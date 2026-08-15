"""里程碑 0：在真实环境里用 PPO 放大「砍到原木」这件事。

    python scripts/train_online.py --envs 12 --init-from /data/wam/checkpoints/stage1b_cl.pt

**这一步验证的是机制，不是能力。** 不指望它学会"过河砍树" —— 那个奖励太稀疏，
当前策略 150 步只走 21 格、forward 只按 7-9%，整个训练可能一次奖励都拿不到。

所以奖励它**已经偶尔会做**的事：``mined *_log`` 触发 +1。实测 BC 策略约 150 步
一次，不是零 —— 这正是关键，PPO 有东西可放大。判据只有一个：**这个率有没有上升**。
上升说明在线回路是活的，可以往难任务加码；不动说明回路本身有问题，跟任务无关。

前面三次目标条件化 BC 全部失败（见 docs/hindsight-goals.md §7），根因是 BC 里没有
搜索 —— 它只会模仿"这个画面下人通常按什么"，学不会"为了拿到木头而做一系列本身
不像木头的动作"。在线交互把后果补上了。
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pipe, Process
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def env_worker(conn, seed: int, cfg_dict: dict):
    """一个长命 Minecraft 实例。只 reset 一次 —— 反复 reset 会崩（见 online.py）。

    用 spawn 起，且在主进程建模型**之前**起：fork 一个已经初始化了 CUDA 的进程，
    子进程会继承那个上下文，12 个并发时直接把 worker 弄死（4 个时侥幸能过）。
    """
    sys.path.insert(0, "/home/pengce/MineWorld")
    from wam.config import WAMConfig
    from wam.envs.minerl import MineRLEnv
    from wam.model.action import ActionChunk

    cfg = WAMConfig.from_dict(cfg_dict)
    env = MineRLEnv(cfg, seed=seed, vocab_path=None)
    obs = env.reset()
    conn.send((obs.rgb, obs.event_ids, ""))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        b, c, h = msg
        obs = env.step(ActionChunk(torch.from_numpy(b), torch.from_numpy(c), torch.from_numpy(h)))
        conn.send((obs.rgb, obs.event_ids, obs.event_text or ""))
    env.close()


def _boot(conn, seed, cfg_dict):
    """把 worker 的异常送回主进程，否则只能看到一个光秃秃的 EOFError。"""
    try:
        env_worker(conn, seed, cfg_dict)
    except Exception as exc:  # noqa: BLE001
        import traceback
        try:
            conn.send(("ERROR", f"{type(exc).__name__}: {exc}", traceback.format_exc()[-400:]))
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=150,
                    help="每次更新前采多少步 —— 这是**信用分配**的视野。"
                         "一步 = 8 tick = 0.4 秒，150 步 = 60 秒，覆盖 BC 实测"
                         "「找到并砍到一棵树」的完整耗时（150 步 0.83 次）。"
                         "采样不带梯度，所以加长它几乎不花显存。")
    ap.add_argument("--env-batch", type=int, default=2,
                    help="更新时一次反传几个环境的**完整**轨迹。"
                         "梯度累积沿**环境**切，不沿时间切 —— 上下文长度扫描"
                         "（docs/online-rl-log.md §3）显示按时间切会直接毁掉能力："
                         "每 8/16/32 步清一次 cache，5400 步里原木全是 0；"
                         "不清则 1800 步出 17 次。原因不是砍不完一根（那只要 7.5 步），"
                         "而是策略需要整段不被打断的上下文。所以时间轴不能切，"
                         "只能切环境 —— 而且这样采集和更新看到的上下文天然一致，"
                         "连 ratio 对不上的问题也一起没了。")
    ap.add_argument("--bptt", type=int, default=16,
                    help="更新时每这么多步 detach 一次计算图。**只 detach 不清 cache**，"
                         "前向仍看完整 horizon —— 清 cache 会毁能力（原木归零），"
                         "detach 只是不回传长程梯度。见 online.py 的 detach_state。")
    ap.add_argument("--epochs", type=int, default=1,
                    help="每批数据重用几遍。取 1 是因为这里更新一次很贵："
                         "12 个 shard × 19 段 × 一次前向反传 ≈ 84 秒，"
                         "而采集只要 64 秒 —— 与其重用旧数据不如多采新数据。")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.98)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bc-kl", type=float, default=0.5)
    ap.add_argument("--reward", default="tree", choices=["log", "any_mine", "tree"])
    ap.add_argument("--out", default="/data/wam/checkpoints/online.pt")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--rotate-every", type=int, default=200,
                    help="每这么多次更新换掉三分之一的世界（重启 worker 用新种子）。"
                         "不换的话奖励会因为**资源枯竭**而下降 —— tree 那轮实测从 4.31 "
                         "单调掉到 0.44，而 KL 只有 0.4（策略几乎没动），说明掉的不是"
                         "策略而是出生点附近的树被砍光了。重启进程而不是 reset 同一个"
                         "实例，同时避开了反复 reset 会崩的老问题。")
    ap.add_argument("--ckpt-every", type=int, default=250,
                    help="每这么多次更新另存一个带编号的 checkpoint，供事后画"
                         "「真实进展曲线」—— 训练奖励曲线在这个设定下不可信，"
                         "它同时反映策略和环境状态，两者分不开。")
    args = ap.parse_args()

    from wam.config import WAMConfig
    from wam.model.action import ActionChunk
    from wam.model.wam import WorldActionModel
    from wam.training.online import PPOConfig, chunk_log_prob, gae, ppo_update

    blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = WAMConfig.from_dict(blob["config"])

    # --- 环境先起，之后才碰 CUDA ---
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    N, T, EB = args.envs, args.horizon, args.env_batch
    print(f"起 {N} 个环境（每个约 40s）...", flush=True)
    conns, procs = [], []
    for i in range(N):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_boot, args=(child, i, cfg.to_dict()), daemon=True)
        p.start()
        conns.append(parent)
        procs.append(p)
    frames = []
    for i, c in enumerate(conns):
        f = c.recv()
        if isinstance(f[0], str) and f[0] == "ERROR":
            print(f"环境 {i} 启动失败: {f[1]}\n{f[2]}")
            return 1
        frames.append(f)
    print(f"{N} 个环境就绪", flush=True)

    dev = torch.device("cuda")
    model = WorldActionModel(cfg).to(dev)
    model.load_state_dict(blob["model"])
    model.train()
    # 价值头从没训练过，冷启动会震荡；给它一个更大的学习率追上去
    ppo = PPOConfig(lr=args.lr, bc_kl_coef=args.bc_kl, gamma=args.gamma, lam=args.lam,
                    bptt=args.bptt, epochs=args.epochs)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("value_head")], "lr": ppo.lr},
        {"params": list(model.value_head.parameters()), "lr": ppo.lr * 10},
    ])

    # 起点策略的副本，用于 KL 护栏
    ref = WorldActionModel(cfg).to(dev)
    ref.load_state_dict(blob["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    print(f"奖励 = {args.reward} | 视野 {T} 步 ({T*0.4:.0f}s，上下文不打断) | "
          f"梯度按环境累积 {EB} 个/批 ({-(-N//EB)} 批) | BPTT {args.bptt} 步 | "
          f"每次更新 {N * T} 个样本 | "
          f"γ={ppo.gamma} λ={ppo.lam}", flush=True)

    # 12 个环境跑一小时，有一个卡住是必然的。MineStudio 超时后会"返回一个随机观测"
    # （它自己的容错），而那个观测形状和别的不一样 —— 上一轮 1040/2000 就是被
    # np.stack 的 shape 不匹配打断的。这里改成：坏帧用上一帧顶，连续坏就重启该 worker。
    shape0 = frames[0][0].shape
    last_good = [f for f in frames]
    fails = [0] * N

    def sanitize(fr):
        for i, f in enumerate(fr):
            if isinstance(f[0], np.ndarray) and f[0].shape == shape0:
                last_good[i] = f
                fails[i] = 0
            else:
                fr[i] = last_good[i]
                fails[i] += 1
        return fr

    seed_next = [N]

    def restart(i, why="异常"):
        print(f"  环境 {i} {why}，重启（新种子 {seed_next[0]}）", flush=True)
        try:
            conns[i].send(None)
        except Exception:
            pass
        try:
            procs[i].terminate()
            procs[i].join(timeout=10)
        except Exception:
            pass
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=_boot, args=(child, seed_next[0], cfg.to_dict()), daemon=True)
        seed_next[0] += 1
        pr.start()
        conns[i], procs[i] = parent, pr
        f = parent.recv()
        last_good[i] = f
        fails[i] = 0
        return f

    def to_batch(fr):
        px = torch.from_numpy(np.stack([f[0] for f in fr]).astype(np.float32) / 255.0)
        px = px.permute(0, 3, 1, 2).to(dev)
        ev = torch.from_numpy(np.stack([f[1] for f in fr])).long().to(dev)
        return px, ev

    def reward_of(text: str) -> float:
        """``tree`` 是带塑形的树木奖励，其余两个是纯的。

        塑形当初的理由**已经不成立了**，记在这里免得再走一遍：那时说纯 ``log``
        奖励一次更新（96 步）只期望 0.43 次事件、优势基本是噪声，而加长 rollout
        被显存 O(T^2) 堵死。后来发现堵死的前提是错的 —— 优势估计根本不需要梯度，
        视野和反传跨度可以解耦（见 ``online.py`` 的 ``gae``）。

        而 ``tree`` 那轮（0.3 × 树叶）为什么把原木打到 0.00，现在有确切答案了。
        实测 1728 步的事件分布：

            oak leaves×26, spruce leaves×11, jungle leaves×5, birch leaves×5 ...
            原木 ×0

        树叶约 50 次/1728 步，原木约 0 次。于是 ``0.3 × 50 = 15`` 对 ``1.0 × 0``，
        **树叶项把奖励信号按 100:1 淹没了**。问题不在"该不该塑形"，在权重
        差了两个数量级 —— 塑形权重必须按事件**频率**定，不能按语义重要性拍。
        真要用，树叶该是 0.01 量级（总贡献 0.5，和一次原木可比）。

        本轮先不塑形：一次只改一个变量，这轮隔离的是视野。

        所以给树叶部分分：**树叶和原木在空间上是同一棵树**，砍到树叶说明已经站在
        树前面了，离原木只差瞄准。这是有依据的塑形，不是随便加权 ——
        而上一轮用 ``any_mine`` 的教训正好相反：泥土草方块石头都算分，
        模型就去挖脚下最近的方块，原木事件反而归零（15.3 挖掘 vs 0.00 原木）。
        """
        if not text:
            return 0.0
        parts = text.split(", ")
        if args.reward == "any_mine":
            return float(sum(1 for p in parts if p.startswith("mined")))
        r = float(sum(1 for p in parts if "log" in p))
        if args.reward == "tree":
            r += 0.3 * sum(1 for p in parts if "leaves" in p or "sapling" in p)
        return r

    state = model.initial_state(N, device=dev)
    t0 = time.time()
    ema_r = None
    total_steps = 0

    t_roll = t_upd = 0.0
    for upd in range(1, args.updates + 1):
        _t = time.time()
        P, E, A_b, A_c, A_h, LP, V, R = [], [], [], [], [], [], [], []
        seen: dict[str, int] = {}
        # 整段只在起点存一次状态。**采集全程不清 cache** —— 更新时也会重放
        # 完整的 T 步，两边上下文天然一致，不存在 ratio 对不上的问题。
        state0 = type(state)(memory=state.memory.detach(), prev_action=state.prev_action,
                             goal=state.goal, goal_is_external=state.goal_is_external,
                             cache=None)

        with torch.no_grad():
            st = state
            for t in range(T):
                px, ev = to_batch(frames)
                P.append(px)
                E.append(ev)
                out, st = model.step(st, px, ev)
                logits = model.action_head(out.readout)
                act = model.action_head.sample(out.readout, temperature=1.0)
                lp, _ = chunk_log_prob(logits, act)
                v = model.value_head(out.readout)
                if v.dim() > 1:
                    v = model.value_head.expectation(v)
                st = type(st)(memory=st.memory, prev_action=act, goal=st.goal,
                              goal_is_external=st.goal_is_external, cache=st.cache)

                # .float()/.long() 不能省：模型跑在 bfloat16 上，numpy 没有这个 dtype。
                # 这是 docs/measurements.md「静默失败」清单的第三条，本会话已撞过三次
                # （eval_stage1a.py、这里、以及适配器本身早就防住了的那处）。
                b = act.buttons.float().cpu().numpy()
                c = act.camera.long().cpu().numpy()
                h = act.hotbar.long().cpu().numpy()
                # 发送侧也要容错。上一轮只包了 recv，结果 worker 死掉之后
                # 主进程往关闭的管道写，BrokenPipe 直接打断整轮训练（320/2500）。
                dead = []
                for i, conn in enumerate(conns):
                    try:
                        conn.send((b[i], c[i], h[i]))
                    except (BrokenPipeError, OSError):
                        fails[i] += 1
                        dead.append(i)
                frames = []
                for i, conn in enumerate(conns):
                    if i in dead:
                        frames.append(last_good[i])
                        continue
                    try:
                        frames.append(conn.recv())
                    except (EOFError, OSError):
                        frames.append(last_good[i])
                        fails[i] += 1
                frames = sanitize(frames)
                for i in range(N):
                    if fails[i] >= 3:
                        frames[i] = restart(i)

                A_b.append(act.buttons); A_c.append(act.camera); A_h.append(act.hotbar)
                LP.append(lp); V.append(v.float())
                R.append(torch.tensor([reward_of(f[2]) for f in frames], device=dev))
                for f in frames:                       # 诊断：这一轮到底挖到了什么
                    for part in (f[2] or "").split(", "):
                        if part.startswith("mined"):
                            seen[part] = seen.get(part, 0) + 1
                total_steps += N

            px, ev = to_batch(frames)
            out_last, _ = model.step(st, px, ev)
            v_last = model.value_head(out_last.readout)
            if v_last.dim() > 1:
                v_last = model.value_head.expectation(v_last)
            # 采集完立刻扔掉 KV cache。150 步 × 12 环境 ≈ 16 GB，它要是活到
            # 反传阶段就和更新的显存叠在一起，必 OOM。memory 槽位要留着传给下一轮。
            del out_last
            st = type(st)(memory=st.memory.detach(), prev_action=st.prev_action,
                          goal=st.goal, goal_is_external=st.goal_is_external, cache=None)
            torch.cuda.empty_cache()

        stack = lambda xs: torch.stack(xs, dim=1)                       # noqa: E731
        rewards = stack(R)
        values = stack(V)
        # 优势在**整个 horizon** 上算，这是这次改动的全部意义所在
        adv, returns = gae(rewards, values, v_last.float(), ppo.gamma, ppo.lam)

        px_all, ev_all = stack(P), stack(E)
        ab, ac, ah = stack(A_b), stack(A_c), stack(A_h)
        lp_all = stack(LP).detach()
        adv, returns = adv.detach(), returns.detach()

        def slice_state(s0, lo, hi):
            pa = s0.prev_action
            return type(s0)(
                memory=s0.memory[lo:hi],
                prev_action=ActionChunk(pa.buttons[lo:hi], pa.camera[lo:hi], pa.hotbar[lo:hi]),
                goal=None if s0.goal is None else s0.goal[lo:hi],
                goal_is_external=(s0.goal_is_external[lo:hi]
                                  if torch.is_tensor(s0.goal_is_external)
                                  else s0.goal_is_external),
                cache=None)

        chunks = []
        for lo in range(0, N, EB):
            hi = min(lo + EB, N)
            chunks.append({
                "pixels": px_all[lo:hi], "event_ids": ev_all[lo:hi],
                "actions": ActionChunk(ab[lo:hi], ac[lo:hi], ah[lo:hi]),
                "logp_old": lp_all[lo:hi], "adv": adv[lo:hi],
                "returns": returns[lo:hi], "state0": slice_state(state0, lo, hi),
            })

        def ref_logits_fn(r):
            o = ref(r["pixels"], r["actions"], r["event_ids"])
            return ref.action_head(o.readout)

        t_roll += time.time() - _t
        _t = time.time()
        logs = ppo_update(model, ref_logits_fn, opt, ppo, chunks)
        t_upd += time.time() - _t
        state = type(state)(memory=st.memory.detach(), prev_action=st.prev_action,
                            goal=st.goal, goal_is_external=st.goal_is_external, cache=None)

        # 轮换世界：错开着换三分之一，避免一次全停
        if args.rotate_every and upd % args.rotate_every == 0:
            k = max(1, N // 3)
            off = (upd // args.rotate_every * k) % N
            for j in range(k):
                i = (off + j) % N
                frames[i] = restart(i, why="轮换")
            state = model.initial_state(N, device=dev)   # 换了世界，记忆作废

        if args.ckpt_every and upd % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "update": upd},
                       str(args.out).replace(".pt", f"_u{upd}.pt"))

        r_per_1k = float(rewards.sum()) / (N * T) * 1000
        ema_r = r_per_1k if ema_r is None else 0.9 * ema_r + 0.1 * r_per_1k
        if upd % args.log_every == 0 or upd == 1:
            sps = total_steps / (time.time() - t0)
            print(f"{upd:>5} 奖励/千步 {ema_r:>7.2f} (本次 {r_per_1k:>6.1f}) "
                  f"pg={logs['pg']:+.4f} vf={logs['vf']:.4f} ent={logs['ent']:.1f} "
                  f"kl={logs['kl']:.2f}(c={logs['kl_coef']:.2f}) | {sps:.1f} 步/s "
                  f"[采样 {100*t_roll/(t_roll+t_upd):.0f}% / 更新 {100*t_upd/(t_roll+t_upd):.0f}%]",
                  flush=True)
            top = sorted(seen.items(), key=lambda kv: -kv[1])[:6]
            print("        挖到: " + (", ".join(f"{k[6:]}×{v}" for k, v in top) or "无"),
                  flush=True)
        if upd % 100 == 0:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "update": upd},
                       args.out)

    for c in conns:
        c.send(None)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
