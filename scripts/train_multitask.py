"""多任务、目标条件化的在线 RL。为 2×H100 写的。

    python scripts/train_multitask.py --init-from /path/stage1b.pt --envs 16

**跑法：一张卡一个独立实验，不要 DDP。** 这个仓库在 2×RTX6000Ada 上撞过
NCCL 死锁，而 RL 这条链路（12+ 个 Minecraft 子进程 + spawn 上下文 + 长 KV
cache）本来就脆。两张 H100 的正确用法是同时跑两组**不同配置**互为对照，
而不是把一个 batch 劈成两半 —— 我们缺的是对照组，不是吞吐。

    CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py --tag A ... &
    CUDA_VISIBLE_DEVICES=1 python scripts/train_multitask.py --tag B ... &

这一版相对单任务版改了什么、为什么，全部写在 ``wam/training/multitask.py``
和 ``docs/multitask-design.md``。一句话：单任务下所有拿到奖励的轨迹共享
"按住攻击键"这个共同成分，实测挖掘量翻倍而原木占比纹丝不动。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_online import _boot  # noqa: E402  复用同一套 env worker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--config", default=None, help="覆盖 checkpoint 里的配置，换骨干/seq_len 时用")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=150,
                    help="采集步数 = 信用分配的视野**上限**，也是上下文长度。"
                         "不能低于 64 —— 实测每 8/16/32 步打断一次上下文，"
                         "5400 步里原木事件全是 0（docs/online-rl-log.md §3）。")
    ap.add_argument("--goal-every", type=int, default=75,
                    help="每这么多步给每个环境重新采一个目标。"
                         "必须明显短于 horizon，否则一次采集里目标不变，"
                         "batch 内就没有'同状态不同目标'的对比。")
    ap.add_argument("--env-batch", type=int, default=1)
    ap.add_argument("--bptt", type=int, default=16)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.90)
    ap.add_argument("--sil-coef", type=float, default=0.5)
    ap.add_argument("--bc-kl", type=float, default=3.0)
    ap.add_argument("--rotate-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--out", default="/data/wam/checkpoints/multitask.pt")
    ap.add_argument("--tag", default="mt")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    from wam.config import WAMConfig
    from wam.model.action import ActionChunk
    from wam.model.wam import WorldActionModel
    from wam.training.multitask import (TASKS, MultiTaskConfig, achieved_tasks,
                                        goal_sensitivity, normalised_reward,
                                        sample_goals, sil_loss)
    from wam.training.online import chunk_log_prob, free_cache, gae

    blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = (WAMConfig.from_yaml(args.config) if args.config
           else WAMConfig.from_dict(blob["config"]))

    import multiprocessing as mp

    ctx = mp.get_context("spawn")          # 环境必须在主进程碰 CUDA 之前起
    N, T, EB = args.envs, args.horizon, args.env_batch
    print(f"起 {N} 个环境...", flush=True)
    conns, procs, seed_next = [], [], [N]
    for i in range(N):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_boot, args=(child, i, cfg.to_dict()), daemon=True)
        p.start()
        conns.append(parent)
        procs.append(p)
    frames = []
    for i, c in enumerate(conns):
        f = c.recv()
        for _ in range(3):
            if not (isinstance(f[0], str) and f[0] == "ERROR"):
                break
            try:
                procs[i].terminate(); procs[i].join(timeout=10)
            except Exception:
                pass
            parent, child = ctx.Pipe()
            pr = ctx.Process(target=_boot, args=(child, seed_next[0], cfg.to_dict()), daemon=True)
            seed_next[0] += 1
            pr.start()
            conns[i], procs[i] = parent, pr
            f = parent.recv()
        else:
            print(f"环境 {i} 三次都起不来：{f[1]}", flush=True)
            return 1
        frames.append(f)
    print(f"{N} 个环境就绪", flush=True)

    dev = torch.device("cuda")
    model = WorldActionModel(cfg).to(dev)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        print(f"权重不完全匹配（换了骨干时正常）: 缺 {len(missing)} 多 {len(unexpected)}",
              flush=True)
    model.train()

    mt = MultiTaskConfig(lr=args.lr, gamma=args.gamma, lam=args.lam,
                         sil_coef=args.sil_coef, bc_kl_coef=args.bc_kl, bptt=args.bptt)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("value_head")], "lr": mt.lr},
        {"params": list(model.value_head.parameters()), "lr": mt.lr * 10},
    ])
    ref = WorldActionModel(cfg).to(dev)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] {len(TASKS)} 个任务 | 视野 {T} 步 ({T*0.4:.0f}s) | "
          f"目标每 {args.goal_every} 步重采 | γ={mt.gamma} λ={mt.lam} | "
          f"可训练 {tr/1e6:.0f}M / {tot/1e6:.0f}M", flush=True)

    rng = np.random.default_rng(0)
    shape0 = frames[0][0].shape
    last_good, fails = list(frames), [0] * N

    def restart(i, why="异常"):
        print(f"  环境 {i} {why}，重启（种子 {seed_next[0]}）", flush=True)
        for fn in (lambda: conns[i].send(None), lambda: procs[i].terminate(),
                   lambda: procs[i].join(timeout=10)):
            try:
                fn()
            except Exception:
                pass
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=_boot, args=(child, seed_next[0], cfg.to_dict()), daemon=True)
        seed_next[0] += 1
        pr.start()
        conns[i], procs[i] = parent, pr
        f = parent.recv()
        last_good[i], fails[i] = f, 0
        return f

    def to_batch(fr):
        px = torch.from_numpy(np.stack([f[0] for f in fr]).astype(np.float32) / 255.0)
        ev = torch.from_numpy(np.stack([f[1] for f in fr])).long()
        return px.permute(0, 3, 1, 2).to(dev), ev.to(dev)

    goals = sample_goals(N, rng)
    state = model.initial_state(N, device=dev)
    state = model.set_goal(state, [g.goal_text for g in goals])
    t0, total_steps = time.time(), 0

    for upd in range(1, args.updates + 1):
        state0 = type(state)(memory=state.memory.detach(), prev_action=state.prev_action,
                             goal=state.goal, goal_is_external=state.goal_is_external,
                             cache=None)
        P, E, A_b, A_c, A_h, LP, V, R = [], [], [], [], [], [], [], []
        ev_hist = [[] for _ in range(N)]          # 给后见之明重标注用
        goal_hist = [list(goals)]
        per_task = {t.key: [0.0, 0] for t in TASKS}

        with torch.no_grad():
            st = state
            for t in range(T):
                # 目标在采集**中途**重采：一次更新的 batch 里要有同一个环境
                # 在不同目标下的行为，否则 goal 这个输入拿不到区分性的梯度。
                if t and t % args.goal_every == 0:
                    goals = sample_goals(N, rng)
                    st = model.set_goal(st, [g.goal_text for g in goals])
                    goal_hist.append(list(goals))

                px, ev = to_batch(frames)
                P.append(px); E.append(ev)
                out, st = model.step(st, px, ev)
                act = model.action_head.sample(out.readout, temperature=1.0)
                lp, _ = chunk_log_prob(model.action_head(out.readout), act)
                v = model.value_head(out.readout)
                if v.dim() > 1:
                    v = model.value_head.expectation(v)
                st = type(st)(memory=st.memory, prev_action=act, goal=st.goal,
                              goal_is_external=st.goal_is_external, cache=st.cache)

                # .float()/.long()：模型跑在 bfloat16 上，numpy 没有这个 dtype
                b = act.buttons.float().cpu().numpy()
                c = act.camera.long().cpu().numpy()
                h = act.hotbar.long().cpu().numpy()
                dead = []
                for i, conn in enumerate(conns):
                    try:
                        conn.send((b[i], c[i], h[i]))
                    except (BrokenPipeError, OSError):
                        fails[i] += 1; dead.append(i)
                new = []
                for i, conn in enumerate(conns):
                    if i in dead:
                        new.append(last_good[i]); continue
                    try:
                        # 带超时：worker **死掉**会抛 EOFError，worker **卡住**
                        # 什么都不抛，recv 会永远阻塞而 CPU 占用看着完全正常。
                        if not conn.poll(30):
                            raise TimeoutError
                        f = conn.recv()
                    except (EOFError, OSError, TimeoutError):
                        f = last_good[i]; fails[i] += 1
                    if isinstance(f[0], np.ndarray) and f[0].shape == shape0:
                        last_good[i], fails[i] = f, 0
                    else:
                        f = last_good[i]; fails[i] += 1
                    new.append(f)
                frames = new
                for i in range(N):
                    if fails[i] >= 3:
                        frames[i] = restart(i)

                A_b.append(act.buttons); A_c.append(act.camera); A_h.append(act.hotbar)
                LP.append(lp); V.append(v.float())
                rew = []
                for i, f in enumerate(frames):
                    ev_hist[i].append(f[2] or "")
                    r = normalised_reward(goals[i], f[2] or "", 0.0)
                    rew.append(r)
                    per_task[goals[i].key][0] += r
                    per_task[goals[i].key][1] += 1
                R.append(torch.tensor(rew, device=dev, dtype=torch.float32))
                total_steps += N

            px, ev = to_batch(frames)
            out_last, _ = model.step(st, px, ev)
            v_last = model.value_head(out_last.readout)
            if v_last.dim() > 1:
                v_last = model.value_head.expectation(v_last)
            # 采集完立刻清 cache。N×T 的 KV 是几十 GB，活到反传阶段必 OOM，
            # 而且 cache=None 只丢引用**不释放** —— 必须就地清空（实测 20.32→4.25 GB）。
            del out_last
            free_cache(st)
            st = type(st)(memory=st.memory.detach(), prev_action=st.prev_action,
                          goal=st.goal, goal_is_external=st.goal_is_external, cache=None)
            torch.cuda.empty_cache()

        stack = lambda xs: torch.stack(xs, dim=1)                      # noqa: E731
        adv, returns = gae(stack(R), stack(V), v_last.float(), mt.gamma, mt.lam)
        px_all, ev_all = stack(P), stack(E)
        ab, ac, ah = stack(A_b), stack(A_c), stack(A_h)
        lp_all = stack(LP).detach()
        del P, E, A_b, A_c, A_h, LP, V, R

        # --- 后见之明重标注：这段轨迹实际完成了什么，就按什么重新贴标签 ---
        # 只喂给自我模仿（监督），**不喂给 PPO** —— 换了目标之后行为策略和
        # 目标策略不再是同一个，PPO 的重要性比值就失去意义了。见 sil_loss。
        sil_w, sil_goal = torch.zeros(N, T, device=dev), [None] * N
        for i in range(N):
            got = achieved_tasks(ev_hist[i], 0.0)
            if got:
                sil_goal[i] = got[0]
                sil_w[i] = 1.0

        chunks = []
        for lo in range(0, N, EB):
            hi = min(lo + EB, N)
            pa = state0.prev_action
            s0 = type(state0)(
                memory=state0.memory[lo:hi],
                prev_action=ActionChunk(pa.buttons[lo:hi], pa.camera[lo:hi], pa.hotbar[lo:hi]),
                goal=None if state0.goal is None else state0.goal[lo:hi],
                goal_is_external=state0.goal_is_external, cache=None)
            chunks.append({
                "pixels": px_all[lo:hi], "event_ids": ev_all[lo:hi],
                "actions": ActionChunk(ab[lo:hi], ac[lo:hi], ah[lo:hi]),
                "logp_old": lp_all[lo:hi], "adv": adv[lo:hi].detach(),
                "returns": returns[lo:hi].detach(), "state0": s0,
                "sil_w": sil_w[lo:hi],
                "sil_goal": [sil_goal[j] for j in range(lo, hi)],
            })

        logs = multitask_update(model, ref, opt, mt, chunks)
        state = type(state)(memory=st.memory.detach(), prev_action=st.prev_action,
                            goal=st.goal, goal_is_external=st.goal_is_external, cache=None)

        if args.rotate_every and upd % args.rotate_every == 0:
            for j in range(max(1, N // 3)):
                i = (upd // args.rotate_every * (N // 3) + j) % N
                frames[i] = restart(i, why="轮换")
            state = model.initial_state(N, device=dev)
            state = model.set_goal(state, [g.goal_text for g in goals])
        if args.ckpt_every and upd % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "update": upd},
                       str(args.out).replace(".pt", f"_u{upd}.pt"))

        if upd % args.log_every == 0 or upd == 1:
            rates = " ".join(f"{k}={per_task[k][0]/max(per_task[k][1],1)*1000:.2f}"
                             for k in per_task)
            print(f"{upd:>5} [{args.tag}] {rates} | pg={logs['pg']:+.4f} "
                  f"vf={logs['vf']:.3f} sil={logs['sil']:.3f} kl={logs['kl']:.2f} "
                  f"目标敏感度={logs['goal_sens']:.3f} | "
                  f"{total_steps/(time.time()-t0):.1f} 步/s", flush=True)

    for c in conns:
        try:
            c.send(None)
        except Exception:
            pass
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, args.out)
    print(f"saved {args.out}")
    return 0


def multitask_update(model, ref, opt, cfg, chunks) -> dict:
    """PPO（按实际目标，on-policy）+ 自我模仿（按重标注目标，监督）。

    分段前向时 **cache 只 detach 不清空** —— 前向仍是完整上下文。
    清 cache 会毁掉能力（原木事件直接归零），detach 只是不回传长程梯度。
    """
    import torch.nn.functional as F

    from wam.model.action import ActionChunk
    from wam.training.multitask import TASK_BY_KEY, goal_sensitivity, sil_loss
    from wam.training.online import chunk_log_prob, detach_state

    all_adv = torch.cat([c["adv"].reshape(-1) for c in chunks])
    mu, sd = all_adv.mean(), all_adv.std() + 1e-8
    T = chunks[0]["adv"].shape[1]
    K = max(1, min(cfg.bptt, T))
    scale = 1.0 / (len(chunks) * -(-T // K))
    acc = {k: 0.0 for k in ("pg", "vf", "sil", "kl", "goal_sens")}

    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        for ch in chunks:
            with torch.no_grad():
                o = ref(ch["pixels"], ch["actions"], ch["event_ids"])
                ref_all = ref.action_head(o.readout)
            st = ch["state0"]
            for a in range(0, T, K):
                b = min(a + K, T)
                sl = lambda x: x[:, a:b]                               # noqa: E731
                acts = ActionChunk(sl(ch["actions"].buttons), sl(ch["actions"].camera),
                                   sl(ch["actions"].hotbar))
                out = model(sl(ch["pixels"]), acts, sl(ch["event_ids"]), state=st)
                logits = model.action_head(out.readout)
                lp, ent = chunk_log_prob(logits, acts)
                value = model.value_head(out.readout)
                if value.dim() > lp.dim():
                    value = model.value_head.expectation(value)

                adv = (sl(ch["adv"]) - mu) / sd
                ratio = (lp - sl(ch["logp_old"])).exp()
                pg = -torch.min(ratio * adv,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                vf = F.mse_loss(value.float(), sl(ch["returns"]))
                kl = (F.kl_div(logits["camera"].float().log_softmax(-1),
                               sl(ref_all["camera"]).float().log_softmax(-1),
                               log_target=True, reduction="batchmean")
                      + F.binary_cross_entropy_with_logits(
                          logits["buttons"].float(), torch.sigmoid(sl(ref_all["buttons"]).float())))

                sil = torch.zeros((), device=lp.device)
                if cfg.sil_coef > 0 and any(g is not None for g in ch["sil_goal"]):
                    texts = [TASK_BY_KEY[g].goal_text if g else "" for g in ch["sil_goal"]]
                    st_h = model.set_goal(detach_state(st), texts)
                    out_h = model(sl(ch["pixels"]), acts, sl(ch["event_ids"]), state=st_h)
                    sil = sil_loss(model.action_head(out_h.readout), acts, sl(ch["sil_w"]))
                    acc["goal_sens"] += float(
                        goal_sensitivity(model, out.readout, out_h.readout.detach())) * scale

                loss = (pg + cfg.value_coef * vf - cfg.entropy_coef * ent.mean()
                        + cfg.bc_kl_coef * kl + cfg.sil_coef * sil)
                (loss * scale).backward()
                st = detach_state(out.state)

                acc["pg"] += float(pg.detach()) * scale
                acc["vf"] += float(vf.detach()) * scale
                acc["sil"] += float(sil.detach()) * scale
                acc["kl"] += float(kl.detach()) * scale
                del out, logits, value, lp, ent

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        opt.step()

    k = acc["kl"]
    if k > 2.0 * cfg.bc_kl_target:
        cfg.bc_kl_coef = min(cfg.bc_kl_max, cfg.bc_kl_coef * 1.5)
    elif k < 0.5 * cfg.bc_kl_target:
        cfg.bc_kl_coef = max(cfg.bc_kl_min, cfg.bc_kl_coef / 1.5)
    return acc


if __name__ == "__main__":
    raise SystemExit(main())
