"""在真实环境里做 PPO —— 不用想象，不用学出来的奖励。

三个决定，每个都有实测依据：

**不用想象。** 想象存在的理由是环境太贵，而这里不贵：12 个并行 Minecraft 实例
实测聚合 372 tick/s = **46 模型步/秒**，一小时 16.7 万步。而想象只在 5 步（2 秒）内
可信（mAP 0.350 vs 基线 0.202），误差复合很快。用它换我们不缺的采样效率是纯亏。

**不用学出来的奖励。** ``MineRLEnv`` 吐的是游戏自己的事件，``mined oak log`` 是真值
不是预测。奖励模型的误差在这里可以完全避开。

**不 reset。** docs/stage1b-log.md §1 记着：反复 reset 会让 Minecraft 的世界生成
线程批量死亡（24 次就崩）。而 RL 是不停 reset 的。所以走 continuing task ——
长回合、折扣回报、bootstrap，不用 episodic return。对"砍树"这种任务本来也不需要
重置世界，砍完一棵接着找下一棵。

递归性是这里最麻烦的地方：模型第 t 步要用第 t−1 步的记忆和 KV cache，所以 PPO 的
小批量**不能打乱**。采样时记下每段 rollout 的起始状态，更新时从那个状态重新跑一遍
整段（截断 BPTT），这样 log-prob 和价值都是当前策略下重算的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..model.action import ActionChunk

_DEBUG_MEM = bool(__import__("os").environ.get("WAM_DEBUG_MEM"))


def chunk_log_prob(logits: dict, action: ActionChunk) -> tuple[torch.Tensor, torch.Tensor]:
    """整个动作块的 log-prob 与熵。

    一个块是 8 个 tick × (11 个伯努利按钮 + 2 个 21 类相机 + 1 个 10 类快捷栏)。
    **必须在整块上求和**，不能当成 8 个独立决策 —— 环境是按块给奖励的，
    拆开会让优势估计错位。
    """
    b = logits["buttons"].float()
    lp_b = -F.binary_cross_entropy_with_logits(b, action.buttons.float(), reduction="none")
    p = torch.sigmoid(b)
    ent_b = -(p * F.logsigmoid(b) + (1 - p) * F.logsigmoid(-b))

    def cat(lg, act):
        lg = lg.float().log_softmax(-1)
        lp = lg.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        ent = -(lg.exp() * lg).sum(-1)
        return lp, ent

    lp_c, ent_c = cat(logits["camera"], action.camera)
    lp_h, ent_h = cat(logits["hotbar"], action.hotbar)

    dims = tuple(range(lp_b.dim() - 2, lp_b.dim()))          # (chunk, n_buttons)
    lp = lp_b.sum(dims) + lp_c.sum((-2, -1)) + lp_h.sum(-1)
    ent = ent_b.sum(dims) + ent_c.sum((-2, -1)) + ent_h.sum(-1)
    return lp, ent


def gae(rewards: torch.Tensor, values: torch.Tensor, last_value: torch.Tensor,
        gamma: float = 0.999, lam: float = 0.98) -> tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计。没有回合边界 —— continuing task，所以不需要 done 掩码。

    **这个函数不需要梯度，这一点决定了整个训练的视野。** 算优势只用 reward 和
    value 两个标量序列，存下来几乎不花显存；真正吃显存的是反传（O(T^2)，
    rollout 32 就 OOM）。前四轮训练把这两个视野绑成了同一个 T=8，后果是致命的：

    一个模型步是 8 tick = 0.4 秒，T=8 就是 **3.2 秒**。而实测 BC 策略 150 步
    （60 秒）才砍到 0.83 次原木 —— 也就是说在 3.2 秒的窗口里，砍树奖励**恒等于
    零**，一次都不会出现。于是"往前走"的优势完全来自价值头自举，而价值头从没在
    BC 阶段训练过，是随机初始化 —— 纯噪声。而"挖脚下的方块"1~2 秒就有回报，
    稳稳落在窗口内，有真信号。一边噪声一边信号，四轮 RL 全是挖掘↑位移↓原木↓
    （BC 位移 17.3 vs RL 5.2~9.0），这不是奖励设计的偏好，是**只有一种行为
    拿得到梯度**。

    所以默认值跟着长视野一起改了：

    * ``gamma`` 0.99 → 0.999。0.99 的半衰期是 69 步（27.6 秒），130 步外的奖励
      只剩 ``0.99^130 = 0.27``；0.999 剩 ``0.88``。跨整局做信用分配时，0.99
      本身就是个衰减器。
    * ``lam`` 0.95 → 0.98。λ 越接近 1，GAE 越靠真实回报、越不靠 V。既然 V 是
      废的就干脆绕开它，用接近蒙特卡洛的回报。短 rollout 时不能这么干（方差
      爆炸），150 步 × 12 环境 = 1800 样本摊得住。
    """
    T = rewards.shape[1]
    adv = torch.zeros_like(rewards)
    nxt = last_value
    run = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * nxt - values[:, t]
        run = delta + gamma * lam * run
        adv[:, t] = run
        nxt = values[:, t]
    return adv, adv + values


@dataclass
class PPOConfig:
    gamma: float = 0.999
    lam: float = 0.98
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    # 从 BC 起步，RL 很容易把已有能力冲掉。对初始策略的 KL 惩罚是护栏，
    # 但**固定系数不够** —— 冒烟测试里三步内 KL 就从 0.73 涨到 8.20。
    # 改成自适应：超过目标值就加大系数，低于一半就减小，这是 PPO 的标准做法。
    bc_kl_coef: float = 0.5
    bc_kl_target: float = 0.5
    bc_kl_min: float = 0.05
    bc_kl_max: float = 20.0
    lr: float = 3e-5
    # 更新时每这么多步 detach 一次计算图（截断 BPTT）。**只 detach 不清 cache** ——
    # 前向仍是完整上下文，见 detach_state。整段 150 步不 detach 要 44 GB。
    bptt: int = 16
    epochs: int = 2
    max_grad_norm: float = 1.0


def detach_state(state):
    """把递归状态从计算图上摘下来，但**保留 KV cache 的内容**。

    这个区分是整个训练能不能跑的关键，我用一次上下文扫描才把它分清楚：

    * **清掉 cache** —— 模型真的看不到历史了。同一个 checkpoint 只改这一项：
      每 8/16/32 步清一次，5400 步里原木 **0 次**；从不清，1800 步 **17 次**。
      快方块（草、泥土、树叶，都是一步就碎）完全不受影响，只有原木（约 3 秒
      ≈ 7.5 个模型步）归零。清 cache 等于把策略的能力直接砍掉一块。
    * **detach cache** —— 前向完全不变，模型照样看到全部历史，只是梯度不再
      往更早的时间步回传。这是标准的截断 BPTT，代价只是长程梯度，不是能力。

    之前我把这两件事混为一谈，得出"时间轴不能切"的结论 —— 那只对清 cache 成立。
    """
    from dataclasses import replace

    cache = state.cache
    if cache is not None:
        for attr in ("key_cache", "value_cache"):          # 老版 transformers
            if hasattr(cache, attr):
                setattr(cache, attr, [t.detach() for t in getattr(cache, attr)])
        for layer in getattr(cache, "layers", []) or []:   # 新版：Cache.layers
            for attr in ("keys", "values"):
                if getattr(layer, attr, None) is not None:
                    setattr(layer, attr, getattr(layer, attr).detach())
    # goal 也要摘 —— ``step()`` 里 ``goal=next_goal`` 是目标头产出的，同样挂在图上。
    # 漏掉它就会在第二段反传时报 "backward through the graph a second time"。
    d = lambda x: x.detach() if torch.is_tensor(x) else x            # noqa: E731
    return replace(state, memory=state.memory.detach(), cache=cache,
                   goal=d(state.goal), goal_is_external=d(state.goal_is_external))


def free_cache(state):
    """把 KV cache 里的张量**就地清空**，不依赖引用计数。

    实测教训：采集完把 ``cache=None`` 塞进一个新 state，``memory_allocated``
    纹丝不动（前后都是 20.32 GB）—— 丢引用不等于释放，那个 Cache 对象还被别处
    持有着。12 环境 × 150 步的 cache 约 14 GB，它要是活到反传阶段就会和更新的
    峰值叠在一起，必 OOM。这里直接把 layer.keys/values 置空，谁持有都没用。
    """
    cache = getattr(state, "cache", None)
    if cache is None:
        return
    for attr in ("key_cache", "value_cache"):
        if hasattr(cache, attr):
            setattr(cache, attr, [])
    for layer in getattr(cache, "layers", []) or []:
        for attr in ("keys", "values"):
            if getattr(layer, attr, None) is not None:
                setattr(layer, attr, None)


def ppo_update(model, ref_logits_fn, opt, cfg: PPOConfig, chunks: list[dict]) -> dict:
    """在长视野轨迹上做 PPO 更新。

    ``chunks`` 按**环境**切（不是按时间），每个含若干环境的完整 T 步轨迹。
    每个 shard 内部再按 ``cfg.bptt`` 步分段前向 —— 但段与段之间 **cache 是接着
    往下传的**（只 detach，不清空），所以第 100 步依然能注意到第 1 步。
    见 ``detach_state`` 的说明：清 cache 会毁能力，detach 不会。

    这样三件事同时成立：

    * 前向和采集时**逐位相同**（都是完整上下文），PPO 的 ratio 才有意义
    * 激活显存可控 —— 但**不是被 bptt 完全封住**，实测：

      ==========  ==========  ==========
      段          反传后残留   段内峰值
      ==========  ==========  ==========
      0-16          3.73 GB    12.06 GB
      16-32         4.02 GB    21.76 GB
      32-48         4.30 GB    31.33 GB
      48-64         4.59 GB    40.90 GB
      ==========  ==========  ==========

      残留只涨 0.29 GB/段（就是 KV cache 本身），**没有泄漏**；涨的是段内峰值。
      原因是注意力要看完整上下文而 sdpa 走了 math 后端，把注意力矩阵存下来给
      反传：q=83 × kv≈4000 × 16 头 × 2 环境 × 2 字节 = 21.2 MB/层，
      × 28 层 × 16 步 = 9.6 GB/段，和实测逐位吻合。强制
      ``EFFICIENT_ATTENTION`` 后端在这条路径上直接报 "No available kernel"。

      所以峰值 ∝ ``bptt × env_batch × 上下文长度``，按这三个量调。装上
      flash-attn 能从根上去掉这一项（它不保存注意力矩阵），是后续优化。
    * 信用分配仍是整个 horizon —— 优势在 ``gae`` 里已经算好了

    优势归一化必须**跨所有 shard 一起做**。各自归一会抹平 shard 间差异，
    而"这个环境这一段比别的好"正是长视野带来的信息。
    """
    logs: dict[str, float] = {}
    trainable = [p for p in model.parameters() if p.requires_grad]

    all_adv = torch.cat([c["adv"].reshape(-1) for c in chunks])
    mu, sd = all_adv.mean(), all_adv.std() + 1e-8
    T = chunks[0]["adv"].shape[1]
    K = max(1, min(cfg.bptt, T))
    n_seg = -(-T // K)
    scale = 1.0 / (len(chunks) * n_seg)

    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ("pg", "vf", "ent", "kl", "ratio")}
        for ch in chunks:
            with torch.no_grad():
                ref_all = ref_logits_fn(ch) if (cfg.bc_kl_coef > 0 and ref_logits_fn) else None
            st = ch["state0"]
            for a in range(0, T, K):
                b = min(a + K, T)
                sl = lambda x: x[:, a:b]                              # noqa: E731
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
                ent_loss = -ent.mean()

                kl = torch.zeros((), device=lp.device)
                if ref_all is not None:
                    kl = (
                        F.kl_div(logits["camera"].float().log_softmax(-1),
                                 sl(ref_all["camera"]).float().log_softmax(-1),
                                 log_target=True, reduction="batchmean")
                        + F.binary_cross_entropy_with_logits(
                            logits["buttons"].float(),
                            torch.sigmoid(sl(ref_all["buttons"]).float()))
                    )

                loss = (pg + cfg.value_coef * vf + cfg.entropy_coef * ent_loss
                        + cfg.bc_kl_coef * kl)
                # 除以总段数：累积后的梯度幅度要和"一次算完"可比，
                # 否则等于偷偷把学习率放大了 len(chunks)*n_seg 倍
                (loss * scale).backward()
                # 反传完立刻摘图并释放本段激活；cache 内容保留给下一段
                st = detach_state(out.state)
                del out, logits, value, lp, ent
                if _DEBUG_MEM:
                    print(f"    seg {a:>3}-{b:<3} 反传后 {torch.cuda.memory_allocated()/2**30:5.2f} GB "
                          f"峰值 {torch.cuda.max_memory_allocated()/2**30:5.2f}", flush=True)

                acc["pg"] += float(pg.detach()) * scale
                acc["vf"] += float(vf.detach()) * scale
                acc["ent"] += float(ent_loss.detach()) * -scale
                acc["kl"] += float(kl.detach()) * scale
                acc["ratio"] += float(ratio.mean().detach()) * scale

        gn = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        opt.step()
        logs = {**acc, "grad": float(gn), "adv_std": float(all_adv.std())}

    # 自适应 KL 系数：偏离太多就收紧，偏离太少就放松
    if cfg.bc_kl_coef > 0:
        k = logs["kl"]
        if k > 2.0 * cfg.bc_kl_target:
            cfg.bc_kl_coef = min(cfg.bc_kl_max, cfg.bc_kl_coef * 1.5)
        elif k < 0.5 * cfg.bc_kl_target:
            cfg.bc_kl_coef = max(cfg.bc_kl_min, cfg.bc_kl_coef / 1.5)
    logs["kl_coef"] = cfg.bc_kl_coef
    return logs
