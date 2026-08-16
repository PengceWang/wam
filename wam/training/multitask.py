"""多任务、目标条件化的在线 RL。

单任务版本失败在一个很具体的地方，这个模块的每一处设计都是冲着那个失败去的。
实测（各 6000 步，训练没碰过的种子）：

    checkpoint   挖掘   原木   原木/挖掘
    BC 起点      198    12     6.1%
    U25          373    24     6.4%
    U100         356    18     5.1%

挖掘量翻倍（p < 1e-12，回路确实活了），但**原木占挖掘的比例从头到尾没涨**。
多出来的挖掘量是 ``picked up dirt×144, mined dirt×138, mined grass block×125``。
奖励只给原木，学到的却是"按住攻击键"。

根因：单任务下**所有拿到奖励的轨迹共享同一个共同成分**。信用一摊开
（γ=0.999、λ=0.98、150 步窗口），策略梯度收敛到的就是那个共同成分。

多任务拆掉这个共同成分 —— 只要任务集合里有一个任务的最优解**不包含**攻击
（这里是 ``travel``），"按住攻击键"就不再是所有奖励轨迹的共同解。
这不是"顺便多学几个技能"，是给信用分配一个它现在没有的对比组。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..model.action import ActionChunk
from .online import chunk_log_prob


# --------------------------------------------------------------------------
# 任务集合
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    key: str
    goal_text: str          # 交给 ``model.set_goal`` 编码的指令
    base_rate: float        # BC 策略下每 1000 步的期望事件数，实测
    match: Callable[[str], bool] | None = None   # 事件文本 -> 算不算数
    positional: bool = False                     # 用位移而不是事件算奖励


def _has(*subs: str) -> Callable[[str], bool]:
    return lambda e: any(s in e for s in subs)


# base_rate 全部来自 BC 起点在 6000 步评测里的实测计数，换算成每千步。
# **这些数字是这个模块最重要的部分**，理由见 normalised_reward。
TASKS: tuple[Task, ...] = (
    Task("wood",   "chop wood from a tree",      2.0,  _has("log")),
    Task("dig",    "dig into the ground",       18.2,  _has("dirt", "grass block", "sand", "gravel")),
    Task("leaves", "clear leaves and foliage",  11.7,  _has("leaves", "grass", "fern", "vine")),
    Task("gather", "pick up items nearby",       9.0,  _has("picked up")),
    Task("travel", "travel far from here",       1.0,  None, positional=True),
)
TASK_BY_KEY = {t.key: t for t in TASKS}


def normalised_reward(task: Task, event_text: str, disp: float) -> float:
    """把每个任务的奖励缩放到"BC 策略下每千步期望 1.0"。

    **不做这一步，多任务必然退化成单任务。** 这是 ``tree`` 塑形失败的教训的推广：
    那次用 ``0.3 × 树叶`` 加 ``1.0 × 原木``，而树叶约 50 次/1728 步、原木约 0 次，
    塑形项按 **100:1** 淹没了目标信号，原木直接归零。

    这里的风险一模一样，只是换了个轴：``dig`` 的自然发生率是 ``wood`` 的 **9 倍**
    （18.2 vs 2.0 次/千步）。不归一化的话，``dig`` 的优势幅度压倒一切 ——
    而 ``ppo_update`` 的优势归一化是**跨整个 batch 全局**做的（必须如此，
    否则抹掉任务间差异），所以幅度差会直接变成梯度权重差。

    权重要按**事件频率**定，不能按语义重要性拍 —— 这条已经付过一次学费了。
    """
    if task.positional:
        return disp / task.base_rate * 1e-3 * 1000.0
    if not event_text:
        return 0.0
    hits = sum(1 for p in event_text.split(", ") if p and task.match(p))
    return hits / task.base_rate


# --------------------------------------------------------------------------
# 目标采样与后见之明重标注
# --------------------------------------------------------------------------

def sample_goals(n: int, rng: np.random.Generator) -> list[Task]:
    """每个环境独立采一个任务。

    独立采样而不是整批同一个任务，是为了让每次更新的 batch 里**同时存在**
    不同目标下的行为。目标向量只有在 batch 内它真的变化时，才会拿到
    "这个输入有用"的梯度。
    """
    return [TASKS[i] for i in rng.integers(0, len(TASKS), size=n)]


def achieved_tasks(event_texts: list[str], disp: float) -> list[str]:
    """这段轨迹**实际上**完成了哪些任务（不管当时的目标是什么）。"""
    text = ", ".join(t for t in event_texts if t)
    out = []
    for t in TASKS:
        if t.positional:
            if disp / t.base_rate >= 1.0:
                out.append(t.key)
        elif t.match and any(t.match(p) for p in text.split(", ") if p):
            out.append(t.key)
    return out


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------

@dataclass
class MultiTaskConfig:
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    bc_kl_coef: float = 3.0
    bc_kl_target: float = 0.5
    bc_kl_min: float = 0.05
    bc_kl_max: float = 20.0
    # 自我模仿：对**后见之明重标注过的成功片段**做监督学习。
    # 见 sil_loss 的说明 —— 这是唯一能安全用上 HER 的地方。
    sil_coef: float = 0.5
    # 信用视野。单任务那轮用的是 0.999/0.98，事后诊断是"信号太散"：
    # 第 140 步砍到一根原木，前面 56 秒几乎被同等奖励。这里收紧。
    # 上下文长度**不跟着收** —— 那是能力的硬下限（见 docs/online-rl-log.md §3）。
    gamma: float = 0.99
    lam: float = 0.90
    lr: float = 3e-5
    bptt: int = 16
    epochs: int = 1
    max_grad_norm: float = 1.0


def sil_loss(logits: dict, actions: ActionChunk, weight: torch.Tensor) -> torch.Tensor:
    """自我模仿：在重标注后的成功片段上做行为克隆。

    **为什么 HER 不能直接喂给 PPO。** PPO 的比值 ``π(a|s,g)/π_old(a|s,g)`` 假设
    数据是当前策略在**同一个目标**下采的。把目标从 g 改成 g' 之后，行为策略
    变成了 ``π(·|s,g)`` 而目标策略是 ``π(·|s,g')``，两者不同，比值不再是那个含义，
    clip 的信赖域保证也就失效了。

    所以这里只把 HER 用在**监督**目标上：片段既然真的达成了 g'，那这些动作
    对 g' 就是好的 —— 直接最大化 ``log π(a|s,g')``，不需要任何重要性比值。
    这就是 self-imitation learning，加上后见之明标注就自然是多任务的。

    这也正面回答了之前三次目标条件化 BC 为什么失败（docs/hindsight-goals.md §7）：
    那三次是拿**人类演示**做目标条件 BC，而 BC 里没有搜索。这里的片段来自
    **agent 自己的成功经验**，搜索由旁边的 PPO 提供 —— 监督只负责把搜到的东西
    固化下来。
    """
    lp, _ = chunk_log_prob(logits, actions)
    return -(lp * weight).sum() / weight.sum().clamp(min=1.0)


def goal_sensitivity(model, out_readout, goal_shuffled_readout) -> torch.Tensor:
    """诊断：把目标换掉，动作分布会变吗？

    **这是前三次目标条件化失败时缺的那个仪表。** 那三次是靠"同义句迁移率"
    这种下游指标发现问题的，而那时候已经训完了。这个量每次更新都能算，
    而且含义没有歧义：

        ≈ 0   模型在**无视** goal 输入，多任务退化成"所有任务的平均策略"
        > 0   goal 真的进入了决策

    它只是**指标不是损失**。做成损失会被轻易钻空子 —— 模型只要让输出随目标
    随机抖动就能把它拉高，而那不是"听懂了目标"。目标该不该被使用，
    应该由奖励说了算：奖励依赖目标，无视目标就拿不到分。
    """
    a = model.action_head(out_readout)["camera"].float().log_softmax(-1)
    b = model.action_head(goal_shuffled_readout)["camera"].float().log_softmax(-1)
    return F.kl_div(a, b, log_target=True, reduction="batchmean").detach()
