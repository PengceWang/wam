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
    goal_text: str           # 交给 ``model.set_goal`` 编码的指令
    scale: float             # 把奖励缩放到"每千步量级 1.0"，见 reward_of
    items: tuple[str, ...] = ()   # 物品栏里这些东西变多就算数
    placed: bool = False          # 放置方块（``used *`` 事件）
    positional: bool = False      # 用位移算


# **这张表是这个模块最重要的部分，也是上一版最烂的部分。** 上一版列了
# dig / leaves / gather 三个任务，问题是：
#
#   * 单任务失败的病灶就是"去挖泥土草方块而不是砍树"，而 dig / leaves
#     **在付钱让它继续这么干**；
#   * gather = ``picked up *``，几乎是所有任务的超集 —— 那就是换了皮的
#     ``any_mine``，而 ``any_mine`` 实测会让原木事件归零；
#   * 没有一个通向"砍树然后盖个小房子"。"clear leaves and foliage" 不是
#     任何人会提的需求，只是把 agent 本来就在乱做的事重新贴了个标签。
#
# 现在这四个都是**人真的会提的要求**，而且都在那条路径上。关键仍然是
# ``place`` 和 ``travel`` —— 它们的最优解**不包含攻击**，所以"按住攻击键"
# 不再是所有奖励轨迹的共同解。这是多任务在这里的全部意义。
TASKS: tuple[Task, ...] = (
    Task("wood",   "chop wood from a tree",    0.5, items=("log",)),
    Task("stone",  "dig down and get stone",   0.5, items=("cobblestone", "stone")),
    Task("place",  "build: place blocks here", 0.3, placed=True),
    Task("travel", "travel far from here",     0.1, positional=True),
)
TASK_BY_KEY = {t.key: t for t in TASKS}


def _gained(inv_before: dict, inv_after: dict, keys: tuple[str, ...]) -> int:
    """物品栏里这些东西**多了几个**。

    用物品栏而不是事件文本，是因为 ``mined oak log`` 和 ``picked up oak log``
    是两条事件、同一根木头 —— 按文本子串计数会算两分。上一版的 ``"log" in p``
    正是这个毛病。物品栏只记你手里真的有什么，不重不漏。
    """
    n = 0
    for name, cnt in inv_after.items():
        if any(k in name for k in keys):
            n += max(0, cnt - inv_before.get(name, 0))
    return n


def reward_of(task: Task, inv_before: dict, inv_after: dict,
              event_text: str, disp: float) -> float:
    """一步的奖励，已按任务缩放。

    ``scale`` 的作用和上一版的 base_rate 一样：把不同任务的奖励拉到同一量级。
    **不做这一步多任务必然退化成单任务** —— 这是 ``tree`` 塑形失败的推广，
    那次 ``0.3 × 树叶`` 加 ``1.0 × 原木``，实际频率是 100:1，原木直接归零。
    而 ``ppo_update`` 的优势归一化是**跨 batch 全局**做的（必须如此，否则抹掉
    任务间差异），所以幅度差会一比一变成梯度权重差。

    但和上一版有个重要区别：上一版的 base_rate 是**拟合 BC 策略**量出来的，
    而策略一变它就过期了。这里的 scale 是固定常数，只表达"一次事件值多少"，
    不假设任何策略。
    """
    if task.positional:
        return disp * task.scale
    if task.placed:
        if not event_text:
            return 0.0
        return task.scale * sum(1 for p in event_text.split(", ") if p.startswith("used "))
    return task.scale * _gained(inv_before, inv_after, task.items)


# --------------------------------------------------------------------------
# 目标采样与后见之明重标注
# --------------------------------------------------------------------------

def packed_hindsight_goals(T: int, rng: np.random.Generator,
                           min_btwn: int = 15, max_btwn: int = 200) -> np.ndarray:
    """给 T 个时间步分配**变长的连续目标区间**，返回每步的目标锚点下标。

    从第 0 步开始，反复"当前位置 + 一个 [min_btwn, max_btwn] 的随机数"取锚点；
    区间 ``(上一个锚点, 这个锚点]`` 内所有步的目标，都设成**这个锚点处的状态**。

    为什么必须变长、必须连续 —— 这是我们上一版和 STEVE-1 的关键差距。
    上一版是 150 步整段贴一个目标（``achieved_tasks`` 扫全段取第一个达成的任务）。
    那样只教会"这一大段最后达成了什么"，教不会**哪些动作导致哪个状态**。
    变长区间让同一批数据里同时有 15 步的短程映射和 200 步的长程映射，
    模型才能学到"目标有多远"这个维度。

    区间上界 200 步不是拍的：一个模型步 0.4 秒，200 步 = 80 秒，
    和我们实测"找到并砍到一棵树"的耗时同量级。
    """
    idx = np.zeros(T, dtype=np.int64)
    lo = 0
    while lo < T:
        hi = min(T - 1, lo + int(rng.integers(min_btwn, max_btwn + 1)))
        idx[lo:hi + 1] = hi
        lo = hi + 1
    return idx


def sample_goals(n: int, rng: np.random.Generator) -> list[Task]:
    """每个环境独立采一个任务。

    独立采样而不是整批同一个任务，是为了让每次更新的 batch 里**同时存在**
    不同目标下的行为。目标向量只有在 batch 内它真的变化时，才会拿到
    "这个输入有用"的梯度。
    """
    return [TASKS[i] for i in rng.integers(0, len(TASKS), size=n)]


def achieved_tasks(inv_first: dict, inv_last: dict,
                   event_texts: list[str], disp: float) -> list[str]:
    """这段轨迹**实际上**完成了哪些任务（不管当时的目标是什么）。"""
    text = ", ".join(t for t in event_texts if t)
    out = []
    for t in TASKS:
        if t.positional:
            if disp >= 20.0:                       # 走出 20 格才算真的"走远了"
                out.append(t.key)
        elif t.placed:
            if any(p.startswith("used ") for p in text.split(", ") if p):
                out.append(t.key)
        elif _gained(inv_first, inv_last, t.items) > 0:
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


# 训练时以这个概率把目标换成 null，推理时才能做 classifier-free guidance。
# **这是 STEVE-1 里最大的单项杠杆**：CFG 让"挖泥土"提升 7.5 倍、"砍木头" 15 倍。
# 我上一版只做了 goal_sensitivity 这个**指标**去观察目标有没有被用上 ——
# 那是温度计，这才是发动机。
GOAL_DROPOUT_P = 0.1


def cfg_logits(cond: dict, uncond: dict, lam: float = 3.0) -> dict:
    """classifier-free guidance：把"有目标"相对"无目标"的差外推出去。

        logits = (1 + λ) · f(o, g) − λ · f(o)

    λ=0 就是普通的条件策略。STEVE-1 的图 5 显示 λ>0 带来的提升是数量级的，
    这也解释了为什么"模型看起来在无视目标"—— 条件信号本来就弱，
    需要在推理时放大，而不是指望它在训练里自己变强。
    """
    return {k: (1.0 + lam) * cond[k].float() - lam * uncond[k].float() for k in cond}


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
