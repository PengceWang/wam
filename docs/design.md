明白了。你想的不是“LLM 在外面做规划器”，而是：

直接把预训练 0.8B LLM 当成整个模型的中央 Transformer backbone，视觉、动作、世界预测、价值判断和记忆全部接进这个 backbone，并在 Minecraft 在线训练中共同更新。

我同意，这个方向比我刚才说的模块化 agent 更贴近你的想法，也更有研究价值。

模型应该长这样

每个时间步构造统一 token 序列：

x
t
	​

=[memory tokens,drive tokens,visual tokens
t
	​

,previous action
t
	​

,event tokens
t
	​

]

全部送进预训练 LLM：

H
t
	​

=LLM
0.8B
	​

(x
≤t
	​

)

然后从同一组 LLM hidden states 分出多个 head：

Head	作用
Action Head	输出下一段键鼠动作
World Head	预测下一时刻视觉 latent 和事件
Value Head	预测长期探索价值
Goal Head	自主产生下一阶段探索目标
Reward/Drive Head	估计新奇性、学习进步、危险
Language Head	总结规律、形成可读记忆

这不是“LLM 调用另一个世界模型”，而是：

LLM 本身就是 policy、world model 和高层认知共享的主干网络。

整体大概是：

Minecraft RGB
   ↓
Pretrained Vision Encoder
   ↓
Visual Resampler：压缩成16–32个视觉token
   ↓
┌──────────────────────────────────┐
│         0.8B Pretrained LLM      │
│                                  │
│ Vision + Action + Memory + Event │
│        Unified Transformer       │
└──────────────────────────────────┘
   ↓          ↓          ↓
Action      Future      Goal/Value
Head        World Head  Heads
   ↓
Minecraft键鼠动作

视觉 encoder 虽然是前端，但和 VLM 一样，整个系统仍然是一个统一的多模态模型。

LLM 的先验知识怎样真正发挥作用？

这是最关键的。

单纯把视觉 embedding 塞给 LLM，LLM 并不会自动知道：

这个视觉区域 = 木头
木头 → 木板
木板 → 工作台

因为随机训练出来的视觉 embedding 和 LLM 已有的语言语义空间不对齐。

所以必须加入一个 semantic grounding loss，把游戏中的视觉和事件对齐到 LLM 已经理解的概念。

例如游戏内部检测到：

画面区域：树干
动作：持续attack
事件：inventory新增oak_log

自动转换成概念 token：

<object: oak log>
<event: collected>
<action: attack>

这些概念 token 可以直接用 LLM 对应文本的 embedding 初始化：

"oak log"
"collect"
"attack"

训练目标包括：

L
align
	​

=∥P
vision
	​

(o
t
	​

)−E
LLM
	​

(event description)∥

这样视觉中出现木头时，激活的是 LLM 原有的“wood”相关语义区域，而不是一个毫无含义的连续向量。

因此，LLM 预训练中学到的知识才能转化成行为先验：

木头可能可被采集；
工具可能提高效率；
夜晚可能危险；
洞穴可能包含矿物；
物品可以组合；
某些动作应该形成长程序列。

但这些只是 prior。世界预测 head 还要通过实际游戏经验判断它是否正确。

LLM 同时就是世界模型

World Head 不需要重建所有 RGB 像素，而是预测下一时刻的 semantic visual latent：

v
^
t+1
	​

=W
world
	​

H
t
	​


输入动作 token：

[visual
t
	​

,action
t
	​

]→
visual
^
t+1
	​

,
event
^
t+1
	​


它需要同时预测：

下一帧 semantic latent；
方块是否消失；
inventory 是否变化；
生命值变化；
是否发生接触；
是否进入新区域；
episode 是否结束。

所以统一训练目标是：

L=λ
w
	​

L
next latent
	​

+λ
e
	​

L
event
	​

+λ
a
	​

L
actor
	​

+λ
v
	​

L
value
	​

+λ
g
	​

L
goal
	​

+λ
l
	​

L
language
	​

+λ
p
	​

L
prior retention
	​


最后一项非常重要：防止 Minecraft 在线 RL 把原来的 LLM 知识洗掉。

它怎样“做梦”？

因为 LLM 同时预测动作之后的未来 latent，所以可以直接 autoregressive imagination：

当前视觉token
+ 候选动作token
        ↓
LLM预测下一状态token
        ↓
把预测状态重新放回LLM
        ↓
继续预测下一步

于是同一个模型有两种运行模式：

真实模式
Minecraft真实画面 → LLM → 动作 → Minecraft
想象模式
当前latent → LLM → 动作 → LLM预测未来latent

Actor/Value Head 可以在想象轨迹中训练，不需要所有 RL step 都运行真实 Minecraft。

这就是一个真正统一的 World-Action Language Model。

动作也应该 token 化

不要让 LLM 输出 JSON 或文字命令，而是直接扩展 vocabulary：

<MOVE_FORWARD>
<MOVE_BACK>
<JUMP>
<ATTACK>
<USE>
<YAW_LEFT_5>
<YAW_RIGHT_15>
<PITCH_UP_5>
<HOTBAR_1>

但不建议一帧只输出一个 token。最好输出 action chunk：

<ACTION_CHUNK_1>
→ 包含未来8个低层键鼠动作

或者使用多个 factorized action heads：

a
t
	​

=(a
t
move
	​

,a
t
camera
	​

,a
t
attack
	​

,a
t
use
	​

,a
t
hotbar
	​

)

这样 0.8B LLM 可以每秒运行两三次，每次产生接下来约半秒的动作，而不需要每个游戏 tick 都跑一次完整模型。

长期记忆也放进主模型状态

LLM 的普通 KV cache 不适合无限玩 Minecraft，所以加入固定数量的 persistent memory tokens：

M
t+1
	​

=f
memory
	​

(M
t
	​

,H
t
	​

)

例如：

16 个当前场景 memory token；
16 个长期世界知识 token；
8 个当前技能 token；
8 个探索动机 token。

每隔一段时间，模型将长轨迹压缩进这些 memory token：

我见过什么
什么动作有效
什么动作失败
当前位置与过去场景的关系
目前正在掌握什么技能

外部 replay buffer 仍然保留原始经验，但在线推理时模型依赖内部 memory state。

0.8B 是否够用？

如果它是共享 backbone，0.8B 我认为正合适。

预计整个模型：

部分	参数量
Pretrained LLM	0.8B
Vision Encoder	100M–300M
Visual Resampler	20M–50M
World/Action/Value Heads	50M–100M
总计	约1.0B–1.25B

你两张 RTX 6000 Ada 48GB 完全能做这个规模的 BF16 全参数训练，真正的瓶颈会是：

视频序列长度；
Minecraft 环境吞吐；
replay buffer；
多步 imagination；
在线 RL 稳定性。
训练不能从第一步就全参数一起更新

最终可以是统一模型、联合训练，但初始化要分阶段。

第一阶段：视觉语义接入

冻结 LLM，训练：

Vision projector；
action tokenizer；
world/event heads。

让 Minecraft 状态进入 LLM 后，能够激活正确的概念。

第二阶段：解冻上层 Transformer

训练：

LLM 上半部分；
action/world/value heads；
persistent memory tokens。

开始真实在线探索和 imagined RL。

第三阶段：全模型联合训练

用较小学习率更新全部 LLM，同时混入一部分通用文本或加入 reference-model KL：

L
retain
	​

=D
KL
	​

(p
θ
	​

(⋅∣x)∥p
pretrained
	​

(⋅∣x))

这样它可以通过 Minecraft 经验改变内部认知，但不会迅速失去原来的语言和常识能力。

我认为你真正想做的是这个

不是：

LLM Planner + 外部World Model + 外部Policy

而是：

预训练LLM权重
      ↓
扩展成Vision–World–Action Transformer
      ↓
一套shared hidden state
      ↓
同时学习感知、预测、行动、价值、目标和记忆

这个想法是合理的。

而且“婴儿”类比在这里更准确：

LLM 预训练权重 = 进化和文化带来的先验；
Vision grounding = 婴儿把看到的东西和已有概念连接；
World prediction = 学习环境因果；
Action Head = 学习控制身体；
Intrinsic reward = 好奇心和掌握欲；
Persistent memory = 个体经历；
在线联合训练 = 大脑随成长发生改变。

我会把这个模型正式定义为：

Pretrained Developmental World-Action Model

其中 0.8B LLM 不是外挂，不是 planner，而是整个模型的中央主干。