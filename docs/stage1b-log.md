# Linux 迁移与 stage 1b：做了什么，测到了什么

承接 [measurements.md](measurements.md)。那一份记的是 RTX 3060 + WSL2 上的数字；
这一份记 2×RTX 6000 Ada 的 Linux 机器上重新跑通、以及把 world head 的事件预测打开
之后的结果。**凡是本文给出的数字都是本机实测**，估计值会明说。

---

## 0. 一句话

事件预测打开了，而且是这次唯一一个**明确成功**的改动：mAP 0.647，最强基线
0.305。其余几项实验（更大主干、更宽 adapter、语言语义）**全部得到负结果或零结果**，
但它们缩小了问题范围，价值不比正结果小。

---

## 1. 环境：从 Windows/WSL2 到 Linux

两个环境，刻意分开，理由在最后。

| | 环境 A（模型） | 环境 B（模拟器） |
|---|---|---|
| 位置 | `.venv`（uv） | `/data/wam/mamba/envs/minestudio`（micromamba） |
| Python | 3.13.6 | 3.10.20 |
| torch | 2.13.0+cu130 | 2.8.0+cu128 |
| 其它 | transformers 5.14.1 | JDK 8 (Zulu 1.8.0_472) + MineStudio 1.1.6 |

**全程不需要 sudo** —— conda-forge 的 `openjdk=8` 就够了。

不能合并的原因：MineStudio 硬 pin 了 `opencv-python==4.8.0.74`、`av==13.1.0`、
`pyrender==0.1.25`、老 `gym`/`gym3`，且是 transformers 4.x 时代的代码。

### 三个本机特有的坑

**user-site 泄漏。** 环境 B 的 Python 3.10 与 Ubuntu 22.04 的系统 Python 同版本，
`~/.local/lib/python3.10/site-packages`（453 个包）被挂进 `sys.path` 且**排在 env 之前**。
pip 因此认为依赖已满足，`numpy` / `transformers` / `gymnasium` **根本没装进 env**。
用 `PYTHONNOUSERSITE=1` 重装补进 60 个包，并写 `etc/conda/activate.d/` 固化。

**xvfb 缺失。** `launchClient.sh:41` 写死 `xvfb-run -a java`。本机没装 xvfb（需 sudo），
conda-forge 的 cos7 版缺 `libcrypto.so.10`。解法：`:10` 上有个属主为本用户的 xrdp Xorg，
带 GLX + DRI2、走 Mesa llvmpipe，写一个 `xvfb-run` 垫片直接用它。**实测 42.25 FPS**，
与 WSL 的 llvmpipe 基线（42.3 / 44.3）一致。这是对上游的偏离，重装会被冲掉。

**世界重新生成会崩。** `eval_stage1a.py` 每条指令都 `env.reset()`（8 指令 × 3 次 = 24 次），
某次 reset 触发世界重新生成后 Minecraft 内部 `Worker-Main-*` 线程批量死亡，主进程失去响应。
`play_server` 只 reset 一次所以从不触发。**任何频繁 reset 的任务都会撞上，包括 stage 2 的在线 RL。**

### transformers 4.x → 5.14.1

通过。`backbone.py` 用的是 `dtype=`（v5 保留），不是 v4 的 `torch_dtype=`（v5 已移除）。
**KV cache 在 `model.train()` 下实测长度 664 = 83 × 8**，跨时间步注意力没有退化 ——
这一条必须在 train 模式下验，getting-started.md 警告过 checkpointing 会静默关掉
`use_cache` 且 eval 模式看不出来。

---

## 2. `wam/data/` 是重建的

**原包从未进过 git。** `.gitignore` 的 `data/` 不带前导斜杠，匹配任意层级的同名目录，
把整个 `wam/data/` 吞掉了（`git check-ignore -v wam/data/__init__.py` → `.gitignore:25:data/`）。
本次已修为 `/data/`。

重建按 measurements.md 的规格做，并以其中的确切计数为验收标准。

### 精确复现的部分

| 核对项 | 本次 | 文档 |
|---|---|---|
| 候选窗口 | **423,316** | 423,316 ✓ |
| image episodes / frames | 3,658 / 27,267,072 | 同 ✓ |
| 时长 | 378.7 小时 | 378.7 ✓ |
| 长度自相矛盾的 episode | 15 个，最大 9.1× | "15 episodes, one by 9×" ✓ |

窗口切法 `(min(image, action) - 1) // 64` 差 0。那个 `-1` 不是差一错误：timestep t
观察其 chunk 的**起始**帧，最后一个 chunk 之后还需留一帧。

### 近似的部分

**指令标注阈值是反解的** —— 文档只给了各类最终窗口数，没给规则。坐标下降拟合结果：

| 指令 | 本次 | 文档 |
|---|---|---|
| move forward | 75,662 | 77,910 |
| mine the block in front | 43,407 | 44,183 |
| turn left / right | 12,501 / 10,667 | 11,400 / 12,200 |
| jump | 2,075 | 2,362 |
| move backward | 1,767 | 1,868 |
| strafe left / right | 759 / 682 | 814 / ~780 |
| **合计** | **147,520 (34.8%)** | 151,517 (35.8%) |

总体 −2.6%。**拿这套标签得出的结论都应连着这张表一起读**（同一份表也写在
`contractor.FIT_QUALITY` 里）。

---

## 3. 硬件：s/step 几乎只由 seq 决定

单卡 RTX 6000 Ada 48GB，带 warmup，stage 1a 两项损失：

| batch | seq | s/step | peak GiB | windows/s |
|---|---|---|---|---|
| 1 | 8 | 0.619 | 3.69 | 1.6 |
| 8 | 8 | 0.627 | 15.00 | 12.8 |
| 16 | 8 | 0.697 | 27.74 | 23.0 |
| 24 | 8 | 0.785 | 40.86 | 30.6 |
| 32 | 8 | OOM | | |
| 1 | 32 | 2.662 | 15.47 | 0.4 |
| 1 | 64 | OOM | | |

**batch 1→12 步时几乎不变（0.619→0.629）** —— rollout 按时间步串行，每步仅 83 token 宽，
GPU 严重喂不饱。所以**加 batch 近乎免费**，只花显存。

`configs/qwen3-0.6b.yaml` 注释里"2×RTX6000Ada 能上 seq 32-64 / batch 8"两头都不成立：
**seq 64 在 batch 1 就 OOM**（显存 O(T²)：KV cache 增长 + 全 T 步激活）。
`gradient_checkpointing` 又被禁用，所以 seq 32 是现实上限。

一个 epoch（147,520 窗口）：3060 约 24h → 本机 batch16 约 **1.8h**。

---

## 4. 三个负结果

### 主干放大 3 倍：无改善

同 batch、同步数、同索引，只换主干：

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| 可训练参数 | 110.2M | 430.2M（3.9×） |
| actor（末 2000 步） | **1.2266** | **1.2559** |

注意这**不是纯粹的主干消融** —— head 宽度随 `d_model` 一起变了，两个变量绑定。
但既然结论是"没差别"，混淆不影响方向。

### 随机向量替代文本嵌入：无差别

八条指令的 goal token 换成八个**匹配了真实嵌入均值方差**的随机向量（不匹配的话
"随机"和"尺度不同"会混在一起）：

| | actor（末 2000 步） |
|---|---|
| 真实文本嵌入 | 1.2266 |
| 八个随机向量 | 1.2501 |

README 的怀疑得到证实：*"Eight random vectors would probably work as well."*

### 同义句泛化：18%，接近随机

但上面那个实验对语言不公平 —— 模型只需区分 8 个类，任何可区分向量都够。
更对口的测试是**训练时没见过的同义句**：

```
基准 move forward       forward=0.91
  ✗ go straight ahead      （什么都不按）
  ✗ walk forwards          （什么都不按）
基准 mine the block in front   attack=0.92
  ✓ dig the block ahead                attack=0.86
  ✓ break the block in front of you    attack=0.86

同义句命中 4/22 (18%)，随机猜 12.5%
```

无关句子（"eat a sandwich"）的表现和失败的同义句**完全一样**。

**根因**：`GoalEncoder.from_text` 用的是 `backbone.get_input_embeddings()` ——
**只查 token 嵌入表，28 层 transformer 一次都没被调用**。所以 goal 向量是"一袋 token 嵌入"，
没有语义组合。两个成功的例子共享字面 token（`the` `block`），是词面重叠不是语义理解。

**这一条解释了前两个负结果**：更大的 LLM 不改变查表行为；随机向量和一袋 token 嵌入
确实差不多。三个实验其实在测同一个断点。

---

## 5. 世界模型：过关，但很薄

### 单步（stage 1a checkpoint）

| 预测器 | 损失 | 余弦 |
|---|---|---|
| **WorldHead** | **0.5011** | **0.8223** |
| copy：下一帧=这一帧 | 0.6108 | 0.7972 |
| mean：训练集平均 | 1.0546 | 0.5947 |
| shuffle：别人的下一帧 | 1.9353 | 0.3574 |

shuffle 只有 0.357 证明 latent **没有塌缩**（若塌缩，任意两条序列都会很像）。

### 多步想象

| horizon | imagine | copy | teacher |
|---|---|---|---|
| 1 | 0.8000 | 0.7971 | 0.8207 |
| 3 | 0.7527 | 0.7352 | 0.8093 |
| 6 | 0.7173 | 0.6986 | 0.8225 |

6 步内始终优于 copy，但领先只有 ~0.02 余弦。**teacher 全程平在 0.82 不衰减而 imagine
掉到 0.717** —— 这就是误差复合。换个说法：推 6 步的信息量约等于"知道 3 步前那一帧"。

**2.4 秒是当前数据切法能测到的上限**（seq 8 − prime 2）。要知道再往后如何，必须扩 context。

训练中 `next_latent` 损失上涨（0.20 → 0.49）**不代表世界模型变差** —— latent 目标由
可训练的 resampler 产生，虽然 `.detach()` 了，分布仍随训练漂移。靶子在动。

---

## 6. Stage 1b：事件预测（本次唯一的正结果）

### 做了什么

`WorldHead` 有四个输出，stage 1a 只训了 `next_latent`；另外三个前向照跑却从没收到梯度，
因为 1a 只加载 image 和 action 两棵树。

新增：
- `scripts/build_concept_vocab.py` —— 从 event 树建 1,023 个概念，**措辞与
  `MineRLEnv._EVENT_VERB` 严格一致**，离线与在线共用同一套 id
- `wam/data/events.py` + `stage1b.py` —— 读 `meta_info`，产出
  `event_ids` / `event_targets` / `flags`
- `scripts/train_stage1b.py` —— 四项损失

**接上了 `EventEmbedding.init_from_text`** —— 这个函数在仓库里从来没有调用点
（getting-started.md 记过这个缺口）。此前 1024 行是随机初始化，`mined oak log` 和
`mined dark oak log` 毫无关系。现在每行由**短语过一遍主干取隐状态**得到，不是像
`from_text` 那样只查嵌入表。

### 两个数据侧的事实

**contractor 的事件是逐帧增量，实时模拟器是整局累计** —— 实测 `sprint_one_cm` 连续帧
27/28/28，20fps × 28cm = 5.6 m/s 正是冲刺速度。适配器那边必须差分，这边绝对不能。

**`contact` 和 `new_area` 无法监督** —— meta_info 里没有血量也没有生物群系。
由 `FLAG_SUPERVISED` 屏蔽，而不是编假标签。

**全量模式**：事件预测是自监督的，不需要"一条行为占优"这个筛选。放开后从
115,263 涨到 **316,436** 个窗口（2.7×），无标签的窗口 goal 槽填 `null`，
actor 于是学到"没人指挥时人会怎么做"。

### 训练（hidden_mult=4，176.3M 可训练，全量，batch 20，20,000 步 ≈ 1 epoch）

| 步数 | actor | event | flag | latent |
|---|---|---|---|---|
| 0–999 | 2.3127 | 0.03653 | 0.1639 | 0.1878 |
| 10000–20000 | 1.9942 | 0.00113 | 0.1116 | 0.2421 |

⚠️ **`event=0.0011` 是假的好看** —— 1024 维多标签、一步平均 1.57 个正样本，
全输出负值就能拿到它。

### 真实质量

| 预测器 | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| **WorldHead** | **0.402** | **0.764** | **0.838** | **0.647** |
| persistence：下一步=这一步 | 0.234 | 0.417 | 0.534 | 0.305 |
| prior：按全局频率 | 0.045 | 0.178 | 0.317 | 0.155 |

正样本位预测概率 **0.2760**，负样本位 **0.00036** —— 相差 766 倍，模型不是在全局压低输出。

persistence 是很强的基线（事件时间自相关强，在挖石头就会一直挖），**mAP 翻倍还多**。

6,720 个时间步，20.6% 有事件，平均每步 1.57 个。

---

## 7. 后续

第二步（hindsight goals）的完整记录见 **[hindsight-goals.md](hindsight-goals.md)** ——
包括一次因 LLM 隐状态各向异性而白跑的训练、以及一个被自己实验推翻的诊断。
结果：GoalHead 输出的语义分离度 0.027 -> 0.491。

## 8. 还开着的问题

- **`init_from_text` 的贡献未做对照。** 上面的提升有多少来自语义初始化，未知。
- **只测了单步事件预测。** "砍树"需要的是"未来 N 步内会不会掉原木"。
- **`L_align` 仍未启用** —— 项目的中心主张，需要 `event` + `meta_info`（都已就绪）。
- **`hindsight_goals()` 仍无调用点** —— 和 `init_from_text` 当初一样。
- **goal 空间仍是查找表**，`from_text` 的断点未修。
- **相机时间结构有结构性天花板**：人类 lag-1 自相关 0.782，而 `ActionHead.sample`
  把一个 chunk 的 8 个 tick 从同一隐状态独立采样，结构上只能得到 ~0.00。这要改架构。
- **`eval_stage1a.py` 无法完成**（见 §1 世界重新生成崩溃）。真实服从率仍未量化。
