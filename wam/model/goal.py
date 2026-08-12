"""External goals, encoded into the same slot the model writes its own into.

The backbone is a pretrained LLM, so "chop oak wood" needs no new modality and no
alignment loss -- it is already a sequence of vectors the model can read. That is
the whole reason an external goal is cheap here.

The two sources have to be *interchangeable*, not merely both present. If
supplied goals and self-proposed goals differ in norm or distribution, the model
learns to tell them apart, and withdrawing the scaffolding in stage 3 becomes a
distribution shift instead of a handover. Both paths therefore pass through the
same normalisation before they reach the sequence -- see ``goal_norm`` in
``WorldActionModel``.
"""

from __future__ import annotations

import torch
from torch import nn


class GoalEncoder(nn.Module):
    """Text -> ``n_goal_tokens`` vectors, plus the "no goal given" default."""

    def __init__(self, n_goal_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n_goal_tokens = n_goal_tokens
        self.d_model = d_model
        # What fills the slot when nobody has set a goal yet: a learned token
        # rather than zeros, so "no goal" is a state the model can represent
        # instead of an absence it has to infer from a dead input.
        self.null_goal = nn.Parameter(torch.randn(n_goal_tokens, d_model) * 0.02)

    def null(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        out = self.null_goal.expand(batch_size, -1, -1)
        return out.to(device=device or out.device, dtype=dtype or out.dtype)

    def from_text(self, backbone, texts: list[str], contextual: bool = False) -> torch.Tensor:
        """(B strings) -> (B, n_goal_tokens, d).

        ``contextual=True`` 让指令**真的过一遍主干**，取最后一层隐状态；
        ``False`` 是原来的行为：只查输入嵌入表。

        原来那条路是这个模块的中心断点。它只调用 ``get_input_embeddings()``，
        也就是说 **28 层 transformer 在编码指令时一次都没被用到** —— goal 向量
        本质是一袋 token 嵌入，没有任何语义组合。实测后果：

        * 训练时没见过的同义句只有 18% 迁移成功（随机猜 12.5%）。
          "go straight ahead" 触发不了任何动作，因为它和 "move forward"
          一个 token 都不共享；而 "dig the block ahead" 成功，是因为它和
          "mine the block in front" 共享 ``the`` ``block`` —— 词面重叠，不是语义。
        * 八个随机向量与真实文本嵌入打平（1.2501 vs 1.2266）。
        * 换 3 倍大的主干毫无改善（1.2559 vs 1.2266）—— 更大的 LLM 不改变查表行为。

        —— 以上是修改前的诊断。**实测证明这个诊断是错的，所以默认值仍是 False。**

        把指令真的过一遍主干（``contextual=True``）重训一轮 20,000 步之后：

        * 同义句命中率 **23% -> 18%**，比不过主干还差；
        * 连训练见过的指令都退化了：``jump`` 从 0.90 掉到 0.33，
          ``turn left`` / ``strafe right`` 的按键几乎归零；
        * 无关句子反而开始触发强动作：``eat a sandwich`` -> ``attack=0.63``。

        推测：过主干后八个短句的最后一层表示被通用句法特征主导，彼此**更相似**
        而非更可分，于是连"区分八个类"这件本来轻松的事都变难了。

        真正的根因更可能在训练信号而非编码方式：用**八个固定类别**训练，goal 空间
        只需要八个可区分的点，无论向量怎么来，模型都只会学查表 —— 目标函数从未
        要求它泛化。改变这一点要靠 hindsight goals（目标变成连续的"实际达成了什么"），
        不是靠换编码器。详见 docs/stage1b-log.md。
        """
        if backbone.tokenizer is None:
            raise RuntimeError("external goals need a tokenizer; backbone.model_name is 'random'")
        emb = backbone.get_input_embeddings()
        rows = []
        for text in texts:
            if not text:
                rows.append(self.null_goal.to(emb.weight.device))
                continue
            ids = backbone.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            vectors = emb(ids["input_ids"].to(emb.weight.device))[0]  # (L, d)
            if contextual:
                hidden, _ = backbone(vectors.unsqueeze(0).to(backbone.dtype))
                vectors = hidden[0].to(self.null_goal.dtype)
            rows.append(self._fit(vectors))
        return torch.stack(rows)

    def _fit(self, vectors: torch.Tensor) -> torch.Tensor:
        """(L, d) -> (n_goal_tokens, d) by truncating or padding with the mean."""
        n, length = self.n_goal_tokens, vectors.shape[0]
        if length >= n:
            return vectors[:n]
        pad = vectors.mean(dim=0, keepdim=True).expand(n - length, -1)
        return torch.cat([vectors, pad], dim=0)
