"""从 contractor 的 ``meta_info`` 树取出事件与世界标志，供 world head 监督。

**这里的事件是逐帧增量，不是累计值** —— 和实时模拟器恰好相反。实测
``sprint_one_cm`` 在连续帧上是 27 / 28 / 28，20fps × 28cm = 5.6 m/s，正是 Minecraft
的冲刺速度，所以这是"这一帧发生了多少"。:class:`wam.envs.minerl.MineRLEnv` 那边
拿到的是整局累计值，必须做差分；**同一套代码不能同时对付两边**，这也是
docs/measurements.md 专门记下这条的原因。

覆盖率：``meta_info`` 只有 2,944 / 3,658 个 episode（80.4%），按窗口算 74.8%。
所以事件监督天然只能作用在一部分数据上，:class:`Stage1bData` 会据此过滤。
"""

from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path

import numpy as np
import torch

CHUNK_FRAMES = 32

# WorldHead.DISCRETE_FLAGS 的顺序，不能改
FLAGS = ("block_removed", "inventory_changed", "contact", "new_area", "done")
# meta_info 里没有 health，也没有 biome，所以这两位无法从离线数据推出。
# 与其编一个假标签，不如把它们屏蔽掉 —— 对着不存在的标签算出来的项不是正则化，
# 是带权重的噪声（docs 原话）。
FLAG_SUPERVISED = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)


def meta_table(root: str | Path) -> dict[str, tuple[str, int, int]]:
    """episode -> (meta_info 分片路径, 分片内下标, 帧数)。"""
    import lmdb

    out: dict[str, tuple[str, int, int]] = {}
    for p in sorted(glob.glob(str(Path(root) / "meta_info" / "part-*"))):
        e = lmdb.open(p, readonly=True, lock=False)
        with e.begin() as t:
            for i in pickle.loads(t.get(b"__chunk_infos__")):
                out[i["episode"]] = (p, i["episode_idx"], i["num_frames"])
        e.close()
    return out


def load_concepts(path: str | Path) -> dict[str, int]:
    """``scripts/build_concept_vocab.py`` 产出的 key -> concept id 表。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: int(v) for k, v in raw["key_to_id"].items()}


def read_meta(shard: str, episode_idx: int, start: int, n: int) -> list[dict]:
    """取 [start, start+n) 的逐帧 meta_info。"""
    import lmdb

    env = lmdb.open(shard, readonly=True, lock=False, readahead=False)
    first = (start // CHUNK_FRAMES) * CHUNK_FRAMES
    last = ((start + n + CHUNK_FRAMES - 1) // CHUNK_FRAMES) * CHUNK_FRAMES
    rows: list[dict] = []
    with env.begin() as txn:
        for off in range(first, last, CHUNK_FRAMES):
            raw = txn.get(str((episode_idx, off)).encode())
            if raw is None:
                break
            rows.extend(pickle.loads(raw))
    env.close()
    lo = start - first
    return rows[lo:lo + n]


@torch.no_grad()
def phrase_embedding_table(model, concepts_path: str | Path, device,
                           center: bool = True) -> torch.Tensor:
    """(vocab_size, d_model)：每个概念 id 对应的 LLM 短语嵌入。

    ``center=True`` 会减去全局均值，**这对 hindsight 目标是必需的**。

    LLM 的隐状态有各向异性：所有向量挤在一个很窄的锥里。实测这 1,023 个短语
    两两余弦都在 0.89~0.97 之间，按活动类型分组后同类 0.950 / 异类 0.913，
    只差 **0.037** —— 余弦损失在这样的目标上几乎没有信号，模型只要输出那个
    公共方向就能拿到 0.93，根本学不到"目标是什么"。（现象上就是 goal 损失
    100 步内从 1.03 掉到 0.07，掉得可疑地快。）

    减去全局均值之后：同类 0.511 / 异类 0.098，差 **0.413**，分离度提高 11 倍。
    矩阵也变得有意义了 —— 砍树↔杀怪 -0.042（几乎正交），挖石头↔挖矿 0.276
    （都属采矿）。

    ``seed_concept_embeddings`` 那边**不做中心化**：它只是 ``EventEmbedding``
    的初值，之后还要被梯度继续训练，而且事件预测走的是 BCE 不是余弦，
    实测未中心化就能拿到 mAP 0.647。不去动一个已经验证有效的东西。
    """
    import json

    raw = json.loads(Path(concepts_path).read_text(encoding="utf-8"))
    id_to_phrase = {int(v): k for k, v in raw["phrases"].items()}
    backbone = model.backbone
    d = model.d_model
    table = torch.zeros(model.cfg.event.vocab_size, d, device=device)
    if backbone.tokenizer is None:
        return table
    for cid, phrase in sorted(id_to_phrase.items()):
        tok = backbone.tokenizer(phrase, add_special_tokens=False, return_tensors="pt")
        emb = backbone.get_input_embeddings()(tok["input_ids"].to(device))
        hidden, _ = backbone(emb.to(backbone.dtype))
        table[cid] = hidden[0].float().mean(dim=0)
    if center:
        live = table.norm(dim=-1) > 0
        table[live] = table[live] - table[live].mean(0, keepdim=True)
    return table


@torch.no_grad()
def seed_concept_embeddings(model, concepts_path: str | Path, device) -> int:
    """把概念行从「随机初始化」换成「LLM 对该短语的理解」。

    ``EventEmbedding.init_from_text`` 一直没有调用点 —— docs/getting-started.md 明确
    记着这个缺口：*"nothing yet feeds those phrases through the LLM's embedding layer
    into EventEmbedding.init_from_text at model construction. Until that is wired,
    concept ids are learned from scratch."* 不接上的话，1024 行就是一张随机查找表，
    ``mined oak log`` 和 ``mined dark oak log`` 之间没有任何关系。

    这里**让短语真的过一遍主干**，取最后一层隐状态的均值，而不是像
    ``GoalEncoder.from_text`` 那样只查 embedding 表。差别是实打实的：只查表的话
    "dark oak log" 只是三个词向量堆在一起，没有组合语义。
    """
    import json

    import torch as _t

    raw = json.loads(Path(concepts_path).read_text(encoding="utf-8"))
    id_to_phrase = {int(v): k for k, v in raw["phrases"].items()}
    backbone = model.backbone
    if backbone.tokenizer is None:
        return 0

    ids, vecs = [], []
    for cid, phrase in sorted(id_to_phrase.items()):
        tok = backbone.tokenizer(phrase, add_special_tokens=False, return_tensors="pt")
        emb = backbone.get_input_embeddings()(tok["input_ids"].to(device))
        hidden, _ = backbone(emb.to(backbone.dtype))
        ids.append(cid)
        vecs.append(hidden[0].float().mean(dim=0))
    model.event_embedding.init_from_text(ids, _t.stack(vecs))
    return len(ids)


def _inventory_key(frame: dict) -> tuple:
    inv = frame.get("inventory") or []
    return tuple(sorted((str(s.get("type")), int(s.get("quantity", 0))) for s in inv))


def window_events(
    frames: list[dict],
    key_to_id: dict[str, int],
    seq_len: int,
    chunk: int,
    n_event_tokens: int,
    vocab_size: int,
    is_last_window: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐帧 meta_info -> (event_ids, event_targets, flags, hindsight)。

    ``event_ids[t]``      第 t 步观察到的概念（进入序列的输入 token）
    ``event_targets[t]``  第 t+1 步会发生什么（world head 的多标签目标）
    ``flags[t]``          第 t+1 步的世界标志
    ``hindsight[t]``      第 t+1 步到窗口末尾之间**实际达成了什么**（多热）。

    hindsight 是「事后重标注」的原料：任何一条轨迹都是「到达它最终所处状态」的
    成功示范，所以不需要人工编写目标标签 —— 事件流已经说明了达成了什么。
    这也是为什么它必须排在事件管道之后：没有事件，「达成了什么」根本无从定义。
    """
    ids = np.zeros((seq_len, n_event_tokens), dtype=np.int64)
    targets = np.zeros((seq_len, vocab_size), dtype=np.float32)
    flags = np.zeros((seq_len, len(FLAGS)), dtype=np.float32)
    hind = np.zeros((seq_len, vocab_size), dtype=np.float32)

    per_step: list[list[int]] = []
    mined: list[bool] = []
    inv_changed: list[bool] = []

    for t in range(seq_len):
        lo, hi = t * chunk, (t + 1) * chunk
        seen: dict[int, None] = {}
        saw_mine = False
        for f in frames[lo:hi]:
            for key, count in (f.get("events") or {}).items():
                if not count:
                    continue
                cid = key_to_id.get(key)
                if cid is None:  # custom 统计量，或没进词表的稀有概念
                    continue
                seen[cid] = None
                if key.startswith("minecraft.mine_block"):
                    saw_mine = True
        per_step.append(list(seen))
        mined.append(saw_mine)
        a = _inventory_key(frames[lo]) if lo < len(frames) else ()
        b = _inventory_key(frames[min(hi, len(frames)) - 1]) if frames else ()
        inv_changed.append(a != b)

    for t in range(seq_len):
        for i, cid in enumerate(per_step[t][:n_event_tokens]):
            ids[t, i] = cid
        nxt = t + 1
        if nxt < seq_len:
            for cid in per_step[nxt]:
                targets[t, cid] = 1.0
            flags[t, 0] = float(mined[nxt])
            flags[t, 1] = float(inv_changed[nxt])
            flags[t, 4] = 0.0
        else:
            # 最后一步没有"下一步"可看；done 只有在这确实是本集最后一个窗口时才为真
            flags[t, 4] = float(is_last_window)
        # 从 t+1 到窗口末尾，累计实际发生过的一切
        for u in range(t + 1, seq_len):
            for cid in per_step[u]:
                hind[t, cid] = 1.0
    return ids, targets, flags, hind
