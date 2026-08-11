"""从 contractor 的 event 树建概念词表，供 world head 的事件预测使用。

    python scripts/build_concept_vocab.py --root /data/wam/6xx --out /data/wam/6xx_concepts.json

两个约束决定了这个脚本的做法：

* **措辞必须和 :mod:`wam.envs.minerl` 一致。** ``_EVENT_VERB`` 把 ``mine_block`` 写成
  "mined"，``pickup`` 写成 "picked up"。离线训练若用另一套写法，同一个概念在离线和
  在线就是两个 id，而 ``EventEmbedding.init_from_text`` 是按 id 取 LLM 词嵌入来初始化的
  —— id 对不上等于把概念指向了别人的语义。
* **id 一经分配不得变动。** 与 ``ConceptVocab`` 同理：追加式，按频率降序固定顺序，
  id 0 永远是 PAD。

``minecraft.custom:*`` 全部排除：它们是统计量（play_one_minute、sprint_one_cm），
每帧都触发，既不是"发生了什么"，塞进词表还会挤掉真正稀有的概念。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

# 与 wam/envs/minerl.py 的 _EVENT_VERB 保持一致
VERB = {
    "mine_block": "mined",
    "pickup": "picked up",
    "craft_item": "crafted",
    "break_item": "broke",
    "kill_entity": "killed",
    "damage_dealt": "damaged",
    "use_item": "used",
    "drop": "dropped",
    "entity_killed_by": "killed by",
}
EXCLUDE_PREFIX = ("minecraft.custom",)


def phrase_of(key: str) -> str | None:
    """``minecraft.mine_block:minecraft.oak_log`` -> ``mined oak log``。"""
    if key.startswith(EXCLUDE_PREFIX):
        return None
    kind, _, item = key.partition(":")
    kind = kind.removeprefix("minecraft.")
    verb = VERB.get(kind)
    if verb is None:
        return None
    item = item.removeprefix("minecraft.").replace("_", " ")
    return f"{verb} {item}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/wam/6xx")
    ap.add_argument("--out", default="/data/wam/6xx_concepts.json")
    ap.add_argument("--size", type=int, default=1024, help="要和 EventConfig.vocab_size 一致")
    args = ap.parse_args()

    import lmdb

    env = lmdb.open(str(Path(args.root) / "event"), readonly=True, lock=False)
    with env.begin() as txn:
        info = pickle.loads(txn.get(b"__event_info__"))
    env.close()
    print(f"event 树里 {len(info):,} 种事件")

    rows = []
    for key, meta in info.items():
        p = phrase_of(key)
        if p is None:
            continue
        rows.append((p, key, int(meta.get("__num_items__", 0)), int(meta.get("__num_episodes__", 0))))

    # 同一个短语可能对应多个 key（理论上不该有，但别假设），合并计数
    merged: dict[str, dict] = {}
    for p, key, n_items, n_eps in rows:
        m = merged.setdefault(p, {"keys": [], "n_items": 0, "n_episodes": 0})
        m["keys"].append(key)
        m["n_items"] += n_items
        m["n_episodes"] = max(m["n_episodes"], n_eps)

    # 按出现次数降序 —— 顺序即 id，所以这个排序一旦定下就不能再变
    order = sorted(merged.items(), key=lambda kv: (-kv[1]["n_items"], kv[0]))
    keep = order[: args.size - 1]  # id 0 留给 PAD

    phrases = {p: i + 1 for i, (p, _) in enumerate(keep)}
    key_to_id = {}
    for p, m in keep:
        for k in m["keys"]:
            key_to_id[k] = phrases[p]

    print(f"排除 custom 统计量后 {len(merged):,} 个概念，收进词表 {len(phrases):,} 个")
    dropped = len(merged) - len(phrases)
    if dropped:
        tail = order[args.size - 1:]
        print(f"丢弃 {dropped} 个最稀有的概念（最高频的被丢者出现 {tail[0][1]['n_items']:,} 次）")

    print("\n最高频的 12 个：")
    for p, m in order[:12]:
        print(f"  {phrases[p]:>4}  {p:<34} {m['n_items']:>12,} 次 / {m['n_episodes']:>5} 集")

    Path(args.out).write_text(
        json.dumps({"phrases": phrases, "key_to_id": key_to_id}, indent=2, sort_keys=True,
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
