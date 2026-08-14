"""把 1023 个概念聚成 K 个活动类，给 hindsight 目标一个离散的形式。

    python scripts/cluster_concepts.py --k 16 --out /data/wam/6xx_clusters.npz

为什么要离散化：实测**指令目标能驱动行为，hindsight 目标不能**。

    注入                          泥土事件   attack
    指令:mine the block in front     9.0     0.512
    不注入                           5.2     0.628
    指令:move forward                0.6     0.252     <- 单调，方向正确

    hindsight「木头」                 —      木头事件 0.4，与不注入完全相同

区别不在通道带宽（两者都是 4 个 goal token），在**形式**：指令是 8 个边界清晰的
离散类，``goal -> action`` 的映射干净；hindsight 目标是几百上千种概念组合的连续
平均，每个都略有不同，模型无法把它当作一个明确的指令来响应。

所以这里做的是：保留 hindsight 的优点（无需人工标注、覆盖全量数据、类别数远多于 8），
拿回指令的优点（边界清晰、可驱动）。聚类跑在**中心化过的**短语嵌入上 ——
未中心化时任意两个概念的余弦都在 0.9 以上，聚出来的类没有意义。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def kmeans(x: torch.Tensor, k: int, iters: int = 100, seed: int = 0) -> tuple:
    """余弦距离下的 k-means（向量先归一化，等价于球面 k-means）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.nn.functional.normalize(x, dim=-1)
    # k-means++ 式的初始化：第一个随机，之后每个都挑离已有中心最远的
    idx = [int(torch.randint(len(x), (1,), generator=g))]
    for _ in range(k - 1):
        d = 1.0 - (x @ x[idx].T).max(dim=-1).values
        idx.append(int(d.argmax()))
    c = x[idx].clone()

    assign = torch.zeros(len(x), dtype=torch.long)
    for _ in range(iters):
        new = (x @ c.T).argmax(dim=-1)
        if torch.equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = torch.nn.functional.normalize(x[m].mean(0), dim=-1)
    return assign, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3-0.6b.yaml")
    ap.add_argument("--concepts", default="/data/wam/6xx_concepts.json")
    ap.add_argument("--out", default="/data/wam/6xx_clusters.npz")
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    from wam.config import WAMConfig
    from wam.data.events import phrase_embedding_table
    from wam.model.wam import WorldActionModel

    cfg = WAMConfig.from_yaml(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldActionModel(cfg).to(dev)
    model.eval()

    emb = phrase_embedding_table(model, args.concepts, dev, center=True).cpu()
    raw = json.loads(Path(args.concepts).read_text(encoding="utf-8"))
    i2p = {int(v): k for k, v in raw["phrases"].items()}
    live = sorted(i2p)                      # id 0 是 PAD，不参与

    assign, cent = kmeans(emb[live], args.k)
    cluster_of = np.zeros(cfg.event.vocab_size, dtype=np.int64) - 1
    for i, cid in enumerate(live):
        cluster_of[cid] = int(assign[i])

    print(f"{len(live)} 个概念 -> {args.k} 个活动类\n")
    for j in range(args.k):
        members = [i2p[c] for c in live if cluster_of[c] == j]
        print(f"  类 {j:>2}  ({len(members):>3} 个)  {', '.join(members[:6])}"
              + (" ..." if len(members) > 6 else ""))

    np.savez_compressed(args.out, cluster_of=cluster_of, centroids=cent.numpy(), k=args.k)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
