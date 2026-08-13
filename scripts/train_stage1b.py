"""Stage 1b：把 world head 的事件与标志预测打开。

    python scripts/train_stage1b.py --max-steps 20000 --batch-size 16

stage 1a 只训了 world head 四个输出里的一个（``next_latent``），另外三个
（``event_logits`` / ``flag_logits`` / ``health_delta``）前向照跑，却从没收到过梯度 ——
因为 1a 只加载 image 和 action 两棵树，那些标签根本不存在。

打开事件预测，模型才能回答"这么做**会发生什么**"，而不只是"下一帧长什么样"。
这是任务级规划的地基：要判断"继续挥砍会不会掉出原木"，需要的是事件粒度的预测，
不是帧粒度的。

``health_delta`` 仍然关着 —— meta_info 里没有血量。``contact`` 和 ``new_area``
两个标志同理（没有血量、没有生物群系），由 ``FLAG_SUPERVISED`` 屏蔽掉：
对着不存在的标签算出来的项不是正则化，是带权重的噪声。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam.config import WAMConfig
from wam.data.stage1a import goal_for, instruction_goal_tokens
from wam.data.stage1b import Stage1bData
from wam.model.wam import WorldActionModel
from wam.training.losses import action_loss, goal_loss, next_latent_loss
from wam.training.trainer import configure_stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3-0.6b.yaml")
    ap.add_argument("--root", default="/data/wam/6xx")
    ap.add_argument("--index", default="/data/wam/6xx_index_seq8.npz")
    ap.add_argument("--concepts", default="/data/wam/6xx_concepts.json")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default="/data/wam/checkpoints/stage1b.pt")
    ap.add_argument("--init-from", default=None,
                    help="从已有 checkpoint 接着训。同一个模型、同一套权重，"
                         "只是多打开两项损失，没必要从头再来。")
    ap.add_argument("--hidden-mult", type=int, default=None,
                    help="覆盖 heads.hidden_mult。这是各个头内部 MLP 的宽度，也就是"
                         "「LLM 到动作之间那段网络」的容量旋钮。默认 2。")
    ap.add_argument("--all-windows", action="store_true",
                    help="用上每一个有 meta_info 的窗口，不再要求带指令标签。"
                         "事件预测是自监督的，那个筛选是 stage 1a 的遗留，砍掉了 65% 的数据。")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--event-weight", type=float, default=1.0)
    ap.add_argument("--flag-weight", type=float, default=1.0)
    ap.add_argument("--latent-weight", type=float, default=0.5)
    ap.add_argument("--goal-cond-prob", type=float, default=0.0,
                    help="以这个概率把 hindsight 目标**放进 goal 槽**，让 actor 学会"
                         "以它为条件行动。这是 hindsight replay 缺掉的另一半：只训 GoalHead "
                         "产出目标向量，actor 却从没以这种形状的目标为输入训练过，"
                         "实测注入目标完全无法改变行为（注入「木头」得到 0.4 次木头事件，"
                         "和注入「泥土」、和不注入完全一样）。")
    ap.add_argument("--goal-weight", type=float, default=0.0,
                    help="事后重标注（hindsight goals）的权重。>0 时把「接下来实际达成了"
                         "什么」当作「它本来的目标」，让 goal 空间以结果为坐标，而不是"
                         "查找表的第几行。默认 0（关闭），因为它必须排在事件管道之后 —— "
                         "没有事件，「达成了什么」无从定义。")
    args = ap.parse_args()

    cfg = WAMConfig.from_yaml(args.config)
    cfg.train.stage = 1
    cfg.train.batch_size = args.batch_size
    if args.hidden_mult:
        cfg.heads.hidden_mult = args.hidden_mult
    cfg.event.concept_vocab_path = args.concepts

    import numpy as np

    index_seq = int(np.load(Path(args.index), allow_pickle=True)["seq_len"])
    if cfg.train.seq_len != index_seq:
        print(f"seq_len {cfg.train.seq_len} -> {index_seq} (由索引决定)")
        cfg.train.seq_len = index_seq

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = WorldActionModel(cfg).to(device)
    configure_stage(model, cfg)

    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        print(f"接续自 {args.init_from}（缺失 {len(missing)} / 多余 {len(unexpected)} 个张量）")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device {device} | 总计 {total / 1e6:.1f}M，可训练 {trainable / 1e6:.1f}M "
          f"({100 * trainable / total:.1f}%) | hidden_mult={cfg.heads.hidden_mult}")
    print(f"  ├ WorldHead      {sum(p.numel() for p in model.world_head.parameters()) / 1e6:.1f}M")
    print(f"  ├ vision 适配层  {sum(p.numel() for p in model.vision.parameters() if p.requires_grad) / 1e6:.1f}M")
    print(f"  └ ActionHead     {sum(p.numel() for p in model.action_head.parameters()) / 1e6:.1f}M")

    # 概念行从随机换成「LLM 对该短语的理解」。不接这一步，1024 行就是随机查找表，
    # mined oak log 和 mined dark oak log 之间毫无关系。
    from wam.data.events import seed_concept_embeddings

    n_seed = seed_concept_embeddings(model, args.concepts, device)
    print(f"用 LLM 短语嵌入初始化了 {n_seed:,} 个概念行（init_from_text，此前从未被调用）")

    phrase_emb = None
    if args.goal_weight > 0:
        from wam.data.events import phrase_embedding_table

        # 和 EventEmbedding 用同一张表：否则「模型脑子里的 mined oak log」和
        # 「hindsight 说你达成了 mined oak log」会是两个不同的向量。
        phrase_emb = phrase_embedding_table(model, args.concepts, device)
        print(f"hindsight goals 已启用，权重 {args.goal_weight}"
              f"（短语嵌入表 {tuple(phrase_emb.shape)}）")

    data = Stage1bData(cfg, args.root, args.index, args.concepts,
                       workers=args.workers, all_windows=args.all_windows)
    goal_table = instruction_goal_tokens(model, device)
    flag_mask = data.flag_mask.to(device)
    print(f"监督的标志位: {[n for n, m in zip(('block_removed','inventory_changed','contact','new_area','done'), data.flag_mask) if m]}")

    optimiser = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.train.weight_decay)
    model.train()
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        batch, labels = data.batch(args.batch_size)
        batch = batch.to(device)
        b = batch.pixels.shape[0]

        # 有指令标签的取 goal 表，没有的填 null —— 两种来源本来就共用同一个 LayerNorm，
        # 架构上就是为这个交接设计的。
        null = model.goal_encoder.null(1, device=device, dtype=goal_table.dtype)
        goal = torch.stack([goal_table[l] if l >= 0 else null[0] for l in labels])

        if args.goal_cond_prob > 0 and phrase_emb is not None and batch.hindsight is not None:
            # 目标条件化的行为克隆：告诉模型「你接下来会达成 X」，让它预测那些
            # 真的达成了 X 的动作。hindsight[:, 0] 是整个窗口内达成的一切。
            h0 = batch.hindsight[:, 0].float()
            n0 = h0.sum(-1, keepdim=True)
            tgt0 = (h0 @ phrase_emb) / n0.clamp(min=1.0)
            use = (torch.rand(b, device=device) < args.goal_cond_prob) & (n0.squeeze(-1) > 0)
            repl = tgt0.unsqueeze(1).expand(-1, cfg.heads.n_goal_tokens, -1).to(goal.dtype)
            goal = torch.where(use[:, None, None], repl, goal)
        state = model.initial_state(b, device=device)
        state = type(state)(memory=state.memory, prev_action=state.prev_action,
                            goal=goal, goal_is_external=True)

        out = model(batch.pixels, batch.actions, batch.event_ids, state=state)

        actor = action_loss(out.action, batch.actions)
        loss = cfg.loss.actor * actor
        parts = {"actor": actor.detach()}

        if out.visual_latent.shape[1] > 1:
            latent = next_latent_loss(out.world["next_latent"][:, :-1], out.visual_latent[:, 1:])
            loss = loss + args.latent_weight * latent
            parts["latent"] = latent.detach()

        # 事件：1024 维多标签。正样本极稀疏，所以用 BCE 而不是 softmax。
        event = F.binary_cross_entropy_with_logits(
            out.world["event_logits"][:, :-1].float(), batch.event_targets[:, :-1].float())
        loss = loss + args.event_weight * event
        parts["event"] = event.detach()

        # 标志：只在能推出来的那几位上算，其余被 flag_mask 置零
        fl = F.binary_cross_entropy_with_logits(
            out.world["flag_logits"].float(), batch.flags.float(), reduction="none")
        flag = (fl * flag_mask).sum() / (flag_mask.sum() * fl.shape[0] * fl.shape[1])
        loss = loss + args.flag_weight * flag
        parts["flag"] = flag.detach()

        if phrase_emb is not None and batch.hindsight is not None:
            # 多热 -> goal 空间的一个向量：把这一步之后实际达成的所有概念的
            # 短语嵌入取平均。没有任何事发生的步被 mask 掉 —— 不是每个时刻都有
            # 「达成」，对着空集编一个目标是噪声。
            h = batch.hindsight.float()
            n = h.sum(-1, keepdim=True)
            tgt = (h @ phrase_emb) / n.clamp(min=1.0)
            gmask = (n.squeeze(-1) > 0).float()
            g = goal_loss(out.goal, tgt, gmask)
            loss = loss + args.goal_weight * g
            parts["goal"] = g.detach()
            parts["goal%"] = gmask.mean().detach()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
        optimiser.step()

        if step % args.save_every == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step},
                       args.out)

        if step % args.log_every == 0 or step == 1:
            rate = step / (time.time() - t0)
            shown = " ".join(f"{k}={float(v):.4f}" for k, v in sorted(parts.items()))
            print(f"{step:>6} {shown} grad={float(grad):.2f} total={float(loss):.4f} "
                  f"{rate:.2f} it/s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, args.out)
    print(f"saved {args.out}")
    data.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
