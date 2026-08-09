"""The pretrained LLM, used as the shared trunk for everything.

This is deliberately *not* a planner that gets called from the outside. The
decoder runs on ``inputs_embeds`` built from vision / action / memory / event
tokens, and every head reads the same hidden states. The token embedding matrix
and the LM head stay reachable so that:

* concept tokens can be initialised from the LLM's own text embeddings
  (``embed_text``), which is what makes the grounding loss meaningful; and
* the language head and the prior-retention KL can reuse the original vocabulary.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import BackboneConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class LLMBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dtype = _DTYPES[cfg.dtype]
        self.model, self.tokenizer = _load(cfg, self.dtype)
        self.decoder = self.model.get_decoder()
        self.d_model = self.model.config.hidden_size

        if cfg.gradient_checkpointing:
            # transformers silently forces use_cache=False whenever gradient
            # checkpointing is on *and* the module is in training mode. Our
            # rollout carries history across timesteps in that very cache, so the
            # combination would quietly train a model with no temporal context
            # beyond the memory tokens -- and only in training mode, so eval would
            # look fine. Refuse instead of being subtly wrong.
            raise ValueError(
                "backbone.gradient_checkpointing is incompatible with the KV-cache "
                "rollout in WorldActionModel.step(): transformers disables use_cache "
                "under checkpointing, which drops all cross-timestep attention. "
                "Set it to false, or shorten train.seq_len if you are out of memory."
            )
        self.set_trainable_top_layers(cfg.n_trainable_top_layers)

    # -- parameter freezing -------------------------------------------------

    @property
    def layers(self) -> nn.ModuleList:
        return self.decoder.layers

    def set_trainable_top_layers(self, k: int) -> None:
        """Stage 1 -> k=0 (fully frozen); stage 2 -> top k blocks; stage 3 -> all."""
        for p in self.model.parameters():
            p.requires_grad = False
        if k <= 0:
            return
        layers = self.layers
        for layer in layers[max(0, len(layers) - k) :]:
            for p in layer.parameters():
                p.requires_grad = True
        # The final norm belongs with the top of the stack.
        norm = getattr(self.decoder, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True

    # -- embeddings ---------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled token embedding of each string, in LLM space.

        Used to initialise concept tokens ("oak log", "collect", "attack") and as
        the target of the semantic grounding loss.
        """
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer available (backbone.model_name == 'random')")
        emb = self.get_input_embeddings()
        out = []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            ids = ids.to(emb.weight.device)
            out.append(emb(ids)[0].mean(dim=0))
        return torch.stack(out)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """(B, L, d) -> last hidden state (B, L, d), plus the updated cache."""
        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return out.last_hidden_state, getattr(out, "past_key_values", None)

    def lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.model.get_output_embeddings()
        return head(hidden.to(head.weight.dtype))


def _load(cfg: BackboneConfig, dtype: torch.dtype):
    """Build the causal LM. ``model_name == "random"`` avoids any download."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    if cfg.model_name == "random":
        llama_cfg = LlamaConfig(
            vocab_size=cfg.random_vocab_size,
            hidden_size=cfg.random_hidden_size,
            intermediate_size=cfg.random_hidden_size * 2,
            num_hidden_layers=cfg.random_n_layers,
            num_attention_heads=cfg.random_n_heads,
            num_key_value_heads=cfg.random_n_heads,
            max_position_embeddings=8192,
            attn_implementation=cfg.attn_implementation,
        )
        return LlamaForCausalLM(llama_cfg).to(dtype), None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return model, tokenizer


class FrozenReference(nn.Module):
    """A frozen copy of the pretrained LLM for the prior-retention KL.

    L_retain = KL( p_theta(.|x) || p_pretrained(.|x) ) keeps online Minecraft RL
    from washing out the language and common-sense knowledge we started from.
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        ref_cfg = BackboneConfig(**{**cfg.__dict__, "n_trainable_top_layers": 0})
        self.backbone = LLMBackbone(ref_cfg)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> FrozenReference:
        return super().train(False)

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.backbone.get_input_embeddings()(input_ids)
        hidden, _ = self.backbone(emb)
        return self.backbone.lm_logits(hidden)
