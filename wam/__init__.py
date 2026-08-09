"""Pretrained Developmental World-Action Model.

A pretrained ~0.8B LLM used as the *central backbone* of a Minecraft agent:
vision, action, world prediction, value, goals and memory all share one
transformer and one hidden state. See ``readme.md`` for the design it implements.
"""

from .config import WAMConfig
from .model import ActionChunk, RolloutState, WorldActionModel

__version__ = "0.1.0"

__all__ = ["ActionChunk", "RolloutState", "WAMConfig", "WorldActionModel"]
