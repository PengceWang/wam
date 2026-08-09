from .base import MinecraftEnv, Observation
from .dummy import DummyMinecraftEnv

__all__ = [
    "BrowserCapture",
    "ConceptVocab",
    "Crop",
    "DummyMinecraftEnv",
    "MineRLEnv",
    "MinecraftEnv",
    "Observation",
]


def __getattr__(name: str):
    # Windows-only and needs the `capture` extra, so keep it out of the eager
    # import path -- tests and Linux checkouts must not need pywin32.
    if name in ("BrowserCapture", "Crop"):
        from . import browser

        return getattr(browser, name)
    # The mirror image: Linux-only, and importing it starts a JVM. Windows
    # checkouts must still be able to `import wam.envs`.
    if name in ("MineRLEnv", "ConceptVocab"):
        from . import minerl

        return getattr(minerl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
