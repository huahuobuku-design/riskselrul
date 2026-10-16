"""Deployment of lean, numeric-only states for edge inference."""

from .lean import decide, features, load, predict, risk

__all__ = ["decide", "features", "load", "predict", "risk"]
