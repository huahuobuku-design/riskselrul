"""Explicit RNG key serialization (no Python ``hash()``).

Every stochastic draw in the package is addressed by a canonical key string built
from the run coordinates (dataset, seed, split, purpose, stream, engine, window
end, fault tag). The key is hashed with SHA-256 and the first 64 bits seed a
``numpy.random.default_rng``, so the whole pipeline is reproducible across
platforms and Python builds.
"""

import hashlib

import numpy as np

# NOTE: the "WP05A|v1|" namespace prefix is FROZEN. It encodes the historical
# origin of the seeds. Changing it would alter every derived RNG key and thereby
# invalidate all released splits, fault draws and result files. Do not edit.
FMT = ("WP05A|v1|ds={ds}|seed={seed}|split={split}|purpose={purpose}|"
       "stream={stream}|engine={engine}|wend={wend}|fault={fault}")


def key_str(ds, seed, split, purpose, stream, engine, wend, fault):
    """Build the canonical key string for one stochastic draw."""
    for v in (ds, split, purpose, engine, fault):
        assert "|" not in str(v) and "=" not in str(v)
    return FMT.format(ds=ds, seed=int(seed), split=split, purpose=purpose,
                      stream=int(stream), engine=engine, wend=wend, fault=fault)


def key_int(s):
    """Deterministic 64-bit integer seed from a key string."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def rng_for(*a):
    """Return ``(rng, key_string)`` for the given run coordinates."""
    s = key_str(*a)
    return np.random.default_rng(key_int(s)), s


def engine_order_key(engine):
    """Label-free tie-break key: SHA-256 hex of the engine identifier."""
    return hashlib.sha256(engine.encode("utf-8")).hexdigest()
