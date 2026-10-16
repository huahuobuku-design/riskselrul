"""Explicit RNG key serialization (no Python hash())."""
import hashlib, numpy as np
FMT="WP05A|v1|ds={ds}|seed={seed}|split={split}|purpose={purpose}|stream={stream}|engine={engine}|wend={wend}|fault={fault}"
def key_str(ds,seed,split,purpose,stream,engine,wend,fault):
    for v in (ds,split,purpose,engine,fault): assert "|" not in str(v) and "=" not in str(v)
    return FMT.format(ds=ds,seed=int(seed),split=split,purpose=purpose,stream=int(stream),engine=engine,wend=wend,fault=fault)
def key_int(s): return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8],"big")
def rng_for(*a): s=key_str(*a); return np.random.default_rng(key_int(s)), s
def engine_order_key(engine): return hashlib.sha256(engine.encode("utf-8")).hexdigest()
