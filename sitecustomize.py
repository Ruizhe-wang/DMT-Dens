# sitecustomize.py —— Python 启动时自动导入
import numpy as np
import torch

allowed = []

# 最常见的需要白名单的 numpy 类型
try:
    from numpy.core.multiarray import scalar as numpy_scalar
    allowed.append(numpy_scalar)
except Exception:
    pass

# 一些老 ckpt 里常见的对象
for t in [np.dtype, np.generic, np.number, np.integer, np.floating, np.bool_]:
    try:
        allowed.append(t)
    except Exception:
        pass

# 注册（幂等，重复调用没问题）
if allowed:
    torch.serialization.add_safe_globals(allowed)

# 调试：确认确实加载过（跑通后可删）
print("[sitecustomize] registered:", [getattr(a, "__name__", str(a)) for a in allowed])