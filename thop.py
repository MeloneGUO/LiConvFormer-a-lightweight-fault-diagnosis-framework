"""
Lightweight fallback stub for `thop` (FLOPS profiler).

This module provides `profile` and `clever_format` functions as a fallback
when the official `thop` package is not available or cannot compute FLOPS
(e.g., in offline environments or due to model/package compatibility issues).

In such cases, this stub returns:
- FLOPS: 0 (placeholder; accurate FLOPS requires the official thop package)
- Params: actual parameter count from the model

For accurate FLOPS computation, install the official thop:
    pip install thop
    conda install -c conda-forge thop

Once installed, Python will use the official package instead of this stub.
"""

def profile(model, inputs=None):
    """
    Fallback profile function.
    
    Returns (flops, params) where:
    - flops: 0 (placeholder value)
    - params: actual model parameter count if available, else 0
    """
    try:
        params = sum(p.numel() for p in getattr(model, 'parameters', lambda: [])())
    except Exception:
        params = 0
    flops = 0
    return flops, params


def clever_format(vals, fmt="%0.3f"):
    """
    Fallback clever_format function.
    
    Converts a list of values to readable strings.
    """
    out = []
    for v in vals:
        try:
            out.append(str(v))
        except Exception:
            out.append('0')
    return out
