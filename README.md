# FA2 With 4D Mask/Score in TileLang

Training-oriented FlashAttention-2 forward/backward kernels written in TileLang.
The public wrapper accepts Q/K/V from model attention code and supports:

- GQA/MQA: `Hq % Hk == 0`
- optional 4D bool mask `[B, Hq, Tq, Tk]`, where `True` means keep
- optional 4D float score `[B, Hq, Tq, Tk]`, added to `QK` before softmax scale
- causal mode without materializing a 4D causal mask
- BHSD `[B, H, T, D]` inputs backed by BTHD-contiguous tensors, matching common LLM attention states

## Install

Use an environment with PyTorch, CUDA, Triton, and einops available.  The pip
package currently pins only the TileLang runtime pieces used in this environment:
`tilelang==0.1.11` and `apache-tvm-ffi==0.1.11`.

Install directly from GitHub:

```bash
pip install "git+https://github.com/Sophie10001b/FA2-with-4D-mask.git"
```

For local development:

```bash
cd /path/to/FA2-with-4D-mask
pip install -e .
```

Quick syntax check:

```bash
python -m py_compile python/fa2.py python/modeling_attention.py tilelang_fa2_4d_mask/*.py benchmark/precision_check.py
```

## Use In `modeling_llm.py`

Low-level tensor API:

```python
from tilelang_fa2_4d_mask import tilelang_flash_attention

# query_states: [B, Hq, Tq, D]
# key_states:   [B, Hk, Tk, D]
# value_states: [B, Hk, Tk, D]
attn_output = tilelang_flash_attention(
    query_states,
    key_states,
    value_states,
    is_causal=True,
    scale=head_dim ** -0.5,
)
# attn_output is [B, Hq, Tq, D]
```

HuggingFace-style adapter:

```python
from tilelang_fa2_4d_mask import tilelang_attention_forward

attn_output, attn_weights = tilelang_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask=None,      # prefer None + is_causal=True for pure causal
    scaling=self.scaling,
    dropout=0.0,
    is_causal=True,
)
attn_output = attn_output.transpose(1, 2).contiguous()
```

If the existing attention path passes a 4D causal mask only to express causality,
prefer `attention_mask=None, is_causal=True` to avoid allocating `[B, H, T, T]`.
Pass a real 4D mask/score only when the model needs that extra structure.

## Parameters

`tilelang_flash_attention(...)` accepts:

| Parameter | Meaning |
|---|---|
| `query`, `key`, `value` | fp16/bf16 rank-4 tensors. Default layout is BHSD `[B, H, T, D]`. |
| `layout` | `bhsd` for `[B, H, T, D]`, `bshd` for `[B, T, H, D]`. Output uses the same layout. |
| `is_causal` | Uses the causal kernel path. Use this instead of a materialized causal mask when possible. |
| `attention_mask` | Optional bool mask or additive mask `[B, Hq or 1, Tq, Tk]`. Bool `True` means keep. Float masks follow PyTorch/HF scaled-logit convention by default. |
| `mask` | Explicit bool 4D mask. Can be combined with `score`. |
| `score` | Explicit float score added to raw `QK` before multiplying by `scale`. |
| `attention_mask_is_scaled` | If true, float `attention_mask` is divided by `scale` before passing to the kernel. |
| `use_atomic_backward` | Selects the atomic backward path. Default: true. |

Dropout and returning attention weights are not implemented.

## Checks At Sequence Length 8192

Environment:

- GPU: NVIDIA RTX PRO 5000 72GB Blackwell
- Torch: 2.9.1+cu128
- dtype: bf16
- shape: `B=1, Tq=Tk=8192, Hq=32, Hk=8, D=128`
- case: no 4D mask, no score

Precision command:

```bash
PYTHONPATH=. micromamba run -n llm python benchmark/precision_check.py \
  --device cuda:0 --dtype bfloat16 \
  --B 1 --Tq 8192 --Tk 8192 --Hq 32 --Hk 8 --D 128 \
  --cases no_causal causal --ref-block 128
```

Precision result:

| Case | TileLang vs eager max_abs | TileLang vs eager mean_abs | TileLang vs SDPA max_abs | TileLang vs SDPA mean_abs | Result |
|---|---:|---:|---:|---:|---|
| no_causal | 4.88281e-04 | 2.30019e-05 | 4.88281e-04 | 3.24496e-05 | PASS |
| causal | 1.56250e-02 | 4.14475e-05 | 3.90625e-03 | 5.41958e-05 | PASS |

Throughput command:

```bash
PYTHONPATH=. micromamba run -n llm python benchmark/throughput_check.py \
  --device cuda:0 --dtype bfloat16 \
  --B 1 --Tq-values 8192 --Tk 0 --Hq 32 --Hk 8 --D 128 \
  --cases no_causal causal --providers tilelang sdpa \
  --bench-modes fwd fwd_bwd --rep 10 --warmup 3
```

Estimated TFLOPS:

| Case | TileLang fwd | SDPA flash fwd | TileLang fwd+bwd | SDPA flash fwd+bwd |
|---|---:|---:|---:|---:|
| no_causal | 170.95 | 208.70 | 169.43 | 187.96 |
| causal | 155.25 | 191.67 | 163.37 | 188.11 |

Notes:

- TFLOPS use the benchmark script estimate: fwd counts `4 * B * Hq * Tq * Tk * D`, fwd+bwd counts `14 * ...`, with a 0.5 causal factor.
- The 4D mask/score path avoids constructing any `[B, H, T, T]` tensor when both mask and score are `None`.
- If a finite additive bias is already in scaled-logit units, pass it via `attention_mask` with `attention_mask_is_scaled=True` or divide it yourself before using `score`.
