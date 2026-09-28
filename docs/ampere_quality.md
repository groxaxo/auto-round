# Ampere quality profile

`auto-round-ampere-best` is a quality-first post-training quantization
profile for NVIDIA Ampere GPUs such as RTX 30-series and A100.

The profile intentionally combines only optimizations that have a compatible
deployment path:

| Setting | Default | Reason |
|---|---|---|
| Weight scheme | W8A16 | Conservative weight-only compression for fidelity |
| Group size | 128 | Good scale locality with mature kernels |
| Symmetry | symmetric | Required by the preferred Marlin GPTQ path |
| Tuning recipe | AutoRoundBest | 1000 iters, 512 samples, 2048-token calibration |
| Batch / accumulation | 1 / 8 | Preserves effective tuning batch with low VRAM |
| GPU memory mode | low | Keeps 4B-8B tuning practical on 12-24 GB cards |
| Reproducibility | deterministic | Makes A/B comparisons repeatable |
| Export | auto_gptq | Compatible with CUDA GPTQ/Marlin runtimes |
| Protected state paths | linear_attn, mtp | Keep hybrid state-sensitive and MTP modules in source precision |

## Reference run

The reference model for this profile is `Qwen/Qwen3.5-4B`:

```bash
auto-round-ampere-best Qwen/Qwen3.5-4B \
  --output_dir ./qwen35-4b-ampere-w8-g128
```
For production use, point --dataset at a representative local calibration
set containing the same languages, coding/tool patterns, and prompt lengths as
the target workload.

Explicit flags override profile defaults:

```bash
auto-round-ampere-best Qwen/Qwen3.5-4B \
  --scheme W4A16 \
  --group_size 64 \
  --format auto_round \
  --batch_size 4 \
  --no-low_gpu_mem_usage
```

The profile is warning-only on non-Ampere hardware, so it remains usable for
debug/export work while making accidental execution on the wrong accelerator
visible.

## Why these methods are not blindly stacked

- AWQ scaling is not forced for this W8A16 weight-only profile. AutoRound can
  combine with AWQ, but the strongest reason to add activation-aware smoothing
  is more aggressive activation quantization.
- FP8 is not selected because Ampere does not provide the Hopper/Blackwell
  native FP8 Transformer Engine path.
- Hadamard/SpinQuant rotations are not enabled because the current AutoRound
  deployment matrix marks the rotation path as lacking a production kernel.
- KV-cache precision is deliberately left unchanged. Weight quantization and
  KV quantization are independent error sources and should be validated
  separately.
- The language-model head remains unquantized unless --quant_lm_head is
  supplied.
- Qwen3.5 linear-attention modules and the entire MTP branch are kept in
  source precision by default via --ignore_layers linear_attn,mtp. An explicit
  --ignore_layers/--fp_layers value replaces this profile default.

## Marlin serving path

The auto_gptq artifact uses a layout accepted by AutoRound's CUDA inference
backends. For Marlin, install a compatible GPTQModel build:

```bash
python -m pip install "gptqmodel>=2.0"
```

Marlin support is limited by the installed runtime and model architecture.
Always load and benchmark the exported artifact on the target GPU before
promotion.

## Acceptance criteria

Compare the quantized artifact with the BF16 reference using the same prompts
and seeds. Record task correctness plus TTFT, output tokens/s, peak VRAM and
long-context behavior. A speedup is only useful when the task-quality floor is
maintained.
