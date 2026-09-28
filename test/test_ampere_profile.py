from auto_round.cli.parser import build_quantize_parser
from auto_round.cli.profiles import ampere_quality_argv

MODEL = "Qwen/Qwen3.5-4B"


def _parse(extra=None):
    argv = [MODEL]
    if extra:
        argv.extend(extra)
    return build_quantize_parser().parse_args(ampere_quality_argv(argv))

def test_ampere_quality_profile_defaults():
    args = _parse()

    assert args.model == MODEL
    assert args.scheme == "W8A16"
    assert args.group_size == 128
    assert args.format == "auto_gptq"
    assert args.ignore_layers == "linear_attn,mtp"
    assert args.batch_size == 1
    assert args.gradient_accumulate_steps == 1
    assert args.low_gpu_mem_usage is True
    assert args.enable_deterministic_algorithms is True

def test_ampere_quality_profile_explicit_overrides_win():
    args = _parse(
        [
            "--scheme",
            "W4A16",
            "--group_size",
            "64",
            "--format",
            "auto_round",
            "--ignore_layers",
            "vision_tower",
            "--batch_size",
            "4",
            "--gradient_accumulate_steps",
            "2",
            "--no-low_gpu_mem_usage",
            "--no-enable_deterministic_algorithms",
        ]
    )

    assert args.scheme == "W4A16"
    assert args.group_size == 64
    assert args.format == "auto_round"
    assert args.ignore_layers == "vision_tower"
    assert args.batch_size == 4
    assert args.gradient_accumulate_steps == 2
    assert args.low_gpu_mem_usage is False
    assert args.enable_deterministic_algorithms is False
