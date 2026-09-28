# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Hardware-oriented CLI profiles.

Profiles inject conservative defaults before the user's argv. Because argparse
uses the last occurrence for scalar options, explicit user flags still win.
"""

from __future__ import annotations

from collections.abc import Sequence

from auto_round.logger import logger


AMPERE_QUALITY_DEFAULTS = (
    (("--scheme",), ("--scheme", "W8A16")),
    (("--group_size",), ("--group_size", "128")),
    (("--format", "--formats"), ("--format", "auto_gptq")),
    (("--batch_size", "--train_bs", "--bs"), ("--batch_size", "1")),
    (("--gradient_accumulate_steps",), ("--gradient_accumulate_steps", "8")),
    (
        ("--low_gpu_mem_usage", "--disable_low_gpu_mem_usage", "--no-low_gpu_mem_usage"),
        ("--low_gpu_mem_usage",),
    ),
    (
        (
            "--enable_deterministic_algorithms",
            "--disable_deterministic_algorithms",
            "--no-enable_deterministic_algorithms",
        ),
        ("--enable_deterministic_algorithms",),
    ),
)


def _option_present(argv: Sequence[str], names: Sequence[str]) -> bool:
    for token in argv:
        if token in names:
            return True
        for name in names:
            if token.startswith(f"{name}="):
                return True
    return False


def _option_value(argv: Sequence[str], names: Sequence[str], default: str) -> str:
    for index, token in enumerate(argv):
        for name in names:
            if token == name and index + 1 < len(argv):
                return argv[index + 1]
            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix):]
    return default


def inject_profile_defaults(
    argv: Sequence[str],
    defaults=AMPERE_QUALITY_DEFAULTS,
) -> list[str]:
    """Prepend profile defaults that were not explicitly supplied by the user."""
    argv = list(argv)
    prefix: list[str] = []
    for aliases, tokens in defaults:
        if not _option_present(argv, aliases):
            prefix.extend(tokens)
    return prefix + argv


def ampere_quality_argv(argv: Sequence[str]) -> list[str]:
    """Return argv for the quality-first Ampere W8A16/G128 profile."""
    return inject_profile_defaults(argv)


def validate_ampere_runtime(argv: Sequence[str]) -> None:
    """Report whether the selected CUDA device is an Ampere GPU."""
    if any(flag in argv for flag in ("-h", "--help")):
        return

    device_map = _option_value(argv, ("--device_map", "--device", "--devices"), "0")
    first_device = str(device_map).split(",", maxsplit=1)[0].strip()
    if first_device in {"cpu", "xpu", "hpu"}:
        logger.warning("Ampere quality profile selected with non-CUDA device %s", first_device)
        return

    try:
        import torch
    except ImportError:
        logger.warning("Unable to validate Ampere hardware because PyTorch is unavailable.")
        return

    if not torch.cuda.is_available():
        logger.warning("Ampere quality profile selected but CUDA is not available.")
        return

    try:
        device_index = int(first_device.removeprefix("cuda:"))
    except ValueError:
        device_index = 0

    major, minor = torch.cuda.get_device_capability(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    if major != 8:
        logger.warning(
            "Ampere quality profile is tuned for compute capability 8.x; "
            "selected %s has capability %d.%d.",
            device_name,
            major,
            minor,
        )
        return

    logger.info(
        "Ampere quality profile: %s (sm_%d%d), "
        "W8A16/G128 symmetric + AutoRoundBest + GPTQ/Marlin export path.",
        device_name,
        major,
        minor,
    )
