"""Check the fused AdamW and Lightning clipping compatibility policy."""

from __future__ import annotations

from joint_event_reconstruction.training.joint import resolve_fused_adamw


def main() -> None:
    cases = (
        (
            "cpu",
            {
                "requested": True,
                "device_type": "cpu",
                "precision": "bf16-mixed",
                "gradient_clip": 1.0,
            },
            (False, "fused AdamW is only enabled for CUDA parameters"),
        ),
        (
            "cuda_bf16_with_clipping",
            {
                "requested": True,
                "device_type": "cuda",
                "precision": "bf16-mixed",
                "gradient_clip": 1.0,
            },
            (
                False,
                "Lightning mixed-precision gradient clipping incompatible with fused AdamW",
            ),
        ),
        (
            "cuda_without_clipping",
            {
                "requested": True,
                "device_type": "cuda",
                "precision": "bf16-mixed",
                "gradient_clip": 0.0,
            },
            (True, None),
        ),
    )
    for name, arguments, expected in cases:
        observed = resolve_fused_adamw(**arguments)
        if observed != expected:
            raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")
        print(f"{name}: PASS effective_fused={observed[0]} reason={observed[1]}")


if __name__ == "__main__":
    main()
