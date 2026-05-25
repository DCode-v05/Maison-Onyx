"""Stage 8 — Decision engine.

Combines the three check verdicts plus registration reliability into a
single ACCEPT / REVIEW / REJECT outcome.
"""

from __future__ import annotations

from typing import List, Tuple

from .types import (
    DecorationCheckResult,
    ProfileCheckResult,
    RegistrationResult,
    SurfaceCheckResult,
)


def decide(
    registration: RegistrationResult,
    profile: ProfileCheckResult,
    decoration: DecorationCheckResult,
    surface: SurfaceCheckResult,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not registration.reliable:
        reasons.append(
            f"Registration unreliable (inliers={registration.num_inliers}, "
            f"ratio={registration.inlier_ratio:.2f}, NCC={registration.ncc:.2f})"
        )
        return "REVIEW", reasons

    verdicts = {
        "profile": profile.verdict,
        "decoration": decoration.verdict,
        "surface": surface.verdict,
    }

    fails = [k for k, v in verdicts.items() if v == "FAIL"]
    borders = [k for k, v in verdicts.items() if v == "BORDERLINE"]

    if fails:
        reasons.extend(f"{k} FAIL" for k in fails)
        return "REJECT", reasons
    if borders:
        reasons.extend(f"{k} BORDERLINE" for k in borders)
        return "REVIEW", reasons
    return "ACCEPT", ["all checks PASS"]
