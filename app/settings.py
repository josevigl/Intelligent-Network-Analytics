"""Runtime settings shared by the API and persistence layer."""

import os


def public_demo_enabled() -> bool:
    value = os.environ.get("INA_PUBLIC_DEMO", "false").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no", ""}:
        return False
    raise RuntimeError("INA_PUBLIC_DEMO must be set to true or false.")
