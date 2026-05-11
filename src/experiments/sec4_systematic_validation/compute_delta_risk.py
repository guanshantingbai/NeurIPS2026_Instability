"""Decision consequence / delta risk placeholder."""

from __future__ import annotations

from src.core.selection_failure import delta_risk


def main() -> None:
    print(f"sec4: demo delta risk {delta_risk(0.21, 0.18):.4f}")


if __name__ == "__main__":
    main()
