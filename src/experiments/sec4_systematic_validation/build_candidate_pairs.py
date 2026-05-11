"""Build near-AUROC candidate pairs for section 4."""

from __future__ import annotations

from src.core.pairwise import build_pairs


def main() -> None:
    pairs = build_pairs(["cand_1", "cand_2", "cand_3"])
    print(f"sec4: built {len(pairs)} candidate pairs")


if __name__ == "__main__":
    main()
