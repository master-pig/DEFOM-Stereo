import argparse
from pathlib import Path


def collect_scene_names(middeval3_root: Path, preferred_split: str):
    split_dirs = [f"training{preferred_split}", "trainingF", "trainingH", "trainingQ"]
    names = set()

    for split_dir in split_dirs:
        path = middeval3_root / split_dir
        if not path.is_dir():
            continue
        for child in path.iterdir():
            if child.is_dir():
                names.add(child.name)

    return sorted(names)

def main():
    parser = argparse.ArgumentParser(
        description="Generate MiddEval3 official_train.txt from existing training scene folders."
    )
    parser.add_argument(
        "--middeval3_root",
        type=Path,
        default=Path("datasets/Middlebury/MiddEval3"),
        help="Path to the MiddEval3 directory containing trainingF/trainingH/trainingQ.",
    )
    parser.add_argument(
        "--preferred_split",
        type=str,
        default="H",
        choices=["F", "H", "Q"],
        help="Preferred split to scan first.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output txt path. Defaults to <middeval3_root>/official_train.txt.",
    )
    args = parser.parse_args()

    root = args.middeval3_root
    if not root.is_dir():
        raise FileNotFoundError(f"MiddEval3 root not found: {root}")

    names = collect_scene_names(root, args.preferred_split)
    if not names:
        raise RuntimeError(
            f"No scene folders found under {root}/training{args.preferred_split} "
            "or trainingF/trainingH/trainingQ."
        )

    output = args.output or (root / "official_train.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(names) + "\n", encoding="utf-8")

    print(f"Wrote {len(names)} scene names to: {output}")
    for name in names:
        print(name)


if __name__ == "__main__":
    main()
