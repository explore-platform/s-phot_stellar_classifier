"""Build a small PySSED .dat file for inference demos (same header/schema as training)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data" / "df_plus_ms.dat"
SOURCE_LABELS = REPO / "data" / "classes_plus_ms.dat"
OUT_DATA = REPO / "data" / "mock_predict.dat"
OUT_LABELS = REPO / "data" / "mock_predict_labels.txt"

# Real stars copied from training; renamed for mock inference. Labels kept for optional evaluation.
MOCK_STARS = [
    ("CD-60__1960", "MOCK_STAR_001"),
    ("CD-60__1971", "MOCK_STAR_002"),
    ("CD-60__2373", "MOCK_STAR_003"),
]


def load_label_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Training table not found: {SOURCE}")

    label_map = load_label_map(SOURCE_LABELS) if SOURCE_LABELS.exists() else {}
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    header = lines[:9]
    by_id = {line.split("\t", 1)[0]: line for line in lines[9:] if line.strip()}

    data_rows = []
    label_rows = []
    for src_id, mock_id in MOCK_STARS:
        if src_id not in by_id:
            raise KeyError(f"Star {src_id} not found in {SOURCE}")
        row = by_id[src_id].split("\t")
        row[0] = mock_id
        data_rows.append("\t".join(row))
        if src_id in label_map:
            label_rows.append(f"{mock_id}\t{label_map[src_id]}")

    OUT_DATA.write_text("\n".join(header + data_rows) + "\n", encoding="utf-8")
    OUT_LABELS.write_text("\n".join(label_rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(data_rows)} mock stars to {OUT_DATA}")
    print(f"Wrote reference labels to {OUT_LABELS} (optional, for evaluation only)")


if __name__ == "__main__":
    main()
