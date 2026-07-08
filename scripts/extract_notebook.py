"""Extract code cells from SemanticExtractionLLms.ipynb for reference."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "reference" / "SemanticExtractionLLms.ipynb"
OUT = Path(__file__).resolve().parents[1] / "reference" / "extracted_cells.py"

def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    parts: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue
        parts.append(f"# --- cell {i} ---\n{src}\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {len(parts)} code cells to {OUT}")

if __name__ == "__main__":
    main()
