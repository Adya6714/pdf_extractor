from pathlib import Path
import json
from app.extractor import PDFOutlineExtractor
import os

INPUT_DIR = Path(os.getenv("INPUT_DIR", "input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in /app/input")
        return

    extractor = PDFOutlineExtractor()

    for pdf in pdf_files:
        print(f"📄 Processing {pdf.name} …")
        result = extractor.process_pdf(pdf)
        with (OUTPUT_DIR / f"{pdf.stem}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ Done!")

if __name__ == "__main__":
    main()