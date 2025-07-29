PDF Outline Extractor - Project 1A

Overview

This project implements an intelligent PDF outline extraction system that automatically identifies and extracts structured headings (H1, H2, H3) and document titles from PDF files. It works offline and is optimized for accuracy and versatility across various document types.

Problem Statement
	•	Accept PDF files (≤50 pages)
	•	Extract document title
	•	Identify H1–H3 headings with proper levels and page numbers
	•	Output a structured JSON
	•	Handle forms, reports, academic docs
	•	Work without network dependency

Directory Structure

pdf_extractor/
└── 1A/
    ├── app/
    │   ├── __init__.py
    │   ├── extractor.py           # Core logic
    │   └── process_pdfs.py        # Batch runner
    ├── input/                    # PDF input files
    ├── output/                   # Extracted JSON
    ├── Dockerfile
    └── README.md

Architecture Diagram

┌────────────┐    ┌─────────────────┐    ┌────────────────┐
│ Input PDFs │──▶│ PDF Extractor   │──▶│ JSON Outlines  │
└────────────┘    │ extractor.py    │    │ output/*.json  │
                 └─────────────────┘    └────────────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ Processing Pipeline│
                   ├────────────────────┤
                   │ - Text & Font Info │
                   │ - Header Filter    │
                   │ - Merge Headings   │
                   │ - Title Extract    │
                   └────────────────────┘

Pipeline Workflow

1. Text & Formatting Extraction
	•	Uses PyMuPDF to read block-level text and font metadata
	•	Collects font size, bold status, coordinates

2. Recurring Header/Footer Detection
	•	Filters out repeated headers using first-line frequency analysis across pages

3. Table Content Filtering
	•	Skips repeated, dense font blocks indicative of table content

4. Font Analysis
	•	Determines potential heading levels based on font size rarity

5. Heading Classification

Scoring:
	•	+5: Font size rarity
	•	+4: Colon-style titles
	•	+3: Numbered headings (e.g. 1.1. Title)
	•	+2: Bold formatting
	•	+2: All caps short strings

Threshold ≥4 qualifies as a heading

6. Heading Merging
	•	Merges visually adjacent heading fragments with matching format

7. Title Extraction
	•	First-page font prominence analysis
	•	Fallback to metadata
	•	Fragment merging with duplicate-checking

8. Output
	•	JSON with keys: title, outline (list of headings with level, text, page)

Special Document Handling

Forms
	•	If >95% font uniformity: classified as form → returns only title, no outline

Reports/Articles
	•	If font distribution is hierarchical and title is prominent → normal extraction

Code Highlights

extractor.py
	•	PDFOutlineExtractor class with:
	•	process_pdf() - main function
	•	extract_title_with_merging()
	•	merge_consecutive_headings()
	•	is_potential_heading() with multi-criteria analysis
	•	is_form_field_number() for form exclusion

process_pdfs.py

from pathlib import Path
import json
from extractor import PDFOutlineExtractor

INPUT_DIR = Path("../input")
OUTPUT_DIR = Path("../output")

extractor = PDFOutlineExtractor()

for pdf in INPUT_DIR.glob("*.pdf"):
    result = extractor.process_pdf(pdf)
    with open(OUTPUT_DIR / f"{pdf.stem}.json", "w") as f:
        json.dump(result, f, indent=2)

Local Execution

cd pdf_extractor/1A
pip install PyMuPDF pathlib
mkdir -p input output
cp my.pdf input/
cd app
python process_pdfs.py

Docker Usage

Build

docker build --platform linux/amd64 -t pdf-extractor:latest .

Run

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-extractor:latest

Configurable Parameters
	•	heading_percentage_threshold = 5
	•	max_heading_length = 200
	•	min_heading_length = 3
	•	form_detection_threshold = 95

Limitations
	•	No OCR: does not process scanned image PDFs
	•	Max 50 pages recommended for performance
	•	Optimized for English documents

Output Format

{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "Background", "page": 2 },
    ...
  ]
}

Troubleshooting
	•	Missing headings → Ensure formatting is distinct
	•	False positives → Tune scoring logic or debug via print statements
	•	Title mismatch → Check merging and duplication checks

License

[Add license here]

Contributors

Maintained by internal team. Contributions welcome with PR and test PDFs.
