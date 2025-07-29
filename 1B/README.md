# PDF Processor 1B

## 1. Project Overview
This project processes collections of PDF documents, extracting and analyzing relevant sections based on a specified persona and job-to-be-done. It is designed for offline, CPU-only execution and is fully containerized for reproducibility.

---

## 2. Folder Structure

| Folder/File                | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `src/`                     | Main source code (pipeline, processors, models, utils)                  |
| `Collections/`             | Input data collections (each with PDFs and input JSON)                  |
| `models/`                  | Pretrained model files (≤1GB, e.g., TinyLlama)                          |
| `outputs/`                 | Output results (JSON, PDF) for each collection                          |
| `logs/`                    | Log files generated during processing                                   |
| `requirements.txt`         | Python dependencies                                                     |
| `Dockerfile`               | Docker build instructions                                               |
| `.dockerignore`            | Files/folders to exclude from Docker build context                      |
| `approach_explanation.md`  | Detailed methodology and constraint compliance                          |
| `README.md`                | This documentation                                                      |

---

## 3. Technical Architecture

**High-Level Pipeline:**
1. **Input Loader**: Reads collection input JSON and PDF files.
2. **Persona Analyzer**: Adapts extraction to the specified persona and task.
3. **PDF Processor**: Extracts structured text and sections from PDFs.
4. **Embedding & Relevance Scorer**: Scores document sections for relevance.
5. **Selection Processor**: Selects top sections and performs sub-section analysis.
6. **Output Generator**: Produces structured JSON and PDF reports.

---

## 4. Input & Output Format

### **Input**
- JSON file (see `Collections/Collection 1/input.json`):
  ```json
  {
    "challenge_info": { ... },
    "documents": [
      {"filename": "doc1.pdf", "title": "Doc 1"},
      ...
    ],
    "persona": {"role": "Travel Planner"},
    "job_to_be_done": {"task": "Plan a trip..."}
  }
  ```

### **Output**
- JSON file (see `outputs/Collection 1/output.json`):
  ```json
  {
    "metadata": { ... },
    "extracted_sections": [
      {"document": "...", "section_title": "...", "importance_rank": 1, "page_number": 1},
      ...
    ],
    "subsection_analysis": [
      {"document": "...", "refined_text": "...", "page_number": 1},
      ...
    ]
  }
  ```
- PDF report (optional, for human consumption).

---

## 5. How the Pipeline Works

1. **Load Input**: Reads the collection’s input JSON and finds referenced PDFs.
2. **Persona & Task Analysis**: Adapts extraction to the persona and job.
3. **PDF Parsing**: Extracts text, structure, and tables from each PDF.
4. **Chunking & Scoring**: Splits documents into sections, scores for relevance.
5. **Selection**: Picks top sections, ensuring diversity and coverage.
6. **Sub-section Analysis**: Summarizes or refines selected sections.
7. **Output Generation**: Writes results to JSON and PDF.

---

## 6. How to Build and Run

### **A. Using Docker (Recommended)**
1. **Build the image:**
   ```bash
   docker build -t pdf-processor-1b .
   ```
2. **Run the pipeline (batch mode):**
   ```bash
   docker run --rm -v $(pwd)/outputs:/app/outputs pdf-processor-1b
   ```
3. **Run for a single collection:**
   ```bash
   docker run --rm -v $(pwd)/outputs:/app/outputs pdf-processor-1b --mode single --collection "Collection 1"
   ```

### **B. Run Locally (No Docker)**
1. **Install Python 3.9+ and pip.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline:**
   ```bash
   python main.py --mode batch
   ```
   or for a single collection:
   ```bash
   python main.py --mode single --collection "Collection 1"
   ```

---

## 7. Sample Usage

- **Sample Input:** `Collections/Collection 1/input.json`
- **Sample Output:** `outputs/Collection 1/output.json`
- **Methodology:** See `approach_explanation.md`

---

## 8. References

- [approach_explanation.md](./approach_explanation.md) — Detailed methodology and compliance
- [Sample Input](./Collections/Collection%201/input.json)
- [Sample Output](./outputs/Collection%201/output.json)

---

For any issues, please refer to the code or contact the maintainer.
