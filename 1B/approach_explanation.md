# Approach Explanation

## Methodology

This project implements a robust, modular PDF processing pipeline designed to extract, analyze, and summarize information from collections of PDF documents according to a specified persona and job-to-be-done. The pipeline is tailored for offline, CPU-only environments and is fully containerized for reproducibility and ease of deployment.

### Input Structure
- **Input documents**: Provided as a list of PDF filenames in a collection directory, referenced in a JSON input file.
- **Persona**: The role (e.g., "Travel Planner") is specified in the input and used to guide the analysis and extraction process.
- **Job to be done**: A task description (e.g., "Plan a trip of 4 days for a group of 10 college friends") is also provided in the input JSON.

### Output Structure
The pipeline produces a structured JSON output with three main sections:
1. **Metadata**: Includes the list of input documents, persona, job to be done, and a processing timestamp.
2. **Extracted Sections**: For each selected section, the output includes the document name, page number, section title, and an importance rank.
3. **Sub-section Analysis**: For each selected section, a refined summary or analysis is provided, along with the document name and page number.

### Processing Pipeline
- **PDF Parsing**: Each PDF is parsed using `pdfplumber` and `PyPDF2` to extract text, structure, and tables.
- **Chunking & Sectioning**: Documents are split into semantic chunks and sections using heuristics and layout analysis.
- **Relevance Scoring**: Each chunk is scored for relevance to the persona and job using embeddings and keyword matching.
- **Diverse Selection**: The top-ranked, diverse sections are selected to maximize coverage and minimize redundancy.
- **Sub-section Analysis**: Each selected section is further summarized or refined for clarity and focus.

### Persona & Task Handling
A persona analyzer module tailors the extraction and scoring process to the specified role and task, ensuring that the output is relevant to the user's needs.

## Constraints & Compliance
- **CPU-only**: The Docker image and all dependencies are CPU-only; no GPU libraries are installed or required.
- **Model Size ≤ 1GB**: The embedded model file is under 1GB, ensuring compliance with resource constraints.
- **Processing Time ≤ 60s**: The pipeline is optimized for speed, using efficient batch processing and limiting the number of extracted sections to ensure that a collection of 3-5 documents is processed in under 60 seconds on typical CPUs.
- **Offline Execution**: All models and dependencies are baked into the Docker image. No internet access is required or used at runtime.

## Reproducibility & Testing
- The pipeline is fully containerized via Docker. All code, models, and sample data are included in the image for offline, reproducible execution.
- Sample input and output files are provided for testing and validation.

---

This approach ensures robust, efficient, and reproducible PDF analysis tailored to specific user personas and tasks, while strictly adhering to all technical and resource constraints. 