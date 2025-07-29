# PDF Processor 1B

## Overview
This project processes collections of PDF documents, extracting and analyzing relevant sections based on a specified persona and job-to-be-done. It is fully containerized for offline, CPU-only execution.

## Quick Start (Docker)

### 1. Build the Docker Image
```bash
docker build -t pdf-processor-1b .
```
**OR Download Prebuilt Docker Image**
   - [Download pdf-processor-1b.tar.gz](https://drive.google.com/file/d/14kM-yPaMNVjwIS-4-waujZV_-87zMjyU/view)
   - Load the image:
     ```bash
     gunzip -c pdf-processor-1b.tar.gz | docker load
     ```

### 2. Run the Pipeline (Batch Mode)
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs pdf-processor-1b
```
- This will process all collections in `/app/Collections` and write outputs to the `outputs/` directory.

### 3. Run for a Single Collection
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs pdf-processor-1b --mode single --collection "Collection 1"
```

### 4. Output
- Results are saved as `output.json` and `report.pdf` in `outputs/<Collection Name>/`.
- See `outputs/Collection 1/output.json` for a sample output.

### 5. Input Format
- See `Collections/Collection 1/input.json` for a sample input file.

## Constraints
- CPU-only, no GPU required
- Model size < 1GB
- No internet access required at runtime
- Processing time: <60s for 3-5 documents

## Methodology
See [approach_explanation.md](./approach_explanation.md) for a detailed explanation of the pipeline and methodology.

## Sample Input/Output
- **Sample Input:** `Collections/Collection 1/input.json`
- **Sample Output:** `outputs/Collection 1/output.json`

---

For any issues, please refer to the code or contact the maintainer.
