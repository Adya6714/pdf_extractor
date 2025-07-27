# PDF Outline Extractor - Adobe India Hackathon 2025

## Overview

This solution extracts structured outlines from PDF documents, identifying titles and hierarchical headings (H1, H2, H3) with their corresponding page numbers. The implementation uses advanced heuristics combining font size analysis, text patterns, and formatting characteristics to accurately detect document structure.

## Approach

### 1. Text Extraction with Formatting
- Uses PyMuPDF to extract text blocks with complete formatting information
- Preserves font size, font name, bold/italic flags, and position data
- Maintains page number associations for all text blocks

### 2. Heading Detection Strategy
The solution employs multiple techniques to identify headings:

#### Pattern Recognition
- Numbered sections (1., 1.1, 1.1.1)
- Chapter/Section keywords
- Roman numerals and lettered lists

#### Font Analysis
- Calculates document-wide font statistics (mean, median, mode)
- Identifies text with font sizes significantly above average
- Detects bold formatting and uppercase text

#### Text Characteristics
- Length constraints (headings are typically shorter)
- Lack of ending punctuation (except colons)
- Position in document structure

### 3. Hierarchical Classification
- **H1**: Major sections, largest font sizes, numbered chapters
- **H2**: Subsections, medium font sizes, numbered subsections (x.x)
- **H3**: Sub-subsections, smaller headings, numbered (x.x.x)

### 4. Title Extraction
- Checks PDF metadata first
- Analyzes first 10 text blocks for largest, shortest text
- Applies heuristics to identify most likely title

## Technical Implementation

### Libraries Used
- **PyMuPDF (1.23.8)**: High-performance PDF parsing with low memory footprint
  - Chosen for speed and accuracy in text extraction
  - Provides detailed formatting information
  - Supports complex PDF structures

### Key Features
1. **Robust Pattern Matching**: Handles various numbering schemes and formats
2. **Statistical Analysis**: Adapts to document-specific font usage
3. **Multi-factor Scoring**: Combines multiple signals for accurate detection
4. **Error Handling**: Gracefully handles corrupted or unusual PDFs

## Performance Optimizations

1. **Efficient Memory Usage**
   - Processes pages sequentially
   - Releases resources immediately after use
   - Minimal data structures in memory

2. **Speed Optimizations**
   - Single-pass document analysis
   - Pre-compiled regex patterns
   - Optimized font statistics calculation

3. **Scalability**
   - Handles documents up to 50 pages within 10-second limit
   - Linear time complexity O(n) where n is number of text blocks

## Build and Run Instructions

### Building the Docker Image
```bash
docker build --platform linux/amd64 -t pdf-processor:latest .
```

### Running the Solution
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-processor:latest
```

### Testing Locally
1. Place PDF files in `./input` directory
2. Run the Docker container
3. Check `./output` directory for JSON files

## Output Format

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

## Multilingual Support

The solution includes basic support for:
- Unicode text extraction
- Non-Latin scripts (tested with Japanese)
- Mixed language documents
- UTF-8 encoding throughout

## Edge Cases Handled

1. **No Clear Headings**: Falls back to font-size based detection
2. **Complex Layouts**: Handles multi-column PDFs
3. **Missing Metadata**: Extracts title from document content
4. **Unusual Formatting**: Adapts to document-specific patterns
5. **Large Documents**: Optimized for 50-page processing

## Future Improvements

1. Machine learning-based heading detection
2. Support for more heading patterns (e.g., lettered sections)
3. Table of contents analysis for validation
4. Enhanced multilingual support with language-specific rules

## Compliance

- ✅ Execution time: < 10 seconds for 50-page PDFs
- ✅ No external dependencies or network calls
- ✅ Model size: < 200MB (no ML models used)
- ✅ CPU-only implementation (AMD64 compatible)
- ✅ Offline operation
- ✅ Open-source libraries only