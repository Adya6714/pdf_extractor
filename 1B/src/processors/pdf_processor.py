# src/processors/pdf_processor.py
import re
import pdfplumber
import PyPDF2
from typing import List, Tuple, Dict
from src.models.document_models import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Advanced PDF processing with structure detection"""
    def __init__(self, config):
        self.config = config
        self.header_patterns = [
            (0, r'^#{1,3}\s+(.+)$'),  # Markdown headers
            (1, r'^(\d+\.?\s+[A-Z].+)$'),  # Numbered sections
            (1, r'^([A-Z][A-Z\s]+):?\s*$'),  # All caps headers
            (2, r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):$'),  # Title case with colon
            (2, r'^\s*\*\*(.+)\*\*\s*$'),  # Bold text
        ]
    
    def extract_pdf_with_structure(self, pdf_path: str) -> List[Dict]:
        """Extract PDF with layout and structure information"""
        structured_pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with positioning
                    text = page.extract_text()
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    
                    # Get text with layout info
                    layout_info = self._analyze_layout(page)
                    
                    structured_pages.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'tables': tables,
                        'layout': layout_info
                    })
                    
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            # Fallback to PyPDF2
            structured_pages = self._fallback_extraction(pdf_path)
        
        return structured_pages
    
    def _analyze_layout(self, page) -> Dict:
        """Analyze page layout for structure detection"""
        layout = {
            'headers': [],
            'paragraphs': [],
            'lists': [],
            'emphasis': []
        }
        
        # Extract characters with formatting
        chars = page.chars if hasattr(page, 'chars') else []
        current_line = []
        current_y = None
        
        for char in chars:
            if current_y is None:
                current_y = char['top']
            
            # New line detection
            if abs(char['top'] - current_y) > 2:
                if current_line:
                    line_text = ''.join([c['text'] for c in current_line])
                    self._classify_line(line_text, current_line[0], layout)
                current_line = [char]
                current_y = char['top']
            else:
                current_line.append(char)
        
        return layout
    
    def _classify_line(self, text: str, first_char: Dict, layout: Dict):
        """Classify line based on formatting"""
        # Check if header (larger font or bold)
        if first_char.get('fontname', '').lower().endswith('bold') or first_char.get('size', 0) > 12:
            layout['headers'].append(text)
        # Check if list item
        elif re.match(r'^\s*[\•\-\*]\s+', text):
            layout['lists'].append(text)
        # Regular paragraph
        else:
            layout['paragraphs'].append(text)
    
    def _fallback_extraction(self, pdf_path: str) -> List[Dict]:
        """Fallback extraction using PyPDF2"""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'tables': [],
                        'layout': {'headers': [], 'paragraphs': [], 'lists': []}
                    })
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
        return pages
    
    def create_semantic_chunks(self, pages: List[Dict], doc_name: str) -> List[DocumentChunk]:
        """Create semantically coherent chunks from pages"""
        all_chunks = []
        
        for page_data in pages:
            page_num = page_data['page_number']
            text = page_data['text']
            layout = page_data.get('layout', {})
            
            # Smart chunking based on structure
            chunks = self._smart_chunk_page(text, layout, page_num, doc_name)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _detect_header(self, line: str, layout: Dict) -> Tuple[bool, int, str]:
        """Detect if line is a header and return level"""
        
        # Skip very short lines or numbers
        if len(line.strip()) < 3 or line.strip().replace('.', '').isdigit():
            return False, 3, ""
        
        # Skip lines that are just numbered steps
        if re.match(r'^\d+\.\s+', line) and len(line) < 100:
            # This might be a step, not a header
            return False, 3, ""
        
        # Priority patterns for different document types
        header_patterns = [
            # Document type headers (highest priority)
            (0, r'^([A-Z][A-Za-z\s\-&]+)$'),
            (0, r'^([A-Z][A-Za-z\s\-&]+):?\s*$'),
            
            # Food/Recipe headers
            (1, r'^([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s*$'),
            (1, r'^([A-Za-z\s]+):\s*$'),
            
            # Section headers
            (1, r'^(?:Chapter|Section|Part)\s+\d+[:\s]+(.+)$'),
            (1, r'^(\d+\.\d+\s+.+)$'),
            
            # Functional headers
            (2, r'^(Introduction|Overview|Summary|Conclusion|Instructions|Ingredients|Tips?).*$'),
            (2, r'^(Must-[A-Za-z\s]+|How to[A-Za-z\s]+|Guide to[A-Za-z\s]+).*$'),
        ]
        
        # Check against layout info first
        if line in layout.get('headers', []):
            return True, 1, line
        
        # Check font-based detection
        for char_info in layout.get('chars', []):
            if char_info.get('text', '') in line:
                if 'bold' in char_info.get('fontname', '').lower() or char_info.get('size', 0) > 12:
                    return True, 1, line
        
        # Check patterns
        for level, pattern in header_patterns:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                header_text = match.group(1) if match.groups() else line.strip()
                return True, level, header_text
        
        return False, 3, ""
    
    def _smart_chunk_page(self, text: str, layout: Dict, page_num: int, doc_name: str) -> List[DocumentChunk]:
        """Create chunks respecting document structure"""
        chunks = []
        lines = text.split('\n') if text else []
        
        # Determine page-level header
        page_header = None
        for line in lines[:10]:
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) > 3:
                is_h, lvl, hdr = self._detect_header(line_stripped, layout)
                if is_h and lvl <= 1:
                    page_header = hdr
                    break
        if not page_header:
            lower_name = doc_name.lower()
            if any(tag in lower_name for tag in ("recipe", "dinner")):
                for line in lines[:20]:
                    clean = line.strip()
                    if clean and not any(skip in clean.lower() for skip in ("ingredients","instructions","serves")):
                        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*$', clean):
                            page_header = clean
                            break
        
        current_section = page_header or "Content"
        current_chunk = []
        current_level = 3
        chunk_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            is_h, lvl, hdr = self._detect_header(line_stripped, layout)
            if is_h and hdr and hdr != current_section:
                if current_chunk:
                    text_chunk = '\n'.join(current_chunk)
                    if len(text_chunk.strip()) > self.config.MIN_CHUNK_SIZE:
                        chunks.append(DocumentChunk(
                            document_name=doc_name,
                            page_number=page_num,
                            text=text_chunk,
                            section_title=current_section,
                            structural_level=current_level,
                            start_char=chunk_start,
                            end_char=chunk_start + len(text_chunk)
                        ))
                current_section = hdr
                current_level = lvl
                current_chunk = [line]
                chunk_start = sum(len(l) + 1 for l in lines[:i])
            elif line_stripped:
                current_chunk.append(line)
                text_chunk = '\n'.join(current_chunk)
                if len(text_chunk) > self.config.MAX_CHUNK_SIZE:
                    chunks.append(DocumentChunk(
                        document_name=doc_name,
                        page_number=page_num,
                        text=text_chunk,
                        section_title=current_section,
                        structural_level=current_level,
                        start_char=chunk_start,
                        end_char=chunk_start + len(text_chunk)
                    ))
                    current_chunk = []
                    chunk_start += len(text_chunk)
        
        if current_chunk:
            text_chunk = '\n'.join(current_chunk)
            if len(text_chunk.strip()) > self.config.MIN_CHUNK_SIZE:
                chunks.append(DocumentChunk(
                    document_name=doc_name,
                    page_number=page_num,
                    text=text_chunk,
                    section_title=current_section,
                    structural_level=current_level,
                    start_char=chunk_start,
                    end_char=chunk_start + len(text_chunk)
                ))
        
        return chunks