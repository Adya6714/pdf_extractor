# processors/pdf_processor.py
import re
import pdfplumber
import PyPDF2
from typing import List, Tuple, Dict
from models.document_models import DocumentChunk
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
        chars = page.chars
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
    
    def _smart_chunk_page(self, text: str, layout: Dict, page_num: int, doc_name: str) -> List[DocumentChunk]:
        """Create chunks respecting document structure"""
        chunks = []
        lines = text.split('\n')
        
        current_section = "General"
        current_chunk = []
        current_level = 3
        chunk_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this is a header
            is_header, level, header_text = self._detect_header(line_stripped, layout)
            
            if is_header:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text.strip()) > self.config.MIN_CHUNK_SIZE:
                        chunks.append(DocumentChunk(
                            document_name=doc_name,
                            page_number=page_num,
                            text=chunk_text,
                            section_title=current_section,
                            structural_level=current_level,
                            start_char=chunk_start,
                            end_char=chunk_start + len(chunk_text)
                        ))
                
                # Start new section
                current_section = header_text
                current_level = level
                current_chunk = []
                chunk_start = sum(len(l) + 1 for l in lines[:i])
            
            elif line_stripped:  # Non-empty line
                current_chunk.append(line)
                
                # Check if chunk is getting too large
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > self.config.MAX_CHUNK_SIZE:
                    # Find good breaking point
                    break_point = self._find_break_point(chunk_text)
                    
                    # Save chunk up to break point
                    chunk_to_save = chunk_text[:break_point]
                    chunks.append(DocumentChunk(
                        document_name=doc_name,
                        page_number=page_num,
                        text=chunk_to_save,
                        section_title=current_section,
                        structural_level=current_level,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_to_save)
                    ))
                    
                    # Keep remainder for next chunk
                    remainder = chunk_text[break_point:].strip()
                    current_chunk = [remainder] if remainder else []
                    chunk_start += len(chunk_to_save)
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text.strip()) > self.config.MIN_CHUNK_SIZE:
                chunks.append(DocumentChunk(
                    document_name=doc_name,
                    page_number=page_num,
                    text=chunk_text,
                    section_title=current_section,
                    structural_level=current_level,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text)
                ))
        
        return chunks
    
    def _detect_header(self, line: str, layout: Dict) -> Tuple[bool, int, str]:
        """Detect if line is a header and return level"""
        # Check against layout info first
        if line in layout.get('headers', []):
            return True, 1, line
        
        # Check patterns
        for level, pattern in self.header_patterns:
            match = re.match(pattern, line)
            if match:
                header_text = match.group(1) if match.groups() else line
                return True, level, header_text
        
        return False, 3, ""
    
    def _find_break_point(self, text: str) -> int:
        """Find natural breaking point in text"""
        # Try to break at sentence boundary
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            # Find break point around middle
            mid_point = len(text) // 2
            best_break = mid_point
            min_distance = float('inf')
            
            current_pos = 0
            for sent in sentences[:-1]:
                current_pos += len(sent) + 1
                distance = abs(current_pos - mid_point)
                if distance < min_distance:
                    min_distance = distance
                    best_break = current_pos
            
            return best_break
        
        # Fallback to paragraph break
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            return len(paragraphs[0]) + 2
        
        # Last resort: break at space near middle
        mid_point = len(text) // 2
        space_pos = text.find(' ', mid_point)
        return space_pos if space_pos != -1 else mid_point
