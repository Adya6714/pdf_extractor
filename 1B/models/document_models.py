# models/document_models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import hashlib

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    document_name: str
    page_number: int
    text: str
    section_title: str
    chunk_id: str = field(default="")
    structural_level: int = 3  # 0=title, 1=h1, 2=h2, 3=text
    start_char: int = 0
    end_char: int = 0
    entities: Dict[str, List[str]] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.chunk_id:
            # Generate unique ID
            content = f"{self.document_name}_{self.page_number}_{self.start_char}"
            self.chunk_id = hashlib.md5(content.encode()).hexdigest()[:8]
    
    def get_text_preview(self, max_length: int = 100) -> str:
        """Get text preview for display"""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."

@dataclass
class PersonaProfile:
    """Represents a user persona with task context"""
    role: str
    task: str
    domain_keywords: List[str] = field(default_factory=list)
    task_keywords: List[str] = field(default_factory=list)
    intent_keywords: List[str] = field(default_factory=list)
    preferred_sections: List[str] = field(default_factory=list)
    
    def get_all_keywords(self) -> List[str]:
        """Get all keywords combined"""
        return list(set(self.domain_keywords + self.task_keywords + self.intent_keywords))

@dataclass
class ProcessingResult:
    """Represents the final processing result"""
    metadata: Dict
    extracted_sections: List[Dict]
    subsection_analysis: List[Dict]
    processing_time: float
    insights: List[str] = field(default_factory=list)
