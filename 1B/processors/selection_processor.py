# processors/selection_processor.py
import numpy as np
from typing import List, Tuple, Dict
from models.document_models import DocumentChunk, ProcessingResult
import json
from datetime import datetime
from models.document_models import DocumentChunk, ProcessingResult, PersonaProfile

class SelectionProcessor:
    """Handles diverse selection and output generation"""
    
    def __init__(self, config):
        self.config = config
    
    def select_diverse_sections(self, 
                              scored_chunks: List[Tuple[DocumentChunk, Dict[str, float]]],
                              embeddings: np.ndarray,
                              max_sections: int = None) -> List[Tuple[DocumentChunk, float]]:
        """Select diverse, high-quality sections using MMR"""
        if not scored_chunks:
            return []
        
        max_sections = max_sections or self.config.MAX_SECTIONS
        
        # Sort by final score
        scored_chunks.sort(key=lambda x: x[1]['final_score'], reverse=True)
        
        # MMR selection
        selected = []
        selected_indices = []
        selected_embeddings = []
        
        # Select first (highest scoring) chunk
        first_chunk, first_scores = scored_chunks[0]
        chunk_idx = self._find_chunk_index(first_chunk, scored_chunks)
        
        selected.append((first_chunk, first_scores['final_score']))
        selected_indices.append(chunk_idx)
        selected_embeddings.append(embeddings[chunk_idx])
        
        # MMR parameters
        lambda_param = 0.7  # Balance between relevance and diversity
        
        # Select remaining chunks
        for chunk, scores in scored_chunks[1:]:
            if len(selected) >= max_sections:
                break
            
            chunk_idx = self._find_chunk_index(chunk, scored_chunks)
            chunk_embedding = embeddings[chunk_idx]
            
            # Calculate similarity to already selected chunks
            similarities = [
                np.dot(chunk_embedding, sel_emb) 
                for sel_emb in selected_embeddings
            ]
            max_similarity = max(similarities) if similarities else 0
            
            # MMR score
            relevance_score = scores['final_score']
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
            
            # Diversity threshold
            if mmr_score > 0.3 and max_similarity < 0.85:
                selected.append((chunk, relevance_score))
                selected_indices.append(chunk_idx)
                selected_embeddings.append(chunk_embedding)
        
        return selected
    
    def _find_chunk_index(self, chunk: DocumentChunk, 
                         all_chunks: List[Tuple[DocumentChunk, Dict]]) -> int:
        """Find index of chunk in the original list"""
        for i, (c, _) in enumerate(all_chunks):
            if c.chunk_id == chunk.chunk_id:
                return i
        return -1
    
    def refine_text(self, chunk: DocumentChunk, 
                   persona_profile: 'PersonaProfile') -> str:
        """Refine and summarize chunk text for the specific persona"""
        import nltk
        sentences = nltk.sent_tokenize(chunk.text)
        
        if not sentences:
            return chunk.text
        
        # Score sentences
        sentence_scores = []
        for sent in sentences:
            score = self._score_sentence(sent, persona_profile)
            sentence_scores.append((sent, score))
        
        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        max_sentences = self.config.MAX_REFINED_SENTENCES
        selected_sentences = sentence_scores[:max_sentences]
        
        # Reorder to maintain flow
        selected_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        # Combine sentences
        refined_text = ' '.join([sent for sent, _ in selected_sentences])
        
        # Add ellipsis if truncated
        if len(selected_sentences) < len(sentences):
            refined_text += '...'
        
        return refined_text
    
    def _score_sentence(self, sentence: str, profile: 'PersonaProfile') -> float:
        """Score sentence relevance for persona"""
        score = 0.0
        sent_lower = sentence.lower()
        
        # Keyword matching
        for keyword in profile.get_all_keywords():
            if keyword in sent_lower:
                score += 2.0
        
        # Length preference
        word_count = len(sentence.split())
        if 10 <= word_count <= 30:
            score += 1.0
        elif word_count > 50:
            score -= 0.5
        
        # Information density (entities, numbers, etc.)
        import re
        
        # Numbers and statistics
        if re.search(r'\d+', sentence):
            score += 0.5
        
        # Quoted text
        if '"' in sentence or "'" in sentence:
            score += 0.5
        
        # Lists or enumerations
        if re.search(r'^\s*[\d\-\•]', sentence):
            score += 0.5
        
        return score
    
    def generate_output(self, 
                       selected_sections: List[Tuple[DocumentChunk, float]],
                       persona_profile: 'PersonaProfile',
                       input_data: Dict,
                       processing_time: float,
                       insights: List[str] = None) -> ProcessingResult:
        """Generate final output in required format"""
        
        # Create metadata
        metadata = {
            "input_documents": [doc['filename'] for doc in input_data['documents']],
            "persona": persona_profile.role,
            "job_to_be_done": persona_profile.task,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_time_seconds": round(processing_time, 2)
        }
        
        # Create extracted sections
        extracted_sections = []
        for rank, (chunk, score) in enumerate(selected_sections):
            extracted_sections.append({
                "document": chunk.document_name,
                "section_title": chunk.section_title,
                "importance_rank": rank + 1,
                "page_number": chunk.page_number,
                "relevance_score": float(score)  # Ensure Python float
            })
        
        # Create subsection analysis
        subsection_analysis = []
        for chunk, _ in selected_sections:
            refined_text = self.refine_text(chunk, persona_profile)
            subsection_analysis.append({
                "document": chunk.document_name,
                "refined_text": refined_text,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title
            })
        
        # Create result
        result = ProcessingResult(
            metadata=metadata,
            extracted_sections=extracted_sections,
            subsection_analysis=subsection_analysis,
            processing_time=processing_time,
            insights=insights or []
        )
        
        return result
    
    def save_output(self, result: ProcessingResult, output_path: str):
        """Save output to JSON file"""
        output_dict = {
            "metadata": result.metadata,
            "extracted_sections": result.extracted_sections,
            "subsection_analysis": result.subsection_analysis
        }
        
        # Add insights if available
        if result.insights:
            output_dict["insights"] = result.insights
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Output saved to {output_path}")
