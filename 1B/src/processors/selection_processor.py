# src/processors/selection_processor.py
"""
Handles diverse selection and output generation - IMPROVED VERSION
"""

import numpy as np
from typing import List, Tuple, Dict
from src.models.document_models import DocumentChunk, ProcessingResult, PersonaProfile
import json
from datetime import datetime, timezone
import re
import nltk
import logging

logger = logging.getLogger(__name__)

class SelectionProcessor:
    """Handles diverse selection and output generation"""
    
    def __init__(self, config):
        self.config = config
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    def select_diverse_sections(self, 
                              scored_chunks: List[Tuple[DocumentChunk, Dict[str, float]]],
                              embeddings: np.ndarray,
                              max_sections: int = None) -> List[Tuple[DocumentChunk, float]]:
        """Select diverse, high-quality sections using improved logic"""
        if not scored_chunks:
            return []
        
        max_sections = max_sections or self.config.MAX_SECTIONS
        
        # Sort by final score
        scored_chunks.sort(key=lambda x: x[1]['final_score'], reverse=True)
        
        # Log top scoring chunks for debugging
        logger.info(f"Top 10 chunks by score:")
        for i, (chunk, scores) in enumerate(scored_chunks[:10]):
            logger.info(f"  {i+1}. {chunk.document_name} - {chunk.section_title} "
                       f"(score: {scores['final_score']:.3f}, page: {chunk.page_number})")
        
        # Improved selection strategy
        selected = []
        selected_docs = set()
        selected_sections = set()
        selected_indices = []
        
        # First pass: Get best unique section from each document
        for chunk, scores in scored_chunks:
            if len(selected) >= max_sections:
                break
            
            # Skip if we already have this document
            if chunk.document_name in selected_docs and len(selected_docs) >= 3:
                continue
            
            # Skip if we already have this exact section title
            section_key = f"{chunk.document_name}:{chunk.section_title}"
            if section_key in selected_sections:
                continue
            
            # Skip generic section titles in first pass
            if chunk.section_title.lower() in ['general', 'content', 'ingredients', 'instructions']:
                continue
            
            # Add to selected
            selected.append((chunk, scores['final_score']))
            selected_docs.add(chunk.document_name)
            selected_sections.add(section_key)
            chunk_idx = self._find_chunk_index(chunk, scored_chunks)
            selected_indices.append(chunk_idx)
        
        # Second pass: Fill remaining slots with best available chunks
        if len(selected) < max_sections:
            for chunk, scores in scored_chunks:
                if len(selected) >= max_sections:
                    break
                
                # Skip already selected chunks
                chunk_idx = self._find_chunk_index(chunk, scored_chunks)
                if chunk_idx in selected_indices:
                    continue
                
                # Skip if this exact section is already selected
                section_key = f"{chunk.document_name}:{chunk.section_title}"
                if section_key in selected_sections:
                    continue
                
                # Limit chunks per document (max 2)
                doc_count = sum(1 for s, _ in selected if s.document_name == chunk.document_name)
                if doc_count >= 2:
                    continue
                
                # Check diversity using embeddings
                if selected_indices and embeddings is not None:
                    chunk_embedding = embeddings[chunk_idx]
                    
                    # Calculate max similarity to already selected chunks
                    max_similarity = 0
                    for sel_idx in selected_indices:
                        similarity = np.dot(chunk_embedding, embeddings[sel_idx])
                        max_similarity = max(max_similarity, similarity)
                    
                    # Skip if too similar (threshold: 0.85)
                    if max_similarity > 0.85:
                        continue
                
                # Add to selected
                selected.append((chunk, scores['final_score']))
                selected_docs.add(chunk.document_name)
                selected_sections.add(section_key)
                selected_indices.append(chunk_idx)
        
        logger.info(f"Selected {len(selected)} sections from {len(selected_docs)} documents")
        
        return selected
    
    def _find_chunk_index(self, chunk: DocumentChunk, 
                         all_chunks: List[Tuple[DocumentChunk, Dict]]) -> int:
        """Find index of chunk in the original list"""
        for i, (c, _) in enumerate(all_chunks):
            if c.chunk_id == chunk.chunk_id:
                return i
        return -1
    
    def refine_text(self, chunk: DocumentChunk, 
                   persona_profile: PersonaProfile) -> str:
        """Refine and extract meaningful content for the specific persona"""
        
        text = chunk.text.strip()
        
        # For HR professional, extract procedural content
        if "hr" in persona_profile.role.lower():
            # Look for instructional content
            lines = text.split('\n')
            refined_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and very short lines
                if len(line) < 10:
                    continue
                
                # Skip numbered steps that are too generic
                if re.match(r'^\d+\.\s+\w+$', line):
                    continue
                
                # Include lines with instructions, steps, or important information
                if any(keyword in line.lower() for keyword in 
                       ['to', 'select', 'choose', 'click', 'enable', 'create', 'fill', 'form', 
                        'field', 'signature', 'document', 'tool', 'option', 'prepare', 'acrobat',
                        'pdf', 'interactive', 'can', 'use', 'see']):
                    refined_lines.append(line)
            
            if refined_lines:
                refined_text = ' '.join(refined_lines)
                # Clean up excessive whitespace
                refined_text = re.sub(r'\s+', ' ', refined_text)
                return refined_text
            else:
                # Fallback: return cleaned original text
                text = re.sub(r'\s+', ' ', text)
                return text[:800] if len(text) > 800 else text
        
        # For food contractor, extract complete recipes
        elif "food" in persona_profile.role.lower() or "contractor" in persona_profile.role.lower():
            # Clean up the text first
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Look for recipe structure
            has_ingredients = "ingredient" in text.lower()
            has_instructions = "instruction" in text.lower()
            
            if has_ingredients or has_instructions:
                # This looks like recipe content, preserve more of it
                # Extract sentences that contain useful information
                try:
                    sentences = nltk.sent_tokenize(text)
                except:
                    sentences = text.split('. ')
                
                refined_sentences = []
                for sent in sentences:
                    sent = sent.strip()
                    # Include sentences with recipe-related content
                    if any(keyword in sent.lower() for keyword in 
                           ['ingredient', 'cup', 'tablespoon', 'teaspoon', 'pound', 'ounce',
                            'mix', 'blend', 'cook', 'bake', 'serve', 'preheat', 'combine',
                            'stir', 'add', 'heat', 'cool', 'refrigerate', 'garnish']):
                        refined_sentences.append(sent)
                
                if refined_sentences:
                    return ' '.join(refined_sentences)
            
            # For non-recipe content, return cleaned text
            return text[:1000] if len(text) > 1000 else text
        
        # For travel planner, extract descriptive content
        elif "travel" in persona_profile.role.lower() or "planner" in persona_profile.role.lower():
            # Clean up formatting
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Try to preserve list-like structures for travel content
            if any(marker in text for marker in ['•', '-', ':', ';']):
                # This might be a list of places or activities
                # Preserve more structure
                lines = chunk.text.strip().split('\n')
                cleaned_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:
                        # Clean up the line but preserve some structure
                        line = re.sub(r'\s+', ' ', line)
                        cleaned_lines.append(line)
                
                if cleaned_lines:
                    # Join with semicolons for better readability
                    return '; '.join(cleaned_lines)
            
            # Default: return substantial content
            return text[:1000] if len(text) > 1000 else text
        
        # Default: clean up and return meaningful content
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Try sentence tokenization for better content
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 3:
                # Take first several complete sentences
                num_sentences = min(len(sentences), 10)
                return ' '.join(sentences[:num_sentences])
        except:
            pass
        
        # Fallback: return first 800 characters or full text if shorter
        return text[:800] + '...' if len(text) > 800 else text
    
    def _score_sentence(self, sentence: str, profile: PersonaProfile) -> float:
        """Score sentence relevance for persona"""
        score = 0.0
        sent_lower = sentence.lower()
        
        # Keyword matching
        for keyword in profile.get_all_keywords():
            if keyword.lower() in sent_lower:
                score += 2.0
        
        # Length preference
        word_count = len(sentence.split())
        if 10 <= word_count <= 40:
            score += 1.0
        elif word_count > 60:
            score -= 0.5
        
        # Information density
        # Numbers and measurements (good for recipes, travel info)
        if re.search(r'\b\d+\b', sentence):
            score += 0.5
        
        # Specific terms based on persona
        if "food" in profile.role.lower():
            food_terms = ['cup', 'tablespoon', 'teaspoon', 'ounce', 'pound', 'degree',
                         'minute', 'hour', 'serve', 'portion']
            for term in food_terms:
                if term in sent_lower:
                    score += 0.5
        
        elif "travel" in profile.role.lower():
            travel_terms = ['visit', 'explore', 'beach', 'city', 'restaurant', 'hotel',
                           'activity', 'tour', 'experience', 'enjoy']
            for term in travel_terms:
                if term in sent_lower:
                    score += 0.5
        
        elif "hr" in profile.role.lower():
            hr_terms = ['form', 'field', 'create', 'fill', 'sign', 'document', 'pdf',
                       'employee', 'compliance', 'process']
            for term in hr_terms:
                if term in sent_lower:
                    score += 0.5
        
        return score
    
    def generate_output(self, 
                       selected_sections: List[Tuple[DocumentChunk, float]],
                       persona_profile: PersonaProfile,
                       input_data: Dict,
                       processing_time: float,
                       insights: List[str] = None) -> ProcessingResult:
        """Generate final output in required format"""
        
        # Create metadata
        metadata = {
            "collection_name": input_data.get("collection_name", "unknown"),
            "input_documents": [doc['filename'] for doc in input_data['documents']],
            "persona": persona_profile.role,
            "job_to_be_done": persona_profile.task,
            "processing_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "processing_time_seconds": round(processing_time, 2)
        }
        
        # Create extracted sections
        extracted_sections = []
        for rank, (chunk, score) in enumerate(selected_sections):
            # Clean up section title
            section_title = chunk.section_title
            
            # Avoid generic titles
            if section_title.lower() in ['general', 'content', 'text']:
                # Try to extract a better title from the chunk content
                lines = chunk.text.strip().split('\n')
                for line in lines[:5]:  # Check first 5 lines
                    line = line.strip()
                    if line and len(line) > 5 and len(line) < 100:
                        # This could be a title
                        if not re.match(r'^\d+\.', line) and not line.lower().startswith(('the', 'a', 'an')):
                            section_title = line
                            break
            
            extracted_sections.append({
                "document": chunk.document_name,
                "section_title": section_title,
                "importance_rank": rank + 1,
                "page_number": chunk.page_number,
                "relevance_score": float(score)
            })
        
        # Create subsection analysis
        subsection_analysis = []
        for chunk, _ in selected_sections:
            refined_text = self.refine_text(chunk, persona_profile)
            
            # Skip if refined text is too short
            if len(refined_text) < 20:
                continue
            
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
        
        logger.info(f"Output saved to {output_path}")