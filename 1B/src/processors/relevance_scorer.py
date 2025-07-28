# src/processors/relevance_scorer.py
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.document_models import DocumentChunk, PersonaProfile
import networkx as nx
from collections import Counter
import re

class RelevanceScorer:
    """Multi-factor relevance scoring system"""
    
    def __init__(self, config):
        self.config = config
        self.tfidf_vectorizer = None
        self.document_graph = None
    
    def initialize_tfidf(self, chunks: List[DocumentChunk]):
        """Initialize TF-IDF vectorizer with all chunks"""
        texts = [chunk.text for chunk in chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def calculate_multi_factor_score(self, 
                                   chunk: DocumentChunk,
                                   chunk_idx: int,
                                   query_embedding: np.ndarray,
                                   chunk_embeddings: np.ndarray,
                                   persona_profile: PersonaProfile) -> Dict[str, float]:
        """Calculate comprehensive relevance score with multiple factors"""
        
        scores = {}
        
        # 1. Semantic similarity score
        scores['semantic'] = self._calculate_semantic_score(
            chunk_embeddings[chunk_idx], query_embedding
        )
        
        # 2. Keyword relevance score
        scores['keyword'] = self._calculate_keyword_score(chunk, persona_profile)
        
        # 3. Structural importance score
        scores['structural'] = self._calculate_structural_score(chunk)
        
        # 4. Contextual relevance score
        scores['contextual'] = self._calculate_contextual_score(
            chunk_idx, chunk, persona_profile
        )
        
        # 5. Cross-reference score
        if self.document_graph:
            scores['cross_reference'] = self._calculate_cross_reference_score(
                chunk_idx, self.document_graph
            )
        else:
            scores['cross_reference'] = 0.0
        
        # Calculate weighted final score
        weights = self._get_persona_weights(persona_profile)
        final_score = sum(scores[key] * weights.get(key, 0) for key in scores)
        
        return {
            'final_score': final_score,
            'component_scores': scores,
            'weights': weights
        }
    
    def _calculate_semantic_score(self, chunk_embedding: np.ndarray, 
                                 query_embedding: np.ndarray) -> float:
        """Calculate semantic similarity score"""
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(chunk_embedding, query_embedding)
        
        # Apply non-linear transformation to spread scores
        return self._sigmoid_transform(similarity, midpoint=0.5, steepness=10)
    
    def _calculate_keyword_score(self, chunk: DocumentChunk, profile: PersonaProfile) -> float:
        """Calculate keyword relevance score"""
        text_lower = chunk.text.lower()
        title_lower = chunk.section_title.lower()
        score = 0.0
        
        # Persona-specific scoring
        if "travel" in profile.role.lower():
            # Travel-specific keywords
            travel_keywords = ["beach", "city", "cities", "hotel", "restaurant", "activity", "activities",
                              "nightlife", "entertainment", "tip", "trick", "pack", "culinary", "coastal",
                              "adventure", "things to do", "cuisine", "culture", "tradition"]
            for keyword in travel_keywords:
                if keyword in text_lower:
                    score += 2.0
                if keyword in title_lower:
                    score += 5.0
        
        elif "hr" in profile.role.lower():
            # HR/Form-specific keywords
            hr_keywords = ["form", "fillable", "sign", "signature", "create", "convert", "pdf", 
                          "document", "field", "interactive", "compliance", "onboarding", "request",
                          "e-signature", "prepare", "manage", "employee", "process"]
            for keyword in hr_keywords:
                if keyword in text_lower:
                    score += 2.0
                if keyword in title_lower:
                    score += 5.0
        
        elif "food" in profile.role.lower() or "contractor" in profile.role.lower():
            # Food-specific keywords
            food_keywords = ["vegetarian", "vegan", "gluten-free", "recipe", "ingredient", "buffet",
                            "menu", "dish", "serve", "portion", "cook", "prepare", "dinner", "side",
                            "main", "appetizer", "salad", "soup"]
            for keyword in food_keywords:
                if keyword in text_lower:
                    score += 2.0
                if keyword in title_lower:
                    score += 5.0
            
            # Boost for actual dish names (capitalized words that aren't common headers)
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*$', chunk.section_title):
                if not any(skip in title_lower for skip in ['ingredients', 'instructions', 'serves']):
                    score += 10.0  # Big boost for dish names
        
        # Original keyword scoring
        for keyword in profile.domain_keywords:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            score += count * 1.5
        
        # Task keywords
        for keyword in profile.task_keywords:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            score += count * 2.0
        
        # Normalize and scale
        word_count = len(text_lower.split())
        normalized_score = score / (word_count + 10)
        
        return min(normalized_score * 2, 1.0)
    
    def _calculate_structural_score(self, chunk: DocumentChunk) -> float:
        """Calculate structural importance score"""
        # Headers and titles get higher scores
        level_scores = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        base_score = level_scores.get(chunk.structural_level, 0.3)
        
        # Boost for chunks at beginning of sections
        position_boost = 0.0
        if chunk.start_char < 500:  # Early in document
            position_boost = 0.2
        
        # Boost for chunks with structured content
        structure_indicators = [
            r'^\d+\.',  # Numbered lists
            r'^\s*[•\-\*]',  # Bullet points
            r':$',  # Headers ending with colon
            r'\b(?:Table|Figure|Chart)\s+\d+',  # References to visuals
        ]
        
        structure_boost = 0.0
        for pattern in structure_indicators:
            if re.search(pattern, chunk.text, re.MULTILINE):
                structure_boost += 0.1
        
        return min(base_score + position_boost + structure_boost, 1.0)
    
    def _calculate_contextual_score(self, chunk_idx: int, chunk: DocumentChunk,
                                   profile: PersonaProfile) -> float:
        """Calculate contextual relevance score"""
        score = 0.0
        
        # Check if chunk contains preferred sections
        text_lower = chunk.text.lower()
        title_lower = chunk.section_title.lower()
        
        for preferred in profile.preferred_sections:
            if preferred.lower() in title_lower:
                score += 0.5
            elif preferred.lower() in text_lower:
                score += 0.2
        
        # Check for persona-specific patterns
        if "researcher" in profile.role.lower():
            # Look for research-specific patterns
            research_patterns = [
                r'\b(?:study|studies|research|experiment)\b',
                r'\b(?:n\s*=\s*\d+)\b',  # Sample size
                r'\b(?:p\s*[<>]\s*0\.\d+)\b',  # P-values
                r'\b(?:methodology|method)\b'
            ]
            for pattern in research_patterns:
                if re.search(pattern, text_lower):
                    score += 0.1
        
        elif "student" in profile.role.lower():
            # Look for educational patterns
            edu_patterns = [
                r'\b(?:example|for instance|such as)\b',
                r'\b(?:definition|defined as|means)\b',
                r'\b(?:key concepts?|important|remember)\b'
            ]
            for pattern in edu_patterns:
                if re.search(pattern, text_lower):
                    score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_cross_reference_score(self, chunk_idx: int, 
                                       graph: nx.Graph) -> float:
        """Calculate cross-reference importance score"""
        if chunk_idx not in graph:
            return 0.0
        
        # Use PageRank for importance
        try:
            pagerank_scores = nx.pagerank(graph, alpha=0.85)
            score = pagerank_scores.get(chunk_idx, 0.0)
            
            # Normalize to [0, 1]
            max_score = max(pagerank_scores.values())
            if max_score > 0:
                score = score / max_score
            
            return score
        except:
            # Fallback to degree centrality
            degree = graph.degree(chunk_idx)
            max_degree = max(dict(graph.degree()).values())
            return degree / max_degree if max_degree > 0 else 0.0
    
    def _get_persona_weights(self, profile: PersonaProfile) -> Dict[str, float]:
        """Get scoring weights based on persona"""
        # For travel planner, prioritize keyword matching and structural importance
        if "travel" in profile.role.lower() or "planner" in profile.role.lower():
            weights = {
                'semantic': 0.20,      # Reduced from 0.35
                'keyword': 0.45,       # Increased from 0.35
                'structural': 0.25,    # Increased from 0.15
                'contextual': 0.10,    # Same
                'cross_reference': 0.00 # Removed
            }
        else:
            # Default weights
            weights = {
                'semantic': 0.35,
                'keyword': 0.35,
                'structural': 0.15,
                'contextual': 0.10,
                'cross_reference': 0.05
            }
        
        # Adjust based on persona
        if "researcher" in profile.role.lower():
            weights['semantic'] = 0.40
            weights['structural'] = 0.20
            weights['cross_reference'] = 0.10
            weights['keyword'] = 0.20
            weights['contextual'] = 0.10
        
        elif "student" in profile.role.lower():
            weights['semantic'] = 0.30
            weights['keyword'] = 0.35
            weights['structural'] = 0.20
            weights['contextual'] = 0.15
            weights['cross_reference'] = 0.00
        
        elif "analyst" in profile.role.lower():
            weights['semantic'] = 0.30
            weights['keyword'] = 0.40
            weights['structural'] = 0.10
            weights['contextual'] = 0.15
            weights['cross_reference'] = 0.05
        
        return weights
    
    def _sigmoid_transform(self, x: float, midpoint: float = 0.5, 
                          steepness: float = 10) -> float:
        """Apply sigmoid transformation to spread scores"""
        return 1 / (1 + np.exp(-steepness * (x - midpoint)))
    
    def build_document_graph(self, chunks: List[DocumentChunk], 
                           embeddings: np.ndarray) -> nx.Graph:
        """Build graph representation of document chunks"""
        G = nx.Graph()
        
        # Add nodes
        for i, chunk in enumerate(chunks):
            G.add_node(i, 
                      chunk_id=chunk.chunk_id,
                      document=chunk.document_name,
                      section=chunk.section_title,
                      level=chunk.structural_level)
        
        # Add edges based on similarity
        similarity_threshold = 0.5
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                similarity = np.dot(embeddings[i], embeddings[j])
                
                if similarity > similarity_threshold:
                    # Also check for same document or section
                    same_doc = chunks[i].document_name == chunks[j].document_name
                    same_section = chunks[i].section_title == chunks[j].section_title
                    
                    # Adjust weight based on relationship
                    weight = similarity
                    if same_section:
                        weight += 0.2
                    elif same_doc:
                        weight += 0.1
                    
                    G.add_edge(i, j, weight=weight)
        
        self.document_graph = G
        return G