# src/processors/persona_analyzer.py
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Set
from src.models.document_models import PersonaProfile
import re

class PersonaAnalyzer:
    """Analyzes and expands persona profiles"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Persona templates
        self.persona_templates = {
            "researcher": {
                "keywords": ["methodology", "hypothesis", "data", "analysis", "findings", 
                           "results", "literature", "study", "research", "experiment"],
                "preferred_sections": ["abstract", "methodology", "results", "discussion"],
                "focus": "scientific_rigor"
            },
            "student": {
                "keywords": ["learn", "understand", "concept", "example", "practice", 
                           "definition", "basics", "fundamental", "exercise", "quiz"],
                "preferred_sections": ["introduction", "examples", "summary", "key concepts"],
                "focus": "educational_clarity"
            },
            "analyst": {
                "keywords": ["trend", "metric", "performance", "comparison", "insight",
                           "revenue", "growth", "market", "competitive", "strategy"],
                "preferred_sections": ["executive summary", "financial", "market analysis"],
                "focus": "business_insights"
            },
            "travel planner": {
                "keywords": ["itinerary", "destination", "accommodation", "restaurant",
                           "attraction", "transport", "budget", "tips", "guide", "visit"],
                "preferred_sections": ["attractions", "dining", "accommodation", "transportation"],
                "focus": "practical_planning"
            },
            "hr professional": {
                "keywords": ["form", "compliance", "onboarding", "policy", "procedure",
                           "documentation", "employee", "benefit", "training", "process"],
                "preferred_sections": ["forms", "procedures", "policies", "guidelines"],
                "focus": "compliance_efficiency"
            },
            "food contractor": {
                "keywords": ["recipe", "ingredient", "preparation", "serving", "menu",
                           "catering", "vegetarian", "dietary", "portion", "cooking"],
                "preferred_sections": ["recipes", "ingredients", "preparation", "serving suggestions"],
                "focus": "culinary_execution"
            }
        }
    
    def create_persona_profile(self, role: str, task: str) -> PersonaProfile:
        """Create enriched persona profile from role and task"""
        profile = PersonaProfile(role=role, task=task)
        
        # Extract base keywords from role
        role_lower = role.lower()
        for persona_type, template in self.persona_templates.items():
            if persona_type in role_lower or any(word in role_lower for word in persona_type.split()):
                profile.domain_keywords = template["keywords"]
                profile.preferred_sections = template["preferred_sections"]
                break
        
        # Extract task keywords
        profile.task_keywords = self._extract_task_keywords(task)
        
        # Extract intent keywords
        profile.intent_keywords = self._extract_intent_keywords(task)
        
        # Expand keywords using WordNet
        expanded_keywords = self._expand_keywords_wordnet(profile.get_all_keywords())
        profile.domain_keywords.extend(expanded_keywords)
        
        return profile
    
    def _extract_task_keywords(self, task: str) -> List[str]:
        """Extract important keywords from task description"""
        # Tokenize and POS tag
        tokens = nltk.word_tokenize(task.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract nouns and important verbs
        keywords = []
        stopwords = set(nltk.corpus.stopwords.words('english'))
        
        for word, pos in pos_tags:
            if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG']:  # Nouns and verbs
                if len(word) > 3 and word not in stopwords:
                    keywords.append(word)
        
        # Extract phrases (bigrams/trigrams)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if all(len(w) > 2 for w in [tokens[i], tokens[i+1]]):
                keywords.append(bigram)
        
        return list(set(keywords))
    
    def _extract_intent_keywords(self, task: str) -> List[str]:
        """Extract action/intent keywords from task"""
        intent_patterns = [
            r'\b(plan|create|analyze|prepare|find|identify|extract|summarize|compare)\b',
            r'\b(evaluate|assess|review|compile|organize|develop|design)\b'
        ]
        
        intents = []
        task_lower = task.lower()
        
        for pattern in intent_patterns:
            matches = re.findall(pattern, task_lower)
            intents.extend(matches)
        
        return list(set(intents))
    
    def _expand_keywords_wordnet(self, keywords: List[str], max_expand: int = 2) -> List[str]:
        """Expand keywords using WordNet synonyms"""
        expanded = set()
        
        for keyword in keywords[:10]:  # Limit to avoid explosion
            synsets = wordnet.synsets(keyword)
            
            for synset in synsets[:max_expand]:
                # Add synonyms
                for lemma in synset.lemmas()[:3]:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != keyword and len(synonym) > 3:
                        expanded.add(synonym)
        
        return list(expanded)
    
    def calculate_persona_alignment(self, text: str, profile: PersonaProfile) -> float:
        """Calculate how well text aligns with persona needs"""
        text_lower = text.lower()
        score = 0.0
        
        # Check domain keywords
        for keyword in profile.domain_keywords:
            score += text_lower.count(keyword) * 2.0
        
        # Check task keywords
        for keyword in profile.task_keywords:
            score += text_lower.count(keyword) * 3.0
        
        # Check preferred sections
        for section in profile.preferred_sections:
            if section.lower() in text_lower:
                score += 5.0
        
        # Normalize by text length
        score = score / (len(text_lower.split()) + 1)
        
        return min(score, 1.0)  # Cap at 1.0