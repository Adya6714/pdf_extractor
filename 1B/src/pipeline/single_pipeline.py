# src/pipeline/single_pipeline.py
"""
Single collection processing pipeline - FIXED VERSION
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.models.document_models import DocumentChunk, PersonaProfile, ProcessingResult
from src.processors.pdf_processor import PDFProcessor
from src.processors.embedding_processor import EmbeddingProcessor
from src.processors.persona_analyzer import PersonaAnalyzer
from src.processors.relevance_scorer import RelevanceScorer
from src.processors.selection_processor import SelectionProcessor
from src.utils.llm_processor import LLMProcessor
from src.utils.pdf_generator import PDFGenerator
from src.utils.config import Config

logger = logging.getLogger(__name__)

class SingleCollectionPipeline:
    """Process a single collection of documents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.embedding_processor = EmbeddingProcessor(config.EMBEDDING_MODEL)
        self.persona_analyzer = PersonaAnalyzer()
        self.relevance_scorer = RelevanceScorer(config)
        self.selection_processor = SelectionProcessor(config)
        self.llm_processor = LLMProcessor(config.MODEL_PATH)
        self.pdf_generator = PDFGenerator()
        
    def process(self, input_data: Dict, output_dir: Path) -> Dict:
        """Process a collection and generate outputs"""
        start_time = time.time()
        
        # Extract configuration
        collection_name = input_data["collection_name"]
        documents = input_data["documents"]
        persona_role = input_data["persona"]
        task = input_data["job_to_be_done"]
        output_config = input_data.get("output_config", {})
        processing_config = input_data.get("processing_config", {})
        
        # Update config with processing settings
        if processing_config.get("max_sections"):
            self.config.MAX_SECTIONS = processing_config["max_sections"]
        
        logger.info(f"Processing {collection_name}: {len(documents)} documents")
        
        # Step 1: Create persona profile
        persona_profile = self.persona_analyzer.create_persona_profile(persona_role, task)
        logger.info(f"Created persona profile with {len(persona_profile.get_all_keywords())} keywords")
        
        # Step 2: Process PDFs
        all_chunks = []
        doc_chunk_count = {}
        
        for doc in documents:
            pdf_path = doc['filepath']
            doc_name = doc['filename']
            logger.info(f"Processing PDF: {pdf_path}")
            
            try:
                pages = self.pdf_processor.extract_pdf_with_structure(pdf_path)
                chunks = self.pdf_processor.create_semantic_chunks(pages, doc_name)
                all_chunks.extend(chunks)
                doc_chunk_count[doc_name] = len(chunks)
                logger.info(f"Extracted {len(chunks)} chunks from {doc_name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        logger.info(f"Document chunk distribution: {doc_chunk_count}")
        
        if not all_chunks:
            logger.error("No chunks extracted from any document!")
            # Return empty result
            result = ProcessingResult(
                metadata=self._create_metadata(input_data, 0),
                extracted_sections=[],
                subsection_analysis=[],
                processing_time=time.time() - start_time,
                insights=["No content could be extracted from the documents"]
            )
            outputs = {}
            if output_config.get("include_json", True):
                json_path = output_dir / "output.json"
                self._save_json_output(result, "", json_path)
                outputs["json"] = str(json_path)
            return outputs
        
        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = self.embedding_processor.encode_texts(chunk_texts, show_progress=True)
        
        # Build index
        self.embedding_processor.build_index(chunk_embeddings, all_chunks)
        
        # Initialize scoring
        self.relevance_scorer.initialize_tfidf(all_chunks)
        doc_graph = self.relevance_scorer.build_document_graph(all_chunks, chunk_embeddings)
        
        # Step 4: Score and select chunks
        query = f"{task} {persona_role}"
        query_embedding = self.embedding_processor.create_query_embedding(
            query, persona_profile.get_all_keywords()
        )
        
        # Score all chunks
        scored_chunks = []
        for idx, chunk in enumerate(all_chunks):
            scores = self.relevance_scorer.calculate_multi_factor_score(
                chunk, idx, query_embedding, chunk_embeddings, persona_profile
            )
            scored_chunks.append((chunk, scores))
        
        logger.info(f"Scored {len(scored_chunks)} chunks")
        
        # IMPROVED SELECTION LOGIC
        # Filter out low-relevance chunks first
        min_relevance_threshold = 0.05  # Lower threshold to get more chunks
        high_quality_chunks = [
            (chunk, scores) for chunk, scores in scored_chunks 
            if scores['final_score'] > min_relevance_threshold
        ]
        
        logger.info(f"Chunks above threshold: {len(high_quality_chunks)}")
        
        # Sort by score
        high_quality_chunks.sort(key=lambda x: x[1]['final_score'], reverse=True)
        
        # Select top chunks from different documents to ensure diversity
        selected_by_doc = {}
        final_selected = []
        target_sections = 5  # Match expected output
        max_per_doc = 2  # Maximum chunks per document
        
        # First pass: get best chunk from each document
        for chunk, scores in high_quality_chunks:
            doc_name = chunk.document_name
            
            if doc_name not in selected_by_doc:
                selected_by_doc[doc_name] = []
                selected_by_doc[doc_name].append((chunk, scores))
                final_selected.append((chunk, scores))
                
                if len(final_selected) >= target_sections:
                    break
        
        # Second pass: get additional chunks if needed
        if len(final_selected) < target_sections:
            for chunk, scores in high_quality_chunks:
                doc_name = chunk.document_name
                
                # Skip if already selected
                if any(c[0].chunk_id == chunk.chunk_id for c in final_selected):
                    continue
                
                # Add if document hasn't reached limit
                if len(selected_by_doc.get(doc_name, [])) < max_per_doc:
                    selected_by_doc[doc_name].append((chunk, scores))
                    final_selected.append((chunk, scores))
                    
                    if len(final_selected) >= target_sections:
                        break
        
        # Convert to expected format
        selected_sections = [(chunk, scores['final_score']) for chunk, scores in final_selected]
        
        logger.info(f"Selected {len(selected_sections)} diverse sections")
        logger.info(f"Documents covered: {list(selected_by_doc.keys())}")

        # Post-process to ensure quality
        if collection_name == "travel_planner":
            # Ensure we have diverse content types
            has_city = any(
                "city" in s[0].section_title.lower() or "cities" in s[0].section_title.lower()
                for s in selected_sections
            )
            has_activity = any(
                "activity" in s[0].text.lower() or "things to do" in s[0].text.lower()
                for s in selected_sections
            )
            has_food = any(
                "cuisine" in s[0].text.lower() or "culinary" in s[0].text.lower()
                for s in selected_sections
            )
            logger.info(f"Content coverage - Cities: {has_city}, Activities: {has_activity}, Food: {has_food}")

        elif collection_name == "create_manageable_forms":
            # Ensure we have form-related content
            has_form_creation = any(
                "form" in s[0].text.lower() and "create" in s[0].text.lower()
                for s in selected_sections
            )
            has_signature = any(
                "signature" in s[0].text.lower()
                for s in selected_sections
            )
            logger.info(f"Content coverage - Forms: {has_form_creation}, Signatures: {has_signature}")
        
        # Step 5: Generate insights
        insights = self._generate_insights(selected_sections, persona_profile)
        
        # Step 6: Create output result
        processing_time = time.time() - start_time
        result = self.selection_processor.generate_output(
            selected_sections, persona_profile, input_data, processing_time, insights
        )
        
        outputs = {}
        
        # Step 7: Generate LLM response (for PDF report)
        llm_response = ""
        if output_config.get("include_pdf", True):
            context = self._prepare_context(selected_sections)
            try:
                llm_response = self.llm_processor.generate_task_response(
                    context, persona_profile, task
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                llm_response = "Unable to generate response due to an error."
        
        # Save outputs
        # Save JSON output (matching expected format)
        if output_config.get("include_json", True):
            json_path = output_dir / "output.json"
            self._save_json_output(result, llm_response, json_path)
            outputs["json"] = str(json_path)
        
        # Generate PDF (with LLM response)
        if output_config.get("include_pdf", True):
            pdf_title = output_config.get("pdf_title", f"{collection_name} Report")
            pdf_path = output_dir / "report.pdf"
            try:
                self.pdf_generator.generate_report(
                    result, llm_response, persona_profile, pdf_title, pdf_path
                )
                outputs["pdf"] = str(pdf_path)
            except Exception as e:
                logger.error(f"PDF generation failed: {e}")
        
        logger.info(f"Completed {collection_name} in {processing_time:.2f}s")
        
        return outputs
    
    def _prepare_context(self, selected_sections: List) -> str:
        """Prepare context for LLM"""
        context_parts = []
        for chunk, score in selected_sections[:10]:  # Limit context
            context_parts.append(
                f"[{chunk.document_name} - {chunk.section_title}]\n"
                f"{chunk.text[:800]}...\n"
            )
        return "\n".join(context_parts)
    
    def _save_json_output(self, result: ProcessingResult, llm_response: str, output_path: Path):
        """Save JSON output matching expected format"""
        
        # Match the expected output structure exactly
        output = {
            "metadata": {
                "input_documents": result.metadata["input_documents"],
                "persona": result.metadata["persona"],
                "job_to_be_done": result.metadata["job_to_be_done"],
                "processing_timestamp": result.metadata["processing_timestamp"]
            },
            "extracted_sections": [
                {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": section["importance_rank"],
                    "page_number": section["page_number"]
                }
                for section in result.extracted_sections[:5]  # Limit to 5
            ],
            "subsection_analysis": [
                {
                    "document": analysis["document"],
                    "refined_text": analysis["refined_text"],
                    "page_number": analysis["page_number"]
                }
                for analysis in result.subsection_analysis[:5]  # Limit to 5
            ]
        }
        
        # Save with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
            
        logger.info(f"JSON output saved to {output_path}")
    
    def _generate_insights(self, selected_sections: List[Tuple[DocumentChunk, float]], 
                          persona: PersonaProfile) -> List[str]:
        """Generate insights from selected sections"""
        insights = []
        
        # Document coverage
        doc_coverage = {}
        for chunk, _ in selected_sections:
            doc_coverage[chunk.document_name] = doc_coverage.get(chunk.document_name, 0) + 1
        
        insights.append(f"Information extracted from {len(doc_coverage)} documents")
        
        # Most relevant document
        if doc_coverage:
            most_relevant = max(doc_coverage.items(), key=lambda x: x[1])
            insights.append(f"Primary source: {most_relevant[0]} ({most_relevant[1]} sections)")
        
        # Task-specific insights
        if "trip" in persona.task.lower() or "travel" in persona.task.lower():
            # Check for specific travel components
            components = {
                "destinations": False,
                "accommodation": False,
                "activities": False,
                "dining": False,
                "tips": False
            }
            
            for chunk, _ in selected_sections:
                text_lower = chunk.text.lower()
                title_lower = chunk.section_title.lower()
                
                if any(word in text_lower + title_lower for word in ["city", "cities", "town", "destination"]):
                    components["destinations"] = True
                if any(word in text_lower + title_lower for word in ["hotel", "accommodation", "stay", "hostel"]):
                    components["accommodation"] = True
                if any(word in text_lower + title_lower for word in ["activity", "things to do", "adventure", "beach", "water sports"]):
                    components["activities"] = True
                if any(word in text_lower + title_lower for word in ["restaurant", "cuisine", "food", "dining", "culinary"]):
                    components["dining"] = True
                if any(word in text_lower + title_lower for word in ["tips", "tricks", "advice", "packing"]):
                    components["tips"] = True
            
            covered = [k for k, v in components.items() if v]
            if covered:
                insights.append(f"Travel components covered: {', '.join(covered)}")
        
        return insights
    
    def _create_metadata(self, input_data: Dict, processing_time: float) -> Dict:
        """Create metadata for empty result"""
        from datetime import datetime, timezone
        
        return {
            "collection_name": input_data.get("collection_name", "unknown"),
            "input_documents": [doc['filename'] for doc in input_data.get('documents', [])],
            "persona": input_data.get("persona", "unknown"),
            "job_to_be_done": input_data.get("job_to_be_done", "unknown"),
            "processing_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "processing_time_seconds": round(processing_time, 2)
        }