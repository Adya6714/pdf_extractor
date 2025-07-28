# main.py
import os
import json
import time
import logging
import warnings
from typing import List
warnings.filterwarnings('ignore')

from config import Config
from models.document_models import DocumentChunk, PersonaProfile, ProcessingResult
from processors.pdf_processor import PDFProcessor
from processors.persona_analyzer import PersonaAnalyzer
from processors.embedding_processor import EmbeddingProcessor
from processors.relevance_scorer import RelevanceScorer
from processors.selection_processor import SelectionProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonaDocumentIntelligence:
    def __init__(self):
        self.config              = Config()
        self.pdf_processor       = PDFProcessor(self.config)
        self.persona_analyzer    = PersonaAnalyzer()
        self.embedding_processor = EmbeddingProcessor(self.config.EMBEDDING_MODEL)
        self.relevance_scorer    = RelevanceScorer(self.config)
        self.selection_processor = SelectionProcessor(self.config)

    def process_documents(self, input_path: str) -> ProcessingResult:
        start = time.time()
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs = data['documents']
        role = data['persona']['role']
        task = data['job_to_be_done']['task']

        # Persona profile
        profile = self.persona_analyzer.create_persona_profile(role, task)

        # Load & chunk PDFs
        all_chunks = []
        for d in docs:
            pdf_file = os.path.join(self.config.PDF_DIR, d['filename'])
            if not os.path.exists(pdf_file):
                logger.warning(f"Missing PDF: {pdf_file}")
                continue
            pages  = self.pdf_processor.extract_pdf_with_structure(pdf_file)
            chunks = self.pdf_processor.create_semantic_chunks(pages, d['filename'])
            all_chunks.extend(chunks)

        # Embedding & index
        texts  = [c.text for c in all_chunks]
        embeds = self.embedding_processor.encode_texts(texts)
        self.embedding_processor.build_index(embeds, all_chunks)

        # Query embedding
        q_emb = self.embedding_processor.create_query_embedding(task, profile.get_all_keywords())

        # Scoring setup
        self.relevance_scorer.initialize_tfidf(all_chunks)
        self.relevance_scorer.build_document_graph(all_chunks, embeds)

        # Score chunks
        scored = [
            (chunk, self.relevance_scorer.calculate_multi_factor_score(
                chunk, idx, q_emb, embeds, profile
            ))
            for idx, chunk in enumerate(all_chunks)
        ]

        # Select + output
        selected = self.selection_processor.select_diverse_sections(scored, embeds)
        result   = self.selection_processor.generate_output(
            selected, profile, data, time.time() - start
        )
        return result

def main():
    system = PersonaDocumentIntelligence()

    for col in sorted(os.listdir(".")):
        if not os.path.isdir(col) or not col.lower().startswith("collection"):
            continue

        input_json = os.path.join(col, "challenge1b_input.json")
        pdf_dir    = os.path.join(col, "PDFs")
        out_json   = os.path.join(col, "challenge1b_output.json")  # ← save directly in CollectionX

        if not os.path.exists(input_json):
            print(f"→ skip {col}: no challenge1b_input.json")
            continue

        if not os.path.isdir(pdf_dir):
            print(f"→ skip {col}: PDFs folder missing")
            continue

        system.config.PDF_DIR    = pdf_dir
        system.config.OUTPUT_DIR = col  # just for consistency if any part uses it

        print(f"\n=== Processing {col} ===")
        try:
            result = system.process_documents(input_json)
            system.selection_processor.save_output(result, out_json)
            print(f"✅ Output saved to: {out_json}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Error in {col}: {e}")

    print("\nAll collections complete.")

if __name__ == "__main__":
    main()