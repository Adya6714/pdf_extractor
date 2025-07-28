"""
Batch processing pipeline adapted for challenge format
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import concurrent.futures
from datetime import datetime

from src.pipeline.single_pipeline import SingleCollectionPipeline
from src.utils.config import Config

logger = logging.getLogger(__name__)

class BatchPipeline:
    """Process multiple collections in batch"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def process_all_collections(self, collections_dir: str = "collections", 
                              parallel: bool = True, max_workers: int = 3) -> Dict:
        """Process all collections in the directory"""
        collections_path = Path(collections_dir)
        collection_dirs = [d for d in collections_path.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(collection_dirs)} collections to process")
        
        if parallel and len(collection_dirs) > 1:
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_collection = {
                    executor.submit(self._process_single_collection, collection_dir): collection_dir
                    for collection_dir in collection_dirs
                }
                
                for future in concurrent.futures.as_completed(future_to_collection):
                    collection_dir = future_to_collection[future]
                    try:
                        result = future.result()
                        self.results[collection_dir.name] = result
                    except Exception as e:
                        logger.error(f"Failed to process {collection_dir.name}: {e}")
                        self.results[collection_dir.name] = {"status": "failed", "error": str(e)}
        else:
            # Process sequentially
            for collection_dir in collection_dirs:
                try:
                    result = self._process_single_collection(collection_dir)
                    self.results[collection_dir.name] = result
                except Exception as e:
                    logger.error(f"Failed to process {collection_dir.name}: {e}")
                    self.results[collection_dir.name] = {"status": "failed", "error": str(e)}
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.results
    
    def _process_single_collection(self, collection_dir: Path) -> Dict:
        """Process a single collection"""
        logger.info(f"Processing collection: {collection_dir.name}")
        
        # Try to load input.json first, then adapted version
        input_file = collection_dir / "input.json"
        if not input_file.exists():
            # Try challenge format
            input_file = collection_dir / "challenge1b_input.json"
            if not input_file.exists():
                raise FileNotFoundError(f"No input.json found in {collection_dir}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Check if it's challenge format
        if "challenge_info" in data:
            # Transform challenge format
            collection_config = {
                "collection_name": data["challenge_info"]["test_case_name"],
                "persona": data["persona"]["role"],
                "job_to_be_done": data["job_to_be_done"]["task"],
                "documents": data["documents"]
            }
        else:
            collection_config = data
        
        # Get PDFs
        pdfs_dir = collection_dir / "PDFs"
        if not pdfs_dir.exists():
            raise FileNotFoundError(f"No PDFs directory found in {collection_dir}")
        
        # Prepare documents list with full paths
        documents = []
        for doc_info in collection_config.get("documents", []):
            pdf_filename = doc_info["filename"]
            pdf_path = pdfs_dir / pdf_filename
            if pdf_path.exists():
                documents.append({
                    "filename": pdf_filename,
                    "filepath": str(pdf_path)
                })
            else:
                logger.warning(f"PDF not found: {pdf_path}")
        
        if not documents:
            # Fallback: get all PDFs in directory
            pdf_files = list(pdfs_dir.glob("*.pdf"))
            documents = [
                {"filename": pdf.name, "filepath": str(pdf)}
                for pdf in pdf_files
            ]
        
        # Prepare input data
        input_data = {
            "collection_name": collection_config.get("collection_name", collection_dir.name),
            "documents": documents,
            "persona": collection_config.get("persona", "analyst"),
            "job_to_be_done": collection_config.get("job_to_be_done", "Analyze documents"),
            "output_config": {
                "include_json": True,
                "include_pdf": True,
                "pdf_title": f"{collection_config.get('collection_name', 'Report')} - {collection_config.get('persona', 'Analysis')}"
            },
            "processing_config": collection_config.get("processing_config", {
                "max_sections": 25,
                "min_relevance_score": 0.5
            })
        }
        
        # Create output directory
        output_dir = Path(self.config.OUTPUT_DIR) / collection_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process collection
        pipeline = SingleCollectionPipeline(self.config)
        result = pipeline.process(input_data, output_dir)
        
        return {
            "status": "completed",
            "collection": collection_dir.name,
            "challenge_id": data.get("challenge_info", {}).get("challenge_id", "unknown"),
            "outputs": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_summary_report(self):
        """Generate a summary report of all processing"""
        summary = {
            "processing_summary": {
                "total_collections": len(self.results),
                "successful": sum(1 for r in self.results.values() if r.get("status") == "completed"),
                "failed": sum(1 for r in self.results.values() if r.get("status") == "failed")
            },
            "collections": self.results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save summary
        summary_path = Path(self.config.OUTPUT_DIR) / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")