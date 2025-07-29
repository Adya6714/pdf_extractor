# main.py
"""
Main entry point for the PDF processing pipeline
Modified to handle challenge input format
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline.batch_pipeline import BatchPipeline
from src.pipeline.single_pipeline import SingleCollectionPipeline
from src.utils.config import Config

# Configure logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/pipeline.log')
        ]
    )

# def download_model_if_needed(config: Config):
#     """Download model if not present"""
#     model_path = Path(config.MODEL_PATH)
#     if not model_path.exists():
#         print("Model not found. Downloading TinyLlama...")
#         import urllib.request
        
#         model_path.parent.mkdir(exist_ok=True)
#         url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        
#         def download_progress(block_num, block_size, total_size):
#             downloaded = block_num * block_size
#             percent = min(downloaded * 100 / total_size, 100)
#             sys.stdout.write(f'\rDownload progress: {percent:.1f}%')
#             sys.stdout.flush()
        
#         urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
#         print(f"\nModel downloaded to {model_path}")

def transform_challenge_format(challenge_data: dict, collection_path: Path) -> dict:
    """Transform challenge JSON format to pipeline format"""
    
    # Get PDFs directory
    pdfs_dir = collection_path / "PDFs"
    
    # Build documents list with full paths
    documents = []
    for doc_info in challenge_data.get("documents", []):
        pdf_filename = doc_info["filename"]
        pdf_path = pdfs_dir / pdf_filename
        
        if pdf_path.exists():
            documents.append({
                "filename": pdf_filename,
                "filepath": str(pdf_path),
                "title": doc_info.get("title", pdf_filename)
            })
        else:
            logging.warning(f"PDF not found: {pdf_path}")
    
    # Extract persona and task
    persona = challenge_data.get("persona", {}).get("role", "analyst")
    task = challenge_data.get("job_to_be_done", {}).get("task", "Analyze documents")
    
    # Get challenge info
    challenge_info = challenge_data.get("challenge_info", {})
    
    # Build pipeline-compatible format
    return {
        "collection_name": challenge_info.get("test_case_name", "unknown"),
        "challenge_id": challenge_info.get("challenge_id", "unknown"),
        "description": challenge_info.get("description", ""),
        "documents": documents,
        "persona": persona,
        "job_to_be_done": task,
        "output_config": {
            "include_json": True,
            "include_pdf": True,
            "pdf_title": f"{challenge_info.get('description', 'Report')} - {persona}"
        },
        "processing_config": {
            "max_sections": 25,
            "min_relevance_score": 0.5
        }
    }

def load_collection_config(collection_path: Path) -> dict:
    """Load and parse collection configuration"""
    
    # Try different input file names
    input_files = [
        "input.json",
        "challenge1b_input.json",
        "challenge_input.json"
    ]
    
    input_data = None
    input_file = None
    
    for filename in input_files:
        test_path = collection_path / filename
        if test_path.exists():
            input_file = test_path
            break
    
    if not input_file:
        raise FileNotFoundError(f"No input JSON found in {collection_path}")
    
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    # Check if it's challenge format
    if "challenge_info" in input_data:
        return transform_challenge_format(input_data, collection_path)
    else:
        # Already in correct format
        return input_data

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PDF Processing Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["batch", "single"], 
        default="batch",
        help="Processing mode: batch (all collections) or single"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Collection name for single mode"
    )
    parser.add_argument(
        "--collections-dir",
        type=str,
        default="collections",
        help="Directory containing collections"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Enable parallel processing"
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create config
    config = Config()
    
    # Download model if needed
    # download_model_if_needed(config)
    
    try:
        if args.mode == "batch":
            # Process all collections
            logger.info("Starting batch processing")
            
            # Custom batch processing for challenge format
            collections_path = Path(args.collections_dir)
            collection_dirs = [d for d in collections_path.iterdir() if d.is_dir()]
            
            results = {}
            
            for collection_dir in collection_dirs:
                try:
                    logger.info(f"Processing {collection_dir.name}...")
                    
                    # Load configuration
                    input_data = load_collection_config(collection_dir)
                    
                    # Create output directory
                    output_dir = Path(config.OUTPUT_DIR) / collection_dir.name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process collection
                    pipeline = SingleCollectionPipeline(config)
                    outputs = pipeline.process(input_data, output_dir)
                    
                    results[collection_dir.name] = {
                        "status": "completed",
                        "collection": collection_dir.name,
                        "challenge_id": input_data.get("challenge_id", "unknown"),
                        "outputs": outputs
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to process {collection_dir.name}: {e}", exc_info=True)
                    results[collection_dir.name] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Print summary
            print("\n" + "="*50)
            print("BATCH PROCESSING COMPLETE")
            print("="*50)
            
            successful = sum(1 for r in results.values() if r.get("status") == "completed")
            failed = sum(1 for r in results.values() if r.get("status") == "failed")
            
            print(f"\nTotal collections: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            for collection, result in results.items():
                status = result.get("status", "unknown")
                print(f"\n{collection}: {status.upper()}")
                
                if status == "completed":
                    outputs = result.get("outputs", {})
                    if "json" in outputs:
                        print(f"  - JSON: {outputs['json']}")
                    if "pdf" in outputs:
                        print(f"  - PDF: {outputs['pdf']}")
                    print(f"  - Challenge ID: {result.get('challenge_id', 'unknown')}")
                elif status == "failed":
                    print(f"  - Error: {result.get('error', 'Unknown error')}")
            
            # Save batch summary
            summary = {
                "processing_summary": {
                    "total_collections": len(results),
                    "successful": successful,
                    "failed": failed
                },
                "collections": results,
                "timestamp": datetime.now(datetime.UTC).isoformat()
            }
            
            summary_path = Path(config.OUTPUT_DIR) / "batch_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {summary_path}")
            
        else:
            # Single collection mode
            if not args.collection:
                parser.error("--collection is required in single mode")
            
            collection_path = Path(args.collections_dir) / args.collection
            if not collection_path.exists():
                raise ValueError(f"Collection not found: {collection_path}")
            
            logger.info(f"Processing single collection: {args.collection}")
            
            # Load configuration
            input_data = load_collection_config(collection_path)
            
            logger.info(f"Loaded configuration for: {input_data.get('collection_name', args.collection)}")
            logger.info(f"Persona: {input_data.get('persona', 'unknown')}")
            logger.info(f"Task: {input_data.get('job_to_be_done', 'unknown')}")
            logger.info(f"Documents: {len(input_data.get('documents', []))}")
            
            # Create output directory
            output_dir = Path(config.OUTPUT_DIR) / args.collection
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process
            pipeline = SingleCollectionPipeline(config)
            outputs = pipeline.process(input_data, output_dir)
            
            print(f"\nProcessing complete for {args.collection}!")
            print(f"Challenge ID: {input_data.get('challenge_id', 'unknown')}")
            print(f"Outputs saved to: {output_dir}")
            
            if "json" in outputs:
                print(f"  - JSON: {outputs['json']}")
            if "pdf" in outputs:
                print(f"  - PDF: {outputs['pdf']}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()