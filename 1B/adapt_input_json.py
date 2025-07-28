import json
from pathlib import Path

def adapt_input_json(collection_dir):
    input_path = Path(collection_dir) / "input.json"
    with open(input_path, "r") as f:
        data = json.load(f)

    adapted = {
        "collection_name": data["challenge_info"]["test_case_name"],
        "persona": data["persona"]["role"],
        "job_to_be_done": data["job_to_be_done"]["task"],
        "output_format": {
            "include_json": True,
            "include_pdf": True,
            "pdf_title": f"{data['challenge_info']['description']} - {data['persona']['role']}"
        },
        "processing_config": {
            "max_sections": 25,
            "min_relevance_score": 0.5
        }
    }

    adapted_path = Path(collection_dir) / "input_adapted.json"
    with open(adapted_path, 'w') as f:
        json.dump(adapted, f, indent=2)

    print(f"Created adapted input for {collection_dir}")
    return adapted

# Process all collections
for i in range(1, 4):
    collection_dir = f"Collections/Collection {i}"
    adapt_input_json(collection_dir)
