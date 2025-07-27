# test_system.py
import json
import os
from main import PersonaDocumentIntelligence

def create_test_input():
    """Create a test input file"""
    test_input = {
        "challenge_info": {
            "challenge_id": "round_1b_test",
            "test_case_name": "test_case"
        },
        "documents": [
            {
                "filename": "test_doc1.pdf",
                "title": "Test Document 1"
            },
            {
                "filename": "test_doc2.pdf", 
                "title": "Test Document 2"
            }
        ],
        "persona": {
            "role": "Research Analyst"
        },
        "job_to_be_done": {
            "task": "Analyze market trends and competitive landscape"
        }
    }
    
    with open("test_input.json", "w") as f:
        json.dump(test_input, f, indent=2)
    
    return "test_input.json"

def run_test():
    """Run a test of the system"""
    # Create test input
    input_path = create_test_input()
    
    # Initialize system
    system = PersonaDocumentIntelligence()
    
    # Process
    result = system.process_documents(input_path)
    
    # Display results
    print("\nTest Results:")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Sections found: {len(result.extracted_sections)}")
    
    # Show top 3 sections
    print("\nTop 3 sections:")
    for section in result.extracted_sections[:3]:
        print(f"- {section['section_title']} (score: {section.get('relevance_score', 'N/A')})")

if __name__ == "__main__":
    run_test()
