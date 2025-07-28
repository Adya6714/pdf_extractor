# validate_outputs.py - FIXED VERSION
"""
Validation script to compare generated outputs with expected challenge outputs
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import difflib
from datetime import datetime
import pandas as pd
from collections import defaultdict

class OutputValidator:
    """Validate generated outputs against expected outputs"""
    
    def __init__(self):
        self.results = {}
        self.detailed_diffs = {}
        
    def load_json(self, filepath: Path) -> Dict:
        """Load JSON file safely"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compare_structures(self, generated: Dict, expected: Dict) -> Dict:
        """Compare the structure of two JSON objects"""
        results = {
            "missing_keys": [],
            "extra_keys": [],
            "type_mismatches": {},
            "structure_match_score": 0.0
        }
        
        gen_keys = set(self._get_all_keys(generated))
        exp_keys = set(self._get_all_keys(expected))
        
        # Find missing and extra keys
        results["missing_keys"] = list(exp_keys - gen_keys)
        results["extra_keys"] = list(gen_keys - exp_keys)
        
        # Check type mismatches for common keys
        common_keys = gen_keys & exp_keys
        for key in common_keys:
            gen_val = self._get_nested_value(generated, key)
            exp_val = self._get_nested_value(expected, key)
            
            if type(gen_val) != type(exp_val):
                results["type_mismatches"][key] = {
                    "generated": type(gen_val).__name__,
                    "expected": type(exp_val).__name__
                }
        
        # Calculate structure match score
        total_keys = len(exp_keys)
        matched_keys = len(common_keys) - len(results["type_mismatches"])
        results["structure_match_score"] = (matched_keys / total_keys * 100) if total_keys > 0 else 0
        
        return results
    
    def compare_content(self, generated: Dict, expected: Dict) -> Dict:
        """Compare the content of two JSON objects"""
        results = {
            "content_differences": {},
            "similarity_scores": {},
            "array_length_mismatches": {},
            "overall_similarity": 0.0
        }
        
        # Get common keys
        gen_keys = set(self._get_all_keys(generated))
        exp_keys = set(self._get_all_keys(expected))
        common_keys = gen_keys & exp_keys
        
        similarities = []
        
        for key in common_keys:
            gen_val = self._get_nested_value(generated, key)
            exp_val = self._get_nested_value(expected, key)
            
            if isinstance(gen_val, str) and isinstance(exp_val, str):
                # Compare strings
                similarity = self._calculate_string_similarity(gen_val, exp_val)
                results["similarity_scores"][key] = similarity
                similarities.append(similarity)
                
                if similarity < 0.9:  # Less than 90% similar
                    results["content_differences"][key] = {
                        "generated": gen_val[:100] + "..." if len(gen_val) > 100 else gen_val,
                        "expected": exp_val[:100] + "..." if len(exp_val) > 100 else exp_val,
                        "similarity": similarity
                    }
            
            elif isinstance(gen_val, list) and isinstance(exp_val, list):
                # Compare arrays
                if len(gen_val) != len(exp_val):
                    results["array_length_mismatches"][key] = {
                        "generated_length": len(gen_val),
                        "expected_length": len(exp_val)
                    }
                
                # Compare array contents
                array_similarity = self._compare_arrays(gen_val, exp_val)
                results["similarity_scores"][key] = array_similarity
                similarities.append(array_similarity)
            
            elif isinstance(gen_val, dict) and isinstance(exp_val, dict):
                # Recursively compare nested dictionaries
                nested_similarity = self._compare_dicts(gen_val, exp_val)
                results["similarity_scores"][key] = nested_similarity
                similarities.append(nested_similarity)
            
            elif gen_val != exp_val:
                # Direct comparison for other types
                results["content_differences"][key] = {
                    "generated": str(gen_val),
                    "expected": str(exp_val),
                    "similarity": 0.0
                }
                similarities.append(0.0)
        
        # Calculate overall similarity
        if similarities:
            results["overall_similarity"] = sum(similarities) / len(similarities) * 100
        
        return results
    
    def validate_semantic_content(self, generated: Dict, expected: Dict) -> Dict:
        """Validate semantic content like extracted sections and analysis"""
        results = {
            "extracted_sections_validation": {},
            "llm_response_validation": {},
            "metadata_validation": {},
            "insights_validation": {}
        }
        
        # Print structure for debugging
        print(f"\nDebug - Generated keys: {list(generated.keys())}")
        print(f"Debug - Expected keys: {list(expected.keys())}")
        
        # Validate extracted sections
        if "extracted_sections" in generated and "extracted_sections" in expected:
            gen_sections = generated["extracted_sections"]
            exp_sections = expected["extracted_sections"]
            
            results["extracted_sections_validation"] = {
                "count_match": len(gen_sections) == len(exp_sections),
                "generated_count": len(gen_sections),
                "expected_count": len(exp_sections),
                "document_coverage": self._calculate_document_coverage(gen_sections, exp_sections),
                "section_overlap": self._calculate_section_overlap(gen_sections, exp_sections)
            }
        
        # Validate LLM response - handle different possible structures
        gen_llm = None
        exp_llm = None
        
        # Check different possible locations for LLM content
        if "llm_generated_content" in generated:
            gen_llm = generated.get("llm_generated_content", {}).get("response", "")
        elif "analysis" in generated:
            gen_llm = generated.get("analysis", "")
        elif "response" in generated:
            gen_llm = generated.get("response", "")
        
        if "llm_generated_content" in expected:
            exp_llm = expected.get("llm_generated_content", {}).get("response", "")
        elif "analysis" in expected:
            exp_llm = expected.get("analysis", "")
        elif "response" in expected:
            exp_llm = expected.get("response", "")
        
        if gen_llm and exp_llm:
            similarity = self._calculate_string_similarity(str(gen_llm), str(exp_llm))
            results["llm_response_validation"] = {
                "found": True,
                "length_ratio": len(str(gen_llm)) / len(str(exp_llm)) if exp_llm else 0,
                "content_similarity": similarity,
                "key_terms_match": self._check_key_terms(str(gen_llm), str(exp_llm))
            }
        else:
            results["llm_response_validation"] = {
                "found": False,
                "message": "LLM response not found in expected locations"
            }
        
        # Validate metadata
        if "metadata" in generated and "metadata" in expected:
            gen_meta = generated["metadata"]
            exp_meta = expected["metadata"]
            
            results["metadata_validation"] = {
                "persona_match": gen_meta.get("persona") == exp_meta.get("persona"),
                "task_match": gen_meta.get("job_to_be_done") == exp_meta.get("job_to_be_done"),
                "document_count_match": len(gen_meta.get("input_documents", [])) == len(exp_meta.get("input_documents", []))
            }
        
        return results
    
    def _get_all_keys(self, obj: Any, prefix: str = "") -> List[str]:
        """Get all keys from nested dictionary"""
        keys = []
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                keys.append(new_key)
                if isinstance(v, dict):
                    keys.extend(self._get_all_keys(v, new_key))
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    # For lists of objects, get keys from first item
                    keys.extend(self._get_all_keys(v[0], f"{new_key}[0]"))
        
        return keys
    
    def _get_nested_value(self, obj: Dict, key_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = key_path.split('.')
        value = obj
        
        try:
            for key in keys:
                if '[' in key:
                    # Handle array index
                    key_name, index = key.split('[')
                    index = int(index.rstrip(']'))
                    value = value.get(key_name, [])[index] if key_name else value[index]
                else:
                    value = value.get(key, None)
                    if value is None:
                        break
        except (IndexError, KeyError, TypeError):
            return None
        
        return value
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        # Use sequence matcher for similarity
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    def _compare_arrays(self, arr1: List, arr2: List) -> float:
        """Compare two arrays and return similarity score"""
        if not arr1 and not arr2:
            return 1.0
        if not arr1 or not arr2:
            return 0.0
        
        # For arrays of dictionaries
        if arr1 and isinstance(arr1[0], dict):
            similarities = []
            for i in range(min(len(arr1), len(arr2))):
                sim = self._compare_dicts(arr1[i], arr2[i])
                similarities.append(sim)
            
            # Penalize for length difference
            length_penalty = abs(len(arr1) - len(arr2)) / max(len(arr1), len(arr2))
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            return avg_similarity * (1 - length_penalty * 0.5)
        
        # For simple arrays
        else:
            matches = sum(1 for i in range(min(len(arr1), len(arr2))) if arr1[i] == arr2[i])
            return matches / max(len(arr1), len(arr2))
    
    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> float:
        """Compare two dictionaries and return similarity score"""
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        similarities = []
        
        for key in all_keys:
            if key in dict1 and key in dict2:
                val1, val2 = dict1[key], dict2[key]
                
                if isinstance(val1, str) and isinstance(val2, str):
                    similarities.append(self._calculate_string_similarity(val1, val2))
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # For numbers, check if they're close
                    max_val = max(abs(val1), abs(val2), 1)
                    similarity = 1 - abs(val1 - val2) / max_val
                    similarities.append(similarity)
                elif val1 == val2:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0
    
    def _calculate_document_coverage(self, gen_sections: List[Dict], exp_sections: List[Dict]) -> Dict:
        """Calculate document coverage comparison"""
        gen_docs = set(s.get("document", "") for s in gen_sections)
        exp_docs = set(s.get("document", "") for s in exp_sections)
        
        return {
            "generated_documents": list(gen_docs),
            "expected_documents": list(exp_docs),
            "missing_documents": list(exp_docs - gen_docs),
            "extra_documents": list(gen_docs - exp_docs),
            "coverage_score": len(gen_docs & exp_docs) / len(exp_docs) * 100 if exp_docs else 0
        }
    
    def _calculate_section_overlap(self, gen_sections: List[Dict], exp_sections: List[Dict]) -> float:
        """Calculate overlap between extracted sections"""
        gen_titles = set(s.get("section_title", "") for s in gen_sections)
        exp_titles = set(s.get("section_title", "") for s in exp_sections)
        
        if not exp_titles:
            return 0.0
        
        overlap = len(gen_titles & exp_titles) / len(exp_titles)
        return overlap * 100
    
    def _check_key_terms(self, gen_text: str, exp_text: str) -> Dict:
        """Check if key terms from expected text appear in generated text"""
        # Extract key terms (simple approach - can be enhanced)
        import re
        
        # Extract capitalized words and important terms
        exp_terms = set(re.findall(r'\b[A-Z][a-z]+\b', exp_text))
        exp_terms.update(re.findall(r'\b\d+\s*days?\b', exp_text))
        exp_terms.update(re.findall(r'\b\d+\s*people\b', exp_text))
        
        if not exp_terms:
            return {"score": 100, "details": "No key terms found"}
        
        found_terms = sum(1 for term in exp_terms if term.lower() in gen_text.lower())
        
        return {
            "score": (found_terms / len(exp_terms)) * 100,
            "expected_terms": list(exp_terms),
            "found_terms": found_terms,
            "total_terms": len(exp_terms)
        }
    
    def validate_collection(self, collection_name: str) -> Dict:
        """Validate a single collection"""
        # Paths
        generated_path = Path(f"outputs/{collection_name}/output.json")
        expected_path = Path(f"collections/{collection_name}/challenge1b_output.json")
        
        # Check if files exist
        if not generated_path.exists():
            return {
                "status": "error",
                "message": f"Generated output not found: {generated_path}"
            }
        
        if not expected_path.exists():
            return {
                "status": "error",
                "message": f"Expected output not found: {expected_path}"
            }
        
        # Load JSONs
        generated = self.load_json(generated_path)
        expected = self.load_json(expected_path)
        
        if not generated or not expected:
            return {
                "status": "error",
                "message": "Failed to load JSON files"
            }
        
        # Perform validations
        structure_results = self.compare_structures(generated, expected)
        content_results = self.compare_content(generated, expected)
        semantic_results = self.validate_semantic_content(generated, expected)
        
        # Calculate overall score
        scores = [
            structure_results["structure_match_score"],
            content_results["overall_similarity"]
        ]
        
        # Add semantic scores if available
        if semantic_results.get("extracted_sections_validation"):
            scores.append(semantic_results["extracted_sections_validation"].get("coverage_score", 0))
        
        if semantic_results.get("llm_response_validation", {}).get("found"):
            scores.append(semantic_results["llm_response_validation"].get("content_similarity", 0) * 100)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "status": "success",
            "overall_score": overall_score,
            "structure_validation": structure_results,
            "content_validation": content_results,
            "semantic_validation": semantic_results,
            "grade": self._get_grade(overall_score)
        }
    
    def _get_grade(self, score: float) -> str:
        """Get letter grade based on score"""
        if score >= 90:
            return "A - Excellent"
        elif score >= 80:
            return "B - Good"
        elif score >= 70:
            return "C - Satisfactory"
        elif score >= 60:
            return "D - Needs Improvement"
        else:
            return "F - Poor"
    
    def generate_report(self, output_file: str = "validation_report.md"):
        """Generate a comprehensive validation report"""
        report_lines = []
        report_lines.append("# Output Validation Report")
        report_lines.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary table
        report_lines.append("## Summary\n")
        report_lines.append("| Collection | Overall Score | Grade | Status |")
        report_lines.append("|------------|---------------|-------|---------|")
        
        for collection, result in self.results.items():
            if result["status"] == "success":
                score = f"{result['overall_score']:.1f}%"
                grade = result['grade']
                status = "✅ Validated"
            else:
                score = "N/A"
                grade = "N/A"
                status = "❌ Error"
            
            report_lines.append(f"| {collection} | {score} | {grade} | {status} |")
        
        # Detailed results for each collection
        for collection, result in self.results.items():
            report_lines.append(f"\n## {collection} Details\n")
            
            if result["status"] == "error":
                report_lines.append(f"**Error**: {result['message']}\n")
                continue
            
            # Structure validation
            report_lines.append("### Structure Validation")
            struct_val = result["structure_validation"]
            report_lines.append(f"- Structure Match Score: {struct_val['structure_match_score']:.1f}%")
            
            if struct_val["missing_keys"]:
                report_lines.append(f"- Missing Keys ({len(struct_val['missing_keys'])}):")
                for key in struct_val["missing_keys"][:5]:  # Show first 5
                    report_lines.append(f"  - `{key}`")
            
            if struct_val["extra_keys"]:
                report_lines.append(f"- Extra Keys ({len(struct_val['extra_keys'])}):")
                for key in struct_val["extra_keys"][:5]:
                    report_lines.append(f"  - `{key}`")
            
            # Content validation
            report_lines.append("\n### Content Validation")
            content_val = result["content_validation"]
            report_lines.append(f"- Overall Content Similarity: {content_val['overall_similarity']:.1f}%")
            
            if content_val["content_differences"]:
                report_lines.append(f"- Major Content Differences ({len(content_val['content_differences'])}):")
                for key, diff in list(content_val["content_differences"].items())[:3]:
                    report_lines.append(f"  - `{key}`: {diff['similarity']*100:.1f}% similar")
            
            # Semantic validation
            report_lines.append("\n### Semantic Validation")
            semantic_val = result["semantic_validation"]
            
            if "extracted_sections_validation" in semantic_val and semantic_val["extracted_sections_validation"]:
                sections_val = semantic_val["extracted_sections_validation"]
                report_lines.append("- **Extracted Sections**:")
                report_lines.append(f"  - Generated: {sections_val['generated_count']} sections")
                report_lines.append(f"  - Expected: {sections_val['expected_count']} sections")
                report_lines.append(f"  - Document Coverage: {sections_val['document_coverage']['coverage_score']:.1f}%")
                report_lines.append(f"  - Section Overlap: {sections_val['section_overlap']:.1f}%")
            
            if "llm_response_validation" in semantic_val and semantic_val["llm_response_validation"]:
                llm_val = semantic_val["llm_response_validation"]
                report_lines.append("- **LLM Response**:")
                if llm_val.get("found"):
                    report_lines.append(f"  - Content Similarity: {llm_val.get('content_similarity', 0)*100:.1f}%")
                    report_lines.append(f"  - Length Ratio: {llm_val.get('length_ratio', 0):.2f}")
                    if "key_terms_match" in llm_val:
                        report_lines.append(f"  - Key Terms Match: {llm_val['key_terms_match']['score']:.1f}%")
                else:
                    report_lines.append(f"  - {llm_val.get('message', 'Not found')}")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📊 Validation report saved to: {output_file}")
        
        # Also save as JSON
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📋 Detailed JSON results saved to: {json_file}")
    
    def print_summary(self):
        """Print a summary to console"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        total_score = 0
        valid_collections = 0
        
        for collection, result in self.results.items():
            if result["status"] == "success":
                score = result["overall_score"]
                total_score += score
                valid_collections += 1
                
                print(f"\n{collection}:")
                print(f"  Overall Score: {score:.1f}%")
                print(f"  Grade: {result['grade']}")
                print(f"  Structure Match: {result['structure_validation']['structure_match_score']:.1f}%")
                print(f"  Content Similarity: {result['content_validation']['overall_similarity']:.1f}%")
            else:
                print(f"\n{collection}: ❌ {result['message']}")
        
        if valid_collections > 0:
            avg_score = total_score / valid_collections
            print(f"\n{'='*60}")
            print(f"AVERAGE SCORE: {avg_score:.1f}%")
            print(f"OVERALL GRADE: {self._get_grade(avg_score)}")
        
        print("="*60)


def main():
    """Main validation function"""
    validator = OutputValidator()
    
    # Validate all collections
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    print("🔍 Starting output validation...\n")
    
    for collection in collections:
        print(f"Validating {collection}...")
        result = validator.validate_collection(collection)
        validator.results[collection] = result
    
    # Generate reports
    validator.print_summary()
    validator.generate_report()
    
    # Generate detailed diff files
    print("\n📄 Generating detailed diff files...")
    
    for collection in collections:
        if validator.results[collection]["status"] == "success":
            # Create detailed diff
            generated_path = Path(f"outputs/{collection}/output.json")
            expected_path = Path(f"collections/{collection}/challenge1b_output.json")
            
            if generated_path.exists() and expected_path.exists():
                generated = validator.load_json(generated_path)
                expected = validator.load_json(expected_path)
                
                # Save pretty-printed versions for manual comparison
                with open(f"outputs/{collection}/generated_pretty.json", 'w') as f:
                    json.dump(generated, f, indent=2)
                
                with open(f"outputs/{collection}/expected_pretty.json", 'w') as f:
                    json.dump(expected, f, indent=2)
                
                print(f"  ✅ Created diff files for {collection}")


if __name__ == "__main__":
    main()