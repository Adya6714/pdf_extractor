import re
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
import statistics

class PDFOutlineExtractor:
    def __init__(self):
        self.heading_patterns = {
            'numbered': [
                (r'^\d+\.?\s+[A-Z]', 1),
                (r'^\d+\.\d+\.?\s+', 2),
                (r'^\d+\.\d+\.\d+\.?\s+', 3),
            ],
            'keywords': {
                1: ['chapter', 'section', 'part'],
                2: ['subsection', 'topic'],
                3: ['sub-topic', 'point']
            }
        }

    def extract_text_with_formatting(self, page):
        blocks = []
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    line_text = ""
                    line_info = {"bbox": line["bbox"], "spans": []}
                    for span in line["spans"]:
                        line_text += span["text"]
                        line_info["spans"].append({
                            "text": span["text"],
                            "font": span["font"],
                            "size": round(span["size"], 1),
                            "flags": span["flags"]
                        })
                    if line_text.strip():
                        blocks.append({
                            "text": line_text.strip(),
                            "info": line_info,
                            "page": page.number + 1
                        })
        return blocks

    def calculate_font_statistics(self, blocks):
        font_sizes = [span["size"] for block in blocks for span in block["info"]["spans"]]
        if not font_sizes:
            return {}
        return {
            "mean": statistics.mean(font_sizes),
            "median": statistics.median(font_sizes),
            "mode": statistics.mode(font_sizes) if len(set(font_sizes)) < len(font_sizes) else statistics.median(font_sizes),
            "max": max(font_sizes),
            "min": min(font_sizes)
        }

    def is_potential_heading(self, block, font_stats):
        text = block["text"]
        if len(text) > 150:
            return False, 0
        max_font_size = 0
        is_bold = False
        is_uppercase = False
        for span in block["info"]["spans"]:
            max_font_size = max(max_font_size, span["size"])
            if span["flags"] & 2**4:
                is_bold = True
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7:
            is_uppercase = True
        score = 0
        if font_stats:
            if max_font_size > font_stats["mean"] * 1.2:
                score += 3
            elif max_font_size > font_stats["mean"]:
                score += 1
        if is_bold:
            score += 2
        if is_uppercase and len(text) < 50:
            score += 2
        for pattern, level in self.heading_patterns['numbered']:
            if re.match(pattern, text):
                score += 3
                return True, level
        if len(text) < 60:
            score += 1
        if text and text[-1] not in '.!?,;':
            score += 1
        return score >= 3, 0

    def classify_heading_level(self, block, all_headings, font_stats):
        text = block["text"]
        for pattern, level in self.heading_patterns['numbered']:
            if re.match(pattern, text):
                return level
        max_font_size = max(span["size"] for span in block["info"]["spans"])
        heading_sizes = [max(span["size"] for span in h["info"]["spans"]) for h in all_headings]
        if not heading_sizes:
            return 1
        unique_sizes = sorted(set(heading_sizes), reverse=True)
        if len(unique_sizes) == 1:
            return 1
        elif len(unique_sizes) == 2:
            return 1 if max_font_size == unique_sizes[0] else 2
        else:
            if max_font_size in unique_sizes[:len(unique_sizes)//3]:
                return 1
            elif max_font_size in unique_sizes[len(unique_sizes)//3:2*len(unique_sizes)//3]:
                return 2
            else:
                return 3

    def extract_title(self, blocks, doc_info):
        if doc_info.get("title"):
            return doc_info["title"]
        candidates = []
        for i, block in enumerate(blocks[:10]):
            if len(block["text"]) < 150:
                max_size = max(span["size"] for span in block["info"]["spans"])
                candidates.append((max_size, i, block["text"]))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][2]
        return "Untitled Document"

    def process_pdf(self, pdf_path: Path) -> Dict:
        try:
            doc = fitz.open(pdf_path)
            all_blocks = []
            for page in doc:
                all_blocks.extend(self.extract_text_with_formatting(page))
            font_stats = self.calculate_font_statistics(all_blocks)
            potential_headings = []
            for block in all_blocks:
                is_heading, numbered_level = self.is_potential_heading(block, font_stats)
                if is_heading:
                    block["numbered_level"] = numbered_level
                    potential_headings.append(block)
            outline = []
            for heading in potential_headings:
                level = heading["numbered_level"] if heading["numbered_level"] > 0 else self.classify_heading_level(heading, potential_headings, font_stats)
                outline.append({
                    "level": f"H{level}",
                    "text": heading["text"],
                    "page": heading["page"]
                })
            title = self.extract_title(all_blocks, doc.metadata)
            doc.close()
            return {"title": title, "outline": outline}
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Error Processing Document", "outline": []}