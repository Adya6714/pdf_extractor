# src/utils/pdf_generator.py
"""
Enhanced PDF generator with better formatting - FIXED VERSION
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import logging

logger = logging.getLogger(__name__)

class PDFGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom styles for the PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a472a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=20,
            spaceAfter=12
        ))
        
        # Info box style
        self.styles.add(ParagraphStyle(
            name='InfoBox',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            backColor=colors.HexColor('#f7fafc')
        ))
    
    def generate_report(self, result, llm_response: str, persona, title: str, output_path: Path):
        """Generate a professional PDF report"""
        # Ensure output_path is a Path object
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        # Create the document - THIS IS THE FIX
        doc = SimpleDocTemplate(
            str(output_path),  # Convert Path to string for reportlab
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title page
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Paragraph(f"Generated for: {persona.role}", self.styles['Subtitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Task description
        story.append(Paragraph("Task Description", self.styles['SectionHeader']))
        story.append(Paragraph(persona.task, self.styles['InfoBox']))
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata table
        metadata_data = [
            ['Processing Date', result.metadata['processing_timestamp'][:10]],
            ['Documents Analyzed', str(len(set(s['document'] for s in result.extracted_sections)))],
            ['Sections Extracted', str(len(result.extracted_sections))],
            ['Processing Time', f"{result.metadata['processing_time_seconds']:.1f} seconds"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f2ff')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(metadata_table)
        story.append(PageBreak())
        
        # Generated content
        story.append(Paragraph("Generated Solution", self.styles['SectionHeader']))
        
        # Process LLM response
        self._add_formatted_content(story, llm_response)
        
        story.append(PageBreak())
        
        # Supporting information
        story.append(Paragraph("Supporting Information", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key insights
        if result.insights:
            story.append(Paragraph("Key Insights", self.styles['Heading3']))
            for insight in result.insights:
                story.append(Paragraph(f"• {insight}", self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Document coverage
        story.append(Paragraph("Document Analysis", self.styles['Heading3']))
        
        # Create document summary table
        doc_summary = {}
        for section in result.extracted_sections:
            doc_name = section['document']
            if doc_name not in doc_summary:
                doc_summary[doc_name] = {'count': 0, 'avg_score': 0}
            doc_summary[doc_name]['count'] += 1
            doc_summary[doc_name]['avg_score'] += section['relevance_score']
        
        doc_data = [['Document', 'Sections Used', 'Avg. Relevance']]
        for doc_name, info in doc_summary.items():
            avg_score = info['avg_score'] / info['count']
            # Truncate long document names
            display_name = doc_name[:40] + '...' if len(doc_name) > 40 else doc_name
            doc_data.append([
                display_name,
                str(info['count']),
                f"{avg_score:.2f}"
            ])
        
        doc_table = Table(doc_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        doc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        
        story.append(doc_table)
        
        # Build PDF - THIS IS WHERE THE ERROR WAS
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
    
    def _add_formatted_content(self, story, content: str):
        """Add formatted content to the story"""
        if not content:
            story.append(Paragraph("No content generated.", self.styles['Normal']))
            return
            
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:
                story.append(Spacer(1, 0.1*inch))
                continue
            
            # Detect headers (lines ending with :)
            if line.endswith(':') and len(line) < 50:
                story.append(Paragraph(line, self.styles['Heading3']))
            # Detect numbered items
            elif len(line) > 2 and line[:2].strip() in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
                story.append(Paragraph(line, self.styles['Normal']))
            # Detect bullet points
            elif line.startswith(('•', '-', '*')):
                story.append(Paragraph(line, self.styles['Normal']))
            # Regular text
            else:
                story.append(Paragraph(line, self.styles['Normal']))