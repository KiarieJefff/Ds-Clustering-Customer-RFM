"""
Convert Segmentation Report to PDF
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def create_pdf_report(text_file_path, pdf_output_path):
    """
    Convert text report to formatted PDF
    
    Args:
        text_file_path (str): Path to the text report file
        pdf_output_path (str): Path for the output PDF file
    """
    # Read the text file
    with open(text_file_path, 'r') as f:
        content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Process content
    lines = content.split('\n')
    current_section = None
    table_data = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Title
        if 'CUSTOMER SEGMENTATION REPORT' in line:
            story.append(Paragraph(line, title_style))
            story.append(Spacer(1, 20))
            
        # Section headers
        elif line in ['SEGMENT SUMMARY', 'KEY INSIGHTS']:
            current_section = line
            story.append(Paragraph(line, heading_style))
            story.append(Spacer(1, 12))
            
        # Separator lines
        elif line.startswith('---') or line.startswith('==='):
            continue
            
        # Table data
        elif current_section == 'SEGMENT SUMMARY' and ('Count' in line or 'At-Risk' in line or 'Recent' in line or 'Regular' in line):
            if 'Count' in line:  # Header row
                headers = [col.strip() for col in line.split() if col.strip()]
                table_data.append(headers)
            else:  # Data rows
                parts = line.split()
                segment_name = ' '.join(parts[:3])  # Handle multi-word segment names
                values = parts[3:]
                row = [segment_name] + values
                table_data.append(row)
                
        # Regular text content
        else:
            if line and not line.startswith('Segment'):
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
    
    # Add table if we have data
    if table_data:
        # Create table
        table = Table(table_data)
        
        # Style the table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report saved to: {pdf_output_path}")

if __name__ == "__main__":
    # Paths
    text_report = "../reports/segmentation_report.txt"
    pdf_report = "../reports/segmentation_report.pdf"
    
    # Create PDF
    create_pdf_report(text_report, pdf_report)
