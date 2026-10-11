"""
Document loader utility for ingesting files into ChromaDB.
Supports loading text, markdown, and PDF files from a documents directory.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def load_text_file(filepath: str) -> str:
    """Load a text or markdown file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(filepath: str) -> str:
    """Load a PDF file (requires PyPDF2 or falls back to basic reading)."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("PyPDF2 not installed — skipping PDF: %s", filepath)
        return ""
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", filepath, e)
        return ""


def load_documents_into_chroma(
    documents_dir: str = "./documents",
    chroma_service=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Load all documents from a directory into ChromaDB.

    Args:
        documents_dir: Path to directory containing documents
        chroma_service: ChromaSearchService instance
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    if chroma_service is None:
        from services.chroma_search_service import ChromaSearchService
        chroma_service = ChromaSearchService()

    docs_path = Path(documents_dir)
    if not docs_path.exists():
        logger.warning(f"Documents directory not found: {documents_dir}")
        return

    supported = {".txt", ".md", ".pdf", ".text"}
    files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported documents found in {documents_dir}")
        return

    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_idx = 0

    for filepath in files:
        logger.info(f"Processing: {filepath.name}")

        if filepath.suffix.lower() == ".pdf":
            text = load_pdf_file(str(filepath))
        else:
            text = load_text_file(str(filepath))

        if not text.strip():
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "name": filepath.name,
                "document_name": filepath.name,
                "source": str(filepath),
                "page_number": i + 1,
                "path": str(filepath.parent),
                "document_type": filepath.suffix.lstrip("."),
            })
            all_ids.append(f"doc_{doc_idx}_{i}")

        doc_idx += 1

    if all_chunks:
        chroma_service.add_documents(all_chunks, all_metadatas, all_ids)
        logger.info(f"Loaded {len(all_chunks)} chunks from {doc_idx} documents")
    else:
        logger.warning("No text content extracted from documents")


def create_sample_documents(documents_dir: str = "./documents"):
    """Create sample healthcare documents for testing."""
    docs_path = Path(documents_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    samples = {
        "philips_mx800_overview.md": """# Philips IntelliVue MX800 Patient Monitor

## Overview
The Philips IntelliVue MX800 is a high-acuity patient monitoring solution designed for critical care environments. It combines patient monitoring with clinical informatics at the bedside.

## Key Features
- 19-inch widescreen display with multi-touch capability
- Integrated PC for clinical applications
- Full parameter monitoring including ECG, SpO2, NBP, IBP, CO2, temperature
- Smart alarm management to reduce alarm fatigue
- Connectivity with Philips IntelliSpace Critical Care and Anesthesia (ICCA)

## Clinical Benefits
- Decision support tools for early detection of patient deterioration
- Customizable display layouts for different care areas
- Seamless data flow to EMR systems
- Remote monitoring capabilities via IntelliVue Information Center

## Technical Specifications
- Display: 19" TFT LCD, 1920x1080 resolution
- Weight: approximately 10.5 kg (monitor only)
- Battery backup: up to 30 minutes
- Network: Wired and wireless connectivity
- Mounting: VESA compatible

## Target Market
- ICU, CCU, OR, and ED environments
- Hospitals seeking integrated monitoring and clinical computing
- Facilities upgrading from older Philips monitoring platforms
""",
        "mindray_competitor_analysis.md": """# Mindray Patient Monitoring — Competitive Analysis

## Company Overview
Mindray Medical International is a global medical device manufacturer headquartered in Shenzhen, China. It is a significant competitor in the patient monitoring market.

## Key Product Lines
### BeneVision N-Series
- N22/N19: High-acuity monitors for ICU/OR
- N17/N15: Mid-acuity monitors
- N12/N1: Transport and spot-check monitors

### BeneVision Distributed Monitoring
- Central monitoring station (BeneVision CMS)
- TeleBed system for step-down units

## Competitive Strengths
- Aggressive pricing: 20-40% lower than Philips equivalent products
- Growing installed base in emerging markets (Asia, Africa, South America)
- Rapid product development cycle
- Strong government hospital contracts in China

## Competitive Weaknesses
- Limited clinical decision support compared to Philips
- Smaller ecosystem of connected devices
- Lower brand recognition in North American/European markets
- Fewer FDA 510(k) clearances vs established competitors

## Market Position
- #3 globally in patient monitoring (after Philips and GE Healthcare)
- Dominant position in China market (~40% share)
- Expanding rapidly in Southeast Asia, Middle East, and Latin America

## Recent Developments (2024-2025)
- Launched AI-powered arrhythmia detection
- Expanded US sales force
- New partnerships with regional distributors in Europe
""",
        "ge_healthcare_comparison.md": """# GE Healthcare Patient Monitoring Portfolio

## Overview
GE Healthcare (now GE HealthCare Technologies) is a major competitor in patient monitoring, particularly strong in North America and Europe.

## Key Products
### CARESCAPE Monitors
- B850: Flagship high-acuity monitor
- B650: Mid-range critical care
- B450: General ward monitoring
- B125: Basic vital signs

### CARESCAPE Central Station
- Centralized alarm management
- Trend analysis and reporting

## Comparison vs Philips IntelliVue
| Feature | GE CARESCAPE B850 | Philips MX800 |
|---------|-------------------|---------------|
| Display | 15.6" | 19" |
| Touch | Yes | Multi-touch |
| PC Integration | Optional | Built-in |
| Alarm Management | Basic | Smart Alarms |
| Price Point | $$$ | $$$$ |

## Market Position
- #2 globally in patient monitoring
- Strong in North America (especially US hospital networks)
- Established relationships with large health systems
- Focus on enterprise-wide monitoring solutions

## Strategic Considerations
- GE HealthCare became independent company in 2024
- Increased R&D investment in AI and predictive analytics
- Growing telehealth and remote monitoring capabilities
- Partnership strategy with EMR vendors
""",
    }

    for filename, content in samples.items():
        filepath = docs_path / filename
        if not filepath.exists():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Created sample document: {filename}")

    logger.info(f"Sample documents ready in {documents_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_sample_documents()
    load_documents_into_chroma()
    print("Documents loaded successfully!")
