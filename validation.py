import requests
from typing import List, Dict
import re
import os

def validate_doi_links(report_text: str) -> Dict:
    """
    Extract all DOI links from report and validate they work.
    Returns: {
        "total_citations": N,
        "valid_links": M,
        "broken_links": [...],
        "validation_passed": bool
    }
    """
    # Extract DOI links (standard format: https://doi.org/...)
    doi_pattern = r'https?://doi\.org/[^\s\)"\]]+'
    dois = re.findall(doi_pattern, report_text)
    
    broken_links = []
    # Use a session for faster connection pooling
    session = requests.Session()
    
    for doi in dois:
        try:
            # We use HEAD request to check if link exists without downloading the page
            response = session.head(doi, timeout=5, allow_redirects=True)
            if response.status_code >= 400:
                broken_links.append(doi)
        except:
            broken_links.append(doi)
    
    return {
        "total_citations": len(dois),
        "valid_links": len(dois) - len(broken_links),
        "broken_links": broken_links,
        "validation_passed": len(broken_links) == 0
    }

def check_placeholders(text: str) -> List[str]:
    """Detect placeholder text that shouldn't be in final report."""
    placeholders = [
        r'\[Topic\]',
        r'\[Author\]',
        r'\[Year\]',
        r'\[Dataset\]',
        r'TODO',
        r'TBD',
        r'XXX',
        r'\?\?\?'
    ]
    
    found = []
    for pattern in placeholders:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found.extend(matches)
    
    return found

def validate_report(report_path: str) -> Dict:
    """
    Full validation suite for research report.
    Reads the report file and runs all checks.
    """
    if not os.path.exists(report_path):
        return {"validation_passed": False, "issues": ["Report file not found."]}

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check 1: Citation validation
    citation_check = validate_doi_links(content)
    if not citation_check['validation_passed']:
        issues.append(f"Broken DOI links found: {citation_check['broken_links']}")
    
    # Check 2: Placeholder detection
    placeholders = check_placeholders(content)
    if placeholders:
        issues.append(f"Placeholder text found: {placeholders}")
    
    # Check 3: Length check (Basic heuristic)
    word_count = len(content.split())
    if word_count < 100:
        issues.append(f"Report suspiciously short: {word_count} words")
    
    # Check 4: Required sections
    required_sections = [
        "# Executive Summary",
        "# References"
    ]
    # Simple check if specific headers exist (case-insensitive partial match)
    missing_sections = []
    lower_content = content.lower()
    if "executive summary" not in lower_content and "tl;dr" not in lower_content:
        missing_sections.append("Executive Summary")
    if "references" not in lower_content and "verified references" not in lower_content:
        missing_sections.append("References")

    if missing_sections:
        issues.append(f"Missing sections: {missing_sections}")
    
    return {
        "validation_passed": len(issues) == 0,
        "issues": issues,
        "citation_stats": citation_check,
        "word_count": word_count
    }