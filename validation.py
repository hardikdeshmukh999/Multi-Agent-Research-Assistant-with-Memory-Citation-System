# validation.py (NEW FILE)
import requests
from typing import List, Dict
import re

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
    # Extract DOI links
    doi_pattern = r'https?://doi\.org/[^\s\)"\]]+'
    dois = re.findall(doi_pattern, report_text)
    
    broken_links = []
    for doi in dois:
        try:
            response = requests.head(doi, timeout=5, allow_redirects=True)
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
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check 1: Citation validation
    citation_check = validate_doi_links(content)
    if not citation_check['validation_passed']:
        issues.append(f"Broken DOI links: {citation_check['broken_links']}")
    
    # Check 2: Placeholder detection
    placeholders = check_placeholders(content)
    if placeholders:
        issues.append(f"Placeholder text found: {placeholders}")
    
    # Check 3: Length check
    word_count = len(content.split())
    if word_count < 2000:
        issues.append(f"Report too short: {word_count} words (target: 4000-6000)")
    elif word_count > 8000:
        issues.append(f"Report too long: {word_count} words (target: 4000-6000)")
    
    # Check 4: Required sections
    required_sections = [
        "# Executive Summary",
        "# Key Findings",
        "# References"
    ]
    missing_sections = [s for s in required_sections if s not in content]
    if missing_sections:
        issues.append(f"Missing sections: {missing_sections}")
    
    return {
        "validation_passed": len(issues) == 0,
        "issues": issues,
        "citation_stats": citation_check,
        "word_count": word_count
    }