"""
Test Flask web display.

Smoke tests để kiểm tra:
- web.py module imports correctly
- Templates exist and have correct structure
- HTML includes all required elements
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_web_module_structure():
    """Kiểm tra web.py có tồn tại và có create_app()."""
    from src.display import web

    assert hasattr(web, "create_app"), "Missing create_app function"
    assert hasattr(web, "run_server"), "Missing run_server function"
    print("[OK] web.py module structure correct")


def test_template_file_exists():
    """Kiểm tra template file tồn tại."""
    template_path = Path(__file__).parent.parent / "src" / "display" / "templates" / "index.html"
    assert template_path.exists(), f"Template not found: {template_path}"
    print("[OK] Template file exists")


def test_template_structure():
    """Kiểm tra template HTML structure."""
    template_path = Path(__file__).parent.parent / "src" / "display" / "templates" / "index.html"

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Check for key elements
    checks = {
        "Title": "VinFast Evo 200",
        "Battery icons": "battery-bms",
        "Coulomb counter": "battery-cc",
        "Model SoC": "battery-model",
        "Range stat": "stat-range",
        "SoH stat": "stat-soh",
        "Energy consumption": "stat-wh",
        "API endpoint": "api_endpoint",
        "AJAX polling": "setInterval",
        "Status indicator": "status-indicator",
    }

    for check_name, check_str in checks.items():
        assert check_str in html, f"Missing: {check_name}"

    print("[OK] Template structure complete")
    print(f"    - Title, 3 battery icons, stats section")
    print(f"    - AJAX polling, error handling, responsive design")


def test_html_validity():
    """Kiểm tra HTML cơ bản (DOCTYPE, tags)."""
    template_path = Path(__file__).parent.parent / "src" / "display" / "templates" / "index.html"

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Basic HTML structure checks
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html
    assert "<head>" in html
    assert "<body>" in html
    assert "</body>" in html

    print("[OK] HTML validity checks passed")


def test_css_styling():
    """Kiểm tra CSS được include."""
    template_path = Path(__file__).parent.parent / "src" / "display" / "templates" / "index.html"

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Check for CSS
    assert "<style>" in html, "Missing <style> tag"
    assert ".battery-icon" in html, "Missing battery CSS"
    assert ".dashboard" in html, "Missing dashboard grid CSS"
    assert "@media" in html, "Missing responsive design"

    print("[OK] CSS styling present and responsive")


def test_javascript_logic():
    """Kiểm tra JavaScript logic."""
    template_path = Path(__file__).parent.parent / "src" / "display" / "templates" / "index.html"

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Check for JavaScript
    checks = [
        ("Fetch API", "fetch(API_ENDPOINT)"),
        ("Update function", "updateDashboard"),
        ("Battery fill update", "battery-bms"),
        ("Error handling", "showError"),
        ("Polling interval", "setInterval"),
    ]

    for check_name, check_str in checks:
        assert check_str in html, f"Missing JS: {check_name}"

    print("[OK] JavaScript logic present")


if __name__ == "__main__":
    print("Testing Flask display module...")
    print()

    test_web_module_structure()
    test_template_file_exists()
    test_template_structure()
    test_html_validity()
    test_css_styling()
    test_javascript_logic()

    print()
    print("All smoke tests passed!")
