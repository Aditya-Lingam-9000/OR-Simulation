"""Smoke test to verify project structure and imports."""


def test_project_structure_exists():
    """Verify the project root directories exist."""
    from pathlib import Path

    root = Path(__file__).parent.parent
    required_dirs = ["src", "tests", "configs", "schemas", "logs", "data", "reports"]
    for d in required_dirs:
        assert (root / d).exists(), f"Directory '{d}' missing from project root"


def test_src_packages_exist():
    """Verify all src sub-packages have __init__.py."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "src"
    packages = ["ingest", "asr", "state", "llm", "api", "workers", "utils"]
    for pkg in packages:
        init_file = src / pkg / "__init__.py"
        assert init_file.exists(), f"Package 'src/{pkg}' missing __init__.py"
