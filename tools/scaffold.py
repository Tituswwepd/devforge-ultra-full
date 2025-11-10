from pathlib import Path

def scaffold_project(kind: str, name: str, dest: Path, templates_dir: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(f"# {name}\n\nScaffolded kind: {kind}\n")
    return dest
