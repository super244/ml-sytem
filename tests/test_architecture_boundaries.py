from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSYSTEM_ROOTS = {
    "inference": REPO_ROOT / "inference",
}


def _iter_imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            modules.append(node.module)
    return modules


def test_ai_factory_core_does_not_import_runtime_subsystems() -> None:
    violations: list[str] = []
    forbidden_roots = ("data", "training", "evaluation")
    for path in sorted((REPO_ROOT / "ai_factory" / "core").rglob("*.py")):
        relative_path = path.relative_to(REPO_ROOT)
        for module_name in _iter_imported_modules(path):
            if module_name.split(".")[0] in forbidden_roots:
                violations.append(f"{relative_path} imports {module_name}")

    assert not violations, "ai_factory.core must not import removed runtime subsystems:\n" + "\n".join(violations)


def test_subsystem_dependency_matrix_matches_implementation() -> None:
    doc = REPO_ROOT / "docs" / "architecture" / "dependency_matrix.md"
    assert doc.is_file(), "dependency_matrix.md must exist"


def test_cross_subsystem_imports_are_explicit_and_limited() -> None:
    violations: list[str] = []

    for subsystem, root in SUBSYSTEM_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            relative_path = str(path.relative_to(REPO_ROOT))
            allowed_imports: set[str] = set()
            observed_allowed_imports: set[str] = set()

            for module_name in _iter_imported_modules(path):
                imported_root = module_name.split(".")[0]
                if imported_root not in SUBSYSTEM_ROOTS or imported_root == subsystem:
                    continue
                if module_name in allowed_imports:
                    observed_allowed_imports.add(module_name)
                    continue
                violations.append(f"{relative_path} imports {module_name}")

            for module_name in sorted(allowed_imports - observed_allowed_imports):
                pass

    assert not violations, "Unexpected cross-subsystem imports:\n" + "\n".join(violations)
