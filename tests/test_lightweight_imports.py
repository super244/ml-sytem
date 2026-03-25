import importlib
import sys


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_generation_parameters_import_is_lightweight():
    _clear_modules("inference.app.parameters", "inference.app.prompts")

    importlib.import_module("inference.app.parameters")

    assert "inference.app.prompts" not in sys.modules


def test_dependencies_import_is_lightweight():
    _clear_modules(
        "inference.app.dependencies",
        "inference.app.generation",
        "inference.app.model_catalog",
        "inference.app.model_loader",
        "inference.app.prompts",
    )

    importlib.import_module("inference.app.dependencies")

    assert "inference.app.generation" not in sys.modules
    assert "inference.app.model_catalog" not in sys.modules
    assert "inference.app.model_loader" not in sys.modules
    assert "inference.app.prompts" not in sys.modules
