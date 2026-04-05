import importlib
import sys


def _clear_modules(*names: str) -> None:
    prefixes = tuple(f"{name}." for name in names)
    for module_name in list(sys.modules):
        if module_name in names or module_name.startswith(prefixes):
            sys.modules.pop(module_name, None)


def _assert_module_stays_lightweight(target: str, *heavy_modules: str) -> None:
    previous = {name: sys.modules.get(name) for name in heavy_modules}
    _clear_modules(target)
    importlib.import_module(target)
    for name, module in previous.items():
        if module is None:
            assert name not in sys.modules
        else:
            assert sys.modules.get(name) is module


def test_generation_parameters_import_is_lightweight() -> None:
    _clear_modules("inference.app.parameters", "inference.app.prompts")

    importlib.import_module("inference.app.parameters")

    assert "inference.app.prompts" not in sys.modules


def test_dependencies_import_is_lightweight() -> None:
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


def test_orchestration_service_import_is_lightweight() -> None:
    _assert_module_stays_lightweight("ai_factory.core.orchestration.service", "torch", "transformers")


def test_tui_import_is_lightweight() -> None:
    _assert_module_stays_lightweight("ai_factory.tui", "torch", "transformers")
