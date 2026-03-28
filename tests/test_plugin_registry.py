from __future__ import annotations

from pathlib import Path

from ai_factory.core.config.schema import OrchestrationConfig
from ai_factory.core.execution.base import CommandSpec
from ai_factory.core.execution.commands import build_command
from ai_factory.core.instances.models import EnvironmentSpec, InstanceManifest
from ai_factory.core.plugins.registry import build_default_plugin_registry


def test_plugin_registry_can_load_external_modules(tmp_path: Path, monkeypatch):
    plugin_path = tmp_path / "ai_factory_test_plugin.py"
    plugin_path.write_text(
        "\n".join(
            [
                "from ai_factory.core.execution.base import CommandSpec",
                "from ai_factory.core.plugins.base import PluginDescriptor",
                "",
                "class CustomReportHandler:",
                "    descriptor = PluginDescriptor(",
                "        kind='instance_handler',",
                "        name='external.custom_report',",
                "        label='Custom report handler',",
                "        version='test',",
                "        capabilities=['report'],",
                "    )",
                "    instance_types = ('report',)",
                "",
                "    def build_command(self, config, manifest, registry):",
                "        return CommandSpec(argv=['echo', 'custom-report'])",
                "",
                "def register_plugins(registry):",
                "    registry.register_instance_handler(CustomReportHandler())",
                "",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = build_default_plugin_registry(("ai_factory_test_plugin",))
    config = OrchestrationConfig.model_validate(
        {
            "instance": {"type": "report", "environment": {"kind": "local"}},
            "subsystem": {"source_artifact_ref": "evaluation/results/failure_analysis.json"},
        }
    )
    manifest = InstanceManifest(
        id="report-001",
        type="report",
        name="report",
        environment=EnvironmentSpec(kind="local"),
    )

    command = build_command(config, manifest, plugin_registry=registry)

    assert isinstance(command, CommandSpec)
    assert command.argv == ["echo", "custom-report"]
    assert any(plugin.name == "external.custom_report" for plugin in registry.list_plugins())
