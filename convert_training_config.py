import re

with open("training/src/config.py") as f:
    content = f.read()

content = content.replace(
    "from dataclasses import asdict, dataclass, field", "from pydantic import BaseModel, ConfigDict, Field"
)
content = content.replace("@dataclass\nclass", "class")

# Add model_config to all classes that inherit from nothing
content = re.sub(
    r"class ([A-Z][a-zA-Z0-9_]*):", r"class \1(BaseModel):\n    model_config = ConfigDict(strict=True)", content
)

# Remove field(default_factory=...) and replace with Field(default_factory=...)
content = content.replace("field(default_factory=", "Field(default_factory=")

# Special cases
content = content.replace(
    "def _construct(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:\n    return dataclass_type(**(payload or {}))",
    "def _construct(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:\n    return dataclass_type.model_validate(payload or {})",
)

content = content.replace(
    "def to_dict(self) -> dict[str, Any]:\n        return asdict(self)",
    "def to_dict(self) -> dict[str, Any]:\n        return self.model_dump()",
)

# Exception should not inherit from BaseModel
content = content.replace(
    "class ConfigValidationError(BaseModel):\n    model_config = ConfigDict(strict=True)\n    pass",
    "class ConfigValidationError(ValueError):\n    pass",
)
content = content.replace(
    "class ConfigValidationError(BaseModel):\n    model_config = ConfigDict(strict=True)",
    "class ConfigValidationError(ValueError):",
)

with open("training/src/config.py", "w") as f:
    f.write(content)
