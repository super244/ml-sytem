from __future__ import annotations


def fix_schemas() -> None:
    filepath = "inference/app/schemas.py"
    with open(filepath) as f:
        content = f.read()

    if "ConfigDict" not in content:
        content = content.replace(
            "from pydantic import BaseModel, Field", "from pydantic import BaseModel, Field, ConfigDict"
        )

        # Add model_config to all classes
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.startswith("class ") and ":" in line:
                new_lines.append("    model_config = ConfigDict(strict=True)")

        with open(filepath, "w") as f:
            f.write("\n".join(new_lines))


fix_schemas()
