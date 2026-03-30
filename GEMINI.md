# AI-Factory

## Project Overview

AI-Factory is a unified AI operating system designed to manage the complete lifecycle of large language models (LLMs). It provides a comprehensive platform spanning data preparation, model training, evaluation, monitoring, and multi-target deployment. 

The project features a multi-domain architecture capable of handling tasks in mathematics, code generation, reasoning, and creative writing. It offers multiple interfaces including a robust Command-Line Interface (CLI), a Terminal UI (TUI), a modern Next.js web dashboard, and an Electron-based desktop application.

### Key Technologies
*   **Backend & ML Stack**: Python 3.10+, PyTorch, Transformers, PEFT, Accelerate, bitsandbytes
*   **API & Data Validation**: FastAPI, Pydantic
*   **Frontend**: Next.js, React, Tailwind CSS (in `frontend/`)
*   **Desktop**: Electron (in `desktop/`)
*   **Tooling**: Ruff (linting/formatting), Mypy (type checking), Pytest (testing)

### Architecture
The repository is modularized into several core subsystems:
*   `ai_factory/`: Core shared foundation, domain definitions, unified interfaces, and platform capabilities.
*   `data/`: Data adapters, synthesis pipelines, quality control, and tools.
*   `training/`: Training system, configurations, distributed orchestration, and scripts.
*   `evaluation/`: Evaluation framework, benchmark registries, and analysis tools.
*   `inference/`: FastAPI-based inference server and model management.
*   `frontend/`: Web-based management dashboard.
*   `desktop/`: Native application wrapper.

## Building and Running

### Prerequisites
*   Python >= 3.10
*   Node.js & npm (for frontend/desktop)
*   Docker & Docker Compose (optional, for containerized environments)

### Installation
```bash
# Clone and install Python dependencies in editable mode with dev tools
git clone https://github.com/super244/ai-factory.git
cd ai-factory
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend && npm install
```

### Key Commands (via Makefile)
*   **Start Backend Server**: `make serve` (runs FastAPI via Uvicorn on port 8000)
*   **Start Frontend Server**: `make frontend-dev` (runs Next.js dev server)
*   **Run Tests**: `make test` (runs pytest)
*   **Run Linter & Typechecker**: `make lint` (runs ruff and mypy)
*   **Format Code**: `make format` (runs ruff format)
*   **Clean Workspace**: `make clean`
*   **Docker Deployment**: `make docker-up` and `make docker-down`

*Alternatively, use the unified CLI:*
```bash
ai-factory serve   # Starts the backend server
ai-factory tui     # Starts the terminal UI
ai-factory --help  # Lists all available CLI commands
```

## Development Conventions

*   **Code Style**: The project enforces a strict coding style using `ruff` with a line length of 120 characters. Always run `make format` and `make lint` before committing.
*   **Type Hinting**: Python code uses heavy type hinting. `mypy` is used for static type checking. Pydantic is used extensively for schema validation and settings management.
*   **Testing**: Tests are written using `pytest` and reside in the `tests/` directory. `pytest-asyncio` is used for asynchronous testing. Maintain high coverage when adding new features.
*   **Extensibility**: The system relies on a multi-domain plugin architecture (`ai_factory.domains`). When adding new capabilities, adhere to the established Factory and Protocol/Interface patterns.
*   **Configuration**: Settings are primarily managed via YAML configuration files located in the `configs/` directory or within specific subsystem config directories (e.g., `training/configs/`).