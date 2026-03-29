# 🏭 AI-Factory

> **Unified AI Operating System for Complete LLM Lifecycle Management**

AI-Factory is a comprehensive platform designed to manage the entire lifecycle of large language models. From training to deployment, monitoring to iteration, AI-Factory provides a unified ecosystem for AI development across multiple domains.

## 🌟 **What Makes AI-Factory Special**

### **🎯 Multi-Domain Architecture**
- **Mathematics**: Calculus, algebra, olympiad reasoning, statistics
- **Code Generation**: Python, JavaScript, system design, debugging
- **Reasoning**: Logic, pattern recognition, causal reasoning
- **Creative**: Writing, content generation, creative problem-solving
- **Custom Domains**: Extensible framework for specialized domains

### **🔄 Complete Lifecycle Management**
- **Training**: From data preparation to model training
- **Monitoring**: Real-time metrics, logs, and health monitoring
- **Evaluation**: Comprehensive benchmarking and analysis
- **Iteration**: AI-assisted improvement recommendations
- **Deployment**: Multi-target deployment with one click

### **🎨 Unified Interface Experience**
- **CLI**: Powerful command-line interface for automation
- **TUI**: Interactive terminal dashboard
- **Web**: Modern web-based management interface
- **Desktop**: Native desktop application

### **⚡ Scalable & Extensible**
- **Local to Cloud**: Scale from laptop to distributed systems
- **Platform Capabilities**: Distributed training, real-time monitoring
- **Plugin Architecture**: Easy to extend with new domains and features

## 🚀 **Key Features**

### **📊 Data Management**
- **Universal Schema**: Canonical v2 dataset records with rich metadata
- **Multi-Source Integration**: Local, HuggingFace, S3, web sources
- **Quality Control**: Automated scoring, deduplication, contamination detection
- **Synthetic Generation**: Template-based and curriculum-driven data synthesis

### **🎓 Training System**
- **Flexible Profiles**: QLoRA, full fine-tuning, curriculum learning
- **Distributed Training**: Multi-node scaling with resource management
- **Experiment Tracking**: Comprehensive run manifests and metadata
- **Model Comparison**: Side-by-side performance analysis

### **🔍 Evaluation & Monitoring**
- **Benchmark Registry**: Standardized evaluation benchmarks
- **Real-time Metrics**: Live performance and system health monitoring
- **Failure Analysis**: Automated error taxonomy and mining
- **Performance Reports**: Detailed analytics and insights

### **🌐 Deployment & Inference**
- **Multi-Target**: HuggingFace, Ollama, LM Studio, custom APIs, edge devices
- **FastAPI Backend**: High-performance inference server
- **Prompt Management**: Configurable prompt presets and templates
- **Model Registry**: Centralized model versioning and management

## 🏗️ **Architecture Overview**

```text
AI-Factory/
├── 🎯 ai_factory/
│   ├── core/           # Shared foundation (schemas, artifacts, hashing)
│   ├── domains/        # Multi-domain support (math, code, reasoning)
│   ├── interfaces/     # Unified interfaces (CLI, TUI, Web, Desktop)
│   └── platform/       # Platform capabilities (scaling, monitoring, deployment)
├── 📊 data/            # Data layer (adapters, synthesis, quality, tools)
├── 🎓 training/        # Training system (configs, scripts, extensions)
├── 🔍 evaluation/      # Evaluation framework (benchmarks, metrics, analysis)
├── 🌐 inference/       # Inference server (FastAPI, models, prompts)
├── 🎨 frontend/        # Web interface (Next.js dashboard and tools)
├── 📱 desktop/        # Desktop application (Electron)
├── 📓 notebooks/       # Research notebooks and tutorials
├── 📚 docs/           # Documentation and guides
├── 🧪 tests/          # Comprehensive test suite
└── ⚙️ configs/        # Configuration files and profiles
```

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone the repository
git clone https://github.com/super244/ml-sytem.git
cd ai-factory

# Install in development mode
pip install -e .[dev]

# Verify installation
ai-factory --help
```

### **Basic Usage**
```bash
# List available domains
ai-factory domain list

# Check platform status
ai-factory platform status

# Start training with default config
ai-factory new --config configs/finetune.yaml

# Monitor progress
ai-factory tui

# Launch web interface
ai-factory serve
```

### **Development Setup**
```bash
# Install frontend dependencies
cd frontend && npm install

# Run development servers
ai-factory serve           # Backend server
npm run dev               # Frontend development

# Run tests
pytest                    # Python tests
npm test                   # Frontend tests
```

## 📚 **Documentation**

### **Core Guides**
- **[Architecture Guide](docs/architecture.md)** - System design and principles
- **[Data System Guide](docs/data-system.md)** - Data layer and processing
- **[Training Guide](tune-guide.md)** - Model training and fine-tuning
- **[Deployment Guide](docs/deployment-guide.md)** - Deployment and production

### **API Documentation**
- **[API Reference](docs/api/README.md)** - REST API and endpoints
- **[CLI Reference](docs/cli-guide.md)** - Command-line interface
- **[Configuration Guide](docs/config-guide.md)** - Configuration options

### **Tutorials & Examples**
- **[Notebooks](notebooks/)** - Interactive tutorials and explorations
- **[Examples](examples/)** - Code examples and templates
- **[Best Practices](docs/best-practices.md)** - Development guidelines

## 🛠️ **Development**

### **Code Quality**
```bash
# Linting and formatting
ruff check .              # Lint code
ruff format .             # Format code
mypy ai_factory/         # Type checking

# Testing
pytest                    # Run tests
pytest --cov=ai_factory  # With coverage
```

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

See [Contributor Guide](docs/contributor-guide.md) for detailed guidelines.

## 🎯 **Use Cases**

### **Research & Development**
- **Academic Research**: Experiment with new training methods and architectures
- **Model Development**: Develop and fine-tune domain-specific models
- **Benchmarking**: Compare model performance across tasks and domains

### **Production Deployment**
- **API Services**: Deploy models as REST APIs
- **Edge Computing**: Deploy to edge devices and IoT systems
- **Batch Processing**: Large-scale data processing and inference

### **Education & Learning**
- **Teaching**: Interactive tools for AI/ML education
- **Prototyping**: Rapid prototyping of AI applications
- **Experimentation**: Safe environment for learning and exploration

## 🔧 **Configuration**

### **Environment Variables**
```bash
export AI_FACTORY_REPO_ROOT="/path/to/ai-factory"
export AI_FACTORY_ARTIFACTS_DIR="/path/to/artifacts"
export AI_FACTORY_LOG_LEVEL="INFO"
```

### **Key Configuration Files**
- `configs/finetune.yaml` - Fine-tuning configuration
- `configs/eval.yaml` - Evaluation settings
- `configs/inference.yaml` - Inference server settings
- `data/configs/processing.yaml` - Data processing pipeline
- `training/configs/profiles/` - Training profiles

## 🌟 **Community & Support**

### **Getting Help**
- **Issues**: [GitHub Issues](https://github.com/super244/ml-sytem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/super244/ml-sytem/discussions)
- **Documentation**: [Full Documentation](docs/)

### **Contributing**
- **Code Contributions**: See [Contributor Guide](docs/contributor-guide.md)
- **Bug Reports**: Use GitHub Issues with detailed information
- **Feature Requests**: Use GitHub Discussions for ideas and proposals

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 **Acknowledgments**

- Built with modern Python ecosystem (FastAPI, Pydantic, Next.js)
- Inspired by best practices in ML engineering and research
- Community-driven development and improvement

---

**AI-Factory** - *Your complete AI development platform* 🚀

> For more information, visit our [documentation](docs/) or join our [community discussions](https://github.com/super244/ml-sytem/discussions).
