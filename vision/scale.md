# AI-Factory: Scaling Architecture & Implementation Plan

## 🎯 Executive Summary

This document outlines the precise steps to evolve the current "Atlas Math Lab" codebase into the full "AI-Factory" vision described in `goal.md`. The transformation involves architectural expansion, interface unification, and capability extension while preserving the solid foundation already built.

---

## 📊 Current State vs Target Vision

### Current State (Atlas Math Lab)
- **Focus**: Specialized mathematics/calculus reasoning models
- **Scope**: Research-grade monorepo with data, training, inference, evaluation
- **Interfaces**: CLI, Web Dashboard, nascent Desktop app
- **Architecture**: Solid shared-core platform with orchestration control plane

### Target State (AI-Factory)
- **Focus**: General LLM lifecycle management platform
- **Scope**: Full-cycle AI operating system (train → monitor → evaluate → iterate → deploy)
- **Interfaces**: CLI, TUI, Web Dashboard, Desktop App (unified)
- **Architecture**: Extensible, multi-domain, infinitely scalable platform

---

## 🏗️ Scaling Strategy: Phased Evolution

### Phase 1: Foundation Expansion (Weeks 1-2)

#### 1.1 Rebranding & Restructuring
```bash
# Rename project identity
- "Atlas Math Lab" → "AI-Factory" 
- Update pyproject.toml name: "atlas-math-lab" → "ai-factory"
- Update README.md to reflect general LLM platform
- Preserve math capabilities as first "domain module"
```

#### 1.2 Domain Architecture Introduction
```
ai_factory/
├── core/                    # Shared foundation (existing)
├── domains/                 # NEW: Domain-specific modules
│   ├── mathematics/         # Move existing math logic here
│   ├── code/               # NEW: Code generation domain
│   ├── reasoning/          # NEW: General reasoning domain
│   └── creative/           # NEW: Creative writing domain
├── interfaces/             # NEW: Unified interface layer
│   ├── cli/               # Move existing CLI here
│   ├── tui/               # NEW: Terminal UI
│   ├── web/               # Move frontend here
│   └── desktop/           # Move desktop here
└── platform/              # NEW: Platform orchestration
    ├── scaling/           # NEW: Distributed training
    ├── monitoring/        # NEW: Real-time monitoring
    └── deployment/        # NEW: Multi-target deployment
```

#### 1.3 Core Platform Extensions
```python
# Extend core schemas for general LLM support
class DomainConfig(BaseModel):
    name: str
    version: str
    datasets: List[DatasetSpec]
    metrics: List[MetricSpec]
    default_models: List[str]

class MultiDomainRegistry(BaseModel):
    domains: Dict[str, DomainConfig]
    cross_domain_tasks: List[CrossDomainTask]
```

### Phase 2: Interface Unification (Weeks 3-4)

#### 2.1 Unified Backend API
```python
# New unified API router structure
interfaces/web/api/
├── v1/
│   ├── training.py      # General training endpoints
│   ├── inference.py     # General inference endpoints  
│   ├── evaluation.py    # General evaluation endpoints
│   ├── monitoring.py    # Real-time monitoring
│   └── domains/         # Domain-specific endpoints
│       ├── mathematics.py
│       ├── code.py
│       └── reasoning.py
└── shared/              # Shared utilities
```

#### 2.2 Terminal User Interface (TUI)
```python
# NEW: Rich-based TUI for system management
interfaces/tui/
├── main.py              # Entry point
├── panels/
│   ├── training.py      # Training monitoring
│   ├── evaluation.py    # Results dashboard
│   ├── models.py        # Model management
│   └── system.py        # System overview
└── components/          # Reusable UI components
```

#### 2.3 Desktop App Enhancement
```javascript
// Enhance existing desktop app for general platform
desktop/
├── main.js             # Existing
├── src/
│   ├── components/     # NEW: Modular components
│   ├── domains/        # NEW: Domain-specific UIs
│   └── platform/       # NEW: Platform management
└── assets/             # NEW: Unified branding
```

### Phase 3: Platform Capabilities (Weeks 5-6)

#### 3.1 Multi-Model Training Support
```yaml
# training/configs/models/
# NEW: General model architecture configs
transformer_base.yaml
transformer_large.yaml
moe_config.yaml
multimodal_config.yaml

# training/configs/domains/
# NEW: Domain-specific training profiles
mathematics_specialist.yaml
code_generation.yaml
reasoning_general.yaml
creative_writing.yaml
```

#### 3.2 Real-Time Monitoring System
```python
# platform/monitoring/
realtime/
├── metrics_collector.py    # Collect training metrics
├── dashboard_server.py     # WebSocket streaming
├── alerting.py            # Anomaly detection
└── storage.py             # Time-series storage
```

#### 3.3 Distributed Training Infrastructure
```python
# platform/scaling/
distributed/
├── orchestrator.py       # Multi-node coordination
├── resource_manager.py   # GPU/CPU resource allocation
├── data_sync.py         # Dataset distribution
└── checkpoint_sync.py   # Model checkpoint management
```

### Phase 4: Advanced Features (Weeks 7-8)

#### 4.1 Intelligent Iteration System
```python
# core/decisions/intelligent_iteration.py
class IterationAnalyzer:
    def analyze_results(self, results: EvaluationResults) -> List[Recommendation]:
        # AI-powered analysis of training results
        # Suggest: retrain, finetune, deploy, or architecture changes
        
class AutoIterationEngine:
    def execute_recommendations(self, recommendations: List[Recommendation]):
        # Execute approved recommendations automatically
```

#### 4.2 Multi-Target Deployment Pipeline
```python
# platform/deployment/
targets/
├── huggingface.py       # HuggingFace Hub
├── ollama.py           # Ollama integration
├── lm_studio.py        # LM Studio format
├── custom_api.py       # Custom API endpoints
└── edge_devices.py     # Edge deployment
```

#### 4.3 Cross-Domain Capabilities
```python
# domains/cross_domain/
multitask/
├── trainers.py         # Multi-task training
├── evaluators.py       # Cross-domain evaluation
└── adapters.py         # Domain adapters
```

---

## 🔧 Technical Implementation Details

### Core Schema Extensions

#### Universal Model Registry
```python
# core/registry/model_registry.py
class ModelSpec(BaseModel):
    name: str
    domain: str
    architecture: ArchitectureSpec
    capabilities: List[str]
    deployment_targets: List[str]
    resource_requirements: ResourceSpec
    performance_profile: PerformanceProfile
```

#### Domain-Agnostic Training Configs
```python
# training/configs/base_training.yaml
model:
  architecture: ${MODEL_ARCHITECTURE}
  domain: ${DOMAIN}
  
training:
  type: ${TRAINING_TYPE}  # supervised, unsupervised, rlhf
  domains: [${DOMAINS}]
  cross_domain_tasks: ${CROSS_DOMAIN_TASKS}

data:
  primary_domain: ${PRIMARY_DOMAIN}
  auxiliary_domains: ${AUXILIARY_DOMAINS}
  cross_domain_ratio: ${CROSS_DOMAIN_RATIO}
```

### Interface Layer Architecture

#### Unified CLI Commands
```bash
# Generalized CLI structure
ai-factory train --domain mathematics --config calculus_specialist
ai-factory train --domain code --config code_generation
ai-factory evaluate --model math_v2 --domains mathematics,reasoning
ai-factory deploy --model code_v1 --target huggingface --public
ai-factory monitor --instance-id <id> --realtime
ai-factory compare --model-a math_v1 --model-b math_v2 --domains mathematics
```

#### TUI Panel Structure
```
┌─────────────────────────────────────────────────────────────┐
│ AI-Factory Control Panel                                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Active Runs     │ System Metrics  │ Domain Overview         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────┐ │
│ │math_train   │ │ │GPU: 87%     │ │ │Mathematics: 3 runs  │ │
│ │code_train   │ │ │CPU: 45%     │ │ │Code: 2 runs         │ │
│ │reason_eval  │ │ │Memory: 62%  │ │ │Reasoning: 1 run     │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────┘ │
├─────────────────┴─────────────────┴─────────────────────────┤
│ Real-time Logs                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │[math_train] Step: 1234/10000 | Loss: 0.234 | Acc: 0.89 │ │
│ │[code_train] Step: 567/5000   | Loss: 0.456 | Acc: 0.76 │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Scaling Infrastructure

#### Horizontal Scaling Architecture
```python
# platform/scaling/cluster_manager.py
class ClusterManager:
    def __init__(self, config: ClusterConfig):
        self.nodes = []  # List of compute nodes
        self.scheduler = JobScheduler()
        self.resource_monitor = ResourceMonitor()
    
    def schedule_training_job(self, job: TrainingJob) -> ScheduledJob:
        # Distribute training across multiple nodes
        
    def scale_cluster(self, target_nodes: int):
        # Auto-scale cluster based on workload
```

#### Monitoring & Alerting
```python
# platform/monitoring/alerts.py
class AlertManager:
    def check_training_health(self, run_id: str) -> List[Alert]:
        # Check for training anomalies
        
    def check_system_resources(self) -> List[Alert]:
        # Check for resource bottlenecks
        
    def send_alert(self, alert: Alert):
        # Send notifications via configured channels
```

---

## 📋 Migration Checklist

### Data Migration
- [ ] Preserve existing math datasets in new domain structure
- [ ] Update dataset manifests to include domain metadata
- [ ] Migrate training checkpoints to new model registry
- [ ] Update evaluation results with domain tags

### Code Migration
- [ ] Move math-specific code to `domains/mathematics/`
- [ ] Update imports across the codebase
- [ ] Extend core schemas for multi-domain support
- [ ] Update CLI commands to be domain-aware
- [ ] Migrate frontend to support multiple domains

### Configuration Migration
- [ ] Restructure config files by domain
- [ ] Update model registry for general LLM support
- [ ] Create domain-specific training profiles
- [ ] Update deployment configs for multiple targets

### Testing & Validation
- [ ] Ensure all existing math functionality works
- [ ] Test new domain architecture
- [ ] Validate interface unification
- [ ] Performance testing for scaling features
- [ ] Integration testing for end-to-end workflows

---

## 🚀 Success Metrics

### Technical Metrics
- **Zero Regression**: All existing math functionality preserved
- **Domain Extensibility**: New domain addition in < 4 hours
- **Interface Consistency**: Unified experience across CLI/TUI/Web/Desktop
- **Scaling Performance**: Linear scaling up to 10 nodes
- **Monitoring Latency**: < 100ms metric collection to display

### User Experience Metrics
- **Onboarding Time**: < 15 minutes for new domains
- **Training Visibility**: Real-time metrics for all runs
- **Deployment Simplicity**: One-command deployment to any target
- **Iteration Speed**: < 5 minutes from evaluation to retraining

### Platform Metrics
- **Model Registry**: Support for 10+ model architectures
- **Domain Support**: 4+ domains out of the box
- **Deployment Targets**: 5+ deployment destinations
- **Monitoring Coverage**: 100% system component visibility

---

## 🎯 Next Steps

1. **Week 1**: Execute Phase 1.1 (Rebranding) and begin Phase 1.2 (Domain Architecture)
2. **Week 2**: Complete Phase 1 and start Phase 2.1 (Unified Backend API)
3. **Week 3**: Continue Phase 2 and begin Phase 3.1 (Multi-Model Support)
4. **Week 4**: Complete interface unification and monitoring basics
5. **Week 5-6**: Focus on platform capabilities and scaling
6. **Week 7-8**: Advanced features and cross-domain capabilities

The key is to **preserve the excellent foundation** while systematically adding the layers needed for the full AI-Factory vision. Each phase builds incrementally without breaking existing functionality.
