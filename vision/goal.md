# AI-FACTORY: V2 Ultimate Vision & Goal

## 🌌 Vision

AI-Factory has evolved from a foundational ML lifecycle manager into a **Unified, Autonomous AI Research Lab**. 

It is designed to manage large language models, *but now, the system operates itself*. It is built to be:

* **Autonomous** enough to iterate models while you sleep.
* **Scalable** enough to coordinate distributed GPU clusters.
* **Structured** enough to maintain perfect data lineage from dataset generation to inference deployment.

The goal is to eliminate friction in AI development and automate the tedious parts of the scientific method.

---

## 🧠 Core Idea (The V2 Expansion)

V1 successfully established a **full-cycle LLM platform** for training, evaluating, and deploying models. 
V2 introduces the **Autonomous Loop**, enabling intelligent agents to guide that lifecycle:

1. **Dataset Generation & Curation**: Automated pipelines that synthesize rigorous training data, validate it, and pack it.
2. **AutoML & Search**: Intelligent hyperparameter and architecture searches that systematically explore the loss landscape.
3. **Multi-Agent Orchestration**: Agent swarms that write self-improving evaluation benchmarks, run training scripts, and analyze metrics to queue the next best experiment.
4. **Distributed Cluster Orchestration**: Seamless management of compute workloads across a fleet of local and remote GPU nodes.

All of this happens inside a single, connected system accessible through web and native shells.

---

## 🔁 The AI Lifecycle (End-to-End V2)

### 1. Data Assembly (Datasets)
AI-Factory does not just train on data; it builds it.
* Visualize the data curation pipeline.
* Filter, clean, and deduplicate synthetic datasets in real-time.
* Track lineage from raw internet dumps to `pack_summary.json`.

### 2. Autonomous Experimentation (AutoML & Agents)
Instead of manually tweaking learning rates:
* Launch an `AutoML` search tree. The system trains 5 LoRAs concurrently across different hyperparameter bands.
* **Agents** continuously evaluate these models. The swarm prunes failing runs and spawns deeper finetuning runs on promising candidates (e.g., using DPO or RLHF on the winner).

### 3. Distributed Training (Cluster)
* The same simple interface from V1, but now scaled. 
* Transparently dispatch training runs to local Macs, remote EC2/Lambda GPUs, or local Linux rigs.
* Health monitoring for distributed compute limits (`nvidia-smi` metrics beamed straight to the dashboard).

### 4. Seamless Inference & Deployment
* Zero-click deployments to local API shims, Ollama, LM Studio, or Hugging Face.
* Interactive chat UI with built-in telemetry to automatically flag user-prompt failures as future dataset additions.

---

## 🧩 Interfaces V2

AI-Factory is accessible through multiple perfectly-synced interfaces:

* **Web Control Center (Next.js)** → The ultimate window into the laboratory. Includes separate spheres for the *Lifecycle* (V1) and the *Lab* (V2: Datasets, Agents, AutoML, Clusters).
* **Desktop Shell (Electron Native)** → A robust native Mac application wrapping the Web Control Center. It features native OS menus, deep-linking, background auto-retries, and native file management.
* **TUI (Terminal UI)** → Low-latency, fast keyboard-driven matrix for cluster ops and live metric monitoring.
* **CLI** → Core scriptable control layer for pipeline integrations.

---

## 🎚️ User Levels

The system adapts to the user's operational depth:

### Beginner
* **“Click and run”**: Uses guided workflows, out-of-the-box synthetic datasets, and one-click training presets.
* Minimal exposure to distributed complexities.

### Hobbyist
* **Experimenter**: Adjusts parameters, links customized local datasets, manages single-node multi-GPU runs.

### Developer / Researcher
* **Architect**: Orchestrates agent swarms, overrides raw `peft` and `transformers` architectures, maps complex distributed clusters, and authors fully custom AutoML traversal strategies.

---

## 🏗️ Design Principles

* **Lineage Above All** → Every model must track exactly what dataset it used, what its parent was, and what its evaluation scores were.
* **Immaculate Codebase** → Perfect PEP-8 styling, 100% strict `mypy` typing, and ruthless enforcement of clean architecture interfaces.
* **Modularity** → Every component is replaceable. The frontend calls an API block; the CLI calls the same block.

---

## 🚀 Ultimate Goal

AI-Factory becomes:

> An autonomous AI laboratory that rivals enterprise platform systems (like NVIDIA NeMo), but is built natively for hackers, researchers, and individuals operating at hyperspeed.

It is a system that amplifies your ability to build intelligence continuously.
