# AiderVer-KE: Multi-Agent Collaborative Knowledge Extraction Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## 📖 Project Overview

AiderVer-KE is a multi-agent collaborative knowledge extraction framework that coordinates five specialized agents to complete the end-to-end extraction pipeline:

1. **Schema Agent** — Parses user-defined entity/relation specs and generates closed-set extraction schemas
2. **Aider Agent** — Retrieves external background knowledge via RAG to assist entity disambiguation
3. **Extraction Agent** — Executes structured information extraction under schema constraints, enhanced with historical error context
4. **Probe Agent** — Performs independent quality inspection, generates structured defect feedback without modifying results
5. **Verifier Agent** — Conducts triple-check verification (entity truthfulness, type consistency, graph integrity) and triggers re-extraction on failure

The framework also introduces a **Collaborative Memory** module for cross-agent error experience sharing, enabling the system to self-evolve and reduce repeated extraction errors over time.

Supported tasks: Named Entity Recognition (NER), Relation Extraction (RE), Event Extraction (EE), and Triple Extraction.

## ✨ Core Features

### 🤖 Multi-Agent Architecture (Five-Agent Collaborative)
- **Schema Agent**: Parses predefined entity/relation specs, generates closed-set extraction schemas
- **Aider Agent**: Retrieves external knowledge via RAG for entity disambiguation
- **Extraction Agent**: Executes structured information extraction with schema constraints
- **Probe Agent**: Quality inspection agent that analyzes defects and generates structured feedback (without modifying results)
- **Verifier Agent**: Final verification with triple-check rules (entity truthfulness, type consistency, graph integrity)

### 🧠 Collaborative Memory
- Persistent storage for full-pipeline historical extraction error logs
- Cross-agent shared error experience pool for knowledge sharing
- Historical error case retrieval for Extraction Agent during inference
- Supports system self-evolution: reduces repeated errors without new manual annotations

### 🔧 Flexible Operating Modes
- **Quick Mode**: Fast extraction mode for simple tasks (Schema → Extraction)
- **Standard Mode**: Standard extraction mode balancing efficiency and accuracy
- **Enhanced Mode**: Full five-agent collaborative pipeline (Schema → Aider → Extraction → Probe → Verifier)
- **Customized Mode**: Custom mode with flexible agent configuration
- **Ablation Modes**: Ablation experiment modes for studying component contributions (no-aider, no-verifier)

### 🌐 Multi-Model Support
- Support for various Large Language Models (LLMs)
- Support for local models and API calls
- Support for VLLM service deployment

## 🚀 Quick Start

### Requirements
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Configure Model and Task Parameters**

Edit the `src/config.yaml` file to configure your model and extraction tasks:

```yaml
model:
  embedding_model: all-MiniLM-L6-v2
  # Other model configurations...

extraction:
  task: "ner"  # or "re", "ee", "triple"
  instruction: "Extract the Named Entities in the given text."
  text: "Your input text here"
  mode: "enhanced"  # Choose operating mode
```

2. **Run Extraction Tasks**

```bash
cd src
python run.py --config config.yaml
```

### Running Experiments

The project provides pre-configured experiment scripts:

```bash
# Run NER experiments
cd experiments
python run_ner.py

# Run relation extraction experiments
python run_re.py
```

## 🏛️ Architecture

### Pipeline Flow

```
Input Text
    │
    ▼
┌─────────────────┐
│  Schema Agent   │  Parse specs → Generate closed-set extraction schema
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Aider Agent   │  (Optional) RAG external knowledge retrieval
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extraction Agent│  Structured extraction with schema constraints
│                 │  + Collaborative Memory historical error context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Probe Agent   │  Quality inspection → Structured defect feedback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verifier Agent  │  Triple-check verification
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  Pass      Fail
    │         │
    ▼         ▼
  Output   Re-extraction (with Probe feedback +
           Collaborative Memory error context)
```

### Verification Feedback Loop

When the Verifier Agent detects issues (low confidence, entity type errors, missing entities, etc.), the pipeline triggers a re-extraction cycle:
1. Probe Agent's defect feedback is injected into the Extraction Agent's context
2. Collaborative Memory provides historical error patterns to avoid
3. Extraction Agent re-runs with enhanced constraints
4. Maximum reprocessing attempts prevent infinite loops

### 🧠 Collaborative Memory (Cross-Agent Shared)

The Collaborative Memory module is a key innovation of AiderVer-KE, solving the problem of traditional multi-agent systems having no error inheritance mechanism:

| Capability | Description |
|-----------|-------------|
| Error Logging | Persistent archiving of extraction errors, Probe defect reports, and Verifier feedback |
| Cross-Agent Sharing | All agents share the same memory pool, enabling error experience flow across roles |
| Inference Retrieval | Extraction Agent retrieves similar historical error cases during inference |
| Self-Evolution | Continuously accumulates error experience, reducing repeated errors without new annotations |
| Session Tracking | Records complete extraction session summaries for pattern analysis |

Memory is stored in `src/modules/knowledge_base/collaborative_memory.json` and persists across sessions.

## 📁 Project Structure

```
AiderVer-KE/
├── data/
│   ├── datasets/                  # Standard evaluation datasets
│   │   ├── CrossNER/             # Cross-domain NER (ai, literature, music, politics, science)
│   │   └── NYT11/                # NYT relation extraction dataset
│   └── input_files/              # Example input files (PDF, TXT, HTML, JSON)
├── experiments/
│   ├── run_ner.py                # NER benchmark experiment script
│   └── run_re.py                 # RE benchmark experiment script
├── src/
│   ├── modules/
│   │   ├── schema_agent.py       # Schema Agent: extraction schema generation
│   │   ├── aider_agent.py        # Aider Agent: RAG knowledge retrieval
│   │   ├── extraction_agent.py   # Extraction Agent: structured extraction
│   │   ├── probe_agent.py        # Probe Agent: quality inspection & defect feedback
│   │   ├── verifier_agent.py     # Verifier Agent: triple-check verification
│   │   ├── collaborative_memory.py # Collaborative Memory: cross-agent error sharing
│   │   └── knowledge_base/       # Knowledge base & case repository
│   ├── models/                   # LLM engine definitions & prompt templates
│   ├── utils/                    # Data definitions & processing utilities
│   ├── construct/                # Knowledge graph construction (Neo4j)
│   ├── pipeline.py               # Main pipeline orchestrator
│   ├── run.py                    # CLI entry point
│   └── config.yaml               # Configuration file
├── requirements.txt
└── README.md
```

## 🔧 Configuration Guide

### Model Configuration

```yaml
model:
  embedding_model: all-MiniLM-L6-v2  # Embedding model
  category: "OpenAI"                  # Model category
  model_name_or_path: "gpt-3.5-turbo" # Model name or path
  api_key: "your-api-key"            # API key
  base_url: "https://api.openai.com" # API base URL
  vllm_serve: false                  # Whether to use VLLM service
```

### Agent Configuration

```yaml
agent:
  aider_agent:
    auto_decision: true              # Automatic decision-making
    confidence_threshold: 0.6        # Confidence threshold
    max_search_terms: 3             # Maximum search terms
  
  chunk_token_limit: 1024           # Text chunk token limit
```

### Operating Mode Configuration

- **enhanced**: Complete pipeline using all agents
- **ablation_no_aider**: Ablation experiment removing Aider Agent
- **ablation_no_verifier**: Ablation experiment removing Verifier Agent
- **minimal**: Minimized process using only core components

## 📂 File Input Support

AiderVer-KE supports multiple file formats for extraction:

| Format | Extension | Description |
|--------|-----------|-------------|
| Text | `.txt` | Plain text files |
| PDF | `.pdf` | PDF document files |
| Word | `.docx` | Microsoft Word documents |
| HTML | `.html` | Web page files |
| JSON | `.json` | JSON data files |

To use file input, set `use_file: true` and provide the `file_path` in your config:

```yaml
extraction:
  task: "NER"
  use_file: true
  file_path: "../data/input_files/Artificial_Intelligence_Wikipedia.txt"
  mode: "enhanced"
```

Long texts are automatically chunked based on `chunk_token_limit` (default: 1024 tokens).

## 🌐 Knowledge Graph Construction

AiderVer-KE supports direct construction of knowledge graphs in Neo4j from extraction results:

```yaml
construct:
  database: "neo4j"
  url: "bolt://localhost:7687"
  username: "neo4j"
  password: "your-password"
```

When the `construct` section is present in config, extraction results are automatically converted to Cypher statements and loaded into the specified Neo4j database.

## 📊 Supported Task Types

| Task Type | Description | Example |
|-----------|-------------|----------|
| NER | Named Entity Recognition | Identify person names, locations, organizations, etc. |
| RE | Relation Extraction | Extract semantic relationships between entities |
| EE | Event Extraction | Identify and extract event information |
| Triple | Triple Extraction | Extract subject-relation-object triples |

## 🎯 Experiments and Evaluation

The project supports experiments on multiple standard datasets:

- **CrossNER**: Cross-domain Named Entity Recognition (5 domains: AI, Literature, Music, Politics, Science)
- **NYT11**: New York Times Relation Extraction Dataset

### Running Experiments

```bash
# Run NER experiments across all CrossNER domains
cd experiments
python run_ner.py

# Run relation extraction experiments on NYT11
python run_re.py
```

### Ablation Study Modes

AiderVer-KE provides built-in ablation modes to study each component's contribution:

| Mode | Description | Pipeline |
|------|-------------|----------|
| `enhanced` | Full five-agent pipeline | Schema → Aider → Extraction → Probe → Verifier |
| `ablation_no_aider` | Remove Aider Agent | Schema → Extraction → Probe → Verifier |
| `ablation_no_verifier` | Remove Verifier Agent | Schema → Aider → Extraction → Probe |
| `minimal` | Core components only | Extraction → Verifier |

Compare F1 scores across modes to quantify the contribution of RAG knowledge enhancement, independent verification, and collaborative memory.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Thanks to the open-source community for providing excellent tools and libraries
- Thanks to all contributors for their support

## 📞 Contact Us

If you have any questions or suggestions, please contact us through:

- Submit an Issue
- Send email to project maintainers
- Participate in project discussions

---

**Note**: Please ensure proper configuration of API keys and model paths before use. Some features may require additional permissions or resources.