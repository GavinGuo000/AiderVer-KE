# AiderVer-KE: Multi-Agent Collaborative Knowledge Extraction Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## 📖 Project Overview

AiderVer-KE is a multi-agent collaborative knowledge extraction framework designed to improve the accuracy and efficiency of knowledge extraction tasks through the coordinated work of multiple specialized agents. The framework supports various knowledge extraction tasks including Named Entity Recognition (NER), Relation Extraction (RE), Event Extraction (EE), and Triple Extraction.

## ✨ Core Features

### 🤖 Multi-Agent Architecture
- **Schema Agent**: Automatically generates and optimizes extraction schemas
- **Aider Agent**: Provides external knowledge support and intelligent decision-making
- **Extraction Agent**: Executes core information extraction tasks
- **Reflection Agent**: Reflects on and improves extraction results
- **Verifier Agent**: Verifies the accuracy of extraction results

### 🔧 Flexible Operating Modes
- **Quick Mode**: Fast extraction mode for simple tasks
- **Standard Mode**: Standard extraction mode balancing efficiency and accuracy
- **Enhanced Mode**: Enhanced mode using full agent collaboration
- **Customized Mode**: Custom mode with flexible agent configuration
- **Ablation Modes**: Ablation experiment modes for studying component contributions

### 🌐 Multi-Model Support
- Support for various Large Language Models (LLMs)
- Support for local models and API calls
- Support for VLLM service deployment
- Support for LoRA fine-tuned models

### 📊 Knowledge Graph Construction
- Automatic knowledge graph construction
- Support for Neo4j graph database
- Visualized knowledge graph display

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

## 📁 Project Structure

AiderVer-KE/
├── data/                          # Data directory
│   ├── datasets/                  # Datasets
│   │   ├── CrossNER/             # CrossNER dataset
│   │   └── NYT11/                # NYT11 dataset
│   └── input_files/              # Input file examples
├── experiments/                   # Experiment scripts
│   ├── run_ner.py                # NER experiments
│   └── run_re.py                 # Relation extraction experiments
├── src/                          # Source code
│   ├── config.yaml               # Configuration file
│   ├── pipeline.py               # Main pipeline
│   ├── run.py                    # Entry point
│   ├── models/                   # Model definitions
│   │   ├── llm_def.py           # LLM definitions
│   │   ├── prompt_template.py    # Prompt templates
│   │   └── vllm_serve.py        # VLLM service
│   ├── modules/                  # Agent modules
│   │   ├── aider_agent.py       # Aider agent
│   │   ├── extraction_agent.py  # Extraction agent
│   │   ├── reflection_agent.py  # Reflection agent
│   │   ├── schema_agent.py      # Schema agent
│   │   └── verifier_agent.py    # Verifier agent
│   ├── construct/               # Knowledge graph construction
│   └── utils/                   # Utility functions
├── requirements.txt             # Dependencies list
└── README.md                   # Project documentation


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

## 📊 Supported Task Types

| Task Type | Description | Example |
|-----------|-------------|----------|
| NER | Named Entity Recognition | Identify person names, locations, organizations, etc. |
| RE | Relation Extraction | Extract semantic relationships between entities |
| EE | Event Extraction | Identify and extract event information |
| Triple | Triple Extraction | Extract (subject, relation, object) triples |

## 🎯 Experiments and Evaluation

The project supports experiments on multiple standard datasets:

- **CrossNER**: Cross-domain Named Entity Recognition
- **NYT11**: New York Times Relation Extraction Dataset

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the ZJUNLP team for their contributions
- Thanks to the open-source community for providing excellent tools and libraries
- Thanks to all contributors for their support

## 📞 Contact Us

If you have any questions or suggestions, please contact us through:

- Submit an Issue
- Send email to project maintainers
- Participate in project discussions

---

**Note**: Please ensure proper configuration of API keys and model paths before use. Some features may require additional permissions or resources.