# ChemOrch - A Chemical Task Orchestration Framework
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

[![MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)

[![LLM Provider](https://img.shields.io/badge/LLM%20Provider-OpenAI-white.svg)](https://openai.com/)

## Key Features

- **Smart Tool Selection**: Embedding-based matching and accurate tools distillation with metadata support
- **Automatic Tool Invocation**: Effective code script generation aligns with the targets of instructions
- **Resilient Execution**: Multi-layer error recovery (syntax → logic → documentation-based)
- **Multi-modal Input**: Supports text/JSON/CSV/Pickle metadata formats

## Installation
```bash
# Clone the repository
git clone https://github.com/HowieHwong/ChemOrch.git
cd ChemOrch/ChemGen/generation

# Create a virtual environment
conda create -n ChemOrch python=3.9

#Install the requirements
pip install -r requirements.txt
