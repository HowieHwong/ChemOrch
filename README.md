# ChemOrch - A Chemical Task Orchestration Framework
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

[![MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)

[![LLM Provider](https://img.shields.io/badge/LLM%20Provider-OpenAI-orange.svg)](https://openai.com/)

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
```

## Quick Start
```python
import asyncio
from ChemOrch import ChemOrch

async def main():
    await ChemOrch(
        task="AI4Chemistry question answering",
        task_description="The application of AI in chemistry domain.",
        instruction_file="ChemGen/results/instructions.json",
        output_file="ChemGen/results/results.json",
        num=5,
    )

asyncio.run(main())
```
## Metadata File Specification

ChemOrch supports four metadata formats: JSON, CSV, Pickle, and text.

**The IG model accepts CSV, JSON, and text-format data, and the RG model accepts Pickle and text-format data.**

### 1. Text

You can input your text into a Python string.

### 2. CSV

Currently, ChemOrch only accepts CSV files with two columns. The first column indicates the auxiliary data for the Instruction Generation, while the second column is optional, representing the ground truth of the instruction.
```csv
SMILES,IUPAC Name
CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C,3-acetyloxy-4-(trimethylazaniumyl)butanoate
CC(=O)OC(CC(=O)O)C[N+](C)(C)C,(2-acetyloxy-3-carboxypropyl)-trimethylazanium
C1=CC(C(C(=C1)C(=O)O)O)O,5,6-dihydroxycyclohexa-1,3-diene-1-carboxylic acid
```

### 3. JSON

Currently, ChemOrch only accepts JSON files with two keys. The first key indicates the auxiliary data for the Instruction Generation, while the second key is optional, representing the ground truth of the instruction.

```python
[
    {
        "SMILES": "CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
        "IUPAC Name": "3-acetyloxy-4-(trimethylazaniumyl)butanoate"
    },
    {
        "SMILES": "CC(=O)OC(CC(=O)O)C[N+](C)(C)C",
        "IUPAC Name": "(2-acetyloxy-3-carboxypropyl)-trimethylazanium"
    },
    {
        "SMILES": "C1=CC(C(C(=C1)C(=O)O)O)O",
        "IUPAC Name": "5,6-dihydroxycyclohexa-1,3-diene-1-carboxylic acid"
    }
]
```

### 4. Pickle

ChemOrch holds an abundant tool pool that integrates the RDKit and PubChem toolkits. You can extend the tool pool by adding the Pickle file with the required information.
**The format of our tool pool**
```python
metadata = [{
    "tool": "mol_from_smiles",
    "module": "rdkit_tool",
    "description": "Creates an RDKit molecule object from a SMILES string.",
    "embedding": "The embedding of the tool description",  # 修正拼写错误
    "parameters": {
        "smiles": {
            'smiles': '(str) The SMILES representation of the molecule.'
        }
    }
}]
```
**The format of the additional tools**
```python
metadata = [{
    "tool": "smiles_from_compound",
    "module": "ord_schema.message_helpers",
    "description": "Fetches or generates a SMILES identifier for a compound. If a SMILES identifier already exists, it is simply returned.",
    "parameters": {"compound": "reaction_pb2.Compound message."},
    "documentation": "https://docs.open-reaction-database.org/en/latest/ord_schema/ord_schema.html#module-ord_schema.message_helpers"
}]
```
After you have made the metadata, you should convert it into a Pickle file.

```python
import pickle

with open('./additional_tools.pkl', 'wb') as f:
    pickle.dump(metadata, f)
```














