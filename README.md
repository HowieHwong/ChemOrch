<div align="center"> <h1>ChemOrch 🧪</h1> <h3>Towards Groundbreaking Chemistry Instruction Data Generation</h3> <p> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg"></a> <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a> <a href="https://openai.com/"><img src="https://img.shields.io/badge/LLM%20Provider-OpenAI-orange.svg"></a> </p> </div>



## ✨ Key Features

- **Smart Tool Selection**: Embedding-based matching and accurate tools distillation with metadata support
- **Automatic Tool Invocation**: Effective code script generation aligns with the targets of instructions
- **Resilient Execution**: Multi-layer error recovery (syntax → logic → documentation-based)
- **Multi-modal Input**: Supports text/JSON/CSV/Pickle metadata formats

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/HowieHwong/ChemOrch.git
cd ChemOrch/ChemGen/generation

# Create a virtual environment
conda create -n ChemOrch python=3.9

#Install the requirements
pip install -r requirements.txt
```

## 🚀 Quick Start
```python
import asyncio
from ChemOrch import ChemOrch

async def main():
    await ChemOrch(
        task="SMILES conversion",
        task_description="Predict the IUPAC name of a given compound's SMILES string.",
        instruction_file="ChemGen/example/instructions.json",
        output_file="ChemGen/example/instructions_response_pairs.json",
        num=10,
        batchsize=10,
        IG_metadata_type="json",
        IG_metadata_content="ChemGen/example/example.json",
        RG_metadata_type="text",
        RG_metadata_content="Please use the `get_compounds` tool in PubChem module to solve the tasks",
        constraint="Your instruction should focus on specific tasks and give specific values based on the metadata content."
    )

asyncio.run(main())
```

## 📦 Output Formats

**Output format of the IG model**
```json
    {
        "task": "SMILES conversion",
        "task_description": "Predict the IUPAC name of a given compound's SMILES string.",
        "constraint": "Your instruction should focus on specific tasks and give specific values based on the metadata content.",
        "instruction": "Predict the IUPAC name for the compound with the SMILES string 'CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C'."
    }
```
**Output format of the RG model**
```json
    {
        "task": "SMILES conversion",
        "instruction": "Translate the SMILES 'CCC(C)(C(C(=O)O)O)O' into its IUPAC nomenclature.",
        "response": "The IUPAC nomenclature for the provided SMILES string 'CCC(C)(C(C(=O)O)O)O' is: 2,3-dihydroxy-3-methylpentanoic acid.\n\nThis nomenclature breaks down as follows: the main chain consists of five carbon atoms (pentanoic acid), the third carbon atom has a methyl group attached (3-methyl), and there are hydroxyl groups on the second and third carbons (2,3-dihydroxy). The structure ends with a carboxylic acid functional group."
    }
```

## Metadata File Specification

ChemOrch supports four metadata formats: JSON, CSV, Pickle, and text.

**The IG model accepts CSV, JSON, and text-format data, and the RG model accepts Pickle and text-format data.**

### 1. Text

Text data should be provided as a raw Python string. The system will automatically parse the content based on the task type.

### 2. CSV

Currently, ChemOrch only accepts CSV files with two columns. The first column indicates the auxiliary data for the Instruction Generation, while the second column is *optional*, representing the ground truth of the instruction.
```csv
SMILES,IUPAC Name
CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C,3-acetyloxy-4-(trimethylazaniumyl)butanoate
CC(=O)OC(CC(=O)O)C[N+](C)(C)C,(2-acetyloxy-3-carboxypropyl)-trimethylazanium
C1=CC(C(C(=C1)C(=O)O)O)O,5,6-dihydroxycyclohexa-1,3-diene-1-carboxylic acid
```

### 3. JSON

Currently, ChemOrch only accepts JSON files with two keys. The first key indicates the auxiliary data for the Instruction Generation, while the second key is *optional*, representing the ground truth of the instruction.

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
    "embedding": "The embedding of the tool description",
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
The **`example`** document folder contains the example usage, metadata, and output files. Please read them for more details.

## 📁 Execution Parameters

| Parameter                 | Type | Description                                                  | Example Value&Detailed Requirements                          |
| :------------------------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`task`**                | str  | **Required.** General task category.                         | SMILES Conversion                                            |
| **`task_description`**    | str  | **Required.** Detailed task objective.                        | Convert canonical SMILES to IUPAC names                      |
| **`instruction_file`**    | str  | **Required.** Path to the JSON file where generated instructions will be stored. **Only supports JSON format.** | ./instructions/SMILES_to_IUPAC_instructions.json             |
| **`output_file`**         | str  | **Required.** Path for final output. **Only supports JSON format.** | ./results/SMILES_to_IUPAC_data_pairs.json                    |
| **`num`**                 | int  | **Required**. Number of instruction-response pairs to generate. | A number range from 1 to 1000. For large values(>100), use **`batchsize`** to optimize performance. |
| **`batchsize`**           | int  | *Optional*. Number of instructions generated per API call.   | Default to 10. You can choose a suitable value according to your API key limitation. |
| **`IG_metadata_type`**    | str  | *Conditional*. Metadata format for Instruction Generation. Required if `IG_metadata_content` is provided. | json/csv/text                                                 |
| **`IG_metadata_content`** | str  | *Conditional*. Content source for the IG model.              | - For `json`/`csv`: File path (e.g., `"./data/compounds.csv"`)<br>- For `text`: Direct string input (e.g., `"aspirin, paracetamol"`) |
| **`RG_metadata_type`**    | str  | *Conditional*. Metadata format for Response Generation.      | pickle/text                                                   |
| **`RG_metadata_content`** | str  | *Conditional*. Content source for the RG model.              | \- For `pickle`: Path to serialized file (e.g., `"./tools/additional_tools.pkl"`)<br/>\- For `text`: Direct string input(e.g., Please use the **`get_compounds`** function in the PubChem module to solve these tasks.) |
| **`constraint`**          | str  | *Optional*. Custom generation constraints in natural language for the IG model. | 1. The generated instructions should be pitched at the knowledge level of senior high school students.<br>2. Each instruction has to convey the task details in fewer words while ensuring the instruction is clear. |

## License
MIT - See [MIT License](https://opensource.org/license/mit) for details.












