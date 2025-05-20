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

