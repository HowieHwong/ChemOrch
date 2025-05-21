from typing import Union
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier, ForwardSDMolSupplier, SDWriter

def mol_from_smiles(smiles: str) -> Chem.Mol:
    """
    Creates an RDKit molecule object from a SMILES string.

    Parameters:
    smiles (str): The SMILES representation of the molecule.

    Returns:
    rdkit.Chem.rdchem.Mol: The RDKit molecule object, or None if invalid.

    #Example
    #Example1:Get Ethanol Mol object from SMILES string and get its properties.
    mol = mol_from_smiles("CCO") # Ethanol
    if mol:
        print(f"Number of atoms:{mol.GetNumAtoms()}, Number of bonds:{mol.GetNumBonds()}")

    #Real-world use case:
    #A researcher wants to get a molecule object from a SMILES string.
    mol = mol_from_smiles("c1ccccc1") # Benzene
    if mol:
        print(f"Number of atoms:{mol.GetNumAtoms()}, Number of bonds:{mol.GetNumBonds()}")
    """
    return Chem.MolFromSmiles(smiles)


def mol_from_molfile(filename: str) -> Chem.Mol:
    """
    Reads a molecule from a Mol file.

    Parameters:
    filename (str): Path to the .mol file.

    Returns:
    rdkit.Chem.rdchem.Mol: The RDKit molecule object, or None if invalid.

    #Example
    #Example1:Get Ethanol Mol object from Mol file.
    mol = mol_from_molfile("Ethanol.mol")
    if mol:
        print(f"SMILES:{Chem.MolToSmiles(mol)},Number of atoms:{mol.GetNumAtoms()}, Number of bonds:{mol.GetNumBonds()}")

    #Real-world use case:
    #A researcher wants to get a molecule object from a Mol file.
    mol = mol_from_molfile("Your .mol file path")
    if mol:
        print(f"SMILES:{Chem.MolToSmiles(mol)}")
    """
    return Chem.MolFromMolFile(filename)


def mol_from_molblock(mol_block: str) -> Chem.Mol:
    """
    Reads a molecule from a Mol block.

    Parameters:
    mol_block (str): The Mol block data as a string.

    Returns:
    rdkit.Chem.rdchem.Mol: The RDKit molecule object, or None if invalid.

    #Example
    #Example1:Get Ethanol Mol object from Mol block.
    mol_block = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO")) # Generate Ethanol Mol block through RDkit
    mol = mol_from_molblock(mol_block)
    if mol:
        print(f"SMILES:{Chem.MolToSmiles(mol)},Number of atoms:{mol.GetNumAtoms()}, Number of bonds:{mol.GetNumBonds()}")

    #Real-world use case:
    #A researcher wants to get a molecule object from a Mol block.
    mol_block = "Your mol block"
    mol = mol_from_molblock(mol_block)
    if mol:
        print(f"SMILES:{Chem.MolToSmiles(mol)}")
    """
    return Chem.MolFromMolBlock(mol_block)


def draw_molecule(mol: Chem.Mol): # picture related
    """
    Generates an image of a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
    PIL.Image.Image: Image representation of the molecule.
    """
    return Draw.MolToImage(mol)


def read_sdf_file(filename: str) -> list:
    """
    Reads molecules from an SDF file.

    Parameters:
    filename (str): Path to the .sdf file.

    Returns:
    list[rdkit.Chem.rdchem.Mol]: List of RDKit molecule objects.

    #Example
    #Example1:Get Ethanol Mol objects from SDF file.
    mol = read_sdf_file("Ethanol.sdf")
    if mol:
        print(f"Number of molecules:{len(mol)},First molecule:{Chem.MolToSmiles(mol[0])}")

    #Real-world use case:
    #A researcher wants to get a list of molecule objects from an SDF file.
    mols = read_sdf_file("Multiple_Molecules.sdf")
    if mols:
        print(f"Number of molecules:{len(mols)}")
        for i,mol in enumerate(mols):
            print(f"Molecule {i+1} is {Chem.MolToSmiles(mol)}")
    """
    return [mol for mol in SDMolSupplier(filename) if mol is not None]


def read_smiles_file(filename: str) -> list:
    """
    Reads molecules from a SMILES file.

    Parameters:
    filename (str): Path to the .smi file.

    Returns:
    list[rdkit.Chem.rdchem.Mol]: List of RDKit molecule objects.

    #Example
    #Example1:Get Ethanol Mol objects from SMILES file.
    #In your SMILES file,the first line must be String"SMILES Name",or it will not operate correctly.
    mols = read_smiles_file("Ethanol.smi")
    if mols:
        print(f"SMILES:{Chem.MolToSmiles(mols[0])}, Number of atoms:{mols[0].GetNumAtoms()}, Number of bonds:{mols[0].GetNumBonds()}")

    #Real-world use case:
    #A researcher wants to get a list of molecule objects from a SMILES file.
    mols = rt.read_smiles_file("Multiple_Molecules.smi")
    if mols:
        for i,mol in enumerate(mols):
            print(f"Molecule {i+1} is {Chem.MolToSmiles(mol)}")
    """
    return [mol for mol in SmilesMolSupplier(filename) if mol is not None]


def read_sdf_gzip(filename: str) -> list:
    """
    Reads molecules from a compressed SDF file.

    Parameters:
    filename (str): Path to the .sdf.gz file.

    Returns:
    list[rdkit.Chem.rdchem.Mol]: List of RDKit molecule objects.

    #Example
    #Example1:Get Ethanol Mol objects from .sdf.gz file.
    mols = read_sdf_gzip("Ethanol.sdf.gz")
    if mols:
        print(f"SMILES:{Chem.MolToSmiles(mols[0])},Number of atoms:{mols[0].GetNumAtoms()},Number of bonds:{mols[0].GetNumBonds()}")

    #Real-World use case:
    #A researcher wants to get a list of molecule objects from a .sdf.gz file.
    mols = read_sdf_gzip("Multiple_Molecules.sdf.gz")
    if mols:
        print(f"Number of molecules:{len(mols)}")
        for i,mol in enumerate(mols):
            print(f"Molecule {i+1} is {Chem.MolToSmiles(mol)}")
    """
    import gzip
    with gzip.open(filename, 'rb') as inf:
        return [mol for mol in ForwardSDMolSupplier(inf) if mol is not None]


def mol_to_smiles(mol: Chem.Mol, isomeric: bool = True, kekule: bool = False) -> str:
    """
    Converts an RDKit molecule to a SMILES string.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
    isomeric (bool): Whether to include stereochemistry information.
    kekule (bool): Whether to output the Kekule form.

    Returns:
    str: The SMILES representation of the molecule.

    #Example
    #Example1:Get SMILES String of Benzene from Mol object in default parameters.
    mol = Chem.MolFromSmiles("c1ccccc1") # Generate Benzene's Mol object through RDkit
    smiles = mol_to_smiles(mol)
    if smiles:
        print(f"SMILES of Benzene:{smiles}")

    #Example2:Get SMILES String of Benzene from Mol object in Kekule form and without stereochemistry information.
    mol = Chem.MolFromSmiles("c1ccccc1")
    smiles = mol_to_smiles(mol,isomeric=False,kekule=True)
    if smiles:
        print(f"SMILES of Benzene:{smiles}")

    #Real-World use case:
    #A researcher wants to get a SMILES string from a Mol object in Kekule form.
    mol = Chem.MolFromSmiles("c1ccc(cc1)Cl") # Chlorobenzene
    smiles = mol_to_smiles(mol,isomeric=False,kekule=True)
    if smiles:
        print(f"SMILES:{smiles}")
    """
    if kekule:
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def mol_to_molblock(mol: Chem.Mol) -> str:
    """
    Converts an RDKit molecule to a Mol block string.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
    str: The Mol block representation of the molecule.

    #Example:
    #Example1:Get Mol block of Benzene from Mol object.
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol_block = mol_to_molblock(mol)
    if mol_block:
        print(f"Mol block information:{mol_block}")

    #Real-World use case:
    #A researcher wants to get spacial information of a molecule from Mol block.
    mol = Chem.MolFromSmiles("CCO")
    mol_block = mol_to_molblock(mol)
    if mol_block:
        print(f"Mol block information:{mol_block}")
    """
    return Chem.MolToMolBlock(mol)


def add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Adds explicit hydrogen atoms to a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
    rdkit.Chem.rdchem.Mol: The modified molecule with explicit hydrogen atoms.

    #Example:
    #Example1:Add Hydrogen atoms to Ethanol.
    mol = Chem.MolFromSmiles("CCO")
    mol_with_hydrogens = add_hydrogens(mol)
    if mol_with_hydrogens:
        print(f"Number of atoms before adding hydrogens:{mol.GetNumAtoms()}\n"
              f"Number of atoms after adding hydrogens:{mol_with_hydrogens.GetNumAtoms()}")

    #Real-World use case:
    #A researcher wants to add hydrogen atoms to a molecule to see the whole structure.
    mol = Chem.MolFromSmiles("c1ccccc1") # Benzene
    mol_with_hydrogens = add_hydrogens(mol)
    if mol_with_hydrogens:
        print(f"Number of atoms with hydrogen atoms:{mol_with_hydrogens.GetNumAtoms()}")
    """
    return Chem.AddHs(mol)


def remove_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Removes explicit hydrogen atoms from a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
    rdkit.Chem.rdchem.Mol: The modified molecule without explicit hydrogen atoms.

    #Example:
    #Example1:Remove Hydrogen atoms from Ethanol.
    mol_with_hydrogens = Chem.AddHs(Chem.MolFromSmiles("CCO")) # Generate Ethanol with Hydrogen atoms using Rdkit
    mol_without_hydrogens = remove_hydrogens(mol_with_hydrogens)
    if mol_without_hydrogens:
        print(f"Number of atoms with hydrogen atoms:{mol_with_hydrogens.GetNumAtoms()}\n"
            f"Number of atoms without hydrogen atoms:{mol_without_hydrogens.GetNumAtoms()}")

    #Real-World use case:
    #A researcher wants to remove hydrogen atoms from a molecule to see the core structure.
    mol_with_hydrogens = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")) # Benzene
    mol_without_hydrogens = remove_hydrogens(mol_with_hydrogens)
    if mol_without_hydrogens:
        print(f"Number of atoms with hydrogen atoms:{mol_with_hydrogens.GetNumAtoms()}\n"
              f"Number of atoms without hydrogen atoms:{mol_without_hydrogens.GetNumAtoms()}")
    """
    return Chem.RemoveHs(mol)


def kekulize_molecule(mol: Chem.Mol):
    """
    Kekulizes a molecule, converting aromatic bonds to alternating single/double bonds.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    #Example:
    #Example1:Kekulize a Benzene ring and show its bond type change.
    mol = Chem.MolFromSmiles("c1ccccc1")
    print("before kekulization:")
    for bond in mol.GetBonds():
        print(f"{bond.GetBondType()}")
    kekulize_molecule(mol)
    print("after kekulization:")
    for bond in mol.GetBonds():
        print(f"{bond.GetBondType()}")

    #Real-World use case:
    #When you are processing a aromatic compound, you want to kekulize it to study its chemical reactivity.
    Furan = Chem.MolFromSmiles("o1cccc1")
    kekulize_molecule(Furan)
    for bond in Furan.GetBonds():
        print(f"{bond.GetBondType()}")
    """
    Chem.Kekulize(mol)


def sanitize_molecule(mol: Chem.Mol):
    """
    Sanitizes a molecule, ensuring valence rules and aromaticity are properly applied.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    #Example:
    #Example1:Sanitize Ethanol to ensure its structure is valid.
    mol = Chem.MolFromSmiles("CCO")
    try:
        rt.sanitize_molecule(mol)
        print(f"Sanitize successfully,SMILES:{Chem.MolToSmiles(mol)}")
    except Exception as e:
        print(f"Sanitize failed,error message:{e}")

    #Real-World use case:
    #A researcher wants to sanitize a compound to ensure its structure is valid.
    mol = Chem.MolFromSmiles("c1ccccc1") # Benzene
    try:
        rt.sanitize_molecule(mol)
        print(f"Sanitize successfully,SMILES:{Chem.MolToSmiles(mol)}")
    except Exception as e:
        print(f"Sanitize failed,error message:{e}")
    """
    Chem.SanitizeMol(mol)


def compute_2d_coords(mol: Chem.Mol):
    """
    Computes 2D coordinates for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    #Example:
    #Example1:Compute 2D coordinates for Benzene and get its x,y coordinates.
    mol = Chem.MolFromSmiles("c1ccccc1")
    compute_2d_coords(mol)
    positions = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if positions:
        for i,pos in enumerate(positions):
            print(f"Atom {i+1} position in Benzene is:(x={pos.x},y={pos.y})")

    #Real-World use case:
    #A researcher wants to get 2D coordinates of a molecule to visualize it.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") #Aspirin
    compute_2d_coords(mol)
    positions = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if positions:
        for i,pos in enumerate(positions):
            print(f"Atom {i+1} position is:(x={pos.x},y={pos.y})")
    """
    AllChem.Compute2DCoords(mol)


def compute_3d_coords(mol: Chem.Mol, random_seed: int = 0xf00d):
    """
    Computes 3D coordinates for a molecule using ETKDG method.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
    random_seed (int): Random seed for reproducibility.

    Returns:
    int: Status of the embedding (0 for success, -1 for failure).

    #Example:
    #Example1:Compute 3D coordinates for Benzene without hydrogen atoms and get its x,y,z coordinates.
    mol = Chem.MolFromSmiles("c1ccccc1") #Benzene
    compute_3d_coords(mol)
    coordinates = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if coordinates:
        for i,coord in enumerate(coordinates):
            print(f"Atom {i+1} coordinates are:(x={coord.x},y={coord.y},z={coord.z})")

    #Example2:Compute 3D coordinates for Benzene with hydrogen atoms and get its x,y,z coordinates.
    mol_with_hydrogens =Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")) #Benzene
    compute_3d_coords(mol_with_hydrogens)
    coordinates = [mol_with_hydrogens.GetConformer().GetAtomPosition(i) for i in range(mol_with_hydrogens.GetNumAtoms())]
    if coordinates:
        for i,coord in enumerate(coordinates):
            print(f"Atom {i+1} coordinates are:(x={coord.x},y={coord.y},z={coord.z})")

    #Real-World use case:
    #A researcher wants to get core 3D coordinates of a molecule to visualize it.
    mol =Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") #Aspirin
    compute_3d_coords(mol)
    coordinates = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if coordinates:
        for i,coord in enumerate(coordinates):
            print(f"Atom {i+1} coordinates are:(x={coord.x},y={coord.y},z={coord.z})")
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    return AllChem.EmbedMolecule(mol, params)


def write_sdf_file(filename: str, mols: list):
    """
    Writes molecules to an SDF file.

    Parameters:
    filename (str): Path to the output .sdf file.
    mols (list[rdkit.Chem.rdchem.Mol]): List of molecules to write.

    #Example:
    #Example1:Write a list of molecules to an SDF file.
    SMILES_list = ["CCO","c1ccccc1","CC(=O)OC1=CC=CC=C1C(O)=O"]
    Mol_list = [Chem.MolFromSmiles(smiles) for smiles in SMILES_list]
    write_sdf_file("Multiple_Molecules.sdf",Mol_list)

    #Real-World use case:
    #A researcher wants to save a list of molecules to an SDF file.
    SMILES_list = ["C","N","O"]
    Mol_list = [Chem.MolFromSmiles(smiles) for smiles in SMILES_list]
    write_sdf_file("Multiple_Molecules.sdf",Mol_list)
    """
    with SDWriter(filename) as writer:
        for mol in mols:
            writer.write(mol)

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


def mol_from_png_string(png_data: bytes) -> Chem.Mol:
    """
    Extracts a molecule from PNG image metadata.

    Parameters:
    png_data (bytes): PNG image data containing molecule metadata.

    Returns:
    rdkit.Chem.rdchem.Mol: The extracted molecule.

    #Function for generating a PNG image with metadata containing a molecule:
    def Generate_PNG_with_molecule_metadata(mol:Chem.Mol, filename: str):  # input mol list, output filename
      with open(filename, "wb") as f:
        d2d = Draw.MolDraw2DCairo(300, 300)
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        PNG_data = d2d.GetDrawingText()
        PNG_string = Chem.MolMetadataToPNGString(mol, PNG_data, includePkl=True, includeSmiles=True, includeMol=True)
        f.write(PNG_string)
     f.close()

    #Example:
    #Example1:Extract Ethanol from a PNG image with metadata.
    filename = "Ethanol.png"
    Generate_PNG_with_molecule_metadata(Chem.MolFromSmiles("CCO"),filename)
    with open(filename, "rb") as f:
        png_data = f.read()
    mol = mol_from_png_string(png_data)
    if mol:
        print(f"SMILES:{Chem.MolToSmiles(mol)}")

    #Real-World use case:
    #A researcher wants to extract a molecule from a PNG image with metadata.
    filename = "Benzene.png"
    Generate_PNG_with_molecule_metadata(Chem.MolFromSmiles("c1ccccc1"),filename) #Generate PNG image with Benzene metadata
    with open(filename, "rb") as f:
        png_data = f.read()
    mol = mol_from_png_string(png_data)
    if mol:
        print(f"SMILES:{Chem.MolToSmiles(mol)}")
    """
    return Chem.MolFromPNGString(png_data)


def mols_from_png_string(png_data: bytes) -> tuple:
    """
    Extracts multiple molecules from PNG image metadata.

    Parameters:
    png_data (bytes): PNG image data containing multiple molecule metadata.

    Returns:
    list[rdkit.Chem.rdchem.Mol]: List of extracted molecules.

    #Function for generating a PNG image with metadata containing multiple molecules:
    def Generate_PNG_with_molecules_metadata(mols:list,filename: str = None):
    # input a mol list, output PNG String.If filename is provided,PNG String will be written to file and return None.
    d2d = Draw.MolDraw2DCairo(300, 300)
    d2d.FinishDrawing()
    PNG_String = d2d.GetDrawingText()
    for mol in mols:
        PNG_String = Chem.MolMetadataToPNGString(mol, PNG_String)
    if filename:
        with open(filename, "wb") as f:
            f.write(PNG_String)
        f.close()
        return None
    return PNG_String

    #Example:
    #Example1:Extract Ethanol and Benzene from a PNG image metadata.
    mol_list = [Chem.MolFromSmiles("CCO"),Chem.MolFromSmiles("c1ccccc1")]
    PNG_String = Generate_PNG_with_molecules_metadata(mol_list)
    mols = mols_from_png_string(PNG_String)
    if mols:
        for i in range(len(mols)):
            print(f"Molecule {i+1} is {Chem.MolToSmiles(mols[i])}")
    else:
        print("There is no molecule in the PNG string")

    #Real-World use case:
    #You have a PNG image with metadata containing multiple molecules, you want to extract them.
    #Write two molecules' metadata to a PNG file.
    mol_list = [Chem.MolFromSmiles("CCO"),Chem.MolFromSmiles("c1ccccc1")]
    Generate_PNG_with_molecules_metadata(mol_list, "Multiple_Molecules.png")

    with open("Multiple_Molecules.png", "rb") as f:
        PNG_String = f.read()
    mols = mols_from_png_string(PNG_String)
    if mols:
        for i in range(len(mols)):
            print(f"Molecule {i+1} is {Chem.MolToSmiles(mols[i])}")
    else:
        print("There is no molecule in the PNG file")
    """
    return Chem.MolsFromPNGString(png_data)


def delete_substructure(mol: Chem.Mol, substructure: Chem.Mol) -> Chem.Mol:
    """
    Removes a substructure from a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    substructure (rdkit.Chem.rdchem.Mol): The substructure to remove.

    Returns:
    rdkit.Chem.rdchem.Mol: The modified molecule.

    #Example:
    #Example1:Delete -OH substructure from Ethanol.
    Ethanol_compound = Chem.MolFromSmiles("CCO")
    query = Chem.MolFromSmarts("[OH]")
    Compound_after_deletion = delete_substructure(Ethanol_compound,query)
    if Compound_after_deletion:
        print(f"After deletion, the compound is {Chem.MolToSmiles(Compound_after_deletion)}")

    #Example2:Delete -CH3 substructure from Toluene.
    Toluene_compound = Chem.MolFromSmiles("c1ccccc1C")
    query = Chem.MolFromSmarts("[CH3]")
    Compound_after_deletion = delete_substructure(Toluene_compound,query)
    if Compound_after_deletion:
        print(f"After deletion, the compound is {Chem.MolToSmiles(Compound_after_deletion)}")

    #Real-World use case:
    #You want to delete a substructure from a drug molecule to test its effectiveness.
    Iburofeno_compound = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    query = Chem.MolFromSmarts("C(=O)[OH]")
    Compound_after_deletion = delete_substructure(Iburofeno_compound,query)
    if Compound_after_deletion:
        print(f"After deletion, the compound is {Chem.MolToSmiles(Compound_after_deletion)}")
    """
    return AllChem.DeleteSubstructs(mol, substructure)


def replace_substructure(mol: Chem.Mol, substructure: Chem.Mol, replacement: Chem.Mol) -> Chem.Mol:
    """
    Replaces a substructure in a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    substructure (rdkit.Chem.rdchem.Mol): The substructure to replace.
    replacement (rdkit.Chem.rdchem.Mol): The replacement structure.

    Returns:
    rdkit.Chem.rdchem.Mol: The modified molecule.

    #Example:
    #Example1:Replace -CH2OH with -COOH in Ethanol to form Acetic acid.
    Ethanol_compound = Chem.MolFromSmiles("CCO")
    substructure = Chem.MolFromSmarts("[CH2][OH]")
    Replacement = Chem.MolFromSmarts("C(=O)[OH]")
    Compound_after_replacement = replace_substructure(Ethanol_compound,substructure,Replacement)
    if Compound_after_replacement:
        print(f"After replacement, the compound is {Chem.MolToSmiles(Compound_after_replacement)}")

    #Example2:Replace -CH3 with -OH in Toluene to form Phenol.
    Toluene_compound = Chem.MolFromSmiles("c1ccccc1C")
    substructure = Chem.MolFromSmarts("[CH3]")
    Replacement = Chem.MolFromSmarts("[OH]")
    Compound_after_replacement = replace_substructure(Toluene_compound,substructure,Replacement)
    if Compound_after_replacement:
        print(f"After replacement, the compound is {Chem.MolToSmiles(Compound_after_replacement)}")

    #Real-World use case:
    #You want to replace a substructure in a drug molecule to explore the change of properties.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Replace the -COOH with -OH in Aspirin
    substructure = Chem.MolFromSmarts("C(=O)[OH]")
    Replacement = Chem.MolFromSmarts("[OH]")
    Compound_after_replacement = replace_substructure(Aspirin_compound,substructure,Replacement)
    if Compound_after_replacement:
        print(f"After replacement, the compound is {Chem.MolToSmiles(Compound_after_replacement)}")
    """
    return AllChem.ReplaceSubstructs(mol, substructure, replacement)[0]


def replace_sidechains(mol: Chem.Mol, core: Chem.Mol) -> Chem.Mol:
    """
    Removes sidechains from a molecule based on a core.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    core (rdkit.Chem.rdchem.Mol): The core structure.

    Returns:
    rdkit.Chem.rdchem.Mol: The molecule with sidechains removed.

    #Example:
    #Example1:Remove sidechains from Glycol as carbon chain is the core.
    Glycol_compound = Chem.MolFromSmiles("C(O)CO")
    Core = Chem.MolFromSmiles("CC")
    Compound_removed_sidechains = replace_sidechains(Glycol_compound,Core)
    if Compound_removed_sidechains:
        print(f"After replace sidechains, the compound is {Chem.MolToSmiles(Compound_removed_sidechains)}")
    else:
        print("Failed to remove sidechains")

    #Example2:Remove sidechains from a core does not exist in the molecule.
    Glycol_compound = Chem.MolFromSmiles("C(O)CO")
    Core = Chem.MolFromSmiles("N#CC#N")
    Compound_removed_sidechains = rt.replace_sidechains(Glycol_compound,Core)
    if Compound_removed_sidechains:
        print(f"After replace sidechains, the compound is {Chem.MolToSmiles(Compound_removed_sidechains)}")
    else:
        print("Failed to remove sidechains")

    #Real-World use case:
    #When studying a compound with multiple functional groups,you want to remove them to focus on the core structure.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    Core = Chem.MolFromSmiles("c1ccccc1")
    Compound_removed_sidechains = rt.replace_sidechains(Aspirin_compound,Core)
    if Compound_removed_sidechains:
        print(f"After replace sidechains, the compound is {Chem.MolToSmiles(Compound_removed_sidechains)}")
    else:
        print("Failed to remove sidechains")
    """
    return AllChem.ReplaceSidechains(mol, core)


def replace_core(mol: Chem.Mol, core: Chem.Mol, label_by_index: bool = False) -> Chem.Mol:
    """
    Removes the core of a molecule, leaving labeled sidechains.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    core (rdkit.Chem.rdchem.Mol): The core structure to remove.
    label_by_index (bool): Whether to label sidechains based on attachment index.

    Returns:
    rdkit.Chem.rdchem.Mol: The remaining sidechains.

    #Example:
    #Example1:Remove the core of Glycol as carbon chain is the core.
    Glycol_compound = Chem.MolFromSmiles("C(O)CO")
    Core = Chem.MolFromSmiles("CC")
    Compound_removed_core = rt.replace_core(Glycol_compound,Core)
    if Compound_removed_core:
        print(f"After replace core, the compound is {Chem.MolToSmiles(Compound_removed_core)}")
    else:
        print("Failed to remove core")

    #Example2:Remove a core does not exist in Glycol.
    Glycol_compound = Chem.MolFromSmiles("C(O)CO")
    Core = Chem.MolFromSmiles("N#CC#N")
    Compound_removed_core = rt.replace_core(Glycol_compound,Core)
    if Compound_removed_core:
        print(f"After replace core, the compound is {Chem.MolToSmiles(Compound_removed_core)}")
    else:
        print("Failed to remove core")

    #Real-World use case:
    #When studying a compound with multiple functional groups,you want to remove the core to identify the sidechains.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    Core = Chem.MolFromSmiles("c1ccccc1")
    Compound_removed_core = rt.replace_core(Aspirin_compound,Core)
    if Compound_removed_core:
        print(f"After replace core, the compound is {Chem.MolToSmiles(Compound_removed_core)}")
    else:
        print("Failed to remove core")
    """
    return AllChem.ReplaceCore(mol, core, labelByIndex=label_by_index)


def get_molecule_fragments(mol: Chem.Mol) -> list:
    """
    Splits a molecule into fragments.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    list[rdkit.Chem.rdchem.Mol]: List of fragments as molecules.

    Example:
    #Example1:Test single molecule (SMILES 'CCO')
    mol = Chem.MolFromSmiles("CCO")
    frags = get_molecule_fragments(mol)
    if frags:
        print(f'Found {len(frags)} fragments')

    #Example2:Test diconnected molecule (SMILES 'CCO.CC')
    mol = Chem.MolFromSmiles("CCO.CC")
    frags = get_molecule_fragments(mol)
    if frags:
        print(f'Found {len(frags)} fragments')

    #Real-World use case:
    #A researcher wants to split a mixture into fragments to analyze its structure.
    mol = Chem.MolFromSmiles("c1ccccc1.CC(=O)N") #Benzene and Acetamide mixture
    frags = rt.get_molecule_fragments(mol)
    if frags:
        print(f'Found {len(frags)} fragments')
        for i,frag in enumerate(frags):
            print(f'Fragment {i+1} is {Chem.MolToSmiles(frag)}')
    """

    return list(AllChem.GetMolFrags(mol, asMols=True))


def murcko_scaffold(mol: Chem.Mol) -> Chem.Mol:
    """
    Extracts the Murcko scaffold from a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    rdkit.Chem.rdchem.Mol: The extracted scaffold.

    #Example:
    #Example1:Extract Murcko scaffold from Toluene.
    mol = Chem.MolFromSmiles("c1ccccc1C") #Toluene
    Murcko_scaffold = murcko_scaffold(mol)
    if Murcko_scaffold:
        print(f'Murcko scaffold for {Chem.MolToSmiles(mol)} is {Chem.MolToSmiles(scaffold)}')

    #Real-World use case:
    #A researcher wants to extract the Murcko scaffold from Aspirin to identify the core structure.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") #Aspirin
    Murcko_scaffold = murcko_scaffold(mol)
    if Murcko_scaffold:
        print(f'Murcko scaffold for {Chem.MolToSmiles(mol)} is {Chem.MolToSmiles(scaffold)}')
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return MurckoScaffold.GetScaffoldForMol(mol)


def draw_highlighted_substructure(mol: Chem.Mol, query: Chem.Mol, filename: str): # picture related
    """
    Highlights a substructure in a molecule and saves the image.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    query (rdkit.Chem.rdchem.Mol): The substructure query.
    filename (str): Output image file path.

    Returns:
    None

    #Example:
    #Example1:Highlight Benzene ring in Toluene.
    mol = Chem.MolFromSmiles("c1ccc(cc1)C") # Toluene
    query = Chem.MolFromSmarts("c1ccc(cc1)") # Benzene ring
    draw_highlighted_substructure(mol, query,filename='highlighted_toluene.png')

    #Example2:Highlight Hydroxyl groups in Ethanol.
    mol = Chem.MolFromSmiles("CCO") # Ethanol
    query = Chem.MolFromSmarts("[OH]") # Hydroxyl group
    draw_highlighted_substructure(mol, query,filename='highlighted_ethanol.png')

    #Real-World use case:
    #A drug developer highlights the molecular substructure to visualize the differences between different compounds.
    mol = Chem.MolFromSmiles("c1cccnc1CC")
    query = Chem.MolFromSmarts("c1cccnc1")
    draw_highlighted_substructure(mol, query,filename='highlighted_pyridine.png')
    """
    hit_ats = list(mol.GetSubstructMatch(query))
    hit_bonds = [mol.GetBondBetweenAtoms(hit_ats[b.GetBeginAtomIdx()], hit_ats[b.GetEndAtomIdx()]).GetIdx() for b in query.GetBonds()]

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=hit_ats, highlightBonds=hit_bonds)
    drawer.WriteDrawingText(filename)


def highlight_multiple_substructures(mol: Chem.Mol, queries: list, filename: str): # picture related
    """
    Highlights multiple substructures in a molecule with different colors and saves the image.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    queries (list[rdkit.Chem.rdchem.Mol]): List of substructure queries.
    filename (str): Output image file path.

    Returns:
    None

    #Example:
    #Example1:Highlight Benzene ring and Methyl in Toluene.
    mol = Chem.MolFromSmiles("c1ccc(cc1)C") # Toluene
    queries = [
        Chem.MolFromSmarts("c1ccc(cc1)"), # Benzene ring
        Chem.MolFromSmarts("[CH3]") # Methyl group
    ]
    highlight_multiple_substructures(mol, queries, "highlighted_toluene.png")

    #Example2:Highlight Benzene ring,Carboxyl,Methyl in 4-Methylbenzoic acid.
    mol = Chem.MolFromSmiles("O=C(O)c1ccc(cc1)C") # 4-Methylbenzoic acid
    queries = [
        Chem.MolFromSmarts("c1ccc(cc1)"), # Benzene ring
        Chem.MolFromSmarts("[C,c](=O)[O,o]"), # Carboxyl
        Chem.MolFromSmarts("[CH3]") # Methyl
    ]
    highlight_multiple_substructures(mol, queries, "highlighted_4-Methylbenzoic_acid_molecule.png")

    #Real-World use case:
    #A drug developer highlights multiple molecular substructure to visualize the differences between different compounds.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=C(C=C1)C(=O)O") # Aspirin
    queries = [
        Chem.MolFromSmarts("C(=O)C"), # Acetyl
        Chem.MolFromSmarts("[C,c](=O)[OH]"), # Carboxyl
    ]
    highlight_multiple_substructures(mol, queries, "highlighted_aspirin.png")
    """
    atom_colors = {}
    bond_colors = {}
    colors = [(0.8, 0.0, 0.8), (0.8, 0.8, 0), (0, 0.8, 0.8), (0, 0, 0.8)]

    for i, query in enumerate(queries):
        hit_ats = list(mol.GetSubstructMatch(query))
        hit_bonds = [mol.GetBondBetweenAtoms(hit_ats[b.GetBeginAtomIdx()], hit_ats[b.GetEndAtomIdx()]).GetIdx() for b in query.GetBonds()]
        
        for at in hit_ats:
            atom_colors[at] = colors[i % 4]
        for bd in hit_bonds:
            bond_colors[bd] = colors[3 - (i % 4)]

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=atom_colors.keys(), highlightAtomColors=atom_colors, highlightBonds=bond_colors.keys(), highlightBondColors=bond_colors)
    drawer.WriteDrawingText(filename)


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFMCS, rdRascalMCES


def find_mcs(mols: list, ring_matches_ring_only: bool = False, complete_rings_only: bool = False, timeout: int = 0) -> str:
    """
    Finds the Maximum Common Substructure (MCS) for a list of molecules.

    Parameters:
    mols (list[rdkit.Chem.rdchem.Mol]): List of molecules to compare.
    ring_matches_ring_only (bool): If True, ensures that ring atoms only match ring atoms.
    complete_rings_only (bool): If True, ensures that the MCS contains complete rings.
    timeout (int): Timeout in seconds for the search.

    Returns:
    str: SMARTS string representing the MCS.

    #Example:
    #Example1:Test Ethane and Benzene.
    mol1 = Chem.MolFromSmiles("c1ccccc1") # Benzene
    mol2 = Chem.MolFromSmiles("CC") # Ethane
    mols = [mol1,mol2]
    MCS = find_mcs(mols)
    if MCS:
        print(f'MCS for {Chem.MolToSmiles(mol1)} and {Chem.MolToSmiles(mol2)} is {Chem.MolToSmiles(Chem.MolFromSmarts(MCS))}')
    else:
        print("No MCS found")

    #Example2:Test Ethane and Benzene and ensures that ring atoms only match ring atoms.
    mol1 = Chem.MolFromSmiles("c1ccccc1") # Benzene
    mol2 = Chem.MolFromSmiles("CC") # Ethane
    mols = [mol1,mol2]
    MCS = find_mcs(mols,ring_matches_ring_only=True)
    if MCS:
        print(f'MCS for {Chem.MolToSmiles(mol1)} and {Chem.MolToSmiles(mol2)} is {Chem.MolToSmiles(Chem.MolFromSmarts(MCS))}')
    else:
        print("No MCS found")

    #Example3:Test Cyclopropane and Benzene acid.
    mol1 = Chem.MolFromSmiles("c1ccccc1") # Benzene
    mol2 = Chem.MolFromSmiles("C1CC1") # Cyclopropane
    mols = [mol1,mol2]
    MCS = find_mcs(mols)
    if MCS:
        print(f'MCS for {Chem.MolToSmiles(mol1)} and {Chem.MolToSmiles(mol2)} is {Chem.MolToSmiles(Chem.MolFromSmarts(MCS))}')
    else:
        print("No MCS found")

    #Example4:Test Cyclopropane and Benzene acid and ensures that the MCS contains complete rings.
    mol1 = Chem.MolFromSmiles("c1ccccc1") # Benzene
    mol2 = Chem.MolFromSmiles("C1CC1") # Cyclopropane
    mols = [mol1,mol2]
    MCS = find_mcs(mols,complete_rings_only=True)
    if MCS:
        print(f'MCS for {Chem.MolToSmiles(mol1)} and {Chem.MolToSmiles(mol2)} is {Chem.MolToSmiles(Chem.MolFromSmarts(MCS))}')
    else:
        print("No MCS found")

    #Real-World use case:
    #A drug developer finds a common structure in a series of molecules derived from a lead compound and ensures that the MCS contains complete rings.
    mol1 = Chem.MolFromSmiles("c1ccc(cc1)C(=O)O")
    mol2 = Chem.MolFromSmiles("C1CC1C(=O)O")
    mol3 = Chem.MolFromSmiles("C1CCC1C(=O)O")
    mols = [mol1,mol2,mol3]
    MCS = find_mcs(mols,complete_rings_only=False)
    if MCS:
        print(f'MCS for {Chem.MolToSmiles(mol1)} and {Chem.MolToSmiles(mol2)} is {Chem.MolToSmiles(Chem.MolFromSmarts(MCS))}')
    else:
        print("No MCS found")
    """
    return rdFMCS.FindMCS(mols, ringMatchesRingOnly=ring_matches_ring_only, completeRingsOnly=complete_rings_only, timeout=timeout).smartsString


def find_mces(mol1: Chem.Mol, mol2: Chem.Mol) -> list:
    """
    Finds the Maximum Common Edge Substructure (MCES) between two molecules.

    Parameters:
    mol1 (rdkit.Chem.rdchem.Mol): First molecule.
    mol2 (rdkit.Chem.rdchem.Mol): Second molecule.

    Returns:
    list: List of RascalResult objects containing MCES details.

    #Example:
    #Example1:Test two molecules with MCES.
    mol1 = Chem.MolFromSmiles("OC(=O)CO") # Glycolic acid
    mol2 = Chem.MolFromSmiles("OC(=O)CCO") # 3-Hydroxypropionic acid
    MCES = find_mces(mol1, mol2)
    if MCES:
        for result in MCES:
            print(f"MCES SMART: {result.smartsString},atom matches: {result.atomMatches()},bond matches: {result.bondMatches()}")
    else:
        print("No MCES found")

    #Example2:Test two molecules without MCES.
    mol1 = Chem.MolFromSmiles("OC(=O)CO") # Glycolic acid
    mol2 = Chem.MolFromSmiles("COC(=O)") # Methyl formate
    MCES = find_mces(mol1, mol2)
    if MCES:
        for result in MCES:
            print(f"MCES SMART: {result.smartsString},atom matches: {result.atomMatches()},bond matches: {result.bondMatches()}")
    else:
        print("No MCES found")

    #Real-World use case:
    #A researcher found that there are two compounds with similar properties but slightly different structures, he need to find the MCES of these two compounds to determine their common structural features.
    mol1 = Chem.MolFromSmiles("OC(=O)C[NH]CO") # Two molecules both with Carboxyl and Imino group, but different bonding patterns.
    mol2 = Chem.MolFromSmiles("OC(=O)[NH]CCO")
    MCES = find_mces(mol1, mol2)
    if MCES:
        for result in MCES:
            print(f"MCES SMART: {result.smartsString},atom matches: {result.atomMatches()},bond matches: {result.bondMatches()}")
    else:
        print("No MCES found")
    """
    return rdRascalMCES.FindMCES(mol1, mol2)


def find_mces_with_options(mol1: Chem.Mol, mol2: Chem.Mol, similarity_threshold: float = 0.5, min_frag_size: int = -1, complete_aromatic_rings: bool = True) -> list:
    """
    Finds the Maximum Common Edge Substructure (MCES) between two molecules with additional options.

    Parameters:
    mol1 (rdkit.Chem.rdchem.Mol): First molecule.
    mol2 (rdkit.Chem.rdchem.Mol): Second molecule.
    similarity_threshold (float): Minimum similarity threshold for MCES detection.
    min_frag_size (int): Minimum fragment size to consider in MCES.
    complete_aromatic_rings (bool): Whether to require complete aromatic rings.

    Returns:
    list: List of RascalResult objects containing MCES details.

    #Example:
    #Example1:Test two molecules with different similarity thresholds.
    mol1 = Chem.MolFromSmiles("OC(=O)C[NH]CO") # Methyl Carbamate
    mol2 = Chem.MolFromSmiles("OC(=O)[NH]CCO") # Methyl Aminopropionate
    thresholds = [0.8,0.85,0.9]
    for threshold in thresholds:
        MCES = find_mces_with_options(mol1, mol2,similarity_threshold=threshold)
        if MCES:
            for result in MCES:
                print(f"MCES SMART: {result.smartsString},atom matches: {result.atomMatches()},bond matches: {result.bondMatches()}")
        else:
            print("No MCES found")
    #Example2:Test two molecules with different minimum fragment sizes.
    mol1 = Chem.MolFromSmiles("c1ccc(cc1)C(=O)O") # Benzoic acid
    mol2 = Chem.MolFromSmiles("c1ccc(cc1)C(=O)OC") # Methyl benzoate
    min_frag_sizes = [1,5,10]
    for min_frag_size in min_frag_sizes:
        MCES = find_mces_with_options(mol1, mol2,min_frag_size=min_frag_size)
        if MCES:
            for result in MCES:
                print(f"MCES SMART: {result.smartsString},atom matches: {result.atomMatches()},bond matches: {result.bondMatches()}")
        else:
            print("No MCES found")

    #Real-World use case:
    #A drug developer who encounters a series of molecules with similar properties in the course of research needs to analyze under a certain similarity threshold and the minimum number of molecular fragments.
    mol1 = Chem.MolFromSmiles("c1ccc(cc1)C(=O)O") # Benzoic acid
    mol2 = Chem.MolFromSmiles("c1ccc(cc1)C(=O)N")
    mol_pairs = [(mol1,mol2)] # You can add more pairs of molecules to compare.
    for mol_pair in mol_pairs:
        MCES = find_mces_with_options(mol_pair[0], mol_pair[1],similarity_threshold=0.7,min_frag_size=5)
        if MCES:
            for result in MCES:
                print(f"MCES SMART: {result.smartsString},atom matches: {result.atomMatches()},bond matches: {result.bondMatches()}")
        else:
            print("No MCES found")
    """
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = similarity_threshold
    opts.minFragSize = min_frag_size
    opts.completeAromaticRings = complete_aromatic_rings
    return rdRascalMCES.FindMCES(mol1, mol2, opts)


def get_rdkit_fingerprint(mol: Chem.Mol, fp_size: int = 2048) -> DataStructs.ExplicitBitVect:
    """
    Computes the RDKit topological fingerprint for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to fingerprint.
    fp_size (int): Fingerprint size in bits.

    Returns:
    rdkit.DataStructs.ExplicitBitVect: The RDKit fingerprint.

    #Example:
    #Example1:Get Fingerprint with default size.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    fingerprint = get_rdkit_fingerprint(mol)
    print(f'Fingerprint size: {len(fingerprint)},Fingerprint preview: {fingerprint.ToBitString()[:32]}')

    #Example2:Get Fingerprint with custom size.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    fingerprint = get_rdkit_fingerprint(mol,fp_size=1024)
    print(f'Fingerprint size: {len(fingerprint)},Fingerprint preview: {fingerprint.ToBitString()[:32]}')

    #Real-World use case:
    #A drug developer uses molecular fingerprints from a library of compounds to screen out molecules that are similar to the target drug.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    compound_library = [mol] # You can add more compounds to the library.
    fingerprints = []
    for i,compound in enumerate(compound_library):
        fingerprints.append(get_rdkit_fingerprint(compound,fp_size=1024))
        print(f'Fingerprint size: {len(fingerprints[i])},Fingerprint preview: {fingerprints[i].ToBitString()[:32]}')

    """
    fpgen = AllChem.GetRDKitFPGenerator(fpSize=fp_size)
    return fpgen.GetFingerprint(mol)


def tanimoto_similarity(fp1, fp2) -> float:
    """
    Computes the Tanimoto similarity between two fingerprints.

    Parameters:
    fp1 (rdkit.DataStructs.ExplicitBitVect): First fingerprint.
    fp2 (rdkit.DataStructs.ExplicitBitVect): Second fingerprint.

    Returns:
    float: Tanimoto similarity score.

    #Example:
    #Example1: Test three fingerprints, two of which are similar to each other, and one of which is dissimilar.
    fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)N")) # fp1 is similar to fp2 but dissimilar to fp3
    fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)O"))
    fp3 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(C)Nc1ccc(cc1)C(=O)N"))
    Similarity_1and2 = tanimoto_similarity(fp1,fp2)
    Similarity_1and3 = tanimoto_similarity(fp1,fp3)
    print(f"Tanimoto similarity between mol1 and mol2: {Similarity_1and2}")
    print(f"Tanimoto similarity between mol1 and mol3: {Similarity_1and3}")

    #Real-World use case:
    #A researcher compares two compounds using tanimoto similarity metrics to determine their similarity.
    fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)N"))
    fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)O"))
    Similarity_1and2 = tanimoto_similarity(fp1,fp2)
    print(f"Similarity between mol1 and mol2: {Similarity_1and2}")
    """
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def dice_similarity(fp1, fp2) -> float:
    """
    Computes the Dice similarity between two fingerprints.

    Parameters:
    fp1 (rdkit.DataStructs.ExplicitBitVect): First fingerprint.
    fp2 (rdkit.DataStructs.ExplicitBitVect): Second fingerprint.

    Returns:
    float: Dice similarity score.

    #Example:
    #Example1: Test three fingerprints, two of which are similar to each other, and one of which is dissimilar.
    fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)N")) # fp1 is similar to fp2 but dissimilar to fp3
    fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)O"))
    fp3 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(C)Nc1ccc(cc1)C(=O)N"))
    Similarity_1and2 = dice_similarity(fp1,fp2)
    Similarity_1and3 = dice_similarity(fp1,fp3)
    print(f"Dice similarity between mol1 and mol2: {Similarity_1and2}")
    print(f"Dice similarity between mol1 and mol3: {Similarity_1and3}")

    #Real-World use case:
    #A researcher compares two compounds using dice similarity metrics to determine their similarity.
    fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)N"))
    fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)S(=O)(=O)O"))
    Similarity_1and2 = dice_similarity(fp1,fp2)
    print(f"dice similarity between mol1 and mol2: {Similarity_1and2}")
    """
    return DataStructs.DiceSimilarity(fp1, fp2)


def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, fp_size: int = 2048) -> DataStructs.ExplicitBitVect:
    """
    Computes the Morgan fingerprint (circular fingerprint) for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to fingerprint.
    radius (int): Radius of the circular environment.
    fp_size (int): Fingerprint size in bits.

    Returns:
    rdkit.DataStructs.ExplicitBitVect: The Morgan fingerprint.

    #Example:
    #Example1:Get Fingerprint for Aspirin with default parameters.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    morgan_fingerprint = get_morgan_fingerprint(mol)
    print(f"Morgan fingerprint size: {len(morgan_fingerprint)},Fingerprint preview: {morgan_fingerprint.ToBitString()[:128]}")

    #Example2:Get Fingerprint for Aspirin with different radius.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    radius = [0,2,4,8]
    for r in radius:
        morgan_fingerprint = get_morgan_fingerprint(mol,radius=r)
        print(f"Radius: {r},Morgan fingerprint size: {len(morgan_fingerprint)},Fingerprint preview: {morgan_fingerprint.ToBitString()[:128]}")

    #Example3:Get Fingerprint for Aspirin with different fingerprint size.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    fp_sizes = [1024,2048,4096]
    for fp_size in fp_sizes:
        morgan_fingerprint = get_morgan_fingerprint(mol,fp_size=fp_size)
        print(f"Fingerprint size:{fp_size},Morgan fingerprint size: {len(morgan_fingerprint)},Fingerprint preview: {morgan_fingerprint.ToBitString()[:128]}")

    #Real-World use case:
    #A researcher wants to get a morgan fingerprint for a compound with a specific radius and fingerprint size.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    morgan_fingerprint = get_morgan_fingerprint(mol,radius=4,fp_size=4096)
    print(f"Morgan fingerprint size: {len(morgan_fingerprint)},Fingerprint preview: {morgan_fingerprint.ToBitString()[:128]}")
    """
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return fpgen.GetFingerprint(mol)


def get_maccs_keys_fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """
    Computes the MACCS keys fingerprint for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to fingerprint.

    Returns:
    rdkit.DataStructs.ExplicitBitVect: The MACCS keys fingerprint.

    #Example:
    #Example1:Get MACCS keys fingerprint for Aspirin.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    maccs_keys_fingerprint = get_maccs_keys_fingerprint(mol)
    print(f"Length of MACCS keys: {len(maccs_keys_fingerprint)},MACCS keys: {maccs_keys_fingerprint.ToBitString()}")

    #Real-World use case:
    #A researcher wants to get a MACCS keys fingerprint for a compound.
    mol = Chem.MolFromSmiles("CCO") # Ethanol
    maccs_keys_fingerprint = get_maccs_keys_fingerprint(mol)
    print(f"Length of MACCS keys: {len(maccs_keys_fingerprint)},MACCS keys: {maccs_keys_fingerprint.ToBitString()}")
    """
    from rdkit.Chem import MACCSkeys
    return MACCSkeys.GenMACCSKeys(mol)


def get_atom_pair_fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """
    Computes the Atom-Pair fingerprint for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to fingerprint.

    Returns:
    rdkit.DataStructs.ExplicitBitVect: The Atom-Pair fingerprint.

    #Example:
    #Example1:Get Atom-Pair fingerprint for Aspirin.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    atom_pair_fingerprint = get_atom_pair_fingerprint(mol)
    print(f"Length of atom pair fingerprint: {len(atom_pair_fingerprint)},Fingerprint preview: {atom_pair_fingerprint.ToBitString()[:128]}")

    #Real-World use case:
    #A researcher wants to get an atom-pair fingerprint for a compound.
    mol = Chem.MolFromSmiles("CCO") # Ethanol
    atom_pair_fingerprint = get_atom_pair_fingerprint(mol)
    print(f"Length of atom pair fingerprint: {len(atom_pair_fingerprint)},Fingerprint preview: {atom_pair_fingerprint.ToBitString()[:128]}")
    """
    fpgen = AllChem.GetAtomPairGenerator()
    return fpgen.GetFingerprint(mol)


def get_topological_torsion_fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """
    Computes the Topological Torsion fingerprint for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to fingerprint.

    Returns:
    rdkit.DataStructs.ExplicitBitVect: The Topological Torsion fingerprint.

    #Example:
    #Example1:Get Topological Torsion fingerprint for Aspirin.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    torsion_fingerprint = get_topological_torsion_fingerprint(mol)
    print(f"Length of topological torsion fingerprint: {len(torsion_fingerprint)},Fingerprint preview: {torsion_fingerprint.ToBitString()[:128]}")

    #Real-World use case:
    #A researcher wants to get a topological torsion fingerprint for a compound.
    mol = Chem.MolFromSmiles("c1ccccc1") # Benzene
    torsion_fingerprint = get_topological_torsion_fingerprint(mol)
    print(f"Length of topological torsion fingerprint: {len(torsion_fingerprint)},Fingerprint preview: {torsion_fingerprint.ToBitString()[:128]}")
    """
    fpgen = AllChem.GetTopologicalTorsionGenerator()
    return fpgen.GetFingerprint(mol)


def explain_morgan_fingerprint(mol: Chem.Mol, radius: int = 2) -> dict:
    """
    Generates an explanation for the Morgan fingerprint, detailing atom contributions.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule.
    radius (int): Radius of the circular environment.

    Returns:A dictionary mapping bit IDs to contributing atoms and radii.

    #Example:
    #Example1:Explain Morgan fingerprint(Bit ID,Atom ID,Radius) for Aspirin with default parameters.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    explaination = rt.explain_morgan_fingerprint(mol)
    print(f"Length of explaination: {len(explaination)}")
    print("Preview of explaination:")
    for bit_id,info in list(explaination.items())[0:3]:
        print(f"Bit ID: {bit_id}")
        for atom_id, radius in info:
            print(f"  Atom ID: {atom_id}, Radius: {radius}")

    #Example2:Explain Morgan fingerprint(Bit ID,Atom ID,Radius) for Aspirin with different radius.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    radius = [2,4,8]
    for r in radius:
        explaination = rt.explain_morgan_fingerprint(mol,radius=r)
        print(f"Radius: {r},Length of explaination: {len(explaination)}")
        print("Preview of explaination:")
        for bit_id,info in list(explaination.items())[0:3]:
            print(f"Bit ID: {bit_id}")
            for atom_id, radius in info:
                print(f"  Atom ID: {atom_id}, Radius: {radius}")

    #Real-World use case:
    #A researcher wants to get an explanation for a specific Morgan fingerprint bit with a certain radius.
    mol = Chem.MolFromSmiles("c1ccccc1") # Benzene
    explaination = rt.explain_morgan_fingerprint(mol,radius=3)
    print(f"Length of explaination: {len(explaination)}")
    print("Preview of explaination:")
    for bit_id,info in list(explaination.items())[0:3]:
        print(f"Bit ID: {bit_id}")
        for atom_id, radius in info:
            print(f"  Atom ID: {atom_id}, Radius: {radius}")
    """
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    ao = AllChem.AdditionalOutput()
    ao.CollectBitInfoMap()
    _ = fpgen.GetSparseCountFingerprint(mol, additionalOutput=ao)
    return ao.GetBitInfoMap()


def draw_morgan_bit(mol: Chem.Mol, bit_id: int, bit_info: dict, filename: str):#picture related
    """
    Draws an image highlighting the environment of a specific Morgan fingerprint bit.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule.
    bit_id (int): The bit ID to visualize.
    bit_info (dict): Dictionary containing bit information from `explain_morgan_fingerprint`.
    filename (str): Output image file path.

    Returns:
    None
    """
    img = Draw.DrawMorganBit(mol, bit_id, bit_info, useSVG=False)
    img.save(filename)


def draw_rdkit_bit(mol: Chem.Mol, bit_id: int, bit_info: dict, filename: str):# picture related
    """
    Draws an image highlighting the bond path of a specific RDKit fingerprint bit.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule.
    bit_id (int): The bit ID to visualize.
    bit_info (dict): Dictionary containing bit information from RDKit fingerprinting.
    filename (str): Output image file path.

    Returns:
    None
    """
    img = Draw.DrawRDKitBit(mol, bit_id, bit_info, useSVG=False)
    img.save(filename)


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator, rdMolDescriptors,Descriptors
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.Chem.Draw import SimilarityMaps,MolDraw2DSVG


def pick_diverse_molecules(mols: list, num_picks: int, radius: int = 3, seed: int = 23) -> list:
    """
    Selects a diverse set of molecules using the MaxMin algorithm.

    Parameters:
    mols (list[rdkit.Chem.rdchem.Mol]): List of RDKit molecules.
    num_picks (int): Number of diverse molecules to pick.
    radius (int): Radius for Morgan fingerprint generation.
    seed (int): Random seed for reproducibility.

    Returns:
    list[rdkit.Chem.rdchem.Mol]: List of selected diverse molecules.

    #Example:
    #Example1:Select 2 diverse molecules from a list of 6 molecules in different radiuses.
    smiles_list = [
        "CCO",  # Ethanol 乙醇
        "CC(C)O",  # Isopropanol 异丙酸
        "CC(C)C(O)=O",  # Isobutyric acid 异丁酸
        "C1CCCCC1",  # Cyclohexane 环己烷
        "c1ccccc1",  # Benzene 苯
        "C1=CC=C(C=C1)C2=CC=CC=C2"]  # Biphenyl 联苯
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    radius = [0,1,2]
    for r in radius:
            diverse_mol_list = pick_diverse_molecules(mol_list, num_picks=2,radius=r)
            print(f"Radius: {r}, Diverse molecules SMILES: {[Chem.MolToSmiles(mol) for mol in diverse_mol_list]}")

    #Real-World use case:
    #A researcher wants to select a diverse set of molecules from a list of compounds in radius 2.
    smiles_list = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "CC(=O)OC1=CC=CC=C1C(O)=O", # Aspirin
        "C1CCCCC1",  # Cyclohexane
        "c1ccccc1"] # Benzene
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    diverse_mol_list = pick_diverse_molecules(mol_list, num_picks=2,radius=2)
    print(f"Diverse molecules SMILES: {[Chem.MolToSmiles(mol) for mol in diverse_mol_list]}")
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    fps = [fpgen.GetFingerprint(mol) for mol in mols]
    picker = MaxMinPicker()
    pick_indices = picker.LazyBitVectorPick(fps, len(fps), num_picks, seed=seed)
    return [mols[i] for i in pick_indices]


def get_similarity_map(mol: Chem.Mol, ref_mol: Chem.Mol, fp_type: str = 'bv') -> tuple:
    """
    Generates a similarity map highlighting atomic contributions.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): Target molecule.
    ref_mol (rdkit.Chem.rdchem.Mol): Reference molecule.
    fp_type (str): Type of fingerprint ('bv' for bit vector, 'count' for count vector).

    Returns:
    float:Maximum weight.

    #Example:
    #Example1:Get similarity map for Ethanol and Propane.
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("CCC")
    maximum_weight = get_similarity_map(mol1,mol2)
    if maximum_weight:
        print(f"Maximum weight between mol1 and mol2: {maximum_weight}")
    else:
        print("There is no similarity between mol1 and mol2")

    #Example2:Get similarity map for two molecules do not have identical atoms.
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("NN")
    maximum_weight = get_similarity_map(mol1,mol2)
    if maximum_weight:
        print(f"Maximum weight between mol1 and mol2: {maximum_weight}")
    else:
        print("There is no similarity between mol1 and mol2")

    #Real-World use case:
    #When comparing two molecules,you want to know the maximum weight of their similarity map.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    Ibuprofen_compound = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    maximum_weight = rt.get_similarity_map(Aspirin_compound,Ibuprofen_compound)
    if maximum_weight:
        print(f"Maximum weight between mol1 and mol2: {maximum_weight}")
    else:
        print("There is no similarity between mol1 and mol2")
    """
    draw2d = MolDraw2DSVG(400, 400)
    fig, max_weight = SimilarityMaps.GetSimilarityMapForFingerprint(ref_mol, mol, SimilarityMaps.GetMorganFingerprint, fpType=fp_type,draw2d=draw2d)
    return max_weight


def get_atomic_weights(mol: Chem.Mol, ref_mol: Chem.Mol) -> list:
    """
    Retrieves atomic contribution weights for similarity mapping.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): Target molecule.
    ref_mol (rdkit.Chem.rdchem.Mol): Reference molecule.

    Returns:
    list: Atomic contribution weights.

    #Example:
    #Example1:Get atomic contribution weights for Ethanol in similarity mapping Ethanol and Propane.
    mol = Chem.MolFromSmiles("CCO")
    ref_mol = Chem.MolFromSmiles("CCC")
    weights = get_atomic_weights(mol,ref_mol)
    for i in range(len(weights)):
        print(f"Atom {mol.GetAtomWithIdx(i).GetSymbol()} weight: {weights[i]}")

    #Real-World use case:
    #A drug developer wants to get atomic contribution weights for a drug molecule in similarity mapping with another drug molecule.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") #Aspirin
    ref_mol = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O") #Ibuprofen
    weights = get_atomic_weights(mol,ref_mol)
    for i in range(len(weights)):
        print(f"Atom {mol.GetAtomWithIdx(i).GetSymbol()} weight: {weights[i]}")
    """
    return SimilarityMaps.GetAtomicWeightsForFingerprint(ref_mol, mol, SimilarityMaps.GetMorganFingerprint)


def compute_gasteiger_charges(mol: Chem.Mol):
    """
    Computes Gasteiger partial charges for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    None: Modifies the molecule in place.

    #Example:
    #Example1:Compute Gasteiger partial charges for Toluene.
    mol = Chem.MolFromSmiles("c1ccccc1C") # Toluene
    compute_gasteiger_charges(mol)
    for i,atom in enumerate(mol.GetAtoms()):
        Gasteiger_charge = atom.GetProp("_GasteigerCharge")
        print(f"Gasteiger charge for atom {i} {atom.GetSymbol()}: {Gasteiger_charge}")

    #Real-World use case:
    #A researcher wants to compute Gasteiger partial charges for a complex molecule.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    compute_gasteiger_charges(mol)
    for i,atom in enumerate(mol.GetAtoms()):
        Gasteiger_charge = atom.GetProp("_GasteigerCharge")
        print(f"Gasteiger charge for atom {i} {atom.GetSymbol()}: {Gasteiger_charge}")
    """
    AllChem.ComputeGasteigerCharges(mol)


def get_partial_charge(mol: Chem.Mol, atom_idx: int) -> float:
    """
    Retrieves the Gasteiger charge of a specific atom.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    atom_idx (int): Atom index.

    Returns:
    float: The computed partial charge.

    #Example:
    #Example1:Get Gasteiger charge for atom indexed 2 in Aspirin.
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O") # Aspirin
    atom_idx = 2
    compute_gasteiger_charges(mol) # You need to compute Gasteiger charges first
    partial_charge = get_partial_charge(mol,atom_idx)
    if partial_charge:
        print(f"Partial charge for atom {atom_idx} {mol.GetAtomWithIdx(atom_idx).GetSymbol()}: {partial_charge}")

    #Real-World use case:
    #After computing Gasteiger partial charges,you want to retrieve an atom's partial charge.
    mol = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O") # Ibuprofen
    atom_idx = 2
    compute_gasteiger_charges(mol)
    partial_charge = get_partial_charge(mol,atom_idx)
    if partial_charge:
        print(f"Partial charge for atom {atom_idx} {mol.GetAtomWithIdx(atom_idx).GetSymbol()}: {partial_charge}")
    """
    return mol.GetAtomWithIdx(atom_idx).GetDoubleProp('_GasteigerCharge')


def calculate_descriptor(mol: Chem.Mol, descriptor: str) -> float:
    """
    Computes a specified molecular descriptor.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    descriptor (str): Name of the descriptor.

    Returns:
    float: Descriptor value.

    #Example:
    #Example1:Get MolWt, NumHAcceptors, and NumHDonors for Aspirin.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    target_descriptors = ["CalcExactMolWt","CalcNumHBA","CalcNumHBD"]
    for descriptor in target_descriptors:
        descriptor_value = calculate_descriptor(Aspirin_compound,descriptor)
        print(f"{descriptor} of Aspirin: {descriptor_value}")

    #Real-World use case:
    #When researching a molecule,you want to calculate a specific descriptor.
    Naphthalene = Chem.MolFromSmiles("c1ccc2c(c1)cccc2")
    target_descriptor = "CalcNumRings"
    descriptor_value = calculate_descriptor(Naphthalene,target_descriptor)
    if descriptor_value:
        print(f"{target_descriptor} of Benzene ring: {descriptor_value}")
    """
    return getattr(rdMolDescriptors, descriptor)(mol)


def calculate_all_descriptors(mol: Chem.Mol) -> dict:
    """
    Computes all available molecular descriptors.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    dict: Dictionary of descriptor names and values.

    #Example:
    #Example1:Calculate all available descriptors for Aspirin.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    All_descriptors = calculate_all_descriptors(Aspirin_compound)
    if All_descriptors:
        for descriptor,value in All_descriptors.items():
            print(f"{descriptor}: {value}")

    #Real-World use case:
    #You want to ensure a drug's properties before you do experiments.
    Ibuprofen_compound = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    All_descriptors = rt.calculate_all_descriptors(Ibuprofen_compound)
    if All_descriptors:
        for descriptor,value in All_descriptors.items():
            print(f"{descriptor}: {value}")
    """
    return Descriptors.CalcMolDescriptors(mol)


def run_reaction(reactants: list, reaction_smarts: str) -> list:
    """
    Applies a chemical reaction to a set of reactants.

    Parameters:
    reactants (list[rdkit.Chem.rdchem.Mol]): List of reactant molecules.
    reaction_smarts (str): SMARTS string defining the reaction.

    Returns:
    list[list[rdkit.Chem.rdchem.Mol]]: List of possible product sets.

    #Example:
    #Example1:Run a reaction between Hydrogen and iodine.
    reactants = [Chem.MolFromSmiles("[H][H]"),Chem.MolFromSmiles("I[I]")]
    Hydrogen_reacts_with_Iodine = "[H:1][H:2].[I:3][I:4]>>[H:1][I:3].[H:2][I:4]"
    possible_products = run_reaction(reactants,Hydrogen_reacts_with_Iodine )
    if possible_products:
        for i,product in enumerate(possible_products):
            print(f"Product {i+1} SMILES:")
            for i in range(len(product)):
                print(Chem.MolToSmiles(product[i]))
    else:
        print("No products found")

    #Example2:Run an esterification reaction between ethanol and acetic acid.
    reactants = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CC(=O)O")
        ]
    esterification_reaction = "[C:1]-[C:2]-[O:3].[C:4]-[C:5](=[O:6])-[O:7]>>[C:4]-[C:5](=[O:6])-[O:3]-[C:2]-[C:1].[O:7]"
    possible_products = run_reaction(reactants,esterification_reaction )
    if possible_products:
        for i,product in enumerate(possible_products):
            print(f"Possible Product {i+1},Number of products: {len(product)}, SMILES:")
            for i in range(len(product)):
                print(Chem.MolToSmiles(product[i]))
    else:
        print("No products found")

    #Real-World use case:
    #You are a drug developer.Before you synthesize aspirin,you want identify possible products.
    reactants = [
            Chem.MolFromSmiles("Oc1ccccc1C(=O)O"), # salicylic acid
            Chem.MolFromSmiles("CC(=O)OC(=O)C") # acetic anhydride
    ]
    Synthesize_aspirin = ("[O:1]-[c:2]1:[c:3]:[c:4]:[c:5]:[c:6]:[c:7]:1-[C:8](=[O:9])-[O:10].[C:11]-[C:12](=[O:13])-[O:14]-[C:15](=[O:16])-[C:17]"
                         ">>[C:11]-[C:12](=[O:13])-[O:14]-[c:2]1:[c:3]:[c:4]:[c:5]:[c:6]:[c:7]:1-[C:8](=[O:9])-[O:10].[C:17]-[C:15](=[O:16])-[O:1]")
    possible_products = run_reaction(reactants,Synthesize_aspirin)
    if possible_products:
        for i,product in enumerate(possible_products):
            print(f"Possible Product {i+1},Number of products: {len(product)}, SMILES:")
            for i in range(len(product)):
                print(Chem.MolToSmiles(product[i]))
    else:
        print("No products found")
    """
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    return rxn.RunReactants(tuple(reactants))


def run_reaction_from_file(reactants: list, rxn_file: str) -> list:
    """
    Applies a chemical reaction defined in an MDL RXN file.

    Parameters:
    reactants (list[rdkit.Chem.rdchem.Mol]): List of reactant molecules.
    rxn_file (str): Path to the MDL RXN file.

    Returns:
    list[list[rdkit.Chem.rdchem.Mol]]: List of possible product sets.

    #Function:
    #This function used for generate an MDL RXN file with the reaction you want to apply.
    def Generate_rxn_file(reaction:str,filename:str):
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    rxn_block = rdChemReactions.ReactionToRxnBlock(rxn)
    with open(filename, "w") as f:
        f.write(rxn_block)
    f.close()

    #Example:
    #Example1:Run a reaction between ethanol and acetic acid from an MDL RXN file.
    reactants = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CC(=O)O")
    ]
    rxn_file_path = "esterification_reaction.rxn"
    reaction = "[C:1]-[C:2]-[O:3].[C:4]-[C:5](=[O:6])-[O:7]>>[C:4]-[C:5](=[O:6])-[O:3]-[C:2]-[C:1].[O:7]"
    Generate_rxn_file(reaction,rxn_file_path) # Generate an MDL RXN file
    possible_products = run_reaction_from_file(reactants,rxn_file_path)
    if possible_products:
        for i,product in enumerate(possible_products):
            print(f"Possible Product {i+1},Number of products: {len(product)}, SMILES:")
            for i in range(len(product)):
                print(Chem.MolToSmiles(product[i]))
    else:
        print("No products found")

    #Real-World use case:
    #You are a drug developer.Before you synthesize a drug(e.g aspirin),you want identify possible products from an MDL RXN file.
    reactants = [
            Chem.MolFromSmiles("Oc1ccccc1C(=O)O"),
            Chem.MolFromSmiles("CC(=O)OC(=O)C")
    ]
    rxn_file_path = "Synthesize_aspirin.rxn" #A MDL RXN file containing the reaction you want to apply
    possible_products = run_reaction_from_file(reactants,rxn_file_path)
    if possible_products:
        for i,product in enumerate(possible_products):
            print(f"Possible Product {i+1},Number of products: {len(product)}, SMILES:")
            for i in range(len(product)):
                print(Chem.MolToSmiles(product[i]))
    else:
        print("No products found")
    """
    rxn = AllChem.ReactionFromRxnFile(rxn_file)
    return rxn.RunReactants(tuple(reactants))


def draw_reaction(reaction_smarts: str, filename: str):
    """
    Draws a chemical reaction and saves it as an image.

    Parameters:
    reaction_smarts (str): SMARTS string defining the reaction.
    filename (str): Output image file path.

    Returns:
    None
    """
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    d2d = Draw.MolDraw2DCairo(800, 300)
    d2d.DrawReaction(rxn)
    d2d.WriteDrawingText(filename)


def draw_reaction_highlighted(reaction_smarts: str, filename: str):
    """
    Draws a chemical reaction with reactant highlights and saves it as an image.

    Parameters:
    reaction_smarts (str): SMARTS string defining the reaction.
    filename (str): Output image file path.

    Returns:
    None
    """
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    d2d = Draw.MolDraw2DCairo(800, 300)
    d2d.DrawReaction(rxn, highlightByReactant=True)
    d2d.WriteDrawingText(filename)


from rdkit import Chem, DataStructs
from rdkit.DataStructs import cDataStructs
from rdkit.Chem import AllChem, FilterCatalog, rdRGroupDecomposition
from rdkit.Chem import rdMolDescriptors


def rgroup_decompose_multiple_cores(cores: list, mols: list) -> list:
    """
    Performs R-group decomposition using multiple core scaffolds.

    Parameters:
    cores (list[rdkit.Chem.rdchem.Mol]): List of core molecules.
    mols (list[rdkit.Chem.rdchem.Mol]): List of molecules to decompose.

    Returns:
    list[dict]: Decomposition results as a list of dictionaries.

    #Annotation:
    #The molecule to be decomposed will not continue to match if it already successfully matches a core scaffold.
    #Example:
    #Example1:Decompose a set of molecules using Benzene as a core scaffold.
    Core_SMILES_list = ["c1ccccc1"]
    Mol_SMILES_list = ["c1ccccc1C","c1ccccc1Cl"]
    Core_list = [Chem.MolFromSmiles(SMILES) for SMILES in Core_SMILES_list]
    Mols_list = [Chem.MolFromSmiles(SMILES) for SMILES in Mol_SMILES_list]
    Discomposed_Mols_list = rgroup_decompose_multiple_cores(Core_list,Mols_list)
    if Discomposed_Mols_list:
        for i,discomposed_mol in enumerate(Discomposed_Mols_list):
            print(f"Discomposed molecule for {Mol_SMILES_list[i]}:")
            for key,value in discomposed_mol.items():
                print(f"{key}: {value}")

    #Real-World use case:
    #When conducting a drug analysis, you want to decompose multiple molecules according to a list of core scaffolds.
    Drug_compounds = [Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O"),Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")]
    Core_scaffold = [Chem.MolFromSmiles("c1ccccc1"),Chem.MolFromSmiles("c1ccccc1")]
    Discomposed_Mols_list = rgroup_decompose_multiple_cores(Core_scaffold,Drug_compounds)
    if Discomposed_Mols_list:
        for i,discomposed_mol in enumerate(Discomposed_Mols_list):
            print(f"Discomposed molecule for {Chem.MolToSmiles(Drug_compounds[i])}:")
            for key,value in discomposed_mol.items():
                print(f"{key}: {value}")
    """
    res, _ = rdRGroupDecomposition.RGroupDecompose(cores, mols, asSmiles=True)
    return res


def get_bit_vector(mol: Chem.Mol, sparse: bool = True, radius: int = 2, bv_size: int = 2048) -> Union[cDataStructs.ExplicitBitVect, cDataStructs.SparseBitVect]:
    """
    Generates a bit vector fingerprint for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    sparse (bool): Whether to use a SparseBitVect (True) or ExplicitBitVect (False).
    radius (int): Morgan fingerprint radius.
    bv_size (int): Bit vector size.

    Returns:
    Union[rdkit.cDataStructs.ExplicitBitVect,rdkit.cDataStructs.SparseBitVect]: Bit vector fingerprint.

    #Example:
    #Example1:Generate bit vector fingerprint for Aspirin with default parameters.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    bit_vector = get_bit_vector(Aspirin_compound)
    if bit_vector:
        print(f"Bit vector length: {len(bit_vector)}, Bit vector preview: {bit_vector.ToBitString()[0:128]}")
    else:
        print("Failed to generate bit vector")

    #Example2:Generate sparse bit vector fingerprint for Aspirin with radius = 3 and bv_size = 1024.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    bit_vector = get_bit_vector(Aspirin_compound,sparse=False,radius=3,bv_size=1024)
    if bit_vector:
        print(f"Bit vector length: {len(bit_vector)}, Bit vector preview: {bit_vector.ToBitString()[0:128]}")
    else:
        print("Failed to generate bit vector")

    #Real-World use case:
    #When processing a large set of molecules, you want to generate bit vectors for each molecule to enable efficient similarity search.
    Smiles_list = ["CCO","c1ccccc1","CC(=O)OC1=CC=CC=C1C(O)=O","CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
    Molecule_set = [Chem.MolFromSmiles(smiles) for smiles in Smiles_list]
    for i,mol in enumerate(Molecule_set):
        bit_vector = get_bit_vector(mol,sparse=False,radius=3,bv_size=1024)
        if bit_vector:
            print(f"{i+1}. SMILES: {Chem.MolToSmiles(mol)}, Bit vector length: {len(bit_vector)}, Bit vector preview: {bit_vector.ToBitString()[0:128]}")
        else:
            print("Failed to generate bit vector")
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bv_size)
    bit_vector = generator.GetFingerprint(mol)
    if sparse:
        sparse_bv = cDataStructs.SparseBitVect(bv_size)
        for bit in sparse_bv.GetOnBits():
            sparse_bv.SetBit(bit)
        return sparse_bv
    return bit_vector


def edit_atom_in_molecule(mol: Chem.Mol, atom_idx: int, atomic_num: int) -> Chem.Mol:
    """
    Edits an atom in a molecule, changing its atomic number.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    atom_idx (int): Index of the atom to modify.
    atomic_num (int): New atomic number.

    Returns:
    rdkit.Chem.rdchem.Mol: Modified molecule.

    #Example:
    #Example1:Edit Toluene to replace the methyl group with a hydroxy group.
    Toluene_compound = Chem.MolFromSmiles("c1ccccc1C")
    New_compound = edit_atom_in_molecule(Toluene_compound,6,8)
    if New_compound:
        print(f"New compound after editing: {Chem.MolToSmiles(New_compound)}")

    #Real-World use case:
    #When studying a molecule,you want to replace a specific atom with a different atom to compare the effects of the change.
    Ethanol_compound = Chem.MolFromSmiles("CCO")
    New_compound = rt.edit_atom_in_molecule(Ethanol_compound,2,17) # You can get decriptors for the new compound through other methods.
    if New_compound:
        print(f"New compound after editing: {Chem.MolToSmiles(New_compound)}")

    """
    rw_mol = Chem.RWMol(mol)
    rw_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(atomic_num)
    Chem.SanitizeMol(rw_mol)
    return rw_mol


def batch_edit_molecule(mol: Chem.Mol, atom_removals: list, bond_removals: list) -> Chem.Mol:
    """
    Performs batch edits on a molecule by removing specified atoms and bonds.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.
    atom_removals (list[int]): List of atom indices to remove.
    bond_removals (list[tuple[int, int]]): List of bond (start, end) atom pairs to remove.

    Returns:
    rdkit.Chem.rdchem.Mol: Edited molecule.

    #Example:
    #Example1:Remove the second oxygen atom and C-C bond from Acetic acid.
    Acetic_acid_compound = Chem.MolFromSmiles("CC(=O)O")
    atom_removals = [3]
    bond_removals = [(0,1)]
    New_compound = batch_edit_molecule(Acetic_acid_compound,atom_removals,bond_removals)
    if New_compound:
        print(f"New compound after editing: {Chem.MolToSmiles(New_compound)}")

    #Real-World use case:
    #When conducting a drug analysis,you want to remove an atom or a bond to compare the effects of the change.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    atom_removals = [0,2] # Remove the Methyl group and Oxygen atom with doule bond in Acetyl.
    bond_removals = [(10,11)] # Remove the C-O bond in Carboxyl.
    New_compound = rt.batch_edit_molecule(Aspirin_compound,atom_removals,bond_removals)
    if New_compound:
        print(f"New compound after editing: {Chem.MolToSmiles(New_compound)}")
    """
    rw_mol = Chem.RWMol(mol)
    rw_mol.BeginBatchEdit()
    for atom in atom_removals:
        rw_mol.RemoveAtom(atom)
    for bond in bond_removals:
        rw_mol.RemoveBond(*bond)
    rw_mol.CommitBatchEdit()
    Chem.SanitizeMol(rw_mol)
    return rw_mol


def apply_lipinski_rule_of_five(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule satisfies Lipinski's Rule of Five.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    bool: True if the molecule satisfies Lipinski's rule, False otherwise.

    #Example:
    #Example1:Test a molecule satisfies Lipinski's Rule of Five.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    result = apply_lipinski_rule_of_five(Aspirin_compound)
    if result == True:
        print("The compound satisfies the Lipinski rule of five")
    elif result == False:
        print("The compound does not satisfy the Lipinski rule of five")
    else:
        print("The compound is not valid")

    #Example2:Test a molecule does not satisfy Lipinski's Rule of Five(Excessive molecular weight).
    molecule_with_high_weight = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    result = apply_lipinski_rule_of_five(molecule_with_high_weight)
    if result == True:
        print("The compound satisfies the Lipinski rule of five")
    elif result == False:
        print("The compound does not satisfy the Lipinski rule of five")
    else:
        print("The compound is not valid")

    #Real-World use case:
    #When developing a new drug,you want to check if it satisfies Lipinski's Rule of Five to ensure it is safe to prescribe.
    Ibuprofen_compound = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    result = rt.apply_lipinski_rule_of_five(Ibuprofen_compound)
    if result == True:
        print("The compound satisfies the Lipinski rule of five")
    elif result == False:
        print("The compound does not satisfy the Lipinski rule of five")
    """
    from rdkit.Chem import Descriptors

    mw = Descriptors.MolWt(mol)
    hba = Descriptors.NOCount(mol)
    hbd = Descriptors.NHOHCount(mol)
    logp = Descriptors.MolLogP(mol)

    conditions = [mw <= 500, hba <= 10, hbd <= 5, logp <= 5]
    return conditions.count(True) >= 3


def filter_pains(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule contains PAINS (Pan-Assay Interference Compounds) substructures.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    bool: True if the molecule contains PAINS substructures, False otherwise.

    #Example:
    #Example1:Test a molecule contains PAINS substructures(e.g p-benzoquinone).
    Compound = Chem.MolFromSmiles("O=C1C=CC(=O)C=C1")
    result = filter_pains(Compound)
    if result == True:
        print(f"{Chem.MolToSmiles(Compound)} contains PAINS substructure")
    elif result == False:
        print(f"{Chem.MolToSmiles(Compound)} does not contain PAINS substructure")
    else:
        print("Failed to filter PAINS substructure")

    #Example2:Test a molecule does not contain PAINS substructures(e.g Ibuprofen).
    Ibuprofen_compound = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    result = filter_pains(Ibuprofen_compound)
    if result == True:
        print(f"{Chem.MolToSmiles(Ibuprofen_compound)} contains PAINS substructure")
    elif result == False:
        print(f"{Chem.MolToSmiles(Ibuprofen_compound)} does not contain PAINS substructure")
    else:
        print("Failed to filter PAINS substructure")

    #Real-World use case:
    #When studying a drug,you want to check if it contains PAINS substructures to ensure it is safe to prescribe.
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    result = filter_pains(Aspirin_compound)
    if result:
        print(f"{Chem.MolToSmiles(Aspirin_compound)} contains PAINS substructure")
    else:
        print(f"{Chem.MolToSmiles(Aspirin_compound)} does not contain PAINS substructure")
    """
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog.HasMatch(mol)


def filter_nih(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule contains NIH filter substructures (reactive or undesirable).

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule.

    Returns:
    bool: True if the molecule contains NIH substructures, False otherwise.

    #Example:
    #Example1:Test a molecule contains NIH filter substructures(e.g hyperoxide)
    Compound = Chem.MolFromSmiles("COOOC")
    result = filter_nih(Compound)
    if result == True:
        print(f"{Chem.MolToSmiles(Compound)} contains NIH substructure")
    elif result == False:
        print(f"{Chem.MolToSmiles(Compound)} does not contain NIH substructure")
    else:
        print("Failed to filter NIH substructure")

    #Example2:Test a molecule does not contain NIH filter substructures
    Aspirin_compound = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O")
    result = filter_nih(Aspirin_compound)
    if result == True:
        print(f"{Chem.MolToSmiles(Aspirin_compound)} contains NIH substructure")
    elif result == False:
        print(f"{Chem.MolToSmiles(Aspirin_compound)} does not contain NIH substructure")
    else:
        print("Failed to filter NIH substructure")

    #Real-World use case:
    #When studying a drug,you want to check if it contains NIH filter substructures to ensure it is safe to prescribe.
    acetaminophen = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    result = filter_nih(acetaminophen)
    if result :
        print(f"{Chem.MolToSmiles(acetaminophen)} contains NIH substructure")
    else :
        print(f"{Chem.MolToSmiles(acetaminophen)} does not contain NIH substructure")
    """
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog.HasMatch(mol)

