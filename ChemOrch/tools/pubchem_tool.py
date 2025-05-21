from typing import List, Union, Dict, Any
import pubchempy as pcp
import pandas as pd


def get_compounds(identifier: str, namespace: str = 'cid', searchtype: str = None,
                  as_dataframe: bool = False, **kwargs) -> Union[List[pcp.Compound], pd.DataFrame]:
    """
    Retrieve compound records from PubChem.

    Parameters:
    identifier (str): The compound identifier (e.g., CID, name, SMILES).  This is your search term.
    namespace (str, optional): The identifier type.  Defaults to 'cid'.
        Possible values: 'cid', 'name', 'smiles', 'sdf', 'inchi', 'inchikey', 'formula'.
        - 'cid': PubChem Compound Identifier.  A unique integer assigned to each compound.
        - 'name':  Compound name (e.g., "Aspirin").
        - 'smiles': SMILES string representing the compound's structure.
        - 'sdf':  SDF (Structure Data Format) string (multiline). Rarely used as input.
        - 'inchi': InChI string (International Chemical Identifier).
        - 'inchikey':  InChIKey, a hashed version of the InChI.
        - 'formula': Molecular formula (e.g., "C9H8O4").
    searchtype (str, optional): Advanced search type (for structure searches). Defaults to None.
        Possible values: 'substructure', 'superstructure', 'similarity'.
        - 'substructure':  Finds compounds containing the given structure as a substructure.  Use with SMILES, InChI.
        - 'superstructure': Finds compounds that are substructures of the given structure. Use with SMILES, InChI.
        - 'similarity': Finds compounds similar to the given structure. Use with SMILES, InChI, CID.
    as_dataframe (bool, optional): Return results as a pandas DataFrame. Defaults to False.
    **kwargs: Additional keyword arguments passed to `pubchempy.get_compounds`. See pubchempy documentation for details.

    Returns:
    Union[List[pcp.Compound], pd.DataFrame]:
        - If as_dataframe is False: A list of Compound objects.
        - If as_dataframe is True: A pandas DataFrame containing compound properties.
        - Returns an empty list if no compounds are found.  Does NOT raise an exception.

    Example:
    # Example 1: Get compound object for Aspirin by name
    aspirin_compounds = get_compounds('Aspirin', namespace='name')
    if aspirin_compounds:
        print(f"Found compound for Aspirin: {aspirin_compounds[0].cid}, IUPAC name: {aspirin_compounds[0].iupac_name}")

    # Example 2: Get compound data as pandas DataFrame for CID 2244 (Aspirin)
    aspirin_df = get_compounds(2244, namespace='cid', as_dataframe=True)
    if not aspirin_df.empty:
        print(f"DataFrame for Aspirin (CID 2244) columns: {aspirin_df.columns.tolist()}")

    # Example 3: Substructure search for compounds containing benzene ring (SMILES 'C1=CC=CC=C1')
    benzene_substructures = get_compounds('C1=CC=CC=C1', namespace='smiles', searchtype='substructure')
    if benzene_substructures:
        print(f"Found {len(benzene_substructures)} compounds with benzene substructure. First one CID: {benzene_substructures[0].cid}")

    # Real-world use case:
    # A researcher wants to find information about caffeine (by name) and store it in a DataFrame for further analysis.
    caffeine_df = get_compounds('caffeine', namespace='name', as_dataframe=True)
    if not caffeine_df.empty:
        print(f"Caffeine data retrieved as DataFrame with shape: {caffeine_df.shape}")
    else:
        print("Caffeine not found.")
    """
    return pcp.get_compounds(identifier, namespace, searchtype, as_dataframe, **kwargs)


def get_substances(identifier: str, namespace: str = 'sid',
                  as_dataframe: bool = False) -> Union[List[pcp.Substance], pd.DataFrame]:
    """
    Retrieve substance records from PubChem.  Substances are the raw deposited data.

    Parameters:
    identifier (str): The substance identifier (e.g., SID, name, source ID).
    namespace (str, optional): The identifier type. Defaults to 'sid'.
        Possible values: 'sid', 'name', 'sourceid/<source name>'.
        - 'sid': PubChem Substance Identifier (a unique integer).
        - 'name': Substance name.
        - 'sourceid/<source name>':  The ID used by the data depositor (source).  Format is "sourceid/sourcename".
    as_dataframe (bool, optional): Return results as a pandas DataFrame. Defaults to False.

    Returns:
    Union[List[pcp.Substance], pd.DataFrame]:
       - If as_dataframe is False: A list of Substance objects.
       - If as_dataframe is True: A pandas DataFrame containing substance properties.
       - Returns an empty list if no substances are found.

    Example:
    # Example 1: Get substance object for Aspirin by name
    aspirin_substances = get_substances('Aspirin', namespace='name')
    if aspirin_substances:
        print(f"Found substances for Aspirin. First one SID: {aspirin_substances[0].sid}, Source ID: {aspirin_substances[0].source_id}")

    # Example 2: Get substance data as pandas DataFrame for SID 135 (Acetic Acid)
    acetic_acid_substance_df = get_substances(135, namespace='sid', as_dataframe=True)
    if not acetic_acid_substance_df.empty:
        print(f"DataFrame for Acetic Acid Substance (SID 135) columns: {acetic_acid_substance_df.columns.tolist()}")

    # Real-world use case:
    # A material scientist is interested in raw data depositions related to 'Gold Nanoparticles'.
    gold_nano_substances_df = get_substances('Gold Nanoparticles', namespace='name', as_dataframe=True)
    if not gold_nano_substances_df.empty:
        print(f"Gold Nanoparticles substance data retrieved as DataFrame with shape: {gold_nano_substances_df.shape}")
    else:
        print("Gold Nanoparticles substances not found.")
    """
    return pcp.get_substances(identifier, namespace, as_dataframe)


def get_assays(identifier: str, namespace: str = 'aid') -> List[pcp.Assay]:
    """
    Retrieve assay records from PubChem.  Assays describe bioactivity experiments.

    Parameters:
    identifier (str): The assay identifier (typically the AID).
    namespace (str, optional): The identifier type. Defaults to 'aid'.
        - 'aid': PubChem Assay Identifier (a unique integer).

    Returns:
    List[pcp.Assay]: A list of Assay objects.  Returns an empty list if no assays are found.

    Example:
    # Example 1: Get assay object for AID 1831
    assay_1831 = get_assays(1831, namespace='aid')
    if assay_1831:
        print(f"Found assay with AID 1831. Description preview: {assay_1831[0].description[:50]}...")

    # Real-world use case:
    # A biologist wants to investigate assays related to a specific target (e.g., a protein),
    # but they often start with a known AID for a preliminary investigation.
    initial_assay = get_assays(1, namespace='aid') # AID 1 is a very general assay for demonstration.
    if initial_assay:
        print(f"Retrieved initial assay AID: {initial_assay[0].aid}, Type: {initial_assay[0].assay_type}")
    else:
        print("Assay with AID 1 not found.") # This AID 1 should exist, but handle cases where it might not in real-world scenarios.
    """
    return pcp.get_assays(identifier, namespace)



def get_properties(properties: Union[str, List[str]], identifier: str, namespace: str = 'cid',
                   searchtype: str = None, as_dataframe: bool = False) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
     Retrieve specified properties for compounds, substances, or assays from PubChem.
     This function is more general, getting properties instead of entire record objects.

     Parameters:
     properties (Union[str, List[str]]): A single property name (string) or a list of property names.
         Example: ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES']
         Available properties for Compounds can be found as attributes of the Compound class (e.g., 'molecular_formula').
     identifier (str): The identifier (e.g., CID, SID, AID).
     namespace (str, optional): The identifier type. Defaults to 'cid'.
         Possible values depend on the type of record being retrieved.
         See get_compounds(), get_substances(), get_assays() for valid namespaces.
     searchtype (str, optional): Advanced search type for structure-based searches (same as in get_compounds).
     as_dataframe (bool, optional): Return results as a pandas DataFrame. Defaults to False.

     Returns:
     Union[List[Dict[str, Any]], pd.DataFrame]:
         - If as_dataframe is False: A list of dictionaries, where each dictionary contains the requested properties
           for a single record.
         - If as_dataframe is True: A pandas DataFrame where each row represents a record and each column a property.
         - Returns an empty list/DataFrame if no records are found or properties don't exist for a given record.

    Example:
    # Example 1: Get Molecular Formula and Molecular Weight for Aspirin (by name) as list of dictionaries
    aspirin_props_list = get_properties(['MolecularFormula', 'MolecularWeight'], 'Aspirin', namespace='name')
    if aspirin_props_list:
        print(f"Aspirin Properties (list of dicts): {aspirin_props_list}")

    # Example 2: Get Canonical SMILES for CID 2244 (Aspirin) as DataFrame
    aspirin_smiles_df = get_properties('CanonicalSMILES', 2244, namespace='cid', as_dataframe=True)
    if not aspirin_smiles_df.empty:
        print(f"Aspirin Canonical SMILES (DataFrame):\n{aspirin_smiles_df}")

    # Example 3: Get multiple properties for a list of CIDs (e.g., CID 2244 and 2245) as DataFrame
    cid_list = ['2244', '2245'] # Using string CIDs for demonstration, can be integers too.
    multi_compound_props_df = get_properties(['MolecularFormula', 'IUPACName'], cid_list, namespace='cid', as_dataframe=True)
    if not multi_compound_props_df.empty:
        print(f"Properties for multiple CIDs (DataFrame):\n{multi_compound_props_df}")

    # Real-world use case:
    # A chemist needs to quickly retrieve IUPAC names and SMILES for a batch of compounds identified by their CIDs.
    batch_cids = [962, 2244, 3384] # Example CIDs: Water, Aspirin, Ethanol
    batch_props_df = get_properties(['IUPACName', 'CanonicalSMILES'], batch_cids, namespace='cid', as_dataframe=True)
    if not batch_props_df.empty:
        print(f"Batch property retrieval DataFrame shape: {batch_props_df.shape}")
        print(f"First few rows of batch properties:\n{batch_props_df.head()}")
    else:
        print("No properties found for the given CIDs.")
    """
    return pcp.get_properties(properties, identifier, namespace, searchtype, as_dataframe)

def compound_from_cid(cid: int) -> pcp.Compound:
    """
    Create a Compound object from a PubChem Compound Identifier (CID).

    Parameters:
    cid (int): The PubChem Compound Identifier (CID).

    Returns:
    pcp.Compound: The Compound object.

    Raises:
        pubchempy.NotFoundError: If the CID is invalid.

    Example:
    # Example 1: Create Compound object for CID 2244 (Aspirin)
    aspirin_compound = compound_from_cid(2244)
    print(f"Compound name for CID 2244: {aspirin_compound.iupac_name}")

    # Real-world use case:
    # In a drug discovery pipeline, after identifying a potential drug candidate by its CID,
    # you would create a Compound object to access its properties.
    drug_candidate_cid = 5090 # Example CID for Rofecoxib
    drug_compound = compound_from_cid(drug_candidate_cid)
    print(f"Drug candidate (CID {drug_candidate_cid}) retrieved: {drug_compound.iupac_name}, Formula: {drug_compound.molecular_formula}")

    # Example of handling NotFoundError (for an invalid CID, e.g., 0)
    try:
        invalid_compound = compound_from_cid(0) # CID 0 is invalid
    except pcp.NotFoundError:
        print("Error: Invalid CID provided. Compound not found.")
    """
    return pcp.Compound.from_cid(cid)

def substance_from_sid(sid: int) -> pcp.Substance:
    """
    Create a Substance object from a PubChem Substance Identifier (SID).

    Parameters:
    sid (int): The PubChem Substance Identifier (SID).

    Returns:
    pcp.Substance: The Substance object.

    Example:
    # Example 1: Create Substance object for SID 135 (Acetic Acid substance)
    acetic_acid_substance = substance_from_sid(135)
    print(f"Substance SID: {acetic_acid_substance.sid}, Source ID: {acetic_acid_substance.source_id}")

    # Real-world use case:
    # When working with raw data depositions or needing source-specific information,
    # you'd use Substance objects. For example, to investigate a specific deposition (SID).
    deposition_sid = 223766453 # Example SID
    deposition_substance = substance_from_sid(deposition_sid)
    print(f"Deposition Substance (SID {deposition_sid}) retrieved. Synonyms preview: {deposition_substance.synonyms[:3]}")
    """
    return pcp.Substance.from_sid(sid)

def assay_from_aid(aid: int) -> pcp.Assay:
    """
    Create an Assay object from a PubChem Assay Identifier (AID).

    Parameters:
    aid (int): The PubChem Assay Identifier (AID).

    Returns:
    pcp.Assay: The Assay object.

    Example:
    # Example 1: Create Assay object for AID 1831
    assay_1831_obj = assay_from_aid(1831)
    print(f"Assay AID: {assay_1831_obj.aid}, Assay type: {assay_1831_obj.assay_type}")

    # Real-world use case:
    # To programmatically access details and results of a specific bioassay experiment in PubChem,
    # you would create an Assay object using its AID.
    target_assay_aid = 1  # Example AID
    target_assay = assay_from_aid(target_assay_aid)
    print(f"Target Assay (AID {target_assay_aid}) retrieved. Description preview: {target_assay.description[:40]}...")
    """
    return pcp.Assay.from_aid(aid)


def compounds_to_frame(compounds: List[pcp.Compound], properties: List[str] = None) -> pd.DataFrame:
    """
    Convert a list of Compound objects to a pandas DataFrame.

    Parameters:
    compounds (List[pcp.Compound]): A list of Compound objects.
    properties (List[str], optional): A list of specific properties to include.  If None, includes
        many common properties, but not those requiring extra requests (synonyms, sids, aids).

    Returns:
    pd.DataFrame: A DataFrame containing the specified (or default) properties of the compounds.

    Example:
    # Example 1: Convert a list of Compound objects to DataFrame with default properties
    aspirin_compounds = get_compounds('Aspirin', namespace='name') # Get a list of compound objects
    if aspirin_compounds:
        aspirin_df_default = compounds_to_frame(aspirin_compounds)
        print(f"DataFrame from Compound objects (default properties) columns: {aspirin_df_default.columns.tolist()}")

    # Example 2: Convert a list to DataFrame with specific properties (IUPACName, MolecularWeight)
    ethanol_compounds = get_compounds('Ethanol', namespace='name')
    if ethanol_compounds:
        ethanol_df_specific = compounds_to_frame(ethanol_compounds, properties=['IUPACName', 'MolecularWeight'])
        print(f"DataFrame from Compound objects (specific properties) columns: {ethanol_df_specific.columns.tolist()}")

    # Real-world use case:
    # After performing a search and getting a list of Compound objects, you want to analyze them in a tabular format.
    drug_like_compounds = get_compounds('drug-like', namespace='name') # Hypothetical search for drug-like compounds
    if drug_like_compounds:
        drug_like_df = compounds_to_frame(drug_like_compounds, properties=['IUPACName', 'MolecularWeight', 'XLogP'])
        print(f"Drug-like compounds DataFrame shape: {drug_like_df.shape}")
        print(f"First few rows of drug-like compounds data:\n{drug_like_df.head()}")
    else:
        print("No drug-like compounds found.")
    """
    return pcp.compounds_to_frame(compounds, properties)

def substances_to_frame(substances: List[pcp.Substance], properties: List[str] = None) -> pd.DataFrame:
    """
    Convert a list of Substance objects to a pandas DataFrame.

    Parameters:
    substances (List[pcp.Substance]):  A list of Substance objects.
    properties (List[str], optional): A list of specific properties to include. If None, includes
        many common properties, but not those requiring extra requests (cids, aids).

    Returns:
    pd.DataFrame: A DataFrame containing the specified (or default) properties of the substances.

    Example:
    # Example 1: Convert a list of Substance objects to DataFrame with default properties
    aspirin_substances = get_substances('Aspirin', namespace='name') # Get a list of substance objects
    if aspirin_substances:
        aspirin_substance_df_default = substances_to_frame(aspirin_substances)
        print(f"DataFrame from Substance objects (default properties) columns: {aspirin_substance_df_default.columns.tolist()}")

    # Example 2: Convert a list to DataFrame with specific properties (Synonyms, SourceID)
    ethanol_substances = get_substances('Ethanol', namespace='name')
    if ethanol_substances:
        ethanol_substance_df_specific = substances_to_frame(ethanol_substances, properties=['Synonyms', 'SourceID'])
        print(f"DataFrame from Substance objects (specific properties) columns: {ethanol_substance_df_specific.columns.tolist()}")

    # Real-world use case:
    # After retrieving a list of Substance objects related to a material, you want to organize and analyze their source information.
    nano_material_substances = get_substances('Nanoparticles', namespace='name') # Hypothetical search for nanoparticle substances
    if nano_material_substances:
        nano_material_df = substances_to_frame(nano_material_substances, properties=['Synonyms', 'SourceID', 'DepositorName'])
        print(f"Nanomaterial substances DataFrame shape: {nano_material_df.shape}")
        print(f"First few rows of nanomaterial substance data:\n{nano_material_df.head()}")
    else:
        print("No nanoparticle substances found.")
    """
    return pcp.substances_to_frame(substances, properties)



# --- Compound Object Methods (for accessing properties of a single Compound) ---

def get_compound_properties(compound: pcp.Compound, properties: List[str] = None) -> Dict[str, Any]:
    """
    Retrieves specified properties from a single Compound object.  This is useful if you
    already *have* a Compound object and want to get specific data from it without
    re-querying PubChem.

    Parameters:
        compound (pcp.Compound): The Compound object.
        properties (List[str], optional):  A list of property names to retrieve. If None,
            returns a dictionary with many default properties (but *not* synonyms, sids, aids,
            which require extra requests). See the Compound class attributes for a list of possible values.

    Returns:
        Dict[str, Any]: A dictionary where keys are property names and values are the corresponding
            property values for the Compound.

    Example:
    # Example 1: Get default properties of an Aspirin Compound object
    aspirin_compound = compound_from_cid(2244)
    default_props = get_compound_properties(aspirin_compound)
    print(f"Default properties of Aspirin (keys preview): {list(default_props.keys())[:5]}...")

    # Example 2: Get specific properties (IUPACName, MolecularFormula) of an Ethanol Compound object
    ethanol_compound = compound_from_cid(689)
    specific_props = get_compound_properties(ethanol_compound, properties=['IUPACName', 'MolecularFormula'])
    print(f"Specific properties of Ethanol: {specific_props}")

    # Real-world use case:
    # After obtaining a Compound object, you need to access specific properties for analysis or display.
    caffeine_compound = compound_from_cid(2519)
    important_props = ['IUPACName', 'MolecularWeight', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount']
    caffeine_details = get_compound_properties(caffeine_compound, properties=important_props)
    print(f"Caffeine details: {caffeine_details}")
    """
    return compound.to_dict(properties)

def get_compound_synonyms(compound: pcp.Compound) -> List[str]:
    """Gets the synonyms for a given compound. (Requires an extra request)

    Example:
    # Example 1: Get synonyms for Aspirin Compound object
    aspirin_compound = compound_from_cid(2244)
    aspirin_synonyms = get_compound_synonyms(aspirin_compound)
    if aspirin_synonyms:
        print(f"Synonyms for Aspirin (preview): {aspirin_synonyms[:5]}...")

    # Real-world use case:
    # For text mining or information retrieval tasks, synonyms are crucial to identify mentions of a compound under different names.
    glucose_compound = compound_from_cid(5793)
    glucose_synonyms = get_compound_synonyms(glucose_compound)
    if glucose_synonyms:
        print(f"Glucose synonyms count: {len(glucose_synonyms)}")
    else:
        print("No synonyms found for Glucose.")
    """
    return compound.synonyms
def get_compound_sids(compound: pcp.Compound) -> List[int]:
    """Gets the sids for a given compound. (Requires an extra request)

    Example:
    # Example 1: Get SIDs for Aspirin Compound object
    aspirin_compound = compound_from_cid(2244)
    aspirin_sids = get_compound_sids(aspirin_compound)
    if aspirin_sids:
        print(f"SIDs associated with Aspirin: {aspirin_sids}")

    # Real-world use case:
    # To trace back from a standardized compound to its original substance depositions in PubChem.
    caffeine_compound = compound_from_cid(2519)
    caffeine_sids = get_compound_sids(caffeine_compound)
    if caffeine_sids:
        print(f"Number of SIDs for Caffeine: {len(caffeine_sids)}")
    else:
        print("No SIDs found for Caffeine.")
    """
    return compound.sids
def get_compound_aids(compound: pcp.Compound) -> List[int]:
    """Gets the aids for a given compound. (Requires an extra request)

    Example:
    # Example 1: Get AIDs for Aspirin Compound object
    aspirin_compound = compound_from_cid(2244)
    aspirin_aids = get_compound_aids(aspirin_compound)
    if aspirin_aids:
        print(f"AIDs associated with Aspirin (preview): {aspirin_aids[:5]}...")

    # Real-world use case:
    # To find bioactivity assays related to a compound, useful for drug repurposing or understanding biological effects.
    ibuprofen_compound = compound_from_cid(3652)
    ibuprofen_aids = get_compound_aids(ibuprofen_compound)
    if ibuprofen_aids:
        print(f"Number of AIDs for Ibuprofen: {len(ibuprofen_aids)}")
    else:
        print("No AIDs found for Ibuprofen.")
    """
    return compound.aids

# --- Substance Object Methods ---
def get_substance_properties(substance: pcp.Substance, properties: List[str] = None) -> Dict[str, Any]:
    """Retrieves properties from a Substance object.

    Example:
    # Example 1: Get default properties of an Acetic Acid Substance object
    acetic_acid_substance = substance_from_sid(135)
    default_substance_props = get_substance_properties(acetic_acid_substance)
    print(f"Default substance properties (keys preview): {list(default_substance_props.keys())[:5]}...")

    # Example 2: Get specific properties (Synonyms, SourceID) of a Substance object
    ethanol_substance = substance_from_sid(223766453)
    specific_substance_props = get_substance_properties(ethanol_substance, properties=['Synonyms', 'SourceID'])
    print(f"Specific substance properties: {specific_substance_props}")

    # Real-world use case:
    # To access specific deposition details or source information directly from a Substance object.
    gold_nano_substance = get_substance_from_sid(24864499) # Example SID for Gold Nanoparticles substance
    source_details = get_substance_properties(gold_nano_substance, properties=['SourceID', 'DepositorName', 'SourceName'])
    print(f"Gold Nanoparticle substance source details: {source_details}")
    """
    return substance.to_dict(properties)

def get_substance_cids(substance: pcp.Substance) -> List[int]:
    """Gets the cids for a given substance. (Requires an extra request)

    Example:
    # Example 1: Get CIDs for a Substance object (e.g., Aspirin substance)
    aspirin_substance = get_substances('Aspirin', namespace='name')[0] # Get the first substance for Aspirin
    aspirin_substance_cids = get_substance_cids(aspirin_substance)
    if aspirin_substance_cids:
        print(f"CIDs associated with Aspirin substance: {aspirin_substance_cids}")

    # Real-world use case:
    # To find the standardized compound(s) derived from a raw substance deposition.
    raw_material_substance = get_substances('Impure Compound X', namespace='name')[0] # Example impure substance
    compound_cids_from_substance = get_substance_cids(raw_material_substance)
    if compound_cids_from_substance:
        print(f"Standardized CIDs from impure substance: {compound_cids_from_substance}")
    else:
        print("No standardized CIDs found for this substance.")
    """
    return substance.cids

def get_substance_standardized_compound(substance: pcp.Substance) -> pcp.Compound:
    """Gets standardized compound from substance. (Requires an extra request)

    Example:
    # Example 1: Get standardized compound for a Substance object (e.g., Aspirin substance)
    aspirin_substance = get_substances('Aspirin', namespace='name')[0] # Get the first substance for Aspirin
    standard_aspirin_compound = get_substance_standardized_compound(aspirin_substance)
    if standard_aspirin_compound:
        print(f"Standardized compound for Aspirin substance: CID {standard_aspirin_compound.cid}, Name: {standard_aspirin_compound.iupac_name}")

    # Real-world use case:
    # To obtain the standardized chemical representation (Compound object) from a raw Substance deposition for further analysis.
    raw_sample_substance = substance_from_sid(223766453) # Example SID
    standard_compound = get_substance_standardized_compound(raw_sample_substance)
    if standard_compound:
        print(f"Standardized compound from raw sample substance: CID {standard_compound.cid}, Formula: {standard_compound.molecular_formula}")
    else:
        print("No standardized compound found for this substance.")
    """
    return substance.standardized_compound


# --- Assay Object Methods ---
def get_assay_properties(assay: pcp.Assay, properties: List[str] = None) -> Dict[str, Any]:
      """Retrieves properties from an Assay object.

      Example:
      # Example 1: Get default properties of an Assay object (AID 1831)
      assay_1831 = assay_from_aid(1831)
      default_assay_props = get_assay_properties(assay_1831)
      print(f"Default assay properties (keys preview): {list(default_assay_props.keys())[:5]}...")

      # Example 2: Get specific properties (Description, AssayType) of an Assay object
      assay_1 = assay_from_aid(1) # AID 1
      specific_assay_props = get_assay_properties(assay_1, properties=['Description', 'AssayType'])
      print(f"Specific assay properties: {specific_assay_props.keys()}") # Just printing keys to show available properties.
      if 'Description' in specific_assay_props:
          print(f"Assay Description preview: {specific_assay_props['Description'][:50]}...")

      # Real-world use case:
      # To programmatically extract detailed information about a bioassay experiment for analysis or integration into a database.
      target_assay = assay_from_aid(1850) # Example AID for demonstration
      assay_details = get_assay_properties(target_assay, properties=['Description', 'Targets', 'OutcomeClasses'])
      print(f"Assay details: {assay_details.keys()}") # Previewing available details.
      if 'Description' in assay_details:
          print(f"Assay description preview: {assay_details['Description'][:60]}...")
      """
      return assay.to_dict(properties)

