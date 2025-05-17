import json
import ast

def extract_instructions(llm_response: str):
    """
    Extracts a Python list of instructions from an LLM-generated response.
    
    Args:
        llm_response (str): The raw response from the LLM containing the list.
    
    Returns:
        list: A list of generated instructions.
    """
    try:
        # Attempt to parse the response as a valid Python list
        llm_response = llm_response.strip('```')
        llm_response = llm_response.strip('python')
        
        # If the response is already formatted correctly, use literal_eval
        if llm_response.startswith("[") and llm_response.endswith("]"):
            instructions = ast.literal_eval(llm_response)
        else:
            raise ValueError("Response is not in expected list format.")
        
        if isinstance(instructions, list):
            return instructions
        else:
            raise ValueError("Extracted data is not a list.")
    
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing LLM response: {e}. Attempting JSON parsing.")

        # Attempt to parse as JSON (which is more robust)
        try:
            instructions = json.loads(llm_response)
            if isinstance(instructions, list):
                return instructions
        except json.JSONDecodeError as json_e:
            print(f"JSON parsing also failed: {json_e}")
    
    return []