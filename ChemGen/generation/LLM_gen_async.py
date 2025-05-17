
# Local module imports
import ChemGen.generation.api_tools_async as api_tools
from ChemGen.utils.extraction import extract_instructions

# ------------------- System Prompts -------------------

INSTRUCTION_GENERATION_SYSTEM_PROMPT = (
    "You are an advanced AI assistant tasked with generating high-quality instructions for synthetic dataset creation."
)

INSTRUCTION_PLANNING_SYSTEM_PROMPT = (
    "You are an advanced AI assistant tasked with structured problem-solving planning."
)

IDEAL_TOOL_DESCRIPTION_SYSTEM_PROMPT = (
    "You are an advanced AI assistant tasked with defining the ideal tools for executing a plan."
)

# ------------------- User Prompts -------------------

INSTRUCTION_GENERATION_PROMPT = """
Your goal is to produce a diverse set of instructions (or questions) based on a given user task. The corresponding answers will be generated later to form a dataset.

### **Instructions:**
1. **Task Understanding:** Carefully analyze the provided task and determine its core objective.
2. **Instruction Generation:** Create exactly `{n}` unique instructions related to the task. The instructions should be diverse in phrasing and complexity.
3. **Clarity & Context:** Ensure each instruction is clear and provides enough context for an AI model to generate a meaningful response.
4. **Format:** Return the instructions strictly as a Python-style list of strings.
5. **Generation Mode:** There are two possible modes for generating instructions: metadata, free.
Metadata: Your instructions should strictly adhere to the metadata provided below.
Free: Your instructions should only focus on the task, task description and possible constraints.
**Mode**: {mode}
6. **Custom Constraint:** {custom_constraint}


### **Example:**
#### **User Task:** Toxicity Prediction  
#### **Generated Instructions (Example Output):**  

[
    "Does benzo[a]pyrene exhibit toxicity to humans?",
    "What is the acute toxicity of trichloroethylene?",
    "Does bisphenol A have endocrine-disrupting effects?",
    "Do pyridine compounds have neurotoxic effects?",
    "Does tetraethyl lead pose long-term toxicity risks to the environment and humans?"
]
"""

INSTRUCTION_PLANNING_PROMPT = """
You are an advanced AI assistant tasked with planning how to solve a given instruction. Your goal is to **break down the problem into structured steps** that can be executed using external tools or reasoning. You should **not** provide an answer—only a plan.

### **Instructions:**
1. **Understand the Instruction:** Carefully analyze the given instruction to determine its requirements.
2. **Identify Key Elements:** Identify key components such as subject, method, and expected output.
3. **Break Down into Steps:** Generate a structured plan consisting of logical steps that guide the problem-solving process.
4. **Ensure Tool Compatibility:** If an external tool is likely required (e.g., a chemical database, scientific literature, mathematical solver), indicate it explicitly.
5. **Format:** Return the planning steps strictly as a Python-style list of strings.

Now, generate a structured plan for the following instruction:

Instruction: {instruction}

Ensure the output is formatted strictly as a Python list of strings. 
"""

IDEAL_TOOL_DESCRIPTION_PROMPT = """
Your goal is to describe the functionalities of these tools concisely, ensuring that each tool serves **one specific purpose**.

### **Instructions:**
1. **Analyze the Planning Steps:** Carefully review the provided planning steps to determine what kind of external tools would be needed to complete them.
2. **Define the Ideal Toolset:** Describe **only the necessary** tools, ensuring that each tool performs only **one function**.
3. **Keep Descriptions Concise:** Each tool description should be brief and focused on its function.
4. **Limit the Number of Tools:** Minimize the number of tools by **combining related functionalities** into single tools where applicable.
5. **Format:** Return the tool descriptions strictly as a Python-style list of strings.

Now, generate a structured list of ideal tool descriptions for the following planning steps:

Planning Steps: {planning_steps}

Ensure the output is formatted strictly as a Python list of strings, with each tool description containing only one function.
"""

# ------------------- Function Definitions -------------------

async def generate_planning_steps(instruction: str, model: str = "gpt-4o", metadata: str = None) -> list:
    """Generate a structured planning sequence for solving a given instruction.

    Args:
        instruction (str): The instruction for which planning is needed.
        model (str): The language model to use.

    Returns:
        list: A structured list of planning steps.
    """
    metadata = metadata or "No additional metadata."
    user_prompt = f"Metadata: {metadata}\n{INSTRUCTION_PLANNING_PROMPT.format(instruction=instruction)}"

    response =  await api_tools.get_response(
        prompt=user_prompt,
        system_prompt=INSTRUCTION_PLANNING_SYSTEM_PROMPT,
        model=model
    )
    return extract_instructions(response)


async def generate_instructions(user_task: str, task_description: str, n: int, mode: str, metadata: str = None, custom_constraint: str = "None", model: str = "gpt-4o") -> list:
    """Generate a diverse set of instructions for a given user task using a language model.

    Args:
        user_task (str): The user task or prompt for instruction generation.
        n (int): The number of instructions to generate.
        custom_constraint (str): A custom constraint to apply to the instruction generation process.
        instruction_examples (str): Examples of instructions to apply to the instruction generation process.
        model (str): The language model to use for instruction generation.

    Returns:
        list: A list of generated instructions.
    """
    custom_constraint = custom_constraint or "No additional constraints."
    user_prompt = f"{user_task}\ntask description: {task_description}\nmetadata: {metadata}\n{INSTRUCTION_GENERATION_PROMPT.format(n=n, custom_constraint=custom_constraint, mode=mode)}"

    response = await api_tools.get_response(
        prompt=user_prompt,
        system_prompt=INSTRUCTION_GENERATION_SYSTEM_PROMPT,
        model=model
    )
    return extract_instructions(response)


async def generate_tool_descriptions(planning_steps: list, model: str = "gpt-4o", metadata: str = None) -> list:
    """Generate ideal tool descriptions based on a given set of planning steps.

    Args:
        planning_steps (list): The structured planning steps for solving a problem.
        model (str): The language model to use.

    Returns:
        list: A structured list of tool descriptions.
    """
    formatted_steps = "\n".join(f"- {step}" for step in planning_steps)
    metadata = metadata or "No additional metadata."
    user_prompt = f"Metadata: {metadata}\n**Planning Steps:**\n{formatted_steps}\n\n{IDEAL_TOOL_DESCRIPTION_PROMPT}"

    response = await api_tools.get_response(
        prompt=user_prompt,
        system_prompt=IDEAL_TOOL_DESCRIPTION_SYSTEM_PROMPT,
        model=model
    )
    return extract_instructions(response)

# ------------------- Execution Example -------------------

if __name__ == "__main__":
    user_task = "User Task: Toxicity Prediction"
    instruction_examples = "Instruction Examples:\nExample1:Is methane toxic?\nExample2:What is the acute toxicity of white phosphorus"
    custom_constraint = "Each instruction must be concise and should not exceed 20 words."

    generated_instructions = generate_instructions(user_task, n=5,instruction_examples=instruction_examples, custom_constraint=custom_constraint)
    print("Generated Instructions:")
    for idx, instruction in enumerate(generated_instructions, 1):
        print(f"{idx}. {instruction}")

    test_instruction = "Assess the acute toxicity of benzene exposure through inhalation."
    generated_steps = generate_planning_steps(test_instruction)

    print("\nGenerated Planning Steps:")
    for idx, step in enumerate(generated_steps, 1):
        print(f"{idx}. {step}")
    
    tool_descrition = generate_tool_descriptions(generated_steps)
    print("\nGenerated Tool Descriptions:")
    for idx, tool in enumerate(tool_descrition, 1):
        print(f"{idx}. {tool}")
