import yaml
import subprocess
import asyncio
import pickle
import os
import torch
import torch.nn.functional as F
import json
import pandas as pd
from openai import AsyncOpenAI
from ChemGen.generation.LLM_gen_async import generate_instructions, generate_planning_steps, generate_tool_descriptions
from ChemGen.generation.Web_retrieval_async import web_search_retrieve

# Define the threshold for tool distilling. If the number of tools overnumbers the threshold, the probability of removing indirectly related tools will be higher.
threshold_for_tool_distilling = 5
# create a AsyncOpenAI client
aclient = AsyncOpenAI(api_key='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')
# load tools embedding and infomation
with open('ChemGen/generation/tools_embedding.pkl', 'rb') as f:
    tools_info = pickle.load(f)
    f.close()

# ------------------- System Prompts -------------------
Tool_Selection_Prompt = """
I will give you the task, a tool name, and its description.
Your goal is to confirm whether the tool can be used to solve the task.
Instructions:
1. If metadata is provided, your choice should **prioritize the metadata's requirements**.
2. You need to extract the final targets of the task and determine whether it requires a specific
tool or multiple tools.
3. First, you need to focus on solving the final targets of the task.
4. Second, if the task requires multiple tools and this tool excels in one aspect of the task, it is
also useful.
Output format:
1. If the tool can be used for solving the task, return the tool index only. It should be an integer.
2. If the tool can't be used for solving the task, return the string "no" only. It should be a string.
"""
Tool_Distillation_Prompt = f"""
I will give you a list of tools that have been screened, and they are all related to the task. I will
also give you the raw task.
Problems:
1. Although these tools are all related to the task, some may be indirectly related to the task, or
the tool may not be an expert in the task.
2. Some tools may not be able to solve the final targets of the task.
Your goal is to check the tools and confirm whether they need to remove some indirectly
related tools.
Strategies for tool selection:
1. **Check if metadata is provided and prioritize the metadata's requirements.**
2. Pay attention to the tools' names. The tool name contains its function, and if the task needs
the tool, the name often appears in the tool description.
3. Throw light on the task content. The content may clarify what tools or what kinds of tools
are needed for the task.
Instructions:
1. If metadata is provided, your choice of tool should **prioritize the metadata's requirements**.
2. Read the tools list and the task carefully, compare the tools' functions with the task, and
check if the task marks specific tools to use.
3. Analyse the task and extract the final targets of the task. Regarding the tools can't solve the
final targets of the task as useless tools, you should focus on the final targets of the task.
4. If the number of tools overnumbers the threshold:{threshold_for_tool_distilling}, you
should think more about finding and removing indirectly related tools. But if there is only one tool left,
you should think more about retaining it.
Output format:
1. If the indirectly related tools are found, please return only the most indirectly related tool
index.
2. If no indirectly related tools are found, please return the string "no" only.
3. You should return the content described above without any prefixes or suffixes."""

Parameters_Filling_Prompt = """
I will give you a task and the functions and parameters, with descriptions, needed for solving
the task.
Your goal is to fill the parameters with specific and correct values used to solve the task.
Instructions:
1. You need to read the task carefully and determine the parameters' values needed for solving
the task.
2. Fill the parameters with specific values and return in this format:
"parameter name: parameter value".
3. Check and ensure the parameter value is in the correct format. If the parameter to fill is not
described explicitly in the description, please use the default value.
Output format:
Return the results in JSON format and do not include any prefixes or suffixes."""

Parameters_Validation_Prompt = """
I will give you parameters with their values in JSON format. I will also give you the parameter
description which regulates its format.
Your goal is to check whether the given parameters are in the correct format and modify the
wrong ones.
Instructions:
1. Carefully read the parameter description and check the format of the given parameters.
2. If the parameter is in the correct format, do not modify it.
3. If the format is incorrect, you should modify the parameter values.
4. Your modification should completely adhere to the parameter description and not create
parameter formats yourself.
Important:
If the format is "rdkit.Chem.rdchem.Mol", you should generate the SMILES string and use
"rdkit.Chem.MolFromSmiles" to convert the SMILES to "rdkit.Chem.rdchem.Mol". You
should use the function expression.
Output format:
Return the modified parameters in JSON format and do not include any prefixes or suffixes."""

Script_Generation_Prompt = """
I will give you some key-value pairs that describe the task, module name, function name, and
parameters with specific values.
Your goal is to write a script for calling the function with the given parameters.
Instructions:
1. Import the module in this format:
"import ChemGen.tools.module_name" or "import module_name".
The module name will be given in the user prompt under the "module_name" key.
2. Some parameters may need other packages. Please check the parameters and import the
required packages.
3. Create variables for the parameters and fill them with the given values.
4. Call the function with the parameters and print the result. When printing the result, you
need to describe what it means and not just print it.
Important:
The function name will be in the user prompt under the "function name" key.
Output format:
Return the script content only without any useless prefixes or suffixes.
"""

Error_Fixing_Prompt = """
I will give you a Python script and its error message.
Your goal is to fix the error in the script according to the error message.
Output format:
Return the fixed script content only, without any useless prefixes or suffixes like double
quotation or back quote marks to mark this as a Python file."""

Effectiveness_Checking_Prompt = """
I will give you the task, the planning steps for solving the task, the script for the task, and its
output.
Your goal is to determine whether the output is useful for solving the task.
The criteria for judging the uselessness of the output:
1. The output is an object without valid characters or numeric information. This one is
important and often appears. Please pay attention.
2. The output is discordant or irrelevant to the task.
3. The script does not follow the planning steps, focusing on checking the input variables and
output format.
4. The output is not the accurate data the task requires.
If you find the output is useless, you can modify the script according to the website given
below:
{website}
Output format:
1. Return the "useful" string only if the script output is useful.
2. Return the modified script content only if the output script is useless.
3. The modified script content should be without any useless prefixes or suffixes like double
quotation or back quote marks."""

Script_Fixing_Prompt = """
I will give you a Python script.
Your goal is to check whether the script can be executed successfully and fix it if it can't.
The situations you may encounter:
1. The script adds useless prefixes or suffixes and can't be executed successfully. For example,
the script contains **"```python" or "```"** at the beginning or end of the script.
2. The script does not have a try-except block to catch any exceptions and can't be executed
successfully.
Instructions:
1. If the script has "```python" or "```" at the beginning or end of the script, you must remove them.
2. I will give you the error message if an error exists. You should fix the error according to the error message.
3. When printing the error message, use the format: "Error: error message" to print the error
message in the try-except block.
Output format:
1. Return the script content only, ensuring your output can be executed directly.
2. You must check again whether there is useless prefixes or suffixes like ```python or ```.
If there is, you should remove them."""

Sufficiency_Validation_Prompt = """I will give you a task and the results of some tools used to solve the task.
Your goal is to judge whether the present results are sufficient for solving the task.
Output format:
1. Return the string "yes" only if the results are sufficient.
2. Return the string "no" only if the results are insufficient."""

Web_Search_Prompt = """
I will give you a task and the planning steps for solving the task.
Your goal is to search for the related information to solve the task online."""

Answer_Generation_Prompt = """
I will give you a task and some information generated from some tools for the task.
Your goal is to analyze and solve the task. You can choose useful information generated from
the tools to make your answer accurate and correct.
Instructions:
1. Read the task carefully and analyze its requirements.
2. Read the information given by the tools carefully and determine whether it can be used
directly.
3. If the information cannot be used directly, you should transform it according to the task’s
requirements.
4. If you receive multiple answers but they are different, you can process them in two ways:
(1) Choose the most accurate answer based on your judgment.
(2) If the answers have descriptions about how they are generated, you can output all answers
with their descriptions and let the user choose the most accurate one.
5. Ensure the answer has good readability. You can change the illustration format if needed. """

Polishment_Prompt = """
I will give you a task and the response generated by an LLM to solve the task. The response is based on the information provided by utility tools and web searches. 
Your goal is to polish the response. Please follow the instructions to polish.
Instructions:
1. Read the task and response carefully.
2. **Give the explicit answer to the task.**
3. You should output the **explicit** final answer of the task, extract the additional reasoning process, and supplement it after the final answer.
4. You should delete the unrelated information according to the task.

Output format:
Return the polished response only. **Do not add any useless prefixes or suffixes.**
"""

# ------------------- Function Definitions -------------------
async def instruction_generation(task: str, task_description: str, mode: str, instruction_file: str, constraint: str = None, batchsize: int = 10,  num: int = 10, metadata: str = None, metadata_type: str = None):

    if metadata_type and metadata:
        if metadata_type == "json":
            with open(metadata, 'r') as f:
                metadata_gen = json.load(f)
        elif metadata_type == "csv":
            with open(metadata, 'r') as f:
                df = pd.read_csv(f)
                metadata_gen = df.to_dict(orient='records')
        elif metadata_type == "txt":
            metadata_gen = metadata
        else: metadata_gen = None
    
    if num % batchsize!= 0:
        cycle = num // batchsize + 1
    else:
        cycle = num // batchsize
    
    for i in range(cycle):
        start = i * batchsize
        end = (i + 1) * batchsize
        if end > num:
            end = num

        if metadata_type == "txt":
            batch_length = end - start
            instructions = await generate_instructions(user_task=task, task_description=task_description, n=batch_length, mode=mode, metadata=metadata_gen, custom_constraint=constraint, model="gpt-4o")
        elif metadata_type == "json" or metadata_type == "csv":
            metadata_batch = metadata_gen[start:end]
            batch_length = len(metadata_batch)
            key1 = list(metadata_batch[0].keys())[0]
            if len(list(metadata_batch[0].keys()))>1:
                key2 = list(metadata_batch[0].keys())[1]
            query_batch = [m[key1] for m in metadata_batch]
            instructions = await generate_instructions(user_task=task, task_description=task_description, n=batch_length, mode=mode, metadata=query_batch, custom_constraint=constraint, model="gpt-4o")
        else:
            instructions = await generate_instructions(user_task=task, task_description=task_description, n=end-start, mode=mode, metadata=None, custom_constraint=constraint, model="gpt-4o")

        for idx, instruction in enumerate(instructions):
            info = {
                "task": task,
                "task_description": task_description,
                "constraint": constraint,
                "instruction": instruction,
            }

            if os.path.exists(instruction_file) == False:
                with open(instruction_file, 'w') as f:
                    json.dump([info], f, indent=4)
            else:
                with open(instruction_file, 'r') as f:
                    existing_data = json.load(f)
                    data = existing_data + [info]
                with open(instruction_file, 'w') as f:
                    json.dump(data, f, indent=4)
    
    print("Instruction generation completed.")
    
async def chat_with_model(system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> str:
    

    response = await aclient.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
           
    return response.choices[0].message.content

async def chat_with_CoT_model(user_prompt: str, model: str = "o3-mini") -> str:
    
    global total_tokens
    response = await aclient.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
       
    return response.choices[0].message.content

def script_error_logging(instruction: str, tool_name: str, module_name: str, file_name: str) -> None:
    
    logging_info = [
        {
            "instruction": instruction,
            "module_name": module_name,
            "tool_name": tool_name,
        }
    ]
    if os.path.exists(f'ChemGen/generation/{file_name}.json') == False:
        with open(f'ChemGen/generation/{file_name}.json', 'w') as f:
            json.dump(logging_info, f, indent=4)
    else:
        with open(f'ChemGen/generation/{file_name}.json', 'r') as f:
            existing_data = json.load(f)
            data = existing_data + logging_info
        with open(f'ChemGen/generation/{file_name}.json', 'w') as f:
            json.dump(data, f, indent=4)
            

async def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Get the embedding of the given text using the openai embedding API."""
    
    text = text.replace("\n", " ")
    response = await aclient.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def tools_embedding(file_path: str) -> None:
    """
    Generate the embedding of the tools' descriptions and save the embedding and tool information to a pickle file.
    For conviently using the tools' information, the tool information is saved as key-value pairs in a pickle file."""
    
    tools_descriptions = []
    tools_embedding = []
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        modules = list(config.keys())

        for module in modules:
            tools = list(config[module].keys())
            for tool in tools:
                tools_descriptions.append(config[module][tool]['description'])
                tools_embedding.append(
                    {
                        "tool": tool,
                        "module": module,
                        "description": config[module][tool]['description'],
                        "embedding": get_embedding(config[module][tool]['description']),
                        "parameters":{
                            f"{key}": config[module][tool][key]['description'] for key in list(config[module][tool].keys()) if key != "description"
                         }
                }
                )

        with open('tools_embedding.pkl', 'wb') as f:
            pickle.dump(tools_embedding, f)
            print(f"Tools embedding saved to tools_embedding.pkl")

async def script_fixing(script: str, script_fixing_num: int = 3, model: str = "o1-mini") -> str:
    """
    Check and fix the generated script for grammar and encoding errors.
    
    Addtional tips for this function:
    1. This function is used for ensuring there is no useless prefix or suffix and no encoding errors in the generated script.
    2. After numerous testing, the model o3-mini seems to have good performance in fixing the errors.
    3. You can change the model to o3-mini by changing the model parameter in the function.
    4. If the effectiveness is stable, you can consider delete the try-except block and the parameter checking_num."""

    while script_fixing_num >= 0:
        
        try:
            with open('ChemGen/generation/temp.py', 'w', encoding = 'UTF-8') as f:
                f.write(script)
            subprocess.run(['python', f'ChemGen/generation/temp.py'], capture_output=True, text=True, check=True)
            error = None
        except subprocess.CalledProcessError as e:
            error = e.stderr
            #If the error still exists after the maximum number of checking, return None.
            if script_fixing_num == 0:
                return None
        
        if error == None:
            file_path = os.path.join(os.getcwd(), f"ChemGen/generation/temp.py")
            os.remove(file_path)
            return script
       
        prompt = f"{Script_Fixing_Prompt}\n\nerror message: {error}\n\nscript: {script}"
        script = await chat_with_CoT_model(user_prompt=prompt)
        script_fixing_num -= 1
        print(f"fixed script: {script}")
    
    file_path = os.path.join(os.getcwd(), f"ChemGen/generation/temp.py")
    os.remove(file_path)
    return script

def calculate_cosine_similarity(description_embedding: list[float], tool_embedding: list[float]) -> float:
    """
    Calculate the cosine similarity between the needed tool description and the tool function embedding."""
    
    vec1 = torch.FloatTensor(description_embedding)
    vec2 = torch.FloatTensor(tool_embedding)
    cosine_similarity = F.cosine_similarity(vec1, vec2, dim=0)

    return cosine_similarity.item()

async def topk_tools_selection(tool_descriptions: list[str], top_k: int = 5) -> list:
    """
    Select the top k tools based on the cosine similarity between the needed tool description and the tool function embedding."""
    
    tool_index_list = []
    descriptions_embedding = [await get_embedding(description) for description in tool_descriptions]

    for i, d_embedding in enumerate(descriptions_embedding):
        similarity_scores = [
            {
                "index": idx,
                "similarity_score": calculate_cosine_similarity(d_embedding, t_embedding['embedding'])
            }
            for idx, t_embedding in enumerate(tools_info)
        ]
        similarity_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        tool_index = []
        for similarity_score in similarity_scores[0:top_k]:
            tool_index.append(similarity_score["index"])
        tool_index_list.append(tool_index)
        print(f"{i+1}th insturction has finished calculating similarity scores")

    return tool_index_list

async def tool_selection(instruction: str, tool_descriptions: list[str], k: int = 5, metadata: str = None, model: str = "gpt-4o") -> list[int]:
    """
    Select the tools for solving the user task with reference to the raw instruction and the tools' descriptions."""
    
    tool_index_list = await topk_tools_selection(tool_descriptions, k)  
    selected_tools_with_description = []

    for i in range(len(tool_index_list)):
        tools = []
        for j in range(k):
            tools.append(
                {
                    "index": tool_index_list[i][j],
                    "tool name": tools_info[tool_index_list[i][j]]['tool'],
                    "description": tools_info[tool_index_list[i][j]]['description']
                }
            )
        selected_tools_with_description.append(tools)
    
    selected_tools_with_description = [item for sublist in selected_tools_with_description for item in sublist]
    selected_tools_with_description_final = []
    for tool in selected_tools_with_description:
        if tool not in selected_tools_with_description_final:
            selected_tools_with_description_final.append(tool)
    tool_selection_prompts = []
        
    if metadata == None:
        metadata = "No additional metadata provided."

    print(f"metadata: {metadata}\n\n")

    for tool in selected_tools_with_description_final:
        tool_selection_prompts.append(f"metadata: {metadata}\nraw task: {instruction}\nThe tool you need to judge: {tool}")

    tools_list = []
    for tool_selection_prompt in tool_selection_prompts:
        response = await chat_with_model(system_prompt=Tool_Selection_Prompt, user_prompt=tool_selection_prompt, model=model)
        
        if response == "no":
            tools_list.append(response)
        else:
            tools_list.append(int(response))
    
    name = []
    for idx in tools_list:
        if idx != "no":
            name.append(tools_info[idx]['tool'])
    print(f"selected tools: {name}")
       
    return await tools_distilling(instruction, tools_list)

async def tools_distilling(instruction: str, tools_list: list[int], metadata: str = None, model: str = "o3-mini") -> list[int]:
    """
    Distill the tools after the tool selection process, remove the tools that are not the expert for the task."""
    tools = []
    for tool in tools_list:
        if tool!= "no":
            tools.append(
                {
                    "index": tool,
                    "tool name": tools_info[tool]['tool'],
                    "description": tools_info[tool]['description']
                }
            )
    mark = "to be confirmed"
    removed_tools = []
    metadata = metadata if metadata != None else "No additional metadata provided."
    while mark != "no":
        prompt = f"metadata: {metadata}\nraw task: {instruction}\nThe tools you need to check: {tools}\n{Tool_Distillation_Prompt}"
        mark = await chat_with_CoT_model(user_prompt=prompt, model=model)
        print(f"mark for distilling: {mark}")
        if mark != "no":
            redundant_tool_index = int(mark)
            for tool in tools:
                if tool['index'] == redundant_tool_index:
                    removed_tools.append(tool)
                    break
            
            tools = [tool for tool in tools if tool['index'] != redundant_tool_index]

            
    return [tool['index'] for tool in tools]
    
async def sufficiency_validation(instruction: str, result: list[str], model: str = "gpt-4o") -> bool:
    """
    Judge whether the present results are sufficient for solving the task."""

    prompt = f"raw task: {instruction}\nresult: {result}"
    response = await chat_with_model(system_prompt=Sufficiency_Validation_Prompt, user_prompt=prompt, model=model)
    
    if response == "yes":
        return True
    else:
        return False

async def tool_calling(instruction: str, planning_steps: list[str], tool_list: list, metadata: str = None, metadata_type: str = None, model: str = "gpt-4o", error_fixing_num: int = 3, effectiveness_checking_num: int = 5, diversity_generation: bool = True) -> list[str]:
    """
    Call the tools have been selected for solving the user task.There are four module in this function:
    1. Parameters filling and checking: Fill the parameters with specific values and check whether the parameters are in correct format according to tools' information.
    2. Script generation: Generate the script for calling the tool with the given parameters.
    3. Error processing: If the script can't be executed successfully, fix the error according to the error message and try again.
    4. Effectiveness confirmation: Judge whether the results generated by the tools are useful and correct for solving the task. Fix the script according to the documentations(pubchempy and rdkit) online if necessary.
    
    Args:
        instruction (str): The user instruction for the task.
        tool_list (list[int]): The list of tool indexes selected for solving the task.
        model (str, optional): The model used for the chatbot. Defaults to "gpt-4o".
        planning_steps (list[str], optional): The planning steps for solving the task.
        error_fixing_num (int, optional): The maximum number of error fixing steps. Defaults to 2.
        effectiveness_confirmation_num (int, optional): The maximum number of effectiveness confirmation steps. Defaults to 3.
        diversity_generation (bool, optional): Whether to generate diversity results. Defaults to True.

    Returns:
        list[str]: The results generated by the tools.
    """
    tool_calling_result = []
    if metadata_type == "pickle":
        tools = metadata
    else: 
        tool_called_index = [int(tool) for tool in tool_list if tool!= "no"]
        tool_called_index = list(set(tool_called_index))
        tools = [tools_info[tool] for tool in tool_called_index]

    if len(tools) == 0:
        print("no tool needed")
        return None
    else:
        for tool in tools:
            tool_name = tool['tool']
            module_name = tool['module']
            parameters_with_description = tool['parameters']
            fixing = error_fixing_num
            retrieval = effectiveness_checking_num
            flag_continue = False
            # Parameters filling and checking-------------------------------------------------
            prompt = f"task: {instruction}\nfunction name: {tool_name}, parameters with description: {parameters_with_description}"
    
            response = await chat_with_model(system_prompt=Parameters_Filling_Prompt, user_prompt=prompt, model=model)
        
            print(response + '\n')
            parameters = response
            
            prompt = f"parameters in json format: {parameters}, parameters with description: {parameters_with_description}"
            modified_parameters = await chat_with_model(system_prompt=Parameters_Validation_Prompt, user_prompt=prompt, model=model)
            
            print(f"modifying step: {modified_parameters} + '\n'")
            
            # Script generation---------------------------------------------------------------
            
            prompt = f"{Script_Generation_Prompt}\n\ntask: {instruction}\nmodule name: {module_name}, function name: {tool_name}, parameters: {modified_parameters}"
            response = await chat_with_CoT_model(user_prompt=prompt)
            print(response + '\n')
           
            #------------------------------------------------------------------
            script = await script_fixing(response)
            if script == None:
                
                script_error_logging(instruction, tool_name, module_name, "error_log")
                print("**script generation failed in script legality check module, continue to the next tool**\n\n")
                continue
            #---------------------------------------------------------------------------------
            
            with open(f"ChemGen/generation/{tool_name}.py", 'w', encoding='UTF-8') as f:
                f.write(script)
            result = subprocess.run(['python', f"ChemGen/generation/{tool_name}.py"], capture_output=True, text=True, check=True)
            # Error processing----------------------------------------------------------------
            
            while result.stdout[0:5] == "Error" and fixing > 0:
                
                prompt = f"script: {script}, error message: {result.stdout[7:]}"
                response = await chat_with_model(system_prompt=Error_Fixing_Prompt, user_prompt=prompt, model=model)
                print(f"fixing step according to error message({error_fixing_num-fixing+1}/{error_fixing_num}):\n{response} + '\n'")
                fixing -= 1
                script = await script_fixing(response)
                if script == None:
    
                    script_error_logging(instruction, tool_name, module_name, "error_log")
                    print("**script generation failed in script legality check module, continue to the next tool**\n\n")
                    flag_continue = True
                    break

                with open(f"ChemGen/generation/{tool_name}.py", 'w', encoding = 'UTF-8') as f:
                    f.write(script)
                result = subprocess.run(['python', f"ChemGen/generation/{tool_name}.py"], capture_output=True, text=True, check=True)
            
            if flag_continue == True:
                continue
            # Effectiveness confirmation according to documents-------------------------------

            response = "To be confirmed"
            global Effectiveness_Checking_Prompt
            if metadata_type == "pickle":
                Effectiveness_Checking_Prompt = Effectiveness_Checking_Prompt.format(website = tool['documentation'])
            else:
                website = """1. PubChemPy documentation(this is used in the pubchem_tool):\
                https://pubchempy.readthedocs.io/en/latest/\
                2. RDKit documentation(this is used in the rdkit_tool):\
                https://www.rdkit.org/docs/index.html"""
                Effectiveness_Checking_Prompt = Effectiveness_Checking_Prompt.format(website = website)
    
            while response[0:6] != "useful" and retrieval > 0:
                if planning_steps != None:
                    prompt = Effectiveness_Checking_Prompt + f"raw task: {instruction}\nscript: {script}\nplanning steps: {planning_steps}\nresult: {result.stdout}"
                else:
                    prompt = Effectiveness_Checking_Prompt + f"raw task: {instruction}\nscript: {script}\nresult: {result.stdout}"
                
                response = await web_search_retrieve(prompt = prompt, search_context_size= "high")
                
                if response != "useful":
                    script = await script_fixing(response)
                    if script == None:
                        
                        script_error_logging(instruction, tool_name, module_name, "error_log")
                        print("**script generation failed in script legality check module, continue to the next tool**\n\n")
                        flag_continue = True
                        break
                print(f"effectiveness step:\n{script} + '\n'")

                with open(f"ChemGen/generation/{tool_name}.py", 'w', encoding = 'UTF-8') as f:
                        f.write(script)
                result = subprocess.run(['python', f"ChemGen/generation/{tool_name}.py"], capture_output=True, text=True, check=True)
                retrieval -= 1
            
            if flag_continue == True:
                continue
               
            tool_calling_result.append(result.stdout)
            #Delete the generated script after the effectiveness confirmation process.
            file_path = os.path.join(os.getcwd(), f"ChemGen/generation/{tool_name}.py")
            os.remove(file_path)
        
            if diversity_generation == False and await sufficiency_validation(instruction, tool_calling_result):
                break
        print(f"result: {tool_calling_result}\n")
        return tool_calling_result

async def answer_generation(instruction: str, tool_calling_result: str, model="gpt-4o") -> str:
    """
    Generate the answer based on the tool calling results
    
    Args:
        instruction (str): The user instruction for the task.
        tool_calling_result (str): The results generated by the tools.
        model (str, optional): The model used for the chatbot. Defaults to "gpt-4o".

    Returns:
        str: The answer generated by the tools retrieval system.
    """
    
    prompt = f"task: {instruction}\nThe information given by the tools is: {tool_calling_result}"
    response = await chat_with_model(system_prompt=Answer_Generation_Prompt, user_prompt=prompt, model=model)
    return response

async def polish(instruction: str, raw_response: str) -> str:

    
    user_prompt = f"Task: {instruction}\nResponse: {raw_response}\n"
    response = await chat_with_model(system_prompt=Polishment_Prompt, user_prompt=user_prompt, model="gpt-4o")
    return response

async def tools_retrieval(instruction: str, RG_metadata_type: str = None, RG_metadata_content: str = None, error_fixing_num: int = 2, effectiveness_confirmation_num: int = 3, diversity_generation: bool = False) -> str:
    """
    The main function for the tools retrieval system.
    
    Args:
        instruction (str): The user instruction for the task.
        model (str, optional): The model used for the chatbot. Defaults to "gpt-4o".
        error_fixing_num (int, optional): The maximum number of error fixing steps. Defaults to 2.
        effectiveness_confirmation_num (int, optional): The maximum number of effectiveness confirmation steps. Defaults to 3.
        diversity_generation (bool, optional): Whether to generate diversity results. Defaults to False.

    Returns:
        str: The answer generated by the tools retrieval system.
    """
    if RG_metadata_type == "text":
        metadata = RG_metadata_content
    elif RG_metadata_type == "pickle":
        with open(RG_metadata_content, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = None
    
    if RG_metadata_type == "pickle":
        planning_steps = await generate_planning_steps(instruction)
        tools_list = None
    else:
        planning_steps = await generate_planning_steps(instruction, metadata = metadata)
        tool_descriptions = await generate_tool_descriptions(planning_steps, metadata = metadata)
        tools_list = await tool_selection(instruction, tool_descriptions, metadata=metadata)
        print(f"tools_list: {[tools_info[tool]['tool'] for tool in tools_list if tool != 'no']}")
        
    tool_calling_result =await tool_calling(instruction, planning_steps, tools_list, metadata=metadata, metadata_type=RG_metadata_type, error_fixing_num=error_fixing_num, effectiveness_checking_num=effectiveness_confirmation_num, diversity_generation=diversity_generation)

    prompt = f"{Web_Search_Prompt}\n\nTask: {instruction}\nPlanning steps: {planning_steps}"
     
    #Judge whether the web retrieval is needed.
    #If no tool is selected or the tool calling result is not effective, the web retrieval is needed.
    
    if tool_calling_result == None:
        print("***web retreival is needed***\n")
        tool_calling_result = await web_search_retrieve(prompt)
    else:
        # sufficiency experiment
        if await sufficiency_validation(instruction, tool_calling_result) == False:
            print("***web retreival is needed***\n")
            tool_calling_result = f"The information generated by the tools is: {tool_calling_result}\nThe information from the web is: {await web_search_retrieve(prompt)}"
                   
    answer = await answer_generation(instruction, tool_calling_result, model="gpt-4o")
    return await polish(instruction, answer)
        
async def instruction_response_pair_generation(user_query: str, instruction: str, output_file_path: str, idx: int, metadata: str = None, metadata_type: str = None):
    """
    Generate the instruction response pair for the given user query and instruction and save it to a file.
    
    Args:
        user_query (str): The general description of the task.
        instruction (str): The specific instruction for the task.
        output_file_path (str): The path to the output file.
        idx (int): The index of the instruction response pair.
    
    Returns:
        None
    """
    
    instruction_response_pairs = []
     
    tools_retrieval_result = await tools_retrieval(instruction, RG_metadata_content=metadata, RG_metadata_type=metadata_type)
    instruction_response_pairs.append(
        {
            "task": user_query,
            "instruction": instruction,
            "response": tools_retrieval_result,
        }
    )
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding= 'UTF-8') as f:
            data = json.load(f)
            data = data + instruction_response_pairs
        with open(output_file_path, 'w', encoding= 'UTF-8') as f:
            json.dump(data, f, indent=4)
    else:
        with open(output_file_path, 'w', encoding= 'UTF-8') as f:
            json.dump(instruction_response_pairs, f, indent=4)

    print(f"index: {idx}, instruction response pair has been created successfully.\n")
    
async def concurrent_instruction_response_pair_generation(file_path: str, output_file_path: str, metadata: str = None, metadata_type: str = None, concurrent_num: int = 10):
    """
    Generate the instruction response pairs concurrently for the given file.

    Args:
        file_path (str): The path to the input file containing the instructions.
        output_file_path (str): The path to the output file.
        concurrent_num (int, optional): The number of concurrent tasks. Defaults to 10.
    
    Returns:
        None
    """

    with open(file_path, 'r', encoding= 'UTF-8') as f:
        data = json.load(f)
    
    tasks = []
    for idx, item in enumerate(data):
        tasks.append(asyncio.create_task(instruction_response_pair_generation(item['task'], item['instruction'], output_file_path, idx, metadata=metadata, metadata_type=metadata_type)))
        if len(tasks) == concurrent_num or idx == len(data) - 1:
            await asyncio.gather(*tasks)
            tasks = []
    
    print(f"All instruction response pairs have been created successfully.\n")

async def generate_response(instruction: str, mode: str = None, metadata_type: str = None, metadata_content: str = None, model: str = "gpt-4o"):

    planning_steps = await generate_planning_steps(instruction)
    #if mode == "metadata" and metadata_type == "pickle":
        #answer = await answer_generation(instruction, await tool_calling(instruction, planning_steps, additional_tool, metadata=additional_tool, metadata_type=metadata_type, model=model))
        #print(answer)
    
    #if mode == "metadata" and metadata_type == "json":
        #response = await tools_retrieval(instruction, metadata=metadata_content, metadata_type=metadata_type, model=model)

async def ChemOrch(task: str, task_description: str, IGmode: str, RGmode: str, instruction_file: str, output_file: str, num: int, 
                   batchsize: int = 10, IG_metadata_type: str = None, RG_metadata_type: str = None, IG_metadata_content: str = None, RG_metadata_content: str = None, 
                   constraint: str = None, model: str = "gpt-4o",):
    
    await instruction_generation(task, task_description, IGmode, instruction_file, num = num, batchsize = batchsize, metadata = IG_metadata_content, metadata_type = IG_metadata_type, constraint = constraint)
   
    await concurrent_instruction_response_pair_generation(instruction_file, output_file, metadata=RG_metadata_content, metadata_type = RG_metadata_type)
    
    
    


#Example usage:
if __name__ == '__main__':
    
    task = "SMILES conversion"
    task_description = "predict the SMILES string of a given compound."
    instruction_file = "ChemGen/results/test_IG_pkl.json"
    output_file = "ChemGen/results/test_IG_pkl_output.json"
    ig_metadata_type = "txt"
    ig_metadata_content = "ethanol, benzene, aspirin"
    constraint = "Your instruction should provide specific value and focus on the specific task.Please use the provided tool in metadata for solving the task."
    rg_metadata_type = "pickle"
    rg_metadata_content = "ChemGen/metadata/additional_tool.pickle"
    """
    task = "Property prediction"
    task_description = "Predict the blood-brain barrier penetration of a given compound."
    instruction_file = "ChemGen/generation/experiment/baseline_contrast/contrast_dataset_instructions_PP.json"
    output_file = "ChemGen/generation/experiment/baseline_contrast/contrast_dataset_v4.0.json"
    ig_metadata_content = "ChemGen/generation/experiment/baseline_contrast/PP_metadata.json"
    constraint = "Please generate instructions adhere to the metadata and keep the sequence of instructions consistent with metadata infomation."
    rg_metadata_content = "Please select suitable tools to solve the task as far as possible."
    """
    asyncio.run(ChemOrch(task, task_description, "metadata", "text", instruction_file, output_file, num = 2, batchsize = 10,
                         IG_metadata_type = ig_metadata_type, IG_metadata_content = ig_metadata_content,
                        RG_metadata_type = rg_metadata_type, RG_metadata_content = rg_metadata_content,
                         constraint = constraint, model = "gpt-4o"))
    #task_name = "Transition State Identification"
    #task_description = "The task is identifying the likely transition state structure for a given chemical reaction."
    #constraint = "The instruction should focus on specific chemical problems but not general problems."
    """
    instruction = "Give me the SMILES string of ethanol."
    mode = "metadata" # metadata, tool, hybrid
    metadata_type = "pickle"# text, csv, json, pickle
    metadata = "ChemGen/generation/experiment/case_study/additional_tool.pickle"
    asyncio.run(generate_response(instruction, mode, metadata_type, metadata))
    """
    """
    additional_tool = [{
        "tool": "smiles_from_compound",
        "module": "ord_schema.message_helpers",
        "description": "Fetches or generates a SMILES identifier for a compound. If a SMILES identifier already exists, it is simply returned.",
        "parameters": {"compound": "reaction_pb2.Compound message."},
        "documentation": "https://docs.open-reaction-database.org/en/latest/ord_schema/ord_schema.html#module-ord_schema.message_helpers"
    }]

    with open("ChemGen/metadata/additional_tool.pickle", "wb") as f:
        pickle.dump(additional_tool, f)
    """
    
    
    
    







