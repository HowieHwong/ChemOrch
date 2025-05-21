import os
import functools
import time
import re
import json
import logging
import tiktoken
import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from functools import wraps
from openai import AsyncOpenAI
import asyncio

load_dotenv()

# os.environ['http_proxy'] = os.getenv('HTTP_PROXY')
# os.environ['https_proxy'] = os.getenv('HTTPS_PROXY')


class TokenLogger:
    def __init__(self, filename="token_log.txt"):
        self.filename = filename
        self.model_tokens = {}
        self.load_tokens()

    def log_tokens(self, model_name, input_tokens, output_tokens):
        if model_name not in self.model_tokens:
            self.model_tokens[model_name] = {"input_total": 0, "output_total": 0, "last_input": 0, "last_output": 0}
        
        self.model_tokens[model_name]["input_total"] += input_tokens
        self.model_tokens[model_name]["output_total"] += output_tokens
        self.model_tokens[model_name]["last_input"] = input_tokens
        self.model_tokens[model_name]["last_output"] = output_tokens

        self.save_tokens()

    def get_total_tokens(self, model_name):
        if model_name in self.model_tokens:
            return self.model_tokens[model_name]["input_total"], self.model_tokens[model_name]["output_total"]
        else:
            return 0, 0

    def save_tokens(self):
        with open(self.filename, 'w') as file:
            for model_name, tokens in self.model_tokens.items():
                file.write(f"{model_name},{tokens['input_total']},{tokens['output_total']},{tokens['last_input']},{tokens['last_output']}\n")

    def load_tokens(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                for line in file:
                    model_name, input_total, output_total, last_input, last_output = line.strip().split(',')
                    self.model_tokens[model_name] = {
                        "input_total": int(input_total),
                        "output_total": int(output_total),
                        "last_input": int(last_input),
                        "last_output": int(last_output)
                    }
        else:
            self.model_tokens = {}

token_logger = TokenLogger()

def num_tokens_from_string(string: str, encoding_name='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def retry_on_failure(max_retries=3, delay=1, backoff=2):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    print(f"Exception occurred: {e}")

                retries += 1
                print(f"Retrying ({retries}/{max_retries}) in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff
            return None
        return wrapper_retry
    return decorator_retry

def token_logger_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model_name = args[0] if len(args) > 0 else kwargs.get('model', '')
        prompt = args[1] if len(args) > 1 else kwargs.get('prompt', '')
        
        input_tokens = num_tokens_from_string(prompt)
        response = func(*args, **kwargs)
        output_tokens = num_tokens_from_string(response)
        token_logger.log_tokens(model_name, input_tokens, output_tokens)
        return response
    return wrapper

model_dict = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o3-mini': 'o3-mini',
    'chatgpt-4o-latest': 'chatgpt-4o-latest',
    'llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct',
    'gemma-2-27B': 'google/gemma-2-27b-it',
    'qwen-2.5-72B': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen-2.5-7B': 'Qwen/Qwen2.5-7B-Instruct',
    'yi-lightning': 'yi-lightning',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
}

#@retry_on_failure()
#@token_logger_decorator
async def get_response(model='chatgpt-4o-latest', prompt=None, temperature=0.001, history=[], system_prompt=''):
    
    if model in ['claude-3.5-sonnet']:
        client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
        )
        messages = []
        for ms in history:
            messages.append({'role': ms['role'], 'content': ms['content']})
        messages.append({'role': 'user', 'content': prompt})

        message = await client.messages.create(
            model=model_dict[model],
            max_tokens=2048,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        return message.content[0].text
    if model in ['llama-3.1-70B', 'llama-3.1-8B', 'qwen-2.5-72B', 'gemma-2-27B', 'llama-3.3-70B', 'qwen-2.5-7B', 'deepseek-r1']:
        aclient = AsyncOpenAI(
            api_key=os.getenv('DEEPINFRA_API_KEY'),
            base_url=os.getenv('DEEPINFRA_BASE_URL')
        )
    elif model == 'yi-lightning':
        aclient = AsyncOpenAI(
            api_key=os.getenv('YI_API_KEY'),
            base_url=os.getenv('YI_BASE_URL')
        )
    else:
        print('Using OpenAI API')
        aclient = AsyncOpenAI(api_key="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")
    
    if model in ['o3-mini', 'gpt-4o', 'gpt-4o-mini']:
        messages = []
        for ms in history:
            messages.append({'role': ms['role'], 'content': ms['content']})
        messages.append({'role': 'user', 'content': prompt})
        response = await aclient.chat.completions.create(
            model=model_dict[model],
            messages=messages,
        )
        
        return response.choices[0].message.content
    messages = []
    for ms in history:
        messages.append({'role': ms['role'], 'content': ms['content']})
    messages.append({'role': 'user', 'content': prompt})
    
    response = await aclient.chat.completions.create(
        model=model_dict[model],
        messages=messages,
        temperature=temperature,
    )

    result_content = response.choices[0].message.content
    return result_content

@retry_on_failure()
@token_logger_decorator
def get_structured_response(model='gpt-4o', prompt=None, history=[], temperature=0.5, response_format=None):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    messages = [{"role": message['role'], "content": message['content']} for message in history]
    messages.append({'role': 'user', 'content': prompt})
    # messages.insert(0, {"role": "system", "content": "You are a skilled math probelm solver. Solve the following problem step by step."})

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=response_format
    )

    result_content = response.choices[0].message.parsed
    return result_content.model_dump_json(indent=4)

def num_tokens_from_string(string: str, encoding_name='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
async def main():
    prompt = "What is the capital of France?"
    response = await get_response(model='gpt-4o', prompt=prompt)
    print(response)

if __name__ == '__main__':
    asyncio.run(main())
    #print(get_response(model='gpt-4o-mini', prompt='What is the capital of France?'))