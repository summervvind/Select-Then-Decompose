from openai import AsyncOpenAI, OpenAI
import asyncio
from typing import List
import json
import re
import ast

API_KEY = ''
BASE_URL = ''

FORMAT_PROMPTS = {
    "GSM8K": "The final numerical answer",
    "Multiarith": "The final numerical answer",
    "MATH": "The final answer in LaTeX format wrapped in \boxed{} (If it is a percentage, please end with \%)",
    "HumanEval": "The final python code",
    "HotpotQA": "The final concise answer in a few words",
    "DROP": "The final concise answer contains only one word/few words",
    "Trivia_Creative_Writing": "The final story",
    "MT_Bench_Writing": "The final text",
    "MT_Bench_Math": "The final concise answer",
    "MT_Bench_Coding": "The final summary and code",
}
NEED_REF_CATS = ["math", "reasoning", "coding"]

def generate_format_prompt(benchmark_name:str, question: str, solution: str) -> str:
        """
        Generates a prompt for the OpenAI API using a predefined template.
        """
        format_prompt = FORMAT_PROMPTS[benchmark_name]
        template = """
For the question described as {question},
please extract {format_prompt} from the following solution: 

{solution}

Ensure that your response does not modify the answers in the solution and there are no other comments or explanations.

Please reply in this format:
<Answer>
Place the extracted content here
</Answer>
"""
        return template.format(format_prompt=format_prompt, question=question, solution=solution)

async def get_format_output(benchmark_name:str, question: str, solution: str) -> tuple:
    """
    Executes the logic for generating a prompt, calling the OpenAI API, and returning the result.
    """
    prompt = generate_format_prompt(benchmark_name, question, solution)
    api_key = API_KEY
    base_url = BASE_URL
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.0,
        stream=False,
    )
    result = response.choices[0].message.content
    # Return the result and usage statistics
    return result


def generate_judge_prompts(type:str, questions:str, outputs:str, references:List[str]):
    judge_prompts_file = 'dataset/mt_bench/judge_prompts.jsonl'
    with open(judge_prompts_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data.get('name') == type:
                judge_dicts = data
    system_prompt = judge_dicts["system_prompt"]
    prompt_template = judge_dicts["prompt_template"]
    if type == "single-v1":
        judge_prompt = prompt_template.format(question=questions[0], answer=outputs[0])
    elif type == "single-math-v1":
        judge_prompt = prompt_template.format(question=questions[0], answer=outputs[0], ref_answer_1=references[0])
    elif type == "single-v1-multi-turn":
        judge_prompt = prompt_template.format(question_1=questions[0], answer_1=outputs[0], question_2=questions[1], answer_2=outputs[1])
    elif type == "single-math-v1-multi-turn":
        judge_prompt = prompt_template.format(question_1=questions[0], answer_1=outputs[0], question_2=questions[1], answer_2=outputs[1], ref_answer_1=references[0], ref_answer_2=references[1])
    return system_prompt, judge_prompt

async def get_llm_score(category, questions, outputs, expected_outputs, i) -> tuple:
    """
    For mt-bench, executes the logic for generating a prompt, calling the OpenAI API, and returning the score.
    """
    if i == 0:
        if category in NEED_REF_CATS: type = "single-math-v1"
        else: type = "single-v1"
    if i == 1:
        if category in NEED_REF_CATS: type = "single-math-v1-multi-turn"
        else: type = "single-v1-multi-turn"
    
    system_prompt, judge_prompt = generate_judge_prompts(type, questions, outputs, expected_outputs)
    api_key = API_KEY
    base_url = BASE_URL
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await async_client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": judge_prompt},
        ],
        max_tokens=2048,
        temperature=0.0,
        stream=False,
    )
    judgment = response.choices[0].message.content

    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)
    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
    # Return the result and usage statistics
    return rating, judgment