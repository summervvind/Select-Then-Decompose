import sys
import os
import warnings
import asyncio

if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai_call import get_openai_response
import argparse
from collections import defaultdict, deque
import json
import re
from method.basemethod import BaseMethod  # Assuming basemethod.py contains the BaseMethod class
import copy
from typing import Optional

class DAGFlow(BaseMethod):
    def extract_json(self, text: str) -> Optional[str]:
        """
        Extracts the JSON portion from the text.
        """
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def extract_subtasks_from_json(self, text: str) -> dict:
        """
        Extracts subtasks and their dependencies from the JSON-formatted text.
        """
        json_obj = self.extract_json(text)
        if not json_obj:
            return {}

        if isinstance(json_obj, str):
            try:
                data = json.loads(json_obj)
            except Exception as e:
                print("JSON decode error:", e)
                print("json_str:", json_obj)
                return {}
        else:
            data = json_obj

        try:
            subtasks = {}
            for task in data["subtasks"]:
                subtasks[task["id"]] = (task["description"], task["dependencies"])
            return subtasks
        except Exception as e:
            print("Subtasks extraction error:", e)
            print("data:", data)
            return {}

    def topological_sort(self, subtasks: dict) -> list:
        """
        Performs topological sorting on the subtasks based on their dependencies.

        :param subtasks: Dictionary where keys are subtask IDs and values are (description, dependencies).
        :return: List of subtask IDs in topological order.
        :raises ValueError: If there is a circular dependency.
        """
        graph = defaultdict(list)  # Adjacency list representation of the graph
        in_degree = {task: 0 for task in subtasks}  # In-degree of each task

        # Build the graph and in-degree counts
        for task, (_, dependencies) in subtasks.items():
            for dep in dependencies:
                graph[dep].append(task)
                in_degree[task] += 1

        # Initialize queue with tasks having no dependencies
        queue = deque([task for task in in_degree if in_degree[task] == 0])
        sorted_tasks = []  # Result of topological sort

        # Perform topological sort
        while queue:
            task = queue.popleft()
            sorted_tasks.append(task)

            for neighbor in graph[task]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for circular dependencies
        if len(sorted_tasks) != len(subtasks):
            raise ValueError("Circular dependencies detected in subtasks.")

        return sorted_tasks

    def generate_prompt(self, question: str) -> str:
        """
        Generates the initial prompt for the DAG flow method.
        """
        template = '''Let's first understand the following problem and devise a directed acyclic graph (DAG) of subtasks to solve the problem.

{input}

Use the following JSON format to break down the problem into subtasks:

{subtasks_example}

Rules:
1. Each subtask must have a unique ID (e.g., "Subtask 1").
2. Each subtask must have a clear description of what needs to be done.
3. If a subtask depends on other subtasks, list their IDs in the "dependencies" field.
4. Ensure the subtasks form a directed acyclic graph (DAG) with no circular dependencies.
5. Provide only the JSON output. Do not include any additional text.

Begin!
'''
        subtasks_example = '''
{
    "subtasks": [
        {
            "id": "Subtask 1",
            "description": "[First step to solve the problem]",
            "dependencies": []
        },
        {
            "id": "Subtask 2",
            "description": "[Second step to solve the problem]",
            "dependencies": ["Subtask 1"]
        },
        {
            "id": "Subtask 3",
            "description": "[Third step to solve the problem]",
            "dependencies": ["Subtask 1", "Subtask 2"]
        }
    ]
}'''
        return template.format(input=question, subtasks_example=subtasks_example)

    async def execute(self, question: str) -> tuple:
        """
        Executes the DAG flow method by generating a plan, performing topological sorting, and executing subtasks.
        """
        prompt_tokens = 0
        completion_tokens = 0
        calls = 0
        # Generate the plan
        try:
            plan_prompt = self.generate_prompt(question)
            # plan_args = copy.copy(self.args)  
            # plan_model = getattr(self.args, 'plan_model', self.args.model)
            # plan_args.model = plan_model
            plan, usage = await get_openai_response(self.args, plan_prompt)

            # Check if plan is None
            if plan is None:
                raise ValueError("Error: Failed to generate plan from LLM response.")

            # Extract subtasks and their dependencies
            subtask_dict = self.extract_subtasks_from_json(plan)
            if not subtask_dict:
                raise ValueError("Error: Failed to extract subtasks from the plan.")
            # Perform topological sorting

            sorted_tasks = self.topological_sort(subtask_dict)
            
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            calls += 1

            # Initialize context
            context = ""
            results = {}

            # Execute subtasks in topological order
            for subtask_id in sorted_tasks:
                subtask_description, subtask_dependencies = subtask_dict[subtask_id]

                # Build subtask prompt
                subtask_prompt = f"""Here is the original question: {question}

Here is the context of previous subtasks and their results:
{context}

The current subtask is: {subtask_id}: {subtask_description}

Dependencies: {subtask_dependencies}

Please execute this subtask and provide the result. If the subtask depends on previous subtasks, use their results to complete the task.
"""
                # Execute subtask
                # subtask_args = copy.copy(self.args)  
                # execute_model = getattr(self.args, 'execute_model', self.args.model)
                # subtask_args.model = execute_model
                result, usage = await get_openai_response(self.args, subtask_prompt)
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                calls += 1

                # Save result
                results[subtask_id] = result

                # Update context
                context += f"{subtask_id}: {subtask_description} Dependencies: {subtask_dependencies}\n\nResult: {result}\n\n"

            # Format the final answer
            answer = f"Plan:\n\n{plan}\n\nExecution:\n\n{context}"
        
        except Exception as e:
            raise ValueError(f"Error: Failed to execute the question, {str(e)}") from e

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += calls
        # Return the result and usage statistics
        return answer, self.get_usage_stats()

def parse_arguments():
    parser = argparse.ArgumentParser(description="task_decomposition")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--model", type=str, default="qwen2.5-32b-instruct")
    parser.add_argument("--plan_model", type=str, help="Model for planning phase (defaults to --model if not specified)")
    parser.add_argument("--execute_model", type=str, help="Model for execution phase (defaults to --model if not specified)")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "PS"], help="method"
    )
    parser.add_argument(
        "--dataset", type=str, default="multiarith", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument("--parallel_num", type=int, default=1, help="number for api calling")
    parser.add_argument("--question", type=str, default="For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?", help="Question to solve")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import asyncio
    
    async def main():
        args = parse_arguments()

        plan_model = getattr(args, 'plan_model', args.model)
        execute_model = getattr(args, 'execute_model', args.model)
        args.question= "How many factors of $2^5\cdot3^6$ are perfect squares?"
        
        # Create an instance of DAGFlowMethod
        dag_flow_method = DAGFlow(args)

        # Execute the method with the question
        question = args.question
        result, usage = await dag_flow_method.execute(question)

    asyncio.run(main())