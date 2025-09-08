from openai_call import get_openai_response
import argparse
from method.basemethod import BaseMethod  # Assuming basemethod.py contains the BaseMethod class

class LinearFlow(BaseMethod):
    def extract_subtasks_to_dict(self, text: str) -> dict:
        """
        Extracts subtasks from the generated plan and returns them as a dictionary.
        """
        subtasks = {}
        lines = text.strip().split("\n")  # Split text into lines
        for line in lines:
            if line.startswith("Subtask"):
                # Extract Subtask ID and description
                parts = line.split(":", 1)  # Split only at the first colon
                if len(parts) == 2:
                    key = parts[0].strip()  # Subtask ID
                    value = parts[1].strip()  # Subtask description
                    subtasks[key] = value
        return subtasks

    def generate_prompt(self, question: str) -> str:
        """
        Generates the initial prompt for the linear flow method.
        """
        template = '''Let's first understand the following problem and devise a linear plan to solve the problem.
        
        {input}

        Use the following format:

        Subtask 1: [First step to solve the problem]
        Subtask 2: [Second step to solve the problem]
        ...(repeat as needed)
        
        Provide only the subtasks as a plan. Do not execute or generate results for any subtask.

        Begin!
        '''
        return template.format(input=question)

    async def execute(self, question: str) -> tuple:
        """
        Executes the linear flow method by generating a plan and executing subtasks sequentially.
        """
        # Generate the plan
        prompt_tokens = 0
        completion_tokens = 0
        calls = 0
        try:
            plan_prompt = self.generate_prompt(question)
            plan, usage = await get_openai_response(self.args, plan_prompt)
            
            # Extract subtasks from the plan
            subtask_dict = self.extract_subtasks_to_dict(plan)

            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            calls += 1

            # Initialize context
            context = ""

            # Execute subtasks sequentially
            results = {}
            for subtask_id, subtask_description in subtask_dict.items():
                # Build subtask prompt
                subtask_prompt = f"{question}\n\n{context}{subtask_id}: {subtask_description}\n\nPlease execute this {subtask_id} and provide the result: "
                
                # Execute subtask
                result, usage = await get_openai_response(self.args, subtask_prompt)
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                calls += 1

                # Save result
                results[subtask_id] = result

                # Update context
                context += f"{subtask_id}: {subtask_description}\n\nResult: {result}\n\n"

            # Format the final answer
            answer = f"Plan: \n{plan}\n\nExecution: \n{context}"
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
    parser.add_argument("--model", type=str, default="deepseek-chat")
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    linear_flow_method = LinearFlow(args)

    # Execute the method with a question
    question = "For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?"
    result, usage = linear_flow_method.execute(question)

    # Print the result and usage statistics
    print("Result:", result)
    print("Usage:", usage)