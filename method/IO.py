import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai_call import get_openai_response
import argparse
import asyncio
from method.basemethod import BaseMethod  # Assuming basemethod.py contains the BaseMethod class

class DirectOutput(BaseMethod):
    def generate_prompt(self, question: str) -> str:
        """
        Generates a prompt for the OpenAI API using a predefined template.
        """
        template = """
        Q: {question}
        A: Please output the final answer directly.
        """
        return template.format(question=question)

    async def execute(self, question: str) -> tuple:
        """
        Executes the logic for generating a prompt, calling the OpenAI API, and returning the result.
        """
        try:
            prompt = self.generate_prompt(question)
            result, usage = await get_openai_response(self.args, prompt)
        except Exception as e:
            raise ValueError(f"Error: Failed to execute the question, {str(e)}") from e
        # Update token usage and call count
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_calls += 1

        # Return the result and usage statistics
        return result, self.get_usage_stats()
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="task_decomposition")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
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

async def main():
    args = parse_arguments()

    method = DirectOutput(args)

    question = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market."
    
    result, usage = await method.execute(question)
    print("Result:", result)
    print("Usage:", usage)

if __name__ == "__main__":
    asyncio.run(main())