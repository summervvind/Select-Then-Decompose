from openai_call import get_openai_response
import argparse
import asyncio
from method.basemethod import BaseMethod  # Assuming basemethod.py contains the BaseMethod class

class ReAct(BaseMethod):
    def generate_prompt(self, question: str) -> str:
        """
        Generates the initial prompt for the ReAct method.
        """
        template = '''Answer the following questions as best you can. 

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Subtask: your subtask to carry out
        Result: the result of the subtask
        ... (this Thought/Subtask/Result can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}'''
        return template.format(input=question)

    async def execute(self, question: str) -> tuple:
        """
        Executes the ReAct method by iteratively generating thoughts, subtasks, and results.
        """
        prompt_tokens = 0
        completion_tokens = 0
        calls = 0
        try:
            prompt = self.generate_prompt(question)

            # Termination flag
            final_answer_found = False
            all_steps = ""

            for i in range(1, 8):  # Maximum of 7 attempts

                # Generate Thought and Subtask
                thought_subtask, usage = await get_openai_response(self.args, prompt + f"\nThought:", stop=[f"\nResult:"])
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                calls += 1

                # Check if the final answer is found
                if "Final Answer:" in thought_subtask:
                    thought, final_answer = thought_subtask.strip().split(f"\nFinal Answer:")
                    all_steps += f"Thought: {thought}\nFinal Answer: {final_answer}" + "\n"
                    final_answer_found = True
                    break  # Terminate the loop

                try:
                    # Attempt to split Thought and Subtask
                    thought, subtask = thought_subtask.strip().split(f"\nSubtask:")
                except:
                    # Handle parsing errors
                    thought = thought_subtask.strip().split('\n')[0]  # Use the first line as Thought
                    subtask, usage = await get_openai_response(self.args, prompt + f"Thought: {thought}\nSubtask:", stop=[f"\n"])
                    prompt_tokens += usage.get("prompt_tokens", 0)
                    completion_tokens += usage.get("completion_tokens", 0)
                    calls += 1

                # Generate Result
                result, usage = await get_openai_response(self.args, prompt + f"Thought: {thought}\nSubtask: {subtask}\nResult:", stop=[f"\nThought:"])
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                calls += 1

                # Update the prompt with the current step
                step_str = f"Thought: {thought}\nSubtask: {subtask}\n\nResult: {result}"
                all_steps += step_str + "\n"
                prompt += step_str + "\n"

            # Handle cases where the final answer is not found
            if not final_answer_found:
                all_steps += "\nReached maximum attempts. Final answer not found.\n"
        except Exception as e:
            raise ValueError(f"Error: Failed to execute the question, {str(e)}") from e
                
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += calls
        # Return the result and usage statistics
        return all_steps, self.get_usage_stats()

    

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

async def main():
    args = parse_arguments()

    method = ReAct(args)

    question = "For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?"
    
    result, usage = await method.execute(question)
    print("Result:", result)
    print("Usage:", usage)

if __name__ == "__main__":
    asyncio.run(main())