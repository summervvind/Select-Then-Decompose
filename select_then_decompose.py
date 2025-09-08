import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Tuple, Optional
from openai_call import get_openai_response
import argparse
import asyncio
import re
import random  
from method.IO import DirectOutput
from method.zero_shot_cot import COT
from method.plan_solve import PlanSolve
from method.react import ReAct
from method.linear_flow import LinearFlow
from method.dag_flow import DAGFlow


class DecompositionMethods:
    def __init__(self, args):
        self.args = args
    async def io(self, problem: str) -> str:

        method = DirectOutput(self.args)
        output, cost = await method.execute(problem)

        return output, cost


    async def cot(self, problem: str) -> str:
        
        method = COT(self.args)
        output, cost = await method.execute(problem)

        return output, cost


    async def ps(self, problem: str) -> str:
        
        method = PlanSolve(self.args)
        output, cost = await method.execute(problem)

        return output, cost


    async def linear_flow(self, problem: str) -> str:
       
        method = LinearFlow(self.args)
        output, cost = await method.execute(problem)

        return output, cost


    async def react(self, problem: str) -> str:
        
        method = ReAct(self.args)
        output, cost = await method.execute(problem)

        return output, cost


    async def dag_flow(self, problem: str) -> str:
        
        method = DAGFlow(self.args)
        output, cost = await method.execute(problem)

        return output, cost


class ConfidenceValidator:
    def __init__(self, args,  llm_api=None):
        self.args = args
        self.threshold = self.args.confidence_threshold
        self.llm_api = llm_api  
        
    async def calculate_confidence(self, problem: str, solution: str) -> float:
        prompt = f"""Please, as a serious evaluator, rate the quality of the following "solution".

Problem:
{problem}

Solution:
{solution}

Please give your **confidence score** for the solution, give your explanation, and return a floating point number between 0 and 1.
Please strictly follow the following format:
<think>
Your analysis
</think>
<score>
confidence score
</score>"""

        response, usage = await self.llm_api(self.args, prompt)
        
        if "<score>" in response:
            match = re.search(r"<score>\s*(.*?)\s*</score>", response, re.DOTALL)
            if match:
                try:
                    confidence_score = float(match.group(1).strip())
                    
                    return confidence_score, usage
                except ValueError:
                    return 0.0, 0.0
        
        return 0.0, 0.0

    async def validate(self, problem: str, solution: str) -> bool:
        
        confidence_score, usage = await self.calculate_confidence(problem, solution)
        
        return  confidence_score >= self.threshold, usage



class FallbackStrategy:
    
    METHOD_GROUPS = [
        [("io", 1.0)],  
        [("cot", 1.0), ("ps", 1.0)], 
        [("react", 1.0), ("linear_flow", 1.0), ("dag_flow", 1.0)]  
    ]

    @classmethod
    def get_next_method(cls, current_method: str) -> Optional[str]:
        for group_idx, group in enumerate(cls.METHOD_GROUPS):
            method_names = [m[0] for m in group]
            if current_method in method_names:
                if group_idx + 1 < len(cls.METHOD_GROUPS):
                    next_group = cls.METHOD_GROUPS[group_idx + 1]
                    methods, weights = zip(*next_group)  
                    next_method = random.choices(methods, weights=weights, k=1)[0]
                    return next_method
                else:
                    return None
        
        methods, weights = zip(*cls.METHOD_GROUPS[0])
        return random.choices(methods, weights=weights, k=1)[0]


class SelectThenDecompose:
    def __init__(self, args, router_model_path=None):
        
        self.args = args
        self.llm_api = get_openai_response
        self.validator = ConfidenceValidator(self.args, self.llm_api)
        self.methods = DecompositionMethods(self.args)
        self.token_usage = 0
        self.api_calls = 0

    def get_usage_stats(self) -> dict:
        return {
            "total_tokens": self.token_usage,
            "total_calls": self.api_calls,
        }
        
    async def _select_final_method(self, problem: str, initial_method: str) -> str:
        
        prompt = f"""Please analyze the characteristics of the task description and select the most suitable method to solve the task from the candidate methods.

Task description: {problem}

Please analyze the characteristics of the task from the following dimensions:
- Whether it has clear goals and solution steps (logic)
- Whether it may require multiple rounds of attempts, corrections or dynamic adjustments (iterative)
- Whether it involves information collection, viewpoint exploration (divergent)

Candidate methods and introduction:
- io: Input-Output, directly output the answer, suitable for simple problems
- cot: Chain of Thought, step-by-step thinking and reasoning to generate answers, suitable for problems that require logical deduction
- ps: Plan & Solve, make a plan first and then execute, suitable for problems that require logical deduction
- react: Reason+Act, alternate reasoning and execution, suitable for iterative code generation tasks
- linear_flow: generate a plan and execute it in sequence, suitable for vertical tasks with strict logic
- dag_flow: build a task structure of directed acyclic graph, suitable for divergent tasks of parallel processing and extensive information collection

When choosing a method, please combine the specific characteristics of the task with the applicable scenarios of the above methods to explain your reasons for choosing

Please strictly follow the following format:
<think>
Your analysis
</think>
<answer>
Your choice (only fill in the method name, such as: cot, ps, etc.)
</answer>"""
        
        response, usage = await self.llm_api(self.args, prompt)
        

        if "<answer>" in response:
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        # return random.choice(["io","cot","ps","react","linear_flow","dag_flow"])
        return initial_method
        
    async def execute(self, problem: str, max_retry: int = 3) -> Tuple[str, str]:
        
        final_method = await self._select_final_method(problem, "io")
        
        tried_methods = []
        for _ in range(max_retry):
        
            solution, usage = await getattr(self.methods, final_method)(problem)
            
            self.token_usage += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            self.api_calls += usage.get("total_calls", 0)
            tried_methods.append(final_method)
            
            validation, usage = await self.validator.validate(problem, solution)
            
            if validation:
                return solution, final_method, self.get_usage_stats()
                
            next_method = FallbackStrategy.get_next_method(final_method)
            if not next_method or next_method in tried_methods:
                break
            final_method = next_method
            
        return solution, final_method, self.get_usage_stats()
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="task_decomposition")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--method", type=str, default="cot", choices=["io", "cot", "ps", "react", "linear_flow", "dag_flow", "select_then_decompose"], help="method"
    )
    parser.add_argument(
        "--dataset", type=str, default="GSM8K", choices=["GSM8K", "Multiarith", "MATH", "HumanEval", "HotpotQA", "Trivia_Creative_Writing", "MT_Bench"], help="dataset used for experiment"
    )
    parser.add_argument("--parallel_num", type=int, default=1, help="number for api calling")
    parser.add_argument("--is_test", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--category", type=str, default="writing", choices=["writing", "roleplay", "extraction", "math", "coding", "reasoning", "stem", "humanities"], help="the category for mt_bench experiment which is not need when experimenting on other benchmarks")
    parser.add_argument("--confidence_threshold", type=float, default=1.0, help="the validation confidence_threshold for select_then_decompose")

    args = parser.parse_args()

    return args

async def main():
    args = parse_arguments()
    processor = SelectThenDecompose(
        args=args,
    )
    problem = "Write a short and coherent story about Cinderella that incorporates the answers to the following 5 questions: What mythological beast has the head of a man, the body of a lion, and the tail and feet of a dragon? In Greek mythology, who were Arges, Brontes and Steropes? Which musician founded the Red Hot Peppers? Where did the Shinning Path terrorists operate? Which Brit broke the land speed record in 1990 in Thrust 2?"
    problem = "Write a short and coherent story about Pikachu that incorporates the answers to the following 5 questions: Who directed the classic 30s western Stagecoach? Dave Gilmore and Roger Waters were in which rock group? Which highway was Revisited in a classic 60s album by Bob Dylan? Which was the only eastern bloc country to participate in the 1984 LA Olympics? Which 90s sci fi series with James Belushi was based on Bruce Wagner's comic strip of the same name?"
    result = await processor.execute(problem)
    solution, method, cost = result  
    print("\n**Final Solution**:")
    print(solution)
    print("\n**Final Approach**:")
    print(method)
    print("\ncost:", cost)


if __name__ == "__main__":
    asyncio.run(main())
    
