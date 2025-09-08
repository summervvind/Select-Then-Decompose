# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.
import asyncio
import argparse
from typing import Dict, List
import os
from evaluator import Evaluator

if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


def parse_arguments():
    parser = argparse.ArgumentParser(description="task_decomposition")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--plan_model", type=str, help="Model for planning phase (defaults to --model if not specified)")
    parser.add_argument("--execute_model", type=str, help="Model for execution phase (defaults to --model if not specified)")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--method", type=str, default="cot", choices=["io", "cot", "ps", "react", "linear_flow", "dag_flow", "select_then_decompose"], help="method"
    )
    parser.add_argument(
        "--dataset", type=str, default="GSM8K", choices=["GSM8K", "Multiarith", "MATH", "HumanEval", "HotpotQA", "Trivia_Creative_Writing", "MT_Bench", "DROP"], help="dataset used for experiment"
    )
    parser.add_argument("--parallel_num", type=int, default=1, help="number for api calling")
    parser.add_argument("--is_test", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--category", type=str, default="writing", choices=["writing", "roleplay", "extraction", "math", "coding", "reasoning", "stem", "humanities"], help="the category for mt_bench experiment which is not need when experimenting on other benchmarks")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="the validation confidence_threshold for select_then_decompose")

    args = parser.parse_args()

    return args

async def main():
    args = parse_arguments()

    evaluator = Evaluator()
    log_path = f"logs/{args.dataset}/{args.method}"
    os.makedirs(log_path, exist_ok=True)

    try:
        result = await evaluator.evaluate(
            dataset=args.dataset,
            method=args.method,      
            params=args,    
            path=log_path,      
            is_test=args.is_test     
        )
        print(f"Results of {args.method} method on {args.dataset} dataset: ")
        if args.dataset == "MT_Bench":
            print(f"Average turn 1 score: {result[0]:.5f}")
            print(f"Average turn 2 score: {result[1]:.5f}")
        else:
            print(f"Average score: {result[0]:.5f}")
        print(f"Average Cost: {result[-4]:.5f}")
        print(f"Total Cost: {result[-3]:.5f}")
        print(f"Average Calls: {result[-2]:.5f}")
        print(f"Total Calls: {result[-1]: .5f}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":

    asyncio.run(main())

  