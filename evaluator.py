# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Literal, Tuple, Callable

from benchmark.benchmark import BaseBenchmark

from benchmark.multiarith import MultiarithBenchmark
from benchmark.gsm8k import GSM8KBenchmark
from benchmark.hotpotqa import HotpotQABenchmark
from benchmark.humaneval import HumanEvalBenchmark
from benchmark.math import MATHBenchmark
from benchmark.trivia_creative_writing import TriviaCreativeWritingBenchmark
from benchmark.mt_bench import MTBenchmark
from benchmark.drop import DROPBenchmark

from method.basemethod import BaseMethod

from method.IO import DirectOutput
from method.zero_shot_cot import COT
from method.plan_solve import PlanSolve
from method.react import ReAct
from method.linear_flow import LinearFlow
from method.dag_flow import DAGFlow

from select_then_decompose import SelectThenDecompose




# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["Multiarith", "HumanEval", "GSM8K", "MATH", "HotpotQA", "Trivia_Creative_Writing", "MT_Bench", "DROP"]
MethodType = Literal["io", "cot", "ps", "react", "linear_flow", "dag_flow"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self):
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "Multiarith": MultiarithBenchmark,
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "Trivia_Creative_Writing": TriviaCreativeWritingBenchmark,
            "MT_Bench": MTBenchmark,
            "DROP": DROPBenchmark,
        }
        self.method_configs: Dict[MethodType, BaseMethod] = {
            "io": DirectOutput,  # Input-Output method
            "cot": COT,  # Chain-of-Thought method
            "ps": PlanSolve,  # Plan-and-Solve method
            "react": ReAct,  # ReAct method
            "linear_flow": LinearFlow,  # Linear Flow method
            "dag_flow": DAGFlow,  # DAG Flow method
        }

    async def evaluate(
        self, dataset: DatasetType, method: str, params: dict, path: str, is_test: bool = False
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(params=params, file_path=data_path, log_path=path)
        if method == "select_then_decompose":
            method = SelectThenDecompose(params)
        else:
            method_class = self.method_configs[method]
            method = method_class(params)
        # The MT_Bench set is divided by category, and the others are divided by is_test
        if dataset == "MT_Bench":
            if params.category == "writing": va_list = list(range(10))   
            elif params.category == "roleplay": va_list = list(range(10, 20))
            elif params.category == "reasoning": va_list = list(range(20, 30))
            elif params.category == "math": va_list = list(range(30, 40))
            elif params.category == "coding":  va_list = list(range(40, 50))
            elif params.category == "extraction": va_list = list(range(50, 60))
            elif params.category == "stem": va_list = list(range(60, 70))
            elif params.category == "humanities": va_list = list(range(70, 80))   
        else:
            if is_test:
                va_list = None # list(range(200))  # For test data, generally use None to test all
            else:
                va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(method, va_list, max_concurrent_tasks = params.parallel_num)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"dataset/{dataset.lower()}/{dataset.lower()}"
        # return f"{base_path}.jsonl"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"