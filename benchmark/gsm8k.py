# -*- coding: utf-8 -*-
# @Date    :
# @Author  : all
# @Desc    : test on gsm8k
import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from utils import get_format_output
from benchmark.benchmark import BaseBenchmark



class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, params: str, file_path: str, log_path: str):
        super().__init__(params, file_path, log_path)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, method, input_text):
        return await method.execute(input_text)

    async def evaluate_problem(self, problem: dict, method) -> Tuple[str, str, float, float, float, float, float]:
        input_text = problem["question"]
        expected_output = self.extract_number(problem["answer"])

        try:
            if self.params.method == "select_then_decompose":
                output, final_method, cost = await self._generate_output(method, input_text)
            else:
                output, cost = await self._generate_output(method, input_text)
                final_method = self.params.method

            format_output = await get_format_output(self.params.dataset, input_text, output)
            predicted_number = self.extract_number(format_output)
            score, extracted_output = self.calculate_score(expected_output, predicted_number)
            if self.params.method == "select_then_decompose":
                total_cost = cost.get("total_tokens")
            else:
                total_cost = cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
            total_calls = cost.get("total_calls", 0)
            return input_text, final_method, output, extracted_output, expected_output, score, total_cost, total_calls

        except Exception as e:
            print(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), str(e), 0.0, expected_output, 0.0, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "final_method", "prediction", "predicted_number", "expected_output", "score", "cost", "total_calls"]
