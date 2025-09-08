import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmark.benchmark import BaseBenchmark
from utils import get_format_output


class MATHBenchmark(BaseBenchmark):
    def __init__(self, params: str, file_path: str, log_path: str):
        super().__init__(params, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = expected_output
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if re.search(r'\d', num) and re.search(r'\\text\{.*?\}', num):
                # If answer has unit, remove it
                num = re.sub(r'\\text\{.*?\}', '', num).strip()
                try:
                    return float(num)
                except:
                    pass
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, method, input_text):
        return await method.execute(input_text)

    async def evaluate_problem(self, problem: dict, method) -> Tuple[str, str, float, float, float, float, float]:
        input_text = problem["problem"]
        expected_output = self.extract_model_answer(problem["solution"])

        try:
            if self.params.method == "select_then_decompose":
                output, final_method, cost = await self._generate_output(method, input_text)
            else:
                output, cost = await self._generate_output(method, input_text)
                final_method = self.params.method

            format_output = await get_format_output(self.params.dataset, input_text, output)
            uni_score, extracted_output = self.calculate_score(expected_output, format_output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )
            if self.params.method == "select_then_decompose":
                total_cost = cost.get("total_tokens")
            else:
                total_cost = cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
            total_calls = cost.get("total_calls", 0)
            return input_text, final_method, output, extracted_output, expected_output, uni_score, total_cost, total_calls

        except Exception as e:
            # print(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), str(e), 0.0, expected_output, 0.0, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "final_method", "prediction", "predicted_number", "expected_output", "score", "cost", "total_calls"]

