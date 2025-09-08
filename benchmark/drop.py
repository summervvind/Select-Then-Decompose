import re
import string
from collections import Counter
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmark.benchmark import BaseBenchmark
from utils import get_format_output


class DROPBenchmark(BaseBenchmark):
    def __init__(self, params: str, file_path: str, log_path: str):
        super().__init__(params, file_path, log_path)

    def remove_answer_tags(self, text: str) -> str:
        """
        Removes <Answer> and </Answer> tags from the given text.
        """
        return re.sub(r"<Answer>(.*?)</Answer>", r"\1", text, flags=re.DOTALL).strip()

    def normalize_answer(self, s: str) -> str:
        """
        Normalize answers for evaluation.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        s = self.remove_answer_tags(s)

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, method, input_text):
        return await method.execute(input_text)

    async def evaluate_problem(self, problem: dict, method) -> Tuple[str, str, str, str, str, float, float, float]:
        input_text = problem["context"]
        expected_output = problem["ref_text"]
        answers = expected_output.split("|")

        try:
            if self.params.method == "select_then_decompose":
                output, final_method, cost = await self._generate_output(method, input_text)
            else:
                output, cost = await self._generate_output(method, input_text)
                final_method = self.params.method
            format_output = await get_format_output(self.params.dataset, input_text, output)
            f1_scores = []

            for answer in answers:
                if answer.strip() != "":
                    if isinstance(format_output, str):
                        output_parts = format_output.split("|")
                    else:
                        output_parts = [str(format_output)]
                    for output_part in output_parts:
                        f1_score, _ = self.calculate_score(answer, output_part)
                        f1_scores.append(f1_score)

            uni_score = max(f1_scores) if f1_scores else 0.0
            if self.params.method == "select_then_decompose":
                total_cost = cost.get("total_tokens")
            else:
                total_cost = cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
            total_calls = cost.get("total_calls", 0)

            if uni_score < 0.3:
                self.log_mismatch(input_text, expected_output, output, str(format_output))

            return input_text, final_method, output, str(format_output), expected_output, uni_score, total_cost, total_calls

        except Exception as e:
            print(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), str(e), str(e), expected_output, 0.0, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "final_method", "prediction", "format_output", "expected_output", "score", "cost", "total_calls"]