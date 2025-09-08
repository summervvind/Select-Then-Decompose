import re
from typing import Callable, List, Optional, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from utils import get_format_output
from benchmark.benchmark import BaseBenchmark


class TriviaCreativeWritingBenchmark(BaseBenchmark):
    def __init__(self, params: str, file_path: str, log_path: str):
        super().__init__(params, file_path, log_path)

    def remove_answer_tags(self, text: str) -> str:
        """
        Removes <Answer> and </Answer> tags from the given text.
        """
        return re.sub(r"<Answer>(.*?)</Answer>", r"\1", text, flags=re.DOTALL).strip()

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        correct_count = 0
        question_count = len(ground_truth)
        prediction = self.remove_answer_tags(prediction)
        for ans_to_question in ground_truth:
            for ans in ans_to_question:
                if ans.lower() in prediction.lower():
                    correct_count += 1
                    break
        score = 1.0 * correct_count / question_count
        return score, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, method, input_text):
        return await method.execute(input_text)

    async def evaluate_problem(self, problem: dict, method) -> Tuple[str, str, str, str, str, float, float, float]:
        questions = problem["questions"]
        topic = problem["topic"]
        n = len(questions)
        questions_str = " ".join(questions)
        expected_output = problem["answers"]
        prompt_template = '''Write a short and coherent story about {topic} that incorporates the answers to the following {n} questions: {questions}'''
        input_prompt = prompt_template.format(n=n, questions=questions_str, topic=topic)
        try:
            if self.params.method == "select_then_decompose":
                output, final_method, cost = await self._generate_output(method, input_prompt)
            else:
                output, cost = await self._generate_output(method, input_prompt)
                final_method = self.params.method
            format_output = await get_format_output(self.params.dataset, input_prompt, output)
            score, extracted_output = self.calculate_score(expected_output, format_output)
            if self.params.method == "select_then_decompose":
                total_cost = cost.get("total_tokens")
            else:
                total_cost = cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
            total_calls = cost.get("total_calls", 0)
            if (
                score <= 0.2 ## only 1/5 or 1/10 is right
            ):  # We set the threshold for collecting incorrect questions to 0.3, as F1 Score cannot be simply judged using 0-1
                self.log_mismatch(input_prompt, expected_output, output, extracted_output)

            return questions, input_prompt, final_method, output, format_output, expected_output, score, total_cost, total_calls

        except Exception as e:
            print(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return questions, input_prompt, str(e), str(e), str(e), expected_output, 0.0, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "input_prompt", "final_method", "prediction", "format_output", "expected_output", "score", "cost", "total_calls"]
