# -*- coding: utf-8 -*-
# @Date    :
# @Author  : all
# @Desc    : test on gsm8k
import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from utils import get_format_output
from utils import get_llm_score
from benchmark.benchmark import BaseBenchmark
import traceback

NEED_REF_CATS = ["math", "reasoning", "coding"]

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

class MTBenchmark(BaseBenchmark):
    def __init__(self, params: str, file_path: str, log_path: str):
        super().__init__(params, file_path, log_path)

    def remove_answer_tags(self, text: str) -> str:
        """
        Removes <Answer> and </Answer> tags from the given text.
        """
        return re.sub(r"<Answer>(.*?)</Answer>", r"\1", text, flags=re.DOTALL).strip()

    async def calculate_score(self, category, questions, outputs, expected_outputs, i) -> Tuple[float, float]:
        if not outputs:
            return 0.0
        score = await get_llm_score(category, questions, outputs, expected_outputs, i)
        return score

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, method, input_text):
        return await method.execute(input_text)

    async def evaluate_problem(self, problem: dict, method) -> Tuple[str, str, str, str, float, str, str, str, str, float, float, float]:
        questions = problem["turns"]
        category = problem["category"]
        
        if category in NEED_REF_CATS and "reference" in problem:
            expected_outputs = problem["reference"]
        else: 
            expected_outputs = [None, None]
        try:
            messages = []
            outputs = []
            scores = []
            judgments = []
            for i in range(len(questions)):
                qs = questions[i]
                input_text = "\n***Your last turn of interaction history***:\n\n" + "\n".join(messages) + "\n\n***Your question this turn***:\n\n" + qs if messages else qs
                output, cost = await self._generate_output(method, input_text)
                
                if category in ["math", "reasoning"]:
                    format_output = output
                elif category == "coding":
                    format_output = await get_format_output("MT_Bench_Coding", input_text, output)
                    format_output = self.remove_answer_tags(format_output)
                else:
                    format_output = await get_format_output("MT_Bench_Writing", input_text, output)
                    format_output = self.remove_answer_tags(format_output)
                outputs.append(format_output)
                ## use llm to score
                score, judgment = await self.calculate_score(category, questions, outputs, expected_outputs, i)
                scores.append(score)
                judgments.append(judgment)
                messages.extend([f"**User**: \n{qs}", f"**Assistent**: \n{format_output}"])
                
                total_cost = cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
                total_calls = cost.get("total_calls", 0)

            return questions[0], outputs[0], expected_outputs[0], judgments[0], scores[0], questions[1], outputs[1], expected_outputs[1], judgments[1], scores[1], total_cost, total_calls

        except Exception as e:
            print(f"Maximum retries reached. Skipping this sample. Error: {e}")
            # traceback.print_exc()
            return questions[0], str(e), expected_outputs[0], str(e), 0.0, questions[1], str(e), expected_outputs[1], str(e), 0.0, 0.0, 0.0
    def get_result_columns(self) -> List[str]:
        return ["question_1", "output_1", "expected_output_1", "judgment_1", "score_1", "question_2", "output_2", "expected_output_2", "judgment_2", "score_2", "cost", "total_calls"]
