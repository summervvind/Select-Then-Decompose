import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio



class BaseBenchmark(ABC):
    def __init__(self, params, file_path: str, log_path: str):
        self.params = params
        self.file_path = file_path
        self.log_path = log_path

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        t_calls = df["total_calls"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        a_calls = t_calls / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}_{self.params.model}_{self.params.is_test}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost, a_calls, t_calls
    
    def save_results_to_csv_mt(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        average_score_turn_1 = df["score_1"].mean()
        average_score_turn_2 = df["score_2"].mean()
        t_cost = df["cost"].max()
        t_calls = df["total_calls"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        a_calls = t_calls / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{average_score_turn_1:.5f}_{average_score_turn_2:.5f}_{current_time}_{self.params.model}_{self.params.category}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return average_score_turn_1, average_score_turn_2, a_cost, t_cost, a_calls, t_calls

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
            "time": current_time,
        }

        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)

        with log_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        


    @abstractmethod
    async def evaluate_problem(self, problem: dict, method) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], method, max_concurrent_tasks: int = 50):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, method)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.params.dataset} problems", total=len(data))

    async def run_evaluation(self, method, va_list: List[int], max_concurrent_tasks: int = 10):
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, method, max_concurrent_tasks)
        columns = self.get_result_columns()
        if self.params.dataset == "MT_Bench":
            results_data = self.save_results_to_csv_mt(results, columns)
        else:
            results_data = self.save_results_to_csv(results, columns)
        return results_data
