from abc import ABC, abstractmethod
from openai_call import get_openai_response
import argparse

class BaseMethod(ABC):
    def __init__(self, args):
        self.args = args
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    @abstractmethod
    def generate_prompt(self, question: str) -> str:
        pass

    @abstractmethod
    def execute(self, question: str) -> tuple:
        pass

    def get_usage_stats(self) -> dict:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_calls": self.total_calls,
        }
