import pandas as pd

from typing import List, Dict


from ragas import evaluate
from ragas.metrics import Metric


class Evaluator:
    def __init__(self, metrics: List[Metric], llm):
        self.metrics = metrics
        self.llm = llm
        for metric in self.metrics:
            metric.llm = llm  # Ensure LLM is set for each metric

    def evaluate(self, predictions: List[Dict]) -> pd.DataFrame:
        eval_results = evaluate(predictions, metrics=self.metrics)
        return eval_results.to_pandas()
