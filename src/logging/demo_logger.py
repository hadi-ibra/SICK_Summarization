from typing import Any, Dict
from overrides import overrides
from src.logging.logger import BaseDecorator, Logger


class DemoLogger(BaseDecorator):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.summaries = None
        self.results = None

    @overrides
    def save_results(self, data: Dict[str, Any]) -> None:
        if "summaries" in data.keys():
            self.summaries = data["summaries"]
        else:
            self.results = data
        self.logger.save_results(data)

    def get_saved_results(self):
        return self.summaries, self.results
