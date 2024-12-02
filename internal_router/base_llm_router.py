from abc import ABC, abstractmethod

import requests


class BaseLLMRouter(ABC):
    def __init__(self):
        pass

    def call(self, model_name, base_url, prompt):
        pass

    def inference_call(self, model, base_url, prompt):
        pass