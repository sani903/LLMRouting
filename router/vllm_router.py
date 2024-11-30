import requests
from transformers import AutoTokenizer
import numpy as np


class VLLMRouter:
    def __init__(self):
        pass

    def call(self, model, base_url, prompt, system_prompt=None):
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.9
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.9
            }

        try:
            response = requests.post(base_url, json=payload)

            response.raise_for_status()
            data = response.json()

            response_text = data["choices"][0]["text"]
            return response_text.strip()

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected response format: Missing key {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_preference(self, model, base_url, prompt, system_prompt=None):
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.9
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.9
            }

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1000,
            "echo": False
        }
        response = requests.post(f"{base_url}", json=payload)
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['text']
            return generated_text
        return None

    