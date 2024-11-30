import ollama
import signal


class OllamaRouter:
    def __init__(self):
        pass

    def timeout_handler(self, signum, frame):
        raise TimeoutError("LLM inference call timed out.")

    def call(self, model, base_url, prompt, timeout_seconds=60):
        signal.signal(signal.SIGALRM, self.timeout_handler)

        # Start the countdown for the timeout

        try:
            local_response = ollama.generate(model=model, prompt=prompt)
            signal.alarm(0)

            return local_response['response']
        except TimeoutError as e:
            print(f"Inference call timed out after {timeout_seconds} seconds.")
            raise e  # Handle the timeout (return None or handle it as needed)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e

    def get_preference(self, model_name, base_url, prompt):
        pass
