import os

import google.generativeai as genai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key


class GeminiRouter:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    def call(self, model, base_url, prompt):
        model = genai.GenerativeModel('gemini-1.5-pro-002')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1,
                temperature=0.0,  # Set to 0 for deterministic output
                top_k=10,  # Consider top 5 tokens
                top_p=1.0,
                response_logprobs=True,
                logprobs=3
            ),
            safety_settings=[],  # Disable safety filters for this example
            stream=False,
        )

        # Access the log probabilities
        if response.candidates and response.candidates[0].logprobs:
            top_tokens = response.candidates[0].logprobs.top_logprobs[0]
            print("Top 5 candidate tokens and their log probabilities:")
            for token, logprob in top_tokens.items():
                print(f"Token: {token}, Log Probability: {logprob}")
        else:
            print("Log probabilities not available in the response.")
    def get_preference(self, model, base_url, prompt):
        pass

