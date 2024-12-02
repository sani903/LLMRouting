import os
import openai
import re


# Replace 'YOUR_API_KEY' with your actual OpenAI API key


class OpenAIRouter:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        pass

    def call(self, model, base_url, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "<<>"
                )
            },
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=1,
            n=1,
            logprobs=True,
            top_logprobs=3,
            stop=None
        )

        unique_next_words = set()

        for choice in response['choices']:
            assistant_message = choice['message']['content'].strip()

            # Extract the first word from the assistant's message
            next_word_match = re.match(r'^\s*(\S+)', assistant_message)
            if next_word_match:
                next_word = next_word_match.group(1)
            else:
                next_word = assistant_message

            unique_next_words.add(next_word)

        # Remove duplicates

        print("Next predicted words:", unique_next_words)
        return unique_next_words, ""

    def get_preference(self, validator, model, base_url, prompt, top_k):
        # Prepare the messages for ChatCompletion
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a word prediction assistant. Given a text prompt, "
                    "predict the next word that is most likely to come next. "
                    "Only provide the next word and nothing else."
                )
            },
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=2,
            temperature=1,
            n=1,
            logprobs=True,
            top_logprobs=3,
            stop=None
        )
        pass
