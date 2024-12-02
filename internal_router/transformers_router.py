import re

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F

device = "cuda"

class TransformerRouter:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda"

    def call(self, model, base_url, prompt):
        pass

    def initiate_model(self, model_name):
        tokenizer = None
        model = None
        if model_name == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        elif model_name == "EleutherAI/pythia-160m":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
        elif model_name == "google/gemma-2-2b-it":
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
            tokenizer.pad_token = tokenizer.eos_token
        # Device selection logic
        try:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")

            self.tokenizer = tokenizer
            self.model = model.to(self.device)
        except RuntimeError:
            print("Error moving model to MPS. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.model = model.to(self.device)

    def get_preference(self, model_name, base_url, prompt):
        self.initiate_model(model_name)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=1000,
            num_return_sequences=1,
            temperature=0.7,
            top_k=100,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    def get_log_probs(self, validator, model_name, base_url, prompt, top_k):
        # Initialize the model and tokenizer
        self.initiate_model(model_name)

        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        final_word_predictions = set()
        i = 0
        while True:
            try:
                # Initialize a list to store the predicted tokens
                predicted_tokens = []
                current_input_ids = input_ids  # Start with the prompt

                # Predict tokens until a non-'Ġ' token is found (i.e., word boundary)
                count = 0
                while True:
                    # Get logits for the last token
                    with torch.no_grad():
                        outputs = self.model(input_ids=current_input_ids, output_attentions=False,
                                             output_hidden_states=False)

                    logits = outputs.logits[:, -1, :]  # Logits for the last token in the sequence
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get the top-k predictions
                    top_k_values, top_k_indices = torch.topk(log_probs, top_k + 50)
                    top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices.squeeze().tolist())

                    # Add the most probable valid token to the predictions
                    if count == 0:
                        predicted_token = top_k_tokens[i]
                    else:
                        predicted_token = top_k_tokens[0]
                    if (predicted_token.startswith("Ġ") and count > 0) or count >= 10:
                        break
                    count += 1
                    predicted_tokens.append(predicted_token)

                    # Append this token to input_ids to predict the next token
                    new_input_id = self.tokenizer.convert_tokens_to_ids([predicted_token])
                    current_input_ids = torch.cat([current_input_ids, torch.tensor([new_input_id]).to(self.device)], dim=-1)

                predicted_word = self.tokenizer.convert_tokens_to_string(predicted_tokens).strip()
                predicted_word = re.sub(r"[^\w'-]", '', predicted_word)

                if validator.validate_output(predicted_word):
                    final_word_predictions.add(predicted_word)
                i += 1
                if len(final_word_predictions) == top_k or i >= 50:
                    break
            except Exception as e:
                print("Error during inference: ", e)
                break
        print("predictions from huggingface: ", final_word_predictions)
        return final_word_predictions, 



