import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    set_seed
)
from transformers.generation.utils import _relative_top_filter
torch.set_grad_enabled(False)

class CustomRegressionLogitsProcessor(LogitsProcessor):
    """
    A custom logits processor to process logits using regression-based predictions.
    """

    def __init__(self, mode, model, M: int=31, L: int=32, step: float=1.0, device: torch.device=None,
            lm_head=None, final_norm=None, mean_X_reg=None, var_X_reg=None, X_reg_=None,
        ):
        self.mode = mode
        self.model = model
        self.M = M
        self.L = L
        self.step = step
        self.device = device
        self.lm_head = lm_head
        self.final_norm = final_norm

        self.mean_X_reg = mean_X_reg
        self.var_X_reg = var_X_reg
        self.X_reg_ = X_reg_

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids (torch.LongTensor): Indices of input sequence tokens in the vocabulary.
            scores (torch.FloatTensor): Prediction scores of a language modeling head.

        Returns:
            torch.FloatTensor: The processed prediction scores.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_hidden_states=True,
            )
            # Extract hidden states
            if self.mode == "filter":
                logits = outputs.logits[0, -1, :]
                scores[:, :], _ = _relative_top_filter(logits, logits)

            elif self.mode == "extrapolation":
                hidden_states = torch.stack(outputs.hidden_states).squeeze(1)  # (N_layer, N_token, d_model).
                hidden_states = hidden_states[self.M:, :, :]  # (N_layer - M, N_token, d_model)
                # Normalization and transformation
                norms = self.final_norm(hidden_states[:-1])  # (N_layer, N_token, d_model)
                norms = torch.cat([norms, hidden_states[-1].unsqueeze(0)], dim=0)
                Y = self.lm_head(norms)
                del outputs, hidden_states
                
                # Regression logic
                Y_reg = Y[:, -1, :].float() # (N_layer - M, vocab_size)
                mean_Y_reg = torch.mean(Y_reg, dim=0)  # (vocab_size)
                cov_XY_reg = torch.mean(self.X_reg_ * (Y_reg - mean_Y_reg), dim=0)  # (1, vocab_size)

                beta_1 = cov_XY_reg / self.var_X_reg  # (vocab_size)
                beta_0 = mean_Y_reg - beta_1 * self.mean_X_reg  # (vocab_size)

                Y_pred = beta_1[None, :] * self.L + beta_0[None, :]  # (vocab_size)
                # Final logits for the target layer
                scores[:, :], _ = _relative_top_filter(Y_pred, Y_reg[None, -1])
            if self.device:
                torch.cuda.empty_cache()
        return scores

class LLM:
    def __init__(self, model_id, device_map="auto"):
        self.model_id = model_id
        self.device_map = device_map

        self.model, self.tokenizer, self.N_layer = self.load_model(model_id)
        self.device = self.model.device
        
        if not hasattr(self.model, "base_model"):
            raise ValueError("Model does not have a `base_model` attribute.")

    def load_model(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        )
        model.eval()
        model.requires_grad_(False)
        N_layer = model.config.num_hidden_layers
        return model, tokenizer, N_layer

    def gen(
            self, 
            input_text, 
            max_new_tokens=None, 
            generate_kwargs: dict = None,
        ):
        with torch.no_grad():
            set_seed(42)
            assert max_new_tokens is not None, "Please specify the maximum number of tokens to generate."
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens
            
            outputs = self.model.generate(
                input_ids, 
                max_length=max_len,
                **generate_kwargs
            )
        sequences = outputs.sequences
        gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
        output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

        if self.device:
            torch.cuda.empty_cache()

        return output_str, len(gen_sequences) + input_ids.shape[-1]