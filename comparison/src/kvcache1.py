import torch
from .util import norm_logits, sample
from transformers.generation.utils import _crop_past_key_values


@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    """
    PLD function to find candidate tokens by matching n-grams in the prompt.
    Returns candidate tokens for speculative verification.
    """
    input_length = input_ids.size(1)

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                candidate_tokens = input_ids[0, start_idx:end_idx]
                return candidate_tokens

    # If no match is found, return empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)


class KVCacheModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        # PLD-specific state
        self._pld_tokens = None  # Store PLD retrieved tokens
        self._pld_probs = None  # Store one-hot probabilities for PLD tokens
        self._pld_length = 0  # Number of PLD tokens in current sequence

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor, max_ngram_size: int = 3,
                              num_pred_tokens: int = 10) -> torch.Tensor:
        """Standard forward pass without PLD enhancement - used for single token generation"""
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k,
                                                          self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = self._past_key_values.get_seq_length()

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)

            not_cached_q = outputs.logits[:, :, :self.vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values

        return last_q

    def _generate_with_kvcache(self, prefix: torch.Tensor,
                               gamma: int) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output

    def _generate_with_pld(self, prefix: torch.Tensor, gamma: int, max_ngram_size: int = 3,
                           num_pred_tokens: int = 10) -> torch.Tensor:
        """
        Generate tokens using PLD following the reference implementation logic.
        gamma = number of PLD steps
        Each step: try PLD retrieval → model verifies → accept/reject following reference logic
        Returns: prefix + all generated tokens from gamma steps
        """
        input_ids = prefix

        # Reset PLD tracking
        self._pld_tokens = None
        self._pld_probs = None
        self._pld_length = 0

        # Track actual tokens generated for verification
        self._actual_tokens_generated = 0

        # Execute gamma PLD steps following reference pld.py logic
        for step in range(gamma):
            cur_len = input_ids.shape[-1]

            # Step 1: Find PLD candidate tokens
            candidate_pred_tokens = find_candidate_pred_tokens(input_ids, max_ngram_size, num_pred_tokens)

            if len(candidate_pred_tokens) == 0:
                # No PLD candidates found, generate single token with draft model
                q = self._forward_with_kvcache(input_ids)
                next_tok = sample(q)
                input_ids = torch.cat((input_ids, next_tok), dim=1)
                self._actual_tokens_generated += 1
            else:
                # PLD candidates found, follow reference verification logic
                candidate_pred_tokens = candidate_pred_tokens.unsqueeze(0)
                candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)
                candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

                # Forward pass through model
                if self._past_key_values is None:
                    outputs = self._model(candidate_input_ids)
                    self._prob_history = outputs.logits[:, :, :self.vocab_size]
                    for i in range(self._prob_history.shape[-2]):
                        self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature,
                                                                  self._top_k, self._top_p)
                    self._past_key_values = outputs.past_key_values
                else:
                    cached_len = self._past_key_values.get_seq_length()
                    new_tokens = candidate_input_ids[:, cached_len:]
                    outputs = self._model(new_tokens, past_key_values=self._past_key_values, use_cache=True)
                    new_logits = outputs.logits[:, :, :self.vocab_size]
                    for i in range(new_logits.shape[-2]):
                        new_logits[:, i, :] = norm_logits(new_logits[:, i, :], self._temperature, self._top_k,
                                                          self._top_p)
                    self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
                    self._past_key_values = outputs.past_key_values

                # Verification logic following reference pld.py
                new_logits = outputs.logits[:, -candidate_length - 1:]  # excludes the input prompt if present
                selected_tokens = new_logits.argmax(dim=-1)
                candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
                n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

                # Accept valid tokens following reference logic
                valid_tokens = selected_tokens[:, : n_matches + 1]
                input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
                new_cur_len = input_ids.shape[-1]

                # Crop cache to new length following reference logic
                new_cache_size = new_cur_len - 1
                if self._past_key_values is not None:
                    self._past_key_values = _crop_past_key_values(self._model, self._past_key_values, new_cache_size)

                # Track actual tokens generated
                accept_length = new_cur_len - cur_len
                self._actual_tokens_generated += accept_length

        return input_ids

    def get_pld_info(self):
        """Return PLD tokens info - simplified since PLD is now internal to draft model"""
        return self._pld_tokens, self._pld_probs, self._pld_length

    def get_actual_tokens_generated(self):
        """Return the actual number of tokens generated in the last PLD generation"""
        return getattr(self, '_actual_tokens_generated', 0)

    @torch.no_grad()
    def generate_with_pld(self, input: torch.Tensor, gamma: int, max_ngram_size: int = 3,
                          num_pred_tokens: int = 10) -> torch.Tensor:
        """Generate tokens using PLD + draft model"""
        output = self._generate_with_pld(input, gamma, max_ngram_size, num_pred_tokens)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)

        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]