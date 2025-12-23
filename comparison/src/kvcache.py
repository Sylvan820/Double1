import torch
from .util import norm_logits, sample
# from transformers import _crop_past_key_values  <-- 已移除
from transformers import DynamicCache
from typing import Optional, TYPE_CHECKING
import ipdb

# 类型提示用，避免循环导入
if TYPE_CHECKING:
    from .retrieval_cache import RetrievalCache


def _crop_past_key_values(model, past_key_values, new_cache_size):
    """
    Manually crops a KV cache tuple to a new size.
    This replaces the internal transformers function that was removed.
    """
    if past_key_values is None:
        return None
    
    past_key_values_trimmed = []
    for kv_layer in past_key_values:
        k, v = kv_layer
        current_len = k.shape[2]
        
        if new_cache_size >= current_len:
            past_key_values_trimmed.append(kv_layer)
        else:
            k_trimmed = k[:, :, :new_cache_size, :]
            v_trimmed = v[:, :, :new_cache_size, :]
            past_key_values_trimmed.append((k_trimmed, v_trimmed))
            
    return tuple(past_key_values_trimmed) 



@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10, 
                                retrieval_cache: Optional["RetrievalCache"] = None,
                                for_draft: bool = True):
    """
    PLD function to find candidate tokens by matching n-grams.
    
    增强版：支持Double Datastore检索库
    
    Double Datastore 策略:
    - for_draft=True  → 使用 Draft Datastore (D_q)：小模型用，包含大小模型所有输出，高覆盖率
    - for_draft=False → 使用 Target Datastore (D_p)：大模型用，只有验证后的tokens，高精度
    
    搜索优先级（在每个 Datastore 内部）：
    1. prefix_cache - 持久化前缀库（warmup + 历史任务积累）
    2. dynamic_cache - 动态历史库（当前任务已生成的tokens）
    3. input_ids - 当前输入序列（原始PLD逻辑）
    
    Args:
        input_ids: 输入token序列
        max_ngram_size: 最大n-gram大小
        num_pred_tokens: 预测tokens数量
        retrieval_cache: 可选的检索库实例
        for_draft: True=小模型用Draft Datastore, False=大模型用Target Datastore
        
    Returns:
        候选tokens序列
    """
    # 如果有检索库，根据 for_draft 选择对应的 datastore
    if retrieval_cache is not None:
        candidate = retrieval_cache.find_candidate_tokens(input_ids, for_draft=for_draft)
        if len(candidate) > 0:
            return candidate
    
    # 回退到原始逻辑：在输入序列中搜索
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
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0,
                 retrieval_cache: Optional["RetrievalCache"] = None) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        # Get vocab size from model config
        self.vocab_size = model.config.vocab_size

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        
        # 检索库支持
        self._retrieval_cache = retrieval_cache
        
        # 跟踪被拒绝的tokens用于添加到检索库
        self._last_rejected_tokens = None
        self._last_candidate_tokens = None
        
    def set_retrieval_cache(self, retrieval_cache: Optional["RetrievalCache"]):
        """设置检索库"""
        self._retrieval_cache = retrieval_cache

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
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
            self._past_key_values= DynamicCache.from_legacy_cache(self._past_key_values)
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
            # ipdb.set_trace()

        return last_q

    def _generate_with_kvcache(self, prefix: torch.Tensor,
                               gamma: int, eos_token_id: int = None) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses
            eos_token_id (int): EOS token id to stop generation

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)

            # Check for EOS token to stop generation early
            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

        # ipdb.set_trace()
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int, eos_token_id: int = None) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, eos_token_id)
        return output

    def _generate_with_pld(self, prefix: torch.Tensor, gamma: int, max_ngram_size: int = 3,
                           num_pred_tokens: int = 10, eos_token_id: int = None) -> torch.Tensor:
        """
        Generate tokens using PLD following the reference implementation logic.
        gamma = number of PLD steps
        Each step: try PLD retrieval → model verifies → accept/reject following reference logic
        Returns: prefix + all generated tokens from gamma steps
        
        增强：使用检索库并记录被拒绝的tokens
        """
        input_ids = prefix
        self._actual_tokens_generated = 0
        self._last_rejected_tokens = []  # 记录本次生成中被拒绝的tokens

        # Execute gamma PLD steps following reference pld.py logic
        for step in range(gamma):
            cur_len = input_ids.shape[-1]

            # Step 1: Find PLD candidate tokens (使用检索库 - 小模型用 Draft Datastore)
            candidate_pred_tokens = find_candidate_pred_tokens(
                input_ids, max_ngram_size, num_pred_tokens, 
                retrieval_cache=self._retrieval_cache,
                for_draft=True  # 小模型使用 Draft Datastore (高覆盖率)
            )
            
            # 保存候选tokens用于后续添加到检索库
            self._last_candidate_tokens = candidate_pred_tokens.clone() if len(candidate_pred_tokens) > 0 else None

            if len(candidate_pred_tokens) == 0:
                candidate_pred_tokens = torch.tensor([100], device=input_ids.device).unsqueeze(0)
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
                self._past_key_values= DynamicCache.from_legacy_cache(self._past_key_values)

                current_len = self._past_key_values.get_seq_length()
                new_tokens = candidate_input_ids[:, current_len:]
                outputs = self._model(new_tokens, past_key_values=self._past_key_values, use_cache=True)
                new_logits = outputs.logits[:, :, :self.vocab_size]
                for i in range(new_logits.shape[-2]):
                    new_logits[:, i, :] = norm_logits(new_logits[:, i, :], self._temperature, self._top_k,
                                                      self._top_p)
                self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
                self._past_key_values = outputs.past_key_values

            # Verification logic following reference pld.py
            new_logits = outputs.logits[:, -candidate_length - 1:]
            selected_tokens = new_logits.argmax(dim=-1)
            candidate_new_tokens = candidate_input_ids[:, -candidate_length:]

            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

            # Accept valid tokens following reference logic
            valid_tokens = selected_tokens[:, : n_matches + 1]
            
            # 记录被拒绝的tokens到检索库
            if self._retrieval_cache is not None and n_matches < candidate_length:
                # 被拒绝的tokens = 候选tokens中未被接受的部分
                rejected = candidate_new_tokens[0, n_matches:]
                if len(rejected) > 0:
                    self._last_rejected_tokens.append(rejected)
                    # 将被拒绝的tokens（以及接受的tokens）添加到检索库
                    # 这样即使被拒绝，这些模式也可能在未来被匹配到
                    self._retrieval_cache.add_rejected_tokens(candidate_new_tokens[0], source="draft")
            
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            new_cur_len = input_ids.shape[-1]

            # Crop cache to new length following reference logic
            new_cache_size = new_cur_len - 1
            if self._past_key_values is not None:
                
                self._past_key_values = _crop_past_key_values(self._model, self._past_key_values, new_cache_size)

            # Crop prob_history to match the accepted tokens length
            if self._prob_history is not None:
                self._prob_history = self._prob_history[:, :new_cur_len - 1, :]

                # Set PLD tokens to one-hot probability for accepted tokens (except the last one)
                accept_length = new_cur_len - cur_len
                if accept_length > 1:
                    for i in range(accept_length - 1):
                        pos = cur_len - 1 + i
                        if pos >= 0 and pos < self._prob_history.shape[1]:
                            accepted_token = valid_tokens[0, i].item()
                            self._prob_history[0, pos, :] = 0.0
                            self._prob_history[0, pos, accepted_token] = 1.0

            # Track actual tokens generated
            self._actual_tokens_generated += accept_length
            
            # 将接受的tokens添加到 Draft Datastore（小模型生成的所有tokens，包括被验证的）
            # 注意：小模型生成的tokens只加到 Draft Datastore，不加到 Target Datastore
            if self._retrieval_cache is not None and accept_length > 0:
                self._retrieval_cache.add_draft_tokens(valid_tokens[0])

            # Check for EOS token to stop generation early
            if eos_token_id is not None:
                if (valid_tokens == eos_token_id).any():
                    break
        return input_ids

    def _generate_with_pldt(self, prefix: torch.Tensor, gamma: int, max_ngram_size: int = 3,
                            num_pred_tokens: int = 10, eos_token_id: int = None) -> torch.Tensor:
        """
        Generate tokens using PLD following the reference implementation logic.
        gamma = number of PLD steps
        Each step: try PLD retrieval → model verifies → accept/reject following reference logic
        Returns: prefix + all generated tokens from gamma steps
        
        增强：使用检索库并记录被拒绝的tokens（用于target model）
        """
        input_ids = prefix
        self._actual_tokens_generated = 0
        self._last_rejected_tokens = []  # 记录本次生成中被拒绝的tokens

        cur_len = input_ids.shape[-1]

        # Step 1: Find PLD candidate tokens (使用检索库 - 大模型用 Target Datastore)
        candidate_pred_tokens = find_candidate_pred_tokens(
            input_ids, max_ngram_size, num_pred_tokens,
            retrieval_cache=self._retrieval_cache,
            for_draft=False  # 大模型使用 Target Datastore (高精度)
        )
        
        # 保存候选tokens
        self._last_candidate_tokens = candidate_pred_tokens.clone() if len(candidate_pred_tokens) > 0 else None

        if len(candidate_pred_tokens) == 0:
            candidate_pred_tokens = torch.tensor([100], device=input_ids.device).unsqueeze(0)
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
            self._past_key_values= DynamicCache.from_legacy_cache(self._past_key_values)
            current_len = self._past_key_values.get_seq_length()
            new_tokens = candidate_input_ids[:, current_len:]

            # Handle edge case where KV cache is ahead
            if new_tokens.shape[1] == 0:
                self._past_key_values = _crop_past_key_values(self._model, self._past_key_values, input_ids.shape[1])

                current_len = self._past_key_values.get_seq_length()
                new_tokens = candidate_input_ids[:, current_len:]

            if new_tokens.dim() == 1:
                new_tokens = torch.unsqueeze(new_tokens, 0)

            outputs = self._model(new_tokens, past_key_values=self._past_key_values, use_cache=True)
            new_logits = outputs.logits[:, :, :self.vocab_size]

            if new_logits.dim() == 2:
                new_logits = torch.unsqueeze(new_logits, 0)

            for i in range(new_logits.shape[-2]):
                new_logits[:, i, :] = norm_logits(new_logits[:, i, :], self._temperature, self._top_k,
                                                  self._top_p)
            self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
            self._past_key_values = outputs.past_key_values

        # Verification logic following reference pld.py
        verification_logits = self._prob_history[:, -candidate_length-1:, :]
        selected_tokens = verification_logits.argmax(dim=-1)
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]

        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

        # Accept valid tokens following reference logic
        valid_tokens = selected_tokens[:, : n_matches + 1]
        
        # 记录被拒绝的tokens到检索库（用于target model）
        if self._retrieval_cache is not None and n_matches < candidate_length:
            rejected = candidate_new_tokens[0, n_matches:]
            if len(rejected) > 0:
                self._last_rejected_tokens.append(rejected)
                self._retrieval_cache.add_rejected_tokens(candidate_new_tokens[0], source="target")
        
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        new_cur_len = input_ids.shape[-1]

        # Crop cache to new length following reference logic
        new_cache_size = new_cur_len - 1
        if self._past_key_values is not None:
            # *** 使用我们新增的本地函数 ***
            self._past_key_values = _crop_past_key_values(self._model, self._past_key_values, new_cache_size)

        # Crop prob_history to match the accepted tokens length
        if self._prob_history is not None:
            self._prob_history = self._prob_history[:, :new_cur_len - 1, :]

            # Set PLD tokens to one-hot probability for accepted tokens (except the last one)
            accept_length = new_cur_len - cur_len
            if accept_length > 1:
                for i in range(accept_length - 1):
                    pos = cur_len - 1 + i
                    if pos >= 0 and pos < self._prob_history.shape[1]:
                        accepted_token = valid_tokens[0, i].item()
                        self._prob_history[0, pos, :] = 0.0
                        self._prob_history[0, pos, accepted_token] = 1.0

        # Track actual tokens generated
        self._actual_tokens_generated += accept_length
        
        # 将大模型验证后的tokens添加到 Target Datastore（只有verified tokens才加到Target）
        # 同时也添加到 Draft Datastore（保持高覆盖率）
        if self._retrieval_cache is not None and accept_length > 0:
            self._retrieval_cache.add_verified_tokens(valid_tokens[0])

        return input_ids

    def get_actual_tokens_generated(self):
        """Return the actual number of tokens generated in the last PLD generation"""
        return getattr(self, '_actual_tokens_generated', 0)

    @torch.no_grad()
    def generate_with_pld(self, input: torch.Tensor, gamma: int, max_ngram_size: int = 3,
                          num_pred_tokens: int = 10, eos_token_id: int = None) -> torch.Tensor:
        """Generate tokens using PLD + draft model"""
        output = self._generate_with_pld(input, gamma, max_ngram_size, num_pred_tokens, eos_token_id)
        return output

    def generate_with_pldt(self, input: torch.Tensor, gamma: int, max_ngram_size: int = 3,
                           num_pred_tokens: int = 10, eos_token_id: int = None) -> torch.Tensor:
        """Generate tokens using PLD + draft model"""
        output = self._generate_with_pldt(input, gamma, max_ngram_size, num_pred_tokens, eos_token_id)
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

        # *** 修改：赋值为 tuple ***
        self._past_key_values = tuple(past_key_values_trimmed)
        self._prob_history = self._prob_history[:, :end_pos, :]