import torch
import transformers
import warnings

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel
from .kvcache4RC import KVCacheModel as KVCache2Model
from .util import seed_everything, norm_logits, sample, max_fn
from .retrieval_cache import RetrievalCache, RetrievalCacheManager
import ipdb
import time
from typing import Optional


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()

        # ! only parallel speculative decoding can use 2 processes
        assert (self.accelerator.num_processes == 1 and args.eval_mode in ["small", "large", "sd"]) or (
                self.accelerator.num_processes == 2 and args.eval_mode in ["para_sd", "para_sd_wo_1",
                                                                           "para_sd_wo_1", "rc_para_sd"])

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        
        # 检索库支持
        self._retrieval_cache: Optional[RetrievalCache] = None
        self._retrieval_cache_manager: Optional[RetrievalCacheManager] = None
        self._warmup_task_count = getattr(args, 'warmup_task_count', 0)  # 热启动任务数
        self._current_task_idx = 0
        self._use_retrieval_cache = getattr(args, 'use_retrieval_cache', False)
        
    def init_retrieval_cache(self, cache_dir: str = "./retrieval_cache", 
                             max_ngram_size: int = 3, 
                             num_pred_tokens: int = 10,
                             load_from: str = None):
        """
        初始化检索库
        
        Args:
            cache_dir: 缓存目录
            max_ngram_size: 最大n-gram大小
            num_pred_tokens: 预测tokens数量
            load_from: 可选，从指定文件加载预热的缓存
        """
        self._retrieval_cache_manager = RetrievalCacheManager(
            cache_dir=cache_dir,
            max_ngram_size=max_ngram_size,
            num_pred_tokens=num_pred_tokens
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._retrieval_cache = self._retrieval_cache_manager.create_cache(device)
        
        if load_from:
            self._retrieval_cache_manager.load_cache(load_from, device)
            self.color_print(f"Loaded retrieval cache from {load_from}", 2)
            
        self._use_retrieval_cache = True
        self.color_print(f"Retrieval cache initialized (max_ngram={max_ngram_size}, num_pred={num_pred_tokens})", 2)
        
    def get_retrieval_cache(self) -> Optional[RetrievalCache]:
        """获取当前检索库"""
        return self._retrieval_cache
    
    def on_task_start(self, task_idx: int, prompt_tokens: torch.Tensor = None):
        """
        任务开始时的回调 - 设置检索库状态
        
        Args:
            task_idx: 任务索引
            prompt_tokens: 可选，任务的prompt tokens，添加到动态库
        """
        self._current_task_idx = task_idx
        
        if self._retrieval_cache is not None:
            # 如果是热启动阶段，不清除动态缓存
            if task_idx >= self._warmup_task_count:
                self._retrieval_cache_manager.on_task_start()
            
            # 将prompt添加到动态库
            if prompt_tokens is not None:
                self._retrieval_cache.add_to_dynamic_cache(prompt_tokens)
                
    def on_task_end(self, output_tokens: torch.Tensor = None):
        """
        任务结束时的回调 - 更新检索库
        
        Args:
            output_tokens: 可选，任务生成的完整输出tokens
        """
        if self._retrieval_cache is not None:
            # 热启动阶段：将所有tokens合并到持久化前缀库
            if self._current_task_idx < self._warmup_task_count:
                if output_tokens is not None:
                    self._retrieval_cache.add_to_prefix_cache(output_tokens)
                # 合并动态和被拒绝的tokens到前缀库
                self._retrieval_cache_manager.on_task_end(keep_dynamic=True, keep_rejected=True)
                self.color_print(f"[Warmup {self._current_task_idx + 1}/{self._warmup_task_count}] Added to prefix cache", 3)
            else:
                # 正常阶段：只合并动态缓存（可选）
                if output_tokens is not None:
                    self._retrieval_cache.add_to_prefix_cache(output_tokens)
                self._retrieval_cache_manager.on_task_end(keep_dynamic=True, keep_rejected=True)
                
    def save_retrieval_cache(self, name: str = "default"):
        """保存检索库"""
        if self._retrieval_cache_manager is not None:
            self._retrieval_cache_manager.save_cache(name)
            self.color_print(f"Retrieval cache saved as '{name}'", 2)
            
    def print_retrieval_stats(self):
        """打印检索库统计信息"""
        if self._retrieval_cache is not None:
            self._retrieval_cache.print_stats()

    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        if self.args.eval_mode == "small":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="auto",
                                                                    torch_dtype=torch.bfloat16,
                                                                    trust_remote_code=True).eval()
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto",
                                                                     torch_dtype=torch.bfloat16,
                                                                     trust_remote_code=True).eval()
        elif self.args.eval_mode == "sd":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0",
                                                                    torch_dtype=torch.bfloat16,
                                                                    trust_remote_code=True).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model,
                                                                     device_map="balanced_low_0",
                                                                     torch_dtype=torch.bfloat16,
                                                                     trust_remote_code=True).eval()

        elif self.args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1"]:
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0",
                                                                        torch_dtype=torch.bfloat16,
                                                                        trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model,
                                                                         device_map="balanced_low_0",
                                                                         torch_dtype=torch.bfloat16,
                                                                         trust_remote_code=True).eval()

        elif self.args.eval_mode == "rc_para_sd":
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0",
                                                                        torch_dtype=torch.bfloat16,
                                                                        trust_remote_code=True).eval()
                self.draft_model_2 = AutoModelForCausalLM.from_pretrained(self.args.draft_model,
                                                                          device_map=f"cuda:{torch.cuda.device_count() - 1}",
                                                                          torch_dtype=torch.bfloat16,
                                                                          trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto",
                                                                         torch_dtype=torch.bfloat16,
                                                                         trust_remote_code=True).eval()

        self.vocab_size = self.args.vocab_size

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"

        # for llama models
        self.tokenizer.pad_token_id = 2
        # print(self.tokenizer.eos_token_id)
        # stop_token_list = [tokenizer.eos_token_id]
        # # hardcode eos_token_id for Llama 3.3 to fix generation issues
        # if ("Llama-3.3" in self.args.draft_model or "llama-3.3" in self.args.draft_model.lower() or
        #     "Llama-3.3" in self.args.target_model or "llama-3.3" in self.args.target_model.lower()):
        #     self.tokenizer.eos_token_id = "<|eot_id|>"
        #     self.color_print(f"Hardcoded EOS token ID to 128009 for Llama 3.3", 2)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def autoregressive_sampling(self, prefix):
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError("Auto-Regressive Decoding can be used only in small / large eval mode!")

        prefix = prefix.to(model.device)

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens

        x = prefix
        past_key_values = None
        while x.shape[1] < max_tokens:
            if past_key_values:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids.unsqueeze(0)
                outputs = model(last_ids, past_key_values=past_key_values, use_cache=True)
            else:
                outputs = model(x)

            if self.accelerator.is_main_process:
                if self.args.eval_mode == "small":
                    self.draft_forward_times += 1
                elif self.args.eval_mode == "large":
                    self.target_forward_times += 1

            last_p = norm_logits(outputs.logits[::, -1, :], self.args.temp, self.args.top_k, self.args.top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)

        return x

    @torch.no_grad()
    # def speculative_decoding(self, prefix):
    #     max_tokens = prefix.shape[1] + self.args.max_tokens
    #
    #     draft_device = self.draft_model.device
    #     target_device = self.target_model.device
    #
    #     approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
    #     approx_model_cache.vocab_size = self.vocab_size
    #     target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
    #     target_model_cache.vocab_size = self.vocab_size
    #
    #     while prefix.shape[1] < max_tokens:
    #         prefix_len = prefix.shape[1]
    #         x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
    #         _ = target_model_cache.generate(x.to(target_device), 1)
    #         if self.accelerator.is_main_process:
    #             self.draft_forward_times += self.args.gamma
    #             self.target_forward_times += 1
    #
    #         n = prefix_len + self.args.gamma - 1
    #         for i in range(self.args.gamma):
    #             r = torch.rand(1, device=draft_device)
    #             j = x[:, prefix_len + i]
    #
    #             if r > (target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
    #                 n = prefix_len + i - 1
    #                 break
    #
    #         self.num_acc_tokens.append(n - prefix_len + 1)
    #
    #         assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
    #         prefix = x[:, :n + 1]
    #
    #         approx_model_cache.rollback(n+1)
    #
    #         if n < prefix_len + self.args.gamma - 1:
    #             # reject someone, sample from the pos n
    #             t = sample(max_fn(target_model_cache._prob_history[:, n, :self.vocab_size].to(draft_device) - approx_model_cache._prob_history[:, n, :self.vocab_size]))
    #             target_model_cache.rollback(n+1)
    #         else:
    #             # all approx model decoding accepted
    #             t = sample(target_model_cache._prob_history[:, -1, :self.vocab_size]).to(draft_device)
    #             target_model_cache.rollback(n+2)
    #         prefix = torch.cat((prefix, t), dim=1)
    #     return prefix

    def speculative_decoding(self, prefix):
        max_tokens = prefix.shape[1] + self.args.max_tokens
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p,
                                          retrieval_cache=self._retrieval_cache)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p,
                                          retrieval_cache=self._retrieval_cache)
        target_model_cache.vocab_size = self.vocab_size
        # 改进文件读取逻辑
        try:
            with open('token.txt', 'r') as f:
                content = f.read().strip()
                acceptance_stats = list(map(int, content.split(','))) if content else [0] * 17
        except (FileNotFoundError, ValueError):
            acceptance_stats = [0] * 17
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_model_cache.generate(x.to(target_device), 1)
            if self.accelerator.is_main_process:
                self.draft_forward_times += self.args.gamma
                self.target_forward_times += 1

            n = prefix_len + self.args.gamma - 1
            accepted_count = 0
            for i in range(self.args.gamma):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]
                if r > (target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]) / (
                        approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break
                accepted_count += 1
            # 更新统计数组并保存到文件
            if self.accelerator.is_main_process and accepted_count < 17:
                acceptance_stats[accepted_count] += 1
                with open('token.txt', 'w') as f:
                    f.write(','.join(map(str, acceptance_stats)))

            self.num_acc_tokens.append(n - prefix_len + 1)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            approx_model_cache.rollback(n + 1)

            if n < prefix_len + self.args.gamma - 1:
                t = sample(max_fn(target_model_cache._prob_history[:, n, :self.vocab_size].to(
                    draft_device) - approx_model_cache._prob_history[:, n, :self.vocab_size]))
                target_model_cache.rollback(n + 1)
            else:
                t = sample(target_model_cache._prob_history[:, -1, :self.vocab_size]).to(draft_device)
                target_model_cache.rollback(n + 2)
            prefix = torch.cat((prefix, t), dim=1)

            stop_flag = 0
            # print(self.tokenizer.eos_token_id)
            for token in prefix[0, prefix_len:].tolist():
                if token == self.tokenizer.eos_token_id:
                    stop_flag = 1
                    break
        
            if stop_flag == 1:
                break    
        return prefix

    @torch.no_grad()
    def parallel_speculative_decodingP(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma * 2] = x[:, prefix_len - self.args.gamma + 1:prefix_len + self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob = prob.to("cuda:1")
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            draft_ids = all_prob[0, [0], 1:self.args.gamma * 2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)

                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0

                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, :prefix_len - self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma + n + 1)
                    
            stop_flag = 0
            # print(self.tokenizer.eos_token_id)
            for token in prefix[0, prefix_len:].tolist():
                if token == self.tokenizer.eos_token_id:
                    stop_flag = 1
                    break
        
            if stop_flag == 1:
                break    

        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding1(self, prefix):
        """
        PLD + PEARL combined parallel speculative decoding.
        Maintains PEARL structure but uses PLD-enhanced generation.
        """
        # parallel speculative decoding with PLD
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # PLD parameters
        max_ngram_size = getattr(self.args, 'pld_max_ngram_size', 3)
        num_pred_tokens = getattr(self.args, 'pld_num_pred_tokens', 10)

        # this flag is used to determine the current verify mode.
        cur_mode = True  # True: Pre-verify, False: Post-verify
        num_acc_token = 0

        # Track tokens for two-round mechanism
        previous_tokens = self.args.gamma  # Initialize with gamma for first round
        current_tokens = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)
            # ipdb.set_trace()

            if self.accelerator.is_main_process:
                # Draft model side: Use PLD-enhanced generation (gamma = PLD steps, not token count)
                x = model.generate_with_pld(input_ids, self.args.gamma, max_ngram_size, num_pred_tokens,
                                            eos_token_id=self.tokenizer.eos_token_id)

                # Get current round's actual number of tokens generated
                current_tokens = model.get_actual_tokens_generated()

                # Store probabilities like standard PEARL using previous_tokens for window size
                prob = model._prob_history[:, prefix_len - previous_tokens - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 3:previous_tokens + current_tokens + 2] = x[:,
                                                                     prefix_len - previous_tokens + 1:prefix_len + current_tokens]

                # Store current_tokens count for next round verification
                prob[:, 0, 1] = current_tokens
                prob[:, 0, 2] = previous_tokens

                # Count PLD steps as forward passes
                self.draft_forward_times += self.args.gamma

            else:
                # Target model side: use PLD-enhanced forward (same as draft model)
                x = model.generate(input_ids, 1, eos_token_id=self.tokenizer.eos_token_id)

                # Get current round's actual number of tokens generated
                current_tokens = 1

                # Store probabilities like standard PEARL using previous_tokens for window size
                prob = model._prob_history[:, prefix_len - previous_tokens - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 1] = current_tokens
                prob[:, 0, 2] = previous_tokens
                prob = prob.to("cuda:1")
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()

            # Verification phase - gather probabilities from both processes
            all_prob = self.accelerator.gather(prob).to(device)
            previous_tokens = all_prob[0, [0], 2].int().item()
            current_tokens = all_prob[0, [0], 1].int().item()

            # ipdb.set_trace()

            # Extract tokens and probabilities like standard PEARL using previous_tokens
            draft_ids = all_prob[0, [0], 3:previous_tokens + current_tokens + 2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]

            if cur_mode:
                # Pfirst_token = draft_ids[:, -self.args.gamma]
                first_token = draft_ids[:, -current_tokens]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)

                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0

                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                    num_acc_token += 1

            else:
                # Post-verify mode: enhanced to verify all tokens including PLD tokens

                n = previous_tokens
                for i in range(previous_tokens):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - previous_tokens + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == previous_tokens:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                    num_acc_token += previous_tokens
                else:
                    # reject someone, change the mode
                    assert n < previous_tokens
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, :prefix_len - previous_tokens + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - previous_tokens + n + 1)

            # Update for next round: current becomes previous
            previous_tokens = current_tokens
            stop_flag = 0
            # print(self.tokenizer.eos_token_id)
            for token in prefix[0, prefix_len:].tolist():
                if token == self.tokenizer.eos_token_id:
                    stop_flag = 1
                    break
        
            if stop_flag == 1:
                break    
        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        """
        PLD + PEARL combined parallel speculative decoding.
        Maintains PEARL structure but uses PLD-enhanced generation.
        
        增强：支持检索库
        """
        # parallel speculative decoding with PLD
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # PLD parameters
        max_ngram_size = getattr(self.args, 'pld_max_ngram_size', 3)
        num_pred_tokens = getattr(self.args, 'pld_num_pred_tokens', 10)

        # this flag is used to determine the current verify mode.
        cur_mode = True  # True: Pre-verify, False: Post-verify
        num_acc_token = 0

        # Track tokens for two-round mechanism
        previous_tokens = self.args.gamma  # Initialize with gamma for first round
        current_tokens = 0
        step = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)
            # start_time = time.perf_counter()  # Start timing

            if self.accelerator.is_main_process:
                # Draft model side: Use PLD-enhanced generation
                # start_time = time.perf_counter()  # Start timing
                x = model.generate_with_pld(input_ids, self.args.gamma, max_ngram_size, num_pred_tokens,
                                            eos_token_id=self.tokenizer.eos_token_id)
                # elapsed = time.perf_counter() - start_time
                # print(f"draft time in {elapsed:.6f} seconds (fast path)")
                current_tokens = model.get_actual_tokens_generated()

                prob = model._prob_history[:, prefix_len - previous_tokens - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 3:previous_tokens + current_tokens + 2] = x[:,
                                                                     prefix_len - previous_tokens + 1:prefix_len + current_tokens]
                prob[:, 0, 1] = current_tokens
                prob[:, 0, 2] = previous_tokens
                self.draft_forward_times += current_tokens

            else:
                # Target model side: use PLD-enhanced forward
                # start_time = time.perf_counter()  # Start timing
                x = model.generate_with_pldt(input_ids, 1, max_ngram_size, num_pred_tokens = 10,
                                             eos_token_id=self.tokenizer.eos_token_id)
                current_tokens = model.get_actual_tokens_generated()

                # Target model side: use autogressive forward
                # x = model.generate(input_ids, 1, eos_token_id=self.tokenizer.eos_token_id)
                # current_tokens = 1

                # elapsed = time.perf_counter() - start_time
                # print(f"target time in {elapsed:.6f} seconds (fast path)")

                prob = model._prob_history[:, prefix_len - previous_tokens - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 1] = current_tokens
                prob[:, 0, 2] = previous_tokens
                prob[:, 0, 3:previous_tokens + current_tokens + 2] = x[:,
                                                                     prefix_len - previous_tokens + 1:prefix_len + current_tokens]
                prob = prob.to("cuda:1")
                self.target_forward_times += current_tokens

            self.accelerator.wait_for_everyone()

            # Verification phase - gather probabilities from both processes
            all_prob = self.accelerator.gather(prob).to(device)

            previous_tokens = all_prob[0, [0], 2].int().item()
            current_tokens = all_prob[0, [0], 1].int().item()
            target_tokens = all_prob[[1], [0], 1].int().item()

            # Extract tokens and probabilities
            draft_ids = all_prob[0, [0], 3:previous_tokens + current_tokens + 2].int()
            target_ids = all_prob[[1], [0], 2+previous_tokens:previous_tokens + target_tokens + 2].int()

            # if self.accelerator.is_main_process:
            #     ipdb.set_trace()

            # Second gather: ensure both processes send same-length tensors
            slice_start = prefix_len - previous_tokens - 1
            slice_end = prefix_len + target_tokens - 1

            # Check if we have enough prob_history and pad if needed
            available_len = model._prob_history.shape[1]
            if slice_end > available_len:
                actual_slice = model._prob_history[:, slice_start:available_len, :self.vocab_size]
                padding_len = slice_end - available_len
                padding = torch.zeros(1, padding_len, self.vocab_size, device=actual_slice.device,
                                      dtype=actual_slice.dtype)
                prob1 = torch.cat([actual_slice, padding], dim=1).to(torch.float32)
            else:
                prob1 = model._prob_history[:, slice_start:slice_end, :self.vocab_size].to(torch.float32)

            all_prob1 = self.accelerator.gather(prob1).to(device)
            draft_prob = all_prob1[[0], 1:, :]
            target_prob = all_prob1[[1], 1:, :]

            if cur_mode:
                # Pre-verify mode
                n = target_tokens

                if n >= current_tokens:
                    prefix = torch.cat((input_ids, target_ids[:, :]), dim=1)
                    num_acc_token += target_tokens
                    if self.accelerator.is_main_process:
                        model.rollback(prefix_len)
                    previous_tokens = target_tokens
                else:
                    for i in range(target_tokens):
                        token = draft_ids[:, -current_tokens + i]
                        torch.manual_seed(self.seed + prefix_len - target_tokens + i + 1)
                        r = torch.rand(1, device=device)

                        if r > target_prob[:, -target_tokens + i, token] / draft_prob[:, -target_tokens + i, token]:
                            n = i
                            break

                    if n == target_tokens:
                        # Accept all guess tokens
                        cur_mode = False
                        prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                        num_acc_token += target_tokens
                        previous_tokens = current_tokens
                        # temp != 0
                        # if not self.accelerator.is_main_process:
                        #     model.rollback(prefix_len)
                    else:
                        # Reject someone
                        assert n < target_tokens
                        prefix = torch.cat((input_ids, target_ids[:, :]), dim=1)
                        self.num_acc_tokens.append(num_acc_token + n)
                        num_acc_token = 0

                        if self.accelerator.is_main_process:
                            # temp != 0
                            # model.rollback(prefix_len)
                            model.rollback(prefix_len + n)
                        previous_tokens = target_tokens

            else:
                # Post-verify
                n = previous_tokens - 1

                for i in range(previous_tokens - 1):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - previous_tokens + i)
                    r = torch.rand(1, device=device)

                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break

                if n == previous_tokens - 1:
                    # Pre-verify in Post-verify
                    if target_tokens >= current_tokens:
                        prefix = torch.cat((input_ids, target_ids[:, :]), dim=1)
                        cur_mode = True
                        num_acc_token += previous_tokens + target_tokens - 1
                        if self.accelerator.is_main_process:
                            model.rollback(prefix_len)
                        previous_tokens = target_tokens
                    else:
                        n = previous_tokens + target_tokens - 1
                        for i in range(previous_tokens - 1, previous_tokens + target_tokens - 1):
                            token = draft_ids[:, i]
                            torch.manual_seed(self.seed + prefix_len - previous_tokens + i)
                            r = torch.rand(1, device=device)

                            if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                                n = i
                                break

                        if n == previous_tokens + target_tokens - 1:
                            prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                            num_acc_token += previous_tokens + target_tokens - 1
                            # if not self.accelerator.is_main_process:
                            #     model.rollback(prefix_len)
                            previous_tokens = current_tokens
                        else:
                            assert n < previous_tokens + target_tokens - 1
                            cur_mode = True
                            prefix = torch.cat((input_ids, target_ids[:, :]), dim=1)
                            if self.accelerator.is_main_process:
                                # model.rollback(prefix_len)
                                model.rollback(prefix_len - previous_tokens + n + 1)
                            previous_tokens = target_tokens

                else:
                    # reject someone, change the mode
                    assert n < previous_tokens - 1
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
                    prefix = torch.cat((input_ids[:, :prefix_len - previous_tokens + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - previous_tokens + n + 1)
                    previous_tokens = current_tokens

     
            stop_flag = 0
            # print(self.tokenizer.eos_token_id)
            for token in prefix[0, prefix_len:].tolist():
                if token == self.tokenizer.eos_token_id:
                    stop_flag = 1
                    break
        
            if stop_flag == 1:
                break                             
                                
        return prefix

    def parallel_speculative_decoding3(self, prefix):
        """
        PLD + PEARL combined parallel speculative decoding.
        Maintains PEARL structure but uses PLD-enhanced generation.
        """
        # parallel speculative decoding with PLD
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p,
                                 retrieval_cache=self._retrieval_cache)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # PLD parameters
        max_ngram_size = getattr(self.args, 'pld_max_ngram_size', 3)
        num_pred_tokens = getattr(self.args, 'pld_num_pred_tokens', 10)

        # this flag is used to determine the current verify mode.
        cur_mode = True  # True: Pre-verify, False: Post-verify
        num_acc_token = 0

        # Track tokens for two-round mechanism
        previous_tokens = self.args.gamma  # Initialize with gamma for first round
        current_tokens = 0
        step = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)

            if self.accelerator.is_main_process:
                # Draft model side: Use PLD-enhanced generation
                x = model.generate_with_pld(input_ids, self.args.gamma, max_ngram_size, num_pred_tokens,
                                            eos_token_id=self.tokenizer.eos_token_id)
                current_tokens = model.get_actual_tokens_generated()

                prob = model._prob_history[:, prefix_len - previous_tokens - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 3:previous_tokens + current_tokens + 2] = x[:,
                                                                     prefix_len - previous_tokens + 1:prefix_len + current_tokens]
                prob[:, 0, 1] = current_tokens
                prob[:, 0, 2] = previous_tokens
                self.draft_forward_times += current_tokens

            else:
                # Target model side: use PLD-enhanced forward
                x = model.generate_with_pldt(input_ids, 1, max_ngram_size, num_pred_tokens = 10,
                                             eos_token_id=self.tokenizer.eos_token_id)

                # ipdb.set_trace()

                current_tokens = model.get_actual_tokens_generated()

                # Target model side: use autogressive forward
                # x = model.generate(input_ids, 1, eos_token_id=self.tokenizer.eos_token_id)
                # current_tokens = 1

                prob = model._prob_history[:, prefix_len - previous_tokens - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 1] = current_tokens
                prob[:, 0, 2] = previous_tokens
                prob[:, 0, 3:previous_tokens + current_tokens + 2] = x[:,
                                                                     prefix_len - previous_tokens + 1:prefix_len + current_tokens]
                prob = prob.to("cuda:1")
                self.target_forward_times += current_tokens

            self.accelerator.wait_for_everyone()

            # Verification phase - gather probabilities from both processes
            all_prob = self.accelerator.gather(prob).to(device)

            previous_tokens = all_prob[0, [0], 2].int().item()
            current_tokens = all_prob[0, [0], 1].int().item()
            target_tokens = all_prob[[1], [0], 1].int().item()

            # Extract tokens and probabilities
            draft_ids = all_prob[0, [0], 3:previous_tokens + current_tokens + 2].int()
            target_ids = all_prob[[1], [0], 2+previous_tokens:previous_tokens + target_tokens + 2].int()

            # if self.accelerator.is_main_process:
            #     ipdb.set_trace()

            # Second gather: ensure both processes send same-length tensors
            slice_start = prefix_len - previous_tokens - 1
            slice_end = prefix_len + target_tokens - 1

            # Check if we have enough prob_history and pad if needed
            available_len = model._prob_history.shape[1]
            if slice_end > available_len:
                actual_slice = model._prob_history[:, slice_start:available_len, :self.vocab_size]
                padding_len = slice_end - available_len
                padding = torch.zeros(1, padding_len, self.vocab_size, device=actual_slice.device,
                                      dtype=actual_slice.dtype)
                prob1 = torch.cat([actual_slice, padding], dim=1).to(torch.float32)
            else:
                prob1 = model._prob_history[:, slice_start:slice_end, :self.vocab_size].to(torch.float32)

            all_prob1 = self.accelerator.gather(prob1).to(device)
            draft_prob = all_prob1[[0], 1:, :]
            target_prob = all_prob1[[1], 1:, :]

            if cur_mode:
                # Pre-verify mode

                if target_tokens < current_tokens:

                    n = target_tokens
                    for i in range(target_tokens):
                        token = draft_ids[:, -current_tokens + i]
                        torch.manual_seed(self.seed + prefix_len - target_tokens + i + 1)
                        r = torch.rand(1, device=device)

                        if r > target_prob[:, -target_tokens + i, token] / draft_prob[:, -target_tokens + i, token]:
                            n = i
                            break

                    if n == target_tokens:
                        # Accept all guess tokens
                        cur_mode = False
                        prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                        num_acc_token += target_tokens
                        # if not self.accelerator.is_main_process:
                            # model.rollback(prefix_len)
                    else:
                        # Reject someone
                        assert n < target_tokens
                        prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:-current_tokens + n]), dim=1)
                        self.num_acc_tokens.append(num_acc_token + n)
                        num_acc_token = n
                        if self.accelerator.is_main_process:
                            model.rollback(prefix_len + n)
                        # else:
                        #     model.rollback(prefix_len)

                else:

                    n = current_tokens
                    for i in range(current_tokens):
                        token = draft_ids[:, -current_tokens + i]
                        torch.manual_seed(self.seed + prefix_len - target_tokens + i + 1)
                        r = torch.rand(1, device=device)

                        if r > target_prob[:, -target_tokens + i, token] / draft_prob[:, -target_tokens + i, token]:
                            n = i
                            break

                    if n == current_tokens:
                        prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                        num_acc_token += target_tokens
                        # if not self.accelerator.is_main_process:
                        #     model.rollback(prefix_len)
                    else:
                        # Reject someone
                        assert n < current_tokens
                        prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:-current_tokens + n]), dim=1)
                        self.num_acc_tokens.append(num_acc_token + n)
                        num_acc_token = 0

                        if self.accelerator.is_main_process:
                            model.rollback(prefix_len + n)
                        # else:
                        #     model.rollback(prefix_len)


            else:
                # Post-verify
                n = previous_tokens - 1
                # Post-verify
                for i in range(previous_tokens - 1):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - previous_tokens + i)
                    r = torch.rand(1, device=device)

                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break

                if n == previous_tokens - 1:
                    # Pre-verify in Post-verify
                    if target_tokens < current_tokens:

                        n = previous_tokens + target_tokens - 1
                        for i in range(previous_tokens - 1, previous_tokens + target_tokens - 1):
                            token = draft_ids[:, i]
                            torch.manual_seed(self.seed + prefix_len - previous_tokens + i)
                            r = torch.rand(1, device=device)

                            if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                                n = i
                                break

                        if n == previous_tokens + target_tokens - 1:
                            prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                            num_acc_token += previous_tokens + target_tokens - 1
                            # if not self.accelerator.is_main_process:
                            #     model.rollback(prefix_len)
                        else:
                            assert n < previous_tokens + target_tokens - 1
                            cur_mode = True
                            prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:-current_tokens + n - previous_tokens + 1]), dim=1)
                            self.num_acc_tokens.append(num_acc_token + n)
                            num_acc_token = 0
                            if self.accelerator.is_main_process:
                                model.rollback(prefix_len - previous_tokens + n + 1)
                            # else:
                            #     model.rollback(prefix_len)

                    else:

                        n = previous_tokens + current_tokens - 1
                        for i in range(previous_tokens, previous_tokens + current_tokens - 1):
                            token = draft_ids[:, i]
                            torch.manual_seed(self.seed + prefix_len - previous_tokens + i)
                            r = torch.rand(1, device=device)

                            if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                                n = i
                                break

                        if n == previous_tokens + current_tokens - 1:
                            prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:]), dim=1)
                            num_acc_token += previous_tokens + target_tokens - 1
                            # if not self.accelerator.is_main_process:
                            #     model.rollback(prefix_len)
                        else:
                            assert n < previous_tokens + current_tokens - 1
                            cur_mode = True
                            prefix = torch.cat((input_ids, draft_ids[:, -current_tokens:-current_tokens + n - previous_tokens + 1]), dim=1)
                            self.num_acc_tokens.append(num_acc_token + n)
                            num_acc_token = 0

                            if self.accelerator.is_main_process:
                                model.rollback(prefix_len - previous_tokens + n + 1)
                            # else:
                            #     model.rollback(prefix_len)


                else:
                    # reject someone, change the mode
                    assert n < previous_tokens - 1
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, :prefix_len - previous_tokens + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - previous_tokens + n + 1)

            previous_tokens = current_tokens

            # for llama-3.3
            for token in prefix[0, prefix_len:].tolist():
                if token == self.tokenizer.convert_tokens_to_ids("<|end_of_text|>") or token == self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"):
                    break

        return prefix

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")