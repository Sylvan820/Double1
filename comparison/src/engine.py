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
                
    def on_task_end(self, output_tokens: torch.Tensor = None, input_length: int = 0):
        """
        任务结束时的回调 - 更新检索库
        
        Args:
            output_tokens: 可选，任务生成的完整输出tokens（包括prompt）
            input_length: 输入的长度（保留参数但不使用，保存完整序列）
        """
        if self._retrieval_cache is not None:
            # 保存完整序列（包括prompt），因为prompt中的模式也可能被匹配
            if output_tokens is not None:
                if output_tokens.dim() == 2:
                    tokens_to_save = output_tokens[0]
                else:
                    tokens_to_save = output_tokens
                    
                if len(tokens_to_save) > 0:
                    # ========== 并行模式：同步两个进程的tokens ==========
                    if self.accelerator.num_processes == 2:
                        # 准备数据用于同步
                        # 使用固定长度的tensor来gather（取最大长度）
                        max_len = 4096  # 假设最大序列长度
                        
                        # 创建填充后的tensor
                        padded_tokens = torch.zeros(max_len, dtype=tokens_to_save.dtype, device=tokens_to_save.device)
                        actual_len = min(len(tokens_to_save), max_len)
                        padded_tokens[:actual_len] = tokens_to_save[:actual_len]
                        
                        # 创建长度tensor
                        len_tensor = torch.tensor([actual_len], device=tokens_to_save.device)
                        
                        # 同步等待
                        self.accelerator.wait_for_everyone()
                        
                        # Gather所有进程的tokens
                        all_tokens = self.accelerator.gather(padded_tokens)  # [2, max_len]
                        all_lens = self.accelerator.gather(len_tensor)  # [2]
                        
                        # 提取两个进程的实际tokens
                        tokens_rank0 = all_tokens[0, :all_lens[0].item()]
                        tokens_rank1 = all_tokens[1, :all_lens[1].item()]
                        
                        # 两个进程都添加双方的tokens到各自的检索库
                        # 这样两边的 datastore 都有完整数据
                        if self._current_task_idx < self._warmup_task_count:
                            # Warmup阶段：添加到prefix
                            self._retrieval_cache.add_to_prefix_verified(tokens_rank0)
                            self._retrieval_cache.add_to_prefix_verified(tokens_rank1)
                            self._retrieval_cache_manager.on_task_end(keep_dynamic=True)
                            if self.accelerator.is_main_process:
                                self.color_print(f"[Warmup {self._current_task_idx + 1}/{self._warmup_task_count}] Added to prefix cache (synced)", 3)
                        else:
                            # 正常阶段
                            self._retrieval_cache.add_to_prefix_verified(tokens_rank0)
                            self._retrieval_cache.add_to_prefix_verified(tokens_rank1)
                            self._retrieval_cache_manager.on_task_end(keep_dynamic=True)
                    else:
                        # ========== 单进程模式：原有逻辑 ==========
                        if self._current_task_idx < self._warmup_task_count:
                            self._retrieval_cache.add_to_prefix_verified(tokens_to_save)
                            self._retrieval_cache_manager.on_task_end(keep_dynamic=True)
                            self.color_print(f"[Warmup {self._current_task_idx + 1}/{self._warmup_task_count}] Added to prefix cache", 3)
                        else:
                            self._retrieval_cache.add_to_prefix_verified(tokens_to_save)
                            self._retrieval_cache_manager.on_task_end(keep_dynamic=True)
                
    def save_retrieval_cache(self, name: str = "default"):
        """保存检索库"""
        if self._retrieval_cache_manager is not None:
            self._retrieval_cache_manager.save_cache(name)
            self.color_print(f"Retrieval cache saved as '{name}'", 2)
            
    def print_retrieval_stats(self, process_name: str = None):
        """
        打印检索库统计信息
        
        Args:
            process_name: 可选，进程名称（如 "Draft Model" 或 "Target Model"）
        """
        if self._retrieval_cache is not None:
            if process_name:
                print(f"\n{'='*50}")
                print(f"Retrieval Cache Stats for: {process_name}")
                print(f"Process rank: {self.accelerator.process_index}")
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

    

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")