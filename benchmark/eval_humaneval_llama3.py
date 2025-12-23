import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import ipdb
import random
from src.util import seed_everything, parse_arguments
from src.engine import Decoding
from collections import Counter


class EvalHumaneval(Decoding):
    def __init__(self, args):
        super().__init__(args)

        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()
        
        # 初始化检索库（如果启用）
        if getattr(args, 'use_retrieval_cache', False):
            load_from = getattr(args, 'retrieval_cache_name', None)
            if load_from and load_from != 'new':
                self.init_retrieval_cache(
                    cache_dir=getattr(args, 'retrieval_cache_dir', './retrieval_cache'),
                    max_ngram_size=getattr(args, 'pld_max_ngram_size', 3),
                    num_pred_tokens=getattr(args, 'pld_num_pred_tokens', 10),
                    load_from=load_from
                )
            else:
                self.init_retrieval_cache(
                    cache_dir=getattr(args, 'retrieval_cache_dir', './retrieval_cache'),
                    max_ngram_size=getattr(args, 'pld_max_ngram_size', 3),
                    num_pred_tokens=getattr(args, 'pld_num_pred_tokens', 10),
                    load_from=None
                )

        self.draft_time = []
        self.target_time = []
        self.acc_num = []

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading HumanEval data...", 3)
        data = []
        with open(os.path.join(self.args.data_path, "humaneval.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["prompt"])
                encode_special_token_flag = not (
                            "Llama-3" in self.args.draft_model and "Llama-3" in self.args.target_model)
                input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=encode_special_token_flag)
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        text = input_text.strip()
        return text

    def postprocess(self, input_text, output_text):
        # 检查bos_token是否存在且在output_text中
        if self.tokenizer.bos_token and output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text) + len(self.tokenizer.bos_token) + 1:]
        else:
            generation = output_text[len(input_text):]

        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                next_line = generation.index(stop_word)
                generation = generation[:next_line].strip()
        output_text = input_text + '\n    ' + generation
        output_text = output_text.replace("\t", "    ")

        return output_text

    @torch.no_grad()
    def eval(self):
        if self.args.eval_mode == "small" or self.args.eval_mode == "large":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "sd":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        elif self.args.eval_mode == "para_sd_wo_1":
            decoding = self.parallel_speculative_decoding_without_strategy_1
        elif self.args.eval_mode == "para_sd_wo_2":
            decoding = self.parallel_speculative_decoding_without_strategy_2
        elif self.args.eval_mode == "rc_para_sd":
            decoding = self.parallel_speculative_decoding_RC
        else:
            raise NotImplementedError

        out_path = os.path.join(self.args.exp_name, f"{self.args.eval_mode}_humaneval.jsonl")
        out_f = open(out_path, "a")
        wall_times = {"time": [], "num_tokens": []}
        
        task_idx = 0  # 任务计数器（用于检索库）
        
        for _ in range(self.args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)

            for datum in tqdm.tqdm(self.data, total=len(self.data), disable=not self.accelerator.is_main_process,
                                   ncols=50):
                input_ids = datum["input_ids"]
                
                # 检索库：任务开始回调
                if self._use_retrieval_cache:
                    self.on_task_start(task_idx, input_ids[0] if input_ids.dim() == 2 else input_ids)
                
                torch.cuda.synchronize()
                start_time = time.time()
                generate_ids = decoding(input_ids)
                torch.cuda.synchronize()
                end_time = time.time()
                
                # 检索库：任务结束回调
                if self._use_retrieval_cache:
                    self.on_task_end(generate_ids[0] if generate_ids.dim() == 2 else generate_ids)
                
                if self.accelerator.is_main_process:
                    wall_times["time"].append(end_time - start_time)
                    wall_times["num_tokens"].append(generate_ids.shape[1] - input_ids.shape[1])
                    output = self.postprocess(datum["input_text"], self.tokenizer.decode(generate_ids[0, :]))
                    out_f.write(json.dumps({"task_id": datum["task_id"], "time": end_time - start_time,
                                            "new_tokens": generate_ids.shape[1] - input_ids.shape[1],
                                            "completion": output}, ensure_ascii=False) + "\n")
                out_f.flush()
                
                task_idx += 1

        out_f.close()

        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)

        self.accelerator.wait_for_everyone()

        if (self.accelerator.num_processes == 1 and self.accelerator.is_main_process) or (
                self.accelerator.num_processes == 2 and not self.accelerator.is_main_process):
            print(f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
            speed_std = (torch.tensor(wall_times["num_tokens"]) / torch.tensor(wall_times["time"])).std().item()
            self.color_print(f"generate speed (tokens / second):  {speed:.2f} with std {speed_std}", 2)

        if self.accelerator.is_main_process:
            try:
                self.color_print(f"Mean accepted tokens: {sum(self.num_acc_tokens) / len(self.num_acc_tokens)}")
            except:
                pass
        
        # 保存检索库统计和缓存
        if self._use_retrieval_cache and self.accelerator.is_main_process:
            self.print_retrieval_stats()
            if getattr(self.args, 'save_retrieval_cache', False):
                cache_name = getattr(self.args, 'retrieval_cache_name', 'default')
                self.save_retrieval_cache(f"{cache_name}_after_eval")
                self.color_print(f"Retrieval cache saved as '{cache_name}_after_eval'", 2)


if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalHumaneval(args)
    alg.eval()
