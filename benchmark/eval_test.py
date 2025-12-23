import os
import sys

# 假设你的父目录设置是正确的
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


class TestSinglePrompt(Decoding):
    """
    一个修改后的版本，用于测试单个 prompt，而不是加载整个 HumanEval 数据集。
    """
    def __init__(self, args):
        super().__init__(args)

        # 加载必要组件
        self.color_print("Loading tokenizer and model...", 3)
        self.load_tokenizer()
        self.load_model()
        
        # 加载我们自定义的单个 prompt 数据
        self.load_data()

        # 初始化统计数据
        self.draft_time = []
        self.target_time = []
        self.acc_num = []
        self.num_acc_tokens = [] # 确保这个列表存在，以便在 eval 结束时打印统计信息

    def load_data(self):
        """
        重写 load_data 方法，只加载和处理你指定的单个 prompt。
        """
        self.color_print(f"Preparing single prompt...", 3)
        
        # prompt_text = "The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle? What's area of the circle circumscribing the triangle?"
        prompt_text = "Write a function to find the highest common ancestor (not LCA) of two nodes in a binary tree. What if it is not a binary tree? reference: Very simple. The function should just return the root of the tree. Same answer. It's still the root of the tree."]
        
        # 1. 应用聊天模板
        # 我们在这里存储原始文本，用于后续的 postprocess
        self.input_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        
        # 2. 编码
        # 检查 Llama-3 的特殊 token 标志，就像你的原始代码一样
        encode_special_token_flag = not (
            "Llama-3" in self.args.draft_model and "Llama-3" in self.args.target_model
        )
        
        input_ids_list = self.tokenizer.encode(self.input_text, add_special_tokens=encode_special_token_flag)
        
        # 3. 转换为 PyTorch 张量
        # 你的解码函数期望一个 [1, seq_len] 的 batch
        self.input_ids = torch.tensor(input_ids_list).unsqueeze(0)

        self.color_print(f"--- Prompt Text ---", 4)
        print(self.input_text)
        self.color_print(f"---------------------", 4)
        self.color_print(f"Input IDs shape: {self.input_ids.shape}", 4)

        # 将 self.data 设置为包含一个 "dummy" 元素的列表，以便 eval 循环可以运行
        # 或者，我们可以直接修改 eval 循环
        # 为了更简洁，我们将修改 eval 循环
        
    def preprocess(self, input_text):
        # 这个方法现在不再需要了，因为 load_data 是自定义的
        pass

    def postprocess(self, input_text, output_text):
        """
        简化的 postprocess，只移除输入的 prompt。
        """
        generation = output_text
        
        # 尝试从开头移除 prompt 文本
        if generation.startswith(input_text):
            generation = generation[len(input_text):]
        
        # 你的原始代码检查了 bos_token，但如果我们在 decode 时使用
        # skip_special_tokens=True，这通常不是必需的。
        # 为安全起见，我们保留一个类似的检查：
        if self.tokenizer.bos_token and generation.startswith(self.tokenizer.bos_token):
            generation = generation[len(self.tokenizer.bos_token):]

        # 移除了 HumanEval 特定的停止词（如 \nclass, \ndef 等）
        
        return generation.strip()

    @torch.no_grad()
    def eval(self):
        # 1. 选择解码方法 (与你的代码相同)
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
            
        # 2. 移除文件输出，我们将直接打印到控制台
        # out_path = ...
        # out_f = ...

        wall_times = {"time": [], "num_tokens": []}
        self.color_print(f"Starting evaluation for {self.args.num_samples_per_task} sample(s) on the single prompt...", 3)

        for i in range(self.args.num_samples_per_task):
            # 3. 设置随机种子 (与你的代码相同)
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 100000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)
            
            self.color_print(f"\n--- Sample {i+1}/{self.args.num_samples_per_task} (Seed: {self.seed}) ---", 1)

            # 4. **核心修改**: 不再遍历 self.data，而是直接使用 self.input_ids
            input_ids = self.input_ids
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 运行解码
            generate_ids = decoding(input_ids)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 5. 处理和打印输出
            if self.accelerator.is_main_process:
                run_time = end_time - start_time
                new_tokens_count = generate_ids.shape[1] - input_ids.shape[1]
                
                wall_times["time"].append(run_time)
                wall_times["num_tokens"].append(new_tokens_count)
                
                # 使用 skip_special_tokens=True 来获得更清晰的聊天输出
                output_full = self.tokenizer.decode(generate_ids[0, :], skip_special_tokens=True)
                
                # 使用我们简化的 postprocess
                generation_text = self.postprocess(self.input_text, output_full)
                
                # 直接打印结果
                self.color_print(f"Generated Output:", 2)
                print(generation_text)
                self.color_print(f"---------------------", 2)
                self.color_print(f"Time: {run_time:.4f}s | New Tokens: {new_tokens_count} | Speed: {new_tokens_count/run_time:.2f} T/s", 3)
                
                # 你的原始代码将结果写入文件，我们在这里跳过
                # out_f.write(...)
                # out_f.flush()

        # 6. 打印最终统计信息 (与你的代码相同)
        
        # 移除 out_f.close()
        # out_f.close()

        self.color_print(f"\n--- Final Stats ---", 1)
        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)

        self.accelerator.wait_for_everyone()

        if (self.accelerator.num_processes == 1 and self.accelerator.is_main_process) or \
           (self.accelerator.num_processes == 2 and not self.accelerator.is_main_process):
            print(f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if sum(wall_times["time"]) > 0:
                speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
                speed_std = (torch.tensor(wall_times["num_tokens"]) / torch.tensor(wall_times["time"])).std().item()
                self.color_print(f"Avg. generate speed (tokens / second):  {speed:.2f} with std {speed_std:.2f}", 2)
            else:
                self.color_print("No time recorded for speed calculation.", 5)

        if self.accelerator.is_main_process:
            try:
                if len(self.num_acc_tokens) > 0:
                    self.color_print(f"Mean accepted tokens: {sum(self.num_acc_tokens) / len(self.num_acc_tokens):.2f}", 2)
                else:
                    self.color_print("No accepted tokens recorded (e.g., running non-SD mode).", 4)
            except Exception as e:
                self.color_print(f"Could not calculate mean accepted tokens: {e}", 5)
                pass


if __name__ == "__main__":
    args = parse_arguments()
    # 使用我们的新类
    alg = TestSinglePrompt(args)
    alg.eval()