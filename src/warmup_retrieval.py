"""
热启动检索库的示例脚本

用法:
1. 运行热启动来构建前缀库
2. 保存缓存
3. 在后续任务中加载并使用

示例:
    # 热启动 (预运行10个任务)
    python warmup_retrieval_cache.py --warmup --num_tasks 10
    
    # 使用已有的缓存
    python warmup_retrieval_cache.py --load default
"""

import argparse
import torch
import json
import os
from transformers import AutoTokenizer
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval_cache import RetrievalCache, RetrievalCacheManager


def load_dataset_samples(data_path: str, num_samples: int = 100, dataset_type: str = "humaneval"):
    """
    从数据集加载样本用于热启动
    
    Args:
        data_path: 数据文件路径
        num_samples: 加载的样本数量
        dataset_type: 数据集类型 (humaneval, gsm8k, mt_bench)
    """
    samples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            
            if dataset_type == "humaneval":
                # HumanEval格式
                prompt = data.get("prompt", "")
                samples.append(prompt)
            elif dataset_type == "gsm8k":
                # GSM8K格式
                question = data.get("question", "")
                samples.append(question)
            elif dataset_type == "mt_bench":
                # MT-Bench格式
                question = data.get("question", data.get("prompt", ""))
                samples.append(question)
            else:
                # 通用格式
                text = data.get("prompt", data.get("text", data.get("question", "")))
                samples.append(text)
                
    return samples


def warmup_from_dataset(cache_manager: RetrievalCacheManager, 
                        tokenizer, 
                        data_path: str,
                        num_samples: int = 100,
                        dataset_type: str = "humaneval",
                        device: str = "cuda"):
    """
    从数据集进行热启动
    """
    print(f"Loading {num_samples} samples from {data_path}...")
    samples = load_dataset_samples(data_path, num_samples, dataset_type)
    
    if len(samples) == 0:
        print("No samples loaded!")
        return
        
    print(f"Loaded {len(samples)} samples")
    
    # 创建缓存并热启动
    cache = cache_manager.create_cache(device)
    
    for i, sample in enumerate(samples):
        if len(sample) == 0:
            continue
            
        # Tokenize
        tokens = tokenizer.encode(sample, return_tensors="pt").to(device)
        
        # 添加到前缀库
        cache.add_to_prefix_cache(tokens[0])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")
            
    print(f"\nWarmup complete!")
    cache.print_stats()
    
    return cache


def warmup_from_generation_results(cache_manager: RetrievalCacheManager,
                                    results_path: str,
                                    tokenizer,
                                    device: str = "cuda"):
    """
    从之前的生成结果进行热启动
    
    Args:
        results_path: 生成结果文件路径 (jsonl格式，每行包含 {"output": "..."})
    """
    print(f"Loading generation results from {results_path}...")
    
    cache = cache_manager.create_cache(device)
    
    count = 0
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            output = data.get("output", data.get("response", data.get("completion", "")))
            
            if len(output) == 0:
                continue
                
            tokens = tokenizer.encode(output, return_tensors="pt").to(device)
            cache.add_to_prefix_cache(tokens[0])
            count += 1
            
            if count % 10 == 0:
                print(f"  Processed {count} samples")
                
    print(f"\nWarmup from generation results complete! Loaded {count} samples")
    cache.print_stats()
    
    return cache


def main():
    parser = argparse.ArgumentParser(description="Warmup Retrieval Cache")
    parser.add_argument("--warmup", action="store_true", help="Run warmup from dataset")
    parser.add_argument("--data_path", type=str, default="./data/humaneval.jsonl", help="Dataset path")
    parser.add_argument("--dataset_type", type=str, default="humaneval", 
                        choices=["humaneval", "gsm8k", "mt_bench", "other"])
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for warmup")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer name")
    parser.add_argument("--cache_dir", type=str, default="./retrieval_cache", help="Cache directory")
    parser.add_argument("--cache_name", type=str, default="default", help="Cache name to save/load")
    parser.add_argument("--load", type=str, default=None, help="Load existing cache by name")
    parser.add_argument("--results_path", type=str, default=None, help="Load from generation results")
    parser.add_argument("--max_ngram_size", type=int, default=3, help="Maximum n-gram size")
    parser.add_argument("--num_pred_tokens", type=int, default=10, help="Number of predicted tokens")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 初始化缓存管理器
    cache_manager = RetrievalCacheManager(
        cache_dir=args.cache_dir,
        max_ngram_size=args.max_ngram_size,
        num_pred_tokens=args.num_pred_tokens
    )
    
    # 加载tokenizer
    print(f"Loading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    if args.load:
        # 加载已有的缓存
        print(f"Loading cache: {args.load}")
        cache_manager.load_cache(args.load, device)
        cache = cache_manager.get_cache()
        if cache:
            cache.print_stats()
    elif args.results_path:
        # 从生成结果热启动
        cache = warmup_from_generation_results(cache_manager, args.results_path, tokenizer, device)
        cache_manager.save_cache(args.cache_name)
        print(f"Cache saved as '{args.cache_name}'")
    elif args.warmup:
        # 从数据集热启动
        cache = warmup_from_dataset(
            cache_manager, 
            tokenizer, 
            args.data_path,
            args.num_samples,
            args.dataset_type,
            device
        )
        cache_manager.save_cache(args.cache_name)
        print(f"Cache saved as '{args.cache_name}'")
    else:
        print("Please specify --warmup, --load, or --results_path")
        parser.print_help()


if __name__ == "__main__":
    main()
