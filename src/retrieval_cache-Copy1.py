"""
RetrievalCache: 高效的n-gram检索库，支持热启动和动态tokens扩展

特性：
1. 持久化前缀库 (prefix_cache): 通过预运行任务热启动，任务结束后不清除
2. 动态历史库 (dynamic_cache): 当前任务的所有tokens，包括被拒绝的tokens
3. 支持大小模型的所有生成tokens，不仅仅是被验证接受的tokens
"""

import torch
import pickle
import os
from typing import Optional, List, Tuple, Dict
from collections import defaultdict


class RetrievalCache:
    """
    高效的n-gram检索库
    
    结构:
    - prefix_cache: 持久化前缀库，热启动后不会被清除
    - dynamic_cache: 当前任务的动态历史tokens
    - rejected_tokens_cache: 被拒绝的tokens缓存
    """
    
    def __init__(self, max_ngram_size: int = 3, num_pred_tokens: int = 10, device: str = "cuda"):
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
        self.device = device
        
        # 持久化前缀库 - 热启动后保持不变
        self.prefix_cache: List[torch.Tensor] = []  # 存储多个序列的tokens
        self.prefix_ngram_index: Dict[tuple, List[Tuple[int, int]]] = defaultdict(list)  # ngram -> [(seq_idx, pos), ...]
        
        # 动态历史库 - 当前任务的tokens，每个任务结束后可选择保留或清除
        self.dynamic_cache: List[torch.Tensor] = []  # 当前任务的accepted tokens
        self.dynamic_ngram_index: Dict[tuple, List[Tuple[int, int]]] = defaultdict(list)
        
        # 被拒绝tokens缓存 - 大小模型生成但被拒绝的tokens
        self.rejected_cache: List[torch.Tensor] = []
        self.rejected_ngram_index: Dict[tuple, List[Tuple[int, int]]] = defaultdict(list)
        
        # 统计信息
        self.stats = {
            "prefix_hits": 0,
            "dynamic_hits": 0,
            "rejected_hits": 0,
            "total_queries": 0,
            "fallback_to_input": 0,
        }
        
    def _build_ngram_index(self, tokens: torch.Tensor, seq_idx: int, 
                           index_dict: Dict[tuple, List[Tuple[int, int]]]):
        """为单个序列构建n-gram索引"""
        if tokens.dim() == 2:
            tokens = tokens[0]
        tokens_list = tokens.tolist()
        seq_len = len(tokens_list)
        
        for ngram_size in range(1, self.max_ngram_size + 1):
            for i in range(seq_len - ngram_size + 1):
                ngram = tuple(tokens_list[i:i + ngram_size])
                index_dict[ngram].append((seq_idx, i))
    
    def add_to_prefix_cache(self, tokens: torch.Tensor):
        """
        添加tokens到持久化前缀库（用于热启动）
        
        Args:
            tokens: 要添加的token序列 [1, seq_len] or [seq_len]
        """
        if tokens.dim() == 2:
            tokens = tokens[0]
        tokens = tokens.to(self.device)
        
        seq_idx = len(self.prefix_cache)
        self.prefix_cache.append(tokens.clone())
        self._build_ngram_index(tokens, seq_idx, self.prefix_ngram_index)
        
    def add_to_dynamic_cache(self, tokens: torch.Tensor):
        """
        添加tokens到动态历史库（当前任务的tokens）
        
        Args:
            tokens: 要添加的token序列 [1, seq_len] or [seq_len]
        """
        if tokens.dim() == 2:
            tokens = tokens[0]
        tokens = tokens.to(self.device)
        
        seq_idx = len(self.dynamic_cache)
        self.dynamic_cache.append(tokens.clone())
        self._build_ngram_index(tokens, seq_idx, self.dynamic_ngram_index)
        
    def add_rejected_tokens(self, tokens: torch.Tensor, source: str = "draft"):
        """
        添加被拒绝的tokens到缓存
        
        Args:
            tokens: 被拒绝的token序列
            source: 来源 ("draft" 或 "target")
        """
        if tokens.dim() == 2:
            tokens = tokens[0]
        if len(tokens) == 0:
            return
        tokens = tokens.to(self.device)
        
        seq_idx = len(self.rejected_cache)
        self.rejected_cache.append(tokens.clone())
        self._build_ngram_index(tokens, seq_idx, self.rejected_ngram_index)
        
    def _search_in_index(self, ngram: tuple, cache: List[torch.Tensor], 
                         index: Dict[tuple, List[Tuple[int, int]]],
                         exclude_last: bool = True) -> Optional[torch.Tensor]:
        """
        在指定的索引中搜索n-gram并返回后续tokens
        
        Args:
            ngram: 要搜索的n-gram元组
            cache: token缓存列表
            index: n-gram索引
            exclude_last: 是否排除最后一个匹配（避免自匹配）
            
        Returns:
            匹配到的后续tokens，或None
        """
        if ngram not in index:
            return None
            
        matches = index[ngram]
        ngram_size = len(ngram)
        
        for seq_idx, pos in matches:
            seq = cache[seq_idx]
            start_idx = pos + ngram_size
            end_idx = start_idx + self.num_pred_tokens
            
            # 确保有足够的后续tokens
            if end_idx <= len(seq) and start_idx < len(seq):
                # 排除自匹配检查（如果需要）
                candidate = seq[start_idx:end_idx]
                if len(candidate) > 0:
                    return candidate
                    
        return None
    
    def find_candidate_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        在所有缓存中查找候选tokens
        
        搜索优先级：
        1. 首先在持久化前缀库中搜索（热启动数据）
        2. 然后在动态历史库中搜索（当前任务的tokens）
        3. 最后在被拒绝tokens库中搜索
        4. 如果都没找到，回退到原始输入序列搜索
        
        Args:
            input_ids: 当前输入序列 [1, seq_len]
            
        Returns:
            候选tokens序列
        """
        self.stats["total_queries"] += 1
        
        if input_ids.dim() == 2:
            tokens = input_ids[0]
        else:
            tokens = input_ids
            
        input_length = len(tokens)
        
        if input_length < 1:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # 从大到小尝试不同的n-gram大小
        for ngram_size in range(min(self.max_ngram_size, input_length), 0, -1):
            ngram = tuple(tokens[-ngram_size:].tolist())
            
            # 1. 搜索持久化前缀库
            if self.prefix_cache:
                result = self._search_in_index(ngram, self.prefix_cache, self.prefix_ngram_index)
                if result is not None:
                    self.stats["prefix_hits"] += 1
                    return result
                    
            # 2. 搜索动态历史库
            if self.dynamic_cache:
                result = self._search_in_index(ngram, self.dynamic_cache, self.dynamic_ngram_index)
                if result is not None:
                    self.stats["dynamic_hits"] += 1
                    return result
                    
            # 3. 搜索被拒绝tokens库
            if self.rejected_cache:
                result = self._search_in_index(ngram, self.rejected_cache, self.rejected_ngram_index)
                if result is not None:
                    self.stats["rejected_hits"] += 1
                    return result
        
        # 4. 回退到原始输入序列搜索（原始PLD逻辑）
        self.stats["fallback_to_input"] += 1
        return self._fallback_search_in_input(input_ids)
    
    def _fallback_search_in_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """回退到在输入序列中搜索（原始PLD逻辑）"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_length = input_ids.size(1)
        
        if self.max_ngram_size <= 0 or self.num_pred_tokens <= 0 or self.max_ngram_size > input_length:
            return torch.tensor([], dtype=torch.long, device=input_ids.device)

        for ngram_size in range(self.max_ngram_size, 0, -1):
            ngram = input_ids[0, -ngram_size:].tolist()
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
            ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)
            matches = (windows == ngram_tensor).all(dim=2)
            match_indices = matches.nonzero(as_tuple=True)[1]

            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_pred_tokens
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return input_ids[0, start_idx:end_idx]

        return torch.tensor([], dtype=torch.long, device=input_ids.device)
    
    def clear_dynamic_cache(self):
        """清除动态历史库（任务结束时调用）"""
        self.dynamic_cache.clear()
        self.dynamic_ngram_index.clear()
        
    def clear_rejected_cache(self):
        """清除被拒绝tokens缓存"""
        self.rejected_cache.clear()
        self.rejected_ngram_index.clear()
        
    def clear_task_caches(self):
        """清除当前任务的所有缓存（动态+被拒绝），保留前缀库"""
        self.clear_dynamic_cache()
        self.clear_rejected_cache()
        
    def merge_dynamic_to_prefix(self):
        """将动态历史库合并到持久化前缀库"""
        for tokens in self.dynamic_cache:
            self.add_to_prefix_cache(tokens)
        self.clear_dynamic_cache()
        
    def merge_rejected_to_prefix(self):
        """将被拒绝tokens合并到持久化前缀库"""
        for tokens in self.rejected_cache:
            self.add_to_prefix_cache(tokens)
        self.clear_rejected_cache()
        
    def save(self, path: str):
        """保存检索库到文件"""
        state = {
            "prefix_cache": [t.cpu() for t in self.prefix_cache],
            "prefix_ngram_index": dict(self.prefix_ngram_index),
            "max_ngram_size": self.max_ngram_size,
            "num_pred_tokens": self.num_pred_tokens,
            "stats": self.stats,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"RetrievalCache saved to {path}")
        
    def load(self, path: str):
        """从文件加载检索库"""
        if not os.path.exists(path):
            print(f"Cache file {path} not found, starting with empty cache")
            return
            
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.prefix_cache = [t.to(self.device) for t in state["prefix_cache"]]
        self.prefix_ngram_index = defaultdict(list, state["prefix_ngram_index"])
        self.max_ngram_size = state.get("max_ngram_size", self.max_ngram_size)
        self.num_pred_tokens = state.get("num_pred_tokens", self.num_pred_tokens)
        self.stats = state.get("stats", self.stats)
        print(f"RetrievalCache loaded from {path}, prefix_cache size: {len(self.prefix_cache)}")
        
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
            
        stats_with_rates = self.stats.copy()
        stats_with_rates["prefix_hit_rate"] = self.stats["prefix_hits"] / total
        stats_with_rates["dynamic_hit_rate"] = self.stats["dynamic_hits"] / total
        stats_with_rates["rejected_hit_rate"] = self.stats["rejected_hits"] / total
        stats_with_rates["fallback_rate"] = self.stats["fallback_to_input"] / total
        stats_with_rates["total_cache_size"] = (len(self.prefix_cache) + 
                                                 len(self.dynamic_cache) + 
                                                 len(self.rejected_cache))
        return stats_with_rates
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n=== RetrievalCache Statistics ===")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Prefix hits: {stats['prefix_hits']} ({stats.get('prefix_hit_rate', 0):.2%})")
        print(f"Dynamic hits: {stats['dynamic_hits']} ({stats.get('dynamic_hit_rate', 0):.2%})")
        print(f"Rejected hits: {stats['rejected_hits']} ({stats.get('rejected_hit_rate', 0):.2%})")
        print(f"Fallback to input: {stats['fallback_to_input']} ({stats.get('fallback_rate', 0):.2%})")
        print(f"Cache sizes - Prefix: {len(self.prefix_cache)}, Dynamic: {len(self.dynamic_cache)}, Rejected: {len(self.rejected_cache)}")
        print("=================================\n")


class RetrievalCacheManager:
    """
    检索库管理器 - 管理热启动和任务间的检索库传递
    """
    
    def __init__(self, cache_dir: str = "./retrieval_cache", 
                 max_ngram_size: int = 3, 
                 num_pred_tokens: int = 10):
        self.cache_dir = cache_dir
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # 全局检索库实例
        self.cache: Optional[RetrievalCache] = None
        
    def create_cache(self, device: str = "cuda") -> RetrievalCache:
        """创建新的检索库"""
        self.cache = RetrievalCache(
            max_ngram_size=self.max_ngram_size,
            num_pred_tokens=self.num_pred_tokens,
            device=device
        )
        return self.cache
        
    def get_cache(self) -> Optional[RetrievalCache]:
        """获取当前检索库"""
        return self.cache
        
    def warmup_from_dataset(self, tokenizer, dataset_samples: List[str], 
                            device: str = "cuda"):
        """
        从数据集样本进行热启动
        
        Args:
            tokenizer: tokenizer实例
            dataset_samples: 数据集样本文本列表
            device: 设备
        """
        if self.cache is None:
            self.create_cache(device)
            
        print(f"Warming up retrieval cache with {len(dataset_samples)} samples...")
        
        for i, sample in enumerate(dataset_samples):
            tokens = tokenizer.encode(sample, return_tensors="pt").to(device)
            self.cache.add_to_prefix_cache(tokens)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset_samples)} samples")
                
        print(f"Warmup complete. Prefix cache size: {len(self.cache.prefix_cache)}")
        
    def warmup_from_generation(self, generated_tokens: torch.Tensor):
        """
        从生成的tokens进行热启动（用于任务运行后积累）
        
        Args:
            generated_tokens: 生成的token序列
        """
        if self.cache is not None:
            self.cache.add_to_prefix_cache(generated_tokens)
            
    def save_cache(self, name: str = "default"):
        """保存检索库"""
        if self.cache is not None:
            path = os.path.join(self.cache_dir, f"{name}.pkl")
            self.cache.save(path)
            
    def load_cache(self, name: str = "default", device: str = "cuda"):
        """加载检索库"""
        path = os.path.join(self.cache_dir, f"{name}.pkl")
        if self.cache is None:
            self.create_cache(device)
        self.cache.load(path)
        
    def on_task_start(self):
        """任务开始时的回调"""
        if self.cache is not None:
            # 清除上一个任务的动态缓存，保留前缀库
            self.cache.clear_task_caches()
            
    def on_task_end(self, keep_dynamic: bool = True, keep_rejected: bool = True):
        """
        任务结束时的回调
        
        Args:
            keep_dynamic: 是否将动态缓存合并到前缀库
            keep_rejected: 是否将被拒绝tokens合并到前缀库
        """
        if self.cache is not None:
            if keep_dynamic:
                self.cache.merge_dynamic_to_prefix()
            else:
                self.cache.clear_dynamic_cache()
                
            if keep_rejected:
                self.cache.merge_rejected_to_prefix()
            else:
                self.cache.clear_rejected_cache()


# 全局函数，替代原来的 find_candidate_pred_tokens
@torch.no_grad()
def find_candidate_pred_tokens_with_cache(
    input_ids: torch.Tensor, 
    retrieval_cache: Optional[RetrievalCache] = None,
    max_ngram_size: int = 3, 
    num_pred_tokens: int = 10
) -> torch.Tensor:
    """
    带检索库的候选tokens查找函数
    
    Args:
        input_ids: 输入token序列
        retrieval_cache: 检索库实例，如果为None则回退到原始逻辑
        max_ngram_size: 最大n-gram大小
        num_pred_tokens: 预测tokens数量
        
    Returns:
        候选tokens序列
    """
    # 如果有检索库，使用检索库查找
    if retrieval_cache is not None:
        return retrieval_cache.find_candidate_tokens(input_ids)
    
    # 否则回退到原始PLD逻辑
    input_length = input_ids.size(1) if input_ids.dim() == 2 else input_ids.size(0)
    
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    for ngram_size in range(max_ngram_size, 0, -1):
        ngram = input_ids[0, -ngram_size:].tolist()
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)
        matches = (windows == ngram_tensor).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]

        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    return torch.tensor([], dtype=torch.long, device=input_ids.device)
