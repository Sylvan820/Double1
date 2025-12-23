"""
RetrievalCache: Double Datastore 高效n-gram检索库

Double Datastore 策略:
1. Target Datastore (D_p): 只存储大模型验证后的tokens，保持高精度
2. Draft Datastore (D_q): 存储大小模型的所有输出，让小模型学习大模型的分布

特性：
- 持久化前缀库 (prefix): 通过预运行任务热启动，任务结束后保留
- 动态历史库 (dynamic): 当前任务的tokens，任务结束可合并到prefix
- 大模型Datastore只接收verified tokens
- 小模型Datastore接收所有tokens (draft + target)
"""

import torch
import pickle
import os
from typing import Optional, List, Tuple, Dict
from collections import defaultdict


class SingleDatastore:
    """单个Datastore的实现"""
    
    def __init__(self, name: str, max_ngram_size: int = 3, num_pred_tokens: int = 10, device: str = "cuda"):
        self.name = name
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
        self.device = device
        
        # 持久化前缀库 - warmup阶段和历史任务积累的tokens
        self.prefix_cache: List[torch.Tensor] = []
        self.prefix_ngram_index: Dict[tuple, List[Tuple[int, int]]] = defaultdict(list)
        
        # 动态历史库 - 当前任务的tokens
        self.dynamic_cache: List[torch.Tensor] = []
        self.dynamic_ngram_index: Dict[tuple, List[Tuple[int, int]]] = defaultdict(list)
        
        # 统计信息
        self.stats = {
            "prefix_hits": 0,
            "dynamic_hits": 0,
            "input_hits": 0,
            "total_queries": 0,
            "miss": 0,
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
    
    def add_to_prefix(self, tokens: torch.Tensor):
        """添加tokens到持久化前缀库"""
        if tokens.dim() == 2:
            tokens = tokens[0]
        if len(tokens) == 0:
            return
        tokens = tokens.to(self.device)
        
        seq_idx = len(self.prefix_cache)
        self.prefix_cache.append(tokens.clone())
        self._build_ngram_index(tokens, seq_idx, self.prefix_ngram_index)
    
    def add_to_dynamic(self, tokens: torch.Tensor):
        """添加tokens到动态历史库"""
        if tokens.dim() == 2:
            tokens = tokens[0]
        if len(tokens) == 0:
            return
        tokens = tokens.to(self.device)
        
        seq_idx = len(self.dynamic_cache)
        self.dynamic_cache.append(tokens.clone())
        self._build_ngram_index(tokens, seq_idx, self.dynamic_ngram_index)
    
    def _search_in_index(self, ngram: tuple, cache: List[torch.Tensor], 
                         index: Dict[tuple, List[Tuple[int, int]]]) -> Optional[torch.Tensor]:
        """在指定索引中搜索n-gram并返回后续tokens"""
        if ngram not in index:
            return None
            
        matches = index[ngram]
        ngram_size = len(ngram)
        
        for seq_idx, pos in matches:
            seq = cache[seq_idx]
            start_idx = pos + ngram_size
            end_idx = start_idx + self.num_pred_tokens
            
            if end_idx <= len(seq) and start_idx < len(seq):
                candidate = seq[start_idx:end_idx]
                if len(candidate) > 0:
                    return candidate
        return None
    
    def _search_in_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """在输入序列中搜索（原始PLD逻辑）"""
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
    
    def find_candidate_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        查找候选tokens
        搜索优先级: prefix_cache -> dynamic_cache -> input_ids
        """
        self.stats["total_queries"] += 1
        
        if input_ids.dim() == 2:
            tokens = input_ids[0]
        else:
            tokens = input_ids
            
        input_length = len(tokens)
        
        if input_length < 1:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # 1. 搜索持久化前缀库 (最高优先级 - warmup + 历史任务)
        for ngram_size in range(min(self.max_ngram_size, input_length), 0, -1):
            ngram = tuple(tokens[-ngram_size:].tolist())
            if self.prefix_cache:
                result = self._search_in_index(ngram, self.prefix_cache, self.prefix_ngram_index)
                if result is not None:
                    self.stats["prefix_hits"] += 1
                    return result
        
        # 2. 搜索动态历史库 (当前任务已生成的tokens)
        for ngram_size in range(min(self.max_ngram_size, input_length), 0, -1):
            ngram = tuple(tokens[-ngram_size:].tolist())
            if self.dynamic_cache:
                result = self._search_in_index(ngram, self.dynamic_cache, self.dynamic_ngram_index)
                if result is not None:
                    self.stats["dynamic_hits"] += 1
                    return result
        
        # 3. 搜索当前输入序列 (原始PLD逻辑 - 最低优先级)
        input_result = self._search_in_input(input_ids)
        if len(input_result) > 0:
            self.stats["input_hits"] += 1
            return input_result
        
        # 没找到
        self.stats["miss"] += 1
        return torch.tensor([], dtype=torch.long, device=self.device)
    
    def clear_dynamic(self):
        """清除动态历史库"""
        self.dynamic_cache.clear()
        self.dynamic_ngram_index.clear()
    
    def merge_dynamic_to_prefix(self):
        """将动态历史库合并到持久化前缀库"""
        for tokens in self.dynamic_cache:
            self.add_to_prefix(tokens)
        self.clear_dynamic()
    
    def get_state(self) -> Dict:
        """获取可序列化的状态"""
        return {
            "name": self.name,
            "prefix_cache": [t.cpu() for t in self.prefix_cache],
            "prefix_ngram_index": dict(self.prefix_ngram_index),
            "max_ngram_size": self.max_ngram_size,
            "num_pred_tokens": self.num_pred_tokens,
            "stats": self.stats.copy(),
        }
    
    def load_state(self, state: Dict):
        """从状态恢复"""
        self.name = state.get("name", self.name)
        self.prefix_cache = [t.to(self.device) for t in state["prefix_cache"]]
        self.prefix_ngram_index = defaultdict(list, state["prefix_ngram_index"])
        self.max_ngram_size = state.get("max_ngram_size", self.max_ngram_size)
        self.num_pred_tokens = state.get("num_pred_tokens", self.num_pred_tokens)
        self.stats = state.get("stats", self.stats)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
            
        stats = self.stats.copy()
        stats["prefix_hit_rate"] = self.stats["prefix_hits"] / total
        stats["input_hit_rate"] = self.stats["input_hits"] / total
        stats["dynamic_hit_rate"] = self.stats["dynamic_hits"] / total
        stats["miss_rate"] = self.stats["miss"] / total
        stats["prefix_size"] = len(self.prefix_cache)
        stats["dynamic_size"] = len(self.dynamic_cache)
        return stats


class RetrievalCache:
    """
    Double Datastore 检索库
    
    结构:
    - target_datastore (D_p): 只存储大模型验证后的tokens，高精度
    - draft_datastore (D_q): 存储大小模型的所有输出，高覆盖率
    """
    
    def __init__(self, max_ngram_size: int = 3, num_pred_tokens: int = 10, device: str = "cuda"):
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
        self.device = device
        
        # Target Datastore (D_p) - 只存储大模型验证后的tokens
        self.target_datastore = SingleDatastore(
            name="target", 
            max_ngram_size=max_ngram_size,
            num_pred_tokens=num_pred_tokens, 
            device=device
        )
        
        # Draft Datastore (D_q) - 存储大小模型的所有输出
        self.draft_datastore = SingleDatastore(
            name="draft",
            max_ngram_size=max_ngram_size,
            num_pred_tokens=num_pred_tokens,
            device=device
        )
        
        # 全局统计信息
        self.stats = {
            "total_queries": 0,
            "target_used": 0,  # 使用target datastore的次数
            "draft_used": 0,   # 使用draft datastore的次数
        }
    
    # ==================== 添加tokens的API ====================
    
    def add_verified_tokens(self, tokens: torch.Tensor):
        """
        添加大模型验证后的tokens
        - 同时加入 target_datastore 和 draft_datastore
        
        Args:
            tokens: 大模型验证后的token序列
        """
        self.target_datastore.add_to_dynamic(tokens)
        self.draft_datastore.add_to_dynamic(tokens)
    
    def add_draft_tokens(self, tokens: torch.Tensor):
        """
        添加小模型生成的tokens（包括被拒绝的）
        - 只加入 draft_datastore
        
        Args:
            tokens: 小模型生成的token序列
        """
        self.draft_datastore.add_to_dynamic(tokens)
    
    def add_to_prefix_verified(self, tokens: torch.Tensor):
        """
        添加验证后的tokens到持久化前缀库（warmup阶段）
        - 同时加入两个datastore的prefix
        
        Args:
            tokens: 验证后的token序列
        """
        self.target_datastore.add_to_prefix(tokens)
        self.draft_datastore.add_to_prefix(tokens)
    
    def add_to_prefix_draft(self, tokens: torch.Tensor):
        """
        添加小模型生成的tokens到持久化前缀库（warmup阶段）
        - 只加入 draft_datastore 的prefix
        
        Args:
            tokens: 小模型生成的token序列
        """
        self.draft_datastore.add_to_prefix(tokens)
    
    # ==================== 兼容旧API ====================
    
    def add_to_prefix_cache(self, tokens: torch.Tensor):
        """兼容旧API - 添加到两个datastore的prefix"""
        self.add_to_prefix_verified(tokens)
    
    def add_to_dynamic_cache(self, tokens: torch.Tensor):
        """兼容旧API - 添加验证后的tokens"""
        self.add_verified_tokens(tokens)
    
    def add_rejected_tokens(self, tokens: torch.Tensor, source: str = "draft"):
        """兼容旧API - 添加被拒绝的tokens到draft datastore"""
        self.add_draft_tokens(tokens)
    
    # ==================== 查询API ====================
    
    def find_candidate_tokens_for_target(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        为大模型查找候选tokens (使用 target_datastore)
        高精度查询
        
        Args:
            input_ids: 当前输入序列
            
        Returns:
            候选tokens序列
        """
        self.stats["total_queries"] += 1
        self.stats["target_used"] += 1
        return self.target_datastore.find_candidate_tokens(input_ids)
    
    def find_candidate_tokens_for_draft(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        为小模型查找候选tokens (使用 draft_datastore)
        高覆盖率查询
        
        Args:
            input_ids: 当前输入序列
            
        Returns:
            候选tokens序列
        """
        self.stats["total_queries"] += 1
        self.stats["draft_used"] += 1
        return self.draft_datastore.find_candidate_tokens(input_ids)
    
    def find_candidate_tokens(self, input_ids: torch.Tensor, for_draft: bool = True) -> torch.Tensor:
        """
        查找候选tokens
        
        Args:
            input_ids: 当前输入序列
            for_draft: True使用draft_datastore, False使用target_datastore
            
        Returns:
            候选tokens序列
        """
        if for_draft:
            return self.find_candidate_tokens_for_draft(input_ids)
        else:
            return self.find_candidate_tokens_for_target(input_ids)
    
    # ==================== 缓存管理 ====================
    
    def clear_dynamic_cache(self):
        """清除两个datastore的动态缓存"""
        self.target_datastore.clear_dynamic()
        self.draft_datastore.clear_dynamic()
    
    def clear_task_caches(self):
        """清除当前任务的所有缓存，保留前缀库"""
        self.clear_dynamic_cache()
    
    def merge_dynamic_to_prefix(self):
        """将动态缓存合并到持久化前缀库"""
        self.target_datastore.merge_dynamic_to_prefix()
        self.draft_datastore.merge_dynamic_to_prefix()
    
    # ==================== 保存/加载 ====================
    
    def save(self, path: str):
        """保存检索库到文件"""
        state = {
            "target_datastore": self.target_datastore.get_state(),
            "draft_datastore": self.draft_datastore.get_state(),
            "max_ngram_size": self.max_ngram_size,
            "num_pred_tokens": self.num_pred_tokens,
            "stats": self.stats.copy(),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Double Datastore RetrievalCache saved to {path}")
    
    def load(self, path: str):
        """从文件加载检索库"""
        if not os.path.exists(path):
            print(f"Cache file {path} not found, starting with empty cache")
            return
            
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.target_datastore.load_state(state["target_datastore"])
        self.draft_datastore.load_state(state["draft_datastore"])
        self.max_ngram_size = state.get("max_ngram_size", self.max_ngram_size)
        self.num_pred_tokens = state.get("num_pred_tokens", self.num_pred_tokens)
        self.stats = state.get("stats", self.stats)
        print(f"Double Datastore RetrievalCache loaded from {path}")
        print(f"  Target datastore prefix size: {len(self.target_datastore.prefix_cache)}")
        print(f"  Draft datastore prefix size: {len(self.draft_datastore.prefix_cache)}")
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        target_stats = self.target_datastore.get_stats()
        draft_stats = self.draft_datastore.get_stats()
        
        return {
            "total_queries": self.stats["total_queries"],
            "target_used": self.stats["target_used"],
            "draft_used": self.stats["draft_used"],
            "target_datastore": target_stats,
            "draft_datastore": draft_stats,
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("Double Datastore Statistics")
        print("=" * 50)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Target datastore used: {stats['target_used']}")
        print(f"Draft datastore used: {stats['draft_used']}")
        
        print("\n--- Target Datastore (D_p) - High Precision ---")
        ts = stats['target_datastore']
        print(f"  Queries: {ts['total_queries']}")
        print(f"  Prefix hits: {ts['prefix_hits']} ({ts.get('prefix_hit_rate', 0):.2%})")
        print(f"  Input hits: {ts['input_hits']} ({ts.get('input_hit_rate', 0):.2%})")
        print(f"  Dynamic hits: {ts['dynamic_hits']} ({ts.get('dynamic_hit_rate', 0):.2%})")
        print(f"  Miss: {ts['miss']} ({ts.get('miss_rate', 0):.2%})")
        print(f"  Prefix size: {ts.get('prefix_size', 0)}, Dynamic size: {ts.get('dynamic_size', 0)}")
        
        print("\n--- Draft Datastore (D_q) - High Coverage ---")
        ds = stats['draft_datastore']
        print(f"  Queries: {ds['total_queries']}")
        print(f"  Prefix hits: {ds['prefix_hits']} ({ds.get('prefix_hit_rate', 0):.2%})")
        print(f"  Input hits: {ds['input_hits']} ({ds.get('input_hit_rate', 0):.2%})")
        print(f"  Dynamic hits: {ds['dynamic_hits']} ({ds.get('dynamic_hit_rate', 0):.2%})")
        print(f"  Miss: {ds['miss']} ({ds.get('miss_rate', 0):.2%})")
        print(f"  Prefix size: {ds.get('prefix_size', 0)}, Dynamic size: {ds.get('dynamic_size', 0)}")
        print("=" * 50 + "\n")


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
            # 使用 add_to_prefix_verified 同时加入两个datastore
            self.cache.add_to_prefix_verified(tokens)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset_samples)} samples")
                
        print(f"Warmup complete.")
        print(f"  Target prefix size: {len(self.cache.target_datastore.prefix_cache)}")
        print(f"  Draft prefix size: {len(self.cache.draft_datastore.prefix_cache)}")
        
    def warmup_from_generation(self, generated_tokens: torch.Tensor, is_verified: bool = True):
        """
        从生成的tokens进行热启动（用于任务运行后积累）
        
        Args:
            generated_tokens: 生成的token序列
            is_verified: 是否是验证后的tokens
        """
        if self.cache is not None:
            if is_verified:
                self.cache.add_to_prefix_verified(generated_tokens)
            else:
                self.cache.add_to_prefix_draft(generated_tokens)
            
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
            
    def on_task_end(self, keep_dynamic: bool = True):
        """
        任务结束时的回调
        
        Args:
            keep_dynamic: 是否将动态缓存合并到前缀库
        """
        if self.cache is not None:
            if keep_dynamic:
                self.cache.merge_dynamic_to_prefix()
            else:
                self.cache.clear_dynamic_cache()


# 全局函数，替代原来的 find_candidate_pred_tokens
@torch.no_grad()
def find_candidate_pred_tokens_with_cache(
    input_ids: torch.Tensor, 
    retrieval_cache: Optional[RetrievalCache] = None,
    max_ngram_size: int = 3, 
    num_pred_tokens: int = 10,
    for_draft: bool = True
) -> torch.Tensor:
    """
    带检索库的候选tokens查找函数
    
    Args:
        input_ids: 输入token序列
        retrieval_cache: 检索库实例，如果为None则回退到原始逻辑
        max_ngram_size: 最大n-gram大小
        num_pred_tokens: 预测tokens数量
        for_draft: True使用draft_datastore, False使用target_datastore
        
    Returns:
        候选tokens序列
    """
    # 如果有检索库，使用检索库查找
    if retrieval_cache is not None:
        return retrieval_cache.find_candidate_tokens(input_ids, for_draft=for_draft)
    
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
