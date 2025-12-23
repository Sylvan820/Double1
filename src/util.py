import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np


def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def model_zoo(args):
    vocab_size = {
        "tinyllamacode-1.1b": 32000,
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
        "deepseek-1.3b": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-33b": 32256,
        "vicuna-68m": 32000,
        "vicuna-7b": 32000,
        "llama-68m": 32000,
        "llama-160m": 32000,
        "llama-7b": 32000,
        "llama-13b": 32000,
        "llama-30b": 32000,
        "qwen3-0.6b": 151936,
        "qwen3-1.7b": 151936,
        "qwen3-4b": 151936,
        "qwen3-8b": 151936,
        "qwen3-14b": 151936,
        "qwen3-32b": 151936,
        "qwen2.5-3b": 151936,
        "qwen2.5-7b": 152064,
        "qwen2.5-14b": 152064,
        "qwen2.5-72b": 152064,
        "llama-3.1-8b": 128256,
        "llama-3.3-70b": 128256,
    }

    zoo = {
        "tinyllamacode-1.1b": "/remote-home/security_shenyuhao/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5",
        "codellama-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed",
        "codellama-34b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d",
        "codellama-70b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-70b-Instruct-hf/snapshots/397cae981dffaf5d5c9c90e89a0a75a850528b70",
        "llama-2-7b": "/root/autodl-user/cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
        "llama-2-70b": "/root/autodl-user/cache/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080",
        "deepseek-1.3b": "/root/autodl-user/cache/hub/models--deepseek-ai--deepseek-coder-1.3b-instruct/snapshots/e063262dac8366fc1f28a4da0ff3c50ea66259ca",
        "deepseek-6.7b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-instruct/snapshots/e5d64addd26a6a1db0f9b863abf6ee3141936807",
        "deepseek-33b": "/root/autodl-user/cache/hub/models--deepseek-ai--deepseek-coder-33b-instruct/snapshots/61dc97b922b13995e7f83b7c8397701dbf9cfd4c",
        "vicuna-68m": "/remote-home/security_shenyuhao/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109",
        "vicuna-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
        "llama-68m": "/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-68m/snapshots/964a5d77df908b69f8d6476fb70e940425b04cb5",
        "llama-160m": "/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-160m/snapshots/aca9b687d1425f863dcf5de9a4c96e3fe36266dd",
        "llama-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",
        "llama-13b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba",
        "llama-30b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-30b/snapshots/2b1edcdb3c7ced7bce6c1aa75c94545777c3118b",
        "qwen3-0.6b": "/root/autodl-user/cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "qwen3-1.7b": "/root/autodl-user/cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "qwen3-4b": "/root/autodl-user/cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "qwen3-8b": "/root/autodl-user/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "qwen3-14b": "/root/autodl-user/cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18",
        "qwen3-32b": "/root/autodl-user/cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137",
        "qwen2.5-3b": "/root/autodl-user/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "qwen2.5-7b": "/root/autodl-user/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "qwen2.5-14b": "/root/autodl-user/cache/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "qwen2.5-72b": "/root/autodl-user/cache/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31",
        "llama-3.1-8b": "/root/autodl-user/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        "llama-3.3-70b": "/root/autodl-user/cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    }

    args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    args.target_model = zoo[args.target_model]


def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--data_path', type=str, default="/root/autodl-user/Retrirval2/data")
    # parser.add_argument('--data_path', type=str,
    #                     default="/remote-home/security_shenyuhao/huggingface/hub/datasets--openai--gsm8k")
    parser.add_argument('--draft_model', type=str, default="vicuna-68m")
    parser.add_argument('--target_model', type=str, default="vicuna-7b")

    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="small",
                        choices=["small", "large", "sd", "para_sd", "para_sd_wo_1", "para_sd_wo_2"], help='eval mode.')
    parser.add_argument('--num_samples_per_task', '-n', type=int, default=1,
                        help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234,
                        help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0.2, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    
    # 检索库相关参数
    parser.add_argument('--use_retrieval_cache', action='store_true', help='Enable retrieval cache for PLD')
    parser.add_argument('--retrieval_cache_dir', type=str, default='./retrieval_cache', help='Directory for retrieval cache')
    parser.add_argument('--retrieval_cache_name', type=str, default='default', help='Name of the retrieval cache to load')
    parser.add_argument('--warmup_task_count', type=int, default=0, help='Number of warmup tasks for retrieval cache')
    parser.add_argument('--pld_max_ngram_size', type=int, default=3, help='Max n-gram size for PLD retrieval')
    parser.add_argument('--pld_num_pred_tokens', type=int, default=10, help='Number of predicted tokens for PLD')
    parser.add_argument('--save_retrieval_cache', action='store_true', help='Save retrieval cache after evaluation')
    
    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    return args


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum