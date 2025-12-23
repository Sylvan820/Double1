# 方案 1：启用检索库 + 热启动（推荐

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py \
#     --eval_mode para_sd \
#     --gamma 2 \
#     -n 1 \
#     -e H_DOU_qwen314b_with_cache \
#     --draft_model qwen3-0.6b \
#     --target_model qwen3-14b \
#     --max_tokens 1024 \
#     --temp 0 \
#     --use_retrieval_cache \
#     --warmup_task_count 0 \
#     --pld_max_ngram_size 3 \
#     --pld_num_pred_tokens 10 \
#     --retrieval_cache_name qwen3_humaneval_after_eval

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py \
    --eval_mode para_sd \
    --gamma 2 \
    -n 1 \
    -e H_DOU_qwen314b_with_cache \
    --draft_model qwen3-0.6b \
    --target_model qwen3-14b \
    --max_tokens 1024 \
    --temp 0 \
    --use_retrieval_cache \
    --warmup_task_count 10 \
    --pld_max_ngram_size 3 \
    --pld_num_pred_tokens 10 \
    --save_retrieval_cache \
    --retrieval_cache_name qwen3_humaneval

# # 方案 2：使用已保存的检索库缓存
#     CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py \
#     --eval_mode para_sd \
#     --gamma 2 \
#     -n 1 \
#     -e H_DOU_qwen314b_with_cache \
#     --draft_model qwen3-0.6b \
#     --target_model qwen3-14b \
#     --max_tokens 1024 \
#     --temp 0 \
#     --use_retrieval_cache \
#     --retrieval_cache_name qwen3_humaneval \
#     --warmup_task_count 0