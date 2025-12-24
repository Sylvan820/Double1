# # qwen3 14b psd
# CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py --eval_mode para_sd --gamma 2 -n 1  -e H_PSD_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_qwen14be.log
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 2 -n 1  -e G_PSD_qwe14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_qwen14bg.log
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 2 -n 1  -e M_PSD_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_qwen14bm.log

# # deepseek psd
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_deepseek --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_deepseeke.log
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 4 -n 1  -e G_PSD_deepseek --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_deepseekg.log
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 4 -n 1  -e M_PSD_deepseek --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_deepseekn.log

# # # # qwen3 14b speculative decoding
# # CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 1 benchmark/eval_humaneval_qwen3.py --eval_mode sd --gamma 5 -n 1  -e H_SD_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/qwen14be.log
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode sd --gamma 5 -n 1  -e G_SD_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/qwen14bg.log
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 1 benchmark/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e M_SD_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/qwen14bm.log


# qwen3 14b Dou
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py --eval_mode para_sd --gamma 2 -n 1  -e H_Dou_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_qwen14be.log
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 2 -n 1  -e G_Dou_qwe14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_qwen14bg.log
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 2 -n 1  -e M_Dou_qwen14 --draft_model qwen3-0.6b --target_model qwen3-14b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_qwen14bm.log

# # deepseek Dou
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_Dou_deepseek --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_deepseeke.log
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 4 -n 1  -e G_Dou_deepseek --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_deepseekg.log
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 4 -n 1  -e M_Dou_deepseek --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_deepseekm.log


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_Dou_deepseek-ablation1 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_deepseeke1.log
