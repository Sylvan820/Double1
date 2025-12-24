# # qwen3 32b psd
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_qwen32 --draft_model qwen3-0.6b --target_model qwen3-32b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_qwen32be.log
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 3 -n 1  -e G_PSD_qwe32 --draft_model qwen3-0.6b --target_model qwen3-32b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_qwen32bg.log
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 3 -n 1  -e M_PSD_qwen32 --draft_model qwen3-0.6b --target_model qwen3-32b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_qwen32bm.log

# # # llama3 psd
# CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 2 benchmark/eval_humaneval_llama3.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_llama3 --draft_model llama-3.1-8b --target_model llama-3.3-70b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_llama3e.log
# CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 5 -n 1  -e G_PSD_llama3 --draft_model llama-3.1-8b --target_model llama-3.3-70b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_llama3g.log
# CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 5 -n 1  -e M_PSD_llama3 --draft_model llama-3.1-8b --target_model llama-3.3-70b --max_tokens 1024 --temp 0 2>&1 | tee logs/PSD_llama3m.log

# qwen3 32b Dou
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval_qwen3.py --eval_mode para_sd --gamma 3 -n 1  -e H_Dou_qwen32 --draft_model qwen3-0.6b --target_model qwen3-32b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_qwen32be.log
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 3 -n 1  -e G_Dou_qwe32 --draft_model qwen3-0.6b --target_model qwen3-32b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_qwen32bg.log
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 3 -n 1  -e M_Dou_qwen32 --draft_model qwen3-0.6b --target_model qwen3-32b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_qwen32bm.log

# # llama3 Dou
# CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 2 benchmark/eval_humaneval_llama3.py --eval_mode para_sd --gamma 5 -n 1  -e H_Dou_llama3 --draft_model llama-3.1-8b --target_model llama-3.3-70b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_llama3e.log
# CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 5 -n 1  -e G_Dou_llama3 --draft_model llama-3.1-8b --target_model llama-3.3-70b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_llama3g.log
# CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 5 -n 1  -e M_Dou_llama3 --draft_model llama-3.1-8b --target_model llama-3.3-70b --max_tokens 1024 --temp 0 2>&1 | tee logs/Dou_llama3m.log

# llama2 70b speculative decoding
