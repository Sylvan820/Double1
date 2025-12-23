import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

# from transformers import pipeline
#
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="codellama/CodeLlama-34b-Instruct-hf")
# pipe(messages)
# Use a pipeline as a high-level helper
# from transformers import pipeline
#
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="deepseek-ai/deepseek-coder-6.7b-instruct")
# pipe(messages)


# # Use a pipeline as a high-level helper
# from transformers import pipeline
#
# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama_v1.1_math_code")

from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")
pipe(messages)