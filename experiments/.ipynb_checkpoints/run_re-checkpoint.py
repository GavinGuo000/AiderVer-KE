import sys
sys.path.append("./src")
from models import *
from dataset_def import *
data_dir = "./data/datasets/NYT11/"
# model = DeepSeek(model_name_or_path="deepseek-chat", api_key="sk-7f1a9cd82f29484db6dbc778aa748344", base_url="https://api.deepseek.com")
# model = Qwen3(
#     model_name_or_path="Qwen/Qwen3-8B"  # 修正参数名称
# )
# model = Gpt5(
#     model_name_or_path="gpt-5",  # 或其他 Gemini 模型名称
#     base_url="https://api.kwwai.top/v1"  # 可选，默认值
# )
model = Qwen25(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct"  # 修正参数名称
)
dataset = REDataset(name="NYT11", data_dir=data_dir)
f1_score = dataset.evaluate(llm=model, mode="ablation_no_aider")
print("f1_score: ", f1_score)

