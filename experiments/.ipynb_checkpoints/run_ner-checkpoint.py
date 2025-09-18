import sys
sys.path.append("./src")
from models import *
from dataset_def import *
name = "crossner-"
data_dir = "./data/datasets/CrossNER/"
# model = DeepSeek(model_name_or_path="deepseek-chat", api_key="sk-7f1a9cd82f29484db6dbc778aa748344", base_url="https://api.deepseek.com")
# 使用 LoRA 微调模型
# model = QwenLoRA(
#     base_model_path="Qwen/Qwen3-8B",  # 基础模型
#     lora_path="/root/autodl-fs/qwen3-8b/crossner-nyt11/lora/sft"  # LoRA 权重路径
# )
# model = Qwen3(
#     model_name_or_path="Qwen/Qwen3-8B"  # 修正参数名称
# )

model = Gpt5(
    model_name_or_path="gpt-5",  # 或其他 Gemini 模型名称
    base_url="https://api.kwwai.top/v1"  # 可选，默认值
)

# tasklist = ["literature", "music", "politics", "science"]
tasklist = ["ai"]
for task in tasklist:
    task_name = name + task
    task_data_dir = data_dir + task
    dataset = NERDataset(name=task_name, data_dir=task_data_dir)
    mode = "ablation_no_aider"
    # mode = "enhanced"
    # mode = "ablation_no_verifier"  # 新增消融实验模式：移除verifier_agent
    f1_score = dataset.evaluate(llm=model, mode=mode)
    print(f"Task: {task_name}, f1_score: {f1_score}")
