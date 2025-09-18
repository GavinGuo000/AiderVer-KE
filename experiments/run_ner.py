import sys
sys.path.append("./src")
from models import *
from dataset_def import *
name = "crossner-"
data_dir = "./data/datasets/CrossNER/"

model = Gpt5(
    model_name_or_path="gpt-5",
    base_url="https://api.kwwai.top/v1"
)

tasklist = ["ai", "literature", "music", "politics", "science"]
for task in tasklist:
    task_name = name + task
    task_data_dir = data_dir + task
    dataset = NERDataset(name=task_name, data_dir=task_data_dir)
    mode = "enhanced"
    f1_score = dataset.evaluate(llm=model, mode=mode)
    print(f"Task: {task_name}, f1_score: {f1_score}")
