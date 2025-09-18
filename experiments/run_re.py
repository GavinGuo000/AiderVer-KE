import sys
sys.path.append("./src")
from models import *
from dataset_def import *
data_dir = "./data/datasets/NYT11/"
model = Gpt5(
    model_name_or_path="gpt-5",
    base_url="https://api.kwwai.top/v1"
)
dataset = REDataset(name="NYT11", data_dir=data_dir)
f1_score = dataset.evaluate(llm=model, mode="ablation_no_aider")
print("f1_score: ", f1_score)

