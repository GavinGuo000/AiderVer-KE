from typing import Literal
from models import *
from .process import *
# predefined processing logic for routine extraction tasks
TaskType = Literal["NER", "RE", "EE", "Base"]

class DataPoint:
    def __init__(self,
                 task: TaskType = "Base",
                 instruction: str = "",
                 text: str = "",
                 output_schema: str = "",
                 constraint: str = "",
                 use_file: bool = False,
                 file_path: str = "",
                 truth: str = ""):
        """
        Initialize a DataPoint instance.
        """
        # task information
        self.task = task
        self.instruction = instruction
        self.text = text
        self.output_schema = output_schema
        self.constraint = constraint
        self.use_file = use_file
        self.file_path = file_path
        self.truth = extract_json_dict(truth)
        # temp storage
        self.print_schema = ""
        self.distilled_text = ""
        self.chunk_text_list = []
        # result feedback
        self.result_list = []
        self.result_trajectory = {}
        self.pred = ""
        
        # 新增验证反馈相关属性
        self.verification_feedback = None  # 验证器反馈信息
        self.reprocessing_context = None   # 重新处理上下文
        self.previous_errors = []          # 历史错误记录
        
    def set_verification_feedback(self, feedback):
        """设置验证器反馈信息"""
        self.verification_feedback = feedback
        if feedback and 'issues' in feedback:
            self.previous_errors.extend(feedback['issues'])
    
    def get_reprocessing_context(self):
        """获取重新处理上下文"""
        if not self.verification_feedback:
            return ""
        
        context = "\n=== 重新处理上下文信息 ===\n"
        context += f"上次验证发现的问题: {self.verification_feedback.get('issues', [])}\n"
        context += f"改进建议: {self.verification_feedback.get('suggestions', [])}\n"
        context += f"置信度评分: {self.verification_feedback.get('confidence_score', 'N/A')}\n"
        
        if self.previous_errors:
            context += f"历史错误记录: {self.previous_errors}\n"
        
        context += "请根据以上反馈信息进行针对性改进。\n"
        return context

    def set_constraint(self, constraint):
        self.constraint = constraint

    def set_schema(self, output_schema):
        self.output_schema = output_schema

    def set_pred(self, pred):
        self.pred = pred

    def set_result_list(self, result_list):
        self.result_list = result_list

    def set_distilled_text(self, distilled_text):
        self.distilled_text = distilled_text

    def update_trajectory(self, function, result):
        if function not in self.result_trajectory:
            self.result_trajectory.update({function: result})

    def get_result_trajectory(self):
        return {"instruction": self.instruction, "text": self.text, "constraint": self.constraint,  "trajectory": self.result_trajectory, "pred": self.pred}
