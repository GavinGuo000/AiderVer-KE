from typing import Literal
from models import *
from .process import *
# predefined processing logic for routine extraction tasks
TaskType = Literal["NER", "RE", "EE", "Triple", "Base"]

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
        
        # Verification feedback (from Verifier Agent)
        self.verification_feedback = None
        self.reprocessing_context = None
        self.previous_errors = []
        
        # Probe feedback (from Probe Agent, MRD Section 5.4)
        self.probe_feedback = []
        self.probe_guidance = ""
        
        # Collaborative Memory integration (MRD Section 6)
        self.collaborative_memory_context = ""
        self.needs_reprocessing = False
        self.reprocessing_reason = ""
        self.reprocessing_suggestions = []
        self.verification_result = None
        self.verification_failure_details = None
        
    def set_verification_feedback(self, feedback):
        """Set verifier feedback information"""
        self.verification_feedback = feedback
        if feedback and 'issues' in feedback:
            self.previous_errors.extend(feedback['issues'])
    
    def get_reprocessing_context(self):
        """Get reprocessing context combining verifier and probe feedback"""
        context_parts = []
        
        # Add verifier feedback
        if self.verification_feedback:
            context_parts.append("\n=== Verifier Feedback ===")
            context_parts.append(f"Issues: {self.verification_feedback.get('issues', [])}")
            context_parts.append(f"Suggestions: {self.verification_feedback.get('suggestions', [])}")
            context_parts.append(f"Confidence: {self.verification_feedback.get('confidence_score', 'N/A')}")
        
        # Add probe feedback guidance
        if self.probe_guidance:
            context_parts.append(f"\n=== Probe Agent Feedback ===")
            context_parts.append(self.probe_guidance)
        
        # Add collaborative memory context
        if self.collaborative_memory_context:
            context_parts.append(f"\n=== Historical Error Memory ===")
            context_parts.append(self.collaborative_memory_context)
        
        # Add historical errors
        if self.previous_errors:
            context_parts.append(f"\nPrevious Errors: {self.previous_errors}")
        
        if context_parts:
            return "\n".join(context_parts)
        return ""

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
