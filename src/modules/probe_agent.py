from models import *
from utils import *
from .knowledge_base.case_repository import CaseRepositoryHandler


class ProbeAnalyzer:
    """
    Probe Analyzer class - Responsible for analyzing extraction result defects
    and generating structured quality feedback (MRD Section 5.4)
    
    Key responsibilities:
    1. Analyze extraction result defects based on custom criteria
    2. Locate entity, relation, and structural issues
    3. Generate standardized structured feedback text
    4. Only produce correction guidance, never directly modify extraction output
    """
    
    def __init__(self, llm: BaseEngine):
        """
        Initialize Probe Analyzer
        
        Args:
            llm: Large Language Model engine
        """
        self.llm = llm

    def analyze_extraction_quality(self, instruction="", text="", schema="", result="", task_type=""):
        """
        Analyze extraction result quality and generate structured defect feedback
        
        Args:
            instruction: Task instruction
            text: Original input text
            schema: Output schema definition
            result: Extraction result to analyze
            task_type: Task type (NER/RE/EE/Triple)
            
        Returns:
            dict: Structured quality feedback containing defects and improvement guidance
        """
        # Build task-specific quality criteria
        quality_criteria = self._build_quality_criteria(task_type)
        
        # Convert result to JSON string for analysis
        result_str = json.dumps(result, ensure_ascii=False)
        
        # Build probe analysis prompt
        prompt = f"""You are a strict quality inspector for {task_type} extraction results. 
Analyze the extraction result and identify defects WITHOUT modifying the result directly.

**Task Instruction:**
{instruction}

**Original Text:**
{text}

**Output Schema:**
{schema}

**Extraction Result to Inspect:**
{result_str}

**Quality Criteria:**
{quality_criteria}

**Analysis Instructions:**
1. Identify specific defects in the extraction result (entity errors, relation errors, structural issues)
2. Locate each defect precisely (which entity/relation/field is problematic)
3. Explain WHY each defect is a problem
4. Provide correction guidance (do NOT provide the corrected result directly)

Return JSON format:
{{
    "has_defects": true/false,
    "defect_count": number,
    "defects": [
        {{
            "defect_type": "entity_boundary|entity_type|relation_type|missing_entity|missing_relation|structural",
            "location": "which entity/relation has the issue",
            "description": "what the problem is",
            "severity": "critical|major|minor",
            "correction_guidance": "how to fix it"
        }}
    ],
    "overall_quality_score": 0.0-1.0,
    "summary": "brief overall quality summary"
}}
"""
        
        try:
            response = self.llm.get_chat_response(prompt)
            response = extract_json_dict(response)
            print(f"[PROBE] Quality analysis result: {json.dumps(response, ensure_ascii=False, indent=2)}")
            return response
        except Exception as e:
            print(f"[PROBE] Analysis error: {e}")
            return {
                "has_defects": False,
                "defect_count": 0,
                "defects": [],
                "overall_quality_score": 0.5,
                "summary": "Analysis failed, unable to determine quality",
                "error": str(e)
            }

    def _build_quality_criteria(self, task_type):
        """
        Build task-specific quality criteria
        
        Args:
            task_type: Task type
            
        Returns:
            str: Formatted quality criteria
        """
        criteria_map = {
            "NER": """
1. All entities must exist exactly as written in the original text (entity boundary accuracy)
2. Entity types must match the predefined schema exactly
3. No obvious entities should be missed
4. Entity names should not include surrounding context words
5. Abbreviations and full names should be handled consistently
""",
            "RE": """
1. Head and tail entities must be correct and exist in the text
2. Relation types must match the predefined schema
3. Relation direction (head→tail) must be correct
4. No spurious relations should be introduced
5. All explicit relations in the text should be captured
""",
            "EE": """
1. Event types must match the predefined schema
2. Event triggers must exist in the original text
3. Event arguments must be accurate and complete
4. No spurious events should be introduced
5. Event argument roles must match the schema definition
""",
            "Triple": """
1. Subject and object entities must exist in the text
2. Relation types must match the predefined schema
3. Entity types must be correctly identified
4. Triple completeness - all explicit triples should be captured
5. No hallucinated triples
"""
        }
        return criteria_map.get(task_type, """
1. Extracted content must have textual evidence
2. Types must match the predefined schema
3. No obvious information should be missed
4. No spurious information should be introduced
""")

    def generate_probe_feedback(self, extraction_result, original_text, task_type, schema=""):
        """
        Generate structured probe feedback for downstream agents
        
        Args:
            extraction_result: The extraction result to probe
            original_text: Original input text
            task_type: Task type
            schema: Output schema
            
        Returns:
            dict: Structured probe feedback
        """
        analysis = self.analyze_extraction_quality(
            instruction=f"{task_type} extraction",
            text=original_text,
            schema=schema,
            result=extraction_result,
            task_type=task_type
        )
        
        return analysis


class ProbeAgent:
    """
    Probe Agent - Quality inspection agent for extraction results (MRD Section 5.4)
    
    Core responsibilities:
    1. Analyze extraction result defects based on custom criteria
    2. Locate entity, relation, and structural issues
    3. Generate standardized structured feedback text
    4. Only produce correction guidance, never directly modify extraction output
    5. Sync defect feedback to Collaborative Memory module
    """
    
    def __init__(self, llm: BaseEngine, case_repo: CaseRepositoryHandler):
        """
        Initialize Probe Agent
        
        Args:
            llm: Large Language Model engine
            case_repo: Case repository handler
        """
        self.llm = llm
        self.module = ProbeAnalyzer(llm=llm)
        self.case_repo = case_repo
        self.methods = ["probe_extraction_quality"]

    def probe_extraction_quality(self, data: DataPoint):
        """
        Probe extraction result quality and generate structured feedback
        
        This method analyzes the extraction result, identifies defects,
        and produces correction guidance WITHOUT modifying the result.
        Feedback is synced to the Collaborative Memory for downstream agents.
        
        Args:
            data: DataPoint object containing extraction results
            
        Returns:
            DataPoint: DataPoint with probe feedback attached
        """
        if data.result_list == []:
            print("[PROBE] No extraction results to probe, skipping")
            return data
        
        if not data.chunk_text_list or len(data.chunk_text_list) == 0:
            print("[PROBE] No text chunks to probe, skipping")
            return data
        
        # Initialize probe feedback collection
        all_probe_feedback = []
        
        # Probe each chunk's extraction result
        for idx, (chunk_text, result) in enumerate(
            zip(data.chunk_text_list, data.result_list)
        ):
            # Generate structured quality feedback
            probe_feedback = self.module.generate_probe_feedback(
                extraction_result=result,
                original_text=chunk_text,
                task_type=data.task,
                schema=data.output_schema
            )
            
            all_probe_feedback.append(probe_feedback)
            
            # Print probe summary
            has_defects = probe_feedback.get("has_defects", False)
            defect_count = probe_feedback.get("defect_count", 0)
            quality_score = probe_feedback.get("overall_quality_score", 0.0)
            print(f"[PROBE] Chunk {idx+1}: defects={has_defects}, "
                  f"count={defect_count}, quality={quality_score:.2f}")
        
        # Store probe feedback in data point
        data.probe_feedback = all_probe_feedback
        
        # Convert probe feedback to reprocessing guidance for extraction agent
        data.probe_guidance = self._build_probe_guidance(all_probe_feedback)
        
        # Record trajectory
        function_name = current_function_name()
        data.update_trajectory(function_name, {
            "probe_feedback": all_probe_feedback,
            "total_chunks_probed": len(all_probe_feedback),
            "defects_found": sum(
                1 for fb in all_probe_feedback if fb.get("has_defects", False)
            )
        })
        
        return data
    
    def _build_probe_guidance(self, probe_feedback_list):
        """
        Build structured correction guidance from probe feedback
        This guidance is used by the Extraction Agent in reprocessing
        
        Args:
            probe_feedback_list: List of probe feedback dicts
            
        Returns:
            str: Formatted correction guidance
        """
        guidance_parts = []
        
        for idx, feedback in enumerate(probe_feedback_list):
            if not feedback.get("has_defects", False):
                continue
            
            defects = feedback.get("defects", [])
            if not defects:
                continue
            
            chunk_guidance = f"--- Chunk {idx+1} Defects ---\n"
            for defect in defects:
                defect_type = defect.get("defect_type", "unknown")
                location = defect.get("location", "unknown")
                description = defect.get("description", "")
                correction = defect.get("correction_guidance", "")
                severity = defect.get("severity", "minor")
                
                chunk_guidance += (
                    f"  [{severity.upper()}] {defect_type} at '{location}': "
                    f"{description}. Guidance: {correction}\n"
                )
            guidance_parts.append(chunk_guidance)
        
        if guidance_parts:
            return "\n**Probe Agent Quality Feedback - Please address the following issues:**\n" + "\n".join(guidance_parts)
        return ""
