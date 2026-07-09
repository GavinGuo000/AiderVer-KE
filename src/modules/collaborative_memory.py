"""
Collaborative Memory Module (MRD Section 6)

Persistent storage for:
1. Full-pipeline historical extraction error logs
2. Probe verification feedback information
3. Cross-agent shared error experience pool

Supports system self-evolution: continuously accumulates error experience,
reduces repeated errors without requiring new manual annotations.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class CollaborativeMemory:
    """
    Collaborative Memory - Cross-agent shared memory for error experience (MRD Section 6)
    
    Core capabilities:
    1. Persistent archiving of full-pipeline historical extraction error logs and Probe verification feedback
    2. Historical error case retrieval for Extraction Agent during inference
    3. Shared storage pool across all agents for cross-role error experience flow
    4. Support system self-evolution: continuously accumulate error experience
    """
    
    MEMORY_FILE = os.path.join(os.path.dirname(__file__), "knowledge_base", "collaborative_memory.json")
    MAX_RECORDS = 500  # Maximum records to prevent unbounded growth
    
    def __init__(self):
        """Initialize Collaborative Memory with persistent storage"""
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict:
        """Load memory from persistent storage"""
        if os.path.exists(self.MEMORY_FILE):
            try:
                with open(self.MEMORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[CollaborativeMemory] Failed to load memory: {e}")
        
        # Initialize empty memory structure
        return {
            "error_records": [],
            "probe_feedback_records": [],
            "extraction_sessions": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    def _save_memory(self):
        """Persist memory to storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.MEMORY_FILE), exist_ok=True)
            with open(self.MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[CollaborativeMemory] Failed to save memory: {e}")
    
    def record_extraction_errors(self, task_type: str, text: str, 
                                  extraction_result, probe_feedback: Dict,
                                  verification_result: Dict, session_id: str = ""):
        """
        Record extraction errors and probe/verification feedback for a session
        
        Args:
            task_type: Task type (NER/RE/EE/Triple)
            text: Original input text (truncated for storage)
            extraction_result: The extraction result that was produced
            probe_feedback: Probe Agent's defect analysis
            verification_result: Verifier Agent's verification result
            session_id: Optional session identifier
        """
        # Truncate text to prevent excessive storage
        truncated_text = text[:500] + "..." if len(text) > 500 else text
        
        error_record = {
            "session_id": session_id or datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "text_preview": truncated_text,
            "extraction_result": extraction_result,
            "probe_defects": probe_feedback.get("defects", []) if isinstance(probe_feedback, dict) else [],
            "probe_quality_score": probe_feedback.get("overall_quality_score", 0.0) if isinstance(probe_feedback, dict) else 0.0,
            "verification_passed": verification_result.get("is_consistent", False) if isinstance(verification_result, dict) else False,
            "verification_issues": verification_result.get("issues", []) if isinstance(verification_result, dict) else [],
            "verification_suggestions": verification_result.get("suggestions", []) if isinstance(verification_result, dict) else [],
            "error_types": self._classify_errors(probe_feedback, verification_result)
        }
        
        self.memory["error_records"].append(error_record)
        
        # Enforce max records limit
        if len(self.memory["error_records"]) > self.MAX_RECORDS:
            self.memory["error_records"] = self.memory["error_records"][-self.MAX_RECORDS:]
        
        self._save_memory()
        print(f"[CollaborativeMemory] Error record saved for {task_type} task")
    
    def record_probe_feedback(self, task_type: str, probe_feedback_list: List[Dict],
                              session_id: str = ""):
        """
        Record Probe Agent feedback for cross-agent knowledge sharing
        
        Args:
            task_type: Task type
            probe_feedback_list: List of probe feedback from each chunk
            session_id: Optional session identifier
        """
        feedback_record = {
            "session_id": session_id or datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "feedback_count": len(probe_feedback_list),
            "total_defects": sum(
                fb.get("defect_count", 0) for fb in probe_feedback_list if isinstance(fb, dict)
            ),
            "average_quality": (
                sum(fb.get("overall_quality_score", 0.0) for fb in probe_feedback_list if isinstance(fb, dict))
                / max(len(probe_feedback_list), 1)
            ),
            "defect_patterns": self._extract_defect_patterns(probe_feedback_list)
        }
        
        self.memory["probe_feedback_records"].append(feedback_record)
        
        # Enforce limit
        if len(self.memory["probe_feedback_records"]) > self.MAX_RECORDS:
            self.memory["probe_feedback_records"] = self.memory["probe_feedback_records"][-self.MAX_RECORDS:]
        
        self._save_memory()
    
    def record_extraction_session(self, task_type: str, text: str,
                                   extraction_result, final_result,
                                   reprocessing_count: int = 0,
                                   session_id: str = ""):
        """
        Record a complete extraction session summary
        
        Args:
            task_type: Task type
            text: Original input text
            extraction_result: Initial extraction result
            final_result: Final output after all processing
            reprocessing_count: Number of reprocessing attempts
            session_id: Optional session identifier
        """
        session_record = {
            "session_id": session_id or datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "initial_result": extraction_result,
            "final_result": final_result,
            "reprocessing_count": reprocessing_count,
            "result_changed": extraction_result != final_result
        }
        
        self.memory["extraction_sessions"].append(session_record)
        
        if len(self.memory["extraction_sessions"]) > self.MAX_RECORDS:
            self.memory["extraction_sessions"] = self.memory["extraction_sessions"][-self.MAX_RECORDS:]
        
        self._save_memory()
    
    def retrieve_similar_errors(self, task_type: str, text: str = "",
                                 error_types: List[str] = None,
                                 top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar historical error cases for Extraction Agent reference
        
        Args:
            task_type: Task type to filter by
            text: Current input text for similarity matching
            error_types: Optional list of error types to filter by
            top_k: Maximum number of records to return
            
        Returns:
            list: List of similar historical error records
        """
        # Filter by task type
        task_records = [
            r for r in self.memory["error_records"]
            if r.get("task_type") == task_type
        ]
        
        if not task_records:
            return []
        
        # Filter by error types if specified
        if error_types:
            task_records = [
                r for r in task_records
                if any(et in r.get("error_types", []) for et in error_types)
            ]
        
        # Simple text similarity scoring
        if text:
            text_words = set(text.lower().split())
            scored_records = []
            for record in task_records:
                record_text = record.get("text_preview", "").lower()
                record_words = set(record_text.split())
                # Jaccard similarity
                intersection = text_words.intersection(record_words)
                union = text_words.union(record_words)
                similarity = len(intersection) / max(len(union), 1)
                scored_records.append((similarity, record))
            
            # Sort by similarity and return top_k
            scored_records.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored_records[:top_k]]
        
        # If no text provided, return most recent records
        return task_records[-top_k:]
    
    def get_error_summary_for_extraction(self, task_type: str, text: str = "") -> str:
        """
        Generate a summary of historical errors for Extraction Agent to reference
        This is the main interface for Extraction Agent to leverage collaborative memory
        
        Args:
            task_type: Task type
            text: Current input text
            
        Returns:
            str: Formatted error summary for extraction guidance
        """
        similar_errors = self.retrieve_similar_errors(task_type, text, top_k=3)
        
        if not similar_errors:
            return ""
        
        summary_parts = ["**Historical Error Patterns from Collaborative Memory:**"]
        
        for i, error in enumerate(similar_errors, 1):
            error_types = error.get("error_types", [])
            issues = error.get("verification_issues", [])
            suggestions = error.get("verification_suggestions", [])
            quality = error.get("probe_quality_score", 0.0)
            
            summary_parts.append(
                f"\n{i}. [Quality: {quality:.2f}] Error types: {error_types}\n"
                f"   Issues: {issues[:3]}\n"
                f"   Lessons: {suggestions[:2]}"
            )
        
        summary_parts.append(
            "\nPlease avoid these known error patterns in your extraction."
        )
        
        return "\n".join(summary_parts)
    
    def _classify_errors(self, probe_feedback: Dict, verification_result: Dict) -> List[str]:
        """Classify error types from probe and verification results"""
        error_types = []
        
        # From probe feedback
        if isinstance(probe_feedback, dict):
            for defect in probe_feedback.get("defects", []):
                defect_type = defect.get("defect_type", "")
                if defect_type and defect_type not in error_types:
                    error_types.append(defect_type)
        
        # From verification result
        if isinstance(verification_result, dict):
            for issue in verification_result.get("issues", []):
                issue_lower = str(issue).lower()
                if "entity" in issue_lower and "entity_error" not in error_types:
                    error_types.append("entity_error")
                elif "type" in issue_lower and "type_error" not in error_types:
                    error_types.append("type_error")
                elif "miss" in issue_lower and "completeness_error" not in error_types:
                    error_types.append("completeness_error")
                elif "relation" in issue_lower and "relation_error" not in error_types:
                    error_types.append("relation_error")
        
        return error_types
    
    def _extract_defect_patterns(self, probe_feedback_list: List[Dict]) -> List[Dict]:
        """Extract common defect patterns from probe feedback"""
        pattern_counts = {}
        
        for feedback in probe_feedback_list:
            if not isinstance(feedback, dict):
                continue
            for defect in feedback.get("defects", []):
                defect_type = defect.get("defect_type", "unknown")
                if defect_type not in pattern_counts:
                    pattern_counts[defect_type] = 0
                pattern_counts[defect_type] += 1
        
        # Return sorted patterns
        return [
            {"pattern": k, "count": v}
            for k, v in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "total_error_records": len(self.memory.get("error_records", [])),
            "total_probe_records": len(self.memory.get("probe_feedback_records", [])),
            "total_sessions": len(self.memory.get("extraction_sessions", [])),
            "memory_file": self.MEMORY_FILE
        }
