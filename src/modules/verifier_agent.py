from models import *
from utils import *
from .knowledge_base.case_repository import CaseRepositoryHandler
import json
import re

class ResultVerifier:
    """Result Verifier Class - Responsible for verifying consistency and accuracy of information extraction results
    """
    
    def __init__(self, llm: BaseEngine):
        """Initialize the result verifier
        
        Args:
            llm (BaseEngine): Large language model engine
        """
        self.llm = llm
    
    def _basic_consistency_check(self, extraction_result, original_text):
        """Rule-based basic consistency check - Support multiple formats
        
        Args:
            extraction_result: Extraction result, can be string, list or dict
            original_text (str): Original text
            
        Returns:
            tuple: (whether check passed, check reason)
        """
        # Check if extraction result is None
        if extraction_result is None:
            return False, "Extraction result is None"
        
        # Handle different input formats
        entities_to_check = []
        
        # If string format, try to parse as JSON
        if isinstance(extraction_result, str):
            try:
                parsed_result = json.loads(extraction_result)
                extraction_result = parsed_result
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[ERROR] JSON parsing failed: {type(e).__name__}: {e}")
                if extraction_result.strip():
                    if extraction_result.strip().lower() in original_text.lower():
                        return True, f"String entity '{extraction_result.strip()}' found in text"
                    else:
                        return False, f"String entity '{extraction_result.strip()}' not found in text"
                else:
                    return False, "Empty string extraction result"
        
        # Handle parsed results
        if isinstance(extraction_result, list):
            if len(extraction_result) == 0:
                return False, "Extraction result list is empty"
            entities_to_check = extraction_result
        elif isinstance(extraction_result, dict):
            if 'entity_list' in extraction_result:
                entities_to_check = extraction_result['entity_list']
            elif 'name' in extraction_result and 'type' in extraction_result:
                entities_to_check = [extraction_result]
            else:
                for key, value in extraction_result.items():
                    if isinstance(value, list) and len(value) > 0:
                        if all(isinstance(item, dict) and 'name' in item for item in value[:3]):
                            entities_to_check = value
                            break
        else:
            return False, f"Unsupported extraction result format: {type(extraction_result)}"
        
        if not entities_to_check:
            return False, "No entities found in extraction result"
        
        # Verify if entities exist in original text
        valid_entities_found = 0
        total_entities = len(entities_to_check)
        
        for entity in entities_to_check:
            if isinstance(entity, dict) and 'name' in entity:
                entity_name = str(entity['name']).strip()
                if entity_name and self._entity_exists_in_text(entity_name, original_text):
                    valid_entities_found += 1
            elif isinstance(entity, str):
                entity_name = entity.strip()
                if entity_name and entity_name.lower() in original_text.lower():
                    valid_entities_found += 1
        
        if valid_entities_found > 0:
            success_rate = valid_entities_found / total_entities
            return True, f"Basic check passed - found {valid_entities_found}/{total_entities} valid entities (success rate: {success_rate:.2f})"
        else:
            return False, f"No valid entities found in text (checked {total_entities} entities)"
    
    def _entity_exists_in_text(self, entity_name, original_text):
        """Enhanced entity existence check with boundary matching
        """
        # Add boundary exact matching check
        if f" {entity_name} " in f" {original_text} ":
            return True
        # Add regex matching
        return bool(re.search(rf"\b{re.escape(entity_name)}\b", original_text, re.IGNORECASE))
    
    def _enhanced_entity_type_check(self, extraction_result, task_type, constraint):
        """Enhanced entity type checking
        """
        if not isinstance(extraction_result, (list, dict)):
            return True, "No type checking needed for non-structured results"
        
        entities = []
        if isinstance(extraction_result, list):
            entities = extraction_result
        elif isinstance(extraction_result, dict) and 'entity_list' in extraction_result:
            entities = extraction_result['entity_list']
        
        type_issues = []
        for entity in entities:
            if isinstance(entity, dict) and 'type' in entity:
                entity_type = entity.get('type', '').strip()
                if constraint and entity_type not in constraint:
                    type_issues.append(f"Invalid entity type: {entity_type}")
        
        if type_issues:
            return False, f"Entity type validation failed: {'; '.join(type_issues[:3])}"
        return True, "Entity types are valid"
    
    def _check_entity_completeness(self, extraction_result, original_text, task_type):
        """Check entity completeness
        """
        # Simple completeness check - can be extended as needed
        if isinstance(extraction_result, (list, dict)):
            entity_count = 0
            if isinstance(extraction_result, list):
                entity_count = len(extraction_result)
            elif isinstance(extraction_result, dict) and 'entity_list' in extraction_result:
                entity_count = len(extraction_result['entity_list'])
            
            # If text is long but entities are few, there might be omissions
            text_length = len(original_text.split())
            if text_length > 50 and entity_count == 0:
                return False, "Potentially missed entities in long text"
            elif text_length > 20 and entity_count == 0:
                return False, "No entities found in substantial text"
        
        return True, "Entity completeness check passed"
    
    def _generate_enhanced_verification_prompt(self, original_text, extraction_result, task_type, local_issues):
        """Generate enhanced verification prompt
        """
        task_specific_info = {
            "NER": {
                "title": "named entity recognition",
                "standards": [
                    "All identified entities must exist in the original text",
                    "Entity types must match the given constraints exactly",
                    "Entity boundaries must be precise (no partial words)",
                    "No obvious entities should be missed",
                    "Entity types should be semantically appropriate"
                ]
            },
            "RE": {
                "title": "relation extraction",
                "standards": [
                    "Relation triplets have textual evidence",
                    "Head and tail entities are correct",
                    "Relation types are accurate"
                ]
            },
            "EE": {
                "title": "event extraction",
                "standards": [
                    "Events have textual evidence",
                    "Event types are correct",
                    "Event arguments are accurate"
                ]
            }
        }
        
        task_info = task_specific_info.get(task_type, {
            "title": "information extraction",
            "standards": ["Extracted content must have textual evidence", "Extraction must be accurate and complete"]
        })
        
        standards_text = "\n".join([f"{i+1}. {std}" for i, std in enumerate(task_info["standards"])])
        
        local_issues_text = ""
        if local_issues:
            local_issues_text = f"\n\n**Pre-identified Issues:**\n" + "\n".join([f"- {issue}" for issue in local_issues])
        
        return f"""
You are a strict {task_info['title']} verification expert. Please carefully verify the extraction results with high standards:

**Original Text:**
{original_text}

**Extraction Results:**
{json.dumps(extraction_result, ensure_ascii=False, indent=2)}

**Strict Verification Standards:**
{standards_text}
{local_issues_text}

**Verification Instructions:**
- Be strict in your evaluation
- If any entity type is incorrect, mark as inconsistent
- If obvious entities are missed, reduce confidence significantly
- Only mark as consistent if extraction is both accurate and reasonably complete

Return JSON format:
{{
    "is_consistent": true/false,
    "confidence_score": 0.0-1.0,
    "issues": ["specific issue descriptions"],
    "suggestions": ["improvement suggestions"]
}}
"""
    
    def verify_consistency(self, original_text, extraction_result, task_type, instruction="", constraint=""):
        """Improved verification logic
        """
        # Basic consistency check
        basic_check_passed, basic_check_reason = self._basic_consistency_check(extraction_result, original_text)
        
        # Entity type validation
        type_check_passed, type_check_reason = self._enhanced_entity_type_check(extraction_result, task_type, constraint)
        
        # Entity completeness check
        completeness_check_passed, completeness_reason = self._check_entity_completeness(extraction_result, original_text, task_type)
        
        # Collect all issues
        issues = []
        if not basic_check_passed:
            issues.append(basic_check_reason)
        if not type_check_passed:
            issues.append(type_check_reason)
        if not completeness_check_passed:
            issues.append(completeness_reason)
        
        # Calculate comprehensive confidence score
        base_confidence = 0.9 if basic_check_passed else 0.1
        type_penalty = 0.0 if type_check_passed else 0.3
        completeness_penalty = 0.0 if completeness_check_passed else 0.2
        
        confidence_score = max(0.0, base_confidence - type_penalty - completeness_penalty)
        
        # If serious issues exist, return directly
        if not basic_check_passed or not type_check_passed:
            return {
                "is_consistent": False,
                "confidence_score": confidence_score,
                "issues": issues,
                "suggestions": ["Review extraction model and constraints", "Check entity type definitions"]
            }
        
        # Call LLM for deep verification
        try:
            verification_prompt = self._generate_enhanced_verification_prompt(original_text, extraction_result, task_type, issues)
            response = self.llm.get_chat_response(verification_prompt)
            verification_result = extract_json_dict(response)
            
            # Merge local check results
            if issues:
                verification_result.setdefault("issues", []).extend(issues)
            
            # Adjust confidence score
            llm_confidence = verification_result.get("confidence_score", 0.8)
            final_confidence = min(confidence_score, llm_confidence)
            verification_result["confidence_score"] = final_confidence
            
            return verification_result
            
        except Exception as e:
            print(f"VerifierAgent: LLM verification error: {str(e)}")
            return {
                "is_consistent": len(issues) == 0,
                "confidence_score": confidence_score,
                "issues": issues,
                "suggestions": ["Check LLM service status"] + (["Review extraction quality"] if issues else [])
            }

class VerifierAgent:
    """Verifier Agent Class - Responsible for coordinating and managing the entire verification process
    """
    
    def __init__(self, llm: BaseEngine, case_repo: CaseRepositoryHandler):
        """Initialize the verifier agent
        
        Args:
            llm (BaseEngine): Large language model engine
            case_repo (CaseRepositoryHandler): Case repository handler
        """
        self.llm = llm
        self.verifier = ResultVerifier(llm)  # Result verifier instance
        self.case_repo = case_repo  # Case repository handler
        self.methods = ["verify_extraction_result"]  # Available methods list
        self.confidence_threshold = 0.6  # Confidence threshold
        self.strict_confidence_threshold = 0.85  # Strict mode threshold
    
    def verify_extraction_result(self, data: DataPoint):
        """Verify extraction results using improved verification logic
        
        Args:
            data (DataPoint): Data point object containing extraction results to be verified
            
        Returns:
            DataPoint: Updated data point object
        """
        
        # Execute consistency verification
        verification_result = self.verifier.verify_consistency(
            data.text, data.pred, data.task, data.instruction
        )
        
        # Record verification results to data point object
        data.verification_result = verification_result
        
        # Get components of verification results
        is_consistent = verification_result.get("is_consistent", True)
        confidence_score = verification_result.get("confidence_score", 0.8)
        issues = verification_result.get("issues", [])
        suggestions = verification_result.get("suggestions", [])
        
        print(f"VerifierAgent: Consistency check - {is_consistent}, confidence: {confidence_score:.2f}")
        
        # Determine if reprocessing is needed
        needs_reprocessing = self._should_reprocess(is_consistent, confidence_score, issues)
        
        # Set reprocessing flag
        data.needs_reprocessing = needs_reprocessing
        
        # If reprocessing is needed, record relevant information
        if needs_reprocessing:
            data.reprocessing_reason = issues
            data.reprocessing_suggestions = suggestions
            self._record_failure_details(data, is_consistent, confidence_score, issues, suggestions)
        
        # Update processing trajectory
        function_name = current_function_name()
        data.update_trajectory(function_name, {
            "verification_passed": not needs_reprocessing,
            "is_consistent": is_consistent,
            "confidence_score": confidence_score,
            "needs_reprocessing": needs_reprocessing,
            "issues": issues,
            "suggestions": suggestions
        })
        
        return data
    
    def _should_reprocess(self, is_consistent, confidence_score, issues):
        """Improved reprocessing decision logic
        """
        # Critical issue keywords
        critical_keywords = ["empty", "format error", "completely wrong", "Invalid entity types", "Potentially missed"]
        
        # Case 1: Completely inconsistent
        if not is_consistent:
            print("VerifierAgent: Inconsistent results detected, marking for reprocessing")
            return True
        
        # Case 2: Confidence too low
        if confidence_score < self.confidence_threshold:
            print(f"VerifierAgent: Low confidence ({confidence_score:.2f} < {self.confidence_threshold}), marking for reprocessing")
            return True
        
        # Case 3: Critical issues exist
        critical_issues = [issue for issue in issues if any(keyword in issue for keyword in critical_keywords)]
        if critical_issues:
            print(f"VerifierAgent: Critical issues detected: {critical_issues[:2]}, marking for reprocessing")
            return True
        
        # Case 4: Too many issues
        if len(issues) >= 4:
            print(f"VerifierAgent: Too many issues ({len(issues)}), marking for reprocessing")
            return True
        
        # Case 5: High standard check in strict mode
        if confidence_score < self.strict_confidence_threshold and len(issues) > 0:
            print(f"VerifierAgent: Strict mode - confidence {confidence_score:.2f} with issues, marking for reprocessing")
            return True
        
        print("VerifierAgent: Verification passed")
        return False
    
    def _record_failure_details(self, data, is_consistent, confidence_score, issues, suggestions):
        """Record detailed failure information
        """
        data.verification_failure_details = {
            "failure_type": "consistency_check_failed" if not is_consistent else "low_confidence",
            "confidence_score": confidence_score,
            "threshold_used": self.confidence_threshold,
            "detailed_issues": issues,
            "improvement_suggestions": suggestions,
            "original_extraction": data.pred,
            "can_be_corrected": len(suggestions) > 0
        }