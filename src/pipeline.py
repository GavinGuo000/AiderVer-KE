from typing import Literal
from models import *
from utils import *
from modules import *
from modules.collaborative_memory import CollaborativeMemory
from construct import *

class Pipeline:
    """
    Pipeline Class - Main controller for AiderVer-KE knowledge extraction pipeline (MRD Section 4)
    
    Five-agent collaborative architecture:
    - Schema Agent: Generate extraction schema with closed-set constraints
    - Aider Agent: Retrieve external knowledge for disambiguation
    - Extraction Agent: Execute structured information extraction
    - Probe Agent: Quality inspection and defect feedback generation
    - Verifier Agent: Final verification with triple-check rules
    
    Collaborative Memory: Cross-agent shared error experience pool
    """
    
    def __init__(self, llm: BaseEngine):
        """
        Initialize Pipeline instance
        
        Args:
            llm: Base language model engine
        """
        self.llm = llm
        self.case_repo = CaseRepositoryHandler(llm = llm)
        self.collaborative_memory = CollaborativeMemory()
        self.schema_agent = SchemaAgent(llm = llm)
        self.aider_agent = AiderAgent(llm = llm, case_repo = self.case_repo)
        self.extraction_agent = ExtractionAgent(llm = llm, case_repo = self.case_repo)
        self.probe_agent = ProbeAgent(llm = llm, case_repo = self.case_repo)
        self.verifier_agent = VerifierAgent(llm = llm, case_repo = self.case_repo)

    def __should_use_aider_agent(self, data: DataPoint):
        """
        Intelligently determine whether to use aider_agent
        
        Args:
            data: DataPoint object
            
        Returns:
            bool: Whether aider agent is needed
        """
        # Build decision prompt
        decision_prompt = f"""
Please analyze the following text and determine whether external knowledge retrieval is needed to assist with the {data.task} task:

**Text Content:**
{data.text[:500]}{'...' if len(data.text) > 500 else ''}

**Task Type:** {data.task}
**Task Instruction:** {data.instruction}

**Judgment Criteria:**
1. Does the text contain professional terms, person names, place names, organization names, or other entities that require background knowledge
2. Does the text involve domain-specific concepts or events
3. Is external knowledge needed to accurately understand entity types or relationships
4. Does the text contain abbreviations, aliases, or concepts that need explanation

Please return the judgment result in JSON format:
{{
    "need_aider": true/false,
    "confidence": 0.0-1.0,
    "reason": "judgment reason",
    "key_entities": ["list of key entities requiring external knowledge"]
}}
"""
        
        try:
            # Get response from language model
            response = self.llm.get_chat_response(decision_prompt)
            decision_result = extract_json_dict(response)
            
            # Extract decision results
            need_aider = decision_result.get("need_aider", False)
            confidence = decision_result.get("confidence", 0.5)
            reason = decision_result.get("reason", "No reason provided")
            
            # Print decision information
            print(f"AiderAgent Decision: {'Needed' if need_aider else 'Not needed'} (Confidence: {confidence:.2f})")
            print(f"Decision Reason: {reason}")
            
            # Record decision trajectory
            data.result_trajectory["aider_decision"] = {
                "need_aider": need_aider,
                "confidence": confidence,
                "reason": reason,
                "key_entities": decision_result.get("key_entities", [])
            }
            
            # Only use when confidence is high
            return need_aider and confidence > 0.6
            
        except Exception as e:
            # Exception handling, default to not use
            print(f"AiderAgent decision error: {e}, defaulting to not use")
            return False

    def __check_consistancy(self, llm, task, mode, update_case):
        """
        Check configuration consistency
        
        Args:
            llm: Language model
            task: Task type
            mode: Running mode
            update_case: Whether to update cases
            
        Returns:
            tuple: Adjusted mode and update_case flag
        """
        if llm.name == "OneKE":
            if task == "Base" or task == "Triple":
                raise ValueError("The finetuned OneKE only supports quick extraction mode for NER, RE and EE Task.")
            else:
                mode = "quick"
                update_case = False
                print("The fine-tuned OneKE defaults to quick extraction mode without case update.")
                return mode, update_case
        return mode, update_case

    def __init_method(self, data: DataPoint, method, mode: str = "standard"):
        """
        Initialize processing methods and agent execution order
        
        Args:
            data: DataPoint object
            method: Method configuration
            mode: Running mode
            
        Returns:
            dict: Sorted processing methods
        """
        # Base process order (MRD Section 4.2: Schema → Extraction → Probe → Verifier)
        base_order = ["schema_agent", "extraction_agent", "probe_agent", "verifier_agent"]
        
        # Ensure method is a dictionary
        if not isinstance(method, dict):
            method = {
                "schema_agent": "get_default_schema",
                "extraction_agent": "extract_information_direct",
                "probe_agent": "probe_extraction_quality",
                "verifier_agent": "verify_extraction_result"
            }
        
        # Determine process order based on mode
        current_mode = mode
        
        if current_mode == "enhanced":
            # Enhanced mode: complete agent workflow (MRD full pipeline)
            default_order = ["schema_agent", "aider_agent", "extraction_agent", "probe_agent", "verifier_agent"]
        elif current_mode == "ablation_no_aider":
            # Ablation mode: force not using aider_agent
            default_order = base_order
            # Ensure aider_agent is not added
            if "aider_agent" in method:
                del method["aider_agent"]
            print("ablation_no_aider mode: aider_agent forcibly removed")
        elif current_mode == "ablation_no_verifier":
            # Ablation mode: based on enhanced mode but remove verifier_agent
            default_order = ["schema_agent", "aider_agent", "extraction_agent", "probe_agent"]
            # Ensure verifier_agent is not added
            if "verifier_agent" in method:
                del method["verifier_agent"]
            print("ablation_no_verifier mode: verifier_agent forcibly removed")
        else:
            # Other modes: intelligent decision on whether to use aider_agent
            if "aider_agent" not in method:
                if self.__should_use_aider_agent(data):
                    method["aider_agent"] = "enhance_with_knowledge"
                    # If aider_agent is needed, insert it before extraction_agent
                    default_order = ["schema_agent", "aider_agent", "extraction_agent", "probe_agent", "verifier_agent"]
                else:
                    # Workflow without aider_agent
                    default_order = base_order
            else:
                # If aider_agent is explicitly specified, use complete workflow
                default_order = ["schema_agent", "aider_agent", "extraction_agent", "probe_agent", "verifier_agent"]
        
        # Set default methods
        if "schema_agent" not in method:
            method["schema_agent"] = "get_default_schema"
        if data.task != "Base":
            method["schema_agent"] = "get_retrieved_schema"
        if "extraction_agent" not in method:
            method["extraction_agent"] = "extract_information_direct"
        if "probe_agent" not in method:
            method["probe_agent"] = "probe_extraction_quality"
        if "verifier_agent" not in method:
            method["verifier_agent"] = "verify_extraction_result"
            
        # Sort processing methods by order
        sorted_process_method = {key: method[key] for key in default_order if key in method}
        return sorted_process_method

    def __init_data(self, data: DataPoint):
        """
        Initialize task-related configuration for data point
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: Initialized data point
        """
        if data.task == "NER":
            data.instruction = config['agent']['default_ner']
            data.output_schema = "EntityList"
        elif data.task == "RE":
            data.instruction = config['agent']['default_re']
            data.output_schema = "RelationList"
        elif data.task == "EE":
            data.instruction = config['agent']['default_ee']
            data.output_schema = "EventList"
        elif data.task == "Triple":
            data.instruction = config['agent']['default_triple']
            data.output_schema = "TripleList"
        return data

    # Main entry function
    def get_extract_result(self,
                           task: TaskType,
                           three_agents = {},
                           construct = {},
                           instruction: str = "",
                           text: str = "",
                           output_schema: str = "",
                           constraint: str = "",
                           use_file: bool = False,
                           file_path: str = "",
                           truth: str = "",
                           mode: str = "quick",
                           update_case: bool = False,
                           show_trajectory: bool = False,
                           isgui: bool = False,
                           iskg: bool = False,
                           config_name: str = "",
                           max_reprocessing_attempts: int = 2,
                           ):
        """
        Main function to get extraction results
        
        Returns:
            tuple: (result, trajectory, frontend_schema, frontend_res)
        """

        # Check consistency
        mode, update_case = self.__check_consistancy(self.llm, task, mode, update_case)

        # Load data
        data = DataPoint(task=task, instruction=instruction, text=text, output_schema=output_schema, constraint=constraint, use_file=use_file, file_path=file_path, truth=truth)
        data = self.__init_data(data)
        
        # Get processing method configuration
        if mode in config['agent']['mode'].keys():
            process_method = config['agent']['mode'][mode].copy()
        else:
            process_method = mode

        # GUI customized mode
        if isgui and mode == "customized":
            process_method = three_agents
            print("Customized Agents: ", three_agents)

        # Initialize processing methods
        sorted_process_method = self.__init_method(data, process_method, mode)
        print("Process Method: ", sorted_process_method)

        # Initialize variables
        print_schema = False
        frontend_schema = ""
        frontend_res = ""
        reprocessing_attempts = 0
        
        # Inject collaborative memory context for Extraction Agent (MRD Section 6)
        memory_context = self.collaborative_memory.get_error_summary_for_extraction(
            task_type=task, text=text
        )
        if memory_context:
            data.collaborative_memory_context = memory_context
            print(f"[CollaborativeMemory] Injected historical error context")

        # Main processing loop, supports reprocessing
        while reprocessing_attempts <= max_reprocessing_attempts:
            print(f"\n=== Processing Round {reprocessing_attempts + 1} ===")
            
            # Information extraction
            for agent_name, method_name in sorted_process_method.items():
                # Get agent object
                agent = getattr(self, agent_name, None)
                if not agent:
                    continue
                # Get method
                method = getattr(agent, method_name, None)
                if not method:
                    continue
                    
                print(f"Executing {agent_name}.{method_name}")
                data = method(data)
                
                # Print schema information
                if not print_schema and hasattr(data, 'print_schema') and data.print_schema:
                    print("Schema: \n", data.print_schema)
                    frontend_schema = data.print_schema
                    print_schema = True
            
            # Check if reprocessing is needed (MRD Section 4.2: verification failure triggers re-extraction)
            if hasattr(data, 'needs_reprocessing') and data.needs_reprocessing and reprocessing_attempts < max_reprocessing_attempts:
                print(f"\nReprocessing needed, reason: {data.reprocessing_reason}")
                print(f"Suggestions: {data.reprocessing_suggestions}")
                
                # Save verifier feedback to data object
                if hasattr(data, 'verification_result'):
                    data.set_verification_feedback(data.verification_result)
                    print(f"[REPROCESSING] Verification feedback saved for context passing")
                
                reprocessing_attempts += 1
                data.needs_reprocessing = False
                
                # Decide reprocessing strategy based on issue type and severity
                issues = data.verification_result.get('issues', []) if hasattr(data, 'verification_result') else []
            
                # Analyze issue types and adopt different reprocessing strategies
                has_extraction_issues = any('extraction' in str(issue).lower() or 'entity' in str(issue).lower() for issue in issues)
                has_type_issues = any('type' in str(issue).lower() or 'category' in str(issue).lower() for issue in issues)
                has_completeness_issues = any('missed' in str(issue).lower() or 'incomplete' in str(issue).lower() for issue in issues)
                
                if has_extraction_issues or has_type_issues:
                    # Severe issues: restart from extraction_agent (MRD: feedback loop back to extraction)
                    restart_agents = ["extraction_agent", "probe_agent", "verifier_agent"]
                    print(f"[REPROCESSING] Severe issues detected, restarting from extraction_agent")
                elif has_completeness_issues:
                    # Completeness issues: restart from probe_agent
                    restart_agents = ["probe_agent", "verifier_agent"]
                    print(f"[REPROCESSING] Completeness issues detected, restarting from probe_agent")
                else:
                    # Other issues: re-verify only
                    restart_agents = ["verifier_agent"]
                    print(f"[REPROCESSING] Minor issues detected, re-verifying only")
                
                # Update processing method configuration
                sorted_process_method = {key: sorted_process_method[key] for key in restart_agents if key in sorted_process_method}
                
                # Record reprocessing information
                data.update_trajectory("reprocessing_decision", {
                    "attempt": reprocessing_attempts,
                    "issues_detected": issues,
                    "restart_strategy": restart_agents,
                    "has_extraction_issues": has_extraction_issues,
                    "has_type_issues": has_type_issues,
                    "has_completeness_issues": has_completeness_issues
                })
                continue
            else:
                break

        if self.extraction_agent is not None:
            data = self.extraction_agent.summarize_answer(data)
        else:
            # If no extraction agent, set empty result based on task type
            if data.task == "NER":
                data.pred = []
            elif data.task == "RE":
                data.pred = []
            elif data.task == "EE":
                data.pred = []
            elif data.task == "Triple":
                data.pred = []
            else:
                data.pred = []

        # Show results
        if not isgui:
            # Show trajectory
            if show_trajectory:
                print("Extraction Trajectory: \n", json.dumps(data.get_result_trajectory(), indent=4, ensure_ascii=False))
            
            # Console output in formatted JSON
            if type(data.pred) is not str:
                extraction_result = json.dumps(data.pred, indent=4, ensure_ascii=False)
            print("Extraction Result: \n", extraction_result)
            
            # Show verification result
            if hasattr(data, 'verification_result'):
                print("\nVerification Result: \n", json.dumps(data.verification_result, indent=4, ensure_ascii=False))
            
            # Add download functionality
            if config_name:
                import os
                # Create result directory
                result_dir = "examples/results"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                # Extract filename from full path (remove path and extension)
                base_name = os.path.splitext(os.path.basename(config_name))[0]
                
                # Save extraction result as formatted JSON
                result_file_path = os.path.join(result_dir, f"{base_name}.json")
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    f.write(extraction_result)
                print(f"Extraction Result has been saved to: {result_file_path}")

        # Construct knowledge graph
        if iskg:
            myurl = construct['url']
            myusername = construct['username']
            mypassword = construct['password']
            print(f"Construct KG in your {construct['database']} now...")
            cypher_statements = generate_cypher_statements(extraction_result)
            execute_cypher_statements(uri=myurl, user=myusername, password=mypassword, cypher_statements=cypher_statements)

        frontend_res = data.pred

        # Case update
        if update_case:
            if self.case_repo is None:
                print("Warning: Case update not available - CaseRepositoryHandler not loaded.")
            else:
                if (data.truth == ""):
                    truth = input("Please enter the correct answer you prefer, or just press Enter to accept the current answer: ")
                    if truth.strip() == "":
                        data.truth = data.pred
                    else:
                        data.truth = extract_json_dict(truth)
                self.case_repo.update_case(data)
        
        # Persist extraction session to Collaborative Memory (MRD Section 6)
        self._persist_to_collaborative_memory(data, task, text, reprocessing_attempts)

        # Return results
        result = data.pred
        trajectory = data.get_result_trajectory()

        return result, trajectory, frontend_schema, frontend_res

    def _persist_to_collaborative_memory(self, data: DataPoint, task: str, text: str, reprocessing_attempts: int):
        """
        Persist extraction session results to Collaborative Memory (MRD Section 6)
        
        Records:
        1. Extraction errors and probe feedback for failed/low-quality extractions
        2. Complete extraction session summary for system self-evolution
        
        Args:
            data: DataPoint with all processing results
            task: Task type
            text: Original input text
            reprocessing_attempts: Number of reprocessing attempts made
        """
        try:
            # Record probe feedback if available
            if hasattr(data, 'probe_feedback') and data.probe_feedback:
                self.collaborative_memory.record_probe_feedback(
                    task_type=task,
                    probe_feedback_list=data.probe_feedback
                )
            
            # Record extraction errors if verification found issues
            if hasattr(data, 'verification_result') and data.verification_result:
                verification_result = data.verification_result
                is_consistent = verification_result.get("is_consistent", True)
                confidence = verification_result.get("confidence_score", 1.0)
                
                # Only record errors for failed or low-confidence extractions
                if not is_consistent or confidence < 0.8:
                    probe_feedback = {}
                    if hasattr(data, 'probe_feedback') and data.probe_feedback:
                        probe_feedback = data.probe_feedback[0] if data.probe_feedback else {}
                    
                    self.collaborative_memory.record_extraction_errors(
                        task_type=task,
                        text=text,
                        extraction_result=data.pred,
                        probe_feedback=probe_feedback,
                        verification_result=verification_result
                    )
            
            # Record complete extraction session
            self.collaborative_memory.record_extraction_session(
                task_type=task,
                text=text,
                extraction_result=data.result_list[0] if data.result_list else {},
                final_result=data.pred,
                reprocessing_count=reprocessing_attempts
            )
            
            print(f"[CollaborativeMemory] Session persisted successfully")
            
        except Exception as e:
            print(f"[CollaborativeMemory] Failed to persist session: {e}")
