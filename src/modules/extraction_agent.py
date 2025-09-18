from models import *
from utils import *
from .knowledge_base.case_repository import CaseRepositoryHandler

class InformationExtractor:
    """
    Information Extractor class responsible for performing specific information extraction tasks
    """
    
    def __init__(self, llm: BaseEngine):
        """
        Initialize Information Extractor
        
        Args:
            llm: Large Language Model engine
        """
        self.llm = llm

    def extract_information(self, instruction="", text="", examples="", schema="", additional_info=""):
        """
        Main method for extracting information
        
        Args:
            instruction: Extraction instruction
            text: Input text
            examples: Example text
            schema: Output schema
            additional_info: Additional information (such as constraints)
            
        Returns:
            dict: Extraction result
        """
        # Wrap good case examples
        examples = good_case_wrapper(examples)
        # Format extraction instruction
        prompt = extract_instruction.format(
            instruction=instruction, 
            examples=examples, 
            text=text, 
            additional_info=additional_info, 
            schema=schema
        )
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract JSON dictionary
        response = extract_json_dict(response)
        return response

    def extract_information_compatible(self, task="", text="", constraint=""):
        """
        Compatible mode information extraction method for OneKE model
        
        Args:
            task: Task type
            text: Input text
            constraint: Constraints
            
        Returns:
            dict: Extraction result
        """
        # Get instruction based on task type
        instruction = instruction_mapper.get(task)
        # Format JSON extraction instruction
        prompt = extract_instruction_json.format(
            instruction=instruction, 
            constraint=constraint, 
            input=text
        )
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract JSON dictionary
        response = extract_json_dict(response)
        return response

    def summarize_answer(self, instruction="", answer_list="", schema="", additional_info=""):
        """
        Summarize multiple extraction results
        
        Args:
            instruction: Extraction instruction
            answer_list: Answer list
            schema: Output schema
            additional_info: Additional information
            
        Returns:
            dict: Summarized result
        """
        # Format summarization instruction
        prompt = summarize_instruction.format(
            instruction=instruction, 
            answer_list=answer_list, 
            schema=schema, 
            additional_info=additional_info
        )
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract JSON dictionary
        response = extract_json_dict(response)
        return response

class ExtractionAgent:
    """
    Extraction Agent class responsible for managing information extraction process and constraint handling
    """
    
    def __init__(self, llm: BaseEngine, case_repo: CaseRepositoryHandler):
        """
        Initialize Extraction Agent
        
        Args:
            llm: Large Language Model engine
            case_repo: Case repository handler
        """
        self.llm = llm
        self.module = InformationExtractor(llm=llm)  # Information extractor module
        self.case_repo = case_repo  # Case repository
        # Available methods list
        self.methods = ["extract_information_direct", "extract_information_with_case"]

    def __get_constraint(self, data: DataPoint):
        """
        Process and format constraints
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: Processed DataPoint
        """
        # Return directly if constraint is empty
        if data.constraint in ("", [], {}, None):
            return data
            
        # Handle constraints for Named Entity Recognition (NER) task
        if data.task == "NER":
            constraint = json.dumps(data.constraint)
            # Check if already formatted or using OneKE model
            if "**Entity Type Constraint**" in constraint or self.llm.name == "OneKE":
                return data
            # Format entity type constraint
            data.constraint = f"\n**Entity Type Constraint**: The type of entities must be chosen from the following list.\n{constraint}\n"
            
        # Handle constraints for Relation Extraction (RE) task
        elif data.task == "RE":
            constraint = json.dumps(data.constraint)
            # Check if already formatted or using OneKE model
            if "**Relation Type Constraint**" in constraint or self.llm.name == "OneKE":
                return data
            # Format relation type constraint
            data.constraint = f"\n**Relation Type Constraint**: The type of relations must be chosen from the following list.\n{constraint}\n"
            
        # Handle constraints for Event Extraction (EE) task
        elif data.task == "EE":
            constraint = json.dumps(data.constraint)
            # Check if already formatted
            if "**Event Extraction Constraint**" in constraint:
                return data
            if self.llm.name != "OneKE":
                # Format event extraction constraint for non-OneKE models
                data.constraint = f"\n**Event Extraction Constraint**: The event type must be selected from the following dictionary keys, and its event arguments should be chosen from its corresponding dictionary values. \n{constraint}\n"
            else:
                # Special constraint format handling for OneKE model
                try:
                    result = [
                        {
                            "event_type": key,
                            "trigger": True,
                            "arguments": value
                        }
                        for key, value in data.constraint.items()
                    ]
                    data.constraint = json.dumps(result)
                except:
                    print("Invalid Constraint: Event Extraction constraint must be a dictionary with event types as keys and lists of arguments as values.", data.constraint)
                    
        # Handle constraints for Triple Extraction task
        elif data.task == "Triple":
            constraint = json.dumps(data.constraint)
            # Check if already formatted
            if "**Triple Extraction Constraint**" in constraint:
                return data
            if self.llm.name != "OneKE":
                # Handle different cases based on constraint list length
                if len(data.constraint) == 1:  # 1 list means entity
                    data.constraint = f"\n**Triple Extraction Constraint**: Entities type must chosen from following list:\n{constraint}\n"
                elif len(data.constraint) == 2:  # 2 lists mean entity and relation
                    if data.constraint[0] == []:
                        # Only relation constraint
                        data.constraint = f"\n**Triple Extraction Constraint**: Relation type must chosen from following list:\n{data.constraint[1]}\n"
                    elif data.constraint[1] == []:
                        # Only entity constraint
                        data.constraint = f"\n**Triple Extraction Constraint**: Entities type must chosen from following list:\n{data.constraint[0]}\n"
                    else:
                        # Entity and relation constraints
                        data.constraint = f"\n**Triple Extraction Constraint**: Entities type must chosen from following list:\n{data.constraint[0]}\nRelation type must chosen from following list:\n{data.constraint[1]}\n"
                elif len(data.constraint) == 3:  # 3 lists mean subject entity, relation and object entity
                    if data.constraint[0] == []:
                        # Relation and object entity constraints
                        data.constraint = f"\n**Triple Extraction Constraint**: Relation type must chosen from following list:\n{data.constraint[1]}\nObject Entities must chosen from following list:\n{data.constraint[2]}\n"
                    elif data.constraint[1] == []:
                        # Subject and object entity constraints
                        data.constraint = f"\n**Triple Extraction Constraint**: Subject Entities must chosen from following list:\n{data.constraint[0]}\nObject Entities must chosen from following list:\n{data.constraint[2]}\n"
                    elif data.constraint[2] == []:
                        # Subject entity and relation constraints
                        data.constraint = f"\n**Triple Extraction Constraint**: Subject Entities must chosen from following list:\n{data.constraint[0]}\nRelation type must chosen from following list:\n{data.constraint[1]}\n"
                    else:
                        # Complete triple constraints
                        data.constraint = f"\n**Triple Extraction Constraint**: Subject Entities must chosen from following list:\n{data.constraint[0]}\nRelation type must chosen from following list:\n{data.constraint[1]}\nObject Entities must chosen from following list:\n{data.constraint[2]}\n"
                else:
                    # Default entity type constraint
                    data.constraint = f"\n**Triple Extraction Constraint**: The type of entities must be chosen from the following list:\n{constraint}\n"
            else:
                # OneKE model does not support Triple Extraction yet
                print("OneKE does not support Triple Extraction task now, please wait for the next version.")
        return data

    def extract_information_direct(self, data: DataPoint):
        """Improved direct information extraction method with verification feedback support"""
        # Process constraints
        data = self.__get_constraint(data)
        result_list = []
        
        # Get reprocessing context
        reprocessing_context = data.get_reprocessing_context()
        
        # Extract from each text chunk
        for chunk_text in data.chunk_text_list:
            # Build enhanced constraints including verification feedback
            enhanced_constraint = data.constraint
            if reprocessing_context:
                enhanced_constraint += "\n" + reprocessing_context
                print(f"[REPROCESSING] Using enhanced constraint with verification feedback")
            
            if self.llm.name != "OneKE":
                # Extract using enhanced constraints
                extract_direct_result = self.module.extract_information(
                    instruction=data.instruction, 
                    text=chunk_text, 
                    schema=data.output_schema, 
                    examples="", 
                    additional_info=enhanced_constraint
                )
            else:
                # OneKE compatible method also uses enhanced constraints
                extract_direct_result = self.module.extract_information_compatible(
                    task=data.task, 
                    text=chunk_text, 
                    constraint=enhanced_constraint
                )
            result_list.append(extract_direct_result)
        
        # Record function call and results
        function_name = current_function_name()
        data.set_result_list(result_list)
        data.update_trajectory(function_name, {
            "results": result_list,
            "used_verification_feedback": reprocessing_context != "",
            "feedback_context": reprocessing_context if reprocessing_context else None
        })
        return data

    def extract_information_with_case(self, data: DataPoint):
        """
        Case-based information extraction method
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: DataPoint with extraction results
        """
        # Process constraints
        data = self.__get_constraint(data)
        result_list = []  # Result list
        
        # Extract from each text chunk
        for chunk_text in data.chunk_text_list:
            # Query good cases as examples
            examples = self.case_repo.query_good_case(data)
            # Extract using cases
            extract_case_result = self.module.extract_information(
                instruction=data.instruction, 
                text=chunk_text, 
                schema=data.output_schema, 
                examples=examples, 
                additional_info=data.constraint
            )
            result_list.append(extract_case_result)
        
        # Record function call and results
        function_name = current_function_name()
        data.set_result_list(result_list)
        
        # If only one result, set directly as pred
        if len(result_list) == 1:
            data.set_pred(result_list[0])
        elif len(result_list) > 1:
            # If multiple results, need to handle in subsequent summarize_answer
            pass
        
        data.update_trajectory(function_name, result_list)
        return data

    def summarize_answer(self, data: DataPoint):
        # Return directly if result list is empty
        if len(data.result_list) == 0:
            return data
        
        # If only one result, set directly as prediction
        if len(data.result_list) == 1:
            data.set_pred(data.result_list[0])
            return data
        
        # Summarize multiple results
        summarized_result = self.module.summarize_answer(
            instruction=data.instruction, 
            answer_list=data.result_list, 
            schema=data.output_schema, 
            additional_info=data.constraint
        )
        
        # Record function call and results
        function_name = current_function_name()
        data.set_pred(summarized_result)
        data.update_trajectory(function_name, summarized_result)
        return data
