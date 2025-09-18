from models import *
from utils import *
from .extraction_agent import ExtractionAgent
from .knowledge_base.case_repository import CaseRepositoryHandler

class ReflectionGenerator:
    """
    Reflection Generator class for generating reflection results
    """
    
    def __init__(self, llm: BaseEngine):
        """
        Initialize Reflection Generator
        
        Args:
            llm: Large Language Model engine
        """
        self.llm = llm

    def get_reflection(self, instruction="", examples="", text="", schema="", result=""):
        """
        Get reflection results
        
        Args:
            instruction: Instruction text
            examples: Example text
            text: Input text
            schema: Output schema
            result: Extraction result
            
        Returns:
            dict: Reflected result
        """
        # Convert result to JSON string
        result = json.dumps(result)
        # Wrap bad case examples
        examples = bad_case_wrapper(examples)
        # Format reflection instruction
        prompt = reflect_instruction.format(
            instruction=instruction, 
            examples=examples, 
            text=text, 
            schema=schema, 
            result=result
        )
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract JSON dictionary
        response = extract_json_dict(response)
        # Print reflection result
        print(f"[DEBUG] Reflection result: {json.dumps(response, ensure_ascii=False, indent=2)}")
        return response

class ReflectionAgent:
    """
    Reflection Agent class responsible for managing reflection process and self-consistency check
    """
    
    def __init__(self, llm: BaseEngine, case_repo: CaseRepositoryHandler):
        """
        Initialize Reflection Agent
        
        Args:
            llm: Large Language Model engine
            case_repo: Case repository handler
        """
        self.llm = llm
        self.module = ReflectionGenerator(llm=llm)  # Reflection generator module
        self.extractor = ExtractionAgent(llm=llm, case_repo=case_repo)  # Extraction agent
        self.case_repo = case_repo  # Case repository
        self.methods = ["reflect_with_case"]  # Available methods list

    def __select_result(self, result_list):
        """
        Select the best result from result list
        
        Args:
            result_list: Result list
            
        Returns:
            object: Selected best result
        """
        # Filter dictionary type objects
        dict_objects = [obj for obj in result_list if isinstance(obj, dict)]
        if dict_objects:
            # Select dictionary with longest JSON string
            selected_obj = max(dict_objects, key=lambda d: len(json.dumps(d)))
        else:
            # If no dict objects, select object with longest JSON string
            selected_obj = max(result_list, key=lambda o: len(json.dumps(o)))
        return selected_obj

    def __self_consistance_check(self, data: DataPoint):
        """
        Perform self-consistency check
        
        Args:
            data: DataPoint object
            
        Returns:
            list: List of indices that need reflection
        """
        # Get the last used extraction function name
        extract_func = list(data.result_trajectory.keys())[-1]
        
        # Check if extractor has the function
        if hasattr(self.extractor, extract_func):
            result_trails = []  # Result trails list
            result_trails.append(data.result_list)  # Add original result
            
            # Get extraction function
            extract_func = getattr(self.extractor, extract_func)
            
            # Set different temperature parameters for multiple extractions
            temperature = [0.5, 1]
            for index in range(2):
                # Set temperature parameter
                self.module.llm.set_hyperparameter(temperature=temperature[index])
                # Perform extraction
                data = extract_func(data)
                # Add result to trails
                result_trails.append(data.result_list)
            
            # Reset hyperparameters
            self.module.llm.set_hyperparameter()
            
            consistant_result = []  # Consistent result list
            reflect_index = []      # Indices that need reflection
            
            # Check consistency for results at each position
            for index, elements in enumerate(zip(*result_trails)):
                # Normalize elements
                normalized_elements = [normalize_obj(e) for e in elements]
                # Count element occurrences
                element_counts = Counter(normalized_elements)
                
                # Select element that appears >= 2 times
                selected_element = next((elements[i] for i, element in enumerate(normalized_elements)
                                        if element_counts[element] >= 2), None)
                
                if selected_element is None:
                    # If no consistent element, select best result and mark for reflection
                    selected_element = self.__select_result(elements)
                    reflect_index.append(index)
                
                consistant_result.append(selected_element)
            
            # Set consistent result
            data.set_result_list(consistant_result)
            return reflect_index

    def reflect_with_case(self, data: DataPoint):
        """Improved reflection method with verification feedback support"""
        if data.result_list == []:
            return data
        
        if not data.chunk_text_list or len(data.chunk_text_list) == 0:
            print("Warning: chunk_text_list is empty, skipping reflection")
            return data
        
        # Perform self-consistency check
        reflect_index = self.__self_consistance_check(data)
        
        if reflect_index is None:
            print("Warning: reflect_index is None, skipping reflection")
            return data
        
        # Get verification feedback context
        verification_context = data.get_reprocessing_context()
        
        reflected_result_list = data.result_list
        
        for idx in reflect_index:
            if idx >= len(data.chunk_text_list):
                print(f"Warning: reflect_index {idx} exceeds chunk_text_list length {len(data.chunk_text_list)}")
                continue
            
            text = data.chunk_text_list[idx]
            result = data.result_list[idx]
            
            # Query bad cases and combine with verification feedback
            examples = json.dumps(self.case_repo.query_bad_case(data))
            
            # If there's verification feedback, add it to the instruction
            enhanced_instruction = data.instruction
            if verification_context:
                enhanced_instruction += "\n" + verification_context
                print(f"[REPROCESSING] Reflection using verification feedback")
            
            # Generate reflection result
            reflected_res = self.module.get_reflection(
                instruction=enhanced_instruction, 
                examples=examples, 
                text=text, 
                schema=data.output_schema, 
                result=result
            )
            
            reflected_result_list[idx] = reflected_res
        
        data.set_result_list(reflected_result_list)
        
        # Record trajectory including verification feedback usage
        function_name = current_function_name()
        data.update_trajectory(function_name, {
            "results": data.result_list,
            "used_verification_feedback": verification_context != "",
            "feedback_context": verification_context if verification_context else None
        })
        
        return data