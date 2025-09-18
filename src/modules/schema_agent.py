from models import *
from utils import *
from .knowledge_base import schema_repository
from langchain_core.output_parsers import JsonOutputParser

class SchemaAnalyzer:
    """
    Schema Analyzer class for processing and analyzing data schemas
    """
    
    def __init__(self, llm: BaseEngine):
        """
        Initialize Schema Analyzer
        
        Args:
            llm: Large Language Model engine
        """
        self.llm = llm

    def serialize_schema(self, schema) -> str:
        """
        Serialize schema object to string format
        
        Args:
            schema: Schema object to serialize
            
        Returns:
            str: Serialized schema string
        """
        # Return directly if basic data type
        if isinstance(schema, (str, list, dict, set, tuple)):
            return schema
        try:
            # Use JsonOutputParser to parse Pydantic object
            parser = JsonOutputParser(pydantic_object = schema)
            schema_description = parser.get_format_instructions()
            # Extract content from code blocks
            schema_content = re.findall(r'```(.*?)```', schema_description, re.DOTALL)
            # Add example explanation
            explanation = "For example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}}, the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance."
            schema = f"{schema_content}\n\n{explanation}"
        except:
            # Return original schema if parsing fails
            return schema
        return schema

    def redefine_text(self, text_analysis):
        """
        Redefine text analysis results
        
        Args:
            text_analysis: Text analysis results
            
        Returns:
            str: Redefined text description
        """
        try:
            # Extract field and genre information
            field = text_analysis['field']
            genre = text_analysis['genre']
        except:
            # Return original analysis if extraction fails
            return text_analysis
        # Generate text description prompt
        prompt = f"This text is from the field of {field} and represents the genre of {genre}."
        return prompt

    def get_text_analysis(self, text: str):
        """
        Get text analysis results
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Text analysis results
        """
        # Get output schema for text description
        output_schema = self.serialize_schema(schema_repository.TextDescription)
        # Format analysis instruction
        prompt = text_analysis_instruction.format(examples="", text=text, schema=output_schema)
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract JSON dictionary
        response = extract_json_dict(response)
        # Redefine text
        response = self.redefine_text(response)
        return response

    def get_deduced_schema_json(self, instruction: str, text: str, distilled_text: str):
        """
        Get deduced schema in JSON format
        
        Args:
            instruction: Instruction text
            text: Original text
            distilled_text: Distilled text
            
        Returns:
            tuple: (code, response)
        """
        # Format deduced schema instruction
        prompt = deduced_schema_json_instruction.format(
            examples=example_wrapper(json_schema_examples), 
            instruction=instruction, 
            distilled_text=distilled_text, 
            text=text
        )
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract JSON dictionary
        response = extract_json_dict(response)
        code = response
        print(f"Deduced Schema in Json: \n{response}\n\n")
        return code, response

    def get_deduced_schema_code(self, instruction: str, text: str, distilled_text: str):
        """
        Get deduced schema in code format
        
        Args:
            instruction: Instruction text
            text: Original text
            distilled_text: Distilled text
            
        Returns:
            tuple: (code, schema)
        """
        # Format deduced schema instruction
        prompt = deduced_schema_code_instruction.format(
            examples=example_wrapper(code_schema_examples), 
            instruction=instruction, 
            distilled_text=distilled_text, 
            text=text
        )
        # Get LLM response
        response = self.llm.get_chat_response(prompt)
        # Extract code blocks
        code_blocks = re.findall(r'```[^\n]*\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            try:
                # Get the last code block
                code_block = code_blocks[-1]
                namespace = {}
                # Execute code block
                exec(code_block, namespace)
                # Get ExtractionTarget class
                schema = namespace.get('ExtractionTarget')
                if schema is not None:
                    # Extract class definition part
                    index = code_block.find("class")
                    code = code_block[index:]
                    print(f"Deduced Schema in Code: \n{code}\n\n")
                    # Serialize schema
                    schema = self.serialize_schema(schema)
                    return code, schema
            except Exception as e:
                print(e)
                # Fallback to JSON format if execution fails
                return self.get_deduced_schema_json(instruction, text, distilled_text)
        # Fallback to JSON format if no code blocks
        return self.get_deduced_schema_json(instruction, text, distilled_text)

class SchemaAgent:
    """
    Schema Agent class responsible for managing and selecting different schema acquisition methods
    """
    
    def __init__(self, llm: BaseEngine):
        """
        Initialize Schema Agent
        
        Args:
            llm: Large Language Model engine
        """
        self.llm = llm
        self.module = SchemaAnalyzer(llm = llm)  # Schema analyzer module
        self.schema_repo = schema_repository      # Schema repository
        # Available methods list
        self.methods = ["get_default_schema", "get_retrieved_schema", "get_deduced_schema"]

    def __preprocess_text(self, data: DataPoint):
        """
        Preprocess text data
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: Processed DataPoint
        """
        # Chunk text based on data source
        if data.use_file:
            # Chunk from file
            data.chunk_text_list = chunk_file(data.file_path)
        else:
            # Chunk from string
            data.chunk_text_list = chunk_str(data.text)
            
        # Set print schema based on task type
        if data.task == "NER":  # Named Entity Recognition
            data.print_schema = """
class Entity(BaseModel):
    name : str = Field(description="The specific name of the entity. ")
    type : str = Field(description="The type or category that the entity belongs to.")
class EntityList(BaseModel):
    entity_list : List[Entity] = Field(description="Named entities appearing in the text.")
            """
        elif data.task == "RE":  # Relation Extraction
            data.print_schema = """
class Relation(BaseModel):
    head : str = Field(description="The starting entity in the relationship.")
    tail : str = Field(description="The ending entity in the relationship.")
    relation : str = Field(description="The predicate that defines the relationship between the two entities.")

class RelationList(BaseModel):
    relation_list : List[Relation] = Field(description="The collection of relationships between various entities.")
            """
        elif data.task == "EE":  # Event Extraction
            data.print_schema = """
class Event(BaseModel):
    event_type : str = Field(description="The type of the event.")
    event_trigger : str = Field(description="A specific word or phrase that indicates the occurrence of the event.")
    event_argument : dict = Field(description="The arguments or participants involved in the event.")

class EventList(BaseModel):
    event_list : List[Event] = Field(description="The events presented in the text.")
            """
        elif data.task == "Triple":  # Triple Extraction
            data.print_schema = """
class Triple(BaseModel):
    head: str = Field(description="The subject or head of the triple.")
    head_type: str = Field(description="The type of the subject entity.")
    relation: str = Field(description="The predicate or relation between the entities.")
    relation_type: str = Field(description="The type of the relation.")
    tail: str = Field(description="The object or tail of the triple.")
    tail_type: str = Field(description="The type of the object entity.")
class TripleList(BaseModel):
    triple_list: List[Triple] = Field(description="The collection of triples and their types presented in the text.")
"""
        return data

    def get_default_schema(self, data: DataPoint):
        """
        Get default schema
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: DataPoint with default schema set
        """
        # Preprocess text
        data = self.__preprocess_text(data)
        # Get default schema configuration
        default_schema = config['agent']['default_schema']
        # Set schema
        data.set_schema(default_schema)
        # Record function call trajectory
        function_name = current_function_name()
        data.update_trajectory(function_name, default_schema)
        return data

    def get_retrieved_schema(self, data: DataPoint):
        """
        Get retrieved schema
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: DataPoint with retrieved schema set
        """
        # Preprocess text
        self.__preprocess_text(data)
        # Get schema name
        schema_name = data.output_schema
        # Get schema class from repository
        schema_class = getattr(self.schema_repo, schema_name, None)
        if schema_class is not None:
            # Serialize schema
            schema = self.module.serialize_schema(schema_class)
            # Get default schema
            default_schema = config['agent']['default_schema']
            # Combine schemas
            data.set_schema(f"{default_schema}\n{schema}")
            # Record function call trajectory
            function_name = current_function_name()
            data.update_trajectory(function_name, schema)
        else:
            # Use default schema if schema doesn't exist
            return self.get_default_schema(data)
        return data

    def get_deduced_schema(self, data: DataPoint):
        """
        Get deduced schema
        
        Args:
            data: DataPoint object
            
        Returns:
            DataPoint: DataPoint with deduced schema set
        """
        # Preprocess text
        self.__preprocess_text(data)
        # Get target text (first text chunk)
        target_text = data.chunk_text_list[0]
        # Analyze text
        analysed_text = self.module.get_text_analysis(target_text)
        # Add prefix if multiple text chunks
        if len(data.chunk_text_list) > 1:
            prefix = "Below is a portion of the text to be extracted. "
            analysed_text = f"{prefix}\n{target_text}"
        # Redefine text
        distilled_text = self.module.redefine_text(analysed_text)
        # Get deduced code schema
        code, deduced_schema = self.module.get_deduced_schema_code(data.instruction, target_text, distilled_text)
        # Set print schema
        data.print_schema = code
        # Set distilled text
        data.set_distilled_text(distilled_text)
        # Get default schema and combine
        default_schema = config['agent']['default_schema']
        data.set_schema(f"{default_schema}\n{deduced_schema}")
        # Record function call trajectory
        function_name = current_function_name()
        data.update_trajectory(function_name, deduced_schema)
        return data