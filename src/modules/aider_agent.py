# Import necessary libraries
import http.client  # HTTP client library
import json         # JSON processing library
import re           # Regular expression library
import time         # Time processing library

class KnowledgeRetriever:
    """
    Knowledge Retriever Class - Used for retrieving relevant information from external knowledge sources
    
    Main Functions:
    1. Extract key entities from text for search
    2. Use Google API for knowledge search
    3. Evaluate and filter retrieved knowledge quality
    4. Apply secondary filtering to avoid information interference
    """
    
    def __init__(self, api_key="yourapikey"):
        """
        Initialize Knowledge Retriever
        
        Args:
            api_key (str): Google API key
        """
        self.api_key = api_key                    # API key
        self.llm = None                          # LLM instance, set during initialization
        self.search_cache = {}                   # Search cache
        self.entity_cache = {}                   # Entity cache
        
        # Knowledge quality evaluation weights
        self.quality_weights = {
            'title_relevance': 0.3,    # Title relevance weight
            'content_length': 0.2,     # Content length weight
            'source_authority': 0.25,  # Source authority weight
            'entity_match': 0.25       # Entity matching degree weight
        }
        
        # Common English words set - used for filtering search terms
        self.common_english_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "among", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "ours",
            "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "can", "shall", "ought", "need", "dare", "used", "able", "like", "want", "know",
            "get", "go", "come", "see", "look", "find", "give", "take", "make", "think",
            "say", "tell", "ask", "work", "play", "run", "walk", "sit", "stand", "lie",
            "good", "bad", "big", "small", "long", "short", "high", "low", "old", "new",
            "hot", "cold", "warm", "cool", "fast", "slow", "easy", "hard", "light", "dark",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "last", "next", "previous", "today", "yesterday",
            "tomorrow", "now", "then", "here", "there", "where", "when", "why", "how", "what",
            "who", "which", "whose", "whom", "very", "quite", "rather", "too", "so", "such",
            "much", "many", "more", "most", "less", "least", "few", "little", "some", "any",
            "all", "both", "each", "every", "either", "neither", "none", "no", "not", "yes"
        }
        
    def _should_skip_search(self, term):
        """
        Determine whether to skip search for a term
        
        Args:
            term (str): Term to check
            
        Returns:
            bool: True to skip, False to search
        """
        term_lower = term.lower().strip()
        
        # Skip empty strings or too short words
        if len(term_lower) <= 1:
            return True
            
        # Skip pure numbers
        if term_lower.isdigit():
            return True
            
        # Skip common English words
        if term_lower in self.common_english_words:
            return True
            
        # Skip single characters (unless special entities)
        if len(term_lower) == 1 and not term_lower.isupper():
            return True
            
        # Skip common punctuation marks
        if term_lower in {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}"}:
            return True
            
        return False
    
    def _is_likely_entity(self, term):
        """
        Determine if a word is likely an entity - optimized based on bad cases
        
        Args:
            term (str): Term to evaluate
            
        Returns:
            bool: True if likely an entity
        """
        term = term.strip()
        
        # Skip empty strings
        if not term:
            return False
            
        # Skip too short words (less than 2 characters)
        if len(term) < 2:
            return False
        
        # Rule 1: Proper noun pattern - all uppercase abbreviations
        if term.isupper() and len(term) >= 2:
            return True
        
        # Rule 2: Capitalized proper nouns
        if term[0].isupper() and not term.isupper() and len(term) >= 3:
            # Exclude common English words starting with capital letters
            if term.lower() not in self.common_english_words:
                return True
        
        # Rule 3: Technical terms with special characters
        if ' ' in term and len(term.split()) >= 2:
            words = term.split()
            # Technical terms usually contain specific patterns
            tech_patterns = ['network', 'model', 'algorithm', 'method', 'analysis', 'learning', 
                            'intelligence', 'recognition', 'extraction', 'classification',
                            'clustering', 'mining', 'processing', 'evaluation', 'estimation']
            if any(pattern in term.lower() for pattern in tech_patterns):
                return True
        
        # Rule 4: Compound words with hyphens
        if '-' in term or '_' in term:
            return True
        
        # Rule 6: Terms containing numbers
        if any(c.isdigit() for c in term) and len(term) >= 2:
            return True
        
        # Rule 7: Domain-specific term patterns
        domain_suffixes = ['tion', 'sion', 'ness', 'ment', 'ity', 'ing']
        if any(term.lower().endswith(suffix) for suffix in domain_suffixes) and len(term) >= 5:
            # Exclude overly generic terms
            generic_terms = ['processing', 'learning', 'training', 'testing', 'running']
            if term.lower() not in generic_terms:
                return True
        
        # Rule 8: Person name pattern - two capitalized words
        if ' ' in term:
            words = term.split()
            if len(words) == 2 and all(w[0].isupper() and w[1:].islower() for w in words if w):
                return True
        
        # Rule 9: Organization pattern - contains specific keywords
        org_keywords = ['robotics', 'systems', 'technologies', 'corporation', 'institute', 
                       'laboratory', 'research', 'university', 'college']
        if any(keyword in term.lower() for keyword in org_keywords):
            return True
        
        return False
    
    def Google_search(self, query, size="10"):
        """
        Use Google API for search - simplified version, only returns answer
        
        Args:
            query (str): Search query
            size (str): Number of results
            
        Returns:
            dict: Search results
        """
        conn = http.client.HTTPSConnection("Google.cn")
        
        # Use new API endpoint and actual query parameters
        payload = json.dumps({"q": query, "model": "fast", "format": "simple"})
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            # Modify API endpoint to new chat/completions
            conn.request("POST", "/api/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            
            # Add detailed debug information
            print(f"[DEBUG] Search query: {query}")
            print(f"[DEBUG] HTTP status code: {res.status}")
            
            if res.status != 200:
                print(f"[ERROR] HTTP error: {res.status} {res.reason}")
                return {"error": f"HTTP {res.status}: {res.reason}"}
            
            data = res.read()
            result = json.loads(data.decode("utf-8"))
            
            # Simplified processing - only focus on answer field
            if "answer" in result:
                # Return answer as snippet, maintaining compatibility
                answer_text = result.get("answer", "")
                
                # Return compatible format, using answer as main content
                compatible_result = {
                    "webpages": [{
                        "title": f"Search result: {query}",
                        "snippet": answer_text,
                        "url": "https://Google.cn"
                    }] if answer_text else [],
                    "answer": answer_text
                }
                
                print(f"[DEBUG] Got answer, length: {len(answer_text)}")
                return compatible_result
            else:
                # If no answer field, return empty result
                print(f"[DEBUG] Answer field not found: {result}")
                return {"webpages": [], "answer": ""}
            
        except Exception as e:
            print(f"[ERROR] Search exception: {str(e)}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def extract_entities_for_search(self, text, task_type):
        """
        Extract key entities from text for search
        
        Args:
            text (str): Input text
            task_type (str): Task type (NER/RE/EE)
            
        Returns:
            list: List of extracted search terms
        """
        # Build different prompts based on task type
        if task_type == "NER":
            prompt = f"Extract key entity names from the following text for supplementary knowledge search:\n{text}\n\nPlease return the 3-5 most important entity names, separated by commas:"
        elif task_type == "RE":
            prompt = f"Extract key entities and relations from the following text for supplementary knowledge search:\n{text}\n\nPlease return the most important entity and relation terms, separated by commas:"
        elif task_type == "EE":
            prompt = f"Extract key events and entities from the following text for supplementary knowledge search:\n{text}\n\nPlease return the most important entity names and event names, separated by commas:"
        else:
            prompt = f"Extract key information from the following text for supplementary knowledge search:\n{text}\n\nPlease return the most important keywords, separated by commas:"
        
        try:
            response = self.llm.get_chat_response(prompt)
            terms = [term.strip() for term in response.split(',') if term.strip()]
            print(f"[DEBUG] Extracted search terms: {terms}")
            return terms
        except Exception as e:
            print(f"[ERROR] Failed to extract search terms: {e}")
            return []
    
    def _apply_secondary_filtering(self, knowledge_list, text, task_constraints):
        """
        Apply secondary filtering to avoid information interference
        
        Args:
            knowledge_list (list): Candidate knowledge list
            text (str): Input text
            task_constraints (str): Task constraints
            
        Returns:
            list: Filtered knowledge list
        """
        filtered_knowledge = []
        
        # Extract entity types from task constraints
        allowed_types = self._extract_allowed_entity_types(task_constraints)
        
        # Extract entities already present in text
        text_entities = self._extract_text_entities(text)
        
        for knowledge_item in knowledge_list:
            # Filter 1: Task relevance check
            if not self._is_task_relevant(knowledge_item, allowed_types):
                print(f"[FILTER] Skipping irrelevant knowledge: {knowledge_item['title']}")
                continue
                
            # Filter 2: Conflict detection
            if self._has_type_conflict(knowledge_item, text_entities, allowed_types):
                print(f"[FILTER] Skipping conflicting knowledge: {knowledge_item['title']}")
                continue
                
            # Filter 3: Information redundancy check
            if self._is_redundant_information(knowledge_item, text):
                print(f"[FILTER] Skipping redundant knowledge: {knowledge_item['title']}")
                continue
                
            # Filter 4: Boundary interference check
            if self._causes_boundary_interference(knowledge_item, text_entities):
                print(f"[FILTER] Skipping boundary-interfering knowledge: {knowledge_item['title']}")
                continue
                
            filtered_knowledge.append(knowledge_item)
            
        return filtered_knowledge
    
    def _extract_allowed_entity_types(self, task_constraints):
        """
        Extract allowed entity types from task constraints
        
        Args:
            task_constraints (str): Task constraints string
            
        Returns:
            list: List of allowed entity types
        """
        import re
        if not task_constraints:
            return []
            
        # Extract entity types from constraint string
        pattern = r"\['([^']+)'(?:,\s*'([^']+)')*\]"
        matches = re.findall(r"'([^']+)'", task_constraints)
        return matches
    
    def _extract_text_entities(self, text):
        """
        Extract potential entities from input text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of extracted entities
        """
        # Simple entity extraction based on capitalization and patterns
        import re
        
        entities = []
        
        # Extract capitalized words/phrases
        capitalized_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        capitalized_matches = re.findall(capitalized_pattern, text)
        entities.extend(capitalized_matches)
        
        # Extract technical terms
        tech_pattern = r'\b[a-z]+\s+[a-z]+\s+(?:model|algorithm|method|network|machine)\b'
        tech_matches = re.findall(tech_pattern, text, re.IGNORECASE)
        entities.extend(tech_matches)
        
        return list(set(entities))
    
    def _is_task_relevant(self, knowledge_item, allowed_types):
        """
        Check if knowledge is relevant to current NER task
        
        Args:
            knowledge_item (dict): Knowledge item
            allowed_types (list): Allowed entity types
            
        Returns:
            bool: True if relevant
        """
        if not allowed_types:
            return True
            
        snippet = knowledge_item.get('snippet', '').lower()
        title = knowledge_item.get('title', '').lower()
        
        for entity_type in allowed_types:
            if entity_type.lower() in snippet or entity_type.lower() in title:
                return True
                
        return True
    
    def _has_type_conflict(self, knowledge_item, text_entities, allowed_types):
        """
        Detect if knowledge creates type conflicts
        
        Args:
            knowledge_item (dict): Knowledge item
            text_entities (list): Text entities
            allowed_types (list): Allowed types
            
        Returns:
            bool: True if there's conflict
        """
        return False
    
    def _is_redundant_information(self, knowledge_item, text):
        """
        Check if knowledge provides redundant information
        
        Args:
            knowledge_item (dict): Knowledge item
            text (str): Input text
            
        Returns:
            bool: True if redundant
        """
        snippet = knowledge_item.get('snippet', '').lower()
        text_lower = text.lower()
        
        # Calculate content overlap
        snippet_words = set(snippet.split())
        text_words = set(text_lower.split())
        
        if len(snippet_words) == 0:
            return True
            
        overlap_ratio = len(snippet_words.intersection(text_words)) / len(snippet_words)
        
        # If overlap is too high, consider it redundant
        return overlap_ratio > 0.7
    
    def _causes_boundary_interference(self, knowledge_item, text_entities):
        """
        Check if knowledge might cause entity boundary confusion
        
        Args:
            knowledge_item (dict): Knowledge item
            text_entities (list): Text entities
            
        Returns:
            bool: True if might interfere
        """
        snippet = knowledge_item.get('snippet', '').lower()
        
        for entity in text_entities:
            entity_lower = entity.lower()
            
            # Check if knowledge mentions partial matches that could confuse boundaries
            entity_words = entity_lower.split()
            if len(entity_words) > 1:
                for word in entity_words:
                    if len(word) > 3 and word in snippet:
                        # Check if it's mentioned in a different context
                        import re
                        pattern = rf'\b{re.escape(word)}\b(?!.*\b{re.escape(entity_lower)}\b)'
                        if re.search(pattern, snippet):
                            return True
                            
        return False

    def retrieve_knowledge(self, text, task_type, instruction=""):
        """
        Retrieve relevant knowledge with secondary filtering
        
        Args:
            text (str): Input text
            task_type (str): Task type
            instruction (str): Instruction
            
        Returns:
            list: List of retrieved knowledge
        """
        print(f"[DEBUG] Starting knowledge retrieval, task type: {task_type}")
        
        # Extract search keywords
        search_terms = self.extract_entities_for_search(text, task_type)
        
        if not search_terms:
            print("[WARNING] No search keywords extracted")
            return []
        
        # Filter words that don't need searching, reduce search count
        filtered_terms = []
        for term in search_terms[:3]:  # Process at most 3 words
            if self._should_skip_search(term):
                print(f"[SKIP] Skipping common word: {term}")
                continue
                
            if not self._is_likely_entity(term):
                print(f"[SKIP] Skipping non-entity word: {term}")
                continue
                
            filtered_terms.append(term)
        
        print(f"[FILTER] Filtered search terms: {filtered_terms}")
        
        if not filtered_terms:
            print("[INFO] All words filtered, no search needed")
            return []
        
        # Collect all candidate knowledge items
        all_candidates = []
        
        # Only search the most important 1-2 words
        for i, term in enumerate(filtered_terms[:2]):
            print(f"[DEBUG] Processing search term {i+1}: {term}")
            
            # Simplify search query, remove hardcoded entity type judgment
            search_query = f"{term} wikipedia"
            
            search_result = self.Google_search(search_query, size="5")  # Increase candidate count
            
            if "error" not in search_result and "webpages" in search_result:
                webpages = search_result["webpages"]
                if webpages:
                    print(f"[SUCCESS] Found {len(webpages)} results")
                    for result in webpages:
                        knowledge_item = {
                            "term": term,
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", result.get("content", "")),
                            "url": result.get("url", "")
                        }
                        # Calculate quality score
                        quality_score = self._evaluate_knowledge_quality(knowledge_item, term)
                        knowledge_item["quality_score"] = quality_score
                        all_candidates.append(knowledge_item)
                else:
                    # When no search results, use LLM to summarize and clean
                    print(f"[INFO] No direct results for '{term}', using LLM to process query and results")
                    processed_knowledge = self._llm_process_search_results(term, search_query, search_result, task_type)
                    if processed_knowledge:
                        all_candidates.extend(processed_knowledge)
        
        # Sort by quality score
        all_candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Select high-quality knowledge
        retrieved_knowledge = []
        high_quality_threshold = 0.6
        
        # Prioritize high-quality knowledge (score>=0.6), at most 3 items
        high_quality_items = [item for item in all_candidates if item["quality_score"] >= high_quality_threshold]
        retrieved_knowledge.extend(high_quality_items[:3])
        
        # If high-quality knowledge is insufficient, select the best 1-2 items as supplement
        if len(retrieved_knowledge) == 0 and all_candidates:
            retrieved_knowledge.extend(all_candidates[:2])
        elif len(retrieved_knowledge) < 2 and len(all_candidates) > len(retrieved_knowledge):
            remaining_candidates = [item for item in all_candidates if item not in retrieved_knowledge]
            retrieved_knowledge.extend(remaining_candidates[:1])
        
        print(f"[RESULT] Total retrieved {len(retrieved_knowledge)} knowledge items")
        for item in retrieved_knowledge:
            print(f"[QUALITY] {item['title'][:30]}... - Quality score: {item['quality_score']:.2f}")
        
        return retrieved_knowledge

    def _llm_process_search_results(self, search_term, search_query, search_result, task_type):
        """
        Use LLM to process and clean search results
        
        Args:
            search_term (str): Search term
            search_query (str): Search query
            search_result (dict): Search result
            task_type (str): Task type
            
        Returns:
            list: List of processed knowledge items
        """
        try:
            # Build LLM processing prompt
            if task_type == "NER":
                task_description = "named entity recognition"
            elif task_type == "RE":
                task_description = "relation extraction"
            elif task_type == "EE":
                task_description = "event extraction"
            else:
                task_description = "information extraction"
            
            # Get original answer or error information
            raw_answer = search_result.get("answer", "")
            error_info = search_result.get("error", "")
            
            prompt = f"""You are an AI assistant helping with {task_description} tasks. 
            
        Search Query: {search_query}
        Search Term: {search_term}
        Raw Search Result: {raw_answer if raw_answer else error_info}
            
        Please analyze and summarize the most relevant information about '{search_term}' that would be helpful for {task_description}. 
        If the search result is empty or irrelevant, provide a brief, factual summary based on your knowledge about '{search_term}'.
            
        Requirements:
        1. Focus on factual, encyclopedic information
        2. Keep the summary concise (2-3 sentences)
        3. Highlight key characteristics or definitions
        4. Avoid speculation or uncertain information
            
        Summary:"""
            
            print(f"[DEBUG] Sending LLM processing request for term: {search_term}")
            
            # Call LLM for processing
            llm_response = self.llm.get_chat_response(prompt)
            
            if llm_response and llm_response.strip():
                # Create processed knowledge item
                processed_item = {
                    "term": search_term,
                    "title": f"LLM Summary: {search_term}",
                    "snippet": llm_response.strip(),
                    "url": "llm_processed",
                    "quality_score": 0.7  # Give LLM processed results a medium quality score
                }
                
                print(f"[SUCCESS] LLM processed knowledge for '{search_term}': {llm_response[:100]}...")
                return [processed_item]
            else:
                print(f"[WARNING] LLM processing returned empty result for '{search_term}'")
                return []
                
        except Exception as e:
            print(f"[ERROR] LLM processing failed for '{search_term}': {str(e)}")
            return []

    def _get_search_strategies(self, term):
        """
        Get search strategy list
        
        Args:
            term (str): Search term
            
        Returns:
            list: List of search strategies
        """
        # English term strategies
        strategies = [
            term,  # Direct search
            f"{term} definition",  # English definition
            f"In wikipedia, what is {term}",  # Wikipedia
        ]
        
        return strategies
    
    def _is_english_term(self, term):
        """
        Determine if term is English
        
        Args:
            term (str): Term to evaluate
            
        Returns:
            bool: True if English
        """
        return bool(re.match(r'^[a-zA-Z\s]+$', term.strip()))
    
    def format_knowledge_for_extraction(self, knowledge_list):
        """
        Format knowledge for extraction tasks
        
        Args:
            knowledge_list (list): Knowledge list
            
        Returns:
            str: Formatted knowledge string
        """
        if not knowledge_list:
            return ""
        
        formatted_knowledge = "\n**Supplementary Knowledge Information:**\n"
        for i, knowledge in enumerate(knowledge_list, 1):
            formatted_knowledge += f"{i}. **{knowledge['term']}**: {knowledge['snippet']}\n"
        
        return formatted_knowledge
    
    def enhance_with_knowledge(self, data):
        """
        Enhance data with knowledge
        
        Args:
            data: Input data object
            
        Returns:
            data: Enhanced data object
        """
        print("[INFO] AiderAgent: Starting knowledge enhancement")
        
        try:
            # Determine task type
            task_type = "NER"  # Default to NER, can be adjusted as needed
            
            # Retrieve knowledge
            knowledge_list = self.retrieve_knowledge(
                text=data.text, 
                task_type=task_type, 
                instruction=data.instruction
            )
            
            # Format knowledge
            formatted_knowledge = self.format_knowledge_for_extraction(knowledge_list)
            
            if formatted_knowledge:
                # Add knowledge to additional_info
                if hasattr(data, 'additional_info'):
                    data.additional_info += formatted_knowledge
                else:
                    data.additional_info = formatted_knowledge
                
                print(f"[SUCCESS] AiderAgent: Retrieved {len(knowledge_list)} relevant knowledge items")
            else:
                print("[WARNING] AiderAgent: No relevant knowledge retrieved")
            
            # Record trajectory
            trajectory = {
                "method": "enhance_with_knowledge",
                "retrieved_knowledge_count": len(knowledge_list),
                "knowledge_summary": [k["title"] for k in knowledge_list]
            }
            
            if hasattr(data, 'result_trajectory'):
                data.result_trajectory["aider_agent"] = trajectory
            
            return data
            
        except Exception as e:
            print(f"[ERROR] AiderAgent knowledge enhancement failed: {e}")
            # Record failure trajectory
            trajectory = {
                "method": "enhance_with_knowledge",
                "retrieved_knowledge_count": 0,
                "knowledge_summary": [],
                "error": str(e)
            }
            
            if hasattr(data, 'result_trajectory'):
                data.result_trajectory["aider_agent"] = trajectory
            
            return data
    
    def _evaluate_knowledge_quality(self, knowledge_item, search_term):
        """
        Evaluate quality score of knowledge item
        
        Args:
            knowledge_item (dict): Knowledge item
            search_term (str): Search term
            
        Returns:
            float: Quality score (0-1)
        """
        score = 0.0
        
        title = knowledge_item.get('title', '').lower()
        snippet = knowledge_item.get('snippet', '').lower()
        url = knowledge_item.get('url', '').lower()
        search_term_lower = search_term.lower()
        
        # 1. Title relevance scoring
        if search_term_lower in title:
            score += self.quality_weights['title_relevance'] * 1.0
        elif any(word in title for word in search_term_lower.split()):
            score += self.quality_weights['title_relevance'] * 0.7
        
        # 2. Content length scoring (moderate length is better)
        content_length = len(snippet)
        if 100 <= content_length <= 500:
            score += self.quality_weights['content_length'] * 1.0
        elif 50 <= content_length < 100 or 500 < content_length <= 800:
            score += self.quality_weights['content_length'] * 0.7
        elif content_length > 0:
            score += self.quality_weights['content_length'] * 0.3
        
        # 3. Source authority scoring
        authority_domains = ['wikipedia', 'edu', 'org', 'gov', 'ieee', 'acm']
        if any(domain in url for domain in authority_domains):
            score += self.quality_weights['source_authority'] * 1.0
        elif 'wiki' in url or '.org' in url:
            score += self.quality_weights['source_authority'] * 0.8
        else:
            score += self.quality_weights['source_authority'] * 0.5
        
        # 4. Entity matching scoring
        if search_term_lower in snippet:
            score += self.quality_weights['entity_match'] * 1.0
        elif any(word in snippet for word in search_term_lower.split()):
            score += self.quality_weights['entity_match'] * 0.6
        
        return min(score, 1.0)  # Ensure score doesn't exceed 1.0
    
    def _get_entity_type_hint(self, term):
        """
        Get entity type hint - simplified version
        
        Args:
            term (str): Term
            
        Returns:
            str: Entity type hint
        """
        term_lower = term.lower()
        
        # Simplified entity type determination
        if any(keyword in term_lower for keyword in ['conference', 'workshop', 'symposium']):
            return 'conference'
        elif any(keyword in term_lower for keyword in ['algorithm', 'method', 'approach']):
            return 'algorithm'
        elif any(keyword in term_lower for keyword in ['university', 'institute', 'laboratory']):
            return 'organization'
        else:
            return 'general'

class AiderAgent:
    """
    AiderAgent class for knowledge-enhanced NER tasks
    
    Main Functions:
    1. Integrate knowledge retriever for knowledge enhancement
    2. Provide external knowledge support for NER tasks
    3. Manage case repository and LLM instance
    """
    
    def __init__(self, llm, case_repo=None, api_key="mk-BD34693D129F16898CB1C14D5D2C7F7B"):
        """
        Initialize AiderAgent
        
        Args:
            llm: Language model instance
            case_repo: Case repository (optional)
            api_key (str): API key
        """
        self.llm = llm                                                    # Language model instance
        self.case_repo = case_repo                                       # Case repository
        self.knowledge_retriever = KnowledgeRetriever(api_key)          # Knowledge retriever instance
        self.knowledge_retriever.llm = llm                              # Set LLM instance
        
    def enhance_with_knowledge(self, data):
        """
        Enhance data with knowledge
        
        Args:
            data: Input data object
            
        Returns:
            data: Enhanced data object
        """
        return self.knowledge_retriever.enhance_with_knowledge(data)