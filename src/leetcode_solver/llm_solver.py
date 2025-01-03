import os
import json
import re
import requests
import time
import logging
from typing import Any, Dict
from jinja2 import Environment
from dotenv import load_dotenv
from .llm_manager import LLMManager, OutputMode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


load_dotenv()
logger = logging.getLogger(__name__)

system_prompt_str = """
You are an professional coder in solving computer science alogorithm problems. 
* Your favorite language is Python, and mostly conform to PEP8 and Python3 best practises.
* You write professional style code that maintains readability and is easy to read and understand.
* You are a good problem solver, you write code that both clean and efficient.
* You always optimize for time and memory, and you are good at using data structures and algorithms to solve problems.
"""

task_prompt_str = """
Your Input will be a Leetcode problem statement, as well as its default code template.

Your output should be a code in Python3 that solves the problem.
* The code should be self-explanatory and easy to read, use comments if necessary.
* The code should be clean, well-structured, efficient and fast.
* ALWAYS start the code with provided default code template.
* DO NOT INCLUDE ANY OTHER TEXT OR FORMATTING. THE CODE SHOULD BE A VALID RUNNABLE PYTHON3 CODE.

Please provide a solution that:
1. Is well-commented with clear explanations
2. Handles all edge cases from the examples
3. Has optimal time and space complexity
4. Follows Python best practices
"""

format_question_str = """
Problem Statement:
{{ question["description"] }}

Code Template:
{{ question["default_code"] }}
"""

twosum_answer = """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numMap = {}
        n = len(nums)

        for i in range(n):
            complement = target - nums[i]
            if complement in numMap:
                return [numMap[complement], i]
            numMap[nums[i]] = i

        return []  # No solution found
"""

class LeetCodeSubmitter:
    def __init__(self, max_retries=3, backoff_factor=0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session()
    
    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def submit_leetcode(self, question, code, session_cookie, max_attempts=3):
        headers = {
            'Cookie': f'LEETCODE_SESSION={session_cookie}',
            'Content-Type': 'application/json',
            'Referer': f'https://leetcode.com/problems/{question["titleSlug"]}/',
            'Origin': 'https://leetcode.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        
        mutation = '''
        mutation submitCode($titleSlug: String!, $code: String!, $lang: String!) {
            submitCode(titleSlug: $titleSlug, code: $code, lang: $lang) {
                statusCode
                statusMessage
                submissionId
                __typename
            }
        }
        '''
        
        for attempt in range(max_attempts):
            try:
                # Submit solution
                response = self.session.post(
                    'https://leetcode.com/graphql',
                    json={
                        'query': mutation,
                        'variables': {
                            'titleSlug': question['titleSlug'],
                            'code': code,
                            'lang': 'python3'
                        }
                    },
                    headers=headers,
                    timeout=30  # Longer timeout for submission
                )
                
                if response.status_code == 499:
                    logger.warning(f"Got 499 error on attempt {attempt + 1}, retrying...")
                    time.sleep((2 ** attempt) * self.backoff_factor)
                    continue
                    
                response.raise_for_status()
                result = response.json()
                
                # Check for specific LeetCode errors in response
                if 'errors' in result:
                    raise Exception(f"LeetCode API error: {result['errors']}")
                
                return result
            
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to submit code after {max_attempts} attempts: {str(e)}")
                wait_time = (2 ** attempt) * self.backoff_factor
                time.sleep(wait_time)
                

class LLMCodeSolver:
    def __init__(self, llm_manager: LLMManager):
        """Initialize solver with LLM manager"""
        self.env = Environment()
        self.llm_manager = llm_manager
        logger.info("Initialized LLM code solver")
    
    
    def solve(self, example_question: Dict[str, Any], target_question: Dict[str, Any], model_name: str = "gemini-flash") -> str:
        """Solve target question using example question as reference"""
        # Generate prompts
        prompts = self.prompt(example_question, target_question)
        
        # Generate solution using LLM
        try:
            solution = self.llm_manager.generate(
                prompts=prompts,
                model_name=model_name,
                output_mode=OutputMode.RAW
            )
            return _clean_response(solution)
        except Exception as e:
            logger.error(f"Error generating solution: {str(e)}")
            raise
    
    def prompt(self, example_question: Dict[str, Any], target_question: Dict[str, Any]) -> list:
        """Generate prompts for solving a problem using example question"""
        # Format questions using template
        sample_1_question = self.env.from_string(format_question_str).render(
            question=example_question
        )
        question = self.env.from_string(format_question_str).render(
            question=target_question
        )

        # Create prompt sequence
        prompts = [
            {"role": "system", "content": system_prompt_str},
            {"role": "user", "content": task_prompt_str},
            {"role": "user", "content": sample_1_question},
            {"role": "assistant", "content": twosum_answer},
            {"role": "user", "content": question},
        ]

        formatted_prompts = "\n".join([f"{p['role']}: {p['content']}" for p in prompts])
        logger.debug(f"Generated prompts for problem solving: {formatted_prompts}")

        return prompts



def load_problem(filename: str) -> Dict[str, Any]:
    """Load problem data from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def _clean_response(response_text):
    """Clean the response by removing markdown formatting."""
    # Remove ```python3 or ```python markers if present
    cleaned_text = re.sub(r'^```(?:python3|python)\n', '', response_text)
    cleaned_text = re.sub(r'\n```$', '', cleaned_text)
    return cleaned_text


def main():
    # Get data directory
    data_dir = "/Users/hxx/projects/leetcode-solver/data"
    
    # Load problems
    twosum = load_problem(os.path.join(data_dir, "1_two-sum.json"))
    add_two_numbers = load_problem(os.path.join(data_dir, "2_add-two-numbers.json"))
    
    # Initialize LLM manager and solver
    llm_manager = LLMManager()
    llm_manager.setup_default_models(google_api_key=os.getenv("GOOGLE_API_KEY"))
    solver = LLMCodeSolver(llm_manager)
    
    # Generate solution
    try:
        solution = solver.solve(twosum, add_two_numbers)
        print("\nGenerated Solution:")
        print(solution)
    except Exception as e:
        logger.error(f"Failed to solve problem: {str(e)}")

if __name__ == "__main__":
    main()