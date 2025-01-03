import json
import requests
from bs4 import BeautifulSoup
import re
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def get_leetcode_problems(skip=0, limit=50, category="algorithms"):
    """
    Fetch paginated list of LeetCode problems.
    
    Args:
        skip (int): Number of problems to skip (for pagination)
        limit (int): Number of problems per page
        category (str): Problem category ("algorithms", "database", "shell", etc.)
    """
    url = 'https://leetcode.com/graphql'
    
    # GraphQL query for problem list
    payload = {
        "query": """
            query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
                problemsetQuestionList: questionList(
                    categorySlug: $categorySlug
                    limit: $limit
                    skip: $skip
                    filters: $filters
                ) {
                    total: totalNum
                    questions: data {
                        titleSlug
                        title
                        difficulty
                        isPaidOnly
                        frontendQuestionId: questionFrontendId
                    }
                }
            }
        """,
        "variables": {
            "categorySlug": category,
            "limit": limit,
            "skip": skip,
            "filters": {}
        }
    }
    
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    }
    
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    
    if 'errors' in data:
        logger.error(f"GraphQL Error: {data['errors']}")
        raise Exception(f"GraphQL Error: {data['errors']}")
        
    data = data['data']['problemsetQuestionList']
    
    # Format the response
    problems = []
    for q in data['questions']:
        problems.append({
            'id': q['frontendQuestionId'],
            'title': q['title'],
            'titleSlug': q['titleSlug'],
            'url': f"https://leetcode.com/problems/{q['titleSlug']}/",
            'difficulty': q['difficulty'],
            'paid_only': q['isPaidOnly']
        })
    
    return {
        'total': data['total'],
        'problems': problems,
        'page_info': {
            'skip': skip,
            'limit': limit,
            'has_more': skip + limit < data['total']
        }
    }

def parse_test_cases(test_cases_str, num_params):
    """
    Parse the exampleTestcases string into list of test cases
    
    Args:
        test_cases_str: String from exampleTestcases field
        num_params: Number of parameters for the function
        
    Returns:
        List of test cases, where each test case is a list of parameters
    """
    # Split by newline and remove empty strings
    values = [v for v in test_cases_str.split('\n') if v]
    
    # Group values into test cases based on number of parameters
    test_cases = []
    for i in range(0, len(values), num_params):
        test_case = values[i:i + num_params]
        
        # Parse each parameter
        parsed_params = []
        for param in test_case:
            try:
                # Handle quoted strings
                if param.startswith('"') and param.endswith('"'):
                    parsed_params.append(param[1:-1])  # Remove quotes
                else:
                    # Try to parse as JSON/number
                    parsed_params.append(json.loads(param))
            except json.JSONDecodeError:
                # If not valid JSON, keep as is
                parsed_params.append(param)
                
        test_cases.append(parsed_params)
    
    return test_cases

def parse_input_value(value_str, param_type):
    """
    Parse a string value according to its parameter type
    
    Args:
        value_str: String representation of the value
        param_type: Type of the parameter ('integer', 'string', etc.)
        
    Returns:
        Parsed value in appropriate type
    """
    value_str = value_str.strip()
    try:
        if param_type == 'integer':
            # Handle integer arrays
            if value_str.startswith('['):
                return json.loads(value_str)
            return int(value_str)
        elif param_type == 'string':
            # Remove quotes if present
            if value_str.startswith('"') and value_str.endswith('"'):
                return value_str[1:-1]
            elif value_str.startswith("'") and value_str.endswith("'"):
                return value_str[1:-1]
            return value_str
        elif param_type in ['list', 'array']:
            return json.loads(value_str)
        else:
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                return value_str
    except (ValueError, json.JSONDecodeError):
        return value_str


def extract_test_cases_using_llm(gemini_parser, description, function_signature):
    """
    Extract test cases using Gemini LLM
    
    Args:
        gemini_parser: Instance of LeetCodeGeminiParser
        description: Problem description text containing examples
        function_signature: Dictionary containing function name, parameters, and return type
        
    Returns:
        List of dictionaries containing structured input parameters and expected output
    """
    # Get raw test cases from Gemini
    raw_test_cases = gemini_parser.parse_with_gemini(description)
    
    # Convert raw test cases to our structured format
    test_cases = []
    params = function_signature.get('params', [])
    return_type = function_signature.get('return', {}).get('type', 'string')
    
    for raw_case in raw_test_cases:
        # Structure input parameters
        structured_inputs = []
        raw_inputs = raw_case['input']
        
        for param in params:
            param_name = param.get('name')
            param_type = param.get('type', 'string')
            
            if param_name in raw_inputs:
                structured_inputs.append({
                    'name': param_name,
                    'type': param_type,
                    'value': raw_inputs[param_name]
                })
        
        test_cases.append({
            'inputs': structured_inputs,
            'expected_output': {
                'type': return_type,
                'value': raw_case['output']
            }
        })
    
    return test_cases

def extract_test_cases_from_description(description, function_signature):
    """
    Extract test cases and their expected outputs from problem description
    
    Args:
        description: Problem description text containing examples
        function_signature: Dictionary containing function name, parameters, and return type
        
    Returns:
        List of dictionaries containing structured input parameters and expected output
    """
    # First, split the description into sections
    sections = re.split(r'\n\s*(?=Example \d+:|\nConstraints:)', description)
    
    test_cases = []
    params = function_signature.get('params', [])
    return_type = function_signature.get('return', {}).get('type', 'string')
    
    # Process each example section
    for section in sections:
        if not section.strip().startswith('Example'):
            continue
            
        # Extract input and output from the example section
        input_match = re.search(r'Input:?\s*(?:s\s*=\s*)?([^:\n]*?)(?=\s*\n|$)', section)
        output_match = re.search(r'Output:?\s*([^:\n]*?)(?=\s*(?:\n|$))', section)
        
        if not input_match or not output_match:
            continue
            
        input_str = input_match.group(1).strip()
        output_str = output_match.group(1).strip()
        
        # Clean up strings
        input_str = re.sub(r'\s+', ' ', input_str)
        output_str = re.sub(r'\s+', ' ', output_str)
        
        # For single parameter cases, treat the entire input as one value
        if len(params) == 1:
            param = params[0]
            param_type = param.get('type', 'string')
            structured_inputs = [{
                'name': param.get('name', 'param0'),
                'type': param_type,
                'value': parse_input_value(input_str, param_type)
            }]
        else:
            # Extract parameter values
            # Split by comma but preserve arrays and quoted strings
            param_values = []
            current = ''
            bracket_count = 0
            quote_char = None
            
            for char in input_str:
                if char in ['"', "'"] and not quote_char:
                    quote_char = char
                elif char == quote_char:
                    quote_char = None
                elif char == '[' and not quote_char:
                    bracket_count += 1
                elif char == ']' and not quote_char:
                    bracket_count -= 1
                elif char == ',' and bracket_count == 0 and not quote_char:
                    if '=' in current:  # Handle named parameters
                        param_values.append(current.split('=')[1].strip())
                    else:
                        param_values.append(current.strip())
                    current = ''
                    continue
                current += char
            
            if current:
                if '=' in current:  # Handle named parameters
                    param_values.append(current.split('=')[1].strip())
                else:
                    param_values.append(current.strip())
            
            # Structure input parameters
            structured_inputs = []
            for i, (param, value) in enumerate(zip(params, param_values)):
                param_type = param.get('type', 'string')
                structured_inputs.append({
                    'name': param.get('name', f'param{i}'),
                    'type': param_type,
                    'value': parse_input_value(value, param_type)
                })
        
        test_cases.append({
            'inputs': structured_inputs,
            'expected_output': {
                'type': return_type,
                'value': parse_input_value(output_str, return_type)
            }
        })
    
    return test_cases

def scrape_leetcode_problem(url, gemini_parser=None):
    """Scrape problem details from LeetCode
    
    Args:
        url: LeetCode problem URL
        gemini_parser: Optional LeetCodeGeminiParser instance for LLM-based test case extraction
    """
    # Extract problem title slug from URL
    slug = url.split('problems/')[1].rstrip('/')
    
    # GraphQL endpoint
    api_url = 'https://leetcode.com/graphql'
    
    # GraphQL query to get problem details
    query = '''
    query getQuestionDetail($titleSlug: String!) {
        question(titleSlug: $titleSlug) {
            title
            content
            difficulty
            exampleTestcases
            sampleTestCase
            metaData
            codeDefinition
        }
    }
    '''
    
    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    }
    
    # Query variables
    variables = {'titleSlug': slug}
    
    # Make the request
    response = requests.post(
        api_url,
        headers=headers,
        json={'query': query, 'variables': variables}
    )
    
    # Parse response
    data = response.json()['data']['question']
    
    # Clean HTML content
    soup = BeautifulSoup(data['content'], 'html.parser')
    description = soup.get_text()
    
    # Parse metadata for function signature
    metadata = json.loads(data['metaData'])
    num_params = len(metadata.get('params', {}))
    
    # Get test cases - use LLM if available, otherwise fallback to regex
    if gemini_parser:
        example_test_cases = extract_test_cases_using_llm(gemini_parser, description, metadata)
    else:
        example_test_cases = extract_test_cases_from_description(description, metadata)
    
    parsed_test_cases = parse_test_cases(data['exampleTestcases'], num_params)

    # Get code definition in Python
    code_definitions = json.loads(data['codeDefinition'])
    defaultCode = ''
    for definition in code_definitions:
        if definition['value'] == 'python3':
            defaultCode = definition['defaultCode']
    
    result = {
        'url': url,
        'title': data['title'],
        'difficulty': data['difficulty'],
        'description': description,
        'function_signature': {
            'name': metadata.get('name', ''),
            'params': metadata.get('params', {}),
            'return': metadata.get('return', {})
        },
        'default_code': defaultCode,
        'test_cases': parsed_test_cases,
        'test_cases_with_answers': example_test_cases
    }
    return result

def save_problem(result, data_dir):
    """
    Save problem data to a JSON file
    
    Args:
        result: Problem data dictionary
        data_dir: Directory to save the file
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Clean title for filename
    filename = f"{result['id']}_{result['titleSlug'].lower().replace(' ', '_')}.json" if 'titleSlug' in result and 'id' in result else '_'.join(result['title'].lower().split()) + '.json'
    
    # Save to file
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved problem data to {file_path}")

def scrape_all_leetcode_problems(data_dir, limit=None, skip=0, gemini_parser=None, batch_size=50):
    """
    Scrape all LeetCode problems with pagination support
    
    Args:
        data_dir: Directory to save problem data
        limit: Maximum number of problems to scrape (None for all)
        skip: Number of problems to skip
        gemini_parser: Optional LeetCodeGeminiParser instance
        batch_size: Number of problems per page
    """
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Starting to scrape LeetCode problems to {data_dir}")
    
    total_scraped = 0
    while True:
        # Get batch of problems
        result = get_leetcode_problems(skip=skip + total_scraped, limit=batch_size)
        problems = result['problems']
        
        if not problems:
            break
            
        logger.info(f"Scraping problems {skip + total_scraped + 1} to {skip + total_scraped + len(problems)}...")
        
        for problem in problems:
            # Skip paid problems
            # if problem['paid_only']:
            #     logger.info(f"Skipping paid problem: {problem['title']}")
            #     continue
                
            # Check if we've reached the limit
            if limit and total_scraped >= limit:
                logger.info(f"Reached limit of {limit} problems")
                return total_scraped
                
            try:
                # Scrape detailed problem data
                logger.info(f"Scraping {problem['id']}. {problem['title']}...")
                detailed_problem = scrape_leetcode_problem(problem['url'], gemini_parser)
                # Add problem metadata
                detailed_problem.update({
                    'id': problem['id'],
                    'titleSlug': problem['titleSlug'],
                    'paid_only': problem['paid_only']
                })
                
                # Save to file
                save_problem(detailed_problem, data_dir)
                total_scraped += 1

                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error scraping problem {problem['title']}: {str(e)}")
                continue
        
        # Check if we've processed all problems
        if not result['page_info']['has_more']:
            break
            
        # Optional delay between batches
        time.sleep(10)  # Be nice to LeetCode's servers
    
    logger.info(f"Finished scraping {total_scraped} problems")
    return total_scraped
