import google.generativeai as genai
import json
import requests
import re
import typing_extensions as typing
from typing import Any, Dict

from jinja2 import Environment
import os


_parse_test_str = """
You are a LeetCode problem parser. Given a LeetCode problem content, extract and structure its test cases (both inputs and expected outputs)
Format your response as a valid JSONL with these keys: input, output. Each input is a dictionary of key-value pairs, where the key is the parameter name and the value is the parameter value.

Make sure to properly escape strings.

Format the output as JSONL with fields: input, output.
Do not include any other text or formatting.

Problem content:
{{ sample_1 }}

Expected output:
{{ result_1 }}

Problem content:
{{ content }}

Expected output:
"""

sample_1_str = """
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"
Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);

 
Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I

Example 3:

Input: s = "A", numRows = 1
Output: "A"

 
Constraints:

1 <= s.length <= 1000
s consists of English letters (lower-case and upper-case), ',' and '.'.
1 <= numRows <= 1000
"""

result_1_str = """
{"input": {"s": "PAYPALISHIRING", "numRows": 3}, "output": "PAHNAPLSIIGYIR"}
{"input": {"s": "PAYPALISHIRING", "numRows": 4}, "output": "PINALSIGYAHRPI"}
{"input": {"s": "A", "numRows": 1}, "output": "A"}
"""

sample_2_str = """
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
Given a string s, return true if it is a palindrome, or false otherwise.
 
Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.

 
Constraints:

1 <= s.length <= 2 * 105
s consists only of printable ASCII characters.
"""


class LeetCodeGeminiParser:
    def __init__(self, api_key: str):
        # Initialize Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.env = Environment()

    
    def clean_gemini_response(self, response_text):
        """Clean the Gemini API response by removing markdown formatting and parsing JSON."""
        # Remove ```jsonl or ```json markers if present
        cleaned_text = re.sub(r'^```(?:jsonl|json)\n', '', response_text)
        cleaned_text = re.sub(r'\n```$', '', cleaned_text)
        
        # Split into lines and parse each valid JSON line
        result = []
        for line in cleaned_text.strip().split('\n'):
            try:
                # Skip empty lines
                if not line.strip():
                    continue
                parsed_item = json.loads(line)
                result.append(parsed_item)
            except json.JSONDecodeError as e:
                logging.error(f"Couldn't parse line: {line}, Error: {e}")
        
        return result

        
    def parse_with_gemini(self, content: str) -> Dict[str, Any]:
        """Use Gemini to parse problem content into structured format"""
        prompt = self.env.from_string(_parse_test_str).render(
            content=content, sample_1=sample_1_str, result_1=result_1_str)        
        response = self.model.generate_content(prompt)
        cleaned_response = self.clean_gemini_response(response.text)
        return cleaned_response
