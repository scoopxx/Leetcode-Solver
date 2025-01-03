#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from leetcode_solver.llm_solver import LLMCodeSolver, load_problem, LeetCodeSubmitter
from leetcode_solver.llm_manager import LLMManager

load_dotenv()

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def save_solution(solution: str, target_problem: dict, save_dir: str):
    """Save solution to a Python file"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename from problem title slug
    filename = f"{target_problem['id']}_{target_problem['titleSlug']}.py"
    filepath = os.path.join(save_dir, filename)
    
    # Add problem description as docstring
    header = f'''"""
{target_problem['title']}
{'-' * len(target_problem['title'])}

Difficulty: {target_problem['difficulty']}

{target_problem['description']}

LeetCode: https://leetcode.com/problems/{target_problem['titleSlug']}/
"""

'''
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + solution)
    
    return filepath

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='LeetCode Problem Solver')
    parser.add_argument(
        '--data-dir',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/problems'),
        help='Directory containing problem data (default: ../data/problems)'
    )
    parser.add_argument(
        '--save-dir',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/solutions'),
        help='Directory to save generated solutions (default: ../data/solutions)'
    )
    parser.add_argument(
        '--example-id',
        type=int,
        default=1,
        help='ID of the example problem to use'
    )
    parser.add_argument(
        '--problem-id',
        type=int,
        required=True,
        help='ID of the problem to solve'
    )
    parser.add_argument(
        '--model',
        default='gemini-flash',
        help='LLM model to use (default: gemini-flash)'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit solution to LeetCode'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the solution to a file'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Find problem files
        example_files = [f for f in os.listdir(args.data_dir) if f.startswith(f'{args.example_id}_')]
        problem_files = [f for f in os.listdir(args.data_dir) if f.startswith(f'{args.problem_id}_')]
        
        if not example_files:
            raise ValueError(f"Example problem {args.example_id} not found in {args.data_dir}")
        if not problem_files:
            raise ValueError(f"Problem {args.problem_id} not found in {args.data_dir}")
        
        # Load problems
        example_problem = load_problem(os.path.join(args.data_dir, example_files[0]))
        target_problem = load_problem(os.path.join(args.data_dir, problem_files[0]))
        
        # Initialize LLM manager and solver
        llm_manager = LLMManager()
        llm_manager.setup_default_models(google_api_key=os.getenv("GOOGLE_API_KEY"))
        solver = LLMCodeSolver(llm_manager)
        
        # Generate solution
        logger.info(f"Solving problem {args.problem_id} using example {args.example_id}")
        solution = solver.solve(example_problem, target_problem, model_name=args.model)
        logger.info(f"\nGenerated Solution:\n {solution}")
        
        # Save solution if requested
        if not args.no_save:
            filepath = save_solution(solution, target_problem, args.save_dir)
            logger.info(f"Saved solution to: {filepath}")
        
        # Submit if requested
        if args.submit:
            leetcode_session = os.getenv("LEETCODE_SESSION")
            logger.info(f"Using LeetCode session cookie: {leetcode_session}")
            
            logger.info("Submitting solution to LeetCode")
            submitter = LeetCodeSubmitter(max_retries=3, backoff_factor=0.5)
            result = submitter.submit_leetcode(target_problem, solution, leetcode_session)
            logger.info(f"Submission result: {result}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
