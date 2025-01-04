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

def solve_problem(problem_id: int, args, solver: LLMCodeSolver, logger: logging.Logger):
    """Solve a single problem"""
    # Find problem files
    problem_files = [f for f in os.listdir(args.data_dir) if f.startswith(f'{problem_id}_')]
    example_files = [f for f in os.listdir(args.data_dir) if f.startswith(f'{args.example_id}_')]
    
    if not example_files:
        logger.error(f"Example problem {args.example_id} not found in {args.data_dir}")
        return False
    if not problem_files:
        logger.error(f"Problem {problem_id} not found in {args.data_dir}")
        return False
    
    try:
        # Load problems
        example_problem = load_problem(os.path.join(args.data_dir, example_files[0]))
        target_problem = load_problem(os.path.join(args.data_dir, problem_files[0]))
        
        # Generate solution
        logger.info(f"Solving problem {problem_id} using example {args.example_id}")
        solution = solver.solve(example_problem, target_problem, model_name=args.model)
        logger.info(f"\nGenerated Solution for problem {problem_id}:\n{solution}")
        
        # Save solution if requested
        if not args.no_save:
            filepath = save_solution(solution, target_problem, args.save_dir)
            logger.info(f"Saved solution to: {filepath}")
        
        # Submit if requested
        if args.submit:
            leetcode_session = os.getenv("LEETCODE_SESSION")
            logger.info(f"Using LeetCode session cookie: {leetcode_session}")
            
            logger.info(f"Submitting solution for problem {problem_id}")
            submitter = LeetCodeSubmitter(max_retries=3, backoff_factor=0.5)
            result = submitter.submit_leetcode(target_problem, solution, leetcode_session)
            logger.info(f"Submission result for problem {problem_id}: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error solving problem {problem_id}: {str(e)}")
        return False

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
        '--start',
        type=int,
        required=True,
        help='ID of the first problem to solve'
    )
    parser.add_argument(
        '--end',
        type=int,
        help='ID of the last problem to solve (optional, if not provided only solves start problem)'
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
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue solving next problem if current one fails'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize LLM manager and solver
        llm_manager = LLMManager()
        llm_manager.setup_default_models(google_api_key=os.getenv("GOOGLE_API_KEY"))
        solver = LLMCodeSolver(llm_manager)
        
        # Determine problem range
        start_id = args.start
        end_id = args.end if args.end else start_id
        
        if end_id < start_id:
            raise ValueError("End ID must be greater than or equal to Start ID")
        
        # Track success/failure
        results = {
            'total': end_id - start_id + 1,
            'succeeded': 0,
            'failed': 0
        }
        
        # Solve problems in range
        for problem_id in range(start_id, end_id + 1):
            success = solve_problem(problem_id, args, solver, logger)
            if success:
                results['succeeded'] += 1
            else:
                results['failed'] += 1
                if not args.continue_on_error:
                    logger.error("Stopping due to error. Use --continue-on-error to continue on failures.")
                    break
        
        # Print summary
        logger.info("\nSummary:")
        logger.info(f"Total problems: {results['total']}")
        logger.info(f"Succeeded: {results['succeeded']}")
        logger.info(f"Failed: {results['failed']}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
