#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
from dotenv import load_dotenv
from leetcode_solver.scraper import scrape_all_leetcode_problems
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

def prepare_data():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='LeetCode Problem Scraper')
    parser.add_argument(
        '--data-dir',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        help='Directory to save scraped problems (default: ../data)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Maximum number of problems to scrape (default: no limit)'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Number of problems to skip (default: 0)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of problems to fetch per batch (default: 50)'
    )
    parser.add_argument(
        '--no-gemini',
        action='store_true',
        help='Disable Gemini parser for test case extraction'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize Gemini parser if enabled and API key is available
    gemini_parser = None
    if not args.no_gemini:
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            from leetcode_solver.gemini_parser import LeetCodeGeminiParser
            gemini_parser = LeetCodeGeminiParser(api_key)
            logger.info("Initialized Gemini parser")
        else:
            logger.warning("GOOGLE_API_KEY not found. Gemini parser will be disabled.")
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    try:
        # Start scraping
        total_scraped = scrape_all_leetcode_problems(
            data_dir=args.data_dir,
            limit=args.limit,
            skip=args.skip,
            gemini_parser=gemini_parser,
            batch_size=args.batch_size
        )
        logger.info(f"Successfully scraped {total_scraped} problems")
        
    except KeyboardInterrupt:
        logger.info("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        sys.exit(1)

def solve_question():
    setup_logging()
    logger = logging.getLogger(__name__)
    data_dir = "/Users/hxx/projects/leetcode-solver/data"
    
    # Load problems
    twosum = load_problem(os.path.join(data_dir, "1_two-sum.json"))
    add_two_numbers = load_problem(os.path.join(data_dir, "2_add-two-numbers.json"))
    
    # Initialize LLM manager
    llm_manager = LLMManager()
    llm_manager.setup_default_models(google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize solver with manager
    solver = LLMCodeSolver(llm_manager)
    
    # Generate solution
    try:
        proposed_solution = solver.solve(twosum, add_two_numbers)
        print("\nGenerated Solution:")
        print(proposed_solution)
    except Exception as e:
        logger.error(f"Failed to solve problem: {str(e)}")

    # Submit solution
    submitter = LeetCodeSubmitter(max_retries=3, backoff_factor=0.5)
    session_cookie = submitter.get_leetcode_session(
        os.getenv("LEETCODE_USERNAME"), 
        os.getenv("LEETCODE_PASSWORD"))
    result = submitter.submit_leetcode(add_two_numbers['id'], proposed_solution, session_cookie)
    logger.info(f"Submission result: {result}")


if __name__ == "__main__":
    solve_question()