#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from leetcode_solver.scraper import scrape_all_leetcode_problems
from leetcode_solver.gemini_parser import LeetCodeGeminiParser

load_dotenv()

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
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

if __name__ == "__main__":
    main()