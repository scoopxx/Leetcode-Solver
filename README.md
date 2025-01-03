# LeetCode Solver

A tool to scrape LeetCode problems and generate solutions using LLMs.

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Required environment variables:
- `GOOGLE_API_KEY`: Your Google API key for using Gemini
- `LEETCODE_SESSION`: Your LeetCode session cookie for submitting solutions

## Usage

### 1. Fetch LeetCode Problems
Scrape problems from LeetCode and store them locally:

```bash
# Scrape 50 problems starting from the beginning
pdm run python src/scrape_leetcode.py --limit 50

# Skip first 100 problems and scrape next 50
pdm run python src/scrape_leetcode.py --skip 100 --limit 50

# Scrape without using Gemini for test case extraction
pdm run python src/scrape_leetcode.py --no-gemini
```

Problems will be saved to `data/problems/` directory.

### 2. Generate and Submit Solutions
Generate solutions using LLM and optionally submit them to LeetCode:

```bash
# Generate solution for problem #50
pdm run python src/solve_leetcode.py --problem-id 50

# Save solution to a custom directory
pdm run python src/solve_leetcode.py --problem-id 50 --save-dir ./my_solutions

# Generate without saving to file
pdm run python src/solve_leetcode.py --problem-id 50 --no-save

# Generate and submit to LeetCode
# TODO, the submission is not working for now.
pdm run python src/solve_leetcode.py --problem-id 50 --submit
```

Solutions will be saved to `data/solutions/` directory by default.

## Directory Structure

```
leetcode-solver/
├── data/
│   ├── problems/     # Scraped problem data
│   └── solutions/    # Generated solutions
├── src/
│   ├── leetcode_solver/
│   │   ├── gemini_parser.py
│   │   ├── llm_manager.py
│   │   ├── llm_solver.py
│   │   └── scraper.py
│   ├── scrape_leetcode.py
│   └── solve_leetcode.py
├── .env
└── README.md
```

## Features

- Scrape LeetCode problems with test cases
- Use Gemini for enhanced test case extraction
- Generate solutions using LLM with example-based learning
- Save solutions with problem descriptions and metadata
- Submit solutions directly to LeetCode
- Configurable LLM models and parameters