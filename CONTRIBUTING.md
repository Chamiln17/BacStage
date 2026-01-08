# Contributing to SIC

Thank you for your interest in contributing to the Algerian Bac Educational Video Engagement Analysis project!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SIC
   ```

2. **Install uv** (if not already installed)
   ```bash
   pip install uv
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Coding Standards

### Python Style

- **Formatting**: Use `black` for code formatting (88 char line length)
  ```bash
  uv run black src/ tests/
  ```

- **Linting**: Use `ruff` for linting
  ```bash
  uv run ruff check src/ tests/
  ```

- **Type Checking**: Use `mypy` for type checking
  ```bash
  uv run mypy src/
  ```

### Code Requirements

1. **Type Hints**: All functions in `src/` must have type hints
   ```python
   def process_data(input_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
       """Process data from input file."""
       ...
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def function_name(param1: str, param2: int) -> bool:
       """
       Brief description of function.
       
       Longer description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ValueError: When this error occurs
       """
       ...
   ```

3. **Path Handling**: Use `pathlib.Path` for all file operations
   ```python
   from pathlib import Path
   
   # Good
   data_dir = Path("data") / "raw"
   
   # Bad
   data_dir = "data/raw"
   ```

4. **Logging**: Use `logging` module instead of `print` in production code
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   logger.info("Processing started")
   ```

5. **Environment Variables**: Use `.env` file for configuration
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   api_key = os.getenv("YOUTUBE_API_KEY")
   ```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test
uv run pytest tests/test_youtube_collector.py::TestYouTubeCollector::test_init_valid_api_key
```

### Writing Tests

1. **Location**: Tests go in `tests/` directory
2. **Naming**: Test files must start with `test_`
3. **Fixtures**: Use pytest fixtures in `tests/conftest.py`
4. **Mocking**: Mock external API calls

Example:
```python
from unittest.mock import Mock, patch
import pytest

def test_function_success(sample_data: pd.DataFrame) -> None:
    """Test successful function execution."""
    result = my_function(sample_data)
    assert result is not None
    assert len(result) > 0

@patch('module.external_api_call')
def test_with_mocked_api(mock_api: Mock) -> None:
    """Test with mocked external API."""
    mock_api.return_value = {'status': 'success'}
    result = function_using_api()
    assert result['status'] == 'success'
```

## Project Structure

Follow the data science project structure:

```
SIC/
├── data/              # Data files (gitignored)
├── notebooks/         # Jupyter notebooks (exploration only)
├── src/               # Production code
│   ├── data/         # Data collection scripts
│   ├── features/     # Feature engineering
│   ├── models/       # Model training
│   └── visualization/# Visualization scripts
└── tests/            # Unit tests
```

**Rules:**
- Notebooks are for exploration only
- Refactor working code from notebooks into `src/`
- All production logic must have tests

## Commit Messages

Use clear, descriptive commit messages:

```
Good:
- "Add feature engineering for temporal patterns"
- "Fix quota tracking bug in YouTube collector"
- "Update README with new installation instructions"

Bad:
- "update"
- "fix bug"
- "changes"
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `uv run pytest`
4. Run linting: `uv run ruff check src/ tests/`
5. Format code: `uv run black src/ tests/`
6. Commit changes with descriptive message
7. Push to your fork
8. Create pull request with description of changes

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.
