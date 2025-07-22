# Tests

This directory contains mainly integration tests for the `langchain-gradientai` package.

## Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/) for dependency management (recommended)
- DigitalOcean Model API key

## Environment Variables

Tests require the `DIGITALOCEAN_INFERENCE_KEY` environment variable to be set.

1. Login to DigitalOcean Cloud console
2. Go to the **GradienAI Platform** and navigate to **Serverless Inference**.
3. Click on **Create model access key**, enter a name, and create the key.
4. Use the generated key as your `DIGITALOCEAN_INFERENCE_KEY`:

### 1. Copy .env.example file to create .env file
cp .env.example .env
### 2. Edit .env and add your access key:
DIGITALOCEAN_INFERENCE_KEY=your_access_key_here

## Installing Test Dependencies

If you are using Poetry, install the test dependencies with:

```bash
poetry install --with test
```

Or, if you are using pip, install the dependencies listed in `pyproject.toml` under `[tool.poetry.group.test.dependencies]`.

## Running Tests

From the project root, run all tests with:

```bash
poetry run pytest
```

To run only the `test_chat_models.py` file, use:

```bash
poetry run pytest tests/integration_tests/test_chat_models.py
```

## Notes
- Tests will be skipped if the required environment variable is not set.
- For more details, see the main project README. 