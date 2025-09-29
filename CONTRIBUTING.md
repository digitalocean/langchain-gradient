# Contributing to langchain-gradient

Thank you for your interest in contributing to langchain-gradient! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- Git

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/langchain-gradientai.git
   cd langchain-gradientai
   ```

3. **Install dependencies** using Poetry:
   ```bash
   poetry install --with dev,test,lint
   ```

4. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   poetry run pre-commit install
   ```

5. **Configure your environment**:
   Create a `.env` file with your DigitalOcean API key:
   ```bash
   DIGITALOCEAN_INFERENCE_KEY=your_access_key_here
   ```

## Making Changes

### Branch Strategy

- Create a new branch for each feature or bugfix
- Use descriptive branch names (e.g., `feature/add-new-model`, `fix/streaming-issue`)
- Keep branches focused on a single change

```bash
git checkout -b feature/your-feature-name
```

### Code Style

This project uses several tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting and import sorting
- **MyPy** for type checking
- **Codespell** for spell checking

#### Formatting Code

```bash
# Format all code
make format

# Format only changed files
make format_diff
```

#### Linting

```bash
# Lint all code
make lint

# Lint only changed files
make lint_diff

# Lint specific packages
make lint_package  # Lint langchain_gradient package
make lint_tests    # Lint test files
```

#### Type Checking

```bash
# Run MyPy type checking
poetry run mypy langchain_gradient
```

#### Spell Checking

```bash
# Check for spelling errors
make spell_check

# Fix spelling errors automatically
make spell_fix
```


## Testing

### Running Tests

The project has two types of tests:

1. **Unit Tests** - Fast, isolated tests that don't require network access
2. **Integration Tests** - Tests that interact with external APIs

#### Unit Tests

```bash
# Run all unit tests
poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/

# Run tests for a specific file
poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/test_chat_models.py

# Run tests with verbose output
poetry run pytest --disable-socket --allow-unix-socket -v tests/unit_tests/

# Watch mode (re-runs tests on file changes)
poetry run ptw --snapshot-update --now . -- -vv tests/unit_tests/
```

#### Integration Tests

```bash
# Run integration tests (requires API key)
poetry run pytest tests/integration_tests/

# Run specific integration test file
poetry run pytest tests/integration_tests/test_chat_models.py
```

### Writing Tests

- Write tests for new features and bug fixes
- Follow the existing test patterns in the `tests/` directory
- Use descriptive test names
- Include both positive and negative test cases
- Mock external API calls in unit tests

### Test Structure

```
tests/
├── unit_tests/          # Unit tests (no network calls)
│   └── test_chat_models.py
└── integration_tests/   # Integration tests (with network calls)
    ├── test_chat_models.py
    └── test_chat_model_streaming.py
```

## Submitting Changes

### Commit Guidelines

- Write clear, descriptive commit messages
- Use the conventional commit format when possible:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `test:` for test additions/changes
  - `refactor:` for code refactoring
  - `style:` for formatting changes

Example:
```
feat: add support for new model endpoint
fix: resolve streaming response parsing issue
docs: update installation instructions
```

### Pull Request Process

1. **Ensure all checks pass**:
   ```bash
   make lint
   poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/
   poetry run pytest tests/integration_tests/  # If applicable
   ```

2. **Update documentation** if your changes affect user-facing functionality

3. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Provide a detailed description of your changes
   - Reference any related issues
   - Include screenshots or examples if applicable

4. **Respond to feedback** promptly and make requested changes

### Pull Request Template

When creating a PR, please include:

- **Description**: What changes were made and why
- **Type of change**: Bug fix, new feature, documentation, etc.
- **Testing**: How the changes were tested
- **Breaking changes**: Any breaking changes and migration steps
- **Checklist**: Confirm all requirements are met

## Issue Guidelines

### Before Creating an Issue

1. Check existing issues to avoid duplicates
2. Search the documentation and README
3. Try to reproduce the issue with the latest version

### Issue Types

- **Bug Report**: Something isn't working as expected
- **Feature Request**: Suggest a new feature or enhancement
- **Documentation**: Improvements to documentation
- **Question**: Ask a question about usage or implementation

### Bug Report Template

When reporting a bug, please include:

- **Environment**: Python version, OS, package version
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback if applicable
- **Additional context**: Any other relevant information

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] All tests pass (unit and integration)
- [ ] Code is properly formatted and linted
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

**Quick test commands:**
```bash
# Run unit tests
poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/

# Run integration tests (if applicable)
poetry run pytest tests/integration_tests/

# Run linting
make lint
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different environments if needed
4. **Approval** from at least one maintainer

## Release Process

Releases are managed through the main LangChain repository. The process typically involves:

1. Version bumping in `pyproject.toml`
2. Updating changelog
3. Creating a release tag
4. Publishing to PyPI

## Getting Help

- **Documentation**: Check the README and inline documentation
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions and ideas

## Development Commands Reference

```bash
# Setup
poetry install --with dev,test,lint

# Code quality
make format          # Format code
make lint            # Lint code
make spell_check     # Check spelling

# Testing
poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/     # Run unit tests
poetry run pytest tests/integration_tests/                                  # Run integration tests
poetry run ptw --snapshot-update --now . -- -vv tests/unit_tests/           # Watch mode for tests

# Utilities
make check_imports   # Check import structure
make help           # Show all available commands
```

## Thank You

Thank you for contributing to langchain-gradient! Your contributions help make this project better for everyone.

---

For questions about contributing, please open an issue or contact the maintainers.
