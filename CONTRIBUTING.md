# Contributing to SB3-GRPO

Thank you for your interest in contributing to SB3-GRPO! 

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/sb3-grpo.git
   cd sb3-grpo
   ```
3. Install in development mode:
   ```bash
   pip install -e .[dev]
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

Run these before submitting:

```bash
black .
flake8 .
mypy . --ignore-missing-imports
```

## Testing

Make sure your changes don't break existing functionality:

```bash
python -c "from sb3_grpo import GRPO; print('Import test passed!')"
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run the code quality tools
4. Test your changes
5. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:

- Python version
- PyTorch version
- Stable Baselines3 version
- Complete error traceback
- Minimal reproduction example

## Feature Requests

We welcome feature requests! Please open an issue with:

- Clear description of the feature
- Use case examples
- Proposed API (if applicable)

Thank you for contributing!
