# Contributing to V2I-Calib

Thank you for considering contributing to V2I-Calib! We welcome contributions from the community to help improve this project.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10+
- Git
- GitHub account

### Development Setup

1. **Fork the repository**
   
   Click the "Fork" button at the top-right of this page.

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/v2i-calib.git
   cd v2i-calib
   ```

3. **Set up upstream remote**
   ```bash
   git remote add upstream https://github.com/MassimoQu/v2i-calib.git
   ```

4. **Create a development environment**
   ```bash
   conda create -n v2i-calib-dev python=3.10 -y
   conda activate v2i-calib-dev
   pip install -r requirements.txt
   ```

5. **Install development dependencies**
   ```bash
   pip install black isort flake8 mypy pytest pytest-cov
   ```

## Ways to Contribute

There are many ways you can contribute to this project:

### 🐛 Report Bugs

Report bugs by creating a new issue with the "Bug Report" template.

**Before submitting a bug report:**

- Check the existing issues to avoid duplicates
- Verify the bug is not already fixed in the latest version
- Collect information about the bug:
  - Python version
  - Operating system
  - Error messages
  - Steps to reproduce

### 💡 Suggest Features

Suggest new features by creating an issue with the "Feature Request" template.

**When suggesting a feature, please include:**

- Clear description of the feature
- Use cases and motivation
- Any relevant research or references
- Implementation suggestions (optional)

### 📝 Improve Documentation

Help improve our documentation by:

- Fixing typos or grammatical errors
- Adding missing examples
- Improving code comments
- Translating documentation
- Creating tutorials

### 🔧 Submit Pull Requests

Contribute code by following the steps below.

## Submitting Changes

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a new branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow our [coding standards](#coding-standards) when making changes.

### 3. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Check code coverage
pytest tests/ --cov=v2x_calib --cov-report=term-missing
```

### 4. Format Your Code

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check for linting issues
flake8 .
```

### 5. Commit Your Changes

Write a clear commit message following our [commit message guidelines](#commit-messages).

```bash
git add .
git commit -m "feat: Add new calibration algorithm for V2I scenarios"
```

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
```

### 7. Respond to Review

Be responsive to feedback from maintainers and make necessary changes.

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: Maximum 88 characters
- **Quotes**: Use single quotes for strings
- **Imports**: Sort with isort

### Code Formatting

Use these tools to ensure code quality:

```bash
# Install formatting tools
pip install black isort flake8

# Format code
black .
isort .

# Check code
flake8 .
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, List

def calibrate(
    points: np.ndarray,
    config: dict,
    verbose: bool = False
) -> np.ndarray:
    """Calibrate point cloud data.
    
    Args:
        points: Input point cloud
        config: Configuration dictionary
        verbose: Print verbose output
    
    Returns:
        Calibrated point cloud
    """
    ...
```

### Documentation Strings

Write docstrings for all public functions:

```python
def calculate_error(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate the registration error between predicted and ground truth transforms.
    
    Args:
        predicted: Predicted transformation matrix (4x4)
        ground_truth: Ground truth transformation matrix (4x4)
    
    Returns:
        Translation error in meters
    """
    ...
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(calibration): Add new oIoU association method

fix(evaluation): Correct TE calculation for large rotations

docs(readme): Add installation guide

test(tools): Add unit tests for calibration module
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use pytest framework
- Follow naming convention: `test_*.py`

```python
import pytest
from v2x_calib.association import oiou_association

class TestOiouAssociation:
    def test_basic_association(self):
"""Test basic oIoU association functionality."""
        boxes1 = np.array([[0, 0, 0, 1, 1, 1]])
        boxes2 = np.array([[0.1, 0.1, 0.1, 1, 1, 1]])
        
        result = oiou_association(boxes1, boxes2, threshold=0.5)
        
        assert len(result) > 0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_association.py

# Run with coverage
pytest --cov=v2x_calib --cov-report=html

# Run in parallel
pytest -n auto
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx_rtd_theme

# Build HTML docs
cd docs/
make html
```

### Writing Documentation

- Use Markdown for simple docs
- Include code examples
- Add comments for complex logic
- Update README for new features

## Community

### Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and ideas
- **Email**: qqxmassimo@gmail.com

### Recognition

Contributors will be recognized in:

- Our README.md contributors section
- Release notes
- Project documentation

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort! 🙏
