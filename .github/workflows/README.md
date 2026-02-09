# GitHub Actions CI/CD Workflows

This directory contains the CI/CD workflows for the quantA project.

## Workflows Overview

### 1. `test.yml` - Main Testing Workflow
**Triggers**: Push to master/main, Pull Requests to master/main

**Features**:
- Matrix testing across Python 3.9, 3.10, and 3.11
- Code formatting checks (Black, isort)
- Linting (Flake8)
- Type checking (mypy)
- Full test suite execution with pytest
- Coverage reporting (threshold: 70%)
- Uploads coverage reports to Codecov (optional)
- Archives HTML coverage reports as artifacts
- Separate unit tests and integration tests jobs

**Jobs**:
- `test` - Main matrix testing job
- `test-unit` - Quick unit tests only
- `test-integration` - Integration tests (runs after main tests)
- `coverage-summary` - Aggregates coverage from all Python versions
- `notify` - Sends notifications on failure

**Timeout**: 15 minutes per job

### 2. `code-quality.yml` - Code Quality Checks
**Triggers**: Pull Requests, Push to master/main/develop

**Features**:
- Black code formatting verification
- isort import sorting verification
- Flake8 linting (PEP 8 compliance)
- mypy static type checking
- Bandit security scanning
- Safety dependency vulnerability scanning
- Pylint code quality analysis
- Code complexity analysis (radon)
- GitHub Step Summaries for all checks

**Jobs**:
- `black-check` - Verifies Black formatting
- `isort-check` - Verifies import sorting
- `flake8-check` - Lints code style
- `mypy-check` - Type checks code
- `bandit-check` - Security scanning
- `safety-check` - Dependency vulnerability check
- `pylint-check` - Code quality analysis
- `complexity-check` - Cyclomatic complexity analysis
- `quality-summary` - Aggregates all results

**Timeout**: 5-15 minutes per job (varies by check)

### 3. `release.yml` - Release Automation
**Triggers**: Git tags matching `v*` pattern (e.g., v1.0.0)

**Features**:
- Builds Python distribution packages (wheel + source tarball)
- Tests the built package
- Creates GitHub Release with auto-generated changelog
- Publishes to PyPI (optional)
- Publishes to TestPyPI (for testing)
- Builds and pushes Docker images to Docker Hub
- Multi-platform Docker builds (amd64, arm64)

**Jobs**:
- `build` - Creates distribution packages
- `test-release` - Verifies the built package
- `github-release` - Creates GitHub Release
- `pypi-release` - Publishes to PyPI
- `testpypi-release` - Publishes to TestPyPI
- `docker-release` - Builds and pushes Docker images
- `notify-release` - Sends release notifications

**Timeout**: 15 minutes per job

## Required Secrets

### For `test.yml`:
- `CODECOV_TOKEN` (optional) - For uploading coverage to Codecov

### For `code-quality.yml`:
- No secrets required

### For `release.yml`:
- `PYPI_API_TOKEN` - For publishing to PyPI
- `TEST_PYPI_API_TOKEN` - For publishing to TestPyPI
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/access token
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

## Setup Instructions

### 1. Configure GitHub Secrets

Go to your repository settings: `Settings` → `Secrets and variables` → `Actions`

**Add the following secrets**:

```bash
# Optional: For Codecov integration
CODECOV_TOKEN=your_codecov_token

# Required: For PyPI publishing
PYPI_API_TOKEN=pypi-your_token_here

# Required: For TestPyPI publishing
TEST_PYPI_API_TOKEN=pypi-your_test_token_here

# Required: For Docker Hub
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_access_token
```

### 2. Enable PyPI Trusted Publishing (Recommended)

Instead of using API tokens, you can use PyPI's trusted publishing:

1. Go to your PyPI account settings
2. Navigate to "Publishing" tab
3. Add a new publisher with:
   - **GitHub repository URL**: Your repository URL
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`

Then remove the `password` parameter from the `pypi-release` job in `release.yml`.

### 3. Configure Codecov (Optional)

1. Sign up at https://codecov.io
2. Add your repository
3. Copy the token
4. Add it as a GitHub secret: `CODECOV_TOKEN`

### 4. Configure Docker Hub (Optional)

1. Create a Docker Hub account at https://hub.docker.com
2. Create an access token (Settings → Security)
3. Add the token as a GitHub secret: `DOCKER_PASSWORD`
4. Add your username as a GitHub secret: `DOCKER_USERNAME`

## Usage Examples

### Running Tests Locally

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run linting
make lint

# Format code
make format
```

### Creating a Release

```bash
# 1. Update version numbers
vim VERSION  # or your version file

# 2. Commit changes
git add .
git commit -m "Bump version to 1.0.0"

# 3. Create and push tag
git tag v1.0.0
git push origin v1.0.0

# 4. GitHub Actions will automatically:
#    - Build the package
#    - Run tests
#    - Create GitHub Release
#    - Publish to PyPI
#    - Build Docker images
```

### Manual Workflow Trigger

You can manually trigger workflows from GitHub Actions UI:

1. Go to `Actions` tab
2. Select the workflow (Tests, Code Quality, or Release)
3. Click `Run workflow` button
4. Choose branch and click `Run workflow`

## Workflow Status Badges

Add these badges to your README.md:

```markdown
# CI/CD Status

[![Tests](https://github.com/yourusername/quanta/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/quanta/actions/workflows/test.yml)
[![Code Quality](https://github.com/yourusername/quanta/actions/workflows/code-quality.yml/badge.svg)](https://github.com/yourusername/quanta/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/yourusername/quanta/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/quanta)
[![PyPI version](https://badge.fury.io/py/quanta.svg)](https://badge.fury.io/py/quanta)
```

## Configuration Files

### pytest.ini
- Test configuration
- Coverage settings (threshold: 70%)
- Test markers

### Makefile
- Commands for running tests locally
- Matches CI/CD workflow behavior

### requirements.txt
- All project dependencies
- Test dependencies included

## Troubleshooting

### Tests Failing in CI but Passing Locally

```bash
# Check Python version
python --version  # Should be 3.9, 3.10, or 3.11

# Install exact same dependencies
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run tests with same flags
pytest tests/ -v --cov=. --cov-report=xml --cov-fail-under=70
```

### Coverage Upload Failing

```bash
# Check if CODECOV_TOKEN is set correctly
# Go to repository Settings → Secrets → Actions

# Or disable Codecov upload in test.yml:
# Set fail_ci_if_error: false
```

### PyPI Publish Failing

```bash
# Verify PYPI_API_TOKEN is correct
# Token should start with "pypi-"

# Test locally:
pip install twine
twine check dist/*
```

### Docker Build Failing

```bash
# Verify Docker credentials are correct
# Check Dockerfile exists and is valid
docker build -t quanta:test .
```

## Advanced Configuration

### Customizing Python Versions

Edit `test.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']  # Add 3.12
```

### Adjusting Coverage Threshold

Edit environment variable in `test.yml`:

```yaml
env:
  COVERAGE_THRESHOLD: 80  # Change from 70 to 80
```

### Adding Custom Test Markers

Edit `pytest.ini`:

```ini
markers =
    custom: Custom marker description
```

### Disabling Specific Jobs

Comment out or remove jobs from workflow files.

## Performance Optimization

### Caching

Workflows use pip caching to speed up builds:

```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip'  # Automatically caches dependencies
```

### Parallel Jobs

Multiple jobs run in parallel:
- All Python versions in test.yml run concurrently
- All quality checks in code-quality.yml run concurrently

### Fail-Fast Strategy

Matrix tests continue even if one version fails:

```yaml
strategy:
  fail-fast: false  # Don't stop on first failure
```

## Monitoring and Notifications

### GitHub Step Summaries

Workflows generate summaries visible in the Actions UI:
- Test results
- Coverage reports
- Linting issues
- Security scan results

### Artifact Retention

- Coverage reports: 30 days
- Test results: 7 days
- Build artifacts: 30 days

### Custom Notifications

Modify the `notify` jobs to add:
- Slack notifications
- Email alerts
- Discord webhooks
- GitHub Issues on failure

## Related Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Codecov Documentation](https://docs.codecov.com/)

## Support

For issues or questions:
1. Check workflow logs in Actions tab
2. Review this README
3. Open an issue on GitHub
