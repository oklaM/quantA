# GitHub Actions Quick Start Guide

## Quick Setup (5 minutes)

### Step 1: Add Required Secrets (2 min)

Go to: Repository Settings → Secrets and variables → Actions → New repository secret

Add these secrets:

```
Name: PYPI_API_TOKEN
Value: pypi-your_token_here

Name: DOCKER_USERNAME
Value: your_dockerhub_username

Name: DOCKER_PASSWORD
Value: your_dockerhub_token
```

### Step 2: Verify Workflows (1 min)

```bash
# Check workflow files exist
ls -la .github/workflows/

# Should see:
# - test.yml
# - code-quality.yml
# - release.yml
```

### Step 3: Test Workflows (2 min)

```bash
# Create a test branch
git checkout -b test-ci

# Make a small change
echo "# Test" >> README.md

# Commit and push
git add README.md
git commit -m "Test CI workflows"
git push origin test-ci

# Create a Pull Request on GitHub
# Workflows will automatically run!
```

## Workflow Cheatsheet

### Test Workflow (`test.yml`)

**When it runs:**
- Every push to `master` or `main`
- Every pull request to `master` or `main`

**What it does:**
```
1. ✓ Setup Python 3.9, 3.10, 3.11
2. ✓ Install dependencies
3. ✓ Check code formatting (Black)
4. ✓ Check import sorting (isort)
5. ✓ Run linting (Flake8)
6. ✓ Run type checking (mypy)
7. ✓ Run tests (pytest)
8. ✓ Check coverage (must be ≥70%)
9. ✓ Upload reports
```

**Manual trigger:**
- Go to Actions → Tests → Run workflow

### Code Quality Workflow (`code-quality.yml`)

**When it runs:**
- Every pull request
- Every push to `master`, `main`, or `develop`

**What it does:**
```
1. ✓ Black format check
2. ✓ isort import check
3. ✓ Flake8 linting
4. ✓ mypy type checking
5. ✓ Bandit security scan
6. ✓ Safety dependency check
7. ✓ Pylint quality check
8. ✓ Complexity analysis
```

**Manual trigger:**
- Go to Actions → Code Quality → Run workflow

### Release Workflow (`release.yml`)

**When it runs:**
- When you push a version tag (e.g., `v1.0.0`)

**What it does:**
```
1. ✓ Build Python package (wheel + tar.gz)
2. ✓ Test the package
3. ✓ Create GitHub Release
4. ✓ Publish to PyPI
5. ✓ Publish to TestPyPI
6. ✓ Build Docker image
7. ✓ Push to Docker Hub
```

**Manual trigger:**
- Go to Actions → Release → Run workflow

## Common Commands

### Local Testing (match CI behavior)

```bash
# Run tests (like CI)
make test

# Run tests with coverage (like CI)
make test-cov

# Check code formatting
make format-check

# Lint code
make lint

# Format code
make format
```

### Creating a Release

```bash
# 1. Update version
echo "1.0.0" > VERSION

# 2. Commit changes
git add VERSION
git commit -m "Release v1.0.0"

# 3. Create tag
git tag v1.0.0

# 4. Push tag (triggers release workflow)
git push origin v1.0.0
```

### Troubleshooting

```bash
# Tests pass locally but fail in CI?
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v --cov=. --cov-fail-under=70

# Coverage too low?
pytest --cov=. --cov-report=term-missing
# Look at lines with ">>>>>" (not covered)

# Format check fails?
make format  # This will auto-format
git add .
git commit -m "Format code"

# PyPI publish fails?
# Check token starts with "pypi-"
# Test locally: twine check dist/*
```

## Status Badges

Add to README.md:

```markdown
![Tests](https://github.com/YOUR_USERNAME/quanta/actions/workflows/test.yml/badge.svg)
![Code Quality](https://github.com/YOUR_USERNAME/quanta/actions/workflows/code-quality.yml/badge.svg)
```

## Workflow Files Structure

```
.github/workflows/
├── README.md              # Detailed documentation
├── QUICKSTART.md          # This file
├── test.yml               # Main testing workflow
├── code-quality.yml       # Code quality checks
├── release.yml            # Release automation
└── ci.yml                 # Legacy CI workflow (can be removed)
```

## Environment Variables

### In test.yml
- `PYTHON_VERSION_DEFAULT`: 3.10
- `COVERAGE_THRESHOLD`: 70

### In code-quality.yml
- `PYTHON_VERSION`: 3.10

### In release.yml
- `PYTHON_VERSION`: 3.10

## Secrets Required

### Optional
- `CODECOV_TOKEN` - For Codecov integration

### Required for PyPI
- `PYPI_API_TOKEN` - PyPI API token
- `TEST_PYPI_API_TOKEN` - TestPyPI API token

### Required for Docker
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token

## Coverage Reporting

### Current Threshold
- Minimum: 70%
- Enforced in: `pytest.ini` and `test.yml`

### View Coverage
- HTML report: Download artifact from Actions
- Term report: Check workflow logs
- Codecov: https://codecov.io (if configured)

### Improve Coverage
```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Markers

```bash
# Unit tests only
pytest -m "unit"

# Integration tests only
pytest -m "integration"

# Exclude slow tests
pytest -m "not slow"

# Exclude tests requiring external data
pytest -m "not requires_data"
```

## Quick Reference: What to Do When...

### PR Checks Failing
1. Click on "Details" next to failed check
2. Read the error message
3. Fix locally using same commands
4. Commit and push fixes

### Tests Passing but Coverage Low
1. Check which files need coverage: `pytest --cov-report=term-missing`
2. Write tests for uncovered code
3. Re-run tests

### Format Check Failing
1. Run: `make format`
2. Commit formatted code
3. Push changes

### Security Scan Failing
1. Check Bandit/Safety output
2. Fix security issues
3. Commit fixes

### Release Workflow Failing
1. Check PyPI token is correct
2. Verify version tag format (v1.0.0)
3. Check Docker credentials
4. Review build logs

## Getting Help

1. **Workflow logs**: Actions tab → Select workflow run → View logs
2. **This README**: Check detailed documentation in `README.md`
3. **GitHub Actions docs**: https://docs.github.com/en/actions
4. **Open issue**: Describe problem with workflow run link

## Best Practices

### Before Pushing
```bash
# Always run locally first
make format
make lint
make test
```

### Before Creating PR
```bash
# Ensure everything passes
make test-cov  # Check coverage
make format-check  # Verify formatting
```

### Before Releasing
```bash
# Run full test suite
make test
make test-integration

# Verify version
git tag -l

# Test release locally
python -m build
twine check dist/*
```

## Next Steps

1. Configure required secrets (see Step 1)
2. Test workflows with a PR
3. Set up Codecov (optional)
4. Configure PyPI trusted publishing (recommended)
5. Customize workflows as needed

For detailed information, see [README.md](./README.md)
