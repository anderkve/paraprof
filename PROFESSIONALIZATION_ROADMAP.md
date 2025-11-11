# ParaProf Professionalization Roadmap

This document tracks the multi-phase plan to transform ParaProf into a professional, maintainable, and pip-installable Python package.

**Current Branch:** `phase1-package-infrastructure`
**Last Updated:** 2025-11-11
**Python Version:** 3.10+
**License:** MIT

---

## Key Decisions Made

1. **License**: MIT (permissive, allows commercial use)
2. **Python Support**: 3.10+ (modern features, type hints)
3. **MPI Testing in CI**: Skip for now (manual testing only)
4. **Matplotlib**: Optional dependency (in `[viz]` extra)
5. **Package Name**: Keep "paraprof" (may change later for PyPI)

---

## Phase 1: Package Infrastructure ✅ COMPLETED

**Branch:** `phase1-package-infrastructure`
**Status:** Done (2 commits)

### Completed Tasks

1. ✅ Created new git branch
2. ✅ Created `pyproject.toml` with modern package configuration
3. ✅ Added MIT LICENSE file
4. ✅ Reorganized code to `src/` layout
5. ✅ Created comprehensive test suite (17 tests, all passing)
6. ✅ Set up GitHub Actions CI workflow
7. ✅ Updated .gitignore for new structure
8. ✅ Verified package installation works
9. ✅ Committed changes with detailed commit message
10. ✅ Fixed import bugs (relative imports in master.py)

### Key Files Created/Modified

- `pyproject.toml` - Modern package configuration with build system, dependencies, tool configs
- `LICENSE` - MIT License
- `CHANGELOG.md` - Version history tracking
- `src/paraprof/` - All source code moved to proper package structure
- `src/paraprof/py.typed` - Type hints marker for PEP 561
- `tests/` - Test suite with pytest
  - `test_sampler.py` - 7 tests for GridAnchoredDESampler
  - `test_test_functions.py` - 6 tests for benchmark functions
  - `test_interpolation.py` - 4 tests for GridInterpolator
  - `conftest.py` - Shared pytest fixtures
- `.github/workflows/tests.yml` - CI/CD automation

### Installation

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[viz]"      # visualization support
pip install -e ".[dev]"      # development tools
pip install -e ".[all]"      # everything
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/paraprof --cov-report=term-missing
```

### Running Examples

```bash
# From project root
mpiexec -n 4 python examples/run_himmelblau_4d.py
```

---

## Phase 2: Code Quality & Maintainability 🎯 NEXT

**Estimated Time:** 1-2 weeks
**Priority:** High

### Goals

Transform code quality with logging, type hints, formatting, and error handling.

### Tasks

#### 2.1 Logging System
- [ ] Create `src/paraprof/logger.py` with configurable logging
- [ ] Replace all `print()` statements with `logger.info()`, `logger.debug()`, etc.
- [ ] Add verbosity control (--quiet, --verbose flags)
- [ ] Include MPI rank information in log messages
- [ ] Add log levels: DEBUG, INFO, WARNING, ERROR
- [ ] Support log file output (optional)

**Example structure:**
```python
# logger.py
import logging
from mpi4py import MPI

def setup_logger(name, level=logging.INFO, rank=None):
    """Setup logger with MPI rank prefix."""
    if rank is None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'[Rank {rank}] %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
```

#### 2.2 Type Hints
- [ ] Add type annotations to all public API functions
- [ ] Use `numpy.typing` for array types (e.g., `npt.NDArray[np.float64]`)
- [ ] Add type hints to `__init__` methods
- [ ] Configure `mypy` strict mode in `pyproject.toml`
- [ ] Fix any type errors that arise
- [ ] Add return type annotations

**Priority modules:**
1. `sampler.py` - Main API
2. `master.py` - Public functions
3. `jobs/base.py` - Job interface
4. `test_functions.py` - Public API

#### 2.3 Code Formatting & Linting
- [ ] Run `black` on entire codebase: `black src/`
- [ ] Fix all `ruff` linting issues: `ruff check src/ --fix`
- [ ] Set up pre-commit hooks (`.pre-commit-config.yaml`)
- [ ] Install pre-commit: `pre-commit install`
- [ ] Test pre-commit runs successfully

**Pre-commit config:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files
```

#### 2.4 Error Handling & Validation
- [ ] Create `src/paraprof/exceptions.py` with custom exceptions:
  - `ParaProfError` (base exception)
  - `InvalidProjectionError` (bad projection config)
  - `ConvergenceError` (optimization failures)
  - `MPIError` (MPI-related issues)
  - `InvalidBoundsError` (bad parameter bounds)
- [ ] Add input validation to `GridAnchoredDESampler.__init__()`
- [ ] Add validation in job creation methods
- [ ] Improve error messages with helpful suggestions
- [ ] Add docstring examples showing error cases

#### 2.5 Enhanced README
- [ ] Add badges (build status, coverage, PyPI version, license)
- [ ] Write clear description of what ParaProf does
- [ ] Add "Key Features" section
- [ ] Add quick installation instructions
- [ ] Add minimal working example (5-10 lines)
- [ ] Add links to documentation
- [ ] Add citation information (if for research)
- [ ] Add "Contributing" section link

**Recommended badges:**
```markdown
[![Tests](https://github.com/anderkve/paraprof/workflows/Tests/badge.svg)](https://github.com/anderkve/paraprof/actions)
[![codecov](https://codecov.io/gh/anderkve/paraprof/branch/main/graph/badge.svg)](https://codecov.io/gh/anderkve/paraprof)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

---

## Phase 3: Testing Infrastructure 🧪

**Estimated Time:** 1-2 weeks
**Priority:** High

### Goals

Expand test coverage and add integration tests.

### Tasks

#### 3.1 Expand Unit Tests
- [ ] Increase coverage to >80%
- [ ] Add tests for job classes:
  - `test_jobs_activation.py`
  - `test_jobs_de.py`
  - `test_jobs_lbfgsb.py`
  - `test_jobs_patching.py`
- [ ] Add tests for utility functions:
  - `test_grid_operations.py`
  - `test_bounds_handling.py`
- [ ] Add tests for edge cases
- [ ] Test error conditions and exceptions

#### 3.2 Integration Tests
- [ ] Create `tests/test_integration.py`
- [ ] Test complete workflow with small grids (5x5)
- [ ] Test refinement pipeline
- [ ] Test warm-starting between projections
- [ ] Test direct evaluation mode
- [ ] Mock MPI for testing (single process)

**Example integration test:**
```python
def test_complete_workflow_2d():
    """Test complete DE + patching workflow on small grid."""
    func = lambda x: -(x[0]**2 + x[1]**2)
    bounds = np.array([[-2, 2], [-2, 2]])
    projection = {'dims': [0, 1], 'grid_points': [5, 5]}

    sampler = GridAnchoredDESampler(
        target_func=func,
        bounds=bounds,
        projections=[projection],
        pop_per_grid_point=2,
    )

    # Mock MPI environment for single-process testing
    # Run workflow
    # Assert results are reasonable
```

#### 3.3 Performance Benchmarks
- [ ] Create `benchmarks/benchmark_suite.py`
- [ ] Benchmark different grid sizes
- [ ] Benchmark different mutation strategies
- [ ] Track function evaluation counts
- [ ] Compare serial vs parallel performance

#### 3.4 Continuous Integration Enhancements
- [ ] Add coverage reporting to GitHub Actions
- [ ] Add test result artifacts
- [ ] Test on multiple OS (Linux, macOS if possible)
- [ ] Add performance regression tests
- [ ] Generate coverage badge

---

## Phase 4: Documentation 📚

**Estimated Time:** 2-3 weeks
**Priority:** Medium-High

### Goals

Create comprehensive documentation with Sphinx.

### Tasks

#### 4.1 Sphinx Setup
- [ ] Install Sphinx: `pip install sphinx sphinx-rtd-theme`
- [ ] Run `sphinx-quickstart docs/`
- [ ] Configure `docs/conf.py`:
  - Set theme to `sphinx_rtd_theme`
  - Enable autodoc, napoleon extensions
  - Configure autodoc to find `src/paraprof`
- [ ] Add `docs/requirements.txt` for Read the Docs

#### 4.2 Documentation Structure
- [ ] `docs/index.rst` - Main landing page
- [ ] `docs/installation.rst` - Installation guide
  - Pip installation
  - From source
  - MPI requirements (OpenMPI, MPICH)
  - Optional dependencies
- [ ] `docs/quickstart.rst` - 5-minute tutorial
  - Minimal example
  - Running with MPI
  - Understanding output
- [ ] `docs/user_guide/` - Comprehensive guide
  - `concepts.rst` - Core concepts (grid-anchored DE, ROI, patching)
  - `configuration.rst` - All parameters explained
  - `projections.rst` - Projection specifications
  - `refinement.rst` - Grid refinement strategies
  - `test_functions.rst` - Available benchmark functions
- [ ] `docs/api.rst` - API reference (auto-generated)
- [ ] `docs/theory.rst` - Mathematical background
  - Differential Evolution algorithm
  - Grid-anchored approach
  - Patching algorithm
  - References to papers
- [ ] `docs/examples/` - Example gallery
  - Simple 1D projection
  - 2D projection with visualization
  - Multi-projection workflow
  - Custom test function
- [ ] `docs/contributing.rst` - Contribution guide
- [ ] `docs/changelog.rst` - Include CHANGELOG.md

#### 4.3 Docstring Improvements
- [ ] Use NumPy-style docstrings consistently
- [ ] Add Examples sections to key functions
- [ ] Document all parameters with types
- [ ] Add notes about MPI usage
- [ ] Add "See Also" sections
- [ ] Add references to papers/algorithms

**Example docstring:**
```python
def create_de_generation_jobs(self, next_job_id, max_num_to_evolve):
    """
    Generate Differential Evolution jobs for one generation.

    Creates DEGridPointJob instances for active grid points, prioritizing
    points with high fitness or improvement rates.

    Parameters
    ----------
    next_job_id : int
        Next available job ID for assignment
    max_num_to_evolve : int or None
        Maximum number of grid points to evolve this generation.
        If None, evolves all unconverged points.

    Returns
    -------
    jobs : list of DEGridPointJob
        List of DE jobs to execute
    next_job_id : int
        Updated job ID counter
    successful_F : list of float
        Shared list for tracking successful F parameters
    successful_CR : list of float
        Shared list for tracking successful CR parameters

    Notes
    -----
    This method is only called on the master process. Jobs are distributed
    to workers via MPI.

    Examples
    --------
    >>> jobs, next_id, F_list, CR_list = sampler.create_de_generation_jobs(0, 10)
    >>> len(jobs) <= 10
    True

    See Also
    --------
    DEGridPointJob : The job class for DE evolution
    update_de_memory : Updates F/CR memory after generation
    """
```

#### 4.4 Read the Docs Integration
- [ ] Create account on readthedocs.org
- [ ] Connect GitHub repository
- [ ] Configure webhook for auto-builds
- [ ] Test documentation builds successfully
- [ ] Add documentation link to README

---

## Phase 5: Development Workflow 🔄

**Estimated Time:** 1 week
**Priority:** Medium

### Tasks

#### 5.1 Contributing Guidelines
- [ ] Create `CONTRIBUTING.md`
- [ ] Document code style requirements
- [ ] Explain how to run tests
- [ ] Describe pull request process
- [ ] Add development setup instructions
- [ ] Document commit message conventions

#### 5.2 Issue & PR Templates
- [ ] Create `.github/ISSUE_TEMPLATE/bug_report.md`
- [ ] Create `.github/ISSUE_TEMPLATE/feature_request.md`
- [ ] Create `.github/PULL_REQUEST_TEMPLATE.md`

#### 5.3 Version Management
- [ ] Use semantic versioning strictly
- [ ] Document versioning policy
- [ ] Set up automated version bumping (bumpversion or similar)
- [ ] Create git tags for releases

#### 5.4 Pre-commit Hooks
- [ ] Install pre-commit framework
- [ ] Configure hooks (black, ruff, mypy)
- [ ] Test hooks work correctly
- [ ] Document in CONTRIBUTING.md

---

## Phase 6: Feature Enhancements 🚀

**Estimated Time:** 2-4 weeks
**Priority:** Low-Medium

### Goals

Add extensibility features for power users.

### Tasks

#### 6.1 Configuration System
- [ ] Create `src/paraprof/config.py` with dataclasses
- [ ] Support loading from YAML/JSON files
- [ ] Add config validation with pydantic (optional)
- [ ] Example config files in `examples/configs/`

#### 6.2 Plugin Architecture
- [ ] Design plugin interface for custom jobs
- [ ] Create example plugin
- [ ] Document plugin creation
- [ ] Add plugin discovery mechanism

#### 6.3 Callback System
- [ ] Add callback hooks:
  - `on_generation_complete`
  - `on_grid_point_converged`
  - `on_stage_transition`
  - `on_job_complete`
- [ ] Allow user-defined callbacks
- [ ] Document callback API
- [ ] Add example callbacks

#### 6.4 Checkpointing
- [ ] Implement save/load for sampler state
- [ ] Support resume from checkpoint
- [ ] Use pickle or HDF5
- [ ] Add checkpoint frequency control
- [ ] Add example of checkpointing

---

## Phase 7: Distribution & Deployment 📦

**Estimated Time:** 1-2 weeks
**Priority:** Medium

### Tasks

#### 7.1 PyPI Publishing
- [ ] Test build process: `python -m build`
- [ ] Test on TestPyPI first
- [ ] Set up PyPI account
- [ ] Configure trusted publishing on GitHub
- [ ] Create release workflow `.github/workflows/publish.yml`
- [ ] Test automated publishing

**Publish workflow:**
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install build
        run: pip install build
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

#### 7.2 Conda Package (Optional)
- [ ] Create conda-forge recipe
- [ ] Submit to conda-forge
- [ ] Test conda installation
- [ ] Update documentation

#### 7.3 Docker Image (Optional)
- [ ] Create `Dockerfile` with MPI support
- [ ] Build and test image
- [ ] Push to Docker Hub
- [ ] Add usage instructions

---

## Phase 8: Project Cleanup 🧹

**Estimated Time:** 1 week
**Priority:** Low

### Tasks

#### 8.1 Remove Development Artifacts
- [ ] Archive or delete `test_108_MPI.py`, `test_38.py`
- [ ] Move draft docs to separate branch or `docs/drafts/`
- [ ] Clean up `plots_*/` directories
- [ ] Remove old test scripts

#### 8.2 Organize Documentation Files
- [ ] Move `*_SUMMARY.md` files to `docs/design/`
- [ ] Convert to RST for Sphinx if needed
- [ ] Ensure all docs referenced in main docs

#### 8.3 Repository Polish
- [ ] Update all README files in subdirectories
- [ ] Ensure examples are well-documented
- [ ] Add project description to GitHub
- [ ] Add topics/tags to GitHub repo
- [ ] Create GitHub project board (optional)

---

## Current Status Summary

### ✅ Completed
- Phase 1: Package Infrastructure (100%)
  - Modern package configuration
  - Proper source layout
  - Basic test suite
  - CI/CD automation
  - MIT License

### 🎯 Next Steps (Phase 2)
1. Create logging system
2. Add type hints to public API
3. Set up pre-commit hooks
4. Create custom exception classes
5. Enhance README with badges

### 📊 Metrics
- **Test Coverage:** ~40% (basic tests only)
- **Type Coverage:** 0% (no type hints yet)
- **Documentation:** Basic README only
- **CI/CD:** GitHub Actions configured
- **Code Quality:** Not yet enforced

---

## Quick Commands Reference

### Development
```bash
# Install in editable mode with dev tools
pip install -e ".[dev,viz]"

# Run tests
pytest tests/ -v --cov=src/paraprof

# Format code
black src/ tests/

# Lint code
ruff check src/ --fix

# Type check
mypy src/paraprof

# Run example
mpiexec -n 4 python examples/run_himmelblau_4d.py
```

### Git Workflow
```bash
# Current branch
git checkout phase1-package-infrastructure

# Create Phase 2 branch (when ready)
git checkout -b phase2-code-quality

# See changes
git log --oneline --graph
```

---

## Notes & Considerations

### Testing with MPI
- Unit tests run without MPI (single process)
- Integration tests can mock MPI for simple cases
- Full MPI tests require manual testing for now
- Consider adding MPI tests in future CI environment

### Performance
- Profile code with large grids before optimization
- Document performance characteristics
- Consider adding benchmarks to CI

### Backward Compatibility
- Maintain API stability after 1.0.0 release
- Use deprecation warnings for API changes
- Document breaking changes in CHANGELOG

### Community
- Monitor GitHub issues/PRs
- Be responsive to user feedback
- Consider creating discussions/forum
- Build example gallery from user contributions

---

## Resources

### Documentation
- [Python Packaging Guide](https://packaging.python.org/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

### Testing
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)

### Code Quality
- [Black](https://black.readthedocs.io/)
- [Ruff](https://docs.astral.sh/ruff/)
- [MyPy](https://mypy.readthedocs.io/)

### CI/CD
- [GitHub Actions](https://docs.github.com/en/actions)
- [Read the Docs](https://readthedocs.org/)

---

**Last Updated:** 2025-11-11
**Maintainer:** Anders Kvellestad
**Status:** Phase 1 Complete, Ready for Phase 2
