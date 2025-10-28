# ParaProf: Parallel Profile Likelihood Computation

A parallel implementation of Grid-Anchored Differential Evolution for computing profile likelihoods using MPI.

## Directory Structure

```
paraprof/
├── __init__.py                    # Package initialization
├── constants.py                   # MPI task constants
├── jobs/                          # Job classes for async execution
│   ├── __init__.py                # Job package exports
│   ├── base.py                    # Abstract Job base class
│   ├── lbfgsb_job.py              # L-BFGS-B optimization job
│   ├── activation_job.py          # Grid point activation job
│   └── de_job.py                  # Differential Evolution job
├── sampler.py                     # Main GridAnchoredDESampler class
├── master.py                      # Master process orchestration
├── worker.py                      # Worker process execution
├── visualization.py               # Plotting utilities
├── test_functions.py              # Benchmark test functions
└── examples/
    └── run_himmelblau_4d.py       # Example usage script
```

## Module Responsibilities

### Core Modules

- **`sampler.py`**: Contains `GridAnchoredDESampler` class with algorithm state and job factory methods
- **`master.py`**: Master process orchestration with workflow state machine
- **`worker.py`**: Worker process event loop for task execution
- **`constants.py`**: MPI communication constants

### Job System (`jobs/`)

- **`base.py`**: Abstract `Job` class defining the job interface
- **`lbfgsb_job.py`**: L-BFGS-B optimization job with gradient computation and line search
- **`activation_job.py`**: Grid point initialization with Latin Hypercube Sampling
- **`de_job.py`**: Differential Evolution generation for single grid points

### Utilities

- **`visualization.py`**: 2D profile likelihood plotting
- **`test_functions.py`**: Benchmark optimization problems (Rosenbrock, Himmelblau)

## Usage

### Running the Example

```bash
mpiexec -n <num_cores> python examples/run_himmelblau_4d.py
```

Replace `<num_cores>` with the number of MPI processes (1 master + N workers).

### Creating Custom Scripts

```python
from mpi4py import MPI
from sampler import GridAnchoredDESampler
from master import master_main
from worker import worker_main
from test_functions import get_test_function

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# Define your target function and bounds
log_likelihood, bounds, _ = get_test_function("himmelblau_4d")

# Configure projections
projections = [
    {'dims': [0, 1], 'grid_points': [100, 100]},
]

if myrank == 0:
    # Master process
    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=bounds,
        projections=projections,
        # ... other parameters
    )

    master_main(comm, sampler, num_generations=1000, ...)
else:
    # Worker process
    worker_main(comm, myrank)
```

## Algorithm Workflow

The master process executes a 4-stage workflow:

1. **INITIAL_OPTIMIZATION**: Global L-BFGS-B optimization to find maxima
2. **ACTIVATION**: Initialize grid point populations near found maxima
3. **DE_LOOP**: Differential Evolution with dynamic grid activation
4. **PATCHING**: Gradient-based refinement using neighbor information

## Development Guide

### Understanding the Logic Flow

**For new developers, read in this order:**

1. `examples/run_himmelblau_4d.py` - See how components fit together
2. `sampler.py` - Understand algorithm configuration and state
3. `master.py` - See job orchestration and scheduling
4. `jobs/base.py` - Understand the job abstraction
5. `jobs/*.py` - Deep dive into specific job implementations
6. `worker.py` - Understand task execution

### Adding New Job Types

1. Create new file in `jobs/` inheriting from `Job`
2. Implement `start()`, `process_result()`, and `on_finish()` methods
3. Add job factory method to `sampler.py`
4. Update `jobs/__init__.py` to export the new class
5. Update master's priority queue logic if needed

### Key Design Patterns

- **Job-based Asynchronous Execution**: Each job manages its own multi-step workflow
- **Priority Queues**: Optimization jobs get priority over population evaluations
- **State Machine**: Master orchestrates workflow through well-defined stages
- **Sparse Grid**: Only activated grid points consume memory

## Original File

The original monolithic implementation is preserved as `test_108_MPI.py`.

## Dependencies

- Python 3.6+
- NumPy
- SciPy
- mpi4py
- Matplotlib (optional, for visualization)