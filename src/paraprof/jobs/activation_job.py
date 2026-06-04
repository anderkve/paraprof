"""Grid point activation job for initializing populations."""
import numpy as np
import collections
from scipy.stats.qmc import LatinHypercube as LHS
from .base import Job


WARM_START_PERTURBATION_STD = 0.1  # Perturbation σ around warm-start params (fraction of bounds extent).
LHS_SEED_MIN = 1_000_000
LHS_SEED_MAX = 1_000_000_000_000


class ActivationJob(Job):
    """Evaluate the initial population for one grid cell (optionally warm-started)."""
    def __init__(self, job_id, sampler, grid_idx, warm_start_params=None, mark_converged=False):
        super().__init__(job_id, 'ACTIVATE_GRID_POINT', sampler)
        self.grid_idx = grid_idx
        self.warm_start_params = warm_start_params
        self.mark_converged = mark_converged

        if self.sampler.direct_eval_mode:
            # Direct-eval mode: a single target evaluation at the grid centre.
            self.pop_size = 1
            self.n_prof_dims = 0
            self.all_profiled_params = np.empty((1, 0))
            grid_coords = self.sampler._get_grid_coords_from_indices(self.grid_idx)
            self.all_full_params = [grid_coords]
            self.fitnesses = np.full(1, -np.inf)
            self.evals_remaining = 1
            return

        self.pop_size = self.sampler.pop_per_grid_point
        self.n_prof_dims = self.sampler.n_prof_dims
        prof_bounds = self.sampler.bounds[self.sampler.profiled_dims]

        # Mixed initialization: warm-start neighbour + global-pool + random LHS.
        # Slots that can't be filled fall through to the random LHS bucket.
        mix_ratios = self.sampler.activation_mix_ratios
        n_from_neighbors = int(self.pop_size * mix_ratios['neighbors'])
        n_from_global = int(self.pop_size * mix_ratios['global'])
        n_from_random = self.pop_size - n_from_neighbors - n_from_global

        samples_list = []

        # Number of leading population slots seeded from the neighbour
        # warm-start (the warm-start itself plus its perturbations). Used by
        # the allow_early_DE_exit multimodality guard: if the best activation
        # individual came from one of these slots, the neighbour seed wasn't
        # beaten by a cold random/pool seed.
        self._n_warm_start_slots = (
            n_from_neighbors if (self.warm_start_params is not None
                                 and n_from_neighbors > 0) else 0
        )

        if self.warm_start_params is not None and n_from_neighbors > 0:
            samples_list.append(self.warm_start_params)
            for _ in range(n_from_neighbors - 1):
                perturbation = np.random.normal(0, WARM_START_PERTURBATION_STD, size=self.n_prof_dims)
                perturbed = self.warm_start_params + perturbation * (prof_bounds[:, 1] - prof_bounds[:, 0])
                perturbed = self.sampler._ensure_bounds(perturbed, self.sampler.profiled_dims)
                samples_list.append(perturbed)
        else:
            n_from_random += n_from_neighbors

        global_samples = self.sampler._sample_from_global_pool(n_from_global)
        if global_samples is not None:
            samples_list.extend(global_samples)
        else:
            n_from_random += n_from_global

        if n_from_random > 0:
            lhs_sampler = LHS(d=self.n_prof_dims, seed=np.random.randint(LHS_SEED_MIN, LHS_SEED_MAX))
            unit_samples = lhs_sampler.random(n=n_from_random)
            samples_list.extend(prof_bounds[:, 0] + unit_samples * (prof_bounds[:, 1] - prof_bounds[:, 0]))

        # Cross-projection knowledge transfer: replace the last random LHS slot
        # with the past evaluation closest to this cell in projection-dim space.
        # Silent no-op when the pool is empty or proximity_warm_start is off.
        if (self.sampler.proximity_warm_start
                and not self.sampler.is_refinement_run
                and n_from_random > 0
                and len(self.sampler.global_solution_pool) > 0):
            cell_coords = self.sampler._get_grid_coords_from_indices(self.grid_idx)
            prox = self.sampler._sample_proximity_from_global_pool(1, cell_coords)
            if prox is not None and len(prox) > 0:
                samples_list[-1] = prox[0]

        self.all_profiled_params = np.array(samples_list)
        self.all_full_params = [
            self.sampler._construct_params(self.grid_idx, prof_params)
            for prof_params in self.all_profiled_params
        ]
        self.fitnesses = np.full(self.pop_size, -np.inf)
        self.evals_remaining = self.pop_size

    def start(self):
        return [
            {'params': full_params,
             'context': {'type': self.type, 'job_id': self.id, 'point_idx': i}}
            for i, full_params in enumerate(self.all_full_params)
        ]

    def process_result(self, result):
        self.fitnesses[result['context']['point_idx']] = result['target_val']
        self.evals_remaining -= 1
        if self.evals_remaining == 0:
            self.success = True
            self._is_finished = True
        return []

    def on_finish(self, next_job_id):
        """Promote this grid point into the sampler's population."""
        self.sampler.pending_activation_indices.discard(self.grid_idx)

        if not self.success or self.grid_idx in self.sampler.population:
            return None

        best_fitness = np.max(self.fitnesses)
        self.sampler.profile_likelihood_grid[self.grid_idx] = best_fitness

        status = 'converged' if (self.sampler.direct_eval_mode or self.mark_converged) else 'active'

        state = {
            'profiled_params': self.all_profiled_params,
            'fitnesses': self.fitnesses,
            'best_fitness': best_fitness,
            'status': status,
            'improvement_history': collections.deque(maxlen=self.sampler.convergence_window),
            'last_update_gen': 0,
            'optimizer_state': None,
            # True if the neighbour warm-start (or a perturbation of it) was the
            # best seed in this cell's activation population -- a cheap
            # single-cell unimodality signal reused by allow_early_DE_exit.
            'warm_start_best': (
                self._n_warm_start_slots > 0
                and int(np.argmax(self.fitnesses)) < self._n_warm_start_slots
            ),
        }
        if self.sampler.direct_eval_mode:
            # Stash the full param vector since direct-eval cells skip
            # _construct_params downstream (no profiled dims to recombine).
            state['full_params'] = self.all_full_params[0]
        self.sampler.population[self.grid_idx] = state

        self.sampler.activated_grid_indices.add(self.grid_idx)
        return None
