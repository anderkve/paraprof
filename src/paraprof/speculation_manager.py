"""
Speculation manager for coordinating speculative parallel evaluation.

This module manages the lifecycle of speculative work, including prediction,
state tracking, merging, and discarding.
"""
import time
import collections
import numpy as np
from .logger import get_logger
from . import prediction_models

logger = get_logger()


class SpeculationMetrics:
    """Track speculation performance metrics."""

    def __init__(self):
        self.predictions_made = 0
        self.predictions_correct = 0
        self.predictions_incorrect = 0

        self.speculative_evals_total = 0
        self.speculative_evals_merged = 0
        self.speculative_evals_discarded = 0
        self.speculative_evals_salvaged = 0

        self.merge_events = []  # [(grid_idx, timestamp, evals_saved)]
        self.discard_events = []  # [(grid_idx, timestamp, reason)]

        self.worker_idle_time_saved = 0.0  # Estimate

    def report(self):
        """
        Generate speculation performance report.

        Returns
        -------
        dict
            Performance metrics
        """
        accuracy = self.predictions_correct / max(1, self.predictions_made)
        efficiency = self.speculative_evals_merged / max(1, self.speculative_evals_total)

        return {
            'prediction_accuracy': accuracy,
            'speculation_efficiency': efficiency,
            'evaluations_saved': self.speculative_evals_merged,
            'evaluations_wasted': self.speculative_evals_discarded - self.speculative_evals_salvaged,
            'worker_idle_time_saved': self.worker_idle_time_saved,
            'predictions_made': self.predictions_made,
            'predictions_correct': self.predictions_correct
        }

    def log_summary(self):
        """Log a summary of speculation metrics."""
        report = self.report()
        logger.info(f"Speculation Metrics: "
                   f"Accuracy={report['prediction_accuracy']:.2%}, "
                   f"Efficiency={report['speculation_efficiency']:.2%}, "
                   f"Evals Saved={report['evaluations_saved']}, "
                   f"Predictions={self.predictions_made} "
                   f"({self.predictions_correct} correct)")


class SpeculationManager:
    """
    Manages speculative evaluation state and predictions.

    This class coordinates all speculative work including:
    - Predicting likely-to-be-activated grid points
    - Managing shadow speculative state
    - Merging speculative state when predictions are correct
    - Discarding and salvaging when predictions are incorrect
    """

    def __init__(self, sampler):
        """
        Initialize the speculation manager.

        Parameters
        ----------
        sampler : ProfileProjector
            The main sampler instance
        """
        self.sampler = sampler

        # Store speculative state in sampler for access from jobs
        if not hasattr(sampler, 'speculative_population'):
            sampler.speculative_population = {}
        if not hasattr(sampler, 'speculative_eval_cache'):
            sampler.speculative_eval_cache = {}

        self.speculative_population = sampler.speculative_population
        self.speculative_eval_cache = sampler.speculative_eval_cache

        self.speculation_predictions = {}  # {grid_idx: {'priority', 'reason', 'timestamp'}}
        self.speculative_gradients = {}  # {(job_id, position_hash): gradient_data}

        self.metrics = SpeculationMetrics()

        # Track when predictions were made (for aging)
        self.prediction_generation = {}  # {grid_idx: generation_number}

        # Auto-tuning state
        self.last_metrics_log = time.time()
        self.metrics_log_interval = 30.0  # seconds

    def predict_next_activations(self, n_predictions=10):
        """
        Predict which grid points are likely to be activated next.

        Combines multiple prediction strategies and ranks targets.

        Parameters
        ----------
        n_predictions : int
            Maximum number of predictions to return

        Returns
        -------
        list of (grid_idx, priority, reason)
            Sorted by priority (highest first)
        """
        if not self.sampler.enable_speculation:
            return []

        active_indices = self.sampler.active_grid_indices

        if not active_indices:
            return []

        # Collect predictions from all enabled strategies
        all_predictions = []

        if 'neighbor_proximity' in self.sampler.speculation_strategies:
            neighbor_preds = prediction_models.predict_neighbor_activations(
                self.sampler, active_indices
            )
            all_predictions.extend(neighbor_preds)

        if 'wavefront_extrapolation' in self.sampler.speculation_strategies:
            wavefront_preds = prediction_models.predict_wavefront_extrapolation(
                self.sampler, active_indices
            )
            all_predictions.extend(wavefront_preds)

        if 'neighbor_interpolation' in self.sampler.speculation_strategies:
            interpolation_preds = prediction_models.predict_by_neighbor_interpolation(
                self.sampler, active_indices
            )
            all_predictions.extend(interpolation_preds)

        # Rank and limit predictions
        ranked_predictions = prediction_models.rank_speculation_targets(
            all_predictions, max_targets=n_predictions
        )

        # Update tracking
        for grid_idx, priority, reason in ranked_predictions:
            if grid_idx not in self.speculation_predictions:
                self.metrics.predictions_made += 1

            self.speculation_predictions[grid_idx] = {
                'priority': priority,
                'reason': reason,
                'timestamp': time.time()
            }
            self.prediction_generation[grid_idx] = self.sampler.current_generation

        return ranked_predictions

    def create_speculative_tasks(self, free_worker_count, active_jobs):
        """
        Generate speculative tasks when workers are idle.

        Parameters
        ----------
        free_worker_count : int
            Number of currently idle workers
        active_jobs : dict
            Currently active jobs

        Returns
        -------
        list of task dicts
            Speculative tasks to dispatch
        """
        if not self.sampler.enable_speculation or free_worker_count == 0:
            return []

        # Limit speculative work based on worker fraction
        max_spec_workers = max(1, int(free_worker_count * self.sampler.speculation_worker_fraction))

        # Get predictions
        predictions = self.predict_next_activations(n_predictions=max_spec_workers * 2)

        if not predictions:
            return []

        tasks = []

        for grid_idx, priority, reason in predictions[:max_spec_workers]:
            # Skip if already being speculatively evaluated
            if grid_idx in self.speculative_population:
                continue

            # Skip if already activated
            if grid_idx in self.sampler.active_grid_indices:
                continue

            # Create speculative activation task
            spec_tasks = self._create_speculative_activation_tasks(grid_idx, reason)
            tasks.extend(spec_tasks)

            if len(tasks) >= max_spec_workers:
                break

        return tasks

    def _create_speculative_activation_tasks(self, grid_idx, reason):
        """
        Create tasks for speculative grid point activation.

        Parameters
        ----------
        grid_idx : int
            Grid index to activate speculatively
        reason : str
            Reason for speculation

        Returns
        -------
        list of task dicts
            Tasks for speculative activation
        """
        # Initialize speculative state
        pop_size = self.sampler.pop_per_grid_point

        # Warm-start from best neighbor if available
        best_neighbor_params = None
        best_neighbor_fitness = -np.inf

        for neighbor_idx in self.sampler._get_valid_neighbors(grid_idx):
            if neighbor_idx in self.sampler.population:
                neighbor_state = self.sampler.population[neighbor_idx]
                if neighbor_state['best_fitness'] > best_neighbor_fitness:
                    best_neighbor_fitness = neighbor_state['best_fitness']
                    best_idx = np.argmax(neighbor_state['fitnesses'])
                    best_neighbor_params = neighbor_state['continuous_params'][best_idx].copy()

        # Initialize speculative population
        if best_neighbor_params is not None:
            # Use neighbor's params with small perturbations
            continuous_params = np.tile(best_neighbor_params, (pop_size, 1))
            # Add small random perturbations
            if self.sampler.n_cont_dims > 0:
                perturbations = np.random.randn(pop_size, self.sampler.n_cont_dims) * 0.1
                continuous_params += perturbations
                # Ensure bounds
                for i in range(pop_size):
                    continuous_params[i] = self.sampler._ensure_bounds(
                        continuous_params[i], self.sampler.continuous_dims
                    )
        else:
            # Random initialization
            if self.sampler.n_cont_dims > 0:
                continuous_params = self.sampler._random_continuous_params(pop_size)
            else:
                continuous_params = np.zeros((pop_size, 0))

        # Create speculative state
        self.speculative_population[grid_idx] = {
            'continuous_params': continuous_params,
            'fitnesses': np.full(pop_size, -np.inf),
            'best_fitness': -np.inf,
            'status': 'speculative_active',
            'speculation_reason': reason,
            'creation_timestamp': time.time(),
            'eval_count': 0,
            'can_merge': False
        }

        # Create evaluation tasks
        tasks = []
        for i in range(pop_size):
            full_params = self.sampler._construct_params(grid_idx, continuous_params[i])

            context = {
                'type': 'SPECULATIVE_ACTIVATION',
                'grid_idx': grid_idx,
                'individual_idx': i,
                'speculation_reason': reason,
                'is_speculative': True
            }

            tasks.append({'params': full_params, 'context': context})

        self.speculative_eval_cache[grid_idx] = []

        logger.debug(f"Created {len(tasks)} speculative activation tasks for grid {grid_idx} "
                    f"(reason: {reason})")

        return tasks

    def process_speculative_result(self, result):
        """
        Process a result from speculative evaluation.

        Parameters
        ----------
        result : dict
            Worker result from speculative task
        """
        context = result['context']
        result_type = context.get('type', '')

        if result_type == 'SPECULATIVE_ACTIVATION':
            self._process_speculative_activation_result(result)
        elif result_type == 'SPECULATIVE_GRADIENT':
            self._process_speculative_gradient_result(result)

    def _process_speculative_activation_result(self, result):
        """Process result from speculative activation task."""
        grid_idx = result['context']['grid_idx']
        individual_idx = result['context']['individual_idx']
        target_val = result['target_val']

        if grid_idx not in self.speculative_population:
            # State was already discarded or merged
            return

        spec_state = self.speculative_population[grid_idx]

        # Update speculative state
        spec_state['fitnesses'][individual_idx] = target_val
        spec_state['eval_count'] += 1

        if target_val > spec_state['best_fitness']:
            spec_state['best_fitness'] = target_val

        # Store evaluation in cache
        eval_record = {
            'params': result['params'],
            'fitness': target_val,
            'timestamp': time.time()
        }
        self.speculative_eval_cache[grid_idx].append(eval_record)

        self.metrics.speculative_evals_total += 1

        # Check if all individuals evaluated
        if spec_state['eval_count'] >= self.sampler.pop_per_grid_point:
            spec_state['can_merge'] = True
            logger.debug(f"Speculative activation complete for grid {grid_idx}, "
                        f"best fitness: {spec_state['best_fitness']:.4e}")

    def _process_speculative_gradient_result(self, result):
        """Process result from speculative gradient computation."""
        # Store gradient component for potential merge
        context = result['context']
        job_id = context.get('job_id')
        position = context.get('predicted_position')

        if position is not None:
            position_hash = hash(position.tobytes())
            key = (job_id, position_hash)

            if key not in self.speculative_gradients:
                self.speculative_gradients[key] = {
                    'components': {},
                    'position': position,
                    'timestamp': time.time()
                }

            dim = context.get('dimension')
            eps = context.get('epsilon')
            self.speculative_gradients[key]['components'][dim] = {
                'fitness': result['target_val'],
                'epsilon': eps
            }

            self.metrics.speculative_evals_total += 1

    def merge_speculative_state(self, grid_idx):
        """
        Merge shadow state into main population when speculation proves correct.

        Parameters
        ----------
        grid_idx : int
            Grid index to merge
        """
        if grid_idx not in self.speculative_population:
            return

        spec_state = self.speculative_population[grid_idx]

        # Move speculative population to main population
        self.sampler.population[grid_idx] = {
            'continuous_params': spec_state['continuous_params'].copy(),
            'fitnesses': spec_state['fitnesses'].copy(),
            'best_fitness': spec_state['best_fitness'],
            'status': 'active',
            'improvement_history': collections.deque(maxlen=self.sampler.convergence_window),
            'last_update_gen': self.sampler.current_generation,
            'optimizer_state': None
        }

        # Update metrics
        self.metrics.predictions_correct += 1
        self.metrics.speculative_evals_merged += spec_state['eval_count']
        self.metrics.merge_events.append((grid_idx, time.time(), spec_state['eval_count']))

        evals_saved = spec_state['eval_count']

        # Clean up speculative state
        del self.speculative_population[grid_idx]
        if grid_idx in self.speculative_eval_cache:
            del self.speculative_eval_cache[grid_idx]
        if grid_idx in self.speculation_predictions:
            del self.speculation_predictions[grid_idx]

        logger.info(f"✓ Speculation SUCCESS! Merged grid {grid_idx} "
                   f"(saved {evals_saved} evaluations, reason: {spec_state['speculation_reason']})")

    def discard_speculative_state(self, grid_idx, reason='stale'):
        """
        Discard speculative state but salvage useful evaluations.

        Parameters
        ----------
        grid_idx : int
            Grid index to discard
        reason : str
            Reason for discarding
        """
        if grid_idx not in self.speculative_population:
            return

        spec_state = self.speculative_population[grid_idx]

        # Salvage evaluations for emulator training
        evals_salvaged = 0
        if self.sampler.use_de_prescreening and grid_idx in self.speculative_eval_cache:
            if grid_idx not in self.sampler.local_eval_caches:
                self.sampler.local_eval_caches[grid_idx] = []

            # Add speculative evaluations to cache
            for eval_record in self.speculative_eval_cache[grid_idx]:
                self.sampler.local_eval_caches[grid_idx].append(eval_record)
                evals_salvaged += 1

        # Update metrics
        self.metrics.predictions_incorrect += 1
        self.metrics.speculative_evals_discarded += spec_state['eval_count']
        self.metrics.speculative_evals_salvaged += evals_salvaged
        self.metrics.discard_events.append((grid_idx, time.time(), reason))

        # Clean up
        del self.speculative_population[grid_idx]
        if grid_idx in self.speculative_eval_cache:
            del self.speculative_eval_cache[grid_idx]
        if grid_idx in self.speculation_predictions:
            del self.speculation_predictions[grid_idx]

        logger.debug(f"✗ Discarded speculative state for grid {grid_idx} "
                    f"(reason: {reason}, salvaged {evals_salvaged} evals)")

    def cleanup_stale_speculative_state(self):
        """
        Remove speculative state that's too old or beyond depth limit.

        This prevents unbounded memory growth and ensures speculative
        work stays relevant.
        """
        current_gen = self.sampler.current_generation
        max_age_seconds = self.sampler.speculation_discard_stale_after
        max_gen_depth = self.sampler.speculation_max_depth

        current_time = time.time()
        stale_indices = []

        for grid_idx, spec_state in self.speculative_population.items():
            # Check time-based staleness
            age = current_time - spec_state['creation_timestamp']
            if age > max_age_seconds:
                stale_indices.append((grid_idx, 'time_limit'))
                continue

            # Check generation-based staleness
            if grid_idx in self.prediction_generation:
                pred_gen = self.prediction_generation[grid_idx]
                gen_depth = current_gen - pred_gen
                if gen_depth > max_gen_depth:
                    stale_indices.append((grid_idx, 'generation_limit'))

        for grid_idx, reason in stale_indices:
            self.discard_speculative_state(grid_idx, reason=reason)

    def check_and_merge_activations(self):
        """
        Check if any speculative grid points were actually activated.

        This should be called periodically to detect when speculation
        was correct.
        """
        to_merge = []

        for grid_idx in list(self.speculative_population.keys()):
            # Check if grid point was activated
            if grid_idx in self.sampler.active_grid_indices:
                to_merge.append(grid_idx)

        for grid_idx in to_merge:
            # Only merge if speculative state is ready
            spec_state = self.speculative_population[grid_idx]
            if spec_state.get('can_merge', False):
                self.merge_speculative_state(grid_idx)

    def log_metrics_if_needed(self):
        """Log metrics periodically."""
        current_time = time.time()
        if current_time - self.last_metrics_log > self.metrics_log_interval:
            if self.metrics.predictions_made > 0:
                self.metrics.log_summary()
            self.last_metrics_log = current_time

    def get_speculative_gradient(self, job_id, position):
        """
        Get pre-computed speculative gradient if available.

        Parameters
        ----------
        job_id : int
            L-BFGS-B job ID
        position : np.ndarray
            Current position to check

        Returns
        -------
        np.ndarray or None
            Gradient vector if available and matches position
        """
        position_hash = hash(position.tobytes())
        key = (job_id, position_hash)

        if key not in self.speculative_gradients:
            return None

        grad_data = self.speculative_gradients[key]

        # Check if all components are available
        n_dims = len(position)
        if len(grad_data['components']) < n_dims:
            return None

        # Construct gradient vector
        # This is a simplified version - real implementation would need
        # to compute finite differences properly
        return None  # Placeholder - full gradient computation not implemented
