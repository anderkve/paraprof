"""Provenance phases for the unified sample log.

Every evaluation paraprof records streams into the single
``samples_output_file`` as a row ``[params..., logL, phase]``, where
``phase`` is one of the integer constants below identifying the algorithm
stage that produced the point. Filtering on this column recovers any
subset, e.g. the volume-sampling evaluations or the refinement points.

The master derives the phase from the producing job's ``type`` (see
:data:`JOB_TYPE_TO_PHASE`).
"""

PHASE_UNKNOWN = -1        # producing stage not identified
PHASE_INITIAL = 0         # initial optimization (global maxima / basin detection)
PHASE_SCAN = 1            # projection-grid scan (activation, DE, grid polish)
PHASE_REFINE = 2          # grid refinement and patching waves
# 3 was the suspect-cell recheck, a stage that no longer exists. Sample files
# of earlier versions can hold it.
PHASE_VOLUME = 4          # volume sampling (umbrella-ensemble walkers)

# Map each job's ``type`` string to the phase its evaluations belong to.
JOB_TYPE_TO_PHASE = {
    'NOISE_ESTIMATE': PHASE_INITIAL,
    'PERIODIC_CHECK': PHASE_INITIAL,
    'INITIAL_POINT_EVAL': PHASE_INITIAL,
    'INITIAL_OPTIMIZATION': PHASE_INITIAL,
    'ACTIVATE_GRID_POINT': PHASE_SCAN,
    'DE_GRID_POINT': PHASE_SCAN,
    'LBFGSB': PHASE_SCAN,
    'LBFGSB_LOOP': PHASE_SCAN,
    'PATCHING_TEST': PHASE_REFINE,
    'PATCHING_LBFGSB': PHASE_REFINE,
    'REFINEMENT_LBFGSB': PHASE_REFINE,
    'VOLUME': PHASE_VOLUME,
}

# Human-readable description per phase (e.g. for documentation / summaries).
PHASE_LEGEND = {
    PHASE_UNKNOWN: 'unidentified phase',
    PHASE_INITIAL: 'initial optimization (global maxima / basin detection)',
    PHASE_SCAN: 'projection-grid scan (activation, DE, grid polish)',
    PHASE_REFINE: 'grid refinement and patching',
    PHASE_VOLUME: 'volume sampling (umbrella-ensemble walkers)',
}


def phase_for_job_type(job_type):
    """The phase a job's evaluations belong to; ``PHASE_UNKNOWN`` if unmapped."""
    return JOB_TYPE_TO_PHASE.get(job_type, PHASE_UNKNOWN)
