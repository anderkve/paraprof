"""Provenance phases for the unified sample log.

Every evaluation paraprof records streams into the single
``samples_output_file`` as a row ``[params..., logL, phase]``, where
``phase`` is one of the integer constants below identifying the algorithm
stage that produced the point. Filtering on this column recovers any
subset — e.g. the volume-sampling probes (the uniform draw used for the
band volume estimate), or just the refinement points.

The phase is derived master-side from the producing job's ``type`` (see
:data:`JOB_TYPE_TO_PHASE`), so no MPI payload changes are needed.
"""

PHASE_UNKNOWN = -1        # producing stage not identified
PHASE_INITIAL = 0         # initial optimization (global maxima / basin detection)
PHASE_SCAN = 1            # projection-grid scan (activation, DE, grid polish)
PHASE_REFINE = 2          # grid refinement and patching waves
PHASE_SUSPECT = 3         # suspect-cell recheck
PHASE_VOLUME_PROBE = 4    # volume sampling: anchor probe (the uniform subset)
PHASE_VOLUME_SEARCH = 5   # volume sampling: anchored search (incl. interior walk)

# Map each job's ``type`` string to the phase its evaluations belong to.
JOB_TYPE_TO_PHASE = {
    'INITIAL_POINT_EVAL': PHASE_INITIAL,
    'INITIAL_OPTIMIZATION': PHASE_INITIAL,
    'ACTIVATE_GRID_POINT': PHASE_SCAN,
    'DE_GRID_POINT': PHASE_SCAN,
    'LBFGSB': PHASE_SCAN,
    'LBFGSB_LOOP': PHASE_SCAN,
    'POST_ACTIVATION_LBFGSB': PHASE_SCAN,
    'PATCHING_TEST': PHASE_REFINE,
    'PATCHING_LBFGSB': PHASE_REFINE,
    'REFINEMENT_LBFGSB': PHASE_REFINE,
    'SUSPECT_RECHECK': PHASE_SUSPECT,
    'SUSPECT_RECHECK_LBFGSB': PHASE_SUSPECT,
    'VOLUME_PROBE': PHASE_VOLUME_PROBE,
    'VOLUME_SEARCH': PHASE_VOLUME_SEARCH,
}

# Human-readable description per phase (e.g. for documentation / summaries).
PHASE_LEGEND = {
    PHASE_UNKNOWN: 'unidentified phase',
    PHASE_INITIAL: 'initial optimization (global maxima / basin detection)',
    PHASE_SCAN: 'projection-grid scan (activation, DE, grid polish)',
    PHASE_REFINE: 'grid refinement and patching',
    PHASE_SUSPECT: 'suspect-cell recheck',
    PHASE_VOLUME_PROBE: 'volume sampling: anchor probe (the uniform subset)',
    PHASE_VOLUME_SEARCH: 'volume sampling: anchored search',
}


def phase_for_job_type(job_type):
    """The phase a job's evaluations belong to; ``PHASE_UNKNOWN`` if unmapped."""
    return JOB_TYPE_TO_PHASE.get(job_type, PHASE_UNKNOWN)
