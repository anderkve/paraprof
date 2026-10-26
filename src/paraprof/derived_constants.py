"""Fixed parameters of projections over derived quantities (see
ProfileProjector and docs/derived_projections_plan.md).

The polish keeps its constrained steps PROJECT_MARGIN cell widths inside a
cell's faces, a margin for the first-order projection (0.005 beat 0.001,
0.002 and 0.02 on the benchmark sets), and a run that converged on a face
ends with one evaluation EDGE_MARGIN inside the true face, which recovers
most of that margin. The recheck's extra polishes
(advanced_config['derived']['recheck_max_polishes']) take seeds within
RECHECK_TOLERANCE of the cell's value.
"""

PROJECT_MARGIN = 0.005
EDGE_MARGIN = 0.002
RECHECK_TOLERANCE = 1.0
