"""
Per-projection analysis of the secant-predictor benchmark JSONs.

Given a directory of JSONs produced by ``run_secant_benchmark.py`` (the
``--keep-tmp`` flag of the driver leaves them in a ``/tmp/paraprof_secant_*``
directory), this script:

  * groups runs by (target, grid, projection),
  * computes per-projection mean baseline vs secant deficit,
  * shows how often the predictor candidate beat the legacy
    best-neighbor seed across the projections,
  * highlights projections where the predictor helps the most and
    the most-helped vs least-helped projections per target.

The intent is to understand WHERE the secant predictor pays off — by
projection (some dim-pairs are stiffer than others), by grid resolution,
and by target shape.

Run with:
    python examples/analyze_secant_benchmark.py /tmp/paraprof_secant_xxx
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load_runs(directory):
    """Load every JSON in ``directory`` and group by (target, grid)."""
    by_setting = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.json'))):
        with open(path) as f:
            d = json.load(f)
        key = (d['target'], d['grid'])
        by_setting.setdefault(key, []).append(d)
    return by_setting


def per_projection_breakdown(runs):
    """Per-projection mean stats across seeds, split by config.

    Returns dict[proj_idx] -> {
        'dims': list,
        'mean_calls_baseline': float, 'mean_calls_secant': float,
        'mean_deficit_baseline': float, 'mean_deficit_secant': float,
        'secant_tested': int, 'secant_won': int,
    }
    """
    # Group by (config, projection_idx)
    grids_per_proj = {}  # (config, p) -> list of arrays (one per seed)
    calls_per_proj = {}  # (config, p) -> list of per-projection delta calls (per seed)
    diag_per_proj = {}   # (config, p) -> dict accumulators

    for r in runs:
        cfg = r['config']
        cum_calls = [p['cumulative_target_calls'] for p in r['projections']]
        per = [cum_calls[0]] + [cum_calls[i] - cum_calls[i - 1]
                                for i in range(1, len(cum_calls))]
        # NOTE on diagnostic counters: ``secant_tested`` and ``secant_won``
        # are reset to zero by _reset_for_new_projection(), so the JSON
        # already stores per-projection values. ``cumulative_target_calls``
        # is genuinely cumulative across projections (master tally),
        # hence the per-projection delta above.
        for p, proj in enumerate(r['projections']):
            key = (cfg, p)
            grids_per_proj.setdefault(key, []).append(
                np.asarray(proj['coarse_grid_values']))
            calls_per_proj.setdefault(key, []).append(per[p])
            d = diag_per_proj.setdefault(key, {'tested': [], 'won': []})
            d['tested'].append(proj['diagnostics']['secant_tested'])
            d['won'].append(proj['diagnostics']['secant_won'])

    # Pointwise best reference per projection across all configs+seeds.
    n_proj = len(next(iter(runs))['projections'])
    refs = []
    for p in range(n_proj):
        stack = []
        for cfg in ('baseline', 'secant'):
            for g in grids_per_proj.get((cfg, p), []):
                stack.append(g)
        if not stack:
            refs.append(None)
            continue
        s = np.stack(stack, axis=0)
        with np.errstate(invalid='ignore'):
            refs.append(np.nanmax(s, axis=0))

    out = {}
    for p in range(n_proj):
        dims = next(iter(runs))['projections'][p]['dims']
        row = {'dims': dims, 'proj_idx': p}
        for cfg in ('baseline', 'secant'):
            grids = grids_per_proj.get((cfg, p), [])
            calls = calls_per_proj.get((cfg, p), [])
            if not grids:
                continue
            # Per-seed deficit, then mean across seeds.
            defs = []
            for g in grids:
                mask = ~(np.isnan(g) | np.isnan(refs[p]))
                if not mask.any():
                    defs.append(float('nan'))
                    continue
                d = np.clip(refs[p][mask] - g[mask], 0.0, None)
                defs.append(float(d.mean()))
            row[f'mean_deficit_{cfg}'] = float(np.nanmean(defs))
            row[f'mean_calls_{cfg}'] = float(np.mean(calls)) if calls else float('nan')
        d = diag_per_proj.get(('secant', p), {'tested': [], 'won': []})
        row['secant_tested'] = int(np.sum(d['tested']))
        row['secant_won'] = int(np.sum(d['won']))
        row['win_rate'] = (row['secant_won'] / row['secant_tested']
                           if row['secant_tested'] else float('nan'))
        out[p] = row
    return out


def print_target_grid_table(target, grid, rows):
    print(f"\n--- {target} @ grid={grid} ---")
    print(f"  {'proj':>4s}  {'dims':<10s}  {'calls(base)':>12s}  {'calls(sec)':>11s}  "
          f"{'meanΔ(base)':>13s}  {'meanΔ(sec)':>12s}  {'ratio':>7s}  "
          f"{'tested':>8s}  {'won':>6s}  {'win_rate':>9s}")
    for p, row in sorted(rows.items()):
        ratio = (row['mean_deficit_secant'] / row['mean_deficit_baseline']
                 if row.get('mean_deficit_baseline', 0) > 0 else float('nan'))
        dims_s = str(row['dims'])
        print(f"  {p:>4d}  {dims_s:<10s}  "
              f"{row.get('mean_calls_baseline', float('nan')):>12.0f}  "
              f"{row.get('mean_calls_secant', float('nan')):>11.0f}  "
              f"{row.get('mean_deficit_baseline', float('nan')):>13.3e}  "
              f"{row.get('mean_deficit_secant', float('nan')):>12.3e}  "
              f"{ratio:>6.3f}x  "
              f"{row['secant_tested']:>8d}  "
              f"{row['secant_won']:>6d}  "
              f"{row['win_rate']:>8.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        help='Directory of run_secant_benchmark JSONs '
                             '(typically /tmp/paraprof_secant_*).')
    parser.add_argument('--target', default=None,
                        help='Restrict analysis to one target.')
    args = parser.parse_args()

    by_setting = load_runs(args.directory)
    if not by_setting:
        print(f"No JSONs found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Summary across all (target, grid) projections: where does secant help most?
    all_proj_rows = []
    for (target, grid), runs in sorted(by_setting.items()):
        if args.target and target != args.target:
            continue
        rows = per_projection_breakdown(runs)
        print_target_grid_table(target, grid, rows)
        for r in rows.values():
            r2 = dict(r)
            r2['target'] = target
            r2['grid'] = grid
            all_proj_rows.append(r2)

    if not all_proj_rows:
        return

    # Distribution summary: per-projection mean-deficit ratio.
    # Filter out trivially-perfect projections (baseline deficit ~ 0)
    # where the ratio is dominated by numerical noise. We require the
    # baseline deficit to exceed a meaningful logL threshold.
    DEFICIT_FLOOR = 1e-6
    ratios = []
    for r in all_proj_rows:
        base = r.get('mean_deficit_baseline')
        sec = r.get('mean_deficit_secant')
        if base and base > DEFICIT_FLOOR and sec is not None:
            ratios.append((sec / base, r))
    ratios.sort(key=lambda x: x[0])

    print("\n" + "=" * 100)
    print(" Top-5 projections most helped by secant (lowest deficit ratio)")
    print("=" * 100)
    for ratio, r in ratios[:5]:
        print(f"  {r['target']:<22s} g{r['grid']:>3d} "
              f"dims={str(r['dims']):>10s} "
              f"baseΔ={r['mean_deficit_baseline']:.3e} -> "
              f"secΔ={r['mean_deficit_secant']:.3e} "
              f"({ratio:.3f}x), win_rate={r['win_rate']:.1%}")

    print("\n Bottom-5 projections (where secant helps least or regresses)")
    print("=" * 100)
    for ratio, r in ratios[-5:]:
        print(f"  {r['target']:<22s} g{r['grid']:>3d} "
              f"dims={str(r['dims']):>10s} "
              f"baseΔ={r['mean_deficit_baseline']:.3e} -> "
              f"secΔ={r['mean_deficit_secant']:.3e} "
              f"({ratio:.3f}x), win_rate={r['win_rate']:.1%}")

    # Correlation summary
    print("\n" + "=" * 100)
    print(" Win-rate vs deficit-ratio correlation")
    print("=" * 100)
    win_rates = np.array([r['win_rate'] for _, r in ratios if np.isfinite(r['win_rate'])])
    deficit_ratios = np.array([d for d, r in ratios if np.isfinite(r['win_rate'])])
    if len(win_rates) > 1:
        corr = float(np.corrcoef(win_rates, deficit_ratios)[0, 1])
        print(f"  Pearson r ({len(win_rates)} projections): {corr:.3f}")
        print(f"  (negative means: higher win rate -> lower deficit, "
              f"i.e. predictor helps where it fires)")
    # Print histogram of win rates.
    if len(win_rates):
        bins = [0, .1, .25, .5, .75, .9, 1.0]
        counts, _ = np.histogram(win_rates, bins=bins)
        print(f"\n  Win-rate histogram across {len(win_rates)} projections:")
        for i, c in enumerate(counts):
            lo = bins[i]; hi = bins[i + 1]
            bar = '#' * c
            print(f"    [{lo:>4.0%}, {hi:>4.0%})  {c:>3d}  {bar}")


if __name__ == '__main__':
    main()
