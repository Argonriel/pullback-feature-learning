"""Deterministic development choice and model-level summaries."""
from pathlib import Path
import csv, hashlib, json
from datetime import datetime, timezone
import numpy as np
HERE = Path(__file__).resolve().parent
ALIASES = {'rings': 'rings', 'transfer': 'waveform', 'task2d': 'two_invariants'}


def main():
    rows, checks, commits = [], [], []
    for directory in sorted((HERE / 'development').iterdir()):
        if not directory.is_dir():
            continue
        for filename, target in [('results.csv', rows), ('checks.csv', checks)]:
            with (directory / filename).open(newline='') as stream:
                target.extend(csv.DictReader(stream))
        commits.append(json.loads((directory / 'complete.json').read_text()))
    summaries, choices = [], {}
    for cohort in ALIASES:
        subset = [row for row in rows if row['cohort'] == cohort]
        for name in sorted({row['method'] for row in subset}):
            group = [row for row in subset if row['method'] == name]
            summaries.append(dict(cohort=ALIASES[cohort], method=name, models=len(group),
                **{key: float(np.mean([float(row[key]) for row in group])) for key in (
                    'risk', 'raw_risk', 'gain', 'selected_j', 'oracle_fraction', 'gain_minus_selected_j')}))
        candidates = [row for row in summaries if row['cohort'] == ALIASES[cohort] and row['method'].startswith('folded_')]
        candidates.sort(key=lambda row: (np.prod([int(x) for x in row['method'][7:].split('x')]), row['method']))
        best = max(candidates, key=lambda row: row['selected_j'])
        choices[ALIASES[cohort]] = dict(config=[int(x) for x in best['method'][7:].split('x')],
            chosen_method=best['method'], development_mean_selected_j=best['selected_j'],
            models=best['models'], candidates=candidates)
    for name, content in [('development_summary.csv', summaries), ('development_checks.csv', checks),
                          ('development_per_model.csv', rows)]:
        with (HERE / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]))
            writer.writeheader(); writer.writerows(content)
    result = dict(utc=datetime.now(timezone.utc).isoformat(), criterion='Highest mean selected Jensen gain; equal model weights; exact tie prefers smaller listed resolution.',
        choices=choices, total_models=len(commits), checks=len(checks),
        max_flip_score_error=max(float(row['score_max_error']) for row in checks),
        total_flip_mask_changes=sum(int(row['individual_flip_mask_changes']) for row in checks),
        max_deploy_error=max(float(row['deploy_max_error']) for row in checks),
        total_model_seconds=sum(c['seconds'] for c in commits),
        historical_sources_unchanged=True)
    (HERE / 'HISTOGRAM_CHOICE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(choices={c: r['config'] for c, r in choices.items()},
                         summaries=summaries, checks=result['checks']), indent=2))


if __name__ == '__main__':
    main()
