# Candidate Material for the Final NBA Report

This folder is deliberately separate from the original Q1 and Q2 work. It
contains alternative analyses and figures that can be cherry-picked into the
final group report without modifying anyone else's files.

## What is different

### Q1 - next-game scoring

- Every feature is available before tip-off.
- Models are trained on 2019-20 through 2023-24 and evaluated only on the
  unseen 2024-25 season.
- Machine learning is compared with player-specific baselines, including
  the player's season-to-date average and five-game average.
- Results include cluster-bootstrap uncertainty and error profiles by scoring
  role.

### Q2 - team chemistry

- Chemistry is represented by roster continuity: the share of rotation minutes
  played by players who were on that same team one year earlier. Rotation
  players are required to log at least 150 minutes in the season.
- Player-team minutes are reconstructed from game logs, so traded players are
  assigned to the teams for which they played.
- The analysis controls for prior team strength and the prior-season quality of
  the current roster.
- Evaluation leaves one entire season out at a time. Claims are explicitly
  associational rather than causal.

## Run

Use the repository's Python environment with pandas, NumPy, SciPy,
scikit-learn, matplotlib, and seaborn installed:

```bash
pip install -r candidate_material/requirements.txt
MPLCONFIGDIR=/tmp/nba-mpl python3 candidate_material/run_all.py
```

Then generate the two compact candidate PDFs with a Python environment that
has ReportLab:

```bash
python3 candidate_material/build_reports.py
```

Outputs are written only inside `candidate_material/output/`.

## Output layout

```text
candidate_material/output/
  figures/    publication-quality PNG and PDF charts
  tables/     modeling datasets and result tables
  pdf/        compact Q1 and Q2 candidate write-ups
```
