# Research-inspired extensions

This folder contains three new analyses that can be used as independent pages
in the final report. They do not replace or edit the original Q1, Q2, or Q3
work.

## What was added

1. **Q1 forecast ranges:** A point prediction is converted into 50%, 80%, and
   90% scoring ranges. The range sizes are learned on 2023-24 and checked on
   2024-25. The 80% range is also shown by scoring role.
2. **Q2 continuity adjustment:** The estimated continuity effect is shown as
   prior team strength, prior roster talent, and roster composition are added
   to the regression. This makes the confounding visible.
3. **Q3 playoff scoring decomposition:** Each change in points per game is
   split exactly into a minutes effect and a points-per-minute effect.

## Why these ideas were selected

- They answer a question that the existing reports leave open.
- They can be calculated with the data already in the repository.
- Each result has one main visual message and can fit on one report page.
- The methods avoid causal claims that the data cannot support.

## Research used for inspiration

- Yeh, Rice, and Dubin study calibration and skill in NBA probability
  forecasts. Their main visual idea is that a forecast should be checked
  against how often the event actually occurs:
  https://arxiv.org/abs/2010.00781
- Chernozhukov, Wuthrich, and Zhu develop prediction intervals that account for
  changing uncertainty. This motivated checking different interval widths for
  different scoring roles:
  https://arxiv.org/abs/1909.07889
- NBA.com's continuity rankings use returning minutes as the continuity
  measure and note that continuity is more strongly related to prior and
  current quality than to improvement:
  https://www.nba.com/news/2025-continuity-rankings
- Wang, Sarker, and Hosoi also define roster continuity as the percentage of
  minutes filled by players from the previous roster and use team-clustered
  regression models:
  https://journals.sagepub.com/doi/10.1177/15270025251328264
- A study of NBA regular-season and playoff games reports lower playoff points
  and field-goal percentage, which motivated separating opportunity from
  scoring rate:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9390892/
- FiveThirtyEight compares regular-season and playoff performance with
  per-minute advanced measures. This is a useful reminder that points per game
  can change because minutes change:
  https://fivethirtyeight.com/features/lebron-doesnt-get-better-in-the-playoffs-hes-always-this-good/

## Reproduce

From the repository root:

```bash
MPLCONFIGDIR=/tmp/nba-mpl python3 candidate_material/inspired_extensions/analysis.py
/Users/nimrodsegev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 candidate_material/inspired_extensions/build_report.py
```

Outputs are written under `output/figures`, `output/tables`, and `output/pdf`.

