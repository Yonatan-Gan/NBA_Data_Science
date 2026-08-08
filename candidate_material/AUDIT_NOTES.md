# Q1 and Q2 Audit Notes

These notes explain why the alternative material exists. They are not intended
as criticism of the original authors; the original work supplied the data,
question framing, and many useful visual ideas.

## Original Q1

### Strong parts

- Clear modular pipeline for loading, preprocessing, feature engineering,
  experiments, statistics, visualization, and reporting.
- Rolling player features are shifted before calculation.
- Chronological splitting is much better than a random game split.
- The report already communicates model metrics and practical limitations.

### Reasons to keep an alternative

- The career-stat table appears to contain full-career information. Joining its
  averages to earlier games can leak future seasons into the predictors.
- `experience_years` is the rank of the season in this six-season dataset, not
  the individual player's NBA experience.
- The class named `NaiveBaselineModel` uses a global-mean dummy regressor even
  though its documentation describes a player season-to-date baseline.
- A large block of old opponent-feature code appears after a `return`, making it
  unreachable.
- A single 80/20 date cut can place the same season on both sides of the split.
  The alternative holds out the complete 2024-25 season.
- Ten pages are too long for the final group report. The candidate report is
  organized as independent one-page modules.

## Original Q2

### Strong parts

- Ambitious attempt to operationalize an inherently vague concept.
- Interesting roster archetypes and an accessible visual narrative.
- Good instinct to separate raw team success from overperformance.
- Age, role hierarchy, and international background are reasonable roster
  composition candidates.

### Reasons to keep an alternative

- Random-forest feature importance is interpreted without held-out predictive
  performance, uncertainty, or a baseline comparison.
- Importance is not direction: a 31% feature weight does not show that higher
  hierarchy improves success.
- Pythagorean overperformance is mostly close-game residual variation; calling
  it "true chemistry" is stronger than the measurement supports.
- Statements such as "proves," "drastically changes," and "definitive
  blueprint" imply causal evidence that the observational design cannot supply.
- Missing countries are assigned to the USA, which can bias international-share
  estimates.
- The alternative introduces roster continuity, reconstructs traded-player
  minutes from game logs, controls for lagged talent, and evaluates by leaving
  out entire seasons.

## Recommended use in the final report

- Keep the original motivation and whichever original visual best introduces
  each question.
- Prefer the candidate Q1 forecast ladder because it makes the baseline and
  future-season test explicit.
- Prefer the candidate Q2 incremental prediction chart or the talent-continuity
  matrix because both support a precise, defensible conclusion.
- Phrase all Q2 conclusions as associations, not prescriptions or causal laws.

