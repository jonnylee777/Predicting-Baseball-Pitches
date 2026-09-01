# Pitch Prediction — Modeling Experiments

Last updated: August 20, 2026

## 1. Purpose

This document tracks modeling experiments, results, and design decisions for the
MLB Pitch Prediction project.

The goal is to avoid making modeling changes based only on intuition and to
maintain a record of:

- the hypothesis behind each experiment,
- the exact modeling change,
- how the experiment was evaluated,
- the results,
- conclusions,
- and whether the change was promoted to the production pipeline.

The production objective is to train a separate Random Forest model for each
starting pitcher and use that model to predict pitch type during future games.

---

# 2. Evaluation Methodology

## Chronological train/test split

Models are evaluated using games in chronological order.

Approximately:

- Oldest 80% of games → training
- Newest 20% of games → testing

This was chosen instead of a random game split because the real application
always predicts future games using past information.

Example:

2018 ───────────── 2024 | 2025 ─── 2026
       TRAINING             TEST

This prevents the evaluation model from learning from future games.

After evaluation, the eventual production model can be retrained on all
available historical data through the day before the upcoming game.

---

# 3. Baseline Model

The primary model is a Random Forest classifier trained separately for each
pitcher.

Current Random Forest configuration:

| Parameter | Value |
|---|---:|
| n_estimators | 800 |
| max_depth | 15 |
| min_samples_split | 20 |
| min_samples_leaf | 5 |
| max_features | log2 |
| bootstrap | True |
| random_state | 42 |
| n_jobs | -1 |

The preprocessing pipeline performs:

- numeric missing-value imputation,
- categorical missing-value imputation,
- one-hot encoding,
- Random Forest classification.

The saved sklearn Pipeline contains both preprocessing and the model.

---

# 4. Experiment 1 — Career Recency Weighting

## Hypothesis

Recent pitches may be more representative of how a pitcher currently pitches
than pitches thrown many years ago.

Instead of treating all historical pitches equally, newer seasons receive
higher training weights.

The initial adaptive decay equation was:

decay = max(
    0.65,
    0.92 - 0.02 × (career_seasons - 1)
)

and:

sample_weight = max(
    0.05,
    decay ^ seasons_ago
)

Current season receives weight 1.0.

The original hypothesis was that pitchers with long careers should have faster
decay because very old data may be less representative.

---

## Initial 10-pitcher test

The initial sample contained:

| Pitcher |
|---|
| Kyle Bradish |
| Gerrit Cole |
| Shane Bieber |
| Anthony Kay |
| Ian Seymour |
| Randy Dobnak |
| Grayson Rodriguez |
| Gage Jump |
| George Kirby |
| Jacob deGrom |

Results:

- Weighted model beat unweighted: 4/10 pitchers
- Unweighted beat weighted: 6/10 pitchers
- Mean weighting effect: approximately +0.27 percentage points
- Median effect: approximately -0.09 percentage points

Anthony Kay represented a large positive outlier at approximately +3.74 pp.

Without Anthony Kay, the mean effect was approximately -0.12 pp.

## Conclusion

The aggressive career weighting formula was not strongly supported.

The hypothesis that longer-career pitchers automatically require stronger
decay was also not supported by this sample.

---

# 5. Experiment 2 — Mild Recency Weighting

## Hypothesis

Recency is still intuitively useful, but the original decay was too aggressive.

A milder formula was tested:

decay = max(
    0.80,
    0.98 - 0.01 × (career_seasons - 1)
)

with:

minimum_sample_weight = 0.10

For example, a pitcher with approximately 10 seasons has a decay near 0.89
instead of approximately 0.74 under the aggressive formula.

---

## Results

The mild version produced relatively small changes in accuracy.

The overall result was close to neutral, but it avoided aggressively
discounting useful historical data.

## Decision

Mild recency weighting is currently preferred over aggressive recency
weighting as the working design.

However, it remains part of the experimental modeling work until the final
production model is updated.

---

# 6. Experiment 3 — Categorical Repertoire Weighting

## Motivation

Generic recency may not be the main reason old observations become obsolete.

A more important issue is that pitchers change their repertoire.

Example:

| Season | FF | SL | CH | CU |
|---|---:|---:|---:|---:|
| 2023 | 48% | 27% | 25% | 0% |
| 2024 | 45% | 25% | 20% | 10% |
| 2025 | 43% | 30% | 2% | 25% |
| 2026 | 42% | 31% | 0% | 27% |

Old changeup observations may no longer represent the pitcher's current
behavior even though old fastball observations may still be useful.

This motivated pitch-specific weighting.

---

## Initial categorical approach

Each pitch was classified using recent and historical usage.

Example statuses:

| Status | Historical multiplier |
|---|---:|
| Stable | 1.00 |
| Emerging | 1.00 |
| Declining | 0.50 |
| Inactive | 0.10 |

Recent examples were not penalized.

Historical evaluation used only repertoire information available inside the
training period.

No information from the test games was used to determine repertoire status.

A minimum recent-season sample was required before repertoire judgments were
made.

---

## Results

Across the original 10-pitcher development sample:

| Model | Mean Test Accuracy |
|---|---:|
| Unweighted | ~40.11% |
| Mild recency | ~40.21% |
| Repertoire only | ~40.32% |
| Mild + repertoire | ~40.39% |

The categorical repertoire layer added approximately +0.18 percentage points
on top of mild recency on average.

The largest improvement occurred for Anthony Kay.

His repertoire showed several major changes:

- CU inactive
- FF declining
- SI emerging
- ST emerging

For Anthony Kay:

| Model | Accuracy |
|---|---:|
| Unweighted | 23.02% |
| Mild recency | 24.30% |
| Repertoire only | 25.32% |
| Mild + repertoire | 26.08% |

## Conclusion

Pitch-specific repertoire changes appeared more promising than simply making
all old pitches decay rapidly.

However, categorical thresholds were considered too coarse.

---

# 7. Experiment 4 — Continuous Repertoire Weighting

## Hypothesis

Pitch usage changes occur continuously rather than falling naturally into
fixed categories.

Instead of:

stable → 1.00  
declining → 0.50  
inactive → 0.10

historical weights were calculated from the magnitude of the change.

For declining pitches:

old_multiplier =
    clip(
        (recent_usage + 0.01)
        /
        (prior_usage + 0.01),
        0.15,
        1.00
    )

Example:

| Prior Usage | Recent Usage | Approx. Old Multiplier |
|---:|---:|---:|
| 20% | 18% | ~0.90 |
| 20% | 12% | ~0.62 |
| 20% | 5% | ~0.29 |
| 20% | 0% | ~0.15 |

Increasing pitches could also receive a modest boost to current-season
examples.

Maximum emerging-pitch boost:

1.50

---

## Development-sample results

Across the original 10 pitchers:

| Model | Mean Accuracy |
|---|---:|
| Unweighted | 40.11% |
| Mild recency | 40.21% |
| Continuous repertoire + recency | 40.73% |

Continuous repertoire weighting added approximately:

+0.52 percentage points over mild recency.

However, much of the gain came from Anthony Kay.

Anthony Kay:

| Model | Accuracy |
|---|---:|
| Unweighted | 23.02% |
| Mild recency | 24.30% |
| Continuous repertoire | 30.76% |

Gain over mild recency:

+6.46 percentage points.

There was also a major negative result for Kyle Bradish:

| Model | Accuracy |
|---|---:|
| Mild recency | 36.42% |
| Continuous repertoire | 34.18% |

Difference:

-2.24 percentage points.

## Interpretation

The continuous model appeared capable of detecting real repertoire
transformations, but it also reacted to normal year-to-year variation.

---

# 8. Experiment 5 — Gated Continuous Repertoire

## Hypothesis

Continuous weighting should only activate after detecting a sufficiently large
repertoire shift.

Small changes should be ignored.

Frozen development rules:

### Major decline

Activate only if:

prior_usage >= 8%

AND

recent_usage < 50% × prior_usage

### Emerging pitch

Activate only if:

prior_usage <= 2%

AND

recent_usage >= 8%

Once activated, continuous weighting is used.

---

## Development-sample results

Across the original 10 pitchers:

| Model | Mean Accuracy |
|---|---:|
| Unweighted | 40.11% |
| Mild recency | 40.21% |
| Ungated continuous | 40.73% |
| Gated continuous | 40.77% |

Gated repertoire added approximately:

+0.56 percentage points over mild recency.

Median improvement was 0.00 pp.

The gate successfully removed the large Kyle Bradish failure.

Kyle Bradish:

| Model | Accuracy |
|---|---:|
| Mild recency | 36.42% |
| Ungated repertoire | 34.18% |
| Gated repertoire | 36.42% |

Anthony Kay still improved substantially:

| Model | Accuracy |
|---|---:|
| Mild recency | 24.30% |
| Gated repertoire | 30.08% |

However, George Kirby performed worse under the gated approach.

## Conclusion

The gated model looked promising on the development sample, but this sample
had already been repeatedly examined while designing the method.

A fresh validation sample was therefore required.

---

# 9. Fresh Validation Experiment

## Experimental design

The original 10 pitchers were permanently excluded.

No repertoire thresholds were changed before running validation.

A new random seed was used.

Because only eight additional probable starters were available on
August 20, 2026, the fresh validation sample contained eight pitchers:

| Pitcher |
|---|
| Brady Singer |
| Grant Holmes |
| Robert Gasser |
| Andrew Alvarez |
| Gavin Williams |
| Peter Lambert |
| Landen Roupp |
| Michael McGreevy |

These pitchers were not used to design the weighting rules.

---

## Validation results

| Model | Mean Accuracy |
|---|---:|
| Unweighted | 34.20% |
| Mild recency | 34.40% |
| **Ungated continuous repertoire** | **34.90%** |
| Gated continuous repertoire | 34.42% |

### Ungated repertoire vs mild recency

Mean improvement:

+0.50 percentage points.

Median improvement:

approximately +0.13 percentage points.

Pitcher-level outcome:

| Outcome | Count |
|---|---:|
| Improved | 4 |
| Hurt | 1 |
| Tie | 3 |

Notable improvements:

| Pitcher | Improvement vs Mild |
|---|---:|
| Landen Roupp | +2.48 pp |
| Gavin Williams | +1.29 pp |
| Brady Singer | +0.45 pp |
| Andrew Alvarez | +0.26 pp |

Robert Gasser declined approximately:

-0.46 pp.

Even excluding Landen Roupp, the ungated repertoire model remained
approximately +0.22 pp better than mild recency on average.

---

## Gated validation result

The gated method did not reproduce its development-sample advantage.

Mean gated accuracy:

34.42%

Mean mild-recency accuracy:

34.40%

Difference:

approximately +0.02 percentage points.

The strict gate appeared to remove several useful moderate repertoire changes.

---

# 10. Current Interpretation

The experiments currently support the following working hypothesis:

FinalSampleWeight =
    MildSeasonRecencyWeight
    ×
    ContinuousRepertoireWeight

Mild season recency provides a small general preference for newer data.

Continuous repertoire weighting provides a pitch-specific correction when the
pitcher's pitch mix changes over time.

The ungated continuous repertoire method has now:

1. performed well on the 10-pitcher development sample, and
2. remained the strongest model on an independent eight-pitcher validation
   sample.

This is stronger evidence than the earlier weighting experiments.

However, the sample size remains too small to consider the method fully
validated.

---

# 11. Current Modeling Decision

## Preferred experimental model

The current leading experimental model is:

Mild Recency
+
Ungated Continuous Repertoire Weighting
+
Random Forest

The gated version is not currently preferred because it removed useful
repertoire information in the fresh validation sample.

## Production status

IMPORTANT:

The repertoire system has NOT yet been added to the production modeling
pipeline.

The experiments remain isolated in scripts under:

scripts/

The production `pitch_prediction/model.py` should not be changed until the
larger validation experiment is complete.

The production pipeline therefore remains separate from the experimental
repertoire implementation.

---

# 12. Next Validation Experiment

The repertoire equations should now be frozen.

No thresholds or multipliers should be changed based on additional individual
pitchers until a larger validation experiment is complete.

Next experiment:

Target sample:

30–50 previously unseen pitchers.

Use pitchers across multiple MLB game dates so the experiment is not limited
by the number of starters on one day.

Exclude all pitchers already examined during development or validation.

Compare only:

| Model | Purpose |
|---|---|
| Unweighted RF | Control |
| Mild recency RF | Test general time weighting |
| Mild + continuous repertoire RF | Leading candidate |

Primary evaluation metrics:

Mean accuracy  
Median accuracy improvement  
Pitcher-level wins/losses/ties  
Accuracy above baseline  
Distribution of improvement across pitchers

Particular attention should be paid to whether gains occur broadly or are
driven by a small number of pitchers undergoing dramatic repertoire changes.

---

# 13. Potential Future Improvements

The current method modifies training sample weights.

A future alternative is to explicitly model the pitcher's current repertoire
and combine that information with Random Forest probabilities.

For example:

Random Forest
    ↓
P(pitch | count, batter, previous pitch, game state, ...)

Current repertoire model
    ↓
P(current pitch usage)

Combine / calibrate probabilities
    ↓
Final pitch probabilities

This may eventually provide a cleaner separation between:

1. which pitches the pitcher currently throws, and
2. when the pitcher chooses each pitch.

This should not be implemented until the simpler weighting method has been
fully validated.

---

# 14. Important Experimental Rules

To maintain valid results:

1. Repertoire calculations must use training data only during historical
   evaluation.
2. Test-game repertoire information must never influence training weights.
3. Pitchers used for rule development should not later be treated as unseen
   validation pitchers.
4. Once a validation experiment begins, equations and thresholds must remain
   frozen.
5. Production changes should only be made after experimental results justify
   them.
6. All experiments should use the same chronological split and Random Forest
   settings when comparing weighting methods.
7. Large improvements from individual pitchers should be investigated rather
   than automatically treated as evidence of general improvement.

---

# 15. Experiment Status Summary

| Experiment | Result | Status |
|---|---|---|
| Aggressive recency weighting | Weak / inconsistent | Rejected |
| Mild recency weighting | Small, generally reasonable effect | Keep as candidate |
| Categorical repertoire | Promising but coarse | Superseded |
| Continuous repertoire | Strongest general repertoire approach | Leading candidate |
| Gated continuous repertoire | Good development result, weak validation | Not preferred |
| Mild + continuous repertoire | Best fresh-sample mean accuracy | Validate further |
| Production repertoire integration | Not yet performed | Pending |

---

# 16. Current Next Step

Run a frozen 30–50 pitcher validation experiment across multiple dates.

If mild recency + continuous repertoire weighting continues to outperform the
unweighted and mild-recency-only models without being driven by a few extreme
outliers, promote the method into the production `PitchModelTrainer`.

At that point:

1. move repertoire logic into a dedicated production module,
2. integrate repertoire weights into model training,
3. add unit tests,
4. save repertoire diagnostics with model metadata,
5. rerun the complete test suite,
6. proceed to live-game prediction and accuracy logging.