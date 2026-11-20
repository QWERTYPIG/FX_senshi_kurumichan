# Feature changes require walk-forward evidence

An earlier agent session picked model inputs from prior knowledge alone, with nothing in the data showing those inputs helped. We decided that model inputs are chosen by evidence from validation, never by belief, so every input in a model traces to either the default rule or a recorded ablation.

## Rule

- **Default inputs:** every column the data provides that is available at the information cutoff, enters the model. Columns unavailable at the cutoff, such as `y`, are excluded on timing grounds alone.
- **Uniform transforms:** a transform is chosen by a column's unit (for example, price-denominated columns versus counts or ratios) and applied to every column of that unit. A column's expected usefulness plays no part.
- **Feature change:** adding a derived input, removing a column, or giving one column its own transform is a feature change. Adopt it only when a walk-forward ablation, with everything else held fixed, raises the mean contest score across validation folds *and* raises it in a majority of those folds.
- **Untouched test:** ablations read validation folds only. The final test period stays out of every feature decision.
- **Record:** write each ablation's configuration and per-fold contest scores into the model's docs, adopted or rejected, so the survey report can cite it.

Literature, intuition, or a teammate's model can motivate *trying* a feature change; only the ablation result adopts it.

## Scope

Applies to every feature change made in this repo from 2026-11-20 on. Existing models keep their inputs until someone changes them.
