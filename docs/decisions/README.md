# Architecture decision records

Every choice that can change a result (data splits, thresholds, models,
preprocessing, simulation assumptions) gets a short record here, numbered in
order: `NNNN-short-title.md`. Records are not edited after acceptance; a later
record supersedes an earlier one and says so.

Template:

```markdown
# NNNN. Title

- Status: proposed | accepted | superseded by NNNN
- Date: YYYY-MM-DD

## Context
What forces the decision and what it affects.

## Decision
What was chosen, with the parameters that matter.

## Consequences
What becomes valid, what becomes invalid, what must be checked.
```
