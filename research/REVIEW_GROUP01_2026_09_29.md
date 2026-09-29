# Four-batch data review milestone

29 September 2026. Review and triage completed for four sequential batches of the 300-family curriculum candidate. Each batch was independently checked and checkpointed before the next. Work stopped at the agreed four-batch boundary.

| Batch | Families | Provisionally supported | Held |
|---|---:|---:|---:|
| 02 | 12 | 8 | 4 |
| 03 | 12 | 11 | 1 |
| 04 | 12 | 8 | 4 |
| 05 | 12 | 9 | 3 |
| **Total** | **48** | **36** | **12** |

## Meaning

These are data-quality outcomes, not solver-accuracy results. A family is one source problem with linked teaching exercises. Ninety-six primary machine reviews were collected using Muse Spark 1.3 Free and independent Luna reviewers, followed by coordinator source inspection and mathematical checks where applicable. This group was risk-prioritized: 43 high-risk and 5 medium-risk families. Its hold rate is not an estimate of the full dataset's error rate.

The primary hold categories are five scientific reviewer disagreements, three reviewer-explanation checks, three source defects and one candidate-derivation correction. Some holds identify reviewer mistakes, not wrong training answers. One case has both a candidate derivation defect and a reviewer disagreement. Correct final answers were not sufficient to pass flawed reasoning or incomplete inputs.

A missing shared source paragraph was restored in a separate crop and reviewed independently, but its active teaching views have not yet been rebuilt. All raw evidence is preserved privately. An initial coordinator symbol-reading concern was disproved by a higher-resolution original source; the existing target was correct.

## Release boundary

The 36 supported families are provisional review outcomes, not a released training set or human expert certification. The 12 held families are explicitly excluded from a future release until resolved or replaced and revalidated. They remain in the unreleased draft, which stays training_ready:false. No weights were trained, changed or promoted in this milestone.

The active candidate index and component hashes remained unchanged. Including the previously completed first batch, 60 current families now have review evidence; 240 families remain unreviewed, in twenty batches or five groups of four. Repairs and replacement work is separate from that count.

Recommended next action: resolve or replace the twelve held families, rebuild all affected linked exercises and validate them, then review the next four batches after the user's checkpoint. Do not use reviewer majority voting to manufacture certainty. The unchanged-base comparison and no-regression training gates remain required.

Private questions, answers, raw reviews and held-out evaluation material are not included in this publication.
