# Training-question inventory

18 October2026. Dataset bookkeeping milestone; no training or model improvement claim.

A private searchable inventory now tracks each active source problem once, with its three teaching exercises separately linked. It distinguishes authoring batches from review batches and records source images, recorded answer formats, methods, review state and metadata provenance. Immutable snapshots allow later comparisons without overwriting earlier counts.

|Measure|Count|
|---|---:|
|Active source-problem families|300|
|Teaching exercises|900|
|Families in each subject × Main/Advanced cell|50|
|Recorded option-based questions|242|
|Recorded numeric-entry questions|58|
|Provisionally supported families|132|
|Held for follow-up|72|
|Partially reviewed|12|
|Not yet reviewed|84|
|Difficulty unrated|300|

The three teaching views cover checking a proposed step, composing a method and solving with a final answer. They are not three independent source problems. Supported remains provisional automated-review evidence; the candidate is not released for training.

Difficulty was not consistently annotated. A separate attributed, rationale-based annotation workflow is defined; exam label, review risk and solution length are not used as automatic difficulty labels. Source single-correct versus multiple-correct format also needs explicit annotation: one selected answer does not identify the question format. Topic/method suggestions are transparent lexical matches, can overlap or be wrong, and are not validated curriculum coverage. Twenty-seven families have no topic suggestion; sixty legacy source years remain unknown.

Verification:300unique family IDs,900unique view IDs, exact three-view contract and six balanced cells.408read source files retain their hashes;198prior preservation checks passed. Eleven checks exercised filtering, counts, search and pagination logic, and local source links resolve. Rendered browser/layout QA was not performed. No original question, answer, image, private identifier or reviewer evidence is included in this public update.

Next: maintain the inventory as reviews and repairs progress; add source-checked question subtypes and reasoned difficulty annotations; freeze a release only after scientific, split, format and processor gates. These are data-quality counts, not solver accuracy or a history of actual optimizer exposures.
