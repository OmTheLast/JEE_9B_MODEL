# Training safety gates for the 4B program

24 September 2026. This contract exists to prevent an experimental adapter from silently replacing a stronger unchanged baseline. It cannot guarantee that every optimizer checkpoint improves; it guarantees that a regressed checkpoint is rejected and the immutable base remains deployable.

## Immutable control

The control is pinned `mlx-community/Qwen3.5-4B-4bit` revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` with the selected solve→verify→commit harness revision. Base and vision tensors remain frozen and their recorded files are hash-checked before and after every run. Adapters use separately named experiment directories and are never written into the base snapshot.

The current reused-development reference is provisional until the state-machine harness is frozen:

- raw single pass:13/24 strict,14/24 parseable,11 caps;
- trigger-only forced serializer/normalizer:15/24 strict,24/24 parseable;
- ten questions were correct for both raw4B and raw9B and form the first preservation canary set.

These24 questions and ten canaries are development-only and never optimizer input. A new independent confirmation set is required for promotion.

## Data gates before a real run

No optimizer run beyond the isolated one-update export/recovery preflight may start until:

1. at least300 independent training families are allocated,50 per subject × Main/Advanced cell;
2. the families have zero recorded overlap with all development, canary, confirmation and protected2026 families;
3. every main target uses one concise solve→verify→commit contract;
4. source image, response contract and answer are independently checked;
5. high-risk organic structures, diagrams, thermodynamic conventions and ambiguous keys receive explicit review;
6. target-length, repetition, arithmetic, unit, option-coverage and answer-format audits pass;
7. a signed `TRAINING_RELEASE.json` records exact hashes and is created separately. This specification is not that release.

## Conservative first recipe

- Base and vision frozen.
- LoRA starts with final three language MLP layers, rank4, dropout0.
- One shuffled exposure to each independent family.
- First one-update preflight uses learning rate2e-6; a move to5e-6 is a separately named one-variable experiment only if gradient/update magnitude is too small and all integrity checks pass.
- AdamW, constant rate, batch1, gradient clipping1.0.
- Checkpoint and fully flush adapter, optimizer, scheduler metadata, MLX/Python RNG, data cursor, history and identity after every update.
- Eligible behavioral checks occur at25%,50%,75% and100% of one exposure. No second pass is scheduled in advance.

## Stop rules during training

Stop before the next segment when any eligible checkpoint has:

- a non-finite tensor/loss or identity/recovery failure;
- any preservation-canary loss versus the frozen control;
- lower strict score on the checkpoint-selection development slice;
- lower procedure points, a new zero-scoring subject/exam cell, or a cell loss greater than one answer;
- more loops, malformed final answers or answer-stage caps than control;
- two or more normal replies at20 tokens or fewer without a valid final contract.

Training loss is recorded for engineering diagnosis only. It cannot override a stop rule.

## Pilot continuation gate

The adapter and unchanged4B use identical images, state-machine harness, decoding and budgets. To proceed from a small pilot to the full300-family run, a checkpoint must:

- retain all preservation canaries;
- score at least the unchanged4B on strict answers and procedure;
- have no zero-scoring cell and no cell loss greater than one;
- preserve24/24 parseable commit-stage outputs and introduce no short corruption mode;
- either gain at least2/24 strict answers, or tie strict answers while gaining at least4/48 procedure points with fewer unresolved verification failures.

Otherwise reject the recipe and keep unchanged4B plus harness.

## Independent promotion gate

Checkpoint selection uses development data only. After selection is locked, one fresh source-reviewed120-family confirmation is opened once. Promotion requires:

- at least6 net strict-answer gains over unchanged4B plus the same harness;
- at least8/240 additional procedure points;
- no subject or Main/Advanced cell worse by more than one answer;
- no catastrophic topic cluster, new output-collapse mode or canary loss;
- native adapter recovery, portable-weight parity and browser quantization/parity gates passed.

If this gate fails, keep the unchanged model. Do not average away a failing cell or choose a later checkpoint because its training loss is lower.

## Deployment boundary

An adapter that passes development but lacks a reproducible trained-weight ONNX/Transformers.js route can remain a local research artifact only. It cannot replace the browser base. Merging, model-card promotion and website selection occur only after independent confirmation and export parity.

