# Separate visual frontend: objective ablation plan

## Decision in advance

A small visual model is useful only if it reduces visual mistakes without becoming a lossy single point of failure. The default hypothesis is **hybrid assistance**, not replacement:

```text
question image ──► small visual/OCR model ──► structured transcription + uncertainty
       └────────────────────────────────────► solver also receives original image
```

Why it may help: crop/readability checks, OCR of tables and dense text, normalized option/value extraction, explicit uncertainty, and the ability to unload the visual frontend before the solver generates. Why it may hurt: transcription can erase arrows, graph curvature, spatial relations, stereochemistry and handwritten/superscript distinctions; errors propagate confidently; two browser models increase download, memory and latency.

**Current product recommendation:** start with the selected solver's integrated vision path. Do not ship a second visual model in the first browser build. A roughly770MB helper beside an approximately2.5–3GB quantized solver is a material phone/browser cost, while every main candidate already accepts images. Keep the separate model only if the ablation below shows a meaningful end-to-end gain, and load it on routed image categories rather than every question.

## Candidates

- First: `LiquidAI/LFM2.5-VL-450M-ONNX`, because its official browser variant is approximately770MB and it has documented ONNX/WebGPU support.
- Fallback comparison: a500M or2.2B SmolVLM build if the first model loses JEE symbols or relations.
- Traditional OCR may later be a cheaper fourth condition for clean printed text, but formula/diagram handling must be measured rather than assumed.

## Twelve-item visual set

Use12 development-only questions: four equation/table dense pages, four geometry/physics graphs or diagrams, and four chemistry structures/reaction schemes. Include subscripts/superscripts, Greek letters, fractions, units, option labels and at least one ambiguous crop.

The visual frontend must output only:

- verbatim visible text and equations;
- values, units and option labels;
- diagram entities and directed/spatial relations;
- unreadable/uncertain regions with coordinates or descriptions;
- no solution, answer guess or hidden correction.

## Three solver conditions

1. **Integrated:** solver receives only the original question image and common prompt.
2. **Text bottleneck:** solver receives only the small model's structured transcription.
3. **Hybrid:** solver receives the original image plus the structured transcription and uncertainty flags.

Run the same solver checkpoint, decoding and budget. Score symbol/value accuracy, critical-fact recall, diagram-relation accuracy, end-to-end strict answers, procedure, caps, latency, downloads and peak memory.

## Keep/reject rule

Keep the frontend as an optional hybrid component only if condition3 improves by at least2/12 strict answers or3/24 procedure points over condition1, introduces no category regression greater than one answer, has at most one critical transcription error, and fits the device budget when loaded sequentially. Do not make condition2 the default unless it matches integrated/hybrid accuracy and provides a substantial measured memory benefit. If gains occur only on text/table items, route it only for those detected layouts rather than all questions.
