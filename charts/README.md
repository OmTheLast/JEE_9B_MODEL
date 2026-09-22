# What training changed, in charts

![Answer scores at checkpoints in three separate studies](training_trajectories.png)

**Checkpoint trajectories.** Each panel uses a different development set; the dashed line is that study's unchanged 9B result. The first 30-family run gained on its 12-question validation at step 60 but dropped sharply later. Recipe02's two exercise mixtures reached 13/18 early, then developed short-output failures. Reliable01's lower-rate, uniform-target run remained stable but tied the base at 7/12. The first run's separate 29-question evaluation tied 22/29; Recipe02's separate 36-question comparison was base 19, A 19, B 20. The graph does **not** combine those evaluation sets into a single trend. Sources: [first controlled report](../EXPERIMENT_HISTORY.md), [Recipe02 report](https://github.com/OmTheLast/JEE_9B_MODEL/blob/main/EXPERIMENT_HISTORY.md) and the original local experiment records described there. Values are transcribed into [`training_trajectories.csv`](training_trajectories.csv).

![Exact answers and procedural validity for the latest series](latest_tradeoffs.png)

**Latest controlled series.** Every row uses the same 12-question Validation01 development slice, so the two panels can be read together. The 16-layer arm gives one additional exact answer but substantially fewer procedure points. The 60-family arm ties the base on answers and has fewer procedure points. Procedure points are coordinator judgments: valid=2, partly valid=1, invalid=0; they are not independent expert certification. The set was reused for checkpoint decisions and cannot establish broad JEE accuracy. Data come from [`history.csv`](../history.csv) and the per-arm decisions in [EXPERIMENT_HISTORY.md](../EXPERIMENT_HISTORY.md).

Rebuild with `python3 -m pip install matplotlib` then `python3 charts/make_charts.py` from the repository root. Both SVG and PNG files are generated. The charts contain no exam questions, model response text, private keys or user attempts. The protected 2026 final set was not used.
