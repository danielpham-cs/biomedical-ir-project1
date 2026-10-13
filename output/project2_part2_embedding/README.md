Project 2 part 2 submission: word embeddings (CBOW vs Skip-gram)

report.pdf / report.md: full report with the background table and RQ1-RQ5.
summary.pdf: one-page summary covering every item of the brief.
experiments.json: every measured number used in the report.
tables/: training summary, top-5 neighbors, rare-word neighbors, pair and group cosines, neighbor overlap, co-occurrence, projection purity and coordinates.
figures/: loss curves, PCA and t-SNE maps, parameter and perplexity comparisons, seed stability, group cosines.
models/: the ten trained models (NumPy .npz; not in submission.zip, rebuilt by the script).
run_metadata.json: settings, environment and checksums.
Reproduce: .\.venv\python.exe scripts\run_project2_part2.py
