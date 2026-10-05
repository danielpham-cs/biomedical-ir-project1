Project 2 submission

report.pdf: full report, including Top 50 tables for A–D.
executive_summary.pdf: one-page summary.
analysis.json: complete statistics, settings and fits.
corpus_snapshot.json: frozen abstract texts and source hashes.
tables/: all frequencies, selected terms, Porter groups, model neighbors and PCA.
figures/: plotted evidence.
skipgram.npz and training.json: trained model/config/history.
run_metadata.json: checksums and environment.
Reproduce the current manifest: .\.venv\python.exe scripts\run_project2.py
Or replay frozen texts: python scripts/run_project2.py --snapshot output/project2_glp1/corpus_snapshot.json
