"""Local web app for interactive review of QP wave events.

A small FastAPI server serving JSON for round-8 events plus their MFA
waveforms and Welch power spectra, paired with a vanilla-JS / uPlot
front-end. Run with ``uv run python -m qp.webapp``.
"""
