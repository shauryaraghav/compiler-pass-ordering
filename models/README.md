# Trained models

None. The Gaussian Process surrogate (`run_all.py`: `featurize`, `kern`, `gp_fit`, `gp_predict`) is fitted from scratch
inside every search from at most 20 observations; nothing is pre-trained or saved.
The only persisted learned artefact is the warm-start sequence store `results/warm.json`
(best verified sequence per program, produced by the proposed method).
