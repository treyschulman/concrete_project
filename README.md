# Concrete Compressive Strength Prediction Project

This repo contains the full workflow to predict **concrete compressive strength** using mixture composition and curing age.
The dataset originates from the UCI Concrete Compressive Strength dataset. The objective was to model the nonlinear relationships between mix proportions and strength using a variety of regression and machine learning methods.
Performance was evaluated using Mean Squared Error (MSE) on a fixed 70/30 train-test split (random seed = 598).

## Repo Structure
- data/ # split train/test data CSVs
- reports/ # PNGs exported by notebooks (pred vs actual, residuals, CV curves, etc.)
  - figures/  # visualizations related to EDA and modeling
- notebooks/ # Reproducible notebooks for EDA, 5 modeling techniques, and output summary
- src/ # auxiliary plotting functions
