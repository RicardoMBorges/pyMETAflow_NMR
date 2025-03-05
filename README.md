# PyMetaboFlow_NMR
This provides a comprehensive toolkit for NMR/metabolomics data processing and analysis. It includes modules for data extraction from MNova exports, spectral filtering and masking, referencing and alignment, data transformation, normalization, scaling, and advanced multivariate analyses. The processed data can also be exported to MetaboAnalyst.

# NMR/Metabolomics Data Processing Toolkit

This repository contains a comprehensive Python toolkit for processing, analyzing, and visualizing NMR and metabolomics data. It was developed to streamline data workflows in metabolomics research by providing a collection of functions for:

- **Data Extraction & Import:** Reading and converting data exported from MNova (TXT to CSV) and organizing multiple CSV files.
- **Preprocessing:** Filtering chemical shift ranges, masking unwanted spectral regions, and removing unwanted samples.
- **Spectral Referencing & Alignment:** Aligning spectra by referencing a common peak (e.g., DSS at 0.0 ppm) and applying several alignment methods (ICOSHIFT, RAFFT, PAFFT) to correct spectral shifts.
- **Visualization:** Creating interactive plots to display raw, filtered, and aligned spectra, including overlapping plots, vertical multiplots, and histograms.
- **STOCSY Analysis:** Performing Statistical Total Correlation Spectroscopy (STOCSY) to analyze covariance and correlation across the spectra.
- **Data Transformation, Normalization & Scaling:** Applying different data transformations (log, square root, cube root), normalization (Z-score, PQN, etc.), and scaling (mean centering, Pareto, range scaling, etc.) techniques.
- **Multivariate Analysis:** Running Principal Component Analysis (PCA) and Partial Least Squares Discriminant Analysis (PLS-DA) for dimensionality reduction, group separation, and evaluating variable importance (VIP).
- **Export:** Preparing the processed data for further analysis in tools such as MetaboAnalyst.

## Features

- **Modular Design:** The code is organized into clear sections and functions within the `data_processing_NMR.py` module.
- **Interactive Visualizations:** Utilize Plotly and Matplotlib for interactive and static plotting.
- **Advanced Alignment Methods:** Compare multiple spectral alignment approaches to select the most suitable one for your data.
- **Comprehensive Data Workflow:** From raw data extraction to final multivariate analysis and VIP evaluation, the toolkit covers all major steps in metabolomics data processing.
- **Easy Integration:** Designed to be incorporated into Jupyter Notebooks for an interactive and reproducible analysis workflow.

## Installation

