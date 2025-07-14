# Improving Spectral Clustering through Chernoff-Guided Edge Weight Transformations

## Overview

This project implements a framework for Weighted Stochastic Block Models (WSBM) with Chernoff-guided edge-weight transformations. It provides:

* Core WSBM models with different edge-weight distributions:

  * [`BetaWSBM`](src/models/beta.py)
  * [`LognormWSBM`](src/models/lognorm.py)
* A suite of transformations (`IdentityTransform`, `LogTransform`, `QuantileTransform`, `PowerTransform`, ...)
* [`twsbm`](src/twsbm.py): contains the `TWSBM` class, which models a transformed weighted stochastic block model and computes a range of structural and embedding-based metrics.
* Visualization utilities for embedding plots via [`plot_embeddings`](src/visualization/plot_embeddings.py)

## Repository Structure

```
├── data/                           # raw and processed datasets  
├── notebooks/                      # example workflows  
│   └── plot_embeddings.ipynb  
├── reports/                        # generated figures and reports  
├── src/                            # source code  
│   ├── metrics.py                  # graph metrics  
│   ├── transformations.py          # weight transforms  
│   ├── twsbm.py                    # Transformed WSBM base classes  
│   ├── models/                     # WSBM implementations  
│   │   ├── beta.py                 # Beta WSBM  
│   │   ├── lognorm.py              # Lognormal WSBM  
│   │   └── wsbm.py                 # Base WSBM  
│   ├── utils/                      # string formatting & helpers  
│   └── visualization/              # plotting code  
│       └── plot_embeddings.py  
├── requirements.txt                # Python dependencies  
└── pyrightconfig.json              # type-checker config  
```

## Installation

1. Clone the repository
2. *(Optional)* Create and activate a virtual environment
3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

Open and run the example in `notebooks/plot_embeddings.ipynb` to reproduce figures and explore different parameter settings.

## Data

* Place raw data in `data/raw/`
* Processed outputs in `data/processed/`

## Contributing

Contributions are welcome via pull requests. Please ensure new code is documented and adheres to existing style.