# Improving Spectral Clustering through Chernoff-Guided Edge Weight Transformations

## Overview

This project implements a framework for Weighted Stochastic Block Models (WSBM) with Chernoff-guided edge-weight transformations.

...


## Repository Structure

```
/
├── data/
│   ├── processed/
│   │   ├── graphs/
│   │   ├── real_world_graphs/
│   │   └── synthetic_graphs/
│   └── raw/
│       └── real_world_graphs/
│           ├── cifar10/
│           ├── fashionmnist/
│           ├── high_school_2011/
│           ├── high_school_2012/
│           ├── high_school_2013/
│           ├── mnist/
│           ├── primary_school/
│           ├── workplace_2013/
│           └── workplace_2015/
├── LICENSE
├── notebooks/
│   ├── compute.ipynb
│   ├── plot_embeddings.ipynb
│   └── plotting.ipynb
├── README.md
├── repo_tree.txt
├── reports/
│   └── figures/
│       └── stacking_vs_chernoff_argmax/
│           ├── by_graph/
│           │   ├── Beta/
│           │   ├── cifar10/
│           │   ├── fashionmnist/
│           │   ├── high_school_2011/
│           │   ├── high_school_2012/
│           │   ├── high_school_2013/
│           │   ├── LogN/
│           │   ├── mnist/
│           │   ├── primary_school/
│           │   ├── workplace_2013/
│           │   └── workplace_2015/
│           └── by_plot/
│               ├── bars_families/
│               ├── bars_families_stacked_vs_argmaxc/
│               ├── bars_transforms/
│               └── heatmap_families/
├── requirements.txt
└── src/
    ├── metrics.py
    ├── models/
    │   ├── bc_registry.py
    │   ├── beta.py
    │   ├── lognorm.py
    │   ├── wdcsbm.py
    │   └── wsbm.py
    ├── tdcsbm.py
    ├── transformations.py
    ├── twsbm.py
    ├── utils/
    │   ├── EGMM.py
    │   ├── string_utils.py
    │   └── utils.py
    └── visualization/
        └── plot_embeddings.py

```

## Installation

1. Clone the repository
2. *(Optional)* Create and activate a virtual environment
3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

...

## Data

* Place raw data in `data/raw/`
* Processed outputs in `data/processed/`

## Contributing

Contributions are welcome via pull requests. Please ensure new code is documented and adheres to existing style.