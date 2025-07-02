# Bayesian On-line Learning of the Hazard Rate

This repository provides simple Python implementations for Bayesian on-line learning algorithms that estimate changepoint locations using hazard rate models. Two example classes are included:

- `ConstantHazardRate` implements a single-level model with a constant hazard rate.
- `ThreeLevelChangePointHierarchy` implements a three-level hierarchy of hazard rates.
These classes are implemented in `constant_hazard_rate.py` and `hierarchical_changepoint.py` respectively.

An additional example of an Expectation-Maximization algorithm is provided in
`em.py`. Running the module will show how means are updated over several
iterations.

Each module contains a small ``__main__`` block demonstrating usage. Run any of
the files directly to see how weights are propagated and normalized to produce
predictions over streaming data.

## Requirements

The hazard rate examples rely on `numpy` for generating random data.
The EM algorithm implementation in :mod:`em.py` uses only the Python
standard library.

## Installation

Install the project using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

This will create a virtual environment with Python 3.10 and install all
runtime and development dependencies.

## Author

Diogo Ribeiro ([@DiogoRibeiro7](https://github.com/DiogoRibeiro7))  
ESMAD - Instituto Politécnico do Porto  
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)  
Contact: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
Alternate contact: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)

## Testing

Run the tests with `pytest`:

```bash
pytest
```

Tests are also executed automatically on each push or pull request via
GitHub Actions.

All functions are documented using Google-style docstrings with inline
comments for clarity.

