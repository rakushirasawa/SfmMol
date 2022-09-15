# SfmMol

[//]: # (Badges)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Introduction
SFMMOL: Submodular Function Maximization for Molecules

 * Pytest script (run_pytest.sh)
    * pytest: with coverage check
    * pdoc: auto document generation
 * Pre-commit: auto format check @ commit
 * Coverage: upload coverage.xml use github action

## Installation
Clone the repository:
```
git clone https://github.com/rakushirasawa/SfmMol
cd SfmMol
conda env create -f environment.yml
conda activate molgenerator
pip install -e .
```

## Test (and generate documents)
```
./run_pytest.sh
```

## Commit with pre-commit
```
pre-commit install #only fisrtime
```