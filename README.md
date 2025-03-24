# NLPInitiative

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Codebase for training, evaluating and deploying NLP models used to detect discriminatory language targeting marginallized individuals or communities and the type(s) of discrimination detected.

## Project Organization

```
├── data
│   ├── interim                 <- Intermediate datasets that have been normalized
│   ├── normalization_schema    <- The schema used for normalizing 3rd party datasets
│   ├── processed               <- The final merged dataset consisting of all normalized datasets
│   └── raw                     <- The original, raw datasets prior to normalization
│
├── docs            <- A directory containing documentation used for generating and serving 
│                      project documentation
│
├── models          <- Trained and serialized models, model predictions, or model summaries
│   │
│   ├── binary_classification        <- Trained and serialized binary classification 
│   │                                   models/model predictions/model summaries
│   └── multilabel_regression        <- Trained and serialized multilabel regression 
│                                       models/model predictions/model summaries
│
├── nlpinitiative   <- Source code for use in this project
│   │
│   ├── __init__.py             <- Makes nlpinitiative a Python module
│   ├── config.py               <- Store useful variables and configuration
│   └── modeling                <- Source code for model training and inference
│       │                
│       ├── __init__.py         <- Makes modeling a Python module
│       ├── predict.py          <- Code to run model inference with trained models          
│       └── train.py            <- Code to train models
│
├── notebooks           <- Directory containing Jupyter notebooks detailing research, testing and 
│                          example usage of project modules 
│
├── references          <- Directory containing Data dictionaries, manuals, and all other 
│                          explanatory materials
│
├── LICENSE             <- Open-source license if one is chosen
│
├── Makefile            <- Makefile with convenience commands 
│
├── mkdocs.yml          <- mkdocs project configuration
│
├── Pipfile             <- The project dependency file for reproducing the analysis environment, 
│                          e.g., generated with `pipenv install`
│
├── Pipfile.lock        <- Locked file containing hashes for dependencies
│
├── pyproject.toml      <- Project configuration file with package metadata for nlpinitiative and 
│                          configuration for tools like black
│
├── README.md           <- The top-level README for developers using this project
│
├── setup.cfg           <- Configuration file for flake8
│
└── setup.sh            <- Bash script containing convenience commands for managing the project
```

--------

