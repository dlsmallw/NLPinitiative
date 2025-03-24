# NLPInitiative Documentation

***

## Project Details

### Description
Codebase for training, evaluating and deploying NLP models used to detect discriminatory language targeting marginallized individuals or communities and the type(s) of discrimination detected.

### Organization

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
<span>
    Based on the CookieCutter Data Science project structure template 
    <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
        <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
    </a>
</span>

### Project Model and Dataset Repositories

#### Model Repositories

| Fine-tuned Model   | Base Model | Repository Link |
| ------------------ | ---------- | --------------- |
| Binary Classification Model | BERT | [NLPinitiative-Binary-Classification](https://huggingface.co/dlsmallw/NLPinitiative-Binary-Classification) |
| Multilabel Regression Model | BERT | [NLPinitiative-Multilabel-Regression](https://huggingface.co/dlsmallw/NLPinitiative-Multilabel-Regression) |

#### Dataset Repository

|                    | Repository Link |
| ------------------ | --------------- |
| Dataset Repository | [NLPinitiative-Dataset](https://huggingface.co/datasets/dlsmallw/NLPinitiative-Dataset) |

***

## Project Setup

The Makefile contains the central entry points for common tasks related to this project.

***

## Datasets Used

### [Ethos](https://doi.org/10.1007/s40747-021-00608-2) - multi-lab**E**l ha**T**e speec**H** detecti**O**n data**S**et
A collection consisting of binary and multilabel data containing hate speech from social media.

#### Links
 - [GitHub Repository](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)
    - [Binary Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv)
    - [Multilabel Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv)

#### BibTeX Reference
```bibtex
@article{mollas_ethos_2022,
    title = {{ETHOS}: a multi-label hate speech detection dataset},
    issn = {2198-6053},
    url = {https://doi.org/10.1007/s40747-021-00608-2},
    doi = {10.1007/s40747-021-00608-2},
    journal = {Complex \& Intelligent Systems},
    author = {Mollas, Ioannis and Chrysopoulou, Zoe and Karlos, Stamatis and Tsoumakas, Grigorios},
    month = jan,
    year = {2022},
}
```

### [Multitarget-CONAN](https://doi.org/10.1007/s40747-021-00608-2) 
Multi-Target CONAN is a dataset of hate speech/counter-narrative pairs for English comprising several hate targets, collected using a Human-in-the-Loop approach.

#### Links
 - [GitHub Repository](https://github.com/marcoguerini/CONAN)
    - [Multitarget-CONAN Dataset](https://github.com/marcoguerini/CONAN/blob/master/Multitarget-CONAN/Multitarget-CONAN.csv)


#### BibTeX Reference
```bibtex
@inproceedings{fanton-2021-human,
  title="{Human-in-the-Loop for Data Collection: a Multi-Target Counter Narrative Dataset to Fight Online Hate Speech}",
  author="{Fanton, Margherita and Bonaldi, Helena and Tekiroğlu, Serra Sinem and Guerini, Marco}",
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
  month = aug,
  year = "2021",
  publisher = "Association for Computational Linguistics",
}
```

***



