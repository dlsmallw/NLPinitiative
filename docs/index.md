# NLPInitiative Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

Codebase for training, evaluating and deploying NLP models used to detect discriminatory language targeting marginallized individuals or communities and the type(s) of discrimination detected.

This project was developed in coordination with the **<a href="https://www.j-initiative.org/" style="text-decoration:none">The J-Healthcare Initiative</a>** for the purposes of detecting discriminatory language in textual media from public officials/organizations and news agencies targetting marginalized communities communities.

## Project Links
- **<a href="https://huggingface.co/dlsmallw/NLPinitiative-Binary-Classification" style="text-decoration:none">🤗 NLPinitiative-Binary-Classification</a>** - The fine-tuned binary classfication models HF Model Repository.
- **<a href="https://huggingface.co/dlsmallw/NLPinitiative-Multilabel-Regression" style="text-decoration:none">🤗 NLPinitiative-Multilabel-Regression</a>** - The fine-tuned multilabel regression models HF Model Repository.
- **<a href="https://huggingface.co/datasets/dlsmallw/NLPinitiative-Dataset" style="text-decoration:none">🤗 NLPinitiative-Dataset</a>** - The HF hosted Dataset Repository.
- **<a href="https://huggingface.co/spaces/dlsmallw/NLPinitiative-Streamlit-App" style="text-decoration:none">🤗 HF Spaces Streamlit Web Application</a>** - The HF Space hosting the served models.

***

## Table of Contents

- [Setup](#setup)
- [Structure](#structure)
- [Datasets](#datasets)
- [License](#license)

***

## Setup

For the purposes of easily building, setting up and managing the project codebase, a bash script, `setup.sh`, has been created which contains a suite of custom commands for running various development-related processes (defined below). Use of this script requires that a bash shell is installed and set up (git bash for Windows users). For configuring and setting up your system to enable the use of Linux subsystems (Windows users), please see [this](https://www.google.com/search?client=firefox-b-d&q=Microsoft+windows+bash) for details on how to install and enabling WSL.

This script can be activated by entering `source ./setup.sh` within the bash shell while within the project source directory.

#### Commands
 - `help`: Displays all of the commands that can be used.
 - `build`: This will setup a virtual environment within the project source directory and install all necessary dependencies for development.
 - `clean`: This will deactivate the virutal environment, and remove the .venv directory (uninstalling all dependencies).
 - `docs build`: Parses the docstrings in the project and generates the project documentation using mkdocs.
 - `docs serve`: Serves the mkdocs documentation to a local dev server that can be opened in a browser.
 - `docs deploy`: Deploys the mkdocs documentation to the linked GitHub repositories 'GitHub Pages'.
 - `lint`: Lints (analyzes and identifies style/format issues to correct) the project files.
 - `format`: Corrects the issues identified from running the lint command.
 - `run tests`: Runs the test suite..
 - `set bin_repo <HF Model Repository ID>`: Sets the binary model repository ID to the specified string.
    - This is the source for downloading the model tensor file.
 - `set ml_repo <HF Model Repository ID>`: Sets the multilabel regression model repository ID to the specified string.
    - This is the source for downloading the model tensor file.
 - `set ds_repo <HF Dataset Repository ID>`: Sets the dataset repository ID to the specified string.
    - This is the source for downloading the datasets.
 - `set streamlit_repo <HF Spaces Streamlit App Repository ID>`: Sets the Streamlit App repo ID in the pyproject.toml file.
 - `set space_url <HF Spaces base URL>`: Sets the base URL for HF Spaces in the pyproject.toml file.
 - `set model_url <HF Model Repo base URL>`: Sets the base URL for HF Model Repos in the pyproject.toml file.
 - `set dataset_url <HF Dataset Repo base URL>`:  Sets the base URL for HF Dataset Repos in the pyproject.toml file.
 - `set hf_token <HF Token>`: Sets the HF personal token in the pyproject.toml file.

***

## Structure

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

***

## Datasets

### ETHOS
A collection consisting of binary and multilabel data containing hate speech from social media.

 - [Academic Article](https://doi.org/10.1007/s40747-021-00608-2)
 - [Link to Source GitHub Repository](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)
    - [Direct Link to the Binary Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv)
    - [Direct Link to the Multilabel Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv)

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

### Multitarget-CONAN
Multi-Target CONAN is a dataset of hate speech/counter-narrative pairs for English comprising several hate targets, collected using a Human-in-the-Loop approach.

 - [Academic Article](https://doi.org/10.1007/s40747-021-00608-2)
 - [Link to Source GitHub Repository](https://github.com/marcoguerini/CONAN)
    - [Direct Link to the Dataset](https://github.com/marcoguerini/CONAN/blob/master/Multitarget-CONAN/Multitarget-CONAN.csv)

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

## License

The MIT License (MIT)
Copyright (c) 2025, ASU Fall-2024/Spring-2025 Capstone Group 8 and The J Healthcare Initiative

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

***