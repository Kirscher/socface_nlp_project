# About Socface

The [Socface](https://socface.site.ined.fr/) project aims to analyze French census documents and extract information on a large scale. The goal is to create a database using handwriting recognition to process handwritten nominal lists from the census.

## Task

One specific task is to predict the household head status, which involves determining whether an individual is the head of the household based on various characteristics extracted from the census data. This prediction task is crucial for grouping individuals into households and analyzing social changes over time.

## Installation

To use this project, please follow these installation steps:

1. Clone this repository to your local machine.
2. Install dependencies by running `pip install -r requirements.txt`.

## Data Preprocessing

Data preprocessing is performed using the `preprocessing.py` script. This script loads data from JSON and YAML files, prepares it for analysis, and converts it into a format suitable for model training.

## Modeling

We explored the use of RandomForest and pre-trained models like BERT for predicting the household head status. Details on the implementation and training of the models are available in the `modelling.ipynb` notebook.


## License

This project is licensed under the [MIT](LICENSE) license.
