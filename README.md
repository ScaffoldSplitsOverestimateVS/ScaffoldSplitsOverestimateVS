# Scaffold Split Overestimate Virutal Screening

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

This repo contains the all the codes that used in the paper: xxx, using traditional machince learning models and deep learning models to illustrate the limitaion of using Scaffold Split in Drug Discovery. This repo can help researcher to reproduce what have been done in the article.

## Installation
These instructions will guide you through setting up the Conda environment for the project.

### Prerequisites

Make sure you have Conda installed on your system. If not, you can download and install it from [here](https://www.anaconda.com/download).

### Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/Rong830/ScaffoldOverestimateVS.git
cd ScaffoldOverestimateVS
```

### Set Up Conda Environment
Create a Conda environment using the provided `requirements.txt` file. Run the following commands in the project root:

```bash
conda create --name my_environment
conda activate my_environment
conda install --file requirements.txt
```

## Usage/Examples

Run the Linear Regression and Random Forest models on all 60 cell lines:
```bash
bash run_sklearn.sh
```

Run the GEM models:
```bash
bash run_gem.sh
```

You modified the arguments to use different splitting method (including scaffold split and UMAP split); or specified the cell line you want to run the model; or if you want to do hyperparamters tunning.
## Authors

- [@Rong830](https://www.github.com/Rong830)
## Contributing

Contributions are always welcome! If you'd like to contribute to this project, please follow the standard procedures:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make changes and commit
4. Push to your fork and submit a pull request

Please adhere to this project's `code of conduct`.
