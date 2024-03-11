# Scaffold Split Overestimate Virtual Screening

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PyPI version](https://badge.fury.io/py/pypi.svg)](https://badge.fury.io/py/pypi)

This repo contains all the codes used in the paper: Scaffold splits overestimate virtual screening performance, using traditional machine learning models and deep learning models to illustrate the limitation of using Scaffold Splits in Drug Discovery. This repo can help researchers to reproduce what has been done in the article.
![Figure](https://github.com/ScaffoldSplitsOverestimateVS/ScaffoldSplitsOverestimateVS/assets/162518242/428a2a6c-4e55-435d-8a8c-c6b01e46a377)


## Installation
These instructions will guide you through setting up the Conda environment for the project.

### Prerequisites

Make sure you have Conda installed on your system. If not, you can download and install it from [here](https://www.anaconda.com/download).

### Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/ScaffoldSplitsOverestimateVS/ScaffoldSplitsOverestimateVS.git
cd ScaffoldSplitsOverestimateVS
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
Modified the arguments to use different splitting methods (including scaffold split and UMAP split); or specified the cell line you want to run the model; or if you want to do hyperparamters tunning.

## Tables
### The RDKit functions are used to extract descriptors from molecules and their descriptions.
| Package                                 	| Function                                                                    	|
|-----------------------------------------	|-----------------------------------------------------------------------------	|
| AllChem.GetMorganFingerprintAsBitVect   	| Generate the Morgan Fingerprints for the molecules. [^1]                   	|
| rdMolDescriptors.CalcTPSA               	| Calculate the area of the total polar surface.                              	|
| rdMolDescriptors.CalcExactMolWt         	| Calculate the molecular weight.                                             	|
| rdMolDescriptors.CalcCrippenDescriptors 	| Calculate the Crippen-Wildman partition coefficient (logP) parameters [^2]. 	|
| rdMolDescriptors.CalcNumAliphaticRings  	| The number of aliphatic rings.                                              	|

[^1]: Rogers, D., Hahn, M.: Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50, 742–754 (2010). [https://doi.org/10.1021/ci100050t](https://doi.org/10.1021/ci100050t).

[^2]: Wildman, S.A., Crippen, G.M.: Prediction of physicochemical parameters by atomic contributions. Journal of Chemical Information and Computer Sciences 1999, 39 (5), 868-873. [https://doi.org/10.1021/ci990307l](https://doi.org/10.1021/ci990307l).


### Table of Splitting Size for Each Cell Line

| Cell Line       	| Total Size 	| Scaffold Split Train Size 	| Scaffold Split Test Size 	| UMAP Split Train Size 	| UMAP Split Test Size 	|
|-----------------	|------------	|---------------------------	|--------------------------	|-----------------------	|----------------------	|
| MCF7            	| 24264      	| 21019                     	| 3245                     	| 21310                 	| 2954                 	|
| MDA-MB-231_ATCC 	| 23907      	| 20726                     	| 3181                     	| 20986                 	| 2921                 	|
| HS_578T         	| 22980      	| 19934                     	| 3046                     	| 20203                 	| 2777                 	|
| BT-549          	| 22162      	| 19214                     	| 2948                     	| 19473                 	| 2689                 	|
| T-47D           	| 22931      	| 19897                     	| 3034                     	| 20076                 	| 2855                 	|
| SF-268          	| 31647      	| 27470                     	| 4177                     	| 27461                 	| 4186                 	|
| SF-295          	| 31678      	| 27488                     	| 4190                     	| 27477                 	| 4201                 	|
| SF-539          	| 30125      	| 26118                     	| 4007                     	| 26121                 	| 4004                 	|
| SNB-19          	| 31436      	| 27267                     	| 4169                     	| 27240                 	| 4196                 	|
| SNB-75          	| 29572      	| 25643                     	| 3929                     	| 25599                 	| 3973                 	|
| U251            	| 31847      	| 27642                     	| 4205                     	| 27581                 	| 4266                 	|
| COLO_205        	| 31596      	| 27405                     	| 4191                     	| 27379                 	| 4217                 	|
| HCC-2998        	| 28814      	| 24975                     	| 3839                     	| 25011                 	| 3803                 	|
| HCT-116         	| 31712      	| 27522                     	| 4190                     	| 27516                 	| 4196                 	|
| HCT-15          	| 31719      	| 27512                     	| 4207                     	| 27520                 	| 4199                 	|
| HT29            	| 31639      	| 27470                     	| 4169                     	| 27405                 	| 4234                 	|
| KM12            	| 31672      	| 27493                     	| 4179                     	| 27472                 	| 4200                 	|
| SW-620          	| 31987      	| 27765                     	| 4222                     	| 27760                 	| 4227                 	|
| CCRF-CEM        	| 30228      	| 26236                     	| 3992                     	| 26231                 	| 3997                 	|
| HL-60(TB)       	| 28788      	| 24980                     	| 3808                     	| 24974                 	| 3814                 	|
| K-562           	| 31091      	| 26969                     	| 4122                     	| 26978                 	| 4113                 	|
| MOLT-4          	| 31388      	| 27254                     	| 4134                     	| 27232                 	| 4156                 	|
| RPMI-8226       	| 29912      	| 25963                     	| 3949                     	| 25958                 	| 3954                 	|
| SR              	| 26483      	| 22986                     	| 3497                     	| 22963                 	| 3520                 	|
| LOX_IMVI        	| 30089      	| 26100                     	| 3989                     	| 26093                 	| 3996                 	|
| MALME-3M        	| 29271      	| 25444                     	| 3827                     	| 25338                 	| 3933                 	|
| M14             	| 31416      	| 27263                     	| 4153                     	| 27199                 	| 4217                 	|
| SK-MEL-2        	| 29932      	| 25971                     	| 3961                     	| 26017                 	| 3915                 	|
| SK-MEL-28       	| 31373      	| 27226                     	| 4147                     	| 27214                 	| 4159                 	|
| SK-MEL-5        	| 31199      	| 27088                     	| 4111                     	| 27088                 	| 4111                 	|
| UACC-257        	| 31544      	| 27343                     	| 4201                     	| 27342                 	| 4202                 	|
| UACC-62         	| 31127      	| 27002                     	| 4125                     	| 26964                 	| 4163                 	|
| MDA-MB-435      	| 24347      	| 21111                     	| 3236                     	| 21384                 	| 2963                 	|
| MDA-N           	| 17948      	| 15562                     	| 2386                     	| 15777                 	| 2171                 	|
| A549_ATCC       	| 32080      	| 27843                     	| 4237                     	| 27819                 	| 4261                 	|
| EKVX            	| 30060      	| 26096                     	| 3964                     	| 26037                 	| 4023                 	|
| HOP-62          	| 31147      	| 27042                     	| 4105                     	| 27034                 	| 4113                 	|
| HOP-92          	| 28213      	| 24481                     	| 3732                     	| 24529                 	| 3684                 	|
| NCI-H226        	| 29739      	| 25805                     	| 3934                     	| 25801                 	| 3938                 	|
| NCI-H23         	| 31705      	| 27541                     	| 4164                     	| 27485                 	| 4220                 	|
| NCI-H322M       	| 30895      	| 26794                     	| 4101                     	| 26809                 	| 4086                 	|
| NCI-H460        	| 31050      	| 26916                     	| 4134                     	| 26913                 	| 4137                 	|
| NCI-H522        	| 29232      	| 25398                     	| 3834                     	| 25317                 	| 3915                 	|
| IGROV1          	| 31413      	| 27256                     	| 4157                     	| 27299                 	| 4114                 	|
| OVCAR-3         	| 31105      	| 27024                     	| 4081                     	| 26965                 	| 4140                 	|
| OVCAR-4         	| 30423      	| 26395                     	| 4028                     	| 26377                 	| 4046                 	|
| OVCAR-5         	| 31249      	| 27127                     	| 4122                     	| 27098                 	| 4151                 	|
| OVCAR-8         	| 32050      	| 27795                     	| 4255                     	| 27783                 	| 4267                 	|
| SK-OV-3         	| 30204      	| 26208                     	| 3996                     	| 26178                 	| 4026                 	|
| NCI_ADR-RES     	| 24312      	| 21059                     	| 3253                     	| 21331                 	| 2981                 	|
| PC-3            	| 24187      	| 20978                     	| 3209                     	| 21259                 	| 2928                 	|
| DU-145          	| 24117      	| 20935                     	| 3182                     	| 21144                 	| 2973                 	|
| 786-0           	| 31483      	| 27333                     	| 4150                     	| 27283                 	| 4200                 	|
| A498            	| 27887      	| 24149                     	| 3738                     	| 24182                 	| 3705                 	|
| ACHN            	| 31650      	| 27494                     	| 4156                     	| 27421                 	| 4229                 	|
| CAKI-1          	| 30144      	| 26161                     	| 3983                     	| 26175                 	| 3969                 	|
| RXF_393         	| 28620      	| 24782                     	| 3838                     	| 24806                 	| 3814                 	|
| SN12C           	| 31667      	| 27471                     	| 4196                     	| 27429                 	| 4238                 	|
| TK-10           	| 30974      	| 26905                     	| 4069                     	| 26837                 	| 4137                 	|
| UO-31           	| 31508      	| 27364                     	| 4144                     	| 27295                 	| 4213                 	|

## Contributing

Contributions are always welcome! If you'd like to contribute to this project, please follow the standard procedures:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make changes and commit
4. Push to your fork and submit a pull request

Please adhere to this project's `code of conduct`.
