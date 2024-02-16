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
Modified the arguments to use different splitting methods (including scaffold split and UMAP split); or specified the cell line you want to run the model; or if you want to do hyperparamters tunning.


## Table of Splitting Size for Each Cell Line
| Cell Line       	| Total Size 	| Scaffold Split Train Size 	| Scaffold Split Validation Size 	| Scaffold Split Test Size 	| UMAP Split Train Size 	| UMAP Split Validation Size 	| UMAP Split Test Size 	|
|-----------------	|------------	|---------------------------	|--------------------------------	|--------------------------	|-----------------------	|----------------------------	|----------------------	|
| MCF7            	| 24264      	| 17631                     	| 3388                           	| 3245                     	| 19805                 	| 1505                       	| 2954                 	|
| MDA-MB-231_ATCC 	| 23907      	| 17373                     	| 3353                           	| 3181                     	| 19526                 	| 1460                       	| 2921                 	|
| HS_578T         	| 22980      	| 16723                     	| 3211                           	| 3046                     	| 18807                 	| 1396                       	| 2777                 	|
| BT-549          	| 22162      	| 16111                     	| 3103                           	| 2948                     	| 18102                 	| 1371                       	| 2689                 	|
| T-47D           	| 22931      	| 16689                     	| 3208                           	| 3034                     	| 18667                 	| 1409                       	| 2855                 	|
| SF-268          	| 31647      	| 23141                     	| 4329                           	| 4177                     	| 25739                 	| 1722                       	| 4186                 	|
| SF-295          	| 31678      	| 23152                     	| 4336                           	| 4190                     	| 25739                 	| 1738                       	| 4201                 	|
| SF-539          	| 30125      	| 22004                     	| 4114                           	| 4007                     	| 24459                 	| 1662                       	| 4004                 	|
| SNB-19          	| 31436      	| 22937                     	| 4330                           	| 4169                     	| 25495                 	| 1745                       	| 4196                 	|
| SNB-75          	| 29572      	| 21624                     	| 4019                           	| 3929                     	| 23952                 	| 1647                       	| 3973                 	|
| U251            	| 31847      	| 23293                     	| 4349                           	| 4205                     	| 25827                 	| 1754                       	| 4266                 	|
| COLO_205        	| 31596      	| 23067                     	| 4338                           	| 4191                     	| 25649                 	| 1730                       	| 4217                 	|
| HCC-2998        	| 28814      	| 21017                     	| 3958                           	| 3839                     	| 23378                 	| 1633                       	| 3803                 	|
| HCT-116         	| 31712      	| 23187                     	| 4335                           	| 4190                     	| 25772                 	| 1744                       	| 4196                 	|
| HCT-15          	| 31719      	| 23139                     	| 4373                           	| 4207                     	| 25776                 	| 1744                       	| 4199                 	|
| HT29            	| 31639      	| 23160                     	| 4310                           	| 4169                     	| 25683                 	| 1722                       	| 4234                 	|
| KM12            	| 31672      	| 23161                     	| 4332                           	| 4179                     	| 25715                 	| 1757                       	| 4200                 	|
| SW-620          	| 31987      	| 23377                     	| 4388                           	| 4222                     	| 26011                 	| 1749                       	| 4227                 	|
| CCRF-CEM        	| 30228      	| 22111                     	| 4125                           	| 3992                     	| 24556                 	| 1675                       	| 3997                 	|
| HL-60(TB)       	| 28788      	| 21099                     	| 3881                           	| 3808                     	| 23374                 	| 1600                       	| 3814                 	|
| K-562           	| 31091      	| 22734                     	| 4235                           	| 4122                     	| 25264                 	| 1714                       	| 4113                 	|
| MOLT-4          	| 31388      	| 22974                     	| 4280                           	| 4134                     	| 25501                 	| 1731                       	| 4156                 	|
| RPMI-8226       	| 29912      	| 21840                     	| 4123                           	| 3949                     	| 24308                 	| 1650                       	| 3954                 	|
| SR              	| 26483      	| 19310                     	| 3676                           	| 3497                     	| 21497                 	| 1466                       	| 3520                 	|
| LOX_IMVI        	| 30089      	| 21964                     	| 4136                           	| 3989                     	| 24433                 	| 1660                       	| 3996                 	|
| MALME-3M        	| 29271      	| 21406                     	| 4038                           	| 3827                     	| 23754                 	| 1584                       	| 3933                 	|
| M14             	| 31416      	| 22923                     	| 4340                           	| 4153                     	| 25506                 	| 1693                       	| 4217                 	|
| SK-MEL-2        	| 29932      	| 21863                     	| 4108                           	| 3961                     	| 24353                 	| 1664                       	| 3915                 	|
| SK-MEL-28       	| 31373      	| 22948                     	| 4278                           	| 4147                     	| 25518                 	| 1696                       	| 4159                 	|
| SK-MEL-5        	| 31199      	| 22793                     	| 4295                           	| 4111                     	| 25359                 	| 1729                       	| 4111                 	|
| UACC-257        	| 31544      	| 23025                     	| 4318                           	| 4201                     	| 25623                 	| 1719                       	| 4202                 	|
| UACC-62         	| 31127      	| 22716                     	| 4286                           	| 4125                     	| 25231                 	| 1733                       	| 4163                 	|
| MDA-MB-435      	| 24347      	| 17682                     	| 3429                           	| 3236                     	| 19901                 	| 1483                       	| 2963                 	|
| MDA-N           	| 17948      	| 13065                     	| 2497                           	| 2386                     	| 14688                 	| 1089                       	| 2171                 	|
| A549_ATCC       	| 32080      	| 23434                     	| 4409                           	| 4237                     	| 26070                 	| 1749                       	| 4261                 	|
| EKVX            	| 30060      	| 21928                     	| 4168                           	| 3964                     	| 24382                 	| 1655                       	| 4023                 	|
| HOP-62          	| 31147      	| 22800                     	| 4242                           	| 4105                     	| 25323                 	| 1711                       	| 4113                 	|
| HOP-92          	| 28213      	| 20587                     	| 3894                           	| 3732                     	| 22902                 	| 1627                       	| 3684                 	|
| NCI-H226        	| 29739      	| 21728                     	| 4077                           	| 3934                     	| 24136                 	| 1665                       	| 3938                 	|
| NCI-H23         	| 31705      	| 23199                     	| 4342                           	| 4164                     	| 25748                 	| 1737                       	| 4220                 	|
| NCI-H322M       	| 30895      	| 22545                     	| 4249                           	| 4101                     	| 25091                 	| 1718                       	| 4086                 	|
| NCI-H460        	| 31050      	| 22637                     	| 4279                           	| 4134                     	| 25195                 	| 1718                       	| 4137                 	|
| NCI-H522        	| 29232      	| 21435                     	| 3963                           	| 3834                     	| 23666                 	| 1651                       	| 3915                 	|
| IGROV1          	| 31413      	| 22941                     	| 4315                           	| 4157                     	| 25551                 	| 1748                       	| 4114                 	|
| OVCAR-3         	| 31105      	| 22763                     	| 4261                           	| 4081                     	| 25265                 	| 1700                       	| 4140                 	|
| OVCAR-4         	| 30423      	| 22210                     	| 4185                           	| 4028                     	| 24690                 	| 1687                       	| 4046                 	|
| OVCAR-5         	| 31249      	| 22818                     	| 4309                           	| 4122                     	| 25365                 	| 1733                       	| 4151                 	|
| OVCAR-8         	| 32050      	| 23390                     	| 4405                           	| 4255                     	| 26032                 	| 1751                       	| 4267                 	|
| SK-OV-3         	| 30204      	| 22066                     	| 4142                           	| 3996                     	| 24501                 	| 1677                       	| 4026                 	|
| NCI_ADR-RES     	| 24312      	| 17667                     	| 3392                           	| 3253                     	| 19855                 	| 1476                       	| 2981                 	|
| PC-3            	| 24187      	| 17574                     	| 3404                           	| 3209                     	| 19792                 	| 1467                       	| 2928                 	|
| DU-145          	| 24117      	| 17562                     	| 3373                           	| 3182                     	| 19683                 	| 1461                       	| 2973                 	|
| 786-0           	| 31483      	| 22992                     	| 4341                           	| 4150                     	| 25581                 	| 1702                       	| 4200                 	|
| A498            	| 27887      	| 20312                     	| 3837                           	| 3738                     	| 22566                 	| 1616                       	| 3705                 	|
| ACHN            	| 31650      	| 23116                     	| 4378                           	| 4156                     	| 25728                 	| 1693                       	| 4229                 	|
| CAKI-1          	| 30144      	| 22002                     	| 4159                           	| 3983                     	| 24479                 	| 1696                       	| 3969                 	|
| RXF_393         	| 28620      	| 20869                     	| 3913                           	| 3838                     	| 23191                 	| 1615                       	| 3814                 	|
| SN12C           	| 31667      	| 23135                     	| 4336                           	| 4196                     	| 25704                 	| 1725                       	| 4238                 	|
| TK-10           	| 30974      	| 22670                     	| 4235                           	| 4069                     	| 25201                 	| 1636                       	| 4137                 	|
| UO-31           	| 31508      	| 23040                     	| 4324                           	| 4144                     	| 25576                 	| 1719                       	| 4213                 	|

## Authors

- [@Rong830](https://www.github.com/Rong830)
## Contributing

Contributions are always welcome! If you'd like to contribute to this project, please follow the standard procedures:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make changes and commit
4. Push to your fork and submit a pull request

Please adhere to this project's `code of conduct`.
