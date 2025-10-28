# CS 7180: Special Topics in AI Final Project

From yours truly

## Rayleigh Quotient Analysis

### Run Experiments

0. This repository contains a submodule. To run this repository and access the submodule, run `git clone https://github.com/EdwardBerman/rayleigh_analysis.git` followed by `git submodule update --init --recursive`. The repo will require non Pythonic dependencies, you will need to run `sudo apt install cmake gfortran`.
1. Install poetry and run `poetry install`
2. Install wandb via and login via `wandb login [api key]`
3. Some data sets can be downloaded by running `python3 -m data_preprocessing.long_range_graph_benchmark`
4. [Optional] Poke around with the datasets. Run `python3 -m data_preprocessing.homophily` to get the homophily distribution of graphs for the node level classification tasks. 

To reproduce, install the poetry env and wandb
