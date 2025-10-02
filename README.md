# CS 7180: Special Topics in AI Final Project

From yours truly

## Rayleigh Quotient Analysis

### Run Experiments

1. Install poetry and run `poetry install`
2. Install wandb via and login via `wandb login [api key]`
3. Download the data sets by running `python3 -m data_preprocessing.longe_range_graph_benchmark`. If benchmarking unitary convolution, also run `python3 -m data_preprocessing.cache_matrix_exponentials`
4. [Optional] Poke around with the datasets. Run `python3 -m data_preprocessing.homophily` to get the homophily distribution of graphs for the node level classification tasks. 

To reproduce, install the poetry env and wandb
