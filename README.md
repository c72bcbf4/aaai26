# Graph Recognition via Subgraph Prediction (GraSP)
This is the official repository of the paper "Graph Recognition via Subgraph Prediction"
submitted to AAAI26.
We encourage reviewers to explore our models interactively as explained below.

### Examples
In case it is not possible to run the code as explained below, we provided
many examples of sample trajectories under [results/examples](results/examples)
for each different experiment.


### Installation

Our code uses [uv](https://docs.astral.sh/uv/getting-started/installation/)
as fast dependency manager for reproducible environments.
Run the commands below to install the dependencies and activate the environment.
```
uv venv --python 3.12.10
uv sync
source .venv/bin/activate
```

### Interactive model usage
We provide our models under [results/models](results/models).
[test.py](test.py) is a small CLI wrapper for all experiments which
can be used as follows.

`trees_sm` is configured for graphs with 6-9 nodes and `trees_lg`
is configured for graphs with 10-15 nodes. 
It is possible to define custom ranges with a custom suffix in 
[line 30](test.py#L30).
```
python test.py --experiment qm9 
python test.py --experiment trees_sm --NC 3 --EC 3  
python test.py --experiment trees_lg --NC 3 --EC 3 
```

Running the trees examples is not specifically bound to any size
for inference, therefore with the `--model` flag, it is possible to 
switch models to use those trained on the smaller or larger dataset.
This applies to any custom range.
```
python test.py --experiment trees_sm --NC 3 --EC 3 --model lg  
python test.py --experiment trees_lg --NC 3 --EC 3 --model sm
```
