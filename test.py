import uuid
from pathlib import Path

import torch
from loguru import logger

from src.data import QM9Generator, TreeGenerator, GraphDataset
from src.eval import ModelEvaluator
from src.logging import configure_logging


def get_dataset(name):
    if name == "qm9":
        return QM9Generator(train=False)

    if name.startswith("trees"):
        _, size, n, m = name.split("_")

        sizes = {"sm": (6, 9), "lg": (10, 15)}

        min_nodes, max_nodes = sizes[size]
        return TreeGenerator(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            n_node_colors=int(n),
            n_edge_colors=int(m),
        )

    raise ValueError(f"unknown dataset '{name}'")


def get_model(name):
    return torch.load(f"results/models/{name}.pt", weights_only=False)


def test():
    # set this to True if you would like to see the outputs
    plot_trajectories = True

    # for available experiments, see models under results/models
    experiment = "trees_lg_5_5"

    tmp_dir = Path("tmp") / experiment
    tmp_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(tmp_dir / f"{experiment}.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = get_dataset(experiment)
    data = GraphDataset(
        generator,
        image_size=256,
        fake_size=100,
        min_size=100,
        batch_size=1,
        cache_size=100,
        device=device,
        max_node_degree=4 if experiment == "qm9" else None,
    )

    evaluator = ModelEvaluator(tmp_dir, data)
    model = get_model(experiment).to(device)

    # how many trajectories to run in total
    total = 100

    # number of parallel trajectories
    parallel = 4

    # number of parallel evaluations of transitions
    batch_size = 32

    n_iters = total // parallel

    done = 0
    correct = 0

    logger.info("running detection")

    for i in range(n_iters):
        start = i * parallel
        end = start + parallel

        indexes = list(range(start, end))
        trajectories, _, _ = evaluator.evaluate(
            model, rollouts=0, batch_size=batch_size, indexes=indexes
        )

        if plot_trajectories:
            for t in trajectories:
                evaluator.plot(str(uuid.uuid4()), t)

        correct += sum(int(t.correct) for t in trajectories)
        done += parallel

        logger.info(f"{correct}/{done} correct")


if __name__ == "__main__":
    test()
