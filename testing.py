import uuid

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
            min_nodes=min_nodes, max_nodes=max_nodes, n_node_colors=n, n_edge_colors=m
        )

    raise ValueError(f"unknown dataset '{name}'")


def get_model(name):
    return torch.load(f"results/models/{name}.pt", weights_only=False)


def test():
    configure_logging("logs/test.log")
    name = "qm9"
    generator = get_dataset(name)
    data = GraphDataset(
        generator,
        image_size=256,
        fake_size=100,
        min_size=100,
        batch_size=1,
        cache_size=100,
        device="cpu",
        max_node_degree=4 if name == "qm9" else None,
    )

    evaluator = ModelEvaluator(f"tmp/{uuid.uuid4()}", data)
    model = get_model(name).to("cuda:0")

    total = 100 if name != "qm9" else len(generator.data)

    parallel = 100
    batch_size = 256
    n_iters = total // parallel

    done = 0
    correct = 0

    for i in range(n_iters):
        start = i * parallel
        end = start + parallel

        indexes = list(range(start, end))
        x, _, _ = evaluator.evaluate(
            model, rollouts=0, batch_size=batch_size, indexes=indexes
        )

        correct += sum(int(x_.correct) for x_ in x)
        done += parallel

        logger.info(f"{correct}/{done} correct")


test()
