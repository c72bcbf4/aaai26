import argparse
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from src.data import QM9Generator, TreeGenerator, GraphDataset
from src.eval import ModelEvaluator
from src.logging import configure_logging


@dataclass
class CLIArgs:
    experiment: str
    NC: int
    EC: int
    save: bool
    model: str


def get_dataset(name):
    if name == "qm9":
        return QM9Generator(train=False)

    if name.startswith("trees"):
        _, size, n, m = name.split("_")

        sizes = {
            "sm": (6, 9),
            "lg": (10, 15),
        }

        min_nodes, max_nodes = sizes[size]
        return TreeGenerator(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            n_node_colors=int(n),
            n_edge_colors=int(m),
        )

    raise ValueError(f"unknown dataset '{name}'")


def get_model(args):
    if args.experiment == "qm9":
        name = "qm9"
    else:
        task, size = args.experiment.split("_")
        nc = args.NC
        ec = args.EC
        size = args.model or size
        name = f"{task}_{size}_{nc}_{ec}"
    return name


def get_experiment(args):
    if args.experiment == "qm9":
        return "qm9"

    exp = args.experiment
    nc = args.NC
    ec = args.EC

    experiment = f"{exp}_{nc}_{ec}"

    return experiment


def validate_args(args):
    if args.experiment == "qm9":
        if args.NC is not None:
            logger.warning(f"ignoring node colors for qm9")
        if args.EC is not None:
            logger.warning(f"ignoring edge colors for qm9")
        if args.model is not None:
            logger.warning(f"ignoring model for qm9")

        return

    assert args.NC in [1, 3, 5], "models support only 1, 3 and 5 node colors"
    assert args.EC in [1, 3, 5], "models support only 1, 3 and 5 edge colors"


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(
        description="Deep learning tool for the recognition of graphs in images."
    )
    parser.add_argument("--experiment", type=str, default="trees_sm", required=True)
    parser.add_argument("--model", type=str, default=None, required=False)
    parser.add_argument(
        "--NC", type=int, default=None, help="Number of node colors.", required=False
    )
    parser.add_argument(
        "--EC", type=int, default=None, help="Number of edge colors.", required=False
    )
    parser.add_argument(
        "--save", action="store_true", dest="save", help="Save outputs."
    )
    args = parser.parse_args()
    return CLIArgs(
        experiment=args.experiment,
        NC=args.NC,
        EC=args.EC,
        save=args.save,
        model=args.model,
    )


def test():
    args = parse_args()

    # for available experiments, see models under results/models
    experiment = get_experiment(args)

    tmp_dir = Path("tmp") / experiment
    tmp_dir.mkdir(parents=True, exist_ok=True)

    configure_logging()
    validate_args(args)

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

    model_name = get_model(args)

    model = torch.load(
        f"results/models/{model_name}.pt", weights_only=False, map_location=device
    )

    logger.info(f"running experiment '{experiment}' with model '{model_name}'")

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

        if args.save:
            for t in trajectories:
                evaluator.plot(str(uuid.uuid4()), t)

        correct += sum(int(t.correct) for t in trajectories)
        done += parallel

        logger.info(f"{correct}/{done} correct")


if __name__ == "__main__":
    test()
