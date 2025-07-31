import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import imageio
import matplotlib
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchmetrics import MeanMetric

from src.data import Graph, GraphDataset, GraphTester
from src.nn import GraphRecognitionModel
from src.util import batch_stream, batch_fast


@dataclass
class TrajectoryContainer:
    true_graph_hash: str
    true_graph_pt: Graph
    true_graph_nx: nx.Graph
    pred_graph_pt: Graph
    pred_graph_nx: nx.Graph
    steps: List
    correct: bool | None
    match: bool | None


class ModelEvaluator:
    def __init__(
        self,
        directory: str | Path,
        dataset: GraphDataset,
    ):
        self.dataset = dataset
        self.directory = directory
        self.max_entries = 75

    def get_root(self, true_graph_pt):
        pred_nodes = true_graph_pt.nodes[:1]
        pred_edges = torch.empty((2, 0), dtype=torch.int64)
        pred_edges_f = true_graph_pt.edges_f[:0]
        pred_graph = Graph(
            true_graph_pt.image,
            pred_nodes,
            pred_edges,
            pred_edges_f,
            torch.tensor(False),
        )

        trajectory = [(pred_graph, None, None, -1, True, None, [])]

        true_graph_nx = self.dataset.to_nx(true_graph_pt)
        true_graph_hash = self.dataset.hash(true_graph_nx)

        pred_graph_nx = self.dataset.to_nx(pred_graph)

        return TrajectoryContainer(
            true_graph_hash=true_graph_hash,
            true_graph_pt=true_graph_pt,
            true_graph_nx=true_graph_nx,
            pred_graph_pt=pred_graph,
            pred_graph_nx=pred_graph_nx,
            steps=trajectory,
            correct=None,
            match=None,
        )

    def is_correct(self, G1: nx.Graph, G2: nx.Graph):
        return self.dataset.hash(G1) == self.dataset.hash(G2)

    @torch.inference_mode()
    def evaluate(
        self,
        model: GraphRecognitionModel,
        rollouts: int,
        batch_size: int,
        indexes: List[int] = None,
    ):
        # must be done only for nodes since edges don't have any other features
        n = len(self.dataset.generator.node_colors)

        if indexes is None:
            indexes = [None] * rollouts

        states = {
            i: self.get_root(self.dataset.get_item(encode_image=False, idx=idx))
            for i, idx in enumerate(indexes)
        }
        dones = set()

        x_ent = MeanMetric().to(model.device)
        top_k_preds = MeanMetric().to(model.device)

        tester = GraphTester()

        while len(dones) != len(states):
            current = {k: v for k, v in states.items() if k not in dones}
            successors_nx = {
                k: list(self.dataset.successors_nx(v.pred_graph_nx).values())
                for k, v in current.items()
            }

            successors_pt = {
                k: [self.dataset.from_nx(x) for x in v]
                for k, v in successors_nx.items()
            }

            successors_len = [len(v) for v in successors_nx.values()]

            labels = {}
            for k, v in current.items():
                labels[k] = torch.tensor(
                    [
                        (
                            self.dataset.hash(s_nx) == v.true_graph_hash
                            if s_nx.graph["terminal"]
                            else tester.approximate_subgraph_match(
                                v.true_graph_nx, s_nx
                            )
                        )
                        for s_nx in successors_nx[k]
                    ],
                    dtype=torch.float32,
                    device=model.device,
                )

            labels_pt = torch.cat(list(labels.values()))

            preds = []
            for batch in batch_stream(
                itertools.chain.from_iterable(successors_pt.values()), batch_size
            ):
                successors_batch = batch_fast(batch, model.device)
                preds.append(model(successors_batch))

            preds = torch.cat(preds, dim=-1)

            x_ent.update(
                torch.nn.functional.binary_cross_entropy_with_logits(preds, labels_pt)
            )

            preds = {
                k: v for k, v in zip(current.keys(), torch.split(preds, successors_len))
            }

            for k, v in current.items():
                idx = preds[k].argmax()
                pred_graph_pt = successors_pt[k][idx]
                pred_graph_nx = successors_nx[k][idx]

                true_node_dist = v.true_graph_pt.nodes[:, :n].sum(0)
                pred_node_dist = v.pred_graph_pt.nodes[:, :n].sum(0)

                true_edge_dist = v.true_graph_pt.edges_f.sum(0)
                pred_edge_dist = v.pred_graph_pt.edges_f.sum(0)

                node_valid = ((true_node_dist - pred_node_dist) >= 0).all()
                edge_valid = ((true_edge_dist - pred_edge_dist) >= 0).all()
                valid = node_valid and edge_valid

                curr_order = preds[k].argsort(descending=True)
                curr_labels = labels[k][curr_order]
                curr_valid = curr_labels.int().sum()
                curr_frac = curr_labels[:curr_valid].float().mean()
                top_k_preds.update(curr_frac)

                connections = [g.graph["connection"] for g in successors_nx[k]]

                top_k = [(i, successors_nx[k][i]) for i in curr_order[:5].cpu().numpy()]

                v.steps.append(
                    (pred_graph_pt, labels[k], preds[k], idx, valid, connections, top_k)
                )

                current[k].pred_graph_pt = pred_graph_pt
                current[k].pred_graph_nx = pred_graph_nx

                valid = bool(labels[k][idx].cpu())

                if not valid or pred_graph_pt.terminal:
                    dones.add(k)

        for k, v in states.items():
            pred_graph_pt = v.steps[-1][0]

            true_nx = self.dataset.to_nx(v.true_graph_pt)
            pred_nx = self.dataset.to_nx(pred_graph_pt)

            correct = self.is_correct(true_nx, pred_nx) and pred_graph_pt.terminal

            true_node_dist = v.true_graph_pt.nodes[:, :n].sum(0)
            pred_node_dist = v.pred_graph_pt.nodes[:, :n].sum(0)

            true_edge_dist = v.true_graph_pt.edges_f.sum(0)
            pred_edge_dist = v.pred_graph_pt.edges_f.sum(0)

            node_match = (true_node_dist == pred_node_dist).all().item()
            edge_match = (true_edge_dist == pred_edge_dist).all().item()

            match = node_match and edge_match

            states[k].correct = correct
            states[k].match = match
        return list(states.values()), x_ent.compute(), top_k_preds.compute()

    def round(self, values):
        return list(map(lambda x: round(x, 3), values))

    def norm(self, array, cvt):
        if cvt:
            array = array.tolist()

        array = self.round(array)
        return mcolors.Normalize(vmin=min(array), vmax=max(array)), array

    def plot_table(
        self,
        axes: matplotlib.axes.Axes,  # noqa
        preds: torch.Tensor,
        labels: torch.Tensor,
        match: bool,
        correct: bool,
        valid: bool,
        idx: int,
        table_size: int,
        connections: List[Dict],
    ):
        all_preds = preds.cpu()
        all_probs = all_preds.sigmoid().cpu()
        all_labels = labels.cpu()
        n_correct = int(all_labels.sum().item())

        conn_types, connections = zip(
            *[(c["type"], c["connection"]) for c in connections]
        )
        us, es, vs = zip(*connections)
        color = "#4287f5"

        n = len(all_preds)
        all_ids = list(range(n))
        all_indices = list(reversed(np.argsort(all_preds)))[: self.max_entries]

        cmap = cm.Purples  # noqa

        if not valid:
            cmap = cm.Reds  # noqa

        if match:
            cmap = cm.Blues  # noqa

        if correct:
            cmap = cm.Greens  # noqa

        col_data = {
            0: self.norm(all_ids, cvt=False),
            1: self.norm(all_preds, cvt=True),
            2: self.norm(all_probs, cvt=True),
            3: self.norm(all_labels, cvt=True),
            4: (None, us),
            5: (None, es),
            6: (None, vs),
        }

        all_columns_data = [v[1] for k, v in col_data.items()]
        columns = [
            f"[{n}]",
            r"$y$",
            r"$\sigma$",
            r"$\checkmark$" + f" ({n_correct})",
            "u",
            "-",
            "v",
        ]

        for i, ax in enumerate(axes):
            table_indices = all_indices[i * table_size : i * table_size + table_size]

            if not table_indices:
                return

            (ids, labels, preds, fracs, us, es, vs) = [
                [x[i] for i in table_indices] for x in all_columns_data
            ]

            empty = [""] * len(us)

            # connections don't have a cell text, leave them empty
            cell_text = list(zip(ids, labels, preds, fracs, empty, empty, empty))

            frac = 0.85

            table = ax.table(
                colLabels=columns,
                cellText=cell_text,
                cellLoc="center",
                colWidths=[frac / 4] * 4 + [(1.0 - frac) / 3] * 3,
                loc="upper left",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)

            if idx in ids:
                i = ids.index(idx) + 1
                table[(i, 0)].set_facecolor(color)

            for (row, col), cell in table.get_celld().items():
                cell.set_height(0.04)

                # must be placed at the beginning for proper header handling
                if 4 <= col <= 6:

                    if row == 0:
                        continue

                    cs = {4: us, 5: es, 6: vs}
                    bg = cs[col][row - 1]
                    if conn_types[0] == "color":
                        cell.set_facecolor(bg)

                    if conn_types[0] == "text":
                        cell.get_text().set_text(bg)

                    continue

                if row == 0 or col == 0:
                    continue

                col_norm, col_values = col_data[col]

                col_values = [col_values[i] for i in table_indices]

                row_value = col_values[row - 1]
                row_norm_v = col_norm(row_value)

                cell.set_facecolor(cmap(row_norm_v))

                if row_norm_v > 0.5:
                    cell.get_text().set_color("white")

    def plot_step(
        self,
        t: int,
        step,
        match: bool,
        correct: bool,
        n_per_table: int,
        n_tables,
    ):
        graph, labels, probs, idx, valid, connections, top_k = step

        fig, axes = pyplot.subplots(2, max(2 + n_tables, len(top_k)), figsize=(24, 12))

        for ax in axes.ravel():
            ax.axis("off")

        axes[0, 0].imshow(graph.image[0].permute(1, 2, 0).numpy())

        G = self.dataset.to_nx(graph)

        self.dataset.generator.draw_graph(G, axes[0, 1])

        if probs is not None:
            self.plot_table(
                axes[0, 2:],
                probs,
                labels,
                match,
                correct,
                valid,
                idx,
                n_per_table,
                connections,
            )

        for i, (idx, g) in enumerate(top_k):
            self.dataset.generator.draw_graph(g, axes[1, i])
            axes[1, i].set_title(f"Graph {idx}")

        header = [
            f"step: {t}",
            f"terminal: {graph.terminal}",
            f"match: {match}",
            f"correct: {correct}",
            f"valid: {valid}",
        ]

        fig.suptitle(", ".join(header))
        fig.tight_layout()

        canvas = FigureCanvasAgg(fig)
        s, (width, height) = canvas.print_to_buffer()
        frame = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        return frame

    def plot(self, filename: str | Path, trajectory: TrajectoryContainer):
        npt = 25
        nt = math.ceil(min(len(trajectory.steps[-1][1]), self.max_entries) / npt)

        file = os.path.join(self.directory, f"{filename}.mp4")

        steps = trajectory.steps
        frames = [
            self.plot_step(i, step, False, False, npt, nt)
            for i, step in enumerate(steps[:-1])
        ]
        frames += [
            self.plot_step(
                len(steps) - 1,
                steps[-1],
                trajectory.match,
                trajectory.correct,
                npt,
                nt,
            )
        ]

        with imageio.get_writer(file, fps=10) as writer:
            for frame in frames:
                writer.append_data(frame)

        pyplot.close("all")
