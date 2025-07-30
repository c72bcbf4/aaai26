import io
import itertools
import os
import random
import time
from abc import abstractmethod
from collections import defaultdict, deque
from multiprocessing import Process, Queue
from threading import Thread
from typing import List, NamedTuple, Dict

import networkx as nx
import numpy as np
import torch
from PIL import Image
from loguru import logger
from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import AtomValenceException
from torch import Tensor
from torch.utils.data import Dataset
from torchrl.data import ReplayBuffer, ListStorage
from torchvision.io import encode_jpeg, decode_jpeg
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm

from src.util import batch_fast


class Graph(NamedTuple):
    image: Tensor
    nodes: Tensor
    edges: Tensor
    edges_f: Tensor
    terminal: Tensor


class GraphTester:
    def filter_candidates(self, data_graph, query_graph):
        label_to_nodes = defaultdict(list)
        for n, data in data_graph.nodes(data=True):
            label = data.get("color", None)
            if label is not None:
                label_to_nodes[label].append(n)

        candidates = {}
        for qn, qdata in query_graph.nodes(data=True):
            qlabel = qdata.get("color", None)
            candidates[qn] = set(label_to_nodes.get(qlabel, []))
        return candidates

    def bfs_order(self, graph, root):
        visited = set()
        order = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                order.append(node)
                queue.extend(sorted(set(graph.neighbors(node)) - visited))
        return order

    def pair_hist(self, G: nx.Graph):
        hist = defaultdict(int)
        for u, v, edata in G.edges(data=True):
            c_u = G.nodes[u]["color"]
            c_v = G.nodes[v]["color"]
            e_color = edata.get("color", None)
            key = (tuple(sorted((c_u, c_v))), e_color)
            hist[key] += 1
        return hist

    def approximate_subgraph_match(self, data_graph, query_graph):
        data_hist = self.pair_hist(data_graph)
        query_hist = self.pair_hist(query_graph)

        for k in query_hist.keys():
            if query_hist[k] > data_hist[k]:
                return False

        candidates = self.filter_candidates(data_graph, query_graph)
        root = list(query_graph.nodes())[0]
        matching_order = self.bfs_order(query_graph, root)

        def backtrack(mapping, depth=0):
            if len(mapping) == len(query_graph):
                return True

            qnode = matching_order[depth]
            for dnode in candidates[qnode]:
                if dnode in mapping.values():
                    continue
                # local consistency check: preserve edges and edge colors
                consistent = True
                for qnbr in query_graph.neighbors(qnode):
                    if qnbr in mapping:
                        dqnbr = mapping[qnbr]
                        if not data_graph.has_edge(dnode, dqnbr):
                            consistent = False
                            break
                        q_edge_color = query_graph[qnode][qnbr].get("color", None)
                        d_edge_color = data_graph[dnode][dqnbr].get("color", None)
                        if q_edge_color != d_edge_color:
                            consistent = False
                            break
                if consistent:
                    mapping[qnode] = dnode
                    if backtrack(mapping, depth + 1):
                        return True
                    del mapping[qnode]
            return False

        return backtrack({})


class GraphGenerator:
    def __init__(self):
        self.node_colors = None
        self.edge_colors = None

        self.node_color_idx = None
        self.edge_color_idx = None

    @abstractmethod
    def get_graph(self, idx=None) -> nx.Graph:
        pass

    @abstractmethod
    def get_frame(self, graph):
        pass

    @abstractmethod
    def draw_graph(self, G, ax):
        pass

    @abstractmethod
    def get_connection(self, target_color, edge_color, node_color):
        pass


class TreeGenerator(GraphGenerator):
    def __init__(
        self,
        min_nodes: int,
        max_nodes: int,
        n_node_colors: int,
        n_edge_colors: int,
    ):
        super().__init__()
        self.markers = ["o", "v", "^", "D", ">", "8", "s", "p", "*", "h", "X"]

        colors = ["black", "red", "green", "blue", "cyan", "magenta", "grey"]

        self.node_colors = colors[:n_node_colors]
        self.edge_colors = colors[:n_edge_colors]

        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

        self.range = np.arange(min_nodes, max_nodes + 1).astype(float)

        self.node_color_idx = {c: i for i, c in enumerate(self.node_colors)}
        self.edge_color_idx = {c: i for i, c in enumerate(self.edge_colors)}

        self.styles = ["solid"]

    def get_connection(self, target_color, edge_color, node_color):
        if target_color is None:
            return {"type": "color", "connection": ("white",) * 3}

        return {"type": "color", "connection": (target_color, edge_color, node_color)}

    def get_graph(self, idx=None):
        generators = [
            lambda n: nx.random_labeled_tree(n),  # noqa
            # lambda n: nx.barabasi_albert_graph(n, np.random.randint(1, 3)),
        ]
        n_nodes = int(np.random.choice(self.range))
        graph = np.random.choice(generators)(n_nodes)

        for node in graph.nodes:
            graph.nodes[node]["color"] = random.choice(self.node_colors)
            graph.nodes[node]["marker"] = random.choice(self.markers)

        for edge in graph.edges:
            graph.edges[edge]["color"] = random.choice(self.edge_colors)
            graph.edges[edge]["style"] = random.choice(self.styles)

        return graph

    def draw_graph(self, G, ax):
        node_colors = [n["color"] for _, n in G.nodes(data=True)]
        edge_colors = [e["color"] for _, _, e in G.edges(data=True)]
        nx.draw_kamada_kawai(G, ax=ax, node_color=node_colors, edge_color=edge_colors)

    def get_frame(self, graph):
        fig = pyplot.figure(figsize=(3, 3))

        layout = nx.kamada_kawai_layout(graph)

        canvas = FigureCanvasAgg(fig)

        for node, data in graph.nodes(data=True):
            nx.draw_networkx_nodes(
                graph,
                layout,
                [node],
                node_color=data["color"],
                node_shape=data["marker"],
                node_size=100,
            )

        for u, v, data in graph.edges(data=True):
            nx.draw_networkx_edges(
                graph,
                layout,
                edgelist=[(u, v)],
                style=data["style"],
                edge_color=data["color"],
                width=3,
            )

        pyplot.axis("off")
        pyplot.tight_layout()
        canvas.draw()

        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(height, width, 4)[:, :, :3]

        pyplot.close(fig)
        frame = Image.fromarray(frame)
        rgb = Image.new("RGB", frame.size, (255, 255, 255))
        rgb.paste(frame)
        return rgb


class QM9Generator(GraphGenerator):
    def __init__(self, train: bool):
        super().__init__()

        self.train = train

        self.node_color_idx = {
            6: 0,  # C
            1: 1,  # H
            8: 2,  # O
            7: 3,  # N
            9: 4,  # F
        }

        self.node_label_idx = {
            6: "C",  # C
            1: "H",  # H
            8: "O",  # O
            7: "N",  # N
            9: "F",  # F
        }

        self.edge_color_idx = {
            1: 0,  # single
            2: 1,  # double
            3: 2,  # triple
        }

        self.edge_label_idx = {
            1: "-",  # single
            2: "=",  # double
            3: r"$\equiv$",  # triple
        }

        self.node_colors = list(self.node_color_idx.keys())
        self.edge_colors = list(self.edge_color_idx.keys())

        self.source_split = "data/qm9_idx.pt"
        self.source_file = "data/qm9.pt"

        if not os.path.exists(self.source_file):
            self.process_raw_data(f"data/qm9.csv")

        logger.info(f"loading data from {self.source_file}")
        self.source = torch.load(self.source_file, weights_only=False)

        if not os.path.exists(self.source_split):
            idx = torch.randperm(len(self.source))
            torch.save(idx, self.source_split)

        self.n = 10_000
        self.idx = torch.load(self.source_split)
        self.idx = self.idx[: -self.n] if self.train else self.idx[-self.n :]

        self.data = [self.source[i] for i in self.idx]

    def get_connection(self, target_color, edge_color, node_color):
        if target_color is None:
            return {"type": "text", "connection": ("", "", "")}

        u = self.node_label_idx[target_color]
        e = self.edge_label_idx[edge_color]
        v = self.node_label_idx[node_color]

        return {"type": "text", "connection": (u, e, v)}

    def smiles_to_nx(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), color=atom.GetAtomicNum())

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                color=int(bond.GetBondTypeAsDouble()),
            )

        return G

    def process_raw_data(self, file):
        with open(file, f"r") as f:
            data = f.readlines()

        data = [x.strip() for x in data]

        result = [(x, self.smiles_to_nx(x)) for x in tqdm(data)]
        results = [(s, g) for s, g in result if g is not None]
        torch.save(results, self.source_file)

    def prepare_data(self):
        pass

    def nx_to_rdkit(self, graph: nx.Graph):
        mol = Chem.RWMol()
        node_to_idx = {}

        for node, data in graph.nodes(data=True):
            atom_symbol = data["color"]
            atom = Chem.Atom(atom_symbol)
            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx

        bond_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            12: Chem.BondType.AROMATIC,
        }

        for u, v, data in graph.edges(data=True):
            bond_type = bond_map[data["color"]]
            mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

        mol = mol.GetMol()

        for atom in mol.GetAtoms():
            atom.SetFormalCharge(0)

        return mol

    def get_graph(self, idx=None) -> nx.Graph:
        if idx is None:
            return random.choice(self.data)[1]
        return self.data[idx][1]

    def get_drawer(self):
        drawer = rdMolDraw2D.MolDraw2DCairo(512, 512)
        options = drawer.drawOptions()

        options.useBWAtomPalette()

        options.addAtomIndices = False
        options.addStereoAnnotation = False
        options.includeAtomTags = False
        options.explicitMethyl = False
        options.suppressHydrogens = True
        options.clearBackground = False
        options.continuousHighlight = True
        options.dontUseAtomChargeColors = True
        options.includeAtomNumbers = False

        return drawer

    def draw_mol_frame(self, graph: nx.Graph):
        # TODO consider batch-plotting
        mol = self.nx_to_rdkit(graph)

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)

        drawer = self.get_drawer()
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        image = Image.open(io.BytesIO(drawer.GetDrawingText()))

        rgb = Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.getchannel("A"))
        return rgb

    def draw_graph(self, G, ax):
        try:
            img = self.draw_mol_frame(G)
            ax.imshow(img)
        except AtomValenceException:
            ax.text(x=0.5, y=0.5, s="invalid mol", fontsize=24)

    def get_frame(self, graph):
        return self.draw_mol_frame(graph)


class GraphDataset(Dataset):
    def __init__(
        self,
        generator: GraphGenerator,
        image_size: int,
        fake_size: int,
        min_size: int,
        batch_size: int,
        cache_size: int,
        device: str,
        max_node_degree: int = None,
    ):
        self.generator = generator
        self.image_size = image_size
        self.fake_size = fake_size
        self.min_size = min_size
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.device = device
        self.max_node_degree = max_node_degree

        self.t = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.RandomAffine(
                    degrees=0,
                    scale=(0.8, 1.0),
                    shear=15,
                    fill=255,
                    interpolation=InterpolationMode.BICUBIC,
                ),
            ]
        )

        self.buffer_pos = None
        self.buffer_neg = None
        self.worker = None

        self.process = None
        self.q = None

        self.size_pos = 0
        self.size_neg = 0
        self.pos_sample = 0.5
        self.pos_true = 0.0
        self.sps = 0.0
        self.gpi = 0.0

    def to_nx(self, graph: Graph):
        G = nx.Graph()

        nodes = graph.nodes.argmax(-1).tolist()
        edges = graph.edges.T.tolist()
        edges_f = graph.edges_f.argmax(-1).tolist()

        for i, x in enumerate(nodes):
            G.add_node(i, color=self.generator.node_colors[x])

        for (u, v), x in zip(edges, edges_f):
            G.add_edge(u, v, color=self.generator.edge_colors[x])

        G.graph["terminal"] = graph.terminal  # noqa
        G.graph["image"] = graph.image  # noqa

        return G

    def copy(self, G):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        H.add_edges_from(G.edges(data=True))
        H.graph.update(G.graph)  # noqa
        return H

    def successors_nx(self, G, terminal=True, max_nodes=None, max_edges=None):
        successors = {}

        nodes = (
            list(G.nodes)
            if self.max_node_degree is None
            else [n for n, c in G.degree() if c < self.max_node_degree]
        )

        missing_nodes = list(
            itertools.product(
                nodes, self.generator.node_colors, self.generator.edge_colors
            )
        )

        if max_nodes is not None and len(missing_nodes) > max_nodes:
            missing_nodes = random.sample(missing_nodes, max_nodes)

        new_node = max(G.nodes, default=-1) + 1
        for target_node, node_color, edge_color in missing_nodes:
            G_new = self.copy(G)
            G_new.add_node(new_node, color=node_color)
            G_new.add_edge(new_node, target_node, color=edge_color)

            target_color = G_new.nodes[target_node]["color"]
            G_new.graph["connection"] = self.generator.get_connection(
                target_color, edge_color, node_color
            )

            G_new_hash = self.hash(G_new)
            if G_new_hash not in successors:
                successors[G_new_hash] = G_new

        nodes = list(G.nodes)
        existing_edges = set(G.edges)
        all_possible_edges = set(itertools.combinations(nodes, 2))
        missing_edges = list(all_possible_edges - existing_edges)

        missing_edges = list(
            itertools.product(missing_edges, self.generator.edge_colors)
        )

        if max_edges is not None and len(missing_edges) > max_edges:
            missing_edges = random.sample(missing_edges, max_edges)

        for (u, v), edge_color in missing_edges:
            G_new = self.copy(G)

            u_color = G.nodes[u]["color"]
            v_color = G.nodes[u]["color"]

            G_new.add_edge(u, v, color=edge_color)

            G_new.graph["connection"] = self.generator.get_connection(
                u_color, edge_color, v_color
            )

            G_new_hash = self.hash(G_new)
            if G_new_hash not in successors:
                successors[G_new_hash] = G_new

        for g in successors.values():
            g.graph["terminal"] = False
            g.graph["image"] = G.graph["image"]

        if terminal:
            G_term = self.copy(G)
            G_term.graph["terminal"] = True  # noqa
            G_term.graph["image"] = G.graph["image"]  # noqa

            G_term.graph["connection"] = self.generator.get_connection(None, None, None)

            G_term_hash = self.hash(G_term)
            successors[G_term_hash] = G_term

        return successors

    def hash(self, graph: nx.Graph):
        return nx.weisfeiler_lehman_graph_hash(
            graph, node_attr="color", edge_attr="color", iterations=3, digest_size=16
        )

    def relabel(self, G: nx.Graph):
        mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
        return nx.relabel_nodes(G, mapping)

    def delete_edges(self, graph: nx.Graph):
        all_nodes = set(list(range(len(graph.nodes))))

        safe_nodes = all_nodes - set(nx.articulation_points(graph))
        safe_edges = list(
            itertools.chain.from_iterable(
                [[e for e in graph.edges() if n in e] for n in safe_nodes]
            )
        )

        sub_graphs = []

        for edge in safe_edges:
            g = self.copy(graph)  # noqa

            g.remove_edge(*edge)
            isolates = list(nx.isolates(g))

            # last removal leaves both nodes unconnected and creates 2 isolates
            if len(g.edges) == 0 and len(isolates) == 2:
                isolates = {np.random.choice(isolates)}

            assert len(isolates) <= 1, "edge removed more than one node"

            if isolates:
                g.remove_nodes_from(isolates)
                g = self.relabel(g)

            sub_graphs.append(g)

        return sub_graphs

    def from_nx(self, graph: nx.Graph):
        image = graph.graph["image"]  # noqa

        nodes = torch.tensor(
            [
                self.generator.node_color_idx[data["color"]]
                for _, data in graph.nodes(data=True)
            ]
        )
        nodes = torch.nn.functional.one_hot(nodes, len(self.generator.node_colors)).to(
            torch.float32
        )

        if len(graph.edges) > 0:
            edges = torch.tensor(list(graph.edges())).T
            edges = torch.cat([edges, edges.flip(0)], dim=1)
        else:
            edges = torch.empty((2, 0), dtype=torch.int64)

        edges_f = torch.tensor(
            [
                self.generator.edge_color_idx[data["color"]]
                for _, _, data in graph.edges(data=True)
            ],
            dtype=torch.int64,
        )
        edges_f = torch.nn.functional.one_hot(
            edges_f, len(self.generator.edge_colors)
        ).to(torch.float32)

        edges_f = torch.cat([edges_f, edges_f])

        terminal = torch.tensor(graph.graph["terminal"])

        assert edges.shape[1] == edges_f.shape[0], "mismatch between number of edges"

        return Graph(image, nodes, edges, edges_f, terminal)

    def get_item(self, encode_image=True, idx=None) -> Graph:
        graph = self.generator.get_graph(idx)

        image = self.generator.get_frame(graph)

        image = np.array(self.t(image))
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        image = (
            encode_jpeg(image, quality=90)
            if encode_image
            else image.unsqueeze(0) / 255.0
        )

        graph.graph["terminal"] = True  # noqa
        graph.graph["image"] = image  # noqa
        graph.graph["connection"] = self.generator.get_connection(None, None, None)

        # leafs are always terminal nodes
        return self.from_nx(graph)

    def get_sub_graphs(self, graph, n):
        sub_graphs = {}

        done = set()

        for _ in range(n):
            sub_graph = graph

            while len(sub_graph.nodes) > 1:
                deletions = self.delete_edges(sub_graph)
                deletions = {self.hash(d): d for d in deletions}
                hashes = set(deletions.keys())
                valid = hashes - done

                if not valid:
                    break

                h = random.choice(list(hashes))
                d = deletions[h]

                done.add(h)
                sub_graphs[h] = d
                sub_graph = d

        return sub_graphs

    def random_walk_subgraph(self, graph: nx.Graph, length=None) -> nx.Graph:
        if length is None:
            length = np.random.randint(1, len(graph.nodes) + 1)

        start_node = random.choice(list(graph.nodes()))
        visited = {start_node}
        current_node = start_node

        while len(visited) < length:
            neighbors = list(graph.neighbors(current_node))
            unvisited_neighbors = [n for n in neighbors if n not in visited]

            if unvisited_neighbors:
                next_node = random.choice(unvisited_neighbors)
            else:
                next_node = random.choice(neighbors)  # revisit if stuck

            visited.add(next_node)
            current_node = next_node

        subgraph = self.copy(graph.subgraph(visited))  # noqa
        return self.relabel(subgraph)

    def get_examples(self, graph: Graph) -> List[Dict]:
        target = self.to_nx(graph)
        target_hash = self.hash(target)

        sub_graphs = self.get_sub_graphs(target, n=10)

        results = deque()

        # correct and terminal graph must be added as it does
        # not occur as correct and terminal below
        # terminal states MUST be treated separately during decoding
        # since subgraph terminal states do not correspond to desired states
        assert graph.terminal, f"graph should be correct and terminal"

        # handle the terminal and correct graph. the correct but not
        # terminal graph is handled via sub-graphs.
        results.append({"x": graph, "y": torch.tensor(1.0)})

        done = set()
        tester = GraphTester()

        term_frac = 0.1

        for graph_hash, graph_nx in sub_graphs.items():
            successors_nx = self.successors_nx(
                graph_nx, terminal=False, max_nodes=5, max_edges=5
            )

            graph_nx.graph["terminal"] = np.random.rand() < term_frac
            graph_pt = self.from_nx(graph_nx)

            # no need to match sub-graph as we know it is one
            graph_pt_label = float(
                graph_hash == target_hash if graph_nx.graph["terminal"] else True
            )
            results.append({"x": graph_pt, "y": torch.tensor(graph_pt_label)})

            for succ_hash, succ_nx in successors_nx.items():
                if succ_hash in done:
                    continue

                done.add(succ_hash)

                succ_nx.graph["terminal"] = np.random.rand() < term_frac
                succ_pt = self.from_nx(succ_nx)
                succ_pt_label = float(
                    succ_hash == target_hash
                    if succ_nx.graph["terminal"]
                    else tester.approximate_subgraph_match(target, succ_nx)
                )
                results.append({"x": succ_pt, "y": torch.tensor(succ_pt_label)})

        return list(results)

    def collate(self, batch):
        batch = batch[0]
        images = [sample["x"].image.contiguous() for sample in batch]
        images = decode_jpeg(images, device=self.device)

        batch = [
            {
                "x": Graph(
                    image=image[None, ...] / 255.0,
                    nodes=sample["x"].nodes,
                    edges=sample["x"].edges,
                    edges_f=sample["x"].edges_f,
                    terminal=sample["x"].terminal,
                ),
                "y": sample["y"],
            }
            for (sample, image) in zip(batch, images)
        ]

        x = batch_fast([d["x"] for d in batch], device=self.device)
        y = torch.cat([d["y"].view(1) for d in batch]).to(self.device)

        return {"x": x, "y": y}

    def __len__(self):
        return self.fake_size // self.batch_size

    def work_process(self):
        while True:
            self.q.put(self.get_item())

    def work(self):
        start = time.time()
        n_images = 0
        n_graphs = 0
        n_pos = 0

        while True:
            graph = self.q.get()
            samples = self.get_examples(graph)

            positives = [s for s in samples if s["y"] > 0.0]
            negatives = [s for s in samples if s["y"] < 1.0]

            self.buffer_pos.add(positives)
            self.buffer_neg.add(negatives)

            n_images += 1
            n_graphs += len(samples)
            n_pos += len(positives)
            now = time.time()

            self.size_pos = len(self.buffer_pos)
            self.size_neg = len(self.buffer_neg)
            self.pos_true = n_pos / n_graphs
            self.sps = n_graphs / (now - start)
            self.gpi = n_graphs / n_images

    @property
    def buffer_size(self):
        return len(self.buffer_pos) + len(self.buffer_neg)

    def __getitem__(self, item):
        if (self.buffer_pos, self.buffer_neg) == (None, None):
            self.q = Queue(maxsize=100)
            self.process = Process(target=self.work_process)
            self.process.start()

            pos_batch_size = int(self.batch_size * self.pos_sample)
            neg_batch_size = self.batch_size - pos_batch_size

            self.buffer_pos = ReplayBuffer(
                storage=ListStorage(max_size=self.cache_size),
                batch_size=pos_batch_size,
                collate_fn=lambda x: x,
            )
            self.buffer_neg = ReplayBuffer(
                storage=ListStorage(max_size=self.cache_size),
                batch_size=neg_batch_size,
                collate_fn=lambda x: x,
            )
            self.worker = Thread(target=self.work)
            self.worker.start()

            while self.buffer_size < self.min_size:
                logger.info(
                    f"waiting for buffer... ({self.buffer_size}/{self.min_size})"
                )
                time.sleep(5)
            logger.info(f"waiting for buffer done ({self.buffer_size}/{self.min_size})")

        pos = [random.choice(x) for x in self.buffer_pos.sample()]
        neg = [random.choice(x) for x in self.buffer_neg.sample()]

        return pos + neg
