import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, Tensor
from torch.nn import init
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

from src.data import Graph
from src.definitions import AgentConf
from src.util import (
    Data,
    make_norm2d,
    make_norm1d,
    RunningNormalize,
    ConcatNet,
)


class GraphRecognitionModel(nn.Module):
    def __init__(self, conf: AgentConf, sample: Graph, normalizer: RunningNormalize):
        super().__init__()
        self.c = conf

        self.activation = torch.nn.Mish()

        self.gnn = GNN(
            node_dim=sample.nodes.shape[-1],
            edge_dim=sample.edges_f.shape[-1],
            hidden_size=self.c.gnn.hidden_size,
            out_size=self.c.gnn.out_size,
            num_layers=self.c.gnn.num_layers,
            degree_embed_dim=self.c.gnn.degree_embed_dim,
            node_embed_dim=self.c.gnn.node_embed_dim,
            edge_embed_dim=self.c.gnn.edge_embed_dim,
            norm=self.c.normalization_gnn,
            mode=self.c.gnn.jk,
            pooling=self.c.gnn.pooling,
            hidden_activation=self.activation,
            dropout=self.c.dropout_gnn,
        )

        self.graph_embedding_size = self.gnn.embedding_size

        self.cnn = ResNetV2(
            sample_image=sample.image,
            sample_embedding=torch.rand((1, self.graph_embedding_size)),
            conf=self.c.cnn.layers,
            norm=self.c.normalization_cnn,
            activation=self.activation,
            dropout=self.c.dropout_cnn,
        )

        self.image_embedding_size = self.cnn.embedding_size

        embedding_size = self.image_embedding_size + 2
        self.mlp_norm = make_norm1d(self.c.normalization_mlp, embedding_size)

        self.mlp = MultiLayerPerceptron(
            input_size=embedding_size,
            hidden_sizes=self.c.mlp.hidden_sizes,
            hidden_activation=self.activation,
            output_size=1,
            norm=self.c.normalization_mlp,
            dropout=self.c.dropout_mlp,
        )

        self.gnn.apply(self.init_gnn)
        self.cnn.apply(self.init_cnn)
        self.mlp.apply(self.init_mlp)

        self.normalizer = normalizer

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def init_gnn(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                init.zeros_(m.bias)

    @torch.no_grad()
    def init_cnn(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                init.zeros_(m.bias)

    @torch.no_grad()
    def init_mlp(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                init.zeros_(m.bias)

    @torch.no_grad()
    def _parameter_count(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @torch.no_grad()
    def _parameter_norm(self, model):
        return torch.as_tensor([p.norm(2) for p in model.parameters()]).mean().item()

    @torch.no_grad()
    def parameter_count(self):
        return {
            "cnn": self._parameter_count(self.cnn),
            "gnn": self._parameter_count(self.gnn),
            "mlp": self._parameter_count(self.mlp),
            "all": self._parameter_count(self),
        }

    @torch.no_grad()
    def parameter_norm(self):
        return {
            "norm_cnn": self._parameter_norm(self.cnn),
            "norm_gnn": self._parameter_norm(self.gnn),
            "norm_mlp": self._parameter_norm(self.mlp),
        }

    def normalize(self, images):
        return self.normalizer(images)

    def forward(self, batch, update=True):
        graph_embeddings = self.gnn(batch)
        images = batch["images"]

        if update:
            self.normalizer.update(images)

        images = self.normalize(images)

        image_embeddings = self.cnn(images, graph_embeddings)

        terminals = batch["terminals"].to(torch.int64)
        terminals = torch.nn.functional.one_hot(terminals, 2).flatten(1)

        embeddings = torch.cat([image_embeddings, terminals], dim=-1)
        embeddings = self.mlp_norm(embeddings)

        return self.mlp(embeddings).reshape(-1)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: callable = None,
        bias: bool = True,
        norm: str = None,
        dropout: float = None,
        output_activation: callable = None,
        hidden_init_fn: callable = lambda x: x,
        output_init_fn: callable = lambda x: x,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.bias = bias
        self.norm = norm
        self.dropout = dropout
        self.hidden_activation = hidden_activation or nn.ReLU()

        in_sizes = [input_size] + hidden_sizes
        out_sizes = hidden_sizes + [output_size]

        hidden_layers = []
        for c_in, c_out in zip(in_sizes[:-1], out_sizes[:-1]):
            linear = nn.Linear(c_in, c_out, bias=bias)
            linear = hidden_init_fn(linear)
            norm = make_norm1d(self.norm, c_out)

            hidden_layers.append(linear)
            hidden_layers.append(norm)
            hidden_layers.append(self.hidden_activation)

            if self.dropout is not None:
                dropout = nn.Dropout(self.dropout)
                hidden_layers.append(dropout)

        linear = nn.Linear(in_sizes[-1], out_sizes[-1])
        linear = output_init_fn(linear)

        hidden_layers.append(linear)

        if output_activation is not None:
            hidden_layers.append(output_activation)

        self.module = nn.Sequential(*hidden_layers)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.module.parameters())

    def forward(self, x):
        return self.module(x)


class GNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_size: int,
        num_layers: int,
        degree_embed_dim: int,
        node_embed_dim: int,
        edge_embed_dim: int,
        mode: str | None,
        pooling: str,
        out_size: int = None,
        norm: str = None,
        hidden_activation: callable = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.degree_embed_dim = degree_embed_dim
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.mode = mode
        self.pooling = pooling
        self.norm = norm
        self.dropout = dropout
        self.hidden_activation = hidden_activation

        self.degree_embedding = (
            torch.nn.Embedding(100, self.degree_embed_dim)
            if self.degree_embed_dim > 0
            else None
        )

        self.node_embedding = torch.nn.Embedding(self.node_dim, self.node_embed_dim)
        self.edge_embedding = torch.nn.Embedding(self.edge_dim, self.edge_embed_dim)

        self.gnn = ConcatNet(
            in_channels=self.node_embed_dim + self.degree_embed_dim,
            edge_dim=self.edge_embed_dim,
            hidden_channels=self.hidden_size,
            num_layers=self.num_layers,
            out_channels=self.out_size or self.hidden_size,
            act=self.hidden_activation,
            jk=self.mode,
            norm=self.norm,
            dropout=self.dropout,
            bias=False,
        )

        poolings = {
            "mean": global_mean_pool,
            "sum": global_add_pool,
        }

        self.pool = poolings[self.pooling]
        self.embedding_size = self.out_size or self.hidden_size

    def forward(self, batch: Data):
        nodes = batch["nodes"]
        edges = batch["edges"]
        edge_index = batch["edge_index"]
        batch = batch["batch"]

        nodes = self.node_embedding(nodes.argmax(-1))
        edges = self.edge_embedding(edges.argmax(-1))

        if self.degree_embed_dim > 0:
            degrees = degree(edge_index[0], num_nodes=len(nodes), dtype=torch.int64)
            degrees = self.degree_embedding(degrees)
            nodes = torch.cat([nodes, degrees], dim=-1)

        node_embedding = self.gnn(x=nodes, edge_index=edge_index, edge_attr=edges)
        graph_embedding = self.pool(node_embedding, batch)

        return graph_embedding


class GlobalMaxPool2d(nn.Module):
    def forward(self, inputs):  # noqa
        return inputs.max(1, keepdim=True).values


class GlobalMeanPool2d(nn.Module):
    def forward(self, inputs):  # noqa
        return inputs.mean(1, keepdim=True)


class GlobalAvgMaxPool(nn.Module):
    def __init__(self):
        super(GlobalAvgMaxPool, self).__init__()
        self.avg_pool = GlobalMeanPool2d()
        self.max_pool = GlobalMaxPool2d()

    def forward(self, x):
        avg_pool = self.avg_pool(x).flatten(1)
        max_pool = self.max_pool(x).flatten(1)
        return torch.cat([avg_pool, max_pool], dim=1)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm: str | None,
        activation: callable,
        scale_factor=2,
        mode="bilinear",
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode=mode, align_corners=False
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = activation
        self.norm = make_norm2d(norm, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNetV2Layer(nn.Module):
    def __init__(
        self,
        embedding: Tensor,
        channels: int,
        kernel: int,
        stride: int,
        activation: nn.Module,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.embedding = embedding
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.conv_1 = self.make_conv()
        self.norm_1 = make_norm2d(self.norm, self.channels)

        self.conv_2 = self.make_conv()
        self.norm_2 = make_norm2d(self.norm, self.channels)

        self.drop = nn.Dropout2d(self.dropout)
        hidden_size = self.embedding.shape[-1]
        self.film_linear = nn.Sequential(
            torch.nn.Linear(hidden_size, 4 * self.channels)
        )

        self.film_norm = nn.LayerNorm(4 * self.channels)

    def make_proj(self, c_in, c_out):
        return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1)

    def make_conv(self):
        return nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.kernel // 2,
            bias=False,
        )

    def film_layer(self, conv, scale, shift):
        scale = scale[..., None, None].expand(*(scale.shape + conv.shape[-2:]))
        shift = shift[..., None, None].expand(*(shift.shape + conv.shape[-2:]))
        return scale * conv + shift

    def forward(self, inputs, embeddings):
        film = self.film_linear(embeddings)
        film = self.film_norm(film)
        scale_1, shift_1, scale_2, shift_2 = film.chunk(4, dim=-1)

        identity = inputs
        conv = inputs

        conv = self.norm_1(conv)
        conv = self.film_layer(conv, scale_1, shift_1)
        conv = self.activation(conv)
        conv = self.conv_1(conv)

        conv = self.norm_2(conv)
        conv = self.film_layer(conv, scale_2, shift_2)
        conv = self.activation(conv)
        conv = self.drop(conv)
        conv = self.conv_2(conv)

        return identity + conv


class ResNetV2(nn.Module):
    def __init__(
        self,
        sample_image: Tensor,
        sample_embedding: Tensor,
        conf: List[dict],
        norm: str | None,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.sample_image = sample_image
        self.sample_embedding = sample_embedding
        self.conf = conf
        self.norm = norm
        self.activation = activation
        self.dropout = dropout

        self.types = []
        self.encoder_layers = []

        down_samples = 0
        for i, l in enumerate(self.conf[:-1]):
            layer = self.make_layer(l)
            if l["type"] == "skip":
                self.sample_image = layer(self.sample_image, self.sample_embedding)
            else:
                self.sample_image = layer(self.sample_image)

            self.types.append(l["type"])
            self.encoder_layers.append(layer)

            logger.info(
                f"layer {i} shape: {tuple(self.sample_image.shape)}={tuple(self.sample_image.flatten(1).shape)}"
            )

            if l["type"] in ("conv", "skip"):
                down_samples += l["stride"] // 2
            if l["type"] in ("hidden-pool",):
                down_samples += 1

        self.encoder_layers = torch.nn.ModuleList(self.encoder_layers)

        assert len(self.types) == len(self.encoder_layers), "types do not match layers"
        self.global_pool = self.make_layer(self.conf[-1])

        self.sample_image = self.global_pool(self.sample_image)

        logger.info(
            f"final shape: {tuple(self.sample_image.shape)}={tuple(self.sample_image.flatten(1).shape)}"
        )

        self.embedding_size = self.sample_image.flatten(1).shape[-1]

    def make_layer(self, layer):
        if layer["type"] == "conv":
            conv = nn.Conv2d(
                in_channels=self.sample_image.shape[1],
                out_channels=layer["channels"],
                kernel_size=layer["kernel"],
                stride=layer["stride"],
                padding=layer["kernel"] // 2,
            )
            norm = make_norm2d(self.norm, layer["channels"])
            act = self.activation
            drop = nn.Dropout2d(self.dropout)
            return nn.Sequential(conv, norm, act, drop)

        if layer["type"] == "skip":
            return ResNetV2Layer(
                embedding=self.sample_embedding,
                channels=layer["channels"],
                kernel=layer["kernel"],
                stride=layer["stride"],
                activation=self.activation,
                norm=self.norm,
                dropout=self.dropout,
            )

        if layer["type"] == "hidden-pool":
            if layer["kind"] == "mean":
                return torch.nn.AvgPool2d(2)
            if layer["kind"] == "max":
                return torch.nn.MaxPool2d(2)
            raise ValueError(f"hidden pooling '{layer['kind']}'")

        if layer["type"] == "global-pool":
            if layer["kind"] == "mean":
                return GlobalMeanPool2d()
            if layer["kind"] == "max":
                return GlobalMaxPool2d()
            if layer["kind"] == "mean-max":
                return GlobalAvgMaxPool()

            raise ValueError(f"hidden global pooling '{layer['kind']}'")

        raise ValueError(f"invalid layer type '{layer['type']}'")

    def forward(self, inputs, embeddings=None):
        outputs = inputs

        for t, l in zip(self.types, self.encoder_layers):
            if t == "skip":
                outputs = l(outputs, embeddings)
                continue
            outputs = l(outputs)

        flat = self.global_pool(outputs).flatten(1)

        return flat
