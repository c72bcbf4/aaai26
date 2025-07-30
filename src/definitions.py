from dataclasses import dataclass
from typing import List


@dataclass
class MLPConf:
    hidden_sizes: List[int]


@dataclass
class CNNConf:
    layers: List[dict]


@dataclass
class GNNConf:
    hidden_size: int
    out_size: int
    num_layers: int
    degree_embed_dim: int
    node_embed_dim: int
    edge_embed_dim: int
    pooling: str
    jk: str


@dataclass
class AgentConf:
    gnn: GNNConf
    cnn: CNNConf
    mlp: MLPConf

    normalization_cnn: str | None
    normalization_gnn: str | None
    normalization_mlp: str | None

    dropout_cnn: float
    dropout_gnn: float
    dropout_mlp: float


@dataclass
class EnvironmentConf:
    train: object
    val: object
    max_node_degree: int | None


@dataclass
class TrainingConf:
    epochs: int
    samples_per_epoch: int
    evals_per_epoch: int
    batch_size: int
    weight_decay: int
    lr: float
    lr_cycle: int
    lr_warmup: int
    lr_factor: int
    max_grad_norm: float
    cache_size: int
    cache_warmup: float
    image_size: int
    image_warmup: int
    label_smoothing: float


@dataclass
class AppConfig:
    output_dir: str
    overrides: List[str]
    agent: AgentConf
    environment: EnvironmentConf
    training: TrainingConf
