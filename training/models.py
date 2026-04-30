from __future__ import annotations

import numpy as np
from keras import losses, metrics, ops, optimizers
from keras import Input, Model
from keras.layers import (
    Add,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Layer,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
    Reshape,
    Bidirectional,
)
from keras.ops import einsum
from keras.saving import register_keras_serializable


@register_keras_serializable(package="training")
class GraphConvolution(Layer):
    """Simple graph convolution over the node axis with learnable adjacency."""

    def __init__(self, units: int, activation: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                "GraphConvolution expects input shape (batch, time, nodes, channels)."
            )
        node_count = int(input_shape[-2])
        input_channels = int(input_shape[-1])
        self.adjacency = self.add_weight(
            name="adjacency",
            shape=(node_count, node_count),
            initializer="identity",
            trainable=True,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_channels, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        mixed_nodes = einsum("ij,btjc->btic", self.adjacency, inputs)
        outputs = einsum("btic,cu->btiu", mixed_nodes, self.kernel)
        outputs = outputs + self.bias
        if self.activation == "relu":
            return ops.relu(outputs)
        if self.activation is None:
            return outputs
        raise ValueError(f"Unsupported activation for GraphConvolution: {self.activation}")

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "units": self.units,
            "activation": self.activation,
        }


def build_cnn1d(input_shape: tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=input_shape, name="skeleton_sequence")
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs")(x)
    return Model(inputs=inputs, outputs=outputs, name="cnn1d_sign_classifier")


def build_lstm(input_shape: tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=input_shape, name="skeleton_sequence")
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs")(x)
    return Model(inputs=inputs, outputs=outputs, name="lstm_sign_classifier")


def _transformer_block(x, embed_dim: int, num_heads: int, ff_dim: int):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(int(x.shape[-1]))(ff)
    x = Add()([x, ff])
    return LayerNormalization()(x)


def build_transformer(input_shape: tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=input_shape, name="skeleton_sequence")
    x = Dense(128)(inputs)
    x = _transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256)
    x = _transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs")(x)
    return Model(inputs=inputs, outputs=outputs, name="transformer_sign_classifier")


def build_gcn(input_shape: tuple[int, int], num_classes: int) -> Model:
    time_steps, feature_count = input_shape
    if feature_count % 5 != 0:
        raise ValueError(
            "GCN requires the flattened feature dimension to be divisible by 5 "
            f"(got {feature_count})."
        )

    node_count = feature_count // 5
    inputs = Input(shape=input_shape, name="skeleton_sequence")
    x = Reshape((time_steps, node_count, 5), name="reshape_to_graph")(inputs)
    x = GraphConvolution(64, activation="relu", name="graph_conv_1")(x)
    x = LayerNormalization(name="graph_norm_1")(x)
    x = Dropout(0.2, name="graph_dropout_1")(x)
    x = GraphConvolution(128, activation="relu", name="graph_conv_2")(x)
    x = LayerNormalization(name="graph_norm_2")(x)
    x = GlobalAveragePooling2D(name="spatiotemporal_pool")(x)
    x = Dense(128, activation="relu", name="graph_projection")(x)
    x = Dropout(0.2, name="graph_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs")(x)
    return Model(inputs=inputs, outputs=outputs, name="gcn_sign_classifier")


def build_model(
    model_name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float,
) -> Model:
    builders = {
        "cnn1d": build_cnn1d,
        "lstm": build_lstm,
        "transformer": build_transformer,
        "gcn": build_gcn,
    }
    if model_name not in builders:
        raise ValueError(
            f"Unsupported model '{model_name}'. Expected one of: {', '.join(sorted(builders))}"
        )
    model = builders[model_name](input_shape=input_shape, num_classes=num_classes)
    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
            metrics.F1Score(name="f1_score", average="macro"),
        ],
    )
    return model


def model_parameter_count(model: Model) -> int:
    return int(np.sum([np.prod(weight.shape) for weight in model.trainable_weights]))
