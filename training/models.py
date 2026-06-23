from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from keras import Input, Model, losses, metrics, ops, optimizers
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
from keras.metrics import Metric
from keras.saving import register_keras_serializable


MODEL_DTYPE = "float32"


@register_keras_serializable(package="training")
class SparseMacroF1Score(Metric):
    def __init__(self, num_classes: int, name: str = "f1_score", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=MODEL_DTYPE)
        self.false_positives = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=MODEL_DTYPE)
        self.false_negatives = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=MODEL_DTYPE)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        del sample_weight
        y_true = ops.cast(ops.reshape(y_true, (-1,)), "int32")
        y_pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        y_pred = ops.reshape(y_pred, (-1,))

        y_true_one_hot = ops.cast(ops.one_hot(y_true, self.num_classes), MODEL_DTYPE)
        y_pred_one_hot = ops.cast(ops.one_hot(y_pred, self.num_classes), MODEL_DTYPE)

        true_positives = ops.sum(y_true_one_hot * y_pred_one_hot, axis=0)
        false_positives = ops.sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        false_negatives = ops.sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision_denominator = self.true_positives + self.false_positives
        recall_denominator = self.true_positives + self.false_negatives
        precision = ops.where(precision_denominator > 0, self.true_positives / precision_denominator, 0.0)
        recall = ops.where(recall_denominator > 0, self.true_positives / recall_denominator, 0.0)
        f1_denominator = precision + recall
        f1 = ops.where(f1_denominator > 0, 2.0 * precision * recall / f1_denominator, 0.0)
        return ops.mean(f1)

    def reset_state(self) -> None:
        zeros = ops.zeros((self.num_classes,), dtype=MODEL_DTYPE)
        self.true_positives.assign(zeros)
        self.false_positives.assign(zeros)
        self.false_negatives.assign(zeros)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "num_classes": self.num_classes}


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
            dtype=MODEL_DTYPE,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_channels, self.units),
            initializer="glorot_uniform",
            trainable=True,
            dtype=MODEL_DTYPE,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            dtype=MODEL_DTYPE,
        )
        super().build(input_shape)

    def call(self, inputs):
        mixed_nodes = ops.einsum("ij,btjc->btic", self.adjacency, inputs)
        outputs = ops.einsum("btic,cu->btiu", mixed_nodes, self.kernel)
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
    inputs = Input(shape=input_shape, name="skeleton_sequence", dtype=MODEL_DTYPE)
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu", dtype=MODEL_DTYPE)(inputs)
    x = Dropout(0.2)(x)
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu", dtype=MODEL_DTYPE)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu", dtype=MODEL_DTYPE)(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs", dtype=MODEL_DTYPE)(x)
    return Model(inputs=inputs, outputs=outputs, name="cnn1d_sign_classifier")


def build_lstm(input_shape: tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=input_shape, name="skeleton_sequence", dtype=MODEL_DTYPE)
    x = Bidirectional(LSTM(128, return_sequences=True, dtype=MODEL_DTYPE))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, dtype=MODEL_DTYPE))(x)
    x = Dense(128, activation="relu", dtype=MODEL_DTYPE)(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs", dtype=MODEL_DTYPE)(x)
    return Model(inputs=inputs, outputs=outputs, name="lstm_sign_classifier")


def _transformer_block(x, embed_dim: int, num_heads: int, ff_dim: int):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype=MODEL_DTYPE)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(dtype=MODEL_DTYPE)(x)
    ff = Dense(ff_dim, activation="relu", dtype=MODEL_DTYPE)(x)
    ff = Dense(int(x.shape[-1]), dtype=MODEL_DTYPE)(ff)
    x = Add()([x, ff])
    return LayerNormalization(dtype=MODEL_DTYPE)(x)


def build_transformer(input_shape: tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=input_shape, name="skeleton_sequence", dtype=MODEL_DTYPE)
    x = Dense(128, dtype=MODEL_DTYPE)(inputs)
    x = _transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256)
    x = _transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs", dtype=MODEL_DTYPE)(x)
    return Model(inputs=inputs, outputs=outputs, name="transformer_sign_classifier")


def build_gcn(input_shape: tuple[int, int], num_classes: int) -> Model:
    time_steps, feature_count = input_shape
    if feature_count % 5 != 0:
        raise ValueError(
            "GCN requires the flattened feature dimension to be divisible by 5 "
            f"(got {feature_count})."
        )

    node_count = feature_count // 5
    inputs = Input(shape=input_shape, name="skeleton_sequence", dtype=MODEL_DTYPE)
    x = Reshape((time_steps, node_count, 5), name="reshape_to_graph")(inputs)
    x = GraphConvolution(64, activation="relu", name="graph_conv_1", dtype=MODEL_DTYPE)(x)
    x = LayerNormalization(name="graph_norm_1", dtype=MODEL_DTYPE)(x)
    x = Dropout(0.2, name="graph_dropout_1")(x)
    x = GraphConvolution(128, activation="relu", name="graph_conv_2", dtype=MODEL_DTYPE)(x)
    x = LayerNormalization(name="graph_norm_2", dtype=MODEL_DTYPE)(x)
    x = GlobalAveragePooling2D(name="spatiotemporal_pool")(x)
    x = Dense(128, activation="relu", name="graph_projection", dtype=MODEL_DTYPE)(x)
    x = Dropout(0.2, name="graph_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probs", dtype=MODEL_DTYPE)(x)
    return Model(inputs=inputs, outputs=outputs, name="gcn_sign_classifier")


def build_model(
    model_name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.0001,
    global_clipnorm: float = 1.0,
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
    optimizer_builders = {
        "adamw": lambda: optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            global_clipnorm=global_clipnorm,
        ),
        "adam": lambda: optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=global_clipnorm,
        ),
        "sgd": lambda: optimizers.SGD(
            learning_rate=learning_rate,
            global_clipnorm=global_clipnorm,
        ),
    }
    if optimizer_name not in optimizer_builders:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Expected one of: {', '.join(sorted(optimizer_builders))}"
        )

    model = builders[model_name](input_shape=input_shape, num_classes=num_classes)
    model.compile(
        optimizer=optimizer_builders[optimizer_name](),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMacroF1Score(num_classes=num_classes, name="f1_score"),
        ],
    )
    return model


def model_parameter_count(model: Model) -> int:
    return int(np.sum([np.prod(weight.shape) for weight in model.trainable_weights]))
