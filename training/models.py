from __future__ import annotations

import numpy as np
from keras import losses, metrics, optimizers
from keras import Input, Model
from keras.layers import (
    Add,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
    Bidirectional,
)


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
