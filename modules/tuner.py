"""
Author: Ahmad Ramadhan
Date: 2024-05-02
This is the tuner.py module.
Usage:
- For tuning best parameter ML
"""
from typing import NamedTuple, Dict, Text, Any
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from transform import (
    FEATURE_KEY,
    transformed_name,
    input_fn,
    VOCAB_SIZE,
    vectorize_layer
)


def model_builder(hp):
    """Build keras tuner model"""
    embedding_dim = hp.Int(
        'embedding_dim',
        min_value=16,
        max_value=128,
        step=16)
    lstm_units = hp.Int('lstm_units', min_value=16, max_value=128, step=16)
    num_layers = hp.Choice('num_layers', values=[1, 2, 3])
    dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float(
        'dropout_rate',
        min_value=0.1,
        max_value=0.5,
        step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

    inputs = tf.keras.Input(
        shape=(
            1,
        ),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.string)

    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(
        VOCAB_SIZE,
        embedding_dim,
        name='embedding')(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(x)
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model


TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    verbose=1,
    patience=2,
    min_delta=0,
    baseline=0.9,
    restore_best_weights=True
)


def tuner_fn(fn_args: FnArgs) -> None:
    """ Running Tuner for best training """
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, 10)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)
        ]]
    )

    # Build the model tuner
    model_tuner =kt.Hyperband(
        hypermodel=model_builder,
        objective=kt.Objective('val_binary_accuracy', direction='max'),
        max_epochs=5,
        factor=3,
        # directory    = fn_args.PIPELINE_DIR,
        project_name='tuner_trials',
    )

    model_tuner.oracle.max_trials = 3

    return TunerFnResult(
        tuner=model_tuner,
        fit_kwargs={
            'callbacks': [early_stop_callback],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
