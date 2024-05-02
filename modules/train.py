"""
Author: Ahmad Ramadhan
Date: 2024-05-02
This is the train.py module.
Usage:
- For training model
"""
import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from keras.utils.vis_utils import plot_model
from transform import (
    LABEL_KEY,
    FEATURE_KEY,
    transformed_name,
    input_fn,
    VOCAB_SIZE,
    vectorize_layer
)


def model_builder(hp):
    """Build machine learning model"""
    inputs = tf.keras.Input(
        shape=(
            1,
        ),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.string)

    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, hp['embedding_dim'], name='embedding')(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp['lstm_units']))(x)
    for _ in range(hp['num_layers']):
        x = tf.keras.layers.Dense(hp['dense_units'], activation='relu')(x)
    x = tf.keras.layers.Dropout(hp['dropout_rate'])(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Running Trainer"""
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    hp = fn_args.hyperparameters['values']

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        patience=10
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    callbacks = [
        tensorboard_callback,
        early_stop_callback,
        model_checkpoint_callback
    ]

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(
        fn_args.train_files,
        tf_transform_output,
        hp['tuner/epochs'])
    val_set = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        hp['tuner/epochs'])

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)
        ]]
    )

    # Build the model
    model = model_builder(hp)

    # Train the model
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=callbacks,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=hp['tuner/epochs']
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )

    plot_model(
        model,
        to_file='model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
