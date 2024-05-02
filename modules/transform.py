"""
Author: Ahmad Ramadhan
Date: 2024-05-02
This is the transform.py module.
Usage:
- Change text to int for tensorflow
"""
import tensorflow as tf

# Declare Variabel
LABEL_KEY = "Category"
FEATURE_KEY = "Message"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs,
        batch_size=64) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset


# Vocabulary size and number of words in a sequence
VOCAB_SIZE = 8000
SEQUENCE_LENGTH = 100
# embedding_dim   = 16

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Preprocess input features into transformed features


def preprocessing_fn(inputs):
    """
    inputs:  map from feature keys to raw features
    outputs: map from feature keys to transformed features
    """

    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
