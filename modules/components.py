"""
Author: Ahmad Ramadhan
Date: 2024-05-02
This is the components.py module.
Usage:
- COMPONENTS TFX PIPELINE
"""
import os
import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def init_components(config):
    """
    Initializes components for a machine learning pipeline, optionally with hyperparameter tuning.

    Args:
        data_dir: Path to the directory containing training and evaluation data.
        transform_module: Module containing data transformation functions.
        training_module: Module containing training functions.
        training_steps: Number of training steps.
        eval_steps: Number of evaluation steps.
        serving_model_dir: Directory to save the serving model.
        tuner_module: Module containing tuner functions.
    """

    # Examplegen
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ])
    )

    example_gen = CsvExampleGen(input_base=config['data_dir'], output_config=output)

    # Statisticgen
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # Schemagen
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics']
    )

    # Validator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Transform
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(config['transform_module'])
    )

    # Tuner
    tuner = Tuner(
        module_file=os.path.abspath(config['tuner_module']),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config["training_steps"]
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config["eval_steps"]
        )
    )

    # Trainer
    trainer = Trainer(
        module_file=os.path.abspath(config['training_module']),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config["training_steps"]
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config["eval_steps"]
        )
    )

    # Resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # Evaluator
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Category')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                                  threshold=tfma.MetricThreshold(
                                      value_threshold=tfma.GenericValueThreshold(
                                          lower_bound={'value': 0.5}
                                      ),
                                      change_threshold=tfma.GenericChangeThreshold(
                                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                          absolute={'value': 0.0001}
                                      )
                                  )
                                  )
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # Pusher
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config['serving_model_dir']
            )
        )
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    return components
