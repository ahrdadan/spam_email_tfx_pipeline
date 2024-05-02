"""
Author: Ahmad Ramadhan
Date: 2024-05-02
This is the local-pipeline.py module.
Usage:
- For running Components module with pipeline orchestrator: Apache Beam.
"""
import os
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


MODEL_NAME = 'spam_email_detection'
PIPELINE_NAME = 'dhadhan-pipeline'

DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/transform.py'
TRAINER_MODULE_FILE = 'modules/train.py'
TUNER_MODULE_FILE = 'modules/tuner.py'

OUTPUT_BASE = 'output'
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')


def init_local_pipeline(
    components_tf, pipeline_root_dir: Text
) -> pipeline.Pipeline:
    """
    Initialize a local TFX pipeline.

    Args:
        components_tf: A dictionary of TFX components to be included in the pipeline.
        pipeline_root_dir: Root directory for pipeline output artifacts.

    Returns:
        A TFX pipeline.
    """
    logging.info(f'Pipeline root set to: {pipeline_root_dir}')
    beam_args = [
        '--direct_running_mode=multi_processing'
        # 0 auto-detect based on the number of CPUs available
        # duraing execution time
        '----direct_num_workers=0'
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root_dir,
        components=components_tf,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    config = {
        "data_dir": DATA_ROOT,
        "training_steps": 1000,
        "eval_steps": 500,
        "serving_model_dir": serving_model_dir,
        "training_module": TRAINER_MODULE_FILE,
        "transform_module": TRANSFORM_MODULE_FILE,
        "tuner_module": TUNER_MODULE_FILE
    }

    from modules.components import init_components
    components = init_components(config)
    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
