from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainigPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainigPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

import subprocess

logger.info('>>>>>>>> Start Model Pipeline <<<<<<<')

# STEP 01
STAGE_NAME = 'Data ingestion stage'
try:
    logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    # call data injestion
    # obj = DataIngestionTrainigPipeline()
    # obj.main()
    logger.info(f'>>>>>>>>>> {STAGE_NAME} complete <<<<<<<<<<\n\nx===========x')
except Exception as e:
    logger.exception(e)
    raise e

# STEP 02
STAGE_NAME = 'PREPARE BASE MODEL'
try:
    logger.info(15 * '*')
    logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
    obj = PrepareBaseModelTrainigPipeline()
    obj.main()
    logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
except Exception as e:
    logger.info(f'ERROR >>> {e}')
    raise e

# STEP 03
STAGE_NAME = 'TRAINING'
try:
    logger.info(15 * '*')
    logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
except Exception as e:
    logger.info(f'ERROR >>> {e}')
    raise e

# STEP 04 
STAGE_NAME = 'EVALUATION STAGE'
try:
    logger.info(15 * '*')
    logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
except Exception as e:
    logger.info(f'ERROR >>> {e}')
    raise e