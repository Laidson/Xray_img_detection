from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainigPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainigPipeline

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