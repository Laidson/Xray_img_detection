from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainigPipeline

logger.info('Start LOG custom')


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