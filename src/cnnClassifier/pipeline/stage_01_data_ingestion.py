
from cnnClassifier import logger

STAGE_NAME = 'Data ingestion stage'

class DataIngestionTrainigPipeline:

    def __init__(self) -> None:
        pass

    def main(self):
        #DATA INGESTION STEPS
        # insert the data ingestion steps from data_ingestion on components
        # connect
        # query
        # save

        pass

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
        # call data injestion
        # obj = DataIngestionTrainigPipeline()
        # obj.main()
        logger.info(f'>>>>>>>>>> {STAGE_NAME} complete <<<<<<<<<<\n\nx===========x')
    except Exception as e:
        logger.exception(e)
        raise e
      