from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel

from cnnClassifier import logger


STAGE_NAME = 'PREPARE BASE MODEL'

class PrepareBaseModelTrainigPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()



if __name__ == '__main__':
    try:
        logger.info(15 * '*')
        logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = PrepareBaseModelTrainigPipeline()
        obj.main()
        logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
    except Exception as e:
        logger.info(f'ERROR >>> {e}')
        raise e