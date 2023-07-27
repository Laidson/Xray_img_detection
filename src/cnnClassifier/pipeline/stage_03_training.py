from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallBack
from cnnClassifier.components.trainig import Training
from cnnClassifier import logger

STAGE_NAME = 'TRAINING'

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallBack(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(
            callback_list=callback_list
        )
    
if __name__ == '__main__':
    try:
        logger.info(15 * '*')
        logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
    except Exception as e:
        logger.info(f'ERROR >>> {e}')
        raise e