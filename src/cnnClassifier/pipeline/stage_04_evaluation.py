from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger


STAGE_NAME = 'EVALUATION STAGE'
class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == '__main__':
    try:
        logger.info(15 * '*')
        logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
    except Exception as e:
        logger.info(f'ERROR >>> {e}')
        raise e