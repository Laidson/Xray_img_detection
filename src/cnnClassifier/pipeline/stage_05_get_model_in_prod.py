import os
from pathlib import Path
import tensorflow as tf
import mlflow
from cnnClassifier import logger
from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.production_model import Production

STAGE_NAME = 'GET PRODUCTION MODEL'
class GetModelProdPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prod_config = config.get_production_model()
        get_prod_model = Production(prod_config)
        get_prod_model.get_mlflow_model()


if __name__ == '__main__':
    try:
        logger.info(15 * '*')
        logger.info(f'>>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = GetModelProdPipeline()
        obj.main()
        logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x')
    except Exception as e:
        logger.info(f'ERROR >>> {e}')
        raise e