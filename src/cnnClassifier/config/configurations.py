import os 
from pathlib import Path

from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories

#import the ENtity classes
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig, 
                                                PrepareCallBacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig,
                                                PredictConfig,
                                                ModelProdConfig,)


class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallBacksConfig:
        config = self.config.prepare_callbacks

        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([Path(model_ckpt_dir), Path(config.tensorboard_root_log_dir)])

        prepare_callback_config = PrepareCallBacksConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.data_ingestion
        validation_data = self.config.data_ingestion
        create_directories([training.root_dir])

        training_config = TrainingConfig(
            root_dir = Path(training.root_dir)
            ,trained_model_path = Path(training.trained_model_path)
            ,updated_base_model_path = Path(prepare_base_model.updated_base_model_path)
            ,training_data = Path(training_data.train_dir)
            ,validation_data = Path(validation_data.val_dir)
            ,params_epochs = params.EPOCHS
            ,params_batch_size = params.BATCH_SIZE
            ,params_is_augmentation = params.AUGMENTATION
            ,params_image_size = params.IMAGE_SIZE
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        config = self.config
        eval_config = EvaluationConfig(
            path_of_model = config.training.trained_model_path,
            training_data = config.data_ingestion.train_dir,
            validation_data = config.data_ingestion.test_dir,
            all_params = self.params,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
                       )
        return eval_config
    
    def get_predict_config(self) -> PredictConfig:
        config = self.config
        predict_config = PredictConfig(
            path_of_model = config.training.trained_model_path,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
        )
        return predict_config
    
    def get_production_model(self) -> ModelProdConfig:
        config = self.config.production        
        create_directories([config.root_dir])

        prod_model = ModelProdConfig(
            path_to_model = config.root_dir,
            prod_model_path = config.prod_model_path,
        )
        return prod_model