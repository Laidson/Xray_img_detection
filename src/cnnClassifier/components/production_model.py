import os
import mlflow
import pickle
from pathlib import Path
from dotenv import load_dotenv

from cnnClassifier.entity.config_entity import ModelProdConfig

class Production:
    def __init__(self, config:ModelProdConfig):
        self.config = config
        pass

    @staticmethod
    def save_model(path:Path, model: mlflow.pyfunc.PyFuncModel):
        model_path = os.path.join(path, 'model.pkl')
        with open(model_path, 'wb') as f:
             pickle.dump(model, f)   

    def get_mlflow_model(self):
        load_dotenv()
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

        model_name = "XRAY"  # Name of the registered model
        stage = 'Production'

        model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_name}/{stage}')

        self.save_model(path=self.config.prod_model_path, model=model)