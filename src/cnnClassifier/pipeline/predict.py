import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier.config.configurations import ConfigurationManager

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        config = ConfigurationManager()
        predict_configs = config.get_predict_config()

        model = load_model(os.path.join('artifacts', 'training', 'model.h5'))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(predict_configs.params_image_size[:-1]))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(model.predict(test_image))
        print(result)

        if result[0] == 0:
            prediction = 'HEALTH'
            return [{'image': prediction}]
        else:
            prediction = 'Pneumonia'
            return [{'image': prediction}]
