from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    @staticmethod
    def _validate_image_transformation(data_generator, img_idx):

        # Get the list of class names (categories)
        class_names = list(data_generator.class_indices.keys())      

        # Display the first batch of images and their labels
        images, labels = next(data_generator)

        # Convert one-hot encoded labels to class names
        class_labels = [class_names[idx] for idx in tf.argmax(labels, axis=1)]
        print(f'LOG: df size: {len(data_generator.filenames)}')

        # Visualize the first image and its label
        plt.imshow(images[img_idx])
        plt.title(f'Class: {format(class_labels[img_idx])} | Label: {format(labels[img_idx])}')
        plt.axis('off')
        plt.show()

    @staticmethod
    def save_model(path:Path, model: tf.keras.Model):
        model.save(path)    
    
    def get_base_model(self):
        """loading the base model done before | 2:16 min"""

        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        print()
    
    def train_valid_generator(self):
        """Create the tranin and test data split if is not done before"""

        datagenerator_kwargs = dict(rescale = 1./225, 
                                    #validation_split = 0.20
                                    )

        dataflow_kwargs = dict(target_size = self.config.params_image_size[:-1],
                               batch_size = self.config.params_batch_size,
                               interpolation = 'bilinear',
                               class_mode='categorical',
                              ) 
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(directory=self.config.validation_data,
                                                                       #subset='validation',
                                                                       shuffle=False, # Set to False to ensure the order of samples is preserved
                                                                       **dataflow_kwargs)
        # Display the first batch of images and their labels
        # Visualize the first image and its label
        #self._validate_image_transformation(data_generator=self.valid_generator, img_idx=0)
        
        
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
        
        self.train_generator = train_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            #subset='training',
            shuffle=True,
            **dataflow_kwargs
        )
        # Display the first batch of images and their labels
        # Visualize the first image and its label
        #self._validate_image_transformation(data_generator=self.train_generator, img_idx=0)


    def train(self, callback_list:list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epochs,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
            validation_data = self.valid_generator,
            callbacks = callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )