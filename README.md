# Xray_img_detection
Pneumunia Xray image detection

This project focuses on building a machine learning model to classify pulmonary X-ray images as either "healthy" or "sick." 
Given a pulmonary X-ray image, the goal is to predict whether the image indicates a healthy condition or a potential health issue.

## Project Goal

The main objective of this project is to create a functional X-ray image classification system. We emphasize that while model precision is important, the primary focus is on understanding the end-to-end implementation, handling data, training models, and making predictions.

## Dataset

We use a dataset of labeled pulmonary X-ray images, comprising both healthy and sick cases. The dataset has been preprocessed and split into training and testing subsets.

## Project Structure

The project is organized as follows:
- **artifacts/chest_xray**: Contains the training and testing datasets.
- **artifacts/training**: Stores trained machine learning models. Tacking using MLFLOW
- **src/cnnClassifier/pipeline**: Source code for model training, evaluation, and prediction.
- **inputImage.jpg**: Example X-ray image for demonstration purposes.

## How to Run?
### STEPS:

Clone the repository

``` https://github.com/Laidson/Xray_img_detection.git ```

### STEP 01- Create a conda environment after opening the repository

```conda create -n cnncls python=3.8 -y ```

```conda activate cnncls ```

### STEP 02- install the requirements 

```pip install -r requirements.txt ```

Finally run the following

```command python app.py```

```open up you local host and port```

## Workflow

1.Update config.yaml

2.Update secrets.yaml [Optional]

3.Update params.yaml

4.Update the entity - src\cnnClassifier\entity

5.Update the configuration manager in src config - src\cnnClassifier\config\configurations.py

6.Update the components - src\cnnClassifier\components

7.Update the pipeline - src\cnnClassifier\pipeline

8.Update the main.py

9.Update the dvc.yaml

## Tensor LOG acess
### on terminal
tensorboard --logdir artifacts/prepare_callbacks/tensorboard_log_dir/
