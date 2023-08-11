# Xray_img_detection
Pneumunia Xray image detection

## How to Run?
### STEPS:

Clone the repository

``` https://github.com/Laidson/Xray_img_detection.git ```

### STEP 01- Create a conda environment after opening the repository

``` conda create -n cnncls python=3.8 -y ```

``` conda activate cnncls ```

### STEP 02- install the requirements 

``` pip install -r requirements.txt ```

``` # Finally run the following 
    command python app.py```

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
