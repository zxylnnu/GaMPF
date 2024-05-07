# GaMPF
Code for paper 'GaMPF: A Full-Scale Gated Message Passing Framework Based on Collaborative Estimation for VHR Remote Sensing Image Change Detection'

## Train
1. Use following command for training

    ```python train.py --datadir (dataset_path) --checkpointdir (checkpoint_path)```
2. The trained models will be saved at checkpoint_path.

## Test

1. According to the trained models, use following command for testing

   ```python eval.py --datadir (testdata_path) --checkpointdir (model_path) --resultdir (result_path) --store-imgs```

2. The predictions will be saved at result_path.

3. If you need some of the predictions, please contact me.


## Acknowledgements
Thank related open source projects for supporting this repository.
