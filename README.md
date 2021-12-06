# Exploring Neural Networks Approximation of Hierarchically Compositional Functions Through Random Generated Trees

## Prerequisites
* Python >= 3.5

## Installation
```sh
pip install -r requirements.txt
```

## Generate Data
The `dataset.py` generates a dataset `DATSET_NAME` with the following arguments:
* NUM_FEATURES : the dimension of the input data
* DENSITY: the expected factor in which the tree shrinks
* NUM_EXAMPLES: the number of total examples in the dataset. by default the training ratio is 0.9


```sh
python dataset.py DATASET_NAME -f NUM_FEATURES -d DENSITY -n NUM_EXAMPLES 
```

## Train Algorithms
After generating the dataset `DATASET_NAME` we run an experiment using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) by running:
```sh
python train.py DATASET_NAME
```

## Results
Results should be automatically accessible through [TensorBoard](https://github.com/tensorflow/tensorboard) by running:
```
tensorboard --logdir lightning_logs
```
TensorBoard should now run on [http://localhost:6006](http://localhost:6006)
