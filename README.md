
# Syntax and Semantics ensemble model for Aspect-based Sentiment Analysis

## Directory structure:

```
|   README.md
|   requirement.txt
|
|--- dataset
|   |-- preprocessed_data
|   |-- raw_data
|
|--- model
|   |-- laptops
|   |-- restaurants
|
|--- ouput
|--- src
|   |-- config
|   |   bert_laptop_config.json
|   |   bert_restaurant_config.json
|   attention.py
|   dataset.py
|   inference.py
|   main.py
|   model.py
|   parser.py
|   tuner.py
|   utils.py
```

## Installation

### Libraries

To create all neccessary library, please run:

```bash
conda env create -f environment.yml
```

In case, the version of Pytorch and Cuda are not compatible on your machine, please remove all related lib in the `.yml` file; then install Pytorch , Pytorch Geometric separately.


### PyTorch
Please follow Pytorch installation instruction in this [link](https://pytorch.org/get-started/locally/).

### Torch Geometric
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` is version of Pytorch and Cuda.


### HuggingFace Transformers

```bash
conda install -c huggingface transformers
```

## Model Architecture

![Model architecture](/figure/model_architecture.png)

### Data preparation
Data preparation consists of three steps: parse XML, singularize data to (text, term, label), parse dependency tree.

To prepare data, run:

```bash
python src/parser.py
```

### Fine tuning
Change the variable `MODEL_NAME = 'bert'/'syn'/'sem'`, then run:

```bash
python src/tuner.py -e experiment_name
```

The experiments will be saved in directory: `experiment/${experiment_name}/${MODEL_NAME}`

To track experiment results: 
```bash
cd experiment/${experiment_name}/${MODEL_NAME}
tensorboard dev upload --logdir './'
```
### Training and Testing

After having the best hyperparameters, edit config files in `./src/config/`

To run training and testing, run:

```bash
sh script/train_test.sh
```

### Ensemble
Ensembling consists of two steps: 

1. Infer the prediction from the base models
- Edit variable `ckpt_filename` in `./src/inference.py` with the path of trained base model
- Edit variable `save_path` in `./src/inference.py` with the path to save the prediction of base model
- Run: 
```bash 
python src/inference.py
```

2. Learn an ensemble model: notebook `experiment/4_ensemble.ipynb`


