# DeepPower


## Dependencies

- Python 3.8
    - matplotlib
    - numpy 
    - pandas
    - plotly
    - scikit-learn
    - Tensorflow 2.0
    - pyyaml
    - scipy
    - dvc


## Experiment pipeline

Stages:

1. **Restructure**: Restructure raw data into dataframes.
2. **Featurize**: Add features to data set.
4. **Split**: Split data set into training and test data.
5. **Scale**: Scale input data.
3. **Sequentialize**: Split data into input/output sequences.
6. **Train**: Train model.
7. **Evaluate**: Evaluate model.

### Restructure

Restructure raw data into dataframes.

### Featurize

This stage consists of these steps:

1. Remove data that should not be a part of the model:
    - Time (the time of the workout do not matter).
    - Calories. Currently the calories are not a part of the model, but might be
      interesting as target values later on.
2. Optionally scale input features.
3. Feature engineering. Examples:
    - Rolling range of breathing data.
    - Gradient of breathing data.
    - Slope angle of the breathing pattern.
4. Optionally delete raw inputs that may be less important, since we prefer a
   simpler model.  This might for example be raw breathing data, if we have
   engineered features that work better.

### Split

This stage splits the data set into a training and test set.

### Scale

In this stage the data is scaled.

### Sequentialize

In this stage the data is divided into sequences based on a chosen history
size.


## Usage

### Run experiment

All stages are defined in the file `dvc.yaml`, and the parameters to be used
are saved in `params.yaml`.

To run/reproduce an experiment with any given parameters specified in
`params.yaml`, run:

```
dvc repro
```

### Evaluate

If a model already exists and you want to test it on a test set, run:

```
python3 src/evaluate.py
```

This requires that the test data already is present in the correct folder.
Because of this, it is usually better to use the command `dvc repro` when
evaluating models.

### Check previous experiments

N/A


### Change dataset

To run experiments with another dataset, just change the content of
`assets/data/raw/` to the files you want to use.


### Visualize data set

Data set can be visualized by running

```
python3 visualize.py
```

