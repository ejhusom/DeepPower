# DeepPower


## Dependencies

- Python 3.8
    - pandas
    - numpy 
    - matplotlib
    - scipy
    - scikit-learn
    - Tensorflow 2.0


## Experiment pipeline

Stages:

1. **Restructure**: Restructure raw data into dataframes.
2. **Featurize**: Add features to data set.
3. **Sequentialize**: Split data into input/output sequences.
4. **Split**: Split data set into training and test set.
5. **Scale**: Scale input data.
6. **Train**: Train model.
7. **Evaluate**: Evaluate model.

### Restructure

Restructure raw data into dataframes.

### Featurize

This stage consists of three steps:

1. Remove data that should not be a part of the model:
    - Time (the time of the workout do not matter).
    - Calories. Currently the calories are not a part of the model, but might be
      interesting as target values later on.
2. Feature engineering. Examples:
    - Rolling minimum.
    - Rolling maximum.
    - Rolling range.
3. Delete features that may be less important, since we prefer a simpler model.
   This might for example be raw breathing data, if we have engineered features
   that work better.

### Sequentialize

In this stage the data is divided into sequences based on a chosen history
size.

### Split

This stage first combines the data from all workouts in the data set, and then
splits the data set into a training and test set.

### Scale

In this stage the data is scaled.

## Usage

### Run experiment

All stages are defined in the file `dvc.yaml`, and the parameters to be used
are saved in `params.yaml`.

To reproduce an experiment with any given parameters specified in
`params.yaml`, run:

```
dvc repro
```

### Evaluate

To evaluate model, run:

```
python3 src/evaluate.py
```

### Change dataset

To run experiments with another dataset, just change the content of
`assets/data/raw/` to the files you want to use.


### Visualize data set

Data set can be visualized in to stages
