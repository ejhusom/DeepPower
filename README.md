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
4. **Combine**: Combine data from multiple workouts.
5. **Split**: Split data set into training and test set.
6. **Scale**: Scale input data.


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


###
