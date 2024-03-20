import os

import numpy as np
import pandas as pd

from source import data_utils, \
    features, \
    model
from source.model import BoostingModel


def build(config):
    '''
    Build an NCAA Tournament prediction model.
    Parameters controlling the build are in config.
    Once the build is run, the predict pipeline can be
    activated to generate predictions for new games.

    :param config: dict of paramters controlling the build (data, features,
        pre- and post-processing, etc.)
    '''
    os.environ['BUILD_DIR'] = config.get('build_dir', '_build/')
    if not os.path.isdir(os.environ['BUILD_DIR']):
        os.mkdir(os.environ['BUILD_DIR'])

    plot_dir = os.path.join(os.environ['BUILD_DIR'], 'plots/')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    target = config.get('target', 'Winner')
    feats = config['features']

    model_params = config.get('model_params', {})
    model_params['target'] = target

    # Load training data:
    start_year = config.get('start_year', 2011)
    end_year = config.get('end_year', 2023)
    df = data_utils.compile_training_data(start_year, end_year)

    # Add kenpom features to master df
    features.add_kenpom_features(df)

    # Training set dataframe:
    df_train = df[feats + [target]]
    df_train.to_csv(os.path.join(os.environ['BUILD_DIR'], 'features.csv'))

    # Train model on entire data set and save to build dir:
    m = BoostingModel()
    m.build(df_train, model_params)


def predict(df, config):
    '''
    Predict using a built model to produce a game prediction.

    :param df: Pandas DataFrame with input data to predict on.
    :param config: dict of paramters (e.g., where build is stored, params
        that control prediction return dict)
    :returns: list of dicts of prediction output, one per input df row.
    '''

    # Set build directory:
    os.environ['BUILD_DIR'] = config.get('build_dir', '_build/')

    print('Loading energy benchmarking model.')
    m = BoostingModel()
    m.load()

    # Get model parameters from config:
    target = config.get('target', 'Winner')
    model_params = config.get('model_params', {})
    model_params['target'] = target

    # Store target column, if available, to validate:
    obs_target = df[target]

    # Check features:
    _check_features(df, config['features'])

    print('Running predictions on %i games.' % len(df))
    pred = m.predict(df, model_params)
    pred_probs = pd.Series(pred, index=obs_target.index, name='predicted_eui')

    # Create df to store all predictions and output:
    df_out = pd.concat([obs_target, pred_target], axis=1)
    df_out['efficiency_ratio'] = eff_ratio

    # Set index as a column to preserve it in dict output:
    df_out.reset_index(drop=False, inplace=True)

    return df_out.to_dict(orient='records')


def _check_features(df, feature_list):
    '''
    Predict-time check that:
    (a) all required features are present in df, and
    (b) any extraneous features are dropped.
    '''
    input_feats = df.columns

    missing_feats = set(feature_list) - set(input_feats)
    if missing_feats:
        raise Exception(
            '%i Missing feature(s) found: %s' %
            (len(missing_feats), ','.join(missing_feats)))

    extra_feats = set(input_feats) - set(feature_list)
    if extra_feats:
        print('%i Extra feature(s) found: %s. Dropping.' %
              (len(extra_feats), ','.join(extra_feats)))
        df.drop(extra_feats, axis=1, inplace=True)
