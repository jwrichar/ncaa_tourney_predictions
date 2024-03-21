import json
import os
import pickle

import numpy as np
import pandas as pd

import lightgbm


class BoostingModel:
    '''
    The BoostingModel object uses the lightgbm python package for learning and
    prediction.

    * lightgbm is a gradient boosting framework that uses tree based learning
    algorithms. It is designed to be faster with lower memory usage than
    other GBM implementations.

    This implementation uses the scikit-learn wrapper interface for lightgbm.
    https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
    '''

    def __init__(self):

        self.default_params = {
            'target': None,  # Model target var, needs to be specified
            'sample_weight': None,  # For train-time sample weighting
            'boosting_type': 'gbdt',  # ‘gbdt’, ‘dart’, 'rf'
            'num_leaves': 31,  # Maximum number of leaves for base learners
            'max_depth': -1,  # Maximum tree depth for base learners
            'learning_rate': 0.1,  # Boosting learning rate
            'n_estimators': 100,  # Number of trees to fit.
            'early_stopping_rounds': 0,  # Early stopping parameter
            'eval_metric': '',  # Eval metric for early stopping, e.g. 'l2'
            'categorical_features': "",  # Cat features list: "name:c1,c2,c3"
            'cat_l2': 10.0,  # L2 Regularization for categorical variables
            'cat_smooth': 10.0,  # L2 cat smoothing parameter
            'min_data_per_group': 10,  # Smallest category size
            'feature_fraction': 1,
            'max_delta_step': 0,
            'min_child_weight': 1,
            'min_child_samples': 10,
            'use_missing': True,  # LIGHTGBM ignores NAs during a split, then
                                  # allocates them to the side that mins loss
            'importance_type': 'gain',
            'num_threads': 0,
            'sparsify': False,  # Whether to use sparse features
            'verbose': 2
        }

    def build(self, df, custom_params):
        '''
        Boosting model build

        :param df: Input training set dataframe (features).  Note that
            all model inputs (x's) must be numerical (float or int) or
            categorical (in the case of lightgbm). The target (y)
            variable is allowed to be numerical.
        :param custom_params: Optional params to override model default_params
        :returns: None
        '''

        params = self._get_params(custom_params)

        self.model = lightgbm.LGBMClassifier(**params)

        if not params.get('target'):
            raise KeyError(
                'Parameter "target" must be specified in build params.')
        target = params['target']

        early_stopping_rounds = params['early_stopping_rounds']

        model_kwargs = {}

        x, y, w = self._prepare_data(df, target, params['sample_weight'])

        # Prepare categorical features for learning
        categorical_feat_locs = self._prepare_categorical(x, params, 'build')
        if categorical_feat_locs:
            model_kwargs['categorical_feature'] = categorical_feat_locs

        feature_names = x.columns

        if params['sparsify']:
            # SparseDataFrame must be applied before splitting operations
            # Note: float conversion is necessary to avoid mixed dtypes
            x = pd.SparseDataFrame(x.astype(float))

        # Create a evaluation set if early stopping is enabled
        if early_stopping_rounds:
            print('Creating evaluation set')
            x, y, w, x_eval, y_eval, w_eval = create_eval_set(x, y, w)

        if params['sparsify']:
            print('Converting features to scipy COO matrix for training.')
            x = x.to_coo()
            if early_stopping_rounds:
                x_eval = x_eval.to_coo()

        print('Training lightgbm model on %i x %i data set' %
              (len(y), len(feature_names)))

        if early_stopping_rounds:
            print('Early stopping in effect (rounds = %i)' %
                  (early_stopping_rounds))
            eval_set = [(x_eval, y_eval)]
            model_kwargs['eval_set'] = eval_set
            model_kwargs['early_stopping_rounds'] = early_stopping_rounds
            model_kwargs['eval_metric'] = params['eval_metric']

        self.model.fit(X=x,
                       y=y,
                       sample_weight=w,
                       **model_kwargs)

        print('Training complete')

        model_info = {
            'importances': dict(zip(
                feature_names, self.model.feature_importances_.tolist())),
            'params': params
        }
        with open(os.path.join(os.environ['BUILD_DIR'],
                               'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)

        self.save()

    def predict(self, df, custom_params):
        '''
        Boosting model predict

        :param df: Input prediction set dataframe (features)
        :param custom_params: Optional params to override model default_params
        :returns: numpy array of model predictions on df
        '''
        params = self._get_params(custom_params)

        target = params['target']
        if target in df.columns:
            df = df.drop(target, axis=1)

        # At predict time, remove any sample weighting columns
        if params['sample_weight']:
            if params['sample_weight'] in df.columns:
                df = df.drop(params['sample_weight'], axis=1)

        # Prepare categorical features for prediction:
        # Note: feature ordering is preserved between build and
        # predict by data.required_columns pre-processing
        self._prepare_categorical(df, params, 'predict')

        if params['sparsify']:
            print('Converting features to scipy COO matrix for prediction.')
            df = pd.SparseDataFrame(df.astype(float)).to_coo()

        out = self.model.predict_proba(df)

        scores = pd.DataFrame(out, columns=self.model.classes_)

        values = scores.idxmax(axis=1).values

        return {'values': values, 'scores': scores}

    def save(self):
        '''
        Save a boosting model to file
        '''
        print('Saving model to file')
        fname = self._get_file_name()
        with open(fname, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        '''
        Load a boosting model from file
        '''
        print('Loading model from file')
        fname = self._get_file_name()
        with open(fname, 'rb') as f:
            self.model = pickle.load(f)

    def get_nearest_training(self, df, train_leaf_indices, k, custom_params):
        '''
        Get the k nearest training examples to each instance in
        data set df based on the co-occurrence in leaves in the
        model.

        :param df: Input prediction set dataframe (features)
        :param train_leaf_indices: DataFrame of (pre-computed)
            training set leaf indices in the model.
        :param k: Number of nearest training examples to return
            for each row of data in df.
        :param custom_params: Optional params to override model default_params
        :returns: list of dicts. Each dict contains {index: score} where the
            index is the df index of the nearest training set examples (ordered
            by decreasing score) for a row in df and the score is the leaf node
            co-occurence score.
        '''

        # Prediction with pred_leaf=True returns numpy array of leaf indices
        # with one row per row in df.
        pred_leaf_indices = self.predict(df,
                                         custom_params,
                                         pred_leaf=True)

        return [_k_nearest(row, k, train_leaf_indices)
                for row in pred_leaf_indices]

    def _get_file_name(self):
        ''' Get the filename for the model file '''
        return os.path.join(os.environ['BUILD_DIR'],
                            'model.lgbm')

    def _prepare_data(self, df, target, weight_col):
        ''' Prepare data triplets (x, y, w) for training '''

        # Pull out target and drop from input covariates
        y = df[target]
        x = df.drop(target, axis=1)

        # If weights are enabled, pull them out separately
        w = None
        if weight_col:
            if weight_col not in x.columns:
                raise KeyError(('Weight column "%s" not found in input data. '
                                'Please check config file.') % weight_col)
            w = x[weight_col]
            x.drop(weight_col, axis=1, inplace=True)

        return x, y, w

    def _prepare_categorical(self, df, params, context):
        ''' Prepare categorical data for learning and prediction '''

        categorical_feat_locs = []  # Indices of categorical features
        if params['categorical_features']:
            print('Preparing categorical features: %s' %
                  ', '.join(params['categorical_features']))
            mappings = {}
            if context == 'predict':
                mappings = self._read_mapping_file()
            for feature in params['categorical_features']:
                if context == 'build':
                    mappings[feature] = self._compute_cat_mapping(df[feature])

                # Perform feature mapping
                df[feature] = self._apply_mapping(df[feature],
                                                  mappings[feature])

                # Keep track of column indices of categorical features
                categorical_feat_locs.append(df.columns.get_loc(feature))

            if context == 'build':
                self._write_mapping_file(mappings)

        return categorical_feat_locs

    def _compute_cat_mapping(self, col):
        ''' Compute categorical-to-int mapping for a column '''
        return dict(zip(col.unique(), range(col.nunique())))

    def _apply_mapping(self, col, mapping):
        ''' Apply a mapping dict to a categorical column '''
        # Apply mapping
        output = col.map(mapping)
        # Set NaNs to an int outside of range:
        nan_val = max(mapping.values()) + 1
        output.loc[output.isnull()] = nan_val

        return output

    def _write_mapping_file(self, mappings):
        fname = os.path.join(os.environ['BUILD_DIR'],
                             'categorical_mapping.json')
        with open(fname, 'w') as f:
            json.dump(mappings, f)

    def _read_mapping_file(self):
        fname = os.path.join(os.environ['BUILD_DIR'],
                             'categorical_mapping.json')
        with open(fname, 'r') as f:
            return json.load(f)

    def _get_params(self, custom_params):
        '''
        Get parameters for a model using config-level custom parameters,
        if set else defaulting to the self.default_params.
        '''
        out = {}
        for k, v in self.default_params.items():
            out[k] = custom_params.get(k, self.default_params[k])
        return out


def train_test(df, params, holdout_sort='random', frac=0.2, column=None):
    '''
    Run a train-test model validation using a BoostingModel
    :param df: input pandas DataFrame with features and target
    :param params: learning parameters for BoostingModel
    :param holdout_sort: how to perform train/test split (see split_data)
    :param frac: fraction of data to leave out
    :param column: (optional) column to sort on in holdout_sort='column'
    :returns: pandas Series of true y values for holdout set,
        numpy array of predicted values for holdout data
    '''

    split_params = {
        'sort': holdout_sort,
        'fraction': frac,
        'column': column

    }
    df_train, df_test = split_data(df, split_params)

    # Instantiate model:
    m = BoostingModel()

    # Train model:
    m.build(df_train, params)

    # Test predictions:
    y_pred = m.predict(df_test, params)

    y_true = df_test[params['target']]

    return y_true, y_pred


def k_fold_cv(df, params, k=10, verbose=True):
    '''
    Perform k-fold cross-validation using a BoostingModel.
    :param df: input pandas DataFrame with features and target
    :param params: learning parameters for BoostingModel
    :param k: int, number of folds
    :returns: Series of cross-validated predictions across entire df.
    '''

    n = len(df)
    folds = np.random.randint(0, k, size=n)

    # Instantiate model:
    m = BoostingModel()

    # Empty predictions Series to fill in each fold:
    y_pred = pd.Series(None, index=df.index, dtype=float)

    if verbose:
        print('Beginning %i-fold cross-validation' % k)
    for fold in np.arange(k):
        if verbose:
            print('Fold %i' % (fold + 1))
        df_train = df.loc[folds != fold, ]
        df_test = df.loc[folds == fold, ]

        # Train model:
        m.build(df_train, params)

        # Predict on test set:
        fold_preds = m.predict(df_test, params)

        y_pred.loc[df_test.index] = fold_preds

    if verbose:
        print('K-fold cross-validation complete.')

    return y_pred


def grid_search_kfold(df, params, grid, k=10, verbose=False):
    '''
    Perform a grid search over specified parameter values, returning
    a DataFrame of k-fold CV MSE & R-squared values for each
    choice of parameters.

    :param df: input pandas DataFrame with features and target.
    :param params: dict, learning parameters for BoostingModel (values of grid
        will overwrite any parameters in the params dict)
    :param grid: dict of lists. This is the grid to search over, where key is
        the parameter name and value is a list of values to search over.
    :param k: int, number of folds for cross-validation
    :returns: results, best_results. results is a dataframe of cross-validated
        MSE and R-squared values for each combo of parameters in the grid.
        best_results is the row containing the best result.
    '''
    # Create DataFrame to hold results:
    idx = pd.MultiIndex.from_product(grid.values(), names=grid.keys())
    results = pd.DataFrame(index=idx).reset_index()
    mse = []
    r2 = []

    y_true = df[params['target']]

    for i in results.index:
        test_param = results.astype(object).iloc[i].to_dict()
        print('Running values: %s' % test_param)

        # Update learning params with test params
        params.update(test_param)

        # Run k-fold CV:
        y_pred = k_fold_cv(df, params, k=k, verbose=verbose)

        # Save output metrics:
        mse.append(get_mse(y_true, y_pred))
        r2.append(get_r_squared(y_true, y_pred))

    results['mse'] = mse
    results['r2'] = r2

    best_result = results.loc[results['mse'].idxmin()]
    print("Best result:")
    print(best_result)

    return results, best_result


def get_mse(y_true, y_pred):
    ''' Compute MSE of regression predictions '''
    mse = np.mean((y_pred - y_true)**2)
    return mse


def get_r_squared(y_true, y_pred):
    ''' Compute r-squared of regression predictions '''
    tar_av = np.mean(y_true)
    r2 = 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - tar_av)**2)
    return r2


def score_transform(eff_ratios):
    '''
    Transform a series of Efficiency Ratios (actual EUI / predicted EUI)
    to 1 - 100 energy rating values based on the empirical CDF of the
    input efficiency ratios, and applying the transform:

        Score = 100 * (1 - ECDF(eff_ratio))

    so that large eff_ratios receive high scores and vice versa.

    :input eff_ratios: Series of cross-validated efficiency ratios.
    :return: Series of scores, with the same index as eff_ratios input.
    '''
    return eff_ratios.apply(
        lambda x: np.ceil(100. * (1. - (x > eff_ratios).mean())))


def split_data(df, split_params, verbose=True):
    '''
    Split a loaded pandas DataFrame into training and
    testing sets based on the recipe specified in a config
    :param df: Pandas DataFrame with the entire data set
    :param split_params: dict, validation data splitting params
    :param verbose: bool, verbosity level
    :returns: Pandas DataFrames of training and testing sets
    '''
    # only support holdout
    holdout_sort = split_params.get('sort', 'index')
    # index, column
    if holdout_sort == 'index':
        data_index = df.index
    elif holdout_sort == 'column':
        sorting_col = split_params['column']
        data_index = df[sorting_col].sort_values().index
    elif holdout_sort == 'random':
        data_index = df.sample(len(df)).index
    else:
        raise ValueError(
            'Mis-specified validation sorting: %s' % (holdout_sort))

    holdout_frac = split_params.get('fraction', 0.2)
    cutoff = int(len(df) * (1. - holdout_frac))

    if verbose:
        print(
            'Performing holdout split of %.2f fraction of data of type "%s".' %
            (holdout_frac, holdout_sort))

    df_train_idx = data_index[:cutoff]
    df_test_idx = data_index[cutoff:]
    return df.loc[df_train_idx], df.loc[df_test_idx]


def create_eval_set(x, y, w=None, holdout_sort='random', frac=0.2, column=None):
    '''
    Create training and evaluation data sets from an input data set
    consisting of features (DataFrame x) and target (Series y)
    :param x: pd DataFrame consisting of features
    :param y: pd Series with target variable
    :param w: pd Series with instance weights (optional)
    :param frac: float, splitting fraction
    :param column: (optional) column to sort on in holdout_sort='column'
    :returns: x_train, y_train, x_eval, y_eval, w_train, w_eval
    '''

    # Perform data splitting on y:
    eval_split_params = {
        'sort': holdout_sort,
        'fraction': frac,
        'column': column
    }
    y_train, y_eval = split_data(y, eval_split_params,
                                 verbose=False)

    idx_train = y_train.index.tolist()
    idx_eval = y_eval.index.tolist()

    # Compute outputs:
    x_train = x.loc[idx_train]
    x_eval = x.loc[idx_eval]
    y_train = y.loc[idx_train]
    y_eval = y.loc[idx_eval]
    if w is None:
        w_train, w_eval = None, None
    else:
        w_train = w.loc[idx_train]
        w_eval = w.loc[idx_eval]

    return x_train, y_train, w_train, x_eval, y_eval, w_eval


def _k_nearest(row_leaf_indices, k, train_leaf_indices):
    '''
    Returns indices of the k nearest training set examples to the
    given row of leaf indices based on the percentage of trees with
    leaf node co-occurrence.
    :param row_leaf_indices: numpy array of leaf indices for one new
        example.
    :param k: int, number of closest training examples to return.
    :param train_leaf_indices: Pandas DataFrame of all leaf indices
        for training data.
    :return: Dict of the highest k index->leaf co-occurence scores pairs.
    '''
    percent_leaf_coocurrence = train_leaf_indices.eq(
        row_leaf_indices, axis=1).mean(axis=1)
    return percent_leaf_coocurrence.nlargest(k).to_dict()
