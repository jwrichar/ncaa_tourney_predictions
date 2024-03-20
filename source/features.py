import os

import pandas as pd

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(FILEPATH, '../data/'))


def add_kenpom_features(df):
    '''
    Add KenPom features to games using the data compiled by
    kenpom_scraper.py. Note: the scraping code must be run
    prior to compiling the features.

    Modifies df in place.
    '''

    df_kenpom = pd.read_csv(
        os.path.join(DATA_DIR, '../data/kenpom_data.csv'))
    # Drop entries with missing team IDs
    df_kenpom.dropna(subset=['TeamID'], inplace=True)
    # ID should be an int:
    df_kenpom['TeamID'] = df_kenpom['TeamID'].astype(int)

    kenpom_team1 = pd.merge(
        df, df_kenpom,
        how='left',
        left_on=['Season', 'TeamID1'],
        right_on=['Season', 'TeamID'])

    kenpom_team2 = pd.merge(
        df, df_kenpom,
        how='left',
        left_on=['Season', 'TeamID2'],
        right_on=['Season', 'TeamID'])

    # Add KenPom features (and diff features)
    for metric in ["kenpom_adj_o", "kenpom_adj_o_rank",
                   "kenpom_adj_d", "kenpom_adj_d_rank",
                   "kenpom_rank", "kenpom_luck", "kenpom_adjem"
                   ]:

        df[f'{metric}_team1'] = kenpom_team1[metric].values
        df[f'{metric}_team2'] = kenpom_team2[metric].values
        df[f'{metric}_diff'] = (
            kenpom_team1[metric] - kenpom_team2[metric]).values
