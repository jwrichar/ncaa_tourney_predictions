import os
import random
import re

import pandas as pd

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(FILEPATH, '../data/'))


def compile_training_data(start_year=2011, end_year=2023):

    df = load_game_data(start_year=start_year, end_year=end_year)

    df = add_seeding(df)

    compute_outcome(df)

    return df


def load_game_data(start_year=2011, end_year=2023):
    '''
    Get all data and result for all NCAA tournament games from a range
    of years.
    :param start_year: First year to select.
    :param end_year: Last year to select.

    '''
    df = pd.read_csv(os.path.join(DATA_DIR, 'MNCAATourneyCompactResults.csv'))
    df.drop(df[(df['Season'] < start_year) | (df['Season'] > end_year)].index,
            inplace=True)
    return df


def compute_outcome(df):
    '''
    From set of game data, compute the target variable(s). This requires:
    + Rearrange the team IDs into numeric order, not Winner/Loser
        - Put the better-seeded team as Team1 and worse-seeded as Team2
    + Compute boolean target about who won
    + Add margin of victory

    :param df: DataFrame as loaded by load_game_data

    :Returns: None, modify df in place
    '''
    # Add TeamID1 and TeamID2 to keep track of teams 1 and 2, where
    # Team 1 is the better-seeded team
    team1_list = []
    team2_list = []
    seed1_list = []
    seed2_list = []
    for index, row in df.iterrows():
        # Better-seeded team won:
        if (row['seed_num_team_W'] < row['seed_num_team_L']):
            team1_list.append(row['WTeamID'])
            seed1_list.append(row['seed_num_team_W'])
            team2_list.append(row['LTeamID'])
            seed2_list.append(row['seed_num_team_L'])
        # Upset happened:
        elif (row['seed_num_team_W'] > row['seed_num_team_L']):
            team1_list.append(row['LTeamID'])
            seed1_list.append(row['seed_num_team_L'])
            team2_list.append(row['WTeamID'])
            seed2_list.append(row['seed_num_team_W'])
        # Same seed - randomly assign:
        else:
            options = ['W', 'L']
            team1 = random.choice(options)
            options.remove(team1)
            team2 = options[0]
            team1_list.append(row[f'{team1}TeamID'])
            seed1_list.append(row[f'seed_num_team_{team1}'])
            team2_list.append(row[f'{team2}TeamID'])
            seed2_list.append(row[f'seed_num_team_{team2}'])

    df['TeamID1'] = team1_list
    df['seed_num_team_1'] = seed1_list
    df['TeamID2'] = team2_list
    df['seed_num_team_2'] = seed2_list

    # Add Winner target variable
    df['Winner'] = (df['WTeamID'] == df['TeamID1']).replace(
        {True: 'Team1', False: 'Team2'})

    # Add Margin target variable
    df['Margin'] = df['WScore'] - df['LScore']

    # Drop superfluous columns:
    df.drop(['DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT',
             'seed_num_team_W', 'seed_num_team_L'],
            axis=1, inplace=True)


def add_seeding(df):
    '''
    Add columns to df representing the tournament seedings
    for both team 1 and team 2.
    '''

    df_seeds = pd.read_csv(os.path.join(DATA_DIR, 'MNCAATourneySeeds.csv'))
    df_seeds['seed_number'] = df_seeds['Seed'].apply(
        lambda x: int(re.sub("[^0-9]", "", x)))
    df_seeds.drop('Seed', axis=1, inplace=True)

    # Add Team 1/2 seedings:
    for res in ['W', 'L']:
        seed_col = pd.merge(
            df, df_seeds,
            how='left',
            left_on=['Season', f'{res}TeamID'],
            right_on=['Season', 'TeamID'])['seed_number']
        df[f'seed_num_team_{res}'] = seed_col.values

    return df
