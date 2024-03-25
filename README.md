# ncaa_tourney_predictions
Machine Learning code to predict the NCAA college bball tourney using (primarily) the KenPom (kenpom.com) college basketball ratings.


## How it works

Using NCAA Men's College Basketball Tournament game-by-game results dating back to 2011 (available on Kaggle) and pre-tournament KenPom team-by-team ratings, this project builds machine learning models to predict the result of every game in the current tournament.

## To run:

First, run the KenPom scraper (`source/kenpom_scraper.py`). You can update the `base_urls` in the code itself to compile data from different seasons. To run:

```
> python source/kenpom_scraper.py
```

This saves all of the team-by-team pre-tournament KenPom metrics in `data/kenpom_data.csv`. From here, we can run the pipeline code to build machine learning models (using `lightgbm`) and use them for prediction on new games.

For instance, to build a model that uses the KenPom offensive and defensive metrics of the two competing teams (and their differences) to predict a winner, we would use a config like:

```
config = {
    'build_dir': '_build/',
    'target': 'Winner',
    'features':  ['kenpom_adj_o_team1',
                  'kenpom_adj_o_team2',
                  'kenpom_adj_o_diff',
                  'kenpom_adj_d_team1',
                  'kenpom_adj_d_team2',
                  'kenpom_adj_d_diff'],
    'model_params': {
        'categorical_features': None,
        'n_estimators': 250,  # Number of trees to fit.
        'early_stopping_rounds':20,
        'eval_metric': 'binary_logloss',
        'min_data_per_group': 3,
        'learning_rate': 0.01,
        'num_leaves': 10
    }
}
```

And then to run the model build:

```
from source import pipeline
pipeline.build(config)
```

To then use that model to predict the first round of the 2024 Tournament [TODO: Move this code to `pipeline`!]:

```
from source import features

df_seeds = pd.read_csv('data/MNCAATourneySeeds.csv')
df_seeds = df_seeds.loc[df_seeds['Season']==2024]
regions=['W', 'X', 'Y', 'Z']

team1_list = []
seed1_list = []
team2_list = []
seed2_list = []

# Compile the games for round 1 of the tournament:
for reg in regions:
    for seed1 in range(1,9):
        seed1_list.append(seed1)
        seed1_str = f'0{seed1}'
        team1_list.append(df_seeds.loc[df_seeds['Seed']==f'{reg}{seed1_str}']['TeamID'].iloc[0])
        
        seed2 = 17-seed1
        seed2_list.append(seed2)
        seed2_str = f'0{seed2}' if seed2==9 else str(seed2)
        reg_seed_str = f'{reg}{seed2_str}'
        if (reg_seed_str == 'X16') or (reg_seed_str == 'Y16') or (reg_seed_str == 'Y10') or (reg_seed_str == 'Z10'):
            reg_seed_str += 'a'
        team2_list.append(df_seeds.loc[df_seeds['Seed']==reg_seed_str]['TeamID'].iloc[0])

df_pred = pd.DataFrame({'TeamID1': team1_list, 'seed_num_team_1': seed1_list,
                       'TeamID2': team2_list, 'seed_num_team_2': seed2_list})
df_pred_input = df_pred.copy(deep=True)
df_pred['Season'] = 2024
df_pred['Winner'] = None  # Placeholder variable

# Add kenpom features to master df
features.add_kenpom_features(df_pred)

# Run predictions
preds = pipeline.predict(df_pred, config)
```

To see this in action, check out the Notebook `notebooks/NCAA Tournament Modelling.ipynb`.
