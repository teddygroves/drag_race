import arviz
import pandas as pd
import numpy as np
import pystan
import pickle
from hashlib import md5

PLACE_MAP = {
    'SAFE': 2,
    'WIN': 1,
    'LOW': 2,
    'HIGH': 2,
    'BTM2': 3,
    'ELIM': 3,
    'Winner': 1,
    'Runner-up': 2,
    'Eliminated': 3,
    'Guest': 2,
    'Miss C': 2,
    'DISQ': 2,
    'OUT': 2
}
ELIMINATED = ['Eliminated', 'DISQ', 'OUT', 'ELIM']
PREDICTORS = ['age_z', 'is_ny']
Z_COLS = ['age']


def z_score(s):
    return (s - s.mean()) / s.std()


def prepare_data():
    episodes = (
        pd.read_csv('data/all_episodes.csv')
        .drop('Unnamed: 0', axis=1)
        .assign(episode_id=lambda df: (
            df[['season_number', 'episode_number']]
            .astype(str)
            .apply('-'.join, axis=1)
            .factorize()[0] + 1
            )
        )
        .set_index(['season_number', 'episode_number'])
    )
    rankings = (
        pd.read_csv('data/all_rankings.csv')
        .drop('Unnamed: 0', axis=1)
        .assign(rank=lambda df: df['episode_placement'].map(PLACE_MAP))
        .join(episodes[['episode_type', 'episode_id']],
              on=['season_number', 'episode_number'])
        .sort_values(['season_number', 'episode_number', 'rank'])
    )

    double_winner = (rankings
                     .groupby('episode_id')
                     ['rank']
                     .apply(lambda s: s.eq(1).sum() == 2)
                     .astype(int)
                     .rename('double_winner'))
    rankings = rankings.join(double_winner, on='episode_id')
    season_starts = (
        episodes.groupby('season_number')['episode_airdate'].min()
        .rename('season_start')
    )
    contestants = (
        pd.read_csv('data/all_contestants.csv')
        .drop('Unnamed: 0', axis=1)
        .set_index('contestant_id')
        .join(season_starts, on='season_number')
    )
    social_media = (
        pd.read_csv('data/all_social_media.csv', skiprows=[1])
        .drop('Unnamed: 0', axis=1)
        .join(contestants['season_start'], on='contestant_id')
        .loc[lambda df: (pd.to_datetime(df['datetime'])
                         - pd.to_datetime(df['season_start'])).dt.days <= 0]
        .assign(latest=lambda df: df.groupby('contestant_id')['datetime'].transform('max'))
        .loc[lambda df: df['datetime'] == df['latest']]
        .set_index('contestant_id')
    )

    # some extra columns
    for col in Z_COLS:
        contestants[col + '_z'] = z_score(contestants[col])
    contestants['is_ny'] = (contestants['hometown_city']
                            .isin(['New York', 'Brooklyn'])
                            .astype(int))

    # filtering
    rankings = rankings.loc[lambda df: df['episode_type'] == 'Competition']

    # get contestants for next episode
    latest_episode = episodes.sort_values('episode_airdate')['episode_id'].iloc[-1]
    contestants_next = rankings.loc[lambda df: (
        (df['episode_id'] == latest_episode)
        & (~df['episode_placement'].isin(ELIMINATED))
    ), 'contestant_id'].values

    return contestants, rankings, contestants_next


def get_stan_input(contestants, rankings, contestants_next):
    return {
        'N': len(rankings),
        'K': len(PREDICTORS),
        'N_episode': len(rankings['episode_id'].unique()),
        'N_contestant': len(contestants),
        'N_contestant_next': len(contestants_next),
        'contestant': rankings['contestant_id'],
        'contestant_next': contestants_next,
        'N_episode_contestant': rankings.groupby('episode_id').size(),
        'double_winner': rankings.groupby('episode_id')['double_winner'].first(),
        'X': contestants[PREDICTORS]
    }


def StanModel_cache(file, model_name=None, **kwargs):
    with open(file, 'r') as f:
        model_code = f.read()
        f.close()
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'data/cached_stan_models/{}.pkl'.format(code_hash)
    else:
        cache_fn = 'data/cached_stan_models/{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm


def probabilise(df):
    return df.sum().div(len(df))


def run_model():
    contestants, rankings, contestants_next = prepare_data()
    model = StanModel_cache(file='model.stan')
    model_input = get_stan_input(contestants, rankings, contestants_next)
    fit = model.sampling(data=model_input)
    infd = arviz.from_pystan(fit,
                             coords={'contestant': contestants.index,
                                     'predictor': PREDICTORS},
                             dims={'ability_maxi': ['contestant'],
                                   'ability_lipsync': ['contestant'],
                                   'beta_lipsync': ['predictor'],
                                   'beta_maxi': ['predictor'],
                                   'is_eliminated': ['contestant']})
    ability_maxi = infd.posterior['ability_maxi'].to_series().unstack()
    ability_lipsync = infd.posterior['ability_lipsync'].to_series().unstack()

    elim_prob = (
        infd.posterior['is_eliminated']
        .to_series()
        .unstack()
        .pipe(probabilise)
        .rename('elim_prob')
    )
    
    contestants['maxi_mean'] = ability_maxi.mean()
    contestants['maxi_sd'] = ability_maxi.std()
    contestants['lipsync_mean'] = ability_lipsync.mean()
    contestants['lipsync_sd'] = ability_lipsync.std()
    contestants = contestants.join(elim_prob)
    infd.to_netcdf('data/output.nd')
    contestants.to_csv('data/output_contestants.csv')
    print(contestants.loc[contestants_next]
          .set_index('contestant_name')
          .sort_values('elim_prob')
          [['maxi_mean', 'maxi_sd', 'lipsync_mean', 'lipsync_sd', 'elim_prob']])

if __name__ == '__main__':
    run_model()
