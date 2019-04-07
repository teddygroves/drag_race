import arviz
import data_prep
from hashlib import md5
import numpy as np
import pandas as pd
import pickle
import pystan
from functools import reduce

PREDICTORS = ['age_z', 'twitter_rank']
PREPARED_DATA_SAVE_FILE = 'data/rankings_prepared.csv'
POSTERIOR_SAVE_FILE = 'data/output_posterior.nd'
CONTESTANT_SAVE_FILE = 'data/output_contestants.csv'
STAN_FILE = 'model.stan'


def proportionise(s):
    return s.div(s.sum())


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



def run_model(rankings):
    n_episode_contestant = rankings.groupby('episode_id')['contestant_id'].nunique()
    episode_rank_counts = (rankings
                           .groupby(['episode_id', 'rank'])
                           .size()
                           .unstack()
                           .fillna(0)
                           .astype(int))
    contestants = rankings.groupby('contestant_id').first()
    con_to_stan = dict(zip(
        rankings['contestant_id'].unique(),
        range(1, len(rankings['contestant_id'].unique()) + 1)
    ))
    input_data = {
        'N': len(rankings),
        'K': len(PREDICTORS),
        'C': rankings['contestant_id'].nunique(),
        'E': rankings['episode_id'].nunique(),
        'X': contestants[PREDICTORS],
        'N_episode_contestant': n_episode_contestant,
        'N_episode_winner': episode_rank_counts[1],
        'N_episode_safe': episode_rank_counts[[2, 3, 4]].sum(axis=1),
        'N_episode_bottom': episode_rank_counts[5],
        'contestant': rankings['contestant_id'].map(con_to_stan)
    }
    model = StanModel_cache(file=STAN_FILE)
    fit = model.sampling(data=input_data)
    return arviz.from_pystan(fit,
                             coords={'contestant': list(con_to_stan.keys()),
                                     'predictor': PREDICTORS},
                             dims={'ability': ['contestant'],
                                   'beta': ['predictor']})


def is_next_episode(df):
    return (df['episode_id'] == df['episode_id'].max()) & (df['eliminated'] == False)


def get_predictions(infd, rankings):
    next_episode_contestants = rankings.loc[is_next_episode, 'contestant_id'].tolist()
    next_abilities = infd.posterior['ability'].sel(contestant=next_episode_contestants)
    best_probs, worst_probs = (
        next_abilities
        .where(next_abilities==compare)
        .to_series()
        .dropna()
        .reset_index()
        ['contestant']
        .value_counts()
        .pipe(proportionise)
        .rename(colname)
        for compare, colname in [
            (next_abilities.max(dim='contestant'), 'prob_best'),
            (next_abilities.min(dim='contestant'), 'prob_worst')
        ]
    )
    ability_means = infd.posterior['ability'].mean(dim=['chain', 'draw']).to_series().rename('ability_mean')
    ability_sds = infd.posterior['ability'].std(dim=['chain', 'draw']).to_series().rename('ability_sd')
    names = rankings.groupby('contestant_id')['contestant_name'].first()
    outputs = [names, ability_means, ability_sds, best_probs, worst_probs]
    return reduce(lambda a, b: pd.DataFrame(a).join(b), outputs)
    
    
if __name__ == '__main__':

    print('Preparing data...')
    rankings = data_prep.prepare_data()

    print(f'Saving prepared data to {PREPARED_DATA_SAVE_FILE}...')
    rankings.to_csv(PREPARED_DATA_SAVE_FILE)

    print('Running model...')
    infd = run_model(rankings)

    print('Tidying up output...')
    contestants = get_predictions(infd, rankings)

    print('Next episode predictions:')
    print(contestants.dropna(how='all', subset=['prob_best', 'prob_worst']))

    print(f'Saving poterior to {POSTERIOR_SAVE_FILE}...')
    infd.to_netcdf(POSTERIOR_SAVE_FILE)

    print(f'Saving contestant data to {CONTESTANT_SAVE_FILE}...')
    contestants.to_csv(CONTESTANT_SAVE_FILE)
    
