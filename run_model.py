import arviz
import numpy as np
import pandas as pd
from stan_utils import StanModel_cache

BASE_URL = "https://docs.google.com/spreadsheets/d/1Sotvl3o7J_ckKUg5sRiZTqNQn3hPqhepBSeOpMTK15Q/export?format=csv&gid={gid}"
POSTERIOR_SAVE_FILE = 'data/output_posterior.nd'
CONTESTANT_SAVE_FILE = 'data/output_contestants.csv'
STAN_FILE = 'model.stan'
OUTPUT_QUANTILES = {'low': 0.1, 'median': 0.5, 'high': 0.9}
GIDS = {
    'episodes': 0,
    'contestants': 1613421713,
    'rankings': 102708949,
    'social_media': 1915800778,
    'survey_votes': 810757234,
    'survey_contestants': 516773740
}
RANKS = {
    'WIN': 1,
    'Winner': 1,
    'Runner-up': 2,
    'HIGH': 2,
    'SAFE': 3,
    'LOW': 4,
    'BTM2': 5,
    'BTM6': 5,
    'ELIM': 6,
    'Eliminated': 6,
    'Guest': 6,
    'Miss C': 6,
    'DISQ': 6,
    'OUT': 6
}
IGNORE = ['Guest', 'DISQ', 'Miss C']
ELIMINATED = ['Eliminated', 'DISQ', 'OUT', 'ELIM']
Z_COLS = ['age', 'twitter_rank']
PREDICTORS = ['age_std', 'twitter_rank_std']
USE_EPISODES = True
USE_SURVEY = False
FETCH_RAW_DATA = True


def main(fetch_raw_data=FETCH_RAW_DATA,
         stan_file=STAN_FILE,
         use_episodes=USE_EPISODES,
         use_survey=USE_SURVEY):
    raw_data = get_raw_data(fetch=fetch_raw_data)

    twitter_followers_earliest = (
        raw_data['social_media']
        .sort_values('datetime')
        .groupby('contestant_id')
        ['followers_twitter']
        .first()
    )

    survey_head_to_heads = get_survey_head_to_heads(
        raw_data['survey_votes']
        .loc[lambda df: df['survey_id'] == df['survey_id'].max()]
    )

    contestants = (
        raw_data['contestants']
        .join(twitter_followers_earliest)
        .assign(
            twitter_rank=lambda df: (df.groupby('season_number')
                                    ['followers_twitter']
                                    .transform(lambda s: s.rank())),
            twitter_rank_std=lambda df: df['twitter_rank'].pipe(standardise),
            age_std=lambda df: df['age'].pipe(standardise)
        )
        .set_index('contestant_id')
        .drop(['season_number'], axis=1)
    )
    if 'Unnamed: 0' in contestants.columns:
        contestants = contestants.drop('Unnamed: 0', axis=1)

    rankings = (
        raw_data['rankings']
        .assign(rank=lambda df: df['episode_placement'].map(RANKS),
                eliminated=lambda df: df['episode_placement'].isin(ELIMINATED),
                episode_id=lambda df: (df[['season_number', 'episode_number']]
                                       .astype(str)
                                       .apply('-'.join, axis=1)
                                       .factorize()[0] + 1))
        .join(contestants, on='contestant_id')
        .loc[lambda df: ~df['episode_placement'].isin(IGNORE)]
        .sort_values(['season_number', 'episode_number', 'rank'])
    )


    # run model
    model_config = {
        'use_survey': int(use_survey),
        'use_episodes': int(use_episodes)
    }
    infd = run_model(rankings, survey_head_to_heads, stan_file, model_config)

    # print summary
    print('Fit summary:')
    print(arviz.summary(infd.posterior, var_names=['beta', 'sigma_ability']))

    # get contestant level output
    contestants_out = (
        pd.DataFrame({f'ability_{n}': (infd.posterior['ability']
                                       .quantile(q, dim=['chain', 'draw'])
                                       .to_series())
                      for n, q in OUTPUT_QUANTILES.items()})
        .join(rankings
              .groupby('contestant_id')[['contestant_name'] + PREDICTORS]
              .first())
    )

    # print information about next episode contestants
    latest_season, latest_episode = (
        raw_data['rankings']
        .sort_values(['season_number', 'episode_number'])
        [['season_number', 'episode_number']]
        .iloc[-1]
    )
    next_episode_contestant_ids = rankings.loc[lambda df: (
        (df['season_number'] == latest_season)
        & (df['episode_number'] == latest_episode)
        & ~df['eliminated']
    ), 'contestant_id']
    print('Next episode contestants:')
    print(contestants_out
          .loc[next_episode_contestant_ids]
          .set_index('contestant_name')
          .sort_values('ability_median', ascending=False)
          [['ability_low', 'ability_median', 'ability_high']].round(2))
    print('Top 20 all time queens:')
    print(contestants_out
          .set_index('contestant_name')
          .sort_values('ability_median', ascending=False)
          [['ability_low', 'ability_median', 'ability_high']].round(2)
          .head(20))
    # save output
    print(f'Saving poterior to {POSTERIOR_SAVE_FILE}...')
    infd.to_netcdf(POSTERIOR_SAVE_FILE)
    print(f'Saving contestant data to {CONTESTANT_SAVE_FILE}...')
    contestants_out.to_csv(CONTESTANT_SAVE_FILE)


def run_model(rankings, survey_head_to_heads, stan_file=STAN_FILE, model_config='combined'):
    n_episode_contestant = rankings.groupby('episode_id')['contestant_id'].nunique()
    episode_rank_counts = (rankings
                           .groupby(['episode_id', 'rank'])
                           .size()
                           .unstack()
                           .fillna(0)
                           .astype(int))
    contestants = rankings.groupby('contestant_id').first()
    contestants['id_stan'] = range(1, len(contestants) + 1)
    rankings = rankings.join(contestants['id_stan'], on='contestant_id')
    survey_head_to_heads = (
        survey_head_to_heads
        .join(contestants['id_stan'].rename('id_stan_own'), on='own')
        .join(contestants['id_stan'].rename('id_stan_opp'), on='opp')
    )
    input_data = {
        'N': len(rankings),
        'K': len(PREDICTORS),
        'C': len(contestants),
        'E': rankings['episode_id'].nunique(),
        'X': contestants[PREDICTORS].fillna(0).values,
        'N_episode_contestant': n_episode_contestant.values,
        'episode_rank': rankings['rank'].values,
        'contestant': rankings['id_stan'].values,
        'N_survey': len(survey_head_to_heads),
        'survey_contestant': survey_head_to_heads['id_stan_own'].values,
        'survey_opponent': survey_head_to_heads['id_stan_opp'].values,
        'survey_count': survey_head_to_heads['count'].values,
        'survey_wins': survey_head_to_heads['wins'].values
    }
    model = StanModel_cache(file=stan_file)
    fit = model.sampling(data={**input_data, **model_config})
    return arviz.from_pystan(fit,
                             coords={'contestant': contestants.index,
                                     'predictor': PREDICTORS},
                             dims={'ability': ['contestant'],
                                   'beta': ['predictor']})


def standardise(s: pd.Series):
    return s.subtract(s.mean()).div(2*s.std())


def get_survey_head_to_heads(votes):
    counts = (
        votes
        .set_index(['vote_id', 'vote_result'])
        ['contestant_id']
        .unstack()
        .groupby(['WIN', 'LOSE'])
        .size()
    )
    lower_tri = counts.unstack().fillna(0).where(np.tril(np.ones(counts.unstack().shape), k=-1).astype(bool))
    upper_tri = counts.unstack().fillna(0).where(np.triu(np.ones(counts.unstack().shape), k=1).astype(bool))
    votes = pd.DataFrame({'count': lower_tri.stack() + upper_tri.stack().values,
                          'wins': lower_tri.stack()}).astype(int)
    votes.index.names = ['own', 'opp']
    votes = votes.reset_index()
    return votes


def get_raw_data(fetch=True):
    out = {}
    for table_name, gid in GIDS.items():
        if fetch:
            table = pd.read_csv(BASE_URL.format(gid=str(gid)))
            table.to_csv(f'data/{table_name}.csv')
            out[table_name] = table
        else:
            out[table_name] = pd.read_csv(f'data/{table_name}.csv')
    return out


if __name__ == '__main__':
    main()
