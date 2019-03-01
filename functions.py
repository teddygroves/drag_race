import pandas as pd
import numpy as np
import pystan

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
Z_COLS = ['age']


def z_score(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def prepare_data():
    episodes = (
        pd.read_csv('data/all_episodes.csv')
        .drop('Unnamed: 0', axis=1)
        .set_index(['season_number', 'episode_number'])
    )
    rankings = (
        pd.read_csv('data/all_rankings.csv')
        .drop('Unnamed: 0', axis=1)
        .assign(rank=lambda df: df['episode_placement'].map(PLACE_MAP),
                episode_id=lambda df: (
                    df[['season_number', 'episode_number']]
                    .astype(str)
                    .apply('-'.join, axis=1)
                    .factorize()[0] + 1
                ))
        .sort_values(['season_number', 'episode_number', 'rank'])
    )
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
    rankings = rankings.join(episodes['episode_type'], on=['season_number', 'episode_number'])
    rankings = rankings.loc[lambda df: df['episode_type'] == 'Competition']
    return contestants, rankings, social_media, episodes


def get_stan_input(contestants, rankings):
    predictors = ['age_z', 'is_ny']
    return {
        'N': len(rankings),
        'K': len(predictors),
        'N_episode': len(rankings['episode_id'].unique()),
        'N_contestant': len(contestants),
        'contestant': rankings['contestant_id'],
        'N_episode_contestant': rankings.groupby('episode_id').size(),
        'X': contestants[predictors]
    }
    


c, r, s, e = prepare_data()
model_data = get_stan_input(c, r)
model = pystan.StanModel(file='simple_model.stan')
fit = model.sampling(data=model_data)


print(fit.stansummary(pars=['lp__', 'beta', 'sigma_ability_maxi', 'sigma_ability_lipsync']))
c['maxi_ability_mean'] = fit['ability_maxi'].mean(axis=0)
c['lipsync_ability_mean'] = fit['ability_lipsync'].mean(axis=0)
c['lipsync_ability_sd'] = fit['ability_lipsync'].std(axis=0)
c['maxi_rank'] = c['maxi_ability_mean'].rank(ascending=False)
c['lipsync_rank'] = c['lipsync_ability_mean'].rank(ascending=False)
print(c.sort_values('lipsync_ability_mean', ascending=False)
      .head(20)
      [['season_number', 'contestant_name', 'maxi_ability_mean',
        'lipsync_ability_mean', 'maxi_rank', 'lipsync_rank']]
)

