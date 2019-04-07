import numpy as np
import pandas as pd
from itertools import chain, permutations, product

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
Z_COLS = ['age', 'followers_twitter']


def prepare_data():
    social_media = pd.read_csv('data/all_social_media.csv',
                               skiprows=[1],
                               index_col=0)
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
        .assign(rank=lambda df: df['episode_placement'].map(RANKS),
                eliminated=lambda df: df['episode_placement'].isin(ELIMINATED))
        .join(episodes[['episode_type', 'episode_id']],
              on=['season_number', 'episode_number'])
        .sort_values(['season_number', 'episode_number', 'rank'])
    )
    season_starts = (
        episodes.groupby('season_number')['episode_airdate'].min()
        .rename('season_start')
    )
    contestants = (
        pd.read_csv('data/all_contestants.csv')
        .drop('Unnamed: 0', axis=1)
        .join(season_starts, on='season_number')
        .join(get_earliest_twitter(social_media), on='contestant_id')
        .assign(twitter_rank=lambda df: (df
                                         .groupby('season_number')
                                         ['followers_twitter']
                                         .transform(lambda s: s.rank())
                                         .pipe(z_score, fill=0)))
        .set_index(['contestant_id', 'season_number'])
    )
    # some extra columns
    for col in Z_COLS:
        contestants[col + '_z'] = z_score(contestants[col], fill=0)
    # filtering
    rankings = rankings.loc[lambda df: (
        (df['episode_type'] == 'Competition')
        & ~(df['episode_placement'].isin(IGNORE))
    )]

    return rankings.join(contestants, on=['contestant_id', 'season_number'])


def z_score(s, fill=np.nan):
    return (s - s.mean()).divide(s.std()).fillna(fill)


def get_earliest_twitter(social_media: pd.DataFrame):
    def is_earliest(sm):
        return sm['datetime'] == (
            sm.groupby('contestant_id')['datetime'].transform('min')
        )
    return (
        social_media
        .dropna(subset=['followers_twitter'])
        .set_index('contestant_id')
        .loc[lambda df: is_earliest(df), 'followers_twitter']
    )
