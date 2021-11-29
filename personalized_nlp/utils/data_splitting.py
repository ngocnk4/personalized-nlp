import numpy as np
import pandas as pd


def split_texts(df, sizes):
    present_ratio, past_ratio, future1_ratio, future2_ratio = sizes

    present_idx = int(present_ratio * len(df.index))
    past_idx = int(past_ratio * len(df.index)) + present_idx
    future1_idx = int(future1_ratio * len(df.index)) + past_idx

    indexes = np.arange(len(df.index))
    np.random.shuffle(indexes)

    df = df.copy()
    df['split'] = ''
    df.iloc[indexes[:present_idx], df.columns.get_loc('split')] = 'present'
    df.iloc[indexes[present_idx:past_idx],
            df.columns.get_loc('split')] = 'past'
    df.iloc[indexes[past_idx:future1_idx],
            df.columns.get_loc('split')] = 'future1'
    df.iloc[indexes[future1_idx:], df.columns.get_loc('split')] = 'future2'

    return df


def split_texts_by_original(df, sizes):
    present_ratio, past_ratio, future1_ratio, future2_ratio = sizes

    past_present_count = len(df[df['split'] == 'train'])

    present_past_ratio = int(present_ratio / (present_ratio + past_ratio))

    present_count = int(present_past_ratio * past_present_count)

    indexes = df.index[df['split'] == 'train'].tolist()
    np.random.shuffle(indexes)

    df.iloc[indexes[:present_count], df.columns.get_loc('split')] = 'present'
    df.iloc[indexes[present_count:], df.columns.get_loc('split')] = 'past'
    df.loc[df['split'] == 'dev', ['split']] = 'future1'
    df.loc[df['split'] == 'test', ['split']] = 'future2'

    return df
