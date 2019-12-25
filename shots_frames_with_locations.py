# grab frames with player locations
import pandas as pd
import numpy as np
from os import listdir

files = [f for f in listdir('data/shot_moments/')]

for file in files:
    try:
        print('starting {}'.format(file))

        shot_frames_pd = pd.read_csv('data/shot_moments/{}'.format(file))
        frames_pd = pd.read_csv('data/csv/{}'.format(file))

        # format shot frame data
        shot_frames_pd.drop(columns=['game_id_y'], inplace=True)
        shot_frames_pd.rename(columns=
                    {'game_id_x': 'game_id',
                    'game_clock': 'calculated_peak_game_clock',
                    }, inplace=True)

        # format frame data
        frames_pd['quarter_clock'] = 720 - frames_pd['game_clock']
        frames_pd['game_clock'] = (frames_pd['quarter'] - 1) * 720 + frames_pd['quarter_clock']
        frames_pd.drop(columns=['quarter_clock'], inplace=True)
        frames_pd.rename(columns=
            {'radius': 'z_loc'
            }, inplace=True)

        # join data together
        result = frames_pd.merge(shot_frames_pd, on='event_id', how='left', left_index=True)
        result = result[(result.game_clock < result.calculated_peak_game_clock + .1) & (result.game_clock > result.calculated_peak_game_clock)]

        # filter to last frame in tenth of a second window
        result = result.groupby(by=['event_id'])
        result = result.apply(lambda g: g[g['game_clock'] == g['game_clock'].max()])

        # save data
        result.to_csv('data/shot_frames_with_locations/{}'.format(file), index=False)
    
    except:
        print('error on {}'.format(file))
        continue