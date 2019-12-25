import pandas as pd
import numpy as np
from os import listdir

files = [f for f in listdir('data/events_ball_filtered/')]

for file in files:
    try:
        print('starting {}'.format(file))
        # ingest data as pandas dataframes
        ball_locations_pd = pd.read_csv('data/events_ball_filtered/{}'.format(file))
        shots_pd = pd.read_csv('data/shots/shots.csv')

        # formatting ball data
        ball_locations_pd['quarter_clock'] = 720 - ball_locations_pd['game_clock']
        ball_locations_pd['game_clock'] = (ball_locations_pd['quarter'] - 1) * 720 + ball_locations_pd['quarter_clock']
        ball_locations_pd.rename(columns=
            {'radius': 'z_loc'
            }, inplace=True)
        # take top ball location over tenth of second 
        ball_locations_pd['times_ten'] = (ball_locations_pd['game_clock'] * 10).astype(int)
        ball_locations_pd = ball_locations_pd.groupby(['game_id', 'event_id', 'quarter', 'times_ten'], as_index=False)['z_loc'].mean()
        ball_locations_pd['game_clock'] = ball_locations_pd['times_ten'] / 10
        ball_locations_pd.drop(columns=['times_ten'], inplace=True)


        # formatting shot data
        shots_pd = shots_pd[shots_pd['GAME_ID'] == int(file[:-4])]
        shots_pd['recorded_game_clock'] = (shots_pd['PERIOD'] - 1)*720 + (11 - shots_pd['MINUTES_REMAINING'])*60 +  60 - shots_pd['SECONDS_REMAINING']
        shots_pd.rename(columns=
            {'GAME_ID': 'game_id', 
            'GAME_EVENT_ID': 'event_id',
            'PLAYER_ID': 'recorded_player_id',
            'PLAYER_NAME': 'recorded_player_name',
            'LOC_X': 'recorded_loc_x',
            'LOC_Y': 'recorded_loc_y'
            }, inplace=True)
        shots_pd = shots_pd[['game_id','event_id', 'recorded_game_clock', 'recorded_player_id', 'recorded_player_name', 'recorded_loc_x', 'recorded_loc_y']]

        # join data and filter frames to preceeding 5 seconds of recorded shot (empirical testing)
        result = ball_locations_pd.merge(shots_pd, on='event_id', how='left', left_index=True)
        result = result[(result.game_clock < result.recorded_game_clock) & (result.game_clock > result.recorded_game_clock - 5)]

        # detect peak shot
        max_shot = result.groupby(by=['event_id'])
        max_shot = max_shot.apply(lambda g: g[g['z_loc'] == g['z_loc'].max()])
        max_shot = max_shot[['event_id', 'game_clock']].reset_index(drop=True)
        max_shot.rename(columns=
            {'game_clock': 'peak_shot_game_clock'
            }, inplace=True)

        # filter to only moments before peak shot
        result = result.merge(max_shot, on='event_id', how='inner')
        result = result[result.game_clock <= result.peak_shot_game_clock]

        # filter to shot frame by finding frame where ball is at its last trough before shot
        result['last_z'] = (result.sort_values(by=['game_clock'], ascending=True)
                            .groupby(['event_id'])['z_loc'].shift(1))
        result['next_z'] = (result.sort_values(by=['game_clock'], ascending=True)
                            .groupby(['event_id'])['z_loc'].shift(-1))
        result = result[(result.last_z > result.z_loc) & (result.next_z > result.z_loc)]
        result = result.groupby(by=['event_id'])
        result = result.apply(lambda g: g[g['game_clock'] == g['game_clock'].max()])

        # formatting
        result.drop(columns=['last_z', 'next_z'], inplace=True)

        result.to_csv('data/shot_moments/{}'.format(file), index=False)
    
    except:
        print('error on {}'.format(file))
        continue

