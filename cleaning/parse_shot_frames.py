__author__ = “Connor Landy”

""" Python script to clean, manipulate, 
    and generate shot frame data for the following project: 
    ############
    
    Parameters
    ----------
    csv_data: CSV containing NBA player shot data saved from ###### 
        and located in the 'data/csv' folder

    Returns
    -------
    csv_data: CSV containing the location of every player on the court as well
        as the ball at the time the ball is determined to have been shot  

    Notes
    -----
    Output of this file is the input into generate_training_data.py.
    
    Composed primarily in Pandas due to network speed limitations to upload full 
    data to distributed environment as opposed to local environment
"""



import pandas as pd
from os import listdir



# filter raw data to only necessary columns and location of the ball
# save resulting data into data/filtered folder
files = [f for f in listdir('data/csv/')]
print(files)

for file in files:
    data = pd.read_csv('data/csv/{}'.format(file))

    # filter to only the location of the ball and needed columns
    data = data[data['player_id'] == -1]
    data = data[['game_id', 'event_id', 'quarter', 'game_clock', 'radius']]

    # output file
    data.to_csv('data/events_ball_filtered/{}'.format(file), index=False)



# determine frame that contains occurrence of shot and then filter to this frame
# save resulting in data/shot_moments folder

files = [f for f in listdir('data/events_ball_filtered/')]

for file in files:
    try:
        print('starting {}'.format(file))

        # ingest data as pandas dataframes
        ball_locations_pd = pd.read_csv('data/events_ball_filtered/{}'.format(file))
        shots_pd = pd.read_csv('data/shots/shots.csv')

        # format ball data
        ball_locations_pd['quarter_clock'] = 720 - ball_locations_pd['game_clock']
        ball_locations_pd['game_clock'] = (ball_locations_pd['quarter'] - 1) * 720 + ball_locations_pd['quarter_clock']
        ball_locations_pd.rename(columns=
                                 {'radius': 'z_loc'
                                  }, inplace=True)
        # filter to frame with peak height of the ball over tenth of second
        ball_locations_pd['times_ten'] = (ball_locations_pd['game_clock'] * 10).astype(int)
        ball_locations_pd = ball_locations_pd.groupby(['game_id', 'event_id', 'quarter', 'times_ten'], as_index=False)[
            'z_loc'].mean()
        ball_locations_pd['game_clock'] = ball_locations_pd['times_ten'] / 10
        ball_locations_pd.drop(columns=['times_ten'], inplace=True)

        # formatting shot data
        shots_pd = shots_pd[shots_pd['GAME_ID'] == int(file[:-4])]
        shots_pd['recorded_game_clock'] = (shots_pd['PERIOD'] - 1) * 720 + (
                    11 - shots_pd['MINUTES_REMAINING']) * 60 + 60 - shots_pd['SECONDS_REMAINING']
        shots_pd.rename(columns=
                        {'GAME_ID': 'game_id',
                         'GAME_EVENT_ID': 'event_id',
                         'PLAYER_ID': 'recorded_player_id',
                         'PLAYER_NAME': 'recorded_player_name',
                         'LOC_X': 'recorded_loc_x',
                         'LOC_Y': 'recorded_loc_y'
                         }, inplace=True)
        shots_pd = shots_pd[['game_id', 'event_id', 'recorded_game_clock', 'recorded_player_id', 'recorded_player_name',
                             'recorded_loc_x', 'recorded_loc_y']]

        # join data and filter frames to proceeding 5 seconds of recorded shot (based on empirical testing)
        result = ball_locations_pd.merge(shots_pd, on='event_id', how='left', left_index=True)
        result = result[
            (result.game_clock < result.recorded_game_clock) & (result.game_clock > result.recorded_game_clock - 5)]

        # detect peak shot
        max_shot = result.groupby(by=['event_id'])
        max_shot = max_shot.apply(lambda g: g[g['z_loc'] == g['z_loc'].max()])
        max_shot = max_shot[['event_id', 'game_clock']].reset_index(drop=True)
        max_shot.rename(columns=
                        {'game_clock': 'peak_shot_game_clock'
                         }, inplace=True)

        # filter to moments before peak shot
        result = result.merge(max_shot, on='event_id', how='inner')
        result = result[result.game_clock <= result.peak_shot_game_clock]

        # filter to frame where shot is initiated by finding frame where ball is at its last trough before shot
        result['last_z'] = (result.sort_values(by=['game_clock'], ascending=True)
                            .groupby(['event_id'])['z_loc'].shift(1))
        result['next_z'] = (result.sort_values(by=['game_clock'], ascending=True)
                            .groupby(['event_id'])['z_loc'].shift(-1))
        result = result[(result.last_z > result.z_loc) & (result.next_z > result.z_loc)]
        result = result.groupby(by=['event_id'])
        result = result.apply(lambda g: g[g['game_clock'] == g['game_clock'].max()])

        # formatting
        result.drop(columns=['last_z', 'next_z'], inplace=True)

        # save data
        result.to_csv('data/shot_moments/{}'.format(file), index=False)

    except:
        print('error on {}'.format(file))
        continue



# isolate and filter full tracking data to frame where the ball was determined to be shot
# output will contain ball and player locations and be saved to 'data/shot_frames_with_locations'

files = [f for f in listdir('data/shot_moments/')]

for file in files:
    try:
        print('starting {}'.format(file))

        # read data
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

        # join
        result = frames_pd.merge(shot_frames_pd, on='event_id', how='left', left_index=True)
        result = result[(result.game_clock < result.calculated_peak_game_clock + .1) & (
                    result.game_clock > result.calculated_peak_game_clock)]

        # filter to last frame in tenth of a second window
        result = result.groupby(by=['event_id'])
        result = result.apply(lambda g: g[g['game_clock'] == g['game_clock'].max()])

        # save data
        result.to_csv('data/shot_frames_with_locations/{}'.format(file), index=False)

    except:
        print('error on {}'.format(file))
        continue



# concatenate all data within 'data/shot_frames_with_locations/' folder into one file
# save in 'data/shot_frames_with_locations/' folder
files = [f for f in listdir('data/shot_frames_with_locations/')]

result = pd.DataFrame()

for file in files:
    print('starting {}'.format(file))
    data = pd.read_csv('data/shot_frames_with_locations/{}'.format(file))
    result = pd.concat([result, data])

result.to_csv('data/shot_frames_with_locations/synthesized.csv', index=False)