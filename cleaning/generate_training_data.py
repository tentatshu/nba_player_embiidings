__author__ = “Connor Landy”

""" Pyspark functions to clean, manipulate, 
    and generate training data for the following project: 
    ############
"""

from pyspark.sql.window import Window
import pyspark.sql.functionas as F

def generate_aggregate_player_data(shots_fixed):
    """ Generates aggregate information and unique count for NBA players based on shots taken

    Parameters
    ----------
    shots_fixed: Dataframe containing NBA player shot data and the following columns:
        'GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID', 'PLAYER_NAME',
        'EVENT_TYPE', 'LOC_X', 'LOC_Y', and 'SHOT_DISTANCE'

    Returns
    -------
    Aggregated player data

    Notes
    -----
    """

    # filter to relevant columns
    result =  shots_fixed.select('GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID', 'PLAYER_NAME', 'EVENT_TYPE', 'LOC_X', 'LOC_Y', 'SHOT_DISTANCE')

    # categorize shots
    result = result.withColumn('EVENT_TYPE', F.when(F.col('EVENT_TYPE')== 'Missed Shot', 0).otherwise(1))

    # aggregate all player data
    result = result.groupBy('PLAYER_ID', 'PLAYER_NAME').pivot('EVENT_TYPE').count()
    result = result.withColumnRenamed('0', 'missed_shot').withColumnRenamed('1', 'made_shot')

    # sort and add unique numerical id for training in TensorFlow
    w1 = Window.orderBy("PLAYER_ID")
    result = result.withColumn('rank', F.rank().over(w1))
    result = result.withColumn('rank', F.col('rank')- 1)
    
    return result


### converting from shots.csv and synthesized.csv to shooter and defender per shot
def shooter_and_defender_per_shot(synthesized, shots, player_aggregates):
    """ Generates Dataframe where each row contains the exact frame when a player took a shot
        based on synthesized data outputted by XXXXXXX and the shots.csv file

    Parameters
    ----------
    synthesized: Dataframe containing tracking data filtered to just the frame where the
        shot is initiated
    shots: Dataframe containing information around whether shot was successful
    player_aggregates: output of generate_aggregate_player_data function

    Returns
    -------
    Dataframe where each row contains the exact frame when a player took a shot enhanced with information
        such as the nearest defender and whether the shot was successful

    Notes
    -----
    """

    # prepare the data

    # filter only to games with more than 140 shots to remove games where SHOT VU has issues with game calculated_peak_game_clock
    filtered_games = synthesized.groupBy('game_id_x').agg(F.countDistinct("event_id")).withColumnRenamed('count(DISTINCT event_id)', 'count_of_events')
    filtered_games = filtered.filter(F.col('count_of_events') > 140)
    filtered_games = synthesized.join(filtered, ['game_id_x'], 'inner')

    # generate Dataframe containing only the location of the ball in each frame
    ball_locations = filtered_games.filter(F.col('player_id') == -1).select('game_id_x', 'event_id', 'player_id', 'x_loc', 'y_loc', 'z_loc_x')\
        .withColumnRenamed('game_id_x', 'game_id').withColumnRenamed('z_loc_x', 'z_loc')\
        .withColumnRenamed('x_loc', 'x_loc_ball').withColumnRenamed('y_loc', 'y_loc_ball')\
        .drop('player_id')

    # generate Dataframe containing only the location of players in each frame
    player_shooter_location = filtered_games.filter(F.col('player_id') == F.col('recorded_player_id'))\
        .select('game_id_x', 'event_id', 'team_id', 'player_id', 'x_loc', 'y_loc')\
        .withColumnRenamed('game_id_x', 'game_id').withColumnRenamed('player_id', 'shooting_player_id')\
        .withColumnRenamed('x_loc', 'x_loc_shooter').withColumnRenamed('y_loc', 'y_loc_shooter') \
        .withColumnRenamed('team_id', 'team_id_shooter')

    # generate Dataframe containing each shot dataframe enhanced with the closest defender

    # join player data with ball data
    frame_filtered = filtered_games.select('game_id_x', 'team_id', 'game_clock', 'event_id', 'shot_clock', 'player_id', 'x_loc', 'y_loc', 'recorded_player_id', 'recorded_loc_x', 'recorded_loc_y')
    frame_filtered = frame_filtered.filter(F.col('player_id') != -1).withColumnRenamed('game_id_x', 'game_id')
    players_joined_with_ball_location = frame_filtered.join(ball_locations, ['game_id', 'event_id'], 'left')
    players_joined_with_ball_location = players_joined_with_ball_location.join(player_shooter_location, ['game_id', 'event_id'], 'left')

    # generate Dataframe containing player distance from ball
    distance_from_ball = players_joined_with_ball_location.withColumn('distance_from_ball', F.sqrt(F.pow(F.col('x_loc')- F.col('x_loc_ball'), 2) + F.pow(F.col('y_loc')- F.col('y_loc_ball'),2)))
    distance_from_ball = distance_from_ball.withColumn('distance_from_shooter', F.sqrt(F.pow(F.col('x_loc')- F.col('x_loc_shooter'), 2) + F.pow(F.col('y_loc')- F.col('y_loc_shooter'),2)))

    # determine which player was the closest distance from ball
    closest_player_on_opposing_team = distance_from_ball.withColumn('opposing_team', F.col('team_id') != F.col('team_id_shooter')) \
        .filter(F.col('opposing_team'))
    w1 = Window.partitionBy('game_id', 'event_id')
    closest_player_on_opposing_team = closest_player_on_opposing_team.withColumn('closest_player_location', F.min('distance_from_shooter').over(w1)) \
        .withColumn('closest_player_flag', F.col('closest_player_location') == F.col('distance_from_shooter')) \
        .filter(F.col('closest_player_flag'))
    closest_player_on_opposing_team = closest_player_on_opposing_team.select('game_id', 'event_id', 'player_id', 'distance_from_shooter')\
        .withColumnRenamed('distance_from_shooter', 'closest_distance_from_shooter')\
        .withColumnRenamed('player_id', 'player_id_closest_defender')

    # final preparation of data

    # filter to only the shooter and join on closest defender
    shooter_and_defender_per_shot = distance_from_ball.filter(F.col('player_id') == F.col('recorded_player_id')) \
        .join(closest_player_on_opposing_team, ['game_id', 'event_id'], 'left')
    
    # flip so shots appear as if on same side of court
    # dimensions of court are 94 feet (horizontal) by 50 feet (vertical)
    shooter_and_defender_per_shot = shooter_and_defender_per_shot.withColumn('x_loc', F.when(F.col('x_loc') > 47, 94 - F.col('x_loc')).otherwise(F.col('x_loc')))
    shooter_and_defender_per_shot = shooter_and_defender_per_shot.withColumn('y_loc', F.when(F.col('x_loc') > 47, 50 - F.col('y_loc')).otherwise(F.col('y_loc')))

    # calculate distance from basket which is located 3 feet from out of bounds and 25 feet from either edge of court
    shooter_and_defender_per_shot = shooter_and_defender_per_shot.withColumn('distance_from_basket', F.sqrt(F.pow(F.col('x_loc')- 3, 2) + F.pow(F.col('y_loc')- 25,2)))

    # enhance data with information about whether shot is successful or not
    shots_filtered = shots.select('GAME_ID','GAME_EVENT_ID', 'SHOT_MADE_FLAG', 'SHOT_DISTANCE') \
        .withColumnRenamed('GAME_ID', 'game_id') \
        .withColumnRenamed('GAME_EVENT_ID', 'event_id') \
        .withColumnRenamed('SHOT_MADE_FLAG', 'shot_made_flag') \
        .withColumnRenamed('SHOT_DISTANCE', 'recorded_shot_distance')

    shooter_and_defender_per_shot = shooter_and_defender_per_shot.join(shots_filtered, ['game_id', 'event_id'], 'left') \
        .select('game_id', 'event_id', 'game_clock', 'shot_clock', 'player_id', 'x_loc', 'y_loc', 'player_id_closest_defender', 'closest_distance_from_shooter' ,'distance_from_basket', 'shot_made_flag', 'recorded_shot_distance')

    # join in embedding index required for tensorflow embedding
    player_aggregates_filtered = player_aggregates.select('PLAYER_ID', 'rank').withColumnRenamed('PLAYER_ID', 'player_id').withColumnRenamed('rank', 'embedding_index')
    result = shooter_and_defender_per_shot.join(player_aggregates_filtered, 'player_id', 'left').withColumnRenamed('embedding_index', 'shooter_embedding_index')
    result = result.join(player_aggregates_filtered, [result.player_id_closest_defender == player_aggregates_filtered.player_id], 'left').withColumnRenamed('embedding_index', 'defender_embedding_index').drop(player_aggregates_filtered.player_id)
    
    return result

