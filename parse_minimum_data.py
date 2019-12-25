import pandas as pd
import numpy as np
from os import listdir

files = [f for f in listdir('data/csv/')]
print(files)

for file in files:
    data = pd.read_csv('data/csv/{}'.format(file))

    data = data[data['player_id'] == -1]
    data = data[['game_id', 'event_id', 'quarter', 'game_clock', 'radius']]

    # output file
    data.to_csv('data/filtered/{}'.format(file), index=False)
