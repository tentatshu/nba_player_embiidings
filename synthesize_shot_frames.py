import pandas as pd
import numpy as np
from os import listdir

files = [f for f in listdir('data/shot_frames_with_locations/')]

result = pd.DataFrame()

for file in files:
    print('starting{}'.format(file))
    data = pd.read_csv('data/shot_frames_with_locations/{}'.format(file))
    result = pd.concat([result, data])

result.to_csv('data/shot_frames_with_locations/synthesized.csv', index=False)
