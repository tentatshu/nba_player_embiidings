import pandas as pd
import numpy as np
from os import listdir

files = [f for f in listdir('data/filtered/')]
print(files)

data = pd.DataFrame()

for idx, file in enumerate(files):
    print(idx)
    temp_data = pd.read_csv('data/filtered/{}'.format(file))
    data = pd.concat([data,temp_data])

    # output file
    if (idx % 26) == 25:
        data.to_csv('data/filtered/consolidated_{}.csv'.format(idx), index=False)
        print('saved')
        data = pd.DataFrame()
    
