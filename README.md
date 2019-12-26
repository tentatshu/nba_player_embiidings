# NBA Player Embiidings
This project contains the backing code and data for my project at #######. For background and commentary on the project, please refer to the website.   

Not all data is contained in this folder due to size limitations within Github. Most of the data near the end of the pipeline has been preserved due to its filtered nature. Early phases of the pipeline such as the original data has been deleted due to Github limitations. The original shot and PlayerVU tracking data that is an input to this project can be found in either my fork of the original data (#######) or the original Github repo by Neil #### (#####).

## Background


## Project Breakdown
- `cleaning\...`: contains cleaning and transformation scripts to prepare the model for training 
    - `parse_shot_frames.py`: Python script to clean, manipulate, and generate shot frame data
    - `generate_training_data.py`: Pyspark functions to clean, manipulate, and generate training data based on synthesized output of `parse_shot_frames.py`
    - `data/...`: contains some of the output files (as permitted by size limitations) of `parse_shot_frames.py` 

- `model\...`: contains the Google Collaboratory Notebook, inputs, and outputs
    - `...tsv`: embeddings and meta data files to be inputed into Google Embedding Projector
    - `offensive_defensive_embedding_model.h5`: saved TensorFlow model 
    - `player_aggregates.csv`: one of the output files of `generate_training_data.py` 

## Requirements
- pandas
- os
- pyspark  

