# NBA Player Embiidings
This project contains the backing code and data for my project at https://www.connorlandy.com/projects/nba-player-embiidings. For background and commentary on the project, please refer to the website.   

Not all data is contained in this folder due to size limitations within Github. Most of the data near the end of the pipeline has been preserved due to its filtered nature. Early phases of the pipeline such as the original data has been deleted due to Github limitations. The original shot and PlayerVU tracking data that is an input to this project can be found in either my fork of the original data or the original Github repo by Neil Seward at https://github.com/sealneaward/nba-movement-data.

## Background

Copied from website:
```
While most of my hometown state of Texas adored (American) football growing up, my sports love was far and away basketball. Coming into sentience during the final peak years of his Airness, the seed in my heart for the game (to me, the game will always refer to basketball) continued to bloom over the years as the sport grew into the global phenomenon it is today.  
‍
So, with the NBA season back in full swing, analyzing the best game in the world (in my biased unbiased opinion) was top of mind as I considered the next project to bite into. Conveniently, at the start of every NBA season, me and a group of friends make bets on which teams will have the best reputation by the end of the season. So, I thought, why not try and augment my bet this year with a bit of data.

The project started with a quick and dirty time series model that approached the game from a 10,000 ft level. It took prior year records to predict the following season’s records. Unfortunately, the predictions didn’t have much alpha over existing public models and suffered from many of the problems also experienced by FiveThirtyEight in their own early models. So, I decided to approach this from a more micro level. Let’s measure and predict the impact a player has on a game, which then enables you to predict the immediate impact a player will have on a new team. In other words, rather than build a model from the top down, let’s build a model from the bottom up.

This signal is especially important to capture given all the trades that happen in the offseason (e.g., AD, Kyrie, KD, Kawhi, Russell Westbrook, Jimmy Butler this year alone) and will undoubtedly happen in the upcoming trade windows. Rather than approach this from a nearest neighbor model + Monte Carlo simulations like FiveThirtyEight, my hypothesis was that a deep neural embedding would be even more valuable in capturing the many semantics of a player.

For the data scientists out there, this would be like how word embeddings changed the game versus prior state of the art via one-hot encodings. There was some prior art on creating deep embeddings but for shot selection, so I had reasonable confidence this experiment would spit out logical results...
```

## Project Breakdown
- `cleaning\...`: contains cleaning and transformation scripts to prepare the model for training 
    - `parse_shot_frames.py`: Python script to clean, manipulate, and generate shot frame data
    - `generate_training_data.py`: Pyspark functions to clean, manipulate, and generate training data based on synthesized output of `parse_shot_frames.py`
    - `data/...`: contains some of the output files (as permitted by size limitations) of `parse_shot_frames.py` 

- `model\...`: contains the Google Collaboratory Notebook, inputs, and outputs
    - `...tsv`: embeddings and meta data files to be inputed into Google Embedding Projector
    - `offensive_defensive_embedding_model_best.h5`: saved TensorFlow model 
    - `player_aggregates.csv`: one of the output files of `generate_training_data.py` 

## Requirements
- pandas
- os
- pyspark  

