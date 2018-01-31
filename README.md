# Classify_Movie_Reviews_Naive_bayes

Download labeledTrainData.tsv, which contains 25000 IMDB movie reviews from here: https://goo.gl/6Qn2vM

Use Pandas http://pandas.pydata.org python package to read this file. Pandas package is preinstalled in your canopy python distribution.

import pandas as pd  
train = pd.read_csv('labeledTrainData', header=0, delimiter='\t', quoting=3) 
