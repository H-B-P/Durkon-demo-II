import pandas as pd
import numpy as np
import durkon as du

#Load in.

trainDf = pd.read_csv("train.csv")
testDf = pd.read_csv("test.csv")
print(trainDf)

cats=["Flavour","Wedding","Fancy"]
conts=["Width","Height","Icing Thickness"]

#Construct dummy model.

model = du.wraps.prep_model(trainDf, "Price", cats, conts)

print(model)

#Train model

model = du.wraps.train_gamma_model(trainDf, "Price", 200, 0.1, model)

print(model)

#Visualize model

du.wraps.viz_multiplicative_model(model)

#Predict with model

preds = du.misc.predict(testDf)