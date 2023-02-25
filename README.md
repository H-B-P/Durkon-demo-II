# Introduction

Durkon is a package which allows users to create inherently-interpretable predictive models. It's primarily intended for use in an Insurance context, but is potentially useful for anyone who wants to build models they can understand.

# Installing Durkon

To get the latest release of Durkon, open a terminal and run

```python
pip install durkon
```
This will also install the three packages Durkon depends on: numpy, pandas and plotly.

# Building a Gamma GLM

Gamma-distributed multiplicative models are conventional for Market models and cost-per-claim Risk models; more generally, for any model predicting how much something will cost. This is the most common model used in Insurance contexts, so it's the one I'll start with.

This first tutorial will walk you through what happens in this demo, where a model is created to predict how much cakes will cost.

First, open a python REPL in the folder you put [the](https://raw.githubusercontent.com/H-B-P/Durkon-demo-II/main/train.csv) [data](https://raw.githubusercontent.com/H-B-P/Durkon-demo-II/main/test.csv) in and import the packages you'll need: 
```python
import pandas as pd
import numpy as np
import durkon as du
```
Then, load in the training data as a Pandas dataframe.

```python
trainDf = pd.read_csv("train.csv")
print(trainDf)
>>>    Width  Height  Icing Thickness     Flavour Wedding     Fancy   Price
0         13       7                6     vanilla       No  Somewhat  63.50
1          8       6                9       lemon       No  Somewhat  13.26
2         13       6                6     vanilla       No  Somewhat  88.26
3          6       6                4    marzipan       No  Somewhat   8.76
4          8      10                5   chocolate      Yes  Somewhat  24.95
...      ...     ...              ...         ...      ...       ...    ...
12340     12       5               12     vanilla       No        No  11.58
12341      5       7               10      coffee       No      Very   5.12
12342      7       6                5  strawberry       No  Somewhat  14.95
12343     12       8               11    marzipan       No  Somewhat  35.78
12344     11       8               11   chocolate       No  Somewhat  24.22

[12345 rows x 7 columns]
```

Examining the dataset, you can see at a glance that Width, Height and Icing Thickness are continuous variables, while Flavour, Wedding and Fancy are categorical. Add the column names to the appropriate lists.

```python
cats=["Flavour","Wedding","Fancy"]
conts=["Width","Height","Icing Thickness"]
```


Use the dataset to produce an untrained "dummy" model.

```python
model = du.wraps.prep_model(trainDf, "Price", cats, conts)
```


If you look at the model with 


```python
print(model)
>>>{'BASE_VALUE': 30.06318023491292, 'featcomb': 'mult', 'cats': {'Flavour': {'OTHER': 1, 'uniques': {'vanilla': 1, 'lemon': 1, 'marzipan': 1, 'chocolate': 1, 'strawberry': 1}}, 'Wedding': {'OTHER': 1, 'uniques': {'No': 1, 'Yes': 1}}, 'Fancy': {'OTHER': 1, 'uniques': {'Somewhat': 1, 'Very': 1, 'No': 1}}}, 'conts': {'Width': [[4, 1], [7, 1], [9, 1], [11, 1], [14, 1]], 'Height': [[4, 1], [6, 1], [7, 1], [9, 1], [13, 1]], 'Icing Thickness': [[4, 1], [7, 1], [8, 1], [9, 1], [12, 1]]}}
```

you'll see that it's represented as a simple collection of nested lists and dictionaries. This can be viewed and edited however and whenever you please.

We'll get into exactly what this structure means shortly. For now, all you should know is that this model currently says "to make a prediction, take the mean value in the training dataset for everything, then multiply by 1 for each feature regardless of what value that feature takes". This isn't very useful, since that's just a fancy way of saying "predict the mean value for everything". So it needs to be fit to the data:

```python
model = du.wraps.train_gamma_model(trainDf, "Price", 200, 0.1, model)
```

If you look at the model after training

```python
print(model)
>>>{'BASE_VALUE': 30.06318023491292, 'featcomb': 'mult', 'cats': {'Flavour': {'OTHER': 1.2048980212766978, 'uniques': {'vanilla': 0.9639571571469416, 'lemon': 1.168822426896769, 'marzipan': 0.9869494596061243, 'chocolate': 1.0652591408488579, 'strawberry': 0.9746296704984564}}, 'Wedding': {'OTHER': 1, 'uniques': {'No': 1.078724828647158, 'Yes': 0.6602304522413379}}, 'Fancy': {'OTHER': 1, 'uniques': {'Somewhat': 1.1246799292529732, 'Very': 1.0252630400928096, 'No': 0.8873356168622061}}}, 'conts': {'Width': [[4, 0.1589218144658377], [7, 0.4906992596089667], [9, 0.8375359017447552], [11, 1.2428056483466765], [14, 2.0328306527991447]], 'Height': [[4, 0.5245772198160243], [6, 0.7828461970778591], [7, 0.9105828044846489], [9, 1.1481691378017878], [13, 1.9265640919381362]], 'Icing Thickness': [[4, 1.13972173517173], [7, 1.2534223023683586], [8, 1.0554688550040638], [9, 0.923582790214725], [12, 0.5350731839837932]]}}
```

you'll see it's changed. You'll also see that this isn't the most enlightening way to view a trained model, which makes this a good opportunity to introduce Durkon's use of plotly visualizations:

```python
du.wraps.viz_multiplicative_model(model)
```
Running the above code should produce a folder of output graphs which look like this.

Note that these graphs do not describe the model post-hoc, like with AvE plots. They (together with the BASE_VALUE) *are* the model. Whenever the model makes a prediction, it starts with the BASE_VALUE, looks up the multiplier for each feature, and applies that multiplier.

For example, in the first row of {the test dataset}, the multiplier for Width would be ~0.38, because that's the y-coordinate of the relevant line when the value of x is 6. 

![enter image description here](https://h-b-p.github.io/Durkon-demo-II/graphs/Width.png)

Similarly, the multiplier for Flavour would be ~1.07, since that's the y-coordinate of the relevant bar when Flavour = Chocolate.

![enter image description here](https://h-b-p.github.io/Durkon-demo-II/graphs/Flavour.png)

Durkon would therefore make a prediction for this row by starting with the BASE_VALUE of ~30.06, multiplying by ~0.38 for Width, multplying by ~1.07 for Flavour, and multiplying by four other factors for the four other explanatory variables. In other words,

$$ [Prediction] = [Base Value] * [Multiplier from Width] * [Multiplier from Height] * [Multiplier from Icing Thickness] * [Multiplier from Flavour] * [Multiplier from Wedding] * [Multiplier from Fancy]

You can use your model to predict new data as shown below:

```python
testDf = pd.read_csv('test.csv')
preds = du.misc.predict(testDf, model)
```

(If you like, you could confirm your understanding of how Durkon works by comparing the output of this process to a prediction calculated by hand using the graphs.)

# Parameters

## prep_model()
prep_model() has the following required parameters:

 - **inputDf:** the dataframe used to specify the model.
 - **resp:** the column name for the response variable in inputDf; the thing you're trying to use explanatory columns to predict.
 - **cats:** the categorical explanatory columns.
 - **conts:** the continuous explanatory columns.


It also has the following optional parameters:

- **catMinPrev:** the minimum prevalence a category needs before it gets its own bar (note how in the Flavour graph, "raspberry" and "coffee" get lumped into "OTHER" instead of getting their own bars, because they aren't a large enough fraction of the dataset). 1% by default.
- **contTargetPts:** the number of points (joints, hinges) each continuous graph will aim to have (sometimes it will be lower than this when the generated points coincide). 5 by default; increase if you want to model continuous features in more detail, and decrease if you want to avoid overfit.
- **edge:** the extent to which each continuous graph rounds off its extreme values. (note how the lowest point in the Width graph is 4 even though the lowest value is 3, because less than 1/100 of the data is below 4). 1% by default.

## train_gamma_model()
train_gamma_model() has the following required parameters:

- **inputDf:** the dataframe used to specify the model.
- **resp:** the column name for the response variable in inputDf; the thing you're trying to use explanatory columns to predict.
- **nrounds:** the number of rounds you want to train the model for. Increasing this will almost invariably make the model better, but returns will diminish rapidly.
- **lr:** the learning rate you want the model to be trained with. Increasing this makes the model learn more in each round, which speeds things up but makes the training process less stable. Lower this if the model glitches or crashes.
- **model:** the model being used as a starting point. This can be a blank model created by prep_model(), or a model which has already been trained (or a model you built by hand, or a model someone gave you . . .).

It also has the following optional parameters:

- **pen:** the level of LASSO penalization applied during training. Useful for featureselection and accounting for overfit. 0 by default.
- **weightCol:** the column which specifies how the rows should be weighted while training (if a row has a weight of 100, that's the same as if the dataset had 100 copies of that row, each with weight 1). *None* by default; in other words, when this isn't specified, all rows have an equal weight of 1.
- **staticFeats:** a list of explanatory features you whose graphs you want *not* to be trained while modelling. If you wanted to preserve how your model treats Height and Width, you could set staticFeats=["Height", "Width"] and all the other feature effects would be trained while holding those two static. Useful for if a stakeholder has already signed off on part of a model and you want to make limited changes without needing them to re-certify the whole thing. An empty list, by default.
- **prints:** a variable specifying how 'loud' the training process should be. It takes three values: "silent" (output nothing to terminal while training), "normal" (output basic progress metrics while training), and "verbose" (output everything which could concievably be useful while training). Useful for if you want a clean terminal, or if you want a better idea of how the model changes over the course of a run (especially for bughunting). "normal" by default.

## viz_multiplicative_model()
viz_multiplicative_model() has one required parameter:

- **model:** the model being visualized.

It also has the following optional parameters:

- **subFolder:** the subfolder inside the 'graphs' folder which it writes the graphs to. None by default; this just sends the graphs to 'graphs'.
- **targetSpan:** the range of y-values the graphs treat as 'normal'. 0.5 by default; this makes the default range of the graphs' y-axes 0.5 (1-0.5) to 1.5 (1+0.5).
- **otherName:** the name used for categories which don't qualify for a bar of their own in the bar chart. "OTHER" by default.


# Other Simple GLMs

## Poisson
Poisson-distributed multiplicative models are conventional for claims-per-customer Risk models; more generally, for any model predicting how many times something will happen.

To fit a Poisson-distributed GLM, follow the instructions in the above section. The only things you need to change are the model training function (train_poisson_model() instead of train_gamma_model()) and the learning rate (different error functions treat learning differently).

## Tweedie

Tweedie-distributed multiplicative models are conventional for cost-per-customer Risk models; more generally, for any model which is a hybrid of Poisson and Gamma models.

To fit a Tweedie-distributed GLM, follow the instructions in the above section. The only things you need to change are the model training function (train_tweedie_model() instead of train_gamma_model()) and the learning rate (different error functions treat learning differently).

train_tweedie_model has one extra (optional) parameter: pTweedie. When this is 1, train_tweedie_model is just a slower version of train_poisson_model; when it's 2, train_tweedie_model is just a slower version of train_gamma_model; when it's somewhere inbetween, it's somewhere inbetween. pTweedie is 1.5 by default.

## Normal

Normally-distributed (equivalently, squared-error-minimizing) additive models are the default regression model used outside Insurance. These models are an implementation of segmented linear regression in the Durkon paradigm.

When predicting, additive models add to the BASE_VALUE instead of multiplying it.

Note that an additive model has a "featcomb" parameter specifying that it combines feature effects additively and not multiplicatively. Note also that an untrained additive model will have feature effects of 0 ("add 0 to everything!") instead of 1 ("multiply everything by 1!").

To fit a Normally-distributed additive model, follow the instructions in the above section. The only things you need to change are the model prep function (prep_additive_model() instead of prep_model()), the model training function (train_additive_model() instead of train_gamma_model()), and the learning rate (different error functions treat learning differently).

To visualize a Normally-distributed additive model, you should use viz_additive_model() instead of viz_multiplicative_model().

## Logistic

Logistic models are conventional for conversion and retention models; more generally, for any model predicting a category for the response variable.

When predicting, logistic models behave like additive models, but apply {the logistic function} to their predictions after applying all the feature effects.

To get data in the right format, create a response column where entries take a value of 1 if they show the behavior you're trying to predict, and 0 otherwise. For example, if you were trying to use other features to predict whether a cake would be marzipan-flavoured, you could construct a valid response variable "MARZIPAN" as shown below:

```python
trainDf["MARZIPAN"] = (trainDf["Flavour"]=="marzipan").apply(int)
```

To fit a Logistic model, follow the instructions in the above section. The only things you need to change are the prep function (prep_logistic_model() instead of prep_model()), the model training function (train_logistic_model() instead of train_gamma_model()), and the learning rate (different error functions treat learning differently).

To predict with a Logistic model, the predict function needs another parameter to specify the linkage applied after applying feature effects. The Logistic function is applied, and the inverse of a logistic function is a Logit function, so predict(df, model) needs to become predict(df, model, "Logit"). This will produce predictions of the probability that a row takes the value you assigned to the 1 value. For example, a prediction of 0.54 would be the model saying "I think there's a 54% chance this cake will be marzipan-flavoured".

To visualize a Logistic model, you should use viz_logistic_model() instead of viz_multiplicative_model(). This will show you how each feature adds to or subtracts from the quantity which will be fed into the Logit function (denominated here in Logistic Probability Units, or LPUs).

# Saving and Loading

Models can be copied and pasted by hand, so there's no need to save or load. However, it's useful to have these functions available for large models and for automatic archiving.

To save a model, use the function
```python
du.misc.save_model(model)
```
to write it to the folder "models" as a JSON file.

This function has the following optional parameters:

- **name:** a name to be assigned to the model; "model" by default.
- **timing:** a boolean specifying whether the model's title should be timestamped; True by default.

To load a model, use the function
```python
model = du.misc.load_model("MODEL_FILENAME_HERE")
```
# Exporting models

Models can be exported in WTW-style csv format, which can be opened and examined like any other spreadsheet.

To export a model, use
```python
du.export.model_to_lines(model)
```
model_to_lines has the following optional parameters:

- **detail:** the number of interpolated points placed between the points that define the model. 1 by default.
- **filename:** the name of the file the function will output. "op.csv" by default.
