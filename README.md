# lazytransform
Automatically transform all categorical, date-time, NLP variables in your data set to numeric in a single line of code for any data set any size.

![banner](lazy_logo5.png)

# Table of Contents
<ul>
<li><a href="#What is lazytransform">What is lazytransform</a></li>
<li><a href="#How to use lazytransform">How to use lazytransform</a></li>
<li><a href="#How to install">How to install lazytransform</a></li>
<li><a href="#Usage">Usage</a></li>
<li><a href="#api">API</a></li>
<li><a href="#maintainers">Maintainers</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
</ul>
<p>

## What is lazytransform?
`lazytransform` is a new python library for automatically transforming your entire dataset to numeric format using category encoders, NLP text vectorizers and pandas date time processing functions. All in a single line of code!


## How to use lazytransform
`lazytransform` can be used in many ways. Let us look at each of them below.

### 1.  Using lazytransform as a simple pandas data transformation pipeline 

<p>The first method is probably the most popular way to use lazytransform. The transformer within lazytransform can be used to transform and create new features from categorical, date-time and NLP (text) features in your dataset. This transformer pipeline is fully scikit-learn Pipeline compatible and can be used to build even more complex pipelines by you based on `make_pipeline` statement from `sklearn.pipeline` library. Let us see an example:<p>

![lazy_code1](lazy_code1.png)

### 2.  Using lazytransform as a sklearn pipeline with sklearn models or XGBoost or LightGBM models

<p>The second method is a great way to create an entire data transform and model training pipeline. `lazytransform` allows you to send in a model object (only the following are supported) and it will automatically transform, create new features and train a model using sklearn pipelines. This method can be seen as follows:<br>

![lazy_code2](lazy_code2.png)

### 3.  Using lazytransform in GridSearchCV to find the best model pipeline
<p>The third method is a great way to find the best data transformation and model training pipeline using GridSearchCV or RandomizedSearchCV along with a LightGBM or XGBoost or scikit-learn model. This is explained very clearly in the `LazyTransformer_with_GridSearch_Pipeline.ipynb` notebook in the same github here. Make sure you check it out!

![lazy_gridsearch](lazy_gridsearch.png)

<p>
The following models are currently supported:
<ol>
<li>All sklearn models</li>
<li>All MultiOutput models from sklearn.multioutput library</li>
<li>XGboost models</li>
<li>LightGBM models</li>
</ol>
However, you must install and import those models on your own and define them as model variables before passing those variables to lazytransform.

## How to install lazytransform
<p>

**Prerequsites:**
<ol>
<li><b>lazytransform is built using pandas, numpy, scikit-learn, category_encoders and imb-learn libraries.</b> It should run on most Python 3 Anaconda installations without additional installs. You won't have to import any special libraries other than "imb-learn" and "category_encoders".</li>
</ol>
[Anaconda](https://docs.anaconda.com/anaconda/install/)
<p>
On your local machine, it is easy to install lazytransform from PyPi:

```
pip install lazytransform 
```
But on Kaggle Notebooks, you must slightly modify the installation into two steps. If you don't do this, you will get an error!
```
pip install lazytransform --ignore-installed --no-deps
pip install category-encoders --ignore-installed --no-deps

```

To install from source:

```
cd <lazytransform_Destination>
git clone git@github.com:AutoViML/lazytransform.git
```
or download and unzip https://github.com/AutoViML/lazytransform/archive/master.zip
```
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
cd lazytransform
pip install -r requirements.txt
```

## Usage
<p>
You can invoke `lazytransform` as a scikit-learn compatible fit and transform or a fit and predict pipeline. See syntax below.<p>

```
from lazytransform import LazyTransformer
lazy = LazyTransformer(model=None, encoders='auto', scalers=None, 
        date_to_string=False, transform_target=False, imbalanced=False)
```

### if you are not using a model in pipeline, you must use fit and transform
```
X_trainm, y_trainm = lazy.fit_transform(X_train, y_train)
X_testm = lazy.transform(X_test)
```
### If you are using a model in pipeline, use must use fit and predict only
```
lazy = LazyTransformer(model=RandomForestClassifier(), encoders='auto', scalers=None, 
        date_to_string=False, transform_target=False, imbalanced=False)
```

```
lazy.fit(X_train, y_train)
lazy.predict(X_test)
```

## API

<p>
lazytransform has a very simple API with the following inputs. You need to create a sklearn-compatible transformer pipeline object by importing LazyTransformer from lazytransform library. <p>
Once you import it, you can define the object by giving several options such as:

**Arguments**

<b>Caution:</b> X_train and y_train must be pandas Dataframes or pandas Series. DO NOT send in numpy arrays. They won't work.

- `model`: default is None. Or it could be any scikit-learn model (including multioutput models) as well as the popular XGBoost and LightGBM libraries. You need to install those libraries if you want to use them.
- `encoders`: could be one more encoders in a string or a list. Each encoder string can be any one of the 10+ encoders from `category_encoders` library below.  Available encoders are listed here as strings so that you can input them in lazytransform:
  - `auto` - It uses `onehot` encoding for low-cardinality variables and `label` encoding for high cardinality variables.
  - `onehot` - One Hot encoding - it will be used for all categorical features irrespective of cardinality
  - `label` - Label Encoding - it will be used for all categorical features irrespective of cardinality
  - `hashing` or `hash` - Hashing (or Hash) Encoding
  - `helmert` - Helmert Encoding
  - `bdc` - BDC Encoding
  - `sum` - Sum Encoding
  - `loo` - Leave one out Encoding
  - `base` - Base encoding
  - `woe` - Weight of Evidence Encoding
  - `james` - James Encoding
  - `target` - Target Encoding
  - `count` - Count Encoding
  - `glm`,`glmm` - Generalized Linear Model Encoding
- Here is a description of various encoders and their uses from the excellent <a href="https://contrib.scikit-learn.org/category_encoders/"> category_encoders</a> python library:<br>
    - `HashingEncoder`: HashingEncoder is a multivariate hashing implementation with configurable dimensionality/precision. The advantage of this encoder is that it does not maintain a dictionary of observed categories. Consequently, the encoder does not grow in size and accepts new values during data scoring by design.
    - `SumEncoder`: SumEncoder is a Sum contrast coding for the encoding of categorical features.
    - `PolynomialEncoder`: PolynomialEncoder is a Polynomial contrast coding for the encoding of categorical features.
    - `BackwardDifferenceEncoder`: BackwardDifferenceEncoder is a Backward difference contrast coding for encoding categorical variables.
    - `OneHotEncoder`: OneHotEncoder is the traditional Onehot (or dummy) coding for categorical features. It produces one feature per category, each being a binary.
    - `HelmertEncoder`: HelmertEncoder uses the Helmert contrast coding for encoding categorical features.
    - `OrdinalEncoder`: OrdinalEncoder uses Ordinal encoding to designate a single column of integers to represent the categories in your data. Integers however start in the same order in which the categories are found in your dataset. If you want to change the order, just sort the column and send it in for encoding.
    - `FrequencyEncoder`: FrequencyEncoder is a count encoding technique for categorical features. For a given categorical feature, it replaces the names of the categories with the group counts of each category.
    - `BaseNEncoder`: BaseNEncoder encodes the categories into arrays of their base-N representation. A base of 1 is equivalent to one-hot encoding (not really base-1, but useful), a base of 2 is equivalent to binary encoding. N=number of actual categories is equivalent to vanilla ordinal encoding.
    - `TargetEncoder`: TargetEncoder performs Target encoding for categorical features. It supports following kinds of targets: binary and continuous. For multi-class targets it uses a PolynomialWrapper.
    - `CatBoostEncoder`: CatBoostEncoder performs CatBoost coding for categorical features. It supports the following kinds of targets: binary and continuous. For polynomial target support, it uses a PolynomialWrapper. This is very similar to leave-one-out encoding, but calculates the values “on-the-fly”. Consequently, the values naturally vary during the training phase and it is not necessary to add random noise.
    - `WOEEncoder`: WOEEncoder uses the Weight of Evidence technique for categorical features. It supports only one kind of target: binary. For polynomial target support, it uses a PolynomialWrapper. It cannot be used for Regression.
    - `JamesSteinEncoder`: JamesSteinEncoder uses the James-Stein estimator. It supports 2 kinds of targets: binary and continuous. For polynomial target support, it uses PolynomialWrapper.
    For feature value i, James-Stein estimator returns a weighted average of:
    The mean target value for the observed feature value i.
    The mean target value (regardless of the feature value).
    - `QuantileEncoder`: This is a very good encoder for Regression tasks. See Paper and article:
    https://towardsdatascience.com/quantile-encoder-eb33c272411d

- `scalers`: could be one of three main scalers used in scikit-learn models to transform numeric features. Default is None. Scalers are used in the last step of the pipeline to scale all features that have transformed. However, you might want to avoid scaling in NLP datasets since after TFiDF vectorization, scaling them may not make sense. But it is up to you. The 4 options are:
  - `None` No scaler. Great for almost all datasets. Test it first and then try one of the scalers below.
  - `std` standard scaler. Great for almost all datasets.
  - `minmax` minmax scaler. Great for datasets where you need to see the distribution between 0 and 1.
  - `maxabs` max absolute scaler. Great for scaling but leaves the negative values as they are (negative). 
- `date_to_string`: default is False. If you want to use date variables as strings (categorical), then set it as True.You can use this option when there are very few dates in your dataset. If you set it as False, it will convert it into date time format and extract up to 20 features from your date time column. This is the default option and best option.
- `transform_target`: default is False. If you want to transform your target variable(s), then set it as True and we will transform your target(s) as numeric using Label Encoding as well as multi-label Binary classes. This is a great option when you have categorical target variables.
- `imbalanced`: default is False. If you have an imbalanced dataset, then set it to True and we will transform your train data using BorderlineSMOTE or SMOTENC which are both great options. We will select the right SMOTE function automatically.
- `verbose`: This has 3 possible states:
  - `0` silent output. Great for running this silently and getting fast results.
  - `1` more verbiage. Great for knowing how results were and making changes to flags in input.
  - `2` highly verbose output. Great for finding out what happens under the hood in lazytransform pipelines.
<p>
To view the text pipeline, the default display is 'text', do:<br>

```
from sklearn import set_config
set_config(display="text")
lazy.xformer
```

<p>
To view the pipeline in a diagram (visual format), do:<br>

```
from sklearn import set_config
set_config(display="diagram")
lazy.xformer
# If you have a model in the pipeline, do:
lazy.modelformer
```
![lazy_pipe](lazy_pipe.png)

To view the feature importances of the model in the pipeline, you can do:
```
lazy.plot_importance()
```
![lazy_feat_imp](lazy_feat_imp.png)

## Maintainers

* [@AutoViML](https://github.com/AutoViML)

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

## License

Apache License 2.0 © 2020 Ram Seshadri

## Note of Gratitude

This libray would not have been possible without the following great libraries:
<ol>
<li><b>Category Encoders library:</b> Fantastic library https://contrib.scikit-learn.org/category_encoders/index.html</li>
<li><b>Imbalanced Learn library:</b> Another fantastic library https://imbalanced-learn.org/stable/index.html</li>
<li><b>The amazing `lazypredict`</b> was an inspiration for `lazytransform`. You can check out the library here:
https://github.com/shankarpandala/lazypredict
</li>
<li><b>The amazing `Kevin Markham`</b> was another inspiration for lazytransform. You can check out his classes here:
https://www.dataschool.io/about/
</li>
</ol>

## DISCLAIMER
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.

