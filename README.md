# lazytransform
Automatically transform all categorical, date-time, NLP variables in your data set to numeric in a single line of code for any data set any size.

<a href="https://ibb.co/JpnTdC1"><img src="https://i.ibb.co/09qLXQ7/lazy-logo5.png" alt="lazy-logo5" border="0"></a>

# Table of Contents
<ul>
<li><a href="#introduction">What is lazytransform</a></li>
<li><a href="#uses">How to use lazytransform</a></li>
<li><a href="#install">How to install lazytransform</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#api">API</a></li>
<li><a href="#maintainers">Maintainers</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
</ul>
<p>

## Introduction
### What is lazytransform?
`lazytransform` is a new python library for automatically transforming your entire dataset to numeric format using category encoders, NLP text vectorizers and pandas date time processing functions. All in a single line of code!

## Uses
`lazytransform` has two important uses in the Data Science process. It can be used in Feature Engg to transform features or add features (see API below). It can also be used to train and evaluate models in MLOps pipelines with multiple models being trained simultaneusly using the same train/test split and the same feature engg strategies. This way there is absolutely zero or minimal data leakage. 

### 1.  Using lazytransform as a simple pandas data transformation pipeline 

<p>The first method is probably the most popular way to use lazytransform. The transformer within lazytransform can be used to transform and create new features from categorical, date-time and NLP (text) features in your dataset. This transformer pipeline is fully scikit-learn Pipeline compatible and can be used to build even more complex pipelines by you based on `make_pipeline` statement from `sklearn.pipeline` library. <a href="https://github.com/AutoViML/lazytransform/blob/main/Featurewiz_LazyTransform_Demo1.ipynb">Let us see an example</a>:<p>

<a href="https://ibb.co/xfMQnNz"><img src="https://i.ibb.co/ZYhCQ0c/lazy-code1.png" alt="lazy-code1" border="0"></a>

### 2.  Using lazytransform as a sklearn pipeline with sklearn models or XGBoost or LightGBM models

<p>The second method is a great way to create an entire data transform and model training pipeline with absolutely zero data leakage. `lazytransform` allows you to send in a model object (only the following are supported) and it will automatically transform, create new features and train a model using sklearn pipelines. <a href="https://github.com/AutoViML/lazytransform/blob/main/Featurewiz_LazyTransform_Demo2.ipynb">This method can be seen as follows</a>:<br>

<a href="https://ibb.co/T1WNhzT"><img src="https://i.ibb.co/0KszJPX/lazy-code2.png" alt="lazy-code2" border="0"></a>

### 3.  Using lazytransform in GridSearchCV to find the best model pipeline
<p>The third method is a great way to find the best data transformation and model training pipeline using GridSearchCV or RandomizedSearchCV along with a LightGBM or XGBoost or scikit-learn model. This is explained very clearly in the <a href="https://github.com/AutoViML/lazytransform/blob/main/LazyTransformer_with_GridSearch_Pipeline.ipynb">LazyTransformer_with_GridSearch_Pipeline.ipynb</a> notebook in the same github here. Make sure you check it out!

<a href="https://ibb.co/WGvnqjs"><img src="https://i.ibb.co/xXqhPd3/lazy-gridsearch.png" alt="lazy-gridsearch" border="0"></a><br />

<p>
The following models are currently supported:
<ol>
<li>All sklearn models</li>
<li>All MultiOutput models from sklearn.multioutput library</li>
<li>XGboost models</li>
<li>LightGBM models</li>
</ol>
However, you must install and import those models on your own and define them as model variables before passing those variables to lazytransform.

## Install
<p>

**Prerequsites:**
<ol>
<li><b>lazytransform is built using pandas, numpy, scikit-learn, category_encoders and imb-learn libraries.</b> It should run on most Python 3 Anaconda installations without additional installs. You won't have to import any special libraries other than "imb-learn" and "category_encoders".</li>
</ol>
The best method to install lazytransform is to use conda:<p>

```
conda install -c conda-forge lazytransform
```
<a href="https://ibb.co/fXnbPd6"><img src="https://i.ibb.co/qDWzPYq/conda-install.png" alt="conda-install" border="0"></a><br>
The second best installation method is to use "pip install".

```
pip install lazytransform 
```
Alert! When using Colab or Kaggle Notebooks, you must slightly modify installation. If you don't do this, you will get weird errors in those platforms!

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
        date_to_string=False, transform_target=False, imbalanced=False,
        combine_rare=False, verbose=0)
```

### if you are not using a model in pipeline, you must use fit and transform
```
X_trainm, y_trainm = lazy.fit_transform(X_train, y_train)
X_testm = lazy.transform(X_test)
```
### If you are using a model in pipeline, use must use fit and predict only
```
lazy = LazyTransformer(model=RandomForestClassifier(), encoders='auto', scalers=None, 
        date_to_string=False, transform_target=False, imbalanced=False,
        combine_rare=False, verbose=0)
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
  - `hashing` or `hash` - Hashing (or Hash) Encoding - will be used for all categorical variables
  - `helmert` - Helmert Encoding - will be used for all categorical variables
  - `bdc` - BDC Encoding - will be used for all categorical variables
  - `sum` - Sum Encoding - will be used for all categorical variables
  - `loo` - Leave one out Encoding - will be used for all categorical variables
  - `base` - Base encoding - will be used for all categorical variables
  - `woe` - Weight of Evidence Encoding - will be used for all categorical variables
  - `james` - James Encoding - will be used for all categorical variables
  - `target` - Target Encoding - will be used for all categorical variables
  - `count` - Count Encoding - will be used for all categorical variables
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
  - `robust` Robust scaler. Great for datasets where you have outliers.
  - `maxabs` max absolute scaler. Great for scaling but leaves the negative values as they are (negative). 
- `date_to_string`: default is False. If you want to use date variables as strings (categorical), then set it as True.You can use this option when there are very few dates in your dataset. If you set it as False, it will convert it into date time format and extract up to 20 features from your date time column. This is the default option and best option.
- `transform_target`: default is False. If you want to transform your target variable(s), then set it as True and we will transform your target(s) as numeric using Label Encoding as well as multi-label Binary classes. This is a great option when you have categorical target variables.
- `imbalanced`: default is False. If you have an imbalanced dataset, then set it to True and we will transform your train data using BorderlineSMOTE or SMOTENC which are both great options. We will select the right SMOTE function automatically.
- `combine_rare`: default is False. This is a great option if you have too many rare categories in your categorical variables. It will automatically combine those categories which are less than 1% of the dataset into one combined category called "rare_categories". You can also set it to False and we will not transform.
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
<a href="https://imgbb.com/"><img src="https://i.ibb.co/Bn7V4px/lazy-pipe.png" alt="lazy-pipe" border="0"></a>

To view the feature importances of the model in the pipeline, you can do:
```
lazy.plot_importance()
```
<a href="https://ibb.co/jhpsVtJ"><img src="https://i.ibb.co/cJmVbqY/lazy-feat-imp.png" alt="lazy-feat-imp" border="0"></a>

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

