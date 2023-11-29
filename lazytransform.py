#!/usr/bin/env python
# coding: utf-8
############################################################################################
# Copyright [2022] [Ram Seshadri]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###########################################################################################
##### LazyTransformer is a simple transformer pipeline for all kinds of ML problems. ######
###########################################################################################
# What does this pipeline do? Here's the major steps:
# 1. Takes categorical variables and encodes them using my special label encoder which can 
#    handle NaNs and future categories
# 1. Takes integer and float variables and imputes them using a simple imputer (with a default)
# 1. Takes NLP and time series (as string) variables and vectorizes them using TFiDF
# 1. Takes pandas date-time (actual date-time) variables and extracts more features from them
# 1. Completely Normalizes all of the above using AbsMaxScaler which preserves the 
#     relationship of label encoded vars
# 1. Optionally adds any model to pipeline so the entire pipeline can be fed to a 
#     cross validation scheme (just set model=any_sklearn_model() when defining the pipeline)
# The initial results from this pipeline in real world data sets is promising indeed.
############################################################################################
####### This Transformer is inspired by Kevin Markham's class on Scikit-Learn pipelines. ###
#######   You can sign up for his Data School class here: https://www.dataschool.io/  ######
############################################################################################
import numpy as np
import pandas as pd
np.random.seed(99)
import random
random.seed(42)
################################################################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import pdb
from collections import defaultdict, Counter
### These imports give fit_transform method for free ###
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import column_or_1d
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, mean_squared_error,balanced_accuracy_score
import gc
##############  These imports are to make trouble shooting easier #####
import time
import copy
import os
import pickle
import scipy
import warnings
warnings.filterwarnings("ignore")
##########################################################
from category_encoders import HashingEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders import HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
#from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
################################################################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################
import math
from collections import Counter
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import time
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingRegressor, VotingClassifier
import copy
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.base import ClassifierMixin, RegressorMixin
import lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.multioutput import ClassifierChain, RegressorChain
import scipy as sp
import pdb
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
#from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import cycle
import matplotlib.pyplot as plt

#########################################################
class My_LabelEncoder(BaseEstimator, TransformerMixin):
    """
    ################################################################################################
    ######     The My_LabelEncoder class works just like sklearn's Label Encoder but better! #######
    #####  It label encodes any object or category dtype in your dataset. It also handles NaN's.####
    ##  The beauty of this function is that it takes care of encoding unknown (future) values. #####
    ##################### This is the BEST working version - don't mess with it!! ##################
    ################################################################################################
    Usage:
          le = My_LabelEncoder()
          le.fit_transform(train[column]) ## this will give your transformed values as an array
          le.transform(test[column]) ### this will give your transformed values as an array
              
    Usage in Column Transformers and Pipelines:
          No. It cannot be used in pipelines since it need to produce two columns for the next stage in pipeline.
          See my other module called My_LabelEncoder_Pipe() to see how it can be used in Pipelines.
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)
        self.max_val = 0
        
    def fit(self,testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ## testx must still be a pd.Series for this encoder to work!
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            #### Do not change this since I have tested it and it works.
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the object as is
                return self
        ins = np.unique(testx.factorize()[1]).tolist()
        outs = np.unique(testx.factorize()[0]).tolist()
        #ins = testx.value_counts(dropna=False).index        
        if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
            if not np.nan in ins:
                ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs))
        self.inverse_transformer = dict(zip(outs,ins))
        return self

    def transform(self, testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ## testx must still be a pd.Series for this encoder to work!
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            #### Do not change this since I have tested it and it works.
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                #### Do not change this since I have tested it and it works.
                return testx
        ### now convert the input to transformer dictionary values
        new_ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in new_ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                self.transformer[each_missing] = int(self.max_val + 1)
                self.inverse_transformer[int(self.max_val+1)] = each_missing
                self.max_val = int(self.max_val+1)
        else:
            self.max_val = np.max(list(self.transformer.values()))
        ### To handle category dtype you must do the next step #####
        #### Do not change this since I have tested it and it works.
        testk = testx.map(self.transformer) 
        if testx.dtype not in [np.int16, np.int32, np.int64, float, bool, object]:
            if testx.isnull().sum().sum() > 0:
                fillval = self.transformer[np.nan]
                testk = testx.cat.add_categories([fillval])
                testk = testk.fillna(fillval)
                testk = testx.map(self.transformer).values.astype(int)
                return testk
            else:
                testk = testx.map(self.transformer).values.astype(int)
                return testk
        else:
            outs = testx.map(self.transformer).values.astype(int)
            return outs

    def inverse_transform(self, testx, y=None):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################
class My_LabelEncoder_Pipe(BaseEstimator, TransformerMixin):
    """
    ################################################################################################
    ######  The My_LabelEncoder_Pipe class works just like sklearn's Label Encoder but better! #####
    #####  It label encodes any cat var in your dataset. But it can also be used in Pipelines! #####
    ##  The beauty of this function is that it takes care of NaN's and unknown (future) values.#####
    #####  Since it produces an unused second column it can be used in sklearn's Pipelines.    #####
    #####  But for that you need to add a drop_second_col() function to this My_LabelEncoder_Pipe ## 
    #####  and then feed the whole pipeline to a Column_Transformer function. It is very easy. #####
    ##################### This is the BEST working version - don't mess with it!! ##################
    ################################################################################################
    Usage in pipelines:
          le = My_LabelEncoder_Pipe()
          le.fit_transform(train[column]) ## this will give you two columns - beware!
          le.transform(test[column]) ### this will give you two columns - beware!
              
    Usage in Column Transformers:
        def drop_second_col(Xt):
        ### This deletes the 2nd column. Hence col number=1 and axis=1 ###
        return np.delete(Xt, 1, 1)
        
        drop_second_col_func = FunctionTransformer(drop_second_col)
        
        le_one = make_pipeline(le, drop_second_col_func)
    
        ct = make_column_transformer(
            (le_one, catvars[0]),
            (le_one, catvars[1]),
            (imp, numvars),
            remainder=remainder)    

    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)
        self.max_val = 0
        
    def fit(self,testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                return self
        ins = np.unique(testx.factorize()[1]).tolist()
        outs = np.unique(testx.factorize()[0]).tolist()
        #ins = testx.value_counts(dropna=False).index        
        if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
            if not np.nan in ins:
                ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs))
        self.inverse_transformer = dict(zip(outs,ins))
        return self

    def transform(self, testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ## testx must still be a pd.Series for this encoder to work!
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            #### Do not change this since I have tested it and it works.
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                #### Do not change this since I have tested it and it works.
                return testx

        ### now convert the input to transformer dictionary values
        new_ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in new_ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                self.transformer[each_missing] = int(self.max_val + 1)
                self.inverse_transformer[int(self.max_val+1)] = each_missing
                self.max_val = int(self.max_val+1)
        else:
            self.max_val = np.max(list(self.transformer.values()))
        ### To handle category dtype you must do the next step #####
        #### Do not change this since I have tested it and it works.
        testk = testx.map(self.transformer) 
        
        if testx.isnull().sum().sum() > 0:
            try:
                fillval = self.transformer[np.nan]
            except:
                fillval = -1
            if testx.dtype not in [np.int16, np.int32, np.int64, float, bool, object]:
                testk = testk.map(self.transformer).fillna(fillval).values.astype(int)
            else:
                testk = testk.fillna(fillval)
                testk = testx.map(self.transformer).values.astype(int)
            return np.c_[testk,np.zeros(shape=testk.shape)].astype(int)
        else:
            testk = testx.map(self.transformer).values.astype(int)
            return np.c_[testk,np.zeros(shape=testk.shape)].astype(int)

    def inverse_transform(self, testx, y=None):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################
# ## First you need to classify variables into different types. Only then can you do the correct transformations.
def classify_vars_pandas(df, verbose=0):
    """
    ################################################################################################
    Pandas select_dtypes makes classifying variables a breeze. You should check out the links here:
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
    ################################################################################################
    """
    ### number of categories in a column below which it is considered a categorial var ##
    category_limit = 30 
    nlp_threshold = 50
    var_dict = defaultdict(list)
    #### To select all numeric types, use np.number or 'number'
    intvars = df.select_dtypes(include='integer').columns.tolist()
    ### Because of differences in pandas versions, floats don't get detected easily
    ###  Hence I am forced to write clumsily like this. Don't change the next line!!
    floatvars = df.select_dtypes(include='float16').columns.tolist() + df.select_dtypes(
                include='float32').columns.tolist() + df.select_dtypes(include='float64').columns.tolist()
    inf_cols = EDA_find_remove_columns_with_infinity(df)
    numvars = left_subtract(floatvars, inf_cols)
    var_dict['continuous_vars'] = floatvars
    var_dict['int_vars'] = intvars
    #### To select strings you must use the object dtype, but note that this will return all object dtype columns
    stringvars = df.select_dtypes(include='object').columns.tolist()
    discrete_vars = []
    if len(stringvars) > 0:
        copy_string_vars = copy.deepcopy(stringvars)
        for each_string in copy_string_vars:
            if len(df[each_string].unique()) > category_limit:
                discrete_vars.append(each_string)
                stringvars.remove(each_string)
    var_dict['discrete_string_vars'] = discrete_vars
    #### To select datetimes, use np.datetime64, 'datetime' or 'datetime64'
    datevars = df.select_dtypes(include='datetime').columns.tolist()
    var_dict['date_vars'] = datevars
    #### To select timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'
    deltavars = df.select_dtypes(include='timedelta').columns.tolist()
    var_dict['time_deltas'] = deltavars
    #### To select Pandas categorical dtypes, use 'category'
    catvars = df.select_dtypes(include='category').columns.tolist()
    if len(catvars) > 0:
        copy_catvars = copy.deepcopy(catvars)
        for each_cat in copy_catvars:
            if len(df[each_cat].unique()) > category_limit:
                discrete_vars.append(each_cat)
                catvars.remove(each_cat)    
    var_dict['categorical_vars'] = catvars + stringvars
    var_dict['discrete_string_vars'] = discrete_vars
    #### To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0) or 'datetime64[ns, tz]'
    datezvars = df.select_dtypes(include='datetimetz').columns.tolist()
    var_dict['date_zones'] = datezvars
    #### check for nlp variables here ####
    str_vars = var_dict['discrete_string_vars']
    copy_vars = copy.deepcopy(str_vars)
    nlp_vars = []
    for each_str_var in copy_vars:
        try:
            mean_str_size = df[each_str_var].fillna('missing').map(len).max()
        except:
            print('Removing column %s since it is erroring probably due to mixed data types.' %each_str_var)
            str_vars.remove(each_str_var)
            continue
        if  mean_str_size >= nlp_threshold:
            print(f"    since {each_str_var}'s max string size {mean_str_size:.0f} >= {nlp_threshold}, re-classifying it as NLP variable")
            nlp_vars.append(each_str_var)
            str_vars.remove(each_str_var)
    var_dict['nlp_vars'] = nlp_vars ### these are NLP text variables ##########
    var_dict['discrete_string_vars'] = str_vars ### these are high cardinality string variables ##

    if verbose:
        print(f"""    Returning dictionary for variable types with following keys:
                        continuous_vars = {len(floatvars)}, int_vars = {len(intvars)}, 
                        discrete_string_vars = {len(str_vars)}, nlp_vars = {len(nlp_vars)},
                        date_vars = {len(datevars)}, time_deltas = {len(deltavars)},
                        categorical_vars = {len(catvars)+len(stringvars)}, date_zones = {len(datezvars)}""")
    
    return var_dict
##################################################################################################
def left_subtract(l1,l2):
    """This handy function subtracts one list from another. Probably my most popular function."""
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
############################################################################################
import copy
def convert_mixed_datatypes_to_string(df):
    """
    #####################################################################################
    Handy function for Feature Transformation: That's why it's included in LazyTransform
    ########### It converts all mixed data type columns into object columns ############
    Inputs:
    df : pandas dataframe

    Outputs:
    df: this is the transformed DataFrame with all mixed data types now as objects
    #####################################################################################
    """
    df = copy.deepcopy(df)
    cols = df.columns.tolist()
    copy_cols = copy.deepcopy(cols)
    for col in copy_cols:
        if len(df[col].apply(type).value_counts()) > 1:
            print('Mixed data type detected in %s column. Converting it to object type...' %col)
            try:
                df[col] = df[col].map(lambda x: x if isinstance(x, str) else str(x)).values
                if len(df[col].apply(type).value_counts()) > 1:
                    df.drop(col,axis=1,inplace=True)
                    print('    %s still has mixed data types. Dropping it.' %col)
            except:
                df.drop(col,axis=1,inplace=True)
                print('    dropping %s since it gives error.' %col)
    return df
############################################################################################
def convert_all_object_columns_to_numeric(train, test=""):
    """
    This a handy function for Feature Engineering - That's why I have included it in Lazy Transform
    ######################################################################################
    This is a utility that converts string columns to numeric using MY_LABEL ENCODER.
    Make sure test and train have the same number of columns. If you have target in train,
    remove it before sending it through this utility. Otherwise, might blow up during test transform.
    The beauty of My_LabelEncoder is it handles NA's and future values in test that are not in train.
    #######################################################################################
    Inputs:
    train : pandas dataframe
    test: (optional) pandas dataframe

    Outputs:
    train: this is the transformed DataFrame
    test: (optional) this is the transformed test dataframe if given.
    ######################################################################################
    """
    
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)
    #### This is to fill all numeric columns with a missing number ##########
    nums = train.select_dtypes('number').columns.tolist()
    if len(nums) == 0:
        pass
    else:

        if train[nums].isnull().sum().sum() > 0:
            null_cols = np.array(nums)[train[nums].isnull().sum()>0].tolist()
            for each_col in null_cols:
                new_missing_col = each_col + '_Missing_Flag'
                train[new_missing_col] = 0
                train.loc[train[each_col].isnull(),new_missing_col]=1
                train[each_col] = train[each_col].fillna(-9999)
                if not train[each_col].dtype in [np.float64,np.float32,np.float16]:
                    train[each_col] = train[each_col].astype(int)
                if not isinstance(test, str):
                    if test is None:
                        pass
                    else:
                        new_missing_col = each_col + '_Missing_Flag'
                        test[new_missing_col] = 0
                        test.loc[test[each_col].isnull(),new_missing_col]=1
                        test[each_col] = test[each_col].fillna(-9999)
                        if not test[each_col].dtype in [np.float64,np.float32,np.float16]:
                            test[each_col] = test[each_col].astype(int)
    ###### Now we convert all object columns to numeric ##########
    lis = []
    lis = train.select_dtypes('object').columns.tolist() + train.select_dtypes('category').columns.tolist()
    if not isinstance(test, str):
        if test is None:
            pass
        else:
            lis_test = test.select_dtypes('object').columns.tolist() + test.select_dtypes('category').columns.tolist()
            if len(left_subtract(lis, lis_test)) > 0:
                ### if there is an extra column in train that is not in test, then remove it from consideration
                lis = copy.deepcopy(lis_test)
    if not (len(lis)==0):
        for everycol in lis:
            MLB = My_LabelEncoder()
            try:
                train[everycol] = MLB.fit_transform(train[everycol])
                if not isinstance(test, str):
                    if test is None:
                        pass
                    else:
                        test[everycol] = MLB.transform(test[everycol])
            except:
                print('Error converting %s column from string to numeric. Continuing...' %everycol)
                continue
    return train, test
################################################################################################
def drop_second_col(Xt): 
    ### This deletes the 2nd column. Hence col number=1 and axis=1 ###
    
    return np.delete(Xt, 1, 1)

def change_col_to_string(Xt): 
    ### This converts the input column to a string and returns it ##
    return Xt.astype(str)

def create_column_names(Xt, nlpvars=[], catvars=[], discretevars=[], floatvars=[], intvars=[],
                datevars=[], onehot_dict={}, colsize_dict={},datesize_dict={}):
    
    cols_nlp = []
    ### This names all the features created by the NLP column. Hence col number=1 and axis=1 ###
    for each_nlp in nlpvars:
        colsize = colsize_dict[each_nlp]
        nlp_add = [each_nlp+'_'+str(x) for x in range(colsize)]
        cols_nlp += nlp_add
    ### this is for discrete column names ####
    cols_discrete = []
    for each_discrete in discretevars:
        ### for anything other than one-hot we should just use label encoding to make it simpler ##
        discrete_add = each_discrete+'_encoded'
        cols_discrete.append(discrete_add)
    ## do the same for datevars ###
    cols_date = []
    for each_date in datevars:
        #colsize = datesize_dict[each_date]
        #date_add = [each_date+'_'+str(x) for x in range(colsize)]
        date_add = datesize_dict[each_date]
        cols_date += date_add

    #### this is where we put all the column names together #######
    cols_names = catvars+cols_discrete+intvars+cols_date
    num_vars = cols_nlp+floatvars
    num_len = len(num_vars)

    ### Xt is a Sparse matrix array, we need to convert it  to dense array ##
    if scipy.sparse.issparse(Xt):
        Xt = Xt.toarray()

    ### Xt is already a dense array, no need to convert it ##
    if num_len == 0:
        Xint = pd.DataFrame(Xt[:,:], columns = cols_names, dtype=np.int16)
        return Xint
    else:
        Xint = pd.DataFrame(Xt[:,:-num_len], columns = cols_names, dtype=np.int16)
        Xnum = pd.DataFrame(Xt[:,-num_len:], columns = num_vars, dtype=np.float32)
        #### this is where we put all the column names together #######
        df = pd.concat([Xint, Xnum], axis=1)
        return df
#############################################################################################################
import collections
def make_column_names_unique(cols):
    ser = pd.Series(cols)
    ### This function removes all special chars from a list ###
    remove_special_chars =  lambda x:re.sub('[^A-Za-z0-9_]+', '', x)
    
    newls = ser.map(remove_special_chars).values.tolist()
    ### there may be duplicates in this list - we need to make them unique by randomly adding strings to name ##
    seen = [item for item, count in collections.Counter(newls).items() if count > 1]
    cols = [x+str(random.randint(1,1000)) if x in seen else x for x in newls]
    return cols
#############################################################################################################
def create_column_names_onehot(Xt, nlpvars=[], catvars=[], discretevars=[], floatvars=[], intvars=[],
                        datevars=[], onehot_dict={},
                        colsize_dict={}, datesize_dict={}):
    ### This names all the features created by the NLP column. Hence col number=1 and axis=1 ###
    ### Once you get back names of one hot encoded columns, change the column names
    cols_cat = []
    x_cols = []
    
    for each_cat in catvars:
        categs = onehot_dict[each_cat]
        if isinstance(categs, str):
            if categs == 'label':
                cat_add = each_cat+'_encoded'
                cols_cat.append(cat_add)
            else:
                categs = [categs]
                cat_add = [each_cat+'_'+str(categs[i]) for i in range(len(categs))]
                cols_cat += cat_add
        else:
            cat_add = [each_cat+'_'+str(categs[i]) for i in range(len(categs))]
            cols_cat += cat_add
    
    cols_cat = make_column_names_unique(cols_cat)
    
    cols_discrete = []
    discrete_add = []
    for each_discrete in discretevars:
        ### for anything other than one-hot we should just use label encoding to make it simpler ##
        try:
            categs = onehot_dict[each_discrete]
            if isinstance(categs, str):
                if categs == 'label':
                    discrete_add = each_discrete+'_encoded'
                    cols_discrete.append(discrete_add)
                else:
                    categs = [categs]
                    discrete_add = [each_discrete+'_'+x for x in categs]
                    cols_discrete += discrete_add
            else:
                discrete_add = [each_discrete+'_'+x for x in categs]
                cols_discrete += discrete_add
        except:
            ### if there is no new var to be created, just use the existing discrete vars itself ###
            cols_discrete.append(each_discrete)
    cols_discrete = make_column_names_unique(cols_discrete)
    
    cols_nlp = []
    nlp_add = []
    for each_nlp in nlpvars:
        colsize = colsize_dict[each_nlp]
        nlp_add = [each_nlp+'_'+str(x) for x in range(colsize)]
        cols_nlp += nlp_add
    ## do the same for datevars ###
    cols_date = []
    date_add = []
    for each_date in datevars:
        #colsize = datesize_dict[each_date]
        #date_add = [each_date+'_'+str(x) for x in range(colsize)]
        date_add = datesize_dict[each_date]
        cols_date += date_add
    ### Remember don't combine the next 2 lines into one. That will be a disaster.
    ### Pandas infers data types autmatically and they always are float64. So
    ###  to avoid that I have split the data into two or three types 
    cols_names = cols_cat+cols_discrete+intvars+cols_date
    num_vars = cols_nlp+floatvars
    num_len = len(num_vars)
    
    ### Xt is a Sparse matrix array, we need to convert it  to dense array ##
    if scipy.sparse.issparse(Xt):
        Xt = Xt.toarray()
    ### Xt is already a dense array, no need to convert it ##
    ### Remember don't combine the next 2 lines into one. That will be a disaster.
    ### Pandas infers data types autmatically and they always are float64. So
    ###  to avoid that I have split the data into two or three types 
    if num_len == 0:
        Xint = pd.DataFrame(Xt[:,:], columns = cols_names, dtype=np.int16)
        return Xint
    else:
        Xint = pd.DataFrame(Xt[:,:-num_len], columns = cols_names, dtype=np.int16)
        Xnum = pd.DataFrame(Xt[:,-num_len:], columns = num_vars, dtype=np.float32)
        #### this is where we put all the column names together #######
        df = pd.concat([Xint, Xnum], axis=1)
        return df
######################################################################################
def find_remove_duplicates(list_of_values):
    """
    # Removes duplicates from a list to return unique values - USED ONLY ONCE
    """
    output = []
    seen = set()
    for value in list_of_values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def convert_ce_to_pipe(Xt):    
    ### This converts a series to a dataframe to make category encoders work in sklearn pipelines ###
    if str(Xt.dtype) == 'category':
        Xtx = Xt.cat.rename_categories(str).values.tolist()
        return pd.DataFrame(Xtx, columns=[Xt.name])
    else:
        Xtx = Xt.fillna('missing')
        return pd.DataFrame(Xtx)
#####################################################################################
#### Regression or Classification type problem
def analyze_problem_type(y_train, target, verbose=0) :  
    y_train = copy.deepcopy(y_train)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    if isinstance(target, str):
        multi_label = False
        string_target = True
    else:
        if len(target) == 1:
            multi_label = False
            string_target = False
        else:
            multi_label = True
            string_target = False
    ####  This is where you detect what kind of problem it is #################
    if string_target:
        ## If target is a string then we should test for dtypes this way #####
        if  y_train.dtype in ['int64', 'int32','int16']:
            if len(np.unique(y_train)) <= 2:
                model_class = 'Binary_Classification'
            elif len(y_train.unique()) > 2 and len(y_train.unique()) <= cat_limit:
                model_class = 'Multi_Classification'
            else:
                model_class = 'Regression'
        elif  y_train.dtype in ['float16','float32','float64']:
            if len(y_train.unique()) <= 2:
                model_class = 'Binary_Classification'
            elif len(y_train.unique()) > 2 and len(y_train.unique()) <= float_limit:
                model_class = 'Multi_Classification'
            else:
                model_class = 'Regression'
        else:
            if len(y_train.unique()) <= 2:
                model_class = 'Binary_Classification'
            else:
                model_class = 'Multi_Classification'
    else:
        for i in range(y_train.shape[1]):
            ### if target is a list, then we should test dtypes a different way ###
            if y_train.dtypes.values.all() in ['int64', 'int32','int16']:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train.iloc[:,0])) > 2 and len(np.unique(y_train.iloc[:,0])) <= cat_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            elif  y_train.dtypes.values.all() in ['float16','float32','float64']:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train.iloc[:,0])) > 2 and len(np.unique(y_train.iloc[:,0])) <= float_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            else:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                else:
                    model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose:
        if multi_label:
            print('''#### %s %s problem ####''' %('Multi_Label', model_class))
        else:
            print('''#### %s %s problem ####''' %('Single_Label', model_class))
    return model_class, multi_label
####################################################################################################
import copy
import time
from dateutil.relativedelta import relativedelta
from datetime import date
def _create_ts_features(df, verbose=0):
    """
    This takes in input a dataframe and a date variable.
    It then creates time series features using the pandas .dt.weekday kind of syntax.
    It also returns the data frame of added features with each variable as an integer variable.
    """
    df = copy.deepcopy(df)
    tscol = df.name
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    elif isinstance(df, np.ndarray):
        print('    input cannot be a numpy array for creating date-time features. Returning')
        return df
    dt_adds = []
    ##### This is where we add features one by one #########
    try:
        df[tscol+'_hour'] = df[tscol].dt.hour.fillna(0).astype(int)
        df[tscol+'_minute'] = df[tscol].dt.minute.fillna(0).astype(int)
        dt_adds.append(tscol+'_hour')
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating hour-second derived features. Continuing...')
    try:
        df[tscol+'_dayofweek'] = df[tscol].dt.dayofweek.fillna(0).astype(int)
        dt_adds.append(tscol+'_dayofweek')
        if tscol+'_hour' in dt_adds:
            DAYS = dict(zip(range(7),['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']))
            df[tscol+'_dayofweek'] = df[tscol+'_dayofweek'].map(DAYS)
            df.loc[:,tscol+'_dayofweek_hour_cross'] = df[tscol+'_dayofweek'] +" "+ df[tscol+'_hour'].astype(str)
            dt_adds.append(tscol+'_dayofweek_hour_cross')
        df[tscol+'_quarter'] = df[tscol].dt.quarter.fillna(0).astype(int)
        dt_adds.append(tscol+'_quarter')
        df[tscol+'_month'] = df[tscol].dt.month.fillna(0).astype(int)
        MONTHS = dict(zip(range(1,13),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                    'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
        df[tscol+'_month'] = df[tscol+'_month'].map(MONTHS)
        dt_adds.append(tscol+'_month')
        #### Add some features for months ########################################
        festives = ['Oct','Nov','Dec']
        name_col = tscol+"_is_festive"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in festives else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        summer = ['Jun','Jul','Aug']
        name_col = tscol+"_is_summer"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in summer else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        winter = ['Dec','Jan','Feb']
        name_col = tscol+"_is_winter"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in winter else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        cold = ['Oct','Nov','Dec','Jan','Feb','Mar']
        name_col = tscol+"_is_cold"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in cold else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        warm = ['Apr','May','Jun','Jul','Aug','Sep']
        name_col = tscol+"_is_warm"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in warm else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        #########################################################################
        if tscol+'_dayofweek' in dt_adds:
            df.loc[:,tscol+'_month_dayofweek_cross'] = df[tscol+'_month'] +" "+ df[tscol+'_dayofweek']
            dt_adds.append(tscol+'_month_dayofweek_cross')
        df[tscol+'_year'] = df[tscol].dt.year.fillna(0).astype(int)
        dt_adds.append(tscol+'_year')
        today = date.today()
        df[tscol+'_age_in_years'] = today.year - df[tscol].dt.year.fillna(0).astype(int)
        dt_adds.append(tscol+'_age_in_years')
        df[tscol+'_dayofyear'] = df[tscol].dt.dayofyear.fillna(0).astype(int)
        dt_adds.append(tscol+'_dayofyear')
        df[tscol+'_dayofmonth'] = df[tscol].dt.day.fillna(0).astype(int)
        dt_adds.append(tscol+'_dayofmonth')
        df[tscol+'_weekofyear'] = df[tscol].dt.weekofyear.fillna(0).astype(int)
        dt_adds.append(tscol+'_weekofyear')
        weekends = (df[tscol+'_dayofweek'] == 'Sat') | (df[tscol+'_dayofweek'] == 'Sun')
        df[tscol+'_typeofday'] = 'weekday'
        df.loc[weekends, tscol+'_typeofday'] = 'weekend'
        dt_adds.append(tscol+'_typeofday')
        if tscol+'_typeofday' in dt_adds:
            df.loc[:,tscol+'_month_typeofday_cross'] = df[tscol+'_month'] +" "+ df[tscol+'_typeofday']
            dt_adds.append(tscol+'_month_typeofday_cross')
    except:
        print('    Error in creating date time derived features. Continuing...')
    if verbose:
        print('    created %d columns from time series %s column' %(len(dt_adds),tscol))
    return df[dt_adds]
##################################################################################
### This wrapper was proposed by someone in Stackoverflow which works well #######
###  Many thanks to: https://stackoverflow.com/questions/63000388/how-to-include-simpleimputer-before-countvectorizer-in-a-scikit-learn-pipeline
##################################################################################
class Make2D:
    """One dimensional wrapper for sklearn Transformers"""
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(np.array(X).reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        return self.transformer.transform(
            np.array(X).reshape(-1, 1)).ravel()

    def inverse_transform(self, X, y=None):
        return self.transformer.inverse_transform(
            np.expand_dims(X, axis=1)).ravel()
##################################################################################
def return_default():
    missing_value = "missing_"+str(random.randint(1,1000))
    return missing_value
##################################################################################
def make_simple_pipeline(X_train, y_train, encoders='auto', scalers='', 
            date_to_string='', save_flag=False, combine_rare_flag=False, verbose=0):
    """
    ######################################################################################################################
    # # This is the SIMPLEST best pipeline for NLP and Time Series problems - Created by Ram Seshadri
    ###### This pipeline is inspired by Kevin Markham's class on Scikit-Learn pipelines. 
    ######      You can sign up for his Data School class here: https://www.dataschool.io/
    ######################################################################################################################
    #### What does this pipeline do. Here's the major steps:
    # 1. Takes categorical variables and encodes them using my special label encoder which can handle NaNs and future categories
    # 2. Takes numeric variables and imputes them using a simple imputer
    # 3. Takes NLP and time series (string) variables and vectorizes them using CountVectorizer
    # 4. Completely standardizing all of the above using AbsMaxScaler which preserves the relationship of label encoded vars
    # 5. Finally adds an RFC or RFR to the pipeline so the entire pipeline can be fed to a cross validation scheme
    #### The results are yet to be THOROUGHLY TESTED but preliminary results OF THIS PIPELINE ARE very promising INDEED.
    ######################################################################################################################
    """
    start_time = time.time()
    ### Now decide which pipeline you want to use ###########
    ##### Now set up all the encoders ##################
    if isinstance(encoders, list):
        basic_encoder = encoders[0]
        encoder = encoders[1]
        if not isinstance(basic_encoder, str) or not isinstance(encoder, str):
            print('encoders must be either string or list of strings. Please check your input and try again.')
            return
    elif isinstance(encoders, str):
        if encoders == 'auto':
            basic_encoder = 'onehot'
            encoder = 'label'
        else:
            basic_encoder = copy.deepcopy(encoders)
            encoder = copy.deepcopy(encoders)
    else:
        print('encoders must be either string or list of strings. Please check your input and try again.')
        return
    if isinstance(X_train, np.ndarray):
        print('X_train input must be a dataframe since we use column names to build data pipelines. Returning')
        return {}, {}
    
    df = pd.concat([X_train, y_train], axis=1)
    if isinstance(y_train, pd.Series):
        target = y_train.name
        targets = [target]
    elif isinstance(y_train, pd.DataFrame):
        target = y_train.columns.tolist()
        targets = copy.deepcopy(target)
    elif isinstance(X_train, np.ndarray):
        print('y_train must be a pd.Series or pd.DataFrame since we use column names to build data pipeline. Returning')
        return {}, {}
    ###### This helps find all the predictor variables 
    cols = X_train.columns.tolist()
    #### Send target variable as it is so that y_train is analyzed properly ###
    modeltype, multi_label = analyze_problem_type(y_train, target, verbose=1)
    print('Shape of dataset: %s. Now we classify variables into different types...' %(X_train.shape,))
    var_dict = classify_vars_pandas(df[cols],verbose)
    #### Once vars are classified bucket them into major types ###
    catvars = var_dict['categorical_vars'] ### these are low cardinality cat variables 
    discretevars = var_dict['discrete_string_vars'] ### these are high cardinality cat variables 
    floatvars = var_dict['continuous_vars']
    intvars = var_dict['int_vars']
    nlpvars = var_dict['nlp_vars']
    datevars = var_dict['date_vars'] + var_dict['time_deltas'] + var_dict['date_zones']
    #### Converting date variable to a string variable if that is requested ###################
    if date_to_string:
        if isinstance(date_to_string, str):
            copy_datevars = [date_to_string]
        else:
            copy_datevars = copy.deepcopy(date_to_string)
        ## if they have given only one string variable 
        for each_date in copy_datevars:
            print('    string variable %s will be treated as NLP var and transformed by TfidfVectorizer' %each_date)
            X_train[each_date] = X_train[each_date].astype(str).values
            if not each_date in nlpvars:
                nlpvars.append(each_date)
            #### Next check to remove them here ###
            if each_date in datevars:
                datevars.remove(each_date)
            elif each_date in discretevars:
                discretevars.remove(each_date)
            elif each_date in catvars:
                catvars.remove(each_date)
    ### Now do the actual processing for datevars ########
    if datevars:
        if verbose:
            print('Date time vars: %s detected. Several features will be extracted...' %datevars)
        else:
            pass
    else:
        if verbose:
            print('    no date time variables detected in this dataset')
        else:
            pass
    ########################################
    ### Set the category encoders here #####
    ########################################
    ### You are going to set Label Encoder as the default encoder since it is most versatile ##
    def default_encoder():
        return My_LabelEncoder()
    encoder_dict = defaultdict(default_encoder)

    ### you must leave drop_invariant = False for catboost since it is not a onhot type encoder. ##
    encoder_dict = {
                    #'onehot': OneHotEncoder(handle_unknown='ignore'), ## this is for sklearn version ##
                    'onehot': OneHotEncoder(handle_unknown='value'), ### this is for category_encoders version
                    'ordinal': OrdinalEncoder(),
                    'hashing': HashingEncoder(n_components=20, drop_invariant=True),
                    'hash': HashingEncoder(n_components=20, drop_invariant=True),
                    'count': CountEncoder(drop_invariant=True),
                    'catboost': CatBoostEncoder(drop_invariant=False),
                    'target': TargetEncoder(min_samples_leaf=3, drop_invariant=True),
                    'glm': GLMMEncoder(drop_invariant=True),
                    'glmm': GLMMEncoder(drop_invariant=True),
                    'sum': SumEncoder(drop_invariant=True),
                    'woe': WOEEncoder(randomized=False, drop_invariant=True),
                    'bdc': BackwardDifferenceEncoder(drop_invariant=True),
                    'bde': BackwardDifferenceEncoder(drop_invariant=True),
                    'loo': LeaveOneOutEncoder(sigma=0.10, drop_invariant=True),
                    'base': BaseNEncoder(base=2, drop_invariant=True),
                    'james': JamesSteinEncoder(drop_invariant=True),
                    'jamesstein': JamesSteinEncoder(drop_invariant=True),
                    'helmert': HelmertEncoder(drop_invariant=True),
                    #'summary': SummaryEncoder(drop_invariant=True, quantiles=[0.25, 0.5, 1.0], m=1.0),
                    'label': My_LabelEncoder(),
                    'auto': My_LabelEncoder(),
                    }

    non_one_hot_list = ['label','james','jamesstein','loo','woe','glmm','glm','target','catboost','count','ordinal']
    ### set the basic encoder for low cardinality vars here ######
    be = encoder_dict[basic_encoder]
    #### These are applied for high cardinality variables ########
    le = encoder_dict[encoder]
    ### How do we make sure that we create one new LE_Pipe for each catvar? Here's one way to do it.
    lep = My_LabelEncoder_Pipe()
    ###### Just a warning in case someone doesn't know about one hot encoding ####
    ### these encoders result in more columns than the original - hence they are considered one hot type ###
    onehot_type_encoders = ['onehot', 'helmert','bdc', 'bde', 'hashing','hash','sum','base', 
                                #'quantile',
                                #'summary'
                                ]

    if basic_encoder in onehot_type_encoders or encoder in onehot_type_encoders:
        if verbose:
            print('    Beware! %s encoding can create hundreds if not 1000s of variables...' %basic_encoder)
        else:
            pass
    ### Now try the other encoders that user has given as input ####################
    if encoder in ['hashing','hash'] or basic_encoder in ['hashing', 'hash']:
        if verbose:
            print('    Beware! Hashing encoders can take a real long time for even small data sets!')
        else:
            pass
    elif encoder in non_one_hot_list or basic_encoder in non_one_hot_list:
        if verbose:
            print('%s encoder selected for transforming all categorical variables' %encoder)
    elif basic_encoder in onehot_type_encoders or encoder in onehot_type_encoders:
        pass
    else:
        print('%s encoder not found in list of encoders. Using auto instead' %encoder)
        encoders = 'auto'
        basic_encoder = 'label'
        encoder = 'label'


    ######################################################################################
    ####          This is where we convert all the encoders to pipeline components    ####
    ######################################################################################
    if verbose:
        print('Using %s and %s as encoders' %(be,le))
    imp_missing = SimpleImputer(strategy='constant', fill_value='missing')
    imp = SimpleImputer(strategy='constant',fill_value=-99)
    convert_ce_to_pipe_func = FunctionTransformer(convert_ce_to_pipe)
    rcc = Rare_Class_Combiner_Pipe()
    if combine_rare_flag:
        print('    combining rare classes for categorical and discrete vars as given...')

    #### lep_one is the basic encoder of cat variables ############
    if basic_encoder == 'label':
        ######  Create a function called drop_second_col that drops the second unnecessary column in My_Label_Encoder
        drop_second_col_func = FunctionTransformer(drop_second_col)
        #### Now combine it with the LabelEncoder to make it run smoothly in a Pipe ##
        if combine_rare_flag:
            ### lep_one uses rare_class_combiner with My_LabelEncoder to first label encode and then drop the second unused column ##
            lep_one = Pipeline([('rare_class_combiner', rcc), ('basic_encoder', lep), ('drop_second_column', drop_second_col_func)])
        else:
            ### lep_one uses My_LabelEncoder to first label encode and then drop the second unused column ##
            lep_one = Pipeline([('basic_encoder', lep), ('drop_second_column', drop_second_col_func)])
    else:
        if combine_rare_flag:
            lep_one = Pipeline([('rare_class_combiner', rcc),('convert_CE_to_pipe', convert_ce_to_pipe_func), ('basic_encoder', be)])
        else:
            lep_one = Pipeline([('convert_CE_to_pipe', convert_ce_to_pipe_func), ('basic_encoder', be)])
    #### lep_two acts as the major encoder of discrete string variables ############
    if encoder == 'label':
        ######  Create a function called drop_second_col that drops the second unnecessary column in My_Label_Encoder
        drop_second_col_func = FunctionTransformer(drop_second_col)
        #### Now combine it with the LabelEncoder to make it run smoothly in a Pipe ##
        if combine_rare_flag:
            ### lep_one uses rare_class_combiner My_LabelEncoder to first label encode and then drop the second unused column ##
            lep_two = Pipeline([('rare_class_combiner', rcc), ('basic_encoder', lep), ('drop_second_column', drop_second_col_func)])
        else:
            ### lep_one uses My_LabelEncoder to first label encode and then drop the second unused column ##
            lep_two = Pipeline([('basic_encoder', lep), ('drop_second_column', drop_second_col_func)])
    else:
        #lep_two = make_pipeline(convert_ce_to_pipe_func, le)
        if combine_rare_flag:
            lep_two = Pipeline([('rare_class_combiner', rcc), ('convert_CE_to_pipe', convert_ce_to_pipe_func), ('basic_encoder', le)])
        else:
            lep_two = Pipeline([('convert_CE_to_pipe', convert_ce_to_pipe_func), ('basic_encoder', le)])
    ####################################################################################
    # CREATE one_dim TRANSFORMER in order to fit between imputer and TFiDF for NLP here ###
    ####################################################################################
    one_dim = Make2D(imp_missing)
    remove_special_chars =  lambda x:re.sub('[^A-Za-z0-9_]+', ' ', x)
    colsize_dict = {}
    ## number of components in SVD ####
    if len(nlpvars) > 0:
        copy_nlps = copy.deepcopy(nlpvars)
        nuniques = []
        for each_nlp in copy_nlps:
            nuniques.append(int(max(2, 3*np.log2(X_train[each_nlp].nunique()))))
        #### find the min and set the TFIDF for that smallest NLP variable ########            
        top_n = np.min(nuniques)
        top_n = int(min(30, top_n*0.1))
        svd_n_iter = int(min(10, top_n*0.1))
        if verbose:
            print('    %d components chosen for TruncatedSVD(n_iter=%d) after TFIDF' %(top_n, svd_n_iter))
        #############   This is where we set defaults for NLP transformers ##########
        if X_train.shape[0] >= 100000:
            tiffd = TfidfVectorizer(strip_accents='unicode',max_features=6000, preprocessor=remove_special_chars)
        elif X_train.shape[0] >= 10000:
            #tiffd = CountVectorizer(strip_accents='unicode',max_features=1000)
            tiffd = TfidfVectorizer(strip_accents='unicode',max_features=3000, preprocessor=remove_special_chars)
            #tiffd = MyTiff(strip_accents='unicode',max_features=300, min_df=0.01)
        else:
            #vect = CountVectorizer(strip_accents='unicode',max_features=100)
            tiffd = TfidfVectorizer(strip_accents='unicode',max_features=1000, preprocessor=remove_special_chars)
            #tiffd = MyTiff(strip_accents='unicode',max_features=300, min_df=0.01)
        ### create a new pipeline with filling with constant missing ##
        tsvd = TruncatedSVD(n_components=top_n, n_iter=svd_n_iter, random_state=3)
        vect = Pipeline([('make_one_dim', one_dim), ('make_tfidf_pipeline', tiffd), ('truncated_svd', tsvd)])
        #vect = make_pipeline(one_dim, tiffd, tsvd)
        ### Similarly you need to create a function that converts all NLP columns to string before feeding to CountVectorizer
        change_col_to_string_func = FunctionTransformer(change_col_to_string)
        ### we try to find the columns created by counttvectorizer ###
        copy_nlp_vars = copy.deepcopy(nlpvars)
        for each_nlp in copy_nlp_vars:
            colsize = vect.fit_transform(X_train[each_nlp]).shape[1]
            colsize_dict[each_nlp] = colsize
        ##### we collect the column size of each nlp variable and feed it to vect_one ##
        #vect_one = make_pipeline(change_col_to_string_func, vect)
        vect_one = Pipeline([('change_col_to_string', change_col_to_string_func), ('tfidf_tsvd_pipeline', vect)])
    #### Now create a function that creates time series features #########
    create_ts_features_func = FunctionTransformer(_create_ts_features)
    #### we need to the same for date-vars #########
    copy_date_vars = copy.deepcopy(datevars)
    datesize_dict = {}
    for each_datecol in copy_date_vars:
        dtx = create_ts_features_func.fit_transform(X_train[each_datecol])
        datesize_dict[each_datecol] = dtx.columns.tolist()
        del dtx
    #### Now we create a pipeline for date-time features as well ####
    olb = OrdinalEncoder()
    mk_dates = Pipeline([('date_time_features', create_ts_features_func), ('ordinal_encoder', olb)])
    ####################################################################################
    ######     C A T E G O R I C A L    E N C O D E R S    H E R E #####################
    ######     we need to create unique column names for one hot variables    ##########
    ####################################################################################
    copy_cat_vars = copy.deepcopy(catvars)
    onehot_dict = defaultdict(return_default)
    ##### This is extremely complicated logic -> be careful before modifying them!
    gc.collect()
    if basic_encoder in onehot_type_encoders:
        for each_catcol in copy_cat_vars:
            copy_lep_one = copy.deepcopy(lep_one)
            if combine_rare_flag:
                rcct = Rare_Class_Combiner_Pipe()
                ### This is to make missing values dont trip up the transformer ####
                X_train[[each_catcol]] = X_train[[each_catcol]].fillna('MISSINGVALUE',axis=1)
                unique_cols = make_column_names_unique(rcct.fit_transform(X_train[each_catcol]).unique().tolist())
                unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                onehot_dict[each_catcol] = unique_cols
            else:
                if basic_encoder == 'onehot':
                    X_train[[each_catcol]] = X_train[[each_catcol]].fillna('MISSINGVALUE',axis=1)
                    unique_cols = X_train[each_catcol].unique().tolist()
                    #unique_cols = np.where(unique_cols==np.nan, 'missing', unique_cols)
                    #unique_cols = np.where(unique_cols == None, 'missing', unique_cols)
                    unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                    unique_cols = make_column_names_unique(unique_cols)
                    onehot_dict[each_catcol] = unique_cols
                else:
                    X_train[[each_catcol]] = X_train[[each_catcol]].fillna('MISSINGVALUE',axis=1)
                    onehot_dict[each_catcol] = copy_lep_one.fit_transform(X_train[each_catcol], y_train).columns.tolist()
    else:
        for each_catcol in copy_cat_vars:
            onehot_dict[each_catcol] = 'label'
    
    
    ### we now need to do the same for discrete variables based on encoder that is selected ##
    ##### This is extremely complicated logic -> be careful before modifying them!
    copy_discrete_vars = copy.deepcopy(discretevars)
    if encoder in onehot_type_encoders:
        for each_discrete in copy_discrete_vars:
            copy_lep_two = copy.deepcopy(lep_two)
            if combine_rare_flag:
                rcct = Rare_Class_Combiner_Pipe()
                X_train[[each_discrete]] = X_train[[each_discrete]].fillna('MISSINGVALUE', axis=1)
                unique_cols = make_column_names_unique(rcct.fit_transform(X_train[each_discrete]).unique().tolist())
                unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                onehot_dict[each_discrete] = unique_cols
            else:
                if encoder == 'onehot':
                    X_train[[each_discrete]] = X_train[[each_discrete]].fillna('MISSINGVALUE', axis=1)
                    unique_cols = X_train[each_discrete].unique().tolist()
                    #unique_cols = np.where(unique_cols==np.nan, 'missing', unique_cols)
                    unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                    unique_cols = make_column_names_unique(unique_cols)
                    onehot_dict[each_discrete] = unique_cols
                else:
                    X_train[[each_discrete]] = X_train[[each_discrete]].fillna('MISSINGVALUE', axis=1)
                    onehot_dict[each_discrete] = copy_lep_two.fit_transform(X_train[each_discrete], y_train).columns.tolist()
    else:
        ### Then mark it as label encoding so it can be handled properly ####
        for each_discrete in copy_discrete_vars:
            onehot_dict[each_discrete] = 'label'
    
    ### if you drop remainder, then leftovervars is not needed.
    remainder = 'drop'
    
    ### If you passthrough remainder, then leftovers must be included 
    #remainder = 'passthrough'

    ####################################################################################
    #############                   S C A L E R S     H E R E             ##############
    ### If you choose StandardScaler or MinMaxScaler, the integer values become stretched 
    ###  as if they are far apart when in reality they are close. So avoid it for now.
    ####################################################################################
    if scalers and verbose:
            print('Caution: ### When you have categorical or date-time vars in data, scaling may not be helpful. ##')
    if scalers=='max' or scalers == 'minmax':
        scaler = MinMaxScaler()
    elif scalers=='standard' or scalers=='std':
        ### You have to set with_mean=False when dealing with sparse matrices ##
        scaler = StandardScaler(with_mean=False)
    elif scalers=='robust':
        scaler = RobustScaler(unit_variance=True)
    elif scalers=='maxabs':
        ### If you choose MaxAbsScaler, then NaNs which were Label Encoded as -1 are preserved as - (negatives). This is fantastic.
        scaler = MaxAbsScaler()
    else:
        if verbose:
            print('    alert: there is no scaler specified. Options are: max, std, robust, maxabs.')
        scalers = ''
    ##########  define numeric vars as combo of float and integer variables    #########
    numvars = intvars + floatvars
    ####################################################################################
    #########          C R E A T I N G      P I P E L I N E      H E R E  ##############
    ### All the imputers work on groups of variables => so they need to be in the ######
    ###    end since they return only 1D arrays. My_LabelEncoder and other        ######
    ###    fit_transformers need 2D arrays since they work on Target labels too.  ######
    ####################################################################################
    init_str = 'make_column_transformer('
    #### lep_one acts as a one-hot encoder of low cardinality categorical variables ########
    middle_str0 = "".join(['(lep_one, catvars['+str(i)+']),' for i in range(len(catvars))])
    #### lep_two acts as a major encoder of high cardinality categorical variables ########
    middle_str1 = "".join(['(lep_two, discretevars['+str(i)+']),' for i in range(len(discretevars))])
    middle_str12 = '(imp, intvars),'
    middle_str2 = "".join(['(mk_dates, datevars['+str(i)+']),' for i in range(len(datevars))])
    ##### Now we can combine the rest of the variables into the pipeline ################
    if nlpvars:
        middle_str3 = "".join(['(vect_one, nlpvars['+str(i)+']),' for i in range(len(nlpvars))])
    else:
        middle_str3 = ''
    end_str = '(imp, floatvars),    remainder=remainder)'
    ### We now put the whole transformer pipeline together ###
    full_str = init_str+middle_str0+middle_str1+middle_str12+middle_str2+middle_str3+end_str
    ct = eval(full_str)
    if verbose >=2:
        print('Check the pipeline creation statement for errors (if any):\n\t%s' %full_str)
    ### Once that is done, we create sequential steps for a pipeline
    if scalers:
        #scaler_pipe = make_pipeline(ct, scaler )
        scaler_pipe = Pipeline([('complete_pipeline', ct), ('scaler', scaler)])
    else:
        ### default is no scaler ##
        scaler_pipe = copy.deepcopy(ct)
    ### The first columns should be whatever is in the Transformer_Pipeline list of columns
    ### Hence they will be 'Sex', "Embarked", "Age", "Fare". Then only other columns that are passed through.
    ### So after the above 4, you will get remainder cols unchanged: "Parch","Name"
    ## So if you do it correctly, you will get the list of names in proper order this way:
    ## first is catvars, then numvars and then leftovervars
    leftovervars = left_subtract(X_train.columns,catvars+nlpvars+datevars+numvars+discretevars)
    if verbose:
        if leftovervars:
            print('    remaining vars %s are in %s status by transformers...' %(leftovervars, remainder))
        else:
            print('    no other vars left in dataset to transform...')
    ### now we create names for the variables transformed by NLP into a function called nlp_col
    params = {"nlpvars": nlpvars,
          "catvars": catvars,
          "discretevars": discretevars,
          "floatvars": floatvars,
          "intvars": intvars,
          "datevars": datevars,
          "onehot_dict": onehot_dict,
          "colsize_dict":colsize_dict,
          "datesize_dict":datesize_dict}
    ### Create a Function using the above function to transform in pipelines ###
    
    if basic_encoder not in onehot_type_encoders and encoder not in onehot_type_encoders:
        nlp_pipe = Pipeline([('NLP', FunctionTransformer(create_column_names, kw_args=params))])
    else:
        ### Use the one-hot anyway since most of the times, you will be doing one-hot encoding ###
        nlp_pipe = Pipeline([('NLP', FunctionTransformer(create_column_names_onehot, kw_args=params))])
    
    #### Chain it together in the above pipeline #########
    data_pipe = Pipeline([('scaler_pipeline', scaler_pipe), ('nlp_pipeline', nlp_pipe)])
    
    ############################################
    #####    S A V E   P I P E L I N E  ########
    ### save the model and or pipeline here ####
    ############################################
    # save the model to disk only if save flag is set to True ##
    if save_flag:
        filename = 'LazyTransformer_pipeline.pkl'
        print('    Data pipeline is saved as: %s in current working directory.' %filename)
        pickle.dump(data_pipe, open(filename, 'wb'))
    difftime = max(1, int(time.time()-start_time))
    print('Time taken to define data pipeline = %s second(s)' %difftime)
    return data_pipe
######################################################################################
#gives fit_transform method for free
from sklearn.base import TransformerMixin 
class MyTiff(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = TfidfVectorizer(*args, **kwargs)
    def fit(self, x, y=None):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=None):
        return self.encoder.transform(x)
########################################################################
import re
class LazyTransformer(TransformerMixin):
    """
    #########################################################################
        #             LazyTransformer                          #
    # LazyTransformer is a simple transformer pipeline for all kinds of ML ##
        #         Created by Ram Seshadri                #
        #            Apache License 2.0                  #
    #########################################################################
    #  This is a simple pipeline for NLP and Time Series problems ##
    #  What does this pipeline do. Here's the major steps:        ##
    # 1. Takes categorical variables and encodes them using my    ##
    #    special label encoder which can handle NaNs and even     ##
    #    future categories that might occur in test but not train ##
    # 1. Takes numeric variables and imputes using simple imputer ##
    # 1. Vectorizes NLP and time series (string) variables and    ##
    #    using CountVectorizer or TfidfVectorizer as the case may be
    # 1. Completely standardizes all above using AbsMaxScaler     ##
    #    which preserves the relationship of label encoded vars   ##
    # 1. Finally adds an RFC or RFR to the pipeline so the entire ##
    #    pipeline can be fed to a cross validation scheme         ##
    #             Try it out and let me know                      ##
    ################################################################
    ## This is not a scikit-learn BaseEstimator, TransformerMixin ##
    ##   But you can use it in all scikit-learn pipelines as is  ###
    ################################################################
    Parameters
    ----------
    X : pandas DataFrame - No numpy arrays are allowed since we need columns.
    y : pandas Series or DataFrame - no numpy arrays are allowed again.
    model : default None. You can send your model to train with the data. 
            Must be an sklearn or multioutput model. You can try other models
            but I have not tested it. It might work.
    encoders : default is "auto". You can leave this as auto and it will 
            automatically choose the right encoder for your dataset. It will
            use one-hot encoding for low cardinality vars and label encoding
            for high cardinality vars. You can also send either one or two encoders
            of your own. It must be a string consisting of one of the following:
            'onehot','ordinal','hashing','hash','count','catboost','target','glm',
             'glmm','sum','woe','bdc','bde','loo','base','james','jamesstein',
             'helmert','quantile','summary', 'label','auto'
    scalers : default is None. You can send one of the following strings:
            'std', 'standard', 'minmax', 'max', 'robust', 'maxabs'
    date_to_string : default is ''. If you want one or more date-time columns which are
            object or string format, you can send those column names here and they will
            be treated as string variables and used as NLP vars for TFIDF. But 
            if you leave it as an empty string then if date-time columns are detected
            they will be automatically converted to a pandas datetime variable and 
            meaningful features will be extracted such as dayoftheweek, from it.
    transform_target : default is False. If True , target column(s) will be 
            converted to numbers and treated as numeric. If False, target(s)
            will be left as they are and not converted.
    imbalanced : default is False. If True, we will try SMOTE if no model is input.
            Alternatively, if a model is given as input, then SMOTE will not be used.
            Instead, we will wrap that given estimator into a
            a Super Learning Optimized (SULO) ensemble model
            called SuloClassifier will will perform stacking that will
            train on your imbalanced data set. If imbalanced=False, your data will be 
            left as is and nothing will be done to it.
    save : default is False. If True, it will save the data and model pipeline in a 
            pickle file in the current working directory under "LazyTransformer_pipeline.pkl"
            file name.
    combine_rare : default is False. If True, it will combine rare categories in your 
            categorical vars and make them in to a single "rare_categories" class. 
            If False, nothing will be done to your categorical vars.
    verbose: If 0, not much will be printed on the terminal. If 1 or 2, lots of 
            steps will be printed on the terminal.

    """
    def __init__(self, model=None, encoders='auto', scalers=None, date_to_string=False, 
                    transform_target=False, imbalanced=False, save=False, combine_rare = False,
                     verbose=0):
        """
        Description of __init__

        Args:
            model=None: Any model esp. scikit-learn model if you want to return pipeline with trained model.
            date_to_string=False: If True, it converts date columns in dataset to string objects
            transform_target=False: Usually models sent as input are sklearn. If not, switch this flag to True.

        """
        self.date_to_string = date_to_string
        self.encoders = encoders
        self.scalers = scalers
        self.imbalanced = imbalanced
        self.verbose = verbose
        
        self.model_input = model
        if not self.model_input:
            self.model_input = None
        self.transform_target = transform_target
        self.fitted = False
        self.save = save
        self.combine_rare = combine_rare

    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"date_to_string": self.date_to_string, "encoders": self.encoders,
            "scalers": self.scalers, "imbalanced": self.imbalanced, 
            "verbose": self.verbose, "transform_target": self.transform_target,
            "model": self.model_input, "combine_rare": self.combine_rare,}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Description of fit method

        Args:
            X (pd.DataFrame): dataset
            y=None (pd.Series or pd.DataFrame): target column(s)

        Returns:
            A Scikit-Learn Pipeline object that you can do the following:
                if modelformer=True, then you can do fit() and predict() with model
                if modelformer=False, then you can do only fit_transform() and predict()
        """
        self.xformer = None
        self.yformer = None
        self.imbalanced_first_done = False
        self.smotex = None
        X = copy.deepcopy(X)
        self.X_index = X.index
        if y is not None:
            self.y_index = y.index
        #X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        
        y = copy.deepcopy(y)
        if y.ndim >= 2:
            modeltype, multi_label = analyze_problem_type(y, target=y.columns.tolist(), verbose=0)
        else:
            modeltype, multi_label = analyze_problem_type(y, target=y.name, verbose=0)
        if modeltype == 'Regression' and self.transform_target:
            print("    Regression models don't need targets to be transformed to numeric...")                 
            self.transform_target = False
        if modeltype == 'Regression' and self.imbalanced:
            print("    Regression models don't need imbalanced flag set to True...")                 
            self.imbalanced = False
        ### if there is a flag and it is a classification, do it ###
        if self.transform_target:
            ### Hence YTransformer converts them before feeding model
            self.yformer = YTransformer()
        #### This is where we build pipelines for X and y #############
        start_time = time.time()
        
        if self.model_input is not None:
            ### If a model is given, then add it to pipeline and fit it ###
            data_pipe = make_simple_pipeline(X, y, encoders=self.encoders, scalers=self.scalers,
                date_to_string=self.date_to_string, save_flag = self.save, 
                combine_rare_flag=self.combine_rare, verbose=self.verbose)
            
            ### There is no YTransformer in this pipeline so targets must be single label only ##
            model_name = str(self.model_input).split("(")[0]            
            if y.ndim >= 2:
                ### In some cases, if y is a DataFrame with one column also, you get these situations.
                if y.shape[1] == 1:
                    ## In this case, y has only one column hence, you can use a model pipeline ##
                    if model_name == '': 
                        print('No model name specified')
                        self.model_input = None
                        ml_pipe = Pipeline([('data_pipeline', data_pipe),])
                    else:
                        ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model_input)])
                elif model_name == '': 
                    print('No model name specified')
                    self.model_input = None
                    ml_pipe = Pipeline([('data_pipeline', data_pipe),])
                elif model_name not in ['MultiOutputClassifier','MultiOutputRegressor']:
                    ### In this case, y has more than 1 column, hence if it is not a multioutput model, give error
                    print('    Alert: Multi-Label problem - make sure your input model can do MultiOutput!')
                    ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model_input)])
                else:
                    ## In this case we have a multi output model. So let's use it ###
                    #ml_pipe = make_pipeline(data_pipe, self.model_input)
                    ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model_input)])
            else:
                ### You don't need YTransformer since it is a simple sklearn model
                #ml_pipe = make_pipeline(data_pipe, self.model_input)
                ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model_input)])
            ##   Now we fit the model pipeline to X and y ###
            try:
                
                #### This is a very important set of statements ####
                if self.transform_target:
                    if y is not None:
                        self.y_index = y.index
                    self.yformer.fit(y)
                    yt = self.yformer.transform(y)
                    print('    transformed target from object type to numeric')

                    if y is not None:
                        yt.index = self.y_index
                    ### Make sure you leave self.model_input as None when there is no model ### 
                    if model_name == '': 
                        self.model_input = None
                    else:
                        ml_pipe.fit(X,yt)
                        self.xformer = ml_pipe
                else:
                    ### Make sure you leave self.model as None when there is no model ### 
                    if model_name == '': 
                        self.model = None
                    else:
                        ml_pipe.fit(X,y)
                        self.xformer = ml_pipe
            except Exception as e:
                print('Erroring due to %s: There may be something wrong with your data types or inputs.' %e)
                self.xformer = ml_pipe
                return self
            print('model pipeline fitted with %s model' %model_name)
            self.fitted = True
        else:
            ### if there is no given model, just use the data_pipeline ##
            data_pipe = make_simple_pipeline(X, y, encoders=self.encoders, scalers=self.scalers,
                date_to_string=self.date_to_string, save_flag = self.save, 
                combine_rare_flag=self.combine_rare, verbose=self.verbose)
            print('No model input given...')
            #### here we check if we should add a model to the pipeline 
            print('Lazy Transformer Pipeline created...')
            if self.transform_target:
                self.yformer.fit(y)
                yt = self.yformer.transform(y)
                print('    transformed target from object type to numeric')
                if y is not None:
                    yt.index = self.y_index
                data_pipe.fit(X,yt)
                self.xformer = data_pipe 
            else:
                data_pipe.fit(X,y)
                self.xformer = data_pipe
            ## we will leave self.model as None ##
            self.fitted = True
        ### print imbalanced ###
        if self.imbalanced:
            print('### Warning! Do not set imbalanced_flag if this is not an imbalanced classification problem! #######')
            try:
                from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
            except:
                print('This function needs Imbalanced-Learn library. Please pip install it first and try again!')
                return
            if isinstance(X, pd.DataFrame):
                var_classes = classify_vars_pandas(X)
                cat_vars = var_classes['categorical_vars']
                if len(cat_vars) > 0:
                    cat_index = [X.columns.tolist().index(x) for x in cat_vars]
                    self.smotex = SMOTENC(categorical_features=cat_index)
                    if self.verbose:
                        print('Imbalanced flag set. SMOTENC with %d categorical vars will be used in pipeline.' %len(cat_vars))
                else:
                    self.smotex = BorderlineSMOTE()
                    if self.verbose:
                        print('Imbalanced flag set. Borderline SMOTE will be added to pipeline.')
            else:
                self.smotex = BorderlineSMOTE()
                if self.verbose:
                    print('Imbalanced flag set. Borderline SMOTE will be added to pipeline.')
        difftime = max(1, int(time.time()-start_time))
        print('    Time taken to fit dataset = %s second(s)' %difftime)
        return self

    def predict(self, X, y=None):
        if self.fitted and self.model_input is not None:
            y_enc = self.xformer.predict(X)
            return y_enc
        else:
            print('Model not fitted or model not provided. Please check your inputs and try again')
            return y

    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        self.X_index = X.index
        
        start_time = time.time()
        if y is None and self.fitted:
            X_enc = self.xformer.transform(X)
            X_enc.index = self.X_index
            ### since xformer only transforms X ###
            difftime = max(1, int(time.time()-start_time))
            print('    Time taken to transform dataset = %s second(s)' %difftime)
            print('    Shape of transformed dataset: %s' %(X_enc.shape,))
            return X_enc
        elif self.fitted and self.model_input is not None:
            print('Error: No transform allowed. You must use fit and predict when using a pipeline with a model.')
            return X, y
        elif not self.fitted:
            print('LazyTransformer has not been fit yet. Fit it first and try again.')
            return X, y
        elif y is not None and self.fitted and self.model_input is None:
            if self.transform_target:
                if y is not None:
                    self.y_index = y.index
                y_enc = self.yformer.transform(y)
                if y is not None:
                    y_enc.index = self.y_index
            else:
                y_enc = y
            X_enc = self.xformer.transform(X)
            X_enc.index = self.X_index
            #### Now check if the imbalanced_flag is True, then apply SMOTE using borderline2 algorithm which works better
            if self.imbalanced_first_done and self.imbalanced:
                pass
            elif not self.imbalanced_first_done and self.imbalanced:
                try:
                    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
                except:
                    print('This function needs Imbalanced-Learn library. Please pip install it first and try again!')
                    return
                sm = self.smotex
                X_enc, y_enc = sm.fit_resample(X_enc, y_enc)
                self.imbalanced_first_done = True
                self.smotex = sm
                if self.verbose:
                    print('    SMOTE transformed data in pipeline. Dont forget to use transformed X and y from output.')
            difftime = max(1, int(time.time()-start_time))
            print('    Time taken to transform dataset = %s second(s)' %difftime)
            print('    Shape of transformed dataset: %s' %(X_enc.shape,))
            return X_enc, y_enc
        else:
            print('LazyTransformer has not been fitted yet. Returning...')
            return X, y

    def fit_transform(self, X, y=None):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        self.X_index = X.index
        if y is not None:
            self.y_index = y.index
        start_time = time.time()
        self.fit(X,y)
        X_trans =  self.xformer.transform(X)
        X_trans.index = self.X_index
        ### Here you can straight away fit and transform y ###
        if self.transform_target:
            y_trans = self.yformer.fit_transform(y)
        else:
            y_trans = y
        #### Now we need to check if imbalanced flag is set ###
        if self.imbalanced_first_done and self.imbalanced:
            pass
        elif not self.imbalanced_first_done and self.imbalanced:
            print('### Warning! Do not use imbalanced_flag if this is not an imbalanced classification problem! #######')
            try:
                from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
            except:
                print('This function needs Imbalanced-Learn library. Please pip install it first and try again!')
                return
            sm = self.smotex
            if self.verbose:
                print('Imbalanced flag set. Using SMOTE to transform X and y...')
            X_trans, y_trans = sm.fit_resample(X_trans, y_trans)
            self.imbalanced_first_done = True
            self.smotex = sm
            if self.verbose:
                print('    SMOTE transformed data in pipeline. Dont forget to use transformed X and y from output.')
        difftime = max(1, int(time.time()-start_time))
        print('    Time taken to transform dataset = %s second(s)' %difftime)
        print('    Shape of transformed dataset: %s' %(X_trans.shape,))
        return X_trans, y_trans

    def fit_predict(self, X, y=None):
        transformer_ = self.fit(X,y)
        y_trans =  transformer_.predict(X)
        return X, y_trans

    def print_pipeline(self):
        from sklearn import set_config
        set_config(display="text")
        return self.xformer

    def plot_pipeline(self):
        from sklearn import set_config
        set_config(display="diagram")
        return self.xformer

    def plot_importance(self, max_features=10):
        import lightgbm as lgbm
        from xgboost import plot_importance
        model_name = str(self.model).split("(")[-2].split(",")[-1]
        if  model_name == ' LGBMClassifier' or model_name == ' LGBMRegressor':
            lgbm.plot_importance(self.model.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == ' XGBClassifier' or model_name == ' XGBRegressor':
            plot_importance(self.model.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == ' LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            import math
            feature_names = self.model.named_steps['model'].feature_names_in_
            model = self.model.named_steps['model']
            w0 = model.intercept_[0]
            w = model.coef_[0]
            feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
            feature_importance["importance"] = pow(math.e, w)
            feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)[:max_features]
            feature_importance.plot.barh(x='feature', y='importance')
        else:
            ### These are for RandomForestClassifier kind of scikit-learn models ###
            try:
                importances = model.feature_importances_
                feature_names = self.model.named_steps['model'].feature_names_in_
                forest_importances = pd.Series(importances, index=feature_names)
                forest_importances.sort_values(ascending=False)[:max_features].plot(kind='barh')
            except:
                print('Could not plot feature importances. Please check your model and try again.')


    def lightgbm_grid_search(self, X_train, y_train, modeltype,
                             params={}, grid_search=False, multi_label=False,
                             ):
        """
        Perform GridSearchCV or RandomizedSearchCV using LightGBM based LazyTransformer pipeline.
        -- Remember that you can only use LightGBM scikit-learn syntax based models here. 
        Optionally you can provide parameters to search (as a dictionary) in the following format:
                params = {
                'model__learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5],
                'model__num_leaves': [20,30,40,50],
                        }
            You can use any LightGBM parameter you want in addition to the above. Please check syntax.
        You can also provide multi_label as True to enable MultiOutputRegressor or MultiOutputClassifier
           to be used as a wrapper around the LightGBM object. The params will be passed to the 
           multioutput estimator which will in turn pass them on to the LGBM estimators during fit.

        This function returns a new LazyTransformer pipeline that contains the best model trained on that dataset.
        You can use this new pipeline to predict on existing or new datasets.

        """

        start_time = time.time()
        if len(params) == 0: 
            if modeltype == 'Regression':
                rand_params = {
                    'model__n_estimators': np.linspace(50,1000,10).astype(int),
                    'model__learning_rate': np.linspace(0.0001,1,10),
                    'model__boosting_type': ['dart','gbdt'],
                    }
                grid_params = {
                    'model__n_estimators': np.linspace(50,1000,5).astype(int),
                    'model__learning_rate': np.linspace(0.0001,1,5),
                    'model__boosting_type': ['dart','gbdt'],
                }
            else:
                rand_params = {
                    'model__n_estimators': np.linspace(50,1000,10).astype(int),
                    'model__class_weight':[None, 'balanced'],
                    'model__learning_rate': np.linspace(0.0001,1,10),
                    'model__boosting_type': ['dart','gbdt'],
                    }
                grid_params = {
                    'model__n_estimators': np.linspace(50,1000,5).astype(int),
                    'model__class_weight':[None, 'balanced'],
                    'model__boosting_type': ['dart','gbdt'],
                        }
        else:
            if grid_search:
                grid_params = copy.deepcopy(params)
            else:
                rand_params = copy.deepcopy(params)
                
        ########   Now let's perform randomized search to find best hyper parameters ######
        init_params = {
              }

        if modeltype == 'Regression':
            scoring = 'neg_mean_squared_error'
            score_name = 'MSE'
            self.model = LGBMRegressor(random_seed=99)
            self.model.set_params(**init_params)            
        else:
            self.model = LGBMClassifier(random_seed=99)
            self.model.set_params(**init_params)            
            if grid_search:
                scoring = 'balanced_accuracy'
                score_name = 'balanced_accuracy'
            else:
                scoring = 'balanced_accuracy'
                score_name = 'balanced_accuracy'

        if X_train.shape[0] <=100000:
            n_iter = 10
        else:
            n_iter = 5
        #### This is where you grid search the pipeline now ##############

        if grid_search:
            search = GridSearchCV(
                    self, 
                    grid_params, 
                    cv=5,
                    scoring=scoring,
                    refit=False,
                    return_train_score = True,
                    n_jobs=-1,
                    verbose=True,
                    )
        else:
            search = RandomizedSearchCV(
                    self,
                    rand_params,
                    n_iter = n_iter,
                    cv = 5,
                    refit=False,
                    return_train_score = True,
                    random_state = 99,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=True,
                    )
            print('Number of iterations in randomized search = %s' %n_iter)
        
        ##### This is where we search for hyper params for model #######
        search.fit(X_train, y_train)
        
        cv_results = pd.DataFrame(search.cv_results_)
        if modeltype == 'Regression':
            print('Mean cross-validated train %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_train_score'].mean()))))
            print('    Mean cross-validated test %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_test_score'].mean()))))
            if cv_results['mean_test_score'].mean()/cv_results['mean_train_score'].mean() >= 1.2:
                print('#### Model is overfitting. You might want to test without GridSearch option as well ####')
        else:
            print('Mean cross-validated train %s = %0.04f' %(score_name, cv_results['mean_train_score'].mean()))
            print('    Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
            if cv_results['mean_train_score'].mean()/cv_results['mean_test_score'].mean() >= 1.2:
                print('#### Model is overfitting. You might want to test without GridSearch option as well ####')
            
        print('Time taken for Hyper Param tuning of LGBM (in minutes) = %0.1f' %(
                                        (time.time()-start_time)/60))
        print('Best params from search:\n%s' %search.best_params_)
        new_model = self.model.set_params(**search.best_params_)
        if multi_label:
            if modeltype == 'Regression':
                new_model = MultiOutputRegressor(new_model)
            else:
                new_model = MultiOutputClassifier(new_model)
        self.model = new_model
        self.fit(X_train, y_train)
        print('    returning a new LazyTransformer pipeline that contains the best model trained on your train dataset!')
        return self
####################################################################################
# This is needed to make this a regular transformer ###
class YTransformer(TransformerMixin):
    def __init__(self, transformers={}, targets=[]):
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  transformers
        self.targets = targets
        
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"transformers": self.transformers, "targets": self.targets}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, y):
        """Fit the model according to the given training data"""        
        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if isinstance(y, pd.Series):
            if y.name not in self.targets:
                self.targets.append(y.name)
        elif isinstance(y, pd.DataFrame):
            self.targets += y.columns.tolist()
            self.targets = find_remove_duplicates(self.targets)
        # transform y and convert back to 1d array if needed
        if isinstance(y, pd.Series):
            y_1d = y.values
            y = pd.DataFrame(y_1d, columns=self.targets)
        elif isinstance(y, pd.DataFrame):
            y_1d = y.values
        #y_1d = column_or_1d(y_1d)
        for i, each_target in enumerate(self.targets):
            lb = My_LabelEncoder()
            lb.fit(y.iloc[:,i])
            self.transformers[each_target] = lb
        return self
    
    def transform(self, y):
        target_len = len(self.targets) - 1
        if y is None:
            return y
        else:
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            y_trans = copy.deepcopy(y)
            for each_target in self.targets:
                y_trans[each_target] = self.transformers[each_target].transform(y[each_target])
            return y_trans
    
    def fit_transform(self, y=None):
        self.fit(y)
        if isinstance(y, pd.Series):
            y_trans = self.transformers[self.targets[0]].transform(y)
            y_trans = pd.Series(y_trans, index=y.index, name=y.name)
        elif isinstance(y, pd.DataFrame):
            y_trans = copy.deepcopy(y)
            for each_target in self.targets:
                y_trans[each_target] = self.transformers[each_target].transform(y[each_target])
        else:
            print('Error: y must be a pandas series or dataframe. Try again after transforming y.')
            return y
        return y_trans
    
    def inverse_transform(self, y):
        for i, each_target in enumerate(self.targets):
            if i == 0:
                transformer_ = self.transformers[each_target]
                if isinstance(y, pd.Series):
                    y = pd.DataFrame(y, columns=self.targets)
                elif isinstance(y, np.ndarray):
                    y = pd.DataFrame(y, columns=self.targets)
                else:
                    ### if it is a dataframe then leave it alone ##
                    pass
                y_t = transformer_.inverse_transform(y.values[:,i])
                y_trans = pd.Series(y_t,name=each_target)
            else:
                transformer_ = self.transformers[each_target]
                y_t = transformer_.inverse_transform(y.values[:,i])
                y_trans = pd.DataFrame(y_trans)
                y_trans[each_target] = y_t
        return y_trans
    
    def predict(self, y=None, **fit_params):
        #print('There is no predict function in Label Encoder. Returning...')
        return y
##############################################################################
from collections import defaultdict
# This is needed to make this a regular transformer ###
from sklearn.base import BaseEstimator, TransformerMixin 
class Rare_Class_Combiner_Pipe(BaseEstimator, TransformerMixin ):
    """
    This is the pipeline version of rare class combiner used in sklearn pipelines.
    """
    def __init__(self, transformers={}  ):
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  transformers
        self.zero_low_counts = defaultdict(bool)
        
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"transformers": self.transformers}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data"""        
        # transformers need a default name for rare categories ##
        def return_cat_value():
            return "rare_categories"
        ### In this case X itself will only be a pd.Series ###
        each_catvar = X.name
        #### if it is already a list, then leave it as is ###
        self.transformers[each_catvar] = defaultdict(return_cat_value)
        ### Then find the unique categories in the column ###
        self.transformers[each_catvar] = dict(zip(X.unique(), X.unique()))
        low_counts = pd.DataFrame(X).apply(lambda x: x.value_counts()[
                (x.value_counts()<=(0.01*x.shape[0])).values].index).values.ravel()
        
        if len(low_counts) == 0:
            self.zero_low_counts[each_catvar] = True
        else:
            self.zero_low_counts[each_catvar] = False
        for each_low in low_counts:
            self.transformers[each_catvar].update({each_low:'rare_categories'})
        return self
    
    def transform(self, X, y=None, **fit_params):
        each_catvar = X.name
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X = X.map(self.transformers[each_catvar])
            ### simply fill in the missing values with the word "missing" ##
            X = X.fillna('missing',inplace=False)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        ### Since X for yT in a pipeline is sent as X, we need to switch X and y this way ##
        self.fit(X, y)
        each_catvar = X.name
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X = X.map(self.transformers[each_catvar])
            ### simply fill in the missing values with the word "missing" ##
            X = X.fillna('missing',inplace=False)
        return X

    def inverse_transform(self, X, **fit_params):
        ### One problem with this approach is that you have combined categories into one.
        ###   You cannot uncombine them since they no longer have a unique category. 
        ###   You will get back the last transformed category when you inverse transform it.
        each_catvar = X.name
        transformer_ = self.transformers[each_catvar]
        reverse_transformer_ = dict([(y,x) for (x,y) in transformer_.items()])
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X[each_catvar] = X[each_catvar].map(reverse_transformer_).values
        return X
    
    def predict(self, X, y=None, **fit_params):
        #print('There is no predict function in Rare class combiner. Returning...')
        return X
##############################################################################
import copy
def EDA_find_remove_columns_with_infinity(df, remove=False):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    If remove flag is set, then it returns a smaller dataframe with inf columns removed.
    """
    nums = df.select_dtypes(include='number').columns.tolist()
    dfx = df[nums]
    sum_rows = np.isinf(dfx).values.sum()
    add_cols =  list(dfx.columns.to_series()[np.isinf(dfx).any()])
    if sum_rows > 0:
        print('    there are %d rows and %d columns with infinity in them...' %(sum_rows,len(add_cols)))
        if remove:
            ### here you need to use df since the whole dataset is involved ###
            nocols = [x for x in df.columns if x not in add_cols]
            print("    Shape of dataset before %s and after %s removing columns with infinity" %(df.shape,(df[nocols].shape,)))
            return df[nocols]
        else:
            ## this will be a list of columns with infinity ####
            return add_cols
    else:
        ## this will be an empty list if there are no columns with infinity
        return add_cols
####################################################################################
def check_if_GPU_exists():
    try:
        os.environ['NVIDIA_VISIBLE_DEVICES']
        print('GPU active on this device')
        return True
    except:
        print('No GPU active on this device')
        return False
############################################################################################
####        This is where SULO CLASSIFIER AND REGRESSOR ARE DEFINED          ###############
############################################################################################
class SuloClassifier(BaseEstimator, ClassifierMixin):
    """
    SuloClassifier stands for Super Learning Optimized (SULO) Classifier.
    -- It works on small as well as big data. It works in Integer mode as well as float-mode.
    -- It works on regular balanced data as well as skewed regression targets.
    The reason it works so well is that Sulo is an ensemble of highly tuned models.
    -- You don't have to send any inputs but if you wanted to, you can spefify multiple options.
    It is fully compatible with scikit-learn pipelines and other models.

    Syntax:
        sulo = SuloClassifier(base_estimator=None, n_estimators=None, pipeline=True, 
                                weights=False, imbalanced=False, verbose=0)
        sulo.fit(X_train, y_train)
        y_preds = sulo.predict(X_test)

    Inputs:
        n_estimators: default is None. Number of models you want in the final ensemble.
        base_estimator: default is None. Base model you want to train in each of the ensembles.
        pipeline: default is False. It will transform all data to numeric automatically if set to True.
        weights: default is False. It will perform weighting of different classifiers if set to True
        imbalanced: default is False. It will activate a special imbalanced classifier if set to True.
        verbose: default is 0. It will print verbose output if set to True.

    Oututs:
        SuloClassifier: returns a classification model highly tuned to your specific needs and dataset.
    """
    def __init__(self, base_estimator=None, n_estimators=None, pipeline=True, weights=False, 
                                       imbalanced=False, verbose=0):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.pipeline = pipeline
        self.weights = weights
        self.imbalanced = imbalanced
        self.verbose = verbose
        self.models = []
        self.multi_label =  False
        self.max_number_of_classes = 1
        self.scores = []
        self.classes = []
        self.regression_min_max = []
        self.model_name = ''
        self.features = []

    def return_worst_fold(self, X):
        """
        This method returns the worst performing train and test rows among all the folds.
        This is very important information since it helps an ML engineer or Data Scientist 
            to trouble shoot classification problems. It helps to find where the model is struggling.

        Inputs:
        --------
        X: Dataframe. This must be the features dataframe of your dataset. It cannot be a numpy array.

        Outputs:
        ---------
        train_rows_dataframe, test_rows_dataframe: Dataframes. 
             This returns the portion of X as train and test to help you understand where model struggled.
        """
        worst_fold = np.argmin(self.scores)
        for i, (tr, tt) in enumerate(self.kf.split(X)):
            if i == worst_fold:
                worst_train_rows = copy.deepcopy(tr)
                worst_ttest_rows = copy.deepcopy(tt)
                print("fold %s: train rows index = %s. Sulo model struggled in this fold." %(i+1,tr))
            else:
                print("fold %s: train rows index = %s" %(i+1,tr))
        return X.iloc[worst_train_rows], X.iloc[worst_ttest_rows]

    def fit(self, X, y):
        X = copy.deepcopy(X)
        print('Input data shapes: X = %s' %(X.shape,))
        print('    y shape = %s' %(y.shape,))
        seed = 42
        shuffleFlag = True
        modeltype = 'Classification'
        features_limit = 50 ## if there are more than 50 features in dataset, better to use LGBM ##
        start = time.time()
        if isinstance(X, pd.DataFrame):
            self.features = X.columns.tolist()
        else:
            print('Cannot operate SuloClassifier on numpy arrays. Must be dataframes. Returning...')
            return self
        # Use KFold for understanding the performance
        if self.weights:
            print('Remember that using class weights will wrongly skew predict_probas from any classifier')
        if self.imbalanced:
            class_weights = None
        else:
            class_weights = get_class_weights(y, verbose=0)
        ### Remember that putting class weights will totally destroy predict_probas ###
        self.classes = print_flatten_dict(class_weights)
        scale_pos_weight = get_scale_pos_weight(y)
        #print('Class weights = %s' %class_weights)
        gpu_exists = check_if_GPU_exists()
        if gpu_exists:
            device="gpu"
        else:
            device="cpu"
        ## Don't change this since it gives an error ##
        metric  = 'auc'
        ### don't change this metric and eval metric - it gives error if you change it ##
        eval_metric = 'auc'
        row_limit = 10000
        if self.imbalanced:
            print('    Imbalanced classes of y = %s' %find_rare_class(y, verbose=self.verbose))
        ################          P I P E L I N E        ##########################
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean", add_indicator=True)), ("scaler", StandardScaler())]
        )

        categorical_transformer_low = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", OneHotEncoder(handle_unknown="value")),
            ]
        )

        categorical_transformer_high = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", LabelEncoder()),
            ]
        )

        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_cardinality(X, categorical_features)
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        ####################################################################################
        if isinstance(y, pd.DataFrame):
            if len(y.columns) >= 2:
                number_of_classes = num_classes(y)
                for each_i in y.columns:
                    number_of_classes[each_i] = int(number_of_classes[each_i] - 1)
                max_number_of_classes = np.max(list(number_of_classes.values()))
            else:
                number_of_classes = int(num_classes(y) - 1)
                max_number_of_classes = np.max(number_of_classes)
        else:
            number_of_classes = int(num_classes(y) - 1)
            max_number_of_classes = np.max(number_of_classes)
        data_samples = X.shape[0]
        self.max_number_of_classes = max_number_of_classes
        if self.n_estimators is None:
            if data_samples <= row_limit:
                self.n_estimators = min(3, int(1.5*np.log10(data_samples)))
            else:
                self.n_estimators = min(10, int(1.5*np.log10(data_samples)))
        self.model_name = 'lgb'
        num_splits = self.n_estimators
        num_repeats = 1
        kfold = RepeatedKFold(n_splits=num_splits, random_state=seed, n_repeats=num_repeats)
        num_iterations = int(num_splits * num_repeats)
        scoring = 'balanced_accuracy'
        print('    Number of estimators used in SuloClassifier = %s' %num_iterations)
        ##### This is where we check if y is single label or multi-label ##
        if isinstance(y, pd.DataFrame):
            ###############################################################
            ### This is for Multi-Label problems only #####################
            ###############################################################
            targets = y.columns.tolist()
            if is_y_object(y):
                print('Cannot perform classification using object or string targets. Please convert to numeric and try again.')
                return self
            if len(targets) > 1:
                self.multi_label = y.columns.tolist()
                ### You need to initialize the class before each run - otherwise, error!
                ### Remember we don't to HPT Tuning for Multi-label problems since it errors ####
                i = 0
                for train_index, test_index in tqdm(kfold.split(X, y), total=kfold.get_n_splits(),
                                            desc="k-fold training"):
                    start_time = time.time()
                    #random_seed = np.random.randint(2,100)
                    random_seed = 9999
                    if self.base_estimator is None:
                        if self.max_number_of_classes <= 1:
                            ##############################################################
                            ###   This is for Binary Classification problems only ########
                            ##############################################################
                            if self.imbalanced:
                                try:
                                    from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                    self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=random_seed)
                                except:
                                    print('pip install imbalanced_ensemble and re-run this again.')
                                    return self
                                self.model_name = 'other'
                            else:
                                ### make it a regular dictionary with weights for pos and neg classes ##
                                class_weights = dict([v for k,v in class_weights.items()][0])
                                if data_samples <= row_limit:
                                    if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                        if self.verbose:
                                            print('    Selecting Label Propagation since it will work great for this dataset...')
                                            print('        however it will skew probabilities and show lower ROC AUC score than normal.')
                                        self.base_estimator =  LabelPropagation()
                                        self.model_name = 'lp'
                                    else:
                                        if len(self.features) <= features_limit:
                                            if self.verbose:
                                                print('    Selecting Bagging Classifier for this dataset...')
                                            ET = ExtraTreeClassifier()
                                            self.base_estimator = BaggingClassifier(base_estimator=ET, n_jobs=-1)
                                            self.model_name = 'bg'
                                        else:
                                            if self.verbose:
                                                print('    Selecting LGBM Regressor as base estimator...')
                                            self.base_estimator = LGBMClassifier(device=device, random_state=random_seed,
                                                               class_weight=class_weights, n_jobs=-1,
                                                                )
                                else:
                                    ### This is for large datasets in Binary classes ###########
                                    if self.verbose:
                                        print('    Selecting LGBM Regressor as base estimator...')
                                    if gpu_exists:
                                        self.base_estimator = XGBClassifier(n_estimators=250, 
                                            n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                                    else:
                                        self.base_estimator = LGBMClassifier(device=device, random_state=random_seed, 
                                                            n_jobs=-1,
                                                            #is_unbalance=True, 
                                                            #max_depth=10, metric=metric,
                                                            #num_class=self.max_number_of_classes,
                                                            #n_estimators=100,  num_leaves=84, 
                                                            #objective='binary',
                                                            #boosting_type ='goss', 
                                                            #scale_pos_weight=scale_pos_weight,
                                                           class_weight=class_weights,
                                                            )
                        else:
                            #############################################################
                            ###   This is for Multi Classification problems only ########
                            ### Make sure you don't put any class weights here since it won't work in multi-labels ##
                            ##############################################################
                            if self.imbalanced:
                                if self.verbose:
                                    print('    Selecting Self Paced ensemble classifier since imbalanced flag is set...')
                                try:
                                    from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                    self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=random_seed)
                                except:
                                    print('pip install imbalanced_ensemble and re-run this again.')
                                    return self
                                self.model_name = 'other'
                            else:
                                if data_samples <= row_limit:
                                    if len(self.features) <= features_limit:
                                        if self.verbose:
                                            print('    Selecting Extra Trees Classifier for small datasets...')
                                        self.base_estimator = ExtraTreesClassifier(random_state=random_seed)
                                        self.model_name = 'rf'
                                    else:
                                        self.base_estimator = LGBMRegressor(n_jobs=-1, device=device, random_state=random_seed)                                    
                                else:
                                    if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                        print('    Selecting Label Propagation since it works great for multiclass problems...')
                                        print('        however it will skew probabilities a little so be aware of this')
                                        self.base_estimator =  LabelPropagation()
                                        self.model_name = 'lp'
                                    else:
                                        self.base_estimator = LGBMClassifier(n_jobs=-1, device=device, random_state=random_seed)
                    else:
                        self.model_name == 'other'
                    ### Now print LGBM if appropriate #######
                    if self.verbose and self.model_name=='lgb':
                        print('    Selecting LGBM Classifier as base estimator...')
                    # Split data into train and test based on folds          
                    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
                    else:
                        y_train, y_test = y[train_index], y[test_index]

                    if isinstance(X, pd.DataFrame):
                        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                    else:
                        x_train, x_test = X[train_index], X[test_index]

                    ###### Do this only the first time ################################################
                    if i == 0:
                        ### It does not make sense to do hyper-param tuning for multi-label models ##
                        ###    since ClassifierChains do not have many hyper params #################
                        #self.base_estimator = rand_search(self.base_estimator, x_train, y_train, 
                        #                        self.model_name, verbose=self.verbose)
                        #print('    hyper tuned base estimator = %s' %self.base_estimator)
                        if self.max_number_of_classes <= 1:
                            est_list = [ClassifierChain(self.base_estimator, order=None, cv=3, random_state=i) 
                                        for i in range(num_iterations)] 
                            if self.verbose:
                                print('Fitting a %s for %s targets with MultiOutputClassifier. This will take time...' %(
                                            str(self.base_estimator).split("(")[0],y.shape[1]))
                        else:
                            if self.imbalanced:
                                if self.verbose:
                                    print('    Training with ClassifierChain since multi_label dataset. This will take time...')
                                est_list = [ClassifierChain(self.base_estimator, order=None, random_state=i)
                                            for i in range(num_iterations)] 
                            else:
                                ### You must use multioutputclassifier since it is the only one predicts probas correctly ##
                                est_list = [MultiOutputClassifier(self.base_estimator)#, order="random", random_state=i) 
                                            for i in range(num_iterations)] 
                                if self.verbose:
                                    print('Training a %s for %s targets with MultiOutputClassifier. This will take time...' %(
                                                str(self.base_estimator).split("(")[0],y.shape[1]))

                    # Initialize model with your supervised algorithm of choice
                    model = est_list[i]

                    # Train model and use it to train on the fold
                    if self.pipeline:
                        ### This is only with a pipeline ########
                        pipe = Pipeline(
                            steps=[("preprocessor", preprocessor), ("model", model)]
                        )

                        pipe.fit(x_train, y_train)
                        self.models.append(pipe)

                        # Predict on remaining data of each fold
                        preds = pipe.predict(x_test)

                    else:
                        #### This is without a pipeline ###
                        model.fit(x_train, y_train)
                        self.models.append(model)

                        # Predict on remaining data of each fold
                        preds = model.predict(x_test)

                    
                    # Use best classification metric to measure performance of model
                    if self.imbalanced:
                        ### Use special priting program here ##
                        score = print_sulo_accuracy(y_test, preds, y_probas="", verbose=self.verbose)
                        if self.verbose:
                            print("    Fold %s: out-of-fold balanced_accuracy: %0.1f%%" %(i+1, 100*score))
                    else:
                        score = print_accuracy(targets, y_test, preds, verbose=self.verbose)
                        if self.verbose:
                            print("    Fold %s: out-of-fold balanced_accuracy: %0.1f%%" %(i+1, 100*score))
                    self.scores.append(score)
                    
                    # Finally, check out the total time taken
                    end_time = time.time()
                    timeTaken = end_time - start_time
                    if self.verbose:
                        print("Time Taken for fold %s: %0.0f (seconds)" %(i+1, timeTaken))

                    i += 1

                # Compute average score
                averageAccuracy = sum(self.scores)/len(self.scores)
                print("Final Balanced Accuracy score of %s-estimator SuloClassifier: %0.1f%%" %(
                                self.n_estimators, 100*averageAccuracy))

                end = time.time()
                timeTaken = end - start
                print("Time Taken overall: %0.0f (seconds)" %(timeTaken))
                self.kf = kfold
                return self
        ########################################################
        #####  This is for Single Label Classification problems 
        ########################################################
        
        if isinstance(y, pd.Series):
            targets = y.name
            if targets is None:
                targets = []
        else:
            targets = []

        est_list = num_iterations*[self.base_estimator]
        
        # Perform CV
        i = 0
        for train_index, test_index in tqdm(kfold.split(X, y), total=kfold.get_n_splits(), 
                                        desc="k-fold training"):
            start_time = time.time()
            random_seed = np.random.randint(2,100)
            ##########  This is where we do multi-seed classifiers #########
            
            if self.base_estimator is None:
                if data_samples <= row_limit:
                    ### For small datasets use RFC for Binary Class   ########################
                    if number_of_classes <= 1:
                        if self.imbalanced:
                            if self.verbose:
                                print('    Selecting Self Paced ensemble classifier as base estimator...')
                            try:
                                from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=random_seed)
                            except:
                                print('pip install imbalanced_ensemble and re-run this again.')
                                return self
                            self.model_name = 'other'
                        else:
                            ### For binary-class problems use RandomForest or the faster ET Classifier ######
                            if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                print('    Selecting Label Propagation since it will work great for this dataset...')
                                print('        however it will skew probabilities and show lower ROC AUC score than normal.')
                                self.base_estimator =  LabelPropagation()
                                self.model_name = 'lp'
                            else:
                                if len(self.features) <= features_limit:
                                    if self.verbose:
                                        print('    Selecting Bagging Classifier for this dataset...')
                                    ### The Bagging classifier outperforms ETC most of the time ####
                                    ET = ExtraTreeClassifier()
                                    self.base_estimator = BaggingClassifier(base_estimator=ET, n_jobs=-1)
                                    self.model_name = 'bg'
                                else:
                                    if self.verbose:
                                        print('    Selecting LGBM Classifier as base estimator...')
                                    self.base_estimator = LGBMClassifier(device=device, random_state=random_seed, n_jobs=-1,
                                                            scale_pos_weight=scale_pos_weight,)
                    else:
                        ### For Multi-class datasets you can use Regressors for numeric classes ####################
                        if self.imbalanced:
                            if self.verbose:
                                print('    Selecting Self Paced ensemble classifier as base estimator...')
                            try:
                                from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=random_seed)
                            except:
                                print('pip install imbalanced_ensemble and re-run this again.')
                                return self
                            self.model_name = 'other'
                        else:
                            ### For multi-class problems use Label Propagation which is faster and better ##
                            if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                    print('    Selecting Label Propagation since it will work great for this dataset...')
                                    print('        however it will skew probabilities and show lower ROC AUC score than normal.')
                                    self.base_estimator =  LabelPropagation()
                                    self.model_name = 'lp'
                            else:
                                if len(self.features) <= features_limit:
                                    if self.verbose:
                                        print('    Selecting Bagging Classifier for this dataset...')
                                    ET = ExtraTreeClassifier()
                                    self.base_estimator = BaggingClassifier(base_estimator=ET, n_jobs=-1)
                                    self.model_name = 'bg'
                                else:
                                    if self.verbose:
                                        print('    Selecting LGBM Classifier as base estimator...')
                                    self.base_estimator = LGBMClassifier(device=device, random_state=random_seed,
                                                    n_jobs=-1,
                                                    #is_unbalance=False,
                                                    #learning_rate=0.3,
                                                    #max_depth=10,
                                                    metric='multi_logloss',
                                                    #n_estimators=130, num_leaves=84,
                                                    num_class=number_of_classes, objective='multiclass',
                                                    #boosting_type ='goss', 
                                                    #scale_pos_weight=None,
                                                    class_weight=class_weights
                                                    )
                else:
                    ### For large datasets use LGBM or Regressors as well ########################
                    if number_of_classes <= 1:
                        if self.imbalanced:
                            if self.verbose:
                                print('    Selecting Self Paced ensemble classifier as base estimator...')
                            try:
                                from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=random_seed)
                            except:
                                print('pip install imbalanced_ensemble and re-run this again.')
                                return self
                            self.model_name = 'other'
                        else:
                            if self.verbose:
                                print('    Selecting LGBM Classifier for this dataset...')
                            #self.base_estimator = LGBMClassifier(n_estimators=250, random_state=random_seed, 
                            #            boosting_type ='goss', scale_pos_weight=scale_pos_weight)
                            self.base_estimator = LGBMClassifier(device=device, random_state=random_seed,
                                                    n_jobs=-1,
                                                    #is_unbalance=True,
                                                    #learning_rate=0.3, 
                                                    #max_depth=10, 
                                                    #metric=metric,
                                                    #n_estimators=230, num_leaves=84, 
                                                    #num_class=number_of_classes,
                                                    #objective='binary',
                                                    #boosting_type ='goss', 
                                                    scale_pos_weight=scale_pos_weight
                                                    )
                    else:
                        ### For Multi-class datasets you can use Regressors for numeric classes ####################
                        if self.imbalanced:
                            if self.verbose:
                                print('    Selecting Self Paced ensemble classifier as base estimator...')
                            try:
                                from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=random_seed)
                            except:
                                print('pip install imbalanced_ensemble and re-run this again.')
                                return self
                            self.model_name = 'other'
                        else:
                            #if self.weights:
                            #    class_weights = None
                            if self.verbose:
                                print('    Selecting LGBM Classifier as base estimator...')
                            self.base_estimator = LGBMClassifier(device=device, random_state=random_seed,
                                                    n_jobs=-1,
                                                    #is_unbalance=False, learning_rate=0.3,
                                                    #max_depth=10, 
                                                    metric='multi_logloss',
                                                    #n_estimators=230, num_leaves=84,
                                                    num_class=number_of_classes, objective='multiclass',
                                                    #boosting_type ='goss', 
                                                    scale_pos_weight=None,
                                                    class_weight=class_weights
                                                    )
            else:
                self.model_name = 'other'


            # Split data into train and test based on folds          
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
            else:
                y_train, y_test = y[train_index], y[test_index]

            if isinstance(X, pd.DataFrame):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            else:
                x_train, x_test = X[train_index], X[test_index]

            
            ##   small datasets processing #####
            if i == 0:
                if self.pipeline:
                    # Train model and use it in a pipeline to train on the fold  ##
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("model", self.base_estimator)])
                    if self.model_name == 'other':
                        print('No HPT tuning performed since base estimator is given by input...')
                        self.base_estimator = copy.deepcopy(pipe)
                    else:
                        if len(self.features) <= features_limit:
                            self.base_estimator = rand_search(pipe, x_train, y_train, 
                                                    self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')
                            self.base_estimator = copy.deepcopy(pipe)
                else:
                    ### This is for without a pipeline #######
                    if self.model_name == 'other':
                        ### leave the base estimator as is ###
                        print('No HPT tuning performed since base estimator is given by input...')
                    else:
                        if len(self.features) <= features_limit:
                            ### leave the base estimator as is ###
                            self.base_estimator = rand_search(self.base_estimator, x_train, 
                                                y_train, self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')

                est_list = num_iterations*[self.base_estimator]
                #print('    base estimator = %s' %self.base_estimator)
            
            # Initialize model with your supervised algorithm of choice
            model = est_list[i]
            
            model.fit(x_train, y_train)
            self.models.append(model)
            
            # Predict on remaining data of each fold
            preds = model.predict(x_test)

            # Use best classification metric to measure performance of model

            if self.imbalanced:
                ### Use Regression predictions and convert them into classes here ##
                score = print_sulo_accuracy(y_test, preds, y_probas="", verbose=self.verbose)
                if self.verbose:
                        print("    Fold %s: out-of-fold balanced accuracy: %0.1f%%" %(i+1, 100*score))
            else:
                #score = balanced_accuracy_score(y_test, preds)
                score = print_accuracy(targets, y_test, preds, verbose=self.verbose)
                if self.verbose:
                        print("    Fold %s: out-of-fold balanced accuracy: %0.1f%%" %(i+1, 100*score))
            self.scores.append(score)

            i += 1

        # Compute average score
        averageAccuracy = sum(self.scores)/len(self.scores)
        print("Final balanced Accuracy of %s-estimator SuloClassifier: %0.1f%%" %(num_iterations, 100*averageAccuracy))

        # Finally, check out the total time taken
        end = time.time()
        timeTaken = end-start
        print("Time Taken: %0.0f (seconds)" %timeTaken)
        self.kf = kfold
        return self

    def predict(self, X):
        from scipy import stats
        weights = self.scores
        if self.multi_label:
            ### In multi-label, targets have to be numeric, so you can leave weights as-is ##
            ypre = np.array([model.predict(X) for model in self.models ])
            y_predis = np.average(ypre, axis=0, weights=weights)
            y_preds = np.round(y_predis,0).astype(int)
            return y_preds
        y_predis = np.column_stack([model.predict(X) for model in self.models ])
        ### This weights the model's predictions according to OOB scores obtained
        #### In single label, targets can be object or string, so weights cannot be applied always ##
        if y_predis.dtype == object or y_predis.dtype == bool:
            ### in the case of predictions that are strings, no need for weights ##
            y_predis = stats.mode(y_predis, axis=1)[0].ravel()
        else:
            if str(y_predis.dtype) == 'category':
                y_predis = stats.mode(y_predis, axis=1)[0].ravel()
            else:
                y_predis = np.average(y_predis, weights=weights, axis=1)
                y_predis = np.round(y_predis,0).astype(int)
        if self.imbalanced:
            y_predis = copy.deepcopy(y_predis)
        return y_predis
    
    def predict_proba(self, X):
        weights = self.scores
        y_probas = [model.predict_proba(X) for model in self.models ]
        y_probas = return_predict_proba(y_probas)
        return y_probas

    def print_pipeline(self):
        from sklearn import set_config
        set_config(display="text")
        return self.xformer

    def plot_pipeline(self):
        from sklearn import set_config
        set_config(display="diagram")
        return self

    def plot_importance(self, max_features=10):
        import lightgbm as lgbm
        from xgboost import plot_importance
        model_name = self.model_name
        feature_names = self.features
        if self.multi_label:
            print('No feature importances available for multi_label problems')
            return
        if  model_name == 'lgb' or model_name == 'xgb':
            for i, model in enumerate(self.models):
                if self.pipeline:
                    model_object = model.named_steps['model']
                else:
                    model_object = model
                feature_importances = model_object.booster_.feature_importance(importance_type='gain')
                if i == 0:
                    feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                else:
                    feature_imp = pd.concat([feature_imp, 
                        pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                #lgbm.plot_importance(model_object, importance_type='gain', max_num_features=max_features)
            feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
            feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            ### This is for XGB ###
            #plot_importance(self.model.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == 'lp':
            print('No feature importances available for LabelPropagation algorithm. Returning...')
            return
        elif model_name == 'rf':
            ### These are for RandomForestClassifier kind of scikit-learn models ###
            try:
                for i, model in enumerate(self.models):
                    if self.pipeline:
                        model_object = model.named_steps['model']
                    else:
                        model_object = model
                    feature_importances = model_object.feature_importances_
                    if i == 0:
                        feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                    else:
                        feature_imp = pd.concat([feature_imp, 
                            pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
                feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            except:
                    print('Could not plot feature importances. Please check your model and try again.')                
        else:
            print('No feature importances available for this algorithm. Returning...')
            return
#####################################################################################################
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
def rand_search(model, X, y, model_name, pipe_flag=False, scoring=None, verbose=0):
    start = time.time()
    if pipe_flag:
        model_string = 'model__'
    else:
        model_string = ''
    ### set n_iter here ####
    if X.shape[0] <= 10000:
        n_iter = 10
    else:
        n_iter = 5
    if model_name == 'rf':
        #criterion = ["gini", "entropy", "log_loss"]
        # Number of trees in random forest
        n_estimators = sp_randInt(100, 300)
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log']
        # Maximum number of levels in tree
        max_depth = sp_randInt(2, 10)
        # Minimum number of samples required to split a node
        min_samples_split = sp_randInt(2, 10)
        # Minimum number of samples required at each leaf node
        min_samples_leaf = sp_randInt(2, 10)
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        ###  These are the RandomForest params ########        
        params = {
            #model_string+'criterion': criterion,
            model_string+'n_estimators': n_estimators,
            model_string+'max_features': max_features,
            #model_string+'max_depth': max_depth,
            #model_string+'min_samples_split': min_samples_split,
            #model_string+'min_samples_leaf': min_samples_leaf,
           #model_string+'bootstrap': bootstrap,
                       }
    elif model_name == 'bg':
        criterion = ["gini", "entropy", "log_loss"]
        # Number of base estimators in a Bagging is very small 
        n_estimators = sp_randInt(2,20)
        # Number of features to consider at every split
        #max_features = ['auto', 'sqrt', 'log']
        #max_features = sp_randFloat(0.3,0.9)
        max_features = [0.3, 0.6, 1.0]
        # Maximum number of levels in tree
        max_depth = sp_randInt(2, 10)
        # Minimum number of samples required to split a node
        min_samples_split = sp_randInt(2, 10)
        # Minimum number of samples required at each leaf node
        min_samples_leaf = sp_randInt(2, 10)
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        ###  These are the Bagging params ########
        params = {
            #model_string+'criterion': criterion,
            model_string+'n_estimators': n_estimators,
            model_string+'max_features': max_features,
            #model_string+'max_depth': max_depth,
            #model_string+'min_samples_split': min_samples_split,
            #model_string+'min_samples_leaf': min_samples_leaf,
            #model_string+'bootstrap': bootstrap,
            #model_string+'bootstrap_features': bootstrap,
                       }
    elif model_name == 'lgb':
        # Number of estimators in LGBM Classifier ##
        n_estimators = sp_randInt(100, 500)
        ### number of leaves is only for LGBM ###
        num_leaves = sp_randInt(5, 300)
        ## learning rate is very important for LGBM ##
        learning_rate = sp.stats.uniform(scale=1)
        params = {
            model_string+'n_estimators': n_estimators,
            #model_string+'num_leaves': num_leaves,
            model_string+'learning_rate': learning_rate,
                    }
    elif model_name == 'lp':
        params =  {
            ### Don't overly complicate this simple model. It works best with no tuning!
            model_string+'gamma': sp_randInt(0, 32),
            model_string+'kernel': ['knn', 'rbf'],
            #model_string+'max_iter': sp_randInt(50, 500),
            #model_string+'n_neighbors': sp_randInt(2, 5),
                }
    else:
        ### Since we don't know what model will be sent, we cannot tune it ##
        params = {}
        return model
    ### You must leave Random State as None since shuffle is False. Otherwise, error! 
    kfold = KFold(n_splits=5, random_state=None, shuffle=False)
    if verbose:
        print("Finding best params for base estimator using RandomizedSearchCV...")
    clf = RandomizedSearchCV(model, params, n_iter=n_iter, scoring=scoring,
                         cv = kfold, n_jobs=-1, random_state=100)
    
    clf.fit(X, y)

    if verbose:
        print("    best score is :" , clf.best_score_)
        #print("    best estimator is :" , clf.best_estimator_)
        print("    best Params is :" , clf.best_params_)
        print("Time Taken for RandomizedSearchCV: %0.0f (seconds)" %(time.time()-start))
    return clf.best_estimator_
##################################################################################
# Calculate class weight
from sklearn.utils.class_weight import compute_class_weight
import copy
from collections import Counter
def find_rare_class(classes, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    counts = OrderedDict(Counter(classes))
    total = sum(counts.values())
    if verbose >= 1:
        print('       Class  -> Counts -> Percent')
        sorted_keys = sorted(counts.keys())
        for cls in sorted_keys:
            print("%12s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
    if type(pd.Series(counts).idxmin())==str:
        return pd.Series(counts).idxmin()
    else:
        return int(pd.Series(counts).idxmin())
##################################################################################
from collections import OrderedDict    
def get_class_weights(y_input, verbose=0):    
    y_input = copy.deepcopy(y_input)
    if isinstance(y_input, np.ndarray):
        y_input = pd.Series(y_input)
    elif isinstance(y_input, pd.Series):
        pass
    elif isinstance(y_input, pd.DataFrame):
        if len(y_input.columns) >= 2:
            ### if it is a dataframe, return only if it is one column dataframe ##
            class_weights = dict()
            for each_target in y_input.columns:
                class_weights[each_target] = get_class_weights(y_input[each_target])
            return class_weights
        else:
            y_input = y_input.values.reshape(-1)
    else:
        ### if you cannot detect the type or if it is a multi-column dataframe, ignore it
        return None
    classes = np.unique(y_input)
    rare_class = find_rare_class(y_input)
    xp = Counter(y_input)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_input)
    class_weights = OrderedDict(zip(classes, np.round(class_weights/class_weights.min()).astype(int)))
    if verbose:
        print('Class weights used in classifier are: %s' %class_weights)
    return class_weights

from collections import OrderedDict
def get_scale_pos_weight(y_input, verbose=0):
    class_weighted_rows = get_class_weights(y_input)
    if isinstance(y_input, np.ndarray):
        y_input = pd.Series(y_input)
    elif isinstance(y_input, pd.Series):
        pass
    elif isinstance(y_input, pd.DataFrame):
        if len(y_input.columns) >= 2:
            ### if it is a dataframe, return only if it is one column dataframe ##
            rare_class_weights = OrderedDict()
            for each_target in y_input.columns:
                rare_class_weights[each_target] = get_scale_pos_weight(y_input[each_target])
            return rare_class_weights
        else:
            y_input = y_input.values.reshape(-1)
    
    rare_class = find_rare_class(y_input)
    rare_class_weight = class_weighted_rows[rare_class]
    if verbose:
        print('    For class %s, weight = %s' %(rare_class, rare_class_weight))
    return rare_class_weight
##########################################################################
from collections import defaultdict
from collections import OrderedDict
def return_minority_samples(y, verbose=0):
    """
    #### Calculates the % count of each class in y and returns a 
    #### smaller set of y based on being 5% or less of dataset.
    It returns the small y as an array or dataframe as input y was.
    """
    import copy
    y = copy.deepcopy(y)
    if isinstance(y, np.ndarray):
        ls = pd.Series(y).value_counts()[(pd.Series(y).value_counts(1)<=0.05).values].index
        return y[pd.Series(y).isin(ls).values]
    else:
        if isinstance(y, pd.Series):
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
        else:
            y = y.iloc[:,0]
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
        return y[y.isin(ls)]

def num_classes(y, verbose=0):
    """
    ### Returns number of classes in y
    """
    import copy
    y = copy.deepcopy(y)
    if isinstance(y, np.ndarray):
        ls = pd.Series(y).nunique()
    else:
        if isinstance(y, pd.Series):
            ls = y.nunique()
        else:
            if len(y.columns) >= 2:
                ls = OrderedDict()
                for each_i in y.columns:
                    ls[each_i] = y[each_i].nunique()
                return ls
            else:
                ls = y.nunique()[0]
    return ls

def return_minority_classes(y, verbose=0):
    """
    #### Calculates the % count of each class in y and returns a 
    #### smaller set of y based on being 5% or less of dataset.
    It returns the list of classes that are <=5% classes.
    """
    import copy
    y = copy.deepcopy(y)
    if isinstance(y, np.ndarray):
        ls = pd.Series(y).value_counts()[(pd.Series(y).value_counts(1)<=0.05).values].index
    else:
        if isinstance(y, pd.Series):
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
        else:
            y = y.iloc[:,0]
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
    return ls
#################################################################################
def get_cardinality(X, cat_features):
    ## pick a limit for cardinal variables here ##
    cat_limit = 30
    mask = X[cat_features].nunique() > cat_limit
    high_cardinal_vars = cat_features[mask]
    low_cardinal_vars = cat_features[~mask]
    return low_cardinal_vars, high_cardinal_vars
################################################################################
def is_y_object(y):
    test1 = (y.dtypes.any()==object) | (y.dtypes.any()==bool)
    test2 = str(y.dtypes.any())=='category'
    return test1 | test2

def print_flatten_dict(dd, separator='_', prefix=''):
    ### this function is to flatten dict to print classes and their order ###
    ### One solution here: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    ### I have modified it to make it work for me #################
    return { prefix + separator + str(k) if prefix else k : v
             for kk, vv in dd.items()
             for k, v in print_flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def print_accuracy(target, y_test, y_preds, verbose=0):
    bal_scores = []
    
    from sklearn.metrics import balanced_accuracy_score, classification_report
    if isinstance(target, str): 
        bal_score = balanced_accuracy_score(y_test,y_preds)
        bal_scores.append(bal_score)
        if verbose:
            print('Bal accu %0.0f%%' %(100*bal_score))
            print(classification_report(y_test,y_preds))
    elif len(target) <= 1:
        bal_score = balanced_accuracy_score(y_test,y_preds)
        bal_scores.append(bal_score)
        if verbose:
            print('Bal accu %0.0f%%' %(100*bal_score))
            print(classification_report(y_test,y_preds))
    else:
        for each_i, target_name in enumerate(target):
            bal_score = balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])
            bal_scores.append(bal_score)
            if verbose:
                if each_i == 0:
                    print('For %s:' %target_name)
                print('    Bal accu %0.0f%%' %(100*bal_score))
                print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
    return np.mean(bal_scores)
##########################################################################################
from collections import defaultdict
def return_predict_proba(y_probas):
    ### This is for detecting what-label what-class problems with probas ####
    problemtype = ""
    if isinstance(y_probas, list):
        ### y_probas is a list when y is multi-label. 
        if isinstance(y_probas[0], list):
            ##    1. If y is multi_label but has more than two classes, y_probas[0] is also a list ##
            problemtype = "multi_label_multi_class"
        else:
            initial = y_probas[0].shape[1]
            if np.all([x.shape[1]==initial for x in y_probas]):
                problemtype =  "multi_label_binary_class"
            else:
                problemtype = "multi_label_multi_class"
    else:
        problemtype = "single_label"
    #### This is for making multi-label multi-class predictions into a dictionary ##
    if problemtype == "multi_label_multi_class":
        probas_dict = defaultdict(list)
        ### Initialize the default dict #############
        for each_target in range(len(y_probas[0])):
            probas_dict[each_target] = []
        #### Now that it is is initialized, compile each class into its own list ###
        if isinstance(y_probas[0], list):
            for each_i in range(len(y_probas)):
                for each_j in range(len(y_probas[each_i])):
                    if y_probas[each_i][each_j].shape[1] > 2:
                        probas_dict[each_j].append(y_probas[each_i][each_j])
                    else:
                        probas_dict[each_j].append(y_probas[each_i][each_j][:,1])
            #### Once all of the probas have been put in a dictionary, now compute means ##
            for each_target in range(len(probas_dict)):
                probas_dict[each_target] = np.array(probas_dict[each_target]).mean(axis=0)
    elif problemtype == "multi_label_binary_class":
        initial = y_probas[0].shape[1]
        if np.all([x.shape[1]==initial for x in y_probas]):
            probas_dict = np.array(y_probas).mean(axis=0)
    return probas_dict   
###############################################################################################
from sklearn.metrics import roc_auc_score
import copy
from sklearn.metrics import balanced_accuracy_score, classification_report
import pdb
def print_sulo_accuracy(y_test, y_preds, y_probas='', verbose=0):
    bal_scores = []
    ####### Once you have detected what problem it is, now print its scores #####
    if y_test.ndim <= 1: 
        ### This is a single label problem # we need to test for multiclass ##
        bal_score = balanced_accuracy_score(y_test,y_preds)
        print('Bal accu %0.0f%%' %(100*bal_score))
        if not isinstance(y_probas, str):
            if y_probas.ndim <= 1:
                print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
            else:
                if y_probas.shape[1] == 2:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                else:
                    print('Multi-class ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
        bal_scores.append(bal_score)
        if verbose:
            print(classification_report(y_test,y_preds))
    elif y_test.ndim >= 2:
        if y_test.shape[1] == 1:
            bal_score = balanced_accuracy_score(y_test,y_preds)
            bal_scores.append(bal_score)
            print('Bal accu %0.0f%%' %(100*bal_score))
            if not isinstance(y_probas, str):
                if y_probas.shape[1] > 2:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
                else:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
            if verbose:
                print(classification_report(y_test,y_preds))
        else:
            if isinstance(y_probas, str):
                ### This is for multi-label problems without probas ####
                for each_i in range(y_test.shape[1]):
                    bal_score = balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])
                    bal_scores.append(bal_score)
                    print('    Bal accu %0.0f%%' %(100*bal_score))
                    if verbose:
                        print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
            else:
                ##### This is only for multi_label_multi_class problems
                num_targets = y_test.shape[1]
                for each_i in range(num_targets):
                    print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])))
                    if len(np.unique(y_test.values[:,each_i])) > 2:
                        ### This nan problem happens due to Label Propagation but can be fixed as follows ##
                        mat = y_probas[each_i]
                        if np.any(np.isnan(mat)):
                            mat = pd.DataFrame(mat).fillna(method='ffill').values
                            bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                        else:
                            bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                    else:
                        if isinstance(y_probas, dict):
                            if y_probas[each_i].ndim <= 1:
                                ## This is caused by Label Propagation hence you must probas like this ##
                                mat = y_probas[each_i]
                                if np.any(np.isnan(mat)):
                                    mat = pd.DataFrame(mat).fillna(method='ffill').values
                                bal_score = roc_auc_score(y_test.values[:,each_i],mat)
                            else:
                                bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i][:,1])
                        else:
                            if y_probas.shape[1] == num_targets:
                                ### This means Label Propagation was used which creates probas like this ##
                                bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[:,each_i])
                            else:
                                ### This means regular sklearn classifiers which predict multi dim probas #
                                bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i])
                    print('Target number %s: ROC AUC score %0.0f%%' %(each_i+1,100*bal_score))
                    bal_scores.append(bal_score)
                    if verbose:
                        print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
    final_score = np.mean(bal_scores)
    if verbose:
        print("final average balanced accuracy score = %0.2f" %final_score)
    return final_score
##############################################################################
import os
def check_if_GPU_exists():
    try:
        os.environ['NVIDIA_VISIBLE_DEVICES']
        print('GPU available on this device. Please activate it to speed up lightgbm.')
        return True
    except:
        print('No GPU available on this device. Using CPU for lightgbm and others.')
        return False
###############################################################################
def get_max_min_from_y(actuals):
    if isinstance(actuals, pd.Series):
        Y_min = actuals.values.min()
        Y_max = actuals.values.max()
    elif isinstance(actuals, pd.DataFrame):
        Y_min = actuals.values.ravel().min()
        Y_max = actuals.values.ravel().max()
    else:
        Y_min = actuals.min()
        Y_max = actuals.max()
    return np.array([Y_min, Y_max])

def convert_regression_to_classes(predictions, actuals):
    if isinstance(actuals, pd.Series):
        Y_min = actuals.values.min()
        Y_max = actuals.values.max()
    elif isinstance(actuals, pd.DataFrame):
        Y_min = actuals.values.ravel().min()
        Y_max = actuals.values.ravel().max()
    else:
        Y_min = actuals.min()
        Y_max = actuals.max()
    predictions = np.round(predictions,0).astype(int)
    predictions = np.where(predictions<Y_min, Y_min, predictions)
    predictions = np.where(predictions>Y_max, Y_max, predictions)
    return predictions
###############################################################################
from sklearn.metrics import mean_squared_error

def rmse(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted, squared=False)
##############################################################################
class SuloRegressor(BaseEstimator, RegressorMixin):
    """
    SuloRegressor stands for Super Learning Optimized (SULO) Regressor.
    -- It works on small as well as big data. It works in Integer mode as well as float-mode.
    -- It works on regular balanced data as well as skewed regression targets.
    The reason it works so well is that Sulo is an ensemble of highly tuned models.
    -- You don't have to send any inputs but if you wanted to, you can spefify multiple options.
    It is fully compatible with scikit-learn pipelines and other models.

    Syntax:
        sulo = SuloRegressor(base_estimator=None, n_estimators=None, pipeline=False, 
                                imbalanced=False, integers_only=False, log_transform=False, 
                                time_series = False, verbose=0)
        sulo.fit(X_train, y_train)
        y_preds = sulo.predict(X_test)

    Inputs:
        n_estimators: default is None. Number of models you want in the final ensemble.
        base_estimator: default is None. Base model you want to train in each of the ensembles.
        pipeline: default is False. It will transform all data to numeric automatically if set to True.
        imbalanced: default is False. It will activate a special imbalanced Regressor if set to True.
        integers_only: default is False. It will perform Integer regression if set to True
        log_transform: default is False. It will perform log transformation of target if set to True.
        time_series: default is False. If set to True, an expanding window Time Series split is performed.
            Instead, use "sliding" if you want it to perform a sliding window time series split.
        verbose: default is 0. It will print verbose output if set to True.

    Oututs:
        regressor: returns a regression model highly tuned to your specific needs and dataset.
    """
    def __init__(self, base_estimator=None, n_estimators=None, pipeline=False, imbalanced=False, 
                                       integers_only=False, log_transform=False, 
                                       time_series = False, verbose=0):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.pipeline = pipeline
        self.imbalanced = imbalanced
        self.integers_only = integers_only
        self.log_transform = log_transform
        self.time_series = time_series 
        self.verbose = verbose
        self.models = []
        self.multi_label =  False
        self.scores = []
        self.integers_only_min_max = []
        self.model_name = ''
        self.features = []

    def fit(self, X, y):
        X = copy.deepcopy(X)
        print('Input data shapes: X = %s' %(X.shape,))
        print('    y shape = %s' %(y.shape,))
        seed = 42
        shuffleFlag = True
        modeltype = 'Regression'
        features_limit = 50 ## if there are more than 50 features in dataset, better to use LGBM ##
        start = time.time()
        if isinstance(X, pd.DataFrame):
            self.features = X.columns.tolist()
        else:
            print('Cannot operate SuloClassifier on numpy arrays. Must be dataframes. Returning...')
            return self
        # Use KFold for understanding the performance
        if self.imbalanced:
            print('Remember that using class weights will wrongly skew predict_probas from any classifier')
        ### Remember that putting class weights will totally destroy predict_probas ###
        gpu_exists = check_if_GPU_exists()
        if gpu_exists:
            device="gpu"
        else:
            device="cpu"
        row_limit = 10000
        if self.integers_only:
            self.integers_only_min_max = get_max_min_from_y(y)
            print('    Min and max values of y = %s' %self.integers_only_min_max)
        ################          P I P E L I N E        ##########################
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean", add_indicator=True)), ("scaler", StandardScaler())])

        categorical_transformer_low = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", OneHotEncoder(handle_unknown="value")),])

        categorical_transformer_high = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", LabelEncoder()),])

        numeric_features = X.select_dtypes(include='number').columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_cardinality(X, categorical_features)
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        ####################################################################################
        data_samples = X.shape[0]
        if self.n_estimators is None:
            if data_samples <= row_limit:
                self.n_estimators = min(5, int(1.5*np.log10(data_samples)))
            else:
                self.n_estimators = min(5, int(1.2*np.log10(data_samples)))
        num_splits = self.n_estimators
        self.model_name = 'lgb'
        num_repeats = 1
        ####################################################################################
        ########## Perform Time Series Split here ########################################
        ####################################################################################
        if self.time_series:
            if isinstance(self.time_series, str):
                if self.time_series == 'sliding':
                    print('Performing sliding window time series split...')
                    kfold = SlidingTimeSeriesSplit(n_splits=num_splits, fixed_length=True, train_splits=2, test_splits=1)
                else:
                    print('Performing expanding window time series split...')
                    kfold = TimeSeriesSplit(n_splits=num_splits)
            else:
                print('Performing expanding window time series split...')
                kfold = TimeSeriesSplit(n_splits=num_splits)
        else:
            print('Performing KFold regular split...')
            kfold = RepeatedKFold(n_splits=num_splits, random_state=seed, n_repeats=num_repeats)
        ####################################################################################
        num_iterations = int(num_splits * num_repeats)
        scoring = 'neg_mean_squared_error'
        print('    Num. estimators = %s (will be larger than n_estimators since kfold is repeated twice)' %num_iterations)
        ##### This is where we check if y is single label or multi-label ##
        if isinstance(y, pd.DataFrame):
            if self.log_transform:
                if self.verbose:
                    print('    log transforming target variable: ensure no zeros in target. Otherwise error.')
                try:
                    y = np.log1p(y)
                except:
                    print('    Error: log transforming targets. Remove zeros in target and try again.')
                    return self
            ###############################################################
            ### This is for Multi-Label Regression problems only #####################
            ###############################################################
            targets = y.columns.tolist()
            if is_y_object(y):
                print('Cannot perform Regression using object or string targets. Please convert to numeric and try again.')
                return self
            if len(targets) > 1:
                self.multi_label = y.columns.tolist()
                ### You need to initialize the class before each run - otherwise, error!
                ### Remember we don't to HPT Tuning for Multi-label problems since it errors ####
                i = 0
                for train_index, test_index in tqdm(kfold.split(X, y), total=kfold.get_n_splits(), 
                                        desc="k-fold training"):
                    start_time = time.time()
                    #random_seed = np.random.randint(2,100)
                    random_seed = 999
                    if self.base_estimator is None:
                        ################################################################
                        ###   This is for Single Label Regression problems only ########
                        ###   Make sure you don't do imbalanced SMOTE work here  #######
                        ################################################################
                        if y.shape[0] <= row_limit:
                            if self.integers_only:
                                if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                    print('    Selecting Ridge as base_estimator. Feel free to send in your own estimator.')
                                    self.base_estimator = Ridge(normalize=False)
                                    self.model_name = 'other'
                                else:
                                    if self.verbose:
                                        print('    Selecting LGBM Regressor as base estimator...')
                                    self.base_estimator = LGBMRegressor(n_jobs=-1, device=device, random_state=random_seed)                                    
                            else:
                                if len(self.features) <= features_limit:
                                    if self.verbose:
                                        print('    Selecting Bagging Regressor since integers_only flag is set...')
                                    ET = ExtraTreeRegressor()
                                    ET = LinearSVR()
                                    ET = DecisionTreeRegressor()
                                    self.base_estimator = BaggingRegressor(base_estimator = ET, bootstrap_features=False,
                                                            n_jobs=-1, random_state=random_seed)
                                    self.model_name = 'bg'
                                else:
                                    if gpu_exists:
                                        if self.verbose:
                                            print('    Selecting XGBRegressor with GPU as base estimator...')
                                        self.base_estimator = XGBRegressor( 
                                            n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                                    else:
                                        if self.verbose:
                                            print('    Selecting LGBM Regressor as base estimator...')
                                        self.base_estimator = LGBMRegressor(n_jobs=-1,device=device, random_state=random_seed)                                    
                        else:
                            if gpu_exists:
                                if self.verbose:
                                    print('    Selecting XGB Regressor with GPU as base estimator...')
                                self.base_estimator = XGBRegressor( 
                                    n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                            else:
                                if self.verbose:
                                    print('    Selecting LGBM Regressor as base estimator...')
                                self.base_estimator = LGBMRegressor(n_jobs=-1,device=device, random_state=random_seed)
                    else:
                        self.model_name == 'other'
                    # Split data into train and test based on folds          
                    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
                    else:
                        y_train, y_test = y[train_index], y[test_index]

                    if isinstance(X, pd.DataFrame):
                        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                    else:
                        x_train, x_test = X[train_index], X[test_index]

                    ###### Do this only the first time ################################################
                    if i == 0:
                        ### It does not make sense to do hyper-param tuning for multi-label models ##
                        ###    since ClassifierChains do not have many hyper params #################
                        #self.base_estimator = rand_search(self.base_estimator, x_train, y_train, 
                        #                        self.model_name, verbose=self.verbose)
                        #print('    hyper tuned base estimator = %s' %self.base_estimator)
                        if self.verbose:
                            print('    Fitting with RegressorChain...')
                        est_list = [RegressorChain(self.base_estimator, order=None, random_state=i)
                                    for i in range(num_iterations)] 

                    # Initialize model with your supervised algorithm of choice
                    model = est_list[i]

                    # Train model and use it to train on the fold
                    if self.pipeline:
                        ### This is only with a pipeline ########
                        pipe = Pipeline(
                            steps=[("preprocessor", preprocessor), ("model", model)]
                        )

                        pipe.fit(x_train, y_train)
                        self.models.append(pipe)

                        # Predict on remaining data of each fold
                        preds = pipe.predict(x_test)

                    else:
                        #### This is without a pipeline ###
                        model.fit(x_train, y_train)
                        self.models.append(model)

                        # Predict on remaining data of each fold
                        preds = model.predict(x_test)

                        if self.log_transform:
                            preds = np.expm1(preds)
                        elif self.integers_only:
                            ### Use Regression predictions and convert them into integers here ##
                            preds = np.round(preds,0).astype(int)
                            
                    score = print_regression_model_stats(y_test, preds, self.verbose)
                    if i == 0:
                        y_stack = copy.deepcopy(y_test)
                        pred_stack = copy.deepcopy(preds)
                    else:
                        y_stack = pd.concat([y_stack,y_test], axis=0)
                        pred_stack = np.r_[pred_stack,preds]

                    if self.verbose:
                        print("    Fold %s: out-of-fold RMSE (smaller is better): %0.3f" %(i+1, score))
                    self.scores.append(score)
                    
                    # Finally, check out the total time taken
                    end_time = time.time()
                    timeTaken = end_time - start_time
                    if self.verbose:
                        print("Time Taken for fold %s: %0.0f (seconds)" %(i+1, timeTaken))
                
                    i += 1

                # Compute average score
                print("######## Final Regressor metrics: ##############")
                normaverageAccuracy = print_regression_model_stats(y_stack, pred_stack, self.verbose)
                end = time.time()
                timeTaken = end - start
                print("Time Taken overall: %0.0f (seconds)" %(timeTaken))
                self.kf = kfold
                return self
        ########################################################
        #####  This is for Single Label Regression problems 
        ########################################################
        if isinstance(y, pd.Series):
            targets = y.name
        else:
            targets = []

        est_list = num_iterations*[self.base_estimator]
        
        ### if there is a need to do SMOTE do it here ##
        smote = False
        #list_classes = return_minority_classes(y)
        #if not list_classes.empty:
        #    smote = True
        #### For now, don't do SMOTE since it is making things really slow ##
        
        # Perform CV
        i = 0
        for train_index, test_index in tqdm(kfold.split(X, y), total=kfold.get_n_splits(), 
                                            desc="k-fold training"):
            start_time = time.time()
            random_seed = np.random.randint(2,100)
            if self.base_estimator is None:
                if data_samples <= row_limit:
                    ### For small datasets use RFR for Regressions   ########################
                    if len(self.features) <= features_limit:
                        if self.verbose:
                            print('    Selecting Bagging Regressor for this dataset...')
                        ### The Bagging Regresor outperforms ETC most of the time ####
                        ET = ExtraTreeRegressor()
                        ET = LinearSVR()
                        ET = DecisionTreeRegressor()
                        self.base_estimator = BaggingRegressor(base_estimator = ET, bootstrap_features=False,
                                                            n_jobs=-1, random_state=random_seed)
                        self.model_name = 'bg'
                    else:
                        if (X.dtypes==float).all():
                            if self.verbose:
                                print('    Selecting Ridge as base_estimator. Feel free to send in your own estimator.')
                            self.base_estimator = Ridge(normalize=False)
                            self.model_name = 'other'
                        else:
                            if self.verbose:
                                print('    Selecting LGBM Regressor as base estimator...')
                            self.base_estimator = LGBMRegressor(n_jobs=-1,  random_state=random_seed)
                else:
                    ### For large datasets Better to use LGBM
                    if data_samples >= 1e5:
                        if gpu_exists:
                            if self.verbose:
                                print('    Selecting XGBRegressor with GPU as base estimator...')
                            self.base_estimator = XGBRegressor(n_jobs=-1,tree_method = 'gpu_hist',
                                gpu_id=0, predictor="gpu_predictor")
                        else:
                            self.base_estimator = LGBMRegressor(n_jobs=-1, random_state=random_seed) 
                    else:
                        ### For smaller than Big Data, use Label Propagation which is faster and better ##
                        if (X.dtypes==float).all():
                            if self.verbose:
                                print('    Selecting Ridge as base_estimator. Feel free to send in your own estimator.')
                            self.base_estimator = Ridge(normalize=False)
                            self.model_name = 'other'
                        else:
                            if len(self.features) <= features_limit:
                                if self.verbose:
                                    print('    Selecting Bagging Regressor for this dataset...')
                                ET = ExtraTreeRegressor()
                                ET = LinearSVR()
                                ET = DecisionTreeRegressor()
                                self.base_estimator = BaggingRegressor(base_estimator = ET, bootstrap_features=False,
                                                            n_jobs=-1, random_state=random_seed)
                                self.model_name = 'bg'
                            else:
                                scoring = 'neg_mean_squared_error'
                                ###   Extra Trees is not so great for large data sets - LGBM is better ####
                                if gpu_exists:
                                    if self.verbose:
                                        print('    Selecting XGBRegressor with GPU as base estimator...')
                                    self.base_estimator = XGBRegressor(
                                        n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                                else:
                                    if self.verbose:
                                        print('    Selecting LGBM Regressor as base estimator...')
                                    self.base_estimator = LGBMRegressor(n_jobs=-1, random_state=random_seed) 
                                self.model_name = 'lgb'
            else:
                self.model_name = 'other'
            # Split data into train and test based on folds          
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
            else:
                y_train, y_test = y[train_index], y[test_index]

            if isinstance(X, pd.DataFrame):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            else:
                x_train, x_test = X[train_index], X[test_index]

            # Convert the data into numpy arrays
            #if not isinstance(x_train, np.ndarray):
            #    x_train, x_test = x_train.values, x_test.values
            
            ##   small datasets processing #####
            if i == 0:
                if self.pipeline:
                    # Train model and use it in a pipeline to train on the fold  ##
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("model", self.base_estimator)])
                    if self.model_name == 'other':
                        print('No HPT tuning performed since base estimator is given by input...')
                        self.base_estimator = copy.deepcopy(pipe)
                    else:
                        if len(self.features) <= features_limit:
                            self.base_estimator = rand_search(pipe, x_train, y_train, 
                                                    self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')
                            self.base_estimator = copy.deepcopy(pipe)
                else:
                    ### This is for without a pipeline #######
                    if self.model_name == 'other':
                        ### leave the base estimator as is ###
                        print('No HPT tuning performed since base estimator is given by input...')
                    else:
                        if len(self.features) <= features_limit:
                            ### leave the base estimator as is ###
                            self.base_estimator = rand_search(self.base_estimator, x_train, 
                                                y_train, self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')

                est_list = num_iterations*[self.base_estimator]
                #print('    base estimator = %s' %self.base_estimator)
                        
            # Initialize model with your supervised algorithm of choice
            model = est_list[i]
            
            model.fit(x_train, y_train)
            self.models.append(model)

            # Predict on remaining data of each fold
            preds = model.predict(x_test)

            if self.log_transform:
                preds = np.expm1(preds)
            elif self.integers_only:
                ### Use Regression predictions and convert them into integers here ##
                preds = np.round(preds,0).astype(int)

            score = print_regression_model_stats(y_test, preds)
            if self.verbose:
                print("    Fold %s: out-of-fold RMSE (smaller is better): %0.3f" %(i+1, score))
                print("              Normalized RMSE (smaller is better): %0.3f" %(score/np.std(y_test)))
            self.scores.append(score)

            i += 1

        # Compute average score
        averageAccuracy = sum(self.scores)/len(self.scores)
        print("Final RMSE of %s-estimator SuloRegressor: %0.3f" %(num_iterations, averageAccuracy))
        print("    Final Normalized RMSE: %0.1f%%" %(100*averageAccuracy/np.std(y_test)))

        # Finally, check out the total time taken
        end = time.time()
        timeTaken = end-start
        print("Time Taken: %0.0f (seconds)" %timeTaken)
        self.kf = kfold

        return self

    def predict(self, X):
        from scipy import stats
        weights = 1/np.array(self.scores)
        if self.multi_label:
            ### In multi-label, targets have to be numeric, so you can leave weights as-is ##
            ypre = np.array([model.predict(X) for model in self.models ])
            y_predis = np.average(ypre, axis=0, weights=weights)
            if self.log_transform:
                y_predis = np.expm1(y_predis)
            ### leave the next line as if since you want to check for it separately
            if self.integers_only:
                y_predis = np.round(y_predis,0).astype(int)
            return y_predis
        y_predis = np.column_stack([model.predict(X) for model in self.models ])
        ### This weights the model's predictions according to OOB scores obtained
        #### In single label, targets can be object or string, so weights cannot be applied always ##
        y_predis = np.average(y_predis, weights=weights, axis=1)
        if self.log_transform:
            y_predis = np.expm1(y_predis)
        if self.integers_only:
            y_predis = np.round(y_predis,0).astype(int)
        return y_predis
    
    def predict_proba(self, X):
        print('Error: In regression problems, no probabilities can be obtained. Returning...')
        return X

    def return_worst_fold(self, X):
        """
        This method returns the worst performing train and test rows among all the folds.
        This is very important information since it helps an ML engineer or Data Scientist 
            to trouble shoot time series problems. It helps to find where the model is struggling.

        Inputs:
        --------
        X: Dataframe. This must be the features dataframe of your dataset. It cannot be a numpy array.

        Outputs:
        ---------
        train_rows_dataframe, test_rows_dataframe: Dataframes. 
             This returns the portion of X as train and test to help you understand where model struggled.
        """
        worst_fold = np.argmax(self.scores)
        for i, (tr, tt) in enumerate(self.kf.split(X)):
            if i == worst_fold:
                worst_train_rows = copy.deepcopy(tr)
                worst_ttest_rows = copy.deepcopy(tt)
                print("fold %s: train rows index = %s. Sulo model struggled in this fold." %(i+1,tr))
            else:
                print("fold %s: train rows index = %s" %(i+1,tr))
        return X.iloc[worst_train_rows], X.iloc[worst_ttest_rows]

    def plot_importance(self, max_features=10):
        import lightgbm as lgbm
        from xgboost import plot_importance
        model_name = self.model_name
        feature_names = self.features

        if  model_name == 'lgb' or model_name == 'xgb':
            for i, model in enumerate(self.models):
                if self.pipeline:
                    if self.multi_label:
                        #model_object = model.named_steps['model'].base_estimator
                        print('No feature importances available for multi_label targets. Returning...')
                        return
                    else:
                        model_object = model.named_steps['model']
                else:
                    if self.multi_label:
                        #model_object = model.base_estimator
                        print('No feature importances available for multi_label targets. Returning...')
                        return
                    else:
                        model_object = model
                feature_importances = model_object.booster_.feature_importance(importance_type='gain')
                if i == 0:
                    feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                else:
                    feature_imp = pd.concat([feature_imp, 
                        pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                #lgbm.plot_importance(model_object, importance_type='gain', max_num_features=max_features)
            feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
            feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            ### This is for XGB ###
            #plot_importance(self.model.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == 'lp':
            print('No feature importances available for LabelPropagation algorithm. Returning...')
            return
        elif model_name == 'rf':
            ### These are for RandomForestClassifier kind of scikit-learn models ###
            try:
                for i, model in enumerate(self.models):
                    if self.pipeline:
                        if self.multi_label:
                            #model_object = model.named_steps['model'].base_estimator
                            print('No feature importances available for multi_label targets. Returning...')
                            return
                        else:
                            model_object = model.named_steps['model']
                    else:
                        if self.multi_label:
                            #model_object = model.base_estimator
                            print('No feature importances available for multi_label targets. Returning...')
                            return
                        else:
                            model_object = model
                    feature_importances = model_object.feature_importances_
                    if i == 0:
                        feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                    else:
                        feature_imp = pd.concat([feature_imp, 
                            pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
                feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            except:
                    print('Could not plot feature importances. Please check your model and try again.')                
        else:
            print('No feature importances available for this algorithm. Returning...')
            return
##########################################################################################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
def print_regression_model_stats(actuals, predicted, verbose=0):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    if isinstance(actuals,pd.Series) or isinstance(actuals,pd.DataFrame):
        actuals = actuals.values
    if isinstance(predicted,pd.Series) or isinstance(predicted,pd.DataFrame):
        predicted = predicted.values
    if len(actuals) != len(predicted):
        if verbose:
            print('Error: Number of rows in actuals and predicted dont match. Continuing...')
        return np.inf
    try:
        ### This is for Multi_Label Problems ###
        assert actuals.shape[1]
        multi_label = True
    except:
        multi_label = False
    if multi_label:
        for i in range(actuals.shape[1]):
            actuals_x = actuals[:,i]
            try:
                predicted_x = predicted[:,i]
            except:
                predicted_x = predicted[:]
            if verbose:
                print('for target %s:' %i)
            each_rmse = print_regression_metrics(actuals_x, predicted_x, verbose)
        final_rmse = np.mean(each_rmse)
    else:
        final_rmse = print_regression_metrics(actuals, predicted, verbose)
    return final_rmse
################################################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def MAPE(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

def print_regression_metrics(y_true, y_preds, verbose=0):
    each_rmse = np.sqrt(mean_squared_error(y_true, y_preds))
    if verbose:
        print('    RMSE = %0.3f' %each_rmse)
        print('    Norm RMSE = %0.0f%%' %(100*np.sqrt(mean_squared_error(y_true, y_preds))/np.std(y_true)))
        print('    MAE = %0.3f'  %mean_absolute_error(y_true, y_preds))
    if len(y_true[(y_true==0)]) > 0:
        if verbose:
            print('    WAPE = %0.0f%%, Bias = %0.0f%%' %(100*np.sum(np.abs(y_true-y_preds))/np.sum(y_true), 
                        100*np.sum(y_true-y_preds)/np.sum(y_true)))
            print('    No MAPE available since zeroes in actuals')
    else:
        if verbose:
            print('    WAPE = %0.0f%%, Bias = %0.0f%%' %(100*np.sum(np.abs(y_true-y_preds))/np.sum(y_true), 
                        100*np.sum(y_true-y_preds)/np.sum(y_true)))
            print('    MAPE = %0.0f%%' %(100*MAPE(y_true, y_preds)))
    print('    R-Squared = %0.0f%%' %(100*r2_score(y_true, y_preds)))
    plot_regression(y_true, y_preds, chart='scatter')
    return each_rmse
################################################################################
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev
################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
def print_rmse(y, y_hat):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def print_mape(y, y_hat):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    To avoid infinity due to division by zero, we select max(0.01, abs(actuals)) to show MAPE.
    """
    ### Wherever there is zero, replace it with 0.001 so it doesn't result in division by zero
    perc_err = (100*(np.where(y==0,0.001,y) - y_hat))/np.where(y==0,0.001,y)
    return np.mean(abs(perc_err))
    
def plot_regression(actuals, predicted, chart='scatter'):
    """
    This function plots the actuals vs. predicted as a line plot.
    You can change the chart type to "scatter' to get a scatter plot.
    """
    figsize = (10, 10)
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    plt.figure(figsize=figsize)
    if not isinstance(actuals, np.ndarray):
        actuals = actuals.values
    dfplot = pd.DataFrame([actuals,predicted]).T
    dfplot.columns = ['Actual','Forecast']
    dfplot = dfplot.sort_index()
    lineStart = actuals.min()
    lineEnd = actuals.max()
    if chart == 'line':
        plt.plot(dfplot)
    else:
        plt.scatter(actuals, predicted, color = next(colors), alpha=0.5,label='Predictions')
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = next(colors))
        plt.xlim(lineStart, lineEnd)
        plt.ylim(lineStart, lineEnd)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Model: Predicted vs Actuals', fontsize=12)
    plt.show();
############################################################################################
#########   This is where SULOCLASSIFIER and SULOREGRESSOR END   ###########################
############################################################################################
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
class SlidingTimeSeriesSplit(TimeSeriesSplit):
    """
    The SlidingTimeSeriesSplit is based on the following response from Stackoverflow:
    https://stackoverflow.com/questions/58295242/sliding-window-train-test-split-for-time-series-data
    Many thanks to the poster and author of this code snippet. 
    I have modified it slightly to suit my needs.
    """

    def __init__(self, n_splits=5, groups=None, fixed_length=None, train_splits=2, test_splits=1):
        self.n_splits = n_splits
        self.groups = groups
        self.fixed_length = fixed_length
        self.train_splits = train_splits
        self.test_splits = test_splits

    def split(self, X, y=None):
        groups = self.groups
        fixed_length = self.fixed_length
        train_splits = self.train_splits
        test_splits = self.test_splits
        n_splits = self.n_splits
        ###################################
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) <= 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],indices[test_start:test_start + test_size])
############################################################################################
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg_log(yhat, y):
    y = np.power(10, y.get_label())
    yhat = np.power(10, yhat)
    return "rmspe", rmspe(y,yhat)

def rmspe_xg(yhat, y):
    return "rmspe", rmspe(y.get_label(),yhat)
#############################################################################################
from sklearn.model_selection import train_test_split
def xgboost_regressor( X_train, y_train, X_test, 
                    log_transform=False, sparsity=False, eta=0.1):
    """
    Perform training using XGBoost with a log based target transformation pipeline.
    ######## Remember that we use this only for Regression problems ###############
    This function uses log transformation of target variable to make it easier.
    If sparsity is set, then target where zeros are set to small value such as 0.1.
    This happens frequently in retail datasets where imbalanced datasets are common. 

    Inputs:
    X_train: can be array or DataFrame of train predictor values
    y_train: can be array or DataFrame of train target values
    X_test: can be array or dataframe of test predictor values

    Output:
    preds1: predictions on X_test which can be compared to y_test or submitted.
    """
    try:
        import xgboost as xgb
    except:
        print('You must have xgboost installed in your machine to try this function.')
        return
    ###### If they give sparsity flag then set the target values toa low number ###
    if sparsity:
        print('Since sparsity is set True, replace zero valued target rows with a small 0.1 value...')
        if isinstance(y_train, np.ndarray):
            y_train[(y_train==0)] = 0.1
        elif isinstance(y_train, pd.Series):
            ### Since we want sparsity we are going to set target values that are zero as near zero
            target = y_train.name
            if len(y_train.loc[(y_train==0)]) > 0:
                y_train.loc[(y_train==0)] = 0.1
        elif isinstance(y_train, pd.DataFrame):
            targets = y_train.columns.tolist()
            for each_target in targets:
                if len(y_train.loc[(y_train[each_target]==0)]) > 0:
                    y_train.loc[(y_train[each_target]==0), each_target] = 0.1

    ##### This is where we start the split into train and valid to tune the model ###
    start_time = time.time()
    X_train1, X_valid, y_train1, y_valid = train_test_split(
                X_train, y_train, test_size=0.02, random_state=10)
    if log_transform:
        y_train1 = np.log10(y_train1)
        y_valid = np.log10(y_valid)
    print(X_train1.shape, X_valid.shape)

    ##### Now let us transform X_train1 and y_train1 with lazytransformer #####
    lazy = LazyTransformer(model=None, encoders='label', scalers='', 
                        transform_target=False, imbalanced=False, verbose=1)
    X_train1, y_train1 = lazy.fit_transform(X_train1, y_train1)
    X_valid1, y_valid1 = lazy.transform(X_valid, y_valid)

    ########   Now let's perform randomized search to find best hyper parameters ######
    print('#### Model training using XGB ######## eta = %s' %eta)
    params = {"objective": "reg:squarederror",
              "booster" : "gbtree",
              "eta": eta,
              "silent": 1,
              "seed": 99
              }
    num_boost_round = 500
    ##### now do the training ###########
    dtrain = xgb.DMatrix(X_train1, y_train1)
    dvalid = xgb.DMatrix(X_valid1, y_valid1)
    ##########   Now we use rmspe_xg ###################
    if log_transform:
        feval = rmspe_xg_log
    else:
        feval = rmspe_xg
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
      early_stopping_rounds=10, feval=feval, verbose_eval=True)
    print('Training Completed')

    #### This is where you grid search the pipeline now ##############
    print("Validating")
    #### If it is not sparse data, you must still transform it into an array ###
    if isinstance(y_valid1, pd.Series):
        y_valid1 = y_valid1.values
    elif isinstance(y_valid1, pd.DataFrame):
        y_valid1 = y_valid1.values.ravel()
    yhat = gbm.predict(xgb.DMatrix(X_valid1))
    error = rmspe(y_valid1, yhat)
    print('RMSPE: {:.6f}'.format(error))

    ##### This is where we search for hyper params for model #######
    X_test1 = lazy.transform(X_test)
    dtest = xgb.DMatrix(X_test1)
    preds1 = gbm.predict(dtest)
    # Return predictions ###
    if log_transform:
        preds1 = np.power(10, preds1)
    from xgboost import plot_importance
    plot_importance(gbm)
    return preds1
#################################################################################
from scipy.stats import probplot,skew
def data_suggestions(data):
    """
    Modified by Ram Seshadri. Original idea for data suggestions module was a Kaggler.
    Many thanks to: https://www.kaggle.com/code/itkin16/catboost-on-gpu-baseline
    """
    maxx = []
    minn = []
    all_cols = list(data)
    cat_cols1 = data.select_dtypes(include='object').columns.tolist()
    cat_cols2 = data.select_dtypes(include='category').columns.tolist()
    cat_cols = list(set(cat_cols1+cat_cols2))
    ### The next line may look odd but due to different versions of pandas which
    ### treat the definition of float differently, I am forced to use this. Don't change it.
    num_cols = data.select_dtypes(include='float16').columns.tolist() + data.select_dtypes(
                    include='float32').columns.tolist() + data.select_dtypes(include='float64').columns.tolist()
    non_num_cols = left_subtract(all_cols, num_cols)
    for i in data.columns:
        if i not in cat_cols:
            ### for float and integer, no need to calculate this ##
            minn.append(0)
        else:
            minn.append(data[i].value_counts().min())
    length = len(data)
    nunik = data.nunique()
    nulls = data.isna().sum()
    df = pd.DataFrame(
        {
         #'column': list(data),
        'Nuniques': nunik,
         'NuniquePercent': (100*(nunik/length)),
         'dtype': data.dtypes,
         'Nulls' : nulls,
         'Nullpercent' : 100*(nulls/length),
         'Value counts Min':minn
        },
        columns = ['Nuniques', 'dtype','Nulls','Nullpercent', 'NuniquePercent',
                       'Value counts Min']).sort_values(by ='Nuniques',ascending = False)
    newcol = 'Data cleaning improvement suggestions'
    print('%s. Complete them before proceeding to ML modeling.' %newcol)
    mixed_cols = [col for col in data.columns if len(data[col].apply(type).value_counts()) > 1]
    df[newcol] = ''
    df['first_comma'] = ''
    if len(cat_cols) > 0:
        mask0 = df['dtype'] == 'object'
        mask1 = df['Value counts Min']/df['Nuniques'] <= 0.05
        mask4 = df['dtype'] == 'category'
        df.loc[mask0&mask1,newcol] += df.loc[mask0&mask1,'first_comma'] + 'combine rare categories'
        df.loc[mask4&mask1,newcol] += df.loc[mask4&mask1,'first_comma'] + 'combine rare categories'
        df.loc[mask0&mask1,'first_comma'] = ', '
        df.loc[mask4&mask1,'first_comma'] = ', '
    mask2 = df['Nulls'] > 0
    df.loc[mask2,newcol] += df.loc[mask2,'first_comma'] + 'fill missing'
    df.loc[mask2,'first_comma'] = ", "
    mask3 = df['Nuniques'] == 1
    df.loc[mask3,newcol] += df.loc[mask3,'first_comma'] + 'invariant values: drop'
    df.loc[mask3,'first_comma'] = ", "
    if len(non_num_cols) > 0:
        for x in non_num_cols:
            if df.loc[x, 'NuniquePercent'] == 100:
                df.loc[x, newcol] += df.loc[x,'first_comma'] + 'possible ID column: drop'
                df.loc[x,'first_comma'] = ", "
    mask5 = df['Nullpercent'] >= 90
    df.loc[mask5,newcol] += df.loc[mask5,'first_comma'] + 'very high nulls percent: drop'
    df.loc[mask5,'first_comma'] = ", "
    #### check for infinite values here #####
    inf_cols1 = np.array(num_cols)[[(data.loc[(data[col] == np.inf)]).shape[0]>0 for col in num_cols]].tolist()
    inf_cols2 = np.array(num_cols)[[(data.loc[(data[col] == -np.inf)]).shape[0]>0 for col in num_cols]].tolist()
    inf_cols = list(set(inf_cols1+inf_cols2))
    ### Check for infinite values in columns #####
    if len(inf_cols) > 0:
        for x in inf_cols:
            df.loc[x,newcol] += df.loc[x,'first_comma'] + 'infinite values: drop'
            df.loc[x,'first_comma'] = ", "
    #### Check for skewed float columns #######
    skew_cols1 = np.array(num_cols)[[(np.abs(np.round(data[col].skew(), 1)) > 1
                    ) & (np.abs(np.round(data[col].skew(), 1)) <= 5) for col in num_cols]].tolist()
    skew_cols2 = np.array(num_cols)[[(np.abs(np.round(data[col].skew(), 1)) > 5) for col in num_cols]].tolist()
    skew_cols = list(set(skew_cols1+skew_cols2))
    ### Check for skewed values in columns #####
    if len(skew_cols1) > 0:
        for x in skew_cols1:
            df.loc[x,newcol] += df.loc[x,'first_comma'] + 'skewed: cap or drop outliers'
            df.loc[x,'first_comma'] = ", "
    if len(skew_cols2) > 0:
        for x in skew_cols2:
            df.loc[x,newcol] += df.loc[x,'first_comma'] + 'highly skewed: drop outliers or do box-cox transform'
            df.loc[x,'first_comma'] = ", "
    ##### Do the same for mixed dtype columns - they must be fixed! ##
    if len(mixed_cols) > 0:
        for x in mixed_cols:
            df.loc[x,newcol] += df.loc[x,'first_comma'] + 'fix mixed data types'
            df.loc[x,'first_comma'] = ", "
    df.drop('first_comma', axis=1, inplace=True)
    return df
###################################################################################
def data_cleaning_suggestions(df):
    """
    This is a simple program to give data cleaning and improvement suggestions in class AV.
    Make sure you send in a dataframe. Otherwise, this will give an error.
    """
    if isinstance(df, pd.DataFrame):
        dfx = data_suggestions(df)
        all_rows = dfx.shape[0]
        ax = dfx.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
        display(ax);
    else:
        print("Input must be a dataframe. Please check input and try again.")
#############################################################################################

############################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '1.8'
print(f"""{module_type} LazyTransformer v{version_number}. 
""")
#################################################################################
