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
# 1. Takes numeric variables and imputes them using a simple imputer
# 1. Takes NLP and time series (string) variables and vectorizes them using CountVectorizer
# 1. Completely standardizing all of the above using AbsMaxScaler which preserves the 
#     relationship of label encoded vars
# 1. Finally adds an RFC or RFR to the pipeline so the entire pipeline can be fed to a 
#     cross validation scheme
# The results are yet to be seen but preliminary results very promising performance
############################################################################################
####### This pipeline is inspired by Kevin Markham's class on Scikit-Learn pipelines. ######
#######   You can sign up for his Data School class here: https://www.dataschool.io/  ######
############################################################################################
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import pdb
from collections import defaultdict
### These imports give fit_transform method for free ###
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import column_or_1d
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import TruncatedSVD

##############  These imports are to make trouble shooting easier #####
import copy
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
from category_encoders.quantile_encoder import QuantileEncoder
from category_encoders.quantile_encoder import SummaryEncoder
from category_encoders import OneHotEncoder
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
from sklearn.pipeline import make_pipeline, Pipeline
#from sklearn.preprocessing import OneHotEncoder
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
            fillval = self.transformer[np.nan]
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
    numvars = df.select_dtypes(include='number').columns.tolist()
    inf_cols = EDA_find_remove_columns_with_infinity(df)
    numvars = left_subtract(numvars, inf_cols)
    var_dict['continuous_vars'] = numvars
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
                        continuous_vars = {len(numvars)}, discrete_string_vars = {len(str_vars)}, nlp_vars = {len(nlp_vars)},
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

def create_column_names(Xt, nlpvars=[], catvars=[], discretevars=[], numvars=[], 
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
        colsize = datesize_dict[each_date]
        date_add = [each_date+'_'+str(x) for x in range(colsize)]
        cols_date += date_add
    cols_names = catvars+cols_nlp+cols_date+numvars
    if nlpvars:
        ### Xt is a Sparse matrix array, we need to convert it  to dense array ##
        if scipy.sparse.issparse(Xt):
            return pd.DataFrame(Xt.toarray(), columns = cols_names)
        else:
            return pd.DataFrame(Xt, columns = cols_names)            
    else:
        ### Xt is already a dense array, no need to convert it ##
        return pd.DataFrame(Xt, columns = cols_names)
#############################################################################################################
import random
import collections
random.seed(10)
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
def create_column_names_onehot(Xt, nlpvars=[], catvars=[], discretevars=[], numvars=[],datevars=[], onehot_dict={},
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
        colsize = datesize_dict[each_date]
        date_add = [each_date+'_'+str(x) for x in range(colsize)]
        cols_date += date_add
    #### this is where we put all the column names together #######
    cols_names = cols_cat+cols_discrete+cols_nlp+cols_date+numvars
    
    ### Xt is a Sparse matrix array, we need to convert it  to dense array ##
    if scipy.sparse.issparse(Xt):
        return pd.DataFrame(Xt.toarray(), columns = cols_names)
    else:
        ### Xt is already a dense array, no need to convert it ##
        return pd.DataFrame(Xt, columns = cols_names)
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
    if Xt.dtype != object:
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
def create_ts_features(df):
    """
    This takes in input a dataframe and a date variable.
    It then creates time series features using the pandas .dt.weekday kind of syntax.
    It also returns the data frame of added features with each variable as an integer variable.
    """
    dfx = pd.to_datetime(df.values)
    col_name = df.name
    dfn = pd.DataFrame(df.values, columns=[col_name])
    dt_adds = []
    
    try:
        dfn['_hour'] = dfx.hour.fillna(0).astype(int)
        dt_adds.append('_hour')
        dfn['_minute'] = dfx.minute.fillna(0).astype(int)
        dt_adds.append('_minute')
    except:
        print('    Error in creating hour-second derived features. Continuing...')
    try:
        dfn['_dayofweek'] = dfx.dayofweek.fillna(0).astype(int)
        dt_adds.append('_dayofweek')
        dfn.drop(col_name, axis=1, inplace=True)
        if '_hour' in dt_adds:
            dfn.loc[:,'_dayofweek_hour_cross'] = dfn['_dayofweek']+dfn['_hour']
            dt_adds.append('_dayofweek_hour_cross')
        dfn['_quarter'] = dfx.quarter.fillna(0).astype(int)
        dt_adds.append('_quarter')
        dfn['_month'] = dfx.month.fillna(0).astype(int)
        dt_adds.append('_month')
        #########################################################################
        if '_dayofweek' in dt_adds:
            dfn.loc[:,'_month_dayofweek_cross'] = dfn['_month'] + dfn['_dayofweek']
            dt_adds.append('_month_dayofweek_cross')
        dfn['_year'] = dfx.year.fillna(0).astype(int)
        dt_adds.append('_year')
        today = date.today()
        dfn['_age_in_years'] = today.year - dfx.year.fillna(0).astype(int)
        dt_adds.append('_age_in_years')
        dfn['_dayofyear'] = dfx.dayofyear.fillna(0).astype(int)
        dt_adds.append('_dayofyear')
        dfn['_dayofmonth'] = dfx.day.fillna(0).astype(int)
        dt_adds.append('_dayofmonth')
        dfn['_weekofyear'] = dfx.week.fillna(0).astype(int)
        dt_adds.append('_weekofyear')
        weekends = (dfn['_dayofweek'] == 6) | (dfn['_dayofweek'] == 5)
        dfn['_typeofday'] = 0
        dfn.loc[weekends, '_typeofday'] = 1
        dt_adds.append('_typeofday')
        if '_typeofday' in dt_adds:
            dfn.loc[:,'_month_typeofday_cross'] = dfn['_month'] + dfn['_typeofday']
            dt_adds.append('_month_typeofday_cross')
    except:
        print('    Error in creating date time derived features. Continuing...')
    #print('    created %d columns from time series column' %len(dt_adds))
    return dfn.values
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
            date_to_string=False, save_flag=False, combine_rare_flag=False, verbose=0):
    """
    ######################################################################################################################
    # # This is the SIMPLEST best pipeline for NLP and Time Series problems - Created by Ram Seshadri
    ###### This pipeline is inspired by Kevin Markham's class on Scikit-Learn pipelines. 
    ######      You can sign up for his Data School class here: https://www.dataschool.io/
    ######################################################################################################################
    #### What does this pipeline do. Here's the major steps:
    # 1. Takes categorical variables and encodes them using my special label encoder which can handle NaNs and future categories
    # 1. Takes numeric variables and imputes them using a simple imputer
    # 1. Takes NLP and time series (string) variables and vectorizes them using CountVectorizer
    # 1. Completely standardizing all of the above using AbsMaxScaler which preserves the relationship of label encoded vars
    # 1. Finally adds an RFC or RFR to the pipeline so the entire pipeline can be fed to a cross validation scheme
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
    numvars = var_dict['continuous_vars']
    nlpvars = var_dict['nlp_vars']
    datevars = var_dict['date_vars'] + var_dict['time_deltas'] + var_dict['date_zones']
    #### Converting date variable to a string variable if that is requested ###################
    if datevars:
        if date_to_string:
            copy_datevars = copy.deepcopy(datevars)
            for each_date in copy_datevars:
                print('    date_var %s will be treated as an NLP object and transformed by TfidfVectorizer' %each_date)
                X_train[each_date] = X_train[each_date].astype(str).values
                nlpvars.append(each_date)
                datevars.remove(each_date)
        else:
            print('    creating time series features from date_vars=%s as date_to_string is set to False' %datevars)
    else:
        print('    no date time variables detected in this dataset')
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
                    'quantile': QuantileEncoder(drop_invariant=True, quantile=0.5, m=1.0),
                    'summary': SummaryEncoder(drop_invariant=True, quantiles=[0.25, 0.5, 1.0], m=1.0),
                    'label': My_LabelEncoder(),
                    'auto': My_LabelEncoder(),
                    }

    ### set the basic encoder for low cardinality vars here ######
    be = encoder_dict[basic_encoder]
    #### These are applied for high cardinality variables ########
    le = encoder_dict[encoder]
    ### How do we make sure that we create one new LE_Pipe for each catvar? Here's one way to do it.
    lep = My_LabelEncoder_Pipe()
    ###### Just a warning in case someone doesn't know about one hot encoding ####
    ### these encoders result in more columns than the original - hence they are considered one hot type ###
    onehot_type_encoders = ['onehot', 'helmert','bdc', 'bde', 'hashing','hash','sum','base', 'quantile',
                                'summary']

    if basic_encoder in onehot_type_encoders or encoder in onehot_type_encoders:
        if verbose:
            print('    Beware! %s encoding can create hundreds if not 1000s of variables...' %basic_encoder)
        else:
            pass
    elif encoder in ['hashing','hash'] or basic_encoder in ['hashing', 'hash']:
        if verbose:
            print('    Beware! Hashing encoders can take a real long time for even small data sets!')
        else:
            pass

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
    ## number of components in SVD
    top_n = int(max(2, 5*np.log2(X_train.shape[0])))
    svd_n_iter = int(max(30, top_n*0.1))
    if len(nlpvars) > 0 and verbose:
        print('    %d components chosen for TruncatedSVD(n_iter=%d) after TFIDF' %(top_n, svd_n_iter))
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
    colsize_dict = {}
    
    for each_nlp in copy_nlp_vars:
        colsize = vect.fit_transform(X_train[each_nlp]).shape[1]
        colsize_dict[each_nlp] = colsize
    ##### we collect the column size of each nlp variable and feed it to vect_one ##
    #vect_one = make_pipeline(change_col_to_string_func, vect)
    vect_one = Pipeline([('change_col_to_string', change_col_to_string_func), ('tfidf_tsvd_pipeline', vect)])
    #### Now create a function that creates time series features #########
    create_ts_features_func = FunctionTransformer(create_ts_features)
    #### we need to the same for date-vars #########
    copy_date_vars = copy.deepcopy(datevars)
    datesize_dict = {}
    for each_datecol in copy_date_vars:
        datesize = create_ts_features_func.fit_transform(X_train[each_datecol]).shape[1]
        datesize_dict[each_datecol] = datesize
    ####################################################################################
    ######     C A T E G O R I C A L    E N C O D E R S    H E R E #####################
    ######     we need to create unique column names for one hot variables    ##########
    ####################################################################################
    copy_cat_vars = copy.deepcopy(catvars)
    onehot_dict = defaultdict(return_default)
    ##### This is extremely complicated logic -> be careful before modifying them!
    
    if basic_encoder in onehot_type_encoders:
        for each_catcol in copy_cat_vars:
            copy_lep_one = copy.deepcopy(lep_one)
            if combine_rare_flag:
                rcct = Rare_Class_Combiner_Pipe()
                unique_cols = make_column_names_unique(rcct.fit_transform(X_train[each_catcol]).unique().tolist())
                unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                onehot_dict[each_catcol] = unique_cols
            else:
                if basic_encoder == 'onehot':
                    unique_cols = X_train[each_catcol].unique().tolist()
                    #unique_cols = np.where(unique_cols==np.nan, 'missing', unique_cols)
                    #unique_cols = np.where(unique_cols == None, 'missing', unique_cols)
                    unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                    unique_cols = make_column_names_unique(unique_cols)
                    onehot_dict[each_catcol] = unique_cols
                else:
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
                unique_cols = make_column_names_unique(rcct.fit_transform(X_train[each_discrete]).unique().tolist())
                unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                onehot_dict[each_discrete] = unique_cols
            else:
                if encoder == 'onehot':
                    unique_cols = X_train[each_discrete].unique().tolist()
                    #unique_cols = np.where(unique_cols==np.nan, 'missing', unique_cols)
                    unique_cols = [str(x) for x in unique_cols] ### just make them all strings
                    unique_cols = make_column_names_unique(unique_cols)
                    onehot_dict[each_discrete] = unique_cols
                else:
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
        ## there is no scaler ###
        scalers = ''

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
    ##### Now we can combine the rest of the variables into the pipeline ################
    middle_str2 = "".join(['(vect_one, nlpvars['+str(i)+']),' for i in range(len(nlpvars))])
    middle_str3 = "".join(['(create_ts_features_func, datevars['+str(i)+']),' for i in range(len(datevars))])
    end_str = '(imp, numvars),    remainder=remainder)'
    ### We now put the whole transformer pipeline together ###
    full_str = init_str+middle_str0+middle_str1+middle_str2+middle_str3+end_str
    ct = eval(full_str)
    if verbose:
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
          "numvars": numvars,
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
    date_to_string : default is False. If True it means date columns will be
            converted to pandas datetime vars and meaningful features will be
            extracted such as dayoftheweek, etc. If False, datetime columns
            will be treated as string variables and used as cat vars or NLP vars.
    transform_target : default is False. If True , target column(s) will be 
            converted to numbers and treated as numeric. If False, target(s)
            will be left as they are and not converted.
    imbalanced : default is False. If True, we will try SMOTE if no model is input.
            If a model is input, then it will be wrapped in a special Classifier
            called SuloClassifier which is a high performance stacking model that
            will work on your imbalanced data set. If False, your data will be 
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
        self.model = model
        if not self.model:
            self.model = None
        self.transform_target = transform_target
        self.fitted = False
        self.save = save
        self.combine_rare = combine_rare

    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"date_to_string": self.date_to_string, "encoders": self.encoders,
            "scalers": self.scalers, "imbalanced": self.imbalanced, 
            "verbose": self.verbose, "transform_target": self.transform_target,
            "model": self.model, "combine_rare": self.combine_rare,}

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
        if self.model is not None:
            ### If a model is given, then add it to pipeline and fit it ###
            data_pipe = make_simple_pipeline(X, y, encoders=self.encoders, scalers=self.scalers,
                date_to_string=self.date_to_string, save_flag = self.save, 
                combine_rare_flag=self.combine_rare, verbose=self.verbose)
            
            ### There is no YTransformer in this pipeline so targets must be single label only ##
            model_name = str(self.model).split("(")[0]            
            if y.ndim >= 2:
                ### In some cases, if y is a DataFrame with one column also, you get these situations.
                if y.shape[1] == 1:
                    ## In this case, y has only one column hence, you can use a model pipeline ##
                    if model_name == '': 
                        print('No model name specified')
                        self.model = None
                        ml_pipe = Pipeline([('data_pipeline', data_pipe),])
                    else:
                        ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model)])
                elif model_name == '': 
                    print('No model name specified')
                    self.model = None
                    ml_pipe = Pipeline([('data_pipeline', data_pipe),])
                elif model_name not in ['MultiOutputClassifier','MultiOutputRegressor']:
                    ### In this case, y has more than 1 column, hence if it is not a multioutput model, give error
                    print('    Alert: Multi-Label problem - make sure your input model can do MultiOutput!')
                    ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model)])
                else:
                    ## In this case we have a multi output model. So let's use it ###
                    #ml_pipe = make_pipeline(data_pipe, self.model)
                    ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model)])
            else:
                ### You don't need YTransformer since it is a simple sklearn model
                #ml_pipe = make_pipeline(data_pipe, self.model)
                ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.model)])
            ##   Now we fit the model pipeline to X and y ###
            try:
                #### This is a very important set of statements ####
                self.xformer = data_pipe.fit(X,y)
                if self.transform_target:
                    if y is not None:
                        self.y_index = y.index
                    self.yformer.fit(y)
                    yt = self.yformer.transform(y)
                    print('    transformed target from object type to numeric')

                    if y is not None:
                        yt.index = self.y_index
                    ### Make sure you leave self.model as None when there is no model ### 
                    if model_name == '': 
                        self.model = None
                    else:
                        self.model = ml_pipe.fit(X,yt)
                else:
                    ### Make sure you leave self.model as None when there is no model ### 
                    if model_name == '': 
                        self.model = None
                    else:
                        self.model = ml_pipe.fit(X,y)
            except Exception as e:
                print('Erroring due to %s: There may be something wrong with your data types or inputs.' %e)
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
            print('X and y Transformer Pipeline created...')
            if self.transform_target:
                self.yformer.fit(y)
                yt = self.yformer.transform(y)
                print('    transformed target from object type to numeric')
                if y is not None:
                    yt.index = self.y_index
                self.xformer = data_pipe.fit(X,yt)
            else:
                self.xformer = data_pipe.fit(X,y)
            ## we will leave self.model as None ##
            self.fitted = True
        ### print imbalanced ###
        if self.imbalanced:
            print('### Alert! Do not use SMOTE if this is not an imbalanced classification problem #######')
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
        if self.fitted and self.model is not None:
            y_enc = self.model.predict(X)
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
            return X_enc
        elif self.fitted and self.model is not None:
            print('Error: No transform allowed. You must use fit and predict when using a pipeline with a model.')
            return X, y
        elif not self.fitted:
            print('LazyTransformer has not been fit yet. Fit it first and try again.')
            return X, y
        elif y is not None and self.fitted and self.model is None:
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
                sm = self.smotex
                X_enc, y_enc = sm.fit_resample(X_enc, y_enc)
                self.imbalanced_first_done = True
                self.smotex = sm
            print('    SMOTE transformed data in pipeline. Dont forget to use transformed X and y from output.')
            difftime = max(1, int(time.time()-start_time))
            print('    Time taken to transform dataset = %s second(s)' %difftime)
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
            print('### Alert! Do not use SMOTE if this is not an imbalanced classification problem #######')
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
        return X_trans, y_trans

    def fit_predict(self, X, y=None):
        transformer_ = self.fit(X,y)
        y_trans =  transformer_.predict(X)
        return X, y_trans

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
        if len(params) == 0 and not multi_label:
            rand_params = {
                'model__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5],
                'model__num_leaves': [5, 10,30,50],
                'model__n_estimators': [100,200,300],
                'model__class_weight':[None, 'balanced'],
                    }
            grid_params = {
                'model__n_estimators': [100, 150, 200, 250, 300],
                'model__class_weight':[None, 'balanced'],
                    }

        else:
            if grid_search:
                grid_params = copy.deepcopy(params)
            else:
                rand_params = copy.deepcopy(params)
                
        ########   Now let's perform randomized search to find best hyper parameters ######
        if modeltype == 'Regression':
            scoring = 'neg_mean_squared_error'
            score_name = 'MSE'
        else:
            if grid_search:
                scoring = 'balanced_accuracy'
                score_name = 'balanced_accuracy'
            else:
                scoring = 'recall'
                score_name = 'recall'

        #### This is where you grid search the pipeline now ##############
        if grid_search:
            search = GridSearchCV(
                    self, 
                    grid_params, 
                    cv=3,
                    scoring=scoring,
                    refit=True,
                    return_train_score = True,
                    n_jobs=-1,
                    verbose=True,
                    )
        else:
            search = RandomizedSearchCV(
                    self,
                    rand_params,
                    n_iter = 10,
                    cv = 3,
                    refit=True,
                    return_train_score = True,
                    random_state = 99,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=True,
                    )
        
        ##### This is where we search for hyper params for model #######
        search.fit(X_train, y_train)
        
        cv_results = pd.DataFrame(search.cv_results_)
        if modeltype == 'Regression':
            print('Mean cross-validated train %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_train_score'].mean()))))
            print('    Mean cross-validated test %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_test_score'].mean()))))
        else:
            print('Mean cross-validated train %s = %0.04f' %(score_name, cv_results['mean_train_score'].mean()))
            print('    Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
            
        print('Time taken for Hyper Param tuning of LGBM (in minutes) = %0.1f' %(
                                        (time.time()-start_time)/60))
        print('Best params from search:\n%s' %search.best_params_)

        newpipe = search.best_estimator_.model.set_params(**search.best_params_)
        print('    returning a new LazyTransformer pipeline that contains the best model trained on your train dataset!')
        return newpipe
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
        elif isinstance(y, pd.DataFrame):
            y_trans = copy.deepcopy(y)
            for each_target in self.targets:
                y_trans[each_target] = self.transformers[each_target].transform(y[each_target])
        else:
            print('Error: Cannot transform numpy arrays. Returning')
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

from sklearn.metrics import mean_squared_log_error, mean_squared_error,balanced_accuracy_score
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy as sp
import time
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter, defaultdict
from tqdm.notebook import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import os
def check_if_GPU_exists():
    try:
        os.environ['NVIDIA_VISIBLE_DEVICES']
        print('GPU active on this device')
        return True
    except:
        print('No GPU active on this device')
        return False

###############################################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '0.60'
print(f"""{module_type} LazyTransformer version:{version_number}. Call by using:
    lazy = LazyTransformer(model=None, encoders='auto', scalers=None, date_to_string=False,
        transform_target=False, imbalanced=False, save=False, combine_rare=False, verbose=0)
    ### if you are not using a model in pipeline, you must use fit and transform ###
        X_trainm, y_trainm = lazy.fit_transform(X_train, y_train)
        X_testm = lazy.transform(X_test)
    ### If using a model in pipeline, use fit and predict only ###
        lazy.fit(X_train, y_train)
        lazy.predict(X_test)
""")
#################################################################################
