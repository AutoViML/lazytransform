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
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import copy
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import scipy
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import TruncatedSVD
##########################################################
from category_encoders import HashingEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders import OneHotEncoder, HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from category_encoders.quantile_encoder import QuantileEncoder
from category_encoders.quantile_encoder import SummaryEncoder
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
import imblearn
from sklearn.pipeline import make_pipeline, Pipeline
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
    inf_cols = EDA_find_columns_with_infinity(df)
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
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

def drop_second_col(Xt): 
    ### This deletes the 2nd column. Hence col number=1 and axis=1 ###
    return np.delete(Xt, 1, 1)

def change_col_to_string(Xt): 
    ### This converts the input column to a string and returns it ##
    return Xt.astype(str)

def create_column_names(Xt, nlpvars=[], catvars=[], numvars=[],datevars=[], onehot_dict={}, colsize_dict={},datesize_dict={}):
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

def create_column_names_onehot(Xt, nlpvars=[], catvars=[], discretevars=[], numvars=[],datevars=[], onehot_dict={},
                        colsize_dict={}, datesize_dict={}):
    ### This names all the features created by the NLP column. Hence col number=1 and axis=1 ###
    ### Once you get back names of one hot encoded columns, change the column names
    
    cols_cat = []
    for each_cat in catvars:
        categs = onehot_dict[each_cat]
        x_cols = [each_cat+'_'+str(categs[i]) for i in range(len(categs))]
        cols_cat += x_cols
    
    cols_discrete = []
    for each_discrete in discretevars:
        ### for anything other than one-hot we should just use label encoding to make it simpler ##
        try:
            categs = onehot_dict[each_discrete]
            discrete_add = [each_discrete+'_'+x for x in categs]
            cols_discrete += discrete_add
        except:
            ### if there is no new var to be created, just use the existing discrete vars itself ###
            cols_discrete.append(each_discrete)
    
    cols_nlp = []
    for each_nlp in nlpvars:
        colsize = colsize_dict[each_nlp]
        nlp_add = [each_nlp+'_'+str(x) for x in range(colsize)]
        cols_nlp += nlp_add
    ## do the same for datevars ###
    cols_date = []
    for each_date in datevars:
        colsize = datesize_dict[each_date]
        date_add = [each_date+'_'+str(x) for x in range(colsize)]
        cols_date += date_add
    #### this is where we put all the column names together #######
    cols_names = cols_cat+cols_discrete+cols_nlp+cols_date+numvars
    if nlpvars:
        ### Xt is a Sparse matrix array, we need to convert it  to dense array ##
        if scipy.sparse.issparse(Xt):
            return pd.DataFrame(Xt.toarray(), columns = cols_names)
        else:
            return pd.DataFrame(Xt, columns = cols_names)            
    else:
        ### Xt is already a dense array, no need to convert it ##
        return pd.DataFrame(Xt, columns = cols_names)

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
import pdb
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
def make_simple_pipeline(X_train, y_train, encoders='auto', scalers='', 
            date_to_string=False, verbose=0):
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
            basic_encoder = 'onehot'
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
    modeltype, multi_label = analyze_problem_type(y_train, target)
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
    if basic_encoder == 'onehot':
        be = OneHotEncoder()
        if verbose:
            print('Beware! one hot encoding can create hundreds if not 1000s of variables...')
    else:
        print('The first encoder in the list must always be onehot since only that is best for low cardinality variables.')
        be = OneHotEncoder()
    #### These are applied for high cardinality variables ########
    if encoder == 'onehot':
        le = OneHotEncoder()
        if verbose:
            print('Beware! one hot encoding can create hundreds if not 1000s of variables...')
    elif encoder == 'ordinal':
        le = OrdinalEncoder()
    elif encoder in ['hashing','hash']:
        le = HashingEncoder(n_components=20, drop_invariant=True)
    elif encoder == 'count':
        le = CountEncoder(drop_invariant=True)
    elif encoder == 'catboost':
        ### you must leave drop_invariant = False for catboost since it is not a onhot type encoder. ##
        le = CatBoostEncoder(drop_invariant=False)
    elif encoder == 'target':
        le = TargetEncoder(drop_invariant=True)
    elif encoder == 'glm' or encoder == 'glmm':
        # Generalized Linear Mixed Model 
        le = GLMMEncoder(drop_invariant=True)
    elif encoder == 'sum' or encoder == 'sumencoder':
        # Sum Encoder 
        le = SumEncoder(drop_invariant=True)
    elif encoder == 'woe':
        le = WOEEncoder(drop_invariant=True)
    elif encoder == 'bdc':
        le = BackwardDifferenceEncoder(drop_invariant=True)
    elif encoder == 'loo':
        le = LeaveOneOutEncoder(drop_invariant=True)
    elif encoder == 'base':
        le = BaseNEncoder()
    elif encoder == 'james' or encoder == 'jamesstein':
        le = JamesSteinEncoder(drop_invariant=True)
    elif encoder == 'helmert':
        le = HelmertEncoder(drop_invariant=True)
    elif encoder == 'quantile':
        le = QuantileEncoder(drop_invariant=True, quantile=0.5, m=1.0)
    elif encoder == 'summary':
        le = SummaryEncoder(drop_invariant=True, quantiles=[0.25, 0.5, 1.0], m=1.0)        
    elif encoder == 'label':
        ### My_LabelEncoder can only work on string and category object types with NaN.
        ### My_LabelEncoder_Pipe() can work with both category and object variables 
        le = My_LabelEncoder()
        ### How do we make sure that we create one new LE_Pipe for each catvar? Here's one way to do it.
        lep = My_LabelEncoder_Pipe()
    else:
        ### The default is Label Encoder
        encoder = 'label'
        le = My_LabelEncoder()
        ### How do we make sure that we create one new LE_Pipe for each catvar? Here's one way to do it.
        lep = My_LabelEncoder_Pipe()
    #### This is where we convert all the encoders to pipeline components ####
    if verbose:
        print('Using %s and %s as encoders' %(be,le))
    imp_missing = SimpleImputer(strategy='constant', fill_value='missing')
    imp = SimpleImputer(strategy='constant',fill_value=-99)
    convert_ce_to_pipe_func = FunctionTransformer(convert_ce_to_pipe)
    #### lep_one is the basic encoder of cat variables ############
    if basic_encoder == 'label':
        ######  Create a function called drop_second_col that drops the second unnecessary column in My_Label_Encoder
        drop_second_col_func = FunctionTransformer(drop_second_col)
        #### Now combine it with the LabelEncoder to make it run smoothly in a Pipe ##
        lep_one = Pipeline([('basic_encoder', lep), ('drop_second_column', drop_second_col_func)])
        #lep_one = make_pipeline(lep, drop_second_col_func)
        ### lep_one uses My_LabelEncoder to first label encode and then drop the second unused column ##
    else:
        lep_one = Pipeline([('convert_CE_to_pipe', convert_ce_to_pipe_func), ('basic_encoder', be)])
        #lep_one = make_pipeline(convert_ce_to_pipe_func, be)
    #### lep_two acts as the major encoder of discrete string variables ############
    if encoder == 'label':
        ######  Create a function called drop_second_col that drops the second unnecessary column in My_Label_Encoder
        drop_second_col_func = FunctionTransformer(drop_second_col)
        #### Now combine it with the LabelEncoder to make it run smoothly in a Pipe ##
        lep_two = Pipeline([('basic_encoder', lep), ('drop_second_column', drop_second_col_func)])
        #lep_two = make_pipeline(lep, drop_second_col_func)
        ### lep_one uses My_LabelEncoder to first label encode and then drop the second unused column ##
    else:
        #lep_two = make_pipeline(convert_ce_to_pipe_func, le)
        lep_two = Pipeline([('convert_CE_to_pipe', convert_ce_to_pipe_func), ('basic_encoder', le)])
    ####################################################################################
    # CREATE one_dim TRANSFORMER in order to fit between imputer and TFiDF for NLP here ###
    ####################################################################################
    one_dim = Make2D(imp_missing)
    if X_train.shape[0] >= 100000:
        #tiffd = CountVectorizer(strip_accents='unicode',max_features=1000)
        tiffd = TfidfVectorizer(strip_accents='unicode',max_features=3000)
        top_n = 100 ## number of components in SVD
        #tiffd = MyTiff(strip_accents='unicode',max_features=300, min_df=0.01)
    else:
        #vect = CountVectorizer(strip_accents='unicode',max_features=100)
        tiffd = TfidfVectorizer(strip_accents='unicode',max_features=1000)
        top_n = 10 ## number of components in SVD
        #tiffd = MyTiff(strip_accents='unicode',max_features=300, min_df=0.01)
    ### create a new pipeline with filling with constant missing ##
    tsvd = TruncatedSVD(n_components=top_n, n_iter=10, random_state=3)
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
    ######     C A T E G O R I C A L    E N C O D E R S    H E R E ####################
    ###### we need to create column names for one hot variables ###
    ####################################################################################
    ### these encoders result in more columns than the original - hence they are considered one hot type ###
    onehot_type_encoders = ['helmert','bdc','hashing','hash','sum','loo','base','woe','james',
                        'target','count','glm','glmm','summary']
    copy_cat_vars = copy.deepcopy(catvars)
    onehot_dict = {}
    
    if basic_encoder == 'onehot':
        for each_catcol in copy_cat_vars:
            onehot_dict[each_catcol] = X_train[each_catcol].unique().tolist()
    elif basic_encoder in onehot_type_encoders:
        for each_catcol in copy_cat_vars:
            copy_lep_one = copy.deepcopy(lep_one)
            onehot_dict[each_catcol] = copy_lep_one.fit_transform(X_train[each_catcol], y_train).columns.tolist()
    ### we now need to do the same for discrete variables based on encoder that is selected ##
    
    copy_discrete_vars = copy.deepcopy(discretevars)
    if encoder == 'onehot':
        for each_discrete in copy_discrete_vars:
            onehot_dict[each_discrete] = X_train[each_discrete].unique().tolist()    
    elif encoder in onehot_type_encoders:
        if encoder in ['hashing','hash']:
            print('Beware! hashing encoders can be really slow for even small datasets. Be patient...')
        for each_discrete in copy_discrete_vars:
            copy_lep_two = copy.deepcopy(lep_two)
            onehot_dict[each_discrete] = copy_lep_two.fit_transform(X_train[each_discrete], y_train).columns.tolist()
    ### if you drop remainder, then leftovervars is not needed.
    remainder = 'drop'
    
    ### If you passthrough remainder, then leftovers must be included 
    #remainder = 'passthrough'

    ### If you choose StandardScaler or MinMaxScaler, the integer values become stretched 
    ###  as if they are far apart when in reality they are close. So avoid it for now.
    if scalers=='max' or scalers == 'minmax':
        scaler = MinMaxScaler()
    elif scalers=='standard' or scalers=='std':
        scaler = StandardScaler()
    elif scalers=='maxabs':
        ### If you choose MaxAbsScaler, then NaNs which were Label Encoded as -1 are preserved as - (negatives). This is fantastic.
        scaler = MaxAbsScaler()
    else:
        ## there is no scaler ###
        scalers = ''

    ### All the imputers work on groups of variables => so they need to be in the end since they return only 1D arrays
    #### My_LabelEncoder and other fit_transformers need 2D arrays since they work on Target labels too.
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
    #if encoder in ['onehot','helmert','bdc','hashing','sum','loo','base','woe','james','label']:
    ### Use the one-hot anyway since most of the times, you will be doing one-hot encoding ###
    nlp_pipe = Pipeline([('NLP', FunctionTransformer(create_column_names_onehot, kw_args=params))])
    #else:
    #    nlp_pipe = Pipeline([('NLP', FunctionTransformer(create_column_names, kw_args=params))])
    #### Chain it together in the above pipeline #########
    data_pipe = Pipeline([('scaler_pipeline', scaler_pipe), ('nlp_pipeline', nlp_pipe)])
    #data_pipe = make_pipeline(scaler_pipe, nlp_pipe)
    #####    S A V E   P I P E L I N E  ########
    ### save the model and or pipeline here ####
    ############################################
    # save the model to disk
    filename = 'LazyTransformer_pipeline.pkl'
    difftime = max(1, int(time.time()-start_time))
    print('    Time taken to define data pipeline = %s second(s)' %difftime)
    print('    Data Pipeline is saved as: %s in current working directory.' %filename)
    pickle.dump(data_pipe, open(filename, 'wb'))
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
import pdb
class LazyTransformer():
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
    X : pandas DataFrame 
    y : pandas Series or DataFrame
    """
    def __init__(self, model=None, encoders='auto', scalers=None, date_to_string=False, 
                    transform_target=False, imbalanced=False, verbose=0):
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
        self.xformer = None
        self.yformer = None
        self.fitted = False
        self.imbalanced_first_done = False
        self.features = []
        self.imbalanced_flag = imbalanced
        self.smotex = None
        self.verbose = verbose
        if model is not None:
            self.modelformer = model
        else:
            self.modelformer = None
        self.transform_target = transform_target

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
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        self.features = X.columns.tolist()
        if self.transform_target:
            ### Non-scikit-learn models need numeric targets. 
            ### Hence YTransformer converts them before feeding model
            yt = YTransformer()
        #### This is where we build pipelines for X and y #############
        if self.modelformer is not None:
            ### If a model is given, then add it to pipeline and fit it ###
            data_pipe = make_simple_pipeline(X, y, encoders=self.encoders, scalers=self.scalers,
                date_to_string=self.date_to_string, verbose=self.verbose)
            ### There is no YTransformer in this pipeline so targets must be single label only ##
            model_name = str(self.modelformer).split("(")[0]            
            if y.ndim >= 2:
                ### In some cases, if y is a DataFrame with one column also, you get these situations.
                if y.shape[1] == 1:
                    ## In this case, y has only one column hence, you can use a model pipeline ##
                    ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.modelformer)])
                elif model_name not in ['MultiOutputClassifier','MultiOutputRegressor']:
                    ### In this case, y has more than 1 column, hence if it is not a multioutput model, give error
                    print('Erroring: please ensure you input a scikit-learn MultiOutput Regressor or Classifier')
                    return self
                else:
                    ## In this case we have a multi output model. So let's use it ###
                    #ml_pipe = make_pipeline(data_pipe, self.modelformer)
                    ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.modelformer)])
            else:
                ### You don't need YTransformer since it is a simple sklearn model
                #ml_pipe = make_pipeline(data_pipe, self.modelformer)
                ml_pipe = Pipeline([('data_pipeline', data_pipe), ('model', self.modelformer)])
            ##   Now we fit the model pipeline to X and y ###
            try:
                self.xformer = data_pipe.fit(X,y)
                if self.transform_target:
                    self.yformer = yt.fit(X,y)
                self.modelformer = ml_pipe.fit(X,y)
            except:
                print('Erroring: please check your input. There may be something wrong with data types or inputs.')
                return self
            print('model pipeline fitted with %s model' %model_name)
            self.fitted = True
        else:
            ### if there is no given model, just use the data_pipeline ##
            data_pipe = make_simple_pipeline(X, y, encoders=self.encoders, scalers=self.scalers,
                date_to_string=self.date_to_string, verbose=self.verbose)
            print('No model input given...')
            #### here we check if we should add a model to the pipeline 
            print('X and y Transformer Pipeline created...')
            self.fitted = True
            if self.transform_target:
                self.yformer = yt.fit(X,y)
            self.xformer = data_pipe.fit(X,y)
            ## we will leave self.modelformer as None ##
        ### print imbalanced ###
        if self.imbalanced_flag:
            #### This is where we do SMOTE #######################
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
        return self

    def predict(self, X, y=None):
        if self.fitted and self.modelformer is not None:
            y_enc = self.modelformer.predict(X)
            return y_enc
        else:
            print('Model not fitted or model not provided. Please check your inputs and try again')
            return y

    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        X_index = X.index
        start_time = time.time()
        
        if y is None and self.fitted:
            X_enc = self.xformer.transform(X)
            X_enc.index = X_index
            ### since xformer only transforms X ###
            difftime = max(1, int(time.time()-start_time))
            print('    Time taken to transform dataset = %s second(s)' %difftime)
            return X_enc
        elif self.fitted and self.modelformer is not None:
            print('Error: No transform allowed. You must use fit and predict when using a pipeline with a model.')
            return X, y
        elif not self.fitted:
            print('LazyTransformer has not been fit yet. Fit it first and try again.')
            return X, y
        elif y is not None and self.fitted and self.modelformer is None:
            if self.transform_target:
                _, y_enc = self.yformer.transform(X, y)
            else:
                y_enc = y
            X_enc = self.xformer.transform(X)
            X_enc.index = X_index
            #### Now check if the imbalanced_flag is True, then apply SMOTE using borderline2 algorithm which works better
            if self.imbalanced_first_done and self.imbalanced_flag:
                pass
            elif not self.imbalanced_first_done and self.imbalanced_flag:
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
        X_index = X.index
        start_time = time.time()
        self.fit(X,y)
        xt = self.xformer
        X_trans =  xt.transform(X)
        X_trans.index = X_index
        if self.transform_target:
            yt = self.yformer
            _, y_trans = yt.transform(X,y)
        else:
            y_trans = y
        if self.imbalanced_first_done and self.imbalanced_flag:
            #sm = SMOTE(sampling_strategy='auto')
            pass
        elif not self.imbalanced_first_done and self.imbalanced_flag:
            sm = self.smotex
            if verbose:
                print('Imbalanced flag set. Using SMOTE to transform X and y...')
            X_enc, y_enc = sm.fit_resample(X_enc, y_enc)
            self.imbalanced_first_done = True
            self.smotex = sm
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
        model_name = str(self.modelformer).split("(")[-2].split(",")[-1]
        if  model_name == ' LGBMClassifier' or model_name == ' LGBMRegressor':
            lgbm.plot_importance(self.modelformer.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == ' XGBClassifier' or model_name == ' XGBRegressor':
            plot_importance(self.modelformer.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == ' LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            import math
            feature_names = self.features
            model = self.modelformer.named_steps['model']
            w0 = model.intercept_[0]
            w = model.coef_[0]
            feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
            feature_importance["importance"] = pow(math.e, w)
            feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)[:max_features]
            feature_importance.plot.barh(x='feature', y='importance')
        else:
            try:
                importances = model.feature_importances_
                feature_names = self.features
                forest_importances = pd.Series(importances, index=feature_names)
                forest_importances.sort_values(ascending=False)[:max_features].plot(kind='barh')
            except:
                print('Could not plot feature importances. Please check your model and try again.')
####################################################################################
from sklearn.utils.validation import column_or_1d
#TransformerMixin, BaseEstimator
class YTransformer():
    def __init__(self):
        self.transformers =  {}
        self.targets = []
        
    def fit(self, X, y):
        """Fit the model according to the given training data"""
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  {}
        self.targets = []
        
        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if isinstance(y, pd.Series):
            self.targets.append(y.name)
        elif isinstance(y, pd.DataFrame):
            self.targets += y.columns.tolist()
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
    
    def transform(self, X, y=None):
        for i, each_target in enumerate(self.targets):
            if y is None:
                return X, y
            else:
                if isinstance(y, pd.Series):
                    y = pd.DataFrame(y)
                if i == 0:
                    y_t = self.transformers[each_target].transform(y.iloc[:,i])
                    y_trans = pd.Series(y_t,name=each_target)
                else:
                    y_t = self.transformers[each_target].transform(y.iloc[:,i])
                    y_trans = pd.DataFrame(y_trans)
                    y_trans[each_target] = y_t
                return X, y_trans
    
    def fit_transform(self, X, y=None):
        ### Since X for yT in a pipeline is sent as X, we need to switch X and y this way ##
        self.fit(y, X)
        for each_target in self.targets:
            y_trans =  self.transformers[each_target].transform(X)
        return y_trans
    
    def inverse_transform(self, X, y=None):
        for i, each_target in enumerate(self.targets):
            if i == 0:
                transformer_ = self.transformers[each_target]
                if isinstance(y, pd.Series):
                    y = pd.DataFrame(y, columns=self.targets)
                if isinstance(y, np.ndarray):
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
        return X, y_trans
    
    def predict(self, X, y=None, **fit_params):
        print('There is no predict function in Label Encoder. Returning...')
        return self
##############################################################################
def EDA_find_columns_with_infinity(df):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    """
    add_cols = []
    sum_cols = 0
    for col in df.columns:
        inf_sum1 = 0 
        inf_sum2 = 0
        inf_sum1 = len(df[df[col]==np.inf])
        inf_sum2 = len(df[df[col]==-np.inf])
        if (inf_sum1 > 0) or (inf_sum2 > 0):
            add_cols.append(col)
            sum_cols += inf_sum1
            sum_cols += inf_sum2
    if sum_cols > 0:
        print('    there are %d rows and %d columns with infinity in them...' %(sum_cols,len(add_cols)))
        print("    after removing columns with infinity, shape of dataset = (%d, %d)" %(df.shape[0],(df.shape[1]-len(add_cols))))
    return add_cols
####################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '0.28'
print(f"""{module_type} LazyTransformer version:{version_number}. Call by using:
    lazy = LazyTransformer(model=False, encoders='auto', scalers=None, 
        date_to_string=False, transform_target=False, imbalanced=False)
    ### if you are not using a model in pipeline, you must use fit and transform ##
    X_trainm, y_trainm = lazy.fit_transform(X_train, y_train)
    X_testm = lazy.transform(X_test)
    ### If using a model in pipeline, use fit and predict only ###
    lazy.fit(X_train, y_train)
    lazy.predict(X_test)
""")
#################################################################################
