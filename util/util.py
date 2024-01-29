import pandas as pd
import numpy as np

def lists_to_dict(key_list,value_list):
    dict_from_lists = dict(zip(key_list, value_list))
    
    return dict_from_lists

def type_value(df):
    cols=df.columns
    type_num_list=[]
    for i in range(df.shape[1]):
        type_value_counts=df[cols[i]].map(type).value_counts()
        key_list=list(type_value_counts.index)
        value_list=list(type_value_counts)
        type_num_dict=lists_to_dict(key_list,value_list)
        type_num_list.append(type_num_dict)
    type_num_dict_by_col=lists_to_dict(cols,type_num_list)
    
    return type_num_dict_by_col

def extract_str_data(df):
    cols=df.columns
    str_cols=[]

    for cols_i in range(len(cols)):
        str_bool=all([type(data_j) == type('str') for data_j in df[cols[cols_i]]])
        if str_bool:
            str_cols.append(cols[cols_i])
    
    str_df=df[str_cols]
    return str_df

def extract_int_float_data(df):
    cols=df.columns
    int_float_cols=[]

    for cols_i in range(len(cols)):
        int_float_bool=all([type(data_j) == type(0) or type(data_j) == type(1.0) for data_j in df[cols[cols_i]]])
        if int_float_bool:
            int_float_cols.append(cols[cols_i])
            
    int_float_df=df[int_float_cols]
    
    return int_float_df
    
def value_to_dummy(value,dic):
    keys_list=list(dic.keys())
    for keys_list_i in range(len(keys_list)):
        key=keys_list[keys_list_i]
        if key==value:
            return dic[key]
        elif keys_list_i==len(keys_list)-1:
            return value
            

    
def df_value_to_dummy(df,dic):
    cols=df.columns
    df_dummy=df.copy()
    for cols_i in cols:
        df_dummy[cols_i]=df_dummy[cols_i].apply(lambda x:value_to_dummy(x,dic))
    return df_dummy

#valueを基準にdictをソートする
def sort_dict_by_value(dic,reverse):
  sorted_list_from_dict=sorted(dic.items(),key=lambda value:value[1],reverse=reverse)
  sorted_dict=dict( (key,value) for key,value in sorted_list_from_dict)

  return sorted_dict

def extract_dict(dic,num):
    extract_dict = {key:dic[key] for key in list(dic)[:num]} 
    
    return extract_dict

def show_value_type(obj):
    if isinstance(obj,list):
        return [show_value_type(val) for val in obj]
    if isinstance(obj,dict):
        return {k:show_value_type(v) for k,v in obj.items()}
    if isinstance(obj,np.ndarray):
        return [show_value_type(val) for val in obj]
        
    return type(obj)
    

