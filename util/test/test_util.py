import pandas as pd
import numpy as np
import pytest
from ..util import *

def test_lists_to_dict():
    input_key_list=['test_key1','test_key2']
    input_value_list=[1,2]
    
    input_key_list_2=['test_key1','test_key2','test_key3']
    input_value_list_2=[1,2,3]
    
    assert lists_to_dict(input_key_list,input_value_list)=={'test_key1':1,'test_key2':2}
    assert lists_to_dict(input_key_list,input_value_list)=={input_key_list[0]:input_value_list[0],input_key_list[1]:input_value_list[1]}
    assert lists_to_dict(input_key_list_2,input_value_list_2)=={input_key_list_2[0]:input_value_list_2[0],input_key_list_2[1]:input_value_list_2[1],input_key_list_2[2]:input_value_list_2[2]}
    

def test_type_value():
    input_column_list=['test']
    input_data_list=['data',2,'']
    sample_df=pd.DataFrame(data=input_data_list,columns=input_column_list)
    
    assert type_value(sample_df)=={'test':{str:2,int:1}}
    
    input_column_list_2=['test0','test1','test2']
    input_data_list_2=[['data',2.2,'a'],
                   [2,2.3,'b'],
                   ['',1.5,'c']]
    sample_df_2=pd.DataFrame(data=input_data_list_2,columns=input_column_list_2)
    
    assert type_value(sample_df_2)=={'test0':{str: 2, int: 1}, 'test1':{float: 3}, 'test2':{str: 3}}

def test_extract_str_data():
    input_columns_list=['str0','str1','int0']
    input_data_list=[['data0','data0',0],
                   ['data1','data1',1],
                   ['data2','data2',0]]
    input_expect_columns_list=['str0','str1']
    input_expect_data_list=[['data0','data0'],
                   ['data1','data1'],
                   ['data2','data2']]
    sample_df=pd.DataFrame(data=input_data_list,columns=input_columns_list)
    expect_df=pd.DataFrame(data=input_expect_data_list,columns=input_expect_columns_list)
    diff_df=pd.concat([extract_str_data(sample_df),expect_df]).drop_duplicates(keep=False)
    #extract_str_data(sample_df)とexpect_dfが一致していれば、dfiif_dfの行数は0,列数はexpect_dfの列数(=2)になるはず
    assert diff_df.shape==(0, 2)
    
    input_data_list_2=[['data0','data0',0],
                   ['data1','data1','data1'],
                   ['data2','data2','data2']]
    sample_df_2=pd.DataFrame(data=input_data_list_2,columns=input_columns_list)
    diff_df=pd.concat([extract_str_data(sample_df_2),expect_df]).drop_duplicates(keep=False)
    assert diff_df.shape==(0, 2)

def test_extract_int_float_data():
    input_columns_list=['str0','float0','int0','float1','int1']
    input_data_list=[['data0',0.5,0,-1.0,-1],
                   ['data1',1.5,1,-2.0,-2],
                   ['data2',2.5,2,-3.0,-3]]
    input_expect_columns_list=['float0','int0','float1','int1']
    input_expect_data_list=[[0.5,0,-1.0,-1],
                   [1.5,1,-2.0,-2],
                   [2.5,2,-3.0,-3]]
    sample_df=pd.DataFrame(data=input_data_list,columns=input_columns_list)
    expect_df=pd.DataFrame(data=input_expect_data_list,columns=input_expect_columns_list)
    diff_df=pd.concat([extract_int_float_data(sample_df),expect_df]).drop_duplicates(keep=False)
    #extract_str_data(sample_df)とexpect_dfが一致していれば、dfiif_dfの行数は0,列数はexpect_dfの列数(=4)になるはず
    assert diff_df.shape==(0, 4)
    
    input_data_list_2=[[0,0.5,0,-1.0,-1],
                   ['data1',1.5,1,-2.0,-2],
                   ['data2',2.5,2,-3.0,-3]]
    sample_df_2=pd.DataFrame(data=input_data_list_2,columns=input_columns_list)
    diff_df=pd.concat([extract_int_float_data(sample_df_2),expect_df]).drop_duplicates(keep=False)
    assert diff_df.shape==(0, 4)

def test_value_to_dummy():
    dic={'?':'NaN','NaN':'--','a':1}
    assert value_to_dummy('?',dic)=='NaN'
    assert value_to_dummy('NaN',dic)=='--'
    assert value_to_dummy('a',dic)==1
    assert value_to_dummy(0,dic)==0


def test_df_value_to_dummy():
    input_columns_list=['test1-0']
    input_data_list=[[1],
                     [2],
                     ['?'],
                     [4]]
    input_expect_columns_list=['test1-0']
    input_expect_data_list=[[1],
                     [2],
                     ['NA'],
                     [4]]
    sample_df=pd.DataFrame(data=input_data_list,columns=input_columns_list)
    expect_df=pd.DataFrame(data=input_expect_data_list,columns=input_expect_columns_list)
    dic={'?':'NA'}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)
    
    input_columns_list_2=['test2-0']
    input_data_list_2=[[1],
                     [2],
                     ['?'],
                     ['?']]
    input_expect_columns_list_2=['test2-0']
    input_expect_data_list_2=[[1],
                     [2],
                     ['NA'],
                     ['NA']]
    sample_df=pd.DataFrame(data=input_data_list_2,columns=input_columns_list_2)
    expect_df=pd.DataFrame(data=input_expect_data_list_2,columns=input_expect_columns_list_2)
    dic={'?':'NA'}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)
    
    input_columns_list_3=['test3-0']
    input_data_list_3=[[1],
                     [2],
                     ['?'],
                     ['---']]
    input_expect_columns_list_3=['test3-0']
    input_expect_data_list_3=[[1],
                     [2],
                     ['NA'],
                     ['NA']]
    sample_df=pd.DataFrame(data=input_data_list_3,columns=input_columns_list_3)
    expect_df=pd.DataFrame(data=input_expect_data_list_3,columns=input_expect_columns_list_3)
    dic={'?':'NA','---':'NA'}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)

    input_columns_list_4=['test4-0']
    input_data_list_4=[[1],
                     [2],
                     ['?'],
                     ['---']]
    input_expect_columns_list_4=['test4-0']
    input_expect_data_list_4=[[1],
                     [2],
                     ['NA'],
                     ['']]
    sample_df=pd.DataFrame(data=input_data_list_4,columns=input_columns_list_4)
    expect_df=pd.DataFrame(data=input_expect_data_list_4,columns=input_expect_columns_list_4)
    dic={'?':'NA','---':''}
    diff_df=pd.concat([df_value_to_dummy(sample_df,dic),expect_df]).drop_duplicates(keep=False) 
    assert diff_df.shape==(0, 1)
    
def test_extract_dict():
    sample_dict={'test1':1,'test2':2,'test3':3}
    expect_dict={'test1':1,'test2':2}
    assert  extract_dict(sample_dict,2)==expect_dict

class Test_untangle_obj:
    
    def test_untangle_obj_int_type(self,return_int):
        assert untangle_obj(type,return_int)==int
   
    @pytest.fixture
    def return_int(self):
        return 1

    def test_untangle_obj_list_type(self,return_list):
        int_list,mixed_list,two_D_list,three_D_list,dict_list,array_list=return_list
        assert untangle_obj(type,int_list)==[int,int,int]
        assert untangle_obj(type,mixed_list)==[int,float,str]
        assert untangle_obj(type,two_D_list)==[[int,int,int],[int,float,str]]
        assert untangle_obj(type,three_D_list)==[[int,int,int],[[int,int,int],[int,float,str]]]
        assert untangle_obj(type,dict_list)==[[int,int,int],{'t1':int,'t2':int,'t3':int}]
        assert untangle_obj(type,array_list)==[[int,int,int],[np.int64,np.int64,np.int64]]
        
    @pytest.fixture
    def return_list(self):
        int_list=[0,1,2]
        mixed_list=[0,1.3,'test1']
        two_D_list=[int_list,mixed_list]
        three_D_list=[int_list,two_D_list]
        dict_list=[int_list,{'t1':0,'t2':1,'t3':2}]
        array_list=[int_list,np.array([0,1,2])]
        
        return int_list,mixed_list,two_D_list,three_D_list,dict_list,array_list

    def test_untangle_obj_dict_type(self,return_dict):
        int_dict,mixed_dict,list_dict,two_D_dict,three_D_dict,array_dict=return_dict
        assert untangle_obj(type,int_dict)=={'t1':int,'t2':int,'t3':int}
        assert untangle_obj(type,mixed_dict)=={'t1':int,'t2':float,'t3':str}
        assert untangle_obj(type,list_dict)=={'t1':[int,int],'t2':[int,int],'t3':[int,int,int]}
        assert untangle_obj(type,two_D_dict)=={'t1':{'t1':int,'t2':float,'t3':str},'t2':{'t1':[int,int],'t2':[int,int],\
                                                't3':[int,int,int]}}
        assert untangle_obj(type,three_D_dict)=={'t1':{'t1':int,'t2':int,'t3':int},'t2':{'t1':{'t1':int,'t2':float,'t3':str},\
                                                't2':{'t1':[int,int],'t2':[int,int],'t3':[int,int,int]}}}
        assert untangle_obj(type,array_dict)=={'t1':{'t1':int,'t2':int,'t3':int},'t2':[np.int64,np.int64,np.int64]}

    @pytest.fixture
    def return_dict(self):
        int_dict={'t1':0,'t2':1,'t3':2}
        mixed_dict={'t1':0,'t2':1.3,'t3':'test'}
        list_dict={'t1':[0,1],'t2':[1,2],'t3':[2,3,4]}
        two_D_dict={'t1':mixed_dict,'t2':list_dict}
        three_D_dict={'t1':int_dict,'t2':two_D_dict}
        array_dict={'t1':int_dict,'t2':np.array([0,1,2])}
        
        return int_dict,mixed_dict,list_dict,two_D_dict,three_D_dict,array_dict

  