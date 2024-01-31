import numpy as np

#listやdictなどが持つそれぞれの要素に関数を適用する（多次元構造に対応）
def untangle_obj(func,obj):
    '''
    引数：
    func=関数
    obj=list or dict or ndaaray
    戻り値：
    関数の適応結果
    '''
    if isinstance(obj,list):
        return [untangle_obj(func,val) for val in obj]
    if isinstance(obj,dict):
        return {k:untangle_obj(func,v) for k,v in obj.items()}
    if isinstance(obj,np.ndarray):
        return [untangle_obj(func,val) for val in obj]
        
    return func(obj)