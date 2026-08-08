import os, json, numpy as np, pandas as pd
from pathlib import Path
os.environ.setdefault("PYTHONPYCACHEPREFIX","/tmp/pyc")
BASE=str(Path(__file__).resolve().parents[1]).replace("\\", "/")
FEATS=["Type_ord","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
TARGETS=["Machine failure","TWF","HDF","PWF","OSF","RNF"]
def load_ai4i():
    df=pd.read_csv(BASE+"/data/ai4i2020.csv")
    df["Type_ord"]=df["Type"].map({"L":0,"M":1,"H":2})
    return df
def ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float)
    e=0.0
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        m=(p>=lo)&(p<hi) if i<bins-1 else (p>=lo)&(p<=hi)
        if m.sum()==0: continue
        e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
def jdump(obj,path):
    tmp=path+".tmp"
    with open(tmp,"w") as f: json.dump(obj,f,indent=1,default=float)
    os.replace(tmp,path)

