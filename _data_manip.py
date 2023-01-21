import pandas as pd
from program import DIR_DATA
from RoughVolatility.scr.black_scholes import impVol

fname = "Raw_data"

cols = ["dtm","Obsdate","spot","K", "price", "IV_old", "IV"]

df = pd.read_csv(DIR_DATA.joinpath(f"{fname}.csv"))
print(df)
df["ttm"]=df["dtm"]/252
df[["spot","K","price"]]=df[["spot","K","price"]]/100

df["iv"]=df.apply(lambda x: impVol(0,x["price"],x["spot"],x["K"],x["ttm"]), axis=1)

df = df[["ttm","spot","Obsdate", "K","iv"]]
print(df)
df.to_csv(DIR_DATA.joinpath(f"VIX_calc.csv"),index=False)
