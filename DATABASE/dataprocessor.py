import pandas as pd
import glob
df_merged = (pd.read_csv("db-epoch.csv").iloc[:,1:])
print(glob.glob("d*.csv"))
for path in glob.glob("*.csv"):
    df_merged = df_merged.to_dict("records")
    print(path)
    df_new = ((pd.read_csv(path)).iloc[:,1:]).to_dict("records")
    print(pd.DataFrame(df_new))
    df_merged =  df_new + df_merged
    df_merged = ((pd.DataFrame(df_merged))).drop_duplicates()
    print(df_merged)
    df_merged = (df_merged.drop_duplicates(subset='time').reset_index(drop=True)).sort_values("time")
df_merged.to_csv("combined.csv")
