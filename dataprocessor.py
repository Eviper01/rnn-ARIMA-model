import pandas as pd
df_old = (pd.read_csv("db.csv")).to_dict("records")
df_new = (pd.read_csv("db-1572670527.csv")).to_dict("records")
df_merged = df_old + df_new
df_merged = ((pd.DataFrame(df_merged)).iloc[:,1:]).drop_duplicates()
df_merged.to_csv("combined.csv", index=False)
