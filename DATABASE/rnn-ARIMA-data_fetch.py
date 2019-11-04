import numpy as np
import pandas as pd
import json
import requests
from lxml import html
import time
df = []
def get_data(time):
    print("Fetching Data")
    url = "https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=USDT&e=BINANCE&aggregate=1&limit=1440&extraParams=PercepertronTrading&api_key=1a95b1abb57132503433dd9be659f9bc7c9c2f7a27f1cf78c0ceb89e59ee6e61"+"&toTs="+str(int(time))
    page = requests.get(url)
    tree = html.fromstring(page.content)
    print("Got It")
    return (json.loads(tree.text)["Data"])["Data"]
df = get_data(time.time())
for i in range(6):
    df = get_data(df[0]["time"]) + df
df = pd.DataFrame(df)
print(df)
df.to_csv("db-"+str(int(time.time()))+".csv")
