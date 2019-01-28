import pandas as pd
import jqdatasdk as jq

jq.auth('18813173861', 'wuyifan1994')
start_date = '2005-02-01'
end_date = '2018-01-01'
def get_data():
    data = jq.get_price('000001.XSHG', start_date=start_date, end_date=end_date,
                     fields=['open', 'close', 'low', 'high', 'volume', 'money', 'high_limit', 'low_limit', 'avg'])
    data['label'] = data['close'].shift(-1)
    data.to_csv("data_set.csv")
get_data()
