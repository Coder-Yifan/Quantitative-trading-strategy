import jqdatasdk as jq
jq.auth('18813173861','wuyifan1994')
ret = jq.get_price("000001.XSHE",start_date='2005-01-01',end_date='2015-01-01')
ret.to_csv("000001.csv")
ret = jq.attribute_history_engine()
ret = jq.get_fundamentals(valuation)