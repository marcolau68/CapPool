import yfinance as yf
import numpy as np
import pandas as pd
from util import constants
import json


# all_sectors = {}
# all_industries = {}
# mapping = {}

# all_sectors_index = 0
# all_industries_index = 0

# for t in constants.TICKERS:
#     try:
#         tmp = yf.Ticker(t).info
#         sector, industry = tmp["sector"], tmp["industry"]
#         print(sector, industry)

#         if sector is not None and sector not in all_sectors:
#             all_sectors[sector] = all_sectors_index
#             all_sectors_index += 1
        
#         if industry is not None and industry not in all_industries:
#             all_industries[industry] = all_industries_index
#             all_industries_index += 1
        
#         sector_index = all_sectors[sector] if sector is not None else -1
#         industry_index = all_industries[industry] if industry is not None else -1
#         mapping[t] = (sector_index, industry_index)
#     except:
#         mapping[t] = (-1, -1)

# save_data = {"all_sectors": all_sectors, "all_industries": all_industries, "mapping": mapping}

# with open("graph_data/sectors_industries.json", "w") as outfile: 
#     json.dump(save_data, outfile)

# NUM_STOCKS = 501
# f = open("graph_data/sectors_industries.json")
# all = json.load(f)

# all_sectors = all["all_sectors"]
# all_industries = all["all_industries"]
# mapping = all["mapping"]

# num_sectors = len(all_sectors)
# num_industries = len(all_industries)

# industry_to_sector = {}

# s_1 = np.zeros((NUM_STOCKS, num_industries))
# s_2 = np.zeros((num_industries, num_sectors))

# df_1 = pd.DataFrame(data=s_1.T, columns=constants.TICKERS)

# for t in mapping:
#     industry = mapping[t][1]

#     if industry < 0:
#         continue

#     df_1.iloc[industry][t] = 1
    
#     if industry not in industry_to_sector:
#         industry_to_sector[industry] = mapping[t][0]

# df_2 = pd.DataFrame(data=s_2.T, columns=list(all_industries.keys()))

# for ind in list(all_industries.keys()):
#     industry = all_industries[ind]

#     if industry not in industry_to_sector:
#         continue

#     sector = industry_to_sector[industry]
#     df_2.iloc[sector][industry] = 1

# s_1 = df_1.to_numpy().T
# s_2 = df_2.to_numpy().T

# print(s_1.shape, s_2.shape)

# np.savetxt("graph_data/stock_pool_s1.csv", s_1,  delimiter = ",")
# np.savetxt("graph_data/stock_pool_s2.csv", s_2,  delimiter = ",")

df = pd.read_csv("graph_data/stock_pool_s1.csv", sep=",", header=None)
s_1 = df.values

print(s_1.shape)
