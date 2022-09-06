from urllib import request 
import pandas as pd

def model_brand_list():
    data_url ="https://raw.githubusercontent.com/aravind9722/datasets-for-ML-projects/main/cardekho_dataset.csv"
    df =pd.read_csv(data_url)
    brand_list = list(df.brand.unique())
    model_list = list(df.model.unique())
    return brand_list, model_list

