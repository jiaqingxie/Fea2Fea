import numpy as np
import pandas as pd
import os

dataset_name = ['NCI1', 'PROTEINS', 'ENZYMES']
for dataset in dataset_name:
    path = os.listdir (r'/home/jiaqing/桌面/FASG_KDD/Result/TUdataset/' + dataset)
    tmp_df = pd.DataFrame()
    for files in path:
        cur_path = r'/home/jiaqing/桌面/FASG_KDD/Result/TUdataset/' + dataset +'/'+ files
        tmp = pd.read_csv(cur_path, sep = '\t',engine='python')
        tmp_df = tmp_df.append(tmp)
    
    tmp_df.to_csv(r'/home/jiaqing/桌面/FASG_KDD/Result/TUdataset/' + dataset + '_property.txt', sep='\t',index=False)