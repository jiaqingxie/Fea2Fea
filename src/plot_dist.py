import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

txt_dir = os.path.join('/home/jiaqing/桌面/Fea2Fea/', 'Result')

def read_file(dataset = 'planetoid', dataset_name = 'Cora', Property ='Degree', method = 'dist', count_zero = False):
    dir_name = []
    planetoid = ['PubMed','Cora', 'Citeseer'] # Must in this order
    tudataset = ['PROTEINS', 'ENZYMES', 'NCI1']
    if dataset == 'planetoid':
        for name in planetoid:
            dir_name.append(txt_dir + '/Planetoid/' + name + '_property.txt')
    elif dataset == 'tudataset':
        for name in tudataset:
            dir_name.append(txt_dir + '/TUdataset/' + name + '_property.txt')

    df = pd.DataFrame()

    if dataset == 'planetoid':
        for (element,name) in zip(dir_name,planetoid):
            temp = pd.read_csv(element, sep='\t')
        #print(temp.describe())
            if Property == 'Degree':   
                prop = temp[temp[Property]<16]
            if Property == 'Clustering_coefficient':
                prop = temp[temp[Property]< 0.001]
            if Property == 'Pagerank':
                prop = temp[temp[Property]< 0.0008]
            if Property == 'Aver_path_len':
                prop = temp[temp[Property]< 16]
            prop = prop[Property]
            df[name] = prop
        df.dropna()
        if count_zero and Property == 'Clustering_coefficient':
            total_zero = df[dataset_name].value_counts(0)
            print("\n")

    if dataset == 'tudataset':
        for (element,name) in zip(dir_name,tudataset):
            temp = pd.read_csv(element, sep='\t')
        #print(temp.describe())
            if Property == 'Degree':   
                prop = temp[temp[Property]<16]
            if Property == 'Clustering_coefficient':
                prop = temp[temp[Property]< 2]
            if Property == 'Pagerank':
                prop = temp[temp[Property]< 0.2]
            if Property == 'Aver_path_len':
                prop = temp[temp[Property]< 16]
            prop = prop[Property]
            df[name] = prop
        df.dropna()
        if count_zero and Property == 'Clustering_coefficient':
            total_zero = df[dataset_name].value_counts(0)
            print("total nodes with zero clustering coefficient in Cora dataset:{}".format(total_zero))
            print("\n")

    if method == 'dist': 
        plt.figure(figsize=(9,7))
        sns.set(font_scale = 1.5) 
        sns.distplot(df[dataset_name], axlabel = Property)
        plt.savefig(txt_dir + '/data distribution/tudataset/' + dataset_name +'_' + Property +'_dis.eps', dpi = 800, format = 'eps', bbox_inches='tight')
        
        return
    df = df.melt(var_name=Property, value_name='value')
    plt.figure(figsize=(10,10))
    sns.set(font_scale = 2) 
    sns.violinplot(x=Property, y='value', data=df)
    plt.savefig(txt_dir + '/violin_plot/' + Property +'_' + dataset +'_vp.eps', dpi = 800, format ='eps') # 

    plt.show()

   
if __name__ == "__main__":
    prop = ['Degree', 'Clustering_coefficient', 'Pagerank', 'Aver_path_len']
    planetoid = ['PubMed','Cora', 'Citeseer']
    tudataset = ['PROTEINS', 'ENZYMES', 'NCI1']
    for dataset_name in planetoid: 
        for i in prop:
            read_file(dataset = 'planetoid', dataset_name = dataset_name, Property=i, count_zero=True)

    
