import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

txt_dir = os.path.join(os.getcwd(), 'Result')





def read_file(dataset = 'planetoid', Property ='Degree', plot_distribution = False, count_zero = False):
    dir_name = []
    planetoid = ['PubMed','Cora', 'Citeseer'] # Must in this order
    if dataset == 'planetoid':
        for name in planetoid:
            dir_name.append(txt_dir + '/Planetoid/' + name + '_property.txt')

    df = pd.DataFrame()

    for (element,name) in zip(dir_name,planetoid):
        temp = pd.read_csv(element, sep='\t')
        print(temp.describe())
        if Property == 'Degree':   
            prop = temp[temp[Property]<10]
        if Property == 'Clustering_coefficient':
            prop = temp[temp[Property]< 0.2]
        if Property == 'Pagerank':
            prop = temp[temp[Property]< 0.0006]
        if Property == 'Aver_path_len':
            prop = temp[temp[Property]< 10]
        prop = prop[Property]
        df[name] = prop
    df.dropna()

    if count_zero and Property == 'Clustering_coefficient':
        total_zero = df['Cora'].value_counts(0)
        print("total nodes with zero clustering coefficient in Cora dataset:{}".format(total_zero))
        total_zero = df['PubMed'].value_counts(0)
        print("total nodes with zero clustering coefficient in PubMed dataset:{}".format(total_zero))
        total_zero = df['Citeseer'].value_counts(0)
        print("total nodes with zero clustering coefficient in Citeseer dataset:{}".format(total_zero))
        print("\n")
        

    if plot_distribution: 
        plt.figure(figsize=(12,7))
        sns.set(font_scale = 1.5) 
        sns.distplot(df['Cora'], axlabel = Property)
        plt.savefig(txt_dir + '/data distribution/planetoid/' + 'Cora' +'_' + Property +'_dis.eps', dpi = 800, format = 'eps')
        plt.show()
    df = df.melt(var_name=Property, value_name='value')
    plt.figure(figsize=(10,10))
    sns.set(font_scale = 2) 
    sns.violinplot(x=Property, y='value', data=df)
    plt.savefig(txt_dir + '/violin_plot/' + Property +'_' + dataset +'_vp.eps', dpi = 800, format ='eps') # 

    plt.show()

   





if __name__ == "__main__":
    prop = ['Degree', 'Clustering_coefficient', 'Pagerank', 'Aver_path_len']
    for i in prop:
        read_file(Property=i,plot_distribution = True,count_zero=True)

    
