from itertools import chain, combinations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import numpy as np

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def subset(a, b):
    # for example, a is [1, 2, 3], b is [(1,2), (2,3), (3,2),..]
    ans = True
    for ii in range(len(a)):
        for jj in range(ii + 1, len(a)):
            if ((a[ii], a[jj]) in b) and ((a[jj], a[ii]) in b):
                continue
            else:
                ans = False
    return ans

def max_len_arr(arr):
    '''
    if arr is an array like: arr = [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (2, 3, 4), (0, 2, 3, 4)]
    then this function will return the miminum element length in this array: 2 and
    the maximum element length in this array: 3
    '''
    max_len = 0
    min_len = 10001
    for i in arr:
        if len(i) >= max_len:
            max_len = len(i)
        if len(i) <= min_len:
            min_len = len(i)
    
    return min_len, max_len

def tSNE_vis(embedding, label, type, d_name, inp, outp, num_label):
    # x: input vector
    # type: initial, graph_embed(before MLP), MLP_embed(after MLP) 
    # label: y1, y2, y3, ... yn
    # d_name: dataset name 
    # inp: input feature index 
    # oup: output feature index
    # num_label: number of labels
    # draw tSNE pictures here:
    x = embedding.cpu().detach().numpy()
    label = label.cpu().detach()
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(x)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1,1,1,)

    values = range(num_label)
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scaMap = plt.cm.ScalarMappable(norm = cNorm  ,cmap = "coolwarm")

    for k in range(num_label):  
        colorval = scaMap.to_rgba(values[k])
        ax.scatter(X_tsne[np.where(label.numpy() == k), 0], X_tsne[np.where(label.numpy() == k), 1] ,label = k, s =3, color = colorval)


    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right',fontsize = 7)
    plt.xlabel("tSNE 1",fontsize = 12)
    plt.ylabel("tSNE 2", fontsize = 12)
    plt.tick_params(labelsize=12)
    name2 = r'/home/jiaqing/桌面/Fea2Fea/Result/tSNE/'
    plt.savefig('{}{},{}_{}to{}_tSNE{}.eps'.format(name2, str(d_name), str(inp[0]), str(inp[1]), str(outp), type), dpi = 800, format = 'eps')
    #plt.show()
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tmp.numpy(), cmap = "rainbow")
    #plt.legend()



if __name__ == '__main__':
    #test1
    a = [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (2, 3, 4), (0, 2, 3, 4)]
    min_len, max_len = max_len_arr(a)
    print(min_len, max_len)

    