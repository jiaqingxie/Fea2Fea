from itertools import chain, combinations
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

if __name__ == '__main__':
    #test1
    a = [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (2, 3, 4), (0, 2, 3, 4)]
    min_len, max_len = max_len_arr(a)
    print(min_len, max_len)

    