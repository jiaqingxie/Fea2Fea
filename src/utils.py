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



if __name__ == '__main__':
    s = list(powerset([1,2,3,4,5]))

    