# row-by-row search
cc = 1


Out = np.zeros(shape=(h_pooling,w_pooling))
for n in range(h_pooling):
    for m in range(w_pooling):
        if In[n,m] == 0:
            Out[n,m] == 0
        elif In[n,m] == 1:
            if n == 0:
                if Out[n,m-1] == 0:
                    Out[n,m] == cc
                    cc += 1
                else:
                    Out[n,m] = Out[n,m-1]
            elif m == 0:
                if Out[n-1,m] == 0:
                    Out[n,m] = cc
                    cc += 1
                elif Out[n-1,m] != 0:
                    Out[n,m] = Out[n-1,m]
            
            else:
                if Out[n-1,m]==0 and Out[n,m-1]==0:
                    Out[n,m] = cc
                    cc += 1

                elif Out[n-1,m]!=0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m]
                elif Out[n-1,m]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n,m-1]
                else:
                    if Out[n-1,m] == Out[n,m-1]:
                        Out[n,m] = Out[n-1,m]
                    elif Out[n-1,m] > Out[n,m-1]:
                        Out[n,m] = Out[n,m-1]
                        cc -= 1
                        v = Out[n-1,m]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n,m-1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                    else:
                        Out[n,m] = Out[n-1,m]
                        v = Out[n,m-1]
                        cc -= 1
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n-1,m]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1