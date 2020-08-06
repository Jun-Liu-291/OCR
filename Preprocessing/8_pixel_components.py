# row-by-row search
cc = 1


Out = np.zeros(shape=(h_pooling,w_pooling))
for n in range(h_pooling):
    for m in range(w_pooling):
        if In[n,m] == 0:
            Out[n,m] == 0
        else:
            if n == 0:
                if Out[n,m-1] == 0:
                    Out[n,m] == cc
                    cc += 1
                else:
                    Out[n,m] = Out[n,m-1]
            elif m == 0:
                if Out[n-1,m] == 0 and Out[n-1,m+1] == 0:
                    Out[n,m] = cc
                    cc += 1
                elif Out[n-1,m] != 0 and Out[n-1,m+1] == 0:
                    Out[n,m] = Out[n-1,m]
                elif Out[n-1,m] == 0 and Out[n-1,m+1] != 0:
                    Out[n,m] = Out[n-1,m+1]
                else:
                    Out[n,m] = Out[n-1,m]
            
            elif m ==  w_line - 1:
                if Out[n-1,m-1]==0 and Out[n-1,m]==0 and Out[n,m-1]==0:
                    Out[n,m] = cc
                    cc += 1
                elif Out[n-1,m-1]!=0 and Out[n-1,m]==0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]==0 and Out[n-1,m]!=0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m]
                elif Out[n-1,m-1]==0 and Out[n-1,m]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n,m-1]
                elif Out[n-1,m-1]!=0 and Out[n-1,m]!=0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]!=0 and Out[n-1,m]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]==0 and Out[n-1,m]!=0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n-1,m]
                else:
                    Out[n,m] = Out[n-1,m-1]
            else:
                if Out[n-1,m-1]==0 and Out[n-1,m]==0 and Out[n-1,m+1]==0 and Out[n,m-1]==0:
                    Out[n,m] = cc
                    cc += 1
                elif Out[n-1,m-1]!=0 and Out[n-1,m]==0 and Out[n-1,m+1]==0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]==0 and Out[n-1,m]!=0 and Out[n-1,m+1]==0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m]
                elif Out[n-1,m-1]==0 and Out[n-1,m]==0 and Out[n-1,m+1]!=0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m+1]
                elif Out[n-1,m-1]==0 and Out[n-1,m]==0 and Out[n-1,m+1]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n,m-1]
                elif Out[n-1,m-1]!=0 and Out[n-1,m]!=0 and Out[n-1,m+1]==0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]!=0 and Out[n-1,m]==0 and Out[n-1,m+1]!=0 and Out[n,m-1]==0:
                    if Out[n-1,m-1] == Out[n-1,m+1]:
                        Out[n,m] = Out[n-1,m-1]
                    elif Out[n-1,m-1] > Out[n-1,m+1]:
                        Out[n,m] = Out[n-1,m+1]
                        cc -= 1
                        v = Out[n-1,m-1]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n-1,m+1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                    else:
                        Out[n,m] = Out[n-1,m-1]
                        cc -= 1
                        v = Out[n-1,m+1]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n-1,m-1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                elif Out[n-1,m-1]!=0 and Out[n-1,m]==0 and Out[n-1,m+1]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]==0 and Out[n-1,m]!=0 and Out[n-1,m+1]!=0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m]
                elif Out[n-1,m-1]==0 and Out[n-1,m]!=0 and Out[n-1,m+1]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n-1,m]
                elif Out[n-1,m-1]==0 and Out[n-1,m]==0 and Out[n-1,m+1]!=0 and Out[n,m-1]!=0:
                    if Out[n-1,m+1] == Out[n,m-1]:
                        Out[n,m] = Out[n-1,m+1]
                    elif Out[n-1,m+1] > Out[n,m-1]:
                        Out[n,m] = Out[n,m-1]
                        cc -= 1
                        v = Out[n-1,m+1]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n,m-1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                    else:                        
                        Out[n,m] = Out[n-1,m+1]
                        cc -= 1
                        v = Out[n,m-1]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n-1,m+1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                elif Out[n-1,m-1]!=0 and Out[n-1,m]!=0 and Out[n-1,m+1]!=0 and Out[n,m-1]==0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]!=0 and Out[n-1,m]==0 and Out[n-1,m+1]!=0 and Out[n,m-1]!=0:
                    if Out[n-1,m-1] == Out[n-1,m+1]:
                        Out[n,m] = Out[n-1,m-1]
                    elif Out[n-1,m-1] > Out[n-1,m+1]:
                        Out[n,m] = Out[n-1,m+1]
                        cc -= 1
                        v = Out[n-1,m-1]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n-1,m+1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                    else:
                        Out[n,m] = Out[n-1,m-1]
                        cc -= 1
                        v = Out[n-1,m+1]
                        for a in range(h_pooling):
                            for b in range(w_pooling):
                                if Out[a,b] == v:
                                    Out[a,b] = Out[n-1,m-1]
                                elif Out[a,b] > v:
                                    Out[a,b] -= 1
                elif Out[n-1,m-1]!=0 and Out[n-1,m]!=0 and Out[n-1,m+1]==0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n-1,m-1]
                elif Out[n-1,m-1]==0 and Out[n-1,m]!=0 and Out[n-1,m+1]!=0 and Out[n,m-1]!=0:
                    Out[n,m] = Out[n-1,m]
                else:
                    Out[n,m] = Out[n-1,m-1]
                    
max_value = [] 
for n in range(h_pooling):
    max_value.append(max(Out[n]))
label_max = max(max_value)
#print(label_max)

for i in range(10):
    ch_image = np.zeros(shape=(h_pooling,w_pooling))
    for n in range(h_pooling):
        for m in range(w_pooling):
            if Out[n,m] == i+30:
                ch_image[n,m] = 1
    plt.subplot(10, 1, i+1)
    plt.imshow(ch_image,cmap=plt.get_cmap('binary'))