import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

files = sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/????/reduced/*MR*.csv"))

x=np.arange(2048)
for file in files:

    order_trace_data = pd.read_csv(file, header=0, index_col=0)
    coffs=(np.flip(order_trace_data.values[:,0:6],axis=1))
    poly_degree = order_trace_data.shape[1]-5

    for i in range(order_trace_data.shape[0]):
        plt.plot(i,order_trace_data.values[i,6],'x')
        plt.plot(i,order_trace_data.values[i,7],'o')
        ord_cen=np.polyval(coffs[i],x)
        
        #plt.plot(x,ord_cen)
    print(file)
    plt.show()
