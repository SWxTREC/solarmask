import os
import pandas as pd

if __name__ == '__main__':
    f = "./labels.csv"
    if not os.path.exists(f):
        print("No full file, try merge.py")
        exit(1)
    
    df = pd.read_csv(f, index_col = 0)
    df.head(5)

    i = 0
    inc = 10000
    while i < df.shape[0]:
        k = min(df.shape[0], i + inc)
        df_ = df.iloc[i:k,:]
        df_.to_csv(os.path.join("./labels", str(i) + "_" + str(k - 1) + ".csv"))
        i += inc
