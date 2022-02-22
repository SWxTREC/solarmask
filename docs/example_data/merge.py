import os
import pandas as pd

if __name__ == '__main__':
    
    
    df = pd.DataFrame()
    
    for i in os.listdir("./labels"):
        df_ = pd.read_csv(os.path.join("./labels", i), index_col = 0)
        df = pd.concat([df, df_])
    
    df.reset_index()
    df.to_csv("./labels.csv", index = False)
