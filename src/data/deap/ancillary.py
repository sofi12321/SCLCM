import numpy as np
import pandas as pd

def deap_metadata():
    num_video = 40
    subj_list = np.array([26, 27, 28, 29, 30, 31])
    video_list = np.array([0,1,2,3,4,5,6,7])
    return num_video, subj_list, video_list

def parse_valence(labels, threshold = 20):
    valence = (labels[:,:,0] > 6)*1 + ((labels[:,:,0] <= 6) & (labels[:,:,0] >= 5))*2
    df = pd.DataFrame(valence, columns = [f"video_{i+1}" for i in range(valence.shape[1])])
    df1 = pd.concat(((df == 0).sum(), (df == 1).sum(), (df == 2).sum()), axis = 1)
    # df2 = pd.concat((df1[0]/df1.sum(axis = 1), df1[1]/df1.sum(axis = 1), df1[2]/df1.sum(axis = 1)), axis = 1)
    # plt.figure()
    # df1.plot.bar(figsize=(18,3))
    df3 = df1[df1 >= threshold].idxmax(axis = 1)
    result = []
    for j in range(40):
        tmp = []
        val = df3.loc[f"video_{j+1}"]
        # print(val)
        for i in range(32):
            if np.isnan(val):
                tmp.append(0)
            elif df.iloc[i, j] == val:
                tmp.append(1)
            else:
                tmp.append(0)
        result.append(np.array(tmp))
    return np.array(result).T
