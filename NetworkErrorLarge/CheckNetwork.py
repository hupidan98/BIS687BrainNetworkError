import numpy as np
import pickle
import lzma
import pandas as pd
import matplotlib.pyplot as plt

def get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./savedoutput/" + name + ".xz", "rb") as fp:
        outsaved = pickle.load(fp)
    return outsaved

if __name__ == '__main__':
    for networktype in ['FeedForward', 'ScaleFree', 'SmallWorld']:
        for layercount in [200]:
            for MeanDelay_noround in np.arange(3, 3.01, 0.2):
                MeanDelay = np.round(MeanDelay_noround,1)
                for stdDelay_noround in np.arange(0, 1.01, 0.05):
                    stdDelay = np.round(stdDelay_noround,2)
                    for model_id in np.arange(0, 10, 1):
                        for input_idx in range(10):
                            for n_run in np.arange(0, 10, 1):
                                try:
                                    get_data(networktype, layercount, MeanDelay, stdDelay, model_id, input_idx, n_run, CellType = "point")
                                except Exception as e:
                                    print("An error occurred: " + str(e))
                                    print(networktype, layercount, MeanDelay, stdDelay, model_id, input_idx, n_run)