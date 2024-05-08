import numpy as np
import pickle
import lzma

import elephant.statistics as estats
import elephant
import neo
from quantities import ms, s, Hz

from elephant.spike_train_dissimilarity import victor_purpura_distance
from elephant.spike_train_dissimilarity import van_rossum_distance

import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing



def get_vpdist(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./VP_every/" + name + "_vp.xz", "rb") as fp:
          outsaved = pickle.load(fp)
    return outsaved
        
def get_vrdist(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./VR_every_decomplex/" + name + "_vr.xz", "rb") as fp:
          outsaved = pickle.load(fp)
    return outsaved


if __name__ == '__main__':
    networktype = 'FeedForward'
    CellType = 'point'

    std_delay_lst = []
    for stdDelay_noround in np.arange(0.0, 1.01, 0.05):
        stdDelay = np.round(stdDelay_noround,2)
        std_delay_lst.append(stdDelay)

    mean_delay_lst = []
    for MeanDelay_noround in np.arange(2, 3.01, 0.2):
        MeanDelay = np.round(MeanDelay_noround,1)
        mean_delay_lst.append(MeanDelay)

    counter = 0
    alloutsaved = []
    # for layercount in [30, 40, 50, 60]:
    for layercount in [30]:
        for MeanDelay in mean_delay_lst:
            for stddelay in std_delay_lst:
                for modelid in range(10):
                    for inputid in range(10):
                        for nrun in range(10):    
                            outsaved = pd.DataFrame(get_vpdist(networktype, layercount, MeanDelay, stddelay, modelid, inputid, nrun, CellType))
                            outsaved.columns = {'avgVP', 'SpikeCountatBase', 'Layer#'}
                            outsaved['avgVP'] = outsaved['avgVP'].astype('float64')
                            outsaved['SpikeCountatBase'] = outsaved['SpikeCountatBase'].astype('int')
                            outsaved['Layer#'] = outsaved['Layer#'].astype('int')
                            outsaved['layercount'] = layercount
                            outsaved['stddelay'] = stddelay
                            outsaved['MeanDelay'] = MeanDelay
                            outsaved['modelid'] = modelid
                            outsaved['inputid'] = inputid
                            outsaved['nrun'] = nrun
                            alloutsaved.append(outsaved)
                            if counter % 1000 == 0 :
                                print(counter)
                            counter += 1

    combined_df = pd.concat(alloutsaved, axis=0, ignore_index=True)

    with lzma.open("./VPofEveryCellEveryRun.xz", "wb") as fp:
        pickle.dump(combined_df, fp)
        
        
#     networktype = 'FeedForward'
#     CellType = 'point'

#     std_delay_lst = []
#     for stdDelay_noround in np.arange(0.0, 1.01, 0.05):
#         stdDelay = np.round(stdDelay_noround,2)
#         std_delay_lst.append(stdDelay)

#     mean_delay_lst = []
#     for MeanDelay_noround in np.arange(2, 3.01, 0.2):
#         MeanDelay = np.round(MeanDelay_noround,1)
#         mean_delay_lst.append(MeanDelay)

#     counter = 0
#     alloutsaved = []
#     for layercount in [30, 40, 50, 60]:
#         for MeanDelay in mean_delay_lst:
#             for stddelay in std_delay_lst:
#                 for modelid in range(10):
#                     for inputid in range(10):
#                         for nrun in range(10):    
#                             outsaved = pd.DataFrame(get_vrdist(networktype, layercount, MeanDelay, stddelay, modelid, inputid, nrun, CellType))
#                             outsaved.columns = {'avgVR', 'SpikeCountatBase', 'Layer#'}
#                             outsaved['avgVR'] = outsaved['avgVR'].astype('float64')
#                             outsaved['SpikeCountatBase'] = outsaved['SpikeCountatBase'].astype('int')
#                             outsaved['Layer#'] = outsaved['Layer#'].astype('int')
#                             outsaved['layercount'] = layercount
#                             outsaved['stddelay'] = stddelay
#                             outsaved['MeanDelay'] = MeanDelay
#                             outsaved['modelid'] = modelid
#                             outsaved['inputid'] = inputid
#                             outsaved['nrun'] = nrun
#                             alloutsaved.append(outsaved)
#                             if counter % 1000 == 0 :
#                                 print(counter)
#                             counter += 1

#     combined_df = pd.concat(alloutsaved, axis=0, ignore_index=True)

#     with lzma.open("./VRofEveryCellEveryRun.xz", "wb") as fp:
#         pickle.dump(combined_df, fp)