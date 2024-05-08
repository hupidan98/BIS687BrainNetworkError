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

def get_vpdist_sec(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, section, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run) + '_section' + str(section)
    with lzma.open("./VP_every_section/" + name + "_vp.xz", "rb") as fp:
          outsaved = pickle.load(fp)
    return outsaved
        

std_delay_lst = []
for stdDelay_noround in np.arange(0.0, 1.01, 0.05):
    stdDelay = np.round(stdDelay_noround,2)
    std_delay_lst.append(stdDelay)

mean_delay_lst = []
for MeanDelay_noround in np.arange(3.0, 3.01, 0.2):
    MeanDelay = np.round(MeanDelay_noround,1)
    mean_delay_lst.append(MeanDelay)
    
if __name__ == '__main__':
  


    networktype = 'ScaleFree'
    CellType = 'point'
    MeanDelay = 3.00

    for section in range(10):
        alloutsaved = []
        for stddelay in std_delay_lst:
            for modelid in range(10):
                for inputid in range(10):
                    for nrun in range(10):    
                        outsaved = pd.DataFrame(get_vpdist_sec(networktype, 200, MeanDelay, stddelay, modelid, inputid, nrun, section, CellType))
                        outsaved.columns = ['avgVP', 'SpikeCountatBase', 'Layer#']
                        outsaved['avgVP'] = outsaved['avgVP'].astype('float64')
                        outsaved['SpikeCountatBase'] = outsaved['SpikeCountatBase'].astype('int')
                        outsaved['Layer#'] = outsaved['Layer#'].astype('int')
                        outsaved['layercount'] = 60
                        outsaved['stddelay'] = stddelay
                        outsaved['MeanDelay'] = MeanDelay
                        outsaved['modelid'] = modelid
                        outsaved['inputid'] = inputid
                        outsaved['nrun'] = nrun
                        alloutsaved.append(outsaved)
        combined_df = pd.concat(alloutsaved, axis=0, ignore_index=True)
        print(str(networktype) + str(section))
        with lzma.open("./VP_agg_section/VPofEvery"+"Networktype_"+str(networktype)+"Sec"+str(section)+".xz", "wb") as fp:
            pickle.dump(combined_df, fp)
    
    
    