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
    with lzma.open("./VR_every/" + name + "_vr.xz", "rb") as fp:
          outsaved = pickle.load(fp)
    return outsaved

def save_vrdist(output_tosave, networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./VR_every_decomplex/" + name + "_vr.xz", "wb") as fp:
        pickle.dump(output_tosave, fp)

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

    alloutsaved = []
    counter = 0 
    for layercount in [50, 60]:
        for MeanDelay in mean_delay_lst:
            for stddelay in std_delay_lst:
                for modelid in range(10):
                    for inputid in range(10):
                        for nrun in range(10):

                            outsaved = pd.DataFrame(get_vrdist(networktype, layercount, MeanDelay, stddelay, modelid, inputid, nrun, CellType))
                            # print(outsaved)
                            outsaved[0] = outsaved[0].apply(lambda x: x.real).astype('float64')
                            outsaved[1] = outsaved[1].apply(lambda x: x.real).astype('int')
                            outsaved[2] = outsaved[2].apply(lambda x: x.real).astype('int')
                            # print(outsaved)
                            save_vrdist(outsaved, networktype, layercount, MeanDelay, stddelay, modelid, inputid, nrun, CellType)
                            if counter % 1000 == 0:
                                print(counter)
                            counter +=1
