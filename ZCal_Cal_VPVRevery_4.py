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



def get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./savedoutput/" + name + ".xz", "rb") as fp:
        outsaved = pickle.load(fp)
    return outsaved

def save_vpdist(output_tosave, networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./VP_every/" + name + "_vp.xz", "wb") as fp:
        pickle.dump(output_tosave, fp)
        
def save_vrdist(output_tosave, networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./VR_every/" + name + "_vr.xz", "wb") as fp:
        pickle.dump(output_tosave, fp)




def get_vpandcount(baseline, distrubed, qnum):
    # Calculating the victor purpora distace of a single cell spiking
    duration = 10000
    train1 = neo.SpikeTrain(baseline * ms, t_stop=duration*ms)
    train2 = neo.SpikeTrain(distrubed * ms, t_stop=duration*ms)
    
    # q = 1.0 / (10.0 * ms) # used in other paper
    q = qnum / ms # cost factor for shifting spikes in the victor purpura distance
    vp_dist = victor_purpura_distance([train1, train2], q)[0, 1]
    
    if len(baseline) == 0:
        return 0,0
    
    return vp_dist/len(baseline), len(baseline)


def get_vrandcount(baseline, distrubed, taunum):
    # Calculating the van rossum distace of a single cell spiking
    duration = 10000
    train1 = neo.SpikeTrain(baseline * ms, t_stop=duration*ms)
    train2 = neo.SpikeTrain(distrubed * ms, t_stop=duration*ms)
    
    # tau = 1s default? 10ms   why 10ms, vab rissom paper
    # tau = 10.0 * ms # time constant for the van rossum distance
    tau = 1000 * taunum * ms
    vr_dist = van_rossum_distance([train1, train2], tau)[0, 1]
    
    if len(baseline) == 0:
        return 0,0

    return vr_dist/ len(baseline), len(baseline)


def get_vpdist_singlerun(baseline_all, distrubed_all, qnum, stddelay, layercount):
    # Getting the vp metrics between layers of network of a single simulation
    vplst_bylayer = []
    spikelst_bylayer = []
    layernumberlst_bylayer = []
    for layer_num in range(10):
        for cell_num in range(layercount):
            baseline = baseline_all[layer_num * layercount + cell_num]
            distrubed = distrubed_all[layer_num * layercount + cell_num]
            cell_vp, cell_spike_count = get_vpandcount(baseline, distrubed, qnum)
            vplst_bylayer.append(cell_vp)
            spikelst_bylayer.append(cell_spike_count)
            layernumberlst_bylayer.append(layer_num)
    # print(vplst_bylayer)
    # print(spikelst_bylayer)
    # print(layernumberlst_bylayer)
    return np.array([vplst_bylayer, spikelst_bylayer, layernumberlst_bylayer]).T

def get_vrdist_singlerun(baseline_all, distrubed_all, taunum, stddelay, layercount):
    # Getting the vr metrics between layers of network of a single simulation
    vrlst_bylayer = []
    spikelst_bylayer = []
    layernumberlst_bylayer = []
    for layer_num in range(10):
        for cell_num in range(layercount):
            baseline = baseline_all[layer_num * layercount + cell_num]
            distrubed = distrubed_all[layer_num * layercount + cell_num]
            cell_vr, cell_spike_count = get_vrandcount(baseline, distrubed, taunum)
            vrlst_bylayer.append(cell_vr)
            spikelst_bylayer.append(cell_spike_count)
            layernumberlst_bylayer.append(layer_num)
    return np.array([vrlst_bylayer, spikelst_bylayer, layernumberlst_bylayer]).T


def cal_VPdistbyStd(networktype, layercount, meandelay, stddelay, qnum):
    all_vp = []
    for model_id in np.arange(0, 10, 1):
        for input_idx in range(10):
            baseline_all = get_data(networktype, layercount, meandelay, 0.0, model_id, input_idx, 0, CellType = "point")
            for n_run in np.arange(0, 10, 1): 
                distrubed_all = get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point")
                singlerun_vpdist = get_vpdist_singlerun(baseline_all, distrubed_all, qnum, stddelay, layercount)
                save_vpdist(singlerun_vpdist,networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point") 
    return 0


def cal_VRdistbyStd(networktype, layercount, meandelay, stddelay, taunum):
    all_vp = []
    for model_id in np.arange(0, 10, 1):
        for input_idx in range(10):
            baseline_all = get_data(networktype, layercount, meandelay, 0.0, model_id, input_idx, 0, CellType = "point")
            for n_run in np.arange(0, 10, 1): 
                distrubed_all = get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point")
                singlerun_vrdist = get_vrdist_singlerun(baseline_all, distrubed_all, taunum, stddelay, layercount)
                save_vrdist(singlerun_vrdist,networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point") 
    return 0


def cal_VPdistbyStd_dicinput(input_dic):
    output = cal_VPdistbyStd(input_dic['networktype'], input_dic['layercount'], input_dic['meandelay'], input_dic['stddelay'], input_dic['qnum'])
    with open('./VP_counter/VP_' + str(input_dic['counter']) + '.pkl', 'wb') as f: pickle.dump([], f)
    return 0


def cal_VRdistbyStd_dicinput(input_dic):
    output = cal_VRdistbyStd(input_dic['networktype'], input_dic['layercount'], input_dic['meandelay'], input_dic['stddelay'], input_dic['taunum'])
    with open('./VR_counter/VP_' + str(input_dic['counter']) + '.pkl', 'wb') as f: pickle.dump([], f)
    return 0


if __name__ == '__main__':
    argdict_lst_vp = []
    argdict_lst_vr = []
    i = 0
    for layercount in [30, 40, 50, 60]:
        for MeanDelay_noround in np.arange(2, 3.01, 0.2):
            MeanDelay = np.round(MeanDelay_noround,1)
            for stdDelay_noround in np.arange(0, 1.01, 0.05):
                stdDelay = np.round(stdDelay_noround,2)
                argdict_vp = {'networktype':'FeedForward', 'layercount':layercount , 'meandelay':MeanDelay, 'stddelay':stdDelay, 'qnum':0.1, 'counter':i}
                argdict_lst_vp.append(argdict_vp)
                argdict_vr = {'networktype':'FeedForward', 'layercount':layercount , 'meandelay':MeanDelay, 'stddelay':stdDelay, 'taunum':0.01, 'counter':i}
                argdict_lst_vr.append(argdict_vr)
                # cal_VPdistbyStd('FeedForward', layercount, MeanDelay, stdDelay, 0.1)
                # cal_VRdistbyStd('FeedForward', layercount, MeanDelay, stdDelay, 0.01)
                # cal_VPdistbyStd_dicinput(argdict_vp)
                # cal_VRdistbyStd_dicinput(argdict_vr)
                i += 1
                
                
    start_idx = 126 + 126 + 126
    end_idx = 126 + 126 + 126 + 126
    with multiprocessing.Pool() as pool:
        # output = pool.map(cal_VPdistbyStd_dicinput, argdict_lst_vp[start_idx:end_idx])
        output = pool.map(cal_VRdistbyStd_dicinput, argdict_lst_vr[start_idx:end_idx])