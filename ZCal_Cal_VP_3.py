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




# for model_id in np.arange(0, 10, 1):
#     for input_idx in range(10):
#         for n_run in np.arange(0, 10, 1):
            
def get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run)
    with lzma.open("./savedoutput/" + name + ".xz", "rb") as fp:
        outsaved = pickle.load(fp)
    return outsaved

def get_vp(baseline, distrubed, qnum):
    # Calculating the victor purpora distace of a single cell spiking
    duration = 10000
    train1 = neo.SpikeTrain(baseline * ms, t_stop=duration*ms)
    train2 = neo.SpikeTrain(distrubed * ms, t_stop=duration*ms)
    
    # q = 1.0 / (10.0 * ms) # used in other paper
    q = qnum / ms # cost factor for shifting spikes in the victor purpura distance
    vp_dist = victor_purpura_distance([train1, train2], q)[0, 1]
    
    
    if len(baseline) == 0:
        return 0
    return vp_dist/ len(baseline)

def get_vr(baseline, distrubed, taunum):
    # Calculating the van rossum distace of a single cell spiking
    duration = 10000
    train1 = neo.SpikeTrain(baseline * ms, t_stop=duration*ms)
    train2 = neo.SpikeTrain(distrubed * ms, t_stop=duration*ms)
  
    # tau = 1s default? 10ms   why 10ms, vab rissom paper
    # tau = 10.0 * ms # time constant for the van rossum distance
    tau = 1000 * taunum * ms
    vr_dist = van_rossum_distance([train1, train2], tau)[0, 1]
    
    if len(baseline) == 0:
        return 0
    return vr_dist/ len(baseline)

# def get_devarvp(baseline, distrubed, qnum, stddelay):
#     # Calculating the devarianced victor purpora distace of a single cell spiking
#     duration = 10000
#     train1 = neo.SpikeTrain(baseline * ms, t_stop=duration*ms)
#     train2 = neo.SpikeTrain(distrubed * ms, t_stop=duration*ms)
#     # q = 1.0 / (10.0 * ms)
#     q = qnum / ms # cost factor for shifting spikes in the victor purpura distance
#     vp_dist = victor_purpura_distance([train1, train2], q)[0, 1]
#     # tau = 10.0 * ms # time constant for the van rossum distance
#     # vr_dist = van_rossum_distance([train1, train2], tau)[0, 1]
#     if len(baseline) == 0:
#         return 0
#     return (vp_dist - qnum * len(baseline) * stddelay * np.sqrt(2/np.pi)) / len(baseline)

def get_vp_singlerun(baseline_all, distrubed_all, qnum, stddelay, layercount):
    # Getting the vp metrics between layers of network of a single simulation
    avgvp_bylayer = []
    # avgdevarvp_bylayer = []
    for layer_num in range(10):
        layer_vp = []
        # layer_devarvp = []
        for cell_num in range(layercount):
            baseline = baseline_all[layer_num * layercount + cell_num]
            distrubed = distrubed_all[layer_num * layercount + cell_num]
            cell_vp = get_vp(baseline, distrubed, qnum)
            # cell_devarvp = get_devarvp(baseline, distrubed, qnum, stddelay)
            layer_vp.append(cell_vp)
            # layer_devarvp.append(cell_devarvp)
        avgvp_bylayer.append(np.mean(layer_vp))
        # avgdevarvp_bylayer.append(np.mean(layer_devarvp))
    # return np.array(avgvp_bylayer), np.array(avgdevarvp_bylayer)
    return np.array(avgvp_bylayer)

def get_vr_singlerun(baseline_all, distrubed_all, taunum, stddelay, layercount):
    # Getting the vp metrics between layers of network of a single simulation
    avgvp_bylayer = []
    # avgdevarvp_bylayer = []
    for layer_num in range(10):
        layer_vp = []
        # layer_devarvp = []
        for cell_num in range(layercount):
            baseline = baseline_all[layer_num * layercount + cell_num]
            distrubed = distrubed_all[layer_num * layercount + cell_num]
            cell_vp = get_vr(baseline, distrubed, taunum)
            # cell_devarvp = get_devarvp(baseline, distrubed, qnum, stddelay)
            layer_vp.append(cell_vp)
            # layer_devarvp.append(cell_devarvp)
        avgvp_bylayer.append(np.mean(layer_vp))
        # avgdevarvp_bylayer.append(np.mean(layer_devarvp))
    # return np.array(avgvp_bylayer), np.array(avgdevarvp_bylayer)
    return np.array(avgvp_bylayer)
    
        
def cal_VPbyStd(networktype, layercount, meandelay, stddelay, qnum):
    all_vp = []
    # all_devarvp = []
    for model_id in np.arange(0, 10, 1):
        for input_idx in range(10):
            baseline_all = get_data(networktype, layercount, meandelay, 0.0, model_id, input_idx, 0, CellType = "point")
            for n_run in np.arange(0, 10, 1): 
                distrubed_all = get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point")
                # singlerun_vp, singlerun_devarvp = get_vps_singlerun(baseline_all, distrubed_all, qnum, stddelay, layercount)
                singlerun_vp = get_vp_singlerun(baseline_all, distrubed_all, qnum, stddelay, layercount)
                all_vp.append(singlerun_vp)
                # all_devarvp.append(singlerun_devarvp)
    with open('./VP_processing/'+ networktype + "_layercount" + str(layercount) + "_meandelay" + str(meandelay) + "_stddelay" + str(stddelay) + "_qum" + str(qnum) + "_VP.pkl",'wb') as f: pickle.dump(all_vp, f)
    # with open('./VP_processing/'+ networktype + "_layercount" + str(layercount) + "_meandelay" + str(meandelay) + "_stddelay" + str(stddelay) + "_qum" + str(qnum) + "_devarVP.pkl",'wb') as f: pickle.dump(all_devarvp, f)
    return 0

def cal_VRbyStd(networktype, layercount, meandelay, stddelay, taunum):
    all_vp = []
    # all_devarvp = []
    for model_id in np.arange(0, 10, 1):
        for input_idx in range(10):
            baseline_all = get_data(networktype, layercount, meandelay, 0.0, model_id, input_idx, 0, CellType = "point")
            for n_run in np.arange(0, 10, 1): 
                distrubed_all = get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point")
                # singlerun_vp, singlerun_devarvp = get_vps_singlerun(baseline_all, distrubed_all, qnum, stddelay, layercount)
                singlerun_vp = get_vr_singlerun(baseline_all, distrubed_all, taunum, stddelay, layercount)
                all_vp.append(singlerun_vp)
                # all_devarvp.append(singlerun_devarvp)
    with open('./VR_processing/'+ networktype + "_layercount" + str(layercount) + "_meandelay" + str(meandelay) + "_stddelay" + str(stddelay) + "_tau" + str(taunum) + "_VR.pkl",'wb') as f: pickle.dump(all_vp, f)
    # with open('./VP_processing/'+ networktype + "_layercount" + str(layercount) + "_meandelay" + str(meandelay) + "_stddelay" + str(stddelay) + "_qum" + str(qnum) + "_devarVP.pkl",'wb') as f: pickle.dump(all_devarvp, f)
    return 0

def cal_VPbyStd_dicinput(input_dic):
    output = cal_VPbyStd(input_dic['networktype'], input_dic['layercount'], input_dic['meandelay'], input_dic['stddelay'], input_dic['qnum'])
    with open('./VR_processing/VP_' + str(input_dic['counter']) + '.pkl', 'wb') as f: pickle.dump([], f)
    return 0

def cal_VRbyStd_dicinput(input_dic):
    output = cal_VRbyStd(input_dic['networktype'], input_dic['layercount'], input_dic['meandelay'], input_dic['stddelay'], input_dic['taunum'])
    with open('./VR_processing/VR_' + str(input_dic['counter']) + '.pkl', 'wb') as f: pickle.dump([], f)
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
                i += 1
                # cal_VPbyStd('FeedForward', layercount, MeanDelay, stdDelay, 0.1)
                # cal_VRbyStd('FeedForward', layercount, MeanDelay, stdDelay, 0.01)
# processes=2
    #Seperate into 4 pieces:
    start_idx = 126
    end_idx = 126 + 63
    with multiprocessing.Pool() as pool:
       # output = pool.map(cal_VRbyStd_dicinput, argdict_lst_vr[start_idx:end_idx])
        output = pool.map(cal_VPbyStd_dicinput, argdict_lst_vp[start_idx:end_idx])

            


        
        
