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

def save_vpdist(output_tosave, networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, section, CellType = "point", ):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run) + '_section' + str(section)
    with lzma.open("./VP_every_section/" + name + "_vp.xz", "wb") as fp:
        pickle.dump(output_tosave, fp)
        
def save_vrdist(output_tosave, networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, section, CellType = "point"):
    name = networktype + '_' + str(CellType) + '_layercount' + str(layercount) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stddelay) + '_meandelay' + str(meandelay) + '_nrun' + str(n_run) + '_section' + str(section)
    with lzma.open("./VR_every_section/" + name + "_vr.xz", "wb") as fp:
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
    for layer_num in range(1):
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
    for layer_num in range(1):
        for cell_num in range(layercount):
            baseline = baseline_all[layer_num * layercount + cell_num]
            distrubed = distrubed_all[layer_num * layercount + cell_num]
            cell_vr, cell_spike_count = get_vrandcount(baseline, distrubed, taunum)
            vrlst_bylayer.append(cell_vr)
            spikelst_bylayer.append(cell_spike_count)
            layernumberlst_bylayer.append(layer_num)
    return np.array([vrlst_bylayer, spikelst_bylayer, layernumberlst_bylayer]).T


def cal_VPdistbyStd(networktype, layercount, meandelay, stddelay, qnum, section):
    all_vp = []
    for model_id in np.arange(0, 10, 1):
        for input_idx in range(10):
            baseline_all = get_data(networktype, layercount, meandelay, 0.0, model_id, input_idx, 0, CellType = "point")
            baseline_all_inrange = [[value for value in baseline_all_singlelist if (((section * 1000) <= value) & ((section * 1000+1000) > value))] for baseline_all_singlelist in baseline_all]
            for n_run in np.arange(0, 10, 1): 
                distrubed_all = get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point")
                distrubed_all_inrange = [[value for value in distrubed_all_singlelist if (((section * 1000) <= value) & ((section * 1000+1000) > value))] for distrubed_all_singlelist in distrubed_all]
                singlerun_vpdist = get_vpdist_singlerun(baseline_all_inrange, distrubed_all_inrange, qnum, stddelay, layercount)
                save_vpdist(singlerun_vpdist,networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, section, CellType = "point") 
    return 0


def cal_VRdistbyStd(networktype, layercount, meandelay, stddelay, taunum, section):
    all_vp = []
    for model_id in np.arange(0, 10, 1):
        for input_idx in range(10):
            baseline_all = get_data(networktype, layercount, meandelay, 0.0, model_id, input_idx, 0, CellType = "point")
            baseline_all_inrange = [[value for value in baseline_all_singlelist if (((section * 1000) <= value) & ((section * 1000+1000) > value))] for baseline_all_singlelist in baseline_all]
            for n_run in np.arange(0, 10, 1): 
                distrubed_all = get_data(networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, CellType = "point")
                distrubed_all_inrange = [[value for value in distrubed_all_singlelist if (((section * 1000) <= value) & ((section * 1000+1000) > value))] for distrubed_all_singlelist in distrubed_all]
                singlerun_vrdist = get_vrdist_singlerun(baseline_all_inrange, distrubed_all_inrange, taunum, stddelay, layercount)
                save_vrdist(singlerun_vrdist,networktype, layercount, meandelay, stddelay, model_id, input_idx, n_run, section, CellType = "point") 
    return 0


def cal_VPdistbyStd_dicinput(input_dic):
    try:
        print(str(input_dic['counter']) + 'vp')
        output = cal_VPdistbyStd(input_dic['networktype'], input_dic['layercount'], input_dic['meandelay'], input_dic['stddelay'], input_dic['qnum'], input_dic['section'])
        with open('./VP_counter_section/VP_' + str(input_dic['counter']) + '.pkl', 'wb') as f: pickle.dump([], f)
        print(str(input_dic['counter']) + 'donevp')
        with open('vp_section_counter.txt', 'a') as file:
            file.write(str(input_dic['counter']) + '_donevp'+"\n")
    except Exception as e:
        print("An error occurred: " + str(e))
        with open('vp_section_counter.txt', 'a') as file:
            file.write(str(input_dic['counter']) + "An error occurred: " + str(e)+"\n")
    return 0


def cal_VRdistbyStd_dicinput(input_dic):
    try:
        print(str(input_dic['counter']) + 'vr')
        output = cal_VRdistbyStd(input_dic['networktype'], input_dic['layercount'], input_dic['meandelay'], input_dic['stddelay'], input_dic['taunum'], input_dic['section'])
        with open('./VR_counter_section/VR_' + str(input_dic['counter']) + '.pkl', 'wb') as f: pickle.dump([], f)
        print(str(input_dic['counter']) + 'donevr')
    except Exception as e:
        print("An error occurred: " + str(e))
    return 0






if __name__ == '__main__':
    argdict_lst_vp = []
    argdict_lst_vr = []
    i = 0
    for networktype in ['SmallWorld']: #'ScaleFree', 
        for layercount in [60]:
            for MeanDelay_noround in np.arange(3.00, 3.01, 0.2):
                MeanDelay = np.round(MeanDelay_noround,1)
                for stdDelay_noround in np.arange(0, 1.01, 0.05):
                    stdDelay = np.round(stdDelay_noround,2)
                    for section in range(1):#10
                        argdict_vp = {'networktype':networktype, 'layercount':layercount , 'meandelay':MeanDelay, 'stddelay':stdDelay, 'qnum':0.1, 'counter':i, 'section': section}
                        argdict_lst_vp.append(argdict_vp)
                        argdict_vr = {'networktype':networktype, 'layercount':layercount , 'meandelay':MeanDelay, 'stddelay':stdDelay, 'taunum':0.01, 'counter':i, 'section': section}
                        argdict_lst_vr.append(argdict_vr)
                        i += 1
    start_idx = 0
    end_idx = len(argdict_lst_vp)
    # with multiprocessing.Pool() as pool:
    #     output = pool.map(cal_VPdistbyStd_dicinput, argdict_lst_vp[start_idx:end_idx])
    #     output = pool.map(cal_VRdistbyStd_dicinput, argdict_lst_vr[start_idx:end_idx])
    with multiprocessing.Pool() as pool:
        output = pool.map(cal_VPdistbyStd_dicinput, argdict_lst_vp[start_idx:end_idx])
        # output = pool.map(cal_VRdistbyStd_dicinput, argdict_lst_vr[start_idx:end_idx])
    print("done")