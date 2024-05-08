# Input: Node Structure, Connection Map, CellType, FirstLayerStim, MeanDelay, stdDelay
# Output: all spiketime as list of list
import numpy as np
from numpy import asarray
from numpy import savetxt
import pickle
import lzma

def runingFFnetowrk_singalparam(param_dic):
    output = runingFFnetwork(param_dic['network_connmat'], param_dic['CellType'], param_dic['total_ruuning_time'],
                          param_dic['stimulations'], param_dic['MeanDelay'], param_dic['stdDelay'])
    with lzma.open('./savedoutput/'+param_dic['unique_id']+".xz", "wb") as fp:
        pickle.dump(output, fp)
    # with open('./savedoutput/'+param_dic['unique_id']+".pkl", "wb") as fp:
    #     pickle.dump(output, fp)
    return 0 #output

def runingFFnetwork(network_connmat, CellType, total_ruuning_time,
                    stimulations, MeanDelay, stdDelay):
    from neuron import h, gui
    from neuron.units import ms, mV
    import Cell
    import spikedelayer as sd
    import Network

    if CellType == "point":
        CellType_obj = Cell.PointCell

    sample_network = Network.Network(connection_mat = network_connmat, cell_type = CellType_obj,
                                    position_info = None, syn_mean_delay = MeanDelay, 
                                     syn_std_delay = stdDelay)
    
    # print(h.topology())
    
    for j in range(len(stimulations)):
        sample_network.cells[j].add_custom_stimulus(stimulations[j])

    t = h.Vector().record(h._ref_t)
    h.finitialize(-65 * mV)
    h.continuerun(total_ruuning_time)
    
    spiketime_all = []
    for idx in range(len(sample_network.cells)):
        spiketime_all.append(list(sample_network.cells[idx].spike_times))
    
    return spiketime_all