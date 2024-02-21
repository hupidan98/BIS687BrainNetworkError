import pickle
import numpy as np
import lzma
import scipy.signal
import tensortools as tt
import pickle
import matplotlib.pyplot as plt
import multiprocessing


def calTCA(paramdict):
    vec_lst = VecbyParm_withGaussian_dic(paramdict)
    ensemble = tt.Ensemble(fit_method="cp_als")
    ensemble.fit(vec_lst, ranks=range(1, 11), replicates=4)
    savefile_tca(ensemble, paramdict)
    return 0

def add_gaussian_kernel(input_vector, kernel_size, sigma):

    # Define the Gaussian kernel
    kernel = scipy.signal.gaussian(kernel_size, sigma)
    
    # Initialize the result vector with zeros as floating-point numbers
    result_vector = np.zeros(len(input_vector), dtype=float)
    
    # Iterate through the input vector
    for i, value in enumerate(input_vector):
        if value == 1.0:
            # Add the Gaussian kernel to the result vector centered around the current index
            start_idx = max(0, i - (kernel_size // 2))
            end_idx = min(len(input_vector), i + (kernel_size // 2) + 1)
            result_vector[start_idx:end_idx] += kernel[
                (kernel_size // 2) - (i - start_idx) : (kernel_size // 2) + (end_idx - i)
            ]
    
    return result_vector

sigma_my = 1.0 / 0.025
kernel_size_my = int(sigma_my) * 5 * 2 + 1

def FiletoVec(name):
    with lzma.open("./savedoutput/" + name + ".xz", "rb") as fp:
        outsaved = pickle.load(fp)
    vec_space = 0.025
    spike_mat = np.zeros((len(outsaved), int(10000/vec_space)))
    for i in range(len(outsaved)):
        for n in range(len(outsaved[i])):
            j = int(outsaved[i][n]/vec_space)
            spike_mat[i, j] = 1
    return spike_mat

def FiletoVec_param(NetworkType, n, model_id, input_idx, MeanDelay, stdDelay, n_run):
    CellType = "point"
    name = str(NetworkType) + '_' + str(CellType) + '_layercount' + str(n) + '_model' + str(model_id) + '_input' + str(input_idx) + '_stddelay' + str(stdDelay) + '_meandelay' + str(MeanDelay) + '_nrun' + str(n_run)
    return FiletoVec(name)

def VecbyParm(NetworkType, n, model_id, input_idx, MeanDelay, stdDelay):
    vec_lst = []
    for n_run in np.arange(0, 10, 1):
        vec_lst.append(FiletoVec_param(NetworkType, n, model_id, input_idx, MeanDelay, stdDelay, n_run))
    return np.stack(vec_lst)

def VecbyParm_withGaussian(NetworkType, n, model_id, input_idx, MeanDelay, stdDelay):
    noguas = VecbyParm(NetworkType, n, model_id, input_idx, MeanDelay, stdDelay)
    result_3d = np.apply_along_axis(add_gaussian_kernel, axis=2, arr=noguas, kernel_size = kernel_size_my, sigma = sigma_my)
    return result_3d

def VecbyParm_withGaussian_dic(paramdict):
    NetworkType, n, model_id, input_idx, MeanDelay, stdDelay = paramdict['NetworkType'], paramdict['n'], paramdict['model_id'], paramdict['input_idx'], paramdict['MeanDelay'], paramdict['stdDelay']
    noguas = VecbyParm(NetworkType, n, model_id, input_idx, MeanDelay, stdDelay)
    result_3d = np.apply_along_axis(add_gaussian_kernel, axis=2, arr=noguas, kernel_size = kernel_size_my, sigma = sigma_my)
    return result_3d

def savefile_tca(dataobj, paramdict):
    NetworkType, cellcount, model_id, input_idx, meandelay, stddelay = paramdict['NetworkType'], paramdict['n'], paramdict['model_id'], paramdict['input_idx'], paramdict['MeanDelay'], paramdict['stdDelay']
    with open('./TCAProcessing/'+ NetworkType + "_cellcount" + str(cellcount) + "_meandelay" + str(meandelay) + "_stddelay" + str(stddelay) + "_modelid" + str(model_id) + "_inputid" + str(input_idx) + "_spectral.pkl",'wb') as f: 
        pickle.dump(dataobj, f)
    with open('./TCACounter/'+ str(paramdict['counter'])+".pkl",'wb') as f: 
        pickle.dump([], f)


if __name__ == '__main__':
    paramlist = []
    counter = 0
    for networktype in ['FeedForward', 'ScaleFree', 'SmallWorld']:
        for cellcount in [30, 50, 60]:
            for modelid in range(10):
                    for inputid in range(10):
                        for MeanDelay_noround in np.arange(2, 3.01, 0.2):
                            MeanDelay = np.round(MeanDelay_noround,1)
                            for stdDelay_noround in np.arange(0, 1.01, 0.05):
                                stdDelay = np.round(stdDelay_noround,2)
                                param = {'NetworkType': networktype, 'n': cellcount, 'model_id': modelid, 'input_idx': inputid, 'MeanDelay': MeanDelay, 'stdDelay': stdDelay, 'counter': counter}
                                paramlist.append(param)
                                counter += 1



    for paradic in paramlist[3*12600+3150: 3*12600+3150+3150]:
        calTCA(paradic)