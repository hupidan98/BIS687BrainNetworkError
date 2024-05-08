

            


import sys
import numpy as np
from numpy.random import default_rng
import random
import ConnectionMapGen
import pickle
from multiprocessing import Pool

import pickle
import RunningFFnetwork
import numpy as np


def run_py(file_counter):
    file_counter = int(file_counter)
                
    counter = 0
    for netname in ['FeedForward', 'SmallWorld', 'ScaleFree', 'Random']:
        for n in np.arange(30, 61, 10):
            for model_id in range(10):
                for input_idx in range(10):
                    if counter == file_counter:
                        print(counter, str(n) + '_' + str(model_id) + '_' + str(input_idx))
                        with open( './finished_counter/starting'+str(file_counter) + '.pkl', 'wb') as outp:
                            pickle.dump([], outp, pickle.HIGHEST_PROTOCOL)

                        with open( './params/'+ netname + '_' + 'point' + '_layercount' + str(n) +  '_model' + str(model_id) + '_input' + str(input_idx) + '_' + 'running_params.pkl', 'rb') as inp:
                            param_list_1 = pickle.load(inp)
                            with Pool(1) as p:
                                p.map(RunningFFnetwork.runingFFnetowrk_singalparam, (param_list_1))
                        with open( './finished_counter/'+str(file_counter) + '.pkl', 'wb') as outp:
                            pickle.dump([], outp, pickle.HIGHEST_PROTOCOL)
                    counter += 1
                        
    
            

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <number>")
    #     sys.exit(1)

    # number = int(sys.argv[1])
    starting = 775 + 400
    end = starting + 2
    for i in (range(starting, end)):
        run_py(i)
#     for i in ([813, 814, 828, 829, 843, 844, 845, 846, 847, 848]):
#         run_py(i)
#     run_py(99)

    