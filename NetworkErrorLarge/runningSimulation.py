

            


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
import os


def run_py(file_counter):
    file_counter = int(file_counter)
                
    counter = 0
    for netname in ['SmallWorld', 'ScaleFree','FeedForward']:
        for n in np.arange(200, 201, 10):
            for model_id in range(10):
                for input_idx in range(10):
                    if counter == file_counter:
                        print(counter, str(n) + '_' + str(model_id) + '_' + str(input_idx))
                        
                        with open('./counter/runningSimulation.txt', 'a') as f:
                            f.write(f'Started job index: {counter}\n')

                        with open( './params/'+ netname + '_' + 'point' + '_layercount' + str(n) +  '_model' + str(model_id) + '_input' + str(input_idx) + '_' + 'running_params.pkl', 'rb') as inp:
                            param_list_1 = pickle.load(inp)
                            with Pool() as p:
                                p.map(RunningFFnetwork.runingFFnetowrk_singalparam, (param_list_1))
                                
                        with open('./counter/runningSimulation.txt', 'a') as f:
                            f.write(f'Completed job index: {counter}\n')
                    counter += 1
                        
    
def check_completed_jobs():
    try:
        with open('./counter/runningSimulation.txt', 'r') as f:
            lines = f.readlines()
            completed_jobs = [line.strip() for line in lines if 'Completed job index' in line]
            print('Completed Jobs:', completed_jobs)
    except FileNotFoundError:
        print('No job has been completed yet.')
        

if __name__ == '__main__':
    if not os.path.exists('./counter'):
        os.makedirs('./counter')

    # starting = 489
    # end = 1000000
    # for i in (range(starting, end)):
    #     run_py(i)
    run_py(259)

    
    check_completed_jobs()