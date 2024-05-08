**************************************************************
Workflow：

1.Running ZGen_GeneratingParams.ipynb to generate parameters of neural networks. Parameter generated includes connection matrix of the neural network, and inital Cell spike time due to external stimulation following poisson process. For Each type of network (Feedforward, Smallword, ScaleFree), each have 4 different network size (30 - 60 cells, in Feedforward case, that is the cell count of a layer. Each will have generated 10 networks with the each sizes. Each will be run on 10 different input for 10 times. ConnectionMapGen.py is used to generate each network. The 

Also, in this notebook, parameter are tunned so that about each cell has about average 300 - 400 spikes, and some of cells has no activity. 

We can visualize the network structure, input to the netowrk, and output of the network of one simulation using ZDisplay_InputVSOutput.ipynb.
    
    
    
2. RunningFFNetwork.py is used to run simulation and save output to savedoutput folder, which have all simulation result. spikedelayer.py is used to introduce random delay between synapse, and is a python holder for mode file spikedelayer.mod.
    
runningSimulation.py is a python multiprocessing file that runs multiple RunningFFnetwork.py simulation at the same time.
        
SLMNetwork.sh is the Slurm script to run runingSimulation.py on HPC. I have seperate into 4 jobs, so there are SLMNetwork_(1-4).sh, crrosponding to runningSimulation_1(1-4).py, and RunningFFNetwork_(1-4).py. All the following jobs with nameing ends from 1-4 are following this convention to running on HPC easily.
  
  
        
3.Calulating Victor-Purpura and Van-Rossu Distance. First, VP and VR of every cell of every run is calculated using ZCal_Cal_VPVRevery.py (ZCal_Cal_VPVRevery_(1-4).py), and running on HPC with SLMCalVPVRevery_(1-4).sh. The calculated VP and VRs are saved in VP_every and VR_every folder. 

After calculation ZCal_VPVRDist_bylayerbystd.ipynb is the notebook that visualize the VP/VR distribution by each layer and each mean delay std. ZCal_Cal_aggVPVRusingEvery.ipynb is the notebook that calculating and visualizing the change of average VP and VR of differencet mean standard delay across layer. 

To see what is VP/VR distribution of a random input, refer to notebook, ZCal_VPVRRandomDistribution.ipynb.



4. Calculation of TCA if down by ZCal_Cal_TCA.py (ZCal_Cal_TCA_(1-3).py), and running on HPC by SLMTCA.sh (SLMTCA_(1-3).sh). The output of calculation is saved in folder TCAProcessing. 



5. Calculation of Spectral Density is down by ZCal_Cal_Spectral.py (ZCal_Cal_Spectral_(1-4).py, and running on HPC by SLMSpectral.sh (SLMSpectral_(1-4).sh). The output of calculation is saved in folder SpectralProcessing.

    
**************************************************************

FolderExplanation:

**************************************************************

FileExplanation：
    Cell.py: Cell class that will be used to running network. The Cell class has method add_custom_stimulus, which take a list of numbers representing a list of spike time of the cell.The default cell used in the simulation is PointCell. The BallandStick cell implemented the same way as in the NEURON tutorial. 
    ConnectionMapGen.py: contains methods that generate connection map of a neural network. The method will be used to generate parameter for running simulation. If there are 100 cells in the network, this will be a 100*100 matrixs, [i,j]entry = w means the connection weight w，if w > 0, this is a excititory connection with weight w, if w < 0, this is a inhibitory connection with weight -w. Method used are FeedForwardMapGen, ScaleFreeMapGen, SmallWorldMapGen.
    delaytest0.py:test whether spikedelayer.py works as expected. the spikedelayer.py is responsible of handleing random delay during synapses.
    Network.py: construct a neural network of given parameter.
    