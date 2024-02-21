from neuron import h
from neuron.units import mV, ms, um

# Cell class that will be used to running network.
class Cell:
    def __init__(self, gid, x, y, z, theta):
        self._gid = gid
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z = 0
        h.define_shape()
        self._rotate_z(theta)
        self._set_position(x, y, z)

        # everything below here in this method is NEW
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

        self._ncs = []
        self._netstims = []

        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)
        
        self._setup_syn()
        self._setup_stim()
        self._setup_out()

    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)
    
    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )
        self.x, self.y, self.z = x, y, z
    
    # method for making cell spike at given list of time.
    def add_custom_stimulus(self, stim_times: list) -> None:
        # add synapse
        # syn = h.ExpSyn(self.soma(0.5))
        # syn.tau = tau
        # syn.e = reversal_potential * mV
        # syn_current = h.Vector().record(syn._ref_i)
        # self.syns.append(syn)
        # self.syn_currents.append(syn_current)
        netstims = [h.NetStim() for _ in stim_times] 
        for netstim, stim_time in zip(netstims, stim_times):
            netstim.number = 1
            netstim.start = stim_time
            netcon = h.NetCon(netstim, self.pos_syn)
            netcon.weight[0] = self.stim_w
            netcon.delay = 0 * ms
            self._ncs.append(netcon)
            self._netstims.append(netstim)

#  Default class                
class BallAndStick(Cell):
    name = "BallAndStick"
    stim_w = 0.0017

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert("pas")
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65  # Leak reversal potential mV
    
    def _setup_syn(self):
        # NEW: the synapse
        self.pos_syn = h.ExpSyn(self.dend(0.5))
        self.pos_syn.tau = 2 * ms
        self.pos_syn.e = 0 * mV
        
        self.neg_syn = h.ExpSyn(self.dend(0.5))
        self.neg_syn.tau = 7.5 * ms
        self.neg_syn.e = -80 * mV
    
    def _setup_stim(self):
        self.pos_stim = h.NetStim()
        self.pos_stim.number = 0
        
        self._posnc = h.NetCon(self.pos_stim, self.pos_syn)
        self._posnc.delay = 0
        self._posnc.weight[0] = BallAndStick.stim_w
        
        self.neg_stim = h.NetStim()
        self.neg_stim.number = 0
        
        self._negnc = h.NetCon(self.neg_stim, self.neg_syn)
        self._negnc.delay = 1
        self._negnc.weight[0] = BallAndStick.stim_w
    
    def _setup_out(self):
        self.out = self.soma(0.5)
        self.out_sec = self.soma

    def _rotate_z(self, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))
                
                

class PointCell(Cell):
    name = 'Pointcell'
    stim_w = 0.000177
    
    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.soma.L = self.soma.diam = 10

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV
    
    def _setup_syn(self):
        # NEW: the synapse
        self.pos_syn = h.ExpSyn(self.soma(0.5))
        self.pos_syn.tau = 5 * ms
        self.pos_syn.e = 0 * mV
        
        self.neg_syn = h.ExpSyn(self.soma(0.5))
        self.neg_syn.tau = 7.5 * ms
        self.neg_syn.e = -80 * mV
    
    def _setup_stim(self):
        self.pos_stim = h.NetStim()
        self.pos_stim.number = 0
        
        self._posnc = h.NetCon(self.pos_stim, self.pos_syn)
        self._posnc.delay = 0
        self._posnc.weight[0] = PointCell.stim_w
        
        self.neg_stim = h.NetStim()
        self.neg_stim.number = 0
        
        self._negnc = h.NetCon(self.neg_stim, self.neg_syn)
        self._negnc.delay = 1
        self._negnc.weight[0] = PointCell.stim_w

    def _setup_out(self):
        self.out = self.soma(0.5)
        self.out_sec = self.soma
    
    def _rotate_z(self, theta):
        pass
    
    