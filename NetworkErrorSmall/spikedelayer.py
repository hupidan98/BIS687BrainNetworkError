from neuron import h
import sys

class NetCon:
    """NetCon proxy that implements delays on a per spike basis from a normal distribution.

    Note: delays that go below 0 are treated as 0.
    """
    def __init__(self, source_ref_v, target, sec=None, delayer = None, nc0_need = True):
        if sec is None:
            sec = h.cas()
        try:
            if delayer == None:
                self._delayer = h.SpikeDelayer()
            else:
                self._delayer = delayer
        except AttributeError:
            print("Couldn't find SpikeDelayer; has spikedelayer.mod been compiled in the current folder?")
            sys.exit()
        if nc0_need:
            self._nc0 = h.NetCon(source_ref_v, self._delayer, sec=sec)
            self._nc0.delay = 0
        else:
            self._nc0 = None
        self._nc1 = h.NetCon(self._delayer, target)
        self._nc1.delay = 0
    
    def __getattr__(self, name):
        # default to nc0's properties and methods
        if name == "delay":
            raise AttributeError("delay not a valid attribute; use mean_delay and std_delay")
        elif name in ("mean_delay", "std_delay"):
            return getattr(self._delayer, name)
        elif name in ("weight",):
            return getattr(self._nc1, name)
        else:
            return getattr(self._nc0, name)
    
    def __setattr__(self, name, value):
        # default to nc0's properties and methods
        if name == "delay":
            raise AttributeError("delay not a valid attribute; use mean_delay and std_delay")
        elif name in ("_nc0", "_nc1", "_delayer"):
            super().__setattr__(name, value)
        elif name in ("mean_delay", "std_delay"):
            return setattr(self._delayer, name, value)
        elif name in ("weight",):
            return setattr(self._nc1, name, value)
        elif name in self._nc0:
            setattr(self._nc0, name, value)
        else:
            raise AttributeError(f"attribute {name} unknown")
    