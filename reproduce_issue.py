
import brian2 as b2
from brian2 import ms, mV, nsiemens, mm2, second
import cleo
from cleo import opto
import pytest

def test_duplicate_opsin_injection():
    ng = b2.NeuronGroup(10, """dv/dt = (-(v - -70*mV) + Iopto*Mohm)/ms : volt
                               Iopto : amp""", 
                        threshold="v>-50*mV", reset="v=-70*mV")
    sim = cleo.CLSimulator(b2.Network(ng))
    
    opsin1 = opto.vfchrimson_4s()
    sim.inject(opsin1, ng)
    
    opsin2 = opto.vfchrimson_4s()
    # This should ideally raise an error during inject, 
    # but currently it succeeds and run() fails.
    with pytest.raises(ValueError, match="already been injected"):
        sim.inject(opsin2, ng)

if __name__ == "__main__":
    test_duplicate_opsin_injection()
