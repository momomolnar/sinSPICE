import sys
import numpy as np
import scipy as sp

sys.path.append('modules/')

from dataLoader import MURaMData
from synthesizeLines import spectralLine, spectrum

modeldir = "/glade/campaign/hao/radmhd/Rempel/SSD_CHR/dyn_25x16Mm_32_pdmp_1_zxy_tvd/3D/"
muramModel = MURaMData(modeldir, 30000)
muramModel.loadModel()


ionNe8  = {"ion":"ne", "IonLevel":8, "lambda0":770, "numWaves":50, "dlambda":0.05}

spectrum_Ne8 = spectrum(ionNe8)
spectrum_Ne8.compute_2Dspectrum(muramModel.Temp,
                                muramModel.ne,
                                muramModel.vz,
                                )