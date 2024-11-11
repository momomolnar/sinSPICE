import sys
import numpy as np
import scipy as sp

sys.path.append('modules/')

from dataLoader import MURaMData
from synthesizeLines import spectralLine, spectrum

modeldir = "/glade/campaign/hao/radmhd/Rempel/SSD_CHR/dyn_25x16Mm_32_pdmp_1_zxy_tvd/3D/"
muramModel = MURaMData(modeldir, 30000)
muramModel.loadModel()


ionNe8  = {"ion":"ne", "IonLevel":8, "lambda0":770, "numWaves":100, "dlambda":0.025}

spectrum_Ne8 = spectrum(ionNe8, loadGTables=True,
                        GTablesFile="/glade/u/home/mmolnar/Projects/sinSPICE/G_Tmin_max_num_4_7.5_100_ne_5_12_100.npz")
spectrum_Ne8.compute_2Dspectrum(muramModel.Temp,
                                muramModel.ne,
                                muramModel.vz,
                                )