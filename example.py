import sys
import numpy as np
import matplotlib.pyplot as pl
import cmcrameri.cm as cmc

import matplotlib.colors as colors

sys.path.append('modules/')

from dataLoader import MURaMData
from synthesizeLines import spectralLine, spectrum

modeldir = "/glade/campaign/hao/radmhd/Rempel/SSD_CHR_GPU_NewEOS/Chromosphere_ng_closed_9sec_1hr/3D/"
model_label = "QS_ng_closed"

workDir = "/glade/work/mmolnar/sinSPICE/"
tStart = 0
tEnd   = 39000
dTimeStep = 100
nTimeSteps = (tEnd - tStart) // dTimeStep + 1

downsample = 3

ionNe8  = {"ion":"ne", "IonLevel":8, "lambda0":770, "numWaves":100, "dlambda":0.0075}

muramModel = MURaMData(modeldir, tStart)

muramModel.loadModel()
nx = ny = muramModel.Temp[::downsample, ::downsample, :].shape[0]

I2D = np.zeros((nx, ny, ionNe8["numWaves"], nTimeSteps))

spectrum_Ne8 = spectrum(ionNe8, loadGTables=False,
                        GTablesFile="/glade/u/home/mmolnar/Projects/sinSPICE/G_Tmin_max_num_4_7.5_100_ne_5_12_100.npz")

for el in range(nTimeSteps):

    tStep = tStart + dTimeStep*el
    muramModel = MURaMData(modeldir, tStep)
    muramModel.loadModel()

    spectrum_Ne8.compute_2Dspectrum(muramModel.Temp[::downsample, ::downsample, :],
                                    muramModel.ne[::downsample, ::downsample, :],
                                    muramModel.vz[::downsample, ::downsample, :])
    I2D[:, :, :, el] = spectrum_Ne8.I2D.reshape(nx, ny, ionNe8["numWaves"])

    pl.plot(I2D[100, 100, :, el])
    pl.savefig(workDir +  model_label + f"_blahspec_{tStep:05}.png")
    pl.clf()

    # im1 = pl.imshow(I2D[:, :, 0, el],
    #                 cmap = cmc.batlow,
    #                 norm=colors.PowerNorm(gamma=0.5, vmin=0.3e-29, vmax=2.5e-28))
    # pl.colorbar(im1)
    # pl.title(f"Continuum 769 Å {el*9:03} sec")
    # pl.savefig(workDir + model_label +  f"_769AA_Icont_{tStep:05}.png")

    pl.clf()

    im2 = pl.imshow(np.sum(I2D[:, :, :, el], axis=2),
                    cmap=cmc.batlow,
                    norm=colors.PowerNorm(gamma=0.5, vmin=1e-22, vmax=6e-21))
    pl.colorbar(im2)
    pl.title(f"Ne VIII 770 Å {el * 9:03} sec")

    # pl.savefig(model_label +  f"_blahIspec_{el:03}.png")
    pl.savefig(workDir + model_label +  f"_NeVIII_770nm_Ispec_{tStep:05}.png")
    pl.clf()

np.savez(workDir + model_label + "_I2D_NeVIII_770AA.npz",
         I2D = I2D, wvlGrid = spectrum_Ne8.lambdaGrid)
