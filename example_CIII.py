import sys

import cmcrameri.cm as cmc
import matplotlib.colors as colors
import matplotlib.pyplot as pl
import numpy as np

sys.path.append('modules/')

from utils import ProgressBar, parse_arguments
from dataLoader import MURaMData
from synthesizeLines import spectrum


# model_dir = "/glade/campaign/hao/radmhd/Rempel/SSD_CHR_GPU_NewEOS/Chromosphere_ng_open_25x25x25Mm_30min_2sec/3D/"
model_dir = "/glade/campaign/hao/radmhd/Rempel/ASD_2023/Run_Corona_1728x1024x1024_ASD_closed/3D/"
model_label = "AR_11158_10kG"

workDir = "/glade/work/mmolnar/sinSPICE/"
tStart = 90000
tEnd = 100000
dTimeStep = 2000
nTimeSteps = (tEnd - tStart) // dTimeStep + 1

downsample = 2
hydrogen_mass_fraction = 0.74  # Match the model EOS composition.
observer = "top"
nSpatialJobs = 120

# ion = {"ion": "c", "IonLevel": 3, "lambda0": 977., "numWaves": 200, "dlambda": 0.0045}
# ion  = {"ion":"c", "IonLevel":3, "lambda0":977., "numWaves":200, "dlambda":0.0045}


def format_bytes(num_bytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num_bytes < 1024 or unit == "TiB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


def print_cube_info(label, cube, sampled_cube):
    print(
        f"{label}: full shape={cube.shape}, dtype={cube.dtype}, "
        f"full logical size={format_bytes(cube.nbytes)}, "
        f"synthesized shape={sampled_cube.shape}, "
        f"synthesized logical size={format_bytes(sampled_cube.nbytes)}"
    )


args = parse_arguments()
ion = {"ion":args.ion,
       "IonLevel": args.ion_level,
       "lambda0": args.lambda0,
       "numWaves": args.numWaves,
       "dlambda": args.dLambda}

muramModel = MURaMData(model_dir,
                       tStart)

muramModel.loadModel()
print(f"Loaded MURaM timestep {tStart} from {model_dir}")
print_cube_info("Temp", muramModel.Temp, muramModel.Temp[::downsample, ::downsample, :])
print_cube_info("ne", muramModel.ne, muramModel.ne[::downsample, ::downsample, :])
print_cube_info("vz", muramModel.vz, muramModel.vz[::downsample, ::downsample, :])
nx, ny = muramModel.Temp[::downsample, ::downsample, :].shape[:2]

I2D = []

spectrum_Ne8 = spectrum(ion, loadGTables=False,
                        GTablesFile="/glade/u/home/mmolnar/Projects/sinSPICE/G_Tmin_max_num_4_7.5_100_ne_5_12_100.npz")


def compute_MURAM_tStep(modeldir, tStep):
    muramModel = MURaMData(modeldir, tStep)
    muramModel.loadModel()

    spectrum_Ne8.compute_2Dspectrum(muramModel.Temp[::downsample, ::downsample, :],
                                    muramModel.ne[::downsample, ::downsample, :],
                                    muramModel.vz[::downsample, ::downsample, :],
                                    rho=muramModel.rho[::downsample, ::downsample, :],
                                    ds=muramModel.ds,
                                    hydrogen_mass_fraction=hydrogen_mass_fraction,
                                    observer=observer,
                                    n_jobs=nSpatialJobs,
                                    progress_label=f"t={tStep:06d} domain chunks")
    return spectrum_Ne8.I2D.reshape(nx, ny, ion["numWaves"])


tSteps = [(tStart + el * dTimeStep) for el in range(nTimeSteps)]

with ProgressBar(len(tSteps), label="Completed timesteps") as timestep_progress:
    for tStep in tSteps:
        I2D.append(compute_MURAM_tStep(model_dir, tStep))
        timestep_progress.update()
I2D = np.asarray(I2D)

ion_label = str(ion["ion"]) + str(ion["IonLevel"])
lambda0 = ion["lambda0"]
output_file = (
    workDir
    + model_label
    + f"_I2D_{ion_label}"
    + f"_{lambda0}AA_t_{tStart}_{tEnd}_dwnsmpl_"
    + str(downsample)
    + ".npz"
)
print(f"Saving {output_file}")
np.savez(output_file,
         I2D=I2D,
         wvlGrid=spectrum_Ne8.lambdaGrid)

# for el in range(nTimeSteps):
#     # pl.plot(I2D[el, 100, 100, el])
#     # pl.savefig(workDir +  model_label + f"_blahspec_{tStep:05}.png")
#     # pl.clf()
#     #
#     # # im1 = pl.imshow(I2D[:, :, 0, el],
#     # #                 cmap = cmc.batlow,
#     # #                 norm=colors.PowerNorm(gamma=0.5, vmin=0.3e-29, vmax=2.5e-28))
#     # # pl.colorbar(im1)
#     # # pl.title(f"Continuum 769 Å {el*9:03} sec")
#     # # pl.savefig(workDir + model_label +  f"_769AA_Icont_{tStep:05}.png")
#     #
#     # pl.clf()
#
#     im2 = pl.imshow(np.sum(I2D[el, :, :, :], axis=2), cmap=cmc.batlow,
#                     norm=colors.PowerNorm(gamma=0.5, vmin=1e-26, vmax=4e-25))
#     pl.colorbar(im2)
#     pl.title(f"C III 977 Å {el * 9:03} sec")
#
#     # pl.savefig(model_label +  f"_blahIspec_{el:03}.png")
#     pl.savefig(workDir + model_label + f"CIII_977Å_Ispec_{el:05}.png")
#     pl.clf()
