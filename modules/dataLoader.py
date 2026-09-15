try:
    from . import muram
except ImportError:
    import muram

class MURaMData:

    def __init__(self, modelDir, timeStep, modelName="MURaM_Model",
                 verbose=False):
        self.modelDir = modelDir
        self.timeStep = timeStep
        self.modelName = modelName
        self.verbose = verbose

    def loadModel(self, timeStep=None):
        if timeStep==None:
            # Load the default timestep
            timeStep = self.timeStep

        self.muramModel = muram.MuramSnap(self.modelDir, timeStep)
        if self.verbose == True:
            print(f"run available props: {self.muramModel.available}")

        self.Temp = self.muramModel.Temp.T
        self.ne   = self.muramModel.ne.T
        # Native x is vertical. Positive vx points toward the top observer.
        self.vz   = self.muramModel.vx.T
        self.rho  = self.muramModel.rho.T
        self.ds   = float(self.muramModel.Temp.dX[0])  # vertical cell length, cm
