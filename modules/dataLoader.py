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

        muramModel = muram.MuramSnap(self.modelDir, timeStep)
        if self.verbose == True:
            print(f"run available props: {run1.available}")

        self.Temp = muramModel.T
        self.ne   = muramModel.ne

