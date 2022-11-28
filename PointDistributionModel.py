import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import svd
from sklearn.decomposition import PCA
from numpy.linalg import eig
import os



class PointDistributionModel:
    """
    A class defining a compact point distribution model based on PCA.
    
    Attributes:
    -----------
    listOfShapes : list
        the list of shapes used as training samples
    _data : array
        the listOfShapes in array format
    _P : array
        the matrix that spans the space of deformations
    _std : array
        the vector of standard deviation for each landmark
    _meanShape : array
        the mean shape resulting from generalized procrustes analysis
    nSample : int
        number of shapes in the training set
    nFeatures : int
        number of features in each shape
    
    Methods:
    --------
    fit()
        generate the model
    setTrainingSet(list Shapes)
        set the data attribute
    generate(int n)
        returns a list of n new shapes (arrays of size (nFeatures/2, 2))    
    GeneralizedProcrustesAnalysis(list Shapes)
        returns the data in the shape space
    to_cco(str filename)
        save the shape into a cco file
    """

    def __init__(self, Shapes):
        """
        Arguments:
        ----------
        Shapes : list
            the list of shapes used to train the model.
        """
        self.listOfShapes = Shapes # The original dataset
        self.nSamples = len(Shapes)
        self.nFeatures = Shapes[0].size

        self.setTrainingSet(Shapes) # Set the aligned data set
        self._isFitted = False
        
        
    
    def setTrainingSet(self, Shapes : list):
        self._data = np.zeros((self.nFeatures, self.nSamples))
        newShapes, self._meanShape = self.GeneralizedProcrustesAnalysis(Shapes, maxIter=5)
        for i, shape in enumerate(newShapes):
            self._data[:,i] = shape.ravel()
        
    def fit(self):
        pca = PCA(whiten=True)
        pca.fit(self._data.T)
        cov = pca.get_covariance()
        eigva, eigve = eig(cov)
        self._P = np.real(eigve).T
        self._stds = np.sqrt(np.abs(eigva))
        self._isFitted = True

    def generate(self, n : int, stdFactor=5) -> list:
        if not self._isFitted:
            raise ValueError("The model has not been trained yet.")
        newShapes = []
        for i in range(n):
            b = np.random.normal(0.0, stdFactor * self._stds)
            xn = (self._meanShape + self._P.dot(b)).reshape((-1, 2))
            newShapes.append(xn)
        return newShapes
        

    @staticmethod
    def GeneralizedProcrustesAnalysis(Shapes:list, error = 1e-1, maxIter = 10) -> list:
        # Shapes is a list of k shapes made of (L, 2) L 2D landmarks
        refShape = Shapes[0] # Arbitrary selected
        k = len(Shapes)
        newShapes = Shapes

        iter = 0
        print(f'Running generalized procruste analysis on the data.')
        while iter < maxIter:
            # Is it ok to reassign refShape every time?    
            meanShape = refShape
            for i in range(1, k):
                refShape, alignedShape, disparity = procrustes(refShape, newShapes[i])
                newShapes[i] = alignedShape
                meanShape += alignedShape 
            meanShape /= k 
            
            dist = np.linalg.norm(meanShape)
            print(f'\tDistance between mean shape and reference shape {dist=}.')
            iter+=1
            if dist < error:
                print('\tError reached tolerance, terminating GPA.')
                print(f'\tTolerance achieved in {iter} iterations. Stopping the alignement procedure.')
                break
        newShapes[0] = refShape
        return newShapes, meanShape.reshape((-1,))

    

    
    # @staticmethod
    # def to_cco(filename, CRA : point, vessels : list, vesselsType = 'DEFORMABLE_PARENT'):
    #     """
    #     Write vessels in cco format filename.

    #     Arguments:
    #     ----------
    #     filename: the name of the file to save the tree
    #     CRA: the location of the CRA, used as root for each vessel in vessels
    #     vessels: a list of list or lists of numpy arrays containing the points (excluding CRA) forming a vessel     
    #     """
                    
    #     ## TODO use a proper length and radius for the CRA 
    #     # Use the length of a vessel segment as an estimate of the CRA's length (because unit is unknown)

    #     if isinstance(vessels[0], float):
    #         vesselsList = [vessels]
    #     elif isinstance(vessels, list):
    #         vesselsList = vessels
    #     else:
    #         raise ValueError("Please provide a valide list of arrays or a single array.")
        
    #     lengthFirstVessel = np.linalg.norm(np.array(vesselsList[0][1])-np.array(vesselsList[0][0]))*3
    #     xperf  = np.array(CRA)-np.array([0.0,0.0,lengthFirstVessel])
    #     qProx  = 2.5e-4 # muL/min? Taken from the results of the DCCO algorithm
    #     psiFr  = 9.6e-6 # psiFactor
    #     dp     = 50*133.3 # Pa
    #     nTerms = len(vesselsList)

    #     refPressure = 18*133.3 # Pa
    #     rootRadius  = 7e-3    # in cm = 70microns
    #     variationTol = 1e-6

    #     branchingTypeDir = {'NO_BRANCHING':0,
    #                         'RIGID_PARENT':1,
    #                         'DEFORMABLE_PARENT':2,
    #                         'DISTAL_BRANCHING':3,
    #                         'ONLY_AT_PARENT_HOTSPOTS':4}
    #     branchingType = branchingTypeDir[vesselsType]


    #     nSegments = 1 # The CRA
    #     for vessel in vessels:
    #         nSegments += np.array(vesselsList).size

    #     nSegment += len(vesselsList)
            
    #     with open(filename, 'w') as f:

    #         f.write("*Tree\n")
    #         f.write(f"{xperf[0]} {xperf[1]} {xperf[2]} {qProx} {psiFr} {dp} {nTerms} {refPressure} 1 {rootRadius} {variationTol}\n")

    #         f.write("\n*Vessels\n")
    #         f.write(f"{nSegments}\n")

    #         vesselId = 0
    #         stage    = 0
    #         vesselConnectivity = dict()
    #         vesselConnectivity[0] = [-1]
            
            
    #         f.write(f"0 {xperf[0]} {xperf[1]} {xperf[2]} {CRA[0]} {CRA[1]} {CRA[2]} 0.0 0.0 0.0 0.0 3 {rootRadius} {qProx} 0.0 0.0 0.0 0 0.0 0.0 {stage}")
    #         stage+=1
    #         vesselId+=1
            
    #         for vessel in vesselsList:
    #             xProx, parent = np.array(CRA), 0
    #             vessel = np.array(vessel).ravel().reshape((-1,2))
    #             for i in range(np.array(vessel).ravel().size):
    #                 distalPoint = vessel[i, :]
    #                 vesselConnectivity[vesselId] = [parent]
                
        
