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
        
        
    
    def setTrainingSet(self, Shapes):
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

    def generate(self, n, stdFactor=5):
        if not self._isFitted:
            raise ValueError("The model has not been trained yet.")
        newShapes = []
        for i in range(n):
            b = np.random.normal(0.0, stdFactor * self._stds)
            xn = (self._meanShape + self._P.dot(b)).reshape((-1, 2))
            newShapes.append(xn)
        return newShapes
        

    @staticmethod
    def GeneralizedProcrustesAnalysis(Shapes, error = 1e-1, maxIter = 10):
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
