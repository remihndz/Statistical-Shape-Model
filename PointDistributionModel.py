import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import svd
from sklearn.decomposition import PCA
from numpy.linalg import eig
import os
import networkx as nx



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
    def ToCCO(shape : list, CCOFileName : str, span : float=2.0, radiusCRA : float = 7e-3, nLandmarks : tuple=(None, None), **kwargs):
        """
        Write the vessel landmarks in shape as a new CCO file. 
        Assumes the first point in the list is the CR vessel's location while the remaining points are the superior and inferior temporal vessel.
        Both superior and inferior vessels are assumed to be composed of the same number of landmarks by default.
        If not, specify those numbers in landmarks.

        Argument:
            shape : list
                the list of 2D points (image landmarks)
            CCOFileName : str
                the output file name
            nLandmarks : tuple
                the number of landmark for each vessel in the shape
            span : float
                the axial length of the vessels (in cm)
        """

        CRVessel = np.pad(shape[0][:2], (0,1)).reshape((3,))
        # Various tree info in the header of the CCO file
        qProx = kwargs.get('qProx', 2.5000000000000001e-04) # 15 microliter/min in cm3/s, probably
        psiFactor = kwargs.get('psiFactor', 9.6040000000001100e-06) # Not sure what that does
        dp = kwargs.get('dp', 9.5453193691098149e+03) # Pressure drop from the single inlet to all outlet vessels, in Pa
        refPressure = kwargs.get('refPressure', 6.6661199999999999e+03) # Occular perfusion pressure, 50mmHg in Pa
        nTerms = 2

        if nLandmarks[0]:
            superiorVessel = shape[1:nLandmarks[0]+1]
            if nLandmarks[1]:
                inferiorVessel = shape[nLandmarks[0]+1 : nLandmarks[0]+nLandmarks[1]+1]
            else:
                inferiorVessel = shape[nLandmarks[0]+1:]
        else:
            nLandmarks = int(len(shape-1)/2)
            superiorVessel = shape[1:nLandmarks[0]+1]
            inferiorVessel = shape[nLandmarks[0]+1:]
        
        # Create the central retinal vessel (out of the plane) to link both vessel to a single inlet
        G = nx.DiGraph()
        G.add_node(0, position=np.array([CRVessel[0], CRVessel[1], -0.1])) # The inlet node
        G.add_node(1, position=CRVessel)
        G.add_edge(0, 1, radius = radiusCRA, key = 0)
        
        # Add the superior temporal vessel
        nProx = 1 # The key of the distal node of the CRVessel
        nodeKey = 2 # Available node key for G
        vesselKey = 1 # Available vessel key for G
        for x in superiorVessel:
            x = np.pad(x, (0,1)) # Position of the distal node
            G.add_node(nodeKey, position=x) 
            G.add_edge(nProx, nodeKey, radius = radiusCRA/2.0, key = vesselKey) # All vessel segments have the same width
            nProx = nodeKey
            nodeKey +=1
            vesselKey +=1

        # Add the inferior vessel
        nProx = 1 # Reset the proximal end to be the CRA
        for x in inferiorVessel:
            x = np.pad(x, (0,1)) # Position of the distal node
            G.add_node(nodeKey, position=x) 
            G.add_edge(nProx, nodeKey, radius = radiusCRA/2.0, key = vesselKey) # All vessel segments have the same width
            nProx = nodeKey
            nodeKey +=1
            vesselKey +=1
        
        with open(CCOFileName, 'w') as f:
            f.write('*Tree\n')
            x = G.nodes[0]['position']
            f.write(f"{x[0]} {x[1]} {x[2]} {qProx} {psiFactor} {dp} {nTerms} {refPressure} {G.number_of_nodes()} {radiusCRA} {1e-6}\n")

            f.write('\n*Vessels\n')
            f.write(f"{G.number_of_edges()}\n")
            for n1, n2, data in G.edges(data=True):
                xProx, xDist = G.nodes[n1]['position'], G.nodes[n2]['position']
                branchingMode = 3 if data['key']==0 else 2 # If the root, then bifurcates at distal end only, else deformable parent
                vesselFunction = 0 # Distribution vessel, bifurcates in the domain only
                f.write(f"{data['key']} {xProx[0]} {xProx[1]} {xProx[2]} {xDist[0]} {xDist[1]} {xDist[2]}")
                f.write(f"0.0 0.0 0.0 0.0 {branchingMode} {data['radius']} {qProx} 0.0 0.0 0.0 {vesselFunction} 0.0 0.0 0\n")

            f.write('\n*Connectivity')
            for n1, n2, data in G.edges(data=True):
                f.write('\n')

                # Sanity check
                parent = G.predecessors(n1)
                assert sum(1 for _ in parent) <= 1, f'Oops... Something went wrong, more than 1 parent was found {parent=}.'
                descendents = G.successors(n2) 
                assert sum(1 for _ in descendents) <= 2, f'Oops... Something went wrong, more than 2 descendents were found {descendents=}.' 
                # End sanity check 

                parent = G.predecessors(n1)
                descendents = G.successors(n2) 
                try: 
                    parentKey = next(parent)
                except:
                    parentKey = -1
                f.write(f"{data['key']} {parentKey}")                    
                
                for descendent in descendents:
                    f.write(f" {G[n2][descendent]['key']}")
            






            
        
        




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