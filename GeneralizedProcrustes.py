import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import os

# arr1 = np.loadtxt('Data/test1.csv', delimiter=',', skiprows=1)[:,1:]
# df = pd.DataFrame([], columns=['Patient', 'Landmark', 'X','Y'])

# for filename in os.listdir('Data'):

#     arr2 = np.loadtxt('Data/'+filename, delimiter=',', skiprows=1)[:,1:]

#     refData, mtx2, Disparity = procrustes(arr1, arr2)
#     tmp = pd.DataFrame(mtx2, columns=['X','Y'])
#     tmp['Patient'] = filename
#     tmp['Landmark'] = (np.arange(mtx2.shape[0])+1).astype(int)
#     df = pd.concat((df,tmp))
    

#     plt.scatter(mtx2[:,0], mtx2[:,1], label=f'{Disparity=}')

# df = df.set_index(['Patient','Landmark'])
# df.to_csv('Landmark_Aligned.csv')
# plt.title(f'{Disparity=}')
# plt.legend()
# plt.show()

def GeneralizedProcrustes(Shapes, error = 1e-1, maxIter = 10):
    # Shapes is a list of k shapes made of (L, 2) L 2D landmarks
    refShape = Shapes[0] # Arbitrary selected
    k = len(Shapes)
    newShapes = Shapes

    iter = 0
    while iter < maxIter:
        # Is it ok to reassign refShape every time?    
        meanShape = refShape
        for i in range(1, k):
            refShape, alignedShape, disparity = procrustes(refShape, newShapes[i])
            newShapes[i] = alignedShape
            meanShape += alignedShape 
        meanShape /= k 
        
        dist = np.linalg.norm(meanShape)
        print(f'Distance between mean shape and reference shape {dist=}.')
        iter+=1
        if dist < error:
            print(f'Tolerance achieved in {iter} iterations. Stopping the alignement procedure.')
            break
    newShapes[0] = refShape
    return newShapes

