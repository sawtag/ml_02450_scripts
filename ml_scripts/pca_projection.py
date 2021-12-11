#project data into pca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Formula for SVD -> X_tilda = U.Sig.V^T
#X_tilda: x - mean(x)
#U:
#V: eigenvector
#Sig: Singular values in diagonal = sqrt(eigenvals)

#Projection:
#Z = X_tilda . V.T
#Z: projected observations into PCA
#Xc = X - mean

#X_tilda = U.Sig.V^T
#Eigenvector
V = np.array([
    [-0.99,-0.13,-0.0],
    [-0.09,0.70,-0.71],
    [0.09, -0.70, -0.71]
              ])
#Original data, cols = variables; rows = observation
X =  pd.DataFrame(
    {"x1":np.array([3,4,0]),
     "x2":np.array([2,1,1]),
     "x3":np.array([1,2,2])
     }
    )
print("X:Original Data\n", X)
#Transformed by mean
X_tilda = X - X.mean(axis=0)
print("\nX_tilda:X-mean(X)\n", X_tilda)
#Project into PCA
Z = pd.DataFrame(X_tilda @ V.T)

#Labeling for easy reading
col = []
for c in range(1,Z.shape[1]+1):
    col.append("PCA"+str(c))

Z.columns = col

indes = []
for o in range(1,len(Z)+1):
    indes.append("obs"+str(o))
Z.index=indes
 
#Results
print("\nZ: Projected obs with PCA")
print(Z)

#Plot PCA of the data
plt.figure()
plt.plot(Z.PCA1,Z.PCA2, "o")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()