import cv2
import numpy as np
import matplotlib.pyplot as plt

def computeSVD(mat):   
    m, n = mat.shape

    T1 = np.dot(mat, mat.transpose())
    T2 = np.dot(mat.transpose(), mat) 
    #print("T1 = ", T1)


    ev1, U = np.linalg.eig(T1)
    ev2, V = np.linalg.eig(T2)


    #sort the eigen values and vectors
    idx1 = np.flip(np.argsort(ev1))
    ev1 = ev1[idx1]
    U = U[:, idx1]

    idx2 = np.flip(np.argsort(ev2))
    ev2 = ev2[idx2]
    V = V[:, idx2]

    E = np.zeros([m, n])

    var = np.minimum(m, n)

    for j in range(var):
        E[j,j] = np.abs(np.sqrt(ev1[j]))  
    
    # verify_term =  np.matrix.round(np.dot(V, np.dot(E.transpose(), np.dot(E, V.transpose()))), 0)
    # verify =  (T2[0,:] == verify_term[0,:])

    # if np.any(verify) == False:
    #     idx = np.where(verify == True)
    #     V[:, idx] = -V[:, idx]
    
    # verify_term =  np.matrix.round(np.dot(U, np.dot(E, np.dot(E.transpose(), U.transpose()))), 0)
    # verify =  (T1[0,:] == -verify_term[0,:])
    # if np.any(verify) == True:
    #     idx = np.where(verify == True)
    #     U[:, idx] = -U[:, idx]

    return U, E, V
    



def testSVD(A):
    mat = A
    #print("Original mat = ", mat)
    u, e, v = computeSVD(mat)
    U,ev, V = np.linalg.svd(A)
    print("Restoring matrix, ", np.dot(u, np.dot(e,v.transpose())))



def computeHomography(set1, set2):

    if (len(set1) < 4) or (len(set2) < 4):
        print("Need atleast four points to compute SVD.")
        return 0

    x = set1[:, 0]
    y = set1[:, 1]
    xp = set2[:, 0]
    yp = set2[:,1]

    nrows = 8
    ncols = 9
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    print("Computing Homography matrix for ")
    print(A)
    U, E, V = computeSVD(A)
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    print("the Homography matrix is")
    print(H)
    #print(cv2.findHomography(set1, set2))
    return H



def main():
    print("Solving question 2 ...")
    set1 = np.array([[5, 5], [150, 5], [150, 150], [5, 150]])
    set2 = np.array([[100, 100], [200, 80], [220, 80], [100, 200]])
    #testSVD()
    computeHomography(set1, set2)


if __name__ == '__main__':
    main()





