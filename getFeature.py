from utils import *
from getFinger import getFinger
from getImage import readImage

class FGFF(getFinger):
    def __init__(self, M0, mFingertip):
        self.mFingertip = mFingertip
        self.M0 = M0

    def getFG(self):
        return [(y - self.M0[1], x - self.M0[0]) for (x, y) in self.mFingertip]
    
    


# #____________Hu moment_________________
# def m_pq(f, p, q):
#     """
#     Two-dimensional (p+q)th order moment of image f(x,y)
#     where p,q = 0, 1, 2, ...
#     """
#     m = 0
#     # Loop in f(x,y)
#     for x in range(0, len(f)):
#         for y in range(0, len(f[0])):
#             # +1 is used because if it wasn't, the first row and column would
#             # be ignored
#             m += ((x+1)**p)*((y+1)**q)*f[x][y]
#     return m


# def centroid(f):
#     """
#     Computes the centroid of image f(x,y)
#     """
#     m_00 = m_pq(f, 0, 0)
#     return [m_pq(f, 1, 0)/m_00, m_pq(f, 0 ,1)/m_00]


# def u_pq(f, p, q):
#     """
#     Centroid moment invariant to rotation.
#     This function is equivalent to the m_pq but translating the centre of image
#     f(x,y) to the centroid.
#     """
#     u = 0
#     centre = centroid(f)
#     for x in range(0, len(f)):
#         for y in range(0, len(f[0])):
#             u += ((x-centre[0]+1)**p)*((y-centre[1]+1)**q)*f[x][y]
#     return u


# def hu(f): # f = mContour
#     """
#     This function computes Hu's seven invariant moments.
#     """
#     u_00 = u_pq(f, 0, 0)

#     # Scale invariance is obtained by normalization.
#     # The normalized central moment is given below
#     eta = lambda f, p, q: u_pq(f, p, q)/(u_00**((p+q+2)/2))

#     # normalized central moments used to compute Hu's seven moments invariat
#     eta_20 = eta(f, 2, 0)
#     eta_02 = eta(f, 0, 2)
#     eta_11 = eta(f, 1, 1)
#     eta_12 = eta(f, 1, 2)
#     eta_21 = eta(f, 2, 1)
#     eta_30 = eta(f, 3, 0)
#     eta_03 = eta(f, 0, 3)

#     # Hu's moments are computed below
#     phi_1 = eta_20 + eta_02
#     phi_2 = 4*eta_11 + (eta_20-eta_02)**2
#     phi_3 = (eta_30 - 3*eta_12)**2 + (3*eta_21 - eta_03)**2
#     phi_4 = (eta_30 + eta_12)**2 + (eta_21 + eta_03)**2
#     phi_5 = (eta_30 - 3*eta_12)*(eta_30 + eta_12)*((eta_30+eta_12)**2 - 3*(eta_21+eta_03)**2) + (3*eta_21 - eta_03)*(eta_21 + eta_03)*(3*(eta_30 + eta_12) - (eta_21 + eta_03)**2)
#     phi_6 = (eta_20 - eta_02)*((eta_30 + eta_12)**2 - (eta_21 + eta_03)**2) + 4*eta_11*(eta_30 + eta_12)*(eta_21 + eta_03)
#     phi_7 = (3*eta_21 - eta_03)*(eta_30 + eta_12)*((eta_30 + eta_12)**2 - 3*(eta_21 + eta_03)**2) - (eta_30 - 3*eta_12)*(eta_21 + eta_03)*(3*(eta_30 + eta_12)**2 - (eta_21 + eta_03)**2)

#     a_1 = np.log10(np.abs(phi_2/(phi_1**2)))
#     a_2 = np.log10(np.abs(phi_3/(phi_1*phi_2)))
#     a_3 = np.log10(np.abs(phi_4/(phi_1*phi_2)))
#     a_4 = np.log10(np.abs(phi_5/(phi_3*phi_4)))
#     a_5 = np.log10(np.abs(phi_6/(phi_1*phi_3)))
#     a_6 = np.log10(np.abs(phi_7/phi_5))
#     return [a_1, a_2, a_3, a_4, a_5, a_6]

def hu_moment(image):
    # Calculate Moments 
    moments = cv2.moments(image) 
    # Calculate Hu Moments 
    phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7 = cv2.HuMoments(moments)
    a_1 = np.log10(np.abs(phi_2/(phi_1**2)))[0]
    a_2 = np.log10(np.abs(phi_3/(phi_1*phi_2)))[0]
    a_3 = np.log10(np.abs(phi_4/(phi_1*phi_2)))[0]
    a_4 = np.log10(np.abs(phi_5/(phi_3*phi_4)))[0]
    a_5 = np.log10(np.abs(phi_6/(phi_1*phi_3)))[0]
    a_6 = np.log10(np.abs(phi_7/phi_5))[0]
    return [a_1, a_2, a_3, a_4, a_5, a_6]

def distance(list1, list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5