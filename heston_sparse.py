import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import pickle
import collections

##Comment sont calcules les erreur dans le cas americains
##Comment sont calcules les ordres (pas d'ordre 2 pour nous)
##Que faut il reprendre sur la partie stabilite exactement

##Financial datas
global E, r, T, Smin, Smax, Ymin, Ymax
E = 10.
r = 0.1
T = 0.25
Smin = 0.
Smax = 20.
Ymin = 0.
Ymax = 1.

global alpha, beta, gamma, rho
alpha = 5.
beta = 0.16
gamma = 0.9
rho = 0.1

##Numerical data
global N, I, K
N = 30.;
I = 255.;
K = 31.;

##Payoff function

def payoff(s):
    payoff = np.fmax(E - s, 0)
    return payoff

##boundary conditions

def u_ymin(t, s):
    ## return K*exp(-r*t)-Smin in case of European option
    return np.fmax(E - s, 0) * np.exp(-r*t)

def u_smin(t):
    ## return K*exp(-r*t)-Smin in case of European option
    return E*np.exp(-r*t) - Smin


## choose SCHEME value in: {'EE' , 'EI' , 'CN', 'BDF2', 'EI-AMER-NEWTON','CN-AMER-NEWTON', 'BDF2-AMER-NEWTON', 'BDF3-AMER-NEWTON'}

global SCHEME, CENTRAGE
SCHEME = 'BDF2-AMER-NEWTON'
CENTRAGE = 'CENTRE'

global A, q


##Define the generator of the scheme and the A matrix (same for each scheme as the focus is on the t de

def Ak_f(s, y, k):
    Ak = np.zeros(I**2)
    Ak = Ak.reshape((I, I))
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)

    for j in range(0, int(I)):
        Ak[j, j] = -gamma**2 * y[k] / (2 * h_y**2) + (alpha * (beta - y[k])) / (2 * h_y)
        # add pkA for Neumann boundary conditions on Smax
        if (j == int(I)-1):
            Ak[j, j] += rho * gamma * y[k] * s[int(I)-1] / (4 * h_s * h_y)
    for j in range(1, int(I)):
        Ak[j, j-1] = -rho * gamma * y[k] * s[j] / (4 * h_s * h_y)
    for j in range(0, int(I-1)):
        Ak[j, j+1] = rho * gamma * y[k] * s[j] / (4 * h_s * h_y)

    return (Ak)

def Bk_f(s, y, k):
    Bk = np.zeros(I**2)
    Bk = Bk.reshape((I, I))
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)

    for j in range(0, int(I)):
        Bk[j, j] = s[j]**2 * y[k] / (h_s**2) + gamma**2 * y[k] / (h_y**2) + r
        # add pkB for Neumann boundary conditions on Smax
        if (j == int(I)-1):
            Bk[j, j] += - s[int(I)-1]**2 * y[k] / (2 * h_s**2) - r * s[int(I)-1] / (2 * h_s)
    for j in range(1, int(I)):
        Bk[j, j-1] = -s[j]**2 * y[k] / (2 * h_s**2) + r * s[j] / (2 * h_s)
    for j in range(0, int(I-1)):
        Bk[j, j+1] = -s[j]**2 * y[k] / (2 * h_s**2) - r * s[j] / (2 * h_s)

    return (Bk)

def Ck_f(s, y, k):
    Ck = np.zeros(I**2)
    Ck = Ck.reshape((I, I))
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)

    for j in range(0, int(I)):
        Ck[j, j] = -gamma**2 * y[k] / (2 * h_y**2) - (alpha * (beta - y[k])) / (2 * h_y)
        # add pkB for Neumann boundary conditions on Smax
        if (j == int(I)-1):
            Ck[j, j] += - rho * gamma * y[k] * s[int(I)-1] / (4 * h_s * h_y)
    for j in range(1, int(I)):
        Ck[j, j-1] = rho * gamma * y[k] * s[j] / (4 * h_s * h_y)
    for j in range(0, int(I-1)):
        Ck[j, j+1] = -rho * gamma * y[k] * s[j] / (4 * h_s * h_y)

    return (Ck)

def centred(s, y):
    global A
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)
    A = np.zeros(I**2 * K**2)
    A = A.reshape((I * K, I * K))

    for k in range(0, int(K)):
        Ak = Ak_f(s, y, k)
        Bk = Bk_f(s, y, k)
        Ck = Ck_f(s, y, k)

        for j in range(0, int(I)):
            A[k*I + j, k*I + j] = Bk[j, j]
            if (k>0): A[k*I + j, (k-1)*I + j] = Ak[j, j]
            if (k<(int(K)-1)): A[(k*I + j, (k+1)*I + j)] = Ck[j, j]
            else: A[k*I + j, k*I + j] += Ck[j, j] # add CK to BK for Neumann boundary conditions on Ymax
        for j in range(1, int(I)):
            A[k*I + j, k*I + j-1] = Bk[j, j-1]
            if (k>0): A[k*I + j, (k-1)*I + j-1] = Ak[j, j-1]
            if (k<(int(K)-1)): A[(k*I + j, (k+1)*I + j-1)] = Ck[j, j-1]
            else: A[k*I + j, k*I + j-1] += Ck[j, j-1]# add CK to BK for Neumann boundary conditions on Ymax
        for j in range(0, int(I-1)):
            A[k*I + j, k*I + j+1] = Bk[j, j+1]
            if (k>0): A[k*I + j, (k-1)*I + j+1] = Ak[j,j+1]
            if (k<(int(K)-1)): A[(k*I + j, (k+1)*I + j+1)] = Ck[j, j+1]
            else: A[k*I + j, k*I + j+1] += Ck[j, j+1] # add CK to BK for Neumann boundary conditions on Ymax

    A = dia_matrix(A)

##Vector of bound conditions

def qt(t, s, y):
    global q
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)

    A_1 = np.zeros(I ** 2)
    A_1 = A_1.reshape((I, I))
    for j in range(0, int(I)):
        A_1[j, j] = -gamma**2 * y[0] / (2 * h_y**2) + (alpha * (beta - y[0])) / (2 * h_y)
    for j in range(1, int(I)):
        A_1[j, j-1] = -rho * gamma * y[0] * s[j] / (4 * h_s * h_y)
    for j in range(0, int(I-1)):
        A_1[j, j+1] = rho * gamma * y[0] * s[j] / (4 * h_s * h_y)
    A_1U = np.dot(A_1, u_ymin(t, s))
    A_1U[0] += -rho * gamma * y[0] * Smin / (4 * h_s * h_y) * u_smin(t)# Dirichlet condition for Smin

    q = np.zeros(I * K)
    for j in range(0, int(I)):
        q[j] = A_1U[j]
    for k in range(0, int(K)):
        a_k = -rho * gamma * y[k] * s[0] / (4 * h_s * h_y)
        b_k = -s[0]**2 * y[k] / (2 * h_s**2) + r * s[0] / (2 * h_s)
        c_k = rho * gamma * y[k] * s[0] / (4 * h_s * h_y)
        q[k*I] += (a_k + b_k + c_k) * u_smin(t)

    return q

## Newton method
def newton(B, b, g, x0, eps, kmax, s):
    k = 0
    x = np.copy(x0)
    x1 = np.copy(x0)
    err = eps + 1
    B1 = np.copy(B.toarray())
    while (k < kmax and err > eps):
        k = k + 1
        F = np.fmin(B.dot(x) - b, x - g)
        Fp = csr_matrix(B, copy=True)

        """
        #############
        F1 = np.fmin(np.dot(B1, x1) - b, x1 - g)
        Fp1 = np.eye(len(B1))
        i1 = np.where((np.dot(B1, x1) - b) <= (x1 - g))
        Fp1[i1, :] = B1[i1, :]
        x1 = x1 - np.linalg.solve(Fp1, F1)
        #############
        """

        i = np.where((B.dot(x) - b) > (x - g))
        for i_row in i[0][:]:
            csr_row_set_nz_to_val(Fp, i_row) # set row to identity
        x = x - sparse.linalg.spsolve(Fp, F)
        #x = x - np.linalg.solve(Fp, F)
        err = np.linalg.norm(np.fmin(B.dot(x) - b , x - g), np.inf)
        err1 = np.linalg.norm(np.fmin(np.dot(B1, x1) - b, x1 - g), np.inf)

        #txt = "Scheme (k=" + str(k) + ")"
        #plt.plot(s, x[992:1024], label=txt )
        #plt.show()
    return x


def bdf2_amer_scheme(Id, dt, P, t, s, y, Pold):
    q1 = np.copy( qt(t + dt, s, y) )
    b = 4 * P / 3 - Pold / 3 - (2 * dt / 3) * q1
    x0 = np.copy(P)
    B = (Id + (2 * dt / 3) * A)
    eps = 10 ** (-10)
    kmax = N + 1

    g = np.zeros(I * K)
    for k in range(0, int(K)):
        g[(k*I):((k+1)*I)] = payoff(s)

    P = newton(B, b, g, x0, eps, kmax, s)
    return P


def csr_row_set_nz_to_val(csr, row):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = 0
    idxdiag = np.where(csr.indices[csr.indptr[row]:csr.indptr[row+1]] == row)[0][0] + csr.indptr[row]
    csr.data[idxdiag] = 1

##Start of the real program

def main():


    start_time = time.time()

    # Mesh
    dt = float(T) / float(N)
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)
    s = Smin + np.transpose(np.linspace(1, I, I)) * h_s
    y = Ymin + np.transpose(np.linspace(1, K, K)) * h_y

    # CFL numbers
#    cfl = ((dt / (h_s ** 2)) * (sigma * Smax) ** 2) / 2
#    print 'Notre condition CFL {!r}.'.format(cfl)

    Pold2 = np.zeros(I * K)
    Pold = np.zeros(I * K)
    P = np.zeros(I * K)
    i = np.where(s<=K)
    for k in range(0, int(K)):
        Pold2[(k*I):((k+1)*I)] = payoff(s)
        Pold[(k*I):((k+1)*I)] = payoff(s)
        P[(k*I):((k+1)*I)] = payoff(s)

    switcher = {
        'CENTRE': centred(s, y),
    }
    switcher.get(CENTRAGE, "CENTRAGE not programmed")

    Id = sparse.eye(I*K)
    for n in range(0, int(N-1)):
        t = n * dt
        qt(t, s, y)
        switcher2 = {
            'BDF2-AMER-NEWTON': bdf2_amer_scheme(Id, dt, np.copy(P), t, s, y, Pold),
        }

        Pold2 = np.copy(Pold)
        Pold = np.copy(P)  ##Attention
        P = switcher2.get(SCHEME, "SCHEME not programmed")

        #plt.figure(1)
        #plt.plot(s, P[(idx1*I):((idx1+1)*I)], label='Scheme (y=0.25)')
        #plt.legend()

    """
    vt = v(t,P,s)
    ref = np.zeros(len(P))
    for i in range(0, (len(s) - 1)): ref[i] = vt[i]

    bla = np.linspace(80, 110, 2.5)
    condition = np.mod(s, 2.5) == 0
    j = np.where((condition == True) * (s >= 10) * (s <= 150))
    vec = P[j] - ref[j]

    errorinf = np.linalg.norm(vec, np.inf)
    error1 = np.linalg.norm(vec, 1)
    error2 = np.linalg.norm(vec, 2)

    print 'Norme un de l erreur {!r}'.format(error1 / 13)
    print 'Norme deux de l erreur {!r}'.format(error2 / np.sqrt(13))
    print 'Norme infini de l erreur {!r}'.format(errorinf)
    """

    print time.time() - start_time

    plt.figure(1)
    plt.axis([5,15,0,5])
    #plt.xlim(0, 200)
    #plt.ylim(0, 100)

    #plt.plot(s, ft, label='deviation')

    # plt.figure(1)
    # plt.plot(s,BS(T,s),label='Closed formula')

    plt.figure(1)
    plt.plot(s, payoff(s), label='Payoff')

    y_plot = np.array([0.0625, 0.25])
    s_plot = np.array([8, 9, 10, 11, 12], int)
    for k in range(0, len(y_plot)):
        idx_y = np.fmax((int)(y_plot[k]/h_y), 1)
        txt = "BDF-2 Scheme (y=" + str(idx_y*h_y) + ")"
        plt.plot(s, P[((idx_y-1)*I):idx_y*I], label=txt)
        print "### y=" + str(idx_y*h_y)
        for j in range(0, len(s_plot)):
            idx_s = np.fmax((int)(s_plot[j]/h_s), 1)
            print "P(S=" + str(idx_s*h_s) + ")=" + str(P[(idx_y-1)*I+idx_s-1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

