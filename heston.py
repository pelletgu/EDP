import numpy as np
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
N = 40.;
I = 32.;
K = 32.;

##Payoff function

def payoff(s):
    payoff = np.fmax(E - s, 0)
    return payoff

##boundary conditions

def u_ymin(t, s):
    ## return K*exp(-r*t)-Smin in case of European option
    payoff = np.fmax(E - s, 0)
    return payoff

def u_ymax(t, s):
    ## return K*exp(-r*t)-Smin in case of European option
    return np.ones(I)*E #*np.exp(-r*t)

def u_smin(t, s):
    ## return K*exp(-r*t)-Smin in case of European option
    return E - s

def u_smax(t, s):
    return 0


## choose SCHEME value in: {'EE' , 'EI' , 'CN', 'BDF2', 'EI-AMER-NEWTON','CN-AMER-NEWTON', 'BDF2-AMER-NEWTON', 'BDF3-AMER-NEWTON'}

global SCHEME, CENTRAGE
SCHEME = 'BDF2-AMER-NEWTON'
CENTRAGE = 'CENTRE'

global A, q


##Define the generator of the scheme and the A matrix (same for each scheme as the focus is on the t de

def centred(s, y):
    global A
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)
    A = np.zeros(I**2 * K**2)
    A = A.reshape((I * K, I * K))

    Ak = np.zeros(I * I)
    Ak = Ak.reshape((I, I))
    Bk = np.zeros(I ** 2)
    Bk = Bk.reshape((I , I))
    Ck = np.zeros(I ** 2)
    Ck = Ck.reshape((I , I))

    for k in range(0, int(K)):
        for j in range(0, int(I)):
            Ak[j, j] = -gamma**2 * y[k] / (2 * h_y**2) + (alpha * (beta + y[k])) / (2 * h_y)
            Bk[j, j] = s[j]**2 * y[k] / (h_s**2) + gamma**2 * y[k] / (h_y**2) + r
            Ck[j, j] = -gamma**2 * y[k] / (2 * h_y**2) - (alpha * (beta + y[k])) / (2 * h_y)
            A[k*K + j, k*K + j] = Bk[j, j]
            if (k>0): A[k*K + j, (k-1)*K + j] = Ak[j, j]
            if (k<(int(I)-1)): A[(k*K + j, (k+1)*K + j)] = Ck[j, j]
        for j in range(1, int(I)):
            Ak[j, j-1] = -rho * gamma * y[k] * s[j] / (4 * h_s * h_y)
            Bk[j, j-1] = -s[j]**2 * y[k] / (2 * h_s**2) + r * s[j] / h_s
            Ck[j, j-1] = rho * gamma * y[k] * s[j] / (4 * h_s * h_y)
            A[k*K + j, k*K + j-1] = Bk[j, j-1]
            if (k>0): A[k*K + j, (k-1)*K + j-1] = Ak[j, j-1]
            if (k<(int(I)-1)): A[(k*K + j, (k+1)*K + j-1)] = Ck[j, j-1]
        for j in range(0, int(I-1)):
            Ak[j, j+1] = rho * gamma * y[k] * s[j] / (4 * h_s * h_y)
            Bk[j, j+1] = -s[j]**2 * y[k] / (2 * h_s**2) - r * s[j] / h_s
            Ck[j, j+1] = -rho * gamma * y[k] * s[j] / (4 * h_s * h_y)
            A[k*K + j, k*K + j+1] = Bk[j, j+1]
            if (k>0): A[k*K + j, (k-1)*K + j+1] = Ak[j,j+1]
            if (k<(int(I)-1)): A[(k*K + j, (k+1)*K + j+1)] = Ck[j, j+1]

##Vector of bound conditions

def qt(t, s, y):
    global q
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)

    A_1 = np.zeros(I ** 2)
    A_1 = A_1.reshape((I , I ))
    C_K = np.zeros(I ** 2)
    C_K = C_K.reshape((I , I ))
    for j in range(0, int(I)):
        A_1[j, j] = -gamma**2 * y[0] / (2 * h_y**2) + (alpha * (beta + y[0])) / (2 * h_y)
        C_K[j, j] = -gamma**2 * y[K-1] / (2 * h_y**2) - (alpha * (beta + y[K-1])) / (2 * h_y)
    for j in range(1, int(I)):
        A_1[j, j-1] = -rho * gamma * y[0] * s[j] / (4 * h_s * h_y)
        C_K[j, j-1] = rho * gamma * y[K-1] * s[j] / (4 * h_s * h_y)
    for j in range(0, int(I-1)):
        A_1[j, j+1] = rho * gamma * y[0] * s[j] / (4 * h_s * h_y)
        C_K[j, j+1] = -rho * gamma * y[K-1] * s[j] / (4 * h_s * h_y)
    A_1U = np.dot(A_1, u_ymin(t, s)) # could use boundary conditions with discount factor
    C_KU = np.dot(C_K, u_ymax(t, s)) # could use boundary conditions with discount factor

    q = np.zeros(I * K)

    for j in range(0, int(I)):
        q[j] = A_1U[j]
        q[(K-1)*I + j] = C_KU[j]
    for k in range(0, int(K)):
        a_k = -rho * gamma * y[k] * s[0] / (4 * h_s * h_y)
        b_k = -s[0]**2 * y[k] / (2 * h_s**2) + r * s[0] / h_s
        c_k = rho * gamma * y[k] * s[0] / (4 * h_s * h_y)
        q[k*I] = q[k*I] + (a_k + b_k + c_k) * u_smin(t, Smin)

    return q


## Newton method
def newton(B, b, g, x0, eps, kmax, s):
    k = 0
    x = np.copy(x0)
    err = eps + 1
    while (k < kmax and err > eps):
        k = k + 1
        F = np.fmin(np.dot(B, x) - b, x - g)
        Fp = np.eye(len(B))
        i = np.where((np.dot(B, x) - b) <=(x - g))
        Fp[i, :] = B[i, :]
        x = x - np.linalg.solve(Fp, F)
        err = np.linalg.norm(np.fmin(np.dot(B, x) - b , x - g), np.inf)

        #plt.plot(s, x[0:32], label='Scheme (y=0.25)')
        #plt.legend()
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


##Start of the real program

def main():
    start_time = time.time()

    # Mesh
    dt = float(T) / float(N)
    h_s = (Smax - Smin) / (I + 1)
    h_y = (Ymax - Ymin) / (K + 1)
    s = Smin + np.transpose(np.linspace(1, I, I)) * h_s
    y = Ymin + np.transpose(np.linspace(1, K, K)) * h_y

    y1_show = 0.0625
    y2_show = 0.25
    y3_show = 0.5
    idx1 = int(y1_show/h_y)
    idx2 = int(y2_show/h_y)
    idx3 = int(y3_show/h_y)
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

    plt.figure(1)
    plt.plot(s, payoff(s), label='payoff')

    switcher = {
        'CENTRE': centred(s, y),
    }
    switcher.get(CENTRAGE, "CENTRAGE not programmed")

    Id = np.eye(len(A))
    for n in range(0, int(N-1)):
        t = n * dt
        qt(t, s, y)
        plt.figure(1)
        switcher2 = {
            'BDF2-AMER-NEWTON': bdf2_amer_scheme(Id, dt, np.copy(P), t, s, y, Pold),
        }

        Pold2 = np.copy(Pold)
        Pold = np.copy(P)  ##Attention
        P = switcher2.get(SCHEME, "SCHEME not programmed")

##        plt.plot(s, P[(idx1*I):((idx1+1)*I)], label='Scheme (y=0.25)')
##        plt.legend()

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
    #plt.xlim(0, 200)
    #plt.ylim(0, 100)

    plt.plot(s, P[(idx1*I):((idx1+1)*I)], label='Scheme (y=0.0625)')
    plt.plot(s, P[(idx2*I):((idx2+1)*I)], label='Scheme (y=0.25)')
    plt.plot(s, P[(idx3*I):((idx3+1)*I)], label='Scheme (y=0.5)')

    #plt.plot(s, ft, label='deviation')

    # plt.figure(1)
    # plt.plot(s,BS(T,s),label='Closed formula')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
