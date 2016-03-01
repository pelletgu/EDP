import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

##Financial datas
global K, r, sigma, T, Smin, Smax
K = 100.;
sigma = 0.2;
r = 0.1;
T = 1.;
Smin = 0.;
Smax = 200.;

##Numerical data
global N, I
N = 10.;
I = 50.-1;


##Payoff function

def payoff(s):
    payoff = 1
    switcher3 = {
        1: np.fmax(K-s, 0),
    }
    return switcher3.get(payoff, "nothing")


##boundary conditions

def ul(t):
    ## return K*exp(-r*t)-Smin in case of European option
    return K*np.exp(-r*t) - Smin


def ur(t):
    return 0


## choose SCHEME value in: {'EE' , 'EI' , 'CN', 'BDF2'}

global SCHEME, CENTRAGE
SCHEME = 'CN'
CENTRAGE = 'CENTRE'

##Parametres de la fenetre graphique

global Xmin, Xmax, Ymin, Ymax
Xmin = Smin;
Xmax = Smax;
Ymin = -20;
Ymax = K;


##Formule de Black-Scholes - method

def BS(t, s):
    if t == 0:
        return np.fmax(K - s, 0)
    else:
        tmp = np.ones(len(s)) * K * np.exp(-r * t)
        i = np.where(s > 0)
        tau = (sigma ** 2) * t
        dm = (np.log(s[i] / K) + r * t - 0.5 * tau) / np.sqrt(tau)
        dp = (np.log(s[i] / K) + r * t + 0.5 * tau) / np.sqrt(tau)
        tmp[i] = K * np.exp(-r * t) * (norm.cdf(-dm)) - s[i] * norm.cdf(-dp)
        return tmp


global alpha, bet, A, q


def centred(s, h):
    global A, alpha, bet
    A = np.zeros(I * I)
    A = A.reshape((I, I))
    alpha = ((sigma ** 2) / 2) * ((s ** 2) / (h ** 2))
    bet = (r * s) / (2 * h)
    for i in range(0, int(I-1)): A[i, i] = 2 * alpha[i] + bet[i] + r
    for i in range(1, int(I-1)): A[i, i - 1] = -alpha[i]
    for i in range(0, int(I-2)): A[i, i + 1] = -alpha[i] - bet[i]


def qt(t):
    global q
    q = np.zeros(I)
    q[0] = (-alpha[0] + bet[0]) * ul(t)
    q[I - 1] = (-alpha[I - 1] - bet[I - 1]) * ur(t)
    return q


def ee_scheme(Id, dt, p):
    return (Id - dt * A) * p - dt * q

def ei_scheme(Id, dt, p,t):
    t1=t+dt
    qt(t1)
    return np.linalg.solve((Id + dt * A), (p - dt * q))

def cn_scheme(Id, dt, p, t):
    q0 = qt(t)
    q1 = qt(t+dt)
    return np.linalg.solve((Id + (dt/2) * A), np.dot(Id - (dt/2) * A,p)-dt*(q0+q1)/2)

def bdf2_scheme(Id,dt,p,pold,t):
    q1 = qt(t+dt)
    return np.linalg.solve((Id+(2*dt/3)*A),4*pold/3-p/3-(2*dt/3)*q1)

def descente(L, b):
    n = len(b)
    y = np.zeros(n)
    y[1] = b[1] / L[1, 1]
    for k in range(2, n):
        y[k] = (b[k] - L[k, k - 1] * y[k - 1]) / L[k, k]


def montee(U, y):
    n = len(y)
    x = np.zeros(n)
    x[n] = y[n]
    for k in range(n - 1, 1, -1):
        x[k] = (y[k] - U[k, k - 1] * x[k + 1])


##Start of the real program

def main():
    # Mesh
    dt = float(T) / float(N)
    h = (Smax - Smin) / (I + 1)
    s = Smin + np.transpose(np.linspace(1, I, I)) * h

    # CFL numbers
    cfl = (dt / (h ** 2)) * (sigma * Smax) ** 2
    Pold = payoff(s)
    P = payoff(s)
    plt.figure(1)
    plt.plot(s,P, label='Payoff')
    switcher = {
        'CENTRE': centred(s, h),
    }
    switcher.get(CENTRAGE, "CENTRAGE not programmed")

    Id = np.eye(len(A))

    for n in range(0, int(N) - 1):
        t = n * dt
        qt(t)
        switcher2 = {
            'EE': ee_scheme(Id, dt, P),
            'EI': ei_scheme(Id,dt,P,t),
            'CN': cn_scheme(Id,dt,P,t),
            'BDF2': bdf2_scheme(Id,dt,P,Pold,t)
        }
        Pold = P
        P = switcher2.get(SCHEME, "SCHEME not programmed")
    plt.figure(1)
    plt.plot(s, P,label='Scheme')


    plt.figure(1)
    plt.plot(s,BS(T,s),label='Closed formula')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
