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
global K, r, sigma, T, Smin, Smax,H, c0
K = 100.
sigma = 0.2
r = 0.1
T = 1.
Smin = 0.
Smax = 200.
H=0.
c0 = 30.

##Numerical data
global N, I
N = 40.;
I = 39.;


##Payoff function

def payoff(s):
    payoff = 1
    switcher3 = {
        1: np.fmax(K - s, 0),
    }
    return switcher3.get(payoff, "nothing")

def conditionsinit(s):
    if (s < K):
        return K - s
    else:
        return 0


def payoffbis_m2(s):
    ct  = C_m2(0,10**(-10))
    xt = xs(0)
    return payoff(xt)-ct*np.arctan((s-xt)/ct)


##boundary conditions

def ul(t):
    ## return K*exp(-r*t)-Smin in case of European option
    return K * np.exp(-r * t) - Smin


def ur(t):
    return 0


## choose SCHEME value in: {'EE' , 'EI' , 'CN', 'BDF2', 'EI-AMER-NEWTON','CN-AMER-NEWTON', 'BDF2-AMER-NEWTON', 'BDF3-AMER-NEWTON'}

global SCHEME, CENTRAGE
SCHEME = 'EI-AMER-NEWTON'
CENTRAGE = 'CENTRE'

global alpha, bet, A, q, ft


##Define the generator of the scheme and the A matrix (same for each scheme as the focus is on the t de

def centred(s, h):
    global A, alpha, bet
    A = np.zeros(I * I)
    A = A.reshape((I, I))
    alpha = ((sigma ** 2) / 2) * ((s ** 2) / (h ** 2))
    bet = (r * s) / (2 * h)
    for i in range(0, int(I)): A[i, i] = 2 * alpha[i] + r
    for i in range(1, int(I)): A[i, i - 1] = -alpha[i] + bet[i]
    for i in range(0, int(I - 1)): A[i, i + 1] = -alpha[i] - bet[i]

##Vector of bound conditions

def qt(t):
    global q
    q = np.zeros(I)
    q[0] = (-alpha[0] + bet[0]) * ul(t)
    q[I - 1] = (-alpha[I - 1] - bet[I - 1]) * ur(t)
    return q

##Function with allow us to compute f

def xs(t):
    return K-c0*t

def C(t):
    x = xs(t)
    bla = payoff(x)
    if bla !=0:
        return (1./float(Smax-x) - 1./float(payoff(x)))**(-1)
    else:
        return 0

def At(t,x):
    ct = C(t)
    xt = xs(t)
    if ct!=0:
        return np.ones(len(x))-(x-xt)/ct
    else:
        return np.zeros(I)

def C_m2(t, eps):
    theta0=1
    theta=1
    xt = xs(t)
    b = payoff(xt)
    a = Smax - xt
    error = eps +1
    while (np.abs(error) > eps):
        theta0 = theta
        theta = theta0-(b*theta0-np.arctan(a*theta0))/(b-a/(1+(a*theta0)**2))
        error = theta - theta0
    if theta!=0:
        return 1./theta
    else:
        return 0

def dC_m2(t, eps):
    xt = xs(t)
    b = payoff(xt)
    a = Smax - xt
    ct = C_m2(t,eps)
    q = 1+(a/ct)**2
    return ct*c0*(q-1)/(q*b-a)


##Compute the f terms at each step t

# f du Model 1 : f(t,x) = min(v_t - lambda^2/2 x^2 v_xx - r x v_x + rv, v - g(x))
def f(t,x,s):
    global ft
    ft = np.zeros(I)
    i = np.where(s < xs(t))
    at = At(t,s)
    xt = xs(t)
    ct = C(t)

    if payoff(xt) != 0:
        dct = (c0/(Smax-xt)**2 - c0/(payoff(xt)**2))*ct**2
    else:
        dct = 0

    lambd = ((sigma ** 2) / 2) * (s ** 2)
    rx = (r * s)
    for j in range (0,len(ft)):
        if np.any(i[0][:] == j):
            vt = 0
            vx = -1
            vxx = 0
            v = payoff(s[j])
            lhs = vt - lambd[j] * vxx - rx[j] * vx + r * v
            rhs = v - payoff(s[j])
            ft[j]= np.fmin(lhs, rhs)
        else:
            vt = c0 - c0/at[j] - ((c0*ct - dct * (s[j] - xt)) * (s[j] - xt)) / (ct**2 * at[j]**2)
            vx = -1. / at[j]**2
            vxx = (-2./ct)/at[j]**3
            v = payoff(xt) - (s[j] - xt) / at[j]
            lhs = vt - lambd[j] * vxx - rx[j] * vx + r * v
            rhs = v - payoff(s[j])
            ft[j]= np.fmin(lhs, rhs)
    return ft

# f du Model 2 : f(t,x) = min(v_t - lambda^2/2 x^2 v_xx - r x v_x + rv, v - g(x))
def f_m2(t,x,s):
    global ft
    ft = np.zeros(I)
    i = np.where(s<xs(t))
    xt = xs(t)
    eps = 10**(-10)
    ct = C_m2(t,eps)
    dct = dC_m2(t,eps)
    lambd = ((sigma ** 2) / 2) * (s ** 2)
    rx = (r * s)
    for j in range (0,len(ft)):
        if np.any(i[0][:] == j):
            vt = 0
            vx = -1
            vxx = 0
            v = payoff(s[j])
            lhs = vt - lambd[j] * vxx - rx[j] * vx + r * v
            rhs = v - payoff(s[j])
            ft[j]= np.fmin(lhs, rhs)
        else:
            vt = c0 - dct * np.arctan((s[j]-xt)/ct) - (c0*ct-dct*(s[j]-xt))/(ct*(1+((s[j]-xt)/ct)**2))
            vx = -1./(1+((s[j]-xt)/ct)**2)
            vxx = ((2./ct) * (s[j]-xt)/ct)/(1+((s[j]-xt)/ct)**2)**2
            v = payoff(xt)-ct * np.arctan((s[j]-xt)/ct)
            lhs = vt - lambd[j] * vxx - rx[j] * vx + r * v
            rhs = v - payoff(s[j])
            ft[j]= np.fmin(lhs, rhs)
    return ft


##We implement now the target function

def v(t,x,s):
    v = np.zeros(I)
    i = np.where(s<xs(t))
    at = At(t,s)
    xt = xs(t)
    for j in range (0,len(s)):
        if np.any(i[0][:] == j):
            v[j] = payoff(s[j])
        else:
            v[j]= payoff(xt) - (s[j]-xt)/(at[j])
    return v

def v_m2(t,x,s):
    v = np.zeros(I)
    i = np.where(s<xs(t))
    xt = xs(t)
    eps = 10**(-10)
    if t==0:
        ct = 0
    else:
        ct = C_m2(t,eps)

    for j in range (0,len(s)):
       if np.any(i[0][:] == j):
            v[j] = payoff(s[j])
       else:
            if ct!=0:
                v[j]= payoff(xt) - ct*np.arctan((s[j]-xt)/ct)
            else:
                v[j]= payoff(xt) - (s[j]-xt)
    return v

## Newton method
def newton(B, b, g, x0, eps, kmax):
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
    return x


def ei_amer_scheme(Id, dt, P, t, s, n, Pold,f):
    B = Id + dt * A
    b = np.copy(P) - f
    x0 = np.copy(P)
    g = payoff(s) + f
    eps = (10) ** (-10)
    kmax = N + 1;
    P = newton(B, b, g, x0, eps, kmax);
    return P

def cn1_amer_scheme(Id, dt, P, t, s, n, Pold, f):
    q0 = np.copy(qt(t))
    q1 = np.copy(qt(t + dt))
    b = np.dot(Id - (dt / 2) * A, P) - dt * (q0 + q1) / 2
    g = payoff(s)
    x0 = np.copy(P)
    B = Id + (dt / 2) * A
    eps = 10**(-10)
    kmax = N + 1
    P = newton(B, b, g, x0, eps, kmax)
    return P

def cn2_amer_scheme(Id, dt, P, t, s, n, Pold, f):
    q0 = np.copy(qt(t))
    q1 = np.copy(qt(t + dt))
    b = np.dot(Id - (dt / 2) * A, P) - dt * (q0 + q1) / 2
    g = payoff(s)
    x0 = np.copy(P)
    B = Id + (dt / 2) * A
    eps = 10**(-10)
    kmax = N + 1
    P = newton(B, b, P, x0, eps, kmax)
    return P


def bdf3_amer_scheme(Id, dt, P, t, s, n, Pold, Pold2, f):
    q1 = np.copy( qt(t + dt) )
    b = 18 * P - 9 * Pold + 2 * Pold2 - (6 * dt / 3) * q1
    x0 = np.copy(P)
    B = (11 * Id + (6 * dt) * A)
    eps = 10 ** (-10)
    kmax = N + 1
    g = payoff(s)
    P = newton(B, b, g, x0, eps, kmax)
    return P


def bdf2_amer_scheme(Id, dt, P, t, s, n, Pold, f):
    q1 = np.copy( qt(t + dt) )
    b = 4 * P / 3 - Pold / 3 - (2 * dt / 3) * (q1 - f)
    x0 = np.copy(P)
    B = (Id + (2 * dt / 3) * A)
    eps = 10 ** (-10)
    kmax = N + 1
    g = payoff(s) + f
    P = newton(B, b, g, x0, eps, kmax)
    return P


##Start of the real program

def main():
    start_time = time.time()
    # Mesh
    dt = float(T) / float(N)
    h = (Smax - Smin) / (I + 1)
    s = Smin + np.transpose(np.linspace(1, I, I)) * h
    global H
    H=h

    # CFL numbers
    cfl = ((dt / (h ** 2)) * (sigma * Smax) ** 2) / 2
    print 'Notre condition CFL {!r}.'.format(cfl)

    Pold2 = np.zeros(len(s))
    Pold = np.zeros(len(s))
    P = np.zeros(len(s))
    i = np.where(s<=K)
    for j in range (0,len(s)):
        if np.any(i[0][:] == j):
            Pold2[j] = conditionsinit(s[j])
            Pold[j] = conditionsinit(s[j])
            P[j] = conditionsinit(s[j])
        else:
            Pold2[j] = conditionsinit(s[j])
            Pold[j] = conditionsinit(s[j])
            P[j] = conditionsinit(s[j])
    plt.figure(1)
    plt.plot(s, P, label='v(0,x)')
    switcher = {
        'CENTRE': centred(s, h),
    }
    switcher.get(CENTRAGE, "CENTRAGE not programmed")

    Id = np.eye(len(A))
    for n in range(0, int(N-1)):
        t = n * dt
        qt(t)
        f(t+dt,P,s)
        plt.figure(1)
        #plt.plot(s, ft)
        #plt.plot(s, P)
        switcher2 = {
            'EI-AMER-NEWTON': ei_amer_scheme(Id, dt, np.copy(P), t, s, n, Pold, ft),
            'CN1-AMER-NEWTON': cn1_amer_scheme(Id, dt, np.copy(P), t, s, n, Pold, ft),
            'CN2-AMER-NEWTON': cn2_amer_scheme(Id, dt, np.copy(P), t, s, n, Pold, ft),
            'BDF2-AMER-NEWTON': bdf2_amer_scheme(Id, dt, np.copy(P), t, s, n, Pold, np.copy(ft)),
            'BDF3-AMER-NEWTON': bdf3_amer_scheme(Id, dt, np.copy(P), t, s, n, Pold, Pold2, ft)
        }
        Pold2 = np.copy(Pold)
        Pold = np.copy(P)  ##Attention
        P = switcher2.get(SCHEME, "SCHEME not programmed")
#        plt.plot(s, P, label='Prix(t)')
#        plt.show()

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
    print time.time() - start_time

    plt.figure(1)
    #plt.xlim(0, 200)
    #plt.ylim(0, 100)
    plt.plot(s, P, label='Scheme')
    plt.plot(s, vt, label='Sol explicite')
    #plt.plot(s, ft, label='deviation')

    # plt.figure(1)
    # plt.plot(s,BS(T,s),label='Closed formula')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
