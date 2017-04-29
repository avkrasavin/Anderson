import numpy as np
from numpy import pi
from numpy.fft import *
from numpy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b

# lattice parametrs
Nc = 4
Nf = 4
Nb = 4
L = int(np.sqrt(Nc))
T = np.array([[1, 1, 1, 1], [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]]) / 2.
t = 1
V = 1
mu = 0
U = 8

# k-meshgrid parametrs
N = 10000

# matsubara grid parametrs
n = 100
beta = 1
matsubara = (2 * np.linspace(0, n, n + 1) + 1) * np.pi / beta
boson_matsubara = (2 * np.linspace(0, n, n + 1)) * np.pi / beta

# fermion k-space
kx = np.linspace(-2 * pi, 2 * pi, 2 * int(np.sqrt(N)) + 1)
kx = np.delete(kx, 0)
ky = np.linspace(-2 * pi, 2 * pi, 2 * int(np.sqrt(N)) + 1)
ky = np.delete(ky, 0)
kx, ky = np.meshgrid(kx, ky)
E_k = -2 * t * (np.cos(kx) + np.cos(ky))

# boson q-space
qx = np.linspace(-2 * pi, 2 * pi, 2 * int(np.sqrt(N)) + 1)
qx = np.delete(qx, 0)
qy = np.linspace(-2 * pi, 2 * pi, 2 * int(np.sqrt(N)) + 1)
qy = np.delete(qy, 0)
qx, qy = np.meshgrid(qx, qy)
V_q = 4 * V - 2 * V * (np.cos(qx) + np.cos(qy))

# cluster K-space
KX = np.linspace(-pi, pi, L + 1)
KX = np.delete(KX, 0)
dK = KX[1] - KX[0]
KY = np.linspace(-pi, pi, L + 1)
KY = np.delete(KY, 0)
KX, KY = np.meshgrid(KX, KY)


def get_start_GFs():
    # Patching
    G0_K = np.empty((matsubara.size, KX.shape[0], KX.shape[1]), dtype=np.complex128)
    P0_K = np.empty((matsubara.size, KX.shape[0], KX.shape[1]), dtype=np.complex128)
    for i in xrange(L):
        for j in xrange(L):
            indexes_k = ((kx > (KX[i, j] - dK / 2.)) & (kx < (KX[i, j] + dK / 2.))) & (
                (ky > (KY[i, j] - dK / 2.)) & (ky < (KY[i, j] + dK / 2.)))
            indexes_q = ((qx > (KX[i, j] - dK / 2.)) & (qx < (KX[i, j] + dK / 2.))) & (
                (qy > (KY[i, j] - dK / 2.)) & (qy < (KY[i, j] + dK / 2.)))
            for k in xrange(matsubara.size):
                G0_K[k, i, j] = Nc * np.sum((1j * matsubara[k] + mu - E_k[indexes_k]) ** (-1)) / N
            P0_K[:, i, j] = Nc * np.sum(V_q[indexes_q]) / N
    return G0_K, P0_K


def get_parametrs(G0_K, P0_K):
    # Fitting fermions
    G0_R = np.zeros_like(G0_K)
    for i in xrange(matsubara.size):
        G0_R[i, :, :] = ifft2(ifftshift(G0_K[i, :, :]))

    G0_IJ = np.zeros((matsubara.size, Nc, Nc), dtype=np.complex128)
    for i in xrange(Nc):
        for j in xrange(Nc):
            x = i % L - j % L
            y = i // L - j // L
            if x < 0: x += L
            if y < 0: y += L
            G0_IJ[:, i, j] = G0_R[:, x, y]

    G0_M = np.zeros_like(G0_IJ)
    for i in xrange(matsubara.size):
        G0_M[i, :, :] = (inv(T).dot(G0_IJ[i, :, :])).dot(T)

    def molecular_GF(parameters):
        em = parameters[0]
        h = parameters[1:Nf / Nc + 1]
        ek = parameters[Nf / Nc + 1:]
        G0 = np.zeros(matsubara.size, dtype=np.complex128)
        for i in xrange(matsubara.size):
            G0[i] = (1j * matsubara[i] + mu - em - np.sum(h ** 2 / (1j * matsubara[i] - ek))) ** (-1)
        return G0

    bounds = [(-4 * t, 4 * t)]
    for i in xrange(Nf / Nc): bounds.append((0.05 * t, 4 * t))
    for i in xrange(Nf / Nc): bounds.append((-4 * t, 4 * t))

    parametrs = []
    for i in xrange(Nc - 1):
        p_fermion = 2 * np.random.random(1 + 2 * Nf / Nc) - 1
        G0_real = G0_M[:, i, i]

        def error_G0(parameters):
            G0 = molecular_GF(parameters)
            return np.sum(np.abs(np.conj(G0 - G0_real) * (G0 - G0_real)) / matsubara)

        p_fermion = fmin_l_bfgs_b(error_G0, x0=p_fermion, approx_grad=True, disp=True, bounds=bounds)
        parametrs.append(p_fermion[0])
    parametrs.append(p_fermion[0])

    em = np.zeros((Nc, Nc))
    h = np.zeros((Nc, Nf))
    ek = np.zeros(Nf)
    for i in xrange(Nc):
        em[i, i] = parametrs[i][0]
        h[i, i * Nf / Nc:(i * Nf / Nc + Nf / Nc)] = parametrs[i][1:Nf / Nc + 1]
        ek[i * Nf / Nc:(i * Nf / Nc + Nf / Nc)] = parametrs[i][Nf / Nc + 1:]

    # Fitting bosons
    P0_R = np.zeros_like(P0_K)
    for i in xrange(matsubara.size):
        P0_R[i, :, :] = ifft2(ifftshift(P0_K[i, :, :]))

    P0_IJ = np.zeros((matsubara.size, Nc, Nc), dtype=np.complex128)
    for i in xrange(Nc):
        for j in xrange(Nc):
            x = i % L - j % L
            y = i // L - j // L
            if x < 0: x += L
            if y < 0: y += L
            P0_IJ[:, i, j] = P0_R[:, x, y]

    P0_M = np.zeros_like(P0_IJ)
    for i in xrange(matsubara.size):
        P0_M[i, :, :] = (inv(T).dot(P0_IJ[i, :, :])).dot(T)

    def molecular_PF(parameters):
        em = parameters[0]
        y = parameters[1:Nb / Nc + 1]
        eq = parameters[Nb / Nc + 1:]
        P0 = np.zeros(matsubara.size, dtype=np.complex128)
        eq[eq == 0] = 10 ** (-8)
        for i in xrange(matsubara.size):
            P0[i] = (em + np.sum(y ** 2 * eq / (boson_matsubara[i] ** 2 + eq ** 2)))
        return P0

    bounds = [(-1000 * V, 1000 * V)]
    for i in xrange(Nb / Nc): bounds.append((0, V))
    for i in xrange(Nb / Nc): bounds.append((0, 8 * V))

    boson_parametrs = []
    for i in xrange(Nc - 1):
        p_boson = 2 * np.random.random(1 + 2 * Nb / Nc)
        P0_real = P0_M[:, i, i]

        def error_P0(parameters):
            P0 = molecular_PF(parameters)
            return np.sum((P0.real - P0_real.real) ** 2)

        p_boson = fmin_l_bfgs_b(error_P0, x0=p_boson, approx_grad=True, disp=True, bounds=bounds)
        boson_parametrs.append(p_boson[0])
    boson_parametrs.append(p_boson[0])

    en = np.zeros((Nc, Nc))
    y = np.zeros((Nc, Nb))
    eq = np.zeros(Nb)
    for i in xrange(Nc):
        en[i, i] = boson_parametrs[i][0]
        y[i, i * Nb / Nc:(i * Nb / Nc + Nb / Nc)] = boson_parametrs[i][1:Nb / Nc + 1]
        eq[i * Nb / Nc:(i * Nb / Nc + Nb / Nc)] = boson_parametrs[i][Nb / Nc + 1:]

    t_renorm = np.round((inv(T).dot(em)).dot(T), 2)[0, 1]
    V_renorm = np.round((inv(T).dot(en)).dot(T), 2)[0, 1]

    hybridization = np.round(T.dot(np.abs(h)), 2)[1, :]
    gamma = np.round(T.dot(np.abs(y)), 2)[1, :]

    delta_U = np.round((inv(T).dot(en)).dot(T), 2)[0, 0]
    delta_mu = np.round((inv(T).dot(em)).dot(T), 2)[0, 0]

    return U + delta_U, mu + delta_mu, t_renorm, V_renorm, hybridization, gamma, ek, eq, G0_K, P0_K


def DCA_loop(G_K, P_K, G0_K, P0_K):
    SE_K = np.zeros_like(G0_K)
    PE_K = np.zeros_like(P0_K)
    for i in xrange(L):
        for j in xrange(L):
            SE_K[:, i, j] = G0_K[:, i, j] ** (-1) - G_K[:, i, j] ** (-1)
            PE_K[:, i, j] = P0_K[:, i, j] ** (-1) - P_K[:, i, j] ** (-1)

    G_K_new = np.zeros_like(G0_K)
    P_K_new = np.zeros_like(P0_K)

    for i in xrange(L):
        for j in xrange(L):
            indexes_k = ((kx > (KX[i, j] - dK / 2.)) & (kx < (KX[i, j] + dK / 2.))) & (
                (ky > (KY[i, j] - dK / 2.)) & (ky < (KY[i, j] + dK / 2.)))
            indexes_q = ((qx > (KX[i, j] - dK / 2.)) & (qx < (KX[i, j] + dK / 2.))) & (
                (qy > (KY[i, j] - dK / 2.)) & (qy < (KY[i, j] + dK / 2.)))
            for k in xrange(matsubara.size):
                G_K_new[k, i, j] = Nc * np.sum((1j * matsubara[k] + mu - E_k[indexes_k] - SE_K[k, i, j]) ** (-1)) / N
            P_K_new[:, i, j] = Nc * np.sum((V_q[indexes_q] ** (-1) - PE_K[:, i, j]) ** (-1)) / N

    G0_K_new = np.zeros_like(G0_K)
    P0_K_new = np.zeros_like(P0_K)
    for i in xrange(L):
        for j in xrange(L):
            G0_K_new = (G_K_new[:, i, j] ** (-1) + SE_K[:, i, j]) ** (-1)
            P0_K_new = (P_K_new[:, i, j] ** (-1) + PE_K[:, i, j]) ** (-1)

    return get_parametrs(G0_K_new, P0_K_new)
