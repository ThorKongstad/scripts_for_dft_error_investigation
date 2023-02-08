# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:37:57 2016

@author: hao
"""
import matplotlib.pyplot as plt
import numpy as np

ECO = np.array([-14.866, -15.207, -34.788, -26.121, -14.333])
EH2O = np.array([-14.245, -14.552, -27.214, -21.493, -13.722])
EH2 = np.array([-6.715, -6.749, -7.988, -7.589, -6.716])
ECO2 = np.array([-23.096, -23.769, -54.739, -40.713, -22.042])
ECH4 = np.array([-23.496, -23.994, -34.034, -29.692, -23.492])
EHCOOH = np.array([-29.696, -30.587, -62.578, -48.078, -28.647])
ECH3OH = np.array([-29.745, -30.524, -52.051, -42.381, -29.204])
ECH3CH2OH = np.array([-45.889, -47.171, -77.568, -63.898, -45.338])
EC3H8 = np.array([-55.477, -56.962, -91.498, -72.337, -55.433])
EC2H6 = np.array([-39.455, -40.439, -59.333, -50.929, -39.435])
EC2H4 = np.array([-31.142, -31.94, -49.805, -41.92, -31.113])
EC4H6 = np.array([-55.344, -56.962, -91.498, -76.11, -55.289])
ECH3COOH = np.array([-46.034, -47.436, -88.247, -69.769, -44.959])
EHCOOCH3 = np.array([-45.332, -46.713, -87.577, -69.148, -44.265])

# we need to correct the DFT energies with the ZPC and CP values
# adding ZPC and CP values from
# "Identifying systematic DFT errors in catalytic reactions"
# By Rune Christensen
ECO += 0.14 + 0.09
EH2O += 0.60 + 0.1
EH2 += 0.32 + 0.09
ECO2 += 0.31 + 0.10
ECH4 += 1.17 + 0.10
EHCOOH += 0.9 + 0.11
ECH3OH += 1.36 + 0.11
ECH3CH2OH += 2.12 + 0.11
EC3H8 += 2.73 + 0.1
EC2H6 += 1.96
EC2H4 += 1.34
ECH3COOH += 1.62
EHCOOCH3 += 1.63 + 0.11
# calculating the reaction enegies.
E0 = ECO + EH2O - EH2 - ECO2  # CO2 + H2 -> CO + H2O
E1 = ECH4 + 2 * EH2O - ECO2 - 4 * EH2  # 4H2 + CO2 -> CH4 + 2H2O
E2 = ECH4 + EH2O - ECO - 3 * EH2  # 3H2 + CO -> CH4 +H2O
E3 = EHCOOH - EH2 - ECO2  # CO2 + H2-> HCOOH
E4 = EHCOOH - EH2O - ECO  # CO + H2O -> HCOOH
E5 = ECH3OH + EH2O - ECO2 - 3 * EH2  # 3H2 + CO2 -> CH3OH + H2O
E6 = ECH3OH - ECO - 2 * EH2  # 2H2 + CO -> CH3OH
E7 = (1. / 2.) * ECH3CH2OH + (3. / 2.) * EH2O - ECO2 - 3 * EH2  # 3H2 + CO2 -> 1/2CH3CH2OH + 3/2H2O
E8 = (1. / 2.) * ECH3CH2OH + (1. / 2.) * EH2O - ECO - 2 * EH2  # 2H2 + CO -> 1/2CH3CH2OH + 1/2H2O
E9 = (1. / 3.) * EC3H8 + 2 * EH2O - (10. / 3.) * EH2 - ECO2  # 10/3H2 + CO2 -> 1/3C3H8 + 2H2O
E10 = (1. / 3.) * EC3H8 + EH2O - (7. / 3.) * EH2 - ECO  # 7/3H2 + CO -> 1/3C3H8 + H2O
E11 = (1. / 2.) * EC2H6 + 2 * EH2O - ECO2 - (7. / 2.) * EH2  # 7/2H2 + CO2 -> 1/2C2H6 + 2H2O
E12 = (1. / 2.) * EC2H6 + EH2O - ECO - (5. / 2.) * EH2  # 5/2H2 + CO -> 1/2C2H6 + H2O
E13 = (1. / 2.) * EC2H4 + 2 * EH2O - ECO2 - 3 * EH2  # 3H2 + CO2 -> 1/2C2H4 + 2H2O
E14 = (1. / 2.) * EC2H4 + EH2O - ECO - 2 * EH2  # 2H2 + CO -> 1/2C2H4 + H2O
E15 = (1. / 4.) * EC4H6 + 2 * EH2O - ECO2 - (11. / 4) * EH2  # 3H2 + CO2 -> 1/4C4H6 + 2H2O
E16 = (1. / 4.) * EC4H6 + EH2O - ECO - (7. / 4.) * EH2  # 2H2 + CO -> 1/4C4H6 + H2O
E17 = (1. / 2.) * ECH3COOH + EH2O - ECO2 - 2 * EH2  # 2H2 + CO2 -> 1/2CH3COOH + H2O
E18 = (1. / 2.) * ECH3COOH - ECO - EH2  # H2 + CO -> 1/2CH3COOH
E19 = (1. / 2.) * EHCOOCH3 + EH2O - ECO2 - 2 * EH2  # 2H2 + CO2 -> 1/2HCOOCH3
E20 = (1. / 2.) * EHCOOCH3 - ECO - EH2  # H2 + CO -> 1/2HCOOCH3

# showing the reaction reference enthalphy energies.
Eref0 = 0.43
Eref1 = -1.71
Eref2 = -2.14
Eref3 = 0.15
Eref4 = -0.27
Eref5 = -0.55
Eref6 = -0.98
Eref7 = -0.89
Eref8 = -1.32
Eref9 = -1.3
Eref10 = -1.72
Eref11 = -1.37
Eref12 = -1.8
Eref13 = -0.66
Eref14 = -1.09
Eref15 = -0.65
Eref16 = -1.08
Eref17 = -0.67
Eref18 = -1.10
Eref19 = -0.17
Eref20 = -0.60


# Defining function to calculate the error
def MeanError(Xcoor, Xreal):
    val = ((Xcoor - Xreal) ** 2) ** (0.5)
    return val


# Defining error function to uptimize: CO2 correction
def MeanSumCO2(CO2coor):
    val = (MeanError(E0 - CO2coor, Eref0) + MeanError(E1 - CO2coor, Eref1) + MeanError(E2, Eref2) + MeanError(
        E3 - CO2coor, Eref3) +
           MeanError(E4, Eref4) + MeanError(E5 - CO2coor, Eref5) + MeanError(E6, Eref6) + MeanError(E7 - CO2coor,
                                                                                                    Eref7) + MeanError(
                E8, Eref8) +
           +MeanError(E9 - CO2coor, Eref9) + MeanError(E10, Eref10) + MeanError(E11 - CO2coor, Eref11) + MeanError(E12,
                                                                                                                   Eref12) + MeanError(
                E13 - CO2coor, Eref13) +
           MeanError(E14, Eref14) + MeanError(E15 - CO2coor, Eref15) + MeanError(E16, Eref16) + MeanError(E17 - CO2coor,
                                                                                                          Eref17) + MeanError(
                E18, Eref18) +
           +MeanError(E19 - CO2coor, Eref19) + MeanError(E20, Eref20)) / 20.0
    return val


# Defining error function to uptimize: CO2 and CO corrections
def MeanSum(CO2coor, COcoor):
    val = (MeanError(E0 - CO2coor + COcoor, Eref0) + MeanError(E1 - CO2coor, Eref1) + MeanError(E2 - COcoor,
                                                                                                Eref2) + MeanError(
        E3 - CO2coor, Eref3) +
           MeanError(E4 - COcoor, Eref4) + MeanError(E5 - CO2coor, Eref5) + MeanError(E6 - COcoor, Eref6) + MeanError(
                E7 - CO2coor, Eref7) + MeanError(E8 - COcoor, Eref8) +
           +MeanError(E9 - CO2coor, Eref9) + MeanError(E10 - COcoor, Eref10) + MeanError(E11 - CO2coor,
                                                                                         Eref11) + MeanError(
                E12 - COcoor, Eref12) + MeanError(E13 - CO2coor, Eref13) +
           MeanError(E14 - COcoor, Eref14) + MeanError(E15 - CO2coor, Eref15) + MeanError(E16 - COcoor,
                                                                                          Eref16) + MeanError(
                E17 - CO2coor, Eref17) + MeanError(E18 - COcoor, Eref18) +
           +MeanError(E19 - CO2coor, Eref19) + MeanError(E20 - COcoor, Eref20)) / 20.0
    return val


# Defining error function to uptimize: CO2 and CO corrections
def MeanSumOxygenbond(CO2coor, COcoor):
    val = (MeanError(E0 - CO2coor + COcoor, Eref0) + MeanError(E1 - CO2coor, Eref1) + MeanError(E2 - COcoor,
                                                                                                Eref2) + MeanError(
        E3 - CO2coor + COcoor, Eref3) +
           MeanError(E4 - COcoor + COcoor, Eref4) + MeanError(E5 - CO2coor, Eref5) + MeanError(E6 - COcoor,
                                                                                               Eref6) + MeanError(
                E7 - CO2coor, Eref7) + MeanError(E8 - COcoor, Eref8) +
           +MeanError(E9 - CO2coor, Eref9) + MeanError(E10 - COcoor, Eref10) + MeanError(E11 - CO2coor,
                                                                                         Eref11) + MeanError(
                E12 - COcoor, Eref12) + MeanError(E13 - CO2coor, Eref13) +
           MeanError(E14 - COcoor, Eref14) + MeanError(E15 - CO2coor, Eref15) + MeanError(E16 - COcoor,
                                                                                          Eref16) + MeanError(
                E17 - CO2coor + 0.5 * COcoor, Eref17) + MeanError(E18 - COcoor + 0.5 * COcoor, Eref18) +
           +MeanError(E19 - CO2coor + 0.5 * COcoor, Eref19) + MeanError(E20 - COcoor + 0.5 * COcoor, Eref20)) / 20.0
    return val


# Defining error function to uptimize: CO2, CO and H2 corrections
def MeanSumH2(CO2coor, COcoor, H2coor):
    val = (MeanError(E0 - CO2coor + COcoor - H2coor, Eref0) + MeanError(E1 - CO2coor - 4 * H2coor, Eref1) + MeanError(
        E2 - COcoor - 3 * H2coor, Eref2) + MeanError(E3 - CO2coor - H2coor + COcoor, Eref3) +
           MeanError(E4 - COcoor + COcoor, Eref4) + MeanError(E5 - CO2coor - 3 * H2coor, Eref5) + MeanError(
                E6 - COcoor - 2 * H2coor, Eref6) + MeanError(E7 - CO2coor - 3 * H2coor, Eref7) + MeanError(
                E8 - COcoor - 2 * H2coor, Eref8) +
           +MeanError(E9 - CO2coor - 10. / 3.0 * H2coor, Eref9) + MeanError(E10 - COcoor - 7. / 3.0,
                                                                            Eref10) + MeanError(
                E11 - CO2coor - 7. / 2 * H2coor, Eref11) + MeanError(E12 - COcoor - 5. / 2. * H2coor,
                                                                     Eref12) + MeanError(E13 - CO2coor - 3 * H2coor,
                                                                                         Eref13) +
           MeanError(E14 - COcoor - 2. * H2coor, Eref14) + MeanError(E15 - CO2coor - 11. / 4. * H2coor,
                                                                     Eref15) + MeanError(
                E16 - COcoor - 7. / 4. * H2coor, Eref16) + MeanError(E17 - CO2coor - 2. * H2coor + 0.5 * COcoor,
                                                                     Eref17) + MeanError(
                E18 - COcoor - H2coor + 0.5 * COcoor, Eref18) +
           +MeanError(E19 - CO2coor - 2. * H2coor + 0.5 * COcoor, Eref19) + MeanError(
                E20 - COcoor - H2coor + 0.5 * COcoor, Eref20)) / 20.0
    return val


xp2 = np.linspace(-1, 1, 100);
s1 = 40;
L = 3;
plt.figure();
Fontsizesub = 15;
Fontsizesub2 = 14;
for k in xp2:
    plt.scatter(k, MeanSumCO2(k)[0], c='k', lw=0, s=s1)
    plt.scatter(k, MeanSumCO2(k)[1], c='b', lw=0, s=s1)
    plt.scatter(k, MeanSumCO2(k)[2], c='r', lw=0, s=s1)
    plt.scatter(k, MeanSumCO2(k)[3], c='c', lw=0, s=s1)
    plt.scatter(k, MeanSumCO2(k)[4], c='m', lw=0, s=s1)

plt.scatter(1, MeanSumCO2(1)[0], c='k', lw=0, s=s1, label='RPBE')
plt.scatter(1, MeanSumCO2(1)[1], c='b', lw=0, s=s1, label='PBE')
plt.scatter(1, MeanSumCO2(1)[2], c='r', lw=0, s=s1, label='BEEF-vdW')
plt.scatter(1, MeanSumCO2(1)[3], c='c', lw=0, s=s1, label='Vdw-DF')
plt.scatter(1, MeanSumCO2(1)[4], c='m', lw=0, s=s1, label='RPBE PW')
plt.plot([0, 0], [0, 1], '-k', linewidth=L)

plt.xlim([-0.5, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc=2)
plt.title('CO2 error correction', fontsize=Fontsizesub)
plt.ylabel('Absolute Mean Error [eV]', fontsize=Fontsizesub)
plt.xlabel('Energy Correction [eV]', fontsize=Fontsizesub)
plt.grid(True)
plt.xticks(fontsize=Fontsizesub2)
plt.yticks(fontsize=Fontsizesub2)

s1 = 40;
L = 3;
plt.figure();
Fontsizesub = 15;
Fontsizesub2 = 14;
for k in xp2:
    plt.scatter(k, MeanSum(0, k)[0], c='k', lw=0, s=s1)
    plt.scatter(k, MeanSum(0, k)[1], c='b', lw=0, s=s1)
    plt.scatter(k, MeanSum(0, k)[2], c='r', lw=0, s=s1)
    plt.scatter(k, MeanSum(0, k)[3], c='c', lw=0, s=s1)
    plt.scatter(k, MeanSum(0, k)[4], c='m', lw=0, s=s1)

plt.scatter(1, MeanSum(0, 1)[0], c='k', lw=0, s=s1, label='RPBE')
plt.scatter(1, MeanSum(0, 1)[1], c='b', lw=0, s=s1, label='PBE')
plt.scatter(1, MeanSum(0, 1)[2], c='r', lw=0, s=s1, label='BEEF-vdW')
plt.scatter(1, MeanSum(0, 1)[3], c='c', lw=0, s=s1, label='Vdw-DF')
plt.scatter(1, MeanSum(0, 1)[4], c='m', lw=0, s=s1, label='RPBE PW')
plt.plot([0, 0], [0, 1], '-k', linewidth=L)

plt.xlim([-1.0, 0.5])
plt.ylim([0.0, 1.0])
plt.legend(loc=2)
plt.title('CO error correction', fontsize=Fontsizesub)
plt.ylabel('Absolute Mean Error [eV]', fontsize=Fontsizesub)
plt.xlabel('Energy Correction [eV]', fontsize=Fontsizesub)
plt.grid(True)
plt.xticks(fontsize=Fontsizesub2)
plt.yticks(fontsize=Fontsizesub2)

s1 = 40;
L = 3;
plt.figure();
Fontsizesub = 15;
Fontsizesub2 = 14;
for k in xp2:
    plt.scatter(k, MeanSumOxygenbond(2 * k, k)[0], c='k', lw=0, s=s1)
    plt.scatter(k, MeanSumOxygenbond(2 * k, k)[1], c='b', lw=0, s=s1)
    plt.scatter(k, MeanSumOxygenbond(2 * k, k)[2], c='r', lw=0, s=s1)
    plt.scatter(k, MeanSumOxygenbond(2 * k, k)[3], c='c', lw=0, s=s1)
    plt.scatter(k, MeanSumOxygenbond(2 * k, k)[4], c='m', lw=0, s=s1)

plt.scatter(1, MeanSumOxygenbond(2, 1)[0], c='k', lw=0, s=s1, label='RPBE')
plt.scatter(1, MeanSumOxygenbond(2, 1)[1], c='b', lw=0, s=s1, label='PBE')
plt.scatter(1, MeanSumOxygenbond(2, 1)[2], c='r', lw=0, s=s1, label='BEEF-vdW')
plt.scatter(1, MeanSumOxygenbond(2, 1)[3], c='c', lw=0, s=s1, label='Vdw-DF')
plt.scatter(1, MeanSumOxygenbond(2, 1)[4], c='m', lw=0, s=s1, label='RPBE PW')
plt.plot([0, 0], [0, 1], '-k', linewidth=L)

plt.xlim([-1.0, 0.5])
plt.ylim([0.0, 1.0])
plt.legend(loc=2)
plt.title('Oxygen dobbelt bond error correction', fontsize=Fontsizesub)
plt.ylabel('Absolute Mean Error [eV]', fontsize=Fontsizesub)
plt.xlabel('Energy Correction [eV]', fontsize=Fontsizesub)
plt.grid(True)
plt.xticks(fontsize=Fontsizesub2)
plt.yticks(fontsize=Fontsizesub2)

# Plotting surface plot of CO2 and CO error
# defining X, Y and Z matrices.
xp2 = np.linspace(-1, 1, 100);
MX = np.zeros((100, 100))
for k in range(0, 100):
    MX[k, :] = xp2

MY = np.zeros((100, 100))
for k in range(0, 100):
    MY[:, k] = xp2

MZ = np.zeros((100, 100))
for i in range(0, 100):
    for j in range(0, 100):
        MZ[i, j] = MeanSum(MX[i, j], MY[i, j])[0]

# importing packages for plotting in 3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(MX, MY, MZ, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0.0, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel('CO energy error [eV]', fontsize=Fontsizesub)
plt.xlabel('CO2 energy error [eV]', fontsize=Fontsizesub)
plt.show()

# Plotting surface plot of CO2, CO and H2 error
# defining X, Y and Z matrices.
xp2 = np.linspace(-1, 1, 100);
MX = np.zeros((100, 100))
for k in range(0, 100):
    MX[k, :] = xp2

MY = np.zeros((100, 100))
for k in range(0, 100):
    MY[:, k] = xp2

MZ = np.zeros((100, 100))
for i in range(0, 100):
    for j in range(0, 100):
        MZ[i, j] = MeanSumH2(MX[i, j] * 2.0, MX[i, j], MY[j, i])[0]

# importing packages for plotting in 3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(MX, MY, MZ, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0.0, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel('H2 energy error [eV]', fontsize=Fontsizesub)
plt.xlabel('Oxygen doble bond energy error [eV]', fontsize=Fontsizesub)
plt.show()