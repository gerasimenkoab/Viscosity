from math import pi
from tokenize import Double
import numpy as np
import scipy as sc
import itertools
from scipy.interpolate import interp1d
from tkinter import *
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename
from scipy.constants import k, N_A, pi, m_u
from PIL import ImageTk, Image
import xml.etree.ElementTree as ET # xml parser

#species:
sps = {
      "N":0,
      "N2":1
      }
num_of_species = int(len(sps))
# x - moll fraciton of each species
x = np.zeros(num_of_species)
# m - mass of species
m = np.zeros(num_of_species)
mij = np.ndarray((num_of_species, num_of_species))
#molecule diameter - sigma
sg = np.ndarray(num_of_species)
sgij = np.ndarray((num_of_species, num_of_species))

eta_i = np.ndarray(num_of_species)
eta_ij = np.ndarray((num_of_species, num_of_species))
coef_ij = np.ndarray((num_of_species, num_of_species))
# H00 - bracket integral with p=0,q=0 for viscosity in first order approximation
H00 = np.zeros((num_of_species, num_of_species))

# Omega integrals ( collision integrals) - obtained from Mutation collision.xml
Omega11 = np.ndarray((num_of_species, num_of_species))
Omega22 = np.ndarray((num_of_species, num_of_species))

# Omega integrals starred * ( collision integrals) - obtained from Mutation collision.xml
Omega11s = np.ndarray((num_of_species, num_of_species))
Omega22s = np.ndarray((num_of_species, num_of_species))


# Astar = Omega22s/Omega11s
Astar = np.ndarray((num_of_species,num_of_species))


#TODO: parse collision.xml for Omega values for desired species
# get N N2 values for testing

Omega11str = np.ndarray((num_of_species,num_of_species,2), dtype=object)
Omega22str = np.ndarray((num_of_species,num_of_species,2), dtype=object)






def InitSpiciesTest():
      #TODO: need values for species constants
      # x - moll fraciton of each species
      x[sps['N']] = 0.0418369
      x[sps['N2']] = 0.958163
      # m - mass of species
      m[sps['N']] = 14 * m_u #N
      m[sps['N2']] = 28 * m_u #N2

      sg[sps['N']] = 1.875e-10
      sg[sps['N2']] = 3.75e-10


def InitSpiciesTest_N2CO():
      #TODO: need values for species constants
      # x - moll fraciton of each species
      x[sps['N2']] = 0.2337
      x[sps['CO']] = 0.7663
      # m - mass of species
      m[sps['N2']] = 28 /(1000*N_A) #N
      m[sps['CO']] = 28 * m_u #N2
      # atomic diameter
      sg[sps['N2']] = 3.681e-10
      sg[sps['CO']] = 3.706e-10

def InitOmegaInt(T):
      """Initialize Omega with values from collision.xml for the given temperature T"""
      tree = ET.parse('collisions.xml')
      root = tree.getroot()
      fit_type = 'quadratic'

      for name1, name2 in itertools.product(sps.keys(), repeat=2):
            i = sps[name1] 
            j = sps[name2]
            mij[i][j] = m[i]*m[j] / (m[i] + m[j])
            sgij[i][j] = (sg[i] + sg[j]) / 2.

            for x in root.findall('pair'):
                  if set(x.attrib.values()) == {name1,name2}:
                        Omega11str[i][j][0],Omega11str[i][j][1] = x.find('Q11').text.split(",")
                        Omega22str[i][j][0],Omega22str[i][j][1] = x.find('Q22').text.split(",")
                        
            x = list(map(float, Omega11str[i][j][0].split()))
            y = list(map(float, Omega11str[i][j][1].split()))
            f11 = interp1d(x,y,kind = fit_type,bounds_error=True)
            try:
                  Omega11s[i][j] = f11(T) * 10**(-20) /(sgij[i][j]**2)
                  Omega11[i][j] = f11(T) * 10**(-20) /(1)
            except ValueError:
                  print("T ",T," out of range. Omega11:",name1,name2)
                  return
            x = list(map(float, Omega22str[i][j][0].split()))
            y = list(map(float, Omega22str[i][j][1].split()))
            f22 = interp1d(x,y,kind = fit_type,bounds_error=True)
            try:
                  Omega22s[i][j] = f22(T) * 10**(-20)/(sgij[i][j]**2 )
                  Omega22[i][j] = f22(T) * 10**(-20)/(1 )
            except ValueError:
                  print("T ",T," out of range. Omega 22:",name1,name2) 
                  return
            Astar[i][j] = Omega22s[i][j] / Omega11s[i][j]

def Init_Eta_Coef(T):
      for name in sps.keys():
            i = sps[name]
            eta_i[i] = 5.0/16.0 * np.sqrt(pi * m[i] * k * T) / (pi*sg[i]**2* Omega22s[i][i])
      for name1,name2 in itertools.product(sps.keys(),repeat = 2):
            i = sps[name1]
            j = sps[name2]
            eta_ij[i][j] = 5.0/16.0 * np.sqrt(pi * mij[i][j] * k * T) / (pi* sgij[i][j]**2 * Omega22s[i][j])
            coef_ij[i][j] = 2.0 *x[i]*x[j] *m[i]*m[j] /( eta_ij[i][j] * (m[i] + m[j])**2 )


# alternative solution: solve system H00*(b*kT/2) = x
# viscosity compute as dot product of x and (b*kT/2)

# symmetry H_ij = H_ji

def CalculateMatrixH00_FK():
      """"Calculation of H00 according to Ferziger Kaper"""
      for i, j in itertools.product(range(0,num_of_species),repeat=2):
                  if i == j :
                        tmp = x[i]**2 / eta_i[i]
                        for l in range(num_of_species):
                              if l != i :
                                    tmp = tmp + coef_ij[i][l]*(5.0 / (3.0 * Astar[i][l]) + m[l]/m[i])
                        H00[i][j] = tmp
                  else:
                        H00[i][j] = -coef_ij[i][j] * (5.0/(3.0 * Astar[i][j]) - 1.)

InitSpiciesTest()
print("! Mole faction are the same for all temperatures !")
for temperature in range(1100,10001,500):
#temperature = 5100
      InitOmegaInt(temperature)
      Init_Eta_Coef(temperature)
      CalculateMatrixH00_FK()
      b = np.linalg.solve(H00,x)
      eta_1 = np.dot(x,b)#*k*temperature/2.
      print("T:",temperature,"eta:",eta_1)


# print("x:",x)
# print("m:",m)
# print("sigma_ij:")
# print(sgij)
# print("Omega11s mut:")
# print(Omega11/pi*1e20)
# print("Omega22s mut:")
# print(Omega22/pi*1e20)
# print("A*:")
# print(Astar)
# print("eta_ij:")
# print(eta_ij)
# print("coef_ij")
# print(coef_ij)
# print("H00:")
# print(H00)
