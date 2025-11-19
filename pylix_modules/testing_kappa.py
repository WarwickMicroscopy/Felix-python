# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:37:23 2025

@author: adamc
"""
import numpy as np
import math
import scipy.integrate as integrate
import scipy.constants as c
from pylix_modules import pylix as px
from pylix_modules import pylix_dicts as fu
import matplotlib.pyplot as plt


#max is 10 angrstrom or bohr radius 

Bohr = 0.52917721067 # in angstrom


def calc_slater_orbitals(z, orbital,r):
    
    
    #ok so now when we calc slater orbital we pass kappa scaled q 
    #r, distance of electron from atomic nucleus#
    #N is normalizing constant 

    delta = np.array(fu.slater_coefficients[z][orbital]['delta'])
    delta = delta / 0.52917721092
       # convert to angstrom 
    
    C= np.array(fu.slater_coefficients[z][orbital]['coeff'])
    
    n = int(orbital[0])
    
    # for now we just state 1s contriutes to the core and 2s contributes to valence with a respective electron occupation of 2,1
  
    # we need array of R values to sample the electron density from so we can actually evaluate a fourier transofrm integral
    
    R_total = 0
   
    #Total radial finction is a superposition of these primitive radial functions and their corresponding expansion coefficent C_jln given 
    #in out hartree fock equation
    #delta is given next to each slater type orbital in the table 
    
    # after we fourier transform radial function to get form factor we need to use motte bethe formula to get to electron scattering factor
    # then compare with kirkland to check agreement and upscale
    
    
    for cj,zj in zip(C,delta):
        Nj = ((2*zj)**(n+0.5))/(np.sqrt(math.factorial(2*n)))
        S_j = Nj*r**(n-1)*np.exp(-zj*r)          # each electron is defined by a primitive slater orbital of this form 
        R_total += cj*S_j
        
    
    return R_total      # now return the radial function for our atom use this function and integrate it to get form factor
    


def xray_form_factor_valence(r, rho, Q,pv,k):
    # rho = electron density at r, Q = array of Q values
    #S= Q/2*np.pi
    
    fQ = []
    for q in Q:
        integrand =  4*np.pi*(k**3)*rho * r**2 * np.sinc(q*r/np.pi)  # np.sinc(x) = sin(pi x)/(pi x)
        fQ.append(np.trapz(integrand, r))
    return pv*np.array(fQ)   #scale by number of electrons in valence



def xray_form_factor_core(r,rho,Q,pc):
    fQ = []
    for q in Q:
        integrand =  4*np.pi*rho * r**2 * np.sinc(q*r/np.pi)  # np.sinc(x) = sin(pi x)/(pi x)
        fQ.append(np.trapz(integrand, r))
    return pc*np.array(fQ)
    

def calc_scattering_amplitudes(q, Z ,pv,kappa):
    
   
    r_max = 20  # in angstrom
    n_points = 10000
    r = np.linspace(1e-6, r_max, n_points)
    
    core_orbitals = fu.elements_info[Z]['core_orbitals']
    valence_orbitals = fu.elements_info[Z]['valence_orbitals']
    core_density =0
    valence_density =0
    print(core_orbitals)
    print(valence_orbitals)
    n_e_core=0
    for orbital in core_orbitals:
       
        n_e_core += fu.elements_info[Z]['occupation'][orbital]
        R=  calc_slater_orbitals(Z,orbital,r)
       
        
        core_density += (R**2)
        
        
        
    core_density /= (4*np.pi)  
   
    
    
    
    for orbital in valence_orbitals:
        
        R= calc_slater_orbitals(Z,orbital,r*kappa)
        
       
        
        valence_density +=  (R**2)
        
        
        
    valence_density /= (4*np.pi)  
    valence_density_n = valence_density / np.trapz(4*np.pi*r**2*valence_density, r)   # need to normalize to 1 electron then scale by pv after 
    

    
    #N_core =  (4*np.pi * np.trapz(r**2 * core_density, r))
   # N_valence = (4*np.pi * np.trapz(r**2 * valence_density, r))
  
   #core_density = (1/(4*np.pi))*(R_core**2)  # this will depend on n,l if multipolar is considered so its more complicate than this , just using this as an example 
   #valence_density = (1/(4*np.pi))*(R_valence**2)# wavefucniton = R(r)Y_l^m(theta,phi)
    pc = n_e_core
    density_total = pc*core_density+ pv*kappa**3*valence_density_n #p_atom(r) in kappa formalism 
    f_valence = xray_form_factor_valence(r, valence_density_n, Q, pv, kappa)
    f_core = xray_form_factor_core(r, core_density, Q, pc)
    

    
    
    f_x_total =  f_core + f_valence    #fourier transform of the calculated radial funciton in 3d r from 0 to infinity 
    
                   
    return f_x_total 

def convert_x(Z,f_x,q):  # q is in angstrom ^-1 
       
    
    f_e = (Z - f_x) / (2 * np.pi**2 * Bohr * q**2)     # at q is 0 must handle singularity defined in kirkland book pg 295
 #motte bethe formula
   
    
   
    return f_e # factor off here not sure why .


# need to carefully look through and fix scaling of function 
#close but not quite


def kappa_factors(q,Z,pv,kappa):
    #return convert_x(Z,calc_scattering_amplitudes(q, Z, pv, kappa),q)
    return calc_scattering_amplitudes(q, Z, pv, kappa)

#should handle values below 0.5 Q using kirkland values or some type of extrapolation

#Z = 3  # Li
Z= 8  #O
pv = 6  # 1 valence electron
kappa = 1

Q = np.linspace(0.5, 10, 100)  # momentum transfer array 1/bohr

S = Q / (2*np.pi)
g= Q*2*np.pi

f_kappa = kappa_factors(Q, Z, pv, kappa).ravel()

f_kirkland = px.f_kirkland(Z, g).ravel() 



plt.plot(Q, f_kappa, label='Kappa/PV')
#plt.plot(Q, f_kirkland, label='Kirkland', linestyle='--')
plt.xlabel('Q (1/Å)')
plt.ylabel('Electron scattering factor f(Q)')
plt.legend()
plt.show()


