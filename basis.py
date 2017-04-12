import numpy as np
import itertools
from math import factorial

class Fermion_Bose_basis(object):
    def __init__(self, lat):
        
        self.size = (factorial(lat.n_fermion)/factorial(lat.n_fermion-lat.n_spin)/factorial(lat.n_spin))**2*lat.n_boson**(lat.n_boson_max+1)
            
        print 'basis size = ',self.size
            
        self.spin_up = np.empty((self.size,lat.n_fermion))
        self.spin_down=np.empty((self.size,lat.n_fermion))
        self.boson = np.empty((self.size,lat.n_boson))

        spin_basis=[]
        for state in itertools.product([0, 1], repeat=lat.n_fermion):
            if np.sum(state)==lat.n_spin:
                spin_basis.append(state)
        spin_basis=np.array(spin_basis)

        boson_basis=[]
        for state in itertools.product(range(lat.n_boson_max+1),repeat=lat.n_boson):
            boson_basis.append(state)
        boson_basis=np.array(boson_basis)

        L=spin_basis.shape[0];M=boson_basis.shape[0]
        print L,M
        for i in xrange(self.size):
            self.spin_up[i,:] = spin_basis[i%L,:]
            self.spin_down[i,:]=spin_basis[i//L%L,:]
            self.boson[i,:]=boson_basis[i//(L*L),:]
            # print self.spin_up[i,:],self.spin_down[i,:],self.boson[i,:]

        self.full=np.hstack((self.spin_up,self.spin_down,self.boson))