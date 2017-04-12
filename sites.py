import numpy as np

class site(object):

    def __init__(self, number,site_type,n_boson_max=0):
        super(site, self).__init__()
        self.number = number
        self.type=site_type
        if self.type=='boson':
            self.n_boson_max=n_boson_max

    def set_on_site_parametrs(self, U,mu):
        self.U=U
        self.mu=mu
        
    def set_long_range_parametrs(self,t,V):
        self.t=t
        self.V=V
        
    def scatering_parametrs(self,gamma):
        self.gamma=gamma
        
    def __repr__(self):
        return "%s %s \n" % (self.type,self.number)

class System:  
    def __init__(self, lattice_type ,n_fermion, n_boson,n_boson_max=0,n_spin=0):
        self.lattice_type=lattice_type
        self.sites=[]
        self.n_fermion=n_fermion
        self.n_boson=n_boson
        self.L=n_fermion+n_boson
        self.n_boson_max=n_boson_max
        self.n_spin = n_spin
        for i in range(n_fermion):
            self.sites.append(site(i,'fermion'))
        if n_boson>0:
            for j in range(n_boson):
                self.sites.append(site(j+n_fermion,'boson',n_boson_max))