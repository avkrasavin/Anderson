# coding: utf-8


import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from math import factorial as fact
import itertools
import progressbar
import copy

class Basis:
    
    def bose(self,m,n):
        '''Возвращает базис для Бозе-статистики. m - число узлов, n - число частиц'''
        R = fact(n + m - 1)/fact(n)/fact(m - 1)
        b = np.zeros((R,m))
        b[0,m-1] = n
        for i in range(R-1):
            j = m - 1
            while j > 0:
                if b[i,j] in range(2,n+1) :
                    b[i+1,:] = b[i,:]
                    b[i+1,j] = 0
                    b[i+1,j-1] = b[i+1,j-1] + 1
                    b[i+1,m-1] = b[i,j] - 1
                    break
                elif b[i,j] > 0:
                    b[i+1,:] = b[i,:]
                    b[i+1,j-1] = b[i+1,j-1] + 1
                    b[i+1,j] = b[i,j] - 1
                    break
                j -= 1
        return b
   
    def limit_basis(self,m,n,n_max):
        '''Возвращает базис для Бозе-статистики с ограничением числа частиц на узле. 
        m - число узлов, n - число частиц, n_max - максимальное число частиц на узле.'''
        # Размерность базиса
        R = fact(n + m - 1)/fact(n)/fact(m - 1)
        b = self.bose(m,n)
        f = np.zeros((R,m))
        j = 0
        # Откиддываем функции, в которых на узлах частиц больше n_max
        for i in range(b.shape[0]):
            if any(b[i] > n_max): 
                continue
            else:
                f[j] = b[i]
                j += 1
        return f[:j]

    def bose_unsave(self, m, n):
        '''Возвращает базис для Бозе статистики с несохраняющимся числом частиц.
        m - число узлов, n - максимальное число частиц на узле
        '''
        return np.array( map(list, itertools.product(range(n+1),repeat=m)) )
    
    def fermi(self, m, n_up, n_down):
        '''Возвращает базис для Ферми-статистики с учетом спина.'''
        R = (fact(m)/fact(n_up)/fact(m-n_up))*(fact(m)/fact(n_down)/fact(m-n_down)) 
        fs = np.zeros((R,2*m))
        part_1 = self.limit_basis(m,n_up,1)
        if n_up == n_down:
            part_2 = copy.copy(part_1)
        else:
            part_2 = self.limit_basis(m,n_down,1)
        size_1, size_2 = part_1.shape[0], part_2.shape[0]
        for i in range(size_1):
            for j in range(size_2):
                fs[i*size_2+j] = np.concatenate((part_1[i],part_2[j]), axis=0)
        return fs
    def full_basis(self,m_d, m_c, m_b,n_down, n_up, n_max):
        '''m_d - число узлов в кластере, m_c - число узлов в Ферми ванне, m_b - число узлов в Бозе ванне,
        n_d_up - число частиц со спином вверх в кластере, n_d_down - число частиц со спином вниз в кластере, 
        n_c_up - число частиц со спином вверх в Ферми ванне, n_c_down - число частиц со спином вниз в Ферми ванне,
        n_max - максимум частиц на узле в Бозе ванне
        '''
        mtx_1 = self.fermi(m_d+m_c, n_up,n_down)
        mtx_2 = self.bose_unsave(m_b,n_max)
        size_1, size_2 = mtx_1.shape[0], mtx_2.shape[0]
        fb = np.zeros((size_1*size_2,mtx_1.shape[1]+m_b))
        for i in range(size_1):
            for j in range(size_2):
                fb[i*size_2+j] = np.concatenate((mtx_1[i],mtx_2[j]), axis=0)
        return fb
class Hamiltonian(Basis):
    def __init__(self,e_c, V_cd, U_d, e_b, gamma_bd, t_d, V_d):
        self.e_c = e_c
        self.V_cd = V_cd
        self.U_d = U_d
        self.e_b = e_b
        self.gamma_bd = gamma_bd
        self.t_d = t_d
        self.V_d = V_d
    # Опертор рождения * уничтожения
    def up_down(self, number_to_up, number_to_down, func): 
        '''Функция возвращает коэффициент и новую функцию'''
        function = copy.copy(func)
        if function[number_to_down] == 0:
            return [0,function]
        elif function[number_to_up] == 1:
            return [0,function]
        else:
            coef_down = np.sqrt(function[number_to_down])
            function[number_to_down] = function[number_to_down] - 1
            coef_up = np.sqrt(function[number_to_up] + 1)
            function[number_to_up] = function[number_to_up] + 1
            return [round(coef_down*coef_up,5), function]

    # Знак 
    def sign(self, left_edge,right_edge,func):
        '''Функция определяет знак перескока'''
        left_edge,right_edge = min(left_edge,right_edge), max(left_edge,right_edge)
        if sum(func[left_edge+1:right_edge])%2:
            return 1
        else:
            return -1
    # Операторы рождения и уничтожения для бозе-ванны 
    def up(self,number_to_up,n_max, function, index):
        if function[number_to_up] == n_max:
            return [0,index]
        else:
            coef_up = np.sqrt(function[number_to_up] + 1)
            return [round(coef_up,5), (index + (n_max+1)**(20 - number_to_up - 1))]

    def down(self,number_to_down, n_max, function, index):
        if function[number_to_down] == 0:
            return [0,index]
        else:
            coef_down = np.sqrt(function[number_to_down])
            return [round(coef_down,5), (index - (n_max+1)**(20 - number_to_down - 1))]
            
    def matrix(self,m_d, m_c, m_b, n_down, n_up, n_max):   
        # Обработка входных параметров
        V_cd = np.ones((4,1)).dot(np.asmatrix(self.V_cd))
        gamma_bd = np.ones((4,1)).dot(np.asmatrix(self.gamma_bd))
        
        bar = progressbar.ProgressBar()
        basis = self.full_basis(m_d, m_c, m_b, n_down, n_up, n_max)
        R = basis.shape
        # Задаем соседей
        neigbors_d_up = {4:[5,6], 5:[4,7], 6:[4,7], 7:[5,6]}
        neigbors_d_down = {12:[13,14], 13:[12,15], 14:[12,15], 15:[13,14]}
        
        #  Число состояний 
        R_up = (fact(m_c+m_d)/fact(n_up)/fact(m_c+m_d-n_up))
        R_down = (fact(m_c+m_d)/fact(n_down)/fact(m_c+m_d-n_down))
        R_bose = (n_max+1)**m_b
        
        # Созданием массивы координат и значений
        line = np.zeros(50*10**6)
        col = np.zeros(50*10**6)
        data = np.zeros(50*10**6)
        
        # Создаем словарь собственных функций
        basis = basis.astype(int)
    
        key = [''.join(map(str,basis[i])) for i in range(R[0])]
        value = range(len(key))
        indexes = dict(zip(key,value))
        s0 = 0
        for i in bar(range(0,R_bose*R_up*R_down, R_bose*R_up)):
            for j in range(m_c,m_c+m_d): # узлы кластера
                
                for p in range(m_c): # узлы ферми-ванны
                    # Спин вверх
                    # перескок с ванны на кластер
                    coef, function = self.up_down(j,p,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        line[s0: s0+R_bose*R_down] = range(i,i+R_bose*R_down)
                        col[s0: s0+R_bose*R_down] = range(index,index+R_bose*R_down)
                        data[s0: s0+R_bose*R_down] = coef*V_cd[j-m_c,p]*self.sign(p,j,basis[i])
                        s0+=R_bose*R_down
                                                    
                    # перескок с кластера на ванну
                    coef, function = self.up_down(p,j,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        line[s0: s0+R_bose*R_down] = range(i,i+R_bose*R_down)
                        col[s0: s0+R_bose*R_down] = range(index,index+R_bose*R_down)
                        data[s0: s0+R_bose*R_down] = coef*V_cd[p,j-m_c]*self.sign(p,j,basis[i])
                        s0+=R_bose*R_down
        bar = progressbar.ProgressBar()
        s0 = np.where(data==0)[0][0]
        # Спин вниз
        for i in bar(range(0,R_bose*R_down,R_bose)):
            for j in range(2*m_c+m_d, 2*(m_c + m_d)):
                for p in range(m_c+m_d, 2*m_c+m_d):
                    # перескок с ванны на кластер
                    coef, function = self.up_down(j,p,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        for T in range(0,R_bose*R_up*R_down,R_bose*R_down):
                            line[s0: s0+R_bose] = range(i+T,i+T+R_bose)
                            col[s0: s0+R_bose] = range(index+T,index+T+R_bose)
                            data[s0: s0+R_bose] = coef*V_cd[j-(2*m_c+m_d),p-(m_c+m_d)]*self.sign(p,j,basis[i])
                            s0 += R_bose                        
                    # перескок с кластера на ванну
                    coef, function = self.up_down(p,j,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        for T in range(0,R_bose*R_up*R_down,R_bose*R_down):
                            line[s0: s0+R_bose] = range(i+T,i+T+R_bose)
                            col[s0: s0+R_bose] = range(index+T,index+T+R_bose)
                            data[s0: s0+R_bose] = coef*V_cd[p-(m_c+m_d),j-(2*m_c+m_d)]*self.sign(p,j,basis[i])
                            s0 += R_bose
        bar = progressbar.ProgressBar()
        print('Complete jumps from claster to fermi bath')
        # Перескоки на кластере
        s0 = np.where(data == 0)[0][0]
        for i in bar(range(0,R_bose*R_up*R_down,R_bose*R_up)):
            iks = 0
            for j in range(m_c,m_c+m_d): # узлы кластера
                for p in neigbors_d_up[j]: # соседи 
                    # Спин вверх
                    # перескок с ванны на кластер
                    coef, function = self.up_down(j,p,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        line[s0: s0+R_bose*R_down] = range(i,i+R_bose*R_down)
                        col[s0: s0+R_bose*R_down] = range(index,index+R_bose*R_down)
                        data[s0: s0+R_bose*R_down] = coef*t_d*self.sign(p,j,basis[i])
                        s0 += R_bose*R_down
                                                    
                    # перескок с кластера на ванну
                    coef, function = self.up_down(p,j,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        line[s0: s0+R_bose*R_down] = range(i,i+R_bose*R_down)
                        col[s0: s0+R_bose*R_down] = range(index,index+R_bose*R_down)
                        data[s0: s0+R_bose*R_down] = coef*t_d*self.sign(p,j,basis[i])
                        s0 += R_bose*R_down
                    iks += 1
        bar = progressbar.ProgressBar()
        s0 = np.where(data==0)[0][0]
        # Спин вниз
        for i in bar(range(0,R_bose*R_down,R_bose)):
            iks = 0
            for j in range(2*m_c+m_d, 2*(m_c + m_d)):
                for p in neigbors_d_down[j]:
                    # перескок с ванны на кластер
                    coef, function = self.up_down(j,p,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        for T in range(0,R_bose*R_up*R_down,R_bose*R_down):
                            line[s0: s0+R_bose] = range(i+T,i+T+R_bose)
                            col[s0: s0+R_bose] = range(index+T,index+T+R_bose)
                            data[s0: s0+R_bose] = coef*t_d*self.sign(p,j,basis[i])
                            s0 += R_bose                      
                    # перескок с кластера на ванну
                    coef, function = self.up_down(p,j,basis[i])
                    index = indexes[''.join(map(str,function))]
                    if coef != 0:
                        for T in range(0,R_bose*R_up*R_down,R_bose*R_down):
                            line[s0: s0+R_bose] = range(i+T,i+T+R_bose)
                            col[s0: s0+R_bose] = range(index+T,index+T+R_bose)
                            data[s0: s0+R_bose] = coef*t_d*self.sign(p,j,basis[i])
                            s0 += R_bose
                iks += 1
        bar = progressbar.ProgressBar()
        print('Complete jumps on the claster')
        # бозоны
        s0 = np.where(data==0)[0][0]
        s = 0
        bb = Basis().fermi(m_c+m_d,n_down, n_up)
        bb = bb[:,m_c:m_c+m_d] + bb[:,2*m_c+m_d:]
        cf_np = np.asarray(bb.dot(gamma_bd))
        for i in bar(range(R_bose)):
            line1 = np.zeros(R_up*R_down*m_b*3)
            col1 = np.zeros(R_up*R_down*m_b*3)
            data1 = np.zeros(R_up*R_down*m_b*3)
            s = 0
            for q in range(2*(m_c+m_d),2*(m_c+m_d)+m_b):
                coef, index = self.up(q,n_max,basis[i],i)
                if coef != 0:
                    line1[s:s+R_up*R_down] = range(i,i+R_up*R_down*R_bose,R_bose)
                    col1[s:s+R_up*R_down] = range(index, index+R_up*R_down*R_bose,R_bose)
                    data1[s:s+R_up*R_down] = coef * cf_np[:,q-2*(m_c+m_d)]
                    s += R_up*R_down
                coef, index = self.down(q,n_max,basis[i],i)
                if coef != 0:
                    line1[s:s+R_up*R_down] = range(i,i+R_up*R_down*R_bose,R_bose)
                    col1[s:s+R_up*R_down] = range(index, index+R_up*R_down*R_bose,R_bose)
                    data1[s:s+R_up*R_down] = coef * cf_np[:,q-2*(m_c+m_d)]
                    s += R_up*R_down
            zeros = np.where(data1 == 0)[0]
            line1 = np.delete(line1, zeros)
            col1 = np.delete(col1,zeros)
            data1 = np.delete(data1,zeros)
            line[s0:s0+len(line1)] = line1
            col[s0:s0+len(col1)] = col1 
            data[s0:s0+len(data1)] = data1
            s0 += len(data1)
        print('Complete up/down on bose bath')        
        bar = progressbar.ProgressBar()           
        # Диагональные элементы 
        s0 = np.where(data==0)[0][0]
        for i in bar(range(0,R[0],R_bose)):
            # Диагональные элементы
            line[s0:s0+R_bose] = range(i,i+R_bose)  
            col [s0:s0+R_bose] = range(i,i+R_bose)
            data[s0:s0+R_bose] = sum(self.e_c[:m_c]*basis[i,0:m_c] + self.e_c[m_c:]*basis[i,(m_d+m_c):(m_d+2*m_c)]) + self.U_d*sum(basis[i,m_c:(m_d+m_c)]*basis[i,(m_d+2*m_c):2*(m_d+m_c)]) 
            for j in neigbors_d_up.keys():
                for k in neigbors_d_up[j]:
                    data[s0+i:s0+i+R_bose] += V_d*(basis[i,j]+basis[i,j+m_c])*(basis[i,k]+basis[i,k+m_c])
            s0 += R_bose
        bar = progressbar.ProgressBar()
        for i in bar(range(R_bose)):
            line[s0:s0+R_up*R_down] = range(i,i+R_up*R_down*R_bose,R_bose)
            col[s0:s0+R_up*R_down] = range(i,i+R_up*R_down*R_bose,R_bose)
            data[s0:s0+R_up*R_down] = sum(self.e_b*basis[i,-m_b:])
            s0 += R_up*R_down
        print('Complete bose bath diag element')
        zeros = np.where(data == 0)[0]
        line = np.delete(line, zeros)
        col = np.delete(col,zeros)
        data = np.delete(data,zeros)
        H = coo_matrix((data, (line, col)), shape=(1254400, 1254400))    
        return H