
# coding: utf-8

# In[1]:

from math import factorial as fact
import numpy as np
import fermi_basis as fb
import progressbar
import copy

# Опертор рождения * уничтожения
def up_down(number_to_up, number_to_down, func):
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
def sign(k,l,m,func):
    '''Функция определяет знак перескока'''
    if k > (m-1):
        (k,l) = (k-m,l-m)
    if sum(func[k+1:l])%2:
        return 1
    else:
        return -1

# Заполнение матрицы
def hamiltonian(U,V,ee,m,n_up,n_down):
    '''Возвращает гамильтонову матрицу.
    U - взаимодействие на узле, V - величина перескока, ee - затравочная энергия, V и ee задаются массивом длинной 2*(m-1) и (m-1)
    соответственно, m - число узлов, n_up - число спинов "вверх", n_dowm - число спинов "вниз".
    '''
    fi = fb.basis_spin(m,n_up,n_down)
    # Переводим тип матрицы в целочисленную для корректной работы метода ''.join(...)
    fi = fi.astype(int)
    H = np.zeros((len(fi),len(fi)))
    
    # Создаем словарь для определения индекса функции 
    key = [''.join(map(str,fi[i])) for i in range(fi.shape[0])]
    value = range(len(key))
    indexes = dict(zip(key,value))
    
    bar = progressbar.ProgressBar()
    for i in bar(range(len(fi))):
        for j in range(1,m,1):
            # Перескоки с ванны на узел и обратно
            H[i,indexes[''.join(map(str,up_down(0,j,fi[i])[1]))]] += V[j-1]*sign(0,j,m,fi[i])*up_down(0,j,fi[i])[0]
            H[i,indexes[''.join(map(str,up_down(m,j+m,fi[i])[1]))]] += V[j+m-2]*sign(m,j+m,m,fi[i])*up_down(m,j+m,fi[i])[0]
            H[i,indexes[''.join(map(str,up_down(j,0,fi[i])[1]))]] += V[j-1]*sign(0,j,m,fi[i])*up_down(j,0,fi[i])[0]
            H[i,indexes[''.join(map(str,up_down(j+m,m,fi[i])[1]))]] += V[j+m-2]*sign(m,j+m,m,fi[i])*up_down(j+m,m,fi[i])[0]
            H[i,i] += ee[j-1] * (fi[i][j] + fi[i][j+m])

        # One-site interactive
        H[i,i] += U * fi[i][0]*fi[i][m]
    return H

