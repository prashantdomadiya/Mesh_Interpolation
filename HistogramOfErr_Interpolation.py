import numpy as np
from os.path import join
import matplotlib.pyplot as plt

Rigname='50025'
path='/media/prashant/DATA/MyCodes/Interpolation/'+Rigname+'/'

#ERR=np.loadtxt(path+Rigname+'_NormalErr.txt',delimiter=',')
ERR=np.loadtxt(path+Rigname+'_AreaErr.txt',delimiter=',')

"""
#ERR=(ERR-np.min(ERR))/np.max(ERR)
mu=np.mean(ERR1)
sd=np.std(ERR1)
ERR=np.zeros(np.shape(ERR1))
for i in range(len(ERR1)):
    if ERR1[i]>(3*sd):
        ERR[i]=3*sd
    else:
        ERR[i]=ERR1[i]
"""
mu=np.mean(ERR)
sd=np.std(ERR)



x=plt.hist(ERR,2000)
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 10,
        }

plt.xlabel("Error")
plt.ylabel("Number of faces")
plt.xlim(0,0.16)
#plt.ylim(0,800)
#plt.title("$\mu$"+str(mu))
plt.text(0.11, 400, r'$\mu_a=$'+('%.4f'% mu), fontdict=font)
plt.text(0.11, 350, r'$\sigma_a=$'+('%.4f'% sd), fontdict=font)
plt.show()
