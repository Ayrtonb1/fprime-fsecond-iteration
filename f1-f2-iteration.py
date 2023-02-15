# code to generate best fit, then use the parameters from the fit to calculate f' and subsequently f" by Kramers-Kronig transform
# step 1: importing necessary values for the iterative process
import numpy as np
import matplotlib.pyplot as plt
import math
import xraydb

ene = np.loadtxt("DAFS_x_data_python.txt")
y_nsb = np.loadtxt("DAFS_y_data_python.txt")
f1_new = np.loadtxt("DAFS_f1_guess_python.txt")

fprime = xraydb.f1_chantler('Zn', ene) 
fsec = xraydb.f2_chantler('Zn', ene)

#%% importing atomic f' and f" as starting guesses. This dkk.kk() command does what I am trying to do. But it only works for some cases, so I am trying to make my own. The end result should look like that printed in the first figure from this section of the script 
import xraydb
from larch.xafs import diffkk 

dkk=diffkk(ene, y_nsb, z=30, edge='K', mback_kws={'e0':9658.6, 'order': 4})
dkk.kk()

plt.plot(dkk.energy,dkk.fpp, label = 'f"(E)')
plt.plot(dkk.energy, dkk.fp, label = "f'(E)")
plt.plot(dkk.energy,dkk.f1, label = "atomic f1")
plt.plot(dkk.energy,dkk.f2, label = "atomic f2")
plt.xlabel("energy / eV")
plt.ylabel("scattering factors")
plt.legend()
plt.show()
#%% fitting a smooth fit to 'Imodel' equation, generating parameters to calculate f'
from lmfit import Model

def intensity(en, abscorr, phi, beta, scale=1, slope=0, offset=0, fprime=-1, fsec=1):
    costerm = np.cos((phi)) + beta*fprime
    sinterm = np.sin((phi)) + beta*fsec
    return scale * (costerm**2 + sinterm**2)*abscorr + slope*en + offset

imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec', 'abscorr'])

params = imodel.make_params(scale=1, offset=0, slope=0, beta=0.1, phi= 5.5)
# We can constrain these parameters as desired below
abscorr = 1
params = imodel.make_params(scale=1, offset=0, slope=0, beta=0.1, phi= 5.5)
params['scale'].min = 0.005 # force scale to be positive
params['scale'].max = 5
params['slope'].vary = False
params['phi'].min = 0.5
params['phi'].max = 10
params['offset'].min = -5
params['offset'].max = 5
params['beta'].min = 0.001
params['beta'].max = 0.5
init_value = imodel.eval(params, en=ene, fprime=fprime, fsec=fsec, abscorr = abscorr)

result = imodel.fit(y_nsb, params, en=ene, fprime=fprime, fsec=fsec, abscorr = abscorr)
# print(' I0 = ',I0, '\n','phi = ', phi, '\n','beta =',beta, '\n','Ioff =', Ioff)
print(result.fit_report())

plt.plot(ene, y_nsb, label='data')
plt.plot(ene, (result.best_fit), '--', label='best fit (lmfit)')
# plt.plot(ene, best_fit , '--', label = 'best fit (least sq.)')
plt.xlabel("Energy / eV")
plt.ylabel("Intensity (bkg subtracted + normalised)")
plt.legend(loc='upper right')
plt.show()

phi = result.params.get('phi').value
beta = result.params.get('beta').value
I0 = result.params.get('scale').value
Ioff = result.params.get('offset').value
print(I0, phi, beta, Ioff) #THESE are the parameters we will use 
#%%
# using the parameters calculated above to calculate a better guess of f', also using the atomic f" as a starting point in this equation below
import sympy as sy

def f1_guess(f2, I, I0, phi, beta, Ioff,abscorr):
    f1_guess = (1/beta) * -(np.sqrt(((I-Ioff)/I0*abscorr) - (math.sin(math.radians(phi))+(beta*fsec))**2) - math.cos(math.radians(phi)))
    return f1_guess

# y_nsb_new = np.interp(ene,ene-3,y_nsb)
f1_new = f1_guess(fsec, y_nsb, I0, phi, beta, Ioff,abscorr)
f1_minus = (fprime - f1_new)
plt.plot(ene, (f1_minus), label='new f1')
plt.plot(ene, fprime, label = 'starting guess f1')
plt.xlabel("Energy / eV")
plt.legend(loc='lower right')
plt.show()
#%%
# using the f' calculated above to generate a better guess for f" by KK transform
from scipy.integrate import quad_vec
ene = np.array(ene)
ene_dash = np.array(ene)
# i = 1

f2KK = lambda ene_dash : ((f1_new - fprime)/ ((ene_dash-1)**2 - (ene[i])**2))

f2KK_arr = []
f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))

fsecond_KK = (fsec - ((2*ene/math.pi) * f2KK_arr))
print(f2KK_arr)

plt.plot(ene,fsecond_KK, label = 'new f2')
plt.plot(ene,fsec, label = 'f2')
plt.plot(ene,f1_minus, label = 'new f1')
plt.plot(ene,fprime, label = 'f1')
plt.plot(dkk.energy,dkk.fpp, label = 'f"(E)')
plt.plot(dkk.energy, dkk.fp, label = "f'(E)")
plt.legend()
plt.show()

# Note: the true f' and f" should match the overall shape of the atomic f' and f", differing only in their fine structure. However, they instead also differ in their magnitude, which is the issue
#%%
#The process is iterated, but the problem arises as the f' and f" diverge away from the atomic values, to a point where they become NaN/infinite because the f' equation requires that you do a square root, but either f" eventually becomes negative or the parameters become negative, such that the iteration cannot proceed.

for i in range(1,10):
    print(i)
    imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec'])

    params = imodel.make_params(scale=I0, offset=Ioff, slope=0, beta=beta, phi=phi, abscorr = abscorr)
    params['scale'].min = 0.05  # force scale to be positive
    params['scale'].max = 2
    params['slope'].vary = False
    # params['phi'].vary =False
    params['phi'].min = 0
    params['phi'].max = 3* math.pi
    params['offset'].min = -5
    params['offset'].max = 5
    params['abscorr'].vary = False
    # params['abscorr'].min = 0.5
    # params['abscorr'].max = 1.5
    params['beta'].min = 0.001
    params['beta'].max = 0.5
    init_value = imodel.eval(params, en=ene, fprime=f1_minus, fsec=fsecond_KK)

    result = imodel.fit(y_nsb, params, en=ene, fprime=f1_minus, fsec=fsecond_KK)
    print(result.fit_report())

    phi = result.params.get('phi').value # - math.pi
    beta = result.params.get('beta').value
    I0 = result.params.get('scale').value
    Ioff = result.params.get('offset').value
    abscorr = result.params.get('abscorr').value 
    print(I0, phi, beta, Ioff, abscorr)

    ##%%
    plt.plot(ene, y_nsb, label='data')
    plt.plot(ene, (result.best_fit), '--', label='best fit (lmfit)')
    plt.xlabel("Energy / eV")
    plt.ylabel("Intensity (bkg subtracted + normalised)")
    plt.legend(loc='upper right')
    plt.show()

    f1_new = f1_guess(fsecond_KK, y_nsb, I0, phi, beta, Ioff, abscorr)
    f1_minus = fprime - f1_new

    plt.plot(ene, (f1_minus), label='new f1 ')
    plt.plot(ene, fprime, label = 'starting guess f1')

    plt.xlabel("Energy / eV")
    plt.legend(loc='upper right')
    plt.show()

    f2KK = lambda ene_dash : ((f1_new - fprime)/ ((ene_dash-1)**2 - (ene[i])**2))

    f2KK_arr = []
    f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))

    fsecond_KK = (fsec - ((2*ene/math.pi) * f2KK_arr)) 

    plt.plot(ene,fsecond_KK, label = 'new f2')
    plt.plot(ene,fsec, label = 'f2')
    plt.plot(ene,f1_minus, label = 'new f1')
    plt.plot(ene,fprime, label = 'f1')
    plt.plot(dkk.energy,dkk.fpp, label = 'f"(E)')
    plt.plot(dkk.energy, dkk.fp, label = "f'(E)")

    plt.legend()
    plt.show()

    plt.plot(ene,fsecond_KK, label = 'new f2')
    plt.plot(ene,f1_minus, label = 'new f1')
    plt.legend()
    plt.show()
# #%%
#     i = str(i)
#     fsec_name = (i + ' fsec ' + filename)
#     np.savetxt((fsec_name),np.column_stack((ene,fsecond_KK)))
#     # np.savetxt(file_name_2, np.column_stack((ene, I_max_norm)))

