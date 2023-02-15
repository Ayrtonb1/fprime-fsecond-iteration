import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from glob import glob
from hdf5_lib import *

#Things to change for each dataset:
    #change filename (1)
    #change configurations per file/sys.argvs (2)
    #change angle correction and dr range (3)
    #change / interpolate I0 array length (4)
    #Change f1/f2 chantler edges (5)
    #Check that bkg function is accurate. If not play around with the points 
#%%
# filename = '553990_ZnAlPhosphate_grazing_stepscanng_after_move_4693.h5' 
filename = '553975_ZnFerrite_grazing_stepscan_4683.h5'
# # filename = '536677_Fe3O4_3000pt_100ms_thresh3.5.hdf5'
# filename = '553988_NiFerrite_grazing_stepscanpscang_after_move_4691.h5'
# filename = '553989_CoFerrite_grazing_stepscanpscang_after_move_4692.h5'

hdf_print_names(filename)
"""""
entry
entry/data
entry/data/data
entry/instrument
entry/instrument/NDAttributes
entry/instrument/NDAttributes/AcqPeriod
entry/instrument/NDAttributes/CameraManufacturer
entry/instrument/NDAttributes/CameraModel
entry/instrument/NDAttributes/DriverFileName
entry/instrument/NDAttributes/ImageCounter
entry/instrument/NDAttributes/ImageMode
entry/instrument/NDAttributes/MaxSizeX
entry/instrument/NDAttributes/MaxSizeY
entry/instrument/NDAttributes/NDArrayEpicsTSSec
entry/instrument/NDAttributes/NDArrayEpicsTSnSec
entry/instrument/NDAttributes/NDArrayTimeStamp
entry/instrument/NDAttributes/NDArrayUniqueId
entry/instrument/NDAttributes/NumExposures
entry/instrument/NDAttributes/NumImages
entry/instrument/NDAttributes/TIFFImageDescription
entry/instrument/NDAttributes/TriggerMode
entry/instrument/detector
entry/instrument/detector/NDAttributes
entry/instrument/performance
entry/instrument/performance/timestamp
"""

XRD=hdf_get_item(filename,'entry/data/data')
z = XRD[0,:,:]

image=hdf_get_item(filename,'entry/data/data')

n_energypoints= shape(XRD)[0]
n_column = shape(XRD) [1]
n_row = shape(XRD) [2]

print(n_energypoints)
y=np.linspace(0,z.shape[0],(z.shape[0]));
x=np.linspace(0,z.shape[1],(z.shape[1]));
#48, 97, 145
buf=z[45,:]; index1=np.where(buf==max(buf));
buf=z[97,:]; index2=np.where(buf==max(buf));
buf=z[145,:]; index3=np.where(buf==max(buf));
print(index1,index2,index3)
#%%
# print((n_energypoints))
print(z.shape)
#%% Defining Functions
e = 2.718

def two_d(two_theta,energy): #this is a function that calculates the two d values based on an input two theta array + energy single value, which returns the array two d values in Ang. may be worth improving the 12398 constant values accuracy
    out = (12398./energy)/np.sin(np.radians(two_theta/2))
    return out

def theta(two_d,energy):
    out = math.asin(12938. / (two_d * energy))
    return out

def norm_I(intensity,I_0): # this function will normalise the intensity by multiplying a single value I0 by an array of XRD intensity values
    out = (intensity/I_0)
    return out

def abs_corr(abs_tot,theta):
    out = (1-e**(-2*abs_tot))/(abs_tot * np.sin(theta)) #function to calculate D
    return out

def lorentz_corr(theta):
    out = 2*(np.sin(theta)**2)*np.cos(theta) #function to calculate L, then must multiply the observed intensity by L/D
    return out

def gaus(x,a,x0,sigma): #initial attempts to fit a Gaussian to each peak
    out = a*np.exp(-(x-x0)**2/(2*sigma**2))
    return out

def gaus_bkg(x,a,x0,sigma,bkg): #initial attempts to fit a Gaussian to each peak
    out = a*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
    return out

def findCircle(x1, y1, x2, y2, x3, y3) :

               x12 = x1 - x2
               x13 = x1 - x3
               y12 = y1 - y2
               y13 = y1 - y3
               y31 = y3 - y1
               y21 = y2 - y1
               x31 = x3 - x1
               x21 = x2 - x1

               sx13 = pow(x1, 2) - pow(x3, 2);
               sy13 = pow(y1, 2) - pow(y3, 2);
               sx21 = pow(x2, 2) - pow(x1, 2);
               sy21 = pow(y2, 2) - pow(y1, 2);

               f = (((sx13) * (x12) + (sy13) *
                              (x12) + (sx21) * (x13) +
                              (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))));    
               g = (((sx13) * (y12) + (sy13) * (y12) +
                              (sx21) * (y13) + (sy21) * (y13)) //
                              (2 * ((x31) * (y12) - (x21) * (y13))));
               c = (-pow(x1, 2) - pow(y1, 2) -
                              2 * g * x1 - 2 * f * y1);
               
               # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
               # where centre is (h = -g, k = -f) and
               # radius r as r^2 = h^2 + k^2 - c
               h = -g;
               k = -f;
               sqr_of_r = h * h + k * k - c;

               # r is the radius
               r = round(sqrt(sqr_of_r), 5);
               print("Centre = (", h, ", ", k, ")");
               print("Radius = ", r);
               return h,k,r 

def radial_profile(data, center):
    # probably faster implementation here https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr;
    return radialprofile

def find_data_in_ROI(x,y,z, Cen_x,Cen_y):
    coeff_line_up = np.polyfit([Cen_x, max(x)],[Cen_y, max(y)],1);
    coeff_line_down=np.polyfit([Cen_x, max(x)],[Cen_y, min(y)],1);
    line_up=np.polyval(coeff_line_up,x); line_down=np.polyval(coeff_line_down,x);
    reduced_data = z
    for index1 in range(len(x)):
        for index2 in range(len(y)):
            if y[index2] > line_up[index1] :
                reduced_data[index2,index1]=0;
            if y[index2] < line_down[index1] :
                reduced_data[index2,index1]=0;
    return reduced_data

# Basic filtering on which indexes are axeptable to work with preferential orientations !!!
if (np.abs(index2[0])-index1[0] + np.abs(index2[0]-index3[0])) < 10: #10 pixels is an arbitrary width   
    Cen_x, Cen_y, rad = findCircle(int(index1[0]),48, int(index2[0]),97, int(index3[0]),145)
else:
    print('Automatic estimation of the centre failed; Please manually select three points on a diffraction ring of interest and press return')
    bufsel =plt.ginput(-1)
    bufsel=np.asarray(bufsel);
    Cen_x, Cen_y, rad = findCircle(int(bufsel[0,0]),int(bufsel[0,1]), int(bufsel[1,0]),int(bufsel[1,1]), int(bufsel[2,0]),int(bufsel[2,1]))

extracted_xrd=np.ones([len(x),np.shape(XRD)[0]+1])
#%%
import sys
import math
import heapq
import statistics
from scipy import optimize

# mainfile = sys.argv[1]
mainfile = "553975_b18.dat" 
energies = []
file_names = [] #this will be a list with 371 pilatus file names
I_0 = []
exafs = []
with open(mainfile, "r") as mythen_file: #opening the XAS file called mainfile and appending the energies and corresponding file names, as well as the I_0 as 3 lists
    for line in mythen_file:
        if line[0] != "#":
            split_line = line.split()  # list of space-separated substrings
            energies.append(float(split_line[0]))
            file_names.append(split_line[-1])
            I_0.append(split_line[1])
            exafs.append(split_line[4])
ene = np.array(energies).astype(float)
I_0 = np.array(I_0).astype(float)
exafs = np.array(exafs).astype(float)

# exafs = exafs[0:n_energypoints]
I_0 = I_0[0:n_energypoints]
# ene = ene[0:n_energypoints]

exafs = np.interp(ene[0:n_energypoints],ene,exafs)
ene = np.interp(ene[0:n_energypoints],ene,ene)

print(len(exafs))
print(len(ene))

# y_new_shift = np.interp(ene,ene-7,y_new)

plt.plot(ene,exafs)
plt.show()

#%%
for index in range(n_energypoints):
    z=XRD[index,:,:]
    y=np.linspace(0,z.shape[0],(z.shape[0]));
    x=np.linspace(0,z.shape[1],(z.shape[1]));
    reduced_data = find_data_in_ROI(x,y,z, Cen_x,Cen_y)
    buffer = radial_profile(reduced_data, [Cen_x,Cen_y])    
    buffer[np.isnan(buffer)] = 0
    buffer=buffer[np.abs(Cen_x -1 ):-1]; # cen_x - 1?
    extracted_xrd[:,index]=buffer
    # print('iteration', index)
#%%
fig, ax = plt.subplots(1)
ax.pcolormesh(x, y, z, cmap='jet')
circle = plt.Circle((Cen_x, Cen_y), rad, edgecolor='r', fill=False)
ax.add_patch(circle)
ax.plot([Cen_x, max(x)],[Cen_y, max(y)],'w')
ax.plot([Cen_x, max(x)],[Cen_y, min(y)],'w')
ax.set_xlim([0,max(x)]); ax.set_ylim([0,max(y)]);
plt.show()

fig, ax = plt.subplots(1)
ax.pcolormesh(x, y, reduced_data, cmap='jet')
plt.show()
# The intensity might need rescaling for the different arc of integration(?)
#%%
#2 theta calibration / alignment step
x1 = [214, 337, 930, 808]
y = [24.988, 29.383, 51.266, 46.836]

coeff = np.polyfit(x1,y,1)
x2 = np.polyval(coeff,x)

def bkg_func(x,a,b):
    model = a*x**2 + b
    return model

def resid(p0):
    return I - bkg_func(x2,p0[0],p0[1])

plt.plot(x2, extracted_xrd)
plt.show()
#%%
print(range(n_energypoints))
#%%
import sys
n = len(sys.argv)
d = sys.argv[2:n] #this is a list of length n containing the d values corresponding to the desired bragg reflections

for dhkl in d: #take a given d value as specified by system argument, and use it to obtain a series of 'max_y' values within a given angle range,
    dhkl = float(dhkl)
    min_y_avg = []
    max_y_avg = []
    area_arr  = []
    for i in range(n_energypoints):
         I = extracted_xrd[:,i]
         x = x2 + 1.5

         p0 = [1,1]
         popt, pcov = optimize.leastsq(resid, p0)
         a = popt[0]
         b = popt[1]
         
         smooth_fit = bkg_func(x, a, b) 
         I_bkg_sub = I - smooth_fit


         d = (two_d(x, ene[i])) / 2  # converts two theta to d using the function at top
         dr = (dhkl / (dhkl * 20))  # define a d specific range for the np.where command to use as its range
         index_d = np.where((d >= (dhkl-dr)) & (d <= (dhkl+dr)))  # stores the number of values of d, that is, the index of d, between the specified range, i.e. the 12th, 13th, 14th value
         d_select = d[index_d[0]]  # stores a list of values for these specified values of index_d in the d values, effectively giving us the actual numerical values within that range, NOT JUST THE MAX values
         y_select = I_bkg_sub[index_d[0]]# must also specify this for y, so lists are same length
         
         n = len(d_select)
         mean = sum(d_select) / n
         bkg = min(y_select)
         sigma = np.sqrt(sum((y_select-bkg)*(d_select-mean)**2) / sum(y_select) )
         amp = max(y_select) - min(y_select)
         p0= [amp, mean, sigma, bkg]
         popt, _ = optimize.curve_fit(gaus_bkg, d_select, y_select,p0,maxfev = 100000)
         plt.plot(d_select, y_select, label=str(i))

         area = ((popt[0]) * (popt[2])) / 0.3989
         area_arr.append(area)
         min_y_avg.append(statistics.mean(heapq.nsmallest(3, y_select)))
         max_y_avg.append(statistics.mean(heapq.nlargest(3, y_select)))
    
    I_max = np.array(max_y_avg)
    I_min = np.array(min_y_avg)
    I_bkg = np.subtract(I_max, I_min)  # subtracting the background, printing 1 value of I for each scan # 250 times and storing in an array
    
    area_arr = np.array(area_arr)
    plt.title(dhkl)
    plt.show()

#%%
    area_norm = norm_I(area_arr,I_0)
    I_max_norm = norm_I(I_bkg,I_0)# normalise maximum intensities by dividing by I_0 using the norm_I function
    abs_tot = np.log(1/area_norm)
    
    theta_range = (math.pi/2) # is this correct??????????
    D = abs_corr(abs_tot, theta_range)  # absorption correction as quoted in Pickering et al.
    #L = lorentz_corr(theta_range)  # lorentzian correction as quoted in Pickering et al.
    abs_tot = abs_tot / D
    I_max_norm = I_max_norm / D
    
    # plotting DAFS spectrum and 'absorption spectrum' 
    # plt.plot(ene, abs_tot, label='corrected I vs. E') # plotting the new absorption graph
    # plt.xlabel("Energy / eV")
    # plt.ylabel("Corrected intensity / Arb. units")
    # plt.show()

    hkl = str(dhkl)
    mainfile = str(mainfile)

    plt.plot(ene,I_max_norm)
    plt.title(mainfile)
    plt.xlabel("Energy / eV")
    plt.ylabel("Intensity / Arb. units")
    plt.show()

    plt.figure()
    y_new = (area_norm - min(area_norm))/(max(area_norm) - min(area_norm))
    plt.plot(ene, (area_norm - min(area_norm))/(max(area_norm) - min(area_norm)), label='corrected area vs. E')# plotting the new absorption graph
    plt.xlabel("Energy / eV")
    plt.ylabel("Corrected area comparison / Arb. units")
    
    file_name = (filename + hkl + '.dat')
    file_name_2 = (filename + hkl + '.dat_I_max_norm')
    np.savetxt(file_name, np.column_stack((ene, (area_norm - min(area_norm))/(max(area_norm) - min(area_norm)))))
    np.savetxt(file_name_2, np.column_stack((ene, I_max_norm)))
    plt.show()   

#%%
import xraydb

fprime = xraydb.f1_chantler('Zn', ene) 
fsec = xraydb.f2_chantler('Zn', ene)
plt.plot(ene, fprime)
plt.plot(ene, fsec)
plt.xlabel("Energy / eV")
plt.ylabel("Intensity / Arb. units")
plt.legend(['f1', 'f2'])
plt.gca().set_xscale('linear')
plt.show()
#%%
# using leastsquares approach
y_new_shift = np.interp(ene,ene-7,y_new)

m = np.polyfit(ene[0:200],y_new_shift[0:200],1)
bkg = np.polyval(m,ene)
y_nsb = y_new_shift - bkg # shifted data (7 eV with background removal)

n = np.polyfit(ene[300:-1], y_new_shift[300:-1],1)
post_bkg = np.polyval(n,ene)
y_nsb_2 = y_new_shift - post_bkg

#%%
t = 1
mew = np.log(I_0/exafs) 
# theta_new = theta(d,ene)
print(dhkl)
sin_theta = []
for i in range(n_energypoints):
    sin_theta.append(12938. / (2* dhkl * ene[i]))
print(len(sin_theta))

check = mew / sin_theta
print(check)

abscorr = (1 - e**((-2*mew*t)/sin_theta)) / (2*mew)

# plt.plot(ene,I_0)
# plt.plot(ene,y_new_shift, label = 'y_new')
# plt.plot(ene,mew, label = 'mew')
plt.plot(ene,abscorr,label = 'abscorr')
plt.legend()
plt.show()
#%%
# def of fucntions for model and residuals
def I_model(f1,f2,I0,phi,beta,Ioff,abscorr):
    model = I0 * ((np.cos(phi)+ (beta * f1))**2+ (np.sin(phi)+(beta*f2))**2) * abscorr + Ioff # * abs_corr * lorentz_corr + Ioff
    return model

def residual(p0):
    return y_nsb - I_model(fprime, fsec, p0[0], p0[1], p0[2], p0[3], abscorr)

y_nsb = y_nsb
p0 = [0.8, np.pi/2, np.pi/2, 0.1, 1] # p0 built with sensible initial values
popt, pcov = optimize.leastsq(residual, p0)

I0   = popt[0]               #obtained from regular EXAFS)
phi  = popt[1]               #How do we obtain the phase?
beta = popt[2]               #alpha / I0     
Ioff = popt[3]               #correction factor of the form a1 + a2E. Energy dependent if a2 is set to 0
# abscorr = popt[4]               #thickness (unstable fitting parameter - see J. Cross)
# dE   = popt[5]               #related to absorption correction in some way?

print(' I0 = ',I0, '\n','phi = ', phi, '\n','beta = ',beta, '\n','Ioff = ', Ioff)

best_fit = I_model(fprime, fsec, I0, phi, beta, Ioff, abscorr) 

plt.plot(ene,y_nsb,'orange')
plt.plot(ene, best_fit , '--')
plt.legend(['background subtracted data','best fit'])
plt.show()

#%%
import xraydb
from larch.xafs import diffkk 

dkk=diffkk(ene, y_nsb, z=30, edge='K', mback_kws={'e0':9658.6, 'order': 4})
# 'whiteline': 20? 
dkk.kk()

plt.plot(dkk.energy,dkk.fpp, label = 'f"(E)')
plt.plot(dkk.energy, dkk.fp, label = "f'(E)")
plt.plot(dkk.energy,dkk.f1, label = "atomic f1")
plt.plot(dkk.energy,dkk.f2, label = "atomic f2")
plt.xlabel("energy / eV")
plt.ylabel("scattering factors")
plt.legend()
plt.show()
#%%
np.savetxt(filename + sys.argv[2] + ".dat", np.column_stack((dkk.energy,dkk.fpp)))
#%%
from lmfit import Model

def intensity(en, abscorr, phi, beta, scale=1, slope=0, offset=0, fprime=-1, fsec=1):
    # costerm = np.cos(math.radians(phi)) + beta*fprime
    # sinterm = np.sin(math.radians(phi)) + beta*fsec
    costerm = np.cos((phi)) + beta*fprime
    sinterm = np.sin((phi)) + beta*fsec
    return scale * (costerm**2 + sinterm**2)*abscorr + slope*en + offset

imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec', 'abscorr'])

params = imodel.make_params(scale=1, offset=0, slope=0, beta=0.1, phi= 5.5)
params['scale'].min = 0  # force scale to be positive
# params['scale'].max = 2
params['slope'].vary = False
# # params['phi'].vary =False
# params['phi'].min = 0.5
# params['phi'].max = 3* math.pi
# params['offset'].min = -5
# params['offset'].max = 5
# # params['offset'].vary = False
# # params['abscorr'].vary = False
# # params['abscorr'].min = 0.5
# # params['abscorr'].max = 1.5
params['beta'].min = 0.001
params['beta'].max = 0.5
init_value = imodel.eval(params, en=ene, fprime=fprime, fsec=fsec, abscorr = abscorr)

result = imodel.fit(y_nsb, params, en=ene, fprime=fprime, fsec=fsec, abscorr = abscorr)
print(' I0 = ',I0, '\n','phi = ', phi, '\n','beta =',beta, '\n','Ioff =', Ioff)
print(result.fit_report())

plt.plot(ene, y_nsb, label='data')
# plt.plot(ene, init_value, label='initial fit')
plt.plot(ene, (result.best_fit), '--', label='best fit (lmfit)')
plt.plot(ene, best_fit , '--', label = 'best fit (least sq.)')
plt.xlabel("Energy / eV")
plt.ylabel("Intensity (bkg subtracted + normalised)")
plt.legend(loc='upper right')
plt.show()

phi = result.params.get('phi').value # - math.pi
beta = result.params.get('beta').value
I0 = result.params.get('scale').value
Ioff = result.params.get('offset').value
# abscorr = result.params.get('abscorr').value 
print(I0, phi, beta, Ioff)
#%%
import sympy as sy

def f1_guess(f2, I, I0, phi, beta, Ioff,abscorr):
    # f1_guess = (1/beta) * -(np.sqrt(((I-Ioff)/I0*abscorr) - (math.sin((phi))+(beta*fsec))**2) - math.cos((phi))) 
    f1_guess = (1/beta) * -(np.sqrt(((I-Ioff)/I0*abscorr) - (math.sin(math.radians(phi))+(beta*fsec))**2) - math.cos(math.radians(phi)))
    return f1_guess

# y_new_shift = np.interp(ene,ene-7,y_new)
y_nsb_new = np.interp(ene,ene-3,y_nsb)

f1_new = f1_guess(fsec, y_nsb_new, I0, phi, beta, Ioff,abscorr)

print(f1_new)

f1_minus = (fprime - f1_new) 
# plt.plot(ene,f1_new)
plt.plot(ene, (f1_minus), label='new f1')
plt.plot(ene, fprime, label = 'starting guess f1')

plt.xlabel("Energy / eV")
plt.legend(loc='lower right')
plt.show()

# np.savetxt("DAFS_x_data_python.txt", np.column_stack(ene))
# np.savetxt("DAFS_y_data_python.txt", np.column_stack(y_nsb))
# np.savetxt("DAFS_f1_guess_python.txt", np.column_stack(ene))
#%%
# import numpy as np
# ene = np.loadtxt("DAFS_x_data_python.txt")
# y_nsb = np.loadtxt("DAFS_y_data_python.txt")
# f1_new = np.loadtxt("DAFS_f1_guess_python.txt")

#%%
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.integrate import trapezoid as trapz
from scipy.integrate import cumulative_trapezoid as ctrapz
from scipy.integrate import simpson as simp
from scipy.integrate import romb

ene = np.array(ene)
ene_dash = np.array(ene)

f2KK = lambda ene_dash : ((f1_new - fprime)/ ((ene_dash-1)**2 - (ene[i])**2))

f2KK_arr = []
f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))

fsecond_KK = (fsec - ((2*ene/math.pi) * f2KK_arr))

# def f2KK(ene_dash,ene,f1_new,fprime):
#     return ((f1_new - fprime)/ ((ene_dash)**2 - (ene)**2))

# f2KK_arr = []
# for i in range(n_energypoints):
#     f2KK_arr.append(trapz(f2KK((ene_dash-1),ene[i],f1_new,fprime),(ene_dash-1)))

# f = f1_minus + fsecond_KK
print(f2KK_arr)

plt.plot(ene,fsecond_KK, label = 'new f2')
plt.plot(ene,fsec, label = 'f2')
plt.plot(ene,f1_minus, label = 'new f1')
plt.plot(ene,fprime, label = 'f1')
plt.plot(dkk.energy,dkk.fpp, label = 'f"(E)')
plt.plot(dkk.energy, dkk.fp, label = "f'(E)")

plt.legend()
plt.show()


#%%
# def I_model(f1,f2,I0,phi,beta,Ioff,abscorr):
#     model = I0 * ((np.cos(phi)+ (beta * f1))**2+ (np.sin(phi)+(beta*f2))**2) * abscorr + Ioff # * abs_corr * lorentz_corr + Ioff
#     return model

# def residual(p0):
#     return y_nsb - I_model(fprime, fsec, p0[0], p0[1], p0[2], p0[3], p0[4])

# y_nsb = y_nsb
# p0 = [I0, phi, beta, Ioff, 1] # p0 built with sensible initial values
# popt, pcov = optimize.leastsq(residual, p0)

# I0   = popt[0]               #obtained from regular EXAFS)
# phi  = popt[1]               #How do we obtain the phase?
# beta = popt[2]               #alpha / I0     
# Ioff = popt[3]               #correction factor of the form a1 + a2E. Energy dependent if a2 is set to 0
# abscorr = popt[4]               #thickness (unstable fitting parameter - see J. Cross)
# # dE   = popt[5]               #related to absorption correction in some way?

# print(' I0 = ',I0, '\n','phi = ', phi, '\n','beta = ',beta, '\n','Ioff = ', Ioff, '\n','abscorr = ', abscorr)

# best_fit = I_model(f1_minus, fsecond_KK, I0, phi, beta, Ioff, abscorr)

# plt.plot(ene,y_nsb,'orange')
# plt.plot(ene, best_fit , '--')
# plt.legend(['background subtracted data','best fit'])
# plt.show()


#%%

# # #%%

# for i in range(1,10):
#     print(i)
#     imodel = Model(intensity, independent_vars=['en', 'fprime', 'fsec'])

#     params = imodel.make_params(scale=I0, offset=Ioff, slope=0, beta=beta, phi=phi, abscorr = abscorr)
#     params['scale'].min = 0.05  # force scale to be positive
#     params['scale'].max = 2
#     params['slope'].vary = False
#     # params['phi'].vary =False
#     params['phi'].min = 0
#     params['phi'].max = 3* math.pi
#     params['offset'].min = -5
#     params['offset'].max = 5
#     params['abscorr'].vary = False
#     # params['abscorr'].min = 0.5
#     # params['abscorr'].max = 1.5
#     params['beta'].min = 0.001
#     params['beta'].max = 0.5
#     init_value = imodel.eval(params, en=ene, fprime=f1_minus, fsec=fsecond_KK)

#     result = imodel.fit(y_nsb, params, en=ene, fprime=f1_minus, fsec=fsecond_KK)
#     # print(' I0 = ',I0_new, '\n','phi = ', phi_new, '\n','beta =',beta_new, '\n','Ioff =', Ioff_new)
#     print(result.fit_report())

#     phi = result.params.get('phi').value # - math.pi
#     beta = result.params.get('beta').value
#     I0 = result.params.get('scale').value
#     Ioff = result.params.get('offset').value
#     abscorr = result.params.get('abscorr').value 
#     print(I0, phi, beta, Ioff, abscorr)

#     ##%%
#     plt.plot(ene, y_nsb, label='data')
#     # plt.plot(ene, init_value, label='initial fit')
#     plt.plot(ene, (result.best_fit), '--', label='best fit (lmfit)')
#     # plt.plot(ene, best_fit , '--', label = 'best fit (least sq.)')
#     plt.xlabel("Energy / eV")
#     plt.ylabel("Intensity (bkg subtracted + normalised)")
#     plt.legend(loc='upper right')
#     plt.show()

#     f1_new = f1_guess(fsecond_KK, y_nsb, I0, phi, beta, Ioff, abscorr)

#     f1_minus = fprime - f1_new

#     # plt.plot(ene, (f1_minus_new), label='new new f1 ')
#     plt.plot(ene, (f1_minus), label='new f1 ')
#     plt.plot(ene, fprime, label = 'starting guess f1')

#     plt.xlabel("Energy / eV")
#     plt.legend(loc='upper right')
#     plt.show()
#     ##%%
#     f2KK = lambda ene_dash : ((f1_new - fprime)/ ((ene_dash-1)**2 - (ene[i])**2))

#     f2KK_arr = []
#     f2KK_arr, err = (quad_vec(f2KK,(ene_dash[0]),(ene_dash[-1])))

#     fsecond_KK = (fsec - ((2*ene/math.pi) * f2KK_arr)) 

#     plt.plot(ene,fsecond_KK, label = 'new f2')
#     # plt.plot(ene,fsecond_KK_new, label = 'newest f2')
#     plt.plot(ene,fsec, label = 'f2')
#     plt.plot(ene,f1_minus, label = 'new f1')
#     # plt.plot(ene,f1_minus_new, label = 'newest f1')
#     plt.plot(ene,fprime, label = 'f1')
#     plt.plot(dkk.energy,dkk.fpp, label = 'f"(E)')
#     plt.plot(dkk.energy, dkk.fp, label = "f'(E)")

#     plt.legend()
#     plt.show()

#     plt.plot(ene,fsecond_KK, label = 'new f2')
#     plt.plot(ene,f1_minus, label = 'new f1')
#     plt.legend()
#     plt.show()
# #%%
#     i = str(i)
#     fsec_name = (i + ' fsec ' + filename)
#     np.savetxt((fsec_name),np.column_stack((ene,fsecond_KK)))
#     # np.savetxt(file_name_2, np.column_stack((ene, I_max_norm)))

    #%% 
    #LDR attempts

    # def struc_fac(struc_fac_exp,phase):
    #     return np.log(struc_fac_exp) * phase

    # phase_LDR = lambda ene_dash :(np.log(struc_fac_exp)/ ((ene_dash-1)**2 - (ene[i])**2))

    # phase_arr = []
    # phase_arr, err = (quad_vec(phase_LDR,(ene_dash[0]),(ene_dash[-1])))

    # zeta = 0
    # xi = 0
    # phase = (zeta(np.log((ene[0]+ene)/(ene[0]-ene)))) - ((2*ene/math.pi) * phase_arr) + (xi*(np.log((ene[-1]+ene)/(ene[-1]-ene))))

    # How do we get zeta and xi values for low and high energy limits? 0 for centrosymmmetric molecules 
    # -Mn exponent is related to debye-waller factor 



































#%%
# import scipy.fftpack as ft # Contains a function that will compute the Hilbert transform

# f2_KK = fsec - ((2/np.pi)*ft.ihilbert((f1_new-fprime)))

# print(f2_KK)

# plt.plot(ene,f2_KK, lw = 0.5, label = "transformed f2")
# # plt.plot(ene, f1_new - fprime, label = "f1 used")

# plt.xlabel("energy / eV",fontsize=18)
# plt.ylabel("f(E)",fontsize=18)
# plt.legend()
# plt.show()

#Need to extend the ranges as in Pickering? 
#%%

# #for i in something?
    ##Why does f1 and exp data look the same?

# math.isclose(phi, phi_new, rel_tol = 0.0005)

# if True: 
#     math.isclose(beta, beta_new, rel_tol = 0.0005)
# elif True:
#     math.isclose(Ioff, Ioff_new, rel_tol = 0.0005)
# elif True:
#     math.isclose(I0, I0_new, rel_tol = 0.0005)
# elif True:
#     print("finished")
# else:


