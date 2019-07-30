############################################
#  CT numbers to density/chem. compo. using 
#  the stoichiometric method (Schneider2000)
# 		   ---
#           P. Lansonneur 2018
############################################

import matplotlib.pyplot as P
import numpy as np
from numpy import genfromtxt
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from scipy.optimize import curve_fit, minimize
#P.rcParams.update({'font.size': 7})
from termcolor import colored

### read inserts and tissus file #######################
tissus 	= genfromtxt('./tissus.csv', delimiter=';')#, dtype=None)
data 	= genfromtxt('./inserts.csv', delimiter=';')
rho_inserts = data[1][3:]
rho_tissus = tissus[1][1:]
Z = data.T[1][2:28]
A = data.T[2][2:28]
W = data[2:28,3:]
W_tissus = tissus[3:,1:]
H_cartilage_os = 200

#Hmeas_inserts, title = data[29][3:], ' SARRP 2018, 60kV'
Hmeas_inserts,title,H_ludo,rho_ludo = data[30][3:], 'Toshiba IC 2017 120kV fantome head FC43', [-1000,-200,-68,-11,38,110,179,3071],[0.00025917,0.79279,0.93129,0.99204,1.0286,1.0999,1.1,2.8019]
#Hmeas_inserts,title,H_ludo,rho_ludo = data[31][3:], 'Toshiba IC 2017 135kV fantome pelvis FC13', [-1000,-200,-79,-23,24,94,139,3071],[0.00016051,0.80407,0.93092,1.0032,1.0292,1.0995,1.0999,3.1282]
#Hmeas_inserts,title,H_ludo,rho_ludo = data[32][3:], 'Toshiba IC 2017 100kV fantome head FC13', [-1000,-200,-77,-13,43,117,198,3071],[0.0001661,0.78809,0.93175,0.99231,1.0274,1.1003,1.0999,2.5936]
#Hmeas_inserts,title,H_ludo,rho_ludo = data[33][3:], 'Siemens IC 2018 120kV fantome body', [-1000,-200,-87,-26,26,97,139,3071],[2.1859e-05,0.80234,0.93193,1.0038,1.029,1.1002,1.1001,3.0564]
#Hmeas_inserts,title,H_ludo,rho_ludo = data[34][3:], 'Siemens IC 2018 120kV fantome head', [-1000,-200,-60,-7,38,110,181,3071],[0.00036957,0.79267,0.93095,0.98881,1.0284,1.0997,1.1,2.8404]

### remove unscanned inserts  ('nan' in the csv file)
tmp = np.argwhere(np.isnan(Hmeas_inserts)).T[0]
rho_inserts = np.delete(rho_inserts,  tmp)
Hmeas_inserts = np.delete(Hmeas_inserts,  tmp)
W = np.delete(W.T,  tmp,0).T

### Minimize Schneider2000 equation (11) to find the parameters k1,k2
def myfunc(j,k1,k2):
	mu_eau = 1*( (0.1119)*(1+k1+k2) + (0.8881/16)*(8+(8**2.86)*k1+(8**4.62)*k2) )
	mu = 0
	for i in range(0,len(Z)):	mu = mu + (W[i,j]/A[i])*(Z[i]+(Z[i]**2.86)*k1+(Z[i]**4.62)*k2)
	mu = rho_inserts[j]*mu 
	H = 1000*((mu/mu_eau)-1)
	return H

def H_calc_vector(x):
	output = 0
	k1, k2 = x[0],x[1]
	for i in range(0,len(Hmeas_inserts)):	output = output+(myfunc(i,k1,k2)-Hmeas_inserts[i])**2
	return output

res = minimize(H_calc_vector, [1E-3,3E-5])
k1_min, k2_min = res.x[0], res.x[1]

### recompute tissus densities #######################
def Hcalc_tissus(j,k1,k2):
	mu_eau = 1*( (0.1119)*(1+k1+k2) + (0.8881/16)*(8+(8**2.86)*k1+(8**4.62)*k2) )
	mu = 0
	for i in range(0,len(Z)):	mu = mu + (W_tissus[i,j]/A[i])*(Z[i]+(Z[i]**2.86)*k1+(Z[i]**4.62)*k2)
	mu = rho_tissus[j]*mu 
	H = 1000*((mu/mu_eau)-1)
	return H

Hcal_inserts = np.zeros(len(rho_inserts))
for j in range(0,len(rho_inserts)):	Hcal_inserts[j] = int(myfunc(j,k1_min, k2_min))

Hcal_tissus = np.zeros(len(rho_tissus))
for j in range(0,len(rho_tissus)):	Hcal_tissus[j] = int(Hcalc_tissus(j,k1_min, k2_min))

### Fit piecewise ################################
def linear(x,a,b):	return a*x + b

popt,pcov = curve_fit(linear,Hcal_tissus[7:23], rho_tissus[7:23],p0=[1000,1]) 
#P.plot(np.linspace(-1024,-200),linear(np.linspace(-1024,-200),*popt),'r:')	# Fit Organs and Muscle 1
H_1000 = linear(-1000,*popt)
H_200 = linear(-200,*popt)
#P.plot([-200,-7],[linear(-200,*popt),],'r:')	# transition1
#P.plot(np.linspace(-7,38),linear(np.linspace(-7,38),*popt),'r:')	# Fit Organs and Muscle 2
H_7 = linear(-7,*popt)
popt,pcov = curve_fit(linear,Hcal_tissus[0:6], rho_tissus[0:6],p0=[1000,1]) # Fit Adipose
#P.plot(np.linspace(-123,0),linear(np.linspace(-123,0),*popt),'r:')
H_123 = linear(-123,*popt)
#P.plot(np.linspace(38,300),rho_tissus[27]*np.ones(50),'r:')
popt,pcov = curve_fit(linear,Hcal_tissus[28:-2], rho_tissus[28:-2],p0=[1000,1]) # Fit Bone
#P.plot(np.linspace(300,3000),linear(np.linspace(300,3000),*popt),'r:')
H_2000 = linear(2000,*popt)
H_3000 = linear(3000,*popt)
H_2995 = linear(2995,*popt)

H_ok = [-1000, -200, -123, -7, int(Hcal_tissus[27]), H_cartilage_os, 3000, 3001]
rho_ok = [H_1000,H_200, H_123, H_7, rho_tissus[27], rho_tissus[27], H_3000, H_3000]

### Plot density #########################################
fig = P.gcf()
ax = fig.gca()

P.plot(Hmeas_inserts, rho_inserts,'b.',label='inserts (measured)')
P.plot(Hcal_inserts, rho_inserts,'bs',label='inserts (calculated)')
P.plot(Hcal_tissus[0:6], rho_tissus[0:6],'gx',label='adipose ICRP')
P.plot(Hcal_tissus[7:23], rho_tissus[7:23],'gs',label='organs and muscle ICRP')
P.plot(Hcal_tissus[24:26], rho_tissus[24:26],'gv',label='breast ICRP')
P.plot(Hcal_tissus[27], rho_tissus[27],'g.',label='cartilage ICRP')
P.plot(Hcal_tissus[28:41], rho_tissus[28:41],'g^',label='bone ICRP')
P.plot(H_ludo,rho_ludo,'r:',label='fit Ludo')
P.plot(H_ok, rho_ok,'b:',label='fit')
P.title(title)
P.xlabel('CT numbers (HU)')
P.ylabel('$\\rho$ (g/cm$^3$)')

#P.xlim((-150,100))
#P.ylim((0.9,1.15))
#fig.set_size_inches(3.54,2.2)
P.legend(frameon=False)
#P.savefig('rho_HU_calib.tif',  bbox_inches='tight', dpi=300)
P.savefig('rho_HU_calib.pdf',  bbox_inches='tight')
P.show()

### Chemical composition ##########################
tissus = genfromtxt('./tissus2.csv', delimiter=';')#, dtype=None)
W_tissus = tissus[3:,1:]
#W_air 	= np.round([0,   0, 0.755,0.232,0,  0,  0,  0,   0,   0.013,0,  0,  0],3)
#W_soft = np.round([0.103, 0.134, 0.03, 0.722, 0, 0.002, 0.002, 0.002, 0, 0, 0, 0.002, 0],3)
W_air 	= np.round([0,   0, 0.755,0.232,0,  0,  0,  0, 0.013, 0, 0, 0, 0],3)
W_soft 	= np.round([0.103, 0.134, 0.03, 0.722, 0, 0.002, 0.002, 0.002, 0.002, 0, 0.002, 0.002, 0],3)
W_Ti 	= np.round([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.],3)

H_AT,H_AG,H_ma,H_bo,H_con,H_SI = int(Hcal_tissus[6]), int(Hcal_tissus[42]), int(Hcal_tissus[43]), int(Hcal_tissus[31]), int(Hcal_tissus[44]), int(Hcal_tissus[12])

print '\nAT:\t\t\t',		H_AT
print 'AG:\t\t\t',		H_AG
print 'Small Intestine:\t',	H_SI
print 'Connective:\t\t',	H_con
print 'Marrow (1:1):\t\t',	H_ma
print 'Cortical bone:\t',	H_bo

def W_ATAG(H): # AT/AG function, Schneider 2000, equation (18),(22)
	W_AT, W_AG = W_tissus[:,6], W_tissus[:,42]
	rhoAT, rhoAG = rho_tissus[6], rho_tissus[42]
	return np.round( (rhoAT*(H_AG - H)*(W_AT-W_AG)/((rhoAT*H_AG - rhoAG*H_AT)+(rhoAG-rhoAT)*H))+W_AG, 3)

def W_MaBo(H): # Marrow/bone function, Schneider 2000, equation (18),(20)
	W_ma, W_bo = W_tissus[:,43], W_tissus[:,31]
	rhoma, rhobo = rho_tissus[43], rho_tissus[31]
	return np.abs(np.round( (rhoma*(H_bo - H)*(W_ma-W_bo)/((rhoma*H_bo - rhobo*H_ma)+(rhobo-rhoma)*H))+W_bo, 3))

HUToMaterialSections 	= [-1024,-950,H_AT-15, H_AT+15, H_AT+45,int(0.5*(H_AG+H_SI)), H_con-20, H_con+20,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,2300,3001]
MaterialsWeight 	= np.append([W_air],[W_tissus[:,8],W_tissus[:,6],W_ATAG(H_AT+30),W_ATAG(H_AT+60),W_soft,W_tissus[:,44],W_MaBo(Hcal_tissus[44]),W_MaBo(150),W_MaBo(250),W_MaBo(350),W_MaBo(450),W_MaBo(550),W_MaBo(650),W_MaBo(750),W_MaBo(850),W_MaBo(950),W_MaBo(1050),W_MaBo(1150),W_MaBo(1250),W_MaBo(1350),W_MaBo(1450),W_MaBo(1600),W_MaBo(1700),W_Ti,W_Ti],axis=0)
MaterialName 		= ['Air','Lung','AT_AG1','AT_AG2','AT_AG3','SoftTissus','ConnectiveTissue','Marrow_Bone01','Marrow_Bone02','Marrow_Bone03','Marrow_Bone04','Marrow_Bone05','Marrow_Bone06','Marrow_Bone07','Marrow_Bone08','Marrow_Bone09','Marrow_Bone10','Marrow_Bone11','Marrow_Bone12','Marrow_Bone13','Marrow_Bone14','Marrow_Bone15','Marrow_Bone16','Marrow_Bone17','MetallImplants','MetallImplants']

# correct small deviations from 1..
for i in range(0,len(HUToMaterialSections)):	MaterialsWeight[i][0] = np.abs(np.round(MaterialsWeight[i][0] + 1. - np.sum(MaterialsWeight[i]),3))

def arr2string(array):	return '\t'.join([str(a) for a in array])
def arr2string_space(array):	return ' '.join([str(a) for a in array])

### write output to file #########################################
# GATE HU to density table
print '\n# ===================='
print '# HU\tdensity g/cm3 '
print '# ===================='
for i in range(0,len(H_ok)-1):	print '{0}\t{1:.5f}'.format( H_ok[i], rho_ok[i])

# GATE HU to material table
#perm = range(0,13)
perm = np.array([0,1,2,3,10,4,5,6,7,8,11,9,12]) # elements are not in the same order in GATE and TOPAS..
print '\n# ============================================================================================================'
print '# HU\tH\tC\tN\tO\tNa\tMg\tP\tS\tCl\tAr\tK\tCa\tTi  Cu  Zn  Ag  Sn'
print '# ============================================================================================================'
for i in range(0,len(HUToMaterialSections)):	
	if (i>1 and i<6):	print HUToMaterialSections[i],'\t', arr2string(MaterialsWeight[i][perm]),'0.0 0.0 0.0 0.0\t', colored(MaterialName[i],'red')
	elif (i>7 and i<len(HUToMaterialSections)-2):	print HUToMaterialSections[i],'\t', arr2string(MaterialsWeight[i][perm]),'0.0 0.0 0.0 0.0\t', colored(MaterialName[i],'green')
	else:	print HUToMaterialSections[i],'\t', arr2string(MaterialsWeight[i][perm]),'0.0 0.0 0.0 0.0\t', MaterialName[i]

# TOPAS HUtoMAT file
"""
DensityOffset,DensityFactor = np.zeros(len(H_ok)-1), np.zeros(len(H_ok)-1) 
for i in range(0,len(H_ok)-1): # schneider 2000, equation (17)
	DensityOffset[i] = (rho_ok[i]*H_ok[i+1] - rho_ok[i+1]*H_ok[i])/(H_ok[i+1]-H_ok[i])
	DensityFactor[i] = (rho_ok[i+1]-rho_ok[i])/(H_ok[i+1]-H_ok[i])

"""
DensityOffset,DensityFactor = np.zeros(len(H_ludo)-1), np.zeros(len(H_ludo)-1) 
for i in range(0,len(H_ludo)-1): # schneider 2000, equation (17)
	DensityOffset[i] = (rho_ludo[i]*H_ludo[i+1] - rho_ludo[i+1]*H_ludo[i])/(H_ludo[i+1]-H_ludo[i])
	DensityFactor[i] = (rho_ludo[i+1]-rho_ludo[i])/(H_ludo[i+1]-H_ludo[i])
HUToMaterialSections[0] = -1000

with open('HUtoMAT.txt', 'w') as f:
	f.write('### Topas parameter settings for conversion of Hounsfield Units to materials using the Schneider paper ###\n')
	f.write('s:Ge/Patient/CT/ImagingtoMaterialConverter = "Schneider"\n\n')

	f.write('### Correction Factor for the relative stopping power of Geant4 and the XiO planning system:\n')
	f.write('dv:Ge/Patient/CT/DensityCorrection = 4001 ')
	for i in range(4001):	f.write('1.0 ')
	f.write('g/cm3\n\n')
	f.write('### Formula : Density = (Offset + (Factor*(FactorOffset + HU[-1000,3000] ))) * DensityCorrection\n')
	f.write('# {0} \n'.format(title))
	f.write('iv:Ge/Patient/CT/SchneiderHounsfieldUnitSections = {0} '.format(len(H_ludo)))
	for i in H_ludo:	f.write('{0} '.format(i))
	f.write('\n')
	f.write('uv:Ge/Patient/CT/SchneiderDensityOffset          = {0} '.format(len(H_ludo)-1) )
	for i in DensityOffset:	f.write('{0:.7f} '.format(i))
	f.write('\n')
	f.write('uv:Ge/Patient/CT/SchneiderDensityFactor          = {0} '.format(len(H_ludo)-1) )
	for i in DensityFactor:	f.write('{0:.7f} '.format(i))
	f.write('\n')
	f.write('uv:Ge/Patient/CT/SchneiderDensityFactorOffset    = {0} '.format(len(H_ludo)-1) )
	for i in range(0,len(H_ok)-1):	f.write('0 '.format(i))
	f.write('\n\n')

	f.write('### Define Materials used for HU\n')
	f.write('sv:Ge/Patient/CT/SchneiderElements\t= 13 "Hydrogen" "Carbon" "Nitrogen" "Oxygen" "Magnesium" "Phosphorus" "Sulfur" "Chlorine" "Argon" "Calcium" "Sodium" "Potassium" "Titanium"\n')
	f.write('iv:Ge/Patient/CT/SchneiderHUToMaterialSections\t= {0} '.format(len(HUToMaterialSections)))
	for i in HUToMaterialSections:	f.write('{0} '.format(i))
	f.write('\n')
	for i in range(1,len(HUToMaterialSections)+1):
		f.write('uv:Ge/Patient/CT/SchneiderMaterialsWeight{0} \t= 13 '.format(i))
		for j in MaterialsWeight[i-1]:	f.write('{0}\t'.format(j))
		f.write('# {0}\n'.format(MaterialName[i-1]))
