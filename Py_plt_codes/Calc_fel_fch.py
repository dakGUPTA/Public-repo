import numpy as np
import matplotlib.pyplot as plt

nx = int(input("Enter nx value: ")) 
ny = nx
step = int(input('Enter the start step value : '))
d8_format = 0 
numsteps = int(input('Enter Numsteps: '))
#nv = int(input("Enter num of variants: "))
#print_vf = int(input('Enter 1 for printing vf at every step else 0: '))
nv = 1
tprof1 = 100 
tprof2 = 1000
tprof3 = 10000
numsteps_prof1 = 1000
numsteps_prof2 = 10000

Iter = np.array([])
Ft0 = np.array([])
Ft1 = np.array([])
Ft2 = np.array([])
Fel0 = np.array([])
Fel1 = np.array([])
Fel2 = np.array([])
filedir0 = "./"
filedir0 = "./"
ft0 = 0.0
ft1 = 0.0
ft2 = 0.0
fel0 = 0.0
fel1 = 0.0
fel2 = 0.0

while step>-1 and step<(numsteps+1):
    
    print(step)
    if (d8_format == 1):
        data = np.loadtxt(f"{filedir0}/prof_gp.{step:08d}")
    else:
        data0 = np.loadtxt(f"{filedir0}/prof_gp.{step}")
        data1 = np.loadtxt(f"./../../../fchemtest_Tload/prof/prof/prof_gp.{step}")
        data2 = np.loadtxt(f"./../../../fchemtest_Cload/prof/prof/prof_gp.{step}")
   
    if (nv == 1):
        #phi = data[:,3].reshape(nx,ny)
        #ftotal = data[:,16].reshape(nx,ny)
        felas0 = data0[:,17].reshape(nx,ny)
        felas1 = data1[:,17].reshape(nx,ny)
        felas2 = data2[:,17].reshape(nx,ny)
    if (nv == 4):
        phi = (data[:,3]+data[:,4]+data[:,5]+data[:,6]).reshape(nx,ny)
    
    #count = 0.0
    #vf_old = vf
    #for i in range (nx):
    #    for j in range(ny):
    #        if (phi[i,j] > 0.5):
    #            count += 1.0
    #
    #vf = count/(nx*ny)
    #dvf = vf - vf_old
    #if (print_vf == 1):
    #    print(f"Vol frac: {vf}")
    #    print(f"dvol frac: {dvf}")
    
    #ft0 = (np.mean(ftotal0))/(nx*ny)
    #ft1 = (np.mean(ftotal1))/(nx*ny)
    #ft2 = (np.mean(ftotal2))/(nx*ny)
    fel0 = (np.mean(felas0))/(nx*ny)
    fel1 = (np.mean(felas1))/(nx*ny)
    fel2 = (np.mean(felas2))/(nx*ny)
    
    #Ft0 = np.append(Ft0,ft0)
    #Ft1 = np.append(Ft1,ft1)
    #Ft2 = np.append(Ft2,ft2)
    Fel0 = np.append(Fel0, fel0)
    Fel1 = np.append(Fel1, fel1)
    Fel2 = np.append(Fel2, fel2)
    Iter = np.append(Iter, step)

    if step == numsteps:
        break;
    if step<numsteps_prof1:
        step += tprof1
    elif step>(numsteps_prof1-1) and step<(numsteps_prof2):
        step += tprof2
    elif step>(numsteps_prof1-2) and step<(numsteps):
        step += tprof3
    #elif step<(numsteps): 
        #step += tprof2
    #plt.show()

#print(f"Volume fraction at eqm:{Vol_frac[-1]}")
#max_step = Iter.max() + 1
#x_interval = (max_step//10)
#plt.plot(Iter,Ft,'--',c='r', label='Ftotal')
plt.plot(Iter,Fel0,'--',c='k', label='NoSA')
plt.plot(Iter,Fel1,c='r', label='Tensile')
plt.plot(Iter,Fel2,c='b', label='Compressive')
plt.legend()
#plt.xticks(np.arange(0,max_step,x_interval))
plt.title('Felas_plot')
plt.ylabel('Elastic energy')
plt.xlabel('Iteration number')
plt.savefig("./Pngs/Fel.png")

#plt.xticks(np.arange(0,max_step,x_interval))
