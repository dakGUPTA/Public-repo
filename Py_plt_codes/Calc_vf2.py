import numpy as np
import matplotlib.pyplot as plt
import sys

nx = int(input("Enter nx value: ")) 
ny = nx
step = int(input('Enter the start step value : '))
d8_format = 0 
numsteps = step 
nv = int(input("Enter num of variants: "))
print_vf = 1 

tprof1 = 100 
tprof2 = 1000
tprof3 = 10000
numsteps_prof1 = 1000
numsteps_prof2 = 10000

Iter = np.array([])
Vol_frac = np.array([])
dvol_frac = np.array([])
filedir0 = "./"
filedir0 = "./"
vf = 0.0

while step>-1 and step<(numsteps+1):
    
    print(step)
    if (d8_format == 1):
        data = np.loadtxt(f"{filedir0}/prof_gp.{step:08d}")
    else:
        data = np.loadtxt(f"{filedir0}/prof_gp.{step}")
   
    if (nv == 0):
        phi = data[:,3].reshape(nx,ny)
    if (nv == 4):
        phi = (data[:,3]+data[:,4]+data[:,5]+data[:,6]).reshape(nx,ny)
    
    count = 0.0
    vf_old = vf
    for i in range (nx):
        for j in range(ny):
            if (phi[i,j] > 0.5):
                count += 1.0
    
    vf = count/(nx*ny)
    dvf = vf - vf_old
    if (print_vf == 1):
        print(f"Vol frac: {vf}")
        print(f"dvol frac: {dvf}")

    Vol_frac = np.append(Vol_frac, vf)
    dvol_frac = np.append(dvol_frac, dvf)
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

sys.exit()
np.savetxt('Vf.txt',Vol_frac)
print(f"Volume fraction at eqm:{Vol_frac[-1]}")
#max_step = Iter.max() + 1
#x_interval = (max_step//10)
plt.plot(Iter,Vol_frac,'--',c='r')
#plt.xticks(np.arange(0,max_step,x_interval))
plt.title('Gamma prime Volume fraction')
plt.ylabel('Volume fraction')
plt.xlabel('Iteration number')
plt.savefig("./Pngs/Vf.png")

plt.plot(Iter, dvol_frac,c='b')
#plt.xticks(np.arange(0,max_step,x_interval))
plt.title('diff Volume fraction')
plt.ylabel('dvol fraction')
plt.xlabel('Iteration number')
plt.savefig("./Pngs/dvf.png")
