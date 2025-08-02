import numpy as np
import matplotlib.pyplot as plt # type: ignore
import os
import subprocess
import shutil
import glob

plotting = 1 
movie = 1
phi_comp_plots = 1 
nph =1 

oneD_plots = 1 
df_plots = 1 
elas = 1 
Z_comp = 1 
sts_plots = 1 
str_plots = 1 
felas_plots = 1 
ftotal_plots = 1 

#plotting = int(input("Enter 1 for making plots or else 0 : "))
#movie = int(input("Enter 1 for making movie out of plots or else 0 : "))
#phi_comp_plots = int(input("Enter 1 for phi & comp plots else 0: "))
#nph = int(input("Enter num of phases:")) 
#
#if (nph == 1):
#    oneD_plots = int(input("Enter 1 for 1D_plots else 0: "))
#    df_plots = int(input("Enter 1 for driving forces plots else 0: "))
#    elas = int(input("Enter 1 for elasticity plots else 0: "))
#    if (elas == 1):
#        Z_comp = int(input("Enter 1 for using 33 components else 0: "))
#        sts_plots = int(input("Enter 1 for stress plots else 0: "))
#        str_plots = int(input("Enter 1 for strain plots else 0: "))
#        felas_plots = int(input("Enter 1 for felas_plots else 0: "))
#        if (felas_plots == 1):
#            felas_calc = int(input("Enter 1 for felas calc from stresses and strains else 0: "))
#        ftotal_plots = int(input("Enter 1 for ftotal_plots else 0: "))

if (plotting == 1):
    nx = int(input("Enter nx value:")) 
    ny = nx
    starting_step = int(input("Enter starting step: "))
    step = starting_step
    numsteps = int(input("Enter Numsteps:"))
    felas_calc = int(input("enter 1 for calc or 0 for taking from prof file:")) 
    
    step_grid = int(input("Enter 0 for step grid default values else 1:"))
    if (step_grid == 0):
        tprof1 = 100
        tprof2 = 1000
        tprof3 = 10000
        numsteps_prof1 = 1000
        numsteps_prof2 = 10000
    else:
        tprof1 = int(input("Enter tprof1: "))
        tprof2 = int(input("Enter tprof2: "))
        tprof3 = int(input("Enter tprof3: "))
        numsteps_prof1 = int(input("Enter numsteps_prof1: "))
        numsteps_prof2 = int(input("Enter numsteps_prof2: "))


#tprof4 = 50000
#numsteps_prof3 = 100000

filedir = os.getcwd()
print(f"Current dir:", filedir)
if os.path.isdir(f'{filedir}/prof'):
    pass
else:
    os.mkdir(f'{filedir}/prof')

filedir0 = f"{filedir}/prof"

if (plotting == 1):
    
    prof_files = glob.glob("prof_gp*")
    for prof_file  in prof_files:
        shutil.move(prof_file,filedir0)

    if (phi_comp_plots == 1):

        if os.path.isdir(f'{filedir0}/phi_plots'):
            shutil.rmtree(f'{filedir0}/phi_plots')
            os.mkdir(f'{filedir0}/phi_plots')
        else:
            os.mkdir(f'{filedir0}/phi_plots')

        if os.path.isdir(f'{filedir0}/comp_plots'):
            shutil.rmtree(f'{filedir0}/comp_plots')
            os.mkdir(f'{filedir0}/comp_plots')
        else:
            os.mkdir(f'{filedir0}/comp_plots')

    if (nph == 1):

        if (oneD_plots == 1):
            
            if os.path.isdir(f'{filedir0}/phi1D_plots'):
                shutil.rmtree(f'{filedir0}/phi1D_plots')
                os.mkdir(f'{filedir0}/phi1D_plots')
            else:
                os.mkdir(f'{filedir0}/phi1D_plots')

            if os.path.isdir(f'{filedir0}/comp1D_plots'):
                shutil.rmtree(f'{filedir0}/comp1D_plots')
                os.mkdir(f'{filedir0}/comp1D_plots')
            else:
                os.mkdir(f'{filedir0}/comp1D_plots')

        if (df_plots == 1):
            
            if os.path.isdir(f'{filedir0}/dfdc_plots'):
                shutil.rmtree(f'{filedir0}/dfdc_plots')
                os.mkdir(f'{filedir0}/dfdc_plots')
            else:
                os.mkdir(f'{filedir0}/dfdc_plots')

            if os.path.isdir(f'{filedir0}/dfdphi_plots'):
                shutil.rmtree(f'{filedir0}/dfdphi_plots')
                os.mkdir(f'{filedir0}/dfdphi_plots')
            else:
                os.mkdir(f'{filedir0}/dfdphi_plots')

            if os.path.isdir(f'{filedir0}/dfeldphi_plots'):
                shutil.rmtree(f'{filedir0}/dfeldphi_plots')
                os.mkdir(f'{filedir0}/dfeldphi_plots')
            else:
                os.mkdir(f'{filedir0}/dfeldphi_plots')

        if (elas == 1):
            if (sts_plots == 1):

                if os.path.isdir(f'{filedir0}/stress_plots'):
                    shutil.rmtree(f'{filedir0}/stress_plots')
                    os.mkdir(f'{filedir0}/stress_plots')
                else:
                    os.mkdir(f'{filedir0}/stress_plots')
                
                if os.path.isdir(f'{filedir0}/stress_plots/S11_plots'):
                    shutil.rmtree(f'{filedir0}/stress_plots/S11_plots')
                    os.mkdir(f'{filedir0}/stress_plots/S11_plots')
                else:
                    os.mkdir(f'{filedir0}/stress_plots/S11_plots')
                
                if os.path.isdir(f'{filedir0}/stress_plots/S22_plots'):
                    shutil.rmtree(f'{filedir0}/stress_plots/S22_plots')
                    os.mkdir(f'{filedir0}/stress_plots/S22_plots')
                else:
                    os.mkdir(f'{filedir0}/stress_plots/S22_plots')
                
                if os.path.isdir(f'{filedir0}/stress_plots/S12_plots'):
                    shutil.rmtree(f'{filedir0}/stress_plots/S12_plots')
                    os.mkdir(f'{filedir0}/stress_plots/S12_plots')
                else:
                    os.mkdir(f'{filedir0}/stress_plots/S12_plots')
                
                if (Z_comp == 1):
                    if os.path.isdir(f'{filedir0}/stress_plots/S33_plots'):
                        shutil.rmtree(f'{filedir0}/stress_plots/S33_plots')
                        os.mkdir(f'{filedir0}/stress_plots/S33_plots')
                    else:
                        os.mkdir(f'{filedir0}/stress_plots/S33_plots')
                
                if os.path.isdir(f'{filedir0}/stress_plots/Hsts_plots'):
                    shutil.rmtree(f'{filedir0}/stress_plots/Hsts_plots')
                    os.mkdir(f'{filedir0}/stress_plots/Hsts_plots')
                else:
                    os.mkdir(f'{filedir0}/stress_plots/Hsts_plots')
           
                
                sts_mat = np.zeros((nx,ny,3,3))
                Eigvals = np.zeros((nx,ny,3))
                Sig1 = np.zeros((nx,ny))
                Sig2 = np.zeros((nx,ny))
                if (Z_comp == 1):
                    Sig3 = np.zeros((nx,ny))
                Hsts = np.zeros((nx,ny))
                
            if (str_plots == 1):

                if os.path.isdir(f'{filedir0}/strain_plots'):
                    shutil.rmtree(f'{filedir0}/strain_plots')
                    os.mkdir(f'{filedir0}/strain_plots')
                else:
                    os.mkdir(f'{filedir0}/strain_plots')

                if os.path.isdir(f'{filedir0}/strain_plots/e11_plots'):
                    shutil.rmtree(f'{filedir0}/strain_plots/e11_plots')
                    os.mkdir(f'{filedir0}/strain_plots/e11_plots')
                else:
                    os.mkdir(f'{filedir0}/strain_plots/e11_plots')
                
                if os.path.isdir(f'{filedir0}/strain_plots/e22_plots'):
                    shutil.rmtree(f'{filedir0}/strain_plots/e22_plots')
                    os.mkdir(f'{filedir0}/strain_plots/e22_plots')
                else:
                    os.mkdir(f'{filedir0}/strain_plots/e22_plots')
                
                if os.path.isdir(f'{filedir0}/strain_plots/e12_plots'):
                    shutil.rmtree(f'{filedir0}/strain_plots/e12_plots')
                    os.mkdir(f'{filedir0}/strain_plots/e12_plots')
                else:
                    os.mkdir(f'{filedir0}/strain_plots/e12_plots')
                
                if (Z_comp == 1):
                    if os.path.isdir(f'{filedir0}/strain_plots/e33_plots'):
                        shutil.rmtree(f'{filedir0}/strain_plots/e33_plots')
                        os.mkdir(f'{filedir0}/strain_plots/e33_plots')
                    else:
                        os.mkdir(f'{filedir0}/strain_plots/e33_plots')

                if os.path.isdir(f'{filedir0}/strain_plots/Hstr_plots'):
                    shutil.rmtree(f'{filedir0}/strain_plots/Hstr_plots')
                    os.mkdir(f'{filedir0}/strain_plots/Hstr_plots')
                else:
                    os.mkdir(f'{filedir0}/strain_plots/Hstr_plots')
               
                str_mat = np.zeros((nx,ny,3,3))
                Eigvals1 = np.zeros((nx,ny,3))
                ## Principal strains.
                Eps1 = np.zeros((nx,ny))
                Eps2 = np.zeros((nx,ny))
                if (Z_comp == 1):
                    Eps3 = np.zeros((nx,ny))
                ## Hydrostatic strains
                Hstr = np.zeros((nx,ny))
           
            if (felas_plots == 1):
                
                if os.path.exists(f'{filedir0}/felas_plots'):
                    shutil.rmtree(f'{filedir0}/felas_plots')
                    os.mkdir(f'{filedir0}/felas_plots')
                else:
                    os.mkdir(f'{filedir0}/felas_plots')
                
                if (felas_calc == 1):
                    felas = np.zeros((nx,ny))
            
            if (ftotal_plots == 1):
                
                if os.path.isdir(f'{filedir0}/ftotal_plots'):
                    shutil.rmtree(f'{filedir0}/ftotal_plots')
                    os.mkdir(f'{filedir0}/ftotal_plots')
                else:
                    os.mkdir(f'{filedir0}/ftotal_plots')
                

if (phi_comp_plots == 1):

    filedir1 = f'{filedir0}/phi_plots'
    filedir2 = f'{filedir0}/comp_plots'

if (nph == 1):
    
    if (oneD_plots == 1):
        filedir3 = f'{filedir0}/phi1D_plots'
        filedir4 = f'{filedir0}/comp1D_plots'

    if (df_plots == 1):
        filedir5 = f'{filedir0}/dfdc_plots'
        filedir6 = f'{filedir0}/dfdphi_plots'
        filedir7 = f'{filedir0}/dfeldphi_plots'

    if (elas == 1):
        if (sts_plots == 1):
            filedir8 = f'{filedir0}/stress_plots/S11_plots'
            filedir9 = f'{filedir0}/stress_plots/S22_plots'
            filedir10 = f'{filedir0}/stress_plots/S12_plots'
            if (Z_comp == 1):
                filedir11 = f'{filedir0}/stress_plots/S33_plots'
            filedir12 = f'{filedir0}/stress_plots/Hsts_plots'
            

        if (str_plots == 1):
            filedir13 = f'{filedir0}/strain_plots/e11_plots'
            filedir14 = f'{filedir0}/strain_plots/e22_plots'
            filedir15 = f'{filedir0}/strain_plots/e12_plots'
            if (Z_comp == 1):
                filedir16 = f'{filedir0}/strain_plots/e33_plots'
            filedir17 = f'{filedir0}/strain_plots/Hstr_plots'
           
        if (felas_plots == 1):
            filedir18 = f'{filedir0}/felas_plots'
        
        if (ftotal_plots == 1):
            filedir19 = f'{filedir0}/ftotal_plots'


if (plotting == 1):
    while step>-1 and step<(numsteps+1):
        
        print(step)
        data = np.loadtxt(f"{filedir0}/prof_gp.{step}")
        comp = (data[:,2]).reshape(nx,ny)
        if (nph == 4):
            phi = (data[:,3]*1+data[:,4]*2+data[:,5]*3+data[:,6]*4).reshape(nx,ny)
        elif (nph == 1):
            phi = (data[:,3]).reshape(nx,ny)
            if (df_plots == 1):
                dfdc = (data[:,4]).reshape(nx,ny)
                dfchdphi = (data[:,5]).reshape(nx,ny)
                dgdphi = (data[:,6]).reshape(nx,ny)
                if (elas == 1):
                    dfeldphi = (data[:,7]).reshape(nx,ny)
                    dfdphi = dfchdphi + dgdphi + dfeldphi
                else:
                    dfdphi = dfchdphi + dgdphi

            
            if (elas == 1):
                if (Z_comp == 1):
                    s11 = data[:,8].reshape(nx,ny)
                    s22 = data[:,9].reshape(nx,ny)
                    s33 = data[:,10].reshape(nx,ny)
                    #s12 = data[:,11].reshape(nx,ny)
                    s12 = 2.0*(data[:,11].reshape(nx,ny))
                    e11 = data[:,12].reshape(nx,ny)
                    e22 = data[:,13].reshape(nx,ny)
                    e33 = data[:,14].reshape(nx,ny)
                    e12 = data[:,15].reshape(nx,ny)
             
                elif (Z_comp == 0):
                    s11 = data[:,8].reshape(nx,ny)
                    s22 = data[:,9].reshape(nx,ny)
                    s12 = 2.0*(data[:,10].reshape(nx,ny))
                    #s12 = data[:,10].reshape(nx,ny)
                    e11 = data[:,11].reshape(nx,ny)
                    e22 = data[:,12].reshape(nx,ny)
                    e12 = data[:,13].reshape(nx,ny)
              
                if (sts_plots == 1):#

             
                    for i in range (nx):
                        for j in range (ny):

                            sts_mat[i,j,0,0] = s11[i,j]
                            sts_mat[i,j,0,1] = s12[i,j]
                            sts_mat[i,j,0,2] = 0.0 
                            sts_mat[i,j,1,0] = s12[i,j]
                            sts_mat[i,j,1,1] = s22[i,j]
                            sts_mat[i,j,1,2] = 0.0 
                            sts_mat[i,j,2,0] = 0.0 
                            sts_mat[i,j,2,1] = 0.0 
                            if (Z_comp == 1):
                                sts_mat[i,j,2,2] = s33[i,j] 
                            
                            Eigvals[i,j] = np.linalg.eigvals(sts_mat[i,j])
                            Sig1[i,j] = Eigvals[i,j,0]
                            Sig2[i,j] = Eigvals[i,j,1]
                            if (Z_comp == 1):
                                Sig3[i,j] = Eigvals[i,j,2]
                            
                            Hsts[i,j] = np.mean(Eigvals[i,j])

                if (str_plots == 1):

                    for i in range (nx):
                        for j in range (ny):
                            
                            str_mat[i,j,0,0] = e11[i,j]
                            str_mat[i,j,0,1] = e12[i,j]
                            str_mat[i,j,0,2] = 0.0 
                            str_mat[i,j,1,0] = e12[i,j]
                            str_mat[i,j,1,1] = e22[i,j]
                            str_mat[i,j,1,2] = 0.0 
                            str_mat[i,j,2,0] = 0.0 
                            str_mat[i,j,2,1] = 0.0 
                            if (Z_comp == 1):
                                str_mat[i,j,2,2] = e33[i,j]  

                            Eigvals1[i,j] = np.linalg.eigvals(str_mat[i,j])
                            Eps1[i,j] = Eigvals1[i,j,0]
                            Eps2[i,j] = Eigvals1[i,j,1]
                            if (Z_comp == 1):
                                Eps3[i,j] = Eigvals1[i,j,2]
                            
                            Hstr[i,j] = np.mean(Eigvals1[i,j])
                
                if (felas_plots == 1):
                   
                    if (felas_calc == 0):
                        felas  = data[:,17].reshape(nx,ny)
                    
                    if (felas_calc == 1):
                        felas[:,:] = 0.5*(s11[:,:]*(e11[:,:]) + \
                                          s22[:,:]*(e22[:,:]) + \
                                          s12[:,:]*(1*e12[:,:]))
                        if (Z_comp == 1):
                            felas[:,:] += (s33[:,:]*e33[:,:])
            
            if (ftotal_plots == 1):
                ftotal  = data[:,16].reshape(nx,ny)

            
            if (oneD_plots == 1):
                plt.close()
                plt.plot(phi[nx//2,:],'-o')
                plt.title(f'phi1D_{step}')
                plt.savefig(f'{filedir3}/phi1D_{step}.png')
            
                plt.close()
                plt.plot(comp[nx//2,:],'-o')
                plt.title(f'comp1D_{step}')
                plt.savefig(f'{filedir4}/comp1D_{step}.png')

            if (df_plots == 1):
                plt.close()
                plt.imshow(dfdc,origin='lower',cmap='jet')
                plt.colorbar()
                plt.title(f'dfdc_{step}')
                plt.savefig(f'{filedir5}/dfdc_{step}.png')

                plt.close()
                plt.imshow(dfdphi,origin='lower',cmap='jet')
                plt.colorbar()
                plt.title(f'dfdphi_{step}')
                plt.savefig(f'{filedir6}/dfdphi_{step}.png')

                plt.close()
                plt.imshow(dfeldphi,origin='lower',cmap='jet')
                plt.colorbar()
                plt.title(f'dfeldphi_{step}')
                plt.savefig(f'{filedir7}/dfeldphi_{step}.png')
     
            if (elas == 1):
                if (sts_plots == 1):
                    
                    plt.close()
                    plt.imshow(s11,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"S11_{step}")
                    plt.savefig(f"{filedir8}/S11_{step}")
                     
                    plt.close()
                    plt.imshow(s22,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"S22_{step}")
                    plt.savefig(f"{filedir9}/S22_{step}")
                    
                    plt.close()
                    plt.imshow(s12,origin='lower',cmap='jet')
                    #plt.gca().invert_yaxis()
                    plt.colorbar()
                    plt.title(f"S12_{step}")
                    plt.savefig(f"{filedir10}/S12_{step}")
                    
                    #plt.close()
                    #plt.imshow(Sig1,cmap = 'jet')
                    #plt.colorbar()
                    #plt.title(f"Principal_stress1_{step}")
                    #plt.savefig(f"{f#iledir8}/Sig1_{step}")
                    #
                    #plt.close()
                    #plt.imshow(Sig2,cmap = 'jet')
                    #plt.colorbar()
                    #plt.title(f"Principal Stress2_{step}")
                    #plt.savefig(f"{filedir9}/Sig2_{step}")
                    
                    if (Z_comp == 1):
                        plt.close()
                        plt.imshow(s33,origin='lower',cmap = 'jet')
                        plt.colorbar()
                        plt.title(f"S33_{step}")
                        plt.savefig(f"{filedir11}/S33_{step}")

                    plt.close()#
                    plt.imshow(Hsts,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"Hydrostatic Stress_{step}")
                    plt.savefig(f"{filedir12}/Hsts_{step}")
                    
                if (str_plots == 1):

                    plt.close()
                    plt.imshow(e11,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"E11_{step}")
                    plt.savefig(f"{filedir13}/E11_{step}")

                    plt.close()
                    plt.imshow(e22,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"E22_{step}")
                    plt.savefig(f"{filedir14}/E22_{step}")
                    
                    plt.close()
                    plt.imshow(e12,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"E12_{step}")
                    plt.savefig(f"{filedir15}/E12_{step}")
                    
                    if (Z_comp == 1):
                        plt.close()
                        plt.imshow(e33,origin='lower',cmap = 'jet')
                        plt.colorbar()
                        plt.title(f"E33_{step}")
                        plt.savefig(f"{filedir16}/E33_{step}")

                    plt.close()
                    plt.imshow(Hstr,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"Hydrostatic strain_{step}")
                    plt.savefig(f"{filedir17}/Hstr_{step}")

                if (felas_plots == 1):

                    plt.close()
                    plt.imshow(felas,origin='lower',cmap = 'jet')
                    plt.colorbar()
                    plt.title(f"felas_{step}")
                    plt.savefig(f"{filedir18}/felas_{step}")

            if (ftotal_plots == 1):
                #
                plt.close()
                plt.imshow(ftotal,origin='lower',cmap = 'jet')
                plt.colorbar()
                plt.title(f"ftotal_{step}")
                plt.savefig(f"{filedir19}/ftotal_{step}")
        
        if (phi_comp_plots == 1):
                    
            plt.close()
            plt.imshow(phi,origin='lower',cmap='jet')
            plt.colorbar()
            plt.title(f'phi_{step}')
            plt.savefig(f'{filedir1}/phi_{step}.png')
            
            plt.close()
            plt.imshow(comp,origin='lower',cmap='jet')
            plt.colorbar()
            plt.title(f'comp_{step}')
            plt.savefig(f'{filedir2}/comp_{step}.png')
            
        
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

if (movie == 1):
    ## Renaming the plots.
    if (phi_comp_plots == 1):
        files1 = sorted(os.listdir(filedir1), key=lambda x: int(x.split('_')[1].split('.')[0]))
        files2 = sorted(os.listdir(filedir2), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if (nph == 1):
        if (oneD_plots == 1):
            files3 = sorted(os.listdir(filedir3), key=lambda x: int(x.split('_')[1].split('.')[0]))
            files4 = sorted(os.listdir(filedir4), key=lambda x: int(x.split('_')[1].split('.')[0]))
#
        if (df_plots == 1):
            files5 = sorted(os.listdir(filedir5), key=lambda x: int(x.split('_')[1].split('.')[0]))
            files6 = sorted(os.listdir(filedir6), key=lambda x: int(x.split('_')[1].split('.')[0]))
            files7 = sorted(os.listdir(filedir7), key=lambda x: int(x.split('_')[1].split('.')[0]))

        if (elas == 1):
            if (sts_plots == 1):
                files8 = sorted(os.listdir(filedir8), key=lambda x: int(x.split('_')[1].split('.')[0]))
                files9 = sorted(os.listdir(filedir9), key=lambda x: int(x.split('_')[1].split('.')[0]))
                files10 = sorted(os.listdir(filedir10), key=lambda x: int(x.split('_')[1].split('.')[0]))
                if (Z_comp == 1):
                    files11 = sorted(os.listdir(filedir11), key=lambda x: int(x.split('_')[1].split('.')[0]))
                files12 = sorted(os.listdir(filedir12), key=lambda x: int(x.split('_')[1].split('.')[0]))

            if (str_plots == 1):
                files13 = sorted(os.listdir(filedir13), key=lambda x: int(x.split('_')[1].split('.')[0]))
                files14 = sorted(os.listdir(filedir14), key=lambda x: int(x.split('_')[1].split('.')[0]))
                files15 = sorted(os.listdir(filedir15), key=lambda x: int(x.split('_')[1].split('.')[0]))
                if (Z_comp == 1):
                    files16 = sorted(os.listdir(filedir16), key=lambda x: int(x.split('_')[1].split('.')[0]))
                files17 = sorted(os.listdir(filedir17), key=lambda x: int(x.split('_')[1].split('.')[0]))

            if (felas_plots == 1):
                files18 = sorted(os.listdir(filedir18), key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if (ftotal_plots == 1):
            files19 = sorted(os.listdir(filedir19), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if (phi_comp_plots == 1):
        print("renaminng the phi_plots")
        for i, file in enumerate(files1):
            os.rename(os.path.join(filedir1, file), os.path.join(filedir1, f'phi_{i}.png'))

        print("renaminng the comp_plots")
        for i, file in enumerate(files2):
            os.rename(os.path.join(filedir2, file), os.path.join(filedir2, f'comp_{i}.png'))
    
    if (nph == 1):
        if (oneD_plots == 1):
            print("renaminng the phi1D_plots")
            for i, file in enumerate(files3):
                os.rename(os.path.join(filedir3, file), os.path.join(filedir3, f'phi1D_{i}.png'))

            print("renaminng the comp1D_plots")
            for i, file in enumerate(files4):
                os.rename(os.path.join(filedir4, file), os.path.join(filedir4, f'comp1D_{i}.png'))

        if (df_plots == 1):
            print("renaminng the dfdc_plots")
            for i, file in enumerate(files5):
                os.rename(os.path.join(filedir5, file), os.path.join(filedir5, f'dfdc_{i}.png'))
            
            print("renaminng the dfdphi_plots")
            for i, file in enumerate(files6):
                os.rename(os.path.join(filedir6, file), os.path.join(filedir6, f'dfdphi_{i}.png'))

            print("renaminng the dfeldphi_plots")
            for i, file in enumerate(files7):
                os.rename(os.path.join(filedir7, file), os.path.join(filedir7, f'dfeldphi_{i}.png'))
 
        if (elas == 1):
            if (sts_plots == 1):
                print("renaminng the stress_plots")
                for i, file in enumerate(files8):
                    os.rename(os.path.join(filedir8, file), os.path.join(filedir8, f'S11_{i}.png'))
                for i, file in enumerate(files9):
                    os.rename(os.path.join(filedir9, file), os.path.join(filedir9, f'S22_{i}.png'))
                for i, file in enumerate(files10):
                    os.rename(os.path.join(filedir10, file), os.path.join(filedir10, f'S12_{i}.png'))
                if (Z_comp == 1):
                    for i, file in enumerate(files11):
                        os.rename(os.path.join(filedir11, file), os.path.join(filedir11, f'S33_{i}.png'))
                for i, file in enumerate(files12):
                    os.rename(os.path.join(filedir12, file), os.path.join(filedir12, f'Hsts_{i}.png'))

            if (str_plots == 1):
                print("renaminng the strain_plots")
                for i, file in enumerate(files13):
                    os.rename(os.path.join(filedir13, file), os.path.join(filedir13, f'E11_{i}.png'))
                for i, file in enumerate(files14):
                    os.rename(os.path.join(filedir14, file), os.path.join(filedir14, f'E22_{i}.png'))
                for i, file in enumerate(files15):
                    os.rename(os.path.join(filedir15, file), os.path.join(filedir15, f'E12_{i}.png'))
                if (Z_comp == 1):
                    for i, file in enumerate(files16):
                        os.rename(os.path.join(filedir16, file), os.path.join(filedir16, f'E33_{i}.png'))
                for i, file in enumerate(files17):
                    os.rename(os.path.join(filedir17, file), os.path.join(filedir17, f'Hstr_{i}.png'))
                
            if (felas_plots == 1):
                for i, file in enumerate(files18):
                    os.rename(os.path.join(filedir18, file), os.path.join(filedir18, f'felas_{i}.png'))
            
        if (ftotal_plots == 1):
            for i, file in enumerate(files19):
                os.rename(os.path.join(filedir19, file), os.path.join(filedir19, f'ftotal_{i}.png'))

    if (phi_comp_plots == 1):
        ## Making video out of the phi plots.
        print("making the phi_plots video")
        os.chdir(f"{filedir1}")
        command1 = "ffmpeg -framerate 10 -i phi_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p phi.mp4"
        subprocess.run(command1, shell=True)

        ## Making video out of the comp plots.
        print("making the comp_plots video")
        os.chdir(f"{filedir2}")
        command2 = "ffmpeg -framerate 10 -i comp_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p comp.mp4"
        subprocess.run(command2, shell=True)

    if (nph == 1):
        ## Making video out of the phi1D & comp1D plots.
        if (oneD_plots == 1):
            print("making the phi1D_plots video")
            os.chdir(f"{filedir3}")
            command3 = "ffmpeg -framerate 10 -i phi1D_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p phi1D.mp4"
            subprocess.run(command3, shell=True)

            print("making the comp1D_plots video")
            os.chdir(f"{filedir4}")
            command4 = "ffmpeg -framerate 10 -i comp1D_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p comp1D.mp4"
            subprocess.run(command4, shell=True)

        if (df_plots == 1):
            print("making the dfdc_plots video")
            os.chdir(f"{filedir5}")
            command5 = "ffmpeg -framerate 10 -i dfdc_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p dfdc.mp4"
            subprocess.run(command5, shell=True)
            
            print("making the dfdphi_plots video")
            os.chdir(f"{filedir6}")
            command6 = "ffmpeg -framerate 10 -i dfdphi_%d.p#ng -c:v libx264 -r 30 -pix_fmt yuv420p dfdphi.mp4"
            subprocess.run(command6, shell=True)

            print("making the dfeldphi_plots video")
            os.chdir(f"{filedir7}")
            command7 = "ffmpeg -framerate 10 -i dfeldphi_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p dfeldphi.mp4"
            subprocess.run(command7, shell=True)
        
        if (elas == 1):
            if (sts_plots == 1):
                print("making the stress_plots video")
                os.chdir(f"{filedir8}")
                command8 = "ffmpeg -framerate 10 -i S11_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p S11.mp4"
                subprocess.run(command8, shell=True)
                
                os.chdir(f"{filedir9}")
                command9 = "ffmpeg -framerate 10 -i S22_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p S22.mp4"
                subprocess.run(command9, shell=True)
                
                os.chdir(f"{filedir10}")
                command10 = "ffmpeg -framerate 10 -i S12_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p S12.mp4"
                subprocess.run(command10, shell=True)
                
                if (Z_comp == 1):
                    os.chdir(f"{filedir11}")
                    command11 = "ffmpeg -framerate 10 -i S33_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p S33.mp4"
                    subprocess.run(command11, shell=True)
             
                os.chdir(f"{filedir12}")
                command12 = "ffmpeg -framerate 10 -i Hsts_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p Hsts.mp4"
                subprocess.run(command12, shell=True)

            if (str_plots == 1):
                print("making the str#ain_plots video")
                os.chdir(f"{filedir13}")
                command13 = "ffmpeg -framerate 10 -i E11_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p E11.mp4"
                subprocess.run(command13, shell=True)
                
                os.chdir(f"{filedir14}")
                command14 = "ffmpeg -framerate 10 -i E22_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p E22.mp4"
                subprocess.run(command14, shell=True)
                
                os.chdir(f"{filedir15}")
                command15 = "ffmpeg -framerate 10 -i E12_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p E12.mp4"
                subprocess.run(command15, shell=True)

                if (Z_comp == 1):
                    os.chdir(f"{filedir16}")
                    command16 = "ffmpeg -framerate 10 -i E33_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p E33.mp4"
                    subprocess.run(command16, shell=True)
                
                os.chdir(f"{filedir17}")
                command17 = "ffmpeg -framerate 10 -i Hstr_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p Hstr.mp4"
                subprocess.run(command17, shell=True)

            if (felas_plots == 1):
                
                print("making the felas video")
                os.chdir(f"{filedir18}")
                command18 = "ffmpeg -framerate 10 -i felas_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p felas.mp4"
                subprocess.run(command18, shell=True)
            
        if (ftotal_plots == 1):
            
            print("making the ftotal video")
            os.chdir(f"{filedir19}")
            command19 = "ffmpeg -framerate 10 -i ftotal_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p ftotal.mp4"
            subprocess.run(command19, shell=True)

print("DONE")
