#include <numeric>
#include <vector>
#include <complex>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cufft.h>
#include <cufftMp.h>
#include <mpi.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "include/box_iterator.hpp"
#include "include/error_checks.hpp"

void writeVTK(FILE *fp, std::vector<std::complex<double>> &phi, 
              std::vector<std::complex<double>> &u, size_t my_nx, size_t my_ny, double dx)
{

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Phasefield\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %lu %lu %d\n", my_ny, my_nx, 1);
    fprintf(fp, "ORIGIN 0 0 0\n");
    fprintf(fp, "SPACING %le %le %le\n", dx, dx, dx);
    fprintf(fp, "POINT_DATA %ld\n", (long)my_nx * (long)my_ny);
    fprintf(fp, "SCALARS COMPOSITION double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto &v : phi)
    {
        fprintf(fp, "%le\n", v.real());
    }

    fprintf(fp, "\n");
    fprintf(fp, "SCALARS DISPLACEMENT_X double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto &v : u)
    {
        fprintf(fp, "%le\n", v.real());
    }
}

__global__ void scaling(BoxIterator<cufftDoubleComplex> Begin,
                        BoxIterator<cufftDoubleComplex> End, int Nx, int Ny)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    Begin += id;
    if (Begin < End)
    {
       *Begin = {Begin->x / (double)(Nx * Ny), Begin->y / (double)(Nx *Ny)};
    }
}

__global__ void evolve(BoxIterator<cufftDoubleComplex> compBegin, BoxIterator<cufftDoubleComplex> compEnd, BoxIterator<cufftDoubleComplex> gBegin,
                       BoxIterator<cufftDoubleComplex> gEnd, int Nx, int Ny, double kappa, double M, double dt, double dx, double dy)
{

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    compBegin += id;
    gBegin += id;

    int x = (int)compBegin.x();
    int y = (int)compBegin.y();

    double kx,ky,k2;

    // double delkx = 2.0 * M_PI / (Nx * dx);
    // double delky = 2.0 * M_PI / (Ny * dx);
    // double delkz = 2.0 * M_PI / (Nz * dx);

    if (x <= Nx / 2)
    kx = (double)x * (double) (2.0 * M_PI /(dx * Nx));
    else
    kx = (double)(x-Nx) * (double) (2.0 * M_PI /(dx * Nx));
    if (y<= Ny / 2)
    ky = (double)y * (double) (2.0 * M_PI /(dy * Ny));
    else
    ky = (double)(y-Ny) * (double) (2.0 * M_PI /(dy * Ny));

    k2 = (kx * kx) + (ky * ky);
    // printf("k2=%lf\n",k2);

    if (compBegin < compEnd)
    {
    
	*compBegin = {(compBegin->x - (M * dt * k2 * gBegin->x))/(1.0 +(2.0 * dt * M * kappa * k2 *k2)),(compBegin->y - 
                (M * dt * k2 *gBegin->y))/(1.0 + (2.0 * dt * M * kappa *k2 *k2))};
        //compBegin->x = (compBegin->x - dt * k2 * gBegin->x) / (1 + dt * M * kappa * k2 * k2);
        //compBegin->y = (compBegin->- dt * k2 * gBegin->y) / (1 + dt * M * kappa * k2 * k2);
    }
}

__global__ void bulkfreenergy(BoxIterator<cufftDoubleComplex> compBegin, BoxIterator<cufftDoubleComplex> compEnd, BoxIterator<cufftDoubleComplex> gBegin,
                              BoxIterator<cufftDoubleComplex> gEnd, double A)
{

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    compBegin += id;
    gBegin += id;

    if (gBegin < gEnd)
    {
        *gBegin = { 2.0 * A * compBegin->x * (1.0 - compBegin->x) * (1 - 2.0 * compBegin->x),
                    2.0 * A * compBegin->y * (1.0 - compBegin->y) * (1 - 2.0 * compBegin->y) };
    }
}

int main(int argc, char **argv)
{
    double dx= 1.0;
    double dy= 1.0;
    double dt= 0.5;
    double A=1.0;
    double M =1.0;
    double kappa=1.0;
    int nt =10000;
    int Nx = 256;
    int Ny = 256; 
    int Nz =1;
    int write =50;
    char name[1000];

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0)
    {
        mkdir ("DATA", 777);
    }

    MPI_Barrier(comm);
    char directory[1000];
    sprintf(directory, "DATA/Processor_%d", rank);
    mkdir (directory, 777);

    int ndevices;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));

    int ranks_cutoff = Nx % size;
    size_t my_nx = (Nx / size) + (rank < ranks_cutoff ? 1 : 0);
    size_t my_ny = Ny;
    size_t my_nz = Nz;

    std ::vector<std::complex<double>> h_comp(my_nx * my_ny * my_nz);
    // std ::vector<std::complex<double>> hk2(my_nx * my_ny * my_nz);

    //long x = (long)rank * (long)my_nx, y = 0;

    for (auto &v : h_comp)
    {
        v = {0.45 + 0.1 * (double)rand() / (double)RAND_MAX, 0.0};
    }

    // for (unsigned int i =0; i<Nx; i++){
        // for(unsigned int j=0; j<Ny; j++){
            // if (j<=Ny/2)
            // hk2[i+Nx*j] += (j*delky) *(j*delky);
            // else
            // hk2[i+Nx +j] += (j-(double)Ny) * delky * (j-(double)Ny)* delky;
            // if (i<=Nx/2)
            // hk2[i+Nx*j] += (i*delkx) * (i*delkx);
            // else        
            // hk2[i+Nx*j] += (i-(double)Nx)*delkx * (i-(double)Nx)*delkx;
        // }
    // }

    cufftHandle plan = 0;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftMpAttachComm(plan, CUFFT_COMM_MPI, &comm));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    size_t workspace;
    CUFFT_CHECK(cufftMakePlan3d(plan, Nx, Ny, Nz, CUFFT_Z2Z, &workspace));

    cudaLibXtDesc *d_comp;
    // cudaLibXtDesc *dk2;
    
    
    CUFFT_CHECK(cufftXtMalloc(plan, &d_comp, CUFFT_XT_FORMAT_INPLACE));
    // CUFFT_CHECK(cufftXtMalloc(plan, &dk2, CUFFT_XT_FORMAT_INPLACE));

    // cufftResult_t result1 = cufftXtMalloc(plan, &d_comp, CUFFT_XT_FORMAT_INPLACE);
        // if (result1 != CUFFT_SUCCESS) {
            // const char* errorString1;
            // cufftGetErrorString(result, &errorString1);
            // printf("cuFFTMp error in cuFFT execution: %s\n", errorString1);
            // }
    
    CUFFT_CHECK(cufftXtMemcpy(plan, (void *)d_comp, (void *)h_comp.data(), CUFFT_COPY_HOST_TO_DEVICE));
    // CUFFT_CHECK(cufftXtMemcpy(plan, (void *)dk2, (void *)hk2.data(), CUFFT_COPY_HOST_TO_DEVICE));

    auto
        [compBegin_r, compEnd_r] = BoxIterators(CUFFT_XT_FORMAT_INPLACE, CUFFT_Z2Z,
                                                rank, size, Nx, Ny, Nz, (cufftDoubleComplex *)d_comp->descriptor->data[0]);

    auto
        [compBegin_f, compEnd_f] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                                rank, size, Nx, Ny, Nz, (cufftDoubleComplex *)d_comp->descriptor->data[0]);

    
    // auto
        // [dk2Begin_r, dk2End_r] = BoxIterators(CUFFT_XT_FORMAT_INPLACE, CUFFT_Z2Z,
                                            // rank, size, Nx, Ny, Nz, (cufftDoubleComplex *)dk2->descriptor->data[0]);
    // auto
        // [dk2Begin_f, dk2End_f] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                            // rank, size, Nx, Ny, Nz, (cufftDoubleComplex *)dk2->descriptor->data[0]);

    cudaLibXtDesc *g;

    CUFFT_CHECK(cufftXtMalloc(plan, &g, CUFFT_XT_FORMAT_INPLACE));
    // cufftResult_t result2 = cufftXtMalloc(plan, &d_comp, CUFFT_XT_FORMAT_INPLACE);
        // if (result2 != CUFFT_SUCCESS) {
            // const char* errorString2;
            // cufftGetErrorString(result, &errorString2);
            // printf("cuFFTMp error in cuFFT execution: %s\n", errorString2);
            // }
    // 

    auto
        [gBegin_r, gEnd_r] = BoxIterators(CUFFT_XT_FORMAT_INPLACE, CUFFT_Z2Z,
                                          rank, size, Nx, Ny, Nz, (cufftDoubleComplex *)g->descriptor->data[0]);

    auto
        [gBegin_f, gEnd_f] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                          rank, size, Nx, Ny, Nz, (cufftDoubleComplex *)g->descriptor->data[0]);

    const size_t num_elements = std::distance(compBegin_r, compEnd_r);
    const size_t num_threads = 128;
    const size_t num_blocks = (num_elements + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < nt+1; t++)
    {
	    printf("t=%d\n",t);
	
        bulkfreenergy<<<num_blocks, num_threads, 0, stream>>>(compBegin_r, compEnd_r, gBegin_r, gEnd_r,A);
        cudaDeviceSynchronize();
	    // cudaError_t kernelError1 = cudaGetLastError();
	    // if (kernelError1 != cudaSuccess) {
    	// printf("CUDA kernel launch error: %s\n", cudaGetErrorString(kernelError1));
    	// }
        
	    CUFFT_CHECK(cufftXtExecDescriptor(plan, g, g, CUFFT_FORWARD));
	    // cufftResult_t result3 = cufftXtExecDescriptor(plan, g, g, CUFFT_FORWARD);
        // if (result3 != CUFFT_SUCCESS) {
            // const char* errorString3;
            // cufftGetErrorString(result, &errorString3);
            // printf("cuFFTMp error in cuFFT execution: %s\n", errorString3);
            // }
	
        CUFFT_CHECK(cufftXtExecDescriptor(plan, d_comp, d_comp, CUFFT_FORWARD));
        // cufftResult_t result4 = cufftXtExecDescriptor(plan, d_comp, d_comp, CUFFT_FORWARD);
        // if (result4 != CUFFT_SUCCESS) {
            // const char* errorString4;
            // cufftGetErrorString(result, &errorString4);
            // printf("cuFFTMp error in cuFFT execution: %s\n", errorString4);
            // }

        evolve<<<num_blocks, num_threads,0, stream>>>(compBegin_f, compEnd_f, gBegin_f, gEnd_f,Nx,Ny, kappa, M,dt,dx,dy);
        cudaDeviceSynchronize();
	    // cudaError_t kernelError2 = cudaGetLastError();
	    // if (kernelError2 != cudaSuccess) {
    	// printf("CUDA kernel launch error: %s\n", cudaGetErrorString(kernelError2));
    	// }

        CUFFT_CHECK(cufftXtExecDescriptor(plan, d_comp, d_comp, CUFFT_INVERSE));
        // cufftResult_t result5 = cufftXtExecDescriptor(plan, d_comp, d_comp, CUFFT_INVERSE);
        // if (result5 != CUFFT_SUCCESS) {
            // const char* errorString5;
            // cufftGetErrorString(result, &errorString5);
            // printf("cuFFTMp error in cuFFT execution: %s\n", errorString5);
            // }
        // 
        
        CUFFT_CHECK(cufftXtExecDescriptor(plan, g, g, CUFFT_INVERSE));
        // cufftResult_t result6 = cufftXtExecDescriptor(plan, g, g, CUFFT_INVERSE);
        // if (result6 != CUFFT_SUCCESS) {
            // const char* errorString6;
            // cufftGetErrorString(result, &errorString6);
            // printf("cuFFTMp error in cuFFT execution: %s\n", errorString6);
            // }
        // 

        scaling<<<num_blocks, num_threads, 0, stream>>>(compBegin_r, compEnd_r,Nx,Ny);
        //scaling<<<num_blocks, num_threads, 0, stream>>>(gBegin_r, gEnd_r,Nx,Ny);
        cudaDeviceSynchronize();
	    // cudaError_t kernelError3 = cudaGetLastError();
	    // if (kernelError3 != cudaSuccess) {
    	// printf("CUDA kernel launch error: %s\n", cudaGetErrorString(kernelError3));
    	// }
    	// 
        if (t % write == 0)
        {
            // Transfer composition profile to host
            CUFFT_CHECK(cufftXtMemcpy(plan, (void *)h_comp.data(), (void *)d_comp, 
                        CUFFT_COPY_DEVICE_TO_HOST));
            
            // Write to VTK
            sprintf(name, "DATA/Processor_%d/Phase_u_%u.vtk", rank, t);
            FILE *fp = fopen(name, "w");

            writeVTK(fp, h_comp, h_comp, my_nx, my_ny,dx);

            fclose(fp);
            if (t == nt)
                break;
        }

    }
    CUDA_CHECK(	cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtFree(g));
    CUFFT_CHECK(cufftXtFree(d_comp)); 
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
}
