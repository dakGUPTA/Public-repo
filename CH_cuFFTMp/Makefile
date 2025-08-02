#MPI_HOME    ?=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/comm_libs/hpcx/hpcx-2.13/ompi/
#NVSHMEM_LIB     =/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/11.8/lib64/compat/nvshmem_2.6.0-1
#CUDA_HOME   ?= /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda
#CUFFT_LIB   ?= /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/lib64
#CUFFT_INC   ?= /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/include/cufftmp
ARCH        ?= $(shell uname -m)
ifeq ($(ARCH), ppc64le)
MPI         ?= mpi_ibm
else
MPI         ?= mpi
endif

NPROCS = 4 

exe = a.out 

DEPS = include/*.*

all: $(exe)

clear:
	rm -rf DATA/

.PHONY: clean

clean:
	rm -rf $(exe)

$(exe): main.cu $(DEPS)
		${NVHPC_CUDA_HOME}/bin/nvcc -arch=sm_70 $< -o $@ -lcuda -lcudart -std=c++17 \
		-L${CUFFT_LIB} -I${CUFFT_INC} -lcufftMp \
		-I${MPI_HOME}/include -L${MPI_HOME}/lib -l${MPI} \
		-I.
build: $(exe)

run: $(exe)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -n  $(NPROCS) $(exe)

reconstruct: reconstruct.c
	gcc reconstruct.c -o reconstruct

