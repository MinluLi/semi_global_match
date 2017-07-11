export prefix = $$(PWD)

export CC = gcc
export CXX = g++
export NVCC = nvcc
export AR = ar

export CFLAGS = -Wall -Wextra -pedantic -O2 -fopenmp
export CXXFLAGS = -Wall -Wextra -pedantic -O2 -fopenmp
export NVCCFLAGS = -g -O3 -Xcompiler "-fopenmp" \
  -gencode=arch=compute_20,code=compute_20 \
	-gencode=arch=compute_20,code=sm_20 \
	-gencode=arch=compute_30,code=compute_30 \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_35,code=compute_35 \
	-gencode=arch=compute_35,code=sm_35 \
	-gencode=arch=compute_50,code=compute_50 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_61,code=compute_61 \
  -gencode=arch=compute_61,code=sm_61
#--ptxas-options=-v

export ARFLAGS = -cru

export top_srcdir = $$(PWD)

.PHONY: all
all: sgm_gpu

.PHONY: clean
clean:
	rm -f *~ *.o sgm_gpu

sgm_gpu: sgm_gpu.cu timer.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^
