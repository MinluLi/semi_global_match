CC=gcc
CXX=g++
NVCC=nvcc
AR=ar

CFLAGS=-Wall -Wextra -pedantic -O2 
CXXFLAGS=-Wall -Wextra -pedantic -O2 
NVCCFLAGS=-O2 -Xcompiler '-fopenmp'\
          -gencode=arch=compute_30,code=compute_30 \
          -gencode=arch=compute_30,code=sm_30 \
          -gencode=arch=compute_35,code=compute_35 \
          -gencode=arch=compute_35,code=sm_35 \
          -gencode=arch=compute_50,code=compute_50 \
          -gencode=arch=compute_50,code=sm_50 \
          -gencode=arch=compute_61,code=compute_61 \
          -gencode=arch=compute_61,code=sm_61
#--ptxas-options=-v

all: sgm_gpu
.PHONY: all

sgm_gpu: sgm_gpu.cu timer.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *~ sgm_gpu 

.PHONY: install
install: sgm_gpu
	mkdir -p $(prefix)/bin; cp sgm_gpu $(prefix)/bin

.PHONY: check
check:
