export prefix = $$(PWD)

export CC = gcc
export CXX = g++
export NVCC = nvcc
export AR = ar

export CFLAGS = -g -Wall -Wextra -pedantic -O3
export CXXFLAGS = -g -Wall -Wextra -pedantic -O3
export NVCCFLAGS = -g -G -O3 -gencode=arch=compute_20,code=compute_20 \
	-gencode=arch=compute_20,code=sm_20
export ARFLAGS = -cru

export top_srcdir = $$(PWD)

SUBDIRS = libArrayUtilities cpu_ref tools gpu_impl test

all: subdirs
.PHONY: all

.PHONY: subdirs $(SUBDIRS)
subdirs: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -w -C $@

SUBCLEAN = $(addsuffix .clean,$(SUBDIRS))
.PHONY: clean $(SUBCLEAN)
clean: $(SUBCLEAN)
$(SUBCLEAN): %.clean:
	$(MAKE) -w -C $* clean

SUBINSTALL = $(addsuffix .install,$(SUBDIRS))
.PHONY: install $(SUBINSTALL)
install: $(SUBINSTALL)
$(SUBINSTALL): %.install:
	$(MAKE) -w -C $* install

SUBCHECK = $(addsuffix .check,$(SUBDIRS))
.PHONY: check $(SUBCHECK)
check: $(SUBCHECK)
$(SUBCHECK): %.check:
	$(MAKE) -w -C $* check

.PHONY: deploy-zip
deploy-zip: sgm_project.zip
sgm_project.zip: libArrayUtilities/Makefile libArrayUtilities/CVector.h libArrayUtilities/CMatrix.h libArrayUtilities/CTensor.h libArrayUtilities/NMath.h libArrayUtilities/NMath.cc libArrayUtilities/timer.h libArrayUtilities/timer.cu tools/Makefile tools/disp-epe.cc tools/float3-to-pgm.cc cpu_ref/Makefile cpu_ref/sgm_cpu.cu gpu_impl/Makefile test/Makefile test/testReadWriteFloatFile.cc bin/evaluateOnSintel.sh Makefile README data
	mkdir -p sgm_project/libArrayUtilities sgm_project/tools sgm_project/cpu_ref sgm_project/gpu_impl sgm_project/test sgm_project/bin; \
	cp libArrayUtilities/Makefile libArrayUtilities/CVector.h libArrayUtilities/CMatrix.h libArrayUtilities/CTensor.h libArrayUtilities/NMath.h libArrayUtilities/NMath.cc libArrayUtilities/timer.h libArrayUtilities/timer.cu sgm_project/libArrayUtilities; \
	cp tools/Makefile tools/disp-epe.cc tools/float3-to-pgm.cc sgm_project/tools; \
	cp cpu_ref/Makefile cpu_ref/sgm_cpu.cu sgm_project/cpu_ref; \
	cp gpu_impl/Makefile sgm_project/gpu_impl; \
	cp test/Makefile test/testReadWriteFloatFile.cc sgm_project/test; \
	cp bin/evaluateOnSintel.sh sgm_project/bin; \
	cp -r Makefile README data sgm_project; \
	zip -r sgm_project.zip sgm_project; \
	rm -r sgm_project

tools: libArrayUtilities
cpu_ref: libArrayUtilities
gpu_impl: libArrayUtilities
