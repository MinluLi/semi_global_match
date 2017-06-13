sgm_cpu_SOURCES = sgm_cpu.cu
sgm_cpu_CPPFLAGS = -I$(top_srcdir)/libArrayUtilities
sgm_cpu_LDFLAGS = -L$(top_srcdir)/libArrayUtilities
sgm_cpu_LIBS = -lArrayUtilities

all: sgm_cpu
.PHONY: all

sgm_cpu: $(sgm_cpu_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(sgm_cpu_LDFLAGS) $(sgm_cpu_CPPFLAGS) -o $@ $^ $(sgm_cpu_LIBS)

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $(sgm_cpu_CPPFLAGS) -ptx --source-in-ptx -o $@ $< 

.PHONY: clean
clean:
	rm -f *~ sgm_cpu $(sgm_cpu_OBJECTS) *.ptx

.PHONY: install
install: sgm_cpu
	mkdir -p $(prefix)/bin; cp sgm_cpu $(prefix)/bin

.PHONY: check
check:
