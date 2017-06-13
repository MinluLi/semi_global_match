libArrayUtilities_a_HEADERS = CVector.h CMatrix.h CTensor.h NMath.h timer.h
libArrayUtilities_a_SOURCES = NMath.cc
libArrayUtilities_a_OBJECTS = NMath.o timer.cuo
libArrayUtilities_a_CPPFLAGS = -I.

all: libArrayUtilities.a
.PHONY: all

libArrayUtilities.a: $(libArrayUtilities_a_OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $(libArrayUtilities_a_CPPFLAGS) -o $@ $<

%.cuo: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(libArrayUtilities_a_CPPFLAGS) -o $@ $<

.PHONY: install
install: libArrayUtilities.a $(libArrayUtilities_a_HEADERS)
	mkdir -p $(prefix)/include; \
	cp $(libArrayUtilities_a_HEADERS) $(prefix)/include
	mkdir -p $(prefix)/lib; \
	cp libArrayUtilities.a $(prefix)/lib; \

.PHONY: clean
clean:
	rm -f *~ libArrayUtilities.a $(libArrayUtilities_a_OBJECTS)

.PHONY: check
check:
