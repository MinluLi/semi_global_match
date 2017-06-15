disp-epe_OBJECTS = disp-epe.o
float3-to-pgm_OBJECTS = float3-to-pgm.o

CPPFLAGS = -I$(top_srcdir)/libArrayUtilities
LDFLAGS = -L$(top_srcdir)/libArrayUtilities
LIBS = -lArrayUtilities

all: disp-epe float3-to-pgm
.PHONY: all

disp-epe: $(disp-epe_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

float3-to-pgm: $(float3-to-pgm_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.PHONY: install
install: all
	mkdir -p $(prefix)/bin; cp disp-epe float3-to-pgm $(prefix)/bin

.PHONY: clean
clean:
	rm -f *~ disp-epe float3-to-pgm $(disp-epe_OBJECTS) $(float3-to-pgm_OBJECTS)

.PHONY: check
check:
