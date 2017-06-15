testReadWriteFloatFile_OBJECTS = testReadWriteFloatFile.o
testReadWriteFloatFile_CPPFLAGS = -I$(top_srcdir)/libArrayUtilities
testReadWriteFloatFile_LDFLAGS = -L$(top_srcdir)/libArrayUtilities
testReadWriteFloatFile_LIBS = -lArrayUtilities
testReadWriteFloatFile_DEPS = \
	$(top_srcdir)/libArrayUtilities/libArrayUtilities.a \
	$(top_srcdir)/libArrayUtilities/CMatrix.h

all: 
.PHONY: all

testReadWriteFloatFile: $(testReadWriteFloatFile_OBJECTS)
	$(CXX) $(CXXFLAGS) $(testReadWriteFloatFile_LDFLAGS) $(testReadWriteFloatFile_CPPFLAGS) -o $@ $^ $(testReadWriteFloatFile_LIBS)

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $(testReadWriteFloatFile_a_CPPFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f *~ testReadWriteFloatFile $(testReadWriteFloatFile_OBJECTS)

.PHONY: install
install:

.PHONY: check
check: testReadWriteFloatFile $(testReadWriteFloatFile_DEPS)
	./testReadWriteFloatFile
