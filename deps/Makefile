CXX_RELEASE_FLAGS = -Wall -std=c++11 -fPIC -mavx2 -O3 -march=native
CXX_DEBUG_FLAGS   = -Wall -std=c++11 -fPIC -mavx2 -O0 -g

.PHONY: release
release: CXXFLAGS = $(CXX_RELEASE_FLAGS)
release: libsimdalign.so

.PHONY: debug
debug: CXXFLAGS = $(CXX_DEBUG_FLAGS)
debug: libsimdalign.so prof

.PHONY: clean
clean:
	rm -rf *.o *.so

libsimdalign.so: simdalign.o paralign.o
	$(CXX) -dynamiclib $(CXXFLAGS) $^ -o $@

prof: prof.cpp simdalign.o paralign.o
	$(CXX) $(CXXFLAGS) -g $^ -o $@

simdalign.o: simdalign.cpp simdalign.h simd.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

paralign.o: paralign.cpp simdalign.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<
