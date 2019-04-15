CXX=g++
COMMON_FLAGS = -O2 -I. 
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler $(COMMON_FLAGS)

%.o: %.c
	nvcc $(NVCCFLAGS) -dc $< -o $@

%.o: %.cu
	nvcc $(NVCCFLAGS) -dc $< -o $@

objects := fft_cuda.o fft.o fft_cuda1.o 

all: ${objects}
	nvcc $(NVCCFLAGS) -o fft ${objects}
clean:
	rm -f fft ${objects}
