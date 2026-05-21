NVCC    = nvcc
ARCH    = sm_80   # Change to sm_86 for RTX 30xx, sm_89 for RTX 40xx
CFLAGS  = -O3 -arch=$(ARCH) -std=c++17 -Xcompiler -Wall

all: matmul softmax

matmul: matmul.cu
	$(NVCC) $(CFLAGS) -o matmul matmul.cu

softmax: softmax.cu
	$(NVCC) $(CFLAGS) -o softmax softmax.cu

clean:
	rm -f matmul softmax

.PHONY: all clean
