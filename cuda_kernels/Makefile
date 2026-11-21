NVCC    = nvcc
# Override per GPU: make ARCH=sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_80 (A100)
ARCH   ?= sm_86
CFLAGS  = -O3 -arch=$(ARCH) -std=c++17 -Xcompiler -Wall

all: matmul softmax

matmul: matmul.cu
	$(NVCC) $(CFLAGS) -o matmul matmul.cu

softmax: softmax.cu
	$(NVCC) $(CFLAGS) -o softmax softmax.cu

clean:
	rm -f matmul softmax

.PHONY: all clean
