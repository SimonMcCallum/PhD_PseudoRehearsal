# Makefile for CUDA-accelerated Hopfield network
# Targets RTX 3070 (Compute 8.6)

NVCC = nvcc
CC = cl

# CUDA architecture for RTX 3070
CUDA_ARCH = -arch=sm_86

# Paths
CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
CUBLAS_LIB = $(CUDA_PATH)/lib/x64

NVCC_FLAGS = $(CUDA_ARCH) -O3 -Xcompiler "/W3 /O2"
LINK_FLAGS = -lcublas

# Source files
CUDA_SRC = gpu_hopfield.cu
C_SOURCES = ../annescode.c ../allocation.c ../display.c ../general.c \
            ../learning.c ../net.c ../param.c ../patternFileIO.c \
            ../pseudolearning.c ../random.c ../sampling.c \
            ../training.c ../unlearning.c

# Build GPU-accelerated version
all: annescode_gpu.exe

# Compile CUDA kernels
gpu_hopfield.obj: $(CUDA_SRC) gpu_hopfield.h
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_SRC) -o gpu_hopfield.obj

# For now, build the standalone GPU library
gpu_lib: gpu_hopfield.obj
	@echo "GPU kernels compiled successfully"

# Test compilation
test_compile: gpu_hopfield.obj
	@echo "CUDA compilation successful"

clean:
	del /Q *.obj *.exe *.exp *.lib 2>nul

.PHONY: all clean gpu_lib test_compile
