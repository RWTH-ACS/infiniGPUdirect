.PHONY: clean

CC = nvcc
LD = nvcc

CUDA_SRC = /usr/local/cuda

INC_FLAGS += -I$(CUDA_SRC)/include 
INC_FLAGS += -I$(CUDA_SRC)/samples/common/inc 

CC_FLAGS += $(INC_FLAGS) -g
LIB_FLAGS += -L$(CUDA_SRC)/lib64
LD_FLAGS = $(LIB_FLAGS) -libverbs -lcudart -lm



ibTest: main.o ib.o oob.o
	$(LD) $(CC_FLAGS) -o $@ $^ $(LD_FLAGS)

ib.o: ib.c
	$(CC) $(CC_FLAGS) -c -o $@ $^ $(LD_FLAGS)

oob.o: oob.c
	$(CC) $(CC_FLAGS) -c -o $@ $^ $(LD_FLAGS)

main.o: main.c 
	$(CC) $(CC_FLAGS) -c -o $@ $^ $(LD_FLAGS)



clean:
	rm -f *.o ibTest	
