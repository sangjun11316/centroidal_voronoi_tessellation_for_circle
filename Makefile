# --------------------------------------------------------------------
# 					simple example Makefile
# --------------------------------------------------------------------
CURRENT_DIR = $(shell pwd)

# Libraries
#HDF5_DIR    = /home/sangjun11316/local/hdf5-1.12.0_oneapi
HDF5_DIR    = /home/sangjun11316/local/hdf5-1.12.0_gpp/
INC_HDF5    = -I/$(HDF5_DIR)/include
LIB_HDF5    = -L/$(HDF5_DIR)/lib -lhdf5_hl -lhdf5
#LIB_HDF5   += -lz

# --------------------------------------------------------------------
# Lists of source, modules, and objects
SOURCES = \
	./io/io.cpp \
	./src/knncuda.cu

INC_SOURCES = \
	-I/$(CURRENT_DIR)/helper \
	-I/$(CURRENT_DIR)/io \
	-I/$(CURRENT_DIR)/src

# --------------------------------------------------------------------
INCLUDE = $(INC_HDF5) $(INC_SOURCES)
LIB     = $(LIB_HDF5) -lcublas -lcuda -lcudart -lcurand -Wno-deprecated-gpu-targets

# C compiler
#LINK    = icc
LINK    = nvcc -std=c++17 -arch=sm_86 --relocatable-device-code=true #-rdc=true # -std=gnu99
#LINK   += --disable-warnings 
LINK   += -Xptxas -O3
#LINK   += -lineinfo
#LINK   += -g -G
FCFLAGS = $(INCLUDE) $(LIB)
#FCFLAGS = -std=c99 -lm $(INCLUDE) $(LIB)
#FCFLAGS = -g -traceback -fp-trap-all=all -std=c99 -lm $(INCLUDE) $(LIB)
#FCFLAGS = -O2 -std=c99 -lm $(INCLUDE) $(LIB)

EXE     = exe

# Executable:
$(EXE): $(SOURCES)
	$(LINK) $(SOURCES) ./main.cu $(FCFLAGS) -o $(EXE)
	rm -f *.o

clean:
	rm -f *.o $(EXE)

