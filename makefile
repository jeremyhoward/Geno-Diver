#-------------------------------------------#
# Make file for Geno-Driver
#
# Supported platforms: Unix/Linux and Mac
#-------------------------------------------#
# Director of the target
OUTPUT = GenoDiver

### which os: 'linux' or 'mac'
SYSTEM = mac

# Compiler
CXX = /usr/local/bin/g++-8

# mkl and eigen paths (change if compiling on own)
MKLPATH = /opt/intel/mkl
EIGEN_PATH = /Users/jeremyhoward/Documents/C++Code/eigen-eigen-b3f3d4950030

#MKLPATH = /home/jthoward/mkl/mkl
#EIGEN_PATH = /home/jthoward/Simulation/Ne1000/eigen-eigen-bdd17ee3b1b3

# Compiler flags
CXXFLAGS = -g -w -lz -O3 -DMKL_ILP64 -m64 -fopenmp -std=c++11 -I ${EIGEN_PATH} -I ${MKLPATH}/include

ifeq ($(SYSTEM),mac)
	LIBS = -L${MKLPATH}/lib
	LDFLAGS = -Wl,${MKLPATH}/lib/libmkl_intel_ilp64.a ${MKLPATH}/lib/libmkl_sequential.a ${MKLPATH}/lib/libmkl_core.a -lpthread -lm -ldl
endif

ifeq ($(SYSTEM),linux)
	LIBS = -L$(MKLPATH)/lib/intel64
	LDFLAGS =  -Wl,--start-group ${MKLPATH}/lib/intel64/libmkl_intel_ilp64.a ${MKLPATH}/lib/intel64/libmkl_core.a ${MKLPATH}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -lpthread -lm -ldl
endif

HDR += Animal.h \
	HaplofinderClasses.h \
	MatingDesignClasses.h \
	ParameterClass.h \
	Genome_ROH.h \
	OutputFiles.h \
	Global_Population.h \
	zfstream.h

SRC = PopulationSimulator.cpp \
	AnimalFun.cpp \
	Simulation_Functions.cpp \
	HaplofinderClasses.cpp \
	MatingDesignClasses.cpp \
	SelectionCullingFunctions.cpp \
	ParameterClass.cpp \
	Genome_ROH.cpp \
	EBV_Functions.cpp \
	SetUpGenome.cpp \
	OutputFiles.cpp \
	Global_Population.cpp \
	zfstream.cpp

OBJ = $(SRC:.cpp=.o)

all : $(OUTPUT)

$(OUTPUT) :
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LIB) $(LDFLAGS)

$(OBJ) : $(HDR)

.cpp.o :
	$(CXX) $(CXXFLAGS) -c $*.cpp
.SUFFIXES : .cpp .c .o $(SUFFIXES)

$(OUTPUT) : $(OBJ)

FORCE:

clean:
	rm -f *.o
