#include <iostream>
#include <vector>
#ifndef MatingDesignClasses_H_
#define MatingDesignClasses_H_

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// Class to store index start and stop site for each chromosome //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
class MatingClass
{
private:
    int ID;                     /* ID of individual */
    int AnimalType;             /* Sex */
    std::vector<int> MateIDs;   /* MateIDs */
    int Matings;                /* Number of matings */
    std::vector<int> OwnIndex;  /* Where at in matingclass relationship matrix */
    std::vector<int> MateIndex; /* Where mates are at in matingclass relationship matrix */
    
public:
    MatingClass();
    MatingClass(int animid = -1, int type = -1, std::vector<int> mateids = std::vector<int>(0),int mates = -1, std::vector<int> ownind = std::vector<int>(0), std::vector<int> mateindex = std::vector<int>(0));
    ~MatingClass();
    int getID_MC(){return ID;}
    int getType_MC(){return AnimalType;}
    int getMatings_MC(){return Matings;}
    void UpdateMateNumber(int temp){Matings = temp;}
    /* Mate Id vector functions */
    const std::vector <int>& get_mateIDs() {return MateIDs;}
    void add_ToMates(int x){MateIDs.push_back(x);}
    void clear_MateIDs(){MateIDs.clear();}
    /* Own Index vector functions */
    const std::vector <int>& get_OwnIndex() {return OwnIndex;}
    void add_ToOwnIndex(int x){OwnIndex.push_back(x);}
    void clear_OwnIndex(){OwnIndex.clear();}
    /* Mate Index vector functions */
    const std::vector <int>& get_mateIndex() {return MateIndex;}
    void add_ToMateIndex(int x){MateIndex.push_back(x);}
    void clear_MateIndex(){MateIndex.clear();}
    
    /* Print portion of variables */
    void showdata(){std::cout<<ID<< " "<< AnimalType << " " << Matings << std::endl;}
};

#endif
