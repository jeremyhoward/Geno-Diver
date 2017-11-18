#ifndef Animal_H_
#define Animal_H_
#include <string>
#include <vector>
/************************************************************************/
/************************************************************************/
/* Animal: class that holds information regarding a particular animal   */
/************************************************************************/
/************************************************************************/
class Animal
{
private:
    int ID;                         /* ID of Animal */
    int Sire;                       /* Sire of Animal; 0 means unknown */
    int Dam;                        /* Dam of Animal; 0 means unknown */
    int Sex;                        /* Sex of Animal */
    int Generation;                 /* Generation Animal was born in */
    int Age;                        /* Age of Animal when removed from breeding population */
    int Progeny;                    /* Number of progeny produced by individual */
    int Matings;                    /* number of matings for a given generation changes based on an animals age */
    int NumDead;                    /* number of progeny that died due to fitness */
    double RndSelection;            /* Random Uniform Deviate Used for Selection */
    double RndCulling;              /* Random Uniform Deviate Used for Culling */
    double Ped_F;                   /* Pedigree based Inbreeding Coeffecient */
    double Gen_F;                   /* Diagonals of Genomic based Relationship Matrix */
    double Hap1_F;                  /* Diagonals of Haplotype based Relationship Matrix (Hickey et al. 2013; H1 matrix) */
    double Hap2_F;                  /* Diagonals of Haplotype based Relationship Matrix (Hickey et al. 2013; H2 matrix) */
    double Hap3_F;                  /* Diagonals of ROH based Relationship Matrix (Similar to Pryce et al. 2012) */
    int Unfav_Homozy_Leth;          /* Number of unfavorable lethal homozygous mutations */
    int Unfav_Heterzy_Leth;         /* Number of unfavorable lethal heterzygous mutations */
    int Unfav_Homozy_Subleth;       /* Number of unfavorable sub-lethal homozygous mutations */
    int Unfav_Heterzy_Subleth;      /* Number of unfavorable sub-lethal homozygous mutations */
    double Lethal_Equivalents;      /* Lethal Equivalents */
    double Homozy;                  /* Proportion of SNP and QTL homozygous */
    double EBV;                     /* Estimated Breeding Value from MME using either pedigree of genomic */
    double Acc;                     /* Accuracy of estimated breeding value */
    double P;                       /* Phenotype of Animal = Genotypic + residual */
    double Fitness;                 /* Fitness value of individual (multiplicative effect) */
    double GV;                      /* True Genotypic Value of Animal  */
    double BV;                      /* True Breeding Value of Animal (Genotype * a) */
    double DD;                      /* True Dominance Deviation */
    double R;                       /* Residual  */
    std::string Marker;             /* Markers of Animal in a string i.e. 02342300302030340 */
    std::string QTL;                /* All QTL's of Animal in a string i.e. 0102030300302020430 */
    std::string PaternalHapl;       /* Paternal Haplotype i.e. 0_40_20 */
    std::string MaternalHapl;       /* Maternal Haplotype i.e. 2_40_20 */
    std::string Pedigree_3_Gen;     /* 3 Generation Pedigree: sire_dam_MGS_MGD_PGS_PGD */
    double propROH;                 /* If ROH function called will do proportion of genome in given ROH cutoff for each individual */
    std::string GenoStatus;         /* Is animal genotyped and if so what density (Full or Reduced) */
public:
    // constructors to create object within class
    Animal();
    Animal(int id = 0, int sire = 0, int dam = 0, int sex = 0, int generation = 0, int age = 0, int progeny = 0, int matings = 0, int dead = 0, double rndsel = 0.0, double rndcul = 0.0, double pedf = 0.0, double genf = 0.0, double hap1f = 0.0, double hap2f = 0.0, double hap3f = 0.0, int unfavhomoleth = 0, int unfavheterleth = 0, int unfavhomosublet = 0, int unfavhetersublet = 0, double lethalequiv = 0.0, double homozy = 0.0, double ebv = 0.0, double acc = 0.0, double pheno = 0.0, double fit = 0.0, double gv = 0.0, double bv = 0.0, double dd = 0.0, double res = 0.0, std::string mark = "0" , std::string qtl = "0", std::string pathap = "0", std::string mathap = "0", std::string ped3g = "",double proproh = 0, std::string genostatus = "No");
    // destructor
    ~Animal();
    // Start of functions
    void BaseGV(double meangv, double meanbv, double meandd);       /* Function to update base generation BV */
    void UpdateInb(double temp);                                    /* Updates Pedigree Inbreeding Value */
    void UpdateGenInb(double temp);                                 /* Updates Genomic Inbreeding Value */
    void UpdateAge();                                               /* Updates Age of animal */
    void UpdateProgeny();                                           /* Updates Number of progeny Created */
    void UpdateRndCulling(double temp);                             /* Updates Culling Uniform Value */
    void UpdateQTLGenotype(std::string temp);                       /* Updates QTL quantitative genotype to include new mutations */
    void UpdateQTLFitGenotype(std::string temp);                    /* Updates QTL fitness genotype to include new mutations */
    void ZeroOutMatings();                                          /* Sets to zero each generation */
    void UpdateMatings(int temp);                                   /* Updates number of matings for a given sire */
    void Update_EBV(double temp);                                   /* Updates Breeding Values */
    void Update_Acc(double temp);                                   /* Updates Accuracy */
    void Update_Dead();                                             /* Updates Dead Progeny count */
    void Update_PatHap(std::string temp);                           /* Updates Paternal Haplotype */
    void Update_MatHap(std::string temp);                           /* Updates Maternal Haplotype */
    void AccumulateH1(double temp);                                 /* Accumulates Haplotype 1 Diagonal Relationship Matrix */
    void AccumulateH2(double temp);                                 /* Accumulates Haplotype 2 Diagonal Relationship Matrix */
    void AccumulateH3(double temp);                                 /* Accumulates Haplotype 2 Diagonal Relationship Matrix */
    void StandardizeH1(double temp);                                /* Standardized by number of haplotypes */
    void StandardizeH2(double temp);                                /* Standardized by number of haplotypes */
    void StandardizeH3(double temp);                                /* Standardized by number of haplotypes */
    void Update3GenPed(std::string temp);                           /* Update 3 Generation Pedigree */
    void UpdatepropROH(double temp);                                /* Update proportion in ROH */
    void UpdateGenoStatus(std::string temp);                        /* Update whether animal is genotyped */
    int getID(){return ID;}
    int getSire(){return Sire;}
    int getDam(){return Dam;}
    int getSex(){return Sex;}
    int getGeneration(){return Generation;}
    int getAge(){return Age;}
    int getProgeny(){return Progeny;}
    int getMatings(){return Matings;}
    int getDead(){return NumDead;}
    double getRndSelection(){return RndSelection;}
    double getRndCulling(){return RndCulling;}
    double getPed_F(){return Ped_F;}
    double getGen_F(){return Gen_F;}
    double getHap1_F(){return Hap1_F;}
    double getHap2_F(){return Hap2_F;}
    double getHap3_F(){return Hap3_F;}
    int getunfavhomolethal(){return Unfav_Homozy_Leth;}
    int getunfavheterolethal(){return Unfav_Heterzy_Leth;}
    int getunfavhomosublethal(){return Unfav_Homozy_Subleth;}
    int getunfavheterosublethal(){return Unfav_Heterzy_Subleth;}
    double getlethalequiv(){return Lethal_Equivalents;}
    double getHomozy(){return Homozy;}
    double getEBV(){return EBV;}
    double getAcc(){return Acc;}
    double getPhenotype(){return P;}
    double getGenotypicValue(){return GV;}
    double getBreedingValue(){return BV;}
    double getDominanceDeviation(){return DD;}
    double getResidual(){return R;}
    double getFitness(){return Fitness;}
    std::string getMarker(){return Marker;}
    std::string getQTL(){return QTL;}
    std::string getPatHapl(){return PaternalHapl;}
    std::string getMatHapl(){return MaternalHapl;}
    std::string getPed3G(){return Pedigree_3_Gen;}
    double getpropROH(){return propROH;}
    std::string getGenoStatus(){return GenoStatus;}
};

/************************************************************************/
/************************************************************************/
/*    QTL_new_old: information regarding the QTL across generations     */
/************************************************************************/
/************************************************************************/
class QTL_new_old
{
private:
    double Location;            /* The location of QTL (Chr.Position) and serves as ID */
    double AdditiveEffect;      /* Additive Effect of QTL */
    double DominanceEffect;     /* Dominance Effect of QTL */
    std::string Type;           /* Whether it is a fitness or quantitative qtl */
    int GenOccured;             /* Which Generated in originated */
    std::string Freq;           /* Frequency for a given Generation (i.e. 0.2_0.19_0.24_etc..) */
    std::string LDDecay;        /* LD decay for a given generation */
public:
    // constructors to create object within class
    QTL_new_old();
    QTL_new_old(double location = 0.0, double add = 0.0, double dom = 0.0, std::string type = "0", int genOccured = 99, std::string freq = "0",std::string lddec = "");
    // destructor
    ~QTL_new_old();
    // Start of functions
    void UpdateFreq(std::string currentFreq);                       /* Function to update the current generation QTL freq */
    void UpdateLDDecay(std::string currentLDDecaya);                 /* Function to update the current generation LD decay */
    void FounderLDDecay(std::string currentLDDecay);                /* Function to update the founder generation LD decay */
    double getLocation(){return Location;}
    double getAdditiveEffect(){return AdditiveEffect;}
    double getDominanceEffect(){return DominanceEffect;}
    std::string getType(){return Type;}
    int getGenOccured(){return GenOccured;}
    std::string getFreq(){return Freq;}
    std::string getLDDecay(){return LDDecay;}
};

/************************************************************************/
/************************************************************************/
/*    hapLibrary: information regarding the haplotype across generations     */
/************************************************************************/
/************************************************************************/
class hapLibrary
{
private:
    int HapID;                  /* ID of haplotype will refer to which loop it is */
    int StartIndex;             /* Start Position snp */
    int EndIndex;               /* End Position snp */
    std::string haplotypes;     /* string of unique haplotypes can grow as generations proceed*/
public:
    hapLibrary();
    hapLibrary(int hapid = 0, int start = 0, int end = 0, std::string haps ="");
    ~hapLibrary();
    int gethapID(){return HapID;}
    int getStart(){return StartIndex;}
    int getEnd(){return EndIndex;}
    std::string getHaplo(){return haplotypes;}
    void UpdateHaplotypes(std::string temp);         /* Updates Culling Uniform Value */
};
#endif
