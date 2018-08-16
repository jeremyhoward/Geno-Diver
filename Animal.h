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
    int ID;                             /* ID of Animal */
    int Sire;                           /* Sire of Animal; 0 means unknown */
    int SireAge;                        /* Age of Sire used for Generation Interval */
    int Dam;                            /* Dam of Animal; 0 means unknown */
    int DamAge;                         /* Age of Dam; used for Generation Interval */
    int Sex;                            /* Sex of Animal */
    int Generation;                     /* Generation Animal was born in */
    int Age;                            /* Age of Animal when removed from breeding population */
    int Progeny;                        /* Number of progeny produced by individual */
    int Matings;                        /* number of matings for a given generation changes based on an animals age */
    int NumDead;                        /* number of progeny that died due to fitness */
    double RndSelection;                /* Random Uniform Deviate Used for Selection */
    double RndCulling;                  /* Random Uniform Deviate Used for Culling */
    double RndPheno1;                   /* Random Uniform Deviate Used for determine whether to phenotype an animal or not for Trait1 */
    double RndPheno2;                   /* Random Uniform Deviate Used for determine whether to phenotype an animal or not for Trait2 */
    double RndGeno;                     /* Random Uniform Deviate Used for determine whether to genotype an animal or not */
    double Ped_F;                       /* Pedigree based Inbreeding Coeffecient */
    double Gen_F;                       /* Diagonals of Genomic based Relationship Matrix */
    double Hap1_F;                      /* Diagonals of Haplotype based Relationship Matrix (Hickey et al. 2013; H1 matrix) */
    double Hap2_F;                      /* Diagonals of Haplotype based Relationship Matrix (Hickey et al. 2013; H2 matrix) */
    double Hap3_F;                      /* Diagonals of ROH based Relationship Matrix (Similar to Pryce et al. 2012) */
    int Unfav_Homozy_Leth;              /* Number of unfavorable lethal homozygous mutations */
    int Unfav_Heterzy_Leth;             /* Number of unfavorable lethal heterzygous mutations */
    int Unfav_Homozy_Subleth;           /* Number of unfavorable sub-lethal homozygous mutations */
    int Unfav_Heterzy_Subleth;          /* Number of unfavorable sub-lethal homozygous mutations */
    double Lethal_Equivalents;          /* Lethal Equivalents */
    double Homozy;                      /* Proportion of SNP and QTL homozygous */
    double Fitness;                     /* Fitness value of individual (multiplicative effect) */
    std::vector <double> Phenvect;      /* Phenotype of Animal = Genotypic + residual */
    std::vector <double> EBVvect;       /* Estimated Breeding Value from MME using either pedigree of genomic */
    std::vector <double> Accvect;       /* Accuracy of estimated breeding value */
    std::vector <double> GVvect;        /* True Genotypic Value of Animal */
    std::vector <double> BVvect;        /* True Breeding Value of Animal (Genotype * a; Biological Parameterization) */
    std::vector <double> DDvect;        /* True Dominance Deviation (heterozygote *d; Biological Parameterization)*/
    std::vector <double> Rvect;         /* Residual */
    std::vector <double> BVvectFalc;    /* True Breeding Value of Animal (Statistical Parameterization) */
    std::vector <double> DDvectFalc;    /* True Dominance Deviation (Statistical Parameterization)*/
    double ebvindex;                    /* EBV based index value */
    double tbvindex;                    /* TBV based index value */
    std::string Marker;                 /* Markers of Animal in a string i.e. 02342300302030340 */
    std::string QTL;                    /* All QTL's of Animal in a string i.e. 0102030300302020430 */
    std::string PaternalHapl;           /* Paternal Haplotype i.e. 0_40_20 */
    std::string MaternalHapl;           /* Maternal Haplotype i.e. 2_40_20 */
    std::string Pedigree_3_Gen;         /* 3 Generation Pedigree: sire_dam_MGS_MGD_PGS_PGD */
    double propROH;                     /* If ROH function called will do proportion of genome in given ROH cutoff for each individual */
    std::string GenoStatus;                 /* Is animal genotyped and if so what density (Full or Reduced) */
    std::vector <std::string> PhenStatus;   /* Whether an animal has a phenotype or not */
    std::string AnimalStage;                /* Used to Figure out when an animal can obtain phenotype; selcand, selcandebv, parent */
public:
    // constructors to create object within class
    Animal();
    Animal(int id = 0, int sire = 0, int sireage = 0, int dam = 0, int damage = 0, int sex = 0, int generation = 0, int age = 0, int progeny = 0, int matings = 0, int dead = 0, double rndsel = 0.0, double rndcul = 0.0, double rndphe1 = 0.0, double rndphe2 = 0.0, double rndgeno = 0.0, double pedf = 0.0, double genf = 0.0, double hap1f = 0.0, double hap2f = 0.0, double hap3f = 0.0, int unfavhomoleth = 0, int unfavheterleth = 0, int unfavhomosublet = 0, int unfavhetersublet = 0, double lethalequiv = 0.0, double homozy = 0.0, double fit = 0.0, std::string mark = "0" , std::string qtl = "0", std::string pathap = "0", std::string mathap = "0", std::string ped3g = "",double proproh = 0, std::string genostatus = "No",std::vector<double> vpheno = std::vector<double>(0),std::vector<double> vebv = std::vector<double>(0),std::vector<double> vacc = std::vector<double>(0),std::vector<double> vgv = std::vector<double>(0),std::vector<double> vbv = std::vector<double>(0),std::vector<double> vdd = std::vector<double>(0),std::vector<double> vr = std::vector<double>(0),double indexebv = 0.0, double indextbv = 0.0,std::vector<double> vbvfalc = std::vector<double>(0),std::vector<double> vddfalc = std::vector<double>(0),std::vector<std::string> phenstat = std::vector<std::string>(0),std::string animstg = "selcand");
    // destructor
    ~Animal();
    // Start of functions
    void UpdateInb(double temp);                                    /* Updates Pedigree Inbreeding Value */
    void UpdateGenInb(double temp);                                 /* Updates Genomic Inbreeding Value */
    void UpdateAge();                                               /* Updates Age of animal */
    void UpdateProgeny();                                           /* Updates Number of progeny Created */
    void UpdateRndCulling(double temp);                             /* Updates Culling Uniform Value */
    void UpdateQTLGenotype(std::string temp);                       /* Updates QTL quantitative genotype to include new mutations */
    void UpdateQTLFitGenotype(std::string temp);                    /* Updates QTL fitness genotype to include new mutations */
    void ZeroOutMatings();                                          /* Sets to zero each generation */
    void UpdateMatings(int temp);                                   /* Updates number of matings for a given sire */
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
    void UpdateSireAge(int temp);                                   /* Update age of sire when animal is born */
    void UpdateDamAge(int temp);                                    /* Update age of dam when animal is born */
    int getID(){return ID;}
    int getSire(){return Sire;}
    int getSireAge(){return SireAge;}
    int getDam(){return Dam;}
    int getDamAge(){return DamAge;}
    int getSex(){return Sex;}
    int getGeneration(){return Generation;}
    int getAge(){return Age;}
    int getProgeny(){return Progeny;}
    int getMatings(){return Matings;}
    int getDead(){return NumDead;}
    double getRndSelection(){return RndSelection;}
    double getRndCulling(){return RndCulling;}
    double getRndPheno1(){return RndPheno1;}
    double getRndPheno2(){return RndPheno2;}
    double getRndGeno(){return RndGeno;}
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
    double getFitness(){return Fitness;}
    std::string getMarker(){return Marker;}
    std::string getQTL(){return QTL;}
    std::string getPatHapl(){return PaternalHapl;}
    std::string getMatHapl(){return MaternalHapl;}
    std::string getPed3G(){return Pedigree_3_Gen;}
    double getpropROH(){return propROH;}
    std::string getGenoStatus(){return GenoStatus;}
    /* Phenotype vector functions */
    const std::vector <double>& get_Phenvect() {return Phenvect;}
    void add_Phenvect(double x){Phenvect.push_back(x);}
    void update_Phenvect(int i, double x);
    void clear_Phenvect(){Phenvect.clear();}
    /* EBV vector functions */
    const std::vector <double>& get_EBVvect() {return EBVvect;}
    void add_EBVvect(double x){EBVvect.push_back(x);}
    void update_EBVvect(int i, double x);
    void clear_EBVvect(){EBVvect.clear();}
    /* Accvect vector functions */
    const std::vector <double>& get_Accvect() {return Accvect;}
    void add_Accvect(double x){Accvect.push_back(x);}
    void update_Accvect(int i, double x);
    void clear_Accvect(){Accvect.clear();}
    /* GVvect vector functions */
    const std::vector <double>& get_GVvect() {return GVvect;}
    void add_GVvect(double x){GVvect.push_back(x);}
    void update_GVvect(int i, double x);
    void clear_GVvect(){GVvect.clear();}
    /* BVvect vector functions */
    const std::vector <double>& get_BVvect() {return BVvect;}
    void add_BVvect(double x){BVvect.push_back(x);}
    void update_BVvect(int i, double x);
    void clear_BVvect(){BVvect.clear();}
    /* DDvect vector functions */
    const std::vector <double>& get_DDvect() {return DDvect;}
    void add_DDvect(double x){DDvect.push_back(x);}
    void update_DDvect(int i, double x);
    void clear_DDvect(){DDvect.clear();}
    /* Rvect vector functions */
    const std::vector <double>& get_Rvect() {return Rvect;}
    void add_Rvect(double x){Rvect.push_back(x);}
    void update_Rvect(int i, double x);
    void clear_Rvect(){Rvect.clear();}
    /* EBV based index value */
    double getebvindex(){return ebvindex;}
    void Updateebvindex(double temp);                                   /* Update index tbv for an animal */
    /* TBV based index value */
    double gettbvindex(){return tbvindex;}
    void Updatetbvindex(double temp);                                   /* Update index tbv for an animal */
    /* True Breeding Value of Animal (Statistical Parameterization) */
    const std::vector <double>& get_BVvectFalc() {return BVvectFalc;}
    void add_BVvectFalc(double x){BVvectFalc.push_back(x);}
    void update_BVvectFalc(int i, double x);
    void clear_BVvectFalc(){BVvectFalc.clear();}
    /* True Dominance Deviation (Statistical Parameterization)*/
    const std::vector <double>& get_DDvectFalc() {return DDvectFalc;}
    void add_DDvectFalc(double x){DDvectFalc.push_back(x);}
    void update_DDvectFalc(int i, double x);
    void clear_DDvectFalc(){Rvect.clear();}
    /* PhenStatus vector functions */
    const std::vector <std::string>& get_PhenStatus() {return PhenStatus;}
    void add_PhenStatus(std::string x){PhenStatus.push_back(x);}
    void update_PhenStatus(int i, std::string x);
    void clear_PhenStatus(){PhenStatus.clear();}
    /* Used to Figure out when an animal can obtain phenotype; selcand, selcandebv, parent */
    std::string getAnimalStage(){return AnimalStage;}
    void UpdateAnimalStage(std::string temp);
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
    std::vector <double> Additivevect;  /* Additive/Selection Coeffecient Effect of QTL */
    std::vector <double> Dominancevect; /* Additive/Selection Coeffecient Effect of QTL */
    double AdditiveEffect;              /* Additive Effect of QTL */
    double DominanceEffect;             /* Dominance Effect of QTL */
    std::string Type;                   /* Whether it is a fitness or quantitative qtl */
    int GenOccured;                     /* Which Generated in originated */
    std::string Freq;                   /* Frequency for a given Generation (i.e. 0.2_0.19_0.24_etc..) */
    std::string LDDecay;                /* LD decay for a given generation */
public:
    // constructors to create object within class
    QTL_new_old();
    QTL_new_old(double location = 0.0,std::vector<double> vadd = std::vector<double>(0),std::vector<double> vdom = std::vector<double>(0), double add = 0.0, double dom = 0.0, std::string type = "0", int genOccured = 99, std::string freq = "0",std::string lddec = "");
    // destructor
    ~QTL_new_old();
    // Start of functions
    void UpdateFreq(std::string currentFreq);                       /* Function to update the current generation QTL freq */
    void UpdateLDDecay(std::string currentLDDecaya);                 /* Function to update the current generation LD decay */
    void FounderLDDecay(std::string currentLDDecay);                /* Function to update the founder generation LD decay */
    double getLocation(){return Location;}
    /* Additive vector functions */
    const std::vector <double>& get_Additivevect() {return Additivevect;}
    void add_Additivevect(double x){Additivevect.push_back(x);}
    void update_Additivevect(int i, double x);
    void clear_Additivevect(){Additivevect.clear();}
    /* Dominance vector functions */
    const std::vector <double>& get_Dominancevect() {return Dominancevect;}
    void add_Dominancevect(double x){Dominancevect.push_back(x);}
    void update_Dominancevect(int i, double x);
    void clear_Dominancevect(){Dominancevect.clear();}
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
