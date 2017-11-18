#ifndef HAPLOFINDERCLASSES_H_
#define HAPLOFINDERCLASSES_H_

// Class to store index start and stop site ##
class ROH_Index
{
private:
    int Chromosome;             /* Chromsome */
    int StartPosition;          /* Start Position nucleotides */
    int EndPosition;            /* End Position Nucleotides */
    int StartIndex;             /* Start Index */
    int EndIndex;               /* End Index */
    int NumberSNP;              /* Number of SNP in ROH */
public:
    ROH_Index();
    ROH_Index(int chr = 0, int stpos = 0, int enpos = 0, int stind = 0, int enind = 0, int numsnp = 0);
    ~ROH_Index();
    int getChr(){return Chromosome;}
    int getStPos(){return StartPosition;}
    int getEnPos(){return EndPosition;}
    int getStInd(){return StartIndex;}
    int getEnInd(){return EndIndex;}
    int getNumSNP(){return NumberSNP;}
};

////Summary of ROH output////////
class Ani_ROH
{
private:
    std::string ROHChrome;      /* Chromsomes for each ROH */
    std::string ROHStart;               /* ROH Start in Base number*/
    std::string ROHEnd;                 /* ROH End in Base number */
    std::string Distance;               /* Length of ROHs in megabases*/
public:
    Ani_ROH();
    Ani_ROH(std::string rohchromeO = "", std::string wsO = "", std::string weO = "", std::string dist_MBO = "");
    ~Ani_ROH();
    std::string getrohchromeO(){return ROHChrome;}
    std::string getwsO(){return ROHStart;}
    std::string getweO(){return ROHEnd;}
    std::string getdist_MBO(){return Distance;}
};


#endif
