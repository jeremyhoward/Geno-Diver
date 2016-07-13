#include <iostream>
#include <string>
#include "Animal.h"

// constructors
Animal::Animal()        /* Animal Class */
{
    ID = 0; Sire = 0; Dam = 0; Sex = 0; Generation = 0; Age = 0; Progeny = 0; Matings = 0; NumDead = 0; RndSelection = 0.0; RndCulling = 0.0;
    Ped_F = 0.0; Gen_F = 0.0; Hap1_F = 0.0; Hap2_F = 0.0; Hap3_F = 0.0; Unfav_Homozy_Leth = 0; Unfav_Heterzy_Leth = 0; Unfav_Homozy_Subleth = 0;
    Unfav_Heterzy_Subleth = 0; Lethal_Equivalents = 0; Homozy = 0.0; EBV = 0.0; Acc = 0.0; P = 0.0; Fitness = 0.0; GV = 0.0; BV = 0.0; DD = 0.0; R = 0.0;
    Marker = "0"; QTL = "0"; PaternalHapl = "0"; MaternalHapl = "0"; Pedigree_3_Gen = "";
}

Animal::Animal(int id, int sire, int dam, int sex, int generation, int age, int progeny, int matings, int dead, double rndsel, double rndcul, double pedf, double genf, double hap1f, double hap2f, double hap3f, int unfavhomoleth, int unfavheterleth, int unfavhomosublet, int unfavhetersublet, double lethalequiv, double homozy, double ebv, double acc, double pheno, double fit, double gv, double bv, double dd, double res, std::string mark, std::string qtl, std::string pathap, std::string mathap, std::string ped3g)
{
    ID = id; Sire = sire; Dam = dam; Sex = sex; Generation = generation; Age = age; Progeny = progeny; Matings = matings;
    NumDead = dead; RndSelection = rndsel; RndCulling = rndcul; Ped_F = pedf; Gen_F = genf; Hap1_F = hap1f; Hap2_F = hap2f; Hap3_F = hap3f;
    Unfav_Homozy_Leth = unfavhomoleth; Unfav_Heterzy_Leth = unfavheterleth; Unfav_Homozy_Subleth = unfavhomosublet;
    Unfav_Heterzy_Subleth = unfavhetersublet; Lethal_Equivalents = lethalequiv;
    Homozy = homozy; EBV = ebv; Acc = acc; P = pheno; Fitness = fit; GV = gv; BV = bv; DD = dd; R = res; Marker = mark; QTL = qtl;
    PaternalHapl = pathap; MaternalHapl = mathap; Pedigree_3_Gen = ped3g;
}
QTL_new_old::QTL_new_old()  /* QTL_new_old Class */
{
    Location = 0.0; AdditiveEffect = 0.0; DominanceEffect = 0.0; Type = "0"; GenOccured = 99; Freq = "0";
}
QTL_new_old::QTL_new_old(double location, double add, double dom, std::string type, int genOccured, std::string freq)
{
    Location = location; AdditiveEffect = add; DominanceEffect = dom; Type = type; GenOccured = genOccured; Freq = freq;
}
hapLibrary::hapLibrary()    /* ROH_Index */
{
    HapID = 0;  StartIndex = 0; EndIndex = 0; haplotypes = "";
}
hapLibrary::hapLibrary(int hapid, int start, int end, std::string haps)
{
    HapID = hapid;  StartIndex = start; EndIndex = end; haplotypes = haps;
}
// destructor
Animal::~Animal(){}                     /* Animal Class */
QTL_new_old::~QTL_new_old(){}           /* QTL_new_old Class */
hapLibrary::~hapLibrary(){}             /* Haplotype Library */

// Start of Functions for Animal Class
void Animal::BaseGV(double meangv, double meanbv, double meandd)
{
    P = P - meangv;
    GV = GV - meangv;
    BV = BV - meanbv;
    DD = DD - meandd;
}
void Animal::UpdateInb(double temp){Ped_F = temp;}
void Animal::UpdateGenInb(double temp){Gen_F = temp;}
void Animal::UpdateAge(){Age = Age + 1;}
void Animal::UpdateProgeny(){Progeny = Progeny + 1;}
void Animal::UpdateRndCulling(double temp){RndCulling = temp;}
void Animal::UpdateQTLGenotype(std::string temp){QTL = temp;}
void Animal::ZeroOutMatings(){Matings = 0;}
void Animal::UpdateMatings(int temp){Matings = Matings + temp;}
void Animal::Update_EBV(double temp){EBV = temp;}
void Animal::Update_Acc(double temp){Acc = temp;}
void Animal::Update_Dead(){NumDead = NumDead + 1;}
void Animal::Update_PatHap(std::string temp){PaternalHapl = temp;}
void Animal::Update_MatHap(std::string temp){MaternalHapl = temp;}
void Animal::AccumulateH1(double temp){Hap1_F += temp;}
void Animal::AccumulateH2(double temp){Hap2_F += temp;}
void Animal::AccumulateH3(double temp){Hap3_F += temp;}
void Animal::StandardizeH1(double temp){Hap1_F = Hap1_F / temp;}
void Animal::StandardizeH2(double temp){Hap2_F = Hap2_F / temp;}
void Animal::StandardizeH3(double temp){Hap3_F = Hap3_F / temp;}
void Animal::Update3GenPed(std::string temp){Pedigree_3_Gen = temp;}


// Start of Functions for QTL class
void QTL_new_old::UpdateFreq(std::string currentFreq){Freq = Freq + "_" + currentFreq;}

// Start of Functions for ROH Class
void hapLibrary::UpdateHaplotypes(std::string temp){haplotypes = temp;}

