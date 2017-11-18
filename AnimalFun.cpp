#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <ctime>
#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <set>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include "Animal.h"
#include "HaplofinderClasses.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"
#include "Genome_ROH.h"

using namespace std;

/************************************************************************/
/************************************************************************/
/* Animal: class that holds information regarding a particular animal   */
/************************************************************************/
/************************************************************************/
// constructors
Animal::Animal()
{
    ID = 0; Sire = 0; Dam = 0; Sex = 0; Generation = 0; Age = 0; Progeny = 0; Matings = 0; NumDead = 0; RndSelection = 0.0;
    RndCulling = 0.0; Ped_F = 0.0; Gen_F = 0.0; Hap1_F = 0.0; Hap2_F = 0.0; Hap3_F = 0.0; Unfav_Homozy_Leth = 0;
    Unfav_Heterzy_Leth = 0; Unfav_Homozy_Subleth = 0; Unfav_Heterzy_Subleth = 0; Lethal_Equivalents = 0; Homozy = 0.0;
    EBV = 0.0; Acc = 0.0; P = 0.0; Fitness = 0.0; GV = 0.0; BV = 0.0; DD = 0.0; R = 0.0;
    Marker = "0"; QTL = "0"; PaternalHapl = "0"; MaternalHapl = "0"; Pedigree_3_Gen = ""; propROH = 0; GenoStatus = "No";
}
Animal::Animal(int id, int sire, int dam, int sex, int generation, int age, int progeny, int matings, int dead, double rndsel, double rndcul, double pedf, double genf, double hap1f, double hap2f, double hap3f, int unfavhomoleth, int unfavheterleth, int unfavhomosublet, int unfavhetersublet, double lethalequiv, double homozy, double ebv, double acc, double pheno, double fit, double gv, double bv, double dd, double res, string mark, string qtl, string pathap, string mathap, string ped3g, double proproh, string genostatus)
{
    ID = id; Sire = sire; Dam = dam; Sex = sex; Generation = generation; Age = age; Progeny = progeny; Matings = matings;
    NumDead = dead; RndSelection = rndsel; RndCulling = rndcul; Ped_F = pedf; Gen_F = genf; Hap1_F = hap1f; Hap2_F = hap2f; Hap3_F = hap3f;
    Unfav_Homozy_Leth = unfavhomoleth; Unfav_Heterzy_Leth = unfavheterleth; Unfav_Homozy_Subleth = unfavhomosublet;
    Unfav_Heterzy_Subleth = unfavhetersublet; Lethal_Equivalents = lethalequiv;
    Homozy = homozy; EBV = ebv; Acc = acc; P = pheno; Fitness = fit; GV = gv; BV = bv; DD = dd; R = res; Marker = mark; QTL = qtl;
    PaternalHapl = pathap; MaternalHapl = mathap; Pedigree_3_Gen = ped3g; propROH = proproh; GenoStatus = genostatus;
}
// destructor
Animal::~Animal(){}                     /* Animal Class */
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
void Animal::UpdatepropROH(double temp){propROH = temp;}
void Animal::UpdateGenoStatus(std::string temp){GenoStatus = temp;}

/************************************************************************/
/************************************************************************/
/*    QTL_new_old: information regarding the QTL across generations     */
/************************************************************************/
/************************************************************************/
// constructors
QTL_new_old::QTL_new_old()  /* QTL_new_old Class */
{
    Location = 0.0; AdditiveEffect = 0.0; DominanceEffect = 0.0; Type = "0"; GenOccured = 99; Freq = "0"; LDDecay = "";
}
QTL_new_old::QTL_new_old(double location, double add, double dom, std::string type, int genOccured, std::string freq, std::string lddec)
{
    Location = location; AdditiveEffect = add; DominanceEffect = dom; Type = type;
    GenOccured = genOccured; Freq = freq; LDDecay = lddec;
}
// destructor
QTL_new_old::~QTL_new_old(){}           /* QTL_new_old Class */
// Start of Functions for QTL class
void QTL_new_old::UpdateFreq(std::string currentFreq){Freq = Freq + "_" + currentFreq;}
void QTL_new_old::UpdateLDDecay(std::string currentLDDecaya){LDDecay = LDDecay + "_" + currentLDDecaya;}
void QTL_new_old::FounderLDDecay(std::string currentLDDecay){LDDecay = currentLDDecay;}


/************************************************************************/
/************************************************************************/
/*    hapLibrary: information regarding the haplotype across generations     */
/************************************************************************/
/************************************************************************/
// constructors
hapLibrary::hapLibrary()    /* ROH_Index */
{
    HapID = 0;  StartIndex = 0; EndIndex = 0; haplotypes = "";
}
hapLibrary::hapLibrary(int hapid, int start, int end, std::string haps)
{
    HapID = hapid;  StartIndex = start; EndIndex = end; haplotypes = haps;
}
// destructor
hapLibrary::~hapLibrary(){}             /* Haplotype Library */
// Start of Functions for ROH Class
void hapLibrary::UpdateHaplotypes(std::string temp){haplotypes = temp;}


/***************************************************************************************************************************************/
/***************************************************************************************************************************************/
/*******                    Window based QTL variance for additive and dominance                                                  ******/
/***************************************************************************************************************************************/
/***************************************************************************************************************************************/
void WindowVariance(parameters &SimParameters,vector <Animal> &population,vector < QTL_new_old > &population_QTL,string foundergen ,string Windowadditive_Output, string Windowdominance_Output)
{
    
    int currentgen;
    vector <int> QTLchr; vector <int> QTLposmb; vector <double> additive; vector <double> dominance; vector <double> freq;
    vector < vector < int > > qtlgenotypes;
    /* Fill QTL parameters */
    for(int i = 0; i < population_QTL.size(); i++)
    {
        int qtlchr = int(population_QTL[i].getLocation());
        double qtlpos = population_QTL[i].getLocation() - qtlchr;
        int qtlposmb = qtlpos * (SimParameters.get_ChrLength())[qtlchr-1];
        QTLchr.push_back(qtlchr); QTLposmb.push_back(qtlposmb); freq.push_back(0.0);
        additive.push_back(population_QTL[i].getAdditiveEffect());
        dominance.push_back(population_QTL[i].getDominanceEffect());
    }
    //cout << QTLchr.size() << endl;
    //for(int i = 0; i < QTLchr.size(); i++)
    //for(int i = 0; i < 25; i++)
    //{
    //    cout << QTLchr[i] << " " << QTLposmb[i] << " " << additive[i] << " " << dominance[i] << " " << freq[i] << endl;
    //}
    /* Fill Genotypes and frequency parameters */
    string tempgeno;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            currentgen = population[i].getGeneration();
            tempgeno = population[i].getQTL();
            vector < int > temp (QTLchr.size(),0);
            for(int j = 0; j < QTLchr.size(); j++)
            {
                int snp = tempgeno[j] - 48;
                if(snp == 3 || snp == 4){snp = 1;}
                temp[j] = snp; freq[j] += snp;
            }
            qtlgenotypes.push_back(temp);
        }
    }
    //cout << qtlgenotypes.size() << " " << qtlgenotypes[0].size() << endl;
    //cout << population.size() << endl;
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++)
    //    {
    //        cout << qtlgenotypes[i][j] << " ";
    //    }
    //    cout << endl;
    //}
    /* Calculate Frequency */
    for(int i = 0; i < freq.size(); i++){freq[i] /= double(2*qtlgenotypes.size());}
    //for(int i = 0; i < 120; i++)
    //{
    //    cout << QTLchr[i] << " " << QTLposmb[i] << " " << additive[i] << " " << dominance[i] << " " << freq[i] << endl;
    //}
    //for(int i = 0; i < population_QTL.size(); i++){cout << freq[i] << " " << population_QTL[i].getFreq() << endl;}
    int leftpos = 0;
    int rightpos = 1000000;
    int indexpos = 0; int currentchr = 1;
    /* start at beginning and move window forward; if reach a different chromosome restart left and right at 0 and 1000000 */
    string stopfull = "no";
    vector < int > outchr;
    vector < int > outposition;
    vector < double > Va;
    vector < double > Vd;
    int totalqtlfound = 0;
    while(stopfull == "no")
    {
        vector < int > qtlindex;
        while(1)
        {
            if(QTLposmb[indexpos] >= leftpos && QTLposmb[indexpos] < rightpos && QTLchr[indexpos] == currentchr)
            {
                qtlindex.push_back(indexpos);
            }
            if(QTLposmb[indexpos] >= leftpos && QTLposmb[indexpos] >= rightpos && QTLchr[indexpos] == currentchr){break;}
            if(QTLchr[indexpos] != currentchr){break;}
            indexpos++;
        }
        if(qtlindex.size() == 0)
        {
            outchr.push_back(currentchr); outposition.push_back(leftpos+500000); Va.push_back(0.0); Vd.push_back(0.0);
            leftpos += 1000000; rightpos += 1000000;
        }
        if(qtlindex.size() > 0)
        {
            totalqtlfound += qtlindex.size();
            outchr.push_back(currentchr); outposition.push_back(leftpos+500000); Va.push_back(0.0); Vd.push_back(0.0);
            //cout << " - " << qtlindex.size() << " - ";
            for(int i = 0; i < qtlindex.size(); i++)
            {
                //cout << QTLchr[qtlindex[i]] << " " << QTLposmb[qtlindex[i]] << "  -  ";
                //cout << freq[qtlindex[i]] << " " << additive[qtlindex[i]] << " " << dominance[qtlindex[i]] << endl;
                Va[outchr.size()-1] += (2*freq[qtlindex[i]]*(1-freq[qtlindex[i]])) * ((additive[qtlindex[i]]+(dominance[qtlindex[i]]*((1-freq[qtlindex[i]])-freq[qtlindex[i]]))) * (additive[qtlindex[i]]+(dominance[qtlindex[i]]*((1-freq[qtlindex[i]])-freq[qtlindex[i]]))));
                Vd[outchr.size()-1] += ((2*freq[qtlindex[i]]*(1-freq[qtlindex[i]])*dominance[qtlindex[i]])*(2*freq[qtlindex[i]]*(1-freq[qtlindex[i]])*dominance[qtlindex[i]]));
                //cout << Va[outchr.size()-1] << " " << Vd[outchr.size()-1] << "  -  ";
                //cout << endl; exit (EXIT_FAILURE);
            }
            leftpos += 1000000; rightpos += 1000000;
        }
        if(rightpos > (SimParameters.get_ChrLength())[currentchr-1])
        {
            leftpos = 0; rightpos = 1000000; currentchr++;
        }
        //cout << totalqtlfound << " " << outchr.size() << ": " << outchr[outchr.size()-1] << " " << outposition[outchr.size()-1] << " ";
        //cout << Va[outchr.size()-1] << " " << Vd[outchr.size()-1] << endl;
        //if(indexpos > 110){exit (EXIT_FAILURE);}
        if(currentchr > SimParameters.getChr()){stopfull = "yes";}
    }
    if(foundergen == "yes")
    {
        fstream checkVafile; checkVafile.open(Windowadditive_Output.c_str(), std::fstream::out | std::fstream::trunc); checkVafile.close();
        fstream checkVdfile; checkVdfile.open(Windowdominance_Output.c_str(), std::fstream::out | std::fstream::trunc); checkVdfile.close();
        /* output Va */
        std::ofstream outVa(Windowadditive_Output.c_str(), std::ios_base::app | std::ios_base::out);
        outVa << "Generation";
        for(int i = 0; i < outchr.size(); i++){outVa << " " << outchr[i] << "_" << outposition[i];}
        outVa << endl;
        outVa << currentgen;
        for(int i = 0; i < outchr.size(); i++){outVa << " " << Va[i];}
        outVa << endl;
        /* output Vd */
        std::ofstream outVd(Windowdominance_Output.c_str(), std::ios_base::app | std::ios_base::out);
        outVd << "Generation";
        for(int i = 0; i < outchr.size(); i++){outVd << " " << outchr[i] << "_" << outposition[i];}
        outVd << endl;
        outVd << currentgen;
        for(int i = 0; i < outchr.size(); i++){outVd << " " << Vd[i];}
        outVd << endl;
    }
    if(foundergen == "no")
    {
        /* output Va */
        std::ofstream outVa(Windowadditive_Output.c_str(), std::ios_base::app | std::ios_base::out);
        outVa << currentgen;
        for(int i = 0; i < outchr.size(); i++){outVa << " " << Va[i];}
        outVa << endl;
        /* output Vd */
        std::ofstream outVd(Windowdominance_Output.c_str(), std::ios_base::app | std::ios_base::out);
        outVd << currentgen;
        for(int i = 0; i < outchr.size(); i++){outVd << " " << Vd[i];}
        outVd << endl;
    }
    //for(int i = 0; i < outchr.size(); i++)
    //{
    //    cout << outchr[i] << " " << outposition[i] << " " << Va[i] << " " << Vd[i] << endl;
    //}
}


