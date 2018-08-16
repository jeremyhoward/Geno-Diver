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
#include <unordered_map>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "HaplofinderClasses.h"
#include "Animal.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"
#include "OutputFiles.h"
#include "Genome_ROH.h"
#include "Global_Population.h"

using namespace std;

/************************************************************************/
/************************************************************************/
/* Animal: class that holds information regarding a particular animal   */
/************************************************************************/
/************************************************************************/
// constructors
Animal::Animal()
{
    ID = 0; Sire = 0; SireAge = 0; Dam = 0; DamAge = 0; Sex = 0; Generation = 0; Age = 0; Progeny = 0; Matings = 0; NumDead = 0; RndSelection = 0.0;
    RndCulling = 0.0; RndPheno1 = 0.0; RndPheno2 = 0.0; RndGeno = 0.0; Ped_F = 0.0; Gen_F = 0.0; Hap1_F = 0.0; Hap2_F = 0.0; Hap3_F = 0.0;
    Unfav_Homozy_Leth = 0; Unfav_Heterzy_Leth = 0; Unfav_Homozy_Subleth = 0; Unfav_Heterzy_Subleth = 0; Lethal_Equivalents = 0; Homozy = 0.0; Fitness = 0.0;
    Marker = "0"; QTL = "0"; PaternalHapl = "0"; MaternalHapl = "0"; Pedigree_3_Gen = ""; propROH = 0; GenoStatus = "No"; Phenvect = vector<double>(0);
    EBVvect = vector<double>(0); Accvect = vector<double>(0); GVvect = vector<double>(0); BVvect = vector<double>(0); DDvect = vector<double>(0);
    Rvect = vector<double>(0); double ebvindex = 0; double tbvindex = 0; BVvectFalc = vector <double>(0); DDvectFalc = vector <double>(0);
    PhenStatus = vector<std::string>(0); AnimalStage = "selcand";
}
Animal::Animal(int id, int sire, int sireage, int dam, int damage, int sex, int generation, int age, int progeny, int matings, int dead, double rndsel, double rndcul, double rndphe1, double rndphe2, double rndgeno, double pedf, double genf, double hap1f, double hap2f, double hap3f, int unfavhomoleth, int unfavheterleth, int unfavhomosublet, int unfavhetersublet, double lethalequiv, double homozy,double fit, string mark, string qtl, string pathap, string mathap, string ped3g, double proproh, string genostatus, vector<double> vpheno, vector<double> vebv, vector<double> vacc, vector<double> vgv, vector<double> vbv, vector<double> vdd, vector<double> vr,double indexebv, double indextbv, vector<double> vbvfalc, vector<double> vddfalc, std::vector<std::string> phenstat, string animstg)
{
    ID = id; Sire = sire; SireAge = sireage; Dam = dam; DamAge = damage; Sex = sex; Generation = generation; Age = age; Progeny = progeny;
    Matings = matings; NumDead = dead; RndSelection = rndsel; RndCulling = rndcul; RndPheno1 = rndphe1; RndPheno2 = rndphe2; RndGeno = rndgeno;
    Ped_F = pedf; Gen_F = genf; Hap1_F = hap1f; Hap2_F = hap2f; Hap3_F = hap3f; Unfav_Homozy_Leth = unfavhomoleth; Unfav_Heterzy_Leth = unfavheterleth;
    Unfav_Homozy_Subleth = unfavhomosublet; Unfav_Heterzy_Subleth = unfavhetersublet; Lethal_Equivalents = lethalequiv; Homozy = homozy; Fitness = fit;
    Marker = mark; QTL = qtl; PaternalHapl = pathap; MaternalHapl = mathap; Pedigree_3_Gen = ped3g; propROH = proproh; GenoStatus = genostatus;
    Phenvect = vpheno; EBVvect = vebv; Accvect = vacc; GVvect = vgv; BVvect = vbv; DDvect = vdd; Rvect = vr; ebvindex = indexebv; tbvindex = indextbv;
    BVvectFalc = vbvfalc; DDvectFalc = vddfalc; PhenStatus = phenstat; AnimalStage = animstg;
}

// destructor
Animal::~Animal(){}                     /* Animal Class */
/* Functions to get or updated values within Animal Class */
void Animal::UpdateInb(double temp){Ped_F = temp;}
void Animal::UpdateGenInb(double temp){Gen_F = temp;}
void Animal::UpdateAge(){Age = Age + 1;}
void Animal::UpdateProgeny(){Progeny = Progeny + 1;}
void Animal::UpdateRndCulling(double temp){RndCulling = temp;}
void Animal::UpdateQTLGenotype(std::string temp){QTL = temp;}
void Animal::ZeroOutMatings(){Matings = 0;}
void Animal::UpdateMatings(int temp){Matings = Matings + temp;}
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
void Animal::update_Phenvect(int i, double x){Phenvect[i] = x;}
void Animal::update_EBVvect(int i, double x){EBVvect[i] = x;}
void Animal::update_Accvect(int i, double x){Accvect[i] = x;}
void Animal::update_GVvect(int i, double x){GVvect[i] = x;}
void Animal::update_BVvect(int i, double x){BVvect[i] = x;}
void Animal::update_DDvect(int i, double x){DDvect[i] = x;}
void Animal::update_Rvect(int i, double x){Rvect[i] = x;}
void Animal::Updatetbvindex(double temp){tbvindex = temp;}
void Animal::Updateebvindex(double temp){ebvindex = temp;}
void Animal::update_BVvectFalc(int i, double x){BVvectFalc[i] = x;}
void Animal::update_DDvectFalc(int i, double x){DDvectFalc[i] = x;}
void Animal::UpdateSireAge(int temp){SireAge = temp;}
void Animal::UpdateDamAge(int temp){DamAge = temp;}
void Animal::update_PhenStatus(int i, std::string x){PhenStatus[i] = x;}
void Animal::UpdateAnimalStage(std::string temp){AnimalStage = temp;}


/////////////////////////////////////////
// Start of Functions for Animal Class //
/////////////////////////////////////////
////////////////////////////////////////////////////////////////////
// Generate based breeding values by centering everything to zero //
////////////////////////////////////////////////////////////////////
void GenerateBaseValues(vector <Animal> &population,vector <double> &meangv, vector <double> &meanbv, vector <double> &meandd)
{
    //cout << meangv[0] << " " << meanbv[0] << " " << meandd[0] << endl;
    for(int i =0; i < population.size(); i++)
    {
        for(int j = 0; j < meangv.size(); j++)
        {
            population[i].update_Phenvect(j,((population[i].get_Phenvect())[j] - meangv[j]));
            population[i].update_GVvect(j,((population[i].get_GVvect())[j]-meangv[j]));
            population[i].update_BVvect(j,((population[i].get_BVvect())[j]-meanbv[j]));
            population[i].update_DDvect(j,((population[i].get_DDvect())[j]-meandd[j]));
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Update true breeding value and if using the statistical parameterization replace BVvect() and DDvect() //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
void UpdateTBV_TDD_Statistical(globalpopvar &Population1, vector <Animal> &population,parameters &SimParameters)
{
    /* First calculate alpha based on frequencies */
    vector< vector <double> > alpha((Population1.get_QTLFreq_AcrossGen()).size(), vector <double> (SimParameters.getnumbertraits(),0.0));
    for(int i = 0; i < (Population1.get_QTLFreq_AcrossGen()).size(); i++)
    {
        if((Population1.get_QTLFreq_AcrossGen())[i] != -5.0)
        {
            double tempfreq = (Population1.get_QTLFreq_AcrossGen())[i] / double(2*Population1.getQTLFreq_Number());
            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
            {
                alpha[i][j] = (Population1.get_qtl_add_quan(i,j)+(Population1.get_qtl_dom_quan(i,j) * ((1 - tempfreq) - (tempfreq))));
            }
        }
    }
    /* Now generate tbv and tdd */
    /* Genotype 0: additive =  2q(alpha); dominance = 2q^2d */
    /* Genotype 1: additive = (q-p)alpha; dominance = 2pqd  */
    /* Genotype 2: additive = -2p(alpha); dominance = 2p^2d */
    for(int i = 0; i < population.size(); i++)
    {
        string geno = population[i].getQTL();
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            double temptbv = 0.0; double temptdd = 0.0;
            for(int k = 0; k < geno.size(); k++)
            {
                if((Population1.get_qtl_type())[k] == 2 || (Population1.get_qtl_type())[k] == 3)
                {
                    int temp = geno[k] - 48;
                    if(temp > 2){temp = 1;}
                    double tempfreq = (Population1.get_QTLFreq_AcrossGen())[k] / double(2*Population1.getQTLFreq_Number());
                    if(temp == 0){
                        temptbv += -2 * (tempfreq) * alpha[k][j];
                        temptdd += -2 * tempfreq * tempfreq * Population1.get_qtl_dom_quan(k,j);
                    } else if(temp == 1){
                        temptbv += ((1 - tempfreq) - (tempfreq)) * alpha[k][j];
                        temptdd += 2 * (1-tempfreq) * tempfreq * Population1.get_qtl_dom_quan(k,j);
                    } else if(temp == 2){
                        temptbv += (2 * (1-tempfreq) * alpha[k][j]);
                        temptdd += -2 * (1-tempfreq) * (1-tempfreq) * Population1.get_qtl_dom_quan(k,j);
                    } else {cout << endl << "Shouldn't Be Here" << endl; exit (EXIT_FAILURE);}
                }
            }
            //cout << temptbv << " " << temptdd << " -- " << (population[i].get_BVvect())[0] << " " << (population[i].get_DDvect())[0] << endl;
            population[i].update_BVvectFalc(j,temptbv); population[i].update_DDvectFalc(j,temptdd);
        }
    }
}
///////////////////////////////////
// Generate breeding value index //
///////////////////////////////////
void GenerateBVIndex(vector <Animal> &population,globalpopvar &Population1,parameters &SimParameters,int Gen, ostream& logfileloc)
{
    /**************************************************/
    /** First Generation calculate sd in ebv and tbv **/
    /**************************************************/
    if(Gen == 1)
    {
        vector < double > mean_ebv(2,0.0); vector < double > mean_bv(2,0.0);
        vector < double > sd_ebv(2,0.0); vector < double > sd_bv(2,0.0);
        for(int i = 0; i < population.size(); i++)
        {
            //cout << population[i].getID() << " " << population[i].getGeneration() << " ";
            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
            {
                mean_ebv[j] += (population[i].get_EBVvect())[j];
                mean_bv[j] += (population[i].get_BVvect())[j];
                //cout << (population[i].get_BVvect())[j] << " " << (population[i].get_EBVvect())[j] << " ";
            }
            //cout << endl;
        }
        for(int j = 0; j < SimParameters.getnumbertraits(); j++){mean_ebv[j] /= population.size(); mean_bv[j] /= population.size();}
        for(int i = 0; i < population.size(); i++)
        {
            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
            {
                sd_ebv[j] += ((population[i].get_EBVvect())[j] - mean_ebv[j])*((population[i].get_EBVvect())[j] - mean_ebv[j]);
                sd_bv[j] += ((population[i].get_BVvect())[j]-mean_bv[j])*((population[i].get_BVvect())[j]-mean_bv[j]);
            }
        }
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            sd_ebv[j] /= (population.size()-1); sd_bv[j] /= (population.size()-1);
            sd_ebv[j] = sqrt(sd_ebv[j]); sd_bv[j] = sqrt(sd_bv[j]);
            if(sd_ebv[j] < 0.00001){sd_ebv[j] = 0;}
        }
        /* Update SD in EBV and BV across both traits in founder population */
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            Population1.update_sdGen0_EBV(j,sd_ebv[j]); Population1.update_sdGen0_TBV(j,sd_bv[j]);
        }
        //for(int j = 0; j < SimParameters.getnumbertraits(); j++){cout << mean_ebv[j] << " " << sd_ebv[j] << endl;}
        //for(int j = 0; j < SimParameters.getnumbertraits(); j++){cout << mean_bv[j] << " " << sd_bv[j] << endl;}
        //for(int i = 0; i < SimParameters.getnumbertraits(); i++){cout<<(Population1.get_sdGen0_EBV())[i]<<" "<<(Population1.get_sdGen0_TBV())[i]<<endl;}
    }
    /*****************************************/
    /** Now Generate index based ebv and bv **/
    /*****************************************/
    for(int i =0; i < population.size(); i++)
    {
        population[i].Updatetbvindex((((population[i].get_BVvect())[0] / double((Population1.get_sdGen0_TBV())[0]))*(SimParameters.get_IndexWeights())[0]) + (((population[i].get_BVvect())[1]/ double((Population1.get_sdGen0_TBV())[1]))*(SimParameters.get_IndexWeights())[1]));
        if((Population1.get_sdGen0_EBV())[0] > 0 && (Population1.get_sdGen0_EBV())[1] > 0)
        {
            population[i].Updateebvindex((((population[i].get_EBVvect())[0] / double((Population1.get_sdGen0_EBV())[0]))*(SimParameters.get_IndexWeights())[0]) + (((population[i].get_EBVvect())[1]/ double((Population1.get_sdGen0_EBV())[1]))*(SimParameters.get_IndexWeights())[1]));
        } else{population[i].Updateebvindex(0.0);}
    }
    //for(int i = 0; i < population.size(); i++)
    //for(int i = 0; i < 15; i++)
    //{
        //cout << population[i].getID() << " " << population[i].getGeneration() << " -- ";
        //for(int j = 0; j < SimParameters.getnumbertraits(); j++){cout << (population[i].get_BVvect())[j] << " ";}
        //cout << population[i].gettbvindex() << " -- ";
        //for(int j = 0; j < SimParameters.getnumbertraits(); j++){cout << (population[i].get_EBVvect())[j] << " ";}
       //cout << population[i].getebvindex() << endl;
    //}
}
////////////////////////////////////////
// Update selection candidate with PA //
////////////////////////////////////////
void Update_selcand_PA(vector <Animal> &population)
{
    for(int i = 0; i < (population[0].get_EBVvect()).size(); i++)
    {
        unordered_map <int, double> EBVlinker;
        for(int j = 0; j < population.size(); j++)
        {
            if(population[j].getAnimalStage() == "parent")
            {
                if((population[j].get_EBVvect())[i] != 0.0){EBVlinker.insert({population[j].getID(),(population[j].get_EBVvect())[i]});}
            }
        }
        /* Only update selection candidates if parens have ebv estimated */
        if(EBVlinker.size() > 0)
        {
            /* Loop through and updated selection candidates with PA */
            for(int j = 0; j < population.size(); j++)
            {
                if(population[j].getAnimalStage() == "selcand")
                {
                    //cout << population[j].getID() << " " << population[j].getSire() << " " << population[j].getDam() << endl;
                    //cout << EBVlinker[population[j].getSire()] << " " << EBVlinker[population[j].getDam()] << endl;
                    //cout << (0.5 * (EBVlinker[population[j].getSire()] + EBVlinker[population[j].getDam()])) << endl;
                    population[j].update_EBVvect(i, (0.5 * (EBVlinker[population[j].getSire()] + EBVlinker[population[j].getDam()])));
                    //cout << (population[j].get_EBVvect())[i] << endl;
                }
            }
        }
    }
}


///////////////////////////////////////
// Update GenoPhenoStatus and Output //
///////////////////////////////////////
void UpdateGenoPhenoStatus(globalpopvar &Population1, vector <Animal> &population,parameters &SimParameters,outputfiles &OUTPUTFILES, int Gen, string stage, vector < string > SelectionVector = vector<string>(0))
{
    /**********************************************************************************************************************************************************/
    /**********************************************************************************************************************************************************/
    /* Initially output geno-pheno status for progeny as soon as all progeny are generated. Only scenario where all animals are phenotyped prior to selection */
    /* is 'pheno_atselection' otherwise set them all as 'No" phenotype. Then depending on the scenario update the phenotype status within each generation for */
    /* a given animal                                                                                                                                         */
    /**********************************************************************************************************************************************************/
    /**********************************************************************************************************************************************************/
    if(stage == "outputselectioncand")
    {
        /* First remove contents from old run then keep appending progeny to file */
        if(Gen == 0){fstream cleargenostatus; cleargenostatus.open(OUTPUTFILES.getloc_GenotypeStatus().c_str(), std::fstream::out | std::fstream::trunc);}
        int ebvtraits;
        if(SimParameters.getSelection() == "ebv" || SimParameters.getSelection() == "tbv" || SimParameters.getSelection() == "phenotype"){ebvtraits = 1;}
        if(SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "index_tbv"){ebvtraits = 2;}
        /**************************************************************************************/
        /* First update genotype status across selection candidates based on default settings */
        /**************************************************************************************/
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getAnimalStage()=="selcand")
            {
                if(SimParameters.getSelection() == "random" || SimParameters.getSelection() == "phenotype" || SimParameters.getSelection() == "tbv"){
                    population[i].UpdateGenoStatus("No");
                } else {
                    /* pblup implies no animals are genotyped */
                    if(SimParameters.getEBV_Calc() == "pblup"){population[i].UpdateGenoStatus("No");}
                    /* gblup, rohblup or bayes implies all animals are genotyped with (i.e. "Full") */
                    if(SimParameters.getEBV_Calc() == "gblup" || SimParameters.getEBV_Calc() == "rohblup" || SimParameters.getEBV_Calc() == "bayes")
                    {
                        population[i].UpdateGenoStatus("Yes");
                    }
                    /* ssgblup implies portion of animals genotyped. Therefore some will have "No" and others will have either "Full" or "Reduced" */
                    if(SimParameters.getEBV_Calc() == "ssgblup") /* initialize to not genotyped; can change depending on the genotype strategy */
                    {
                        population[i].UpdateGenoStatus("No");
                    }
                }
            }
        }
        /**************************************************************************************************/
        /* Then update phenotype status across selection candidates depends only on *_atselection options */
        /**************************************************************************************************/
        for(int trt = 0; trt < ebvtraits; trt++)
        {
            /**************************************************************/
            /* First make a key to figure out whether to phenotype or not */
            /************************************************************************************/
            unordered_map <int, string> PhenoStatuslinker; vector <double> randev_M; vector <double> randev_F;
            /*******************************************************************************************************************************/
            /* if doing random_atselection phenotyping first determine whether an animal is phenotyped prior to looping through population */
            /*******************************************************************************************************************************/
            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_atselection")
            {
                for(int i = 0; i < population.size();  i++)
                {
                    if(population[i].getSex() == 0)
                    {
                        if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_atselection" && population[i].getAnimalStage()=="selcand")
                        {
                            if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                            if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                        }
                    }
                    if(population[i].getSex() == 1)
                    {
                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_atselection" && population[i].getAnimalStage()=="selcand")
                        {
                            if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                            if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                        }
                    }
                }
                //cout << "Output Selection Cand(random): " << randev_F.size() << " " << randev_M.size() << endl;
                if(randev_F.size() > 0 || randev_M.size() > 0)
                {
                    double malegenotypecutoff, femalegenotypecutoff, temp;
                    /* order and take a certain percentage */
                    if(randev_M.size() > 0)
                    {
                        for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                        {
                            for(int j=i+1; j< randev_M.size(); j++)
                            {
                                if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                            }
                        }
                        malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                    }
                    if(randev_F.size() > 0)
                    {
                        for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                        {
                            for(int j=i+1; j< randev_F.size(); j++)
                            {
                                if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                            }
                        }
                        femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                    }
                    //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                    //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                    //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                    //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                    //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                    //int numfemale = 0; int nummale = 0;
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(randev_M.size() > 0)
                        {
                            if(population[i].getSex() == 0)
                            {
                                if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_atselection" && population[i].getAnimalStage()=="selcand")
                                {
                                    if(trt == 0)
                                    {
                                        if(population[i].getRndPheno1() <= malegenotypecutoff){
                                            PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                        } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                    }
                                    if(trt == 1)
                                    {
                                        if(population[i].getRndPheno2() <= malegenotypecutoff){
                                            PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                        } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                    }
                                }
                            }
                        }
                        if(population[i].getSex() == 1)
                        {
                            if(randev_F.size() > 0)
                            {
                                if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_atselection" && population[i].getAnimalStage()=="selcand")
                                {
                                    if(trt == 0)
                                    {
                                        if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                            PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                        } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                    }
                                    if(trt == 1)
                                    {
                                        if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                            PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                        } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                    }
                                }
                            }
                        }
                    }
                    //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                }
            }
            /*******************************************************************************************************************************/
            /* if doing ebv_atselection phenotyping first determine whether an animal is phenotyped prior to looping through population    */
            /*******************************************************************************************************************************/
            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_atselection")
            {
                string whereat;
                if(SelectionVector.size() == 0){
                    whereat = "random";
                } else{whereat = SelectionVector[Gen-1];}
                if(whereat == "random"){
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getSex() == 0)
                        {
                            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                            {
                                if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                            }
                        }
                        if(population[i].getSex() == 1)
                        {
                            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                            {
                                if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                            }
                        }
                    }
                    //cout << "Output Selection Cand (ebv_random): " << randev_F.size() << " " << randev_M.size() << endl;
                    if(randev_F.size() > 0 || randev_M.size() > 0)
                    {
                        double malegenotypecutoff, femalegenotypecutoff, temp;
                        /* order and take a certain percentage */
                        if(randev_M.size() > 0)
                        {
                            for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                            {
                                for(int j=i+1; j< randev_M.size(); j++)
                                {
                                    if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                }
                            }
                            malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                        }
                        if(randev_F.size() > 0)
                        {
                            for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                            {
                                for(int j=i+1; j< randev_F.size(); j++)
                                {
                                    if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                }
                            }
                            femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                        }
                        //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                        //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                        //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                        //cout << endl;
                        //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                        //cout << endl;
                        //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                        //int numfemale = 0; int nummale = 0;
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(randev_M.size() > 0)
                            {
                                if(population[i].getSex() == 0)
                                {
                                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                            if(population[i].getSex() == 1)
                            {
                                if(randev_F.size() > 0)
                                {
                                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                        }
                        //cout << numfemale << " " << nummale << endl;
                        //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                    }
                } else if(whereat == "ebv" || whereat == "index_ebv"){
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getSex() == 0)
                        {
                            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                            {
                                randev_M.push_back((population[i].get_EBVvect())[trt]);
                            }
                        }
                        if(population[i].getSex() == 1)
                        {
                            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                            {
                                randev_F.push_back((population[i].get_EBVvect())[trt]);
                            }
                        }
                    }
                    //cout << "Output Selection Cand (ebv_ebv): " << randev_F.size() << " " << randev_M.size() << endl;
                    //cout << SimParameters.get_PortionofDistribution_vec()[trt] << endl;
                    if(randev_F.size() > 0 || randev_M.size() > 0)
                    {
                        double malegenotypecutoff, femalegenotypecutoff, temp;
                        /* cutoffs for tailed */
                        double malelowercutoff, maleuppercutoff, femalelowercutoff, femaleuppercutoff;
                        /* order and take a certain percentage */
                        if(randev_M.size() > 0)
                        {
                            /* sort lowest to highest */
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                            {
                                for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                                {
                                    for(int j=i+1; j< randev_M.size(); j++)
                                    {
                                        if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                    }
                                }
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")   /* Sort highest to lowest */
                            {
                                for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                                {
                                    for(int j=i+1; j< randev_M.size(); j++)
                                    {
                                        if(randev_M[i] < randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                    }
                                }
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                            {
                                malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                            {
                                malelowercutoff = randev_M[int((((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5)*randev_M.size()+0.5))-1];
                                maleuppercutoff = randev_M[int(((1-((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5))*randev_M.size()))-1];
                            }
                        }
                        if(randev_F.size() > 0)
                        {
                            /* sort lowest to highest */
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                            {
                                for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                                {
                                    for(int j=i+1; j< randev_F.size(); j++)
                                    {
                                        if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                    }
                                }
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")   /* Sort highest to lowest */
                            {
                                for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                                {
                                    for(int j=i+1; j< randev_F.size(); j++)
                                    {
                                        if(randev_F[i] < randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                    }
                                }
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                            {
                                femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                            {
                                femalelowercutoff = randev_F[int((((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5)*randev_F.size()+0.5))-1];
                                femaleuppercutoff = randev_F[int(((1-((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5))*randev_F.size()))-1];
                            }
                        }
                        //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                        //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                        //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                        //cout << endl << endl;
                        //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                        //cout << endl << endl;
                        //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                        int numfemale = 0; int nummale = 0;
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(randev_M.size() > 0)
                            {
                                if(population[i].getSex() == 0)
                                {
                                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                                    {
                                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                        {
                                            if((population[i].get_EBVvect())[trt] >= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "low")
                                        {
                                            if((population[i].get_EBVvect())[trt] <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                        {
                                            if((population[i].get_EBVvect())[trt]<=malelowercutoff || (population[i].get_EBVvect())[trt]>=maleuppercutoff)
                                            {
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                            if(randev_F.size() > 0)
                            {
                                if(population[i].getSex() == 1)
                                {
                                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_atselection" && population[i].getAnimalStage()=="selcand")
                                    {
                                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                        {
                                            if((population[i].get_EBVvect())[trt] >= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "low")
                                        {
                                            if((population[i].get_EBVvect())[trt] <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                        {
                                            if((population[i].get_EBVvect())[trt]<=femalelowercutoff || (population[i].get_EBVvect())[trt]>=femaleuppercutoff)
                                            {
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                        }
                        //cout << nummale << " " << numfemale << endl;
                        //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                        {
                            if(numfemale > int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if((population[i].get_EBVvect())[trt] == femalegenotypecutoff && population[i].getSex() == 1)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                }
                                /* Change to no phenotyped */
                                for(int i = 0; i < (numfemale-int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)); i++)
                                {
                                    PhenoStatuslinker[ID[i]] = "No";
                                }
                            }
                            if(nummale > int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if((population[i].get_EBVvect())[trt] == malegenotypecutoff && population[i].getSex() == 0)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                }
                                /* Change to no phenotyped */
                                for(int i = 0; i < (nummale-int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)); i++)
                                {
                                    PhenoStatuslinker[ID[i]] = "No";
                                }
                            }
                        }
                        if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                        {
                            if(numfemale > int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if((population[i].get_EBVvect())[trt] == femalelowercutoff || (population[i].get_EBVvect())[trt] == femaleuppercutoff && population[i].getSex() == 1)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                }
                                /* Change to no phenotyped */
                                for(int i = 0; i < (numfemale-int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)); i++)
                                {
                                    PhenoStatuslinker[ID[i]] = "No";
                                }
                            }
                            if(nummale > int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if((population[i].get_EBVvect())[trt] == malelowercutoff || (population[i].get_EBVvect())[trt] == maleuppercutoff && population[i].getSex() == 0)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                }
                                /* Change to no phenotyped */
                                for(int i = 0; i < (nummale-int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)); i++)
                                {
                                    PhenoStatuslinker[ID[i]] = "No";
                                }
                            }
                        }
                        //cout << nummale << " " << numfemale << endl;
                    }
                } else {cout << endl << "Shouldn't be Here (Line 613)" << endl; exit (EXIT_FAILURE);}
            }
            /****************************************************************************************************************************************/
            /* if doing litterrandom_atselection phenotyping first determine whether an animal is phenotyped prior to looping through population    */
            /****************************************************************************************************************************************/
            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_atselection")
            {
                /* First get index for Dams and all associated offspring within the pedigree class */
                unordered_map <int, int> Damlinker; unordered_map <int, int> Damlinkerindex; int numdams = 0;  unordered_map <int, int> Offspringlinker;
                std:unordered_map<int,int>::const_iterator got;  /* Search if key exists */
                for(int i = 0; i < population.size();  i++)
                {
                    if(population[i].getAnimalStage()=="selcand")   /* Grab all selection candidates */
                    {
                        /* Grab dams and save in hash-table if not already there */
                        if(population[i].getDam() != 0)
                        {
                            got = Damlinker.find(population[i].getDam());
                            if(got == Damlinker.end())     /* If not found then add */
                            {
                                Damlinker.insert({population[i].getDam(),i});
                                Damlinkerindex.insert({population[i].getDam(),numdams}); numdams++;
                            }
                        }
                        if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && population[i].getSex() == 0)
                        {
                            Offspringlinker.insert({population[i].getID(),i});
                        }
                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && population[i].getSex() == 1)
                        {
                            Offspringlinker.insert({population[i].getID(),i});
                        }
                    }
                }
                /* random phenotype selection this happens in the founders and then defaults to within a litter after that */
                if(Damlinker.size() == 0)
                {
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getSex() == 0)
                        {
                            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && population[i].getAnimalStage()=="selcand")
                            {
                                if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                            }
                        }
                        if(population[i].getSex() == 1)
                        {
                            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && population[i].getAnimalStage()=="selcand")
                            {
                                if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                            }
                        }
                    }
                    //cout << "Output Selection Cand (random_random): " << randev_F.size() << " " << randev_M.size() << endl;
                    if(randev_F.size() > 0 || randev_M.size() > 0)
                    {
                        double malegenotypecutoff, femalegenotypecutoff, temp;
                        /* order and take a certain percentage */
                        if(randev_M.size() > 0)
                        {
                            for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                            {
                                for(int j=i+1; j< randev_M.size(); j++)
                                {
                                    if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                }
                            }
                            malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                        }
                        if(randev_F.size() > 0)
                        {
                            for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                            {
                                for(int j=i+1; j< randev_F.size(); j++)
                                {
                                    if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                }
                            }
                            femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                        }
                        //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                        //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                        //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                        //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                        //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                        //int numfemale = 0; int nummale = 0;
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(randev_M.size() > 0)
                            {
                                if(population[i].getSex() == 0)
                                {
                                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && population[i].getAnimalStage()=="selcand")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                            if(population[i].getSex() == 1)
                            {
                                if(randev_F.size() > 0)
                                {
                                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && population[i].getAnimalStage()=="selcand")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                        }
                        //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                    }
                }
                if(Damlinker.size() > 0)
                {
                    /* Intialize a male and female 2d vector to store ids for all offspring of a given dam */
                    vector< vector < int >> Maleoffspring(Damlinker.size(),vector < int > (0));
                    vector< vector < int >> Femaleoffspring(Damlinker.size(),vector < int > (0));
                    for(int i = 0; i < population.size(); i++)
                    {
                        if(population[i].getAnimalStage()=="selcand")
                        {
                            if(population[i].getSex() == 0){Maleoffspring[Damlinkerindex[population[i].getDam()]].push_back(population[i].getID());}
                            if(population[i].getSex() == 1){Femaleoffspring[Damlinkerindex[population[i].getDam()]].push_back(population[i].getID());}
                        }
                    }
                    //for(int i = 0; i < Maleoffspring.size(); i++){cout<<Maleoffspring[i].size()<<" "<<Femaleoffspring[i].size()<<endl;}
                    int malenumber = int((SimParameters.get_MalePropPhenotype_vec())[trt]*(SimParameters.getOffspring()*0.5)+0.5);
                    int femalenumber = int((SimParameters.get_FemalePropPhenotype_vec())[trt]*(SimParameters.getOffspring()*0.5)+0.5);
                    int phenomale = 0; int phenofemale = 0;
                    /******************/
                    /* Do Males First */
                    /******************/
                    for(int i = 0; i < Damlinker.size(); i++)
                    {
                        if(malenumber < Maleoffspring[i].size())    /* if number of males larger than number genotyped than randomly select */
                        {
                            vector < double > randomvalues;
                            for(int j = 0; j < Maleoffspring[i].size(); j++)
                            {
                                if(trt == 0){randomvalues.push_back(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno1());}
                                if(trt == 1){randomvalues.push_back(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno2());}
                            }
                            double temp; double cutoff;
                            for(int j = 0; j < (randomvalues.size()-1); j++)   /* Sort */
                            {
                                for(int k=j+1; k < randomvalues.size(); k++)
                                {
                                    if(randomvalues[j] > randomvalues[k])
                                    {
                                        temp = randomvalues[j];
                                        randomvalues[j] = randomvalues[k];
                                        randomvalues[k] = temp;
                                    }
                                }
                            }
                            cutoff = randomvalues[malenumber-1];
                            for(int j = 0; j < Maleoffspring[i].size(); j++)
                            {
                                if(trt == 0)
                                {
                                    if(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno1() <= cutoff){
                                        PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); phenomale++;
                                    } else {PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"No"});}
                                }
                                if(trt == 1)
                                {
                                    if(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno2() <= cutoff){
                                        PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); phenomale++;
                                    } else {PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"No"});}
                                }
                            }
                        } else {
                            for(int j = 0; j < Maleoffspring[i].size(); j++)
                            {
                                PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); phenomale++;
                            }
                        }
                    }
                    /*******************/
                    /* Then Do Females */
                    /*******************/
                    for(int i = 0; i < Damlinker.size(); i++)
                    {
                        if(femalenumber < Femaleoffspring[i].size())    /* if number of males larger than number genotyped than randomly select */
                        {
                            vector < double > randomvalues;
                            for(int j = 0; j < Femaleoffspring[i].size(); j++)
                            {
                                if(trt == 0){randomvalues.push_back(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno1());}
                                if(trt == 1){randomvalues.push_back(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno2());}
                            }
                            double temp; double cutoff;
                            for(int j = 0; j < (randomvalues.size()-1); j++)   /* Sort */
                            {
                                for(int k=j+1; k < randomvalues.size(); k++)
                                {
                                    if(randomvalues[j] > randomvalues[k]){temp = randomvalues[j]; randomvalues[j] = randomvalues[k]; randomvalues[k] = temp;}
                                }
                            }
                            cutoff = randomvalues[malenumber-1];
                            for(int j = 0; j < Femaleoffspring[i].size(); j++)
                            {
                                if(trt == 0)
                                {
                                    if(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno1() <= cutoff){
                                        PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); phenofemale++;
                                    } else {PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"No"});}
                                }
                                if(trt == 1)
                                {
                                    if(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno2() <= cutoff){
                                        PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); phenofemale++;
                                    } else {PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"No"});}
                                }
                            }
                        } else {
                            for(int j = 0; j < Femaleoffspring[i].size(); j++)
                            {
                                PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); phenofemale++;
                            }
                        }
                    }
                    //cout << "Output Selection Cand (random_litter): " << phenomale << " " << phenofemale << endl;
                }
            }
            /********************************************************/
            /* Now Update phenotype status for selection candidates */
            /********************************************************/
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getSex() == 0 && population[i].getAnimalStage()=="selcand")
                {
                    /* If proportion == 1.0 and using pheno_atselection update when animal is at stage 'selcand' */
                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="pheno_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trt]==1.0){
                        population[i].update_PhenStatus(trt,"Yes");
                    /* If using 'random_atselection' and is a 'Yes' update when animal is at stage 'selcand' */
                    } else if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_atselection" && PhenoStatuslinker[population[i].getID()] == "Yes"){
                        population[i].update_PhenStatus(trt,"Yes");
                    /* If using 'ebv_atselection' and is a 'Yes' update when animal is at stage 'selcand' */
                    } else if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_atselection" && PhenoStatuslinker[population[i].getID()] == "Yes"){
                        population[i].update_PhenStatus(trt,"Yes");
                    /* If using 'litterrandom_atselection' and is a 'Yes' update when animal is at stage 'selcand' */
                    } else if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && PhenoStatuslinker[population[i].getID()]=="Yes"){
                        population[i].update_PhenStatus(trt,"Yes");
                    } else {population[i].update_PhenStatus(trt,"No");}
                }
                if(population[i].getSex() == 1 && population[i].getAnimalStage()=="selcand")
                {
                    /* If proportion == 1.0 and using pheno_atselection update when animal is at stage 'selcand' */
                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="pheno_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trt]==1.0){
                        population[i].update_PhenStatus(trt,"Yes");
                    /* If using 'random_atselection' and is a 'Yes' update when animal is at stage 'selcand' */
                    } else if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_atselection" && PhenoStatuslinker[population[i].getID()] == "Yes"){
                        population[i].update_PhenStatus(trt,"Yes");
                    /* If using 'ebv_atselection' and is a 'Yes' update when animal is at stage 'selcand' */
                    } else if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_atselection" && PhenoStatuslinker[population[i].getID()] == "Yes"){
                        population[i].update_PhenStatus(trt,"Yes");
                    /* If using 'litterrandom_atselection' and is a 'Yes' update when animal is at stage 'selcand' */
                    } else if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" && PhenoStatuslinker[population[i].getID()]=="Yes"){
                        population[i].update_PhenStatus(trt,"Yes");
                    } else {population[i].update_PhenStatus(trt,"No");}

                }
            }
        }
        /*********************************************************************************************/
        /* Put newly created progeny in GenoStatus file; Save as a continuous string and then output */
        /*********************************************************************************************/
        stringstream outputstringgenostatusfound(stringstream::out);
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getAge() == 1)
            {
                outputstringgenostatusfound << population[i].getID() << " ";
                if(population[i].getGenoStatus() == "Yes"){
                    outputstringgenostatusfound << population[i].getGenoStatus() << " parents ";
                } else {outputstringgenostatusfound << population[i].getGenoStatus() << " -";}
                for(int j = 0; j < SimParameters.getnumbertraits(); j++){outputstringgenostatusfound<<" "<<(population[i].get_PhenStatus())[j];}
                outputstringgenostatusfound << " popselcandidate " << population[i].getGeneration() << " ";
                outputstringgenostatusfound << population[i].getAge() << " " << population[i].getProgeny() << " " << population[i].getSex() << endl;
            }
        }
        /* output genostatus file */
        std::ofstream outputfounder(OUTPUTFILES.getloc_GenotypeStatus().c_str(), std::ios_base::app | std::ios_base::out);
        outputfounder << outputstringgenostatusfound.str(); outputstringgenostatusfound.str(""); outputstringgenostatusfound.clear();
        //for(int i = 0; i < population.size(); i++)
        //{
        //    cout << population[i].getID() << " '" << population[i].getGenoStatus() << "' '";
        //    for(int j = 0; j < SimParameters.getnumbertraits(); j++){cout << (population[i].get_PhenStatus())[j] << "' ";}
        //    cout << endl;
        //}
        if(Gen > 0)
        {
            string madechanges = "NO";
            vector < string > GenoPhenoStatus; int tempid;
            string line;
            ifstream infile;
            infile.open(OUTPUTFILES.getloc_GenotypeStatus().c_str());
            if(infile.fail()){cout << "GenotypeStatus!\n"; exit (EXIT_FAILURE);}
            while (getline(infile,line)){GenoPhenoStatus.push_back(line);}
            /* Now loop through and update parents age and number of progeny */
            unordered_map <int, int> Parentslinker; std::unordered_map<int, int>::const_iterator got;  /* Search if key exists */
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getAnimalStage() == "parent"){Parentslinker.insert({population[i].getID(),i});}
            }
            if(Parentslinker.size() > 0)
            {
                for(int i = 0; i < GenoPhenoStatus.size(); i++)
                {
                    size_t pos = GenoPhenoStatus[i].find(" popparent",0);
                    if(pos != std::string::npos)    /* Only do something if found */
                    {
                        //cout << "'" << GenoPhenoStatus[i] << "'" << "\t";
                        string line = GenoPhenoStatus[i];
                        vector < string > solvervariables(15,"");
                        for(int i = 0; i < 15; i++)
                        {
                            size_t pos = line.find(" ",0);
                            solvervariables[i] = line.substr(0,pos);
                            if(pos != std::string::npos){line.erase(0, pos + 1);}
                            if(pos == std::string::npos){line.clear(); i = 15;}
                        }
                        int start = 0;
                        while(start < solvervariables.size())
                        {
                            if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                            if(solvervariables[start] != ""){start++;}
                        }
                        //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
                        //cout << "  -  " << population[i].getAge() << " "<< population[i].getProgeny() << "  -  ";
                        if(solvervariables.size() == 9){
                            /* See if it exists in Parentslinker; */
                            got = Parentslinker.find(atoi(solvervariables[0].c_str()));
                            if(got == Parentslinker.end()){
                                cout << "Shouldn't be here Line 1135" << endl; exit (EXIT_FAILURE);
                            } else {        /* If found update age and progeny */
                                stringstream s1; s1 << population[Parentslinker[atoi(solvervariables[0].c_str())]].getAge();
                                stringstream s2; s2 << population[Parentslinker[atoi(solvervariables[0].c_str())]].getProgeny();
                                solvervariables[6] = s1.str(); solvervariables[7] = s2.str();
                            }
                        } else if(solvervariables.size() == 10){
                            /* See if it exists in Parentslinker; */
                            got = Parentslinker.find(atoi(solvervariables[0].c_str()));
                            if(got == Parentslinker.end()){
                                cout << "Shouldn't be here Line 1145" << endl; exit (EXIT_FAILURE);
                            } else {        /* If found update age and progeny */
                                stringstream s1; s1 << population[Parentslinker[atoi(solvervariables[0].c_str())]].getAge();
                                stringstream s2; s2 << population[Parentslinker[atoi(solvervariables[0].c_str())]].getProgeny();
                                solvervariables[7] = s1.str(); solvervariables[8] = s2.str();
                            }
                        } else {cout << "Shouldn't be here Line 1151" << endl; exit (EXIT_FAILURE);}
                        /* replace with new line */
                        std::ostringstream updatedline; updatedline << solvervariables[0];
                        for(int i = 1; i < solvervariables.size(); i++){updatedline << " " << solvervariables[i];}
                        GenoPhenoStatus[i] = updatedline.str();
                        //cout << "'" << GenoPhenoStatus[i] << "'" << endl;
                    }
                }
                /* First remove contents from old one then update with new genotype or phenotype status */
                fstream cleargenostatus; cleargenostatus.open(OUTPUTFILES.getloc_GenotypeStatus().c_str(), std::fstream::out | std::fstream::trunc);
                /* output updated pheno_geno status */
                stringstream outputstringgenostatus(stringstream::out);
                for(int i = 0; i < GenoPhenoStatus.size(); i++){outputstringgenostatus << GenoPhenoStatus[i] << endl;}
                std::ofstream outputstatus(OUTPUTFILES.getloc_GenotypeStatus().c_str(), std::ios_base::app | std::ios_base::out);
                outputstatus << outputstringgenostatus.str(); outputstringgenostatus.str(""); outputstringgenostatus.clear();
            }
        }
    }
    if(stage != "outputselectioncand")
    {
        int ebvtraits;
        if(SimParameters.getSelection() == "ebv" || SimParameters.getSelection() == "tbv"){ebvtraits = 1;}
        if(SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "index_tbv"){ebvtraits = 2;}
        /**************************************************************************************/
        /* First read in pheno_geno status and if it changes update otherwise just keep as is */
        /**************************************************************************************/
        string madechanges = "NO";
        vector < string > GenoPhenoStatus; int tempid;
        string line;
        ifstream infile;
        infile.open(OUTPUTFILES.getloc_GenotypeStatus().c_str());
        if(infile.fail()){cout << "GenotypeStatus!\n"; exit (EXIT_FAILURE);}
        while (getline(infile,line)){GenoPhenoStatus.push_back(line);}
        /*****************************************************************************************************************************************/
        /*****************************************************************************************************************************************/
        /******************                            UPDATE GENOTYPE STATUS OF AN INDIVIDUAL                                  ******************/
        /*****************************************************************************************************************************************/
        /*****************************************************************************************************************************************/
        if((Gen > SimParameters.getGenoGeneration() && stage == "preebvcalc") || (Gen >= SimParameters.getGenoGeneration() && stage == "postselcalc"))
        {
            string tempupdanim, tempnewline;
            madechanges = "YES";
            /**************************************************************/
            /* First make a key to figure out whether to Genotype or not  */
            /**************************************************************/
            unordered_map <int, string> GenoStatuslinker; vector <double> randev_M; vector <double> randev_F;
            std::unordered_map <int, string>::const_iterator got;  /* Search if key exists */
            /*****************************************/
            /* Update parents to genotyped if needed */
            /*****************************************/
            if(stage == "postselcalc")
            {
                if(SimParameters.getMaleWhoGenotype()=="parents" || SimParameters.getFemaleWhoGenotype()=="parents" || SimParameters.getMaleWhoGenotype() =="parents_offspring" || SimParameters.getFemaleWhoGenotype()=="parents_offspring" || SimParameters.getMaleWhoGenotype()=="parents_random" ||
                   SimParameters.getFemaleWhoGenotype()=="parents_random" || SimParameters.getMaleWhoGenotype()=="litter_parents_random" ||
                   SimParameters.getFemaleWhoGenotype()=="litter_parents_random" || SimParameters.getMaleWhoGenotype()=="parents_ebv" ||
                   SimParameters.getFemaleWhoGenotype()=="parents_ebv")
                {
                    /* Grab all parents that haven't been genotyped yet */
                    for(int i = 0; i < population.size(); i++)
                    {
                        if(population[i].getSex() == 0 && (SimParameters.getMaleWhoGenotype()=="parents" || SimParameters.getMaleWhoGenotype() =="parents_offspring" || SimParameters.getMaleWhoGenotype()=="parents_random" || SimParameters.getMaleWhoGenotype()=="litter_parents_random" || SimParameters.getMaleWhoGenotype()=="parents_ebv"))
                        {
                            if(population[i].getGenoStatus() == "No" && population[i].getAnimalStage()=="parent"){
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                   GenoStatuslinker.insert({population[i].getID(),"Yes"});
                                } else {GenoStatuslinker[population[i].getID()] = "Yes";} /* If found then update */
                            } else {
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                            }
                        }
                        if(population[i].getSex() == 1 && (SimParameters.getFemaleWhoGenotype()=="parents" || SimParameters.getFemaleWhoGenotype()=="parents_offspring" || SimParameters.getFemaleWhoGenotype()=="parents_random" || SimParameters.getFemaleWhoGenotype()=="litter_parents_random" || SimParameters.getFemaleWhoGenotype()=="parents_ebv"))
                        {
                            if(population[i].getGenoStatus() == "No" && population[i].getAnimalStage()=="parent"){
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),"Yes"});
                                } else {GenoStatuslinker[population[i].getID()] = "Yes";} /* If found then update */
                            } else {
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                            }
                        }
                    }
                }
            }
            /******************************************************/
            /* Update selection candidates to genotyped if needed */
            /******************************************************/
            if(stage == "preebvcalc")
            {
                /******************************************************************************************/
                /* If doing 'offspring' or 'parents_offspring' grab all selection candidates and genotype */
                /******************************************************************************************/
                if(SimParameters.getMaleWhoGenotype()=="offspring" || SimParameters.getFemaleWhoGenotype()=="offspring" ||  SimParameters.getMaleWhoGenotype()== "parents_offspring" || SimParameters.getFemaleWhoGenotype()=="parents_offspring")
                {
                    int animalsgeno = 0;
                    for(int i = 0; i < population.size(); i++)
                    {
                        if(population[i].getSex() == 0)
                        {
                            /************************************************************************/
                            /*** all offspring are genotyped ('offspring' or 'parents_offspring') ***/
                            /************************************************************************/
                            if(population[i].getAnimalStage()=="selcand" && population[i].getGenoStatus()=="No" && (SimParameters.getMaleWhoGenotype()== "offspring" || SimParameters.getMaleWhoGenotype()=="parents_offspring"))
                            {
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),"Yes"}); animalsgeno++;
                                } else {GenoStatuslinker[population[i].getID()] = "Yes";} /* If found then update */
                            } else {
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                            }
                        }
                        if(population[i].getSex() == 1)
                        {
                            /************************************************************************/
                            /*** all offspring are genotyped ('offspring' or 'parents_offspring') ***/
                            /************************************************************************/
                            if(population[i].getAnimalStage()=="selcand" && population[i].getGenoStatus()=="No" && (SimParameters.getFemaleWhoGenotype()== "offspring" || SimParameters.getFemaleWhoGenotype()=="parents_offspring"))
                            {
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),"Yes"}); animalsgeno++;
                                } else {GenoStatuslinker[population[i].getID()] = "Yes";} /* If found then update */
                            } else {
                                got = GenoStatuslinker.find(population[i].getID());
                                if(got == GenoStatuslinker.end()){     /* If not found then add */
                                    GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                            }
                        }
                    }
                }
                /*********************************************************************************/
                /* if doing a certain combination of random_* genotyping of selection candidates */
                /*********************************************************************************/
                if((SimParameters.getMaleWhoGenotype()=="parents_random" || SimParameters.getFemaleWhoGenotype()=="parents_random" || SimParameters.getMaleWhoGenotype()=="random" || SimParameters.getFemaleWhoGenotype()=="random") && stage == "preebvcalc")
                {
                    /* clear randev_M and randev_F in case did one of the previous scenarios */
                    randev_M.clear(); randev_F.clear();
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getAge() == 1)
                        {
                            if(population[i].getSex() == 0){randev_M.push_back(population[i].getRndGeno());}
                            if(population[i].getSex() == 1){randev_F.push_back(population[i].getRndGeno());}
                        }
                    }
                    /* order and take a certain percentage */
                    double temp;
                    for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                    {
                        for(int j=i+1; j< randev_M.size(); j++)
                        {
                            if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                        }
                    }
                    for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                    {
                        for(int j=i+1; j< randev_F.size(); j++)
                        {
                            if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                        }
                    }
                    double malegenotypecutoff = randev_M[int(SimParameters.getMalePropGenotype()*randev_M.size()+0.5)-1];
                    double femalegenotypecutoff = randev_F[int(SimParameters.getFemalePropGenotype()*randev_F.size()+0.5)-1];
                    //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                    //cout << endl << endl;
                    //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                    //cout << endl << endl;
                    //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                    int animalsgeno = 0;
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getAnimalStage()=="selcand" && population[i].getGenoStatus()=="No")
                        {
                            if(population[i].getSex() == 0 && (SimParameters.getMaleWhoGenotype()=="parents_random" || SimParameters.getMaleWhoGenotype()=="random"))
                            {
                                if(population[i].getRndGeno() <= malegenotypecutoff){
                                    got = GenoStatuslinker.find(population[i].getID());
                                    if(got == GenoStatuslinker.end()){     /* If not found then add */
                                        GenoStatuslinker.insert({population[i].getID(),"Yes"}); animalsgeno++;
                                    } else {GenoStatuslinker[population[i].getID()] = "Yes";} /* If found then update */
                                } else {
                                    got = GenoStatuslinker.find(population[i].getID());
                                    if(got == GenoStatuslinker.end()){     /* If not found then add */
                                        GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                    } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                }
                            }
                            if(population[i].getSex() == 1 && (SimParameters.getFemaleWhoGenotype()=="parents_random" ||  SimParameters.getFemaleWhoGenotype()=="random"))
                            {
                                if(population[i].getRndGeno() <= femalegenotypecutoff){
                                    got = GenoStatuslinker.find(population[i].getID());
                                    if(got == GenoStatuslinker.end()){     /* If not found then add */
                                        GenoStatuslinker.insert({population[i].getID(),"Yes"}); animalsgeno++;
                                    } else {GenoStatuslinker[population[i].getID()] = "Yes";} /* If found then update */
                                } else {
                                    got = GenoStatuslinker.find(population[i].getID());
                                    if(got == GenoStatuslinker.end()){     /* If not found then add */
                                        GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                    } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                }
                            }
                        }
                    }
                }
                /*********************************************************************************/
                /* if doing a certain combination of ebv_* genotyping of selection candidates    */
                /*********************************************************************************/
                if(SimParameters.getMaleWhoGenotype() == "ebv" || SimParameters.getMaleWhoGenotype() == "parents_ebv" || SimParameters.getFemaleWhoGenotype() == "ebv" || SimParameters.getFemaleWhoGenotype() == "parents_ebv")
                {
                    /* clear randev_M and randev_F in case did one of the previous scenarios */
                    randev_M.clear(); randev_F.clear();
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getAnimalStage()=="selcand")
                        {
                            if(SimParameters.getSelection() == "index_ebv")
                            {
                                if(population[i].getSex() == 0){randev_M.push_back(population[i].getebvindex());}
                                if(population[i].getSex() == 1){randev_F.push_back(population[i].getebvindex());}
                            }
                            if(SimParameters.getSelection() == "ebv")
                            {
                                if(population[i].getSex() == 0){randev_M.push_back((population[i].get_EBVvect())[0]);}
                                if(population[i].getSex() == 1){randev_F.push_back((population[i].get_EBVvect())[0]);}
                            }
                        }
                    }
                    /* cutoffs for high or low */
                    double malegenotypecutoff, femalegenotypecutoff, temp;
                    /* cutoffs for tailed */
                    double malelowercutoff, maleuppercutoff, femalelowercutoff, femaleuppercutoff;
                    /* sort lowest to highest */
                    if(SimParameters.getGenotypePortionofDistribution() == "low" || SimParameters.getGenotypePortionofDistribution() == "tails")
                    {
                        for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                        {
                            for(int j=i+1; j< randev_M.size(); j++)
                            {
                                if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                            }
                        }
                        /* order and take a certain percentage */
                        for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                        {
                            for(int j=i+1; j< randev_F.size(); j++)
                            {
                                if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                            }
                        }
                    }
                    /* Sort highest to lowest */
                    if(SimParameters.getGenotypePortionofDistribution() == "high")
                    {
                        for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                        {
                            for(int j=i+1; j< randev_M.size(); j++)
                            {
                                if(randev_M[i] < randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                            }
                        }
                        /* order and take a certain percentage */
                        for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                        {
                            for(int j=i+1; j< randev_F.size(); j++)
                            {
                                if(randev_F[i] < randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                            }
                        }
                    }
                    if(SimParameters.getGenotypePortionofDistribution() == "low" || SimParameters.getGenotypePortionofDistribution() == "high")
                    {
                        malegenotypecutoff = randev_M[int(SimParameters.getMalePropGenotype()*randev_M.size()+0.5)-1];
                        femalegenotypecutoff = randev_F[int(SimParameters.getFemalePropGenotype()*randev_F.size()+0.5)-1];
                    }
                    if(SimParameters.getGenotypePortionofDistribution() == "tails")
                    {
                        //cout << "Males: " << randev_M.size() << endl;
                        //cout << randev_M[int(((SimParameters.getMalePropGenotype()*0.5)*randev_M.size()+0.5))-1] << endl;
                        //cout << randev_M[int(((1-(SimParameters.getMalePropGenotype()*0.5))*randev_M.size()+0.5))-1] << endl;
                        malelowercutoff = randev_M[int(((SimParameters.getMalePropGenotype()*0.5)*randev_M.size()+0.5))-1];
                        maleuppercutoff = randev_M[int(((1-(SimParameters.getMalePropGenotype()*0.5))*randev_M.size()+0.5))-1];
                        //cout << endl << "Females: " << randev_F.size() << endl;
                        //cout << randev_F[int(((SimParameters.getFemalePropGenotype()*0.5)*randev_F.size()+0.5))-1] << endl;
                        //cout << randev_F[int(((1-(SimParameters.getFemalePropGenotype()*0.5))*randev_F.size()+0.5))-1] << endl;
                        femalelowercutoff = randev_F[int(((SimParameters.getFemalePropGenotype()*0.5)*randev_F.size()+0.5))-1];
                        femaleuppercutoff = randev_F[int(((1-(SimParameters.getFemalePropGenotype()*0.5))*randev_F.size()+0.5))-1];
                    }
                    //cout << SimParameters.getGenotypePortionofDistribution() << endl;
                    //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                    //cout << endl << endl;
                    //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                    //cout << endl << endl << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                    int femalegeno = 0; int malegeno = 0;
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getSex() == 0 && population[i].getAnimalStage()=="selcand" && population[i].getGenoStatus()=="No" && (SimParameters.getMaleWhoGenotype() == "ebv" || SimParameters.getMaleWhoGenotype() == "parents_ebv"))
                        {
                            if(SimParameters.getGenotypePortionofDistribution() == "high")
                            {
                                if(SimParameters.getSelection() == "index_ebv")
                                {
                                    if(population[i].getebvindex() >= malegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); malegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; malegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                                if(SimParameters.getSelection() == "ebv")
                                {
                                    if((population[i].get_EBVvect())[0] >= malegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); malegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; malegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                           GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                            }
                            if(SimParameters.getGenotypePortionofDistribution() == "low")
                            {
                                if(SimParameters.getSelection() == "index_ebv")
                                {
                                    if(population[i].getebvindex() <= malegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); malegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; malegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                                if(SimParameters.getSelection() == "ebv")
                                {
                                    if((population[i].get_EBVvect())[0] <= malegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); malegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; malegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                            }
                            if(SimParameters.getGenotypePortionofDistribution() == "tails")
                            {
                                if(SimParameters.getSelection() == "index_ebv")
                                {
                                    //cout << population[i].getebvindex() << " " << malelowercutoff << " " << maleuppercutoff <<  endl;
                                    if(population[i].getebvindex()<=malelowercutoff || population[i].getebvindex()>=maleuppercutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); malegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; malegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                                if(SimParameters.getSelection() == "ebv")
                                {
                                    //cout << (population[i].get_EBVvect())[0] << " " << malelowercutoff << " " << maleuppercutoff <<  endl;
                                    if((population[i].get_EBVvect())[0]<=malelowercutoff || (population[i].get_EBVvect())[0]>=maleuppercutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); malegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; malegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                            }
                        }
                        if(population[i].getSex() == 1 && population[i].getAnimalStage()=="selcand" && population[i].getGenoStatus()=="No" && (SimParameters.getFemaleWhoGenotype() == "ebv" || SimParameters.getFemaleWhoGenotype() == "parents_ebv"))
                        {
                            if(SimParameters.getGenotypePortionofDistribution() == "high")
                            {
                                if(SimParameters.getSelection() == "index_ebv")
                                {
                                    if(population[i].getebvindex() >= femalegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); femalegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; femalegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                                if(SimParameters.getSelection() == "ebv")
                                {
                                    if((population[i].get_EBVvect())[0] >= femalegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); femalegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; femalegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                            }
                            if(SimParameters.getGenotypePortionofDistribution() == "low")
                            {
                                if(SimParameters.getSelection() == "index_ebv")
                                {
                                    if(population[i].getebvindex() <= femalegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); femalegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; femalegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                                if(SimParameters.getSelection() == "ebv")
                                {
                                    if((population[i].get_EBVvect())[0] <= femalegenotypecutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); femalegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; femalegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                            }
                            if(SimParameters.getGenotypePortionofDistribution() == "tails")
                            {
                                if(SimParameters.getSelection() == "index_ebv")
                                {
                                    //cout << population[i].getebvindex() << " " << femalelowercutoff << " " << femaleuppercutoff << endl;
                                    if(population[i].getebvindex()<=femalelowercutoff || population[i].getebvindex()>=femaleuppercutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),"Yes"}); femalegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; femalegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                                if(SimParameters.getSelection() == "ebv")
                                {
                                    //cout << (population[i].get_EBVvect())[0] << " " << femalelowercutoff << " " << femaleuppercutoff << " ";
                                    if((population[i].get_EBVvect())[0]<=femalelowercutoff || (population[i].get_EBVvect())[0]>=femaleuppercutoff){
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                             GenoStatuslinker.insert({population[i].getID(),"Yes"}); femalegeno++;
                                        } else {GenoStatuslinker[population[i].getID()] = "Yes"; femalegeno++;} /* If found then update */
                                    } else {
                                        got = GenoStatuslinker.find(population[i].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                             GenoStatuslinker.insert({population[i].getID(),population[i].getGenoStatus()});
                                        } else {GenoStatuslinker[population[i].getID()] = population[i].getGenoStatus();} /* If found then update */
                                    }
                                }
                            }
                        }
                    }
                    if(SimParameters.getGenotypePortionofDistribution() == "low" || SimParameters.getGenotypePortionofDistribution() == "high")
                    {
                        if(stage=="selcand" && (SimParameters.getFemaleWhoGenotype() == "ebv" || SimParameters.getFemaleWhoGenotype() == "parents_ebv"))
                        {
                            if(femalegeno > int(SimParameters.getFemalePropGenotype()*randev_F.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if(SimParameters.getSelection()=="index_ebv" && population[i].getebvindex()==femalegenotypecutoff && population[i].getSex()== 1)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                    if(SimParameters.getSelection()=="ebv" && (population[i].get_EBVvect())[0]==femalegenotypecutoff && population[i].getSex()==1)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                }
                                /* Change to no genotyped */
                                for(int i = 0; i < (femalegeno-int(SimParameters.getFemalePropGenotype()*randev_F.size()+0.5)); i++)
                                {
                                    GenoStatuslinker[ID[i]]="No";
                                }
                            }
                        }
                        if(stage=="selcand" && (SimParameters.getMaleWhoGenotype() == "ebv" || SimParameters.getMaleWhoGenotype() == "parents_ebv"))
                        {
                            if(malegeno > int(SimParameters.getMalePropGenotype()*randev_M.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if(SimParameters.getSelection()=="index_ebv" && population[i].getebvindex()==malegenotypecutoff && population[i].getSex() == 0)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                    if(SimParameters.getSelection()=="ebv" && (population[i].get_EBVvect())[0]==malegenotypecutoff && population[i].getSex() == 0)
                                    {
                                        ID.push_back(population[i].getID());
                                    }
                                }
                                /* Change to no genotyped */
                                for(int i = 0; i < (malegeno-int(SimParameters.getMalePropGenotype()*randev_M.size()+0.5)); i++){GenoStatuslinker[ID[i]]="No";}
                            }
                        }
                    }
                    if(SimParameters.getGenotypePortionofDistribution() == "tails")
                    {
                        if(stage =="selcand" && (SimParameters.getFemaleWhoGenotype() == "ebv" || SimParameters.getFemaleWhoGenotype() == "parents_ebv"))
                        {
                            if(femalegeno > int(SimParameters.getFemalePropGenotype()*randev_F.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if(SimParameters.getSelection() == "index_ebv")
                                    {
                                        if(population[i].getebvindex()==femalelowercutoff || population[i].getebvindex()==femaleuppercutoff && population[i].getSex() == 1){ID.push_back(population[i].getID());}
                                    }
                                    if(SimParameters.getSelection() == "ebv")
                                    {
                                        if((population[i].get_EBVvect())[0]==femalelowercutoff || (population[i].get_EBVvect())[0]==femaleuppercutoff && population[i].getSex() == 1){ID.push_back(population[i].getID());}
                                    }
                                }
                                /* Change to no phenotyped */
                                for(int i = 0; i < (femalegeno-int(SimParameters.getFemalePropGenotype()*randev_F.size()+0.5)); i++)
                                {
                                    GenoStatuslinker[ID[i]]="No";
                                }
                            }
                        }
                        if(stage=="selcand" && (SimParameters.getMaleWhoGenotype() == "ebv" || SimParameters.getMaleWhoGenotype() == "parents_ebv"))
                        {
                            if(malegeno > int(SimParameters.getMalePropGenotype()*randev_M.size()+0.5))
                            {
                                vector < int > ID;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if(SimParameters.getSelection() == "index_ebv")
                                    {
                                        if(population[i].getebvindex()==malelowercutoff || population[i].getebvindex()==maleuppercutoff && population[i].getSex()==0){ID.push_back(population[i].getID());}
                                    }
                                    if(SimParameters.getSelection() == "ebv")
                                    {
                                        if((population[i].get_EBVvect())[0]==malelowercutoff || (population[i].get_EBVvect())[0]==maleuppercutoff && population[i].getSex()==0){ID.push_back(population[i].getID());}
                                    }
                                }
                                /* Change to no phenotyped */
                                for(int i = 0; i < (malegeno-int(SimParameters.getMalePropGenotype()*randev_M.size()+0.5)); i++){GenoStatuslinker[ID[i]]="No";}
                            }
                        }
                    }
                }
                /*******************************************************************************************/
                /* if doing a certain combination of litter_random_* genotyping of selection candidates    */
                /*******************************************************************************************/
                if(SimParameters.getMaleWhoGenotype() == "litter_random" || SimParameters.getFemaleWhoGenotype() == "litter_random" || SimParameters.getMaleWhoGenotype() == "litter_parents_random" || SimParameters.getFemaleWhoGenotype() == "litter_parents_random")
                {
                    /* First get index for Dams and all associated offspring within the pedigree class */
                    unordered_map <int, int> Damlinker; unordered_map <int, int> Damlinkerindex; int numdams = 0;  unordered_map <int, int> Offspringlinker;
                    std::unordered_map<int,int>::const_iterator gota;  /* Search if key exists */
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getAnimalStage()=="selcand")   /* Grab all selection candidates */
                        {
                            /* Grab dams and save in hash-table if not already there */
                            if(population[i].getDam() != 0)
                            {
                                gota = Damlinker.find(population[i].getDam());
                                if(gota == Damlinker.end())     /* If not found then add */
                                {
                                    Damlinker.insert({population[i].getDam(),i});
                                    Damlinkerindex.insert({population[i].getDam(),numdams}); numdams++;
                                }
                            }
                            Offspringlinker.insert({population[i].getID(),i});
                        }
                    }
                    /* Intialize a male and female 2d vector to store ids for all offspring of a given dam */
                    vector< vector < int >> Maleoffspring(Damlinker.size(),vector < int > (0));
                    vector< vector < int >> Femaleoffspring(Damlinker.size(),vector < int > (0));
                    for(int i = 0; i < population.size(); i++)
                    {
                        if(population[i].getAnimalStage()=="selcand")
                        {
                            if(population[i].getSex() == 0){Maleoffspring[Damlinkerindex[population[i].getDam()]].push_back(population[i].getID());}
                            if(population[i].getSex() == 1){Femaleoffspring[Damlinkerindex[population[i].getDam()]].push_back(population[i].getID());}
                        }
                    }
                    int malenumber = int(SimParameters.getMalePropGenotype()*(SimParameters.getOffspring()*0.5)+0.5);
                    int femalenumber = int(SimParameters.getFemalePropGenotype()*(SimParameters.getOffspring()*0.5)+0.5);
                    int genomale = 0; int genofemale = 0;
                    /******************/
                    /* Do Males First */
                    /******************/
                    if(SimParameters.getMaleWhoGenotype() == "litter_random" || SimParameters.getMaleWhoGenotype() == "litter_parents_random")
                    {
                        for(int i = 0; i < Damlinker.size(); i++)
                        {
                            if(malenumber < Maleoffspring[i].size())    /* if number of males larger than number genotyped than randomly select */
                            {
                                vector < double > randomvalues;
                                for(int j = 0; j < Maleoffspring[i].size(); j++)
                                {
                                    randomvalues.push_back(population[Offspringlinker[Maleoffspring[i][j]]].getRndGeno());
                                }
                                double temp; double cutoff;
                                for(int j = 0; j < (randomvalues.size()-1); j++)   /* Sort */
                                {
                                    for(int k=j+1; k < randomvalues.size(); k++)
                                    {
                                        if(randomvalues[j] > randomvalues[k])
                                        {
                                            temp = randomvalues[j]; randomvalues[j] = randomvalues[k]; randomvalues[k] = temp;
                                        }
                                    }
                                }
                                cutoff = randomvalues[malenumber-1];
                                for(int j = 0; j < Maleoffspring[i].size(); j++)
                                {
                                    if(population[Offspringlinker[Maleoffspring[i][j]]].getRndGeno() <= cutoff){
                                        got = GenoStatuslinker.find(population[Offspringlinker[Maleoffspring[i][j]]].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); genomale++;
                                        } else {GenoStatuslinker[population[Offspringlinker[Maleoffspring[i][j]]].getID()] = "Yes";}
                                    } else {
                                        got = GenoStatuslinker.find(population[Offspringlinker[Maleoffspring[i][j]]].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),population[Offspringlinker[Maleoffspring[i][j]]].getGenoStatus()});
                                        } else {GenoStatuslinker[population[Offspringlinker[Maleoffspring[i][j]]].getID()] = population[Offspringlinker[Maleoffspring[i][j]]].getGenoStatus();} /* If found then update */
                                    }
                                }
                            } else {
                                for(int j = 0; j < Maleoffspring[i].size(); j++)
                                {
                                    got = GenoStatuslinker.find(population[Offspringlinker[Maleoffspring[i][j]]].getID());
                                    if(got == GenoStatuslinker.end()){     /* If not found then add */
                                        GenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); genomale++;
                                    } else {GenoStatuslinker[population[Offspringlinker[Maleoffspring[i][j]]].getID()] = "Yes";}
                                }
                            }
                        }
                    }
                    /*******************/
                    /* Then Do Females */
                    /*******************/
                    if(SimParameters.getFemaleWhoGenotype() == "litter_random" || SimParameters.getFemaleWhoGenotype() == "litter_parents_random")
                    {
                        for(int i = 0; i < Damlinker.size(); i++)
                        {
                            if(femalenumber < Femaleoffspring[i].size())    /* if number of males larger than number genotyped than randomly select */
                            {
                                vector < double > randomvalues;
                                for(int j = 0; j < Femaleoffspring[i].size(); j++)
                                {
                                    randomvalues.push_back(population[Offspringlinker[Femaleoffspring[i][j]]].getRndGeno());
                                }
                                double temp; double cutoff;
                                for(int j = 0; j < (randomvalues.size()-1); j++)   /* Sort */
                                {
                                    for(int k=j+1; k < randomvalues.size(); k++)
                                    {
                                        if(randomvalues[j] > randomvalues[k]){temp = randomvalues[j]; randomvalues[j] = randomvalues[k]; randomvalues[k] = temp;}
                                    }
                                }
                                cutoff = randomvalues[malenumber-1];
                                for(int j = 0; j < Femaleoffspring[i].size(); j++)
                                {
                                    if(population[Offspringlinker[Femaleoffspring[i][j]]].getRndGeno() <= cutoff){
                                        got = GenoStatuslinker.find(population[Offspringlinker[Femaleoffspring[i][j]]].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); genofemale++;
                                        } else {GenoStatuslinker[population[Offspringlinker[Femaleoffspring[i][j]]].getID()] = "Yes";}
                                    } else {
                                        got = GenoStatuslinker.find(population[Offspringlinker[Femaleoffspring[i][j]]].getID());
                                        if(got == GenoStatuslinker.end()){     /* If not found then add */
                                            GenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),population[Offspringlinker[Femaleoffspring[i][j]]].getGenoStatus()});
                                        } else {GenoStatuslinker[population[Offspringlinker[Femaleoffspring[i][j]]].getID()] = population[Offspringlinker[Femaleoffspring[i][j]]].getGenoStatus();} /* If found then update */
                                    }
                                }
                            } else {
                                for(int j = 0; j < Femaleoffspring[i].size(); j++)
                                {
                                    got = GenoStatuslinker.find(population[Offspringlinker[Femaleoffspring[i][j]]].getID());
                                    if(got == GenoStatuslinker.end()){     /* If not found then add */
                                        GenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); genofemale++;
                                    } else {GenoStatuslinker[population[Offspringlinker[Femaleoffspring[i][j]]].getID()] = "Yes";}
                                }
                            }
                        }
                    }
                }
            }
            /* Now go through and update the Geno_Pheno Status genotype file */
            for(int i = 0; i < population.size(); i++)
            {
                if(stage == "postselcalc")
                {
                    if((SimParameters.getMaleWhoGenotype()=="parents" || SimParameters.getMaleWhoGenotype() =="parents_offspring" || SimParameters.getMaleWhoGenotype()=="parents_random" || SimParameters.getMaleWhoGenotype()=="parents_ebv" || SimParameters.getMaleWhoGenotype()== "litter_parents_random") && GenoStatuslinker[population[i].getID()] == "Yes")
                    {
                        population[i].UpdateGenoStatus("Yes");
                        /* Split off first three and then update correct one with a Yes*/
                        tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                        size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                        if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                        {
                            cout << "Error Indexing geno-pheno status(male + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                        }
                        tempupdanim.erase(0, pos+1); pos = tempupdanim.find(" ",0); tempnewline += "Yes "; tempupdanim.erase(0, pos+1);
                        pos = tempupdanim.find(" ",0); tempnewline += "parent "; tempupdanim.erase(0, pos+1);
                        tempnewline += tempupdanim; GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                    }
                    if((SimParameters.getFemaleWhoGenotype()=="parents" || SimParameters.getFemaleWhoGenotype() =="parents_offspring" || SimParameters.getFemaleWhoGenotype()=="parents_random" || SimParameters.getFemaleWhoGenotype()=="parents_ebv" || SimParameters.getFemaleWhoGenotype()== "litter_parents_random") && GenoStatuslinker[population[i].getID()] == "Yes")
                    {
                        
                        population[i].UpdateGenoStatus("Yes");
                        /* Split off first three and then update correct one with a Yes*/
                        tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                        size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                        if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                        {
                            cout << "Error Indexing geno-pheno status(male + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                        }
                        tempupdanim.erase(0, pos+1); pos = tempupdanim.find(" ",0); tempnewline += "Yes "; tempupdanim.erase(0, pos+1);
                        pos = tempupdanim.find(" ",0); tempnewline += "parent "; tempupdanim.erase(0, pos+1);
                        tempnewline += tempupdanim; GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                    }
                }
                if(stage == "preebvcalc")
                {
                    if((SimParameters.getMaleWhoGenotype()=="offspring" || SimParameters.getMaleWhoGenotype()== "parents_offspring" || SimParameters.getMaleWhoGenotype()=="random" || SimParameters.getMaleWhoGenotype()=="parents_random" || SimParameters.getMaleWhoGenotype()=="ebv" || SimParameters.getMaleWhoGenotype()=="parents_ebv" || SimParameters.getMaleWhoGenotype()=="litter_random" || SimParameters.getMaleWhoGenotype()=="litter_parents_random") && population[i].getSex() == 0 && GenoStatuslinker[population[i].getID()] == "Yes")
                    {
                        population[i].UpdateGenoStatus("Yes");
                        /* Split off first three and then update correct one with a Yes*/
                        tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                        size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                        if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                        {
                            cout << "Error Indexing geno-pheno status(male + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                        }
                        tempupdanim.erase(0, pos+1); pos = tempupdanim.find(" ",0); tempnewline += "Yes "; tempupdanim.erase(0, pos+1);
                        pos = tempupdanim.find(" ",0); tempnewline += "selectioncandidate "; tempupdanim.erase(0, pos+1);
                        tempnewline += tempupdanim; GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                    }
                    if((SimParameters.getFemaleWhoGenotype()=="offspring" || SimParameters.getFemaleWhoGenotype()== "parents_offspring" || SimParameters.getFemaleWhoGenotype()=="random" || SimParameters.getFemaleWhoGenotype()=="parents_random" || SimParameters.getFemaleWhoGenotype()=="ebv" || SimParameters.getFemaleWhoGenotype()=="parents_ebv" || SimParameters.getFemaleWhoGenotype()=="litter_random" || SimParameters.getFemaleWhoGenotype()=="litter_parents_random") && population[i].getSex() == 1 && GenoStatuslinker[population[i].getID()] == "Yes")
                    {
                        population[i].UpdateGenoStatus("Yes");
                        /* Split off first three and then update correct one with a Yes*/
                        tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                        size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                        if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                        {
                            cout << "Error Indexing geno-pheno status(male + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                        }
                        tempupdanim.erase(0, pos+1); pos = tempupdanim.find(" ",0); tempnewline += "Yes "; tempupdanim.erase(0, pos+1);
                        pos = tempupdanim.find(" ",0); tempnewline += "selectioncandidate "; tempupdanim.erase(0, pos+1);
                        tempnewline += tempupdanim; GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                    }
                }
            }
            //if(Gen == 7 && stage == "preebvcalc")
            //{
            //    fstream Check; Check.open("LHS", std::fstream::out | std::fstream::trunc); Check.close();
            //    std::ofstream output9("LHS", std::ios_base::app | std::ios_base::out);
            //    for(int i = 0; i < population.size(); i++)
            //    {
            //        if(population[i].getAnimalStage()=="selcand")
            //        {
            //            output9 <<population[i].getID()<<" "<<(population[i].get_EBVvect())[0]<<" "<<population[i].getGenoStatus()<<endl;
            //        }
            //    }
            //    exit (EXIT_FAILURE);
            //}
        }
        /*****************************************************************************************************************************************/
        /*****************************************************************************************************************************************/
        /******************                           UPDATE PHENOTYPE STATUS OF AN INDIVIDUAL                                  ******************/
        /*****************************************************************************************************************************************/
        /*****************************************************************************************************************************************/
        for(int trt = 0; trt < ebvtraits; trt++)
        {
            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]!="pheno_atselection" || (SimParameters.get_FemaleWhoPhenotype_vec())[trt]!="pheno_atselection" || (SimParameters.get_FemaleWhoPhenotype_vec())[trt]!="random_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]!="random_atselection" || (SimParameters.get_FemaleWhoPhenotype_vec())[trt]!="ebv_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]!="ebv_atselection" ||
               (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_atselection")
            {
                /* If got to this step then need to update pheno_geno status so delete old one */
                /* loop through population and update animal phenotype status for animals in the population class */
                string tempupdanim, tempnewline, tempnewlinea, tempnewlineb;
                if(madechanges == "NO"){madechanges = "YES";}
                //for(int i = 0; i < GenoPhenoStatus.size(); i++){cout << GenoPhenoStatus[i] << " ";}
                //cout << endl << endl;
                /**************************************************************/
                /* First make a key to figure out whether to phenotype or not */
                /************************************************************************************/
                unordered_map <int, string> PhenoStatuslinker; vector <double> randev_M; vector <double> randev_F;
                /**********************************************************************************************************************************************/
                /**********************************************************************************************************************************************/
                /* if doing a certain combination of random_* phenotyping first determine whether an animal is phenotyped prior to looping through population */
                /**********************************************************************************************************************************************/
                /**********************************************************************************************************************************************/
                if((((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_afterselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_afterselection") && stage == "postebvcalc") || (((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_parents" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_parents") && stage == "postselcalc"))
                {
                    for(int i = 0; i < population.size();  i++)
                    {
                        if(population[i].getSex() == 0)
                        {
                            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_afterselection" && population[i].getAnimalStage()=="selcandebv")
                            {
                                if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                            }
                            /* Age 2 would be the most recently selected parents; Only sampled for phenotype once which occurs when they first produced offspring */
                            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1 && stage == "postselcalc")
                            {
                                if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                            }
                        }
                        if(population[i].getSex() == 1)
                        {
                            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_afterselection" && population[i].getAnimalStage()=="selcandebv")
                            {
                                if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                            }
                            /* Age 2 would be the most recently selected parents */
                            if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                            {
                                if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                            }
                        }
                    }
                    //cout << randev_F.size() << " " << randev_M.size() << endl;
                    if(randev_F.size() > 0 || randev_M.size() > 0)
                    {
                        double malegenotypecutoff, femalegenotypecutoff, temp;
                        /* order and take a certain percentage */
                        if(randev_M.size() > 0)
                        {
                            for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                            {
                                for(int j=i+1; j< randev_M.size(); j++)
                                {
                                    if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                }
                            }
                            malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                        }
                        if(randev_F.size() > 0)
                        {
                            for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                            {
                                for(int j=i+1; j< randev_F.size(); j++)
                                {
                                    if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                }
                            }
                            femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                        }
                        //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                        //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                        //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                        //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                        //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                        //int numfemale = 0; int nummale = 0;
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(randev_M.size() > 0)
                            {
                                if(population[i].getSex() == 0)
                                {
                                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                            if(population[i].getSex() == 1)
                            {
                                if(randev_F.size() > 0)
                                {
                                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                            } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                        }
                                    }
                                }
                            }
                        }
                        //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                    }
                }
                /**********************************************************************************************************************************************/
                /**********************************************************************************************************************************************/
                /* if doing a certain combination of ebv_* phenotyping first determine whether an animal is phenotyped prior to looping through population    */
                /**********************************************************************************************************************************************/
                /**********************************************************************************************************************************************/
                if((((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_afterselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_afterselection") && stage == "postebvcalc") || (((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_parents" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_parents") && stage == "postselcalc"))
                {
                    if((SelectionVector[Gen-1]) == "random"){
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(population[i].getSex() == 0)
                            {
                                if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                {
                                    if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                    if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                                }
                                /* Age 2 would be the most recently selected parents; Only sampled for phenotype once which occurs when they first produced offspring */
                                if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1 && stage == "postselcalc")
                                {
                                    if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                    if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                                }
                            }
                            if(population[i].getSex() == 1)
                            {
                                if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                {
                                    if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                    if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                                }
                                /* Age 2 would be the most recently selected parents */
                                if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                {
                                    if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                    if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                                }
                            }
                        }
                        //cout << randev_F.size() << " " << randev_M.size() << endl;
                        if(randev_F.size() > 0 || randev_M.size() > 0)
                        {
                            double malegenotypecutoff, femalegenotypecutoff, temp;
                            /* order and take a certain percentage */
                            if(randev_M.size() > 0)
                            {
                                for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                                {
                                    for(int j=i+1; j< randev_M.size(); j++)
                                    {
                                        if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                    }
                                }
                                malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                            }
                            if(randev_F.size() > 0)
                            {
                                for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                                {
                                    for(int j=i+1; j< randev_F.size(); j++)
                                    {
                                        if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                    }
                                }
                                femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                            }
                            //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                            //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                            //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                            //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                            //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                            //int numfemale = 0; int nummale = 0;
                            for(int i = 0; i < population.size();  i++)
                            {
                                if(randev_M.size() > 0)
                                {
                                    if(population[i].getSex() == 0)
                                    {
                                        if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                        {
                                            if(trt == 0)
                                            {
                                                if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(trt == 1)
                                            {
                                                if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                        if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                        {
                                            if(trt == 0)
                                            {
                                                if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(trt == 1)
                                            {
                                                if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                    }
                                }
                                if(population[i].getSex() == 1)
                                {
                                    if(randev_F.size() > 0)
                                    {
                                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                        {
                                            if(trt == 0)
                                            {
                                                if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(trt == 1)
                                            {
                                                if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                        {
                                            if(trt == 0)
                                            {
                                                if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(trt == 1)
                                            {
                                                if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                    }
                                }
                            }
                            //cout << numfemale << " " << nummale << endl;
                            //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                        }
                    } else if((SelectionVector[Gen-1]) == "ebv" || (SelectionVector[Gen-1]) == "index_ebv"){
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(population[i].getSex() == 0)
                            {
                                if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                {
                                    randev_M.push_back((population[i].get_EBVvect())[trt]);
                                }
                                /* Age 2 would be the most recently selected parents; Only sampled for phenotype once which occurs when they first produced offspring */
                                if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1 && stage == "postselcalc")
                                {
                                    randev_M.push_back((population[i].get_EBVvect())[trt]);
                                }
                            }
                            if(population[i].getSex() == 1)
                            {
                                if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                {
                                    randev_F.push_back((population[i].get_EBVvect())[trt]);
                                }
                                /* Age 2 would be the most recently selected parents */
                                if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                {
                                    randev_F.push_back((population[i].get_EBVvect())[trt]);
                                }
                            }
                        }
                        //cout << randev_M.size() << " " << randev_F.size() << endl;
                        //cout << SimParameters.get_PortionofDistribution_vec()[trt] << endl;
                        if(randev_F.size() > 0 || randev_M.size() > 0)
                        {
                            double malegenotypecutoff, femalegenotypecutoff, temp;
                            /* cutoffs for tailed */
                            double malelowercutoff, maleuppercutoff, femalelowercutoff, femaleuppercutoff;
                            /* order and take a certain percentage */
                            if(randev_M.size() > 0)
                            {
                                /* sort lowest to highest */
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                {
                                    for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                                    {
                                        for(int j=i+1; j< randev_M.size(); j++)
                                        {
                                            if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                        }
                                    }
                                }
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")   /* Sort highest to lowest */
                                {
                                    for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                                    {
                                        for(int j=i+1; j< randev_M.size(); j++)
                                        {
                                            if(randev_M[i] < randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                        }
                                    }
                                }
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                {
                                    malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                                }
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                {
                                    //cout << randev_M.size() << endl;
                                    //cout << ((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5) << " ";
                                    //cout << (1 - ((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5)) << endl;
                                    //cout << int((((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5)*randev_M.size()+0.5)) << endl;
                                    //cout << int(((1-((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5))*randev_M.size()+0.5)) << endl;
                                    //cout << randev_M[int((((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5)*randev_M.size()+0.5))-1] << endl;
                                    //cout << randev_M[int(((1-((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5))*randev_M.size()+0.5))-1] << endl;
                                    malelowercutoff = randev_M[int((((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5)*randev_M.size()+0.5))-1];
                                    maleuppercutoff = randev_M[int(((1-((SimParameters.get_MalePropPhenotype_vec())[trt]*0.5))*randev_M.size()))-1];
                                }
                            }
                            if(randev_F.size() > 0)
                            {
                                /* sort lowest to highest */
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                {
                                    for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                                    {
                                        for(int j=i+1; j< randev_F.size(); j++)
                                        {
                                            if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                        }
                                    }
                                }
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")   /* Sort highest to lowest */
                                {
                                    for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                                    {
                                        for(int j=i+1; j< randev_F.size(); j++)
                                        {
                                            if(randev_F[i] < randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                        }
                                    }
                                }
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                {
                                    femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                                }
                                if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                {
                                    //cout << randev_F.size() << endl;
                                    //cout << ((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5) << " ";
                                    //cout << (1 - ((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5)) << endl;
                                    //cout << int((((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5)*randev_F.size()+0.5)) << endl;
                                    //cout << int(((1-((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5))*randev_F.size()+0.5)) << endl;
                                    //cout << randev_M[int((((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5)*randev_F.size()+0.5))-1] << endl;
                                    //cout << randev_M[int(((1-((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5))*randev_F.size()+0.5))-1] << endl;
                                    femalelowercutoff = randev_F[int((((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5)*randev_F.size()+0.5))-1];
                                    femaleuppercutoff = randev_F[int(((1-((SimParameters.get_FemalePropPhenotype_vec())[trt]*0.5))*randev_F.size()))-1];
                                }
                            }
                            //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                            //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                            //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                            //cout << endl << endl;
                            //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                            //cout << endl << endl;
                            //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                            int numfemale = 0; int nummale = 0;
                            for(int i = 0; i < population.size();  i++)
                            {
                                if(randev_M.size() > 0)
                                {
                                    if(population[i].getSex() == 0)
                                    {
                                        if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                        {
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                            {
                                                if((population[i].get_EBVvect())[trt] >= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low")
                                            {
                                                if((population[i].get_EBVvect())[trt] <= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                            {
                                                if((population[i].get_EBVvect())[trt]<=malelowercutoff || (population[i].get_EBVvect())[trt]>=maleuppercutoff)
                                                {
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                        if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1  && stage == "postselcalc")
                                        {
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                            {
                                                if((population[i].get_EBVvect())[trt] >= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low")
                                            {
                                                if((population[i].get_EBVvect())[trt] <= malegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                            {
                                                if((population[i].get_EBVvect())[trt]<=malelowercutoff || (population[i].get_EBVvect())[trt]>=maleuppercutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); nummale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                    }
                                }
                                if(randev_F.size() > 0)
                                {
                                    if(population[i].getSex() == 1)
                                    {
                                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                        {
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                            {
                                                if((population[i].get_EBVvect())[trt] >= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low")
                                            {
                                                if((population[i].get_EBVvect())[trt] <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                            {
                                                if((population[i].get_EBVvect())[trt]<=femalelowercutoff || (population[i].get_EBVvect())[trt]>=femaleuppercutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_parents" && population[i].getAnimalStage()=="parent" && population[i].getAge()==1 && stage == "postselcalc")
                                        {
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                                            {
                                                if((population[i].get_EBVvect())[trt] >= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low")
                                            {
                                                if((population[i].get_EBVvect())[trt] <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                                            {
                                                if((population[i].get_EBVvect())[trt] <= femalelowercutoff || (population[i].get_EBVvect())[trt] >= femaleuppercutoff)
                                                {
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                    }
                                }
                            }
                            //cout << nummale << " " << numfemale << endl;
                            //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "low" || SimParameters.get_PortionofDistribution_vec()[trt] == "high")
                            {
                                if(numfemale > int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5))
                                {
                                    vector < int > ID;
                                    for(int i = 0; i < population.size();  i++)
                                    {
                                        if((population[i].get_EBVvect())[trt] == femalegenotypecutoff && population[i].getSex() == 1)
                                        {
                                            ID.push_back(population[i].getID());
                                        }
                                    }
                                    /* Change to no phenotyped */
                                    for(int i = 0; i < (numfemale-int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)); i++)
                                    {
                                        PhenoStatuslinker[ID[i]] = "No";
                                    }
                                }
                                if(nummale > int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5))
                                {
                                    vector < int > ID;
                                    for(int i = 0; i < population.size();  i++)
                                    {
                                        if((population[i].get_EBVvect())[trt] == malegenotypecutoff && population[i].getSex() == 0)
                                        {
                                            ID.push_back(population[i].getID());
                                        }
                                    }
                                    /* Change to no phenotyped */
                                    for(int i = 0; i < (nummale-int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)); i++)
                                    {
                                        PhenoStatuslinker[ID[i]] = "No";
                                    }
                                }
                            }
                            if(SimParameters.get_PortionofDistribution_vec()[trt] == "tails")
                            {
                                if(numfemale > int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5))
                                {
                                    vector < int > ID;
                                    for(int i = 0; i < population.size();  i++)
                                    {
                                        if((population[i].get_EBVvect())[trt] == femalelowercutoff || (population[i].get_EBVvect())[trt] == femaleuppercutoff && population[i].getSex() == 1)
                                        {
                                            ID.push_back(population[i].getID());
                                        }
                                    }
                                    /* Change to no phenotyped */
                                    for(int i = 0; i < (numfemale-int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)); i++)
                                    {
                                        PhenoStatuslinker[ID[i]] = "No";
                                    }
                                }
                                if(nummale > int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5))
                                {
                                    vector < int > ID;
                                    for(int i = 0; i < population.size();  i++)
                                    {
                                        if((population[i].get_EBVvect())[trt] == malelowercutoff || (population[i].get_EBVvect())[trt] == maleuppercutoff && population[i].getSex() == 0)
                                        {
                                            ID.push_back(population[i].getID());
                                        }
                                    }
                                    /* Change to no phenotyped */
                                    for(int i = 0; i < (nummale-int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)); i++)
                                    {
                                        PhenoStatuslinker[ID[i]] = "No";
                                    }
                                }
                            }
                            //cout << nummale << " " << numfemale << endl;
                        }
                    } else {cout << endl << "Shouldn't be Here (Line 613)" << endl; exit (EXIT_FAILURE);}
                }
                /**********************************************************************************************************************************************/
                /**********************************************************************************************************************************************/
                /* if doing a certain combination of litter_* phenotyping, determine whether an animal is phenotyped prior to looping through population    */
                /**********************************************************************************************************************************************/
                /**********************************************************************************************************************************************/
                if((((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection") && stage == "postebvcalc"))
                {
                    if((stage == "postebvcalc" && ((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection")))
                    {
                        /* First get index for Dams and all associated offspring within the pedigree class */
                        unordered_map <int, int> Damlinker; unordered_map <int, int> Damlinkerindex; int numdams = 0;  unordered_map <int, int> Offspringlinker;
                        std::unordered_map<int,int>::const_iterator got;  /* Search if key exists */
                        for(int i = 0; i < population.size();  i++)
                        {
                            if(population[i].getAnimalStage()=="selcandebv")   /* Grab all selection candidates */
                            {
                                /* Grab dams and save in hash-table if not already there */
                                if(population[i].getDam() != 0)
                                {
                                    got = Damlinker.find(population[i].getDam());
                                    if(got == Damlinker.end())     /* If not found then add */
                                    {
                                        Damlinker.insert({population[i].getDam(),i});
                                        Damlinkerindex.insert({population[i].getDam(),numdams}); numdams++;
                                    }
                                }
                                if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && population[i].getSex() == 0)
                                {
                                    Offspringlinker.insert({population[i].getID(),i});
                                }
                                if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && population[i].getSex() == 1)
                                {
                                    Offspringlinker.insert({population[i].getID(),i});
                                }
                            }
                        }
                        /* random phenotype selection this happens in the founders and then defaults to within a litter after that */
                        if(Damlinker.size() == 0)
                        {
                            for(int i = 0; i < population.size();  i++)
                            {
                                if(population[i].getSex() == 0)
                                {
                                    if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                    {
                                        if(trt == 0){randev_M.push_back(population[i].getRndPheno1());}
                                        if(trt == 1){randev_M.push_back(population[i].getRndPheno2());}
                                    }
                                }
                                if(population[i].getSex() == 1)
                                {
                                    if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                    {
                                        if(trt == 0){randev_F.push_back(population[i].getRndPheno1());}
                                        if(trt == 1){randev_F.push_back(population[i].getRndPheno2());}
                                    }
                                }
                            }
                            //cout << randev_F.size() << " " << randev_M.size() << endl;
                            if(randev_F.size() > 0 || randev_M.size() > 0)
                            {
                                double malegenotypecutoff, femalegenotypecutoff, temp;
                                /* order and take a certain percentage */
                                if(randev_M.size() > 0)
                                {
                                    for(int i = 0; i < (randev_M.size() -1); i++)   /* Sort Males */
                                    {
                                        for(int j=i+1; j< randev_M.size(); j++)
                                        {
                                            if(randev_M[i] > randev_M[j]){temp = randev_M[i]; randev_M[i] = randev_M[j]; randev_M[j] = temp;}
                                        }
                                    }
                                    malegenotypecutoff = randev_M[int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5)-1];
                                }
                                if(randev_F.size() > 0)
                                {
                                    for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
                                    {
                                        for(int j=i+1; j< randev_F.size(); j++)
                                        {
                                            if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
                                        }
                                    }
                                    femalegenotypecutoff = randev_F[int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5)-1];
                                }
                                //cout << int((SimParameters.get_FemalePropPhenotype_vec())[trt]*randev_F.size()+0.5) << " ";
                                //cout << int((SimParameters.get_MalePropPhenotype_vec())[trt]*randev_M.size()+0.5) << endl;
                                //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
                                //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
                                //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
                                //int numfemale = 0; int nummale = 0;
                                for(int i = 0; i < population.size();  i++)
                                {
                                    if(randev_M.size() > 0)
                                    {
                                        if(population[i].getSex() == 0)
                                        {
                                            if((SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                            {
                                                if(trt == 0)
                                                {
                                                    if(population[i].getRndPheno1() <= malegenotypecutoff){
                                                        PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                                    } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                                }
                                                if(trt == 1)
                                                {
                                                    if(population[i].getRndPheno2() <= malegenotypecutoff){
                                                        PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //nummale++;
                                                    } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                                }
                                            }
                                        }
                                    }
                                    if(population[i].getSex() == 1)
                                    {
                                        if((SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && population[i].getAnimalStage()=="selcandebv")
                                        {
                                            if(trt == 0)
                                            {
                                                if(population[i].getRndPheno1() <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                            if(trt == 1)
                                            {
                                                if(population[i].getRndPheno2() <= femalegenotypecutoff){
                                                    PhenoStatuslinker.insert({population[i].getID(),"Yes"}); //numfemale++;
                                                } else {PhenoStatuslinker.insert({population[i].getID(),"No"});}
                                            }
                                        }
                                    }
                                }
                                //cout << numfemale << " " << nummale << endl;
                                //for(pair<int,string> element:PhenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
                            }
                        }
                        if(Damlinker.size() > 0)
                        {
                            /* Intialize a male and female 2d vector to store ids for all offspring of a given dam */
                            vector< vector < int >> Maleoffspring(Damlinker.size(),vector < int > (0));
                            vector< vector < int >> Femaleoffspring(Damlinker.size(),vector < int > (0));
                            for(int i = 0; i < population.size(); i++)
                            {
                                if(population[i].getAnimalStage()=="selcandebv")
                                {
                                    if(population[i].getSex() == 0){Maleoffspring[Damlinkerindex[population[i].getDam()]].push_back(population[i].getID());}
                                    if(population[i].getSex() == 1){Femaleoffspring[Damlinkerindex[population[i].getDam()]].push_back(population[i].getID());}
                                }
                            }
                            //for(int i = 0; i < Maleoffspring.size(); i++){cout<<Maleoffspring[i].size()<<" "<<Femaleoffspring[i].size()<<endl;}
                            int malenumber = int((SimParameters.get_MalePropPhenotype_vec())[trt]*(SimParameters.getOffspring()*0.5)+0.5);
                            int femalenumber = int((SimParameters.get_FemalePropPhenotype_vec())[trt]*(SimParameters.getOffspring()*0.5)+0.5);
                            int phenomale = 0; int phenofemale = 0;
                            /******************/
                            /* Do Males First */
                            /******************/
                            for(int i = 0; i < Damlinker.size(); i++)
                            {
                                if(malenumber < Maleoffspring[i].size())    /* if number of males larger than number genotyped than randomly select */
                                {
                                    vector < double > randomvalues;
                                    for(int j = 0; j < Maleoffspring[i].size(); j++)
                                    {
                                        if(trt == 0){randomvalues.push_back(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno1());}
                                        if(trt == 1){randomvalues.push_back(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno2());}
                                    }
                                    double temp; double cutoff;
                                    for(int j = 0; j < (randomvalues.size()-1); j++)   /* Sort */
                                    {
                                        for(int k=j+1; k < randomvalues.size(); k++)
                                        {
                                            if(randomvalues[j] > randomvalues[k])
                                            {
                                                temp = randomvalues[j];
                                                randomvalues[j] = randomvalues[k];
                                                randomvalues[k] = temp;
                                            }
                                        }
                                    }
                                    cutoff = randomvalues[malenumber-1];
                                    for(int j = 0; j < Maleoffspring[i].size(); j++)
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno1() <= cutoff){
                                                PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); phenomale++;
                                            } else {PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[Offspringlinker[Maleoffspring[i][j]]].getRndPheno2() <= cutoff){
                                                PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); phenomale++;
                                            } else {PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"No"});}
                                        }
                                    }
                                } else {
                                    for(int j = 0; j < Maleoffspring[i].size(); j++)
                                    {
                                        PhenoStatuslinker.insert({population[Offspringlinker[Maleoffspring[i][j]]].getID(),"Yes"}); phenomale++;
                                    }
                                }
                            }
                            /*******************/
                            /* Then Do Females */
                            /*******************/
                            for(int i = 0; i < Damlinker.size(); i++)
                            {
                                if(femalenumber < Femaleoffspring[i].size())    /* if number of males larger than number genotyped than randomly select */
                                {
                                    vector < double > randomvalues;
                                    for(int j = 0; j < Femaleoffspring[i].size(); j++)
                                    {
                                        if(trt == 0){randomvalues.push_back(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno1());}
                                        if(trt == 1){randomvalues.push_back(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno2());}
                                    }
                                    double temp; double cutoff;
                                    for(int j = 0; j < (randomvalues.size()-1); j++)   /* Sort */
                                    {
                                        for(int k=j+1; k < randomvalues.size(); k++)
                                        {
                                            if(randomvalues[j] > randomvalues[k]){temp = randomvalues[j]; randomvalues[j] = randomvalues[k]; randomvalues[k] = temp;}
                                        }
                                    }
                                    cutoff = randomvalues[malenumber-1];
                                    for(int j = 0; j < Femaleoffspring[i].size(); j++)
                                    {
                                        if(trt == 0)
                                        {
                                            if(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno1() <= cutoff){
                                                PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); phenofemale++;
                                            } else {PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"No"});}
                                        }
                                        if(trt == 1)
                                        {
                                            if(population[Offspringlinker[Femaleoffspring[i][j]]].getRndPheno2() <= cutoff){
                                                PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); phenofemale++;
                                            } else {PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"No"});}
                                        }
                                    }
                                } else {
                                    for(int j = 0; j < Femaleoffspring[i].size(); j++)
                                    {
                                        PhenoStatuslinker.insert({population[Offspringlinker[Femaleoffspring[i][j]]].getID(),"Yes"}); phenofemale++;
                                    }
                                }
                            }
                            //cout << phenomale << " " << phenofemale << endl;
                        }
                    }
                }
                //cout << PhenoStatuslinker.size() << " " << randev_M.size() << " " << randev_F.size() << endl;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAnimalStage() == "selcandebv" && stage == "postebvcalc")
                    {
                        /***************************************************************************************************/
                        /* If proportion == 1.0 and using pheno_afterselection update when animal is at stage 'selcandebv' */
                        /***************************************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="pheno_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trt]==1.0)
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No"){cout<<"Error Indexing geno-pheno status!!(male + pheno_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No"){cout<<"Error Indexing geno-pheno status!!(male + pheno_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="pheno_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trt]==1.0)
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(female + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No"){cout<<"Error Indexing geno-pheno status!!(female + pheno_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + pheno_afterselection)"<<endl;
                                    exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        /*******************************************************************************/
                        /* If using random_afterselection update when animal is at stage 'postebvcalc' */
                        /*******************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_afterselection" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + random_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No"){cout<<"Error Indexing geno-pheno status!!(male + random_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + random_afterselection)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_afterselection" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(female + random_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No"){cout<<"Error Indexing geno-pheno status!!(female + random_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + random_afterselection)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        /***************************************************************************/
                        /* If using ebv_afterselection update when animal is at stage 'postebvcalc'*/
                        /***************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + ebv_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No"){cout<<"Error Indexing geno-pheno status!!(male + ebv_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No"){cout<<"Error Indexing geno-pheno status!!(male + ebv_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_afterselection" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + pheno_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No"){cout<<"Error Indexing geno-pheno status!!(female + ebv_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No"){cout<<"Error Indexing geno-pheno status!!(female + ebv_afterselection)"<<endl; exit (EXIT_FAILURE);}
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        /*********************************************************************************/
                        /* If using litterrandom_atselection update when animal is at stage 'preebvcalc' */
                        /*********************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + litterrandom_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + litterrandom_afterselection)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + litterrandom_afterselection)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="litterrandom_afterselection" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(female + litterrandom_afterselection)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + litterrandom_afterselection)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + litterrandom_afterselection)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                    }
                    if(population[i].getAnimalStage() == "parent" && stage == "postselcalc" && population[i].getAge() == 1)
                    {
                        /***************************************************************************************************/
                        /* If proportion == 1.0 and using pheno_parents update when animal is at stage 'parent' */
                        /***************************************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="pheno_parents" && (SimParameters.get_MalePropPhenotype_vec())[trt]==1.0)
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + pheno_parents)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + pheno_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + pheno_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="pheno_parents" && (SimParameters.get_FemalePropPhenotype_vec())[trt]==1.0)
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(female + pheno_parents)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + pheno_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + pheno_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        /*******************************************************************************/
                        /* If using random_parents update when animal is at stage 'postselcalc' */
                        /*******************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="random_parents" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + random_parents)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + random_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + random_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="random_parents" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(female + random_parents)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + random_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + random_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        /*******************************************************************************/
                        /* If using random_parents update when animal is at stage 'postselcalc' */
                        /*******************************************************************************/
                        if(population[i].getSex() == 0 && (SimParameters.get_MaleWhoPhenotype_vec())[trt]=="ebv_parents" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(male + ebv_parents)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + ebv_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(male + ebv_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                        if(population[i].getSex() == 1 && (SimParameters.get_FemaleWhoPhenotype_vec())[trt]=="ebv_parents" && PhenoStatuslinker[population[i].getID()] == "Yes")
                        {
                            population[i].update_PhenStatus(trt,"Yes");
                            /* Split off first three and then update correct one with a Yes*/
                            tempupdanim = GenoPhenoStatus[population[i].getID()-1];
                            size_t pos = tempupdanim.find(" ",0); tempnewline = (tempupdanim.substr(0,pos)) + " ";
                            if(atoi((tempupdanim.substr(0,pos).c_str())) != population[i].getID())
                            {
                                cout << "Error Indexing geno-pheno status(female + ebv_parents)!!" << endl; exit (EXIT_FAILURE);
                            }
                            tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            pos = tempupdanim.find(" ",0); tempnewline += (tempupdanim.substr(0,pos)) + " "; tempupdanim.erase(0, pos+1);
                            if(ebvtraits == 1)
                            {
                                pos = tempupdanim.find(" ",0); tempnewlineb = (tempupdanim.substr(0,pos)); tempupdanim.erase(0, pos+1);
                                if(tempnewlineb != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + ebv_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                tempnewline += "Yes "+tempupdanim;
                            }
                            if(ebvtraits == 2)
                            {
                                /* split off each trait then update the correct one based on 'trt' index */
                                vector < string> temptrait;
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                pos = tempupdanim.find(" ",0); temptrait.push_back((tempupdanim.substr(0,pos))); tempupdanim.erase(0, pos+1);
                                //cout << temptrait[0] << " " << temptrait[1] << endl;
                                if(temptrait[trt] != "No")
                                {
                                    cout<<"Error Indexing geno-pheno status!!(female + ebv_parents)"<<endl; exit (EXIT_FAILURE);
                                }
                                temptrait[trt] = "Yes";
                                //cout << temptrait[0] << " " << temptrait[1] << endl << tempnewline << endl << tempupdanim << endl;
                                tempnewline += temptrait[0]+" "+temptrait[1] +" "+tempupdanim;
                                //cout << tempnewline << endl << tempupdanim << endl;
                            }
                            GenoPhenoStatus[population[i].getID()-1] = tempnewline;
                        }
                    }
                }
            }
        }
        /***********************************************************************************/
        /* Update Status of selection candidates to either parents or culled_selcandidates */
        /***********************************************************************************/
        if(stage == "postselcalc")
        {
            madechanges = "YES";
            unordered_map <int, int> Parentslinker; std::unordered_map<int, int>::const_iterator got;  /* Search if key exists */
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getAnimalStage() == "parent"){Parentslinker.insert({population[i].getID(),i});}
            }
            for(int i = 0; i < GenoPhenoStatus.size(); i++)
            {
                size_t pos = GenoPhenoStatus[i].find(" popselcandidate",0);
                if(pos != std::string::npos)        /* Only do something if found */
                {
                    //cout << "'" << GenoPhenoStatus[i] << "'" << "\t";
                    string line = GenoPhenoStatus[i];
                    vector < string > solvervariables(15,"");
                    for(int i = 0; i < 15; i++)
                    {
                        size_t pos = line.find(" ",0);
                        solvervariables[i] = line.substr(0,pos);
                        if(pos != std::string::npos){line.erase(0, pos + 1);}
                        if(pos == std::string::npos){line.clear(); i = 15;}
                    }
                    int start = 0;
                    while(start < solvervariables.size())
                    {
                        if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                        if(solvervariables[start] != ""){start++;}
                    }
                    //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
                    if(solvervariables.size() == 9){
                        /* See if it exists in Parentslinker; */
                        got = Parentslinker.find(atoi(solvervariables[0].c_str()));
                        if(got == Parentslinker.end()){     /* If not found change to 'culled_selcandidate' */
                            solvervariables[4] = "popculled_selcandidate";
                        } else {solvervariables[4] = "popparent";} /* If found change to 'parent' */
                    } else if(solvervariables.size() == 10){
                        /* See if it exists in Parentslinker; */
                        got = Parentslinker.find(atoi(solvervariables[0].c_str()));
                        if(got == Parentslinker.end()){     /* If not found change to 'culled_selcandidate' */
                            solvervariables[5] = "popculled_selcandidate";
                        } else {solvervariables[5] = "popparent";} /* If found change to 'parent' */
                    } else {cout << "Shouldn't be here Yep" << endl; exit (EXIT_FAILURE);}
                    /* replace with new line */
                    std::ostringstream updatedline; updatedline << solvervariables[0];
                    for(int i = 1; i < solvervariables.size(); i++){updatedline << " " << solvervariables[i];}
                    GenoPhenoStatus[i] = updatedline.str();
                    //cout << "'" << GenoPhenoStatus[i] << "'" << endl;
                }
            }
        }
        /****************************************************************/
        /* Update Status of parents to either parents or culled_parents */
        /****************************************************************/
        if(stage == "preebvcalc")
        {
            unordered_map <int, int> Parentslinker; std::unordered_map<int, int>::const_iterator got;  /* Search if key exists */
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getAnimalStage() == "parent"){Parentslinker.insert({population[i].getID(),i});}
            }
            madechanges = "YES";
            for(int i = 0; i < GenoPhenoStatus.size(); i++)
            {
                size_t pos = GenoPhenoStatus[i].find(" popparent",0);
                if(pos != std::string::npos)    /* Only do something if found */
                {
                    //cout << "'" << GenoPhenoStatus[i] << "'" << "\t";
                    string line = GenoPhenoStatus[i];
                    vector < string > solvervariables(15,"");
                    for(int i = 0; i < 15; i++)
                    {
                        size_t pos = line.find(" ",0);
                        solvervariables[i] = line.substr(0,pos);
                        if(pos != std::string::npos){line.erase(0, pos + 1);}
                        if(pos == std::string::npos){line.clear(); i = 15;}
                    }
                    int start = 0;
                    while(start < solvervariables.size())
                    {
                        if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                        if(solvervariables[start] != ""){start++;}
                    }
                    //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
                    if(solvervariables.size() == 9){
                        /* See if it exists in Parentslinker; */
                        got = Parentslinker.find(atoi(solvervariables[0].c_str()));
                        if(got == Parentslinker.end()){     /* If not found change to 'culled_selcandidate' */
                            solvervariables[4] = "popculled_parent";
                        } else {solvervariables[4] = "popparent";} /* If found change to 'parent' */
                    } else if(solvervariables.size() == 10){
                        /* See if it exists in Parentslinker; */
                        got = Parentslinker.find(atoi(solvervariables[0].c_str()));
                        if(got == Parentslinker.end()){     /* If not found change to 'culled_selcandidate' */
                            solvervariables[5] = "popculled_parent";
                        } else {solvervariables[5] = "popparent";} /* If found change to 'parent' */
                    } else {cout << "Shouldn't be here" << endl; exit (EXIT_FAILURE);}
                    /* replace with new line */
                    std::ostringstream updatedline; updatedline << solvervariables[0];
                    for(int i = 1; i < solvervariables.size(); i++){updatedline << " " << solvervariables[i];}
                    GenoPhenoStatus[i] = updatedline.str();
                    //cout << "'" << GenoPhenoStatus[i] << "'" << endl;
                }
            }
        }
        if(madechanges == "YES")
        {
            /* First remove contents from old one then update with new genotype or phenotype status */
            fstream cleargenostatus; cleargenostatus.open(OUTPUTFILES.getloc_GenotypeStatus().c_str(), std::fstream::out | std::fstream::trunc);
            /* output updated pheno_geno status */
            stringstream outputstringgenostatus(stringstream::out);
            for(int i = 0; i < GenoPhenoStatus.size(); i++){outputstringgenostatus << GenoPhenoStatus[i] << endl;}
            std::ofstream outputstatus(OUTPUTFILES.getloc_GenotypeStatus().c_str(), std::ios_base::app | std::ios_base::out);
            outputstatus << outputstringgenostatus.str(); outputstringgenostatus.str(""); outputstringgenostatus.clear();
        }
    }
    
    
}
///////////////////////////////////
// Calculate Generation Interval //
///////////////////////////////////
void GenerateGenerationInterval(vector <Animal> &population,globalpopvar &Population1,parameters &SimParameters,int Gen, ostream& logfileloc)
{
    double L_f = 0.0; double L_m = 0.0;
    /* loop through all progeny and grab sire and dam */
    vector < int > sire; vector < int > dam;
    for(int i =0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1){sire.push_back(population[i].getSire()); dam.push_back(population[i].getDam());}
    }
    /* remove duplicates */
    sort(sire.begin(),sire.end() ); sire.erase(unique(sire.begin(),sire.end()),sire.end());
    sort(dam.begin(),dam.end() ); dam.erase(unique(dam.begin(),dam.end()),dam.end());
    for(int i = 0; i < sire.size(); i++)
    {
        int search = 0;
        while(1)
        {
            if(sire[i] == population[search].getID()){L_m += (population[search].getAge()-1); break;}
            if(sire[i] != population[search].getID()){search++;}
            if(search >= population.size()){cout << "Shouldn't be here!"; exit (EXIT_FAILURE);}
        }
    }
    L_m /= double(sire.size());
    for(int i = 0; i < dam.size(); i++)
    {
        int search = 0;
        while(1)
        {
            if(dam[i] == population[search].getID()){L_f += (population[search].getAge()-1); break;}
            if(dam[i] != population[search].getID()){search++;}
            if(search >= population.size()){cout << "Shouldn't be here!"; exit (EXIT_FAILURE);}
        }
    }
    L_f /= double(dam.size());
    for(int k = 0; k < SimParameters.getnumbertraits(); k++)
    {
        Population1.update_GenInterval_Males(Gen,k,L_m);
        Population1.update_GenInterval_Females(Gen,k,L_f);
    }
    //for(int i = 0; i < Population1.getrow_GenInterval_Males(); i++)
    //{
    //    for(int j = 0; j < Population1.getcol_GenInterval_Males(); j++)
    //    {
    //        cout << i << ": " << Population1.get_GenInterval_Males(i,j) << " " << Population1.get_GenInterval_Females(i,j) << " - ";
    //    }
    //    cout << endl;
    //}
    logfileloc << "   Generation Intervals: Males(" <<L_m<<"); Females("<<L_f<<")." << endl << endl;
}


/************************************************************************/
/************************************************************************/
/*    QTL_new_old: information regarding the QTL across generations     */
/************************************************************************/
/************************************************************************/
// constructors
QTL_new_old::QTL_new_old()  /* QTL_new_old Class */
{
    Location = 0.0; Additivevect = std::vector<double>(0); Dominancevect = std::vector<double>(0);
    AdditiveEffect = 0.0; DominanceEffect = 0.0; Type = "0"; GenOccured = 99; Freq = "0"; LDDecay = "";
}
QTL_new_old::QTL_new_old(double location,std::vector<double> vadd, std::vector<double> vdom, double add, double dom, std::string type, int genOccured, std::string freq, std::string lddec)
{
    Location = location; Additivevect = vadd; Dominancevect = vdom;
    AdditiveEffect = add; DominanceEffect = dom; Type = type;
    GenOccured = genOccured; Freq = freq; LDDecay = lddec;
}
// destructor
QTL_new_old::~QTL_new_old(){}           /* QTL_new_old Class */
// Start of Functions for QTL class
void QTL_new_old::UpdateFreq(std::string currentFreq){Freq = Freq + "_" + currentFreq;}
void QTL_new_old::UpdateLDDecay(std::string currentLDDecaya){LDDecay = LDDecay + "_" + currentLDDecaya;}
void QTL_new_old::FounderLDDecay(std::string currentLDDecay){LDDecay = currentLDDecay;}
void QTL_new_old::update_Additivevect(int i, double x){Additivevect[i] = x;}
void QTL_new_old::update_Dominancevect(int i, double x){Dominancevect[i] = x;}


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


