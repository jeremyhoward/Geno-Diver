#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <random>
#include <algorithm>
#include <cstring>
#include <set>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/LU>
#include <mkl.h>
#include "zfstream.h"
#include "HaplofinderClasses.h"
#include "Animal.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"
#include "OutputFiles.h"
#include "Global_Population.h"

/**********************************************/
/* Functions from Genome_ROH.cpp              */
/**********************************************/
void Genome_ROH_Summary(parameters &SimParameters, outputfiles &OUTPUTFILES, vector <Animal> &population,int Gen, ostream& logfileloc);
void Proportion_ROH(parameters &SimParameters, vector <Animal> &population,outputfiles &OUTPUTFILES, ostream& logfileloc);
void ld_decay_estimator(outputfiles &OUTPUTFILES, vector <Animal> &population, string lineone);
void qtlld_decay_estimator(parameters &SimParameters, vector <Animal> &population, vector <QTL_new_old> &population_QTL,outputfiles &OUTPUTFILES,string foundergen);
void WindowVariance(parameters &SimParameters,vector <Animal> &population,vector < QTL_new_old > &population_QTL,string foundergen ,outputfiles &OUTPUTFILES);

/**********************************************/
/* Functions from HaplofinderClasses.cpp      */
/**********************************************/
void calculate_IIL(vector <double> &inbreedingload, vector <string> const &progenygenotype,vector <Unfavorable_Regions> &trainregions,string unfav_direc);
void calculatecorrelation(vector <double> &inbreedingload, vector <double> &progenyphenotype,vector <double> &progenytgv, vector <double> &progenytbv, vector <double> &progenytdd, vector < double > &outcorrelations);


using namespace std;
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Class Functions       ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
outputfiles::outputfiles(){}
outputfiles::~outputfiles(){}
void outputfiles::Updateloc_lowfitnesspath(std::string temp){loc_lowfitnesspath=temp;}
void outputfiles::Updateloc_snpfreqfile(std::string temp){loc_snpfreqfile=temp;}
void outputfiles::Updateloc_foundergenofile(std::string temp){loc_foundergenofile=temp;}
void outputfiles::Updateloc_qtl_class_object(std::string temp){loc_qtl_class_object=temp;}
void outputfiles::Updateloc_Pheno_Pedigreet(std::string temp){loc_Pheno_Pedigree=temp;}
void outputfiles::Updateloc_Pheno_GMatrix(std::string temp){loc_Pheno_GMatrix=temp;}
void outputfiles::Updateloc_Pheno_GMatrixImp(std::string temp){loc_Pheno_GMatrixImp=temp;}
void outputfiles::Updateloc_Master_DF(std::string temp){loc_Master_DF=temp;}
void outputfiles::Updateloc_Master_Genotype(std::string temp){loc_Master_Genotype=temp;}
void outputfiles::Updateloc_Master_Genotype_zip(std::string temp){loc_Master_Genotype_zip=temp;}
void outputfiles::Updateloc_BinaryG_Matrix(std::string temp){loc_BinaryG_Matrix=temp;}
void outputfiles::Updateloc_Binarym_Matrix(std::string temp){loc_Binarym_Matrix=temp;}
void outputfiles::Updateloc_Binaryp_Matrix(std::string temp){loc_Binaryp_Matrix=temp;}
void outputfiles::Updateloc_BinaryLinv_Matrix(std::string temp){loc_BinaryLinv_Matrix=temp;}
void outputfiles::Updateloc_BinaryGinv_Matrix(std::string temp){loc_BinaryGinv_Matrix=temp;}
void outputfiles::Updateloc_Marker_Map(std::string temp){loc_Marker_Map=temp;}
void outputfiles::Updateloc_Master_DataFrame(std::string temp){loc_Master_DataFrame=temp;}
void outputfiles::Updateloc_Summary_QTL(std::string temp){loc_Summary_QTL=temp;}
void outputfiles::Updateloc_Summary_DF(std::string temp){loc_Summary_DF=temp;}
void outputfiles::Updateloc_GenotypeStatus(std::string temp){loc_GenotypeStatus=temp;}
void outputfiles::Updateloc_ExpRealDeltaG(std::string temp){loc_ExpRealDeltaG=temp;}
void outputfiles::Updateloc_LD_Decay(std::string temp){loc_LD_Decay=temp;}
void outputfiles::Updateloc_QTL_LD_Decay(std::string temp){loc_QTL_LD_Decay=temp;}
void outputfiles::Updateloc_Phase_Persistance(std::string temp){loc_Phase_Persistance=temp;}
void outputfiles::Updateloc_Phase_Persistance_Outfile(std::string temp){loc_Phase_Persistance_Outfile=temp;}
void outputfiles::Updateloc_Summary_Haplofinder(std::string temp){loc_Summary_Haplofinder=temp;}
void outputfiles::Updateloc_Summary_ROHGenome_Freq(std::string temp){loc_Summary_ROHGenome_Freq=temp;}
void outputfiles::Updateloc_Summary_ROHGenome_Length(std::string temp){loc_Summary_ROHGenome_Length=temp;}
void outputfiles::Updateloc_Bayes_MCMC_Samples(std::string temp){loc_Bayes_MCMC_Samples=temp;}
void outputfiles::Updateloc_Bayes_PosteriorMeans(std::string temp){loc_Bayes_PosteriorMeans=temp;}
void outputfiles::Updateloc_Amax_Output(std::string temp){loc_Amax_Output=temp;}
void outputfiles::Updateloc_TraitReference_Output(std::string temp){loc_TraitReference_Output=temp;}
void outputfiles::Updateloc_Windowadditive_Output(std::string temp){loc_Windowadditive_Output=temp;}
void outputfiles::Updateloc_Windowdominance_Output(std::string temp){loc_Windowdominance_Output=temp;}


////////////////////////////////////////////////////////////////////////////////////
///////////        Read in Parameter File and Fill Parameter Class       ///////////
////////////////////////////////////////////////////////////////////////////////////
void GenerateOutputFiles(parameters &SimParameters, outputfiles &OUTPUTFILES,string path)
{
    OUTPUTFILES.Updateloc_lowfitnesspath(path + "/" + SimParameters.getOutputFold() + "/Low_Fitness");
    OUTPUTFILES.Updateloc_snpfreqfile(path + "/" + SimParameters.getOutputFold() + "/SNPFreq");
    OUTPUTFILES.Updateloc_foundergenofile(path + "/" + SimParameters.getOutputFold() + "/FounderGenotypes");
    OUTPUTFILES.Updateloc_qtl_class_object(path + "/" + SimParameters.getOutputFold() + "/QTL_new_old_Class");
    OUTPUTFILES.Updateloc_Pheno_Pedigreet(path + "/" + SimParameters.getOutputFold() + "/Pheno_Pedigree");
    OUTPUTFILES.Updateloc_Pheno_GMatrix(path + "/" + SimParameters.getOutputFold() + "/Pheno_GMatrix");
    OUTPUTFILES.Updateloc_Pheno_GMatrixImp(path + "/" + SimParameters.getOutputFold() + "/Pheno_GMatrixImputed");
    OUTPUTFILES.Updateloc_Master_DF(path + "/" + SimParameters.getOutputFold() + "/Master_DF");
    OUTPUTFILES.Updateloc_Master_Genotype(path + "/" + SimParameters.getOutputFold() + "/Master_Genotypes");
    OUTPUTFILES.Updateloc_Master_Genotype_zip(path + "/" + SimParameters.getOutputFold() + "/Master_Genotypes.gz");
    OUTPUTFILES.Updateloc_BinaryG_Matrix(path + "/" + SimParameters.getOutputFold() + "/G_Matrix");
    OUTPUTFILES.Updateloc_Binarym_Matrix(path + "/" + SimParameters.getOutputFold() + "/m_Matrix");
    OUTPUTFILES.Updateloc_Binaryp_Matrix(path + "/" + SimParameters.getOutputFold() + "/p_Matrix");
    OUTPUTFILES.Updateloc_BinaryLinv_Matrix(path + "/" + SimParameters.getOutputFold() + "/Linv_Matrix");
    OUTPUTFILES.Updateloc_BinaryGinv_Matrix(path + "/" + SimParameters.getOutputFold() + "/Ginv_Matrix");
    OUTPUTFILES.Updateloc_Marker_Map(path + "/" + SimParameters.getOutputFold() + "/Marker_Map");
    OUTPUTFILES.Updateloc_Master_DataFrame(path + "/" + SimParameters.getOutputFold() + "/Master_DataFrame");
    OUTPUTFILES.Updateloc_Summary_QTL(path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_QTL");
    OUTPUTFILES.Updateloc_Summary_DF(path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_DataFrame");
    OUTPUTFILES.Updateloc_GenotypeStatus(path + "/" + SimParameters.getOutputFold() + "/Animal_GenoPheno_Status");
    OUTPUTFILES.Updateloc_ExpRealDeltaG(path + "/" + SimParameters.getOutputFold() + "/ExpRealDeltaG");
    OUTPUTFILES.Updateloc_LD_Decay(path + "/" + SimParameters.getOutputFold() + "/LD_Decay");
    OUTPUTFILES.Updateloc_QTL_LD_Decay(path + "/" + SimParameters.getOutputFold() + "/QTL_LD_Decay");
    OUTPUTFILES.Updateloc_Phase_Persistance(path + "/" + SimParameters.getOutputFold() + "/Phase_Persistance");
    OUTPUTFILES.Updateloc_Phase_Persistance_Outfile(path + "/" + SimParameters.getOutputFold() + "/Phase_Persistance_Generation");
    OUTPUTFILES.Updateloc_Summary_Haplofinder(path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_Haplofinder");
    OUTPUTFILES.Updateloc_Summary_ROHGenome_Freq(path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_ROH_Freq");
    OUTPUTFILES.Updateloc_Summary_ROHGenome_Length(path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_ROH_Length");
    OUTPUTFILES.Updateloc_Bayes_MCMC_Samples(path + "/" + SimParameters.getOutputFold() + "/bayes_mcmc_samples");
    OUTPUTFILES.Updateloc_Bayes_PosteriorMeans(path + "/" + SimParameters.getOutputFold() + "/bayes_posteriormeans");
    OUTPUTFILES.Updateloc_Amax_Output(path + "/" + SimParameters.getOutputFold() + "/AmaxGeneration");
    OUTPUTFILES.Updateloc_TraitReference_Output(path + "/" + SimParameters.getOutputFold() + "/TrainReference");
    OUTPUTFILES.Updateloc_Windowadditive_Output(path + "/" + SimParameters.getOutputFold() + "/WindowAdditiveVariance");
    OUTPUTFILES.Updateloc_Windowdominance_Output(path + "/" + SimParameters.getOutputFold() + "/WindowDominanceVariance");
   
//    cout << OUTPUTFILES.getloc_lowfitnesspath() << endl;
//    cout << OUTPUTFILES.getloc_snpfreqfile() << endl;
//    cout << OUTPUTFILES.getloc_foundergenofile() << endl;
//    cout << OUTPUTFILES.getloc_qtl_class_object() << endl;
//    cout << OUTPUTFILES.getloc_Pheno_Pedigree() << endl;
//    cout << OUTPUTFILES.getloc_Pheno_GMatrix() << endl;
//    cout << OUTPUTFILES.getloc_Master_DF() << endl;
//    cout << OUTPUTFILES.getloc_Master_Genotype() << endl;
//    cout << OUTPUTFILES.getloc_BinaryG_Matrix() << endl;
//    cout << OUTPUTFILES.getloc_Binarym_Matrix() << endl;
//    cout << OUTPUTFILES.getloc_Binaryp_Matrix() << endl;
//    cout << OUTPUTFILES.getloc_BinaryLinv_Matrix() << endl;
//    cout << OUTPUTFILES.getloc_BinaryGinv_Matrix() << endl;
//    cout << OUTPUTFILES.getloc_Marker_Map() << endl;
//    cout << OUTPUTFILES.getloc_Master_DataFrame() << endl;
//    cout << OUTPUTFILES.getloc_Summary_QTL() << endl;
//    cout << OUTPUTFILES.getloc_Summary_DF() << endl;
//    cout << OUTPUTFILES.getloc_GenotypeStatus() << endl;
//    cout << OUTPUTFILES.getloc_LD_Decay() << endl;
//    cout << OUTPUTFILES.getloc_QTL_LD_Decay() << endl;
//    cout << OUTPUTFILES.getloc_Phase_Persistance() << endl;
//    cout << OUTPUTFILES.getloc_Phase_Persistance_Outfile() << endl;
//    cout << OUTPUTFILES.getloc_Summary_Haplofinder() << endl;
//    cout << OUTPUTFILES.getloc_Summary_ROHGenome_Freq() << endl;
//    cout << OUTPUTFILES.getloc_Summary_ROHGenome_Length() << endl;
//    cout << OUTPUTFILES.getloc_Bayes_MCMC_Samples() << endl;
//    cout << OUTPUTFILES.getloc_Bayes_PosteriorMeans() << endl;
//    cout << OUTPUTFILES.getloc_Amax_Output() << endl;
//    cout << OUTPUTFILES.getloc_Correlation_Output() << endl;
//    cout << OUTPUTFILES.getloc_Windowadditive_Output() << endl;
//    cout << OUTPUTFILES.getloc_Windowdominance_Output() << endl;
//    cout << OUTPUTFILES.getloc_ExpRealDeltaG() << endl;
    /* Ensure removes files that aren't always created so wont' confuse users */
    string command = "rm -rf " + OUTPUTFILES.getloc_LD_Decay() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_QTL_LD_Decay() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Phase_Persistance() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Phase_Persistance_Outfile() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Summary_Haplofinder() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Summary_ROHGenome_Freq() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Summary_ROHGenome_Length() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Bayes_MCMC_Samples() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Bayes_PosteriorMeans() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Amax_Output() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_TraitReference_Output() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Windowadditive_Output() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Windowdominance_Output() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_GenotypeStatus() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_ExpRealDeltaG() + " || true"; system(command.c_str());
}
////////////////////////////////////////////////////////////////////////////////////
///////////             Delete Previous Simulation Files                 ///////////
////////////////////////////////////////////////////////////////////////////////////
void DeletePreviousSimulation(outputfiles &OUTPUTFILES,parameters &SimParameters,string logfileloc)
{
    fstream checkmarkermap;
    checkmarkermap.open(OUTPUTFILES.getloc_Marker_Map().c_str(), std::fstream::out | std::fstream::trunc);
    checkmarkermap.close();
    fstream checkphenoped;
    checkphenoped.open(OUTPUTFILES.getloc_Pheno_Pedigree().c_str(), std::fstream::out | std::fstream::trunc);
    checkphenoped.close();
    fstream checkphenogen;
    checkphenogen.open(OUTPUTFILES.getloc_Pheno_GMatrix(), std::fstream::out | std::fstream::trunc);
    checkphenogen.close();
    fstream checkmasterdf;
    checkmasterdf.open(OUTPUTFILES.getloc_Master_DF().c_str(), std::fstream::out | std::fstream::trunc);
    checkmasterdf.close();
    string command = "rm -rf " + OUTPUTFILES.getloc_Master_Genotype() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Master_Genotype_zip() + " || true"; system(command.c_str());
    fstream checkmasterdataframe;
    checkmasterdataframe.open(OUTPUTFILES.getloc_Master_DataFrame().c_str(), std::fstream::out | std::fstream::trunc);
    checkmasterdataframe.close();
    fstream checklog; checklog.open(logfileloc, std::fstream::out | std::fstream::trunc); checklog.close(); /* Deletes log file Previous Simulation */
    fstream checklowfitness;
    checklowfitness.open(OUTPUTFILES.getloc_lowfitnesspath().c_str(), std::fstream::out | std::fstream::trunc);
    checklowfitness.close();
    /* add first line as column ID's */
    std::ofstream outputlow(OUTPUTFILES.getloc_lowfitnesspath().c_str(), std::ios_base::app | std::ios_base::out);
    if(SimParameters.getnumbertraits() == 1)
    {
        outputlow << "Sire Dam Gen Ped_F Gen_F Hap3_F Homozy Homolethal Heterlethal Homosublethal Hetersublethal Letequiv Fitness QTL_Fitness";
        outputlow << " TGV TBV TDD" << endl;
    }
    if(SimParameters.getnumbertraits() == 2)
    {
        outputlow << "Sire Dam Gen Ped_F Gen_F Hap3_F Homozy Homolethal Heterlethal Homosublethal Hetersublethal Letequiv Fitness QTL_Fitness";
        outputlow << " TGV1 TBV1 TDD1 TGV2 TBV2 TDD2" << endl;
    }
}
////////////////////////////////////////////////////////////////////////////////////
///////////      Generate Summary Statistics As Simulation Progresses    ///////////
////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////
// Calculate Expected Heterozygosity //
///////////////////////////////////////
void ExpectedHeterozygosity(vector <Animal> &population, globalpopvar &Population1, int Gen,ostream& logfileloc)
{
    double expectedhet = 0.0; vector < string > population_marker;
    for(int i = 0; i < population.size(); i++){population_marker.push_back(population[i].getMarker());}
    vector < double > tempfreqexphet(population_marker[0].size(),0.0);
    for(int i = 0; i < population_marker.size(); i++)
    {
        string geno = population_marker[i];
        for(int j = 0; j < geno.size(); j++)
        {
            int temp = geno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            tempfreqexphet[j] += temp;
        }
    }
    for(int i = 0; i < tempfreqexphet.size(); i++){tempfreqexphet[i] = tempfreqexphet[i] / (2 * population_marker.size());}
    for(int i = 0; i < tempfreqexphet.size(); i++)
    {
        expectedhet += (1 - ((tempfreqexphet[i]*tempfreqexphet[i]) + ((1-tempfreqexphet[i])*(1-tempfreqexphet[i]))));
    }
    expectedhet /= double(tempfreqexphet.size());
    Population1.update_expectedheter(Gen,expectedhet);
    logfileloc << "   Expected Heterozygosity in progeny calculated: " << (Population1.get_expectedheter())[Gen] << "." << endl;
}
///////////////////////////////////////
// Update Frequency and Va and Vd    //
///////////////////////////////////////
void UpdateFrequency_GenVar(parameters &SimParameters, vector <Animal> &population, globalpopvar &Population1, int Gen, vector < QTL_new_old > &population_QTL,outputfiles &OUTPUTFILES,ostream& logfileloc)
{
    vector < string > genfreqgeno;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            genfreqgeno.push_back(population[i].getQTL());
        }
    }
    vector < double > tempgenfreq(genfreqgeno[0].size(),0.0);
    for(int i = 0; i < genfreqgeno.size(); i++)
    {
        string geno = genfreqgeno[i];
        for(int j = 0; j < geno.size(); j++)
        {
            int temp = geno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            tempgenfreq[j] += temp;
        }
    }
    for(int i = 0; i < tempgenfreq.size(); i++){tempgenfreq[i] = tempgenfreq[i] / (2 * genfreqgeno.size());}
    /* Calculate additive and dominance variance based on allele frequencies in the current generation */
    double tempva = 0.0; double tempvd = 0.0;
    for(int i = 0; i < tempgenfreq.size(); i++)
    {
        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
        {
            tempva += 2*tempgenfreq[i]*(1-tempgenfreq[i])*((Population1.get_qtl_add_quan(i,0)+(Population1.get_qtl_dom_quan(i,0)*((1-tempgenfreq[i])-tempgenfreq[i])))*(Population1.get_qtl_add_quan(i,0)+(Population1.get_qtl_dom_quan(i,0)*((1-tempgenfreq[i])-tempgenfreq[i]))));
            tempvd += (2*tempgenfreq[i]*(1-tempgenfreq[i])*Population1.get_qtl_dom_quan(i,0)) * (2*tempgenfreq[i]*(1-tempgenfreq[i])*Population1.get_qtl_dom_quan(i,0));
        }
    }
    Population1.update_qtl_additivevar(Gen,0,tempva);
    Population1.update_qtl_dominancevar(Gen,0,tempvd);
    /* update frequency */
    for(int j = 0; j < tempgenfreq.size(); j++)                /* Loops across old QTL until map position matches up */
    {
        string currentType;
        if((Population1.get_qtl_type())[j] == 2)
        {
            currentType = "2"; int i = 0;
            while(1)
            {
                double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                if(previousLoc == (Population1.get_qtl_mapposition())[j] && previousType == currentType)
                {
                    stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                    string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                }
                i++;
            }
        }
        if((Population1.get_qtl_type())[j] == 4)
        {
            currentType = "4"; int i = 0;
            while(1)
            {
                double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                if(previousLoc == (Population1.get_qtl_mapposition())[j] && previousType == currentType)
                {
                    stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                    string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                }
                i++;
            }
        }
        if((Population1.get_qtl_type())[j] == 5)
        {
            currentType = "5"; int i = 0;
            while(1)
            {
                double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                if(previousLoc == (Population1.get_qtl_mapposition())[j] && previousType == currentType)
                {
                    stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                    string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                }
                i++;
            }
        }
        if((Population1.get_qtl_type())[j] == 3)
        {
            for(int rep= 0; rep < 2; rep++)
            {
                if(rep == 0){currentType = "2";}
                if(rep == 1){currentType = "5";}
                int i = 0;
                while(1)
                {
                    double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                    string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                    if(previousLoc == (Population1.get_qtl_mapposition())[j] && previousType == currentType)
                    {
                        stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                        string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                    }
                    i++;
                }
            }
        }
    }
    logfileloc << "   QTL frequency updated." << endl;
    /* Add New Mutations to population_QTL; Save to a file that will copy over old one */
    ofstream output10;
    output10.open (OUTPUTFILES.getloc_qtl_class_object().c_str());
    if(SimParameters.getnumbertraits() == 1){output10 << "Location Additive_Selective Dominance Type Gen Freq" << endl;}
    if(SimParameters.getnumbertraits() == 2){output10 << "Location Additive_Selective1 Dominance1 Additive_Selective2 Dominance2 Type Gen Freq" << endl;}
    for(int i = 0; i < population_QTL.size(); i++)
    {
        output10 << population_QTL[i].getLocation() << " ";
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            output10 << (population_QTL[i].get_Additivevect())[j] << " " << (population_QTL[i].get_Dominancevect())[j] << " ";
        }
        output10 << population_QTL[i].getType() << " ";
        output10 << population_QTL[i].getGenOccured() << " " << population_QTL[i].getFreq() << endl;
    }
    output10.close();
}
/*****************************/
/* LD Option Parameter Given */
/*****************************/
void LD_Option(parameters &SimParameters, vector <Animal> &population, vector <QTL_new_old> &population_QTL,outputfiles &OUTPUTFILES,string foundergen,ostream& logfileloc)
{
    time_t intbegin_time = time(0);
    if(foundergen == "no")
    {
        logfileloc << endl << "   Generate Genome Summary Statistics: " << endl;
    }
    if(foundergen == "yes")     /* Clear previous simulation */
    {
        logfileloc << "Generate Genome Summary Statistics: " << endl;
        fstream checkldfile; checkldfile.open(OUTPUTFILES.getloc_LD_Decay().c_str(), std::fstream::out | std::fstream::trunc); checkldfile.close();
        fstream ckqtlldfile; ckqtlldfile.open(OUTPUTFILES.getloc_QTL_LD_Decay().c_str(), std::fstream::out | std::fstream::trunc); ckqtlldfile.close();
        ofstream outputphase;
        outputphase.open(OUTPUTFILES.getloc_QTL_LD_Decay().c_str());
        outputphase << "Generation PhaseCorrelation...." << endl;
        outputphase.close();
    }
    /* Generate genome-wide LD */
    ld_decay_estimator(OUTPUTFILES,population,foundergen);
    qtlld_decay_estimator(SimParameters,population,population_QTL,OUTPUTFILES,foundergen);
    time_t intend_time = time(0);
    if(foundergen == "yes")     /* Clear previous simulation */
    {
        logfileloc << "    - Genome-wide marker LD decay." << endl;
        logfileloc << "    - QTL LD decay and Phase Persistance." << endl;
        logfileloc << "Finished Generating Genome Summary Statistics (Time: " << difftime(intend_time,intbegin_time) << " seconds)."<< endl << endl;
    }
    if(foundergen == "no")
    {
        logfileloc << "      Genome-wide marker LD decay." << endl;
        logfileloc << "      QTL LD decay and Phase Persistance." << endl;
        logfileloc << "   Finished Generating Genome Summary Statistics (Time: " << difftime(intend_time,intbegin_time) << " seconds)."<< endl << endl;
    }
}
/*******************************/
/* Haplofinder Parameter Given */
/*******************************/
void Haplofinder_Option(parameters &SimParameters, vector <Animal> &population,vector <Unfavorable_Regions> &trainregions,string unfav_direc,int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc)
{
    if(trainregions.size() > 0)
    {
        vector <double> progenyphenotype; vector <double> progenytgv; vector <double> progenytbv; vector <double> progenytdd;
        vector <string> progenygenotype; vector <double> inbreedingload; vector <double> outcorrelations(4,0);
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getAge() == 1)
            {
                progenyphenotype.push_back((population[i].get_Phenvect())[0]); progenytgv.push_back((population[i].get_GVvect())[0]);
                progenytbv.push_back((population[i].get_BVvect())[0]); progenytdd.push_back((population[i].get_DDvect())[0]);
                progenygenotype.push_back(population[i].getMarker()); inbreedingload.push_back(0.0);
            }
        }
        calculate_IIL(inbreedingload,progenygenotype,trainregions,unfav_direc);     /* Calculate inbreeding load based on training regions */
        calculatecorrelation(inbreedingload,progenyphenotype,progenytgv,progenytbv,progenytdd,outcorrelations); /* Calculate Correlation */
        logfileloc << "   Correlation between phenotype and haplofinder inbreeding load " << outcorrelations[0] << "." << endl << endl;
        std::ofstream outsummaryhap(OUTPUTFILES.getloc_Summary_Haplofinder().c_str(), std::ios_base::app | std::ios_base::out);
        outsummaryhap<<Gen<<" "<<outcorrelations[0]<<" "<<outcorrelations[1]<<" "<<outcorrelations[2]<<" "<<outcorrelations[3]<<endl;
    }
}
/*******************************/
/* ROH Parameter Given         */
/*******************************/
void ROH_Option(parameters &SimParameters,vector <Animal> &population, int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc)
{
    for(int i = 0; i < (SimParameters.get_rohgeneration()).size(); i++)
    {
        if(Gen == (SimParameters.get_rohgeneration())[i])
        {
            time_t start_roh = time(0);
            logfileloc << "   Generated summary statistics of ROH levels across the genome.";
            Genome_ROH_Summary(SimParameters,OUTPUTFILES,population,Gen,logfileloc);
            time_t end_roh = time(0);
            logfileloc << " (Time: " << difftime(end_roh,start_roh) << " seconds)." << endl << endl;
        }
    }
}
///////////////////////////////////////
// Output pedigree and genomic files //
///////////////////////////////////////
int OutputPedigree_GenomicEBV(outputfiles &OUTPUTFILES, vector <Animal> &population,int TotalAnimalNumber, parameters &SimParameters)
{
    int tempanimnum = TotalAnimalNumber;
    /* Output animals that are of age 1 into pheno_pedigree and Pheno_Gmatrix to use for relationships. That way when you read them back in */
    /* to create relationship matrix don't need to order them.  Save as a continuous string and then output */
    stringstream outputstringpedigree(stringstream::out);
    stringstream outputstringgenomic(stringstream::out); int outputnumpedgen = 0;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            /* For pedigree */
            outputstringpedigree << population[i].getID() << " " << population[i].getSire() << " " << population[i].getDam() << " ";
            if(SimParameters.getSelection() == "random" || SimParameters.getSelection() == "phenotype" || SimParameters.getSelection() == "tbv" || SimParameters.getSelection() == "ebv")
            {
                outputstringpedigree << (population[i].get_Phenvect())[0] << endl;
            }
            if(SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "index_tbv")
            {
                outputstringpedigree << (population[i].get_Phenvect())[0] << " " << (population[i].get_Phenvect())[1] << endl;
            }
            /* For Genomic */
            outputstringgenomic << population[i].getID() << " ";
            if(SimParameters.getSelection() == "random" || SimParameters.getSelection() == "phenotype" || SimParameters.getSelection() == "tbv" || SimParameters.getSelection() == "ebv")
            {
                outputstringgenomic << (population[i].get_Phenvect())[0] << " ";
            }
            if(SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "index_tbv")
            {
                outputstringgenomic << (population[i].get_Phenvect())[0] << " " << (population[i].get_Phenvect())[1] << " ";
            }
            outputstringgenomic <<population[i].getMarker()<<" "<<population[i].getPatHapl()<<" "<<population[i].getMatHapl()<<endl;
            outputnumpedgen++; tempanimnum++;                                /* to keep track of number of animals */
        }
        if(outputnumpedgen % 100 == 0)
        {
            /* Don't need to worry about pedigree file getting big, but output pheno_genomic file */
            std::ofstream output2(OUTPUTFILES.getloc_Pheno_GMatrix().c_str(), std::ios_base::app | std::ios_base::out);
            output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
        }
    }
    /* output pheno pedigree file */
    std::ofstream output1(OUTPUTFILES.getloc_Pheno_Pedigree().c_str(), std::ios_base::app | std::ios_base::out);
    output1 << outputstringpedigree.str(); outputstringpedigree.str(""); outputstringpedigree.clear();
    /* output pheno genomic file */
    std::ofstream output2(OUTPUTFILES.getloc_Pheno_GMatrix().c_str(), std::ios_base::app | std::ios_base::out);
    output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
    return  tempanimnum;
}

////////////////////////////////////////////////////////////////////////////////////
///////////                    Generate Output Files                     ///////////
////////////////////////////////////////////////////////////////////////////////////
/***********************************/
/* Generate Master_DataFrame File  */
/***********************************/
void GenerateMaster_DataFrame(parameters &SimParameters, outputfiles &OUTPUTFILES,globalpopvar &Population1, vector<double> &estimatedsolutions,vector<double> &trueaccuracy,vector < int > &ID_Gen)
{
    vector <int> id_falconer;
    vector< vector <double> > tbv_falconer;
    vector< vector <double> > tdd_falconer;
    if(SimParameters.getQuantParam() == "statistical")
    {
        //cout << (Population1.get_QTLFreq_AcrossGen()).size() << endl;
        //for(int i = 0; i < (Population1.get_QTLFreq_AcrossGen()).size(); i++)
        //{
        //    if((Population1.get_QTLFreq_AcrossGen())[i] != -5.0){cout << (Population1.get_QTLFreq_AcrossGen())[i] << " ";}
        //}
        //cout << endl << endl;
        //cout << Population1.getQTLFreq_Number() << endl;
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
        
        //for(int i = 0; i < alpha.size(); i++)
        //{
        //    if((Population1.get_QTLFreq_AcrossGen())[i] != -5.0)
        //    {
        //        for(int j = 0; j < alpha[i].size(); j++){cout << alpha[i][j] << " ";}
        //        cout << endl;
        //    }
        //}
        /* Figure out max length of buffer */
        int figureoutmaxlength = 1000;
        for(int i = 0; i < (SimParameters.get_Marker_chr()).size(); i++){figureoutmaxlength += (SimParameters.get_Marker_chr())[i];}
        for(int i = 0; i < (SimParameters.get_QTL_chr()).size(); i++){figureoutmaxlength += (SimParameters.get_QTL_chr())[i];}
        for(int i = 0; i < (SimParameters.get_FTL_lethal_chr()).size(); i++){figureoutmaxlength += (SimParameters.get_FTL_lethal_chr())[i];}
        for(int i = 0; i < (SimParameters.get_FTL_sublethal_chr()).size(); i++){figureoutmaxlength += (SimParameters.get_FTL_sublethal_chr())[i];}
        /* Read in zipped genotype file and calculate tbv_falc and tdd_falc */
        const int MAX_LINE_LENGTH = figureoutmaxlength;
        char buf[MAX_LINE_LENGTH];
        int linenumber = 0; string line;  double temptbv, temptdd;
        gzifstream zippedgeno;
        zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str());
        if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
        while(1)
        {
            zippedgeno.getline(buf, MAX_LINE_LENGTH, '\n');
            if(zippedgeno.fail() || !zippedgeno.good()) break;
            if(linenumber > 0)
            {
                stringstream ss(buf);
                string line = ss.str();
                vector < string > solvervariables(10,"");
                for(int i = 0; i < 10; i++)
                {
                    size_t pos = line.find(" ",0);
                    solvervariables[i] = line.substr(0,pos);
                    if(pos != std::string::npos){line.erase(0, pos + 1);}
                    if(pos == std::string::npos){line.clear(); i = 10;}
                }
                int start = 0;
                while(start < solvervariables.size())
                {
                    if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                    if(solvervariables[start] != ""){start++;}
                }
                //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
                id_falconer.push_back(atoi(solvervariables[0].c_str()));
                string geno = solvervariables[2]; //solvervariables.clear();
                vector <double> temptbv(SimParameters.getnumbertraits(),0.0); vector <double> temptdd(SimParameters.getnumbertraits(),0.0);
                for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                {
                    for(int k = 0; k < geno.size(); k++)
                    {
                        if((Population1.get_qtl_type())[k] == 2 || (Population1.get_qtl_type())[k] == 3)
                        {
                            
                            int temp = geno[k] - 48;
                            if(temp > 2){temp = 1;}
                            double tempfreq = (Population1.get_QTLFreq_AcrossGen())[k] / double(2*Population1.getQTLFreq_Number());
                            if(temp == 0){
                                temptbv[j] += -2 * (tempfreq) * alpha[k][j];
                                temptdd[j] += -2 * tempfreq * tempfreq * Population1.get_qtl_dom_quan(k,j);
                            } else if(temp == 1){
                                temptbv[j] += ((1 - tempfreq) - (tempfreq)) * alpha[k][j];
                                temptdd[j] += 2 * (1-tempfreq) * tempfreq * Population1.get_qtl_dom_quan(k,j);
                            } else if(temp == 2){
                                temptbv[j] += (2 * (1-tempfreq) * alpha[k][j]);
                                temptdd[j] += -2 * (1-tempfreq) * (1-tempfreq) * Population1.get_qtl_dom_quan(k,j);
                            } else {cout << endl << "Shouldn't Be Here" << endl; exit (EXIT_FAILURE);}
                        }
                    }
                }
                tbv_falconer.push_back(temptbv); tdd_falconer.push_back(temptdd);
                //cout<<id_falconer[id_falconer.size()-1]<<" "<<tbv_falconer[id_falconer.size()-1][0]<<" "<<tdd_falconer[id_falconer.size()-1][0]<<endl;
                //if(linenumber > 5){exit (EXIT_FAILURE);}
            }
            linenumber++;
        }
    }
    int numbanim = (estimatedsolutions.size() * 0.5);
    /* Save animal ID and generation in a vector and after order so lines up with pheno_gmatrix file */
    vector <int> ID_Gen_id; stringstream outputstring(stringstream::out); string line;
    if(SimParameters.getnumbertraits() == 1)
    {
        if(SimParameters.getQuantParam() == "statistical")
        {
            outputstring << "ID Sire Dam Sex Gen Age Progeny Dead Ped_F Gen_F Hap1_F Hap2_F Hap3_F Homolethal Heterlethal ";
            outputstring << "Homosublethal Hetersublethal Letequiv Homozy PropROH Fitness Phen EBV Acc TGV TBV TDD R TBV_Statistical TDD_Statistical\n";
        } else {
            outputstring << "ID Sire Dam Sex Gen Age Progeny Dead Ped_F Gen_F Hap1_F Hap2_F Hap3_F Homolethal Heterlethal ";
            outputstring << "Homosublethal Hetersublethal Letequiv Homozy PropROH Fitness Phen EBV Acc TGV TBV TDD R\n";
        }
    }
    if(SimParameters.getnumbertraits() == 2)
    {
        if(SimParameters.getQuantParam() == "statistical")
        {
            outputstring << "ID Sire Dam Sex Gen Age Progeny Dead Ped_F Gen_F Hap1_F Hap2_F Hap3_F Homolethal Heterlethal ";
            outputstring << "Homosublethal Hetersublethal Letequiv Homozy PropROH Fitness Phen1 EBV1 Acc1 TGV1 TBV1 TDD1 R1 TBV1_Statistical ";
            outputstring << "TDD1_Statistical Phen2 EBV2 Acc2 TGV2 TBV2 TDD2 R2 TBV2_Statistical TDD2_Statistical";
        } else {
            outputstring << "ID Sire Dam Sex Gen Age Progeny Dead Ped_F Gen_F Hap1_F Hap2_F Hap3_F Homolethal Heterlethal ";
            outputstring << "Homosublethal Hetersublethal Letequiv Homozy PropROH Fitness Phen1 EBV1 Acc1 TGV1 TBV1 TDD1 R1 ";
            outputstring << "Phen2 EBV2 Acc2 TGV2 TBV2 TDD2 R2";
        }
        if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
            outputstring << " index_tbv\n";
        } else{outputstring << "\n";}
    }
    int linenumbera = 0; int outputnumpart2 = 0;
    ifstream infile3;
    infile3.open(OUTPUTFILES.getloc_Master_DF().c_str());
    if(infile3.fail()){cout << "Error Opening File With Animal Information!\n"; exit (EXIT_FAILURE);}
    while (getline(infile3,line))
    {
        vector <string> lineVar;
        if(SimParameters.getnumbertraits() == 1)
        {
            for(int i = 0; i < 28; i++)
            {
                if(i <= 26)
                {
                    size_t pos = line.find(" ",0);
                    lineVar.push_back(line.substr(0,pos));
                    line.erase(0, pos + 1);
                }
                if(i == 27){lineVar.push_back(line);}
            }
            int templine = stoi(lineVar[0]) - 1;
            ID_Gen_id.push_back(stoi(lineVar[0]));
            ID_Gen.push_back(stoi(lineVar[4]));
            outputstring << lineVar[0] << " " <<  lineVar[1] << " " <<  lineVar[2] << " " <<  lineVar[3] << " " << lineVar[4] << " ";
            outputstring << lineVar[5] << " " <<  lineVar[6] << " " <<  lineVar[7] << " " <<  lineVar[8] << " " << lineVar[9] << " ";
            outputstring << lineVar[10] << " " << lineVar[11] << " " << lineVar[12] << " " << lineVar[13] << " " << lineVar[14] << " ";
            outputstring << lineVar[15] << " " << lineVar[16] << " " << lineVar[17] << " " << lineVar[18] << " " << lineVar[19] << " ";
            outputstring << lineVar[20] << " " << lineVar[21] << " ";
            /* If Founder population or genotype frequencies change don't output breeding values and accuracies */
            if(SimParameters.getGener()==SimParameters.getreferencegenblup())
            {
                if(SimParameters.getEBV_Calc()=="pblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[templine] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="gblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct" && SimParameters.getConstructGFreq()=="founder"){
                        outputstring << trueaccuracy[templine] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="rohblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[templine] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="ssgblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[templine] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
            } else {outputstring << 0.0 << " " << 0.0 << " ";}
            outputstring << lineVar[24] << " " << lineVar[25] << " " << lineVar[26] << " " << lineVar[27];
            if(SimParameters.getQuantParam() == "statistical")
            {
                if(stoi(lineVar[0]) != id_falconer[linenumbera])
                {
                    cout << endl << "Error in Generating TBV and TDD for Statistical Parameterization!!" << endl; exit (EXIT_FAILURE);
                }
                outputstring << " " << tbv_falconer[linenumbera][0] << " " << tdd_falconer[linenumbera][0] << endl;
            } else {outputstring << endl;}
        }
        if(SimParameters.getnumbertraits() == 2)
        {
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                for(int i = 0; i < 36; i++)
                {
                    if(i <= 34)
                    {
                        size_t pos = line.find(" ",0);
                        lineVar.push_back(line.substr(0,pos));
                        line.erase(0, pos + 1);
                    }
                    if(i == 35){lineVar.push_back(line);}
                }
            } else {
                for(int i = 0; i < 35; i++)
                {
                    if(i <= 33)
                    {
                        size_t pos = line.find(" ",0);
                        lineVar.push_back(line.substr(0,pos));
                        line.erase(0, pos + 1);
                    }
                    if(i == 34){lineVar.push_back(line);}
                }
            }
            int templine = stoi(lineVar[0]) - 1;
            ID_Gen_id.push_back(stoi(lineVar[0]));
            ID_Gen.push_back(stoi(lineVar[4]));
            outputstring << lineVar[0] << " " <<  lineVar[1] << " " <<  lineVar[2] << " " <<  lineVar[3] << " " << lineVar[4] << " ";
            outputstring << lineVar[5] << " " <<  lineVar[6] << " " <<  lineVar[7] << " " <<  lineVar[8] << " " << lineVar[9] << " ";
            outputstring << lineVar[10] << " " << lineVar[11] << " " << lineVar[12] << " " << lineVar[13] << " " << lineVar[14] << " ";
            outputstring << lineVar[15] << " " << lineVar[16] << " " << lineVar[17] << " " << lineVar[18] << " " << lineVar[19] << " ";
            outputstring << lineVar[20] << " " << lineVar[21] << " ";
            //cout << " --- " << endl;
            //cout << lineVar[0] << " " <<  lineVar[1] << " " <<  lineVar[2] << " " <<  lineVar[3] << " " << lineVar[4] << " ";
            //cout << lineVar[5] << " " <<  lineVar[6] << " " <<  lineVar[7] << " " <<  lineVar[8] << " " << lineVar[9] << " ";
            //cout << lineVar[10] << " " << lineVar[11] << " " << lineVar[12] << " " << lineVar[13] << " " << lineVar[14] << " ";
            //cout << lineVar[15] << " " << lineVar[16] << " " << lineVar[17] << " " << lineVar[18] << " " << lineVar[19] << " ";
            //cout << lineVar[20] << " " << lineVar[21] << " ";
            //cout << " --- " << endl;
            //cout << lineVar[22] << " " << lineVar[23] << " " << endl;
            //cout << " --- " << endl;
            //cout << lineVar[24] << " " << lineVar[25] << " " << lineVar[26] << " " << lineVar[27] << " " << lineVar[28] << " ";
            //cout << endl;
            //for(int i = 0; i < lineVar.size(); i++){cout << lineVar[i] << " ";}
            /* If Founder population or genotype frequencies change don't output breeding values and accuracies */
            if(SimParameters.getGener()==SimParameters.getreferencegenblup())
            {
                if(SimParameters.getEBV_Calc()=="pblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[(templine)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="gblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct" && SimParameters.getConstructGFreq()=="founder"){
                        outputstring << trueaccuracy[(templine)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="rohblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[(templine)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="ssgblup")
                {
                    outputstring << estimatedsolutions[templine] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[(templine)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
            } else {outputstring << 0.0 << " " << 0.0 << " ";}
            outputstring << lineVar[24] << " " << lineVar[25] << " " << lineVar[26] << " " << lineVar[27] << " ";
            if(SimParameters.getQuantParam() == "statistical")
            {
                if(stoi(lineVar[0]) != id_falconer[linenumbera])
                {
                    cout << endl << "Error in Generating TBV and TDD for Statistical Parameterization!!" << endl; exit (EXIT_FAILURE);
                }
                outputstring << tbv_falconer[linenumbera][0] << " " << tdd_falconer[linenumbera][0] << " ";
            }
            outputstring << lineVar[28] << " ";
            if(SimParameters.getGener()==SimParameters.getreferencegenblup())
            {
                if(SimParameters.getEBV_Calc()=="pblup")
                {
                    outputstring << estimatedsolutions[templine+numbanim] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[(templine+numbanim)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="gblup")
                {
                    outputstring << estimatedsolutions[templine+numbanim] << " ";
                    if(SimParameters.getSolver() == "direct" && SimParameters.getConstructGFreq()=="founder"){
                        outputstring << trueaccuracy[(templine+numbanim)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="rohblup")
                {
                    outputstring << estimatedsolutions[templine+numbanim] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[(templine+numbanim)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                if(SimParameters.getEBV_Calc()=="ssgblup")
                {
                    outputstring << estimatedsolutions[templine+numbanim] << " ";
                    if(SimParameters.getSolver() == "direct"){
                        outputstring << trueaccuracy[(templine+numbanim)] << " ";
                    } else {outputstring << 0.0 << " ";}
                }
                
            } else {outputstring << 0.0 << " " << 0.0 << " ";}
            outputstring << lineVar[31] << " "<< lineVar[32] << " " << lineVar[33] << " " << lineVar[34];
            if(SimParameters.getQuantParam() == "statistical")
            {
                if(stoi(lineVar[0]) != id_falconer[linenumbera])
                {
                    cout << endl << "Error in Generating TBV and TDD for Statistical Parameterization!!" << endl; exit (EXIT_FAILURE);
                }
                outputstring << " " << tbv_falconer[linenumbera][1] << " " << tdd_falconer[linenumbera][1];
            }
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                //cout << lineVar[35] << " " << lineVar[24] << " " << lineVar[31] << " ";
                //cout << (((atof(lineVar[24].c_str()) / double((Population1.get_sdGen0_TBV())[0]))*(SimParameters.get_IndexWeights())[0]) + ((atof(lineVar[31].c_str()) / double((Population1.get_sdGen0_TBV())[1]))*(SimParameters.get_IndexWeights())[1])) << endl;
                outputstring << " " << (((atof(lineVar[24].c_str()) / double((Population1.get_sdGen0_TBV())[0]))*(SimParameters.get_IndexWeights())[0]) + ((atof(lineVar[31].c_str()) / double((Population1.get_sdGen0_TBV())[1]))*(SimParameters.get_IndexWeights())[1])) << endl;
            } else {outputstring << endl;}
        }
        lineVar.clear(); outputnumpart2++;
        if(outputnumpart2 % 1000 == 0)
        {
            /* output master df file */
            std::ofstream output20(OUTPUTFILES.getloc_Master_DataFrame().c_str(), std::ios_base::app | std::ios_base::out);
            output20 << outputstring.str(); outputstring.str(""); outputstring.clear();
        }
        linenumbera++;
    }
    std::ofstream output20(OUTPUTFILES.getloc_Master_DataFrame().c_str(), std::ios_base::app | std::ios_base::out);
    output20 << outputstring.str(); outputstring.str(""); outputstring.clear();
    /* Bubble Sort by ID */
    int tempid; int tempgen;
    for(int i = 0; i < ID_Gen_id.size()-1; i++)
    {
        for(int j=i+1; j < ID_Gen_id.size(); j++)
        {
            if(ID_Gen_id[i] > ID_Gen_id[j])
            {
                tempid=ID_Gen_id[i]; tempgen=ID_Gen[i];
                ID_Gen_id[i] = ID_Gen_id[j]; ID_Gen[i] = ID_Gen[j];
                ID_Gen_id[j]=tempid; ID_Gen[j]=tempgen;
            }
        }
    }
    ID_Gen_id.clear();
}
/***********************************/
/* Generate QTL Summary Statistics */
/***********************************/
void generatesummaryqtl(parameters &SimParameters, outputfiles &OUTPUTFILES,vector < int > const &idgeneration,globalpopvar &Population1)
{
    int generations = SimParameters.getGener() + 1;
    /* Compute Number of Haplotypes by Generation */
    /* read in all animals haplotype ID's first; Don't need to really worry about this getting big; already in order */
    vector < vector < int > > PaternalHaplotypeIDs;
    vector < vector < int > > MaternalHaplotypeIDs;
    string line;
    int linenumbera = 0;
    ifstream infile22; string PaternalHap, MaternalHap;
    infile22.open(OUTPUTFILES.getloc_Pheno_GMatrix().c_str());
    if(infile22.fail()){cout << "Error Opening File\n";}
    while (getline(infile22,line))
    {
        vector < string > variables(7,"");
        for(int i = 0; i < 7; i++)
        {
            size_t pos = line.find(" ",0); variables[i] = line.substr(0,pos);
            if(pos != std::string::npos){line.erase(0, pos + 1);}
            if(pos == std::string::npos){line.clear();}
        }
        int start = 0;
        while(start < variables.size())
        {
            if(variables[start] == ""){variables.erase(variables.begin()+start);}
            if(variables[start] != ""){start++;}
        }
        if(variables.size() == 5){PaternalHap = variables[3]; MaternalHap = variables[4];}
        if(variables.size() == 6){PaternalHap = variables[4]; MaternalHap = variables[5];}
        /* Unstring each one and place in appropriate 2-d vector */
        vector < int > temp_pat; string quit = "NO";
        while(quit != "YES")
        {
            size_t pos = PaternalHap.find("_",0);
            if(pos > 0)                                                                         /* hasn't reached last one yet */
            {
                temp_pat.push_back(stoi(PaternalHap.substr(0,pos)));                             /* extend column by 1 */
                PaternalHap.erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}                                         /* has reached last one so now kill while loop */
        }
        PaternalHaplotypeIDs.push_back(temp_pat);                                               /* push back row */
        vector < int > temp_mat; quit = "NO";
        while(quit != "YES")
        {
            size_t pos = MaternalHap.find("_",0);
            if(pos > 0)                                                                         /* hasn't reached last one yet */
            {
                temp_mat.push_back(stoi(MaternalHap.substr(0,pos)));                            /* extend column by 1 */
                MaternalHap.erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}                                         /* has reached last one so now kill while loop */
        }
        MaternalHaplotypeIDs.push_back(temp_mat);                                               /* push back row */
        linenumbera++;
    }
    /* need to loop through population */
    double Number_Haplotypes [generations];
    for(int i = 0; i < (generations); i++){Number_Haplotypes[i] = 0.0;}                         /* Zero out first */
    for(int i = 0; i < PaternalHaplotypeIDs[0].size(); i++)                                   /* Loop acros haplotypes */
    {
        for(int j = 0; j < generations; j++)
        {
            vector < int > temphaps;
            for(int k = 0; k < idgeneration.size(); k++)
            {
                int bin = idgeneration[k];
                if(bin == j)
                {
                    temphaps.push_back(PaternalHaplotypeIDs[k][i]);
                    temphaps.push_back(MaternalHaplotypeIDs[k][i]);
                }
            }
            sort(temphaps.begin(),temphaps.end() );                                             /* Sort by haplotype ID */
            temphaps.erase(unique(temphaps.begin(),temphaps.end()),temphaps.end());             /* keeps only unique ones */
            Number_Haplotypes[j] += temphaps.size();
        }
    }
    for(int i = 0; i < generations; i++)
    {
        Number_Haplotypes[i] = Number_Haplotypes[i] / PaternalHaplotypeIDs[0].size();           /* Compute Average */
    }
    /* Summary Statistics for QTL's across generations */
    fstream checkSumQTL; checkSumQTL.open(OUTPUTFILES.getloc_Summary_QTL().c_str(), std::fstream::out | std::fstream::trunc); checkSumQTL.close();
    int Founder_Quan_Total[generations];                                        /* Total QTL from founder generation */
    int Founder_Quan_Lost[generations];                                         /* Lost QTL from founder generation */
    int Mutations_Quan_Total[generations];                                      /* Total New mutations */
    int Mutations_Quan_Lost[generations];                                       /* Lost New mutations */
    int Founder_Fit_Total[generations];                                         /* Total QTL from founder generation */
    int Founder_Fit_Lost[generations];                                          /* Lost QTL from founder generation */
    int Mutations_Fit_Total[generations];                                       /* Total new mutations */
    int Mutations_Fit_Lost[generations];                                        /* Lost new mutations */
    for(int i = 0; i < generations; i++)
    {
        Founder_Quan_Total[i] = 0; Founder_Quan_Lost[i] = 0; Mutations_Quan_Total[i] = 0; Mutations_Quan_Lost[i] = 0;
        Founder_Fit_Total[i] = 0; Founder_Fit_Lost[i] = 0; Mutations_Fit_Total[i] = 0; Mutations_Fit_Lost[i] = 0;
    }
    int All_QTL_Gen;
    string Type;
    string Freq_Gen;
    /* Import file and put each row into a vector */
    vector <string> numbers;
    line;
    int linenumberqtl = 0;
    ifstream infile5;
    infile5.open(OUTPUTFILES.getloc_qtl_class_object().c_str());
    if(infile5.fail()){cout << "Error Opening File\n";}
    while (getline(infile5,line))
    {
        if(linenumberqtl > 0)
        {
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = line.find(" ",0);
                solvervariables[i] = line.substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){line.erase(0, pos + 1);}
                if(pos == std::string::npos){line.clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() == 8){
                Type = solvervariables[5];
                All_QTL_Gen = stoi(solvervariables[6].c_str());
                Freq_Gen = solvervariables[7];
            } else if(solvervariables.size() == 6){
                Type = solvervariables[3];
                All_QTL_Gen = stoi(solvervariables[4].c_str());
                Freq_Gen = solvervariables[5];
            } else{cout << "Shouldn't be here" << endl; exit (EXIT_FAILURE);}
            for(int i = 0; i < generations; i++)
            {
                size_t pos = Freq_Gen.find("_",0);
                double freq = stod(Freq_Gen.substr(0,pos));
                Freq_Gen.erase(0,pos+1);
                if(Type == "2" && All_QTL_Gen == 0)
                {
                    if(freq > 0.0 && freq < 1.0){Founder_Quan_Total[i] += 1;}
                    if(freq == 0.0 || freq == 1.0){Founder_Quan_Lost[i] += 1;}
                }
                if(Type == "2" && All_QTL_Gen > 0)
                {
                    if(freq > 0.0 && freq < 1.0)
                    {
                        if(i >= All_QTL_Gen){Mutations_Quan_Total[i] += 1;}
                        if(i < All_QTL_Gen){Mutations_Quan_Total[i] += 0;}
                    }
                    if(freq == 0.0 || freq == 1.0)
                    {
                        if(i >= All_QTL_Gen){Mutations_Quan_Lost[i] += 1;}
                        if(i < All_QTL_Gen){Mutations_Quan_Lost[i] += 0;}
                    }
                }
                if((Type == "4" || Type == "5") && All_QTL_Gen == 0)
                {
                    if(freq > 0.0 && freq < 1.0){Founder_Fit_Total[i] += 1;}
                    if(freq == 0.0 || freq == 1.0){Founder_Fit_Lost[i] += 1;}
                }
                if((Type == "4" || Type == "5") && All_QTL_Gen > 0)
                {
                    if(freq > 0.0 && freq < 1.0)
                    {
                        if(i >= All_QTL_Gen){Mutations_Fit_Total[i] += 1;}
                        if(i < All_QTL_Gen){Mutations_Fit_Total[i] += 0;}
                    }
                    if(freq == 0.0 || freq == 1.0)
                    {
                        if(i >= All_QTL_Gen){Mutations_Fit_Lost[i] += 1;}
                        if(i < All_QTL_Gen){Mutations_Fit_Lost[i] += 0;}
                    }
                }
            }
        }
        linenumberqtl++;
    }
    for(int i = 0; i < generations; i++)
    {
        std::ofstream output5(OUTPUTFILES.getloc_Summary_QTL().c_str(), std::ios_base::app | std::ios_base::out);
        if(i == 0)
        {
            output5 << "Generation Quant_Founder_Start Quant_Founder_Lost Mutation_Quan_Total Mutation_Quan_Lost ";
            output5 << "Fit_Founder_Start Fit_Founder_Lost Mutation_Fit_Total Mutation_Fit_Lost ";
            output5 << "Avg_Haplotypes_Window ProgenyDiedFitness" << endl;
        }
        output5 << i << " " << Founder_Quan_Total[i] << " " << Founder_Quan_Lost[i] << " " << Mutations_Quan_Total[i] << " ";
        output5 << Mutations_Quan_Lost[i] <<" "<<Founder_Fit_Total[i]<<" "<<Founder_Fit_Lost[i]<<" "<<Mutations_Fit_Total[i]<<" "<<Mutations_Fit_Lost[i]<<" ";
        output5 << Number_Haplotypes[i] << " " << (Population1.get_numdeadfitness())[i] << endl;
    }
}
/*****************************************/
/* Generate dataframe Summary Statistics */
/*****************************************/
void generatessummarydf(parameters &SimParameters, outputfiles &OUTPUTFILES, globalpopvar &Population1)
{
    int generations = SimParameters.getGener() + 1;
    vector < double > generationcount(generations,0);
    vector < double > generationvalues;
    /* Inbreeding Summaries */
    vector < double > pedigreefvalues; vector < double > pedigreefmean(generations,0.0); vector < double > pedigreefsd(generations,0.0);
    vector < double > genomicfvalues; vector < double > genomicfmean(generations,0.0); vector < double > genomicfsd(generations,0.0);
    vector < double > h1fvalues; vector < double > h1fmean(generations,0.0); vector < double > h1fsd(generations,0.0);
    vector < double > h2fvalues; vector < double > h2fmean(generations,0.0); vector < double > h2fsd(generations,0.0);
    vector < double > h3fvalues; vector < double > h3fmean(generations,0.0); vector < double > h3fsd(generations,0.0);
    vector < double > homozygovalues; vector < double > homozygomean(generations,0.0); vector < double > homozygosd(generations,0.0);
    vector < double > proprohvalues; vector < double > proprohmean(generations,0.0); vector < double > proprohsd(generations,0.0);
    /* Fitness Summaries */
    vector < double > fitnessvalues; vector < double > fitnessmean(generations,0.0); vector < double > fitnesssd(generations,0.0);
    vector < double > homolethalvalues; vector < double > homolethalmean(generations,0.0); vector < double > homolethalsd(generations,0.0);
    vector < double > hetelethalvalues; vector < double > hetelethalmean(generations,0.0); vector < double > hetelethalsd(generations,0.0);
    vector < double > homosublethalvalues; vector < double > homosublethalmean(generations,0.0); vector < double > homosublethalsd(generations,0.0);
    vector < double > hetesublethalvalues; vector < double > hetesublethalmean(generations,0.0); vector < double > hetesublethalsd(generations,0.0);
    vector < double > lethalequivvalues; vector < double > lethalequivmean(generations,0.0); vector < double > lethalequivsd(generations,0.0);
    /* Performance Summaries Trait 1 */
    vector < double > phenovalues; vector < double > phenomean(generations,0.0); vector < double > phenosd(generations,0.0);
    vector < double > ebvvalues; vector < double > ebvmean(generations,0.0); vector < double > ebvsd(generations,0.0);
    vector < double > gvvalues; vector < double > gvmean(generations,0.0); vector < double > gvsd(generations,0.0);
    vector < double > bvvalues; vector < double > bvmean(generations,0.0); vector < double > bvsd(generations,0.0);
    vector < double > ddvalues; vector < double > ddmean(generations,0.0); vector < double > ddsd(generations,0.0);
    vector < double > resvalues; vector < double > resmean(generations,0.0); vector < double > ressd(generations,0.0);
    vector < double > bvfalc_values; vector < double > bvfalc_mean(generations,0.0); vector < double > bvfalc_sd(generations,0.0);
    vector < double > ddfalc_values; vector < double > ddfalc_mean(generations,0.0); vector < double > ddfalc_sd(generations,0.0);
    /* Performance Summaries Trait 2 */
    vector < double > phenovalues2; vector < double > phenomean2(generations,0.0); vector < double > phenosd2(generations,0.0);
    vector < double > ebvvalues2; vector < double > ebvmean2(generations,0.0); vector < double > ebvsd2(generations,0.0);
    vector < double > gvvalues2; vector < double > gvmean2(generations,0.0); vector < double > gvsd2(generations,0.0);
    vector < double > bvvalues2; vector < double > bvmean2(generations,0.0); vector < double > bvsd2(generations,0.0);
    vector < double > ddvalues2; vector < double > ddmean2(generations,0.0); vector < double > ddsd2(generations,0.0);
    vector < double > resvalues2; vector < double > resmean2(generations,0.0); vector < double > ressd2(generations,0.0);
    vector < double > bvfalc_values2; vector < double > bvfalc_mean2(generations,0.0); vector < double > bvfalc_sd2(generations,0.0);
    vector < double > ddfalc_values2; vector < double > ddfalc_mean2(generations,0.0); vector < double > ddfalc_sd2(generations,0.0);
    vector < double > covartrt1_trt2(generations,0.0);
    vector < double > tbvindexvalues; vector < double > tbvindexmean2(generations,0.0); vector < double > tbvindexsd2(generations,0.0);
    /* Read through and fill vector */
    string line; int linenumb = 0;
    ifstream infile22;
    infile22.open(OUTPUTFILES.getloc_Master_DataFrame().c_str());
    if(infile22.fail()){cout << "Error Opening File\n";}
    while (getline(infile22,line))
    {
        if(linenumb > 0)
        {
            size_t pos = line.find(" ", 0); line.erase(0, pos + 1);                                                 /* Don't need animal ID so skip */
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need dam ID so skip*/
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need sire ID so skip*/
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need sex so skip*/
            pos = line.find(" ",0); generationvalues.push_back(atoi((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);   /* Grab Generation */
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need age so skip*/
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need progeny number so skip*/
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need progeny dead so skip*/
            pos = line.find(" ",0); pedigreefvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);    /* Grab Pedigree f */
            pos = line.find(" ",0); genomicfvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab Genomic f */
            pos = line.find(" ",0); h1fvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);          /* Grab h1 f */
            pos = line.find(" ",0); h2fvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);          /* Grab h2 f */
            pos = line.find(" ",0); h3fvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);          /* Grab h3 f */
            pos = line.find(" ",0); homolethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);      /* Grab homozygous lethal */
            pos = line.find(" ",0); hetelethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);      /* Grab heterzygous lethal */
            pos = line.find(" ",0); homosublethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);   /* Grab homozygous sublethal */
            pos = line.find(" ",0); hetesublethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);   /* Grab heterzygous sublethal */
            pos = line.find(" ",0); lethalequivvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab Lethal Equivalents */
            pos = line.find(" ",0); homozygovalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);        /* Grab homozygosity */
            pos = line.find(" ",0); proprohvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);         /* Grab proportion ROH */
            pos = line.find(" ",0); fitnessvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);         /* Grab fitness */
            pos = line.find(" ",0); phenovalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);           /* Grab phenotype */
            pos = line.find(" ",0); ebvvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);             /* Grab ebv */
            pos = line.find(" ",0); line.erase(0,pos + 1);                                                                      /* Don't need accuracy so skip*/
            pos = line.find(" ",0); gvvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);              /* Grab gv */
            pos = line.find(" ",0); bvvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);              /* Grab bv */
            pos = line.find(" ",0); ddvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);              /* Grab dd */
            pos = line.find(" ",0); resvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);             /* Grab residuals */
            if(SimParameters.getQuantParam() == "statistical")
            {
                pos = line.find(" ",0); bvfalc_values.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab TBV_Statistical */
                pos = line.find(" ",0); ddfalc_values.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab TDD_Statistical */
            }
            if(SimParameters.getnumbertraits() == 2)
            {
                pos = line.find(" ",0); phenovalues2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);   /* Grab phenotype */
                pos = line.find(" ",0); ebvvalues2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab ebv */
                pos = line.find(" ",0); line.erase(0,pos + 1);                                                               /* Don't need accuracy so skip*/
                pos = line.find(" ",0); gvvalues2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);      /* Grab gv */
                pos = line.find(" ",0); bvvalues2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);      /* Grab bv */
                pos = line.find(" ",0); ddvalues2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);      /* Grab dd */
                pos = line.find(" ",0); resvalues2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab residuals */
                if(SimParameters.getQuantParam() == "statistical")
                {
                    pos = line.find(" ",0); bvfalc_values2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab TBV_Statistical */
                    pos = line.find(" ",0); ddfalc_values2.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab TDD_Statistical */
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
                {
                    pos = line.find(" ",0); tbvindexvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);     /* Grab tbv index */
                }
            }
        }
        linenumb++;
    }
    /* Calculate Mean */
    for(int i = 0; i < generationvalues.size(); i++)
    {
        /* Inbreeding Summary */
        generationcount[generationvalues[i]] += 1;
        pedigreefmean[generationvalues[i]] += pedigreefvalues[i];
        genomicfmean[generationvalues[i]] += genomicfvalues[i];
        h1fmean[generationvalues[i]] += h1fvalues[i];
        h2fmean[generationvalues[i]] += h2fvalues[i];
        h3fmean[generationvalues[i]] += h3fvalues[i];
        homozygomean[generationvalues[i]] += homozygovalues[i];
        proprohmean[generationvalues[i]] += proprohvalues[i];
        fitnessmean[generationvalues[i]] += fitnessvalues[i];
        homolethalmean[generationvalues[i]] += homolethalvalues[i];
        hetelethalmean[generationvalues[i]] += hetelethalvalues[i];
        homosublethalmean[generationvalues[i]] += homosublethalvalues[i];
        hetesublethalmean[generationvalues[i]] += hetesublethalvalues[i];
        lethalequivmean[generationvalues[i]] += lethalequivvalues[i];
        /* Performance Summary */
        phenomean[generationvalues[i]] += phenovalues[i];
        ebvmean[generationvalues[i]] += ebvvalues[i];
        gvmean[generationvalues[i]] += gvvalues[i];
        bvmean[generationvalues[i]] += bvvalues[i];
        ddmean[generationvalues[i]] += ddvalues[i];
        resmean[generationvalues[i]] += resvalues[i];
        if(SimParameters.getQuantParam() == "statistical")
        {
            bvfalc_mean[generationvalues[i]] += bvfalc_values[i];
            ddfalc_mean[generationvalues[i]] += ddfalc_values[i];
        }
        if(SimParameters.getnumbertraits() == 2)
        {
            phenomean2[generationvalues[i]] += phenovalues2[i];
            ebvmean2[generationvalues[i]] += ebvvalues2[i];
            gvmean2[generationvalues[i]] += gvvalues2[i];
            bvmean2[generationvalues[i]] += bvvalues2[i];
            ddmean2[generationvalues[i]] += ddvalues2[i];
            resmean2[generationvalues[i]] += resvalues2[i];
            if(SimParameters.getQuantParam() == "statistical")
            {
                bvfalc_mean2[generationvalues[i]] += bvfalc_values2[i];
                ddfalc_mean2[generationvalues[i]] += ddfalc_values2[i];
            }
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
            {
                tbvindexmean2[generationvalues[i]] += tbvindexvalues[i];
            }
        }
    }
    for(int i = 0; i < generationcount.size(); i++)
    {
        /* Inbreeding Summary */
        pedigreefmean[i] =  pedigreefmean[i] / generationcount[i];
        genomicfmean[i] = genomicfmean[i] / generationcount[i];
        h1fmean[i] = h1fmean[i] / generationcount[i];
        h2fmean[i] = h2fmean[i] / generationcount[i];
        h3fmean[i] = h3fmean[i] / generationcount[i];
        homozygomean[i] = homozygomean[i] / generationcount[i];
        if(proprohmean[i] != 0){proprohmean[i] = proprohmean[i] / double(generationcount[i]);}
        if(proprohmean[i] == 0){proprohmean[i] = 0.0;}
        fitnessmean[i] = fitnessmean[i] / generationcount[i];
        homolethalmean[i] = homolethalmean[i] / generationcount[i];
        hetelethalmean[i] = hetelethalmean[i] / generationcount[i];
        homosublethalmean[i] = homosublethalmean[i] / generationcount[i];
        hetesublethalmean[i] = hetesublethalmean[i] / generationcount[i];
        lethalequivmean[i] = lethalequivmean[i] / generationcount[i];
        /* Performance Summary */
        phenomean[i] = phenomean[i] / generationcount[i];
        ebvmean[i] = ebvmean[i] / generationcount[i];
        gvmean[i] = gvmean[i] / generationcount[i];
        bvmean[i] = bvmean[i] / generationcount[i];
        ddmean[i] = ddmean[i] / generationcount[i];
        resmean[i] = resmean[i] / generationcount[i];
        if(SimParameters.getQuantParam() == "statistical")
        {
            bvfalc_mean[i] = bvfalc_mean[i] / generationcount[i];
            ddfalc_mean[i] = ddfalc_mean[i] / generationcount[i];
        }
        if(SimParameters.getnumbertraits() == 2)
        {
            phenomean2[i] = phenomean2[i] / generationcount[i];
            ebvmean2[i] = ebvmean2[i] / generationcount[i];
            gvmean2[i] = gvmean2[i] / generationcount[i];
            bvmean2[i] = bvmean2[i] / generationcount[i];
            ddmean2[i] = ddmean2[i] / generationcount[i];
            resmean2[i] = resmean2[i] / generationcount[i];
            if(SimParameters.getQuantParam() == "statistical")
            {
                bvfalc_mean2[i] = bvfalc_mean2[i] / generationcount[i];
                ddfalc_mean2[i] = ddfalc_mean2[i] / generationcount[i];
            }
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
            {
                tbvindexmean2[i] = tbvindexmean2[i] / generationcount[i];
            }
        }
    }
    /* Calculate Variance */
    for(int i = 0; i < generationvalues.size(); i++)
    {
        /* Inbreeding Summary */
        pedigreefsd[generationvalues[i]] += ((pedigreefvalues[i]-pedigreefmean[generationvalues[i]]) * (pedigreefvalues[i]-pedigreefmean[generationvalues[i]]));
        genomicfsd[generationvalues[i]] += ((genomicfvalues[i] - genomicfmean[generationvalues[i]]) * (genomicfvalues[i] - genomicfmean[generationvalues[i]]));
        h1fsd[generationvalues[i]] += ((h1fvalues[i] - h1fmean[generationvalues[i]]) * (h1fvalues[i] - h1fmean[generationvalues[i]]));
        h2fsd[generationvalues[i]] += ((h2fvalues[i] - h2fmean[generationvalues[i]]) * (h2fvalues[i] - h2fmean[generationvalues[i]]));
        h3fsd[generationvalues[i]] += ((h3fvalues[i] - h3fmean[generationvalues[i]]) * (h3fvalues[i] - h3fmean[generationvalues[i]]));
        homozygosd[generationvalues[i]] += ((homozygovalues[i] - homozygomean[generationvalues[i]]) * (homozygovalues[i] - homozygomean[generationvalues[i]]));
        proprohsd[generationvalues[i]] += ((proprohvalues[i] - proprohmean[generationvalues[i]]) * (proprohvalues[i] - proprohmean[generationvalues[i]]));
        fitnesssd[generationvalues[i]] += ((fitnessvalues[i] - fitnessmean[generationvalues[i]]) * (fitnessvalues[i] - fitnessmean[generationvalues[i]]));
        homolethalsd[generationvalues[i]] +=((homolethalvalues[i]-homolethalmean[generationvalues[i]])*(homolethalvalues[i]-homolethalmean[generationvalues[i]]));
        hetelethalsd[generationvalues[i]] +=((hetelethalvalues[i]-hetelethalmean[generationvalues[i]])*(hetelethalvalues[i]-hetelethalmean[generationvalues[i]]));
        homosublethalsd[generationvalues[i]] += ((homosublethalvalues[i]-homosublethalmean[generationvalues[i]]) * (homosublethalvalues[i]-homosublethalmean[generationvalues[i]]));
        hetesublethalsd[generationvalues[i]] += ((hetesublethalvalues[i]-hetesublethalmean[generationvalues[i]]) * (hetesublethalvalues[i]-hetesublethalmean[generationvalues[i]]));
        lethalequivsd[generationvalues[i]] += ((lethalequivvalues[i]-lethalequivmean[generationvalues[i]]) * (lethalequivvalues[i]-lethalequivmean[generationvalues[i]]));
        /* Performance Summary */
        phenosd[generationvalues[i]] += ((phenovalues[i] - phenomean[generationvalues[i]]) * (phenovalues[i] - phenomean[generationvalues[i]]));
        ebvsd[generationvalues[i]] += ((ebvvalues[i] - ebvmean[generationvalues[i]]) * (ebvvalues[i] - ebvmean[generationvalues[i]]));
        gvsd[generationvalues[i]] += ((gvvalues[i] - gvmean[generationvalues[i]]) * (gvvalues[i] - gvmean[generationvalues[i]]));
        bvsd[generationvalues[i]] += ((bvvalues[i] - bvmean[generationvalues[i]]) * (bvvalues[i] - bvmean[generationvalues[i]]));
        ddsd[generationvalues[i]] += ((ddvalues[i] - ddmean[generationvalues[i]]) * (ddvalues[i] - ddmean[generationvalues[i]]));
        ressd[generationvalues[i]] += ((resvalues[i] - resmean[generationvalues[i]]) * (resvalues[i] - resmean[generationvalues[i]]));
        if(SimParameters.getQuantParam() == "statistical")
        {
            bvfalc_sd[generationvalues[i]] += ((bvfalc_values[i]-bvfalc_mean[generationvalues[i]])*(bvfalc_values[i]-bvfalc_mean[generationvalues[i]]));
            ddfalc_sd[generationvalues[i]] += ((ddfalc_values[i]-ddfalc_mean[generationvalues[i]])*(ddfalc_values[i]-ddfalc_mean[generationvalues[i]]));
        }
        if(SimParameters.getnumbertraits() == 2)
        {
            phenosd2[generationvalues[i]] += ((phenovalues2[i] - phenomean2[generationvalues[i]]) * (phenovalues2[i] - phenomean2[generationvalues[i]]));
            ebvsd2[generationvalues[i]] += ((ebvvalues2[i] - ebvmean2[generationvalues[i]]) * (ebvvalues2[i] - ebvmean2[generationvalues[i]]));
            gvsd2[generationvalues[i]] += ((gvvalues2[i] - gvmean2[generationvalues[i]]) * (gvvalues2[i] - gvmean2[generationvalues[i]]));
            bvsd2[generationvalues[i]] += ((bvvalues2[i] - bvmean2[generationvalues[i]]) * (bvvalues2[i] - bvmean2[generationvalues[i]]));
            ddsd2[generationvalues[i]] += ((ddvalues2[i] - ddmean2[generationvalues[i]]) * (ddvalues2[i] - ddmean2[generationvalues[i]]));
            ressd2[generationvalues[i]] += ((resvalues2[i] - resmean2[generationvalues[i]]) * (resvalues2[i] - resmean2[generationvalues[i]]));
            covartrt1_trt2[generationvalues[i]] += ((bvvalues2[i] - bvmean2[generationvalues[i]]) * ((bvvalues[i] - bvmean[generationvalues[i]])));
            if(SimParameters.getQuantParam() == "statistical")
            {
                bvfalc_sd2[generationvalues[i]] += ((bvfalc_values2[i]-bvfalc_mean2[generationvalues[i]])*(bvfalc_values2[i]-bvfalc_mean2[generationvalues[i]]));
                ddfalc_sd2[generationvalues[i]] += ((ddfalc_values2[i]-ddfalc_mean2[generationvalues[i]])*(ddfalc_values2[i]-ddfalc_mean2[generationvalues[i]]));
            }
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
            {
                tbvindexsd2[generationvalues[i]] += ((tbvindexvalues[i] - tbvindexmean2[generationvalues[i]]) * ((tbvindexvalues[i] - tbvindexmean2[generationvalues[i]])));
            }
        }
    }
    for(int i = 0; i < generationcount.size(); i++)
    {
        /* Inbreeding Summary */
        pedigreefsd[i] = pedigreefsd[i] / double(generationcount[i] -1);
        genomicfsd[i] = genomicfsd[i] / double(generationcount[i] -1);
        h1fsd[i] = h1fsd[i] / double(generationcount[i] -1);
        h2fsd[i] = h2fsd[i] / double(generationcount[i] -1);
        h3fsd[i] = h3fsd[i] / double(generationcount[i] -1);
        if(homozygosd[i] != 0){homozygosd[i] = homozygosd[i] / double(generationcount[i] -1);}
        if(homozygosd[i] == 0){homozygosd[i] = 0.0;}
        if(proprohsd[i] != 0){proprohsd[i] = proprohsd[i] / double(generationcount[i] - 1);}
        if(proprohsd[i] == 0){proprohsd[i] = 0.0;}
        fitnesssd[i] = fitnesssd[i] / double(generationcount[i] -1);
        homolethalsd[i] = homolethalsd[i] / double(generationcount[i] -1);
        hetelethalsd[i] = hetelethalsd[i] / double(generationcount[i] -1);
        homosublethalsd[i] = homosublethalsd[i] / double(generationcount[i] -1);
        hetesublethalsd[i] = hetesublethalsd[i] / double(generationcount[i] -1);
        lethalequivsd[i] = lethalequivsd[i] / double(generationcount[i] -1);
        /* Performance Summary */
        phenosd[i] = phenosd[i] / double(generationcount[i] -1);
        ebvsd[i] = ebvsd[i] / double(generationcount[i] -1);
        gvsd[i] = gvsd[i] / double(generationcount[i] -1);
        bvsd[i] = bvsd[i] / double(generationcount[i] -1);
        ddsd[i] = ddsd[i] / double(generationcount[i] -1);
        ressd[i] = ressd[i] / double(generationcount[i] -1);
        if(SimParameters.getQuantParam() == "statistical")
        {
            bvfalc_sd[i] += bvfalc_sd[i] / double(generationcount[i] -1);
            ddfalc_sd[i] += ddfalc_sd[i] / double(generationcount[i] -1);
        }
        if(SimParameters.getnumbertraits() == 2)
        {
            phenosd2[i] = phenosd2[i] / double(generationcount[i] -1);
            ebvsd2[i] = ebvsd2[i] / double(generationcount[i] -1);
            gvsd2[i] = gvsd2[i] / double(generationcount[i] -1);
            bvsd2[i] = bvsd2[i] / double(generationcount[i] -1);
            ddsd2[i] = ddsd2[i] / double(generationcount[i] -1);
            ressd2[i] = ressd2[i] / double(generationcount[i] -1);
            covartrt1_trt2[i] = (covartrt1_trt2[i] / sqrt(bvsd2[i]*bvsd[i])) / double(generationcount[i] -1);
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
            {
                tbvindexsd2[i] = tbvindexsd2[i] / double(generationcount[i] -1);
            }
        }
    }
    string outfileinbreeding = OUTPUTFILES.getloc_Summary_DF() + "_Inbreeding";
    string outfileperformance = OUTPUTFILES.getloc_Summary_DF() + "_Performance";
    fstream checkoutinbreeding; checkoutinbreeding.open(outfileinbreeding, std::fstream::out | std::fstream::trunc); checkoutinbreeding.close();
    fstream checkoutperformance; checkoutperformance.open(outfileperformance, std::fstream::out | std::fstream::trunc); checkoutperformance.close();
    /* output inbreeding */
    for(int i = 0; i < generations; i++)
    {
        cout.setf(ios::fixed);
        std::ofstream output(outfileinbreeding, std::ios_base::app | std::ios_base::out);
        if(i == 0)
        {
            output << "Generation ped_f gen_f h1_f h2_f h3_f homozy PropROH ExpHet fitness homozlethal hetezlethal homozysublethal hetezsublethal lethalequiv" << endl;
        }
        output << i << " ";
        output << setprecision(4) << pedigreefmean[i] << "(" << pedigreefsd[i] << ") ";
        output << setprecision(4) << genomicfmean[i] << "(" << genomicfsd[i] << ") ";
        output << setprecision(4) << h1fmean[i] << "(" << h1fsd[i] << ") ";
        output << setprecision(4) << h2fmean[i] << "(" << h2fsd[i] << ") ";
        output << setprecision(4) << h3fmean[i] << "(" << h3fsd[i] << ") ";
        output << setprecision(4) << homozygomean[i] << "(" << homozygosd[i] << ") ";
        output << setprecision(4) << proprohmean[i] << "(" << proprohsd[i] << ") ";
        output << setprecision(4) << (Population1.get_expectedheter())[i] << " ";
        output << setprecision(4) << fitnessmean[i] << "(" << fitnesssd[i] << ") ";
        output << setprecision(4) << homolethalmean[i] << "(" << homolethalsd[i] << ") ";
        output << setprecision(4) << hetelethalmean[i] << "(" << hetelethalsd[i] << ") ";
        output << setprecision(4) << homosublethalmean[i] << "(" << homosublethalsd[i] << ") ";
        output << setprecision(4) << hetesublethalmean[i] << "(" << hetesublethalsd[i] << ") ";
        output << setprecision(4) << lethalequivmean[i] << "(" << lethalequivsd[i] << ")" << endl;
        cout.unsetf(ios::fixed);
    }
    if(SimParameters.getSelection() == "ebv" && SimParameters.getDamRepl() == 1.0 && SimParameters.getSireRepl() == 1.0)
    {
        /**********************************************************/
        /* DeltaG = r_ebv,bv * ((i_males + i_females)/2) * sd_bv */
        /**********************************************************/
        if(SimParameters.getnumbertraits() == 1)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen r_ebv_tbv intensity_males intensity_females sdtbv ExpDeltaG RealizedDeltaG" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output << i << " " << (Population1.get_accuracydeltaG())[i] << " ";
                output << Population1.get_Intensity_Males(i,0) << " " << Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
                output << (Population1.get_accuracydeltaG())[i] * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*sqrt(bvsd[i]) << " " << bvmean[i] - bvmean[i-1] << endl;
            }
        }
        /**********************************************************/
        /* DeltaG Correlated Trait = r_ebv1,ebv2 * r_ebv1,bv1 * ((i_males + i_females)/2) * sd_bv_2 */
        /**********************************************************/
        if(SimParameters.getnumbertraits() == 2)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen r_ebv_tbv_t1 intensity_males intensity_females sdtbv_t1 ExpDeltaG_t1 RealizedDeltaG_t1 ";
            output << "rg sd_tbv_t2 ExpDeltaG_t2 RealizedDeltaG_t2" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output << i << " " << (Population1.get_accuracydeltaG())[i] << " ";
                output << Population1.get_Intensity_Males(i,0) << " " << Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
                output << (Population1.get_accuracydeltaG())[i] * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*sqrt(bvsd[i]) << " " << bvmean[i] - bvmean[i-1] << " " << covartrt1_trt2[i] << " " << sqrt(bvsd2[i]) << " ";
                output << covartrt1_trt2[i] * (Population1.get_accuracydeltaG())[i] * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2)) * sqrt(bvsd2[i]);
                output << " " << bvmean2[i] - bvmean2[i-1] << endl;
            }
        }
    }
    
    if(SimParameters.getSelection() == "ebv" && SimParameters.getDamRepl() == 0.5 && SimParameters.getSireRepl() == 0.5 && SimParameters.getMaxAge())
    {
        /**********************************************************/
        /* DeltaG = r_ebv,bv * ((i_males + i_females)/2) * sd_bv */
        /**********************************************************/
        if(SimParameters.getnumbertraits() == 1)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen r_ebv_tbv intensity_males intensity_females sdtbv ExpDeltaG RealizedDeltaG" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output << i << " " << (Population1.get_accuracydeltaG())[i] << " ";
                output << Population1.get_Intensity_Males(i,0) << " " << Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
                output << (Population1.get_accuracydeltaG())[i] * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*sqrt(bvsd[i]) << " " << bvmean[i] - bvmean[i-1] << endl;
            }
        }
        /**********************************************************/
        /* DeltaG Correlated Trait = r_ebv1,ebv2 * r_ebv1,bv1 * ((i_males + i_females)/2) * sd_bv_2 */
        /**********************************************************/
        if(SimParameters.getnumbertraits() == 2)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen r_ebv_tbv_t1 intensity_males intensity_females sdtbv_t1 ExpDeltaG_t1 RealizedDeltaG_t1 ";
            output << "rg sd_tbv_t2 ExpDeltaG_t2 RealizedDeltaG_t2" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output << i << " " << (Population1.get_accuracydeltaG())[i] << " ";
                output << Population1.get_Intensity_Males(i,0) << " " << Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
                output << (Population1.get_accuracydeltaG())[i] * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*sqrt(bvsd[i]) << " " << bvmean[i] - bvmean[i-1] << " " << covartrt1_trt2[i] << " " << sqrt(bvsd2[i]) << " ";
                output << covartrt1_trt2[i] * (Population1.get_accuracydeltaG())[i] * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2)) * sqrt(bvsd2[i]);
                output << " " << bvmean2[i] - bvmean2[i-1] << endl;
            }
        }
    }

    
    if(SimParameters.getSelection() == "phenotype" && SimParameters.getDamRepl() == 1.0 && SimParameters.getSireRepl() == 1.0)
    {
        /**********************************************************/
        /* DeltaG = h2 * ((i_males + i_females)/2) * sd_phenotype */
        /**********************************************************/
        if(SimParameters.getnumbertraits() == 1)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen h2 intensity_males intensity_females sdphenotype ExpDeltaG RealizedDeltaG" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output << i << " " << bvsd[i] / double(bvsd[i]+ressd[i]) << " " << Population1.get_Intensity_Males(i,0) << " ";
                output << Population1.get_Intensity_Females(i,0) << " " << sqrt(phenosd[i]) << " ";
                //output << Population1.get_GenInterval_Males(i,0) << " " << Population1.get_GenInterval_Females(i,0) << " ";
                output << ((bvsd[i] / double(bvsd[i]+ressd[i])) * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*(sqrt(phenosd[i]))) << " ";
                output << bvmean[i] - bvmean[i-1] << endl;
            }
        }
        /*********************************************************************************************/
        /* DeltaG Correlated Trait = ((i_males + i_females)/2) * h_1 * h_2 * rg_1_2 * sd_phenotype_2 */
        /*********************************************************************************************/
        if(SimParameters.getnumbertraits() == 2)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen h2_t1 intensity_males_t1 intensity_females_t1 sdphenotype_t1 ExpDeltaG_t1 RealizedDeltaG_t1 ";
            output << "h_t1 h_t2 rg sdphenotype_t2 ExpDeltaG_t2 RealizedDeltaG_t2" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output << i << " " << bvsd[i] / double(bvsd[i]+ressd[i]) << " " << Population1.get_Intensity_Males(i,0) << " ";
                output << Population1.get_Intensity_Females(i,0) << " " << sqrt(phenosd[i]) << " ";
                output << ((bvsd[i] / double(bvsd[i]+ressd[i])) * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*(sqrt(phenosd[i]))) << " ";
                output << bvmean[i] - bvmean[i-1] << " ";
                output << sqrt(bvsd[i] / double(bvsd[i]+ressd[i])) << " " << sqrt(bvsd2[i] / double(bvsd2[i]+ressd2[i])) << " " << covartrt1_trt2[i] << " ";
                output << sqrt(phenosd2[i]) << " ";
                output << ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2)) * sqrt(bvsd[i] / double(bvsd[i]+ressd[i])) * sqrt(bvsd2[i] / double(bvsd2[i]+ressd2[i])) * covartrt1_trt2[i]  * sqrt(phenosd2[i]) << " ";
                output << bvmean2[i] - bvmean2[i-1] << endl;
            }
        }
    }
//    if(SimParameters.getSelection() == "phenotype" && (SimParameters.getDamRepl() < 1.0 || SimParameters.getSireRepl() < 1.0))
//    {
//        /********************************************************************/
//        /* DeltaG = (h2 * (i_males + i_females) * sd_phenotype) / (Lm + Lf) */
//        /********************************************************************/
//        if(SimParameters.getnumbertraits() == 1)
//        {
//            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
//            output << "Gen h2 intensity_males intensity_females sdphenotype L_males L_females ExpDeltaG RealizedDeltaG" << endl;
//            for(int i = 1; i < (generations-1); i++)
//            {
//                output << i << " " << bvsd[i] / double(bvsd[i]+ressd[i]) << " " << Population1.get_Intensity_Males(i,0) << " ";
//                output << Population1.get_Intensity_Females(i,0) << " " << sqrt(phenosd[i]) << " ";
//                output << Population1.get_GenInterval_Males(i,0) << " " << Population1.get_GenInterval_Females(i,0) << " ";
//                output << ((bvsd[i] / double(bvsd[i]+ressd[i])) * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0)))*(sqrt(phenosd[i]))) / (Population1.get_GenInterval_Males(i,0) + Population1.get_GenInterval_Females(i,0))  << " ";
//                output << bvmean[i] - bvmean[i-1] << endl;
//            }
//        }
//        /********************************************************************************************************/
//        /* DeltaG Correlated Trait = ((i_males + i_females) * h_1 * h_2 * rg_1_2 * sd_phenotype_2) / (Lm + Lf)  */
//        /********************************************************************************************************/
//        if(SimParameters.getnumbertraits() == 2)
//        {
//            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
//            output << "Gen h2_t1 intensity_males intensity_females sdphenotype_t1 L_males L_females ExpDeltaG_t1 RealizedDeltaG_t1";
//            output << " h_t1 h_t2 rg sdphenotype_t2 ExpDeltaG_t2 RealizedDeltaG_t2" << endl;
//            for(int i = 1; i < (generations-1); i++)
//            {
//                output << i << " " << bvsd[i] / double(bvsd[i]+ressd[i]) << " " << Population1.get_Intensity_Males(i,0) << " ";
//                output << Population1.get_Intensity_Females(i,0) << " " << sqrt(phenosd[i]) << " ";
//                output << Population1.get_GenInterval_Males(i,0) << " " << Population1.get_GenInterval_Females(i,0) << " ";
//                output << ((bvsd[i] / double(bvsd[i]+ressd[i])) * ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0)))*(sqrt(phenosd[i]))) / (Population1.get_GenInterval_Males(i,0) + Population1.get_GenInterval_Females(i,0))  << " ";
//                output << bvmean[i] - bvmean[i-1] << " ";
//                output << sqrt(bvsd[i] / double(bvsd[i]+ressd[i])) << " " << sqrt(bvsd2[i] / double(bvsd2[i]+ressd2[i])) << " " << covartrt1_trt2[i] << " ";
//                output << sqrt(phenosd2[i]) << " ";
//                output << (((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2)) * sqrt(bvsd[i] / double(bvsd[i]+ressd[i])) * sqrt(bvsd2[i] / double(bvsd2[i]+ressd2[i])) * covartrt1_trt2[i]  * sqrt(phenosd2[i])) / (Population1.get_GenInterval_Males(i,0) + Population1.get_GenInterval_Females(i,0)) << " ";
//                output << bvmean2[i] - bvmean2[i-1] << endl;
//            }
//        }
//    }
    if(SimParameters.getSelection() == "tbv" && SimParameters.getDamRepl() == 1.0 && SimParameters.getSireRepl() == 1.0)
    {
        /****************************************************/
        /* DeltaG = ((i_males + i_females)/2) * sd_tbv */
        /****************************************************/
        if(SimParameters.getnumbertraits() == 1)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen intensity_males intensity_females sd_tbv ExpDeltaG RealizedDeltaG" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output<<i<<" "<<Population1.get_Intensity_Males(i,0) << " "<< Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
                output << ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*sqrt(bvsd[i]) << " ";
                output << bvmean[i] - bvmean[i-1] << endl;
            }
        }
        /*********************************************************************************************/
        /* DeltaG Correlated Trait = ((i_males + i_females)/2) * rg_1_2 * sd_tbv_2 */
        /*********************************************************************************************/
        if(SimParameters.getnumbertraits() == 2)
        {
            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
            output << "Gen intensity_males_t1 intensity_females_t1 sd_tbv_t1 ExpDeltaG_t1 RealizedDeltaG_t1 ";
            output << "rg sd_tbv_t2 ExpDeltaG_t2 RealizedDeltaG_t2" << endl;
            for(int i = 1; i < (generations-1); i++)
            {
                output<<i<<" "<<Population1.get_Intensity_Males(i,0) << " "<< Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
                output << ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2))*sqrt(bvsd[i]) << " ";
                output << bvmean[i] - bvmean[i-1] << " " << covartrt1_trt2[i] << " " << sqrt(bvsd2[i]) << " ";
                output << ((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0))/double(2)) * covartrt1_trt2[i] * sqrt(bvsd2[i]);
                output << " " << bvmean2[i] - bvmean2[i-1] << endl;
            }
        }
    }
//    if(SimParameters.getSelection() == "tbv" && (SimParameters.getDamRepl() < 1.0 || SimParameters.getSireRepl() < 1.0))
//    {
//        /*********************************************************/
//        /* DeltaG = ((i_males + i_females) * sd_tbv) / (Lm + Lf) */
//        /*********************************************************/
//        if(SimParameters.getnumbertraits() == 1)
//        {
//            std::ofstream output(OUTPUTFILES.getloc_ExpRealDeltaG().c_str(), std::ios_base::app | std::ios_base::out);
//            output << "Gen intensity_males intensity_females sd_tbv L_males L_females ExpDeltaG RealizedDeltaG" << endl;
//            for(int i = 1; i < (generations-1); i++)
//            {
//                output<<i<<" "<<Population1.get_Intensity_Males(i,0) << " "<< Population1.get_Intensity_Females(i,0) << " " << sqrt(bvsd[i]) << " ";
//                output << Population1.get_GenInterval_Males(i,0) << " " << Population1.get_GenInterval_Females(i,0) << " ";
//                output << (((Population1.get_Intensity_Males(i,0)+Population1.get_Intensity_Females(i,0)))*sqrt(bvsd[i])) / (Population1.get_GenInterval_Males(i,0) + Population1.get_GenInterval_Females(i,0)) << " ";
//                output << bvmean[i] - bvmean[i-1] << endl;
//            }
//        }
//    
//    }
    /* output performance */
    for(int i = 0; i < generations; i++)
    {
        cout.setf(ios::fixed);
        std::ofstream output1(outfileperformance, std::ios_base::app | std::ios_base::out);
        if(i == 0)
        {
            if(SimParameters.getnumbertraits() == 1)
            {
                if(SimParameters.getQuantParam() == "statistical"){
                    output1 << "Generation phen ebv tgv tbv tdd res bvstat ddstat" << endl;
                } else {output1 << "Generation phen ebv tgv tbv tdd res" << endl;}
            }
            if(SimParameters.getnumbertraits() == 2)
            {
                if(SimParameters.getQuantParam() == "statistical"){
                    output1 << "Generation phen1 ebv1 tgv1 tbv1 tdd1 res1 bvstat1 ddstat1 phen2 ebv2 tgv2 tbv2 tdd2 res2 bvstat2 ddstat2";
                } else {output1 << "Generation phen1 ebv1 tgv1 tbv1 tdd1 res1 phen2 ebv2 tgv2 tbv2 tdd2 res2";}
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    output1 << " index_tbv" << endl;
                } else {output1 << endl;}
            }
        }
        output1 << i << " ";
        output1 << setprecision(4) << phenomean[i]  << "(" << phenosd[i] << ") ";
        output1 << setprecision(4) << ebvmean[i]  << "(" << ebvsd[i] << ") ";
        output1 << setprecision(4) << gvmean[i]  << "(" << gvsd[i] << ") ";
        output1 << setprecision(4) << bvmean[i]  << "(" << bvsd[i] << ") ";
        output1 << setprecision(4) << ddmean[i]  << "(" << ddsd[i] << ") ";
        output1 << setprecision(4) << resmean[i]  << "(" << ressd[i] << ")";
        if(SimParameters.getnumbertraits() == 1)
        {
            if(SimParameters.getQuantParam() == "statistical"){
                output1 << " " << setprecision(4) << bvfalc_mean[i] << "(" << bvfalc_sd[i] << ") ";
                output1 << setprecision(4) << ddfalc_mean[i] << "(" << ddfalc_sd[i] << ")" << endl;
            } else {output1 << endl;}
        }
        if(SimParameters.getnumbertraits() == 2)
        {
            if(SimParameters.getQuantParam() == "statistical")
            {
                output1 << " " << setprecision(4) << bvfalc_mean[i] << "(" << bvfalc_sd[i] << ") ";
                output1 << setprecision(4) << ddfalc_mean[i] << "(" << ddfalc_sd[i] << ") ";
            }
            output1 << " " << setprecision(4) << phenomean2[i]  << "(" << phenosd2[i] << ") ";
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
            {
                output1 << setprecision(4) << ebvmean2[i]  << "(" << ebvsd2[i] << ") ";
            } else {output1 << "-(-) ";}
            output1 << setprecision(4) << gvmean2[i]  << "(" << gvsd2[i] << ") ";
            output1 << setprecision(4) << bvmean2[i]  << "(" << bvsd2[i] << ") ";
            output1 << setprecision(4) << ddmean2[i]  << "(" << ddsd2[i] << ") ";
            output1 << setprecision(4) << resmean2[i]  << "(" << ressd2[i] << ")";
            if(SimParameters.getQuantParam() == "statistical"){
                output1 << " " << setprecision(4) << bvfalc_mean2[i] << "(" << bvfalc_sd2[i] << ") ";
                output1 << setprecision(4) << ddfalc_mean2[i] << "(" << ddfalc_sd2[i] << ")";
            }
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                output1 << setprecision(4) << " " << tbvindexmean2[i] << "(" << tbvindexsd2[i] << ")" << endl;
            } else {output1 << endl;}
        }
        cout.unsetf(ios::fixed);
    }
}
/*****************************************************************************************/
/* Make location be chr and pos instead of in current format to make it easier for user. */
/*****************************************************************************************/
void generateqtlfile(parameters &SimParameters, outputfiles &OUTPUTFILES)
{
    vector <string> numbers; string line;
    ifstream infileqtlreformat;
    infileqtlreformat.open(OUTPUTFILES.getloc_qtl_class_object().c_str());
    if(infileqtlreformat.fail()){cout << "Error Opening File\n";}
    while (getline(infileqtlreformat,line)){numbers.push_back(line);}
    vector < int > qtlout_chr((numbers.size()-1),0);
    vector < int > qtlout_pos((numbers.size()-1),0.0);
    vector < string > restofit((numbers.size()-1),"");
    for(int i = 1; i < numbers.size(); i++)
    {
        size_t pos =  numbers[i].find(" ",0); string temp =  numbers[i].substr(0,pos); numbers[i].erase(0, pos + 1);
        qtlout_chr[i-1] = stoi(temp.c_str());
        qtlout_pos[i-1] = (stof(temp.c_str())-qtlout_chr[i-1])*(SimParameters.get_ChrLength())[qtlout_chr[i-1]-1]; /* convert to nuclotides */
        restofit[i-1] = numbers[i];
    }
    ofstream output10;
    output10.open (OUTPUTFILES.getloc_qtl_class_object().c_str());
    if(SimParameters.getnumbertraits() == 1){output10 << "Chr Pos Additive_Selective Dominance Type Gen Freq" << endl;}
    if(SimParameters.getnumbertraits() == 2){output10 << "Chr Pos Additive_Selective1 Dominance1 Additive_Selective2 Dominance2 Type Gen Freq" << endl;}
    for(int i = 1; i < numbers.size(); i++){output10 << qtlout_chr[i-1] << " " << qtlout_pos[i-1] << " " << restofit[i-1] << endl;}
    output10.close();
}
/**************************************/
/* Clean Up Files Not Needed In Folder*/
/**************************************/
void CleanUpSimulation(outputfiles &OUTPUTFILES)
{
    string command = "rm -rf " + OUTPUTFILES.getloc_Master_DF() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Pheno_GMatrix() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_BinaryG_Matrix() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Binarym_Matrix() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Binaryp_Matrix() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_BinaryLinv_Matrix() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_BinaryGinv_Matrix() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Bayes_MCMC_Samples() + " || true"; system(command.c_str());
    command = "rm -rf " + OUTPUTFILES.getloc_Bayes_PosteriorMeans() + " || true"; system(command.c_str());
}
/***************************************************************************************************************************/
/* If you have multiple replicates create a new directory within this folder to store them and just attach seed afterwards */
/***************************************************************************************************************************/
void SaveReplicates (int reps,parameters &SimParameters, outputfiles &OUTPUTFILES,string logfileloc, string path)
{
    if(reps == 0)               /* First delete replicates folder if exists */
    {
        string systemcall = "rm -rf " + path + "/" + SimParameters.getOutputFold() + "/replicates || true";
        system(systemcall.c_str());
        systemcall = "mkdir " + path + "/" + SimParameters.getOutputFold() + "/replicates";
        system(systemcall.c_str());
    }
    /* make seed a string */
    stringstream stringseed; stringseed << SimParameters.getSeed(); string stringseednumber = stringseed.str();
    /* Logfile */
    string systemcall = "mv "+logfileloc+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/"+"log_file_"+stringseednumber;
    system(systemcall.c_str());
    /* Low Fitness File */
    systemcall = "mv "+OUTPUTFILES.getloc_lowfitnesspath()+" "+path+"/" + SimParameters.getOutputFold() + "/replicates/";
    systemcall += "Low_Fitness_" + stringseednumber; system(systemcall.c_str());
    /* Master DataFrame File */
    systemcall = "mv "+OUTPUTFILES.getloc_Master_DataFrame()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Master_DataFrame_" + stringseednumber; system(systemcall.c_str());
    /* Master Genotype File */
    systemcall = "mv "+OUTPUTFILES.getloc_Master_Genotype_zip()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Master_Genotypes_" + stringseednumber+".gz"; system(systemcall.c_str());
    /* Master QTL File */
    systemcall = "mv "+OUTPUTFILES.getloc_qtl_class_object()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "QTL_new_old_Class_" + stringseednumber; system(systemcall.c_str());
    /* Marker File */
    systemcall = "mv "+OUTPUTFILES.getloc_Marker_Map()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Marker_Map_" + stringseednumber; system(systemcall.c_str());
    /* Output Summary QTL */
    systemcall = "mv "+OUTPUTFILES.getloc_Summary_QTL()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Summary_Statistics_QTL_"+stringseednumber; system(systemcall.c_str());
    /* Output Summary Inbreeding */
    systemcall = "mv "+OUTPUTFILES.getloc_Summary_DF()+"_Inbreeding "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Summary_Statistics_DataFrame_Inbreeding_"+stringseednumber; system(systemcall.c_str());
    /* Output Summary Performance */
    systemcall = "mv "+OUTPUTFILES.getloc_Summary_DF()+"_Performance "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Summary_Statistics_DataFrame_Performance_"+stringseednumber; system(systemcall.c_str());
    
    systemcall = "mv "+OUTPUTFILES.getloc_GenotypeStatus()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
    systemcall += "Animal_GenoPheno_Status_"+stringseednumber; system(systemcall.c_str());
    if((SimParameters.getSelection() == "tbv" && SimParameters.getDamRepl() == 1.0 && SimParameters.getSireRepl() == 1.0) || (SimParameters.getSelection() == "phenotype") || (SimParameters.getSelection() == "ebv" && SimParameters.getDamRepl() == 1.0 && SimParameters.getSireRepl() == 1.0))
    {
        systemcall = "mv "+OUTPUTFILES.getloc_ExpRealDeltaG()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "ExpRealDeltaG_"+stringseednumber; system(systemcall.c_str());
    }
    /* Output Summary Haplofinder */
    if(SimParameters.gettraingen() > 0)
    {
        systemcall = "mv "+OUTPUTFILES.getloc_Summary_Haplofinder()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "Summary_Haplofinder_"+stringseednumber; system(systemcall.c_str());
    }
    /* Output Summary ROH */
    if((SimParameters.get_rohgeneration()).size() > 0)
    {
        systemcall = "mv "+OUTPUTFILES.getloc_Summary_ROHGenome_Freq()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "Summary_ROHGenome_Freq_"+stringseednumber; system(systemcall.c_str());
        systemcall = "mv "+OUTPUTFILES.getloc_Summary_ROHGenome_Length()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "Summary_ROHGenome_Length_"+stringseednumber; system(systemcall.c_str());
    }
    /* Output TrainReferenceGen */
    if(SimParameters.getOutputTrainReference() == "yes")
    {
        systemcall = "mv "+OUTPUTFILES.getloc_Amax_Output()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "AmaxGeneration_"+stringseednumber; system(systemcall.c_str());
        systemcall = "mv "+ OUTPUTFILES.getloc_TraitReference_Output()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "TrainReference_"+stringseednumber; system(systemcall.c_str());
    }
    /* Output Window Variance */
    if(SimParameters.getOutputWindowVariance() == "yes")
    {
        systemcall = "mv "+OUTPUTFILES.getloc_Windowadditive_Output()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "WindowAdditiveVariance_"+stringseednumber; system(systemcall.c_str());
        systemcall = "mv "+OUTPUTFILES.getloc_Windowdominance_Output()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "WindowDominanceVariance_"+stringseednumber; system(systemcall.c_str());
    }
    /* Output LD Metrics */
    if(SimParameters.getLDDecay() == "yes")
    {
        systemcall = "mv "+OUTPUTFILES.getloc_LD_Decay()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "LD_Decay_"+stringseednumber; system(systemcall.c_str());
        systemcall = "mv "+OUTPUTFILES.getloc_QTL_LD_Decay()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "QTL_LD_Decay_"+stringseednumber; system(systemcall.c_str());
        systemcall = "mv "+OUTPUTFILES.getloc_Phase_Persistance()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "Phase_Persistance_"+stringseednumber; system(systemcall.c_str());
        systemcall = "mv "+OUTPUTFILES.getloc_Phase_Persistance_Outfile()+" "+path+"/"+SimParameters.getOutputFold()+"/replicates/";
        systemcall += "Phase_Persistance_Generation_"+stringseednumber; system(systemcall.c_str());
    }
}


