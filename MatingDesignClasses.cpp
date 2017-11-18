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

#include "HaplofinderClasses.h"
#include "Animal.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"

using namespace std;

/********************/
/* Mating Functions */
/********************/
void randommating(vector <MatingClass> &matingindividuals, vector <Animal> &population, parameters &SimParameters);
void randomavoidance(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, string Pheno_Pedigree_File, parameters &SimParameters,ostream& logfileloc);
void minimizepedigree(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, string Pheno_Pedigree_File, parameters &SimParameters,ostream& logfileloc);
void minimizegenomic(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, double* M, float scale, parameters &SimParameters,ostream& logfileloc);
void minimizegenomic_maf(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, double* M, float scale, parameters &SimParameters,ostream& logfileloc);
void minimizeroh(vector <MatingClass> &matingindividuals, vector <Animal> &population,vector < hapLibrary > &haplib, string matingscenario, parameters &SimParameters,ostream& logfileloc);
void assortativemating(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario,parameters &SimParameters ,ostream& logfileloc);
void geneticvalueindex(vector <MatingClass> &matingindividuals,vector <Animal> &population,string tempmatescen,parameters &SimParameters,ostream& logfileloc);
void genetic_inbreedvalueindex(vector <MatingClass> &matingindividuals, vector <Animal> &population,string tempmatescen,parameters &SimParameters,string Pheno_Pedigree_File,double* M, float scale,vector < hapLibrary > &haplib,ostream& logfileloc);
void generate2traitindex(double* mate_value_matrix1, double* mate_value_matrix2, double* mate_index_matrix,parameters &SimParameters, vector <int> const &sireIDs, vector <int> const &damIDs, vector <double> &returnweights);


/*********************************/
/* Relationship Matrix Functions */
/*********************************/
void pedigree_relationship(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship);
void pedigree_relationship_Colleau(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship);
void grm_noprevgrm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);
void generaterohmatrix(vector <Animal> &population,vector < hapLibrary > &haplib,vector <int> const &parentID, double* _rohrm);
void matinggrm_maf(parameters &SimParameters,vector < string > &genotypes,double* output_grm,ostream& logfileloc);


/*********************/
/* Mating Algorithms */
/*********************/
void adaptivesimuanneal(vector <MatingClass> &matingindividuals, double *mate_value_matrix, vector <int> const &sireIDs, vector <int> const &damIDs, string direction,parameters &SimParameters);
void Kuhn_Munkres_Assignment(vector <MatingClass> &matingindividuals, double *mate_value_matrix,vector <int> const &sireIDs, vector <int> const &damIDs, string direction);
void sequentialselection(vector <MatingClass> &matingindividuals, double *mate_value_matrix,vector <int> const &sireIDs, vector <int> const &damIDs, string direction);
void geneticalgorithm(vector <MatingClass> &matingindividuals, double *mate_value_matrix,vector <int> const &sireIDs, vector <int> const &damIDs, string direction,parameters &SimParameters);




////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Class Functions       ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
MatingClass::MatingClass()
{
    ID = -1; AnimalType = -1; MateIDs = std::vector<int>(0); Matings = -1; OwnIndex = std::vector<int>(0); MateIndex = std::vector<int>(0);
}
MatingClass::MatingClass(int animid, int type, std::vector<int> mateids, int mates, std::vector<int> ownind, std::vector<int> mateindex)
{
    ID = animid; AnimalType = type; MateIDs = mateids; Matings = mates; OwnIndex = ownind; MateIndex = mateindex;
}
MatingClass::~MatingClass(){}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////     Catch all Functions    ////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////
//// Figure out Age Distribution ////
/////////////////////////////////////
void agedistribution(vector <Animal> &population,vector <int> &MF_AgeClass, vector <int> &M_AgeClass, vector <int> &MF_AgeID)
{
    for(int i = 0; i < population.size(); i++)
    {
        //if(i == 0){cout << population[i].getMatings() << endl;}
        population[i].ZeroOutMatings();                                 /* Before figuring out number of mating zero out last generations */
        //if(i == 0){cout << population[i].getMatings() << endl;}
        int temp = population[i].getAge();
        if(MF_AgeClass.size() != temp)                                  /* Increase the size of both MF_AgeClass and M_AgeClass to appropriate age length */
        {
            while(MF_AgeClass.size() < temp){MF_AgeClass.push_back(0); M_AgeClass.push_back(0);}
        }
        MF_AgeClass[temp - 1] += 1;                                     /* Adds age to current list */
        if(population[i].getSex() == 0){temp = population[i].getAge(); M_AgeClass[temp - 1] += 1;}
    }
    for(int i = 0; i < M_AgeClass.size(); i++){MF_AgeID.push_back(i+1);}
    //cout << MF_AgeClass.size() << " " << M_AgeClass.size() << endl;
    //for(int i = 0; i < M_AgeClass.size(); i++){cout <<"Age "<<i+1<<" "<<MF_AgeClass[i]<<" -- "<<M_AgeClass[i]<<" -- "<<MF_AgeID[i]<<endl;}
}
/////////////////////////////////////
//// Output Summary Statistics   ////
/////////////////////////////////////
void outputlogsummary(vector <Animal> &population, vector<int> &M_AgeClass, vector<int> &MF_AgeID,ostream& logfileloc)
{
    vector <double> CountMatingsClass(M_AgeClass.size(),0);
    /* Generate output for logfile */
    int mincount = 0; int maxcount = 0; int mincountid = 0; int maxcountid = 0;
    for(int i = 0; i < population.size(); i++)
    {
        int temp = population[i].getAge();
        if(population[i].getSex() == 0)
        {
            int tempa = population[i].getMatings();                                 /* grab current number of matings */
            CountMatingsClass[temp - 1] = CountMatingsClass[temp - 1 ] + tempa;    /* add number based on age */
        }
    }
    logfileloc << "       - Sire Breeding Age Distribution: " << endl;
    for(int i = 0; i < M_AgeClass.size(); i++)
    {
        if(M_AgeClass[i]>0){logfileloc<< "           - Age "<<i+1<<" Number Sires: "<<M_AgeClass[i]<<" and Number of Matings: "<<CountMatingsClass[i]<<endl;}
    }
    int CheckMatings = 0; int CheckSire = 0; int CheckDam = 0;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getSex() == 0){CheckSire += population[i].getMatings();}
        if(population[i].getSex() == 1){CheckDam += population[i].getMatings();}
        CheckMatings += population[i].getMatings();
    }
    logfileloc << "       - Total Matings: " << CheckMatings << "; Sire Matings: " << CheckSire << "; Dam Matings: " << CheckDam << endl;
}
/////////////////////////////////////
//// Summary of Mating Design  //////
/////////////////////////////////////
void summarymatingdesign(vector <MatingClass> &matingindividuals, vector <double> &summarystats, double *mate_value_matrix, vector <int> const &sireIDs, vector <int> const &damIDs)
{
    string firstindyet = "NO";
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 1)
        {
            if((matingindividuals[i].get_mateIDs()).size() != 1){cout << endl << "Female mates greater than 1" << endl; exit (EXIT_FAILURE);}
            //cout << matingindividuals[i].getID_MC() << " " << (matingindividuals[i].get_mateIDs())[0] << " --- ";
            int rowindex, colindex; int searchlocation = 0;
            while(1)
            {
                if(damIDs[searchlocation] == matingindividuals[i].getID_MC()){colindex = searchlocation; break;}
                if(damIDs[searchlocation] != matingindividuals[i].getID_MC()){searchlocation++;}
            }
            //cout << damIDs[searchlocation] << " " << colindex << endl;
            searchlocation = 0;
            while(1)
            {
                if(sireIDs[searchlocation] == (matingindividuals[i].get_mateIDs())[0]){rowindex = searchlocation; break;}
                if(sireIDs[searchlocation] != (matingindividuals[i].get_mateIDs())[0]){searchlocation++;}
            }
            //cout << sireIDs[searchlocation] << " " << rowindex << endl;
            double temp = mate_value_matrix[(rowindex*damIDs.size())+colindex];
            if(firstindyet == "NO"){summarystats[1] = temp; summarystats[2] = temp; firstindyet = "YES";}
            summarystats[0] += temp;
            if(summarystats[1] == 0.0){summarystats[1] = temp;}
            if(temp < summarystats[1]){summarystats[1] = temp;}
            if(temp > summarystats[2]){summarystats[2] = temp;}
            if(temp >= 0.125){summarystats[3] += 1;}
            if(temp >= 0.25){summarystats[4] += 1;}
            if(temp >= 0.50){summarystats[5] += 1;}
            //cout << temp << " ";
        }
    }
    //cout << endl;
    //for(int i = 0; i < 6; i++){cout << summarystats[i] << endl;}
    //exit (EXIT_FAILURE);
}
///////////////////////////////////////////////
//// Update Matings Based on New Mates   //////
///////////////////////////////////////////////
void updatematings(vector <MatingClass> &matingindividuals, vector < int > const &sire_mate_column,vector <int> const &sireIDs, vector <int> const &damIDs)
{
    /* First clear old mate ids from each class along with index locations */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        //cout << (matingindividuals[i].get_OwnIndex()).size() << " " << (matingindividuals[i].get_mateIDs()).size() << endl;
        matingindividuals[i].clear_OwnIndex(); matingindividuals[i].clear_MateIDs(); matingindividuals[i].clear_MateIndex();
        //cout << (matingindividuals[i].get_OwnIndex()).size() << " " << (matingindividuals[i].get_mateIDs()).size() << endl;
    }
    //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
    //cout << endl << endl;
    /* loop through and push_back updated ID to MateIDs() */
    for(int i = 0; i < sire_mate_column.size(); i++)
    {
        if(sire_mate_column[i] != -5)
        {
            //cout << i << " " << sireIDs[i] << " " << damIDs[sire_mate_column[i]] << endl;
            /* Find sire and update */
            int searchlocation = 0;
            while(1)
            {
                if(sireIDs[i] == (matingindividuals[searchlocation].getID_MC()))
                {
                    //cout << matingindividuals[searchlocation].getID_MC() << endl;
                    matingindividuals[searchlocation].add_ToMates(damIDs[sire_mate_column[i]]);
                    //cout << (matingindividuals[searchlocation].get_mateIDs())[0] << endl;
                    break;
                }
                if(sireIDs[i] != (matingindividuals[searchlocation].getID_MC())){searchlocation++;}
            }
            searchlocation = 0;
            while(1)
            {
                if(damIDs[sire_mate_column[i]] == matingindividuals[searchlocation].getID_MC())
                {
                    //cout << matingindividuals[searchlocation].getID_MC() << endl;
                    matingindividuals[searchlocation].add_ToMates(sireIDs[i]);
                    //cout << (matingindividuals[searchlocation].get_mateIDs())[0] << endl;
                    break;
                }
                if(damIDs[sire_mate_column[i]] != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
            }
        }
    }
    int doublecheckmatesmale = 0; int doublecheckmatesfemale = 0;
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0){doublecheckmatesmale += (matingindividuals[i].get_mateIDs()).size();}
        if(matingindividuals[i].getType_MC() == 1){doublecheckmatesfemale += (matingindividuals[i].get_mateIDs()).size();}
    }
    if(doublecheckmatesmale != doublecheckmatesfemale){cout << endl << "Mating Pairs Don't Match Up!" << endl; exit (EXIT_FAILURE);}
}
///////////////////////////////////////////////////////////////////
//// Update Matings in Population class Based on New Mates   //////
///////////////////////////////////////////////////////////////////
void updatematingindex(vector <MatingClass> &matingindividuals, vector <Animal> &population)
{
    for(int i = 0; i < population.size(); i++)
    {
        //cout << population[i].getID() << " " << population[i].getMatings() << " ";
        int searchlocation = 0;
        while(1)
        {
            if(matingindividuals[searchlocation].getID_MC() == population[i].getID())
            {
                //cout << matingindividuals[searchlocation].getID_MC() << " ";
                //cout << matingindividuals[searchlocation].getMatings_MC() << " ";
                population[i].UpdateMatings((matingindividuals[searchlocation].get_mateIDs()).size());
                //cout << population[i].getID() << " " << population[i].getMatings() << endl;
                break;
            }
            if(matingindividuals[searchlocation].getID_MC() != population[i].getID()){searchlocation++;}
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////     Primary function to generate mating design       ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////
// Figures out which mating scenario to utilize //
//////////////////////////////////////////////////
string choosematingscenario(parameters &SimParameters, string tempselectionvector)
{
    string tempmatingscenario;
    if(SimParameters.getMating() == "random" || tempselectionvector == "random"){tempmatingscenario = "RANDOM";}
    if(SimParameters.getMating() == "random5" && tempselectionvector != "random"){tempmatingscenario = "RANDOM5";}
    if(SimParameters.getMating() == "random25" && tempselectionvector != "random"){tempmatingscenario = "RANDOM25";}
    if(SimParameters.getMating() == "random125" && tempselectionvector != "random"){tempmatingscenario = "RANDOM125";}
    if(SimParameters.getMating() == "minPedigree" && tempselectionvector != "random"){tempmatingscenario = "MINPEDIGREE";}
    if(SimParameters.getMating() == "minGenomic" && tempselectionvector != "random"){tempmatingscenario = "MINGENOMIC";}
    if(SimParameters.getMating() == "minROH" && tempselectionvector != "random"){tempmatingscenario = "MINROH";}
    if(SimParameters.getMating() == "pos_assort" && tempselectionvector != "random"){tempmatingscenario = "POS_ASSORTATIVE";}
    if(SimParameters.getMating() == "neg_assort" && tempselectionvector != "random"){tempmatingscenario = "NEG_ASSORTATIVE";}
    if(SimParameters.getMating() == "minGenomic_maf" && tempselectionvector != "random"){tempmatingscenario = "MIN_GENOMICMAF";}
    return tempmatingscenario;
}
////////////////////////////////////////////////
// Based on index values choose mating pairs //
////////////////////////////////////////////////
void indexmatingdesign(vector <MatingClass> &matingindividuals, vector <Animal> &population, vector < hapLibrary > &haplib, parameters &SimParameters, string Pheno_Pedigree_File, double* M,float scale,ostream& logfileloc)
{
    string tempmatescen;
    /* If size of both equal 1 then it is just a function of breeding value */
    if((SimParameters.get_indexweights()).size() == 1 && (SimParameters.get_indexparameters()).size() == 1 && SimParameters.getMating() == "index")
    {
        //for(int i = 0; i < (SimParameters.get_indexweights()).size(); i++)
        //{
        //    cout << (SimParameters.get_indexweights())[i] << " " << (SimParameters.get_indexparameters())[i] << endl;
        //}
        if((SimParameters.get_indexparameters())[0] == "ebv"){tempmatescen = "INDEX_EBV";}
        if((SimParameters.get_indexparameters())[0] == "true_bv"){tempmatescen = "INDEX_TBV";}
        if((SimParameters.get_indexparameters())[0] == "phenotype"){tempmatescen = "INDEX_PHEN";}
        logfileloc << "       - Parent mated based on an index value comprised of " << (SimParameters.get_indexparameters())[0] << " only." << endl;
    }
    if((SimParameters.get_indexweights()).size() == 2 && (SimParameters.get_indexparameters()).size() == 2 && SimParameters.getMating() == "index")
    {
        //for(int i = 0; i < (SimParameters.get_indexweights()).size(); i++)
        //{
        //    cout << (SimParameters.get_indexweights())[i] << " " << (SimParameters.get_indexparameters())[i] << endl;
        //}
        if((SimParameters.get_indexparameters())[0]=="ebv" && (SimParameters.get_indexparameters())[1]=="pedigree"){tempmatescen = "INDEX_EBV_PED";}
        if((SimParameters.get_indexparameters())[0]=="ebv" && (SimParameters.get_indexparameters())[1]=="genomic"){tempmatescen = "INDEX_EBV_GEN";}
        if((SimParameters.get_indexparameters())[0]=="ebv" && (SimParameters.get_indexparameters())[1]=="ROH"){tempmatescen = "INDEX_EBV_ROH";}
        if((SimParameters.get_indexparameters())[0]=="true_bv" && (SimParameters.get_indexparameters())[1]=="pedigree"){tempmatescen = "INDEX_TBV_PED";}
        if((SimParameters.get_indexparameters())[0]=="true_bv" && (SimParameters.get_indexparameters())[1]=="genomic"){tempmatescen = "INDEX_TBV_GEN";}
        if((SimParameters.get_indexparameters())[0]=="true_bv" && (SimParameters.get_indexparameters())[1]=="ROH"){tempmatescen = "INDEX_TBV_ROH";}
        if((SimParameters.get_indexparameters())[0]=="phenotype" && (SimParameters.get_indexparameters())[1]=="pedigree"){tempmatescen = "INDEX_PHEN_PED";}
        if((SimParameters.get_indexparameters())[0]=="phenotype" && (SimParameters.get_indexparameters())[1]=="genomic"){tempmatescen = "INDEX_PHEN_GEN";}
        if((SimParameters.get_indexparameters())[0]=="phenotype" && (SimParameters.get_indexparameters())[1]=="ROH"){tempmatescen = "INDEX_PHEN_ROH";}
    }
    //cout << tempmatescen << endl;
    /* First fill matingclass with male and females */
    int malecandidates= 0; int femalecandidates = 0;
    for(int i = 0; i < population.size(); i++)
    {
        //cout << population[i].getID() << " " << population[i].getSex() << " " << population[i].getMatings() << endl;
        if(population[i].getSex() == 0){malecandidates++;}
        if(population[i].getSex() == 1){femalecandidates++;}
        /* Fill Mating Class vector */
        MatingClass temp(population[i].getID(),population[i].getSex(),vector<int>(0), population[i].getMatings(),vector<int>(0),vector<int>(0));
        matingindividuals.push_back(temp);
        //matingindividuals[i].showdata();
    }
    /* Given number of females figure out max number of matings a sire is allowed */
    int maximummatings = femalecandidates * SimParameters.getmaxsireprop();
    logfileloc << "       - Maximum number of times a sire can be mated " << maximummatings << "." << endl;
    /* Update mate number */
    int checkmale = 0; int checkfemale = 0;
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0)           /* Males max matings */
        {
            matingindividuals[i].UpdateMateNumber(maximummatings); checkmale += matingindividuals[i].getMatings_MC();
        }
        if(matingindividuals[i].getType_MC() == 1)           /* Females a value of 1 */
        {
            matingindividuals[i].UpdateMateNumber(1); checkfemale += matingindividuals[i].getMatings_MC();
        }
    }
    logfileloc << "         - Maximum possible matings for sires: " << checkmale << "." << endl;
    logfileloc << "         - Maximum possible matings for dams: " << checkfemale << "." << endl;
    if(tempmatescen == "INDEX_EBV" || tempmatescen == "INDEX_TBV" || tempmatescen == "INDEX_PHEN")
    {
        /* Generate mating pairs based on ebv only */
        geneticvalueindex(matingindividuals,population,tempmatescen,SimParameters,logfileloc);
    }
    if(tempmatescen == "INDEX_EBV_PED" || tempmatescen == "INDEX_EBV_GEN" || tempmatescen == "INDEX_EBV_ROH" || tempmatescen == "INDEX_TBV_PED" ||
       tempmatescen == "INDEX_TBV_GEN" || tempmatescen == "INDEX_TBV_ROH" || tempmatescen == "INDEX_PHEN_PED" || tempmatescen == "INDEX_PHEN_GEN" ||
       tempmatescen == "INDEX_PHEN_ROH")
    {
        /* Generate mating pairs based on genetic value and inbreeding */
        genetic_inbreedvalueindex(matingindividuals,population,tempmatescen,SimParameters,Pheno_Pedigree_File,M,scale,haplib,logfileloc);
    }
}

////////////////////////////////////////////////
// Given mating scenario chooses mating pairs //
////////////////////////////////////////////////
void generatematingpairs(vector <MatingClass> &matingindividuals, vector <Animal> &population, vector < hapLibrary > &haplib,parameters &SimParameters, string matingscenario, string Pheno_Pedigree_File, double* M,float scale, ostream& logfileloc)
{
    //cout << matingscenario << " " << Pheno_Pedigree_File << endl;
    /* First fill matingclass with male and females */
    int numberofmatings = 0;
    for(int i = 0; i < population.size(); i++)
    {
        //cout << population[i].getID() << " " << population[i].getSex() << " " << population[i].getMatings() << endl;
        if(population[i].getSex() == 0){numberofmatings += population[i].getMatings();}
        /* Fill Mating Class vector */
        MatingClass temp(population[i].getID(),population[i].getSex(),vector<int>(0,0),population[i].getMatings(),vector<int>(0,0),vector<int>(0,0));
        matingindividuals.push_back(temp);
        //matingindividuals[i].showdata();
    }
    if(numberofmatings != SimParameters.getDams())
    {
        cout << endl << "Total number of sire matings does not match up with number of females. Ensure males to females is a whole number!!" << endl;
        exit (EXIT_FAILURE);
    }
    /* Put in mating class now depending on what design you have figure out mating pairs now */
    if(matingscenario == "RANDOM")
    {
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Parents mated randomly." << endl;
        randommating(matingindividuals,population,SimParameters);
    }
    if(matingscenario == "RANDOM5" || matingscenario == "RANDOM25" || matingscenario == "RANDOM125")
    {
        randommating(matingindividuals,population,SimParameters); /* First do random mating then do avoidance */
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Mating was based on avoiding mating pairs with a relationship greater than ";
        if(matingscenario == "RANDOM5"){logfileloc << 0.50 << ", otherwise random mating" << endl;}
        if(matingscenario == "RANDOM25"){logfileloc << 0.25 << ", otherwise random mating" << endl;}
        if(matingscenario == "RANDOM125"){logfileloc << 0.125 << ", otherwise random mating" << endl;}
        randomavoidance(matingindividuals,population,matingscenario,Pheno_Pedigree_File,SimParameters,logfileloc);
    }
    if(matingscenario == "MINPEDIGREE")
    {
        randommating(matingindividuals,population,SimParameters); /* First do random mating then do avoidance */
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Minimize coancestries based on pedigree-based relationship matrix." << endl;
        minimizepedigree(matingindividuals,population,matingscenario,Pheno_Pedigree_File,SimParameters,logfileloc);
    }
    if(matingscenario == "MINGENOMIC")
    {
        randommating(matingindividuals,population,SimParameters); /* First do random mating then do avoidance */
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Minimize coancestries based on genomic-based relationship matrix." << endl;
        minimizegenomic(matingindividuals,population,matingscenario,M,scale,SimParameters,logfileloc);
    }
    if(matingscenario == "MINROH")
    {
        randommating(matingindividuals,population,SimParameters); /* First do random mating then do avoidance */
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Minimize coancestries based on run of homozygosity (ROH)-based relationship matrix." << endl;
        minimizeroh(matingindividuals,population,haplib,matingscenario,SimParameters,logfileloc);
    }
    if(matingscenario == "MIN_GENOMICMAF")
    {
        randommating(matingindividuals,population,SimParameters); /* First do random mating then do avoidance */
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Minimize coancestries based on genomic_maf-based relationship matrix." << endl;
        minimizegenomic_maf(matingindividuals,population,matingscenario,M,scale,SimParameters,logfileloc);
    }
    if(matingscenario == "POS_ASSORTATIVE" || matingscenario == "NEG_ASSORTATIVE")
    {
        logfileloc << "       - Size of mating class: " << matingindividuals.size() << endl;
        logfileloc << "       - Animals are mated based on ";
        if(matingscenario == "POS_ASSORTATIVE"){logfileloc << "positive assortative mating." << endl;}
        if(matingscenario == "NEG_ASSORTATIVE"){logfileloc << "negative assortative mating." << endl;}
        assortativemating(matingindividuals,population,matingscenario,SimParameters,logfileloc);
    }   
    //cout << matingindividuals[0].getMatings_MC() << endl;
    //cout << (matingindividuals[0].get_mateIDs()).size() << endl;
    //matingindividuals[i].add_ToMates(1); matingindividuals[i].add_ToMates(2);
    //cout << (matingindividuals[i].get_mateIDs()).size() << " " << matingindividuals[i].get_mateIDs()[1] << endl;
    //cout << matingindividuals.size() << "---" << endl;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////        Mating Scenario Funcions        ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
//// Generate Number of Mates by Beta Distribution   ////
/////////////////////////////////////////////////////////
void betadistributionmates(vector <Animal> &population,parameters &SimParameters, int M_NumberClassg0, vector <double> const &number, vector <int> const &M_AgeClass,vector <int> const &MF_AgeID)
{
    mt19937 gen(SimParameters.getSeed());
    int i = 0;                                                  /* Counter to determine how many Parity 0 updated */
    int SireCount = 0;                                          /* Total Number of Sires available */
    int DamCount = 0;
    /* Give first parity sires at least one mating and zero out anything older */
    for(int j = 0; j < population.size(); j++)
    {
        if(population[j].getSex() == 0 && population[j].getAge() == 1)  /* First Parity so give it at least one mating */
        {
            population[j].UpdateMatings(1); SireCount++; i++;
        }
        if(population[j].getSex() == 0 && population[j].getAge() > 1)   /* Older animal so reset to zero and Beta Parameters will determine */
        {
            population[j].UpdateMatings(0); SireCount++;
        }
        if(population[j].getSex() == 1){DamCount++;}
        //cout << population[j].getSex() << " " << population[j].getAge() << " " << population[j].getMatings() << endl;
    }
    //cout << i << " " << SireCount << " " << DamCount << endl;
    int MatingPairs = DamCount - i;     /* Number of Mating Pairs left over after giving first parity sires 1 */
    vector <double> MatingByAge(M_NumberClassg0,0.0);
    for(int i = 0; i < M_NumberClassg0; i++)
    {
        if(i == 0){MatingByAge[i] = (1.0 / double(M_NumberClassg0));}
        if(i > 0){MatingByAge[i] = MatingByAge[i-1] + (1.0 / double(M_NumberClassg0));}
    }
    //for(int i = 0; i < M_NumberClassg0; i++){cout << MatingByAge[i] << " ";}
    //cout << endl;
    vector <double> MatingProp(M_NumberClassg0,0.0);            /* proportion of gametes that belong to the particular age class */
    vector <int> NumberGametes(M_NumberClassg0,0.0);            /* Number of gamates that are derived from the particular age class */
    double RunningTotalProp = 0.0;                              /* Used to tally up how far have gone and to subtract of to create interval */
    int RunningTotalGame = 0;                                   /* Used to keep track of total gametes */
    for(int i = 0; i < M_NumberClassg0 - 1; i++)
    {
        int j = 0;
        while(1)            /* Loops across number values from Beta until value is reached then saves value */
        {
            if(MatingByAge[i] < number[j])
            {
                if(i == 0)
                {
                    MatingProp[i] = j / 10000.0;                                    /* Determine proportion that fall in interval */
                    NumberGametes[i] = (MatingProp[i] * MatingPairs) + 0.5;
                }
                if(i > 0)
                {
                    MatingProp[i] = (j / 10000.0) - RunningTotalProp;                /* Determine proportion that fall in interval */
                    NumberGametes[i] = (MatingProp[i] * MatingPairs) + 0.5;
                }
                RunningTotalProp += MatingProp[i];
                RunningTotalGame += NumberGametes[i];
                break;
            }
            j++;
        }
    }
    MatingProp[M_NumberClassg0 - 1] = 1 - RunningTotalProp;
    NumberGametes[M_NumberClassg0 - 1] = MatingPairs - RunningTotalGame;                 /* Remaining gametes have to end up in last age class */
    //for(int i = 0; i < M_NumberClassg0; i++){cout << MatingProp[i] << " " << NumberGametes[i] << endl;}
    int age = 0; int sireageloc = 0;
    while(age < M_NumberClassg0)
    {
        std::uniform_real_distribution<double> distribution(0,1);
        if(M_AgeClass[sireageloc] > 0)
        {
            //cout << NumberGametes[age] << " " << M_AgeClass[sireageloc] << endl;
            int MinGametes = NumberGametes[age] / M_AgeClass[sireageloc];
            int SurplusGametes = NumberGametes[age] - (MinGametes * M_AgeClass[sireageloc]);    /* up to this number will get 1 extra */
            //cout << MinGametes << " " << SurplusGametes << endl;
            vector <int> ID(M_AgeClass[sireageloc],0);                                          /* Stores ID */
            vector <double> UniformValue(M_AgeClass[sireageloc],0);                             /* Stores random Variable */
            int AnimalCounter = 0;
            for(int an = 0; an < population.size(); an++)
            {
                if(population[an].getSex() == 0 && population[an].getAge() ==  MF_AgeID[sireageloc])    /* Add one because loop starts at 0 */
                {
                    ID[AnimalCounter] = population[an].getID();
                    UniformValue[AnimalCounter] = distribution(gen);
                    AnimalCounter++;
                    //cout << population[an].getSex() << " " << population[an].getAge() << endl;
                }
            }
            /* Sort values based on Uniform value */
            int temp;
            double tempa;
            for(int i = 0; i < M_AgeClass[sireageloc] - 1; i++)
            {
                for(int j=i+1; j < M_AgeClass[sireageloc]; j++)
                {
                    if(UniformValue[i] > UniformValue[j])
                    {
                        /* put i values in temp variables */
                        temp = ID[i]; tempa = UniformValue[i];
                        /* swap lines */
                        ID[i] = ID[j]; UniformValue[i] = UniformValue[j];
                        /* put temp values in 1 backward */
                        ID[j] = temp; UniformValue[j] = tempa;
                    }
                }
            }
            /* sort out gametes to animals within a given parity */
            int j = 0; int ExtraGam = 1;
            while(j < M_AgeClass[sireageloc])
            {
                int i = 0;
                while(1)
                {
                    if(population[i].getSex() == 0 && population[i].getAge() == MF_AgeID[sireageloc] && population[i].getID() == ID[j])
                    {
                        if(SurplusGametes == 0){population[i].UpdateMatings(MinGametes);}
                        if(ExtraGam <= SurplusGametes && SurplusGametes != 0){population[i].UpdateMatings(MinGametes + 1);}
                        if(ExtraGam > SurplusGametes && SurplusGametes != 0){population[i].UpdateMatings(MinGametes);}
                        ExtraGam++; j++; break;
                    }
                    i++;
                }
            }
            //for(int i = 0; i < population.size(); i++)
            //{
            //    if(population[i].getAge() == MF_AgeID[sireageloc])
            //    {
            //        cout << population[i].getSex() << " " << population[i].getAge() << " " << population[i].getMatings() << endl;
            //    }
            //}
            age++;
        }
        sireageloc++;
    }
}
/////////////////////////////////////////////////////////
////                Random Mating                    ////
/////////////////////////////////////////////////////////
void randommating(vector <MatingClass> &matingindividuals, vector <Animal> &population, parameters &SimParameters)
{
    /* create two vectors one for sires and one for dams; then generate random uniform value for females sort then mate sorted */
    /* list to respective dams */
    mt19937 gen(SimParameters.getSeed()); uniform_real_distribution<double> distribution(0,1);
    vector < int > sireIDs; vector < int > damIDs; vector < double > random_value; vector < int > parentID;
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC());}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); random_value.push_back(distribution(gen));}
        }
    }
    
    /* Check to see if dams are getting same random deviate values across scenarios */
    //std::ofstream output1("CHECK", std::ios_base::app | std::ios_base::out);
    //for(int i = 0; i < damIDs.size(); i++){output1 << sireIDs[i] << "-" << damIDs[i] << " " << random_value[i] << endl;}
    //cout << endl << endl;
    double temp_uniform; int temp_id;
    for(int i = 0; i < damIDs.size()-1; i++)
    {
        for(int j=i+1; j < damIDs.size(); j++)
        {
            if(random_value[i] > random_value[j])
            {
                temp_id = damIDs[i]; temp_uniform = random_value[i];
                damIDs[i] = damIDs[j]; random_value[i] = random_value[j];
                damIDs[j] = temp_id; random_value[j] = temp_uniform;
            }
        }
    }
    /* Now loop through and fill matingindividuals matingID vector */
    for(int i = 0; i < damIDs.size(); i++)
    {
        //cout << sireIDs[i] << " " << damIDs[i] << endl;
        int searchlocation = 0; string foundsire = "NO"; string founddam = "NO";
        while(1) /* First find sire for a given mating pair */
        {
            if(sireIDs[i] == matingindividuals[searchlocation].getID_MC() && foundsire == "NO")
            {
                //cout << "Sire: " << (matingindividuals[searchlocation].get_mateIDs()).size() << " " << sireIDs[i] << " ";
                matingindividuals[searchlocation].add_ToMates(damIDs[i]); foundsire = "YES";
                //cout <<(matingindividuals[searchlocation].get_mateIDs()).size()<<" "<< foundsire<< "\t";
            }
            if(damIDs[i] == matingindividuals[searchlocation].getID_MC() && founddam == "NO")
            {
                //cout << "Dam: " << (matingindividuals[searchlocation].get_mateIDs()).size() << " " << damIDs[i] << " ";
                matingindividuals[searchlocation].add_ToMates(sireIDs[i]); founddam = "YES";
                //cout << (matingindividuals[searchlocation].get_mateIDs()).size() << " " << founddam << "\t";
            }
            if(foundsire == "YES" && founddam == "YES"){break; /* cout << endl; */}
            searchlocation++;
        }
    }
    //for(int i = 0; i < matingindividuals.size(); i++)
    //{
    //    cout << matingindividuals[i].getID_MC() << " " << matingindividuals[i].getType_MC() << " ";
    //    cout << matingindividuals[i].getMatings_MC() << " " << (matingindividuals[i].get_mateIDs()).size() << endl;
    //}
}
/////////////////////////////////////////////////////////
////          Random Avoidance Mating                ////
/////////////////////////////////////////////////////////
void randomavoidance(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, string Pheno_Pedigree_File, parameters &SimParameters,ostream& logfileloc)
{
    string direction = "minimum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector < int > parentID;                                                                        /* ID of parents */
    vector < int > sireIDs; vector < int > sireA;                                                   /* row in A and associated ID for sire */
    vector < int > damIDs; vector < int > damA;                                                     /* row in A and associated ID for dam */
    double* subsetrelationship = new double[matingindividuals.size()*matingindividuals.size()];     /* Used to store subset of relationship matrix */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC()); sireA.push_back(i);}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); damA.push_back(i);}
        }
        parentID.push_back(matingindividuals[i].getID_MC());
    }
    //cout << sireIDs.size() << " " << sireA.size() << " " << damIDs.size() << " " << damA.size() << endl;
    //for(int i = 0; i < sireA.size(); i++){cout << sireIDs[i] << "-" << sireA[i] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < damA.size(); i++){cout << damIDs[i] << "-" << damA[i] << " ";}
    //cout << endl << endl;
    /* Fill in index array to make it easer for all other functions */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        //cout << i << ": " << matingindividuals[i].getID_MC() << " " << matingindividuals[i].getType_MC() << " ";
        if(matingindividuals[i].getType_MC() == 0)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC() && searchlocation < (sireIDs.size()))
            {
                if(searchlocation > (sireIDs.size()-1)){break;}
                if(matingindividuals[i].getID_MC() == sireIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                    if(searchlocation >= sireIDs.size()){break;}
                }
                if(matingindividuals[i].getID_MC() != sireIDs[searchlocation]){searchlocation++;}
                //cout << searchlocation << " ";
            }
        }
        if(matingindividuals[i].getType_MC() == 1)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC() && searchlocation < (damIDs.size()))
            {
                if(matingindividuals[i].getID_MC() == damIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                    if(searchlocation >= damIDs.size()){break;}
                }
                if(matingindividuals[i].getID_MC() != damIDs[searchlocation]){searchlocation++;}
                //cout << searchlocation << " ";
            }
        }
        //cout << endl;
    }
    double* mate_value_matrix = new double[sireA.size() * damA.size()];                    /* mate allocation matrix based on pedigree */
    //pedigree_relationship(Pheno_Pedigree_File,parentID, subsetrelationship);                        /* Generate Relationship Matrix */
    pedigree_relationship_Colleau(Pheno_Pedigree_File,parentID, subsetrelationship);
    /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
    for(int i = 0; i < sireA.size(); i++)
    {
        for(int j = 0; j < damA.size(); j++){mate_value_matrix[(i*sireA.size())+j] = subsetrelationship[(sireA[i]*parentID.size())+damA[j]];}
    }
    //for(int i = (parentID.size()-10); i < parentID.size(); i++)
    //{
    //    for(int j = (parentID.size()-10); j < parentID.size(); j++){cout << subsetrelationship[(i*parentID.size())+j] << "\t";}
    //    cout << endl;
    //}
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << mate_value_matrix[(i*sireA.size())+j] << "\t";}
    //    cout << endl;
    //}
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
    string matingdesign = "YES";
    if(matingscenario == "RANDOM5")             /* Set all to zero less than 0.5 */
    {
        logfileloc << "       - Number of mating's over threshold (0.5) before mating design: " << summarystats[5] << endl;
        for(int i = 0; i < (sireA.size() * damA.size()); i++){if(mate_value_matrix[i] < 0.50){mate_value_matrix[i] = 0;}}
        if(summarystats[5] == 0){matingdesign = "NO";}
    }
    if(matingscenario == "RANDOM25")             /* Set all to zero less than 0.5 */
    {
        logfileloc << "       - Number of mating's over threshold (0.25) before mating design: " << summarystats[4] << endl;
        for(int i = 0; i < (sireA.size() * damA.size()); i++){if(mate_value_matrix[i] < 0.25){mate_value_matrix[i] = 0;}}
        if(summarystats[4] == 0){matingdesign = "NO";}
    }
    if(matingscenario == "RANDOM125")             /* Set all to zero less than 0.5 */
    {
        logfileloc << "       - Number of mating's over threshold (0.125) before mating design: " << summarystats[3] << endl;
        for(int i = 0; i < (sireA.size() * damA.size()); i++){if(mate_value_matrix[i] < 0.125){mate_value_matrix[i] = 0;}}
        if(summarystats[3] == 0){matingdesign = "NO";}
    }
    if(matingdesign == "YES")
    {
        for(int i = 0; i < 6; i++){summarystats[i] = 0.0;}
        if(SimParameters.getMatingAlg() == "simu_anneal")
        {
            logfileloc << "         - Optimization algorithm utilized was Adaptive Simulated Annealing." << endl;
            adaptivesimuanneal(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction,SimParameters);
        }
        if(SimParameters.getMatingAlg() == "linear_prog")
        {
            logfileloc << "         - Optimization algorithm utilized was Kuhn–Munkres Algorithm." << endl;
            Kuhn_Munkres_Assignment(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        if(SimParameters.getMatingAlg() == "sslr")
        {
            logfileloc << "         - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
            sequentialselection(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
        if(matingscenario == "RANDOM5"){logfileloc << "       - Number of mating's over threshold (0.5) after mating design: " << summarystats[5] << endl;}
        if(matingscenario == "RANDOM25"){logfileloc << "       - Number of mating's over threshold (0.25) after mating design: " << summarystats[4] << endl;}
        if(matingscenario == "RANDOM125"){logfileloc << "       - Number of mating's over threshold (0.125) after mating design: "<<summarystats[3] << endl;}
    }
    delete [] subsetrelationship; delete [] mate_value_matrix;
}
/////////////////////////////////////////////////////////
////          Minimize Based On Pedigree             ////
/////////////////////////////////////////////////////////
void minimizepedigree(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, string Pheno_Pedigree_File, parameters &SimParameters,ostream& logfileloc)
{
    string direction = "minimum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector < int > parentID;                                                                        /* ID of parents */
    vector < int > sireIDs; vector < int > sireA;                                                   /* row in A and associated ID for sire */
    vector < int > damIDs; vector < int > damA;                                                     /* row in A and associated ID for dam */
    double* subsetrelationship = new double[matingindividuals.size()*matingindividuals.size()];     /* Used to store subset of relationship matrix */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC()); sireA.push_back(i);}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); damA.push_back(i);}
        }
        parentID.push_back(matingindividuals[i].getID_MC());
    }
    //cout << sireIDs.size() << " " << sireA.size() << " " << damIDs.size() << " " << damA.size() << endl;
    //for(int i = 0; i < sireA.size(); i++){cout << sireIDs[i] << "-" << sireA[i] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < damA.size(); i++){cout << damIDs[i] << "-" << damA[i] << " ";}
    //cout << endl << endl;
    /* Fill in index array to make it easer for all other functions */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        //cout << matingindividuals[i].getID_MC() << " " << matingindividuals[i].getType_MC() << " ";
        if(matingindividuals[i].getType_MC() == 0)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == sireIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != sireIDs[searchlocation]){searchlocation++;}
            }
        }
        if(matingindividuals[i].getType_MC() == 1)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == damIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != damIDs[searchlocation]){searchlocation++;}
            }
        }
        //cout << endl;
    }
    double* mate_value_matrix = new double[sireA.size() * damA.size()];                    /* mate allocation matrix based on pedigree */
    for(int i = 0; i < sireA.size()*damA.size(); i++){mate_value_matrix[i] = 0.0;}
    //pedigree_relationship(Pheno_Pedigree_File,parentID, subsetrelationship);                        /* Generate Relationship Matrix */
    pedigree_relationship_Colleau(Pheno_Pedigree_File,parentID, subsetrelationship);
    /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
    for(int i = 0; i < sireA.size(); i++)
    {
        for(int j = 0; j < damA.size(); j++){mate_value_matrix[(i*sireA.size())+j] = subsetrelationship[(sireA[i]*parentID.size())+damA[j]];}
    }
    //for(int i = (parentID.size()-10); i < parentID.size(); i++)
    //{
    //   for(int j = (parentID.size()-10); j < parentID.size(); j++){cout << subsetrelationship[(i*parentID.size())+j] << "\t";}
    //    cout << endl;
    //}
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << mate_value_matrix[(i*sireA.size())+j] << "\t";}
    //   cout << endl;
    //}
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
    //for(int i = 0; i < summarystats.size(); i++){cout << summarystats[i] << endl;}
    string matingdesign = "YES";
    logfileloc << "       - Expected Progeny Level Inbreeding (Mean - Min - Max) prior to mating strategy: ";
    logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    if(summarystats[0] == 0.0){matingdesign = "NO";}
    if(matingdesign == "YES")
    {
        for(int i = 0; i < 6; i++){summarystats[i] = 0.0;}
        if(SimParameters.getMatingAlg() == "simu_anneal")
        {
            logfileloc << "         - Optimization algorithm utilized was Adaptive Simulated Annealing." << endl;
            adaptivesimuanneal(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction,SimParameters);
        }
        if(SimParameters.getMatingAlg() == "linear_prog")
        {
            logfileloc << "         - Optimization algorithm utilized was Kuhn–Munkres Algorithm." << endl;
            Kuhn_Munkres_Assignment(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        if(SimParameters.getMatingAlg() == "sslr")
        {
            logfileloc << "         - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
            sequentialselection(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
        logfileloc << "       - Expected Progeny Inbreeding Level (Mean - Min - Max) after to mating strategy: ";
        logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    }
    delete [] subsetrelationship; delete [] mate_value_matrix;
}
//////////////////////////////////////////////////////////
////          Minimize Based On Genomic               ////
//////////////////////////////////////////////////////////
void minimizegenomic(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, double* M, float scale, parameters &SimParameters,ostream& logfileloc)
{
    string direction = "minimum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector < int > parentID;                                                                        /* ID of parents */
    vector < string > genotypeparent;                                                               /* Genotype of parent */
    vector < int > sireIDs; vector < int > sireA;                                                   /* row in A and associated ID for sire */
    vector < int > damIDs; vector < int > damA;                                                     /* row in A and associated ID for dam */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC()); sireA.push_back(i);}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); damA.push_back(i);}
        }
        parentID.push_back(matingindividuals[i].getID_MC()); genotypeparent.push_back("");
    }
    //cout << sireIDs.size() << " " << sireA.size() << " " << damIDs.size() << " " << damA.size() << endl;
    //for(int i = 0; i < sireA.size(); i++){cout << sireIDs[i] << "-" << sireA[i] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < damA.size(); i++){cout << damIDs[i] << "-" << damA[i] << " ";}
    //cout << endl << endl;
    //cout << parentID.size() << " " << genotypeparent.size() << endl;
    /* Fill in index array to make it easer for all other functions */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        //cout << matingindividuals[i].getID_MC() << " " << matingindividuals[i].getType_MC() << " ";
        if(matingindividuals[i].getType_MC() == 0)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == sireIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != sireIDs[searchlocation]){searchlocation++;}
            }
        }
        if(matingindividuals[i].getType_MC() == 1)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == damIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != damIDs[searchlocation]){searchlocation++;}
            }
        }
        //cout << endl;
    }
    /* Now loop through population and get marker genotypes */
    for(int i = 0; i < parentID.size(); i++)
    {
        int searchlocation = 0;
        while(1)
        {
            if(parentID[i] == population[searchlocation].getID()){genotypeparent[i] = population[i].getMarker(); break;}
            if(parentID[i] != population[searchlocation].getID()){searchlocation++;}
        }
    }
    double *_grm_mkl = new double[parentID.size()*parentID.size()];
    //for(int i = 0; i < parentID.size()*parentID.size(); i++){_grm_mkl[i] = 0.0;}
    grm_noprevgrm(M,genotypeparent,_grm_mkl,scale);
    genotypeparent.clear();
    double* mate_value_matrix = new double[sireA.size() * damA.size()];                    /* mate allocation matrix based on pedigree */
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _grm_mkl[(i*parentID.size())+j] << " ";}
    //    cout << endl;
    //}
    for(int i = 0; i < sireA.size(); i++)
    {
        for(int j = 0; j < damA.size(); j++){mate_value_matrix[(i*sireA.size())+j] = _grm_mkl[(sireA[i]*parentID.size())+damA[j]];}
    }
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << mate_value_matrix[(i*sireA.size())+j] << "\t";}
    //   cout << endl;
    //}
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
    //for(int i = 0; i < summarystats.size(); i++){cout << summarystats[i] << endl;}
    string matingdesign = "YES";
    logfileloc << "       - Expected Progeny Level Inbreeding (Mean - Min - Max) prior to mating strategy: ";
    logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    if(summarystats[0] == 0.0){matingdesign = "NO";}
    if(matingdesign == "YES")
    {
        for(int i = 0; i < 6; i++){summarystats[i] = 0.0;}
        if(SimParameters.getMatingAlg() == "simu_anneal")
        {
            logfileloc << "         - Optimization algorithm utilized was Adaptive Simulated Annealing." << endl;
            adaptivesimuanneal(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction,SimParameters);
        }
        if(SimParameters.getMatingAlg() == "linear_prog")
        {
            logfileloc << "         - Optimization algorithm utilized was Kuhn–Munkres Algorithm." << endl;
            Kuhn_Munkres_Assignment(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        if(SimParameters.getMatingAlg() == "sslr")
        {
            logfileloc << "         - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
            sequentialselection(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
        logfileloc << "       - Expected Progeny Inbreeding Level (Mean - Min - Max) after to mating strategy: ";
        logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    }
    delete [] _grm_mkl; delete [] mate_value_matrix;
}
//////////////////////////////////////////////////////////
////            Minimize Based On ROH                 ////
//////////////////////////////////////////////////////////
void minimizeroh(vector <MatingClass> &matingindividuals, vector <Animal> &population,vector < hapLibrary > &haplib, string matingscenario, parameters &SimParameters,ostream& logfileloc)
{
    string direction = "minimum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector < int > parentID;                                                                        /* ID of parents */
    vector < string > genotypeparent;                                                               /* Genotype of parent */
    vector < int > sireIDs; vector < int > sireA;                                                   /* row in A and associated ID for sire */
    vector < int > damIDs; vector < int > damA;                                                     /* row in A and associated ID for dam */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC()); sireA.push_back(i);}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); damA.push_back(i);}
        }
        parentID.push_back(matingindividuals[i].getID_MC()); genotypeparent.push_back("");
    }
    /* Fill in index array to make it easer for all other functions */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        //cout << matingindividuals[i].getID_MC() << " " << matingindividuals[i].getType_MC() << " ";
        if(matingindividuals[i].getType_MC() == 0)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == sireIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != sireIDs[searchlocation]){searchlocation++;}
            }
        }
        if(matingindividuals[i].getType_MC() == 1)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == damIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != damIDs[searchlocation]){searchlocation++;}
            }
        }
        //cout << endl;
    }
    double *_rohrm = new double[parentID.size()*parentID.size()];
    for(int i = 0; i < parentID.size()*parentID.size(); i++){_rohrm[i] = 0.0;}
    generaterohmatrix(population,haplib,parentID,_rohrm);
    double* mate_value_matrix = new double[sireA.size() * damA.size()];                    /* mate allocation matrix based on pedigree */
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _rohrm[(i*parentID.size())+j] << " ";}
    //    cout << endl;
    //}
    //cout << endl << endl;
    for(int i = 0; i < sireA.size(); i++)
    {
        for(int j = 0; j < damA.size(); j++){mate_value_matrix[(i*sireA.size())+j] = _rohrm[(sireA[i]*parentID.size())+damA[j]];}
    }
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << mate_value_matrix[(i*sireA.size())+j] << "\t";}
    //    cout << endl;
    //}
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
    //for(int i = 0; i < summarystats.size(); i++){cout << summarystats[i] << endl;}
    string matingdesign = "YES";
    logfileloc << "       - Expected Progeny Level Inbreeding (Mean - Min - Max) prior to mating strategy: ";
    logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    if(summarystats[0] == 0.0){matingdesign = "NO";}
    if(matingdesign == "YES")
    {
        for(int i = 0; i < 6; i++){summarystats[i] = 0.0;}
        if(SimParameters.getMatingAlg() == "simu_anneal")
        {
            logfileloc << "         - Optimization algorithm utilized was Adaptive Simulated Annealing." << endl;
            adaptivesimuanneal(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction,SimParameters);
        }
        if(SimParameters.getMatingAlg() == "linear_prog")
        {
            logfileloc << "         - Optimization algorithm utilized was Kuhn–Munkres Algorithm." << endl;
            Kuhn_Munkres_Assignment(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        if(SimParameters.getMatingAlg() == "sslr")
        {
            logfileloc << "         - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
            sequentialselection(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
        logfileloc << "       - Expected Progeny Inbreeding Level (Mean - Min - Max) after to mating strategy: ";
        logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    }
    delete [] _rohrm;  delete [] mate_value_matrix;
}
//////////////////////////////////////////////////////////
////         Minimize Based On Genomic - MAF          ////
//////////////////////////////////////////////////////////
void minimizegenomic_maf(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario, double* M, float scale, parameters &SimParameters,ostream& logfileloc)
{
    string direction = "minimum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector < int > parentID;                                                                        /* ID of parents */
    vector < string > genotypeparent;                                                               /* Genotype of parent */
    vector < int > sireIDs; vector < int > sireA;                                                   /* row in A and associated ID for sire */
    vector < int > damIDs; vector < int > damA;                                                     /* row in A and associated ID for dam */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC()); sireA.push_back(i);}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); damA.push_back(i);}
        }
        parentID.push_back(matingindividuals[i].getID_MC()); genotypeparent.push_back("");
    }
    //cout << sireIDs.size() << " " << sireA.size() << " " << damIDs.size() << " " << damA.size() << endl;
    //for(int i = 0; i < sireA.size(); i++){cout << sireIDs[i] << "-" << sireA[i] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < damA.size(); i++){cout << damIDs[i] << "-" << damA[i] << " ";}
    //cout << endl << endl;
    //cout << parentID.size() << " " << genotypeparent.size() << endl;
    /* Fill in index array to make it easer for all other functions */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        //cout << matingindividuals[i].getID_MC() << " " << matingindividuals[i].getType_MC() << " ";
        if(matingindividuals[i].getType_MC() == 0)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == sireIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != sireIDs[searchlocation]){searchlocation++;}
            }
        }
        if(matingindividuals[i].getType_MC() == 1)
        {
            int searchlocation = 0; int matesfound = 0;
            while(matesfound < matingindividuals[i].getMatings_MC())
            {
                if(matingindividuals[i].getID_MC() == damIDs[searchlocation])
                {
                    matingindividuals[i].add_ToOwnIndex(searchlocation);
                    //cout << searchlocation << " ";
                    matesfound++; searchlocation++;
                }
                if(matingindividuals[i].getID_MC() != damIDs[searchlocation]){searchlocation++;}
            }
        }
        //cout << endl;
    }
    /* Now loop through population and get marker genotypes */
    for(int i = 0; i < parentID.size(); i++)
    {
        int searchlocation = 0;
        while(1)
        {
            if(parentID[i] == population[searchlocation].getID()){genotypeparent[i] = population[i].getMarker(); break;}
            if(parentID[i] != population[searchlocation].getID()){searchlocation++;}
        }
    }
    double *_grm_mkl = new double[parentID.size()*parentID.size()];
    for(int i = 0; i < parentID.size()*parentID.size(); i++){_grm_mkl[i] = 0.0;}
    matinggrm_maf(SimParameters,genotypeparent,_grm_mkl,logfileloc);
    genotypeparent.clear();
    double* mate_value_matrix = new double[sireA.size() * damA.size()];                    /* mate allocation matrix based on pedigree */
    //for(int i = 0; i < parentID.size(); i++)
    //{
    //    for(int j = 0; j < parentID.size(); j++){cout << _grm_mkl[(i*parentID.size())+j] << " ";}
    //    cout << endl;
    //}
    for(int i = 0; i < sireA.size(); i++)
    {
        for(int j = 0; j < damA.size(); j++){mate_value_matrix[(i*sireA.size())+j] = _grm_mkl[(sireA[i]*parentID.size())+damA[j]];}
    }
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << mate_value_matrix[(i*sireA.size())+j] << "\t";}
    //   cout << endl;
    //}
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
    //for(int i = 0; i < summarystats.size(); i++){cout << summarystats[i] << endl;}
    string matingdesign = "YES";
    logfileloc << "       - Expected Progeny Level Inbreeding (Mean - Min - Max) prior to mating strategy: ";
    logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    if(summarystats[0] == 0.0){matingdesign = "NO";}
    if(matingdesign == "YES")
    {
        for(int i = 0; i < 6; i++){summarystats[i] = 0.0;}
        if(SimParameters.getMatingAlg() == "simu_anneal")
        {
            logfileloc << "         - Optimization algorithm utilized was Adaptive Simulated Annealing." << endl;
            adaptivesimuanneal(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction,SimParameters);
        }
        if(SimParameters.getMatingAlg() == "linear_prog")
        {
            logfileloc << "         - Optimization algorithm utilized was Kuhn–Munkres Algorithm." << endl;
            Kuhn_Munkres_Assignment(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        if(SimParameters.getMatingAlg() == "sslr")
        {
            logfileloc << "         - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
            sequentialselection(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
        }
        summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
        logfileloc << "       - Expected Progeny Inbreeding Level (Mean - Min - Max) after to mating strategy: ";
        logfileloc <<  summarystats[0] / double(damA.size())<<" - "<<summarystats[1]<<" - "<<summarystats[2]<<"."<<endl;
    }
    delete [] _grm_mkl; delete [] mate_value_matrix;
}
//////////////////////////////////////////////////////////
////               Assortative Mating                 ////
//////////////////////////////////////////////////////////
void assortativemating(vector <MatingClass> &matingindividuals, vector <Animal> &population,string matingscenario,parameters &SimParameters ,ostream& logfileloc)
{
    /* Grab animals */
    vector <int> sireIDs; vector <double> sireVALUE;                        /* Sire and associated value */
    vector < int > damIDs; vector <double> damVALUE;                        /* Dam and associated value */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0){sireIDs.push_back(matingindividuals[i].getID_MC()); sireVALUE.push_back(0.0);}
            if(matingindividuals[i].getType_MC() == 1){damIDs.push_back(matingindividuals[i].getID_MC()); damVALUE.push_back(0.0);}
        }
    }
    /* find value that assortative mating is based on */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        int searchlocation = 0;
        while(1)
        {
            if(population[searchlocation].getID() == sireIDs[i])
            {
                if(SimParameters.getSelection() == "ebv"){sireVALUE[i] = population[searchlocation].getEBV();}
                if(SimParameters.getSelection() == "true_bv"){sireVALUE[i] = population[searchlocation].getGenotypicValue();}
                if(SimParameters.getSelection() == "phenotype"){sireVALUE[i] = population[searchlocation].getPhenotype();}
                break;
                //cout << sireIDs[i] << " " << sireVALUE[i] << endl;
            }
            if(population[searchlocation].getID() != sireIDs[i]){searchlocation++;}
        }
    }
    for(int i = 0; i < damIDs.size(); i++)
    {
        int searchlocation = 0;
        while(1)
        {
            if(population[searchlocation].getID() == damIDs[i])
            {
                if(SimParameters.getSelection() == "ebv"){damVALUE[i] = population[searchlocation].getEBV();}
                if(SimParameters.getSelection() == "true_bv"){damVALUE[i] = population[searchlocation].getGenotypicValue();}
                if(SimParameters.getSelection() == "phenotype"){damVALUE[i] = population[searchlocation].getPhenotype();}
                break;
                //cout << damIDs[i] << " " << damVALUE[i] << endl;
            }
            if(population[searchlocation].getID() != damIDs[i]){searchlocation++;}
        }
    }
    /* Sort males and females by damVALUE or sireVALUE */
    //for(int i = 0; i < sireIDs.size(); i++){cout << sireIDs[i] << "-" << sireVALUE[i] << "  ";}
    //cout << endl << endl;
    /* Sort based on value */
    int tempordera; double temporderb;
    for(int i = 0; i < sireIDs.size()-1; i++)
    {
        for(int j=i+1; j < sireIDs.size(); j++)
        {
            if(sireVALUE[i] > sireVALUE[j])
            {
                tempordera = sireIDs[i]; temporderb = sireVALUE[i];
                sireIDs[i] = sireIDs[j]; sireVALUE[i] = sireVALUE[j];
                sireIDs[j] = tempordera; sireVALUE[j] = temporderb;
            }
        }
    }
    //for(int i = 0; i < sireIDs.size(); i++){cout << sireIDs[i] << "-" << sireVALUE[i] << "  ";}
    //cout << endl << endl;
    //for(int i = 0; i < damIDs.size(); i++){cout << damIDs[i] << "-" << damVALUE[i] << "  ";}
    //cout << endl << endl;
    for(int i = 0; i < damIDs.size()-1; i++)
    {
        for(int j=i+1; j < damIDs.size(); j++)
        {
            if(damVALUE[i] > damVALUE[j])
            {
                tempordera = damIDs[i]; temporderb = damVALUE[i];
                damIDs[i] = damIDs[j]; damVALUE[i] = damVALUE[j];
                damIDs[j] = tempordera; damVALUE[j] = temporderb;
            }
        }
    }
    //for(int i = 0; i < damIDs.size(); i++){cout << damIDs[i] << "-" << damVALUE[i] << "  ";}
    //cout << endl << endl;
    /* loop through and push_back updated ID to MateIDs() */
    int damlocation;
    if(matingscenario == "POS_ASSORTATIVE"){damlocation = 0;}
    if(matingscenario == "NEG_ASSORTATIVE"){damlocation = (damIDs.size()-1);}
    for(int i = 0; i < sireIDs.size(); i++)
    {
        //cout << damlocation << " " << sireIDs[i] << " " << damIDs[damlocation] << " ---- ";
        /* Find sire and update */
        int searchlocation = 0;
        while(1)
        {
            if(sireIDs[i] == (matingindividuals[searchlocation].getID_MC()))
            {
                //cout << matingindividuals[searchlocation].getID_MC() << " ";
                //cout << (matingindividuals[searchlocation].get_mateIDs()).size() << " ";
                matingindividuals[searchlocation].add_ToMates(damIDs[damlocation]);
                //cout << (matingindividuals[searchlocation].get_mateIDs()).size() << " ";
                //cout << (matingindividuals[searchlocation].get_mateIDs())[0] << "-----";
                break;
            }
            if(sireIDs[i] != (matingindividuals[searchlocation].getID_MC())){searchlocation++;}
        }
        /* Find dam and update */
        searchlocation = 0;
        while(1)
        {
            if(damIDs[damlocation] == matingindividuals[searchlocation].getID_MC())
            {
                //cout << matingindividuals[searchlocation].getID_MC() << " ";
                //cout << (matingindividuals[searchlocation].get_mateIDs()).size() << " ";
                matingindividuals[searchlocation].add_ToMates(sireIDs[i]);
                //cout << (matingindividuals[searchlocation].get_mateIDs()).size() << " ";
                //cout << (matingindividuals[searchlocation].get_mateIDs())[0] << endl;
                break;
            }
            if(damIDs[damlocation] != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
        }
        if(matingscenario == "POS_ASSORTATIVE"){damlocation++;}
        if(matingscenario == "NEG_ASSORTATIVE"){damlocation--;}
    }
}
//////////////////////////////////////////////////////////
////          Index mating based on EBV only          ////
//////////////////////////////////////////////////////////
void geneticvalueindex(vector <MatingClass> &matingindividuals,vector <Animal> &population,string tempmatescen,parameters &SimParameters,ostream& logfileloc)
{
    string direction = "maximum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector <int> sireIDs; vector < double > sireVALUE;                               /* row in ebv matrix and associated ID for sire */
    vector <int> damIDs; vector < double > damVALUE;                                 /* row in ebv matrix and associated ID for dam */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        /* find location in population class */
        int searchlocation = 0;
        while(1)
        {
            if(matingindividuals[i].getID_MC() == population[searchlocation].getID()){break;}
            if(matingindividuals[i].getID_MC() != population[searchlocation].getID()){searchlocation++;}
        }
        /* now give value to sire or dam vectors */
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0)
            {
                sireIDs.push_back(matingindividuals[i].getID_MC());
                if(tempmatescen == "INDEX_EBV"){sireVALUE.push_back(population[searchlocation].getEBV());}
                if(tempmatescen == "INDEX_TBV"){sireVALUE.push_back(population[searchlocation].getGenotypicValue());}
                if(tempmatescen == "INDEX_PHEN"){sireVALUE.push_back(population[searchlocation].getPhenotype());}
            }
            if(matingindividuals[i].getType_MC() == 1)
            {
                damIDs.push_back(matingindividuals[i].getID_MC());
                if(tempmatescen == "INDEX_EBV"){damVALUE.push_back(population[searchlocation].getEBV());}
                if(tempmatescen == "INDEX_TBV"){damVALUE.push_back(population[searchlocation].getGenotypicValue());}
                if(tempmatescen == "INDEX_PHEN"){damVALUE.push_back(population[searchlocation].getPhenotype());}
            }
        }
    }
    //for(int i = 0; i < 10; i++){cout << sireIDs[i] << " " << sireVALUE[i] << " ---- " << damIDs[i] << " " << damVALUE[i] << endl;}
    double* mate_value_matrix = new double[sireIDs.size() * damIDs.size()];
    /* Fill mate allocation matrix (sire by dams) */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        for(int j = 0; j < damIDs.size(); j++){mate_value_matrix[(i*damIDs.size())+j] = ((sireVALUE[i] + damVALUE[j]) * double(0.5));}
    }
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << mate_value_matrix[(i*damIDs.size())+j] << " ";}
    //    cout << endl;
    //}
    if(SimParameters.getMatingAlg() == "sslr")
    {
        logfileloc << "       - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
        sequentialselection(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction);
    }
    if(SimParameters.getMatingAlg() == "gp")
    {
        logfileloc << "       - Optimization algorithm utilized was Genetic Programming." << endl;
        geneticalgorithm(matingindividuals,mate_value_matrix,sireIDs,damIDs,direction,SimParameters);
    }
    int sireused = 0;
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0 && (matingindividuals[i].get_mateIDs()).size() > 0){sireused += 1;}
    }
    logfileloc << "         - Total Number of Sires utilized in Mating Design: " << sireused << "." << endl;
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix,sireIDs,damIDs);
    cout << summarystats[0] << endl;
    logfileloc << "         - Final value of function maximized: " << summarystats[0] << endl;
    logfileloc << "         - Expected Progeny Breeding Value (Mean & Min & Max) after mating strategy: ";
    logfileloc <<  summarystats[0] / double(damVALUE.size())<<" & "<<summarystats[1]<<" & "<<summarystats[2]<<"."<<endl;
    delete [] mate_value_matrix;
}
//////////////////////////////////////////////////////////
////     Index mating based on EBV and Inbreeding     ////
//////////////////////////////////////////////////////////
void genetic_inbreedvalueindex(vector <MatingClass> &matingindividuals, vector <Animal> &population,string tempmatescen,parameters &SimParameters,string Pheno_Pedigree_File,double* M, float scale,vector < hapLibrary > &haplib,ostream& logfileloc)
{
    ///******************/
    ///* Check with EVA */
    ///******************/
    //system("rm -rf ./evadf || true"); system("rm -rf ./evarelationship || true"); system("rm -rf ./evapara.prm || true");
    // /* Used for eva */
    //vector <int> tempid; vector <int> tempsire; vector <int> tempdam; vector <int> tempsex;
    //vector <int> tempgen; vector <int> tempmates; vector <double> tempebv;
    //int malecan = 0; int femalecan = 0;
    //for(int i = 0; i < population.size(); i++)
    //{
    //    tempid.push_back(population[i].getID()); tempsire.push_back(population[i].getSire());
    //    tempdam.push_back(population[i].getDam()); tempsex.push_back(population[i].getSex());
    //    tempgen.push_back(population[i].getGeneration());
    //    if(population[i].getSex() == 0){tempmates.push_back(SimParameters.getDams() * SimParameters.getmaxsireprop()); malecan++;}
    //    if(population[i].getSex() == 1){tempmates.push_back(1); femalecan++;}
    //    tempebv.push_back(population[i].getEBV());
    //}
    //cout << "       - Number Males Candidates: " << malecan << "." <<  endl;
    //cout << "       - Number Females Candidates: " << femalecan << "." << endl;
    //std::ofstream outputevadf("evadf", std::ios_base::app | std::ios_base::out);
    //for(int i = 0; i < tempid.size(); i++)
    //{
    //    outputevadf<<tempid[i]<<"\t"<<0<<"\t"<<0<<"\t"<<tempsex[i]+1<<"\t"<<tempgen[i]<<"\t"<<tempmates[i]<<"\t"<<tempebv[i]<<endl;
    //    //outputevadf<<tempid[i]<<"\t"<<tempsire[i]<<"\t"<<tempdam[i]<<"\t"<<tempsex[i]+1<<"\t"<<tempgen[i]<<"\t"<<tempmates[i]<<"\t"<<tempebv[i]<<endl;
    //}
    //cout << tempmatescen << endl;
    string direction = "maximum";
    /* order of relationship matrix is the same as matingindividuals class */
    vector < int > parentID;                                                         /* ID of parents */
    vector < string > genotypeparent;                                                /* Genotype of parent */
    vector <int> sireIDs; vector < double > sireVALUE; vector < int > sireA;         /* row in ebv matrix and associated ID for sire */
    vector <int> damIDs; vector < double > damVALUE; vector < int > damA;            /* row in ebv matrix and associated ID for dam */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        /* find location in population class */
        int searchlocation = 0;
        while(1)
        {
            if(matingindividuals[i].getID_MC() == population[searchlocation].getID()){break;}
            if(matingindividuals[i].getID_MC() != population[searchlocation].getID()){searchlocation++;}
        }
        /* now give value to sire or dam vectors */
        for(int j = 0; j < matingindividuals[i].getMatings_MC(); j++)
        {
            if(matingindividuals[i].getType_MC() == 0)
            {
                sireIDs.push_back(matingindividuals[i].getID_MC()); sireA.push_back(i);
                if(tempmatescen == "INDEX_EBV_PED" || tempmatescen == "INDEX_EBV_GEN" || tempmatescen == "INDEX_EBV_ROH")
                {
                    sireVALUE.push_back(population[searchlocation].getEBV());
                }
                if(tempmatescen == "INDEX_TBV_PED" || tempmatescen == "INDEX_TBV_GEN" || tempmatescen == "INDEX_TBV_ROH")
                {
                    sireVALUE.push_back(population[searchlocation].getGenotypicValue());
                }
                if(tempmatescen == "INDEX_PHEN_PED" || tempmatescen == "INDEX_PHEN_GEN" || tempmatescen == "INDEX_PHEN_ROH")
                {
                    sireVALUE.push_back(population[searchlocation].getPhenotype());
                }
            }
            if(matingindividuals[i].getType_MC() == 1)
            {
                damIDs.push_back(matingindividuals[i].getID_MC()); damA.push_back(i);
                if(tempmatescen == "INDEX_EBV_PED" || tempmatescen == "INDEX_EBV_GEN" || tempmatescen == "INDEX_EBV_ROH")
                {
                    damVALUE.push_back(population[searchlocation].getEBV());
                }
                if(tempmatescen == "INDEX_TBV_PED" || tempmatescen == "INDEX_TBV_GEN" || tempmatescen == "INDEX_TBV_ROH")
                {
                    damVALUE.push_back(population[searchlocation].getGenotypicValue());
                }
                if(tempmatescen == "INDEX_PHEN_PED" || tempmatescen == "INDEX_PHEN_GEN" || tempmatescen == "INDEX_PHEN_ROH")
                {
                    damVALUE.push_back(population[searchlocation].getPhenotype());
                }
            }
        }
        parentID.push_back(matingindividuals[i].getID_MC());
        if(tempmatescen != "INDEX_EBV_PED" && tempmatescen != "INDEX_TBV_PED" && tempmatescen != "INDEX_PHEN_PED"){genotypeparent.push_back("");}
    }
    //cout << parentID.size() << " " << genotypeparent.size() << " " << sireIDs.size() << " " << sireVALUE.size() << " " << sireA.size() << " ";
    //cout << damIDs.size() << " " << damVALUE.size() << " " << damA.size() << endl;
    //for(int i = 0; i < 10; i++){cout << sireIDs[i] << " " << sireVALUE[i] << " ---- " << damIDs[i] << " " << damVALUE[i] << endl;}
    double* mate_value_matrix1 = new double[sireIDs.size() * damIDs.size()];
    double* mate_value_matrix2 = new double[sireA.size() * damA.size()];                    /* mate allocation matrix based on pedigree */
    double* mate_index_matrix = new double[sireA.size() * damA.size()];
    
    /* Fill mate allocation matrix (sire by dams) */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        for(int j = 0; j < damIDs.size(); j++){mate_value_matrix1[(i*damIDs.size())+j] = ((sireVALUE[i] + damVALUE[j]) * double(0.5));}
    }
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << mate_value_matrix1[(i*damIDs.size())+j] << " ";}
    //    cout << endl;
    //}
    if(tempmatescen == "INDEX_EBV_PED" || tempmatescen == "INDEX_TBV_PED" || tempmatescen == "INDEX_PHEN_PED")
    {
        double* subsetrelationship = new double[matingindividuals.size()*matingindividuals.size()];     /* Used to store subset of relationship matrix */
        for(int i = 0; i < (matingindividuals.size()*matingindividuals.size()); i++){subsetrelationship[i] = 0.0;}
        //pedigree_relationship(Pheno_Pedigree_File,parentID, subsetrelationship);                        /* Generate Relationship Matrix */
        pedigree_relationship_Colleau(Pheno_Pedigree_File,parentID, subsetrelationship);
        //std::ofstream outputevarel("evarelationship", std::ios_base::app | std::ios_base::out);
        //for(int i = 0; i < parentID.size(); i++)
        //{
        //    for(int j = i; j < parentID.size(); j++)
        //    {
        //        if(subsetrelationship[(i*parentID.size())+j] != 0.0)
        //        {
        //            outputevarel << parentID[i] << "\t" << parentID[j] << "\t" << subsetrelationship[(i*parentID.size())+j] << endl;
        //        }
        //    }
        //}
        /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
        for(int i = 0; i < sireA.size(); i++)
        {
            for(int j = 0; j < damA.size(); j++)
            {
                mate_value_matrix2[(i*damA.size())+j] = subsetrelationship[(sireA[i]*parentID.size())+damA[j]] * double(0.5);
            }
        }
        //for(int i = (parentID.size()-10); i < parentID.size(); i++)
        //{
        //   for(int j = (parentID.size()-10); j < parentID.size(); j++){cout << subsetrelationship[(i*parentID.size())+j] << "\t";}
        //    cout << endl;
        //}
        //cout << endl << endl;
        //for(int i = 0; i < 10; i++)
        //{
        //    for(int j = 0; j < 10; j++){cout << mate_value_matrix2[(i*sireA.size())+j] << "\t";}
        //   cout << endl;
        //}
        delete [] subsetrelationship;
    }
    if(tempmatescen == "INDEX_EBV_GEN" || tempmatescen == "INDEX_TBV_GEN" || tempmatescen == "INDEX_PHEN_GEN")
    {
        /* Now loop through population and get marker genotypes */
        for(int i = 0; i < parentID.size(); i++)
        {
            int searchlocation = 0;
            while(1)
            {
                if(parentID[i] == population[searchlocation].getID()){genotypeparent[i] = population[i].getMarker(); break;}
                if(parentID[i] != population[searchlocation].getID()){searchlocation++;}
            }
        }
        double *_grm_mkl = new double[parentID.size()*parentID.size()];
        for(int i = 0; i < (parentID.size()*parentID.size()); i++){_grm_mkl[i] = 0.0;}
        grm_noprevgrm(M,genotypeparent,_grm_mkl,scale);
        genotypeparent.clear();
        //for(int i = 0; i < 5; i++)
        //{
        //    for(int j = 0; j < 5; j++){cout << _grm_mkl[(i*parentID.size())+j] << " ";}
        //    cout << endl;
        //}
        for(int i = 0; i < sireA.size(); i++)
        {
            for(int j = 0; j < damA.size(); j++)
            {
                mate_value_matrix2[(i*damA.size())+j] = _grm_mkl[(sireA[i]*parentID.size())+damA[j]] * double(0.5);
            }
        }
        delete [] _grm_mkl;
    }
    if(tempmatescen == "INDEX_EBV_ROH" || tempmatescen == "INDEX_TBV_ROH" || tempmatescen == "INDEX_PHEN_ROH")
    {
        double *_rohrm = new double[parentID.size()*parentID.size()];
        for(int i = 0; i < (parentID.size()*parentID.size()); i++){_rohrm[i] = 0.0;}
        generaterohmatrix(population,haplib,parentID,_rohrm);
        //for(int i = 0; i < 10; i++)
        //{
        //    for(int j = 0; j < 10; j++){cout << _rohrm[(i*parentID.size())+j] << " ";}
        //    cout << endl;
        //}
        for(int i = 0; i < sireA.size(); i++)
        {
            for(int j = 0; j < damA.size(); j++)
            {
                mate_value_matrix2[(i*damA.size())+j] = _rohrm[(sireA[i]*parentID.size())+damA[j]] * double(0.5);
            }
        }
        //for(int i = 0; i < 15; i++)
        //{
        //    for(int j = 0; j < 10; j++){cout << mate_value_matrix2[(i*damA.size())+j] << "\t";}
        //    cout << endl;
        //}
        //cout << endl;
        delete [] _rohrm;
    }
    vector < double > returnweights((SimParameters.get_indexweights()).size(),0.0);
    generate2traitindex(mate_value_matrix1, mate_value_matrix2,mate_index_matrix,SimParameters,sireIDs,damIDs,returnweights);
    logfileloc << "       - Generated Index values based on ";
    if(tempmatescen=="INDEX_EBV_PED" || tempmatescen=="INDEX_EBV_GEN" || tempmatescen=="INDEX_EBV_ROH"){logfileloc << "ebv and ";}
    if(tempmatescen=="INDEX_TBV_PED" || tempmatescen=="INDEX_TBV_GEN" || tempmatescen=="INDEX_TBV_ROH"){logfileloc << "tbv and ";}
    if(tempmatescen=="INDEX_PHEN_PED" || tempmatescen=="INDEX_PHEN_GEN" || tempmatescen=="INDEX_PHEN_ROH"){logfileloc << "phenotype and ";}
    if(tempmatescen=="INDEX_EBV_PED" || tempmatescen=="INDEX_TBV_PED" || tempmatescen=="INDEX_PHEN_PED"){logfileloc << "pedigree relationships.";}
    if(tempmatescen=="INDEX_EBV_GEN" || tempmatescen=="INDEX_TBV_GEN" || tempmatescen=="INDEX_PHEN_GEN"){logfileloc << "genomic relationships.";}
    if(tempmatescen=="INDEX_EBV_ROH" || tempmatescen=="INDEX_TBV_ROH" || tempmatescen=="INDEX_PHEN_ROH"){logfileloc << "roh relationships.";}
    logfileloc << endl << "         - First trait regression coefficient " << returnweights[0] << "." << endl;
    logfileloc << "         - Second trait regression coefficient " << returnweights[1] << "." << endl;
    //std::ofstream outputparameter("evapara.prm", std::ios_base::app | std::ios_base::out);
    //outputparameter << "&DATAPARAMETERS" << endl;
    //outputparameter << " dataFile='evadf'" << endl;
    //outputparameter << " resultsDirectory=''" << endl;
    //outputparameter << " ignoreParentalPedigreeErrors=.false. /" << endl << endl;
    //outputparameter << "&POPULATIONHISTORY  /" << endl << endl;
    //outputparameter << "&RELATIONSHIPMATRIX" << endl;
    //outputparameter << " source='file'" << endl;
    //outputparameter << " gfile=evarelationship  /" << endl << endl;
    //outputparameter << "&OCSPARAMETERS" << endl;
    //outputparameter << " nMatings= " << femalecan << endl;
    //outputparameter << " optimise='penalty'" << endl;
    //outputparameter << " w_merit=   1.0" << endl;
    //outputparameter << " w_relationship=    " << returnweights[1] << endl;
    //outputparameter << " limitMaleMatings=   1  /" << endl << endl;
    //outputparameter << "&ALGORITHMPARAMETERS" << endl;
    //outputparameter << " generations=   " << 5000 << endl;
    //outputparameter << " nGenerationsNoImprovement=   30000" << endl;
    //outputparameter << " popSize=       " << 100 << endl;
    //outputparameter << " n_offspring=       " << double (100) / double(10) << endl;
    //outputparameter << " restart_interval=      " << double(5000) / double(10) << endl;
    //outputparameter << " exchange_algorithm=    " << double(5000) / double(10) << endl;
    //outputparameter << " mutate_probability=    " << 1 / double(2*double(femalecan)) << endl;
    //outputparameter << " crossover_probability=    " << 0.75/double(femalecan) << endl;
    //outputparameter << " directed_mutation_probability=        " << 1 / double(2*double(femalecan)) << endl;
    //outputparameter << " seed_rng=0  /" << endl << endl;
    //outputparameter << "&MATINGOPTIONS" << endl;
    //outputparameter << " matingStrategy='random'" << endl;
    //outputparameter << "  repeatedMatings=.true.  /" << endl;
    //system("./eva evapara.prm > output.txt 2>&1");
    //for(int i = 0; i < sireIDs.size(); i++)
    //{
    //    if(i == 0){cout << mate_value_matrix2[(i*damIDs.size())+0] << " ";}
    //    if(i > 0){if(sireIDs[i] != sireIDs[i-1]){cout << mate_value_matrix2[(i*damIDs.size())+0] << " ";}}
    //}
    //cout << endl << endl;
    //for(int i = 0; i < sireIDs.size(); i++)
    //{
    //    if(i == 0){cout << mate_value_matrix1[(i*damIDs.size())+0] << " ";}
    //    if(i > 0){if(sireIDs[i] != sireIDs[i-1]){cout << mate_value_matrix1[(i*damIDs.size())+0] << " ";}}
    //}
    //cout << endl << endl;
    if(SimParameters.getMatingAlg() == "sslr")
    {
        logfileloc << "       - Optimization algorithm utilized was Sequential Selection of Least Related Mates." << endl;
        sequentialselection(matingindividuals,mate_index_matrix,sireIDs,damIDs,direction);
    }
    if(SimParameters.getMatingAlg() == "gp")
    {
        logfileloc << "       - Optimization algorithm utilized was Genetic Programming." << endl;
        geneticalgorithm(matingindividuals,mate_index_matrix,sireIDs,damIDs,direction,SimParameters);
    }
    int sireused = 0;
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0 && (matingindividuals[i].get_mateIDs()).size() > 0){sireused += 1;}
    }
    logfileloc << "         - Total Number of Sires utilized in Mating Design: " << sireused << "." << endl;
    vector <double> summarystats(6,0.0);
    summarymatingdesign(matingindividuals,summarystats,mate_index_matrix,sireIDs,damIDs);
    logfileloc << "         - Final value of function maximized: " << summarystats[0] << endl;
    //cout << summarystats[0] << endl;
    for(int i = 0; i < 6; i++){summarystats[i] = 0;}
    summarymatingdesign(matingindividuals,summarystats,mate_value_matrix1,sireIDs,damIDs);
    logfileloc << "         - Expected Progeny Breeding Value (Mean & Min & Max) after mating strategy: ";
    logfileloc <<  summarystats[0] / double(damVALUE.size())<<" & "<<summarystats[1]<<" & "<<summarystats[2]<<"."<<endl;
    delete [] mate_index_matrix; delete [] mate_value_matrix1; delete [] mate_value_matrix2;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////          Combining Trait Index            ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////
/* Generate a 2 trait index */
//////////////////////////////
void generate2traitindex(double* mate_value_matrix1, double* mate_value_matrix2, double* mate_index_matrix,parameters &SimParameters, vector <int> const &sireIDs, vector <int> const &damIDs, vector <double> &returnweights)
{
    vector <string> direction(2,""); direction[0] = "positive"; direction[1] = "negative";
    using Eigen::MatrixXd; using Eigen::VectorXd;
    /* Generate a X matrix and a Y matrix; X is a function of the values and Y will be the value depending on the beta values */
    /* Don't need to keep repeated values only keep one for each combination */
    int numberofuniquerows = 0;
    for(int row = 0; row < sireIDs.size(); row++)       /* loop across rows and find unique ones */
    {
        if(row == 0){numberofuniquerows++;}
        if(row > 0){if(sireIDs[row] != sireIDs[row-1]){numberofuniquerows++;}}
    }
    //cout << numberofuniquerows << endl;
    MatrixXd X((numberofuniquerows*damIDs.size()),2); int row = 0; /* only keep one row for each individual */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        if(i == 0)
        {
            for(int j = 0; j < damIDs.size(); j++)
            {
                X(row,0) = mate_value_matrix1[(i*damIDs.size())+j];
                X(row,1) = mate_value_matrix2[(i*damIDs.size())+j];
                row++;
            }
        }
        if(i > 0)
        {
            if(sireIDs[i] != sireIDs[i-1])
            {
                for(int j = 0; j < damIDs.size(); j++)
                {
                    X(row,0) = mate_value_matrix1[(i*damIDs.size())+j];
                    X(row,1) = mate_value_matrix2[(i*damIDs.size())+j];
                    row++;
                }
            }
        }
    }
    //cout << mate_value_matrix1[(0*damIDs.size())+0] << " " << mate_value_matrix2[(0*damIDs.size())+0] << endl;
    //cout << X(0,0) << " " << X(0,1) << endl;
    //cout << mate_value_matrix1[(0*damIDs.size())+1] << " " << mate_value_matrix2[(0*damIDs.size())+1] << endl;
    //cout << X(1,0) << " " << X(1,1) << endl;
    //cout << X.rows() << " " << X.cols() << endl;
    /* now center and scale each one to have a mean of 0 and variance of 1; make it easier to converge */
    vector <double> mean(2,0.0); vector <double> sd(2,0.0);
    double sum1 = 0.0; double sum2 = 0.0;
    for(int i = 0; i < X.rows(); i++){sum1 += X(i,0); sum2 += X(i,1);}
    mean[0] = sum1 / double(X.rows()); sum1 = 0.0;
    mean[1] = sum2 / double(X.rows()); sum2 = 0.0;
    for(int i = 0; i < X.rows(); i++){sum1 += ((X(i,0)-mean[0]) * (X(i,0)-mean[0])); sum2 += ((X(i,1)-mean[1]) * (X(i,1)-mean[1]));}
    sd[0] = sqrt(sum1 / double(X.rows()-1)); sd[1] = sqrt(sum2 / double(X.rows()-1));
    //cout << mean[0] << " " << sd[0] << " " << mean[1] << " " << sd[1] << endl;
    for(int i = 0; i < X.rows(); i++)
    {
        X(i,0) = (X(i,0) - mean[0]) / double(sd[0]);
        X(i,1) = (X(i,1) - mean[1]) / double(sd[1]);
        //X(i,0) = (X(i,0) - mean[0]);
        //X(i,1) = (X(i,1) - mean[1]);
    }
    //cout << X(0,0) << " " << X(0,1) << endl;
    //cout << X(1,0) << " " << X(1,1) << endl;
    /* Compute stuff that is repeatedly used outside of while loop */
    MatrixXd SSFullInverse(2,2); SSFullInverse = (X.transpose() * X).inverse();
    MatrixXd X_SSInbreed(X.rows(),1); X_SSInbreed = X.leftCols(1);
    MatrixXd SSInbreedInverse(1,1); SSInbreedInverse = (X_SSInbreed.transpose() * X_SSInbreed).inverse();
    /* Initialize Parameters */
    double diff1 = 1.0; double diff2 = 1.0;
    VectorXd beta(2);
    beta(0) = 1.0; beta(1) = -0.2;
    double initialbeta = beta(0);
    //cout << beta(0) << " " << beta(1) << endl;
    double new1, new2;
    string kill = "NO"; int iteration = 1;
    VectorXd Y(X.rows());
    while(kill == "NO")
    {
        Y = X * beta;
        /* Generate SS for full model */
        double SSFull = 0.0;
        #pragma omp parallel for default(shared) reduction(+:SSFull)
        for(int i = 0; i < X.rows(); i++){SSFull += (X(i,0) * beta(0) + X(i,1) * beta(1)) * (X(i,0) * beta(0) + X(i,1) * beta(1));}
        /* Generate SS for Inbreeding by only including Genetic Value */
        MatrixXd IntInverse(2,1); IntInverse = (SSInbreedInverse * (X_SSInbreed.transpose()*Y));
        double gvSS = 0.0;
        #pragma omp parallel for default(shared) reduction(+:gvSS)
        for(int i = 0; i < X.rows(); i++){gvSS += (X(i,0) * IntInverse(0,0)) * (X(i,0) * IntInverse(0,0));}
        double inbSS = SSFull - gvSS;
        new2 = inbSS / double(gvSS + inbSS);
        new1 = gvSS / double(gvSS + inbSS);
        /* Figure out which direction to move the regression coefficents */
        if((new2 - (SimParameters.get_indexweights())[1]) < 0.0005)
        {
            if(direction[1] == "negative")
            {
                if(abs(new2 - (SimParameters.get_indexweights())[1]) > 0.05){beta(1) -= 0.10;}
                if(abs(new2 - (SimParameters.get_indexweights())[1]) > 0.015){beta(1) -= 0.01;}
                if(abs(new2-(SimParameters.get_indexweights())[1]) >= 0.0075 && abs(new2-(SimParameters.get_indexweights())[1]) < 0.015){beta(1) -= 0.005;}
                if(abs(new2 - (SimParameters.get_indexweights())[1]) < 0.0075){beta(1) -= 0.0005;}
            }
            if(direction[1] == "positive")
            {
                if(abs(new2 - (SimParameters.get_indexweights())[1]) > 0.05){beta(1) += 0.10;}
                if(abs(new2 - (SimParameters.get_indexweights())[1]) > 0.015){beta(1) += 0.01;}
                if(abs(new2-(SimParameters.get_indexweights())[1]) >= 0.0075 && abs(new2-(SimParameters.get_indexweights())[1]) < 0.015){beta(1) += 0.005;}
                if(abs(new2 - (SimParameters.get_indexweights())[1]) < 0.0075){beta(1) += 0.0005;}
            }
        }
        if((new1 - (SimParameters.get_indexweights())[0]) > 0.0005 && (abs(new2 - (SimParameters.get_indexweights())[1]) < 0.0005))
        {
            if(direction[0] == "positive"){beta(0) += 0.0001;}
            if(direction[0] == "negative"){beta(0) -= 0.0001;}
        }
        if((new2 - (SimParameters.get_indexweights())[1]) > 0.0005 && direction[1] == "negative"){beta(1) += 0.0001;}
        if((new2 - (SimParameters.get_indexweights())[1]) > 0.0005 && direction[1] == "positive"){beta(1) -= 0.0001;}
        diff1 = abs(new1 - (SimParameters.get_indexweights())[0]);
        diff2 = abs(new2 - (SimParameters.get_indexweights())[1]);
        //cout << new1 - indexweights[0] << " " << new2 - indexweights[1] << " ";
        //cout << gvSS << "--" << inbSS << "--" << new1 << " - " << new2 << " \/ " << beta(0) << " \/ "  << beta(1) << endl;
        if(diff1 < 0.0005 && diff2 < 0.0005){kill = "YES";}
        iteration++;
        //if(iteration > 55){exit (EXIT_FAILURE);}
    }
    returnweights[0] = beta(0); returnweights[1] = beta(1);
    /* Now that we know the correct weights that will generate the right variation explained by a trait make index */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        for(int j = 0; j < damIDs.size(); j++)
        {
            //double firstone = (mate_value_matrix1[(i*damIDs.size())+j] - mean[0]) / double(sd[0]);
            //double secondone = (mate_value_matrix2[(i*damIDs.size())+j] - mean[1]) / double(sd[1]);
            double firstone = (mate_value_matrix1[(i*damIDs.size())+j] - mean[0]) / double(sd[0]);
            double secondone = ((mate_value_matrix2[(i*damIDs.size())+j]) - mean[1]) / double(sd[1]);
            mate_index_matrix[(i*damIDs.size())+j] = firstone * double(beta(0)) + secondone * double(beta(1));
            //cout << firstone << " " << secondone << " " << mate_index_matrix[(i*damIDs.size())+j] << endl;
        }
    }
//    for(int i = 0; i < damIDs.size(); i++){cout << mate_value_matrix2[(0*damIDs.size())+i] << " ";}
//    cout << endl << endl;
//    for(int i = 0; i < damIDs.size(); i++){cout << mate_value_matrix1[(0*damIDs.size())+i] << " ";}
//    cout << endl << endl;
//    for(int i = 0; i < damIDs.size(); i++){cout << mate_index_matrix[(0*damIDs.size())+i] << " ";}
//    cout << endl << endl;
//    exit (EXIT_FAILURE);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////       Mating Algorithm Funcions        ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/* Adaptive Simulated Annealing that is less reliant on the cooling schedule */
///////////////////////////////////////////////////////////////////////////////
void adaptivesimuanneal(vector <MatingClass> &matingindividuals, double *mate_value_matrix, vector <int> const &sireIDs, vector <int> const &damIDs, string direction,parameters &SimParameters)
{
    vector <int> sire_mate_column(damIDs.size(),-5);            /* determines which sire gets mated to a particular dam */
    mt19937 gen(SimParameters.getSeed());
    //cout << matingindividuals.size() << endl;
    /* Used to randomly switch two elements */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0)
        {
            for(int j = 0; j < (matingindividuals[i].get_OwnIndex()).size(); j++)
            {
                //cout << matingindividuals[i].get_OwnIndex()[j] << " " << matingindividuals[i].get_mateIDs()[j] <<  endl;
                int searchlocation = 0;
                while(searchlocation < matingindividuals.size())
                {
                    if(matingindividuals[i].get_mateIDs()[j] == matingindividuals[searchlocation].getID_MC())
                    {
                        if((matingindividuals[searchlocation].get_OwnIndex()).size() != 1)
                        {
                            cout << endl << "Female mates greater than 1" << endl; exit (EXIT_FAILURE);
                        }
                        //cout << matingindividuals[searchlocation].getID_MC() << " ";
                        //cout << matingindividuals[searchlocation].get_OwnIndex()[0] << endl;
                        sire_mate_column[matingindividuals[i].get_OwnIndex()[j]] = matingindividuals[searchlocation].get_OwnIndex()[0];
                        //cout << endl;
                        //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
                        //cout << endl << endl;
                        break;
                    }
                    if(matingindividuals[i].get_mateIDs()[j] != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
                }
            }
        }
    }
    //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
    //cout << endl << endl;
    double costprevious = 0.0;
    for(int i = 0; i < sire_mate_column.size(); i++)
    {
        costprevious += mate_value_matrix[(i*sire_mate_column.size())+sire_mate_column[i]];
    }
    //cout << costprevious << endl;
    /* Parameters used in the algorithm */
    double T = 0.5;                                                 /* Temperature */
    double AcceptRate = 1.0;                                        /* Start at 1.0 decrease exponentially */
    int Evalsmax = 300000;                                          /* Maximum number of iteration to do */
    for(int i = 0; i < Evalsmax; i++)
    {
        vector < int > next_sire_mate_column;
        for(int i = 0; i < sire_mate_column.size(); i++){next_sire_mate_column.push_back(sire_mate_column[i]);}
        /* Randomly swap two elements */
        int change[2];
        for(int i = 0; i < 2; i++)
        {
            std::uniform_real_distribution<double> distribution15(0,sire_mate_column.size());
            change[i] = distribution15(gen);
            if(i == 1){if(change[0] == change[1]){i = i-1;}}        /* if same the redo sampling */
        }
        // change elements
        int oldfirstone = next_sire_mate_column[change[0]];
        next_sire_mate_column[change[0]] = next_sire_mate_column[change[1]];
        next_sire_mate_column[change[1]] = oldfirstone;
        // compute new one
        double cost_current = 0.0;
        for(int i = 0; i < sire_mate_column.size(); i++)
        {
            cost_current += mate_value_matrix[(i*sire_mate_column.size())+next_sire_mate_column[i]];
        }
        double diff = cost_current - costprevious;
        if(direction == "minimum")
        {
            if(diff < 0) {
                for(int i = 0; i < sire_mate_column.size(); i++)
                {
                    sire_mate_column[i] = next_sire_mate_column[i]; /*accept copy to current solution*/
                }
                costprevious = cost_current;
                AcceptRate = (1/double(500)) * double(499*AcceptRate+1);
            } else {
                std::uniform_real_distribution<double> distribution16(0,1);
                double sample = distribution16(gen);
                double acceptprob = exp((-1 * diff)/T);
                if(sample < acceptprob){
                    /* Even Though Worse Accept */
                    for(int i = 0; i < sire_mate_column.size(); i++)
                    {
                        sire_mate_column[i] = next_sire_mate_column[i]; /*accept copy to current solution*/
                    }
                    costprevious = cost_current;
                    AcceptRate = (1/double(500)) * double(499*AcceptRate+1);

                } else {
                    AcceptRate = (1/double(500)) * double(499*AcceptRate);
                }
            }
        }
        if(direction == "maximum")
        {
            if(diff > 0) {
                for(int i = 0; i < sire_mate_column.size(); i++)
                {
                    sire_mate_column[i] = next_sire_mate_column[i]; /*accept copy to current solution*/
                }
                costprevious = cost_current;
                AcceptRate = (1/double(500)) * double(499*AcceptRate+1);
            } else {
                std::uniform_real_distribution<double> distribution16(0,1);
                double sample = distribution16(gen);
                double acceptprob = exp((-1 * diff)/T);
                if(sample < acceptprob){
                    /* Even Though Worse Accept */
                    for(int i = 0; i < sire_mate_column.size(); i++)
                    {
                        sire_mate_column[i] = next_sire_mate_column[i]; /*accept copy to current solution*/
                    }
                    costprevious = cost_current;
                    AcceptRate = (1/double(500)) * double(499*AcceptRate+1);

                } else {
                    AcceptRate = (1/double(500)) * double(499*AcceptRate);
                }
            }
        }
        double LamRate;
        if((i / double(Evalsmax)) < 0.15)
        {
            double exponentterm = -1 * ((i+1) / double(Evalsmax) / double(0.15));
            LamRate = 0.44 + 0.56 * pow(560.0,exponentterm );
        }
        if((i / double(Evalsmax)) >= 0.15 && (i / double(Evalsmax)) < 0.65){LamRate = 0.44;}
        if((i / double(Evalsmax)) >= 0.65)
        {
            double exponentterm = (-1* ((((i+1)/double(Evalsmax))-0.65) / double(0.35)));
            LamRate = 0.44 * pow(440.0,exponentterm );
        }
        if(AcceptRate > LamRate){
            T = 0.999 * T;
        } else {
            T = T / double(0.999);
        }
        //if(i % 3000 == 0)
        //{
        //    cout << (i / double(Evalsmax)) << " " << costprevious << " " << T << " " << LamRate << " " << AcceptRate << endl;
        //}
    }
    //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
    //cout << endl << endl;
    //costprevious = 0.0;
    //for(int i = 0; i < sire_mate_column.size(); i++)
    //{
    //    costprevious += mate_value_matrix[(i*sire_mate_column.size())+sire_mate_column[i]];
    //}
    /* Now change vector of mates for males and females */
    updatematings(matingindividuals, sire_mate_column, sireIDs, damIDs);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
/* When all sires mated equally then can use Kuhn-Munkres algorithm to get optimal assignment  */
/////////////////////////////////////////////////////////////////////////////////////////////////
void Kuhn_Munkres_Assignment(vector <MatingClass> &matingindividuals, double *mate_value_matrix,vector <int> const &sireIDs, vector <int> const &damIDs, string direction)
{
    vector <int> sire_mate_column(damIDs.size(),-5);            /* determines which sire gets mated to a particular dam */
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0)
        {
            for(int j = 0; j < (matingindividuals[i].get_OwnIndex()).size(); j++)
            {
                //cout << matingindividuals[i].get_OwnIndex()[j] << " " << matingindividuals[i].get_mateIDs()[j] <<  endl;
                int searchlocation = 0;
                while(searchlocation < matingindividuals.size())
                {
                    if(matingindividuals[i].get_mateIDs()[j] == matingindividuals[searchlocation].getID_MC())
                    {
                        if((matingindividuals[searchlocation].get_OwnIndex()).size() != 1)
                        {
                            cout << endl << "Female mates greater than 1" << endl; exit (EXIT_FAILURE);
                        }
                        //cout << matingindividuals[searchlocation].getID_MC() << " ";
                        //cout << matingindividuals[searchlocation].get_OwnIndex()[0] << endl;
                        sire_mate_column[matingindividuals[i].get_OwnIndex()[j]] = matingindividuals[searchlocation].get_OwnIndex()[0];
                        //cout << endl;
                        //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
                        //cout << endl << endl;
                        break;
                    }
                    if(matingindividuals[i].get_mateIDs()[j] != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
                }
            }
        }
    }
    //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
    //cout << endl << endl;
    int dim = sire_mate_column.size();
    /* Generate Matrices that are used in algorithm */
    vector< vector <double> > costmatrix;   /* refers to values to minimize */
    vector< vector < int > > maskmatrix;    /* generates matching pairs */
    vector < int > R_cov(dim,0);            /* Used to cover rows */
    vector < int > C_cov(dim,0);            /* Used to cover columns */
    int path_row_0, path_col_0;             /* Used in  augmenting path algorithm */
    /* Copy intitialmatrix so don't overwrite */
    for(int i = 0; i < dim; i++)
    {
        vector < double > row(dim,0); costmatrix.push_back(row);
        vector < int > row1(dim,0); maskmatrix.push_back(row1);
    }
    /* Matrix has to be greater than 0 so first check if it is satisfied */
    double intitialminimum = mate_value_matrix[(0*dim)+0];
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++){if(mate_value_matrix[(i*dim)+j] < intitialminimum){intitialminimum = mate_value_matrix[(i*dim)+j];}}
    }
    if(intitialminimum < 0)
    {
        /* add the smallest value to each element to make it 0 */
        /* Fill matrix that will be changed */
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++){costmatrix[i][j] = mate_value_matrix[(i*dim)+j] + intitialminimum;}
        }
    }
    if(intitialminimum >= 0)
    {
        /* Fill matrix that will be changed */
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++){costmatrix[i][j] = mate_value_matrix[(i*dim)+j];}
        }
    }
    vector< vector <int> > path;
    for(int i = 0; i < (dim+200); i++){vector < int > row(2,0);path.push_back(row);}
    if(direction == "maximum")
    {
        /* Maximum is found by finding the largest number and subtracting it off from original matrix */
        double maxvalue = 0.0;
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                if(costmatrix[i][j] > maxvalue){maxvalue = costmatrix[i][j];}
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                costmatrix[i][j] = maxvalue - costmatrix[i][j];
            }
        }
    }
    double objective_cost = 0;
    bool done = false; int step = 1; int iteration = 1;
    while(!done)
    {
        //cout << iteration << "-" << step << endl;
        //for(int i = 0; i < dim; i++)
        //{
        //   for(int j = 0; j < dim; j++){cout << costmatrix[i][j] << " ";}
        //    cout << endl;
        //}
        //cout << "----" << endl;
        //for(int i = 0; i < dim; i++)
        //{
        //    for(int j = 0; j < dim; j++)
        //    {
        //        cout << maskmatrix[i][j] << " ";
        //    }
        //    cout << endl;
        //}
        //cout << "----" << endl;
        //for(int i = 0; i < dim; i++){cout << R_cov[i] << " ";}
        //cout << endl;
        //for(int i = 0; i < dim; i++){cout << C_cov[i] << " ";}
        //cout << endl << endl;
        //if(iteration == 500){exit (EXIT_FAILURE);}
        switch (step)
        {
            case 1:
            {
                double min_in_row;
                /* Loop across rows */
                for(int r = 0; r < dim; r++)
                {
                    /* Find the smallest element for a given row */
                    min_in_row = costmatrix[r][0];
                    for(int c = 1; c < costmatrix[0].size(); c++)
                    {
                        if(costmatrix[r][c] < min_in_row){min_in_row = costmatrix[r][c];}
                    }
                    /* Once found subtract it from every element in row */
                    for(int c = 0; c < costmatrix[0].size(); c++){costmatrix[r][c] -= min_in_row;}
                }
                step = 2; iteration++; break;
            }
            case 2:
            {
                /* Find a zero in the resulting matrix. If there is not a 0 */
                /*in its row or column give it a 1 in maskign matrix */
                for(int r = 0; r < costmatrix.size(); r++)
                {
                    for(int c = 0; c < costmatrix[0].size(); c++)
                    {
                        if(costmatrix[r][c] == 0 && R_cov[r] == 0 && C_cov[c] == 0)
                        {
                            maskmatrix[r][c] = 1; R_cov[r] = 1; C_cov[c] = 1;
                        }
                    }
                }
                /* Before  we proceed to step 3 need to set C_cov and R_cov back to zero */
                for(int r = 0; r < costmatrix.size(); r++){R_cov[r] = 0;}
                for(int c = 0; c < costmatrix[0].size(); c++){C_cov[c] = 0;}
                step = 3; iteration++; break;
            }
            case 3:
            {
                /* cover each column containing a 1 in the masked matrix */
                for(int r = 0; r < costmatrix.size(); r++)
                {
                    for(int c = 0; c < costmatrix[0].size(); c++)
                    {
                        if(maskmatrix[r][c] == 1){C_cov[c] = 1;}
                    }
                }
                /* If k columns covered then can stop otherwise go to step 4 */
                int colcount = 0;
                for(int c = 0; c < costmatrix[0].size(); c++)
                {
                    if(C_cov[c] == 1){colcount+= 1;}
                }
                if(colcount >= costmatrix[0].size() || colcount >= costmatrix.size())
                    step = 7;
                else
                    step = 4;
                iteration++; break;
                
            }
            case 4:
            {
                int row = -1; int col = -1; string stop = "no";
                while (stop != "yes")
                {
                    /* find a non-covered zero and prime it */
                    int r = 0; int c; string stopa = "no"; row = -1; col = -1;
                    while(stopa != "yes")
                    {
                        c = 0;
                        while(true)
                        {
                            if(costmatrix[r][c] == 0 && R_cov[r] == 0 && C_cov[c] == 0)
                            {
                                row = r; col = c; stopa = "yes";
                            }
                            c += 1;
                            if(c >= costmatrix[0].size() || stopa == "yes"){break;}
                        }
                        r += 1;
                        if(r >= costmatrix.size()){stopa = "yes";}
                    }
                    if(row == -1)
                    {
                        stop = "yes"; step = 6;
                    }
                    else
                    {
                        maskmatrix[row][col] = 2;
                        /* Check to see if their is a 1 in corresponding row */
                        bool temp = false;
                        for(int c = 0; c < costmatrix[0].size(); c++)
                        {
                            if(maskmatrix[row][c] == 1){temp = true;}
                        }
                        if(temp == true)
                        {
                            /* now find where the 1 occurs in the row */
                            col = - 1;
                            for(int c = 0; c < costmatrix[0].size(); c++)
                            {
                                if(maskmatrix[row][c] == 1){col = c;}
                            }
                            R_cov[row] = 1;
                            C_cov[col] = 0;
                        }
                        else
                        {
                            stop = "yes"; step = 5; path_row_0 = row; path_col_0 = col;
                        }
                    }
                }
                iteration++; break;
            }
            case 5:
            {
                string stop = "no";
                int r = -1;
                int c = -1;
                int path_count = 1;
                path[path_count -1][0] = path_row_0;
                path[path_count -1][1] = path_col_0;
                while (stop != "yes")
                {
                    /* find 1 in columns */
                    r = -1;
                    for(int i = 0; i < costmatrix[0].size(); i++)
                    {
                        if(maskmatrix[i][(path[path_count-1][1])] == 1){r = i;}
                    }
                    if(r > -1)
                    {
                        path_count += 1;
                        path[path_count -1][0] = r;
                        path[path_count -1][1] = path[path_count - 2][1];
                    }
                    else
                        stop = "yes";
                    if(stop != "yes")
                    {
                        /* Find the 2 in rows */
                        for(int j = 0; j < costmatrix[0].size(); j++)
                        {
                            if(maskmatrix[(path[path_count-1][0])][j] == 2){c = j;}
                        }
                        path_count += 1;
                        //cout << path_count -1 << " ";
                        path[path_count -1][0] = path[path_count - 2][0];
                        path[path_count -1][1] = c;
                    }
                }
                /* Augment Path */
                for(int p = 0; p < path_count; p++)
                {
                    if(maskmatrix[(path[p][0])][(path[p][1])] == 1)
                    {
                        maskmatrix[(path[p][0])][(path[p][1])] = 0;
                    }
                    else
                        maskmatrix[(path[p][0])][(path[p][1])] = 1;
                    
                }
                /* Clear covers */
                for(int r = 0; r < costmatrix.size(); r++){R_cov[r] = 0;}
                for(int c = 0; c < costmatrix[0].size(); c++){C_cov[c] = 0;}
                /* Erase 2 from maskmatrix */
                for(int r = 0; r < costmatrix.size(); r++)
                {
                    for(int c = 0; c < costmatrix[0].size(); c++)
                    {
                        if(maskmatrix[r][c] == 2){maskmatrix[r][c] = 0;}
                    }
                }
                step = 3; iteration++; break;
            }
            case 6:
            {
                /* Find smallest value that was searched for in step 4 */
                double minval = -1;
                for(int r = 0; r < costmatrix.size(); r++)
                {
                    for(int c = 0; c < costmatrix[0].size(); c++)
                    {
                        if(R_cov[r] == 0 && C_cov[c] == 0)
                        {
                            if(minval == -1){minval = costmatrix[r][c];}
                            if(minval > costmatrix[r][c]){minval = costmatrix[r][c];}
                        }
                    }
                }
                /* Once the smallest value is found add to covered rows and subtract from uncovered */
                for(int r = 0; r < costmatrix.size(); r++)
                {
                    for(int c = 0; c < costmatrix[0].size(); c++)
                    {
                        if(R_cov[r] == 1){costmatrix[r][c] += minval;}
                        if(C_cov[c] == 0){costmatrix[r][c] -= minval;}
                    }
                }
                step = 4; iteration++; break;
            }
            case 7:
            {
                for(int i = 0; i < dim; i++)
                {
                    for(int j = 0; j < dim; j++)
                    {
                        if(maskmatrix[i][j] == 1){objective_cost += mate_value_matrix[(i*dim)+j];}
                    }
                }
                done = true; break;
            }
        }
    }
    /* Now place column of where 1 is at for a given row in mate_column vector */
    int numberones = 0;
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            if(maskmatrix[i][j] == 1){sire_mate_column[i] = j; numberones++;}
        }
    }
    if(numberones != dim){cout << "Something went wrong in linear programming algorith!!" << endl; exit (EXIT_FAILURE);}
    //for(int temp = 0; temp < sire_mate_column.size(); temp++){cout << sire_mate_column[temp] << " ";}
    //cout << endl << endl;
    updatematings(matingindividuals, sire_mate_column, sireIDs, damIDs);
}
////////////////////////////////////
/* Sequential Selection of Mates  */
////////////////////////////////////
void sequentialselection(vector <MatingClass> &matingindividuals, double *mate_value_matrix,vector <int> const &sireIDs, vector <int> const &damIDs, string direction)
{
    vector <int> sireusage;                         /* Number of times sire has been used */
    vector <int> maxsireusage;                      /* Maximum number of times a sire can be mated */
    vector <int> sireid;                            /* ID of sire */
    int totalmatings = 0;
    for(int i = 0; i < matingindividuals.size(); i++)
    {
        if(matingindividuals[i].getType_MC() == 0)
        {
            maxsireusage.push_back(matingindividuals[i].getMatings_MC()); sireusage.push_back(0);
            sireid.push_back(matingindividuals[i].getID_MC());
        }
        if(matingindividuals[i].getType_MC() == 1){totalmatings += matingindividuals[i].getMatings_MC();}
    }
    int columns = sireusage.size();
    vector <int> dam_mate_column(totalmatings,-5);            /* determines which sire gets mated to a particular dam */
    //cout << sireusage.size() << " " << maxsireusage.size() << " " << sireid.size() << endl;
    //for(int i = 0; i < maxsireusage.size(); i++){cout << maxsireusage[i] << endl;}
    //cout << dam_mate_column.size() << endl;
    for(int row = 0; row < dam_mate_column.size(); row++)       /* loop across columns (i.e. females) and find row with lowest mating */
    {
        /* grab row and sort */
        vector < double > values(columns,0.0); vector < int > valuerow(columns,0); int currentvalueindex = 0;
        for(int i = 0; i < sireIDs.size(); i++)
        {
            if(i == 0)
            {values[currentvalueindex] = mate_value_matrix[(i*totalmatings)+row]; valuerow[currentvalueindex] = currentvalueindex; currentvalueindex++;
            }
            if(i > 0)
            {
                if(sireIDs[i] != sireIDs[i-1])
                {
                    values[currentvalueindex] = mate_value_matrix[(i*totalmatings)+row];
                    valuerow[currentvalueindex] = currentvalueindex; currentvalueindex++;
                }
            }
        }
        //for(int i = 0; i < values.size(); i++){cout << values[i] << "-" << valuerow[i] << "\t";}
        //cout << endl << endl;
        double tempb; int temp;
        /* Sort by value either from highest to lowest or lowest to highest */
        for(int i = 0; i < (columns - 1); i++)
        {
            for(int j = i+1; j < columns; j++)
            {
                if(direction == "minimum")
                {
                    if(values[i] > values[j])
                    {
                        tempb = values[i]; temp = valuerow[i];
                        values[i] = values[j]; valuerow[i] = valuerow[j];
                        values[j] = tempb; valuerow[j] = temp;
                    }
                }
                if(direction == "maximum")
                {
                    if(values[i] < values[j])
                    {
                        tempb = values[i]; temp = valuerow[i];
                        values[i] = values[j]; valuerow[i] = valuerow[j];
                        values[j] = tempb; valuerow[j] = temp;
                    }
                }
            }
        }
        //for(int i = 0; i < values.size(); i++){cout << values[i] << "-" << valuerow[i] << "\t";}
        //cout << endl << endl;
        //if(row > 5){exit (EXIT_FAILURE);}
        string sirefound = "NO";
        if(sirefound == "NO")
        {
            /* if sire has less than Dam mating use */
            if(sireusage[valuerow[0]] < maxsireusage[valuerow[0]])
            {
                //cout << valuerow[0] << " - ";
                dam_mate_column[row] = valuerow[0]; sireusage[valuerow[0]] += 1; sirefound = "YES";
            }
            /* If this is false then will not do anything and drop down to next if because damfound still = "NO" */
        }
        if(sirefound == "NO")
        {
            /* if sire is over Dam mating use next lowest one */
            if(sireusage[valuerow[0]] >= maxsireusage[valuerow[0]])
            {
                int next = 1;
                string stop = "GO";
                while(stop == "GO")
                {
                    if(sireusage[valuerow[next]] >= maxsireusage[valuerow[next]]){next++;}
                    if(sireusage[valuerow[next]] < maxsireusage[valuerow[next]])
                    {
                        //cout << valuerow[next] << " - ";
                        dam_mate_column[row] = valuerow[next]; sireusage[valuerow[next]] += 1; stop = "NOPE";
                    }
                }
                sirefound = "YES";
            }
            /* if still no than their is an error so kill program */
            if (sirefound == "NO"){cout << endl << "sslr algorithm failed!" << endl; exit (EXIT_FAILURE);}
        }
        //for(int i = 0; i < sireusage.size(); i++){cout << sireusage[i] << " ";}
        //cout << endl << endl;
        //if(row >= 100){exit (EXIT_FAILURE);}
    }
    //cout << endl << endl;
    
    //for(int i = 0; i < dam_mate_column.size(); i++){cout << dam_mate_column[i] << " - ";}
    //cout << endl << endl;
    vector <int> alreadyused(sireIDs.size(),0);                   /* Determines dam has already been chosen for a sire */
    vector <int> sire_mate_column(sireIDs.size(),-5);             /* determines which sire gets mated to a particular dam */
    /* Now fill sire_mate_column and update mating individual class */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        //cout << sireIDs[i] << " ";
        int searchlocation = 0;
        /* now find in sireid */
        while(1)
        {
            if(sireIDs[i] == sireid[searchlocation]){break;}
            if(sireIDs[i] != sireid[searchlocation]){searchlocation++;}
        }
        //cout << searchlocation << " " << sireid[searchlocation] << endl;
        /* once found now now search for it in dam_mate_column */
        int searchlocationa = 0;
        while(searchlocationa < dam_mate_column.size())
        {
            if(dam_mate_column[searchlocationa] == searchlocation && alreadyused[searchlocationa] == 0)
            {
                sire_mate_column[i] = searchlocationa; alreadyused[searchlocationa] = -5; break;
            }
            if(dam_mate_column[searchlocationa] != searchlocation || alreadyused[searchlocationa] != 0){searchlocationa++;}
        }
        //cout << searchlocationa << endl;
    }
    //for(int i = 0; i < sireIDs.size(); i++){cout << sire_mate_column[i] << " - ";}
    //cout << endl;
    int doublecheckmate = 0; int doublechecknomate = 0;
    for(int i = 0; i < sire_mate_column.size(); i++)
    {
        if(sire_mate_column[i] == -5){doublechecknomate += 1;}
        if(sire_mate_column[i] != -5 && sire_mate_column[i] >= 0){doublecheckmate += 1;}
    }
    if(dam_mate_column.size() != doublecheckmate){cout << endl << "Mating Pairs Don't Match Up!" << endl; exit (EXIT_FAILURE);}
    updatematings(matingindividuals, sire_mate_column, sireIDs, damIDs);
}

////////////////////////////////////
/*     Genetic Programming        */
////////////////////////////////////
void geneticalgorithm(vector <MatingClass> &matingindividuals, double *mate_value_matrix,vector <int> const &sireIDs, vector <int> const &damIDs, string direction,parameters &SimParameters)
{
    vector <int> sireid;                            /* ID of sire */
    vector < vector < double >> cost;               /* refers to values to maximize/minimize */
    vector < vector < int >> resource;              /* refers to the resources utilized */
    vector < int > maxresourcerow;                  /* Maximum number of times a sire can be mated */
    /* Fill cost and resource matrix */
    for(int row = 0; row < sireIDs.size(); row++)
    {
        if(row == 0)
        {
            /* grab row and sort */
            vector < double > rows; vector < int > rowa;
            for(int i = 0; i < damIDs.size(); i++){rows.push_back(mate_value_matrix[(row*damIDs.size())+i]); rowa.push_back(1);}
            cost.push_back(rows); resource.push_back(rowa); sireid.push_back(sireIDs[row]);
        }
        if(row > 0)
        {
            if(sireIDs[row] != sireIDs[row-1])
            {
                vector < double > rows; vector < int > rowa;
                for(int i = 0; i < damIDs.size(); i++){rows.push_back(mate_value_matrix[(row*damIDs.size())+i]); rowa.push_back(1);}
                cost.push_back(rows); resource.push_back(rowa); sireid.push_back(sireIDs[row]);
            }
        }
    }
    for(int i = 0; i < sireid.size(); i++)
    {
        int searchlocation = 0;
        while(1)
        {
            if(sireid[i] == matingindividuals[searchlocation].getID_MC())
            {
                maxresourcerow.push_back(matingindividuals[searchlocation].getMatings_MC()); break;
            }
            if(sireid[i] != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
        }
    }
    int femalecand = resource[0].size();
    int malecand = resource.size();
    /* Need to ensure all values greater than zero */
    double globalminimum = 0.0;
    for(int i = 0; i < malecand; i++)
    {
        for(int j = 0; j < femalecand ; j++){if(cost[i][j] < globalminimum){globalminimum = cost[i][j];}}
    }
    //cout << globalminimum << endl;
    /* add absolute value to global minimum and before doing final one subtract off */
    if(globalminimum < 0.0)
    {
        for(int i = 0; i < malecand; i++)
        {
            for(int j = 0; j < femalecand ; j++){cost[i][j] += abs(globalminimum);}
        }
    }
    int size = 100;
    int N_offspring = 100;
    int stopwhenunchanged = 4000;
    int unchanged = 0;
    int generation = 0;
    int exchangealgorithm = 1;
    mt19937 gen(SimParameters.getSeed());
    /***************************************/
    /* Variables used in Genetic Algorithm */
    /***************************************/
    vector < vector < int >> oldpop;                        /* population of solutions based on size */
    vector < double > oldpopfitness(size,0.0);              /* population fitness */
    vector < double > oldpopinfeasability(size,0.0);        /* population infeasability */
    vector < double > oldpopfitnessInf(size,0.0);           /* population fitness + infeasability */
    vector < string > stringmates(size,"");                 /* mating pairs in string */
    /***************************************/
    /*    Generate an initial population   */
    /***************************************/
    for(int i = 0; i < size; i++)
    {
        vector <int> row(femalecand,0);     /* a solution is the number of columns (i.e. jobs or females) */
        std::uniform_real_distribution <double> distribution(0,1);                          /* Randomly pick a number between i and number of sires */
        for(int j = 0; j < femalecand; j++){row[j] = int((distribution(gen))*malecand);}
        oldpop.push_back(row);
    }
    /* Calculate Fitness which is penalized by infeasability */
    for(int i = 0; i < size; i++)
    {
        stringstream strStreamM (stringstream::in | stringstream::out);
        double sum = 0.0;
        for(int j = 0; j < femalecand; j++){sum += cost[oldpop[i][j]][j]; strStreamM << oldpop[i][j];}
        oldpopfitness[i] = sum; stringmates[i] = strStreamM.str();
    }
    /* Calculate Feasability */
    double minvalue = oldpopfitness[0];
    for(int i = 0; i < size; i++)
    {
        vector <int> rowuse(malecand,0);
        for(int j = 0; j < femalecand; j++){rowuse[oldpop[i][j]] += resource[oldpop[i][j]][j];}
        double value = 0;
        for(int j = 0; j < malecand; j++)
        {
            double temp = (rowuse[j] / double(maxresourcerow[j])) - 1;
            if(temp > 0){value += temp;}
            if(temp <= 0){value += 0;}
        }
        oldpopinfeasability[i] = value * (1/double(resource.size()));
        if(oldpopfitness[i] < minvalue){minvalue = oldpopfitness[i];}
    }
    //cout << minvalue << endl;
    for(int i = 0; i < size; i++)
    {
        if(oldpopinfeasability[i] > 0){oldpopfitnessInf[i] = minvalue * (1 - oldpopinfeasability[i]);}
        if(oldpopinfeasability[i] == 0){oldpopfitnessInf[i] = oldpopfitness[i];}
    }
    //cout << "--- INITIAL POPULATION ----" << endl;
    //for(int i = 0; i < size; i++)
    //{
    //    for(int j = 0; j < resource[0].size(); j++){cout << oldpop[i][j] << " ";}
    //    cout << " --- " << oldpopfitness[i] <<"\t"<< oldpopinfeasability[i] <<"\t"<< oldpopfitnessInf[i] <<"\t"<< stringmates[i] << endl;
    //}
    vector < int > best_solution (femalecand,0);
    double bestsolutionvalue = 0.0; int bestsolutionrow; double meanpopulationfitness = 0.0;
    /* Save best solution */
    for(int i = 0; i < size; i++)
    {
        if(oldpopfitness[i] > bestsolutionvalue && oldpopinfeasability[i] == 0)
        {
            bestsolutionvalue = oldpopfitnessInf[i]; bestsolutionrow = i;
        }
        meanpopulationfitness += oldpopfitnessInf[i];
    }
    bestsolutionvalue = oldpopfitness[bestsolutionrow];
    for(int i = 0; i < femalecand; i++){best_solution[i] = oldpop[bestsolutionrow][i];}
    //cout << endl << endl << "Starting Best Solution: ";
    //for(int i = 0; i < best_solution.size(); i++){cout << best_solution[i] << " ";}
    //cout << " --- " << bestsolutionvalue << " Mean Fitness: " << meanpopulationfitness / double(size) << endl;
    std::uniform_real_distribution <double> distribution(0,1);
    while(unchanged < stopwhenunchanged)
    {
        //cout << "Generation " << generation+1 << ":" << endl;
        /* Generate new progeny by randomly assigning mating pairs from population */
        for(int i = 0; i < N_offspring; i++)
        {
            vector < int > offspring(femalecand,0);
            /* Select two parents based on binary tournament selection; randomly grab two and chose one with higher fitness */
            int ParentA, ParentB;
            /* Select Parent A */
            vector <int> parentcan1(2,-5);
            for(int j = 0; j < 2; j++)
            {
                parentcan1[j] = int((distribution(gen)) * size);
                if(parentcan1[1] == parentcan1[0]){j = 0;}
            }
            //cout << parentcan1[0] << " " << parentcan1[1] << endl;
            if(oldpopfitness[parentcan1[0]] > parentcan1[1])
            {
                ParentA = parentcan1[0];
            } else {ParentA = parentcan1[1];}
            /* Select Parent B */
            vector <int> parentcan2(2,-5);
            for(int j = 0; j < 2; j++)
            {
                parentcan2[j] = int((distribution(gen)) * size);
                if(parentcan2[1] == parentcan2[0]){j = 0;}
            }
            //cout << parentcan2[0] << " " << parentcan2[1] << endl;
            if(oldpopfitness[parentcan2[0]] > parentcan2[1])
            {
                ParentB = parentcan2[0];
            } else {ParentB = parentcan2[1];}
            //cout << ParentA << " " << ParentB << endl;
            //for(int j = 0; j < femalecand; j++){cout << oldpop[ParentA][j] << " ";}
            //cout << endl;
            //for(int j = 0; j < femalecand; j++){cout << oldpop[ParentB][j] << " ";}
            //cout << endl;
            /* Random crossover: start at ParentA then go to ParentB */
            int crossoverlocation = int((distribution(gen)) * femalecand);
            //cout << crossoverlocation << endl;
            /* Declare which one to start with */
            double order = distribution(gen);
            //cout << order << endl;
            if(order < 0.5)
            {
                for(int j = 0; j < crossoverlocation; j++){offspring[j] = oldpop[ParentA][j];}
                for(int j = crossoverlocation; j < femalecand; j++){offspring[j] = oldpop[ParentB][j];}
            } else {
                for(int j = 0; j < crossoverlocation; j++){offspring[j] = oldpop[ParentB][j];}
                for(int j = crossoverlocation; j < femalecand; j++){offspring[j] = oldpop[ParentA][j];}
            }
            //for(int j = 0; j < femalecand; j++){cout << offspring[j] << " ";}
            //cout << endl;
            /* Mutation Swap two elements */
            if(stopwhenunchanged > 500)
            {
                for(int mutrep = 0; mutrep < 2; mutrep++)
                {
                    vector < int > swapelements(2,0);
                    for(int j = 0; j < 2; j++)
                    {
                        swapelements[j] = int((distribution(gen)) * femalecand);
                        if(swapelements[1] == swapelements[0]){j = 0;}
                    }
                    //cout << swapelements[0] << " " << swapelements[1] << endl;
                    int temp = offspring[swapelements[0]];
                    offspring[swapelements[0]] = offspring[swapelements[1]]; offspring[swapelements[1]] = temp;
                }
            }
            if(stopwhenunchanged < 500)
            {
                for(int mutrep = 0; mutrep < 1; mutrep++)
                {
                    vector < int > swapelements(2,0);
                    for(int j = 0; j < 2; j++)
                    {
                        swapelements[j] = int((distribution(gen)) * femalecand);
                        if(swapelements[1] == swapelements[0]){j = 0;}
                    }
                    //cout << swapelements[0] << " " << swapelements[1] << endl;
                    int temp = offspring[swapelements[0]];
                    offspring[swapelements[0]] = offspring[swapelements[1]]; offspring[swapelements[1]] = temp;
                }
            }
            //for(int j = 0; j < resource[0].size(); j++){cout << offspring[j] << " ";}
            //cout << endl;
            double offspringfitness = 0.0;
            double feasability = 0.0;
            vector <int> rowuse(malecand,0);
            for(int j = 0; j < femalecand; j++){rowuse[offspring[j]] += resource[offspring[j]][j];}
            double value = 0;
            for(int j = 0; j < malecand; j++)
            {
                double temp = (rowuse[j] / double(maxresourcerow[j])) - 1;
                if(temp > 0){value += temp;}
                if(temp <= 0){value += 0;}
            }
            feasability = value * (1/double(malecand));
            //cout << i << endl;
            if(feasability > 0)
            {
                if(generation % exchangealgorithm == 0)
                {
                    //cout << feasability << endl;
                    //for(int j = 0; j < malecand; j++){cout << rowuse[j] << " ";}
                    //cout << endl;
                    for(int j = 0; j < malecand; j++)
                    {
                        while(rowuse[j] > maxresourcerow[j])
                        {
                            int replace = int((distribution(gen)) * rowuse[j]);
                            //cout << replace << endl;
                            //cout << j << endl;
                            rowuse[j] -= 1;
                            /* First find where replace location is */
                            int searchlocation = 0; int found = 0;
                            int replacelocation = -5;
                            while(1)
                            {
                                if(offspring[searchlocation] == j && found == replace){replacelocation = searchlocation; break;}
                                if(offspring[searchlocation] == j && found != replace){searchlocation++; found++;}
                                if(offspring[searchlocation] != j){searchlocation++;}
                            }
                            //cout << replacelocation << endl;
                            vector <int> candidates;
                            vector <double> values;
                            for(int add = 0; add < rowuse.size(); add++)
                            {
                                if(rowuse[add] < maxresourcerow[j]){candidates.push_back(add); values.push_back(cost[add][replacelocation]);}
                            }
                            //for(int add = 0; add < candidates.size(); add++){cout << candidates[add] << " " << values[add] << endl;}
                            /* Find max value */
                            double maxval = values[0]; int maxcan = candidates[0];
                            for(int add = 1; add < candidates.size(); add++)
                            {
                                if(values[add] > maxval){maxval = values[add]; maxcan = candidates[add];}
                            }
                            //cout << maxval << " " << maxcan << endl;
                            //for(int j = 0; j < oldpop[0].size(); j++){cout << offspring[j] << " ";}
                            //cout << endl;
                            offspring[replacelocation] = maxcan;
                            rowuse[maxcan] += 1;
                            //for(int j = 0; j < oldpop[0].size(); j++){cout << offspring[j] << " ";}
                            //cout << endl;
                            //for(int j = 0; j < malecand; j++){cout << rowuse[j] << " ";}
                            //cout << endl;
                        }
                    }
                    feasability = 0.0;
                }
            }
            /* Calculate Fitness */
            stringstream strStreamM (stringstream::in | stringstream::out);
            for(int j = 0; j < femalecand; j++){offspringfitness += cost[offspring[j]][j]; strStreamM << offspring[j];}
            string offspringstr = strStreamM.str();
            //cout << offspringfitness << " " << offspringstr  << endl;
            if(feasability > 0)
            {
                double minvaluesub = oldpopfitnessInf[0];
                for(int i = 0; i < size; i++)
                {
                    if(oldpopfitnessInf[i] < minvalue){minvalue = oldpopfitnessInf[i];}
                }
                offspringfitness = minvalue * (1 - feasability);
            }
            /* If not already in the population replace the worst one */
            string foundduplicate = "NO";
            double minimumvalue = oldpopfitnessInf[0];
            double minimumvaluelocation = 0;
            for(int j = 0; j < size; j++)
            {
                if(stringmates[j] == offspringstr){foundduplicate = "YES";}
                if(oldpopfitnessInf[j] < minimumvalue){minimumvalue = oldpopfitnessInf[j]; minimumvaluelocation = j;}
            }
            //cout << minimumvalue << " " << minimumvaluelocation << " " << foundduplicate <<  endl;
            /* if founduplicate = no replace lowest value with newest */
            if(foundduplicate == "NO")
            {
                for(int j = 0; j < femalecand; j++){oldpop[minimumvaluelocation][j] = offspring[j];}
                oldpopfitness[minimumvaluelocation] = offspringfitness;
                oldpopinfeasability[minimumvaluelocation] = feasability;
                oldpopfitnessInf[minimumvaluelocation] = offspringfitness;
                stringmates[minimumvaluelocation] = offspringstr;
            }
        }
        //cout << "Generation " << generation << " population ----" << endl;
        //for(int i = 0; i < size; i++)
        //{
        //    for(int j = 0; j < resource[0].size(); j++){cout << oldpop[i][j] << " ";}
        //    cout <<" --- "<< oldpopfitness[i] <<"\t"<< oldpopinfeasability[i] <<"\t"<< oldpopfitnessInf[i] <<"\t"<< stringmates[i] << endl;
        //}
        double newbestsolution = bestsolutionvalue; meanpopulationfitness = 0.0;
        for(int i = 0; i < size; i++)
        {
            if(oldpopfitness[i] > newbestsolution && oldpopinfeasability[i] == 0){newbestsolution = oldpopfitnessInf[i]; bestsolutionrow = i;}
            meanpopulationfitness += oldpopfitnessInf[i];
        }
        meanpopulationfitness = meanpopulationfitness / double(size);
        if(newbestsolution == bestsolutionvalue){unchanged++;}
        if(newbestsolution != bestsolutionvalue){unchanged = 0;}
        bestsolutionvalue = oldpopfitness[bestsolutionrow];
        for(int i = 0; i < femalecand; i++){best_solution[i] = oldpop[bestsolutionrow][i];}
        //if(generation % 100 == 0)
        //{
        //    /* Calculate genetic diversity */
        //    double sdfitness = 0.0;
        //    for(int i = 0; i < size; i++)
        //    {
        //        sdfitness += (oldpopfitnessInf[i] - meanpopulationfitness) * (oldpopfitnessInf[i] - meanpopulationfitness);
        //    }
        //    sdfitness = sdfitness / double(size-1);
        //    cout << "Current Best Solution (" << generation << "-" << unchanged << "): ";
        //    //for(int i = 0; i < best_solution.size(); i++){cout << best_solution[i] << " ";}
        //    cout << " --- " << bestsolutionvalue << " Mean Fitness (+/- SD): " << meanpopulationfitness << " (" << sdfitness << ")" << endl;
        //}
        generation++;
        //if(generation == 100){exit (EXIT_FAILURE);}
    }
    if(globalminimum < 0.0)
    {
        for(int i = 0; i < cost.size(); i++){for(int j = 0; j < cost[0].size(); j++){cost[i][j] -= abs(globalminimum);}}
    }
    for(int i = 0; i < size; i++)
    {
        double sum = 0.0;
        for(int j = 0; j < femalecand; j++){sum += cost[oldpop[i][j]][j];}
        oldpopfitness[i] = sum; oldpopfitnessInf[i] = sum;
    }
    double finalbestsolution = oldpopfitnessInf[0]; int finalbestsolutionrow = 0;
    for(int i = 1; i < size; i++)
    {
        if(oldpopfitnessInf[i] > finalbestsolution && oldpopinfeasability[i] == 0){finalbestsolution = oldpopfitnessInf[i]; bestsolutionrow = i;}
    }
    //cout << "Final Best Solution " << endl;
    //for(int j = 0; j < femalecand; j++){cout << oldpop[bestsolutionrow][j] << " ";}
    //cout << " --- " << finalbestsolution << endl << endl;
    vector <int> finalrowuse(malecand,0);
    for(int j = 0; j < femalecand; j++){finalrowuse[oldpop[bestsolutionrow][j]] += resource[oldpop[bestsolutionrow][j]][j];}
    //for(int j = 0; j < malecand; j++){cout << finalrowuse[j] << " ";}
    //cout << endl;
    vector <int> dam_mate_column(femalecand,-5);
    for(int j = 0; j < femalecand; j++){dam_mate_column[j] = oldpop[bestsolutionrow][j];}
    //for(int j = 0; j < femalecand; j++){cout << dam_mate_column[j] << " ";}
    //cout << endl << endl;
    vector <int> alreadyused(sireIDs.size(),0);                   /* Determines dam has already been chosen for a sire */
    vector <int> sire_mate_column(sireIDs.size(),-5);             /* determines which sire gets mated to a particular dam */
    int matingsfound = 0;
    /* Now fill sire_mate_column and update mating individual class */
    for(int i = 0; i < sireIDs.size(); i++)
    {
        //cout << sireIDs[i] << " ";
        int searchlocation = 0;
        /* now find in sireid */
        while(1)
        {
            if(sireIDs[i] == sireid[searchlocation]){break;}
            if(sireIDs[i] != sireid[searchlocation]){searchlocation++;}
        }
        //cout << searchlocation << " " << sireid[searchlocation] << endl;
        /* once found now now search for it in dam_mate_column */
        int searchlocationa = 0;
        while(searchlocationa < dam_mate_column.size())
        {
            if(dam_mate_column[searchlocationa] == searchlocation && alreadyused[searchlocationa] == 0)
            {
                sire_mate_column[i] = searchlocationa; alreadyused[searchlocationa] = -5; matingsfound++; break;
            }
            if(dam_mate_column[searchlocationa] != searchlocation || alreadyused[searchlocationa] != 0){searchlocationa++;}
        }
        //cout << searchlocationa << endl;
        //for(int j = 0; j < sire_mate_column.size(); j++)
        //{
        //    if(sire_mate_column[j] != -5){cout << sireIDs[j] << " " << sire_mate_column[j] << endl; }
        //}
        //cout << matingsfound << endl;
        //if(matingsfound > 0)
        //{
        //    cout << sireIDs[i] << endl;
        //    cout << searchlocation << " " << sireid[searchlocation] << endl;
        //    exit(EXIT_FAILURE);
        //}
    }
    //for(int i = 0; i < sireIDs.size(); i++){cout << sire_mate_column[i] << " - ";}
    //cout << endl;
    int doublecheckmate = 0; int doublechecknomate = 0;
    for(int i = 0; i < sire_mate_column.size(); i++)
    {
        if(sire_mate_column[i] == -5){doublechecknomate += 1;}
        if(sire_mate_column[i] != -5 && sire_mate_column[i] >= 0){doublecheckmate += 1;}
    }
    if(dam_mate_column.size() != doublecheckmate){cout << endl << "Mating Pairs Don't Match Up!" << endl; exit (EXIT_FAILURE);}
    updatematings(matingindividuals, sire_mate_column, sireIDs, damIDs);
}



