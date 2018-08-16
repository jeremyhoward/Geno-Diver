#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <random>
#include <algorithm>
#include <iterator>
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

using namespace std;

/*********************************/
/* Relationship Matrix Functions */
/*********************************/
void pedigree_relationship(outputfiles &OUTPUTFILES, vector <int> const &parent_id, double* output_subrelationship);
void grm_noprevgrm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);
void generaterohmatrix(vector <Animal> &population,vector < hapLibrary > &haplib,vector <int> const &parentID, double* _rohrm);
void matinggrm_maf(parameters &SimParameters,vector < string > &genotypes,double* output_grm,ostream& logfileloc);

/*********************************/
/* Calculate Selection Intensity */
/*********************************/
double SelectionIntensity(double prob, double mu, double sigma);



//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*******                             Breeding Population Age Distribution                             *******/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void breedingagedistribution(vector <Animal> &population,parameters &SimParameters,ostream& logfileloc)
{
    vector < int > maleagecount(SimParameters.getMaxAge(),0);
    vector < int > femaleagecount(SimParameters.getMaxAge(),0);
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getSex() == 0){maleagecount[population[i].getAge()-1] += 1;}
        if(population[i].getSex() == 1){femaleagecount[population[i].getAge()-1] += 1;}
    }
    logfileloc << "            Males \t Females" << endl;
    for(int i = 0; i < maleagecount.size(); i++)
    {
        logfileloc << "    - Age " << i+1 << ": " << maleagecount[i] << "\t\t" << femaleagecount[i] << endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*******                                  Selection Functions                                         *******/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////
// Truncation Selection //
//////////////////////////
void truncationselection(vector <Animal> &population,parameters &SimParameters,string tempselectionscen,int Gen,outputfiles &OUTPUTFILES,globalpopvar &Population1,ostream& logfileloc)
{
    double MaleCutOff = 0.0; double FemaleCutOff = 0.0;                     /* Value cutoff for males or females */
    int male = 0; int female= 0;                                            /* number of male or female animals that are of age 1 */
    vector < double > MaleValue; vector < double > FemaleValue;             /* array to hold male or female random values that are of age 1*/
    int maleparent = 0; int femaleparent = 0;                               /* number of male or female parents */
    /* if don't have enough parents based on culling proportion than need to save more progeny; only need to worry about this in Gen >= 1 */
    if(Gen > 1)
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){maleparent++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){femaleparent++;}
        }
    }
    for(int i = 0; i < population.size(); i++)                                  /* Place animals into correct array based on sex*/
    {
        if(population[i].getSex() == 0 && population[i].getAge() == 1)                                      /* if male (i.e. 0) */
        {
            if(tempselectionscen == "random"){MaleValue.push_back(population[i].getRndSelection()); male++;}        /* Random Selection */
            if(tempselectionscen == "phenotype"){MaleValue.push_back((population[i].get_Phenvect())[0]); male++;}   /* Phenotypic Selection */
            if(tempselectionscen == "tbv"){MaleValue.push_back((population[i].get_GVvect())[0]); male++;}           /* TBV Selection */
            if(tempselectionscen == "ebv"){MaleValue.push_back((population[i].get_EBVvect())[0]); male++;}          /* EBV Selection */
            if(tempselectionscen == "index_tbv"){MaleValue.push_back(population[i].gettbvindex()); male++;}         /* Index TBV Selection */
            if(tempselectionscen == "index_ebv"){MaleValue.push_back(population[i].getebvindex()); male++;}         /* Index EBV Selection */
        }
        if(population[i].getSex() == 1 && population[i].getAge() == 1)                                      /* if female (i.e. 1) */
        {
            if(tempselectionscen == "random"){FemaleValue.push_back(population[i].getRndSelection()); female++;}     /* Random Selection */
            if(tempselectionscen == "phenotype"){FemaleValue.push_back((population[i].get_Phenvect())[0]); female++;}/* Phenotypic Selection */
            if(tempselectionscen == "tbv"){FemaleValue.push_back((population[i].get_GVvect())[0]); female++;}        /* TBV Selection */
            if(tempselectionscen == "ebv"){FemaleValue.push_back((population[i].get_EBVvect())[0]); female++;}       /* EBV Selection */
            if(tempselectionscen == "index_tbv"){FemaleValue.push_back(population[i].gettbvindex()); female++;}      /* Index TBV Selection */
            if(tempselectionscen == "index_ebv"){FemaleValue.push_back(population[i].getebvindex()); female++;}      /* Index EBV Selection */
        }
    }
    //cout << maleparent << " " << femaleparent << endl;
    //cout << male << " " << female << endl;
    //cout << MaleValue.size() << " " << FemaleValue.size() << endl;
    /* Array with correct value based on how selection is to proceed created now sort */
    if(SimParameters.getSelectionDir() == "low" || tempselectionscen == "random")         /* sort lowest to highest */
    {
        double temp;
        for(int i = 0; i < male -1; i++)                               /* Sort Males */
        {
            for(int j=i+1; j< male; j++){if(MaleValue[i] > MaleValue[j]){temp = MaleValue[i]; MaleValue[i] = MaleValue[j]; MaleValue[j] = temp;}}
        }
        for(int i = 0; i < female -1; i++)                             /* Sort Females */
        {
            for(int j=i+1; j< female; j++)
            {
                if(FemaleValue[i] > FemaleValue[j]){temp = FemaleValue[i]; FemaleValue[i] = FemaleValue[j]; FemaleValue[j] = temp;}
            }
        }
    }
    if(SimParameters.getSelectionDir() == "high" && tempselectionscen != "random")   /* Sort lowest to highest then reverse order so highest to lowest */
    {
        double temp;
        for(int i = 0; i < male -1; i++)                               /* Sort Males */
        {
            for(int j=i+1; j< male; j++){if(MaleValue[i] < MaleValue[j]){temp = MaleValue[i]; MaleValue[i] = MaleValue[j]; MaleValue[j] = temp;}}
        }
        for(int i = 0; i < female -1; i++)                             /* Sort Females */
        {
            for(int j=i+1; j< female; j++)
            {
                if(FemaleValue[i] < FemaleValue[j]){temp = FemaleValue[i]; FemaleValue[i] = FemaleValue[j]; FemaleValue[j] = temp;}
            }
        }
    }
    //for(int i = 0; i < FemaleValue.size(); i++){cout << FemaleValue[i] << " ";}
    //cout << endl;
    //for(int i = 0; i < MaleValue.size(); i++){cout << MaleValue[i] << " ";}
    //cout << endl;
    /* Array sorted now determine cutoff value any animal above this line will be removed */
    logfileloc << "       - Number of Male Selection Candidates: " << male << "." <<  endl;
    logfileloc << "       - Number of Female Selection Candidates: " << female << "." << endl;
    logfileloc << "       - Number of Male Parents: " << maleparent << "." << endl;
    logfileloc << "       - Number of Female Parents: " << femaleparent << "." << endl;
    if(Gen == 1)
    {
        if(SimParameters.getSires() > male)
        {
            logfileloc << endl << "   Program Ended - Not Enough Male Selection Candidates in GEN 1. Add More Founders!" << endl;
            exit (EXIT_FAILURE);
        }
        if(SimParameters.getDams() > female)
        {
            logfileloc << endl << "   Program Ended - Not Enough Female Selection Candidates in GEN 1. Add More Founders!" << endl;
            exit (EXIT_FAILURE);
        }
    }
    int malesadd = 0; int femalesadd = 0;
    if(Gen > 1)
    {
        /* add 0.5 due to potential rounding error */
        int malesneeded=(SimParameters.getSires()*SimParameters.getSireRepl())+((SimParameters.getSires()*(1-SimParameters.getSireRepl()))-maleparent)+0.5;
        int femalesneeded=(SimParameters.getDams()*SimParameters.getDamRepl())+((SimParameters.getDams()*(1-SimParameters.getDamRepl()))-femaleparent)+0.5;
        if(male < malesneeded){logfileloc << "       - Program Ended - Not enough new male progeny. " << endl; exit (EXIT_FAILURE);}
        if(female < femalesneeded){logfileloc << "       - Program Ended - Not enough new female progeny. " << endl; exit (EXIT_FAILURE);}
        if(maleparent < ((SimParameters.getSires() * (1 - SimParameters.getSireRepl()))))
        {
            malesadd = (SimParameters.getSires() * (1 - SimParameters.getSireRepl())) - maleparent;
            logfileloc << "        - Number of male parents too small need to keep more!!" << endl;
            logfileloc << "            - Kept " << malesadd << " extra progeny." << endl;
        }
        if(femaleparent < int(((SimParameters.getDams() * (1 - SimParameters.getDamRepl()))+0.5)))
        {
            femalesadd = (SimParameters.getDams() * (1 - SimParameters.getDamRepl())) - femaleparent;
            logfileloc << "        - Number of female parents too small need to keep more!!" << endl;
            logfileloc << "            - Kept " << femalesadd << " extra progeny." << endl;
        }
    }
    int malepos, femalepos;
    if(Gen == 1){malepos = SimParameters.getSires(); femalepos = SimParameters.getDams();}    /* Grabs Position based on percentile in Males or Females */
    /* If under the male and female value need to keep more selection candidates */
    if(Gen > 1)
    {
        malepos = (SimParameters.getSires() * SimParameters.getSireRepl()) + malesadd;
        femalepos = (SimParameters.getDams() * SimParameters.getDamRepl()) + femalesadd;
    }
    /* vectors to store info on number of full-sib families */
    vector < string > siredamkept(population.size(),"");
    vector <int> keep(population.size(),0);
    vector < double > siredamvalue(population.size(),0.0);
    string Action;                               /* Based on Selection used will result in an action that is common to all */
    //cout << malepos << " " << femalepos << endl;
    //cout << MaleValue[malepos] << " " << FemaleValue[femalepos] << endl;
    int nummaleselected = 0; int numfemaleselected = 0;
    for(int i = 0; i < population.size(); i++)
    {
        while(1)
        {
            if(tempselectionscen == "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getRndSelection()>MaleValue[malepos-1]){Action="RM_M";break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getRndSelection()<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getRndSelection()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getRndSelection()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}   /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "phenotype" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]<=MaleValue[malepos-1]){Action = "KP_M"; break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}   /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "phenotype" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_Phenvect())[0]>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "tbv" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0&&population[i].getAge()==1&&(population[i].get_GVvect())[0]>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0&&population[i].getAge()==1&&(population[i].get_GVvect())[0]<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&(population[i].get_GVvect())[0]>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&(population[i].get_GVvect())[0]<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "tbv" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0&&population[i].getAge()==1&&(population[i].get_GVvect())[0]<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0&&population[i].getAge()==1&&(population[i].get_GVvect())[0]>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&(population[i].get_GVvect())[0]<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&(population[i].get_GVvect())[0]>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "ebv" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "ebv" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && (population[i].get_EBVvect())[0]>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "index_tbv" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].gettbvindex()>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].gettbvindex()<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].gettbvindex()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].gettbvindex()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "index_tbv" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].gettbvindex()<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].gettbvindex()>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].gettbvindex()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].gettbvindex()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "index_ebv" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getebvindex()>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getebvindex()<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getebvindex()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getebvindex()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "index_ebv" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getebvindex()<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getebvindex()>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getebvindex()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getebvindex()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
        }
        stringstream kpmaleparents(stringstream::out); kpmaleparents << population[i].getSire() << "_" << population[i].getDam();
        siredamkept[i] = kpmaleparents.str();
        if(Action == "RM_M" || Action == "RM_F"){keep[i] = 0;}
        if(Action == "KP_M" || Action == "KP_F")
        {
            if(Action == "KP_M"){nummaleselected++;}
            if(Action == "KP_F"){numfemaleselected++;}
            keep[i] = 1;
        }
        if(Action == "Old_Animal"){keep[i] = 2;}
        if(tempselectionscen == "random"){siredamvalue[i] = population[i].getRndSelection();}
        if(tempselectionscen == "phenotype"){siredamvalue[i] = (population[i].get_Phenvect())[0];}
        if(tempselectionscen == "tbv"){siredamvalue[i] = (population[i].get_GVvect())[0];}
        if(tempselectionscen == "ebv"){siredamvalue[i] = (population[i].get_EBVvect())[0];}
        if(tempselectionscen == "index_tbv"){siredamvalue[i] = population[i].gettbvindex();}
        if(tempselectionscen == "index_ebv"){siredamvalue[i] = population[i].getebvindex();}
    }
    /* Sometimes if males and females within a litter have same EBV (i.e. no phenotype) will grab too much */
    if(nummaleselected != malepos || numfemaleselected != femalepos)
    {
        vector <int> maleloc; vector <int> femaleloc;
        for(int i = 0; i < population.size(); i++)
        {
            if(tempselectionscen == "random"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "phenotype"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "tbv"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "index_tbv"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "index_ebv")
            {
                if(population[i].getSex() == 0 && population[i].getebvindex() == MaleValue[malepos-1]){maleloc.push_back(i);}
                if(population[i].getSex() == 1 && population[i].getebvindex() == FemaleValue[femalepos-1]){femaleloc.push_back(i);}
            }
            if(tempselectionscen == "ebv")
            {
                if(population[i].getSex() == 0 && (population[i].get_EBVvect())[0] == MaleValue[malepos-1]){maleloc.push_back(i);}
                if(population[i].getSex() == 1 && (population[i].get_EBVvect())[0] == FemaleValue[femalepos-1]){femaleloc.push_back(i);}
            }
        }
        /* If over randomly remove the required number of animals within a family with same EBV */
        for(int i = 0; i < (nummaleselected - malepos); i++){keep[maleloc[i]] = 0;}
        for(int i = 0; i < (numfemaleselected - femalepos); i++){keep[femaleloc[i]] = 0;}
        int updatedcountmale = 0; int updatedcountfemale = 0;
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() == 1){updatedcountmale += keep[i];}
            if(population[i].getSex() == 1 && population[i].getAge() == 1){updatedcountfemale += keep[i];}
        }
    }
    /* Remove Unselected based on certain selection used Animals from class object and then resize vector from population */
    if(Gen > 1 && SimParameters.getOffspring() > 1)
    {
        string conditionmet = "NO";
        while(conditionmet  == "NO")
        {
            /* now loop through all siredamkept and tabulate all matings */
            vector < string > tempsiredam; vector < int > tempsiredamnum;
            for(int i = 0; i < siredamkept.size(); i++)
            {
                if(keep[i] == 1)
                {
                    if(tempsiredam.size() > 0)
                    {
                        int location_temp = 0;
                        while(location_temp < tempsiredam.size())
                        {
                            if(tempsiredam[location_temp] == siredamkept[i]){tempsiredamnum[location_temp] += 1; break;}
                            if(tempsiredam[location_temp] != siredamkept[i]){location_temp += 1;}
                        }
                        if(location_temp == tempsiredam.size()){tempsiredam.push_back(siredamkept[i]); tempsiredamnum.push_back(1);}
                    }
                    if(tempsiredam.size() == 0){tempsiredam.push_back(siredamkept[i]); tempsiredamnum.push_back(1);}
                }
            }
            /* now tabulate number within each category */
            vector < int > numberwithineachfullsiba (SimParameters.getOffspring(),0);
            for(int i = 0; i < tempsiredam.size(); i++){numberwithineachfullsiba[tempsiredamnum[i]-1] += 1;}
            //for(int i = 0; i < numberwithineachfullsiba.size(); i++){cout << numberwithineachfullsiba[i] << " ";}
            //cout << endl;
            int numgreaterthanmax = 0;
            for(int i = SimParameters.getmaxmating(); i < numberwithineachfullsiba.size(); i++){numgreaterthanmax += numberwithineachfullsiba[i];}
            //cout << numgreaterthanmax << endl;
            /* remove animals with full sib families over the limit */
            if(numgreaterthanmax > 0)
            {
                for(int fam = 0; fam < tempsiredam.size(); fam++)
                {
                    if(tempsiredamnum[fam] > SimParameters.getmaxmating())
                    {
                        vector < double > fullsibvalues;
                        for(int i = 0; i < siredamkept.size(); i++)
                        {
                            if(siredamkept[i] == tempsiredam[fam] && keep[i] == 1){fullsibvalues.push_back(siredamvalue[i]);}
                        }
                        //for(int i = 0; i < fullsibvalues.size(); i++){cout << fullsibvalues[i] << " ";}
                        /* sort lowest to highest */
                        if(SimParameters.getSelectionDir() == "low" && tempselectionscen == "random")
                        {
                            double temp;
                            for(int i = 0; i < fullsibvalues.size() -1; i++)                                     /* Sort Males */
                            {
                                for(int j=i+1; j< fullsibvalues.size(); j++)
                                {
                                    if(fullsibvalues[i] > fullsibvalues[j])
                                    {
                                        temp = fullsibvalues[i]; fullsibvalues[i] = fullsibvalues[j]; fullsibvalues[j] = temp;
                                    }
                                }
                            }
                        }
                        /* Sort lowest to highest then reverse order so highest to lowest */
                        if(SimParameters.getSelectionDir() == "high" && tempselectionscen != "random")
                        {
                            double temp;
                            for(int i = 0; i < fullsibvalues.size() -1; i++)                                     /* Sort Males */
                            {
                                for(int j=i+1; j< fullsibvalues.size(); j++)
                                {
                                    if(fullsibvalues[i] < fullsibvalues[j])
                                    {
                                        temp = fullsibvalues[i]; fullsibvalues[i] = fullsibvalues[j]; fullsibvalues[j] = temp;
                                    }
                                }
                            }
                        }
                        /* now replace with next best animal */
                        for(int i = SimParameters.getmaxmating(); i < fullsibvalues.size(); i++)
                        {
                            /* Find Individual */
                            string found = "NO"; int startpoint = 0;
                            while(found == "NO")
                            {
                                if(tempselectionscen == "random")
                                {
                                    if(fullsibvalues[i] == population[startpoint].getRndSelection() && keep[startpoint] == 1){found = "YES";}
                                    if(fullsibvalues[i] != population[startpoint].getRndSelection()){startpoint++;}
                                }
                                if(tempselectionscen == "phenotype")
                                {
                                    if(fullsibvalues[i] == (population[startpoint].get_Phenvect())[0] && keep[startpoint] == 1){found = "YES";}
                                    if(fullsibvalues[i] != (population[startpoint].get_Phenvect())[0]){startpoint++;}
                                }
                                if(tempselectionscen == "tbv")
                                {
                                    if(fullsibvalues[i] == (population[startpoint].get_GVvect())[0] && keep[startpoint] == 1){found = "YES";}
                                    if(fullsibvalues[i] != (population[startpoint].get_GVvect())[0]){startpoint++;}
                                }
                                if(tempselectionscen == "ebv")
                                {
                                    if(fullsibvalues[i] != (population[startpoint].get_EBVvect())[0]){startpoint++;}
                                    if(fullsibvalues[i] == (population[startpoint].get_EBVvect())[0] && keep[startpoint] == 0){startpoint++;}
                                    if(fullsibvalues[i] == (population[startpoint].get_EBVvect())[0] && keep[startpoint] == 1){found = "YES";}
                                }
                                if(tempselectionscen == "index_tbv")
                                {
                                    if(fullsibvalues[i] != population[startpoint].gettbvindex()){startpoint++;}
                                    if(fullsibvalues[i] == population[startpoint].gettbvindex() && keep[startpoint] == 0){startpoint++;}
                                    if(fullsibvalues[i] == population[startpoint].gettbvindex() && keep[startpoint] == 1){found = "YES";}
                                }
                                if(tempselectionscen == "index_ebv")
                                {
                                    if(fullsibvalues[i] != population[startpoint].getebvindex()){startpoint++;}
                                    if(fullsibvalues[i] == population[startpoint].getebvindex() && keep[startpoint] == 0){startpoint++;}
                                    if(fullsibvalues[i] == population[startpoint].getebvindex() && keep[startpoint] == 1){found = "YES";}
                                }
                            }
                            keep[startpoint] = 0;                   /* No longer select animal */
                            if(population[startpoint].getSex() == 0)
                            {
                                malepos = malepos + 1;        /* next best candidate value */
                                string founda = "NO"; int startpointa = 0;
                                while(founda == "NO")
                                {
                                    if(tempselectionscen == "random")
                                    {
                                        if(MaleValue[malepos-1] == population[startpointa].getRndSelection() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1] != population[startpointa].getRndSelection()){startpointa++;}
                                    }
                                    if(tempselectionscen == "phenotype")
                                    {
                                        if(MaleValue[malepos-1] == (population[startpointa].get_Phenvect())[0] && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1] != (population[startpointa].get_Phenvect())[0]){startpointa++;}
                                    }
                                    if(tempselectionscen == "tbv")
                                    {
                                        if(MaleValue[malepos-1] == (population[startpointa].get_GVvect())[0] && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1] != (population[startpointa].get_GVvect())[0]){startpointa++;}
                                    }
                                    if(tempselectionscen == "ebv")
                                    {
                                        if(MaleValue[malepos-1]==(population[startpointa].get_EBVvect())[0] && keep[startpointa]==0 && population[startpointa].getSex()==0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1]==(population[startpointa].get_EBVvect())[0] && keep[startpointa]==1 && population[startpointa].getSex()==0)
                                        {
                                            startpointa++;
                                        }
                                        if(MaleValue[malepos-1]==(population[startpointa].get_EBVvect())[0] && population[startpointa].getSex()==1){startpointa++;}
                                        if(MaleValue[malepos-1] != (population[startpointa].get_EBVvect())[0]){startpointa++;}
                                        
                                    }
                                    if(tempselectionscen == "index_tbv")
                                    {
                                        if(MaleValue[malepos-1] == population[startpointa].gettbvindex() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1] != population[startpointa].gettbvindex()){startpointa++;}
                                    }
                                    if(tempselectionscen == "index_ebv")
                                    {
                                        if(MaleValue[malepos-1] == population[startpointa].getebvindex() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1]==population[startpointa].getebvindex() && keep[startpointa]==1 && population[startpointa].getSex()==0)
                                        {
                                            startpointa++;
                                        }
                                        if(MaleValue[malepos-1]==population[startpointa].getebvindex() && population[startpointa].getSex()==1){startpointa++;}
                                        
                                        if(MaleValue[malepos-1] != population[startpointa].getebvindex()){startpointa++;}
                                    }
                                }
                            }
                            if(population[startpoint].getSex() == 1)
                            {
                                femalepos  = femalepos + 1;    /* next best candidate value */
                                string founda = "NO"; int startpointa = 0;
                                while(founda == "NO")
                                {
                                    if(tempselectionscen == "random")
                                    {
                                        if(FemaleValue[femalepos-1] == population[startpointa].getRndSelection() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1] != population[startpointa].getRndSelection()){startpointa++;}
                                    }
                                    if(tempselectionscen == "phenotype")
                                    {
                                        if(FemaleValue[femalepos-1] == (population[startpointa].get_Phenvect())[0] && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1] != (population[startpointa].get_Phenvect())[0]){startpointa++;}
                                    }
                                    if(tempselectionscen == "tbv")
                                    {
                                        if(FemaleValue[femalepos-1] == (population[startpointa].get_GVvect())[0] && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1] != (population[startpointa].get_GVvect())[0]){startpointa++;}
                                    }
                                    if(tempselectionscen == "ebv")
                                    {
                                        if(FemaleValue[femalepos-1]==(population[startpointa].get_EBVvect())[0] && keep[startpointa]==0 && population[startpointa].getSex()==1)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1]==(population[startpointa].get_EBVvect())[0] && keep[startpointa]==1 && population[startpointa].getSex()==1)
                                        {
                                            startpointa++;
                                        }
                                        if(FemaleValue[femalepos-1]==(population[startpointa].get_EBVvect())[0] && population[startpointa].getSex()==0){startpointa++;}
                                        if(FemaleValue[femalepos-1] != (population[startpointa].get_EBVvect())[0]){startpointa++;}
                                    }
                                    if(tempselectionscen == "index_tbv")
                                    {
                                        if(FemaleValue[femalepos-1] == population[startpointa].gettbvindex() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1] != population[startpointa].gettbvindex()){startpointa++;}
                                    }
                                    if(tempselectionscen == "index_ebv")
                                    {
                                        if(FemaleValue[femalepos-1] == population[startpointa].getebvindex() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1]==population[startpointa].getebvindex() && keep[startpointa]==1 && population[startpointa].getSex()==1)
                                        {
                                            startpointa++;
                                        }
                                        if(FemaleValue[femalepos-1] == population[startpointa].getebvindex() && population[startpointa].getSex()==0){startpointa++;}
                                        if(FemaleValue[femalepos-1] != population[startpointa].getebvindex()){startpointa++;}
                                    }
                                    //cout << startpointa << " ";
                                }
                                //cout << endl;
                            }
                        }
                    }
                }
            }
            if(numgreaterthanmax == 0)
            {
                logfileloc << "       - Number of Full-Sibs selected within a family: " << endl;
                for(int i = 0; i < numberwithineachfullsiba.size(); i++)
                {
                    logfileloc << "            -" << i + 1 << " sibling selected from family: " << numberwithineachfullsiba[i] << "." <<  endl;
                }
                conditionmet = "YES";
            }
        }
    }
    male = 0; female = 0;
    if(tempselectionscen != "random")
    {
        /* Figure figure out number of age classes */
        vector < int > sire; vector < int > dam; vector <int> agesubclass;
        for(int i =0; i < population.size(); i++)                   /* loop through all progeny and grab sire and dam */
        {
            if(population[i].getAge() == 1)
            {
                if(population[i].getSire() != 0){sire.push_back(population[i].getSireAge());}
                if(population[i].getDam() != 0){dam.push_back(population[i].getDamAge());}
                //if(population[i].getDam() != 0 && population[i].getSire() != 0)
                //{
                //    stringstream converter; converter << population[i].getSireAge() << population[i].getDamAge();
                //    agesubclass.push_back(atoi((converter.str()).c_str()));
                //}
            }
        }
        if(sire.size() > 0 & dam.size() > 0)
        {
            //sort(agesubclass.begin(),agesubclass.end() ); agesubclass.erase(unique(agesubclass.begin(),agesubclass.end()),agesubclass.end());
            //cout << agesubclass.size() << endl;
            //for(int i = 0; i < agesubclass.size(); i++){cout << agesubclass[i] << " ";}
            //cout << endl;
            /* remove duplicates */
            sort(sire.begin(),sire.end() ); sire.erase(unique(sire.begin(),sire.end()),sire.end());
            sort(dam.begin(),dam.end() ); dam.erase(unique(dam.begin(),dam.end()),dam.end());
            vector < int > sirerenum(sire.size(),0); vector < int > damrenum(dam.size(),0); /* in case age skips a number */
            for(int i = 0; i < sire.size(); i++){sirerenum[i] = i;}
            for(int i = 0; i < dam.size(); i++){damrenum[i] = i;}
            //cout << sire.size() << endl;
            //for(int i = 0; i < sire.size(); i++){cout << sire[i] << "-" << sirerenum[i] << " ";}
            //cout << endl;
            //for(int i = 0; i < dam.size(); i++){cout << dam[i] << "-" << damrenum[i] << " ";}
            //cout << endl;
            vector < vector <double>> SC_males; vector <int> SC_males_age; vector < vector <double>> SC_msel; vector <int> SC_msel_age;
            vector < vector <double>> SC_females; vector <int> SC_females_age; vector < vector <double>> SC_fsel; vector <int> SC_fsel_age;
            int numberoftraits;
            if(tempselectionscen == "phenotype" || tempselectionscen == "tbv" || tempselectionscen == "ebv"){numberoftraits = SimParameters.getnumbertraits();}
            if(tempselectionscen == "index_tbv" || tempselectionscen == "index_ebv"){numberoftraits = 1;}
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getSex() == 0 && population[i].getAge() == 1)                                      /* if male (i.e. 0) */
                {
                    vector < double > temp(numberoftraits,0.0); vector < double > tempa(numberoftraits,0.0);
                    for(int k = 0; k < numberoftraits; k++)
                    {
                        if(tempselectionscen == "phenotype")            /* Phenotypic Selection */
                        {
                            temp[k] = (population[i].get_Phenvect())[k];
                            if(keep[i] == 1){tempa[k] = (population[i].get_Phenvect())[k];}
                        }
                        if(tempselectionscen == "tbv")                  /* TBV Selection */
                        {
                            temp[k] = (population[i].get_GVvect())[k];
                            if(keep[i] == 1){tempa[k] = (population[i].get_GVvect())[k];}
                        }
                        if(tempselectionscen == "ebv")                  /* EBV Selection */
                        {
                            temp[k] = (population[i].get_EBVvect())[k];
                            if(keep[i] == 1){tempa[k] = (population[i].get_EBVvect())[k];}
                        }
                        if(tempselectionscen == "index_tbv")
                        {
                            temp[k] = population[i].gettbvindex();
                            if(keep[i] == 1){tempa[k] = population[i].gettbvindex();}
                        }
                        if(tempselectionscen == "index_ebv")
                        {
                            temp[k] = population[i].getebvindex();
                            if(keep[i] == 1){tempa[k] = population[i].getebvindex();}
                        }
                    }
                    SC_males_age.push_back(population[i].getSireAge()); SC_males.push_back(temp);
                    if(keep[i] == 1){SC_msel.push_back(tempa);SC_msel_age.push_back(population[i].getSireAge());}
                }
                if(population[i].getSex() == 1 && population[i].getAge() == 1)                                      /* if female (i.e. 1) */
                {
                    vector < double > temp(numberoftraits,0.0); vector < double > tempa(numberoftraits,0.0);
                    for(int k = 0; k < numberoftraits; k++)
                    {
                        if(tempselectionscen == "phenotype")            /* Phenotypic Selection */
                        {
                            temp[k] = (population[i].get_Phenvect())[k];
                            if(keep[i] == 1){tempa[k] = (population[i].get_Phenvect())[k];}
                        }
                        if(tempselectionscen == "tbv")              /* TBV Selection */
                        {
                            temp[k] = (population[i].get_GVvect())[k];
                            if(keep[i] == 1){tempa[k] = (population[i].get_GVvect())[k];}
                        }
                        if(tempselectionscen == "ebv")                  /* EBV Selection */
                        {
                            temp[k] = (population[i].get_EBVvect())[k];
                            if(keep[i] == 1){tempa[k] = (population[i].get_EBVvect())[k];}
                        }
                        if(tempselectionscen == "index_tbv")
                        {
                            temp[k] = population[i].gettbvindex();
                            if(keep[i] == 1){tempa[k] = population[i].gettbvindex();}
                        }
                        if(tempselectionscen == "index_ebv")
                        {
                            temp[k] = population[i].getebvindex();
                            if(keep[i] == 1){tempa[k] = population[i].getebvindex();}
                        }
                    }
                    SC_females_age.push_back(population[i].getDamAge()); SC_females.push_back(temp);
                    if(keep[i] == 1){SC_fsel.push_back(tempa); SC_fsel_age.push_back(population[i].getDamAge());}
                }
            }
            /* Loop through and rename age to index of where it would be in 2d array's */
            for(int i = 0; i < SC_msel_age.size(); i++)
            {
                int search = 0;
                while(1)
                {
                    if(SC_msel_age[i] == sire[search]){SC_msel_age[i] = sirerenum[search]; break;}
                    if(SC_msel_age[i] != sire[search]){search++;}
                    if(search >= sire.size()){cout << "Shouldn't be here" << endl; exit (EXIT_FAILURE);}
                }
            }
            for(int i = 0; i < SC_males_age.size(); i++)
            {
                int search = 0;
                while(1)
                {
                    if(SC_males_age[i] == sire[search]){SC_males_age[i] = sirerenum[search]; break;}
                    if(SC_males_age[i] != sire[search]){search++;}
                    if(search >= sire.size()){cout << "Shouldn't be here" << endl; exit (EXIT_FAILURE);}
                }
            }
            for(int i = 0; i < SC_fsel_age.size(); i++)
            {
                int search = 0;
                while(1)
                {
                    if(SC_fsel_age[i] == dam[search]){SC_fsel_age[i] = damrenum[search]; break;}
                    if(SC_fsel_age[i] != dam[search]){search++;}
                    if(search >= dam.size()){cout << "Shouldn't be here" << endl; exit (EXIT_FAILURE);}
                }
            }
            for(int i = 0; i < SC_females_age.size(); i++)
            {
                int search = 0;
                while(1)
                {
                    if(SC_females_age[i] == dam[search]){SC_females_age[i] = damrenum[search]; break;}
                    if(SC_females_age[i] != dam[search]){search++;}
                    if(search >= dam.size()){cout << "Shouldn't be here" << endl; exit (EXIT_FAILURE);}
                }
            }
            //cout << SC_males.size() << " " << SC_males[0].size() << " " << SC_males_age.size() << endl;
            //cout << SC_msel.size() << " " << SC_msel[0].size() << " " << SC_msel_age.size() << endl;
            //cout << SC_females.size() << " " << SC_females[0].size() << " " << SC_females_age.size() << endl;
            //cout << SC_fsel.size() << " " << SC_fsel[0].size() << " " << SC_fsel_age.size() << endl;
            //for(int i = 0; i < SC_males[0].size(); i++)
            //{
            //    double propsel = SC_msel.size() / double(SC_males.size());
            //    double selint = SelectionIntensity(propsel,0.0,1.0);
            //    cout << propsel << " " << selint << endl;
            //}
            //for(int i = 0; i < SC_females[0].size(); i++)
            //{
            //    double propsel = SC_fsel.size() / double(SC_females.size());
            //    double selint = SelectionIntensity(propsel,0.0,1.0);
            //    cout << propsel << " " << selint << endl;
            //}
            /* Now Loop Through each age class and calculate selection differential */
            vector < double > sire_numberwithinageclass_sel(sire.size(),0.0); vector < double > sire_numberwithinageclass(sire.size(),0.0);
            vector < double > dam_numberwithinageclass_sel(dam.size(),0.0); vector < double > dam_numberwithinageclass(dam.size(),0.0);
            vector < vector <double> > meanselected_m(sire.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > meanselected_f(dam.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > meanall_m(sire.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > meanall_f(dam.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > sdall_m(sire.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > sdall_f(dam.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > seli_m(sire.size(),std::vector<double>(numberoftraits,0.0));
            vector < vector <double> > seli_f(dam.size(),std::vector<double>(numberoftraits,0.0));
            /******************/
            /* First do Males */
            /******************/
            /* Selected Group */
            for(int i = 0; i < SC_msel.size(); i++)
            {
                for(int k = 0; k < numberoftraits; k++){meanselected_m[SC_msel_age[i]][k] += SC_msel[i][k];}
                sire_numberwithinageclass_sel[SC_msel_age[i]] += 1;
            }
            for(int i = 0; i < meanselected_m.size(); i++)
            {
                for(int j = 0; j < meanselected_m[i].size(); j++)
                {
                    if(sire_numberwithinageclass_sel[i] > 0){meanselected_m[i][j] /= double(sire_numberwithinageclass_sel[i]);}
                }
            }
            /*  All Animals  */
            for(int i = 0; i < SC_males.size(); i++)
            {
                for(int k = 0; k < numberoftraits; k++){meanall_m[SC_males_age[i]][k] += SC_males[i][k];}
                sire_numberwithinageclass[SC_males_age[i]] += 1;
            }
            for(int i = 0; i < meanall_m.size(); i++)
            {
                for(int j = 0; j < meanall_m[i].size(); j++){meanall_m[i][j] /= double(sire_numberwithinageclass[i]);}
            }
            /* All Animals SD */
            for(int i = 0; i < SC_males.size(); i++)
            {
                for(int k = 0; k < numberoftraits; k++)
                {
                    sdall_m[SC_males_age[i]][k] += (SC_males[i][k]-meanall_m[SC_males_age[i]][k])*(SC_males[i][k]-meanall_m[SC_males_age[i]][k]);
                }
            }
            for(int i = 0; i < sdall_m.size(); i++)
            {
                for(int j = 0; j < sdall_m[i].size(); j++)
                {
                    sdall_m[i][j] /= double(sire_numberwithinageclass[i]-1); sdall_m[i][j] = sqrt(sdall_m[i][j]);
                }
            }
            /* Calculate Selection Intensity */
            for(int i = 0; i < sdall_m.size(); i++)
            {
                for(int j = 0; j < sdall_m[i].size(); j++)
                {
                    if(sire_numberwithinageclass_sel[i] > 0){seli_m[i][j] = (meanselected_m[i][j] - meanall_m[i][j]) / double(sdall_m[i][j]);}
                }
            }
            //for(int i = 0; i < meanall_m.size(); i++)
            //{
            //    cout << "Males Age " << i+1 << ":";
            //    for(int j = 0; j < meanselected_m[i].size(); j++){cout << " " << meanselected_m[i][j];}
            //    cout <<"("<<sire_numberwithinageclass_sel[i]<<") -+-";
            //    for(int j = 0; j < meanall_m[i].size(); j++){cout << " " << meanall_m[i][j];}
            //    cout <<"("<<sire_numberwithinageclass[i]<<") -+-";
            //    for(int j = 0; j < sdall_m[i].size(); j++){cout << " " << sdall_m[i][j];}
            //    cout <<" -+-";
            //    for(int j = 0; j < seli_m[i].size(); j++){cout << " " << seli_m[i][j];}
            //    cout << endl;
            //}
            /********************/
            /* First do Females */
            /********************/
            /* Selected Group */
            for(int i = 0; i < SC_fsel.size(); i++)
            {
                for(int k = 0; k < numberoftraits; k++){meanselected_f[SC_fsel_age[i]][k] += SC_fsel[i][k];}
                dam_numberwithinageclass_sel[SC_fsel_age[i]] += 1;
            }
            for(int i = 0; i < meanselected_f.size(); i++)
            {
                for(int j = 0; j < meanselected_f[i].size(); j++)
                {
                    if(dam_numberwithinageclass_sel[i] > 0){meanselected_f[i][j] /= double(dam_numberwithinageclass_sel[i]);}
                }
            }
            /*  All Animals  */
            for(int i = 0; i < SC_females.size(); i++)
            {
                for(int k = 0; k < numberoftraits; k++){meanall_f[SC_females_age[i]][k] += SC_females[i][k];}
                dam_numberwithinageclass[SC_females_age[i]] += 1;
            }
            for(int i = 0; i < meanall_f.size(); i++)
            {
                for(int j = 0; j < meanall_f[i].size(); j++){meanall_f[i][j] /= double(dam_numberwithinageclass[i]);}
            }
            /* All Animals SD */
            for(int i = 0; i < SC_females.size(); i++)
            {
                for(int k = 0; k < numberoftraits; k++)
                {
                    sdall_f[SC_females_age[i]][k] +=(SC_females[i][k]-meanall_f[SC_females_age[i]][k])*(SC_females[i][k]-meanall_f[SC_females_age[i]][k]);
                }
            }
            for(int i = 0; i < sdall_f.size(); i++)
            {
                for(int j = 0; j < sdall_f[i].size(); j++){sdall_f[i][j] /= double(dam_numberwithinageclass[i]-1); sdall_f[i][j] = sqrt(sdall_f[i][j]);}
            }
            /* Calculate Selection Intensity */
            for(int i = 0; i < sdall_f.size(); i++)
            {
                for(int j = 0; j < sdall_f[i].size(); j++)
                {
                    if(dam_numberwithinageclass_sel[i] > 0){seli_f[i][j] = (meanselected_f[i][j] - meanall_f[i][j]) / double(sdall_f[i][j]);}
                }
            }
            //for(int i = 0; i < meanall_f.size(); i++)
            //{
            //    cout << "Females Age " << i+1 << ":";
            //    for(int j = 0; j < meanselected_f[i].size(); j++){cout << " " << meanselected_f[i][j];}
            //    cout <<"("<<dam_numberwithinageclass_sel[i]<<") -+-";
            //    for(int j = 0; j < meanall_f[i].size(); j++){cout << " " << meanall_f[i][j];}
            //    cout <<"("<<dam_numberwithinageclass[i]<<") -+-";
            //    for(int j = 0; j < sdall_f[i].size(); j++){cout << " " << sdall_f[i][j];}
            //    cout <<" -+-";
            //    for(int j = 0; j < seli_f[i].size(); j++){cout << " " << seli_f[i][j];}
            //   cout << endl;
            //}
            /* now make the weighting to add up intensity of selection */
            for(int i = 0; i < dam_numberwithinageclass_sel.size(); i++){dam_numberwithinageclass_sel[i] /= double(SC_fsel.size());}
            for(int i = 0; i < sire_numberwithinageclass_sel.size(); i++){sire_numberwithinageclass_sel[i] /= double(SC_msel.size());}
            //for(int i = 0; i < dam_numberwithinageclass_sel.size(); i++){cout << dam_numberwithinageclass_sel[i] << " ";}
            //cout << endl;
            //for(int i = 0; i < sire_numberwithinageclass_sel.size(); i++){cout << sire_numberwithinageclass_sel[i]  << " ";}
            //cout << endl;
            /* now generated weighted selection intensity */
            vector < double> weighted_i_sire(numberoftraits,0.0); vector < double> weighted_i_dam(numberoftraits,0.0);
            for(int i = 0; i < seli_m.size(); i++)
            {
                for(int j = 0; j < seli_m[i].size(); j++){weighted_i_sire[j] += (seli_m[i][j] * sire_numberwithinageclass_sel[i]);}
            }
            for(int i = 0; i < seli_f.size(); i++)
            {
                for(int j = 0; j < seli_f[i].size(); j++){weighted_i_dam[j] += (seli_f[i][j] * dam_numberwithinageclass_sel[i]);}
            }
            logfileloc << "       - Selection Intensity: " << endl;
            for(int k = 0; k < numberoftraits; k++)
            {
                logfileloc<<"           - Trait "<<k+1<<": Males = "<<weighted_i_sire[k]<<"; Female = " << weighted_i_dam[k] << endl;
            }
            for(int k = 0; k < numberoftraits; k++)
            {
                Population1.update_Intensity_Males(Gen-1,k,weighted_i_sire[k]);
                Population1.update_Intensity_Females(Gen-1,k,weighted_i_dam[k]);
            }
            if(tempselectionscen == "ebv")
            {
                vector < double > mean(2,0.0); vector < double > sd(2,0.0); double covar = 0.0; int age1num = 0;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1)
                    {
                        mean[0] += (population[i].get_EBVvect())[0]; mean[1] += (population[i].get_BVvect())[0]; age1num += 1;
                    }
                }
                mean[0] /= age1num; mean[1] /= age1num;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1)
                    {
                        sd[0] += ((population[i].get_EBVvect())[0]-mean[0])*((population[i].get_EBVvect())[0]-mean[0]);
                        sd[1] += ((population[i].get_BVvect())[0]-mean[1])*((population[i].get_BVvect())[0]-mean[1]);
                        covar += ((population[i].get_EBVvect())[0]-mean[0])*((population[i].get_BVvect())[0]-mean[1]);
                    }
                }
                sd[0] /= double(age1num-1); sd[1] /= double(age1num-1);
                covar = (covar / double(sqrt(sd[0]) * sqrt(sd[1]))) / double (age1num-1);
                Population1.update_accuracydeltaG(Gen-1,covar);
            }
        }
    }
    /* Start from the end of vector and go back that way saves space */
    int i = (population.size()-1); string kill = "NO";
    /* Save as a continuous string and then output */
    stringstream outputstring(stringstream::out);
    stringstream outputstringgeno(stringstream::out); int outputnum = 0;
    while(kill == "NO")
    {
        while(1)
        {
            if(keep[i] == 2){i--; break;}
            if(keep[i] == 0 && population[i].getSex()==0)
            {
                /* Output info into file with everything in it update with real breeding values at the end */
                outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
                outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
                outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
                outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
                outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
                outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness();
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    outputstring <<" "<<(population[i].get_Phenvect())[k]<<" " <<(population[i].get_EBVvect())[k]<<" "<<(population[i].get_Accvect())[k];
                    outputstring <<" "<<(population[i].get_GVvect())[k]<<" " << (population[i].get_BVvect())[k] <<" ";
                    outputstring << (population[i].get_DDvect())[k] << " " << (population[i].get_Rvect())[k];
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    outputstring << " " << population[i].gettbvindex() << endl;
                } else {outputstring << endl;}
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl;
                }
                population.erase(population.begin()+i); siredamkept.erase(siredamkept.begin()+i);
                keep.erase(keep.begin()+i); siredamvalue.erase(siredamvalue.begin()+i); outputnum++; i--; break;
            }
            if(keep[i] == 1 && population[i].getSex()==0){male++; i--; break;}
            if(keep[i] == 0 && population[i].getSex()==1)
            {
                /* Output info into file with everything in it update with real breeding values at the end */
                outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
                outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
                outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
                outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
                outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
                outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness();
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    outputstring <<" "<<(population[i].get_Phenvect())[k]<<" " <<(population[i].get_EBVvect())[k]<<" "<<(population[i].get_Accvect())[k];
                    outputstring <<" "<<(population[i].get_GVvect())[k]<<" " << (population[i].get_BVvect())[k] <<" ";
                    outputstring << (population[i].get_DDvect())[k] << " " << (population[i].get_Rvect())[k];
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    outputstring << " " << population[i].gettbvindex() << endl;
                } else {outputstring << endl;}
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl;
                }
                population.erase(population.begin()+i); siredamkept.erase(siredamkept.begin()+i);
                keep.erase(keep.begin()+i); siredamvalue.erase(siredamvalue.begin()+i); outputnum++; i--; break;
            }
            if(keep[i] == 1 && population[i].getSex()==1){female++; i--; break;}
        }
        if(outputnum % 150 == 0)
        {
            /* output master df file */
            std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
            output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
            /* output master geno file */
            //std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
            //output4 << outputstringgeno.str();
            gzofstream zippedgeno;
            zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
            if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
            zippedgeno << outputstringgeno.str();
            zippedgeno.close(); outputstringgeno.str(""); outputstringgeno.clear();
        }
        if(i == -1){kill = "YES";}
    }
    /* output master df file */
    std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    //std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
    //output4 << outputstringgeno.str();
    gzofstream zippedgeno;
    zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
    if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
    zippedgeno << outputstringgeno.str();
    zippedgeno.close(); outputstringgeno.str(""); outputstringgeno.clear();
    /* Calculate number of males and females */
    male = 0; female = 0;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getSex() == 0 && population[i].getAge() == 1){male++;}
        if(population[i].getSex() == 1 && population[i].getAge() == 1){female++;}
    }
    logfileloc << "       - Number Males Selected: " << male << "." <<  endl;
    logfileloc << "       - Number Females Selected: " << female << "." << endl;
    logfileloc << "       - Breeding Population Size: " << population.size() << "." << endl;
}
////////////////////////////////////
// Optimal Contribution Selection //
////////////////////////////////////
void optimalcontributionselection(vector <Animal> &population,vector <MatingClass> &matingindividuals,vector < hapLibrary > &haplib,parameters &SimParameters,string tempselectionscen,double* M, float scale,outputfiles &OUTPUTFILES,int Gen,ostream& logfileloc)
{
    system("rm -rf ./evadf || true");
    system("rm -rf ./evarelationship || true");
    system("rm -rf ./evapara.prm || true");
    /* all of the animals will either be in Master_DF_File (culled or unselected) or population */
    /* First read MasterData Frame */
    string line; ifstream infile; vector <string> linestring;
    infile.open(OUTPUTFILES.getloc_Master_DF().c_str());                                                 /* This file has all animals in it */
    if(infile.fail()){cout << "Error Opening Pedigree File\n";}
    while (getline(infile,line)){linestring.push_back(line);}
    /* Used for eva */
    vector <int> tempid((linestring.size()+population.size()),-5);
    vector <int> tempsire((linestring.size()+population.size()),-5);
    vector <int> tempdam((linestring.size()+population.size()),-5);
    vector <int> tempsex((linestring.size()+population.size()),-5);
    vector <int> tempgen((linestring.size()+population.size()),-5);
    vector <int> tempmates((linestring.size()+population.size()),-5);
    vector <double> tempebv((linestring.size()+population.size()),-5);
    /* Used to generate relationship matrix */
    vector < int > ParentIDs;                   /* Used to construct relationship */
    vector < string > genotypeparent;           /* Genotype of parent */
    /* ID of animal corresponds to location in tempvectors first do master df*/
    for(int i = 0; i < linestring.size(); i++)
    {
        //cout << linestring[0] << endl;
        vector <string> lineVar;
        for(int j = 0; j < 27; j++)
        {
            if(j <= 25){size_t pos = linestring[i].find(" ",0); lineVar.push_back(linestring[i].substr(0,pos)); linestring[i].erase(0, pos + 1);}
            if(j == 26){lineVar.push_back(linestring[i]);}
        }
        int row = atoi(lineVar[0].c_str());
        //cout << lineVar[0] << " " << lineVar[1] << " " << lineVar[2] << " " << lineVar[3] << " " << lineVar[4] << endl;
        //cout << tempid[row-1]<<" "<<tempsire[row-1]<<" "<<tempdam[row-1]<<" "<<tempsex[row-1]<<" "<<tempgen[row-1]<<" "<<tempmates[row-1]<<" " <<endl;
        tempid[row-1] = atoi(lineVar[0].c_str()); tempsire[row-1] = atoi(lineVar[1].c_str());
        tempdam[row-1] = atoi(lineVar[2].c_str()); tempsex[row-1] = atoi(lineVar[3].c_str());
        tempgen[row-1] = atoi(lineVar[4].c_str()); tempmates[row-1] = 0; tempebv[row-1] = 0.0;
        //cout << tempid[row-1]<<" "<<tempsire[row-1]<<" "<<tempdam[row-1]<<" "<<tempsex[row-1]<<" "<<tempgen[row-1]<<" "<<tempmates[row-1]<<" " <<endl;
    }
    //for(int i = 0; i < tempid.size(); i++)
    //{
    //    cout<<tempid[i]<<" "<<tempsire[i]<<" "<<tempdam[i]<<" "<<tempsex[i]<<" "<<tempgen[i]<<" "<<tempmates[i]<<" "<<tempebv[i]<<endl;
    //}
    int malecan = 0; int femalecan = 0;
    for(int i = 0; i < population.size(); i++)
    {
        int row = population[i].getID();
        ParentIDs.push_back(population[i].getID());
        tempid[row-1] = population[i].getID(); tempsire[row-1] = population[i].getSire();
        tempdam[row-1] = population[i].getDam(); tempsex[row-1] = population[i].getSex();
        tempgen[row-1] = population[i].getGeneration();
        if(population[i].getSex() == 0){tempmates[row-1] = 10; malecan++;}
        if(population[i].getSex() == 1){tempmates[row-1] = 1; femalecan++;}
        tempebv[row-1] = (population[i].get_EBVvect())[0];
        if(SimParameters.getocsrelat() == "genomic" || SimParameters.getocsrelat() == "genomicmaf"){genotypeparent.push_back(population[i].getMarker());}
    }
    logfileloc << "       - Number Males Candidates: " << malecan << "." <<  endl;
    logfileloc << "       - Number Females Candidates: " << femalecan << "." << endl;
    //for(int i = 0; i < tempid.size(); i++)
    //{
    //    cout<<tempid[i]<<" "<<tempsire[i]<<" "<<tempdam[i]<<" "<<tempsex[i]<<" "<<tempgen[i]<<" "<<tempmates[i]<<" "<<tempebv[i]<<endl;
    //}
    /* Generate DataFrame */
    std::ofstream outputevadf("evadf", std::ios_base::app | std::ios_base::out);
    for(int i = 0; i < tempid.size(); i++)
    {
        outputevadf<<tempid[i]<<"\t"<<tempsire[i]<<"\t"<<tempdam[i]<<"\t"<<tempsex[i]+1<<"\t"<<tempgen[i]<<"\t"<<tempmates[i]<<"\t"<<tempebv[i]<<endl;
    }
    if(SimParameters.getocsrelat() == "pedigree")
    {
        /* Once Generated Now Generate Relationship */
        double* subsetrelationship = new double[ParentIDs.size()*ParentIDs.size()];     /* Used to store subset of relationship matrix */
        pedigree_relationship(OUTPUTFILES,ParentIDs, subsetrelationship);                        /* Generate Relationship Matrix */
        /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
        //for(int i = (ParentIDs.size()-15); i < ParentIDs.size(); i++)
        //{
        //    for(int j = (ParentIDs.size()-15); j < ParentIDs.size(); j++){cout << subsetrelationship[(i*ParentIDs.size())+j] << "\t";}
        //    cout << endl;
        //}
        std::ofstream outputevarel("evarelationship", std::ios_base::app | std::ios_base::out);
        for(int i = 0; i < ParentIDs.size(); i++)
        {
            for(int j = i; j < ParentIDs.size(); j++)
            {
                if(subsetrelationship[(i*ParentIDs.size())+j] != 0.0)
                {
                    outputevarel << ParentIDs[i] << "\t" << ParentIDs[j] << "\t" << subsetrelationship[(i*ParentIDs.size())+j] << endl;
                }
            }
        }
        delete [] subsetrelationship;
    }
    if(SimParameters.getocsrelat() == "genomic")
    {
        double *_grm_mkl = new double[ParentIDs.size()*ParentIDs.size()];
        for(int i = 0; i < (ParentIDs.size()*ParentIDs.size()); i++){_grm_mkl[i] = 0.0;}
        grm_noprevgrm(M,genotypeparent,_grm_mkl,scale);
        /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
        //for(int i = (ParentIDs.size()-15); i < ParentIDs.size(); i++)
        //{
        //    for(int j = (ParentIDs.size()-15); j < ParentIDs.size(); j++){cout << _grm_mkl[(i*ParentIDs.size())+j] << "\t";}
        //    cout << endl;
        //}
        std::ofstream outputevarel("evarelationship", std::ios_base::app | std::ios_base::out);
        for(int i = 0; i < ParentIDs.size(); i++)
        {
            for(int j = i; j < ParentIDs.size(); j++)
            {
                if(_grm_mkl[(i*ParentIDs.size())+j] != 0.0)
                {
                    outputevarel << ParentIDs[i] << "\t" << ParentIDs[j] << "\t" << _grm_mkl[(i*ParentIDs.size())+j] << endl;
                }
            }
        }
        delete [] _grm_mkl;
    }
    if(SimParameters.getocsrelat() == "ROH")
    {
        double *_rohrm = new double[ParentIDs.size()*ParentIDs.size()];
        for(int i = 0; i < (ParentIDs.size()*ParentIDs.size()); i++){_rohrm[i] = 0.0;}
        generaterohmatrix(population,haplib,ParentIDs,_rohrm);
        //for(int i = 0; i < 10; i++)
        //{
        //    for(int j = 0; j < 10; j++){cout << _rohrm[(i*ParentIDs.size())+j] << " ";}
        //    cout << endl;
        //}
        std::ofstream outputevarel("evarelationship", std::ios_base::app | std::ios_base::out);
        for(int i = 0; i < ParentIDs.size(); i++)
        {
            for(int j = i; j < ParentIDs.size(); j++)
            {
                if(_rohrm[(i*ParentIDs.size())+j] != 0.0)
                {
                    outputevarel << ParentIDs[i] << "\t" << ParentIDs[j] << "\t" << _rohrm[(i*ParentIDs.size())+j] << endl;
                }
            }
        }
        delete [] _rohrm;
    }
    if(SimParameters.getocsrelat() == "genomicmaf")
    {
        double *_grm_mkl = new double[ParentIDs.size()*ParentIDs.size()];
        for(int i = 0; i < ParentIDs.size()*ParentIDs.size(); i++){_grm_mkl[i] = 0.0;}
        matinggrm_maf(SimParameters,genotypeparent,_grm_mkl,logfileloc);
        //for(int i = 0; i < 10; i++)
        //{
        //    for(int j = 0; j < 10; j++){cout << _rohrm[(i*ParentIDs.size())+j] << " ";}
        //    cout << endl;
        //}
        std::ofstream outputevarel("evarelationship", std::ios_base::app | std::ios_base::out);
        for(int i = 0; i < ParentIDs.size(); i++)
        {
            for(int j = i; j < ParentIDs.size(); j++)
            {
                if(_grm_mkl[(i*ParentIDs.size())+j] != 0.0)
                {
                    outputevarel << ParentIDs[i] << "\t" << ParentIDs[j] << "\t" << _grm_mkl[(i*ParentIDs.size())+j] << endl;
                }
            }
        }
        delete [] _grm_mkl;
    }   
    std::ofstream outputparameter("evapara.prm", std::ios_base::app | std::ios_base::out);
    outputparameter << "&DATAPARAMETERS" << endl;
    outputparameter << " dataFile='evadf'" << endl;
    outputparameter << " resultsDirectory=''" << endl;
    outputparameter << " ignoreParentalPedigreeErrors=.false. /" << endl << endl;
    outputparameter << "&POPULATIONHISTORY  /" << endl << endl;
    outputparameter << "&RELATIONSHIPMATRIX" << endl;
    outputparameter << " source='file'" << endl;
    outputparameter << " gfile=evarelationship  /" << endl << endl;
    outputparameter << "&OCSPARAMETERS" << endl;
    outputparameter << " nMatings= " << SimParameters.getDams() << endl;
    outputparameter << " optimise='" << SimParameters.getocs_optimize() << "'" << endl;
    
    if(SimParameters.getocs_optimize() == "constraint"){outputparameter << " dFconstraint=" << SimParameters.getocs_w_rel() << endl;}
    if(SimParameters.getocs_optimize() == "penalty")
    {
        outputparameter << " w_merit=   " << SimParameters.getocs_w_merit() << endl;
        outputparameter << " w_relationship=    " << SimParameters.getocs_w_rel() << endl;
    }
    outputparameter << " limitMaleMatings=   1  /" << endl << endl;
    outputparameter << "&ALGORITHMPARAMETERS" << endl;
    outputparameter << " generations=   " << SimParameters.getnEVAgen() << endl;
    outputparameter << " nGenerationsNoImprovement=   30000" << endl;
    outputparameter << " popSize=       " << SimParameters.getnEVApop() << endl;
    outputparameter << " n_offspring=       " << SimParameters.getnEVApop()/double(10) << endl;
    outputparameter << " restart_interval=      " << SimParameters.getnEVAgen()/double(10) << endl;
    outputparameter << " exchange_algorithm=    " << SimParameters.getnEVAgen()/double(10) << endl;
    outputparameter << " mutate_probability=    " << 1 / double(2*double(SimParameters.getDams())) << endl;
    outputparameter << " crossover_probability=    " << 0.75/double(SimParameters.getDams()) << endl;
    outputparameter << " directed_mutation_probability=        " << 1 / double(2*double(SimParameters.getDams())) << endl;
    outputparameter << " seed_rng=0  /" << endl << endl;
    outputparameter << "&MATINGOPTIONS" << endl;
    outputparameter << " matingStrategy='random'" << endl;
    outputparameter << "  repeatedMatings=.true.  /" << endl;
    system("./eva evapara.prm > output.txt 2>&1");
    //stringstream s1; s1 << Gen; string tempvar = s1.str();
    //string outfilename = "./F_summary_" + tempvar + ".txt";
    //string command = "mv ./F_summary.txt " + outfilename;
    //system(command.c_str());
    system("rm -rf ./eva_best.txt eva_conv.txt evadf eva.log evapara.prm evarelationship f_coeff.txt F_summary.txt gencont.txt output.txt || true");
    /* Once finished now read back in mating list file */
    vector < int > siremate; vector < int > dammate; vector < int > selectedanimals;
    line; ifstream infile2; int linenumber = 0;
    infile2.open("eva_MatingList.txt");                                                 /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening Pedigree File\n";}
    while (getline(infile2,line))
    {
        if(linenumber > 0)
        {
            istringstream iss(line);
            string word; int iteration = 0;
            while(iss >> word)
            {
                if(iteration == 0){siremate.push_back(atoi(word.c_str())); selectedanimals.push_back(atoi(word.c_str()));}
                if(iteration == 1){dammate.push_back(atoi(word.c_str())); selectedanimals.push_back(atoi(word.c_str()));}
                iteration++;
            }
            //cout << siremate.size() << " " << dammate.size() << endl;
            //cout << siremate[0] << " " << dammate[0] << endl;
        }
        linenumber++;
    }
    /* now remove animals from population that don't have a */
    sort(selectedanimals.begin(),selectedanimals.end());                                                    /* Remove duplicates */
    selectedanimals.erase(unique(selectedanimals.begin(),selectedanimals.end()),selectedanimals.end());     /* Remove duplicates */
    //cout << selectedanimals.size() << endl;
    //for(int i = 0; i < selectedanimals.size(); i++){cout << selectedanimals[i] << "\t";}
    //cout << endl;
    vector <int> keep(population.size(),0);
    int male = 0;
    int female = 0;
    for(int i = 0; i < selectedanimals.size(); i++)
    {
        int searchlocation = 0;
        while(searchlocation < population.size())
        {
            if(selectedanimals[i] == population[searchlocation].getID())
            {
                if(population[searchlocation].getSex() == 0){male++;}
                if(population[searchlocation].getSex() == 1){female++;}
                keep[searchlocation] = 1; break;
            }
            if(selectedanimals[i] != population[searchlocation].getID()){searchlocation++;}
        }
    }
    //for(int i = 0; i < keep.size(); i++){cout << keep[i] << " ";}
    //cout << endl;
    int ROWS = population.size();                           /* Current Size of Population Class */
    int i = 0;
    /* Save as a continuous string and then output */
    stringstream outputstring(stringstream::out); stringstream outputstringgeno(stringstream::out); int outputnum = 0;
    while(i < ROWS)
    {
        if(keep[i] == 0)
        {
            /* Output info into file with everything in it update with real breeding values at the end */
            outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
            outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
            outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
            outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
            outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
            outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
            outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
            outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
            outputstring << (population[i].get_Phenvect())[0] <<" " << (population[i].get_EBVvect())[0] <<" "<<(population[i].get_Accvect())[0]<<" ";
            outputstring << (population[i].get_GVvect())[0] <<" " << (population[i].get_BVvect())[0] <<" ";
            outputstring << (population[i].get_DDvect())[0] << " " << (population[i].get_Rvect())[0] << endl;
            if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
            {
                outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
            }
            population.erase(population.begin()+i); keep.erase(keep.begin()+i);
            ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
        }
        if(keep[i] == 1){i++;}
        if(outputnum % 1000 == 0)
        {
            /* output master df file */
            std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
            output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
            /* output master geno file */
            std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
            output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
        }
    }
    /* output master df file */
    std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
    /* Calculate number of males and females */
    male = 0; female = 0;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getSex() == 0 && population[i].getAge() == 1){male++;}
        if(population[i].getSex() == 1 && population[i].getAge() == 1){female++;}
    }
    time_t end_test = time (0);
    logfileloc << "       - Number Males Selected: " << male << "." <<  endl;
    logfileloc << "       - Number Females Selected: " << female << "." << endl;
    logfileloc << "       - Breeding Population Size: " << population.size() << "." << endl;
    /* Now begin filling mating design class with population IDs */
    for(int i = 0; i < population.size(); i++)
    {
        //cout << population[i].getID() << " " << population[i].getSex() << " " << population[i].getMatings() << endl;
        /* Now go and find how many times shown up in either siremate or dammate */
        int tempnummatings = 0; int searchlocation = 0;
        while(searchlocation < siremate.size())
        {
            if(population[i].getSex() == 0){if(population[i].getID() == siremate[searchlocation]){tempnummatings += 1;}}
            if(population[i].getSex() == 1){if(population[i].getID() == dammate[searchlocation]){tempnummatings += 1;}}
            searchlocation++;
        }
        //cout << tempnummatings << endl;
        /* Fill Mating Class vector */
        MatingClass temp(population[i].getID(),population[i].getSex(),vector<int>(0),tempnummatings,vector<int>(0),vector<int>(0));
        matingindividuals.push_back(temp);
        //matingindividuals[i].showdata();
    }
    /* Now fill mating class with mating pairs */
    for(int i = 0; i < siremate.size(); i++)
    {
        //cout << i << " " << siremate[i] << " " << dammate[i] << "  ----  ";
        int searchlocation = 0;
        while(1)
        {
            if(siremate[i] == (matingindividuals[searchlocation].getID_MC()))
            {
                //cout << matingindividuals[searchlocation].getID_MC() << " ";
                matingindividuals[searchlocation].add_ToMates(dammate[i]);
                //cout << (matingindividuals[searchlocation].get_mateIDs()).size() << "  ----  ";
                break;
            }
            if(siremate[i] != (matingindividuals[searchlocation].getID_MC())){searchlocation++;}
        }
        searchlocation = 0;
        while(1)
        {
            if(dammate[i] == matingindividuals[searchlocation].getID_MC())
            {
                //cout << matingindividuals[searchlocation].getID_MC() << " ";
                matingindividuals[searchlocation].add_ToMates(siremate[i]);
                //cout << (matingindividuals[searchlocation].get_mateIDs())[0] << endl;
                break;
            }
            if(dammate[i] != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
        }
    }
    /* Now update number of mating in population mating class */
    for(int i = 0; i < population.size(); i++)
    {
        //cout << i << " " << population[i].getID() << " ";
        int searchlocation = 0;
        while(1)
        {
            if(population[i].getID() == matingindividuals[searchlocation].getID_MC())
            {
                //cout << matingindividuals[searchlocation].getID_MC() << " ";
                //cout << matingindividuals[searchlocation].getMatings_MC() << " ";
                population[i].UpdateMatings((matingindividuals[searchlocation].get_mateIDs()).size());
                //cout << population[i].getID() << " " << population[i].getMatings() << endl;
                break;
            }
            if(population[i].getID() != matingindividuals[searchlocation].getID_MC()){searchlocation++;}
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*******                                    Culling Functions                                         *******/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////
// Discrete Generations //
//////////////////////////
void discretegenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc)
{
    /* Output all parents due to non-overlapping generations */
    stringstream outputstring(stringstream::out); stringstream outputstringgeno(stringstream::out);
    int ROWS = population.size();                           /* Current Size of Population Class */
    int i = 0; int outputnum = 0;
    while(i < ROWS)
    {
        while(1)
        {
            if(population[i].getAge() > 1)
            {
                /* Output info into master file with everything in it */
                outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
                outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
                outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
                outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
                outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
                outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness();
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    outputstring <<" "<<(population[i].get_Phenvect())[k]<<" " <<(population[i].get_EBVvect())[k]<<" "<<(population[i].get_Accvect())[k];
                    outputstring <<" "<<(population[i].get_GVvect())[k]<<" " << (population[i].get_BVvect())[k] <<" ";
                    outputstring << (population[i].get_DDvect())[k] << " " << (population[i].get_Rvect())[k];
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    outputstring << " " << population[i].gettbvindex() << endl;
                } else {outputstring << endl;}
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl;
                    outputnum++;
                }
                population.erase(population.begin()+i);     /* Delete Animal from population */
                ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                break;
            }
            if(population[i].getAge() == 1){i++; break;}                                       /* since kept animal move to next slot */
        }
    }
    /* output master df file */
    std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    //std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
    //output4 << outputstringgeno.str();
    gzofstream zippedgeno;
    zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
    if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
    zippedgeno << outputstringgeno.str();
    zippedgeno.close(); outputstringgeno.str(""); outputstringgeno.clear();
    logfileloc << "       -Non-overlapping Generations so all parents culled." << endl;
}
/////////////////////////////
// Overlapping Generations //
/////////////////////////////
void overlappinggenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc)
{
    /* Automatically remove Animals that are at the maximum age then the remaining are used used for proportion culled */
    int oldage = 0;                                         /* Counter for old age */
    int ROWS = population.size();                           /* Current Size of Population Class */
    int i = 0; int outputnum = 0;
    stringstream outputstring(stringstream::out); stringstream outputstringgeno(stringstream::out);
    while(i < ROWS)
    {
        while(1)
        {
            if(population[i].getAge() > SimParameters.getMaxAge())
            {
                /* Output info into master file with everything in it */
                outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
                outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
                outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
                outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
                outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
                outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness();
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    outputstring <<" "<<(population[i].get_Phenvect())[k]<<" " <<(population[i].get_EBVvect())[k]<<" "<<(population[i].get_Accvect())[k];
                    outputstring <<" "<<(population[i].get_GVvect())[k]<<" " << (population[i].get_BVvect())[k] <<" ";
                    outputstring << (population[i].get_DDvect())[k] << " " << (population[i].get_Rvect())[k];
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    outputstring << " " << population[i].gettbvindex() << endl;
                } else {outputstring << endl;}
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                }
                population.erase(population.begin()+i);     /* Delete Animal from population */
                ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                oldage++;
                break;
            }
            if(population[i].getAge() <= SimParameters.getMaxAge()){i++; break;}                   /* since kept animal move to next slot */
        }
    }
    /* output master df file */
    std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    //std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
    //output4 << outputstringgeno.str();
    gzofstream zippedgeno;
    zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
    if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
    zippedgeno << outputstringgeno.str();
    zippedgeno.close(); outputstringgeno.str(""); outputstringgeno.clear();
    logfileloc << "       - Culled " << oldage << " Animals Due To Old Age. (New Population Size: " << population.size() <<")" << endl;
    /* now cull remaining animals based on culling criteria */
    double MaleCutOff = 0.0;                                /* Value cutoff for males */
    double FemaleCutOff = 0.0;                              /* Value cutoff for females */
    int male = 0;                                           /* number of male animals that are of age 1 */
    int female= 0;                                          /* number of female animals that are of age 1 */
    vector < double > MaleValue;                            /* vector to hold male random values that are of age greater than 1 (male is 0)*/
    vector < double > FemaleValue;                          /* veector to hold female random values that are of age greater than 1 (female is 1)*/
    if(SimParameters.getCulling() == "random" || tempcullingscen == "random")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getRndCulling()); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getRndCulling()); female++;}
        }
    }
    if(SimParameters.getCulling() == "phenotype" && tempcullingscen !="random" && tempcullingscen == "phenotype")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back((population[i].get_Phenvect())[0]); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back((population[i].get_Phenvect())[0]); female++;}
        }
    }
    if(SimParameters.getCulling() == "tbv" && tempcullingscen != "random" && tempcullingscen == "tbv")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back((population[i].get_GVvect())[0]); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back((population[i].get_GVvect())[0]);female++;}
        }
    }
    if(SimParameters.getCulling() == "ebv" && tempcullingscen != "random")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back((population[i].get_EBVvect())[0]); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back((population[i].get_EBVvect())[0]); female++;}
        }
    }
    if(SimParameters.getCulling() == "index_tbv" && tempcullingscen != "random")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].gettbvindex()); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].gettbvindex()); female++;}
        }
    }
    if(SimParameters.getCulling() == "index_ebv" && tempcullingscen != "random")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getebvindex()); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getebvindex()); female++;}
        }
    }
    /* Array with correct value based on how selection is to proceed created now sort */
    if(SimParameters.getSelectionDir() == "low" || SimParameters.getCulling() == "random" || tempcullingscen =="random")    /* sort lowest to highest */
    {
        double temp;
        for(int i = 0; i < male -1; i++)                                     /* Sort Males */
        {
            for(int j=i+1; j< male; j++)
            {
                if(MaleValue[i] > MaleValue[j]){temp = MaleValue[i]; MaleValue[i] = MaleValue[j]; MaleValue[j] = temp;}
            }
        }
        for(int i = 0; i < female -1; i++)                                   /* Sort Females */
        {
            for(int j=i+1; j< female; j++)
            {
                if(FemaleValue[i] > FemaleValue[j]){temp = FemaleValue[i]; FemaleValue[i] = FemaleValue[j]; FemaleValue[j] = temp;}
            }
        }
    }
    if(SimParameters.getSelectionDir() == "high" && SimParameters.getCulling() != "random" && tempcullingscen != "random")   /* sort highest to lowest */
    {
        double temp;
        for(int i = 0; i < male -1; i++)                                     /* Sort Males */
        {
            for(int j=i+1; j< male; j++)
            {
                if(MaleValue[i] < MaleValue[j]){temp = MaleValue[i]; MaleValue[i] = MaleValue[j]; MaleValue[j] = temp;}
            }
        }
        for(int i = 0; i < female -1; i++)                                   /* Sort Females */
        {
            for(int j=i+1; j< female; j++)
            {
                if(FemaleValue[i] < FemaleValue[j]){temp = FemaleValue[i]; FemaleValue[i] = FemaleValue[j]; FemaleValue[j] = temp;}
            }
        }
    }
    logfileloc << "       - Number Male Parents prior to culling: " << male << "." <<  endl;
    logfileloc << "       - Number Female Parents prior to culling: " << female << "." << endl;
    int malepos; int femalepos;
    //for(int i = 0; i < FemaleValue.size(); i++){cout << FemaleValue[i] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < MaleValue.size(); i++){cout << MaleValue[i] << " ";}
    //cout << endl;
    /* add 0.5 due to rounding errors accumulating and sometimes will give one less i.e. if put replacement rate as 0.8 */
    if(male > (SimParameters.getSires() * (1-SimParameters.getSireRepl())))
    {
        malepos = SimParameters.getSires()*(1-SimParameters.getSireRepl())+0.5;  /* Grabs Position based on percentile in Males */
    }
    if(male <= (SimParameters.getSires() * (1 - SimParameters.getSireRepl()))){malepos = male;}         /* Keep all animals and keeps more progeny */
    if(female > (SimParameters.getDams() * (1 - SimParameters.getDamRepl())))
    {
        femalepos = SimParameters.getDams() * (1 - SimParameters.getDamRepl()) + 0.5;   /* Grabs Position based on percentile in Females */
    }
    if(female <= (SimParameters.getDams() * (1 - SimParameters.getDamRepl()))){femalepos = female;}     /* Keep all animmals and keeps more progeny */
    MaleCutOff =  MaleValue[malepos - 1];          /* Uniform Value cutoff for males (anything greater then remove) */
    FemaleCutOff = FemaleValue[femalepos - 1];     /* Uniform Value cutoff for females (anything greater then remove) */
    /* Generate Identifier if parent should be removed or not */
    vector <int> keep(population.size(),0); string Action;
    //cout << malepos << " " << femalepos << endl;
    //cout << MaleValue[malepos] << " " << FemaleValue[femalepos] << endl;
    int nummalekept = 0; int numfemalekept = 0;
    for(int i = 0; i < population.size(); i++)
    {
        while(1)
        {
            if(SimParameters.getCulling() == "random" || tempcullingscen == "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getRndCulling()>MaleCutOff){Action = "RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getRndCulling()<=MaleCutOff){Action = "KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getRndCulling()>FemaleCutOff){Action = "RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getRndCulling()<=FemaleCutOff){Action = "KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "phenotype" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random" && tempcullingscen == "phenotype")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]>MaleCutOff){Action = "RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]<=MaleCutOff){Action = "KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]>FemaleCutOff){Action = "RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]<=FemaleCutOff){Action = "KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "phenotype" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random" && tempcullingscen == "phenotype")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]<MaleCutOff){Action = "RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]>=MaleCutOff){Action = "KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]<FemaleCutOff){Action = "RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_Phenvect())[0]>=FemaleCutOff){Action = "KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "tbv" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random" && tempcullingscen == "tbv")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_GVvect())[0]>MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_GVvect())[0]<=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_GVvect())[0]>FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_GVvect())[0]<=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge() == 1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "tbv" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random" && tempcullingscen == "tbv")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_GVvect())[0]<MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_GVvect())[0]>=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_GVvect())[0]<FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_GVvect())[0]>=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "ebv" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]>MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]<=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]>FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]<=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "ebv" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]<MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]>=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]<FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && (population[i].get_EBVvect())[0]>=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "index_tbv" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].gettbvindex()>MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].gettbvindex()<=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].gettbvindex()>FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].gettbvindex()<=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "index_tbv" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].gettbvindex()<MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].gettbvindex()>=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].gettbvindex()<FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].gettbvindex()>=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "index_ebv" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getebvindex()>MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getebvindex()<=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getebvindex()>FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getebvindex()<=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "index_ebv" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getebvindex()<MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getebvindex()>=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getebvindex()<FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getebvindex()>=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
        }
        if(Action == "RM_Male" || Action == "RM_Female"){keep[i] = 0;}
        if(Action == "KP_Male" || Action == "KP_Female")
        {
            if(Action == "KP_Male"){nummalekept++;}
            if(Action == "KP_Female"){numfemalekept++;}
            keep[i] = 1;
        }
        if(Action == "Young_Animal"){keep[i] = 2;}
    }
    //cout << nummalekept << " " << malepos << " - " << numfemalekept << " " << femalepos << endl;
    if(nummalekept != malepos || numfemalekept != femalepos)
    {
        vector <int> maleloc; vector <int> femaleloc;
        for(int i = 0; i < population.size(); i++)
        {
            if(SimParameters.getCulling() == "random"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "phenotype"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "tbv"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "index_tbv"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "index_ebv")
            {
                /* If offspring doesn't have phenotype for both traits and don't update ebv after their progeny are born and before culling */
                /* will have equal breeding values within a litter */
                if(population[i].getSex() == 0 && population[i].getebvindex() == MaleValue[malepos-1]){maleloc.push_back(i);}
                if(population[i].getSex() == 1 && population[i].getebvindex() == FemaleValue[femalepos-1]){femaleloc.push_back(i);}
            }
            if(SimParameters.getCulling() == "ebv")
            {
                if(population[i].getSex() == 0 && (population[i].get_EBVvect())[0] == MaleValue[malepos-1]){maleloc.push_back(i);}
                if(population[i].getSex() == 1 && (population[i].get_EBVvect())[0] == FemaleValue[femalepos-1]){femaleloc.push_back(i);}
            }
        }
        /* If over randomly remove the required number of animals within a family with same EBV */
        for(int i = 0; i < (nummalekept - malepos); i++){keep[maleloc[i]] = 0;}
        for(int i = 0; i < (numfemalekept - femalepos); i++){keep[femaleloc[i]] = 0;}
        int updatedcountmale = 0; int updatedcountfemale = 0;
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){updatedcountmale += keep[i];}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){updatedcountfemale += keep[i];}
        }
    }
    /* Now remove animals */
    i = 0; male = 0; female = 0; ROWS = population.size();                           /* Current Size of Population Class */
    string kill = "NO";                        /* Based on Culling used will result in an action that is common to all */
    while(kill == "NO")
    {
        while(1)
        {
            if(keep[i] == 2){i++; break;}
            if(keep[i] == 0 && population[i].getSex()==0)
            {
                /* Output info into master file with everything in it */
                outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
                outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
                outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
                outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
                outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
                outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness();
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    outputstring <<" "<<(population[i].get_Phenvect())[k]<<" " <<(population[i].get_EBVvect())[k]<<" "<<(population[i].get_Accvect())[k];
                    outputstring <<" "<<(population[i].get_GVvect())[k]<<" " << (population[i].get_BVvect())[k] <<" ";
                    outputstring << (population[i].get_DDvect())[k] << " " << (population[i].get_Rvect())[k];
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    outputstring << " " << population[i].gettbvindex() << endl;
                } else {outputstring << endl;}
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                }
                population.erase(population.begin()+i); keep.erase(keep.begin()+i); break;     /* Delete Animal from population */
            }
            if(keep[i] == 1 && population[i].getSex()==0){male++; i++; break;}
            if(keep[i] == 0 && population[i].getSex()==1)
            {
                /* Output info into master file with everything in it */
                outputstring << population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" ";
                outputstring << population[i].getSex() << " " << population[i].getGeneration() <<" "<< population[i].getAge() <<" ";
                outputstring << population[i].getProgeny() << " " << population[i].getDead() <<" "<< population[i].getPed_F() <<" ";
                outputstring << population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" " <<population[i].getHap2_F() <<" ";
                outputstring << population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring << population[i].getunfavheterolethal() <<" "<<population[i].getunfavhomosublethal() <<" ";
                outputstring << population[i].getunfavheterosublethal() <<" "<<population[i].getlethalequiv() <<" ";
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness();
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    outputstring <<" "<<(population[i].get_Phenvect())[k]<<" " <<(population[i].get_EBVvect())[k]<<" "<<(population[i].get_Accvect())[k];
                    outputstring <<" "<<(population[i].get_GVvect())[k]<<" " << (population[i].get_BVvect())[k] <<" ";
                    outputstring << (population[i].get_DDvect())[k] << " " << (population[i].get_Rvect())[k];
                }
                if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                    outputstring << " " << population[i].gettbvindex() << endl;
                } else {outputstring << endl;}
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                }
                population.erase(population.begin()+i); keep.erase(keep.begin()+i); break;     /* Delete Animal from population */
            }
            if(keep[i] == 1 && population[i].getSex()==1){female++; i++; break;}
        }
        if(i >= population.size()){kill = "YES";}
    }
    /* output master df file */
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    //output4 << outputstringgeno.str();
    zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
    if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
    zippedgeno << outputstringgeno.str();
    zippedgeno.close(); outputstringgeno.str(""); outputstringgeno.clear();
    logfileloc << "       - Number Male parents after culling: " << male << "." <<  endl;
    logfileloc << "       - Number Female parents after culling: " << female << "." << endl;
    logfileloc << "       - Size of population after culling: " << population.size() << endl;
}

/*********************************/
/* Calculate Selection Intensity */
/*********************************/
double SelectionIntensity(double prob, double mu, double sigma)
{
    double p = 1 - prob;
    const long double PI = 3.141592653589793238L;
    /* ---------------------- use AS 241 -------------------------- */
    /* double ppnd16_(double *p, long *ifault)                      */
    /* ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3        */
    /* Produces the normal deviate Z corresponding to a given lower */
    /* tail area of P; Z is accurate to about 1 part in 10**16.     */
    /* ------------------------------------------------------------ */
    double q, r, output;
    if (p < 0 || p > 1){cout << "The probality must be bigger than 0 and smaller than 1." << endl; exit (EXIT_FAILURE);}
    if (sigma <= 0){cout << "The standard deviation sigma must be positive and greather than zero." << endl; exit (EXIT_FAILURE);}
    if (p == 0){cout << "The probability can't be zero (i.e. result is infinity)!!" << endl;}
    if (p == 1){cout << "The probability can't be one (i.e. result is infinity)!!" << endl;}
    q = p - 0.5;
    if(abs(q) <= .425)  /* 0.075 <= p <= 0.925 */
    {
        r = .180625 - q * q;
        output = q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608) / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1);
    } else              /* closer than 0.075 from {0,1} boundary */
    {
        /* r = min(p, 1-p) < 0.075 */
        if (q > 0) {
            r = 1 - p;
        } else {r = p;}
        /* r = sqrt(-log(r))  <==>  min(p, 1-p) = exp( - r^2 ) */
        r = sqrt(-log(r));
        if(r <= 5)         /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
        {
            r += -1.6;
            output = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + .24178072517745061177) * r + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r + 1.42343711074968357734) / (((((((r *1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + .0151986665636164571966) * r + 0.14810397642748007459) * r + .68976733498510000455) * r + 1.6763848301838038494) * r + 2.05319162663775882187) * r + 1);
        }
        else
        { /* very close to  0 or 1 */
            r += -5;
            output = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r + .026532189526576123093) * r + .29656057182850489123) * r + 1.7848265399172913358) * r + 5.4637849111641143699) * r + 6.6579046435011037772) / (((((((r *2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r + .0148753612908506148525) * r + .13692988092273580531) * r + 0.59983220655588793769) * r + 1);
        }
        if (q < 0.0){output = -output;}
    }
    output = mu + sigma * output;
    return(((1 / double(sqrt(2*PI))) * exp(-(output*output)/2))/ double(prob));
}
