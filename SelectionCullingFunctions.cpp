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

#include "HaplofinderClasses.h"
#include "Animal.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"

using namespace std;

/*********************************/
/* Relationship Matrix Functions */
/*********************************/
void pedigree_relationship(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship);
void grm_noprevgrm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);
void generaterohmatrix(vector <Animal> &population,vector < hapLibrary > &haplib,vector <int> const &parentID, double* _rohrm);
void matinggrm_maf(parameters &SimParameters,vector < string > &genotypes,double* output_grm,ostream& logfileloc);

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
void truncationselection(vector <Animal> &population,parameters &SimParameters,string tempselectionscen,int Gen,string Master_DF_File, string Master_Genotype_File, ostream& logfileloc)
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
            if(tempselectionscen == "phenotype"){MaleValue.push_back(population[i].getPhenotype()); male++;}        /* Phenotypic Selection */
            if(tempselectionscen == "true_bv"){MaleValue.push_back(population[i].getGenotypicValue()); male++;}     /* TBV Selection */
            if(tempselectionscen == "ebv"){MaleValue.push_back(population[i].getEBV()); male++;}                    /* EBV Selection */
        }
        if(population[i].getSex() == 1 && population[i].getAge() == 1)                                      /* if female (i.e. 1) */
        {
            if(tempselectionscen == "random"){FemaleValue.push_back(population[i].getRndSelection()); female++;}    /* Random Selection */
            if(tempselectionscen == "phenotype"){FemaleValue.push_back(population[i].getPhenotype()); female++;}    /* Phenotypic Selection */
            if(tempselectionscen == "true_bv"){FemaleValue.push_back(population[i].getGenotypicValue()); female++;} /* TBV Selection */
            if(tempselectionscen == "ebv"){FemaleValue.push_back(population[i].getEBV()); female++;}                /* EBV Selection */
        }
    }
    //cout << maleparent << " " << femaleparent << endl;
    //cout << male << " " << female << endl;
    //cout << MaleValue.size() << " " << FemaleValue.size() << endl;
    //for(int i = 0; i < FemaleValue.size(); i++){cout << FemaleValue[i] << " ";}
    //for(int i = 0; i < MaleValue.size(); i++){cout << MaleValue[i] << " ";}
    //cout << endl;
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
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()<=MaleValue[malepos-1]){Action = "KP_M"; break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}   /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "phenotype" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "true_bv" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "true_bv" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "ebv" && SimParameters.getSelectionDir() == "low")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getEBV()>MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getEBV()<=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getEBV()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getEBV()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
            }
            if(tempselectionscen == "ebv" && SimParameters.getSelectionDir() == "high")
            {
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getEBV()<MaleValue[malepos-1]){Action="RM_M"; break;}
                if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getEBV()>=MaleValue[malepos-1]){Action="KP_M";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getEBV()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getEBV()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
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
        if(tempselectionscen == "phenotype"){siredamvalue[i] = population[i].getPhenotype();}
        if(tempselectionscen == "true_bv"){siredamvalue[i] = population[i].getGenotypicValue();}
        if(tempselectionscen == "ebv"){siredamvalue[i] = population[i].getEBV();}
    }
    /* Sometimes if males and females within a litter have same EBV (i.e. no phenotype) will grab too much */
    if(nummaleselected != malepos || numfemaleselected != femalepos)
    {
        vector <int> maleloc; vector <int> femaleloc;
        for(int i = 0; i < population.size(); i++)
        {
            if(tempselectionscen == "random"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "phenotype"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "true_bv"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(tempselectionscen == "ebv")
            {
                if(population[i].getSex() == 0 && population[i].getEBV() == MaleValue[malepos-1]){maleloc.push_back(i);}
                if(population[i].getSex() == 1 && population[i].getEBV() == FemaleValue[femalepos-1]){femaleloc.push_back(i);}
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
                            if(siredamkept[i] == tempsiredam[fam] && keep[i] == 1){fullsibvalues.push_back(siredamvalue[i]);}                        }
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
                                    if(fullsibvalues[i] == population[startpoint].getPhenotype() && keep[startpoint] == 1){found = "YES";}
                                    if(fullsibvalues[i] != population[startpoint].getPhenotype()){startpoint++;}
                                }
                                if(tempselectionscen == "true_bv")
                                {
                                    if(fullsibvalues[i] == population[startpoint].getGenotypicValue() && keep[startpoint] == 1){found = "YES";}
                                    if(fullsibvalues[i] != population[startpoint].getGenotypicValue()){startpoint++;}
                                }
                                if(tempselectionscen == "ebv")
                                {
                                    if(fullsibvalues[i] != population[startpoint].getEBV()){startpoint++;}
                                    if(fullsibvalues[i] == population[startpoint].getEBV() && keep[startpoint] == 0){startpoint++;}
                                    if(fullsibvalues[i] == population[startpoint].getEBV() && keep[startpoint] == 1){found = "YES";}
                                }
                            }
                            /* No longer select animal */
                            keep[startpoint] = 0;
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
                                        if(MaleValue[malepos-1] == population[startpointa].getPhenotype() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1] != population[startpointa].getPhenotype()){startpointa++;}
                                    }
                                    if(tempselectionscen == "true_bv")
                                    {
                                        if(MaleValue[malepos-1] == population[startpointa].getGenotypicValue() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1] != population[startpointa].getGenotypicValue()){startpointa++;}
                                    }
                                    if(tempselectionscen == "ebv")
                                    {
                                        if(MaleValue[malepos-1]==population[startpointa].getEBV() && keep[startpointa]==0 && population[startpointa].getSex()==0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(MaleValue[malepos-1]==population[startpointa].getEBV() && keep[startpointa]==1 && population[startpointa].getSex()==0)
                                        {
                                            startpointa++;
                                        }
                                        if(MaleValue[malepos-1]==population[startpointa].getEBV() && population[startpointa].getSex()==1){startpointa++;}
                                        if(MaleValue[malepos-1] != population[startpointa].getEBV()){startpointa++;}
                                        
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
                                        if(FemaleValue[femalepos-1] == population[startpointa].getPhenotype() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1] != population[startpointa].getPhenotype()){startpointa++;}
                                    }
                                    if(tempselectionscen == "true_bv")
                                    {
                                        if(FemaleValue[femalepos-1] == population[startpointa].getGenotypicValue() && keep[startpointa] == 0)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1] != population[startpointa].getGenotypicValue()){startpointa++;}
                                    }
                                    if(tempselectionscen == "ebv")
                                    {
                                        if(FemaleValue[femalepos-1]==population[startpointa].getEBV() && keep[startpointa]==0 && population[startpointa].getSex()==1)
                                        {
                                            keep[startpointa] = 1; founda = "YES";
                                        }
                                        if(FemaleValue[femalepos-1]==population[startpointa].getEBV() && keep[startpointa]==1 && population[startpointa].getSex()==1)
                                        {
                                            startpointa++;
                                        }
                                        if(FemaleValue[femalepos-1]==population[startpointa].getEBV() && population[startpointa].getSex()==0){startpointa++;}
                                        if(FemaleValue[femalepos-1] != population[startpointa].getEBV()){startpointa++;}
                                    }
                                }
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
    int ROWS = population.size();                           /* Current Size of Population Class */
    int i = 0; string kill = "NO";
    /* Save as a continuous string and then output */
    stringstream outputstring(stringstream::out); stringstream outputstringgeno(stringstream::out); int outputnum = 0;
    while(kill == "NO")
    {
        while(1)
        {
            if(keep[i] == 2){i++; break;}
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
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
                outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
                outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
                outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                }
                population.erase(population.begin()+i); siredamkept.erase(siredamkept.begin()+i);
                keep.erase(keep.begin()+i); siredamvalue.erase(siredamvalue.begin()+i); break;
            }
            if(keep[i] == 1 && population[i].getSex()==0){male++; i++; break;}
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
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
                outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
                outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
                outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
                if(SimParameters.getOutputGeno() == "yes" && Gen >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                }
                population.erase(population.begin()+i); siredamkept.erase(siredamkept.begin()+i);
                keep.erase(keep.begin()+i); siredamvalue.erase(siredamvalue.begin()+i); break;
            }
            if(keep[i] == 1 && population[i].getSex()==1){female++; i++; break;}
        }
        if(outputnum % 100 == 0)
        {
            /* output master df file */
            std::ofstream output3(Master_DF_File.c_str(), std::ios_base::app | std::ios_base::out);
            output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
            /* output master geno file */
            std::ofstream output4(Master_Genotype_File.c_str(), std::ios_base::app | std::ios_base::out);
            output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
        }
        if(i >= population.size()){kill = "YES";}
    }
    /* output master df file */
    std::ofstream output3(Master_DF_File.c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    std::ofstream output4(Master_Genotype_File.c_str(), std::ios_base::app | std::ios_base::out);
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
}
////////////////////////////////////
// Optimal Contribution Selection //
////////////////////////////////////
void optimalcontributionselection(vector <Animal> &population,vector <MatingClass> &matingindividuals,vector < hapLibrary > &haplib,parameters &SimParameters,string tempselectionscen,string Pheno_Pedigree_File,double* M, float scale,string Master_DF_File,string Master_Genotype_File,int Gen,ostream& logfileloc)
{
    system("rm -rf ./evadf || true");
    system("rm -rf ./evarelationship || true");
    system("rm -rf ./evapara.prm || true");
    /* all of the animals will either be in Master_DF_File (culled or unselected) or population */
    /* First read MasterData Frame */
    string line; ifstream infile; vector <string> linestring;
    infile.open(Master_DF_File.c_str());                                                 /* This file has all animals in it */
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
        tempebv[row-1] = population[i].getEBV();
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
        pedigree_relationship(Pheno_Pedigree_File,ParentIDs, subsetrelationship);                        /* Generate Relationship Matrix */
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
            outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
            outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
            outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
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
            std::ofstream output3(Master_DF_File.c_str(), std::ios_base::app | std::ios_base::out);
            output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
            /* output master geno file */
            std::ofstream output4(Master_Genotype_File.c_str(), std::ios_base::app | std::ios_base::out);
            output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
        }
    }
    /* output master df file */
    std::ofstream output3(Master_DF_File.c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    std::ofstream output4(Master_Genotype_File.c_str(), std::ios_base::app | std::ios_base::out);
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
void discretegenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,string Master_DF_File, string Master_Genotype_File, ostream& logfileloc)
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
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
                outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
                outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
                outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
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
    std::ofstream output3(Master_DF_File.c_str(), std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    std::ofstream output4(Master_Genotype_File.c_str(), std::ios_base::app | std::ios_base::out);
    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
    logfileloc << "       -Non-overlapping Generations so all parents culled." << endl;
}
/////////////////////////////
// Overlapping Generations //
/////////////////////////////
void overlappinggenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,string Master_DF_File,string Master_Genotype_File,ostream& logfileloc)
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
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
                outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
                outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
                outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
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
    std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
    /* output master geno file */
    std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
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
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getPhenotype()); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getPhenotype()); female++;}
        }
    }
    if(SimParameters.getCulling() == "true_bv" && tempcullingscen != "random" && tempcullingscen == "true_bv")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getGenotypicValue()); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getGenotypicValue());female++;}
        }
    }
    if(SimParameters.getCulling() == "ebv" && tempcullingscen != "random")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getEBV()); male++;}
            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getEBV()); female++;}
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
    int malepos;
    int femalepos;
    //for(int i = 0; i < FemaleValue.size(); i++){cout << FemaleValue[i] << " ";}
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
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()>MaleCutOff){Action = "RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()<=MaleCutOff){Action = "KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()>FemaleCutOff){Action = "RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()<=FemaleCutOff){Action = "KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "phenotype" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random" && tempcullingscen == "phenotype")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()<MaleCutOff){Action = "RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()>=MaleCutOff){Action = "KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()<FemaleCutOff){Action = "RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()>=FemaleCutOff){Action = "KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "true_bv" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random" && tempcullingscen == "true_bv")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()>MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()<=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()>FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()<=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge() == 1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "true_bv" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random" && tempcullingscen == "true_bv")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()<MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()>=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()<FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()>=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "ebv" && SimParameters.getSelectionDir() == "low" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()>MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()<=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()>FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()<=FemaleCutOff){Action="KP_Female";break;}
                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
            }
            if(SimParameters.getCulling() == "ebv" && SimParameters.getSelectionDir() == "high" && tempcullingscen != "random")
            {
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()<MaleCutOff){Action="RM_Male";break;}
                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()>=MaleCutOff){Action="KP_Male";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()<FemaleCutOff){Action="RM_Female";break;}
                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()>=FemaleCutOff){Action="KP_Female";break;}
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
    if(nummalekept != malepos || numfemalekept != femalepos)
    {
        vector <int> maleloc; vector <int> femaleloc;
        for(int i = 0; i < population.size(); i++)
        {
            if(SimParameters.getCulling() == "random"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "phenotype"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "true_bv"){cout << "Shouldn't Be Here: E-mail Software Developer!!" << endl;}
            if(SimParameters.getCulling() == "ebv")
            {
                if(population[i].getSex() == 0 && population[i].getEBV() == MaleValue[malepos-1]){maleloc.push_back(i);}
                if(population[i].getSex() == 1 && population[i].getEBV() == FemaleValue[femalepos-1]){femaleloc.push_back(i);}
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
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
                outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
                outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
                outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
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
                outputstring << population[i].getHomozy() <<" "<< population[i].getpropROH() <<" "<< population[i].getFitness() <<" ";
                outputstring << population[i].getPhenotype() <<" " << population[i].getEBV() <<" "<< population[i].getAcc() <<" ";
                outputstring << population[i].getGenotypicValue()<<" " << population[i].getBreedingValue() <<" ";
                outputstring << population[i].getDominanceDeviation() << " " << population[i].getResidual() << endl;
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
    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
    logfileloc << "       - Number Male parents after culling: " << male << "." <<  endl;
    logfileloc << "       - Number Female parents after culling: " << female << "." << endl;
    logfileloc << "       - Size of population after culling: " << population.size() << endl;
}
