#include <iostream>
#include <string>
#include <algorithm>
#include <math.h>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <omp.h>
#include <mkl.h>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>

#include "HaplofinderClasses.h"
#include "Animal.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"
#include "OutputFiles.h"



using namespace std;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////     Pedigree Inverse      /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Generate inverse of relationship matrix: pass animal sire and dam as a reference; output is ainverse */
void pedigree_inverse(vector <int> const &f_anim, vector <int> const &f_sire, vector <int> const &f_dam, vector<double> &output,vector < double > &output_f)
{
    int TotalAnimalNumber = f_anim.size();
    vector < double > F((TotalAnimalNumber+1),0.0);
    vector < double > D(TotalAnimalNumber,0.0);
    /* This it makes so D calculate is correct */
    F[0] = -1;
    for(int k = f_anim[0]; k < (f_anim.size()+1); k++)                  /* iterate through each row of l */
    {
        vector < double > L(TotalAnimalNumber,0.0);
        vector < double > AN;                                       // Holds all ancestors of individuals
        double ai = 0.0;                                            /* ai  is the inbreeding */
        AN.push_back(k);                                            /* push back animal in ancestor */
        L[k-1] = 1.0;
        /* Calculate D */
        D[k-1] = 0.5 - (0.25 * (F[f_sire[k-1]] + F[f_dam[k-1]]));
        int j = k;                                          /* start off at K then go down */
        while(AN.size() > 0)
        {
            /* if sire is know add to AN and add to L[j] */
            if((f_sire[j-1]) != 0){AN.push_back(f_sire[j-1]); L[f_sire[j-1]-1] += 0.5 * L[j-1];}
            /* if dam is known add to AN and add to L[j] */
            if((f_dam[j-1]) != 0){AN.push_back(f_dam[j-1]); L[f_dam[j-1]-1] += 0.5 * L[j-1];}
            /* add to inbreeding value */
            ai += (L[j-1] * L[j-1] * D[j-1]);
            /* Delete j from n */
            int found = 0;
            while(1)
            {
                if(AN[found] == j){AN.erase(AN.begin()+found); break;}
                if(AN[found] != j){found++;}
            }
            /* Find youngest animal in AN to see if it has ancestors*/
            j = -1;
            for(int i = 0; i < AN.size(); i++){if(AN[i] > j){j = AN[i];}}
            /* Erase Duplicates */
            sort(AN.begin(),AN.end());
            AN.erase(unique(AN.begin(),AN.end()),AN.end());
        }
        /* calculate inbreeding value */
        F[k] = ai - 1;
        double bi = (1/sqrt(D[k-1])) * (1/sqrt(D[k-1]));
        if(f_sire[k-1] != 0 && f_dam[k-1] != 0) /* indexed by (row * f_anim.size()) + col */
        {
            output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] + bi;
            output[((f_sire[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] = output[((f_sire[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] - (bi/2);
            output[((f_anim[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] - (bi/2);
            output[((f_dam[k-1]-1) *f_anim.size())+(f_anim[k-1]-1)] = output[((f_dam[k-1]-1) *f_anim.size())+(f_anim[k-1]-1)] - (bi/2);
            output[((f_anim[k-1]-1)*f_anim.size())+ (f_dam[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+ (f_dam[k-1]-1)] - (bi/2);
            output[((f_sire[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] = output[((f_sire[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] + (bi/4);
            output[((f_sire[k-1]-1)*f_anim.size())+ (f_dam[k-1]-1)] = output[((f_sire[k-1]-1)*f_anim.size())+ (f_dam[k-1]-1)] + (bi/4);
            output[((f_dam[k-1]-1) *f_anim.size())+(f_sire[k-1]-1)] = output[((f_dam[k-1]-1) *f_anim.size())+(f_sire[k-1]-1)] + (bi/4);
            output[((f_dam[k-1]-1) *f_anim.size())+ (f_dam[k-1]-1)] = output[((f_dam[k-1]-1) *f_anim.size())+ (f_dam[k-1]-1)] + (bi/4);
        }
        if(f_sire[k-1] != 0 && f_dam[k-1] == 0)
        {
            output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] + bi;
            output[((f_sire[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] = output[((f_sire[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] - (bi/2);
            output[((f_anim[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] - (bi/2);
            output[((f_sire[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] = output[((f_sire[k-1]-1)*f_anim.size())+(f_sire[k-1]-1)] + (bi/4);
        }
        if(f_sire[k-1] == 0 && f_dam[k-1] != 0)
        {
            output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] + bi;
            output[((f_dam[k-1]-1) *f_anim.size())+(f_anim[k-1]-1)] = output[((f_dam[k-1]-1) *f_anim.size())+(f_anim[k-1]-1)] - (bi/2);
            output[((f_anim[k-1]-1)*f_anim.size())+ (f_dam[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+ (f_dam[k-1]-1)] - (bi/2);
            output[((f_dam[k-1]-1) *f_anim.size())+ (f_dam[k-1]-1)] = output[((f_dam[k-1]-1) *f_anim.size())+ (f_dam[k-1]-1)] + (bi/4);
        }
        if(f_sire[k-1] == 0 && f_dam[k-1] == 0)
        {
            output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] = output[((f_anim[k-1]-1)*f_anim.size())+(f_anim[k-1]-1)] + bi;
        }
    }
    /* copy vector back to output_ped array */
    for(int i = 1; i < (f_anim.size()+1); i++){output_f[i-1] = F[i];}
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////     Calculate Inbreeding       ///////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Calculates inbreeding: pass animal sire and dam as a reference; output is output_f based on Meuwissen & Luo Algorithm*/
void pedigree_inbreeding(string phenotypefile, double* output_f)
{
    vector < int > animal;
    vector < int > sire;
    vector < int > dam;
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    string line;
    ifstream infile2;
    infile2.open(phenotypefile.c_str());                                                 /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening Pedigree File\n";}
    while (getline(infile2,line))
    {
        /* Fill each array with correct number already in order so don't need to order */
        size_t pos = line.find(" ",0); animal.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);  /* Grab Animal ID */
        pos = line.find(" ",0); sire.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);           /* Grab Sire ID */
        pos = line.find(" ",0); dam.push_back(stoi(line.substr(0,pos)));                                    /* Grab Dam ID */
    }
    int TotalAnimalNumber = animal.size();
    vector < double > F((TotalAnimalNumber+1),0.0);
    vector < double > D(TotalAnimalNumber,0.0);
    /* This it makes so D calculate is correct */
    F[0] = -1;
    for(int k = animal[0]; k < (animal.size()+1); k++)                  /* iterate through each row of l */
    {
        vector < double > L(TotalAnimalNumber,0.0);
        vector < double > AN;                                       // Holds all ancestors of individuals
        double ai = 0.0;                                            /* ai  is the inbreeding */
        AN.push_back(k);                                            /* push back animal in ancestor */
        L[k-1] = 1.0;
        /* Calculate D */
        D[k-1] = 0.5 - (0.25 * (F[sire[k-1]] + F[dam[k-1]]));
        int j = k;                                          /* start off at K then go down */
        while(AN.size() > 0)
        {
            /* if sire is know add to AN and add to L[j] */
            if((sire[j-1]) != 0){AN.push_back(sire[j-1]); L[sire[j-1]-1] += 0.5 * L[j-1];}
            /* if dam is known add to AN and add to L[j] */
            if((dam[j-1]) != 0){AN.push_back(dam[j-1]); L[dam[j-1]-1] += 0.5 * L[j-1];}
            /* add to inbreeding value */
            ai += (L[j-1] * L[j-1] * D[j-1]);
            /* Delete j from n */
            int found = 0;
            while(1)
            {
                if(AN[found] == j){AN.erase(AN.begin()+found); break;}
                if(AN[found] != j){found++;}
            }
            /* Find youngest animal in AN to see if it has ancestors*/
            j = -1;
            for(int i = 0; i < AN.size(); i++){if(AN[i] > j){j = AN[i];}}
            /* Erase Duplicates */
            sort(AN.begin(),AN.end());
            AN.erase(unique(AN.begin(),AN.end()),AN.end());
        }
        /* calculate inbreeding value */
        F[k] = ai - 1;
    }
    /* copy vector back to output_qs_u array */
    for(int i = 1; i < (animal.size()+1); i++){output_f[i-1] = F[i];}
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////     Calculate Inbreeding Lethals       ///////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/* Calculates inbreeding: pass animal sire and dam as a reference; output is output_f based on Meuwissen & Luo Algorithm*/
double lethal_pedigree_inbreeding(outputfiles &OUTPUTFILES, int tempsireid, int tempdamid)
{
    vector < int > animal;
    vector < int > sire;
    vector < int > dam;
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    string line;
    ifstream infile2;
    infile2.open(OUTPUTFILES.getloc_Pheno_Pedigree().c_str());                                                 /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening Pedigree File\n";}
    while (getline(infile2,line))
    {
        /* Fill each array with correct number already in order so don't need to order */
        size_t pos = line.find(" ",0); animal.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);  /* Grab Animal ID */
        pos = line.find(" ",0); sire.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);           /* Grab Sire ID */
        pos = line.find(" ",0); dam.push_back(stoi(line.substr(0,pos)));                                    /* Grab Dam ID */
    }
    /* add dead animal */
    animal.push_back(animal[animal.size()-1]+1); sire.push_back(tempsireid); dam.push_back(tempdamid);
    int TotalAnimalNumber = animal.size();
    vector < double > F((TotalAnimalNumber+1),0.0);
    vector < double > D(TotalAnimalNumber,0.0);
    /* This it makes so D calculate is correct */
    F[0] = -1;
    for(int k = animal[0]; k < (animal.size()+1); k++)                  /* iterate through each row of l */
    {
        vector < double > L(TotalAnimalNumber,0.0);
        vector < double > AN;                                       // Holds all ancestors of individuals
        double ai = 0.0;                                            /* ai  is the inbreeding */
        AN.push_back(k);                                            /* push back animal in ancestor */
        L[k-1] = 1.0;
        /* Calculate D */
        D[k-1] = 0.5 - (0.25 * (F[sire[k-1]] + F[dam[k-1]]));
        int j = k;                                          /* start off at K then go down */
        while(AN.size() > 0)
        {
            /* if sire is know add to AN and add to L[j] */
            if((sire[j-1]) != 0){AN.push_back(sire[j-1]); L[sire[j-1]-1] += 0.5 * L[j-1];}
            /* if dam is known add to AN and add to L[j] */
            if((dam[j-1]) != 0){AN.push_back(dam[j-1]); L[dam[j-1]-1] += 0.5 * L[j-1];}
            /* add to inbreeding value */
            ai += (L[j-1] * L[j-1] * D[j-1]);
            /* Delete j from n */
            int found = 0;
            while(1)
            {
                if(AN[found] == j){AN.erase(AN.begin()+found); break;}
                if(AN[found] != j){found++;}
            }
            /* Find youngest animal in AN to see if it has ancestors*/
            j = -1;
            for(int i = 0; i < AN.size(); i++){if(AN[i] > j){j = AN[i];}}
            /* Erase Duplicates */
            sort(AN.begin(),AN.end());
            AN.erase(unique(AN.begin(),AN.end()),AN.end());
        }
        /* calculate inbreeding value */
        F[k] = ai - 1;
    }
    return(F[TotalAnimalNumber]);
}
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////     Pedigree Relationship Matrix (old)      //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/* Calculates subset of a relationship matrix */
void pedigree_relationship(outputfiles &OUTPUTFILES, vector <int> const &parent_id, double* output_subrelationship)
{
    vector < int > animal; vector < int > sire; vector < int > dam;
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    string line;
    ifstream infile2;
    infile2.open(OUTPUTFILES.getloc_Pheno_Pedigree().c_str());                  /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening Pedigree File\n";}
    while (getline(infile2,line))
    {
        /* Fill each array with correct number already in order so don't need to order */
        size_t pos = line.find(" ",0); animal.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);  /* Grab Animal ID */
        pos = line.find(" ",0); sire.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);           /* Grab Sire ID */
        pos = line.find(" ",0); dam.push_back(stoi(line.substr(0,pos)));                                    /* Grab Dam ID */
    }
    int TotalAnimalNumber = animal.size();
    double* parent_A = new double[TotalAnimalNumber * TotalAnimalNumber];            /* Full A matrix */
    unsigned long i_p, j_p;
    #pragma omp parallel for private(j_p)
    for(i_p = 0; i_p < animal.size(); i_p++)
    {
        for(j_p = 0; j_p < animal.size(); j_p++){parent_A[(i_p*TotalAnimalNumber)+j_p] = 0.0;}
    }
    /* Generate relationship of whole group */
    for(int i = 0; i < animal.size(); i++)
    {
        if (sire[i] != 0 && dam[i] != 0)
        {
            for (int j = 0; j < i; j++)
            {
                parent_A[(i*TotalAnimalNumber)+j] = 0.5 * (parent_A[(j*TotalAnimalNumber)+(sire[i]-1)] + parent_A[(j*TotalAnimalNumber)+(dam[i]-1)]);
                parent_A[(j*TotalAnimalNumber)+i] = 0.5 * (parent_A[(j*TotalAnimalNumber)+(sire[i]-1)] + parent_A[(j*TotalAnimalNumber)+(dam[i]-1)]);
            }
            parent_A[((animal[i]-1)*TotalAnimalNumber)+(animal[i]-1)] = 1 + (0.5 * parent_A[((sire[i]-1)*TotalAnimalNumber)+(dam[i]-1)]);
        }
        if (sire[i] != 0 && dam[i] == 0)
        {
            for (int j = 0; j < i; j++)
            {
                parent_A[(j*TotalAnimalNumber)+i] = 0.5 * (parent_A[(j*TotalAnimalNumber)+(sire[i]-1)]);
                parent_A[(i*TotalAnimalNumber)+j] = 0.5 * (parent_A[(j*TotalAnimalNumber)+(sire[i]-1)]);
            }
            parent_A[((animal[i]-1)*TotalAnimalNumber)+(animal[i]-1)] = 1;
        }
        if (sire[i] == 0 && dam[i] != 0)
        {
            for (int j = 0; j < i; j++)
            {
                parent_A[(j*TotalAnimalNumber)+i] = 0.5 * (parent_A[(j*TotalAnimalNumber)+(dam[i]-1)]);
                parent_A[(i*TotalAnimalNumber)+j] = 0.5 * (parent_A[(j*TotalAnimalNumber)+(dam[i]-1)]);
            }
            parent_A[((animal[i]-1)*TotalAnimalNumber)+(animal[i]-1)] = 1;
        }
        if (sire[i] == 0 && dam[i] == 0){parent_A[((animal[i]-1)*TotalAnimalNumber)+(animal[i]-1)] = 1;}
    }
    /* Full Created now fill subset */
    for(int i = 0; i < parent_id.size(); i++)
    {
        for(int j = i; j < parent_id.size(); j++)
        {
            if(i == j){output_subrelationship[(i*parent_id.size()) + j] = parent_A[((parent_id[i]-1)*TotalAnimalNumber) + (parent_id[j]-1)];}
            if(i != j)
            {
                output_subrelationship[(i*parent_id.size())+ j] = parent_A[((parent_id[i]-1)*TotalAnimalNumber) + (parent_id[j]-1)];
                output_subrelationship[(j*parent_id.size())+ i] = parent_A[((parent_id[i]-1)*TotalAnimalNumber) + (parent_id[j]-1)];
            }
        }
    }
    delete [] parent_A;
}
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////     Pedigree Relationship Matrix (new)      //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
void pedigree_relationship_Colleau(outputfiles &OUTPUTFILES, vector <int> const &parent_id, double* output_subrelationship)
{
    vector < int > animal; vector < int > sire; vector < int > dam; vector < double > f;
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    string line; ifstream infile2;
    infile2.open(OUTPUTFILES.getloc_Pheno_Pedigree().c_str());       /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening Pedigree File\n";}
    while (getline(infile2,line))
    {
        /* Fill each array with correct number already in order so don't need to order */
        size_t pos = line.find(" ",0); animal.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);  /* Grab Animal ID */
        pos = line.find(" ",0); sire.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);           /* Grab Sire ID */
        pos = line.find(" ",0); dam.push_back(stoi(line.substr(0,pos))); f.push_back(0.0);                  /* Grab Dam ID */
    }
    int TotalAnimalNumber = animal.size();
    vector < double > F((TotalAnimalNumber+1),0.0);
    vector < double > D(TotalAnimalNumber,0.0);
    /* This it makes so D calculate is correct */
    F[0] = -1;
    for(int k = animal[0]; k < (animal.size()+1); k++)                  /* iterate through each row of l */
    {
        vector < double > L(TotalAnimalNumber,0.0);
        vector < double > AN;                                       // Holds all ancestors of individuals
        double ai = 0.0;                                            /* ai  is the inbreeding */
        AN.push_back(k);                                            /* push back animal in ancestor */
        L[k-1] = 1.0;
        /* Calculate D */
        D[k-1] = 0.5 - (0.25 * (F[sire[k-1]] + F[dam[k-1]]));
        int j = k;                                          /* start off at K then go down */
        while(AN.size() > 0)
        {
            /* if sire is know add to AN and add to L[j] */
            if((sire[j-1]) != 0){AN.push_back(sire[j-1]); L[sire[j-1]-1] += 0.5 * L[j-1];}
            /* if dam is known add to AN and add to L[j] */
            if((dam[j-1]) != 0){AN.push_back(dam[j-1]); L[dam[j-1]-1] += 0.5 * L[j-1];}
            /* add to inbreeding value */
            ai += (L[j-1] * L[j-1] * D[j-1]);
            /* Delete j from n */
            int found = 0;
            while(1)
            {
                if(AN[found] == j){AN.erase(AN.begin()+found); break;}
                if(AN[found] != j){found++;}
            }
            /* Find youngest animal in AN to see if it has ancestors*/
            j = -1;
            for(int i = 0; i < AN.size(); i++){if(AN[i] > j){j = AN[i];}}
            /* Erase Duplicates */
            sort(AN.begin(),AN.end());
            AN.erase(unique(AN.begin(),AN.end()),AN.end());
        }
        /* calculate inbreeding value */
        F[k] = ai - 1;
    }
    /* copy vector back to output_qs_u array */
    for(int i = 1; i < (animal.size()+1); i++){f[i-1] = F[i];}
    int nGen = parent_id.size(); int n = f.size();
    vector < double > w(n,0.0); vector < double > v(n,0.0);
    /* temporary variables */
    int s, d;
    double di, initialadd, tmp;
    for(int i = 0; i < nGen; i++)
    {
        //cout << parent_id[i] << endl << endl;
        vector < double > q(n,0.0);
        for(int j = 0; j < n; j++){v[j] = 0;}
        v[parent_id[i]-1] =  1.0;
        //cout << parent_id[i] << endl << endl;
        //for(int j = 0; j < n; j++){cout << v[j] << " ";}
        //cout << endl << endl;
        for(int j = (n-1); j >= 0 && j < n; --j)
        {
            q[j] = q[j] + v[j];
            int s = sire[j]; int d = dam[j];
            //for(int j = 0; j < n; j++){cout << q[j] << " ";}
            //cout << " -- " << s << " " << d;
            if(s != 0){q[s-1] = q[s-1] + q[j] * 0.5;}
            if(d != 0){q[d-1] = q[d-1] + q[j] * 0.5;}
            //cout << endl << endl;
        }
        //for(int j = 0; j < n; j++){cout << q[j] << " ";}
        //cout << endl << endl;
        for(int j = 0; j < n; j++)
        {
            s = sire[j]; d = dam[j]; di = 0;
            if(s != 0 && d != 0)
            {
                initialadd = 0;
                if(sire[j] == 0){initialadd += 1;}
                if(dam[j] == 0){initialadd += 1;}
                di = ((initialadd+2)/double(4)) - (0.25*(f[s-1]+f[d-1]));
            }
            if(s == 0 && d == 0)
            {
                initialadd = 0;
                if(sire[j] == 0){initialadd += 1;}
                if(dam[j] == 0){initialadd += 1;}
                di = ((initialadd+2)/double(4));
            }
            if(s != 0 && d == 0)
            {
                initialadd = 0;
                if(sire[j] == 0){initialadd += 1;}
                if(dam[j] == 0){initialadd += 1;}
                di = ((initialadd+2)/double(4)) - (0.25*(f[s-1]+0));
            }
            if(s == 0 && d != 0)
            {
                initialadd = 0;
                if(sire[j] == 0){initialadd += 1;}
                if(dam[j] == 0){initialadd += 1;}
                di = ((initialadd+2)/double(4)) - (0.25*(0+f[d-1]));
            }
            tmp = 0.0;
            if(s != 0){tmp = tmp + w[s-1];}
            if(d != 0){tmp = tmp + w[d-1];}
            w[j] = 0.5 *tmp;
            w[j] = w[j] + (di * q[j]);
        }
        //for(int j = 0; j < n; j++){cout << w[j] << " ";}
        //cout << endl << endl;
        for(int j = 0; j < parent_id.size(); j++){output_subrelationship[(i*parent_id.size())+j] = w[parent_id[j]-1];}
    }
}
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////     Genomic Relationship Matrix (MAF)      //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
void matinggrm_maf(parameters &SimParameters,vector < string > &genotypes,double* output_grm,ostream& logfileloc)
{
    vector < double > obsfreq(genotypes[0].size(),0.0);
    vector < double > obsmaf(genotypes[0].size(),0.0);
    vector < int > included(genotypes[0].size(),0);
    /* First calculate frequency to get MAF */
    for(int i = 0; i < genotypes.size(); i++)
    {
        string geno = genotypes[i];
        for(int j = 0; j < geno.size(); j++)
        {
            int temp = geno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            obsfreq[j] += temp;
        }
    }
    int numberutilized = 0;
    for(int i = 0; i < obsfreq.size(); i++)
    {
        obsfreq[i] = obsfreq[i] / (double(2*genotypes.size()));
        if(obsfreq[i] > 0.5){obsmaf[i] = 1- obsfreq[i];}
        if(obsfreq[i] < 0.5){obsmaf[i] = obsfreq[i];}
        if(SimParameters.getgenmafdir() == "above")
        {
            if(obsmaf[i] >= SimParameters.getgenmafcutoff()){included[i] = 1; numberutilized += 1;}
        }
        if(SimParameters.getgenmafdir() == "below")
        {
            if(obsmaf[i] <= SimParameters.getgenmafcutoff()){included[i] = 1; numberutilized += 1;}
        }
    }
    /* Set up M and scale factor */
    double* M = new double[3*obsfreq.size()]; float scale = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < obsfreq.size(); j++)
        {
            if(included[j] == 1)
            {
                if(i == 0){scale += (1 - obsfreq[j]) * obsfreq[j];}
                M[(i*obsfreq.size())+j] = i - (2 * obsfreq[j]);
            }
            if(included[j] == 0){M[(i*obsfreq.size())+j] = -5;}
        }
    }
    scale = scale * 2;
    /* Set parameters of matrix dimension */
    int n = genotypes.size();                       /* Number of animals */
    int m = numberutilized;                         /* Number of markers */
    int j = 0; int i = 0;
    double *_geno_mkl = new double[n*m];            /* Allocate Memory for Z matrix n x m */
    /* Fill Matrix */
    for(i = 0; i < n; i++)
    {
        int colloc = 0;
        string tempgeno = genotypes[i];
        for(j = 0; j < obsfreq.size(); j++)
        {
            if(included[j] == 1)
            {
                int tempa = tempgeno[j] - 48;
                if(tempa == 3 || tempa == 4){tempa = 1;}
                _geno_mkl[(i*m)+colloc] = M[(tempa*obsfreq.size())+j];
                colloc++;
                if(M[(tempa*obsfreq.size())+j] == -5){cout << M[(tempa*obsfreq.size())+j] << endl;}
            }
        }
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _geno_mkl[(i*m)+j] << " ";}
    //    cout << endl;
    //}
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, m, 1.0, _geno_mkl, m, _geno_mkl, m, 0.0, output_grm, n);
    delete[] _geno_mkl; delete [] M;
    // Standardize relationship matrix by scalar
    #pragma omp parallel for private(j)
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++){output_grm[(i*n)+j] /= scale;}
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << output_grm[(i*n)+j] << " ";}
    //    cout << endl;
    //}
    logfileloc << "             - Number of Markers Utilized: " << numberutilized << endl;
    //cout << endl << endl << numberutilized << endl << endl;
    //for(int i = 0; i < 25; i++)
    //{
    //    cout << obsfreq[i] << " " << obsmaf[i] << " " << included[i] << " --- ";
    //    cout << M[(0*obsfreq.size())+i] << " " << M[(1*obsfreq.size())+i] << " " << M[(2*obsfreq.size())+i] << endl;
    //}
    //cout << scale << endl;
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _geno_mkl[(i*m)+j] << " ";}
    //    cout << endl;
    //}

}













////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 4       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Calculate frequency of a SNPs;  */
void frequency_calc(vector < string > const &genotypes, double* output_freq)
{
    for(int i = 0; i < genotypes[0].size(); i++){output_freq[i] = 0;}
    for(int i = 0; i < genotypes.size(); i++)
    {
        string geno = genotypes[i];
        for(int j = 0; j < geno.size(); j++)
        {
            int temp = geno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            output_freq[j] += temp;
        }
    }
    for(int i = 0; i < genotypes[0].size(); i++){output_freq[i] = output_freq[i] / (2 * genotypes.size());}
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 5       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Generate Genomic Relationship Matrix (GRM) without any prior GRM matrix calculation  */
void grm_noprevgrm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler)
{
    /* Set parameters of matrix dimension */
    int n = genotypes.size();                       /* Number of animals */
    int m = genotypes[0].size();                    /* Number of markers */
    int j = 0; int i = 0;
    double *_geno_mkl = new double[n*m];            /* Allocate Memory for Z matrix n x m */
    /* Fill Matrix */
    //#pragma omp parallel for private(j)
    for(i = 0; i < n; i++)
    {
        string tempgeno = genotypes[i];
        for(j = 0; j < m; j++)
        {
            int tempa = tempgeno[j] - 48;
            if(tempa == 3 || tempa == 4){tempa = 1;}
            _geno_mkl[(i*m) + j] = input_m[(tempa*m)+j];
        }
    }
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, m, 1.0, _geno_mkl, m, _geno_mkl, m, 0.0, output_grm, n);
    delete[] _geno_mkl;
    // Standardize relationship matrix by scalar
    #pragma omp parallel for private(j)
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++){output_grm[(i*n)+j] /= scaler;}
    }
    
    
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 6       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Generate Genomic Relationship Matrix (GRM) with prior GRM matrix calculation  */
void grm_prevgrm(double* input_m, string genofile, vector < string > const &newgenotypes, double* output_grm12, double* output_grm22, float scaler,vector < int > &animalvector, vector < double > &phenotypevector)
{
    /* First create Z for only new individuals that can be used for old-new and new-new */
    int n = newgenotypes.size();
    int m = newgenotypes[0].size();
    // Construct Z matrix n x m
    double *_geno_mkl = new double[n*m];
    /* Fill Matrix */
    int i = 0; int j = 0;
    for(int i = 0; i < n; i++)
    {
        string tempgeno = newgenotypes[i];
        for(j = 0; j < m; j++)
        {
            int tempa = tempgeno[j] - 48;
            if(tempa == 3 || tempa == 4){tempa = 1;}
            _geno_mkl[(i*m) + j] = input_m[(tempa*m)+j];
        }
    }
    /* only need to fill in off-diagonals of new and old animals */
    /* Import file */
    int linenumber = 0;
    string line;
    ifstream infile2;
    infile2.open(genofile);
    if(infile2.fail()){cout << "Error Opening File\n";}
    while (getline(infile2,line))
    {
        /* Grab animal id */
        size_t pos = line.find(" ", 0); animalvector[linenumber] = (std::stoi(line.substr(0,pos))); line.erase(0, pos + 1);
        /* Grap Phenotype */
        pos = line.find(" ",0); phenotypevector[linenumber] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
        pos = line.find(" ",0); string tempgeno = line.substr(0,pos);                   /* Grab Marker Genotypes */
        /* Do not need the paternal and maternal haplotypes */
        if(linenumber < (phenotypevector.size() - n))                   /* once it reaches new animals doesn't do anything to those genotypes */
        {
            double *_geno_line = new double[m];
            for(j = 0; j < m; j++)
            {
                int tempa = tempgeno[j] - 48;
                if(tempa == 3 || tempa == 4){tempa = 1;}
                _geno_line[j] = input_m[(tempa*m)+j];
            }
            double *_grm_mkl = new double[n];
            // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, n, m, 1.0, _geno_line, m, _geno_mkl, m, 0.0, _grm_mkl, m);
            // Once created now standardize
            for(i = 0; i < n; i++){_grm_mkl[i] /= scaler;}
            int newanimalLocation = (phenotypevector.size() - n);
            for(int i = 0; i < n; i++){output_grm12[(linenumber*n)+i] = _grm_mkl[i];}
            delete[] _geno_line;
            delete[] _grm_mkl;
        }
        linenumber++;
    }
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
    // m rows by k columns A
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, m, 1.0, _geno_mkl, m, _geno_mkl, m, 0.0, output_grm22, n);
    delete[] _geno_mkl;
    // Standardize relationship matrix by scalar
    #pragma omp parallel for private(j)
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            output_grm22[(i*n)+j] /= scaler;
        }
    }
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 12      /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Use linear programing to generate mate allocation matrix using Kuhn-Munkres algorithm //
void LinearProgramming(double* matingvalue_matrix, string direction, vector <int> &mate_column)
{
    int dim = mate_column.size();
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
    /* Fill matrix that will be changed */
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            costmatrix[i][j] = matingvalue_matrix[(i*dim)+j] * 100;
        }
    }
    vector< vector <int> > path;
    for(int i = 0; i < (dim+200); i++)
    {
        vector < int > row(2,0);
        path.push_back(row);
    }
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
                        if(maskmatrix[i][j] == 1){objective_cost += matingvalue_matrix[(i*dim)+j];}
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
            if(maskmatrix[i][j] == 1)
            {
                mate_column[i] = j; numberones++;
            }
        }
    }
    if(numberones != dim){cout << "Something went wrong in linear programming algorith!!" << endl; exit (EXIT_FAILURE);}
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 13      /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Use sequential selection of least related mates based on Pryce et al. (2011) //
void sslr_mating(double* matingvalue_matrix, string direction, vector <int> &mate_column)
{
    int dim = mate_column.size();
    for(int i = 0; i < dim; i++){mate_column[i] = -1;}
    /* a 1 if sire has been given a mating pair or 0 if it hasn't */
    vector < int > sireusage(dim,0);
    /* loop across columns (i.e. females) and find row with lowest mating */
    for(int col = 0; col < dim; col++)
    {
        /* grab row and sort */
        vector < double > values(dim,0.0);
        vector < int > valuerow(dim,0);
        for(int i = 0; i < dim; i++){values[i] = matingvalue_matrix[(i*dim)+col]; valuerow[i] = i;}
        //for(int i = 0; i < values.size(); i++){cout << values[i] << "-" << valuerow[i] << "\t";}
        //cout << endl;
        double tempb; int temp;
        /* Sort by inbreeding coeffecient from lowest to highest */
        for(int i = 0; i < (dim - 1); i++)
        {
            for(int j = i+1; j < dim; j++)
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
        //cout << endl << endl;
        //for(int i = 0; i < values.size(); i++){cout << values[i] << "-" << valuerow[i] << "\t";}
        //cout << endl;
        string sirefound = "NO";
        if(sirefound == "NO")
        {
            /* if sire has less than Dam mating use */
            if(sireusage[valuerow[0]] == 0)
            {
                mate_column[valuerow[0]] = col;
                sireusage[valuerow[0]] = 1;
                sirefound = "YES";
            }
            /* If this is false then will not do anything and drop down to next if because damfound still = "NO" */
        }
        if(sirefound == "NO")
        {
            /* if sire is over Dam mating use next lowest one */
            if(sireusage[valuerow[0]] == 1)
            {
                int next = 1;
                string stop = "GO";
                while(stop == "GO")
                {
                    if(sireusage[valuerow[next]] == 1){next++;}
                    if(sireusage[valuerow[next]] == 0)
                    {
                        mate_column[valuerow[next]] = col;
                        sireusage[valuerow[next]] = 1;
                        stop = "NOPE";
                    }
                }
                sirefound = "YES";
            }
            /* if still no than their is an error so kill program */
            if (sirefound == "NO"){cout << endl << "sslr algorithm failed!" << endl; exit (EXIT_FAILURE);}
        }
        //cout << "----" << endl;
        //for(int i = 0; i < dim; i++)
        //{
        //    cout << sireusage[i] << "/" << mate_column[i] << "  ";
        //}
        //cout << endl << endl;
        //cout << "----" << endl;
    }
    /* Double check sire usage should sum up to dimension if not exit out */
    int totalsireusage = 0;
    for(int i = 0; i < dim; i++){totalsireusage += sireusage[i];}
    if(totalsireusage != dim){cout << endl << "SSLR Mate Algorithm Failed" << endl; exit (EXIT_FAILURE);}
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 14      /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Generate an index mate allocation matrix based on two traits //
void generate2trindex(double* mate_matrix1, double* mate_matrix2, double* mate_index_matrix, vector < double > const &indexprop, int dim, vector < int > const &rowindex, vector <double> &returnweights)
{
    using Eigen::MatrixXd; using Eigen::VectorXd;
    /* Generate a X matrix and a Y matrix; X is a function of the values and Y will be the value depending on the beta values */
    MatrixXd X((dim*dim),2); int row = 0; /* only keep one row for each individual */
    for(int i = 0; i < dim; i++)
    {
        if(i == 0){for(int j = 0; j < dim; j++){X(row,0) = mate_matrix1[(i*dim)+j]; X(row,1) = mate_matrix2[(i*dim)+j]; row++;}}
        if(i > 0)
        {
            if(rowindex[i] != rowindex[i-1])
            {
                for(int j = 0; j < dim; j++){X(row,0) = mate_matrix1[(i*dim)+j]; X(row,1) = mate_matrix2[(i*dim)+j]; row++;}
            }
        }
    }
    //cout << mate_matrix1[(0*dim)+0] << " " << mate_matrix2[(0*dim)+0] << endl;
    //cout << X(0,0) << " " << X(0,1) << endl;
    //cout << mate_matrix1[(0*dim)+1] << " " << mate_matrix2[(0*dim)+1] << endl;
    //cout << X(1,0) << " " << X(1,1) << endl;
    
    /* now center and scale each one to have a mean of 0 and variance of 1; make it easier to converge */
    double sum1 = 0.0; double sum2 = 0.0;
    double mean1 = 0.0; double mean2 = 0.0;
    double sd1 = 0.0; double sd2 = 0.0;
    for(int i = 0; i < X.rows(); i++){sum1 += X(i,0); sum2 += X(i,1);}
    mean1 = sum1 / double(X.rows()); sum1 = 0.0;
    mean2 = sum2 / double(X.rows()); sum2 = 0.0;
    for(int i = 0; i < X.rows(); i++)
    {
        sum1 += ((X(i,0) - mean1) * (X(i,0) - mean1));
        sum2 += ((X(i,1) - mean2) * (X(i,1) - mean2));
    }
    sd1 = sqrt(sum1 / double(X.rows()-1));
    sd2 = sqrt(sum2 / double(X.rows()-1));
    //cout << mean1 << " " << sd1 << " " << mean2 << " " << sd2 << endl;
    for(int i = 0; i < X.rows(); i++)
    {
        X(i,0) = (X(i,0) - mean1) / double(sd1);
        X(i,1) = (X(i,1) - mean2) / double(sd2);
    }
    /* Compute stuff that is repeatedly used outside of while loop */
    MatrixXd SSFullInverse(2,2); SSFullInverse = (X.transpose() * X).inverse();
    MatrixXd X_SSInbreed(X.rows(),1); X_SSInbreed = X.leftCols(1);
    MatrixXd SSInbreedInverse(1,1); SSInbreedInverse = (X_SSInbreed.transpose() * X_SSInbreed).inverse();
    /* Initialize Parameters */
    double diff1 = 1.0; double diff2 = 1.0;
    VectorXd beta(2); beta(0) = 1.0; beta(1) = -0.2;
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
        if((new2 - indexprop[1]) < 0.0005)
        {
            if(abs(new2 - indexprop[1]) > 0.025){beta(1) -= 0.01;}
            if(abs(new2 - indexprop[1]) >= 0.0075 && abs(new2 - indexprop[1]) < 0.025){beta(1) -= 0.001;}
            if(abs(new2 - indexprop[1]) < 0.0075){beta(1) -= 0.0001;}
        }
        if((new1 - indexprop[0]) > 0.0005 && (abs(new2 - indexprop[1]) < 0.0005)){beta(0) += 0.0001;}
        if((new2 - indexprop[1]) > 0.0005){beta(1) += 0.0001;}
        diff1 = abs(new1 - indexprop[0]);
        diff2 = abs(new2 - indexprop[1]);
        //cout << gvSS << "--" << inbSS << "--" << new1 << " - " << new2 << " \/ " << beta(0) << " \/ "  << beta(1) << endl;
        if(diff1 < 0.0005 && diff2 < 0.0005){kill = "YES";}
        iteration++;
    }
    returnweights[0] = beta(0); returnweights[1] = beta(1);
    /* Now that we know the correct weights that will generate the right variation explained by a trait make index */
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            mate_index_matrix[(i*dim)+j] = mate_matrix1[(i*dim)+j] * double(beta(0)) + mate_matrix2[(i*dim)+j] * double(beta(1));
        }
    }
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 15      /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Build ROH Relationship Matrix */
void generaterohmatrix(vector <Animal> &population,vector < hapLibrary > &haplib,vector <int> const &parentID, double* _rohrm)
{
    /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
    /* This way you don't have to repeat this step for each haplotype */
    vector < vector < int > > pathaploIDs;
    vector < vector < int > > mathaploIDs;
    for(int i = 0; i < parentID.size(); i++)
    {
        int searchlocation = 0;
        while(1)
        {
            if(parentID[i] == population[searchlocation].getID())
            {
                string PaternalHap = population[searchlocation].getPatHapl();           /* Grab Paternal Haplotype for Individual */
                string MaternalHap = population[searchlocation].getMatHapl();           /* Grab Maternal Haplotype for Individual */
                vector < int > temp_pat;
                string quit = "NO";
                while(quit != "YES")
                {
                    size_t pos = PaternalHap.find("_",0);                               /* search until last one yet */
                    if(pos > 0){temp_pat.push_back(stoi(PaternalHap.substr(0,pos))); PaternalHap.erase(0, pos + 1);}    /* extend column by 1 */
                    if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
                }
                //for(int check = 0; check < temp_pat.size(); check++){cout << temp_pat[check] << "-";}
                //cout << endl << temp_pat.size() << endl;
                pathaploIDs.push_back(temp_pat);                               /* push back row */
                vector < int > temp_mat;
                quit = "NO";
                while(quit != "YES")
                {
                    size_t pos = MaternalHap.find("_",0);                               /* search until last one yet */
                    if(pos > 0){temp_mat.push_back(stoi(MaternalHap.substr(0,pos))); MaternalHap.erase(0, pos + 1);}    /* extend column by 1 */
                    if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
                }
                //for(int check = 0; check < temp_mat.size(); check++){cout << temp_mat[check] << "-";}
                //cout << endl << temp_mat.size() << endl;
                mathaploIDs.push_back(temp_mat);
                break;
            }
            if(parentID[i] != population[searchlocation].getID()){searchlocation++;}
        }
    }
    //cout << "Pat: " << pathaploIDs.size() << " " << pathaploIDs[0].size() << endl;
    //cout << "Mat: " << mathaploIDs.size() << " " << mathaploIDs[0].size() << endl;
    for(int hap = 0; hap < haplib.size(); hap++)
    {
        vector < string > haplotypes;
        /* Unstring haplotypes, seperated by "_" */
        string temphapstring = haplib[hap].getHaplo();
        string quit = "NO";
        while(quit == "NO")
        {
            size_t pos = temphapstring.find("_",0);                         /* hasn't reached last one yet */
            if(pos > 0){haplotypes.push_back(temphapstring.substr(0,pos)); temphapstring.erase(0, pos + 1);}
            if(pos == std::string::npos){quit = "YES";}
        }
        /* ROH haplotype similarity matrix is just a diagonal matrix */
        double* H_matrix = new double[haplotypes.size()*haplotypes.size()];
        int i, j;
        #pragma omp parallel for private(j)
        for(i = 0; i < haplotypes.size(); i++)
        {
            for(j = 0; j < haplotypes.size(); j++)
            {
                if(i == j){H_matrix[(i*haplotypes.size())+j] = 1.0;}
                if(i != j){H_matrix[(i*haplotypes.size())+j] = 0.0;}
            }
        }
        /* fill relationship matrix */
        #pragma omp parallel for private(j)
        for(i = 0; i < parentID.size(); i++)
        {
            for(j = i; j < parentID.size(); j++)
            {
                _rohrm[(i*parentID.size())+j] += (H_matrix[((pathaploIDs[i][hap])*haplotypes.size())+(pathaploIDs[j][hap])] +
                                                  H_matrix[((pathaploIDs[i][hap])*haplotypes.size())+(mathaploIDs[j][hap])] +
                                                  H_matrix[((mathaploIDs[i][hap])*haplotypes.size())+(pathaploIDs[j][hap])] +
                                                  H_matrix[((mathaploIDs[i][hap])*haplotypes.size())+(mathaploIDs[j][hap])]) / 2;
                _rohrm[(j*parentID.size())+i] = _rohrm[(i*parentID.size())+j];
            }
        }
        delete [] H_matrix;
    }
    for(int i = 0; i < parentID.size(); i++)
    {
        for(int j = 0; j < parentID.size(); j++)
        {
            _rohrm[(i*parentID.size())+j] = _rohrm[(i*parentID.size())+j] / double(haplib.size());
        }
    }
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << _rohrm[(i*parentID.size())+j] << " ";}
    //    cout << endl;
    //}
}

