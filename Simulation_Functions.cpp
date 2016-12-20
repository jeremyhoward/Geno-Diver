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

/* Ensures memory is freed with a small drop in performance */

using namespace std;
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 1       /////////////////////////
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
/////////////////////////     FUNCTION 2       /////////////////////////
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
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 3       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Calculate LD decay by binning into windows;  */
void ld_decay_estimator(string outputfile, string mapfile, string lineone, vector < string > const &genotypes)
{
    mt19937 gen(time(0));
    /* Vector to store LD information */
    vector < int > ld_block_start;
    vector < int > ld_block_end;
    vector < double > r2;
    vector < int > numr2;
    for(int i = 0; i < 100; i++)
    {
        if(i == 0)
        {
            ld_block_start.push_back(1);
            ld_block_end.push_back(ld_block_start[i] + 99);
            r2.push_back(0.0);
            numr2.push_back(0);
        }
        if(i > 0)
        {
            ld_block_start.push_back(ld_block_start[i-1]+100);
            ld_block_end.push_back(ld_block_start[i] + 99);
            r2.push_back(0.0);
            numr2.push_back(0);
        }
    }
    /* read in mb and chr genotype information for markers */
    vector < int > markerpositionMb;
    vector < int > markerchromosome;
    int linenumber = 0;
    ifstream infile1;
    string line;
    infile1.open(mapfile.c_str());
    if(infile1.fail()){cout << "Error Opening File\n";}
    while (getline(infile1,line))
    {
        if(linenumber > 0)
        {
            size_t pos = line.find(" ", 0); markerchromosome.push_back((std::stoi(line.substr(0,pos)))); line.erase(0, pos + 1);
            markerpositionMb.push_back((std::stoi(line)));
        }
        linenumber++;
    }
    /* Begin looping across chromosomes and filling in ld-based statistics */
    for(int i = 0; i < markerchromosome[markerchromosome.size()-1]; i++)
    {
        vector < int > subposition; vector < int > subindex;
        for(int j = 0; j < markerchromosome.size(); j++)
        {
            if(markerchromosome[j] == (i+1)){subposition.push_back(markerpositionMb[j]);subindex.push_back(j);}
        }
        int start = 1; string kill = "NO";
        while(kill == "NO")
        {
            /* Loop across blocks of size 10 Mb and randomly grab two snp and calculate ld one done shift by 5 Mb */
            int end = start + ld_block_end[ld_block_end.size()-1];
            if(end*1000 > subposition[subposition.size()-1]){kill = "YES";} /* Once go past last SNP then stop */
            vector < int > samplingindex;
            for(int j = 0; j < subposition.size(); j++)
            {
                if(subposition[j] >= start*1000 && subposition[j] <= end*1000){samplingindex.push_back(subindex[j]);}
            }
            /* Once found a 10 Mb window randomly sample SNP within it */
            for(int k = 0; k < 500; k++)
            {
                int snp[2];
                for(int i = 0; i < 2; i++)
                {
                    std::uniform_real_distribution<double> distribution1(samplingindex[0],samplingindex[samplingindex.size()-1]);
                    snp[i] = distribution1(gen);
                    if(i == 1)
                    {
                        if(snp[0] == snp[1]){i = i-1;}
                    }
                }
                /* calculate difference and figure out where to put it in ld vectors */
                int diff = abs((markerpositionMb[snp[1]] - markerpositionMb[snp[0]]) / 1000);
                if(diff != 0)
                {
                    int j = 0;
                    while(j <  ld_block_start.size())
                    {
                        if(diff >= ld_block_start[j] && diff <= ld_block_end[j]){break;}
                        j++;
                    }
                    if(j == 100){cout << "Killed" << endl; exit (EXIT_FAILURE);}
                    int row = j;
                    /* grab genotypes */
                    double hap11 = 0; double hap12 = 0; double hap21 = 0; double hap22 = 0;
                    double freqsnp1 = 0; double freqsnp2 = 0;
                    for(int j = 0; j < genotypes.size(); j++)
                    {
                        int temp1 = atoi((genotypes[j].substr(snp[0],1)).c_str());
                        int temp2 = atoi((genotypes[j].substr(snp[1],1)).c_str());
                        /* add to haplotype frequencies */
                        if(temp1 == 0 && temp2 == 0){hap11 += 2;}
                        if(temp1 == 0 && temp2 == 2){hap12 += 2;}
                        if(temp1 == 0 && temp2 == 3){hap11 += 1; hap12 += 1;}
                        if(temp1 == 0 && temp2 == 4){hap11 += 1; hap12 += 1;}
                        if(temp1 == 2 && temp2 == 0){hap21 += 2;}
                        if(temp1 == 2 && temp2 == 2){hap22 += 2;}
                        if(temp1 == 2 && temp2 == 3){hap22 += 1; hap21 += 1;}
                        if(temp1 == 2 && temp2 == 4){hap22 += 1; hap21 += 1;}
                        if(temp1 == 3 && temp2 == 0){hap11 += 1; hap21 += 1;}
                        if(temp1 == 3 && temp2 == 2){hap12 += 1; hap22 += 1;}
                        if(temp1 == 3 && temp2 == 3){hap11 += 1; hap22 += 1;}
                        if(temp1 == 3 && temp2 == 4){hap12 += 1; hap21 += 1;}
                        if(temp1 == 4 && temp2 == 0){hap11 += 1; hap21 += 1;}
                        if(temp1 == 4 && temp2 == 2){hap12 += 1; hap22 += 1;}
                        if(temp1 == 4 && temp2 == 3){hap12 += 1; hap21 += 1;}
                        if(temp1 == 4 && temp2 == 4){hap22 += 1; hap11 += 1;}
                        /* convert 3 and 4 to one in order to calculate frequencies */
                        if(temp1 == 3 || temp1 == 4){temp1 = 1;}
                        if(temp2 == 3 || temp2 == 4){temp2 = 1;}
                        freqsnp1 += temp1; freqsnp2 += temp2;
                    }
                    /* Get frequencies */
                    hap11 = hap11 / (2 * genotypes.size()); hap12 = hap12 / (2 * genotypes.size());
                    hap21 = hap21 / (2 * genotypes.size()); hap22 = hap22 / (2 * genotypes.size());
                    freqsnp1 = freqsnp1 / (2 * genotypes.size()); freqsnp2 = freqsnp2 / (2 * genotypes.size());
                    if(hap11 != 0 && hap12 != 0 && hap21 != 0 && hap22 != 0 && (freqsnp1 > 0.0 && freqsnp1 < 1.0) && (freqsnp2 > 0.0 && freqsnp2 < 1.0))
                    {
                        double D = ((hap11*hap22 ) - (hap12*hap21)) * ((hap11*hap22 ) - (hap12*hap21));
                        double den = (freqsnp1*(1-freqsnp1)) *(freqsnp2*(1-freqsnp2));
                        r2[row] += (D / den);
                        numr2[row] += 1;
                    }
                }
            }
            start = start + 5000;
        }
    }
    // Calculate LD for a given window as mean off all LD values within a window //
    for(int i = 0; i < 100; i++){r2[i] = r2[i] / numr2[i];}
    std::ofstream output2(outputfile, std::ios_base::app | std::ios_base::out);
    if(lineone == "yes")
    {
        for(int i = 0; i < 100; i++)
        {
            if(i != 100 - 1){output2 << ld_block_end[i] << " ";}
            if(i == 100 - 1){output2 << ld_block_end[i] << endl;}
        }
    }
    for(int i = 0; i < 100; i++)
    {
        if(i != 100 - 1){output2 << r2[i] << " ";}
        if(i == 100 - 1){output2 << r2[i] << endl;}
    }
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 4       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Calculate frequency of a SNPs;  */
void frequency_calc(vector < string > const &genotypes, double* output_freq)
{
    for(int i = 0; i < genotypes[0].size(); i++)
    {
        output_freq[i] = 0;
    }
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
    for(int i = 0; i < genotypes[0].size(); i++)
    {
        output_freq[i] = output_freq[i] / (2 * genotypes.size());
    }
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
    #pragma omp parallel for private(j)
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
/////////////////////////     FUNCTION 7       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Generate QTL Summary Statistics */
void generatesummaryqtl(string inputfilehap, string inputfileqtl, string outputfile, int generations,vector < int > const &idgeneration, vector < double > const &tempaddvar, vector < double > const &tempdomvar,  vector < int > const &tempdeadfit)
{
    /* Compute Number of Haplotypes by Generation */
    /* read in all animals haplotype ID's first; Don't need to really worry about this getting big; already in order */
    vector < vector < int > > PaternalHaplotypeIDs;
    vector < vector < int > > MaternalHaplotypeIDs;
    string line;
    int linenumbera = 0;
    ifstream infile22;
    infile22.open(inputfilehap);
    if(infile22.fail()){cout << "Error Opening File\n";}
    while (getline(infile22,line))
    {
        size_t pos = line.find(" ", 0); line.erase(0, pos + 1);                                     /* Don't need animal ID so skip */
        pos = line.find(" ",0); line.erase(0,pos + 1);                                              /* Don't need phenotype so skip*/
        pos = line.find(" ",0); line.erase(0,pos + 1);                                              /* Do not need marker genotypes so skip */
        pos = line.find(" ",0); string PaternalHap = line.substr(0,pos);  line.erase(0,pos + 1);    /* Grap paternal haplotype ID's */
        string MaternalHap = line;                                                                  /* Grap maternal haplotype ID's */
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
    fstream checkSumQTL; checkSumQTL.open(outputfile, std::fstream::out | std::fstream::trunc); checkSumQTL.close();
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
    infile5.open(inputfileqtl);
    if(infile5.fail()){cout << "Error Opening File\n";}
    while (getline(infile5,line))
    {
        if(linenumberqtl > 0)
        {
            size_t pos = line.find(" ",0); line.erase(0, pos + 1);
            pos = line.find(" ",0); line.erase(0, pos + 1);
            pos = line.find(" ",0); line.erase(0, pos + 1);
            pos = line.find(" ",0); Type = line.substr(0,pos); line.erase(0, pos + 1);
            pos = line.find(" ",0); All_QTL_Gen = stoi(line.substr(0,pos)); line.erase(0, pos + 1);
            Freq_Gen = line;
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
        std::ofstream output5(outputfile, std::ios_base::app | std::ios_base::out);
        if(i == 0)
        {
            output5 << "Generation Quant_Founder_Start Quant_Founder_Lost Mutation_Quan_Total Mutation_Quan_Lost ";
            output5 << "Additive_Var Dominance_Var Fit_Founder_Start Fit_Founder_Lost Mutation_Fit_Total Mutation_Fit_Lost ";
            output5 << "Avg_Haplotypes_Window ProgenyDiedFitness" << endl;
        }
        output5 << i << " " << Founder_Quan_Total[i] << " " << Founder_Quan_Lost[i] << " " << Mutations_Quan_Total[i] << " ";
        output5 << Mutations_Quan_Lost[i] <<" "<< tempaddvar[i] <<" "<< tempdomvar[i] <<" "<< Founder_Fit_Total[i] <<" "<< Founder_Fit_Lost[i] <<" ";
        output5 << Mutations_Fit_Total[i] << " " << Mutations_Fit_Lost[i] << " " << Number_Haplotypes[i] << " " << tempdeadfit[i] << endl;
    }
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 8       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Generate dataframe Summary Statistics */
void generatessummarydf(string inputfilehap, string outputfile, int generations, vector < double > const &tempexphet)
{
    vector < double > generationcount(generations,0);
    vector < double > generationvalues;
    /* Inbreeding Summaries */
    vector < double > pedigreefvalues; vector < double > pedigreefmean(generations,0.0); vector < double > pedigreefsd(generations,0.0);
    vector < double > genomicfvalues; vector < double > genomicfmean(generations,0.0); vector < double > genomicfsd(generations,0.0);
    vector < double > h1fvalues; vector < double > h1fmean(generations,0.0); vector < double > h1fsd(generations,0.0);
    vector < double > h2fvalues; vector < double > h2fmean(generations,0.0); vector < double > h2fsd(generations,0.0);
    vector < double > h3fvalues; vector < double > h3fmean(generations,0.0); vector < double > h3fsd(generations,0.0);
    vector < double > homozygovalues; vector < double > homozygomean(generations,0.0); vector < double > homozygosd(generations,0.0);
    /* Fitness Summaries */
    vector < double > fitnessvalues; vector < double > fitnessmean(generations,0.0); vector < double > fitnesssd(generations,0.0);
    vector < double > homolethalvalues; vector < double > homolethalmean(generations,0.0); vector < double > homolethalsd(generations,0.0);
    vector < double > hetelethalvalues; vector < double > hetelethalmean(generations,0.0); vector < double > hetelethalsd(generations,0.0);
    vector < double > homosublethalvalues; vector < double > homosublethalmean(generations,0.0); vector < double > homosublethalsd(generations,0.0);
    vector < double > hetesublethalvalues; vector < double > hetesublethalmean(generations,0.0); vector < double > hetesublethalsd(generations,0.0);
    vector < double > lethalequivvalues; vector < double > lethalequivmean(generations,0.0); vector < double > lethalequivsd(generations,0.0);
    /* Performance Summaries */
    vector < double > phenovalues; vector < double > phenomean(generations,0.0); vector < double > phenosd(generations,0.0);
    vector < double > ebvvalues; vector < double > ebvmean(generations,0.0); vector < double > ebvsd(generations,0.0);
    vector < double > gvvalues; vector < double > gvmean(generations,0.0); vector < double > gvsd(generations,0.0);
    vector < double > bvvalues; vector < double > bvmean(generations,0.0); vector < double > bvsd(generations,0.0);
    vector < double > ddvalues; vector < double > ddmean(generations,0.0); vector < double > ddsd(generations,0.0);
    vector < double > resvalues; vector < double > resmean(generations,0.0); vector < double > ressd(generations,0.0);
    /* Read through and fill vector */
    string line;
    ifstream infile22;
    infile22.open(inputfilehap);
    if(infile22.fail()){cout << "Error Opening File\n";}
    while (getline(infile22,line))
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
        pos = line.find(" ",0); homolethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);  /* Grab homozygous lethal */
        pos = line.find(" ",0); hetelethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);  /* Grab heterzygous lethal */
        pos = line.find(" ",0); homosublethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);  /* Grab homozygous sublethal */
        pos = line.find(" ",0); hetesublethalvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);  /* Grab heterzygous sublethal */
        pos = line.find(" ",0); lethalequivvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);    /* Grab Lethal Equivalents */
        pos = line.find(" ",0); homozygovalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);       /* Grab homozygosity */
        pos = line.find(" ",0); fitnessvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);        /* Grab fitness */
        pos = line.find(" ",0); phenovalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);          /* Grab phenotype */
        pos = line.find(" ",0); ebvvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);            /* Grab ebv */
        pos = line.find(" ",0); line.erase(0,pos + 1);                                                          /* Don't need accuracy so skip*/
        pos = line.find(" ",0); gvvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);            /* Grab gv */
        pos = line.find(" ",0); bvvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);            /* Grab bv */
        pos = line.find(" ",0); ddvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);            /* Grab dd */
        pos = line.find(" ",0); resvalues.push_back(atof((line.substr(0,pos)).c_str())); line.erase(0,pos + 1);           /* Grab residuals */
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
        fitnesssd[generationvalues[i]] += ((fitnessvalues[i] - fitnessmean[generationvalues[i]]) * (fitnessvalues[i] - fitnessmean[generationvalues[i]]));
        homolethalsd[generationvalues[i]] += ((homolethalvalues[i]-homolethalmean[generationvalues[i]]) * (homolethalvalues[i]-homolethalmean[generationvalues[i]]));
        hetelethalsd[generationvalues[i]] += ((hetelethalvalues[i]-hetelethalmean[generationvalues[i]]) * (hetelethalvalues[i]-hetelethalmean[generationvalues[i]]));
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
    }
    for(int i = 0; i < generationcount.size(); i++)
    {
        /* Inbreeding Summary */
        pedigreefsd[i] = pedigreefsd[i] / double(generationcount[i] -1);
        genomicfsd[i] = genomicfsd[i] / double(generationcount[i] -1);
        h1fsd[i] = h1fsd[i] / double(generationcount[i] -1);
        h2fsd[i] = h2fsd[i] / double(generationcount[i] -1);
        h3fsd[i] = h3fsd[i] / double(generationcount[i] -1);
        homozygosd[i] = homozygosd[i] / double(generationcount[i] -1);
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
    }
    string outfileinbreeding = outputfile + "_Inbreeding";
    string outfileperformance = outputfile + "_Performance";
    fstream checkoutinbreeding; checkoutinbreeding.open(outfileinbreeding, std::fstream::out | std::fstream::trunc); checkoutinbreeding.close();
    fstream checkoutperformance; checkoutperformance.open(outfileperformance, std::fstream::out | std::fstream::trunc); checkoutperformance.close();
    /* output inbreeding */
    for(int i = 0; i < generations; i++)
    {
        cout.setf(ios::fixed);
        std::ofstream output(outfileinbreeding, std::ios_base::app | std::ios_base::out);
        if(i == 0)
        {
            output << "Generation ped_f gen_f h1_f h2_f h3_f homozy ExpHet fitness homozlethal hetezlethal homozysublethal hetezsublethal lethalequiv" << endl;
        }
        output << i << " ";
        output << setprecision(4) << pedigreefmean[i] << "(" << pedigreefsd[i] << ") ";
        output << setprecision(4) << genomicfmean[i] << "(" << genomicfsd[i] << ") ";
        output << setprecision(4) << h1fmean[i] << "(" << h1fsd[i] << ") ";
        output << setprecision(4) << h2fmean[i] << "(" << h2fsd[i] << ") ";
        output << setprecision(4) << h3fmean[i] << "(" << h3fsd[i] << ") ";
        output << setprecision(4) << homozygomean[i] << "(" << homozygosd[i] << ") ";
        output << setprecision(4) << tempexphet[i] << " ";       
        output << setprecision(4) << fitnessmean[i] << "(" << fitnesssd[i] << ") ";
        output << setprecision(4) << homolethalmean[i] << "(" << homolethalsd[i] << ") ";
        output << setprecision(4) << hetelethalmean[i] << "(" << hetelethalsd[i] << ") ";
        output << setprecision(4) << homosublethalmean[i] << "(" << homosublethalsd[i] << ") ";
        output << setprecision(4) << hetesublethalmean[i] << "(" << hetesublethalsd[i] << ") ";
        output << setprecision(4) << lethalequivmean[i] << "(" << lethalequivsd[i] << ")" << endl;
        cout.unsetf(ios::fixed);
    }
    /* output performance */
    for(int i = 0; i < generations; i++)
    {
        cout.setf(ios::fixed);
        std::ofstream output1(outfileperformance, std::ios_base::app | std::ios_base::out);
        if(i == 0)
        {
            output1 << "Generation phen ebv gv bv dd res" << endl;
        }
        output1 << i << " ";
        output1 << setprecision(4) << phenomean[i]  << "(" << phenosd[i] << ") ";
        output1 << setprecision(4) << ebvmean[i]  << "(" << ebvsd[i] << ") ";
        output1 << setprecision(4) << gvmean[i]  << "(" << gvsd[i] << ") ";
        output1 << setprecision(4) << bvmean[i]  << "(" << bvsd[i] << ") ";
        output1 << setprecision(4) << ddmean[i]  << "(" << ddsd[i] << ") ";
        output1 << setprecision(4) << resmean[i]  << "(" << ressd[i] << ")" << endl;
        cout.unsetf(ios::fixed);
    }
    
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 9       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Solve for solutions using pcg */
void pcg_solver(double* lhs, double* rhs, vector < double > &solutionsa, int dimen, int* solvediter)
{
    //  PCG involves 4 vectors (Notation is like Mrodes book)
    //  e:          vector of residuals (size: number of unknowns)
    //                  - Initially found by: RHS - (LHS * beta)
    //  solutions:  vector of solutions (size: number of unknowns)
    //                  - Initially given based on starting values
    //  d:          vector of search directions
    //                  - Initially given by: Minv * e (Minv is preconditioner matrix diagonals of LHS)
    //  v:          working vector
    /* Initialize vectors or matrices */
    double* solutions = new double[dimen];
    double* Minv = new double[dimen*dimen];
    double* e = new double[dimen];
    double* d = new double[dimen];
    double* v = new double[dimen];
    double* p = new double[dimen];
    double* oldb = new double[dimen];
    double* tau1 = new double[1];
    double* tau2 = new double[1];
    double* alpha = new double[1];
    double* beta = new double[1];
    double* intermediate = new double[1];
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++)                                  /* set solutions to zero & copy rhs to e */
    {
        solutions[i] = solutionsa[i]; e[i] = 0; d[i] = 0; v[i] = 0; p[i] = 0; oldb[i] = 0;
    }
    for(int i = 0; i < dimen; i++)                                  /* inverse of diagonals is just 1 / diagonal of LHS */
    {
        #pragma omp parallel for
        for(int j = i; j < dimen; j++)
        {
            if(i == j){Minv[(i*dimen)+i] = 1 / double(lhs[(i*dimen)+i]);}
            if(i != j){Minv[(i*dimen)+j] = 0.0; Minv[(j*dimen)+i] = 0.0;}
        }
    }
    /* intialize parameters */
    tau1[0] = 0; tau2[0] = 1; alpha[0] = 0; beta[0] = 0;
    int k = 1;
    float diff = 1;
    float tol = 1E-12;
    
    /* Variables used in mkl functions */
    const long long int lhssize = dimen;
    const long long int onesize = 1;
    const long long int increment = int(1);
    int incx = 1; string stop = "NO";
    int incy = 1;
    dcopy(&lhssize,rhs,&increment,e,&increment);                      /* copy rhs to e */
    while(stop == "NO")
    {
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
        /* d = Minv * e */
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,lhssize,onesize,lhssize,1.0,Minv,lhssize,e,onesize,0.0,d,onesize);
        tau1[0] = cblas_ddot(lhssize,d,incx,e,incy);                        /* tau1 = e dot product d */
        if(k == 1)
        {
            dcopy(&lhssize,d,&increment,p,&increment);                      /* copy d to p */
            beta[0] = 0;
        }
        if(k > 1)
        {
            beta[0] = tau1[0] / tau2[0];
            #pragma omp parallel for
            for(int i = 0; i < dimen; i++){p[i] = d[i] + (beta[0]*p[i]);}
        }
        /* v = LHS * p */
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,lhssize,onesize,lhssize,1.0,lhs,lhssize,p,onesize,0.0,v,onesize);
        /* get alpha = tau1 / p.dot(v) */
        intermediate[0] = cblas_ddot(lhssize,p,incx,v,incy);
        alpha[0] = tau1[0] / intermediate[0];
        dcopy(&lhssize,solutions,&increment,oldb,&increment);                      /* copy d to p */
        /* get new solutions and residuals */
        #pragma omp parallel for
        for(int i = 0; i < dimen; i++)
        {
            solutions[i] = solutions[i] + (alpha[0]*p[i]);
            e[i] = e[i] - (alpha[0]*v[i]);
        }
        k++;
        /* Compute difference to determine if converged sum (current - previous beta)^2 / sum(current)^2 */
        float num = 0.0;
        float den = 0.0;
        for(int i = 0; i < dimen; i++)
        {
            num += (solutions[i] - oldb[i]) * (solutions[i] - oldb[i]);
            den += (solutions[i] * solutions[i]);
        }
        diff = num / float(den);
        tau2[0] = tau1[0];
        if(diff < tol){stop = "YES";}
    }
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++){solutionsa[i] = solutions[i];}
    solvediter[0] = k;
    delete[] Minv; delete[] e; delete[] d; delete[] v; delete[] p; delete[] intermediate;
    delete[] oldb; delete[] tau1; delete[] tau2; delete[] alpha; delete[] beta; delete[] solutions;
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 10       /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Solve for solutions using brute force inversion */
void direct_solver(double* lhs, double* rhs, vector < double > &solutionsa, int dimen)
{
    double* solutions = new double[dimen];
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++){solutions[i] = solutionsa[i];}                                 /* set solutions to zero & copy rhs to e */
    /* parameters for mkl */
    unsigned long i_p = 0, j_p = 0;
    unsigned long n = dimen;
    long long int info = 0;
    const long long int int_n =(int)n;
    char lower='L';
    dpotrf(&lower, &int_n, lhs, &int_n, &info);          /* Calculate upper triangular L matrix */
    dpotri(&lower, &int_n, lhs, &int_n, &info);          /* Calculate inverse of lower triangular matrix result is the inverse */
    mkl_thread_free_buffers;
    /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
    #pragma omp parallel for private(j_p)
    for(j_p=0; j_p < n; j_p++)
    {
        for(i_p=0; i_p <= j_p; i_p++)
        {
            lhs[(j_p*n)+i_p] = lhs[(i_p*n)+j_p];
        }
    }
    const long long int lhssize = dimen;
    const long long int onesize = 1;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,lhssize,onesize,lhssize,1.0,lhs,lhssize,rhs,onesize,0.0,solutions,onesize);
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++){solutionsa[i] = solutions[i];}                                 /* set solutions to zero & copy rhs to e */
    delete [] solutions;

}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////     FUNCTION 11      /////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* Calculates subset of a relationship matrix */
void pedigree_relationship(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship)
{
    vector < int > animal; vector < int > sire; vector < int > dam;
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





