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
#include <tuple>
#include <unordered_map>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <mkl.h>

/* Function to create Beta Distribution. Random library does not produce it; stems from a gamma. */
namespace sftrabbit
{
    template <typename RealType = double>
    class beta_distribution
    {
    public:
        typedef RealType result_type;
        class param_type
        {
        public:
            typedef beta_distribution distribution_type;
            explicit param_type(RealType a = 2.0, RealType b = 2.0)
            : a_param(a), b_param(b) { }
            RealType a() const { return a_param; }
            RealType b() const { return b_param; }
            bool operator==(const param_type& other) const
            {
                return (a_param == other.a_param &&
                        b_param == other.b_param);
            }
            bool operator!=(const param_type& other) const
            {
                return !(*this == other);
            }
        private:
            RealType a_param, b_param;
        };
        explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
        : a_gamma(a), b_gamma(b) { }
        explicit beta_distribution(const param_type& param)
        : a_gamma(param.a()), b_gamma(param.b()) { }
        void reset() { }
        param_type param() const
        {
            return param_type(a(), b());
        }
        void param(const param_type& param)
        {
            a_gamma = gamma_dist_type(param.a());
            b_gamma = gamma_dist_type(param.b());
        }
        template <typename URNG>
        result_type operator()(URNG& engine)
        {
            return generate(engine, a_gamma, b_gamma);
        }
        template <typename URNG>
        result_type operator()(URNG& engine, const param_type& param)
        {
            gamma_dist_type a_param_gamma(param.a()),
            b_param_gamma(param.b());
            return generate(engine, a_param_gamma, b_param_gamma);
        }
        result_type min() const { return 0.0; }
        result_type max() const { return 1.0; }
        result_type a() const { return a_gamma.alpha(); }
        result_type b() const { return b_gamma.alpha(); }
        bool operator==(const beta_distribution<result_type>& other) const
        {
            return (param() == other.param() &&
                    a_gamma == other.a_gamma &&
                    b_gamma == other.b_gamma);
        }
        bool operator!=(const beta_distribution<result_type>& other) const
        {
            return !(*this == other);
        }
    private:
        typedef std::gamma_distribution<result_type> gamma_dist_type;
        gamma_dist_type a_gamma, b_gamma;
        template <typename URNG>
        result_type generate(URNG& engine,
                             gamma_dist_type& x_gamma,
                             gamma_dist_type& y_gamma)
        {
            result_type x = x_gamma(engine);
            return x / (x + y_gamma(engine));
        }
    };
    
    template <typename CharT, typename RealType>
    std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os,const beta_distribution<RealType>& beta)
    {
        os << "~Beta(" << beta.a() << "," << beta.b() << ")";
        return os;
    }
    template <typename CharT, typename RealType>
    std::basic_istream<CharT>& operator>>(std::basic_istream<CharT>& is,beta_distribution<RealType>& beta)
    {
        std::string str;
        RealType a, b;
        if (std::getline(is, str, '(') && str == "~Beta" &&
            is >> a && is.get() == ',' && is >> b && is.get() == ')') {
            beta = beta_distribution<RealType>(a, b);
        } else {
            is.setstate(std::ios::failbit);
        }
        return is;
    }
}


#include "Animal.h"
#include "ParameterClass.h"

using namespace std;
/******************************/
/*     Catch All Functions    */
/******************************/
void updateinbreeding(vector <Animal> &population,vector <int> &animal, vector <double> &pedigreeinb);
double proportiongenotyped(parameters &SimParameters,string GenotypeStatus_path);
void update_animgreat(parameters &SimParameters,vector <Animal> &population, vector <int> &trainanimals,string Pheno_Pedigree_File,int Gen);
void Inbreeding_Pedigree(vector <Animal> &population,string Pheno_Pedigree_File);
void MatrixStats(vector <double> &summarystats, double* output_subrelationship, int dimen,ostream& logfileloc);
void MatrixCorrStats(vector <double> &summarystats, double* _grm_mkl, double* A22relationship, int dimen,ostream& logfileloc);
void frequency_calc(vector < string > const &genotypes, double* output_freq);
float update_M_scale_Hinverse(parameters &SimParameters, vector <string> &genotypes, double* M);

/******************************/
/*     Set up Blup MME        */
/******************************/
void Setup_blup(parameters &SimParameters,vector <Animal> &population,vector < tuple <int,int,double> > &sprelationshipinv,vector <double> &Phenotype,vector <int> &animal,double scalinglambda,vector <double> &estimatedsolutions, vector <double> &trueaccuracy,ostream& logfileloc);
void phenotypestatus(parameters &SimParameters,vector <Animal> &population,vector <int> &animal, vector <int> &hasphenotype);

/******************************/
/* Generate Inverse Functions */
/******************************/
/* gblup type matrices */
void newgenomicrecursion(parameters &SimParameters, int relationshipsize, string BinaryG_Matrix_File,string Binarym_Matrix_File, string Binaryp_Matrix_File, string BinaryGinv_Matrix_File,vector < tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal);
void updategenomicrecursion(int TotalAnimalNumber, int TotalOldAnimalNumber, string BinaryG_Matrix_File,string Binarym_Matrix_File, string Binaryp_Matrix_File, string BinaryGinv_Matrix_File,vector <tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal);
void newgenomiccholesky(parameters &SimParameters, int relationshipsize, string BinaryG_Matrix_File,string BinaryLinv_Matrix_File, string BinaryGinv_Matrix_File, vector < tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal);
void updategenomiccholesky(int TotalAnimalNumber, int TotalOldAnimalNumber, string BinaryG_Matrix_File,string BinaryLinv_Matrix_File, string BinaryGinv_Matrix_File, vector < tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal);
/* h-inverse type matrices */
void H_inverse_function(parameters &SimParameters, vector <Animal> &population,string Pheno_Pedigree_File,string Pheno_GMatrix_File,string GenotypeStatus_path, vector <int> &animal, vector <double> &Phenotype, vector <int> trainanimals, vector < tuple<int,int,double> > &sprelationshipinv, ostream& logfileloc);
/* pedigree type matrices */
void Setup_SpmeuwissenluoAinv(string Pheno_Pedigree_File, vector <int> &animal, vector <double> &Phenotype,  vector <int> &trainanimals,vector < tuple <int,int,double> > &sprelationshipinv,vector <double> &pedigreeinb, ostream& logfileloc);
void Meuwissen_Luo_Ainv_Sparse(vector <int> &animal,vector <int> &sire,vector <int> &dam,vector <tuple<int,int,double>> &sprelationshipinv,vector <double> &pedinbreeding);
void DirectInverse(double* relationship, int dimension);

/***********************************/
/* Generate Relationship Functions */
/***********************************/
/* haplotype type matrices */
void newhaplotyperelationship(parameters &SimParameters,vector <Animal> &population, vector < hapLibrary > &haplib, string Pheno_GMatrix_File, vector <int> &animal, vector <double> &Phenotype, vector <int> trainanimals, string BinaryG_Matrix_File, ostream& logfileloc);
void updatehaplotyperelationship(parameters &SimParameters,vector <Animal> &population, vector < hapLibrary > &haplib, string Pheno_GMatrix_File, vector <int> &animal, vector <double> &Phenotype,string BinaryG_Matrix_File,int TotalOldAnimalNumber,int TotalAnimalNumber,ostream& logfileloc);
/* gblup type matrices */
void VanRaden_grm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);
void newgenomicrelationship(parameters &SimParameters, vector <Animal> &population, string Pheno_GMatrix_File,vector <int> &animal, vector <double> &Phenotype, double* M, float scale, vector <int> trainanimals, string BinaryG_Matrix_File, ostream& logfileloc);
void updategenomicrelationship(parameters &SimParameters, vector <Animal> &population, string Pheno_GMatrix_File,vector <int> &animal, vector <double> &Phenotype, double* M, float scale, string BinaryG_Matrix_File,int TotalOldAnimalNumber,int TotalAnimalNumber, ostream& logfileloc);
/* Pedigree type matrices */
void pedigree_relationship_Colleau(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship);
void A22_Colleau(vector <int> const &animal,vector <int> const &sire,vector <int> const &dam,vector <int> const &parentid, vector <double> const &pedinbreeding, double* output_subrelationship);

/***********/
/* Solvers */
/***********/
void pcg_solver_dense(vector <tuple<int,int,double>> &lhs_sparse,vector <double> &rhs_sparse, vector <double> &solutionsa, int dimen, int* solvediter);
void pcg_solver_sparse(vector <tuple<int,int,double>> &lhs_sparse,vector <double> &rhs_sparse, vector <double> &solutionsa, int dimen, int* solvediter);
void direct_solversparse(parameters &SimParameters,vector <tuple<int,int,double>> &lhs_sparse,vector <double> &rhs_sparse, vector <double> &solutionsa, vector <double> &trueaccuracy,int dimen);

/*******************************************************************/
/* Functions to output and input eigen matrices into binary format */
/*******************************************************************/
namespace Eigen
{
    template<class Matrix>
    void writebinary(const char* filename, const Matrix& matrix)
    {
        std::ofstream out(filename,ios::out | ios::binary | ios::trunc); typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index)); out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) ); out.close();
    }
    template<class Matrix>
    void readbinary(const char* filename, Matrix& matrix)
    {
        std::ifstream in(filename,ios::in | std::ios::binary); typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index)); in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols); in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar)); in.close();
    }
}

/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/********************                                     Miscellaneous Functions                                    ********************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
void updateinbreeding(vector <Animal> &population,vector <int> &animal, vector <double> &pedigreeinb)
{
    for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
    {
        int j = 0;                                                                  /* Counter for population spot */
        while(1)
        {
            if(population[i].getID() == animal[j]){population[i].UpdateInb(pedigreeinb[j]); break;}
            j++;                                                                    /* Loop across until animal has been found */
        }
    }
}
/********************************************************************************/
/* Generate pedigree based inbreeding values based on Meuwissen & Luo Algorithm */
/********************************************************************************/
void Inbreeding_Pedigree(vector <Animal> &population,string Pheno_Pedigree_File)
{
    vector < int > animal;
    vector < int > sire;
    vector < int > dam;
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    string line;
    ifstream infile2;
    infile2.open(Pheno_Pedigree_File.c_str());                                                 /* This file has all animals in it */
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
    vector <double> animal_inbreeding(animal.size(),0.0);
    for(int i = 1; i < (animal.size()+1); i++){animal_inbreeding[i-1] = F[i];}
    /* copy vector back to output_qs_u array */
    for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
    {
        int j = 0;                                                                  /* Counter for population spot */
        while(1)
        {
            if(population[i].getID() == animal[j])
            {
                double temp = animal_inbreeding[j]; population[i].UpdateInb(temp); break;
            }
            j++;                                                                    /* Loop across until animal has been found */
        }
    }
    animal.clear(); sire.clear(); dam.clear(); F.clear(); D.clear(); animal_inbreeding.clear();
}
/*****************************************************************************************/
/* Generate new M and scaler for G based on observed frequency only selection candidates */
/*****************************************************************************************/
float update_M_scale(parameters &SimParameters, vector <Animal> &population,double* M)
{
    int mostrecentgen = -5;
    for(int i = 0; i < population.size(); i++){if(population[i].getGeneration() > mostrecentgen){mostrecentgen = population[i].getGeneration();}}
    vector < string > tempgenofreqcalc;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getGeneration() == mostrecentgen){tempgenofreqcalc.push_back(population[i].getMarker());}
    }
    double* freq = new double[tempgenofreqcalc[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
    frequency_calc(tempgenofreqcalc, freq);                              /* Function to calculate snp frequency */
    //for(int i = 0; i < 5; i++){cout << freq[i] << " ";}
    //cout << endl;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < tempgenofreqcalc[0].size(); j++){M[(i*tempgenofreqcalc[0].size())+j] = i - (2 * freq[j]);}
    }
    /* Calculate Scale */
    float scaletemp = 0;
    for (int j = 0; j < tempgenofreqcalc[0].size(); j++){scaletemp += (1 - freq[j])*freq[j];}
    scaletemp = scaletemp *2;
    delete [] freq; tempgenofreqcalc.clear();
    return (scaletemp);
}
/**********************************************************************************************************/
/***       Checks proportion of genotyped animals to ensure you are at the correct ebv estimation place ***/
/**********************************************************************************************************/
double proportiongenotyped(parameters &SimParameters,string GenotypeStatus_path)
{
    string line;
    ifstream infile; int totalanimals = 0; double genotypedanimals = 0;
    infile.open(GenotypeStatus_path.c_str());
    if(infile.fail()){cout << "GenotypeStatus!\n"; exit (EXIT_FAILURE);}
    while (getline(infile,line))
    {
        size_t pos = line.find(" ",0); line.erase(0, pos + 1);
        if(line == "Full" || line == "Reduced"){genotypedanimals++;}
        totalanimals++;
    }
    //cout << totalanimals << " " << genotypedanimals << endl;
    if(genotypedanimals > 0){genotypedanimals /= double(totalanimals);}
    return genotypedanimals;
}
/**********************************************************************************************************/
/* Generate new M and scaler for G based on observed frequency for animals in G when creating Hinv matrix */
/**********************************************************************************************************/
float update_M_scale_Hinverse(parameters &SimParameters, vector <string> &genotypes, double* M)
{
    double* freq = new double[genotypes[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
    frequency_calc(genotypes, freq);                              /* Function to calculate snp frequency */
    //for(int i = 0; i < 15; i++){cout << freq[i] << " ";}
    //cout << endl;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < genotypes[0].size(); j++){M[(i*genotypes[0].size())+j] = i - (2 * freq[j]);}
    }
    /* Calculate Scale */
    float scaletemp = 0;
    for (int j = 0; j < genotypes[0].size(); j++){scaletemp += (1 - freq[j])*freq[j];}
    scaletemp = scaletemp *2;
    delete [] freq;
    return (scaletemp);
}
/******************************************************************/
/***       Update whether an animal is genotyped or not         ***/
/******************************************************************/
void updateanimalgenostatus(parameters &SimParameters,vector <Animal> &population,string GenotypeStatus_path)
{
    /* across all methods first update whether an animal is genotyped in the parent population once that is updated then proceed to */
    /* update the 'GenotypeStatus' file with the genotype status of an animal */
    /* If doing random then first make a key to figure out whether to genotype or not */
    unordered_map <int, int> GenoStatuslinker;
    vector <double> randev_M; vector <double> randev_F;
    if(SimParameters.getMaleWhoGenotype() == "ebv_parentavg" || SimParameters.getFemaleWhoGenotype() == "ebv_parentavg" || SimParameters.getMaleWhoGenotype() == "parents_ebv_parentavg" || SimParameters.getFemaleWhoGenotype() == "parents_ebv_parentavg") /* Generate parent average */
    {
        unordered_map <int, double> parentebv;
        for(int i = 0; i < population.size();  i++)
        {
            if(population[i].getAge() > 1){parentebv.insert({population[i].getID(),population[i].getEBV()});}
        }
        //for(pair<int,double> element:parentebv){cout << element.first << " :: " << element.second << endl;}
        for(int i = 0; i < population.size();  i++)
        {
            if(population[i].getAge() == 1)
            {
                if(population[i].getSex() == 0)
                {
                    randev_M.push_back(((parentebv[population[i].getSire()] + parentebv[population[i].getDam()]) * 0.5));
                }
                if(population[i].getSex() == 1)
                {
                    randev_F.push_back(((parentebv[population[i].getSire()] + parentebv[population[i].getDam()]) * 0.5));
                }
            }
        }
        if(SimParameters.getSelectionDir() == "low")    /* sort lowest to highest */
        {
            double temp;
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
        if(SimParameters.getSelectionDir() == "high")   /* Sort highest to lowest */
        {
            double temp;
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
        //for(int i = 0; i < randev_M.size(); i++){cout << randev_M[i] << " ";}
        //cout << endl << endl;
        double malegenotypecutoff = randev_M[int(SimParameters.getMalePropGenotype()*randev_M.size()+1)-1];
        //for(int i = 0; i < randev_F.size(); i++){cout << randev_F[i] << " ";}
        //cout << endl << endl;
        double femalegenotypecutoff = randev_F[int(SimParameters.getFemalePropGenotype()*randev_F.size()+1)-1];
        //cout << malegenotypecutoff << " " << femalegenotypecutoff << endl;
        for(int i = 0; i < population.size();  i++)
        {
            if(population[i].getAge() == 1)
            {
                if(SimParameters.getSelectionDir() == "high")
                {
                    if(population[i].getSex() == 0)
                    {
                        if(((parentebv[population[i].getSire()] + parentebv[population[i].getDam()]) * 0.5) >= malegenotypecutoff)
                        {
                            GenoStatuslinker.insert({population[i].getID(),1});
                        } else {GenoStatuslinker.insert({population[i].getID(),0});}
                    }
                    if(population[i].getSex() == 1)
                    {
                        if(((parentebv[population[i].getSire()] + parentebv[population[i].getDam()]) * 0.5) >= femalegenotypecutoff)
                        {
                            GenoStatuslinker.insert({population[i].getID(),1});
                        } else {GenoStatuslinker.insert({population[i].getID(),0});}
                    }
                }
                if(SimParameters.getSelectionDir() == "low")
                {
                    if(population[i].getSex() == 0)
                    {
                        if(((parentebv[population[i].getSire()] + parentebv[population[i].getDam()]) * 0.5) <= malegenotypecutoff)
                        {
                            GenoStatuslinker.insert({population[i].getID(),1});
                        } else {GenoStatuslinker.insert({population[i].getID(),0});}
                    }
                    if(population[i].getSex() == 1)
                    {
                        if(((parentebv[population[i].getSire()] + parentebv[population[i].getDam()]) * 0.5) <= femalegenotypecutoff)
                        {
                            GenoStatuslinker.insert({population[i].getID(),1});
                        } else {GenoStatuslinker.insert({population[i].getID(),0});}
                    }
                }
            }
            if(population[i].getAge() > 1)
            {
                if(population[i].getSex() == 0)
                {
                    if(SimParameters.getMaleWhoGenotype() == "ebv_parentavg"){GenoStatuslinker.insert({population[i].getID(),0});}
                    if(SimParameters.getMaleWhoGenotype() == "parents_ebv_parentavg"){GenoStatuslinker.insert({population[i].getID(),1});}
                    
                }
                if(population[i].getSex() == 1)
                {
                    if(SimParameters.getFemaleWhoGenotype() == "ebv_parentavg"){GenoStatuslinker.insert({population[i].getID(),0});}
                    if(SimParameters.getFemaleWhoGenotype() == "parents_ebv_parentavg"){GenoStatuslinker.insert({population[i].getID(),1});}
                }
            }
        }
    }
    if(SimParameters.getMaleWhoGenotype() == "random" || SimParameters.getMaleWhoGenotype() == "parents_random" || SimParameters.getFemaleWhoGenotype() == "random" || SimParameters.getFemaleWhoGenotype() == "parents_random")    /* Randomly sample genotypes */
    {
        for(int i = 0; i < population.size();  i++)
        {
            if(population[i].getAge() == 1)
            {
                if(population[i].getSex() == 0){randev_M.push_back(population[i].getRndSelection());}
                if(population[i].getSex() == 1){randev_F.push_back(population[i].getRndSelection());}
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
        //for(int i = 0; i < randomdeviate_Male.size(); i++){cout << randomdeviate_Male[i] << " ";}
        double malegenotypecutoff = randev_M[int(SimParameters.getMalePropGenotype()*randev_M.size()+1)-1];
        /* order and take a certain percentage */
        for(int i = 0; i < (randev_F.size()-1); i++)   /* Sort Females */
        {
            for(int j=i+1; j< randev_F.size(); j++)
            {
                if(randev_F[i] > randev_F[j]){temp = randev_F[i]; randev_F[i] = randev_F[j]; randev_F[j] = temp;}
            }
        }
        //for(int i = 0; i < randomdeviate_Male.size(); i++){cout << randomdeviate_Male[i] << " ";}
        double femalegenotypecutoff = randev_F[int(SimParameters.getFemalePropGenotype()*randev_F.size()+1)-1];
        //cout << "Male: " << malegenotypecutoff << "  Female: " <<femalegenotypecutoff << endl;
        for(int i = 0; i < population.size();  i++)
        {
            if(population[i].getAge() == 1)
            {
                if(population[i].getSex() == 0)
                {
                    if(population[i].getRndSelection() <= malegenotypecutoff)
                    {
                        GenoStatuslinker.insert({population[i].getID(),1});
                    } else {GenoStatuslinker.insert({population[i].getID(),0});}
                }
                if(population[i].getSex() == 1)
                {
                    if(population[i].getRndSelection() <= femalegenotypecutoff)
                    {
                        GenoStatuslinker.insert({population[i].getID(),1});
                    } else {GenoStatuslinker.insert({population[i].getID(),0});}
                }
            }
            if(population[i].getAge() > 1)
            {
                if(population[i].getSex() == 0)
                {
                    if(SimParameters.getMaleWhoGenotype() == "random"){GenoStatuslinker.insert({population[i].getID(),0});}
                    if(SimParameters.getMaleWhoGenotype() == "parents_random"){GenoStatuslinker.insert({population[i].getID(),1});}

                }
                if(population[i].getSex() == 1)
                {
                    if(SimParameters.getFemaleWhoGenotype() == "random"){GenoStatuslinker.insert({population[i].getID(),0});}
                    if(SimParameters.getFemaleWhoGenotype() == "parents_random"){GenoStatuslinker.insert({population[i].getID(),1});}
                }
            }
        }
        //for(pair<int,int> element:GenoStatuslinker){cout << element.first << " :: " << element.second << endl;}
    }
    /* read in file */
    vector < string > genostatusrow;
    string line, tempid;
    ifstream infile;
    infile.open(GenotypeStatus_path.c_str());
    if(infile.fail()){cout << "GenotypeStatus!\n"; exit (EXIT_FAILURE);}
    while (getline(infile,line)){genostatusrow.push_back(line);}
    /* update file */
    for(int i = 0; i < population.size();  i++)
    {
        if(population[i].getSex() == 0)             /* males */
        {
            if(SimParameters.getMaleWhoGenotype() == "parents")         /* Genotype All Parents */
            {
                if(population[i].getAge() > 1)     /* Animal has been selected so now is genotyped */
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
            }
            if(SimParameters.getMaleWhoGenotype() == "offspring")         /* Genotype All Parents */
            {
                if(population[i].getAge() == 1)     /* Animal has been selected so now is genotyped */
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
            }
            if(SimParameters.getMaleWhoGenotype() == "parents_offspring")         /* Genotype All Parents and Offspring */
            {
                if(population[i].getAge() >= 1)     /* Animal has been selected so now is genotyped */
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
            }
            if(SimParameters.getMaleWhoGenotype() == "random" || SimParameters.getMaleWhoGenotype() == "parents_random" || SimParameters.getMaleWhoGenotype() == "ebv_parentavg" || SimParameters.getMaleWhoGenotype() == "parents_ebv_parentavg")
            {
                if(GenoStatuslinker[population[i].getID()] == 1)
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
                //cout << population[i].getID() << " " << GenoStatuslinker[population[i].getID()] << endl;
                //exit (EXIT_FAILURE);
            }
        }
        if(population[i].getSex() == 1)             /* females */
        {
            if(SimParameters.getFemaleWhoGenotype() == "parents")         /* Genotype All Parents */
            {
                if(population[i].getAge() > 1)     /* Animal has been selected so now is genotyped */
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
            }
            if(SimParameters.getFemaleWhoGenotype() == "offspring")         /* Genotype All Parents */
            {
                if(population[i].getAge() == 1)     /* Animal has been selected so now is genotyped */
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
            }
            if(SimParameters.getFemaleWhoGenotype() == "parents_offspring")         /* Genotype All Parents and Offspring */
            {
                if(population[i].getAge() >= 1)     /* Animal has been selected so now is genotyped */
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
            }
            if(SimParameters.getFemaleWhoGenotype() == "random" || SimParameters.getFemaleWhoGenotype() == "parents_random" || SimParameters.getFemaleWhoGenotype() == "ebv_parentavg" || SimParameters.getFemaleWhoGenotype() == "parents_ebv_parentavg")
            {
                if(GenoStatuslinker[population[i].getID()] == 1)
                {
                    //cout << population[i].getID() << "  --  " << genostatusrow[population[i].getID()-1] << "  --  ";
                    population[i].UpdateGenoStatus("Full");     /* Change genotype status in population */
                    /* Update in file */
                    size_t pos = genostatusrow[population[i].getID()-1].find("No",0);   /* only update if was a 'No'" */
                    if(pos != std::string::npos)
                    {
                        pos = genostatusrow[population[i].getID()-1].find(" ",0); tempid = genostatusrow[population[i].getID()-1].substr(0,pos);
                        genostatusrow[population[i].getID()-1] = tempid + " Full";
                        //cout << genostatusrow[population[i].getID()-1] << endl;
                    }
                }
                //cout << population[i].getID() << " " << GenoStatuslinker[population[i].getID()] << endl;
                //exit (EXIT_FAILURE);
            }
        }
    }
    /* First remove contents from old one then update with new genotype status */
    fstream cleargenostatus; cleargenostatus.open(GenotypeStatus_path.c_str(), std::fstream::out | std::fstream::trunc); cleargenostatus.close();
    stringstream outputstringgenostatus(stringstream::out);
    for(int i = 0; i < genostatusrow.size(); i++){outputstringgenostatus << genostatusrow[i] << endl;}
    /* output genostatus file */
    std::ofstream outputg(GenotypeStatus_path.c_str(), std::ios_base::app | std::ios_base::out);
    outputg << outputstringgenostatus.str(); outputstringgenostatus.str(""); outputstringgenostatus.clear();
}
/******************************************************************/
/***       Proportion of breeding Population Genotyped          ***/
/******************************************************************/
void breedingpopulationgenotyped(vector <Animal> &population, ostream& logfileloc)
{
    int maleoffspring = 0; int femaleoffspring = 0; int maleparents = 0; int femaleparents = 0;
    for(int i = 0; i < population.size();  i++)
    {
        if(population[i].getGenoStatus() != "No")
        {
            if(population[i].getAge() > 1)
            {
                if(population[i].getSex() == 0){maleparents++;}
                if(population[i].getSex() == 1){femaleparents++;}
            }
            if(population[i].getAge() == 1)
            {
                if(population[i].getSex() == 0){maleoffspring++;}
                if(population[i].getSex() == 1){femaleoffspring++;}
            }
        }
    }
    logfileloc << "   Breeding Population Genotype Status: " << endl;
    logfileloc << "     - Parents: " << endl;
    logfileloc << "         - Males: " << maleparents << endl;
    logfileloc << "         - Females: " << femaleparents << endl;
    logfileloc << "     - Selection Candidates: " << endl;
    logfileloc << "         - Males: " << maleoffspring << endl;
    logfileloc << "         - Females: " << femaleoffspring << endl << endl;
}
/******************************************************************/
/***    Total number of animals genotyped across all system     ***/
/******************************************************************/
int getTotalGenotypeCount(string GenotypeStatus_path)
{
    int numberofgenotyped = 0;
    /* read in file */
    string line;
    ifstream infile;
    infile.open(GenotypeStatus_path.c_str());
    if(infile.fail()){cout << "GenotypeStatus!\n"; exit (EXIT_FAILURE);}
    while (getline(infile,line))
    {
        size_t pos = line.find("No",0);   /* only update if was a 'No'" */
        if(pos == std::string::npos){numberofgenotyped++;}
    }
    return numberofgenotyped;
}
/************************************/
/***     Matrix Statistics        ***/
/************************************/
void MatrixStats(vector <double> &summarystats, double* output_subrelationship, int dimen,ostream& logfileloc)
{
    double meandiagonal = 0.0; double mindiag = output_subrelationship[(0*dimen)+0]; double maxdiag = output_subrelationship[(0*dimen)+0];
    double meanoffdiagonal = 0.0; double minoffdiag = output_subrelationship[(0*dimen)+1]; double maxoffdiag = output_subrelationship[(0*dimen)+1];
    int countsoffdiagonal = 0;
    for(int i = 0; i < dimen; i++)
    {
        for(int j = i; j < dimen; j++)
        {
            
            if(i == j)
            {
                meandiagonal += output_subrelationship[(i*dimen)+j];
                if(output_subrelationship[(i*dimen)+j] < mindiag){mindiag = output_subrelationship[(i*dimen)+j];}
                if(output_subrelationship[(i*dimen)+j] > maxdiag){maxdiag = output_subrelationship[(i*dimen)+j];}
            }
            if(i != j)
            {
                meanoffdiagonal += output_subrelationship[(i*dimen)+j]; countsoffdiagonal++;
                if(output_subrelationship[(i*dimen)+j] < minoffdiag){minoffdiag = output_subrelationship[(i*dimen)+j];}
                if(output_subrelationship[(i*dimen)+j] > maxoffdiag){maxoffdiag = output_subrelationship[(i*dimen)+j];}
            }
        }
    }
    meandiagonal /= double(dimen); meanoffdiagonal /= double(countsoffdiagonal);
    summarystats[0] = meandiagonal; summarystats[1] = mindiag; summarystats[2] = maxdiag;
    summarystats[3] = meanoffdiagonal; summarystats[4] = minoffdiag; summarystats[5] = maxoffdiag;
    logfileloc <<"              - Diagonal: \t" <<summarystats[0]<<"\t"<<summarystats[1]<<"\t"<<summarystats[2]<<endl;
    logfileloc <<"              - Off-diagonal: \t"<<summarystats[3]<<"\t"<<summarystats[4]<<"\t"<<summarystats[5]<<endl;
}
/************************************/
/***     Matrix Statistics        ***/
/************************************/
void MatrixCorrStats(vector <double> &summarystats, double* _grm_mkl, double* A22relationship, int dimen,ostream& logfileloc)
{
    double diag_meanA22 = 0.0; double diag_meanG = 0.0; /* Diagaonl */
    double offdiag_meanA22 = 0.0; double offdiag_meanG = 0.0; int offdiagonalcount = 0; /* Off_Diagonal */
    /* Generate Mean */
    for(int i = 0; i < dimen; i++)
    {
        for(int j = i; j < dimen; j++)
        {
            if(i == j){diag_meanA22 += A22relationship[(i*dimen)+j]; diag_meanG += _grm_mkl[(i*dimen)+j];}
            if(i != j){offdiag_meanA22 += A22relationship[(i*dimen)+j]; offdiag_meanG += _grm_mkl[(i*dimen)+j]; offdiagonalcount++;}
        }
    }
    diag_meanA22 /= dimen; diag_meanG /= dimen;
    offdiag_meanA22 /= offdiagonalcount; offdiag_meanG /= offdiagonalcount;
    /* Calculate Correlation */
    double diag_numer = 0.0; double diag_varG = 0.0; double diag_varA22 = 0.0;
    double offdiag_numer = 0.0; double offdiag_varG = 0.0; double offdiag_varA22 = 0.0;
    for(int i = 0; i < dimen; i++)
    {
        for(int j = i; j < dimen; j++)
        {
            if(i == j)
            {
                diag_numer += ((A22relationship[(i*dimen)+j]-diag_meanA22)*(_grm_mkl[(i*dimen)+j]-diag_meanG));
                diag_varG += (_grm_mkl[(i*dimen)+j]-diag_meanG) * (_grm_mkl[(i*dimen)+j]-diag_meanG);
                diag_varA22 += (A22relationship[(i*dimen)+j]-diag_meanA22)*(A22relationship[(i*dimen)+j]-diag_meanA22);
            }
            if(i != j)
            {
                offdiag_numer += ((A22relationship[(i*dimen)+j]-offdiag_meanA22)*(_grm_mkl[(i*dimen)+j]-offdiag_meanG));
                offdiag_varG += (_grm_mkl[(i*dimen)+j]-offdiag_meanG)*(_grm_mkl[(i*dimen)+j]-offdiag_meanG);
                offdiag_varA22 += (A22relationship[(i*dimen)+j]-offdiag_meanA22)*(A22relationship[(i*dimen)+j]-offdiag_meanA22);
            }
        }
    }
    logfileloc << "            - Correlation of elements of G and A22: " << endl;
    logfileloc <<"              - Diagonal: " << diag_numer / double(sqrt(diag_varG*diag_varA22)) << "." << endl;
    logfileloc <<"              - Off-diagonal: " << offdiag_numer / double(sqrt(offdiag_varG*offdiag_varA22)) << "." << endl;
}
/************************************/
/***       Figure Out Training    ***/
/************************************/
void update_animgreat(parameters &SimParameters,vector <Animal> &population, vector <int> &trainanimals,string Pheno_Pedigree_File,int Gen)
{
    vector < int > fullanim; vector < int > fullsire; vector < int > fulldam;
    /*******************************/
    /* First read in full pedigree */
    /*******************************/
    vector <string> numbers; string line;                                               /* Import file and put each row into a vector */
    ifstream infile1;
    infile1.open(Pheno_Pedigree_File.c_str());
    if(infile1.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile1,line)){numbers.push_back(line);}     /* Stores in vector and each new line push back to next space */
    for(int i = 0; i < numbers.size(); i++)
    {
        size_t pos = numbers[i].find(" ",0); fullanim.push_back(atoi(numbers[i].substr(0,pos).c_str())); numbers[i].erase(0, pos + 1); /* Grab Animal */
        pos = numbers[i].find(" ",0); fullsire.push_back(atoi(numbers[i].substr(0,pos).c_str())); numbers[i].erase(0, pos + 1); /* Grab Sire */
        pos = numbers[i].find(" ",0); fulldam.push_back(atoi(numbers[i].substr(0,pos).c_str())); numbers[i].erase(0, pos + 1); /* Grab Dam */
        //cout << fullanim[fullanim.size()-1] << " " << fullsire[fullanim.size()-1] << " " << fulldam[fullanim.size()-1] << endl;
    }
    numbers.clear();
    //cout << fullanim.size() << " " << fullsire.size() << " " << fulldam.size() << endl;
    //cout << SimParameters.getreferencegenblup() << endl;
    /***********************************************/
    /* Grab the animals that are currently progeny */
    /***********************************************/
    vector < int > tempparents;                         /* used to check parentage */
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            trainanimals.push_back(population[i].getID());
            tempparents.push_back(population[i].getSire()); tempparents.push_back(population[i].getDam());
        }
    }
    int traingenback = 0;
    while(traingenback < SimParameters.getreferencegenblup())
    {
        //cout << trainanimals.size() << endl;
        /* First take all parents and add to trainanimals */
        //for(int i = 0; i < tempparents.size(); i++){cout << tempparents[i] << " ";}
        //cout << endl << endl;
        sort(tempparents.begin(),tempparents.end());
        tempparents.erase(unique(tempparents.begin(),tempparents.end()),tempparents.end());
        /* add current parents to training */
        for(int i = 0; i < tempparents.size(); i++){trainanimals.push_back(tempparents[i]);}
        traingenback++;
        //cout << trainanimals.size() << endl;
        //for(int i = 0; i < tempparents.size(); i++){cout << tempparents[i] << " ";}
        //cout << endl << endl;
        /* Find parents of current parents in tempparents */
        vector < int > temporarynewparents;
        for(int i = 0; i < tempparents.size(); i++)
        {
            if(fullanim[tempparents[i]-1]!= tempparents[i]){cout << "Line 224" << endl; exit (EXIT_FAILURE);}
            if(fullsire[tempparents[i]-1] != 0){temporarynewparents.push_back(fullsire[tempparents[i]-1]);}
            if(fulldam[tempparents[i]-1] != 0){temporarynewparents.push_back(fulldam[tempparents[i]-1]);}
            //cout << fullanim[tempparents[i]-1] << " " << fullsire[tempparents[i]-1] << " " << fulldam[tempparents[i]-1] << endl;
        }
        tempparents.clear();
        //cout << tempparents.size() << endl << endl;
        //cout << temporarynewparents.size() << endl;
        //for(int i = 0; i < temporarynewparents.size(); i++){cout << temporarynewparents[i] << " ";}
        //cout << endl << endl;
        /* Sort old parents and put in tempparents */
        sort(temporarynewparents.begin(),temporarynewparents.end());
        temporarynewparents.erase(unique(temporarynewparents.begin(),temporarynewparents.end()),temporarynewparents.end());
        //cout << temporarynewparents.size() << endl;
        //for(int i = 0; i < temporarynewparents.size(); i++){cout << temporarynewparents[i] << " ";}
        //cout << endl << endl;
        for(int i = 0; i < temporarynewparents.size(); i++){tempparents.push_back(temporarynewparents[i]);}
        if(tempparents.size() == 0){break;}
        //cout << trainanimals.size() << endl;
    }
    sort(trainanimals.begin(),trainanimals.end());
    trainanimals.erase(unique(trainanimals.begin(),trainanimals.end()),trainanimals.end());
    /* Find all progeny of animals and include */
    int currentsize = trainanimals.size();
    for(int i = 0; i < currentsize; i++)
    {
        //cout << trainanimals[i] << "-" << trainanimals.size() << "+";
        for(int j = 0; j < fullsire.size(); j++)
        {
            if(fullsire[j] == trainanimals[i] || fulldam[j] == trainanimals[i]){trainanimals.push_back(fullanim[j]);}
        }
        //cout << trainanimals.size() << "\t";
        //if(i > 20) { cout << endl; exit (EXIT_FAILURE);}
    }
    //cout << trainanimals.size() << endl;
    //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << " ";}
    //cout << endl << endl;
    sort(trainanimals.begin(),trainanimals.end());
    trainanimals.erase(unique(trainanimals.begin(),trainanimals.end()),trainanimals.end());
    //cout << trainanimals.size() << endl;
    //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << " ";}
    //cout << endl << endl;
}
/************************************/
/***         Generate Amax        ***/
/************************************/
void getamax (parameters &SimParameters,vector <Animal> &population,string Master_DF_File,string Pheno_Pedigree_File,string Amax_Output)
{
    vector <int> animal; vector < int > sire; vector < int > dam;
    vector <int> generation;
    int recentgen = -5;
    vector <string> numbers; string line;                                               /* Import file and put each row into a vector */
    ifstream infile1;
    infile1.open(Master_DF_File.c_str());
    if(infile1.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile1,line)){numbers.push_back(line);}     /* Stores in vector and each new line push back to next space */
    for(int i = 0; i < numbers.size(); i++)
    {
        vector <string> lineVar;
        for(int j = 0; j < 27; j++)
        {
            if(j <= 25){size_t pos = numbers[i].find(" ",0); lineVar.push_back(numbers[i].substr(0,pos)); numbers[i].erase(0, pos + 1);}
            if(j == 26){lineVar.push_back(numbers[i]);}
        }
        animal.push_back(atoi(lineVar[0].c_str())); sire.push_back(atoi(lineVar[1].c_str())); dam.push_back(atoi(lineVar[2].c_str()));
        generation.push_back(atoi(lineVar[4].c_str()));
        if(generation[generation.size()-1] > recentgen){recentgen = generation[generation.size()-1];}
    }
    numbers.clear();
    //cout << animal.size() << " " << generation.size() << endl;
    /************************************************************************************************/
    /* Grab the animals that are currently progeny or are parent that belong to training generation */
    /************************************************************************************************/
    for(int i = 0; i < population.size(); i++)
    {
        animal.push_back(population[i].getID()); sire.push_back(population[i].getSire()); dam.push_back(population[i].getDam());
        generation.push_back(population[i].getGeneration());
        if(generation[generation.size()-1] > recentgen){recentgen = population[i].getGeneration();}
    }
    /* sort based on animal id to match up with pedigree file */
    int tempa, tempb, tempc, tempd;                     /* Bubble sort */
    for(int i = 0; i < animal.size()-1; i++)
    {
        for(int j=i+1; j < animal.size(); j++)
        {
            if(animal[i] > animal[j])
            {
                tempa = animal[i]; tempb = sire[i]; tempc = dam[i]; tempd = generation[i];
                animal[i] = animal[j]; sire[i] = sire[j]; dam[i] = dam[j]; generation[i] = generation[j];
                animal[j] = tempa; sire[j] = tempb; dam[j] = tempc; generation[j] = tempd;
            }
        }
    }
    //for(int i = 0; i < animal.size(); i++){cout << animal[i] << "-" << sire[i] << "-" << dam[i] << "-" << generation[i] << "  ";}
    //cout << endl;
    vector < double > acrossgenerationAmax(SimParameters.getGener(),0.0);
    if(recentgen == 1)
    {
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
        //for(int i = 0; i < 10; i++)
        //{
        //    for(int j = 0; j < 10; j++){cout << parent_A[(i*TotalAnimalNumber)+j] << " ";}
        //    cout << endl;
        //}
        vector < int > recentindex;
        for(int i = 0; i < animal.size(); i++){if(generation[i] == recentgen){recentindex.push_back(i);}}
        std::vector <double> Amax;
        double indAmax;
        for(int i = 0; i < animal.size(); i++)
        {
            if(generation[i] < recentgen)
            {
                indAmax = 0.0;
                for(int j = 0; j < recentindex.size(); j++)
                {
                    if(parent_A[(i*animal.size())+recentindex[j]] > indAmax){indAmax = parent_A[(i*animal.size())+recentindex[j]];}
                }
                Amax.push_back(indAmax);
            }
        }
        double meanAmax = 0.0;
        for(int i = 0; i < Amax.size(); i++){meanAmax += Amax[i];}
        meanAmax = meanAmax / double(Amax.size());
        acrossgenerationAmax[0] = meanAmax;
        delete [] parent_A;
    }
    if(recentgen > 1)
    {
        /* Dont want to build full A across all animals all at once; instead build for each generation to selection candidates */
        for(int recind = 0; recind < recentgen; recind++)
        {
            vector <int> parentid; vector <int> prevgenindex; vector <int> selecind;
            for(int i = 0; i < generation.size(); i++)
            {
                if(generation[i] == recind || generation[i] == recentgen){parentid.push_back(animal[i]);}
                if(generation[i] == recind){prevgenindex.push_back(parentid.size()-1);}
                if(generation[i] == recentgen){selecind.push_back(parentid.size()-1);}
            }
            double* temprelationship = new double[parentid.size()*parentid.size()];
            vector < double > pedinbreeding(animal.size(),0.0);
            for(int i = 0; i < (parentid.size()*parentid.size()); i++){temprelationship[i] = 0.0;}
            A22_Colleau(animal,sire,dam,parentid,pedinbreeding,temprelationship);
            //for(int i = 0; i < 15; i++)
            //{
            //    for(int j = 0; j < 15; j++){cout << temprelationship[(i*parentid.size())+j] << " ";}
            //    cout << endl;
            //}
            std::vector <double> Amax;
            double indAmax;
            for(int i = 0; i < prevgenindex.size(); i++)
            {
                indAmax = 0.0;
                for(int j = 0; j < selecind.size(); j++)
                {
                    if(temprelationship[(prevgenindex[i]*parentid.size())+selecind[j]] > indAmax)
                    {
                        indAmax = temprelationship[(prevgenindex[i]*parentid.size())+selecind[j]];
                    }
                }
                Amax.push_back(indAmax);
            }
            double meanAmax = 0.0;
            for(int i = 0; i < Amax.size(); i++){meanAmax += Amax[i];}
            meanAmax = meanAmax / double(Amax.size());
            acrossgenerationAmax[recind] = meanAmax;
            delete [] temprelationship;
        }
    }
    std::ofstream outAmax(Amax_Output.c_str(), std::ios_base::app | std::ios_base::out);
    outAmax << recentgen;
    for(int i = 0; i < acrossgenerationAmax.size(); i++)
    {
        outAmax << " " << acrossgenerationAmax[i];
    }
    outAmax<<endl;
}
/********************************************/
/***         Generate Correlations        ***/
/********************************************/
void trainrefcor(parameters &SimParameters,vector <Animal> &population,string Correlation_Output, int Gen)
{
    vector < vector < double > > variables; int n;
    string outputstring;
    /********************/
    /* First do parents */
    /********************/
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() > 1)
        {
            vector < double > temp;
            temp.push_back(population[i].getEBV()); temp.push_back(population[i].getGenotypicValue());
            temp.push_back(population[i].getBreedingValue()); temp.push_back(population[i].getDominanceDeviation());
            variables.push_back(temp);
        }
    }
    n = variables.size();
    //cout << n << endl;
    //for(int i = 0; i < variables.size(); i++)
    //{
    //    cout << i << " ";
    //    for(int j = 0; j < variables[0].size(); j++)
    //    {
    //        cout << variables[i][j] << " ";
    //    }
    //    cout << "   ---   ";
    //}
    //cout << endl << endl;
    vector < double > mean(4,0.0); double bias;
    vector < vector < double > > covar(4, vector<double>(4,0.0));
    for(int i = 0; i < variables.size(); i++)
    {
        for(int j = 0; j < variables[0].size(); j++){mean[j] += variables[i][j];}
    }
    for(int j = 0; j < variables[0].size(); j++){mean[j] /= double(n);}
    //for(int j = 0; j < variables[0].size(); j++){cout << mean[j] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < 4; i++)
    //{
    //    for(int j = 0; j < 4; j++){cout << covar[i][j] << " ";}
    //    cout << endl;
    //}
    //cout << endl;
    for(int ind = 0; ind < n; ind++)
    {
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j <= i; j++){covar[i][j] += ((variables[ind][i] - mean[i]) * (variables[ind][j] - mean[j]));}
        }
    }
    bias = covar[2][0] / double(covar[0][0]);
    for(int i = 0; i < 4; i++){covar[i][i] = sqrt(covar[i][i] / double(n-1));}
    for(int i = 0; i < 4; i++)
    {
        for(int j = i+1; j < 4; j++)
        {
            covar[i][j] = ((covar[j][i] / double(covar[i][i] * covar[j][j])) / double(n-1));
        }
    }
    //for(int i = 0; i < 4; i++)
    //{
    //    for(int j = 0; j < 4; j++){cout << covar[i][j] << "\t" ;}
    //    cout << endl;
    //}
    //for(int i = 1; i < 4; i++){cout << covar[0][i] << endl;}
    /* Save to a string */
    stringstream s1; s1 << Gen-1; string tempGen = s1.str();
    outputstring = outputstring + tempGen;
    //cout << outputstring << endl;
    for(int i = 1; i < 4; i++)
    {
        stringstream s2; s2 << covar[0][i]; string tempvar2 = s2.str();
        outputstring = outputstring + " " + tempvar2;
        //cout << outputstring << endl;
    }
    stringstream sbiasparent; sbiasparent << bias; string tempbiasparent = sbiasparent.str();
    outputstring = outputstring + " " + tempbiasparent;
    //cout << outputstring << endl;
    /* First zero everything out */
    for(int i = 0; i < 4; i++)
    {
        mean[i] = 0.0;
        for(int j = 0; j < 4; j++){covar[i][j] = 0.0;}
    }
    for(int i = 0; i < n; i++){variables[i].clear();}
    variables.clear();
    /********************/
    /* Then do progeny */
    /********************/
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            vector < double > temp;
            temp.push_back(population[i].getEBV()); temp.push_back(population[i].getGenotypicValue());
            temp.push_back(population[i].getBreedingValue()); temp.push_back(population[i].getDominanceDeviation());
            variables.push_back(temp);
        }
    }
    n = variables.size();
    //cout << n << endl;
    //for(int i = 0; i < variables.size(); i++)
    //{
    //    cout << i << " ";
    //    for(int j = 0; j < variables[0].size(); j++)
    //    {
    //        cout << variables[i][j] << " ";
    //    }
    //    cout << endl;
    //}
    //cout << endl << endl;
    for(int i = 0; i < variables.size(); i++)
    {
        for(int j = 0; j < variables[0].size(); j++){mean[j] += variables[i][j];}
    }
    for(int j = 0; j < variables[0].size(); j++){mean[j] /= double(n);}
    //for(int j = 0; j < variables[0].size(); j++){cout << mean[j] << " ";}
    //cout << endl << endl;
    //for(int i = 0; i < 4; i++)
    //{
    //    for(int j = 0; j < 4; j++){cout << covar[i][j] << " ";}
    //    cout << endl;
    //}
    //cout << endl;
    for(int ind = 0; ind < n; ind++)
    {
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j <= i; j++){covar[i][j] += ((variables[ind][i] - mean[i]) * (variables[ind][j] - mean[j]));}
        }
    }
    bias = covar[2][0] / double(covar[0][0]);
    for(int i = 0; i < 4; i++){covar[i][i] = sqrt(covar[i][i] / double(n-1));}
    for(int i = 0; i < 4; i++)
    {
        for(int j = i+1; j < 4; j++)
        {
            covar[i][j] = ((covar[j][i] / double(covar[i][i] * covar[j][j])) / double(n-1));
        }
    }
    //cout << bias << endl;
    
    //using Eigen::MatrixXd; using Eigen::VectorXd;
    //MatrixXd X((variables.size()),2);
    //for(int i = 0; i < X.rows(); i++){X(i,0) = 1; X(i,1) = variables[i][0];}
    //VectorXd Y(X.rows());
    //for(int i = 0; i < X.rows(); i++){Y(i) = variables[i][2];}
    //MatrixXd LHSInverse(2,2); LHSInverse = (X.transpose() * X).inverse();
    //MatrixXd Solutions(2,1); Solutions = (LHSInverse * (X.transpose()*Y));
    //cout << LHSInverse << endl;
    //cout << Solutions << endl;
    
    
    /* Save to a string */
    for(int i = 1; i < 4; i++)
    {
        stringstream s2; s2 << covar[0][i]; string tempvar2 = s2.str();
        outputstring = outputstring + " " + tempvar2;
        //cout << outputstring << endl;
    }
    stringstream sbiasprogeny; sbiasprogeny << bias; string tempbiasprogeny = sbiasprogeny.str();
    outputstring = outputstring + " " + tempbiasprogeny;
    //cout << endl;
    //for(int i = 0; i < 4; i++)
    //{
    //    for(int j = 0; j < 4; j++){cout << covar[i][j] << " ";}
    //    cout << endl;
    //}
    std::ofstream outCorr(Correlation_Output.c_str(), std::ios_base::app | std::ios_base::out);
    outCorr << outputstring << endl;
}
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/********************                       BLUP Based Functions (pblup, gblup, rohblup, ssgblup)                    ********************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
void Generate_BLUP_EBV(parameters &SimParameters,vector <Animal> &population, vector <double> &estimatedsolutions, vector <double> &trueaccuracy,ostream& logfileloc,vector <int> &trainanimals,int TotalAnimalNumber, int TotalOldAnimalNumber, int Gen,string Pheno_Pedigree_File,string GenotypeStatus_path,string Pheno_GMatrix_File,double* M, float scale,string BinaryG_Matrix_File,string Binarym_Matrix_File, string Binaryp_Matrix_File, string BinaryGinv_Matrix_File,string BinaryLinv_Matrix_File,vector < hapLibrary > &haplib)
{
    /* When updating relationship matrix will use TotalAnimalNumber and OldAnimal number, but for everything else use relationshipsize */
    int relationshipsize;
    /* When using full G or P animal with 1 will be at generation 0 and any generation after that will be greater */
    int animgreatkeep;
    if(SimParameters.getGener() > SimParameters.getreferencegenblup())
    {
        /* Truncate either G or P; Figure out which animal id to start keeping genotypes */
        update_animgreat(SimParameters,population,trainanimals,Pheno_Pedigree_File,Gen);
        relationshipsize = trainanimals.size();
        logfileloc << "       - A portion of animals were removed!" << endl;
    } else {relationshipsize = TotalAnimalNumber;}
    logfileloc << "       - Size of Relationship Matrix: " << relationshipsize << " X " << relationshipsize << "." << endl;
    double scalinglambda = (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd());     /* Shrinkage Factor for MME */
    vector < double > Phenotype(relationshipsize,0.0);                  /* Vector of phenotypes */
    vector < int > animal(relationshipsize,0);                          /* Array to store Animal IDs */
    vector < tuple <int,int,double> > sprelationshipinv;                /* Sparse storage of relationship inverse */
    if(SimParameters.getGener() <= SimParameters.getreferencegenblup())
    {
        /**********************************************************/
        /* The h1, h2 and rohblup aren't dependent on frequencies */
        /**********************************************************/
        if(SimParameters.getEBV_Calc() == "h1" || SimParameters.getEBV_Calc() == "h2" || SimParameters.getEBV_Calc() == "rohblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            if(propgeno != 1)
            {
                cout << endl << "Mix of genotyped and non-genotyped animals so shouldn't be at pblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            /* Initialize first then you can update for later generations */
            if(Gen == (SimParameters.getGenfoundsel() + 1))     /* Initialize first then you can update for later generations */
            {
                newhaplotyperelationship(SimParameters,population,haplib,Pheno_GMatrix_File,animal,Phenotype,trainanimals,BinaryG_Matrix_File,logfileloc);
                if(SimParameters.getGeno_Inverse() == "recursion"){newgenomicrecursion(SimParameters,relationshipsize,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                if(SimParameters.getGeno_Inverse() == "cholesky"){newgenomiccholesky(SimParameters,relationshipsize,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
            }
            if(Gen > (SimParameters.getGenfoundsel() + 1))
            {
                updatehaplotyperelationship(SimParameters,population,haplib,Pheno_GMatrix_File,animal,Phenotype,BinaryG_Matrix_File,TotalOldAnimalNumber,TotalAnimalNumber,logfileloc);
                if(SimParameters.getGeno_Inverse() == "recursion"){updategenomicrecursion(TotalAnimalNumber,TotalOldAnimalNumber,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                if(SimParameters.getGeno_Inverse() == "cholesky"){updategenomiccholesky(TotalAnimalNumber,TotalOldAnimalNumber,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File, sprelationshipinv,logfileloc,animal);}
            }
            Inbreeding_Pedigree(population,Pheno_Pedigree_File);        /* Still need to calculate pedigree inbreeding so do it now */
        }
        /*************************************************************************************************/
        /* The gblup is dependent on frequencies (founder, observed) so updating will only work for  */
        /* founder one because will always be the same across generations. Other are dependent on Gen    */
        /*************************************************************************************************/
        if(SimParameters.getEBV_Calc() == "gblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            if(propgeno != 1)
            {
                cout << endl << "Mix of genotyped and non-genotyped animals so shouldn't be at pblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            /* Can be utilized to updated since same freqeuncy used across all generations */
            if(SimParameters.getConstructGFreq() == "founder")
            {
                logfileloc << "       - G constructed based on founder allele frequencies." << endl;
                if(Gen == (SimParameters.getGenfoundsel() + 1))     /* Initialize first then you can update for later generations */
                {
                    newgenomicrelationship(SimParameters,population,Pheno_GMatrix_File,animal,Phenotype,M,scale,trainanimals,BinaryG_Matrix_File,logfileloc);
                    if(SimParameters.getGeno_Inverse() == "recursion"){newgenomicrecursion(SimParameters,relationshipsize,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                    if(SimParameters.getGeno_Inverse() == "cholesky"){newgenomiccholesky(SimParameters,relationshipsize,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                }
                if(Gen > (SimParameters.getGenfoundsel() + 1))
                {
                    updategenomicrelationship(SimParameters,population,Pheno_GMatrix_File,animal,Phenotype,M,scale,BinaryG_Matrix_File,TotalOldAnimalNumber,TotalAnimalNumber,logfileloc);
                    if(SimParameters.getGeno_Inverse() == "recursion"){updategenomicrecursion(TotalAnimalNumber,TotalOldAnimalNumber,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                    if(SimParameters.getGeno_Inverse() == "cholesky"){updategenomiccholesky(TotalAnimalNumber,TotalOldAnimalNumber,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File, sprelationshipinv,logfileloc,animal);}
                }
            }
            if(SimParameters.getConstructGFreq() == "observed")
            {
                /* Can't update G matrix at any generation so always have to recreate G each generation */
                double* Mobs = new double[3*(population[0].getMarker()).size()]; /* Dimension 3 by number of markers */
                float scaleobs =  update_M_scale(SimParameters,population,Mobs);
                logfileloc << "       - G constructed based on observed (i.e. selection candidate) allele frequencies." << endl;
                //cout << scaleobs << endl;
                //for(int i = 0; i < 5; i++)
                //{
                //    for(int j = 0; j < 3; j++){cout << Mobs[(j*(population[0].getMarker()).size())+i] << " ";}
                //    cout << endl;
                //}
                newgenomicrelationship(SimParameters,population,Pheno_GMatrix_File,animal,Phenotype,M,scale,trainanimals,BinaryG_Matrix_File,logfileloc);
                delete [] Mobs;
                if(SimParameters.getGeno_Inverse() == "recursion"){newgenomicrecursion(SimParameters,relationshipsize,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                if(SimParameters.getGeno_Inverse() == "cholesky"){newgenomiccholesky(SimParameters,relationshipsize,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
            }
            Inbreeding_Pedigree(population,Pheno_Pedigree_File);        /* Still need to calculate pedigree inbreeding so do it now */
        }
        if(SimParameters.getEBV_Calc() == "ssgblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            /* Genotyped proportion has to be either 0 (i.e. no genotypes) or any value below 1.0 (i.e. all genotyped) */
            /* It can't be all genotyped because that would be gblup, rohblup or a bayes option */
            if(propgeno == 1.0)
            {
                cout << endl << "All genotyped animals so shouldn't be at ssgblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            if(propgeno == 0.0)
            {
                logfileloc << "       - Begin Constructing Sparse A-Inverse (i.e. No animals are genotyped)." << endl;
                time_t start = time(0); vector <double> pedigreeinb(animal.size());
                Setup_SpmeuwissenluoAinv(Pheno_Pedigree_File,animal,Phenotype,trainanimals,sprelationshipinv,pedigreeinb,logfileloc);
                updateinbreeding(population,animal,pedigreeinb);            /* Update inbreeding for animal */
                time_t end = time(0);
                logfileloc<< "       - Finished Constructing Sparse Ainverse.\n"<<"               - Took: "<<difftime(end,start)<<" seconds."<<endl;
            }
            if(propgeno > 0.0 && propgeno < 1.0)
            {
                logfileloc << "       - Begin Constructing H Inverse." << endl;
                logfileloc << "            - Proportion of population genotyped: " << propgeno << "." << endl;
                time_t start = time(0);
                H_inverse_function(SimParameters,population,Pheno_Pedigree_File,Pheno_GMatrix_File,GenotypeStatus_path,animal,Phenotype,trainanimals,sprelationshipinv,logfileloc);
                Inbreeding_Pedigree(population,Pheno_Pedigree_File);        /* Still need to calculate pedigree inbreeding so do it now */
                time_t end = time(0);
                logfileloc << "       - Finished Constructing Hinv.\n"<<"               - Took: "<<difftime(end,start)<<" seconds."<<endl;
            }
        }
        if(SimParameters.getEBV_Calc() == "pblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            if(propgeno != 0)
            {
                cout << endl << "Mix of genotyped and non-genotyped animals so shouldn't be at pblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            logfileloc << "       - Begin Constructing Sparse A-Inverse." << endl;
            time_t start = time(0); vector <double> pedigreeinb(animal.size());
            Setup_SpmeuwissenluoAinv(Pheno_Pedigree_File,animal,Phenotype,trainanimals,sprelationshipinv,pedigreeinb,logfileloc);
            updateinbreeding(population,animal,pedigreeinb);            /* Update inbreeding for animal */
            time_t end = time(0);
            logfileloc<< "       - Finished Constructing Sparse Ainverse.\n"<<"               - Took: "<<difftime(end,start)<<" seconds."<<endl;
        }
    }
    if(SimParameters.getGener() > SimParameters.getreferencegenblup())
    {
        if(SimParameters.getEBV_Calc() == "h1" || SimParameters.getEBV_Calc() == "h2" || SimParameters.getEBV_Calc() == "rohblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            if(propgeno != 1)
            {
                cout << endl << "Mix of genotyped and non-genotyped animals so shouldn't be at pblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            newhaplotyperelationship(SimParameters,population,haplib,Pheno_GMatrix_File,animal,Phenotype,trainanimals,BinaryG_Matrix_File,logfileloc);
            if(SimParameters.getGeno_Inverse() == "recursion"){newgenomicrecursion(SimParameters,relationshipsize,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
            if(SimParameters.getGeno_Inverse() == "cholesky"){newgenomiccholesky(SimParameters,relationshipsize,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
        }
        if(SimParameters.getEBV_Calc() == "gblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            if(propgeno != 1)
            {
                cout << endl << "Mix of genotyped and non-genotyped animals so shouldn't be at pblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            if(SimParameters.getConstructGFreq() == "founder")
            {
                newgenomicrelationship(SimParameters,population,Pheno_GMatrix_File,animal,Phenotype,M,scale,trainanimals,BinaryG_Matrix_File,logfileloc);
                if(SimParameters.getGeno_Inverse() == "recursion"){newgenomicrecursion(SimParameters,relationshipsize,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                if(SimParameters.getGeno_Inverse() == "cholesky"){newgenomiccholesky(SimParameters,relationshipsize,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
            }
            if(SimParameters.getConstructGFreq() == "observed")
            {
                /* Can't update G matrix at any generation so always have to recreate G each generation */
                double* Mobs = new double[3*(population[0].getMarker()).size()]; /* Dimension 3 by number of markers */
                float scaleobs =  update_M_scale(SimParameters,population,Mobs);
                logfileloc << "       - G constructed based on observed (i.e. selection candidate) allele frequencies." << endl;
                //cout << scaleobs << endl;
                //for(int i = 0; i < 5; i++)
                //{
                //    for(int j = 0; j < 3; j++){cout << Mobs[(j*(population[0].getMarker()).size())+i] << " ";}
                //    cout << endl;
                //}
                newgenomicrelationship(SimParameters,population,Pheno_GMatrix_File,animal,Phenotype,M,scale,trainanimals,BinaryG_Matrix_File,logfileloc);
                delete [] Mobs;
                if(SimParameters.getGeno_Inverse() == "recursion"){newgenomicrecursion(SimParameters,relationshipsize,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
                if(SimParameters.getGeno_Inverse() == "cholesky"){newgenomiccholesky(SimParameters,relationshipsize,BinaryG_Matrix_File,BinaryLinv_Matrix_File,BinaryGinv_Matrix_File,sprelationshipinv,logfileloc,animal);}
            }
        }
        if(SimParameters.getEBV_Calc() == "ssgblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            /* Genotyped proportion has to be either 0 (i.e. no genotypes) or any value below 1.0 (i.e. all genotyped) */
            /* It can't be all genotyped because that would be gblup, rohblup or a bayes option */
            if(propgeno == 1.0)
            {
                cout << endl << "All genotyped animals so shouldn't be at ssgblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            if(propgeno == 0.0)
            {
                logfileloc << "       - Begin Constructing Sparse A-Inverse (i.e. No animals are genotyped)." << endl;
                time_t start = time(0); vector <double> pedigreeinb(animal.size());
                Setup_SpmeuwissenluoAinv(Pheno_Pedigree_File,animal,Phenotype,trainanimals,sprelationshipinv,pedigreeinb,logfileloc);
                updateinbreeding(population,animal,pedigreeinb);            /* Update inbreeding for animal */
                time_t end = time(0);
                logfileloc<< "       - Finished Constructing Sparse Ainverse.\n"<<"               - Took: "<<difftime(end,start)<<" seconds."<<endl;
            }
            if(propgeno > 0.0 && propgeno < 1.0)
            {
                logfileloc << "       - Begin Constructing H Inverse." << endl;
                logfileloc << "            - Proportion of population genotyped: " << propgeno << "." << endl;
                time_t start = time(0);
                H_inverse_function(SimParameters,population,Pheno_Pedigree_File,Pheno_GMatrix_File,GenotypeStatus_path,animal,Phenotype,trainanimals,sprelationshipinv,logfileloc);
                Inbreeding_Pedigree(population,Pheno_Pedigree_File);        /* Still need to calculate pedigree inbreeding so do it now */
                time_t end = time(0);
                logfileloc << "       - Finished Constructing Hinv.\n"<<"               - Took: "<<difftime(end,start)<<" seconds."<<endl;
            }
        }
        if(SimParameters.getEBV_Calc() == "pblup")
        {
            double propgeno = proportiongenotyped(SimParameters,GenotypeStatus_path);
            if(propgeno != 0)
            {
                cout << endl << "Mix of genotyped and non-genotyped animals so shouldn't be at pblup option ebv estimation" << endl;
                exit (EXIT_FAILURE);
            }
            logfileloc << "       - Begin Constructing Sparse A-Inverse." << endl;
            time_t start = time(0); vector <double> pedigreeinb(animal.size());
            Setup_SpmeuwissenluoAinv(Pheno_Pedigree_File,animal,Phenotype,trainanimals,sprelationshipinv,pedigreeinb,logfileloc);
            /* Calculate inbreeding with full pedigree */
            Inbreeding_Pedigree(population,Pheno_Pedigree_File);        /* Still need to calculate pedigree inbreeding so do it now */
            time_t end = time(0);
            logfileloc<< "       - Finished Constructing Sparse Ainverse.\n"<<"               - Took: "<<difftime(end,start)<<" seconds."<<endl;
        }
    }
    logfileloc << "       - Begin Solving for equations using " << SimParameters.getSolver() << " method." << endl;
    Setup_blup(SimParameters,population,sprelationshipinv,Phenotype,animal,scalinglambda,estimatedsolutions,trueaccuracy,logfileloc);
}
/****************************************************************/
/* Feeds into Meuwissen_Luo Ainverse assume animals go from 1:n */
/****************************************************************/
void Setup_SpmeuwissenluoAinv(string Pheno_Pedigree_File, vector <int> &animal, vector <double> &Phenotype,  vector <int> &trainanimals,vector < tuple <int,int,double> > &sprelationshipinv,vector <double> &pedigreeinb, ostream& logfileloc)
{
    vector <int> sire(animal.size(),0); vector <int> dam(animal.size(),0);
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    int linenumber = 0; int tempanim; int tempanimindex = 0; int indexintrainanim = 0;  /* Counter to determine where at in pedigree index's */
    string line;
    ifstream infile2;
    infile2.open(Pheno_Pedigree_File.c_str());                                                  /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line))
    {
        size_t pos = line.find(" ", 0); tempanim = (std::stoi(line.substr(0,pos)));
        if(trainanimals.size() == 0)
        {
            animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
            pos = line.find(" ",0); sire[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);             /* Grab Sire ID */
            //if(sire[tempanimindex] < animgreatkeep){sire[tempanimindex] = 0;}
            pos = line.find(" ",0); dam[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);              /* Grab Dam ID */
            //if(dam[tempanimindex] < animgreatkeep){dam[tempanimindex] = 0;}
            Phenotype[tempanimindex] = stod(line); tempanimindex++;                                                     /* Grab Phenotype */
        } else {
            if(tempanim == trainanimals[indexintrainanim])
            {
                animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
                pos = line.find(" ",0); sire[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);             /* Grab Sire ID */
                //if(sire[tempanimindex] < animgreatkeep){sire[tempanimindex] = 0;}
                pos = line.find(" ",0); dam[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);              /* Grab Dam ID */
                //if(dam[tempanimindex] < animgreatkeep){dam[tempanimindex] = 0;}
                Phenotype[tempanimindex] = stod(line); tempanimindex++;                                                     /* Grab Phenotype */
                indexintrainanim++;
            }
        }
        linenumber++;
    }
    /* if trainanimals size greater than 0 than truncation occured need to renumber animals to go from 1 to n */
    if(trainanimals.size() > 0 )
    {
        /* First number may not begin with 1 therefore need to renumber animalIDs; Will still be in correct order */
        /* First check to make sure numbers go from 1 to n */
        string needtorenumber = "NO";
        for(int i = 0; i < animal.size(); i++){if(animal[i] != i+1){needtorenumber = "YES"; i = animal.size();}}
        if(needtorenumber == "YES")
        {
            vector < int > renum_animal(animal.size(),0); vector < int > renum_sire(animal.size(),0); vector < int > renum_dam(animal.size(),0);
            for(int i = 0; i < animal.size(); i++)
            {
                renum_animal[i] = i+1;
                int temp = animal[i];
                for(int j = 0; j < animal.size(); j++)
                {
                    /* change it if sire or dam */
                    if(temp == sire[j]){renum_sire[j] = renum_animal[i];}
                    if(temp == dam[j]){renum_dam[j] = renum_animal[i];}
                }
            }
            //for(int i = 0; i < renum_animal.size(); i++){cout<<renum_animal[i]<<" "<<renum_sire[i]<<" "<<renum_dam[i] << "\t";}
            //cout << endl;
            Meuwissen_Luo_Ainv_Sparse(renum_animal,renum_sire,renum_dam,sprelationshipinv,pedigreeinb);
        }
        if(needtorenumber == "NO"){Meuwissen_Luo_Ainv_Sparse(animal,sire,dam,sprelationshipinv,pedigreeinb);}
    }
    /* No truncation occured use all animals */
    if(trainanimals.size() == 0){Meuwissen_Luo_Ainv_Sparse(animal,sire,dam,sprelationshipinv,pedigreeinb);}
}
/*********************************************************/
/* Setup sparse pblup, gblup or rohblup representation   */
/*********************************************************/
void Setup_blup(parameters &SimParameters,vector <Animal> &population,vector < tuple <int,int,double> > &sprelationshipinv,vector <double> &Phenotype,vector <int> &animal,double scalinglambda,vector <double> &estimatedsolutions, vector <double> &trueaccuracy,ostream& logfileloc)
{
    vector < int > hasphenotype(animal.size(),1);       /* Initialize as everyone animal has a phenotype */
    if(SimParameters.getAtSelectionHas() != "phenotype")    /* Some animals may not have phenotypes information */
    {
        phenotypestatus(SimParameters,population,animal,hasphenotype);
    }
    int LHSsize = animal.size() + 1;
    if(estimatedsolutions.size() == 0)
    {
        for(int i = 0; i < LHSsize; i++)
        {
            estimatedsolutions.push_back(0.0);
            if(i > 0){trueaccuracy.push_back(0.0);}
        }
    }
    /***************************/
    /* Setup LHS in Tuple Form */
    /***************************/
    vector < tuple <int,int,double> > lhs_sparse; double tempvalue;
    double interc = 0;
    for(int i = 0; i < animal.size(); i++){interc += hasphenotype[i];}
    lhs_sparse.push_back(std::make_tuple(0,0,interc));                                                  /* Fill intercept (X'X) */
    for(int i = 1; i < LHSsize; i++){lhs_sparse.push_back(std::make_tuple(0,i,hasphenotype[i-1]));}     /* Fill intercept-animal (X'Z) */
    for(int i = 1; i < LHSsize; i++){lhs_sparse.push_back(std::make_tuple(i,0,hasphenotype[i-1]));}     /* Fill intercept-animal (Z'X) */
    for(int i = 1; i < LHSsize; i++){lhs_sparse.push_back(std::make_tuple(i,i,hasphenotype[i-1]));}     /* Fill intercept-animal (Z'Z) */
    for(int i = 0; i < sprelationshipinv.size(); i++)                                   /* Fill Relationshipinv*alpha */
    {
        tempvalue = get<2>(sprelationshipinv[i]) * double(scalinglambda);
        lhs_sparse.push_back(std::make_tuple((get<0>(sprelationshipinv[i])+1),(get<1>(sprelationshipinv[i])+1),tempvalue));
    }
    /***************************************/
    /* Setup RHS as a vector in Tuple Form */
    /***************************************/
    vector < double > rhs_sparse (LHSsize,0.0);
    for(int i = 0; i < animal.size(); i++)                      /* row 1 of RHS is sum of phenotypic observations */
    {
        if(hasphenotype[i] != 0){rhs_sparse[0] += Phenotype[i];}
    }
    for(int i = 0; i < animal.size(); i++)                      /* Copy animals with phenotype to RHS */
    {
        if(hasphenotype[i] != 0){rhs_sparse[i+1] = Phenotype[i];}
    }
    logfileloc << "           - RHS created, Dimension (" << LHSsize << " X " << 1 << ")." << endl;
    logfileloc << "           - LHS created, Dimension (" << LHSsize << " X " << LHSsize << ")." << endl;
    if(SimParameters.getSolver() == "direct")                                                   /* Solve equations using direct inversion */
    {
        logfileloc << "           - Starting " << SimParameters.getSolver() << "." << endl;
        time_t start = time(0);
        direct_solversparse(SimParameters,lhs_sparse,rhs_sparse,estimatedsolutions,trueaccuracy,LHSsize);
        time_t end = time(0);
        logfileloc << "       - Finished Solving Equations created." << endl;
        logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
    }
    if(SimParameters.getSolver() == "pcg")                                                      /* Solve equations using pcg */
    {
        logfileloc << "           - Starting " << SimParameters.getSolver() << "." << endl;
        time_t start = time(0);
        int* solvedatiteration = new int[1]; solvedatiteration[0] = 0;
        if(SimParameters.getEBV_Calc()=="pblup" || SimParameters.getEBV_Calc()=="ssgblup")        /* keep it sparse since a large number will be zero */
        {
            pcg_solver_sparse(lhs_sparse,rhs_sparse,estimatedsolutions,LHSsize,solvedatiteration);
        } else{pcg_solver_dense(lhs_sparse,rhs_sparse,estimatedsolutions,LHSsize,solvedatiteration);}   /* convert it back to full matrix form */
        time_t end = time(0);
        logfileloc << "           - PCG converged at iteration " << solvedatiteration[0] << "." << endl;
        logfileloc << "       - Finished Solving Equations created." << endl;
        logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
        delete[] solvedatiteration;
    }
    /* Update Animal Class with EBV's and associated Accuracy (if option is direct) */
    for(int i = 0; i < population.size(); i++)
    {
        int j = 0;                                                  /* Counter for population spot */
        while(j < animal.size())
        {
            if(population[i].getID() == animal[j])
            {
                population[i].Update_EBV(estimatedsolutions[j+1]);
                if(SimParameters.getSolver() == "direct")
                {
                    if(SimParameters.getEBV_Calc()!="gblup" && SimParameters.getConstructGFreq()!="observed" && SimParameters.getGener()==SimParameters.getreferencegenblup())
                    {
                        population[i].Update_Acc(trueaccuracy[j]);
                    }
                }
                break;
            }
            j++;
        }
    }
    //for(int i = 0; i < population.size(); i++)
    //{
    //    if(population[i].getSex() == 0)
    //    {
    //        cout<<population[i].getID()<<" "<<population[i].getProgeny()<<" "<<population[i].getEBV()<<" "<<population[i].getAcc()<<endl;
    //    }
    //}
}
/********************************************************************/
/* Figure out the status of an animal in order to set up ZZ' of LHS */
/********************************************************************/
void phenotypestatus(parameters &SimParameters,vector <Animal> &population,vector <int> &animal, vector <int> &hasphenotype)
{
    /* Selection Candidates (i.e. Progeny Don't have phenotypes */
    if(SimParameters.getAtSelectionHas() == "nophenotype")
    {
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getAge() == 1)
            {
                int currentlookloc = animal.size()-1;
                while(currentlookloc >= 0)
                {
                    if(population[i].getID() == animal[currentlookloc]){hasphenotype[currentlookloc] = 0; break;}
                    if(population[i].getID() != animal[currentlookloc]){currentlookloc--;}
                }
            }
        }
    }
    //int numberzero = 0;
    //for(int i = 0; i < hasphenotype.size(); i++){if(hasphenotype[i] == 0){numberzero++;}}
    //cout << numberzero << endl;
}

/*************************************************************************************/
/*************************************************************************************/
/********************           Relationship Functions           *********************/
/*************************************************************************************/
/*************************************************************************************/
/***************************************************************/
/* Generate genomic relationship matrix based on VanRaden 2008 */
/***************************************************************/
void VanRaden_grm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler)
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
/****************************************************/
/* Generate new haplotype based relationship matrix */
/****************************************************/
void newhaplotyperelationship(parameters &SimParameters,vector <Animal> &population, vector < hapLibrary > &haplib, string Pheno_GMatrix_File, vector <int> &animal, vector <double> &Phenotype, vector <int> trainanimals, string BinaryG_Matrix_File, ostream& logfileloc)
{
    using Eigen::MatrixXd; using Eigen::VectorXd;
    logfileloc << "           - Begin Constructing " << SimParameters.getEBV_Calc() << " Relationship Matrix." << endl;
    /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
    /* This way you don't have to repeat this step for each haplotype */
    time_t start = time(0);
    vector < vector < int > > PaternalHaplotypeIDs;
    vector < vector < int > > MaternalHaplotypeIDs;
    /* read in all animals haplotype ID's; Don't need to really worry about this getting big */
    int linenumber = 0; int indexintrainanim = 0;
    string line; int tempanim; int tempanimindex = 0;
    ifstream infile2;
    infile2.open(Pheno_GMatrix_File);
    if(infile2.fail()){cout << "Error Opening File To Make Genomic Relationship Matrix!\n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line))
    {
        size_t pos = line.find(" ", 0); tempanim = (std::stoi(line.substr(0,pos)));
        if(trainanimals.size() == 0)
        {
            animal[tempanimindex] = (std::stoi(line.substr(0,pos))); line.erase(0, pos + 1);
            pos = line.find(" ",0); Phenotype[tempanimindex] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
            pos = line.find(" ",0); line.erase(0,pos + 1); /* Do not need marker genotypes so skip */
            pos = line.find(" ",0); string PaternalHap = line.substr(0,pos); line.erase(0,pos + 1);
            string MaternalHap = line;
            vector < int > temp_pat;
            string quit = "NO";
            while(quit != "YES")
            {
                size_t pos = PaternalHap.find("_",0);
                if(pos > 0)                                                         /* hasn't reached last one yet */
                {
                    temp_pat.push_back(stoi(PaternalHap.substr(0,pos)));            /* extend column by 1 */
                    PaternalHap.erase(0, pos + 1);
                }
                if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
            }
            PaternalHaplotypeIDs.push_back(temp_pat);                               /* push back row */
            vector < int > temp_mat;
            quit = "NO";
            while(quit != "YES")
            {
                size_t pos = MaternalHap.find("_",0);
                if(pos > 0)                                                         /* hasn't reached last one yet */
                {
                    temp_mat.push_back(stoi(MaternalHap.substr(0,pos)));            /* extend column by 1 */
                    MaternalHap.erase(0, pos + 1);
                }
                if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
            }
            MaternalHaplotypeIDs.push_back(temp_mat);                               /* push back row */
            tempanimindex++;
        } else {
            if(tempanim == trainanimals[indexintrainanim])
            {
                animal[tempanimindex] = (std::stoi(line.substr(0,pos))); line.erase(0, pos + 1);
                pos = line.find(" ",0); Phenotype[tempanimindex] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
                pos = line.find(" ",0); line.erase(0,pos + 1); /* Do not need marker genotypes so skip */
                pos = line.find(" ",0); string PaternalHap = line.substr(0,pos); line.erase(0,pos + 1);
                string MaternalHap = line;
                vector < int > temp_pat;
                string quit = "NO";
                while(quit != "YES")
                {
                    size_t pos = PaternalHap.find("_",0);
                    if(pos > 0)                                                         /* hasn't reached last one yet */
                    {
                        temp_pat.push_back(stoi(PaternalHap.substr(0,pos)));            /* extend column by 1 */
                        PaternalHap.erase(0, pos + 1);
                    }
                    if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
                }
                PaternalHaplotypeIDs.push_back(temp_pat);                               /* push back row */
                vector < int > temp_mat;
                quit = "NO";
                while(quit != "YES")
                {
                    size_t pos = MaternalHap.find("_",0);
                    if(pos > 0)                                                         /* hasn't reached last one yet */
                    {
                        temp_mat.push_back(stoi(MaternalHap.substr(0,pos)));            /* extend column by 1 */
                        MaternalHap.erase(0, pos + 1);
                    }
                    if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
                }
                MaternalHaplotypeIDs.push_back(temp_mat);                               /* push back row */
                tempanimindex++; indexintrainanim++;
            }
        }
        linenumber++;
    }
    //cout << PaternalHaplotypeIDs.size() << " " << MaternalHaplotypeIDs.size() << endl;
    //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << "-" << animal[i] << "  ";}
    //cout << endl << endl;
    /* Initialize Relationship Matrix as 0.0 */
    MatrixXd Relationship(PaternalHaplotypeIDs.size(),PaternalHaplotypeIDs.size());
    Relationship = MatrixXd::Identity(PaternalHaplotypeIDs.size(),PaternalHaplotypeIDs.size());
    for(int i = 0; i < PaternalHaplotypeIDs.size(); i++){Relationship(i,i) = 0;}
    /* Animal Haplotype ID's have been put in 2-D vector to grab */
    for(int i = 0; i < haplib.size(); i++)
    {
        vector < string > haplotypes;
        /* Unstring haplotypes, seperated by "_" */
        string temphapstring = haplib[i].getHaplo();
        string quit = "NO";
        while(quit == "NO")
        {
            size_t pos = temphapstring.find("_",0);
            if(pos > 0)                                                 /* hasn't reached last one yet */
            {
                haplotypes.push_back(temphapstring.substr(0,pos));
                temphapstring.erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}
        }
        /* For each haplotype segment create an H matrix of dimenstion nhaplotypes by nhaplotypes */
        MatrixXd H(haplotypes.size(),haplotypes.size());
        for(int hap1 = 0; hap1 < haplotypes.size(); hap1++)
        {
            string temphap1 = haplotypes[hap1];
            for(int hap2 = hap1; hap2 < haplotypes.size(); hap2++)
            {
                if(SimParameters.getEBV_Calc() == "h1")
                {
                    if(hap1 == hap2){H(hap1,hap2) = 1;}
                    if(hap1 != hap2)
                    {
                        string temphap2 = haplotypes[hap2];
                        float sum=0.0;
                        for(int g = 0; g < temphap1.size(); g++){sum += abs(temphap1[g] - temphap2[g]);}
                        H(hap1,hap2) = 1 - (sum/temphap1.size());
                        H(hap2,hap1) = H(hap1,hap2);
                    }
                }
                if(SimParameters.getEBV_Calc() == "h2")
                {
                    if(hap1 == hap2){H(hap1,hap2) = 1;}
                    if(hap1 != hap2)
                    {
                        string temphap2 = haplotypes[hap2];
                        int match[temphap1.size()];                 /* matrix that has 1 to match and 0 if not match */
                        for(int g = 0; g < temphap1.size(); g++){match[g] = 1 - abs(temphap1[g] - temphap2[g]);}
                        double sumGlobal = 0;
                        double sum = 0;
                        for(int g = 0; g < temphap1.size(); g++)
                        {
                            if(match[g] < 1)
                            {
                                sumGlobal = sumGlobal + sum * sum; sum = 0;
                            } else {
                                sum = sum + 1;
                            }
                        }
                        H(hap1,hap2) = sqrt((sumGlobal + sum*sum) / (temphap1.size()* temphap1.size()));
                        H(hap2,hap1) = H(hap1,hap2);
                    }
                }
                if(SimParameters.getEBV_Calc() == "rohblup")
                {
                    if(hap1 == hap2){H(hap1,hap2) = 1;}
                    if(hap1 != hap2)
                    {
                        string temphap2 = haplotypes[hap2];
                        int match[temphap1.size()];                 /* matrix that has 1 to match and 0 if not match */
                        double sum = 0;
                        for(int g = 0; g < temphap1.size(); g++){match[g] = 1 - abs(temphap1[g] - temphap2[g]);sum += match[g];}
                        if(sum == temphap1.size()){H(hap1,hap2) = 1.0;}
                        if(sum != temphap1.size()){H(hap1,hap2) = 0.0;}
                        H(hap2,hap1) = H(hap1,hap2);
                    }
                }
            }
        }
        for(int ind1 = 0; ind1 < PaternalHaplotypeIDs.size(); ind1++)
        {
            for(int ind2 = ind1; ind2 < PaternalHaplotypeIDs.size(); ind2++)
            {
                Relationship(ind1,ind2) += (H((PaternalHaplotypeIDs[ind1][i]),(PaternalHaplotypeIDs[ind2][i])) +
                                            H((PaternalHaplotypeIDs[ind1][i]),(MaternalHaplotypeIDs[ind2][i])) +
                                            H((MaternalHaplotypeIDs[ind1][i]),(PaternalHaplotypeIDs[ind2][i])) +
                                            H((MaternalHaplotypeIDs[ind1][i]),(MaternalHaplotypeIDs[ind2][i]))) / 2;
                Relationship(ind2,ind1) = Relationship(ind1,ind2);
            } /* Finish loop across ind2 */
        } /* Finish loop across ind1 */
    } /* Loop across haplotypes */
    VectorXd den(1);                                                 /* Scale Relationship Matrix */
    den(0) = haplib.size();
    Relationship = Relationship / den(0);
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationship(i,j) << "\t";}
    //    cout << endl;
    //}
    time_t end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Matrix. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
    Eigen::writebinary(BinaryG_Matrix_File.c_str(),Relationship);   /* Output Relationship Matrix into Binary for next generation */
}
/********************************************/
/* Generate new genomic relationship matrix */
/********************************************/
void newgenomicrelationship(parameters &SimParameters, vector <Animal> &population, string Pheno_GMatrix_File,vector <int> &animal, vector <double> &Phenotype, double* M, float scale, vector <int> trainanimals, string BinaryG_Matrix_File, ostream& logfileloc)
{
    //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << " ";}
    //cout << endl << endl;
    using Eigen::MatrixXd;
    logfileloc << "           - Begin Constructing Genomic Relationship Matrix." << endl;
    time_t start = time(0);
    /* May have started a few generations later so always start by reading in genomic based file */
    vector < string > creategenorel;
    int linenumber = 0; int indexintrainanim = 0;
    string line; int tempanim; int tempanimindex = 0;
    ifstream infile2;
    infile2.open(Pheno_GMatrix_File.c_str());
    if(infile2.fail()){cout << "Error Opening File\n";}
    while(getline(infile2,line))
    {
        size_t pos = line.find(" ", 0); tempanim = (std::stoi(line.substr(0,pos)));
        if(trainanimals.size() == 0)
        {
            animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
            pos = line.find(" ",0); Phenotype[tempanimindex] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);     /* Grap Phenotype */
            pos = line.find(" ",0); creategenorel.push_back(line.substr(0,pos)); tempanimindex++;                       /* Grab Marker Genotypes */
        } else {
            if(tempanim == trainanimals[indexintrainanim])
            {
                animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
                pos = line.find(" ",0); Phenotype[tempanimindex] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);     /* Grap Phenotype */
                pos = line.find(" ",0); creategenorel.push_back(line.substr(0,pos)); tempanimindex++;                        /* Grab Marker Genotypes */
                indexintrainanim++;
            }
        }
        linenumber++;
    }
    //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << "-" << animal[i] << "  ";}
    //cout << endl << endl;
    //exit (EXIT_FAILURE);
    double *_grm_mkl = new double[creategenorel.size()*creategenorel.size()];   /* Allocate Memory for GRM */
    for(int i = 0; i < (creategenorel.size()*creategenorel.size()); i++){_grm_mkl[i] = 0.0;}
    if(SimParameters.getConstructG() == "VanRaden")
    {
        VanRaden_grm(M,creategenorel,_grm_mkl,scale);
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _grm_mkl[(i*creategenorel.size())+j] << "\t";}
    //    cout << endl;
    //}
    //cout << endl << creategenorel.size() << " " << animal.size() << " " << Phenotype.size() << endl;
    /* Output Genomic Relationship Matrix into Binary for next generation */
    MatrixXd Relationship(creategenorel.size(),creategenorel.size());
    for(int i = 0; i < creategenorel.size(); i++)
    {
        for(int j = 0; j <= i; j++){Relationship(i,j) = Relationship(j,i) = _grm_mkl[(i*creategenorel.size())+j];}
    }
    /* Output Genomic Relationship Matrix into Binary for next generation */
    Eigen::writebinary(BinaryG_Matrix_File.c_str(),Relationship);
    Relationship.resize(0,0);
    time_t end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Matrix." << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
    delete[] _grm_mkl; creategenorel.clear();
}
/**************************************/
/* Update genomic relationship matrix */
/**************************************/
void updategenomicrelationship(parameters &SimParameters, vector <Animal> &population, string Pheno_GMatrix_File,vector <int> &animal, vector <double> &Phenotype, double* M, float scale, string BinaryG_Matrix_File,int TotalOldAnimalNumber,int TotalAnimalNumber, ostream& logfileloc)
{
    logfileloc << "           - Begin Constructing Genomic Relationship Matrix." << endl;
    time_t start = time(0);
    //cout << TotalOldAnimalNumber << " " << TotalAnimalNumber << endl;
    using Eigen::MatrixXd; using Eigen::VectorXd;
    MatrixXd Relationship(TotalAnimalNumber,TotalAnimalNumber);
    MatrixXd OldRelationship(TotalOldAnimalNumber,TotalOldAnimalNumber);
    Eigen::readbinary(BinaryG_Matrix_File.c_str(),OldRelationship);
    /* Fill Relationship matrix with previously calcuated cells */
    for(int i = 0; i < TotalOldAnimalNumber; i++)
    {
        for(int j= 0; j < TotalOldAnimalNumber; j++){Relationship(i,j) = OldRelationship(i,j);}
    }
    OldRelationship.resize(0,0);
    /* Calculate relationship between old and new */
    vector < string > newanimalgeno;
    for(int ind = 0; ind < population.size(); ind++)
    {
        if(population[ind].getAge() == 1){newanimalgeno.push_back(population[ind].getMarker());}
    }
    int newanimals = newanimalgeno.size();
    //cout << newanimalgeno.size() << endl;
    /* First create Z for only new individuals that can be used for old-new and new-new */
    int n = newanimalgeno.size();
    int m = newanimalgeno[0].size();
    // Construct Z matrix n x m
    double *_geno_mkl = new double[n*m];
    /* Fill Matrix */
    int i = 0; int j = 0;
    for(int i = 0; i < n; i++)
    {
        string tempgeno = newanimalgeno[i];
        for(j = 0; j < m; j++)
        {
            int tempa = tempgeno[j] - 48;
            if(tempa == 3 || tempa == 4){tempa = 1;}
            _geno_mkl[(i*m) + j] = M[(tempa*m)+j];
        }
    }
    //cout << Phenotype.size() << endl;
    /* only need to fill in off-diagonals of new and old animals */
    /* Import file */
    int linenumber = 0;
    string line;
    ifstream infile2;
    infile2.open(Pheno_GMatrix_File.c_str());
    if(infile2.fail()){cout << "Error Opening File\n";}
    while (getline(infile2,line))
    {
        /* Grab animal id */
        size_t pos = line.find(" ", 0); animal[linenumber] = (std::stoi(line.substr(0,pos))); line.erase(0, pos + 1);
        /* Grap Phenotype */
        pos = line.find(" ",0); Phenotype[linenumber] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
        pos = line.find(" ",0); string tempgeno = line.substr(0,pos);                   /* Grab Marker Genotypes */
        /* Do not need the paternal and maternal haplotypes */
        if(linenumber < (Phenotype.size() - n))                   /* once it reaches new animals doesn't do anything to those genotypes */
        {
            double *_geno_line = new double[m];
            for(j = 0; j < m; j++)
            {
                int tempa = tempgeno[j] - 48;
                if(tempa == 3 || tempa == 4){tempa = 1;}
                _geno_line[j] = M[(tempa*m)+j];
            }
            double *_grm_mkl = new double[n];
            // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, n, m, 1.0, _geno_line, m, _geno_mkl, m, 0.0, _grm_mkl, m);
            // Once created now standardize
            for(i = 0; i < n; i++){_grm_mkl[i] /= scale;}
            int newanimalLocation = (Phenotype.size() - n);
            for(int i = 0; i < n; i++)
            {
                Relationship(linenumber,newanimalLocation) = Relationship(newanimalLocation,linenumber) = _grm_mkl[i]; newanimalLocation++;
            }
            delete[] _geno_line; delete[] _grm_mkl;
        }
        linenumber++;
    }
    /* Need to fill in relationship between new and new animals */
    double *_grm_mkl_22 = new double[newanimals*newanimals];                /* Allocate Memory for G22 */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, m, 1.0, _geno_mkl, m, _geno_mkl, m, 0.0, _grm_mkl_22, n);
    delete[] _geno_mkl;
    // Standardize relationship matrix by scalar
    #pragma omp parallel for private(j)
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++){_grm_mkl_22[(i*n)+j] /= scale;}
    }
    int newanimalLocation_i = TotalOldAnimalNumber;
    for(int i = 0; i < newanimals; i++)
    {
        int newanimalLocation_j = TotalOldAnimalNumber;
        for(int j = 0; j < newanimals; j++){Relationship(newanimalLocation_i,newanimalLocation_j) = _grm_mkl_22[(i*newanimals)+j]; newanimalLocation_j++;}
        newanimalLocation_i++;
    }
    delete [] _grm_mkl_22;
    time_t end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Matrix. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
    Eigen::writebinary(BinaryG_Matrix_File.c_str(),Relationship);  /* Output Genomic Relationship Matrix into Binary for */
    //fstream check; check.open("CHECK", std::fstream::out | std::fstream::trunc); check.close();
    //std::ofstream outputcheck("CHECK", std::ios_base::app | std::ios_base::out);
    //for(int i = 0; i < TotalAnimalNumber; i++)
    //{
    //    for(int j = 0; j < TotalAnimalNumber; j++){outputcheck << Relationship(i,j) << " ";}
    //    outputcheck << endl << endl;
    //}
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _grm_mkl_22[(i*newanimals)+j] << "\t";}
    //    cout << endl;
    //}
    
}
/**********************************************/
/* Update haplotype based relationship matrix */
/**********************************************/
void updatehaplotyperelationship(parameters &SimParameters,vector <Animal> &population, vector < hapLibrary > &haplib, string Pheno_GMatrix_File, vector <int> &animal, vector <double> &Phenotype,string BinaryG_Matrix_File,int TotalOldAnimalNumber,int TotalAnimalNumber,ostream& logfileloc)
{
    using Eigen::MatrixXd; using Eigen::VectorXd;
    logfileloc << "           - Begin Constructing " << SimParameters.getEBV_Calc() << " Relationship Matrix." << endl;
    /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
    /* This way you don't have to repeat this step for each haplotype */
    time_t start = time(0);
    vector < vector < int > > PaternalHaplotypeIDs;
    vector < vector < int > > MaternalHaplotypeIDs;
    /* read in all animals haplotype ID's; Don't need to really worry about this getting big */
    int linenumber = 0; string line; ifstream infile2;
    infile2.open(Pheno_GMatrix_File);
    if(infile2.fail()){cout << "Error Opening File To Make Genomic Relationship Matrix!\n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line))
    {
        size_t pos = line.find(" ", 0); animal[linenumber] = (std::stoi(line.substr(0,pos))); line.erase(0, pos + 1);
        pos = line.find(" ",0); Phenotype[linenumber] = (std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
        pos = line.find(" ",0); line.erase(0,pos + 1); /* Do not need marker genotypes so skip */
        pos = line.find(" ",0); string PaternalHap = line.substr(0,pos); line.erase(0,pos + 1);
        string MaternalHap = line;
        vector < int > temp_pat;
        string quit = "NO";
        while(quit != "YES")
        {
            size_t pos = PaternalHap.find("_",0);
            if(pos > 0)                                                         /* hasn't reached last one yet */
            {
                temp_pat.push_back(stoi(PaternalHap.substr(0,pos)));            /* extend column by 1 */
                PaternalHap.erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
        }
        PaternalHaplotypeIDs.push_back(temp_pat);                               /* push back row */
        vector < int > temp_mat;
        quit = "NO";
        while(quit != "YES")
        {
            size_t pos = MaternalHap.find("_",0);
            if(pos > 0)                                                         /* hasn't reached last one yet */
            {
                temp_mat.push_back(stoi(MaternalHap.substr(0,pos)));            /* extend column by 1 */
                MaternalHap.erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
        }
        MaternalHaplotypeIDs.push_back(temp_mat);                               /* push back row */
        linenumber++;
    }
    /* Initialize Relationship Matrix as 0.0 */
    MatrixXd Relationship(PaternalHaplotypeIDs.size(),PaternalHaplotypeIDs.size());     /* GRM calculated previously */
    Relationship = MatrixXd::Identity(PaternalHaplotypeIDs.size(),PaternalHaplotypeIDs.size());
    for(int i = 0; i < PaternalHaplotypeIDs.size(); i++){Relationship(i,i) = 0;}
    /* Grab old relationship Matrix */
    MatrixXd OldRelationship(TotalOldAnimalNumber,TotalOldAnimalNumber);        /* Used to store old animal G */
    Eigen::readbinary(BinaryG_Matrix_File.c_str(),OldRelationship);            /* Read in old G */
    /* Animal Haplotype ID's have been put in 2-D vector to grab */
    for(int i = 0; i < haplib.size(); i++)
    {
        vector < string > haplotypes;
        /* Unstring haplotypes, seperated by "_" */
        string temphapstring = haplib[i].getHaplo();
        string quit = "NO";
        while(quit == "NO")
        {
            size_t pos = temphapstring.find("_",0);
            if(pos > 0)                                                 /* hasn't reached last one yet */
            {
                haplotypes.push_back(temphapstring.substr(0,pos));
                temphapstring.erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}                 /* has reached last one so now save last one and kill while loop */
        }
        /* For each haplotype segment create an H matrix of dimenstion nhaplotypes by nhaplotypes */
        MatrixXd H(haplotypes.size(),haplotypes.size());
        for(int hap1 = 0; hap1 < haplotypes.size(); hap1++)
        {
            string temphap1 = haplotypes[hap1];
            for(int hap2 = hap1; hap2 < haplotypes.size(); hap2++)
            {
                if(SimParameters.getEBV_Calc() == "h1")
                {
                    if(hap1 == hap2){H(hap1,hap2) = 1;}
                    if(hap1 != hap2)
                    {
                        string temphap2 = haplotypes[hap2];
                        float sum=0.0;
                        for(int g = 0; g < temphap1.size(); g++){sum += abs(temphap1[g] - temphap2[g]);}
                        H(hap1,hap2) = 1 - (sum/temphap1.size());
                        H(hap2,hap1) = H(hap1,hap2);
                    }
                }
                if(SimParameters.getEBV_Calc() == "h2")
                {
                    if(hap1 == hap2){H(hap1,hap2) = 1;}
                    if(hap1 != hap2)
                    {
                        string temphap2 = haplotypes[hap2];
                        int match[temphap1.size()];                 /* matrix that has 1 to match and 0 if not match */
                        for(int g = 0; g < temphap1.size(); g++){match[g] = 1 - abs(temphap1[g] - temphap2[g]);}
                        double sumGlobal = 0;
                        double sum = 0;
                        for(int g = 0; g < temphap1.size(); g++)
                        {
                            if(match[g] < 1)
                            {
                                sumGlobal = sumGlobal + sum * sum; sum = 0;
                            } else {
                                sum = sum + 1;
                            }
                        }
                        H(hap1,hap2) = sqrt((sumGlobal + sum*sum) / (temphap1.size()* temphap1.size()));
                        H(hap2,hap1) = H(hap1,hap2);
                    }
                }
                if(SimParameters.getEBV_Calc() == "rohblup")
                {
                    if(hap1 == hap2){H(hap1,hap2) = 1;}
                    if(hap1 != hap2)
                    {
                        string temphap2 = haplotypes[hap2];
                        int match[temphap1.size()];                 /* matrix that has 1 to match and 0 if not match */
                        double sum = 0;
                        for(int g = 0; g < temphap1.size(); g++){match[g] = 1 - abs(temphap1[g] - temphap2[g]); sum += match[g];}
                        if(sum == temphap1.size()){H(hap1,hap2) = 1.0;}
                        if(sum != temphap1.size()){H(hap1,hap2) = 0.0;}
                        H(hap2,hap1) = H(hap1,hap2);
                    }
                }
            }
        }
        for(int ind1 = 0; ind1 < TotalAnimalNumber; ind1++)
        {
            for(int ind2 = ind1; ind2 < TotalAnimalNumber; ind2++)
            {
                if(ind1 < TotalOldAnimalNumber && ind2 < TotalOldAnimalNumber)              /* Old animal don't need to figure out */
                {
                    Relationship(ind1,ind2) = OldRelationship(ind1,ind2);
                    Relationship(ind2,ind1) = Relationship(ind1,ind2);
                }
                if(ind1 >= TotalOldAnimalNumber || ind2 >= TotalOldAnimalNumber)            /* Relationship with new animal */
                {
                    Relationship(ind1,ind2) += (H((PaternalHaplotypeIDs[ind1][i]),(PaternalHaplotypeIDs[ind2][i])) +
                                                H((PaternalHaplotypeIDs[ind1][i]),(MaternalHaplotypeIDs[ind2][i])) +
                                                H((MaternalHaplotypeIDs[ind1][i]),(PaternalHaplotypeIDs[ind2][i])) +
                                                H((MaternalHaplotypeIDs[ind1][i]),(MaternalHaplotypeIDs[ind2][i]))) / 2;
                    Relationship(ind2,ind1) = Relationship(ind1,ind2);
                    if(i == haplib.size() - 1){Relationship(ind1,ind2) = Relationship(ind1,ind2) / double(haplib.size());}
                }
            } /* Finish loop across ind2 */
        } /* Finish loop across ind1 */
    } /* Loop across haplotypes */
    time_t  end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Matrix. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
    Eigen::writebinary(BinaryG_Matrix_File.c_str(),Relationship);  /* Output Relationship Matrix into Binary */
}
/*******************************************************/
/* Generate subset of relationship matrix for Hinverse */
/*******************************************************/
void A22_Colleau(vector <int> const &animal,vector <int> const &sire,vector <int> const &dam,vector <int> const &parentid, vector <double> const &pedinbreeding, double* output_subrelationship)
{
    int nGen = parentid.size(); int n = pedinbreeding.size();
    vector < double > w(n,0.0); vector < double > v(n,0.0);
    /* temporary variables */
    int s, d; double di, initialadd, tmp;
    for(int i = 0; i < nGen; i++)
    {
        //cout << parentid[i] << endl << endl;
        vector < double > q(n,0.0);
        for(int j = 0; j < n; j++){v[j] = 0;}
        v[parentid[i]-1] =  1.0;
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
                di = ((initialadd+2)/double(4)) - (0.25*(pedinbreeding[s-1]+pedinbreeding[d-1]));
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
                di = ((initialadd+2)/double(4)) - (0.25*(pedinbreeding[s-1]+0));
            }
            if(s == 0 && d != 0)
            {
                initialadd = 0;
                if(sire[j] == 0){initialadd += 1;}
                if(dam[j] == 0){initialadd += 1;}
                di = ((initialadd+2)/double(4)) - (0.25*(0+pedinbreeding[d-1]));
            }
            tmp = 0.0;
            if(s != 0){tmp = tmp + w[s-1];}
            if(d != 0){tmp = tmp + w[d-1];}
            w[j] = 0.5 *tmp;
            w[j] = w[j] + (di * q[j]);
        }
        //for(int j = 0; j < n; j++){cout << w[j] << " ";}
        //cout << endl << endl;
        for(int j = 0; j < parentid.size(); j++){output_subrelationship[(i*parentid.size())+j] = w[parentid[j]-1];}
    }
}

/*************************************************************************************/
/*************************************************************************************/
/********************           Inverse Functions                *********************/
/*************************************************************************************/
/*************************************************************************************/
/******************/
/* Direct Inverse */
/******************/
void DirectInverse(double* relationship, int dimension)
{
    /* Set up parameters used for mkl variables */
    unsigned long i_p = 0, j_p = 0; long long int info = 0; char lower='L'; char diag='N';
    unsigned long n_a = dimension; const long long int int_n =(int)n_a;
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << relationship[(i*dimension)+j] << "\t";}
    //    cout << endl;
    //}
    dpotrf(&lower,&int_n,relationship,&int_n, &info);               /* Calculate upper triangular L matrix */
    dpotri(&lower,&int_n,relationship, &int_n,&info);               /* Calculate inverse of upper triangular matrix result is the inverse */
    #pragma omp parallel for private(j_p)
    for(i_p = 0; i_p < n_a; i_p++)
    {
        for(j_p = 0; j_p < i_p; j_p++){relationship[(i_p*n_a)+j_p] = relationship[(j_p*n_a)+i_p];}
    }
    //cout << endl << endl;
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << relationship[(i*dimension)+j] << "\t";}
    //    cout << endl;
    //}
}
/*****************************************************/
/* Meuwissen_Luo Ainverse assume animals go from 1:n */
/*****************************************************/
void Meuwissen_Luo_Ainv_Sparse(vector <int> &animal,vector <int> &sire,vector <int> &dam,vector < tuple<int,int,double> > &sprelationshipinv,vector <double> &pedinbreeding)
{
    /**************************/
    /* Begin to generate Ainv */
    /**************************/
    int animnumb = animal.size();
    vector < double > F((animnumb+1),0.0); vector < double > D(animnumb,0.0);
    F[0] = -1;                                                      /* This it makes so D is correct */
    for(int k = animal[0]; k < (animnumb+1); k++)             /* iterate through each row of l */
    {
        vector < double > L(animnumb,0.0);
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
        double bi = (1/sqrt(D[k-1])) * (1/sqrt(D[k-1]));
        if(sire[k-1] != 0 && dam[k-1] != 0)     /* both parent known */
        {
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(animal[k-1]-1),bi));
            sprelationshipinv.push_back(std::make_tuple((sire[k-1]-1),(animal[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(sire[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((dam[k-1]-1),(animal[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(dam[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((sire[k-1]-1),(sire[k-1]-1),(bi/double(4))));
            sprelationshipinv.push_back(std::make_tuple((sire[k-1]-1),(dam[k-1]-1),(bi/double(4))));
            sprelationshipinv.push_back(std::make_tuple((dam[k-1]-1),(sire[k-1]-1),(bi/double(4))));
            sprelationshipinv.push_back(std::make_tuple((dam[k-1]-1),(dam[k-1]-1),(bi/double(4))));
        }
        if(sire[k-1] != 0 && dam[k-1] == 0)     /* sire parent known */
        {
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(animal[k-1]-1),bi));
            sprelationshipinv.push_back(std::make_tuple((sire[k-1]-1),(animal[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(sire[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((sire[k-1]-1),(sire[k-1]-1),(bi/double(4))));
        }
        if(sire[k-1] == 0 && dam[k-1] != 0)     /* dam parent known */
        {
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(animal[k-1]-1),bi));
            sprelationshipinv.push_back(std::make_tuple((dam[k-1]-1),(animal[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(dam[k-1]-1),((bi/2)*-1)));
            sprelationshipinv.push_back(std::make_tuple((dam[k-1]-1),(dam[k-1]-1),(bi/double(4))));
        }
        if(sire[k-1] == 0 && dam[k-1] == 0) /* both parent unknown */
        {
            sprelationshipinv.push_back(std::make_tuple((animal[k-1]-1),(animal[k-1]-1),bi));
        }
    }
    /* Save inbreeding values */
    for(int i = 1; i < F.size(); i++){pedinbreeding[i-1] = F[i];}
}
/****************************************************************/
/* Generate new genomic relationship inverse based on recursion */
/****************************************************************/
void newgenomicrecursion(parameters &SimParameters, int relationshipsize, string BinaryG_Matrix_File,string Binarym_Matrix_File, string Binaryp_Matrix_File, string BinaryGinv_Matrix_File,vector < tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal)
{
    logfileloc << "           - Begin Constructing Genomic Relationship Inverse (Using recursion)." << endl;
    time_t start = time(0);
    using Eigen::MatrixXd; using Eigen::VectorXd;
    MatrixXd mg(relationshipsize,1);                               /* m vector in Misztal et al. (2014) */
    MatrixXd pg(relationshipsize,relationshipsize);               /* p matrix in Misztal et al. (2014) */
    MatrixXd Relationship(relationshipsize,relationshipsize);     /* GRM calculated previously */
    MatrixXd Relationshipinv(relationshipsize,relationshipsize);  /* Associated Inverse */
    /* Set matrices to zero */
    Relationship = MatrixXd::Identity(relationshipsize,relationshipsize);
    Relationshipinv = MatrixXd::Identity(relationshipsize,relationshipsize);
    pg = MatrixXd::Identity(relationshipsize,relationshipsize);
    for(int i = 0; i < relationshipsize; i++){Relationship(i,i) = 0; Relationshipinv(i,i) = 0; pg(i,i) = 0; mg(i,0) = 0;}
    Eigen::readbinary(BinaryG_Matrix_File.c_str(),Relationship);
    /* Add small numbers to make it invertible */
    for(int i = 0; i < relationshipsize; i++){Relationship(i,i) += 0.001;}
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationship(i,j) << "\t";}
    //    cout << endl;
    //}
    // Step 1
    Relationshipinv(0,0) = 1 / Relationship(0,0); mg(0,0) = Relationship(0,0);
    // Step 2: start at 2
    for(int i = 1; i < relationshipsize; i++)
    {
        /* block(starting at i, starting at j, size p, size q) */
        pg.block(0,i,i,1) = Relationshipinv.block(0,0,i,i) * Relationship.block(0,i,i,1);
        mg.block(i,0,1,1) = Relationship.block(i,i,1,1) - (pg.block(0,i,i,1)).transpose() * Relationship.block(0,i,i,1);
        VectorXd subp(i+1);
        for(int j = 0; j < i+1; j++)
        {
            subp(j) = 0;                        /* Zero out first so doesn't mess something up */
            if(j < i){subp(j) = -1 * pg(j,i);}
            if(j == i){subp(j) = 1;}
        }
        Relationshipinv.block(0,0,i+1,i+1) = Relationshipinv.block(0,0,i+1,i+1) + (((subp * subp.transpose()) / mg(i,0)));
    }
    if(SimParameters.getEBV_Calc() == "h1" || SimParameters.getEBV_Calc() == "h2" || SimParameters.getEBV_Calc() == "rohblup")
    {
        Eigen::writebinary(Binarym_Matrix_File.c_str(),mg);                 /* Output m Matrix into Binary */
        Eigen::writebinary(Binaryp_Matrix_File.c_str(),pg);                 /* Output p Matrix into Binary */
        Eigen::writebinary(BinaryGinv_Matrix_File.c_str(),Relationshipinv); /* Output Relationship Inverse Matrix into Binary */
    }
    if(SimParameters.getEBV_Calc() == "gblup" && SimParameters.getConstructGFreq() == "founder")
    {
        Eigen::writebinary(Binarym_Matrix_File.c_str(),mg);                 /* Output m Matrix into Binary */
        Eigen::writebinary(Binaryp_Matrix_File.c_str(),pg);                 /* Output p Matrix into Binary */
        Eigen::writebinary(BinaryGinv_Matrix_File.c_str(),Relationshipinv); /* Output Relationship Inverse Matrix into Binary */
    }
    mg.resize(0,0); pg.resize(0,0); Relationship.resize(0,0);
    /* Convert to triplet form */
    for(int i = 0; i < relationshipsize; i++)
    {
        for(int j = i; j < relationshipsize; j++)
        {
            sprelationshipinv.push_back(std::make_tuple((animal[i]-1),(animal[j]-1),Relationshipinv(i,j)));
        }
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationshipinv(i,j) << "\t";}
    //    cout << endl;
    //}
    Relationshipinv.resize(0,0);
    time_t end = time(0);
    logfileloc<<"           - Finished constructing Genomic Relationship Inverse. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
}
/**********************************************************/
/* Update genomic relationship inverse based on recursion */
/**********************************************************/
void updategenomicrecursion(int TotalAnimalNumber, int TotalOldAnimalNumber, string BinaryG_Matrix_File,string Binarym_Matrix_File, string Binaryp_Matrix_File, string BinaryGinv_Matrix_File,vector <tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal)
{
    logfileloc << "           - Begin Constructing Genomic Relationship Inverse (Using recursion)." << endl;
    time_t start = time(0);
    using Eigen::MatrixXd; using Eigen::VectorXd;
    // Need to fill mg, pg and Relationshipinv first with old animals
    MatrixXd Old_m(TotalOldAnimalNumber,1);                                 /* Used to store old animals for matrix m */
    MatrixXd Old_p(TotalOldAnimalNumber,TotalOldAnimalNumber);              /* Used to store old animals for matrix p */
    MatrixXd Old_Ginv(TotalOldAnimalNumber,TotalOldAnimalNumber);           /* Used to store old animals for matrix Ginv */
    // Matrices for current generation
    MatrixXd mg(TotalAnimalNumber,1);                                       /* m vector in Misztal et al. (2014) */
    MatrixXd pg(TotalAnimalNumber,TotalAnimalNumber);                       /* p matrix in Misztal et al. (2014) */
    MatrixXd Relationship(TotalAnimalNumber,TotalAnimalNumber);             /* GRM calculated previously */
    MatrixXd Relationshipinv(TotalAnimalNumber,TotalAnimalNumber);          /* Associated Inverse */
    /* Set matrices to zero */
    Relationship = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
    Relationshipinv = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
    pg = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
    for(int i = 0; i < TotalAnimalNumber; i++){Relationship(i,i) = 0; Relationshipinv(i,i) = 0; pg(i,i) = 0; mg(i,0) = 0;}
    /* Fill old matrices */
    Eigen::readbinary(Binarym_Matrix_File.c_str(),Old_m);
    Eigen::readbinary(Binaryp_Matrix_File.c_str(),Old_p);
    Eigen::readbinary(BinaryGinv_Matrix_File.c_str(),Old_Ginv);
    Eigen::readbinary(BinaryG_Matrix_File.c_str(),Relationship);
    /* Add small numbers to make it invertible */
    for(int i = 0; i < TotalAnimalNumber; i++){Relationship(i,i) += 1e-5;}
    /* Fill full matrices with already computed animals */
    for(int i = 0; i < TotalOldAnimalNumber; i++)
    {
        for(int j = 0; j < TotalOldAnimalNumber; j++){pg(i,j) = Old_p(i,j); Relationshipinv(i,j) = Old_Ginv(i,j);}
        mg(i,0) = Old_m(i,0);
    }
    logfileloc << "               - Filled Inverse Relationship Matrix with old animals." << endl;
    /* Old animals have been uploaded now start from last old animal */
    for(int i = TotalOldAnimalNumber; i < TotalAnimalNumber; i++)
    {
        /* block(starting at i, starting at j, size p, size q) */
        pg.block(0,i,i,1) = Relationshipinv.block(0,0,i,i) * Relationship.block(0,i,i,1);
        mg.block(i,0,1,1) = Relationship.block(i,i,1,1) - (pg.block(0,i,i,1)).transpose() * Relationship.block(0,i,i,1);
        VectorXd subp(i+1);
        for(int j = 0; j < i+1; j++)
        {
            subp(j) = 0;                        /* Zero out first so doesn't mess something up */
            if(j < i){subp(j) = -1 * pg(j,i);}
            if(j == i){subp(j) = 1;}
        }
        Relationshipinv.block(0,0,i+1,i+1) = Relationshipinv.block(0,0,i+1,i+1) + (((subp * subp.transpose()) / mg(i,0)));
    }
    logfileloc << "               - Filled Inverse Relationship Matrix for new animals." << endl;
    Eigen::writebinary(Binarym_Matrix_File.c_str(),mg);                       /* Output m Matrix into Binary for next generation */
    Eigen::writebinary(Binaryp_Matrix_File.c_str(),pg);                       /* Output p Matrix into Binary for next generation */
    Eigen::writebinary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);       /* Output Ginv Matrix into Binary for next generation */
    mg.resize(0,0); pg.resize(0,0); Relationship.resize(0,0);
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationshipinv(i,j) << "\t";}
    //    cout << endl;
    //}
    for(int i = 0; i < TotalAnimalNumber; i++)
    {
        for(int j = i; j < TotalAnimalNumber; j++)
        {
            sprelationshipinv.push_back(std::make_tuple((animal[i]-1),(animal[j]-1),Relationshipinv(i,j)));
        }
    }
    Relationshipinv.resize(0,0);
    time_t end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Inverse. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
}
/***************************************************************************/
/* Generate new genomic relationship inverse based on cholesky rank update */
/***************************************************************************/
void newgenomiccholesky(parameters &SimParameters, int relationshipsize, string BinaryG_Matrix_File,string BinaryLinv_Matrix_File, string BinaryGinv_Matrix_File, vector < tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal)
{
    logfileloc << "           - Begin Constructing Genomic Relationship Inverse (Using cholesky)." << endl;
    time_t start = time(0);
    using Eigen::MatrixXd; using Eigen::VectorXd;
    MatrixXd Relationship(relationshipsize,relationshipsize);     /* GRM calculated previously */
    Relationship = MatrixXd::Identity(relationshipsize,relationshipsize);
    for(int i = 0; i < relationshipsize; i++){Relationship(i,i) = 0;}
    Eigen::readbinary(BinaryG_Matrix_File.c_str(),Relationship);
    /* Add small numbers to make it invertible */
    for(int i = 0; i < relationshipsize; i++){Relationship(i,i) += 0.001;}
    /* Set up parameters used for mkl variables */
    unsigned long i_p = 0, j_p = 0; long long int info = 0; char lower='L'; char diag='N';
    unsigned long n_a = relationshipsize; const long long int int_n =(int)n_a;
    const long long int increment = int(1); const long long int int_na =(int)n_a * (int)n_a;
    double *GRM = new double[n_a*n_a];                      /* GRM */
    /* Copy it to a 2-dim array that is dynamically stored that all the computations will be on */
    #pragma omp parallel for private(j_p)
    for(i_p=0; i_p<n_a; i_p++)
    {
        for(j_p=0; j_p < n_a; j_p++){GRM[i_p*n_a+j_p]=Relationship(i_p,j_p);}
    }
    Relationship.resize(0,0);
    double *Linv = new double[n_a*n_a];                     /* Choleskey Inverse */
    dpotrf(&lower,&int_n,GRM,&int_n, &info);                /* Calculate upper triangular L matrix */
    dcopy(&int_na,GRM,&increment,Linv,&increment);          /* Copy to vector to calculate Linv */
    dtrtri (&lower,&diag,&int_n,Linv,&int_n,&info);         /* Calculate Linv */
    dpotri(&lower,&int_n,GRM, &int_n,&info);                /* Calculate inverse of upper triangular matrix result is the inverse */
    if(SimParameters.getEBV_Calc() == "h1" || SimParameters.getEBV_Calc() == "h2" || SimParameters.getEBV_Calc() == "rohblup")
    {
        MatrixXd LINV(relationshipsize,relationshipsize);     /* need to save for next generation */
        /* Copy Linv to a eigen matrix to save */
        #pragma omp parallel for private(j_p)
        for(i_p = 0; i_p < n_a; i_p++)
        {
            for(j_p = 0; j_p < n_a; j_p++){LINV(j_p,i_p) = Linv[(i_p*n_a)+j_p];}
        }
        for(i_p = 0; i_p < n_a; i_p++)
        {
            for(j_p = (i_p+1); j_p < n_a; j_p++){LINV(i_p,j_p) = 0.0;}
        }
        Eigen::writebinary(BinaryLinv_Matrix_File.c_str(),LINV);                    /* Output Linv Matrix into Binary */
        LINV.resize(0,0);
    }
    if(SimParameters.getEBV_Calc() == "gblup" && SimParameters.getConstructGFreq() == "founder")
    {
        MatrixXd LINV(relationshipsize,relationshipsize);     /* need to save for next generation */
        /* Copy Linv to a eigen matrix to save */
        #pragma omp parallel for private(j_p)
        for(i_p = 0; i_p < n_a; i_p++)
        {
            for(j_p = 0; j_p < n_a; j_p++){LINV(j_p,i_p) = Linv[(i_p*n_a)+j_p];}
        }
        for(i_p = 0; i_p < n_a; i_p++)
        {
            for(j_p = (i_p+1); j_p < n_a; j_p++){LINV(i_p,j_p) = 0.0;}
        }
        Eigen::writebinary(BinaryLinv_Matrix_File.c_str(),LINV);                    /* Output Linv Matrix into Binary */
        LINV.resize(0,0);
    }
    delete[] Linv;
    MatrixXd Relationshipinv(relationshipsize,relationshipsize);              /* GRM calculated previously */
    /* Copy GRM to a eigen matrix to save */
    #pragma omp parallel for private(j_p)
    for(i_p = 0; i_p < n_a; i_p++)
    {
        for(j_p = 0; j_p < n_a; j_p++){Relationshipinv(i_p,j_p) =  GRM[(i_p*n_a)+j_p];}
    }
    delete [] GRM;
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationshipinv(i,j) << "\t";}
    //   cout << endl;
    //}
    if(SimParameters.getEBV_Calc() == "h1" || SimParameters.getEBV_Calc() == "h2" || SimParameters.getEBV_Calc() == "rohblup")
    {
        Eigen::writebinary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);        /* Output Relationship Inverse Matrix into Binary */
    }
    if(SimParameters.getEBV_Calc() == "gblup" && SimParameters.getConstructGFreq() == "founder")
    {
        Eigen::writebinary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);        /* Output Relationship Inverse Matrix into Binary */
    }
    
    for(int i = 0; i < relationshipsize; i++)
    {
        for(int j = i; j < relationshipsize; j++)
        {
            sprelationshipinv.push_back(std::make_tuple((i),(j),Relationshipinv(i,j)));
        }
    }
    Relationshipinv.resize(0,0);
    time_t end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Inverse. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
}
/**********************************************************/
/* Update genomic relationship inverse based on cholesky */
/**********************************************************/
 void updategenomiccholesky(int TotalAnimalNumber, int TotalOldAnimalNumber, string BinaryG_Matrix_File,string BinaryLinv_Matrix_File, string BinaryGinv_Matrix_File, vector < tuple<int,int,double> > &sprelationshipinv,ostream& logfileloc,vector <int> &animal)
{
    logfileloc << "           - Begin Constructing Genomic Relationship Inverse (Using cholesky)." << endl;
    time_t start = time(0);
    using Eigen::MatrixXd; using Eigen::VectorXd;
    MatrixXd LINV_Old(TotalOldAnimalNumber,TotalOldAnimalNumber);               /* Used to store old animals for Linv */
    MatrixXd GINV_Old(TotalOldAnimalNumber,TotalOldAnimalNumber);               /* Used to store old animals for Ginv */
    MatrixXd Relationship(TotalAnimalNumber,TotalAnimalNumber);                 /* GRM calculated previously */
    MatrixXd Relationshipinv(TotalAnimalNumber,TotalAnimalNumber);                 /* GRM calculated previously */
    Eigen::readbinary(BinaryLinv_Matrix_File.c_str(),LINV_Old);
    Eigen::readbinary(BinaryGinv_Matrix_File.c_str(),GINV_Old);
    Eigen::readbinary(BinaryG_Matrix_File.c_str(),Relationship);
    /* Add small numbers to make it invertible */
    for(int i = 0; i < TotalAnimalNumber; i++){Relationship(i,i) += 1e-5;}
    /* Parameters that are used for mkl functions */
    unsigned long i_p = 0, j_p = 0;
    const long long int newanm = (TotalAnimalNumber - TotalOldAnimalNumber);    /* Number of new animals */
    const long long int oldanm = TotalOldAnimalNumber;                          /* Number of old animals */
    const long long int length = int(newanm) * int(newanm);
    const double beta = double(-1.0); const double alpha = double(1.0); const long long int increment = int(1);
    long long int info = 0; const long long int int_n =(int)newanm; const char diag = 'N'; char lower='L';
    /*************/
    /* old G-inv */
    /*************/
    double *G11inv = new double[oldanm*oldanm];
    #pragma omp parallel for private(j_p)
    for(i_p=0; i_p < oldanm; i_p++)
    {
        for(j_p=0; j_p < oldanm; j_p++){G11inv[(i_p*oldanm)+j_p] = GINV_Old(i_p,j_p);}
    }
    GINV_Old.resize(0,0);
    /*************/
    /* old L-inv */
    /*************/
    /* Create Linv to save for next generation */
    MatrixXd LINV(TotalAnimalNumber,TotalAnimalNumber);                 /* need to save for next generation */
    LINV = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
    for(i_p = 0; i_p < (newanm + oldanm); i_p++){LINV(i_p,i_p) = 0.0;}
    double *L11inv = new double[oldanm*oldanm];
    #pragma omp parallel for private(j_p)
    for(i_p=0; i_p < oldanm; i_p++)
    {
        for(j_p=0; j_p < oldanm; j_p++){L11inv[(i_p*oldanm)+j_p] = LINV_Old(i_p,j_p);}
    }
    // L11inv //
    for(int i_p = 0; i_p < oldanm; i_p++)
    {
        for(j_p = 0; j_p < oldanm; j_p++){LINV(i_p,j_p) = L11inv[(i_p*oldanm)+j_p];}
    }
    LINV_Old.resize(0,0);
    /* Grab old verses new relationship and new relationship part of GRM*/
    double *G21 = new double[newanm * oldanm];                              /* Grab GRM-old_new relationship*/
    int row = 0;
    for(i_p = oldanm; i_p < (newanm+oldanm); i_p++)
    {
        for(j_p = 0; j_p < oldanm; j_p++){G21[(row*oldanm) + j_p] = Relationship(i_p,j_p);}
        row++;
    }
    double *G22 = new double[newanm * newanm];                              /* Grab GRM-new_new relationship*/
    row = 0;
    for(i_p = oldanm; i_p < (newanm+oldanm); i_p++)
    {
        int rowj = 0;
        for(j_p = oldanm; j_p < (newanm+oldanm); j_p++)
        {
            G22[(row*newanm) + rowj] = Relationship(i_p,j_p); rowj++;
        }
        row++;
    }
    ////////////
    // Step 1 //
    ////////////
    double *L21 = new double[newanm * oldanm];                              /* Choleskey L- old_new */
    /* Calculate L21 */
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,newanm,oldanm,oldanm,1.0,G21,oldanm,L11inv,oldanm,0.0,L21,oldanm);
    /* Calculate G22.1 */
    double *Intermediate = new double[newanm * newanm];                     /* Intermediate is L21*L21' */
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,newanm,newanm,oldanm,1.0,L21,oldanm,L21,oldanm,0.0,Intermediate,newanm);
    cblas_daxpy(length,beta,Intermediate,increment,G22,increment);          /* G22 - Intermediate */
    delete [] Intermediate;
    ////////////
    // Step 2 //
    ////////////
    double *L22inv = new double[newanm * newanm];                           /* L22 Inverse */
    dcopy(&length,G22,&increment,L22inv,&increment);                        /* Copy to vector to calculate Linv */
    dpotrf(&lower,&int_n,L22inv,&int_n,&info);                              /* Calculate upper triangular L matrix */
    dtrtri(&lower,&diag,&int_n,L22inv,&int_n,&info);                        /* Calculate Linv */
    for(i_p = 0; i_p < newanm; i_p++)
    {
        for(j_p = (i_p+1); j_p < newanm; j_p++){L22inv[(j_p*newanm)+i_p] = L22inv[(i_p*newanm)+j_p];}
    }
    for(i_p = 0; i_p < newanm; i_p++)
    {
        for(j_p = (i_p+1); j_p < newanm; j_p++){L22inv[(i_p*newanm)+j_p] = 0.0;}
    }
    row = 0;
    for(i_p = oldanm; i_p < (oldanm+newanm); i_p++)
    {
        int rowj = 0;
        for(j_p = oldanm; j_p < (oldanm+newanm); j_p++){LINV(i_p,j_p) = L22inv[(row*newanm)+rowj]; rowj++;}
        row++;
    }
    /* Calculate L21 Inverse */
    double *Intermediatea = new double[newanm * oldanm];                    /* is L22inv * L21 */
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,newanm,oldanm,newanm,-1.0,L22inv,newanm,L21,oldanm,0.0,Intermediatea,oldanm);
    double *L21Inv = new double[newanm * oldanm];                           /* L21inv */
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,newanm,oldanm,oldanm,1.0,Intermediatea,oldanm,L11inv,oldanm,0.0,L21Inv,oldanm);
    delete [] Intermediatea;
    // L21inv //
    row = 0;
    for(i_p = oldanm; i_p < (oldanm + newanm); i_p++)
    {
        for(j_p = 0; j_p < oldanm; j_p++){LINV(i_p,j_p) = L21Inv[(row*oldanm)+j_p];}row++;
    }
    ////////////
    // Step 3 //
    ////////////
    /* G11 */
    double *G11invnew = new double[oldanm * oldanm];
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,oldanm,oldanm,newanm,1.0,L21Inv,oldanm,L21Inv,oldanm,0.0,G11invnew,oldanm);
    const long long int lengtha = int(oldanm) * int(oldanm);
    cblas_daxpy(lengtha,alpha,G11inv,increment,G11invnew,increment);
    for(int i_p = 0; i_p < oldanm; i_p++)
    {
        for(int j_p = 0; j_p < oldanm; j_p++){Relationshipinv(i_p,j_p) = G11invnew[(i_p*oldanm)+j_p];}
    }
    delete [] G11invnew;
    /* G21 & G12 */
    double *G21invnew = new double[newanm * oldanm];
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,newanm,oldanm,newanm,1.0,L22inv,newanm,L21Inv,oldanm,0.0,G21invnew,oldanm);
    row = 0;
    for(int i_p = oldanm; i_p < (oldanm+newanm); i_p++)
    {
        for(int j_p = 0; j_p < oldanm; j_p++){Relationshipinv(i_p,j_p) = Relationshipinv(j_p,i_p) = G21invnew[(row*oldanm)+j_p];}
        row++;
    }
    delete [] G21invnew;
    /* G22 */
    double *G22invnew = new double[newanm * newanm];
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,newanm,newanm,newanm,1.0,L22inv,newanm,L22inv,newanm,0.0,G22invnew,newanm);
    row = 0;
    for(i_p = oldanm; i_p < (oldanm+newanm); i_p++)
    {
        int rowj = 0;
        for(j_p = oldanm; j_p < (oldanm+newanm); j_p++){Relationshipinv(i_p,j_p) = G22invnew[(row*newanm)+rowj]; rowj++;}
        row++;
    }
    ///////////////////////////////////////
    // Generate Linv for next generation //
    ///////////////////////////////////////
    Eigen::writebinary(BinaryLinv_Matrix_File.c_str(),LINV);           /* Output Linv Matrix into Binary */
    Eigen::writebinary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);/* Output Relationship Inverse Matrix into Binary */
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationshipinv(i,j) << "\t";}
    //   cout << endl;
    //}
    for(int i = 0; i < TotalAnimalNumber; i++)
    {
        for(int j = i; j < TotalAnimalNumber; j++)
        {
            sprelationshipinv.push_back(std::make_tuple((animal[i]-1),(animal[j]-1),Relationshipinv(i,j)));
        }
    }
    /* Delete Matrices */
    delete [] G22invnew; delete [] G11inv; delete [] L11inv; delete [] G21; delete [] G22; delete [] L22inv; delete [] L21Inv;
    LINV.resize(0,0); Relationshipinv.resize(0,0);
    time_t end = time(0);
    logfileloc << "           - Finished constructing Genomic Relationship Inverse. " << endl;
    logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
}
/*********************************/
/* Generate H inverse in ssgblup */
/*********************************/
void H_inverse_function(parameters &SimParameters, vector <Animal> &population,string Pheno_Pedigree_File,string Pheno_GMatrix_File,string GenotypeStatus_path, vector <int> &animal, vector <double> &Phenotype, vector <int> trainanimals, vector < tuple<int,int,double> > &sprelationshipinv, ostream& logfileloc)
{
    using Eigen::SparseMatrix; using Eigen::MatrixXd; using Eigen::VectorXd;
    /* Hinverse is Ainv + (Ginv -A22) for genotyped animals only */
    /* Store Ainv in sparse triplet form then generate (Ginv - A22) also in triplet form then convert triplet form int full relationship matrix  */
    /********************************************************/
    /*** First generate Ainv; save inbreeding values to use in creation of A22 */
    vector <int> sire(animal.size(),0); vector <int> dam(animal.size(),0);
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    int linenumber = 0; int tempanim; int tempanimindex = 0; int indexintrainanim = 0;  /* Counter to determine where at in pedigree index's */
    string line;
    ifstream infile2;
    infile2.open(Pheno_Pedigree_File.c_str());                                                  /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line))
    {
        size_t pos = line.find(" ", 0); tempanim = (std::stoi(line.substr(0,pos)));
        if(trainanimals.size() == 0)
        {
            animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
            pos = line.find(" ",0); sire[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);             /* Grab Sire ID */
            pos = line.find(" ",0); dam[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);              /* Grab Dam ID */
            Phenotype[tempanimindex] = stod(line); tempanimindex++;                                                     /* Grab Phenotype */
        } else {
            if(tempanim == trainanimals[indexintrainanim])
            {
                animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
                pos = line.find(" ",0); sire[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);             /* Grab Sire ID */
                pos = line.find(" ",0); dam[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);              /* Grab Dam ID */
                Phenotype[tempanimindex] = stod(line); tempanimindex++;                                                     /* Grab Phenotype */
                indexintrainanim++;
            }
        }
        linenumber++;
    }
    vector <int> renum_animal; vector <int> renum_sire; vector <int> renum_dam; /* will be of the same dimension as animal but go from 1:n */
    if(trainanimals.size() > 0 )
    {
        /* First number may not begin with 1 therefore need to renumber animalIDs; Will still be in correct order */
        /* First check to make sure numbers go from 1 to n */
        string needtorenumber = "NO";
        for(int i = 0; i < animal.size(); i++){if(animal[i] != i+1){needtorenumber = "YES"; i = animal.size();}}
        if(needtorenumber == "YES")
        {
            for(int i = 0; i < animal.size(); i++)
            {
                renum_animal.push_back(0); renum_sire.push_back(0); renum_dam.push_back(0);
            }
            for(int i = 0; i < animal.size(); i++)
            {
                renum_animal[i] = i+1;
                int temp = animal[i];
                for(int j = 0; j < animal.size(); j++)
                {
                    /* change it if sire or dam */
                    if(temp == sire[j]){renum_sire[j] = renum_animal[i];}
                    if(temp == dam[j]){renum_dam[j] = renum_animal[i];}
                }
            }
            //for(int i = 0; i < renum_animal.size(); i++){cout<<renum_animal[i]<<" "<<renum_sire[i]<<" "<<renum_dam[i] << "\t";}
            //cout << endl;
       }
    }
    /********************************/
    /* First Generate Full Ainverse */
    /********************************/
    vector <double> pedinbreeding(animal.size(),0.0);
    vector < tuple<int,int,double> > tripletAinv;
    if(renum_animal.size() > 0)
    {
        Meuwissen_Luo_Ainv_Sparse(renum_animal,renum_sire,renum_dam,tripletAinv,pedinbreeding);
    } else {
        Meuwissen_Luo_Ainv_Sparse(animal,sire,dam,tripletAinv,pedinbreeding);
    }
    /*****************************************************************/
    /* Figure out who is genotyped and set up indexing for A22 and G */
    /*****************************************************************/
    vector <int> parentid;          /* ID of parent and location in full Ainv  */
    vector <int> renumparentid;     /* If animals are renumber what is the renumber parent ID */
    vector <int> locationA;         /* location of animal within full A if animals are truncated */
    vector <int> locationA22;       /* location of animal within A22 and G */
    vector <string> genotypes;      /* genotype for an animal */
    ifstream infilegstat;
    infilegstat.open(GenotypeStatus_path.c_str());
    if(infilegstat.fail()){cout << "GenotypeStatus!\n"; exit (EXIT_FAILURE);}
    while (getline(infilegstat,line))
    {
        size_t pos = line.find("No",0);   /* If you can find no skip it */
        if(pos == std::string::npos)
        {
            pos = line.find(" ",0); parentid.push_back(stoi(line.substr(0,pos))); line.erase(0, pos + 1);
        }
    }
    logfileloc << "            - Number of genotyped animals: " << parentid.size() << endl;
    if(renum_animal.size() > 0)
    {
        /* Need to ensure animals that used to be genotyped is removed when truncating animals */
        int currentloc = 0;
        while(currentloc < parentid.size())
        {
            //cout << parentid[currentloc] << endl;
            int lookloc = 0;
            while(1)
            {
                if(animal[lookloc] == parentid[currentloc])
                {
                    locationA.push_back(lookloc); renumparentid.push_back(renum_animal[lookloc]); currentloc++; break;
                }
                if(animal[lookloc] != parentid[currentloc]){lookloc++;}
                if(lookloc >= animal.size()){parentid.erase(parentid.begin()+currentloc); break;}
            }
        }
    }
    for(int i = 0; i < parentid.size(); i++){locationA22.push_back(i); genotypes.push_back("");}
    logfileloc << "            - Number of genotyped animals used: " << parentid.size() << endl;
    int numbgenotyped = parentid.size();
    //cout << parentid.size() << " " << locationA22.size() << " " << genotypes.size() << endl;
    //for(int i = 0; i < 10; i++){cout<<parentid[i]<<" "<<locationA22[i]<<" '"<<genotypes[i]<<"'"<<endl;}
    /**************************************************/
    /* Grab genotypes for animals that were genotyped */
    /**************************************************/
    tempanimindex = 0; int indexgeno = 0;
    ifstream infileG;
    infileG.open(Pheno_GMatrix_File.c_str());
    if(infileG.fail()){cout << "Error Opening 'Pheno_GMatrix' File\n";}
    while(getline(infileG,line))
    {
        if(indexgeno < parentid.size())
        {
            size_t pos = line.find(" ", 0); tempanim = (std::stoi(line.substr(0,pos)));
            if(tempanim == parentid[indexgeno])
            {
                pos = line.find(" ",0); line.erase(0, pos + 1);  pos = line.find(" ",0); line.erase(0, pos + 1);
                pos = line.find(" ",0); genotypes[indexgeno] =line.substr(0,pos);
                //cout << parentid[indexgeno] << " " << locationA22[indexgeno] << " " << genotypes[indexgeno] << endl;
                indexgeno++;
            }
        }
    }
    /* Generate M and scale values */
    double* Mobs = new double[3*(genotypes[0].size())]; /* Dimension 3 by number of markers */
    float scaleobs = update_M_scale_Hinverse(SimParameters,genotypes,Mobs);
    //cout << scaleobs << endl;
    //for(int i = 0; i < 3; i++){cout << Mobs[(i*genotypes[0].size())+0] << endl;}
    double *_grm_mkl = new double[genotypes.size()*genotypes.size()];   /* Allocate Memory for GRM */
    for(int i = 0; i < (genotypes.size()*genotypes.size()); i++){_grm_mkl[i] = 0.0;}
    if(SimParameters.getConstructG() == "VanRaden"){VanRaden_grm(Mobs,genotypes,_grm_mkl,scaleobs);}
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _grm_mkl[(i*genotypes.size())+j] << "\t";}
    //    cout << endl;
    //}
    //cout << endl;
    /**************************************************/
    /* Generate A22 Matrix for genotyped animals only */
    /**************************************************/
    double* A22relationship = new double[genotypes.size()*genotypes.size()];
    for(int i = 0; i < (genotypes.size()*genotypes.size()); i++){A22relationship[i] = 0.0;}
    if(renum_animal.size() > 0)
    {
        A22_Colleau(renum_animal,renum_sire,renum_dam,renumparentid,pedinbreeding,A22relationship);
    } else {
        A22_Colleau(animal,sire,dam,parentid,pedinbreeding,A22relationship);
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << A22relationship[(i*genotypes.size())+j] << "\t";}
    //    cout << endl;
    //}
    //cout << endl << endl;
    logfileloc << "            - Final A22 Matrix Summary Statistics: " << endl;
    vector < double > A22summarystats(6,0.0);
    MatrixStats(A22summarystats,A22relationship,numbgenotyped,logfileloc);
    /*****************************/
    /* Make A22 and G be similar */
    /*****************************/
    logfileloc << "            - Blend G (0.95) with A (0.05). " << endl;
    for(int i = 0; i < (genotypes.size()*genotypes.size()); i++){_grm_mkl[i] = ((_grm_mkl[i] *0.95) + (A22relationship[i]*0.05));}
    vector < double > Gsummarystats(6,0.0);
    logfileloc << "            - Initial G Matrix Summary Statistics: " << endl;
    MatrixStats(Gsummarystats,_grm_mkl,numbgenotyped,logfileloc);
    logfileloc << "            - Force G diagonals and off-diagonals to equal A22 diagonals and off-diagonals. " << endl;
    // - Solve system of equations:
    //  1.) a + b(" << initmeanoffdiagG << ") = " << meanoffdiagonalA22 << endl;
    //  2.) a + b(" << initmeandiagG << ") = " << meandiagonalA22 << endl;
    double a_gadj = ((A22summarystats[3]*Gsummarystats[0])-(Gsummarystats[3]*A22summarystats[0])) / ((1*Gsummarystats[0])-(Gsummarystats[3]*1));
    double b_gadj = ((1*A22summarystats[0])-(A22summarystats[3]*1)) / ((1*Gsummarystats[0])-(Gsummarystats[3]*1));
    logfileloc <<"              - NewG(i,j) = " << a_gadj << "OldG(i,j) * " << b_gadj << ". (Christensen et al. 2012)." << endl;
    for(int i = 0; i < (genotypes.size()*genotypes.size()); i++){_grm_mkl[i] = (a_gadj + (_grm_mkl[i]*b_gadj));}
    for(int i = 0; i < 6; i++){Gsummarystats[i] = 0.0;}
    logfileloc << "            - Final G Matrix Summary Statistics: " << endl;
    MatrixStats(Gsummarystats,_grm_mkl,numbgenotyped,logfileloc);
    for(int i = 0; i < genotypes.size(); i++){_grm_mkl[(i*genotypes.size())+i] += 0.001;}
    /* Estimate correlation between grm and A22 for diagonals and off-diagonals */
    vector < double > GA22summarystats(2,0.0);
    MatrixCorrStats(GA22summarystats,_grm_mkl,A22relationship,numbgenotyped,logfileloc);

    
    //MatrixXd G(genotypes.size(),genotypes.size()); /* stores IBD based relationships */
    //for(int i = 0; i < genotypes.size(); i++)
    //{
    //    for(int j = 0; j < genotypes.size(); j++){G(i,j) = _grm_mkl[(i*genotypes.size())+j];}
    //}
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << _grm_mkl[(i*genotypes.size())+j] << "\t";}
    //    cout << endl;
    //}
    //cout << endl << endl;
    //Eigen::SelfAdjointEigenSolver<MatrixXd> es(G);
    //MatrixXd D = es.eigenvalues().asDiagonal();
    //MatrixXd V = es.eigenvectors();
    //int numneg = 0;
    //for(int i = 0; i < genotypes.size(); i++){if(D(i,i) < 1e-12){numneg += 1;}}
    //cout << "   - Number of negative eigenvalues: " << numneg << endl;
    /****************************/
    /* Generate Ginv and A22inv */
    /****************************/
    DirectInverse(A22relationship,numbgenotyped);
    DirectInverse(_grm_mkl,numbgenotyped);
    logfileloc << "            - Final A22Inv Matrix Summary Statistics: " << endl;
    vector < double > A22Invsummarystats(6,0.0);
    MatrixStats(A22Invsummarystats,A22relationship,numbgenotyped,logfileloc);
    logfileloc << "            - Final GInv Matrix Summary Statistics: " << endl;
    vector < double > GInvsummarystats(6,0.0);
    MatrixStats(GInvsummarystats,_grm_mkl,numbgenotyped,logfileloc);
    /****************************/
    /* Generate Ginv - A22inv   */
    /****************************/
    vector < tuple<int,int,double> > tripletG_A22inv; double tempGinvA22inv;
    if(renum_animal.size() > 0)
    {
        for(int i = 0; i < numbgenotyped; i++)
        {
            for(int j = 0; j < numbgenotyped; j++)
            {
                tempGinvA22inv = _grm_mkl[(i*numbgenotyped)+j] - A22relationship[(i*numbgenotyped)+j];
                tripletG_A22inv.emplace_back(std::make_tuple((locationA[i]),(locationA[j]),tempGinvA22inv));
            }
        }
    } else {
        for(int i = 0; i < numbgenotyped; i++)
        {
            for(int j = 0; j < numbgenotyped; j++)
            {
                tempGinvA22inv = _grm_mkl[(i*numbgenotyped)+j] - A22relationship[(i*numbgenotyped)+j];
                tripletG_A22inv.emplace_back(std::make_tuple((parentid[i]-1),(parentid[j]-1),tempGinvA22inv));
            }
        }
    }
    delete [] Mobs; delete [] _grm_mkl; delete [] A22relationship;
    /**************************************************************/
    /* Combine Ainv and G_A22 inverse triplet form to matrix form */
    /**************************************************************/
    for(int i = 0; i < tripletAinv.size(); i++)
    {
        sprelationshipinv.push_back(std::make_tuple(get<0>(tripletAinv[i]),get<1>(tripletAinv[i]),get<2>(tripletAinv[i])));
    }
    for(int i = 0; i < tripletG_A22inv.size(); i++)
    {
        sprelationshipinv.push_back(std::make_tuple(get<0>(tripletG_A22inv[i]),get<1>(tripletG_A22inv[i]),get<2>(tripletG_A22inv[i])));
    }
    tripletG_A22inv.clear(); tripletAinv.clear();
}




/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/********************                                      Solver Functions                                          ********************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/*********************************/
/* Solve for solutions using pcg */
/*********************************/
void pcg_solver_dense(vector <tuple<int,int,double>> &lhs_sparse,vector <double> &rhs_sparse, vector <double> &solutionsa, int dimen, int* solvediter)
{
    //cout<<get<0>(lhs_sparse[lhs_sparse.size()-1])<<" "<<get<1>(lhs_sparse[lhs_sparse.size()-1])<<" "<<get<2>(lhs_sparse[lhs_sparse.size()-1])<<endl;
    double* lhs = new double[dimen*dimen];                                 /* LHS dimension: animal + intercept by animal + intercept */
    for(int i = 0; i < (dimen*dimen); i++){lhs[i] = 0;}
    for(int i = 0; i < lhs_sparse.size(); i++)
    {
        lhs[(get<0>(lhs_sparse[i])*dimen)+get<1>(lhs_sparse[i])] += get<2>(lhs_sparse[i]);
        if(get<0>(lhs_sparse[i]) != get<1>(lhs_sparse[i]))
        {
            lhs[(get<1>(lhs_sparse[i])*dimen)+get<0>(lhs_sparse[i])] += get<2>(lhs_sparse[i]);
        }
    }
    double* rhs = new double[dimen];                                                     /* RHS dimension: animal + intercept by 1 */
    for(int i = 0; i < rhs_sparse.size(); i++){rhs[i] = rhs_sparse[i];}
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << lhs[(i*dimen)+j] << " ";}
    //    cout << endl;
    //}
    //for(int i = 0; i < 5; i++){cout << rhs[i] << endl;}
    lhs_sparse.clear(); rhs_sparse.clear();
    //  PCG involves 4 vectors (Notation is like Mrodes book, but solving pseudo code is like UGA short course)
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
    int k = 1; float diff = 1; float tol = 1E-10;
    /* Variables used in mkl functions */
    const long long int lhssize = dimen;
    const long long int onesize = 1;
    const long long int increment = int(1);
    int incx = 1; string stop = "NO"; int incy = 1;
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
        dcopy(&lhssize,solutions,&increment,oldb,&increment);                      /* copy current solutions to previous solutions */
        /* get new solutions and residuals */
        #pragma omp parallel for
        for(int i = 0; i < dimen; i++)
        {
            solutions[i] = solutions[i] + (alpha[0]*p[i]);
            e[i] = e[i] - (alpha[0]*v[i]);
        }
        k++;
        /* Compute difference to determine if converged sum (current - previous beta)^2 / sum(current)^2 */
        float num = 0.0; float den = 0.0;
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
    delete[] Minv; delete[] e; delete[] d; delete[] v; delete[] p; delete[] intermediate; delete [] lhs;
    delete[] oldb; delete[] tau1; delete[] tau2; delete[] alpha; delete[] beta; delete[] solutions; delete [] rhs;
}
/********************************************/
/* Solve for solutions using pcg but sparse */
/********************************************/
void pcg_solver_sparse(vector <tuple<int,int,double>> &lhs_sparse,vector <double> &rhs_sparse, vector <double> &solutionsa, int dimen, int* solvediter)
{
    using Eigen::SparseMatrix; using Eigen::MatrixXd; using Eigen::VectorXd;
    /************************************************************/
    /**            Initialize all the variables                **/
    /************************************************************/
    //  PCG involves 4 vectors (Notation is like Mrodes book, but solving pseudo code is like UGA short course)
    //  e:          vector of residuals (size: number of unknowns)
    //                  - Initially found by: RHS - (LHS * beta)
    //  solutions:  vector of solutions (size: number of unknowns)
    //                  - Initially given based on starting values
    //  d:          vector of search directions
    //                  - Initially given by: Minv * e (Minv is preconditioner matrix diagonals of LHS)
    //  v:          working vector
    /* Initialize vectors or matrices */
    VectorXd solutions(dimen); solutions.setZero(dimen);        /* Initialize b vector */
    SparseMatrix <double,Eigen::RowMajor> Minv (dimen,dimen);   /* Initialize Minv */
    VectorXd RHS(dimen);                                        /* RHS of mme */
    for(int i = 0; i < rhs_sparse.size(); i++){RHS(i) = rhs_sparse[i];}
    rhs_sparse.clear();
    VectorXd e(dimen); e.setZero(dimen);                        /* Initialize e vector */
    VectorXd d(dimen); d.setZero(dimen);                        /* Initialize d vector */
    VectorXd v(dimen); v.setZero(dimen);                        /* Initialize v (working vector) */
    VectorXd p(dimen); p.setZero(dimen);                        /* Initialize p */
    VectorXd oldb(dimen); oldb.setZero(dimen);                  /* Initialize b vector from previous iteration */
    VectorXd tau2(1);                                           /* Initialize tau2 */
    VectorXd tau1(1);                                           /* Intialize tau1 */
    VectorXd alpha(1);                                          /* Initialize alpha */
    VectorXd beta(1);                                           /* Initialize second step size */
    /************************************************************/
    /** Convert Triplet LHS to sparse matrix and generate Minv **/
    /************************************************************/
    typedef Eigen::Triplet<double> T;
    SparseMatrix <double,Eigen::RowMajor> LHS (dimen,dimen);   /* Convert triplet to sparse matrix */
    vector<T> tripletListLHS; tripletListLHS.reserve(lhs_sparse.size());
    vector<T> tripletListMinv; tripletListMinv.reserve(dimen);
    for(int i = 0; i < lhs_sparse.size(); i++)
    {
        tripletListLHS.push_back(T(get<0>(lhs_sparse[i]),get<1>(lhs_sparse[i]),get<2>(lhs_sparse[i])));
        if(get<0>(lhs_sparse[i]) == get<1>(lhs_sparse[i]))  /* if diagonal add to Minv */
        {
            tripletListMinv.push_back(T(get<0>(lhs_sparse[i]),get<1>(lhs_sparse[i]),get<2>(lhs_sparse[i])));
        }
    }
    LHS.setFromTriplets(tripletListLHS.begin(),tripletListLHS.end()); lhs_sparse.clear();
    Minv.setFromTriplets(tripletListMinv.begin(),tripletListMinv.end());
    /* inverse of diagonals is just 1 / diagonal of LHS */
    for(int i = 0; i < dimen; i++){Minv.coeffRef(i,i) = 1 / Minv.coeffRef(i,i);}
    /************************************************************/
    /**                      Start PCG                         **/
    /************************************************************/
    tau1(0) = 0; tau2(0) = 1; alpha(0) = 0; beta(0) = 0;
    int k = 1; float diff = 10; float tol = 1E-10;
    e = RHS;
    while(diff > tol)
    {
        d = Minv * e;
        tau1(0) = e.dot(d);
        if(k == 1)
        {
            p = d;
            beta(0) = 0;
        }
        if(k > 1)
        {
            beta(0) = tau1(0) / tau2(0);
            p = d + (beta(0) * p);
        }
        v = LHS * p;
        alpha(0) = tau1(0) / (p.dot(v));
        oldb = solutions;
        solutions = solutions + (alpha(0) * p);
        e = e - (alpha(0) * v);
        k++;
        /* Compute difference to determine if converged sum (current - previous beta)^2 / sum(current)^2 */
        double num = 0;
        double den = 0;
        for(int i = 0; i < dimen; i++)
        {
            num += (solutions(i) - oldb(i)) * (solutions(i) - oldb(i));
            den += solutions(i) * solutions(i);
        }
        diff = num / den;
        tau2(0) = tau1(0);
    }
    /************************************************************/
    /**                      PCG Finished                      **/
    /************************************************************/
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++){solutionsa[i] = solutions[i];}
    solvediter[0] = k;
    /* clear all working matrices or vectors */
    solutions.resize(0); Minv.resize(0,0); RHS.resize(0); e.resize(0); d.resize(0); beta.resize(0);
    v.resize(0); p.resize(0); oldb.resize(0); tau2.resize(0); tau1.resize(0); alpha.resize(0); LHS.resize(0,0);
    tripletListLHS.clear(); tripletListMinv.clear();
}
/******************************************************************/
/* Solve for solutions but need to convert from sparse to dense   */
/******************************************************************/
void direct_solversparse(parameters &SimParameters,vector <tuple<int,int,double>> &lhs_sparse,vector <double> &rhs_sparse, vector <double> &solutionsa, vector <double> &trueaccuracy,int dimen)
{
    double* lhs = new double[dimen*dimen];                                 /* LHS dimension: animal + intercept by animal + intercept */
    for(int i = 0; i < (dimen*dimen); i++){lhs[i] = 0;}
    for(int i = 0; i < lhs_sparse.size(); i++)
    {
        lhs[(get<0>(lhs_sparse[i])*dimen)+get<1>(lhs_sparse[i])] += get<2>(lhs_sparse[i]);
    }
    double* rhs = new double[dimen];                                                     /* RHS dimension: animal + intercept by 1 */
    for(int i = 0; i < rhs_sparse.size(); i++){rhs[i] = rhs_sparse[i];}
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << lhs[(i*dimen)+j] << " ";}
    //    cout << endl;
    //}
    //for(int i = 0; i < 5; i++){cout << rhs[i] << endl;}
    lhs_sparse.clear(); rhs_sparse.clear();
    /*******************************/
    /****    Generate Inverse   ****/
    /*******************************/
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
    /* Generate Accuracy */
    double lambda = (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd());
    //cout << "Lambda: " << (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd()) << endl;
    //for(int i = 1; i < 10; i++){cout << lhs[(i*n)+i] << endl;}
    for(int i = 1; i < dimen; i++){trueaccuracy[i-1] = sqrt(1 - (lhs[(i*n)+i]*lambda));}
    //for(int i = 0; i < 9; i++){cout << accuracy[i-1] << endl;}
    const long long int lhssize = dimen;
    const long long int onesize = 1;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,lhssize,onesize,lhssize,1.0,lhs,lhssize,rhs,onesize,0.0,solutions,onesize);
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++){solutionsa[i] = solutions[i];}                                 /* set solutions to zero & copy rhs to e */
    delete [] solutions; delete [] lhs; delete [] rhs ;
}
/****************************************************/
/* Solve EBV based on Bayesian Marker Effects Model */
/****************************************************/
void bayesianestimates(parameters &SimParameters,vector <Animal> &population,string Master_DF_File, string Pheno_GMatrix_File, string Pheno_Pedigree_File, int Gen,vector <double> &estimatedsolutions,string Bayes_MCMC_Samples,string Bayes_PosteriorMeans, ostream& logfileloc)
{
    /* Initialize Parameters that will be filled */
    int Bayesseednumber = SimParameters.getSeed();                          /* Inititial seed to make sure it is reproducible */
    double R2 = SimParameters.getVarAdd();                                  /* Narrow sense heritability */
    vector < int > animalid;                                                /* ID of individual */
    vector < double > phenotype;                                            /* Phenotype of an individual */
    vector < string > genotype;                                             /* genotype for an individual as a string */
    vector < int > generationnum;                                           /* Generation animal was born */
    vector < int > referencegen;                                            /* Generation number that is included in reference population */
    /* Figure number of generations */
    int recentgen = -5;
    /* Figure out which generations to keep */
    for(int i = 0; i < Gen; i++)
    {
        if((Gen-i) - SimParameters.getreferencegenerations() <= 0){referencegen.push_back(i);}
    }
    //for(int i = 0; i < referencegen.size(); i++){cout << referencegen[i] << " ";}
    /************************************************************************************************/
    /************************************************************************************************/
    /********************                   Figure Out Training Population                    *******/
    /************************************************************************************************/
    /************************************************************************************************/
    vector < int > fullanim; vector < int > fullsire; vector < int > fulldam; vector < int > trainanimals;
    /*******************************/
    /* First read in full pedigree */
    /*******************************/
    vector <string> numbers; string line;                                               /* Import file and put each row into a vector */
    ifstream infile9;
    infile9.open(Pheno_Pedigree_File.c_str());
    if(infile9.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile9,line)){numbers.push_back(line);}     /* Stores in vector and each new line push back to next space */
    for(int i = 0; i < numbers.size(); i++)
    {
        size_t pos = numbers[i].find(" ",0); fullanim.push_back(atoi(numbers[i].substr(0,pos).c_str())); numbers[i].erase(0, pos + 1); /* Grab Animal */
        pos = numbers[i].find(" ",0); fullsire.push_back(atoi(numbers[i].substr(0,pos).c_str())); numbers[i].erase(0, pos + 1); /* Grab Sire */
        pos = numbers[i].find(" ",0); fulldam.push_back(atoi(numbers[i].substr(0,pos).c_str())); numbers[i].erase(0, pos + 1); /* Grab Dam */
        //cout << fullanim[fullanim.size()-1] << " " << fullsire[fullanim.size()-1] << " " << fulldam[fullanim.size()-1] << endl;
    }
    numbers.clear();
    //cout << "--" << referencegen.size() << " " << Gen << endl;
    if(referencegen.size() < Gen)
    {
        //cout << fullanim.size() << " " << fullsire.size() << " " << fulldam.size() << endl;
        //cout << SimParameters.getreferencegenblup() << endl;
        /***********************************************/
        /* Grab the animals that are currently progeny */
        /***********************************************/
        vector < int > tempparents;                         /* used to check parentage */
        for(int i = 0; i < population.size(); i++)
        {
            if(population[i].getAge() == 1)
            {
                trainanimals.push_back(population[i].getID());
                tempparents.push_back(population[i].getSire()); tempparents.push_back(population[i].getDam());
            }
        }
        int traingenback = 0;
        while(traingenback < SimParameters.getreferencegenerations())
        {
            //cout << trainanimals.size() << endl;
            /* First take all parents and add to trainanimals */
            //cout << tempparents.size() << endl;
            //for(int i = 0; i < tempparents.size(); i++){cout << tempparents[i] << " ";}
            //cout << endl << endl;
            sort(tempparents.begin(),tempparents.end());
            tempparents.erase(unique(tempparents.begin(),tempparents.end()),tempparents.end());
            /* add current parents to training */
            for(int i = 0; i < tempparents.size(); i++){trainanimals.push_back(tempparents[i]);}
            traingenback++;
            //cout << trainanimals.size() << endl;
            //for(int i = 0; i < tempparents.size(); i++){cout << tempparents[i] << " ";}
            //cout << endl << endl;
            /* Find parents of current parents in tempparents */
            vector < int > temporarynewparents;
            for(int i = 0; i < tempparents.size(); i++)
            {
                if(fullanim[tempparents[i]-1]!= tempparents[i]){cout << "Line 224" << endl; exit (EXIT_FAILURE);}
                if(fullsire[tempparents[i]-1] != 0){temporarynewparents.push_back(fullsire[tempparents[i]-1]);}
                if(fulldam[tempparents[i]-1] != 0){temporarynewparents.push_back(fulldam[tempparents[i]-1]);}
                //cout << fullanim[tempparents[i]-1] << " " << fullsire[tempparents[i]-1] << " " << fulldam[tempparents[i]-1] << endl;
            }
            tempparents.clear();
            //cout << tempparents.size() << endl << endl;
            //cout << temporarynewparents.size() << endl;
            //for(int i = 0; i < temporarynewparents.size(); i++){cout << temporarynewparents[i] << " ";}
            //cout << endl << endl;
            /* Sort old parents and put in tempparents */
            sort(temporarynewparents.begin(),temporarynewparents.end());
            temporarynewparents.erase(unique(temporarynewparents.begin(),temporarynewparents.end()),temporarynewparents.end());
            //cout << temporarynewparents.size() << endl;
            //for(int i = 0; i < temporarynewparents.size(); i++){cout << temporarynewparents[i] << " ";}
            //cout << endl << endl;
            for(int i = 0; i < temporarynewparents.size(); i++){tempparents.push_back(temporarynewparents[i]);}
            if(tempparents.size() == 0){break;}
            //cout << trainanimals.size() << " " << traingenback << endl;
        }
        //cout << trainanimals.size() << endl;
        //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << " ";}
        //cout << endl << endl;
        sort(trainanimals.begin(),trainanimals.end());
        trainanimals.erase(unique(trainanimals.begin(),trainanimals.end()),trainanimals.end());
        /* Find all progeny of animals and include */
        int currentsize = trainanimals.size();
        for(int i = 0; i < currentsize; i++)
        {
            //cout << trainanimals[i] << "-" << trainanimals.size() << "+";
            for(int j = 0; j < fullsire.size(); j++)
            {
                if(fullsire[j] == trainanimals[i] || fulldam[j] == trainanimals[i]){trainanimals.push_back(fullanim[j]);}
            }
            //cout << trainanimals.size() << "\t";
            //if(i > 20) { cout << endl; exit (EXIT_FAILURE);}
        }
        //cout << trainanimals.size() << endl;
        //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << " ";}
        //cout << endl << endl;
        sort(trainanimals.begin(),trainanimals.end());
        trainanimals.erase(unique(trainanimals.begin(),trainanimals.end()),trainanimals.end());
        //cout << trainanimals.size() << endl;
        //for(int i = 0; i < trainanimals.size(); i++){cout << trainanimals[i] << " ";}
        //cout << endl << endl;
        //exit (EXIT_FAILURE);
        
    } else{
        for(int i = 0; i < fullanim.size(); i++)
        {
            trainanimals.push_back(fullanim[i]);
        }
    }
    /**********************************************************************************/
    /* First read in all animals that didn't get selected from appropriate generation */
    /**********************************************************************************/
    numbers.clear();
    ifstream infile1; int indexintrainanim = 0;
    infile1.open(Pheno_GMatrix_File.c_str());
    if(infile1.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile1,line))
    {
        size_t pos = line.find(" ", 0); int tempanim = (std::stoi(line.substr(0,pos))); line.erase(0,pos + 1);
        if(tempanim == trainanimals[indexintrainanim])
        {
            animalid.push_back(tempanim);
            pos = line.find(" ", 0); phenotype.push_back(std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
            pos = line.find(" ", 0); genotype.push_back(line.substr(0,pos)); line.erase(0,pos + 1); indexintrainanim++;
        }
    }
    //cout << animalid.size() << " " << phenotype.size() << " " << genotype.size() << " " << trainanimals.size() << endl;
    numbers.clear();
    using Eigen::MatrixXd;
    int n = animalid.size();
    int m = genotype[0].size();
    /**********************************/
    /* Default Priors used in Program */
    /**********************************/
    double priordf_res = 5;                 /* Prior degrees of Freedom for residual */
    double scaleres;                        /* Scale parameter for residual */
    double priordf_gen = 5;                 /* Prior degrees of Freedom for variance assigned to markers */
    double MSx = 0.0;                       /* Used in scaling scale parameter */
    MatrixXd ScaleMark(1,1);                /* Scale parameter for variance assigned to markers */
    double rateprior;                       /* rate prior assigned a gamma density for scale of marker */
    double shapeprior;                      /* shape prior assigned to a gamma density for scale of marker */
    /* Bayes B & C */
    double pi;                              /* propabilility of inclusiong (can be fixed or estimated) */
    double priorcount;                      /* Used to calculate variance of pi (i.e. (pi*(1-pi)) / (priorcount + 1)) */
    double priorcountin;                    /* Prior Count included */
    double priorcountout;                   /* Prior Count excluded */
    /*******************************************************/
    /*             arrays used throughout                  */
    /*******************************************************/
    /* Static arrays used throughout */
    MatrixXd add(n,m);                  /* Genotype Matrix */
    MatrixXd y(n,1);                    /* Phenotype vector */
    vector < double> x2(m,0.0);         /* sum of genotypes squared */
    /* Temporary variables used within mcmc part */
    MatrixXd yhat(n,1);                 /* predicted value */
    MatrixXd e(n,1);                    /* residual */
    MatrixXd b(m,1);                    /* Marker estimate */
    MatrixXd bvar(m,1);                 /* Marker variance */
    vector <double> d;                  /* Included or excluded in model (only need for Bayes B & C */
    vector <double> ppa;                /* posterior probability of acceptance */
    MatrixXd mu(1,1);                   /* intercept */
    MatrixXd vare(1,1);                 /* residual variance */
    /* Posterior mean and variances */
    vector < double > postb(m,0.0);
    vector < double > postvarB(m,0.0);
    vector < int > iterationnumber((SimParameters.getnumiter()-SimParameters.getburnin())/SimParameters.getthin());
    vector < double > samples_res((SimParameters.getnumiter()-SimParameters.getburnin())/SimParameters.getthin());
    vector < double > samples_scalemark;                    /* used in Bayes A and B */
    vector < double > samples_varb;                         /* used in Bayes Ridge Regression and C */
    vector < double > samples_pi;                           /* used in Bayes B and Bayes C */
    if(SimParameters.getmethod()=="BayesA" || SimParameters.getmethod()=="BayesB")
    {
        for(int i = 0; i < samples_res.size(); i++){samples_scalemark.push_back(0.0);}
    }
    if((SimParameters.getmethod()=="BayesB"&&SimParameters.getpie_f()=="estimate")||(SimParameters.getmethod()=="BayesC"&&SimParameters.getpie_f()=="estimate"))
    {
        for(int i = 0; i < samples_res.size(); i++){samples_pi.push_back(0.0);}
    }
    if(SimParameters.getmethod()=="BayesRidgeRegression" || SimParameters.getmethod()=="BayesC")
    {
        for(int i = 0; i < samples_res.size(); i++){samples_varb.push_back(0.0);}
    }
    double postintercept = 0.0; double postmarkervar = 0.0; double postvarb = 0.0; double postresidual = 0.0; double postpi = 0.0;
    /*==============================================*/
    /* initialize all random number generators here */
    /*==============================================*/
    mt19937 gen(Bayesseednumber);                                                /* Generate random number to start with */
    chi_squared_distribution<double> distr_chisq_u(priordf_gen+1);          /* sigma^2u chi square for set specific variances */
    chi_squared_distribution<double> distr_chisq_u1(priordf_gen+m);         /* sigma^2u chi square for one variance across all markers */
    chi_squared_distribution<double> distr_chisq_e(n+priordf_res);          /* sigma^2e chi square*/
    std::uniform_real_distribution<double> uniform_bc(0.0,1.0);             /* uniform used in Bayes B_C  */
    /*******************************************************/
    /*******************************************************/
    /*                 Generate prior values               */
    /*******************************************************/
    /*******************************************************/
    double meany = 0.0; double vary = 0.0;
    /* Fill add and y */
    for(int i = 0; i < n; i++)
    {
        string tempgeno = genotype[i];
        for(int j = 0; j < m; j++)
        {
            int temp = tempgeno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            add(i,j) = temp;
        }
        y(i,0) = phenotype[i];
        meany += phenotype[i];
    }
    meany = meany / double(n);
    mu(0,0) = meany;
    for(int i = 0; i < n; i++){yhat(i,0) = meany; e(i,0) = y(i,0) - yhat(i,0);}
    /* Calculate MSx which is used across all priors */
    vector < double > meancolumns(m,0.0);
    double sumMeanXSq = 0.0; double sum_squared = 0.0;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++){x2[i] += (add(j,i)*add(j,i)); meancolumns[i] += add(j,i);}
    }
    for(int i = 0; i < m; i++)
    {
        meancolumns[i] = meancolumns[i] / double(n);
        sumMeanXSq += (meancolumns[i]*meancolumns[i]);
        sum_squared += x2[i];
    }
    MSx = (sum_squared / double(n)) - sumMeanXSq;
    meancolumns.clear();
    /*******************************************/
    /* Read in previous mcmc chain if possible */
    /*******************************************/
    ifstream infile3;
    infile3.open(Bayes_PosteriorMeans.c_str());
    if(infile3.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile3,line)){numbers.push_back(line);}     /* Stores in vector and each new line push back to next space */
    string tagdecreasewindow = numbers[0];
    if(numbers[0] == "GeneratePriorH2")
    {
        /******************************************************************************/
        /* Scale parameter for the residual (i.e. vary(y) * (1-R2) * (priordf_res+2)) */
        /******************************************************************************/
        for(int i = 0; i < n; i++){vary += (y(i,0)-meany) * (y(i,0)-meany);}
        vary = vary / (double(n-1));
        vare(0,0) = vary*(1-R2);
        scaleres = vary*(1-R2)*(priordf_res+ 2);
        /*************************************************************************************/
        /* Prior parameters for the part associated with markers (i.e. varies across methods */
        /*************************************************************************************/
        if(SimParameters.getmethod() == "BayesA")
        {
            /* Scale parameter for the marker variance (i.e. vary(y) * R2 / MSx * (priordf_gen + 2)) */
            ScaleMark(0,0) = vary * R2 /double(MSx) * (priordf_gen + 2);
            /* shape parameter for the gamma for scale parameter for marker effect */
            shapeprior = 1.1;
            /* rate parameter for the gamma (rate) for scale parameter for marker effect */
            rateprior = (shapeprior-1)/ double(ScaleMark(0,0));
            /* Initialzie b and bvar */
            for(int i = 0; i < m; i++)
            {
                b(i,0) = 0;
                bvar(i,0) = ScaleMark(0,0) / double(priordf_gen+2);
            }
        }
        if(SimParameters.getmethod() == "BayesRidgeRegression")
        {
            ScaleMark(0,0) = vary * R2 /double(MSx) * (priordf_gen + 2);
            for(int i = 0; i < m; i++)
            {
                b(i,0) = 0;
                bvar(i,0) = ScaleMark(0,0) / double(priordf_gen+2);
            }
        }
        if(SimParameters.getmethod() == "BayesB")
        {
            pi = SimParameters.getinitpi();
            priorcount = 10;
            priorcountin = priorcount * double(SimParameters.getinitpi());
            priorcountout = priorcount - priorcountin;
            /* Scale parameter for the marker variance (i.e. (vary(y) * R2 / MSx * (priordf_gen + 2))) / pi */
            ScaleMark(0,0) = (vary * R2 /double(MSx) * (priordf_gen + 2)) / double(pi);
            /* shape parameter for the gamma for scale parameter for marker effect */
            shapeprior = 1.1;
            /* rate parameter for the gamma for scale parameter for marker effect */
            rateprior = (shapeprior-1)/ double(ScaleMark(0,0));
            /* Initialzie b, bvar and probability of inclusion */
            std::binomial_distribution<int> initialinout(1,pi);
            for(int i = 0; i < m; i++)
            {
                b(i,0) = 0;
                bvar(i,0) = ScaleMark(0,0) / double(priordf_gen+2);
                d.push_back(double(initialinout(gen)));
                ppa.push_back(0);
            }
        }
        if(SimParameters.getmethod() == "BayesC")
        {
            pi = SimParameters.getinitpi();
            priorcount = 10;
            priorcountin = priorcount * double(SimParameters.getinitpi());
            priorcountout = priorcount - priorcountin;
            /* Scale parameter for the marker variance (i.e. (vary(y) * R2 / MSx * (priordf_gen + 2))) / pi */
            ScaleMark(0,0) = (vary * R2 /double(MSx) * (priordf_gen + 2)) / double(pi);
            /* Initialzie b, bvar and probability of inclusion */
            std::binomial_distribution<int> initialinout(1,pi);
            for(int i = 0; i < m; i++)
            {
                b(i,0) = 0;
                bvar(i,0) = ScaleMark(0,0);
                d.push_back(double(initialinout(gen)));
                ppa.push_back(0);
            }
        }
    } else
    {
        /************************************/
        /* Scale parameter for the residual */
        /************************************/
        for(int i = 0; i < n; i++){vary += (y(i,0)-meany) * (y(i,0)-meany);}
        vary = vary / (double(n-1));
        vare(0,0) = vary*(1-R2);
        scaleres = vary*(1-R2)*(priordf_res+ 2);
        //vare(0,0) = atof(numbers[1].c_str());
        //scaleres = atof(numbers[1].c_str())*(priordf_res+ 2);
        /*************************************************************************************/
        /* Prior parameters for the part associated with markers (i.e. varies across methods */
        /*************************************************************************************/
        if(SimParameters.getmethod() == "BayesA")
        {
            /* Scale parameter for the marker variance (i.e. vary(y) * R2 / MSx * (priordf_gen + 2)) */
            ScaleMark(0,0) = vary * R2 /double(MSx) * (priordf_gen + 2);
            /* shape parameter for the gamma for scale parameter for marker effect */
            shapeprior = 1.1;
            /* rate parameter for the gamma (rate) for scale parameter for marker effect */
            rateprior = (shapeprior-1)/ double(ScaleMark(0,0));
            //* Initialzie b and bvar */
            for(int i = 0; i < m; i++)
            {
                size_t pos = numbers[3].find(" ",0);
                b(i,0) = atof((numbers[3].substr(0,pos)).c_str()); numbers[3].erase(0, pos + 1);
                bvar(i,0) = ScaleMark(0,0) / double(priordf_gen+2);
            }
        }
        if(SimParameters.getmethod() == "BayesRidgeRegression")
        {
            ScaleMark(0,0) = vary * R2 /double(MSx) * (priordf_gen + 2);
            for(int i = 0; i < m; i++)
            {
                size_t pos = numbers[2].find(" ",0);
                b(i,0) = atof((numbers[2].substr(0,pos)).c_str()); numbers[2].erase(0, pos + 1);
                bvar(i,0) = ScaleMark(0,0) / double(priordf_gen+2);
            }
        }
        if(SimParameters.getmethod() == "BayesB")
        {
            pi = atof(numbers[3].c_str());;
            priorcount = 10;
            priorcountin = priorcount * pi;
            priorcountout = priorcount - priorcountin;
            /* Scale parameter for the marker variance (i.e. (vary(y) * R2 / MSx * (priordf_gen + 2))) / pi */
            ScaleMark(0,0) = (vary * R2 /double(MSx) * (priordf_gen + 2)) / double(pi);
            /* shape parameter for the gamma for scale parameter for marker effect */
            shapeprior = 1.1;
            /* rate parameter for the gamma for scale parameter for marker effect */
            rateprior = (shapeprior-1)/ double(ScaleMark(0,0));
            /* Initialzie b, bvar and probability of inclusion */
            std::binomial_distribution<int> initialinout(1,pi);
            for(int i = 0; i < m; i++)
            {
                size_t pos = numbers[4].find(" ",0);
                b(i,0) = atof((numbers[4].substr(0,pos)).c_str()); numbers[4].erase(0, pos + 1);
                bvar(i,0) = ScaleMark(0,0) / double(priordf_gen+2);
                d.push_back(double(initialinout(gen)));
                ppa.push_back(0);
            }
        }
        if(SimParameters.getmethod() == "BayesC")
        {
            pi = atof(numbers[2].c_str());;
            priorcount = 10;
            priorcountin = priorcount * pi;
            priorcountout = priorcount - priorcountin;
            /* Scale parameter for the marker variance (i.e. (vary(y) * R2 / MSx * (priordf_gen + 2))) / pi */
            ScaleMark(0,0) = (vary * R2 /double(MSx) * (priordf_gen + 2)) / double(pi);
            std::binomial_distribution<int> initialinout(1,pi);
            for(int i = 0; i < m; i++)
            {
                size_t pos = numbers[3].find(" ",0);
                b(i,0) = atof((numbers[3].substr(0,pos)).c_str()); numbers[3].erase(0, pos + 1);
                bvar(i,0) = ScaleMark(0,0);
                d.push_back(double(initialinout(gen)));
                ppa.push_back(0);
            }
        }
    }
    logfileloc << "       - Size of training population: " << n << endl;
    logfileloc << "       - Prior Parameters:" << endl;;
    logfileloc << "         - Prior residual degrees of freedom for Residual: " << priordf_res << endl;
    logfileloc << "         - Scale parameter for prior Residual: " << scaleres << endl;
    logfileloc << "         - Prior residual degrees of freedom assigned to the variance of markers: " << priordf_gen << endl;
    logfileloc << "         - Scale parameter for prior assigned to the variance of markers: " << ScaleMark(0,0) << endl;
    if(SimParameters.getmethod() == "BayesA" || SimParameters.getmethod() == "BayesB")
    {
        logfileloc << "         - Gamma shape parameter for prior assigned to scale parameter of variance of markers: " << shapeprior << endl;
        logfileloc << "         - Gamma rate parameter for prior assigned to scale parameter of variance of markers: " << rateprior << endl;
    }
    if(SimParameters.getmethod() == "BayesB" || SimParameters.getmethod() == "BayesC")
    {
        logfileloc << "         - pi value: " << pi << endl;
    }
    /* temporary variables used within mcmc sampling */
    double rhs_int, C_int, mean_int, sd_int, rhs_mark, C_mark, mean_mark, gamma_rate, gamma_shape, bc_RSS, bc_logodds;
    double bc_old_d, bc_markin, tempshape1, tempshape2;
    int currentsavediteration = 1;
    logfileloc << "       - Begin MCMC Sampling." << endl;
    time_t overstart = time(0);
    for (int iter=0; iter < SimParameters.getnumiter(); iter++)
    {
        time_t start = time(0);
        double tempvar = 0.0;
        /*=================*/
        /*sample intercept */
        /*=================*/
        #pragma omp parallel for reduction(+:tempvar)
        for(int i = 0; i < n; i++){e(i,0) +=  mu(0,0); tempvar += e(i,0);}     /* add back intercept to residuals and sum */
        rhs_int = tempvar / double(vare(0,0));
        C_int = n / double(vare(0,0));
        mean_int = rhs_int / double(C_int);
        sd_int = sqrt(1/double(C_int));
        std::normal_distribution<double> normal(mean_int,sd_int);
        mu(0,0) = normal(gen);
        #pragma omp parallel for
        for(int i = 0; i < n; i++){e(i,0) -= mu(0,0);}     /* subtract off new intercept to residuals */
        /*==============================*/
        /* sample effect for each locus */
        /*==============================*/
        MatrixXd tmpa;
        for(int marker = 0; marker < m; marker++)
        {
            if(SimParameters.getmethod() == "BayesA" || SimParameters.getmethod() == "BayesRidgeRegression")
            {
                tmpa =  (add.col(marker).transpose()*e) / vare(0,0);
                rhs_mark = tmpa(0,0);
                double beta = b(marker,0);
                rhs_mark += (x2[marker]*beta) / vare(0,0);
                C_mark = (x2[marker]/vare(0,0)) + (1.0/bvar(marker,0));
                std::normal_distribution<double> normalL((rhs_mark/double(C_mark)),sqrt(1.0/double(C_mark)));
                b(marker,0) = normalL(gen);
                double newbeta = beta - b(marker,0);
                e = e + add.col(marker)*newbeta;  /* correct locus to residuals; everything adjusted except for this locus */
                /* update posterior mean calculation */
                if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
                {
                    postb[marker] = ((postb[marker]*(currentsavediteration-1))+ b(marker,0)) / double(currentsavediteration);
                }
            }
            if(SimParameters.getmethod() == "BayesB" || SimParameters.getmethod() == "BayesC")
            {
                tmpa = (add.col(marker).transpose()*e);
                if(d[marker] == 1)
                {
                    /* included in model */
                    bc_RSS = (-1 * b(marker,0) * b(marker,0) * x2[marker]) - (2 * b(marker,0) * tmpa(0,0));
                } else{
                    /* not included in model */
                    bc_RSS = (b(marker,0) * b(marker,0) * x2[marker]) - (2 * b(marker,0) * tmpa(0,0));
                }
                bc_logodds = log(pi/double(1-pi)) + ((-0.5/double(vare(0,0)))*bc_RSS);
                tempvar = 1.0 / double(1.0+exp(-bc_logodds)); /* probability of being in included in model */
                bc_old_d = d[marker];
                /* Determine whether to include or exclude */
                if(uniform_bc(gen) < tempvar)
                {
                    d[marker] = 1;
                } else{ d[marker] = 0;}
                /* if status of inclusion changed update residuals */
                if(bc_old_d != d[marker])
                {
                    if(bc_old_d == 0 && d[marker] == 1)
                    {
                        e = e - add.col(marker)*b(marker,0);
                        tmpa = (add.col(marker).transpose()*e);
                    } else {
                        /* changed form 1 to 0 */
                        e = e + add.col(marker)*b(marker,0);
                    }
                }
                /* Now sample the effects */
                if(d[marker] == 0)
                {
                    std::normal_distribution<double> normal_0_1(0,1);
                    b(marker,0) = sqrt(bvar(marker,0))*normal_0_1(gen);
                    
                } else {
                    double beta = b(marker,0);
                    rhs_mark = ((x2[marker]*beta) + tmpa(0,0))/ vare(0,0);
                    C_mark = (x2[marker]/vare(0,0)) + (1.0/bvar(marker,0));
                    std::normal_distribution<double> normalL((rhs_mark/double(C_mark)),sqrt(1.0/double(C_mark)));
                    b(marker,0) = normalL(gen);
                    double newbeta = beta - b(marker,0);
                    e = e + add.col(marker)*newbeta;  /* correct locus to residuals; everything adjusted except for this locus */
                }
                ppa[marker] += d[marker];
                /* update posterior mean calculation */
                if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
                {
                    postb[marker] = ((postb[marker]*(currentsavediteration-1))+ b(marker,0)) / double(currentsavediteration);
                }
            }
        }
        /*================================*/
        /*   update variance for markers  */
        /*================================*/
        if(SimParameters.getmethod() == "BayesA" || SimParameters.getmethod() == "BayesB")            /* Different across loci */
        {
            double sumvar = 0.0;
            #pragma omp parallel for reduction(+:sumvar)
            for(int marker = 0; marker < m; marker++)
            {
                bvar(marker,0) = (ScaleMark(0,0) + (b(marker,0)*b(marker,0))) / double(distr_chisq_u(gen));
                if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
                {
                    postvarB[marker] = ((postvarB[marker]*(currentsavediteration-1))+ bvar(marker,0)) / double(currentsavediteration);
                }
                sumvar += 1/bvar(marker,0);               /* used to sample gamma */
            }
            /* Update Scale Marker variance paramter based on gamma */
            gamma_shape = m*priordf_gen/double(2) + shapeprior;
            gamma_rate = sumvar/double(2) + rateprior;
            std::gamma_distribution <double> dist_gamma(gamma_shape,(1/double(gamma_rate)));
            ScaleMark(0,0) = dist_gamma(gen);
            if(SimParameters.getmethod() == "BayesB" && SimParameters.getpie_f() == "estimate")
            {
                bc_markin = 0;
                for(int i = 0; i < m; i++){bc_markin += d[i];}
                tempshape1 = bc_markin + priorcountin + 1;
                tempshape2 = (m - bc_markin + priorcountout + 1);
                sftrabbit::beta_distribution<> beta(tempshape1,tempshape2); /* Beta Distribution */
                pi = beta(gen);
                if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
                {
                    postpi = ((postpi*(currentsavediteration-1))+pi) / double(currentsavediteration);
                    samples_pi[currentsavediteration-1] = pi;
                }
            }
        }
        if(SimParameters.getmethod() == "BayesRidgeRegression" || SimParameters.getmethod() == "BayesC")        /* Same across loci */
        {
            double sumvar = 0.0;
            #pragma omp parallel for reduction(+:sumvar)
            for(int marker = 0; marker < m; marker++){sumvar += ((b(marker,0)*b(marker,0)));}
            tempvar = (sumvar + ScaleMark(0,0)) / double(distr_chisq_u1(gen));
            ScaleMark(0,0) = tempvar;
            #pragma omp parallel for
            for(int i = 0; i < m; i++){bvar(i,0) = tempvar;}
            if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
            {
                postvarb = ((postvarb*(currentsavediteration-1))+bvar(0,0)) / double(currentsavediteration);
            }
            if(SimParameters.getmethod() == "BayesC" && SimParameters.getpie_f() == "estimate")
            {
                bc_markin = 0;
                for(int i = 0; i < m; i++){bc_markin += d[i];}
                tempshape1 = bc_markin + priorcountin + 1;
                tempshape2 = (m - bc_markin + priorcountout + 1);
                sftrabbit::beta_distribution<> beta(tempshape1,tempshape2); /* Beta Distribution */
                pi = beta(gen);
                if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
                {
                    postpi = ((postpi*(currentsavediteration-1))+pi) / double(currentsavediteration);
                    samples_pi[currentsavediteration-1] = pi;
                }
            }
        }
        /* update yhat */
        #pragma omp parallel for
        for(int i = 0; i < n; i++){yhat(i,0) = y(i,0) - e(i,0);}
        /*====================*/
        /*sample Var residual */
        /*====================*/
        tmpa =  (e.transpose()*e);
        vare(0,0) = (tmpa(0,0) + scaleres) / double(distr_chisq_e(gen));
        if((iter+1) > SimParameters.getburnin() && (iter+1) % SimParameters.getthin() == 0)
        {
            /* update posterior mean based on ((mean * (size of current mean)) + current sample) / total samples */
            postintercept = ((postintercept*(currentsavediteration-1))+ mu(0,0)) / double(currentsavediteration);
            postresidual = ((postresidual*(currentsavediteration-1))+ vare(0,0)) / double(currentsavediteration);
            if(SimParameters.getmethod() == "BayesA" || SimParameters.getmethod() == "BayesB")
            {
                postmarkervar = ((postmarkervar*(currentsavediteration-1))+ ScaleMark(0,0)) / double(currentsavediteration);
                samples_scalemark[currentsavediteration-1] = ScaleMark(0,0);
            }
            samples_res[currentsavediteration-1] = vare(0,0);
            if(SimParameters.getmethod() == "BayesRidgeRegression" || SimParameters.getmethod() == "BayesC")
            {
                samples_varb[currentsavediteration-1] = bvar(0,0);
            }
            iterationnumber[currentsavediteration-1] = iter+1;
            currentsavediteration += 1;
        }
        time_t end = time(0);
        if(iter % 1000 == 0 || iter == (SimParameters.getnumiter()-1))
        {
            if(SimParameters.getmethod() == "BayesB" && SimParameters.getpie_f() == "estimate")
            {
                logfileloc << "         - Iter= "<<iter+1<<"; VarE= "<<vare(0,0)<<"; VarS= " <<ScaleMark(0,0)<<"; Pi= "<< pi << endl;
            }
            if((SimParameters.getmethod() == "BayesB" && SimParameters.getpie_f() == "fix") || SimParameters.getmethod() == "BayesA")
            {
                logfileloc << "         - Iter= "<<iter+1<<"; VarE= "<<vare(0,0)<<"; VarS= " <<ScaleMark(0,0)<<endl;
            }
            if(SimParameters.getmethod() == "BayesRidgeRegression")
            {
                logfileloc << "         - Iter= "<<iter+1<<"; VarE= "<<vare(0,0)<<"; VarB= " <<bvar(0,0)<<endl;
            }
            if(SimParameters.getmethod() == "BayesC" && SimParameters.getpie_f() == "fix")
            {
                logfileloc << "         - Iter= "<<iter+1<<"; VarE= "<<vare(0,0)<<"; VarB= " <<bvar(0,0)<<"; Pi= "<< pi << endl;
            }
            if(SimParameters.getmethod() == "BayesC" && SimParameters.getpie_f() == "estimate")
            {
                logfileloc << "         - Iter= "<<iter+1<<"; VarE= "<<vare(0,0)<<"; VarS= " <<bvar(0,0)<<"; Pi= "<< pi << endl;
            }
        }
        //if(iter == 500){exit (EXIT_FAILURE);}
    }
    time_t overend = time(0);
    /* Plot posterior mean of a few summary statistics */
    if(SimParameters.getmethod() == "BayesA")
    {
        logfileloc << "         - Posterior mean residual variance: " << postresidual << endl;
        logfileloc << "         - Posterior mean scale parameter for markers: " << postmarkervar << endl;
        /* Save posteriormeans to file for priors for next generation */
        fstream checkbayes; checkbayes.open(Bayes_PosteriorMeans.c_str(), std::fstream::out | std::fstream::trunc); checkbayes.close();
        std::ofstream flagbayes(Bayes_PosteriorMeans.c_str(), std::ios_base::app | std::ios_base::out);
        flagbayes << "PosteriorMeansPrior" << endl;
        flagbayes << postresidual << endl;
        flagbayes << postmarkervar << endl;
        flagbayes << postb[0];
        for(int i = 1; i < m; i++){flagbayes << " " << postb[i];}
        flagbayes << endl << postvarB[0];
        for(int i = 1; i < m; i++){flagbayes << " " << postvarB[i];}
        flagbayes << endl;
    }
    if(SimParameters.getmethod() == "BayesRidgeRegression")
    {
        logfileloc << "         - Posterior mean residual variance: " << postresidual << endl;
        logfileloc << "         - Posterior mean marker variance: " << postvarb << endl;
        /* Save posteriormeans to file for priors for next generation */
        fstream checkbayes; checkbayes.open(Bayes_PosteriorMeans.c_str(), std::fstream::out | std::fstream::trunc); checkbayes.close();
        std::ofstream flagbayes(Bayes_PosteriorMeans.c_str(), std::ios_base::app | std::ios_base::out);
        flagbayes << "PosteriorMeansPrior" << endl;
        flagbayes << postresidual << endl;
        flagbayes << postb[0];
        for(int i = 1; i < m; i++){flagbayes << " " << postb[i];}
        flagbayes << endl;
        flagbayes << postvarb << endl;
    }
    if(SimParameters.getmethod() == "BayesB")
    {
        logfileloc << "         - Posterior mean residual variance: " << postresidual << endl;
        logfileloc << "         - Posterior mean scale parameter for markers: " << postmarkervar << endl;
        if(SimParameters.getpie_f() == "estimate"){logfileloc << "         - Posterior mean pi parameter: " << postpi << endl;}
        /* Save posteriormeans to file for priors for next generation */
        fstream checkbayes; checkbayes.open(Bayes_PosteriorMeans.c_str(), std::fstream::out | std::fstream::trunc); checkbayes.close();
        std::ofstream flagbayes(Bayes_PosteriorMeans.c_str(), std::ios_base::app | std::ios_base::out);
        flagbayes << "PosteriorMeansPrior" << endl;
        flagbayes << postresidual << endl;
        flagbayes << postmarkervar << endl;
        if(SimParameters.getpie_f() == "estimate"){flagbayes << postpi << endl;}
        if(SimParameters.getpie_f() == "fix"){flagbayes << SimParameters.getinitpi() << endl;}
        flagbayes << postb[0];
        for(int i = 1; i < m; i++){flagbayes << " " << postb[i];}
        flagbayes << endl << postvarB[0];
        for(int i = 1; i < m; i++){flagbayes << " " << postvarB[i];}
        flagbayes << endl << d[0];
        for(int i = 1; i < m; i++){flagbayes << " " << d[i];}
        flagbayes << endl;
    }
    if(SimParameters.getmethod() == "BayesC")
    {
        logfileloc << "         - Posterior mean residual variance: " << postresidual << endl;
        logfileloc << "         - Posterior mean marker variance: " << postvarb << endl;
        if(SimParameters.getpie_f() == "estimate"){logfileloc << "         - Posterior mean pi parameter: " << postpi << endl;}
        /* Save posteriormeans to file for priors for next generation */
        fstream checkbayes; checkbayes.open(Bayes_PosteriorMeans.c_str(), std::fstream::out | std::fstream::trunc); checkbayes.close();
        std::ofstream flagbayes(Bayes_PosteriorMeans.c_str(), std::ios_base::app | std::ios_base::out);
        flagbayes << "PosteriorMeansPrior" << endl;
        flagbayes << postresidual << endl;
        if(SimParameters.getpie_f() == "estimate"){flagbayes << postpi << endl;}
        if(SimParameters.getpie_f() == "fix"){flagbayes << SimParameters.getinitpi() << endl;}
        flagbayes << postb[0];
        for(int i = 1; i < m; i++){flagbayes << " " << postb[i];}
        flagbayes << endl;
        flagbayes << postvarb << endl;
        flagbayes << d[0];
        for(int i = 1; i < m; i++){flagbayes << " " << d[i];}
        flagbayes << endl;
    }
    logfileloc << "       - Finished MCMC Sampling (Took: " << difftime(overend,overstart) << " seconds)."<< endl << endl;
    /* Delete old mcmc run and output saved mcmc samples */
    //stringstream s2; s2 << Gen; string tempvara = s2.str();
    //string tempfile = Bayes_MCMC_Samples + tempvara;
    fstream checkmcmc; checkmcmc.open(Bayes_MCMC_Samples.c_str(), std::fstream::out | std::fstream::trunc); checkmcmc.close();
    int outiteration = SimParameters.getburnin()+1;
    std::ofstream mcmcfile(Bayes_MCMC_Samples.c_str(), std::ios_base::out);               /* open mcmc file to output samples */
    if(SimParameters.getmethod() == "BayesA" || SimParameters.getmethod() == "BayesRidgeRegression"){mcmcfile << "Sample Residual MarkerVar" <<endl;}
    if(SimParameters.getmethod() == "BayesC" && SimParameters.getpie_f() == "estimate"){mcmcfile << "Sample Residual MarkerVar Pi" <<endl;}
    if(SimParameters.getmethod() == "BayesC" && SimParameters.getpie_f() == "fix"){mcmcfile << "Sample Residual MarkerVar" <<endl;}
    if(SimParameters.getmethod() == "BayesB" && SimParameters.getpie_f() == "estimate"){mcmcfile << "Sample Residual MarkerVar Pi" <<endl;}
    if(SimParameters.getmethod() == "BayesB" && SimParameters.getpie_f() == "fix"){mcmcfile << "Sample Residual MarkerVar" <<endl;}
    for (int i = 0; i < samples_res.size(); i++)
    {
        if(SimParameters.getmethod() == "BayesB")
        {
            mcmcfile << iterationnumber[i] << " " << samples_res[i] << " " << samples_scalemark[i];
            if(SimParameters.getpie_f() == "estimate"){mcmcfile << " " << samples_pi[i] << endl;}
            if(SimParameters.getpie_f() == "fix"){mcmcfile << endl;}
        }
        if(SimParameters.getmethod() == "BayesC")
        {
            mcmcfile << iterationnumber[i] << " " << samples_res[i] << " " << samples_varb[i];
            if(SimParameters.getpie_f() == "estimate"){mcmcfile << " " << samples_pi[i] << endl;}
            if(SimParameters.getpie_f() == "fix"){mcmcfile << endl;}
        }
        if(SimParameters.getmethod() == "BayesA"){mcmcfile << iterationnumber[i] << " " << samples_res[i] << " " << samples_scalemark[i] << endl;}
        if(SimParameters.getmethod() == "BayesRidgeRegression"){mcmcfile << iterationnumber[i] << " " << samples_res[i] << " " << samples_varb[i] << endl;}
        outiteration++;
    }
    mcmcfile.close();
    for(int i = 0; i < n; i++){estimatedsolutions.push_back(0.0);}
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++){estimatedsolutions[i] += postb[j]*add(i,j);}
    }
    //for(int i = 0; i < 10; i++){cout << estimatedsolutions[i] << endl;}
    /* Update Animal Class with EBV's */
    for(int i = 0; i < population.size(); i++)
    {
        int j = 0;                                                  /* Counter for population spot */
        while(j < animalid.size())
        {
            if(population[i].getID() == animalid[j]){population[i].Update_EBV(estimatedsolutions[j]); break;}
            j++;
        }
    }
    if(tagdecreasewindow == "GeneratePriorH2")
    {
        /* Since give better prior value and put close to stationary distribution half number of iterations and burnins */
        SimParameters.Updatenumiter(SimParameters.getnumiterstat());
        SimParameters.Updateburnin(SimParameters.getburninstat());
    }
}
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/********************                                  Functions Not Used Anymore                                    ********************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/****************************************************************************************************************************************/
/***********************************************************/
/* Solve for solutions using cholesky decomposition of LHS */
/***********************************************************/
void direct_solver(parameters &SimParameters,vector <Animal> &population,double* lhs, double* rhs, vector < double > &solutionsa, vector <double> &trueaccuracy, int dimen)
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
    /* Generate Accuracy */
    double lambda = (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd());
    //cout << "Lambda: " << (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd()) << endl;
    //for(int i = 1; i < 10; i++){cout << lhs[(i*n)+i] << endl;}
    for(int i = 1; i < dimen; i++){trueaccuracy[i-1] = sqrt(1 - (lhs[(i*n)+i]*lambda));}
    //for(int i = 0; i < 9; i++){cout << accuracy[i-1] << endl;}
    const long long int lhssize = dimen;
    const long long int onesize = 1;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,lhssize,onesize,lhssize,1.0,lhs,lhssize,rhs,onesize,0.0,solutions,onesize);
    #pragma omp parallel for
    for(int i = 0; i < dimen; i++){solutionsa[i] = solutions[i];}                                 /* set solutions to zero & copy rhs to e */
    delete [] solutions;
}
/**************************************************/
/* EBV generated based on pblup, gblup or rohblup */
/**************************************************/
void p_or_g_blup(parameters &SimParameters,vector <Animal> &population,double* relationshipinv,vector <double> &Phenotype,vector <int> &animal,double scalinglambda, vector <double> &estimatedsolutions, vector <double> &trueaccuracy, ostream& logfileloc)
{
    int LHSsize = animal.size() + 1;
    if(estimatedsolutions.size() == 0)
    {
        for(int i = 0; i < LHSsize; i++)
        {
            estimatedsolutions.push_back(0.0);
            if(i > 0){trueaccuracy.push_back(0.0);}
        }
    }
    double* LHSarray = new double[LHSsize*LHSsize];                                 /* LHS dimension: animal + intercept by animal + intercept */
    for(int i = 0; i < LHSsize*LHSsize; i++){LHSarray[i] = 0;}
    /* fill LHSarray */
    for(int i = 0; i < LHSsize; i++)
    {
        for(int j = 0; j < LHSsize; j++)
        {
            if(i == 0 && j == 0){LHSarray[(i*LHSsize)+j] = animal.size();}
            if(i == 0 && j > 0){LHSarray[(i*LHSsize)+j] = 1;}
            if(i > 0 && j == 0){LHSarray[(i*LHSsize)+j] = 1;}
            if(i > 0 && j > 0)
            {
                if(i == j){LHSarray[(i*LHSsize)+j] = 1 + (relationshipinv[((i-1)*animal.size())+(j-1)] * double(scalinglambda));}
                if(i != j){LHSarray[(i*LHSsize)+j] = (relationshipinv[((i-1)*animal.size())+(j-1)] * double(scalinglambda));}
            }
        }
    }
    double* RHSarray = new double[LHSsize];                                                     /* RHS dimension: animal + intercept by 1 */
    /* fill RHSarray */
    for(int i = 0; i < LHSsize; i++){RHSarray[i] = 0;}
    for(int i = 0; i < animal.size(); i++){RHSarray[0] += Phenotype[i];}                    /* row 1 of RHS is sum of observations */
    for(int i = 0; i < animal.size(); i++){RHSarray[i+1] = Phenotype[i];}                   /* Copy phenotypes to RHS */
    logfileloc << "           - RHS created, Dimension (" << LHSsize << " X " << 1 << ")." << endl;
    logfileloc << "           - LHS created, Dimension (" << LHSsize << " X " << LHSsize << ")." << endl;
    if(SimParameters.getSolver() == "direct")                                                   /* Solve equations using direct inversion */
    {
        logfileloc << "           - Starting " << SimParameters.getSolver() << "." << endl;
        time_t start = time(0);
        direct_solver(SimParameters,population,LHSarray,RHSarray,estimatedsolutions,trueaccuracy,LHSsize);
        
        double meanacc = 0.0; double minacc = trueaccuracy[0]; double maxacc = trueaccuracy[0];
        for(int i = 0; i < trueaccuracy.size(); i++)
        {
            meanacc += trueaccuracy[i];
            if(trueaccuracy[i] < minacc){minacc = trueaccuracy[i];}
            if(trueaccuracy[i] > maxacc){maxacc = trueaccuracy[i];}
        }
        meanacc /= trueaccuracy.size();
        if(SimParameters.getEBV_Calc()=="pblup" && SimParameters.getConstructGFreq()!="observed" && SimParameters.getGener()==SimParameters.getreferencegenblup())
        {
            logfileloc << "              - Mean Accuracy: " << meanacc << " (Min: " << minacc << " - Max: " << maxacc << ")" << endl;
        }
        if(SimParameters.getEBV_Calc()=="gblup" && SimParameters.getConstructGFreq()!="observed" && SimParameters.getGener()==SimParameters.getreferencegenblup())
        {
            logfileloc << "              - Mean Accuracy: " << meanacc << " (Min: " << minacc << " - Max: " << maxacc << ")" << endl;
        }
        time_t end = time(0);
        logfileloc << "       - Finished Solving Equations created." << endl;
        logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
    }
    if(SimParameters.getSolver() == "pcg")                                                      /* Solve equations using pcg */
    {
        logfileloc << "           - Starting " << SimParameters.getSolver() << "." << endl;
        time_t start = time(0);
        int* solvedatiteration = new int[1]; solvedatiteration[0] = 0;
        //pcg_solver(LHSarray,RHSarray,estimatedsolutions,LHSsize,solvedatiteration);
        time_t end = time(0);
        logfileloc << "           - PCG converged at iteration " << solvedatiteration[0] << "." << endl;
        logfileloc << "       - Finished Solving Equations created." << endl;
        logfileloc << "               - Took: " << difftime(end,start) << " seconds." << endl;
        delete[] solvedatiteration;
    }
    /* Update Animal Class with EBV's and associated Accuracy (if option is direct) */
    for(int i = 0; i < population.size(); i++)
    {
        int j = 0;                                                  /* Counter for population spot */
        while(j < animal.size())
        {
            if(population[i].getID() == animal[j])
            {
                population[i].Update_EBV(estimatedsolutions[j+1]);
                if(SimParameters.getSolver() == "direct")
                {
                    if(SimParameters.getEBV_Calc()!="gblup" && SimParameters.getConstructGFreq()!="observed" && SimParameters.getGener()==SimParameters.getreferencegenblup())
                    {
                        population[i].Update_Acc(trueaccuracy[j]);
                    }
                }
                break;
            }
            j++;
        }
    }
    //for(int i = 0; i < population.size(); i++)
    //{
    //    if(population[i].getSex() == 0)
    //    {
    //        cout<<population[i].getID()<<" "<<population[i].getProgeny()<<" "<<population[i].getEBV()<<" "<<population[i].getAcc()<<endl;
    //    }
    //}
    delete[] LHSarray; delete[] RHSarray;
}
/*********************************************************/
/* Generate A inverse based on Meuwissen & Luo Algorithm */
/*********************************************************/
void A_inverse_function(vector <Animal> &population,string Pheno_Pedigree_File, vector <int> &animal, vector <double> &Phenotype,  vector <int> trainanimals, double* ainv, ostream& logfileloc)
{
    vector <int> sire(animal.size(),0); vector <int> dam(animal.size(),0);
    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
    int linenumber = 0; int tempanim; int tempanimindex = 0; int indexintrainanim = 0;  /* Counter to determine where at in pedigree index's */
    string line;
    ifstream infile2;
    infile2.open(Pheno_Pedigree_File.c_str());                                                  /* This file has all animals in it */
    if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line))
    {
        size_t pos = line.find(" ", 0); tempanim = (std::stoi(line.substr(0,pos)));
        if(trainanimals.size() == 0)
        {
            animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
            pos = line.find(" ",0); sire[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);             /* Grab Sire ID */
            //if(sire[tempanimindex] < animgreatkeep){sire[tempanimindex] = 0;}
            pos = line.find(" ",0); dam[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);              /* Grab Dam ID */
            //if(dam[tempanimindex] < animgreatkeep){dam[tempanimindex] = 0;}
            Phenotype[tempanimindex] = stod(line); tempanimindex++;                                                     /* Grab Phenotype */
        } else {
            if(tempanim == trainanimals[indexintrainanim])
            {
                animal[tempanimindex] = tempanim; line.erase(0, pos + 1);   /* Grab animal id */
                pos = line.find(" ",0); sire[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);             /* Grab Sire ID */
                //if(sire[tempanimindex] < animgreatkeep){sire[tempanimindex] = 0;}
                pos = line.find(" ",0); dam[tempanimindex] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);              /* Grab Dam ID */
                //if(dam[tempanimindex] < animgreatkeep){dam[tempanimindex] = 0;}
                Phenotype[tempanimindex] = stod(line); tempanimindex++;                                                     /* Grab Phenotype */
                indexintrainanim++;
            }
        }
        linenumber++;
    }
    /* Zero out ainv */
    for(int i = 0; i < animal.size(); i++)
    {
        for(int j = 0; j < animal.size(); j++){ainv[(i*animal.size())+j] = 0.0;}
    }
    if(animal[0] != 1)
    {
        vector < int > renum_animal(animal.size(),0);
        vector < int > renum_sire(animal.size(),0);
        vector < int > renum_dam(animal.size(),0);
        for(int i = 0; i < animal.size(); i++)
        {
            renum_animal[i] = i+1;
            int temp = animal[i];
            for(int j = 0; j < animal.size(); j++)
            {
                /* change it if sire or dam */
                if(temp == sire[j]){renum_sire[j] = renum_animal[i];}
                if(temp == dam[j]){renum_dam[j] = renum_animal[i];}
            }
        }
        //for(int i = 0; i < renum_animal.size(); i++){cout<<renum_animal[i]<<" "<<renum_sire[i]<<" "<<renum_dam[i] << "\t";}
        //cout << endl;
        //exit (EXIT_FAILURE);
        int animnumb = animal.size();
        vector < double > F((animnumb+1),0.0);
        vector < double > D(animnumb,0.0);
        /* This it makes so D calculate is correct */
        F[0] = -1;
        for(int k = renum_animal[0]; k < (animnumb+1); k++)                  /* iterate through each row of l */
        {
            vector < double > L(animnumb,0.0);
            vector < double > AN;                                       // Holds all ancestors of individuals
            double ai = 0.0;                                            /* ai  is the inbreeding */
            AN.push_back(k);                                            /* push back animal in ancestor */
            L[k-1] = 1.0;
            /* Calculate D */
            D[k-1] = 0.5 - (0.25 * (F[renum_sire[k-1]] + F[renum_dam[k-1]]));
            int j = k;                                          /* start off at K then go down */
            while(AN.size() > 0)
            {
                /* if sire is know add to AN and add to L[j] */
                if((renum_sire[j-1]) != 0){AN.push_back(renum_sire[j-1]); L[renum_sire[j-1]-1] += 0.5 * L[j-1];}
                /* if dam is known add to AN and add to L[j] */
                if((renum_dam[j-1]) != 0){AN.push_back(renum_dam[j-1]); L[renum_dam[j-1]-1] += 0.5 * L[j-1];}
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
            if(renum_sire[k-1] != 0 && renum_dam[k-1] != 0) /* indexed by (row * f_anim.size()) + col */
            {
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] + bi;
                ainv[((renum_sire[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_sire[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] - (bi/2);
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] - (bi/2);
                ainv[((renum_dam[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_dam[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] - (bi/2);
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] - (bi/2);
                ainv[((renum_sire[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] = ainv[((renum_sire[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] + (bi/4);
                ainv[((renum_sire[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] = ainv[((renum_sire[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] + (bi/4);
                ainv[((renum_dam[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] = ainv[((renum_dam[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] + (bi/4);
                ainv[((renum_dam[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] = ainv[((renum_dam[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] + (bi/4);
            }
            if(renum_sire[k-1] != 0 && renum_dam[k-1] == 0)
            {
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] + bi;
                ainv[((renum_sire[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_sire[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] - (bi/2);
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] - (bi/2);
                ainv[((renum_sire[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] = ainv[((renum_sire[k-1]-1) * animnumb) + (renum_sire[k-1]-1)] + (bi/4);
            }
            if(renum_sire[k-1] == 0 && renum_dam[k-1] != 0)
            {
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] + bi;
                ainv[((renum_dam[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_dam[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] - (bi/2);
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] - (bi/2);
                ainv[((renum_dam[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] = ainv[((renum_dam[k-1]-1) * animnumb) + (renum_dam[k-1]-1)] + (bi/4);
            }
            if(renum_sire[k-1] == 0 && renum_dam[k-1] == 0)
            {
                ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] = ainv[((renum_animal[k-1]-1) * animnumb) + (renum_animal[k-1]-1)] + bi;
            }
        }
    }
    if(animal[0] == 1)
    {
        int animnumb = animal.size();
        vector < double > F((animnumb+1),0.0);
        vector < double > D(animnumb,0.0);
        /* This it makes so D calculate is correct */
        F[0] = -1;
        for(int k = animal[0]; k < (animnumb+1); k++)                  /* iterate through each row of l */
        {
            vector < double > L(animnumb,0.0);
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
            double bi = (1/sqrt(D[k-1])) * (1/sqrt(D[k-1]));
            if(sire[k-1] != 0 && dam[k-1] != 0) /* indexed by (row * f_anim.size()) + col */
            {
                ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] + bi;
                ainv[((sire[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((sire[k-1]-1) * animnumb) + (animal[k-1]-1)] - (bi/2);
                ainv[((animal[k-1]-1) * animnumb) + (sire[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (sire[k-1]-1)] - (bi/2);
                ainv[((dam[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((dam[k-1]-1) * animnumb) + (animal[k-1]-1)] - (bi/2);
                ainv[((animal[k-1]-1) * animnumb) + (dam[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (dam[k-1]-1)] - (bi/2);
                ainv[((sire[k-1]-1) * animnumb) + (sire[k-1]-1)] = ainv[((sire[k-1]-1) * animnumb) + (sire[k-1]-1)] + (bi/4);
                ainv[((sire[k-1]-1) * animnumb) + (dam[k-1]-1)] = ainv[((sire[k-1]-1) * animnumb) + (dam[k-1]-1)] + (bi/4);
                ainv[((dam[k-1]-1) * animnumb) + (sire[k-1]-1)] = ainv[((dam[k-1]-1) * animnumb) + (sire[k-1]-1)] + (bi/4);
                ainv[((dam[k-1]-1) * animnumb) + (dam[k-1]-1)] = ainv[((dam[k-1]-1) * animnumb) + (dam[k-1]-1)] + (bi/4);
            }
            if(sire[k-1] != 0 && dam[k-1] == 0)
            {
                ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] + bi;
                ainv[((sire[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((sire[k-1]-1) * animnumb) + (animal[k-1]-1)] - (bi/2);
                ainv[((animal[k-1]-1) * animnumb) + (sire[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (sire[k-1]-1)] - (bi/2);
                ainv[((sire[k-1]-1) * animnumb) + (sire[k-1]-1)] = ainv[((sire[k-1]-1) * animnumb) + (sire[k-1]-1)] + (bi/4);
            }
            if(sire[k-1] == 0 && dam[k-1] != 0)
            {
                ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] + bi;
                ainv[((dam[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((dam[k-1]-1) * animnumb) + (animal[k-1]-1)] - (bi/2);
                ainv[((animal[k-1]-1) * animnumb) + (dam[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (dam[k-1]-1)] - (bi/2);
                ainv[((dam[k-1]-1) * animnumb) + (dam[k-1]-1)] = ainv[((dam[k-1]-1) * animnumb) + (dam[k-1]-1)] + (bi/4);
            }
            if(sire[k-1] == 0 && dam[k-1] == 0)
            {
                ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] = ainv[((animal[k-1]-1) * animnumb) + (animal[k-1]-1)] + bi;
            }
        }
    }
    /* copy vector back to output_ped array */
    //for(int i = 1; i < (animnumb+1); i++){fped[i-1] = F[i];}
    /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
    //for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
    //{
    //    int j = 0;                                                                  /* Counter for population spot */
    //    while(1)
    //    {
    //        if(population[i].getID() == animal[j]){double temp = fped[j]; population[i].UpdateInb(temp); break;}
    //        j++;                                                                    /* Loop across until animal has been found */
    //    }
    //}
    //cout << animal.size() << endl;
    //int numbernonzeros = 0;
    //for(int i = 0; i < animal.size(); i++)
    //{
    //    for(int j = 0; j < animal.size(); j++)
    //    {
    //        if(ainv[(i*animal.size())+j] > 0){cout << i << " " << j << " " <<ainv[(i*animal.size())+j] << " -- "; numbernonzeros += 1;}
    //    }
    //}
    //cout << endl;
    //cout << numbernonzeros << endl;
    //for(int i = (animal.size()-15); i < animal.size(); i++)
    //{
    //    for(int j = (animal.size()-15); j < animal.size(); j++){cout << ainv[(i*animal.size())+j] << "\t";}
    //    cout << endl;
    //}
    //cout << endl << endl;
}

