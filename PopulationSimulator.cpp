#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <ctime>
#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "Animal.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include <mkl.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>

using namespace std;
/*******************************************/
/* Functions from Simulation_Functions.cpp */
/*******************************************/
void pedigree_inverse(vector <int> const &f_anim, vector <int> const &f_sire, vector <int> const &f_dam, vector<double> &output,vector < double > &output_f);
void pedigree_inbreeding(string phenotypefile, double* output_f);
void ld_decay_estimator(string outputfile, string mapfile, string lineone, vector < string > const &genotypes);
void frequency_calc(vector < string > const &genotypes, double* output_freq);
void grm_noprevgrm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);
void grm_prevgrm(double* input_m, string genofile, vector < string > const &newgenotypes, double* output_grm12, double* output_grm22, float scaler,vector < int > &animalvector, vector < double > &phenotypevector);
void generatesummaryqtl(string inputfilehap, string inputfileqtl, string outputfile, int generations,vector < int > const &idgeneration, vector < double > const &tempaddvar, vector < double > const &tempdomvar,  vector < int > const &tempdeadfit);
void generatessummarydf(string inputfilehap, string outputfile, int generations, vector < double > const &tempexphet);
void pcg_solver(double* lhs, double* rhs, vector < double > &solutionsa, int dimen, int* solvediter);
void direct_solver(double* lhs, double* rhs, vector < double > &solutionsa, int dimen);

void pedigree_relationship(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship);


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
/* Functions to output and input eigen matrices into binary format */
namespace Eigen
{
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix)
    {
        std::ofstream out(filename,ios::out | ios::binary | ios::trunc); typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index)); out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) ); out.close();
    }
    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix)
    {
        std::ifstream in(filename,ios::in | std::ios::binary); typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index)); in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols); in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar)); in.close();
    }
}

int main(int argc, char* argv[])
{
    /* Ensure all executables have the correct permissions */
    system("chmod 755 macs"); system("chmod 755 msformatter"); system("chmod 755 GenoDiver");
    time_t fullbegin_time = time(0);
    cout<<"########################################################################\n";
    cout<<"#########################################################   ############\n";
    cout<<"#####################################################   /~|   ##########\n";
    cout<<"##################################################   _- `~~~', #########\n";
    cout<<"################################################  _-~       )  #########\n";
    cout<<"#############################################  _-~          |  #########\n";
    cout<<"##########################################  _-~            ;  ##########\n";
    cout<<"################################  __---___-~              |   ##########\n";
    cout<<"#############################   _~   ,,                  ;  `,,  #######\n";
    cout<<"###########################  _-~    ;'                  |  ,'  ; #######\n";
    cout<<"#########################  _~      '                    `~'   ; ########\n";
    cout<<"##################   __---;                                 ,' #########\n";
    cout<<"##############   __~~  ___                                ,' ###########\n";
    cout<<"###########  _-~~   -~~ _          N                    ,' #############\n";
    cout<<"########### `-_         _           C                  ; ###############\n";
    cout<<"#############  ~~----~~~   ;         S                ; ################\n";
    cout<<"###############  /          ;         U               ; ################\n";
    cout<<"#############  /             ;                      ; ##################\n";
    cout<<"###########  /                `                    ; ###################\n";
    cout<<"#########  /                                      ; ####################\n";
    cout<<"#######                                            #####################\n";
    cout<<"########################################################################\n";
    cout<<"------------------------------------------------------------------------\n";
    cout<<"- GENO-DIVER                                                           -\n";
    cout<<"- Complex Genomic Simulator                                            -\n";
    cout<<"- Authors: Jeremy T. Howard (jthoward@ncsu.edu)                        -\n";
    cout<<"-          Francesco Tiezzi (f_tiezzi@ncsu.edu)                        -\n";
    cout<<"-          Jennie Pryce (jennie.pryce@ecodev.vic.gov.au)               -\n";
    cout<<"-          Christian Maltecca (cmaltec@ncsu.edu)                       -\n";
    cout<<"- Institution: NCSU                                                    -\n";
    cout<<"- Date: Ongoing                                                        -\n";
    cout<<"- This program is free software: you can redistribute it and/or modify -\n";
    cout<<"- it under the terms of the GNU General Public License as published by -\n";
    cout<<"- the Free Software Foundation, either version 3 of the License, or    -\n";
    cout<<"- (at your option) any later version.                                  -\n";
    cout<<"------------------------------------------------------------------------\n";
    /* Figure out where you are currently at then just append to string */
    char * cwd;
    cwd = (char*) malloc( FILENAME_MAX * sizeof(char) );
    getcwd(cwd,FILENAME_MAX);
    string path(cwd);
    if(argc != 2){cout << "Program ended due to a parameter file not given!" << endl; exit (EXIT_FAILURE);}
    string paramterfile = argv[1];
    /* Initialize all of the variables that will be called when reading in parameter file */
    string StartSim, Selection, EBV_Calc, Solver, Geno_Inverse, Mating, Culling, SelectionDir, Create_LD_Decay;
    string outputfolder, OutputGeno, Recomb_Distribution, Ne_spec;
    int seednumber, numchra, qtla, qtlb, qtlc, fnd_haplo, haplo_size, ne_founder,GENERATIONS, SIRES, DAMS, OffspringPerMating, MaximumAge;
    int outputgeneration, nthread, maxmating, replicates;
    double ThresholdMAFQTL, ThresholdMAFMark, LowerThresholdMAFFitnesslethal, LowerThresholdMAFFitnesssublethal, Gamma_Shape, Gamma_Scale;
    double Normal_varRelDom, Gamma_Shape_Lethal,Gamma_Scale_Lethal, Normal_meanRelDom_Lethal, Normal_varRelDom_Lethal, Normal_meanRelDom;
    double Gamma_Shape_SubLethal, Gamma_Scale_SubLethal, Normal_meanRelDom_SubLethal, Normal_varRelDom_SubLethal, PropQTL;
    double Variance_Additiveh2, Variance_Dominanceh2, SireReplacement, DamReplacement, BetaDist_alpha;
    double BetaDist_beta, proppleitropic, genetic_correlation;
    float u;
    /* read parameters file */
    vector <string> parm;
    string parline;
    ifstream parfile;
    parfile.open(paramterfile);
    if(parfile.fail()){cout << "Parameter file not found. Check log file." << endl; exit (EXIT_FAILURE);}
    while (getline(parfile,parline)){parm.push_back(parline);} /* Stores in vector and each new line push back to next space */
    int search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("START:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); StartSim = parm[search]; break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'START:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(StartSim != "sequence" && StartSim != "founder")
    {
        cout << endl << "START (" << StartSim << ") didn't equal sequence or founder! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("OUTPUTFOLDER:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); outputfolder = parm[search].c_str(); break;
        }
        search++;
        if(search >= parm.size()){outputfolder = "GenoDiverFiles"; break;}
    }
    /* Need to remove all the files within GenoDiverFiles */
    if(StartSim == "sequence")
    {
        string rmfolder = "rm -rf ./" + outputfolder;
        system(rmfolder.c_str());
        rmfolder = "mkdir " + outputfolder;
        system(rmfolder.c_str());
    }
    if(StartSim == "founder")
    {
        std::string x = path + "/" + outputfolder + "/";
        const char *folderr = x.c_str();
        struct stat sb;
        if (stat(folderr, &sb) == 0 && S_ISDIR(sb.st_mode)){}
        else
        {
            cout << endl << " FOLDER DOESN'T EXIST. CHANGE TO 'START: sequence' TO FILL FOLDER IF USING FOUNDER AS START!\n" << endl;
        }
    }
    /* Files to drop things in */
    string logfileloc = path + "/" + outputfolder + "/log_file.txt";
    string lowfitnesspath = path + "/" + outputfolder + "/Low_Fitness";
    string snpfreqfileloc = path + "/" + outputfolder + "/SNPFreq";
    string foundergenofileloc = path + "/" + outputfolder + "/FounderGenotypes";
    string qtl_class_object = path + "/" + outputfolder + "/QTL_new_old_Class";
    string Pheno_Pedigree_File = path + "/" + outputfolder + "/Pheno_Pedigree";
    string Pheno_GMatrix_File = path + "/" + outputfolder + "/Pheno_GMatrix";
    string Master_DF_File = path + "/" + outputfolder + "/Master_DF";
    string Master_Genotype_File = path + "/" + outputfolder + "/Master_Genotypes";
    string BinaryG_Matrix_File = path + "/" + outputfolder + "/G_Matrix";
    string Binarym_Matrix_File = path + "/" + outputfolder + "/m_Matrix";
    string Binaryp_Matrix_File = path + "/" + outputfolder + "/p_Matrix";
    string BinaryLinv_Matrix_File = path + "/" + outputfolder + "/Linv_Matrix";
    string BinaryGinv_Matrix_File = path + "/" + outputfolder + "/Ginv_Matrix";
    string Marker_Map = path + "/" + outputfolder + "/Marker_Map";
    string LD_Decay_File = path + "/" + outputfolder + "/LD_Decay";
    string Master_DataFrame_path = path + "/" + outputfolder + "/Master_DataFrame";
    string Summary_QTL_path = path + "/" + outputfolder + "/Summary_Statistics_QTL";
    string Summary_DF_path = path + "/" + outputfolder + "/Summary_Statistics_DataFrame";
    /* Read in parameters */
    search = 0; string seedstring;
    while(1)
    {
        size_t fnd = parm[search].find("SEED:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            seedstring = "        - Seed Number:\t\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            seednumber = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            seednumber = time(0); stringstream s1; s1 << seednumber; string tempvar = s1.str();
            seedstring = "        - Seed Number:\t\t\t\t\t\t\t\t\t\t\t'" + tempvar + "' (System Clock)\n"; break;
        }
    }
    search = 0; string threadstring;
    while(1)
    {
        size_t fnd = parm[search].find("NTHREAD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            threadstring = "        - Number of threads:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            nthread = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){nthread = 1; threadstring = "        - Number of threads:\t\t\t\t\t\t\t\t\t\t'1' (Default)\n"; break;}
    }
    omp_set_num_threads(nthread);
    mkl_set_num_threads_local(nthread);
    search = 0; string nrepstring;
    while(1)
    {
        size_t fnd = parm[search].find("NREP:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            nrepstring = "        - Number of replicates:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            replicates = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){replicates = 1; nrepstring = "        - Number of replicates:\t\t\t\t\t\t\t\t\t\t'1'\n"; break;}
    }
    nrepstring = nrepstring + "    - Genome and Marker:\n";
    search = 0; string chrstring;
    while(1)
    {
        size_t fnd = parm[search].find("CHR:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            chrstring = "        - Number of Chromosomes:\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            numchra = atoi(parm[search].c_str()); break;
        }
        search++; if(search > parm.size()){cout << endl << "Couldn't find 'CHR:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    const int numChr=numchra; float ChrLength [numChr];
    /* This should be the same length as CHR_LENGTH if not will exit the program*/
    search = 0; string chrlengthstring;
    while(1)
    {
        size_t fnd = parm[search].find("CHR_LENGTH:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            int timescycled = 0;
            for(int i = 0; i < 31; i++)
            {
                size_t posinner = parm[search].find(" ",0);
                if(posinner > 0)                                                                         /* hasn't reached last one yet */
                {
                    string temp = parm[search].substr(0,posinner);
                    ChrLength[i] = strtod(temp.c_str(),NULL); ChrLength[i] = ChrLength[i] * 1000000;
                    parm[search].erase(0, posinner + 1); timescycled++;
                }
                if(posinner == std::string::npos){i = 31;}
            }
            if(timescycled != numChr){cout << endl << "CHR_LENGTH " << timescycled << " doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);}
            break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'CHR_LENGTH:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    /* Check to ensure isn't 0 which can happen if leave a blank when you give one less */
    for(int i = 0; i < numChr; i++){if(ChrLength[i] == 0){cout << endl << "Incorrect Chromosome Length of 0!" << endl; exit (EXIT_FAILURE);}}
    chrlengthstring = "        - Length of Chromosome: \n";
    for(int i = 0; i < numChr; i++)
    {
        stringstream s1; s1 << ChrLength[i]; string tempvar = s1.str();
        stringstream s2; s2 << i + 1; string tempvara = s2.str();
        chrlengthstring = chrlengthstring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
    }
    int numMark [numChr];
    search = 0; string nummarkerstring;
    while(1)
    {
        size_t fnd = parm[search].find("NUM_MARK:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            int timescycled = 0;
            for(int i = 0; i < 31; i++)
            {
                size_t posinner = parm[search].find(" ",0);
                if(posinner > 0)                                                                         /* hasn't reached last one yet */
                {
                    string temp = parm[search].substr(0,posinner);
                    numMark[i] = atoi(temp.c_str()); parm[search].erase(0, posinner + 1); timescycled++;
                }
                if(posinner == std::string::npos){i = 31;}
            }
            if(timescycled != numChr){cout << endl << "NUM_MARK " << timescycled << " doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);}
            break;
        }
        search++;
        if(search >= parm.size()){cout << endl << "Couldn't find 'NUM_MARK: ' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    for(int i = 0; i < numChr; i++){if(numMark[i] == 0){cout << endl << "Incorrect Number of Marker value of 0!" << endl; exit (EXIT_FAILURE);}}
    nummarkerstring = "        - Number of Markers: \n";
    for(int i = 0; i < numChr; i++)
    {
        stringstream s1; s1 << numMark[i]; string tempvar = s1.str();
        stringstream s2; s2 << i + 1; string tempvara = s2.str();
        nummarkerstring = nummarkerstring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
    }
    search = 0; string markermafstring;
    while(1)
    {
        size_t fnd = parm[search].find("MARKER_MAF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            markermafstring = "        - Minor Allele Frequency Threshold for Markers:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            ThresholdMAFMark = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            ThresholdMAFMark = 0.05; markermafstring = "        - Minor Allele Frequency Threshold for Markers:\t\t\t\t\t\t\t'0.05' (Default)\n"; break;
        }
    }
    if(ThresholdMAFMark >= 0.5){cout << endl << "Marker MAF threshold can't be greater than 0.5! Check parameter file!" << endl; exit (EXIT_FAILURE);}
    search = 0; string numqtlstring;
    while(1)
    {
        size_t fnd = parm[search].find("QTL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            numqtlstring = "        - Number of Quantitative Trait QTL by Chromosome:\t\t\t\t\t\t'" + parm[search] + "'\n";
            qtla = atoi(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'QTL:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    const int numQTL = qtla;
    search = 0; string qtlmafstring;
    while(1)
    {
        size_t fnd = parm[search].find("QUANTITATIVE_MAF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            qtlmafstring = "        - Minor Allele Frequency Threshold for Quantitative Trait QTL's:\t\t\t\t'" + parm[search] + "'\n";
            ThresholdMAFQTL = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            ThresholdMAFQTL = 0.05; qtlmafstring = "        - Minor Allele Frequency Threshold for Quantitative Trait QTL's:\t\t\t\t'0.05' (Default)\n"; break;
        }
    }
    if(ThresholdMAFQTL >= 0.5){cout << endl << "Quantitative MAF threshold can't be greater than 0.5! Check parameter file!" << endl; exit (EXIT_FAILURE);}
    search = 0; string lethfitnessstring;
    while(1)
    {
        size_t fnd = parm[search].find("FIT_LETHAL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); qtlb = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){qtlb = 0; break;}
    }
    const int numFitnessQTLlethal = qtlb;
    if(qtlb > 0)
    {
        stringstream s1; s1 << numFitnessQTLlethal; string tempvar = s1.str();
        lethfitnessstring = "        - Number of Lethal Fitness Trait QTL by Chromosome:\t\t\t\t\t\t'" + tempvar + "'\n";
    }
    if(qtlb == 0)
    {
        stringstream s1; s1 << numFitnessQTLlethal; string tempvar = s1.str();
        lethfitnessstring = "        - Number of Lethal Fitness Trait QTL by Chromosome:\t\t\t\t\t\t'" + tempvar + "' (Default)\n";
    }
    search = 0; string sublethfitnessstring;
    while(1)
    {
        size_t fnd = parm[search].find("FIT_SUBLETHAL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); qtlc = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){qtlc = 0; break;}
    }
    const int numFitnessQTLsublethal = qtlc;
    if(qtlc > 0)
    {
        stringstream s1; s1 << numFitnessQTLsublethal; string tempvar = s1.str();
        sublethfitnessstring = "        - Number of Sub-Lethal Fitness Trait QTL by Chromosome:\t\t\t\t\t\t'" + tempvar + "'\n";
    }
    if(qtlc == 0)
    {
        stringstream s1; s1 << numFitnessQTLsublethal; string tempvar = s1.str();
        sublethfitnessstring = "        - Number of Sub-Lethal Fitness Trait QTL by Chromosome:\t\t\t\t\t\t'" + tempvar + "' (Default)\n";
    }
    if((numQTL+numFitnessQTLlethal+numFitnessQTLsublethal) > 5000)
    {
        cout << endl << "Cannot have greater than 5000 QTL! Check parameter file." << endl; exit (EXIT_FAILURE);
    }
    search = 0; string fitnessmafstring;
    while(1)
    {
        size_t fnd = parm[search].find("FITNESS_MAF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            pos = parm[search].find(" ",0);
            /* should at least be two */
            if(pos == std::string::npos){cout  << endl << "        - Should be two values not one for fitness MAF!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            fitnessmafstring = "        - Upper Minor Allele Frequency Threshold for Fitness Lethal QTL's:\t\t\t\t'" + temp + "'\n";
            LowerThresholdMAFFitnesslethal = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            fitnessmafstring=fitnessmafstring+"        - Upper Minor Allele Frequency Threshold for Fitness Sub Lethal QTL's:\t\t\t\t'"+parm[search]+"'\n";
            LowerThresholdMAFFitnesssublethal = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            LowerThresholdMAFFitnesslethal = 0.02; LowerThresholdMAFFitnesssublethal = 0.08;
            fitnessmafstring = "        - Upper Minor Allele Frequency Threshold for Fitness Lethal QTL's:\t\t\t\t'0.02' (Default)\n";
            fitnessmafstring=fitnessmafstring+"        - Upper Minor Allele Frequency Threshold for Fitness Sub Lethal QTL's:\t\t\t\t'0.08' (Default)\n"; break;
        }
    }
    if(LowerThresholdMAFFitnesslethal >= 0.5 || LowerThresholdMAFFitnesssublethal >= 0.5)
    {
        cout << endl << "Fitness MAF upper threshold can't be greater than 0.5! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0; string haplosizestring;
    while(1)
    {
        size_t fnd = parm[search].find("HAPLOTYPE_SIZE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            haplosizestring = "        - SNP Haplotype Size:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            haplo_size = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){haplo_size=50; haplosizestring = "        - SNP Haplotype Size:\t\t\t\t\t\t\t\t\t\t'50' (Default)\n";break;}
    }
    const int haplotypesize = haplo_size;
    search = 0; string recombinstring;
    while(1)
    {
        size_t fnd = parm[search].find("RECOMBINATION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); Recomb_Distribution = parm[search];
            recombinstring = "        - Recombination Position generated from the following distribution:\t\t\t\t'" + Recomb_Distribution + "'\n"; break;
        }
        search++;
        if(search >= parm.size())
        {
            Recomb_Distribution = "Uniform";
            recombinstring = "        - Recombination Position generated from the following distribution:\t\t\t\t'" + Recomb_Distribution + "' (Default)\n";
            break;
        }
    }
    if(Recomb_Distribution != "Uniform" && StartSim != "Beta")
    {
        cout << endl << "RECOMBINATION (" << Recomb_Distribution << ") didn't equal Uniform or Beta! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    recombinstring = recombinstring + "    - QTL Effects: \n";
    search = 0; string addquanstring;
    while(1)
    {
        size_t fnd = parm[search].find("ADD_QUAN:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            pos = parm[search].find(" ",0);
            /* should at least be two */
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for ADD_QUAN!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            addquanstring = "        - Gamma Shape: Additive Quantitative Trait:\t\t\t\t\t\t\t'" + temp + "'\n";
            Gamma_Shape = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            addquanstring = addquanstring + "        - Gamma Scale: Additive Quantitative Trait:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            Gamma_Scale = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            Gamma_Shape = 0.4; Gamma_Scale = 1.66;
            addquanstring = "        - Gamma Shape: Additive Quantitative Trait:\t\t\t\t\t\t\t'0.4' (Default)\n";
            addquanstring = addquanstring + "        - Gamma Scale: Additive Quantitative Trait:\t\t\t\t\t\t\t'1.66' (Default)\n"; break;
        }
    }
    search = 0; string domquanstring;
    while(1)
    {
        size_t fnd = parm[search].find("DOM_QUAN:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for DOM_QUAN!" << endl; exit (EXIT_FAILURE);}
            
            string temp = parm[search].substr(0,pos);
            domquanstring = "        - Normal Mean: Dominance Quantitative Trait:\t\t\t\t\t\t\t'" + temp + "'\n";
            Normal_meanRelDom = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            domquanstring = domquanstring + "        - Normal SD: Dominance Quantitative Trait:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            Normal_varRelDom = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            Normal_meanRelDom = 0.1; Normal_varRelDom = 0.2;
            domquanstring = "        - Normal Mean: Dominance Quantitative Trait:\t\t\t\t\t\t\t'0.1' (Default)\n";
            domquanstring = domquanstring + "        - Normal SD: Dominance Quantitative Trait:\t\t\t\t\t\t\t'0.2' (Default)\n"; break;
        }
    }
    search = 0; string lethaladdstring;
    while(1)
    {
        size_t fnd = parm[search].find("LTHA:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for LTHA!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            lethaladdstring = "        - Gamma Shape: S value Fitness Lethal:\t\t\t\t\t\t\t\t'"+temp+"'\n";
            Gamma_Shape_Lethal = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            lethaladdstring = lethaladdstring + "        - Gamma Scale: S value Fitness Lethal:\t\t\t\t\t\t\t\t'"+parm[search]+"'\n";
            Gamma_Scale_Lethal = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            Gamma_Shape_Lethal = 1.6; Gamma_Scale_Lethal = 0.1;
            lethaladdstring = "        - Gamma Shape: S value Fitness Lethal:\t\t\t\t\t\t\t\t'1.6' (Default)\n";
            lethaladdstring = lethaladdstring+"        - Gamma Scale: S value Fitness Lethal:\t\t\t\t\t\t\t\t'0.1' (Default)\n"; break;
        }
    }
    search = 0; string lethaldomstring;
    while(1)
    {
        size_t fnd = parm[search].find("LTHD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for LTHD!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            lethaldomstring = "        - Normal Mean: Dominance Fitness Lethal:\t\t\t\t\t\t\t'" + temp + "'\n";
            Normal_meanRelDom_Lethal = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            lethaldomstring = lethaldomstring + "        - Normal Variance: Dominance Fitness Lethal:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            Normal_varRelDom_Lethal = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            Normal_meanRelDom_Lethal = 0.05; Normal_varRelDom_Lethal = 0.1;
            lethaldomstring = "        - Normal Mean: Dominance Fitness Lethal:\t\t\t\t\t\t\t'0.05' (Default)\n";
            lethaldomstring = lethaldomstring + "        - Normal Variance: Dominance Fitness Lethal:\t\t\t\t\t\t\t'0.1' (Default)\n"; break;
        }
    }
    search = 0; string sublethaladdstring;
    while(1)
    {
        size_t fnd = parm[search].find("SUBA:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for SUBA!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            sublethaladdstring = "        - Gamma Shape: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'" + temp + "'\n";
            Gamma_Shape_SubLethal = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            sublethaladdstring = sublethaladdstring + "        - Gamma Scale: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            Gamma_Scale_SubLethal = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            Gamma_Shape_SubLethal  = 0.1; Gamma_Scale_SubLethal = 0.2;
            sublethaladdstring = "        - Gamma Shape: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'0.1' (Default)\n";
            sublethaladdstring = sublethaladdstring + "        - Gamma Scale: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'0.2' (Default)\n"; break;
        }
    }
    search = 0; string sublethaldomstring;
    while(1)
    {
        size_t fnd = parm[search].find("SUBD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for SUBD!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            sublethaldomstring = "        - Normal Mean: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t\t'" + temp + "'\n";
            Normal_meanRelDom_SubLethal = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            sublethaldomstring = sublethaldomstring + "        - Normal Variance: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t'" + parm[search] + "'\n";
            Normal_varRelDom_SubLethal = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            Normal_meanRelDom_SubLethal = 0.3; Normal_varRelDom_SubLethal = 0.1;
            sublethaldomstring = "        - Normal Mean: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t\t'0.3' (Default)\n";
            sublethaldomstring = sublethaldomstring + "        - Normal Variance: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t'0.1' (Default)\n"; break;
        }
    }
    search = 0; string covarstring;
    while(1)
    {
        size_t fnd = parm[search].find("COVAR:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for COVAR!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            covarstring = "        - Proportion of QTL with Pleiotropic Fitness Effects :\t\t\t\t\t\t'" + temp + "'\n";
            proppleitropic = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            covarstring = covarstring + "        - Correlation (Rank) between QTL and Fitness Effects:\t\t\t\t\t\t'" + parm[search] + "'\n";
            genetic_correlation = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            proppleitropic = 0; genetic_correlation = 0;
            covarstring = "        - Proportion of QTL with Pleiotropic Fitness Effects:\t\t\t\t\t\t'0.0' (Default)\n";
            covarstring = covarstring +"        - Additive Genetic Correlation between QTL and Fitness Effects:\t\t\t\t\t'0.0' (Default)\n"; break;
        }
    }
    if(genetic_correlation < 0)
    {
        cout << "        - Correlation can't be negative due to how effects are constructed. Alter Favorable direction!" << endl; exit (EXIT_FAILURE);
    }
    covarstring = covarstring + "    - Population Parameters:\n";
    search = 0; string nefounderstring;
    ne_founder = -10;
    while(1)
    {
        size_t fnd = parm[search].find("FOUNDER_Effective_Size:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            if(parm[search]=="Ne70"||parm[search]=="Ne100_Scen1"||parm[search]=="Ne100_Scen2"||parm[search]=="Ne250"||parm[search]=="Ne1000")
            {
                Ne_spec = parm[search]; ne_founder = -5; break;
            }
            if(parm[search]=="CustomNe")
            {
                Ne_spec = parm[search]; ne_founder = -5; break;
            }
            if(parm[search]!="Ne70"||parm[search]!="Ne100_Scen1"||parm[search]!="Ne100_Scen2"||parm[search]!="Ne250"||parm[search]!="Ne1000"||parm[search]!="CustomNe")
            {
                Ne_spec = ""; ne_founder = atoi(parm[search].c_str()); break;
            }
        }
        search++; if(search >= parm.size()){cout<<endl<<"Couldn't find 'FOUNDER_Effective_Size:' variable in parameter file!"<<endl;exit(EXIT_FAILURE);}
    }
    if(ne_founder == -10 || ne_founder == 0)         /* will exit if 0 because can't be zero and if give a letter value will default to zero */
    {
        cout << endl << "        - Wrong parameter for Founder Effective Population Size. Check Manual." << endl; exit (EXIT_FAILURE);
    }
    const int Ne = ne_founder;
    if(Ne != -5 && Ne_spec == "")
    {
        stringstream s1; s1 << Ne; string tempvar = s1.str();
        nefounderstring = "        - Effective Population Size in Founders:\t\t\t\t\t\t\t'" + tempvar + "'\n";
    }
    if(Ne == -5 && Ne_spec != ""){nefounderstring = "        - Effective Population Modeled from the scenario:\t\t\t\t\t\t'" + Ne_spec + "'\n";}
    search = 0; string mutationstring;
    while(1)
    {
        size_t fnd = parm[search].find("MUTATION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            pos = parm[search].find(" ",0);
            if(pos == std::string::npos){cout << endl << "        - Should be two values not one for MUTATION!" << endl; exit (EXIT_FAILURE);}
            string temp = parm[search].substr(0,pos);
            mutationstring = "        - Mutation Rate:\t\t\t\t\t\t\t\t\t\t'" + temp + "'\n";
            u = strtod(temp.c_str(),NULL); parm[search].erase(0, pos + 1);
            mutationstring = mutationstring + "        - Proportion of mutations that can be QTL:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            PropQTL = atof(parm[search].c_str()); break;

        }
        search++; if(search >= parm.size())
        {
            u = 2.5e-8; PropQTL = 0.0;
            mutationstring = "        - Mutation Rate:\t\t\t\t\t\t\t\t\t\t'2.5e-8' (Default)\n";
            mutationstring = mutationstring + "        - Proportion of mutations that can be QTL:\t\t\t\t\t\t\t'0.0' (Default)\n"; break;
        }
    }
    search = 0; string heritstring;
    while(1)
    {
        size_t fnd = parm[search].find("VARIANCE_A:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            heritstring = "        - Variance due to additive gene action (Phenotypic variance is 1.0):\t\t\t\t'" + parm[search] + "'\n";
            Variance_Additiveh2 = atof(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'VARIANCE_A:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(Variance_Additiveh2 < 0 || Variance_Additiveh2 > 1.0)
    {
        cout << endl << "VARIANCE_A outside of range (0.0 - 1.0)! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VARIANCE_D:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            heritstring = heritstring + "        - Variance due to dominant gene action (Phenotypic variance is 1.0):\t\t\t\t'" + parm[search] + "'\n";
            Variance_Dominanceh2 = atof(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'VARIANCE_D:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(Variance_Dominanceh2 < 0 || Variance_Dominanceh2 > 1.0)
    {
        cout << endl << "VARIANCE_D outside of range (0.0 - 1.0)! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    heritstring = heritstring + "    - Selection and Mating Parameters\n";
    search = 0; string genetstring;
    while(1)
    {
        size_t fnd = parm[search].find("GENERATIONS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            genetstring = "        - Number of Generations to Simulate:\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            GENERATIONS = atoi(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'GENERATIONS:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0; string individualsstring;
    while(1)
    {
        size_t fnd = parm[search].find("INDIVIDUALS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0); string temp = parm[search].substr(0,pos);
            individualsstring = "        - Number of Males in Population per Generation:\t\t\t\t\t\t\t'" + temp + "'\n";
            SIRES = atoi(temp.c_str()); parm[search].erase(0, pos + 1); pos = parm[search].find(" ",0); temp = parm[search].substr(0,pos);
            individualsstring = individualsstring + "        - Sire Replacement Rate:\t\t\t\t\t\t\t\t\t'" + temp + "'\n";
            SireReplacement = atof(temp.c_str()); parm[search].erase(0, pos + 1); pos = parm[search].find(" ",0); temp = parm[search].substr(0,pos);
            individualsstring = individualsstring + "        - Number of Females in Population per Generation:\t\t\t\t\t\t'" + temp + "'\n";
            DAMS = atoi(temp.c_str()); parm[search].erase(0, pos + 1);
            individualsstring = individualsstring + "        - Dam Replacement Rate:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            DamReplacement = atof(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'INDIVIDUALS:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SireReplacement <= 0 || SireReplacement >= 1.0){cout << endl << "Replacement rate outside of range (0.0 - 1.0). Check Parameterfile" << endl;}
    if(DamReplacement <= 0 || DamReplacement >= 1.0){cout << endl << "Replacement rate outside of range (0.0 - 1.0). Check Parameterfile" << endl;}
    search = 0; string founderhaplostring;
    while(1)
    {
        size_t fnd = parm[search].find("FOUNDER_HAPLOTYPES:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            fnd_haplo = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            if(numFitnessQTLlethal == 0 && numFitnessQTLsublethal == 0){fnd_haplo = 2*(2*(DAMS+150)); break;} /* None will die so don't need as much */
            if(numFitnessQTLlethal != 0 || numFitnessQTLsublethal != 0){fnd_haplo = 2*(2*(DAMS+300)); break;} /* Some will die so need more */
        }
    }
    const int FounderHaplotypes = fnd_haplo;
    stringstream s1founderhap; s1founderhap << FounderHaplotypes; string s1founderhaptemp = s1founderhap.str();
    founderhaplostring = "        - Founder Haplotypes:\t\t\t\t\t\t\t\t\t\t'" + s1founderhaptemp + "'\n";
    int numberfoundersa;
    if(numFitnessQTLlethal == 0 && numFitnessQTLsublethal == 0){numberfoundersa = 2*(DAMS+150);}     /* None will die so don't need as much */
    if(numFitnessQTLlethal != 0 || numFitnessQTLsublethal != 0){numberfoundersa = 2*(DAMS+300);}    /* Some will die so need more */
    if(numberfoundersa > (FounderHaplotypes/2))
    {
        numberfoundersa = (FounderHaplotypes/2);
        founderhaplostring = founderhaplostring+"        - Had to reduce number of founder individuals created in order to not run out of haplotypes!\n";
    }
    const int numberfounders = numberfoundersa;
    stringstream s1founderhapb; s1founderhapb << numberfounders; string s1founderhapbtemp = s1founderhapb.str();
    founderhaplostring = founderhaplostring+"        - Number of Founder Individuals:\t\t\t\t\t\t\t\t'"+s1founderhapbtemp+"' (Based on Number of Dams)\n";
    search = 0; string paritymatestring;
    while(1)
    {
        size_t fnd = parm[search].find("PARITY_MATES_DIST:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0); string temp = parm[search].substr(0,pos);
            paritymatestring = "        - Beta Alpha - distribution of mating pairs by age:\t\t\t\t\t\t'" + temp + "'\n";
            BetaDist_alpha = atof(temp.c_str()); parm[search].erase(0, pos + 1);
            paritymatestring = paritymatestring + "        - Beta Beta - distribution of mating pairs by age:\t\t\t\t\t\t'" + parm[search] + "'\n";
            BetaDist_beta = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            BetaDist_alpha = 1.0; BetaDist_beta = 1.0;
            paritymatestring = "        - Beta Alpha - distribution of mating pairs by age:\t\t\t\t\t\t'1.0' (Default)\n";
            paritymatestring = paritymatestring + "        - Beta Beta - distribution of mating pairs by age:\t\t\t\t\t\t'1.0' (Default)\n"; break;
        }
    }
    search = 0; string progenystring;
    while(1)
    {
        size_t fnd = parm[search].find("PROGENY:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            progenystring = "        - Number of offspring for each mating pair:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            OffspringPerMating = atoi(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'PROGENY:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MAXFULLSIB:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            progenystring = progenystring + "        - Maximum Number of Full-Sibling taken per family:\t\t\t\t\t\t'" + parm[search] + "'\n";
            maxmating = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size())
        {
            maxmating = OffspringPerMating; stringstream s1; s1 << maxmating; string tempvar = s1.str();
            progenystring = progenystring + "        - Maximum Number of Full-Sibling taken per family:\t\t\t\t\t\t'" + tempvar + "'(Default)\n"; break;
        }
    }
    search = 0; string selectionstring;
    while(1)
    {
        size_t fnd = parm[search].find("SELECTION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0); string temp = parm[search].substr(0,pos);
            selectionstring = "        - Selection Criteria:\t\t\t\t\t\t\t\t\t\t'" + temp + "'\n";
            Selection = temp; parm[search].erase(0, pos + 1);
            selectionstring = selectionstring + "        - Selection Direction:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SelectionDir = parm[search]; break;
        }
        search++;
        if(search >= parm.size()){cout << endl << "Couldn't find 'SELECTION:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(Selection != "random" && Selection != "phenotype" && Selection != "true_bv" && Selection != "ebv")
    {
        cout << endl << "SELECTION (" << Selection << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    if(SelectionDir != "low" && SelectionDir != "high")
    {
        cout << endl << "SELECTIONDIR (" << SelectionDir << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0; string solverinversestring;
    while(1)
    {
        size_t fnd = parm[search].find("SOLVER_INVERSE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            vector < string > solvervariables(3,"");
            for(int i = 0; i < 3; i++)
            {
                pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear();}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            string kill = "NO";        /* Used to exit out of not in right order or not enough parameters */
            while(kill == "NO")
            {
                /* all three specified than has to be genomic based relationship matrix */
                if(solvervariables.size() == 3)
                {
                    if(solvervariables[0] == "pedigree")
                    {
                        cout << endl << "Chosen pedigree based relationship matrix only need first two options." << endl; exit (EXIT_FAILURE);
                    }
                    /* First check to see if in right order */
                    if(solvervariables[0] != "genomic" && solvervariables[0] != "ROH")
                    {
                        cout << endl << "SOLVER_INVERSE first option (" << EBV_Calc << ") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    if(solvervariables[1] != "direct" && solvervariables[1] != "pcg")
                    {
                        cout << endl << "SOLVER_INVERSE second option (" << Solver << ") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    if(solvervariables[2] != "cholesky" && solvervariables[2] != "recursion")
                    {
                        cout << endl << "SOLVER_INVERSE third option (" << Geno_Inverse << ") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    /* if didn't exit declare variables */
                    EBV_Calc = solvervariables[0]; Solver = solvervariables[1]; Geno_Inverse = solvervariables[2];
                    solverinversestring = "        - EBV estimated by:\t\t\t\t\t\t\t\t\t\t'" + EBV_Calc + "'\n";
                    solverinversestring = solverinversestring + "        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'" + Solver + "'\n";
                    solverinversestring = solverinversestring + "        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'" + Geno_Inverse + "'\n";
                    kill = "YES";
                }
                if(solvervariables.size() == 2)
                {
                    /* First check to see if in right order */
                    if(solvervariables[0] != "pedigree" && solvervariables[0] != "genomic" && solvervariables[0] != "ROH")
                    {
                        cout << endl << "SOLVER_INVERSE first option (" << EBV_Calc << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
                    }
                    if(solvervariables[1] != "direct" && solvervariables[1] != "pcg")
                    {
                        cout << endl << "SOLVER_INVERSE second option (" << Solver << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
                    }
                    EBV_Calc = solvervariables[0]; Solver = solvervariables[1]; Geno_Inverse = "cholesky";
                    
                    solverinversestring = "        - EBV estimated by:\t\t\t\t\t\t\t\t\t\t'" + EBV_Calc + "'\n";
                    solverinversestring = solverinversestring + "        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'" + Solver + "'\n";
                    if(EBV_Calc != "pedigree")
                    {
                        solverinversestring = solverinversestring + "        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'"+Geno_Inverse+"' (Default)\n";
                    }
                    kill = "YES";
                }
                if(solvervariables.size() == 1)
                {
                    /* First check to see if in right order */
                    if(solvervariables[0] != "pedigree" && solvervariables[0] != "genomic" && solvervariables[0] != "ROH")
                    {
                        cout << endl << "SOLVER_INVERSE first option (" << EBV_Calc << ") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    EBV_Calc = solvervariables[0]; Solver = "pcg"; Geno_Inverse = "cholesky";
                    solverinversestring = "        - EBV estimated by:\t\t\t\t\t\t\t\t\t\t'" + EBV_Calc + "'\n";
                    solverinversestring = solverinversestring + "        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'" + Solver + "' (Default)\n";
                    if(EBV_Calc != "pedigree")
                    {
                        solverinversestring = solverinversestring + "        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'"+Geno_Inverse+"' (Default)\n";
                    }
                    kill = "YES";
                }
                if(kill == "NO"){cout << endl << "'SOLVER_INVERSE:' variable incorrect check user manual!" << endl; exit (EXIT_FAILURE);}
            }
            break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'SOLVER_INVERSE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(EBV_Calc != "pedigree" && EBV_Calc != "genomic" && EBV_Calc != "ROH")
    {
        cout << endl << "SOLVER_INVERSE first option (" << EBV_Calc << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    if(Solver != "direct" && Solver != "pcg")
    {
        cout << endl << "SOLVER_INVERSE second option (" << Solver << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    if(Geno_Inverse != "cholesky" && Geno_Inverse != "recursion")
    {
        cout << endl << "SOLVER_INVERSE third option (" << Geno_Inverse << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0; string matingcullingstring;
    while(1)
    {
        size_t fnd = parm[search].find("MATING:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            matingcullingstring = "        - Mating Criteria:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n"; Mating = parm[search]; break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'MATING:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(Mating!="random" && Mating!="random5" && Mating!="random25" && Mating!="random125" && Mating!="minPedigree" && Mating!="minGenomic" && Mating != "minROH")
    {
        cout << endl << "MATING (" << Mating << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("CULLING:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2); pos = parm[search].find(" ",0); string temp = parm[search].substr(0,pos);
            matingcullingstring = matingcullingstring + "        - Culling Criteria:\t\t\t\t\t\t\t\t\t\t'" + temp + "'\n";
            Culling = temp; parm[search].erase(0, pos + 1);
            matingcullingstring = matingcullingstring + "        - Maximum Age Parents can Remain in Population:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            MaximumAge = atoi(parm[search].c_str()); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'CULLING:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(Culling != "random" && Culling != "phenotype" && Culling != "true_bv" && Culling != "ebv")
    {
        cout << endl << "CULLING (" << Culling << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    matingcullingstring = matingcullingstring + "    - Output Options\n";
    search = 0; string outputstring;
    while(1)
    {
        size_t fnd = parm[search].find("OUTPUT_LD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); Create_LD_Decay = parm[search];
            outputstring = "        - LD Parameters by Generation Created:\t\t\t\t\t\t\t\t'" + Create_LD_Decay + "'\n"; break;
        }
        search++;
        if(search >= parm.size())
        {
            Create_LD_Decay = "no";
            outputstring = "        - LD Parameters by Generation Not Created:\t\t\t\t\t\t\t\t'" + Create_LD_Decay + "' (Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("GENOTYPES:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            pos = parm[search].find(" ",0); string temp = parm[search].substr(0,pos);
            OutputGeno = temp;
            if(OutputGeno == "no"){outputgeneration = 0; outputstring = outputstring + "        - No genotypes are kept and placed in file.\n";}
            if(OutputGeno == "yes")
            {
                parm[search].erase(0, pos + 1);
                outputstring = outputstring + "        - Genotypes after generation " + parm[search] + " are kept and placed in file.\n";
                outputgeneration = atoi(parm[search].c_str());
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            OutputGeno="yes";outputgeneration=0;
            outputstring = outputstring +"        - All genotypes are kept and placed in file. (Default)\n";break;
        }
    }
    outputstring = outputstring +"Parameters Read in!!\n";
    parm.clear();
    /* Check to see if make sense if not exit out of program */
    if(numFitnessQTLsublethal == 0 && (proppleitropic > 0.0 || genetic_correlation != 0.0))
    {
        cout << endl << "Cannot have a genetic correlation between fitness and quantitative QTL when no sublethal QTL!" << endl;
        exit (EXIT_FAILURE);
    }
    if(numQTL > 0 && Variance_Additiveh2 == 0.0 && Variance_Dominanceh2 == 0.0)
    {
        cout << endl << "No additive or dominance variance so can't have quantitative QTL be greater than 0!" << endl;
        exit (EXIT_FAILURE);
        
    }
    if(numQTL == 0 && Variance_Additiveh2 == 0.0 && Variance_Dominanceh2 == 0.0 && Selection != "random")
    {
        cout << endl << "Cannot have selection occuring if no additive or dominance variance!" << endl;
        exit (EXIT_FAILURE);
    }
    if(Variance_Additiveh2 == 0.0 && Variance_Dominanceh2 > 0.0)
    {
        cout << endl << "Cannot have dominance variance, but no additive variance!" << endl;
        exit (EXIT_FAILURE);
    }
    /* loop across replicates */
    for(int reps = 0; reps < replicates; reps++)
    {
        time_t repbegin_time = time(0);
        if(replicates > 1)
        {
            cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~ Starting Replicate " << reps + 1 << " ~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        }
        /* add seednumber by one to generate new replicates and reprint out parameter file with updated logfile */
        if(reps > 0)
        {
            seednumber++; StartSim = "founder";
            stringstream s1; s1 << seednumber; string tempvar = s1.str();
            seedstring = "        - Seed Number:\t\t\t\t\t\t\t\t\t\t\t'" + tempvar + "' (Updated for replicate)\n";
        }
        /* Clear out log_file.txt from previous replication or simulation */
        fstream checklog; checklog.open(logfileloc, std::fstream::out | std::fstream::trunc); checklog.close();
        fstream checkmarkermap; checkmarkermap.open(Marker_Map, std::fstream::out | std::fstream::trunc); checkmarkermap.close();
        std::ofstream logfile(logfileloc, std::ios_base::out);               /* open log file to output verbage throughout code */
        logfile << "===============================================\n";
        logfile << "==        Read in Parameters from file       ==\n";
        logfile << "===============================================\n";
        logfile << "Name of parameter file was: '" << paramterfile << "'"<< endl;
        logfile << "Parameters Specified in Paramter File: " << endl;
        logfile << "    - Starting Point:" << endl;
        logfile << "        - Starting Site:\t\t\t\t\t\t\t\t\t\t" << "'" << StartSim << "'" << endl;
        logfile << "        - Files willl be placed in:\t\t\t\t\t\t\t\t\t" << "'" << outputfolder<< "'" << endl;
        logfile << seedstring << threadstring << nrepstring << chrstring << chrlengthstring << nummarkerstring << markermafstring;
        logfile << numqtlstring << qtlmafstring << lethfitnessstring << sublethfitnessstring << fitnessmafstring << haplosizestring;
        logfile << recombinstring << addquanstring << domquanstring << lethaladdstring << lethaldomstring << sublethaladdstring;
        logfile << sublethaldomstring << covarstring << nefounderstring << mutationstring << heritstring << genetstring;
        logfile << individualsstring << founderhaplostring << paritymatestring << progenystring << selectionstring;
        logfile << solverinversestring << matingcullingstring << outputstring << endl;
        /* Variables that need initialized in order to ensure global inheritance */
        using Eigen::MatrixXd; using Eigen::VectorXd;
        /* checks to see if want dominance or add + dominance variance to 0 */
        double scalefactaddh2, scalefactpdomh2;
        if(Variance_Dominanceh2 == 0.0){scalefactpdomh2 = 0.0;}                         /* scaling factor for dominance set to 0 */
        if(Variance_Dominanceh2 > 0.0){scalefactpdomh2 = 1.0;}                          /* scaling factor for dominance set to 1.0 and will change */
        if(Variance_Additiveh2 == 0.0){scalefactaddh2 = 0.0;}                           /* scaling factor for additive set to 0 */
        if(Variance_Additiveh2 > 0.0){scalefactaddh2 = 1.0;}                            /* scaling factor for additive set to 1.0 and will change*/
        string SNPFiles[numChr];                                                        /* Stores names of SNP Files */
        string MapFiles[numChr];                                                        /* Stores names of MAP Files */
        int ChrSNPLength [numChr];                                                      /* number of SNP within each chromosome */
        /* Marker Information */
        int NUMBERMARKERS = 0;
        for (int i = 0; i < numChr; i++){NUMBERMARKERS += numMark[i];}                  /* Figure out number of markers by adding up across chromosome */
        vector < int > MarkerIndex;                                                     /* Index to store where Marker should be located */
        vector < double > MarkerMapPosition;                                            /* Map position for Marker's */
        /* Quantitative QTL Information */
        vector < int > QTL_Index(5000,0);                                               /* Index to store where QTL should be located */
        vector < double > QTL_MapPosition(5000,0.0);                                    /* Map position for QTL's */
        vector < int > QTL_Type(5000,0);                                                /* Whether it was a quantitative, lethal, sublethal or both */
        vector < int > QTL_Allele(5000,0);                                              /* Which allele it is referring to */
        vector < double > QTL_Freq(5000,0.0);                                           /* Freq of QTL */
        vector < double > QTL_Add_Quan(5000,0.0);                                       /* Additive Effect quantitative */
        vector < double > QTL_Dom_Quan(5000,0.0);                                       /* Dominance Effect quantitative */
        vector < double > QTL_Add_Fit(5000,0.0);                                        /* Additive Effect fitness */
        vector < double > QTL_Dom_Fit(5000,0.0);                                        /* Dominance Effect fitness */
        /* Indicator to determine where at */
        int Marker_IndCounter = 0;                                                      /* Counter to determine where you are at in Marker index array */
        int QTL_IndCounter = 0;                                                         /* Counter to determine where you are at in QTL index array */
        int markerperChr[numChr];                                                       /* Number of Markers for each chromosome */
        int qtlperChr[numChr];                                                          /* Number of QTL per chromsome */
        vector < int > NumDeadFitness((GENERATIONS + 1),0);                             /* Number of dead due to fitness by generation */
        vector < double > AdditiveVar((GENERATIONS + 1),0.0);                           /* sum of 2pqa^2 */
        vector < double > DominanceVar((GENERATIONS + 1),0.0);                          /* sum of (2pqd)+^2 */
        vector < double > ExpectedHeter((GENERATIONS + 1),0.0);                         /* Expected Heterozygosity: (1 - p^2 - q^2) / markers */
        vector < QTL_new_old > population_QTL;                                          /* Hold in a vector of QTL_new_old Objects */
        vector < Animal > population;                                                   /* Hold in a vector of Animal Objects */
        vector < hapLibrary > haplib;                                                   /* Vector of haplotype library objects */
        const clock_t intbegin_time = clock();
        logfile << endl;
        if(StartSim == "sequence")
        {
            logfile << "============================================================\n";
            logfile << "===\t MaCS Part of Program (Chen et al. 2009) \t====\n";
            logfile << "============================================================\n";
            logfile << " Begin Generating Sequence Information: " << endl;
            if(Ne == -5 && Ne_spec != "")
            {
                if(Ne_spec == "CustomNe")
                {
                    vector <string> parm_custom_ne;
                    string parline_custom_ne;
                    ifstream parfile_custom_ne;
                    parfile_custom_ne.open("CustomNe");
                    if(parfile_custom_ne.fail()){cout << "Couldn't find 'CustomNe' file!" << endl; exit (EXIT_FAILURE);}
                    while (getline(parfile_custom_ne,parline_custom_ne)){parm_custom_ne.push_back(parline_custom_ne);}
                    int customNe = atoi(parm_custom_ne[0].c_str());
                    string part6 = parm_custom_ne[1];
                    parm_custom_ne.clear();
                    /* Need to first initialize paramters for macs */
                    logfile << "    - Ne that was read in: " << "'" << customNe << "'." << endl;
                    float ScaledMutation = 4 * customNe * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 50 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                           /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                      /* string for Scaled Recombination */
                    stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                       /* string for Number Haplotypes */
                    for(int i = 0; i < numChr; i++)
                    {
                        stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                         /* Convert chromosome length to a string */
                        stringstream s5; s5 << seednumber + i; string macsseed = s5.str();                      /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part2+part6+part7; system(command.c_str());
                        system("rm haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail --lines +6 file1.txt > Intermediate.txt");
                        part1 = "tail --lines +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();                                  /* Convert i loop to string chromosome number */
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str()); /* Store name of genotype file */
                        system("head --lines 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                        command = part1 + mapfile;  MapFiles[i] = mapfile; system(command.c_str());             /* Store name of map file */
                        part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                        part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                        part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(Ne_spec == "Ne70")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 70 * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 70 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                           /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                      /* string for Scaled Recombination */
                    stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                       /* string for Number Haplotypes */
                    for(int i = 0; i < numChr; i++)
                    {
                        stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                         /* Convert chromosome length to a string */
                        stringstream s5; s5 << seednumber + i; string macsseed = s5.str();
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 0.18 0.71 -eN 0.36 1.43 -eN 0.54 2.14 -eN 0.71 2.86 -eN 0.89 3.57 -eN 1.07 4.29 -eN 1.25 5.00 -eN 1.43 5.71";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part7; system(command.c_str());
                        system("rm haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail --lines +6 file1.txt > Intermediate.txt");
                        part1 = "tail --lines +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str()); /* Store name of genotype file */
                        system("head --lines 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());              /* Store name of map file */
                        part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                        part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                        part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to Output Directory */
                        part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(Ne_spec == "Ne100_Scen1")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 100 * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 100 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < numChr; i++)
                    {
                        stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                      /* Convert chromosome length to a string */
                        stringstream s5; s5 << seednumber + i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 0.06 2.0 -eN 0.13 3.0 -eN 0.25 5.0 -eN 0.50 7.0 -eN 0.75 9.0 -eN 1.00 11.0 -eN 1.25 12.5 -eN 1.50 13.0 ";
                        string part6a = "-eN 1.75 13.5 -eN 2.00 14.0 -eN 2.25 14.5 -eN 2.50 15.0 -eN 5.00 20.0 -eN 7.50 25.0 -eN 10.00 30.0 -eN 12.50 35.0 ";
                        string part6b = "-eN 15.00 40.0 -eN 17.50 45.0 -eN 20.00 50.0 -eN 22.50 55.0 -eN 25.00 60.0 -eN 50.00 70.0 -eN 100.00 80.0 -eN 150.00 90.0 ";
                        string part6c = "-eN 200.00 100.0 -eN 250.00 120.0 -eN 500.00 200.0 -eN 1000.00 400.0 -eN 1500.00 600.0 -eN 2000.00 800.0 -eN 2500.00 1000.0";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part6c+part7;
                        system(command.c_str());
                        system("rm haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail --lines +6 file1.txt > Intermediate.txt");
                        part1 = "tail --lines +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                        system("head --lines 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                        part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                        part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                        part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(Ne_spec == "Ne100_Scen2")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 100 * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 100 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < numChr; i++)
                    {
                        stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                      /* Convert chromosome length to a string */
                        stringstream s5; s5 << seednumber + i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 50.00 200.0 -eN 75.00 300.0 -eN 100.00 400.0 -eN 125.00 500.0 -eN 150.00 600.0 -eN 175.00 700.0 -eN 200.00 800.0 ";
                        string part6a = "-eN 225.00 900.0 -eN 250.00 1000.0 -eN 275.00 2000.0 -eN 300.00 3000.0 -eN 325.00 4000.0 -eN 350.00 5000.0 ";
                        string part6b = "-eN 375.00 6000.0 -eN 400.00 7000.0 -eN 425.00 8000.0 -eN 450.00 9000.0 -eN 475.00 10000.0";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part7;
                        system(command.c_str());
                        system("rm haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail --lines +6 file1.txt > Intermediate.txt");
                        part1 = "tail --lines +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                        system("head --lines 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                        part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                        part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                        part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(Ne_spec == "Ne250")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 250 * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 250 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < numChr; i++)
                    {
                        stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                      /* Convert chromosome length to a string */
                        stringstream s5; s5 << seednumber + i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 0 1.04 -eN 0 1.08 -eN 0 1.12 -eN 0 1.16 -eN 0.01 1.2 -eN 0.03 1.6 -eN 0.05 2.0 -eN 0.1 2.8 ";
                        string part6a = "-eN 0.2 4.8 -eN 0.3 5 -eN 0.4 5.2 -eN 0.5 5.4 -eN 0.6 5.6 -eN 0.7 5.7 -eN 0.8 5.8 -eN 0.9 5.9 -eN 1 6 ";
                        string part6b = "-eN 1 4 -eN 2 8 -eN 3 10 -eN 4 12 -eN 5 14 -eN 6 16 -eN 7 18 -eN 8 20 -eN 9 22 -eN 10 24 -eN 20 28 ";
                        string part6c = "-eN 40 32 -eN 60 36 -eN 80 40 -eN 100 48 -eN 200 80 -eN 400 160 -eN 600 240 -eN 800 320 -eN 1000 400";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part6c+part7;
                        system(command.c_str());
                        system("rm haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail --lines +6 file1.txt > Intermediate.txt");
                        part1 = "tail --lines +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str()); /* Store name of genotype file */
                        system("head --lines 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());              /* Store name of map file */
                        part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                        part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                        part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(Ne_spec == "Ne1000")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 1000 * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 1000 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < numChr; i++)
                    {
                        stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                      /* Convert chromosome length to a string */
                        stringstream s5; s5 << seednumber + i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6  = " -eN 0.50 2.00 -eN 0.75 2.50 -eN 1.00 3.00 -eN 1.25 3.20 -eN 1.50 3.50 -eN 1.75 3.80 -eN 2.00 4.00 -eN 2.25 4.20 ";
                        string part6a = "-eN 2.50 4.50 -eN 5.00 5.46 -eN 10.00 7.37 -eN 15.00 9.28 -eN 20.00 11.19 -eN 25.00 13.10 -eN 50.00 22.66 ";
                        string part6b = "-eN 100.00 41.77 -eN 150.00 60.89 -eN 200.00 80.00";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part7; system(command.c_str());
                        system("rm haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail --lines +6 file1.txt > Intermediate.txt"); part1 = "tail --lines +2 Intermediate.txt > ";
                        /* Convert i loop to string chromosome number */
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());        /* Store name of genotype file */
                        system("head --lines 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                     /* Store name of map file */
                        part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                        part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                        part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
            }
            if(Ne != -5 && Ne_spec == "")
            {
                /* Need to first initialize paramters for macs */
                float ScaledMutation = 4 * Ne * u; logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                float ScaledRecombination = 4 * Ne * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                /* Convert Every Value to a string in order to make string */
                stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                stringstream s3; s3 << FounderHaplotypes; string foundhap = s3.str();                    /* string for Number Haplotypes */
                for(int i = 0; i < numChr; i++)
                {
                    stringstream s4; s4 << ChrLength[i]; string SizeChr = s4.str();                      /* Convert chromosome length to a string */
                    stringstream s5; s5 << seednumber + i; string macsseed = s5.str();                   /* Convert seed number to a string */
                    /* Part 1 run the macs simulation program and output it into ms form */
                    string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                    string part6 = " -h 1e3 2>debug.txt | ./msformatter > file1.txt";
                    string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6; system(command.c_str());
                    system("rm haplo* tree.* debug.txt");
                    /* Part 2 put into right format */
                    system("tail --lines +6 file1.txt > Intermediate.txt");
                    part1 = "tail --lines +2 Intermediate.txt > ";
                    stringstream ss; ss << (i + 1); string str = ss.str();
                    part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                    string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str()); /* Store name of genotype file */
                    system("head --lines 1 Intermediate.txt > TempMap.txt");
                    part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail --bytes=+11 TempMap.txt > ";
                    command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());              /* Store name of map file */
                    part1 = "sed -i '1s/^.//' "; command = part1 + mapfile; system(command.c_str());
                    part1 = "sed -i 's/1/2/g' "; command = part1 + genofile; system(command.c_str());
                    part1 = "sed -i 's/0/1/g' "; command = part1 + genofile; system(command.c_str());
                    system("rm Intermediate.txt file1.txt TempMap.txt");
                    /* need to move files to output Directory */
                    part1 = "mv ./"; part2 = " ./" + outputfolder + "/"; command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                    command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                    logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                }
            }
            logfile << " Finished Generating Sequence Information: " << endl << endl;
            system("rm -rf ./nul || true");
        }
        if(StartSim != "sequence")
        {
            logfile << "============================================================\n";
            logfile << "===\t MaCS Part of Program (Chen et al. 2009) \t====\n";
            logfile << "============================================================\n";
            logfile << "    - File already exist do not need to create sequence information." << endl;
            logfile << "    - Need to ensure that parameters related to sequence information " << endl;
            logfile << "    - from previous simulation are what you wanted!!" << endl << endl;
            /* even though files already exist need to get names of files */
            for(int i = 0; i < numChr; i++)
            {
                /* For the SNP and Map files */
                stringstream ss; ss<<(i + 1); string str=ss.str(); string part2="CH"; string part3="SNP.txt"; string genofile=part2+str+part3; SNPFiles[i] = genofile;
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; MapFiles[i] = mapfile;
            }
        }
        if(StartSim == "founder" || StartSim == "sequence")
        {
            time_t t_start = time(0);
            logfile << "====================================================================\n";
            logfile << "===\t GenoDiver Part of Program (Howard et al. 2016) \t====\n";
            logfile << "====================================================================\n";
            logfile << " Begin Generating Founder Individual Genotypes: " << endl;
            /* Generate random number to start with */
            mt19937 gen(seednumber);
            /* read in haplotypes and create founder individuals */
            vector < string > FounderIndividuals;                                           /* Stores haplotype that spans across chromsomes */
            vector < int > haplotype_used(FounderHaplotypes,0);                             /* Makes it so only a single haplotype is used across animals */
            /* First need to randomly generate number to grab from and can only grab 1 unique haplotype out of the library */
            int* linenumber = new int[numberfounders*2];                                    /* Create a list of line number to grab for each founder */
            for(int i = 0; i < numberfounders; i++)
            {
                for (int j = 0; j < 2; j++)                                                         /* Loop through 2x to get two haplotypes */
                {
                    while(1)
                    {
                        std::uniform_real_distribution<double> distribution(0,1);                   /* Generate sample */
                        double temp = (distribution(gen) * (FounderHaplotypes-1));
                        int tempa = (temp + 0.5);
                        if(haplotype_used[tempa] == 0)                                              /* first checks to make sure hasn't been used */
                        {
                            haplotype_used[tempa] = 1;
                            tempa += 1;                                /* index go from 0 to founderhaplotypes -1; need to add one to get back to line numbers*/
                            linenumber[i*2+j] = tempa;
                            break;
                        }
                    }
                }
                FounderIndividuals.push_back("");
            }
            vector < double > SNPFreq;                                                              /* Creates array for SNP Frequencies */
            for(int i = 0; i < numChr; i++)
            {
                /* directory where file is read in */
                string snppath = path + "/" + outputfolder + "/" + SNPFiles[i];
                /* Read in Haplotypes */
                vector < string > Haplotypes;
                string gam;
                ifstream infile; infile.open(snppath);
                if (infile.fail()){logfile << "Error Opening MaCS SNP File. Check Parameter File!\n"; exit (EXIT_FAILURE);}
                while (infile >> gam){Haplotypes.push_back(gam);}
                /* need to push_back GenoSum when start a new chromosome */
                ChrSNPLength[i] = Haplotypes[0].size();
                logfile << "    - Number of SNP for Chromosome " << i + 1 << ": " << ChrSNPLength[i] << " SNP." << endl;
                if((ChrSNPLength[i]* 0.95) < numMark[i])
                {
                    logfile << endl << " Number of Markers too high for chromosome " << i + 1 << ". Alter parameterfile. " << endl; exit (EXIT_FAILURE);
                }
                int GenoFreqStart = SNPFreq.size();                                             /* Index to tell you where you should start summing allele count */
                for(int g = 0; g < Haplotypes[0].size(); g++){SNPFreq.push_back(0);}            /* Starts at zero for each new snp */
                for(int numfoun = 0; numfoun < numberfounders; numfoun++)
                {
                    string fullhomologo1 = Haplotypes[(linenumber[numfoun*2+0]) - 1];            /* Paternal Haplotype */
                    string fullhomologo2 = Haplotypes[(linenumber[numfoun*2+1]) - 1];            /* Maternal Haplotype */
                    stringstream strStreamM (stringstream::in | stringstream::out);              /* Used to put genotype into string */
                    int start = GenoFreqStart;                                                   /* where to start for each founder */
                    for(int g = 0; g < fullhomologo1.size(); g++)
                    {
                        int temp1 = fullhomologo1[g] - 48; int temp2 = fullhomologo2[g] - 48;
                        if(temp1 == 1 && temp2 == 1){strStreamM << 0; SNPFreq[start] += 0;}     /* add genotype a1a1 to string then to genosum count */
                        if(temp1 == 2 && temp2 == 2){strStreamM << 2; SNPFreq[start] += 2;}     /* add genotype a2a2 to string then to genosum count */
                        if(temp1 == 1 && temp2 == 2){strStreamM << 3; SNPFreq[start] += 1;}     /* add genotype a1a2 to string then to genosum count */
                        if(temp1 == 2 && temp2 == 1){strStreamM << 4; SNPFreq[start] += 1;}     /* add genotype a1a2 to string then to genosum count */
                        start++;
                    }
                    string Genotype = strStreamM.str();
                    FounderIndividuals[numfoun] = FounderIndividuals[numfoun] + Genotype;
                }
            }
            logfile << "    - Size of Founder Sequence: " << FounderIndividuals[0].size() << " SNP." << endl;
            /* Founder Sequence Genotypes Created */
            int TotalSNP = SNPFreq.size();
            // Calculate Frequency and output //
            ofstream output15; output15.open (snpfreqfileloc);
            for(int i = 0; i < SNPFreq.size(); i++){SNPFreq[i] = SNPFreq[i] / double(2 * numberfounders); output15 << SNPFreq[i] << " ";}
            output15.close();
            // Part 4: Output into FounderFile
            ofstream output16; output16.open (foundergenofileloc);
            for(int i = 0; i < numberfounders; i++){output16 << linenumber[i*2+0] << "_" << linenumber[i*2+1] << " " << FounderIndividuals[i] << endl;}
            output16.close();
            /* Delete vectors and linenumber array to conserve memory */
            FounderIndividuals.clear(); delete[] linenumber;
            logfile << " Finished Generating Founder Individual Genotypes. " << endl << endl;
            ////////////////////////////////////////////////////////////////////////////////////
            //////          Generate Effects, Marker Geno, QTL geno for Individuals       //////
            ////////////////////////////////////////////////////////////////////////////////////
            // Step 1: Sample appropriate parameters to get location and effect for additive and dominance.
            // Step 2: Construct a genotype file based off given marker density that has a threshold for Marker MAF.
            // Step 3: Split of Marker and QTL alleles in order to make mutations easier to simulate.
            // Step 4: Scale effects so have right additive and dominance based on user specified value.
            // Step 5: Save Founder mutation in QTL_new_old vector class object.
            // Step 6: Generate hapLibrary and tabulate unique haplotypes
            // Step 7: Create founder file that has everything set for Animal Class and save in Animal vector of class objects.
            logfile << "Begin Constructing Trait Architecture: " << endl;
            /* Calculate total number of SNP across all chromsomes */
            /* Convert to MAF (i.e. if greater than 0.5 then is 1 - freq (doesn't matter which allele it is; just need to know minimum frequency) */
            vector < double > MAF(TotalSNP,0.0);
            #pragma omp parallel for
            for(int i = 0; i < TotalSNP; i++)
            {
                MAF[i] = SNPFreq[i];                                        /* If MAF < 0.50 than MAF refers to allele 2 */
                if(MAF[i] > 0.50){MAF[i] = 1 - MAF[i];}                     /* If MAF > 0.50 than MAF refers to allele 1; so flip it */
            }
            /* Loop across chromosomes to get create marker and QTL locations */
            /* Max size needed for Full arrays */
            int SIZEDF = (NUMBERMARKERS) + (numQTL * numChr) + (numFitnessQTLlethal * numChr) + (numFitnessQTLsublethal * numChr);
            vector < double > FullMapPos(SIZEDF,0.0);                           /* Marker Map for Marker + QTL */
            vector < int > FullColNum(SIZEDF,0);                                /* Column number for SNP that were kept */
            vector < double > FullSNPFreq(SIZEDF,0.0);                          /* SNP Frequency in Full Set of SNP */
            vector < int > FullQTL_Mark(SIZEDF,0);                              /* Indicator of whether SNP was a (Marker, QTL, or QTLFitness) */
            vector < double > FullAddEffectQuan(SIZEDF,0.0);                    /* Array of additive_quan effect */
            vector < double > FullDomEffectQuan(SIZEDF,0.0);                    /* Array of dominance_quan effect */
            vector < double > FullAddEffectFit(SIZEDF,0.0);                     /* Array of additive_quan effect */
            vector < double > FullDomEffectFit(SIZEDF,0.0);                     /* Array of dominance_quan effect */
            int TotalSNPCounter = 0;                                            /* Counter to set where SNP is placed in FullMapPos, FullColNum, FullQTL_Mark */
            int TotalFreqCounter = 0;                                           /* Counter to set where SNP is placed in Freq File for QLT Location */
            int TotalFreqCounter1 = 0;                                          /* Counter to set where SNP is placed in Freq File for Marker Location */
            int firstsnpfreq = 0;                                               /* first snp that pertains to a given chromosome */
            /* Vectors used within each chromosome set to nothing now and then within increase to desired size and when done with clear to start fresh */
            vector < double > mappos;                                           /* Map position for a given chromosome;set length by last SNP map position number */
            vector < double > Add_Array_Fit; vector < double > Add_Array_Quan;  /* array that holds additive effects for each SNP */
            vector < double > Dom_Array_Fit; vector < double > Dom_Array_Quan;  /* array that holds dominance effects for each SNP */
            vector < int > QTL;                                                 /* array to hold whether SNP is QTL or can be used as a marker */
            vector < int > markerlocation;                                      /* at what number is marker suppose to be at */
            vector < double > mapmark;                                          /* array with Marker Map for Test Genotype */
            vector < int > colnum;                                              /* array with Column Number for genotypes that were kept to grab SNP Later */
            vector < double > MarkerQTLFreq;                                    /* array with frequency's for marker and QTL SNP */
            vector < int > QTL_Mark;                                            /* array of whether SNP is QTL or Marker */
            vector < double > addMark_Fit; vector < double > addMark_Quan;      /* array of additive effects for each SNP */
            vector < double > domMark_Fit; vector < double > domMark_Quan;      /* array of dominance effects for each SNP */
            /* Indicator for whether QTL is for quantitatve trait or fitness: 1 = marker; 2 = QTLQuanti; 3 = QTLQuant_QTLFitness; 4 = QTLlethal; 5 =QTLsublethal */
            vector < double > covar_add_fitness;                                /* used later on to generate covariance */
            vector < double > covar_add;                                        /* used later on to generate covariance */
            for(int c = 0; c < numChr; c++)
            {
                logfile << "   - Chromosome " << c + 1 << ":" << endl;
                int endspot = ChrSNPLength[c];                              /* Determines number of SNP to have in order to get end position */
                vector < double > mappos;                                   /* Map position for a given chromosome; set length by last SNP map position number */
                string mapfilepath = path + "/" + outputfolder + "/" + MapFiles[c];
                /* READ map file for a given chromsome *************/
                ifstream infile1; infile1.open(mapfilepath);
                if(infile1.fail()){cout << "Error Opening MaCS Map File. Check ParameterFile!\n"; exit (EXIT_FAILURE);}
                for(int i = 0; i < endspot; i++){double temp; infile1 >> temp; mappos.push_back(temp);}
                infile1.close();
                ////////////////////////////////////////////////////////
                ///// Step 1: Sample location and effect of QTL     ////
                ////////////////////////////////////////////////////////
                double lenChr = mappos[endspot - 1 ];               /* length of chromosome used when sampling from uniform distribution to get QTL location*/
                /* Fill vectors to zero */
                for(int i = 0; i < endspot; i++)
                {
                    Add_Array_Fit.push_back(0.0); Dom_Array_Fit.push_back(0.0); Add_Array_Quan.push_back(0.0); Dom_Array_Quan.push_back(0.0); QTL.push_back(0.0);
                }
                /* use uniform distribution to randomly place QTL along genome for quantitative trait */
                int qtlcountertotal = 0; int numbertimesqtlcycled = 0;
                while(qtlcountertotal < numQTL)
                {
                    double pltrpic = 0.0;
                    std::uniform_real_distribution <double> distribution(0,1);
                    pltrpic = distribution(gen);
                    if(pltrpic < proppleitropic)        /* Has pleitrophic effect need find snp below MAF for fitness */
                    {
                        string killwithin = "NO";
                        while(killwithin == "NO")
                        {
                            std::uniform_real_distribution <double> distribution(0,1);
                            int indexid = (distribution(gen)) * endspot;
                            if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid + firstsnpfreq]<(LowerThresholdMAFFitnesssublethal) && MAF[indexid + firstsnpfreq]>0.03)
                            {
                                QTL[indexid] = 3; qtlcountertotal++; killwithin = "YES";
                            }
                            numbertimesqtlcycled++;
                            if(numbertimesqtlcycled > (10 * endspot))
                            {
                                logfile << endl << "Couldn't find enough quant+fitness that pass MAF threshold! increase fitness MAF or decrease QTL Number!" << endl;
                                exit (EXIT_FAILURE);
                            }
                        }
                    }
                    if(pltrpic >= proppleitropic)        /* No pleitrophic effect only need to sample quantitative */
                    {
                        string killwithin = "NO";
                        while(killwithin == "NO")
                        {
                            std::uniform_real_distribution <double> distribution(0,1);
                            int indexid = (distribution(gen)) * endspot;
                            if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid + firstsnpfreq] > ThresholdMAFQTL)
                            {
                                QTL[indexid] = 2; qtlcountertotal++; killwithin = "YES";
                            }
                            numbertimesqtlcycled++;
                        }
                        if(numbertimesqtlcycled > (10 * endspot))
                        {
                            logfile << endl << "Couldn't find enough quantitative that pass MAF threshold! Decrease quantitative MAF or decrease QTL Number!" << endl;
                            exit (EXIT_FAILURE);
                        }
                    }
                }
                /* use uniform distribution to randomly place QTL along genome for Lethal Fitness */
                qtlcountertotal = 0; numbertimesqtlcycled = 0;
                while(qtlcountertotal < numFitnessQTLlethal)
                {
                    std::uniform_real_distribution <double> distribution(0,1);
                    int indexid = (distribution(gen)) * endspot;
                    if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid + firstsnpfreq]<(LowerThresholdMAFFitnesslethal) && MAF[indexid + firstsnpfreq]>0.01)
                    {
                        QTL[indexid] = 4; qtlcountertotal++;
                    }
                    numbertimesqtlcycled++;
                    if(numbertimesqtlcycled > (10 * endspot))
                    {
                        logfile << endl << "Couldn't find enough lethal that pass MAF threshold! Increase lethal MAF or decrease QTL Number!" << endl;
                        exit (EXIT_FAILURE);
                    }
                }
                /* use uniform distribution to randomly place QTL along genome for sub-Lethal Fitness */
                qtlcountertotal = 0; numbertimesqtlcycled = 0;
                while(qtlcountertotal < numFitnessQTLsublethal)
                {
                    std::uniform_real_distribution <double> distribution(0,1);
                    int indexid = (distribution(gen)) * endspot;
                    if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid + firstsnpfreq]<(LowerThresholdMAFFitnesssublethal) && MAF[indexid + firstsnpfreq]>0.01)
                    {
                        QTL[indexid] = 5; qtlcountertotal++;
                    }
                    numbertimesqtlcycled++;
                    if(numbertimesqtlcycled > (10 * endspot))
                    {
                        logfile << endl << "Couldn't find enough sublethal that pass MAF threshold! Increase sublethal MAF or decrease QTL Number!" << endl;
                        exit (EXIT_FAILURE);
                    }
                }
                /* Now that all of them have been placed go through and generate effects for each one depending on what it was called */
                for(int i = 0; i < endspot; i++)
                {
                    if(QTL[i] == 2)                 /* Tagged as quantitative QTL with no relationship to fitness */
                    {
                        /******* QTL Additive Effect *******/
                        std::gamma_distribution <double> distribution1(Gamma_Shape,Gamma_Scale);            /* QTL generated from a gamma */
                        Add_Array_Quan[i] = distribution1(gen);
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                        std::normal_distribution<double> distribution2(Normal_meanRelDom,Normal_varRelDom);
                        double temph = distribution2(gen);
                        Dom_Array_Quan[i] = (Add_Array_Quan[i] * temph);
                        /* Determine sign of additive effect */
                        /* 1 tells you range so can sample from 0 to 1 and 0.5 is the frequency */
                        std::binomial_distribution<int> distribution3(1,0.5);
                        int signadd = distribution3(gen);
                        if(signadd == 1){Add_Array_Quan[i] = Add_Array_Quan[i] * -1;}                     /*Assign negative effect with a 50% probability */
                        /* set fitness ones to 0 */
                        Add_Array_Fit[i] = 0; Dom_Array_Fit[i] = 0;
                    }
                    if(QTL[i] == 3)                 /* Tagged as quantitative QTL with a relationship to fitness */
                    {
                        /******* QTL Additive Effect ********/
                        /* Utilizing a Trivariate Reduction */
                        double sub = genetic_correlation * sqrt(Gamma_Shape * Gamma_Shape_SubLethal);
                        std::gamma_distribution <double> distribution1((Gamma_Shape-sub),1);            /* QTL generated from a gamma */
                        double temp = distribution1(gen);
                        std::gamma_distribution <double> distribution2(sub,1);                          /* Covariance part */
                        double tempa = distribution2(gen);
                        Add_Array_Quan[i] = Gamma_Scale * (temp + tempa);
                        covar_add_fitness.push_back(tempa);
                        /*******   QTL Fit S effect  *******/                       /* Generate effect after standardize add and dominance effect */
                        Add_Array_Fit[i] = -5; Dom_Array_Fit[i] = -5;               /* Set it to -5 first to use as a flag since # has to be greater than 0 */
                        /******* QTL Additive Effect *******/
                        /* 1 tells you range so can sample from 0 to 1 and 0.5 is the frequency */
                        //std::binomial_distribution<int> distribution4(1,0.5);
                        //int signadd = distribution4(gen);
                        //if(signadd == 1){Add_Effect_Quan[i] = Add_Effect_Quan[i] * -1;}                     /*Assign negative effect with a 50% probability */
                        covar_add.push_back(Add_Array_Quan[i]);
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                        std::normal_distribution<double> distribution5(Normal_meanRelDom,Normal_varRelDom);
                        double temph = distribution5(gen);
                        Dom_Array_Quan[i] = abs(Add_Array_Quan[i]) * temph;
                    }
                    if(QTL[i] == 4)                 /* Tagged as lethal fitness QTL with no relationship to quantitative */
                    {
                        /*******     QTL s effect (i.e. selection coeffecient)   *******/
                        std::gamma_distribution <double> distribution1(Gamma_Shape_Lethal,Gamma_Scale_Lethal);
                        Add_Array_Fit[i] = distribution1(gen);
                        /******      QTL h Effect (i.e. degree of dominance)     *******/
                        /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                        std::normal_distribution<double> distribution2(Normal_meanRelDom_Lethal,Normal_varRelDom_Lethal);
                        double temph = distribution2(gen);
                        Dom_Array_Fit[i] = abs(temph);
                        /* Set quantitative additive and dominance to 0 */
                        Add_Array_Quan[i] = 0; Dom_Array_Quan[i] = 0;
                    }
                    if(QTL[i] == 5)                 /* Tagged as sub-lethal fitness QTL with no relationship to quantitative */
                    {
                        /*******     QTL S effect (i.e. selection coeffecient)   *******/
                        std::gamma_distribution <double> distribution1(Gamma_Shape_SubLethal,Gamma_Scale_SubLethal);
                        Add_Array_Fit[i] = distribution1(gen);
                        /******      QTL h Effect (i.e. degree of dominance)     *******/
                        /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                        std::normal_distribution<double> distribution2(Normal_meanRelDom_SubLethal,Normal_varRelDom_SubLethal);
                        double temph = distribution2(gen);
                        Dom_Array_Fit[i] = abs(temph);
                        /* Set quantitative additive and dominance to 0 */
                        Add_Array_Quan[i] = 0; Dom_Array_Quan[i] = 0;
                    }
                }
                /* tabulate number of QTL for each chromosome may be less than what simulated due to two being within two markers*/
                int simulatedQTL = 0;
                for(int i = 0; i < endspot; i++){if(QTL[i] >= 2){simulatedQTL += 1;}}
                qtlperChr[c] = simulatedQTL;
                markerperChr[c] = numMark[c];
                ///////////////////////////////////////////////////////////////////////
                // Step 2: Construct Marker file based on given Marker MAF threshold //
                ///////////////////////////////////////////////////////////////////////
                /* Create Genotypes that has QTL and Markers */
                int markertokeep = markerperChr[c] + qtlperChr[c];      /* Determine length of array based on markers to keep plus number of actual QTL */
                for(int i = 0; i < markertokeep; i++)
                {
                    mapmark.push_back(0.0); colnum.push_back(0); MarkerQTLFreq.push_back(0.0); QTL_Mark.push_back(0);
                    addMark_Fit.push_back(0.0); domMark_Fit.push_back(0.0); addMark_Quan.push_back(0.0); domMark_Quan.push_back(0.0);
                }
                /* use uniform distribution to randomly place markers along genome */
                int markercountertotal = 0; int numbertimescycled = 0;
                while(markercountertotal < numMark[c])
                {
                    std::uniform_real_distribution <double> distribution(0,1);
                    /* randomly sets index location */
                    int indexid = (distribution(gen)) * endspot;
                    if(QTL[indexid] == 0 && mappos[indexid] > 0.0001 && mappos[indexid] < 0.9999 && MAF[indexid + firstsnpfreq] >= ThresholdMAFMark)
                    {
                        QTL[indexid] = 1; markercountertotal++;
                    }
                    numbertimescycled++;
                    if(numbertimescycled > (50 * endspot))
                    {
                        logfile << endl << "Couldn't find enough markers that pass MAF threshold! Decrease Marker MAF or Marker Number!" << endl;
                        exit (EXIT_FAILURE);
                    }
                }
                int markloc = 0;                                        /* Track where you are at in marker file */
                for(int i = 0; i < endspot; i++)
                {
                    if(QTL[i] == 1)     /* Declared as a marker */
                    {
                        /* place position, column, frequency, tag as marker (i.e. 1), give add and dom effect as 0 */
                        mapmark[markloc] = mappos[i]; colnum[markloc] = i; MarkerQTLFreq[markloc] = SNPFreq[i + firstsnpfreq];
                        QTL_Mark[markloc] = 1; addMark_Fit[markloc] = 0; domMark_Fit[markloc] = 0; addMark_Quan[markloc] = 0; domMark_Quan[markloc] = 0;
                        markloc++;   /* Increase position by one in mapMark and colnum and go to next declared marker */
                    }
                    if(QTL[i] > 1)
                    {
                        mapmark[markloc] = mappos[i]; colnum[markloc] = i; MarkerQTLFreq[markloc] = SNPFreq[i + firstsnpfreq];
                        QTL_Mark[markloc] = QTL[i]; addMark_Quan[markloc]=Add_Array_Quan[i]; domMark_Quan[markloc]=Dom_Array_Quan[i];
                        addMark_Fit[markloc]=Add_Array_Fit[i]; domMark_Fit[markloc]=Dom_Array_Fit[i];
                        markloc++;                                /* Increase position by one in mapMark and colnum */
                    }
                }
                int test_mark = 0; int test_QTL_Quan = 0; int test_QTL_Quan_Fitness = 0; int test_QTL_Leth = 0; int test_QTL_SubLeth = 0; int totalkeep = 0;
                for(int i = 0; i < QTL_Mark.size(); i++)
                {
                    if(QTL_Mark[i] == 1){test_mark += 1;}
                    if(QTL_Mark[i] == 2){test_QTL_Quan += 1;}
                    if(QTL_Mark[i] == 3){test_QTL_Quan_Fitness += 1;}
                    if(QTL_Mark[i] == 4){test_QTL_Leth += 1;}
                    if(QTL_Mark[i] == 5){test_QTL_SubLeth += 1;}
                    if(QTL_Mark[i] >= 1){totalkeep += 1;}
                }
                /* Update Number of Markes Per Chromsome */
                markerperChr[c] = test_mark;
                logfile << "        - Number Markers: " << markerperChr[c] << "." << endl;
                logfile << "        - Number of Quantititive QTL: " << test_QTL_Quan << "." << endl;
                logfile << "        - Number of Quantitative QTL with Pleiotropic effects on Fitness: " << test_QTL_Quan_Fitness << endl;
                logfile << "        - Number of Fitness QTL: " << test_QTL_Leth + test_QTL_SubLeth << "." << endl;
                markertokeep = totalkeep;
                /* Copy to full set */
                if(c == 0)
                {
                    int j = 0;                                                  /* where at in arrays for each chromosome */
                    for(int i = 0 ; i < markertokeep; i++)                      /* first chromsome would start at first postion until markertokeep */
                    {
                        FullMapPos[i] = mapmark[j] + c + 1; FullColNum[i] = colnum[j]; FullSNPFreq[i] = MarkerQTLFreq[j]; FullQTL_Mark[i] = QTL_Mark[j];
                        FullAddEffectQuan[i]=addMark_Quan[j]; FullDomEffectQuan[i]=domMark_Quan[j];
                        FullAddEffectFit[i]=addMark_Fit[j]; FullDomEffectFit[i]=domMark_Fit[j]; j++;
                    }
                    TotalSNPCounter = markertokeep;
                }
                if(c > 0)
                {
                    int j = 0;
                    for(int i = TotalSNPCounter; i < (TotalSNPCounter + markertokeep); i++)
                    {
                        FullMapPos[i] = mapmark[j] + c + 1; FullColNum[i] = colnum[j]; FullSNPFreq[i] = MarkerQTLFreq[j]; FullQTL_Mark[i] = QTL_Mark[j];
                        FullAddEffectQuan[i]=addMark_Quan[j]; FullDomEffectQuan[i]=domMark_Quan[j];
                        FullAddEffectFit[i]=addMark_Fit[j]; FullDomEffectFit[i]=domMark_Fit[j]; j++;
                    }
                    TotalSNPCounter = TotalSNPCounter + markertokeep;           /* Updates where should begin to fill for next chromosome */
                }
                firstsnpfreq += endspot;
                mappos.clear(); QTL.clear(); markerlocation.clear(); mapmark.clear(); colnum.clear();
                MarkerQTLFreq.clear(); QTL_Mark.clear(); Add_Array_Fit.clear(); Dom_Array_Fit.clear(); Add_Array_Quan.clear(); Dom_Array_Quan.clear();
                addMark_Quan.clear(); domMark_Quan.clear(); addMark_Fit.clear(); domMark_Fit.clear();
            }
            /* Remove vectors */
            SNPFreq.clear(); MAF.clear();
            vector < int > count_type(5,0);
            for(int i = 0; i < TotalSNPCounter; i++)
            {
                if(FullQTL_Mark[i] == 1){count_type[0] = count_type[0] + 1;}
                if(FullQTL_Mark[i] == 2){count_type[1] = count_type[1] + 1;}
                if(FullQTL_Mark[i] == 3){count_type[2] = count_type[2] + 1;}
                if(FullQTL_Mark[i] == 4){count_type[3] = count_type[3] + 1;}
                if(FullQTL_Mark[i] == 5){count_type[4] = count_type[4] + 1;}
            }
            logfile << "   - QTL's Simulated." << endl;
            logfile << "       - Quantitative QTL's: " << count_type[1] << "." << endl;
            logfile << "       - Quantitative QTL's with Pleiotropic effects on Fitness: " << count_type[2] << "." << endl;
            logfile << "       - Fitness Lethal QTL's: " << count_type[3] << "." << endl;
            logfile << "       - Fitness Sub-Lethal QTL's: " << count_type[4] << "." << endl;
            logfile << "   - Marker Array Created." << endl;
            logfile << "       - Markers: " << count_type[0] << "." << endl;
            //////////////////////////////////////////
            // Step 3: Split off Marker and QTL SNP //
            //////////////////////////////////////////
            /* Split off SNP that are markers and QTL for either Quantitative, Fitness or both by creating an index from 1 to number of SNP based on location */
            /* The markers won't change, but the number of QTL will change based on the accumulation of new mutations. */
            /* Fill arrays */
            for(int i = 0; i < TotalSNPCounter; i++)
            {
                /* If is a marker */
                if(FullQTL_Mark[i] == 1){MarkerIndex.push_back(i); MarkerMapPosition.push_back(FullMapPos[i]); Marker_IndCounter++;}
                if(FullQTL_Mark[i] > 1)     /* If is a qtl */
                {
                    QTL_Index[QTL_IndCounter] = i; QTL_MapPosition[QTL_IndCounter] = FullMapPos[i]; QTL_Type[QTL_IndCounter] = FullQTL_Mark[i];
                    QTL_Freq[QTL_IndCounter] = FullSNPFreq[i];
                    if(QTL_Freq[QTL_IndCounter] > 0.5){QTL_Allele[QTL_IndCounter] = 0;}     /* Indicator for which homozygote impacted */
                    if(QTL_Freq[QTL_IndCounter] < 0.5){QTL_Allele[QTL_IndCounter] = 2;}     /* Indicator for which homozygote impacted */
                    QTL_Add_Quan[QTL_IndCounter] = FullAddEffectQuan[i]; QTL_Dom_Quan[QTL_IndCounter] = FullDomEffectQuan[i];
                    QTL_Add_Fit[QTL_IndCounter] = FullAddEffectFit[i]; QTL_Dom_Fit[QTL_IndCounter] = FullDomEffectFit[i];
                    QTL_IndCounter++;                                                                   /* Move to next cell of array */
                }
            }
            /* used to keep track of where at */
            int MarkerMapPosition_MB[Marker_IndCounter];                                                /* Position in Megabases */
            int MarkerMapPosition_CHR[Marker_IndCounter];                                               /* Chromosome in Megabases */
            for(int i = 0; i < numChr; i++)
            {
                double chr = i + 1;
                for(int j = 0; j < Marker_IndCounter; j++)
                {
                    if(MarkerMapPosition[j] >= i + 1 && MarkerMapPosition[j] < i + 2)
                    {
                        MarkerMapPosition_MB[j] = (MarkerMapPosition[j] - chr) * ChrLength[i]; MarkerMapPosition_CHR[j] =  chr;
                    }
                }
            }
            ofstream output21;
            output21.open (Marker_Map);
            output21 << "chr pos" << endl;
            for(int i = 0; i < Marker_IndCounter; i++){output21 << MarkerMapPosition_CHR[i] << " " << MarkerMapPosition_MB[i] << endl;}
            output21.close();
            /* When create covariance between quantitative and fitness some may fall out therefore need to make sure they are lined up */
            if(covar_add_fitness.size() > 0)
            {
                int counteradd_cov = 0;
                for(int i = 0; i < QTL_IndCounter; i++)
                {
                    if(QTL_Type[i] == 3)
                    {
                        if(QTL_Add_Quan[i] != covar_add[counteradd_cov])
                        {
                            covar_add.erase(covar_add.begin()+counteradd_cov);
                            covar_add_fitness.erase(covar_add_fitness.begin()+counteradd_cov);
                        }
                        if(QTL_Add_Quan[i] == covar_add[counteradd_cov]){counteradd_cov++;}
                    }
                }
                covar_add.clear();
            }
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            // In order to ensure that the population starts out with similar genomes due to some animals having ///
            // a high level of homozygosity which makes it difficult not to generate chromosome of complete ROH  ///
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            vector < string > founder_snp;
            string line;
            ifstream infilefounder;
            infilefounder.open(foundergenofileloc);
            if(infilefounder.fail()){cout << "Error Opening Founder Genotype File!\n"; exit (EXIT_FAILURE);}
            while (getline(infilefounder,line))
            {
                vector < string > numbers;                              /* Stores Line in a vector */
                string temp;                                            /* temporary variable */
                while (line.find(" ",0) !=std::string::npos)            /* Loops until end of line */
                {
                    size_t pos = line.find(" ", 0); temp = line.substr(0,pos); line.erase(0, pos + 1); numbers.push_back(temp);
                }
                numbers.push_back(line);                                /* the last token is all alone */
                string geno = numbers[1];                               /* numbers[0] is ID and numbers[1] is genotype */
                int TotalSNP = geno.size();
                int* Genotype = new int[TotalSNP];                      /* put genotypes into an array */
                for(int i = 0; i < TotalSNP; i++){int temp = geno[i] - 48; Genotype[i]= temp;} /* ASCI value is 48 for 0; and ASCI value for 0 to 9 is 48 to 57 */
                int* MarkerGeno = new int[TotalSNPCounter];             /* Array that holds SNP that were declared as Markers and QTL */
                int markercounter = 0;                                  /* counter for where we are at in SNPLoc */
                int bigmarkercounter = 0;                               /* counter to determine where chromsome ends and begins */
                for(int i = 0; i < numChr; i++)
                {
                    /* fill in temp matrix where column number from SNPLoc should match up within each chromosome */
                    vector < int > temp(ChrSNPLength[i],0);
                    int k = 0;
                    /* grab genotype from right chromosome and go to next temp position */
                    for(int j = bigmarkercounter; j < (bigmarkercounter + ChrSNPLength[i]); j++){temp[k] = Genotype[j]; k++;}
                    for(int j = 0; j < ChrSNPLength[i]; j++)
                    {
                        if(markercounter < TotalSNPCounter)
                        {
                            /* if colnumber lines up with big file then it is a marker or QTL */
                            if(FullColNum[markercounter] == j){MarkerGeno[markercounter] = temp[j]; markercounter++;}
                        }
                    }
                    /* Fix it so it kills one reached TotalSNPCounter */
                    /* determine where next SNP should begin next in in large genotype file */
                    if(i != numChr)
                    {
                        int temp = 0;
                        for(int chr = 0; chr < i + 1; chr++){temp = ChrSNPLength[chr] + temp;}
                        bigmarkercounter = temp;
                    }
                }
                /* Put into string */
                stringstream strStreamgeno (stringstream::in | stringstream::out);
                for (int i=0; i < TotalSNPCounter; i++){strStreamgeno << MarkerGeno[i];}
                geno = strStreamgeno.str();
                delete [] Genotype; delete [] MarkerGeno;
                founder_snp.push_back(geno);
            }
            /* Prior to generating founder generation need to ensure don't start out with complete homozygous genotypes */
            vector < vector < double > > markerhomozygosity;
            vector < int > remove_founder;
            for(int i = 0; i < founder_snp.size(); i++)
            {
                vector < double > animal_homo;                          /* Stores Homozygosity for each animal */
                for(int j = 0; j < numChr; j++){animal_homo.push_back(0);}
                markerhomozygosity.push_back(animal_homo);
                remove_founder.push_back(0);
            }
            #pragma omp parallel for
            for(int ind = 0; ind < founder_snp.size(); ind++)
            {
                string geno = founder_snp[ind];
                vector < int > Geno(geno.size(),0);                 /* declare genotype array */
                /* Convert Genotype string to genotype array */
                for (int m = 0; m < geno.size(); m++){Geno[m] = geno[m] - 48;}
                // From two haplotypes create genotypes: 0 = a1,a1; 2 = a2,a2; 3 = a1,a2; 4 = a2,a1
                int genocounter = 0;
                for(int c = 0; c < numChr; c++)
                {
                    for(int i = genocounter; i < (genocounter + (markerperChr[c] + qtlperChr[c])); i++)
                    {
                        if(Geno[i] == 0 || Geno[i] == 2){markerhomozygosity[ind][c] += 1;}
                    }
                    genocounter = genocounter + markerperChr[c] + qtlperChr[c];
                    markerhomozygosity[ind][c] = markerhomozygosity[ind][c] / (markerperChr[c] + qtlperChr[c]);
                }
            }
            for(int i = 0; i < numChr; i++)
            {
                double sumhomozygosity = 0;
                double meanhomozygosity, sdhomozygosity;
                for(int j = 0; j < markerhomozygosity.size(); j++){sumhomozygosity += markerhomozygosity[j][i];}
                meanhomozygosity = sumhomozygosity / double(markerhomozygosity.size());
                sumhomozygosity = 0;
                for(int j = 0; j < markerhomozygosity.size(); j++)
                {
                    sumhomozygosity += ((markerhomozygosity[j][i]- meanhomozygosity) * (markerhomozygosity[j][i]- meanhomozygosity));
                }
                sdhomozygosity = sqrt(sumhomozygosity / double(markerhomozygosity.size()));
                for(int j = 0; j < markerhomozygosity.size(); j++)
                {
                    if(markerhomozygosity[j][i] > (meanhomozygosity + (3*sdhomozygosity))){if(remove_founder[j] == 0){remove_founder[j] = 1;}}
                }
            }
            for(int i = 0; i < markerhomozygosity.size(); i++){markerhomozygosity[i].clear();}
            markerhomozygosity.clear();
            /* Remove animals with extreme homozygosity */
            int ROWS = founder_snp.size();                           /* Current Size of founder population */
            int i = 0;
            while(i < ROWS)
            {
                while(1)
                {
                    if(remove_founder[i] == 1){founder_snp.erase(founder_snp.begin()+i); remove_founder.erase(remove_founder.begin()+i); ROWS = ROWS -1; break;}
                    if(remove_founder[i] == 0){i++; break;}
                }
            }
            remove_founder.clear();
            /* put into founder qtl and markers then start selecting */
            vector < string > founder_qtl;
            vector < string > founder_markers;
            for(int ind = 0; ind < founder_snp.size(); ind++)
            {
                string snp = founder_snp[ind];
                vector < int > MarkerGeno(snp.size(),0);
                for(int i = 0; i < snp.size(); i++){MarkerGeno[i] = snp[i] - 48;}   /* Put into vector */
                int* MarkerGenotypes = new int[NUMBERMARKERS];                      /* Marker Genotypes */
                int* QTLGenotypes = new int[QTL_IndCounter];                        /* QTL Genotypes */
                Marker_IndCounter = 0;                                              /* Counter to determine where you are at in Marker index array */
                QTL_IndCounter = 0;                                                 /* Counter to determine where you are at in QTL index array */
                /* Fill Genotype Array */
                for(int i = 0; i < TotalSNPCounter; i++)                            /* Loop through MarkerGenotypes array & place Geno based on Index value */
                {
                    if(Marker_IndCounter < MarkerIndex.size())                      /* ensures doesn't go over and cause valgrind error */
                    {
                        if(i == MarkerIndex[Marker_IndCounter]){MarkerGenotypes[Marker_IndCounter] = MarkerGeno[i]; Marker_IndCounter++;}
                    }
                    if(QTL_IndCounter < QTL_Index.size())                           /* ensures doesn't go over and cause valgrind error */
                    {
                        if(i == QTL_Index[QTL_IndCounter]){QTLGenotypes[QTL_IndCounter] = MarkerGeno[i]; QTL_IndCounter++;}
                    }
                }
                /* MarkerGeno array contains Markers and QTL therefore need to split them off based on index arrays that were created previously */
                /* put marker, qtl (quantitative + fitness into string to store) */
                stringstream strStreamM (stringstream::in | stringstream::out);
                for (int i=0; i < Marker_IndCounter; i++){strStreamM << MarkerGenotypes[i];}
                string MA = strStreamM.str();
                stringstream strStreamQt (stringstream::in | stringstream::out);
                for (int i=0; i < QTL_IndCounter; i++){strStreamQt << QTLGenotypes[i];}
                string QT = strStreamQt.str();
                founder_markers.push_back(MA); founder_qtl.push_back(QT);
                delete [] MarkerGenotypes; delete [] QTLGenotypes;
            }
            /* calculate frequency */
            double* tempfreq = new double[founder_qtl[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
            frequency_calc(founder_qtl, tempfreq);                              /* Function to calculate snp frequency */
            for(int i = 0; i < founder_qtl[0].size(); i++){QTL_Freq[i] = tempfreq[i];}
            delete [] tempfreq;
            //////////////////////////////////////////////////
            // Step 4: Scale Additive and Dominance Effects //
            //////////////////////////////////////////////////
            /* Variance Additive = 2pq[a + d(q-p)]^2; Variance Dominance = (2pqd)^2; One depends on the other so therefore do an optimization technique */
            logfile << "   - Begin to scale additive and dominance effects." << endl;
            double obsh2 = 0.0;               /* current iterations h2 for additive */
            double obsph2 = 0.0;              /* current iterations h2 for dominance */
            int quit = 1;                     /* won't quit until = 0 */
            vector < double > tempadd(QTL_IndCounter,0.0);                   /* Temporary array to store scaled additive effects */
            vector < double > tempdom(QTL_IndCounter,0.0);                   /* Temporary array to store scaled dominance effects */
            int interationnumber = 0;
            while(quit != 0)
            {
                for(int i = 0; i < QTL_IndCounter; i++)
                {
                    if(QTL_Type[i] == 2 || QTL_Type[i] == 3)
                    {
                        tempadd[i] = QTL_Add_Quan[i] * scalefactaddh2;     /* based on current iterations scale get temporary additive effect */
                        if(scalefactpdomh2 == 0.0){tempdom[i] = 0;}                                   /* based on current iteration; scale to get temporary dom effect */
                        if(scalefactpdomh2 > 0.0){tempdom[i] = QTL_Dom_Quan[i] * scalefactpdomh2;}   /* based on current iteration; scale to get temporary dom effect */
                    }
                }
                double tempva = 0;                                              /* Variance in additive based on current iterations scale factor */
                double tempvd = 0;                                              /* Variance in dominance based on current iterations scale factor */
                for(int i = 0; i < QTL_IndCounter; i++)
                {
                    if(QTL_Type[i] == 2 || QTL_Type[i] == 3)
                    {
                        tempva += 2 * QTL_Freq[i] * (1 - QTL_Freq[i]) * ((tempadd[i]+(tempdom[i] * ((1 - QTL_Freq[i]) - QTL_Freq[i]))) * (tempadd[i]+(tempdom[i] * ((1 - QTL_Freq[i]) - QTL_Freq[i]))));
                        tempvd += ((2 * QTL_Freq[i] * (1 - QTL_Freq[i]) * tempdom[i]) * (2 * QTL_Freq[i] * (1 - QTL_Freq[i]) * tempdom[i]));
                    }
                }
                if(abs(Variance_Additiveh2 - tempva) < 0.004){scalefactaddh2 = scalefactaddh2;}                /* Add Var Within Window (+/-) don't change */
                if(Variance_Additiveh2 - tempva > 0.004){scalefactaddh2 = scalefactaddh2 + 0.0001;}            /* Add Var Smaller than desired increase */
                if(Variance_Additiveh2 - tempva < -0.004){scalefactaddh2 = scalefactaddh2 - 0.0001;}           /* Add Var Bigger than desired decrease */
                if(abs(Variance_Dominanceh2 - tempvd) < 0.004){scalefactpdomh2 = scalefactpdomh2;}             /* Dom Var Within Window (+/-) don't change */
                if(Variance_Dominanceh2 - tempvd > 0.004){scalefactpdomh2 = scalefactpdomh2 + 0.0001;}         /* Dom Var Smaller than desired increase */
                if(Variance_Dominanceh2 - tempvd < -0.004){scalefactpdomh2 = scalefactpdomh2 - 0.0001;}         /* Dom Var Bigger than desired decrease */
                if(abs(Variance_Additiveh2 - tempva) < 0.004 && abs(Variance_Dominanceh2 - tempvd) < 0.004)    /* Dom & Add Var within window */
                {
                    quit = 0;
                    scalefactaddh2 = scalefactaddh2;
                    scalefactpdomh2 = scalefactpdomh2;
                    logfile << "   - Effects Centered and Scaled:" << endl;
                    logfile << "       - Additive Variance in Founders: " << tempva << "." << endl;
                    logfile << "       - Dominance Variance in Founders: " << tempvd << "." << endl;
                    logfile << "       - Scale factor for additive effects: " << scalefactaddh2 << "." << endl;
                    logfile << "       - Scale factor for dominance effects: " << scalefactpdomh2 << "." << endl;
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if(QTL_Type[i] == 2 || QTL_Type[i] == 3)
                        {
                            QTL_Add_Quan[i] = QTL_Add_Quan[i] * scalefactaddh2;
                            QTL_Dom_Quan[i] = QTL_Dom_Quan[i] * scalefactpdomh2;
                        }
                    }
                    AdditiveVar[0] = tempva;
                    DominanceVar[0] = tempvd;
                }
                tempadd.clear(); tempdom.clear(); interationnumber++;
                //cout << interationnumber << " " << tempvd << " " << scalefactpdomh2 << " " << tempva << " " << scalefactaddh2 << endl << endl;;
                //if(interationnumber > 2){exit (EXIT_FAILURE);}
            }
            if(scalefactaddh2 == 0 && scalefactpdomh2 == 0)
            {
                logfile << "   - Dominance and Additive Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
            }
            if(scalefactaddh2 > 0)
            {
                /* print number of QTL that display partial or over-dominance */
                int partDom = 0; int overDom = 0; int negativesign = 0; int positivesign = 0;
                for(int i = 0; i < QTL_IndCounter; i++)
                {
                    if(QTL_Type[i] == 2 || QTL_Type[i] == 3)
                    {
                        if(abs(QTL_Dom_Quan[i]) < abs(QTL_Add_Quan[i])){partDom += 1;}
                        if(abs(QTL_Dom_Quan[i]) > abs(QTL_Add_Quan[i])){overDom += 1;}
                        if(QTL_Dom_Quan[i] > 0){positivesign += 1;}
                        if(QTL_Dom_Quan[i] < 0){negativesign += 1;}
                    }
                }
                if(scalefactpdomh2 > 0)
                {
                    logfile << "   -After Centering and Scaling Number of QTL with: " << endl;
                    logfile << "       - Partial-Dominance: " << partDom << "." << endl;
                    logfile << "       - Over-Dominance: " << overDom << "." << endl;
                    logfile << "       - Negative Sign: " << negativesign << "." << endl;
                    logfile << "       - Positive Sign: " << positivesign << "." << endl;
                }
                if(scalefactpdomh2 == 0)
                {
                    logfile << "   - Dominance Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
                }
            }
            /////////////////////////////////////////////////////////////////////////////////////////////////
            // Once Standardize now need to generate effects for additive with pleithropic fitness effects //
            /////////////////////////////////////////////////////////////////////////////////////////////////
            if(proppleitropic > 0)
            {
                logfile << "   - Generate correlation (rank) between additive effects of quantitative and fitness." << endl;
                double cor = 0.0;
                while(genetic_correlation - cor >= 0.015)
                {
                    vector < double > quant_rank; vector < double > fitness_rank;
                    int covarcount = 0;
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if(QTL_Type[i] == 3)
                        {
                            double sub = (-1 * genetic_correlation) * sqrt(Gamma_Shape * Gamma_Shape_SubLethal);
                            /******   QTL fit s Effect ********/
                            std::gamma_distribution <double> distribution2((Gamma_Shape_SubLethal - sub),1);
                            double temp = distribution2(gen);                /* Raw Fitness Value need to transform to relative fitness */
                            QTL_Add_Fit[i] = Gamma_Scale_SubLethal * (temp + (covar_add_fitness[covarcount]*scalefactaddh2));
                            /******   QTL Fit h Effect  *******/
                            /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                            std::normal_distribution<double> distribution3(Normal_meanRelDom_SubLethal,Normal_varRelDom_SubLethal);
                            double temph = distribution3(gen);
                            QTL_Dom_Fit[i] = abs(temph);
                            quant_rank.push_back(QTL_Add_Quan[i]); fitness_rank.push_back(QTL_Add_Fit[i]); covarcount++;
                        }
                    }
                    double tempq; double tempf;                     /* Bubble sort */
                    for(int i = 0; i < quant_rank.size()-1; i++)
                    {
                        for(int j=i+1; j < quant_rank.size(); j++)
                        {
                            if(quant_rank[i] > quant_rank[j])
                            {
                                tempq = quant_rank[i]; tempf = fitness_rank[i];
                                quant_rank[i] = quant_rank[j]; fitness_rank[i] = fitness_rank[j];
                                quant_rank[j] = tempq; fitness_rank[j] = tempf;
                            }
                        }
                    }
                    for(int i = 0; i < quant_rank.size(); i++){quant_rank[i] = i + 1;}
                    for(int i = 0; i < fitness_rank.size()-1; i++)
                    {
                        for(int j=i+1; j < fitness_rank.size(); j++)
                        {
                            if(fitness_rank[i] > fitness_rank[j])
                            {
                                tempq = quant_rank[i]; tempf = fitness_rank[i];
                                quant_rank[i] = quant_rank[j]; fitness_rank[i] = fitness_rank[j];
                                quant_rank[j] = tempq; fitness_rank[j] = tempf;
                            }
                        }
                    }
                    for(int i = 0; i < fitness_rank.size(); i++){fitness_rank[i] = i + 1;}
                    double num_sum_diff_ranks = 0;
                    for(int i = 0; i < quant_rank.size(); i++){num_sum_diff_ranks += (fitness_rank[i] - quant_rank[i]) * (fitness_rank[i] - quant_rank[i]);}
                    cor = 1 - ((6 * num_sum_diff_ranks) / (quant_rank.size()*((quant_rank.size()*quant_rank.size())-1)));
                }
                logfile << "       - Rank Correlation between additive effects of quantitative and fitness: " << cor << "." << endl;
            }
            /* generate mean selection and dominance coeffecint for fitness traits */
            double meanlethal_sel = 0.0; double meanlethal_dom = 0.0; int numberlethal_sel = 0;
            double meansublethal_sel = 0.0; double meansublethal_dom = 0.0; int numbersublethal_sel = 0;
            for(int i = 0; i < QTL_IndCounter; i++)
            {
                if(QTL_Type[i] == 4)            /* Lethal */
                {
                    meanlethal_sel += QTL_Add_Fit[i]; meanlethal_dom += QTL_Dom_Fit[i]; numberlethal_sel += 1;
                }
                if(QTL_Type[i] == 3 || QTL_Type[i] == 5)
                {
                    meansublethal_sel += QTL_Add_Fit[i]; meansublethal_dom += QTL_Dom_Fit[i]; numbersublethal_sel += 1;
                }
            }
            if(meanlethal_sel > 0){meanlethal_sel = meanlethal_sel / double(numberlethal_sel);}
            if(meanlethal_dom > 0){meanlethal_dom = meanlethal_dom / double(numberlethal_sel);}
            if(meansublethal_sel > 0){meansublethal_sel = meansublethal_sel / double(numbersublethal_sel);}
            if(meansublethal_dom > 0){meansublethal_dom = meansublethal_dom / double(numbersublethal_sel);}
            logfile << "       - Fitness Lethal: " << endl;
            logfile << "           - Mean Selection Coefficient: " << meanlethal_sel << endl;
            logfile << "           - Mean Degree of Dominance: " << meanlethal_dom << endl;
            logfile << "       - Fitness Sub-Lethal: " << endl;
            logfile << "           - Mean Selection Coefficient: " << meansublethal_sel << endl;
            logfile << "           - Mean Degree of Dominance: " << meansublethal_dom << endl;
            //////////////////////////////////////////////////////////////////////
            // Step 5: Save Founder mutation in QTL_new_old vector class object //
            //////////////////////////////////////////////////////////////////////
            /* Need to add both Quantitative QTL and Fitness QTL to class */
            for(int i = 0; i < QTL_IndCounter; i++)
            {
                if(QTL_Type[i] == 4 || QTL_Type[i] == 5)            /* Fitness QTL */
                {
                    stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<QTL_Freq[i]; string stringfreq=strStreamtempfreq.str();
                    stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<QTL_Type[i]; string stringtype=strStreamtemptype.str();
                    QTL_new_old tempa(QTL_MapPosition[i], QTL_Add_Fit[i], QTL_Dom_Fit[i],stringtype, 0, stringfreq); population_QTL.push_back(tempa);
                }
                if(QTL_Type[i] == 2)                                /* Quantitative QTL */
                {
                    stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<QTL_Freq[i]; string stringfreq=strStreamtempfreq.str();
                    stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<QTL_Type[i]; string stringtype=strStreamtemptype.str();
                    QTL_new_old tempa(QTL_MapPosition[i], QTL_Add_Quan[i], QTL_Dom_Quan[i],stringtype, 0, stringfreq); population_QTL.push_back(tempa);
                }
                if(QTL_Type[i] == 3)                                /* Fitness + Quantitative QTL */
                {
                    /* Quantitative one first */
                    stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<QTL_Freq[i]; string stringfreq=strStreamtempfreq.str();
                    stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<"2"; string stringtype=strStreamtemptype.str();
                    QTL_new_old tempa(QTL_MapPosition[i], QTL_Add_Quan[i], QTL_Dom_Quan[i],stringtype, 0, stringfreq); population_QTL.push_back(tempa);
                    /* Fitness one */
                    stringstream strStreamtemptypea (stringstream::in | stringstream::out); strStreamtemptypea<<"5"; string stringtypea=strStreamtemptypea.str();
                    QTL_new_old tempb(QTL_MapPosition[i], QTL_Add_Fit[i], QTL_Dom_Fit[i],stringtypea, 0, stringfreq);
                    population_QTL.push_back(tempb);
                }
            }
            fstream checkqtl; checkqtl.open(qtl_class_object, std::fstream::out | std::fstream::trunc); checkqtl.close(); /* Deletes Previous Simulation */
            /* Save to a file that will continually get updated based on new mutation and frequencies */
            ofstream output;
            output.open (qtl_class_object);
            for(int i = 0; i < population_QTL.size(); i++)
            {
                output << population_QTL[i].getLocation() << " " << population_QTL[i].getAdditiveEffect() << " ";
                output << population_QTL[i].getDominanceEffect() << " " << population_QTL[i].getType() << " ";
                output << population_QTL[i].getGenOccured() << " " << population_QTL[i].getFreq() << endl;
            }
            output.close();
            logfile << "   - Copied Founder Mutations to QTL class object." << endl;
            logfile << "Finished Constructing Trait Architecture." << endl << endl;
            //////////////////////////////////////////////////////////////////////////
            // Step 7: Generate hapLibrary and tabulate unique haplotypes           //
            //////////////////////////////////////////////////////////////////////////
            vector < int > hapChr(NUMBERMARKERS,0);             /* Stores the chromosome number for a given marker */
            vector < int > hapNum(NUMBERMARKERS,0);             /* Number of SNP based on string of markers */
            #pragma omp parallel for
            for(int i = 0; i < NUMBERMARKERS; i++){hapChr[i] = MarkerMapPosition[i]; hapNum[i] = i;} /* When converted to integer will always round down */
            /* Create index values to grab chunks from */
            int haplotindex = 1;                                /* Initialize haplotype id */
            for(int i = 0; i < numChr; i++)
            {
                vector < int > tempchr;                         /* stores chromosome in vector */
                vector < int > tempnum;                         /* stores index of where it is at */
                int j = 0;
                /* Grab Correct Chromosome and store in three vectors */
                while(j < (NUMBERMARKERS))
                {
                    if(hapChr[j] > i + 1){break;}
                    if(hapChr[j] < i + 1){j++;}
                    if(hapChr[j] == i + 1){tempchr.push_back(hapChr[j]); tempnum.push_back(hapNum[j]); j++;}
                }
                /* Once Have Correct Chromosome then create index based on specified window */
                int start = 0;                                  /* ID to determine where start of current fragment is */
                int end = 0;                                    /* ID to determine where end of current fragment is */
                while(end < tempnum.size())
                {
                    if(end == 0 && start == 0){end = end + (haplotypesize-1);}
                    if(end > 0 && start > 0){end = end + haplotypesize;}
                    if(end > tempnum.size()){break;}
                    int tempst = tempnum[start]; int tempen = tempnum[end]; start = end + 1;
                    hapLibrary hap_temp(haplotindex,tempst,tempen,"0");
                    haplib.push_back(hap_temp);                 /* store in vector of haplotype library objects */
                    haplotindex++;
                }
            }
            hapChr.clear(); hapNum.clear();
            /////////////////////////////////////////////////////////////////////////////////////
            // Step 6: Make Marker Panel and QTL Genotypes place into vector of Animal Objects //
            /////////////////////////////////////////////////////////////////////////////////////
            logfile << "Begin Creating Founder Population: " << endl;
            /* Deletes Previous Simulation founder low fitness individuals */
            fstream checklowfitness; checklowfitness.open(lowfitnesspath, std::fstream::out | std::fstream::trunc); checklowfitness.close();
            /* add first line as column ID's */
            std::ofstream outputlow(lowfitnesspath, std::ios_base::app | std::ios_base::out);
            outputlow << "Sire Dam Fitness QTL_Fitness" << endl;
            int ind = 1;                                                    /* ID of animal increments by one for each line */
            int LethalFounder = 0;                                          /* number of animals dead in founder */
            /* now take marker and qtl from founder vector and turn into a founder animal */
            for(int i = 0; i < founder_markers.size(); i++)
            {
                int* MarkerGenotypes = new int[founder_markers[0].size()];                      /* Marker Genotypes */
                int* QTLGenotypes = new int[founder_qtl[0].size()];                             /* QTL Genotypes */
                /* put marker genotypes into string then fill vector */
                string geno = founder_markers[i];
                for(int i = 0; i < geno.size(); i++){int temp = geno[i] - 48; MarkerGenotypes[i]= temp;}
                geno = founder_qtl[i];
                for(int i = 0; i < geno.size(); i++){int temp = geno[i] - 48; QTLGenotypes[i]= temp;}
                /* Before you put the individual in the founder population need to determine if it dies */
                /* Starts of as a viability of 1.0 */
                double relativeviability = 1.0;                         /* represents the multiplicative fitness effect across lethal and sub-lethal alleles */
                for(int i = 0; i < QTL_IndCounter; i++)
                {
                    if(QTL_Type[i] == 3 || QTL_Type[i] == 4 || QTL_Type[i] == 5)            /* Fitness QTL */
                    {
                        if(QTLGenotypes[i] == QTL_Allele[i]){relativeviability = relativeviability * (1-(QTL_Add_Fit[i]));}
                        if(QTLGenotypes[i] > 2){relativeviability = relativeviability * (1-(QTL_Dom_Fit[i] * QTL_Add_Fit[i]));}
                    }
                }
                /* now take a draw from a uniform and if less than relativeviability than survives if greater than dead */
                std::uniform_real_distribution<double> distribution5(0,1);
                double draw = distribution5(gen);
                if(draw > relativeviability)                                                    /* Animal Died due to a low fitness */
                {
                    string Qf; stringstream strStreamQf (stringstream::in | stringstream::out);
                    for (int i=0; i < QTL_IndCounter; i++){if(QTL_Type[i] == 3 || QTL_Type[i] == 4 || QTL_Type[i] == 5){strStreamQf << QTLGenotypes[i];}}
                    string QF = strStreamQf.str();
                    std::ofstream output1(lowfitnesspath, std::ios_base::app | std::ios_base::out);
                    output1 << 0 << " " << 0 << " " << relativeviability << " " <<  QF << endl;
                    LethalFounder++;
                }
                if(draw < relativeviability)                                                    /* Animal Survived */
                {
                    //////////////////////////////////////////////////////////////////////////
                    // Step 7: Create founder file that has everything set for Animal Class //
                    //////////////////////////////////////////////////////////////////////////
                    /* Declare Variables */
                    double GenotypicValue = 0.0;                                            /* Stores Genotypic value; resets to zero for each line */
                    double BreedingValue = 0.0;                                             /* Stores Breeding Value; resets to zero for each line */
                    double DominanceDeviation = 0.0;                                        /* Stores Dominance Deviation; resets to zero for each line */
                    double Residual = 0.0;                                                  /* Stores Residual Value; resets to zero for each line */
                    double Phenotype = 0.0;                                                 /* Stores Phenotype; resets to zero for each line */
                    double Homoz = 0.0;                                                     /* Stores homozygosity based on marker information */
                    double sex;                                                             /* draw from uniform to determine sex */
                    int Sex;                                                                /* Sex of the animal 0 is male 1 is female */
                    double residvar = 1 - (Variance_Additiveh2 + Variance_Dominanceh2);     /* Residual Variance; Total Variance equals 1 */
                    residvar = sqrt(residvar);                                              /* random number generator need standard deviation */
                    /* Determine Sex of the animal based on draw from uniform distribution; if sex < 0.5 sex is 0 if sex >= 0.5 */
                    sex = distribution5(gen);
                    if(sex < 0.5){Sex = 0;}         /* Male */
                    if(sex >= 0.5){Sex = 1;}        /* Female */
                    /* Calculate Genotypic Value */
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if(QTL_Type[i] == 2 || QTL_Type[i] == 3)            /* Quantitative QTL */
                        {
                            int tempgeno;
                            if(QTLGenotypes[i] == 0 || QTLGenotypes[i] == 2){tempgeno = QTLGenotypes[i];}
                            if(QTLGenotypes[i] == 3 || QTLGenotypes[i] == 4){tempgeno = 1;}
                            /* Breeding value is only a function of additive effects */
                            BreedingValue += tempgeno * QTL_Add_Quan[i];
                            if(tempgeno != 1){GenotypicValue += tempgeno * QTL_Add_Quan[i];}     /* Not a heterozygote so only a function of additive */
                            if(tempgeno == 1)
                            {
                                GenotypicValue += (tempgeno * QTL_Add_Quan[i]) + QTL_Dom_Quan[i];    /* Heterozygote so need to include add and dom */
                                DominanceDeviation += QTL_Dom_Quan[i];
                            }
                        }
                    }
                    /* Calculate Homozygosity only in the MARKERS */
                    for(int i=0; i < Marker_IndCounter; i++)
                    {
                        if(MarkerGenotypes[i] == 0 || MarkerGenotypes[i] == 2){Homoz += 1;}
                        if(MarkerGenotypes[i] == 3 || MarkerGenotypes[i] == 4){Homoz += 0;}
                    }
                    Homoz = (Homoz/(Marker_IndCounter));
                    /* Count number of homozygous fitness loci */
                    int homozygouscount_lethal=0; int homozygouscount_sublethal=0; int heterzygouscount_lethal=0; int heterzygouscount_sublethal=0;
                    double lethalequivalent = 0.0;
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if(QTL_Type[i] == 3 || QTL_Type[i] == 4 || QTL_Type[i] == 5)            /* Fitness QTL */
                        {
                            if(QTLGenotypes[i] == QTL_Allele[i])
                            {
                                if(QTL_Type[i] == 4){homozygouscount_lethal += 1;}
                                if(QTL_Type[i] == 3 || QTL_Type[i] == 5){homozygouscount_sublethal += 1;}
                                lethalequivalent += QTL_Add_Fit[i];
                            }
                            if(QTLGenotypes[i] > 2)
                            {
                                if(QTL_Type[i] == 4){heterzygouscount_lethal += 1;}
                                if(QTL_Type[i] == 3 || QTL_Type[i] == 5){heterzygouscount_sublethal += 1;}
                                lethalequivalent += QTL_Add_Fit[i];
                            }
                        }
                    }
                    /* put marker, qtl and fitness into string to store */
                    stringstream strStreamM (stringstream::in | stringstream::out);
                    for (int i=0; i < Marker_IndCounter; i++){strStreamM << MarkerGenotypes[i];} string MA = strStreamM.str();
                    stringstream strStreamQt (stringstream::in | stringstream::out);
                    for (int i=0; i < QTL_IndCounter; i++){strStreamQt << QTLGenotypes[i];} string QT = strStreamQt.str();
                    /* Sample from standard normal to generate environmental effect */
                    std::normal_distribution<double> distribution6(0.0,residvar);
                    Residual = distribution6(gen);
                    Phenotype = GenotypicValue + Residual;
                    double rndselection = distribution5(gen);
                    double rndculling = distribution5(gen);
                    /* ID Sire Dam Sex Gen Age Prog Mts Dead-Prog RndSel RndCul Ped_F Gen_F H1F H2F H3F FitHomo FitHeter LethEquiv Homozy EBV Acc P GV R M QN QF PH MH*/
                    Animal animal(ind,0,0,Sex,0,1,0,0,0,rndselection,rndculling,0.0,0.0,0.0,0.0,0.0,homozygouscount_lethal, heterzygouscount_lethal, homozygouscount_sublethal, heterzygouscount_sublethal,lethalequivalent,Homoz,0.0,0.0,Phenotype,relativeviability,GenotypicValue,BreedingValue,DominanceDeviation,Residual,MA,QT,"","","");
                    /* Then place given animal object in population to store */
                    population.push_back(animal);
                    ind++;                              /* Increment Animal ID by one for next individual */
                }
                delete [] MarkerGenotypes; delete [] QTLGenotypes;
            }
            logfile << "   - Number of Founder's that Died due to fitness: " << LethalFounder << endl;
            NumDeadFitness[0] = LethalFounder;
            double expectedhet = 0.0; vector < string > population_marker;
            for(int i = 0; i < population.size(); i++){population_marker.push_back(population[i].getMarker());}
            double* tempfreqexphet = new double[population_marker[0].size()];               /* Array that holds SNP frequencies that were declared as Markers*/
            frequency_calc(population_marker, tempfreqexphet);                              /* Function to calculate snp frequency */
            for(int i = 0; i < population_marker[0].size(); i++)
            {
                expectedhet += (1 - ((tempfreqexphet[i]*tempfreqexphet[i]) + ((1-tempfreqexphet[i])*(1-tempfreqexphet[i]))));
            }
            expectedhet /= double(population_marker[0].size());
            population_marker.clear(); delete [] tempfreqexphet;
            ExpectedHeter[0] = expectedhet;
            logfile << "   - Calculated expected heterozygosity: " << expectedhet << endl;
            logfile << "Finished Creating Founder Population (Size: " << population.size() << ")." << endl << endl;
            // Compute mean genotypic value in founder generation
            double BaseGenGV = 0.0; double BaseGenBV = 0.0; double BaseGenDD = 0.0;
            for(int i = 0; i < population.size(); i++)
            {
                BaseGenGV += population[i].getGenotypicValue();
                BaseGenBV += population[i].getBreedingValue();
                BaseGenDD += population[i].getDominanceDeviation();
                if(i == population.size() -1)
                {
                    BaseGenGV = BaseGenGV / population.size();
                    BaseGenBV = BaseGenBV / population.size();
                    BaseGenDD = BaseGenDD / population.size();
                }
            }
            for(int i =0; i < population.size(); i++){population[i].BaseGV(BaseGenGV,BaseGenBV,BaseGenDD);}
            logfile << "Set Mean GV, BV, and DD in Founder Population to Zero." << endl << endl;
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Create Haplotype Library based on Founder Generation and compute diagonals of relationship matrix                //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            logfile << "Begin Creating Haplotype Library and assigning haplotypes IDs to individuals: " << endl;
            vector < string > AnimalPatHap(population.size(),"");                                               /* Stores haplotype ID's in a string */
            vector < string > AnimalMatHap(population.size(),"");                                               /* Stores haplotype ID's in a string */
            for(int i = 0; i < haplib.size(); i++)
            {
                /* First step is to get number of unique haplotypes to set dimension of Haplotype similarity matrix (H) */
                /* stores each unique ROH roh */
                vector < string > haplotypes(population.size()*2,"");
                vector < string > animalpatstring(population.size(),"");
                vector < string > animalmatstring(population.size(),"");
                /* loop across animals */
                #pragma omp parralel for
                for(int j = 0; j < population.size(); j++)
                {
                    if(i == 0){AnimalPatHap.push_back(""); AnimalMatHap.push_back("");}
                    string homo1 =  (population[j].getMarker()).substr(haplib[i].getStart(),haplotypesize);     /* Paternal haplotypes */
                    string homo2 = homo1;                                                                       /* Maternal haplotypes */
                    for(int g = 0; g < homo1.size(); g++)
                    {
                        if(homo1[g] == '0'){homo1[g] = '1';} if(homo2[g] == '0'){homo2[g] = '1';}               /* a1a1 genotype */
                        if(homo1[g] == '2'){homo1[g] = '2';} if(homo2[g] == '2'){homo2[g] = '2';}               /* a2a2 genotype */
                        if(homo1[g] == '3'){homo1[g] = '1';} if(homo2[g] == '3'){homo2[g] = '2';}               /* a1a2 genotype */
                        if(homo1[g] == '4'){homo1[g] = '2';} if(homo2[g] == '4'){homo2[g] = '1';}               /* a2a1 genotype */
                    }
                    animalpatstring[j] = homo1;
                    animalmatstring[j] = homo2;
                    haplotypes[(j*2)+0] = homo1;
                    haplotypes[(j*2)+1] = homo2;
                }
                /* Now Sort them and only keep unique ones */
                sort(haplotypes.begin(),haplotypes.end());
                haplotypes.erase(unique(haplotypes.begin(),haplotypes.end()),haplotypes.end());
                #pragma omp parralel for
                for(int j = 0; j < population.size(); j++)
                {
                    int k = 0;
                    string foundpat = "nope"; string foundmat = "nope";
                    /* assign paternal and maternal string to a numeric value */
                    while(foundpat == "nope" || foundmat == "nope")
                    {
                        if(animalpatstring[j] == haplotypes[k])
                        {
                            if(i <= haplib.size() - 2){std::ostringstream s; s << AnimalPatHap[j] << k << "_" ; AnimalPatHap[j]=s.str();}
                            if(i == haplib.size() - 1){std::ostringstream s; s << AnimalPatHap[j] << k; AnimalPatHap[j]=s.str();}
                            foundpat = "yes";
                        }
                        if(animalmatstring[j] == haplotypes[k])
                        {
                            if(i <= haplib.size() - 2){std::ostringstream s; s << AnimalMatHap[j] << k <<"_"; AnimalMatHap[j]=s.str();}
                            if(i == haplib.size() - 1){std::ostringstream s; s << AnimalMatHap[j] << k; AnimalMatHap[j]=s.str();}
                            foundmat = "yes";
                        }
                        k++;
                    }
                }
                #pragma omp parralel for
                for(int j = 0; j < population.size(); j++)
                {
                    string homo1 =  (population[j].getMarker()).substr(haplib[i].getStart(),haplotypesize);     /* Paternal haplotypes */
                    string homo2 = homo1;                                                                       /* Maternal haplotypes */
                    for(int g = 0; g < homo1.size(); g++)
                    {
                        if(homo1[g] == '0'){homo1[g] = '1';} if(homo2[g] == '0'){homo2[g] = '1';}               /* a1a1 genotype */
                        if(homo1[g] == '2'){homo1[g] = '2';} if(homo2[g] == '2'){homo2[g] = '2';}               /* a2a2 genotype */
                        if(homo1[g] == '3'){homo1[g] = '1';} if(homo2[g] == '3'){homo2[g] = '2';}               /* a1a2 genotype */
                        if(homo1[g] == '4'){homo1[g] = '2';} if(homo2[g] == '4'){homo2[g] = '1';}               /* a2a1 genotype */
                    }
                    /****************************************************/
                    /* Haplotype 1 Matrix (Based on Hickey et al. 2012) */
                    /****************************************************/
                    float sum=0.0;
                    double hvalue = 0.0;
                    for(int g = 0; g < homo1.size(); g++)
                    {
                        sum += abs(homo1[g] - homo2[g]);
                        if(g == homo1.size() - 1){hvalue = (1 - (sum/homo1.size())) + 1;}   /* don't need to divide by 2 because is 1 + 1 + (1-sum) + (1-sum) */
                    }
                    population[j].AccumulateH1(hvalue);                                             /* Add to diagonal of population */
                    /****************************************************/
                    /* Haplotype 2 Matrix (Based on Hickey et al. 2012) */
                    /****************************************************/
                    hvalue = 0.0;
                    int match[homo1.size()];                                                        /* matrix that has 1 to match and 0 if not match */
                    for(int g = 0; g < homo1.size(); g++){match[g] = 1 - abs(homo1[g] - homo2[g]);} /* Create match vector */
                    double sumGlobal = 0;
                    double sumh2=0;
                    for(int g = 0; g < homo1.size(); g++)
                    {
                        if(match[g] < 1)
                        {
                            sumGlobal = sumGlobal + sumh2*sumh2;
                            sumh2 = 0;
                        } else {
                            sumh2 = sumh2 + 1;
                        }
                    }
                    /* don't need to divide by 2 because is 1 + 1 + sqrt(Sum/length) + sqrt(Sum/length) */
                    hvalue = 1 + sqrt((sumGlobal + (sumh2 * sumh2)) / (homo1.size() * homo1.size()));
                    population[j].AccumulateH2(hvalue);                                             /* Add to diagonal of population */
                    /*****************************************************/
                    /* Haplotype 3 Matrix (Similar to Pryce et al. 2011) */
                    /*****************************************************/
                    double sumh3 = 0.0;
                    hvalue = 0.0;
                    for(int g = 0; g < homo1.size(); g++){sumh3 += match[g];}
                    if(sumh3 == homo1.size()){hvalue = 2.0;}
                    if(sumh3 != homo1.size()){hvalue = 1.0;}
                    population[j].AccumulateH3(hvalue);                                             /* Add to diagonal of population */
                    /* Finished with 3 haplotype based methods */
                    /* Once reached last individual put all unique haplotypes into string with a "_" delimter to split them apart later */
                    if(j == population.size() -1)
                    {   string temp;
                        for(int h = 0; h < haplotypes.size(); h++)
                        {
                            if(h == 0){temp = haplotypes[h];}
                            if(h > 0){temp = temp + "_" + haplotypes[h];}
                        }
                        haplib[i].UpdateHaplotypes(temp);
                    }
                    /* Once get to last haplotype segment need to standardize by number of haplotype segments */
                    if(i == haplib.size() - 1)
                    {
                        double denom = haplib.size();
                        population[j].StandardizeH1(denom); population[j].StandardizeH2(denom); population[j].StandardizeH3(denom);
                        population[j].Update_PatHap(AnimalPatHap[j]); population[j].Update_MatHap(AnimalMatHap[j]);
                    }
                }
            }
            logfile << "Finished Creating Haplotype Library and assigning haplotypes IDs to individuals: " << endl << endl;
            if(Create_LD_Decay == "yes")                            /* Estimate LD Decay */
            {
                /* Clear previous simulation */
                fstream checkldfile; checkldfile.open(LD_Decay_File, std::fstream::out | std::fstream::trunc); checkldfile.close();
                /* Vector of string of markers */
                vector < string > markergenotypes;
                for(int i = 0; i < population.size(); i++){markergenotypes.push_back(population[i].getMarker());}
                ld_decay_estimator(LD_Decay_File,Marker_Map,"yes",markergenotypes);      /* Function to calculate ld decay */
                markergenotypes.clear();
            }
            //////////////////////////////////////////////////////////////////////////
            // Create Distribution of Mating Pairs to Draw From                     //
            //////////////////////////////////////////////////////////////////////////
            logfile << "Distribution of Mating Pairs \n";
            /* Plot distribution of mating pairs as an animal gets older */
            const int n=10000;                          /* number of samples to draw */
            const int nstars=100;                      /* maximum number of stars to distribute for plot */
            const int nintervals=20;                   /* number of intervals for plot */
            vector < double > number(10000,0.0);        /* stores samples */
            int p[nintervals] = {};                    /* stores number in each interval */
            sftrabbit::beta_distribution<> beta(BetaDist_alpha, BetaDist_beta);     /* Beta Distribution */
            for (int i = 0; i < n; ++i){number[i] = beta(gen); ++p[int(nintervals*number[i])];}                  /* Get sample and put count in interval */
            /* Plots disribution based on alpha and beta */
            logfile << "Beta distribution (" << BetaDist_alpha << "," << BetaDist_beta << "):" << endl;
            for (int i=0; i<nintervals; ++i)
            {
                logfile << float(i)/nintervals << "-" << float(i+1)/nintervals << ": " << "\t" << std::string(p[i]*nstars/n,'*') << std::endl;
            }
            sort(number.begin(),number.end());
            logfile << "\nAllele Frequencies in Founder \n";
            /* Allele frequencies should be from a population that has been unselected; get allele frequency counts in from unselected base population */
            vector < string > tempgenofreqcalc;
            for(int i = 0; i < population.size(); i++){tempgenofreqcalc.push_back(population[i].getMarker());}
            double* founderfreq = new double[tempgenofreqcalc[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
            frequency_calc(tempgenofreqcalc, founderfreq);                              /* Function to calculate snp frequency */
            double sum = 0; double sq_sum = 0; double cLower = 0.5; double cUpper = 0.5;
            for(int i = 0; i < tempgenofreqcalc[0].size(); i++){sum += founderfreq[i];}
            double mean = sum / tempgenofreqcalc[0].size();
            for(int i = 0; i < tempgenofreqcalc[0].size(); i++){sq_sum += founderfreq[i] * founderfreq[i];}
            double stdev = sqrt(sq_sum / tempgenofreqcalc[0].size() - mean * mean);
            for(int i = 0; i < tempgenofreqcalc[0].size(); i++)
            {
                if(founderfreq[i] > 0.5){if(founderfreq[i] > cUpper){cUpper = founderfreq[i];}}
                if(founderfreq[i] < 0.5){if(founderfreq[i] < cLower){ cLower = founderfreq[i];}}
            }
            logfile << "Statistics for Marker Frequency in Base population." << endl;
            logfile << "   - mean " << mean<< endl << "   - Standard Deviation "<< stdev<< endl;
            logfile << "   - Min "<< cLower<< endl << "   - Max "<< cUpper<< endl;
            /* Create M which is of dimension 3 by number of markers */
            /* Create M which is of dimension 3 by number of markers */
            double* M = new double[3*tempgenofreqcalc[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
            for(int i = 0; i < 3; i++){for(int j = 0; j < tempgenofreqcalc[0].size(); j++){M[(i*tempgenofreqcalc[0].size())+j] = i - (2 * founderfreq[j]);}}
            /* Calculate Scale */
            float scale = 0;
            for (int j=0; j < tempgenofreqcalc[0].size(); j++){scale += (1 - founderfreq[j]) * founderfreq[j];}
            scale = scale * 2;
            logfile << "   - M Matrix Calculated which is based on frequencies from founder generation. " << endl;
            logfile << "   - Scale Factor used to construct G: " << scale << endl << endl;
            /* calculate diagonals of genomic relationship matrix; had to wait until now because needed allele frequencies in base generation */
            for(int i = 0; i < population.size(); i++)
            {
                string Geno = population[i].getMarker();                        /* Grab Genotype for Individual i */
                float S = 0.0;
                for(int k = 0; k < Geno.size(); k++)                            /* Loop Across Markers to create cell within G */
                {
                    double temp = Geno[k] - 48;                                 /* Create Z for a given SNP */
                    if(temp == 3 || temp == 4){temp = 1;}                       /* Convert 3 or 4 to 1 */
                    temp = (temp - 1) - (2 * (founderfreq[k] - 0.5));           /* Convert it to - 1 0 1 and then put it into Z format */
                    S += temp * temp;                                           /* Multipy then add to sum */
                }
                double diagInb = S / scale;
                population[i].UpdateGenInb(diagInb);
            }
            tempgenofreqcalc.clear();
            int TotalAnimalNumber = 0;                  /* Counter to determine how many animals their are for full matrix sizes */
            int TotalOldAnimalNumber = 0;               /* Counter to determine size of old animal matrix */
            /* Deletes files from Previous Simulation */
            fstream checkphenoped; checkphenoped.open(Pheno_Pedigree_File, std::fstream::out | std::fstream::trunc); checkphenoped.close();
            fstream checkphenogen; checkphenogen.open(Pheno_GMatrix_File, std::fstream::out | std::fstream::trunc); checkphenogen.close();
            fstream checkmasterdf; checkmasterdf.open(Master_DF_File, std::fstream::out | std::fstream::trunc); checkmasterdf.close();
            fstream checkmastergeno; checkmastergeno.open(Master_Genotype_File, std::fstream::out | std::fstream::trunc); checkmastergeno.close();
            std::ofstream outputmastgeno(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
            outputmastgeno << "ID Marker QTL" << endl;
            time_t t_end = time(0);
            cout << "Constructed Trait Architecture and Founder Genomes. (Took: " << difftime(t_end,t_start) << " seconds)" << endl << endl;
            ////////////////////////////////////////////////////////////////////////////////////
            //////      Loop through based on Number of Generations you want simulated    //////
            ////////////////////////////////////////////////////////////////////////////////////
            cout << "Begin Simulating Generations:" << endl;
            for(int Gen = 1; Gen < (GENERATIONS + 1); Gen++)
            {
                time_t intbegin_time = time(0);
                if(Gen > 1){TotalOldAnimalNumber = TotalAnimalNumber;}                              /* Size of old animal matrix */
                logfile << "------ Begin Generation " << Gen << " -------- " << endl;
                time_t start_block = time(0); time_t start; time_t end;
                /* Output animals that are of age 1 into pheno_pedigree and Pheno_Gmatrix to use for relationships */
                /* That way when you read them back in to create relationship matrix don't need to order them */
                /* Save as a continuous string and then output */
                stringstream outputstringpedigree(stringstream::out); stringstream outputstringgenomic(stringstream::out); int outputnumpedgen = 0;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1)
                    {
                        /* For pedigree */
                        outputstringpedigree << population[i].getID() << " " << population[i].getSire() << " " << population[i].getDam() << " ";
                        outputstringpedigree << population[i].getPhenotype() << endl;
                        /* For Genomic */
                        outputstringgenomic << population[i].getID() << " " << population[i].getPhenotype() << " " << population[i].getMarker() << " ";
                        outputstringgenomic << population[i].getPatHapl() << " " << population[i].getMatHapl() <<endl; outputnumpedgen++;
                        TotalAnimalNumber++;                                /* to keep track of number of animals */
                    }
                    if(outputnumpedgen % 1000 == 0)
                    {
                        /* output pheno pedigree file */
                        std::ofstream output1(Pheno_Pedigree_File, std::ios_base::app | std::ios_base::out);
                        output1 << outputstringpedigree.str(); outputstringpedigree.str(""); outputstringpedigree.clear();
                        /* output pheno genomic file */
                        std::ofstream output2(Pheno_GMatrix_File, std::ios_base::app | std::ios_base::out);
                        output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
                    }
                }
                /* output pheno pedigree file */
                std::ofstream output1(Pheno_Pedigree_File, std::ios_base::app | std::ios_base::out);
                output1 << outputstringpedigree.str(); outputstringpedigree.str(""); outputstringpedigree.clear();
                /* output pheno genomic file */
                std::ofstream output2(Pheno_GMatrix_File, std::ios_base::app | std::ios_base::out);
                output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
                /* Get ID for last animal; Start ID represent the first ID that will be used for first new animal */
                int StartID = population.size();
                StartID = population[StartID - 1].getID() + 1;
                if(scalefactaddh2 != 0)
                {
                    logfile << "   Generate Estimated Breeding Values based on " << EBV_Calc << " information:" << endl;
                    VectorXd lambda(1);                                                 /* Declare scalar vector for alpha */
                    lambda(0) = (1 - Variance_Additiveh2) / Variance_Additiveh2;        /* Shrinkage Factor for MME */
                    vector < double > Phenotype(TotalAnimalNumber,0.0);                 /* Vector of phenotypes */
                    vector < int > animal(TotalAnimalNumber,0);                         /* Array to store Animal IDs */
                    vector < int > sire(TotalAnimalNumber,0);                           /* Array to store Sire IDs */
                    vector < int > dam(TotalAnimalNumber,0);                            /* Array to store Dam IDs */
                    logfile << "       - Size of Relationship Matrix: " << TotalAnimalNumber << " X " << TotalAnimalNumber << "." << endl;
                    MatrixXd Relationship(TotalAnimalNumber,TotalAnimalNumber);
                    MatrixXd Relationshipinv(TotalAnimalNumber,TotalAnimalNumber);
                    if(EBV_Calc == "h1" || EBV_Calc == "h2" || EBV_Calc == "ROH" || EBV_Calc == "genomic")
                    {
                        /* G is set up different but inverse derivation is the same for Gen1 or when updating */
                        if(Gen == 1)
                        {
                            if(EBV_Calc == "h1" || EBV_Calc == "h2" || EBV_Calc == "ROH")
                            {
                                logfile << "           - Begin Constructing " << EBV_Calc << " Relationship Matrix." << endl;
                                /* Initialize Relationship Matrix as 0.0 */
                                for(int ind1 = 0; ind1 < TotalAnimalNumber; ind1++)
                                {
                                    for(int ind2 = 0; ind2 < TotalAnimalNumber; ind2++){Relationship(ind1,ind2) = 0.0;}
                                }
                                /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
                                /* This way you don't have to repeat this step for each haplotype */
                                start = time(0);
                                vector < vector < int > > PaternalHaplotypeIDs;
                                vector < vector < int > > MaternalHaplotypeIDs;
                                for(int i = 0; i < population.size(); i++)
                                {
                                    animal[i] = population[i].getID();                                      /* Grab Animal ID */
                                    Phenotype[i] = population[i].getPhenotype();                            /* Grab Phenotype for Individual i */
                                    string PaternalHap = population[i].getPatHapl();                        /* Grab Paternal Haplotype for Individual i */
                                    string MaternalHap = population[i].getMatHapl();                        /* Grab Maternal Haplotype for Individual i */
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
                                    temp_pat.clear(); temp_mat.clear();
                                }
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
                                            if(EBV_Calc == "h1")
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
                                            if(EBV_Calc == "h2")
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
                                            if(EBV_Calc == "ROH")
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
                                    for(int ind1 = 0; ind1 < population.size(); ind1++)
                                    {
                                        for(int ind2 = ind1; ind2 < population.size(); ind2++)
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
                                for(int i = 0; i < population.size(); i++){Relationship(i,i) += 1e-5;}
                                end = time(0);
                                logfile << "           - Finished constructing Genomic Relationship Matrix. " << endl;
                                logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                                Eigen::write_binary(BinaryG_Matrix_File.c_str(),Relationship);   /* Output Relationship Matrix into Binary for next generation */
                            }
                            if(EBV_Calc == "genomic")
                            {
                                logfile << "           - Begin Constructing Genomic Relationship Matrix." << endl;
                                start = time(0);
                                vector < string > creategenorel;
                                for(int i = 0; i < population.size(); i++)
                                {
                                    /* Grab Animal ID and Phenotype for each individual; Grab genotype for GRM function */
                                    animal[i] = population[i].getID(); Phenotype[i] = population[i].getPhenotype(); creategenorel.push_back(population[i].getMarker());
                                }
                                double *_grm_mkl = new double[population.size()*population.size()];     /* Allocate Memory for GRM */
                                grm_noprevgrm(M,creategenorel,_grm_mkl,scale);                          /* Function to create GRM, with no previous grm */
                                int i, j;
                                /* Copy to Eigen Matrix */
                                #pragma omp parallel for private(i)
                                for(i = 0; i < population.size(); i++)
                                {
                                    for(j = 0; j <= i; j++)
                                    {
                                        Relationship(i,j) = Relationship(j,i) = _grm_mkl[(i*population.size())+j];
                                        if(i == j){Relationship(i,j) += 1e-5;}
                                    }
                                }
                                delete[] _grm_mkl; creategenorel.clear();                                         /* free memory */
                                end = time(0);
                                logfile << "           - Finished constructing Genomic Relationship Matrix." << endl;
                                logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                                Eigen::write_binary(BinaryG_Matrix_File.c_str(),Relationship);/* Output Genomic Relationship Matrix into Binary for next generation */
                            }
                            logfile << "           - Begin Constructing Genomic Relationship Inverse (Using " << Geno_Inverse << ")." << endl;
                            start = time(0);
                            if(Geno_Inverse == "recursion")
                            {
                                MatrixXd mg(TotalAnimalNumber,1);                              /* m vector in Misztal et al. (2014) */
                                MatrixXd pg(TotalAnimalNumber,TotalAnimalNumber);              /* p matrix in Misztal et al. (2014) */
                                /* Set matrices to zero */
                                Relationshipinv = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                                pg = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                                for(int i = 0; i < TotalAnimalNumber; i++){Relationshipinv(i,i) = 0; pg(i,i) = 0; mg(i,0) = 0;}
                                // Step 1
                                Relationshipinv(0,0) = 1 / Relationship(0,0); mg(0,0) = Relationship(0,0);
                                // Step 2: start at 2
                                for(int i = 1; i < TotalAnimalNumber; i++)
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
                                Eigen::write_binary(Binarym_Matrix_File.c_str(),mg);                 /* Output m Matrix into Binary */
                                Eigen::write_binary(Binaryp_Matrix_File.c_str(),pg);                 /* Output p Matrix into Binary */
                                Eigen::write_binary(BinaryGinv_Matrix_File.c_str(),Relationshipinv); /* Output Relationship Inverse Matrix into Binary */
                            }
                            if(Geno_Inverse == "cholesky")
                            {
                                /* Set up parameters used for mkl variables */
                                unsigned long i_p = 0, j_p = 0; long long int info = 0; char lower='L'; char diag='N';
                                unsigned long n_a = TotalAnimalNumber; const long long int int_n =(int)n_a;
                                const long long int increment = int(1); const long long int int_na =(int)n_a * (int)n_a;
                                double *GRM = new double[n_a*n_a];                      /* GRM */
                                double *Linv = new double[n_a*n_a];                     /* Choleskey Inverse */
                                MatrixXd LINV(TotalAnimalNumber,TotalAnimalNumber);     /* need to save for next generation */
                                /* Copy it to a 2-dim array that is dynamically stored that all the computations will be on */
                                #pragma omp parallel for private(j_p)
                                for(i_p=0; i_p<n_a; i_p++)
                                {
                                    for(j_p=0; j_p < n_a; j_p++){GRM[i_p*n_a+j_p]=Relationship(i_p,j_p);}
                                }
                                dpotrf(&lower,&int_n,GRM,&int_n, &info);            /* Calculate upper triangular L matrix */
                                dcopy(&int_na,GRM,&increment,Linv,&increment);      /* Copy to vector to calculate Linv */
                                dtrtri (&lower,&diag,&int_n,Linv,&int_n,&info);     /* Calculate Linv */
                                dpotri(&lower,&int_n,GRM, &int_n,&info);            /* Calculate inverse of upper triangular matrix result is the inverse */
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
                                /* Copy GRM to a eigen matrix to save */
                                #pragma omp parallel for private(j_p)
                                for(j_p=0; j_p<n_a; j_p++)
                                {
                                    for(i_p=0; i_p<=j_p; i_p++){Relationshipinv(i_p,j_p) = Relationshipinv(j_p,i_p) = GRM[i_p*n_a+j_p];}
                                }
                                Eigen::write_binary(BinaryLinv_Matrix_File.c_str(),LINV);                 /* Output Linv Matrix into Binary */
                                Eigen::write_binary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);/* Output Relationship Inverse Matrix into Binary */
                                delete[] GRM; delete[] Linv; LINV.resize(0,0);
                            }
                            end = time(0);
                            logfile << "           - Finished constructing Genomic Relationship Inverse. " << endl;
                            logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                        }
                        if(Gen > 1)
                        {
                            if(EBV_Calc == "h1" || EBV_Calc == "h2" || EBV_Calc == "ROH")
                            {
                                logfile << "           - Begin Constructing " << EBV_Calc << " Relationship Matrix." << endl;
                                /* Initialize Relationship Matrix as 0.0 */
                                for(int ind1 = 0; ind1 < TotalAnimalNumber; ind1++)
                                {
                                    for(int ind2 = 0; ind2 < TotalAnimalNumber; ind2++){Relationship(ind1,ind2) = 0.0;}
                                }
                                /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
                                /* This way you don't have to repeat this step for each haplotype */
                                start = time(0);
                                vector < vector < int > > PaternalHaplotypeIDs;
                                vector < vector < int > > MaternalHaplotypeIDs;
                                /* read in all animals haplotype ID's; Don't need to really worry about this getting big */
                                int linenumber = 0;
                                string line;
                                ifstream infile2;
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
                                /* Animal Haplotype ID's have been put in 2-D vector to grab */
                                MatrixXd OldRelationship(TotalOldAnimalNumber,TotalOldAnimalNumber);        /* Used to store old animal G */
                                Eigen::read_binary(BinaryG_Matrix_File.c_str(),OldRelationship);            /* Read in old G */
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
                                            if(EBV_Calc == "h1")
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
                                            if(EBV_Calc == "h2")
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
                                            if(EBV_Calc == "ROH")
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
                                                if(i == haplib.size() - 1)
                                                {
                                                    Relationship(ind1,ind2) = Relationship(ind1,ind2) / double(haplib.size());
                                                    if(ind1 == ind2){Relationship(ind1,ind2) += 1e-5;}
                                                    if(ind1 != ind2){Relationship(ind2,ind1) = Relationship(ind2,ind1) / double(haplib.size());}
                                                }
                                            }
                                        } /* Finish loop across ind2 */
                                    } /* Finish loop across ind1 */
                                } /* Loop across haplotypes */
                                end = time(0);
                                logfile << "           - Finished constructing Genomic Relationship Matrix. " << endl;
                                logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                                Eigen::write_binary(BinaryG_Matrix_File.c_str(),Relationship);  /* Output Relationship Matrix into Binary */
                            }
                            if(EBV_Calc == "genomic")
                            {
                                logfile << "           - Begin Constructing Genomic Relationship Matrix." << endl;
                                start = time(0);
                                MatrixXd OldRelationship(TotalOldAnimalNumber,TotalOldAnimalNumber);
                                Eigen::read_binary(BinaryG_Matrix_File.c_str(),OldRelationship);
                                /* Fill Relationship matrix with previously calcuated cells */
                                for(int i = 0; i < TotalOldAnimalNumber; i++)
                                {
                                    for(int j= 0; j < TotalOldAnimalNumber; j++){Relationship(i,j) = OldRelationship(i,j);}
                                }
                                OldRelationship.resize(0,0);
                                vector < string > newanimalgeno;
                                for(int ind = 0; ind < population.size(); ind++)
                                {
                                    if(population[ind].getAge() == 1){newanimalgeno.push_back(population[ind].getMarker());}
                                }
                                /* Figure out number of new animals */
                                int newanimals = 0;
                                for(int i = 0; i < population.size(); i++){if(population[i].getAge() == 1){newanimals++;}}
                                double *_grm_mkl_12 = new double[TotalOldAnimalNumber*newanimals];      /* Allocate Memory for G12 */
                                double *_grm_mkl_22 = new double[newanimals*newanimals];                /* Allocate Memory for G22 */
                                grm_prevgrm(M,Pheno_GMatrix_File,newanimalgeno,_grm_mkl_12,_grm_mkl_22,scale,animal,Phenotype); /* function to create G12 & G22 of GRM */
                                newanimalgeno.clear();
                                /* Fill G12 into eigen relationship matrix */
                                for(int i = 0; i < TotalOldAnimalNumber; i++)
                                {
                                    int startofinner = 0;
                                    for(int j = TotalOldAnimalNumber; j < (TotalOldAnimalNumber+newanimals); j++)
                                    {
                                        Relationship(i,j) = Relationship(j,i) = _grm_mkl_12[(i*newanimals)+startofinner]; startofinner++;
                                    }
                                }
                                delete [] _grm_mkl_12;
                                /* Fill G22 into eigen relationship matrix */
                                int newanimalLocation_i = TotalOldAnimalNumber;
                                for(int i = 0; i < newanimals; i++)
                                {
                                    int newanimalLocation_j = TotalOldAnimalNumber;
                                    for(int j = 0; j < newanimals; j++)
                                    {
                                        Relationship(newanimalLocation_i,newanimalLocation_j) = _grm_mkl_22[(i*newanimals)+j];
                                        if(i == j){Relationship(newanimalLocation_i,newanimalLocation_j) = Relationship(newanimalLocation_i,newanimalLocation_j) + 1e-5;}
                                        newanimalLocation_j++;
                                    }
                                    newanimalLocation_i++;
                                }
                                delete [] _grm_mkl_22;
                                end = time(0);
                                logfile << "           - Finished constructing Genomic Relationship Matrix. " << endl;
                                logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                                Eigen::write_binary(BinaryG_Matrix_File.c_str(),Relationship);  /* Output Genomic Relationship Matrix into Binary for */
                            }
                            logfile << "           - Begin Constructing Genomic Relationship Inverse (Using " << Geno_Inverse << ")." << endl;
                            start = time(0);
                            if(Geno_Inverse == "recursion")
                            {
                                // Need to fill mg, pg and Relationshipinv first with old animals
                                MatrixXd Old_m(TotalOldAnimalNumber,1);                                 /* Used to store old animals for matrix m */
                                MatrixXd Old_p(TotalOldAnimalNumber,TotalOldAnimalNumber);              /* Used to store old animals for matrix p */
                                MatrixXd Old_Ginv(TotalOldAnimalNumber,TotalOldAnimalNumber);           /* Used to store old animals for matrix Ginv */
                                // Matrices for current generation
                                MatrixXd mg(TotalAnimalNumber,1);                                       /* m vector in Misztal et al. (2014) */
                                MatrixXd pg(TotalAnimalNumber,TotalAnimalNumber);                       /* p matrix in Misztal et al. (2014) */
                                /* Set matrices to zero */
                                Relationshipinv = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                                pg = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                                for(int i = 0; i < TotalAnimalNumber; i++){Relationshipinv(i,i) = 0; pg(i,i) = 0; mg(i,0) = 0;}
                                /* Fill old matrices */
                                Eigen::read_binary(Binarym_Matrix_File.c_str(),Old_m); Eigen::read_binary(Binaryp_Matrix_File.c_str(),Old_p);
                                Eigen::read_binary(BinaryGinv_Matrix_File.c_str(),Old_Ginv);
                                /* Fill full matrices with already computed animals */
                                for(int i = 0; i < TotalOldAnimalNumber; i++)
                                {
                                    for(int j = 0; j < TotalOldAnimalNumber; j++){pg(i,j) = Old_p(i,j); Relationshipinv(i,j) = Old_Ginv(i,j);}
                                    mg(i,0) = Old_m(i,0);
                                }
                                logfile << "               - Filled Inverse Relationship Matrix with old animals." << endl;
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
                                logfile << "               - Filled Inverse Relationship Matrix for new animals." << endl;
                                Eigen::write_binary(Binarym_Matrix_File.c_str(),mg);                       /* Output m Matrix into Binary for next generation */
                                Eigen::write_binary(Binaryp_Matrix_File.c_str(),pg);                       /* Output p Matrix into Binary for next generation */
                                Eigen::write_binary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);       /* Output Ginv Matrix into Binary for next generation */
                            }
                            if(Geno_Inverse == "cholesky")
                            {
                                MatrixXd LINV_Old(TotalOldAnimalNumber,TotalOldAnimalNumber);               /* Used to store old animals for Linv */
                                MatrixXd GINV_Old(TotalOldAnimalNumber,TotalOldAnimalNumber);               /* Used to store old animals for Ginv */
                                Eigen::read_binary(BinaryLinv_Matrix_File.c_str(),LINV_Old); Eigen::read_binary(BinaryGinv_Matrix_File.c_str(),GINV_Old);
                                /* Parameters that are used for mkl functions */
                                unsigned long i_p = 0, j_p = 0;
                                const long long int newanm = (TotalAnimalNumber - TotalOldAnimalNumber);    /* Number of new animals */
                                const long long int oldanm = TotalOldAnimalNumber;                          /* Number of old animals */
                                const long long int length = int(newanm) * int(newanm);
                                const double beta = double(-1.0); const double alpha = double(1.0); const long long int increment = int(1);
                                long long int info = 0; const long long int int_n =(int)newanm; const char diag = 'N'; char lower='L';
                                /* Create Linv to save for next generation */
                                MatrixXd LINV(TotalAnimalNumber,TotalAnimalNumber);                 /* need to save for next generation */
                                LINV = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                                for(i_p = 0; i_p < (newanm + oldanm); i_p++){LINV(i_p,i_p) = 0.0;}
                                double *G11inv = new double[oldanm*oldanm];                             /* old G-inv */
                                #pragma omp parallel for private(j_p)
                                for(i_p=0; i_p < oldanm; i_p++)
                                {
                                    for(j_p=0; j_p < oldanm; j_p++){G11inv[(i_p*oldanm)+j_p] = GINV_Old(i_p,j_p);}
                                }
                                GINV_Old.resize(0,0);
                                double *L11inv = new double[oldanm*oldanm];                             /* old L-inv */
                                #pragma omp parallel for private(j_p)
                                for(i_p=0; i_p < oldanm; i_p++)
                                {
                                    for(j_p=0; j_p < oldanm; j_p++){L11inv[(i_p*oldanm)+j_p] = LINV_Old(i_p,j_p);}
                                }
                                /* Save for next generation */
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
                                Eigen::write_binary(BinaryLinv_Matrix_File.c_str(),LINV);           /* Output Linv Matrix into Binary */
                                Eigen::write_binary(BinaryGinv_Matrix_File.c_str(),Relationshipinv);/* Output Relationship Inverse Matrix into Binary */
                                /* Delete Matrices */
                                delete [] G22invnew; delete [] G11inv; delete [] L11inv; delete [] G21; delete [] G22; delete [] L22inv; delete [] L21Inv;
                                LINV.resize(0,0);
                            }
                            end = time(0);
                            logfile << "           - Finished constructing Genomic Relationship Inverse." << endl;
                            logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                        }
                        /* Still need to calculate pedigree inbreeding so do it now */
                        double *f_ped = new double[TotalAnimalNumber];
                        pedigree_inbreeding(Pheno_Pedigree_File,f_ped);                /* Function that calculates inbreeding */
                        /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
                        for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
                        {
                            int j = 0;                                                                  /* Counter for population spot */
                            while(1)
                            {
                                if(population[i].getID() == animal[j]){double temp = f_ped[j]; population[i].UpdateInb(temp); break;}
                                j++;                                                                    /* Loop across until animal has been found */
                            }
                        }
                        delete[] f_ped;
                    }
                    if(EBV_Calc == "pedigree")
                    {
                        logfile << "       - Begin Constructing A Inverse." << endl;
                        start = time(0);
                        /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
                        int linenumber = 0;                                                                 /* Counter to determine where at in pedigree index's */
                        string line;
                        ifstream infile2;
                        infile2.open(Pheno_Pedigree_File);                                                  /* This file has all animals in it */
                        if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
                        while (getline(infile2,line))
                        {
                            /* Fill each array with correct number already in order so don't need to order */
                            size_t pos = line.find(" ",0); animal[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);       /* Grab Animal ID */
                            pos = line.find(" ",0); sire[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                /* Grab Sire ID */
                            pos = line.find(" ",0); dam[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                 /* Grab Dam ID */
                            Phenotype[linenumber] = stod(line);                                                                         /* Grab Phenotype */
                            linenumber++;
                        }
                        vector < double > f_ainv(TotalAnimalNumber * TotalAnimalNumber,0);
                        vector < double > f_ped(TotalAnimalNumber,0);
                        pedigree_inverse(animal,sire,dam,f_ainv,f_ped);                    /* Function that calculates A-inverse */
                        /* Fill eigen matrix of A inverse*/
                        Relationshipinv = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                        for(int i = 0; i < TotalAnimalNumber; i++){Relationshipinv(i,i) = 0;}
                        for(int i = 0; i < TotalAnimalNumber; i++)
                        {
                            for(int j = 0; j < TotalAnimalNumber; j++){Relationshipinv(i,j) = f_ainv[(i*TotalAnimalNumber)+j];}
                        }
                        end = time(0);
                        logfile << "       - Finished Constructing Ainverse created.\n" << "               - Took: " << difftime(end,start) << " seconds." << endl;
                        /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
                        for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
                        {
                            int j = 0;                                                                  /* Counter for population spot */
                            while(1)
                            {
                                if(population[i].getID() == animal[j]){double temp = f_ped[j]; population[i].UpdateInb(temp); break;}
                                j++;                                                                    /* Loop across until animal has been found */
                            }
                        }
                    }
                    logfile << "       - Begin Solving for equations using " << Solver << " method." << endl;
                    ////////////////////////
                    ///    Create MME    ///
                    ////////////////////////
                    MatrixXd C22(TotalAnimalNumber,TotalAnimalNumber);                              /* Declare C22 */
                    MatrixXd Z(TotalAnimalNumber,TotalAnimalNumber);                                /* Declare Z */
                    Z = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);                    /* Z is same as Z'Z (i.e. diagonals of 1's) */
                    C22 = Z + (Relationshipinv * lambda(0));                                        /* C22 is Z'Z + (Ainv * lambda) */
                    Relationshipinv.resize(0,0);
                    Z.resize(0,0);
                    ///////////////////////////////////////////////////////////
                    /////    LHS and RHS for direct or pcg function       /////
                    ///////////////////////////////////////////////////////////
                    int LHSsize = TotalAnimalNumber + 1;
                    double* LHSarray = new double[LHSsize*LHSsize];                                 /* LHS dimension: animal + intercept by animal + intercept */
                    LHSarray[0] = TotalAnimalNumber;                                                /* C11 is number of observations */
                    #pragma omp parallel for
                    for(int i = 1; i < LHSsize; i++)
                    {
                        LHSarray[(0*LHSsize)+i] = 1; LHSarray[(i*LHSsize)+0] = 1;                   /* C12 & C21 total number of phenotypes per animal (i.e. 1) */
                    }
                    int outeri, innerj;
                    #pragma omp parallel for private(innerj)
                    for(int outeri = 0; outeri < TotalAnimalNumber; outeri++)
                    {
                        for(int innerj=0; innerj < TotalAnimalNumber; innerj++){LHSarray[((outeri+1)*LHSsize)+(innerj+1)] = C22(outeri,innerj);}
                    }
                    double* RHSarray = new double[LHSsize];                                                     /* RHS dimension: animal + intercept by 1 */
                    for(int i = 0; i < LHSsize; i++){RHSarray[i] = 0;}
                    for(int i = 0; i < TotalAnimalNumber; i++){RHSarray[0] += Phenotype[i];}                    /* row 1 of RHS is sum of observations */
                    for(int i = 0; i < TotalAnimalNumber; i++){RHSarray[i+1] = Phenotype[i];}                   /* Copy phenotypes to RHS */
                    logfile << "           - RHS created, Dimension (" << TotalAnimalNumber + 1 << " X " << 1 << ")." << endl;
                    logfile << "           - LHS created, Dimension (" << TotalAnimalNumber + 1 << " X " << TotalAnimalNumber + 1 << ")." << endl;
                    
                    vector < double > estimatedsolutions(LHSsize,0);                                            /* Initialize to zero */
                    if(Solver == "direct")                                                          /* Solve equations using direct inversion */
                    {
                        logfile << "           - Starting " << Solver << "." << endl;
                        start = time(0);
                        direct_solver(LHSarray,RHSarray,estimatedsolutions,LHSsize);
                        end = time(0);
                        logfile << "       - Finished Solving Equations created." << endl;
                        logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                   }
                    if(Solver == "pcg")                                                                         /* Solve equations using pcg */
                    {
                        logfile << "           - Starting " << Solver << "." << endl;
                        start = time(0);
                        int* solvedatiteration = new int[1]; solvedatiteration[0] = 0;
                        pcg_solver(LHSarray,RHSarray,estimatedsolutions,LHSsize,solvedatiteration);
                        end = time(0);
                        logfile << "           - PCG converged at iteration " << solvedatiteration[0] << "." << endl;
                        logfile << "       - Finished Solving Equations created." << endl;
                        logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                        delete[] solvedatiteration;
                    }
                    /* Update Animal Class with EBV's and associated Accuracy */
                    for(int i = 0; i < population.size(); i++)
                    {
                        int j = 0;                                                  /* Counter for population spot */
                        while(1)
                        {
                            if(population[i].getID() == animal[j])
                            {
                                population[i].Update_EBV(estimatedsolutions[j+1]);
                                break;
                            }
                            j++;
                        }
                    }
                    delete[] LHSarray; delete[] RHSarray;
                    time_t end_block = time(0);
                    logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
                    animal.clear(); sire.clear(); dam.clear();
                }
                if(scalefactaddh2 == 0)
                {
                    vector < int > animal(TotalAnimalNumber,0);                         /* Array to store Animal IDs */
                    vector < int > sire(TotalAnimalNumber,0);                           /* Array to store Sire IDs */
                    vector < int > dam(TotalAnimalNumber,0);                            /* Array to store Dam IDs */
                    int linenumber = 0;                                                                 /* Counter to determine where at in pedigree index's */
                    string line;
                    ifstream infile2;
                    infile2.open(Pheno_Pedigree_File);                                                  /* This file has all animals in it */
                    if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
                    while (getline(infile2,line))
                    {
                        /* Fill each array with correct number already in order so don't need to order */
                        size_t pos = line.find(" ",0); animal[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);       /* Grab Animal ID */
                        pos = line.find(" ",0); sire[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                /* Grab Sire ID */
                        pos = line.find(" ",0); dam[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                 /* Grab Dam ID */
                        linenumber++;
                    }
                    double *f_qs_u = new double[TotalAnimalNumber];
                    pedigree_inbreeding(Pheno_Pedigree_File,f_qs_u);                /* Function that calculates inbreeding */
                    /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
                    for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
                    {
                        int j = 0;                                                                  /* Counter for population spot */
                        while(1)
                        {
                            if(population[i].getID() == animal[j])
                            {
                                double temp = f_qs_u[j] - 1; population[i].UpdateInb(temp); break;
                            }
                            j++;                                                                    /* Loop across until animal has been found */
                        }
                    }
                    delete[] f_qs_u; animal.clear(); sire.clear(); dam.clear();
                }
                logfile << "   Begin " << Selection << " Selection of offspring: " << endl;
                ///////////////////////////////////////////////////////////////////
                //  Select offspring to serve as parents for next Generation //////
                ///////////////////////////////////////////////////////////////////
                time_t start_block1 = time(0);
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
                        if(Selection == "random"){MaleValue.push_back(population[i].getRndSelection()); male++;}        /* Random Selection */
                        if(Selection == "phenotype"){MaleValue.push_back(population[i].getPhenotype()); male++;}        /* Phenotypic Selection */
                        if(Selection == "true_bv"){MaleValue.push_back(population[i].getGenotypicValue()); male++;}     /* TBV Selection */
                        if(Selection == "ebv"){MaleValue.push_back(population[i].getEBV()); male++;}                    /* EBV Selection */
                    }
                    if(population[i].getSex() == 1 && population[i].getAge() == 1)                                      /* if female (i.e. 1) */
                    {
                        if(Selection == "random"){FemaleValue.push_back(population[i].getRndSelection()); female++;}    /* Random Selection */
                        if(Selection == "phenotype"){FemaleValue.push_back(population[i].getPhenotype()); female++;}    /* Phenotypic Selection */
                        if(Selection == "true_bv"){FemaleValue.push_back(population[i].getGenotypicValue()); female++;} /* TBV Selection */
                        if(Selection == "ebv"){FemaleValue.push_back(population[i].getEBV()); female++;}                /* EBV Selection */
                    }
                }
                /* Array with correct value based on how selection is to proceed created now sort */
                if(SelectionDir == "low" || Selection == "random")                       /* sort lowest to highest */
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
                if(SelectionDir == "high" && Selection != "random")                      /* Sort lowest to highest then reverse order so highest to lowest */
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
                /* Array sorted now determine cutoff value any animal above this line will be removed */
                logfile << "       - Number of Male Selection Candidates: " << male << "." <<  endl;
                logfile << "       - Number of Female Selection Candidates: " << female << "." << endl;
                logfile << "       - Number of Male Parents: " << maleparent << "." << endl;
                logfile << "       - Number of Female Parents: " << femaleparent << "." << endl;
                if(Gen == 1)
                {
                    if(SIRES > male)
                    {
                        logfile << endl << "   Program Ended - Not Enough Male Selection Candidates in GEN 1. Add More Founders!" << endl; exit (EXIT_FAILURE);
                    }
                    if(DAMS > female)
                    {
                        logfile << endl << "   Program Ended - Not Enough Female Selection Candidates in GEN 1. Add More Founders!" << endl; exit (EXIT_FAILURE);
                    }
                }
                time_t start_test = time(0);
                int malesadd = 0; int femalesadd = 0;
                if(Gen > 1)
                {
                    /* add 0.5 due to potential rounding error */
                    int malesneeded = (SIRES * SireReplacement) + ((SIRES * (1 - SireReplacement)) - maleparent) + 0.5;
                    int femalesneeded = (DAMS * DamReplacement) + ((DAMS * (1 - DamReplacement)) - femaleparent) + 0.5;
                    if(male < malesneeded){logfile << "       - Program Ended - Not enough new male progeny. " << endl; exit (EXIT_FAILURE);}
                    if(female < femalesneeded){logfile << "       - Program Ended - Not enough new female progeny. " << endl; exit (EXIT_FAILURE);}
                    if(maleparent < ((SIRES * (1 - SireReplacement))))
                    {
                        malesadd = (SIRES * (1 - SireReplacement)) - maleparent;
                        logfile << "        - Number of male parents too small need to keep more!!" << endl;
                        logfile << "            - Kept " << malesadd << " extra progeny." << endl;
                    }
                    if(femaleparent < int(((DAMS * (1 - DamReplacement))+0.5)))
                    {
                        femalesadd = (DAMS * (1 - DamReplacement)) - femaleparent;
                        logfile << "        - Number of female parents too small need to keep more!!" << endl;
                        logfile << "            - Kept " << femalesadd << " extra progeny." << endl;
                    }
                }
                /* Start out with simulated number of females and males  */
                int malepos, femalepos;
                if(Gen == 1){malepos = SIRES; femalepos = DAMS;}    /* Grabs Position based on percentile in Males or Females */
                /* If under the male and female value need to keep more selection candidates */
                if(Gen > 1){malepos = (SIRES * SireReplacement) + malesadd; femalepos = (DAMS * DamReplacement) + femalesadd;}
                /* vectors to store info on number of full-sib families */
                vector < string > siredamkept(population.size(),""); vector < int > keep(population.size(),0); vector < double > siredamvalue(population.size(),0.0);
                string Action;                               /* Based on Selection used will result in an action that is common to all */
                for(int i = 0; i < population.size(); i++)
                {
                    while(1)
                    {
                        if(Selection == "random")
                        {
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getRndSelection()>MaleValue[malepos-1]){Action="RM_M";break;}
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getRndSelection()<=MaleValue[malepos-1]){Action="KP_M";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getRndSelection()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getRndSelection()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                            if(population[i].getAge() > 1){Action = "Old_Animal"; break;}   /* Old parent so keep can only be culled */
                        }
                        if(Selection == "phenotype" && SelectionDir == "low")
                        {
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()>MaleValue[malepos-1]){Action="RM_M"; break;}
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()<=MaleValue[malepos-1]){Action = "KP_M"; break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                            if(population[i].getAge() > 1){Action = "Old_Animal"; break;}   /* Old parent so keep can only be culled */
                        }
                        if(Selection == "phenotype" && SelectionDir == "high")
                        {
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()<MaleValue[malepos-1]){Action="RM_M"; break;}
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getPhenotype()>=MaleValue[malepos-1]){Action="KP_M";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getPhenotype()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                            if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
                        }
                        if(Selection == "true_bv" && SelectionDir == "low")
                        {
                            if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()>MaleValue[malepos-1]){Action="RM_M"; break;}
                            if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()<=MaleValue[malepos-1]){Action="KP_M";break;}
                            if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                            if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                            if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
                        }
                        if(Selection == "true_bv" && SelectionDir == "high")
                        {
                            if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()<MaleValue[malepos-1]){Action="RM_M"; break;}
                            if(population[i].getSex()==0&&population[i].getAge()==1&&population[i].getGenotypicValue()>=MaleValue[malepos-1]){Action="KP_M";break;}
                            if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()<FemaleValue[femalepos-1]){Action="RM_F";break;}
                            if(population[i].getSex()==1&&population[i].getAge()==1&&population[i].getGenotypicValue()>=FemaleValue[femalepos-1]){Action="KP_F";break;}
                            if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
                        }
                        if(Selection == "ebv" && SelectionDir == "low")
                        {
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getEBV()>MaleValue[malepos-1]){Action="RM_M"; break;}
                            if(population[i].getSex()==0 && population[i].getAge()==1 && population[i].getEBV()<=MaleValue[malepos-1]){Action="KP_M";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getEBV()>FemaleValue[femalepos-1]){Action="RM_F";break;}
                            if(population[i].getSex()==1 && population[i].getAge()==1 && population[i].getEBV()<=FemaleValue[femalepos-1]){Action="KP_F";break;}
                            if(population[i].getAge() > 1){Action = "Old_Animal"; break;}  /* Old parent so keep can only be culled */
                        }
                        if(Selection == "ebv" && SelectionDir == "high")
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
                    if(Action == "KP_M" || Action == "KP_F"){keep[i] = 1;}
                    if(Action == "Old_Animal"){keep[i] = 2;}
                    if(Selection == "random"){siredamvalue[i] = population[i].getRndSelection();}
                    if(Selection == "phenotype"){siredamvalue[i] = population[i].getPhenotype();}
                    if(Selection == "true_bv"){siredamvalue[i] = population[i].getGenotypicValue();}
                    if(Selection == "ebv"){siredamvalue[i] = population[i].getEBV();}
                }
                /* Remove Unselected based on certain selection used Animals from class object and then resize vector from population */
                if(Gen > 1 && OffspringPerMating > 1)
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
                        vector < int > numberwithineachfullsiba (OffspringPerMating,0);
                        for(int i = 0; i < tempsiredam.size(); i++){numberwithineachfullsiba[tempsiredamnum[i]-1] += 1;}
                        int numgreaterthanmax = 0;
                        for(int i = maxmating; i < numberwithineachfullsiba.size(); i++){numgreaterthanmax += numberwithineachfullsiba[i];}
                        /* remove animals with full sib families over the limit */
                        if(numgreaterthanmax > 0)
                        {
                            for(int fam = 0; fam < tempsiredam.size(); fam++)
                            {
                                if(tempsiredamnum[fam] > maxmating)
                                {
                                    vector < double > fullsibvalues;
                                    for(int i = 0; i < siredamkept.size(); i++)
                                    {
                                        if(siredamkept[i] == tempsiredam[fam] && keep[i] == 1)
                                        {
                                            fullsibvalues.push_back(siredamvalue[i]);
                                        }
                                    }
                                    if(SelectionDir == "low" && Selection == "random")          /* sort lowest to highest */
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
                                    if(SelectionDir == "high" && Selection != "random")         /* Sort lowest to highest then reverse order so highest to lowest */
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
                                    for(int i = maxmating; i < fullsibvalues.size(); i++)
                                    {
                                        /* Find Individual */
                                        string found = "NO"; int startpoint = 0;
                                        while(found == "NO")
                                        {
                                            if(Selection == "random")
                                            {
                                                if(fullsibvalues[i] == population[startpoint].getRndSelection() && keep[startpoint] == 1){found = "YES";}
                                                if(fullsibvalues[i] != population[startpoint].getRndSelection()){startpoint++;}
                                            }
                                            if(Selection == "phenotype")
                                            {
                                                if(fullsibvalues[i] == population[startpoint].getPhenotype() && keep[startpoint] == 1){found = "YES";}
                                                if(fullsibvalues[i] != population[startpoint].getPhenotype()){startpoint++;}
                                            }
                                            if(Selection == "true_bv")
                                            {
                                                if(fullsibvalues[i] == population[startpoint].getGenotypicValue() && keep[startpoint] == 1){found = "YES";}
                                                if(fullsibvalues[i] != population[startpoint].getGenotypicValue()){startpoint++;}
                                            }
                                            if(Selection == "ebv")
                                            {
                                                if(fullsibvalues[i] == population[startpoint].getEBV() && keep[startpoint] == 1){found = "YES";}
                                                if(fullsibvalues[i] != population[startpoint].getEBV()){startpoint++;}
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
                                                if(Selection == "random")
                                                {
                                                    if(MaleValue[malepos-1] == population[startpointa].getRndSelection() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
                                                    if(MaleValue[malepos-1] != population[startpointa].getRndSelection()){startpointa++;}
                                                }
                                                if(Selection == "phenotype")
                                                {
                                                    if(MaleValue[malepos-1] == population[startpointa].getPhenotype() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
                                                    if(MaleValue[malepos-1] != population[startpointa].getPhenotype()){startpointa++;}
                                                }
                                                if(Selection == "true_bv")
                                                {
                                                    if(MaleValue[malepos-1] == population[startpointa].getGenotypicValue() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
                                                    if(MaleValue[malepos-1] != population[startpointa].getGenotypicValue()){startpointa++;}
                                                }
                                                if(Selection == "ebv")
                                                {
                                                    if(MaleValue[malepos-1] == population[startpointa].getEBV() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
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
                                                if(Selection == "random")
                                                {
                                                    if(FemaleValue[femalepos-1] == population[startpointa].getRndSelection() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
                                                    if(FemaleValue[femalepos-1] != population[startpointa].getRndSelection()){startpointa++;}
                                                }
                                                if(Selection == "phenotype")
                                                {
                                                    if(FemaleValue[femalepos-1] == population[startpointa].getPhenotype() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
                                                    if(FemaleValue[femalepos-1] != population[startpointa].getPhenotype()){startpointa++;}
                                                }
                                                if(Selection == "true_bv")
                                                {
                                                    if(FemaleValue[femalepos-1] == population[startpointa].getGenotypicValue() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
                                                    if(FemaleValue[femalepos-1] != population[startpointa].getGenotypicValue()){startpointa++;}
                                                }
                                                if(Selection == "ebv")
                                                {
                                                    if(FemaleValue[femalepos-1] == population[startpointa].getEBV() && keep[startpointa] == 0)
                                                    {
                                                        keep[startpointa] = 1; founda = "YES";
                                                    }
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
                            logfile << "       - Number of Full-Sibs selected within a family: " << endl;
                            for(int i = 0; i < numberwithineachfullsiba.size(); i++)
                            {
                                logfile << "            -" << i + 1 << " sibling selected from family: " << numberwithineachfullsiba[i] << "." <<  endl;
                            }
                            conditionmet = "YES";
                        }
                    }
                }
                male = 0; female = 0;
                int ROWS = population.size();                           /* Current Size of Population Class */
                int i = 0;
                /* Save as a continuous string and then output */
                stringstream outputstring(stringstream::out); stringstream outputstringgeno(stringstream::out); int outputnum = 0;
                while(i < ROWS)
                {
                    if(keep[i] == 0 && population[i].getSex()==0)
                    {
                        /* Output info into file with everything in it update with real breeding values at the end */
                        outputstring <<population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" "<< population[i].getSex() << " ";
                        outputstring <<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                        outputstring <<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" ";
                        outputstring <<population[i].getHap2_F() <<" "<< population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                        outputstring <<population[i].getunfavheterolethal()<<" "<<population[i].getunfavhomosublethal()<<" ";
                        outputstring <<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                        outputstring <<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" "<<population[i].getAcc()<<" ";
                        outputstring <<population[i].getGenotypicValue()<<" ";
                        outputstring <<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                        if(OutputGeno == "yes" && Gen >= outputgeneration)
                        {
                            outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                        }
                        population.erase(population.begin()+i); siredamkept.erase(siredamkept.begin()+i);
                        keep.erase(keep.begin()+i); siredamvalue.erase(siredamvalue.begin()+i);
                        ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                    }
                    if(keep[i] == 1 && population[i].getSex()==0){male++; i++;}
                    if(keep[i] == 0 && population[i].getSex()==1)
                    {
                        /* Output info into file with everything in it update with real breeding values at the end */
                        outputstring <<population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" "<< population[i].getSex() << " ";
                        outputstring <<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                        outputstring <<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" "<< population[i].getHap1_F() <<" ";
                        outputstring <<population[i].getHap2_F() <<" "<< population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                        outputstring <<population[i].getunfavheterolethal()<<" "<<population[i].getunfavhomosublethal()<<" ";
                        outputstring <<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                        outputstring <<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" "<<population[i].getAcc()<<" ";
                        outputstring <<population[i].getGenotypicValue()<<" ";
                        outputstring <<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                        if(OutputGeno == "yes" && Gen >= outputgeneration)
                        {
                            outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                        }
                        population.erase(population.begin()+i); siredamkept.erase(siredamkept.begin()+i);
                        keep.erase(keep.begin()+i); siredamvalue.erase(siredamvalue.begin()+i);
                        ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                    }
                    if(keep[i] == 1 && population[i].getSex()==1){female++; i++;}
                    if(keep[i] == 2){i++;}
                    if(outputnum % 1000 == 0)
                    {
                        /* output master df file */
                        std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
                        output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
                        /* output master geno file */
                        std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
                        output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
                    }
                }
                /* output master df file */
                std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
                output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
                /* output master geno file */
                std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
                output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
                /* Calculate number of males and females */
                male = 0; female = 0;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getSex() == 0 && population[i].getAge() == 1){male++;}
                    if(population[i].getSex() == 1 && population[i].getAge() == 1){female++;}
                }
                time_t end_test = time (0);
                logfile << "       - Number Males Selected: " << male << "." <<  endl;
                logfile << "       - Number Females Selected: " << female << "." << endl;
                logfile << "       - Breeding Population Size: " << population.size() << " (Took: " << difftime(end_test,start_test) << ")." << endl;
                vector < int > animal(TotalAnimalNumber,0);                         /* Array to store Animal IDs */
                vector < int > sire(TotalAnimalNumber,0);                           /* Array to store Sire IDs */
                vector < int > dam(TotalAnimalNumber,0);                            /* Array to store Dam IDs */
                int linenumber = 0;                                                                 /* Counter to determine where at in pedigree index's */
                string line;
                ifstream infile2;
                infile2.open(Pheno_Pedigree_File);                                                  /* This file has all animals in it */
                if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
                while (getline(infile2,line))
                {
                    /* Fill each array with correct number already in order so don't need to order */
                    size_t pos = line.find(" ",0); animal[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);       /* Grab Animal ID */
                    pos = line.find(" ",0); sire[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                /* Grab Sire ID */
                    pos = line.find(" ",0); dam[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                 /* Grab Dam ID */
                    linenumber++;
                }
                /* If animal doesn't have a 3 generation pedigree then create it now */
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getPed3G() == "")                  /* hasn't been updated */
                    {
                        int S, D, MGS, MGD, PGS, PGD;
                        S = population[i].getSire();
                        D = population[i].getDam();
                        if(S == 0){PGS = 0; PGD = 0;}
                        if(S != 0)
                        {
                            if(S == animal[(S-1)]){PGS = sire[(S-1)]; PGD = dam[(S-1)];}
                            if(S != animal[(S-1)]){exit (EXIT_FAILURE);}
                        }
                        if(D == 0){MGS = 0; MGD = 0;}
                        if(D != 0)
                        {
                            if(D == animal[(D-1)]){MGS = sire[(D-1)]; MGD = dam[(D-1)];}
                            if(D != animal[(D-1)]){exit (EXIT_FAILURE);}
                        }
                        /* Put into a string */
                        stringstream strStreamPed3G (stringstream::in | stringstream::out);
                        strStreamPed3G << S << "_" << D << "_" << MGS << "_" << MGD << "_" << PGS << "_" << PGD;
                        string ped3g = strStreamPed3G.str();
                        population[i].Update3GenPed(ped3g);
                    }
                }
                animal.clear(); sire.clear(); dam.clear();
                /* All Non-Selected Animals Removed figure out how many sire per progeny based on age */
                /*Figure out Distribution of ages all animals*/
                /* Used so first parity animals get a mating; zero to let Beta control it */
                double proportionParity0 = ((SIRES * SireReplacement) /DAMS);
                int CountAgeClass[15];                              /* Number of animals in total for both male and female */
                int CountSireAgeClass[15];                          /* Number of Sires in a given age class */
                /* Can Skip Age Classes so need to make array that store them based on total so no zero's */
                int CountSireAgeClassTot[15];                       /* Stores count for ages that are > 0 */
                int AgeClassID[15];                                 /* ID for age class; (can skip age classes based on culling */
                double CountSireMateClass[15];
                for(int i = 0; i < 15; i++)                         /* Initialize to zero */
                {
                    CountAgeClass[i] = 0; CountSireAgeClass[i] = 0; CountSireAgeClassTot[i] = 0; AgeClassID[i] = 0; CountSireMateClass[i] = 0.0;
                }
                for(int i = 0; i < population.size(); i++)
                {
                    population[i].ZeroOutMatings();                                 /* Before figuring out number of mating zero out last generations */
                    int temp = population[i].getAge();
                    CountAgeClass[temp - 1] += 1;                                   /* Adds age to current list */
                    if(population[i].getSex() == 0)
                    {
                        int temp = population[i].getAge();
                        CountSireAgeClass[temp - 1] += 1;
                    }
                }
                int AgeClasses = 0;
                for(int i = 0; i < 15; i++)
                {
                    if(CountSireAgeClass[i] > 0)
                    {
                        AgeClassID[AgeClasses] = i + 1;
                        CountSireAgeClassTot[AgeClasses] = CountSireAgeClass[i];
                        AgeClasses += 1;
                    }
                }
                /* Begin to figure out how many gametes to give sire */
                if(proportionParity0 > 0)
                {
                    /* This happens first generation or alpha and beta both equal 1.0 */
                    if(AgeClasses == 1 || (BetaDist_alpha == 1.0 && BetaDist_beta == 1.0))
                    {
                        int temp = DAMS / SIRES;
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getSex() == 0)
                            {
                                population[i].UpdateMatings(temp);
                            }
                        }
                    }
                    /* more than one so need to first give proportion to parity 1 and remaining based of Beta Distribution */
                    if(AgeClasses > 1 && (BetaDist_alpha != 1.0 || BetaDist_beta != 1.0))
                    {
                        /* Proportion of total matings that are available to first parity DAMS * ProportionParity0 grab until set is reached */
                        int i = 0;                                                  /* Counter to determine how many Parity 0 updated */
                        int j = 0;                                                  /* Counter to determine where at in population class */
                        int SireCount = 0;                                          /* Total Number of Sires available */
                        while(i < DAMS * proportionParity0)
                        {
                            if(population[j].getSex() == 0 && population[j].getAge() == 1)  /* First Parity so give it at least one mating */
                            {
                                population[j].UpdateMatings(1);
                                SireCount++;
                                i++;
                            }
                            if(population[j].getSex() == 0 && population[j].getAge() > 1)   /* Older animal so reset to zero and Beta Parameters will determine */
                            {
                                population[j].UpdateMatings(0);
                                SireCount++;
                            }
                            j++;
                        }
                        /* Number of Mating pairs for Beta Distribution */
                        int MatingPairs = DAMS - (DAMS * proportionParity0);
                        double MatingByAge[AgeClasses];
                        for(int i = 0; i < AgeClasses; i++)
                        {
                            if(i == 0){MatingByAge[i] = (1.0 / AgeClasses);}
                            if(i > 0){MatingByAge[i] = MatingByAge[i-1] + (1.0 / AgeClasses);}
                        }
                        /* Figure out proportion that fall in given interval */
                        double MatingProp[AgeClasses];                              /* proportion of gametes that belong to the particular age class */
                        int NumberGametes[AgeClasses];                              /* Number of gamates that are derived from the particular age class */
                        double RunningTotalProp = 0.0;                              /* Used to tally up how far have gone and to subtract of to create interval */
                        int RunningTotalGame = 0;                                   /* Used to keep track of total gametes */
                        for(int i = 0; i < AgeClasses - 1; i++)
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
                        MatingProp[AgeClasses - 1] = 1 - RunningTotalProp;
                        NumberGametes[AgeClasses - 1] = MatingPairs - RunningTotalGame;                 /* Remaining gametes have to end up in last age class */
                        /* Figured out how many gametes fall into each age class now add them to each animal; needs to be randomized because */
                        /* it will never be even so some animals will get an extra gamete                                                    */
                        for(int age = 0; age < AgeClasses; age++)
                        {
                            std::uniform_real_distribution<double> distribution(0,1);
                            int MinGametes = NumberGametes[age] / CountSireAgeClassTot[age];
                            int SurplusGametes = NumberGametes[age] - (MinGametes * CountSireAgeClassTot[age]);    /* up to this number will get 1 extra */
                            int ID[CountSireAgeClassTot[age]];                                                     /* Stores ID */
                            double UniformValue[CountSireAgeClassTot[age]];                                        /* Stores random Variable */
                            int AnimalCounter = 0;
                            for(int an = 0; an < population.size(); an++)
                            {
                                if(population[an].getSex() == 0 && population[an].getAge() ==  AgeClassID[age])    /* Add one because loop starts at 0 */
                                {
                                    ID[AnimalCounter] = population[an].getID();
                                    UniformValue[AnimalCounter] = distribution(gen);
                                    AnimalCounter++;
                                }
                            }
                            /* Sort values based on Uniform value */
                            int temp;
                            double tempa;
                            for(int i = 0; i < CountSireAgeClassTot[age] - 1; i++)
                                for(int j=i+1; j < CountSireAgeClassTot[age]; j++)
                                    if(UniformValue[i] > UniformValue[j])
                                    {
                                        /* put i values in temp variables */
                                        temp = ID[i]; tempa = UniformValue[i];
                                        /* swap lines */
                                        ID[i] = ID[j]; UniformValue[i] = UniformValue[j];
                                        /* put temp values in 1 backward */
                                        ID[j] = temp; UniformValue[j] = tempa;
                                    }
                            /* sort out gametes to animals within a given parity */
                            int j = 0;
                            int ExtraGam = 1;
                            while(j < CountSireAgeClassTot[age])
                            {
                                int i = 0;
                                while(1)
                                {
                                    if(population[i].getSex() == 0 && population[i].getAge() == AgeClassID[age] && population[i].getID() == ID[j])
                                    {
                                        if(SurplusGametes == 0)                                /* will mess up the surplus calculation if equals exactly zero */
                                        {
                                            population[i].UpdateMatings(MinGametes);
                                        }
                                        if(ExtraGam <= SurplusGametes && SurplusGametes != 0)  /* Once reaches total surplus gametes then put minimum*/
                                        {
                                            population[i].UpdateMatings(MinGametes + 1);
                                        }
                                        if(ExtraGam > SurplusGametes && SurplusGametes != 0)
                                        {
                                            population[i].UpdateMatings(MinGametes);
                                        }
                                        ExtraGam++;
                                        j++;
                                        break;
                                    }
                                    i++;
                                }
                            }
                        }
                    }
                }
                /* Loop through females and give them a mating of 1 */
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getSex() == 1)
                    {
                        population[i].ZeroOutMatings();             /* Ensure already set to zero */
                        population[i].UpdateMatings(1);             /* Give a value of 1 mating  */
                    }
                }
                int mincount = 0;
                int maxcount = 0;
                int mincountid = 0;
                int maxcountid = 0;
                for(int i = 0; i < population.size(); i++)
                {
                    int temp = population[i].getAge();
                    if(population[i].getSex() == 0)
                    {
                        int tempa = population[i].getMatings();                                 /* grab current number of matings */
                        CountSireMateClass[temp - 1] = CountSireMateClass[temp - 1 ] + tempa;    /* add number based on age */
                    }
                }
                logfile << "       - Sire Breeding Age Distribution: " << endl;
                for(int i = 0; i < 15; i++)
                {
                    if(CountSireMateClass[i] > 0)
                    {
                        logfile << "           - Age " << i + 1 << " Number Sires: " << CountSireAgeClass[i] << " and Number of Matings: ";
                        logfile << CountSireMateClass[i] << endl;
                    }
                }
                int CheckMatings = 0;
                int CheckSire = 0;
                int CheckDam = 0;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getSex() == 0)
                    {
                        CheckSire += population[i].getMatings();
                    }
                    if(population[i].getSex() == 1)
                    {
                        CheckDam += population[i].getMatings();
                    }
                    CheckMatings += population[i].getMatings();
                }
                logfile << "       - Total Matings: " << CheckMatings << "; Sire Matings: " << CheckSire << "; Dam Matings: " << CheckDam << endl;
                time_t end_block1 = time(0);
                logfile << "   Finished " << Selection << " Selection of parents (Time: " <<difftime(end_block1,start_block1)<< " seconds)." << endl << endl;
                ////////////////////////////////////////////////////////////////
                //  Create Mutation Events and Gametes from parent genotypes  //
                ////////////////////////////////////////////////////////////////
                // Part1: Set up housekeeping arrays.
                // Part2: Combine QTL and Marker genotypes and put into homologues sire dam gametes
                // Part3: Determine number of mutations and where they occur
                // Part4: Sample the recombination parameters
                // Part5: Find the SNP right after a Cx
                // Part6: Create the new haplotype by copying from the parental homologes
                // Part7: Update all genotypes with new mutations that created a new QTL
                // Part8: Reconstruct Index and Update Map
                // Part9: Update genotype with mutations
                logfile << "   Begin creating gametes based on parent genotypes and mutation events: " << endl;
                time_t start_block2 = time(0);
                /////////////////////////////////////////////
                // Step 1: Set up housekeeping arrays.     //
                /////////////////////////////////////////////
                /* Set total number of markers and qtl based on length of marker and QTL string in parent generation */
                int TotalQTL = 0;
                int TotalMarker = 0;
                for(int i = 0; i < numChr; i++)
                {
                    TotalQTL += qtlperChr[i];
                    TotalMarker += markerperChr[i];
                }
                logfile << "       - Gamete size prior to new mutations: " << TotalMarker + TotalQTL << "." << endl;
                /* merge marker and QTL back together to create map file */
                vector < double > FullMap;
                for(int i = 0; i < (TotalQTL + TotalMarker); i++){FullMap.push_back(0.0);}  /* Array to store map position across chromosomes */
                int m_counter = 0;                                                  /* where at in marker array */
                int qtl_counter = 0;                                                /* where at in quantitative array */
                /* Combine to make marker and QTL map */
                for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                {
                    if(m_counter < MarkerIndex.size())                              /* ensures doesn't go over and cause valgrind error */
                    {
                        if(MarkerIndex[m_counter] == i){FullMap[i] = MarkerMapPosition[m_counter]; m_counter++;}  /* If is marker put in full map */
                    }
                    if(qtl_counter < QTL_Index.size())                           /* ensures doesn't go over and cause valgrind error */
                    {
                        if(QTL_Index[qtl_counter] == i){FullMap[i] = QTL_MapPosition[qtl_counter]; qtl_counter++;}  /* If is quant QTL put in full map */
                    }
                }
                /* Put Sequence SNP map information in 2-dim array so only have to read in once; row = numchr col= max number of SNP */
                int col = ChrSNPLength[0];                                                          /* intitialize to figure out max column number */
                for(int i = 1; i < numChr; i++){if(ChrSNPLength[i] > col){col = ChrSNPLength[i];}}   /* Find max number of SNP within a chromosome */
                // store in 2-D vector using this may get large so need to store dynamically
                vector < vector < double > > SNPSeqPos;
                /* read in each chromsome files */
                for(int i = 0; i < numChr; i++)
                {
                    vector < double > temp;
                    string mapfilepath = path + "/" + outputfolder + "/" + MapFiles[i];
                    ifstream infile8;
                    infile8.open(mapfilepath);
                    if (infile8.fail()){cout << "Error Opening MaCS Map File\n"; exit (EXIT_FAILURE);}
                    for(int j = 0; j < ChrSNPLength[i]; j++)
                    {
                        double tempa =0.0;
                        infile8 >> tempa;
                        temp.push_back(tempa);
                    }
                    infile8.close();
                    SNPSeqPos.push_back(temp);
                }
                //////////////////////////////////////////////////
                // Step 2: Combine QTL and Marker genotypes     //
                //////////////////////////////////////////////////
                /* Variables to Store Mutation Events; Collect all of them and then after all gametes have been created add them */
                vector < int > MutationAnim;                                    /* Which animal mutation originated from */
                vector < int > MutationGamete;                                  /* Which gamete it came from */
                vector < int > MutationType;                                    /* Type of mutation quantitative, lethal or sublethal */
                vector < double > MutationLoc;                                  /* Location of Mutation */
                vector < int > MutationChr;                                     /* Chromosome of Mutation */
                vector < double > MutationAdd_quan;                             /* Additive effect of allele quantitative */
                vector < double > MutationDom_quan;                             /* Dominance effect of mutant allele quantitative */
                vector < double > MutationAdd_fit;                              /* Additive effect of allele fitness */
                vector < double > MutationDom_fit;                              /* Dominance effect of mutant allele fitness */
                int CounterMutationIndex = 0;
                /* Variables to Store Gametes that were created; Collect them all then alter them based on mutations that occured */
                vector < int > AnimGam_ID;                                      /* Parent ID of Gamete */
                vector < double > AnimGam_Dev;                                  /* Parent Deviate used to sort for random mating */
                vector < int > AnimGam_GamID;                                   /* Gamete number */
                vector < int > AnimGam_Sex;                                     /* Sex of Parent Gamete */
                vector < string > AnimGam_Gam;                                  /* Gamete without mutations */
                int CounterAnimGamIndex = 0;
                /* Determine number of gametes to produce for each male and female individual */
                int gametesfemale = OffspringPerMating;                         /* female can only be mated to one sire */
                logfile << "       - Number of offspring per mating pair: " << OffspringPerMating << "." << endl;
                /* Array to update old mutations allele frequency */
                vector < double > SNPFreqGen;
                for(int i = 0; i < (TotalQTL + TotalMarker); i++){SNPFreqGen.push_back(0);}     /* declare array to calculate QTL allele frequencies in parents */
                /* while making gametes and seeing if a mutation event happened, drop ordered genotype int vector string to add in new mutations */
                vector <string> parentgeno;
                int countfemalemates = 0; int countmalemates = 0;
                /* Loop across individuals */
                for(int a = 0; a < population.size(); a++)
                {
                    /* Generate random deviation from uniform distribution to use for random mating */
                    uniform_real_distribution<double> distribution(0,1);
                    int matings = population[a].getMatings();                                       /* Number of mating; number of deviates to draw */
                    // Grab marker and QTL genotypes from population class
                    int TEMPID = population[a].getID();                                             /* Grabs Animal ID */
                    int TEMPSEX = population[a].getSex();                                           /* Grabs Sex of Animal */
                    string mark = population[a].getMarker();                                        /* Grabs marker genotypes */
                    string qtl_qn = population[a].getQTL();                                         /* Grabs QTL genotypes */
                    // Declare Variables
                    vector < int > markgeno(m_counter,0);                                           /* Stores marker genotypes in array */
                    vector < int > qtl_geno(qtl_counter,0);                                         /* Stores quantitative qtl genotypes in array */
                    vector < int > geno((m_counter + qtl_counter),0);                               /* Stores full genotype in correct order by location */
                    vector < int > fullhomo1((m_counter + qtl_counter),0);                          /* Stores FULL array for homologue 1 */
                    vector < int > fullhomo2((m_counter + qtl_counter),0);                          /* Stores FULL array for homologue 2 */
                    /* put both marker and QTL genotypes into an array instead of a string; ASCI value is 48 for 0; ASCI value for 0 - 9 is 48 - 57 */
                    for(int i = 0; i < m_counter; i++){markgeno[i] = mark[i] - 48;}                 /* Marker genotypes */
                    for(int i = 0; i < qtl_counter; i++){qtl_geno[i] = qtl_qn[i] - 48;}          /* QTL genotypes */
                    m_counter = 0;                                                                  /* where at in marker array */
                    qtl_counter = 0;                                                             /* where at in quantitative qtl array */
                    /* Combine them into marker and QTL array for crossovers based on index value */
                    for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                    {
                        if(m_counter < MarkerIndex.size())                              /* ensures doesn't go over and cause valgrind error */
                        {
                            if(MarkerIndex[m_counter] == i){geno[i] = markgeno[m_counter]; m_counter++;}            /* If is marker put in geno */
                        }
                        if(qtl_counter < QTL_Index.size())                           /* ensures doesn't go over and cause valgrind error */
                        {
                            if(QTL_Index[qtl_counter] == i){geno[i] = qtl_geno[qtl_counter]; qtl_counter++;}        /* If is QTL put in geno */
                        }
                    }
                    /* but back into string to make storing easier then once gamete formation complete add back in mutations */
                    string parentGENO;
                    stringstream strStream (stringstream::in | stringstream::out);
                    for (int i=0; i < (TotalQTL + TotalMarker); i++)
                    {
                        strStream << geno[i];
                    }
                    parentGENO = strStream.str();
                    parentgeno.push_back(parentGENO);
                    for (int i=0; i < (TotalQTL + TotalMarker); i++)
                    {
                        if(geno[i] == 3 || geno[i] == 4){SNPFreqGen[i] += 1;}
                        if(geno[i] == 2){SNPFreqGen[i] += 2;}
                    }
                    /* put genotypes into haplotypes; fullhomo1 = Sire & fullhomo2 = Dam; */
                    for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                    {
                        if(geno[i] == 0){fullhomo1[i] = 1; fullhomo2[i] = 1;}                   /* genotype a1a1 */
                        if(geno[i] == 2){fullhomo1[i] = 2; fullhomo2[i] = 2;}                   /* genotype a2a2 */
                        if(geno[i] == 3){fullhomo1[i] = 1; fullhomo2[i] = 2;}                   /* genotype a1a2 */
                        if(geno[i] == 4){fullhomo1[i] = 2; fullhomo2[i] = 1;}                   /* genotype a2a1 */
                    }
                    /* Depending on whether male or female need to loop across to create a given number of gametes for each individual */
                    int g = 0;                                                                  /* Counter keep track of number of gamates */
                    /* Loop across number of matings */
                    for(int mat = 0; mat < matings; mat++)
                    {
                        double temp_dev = distribution(gen);                                    /* Each mating pair combo will get same Uniform Deviate */
                        if(TEMPSEX == 1){countfemalemates++;}
                        if(TEMPSEX == 0){countmalemates++;}
                        /* Loop across number of offspring */
                        for(int off = 0; off < OffspringPerMating; off++)
                        {
                            AnimGam_Dev.push_back(temp_dev);                        /* Store Uniform Deviate */
                            population[a].UpdateProgeny();                          /* Adds one to to progeny number for animal */
                            int genocounter = 0;                                    /* Counter for where to start next chromosome in full geno file */
                            vector < int > fullgamete((TotalQTL + TotalMarker),0);
                            AnimGam_ID.push_back(TEMPID);                           /* Store Animal ID */
                            AnimGam_GamID.push_back(g + 1);                         /* Store gamete number */
                            AnimGam_Sex.push_back(TEMPSEX);                         /* Store Sex of Animal */
                            /** Loop across chromosomes: Simulate mutations then generate gamete for each chromosome **/
                            for(int c = 0; c < numChr; c++)
                            {
                                //////////////////////////////////
                                /// Part 3: Simulate Mutations ///
                                //////////////////////////////////
                                /* Simulate Mutations based on an infinite sites model, which assumes no site that is segregating in the        */
                                /* population can receive another mutation. The number of mutations is sampled from a poisson distribution      */
                                /* that has a rate parameter equal to the length (nucleotides) times the mutation rate. The proportion of       */
                                /* mutatation that can be a QTL is determined by the user. Only QTL were kept track of and they were placed     */
                                /* in the QTL pool. Each generation the QTL pool gets reconfigured to remove alleles that were fixed and index  */
                                /* for QTL and markers get updated. For each gamete that gets created store mutation variables in seperate      */
                                /* array then after all gametes are created they get updated by iterating through number of mutations. Do not   */
                                /* need to be included during crossover event and gamete creation because it is seen in the gamete.             */
                                int Nmb_Mutations;                          /* number of new mutations for each individual with a chromosome */
                                /* Sample number of Mutation Events */
                                std::poisson_distribution<int>distribution(u*ChrLength[c]);
                                Nmb_Mutations = distribution(gen);
                                /* Proportion that can be QTL */
                                Nmb_Mutations = (double(Nmb_Mutations) * double(PropQTL));        /* integer */
                                /* If number of mutations > 1 create appropriate samples and place in storage */
                                if(Nmb_Mutations > 0)
                                {
                                    /* sample location based on uniform distribution from 0 to 1 (length of chromsome) */
                                    for(int i = 0; i < Nmb_Mutations; i++)
                                    {
                                        MutationAnim.push_back(population[a].getID());
                                        MutationGamete.push_back(g + 1);
                                        std::uniform_real_distribution<double> distribution1(0,1.0);
                                        /* Determine what is the mutation 2 = quantitative; 4 = lethal; 5 = sublethal */
                                        double temptype = distribution1(gen);
                                        if(numQTL > 0 && numFitnessQTLlethal > 0 && numFitnessQTLsublethal > 0)
                                        {
                                            if(temptype < 0.333){MutationType.push_back(2);}                                    /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.333 && temptype < 0.666){MutationType.push_back(4);}               /* Mutation is a Lethal Fitness QTL */
                                            if(temptype >= 0.666){MutationType.push_back(5);}                                   /* Mutation is a subLethal Fitness QTL */
                                        }
                                        if(numQTL > 0 && numFitnessQTLlethal == 0 && numFitnessQTLsublethal > 0)
                                        {
                                            if(temptype < 0.5){MutationType.push_back(2);}                                    /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.5){MutationType.push_back(5);}                                   /* Mutation is a subLethal Fitness QTL */
                                        }
                                        if(numQTL > 0 && numFitnessQTLlethal > 0 && numFitnessQTLsublethal == 0)
                                        {
                                            if(temptype < 0.5){MutationType.push_back(2);}                                    /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.5){MutationType.push_back(4);}                                   /* Mutation is a Lethal Fitness QTL */
                                        }
                                        if(numQTL > 0 && numFitnessQTLlethal == 0 && numFitnessQTLsublethal == 0){MutationType.push_back(2);}
                                        MutationLoc.push_back(distribution1(gen) + c + 1);
                                        for(int j = 0; j < ChrSNPLength[c]; j++)
                                        {
                                            if(MutationLoc[CounterMutationIndex] == SNPSeqPos[c][j] + c + 1)  /* Make sure it hasn't already been tagged as SNP */
                                            {
                                                logfile << "       - Mutated SNP already is Nuetral, Marker or QTL added to small number" << endl;
                                                double temp = MutationLoc[CounterMutationIndex] + 0.000001;
                                                while(1)
                                                {
                                                    if(temp != (SNPSeqPos[c][j] + c + 1)){MutationLoc[CounterMutationIndex] = temp; break;}
                                                    if(temp == (SNPSeqPos[c][j] + c + 1)){temp = temp + 0.0000000001;}
                                                }
                                            }
                                        }
                                        MutationChr.push_back(c + 1);
                                        /* Depending on the type */
                                        if(MutationType[CounterMutationIndex] == 2)
                                        {
                                            /******* QTL Additive Effect *******/
                                            std::gamma_distribution <double> distribution1(Gamma_Shape,Gamma_Scale);            /* QTL generated from a gamma */
                                            MutationAdd_quan.push_back(distribution1(gen));
                                            /****** QTL Dominance Effect *******/
                                            /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                                            std::normal_distribution<double> distribution2(Normal_meanRelDom,Normal_varRelDom);
                                            double temph = distribution2(gen);
                                            MutationDom_quan.push_back(MutationAdd_quan[CounterMutationIndex] * temph);
                                            /* Determine sign of additive effect */
                                            /* 1 tells you range so can sample from 0 to 1 and 0.5 is the frequency */
                                            std::binomial_distribution<int> distribution4(1,0.5);
                                            int signadd = distribution4(gen);
                                            if(signadd == 1){MutationAdd_quan[CounterMutationIndex] = MutationAdd_quan[CounterMutationIndex] * -1;}
                                            /* Scale effects based on Founder Scale Factor */
                                            /* Should not re-scale it because the effects of the current mutations should not change */
                                            /* Scaling based on the original set of effects should put it close to the effect sizes of the founder QTL */
                                            MutationAdd_quan[CounterMutationIndex] = MutationAdd_quan[CounterMutationIndex] * scalefactaddh2;
                                            MutationDom_quan[CounterMutationIndex] = MutationDom_quan[CounterMutationIndex] * scalefactpdomh2;
                                            MutationAdd_fit.push_back(0.0); MutationDom_fit.push_back(0.0);
                                        }
                                        if(MutationType[CounterMutationIndex] == 4)
                                        {
                                            /*******     QTL s effect (i.e. selection coeffecient)   *******/
                                            std::gamma_distribution <double> distribution1(Gamma_Shape_Lethal,Gamma_Scale_Lethal);
                                            MutationAdd_fit.push_back(distribution1(gen));
                                            /******      QTL h Effect (i.e. degree of dominance)     *******/
                                            /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                                            std::normal_distribution<double> distribution2(Normal_meanRelDom_Lethal,Normal_varRelDom_Lethal);
                                            double temph = distribution2(gen);
                                            MutationDom_fit.push_back(abs(temph));
                                            MutationAdd_quan.push_back(0.0); MutationDom_quan.push_back(0.0);
                                        }
                                        if(MutationType[CounterMutationIndex] == 5)
                                        {
                                            /*******     QTL s effect (i.e. selection coeffecient)   *******/
                                            std::gamma_distribution <double> distribution1(Gamma_Shape_SubLethal,Gamma_Scale_SubLethal);
                                            MutationAdd_fit.push_back(distribution1(gen));
                                            /******      QTL h Effect (i.e. degree of dominance)     *******/
                                            /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                                            std::normal_distribution<double> distribution2(Normal_meanRelDom_SubLethal,Normal_varRelDom_SubLethal);
                                            double temph = distribution2(gen);
                                            MutationDom_fit.push_back(abs(temph));
                                            MutationAdd_quan.push_back(0.0); MutationDom_quan.push_back(0.0);
                                        }
                                        CounterMutationIndex++;
                                    }
                                }
                                ///////////////////////////////////////////////////////////////////////
                                /// Create gamete based on markers and QTL (not including mutation) ///
                                ///////////////////////////////////////////////////////////////////////
                                int homo1[(markerperChr[c] + qtlperChr[c])];
                                int homo2[(markerperChr[c] + qtlperChr[c])];
                                double mappos[(markerperChr[c] + qtlperChr[c])];
                                int newhaplotype[(markerperChr[c] + qtlperChr[c])];
                                /* Copy current chromosome haplotypes int temp homo1 and homo2 and their associated map position */
                                int spot = 0;                                       /* indicator to determine where at in temporary arrays */
                                for(int i = genocounter; i < (genocounter + (markerperChr[c] + qtlperChr[c])); i++)
                                {
                                    homo1[spot] = fullhomo1[i];
                                    homo2[spot] = fullhomo2[i];
                                    mappos[spot] = FullMap[i] - (c+1);
                                    spot++;
                                }
                                // initiate some variables within each chromosome
                                double u;                       /* random uniform[0,1] derivate  */
                                int pointingAt;                 /* variable indicating which homologe is pointed at */
                                int *currentHomologe_ptr;       /* pointer to the current homologe */
                                int countSNP;                   /* while loop counter for SNP */
                                int nCx;                        /* number of Cx */
                                ////////////////////////////////////////////////////////////
                                /// Part 4: Sample Recombination Parameters and Location ///
                                ////////////////////////////////////////////////////////////
                                /* mean of 1.0 crossovers simulated from Poisson distribution and located with uniform distribution across the chromosome. */
                                std::poisson_distribution<int>distribution6(1.3);
                                nCx = distribution6(gen);
                                /* sample the locations of the Cx (and sort them) */
                                int position_dummy;
                                if(nCx > 0)
                                {
                                    position_dummy = 0;
                                }
                                else
                                {
                                    position_dummy = 1;
                                }
                                /* vector of Cx positions (positition_dummy=1 is added so that the array has length > 0 even if nCx == 0 */
                                /* (the element in the last position is ignored) */
                                double lCx[nCx + position_dummy];   /* stores recombination events */
                                /* Only if recombination occured */
                                if(nCx > 0)
                                {
                                    if(Recomb_Distribution == "Uniform")
                                    {
                                        /* For each recombination event sample to get location */
                                        for(int countCx = 0; countCx < nCx; ++countCx)
                                        {
                                            std::uniform_real_distribution<double> distribution7(0,1.0);
                                            lCx[countCx] = distribution7(gen);
                                        }
                                    }
                                    if(Recomb_Distribution == "Beta")
                                    {
                                        /* For each recombination event sample to get location */
                                        for(int countCx = 0; countCx < nCx; ++countCx)
                                        {
                                            sftrabbit::beta_distribution<> beta(0.5,0.5);     /* Beta Distribution */
                                            lCx[countCx] = beta(gen);
                                        }
                                    }
                                }
                                /* Sort if nCx is greater than 2 using bubble sort*/
                                if (nCx > 1)
                                {
                                    double temp;
                                    for(int i = 0; i < nCx-1; i++)
                                        for(int j=i+1; j< nCx; j++)
                                            if(lCx[i] > lCx[j])
                                            {
                                                // These three lines swap the elements list[i] and list[j].
                                                temp = lCx[i];
                                                lCx[i] = lCx[j];
                                                lCx[j] = temp;
                                            }
                                }
                                /* Sample from discrete uniform to decide which homologue to chose to use to track recombination */
                                /* Sample to determine which homologue to start */
                                std::uniform_real_distribution<double> distribution8(0,1);
                                u = distribution8(gen);
                                /*  if u < 0.5 point to homologe1 else to homologe2 */
                                if(u < 0.5){currentHomologe_ptr = homo1; pointingAt = 1;}
                                if(u >= 0.5){currentHomologe_ptr = homo2; pointingAt = 2;}
                                ////////////////////////////////////////////////////////////
                                /// Part 5: find the SNP right after a Cx                ///
                                ////////////////////////////////////////////////////////////
                                /* Only if recombination occured */
                                if(nCx > 0)
                                {
                                    int SNPaftCx[nCx];              /* stores the index of the SNP right after the Cx */
                                    double locNextCx;               /* the location of the next Cx */
                                    locNextCx = lCx[0];
                                    /* number of the Cx ahead (at start, this is 1, because we are at the beginning of */
                                    /* the chromosome, if Cx 1 is passed  countCx will change to 2, because Cx number 2 is now ahead) */
                                    int numNextCx = 1;
                                    /* loop over the whole chromosome, forever or until a break is  encountered */
                                    countSNP = 0;
                                    while(1)
                                    {
                                        /* if the ID of the Cx ahead is larger than nCx, break, because the last Cx was already passed */
                                        if(numNextCx > nCx)
                                        {
                                            break;
                                        }
                                        /* If the number of Loci is equal to countSNP break then recombination happend at last one can't be observed */
                                        if(countSNP > ((markerperChr[c] + qtlperChr[c]) - 1))
                                        {
                                            *(SNPaftCx + numNextCx-1) = countSNP;
                                            {
                                                break;
                                            }
                                        }
                                        /* if the map position of the current SNP is behind the  location of the next Cx  pointer to mapPos index */
                                        if(*(mappos + countSNP) > locNextCx)
                                        {
                                            /* store the position of this SNP */
                                            *(SNPaftCx + numNextCx -1) = countSNP;
                                            /* change the ID of the next Cx that is ahead */
                                            ++numNextCx;
                                            /* break if we are beyond the last Cx*/
                                            if(numNextCx > nCx)
                                            {
                                                break;
                                            }
                                            /* Change the location of next SNP */
                                            locNextCx = lCx[numNextCx -1];
                                            /* if the mapPos of the SNP is also larger than the next Cx  location then decrement countSNP. Outside this */
                                            /* if block countSNP will incremented again, so that countSNP stays the same, until the mapPos of countSNP */
                                            /* is not larger than the next Cx anymore. This assures that multiple Cx between two SNP are detected. */
                                            if(countSNP <= ((markerperChr[c] + qtlperChr[c]) - 1))
                                            {
                                                if(*(mappos + countSNP) > locNextCx)
                                                    --countSNP;
                                            }
                                        }
                                        ++countSNP;             /* go to next SNP */
                                    }
                                    //////////////////////////////////////////////////////////////////////////////////////////////
                                    /// Part 6: create the new haplotype by copying from the parental homologes                ///
                                    //////////////////////////////////////////////////////////////////////////////////////////////
                                    numNextCx = 1;
                                    countSNP = 0;
                                    while(true)
                                    {
                                        /* copy all SNP between the last and the next Cx from the currentHomologe if the last Cx was passed copy until */
                                        /* the end of the chromsome OR if the SNP behind the next Cx is a SNP that is out of the range of existing SNP */
                                        /* then ignore the Cx and copy until the end of the chromosome */
                                        if((numNextCx > nCx) || (*(SNPaftCx + numNextCx - 1) > (markerperChr[c] + qtlperChr[c])))
                                        {
                                            while(countSNP <= ((markerperChr[c] + qtlperChr[c]) - 1))
                                            {
                                                *(newhaplotype + countSNP) = *(currentHomologe_ptr + countSNP);
                                                ++countSNP;
                                            }
                                            break;
                                        }
                                        else
                                        /* else  copy until the next Cx; */
                                        {
                                            while(countSNP <  *(SNPaftCx + numNextCx - 1))
                                            {
                                                *(newhaplotype + countSNP) = *(currentHomologe_ptr + countSNP);
                                                ++countSNP;
                                            }
                                        }
                                        /* switch pointer to other chromsome */
                                        if(pointingAt == 1)
                                        {
                                            currentHomologe_ptr = homo2;
                                            pointingAt = 2;
                                        }
                                        else
                                        {
                                            currentHomologe_ptr = homo1;
                                            pointingAt = 1;
                                        }
                                        ++numNextCx;
                                    }
                                } /* End of if number of recombination was greater than 0 */
                                if(nCx == 0) /* if no cross-over then just copy full homologue that is being pointed too */
                                {
                                    int countSNP;
                                    for(countSNP = 0; countSNP < (markerperChr[c] + qtlperChr[c]); ++countSNP)
                                    {
                                        *(newhaplotype + countSNP) = *(currentHomologe_ptr + countSNP);
                                    }
                                }
                                /* Done with current chromosome copy gamete onto full gamete */
                                int j = 0;                                                      /* j is a counter for location of gamete */
                                /* Copy to full new gamete */
                                for(int i = genocounter; i < (genocounter + (markerperChr[c] + qtlperChr[c])); i++)
                                {
                                    fullgamete[i] = newhaplotype[j];
                                    j++;
                                } /* close for loop */
                                genocounter = genocounter + markerperChr[c] + qtlperChr[c];     /* Update position of where you are at in full gamete */
                            }
                            /* put gamete into string to store */
                            string gamete;
                            stringstream strStream (stringstream::in | stringstream::out);
                            for (int i=0; i < genocounter; i++)
                            {
                                strStream << fullgamete[i];
                            }
                            AnimGam_Gam.push_back(strStream.str());
                            CounterAnimGamIndex++;
                            g++;
                        }               /* looped through all mating for giving mating pair */
                    }                   /* if mated more then once keep on looping */
                }
                logfile << "       - Female's to Mate: " << countfemalemates << "; Male's to Mate: " << countmalemates << "." << endl;
                logfile << "       - Total number of gametes produced: " << CounterAnimGamIndex << "." << endl;
                logfile << "       - Finished Creating Gametes, Total Number of New Mutations: " << CounterMutationIndex << "." << endl;
                /* delete SNPSeqPos 2-D vector */
                for(int i = 0; i < numChr; i++){SNPSeqPos[i].clear();}
                SNPSeqPos.clear();
                if(CounterMutationIndex == 0){logfile << "       - No new mutations don't need to update QTL object and map files." << endl;}
                if(CounterMutationIndex > 0)
                {
                    for(int i = 0; i < CounterMutationIndex; i++)
                    {
                        string temp;
                        if(MutationType[i] == 2){temp = "2";}
                        if(MutationType[i] == 4){temp = "4";}
                        if(MutationType[i] == 5){temp = "5";}
                        string stringfreq;
                        /* Add a 0.0 for every generation before the current one */
                        for(int g = 0; g < Gen; g++)
                        {
                            if(g != (Gen - 1)){stringfreq = stringfreq + "0.0" + "_";}
                            if(g == (Gen - 1)){stringfreq = stringfreq + "0.0";}
                        }
                        if(MutationType[i] == 2)
                        {
                            QTL_new_old tempa(MutationLoc[i], MutationAdd_quan[i], MutationDom_quan[i], temp, Gen, stringfreq);
                            population_QTL.push_back(tempa);
                        }
                        if(MutationType[i] == 4 || MutationType[i] == 5)
                        {
                            QTL_new_old tempa(MutationLoc[i], MutationAdd_fit[i], MutationDom_fit[i], temp, Gen, stringfreq);
                            population_QTL.push_back(tempa);
                        }
                    }
                    logfile << "       - New QTL's Added to QTL class object (Total: " << population_QTL.size() << ")." << endl;
                    /* Update number of qtl and markers per chromosome */
                    for(int i = 0; i < numChr; i++)
                    {
                        int numbqtl = 0;
                        vector < double > allqtllocations;
                        for(int j = 0; j < population_QTL.size(); j++){allqtllocations.push_back(population_QTL[j].getLocation());}
                        /* Remove duplicates */
                        allqtllocations.erase(unique(allqtllocations.begin(),allqtllocations.end()),allqtllocations.end());
                        /* Location is in double (1.2345) and when convert to integer will always round down so can get Chr */
                        for(int j = 0; j < allqtllocations.size(); j++)
                        {
                            int temp = allqtllocations[j]; if(temp == i + 1){numbqtl += 1;}
                        }
                        qtlperChr[i] = numbqtl;
                    }
                    /////////////////////////////////////////////////////////
                    /// Part8: Reconstruct Index and Update Map           ///
                    /////////////////////////////////////////////////////////
                    /* Reconstruct Old QTL Marker indicator array */
                    /* Old QTL or Marker Indicator (1 = marker,2 = quant QTL,3 = quant + fitness QTL, 4 = lethal fitness QTL, 5 = sublethal fitness QTL) */
                    vector < int > Old_QTLMarker((TotalQTL + TotalMarker),0);
                    vector < double > Old_Add_Quan(TotalQTL,0.0);                                   /* Old additive effect quantitative */
                    vector < double > Old_Dom_Quan(TotalQTL,0.0);                                   /* Old dominance effect quantitative */
                    vector < double > Old_Add_Fit(TotalQTL,0.0);                                    /* Old additive effect fitness */
                    vector < double > Old_Dom_Fit(TotalQTL,0.0);                                    /* Old additive effect fitness */
                    vector < int > Old_QTL_Allele(TotalQTL,0);                                      /* Old unfavorable allele for fitness */
                    m_counter = 0; qtl_counter = 0;
                    for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                    {
                        if(MarkerIndex[m_counter] == i){Old_QTLMarker[i] = 1; m_counter++;}                             /* was a marker tagged as 1 */
                        if(QTL_Index[qtl_counter] == i)
                        {
                            Old_QTLMarker[i] = QTL_Type[qtl_counter];
                            Old_Add_Quan[qtl_counter] = QTL_Add_Quan[qtl_counter];
                            Old_Dom_Quan[qtl_counter] = QTL_Dom_Quan[qtl_counter];
                            Old_Add_Fit[qtl_counter] = QTL_Add_Fit[qtl_counter];
                            Old_Dom_Fit[qtl_counter] = QTL_Dom_Fit[qtl_counter];
                            Old_QTL_Allele[qtl_counter] = QTL_Allele[qtl_counter];
                            qtl_counter++;
                        }
                    }
                    /* Need to order mutation by location in genome */
                    int temp_mutAnim; int temp_mutGame; int temp_mutType; double temp_mutLoc; int temp_mutChr;
                    double temp_mutAdd_quan; double temp_mutDom_quan; double temp_mutAdd_fit; double temp_mutDom_fit;
                    for(int i = 0; i < CounterMutationIndex-1; i++)
                    {
                        for(int j=i+1; j < CounterMutationIndex; j++)
                        {
                            if(MutationLoc[i] > MutationLoc[j])
                            {
                                /* put i values in temp variables */
                                temp_mutAnim=MutationAnim[i]; temp_mutGame=MutationGamete[i]; temp_mutType=MutationType[i]; temp_mutLoc=MutationLoc[i];
                                temp_mutChr=MutationChr[i]; temp_mutAdd_quan=MutationAdd_quan[i]; temp_mutDom_quan=MutationDom_quan[i];
                                temp_mutAdd_fit=MutationAdd_fit[i]; temp_mutDom_fit=MutationDom_fit[i];
                                /* swap lines */
                                MutationAnim[i]=MutationAnim[j]; MutationGamete[i]=MutationGamete[j]; MutationType[i]=MutationType[j];
                                MutationLoc[i]=MutationLoc[j]; MutationChr[i]=MutationChr[j]; MutationAdd_quan[i]=MutationAdd_quan[j];
                                MutationDom_quan[i]=MutationDom_quan[j]; MutationAdd_fit[i]=MutationAdd_fit[j]; MutationDom_fit[i]=MutationDom_fit[j];
                                /* put temp values in 1 backward */
                                MutationAnim[j]=temp_mutAnim; MutationGamete[j]=temp_mutGame; MutationType[j]=temp_mutType; MutationLoc[j]=temp_mutLoc;
                                MutationChr[j]=temp_mutChr; MutationAdd_quan[j]=temp_mutAdd_quan; MutationDom_quan[j]=temp_mutDom_quan;
                                MutationAdd_fit[j]=temp_mutAdd_fit; MutationDom_fit[j]=temp_mutDom_fit;
                            }
                        }
                    }
                    /* Zero out old QTL arrays and Marker Index to ensure something doesn't get messed up */
                    for(int i = 0; i < m_counter; i++){MarkerIndex[i] = 0; MarkerMapPosition[i] = 0.0;}
                    /* Zero out all QTL arrays to ensure doesn't mess things up */
                    for(int i = 0; i < 5000; i++)
                    {
                        QTL_Index[i]=0; QTL_MapPosition[i]=0.0; QTL_Type[i]=0; QTL_Allele[i]=0;
                        QTL_Add_Quan[i]=0.0; QTL_Dom_Quan[i]=0.0; QTL_Add_Fit[i]=0.0; QTL_Dom_Fit[i]=0.0;
                    }
                    /* Set up Counters */
                    vector < double > New_Map((TotalQTL + TotalMarker + CounterMutationIndex),0.0);     /* New Marker Map */
                    /* New QTL Marker Indicator; 0 = old marker/qtl; 2 = new qtl */
                    vector < int > New_QTLMarker((TotalQTL + TotalMarker + CounterMutationIndex),0);
                    int NewarrayCounter = 0;                                        /* Keep track of where you are at in new arrays */
                    int MutationCounter = 0;                                        /* Keep track of which mutation you are adding */
                    int Add_DomCounter = 0;                                         /* Keep track where add in old Add and Dom array Quantitative */
                    m_counter = 0;                                                  /* Counter to keep track where at in Marker Index */
                    qtl_counter = 0;                                                /* Counter to keep track where at in quantitative QTL Index */
                    /* Update QTLMapPosition, QTL_Add, QTL_Dom, Marker Index, QTL Index */
                    for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                    {
                        /* if current Map position is less than current mutation than don't need to add new mutation between previous SNP and current SNP */
                        /* Also still has to be mutation event left or else goes to last if statement */
                        if((FullMap[i] < MutationLoc[MutationCounter]) && (MutationCounter < CounterMutationIndex))
                        {
                            New_QTLMarker[NewarrayCounter] = 0;
                            New_Map[NewarrayCounter] = FullMap[i];
                            /* Index it based on whether it is old QTL or marker */
                            if(Old_QTLMarker[i] == 1)                   /* Marker */
                            {
                                MarkerIndex[m_counter] = NewarrayCounter; MarkerMapPosition[m_counter] = FullMap[i]; m_counter++;
                            }
                            if(Old_QTLMarker[i] > 1)
                            {
                                QTL_Index[qtl_counter] = NewarrayCounter; QTL_MapPosition[qtl_counter] = FullMap[i]; QTL_Type[qtl_counter] = Old_QTLMarker[i];
                                QTL_Add_Quan[qtl_counter] = Old_Add_Quan[Add_DomCounter]; QTL_Dom_Quan[qtl_counter] = Old_Dom_Quan[Add_DomCounter];
                                QTL_Add_Fit[qtl_counter] = Old_Add_Fit[Add_DomCounter]; QTL_Dom_Fit[qtl_counter] = Old_Dom_Fit[Add_DomCounter];
                                QTL_Allele[qtl_counter] = Old_QTL_Allele[Add_DomCounter];
                                qtl_counter++; Add_DomCounter++;
                            }
                            NewarrayCounter++;
                        }
                        /* if current Map position in greater than current mutation than need to add new mutation between previous SNP and current SNP */
                        /* Also still has to be mutation event left or else goes to last if statement; New Mutation was a quantitative Trait */
                        if(FullMap[i] > MutationLoc[MutationCounter] && MutationCounter < CounterMutationIndex)
                        {
                            /* add location to new map */
                            New_QTLMarker[NewarrayCounter] = 2; New_Map[NewarrayCounter] = MutationLoc[MutationCounter];
                            /* Index it into quantitative QTL Index */
                            QTL_Index[qtl_counter] = NewarrayCounter; QTL_MapPosition[qtl_counter] = MutationLoc[MutationCounter];
                            QTL_Type[qtl_counter] = MutationType[MutationCounter]; QTL_Allele[qtl_counter] = 2;
                            QTL_Add_Quan[qtl_counter] = MutationAdd_quan[MutationCounter]; QTL_Dom_Quan[qtl_counter] = MutationDom_quan[MutationCounter];
                            QTL_Add_Fit[qtl_counter] = MutationAdd_fit[MutationCounter]; QTL_Dom_Fit[qtl_counter] = MutationDom_fit[MutationCounter];
                            NewarrayCounter++; qtl_counter++; MutationCounter++;
                            while(1) /* repeat until break */
                            {
                                /* Next new mutation is also greater than current map position add new mutation between previous SNP and current SNP */
                                /* Also still has to be mutation event left or else goes to last if statement */
                                if(FullMap[i] > MutationLoc[MutationCounter] && MutationCounter < CounterMutationIndex)
                                {
                                    /* add location to new map */
                                    New_QTLMarker[NewarrayCounter] = 2; New_Map[NewarrayCounter] = MutationLoc[MutationCounter];
                                    /* Index it into quantitative QTL Index */
                                    QTL_Index[qtl_counter] = NewarrayCounter; QTL_MapPosition[qtl_counter] = MutationLoc[MutationCounter];
                                    QTL_Type[qtl_counter] = MutationType[MutationCounter]; QTL_Allele[qtl_counter] = 2;
                                    QTL_Add_Quan[qtl_counter] = MutationAdd_quan[MutationCounter]; QTL_Dom_Quan[qtl_counter] = MutationDom_quan[MutationCounter];
                                    QTL_Add_Fit[qtl_counter] = MutationAdd_fit[MutationCounter]; QTL_Dom_Fit[qtl_counter] = MutationDom_fit[MutationCounter];
                                    NewarrayCounter++; qtl_counter++; MutationCounter++;
                                }
                                /* Current Map position less than current mutation than don't need to add new mutation between previous SNP and current SNP */
                                /* Also still has to be mutation event left or else goes to last if statement */
                                if(FullMap[i] < MutationLoc[MutationCounter] && MutationCounter < CounterMutationIndex)
                                {
                                    New_QTLMarker[NewarrayCounter] = 0;
                                    New_Map[NewarrayCounter] = FullMap[i];
                                    /* Index it based on whether it is old QTL or marker */
                                    if(Old_QTLMarker[i] == 1)                   /* Marker */
                                    {
                                        MarkerIndex[m_counter] = NewarrayCounter; MarkerMapPosition[m_counter] = FullMap[i]; m_counter++;
                                    }
                                    if(Old_QTLMarker[i] > 1)
                                    {
                                        QTL_Index[qtl_counter] = NewarrayCounter; QTL_MapPosition[qtl_counter] = FullMap[i]; QTL_Type[qtl_counter] = Old_QTLMarker[i];
                                        QTL_Add_Quan[qtl_counter] = Old_Add_Quan[Add_DomCounter]; QTL_Dom_Quan[qtl_counter] = Old_Dom_Quan[Add_DomCounter];
                                        QTL_Add_Fit[qtl_counter] = Old_Add_Fit[Add_DomCounter]; QTL_Dom_Fit[qtl_counter] = Old_Dom_Fit[Add_DomCounter];
                                        QTL_Allele[qtl_counter] = Old_QTL_Allele[Add_DomCounter];
                                        qtl_counter++; Add_DomCounter++;
                                    }
                                    NewarrayCounter++;
                                    break;
                                }
                                /* No more mutations so quite while loop */
                                if(MutationCounter >= CounterMutationIndex){break;}
                            }
                        }
                        /* No more mutations remain so just add to new array based on only QTL and marker information */
                        if(MutationCounter >= CounterMutationIndex)
                        {
                            New_QTLMarker[NewarrayCounter] = 0;
                            New_Map[NewarrayCounter] = FullMap[i];
                            /* Index it based on whether it is old QTL or marker */
                            if(Old_QTLMarker[i] == 1)                   /* Marker */
                            {
                                MarkerIndex[m_counter] = NewarrayCounter; MarkerMapPosition[m_counter] = FullMap[i]; m_counter++;
                            }
                            if(Old_QTLMarker[i] > 1)
                            {
                                QTL_Index[qtl_counter] = NewarrayCounter; QTL_MapPosition[qtl_counter] = FullMap[i]; QTL_Type[qtl_counter] = Old_QTLMarker[i];
                                QTL_Add_Quan[qtl_counter] = Old_Add_Quan[Add_DomCounter]; QTL_Dom_Quan[qtl_counter] = Old_Dom_Quan[Add_DomCounter];
                                QTL_Add_Fit[qtl_counter] = Old_Add_Fit[Add_DomCounter]; QTL_Dom_Fit[qtl_counter] = Old_Dom_Fit[Add_DomCounter];
                                QTL_Allele[qtl_counter] = Old_QTL_Allele[Add_DomCounter];
                                qtl_counter++; Add_DomCounter++;
                            }
                            NewarrayCounter++;
                        }
                    }
                    logfile << "       - Updated Index and Map Positions." << endl;
                    /////////////////////////////////////////////////////
                    /// Part9: Update genotype with mutations.        ///
                    /////////////////////////////////////////////////////
                    /* Update new gamete with mutations based on New_QTL array (0 means everyone has it; 2 is a new mutation */
                    for(int an = 0; an < CounterAnimGamIndex; an++)
                    {
                        vector < int > oldhaplotype((TotalQTL + TotalMarker),0);                        /* Gamete without mutations */
                        vector < int > newhaplotype((TotalQTL + TotalMarker + CounterMutationIndex),0); /* Gamete that includes mutations */
                        string tempGamete = AnimGam_Gam[an];
                        /* Put into old gamete into an array */
                        for(int i = 0; i < (TotalQTL + TotalMarker); i++){oldhaplotype[i] = tempGamete[i] - 48;}
                        MutationCounter = 0;                                                        /* Keep track of which mutation you are adding */
                        int OldGameteCounter = 0;                                                   /* Keep track of where you are at in old gamete */
                        for(int i = 0; i < (TotalQTL + TotalMarker + CounterMutationIndex); i++)
                        {
                            /* SNP was a marker everyone in current generation has it */
                            if(New_QTLMarker[i] == 0){newhaplotype[i] = oldhaplotype[OldGameteCounter]; OldGameteCounter++;}
                            /* New Mutation was created therefore only one animal has it */
                            if(New_QTLMarker[i] == 2)
                            {
                                /* Current animal and gamete is where it happened */
                                if(AnimGam_ID[an] == MutationAnim[MutationCounter] && AnimGam_GamID[an] == MutationGamete[MutationCounter]){newhaplotype[i] = 2;}
                                /* Current animal and gamete is not where it happened */
                                if(AnimGam_ID[an] != MutationAnim[MutationCounter] || AnimGam_GamID[an] != MutationGamete[MutationCounter]){newhaplotype[i] = 1;}
                                MutationCounter++;
                            }
                        }
                        /* Put back into String */
                        string gamete;
                        stringstream strStream (stringstream::in | stringstream::out);
                        for (int i=0; i < (TotalQTL + TotalMarker + CounterMutationIndex); i++){strStream << newhaplotype[i];}
                        string temp = strStream.str();
                        AnimGam_Gam[an] = temp;
                        if(an == CounterAnimGamIndex - 1)
                        {
                            logfile << "       - Updated Progeny Genotypes with Mutations (Gamete Size: " << AnimGam_Gam[an].size() << ")." << endl;
                        }
                    }
                    /** need to also update parent genotypes if they stick around in order to keep QTL length the same */
                    for(int i = 0; i < parentgeno.size(); i++)
                    {
                        MutationCounter = 0;                                                        /* Set to first mutation event */
                        string oldgeno = parentgeno[i];                                             /* Grab old genotype */
                        vector < int > geno((TotalQTL + TotalMarker),0);                            /* Full genotype without mutations */
                        vector < int > genomut((TotalQTL + TotalMarker + CounterMutationIndex),0);  /* Full genotype with mutations */
                        /* put both marker and QTL genotypes into an array instead of a string */
                        for(int j = 0; j < TotalMarker + TotalQTL; j++){geno[j] = oldgeno[j] - 48;}
                        /* update Genotype with new mutations */
                        int OldGameteCounter = 0;                                   /* Keep track of where you are at in old gamete */
                        for(int j = 0; j < (TotalQTL + TotalMarker + CounterMutationIndex); j++)
                        {
                            /* SNP was a marker everyone in current generation has it */
                            if(New_QTLMarker[j] == 0){genomut[j] = geno[OldGameteCounter]; OldGameteCounter++;}
                            if(New_QTLMarker[j] == 2){genomut[j] = 0;}       /* New Mutation only in gamete has it there for all animals are genotype 0 (11) */
                        }
                        /* Put back into QTL string */
                        int* updQTL = new int[5000];                                        /* QTL Genotypes */
                        qtl_counter = 0;                                                    /* Counter to keep track where at in quantitative QTL Index */
                        /* Loop through MarkerGenotypes array and place Genotypes based on Index value */
                        for(int j = 0; j < (TotalQTL + TotalMarker + CounterMutationIndex); j++)
                        {
                            if(j == QTL_Index[qtl_counter]){updQTL[qtl_counter] = genomut[j]; qtl_counter++;}
                        }
                        /* Put back in string and update in animal class object */
                        stringstream strStreamQn (stringstream::in | stringstream::out);
                        for(int j=0; j < (qtl_counter); j++){strStreamQn << updQTL[j];}
                        string QTn = strStreamQn.str();
                        population[i].UpdateQTLGenotype(QTn);
                        delete [] updQTL;
                    }
                    logfile << "       - Updated Parent Genotypes with Mutations." << endl;
                }
                FullMap.clear();
                time_t end_block2 = time(0);
                logfile << "   Finished creating gametes based on parent genotypes and mutation events (Time: ";
                logfile << difftime(end_block2,start_block2) << " seconds)." << endl << endl;
                logfile << "   Begin " << Mating << " mating: " << endl;
                time_t start_block3 = time(0);
                int LethalFounder = 0;
                /* First check to see how many male (rows) and female (columns) derived gametes there are are */
                int malegametecount = 0;                    /* Number of male derived gametes */
                vector < int > malerow;                     /* where at in mate allocation value matrix */
                vector < int > maleid;                      /* ID of parent for ID (has to be male) */
                vector < int > maleA;                       /* Where at in Relationship Matrix */
                vector < double > maleUni;                  /* Uniform deviate used for random mating */
                vector < string > malegamete;               /* gamete for row */
                int femalegametecount = 0;                  /* Number of female derived gametes */
                vector < int > femalerow;                   /* where at in mate allocation value matrix */
                vector < int > femaleid;                    /* ID of parent for ID (has to be male) */
                vector < int > femaleA;                     /* Where at in Relationship Matrix */
                vector < double > femaleUni;                /* Uniform deviate used for random mating */
                vector < string > femalegamete;             /* gamete for row */
                vector < int > sire_mate_column;            /* column that corresponds to female that will be mated */
                for(int i = 0; i < CounterAnimGamIndex; i++)
                {
                    if(AnimGam_Sex[i] == 0)                                     /* if Male */
                    {
                        malerow.push_back(malegametecount); maleid.push_back(AnimGam_ID[i]); maleA.push_back(-5);
                        maleUni.push_back(AnimGam_Dev[i]); malegamete.push_back(AnimGam_Gam[i]); malegametecount++;
                    }
                    if(AnimGam_Sex[i] == 1)                                     /* if Female */
                    {
                        femalerow.push_back(femalegametecount); femaleid.push_back(AnimGam_ID[i]); femaleA.push_back(-5);
                        femaleUni.push_back(AnimGam_Dev[i]); femalegamete.push_back(AnimGam_Gam[i]); femalegametecount++;
                    }
                }
                double* mate_value_matrix = new double[malegametecount * femalegametecount];            /* Filled by values which want to minimize or maximize */
                for(int i = 0; i < (malegametecount * femalegametecount); i++){mate_value_matrix[i] = 0;}
                /* Get ID of parents in order to create a relationship matrix of some kind */
                vector < int > parentID;
                for(int i = 0; i < population.size(); i++){parentID.push_back(population[i].getID());}  /* Loop through and grab parent IDs */
                if(Mating == "minROH")
                {
                    vector < int > idgenorel;                                       /* ID row column geno matrix */
                    /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
                    /* This way you don't have to repeat this step for each haplotype */
                    vector < vector < int > > pathaploIDs;
                    vector < vector < int > > mathaploIDs;
                    for(int i = 0; i < population.size(); i++)
                    {
                        /* Grab Animal ID and maternal and paternal for GRM function */
                        idgenorel.push_back(population[i].getID());
                        string PaternalHap = population[i].getPatHapl();                        /* Grab Paternal Haplotype for Individual i */
                        string MaternalHap = population[i].getMatHapl();                        /* Grab Maternal Haplotype for Individual i */
                        vector < int > temp_pat;
                        string quit = "NO";
                        while(quit != "YES")
                        {
                            size_t pos = PaternalHap.find("_",0);                               /* search until last one yet */
                            if(pos > 0){temp_pat.push_back(stoi(PaternalHap.substr(0,pos))); PaternalHap.erase(0, pos + 1);}    /* extend column by 1 */
                            if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
                        }
                        pathaploIDs.push_back(temp_pat);                               /* push back row */
                        vector < int > temp_mat;
                        quit = "NO";
                        while(quit != "YES")
                        {
                            size_t pos = MaternalHap.find("_",0);                               /* search until last one yet */
                            if(pos > 0){temp_mat.push_back(stoi(MaternalHap.substr(0,pos))); MaternalHap.erase(0, pos + 1);}    /* extend column by 1 */
                            if(pos == std::string::npos){quit = "YES";}                         /* has reached last one so now kill while loop */
                        }
                        mathaploIDs.push_back(temp_mat);                               /* push back row */
                    }
                    /* the order of the sub relationship will be that same as the order of the idgenorel vector; so figure where at in sub relationship now */
                    for(int i = 0; i < maleid.size(); i++)
                    {
                        int j = 0;
                        while(j < idgenorel.size())
                        {
                            if(maleid[i] == idgenorel[j]){maleA[i] = j; break;}
                            if(maleid[i] != idgenorel[j]){j++;}
                            if(j == (idgenorel.size())){cout << "Couldn't Find Animal in Mating Design " << j << endl; exit (EXIT_FAILURE);}
                        }
                    }
                    for(int i = 0; i < femaleid.size(); i++)
                    {
                        int j = 0;
                        while(j < idgenorel.size())
                        {
                            if(femaleid[i] == idgenorel[j]){femaleA[i] = j; break;}
                            if(femaleid[i] != idgenorel[j]){j++;}
                            if(j == (idgenorel.size())){cout << "Couldn't Find Animal in Mating Design " << j << endl; exit (EXIT_FAILURE);}
                        }
                    }
                    double *_grm_mkl = new double[population.size()*population.size()];     /* Allocate Memory for GRM */
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
                        for(i = 0; i < population.size(); i++)
                        {
                            for(j = i; j < population.size(); j++)
                            {
                                _grm_mkl[(i*population.size())+j] += (H_matrix[((pathaploIDs[i][hap])*haplotypes.size())+(pathaploIDs[j][hap])] +
                                                                      H_matrix[((pathaploIDs[i][hap])*haplotypes.size())+(mathaploIDs[j][hap])] +
                                                                      H_matrix[((mathaploIDs[i][hap])*haplotypes.size())+(pathaploIDs[j][hap])] +
                                                                      H_matrix[((mathaploIDs[i][hap])*haplotypes.size())+(mathaploIDs[j][hap])]) / 2;
                                _grm_mkl[(j*population.size())+i] =  _grm_mkl[(i*population.size())+j];
                            }
                        }
                        delete [] H_matrix;
                    }
                    for(int i = 0; i < population.size(); i++)
                    {
                        for(int j = 0; j < population.size(); j++)
                        {
                            _grm_mkl[(i*population.size())+j] = _grm_mkl[(i*population.size())+j] / double(haplib.size());
                        }
                    }
                    /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
                    for(int i = 0; i < malegametecount; i++)
                    {
                        for(int j = 0; j < femalegametecount; j++)
                        {
                            mate_value_matrix[(i*malegametecount)+j] = _grm_mkl[(maleA[i]*population.size())+femaleA[j]];
                        }
                    }
                    delete[] _grm_mkl;
                    /* randomly mate male and female gametes; copy male uniform gamete with row then randomly sort */
                    vector < double > copy_male_uni;
                    for(int i = 0; i < maleUni.size(); i++){copy_male_uni.push_back(maleUni[i]); sire_mate_column.push_back(i);}
                    double temp_uniform; int temp_id;
                    for(int i = 0; i < maleUni.size()-1; i++)
                    {
                        for(int j=i+1; j < maleUni.size(); j++)
                        {
                            if(copy_male_uni[i] > copy_male_uni[j])
                            {
                                /* put i values in temp variables */
                                temp_id = sire_mate_column[i]; temp_uniform = copy_male_uni[i];
                                /* swap lines */
                                sire_mate_column[i] = sire_mate_column[j]; copy_male_uni[i] = copy_male_uni[j];
                                /* put temp values in 1 backward */
                                sire_mate_column[j] = temp_id; copy_male_uni[j] = temp_uniform;
                            }
                        }
                    }
                    double mean = 0.0; double min = 1.0; double max = 0.0; double initial_inbreeding;
                    vector < double > coancestoryvalues(maleUni.size(), 0.0);
                    for(int i = 0; i < maleUni.size(); i++)
                    {
                        double temp = mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];
                        coancestoryvalues[i] = temp;
                        mean += temp;
                        if(temp < min){min = temp;}
                        if(temp > max){max = temp;}
                    }
                    logfile << "       - Number of male and female gametes: " << malegametecount << " " << femalegametecount << endl;
                    logfile << "       - Inbreeding Level prior to mating strategy: " << endl;
                    logfile << "            - Initial Mean (Min - Max) Coancestory: " << mean / double(maleUni.size()) << " (" << min << " - " << max << ")." << endl;
                    logfile << "       - Minimize coancestries based on Pedigree information." << endl;
                    double temperature = 10000.0;
                    double epsilon = 0.001;
                    double alpha = 0.9999;
                    /* Simulated Annealing Algorithm */
                    while(temperature > epsilon)
                    {
                        vector < int > next_sire_mate_column;
                        for(int i = 0; i < sire_mate_column.size(); i++)
                        {
                            next_sire_mate_column.push_back(sire_mate_column[i]);
                        }
                        /* Randomly swap two elements */
                        int change[2];
                        for(int i = 0; i < 2; i++)
                        {
                            std::uniform_real_distribution<double> distribution15(0,sire_mate_column.size());
                            change[i] = distribution15(gen);
                            if(i == 1)
                            {
                                if(change[0] == change[1]){i = i-1;}
                            }
                        }
                        // change elements
                        int oldfirstone = next_sire_mate_column[change[0]];
                        next_sire_mate_column[change[0]] = next_sire_mate_column[change[1]];
                        next_sire_mate_column[change[1]] = oldfirstone;
                        // compute previous one
                        double cost_previous = 0.0;
                        for(int i = 0; i < sire_mate_column.size(); i++){cost_previous += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                        // compute new one
                        double cost_current = 0.0;
                        for(int i = 0; i < sire_mate_column.size(); i++){cost_current += mate_value_matrix[(i*malegametecount)+next_sire_mate_column[i]];}
                        // Used to determine if you want to change it it is worse
                        double sa = exp((-cost_previous-cost_current)/temperature);
                        // check the boolean condition
                        if(cost_current < cost_previous)
                        {
                            for(int i = 0; i < sire_mate_column.size(); i++){sire_mate_column[i] = next_sire_mate_column[i];}
                        }
                        else
                        {
                            std::uniform_real_distribution<double> distribution16(0,1);
                            double sample = distribution16(gen);
                            if(sample < sa)
                            {
                                for(int i = 0; i < sire_mate_column.size(); i++)
                                {
                                    sire_mate_column[i] = next_sire_mate_column[i];
                                }
                            }
                        }
                        temperature = temperature * alpha;
                    }
                    mean = 0.0; min = 1.0; max = 0.0;
                    for(int i = 0; i < maleUni.size(); i++)
                    {
                        double temp = mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];
                        coancestoryvalues[i] = temp;
                        mean += temp;
                        if(temp < min){min = temp;}
                        if(temp > max){max = temp;}
                    }
                    logfile << "       - Inbreeding Level after mating strategy: " << endl;
                    logfile << "            - Initial Mean (Min - Max) Coancestory: " << mean / double(maleUni.size()) << " (" << min << " - " << max << ")." << endl;
                    delete [] mate_value_matrix;
                }
                if(Mating == "minGenomic")
                {
                    
                    vector < int > idgenorel;                                       /* ID row column geno matrix */
                    vector < string > creategenorel;                                /* Geno string for row column geno matrix */
                    for(int i = 0; i < population.size(); i++)
                    {
                        /* Grab Animal ID and genotype for GRM function */
                        idgenorel.push_back(population[i].getID()); creategenorel.push_back(population[i].getMarker());
                    }
                    /* the order of the sub relationship will be that same as the order of the idgenorel vector; so figure where at in sub relationship now */
                    for(int i = 0; i < maleid.size(); i++)
                    {
                        int j = 0;
                        while(j < idgenorel.size())
                        {
                            if(maleid[i] == idgenorel[j]){maleA[i] = j; break;}
                            if(maleid[i] != idgenorel[j]){j++;}
                            if(j == (idgenorel.size())){cout << "Couldn't Find Animal in Mating Design " << j << endl; exit (EXIT_FAILURE);}
                        }
                    }
                    for(int i = 0; i < femaleid.size(); i++)
                    {
                        int j = 0;
                        while(j < idgenorel.size())
                        {
                            if(femaleid[i] == idgenorel[j]){femaleA[i] = j; break;}
                            if(femaleid[i] != idgenorel[j]){j++;}
                            if(j == (idgenorel.size())){cout << "Couldn't Find Animal in Mating Design " << j << endl; exit (EXIT_FAILURE);}
                        }
                    }
                    /* Generate G matrix for parental animals */
                    double *_grm_mkl = new double[population.size()*population.size()];     /* Allocate Memory for GRM */
                    grm_noprevgrm(M,creategenorel,_grm_mkl,scale);                          /* Function to create GRM, with no previous grm */
                    creategenorel.clear();
                    
                    /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
                    for(int i = 0; i < malegametecount; i++)
                    {
                        for(int j = 0; j < femalegametecount; j++)
                        {
                            mate_value_matrix[(i*malegametecount)+j] = _grm_mkl[(maleA[i]*population.size())+femaleA[j]];
                        }
                    }
                    delete[] _grm_mkl;
                    /* randomly mate male and female gametes; copy male uniform gamete with row then randomly sort */
                    vector < double > copy_male_uni;
                    for(int i = 0; i < maleUni.size(); i++){copy_male_uni.push_back(maleUni[i]); sire_mate_column.push_back(i);}
                    double temp_uniform; int temp_id;
                    for(int i = 0; i < maleUni.size()-1; i++)
                    {
                        for(int j=i+1; j < maleUni.size(); j++)
                        {
                            if(copy_male_uni[i] > copy_male_uni[j])
                            {
                                /* put i values in temp variables */
                                temp_id = sire_mate_column[i]; temp_uniform = copy_male_uni[i];
                                /* swap lines */
                                sire_mate_column[i] = sire_mate_column[j]; copy_male_uni[i] = copy_male_uni[j];
                                /* put temp values in 1 backward */
                                sire_mate_column[j] = temp_id; copy_male_uni[j] = temp_uniform;
                            }
                        }
                    }
                    double mean = 0.0; double min = 1.0; double max = 0.0; double initial_inbreeding;
                    vector < double > coancestoryvalues(maleUni.size(), 0.0);
                    for(int i = 0; i < maleUni.size(); i++)
                    {
                        double temp = mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];
                        coancestoryvalues[i] = temp;
                        mean += temp;
                        if(temp < min){min = temp;}
                        if(temp > max){max = temp;}
                    }
                    logfile << "       - Number of male and female gametes: " << malegametecount << " " << femalegametecount << endl;
                    logfile << "       - Inbreeding Level prior to mating strategy: " << endl;
                    logfile << "            - Initial Mean (Min - Max) Coancestory: " << mean / double(maleUni.size()) << " (" << min << " - " << max << ")." << endl;
                    logfile << "       - Minimize coancestries based on Pedigree information." << endl;
                    double temperature = 10000.0;
                    double epsilon = 0.001;
                    double alpha = 0.9999;
                    /* Simulated Annealing Algorithm */
                    while(temperature > epsilon)
                    {
                        vector < int > next_sire_mate_column;
                        for(int i = 0; i < sire_mate_column.size(); i++)
                        {
                            next_sire_mate_column.push_back(sire_mate_column[i]);
                        }
                        /* Randomly swap two elements */
                        int change[2];
                        for(int i = 0; i < 2; i++)
                        {
                            std::uniform_real_distribution<double> distribution15(0,sire_mate_column.size());
                            change[i] = distribution15(gen);
                            if(i == 1)
                            {
                                if(change[0] == change[1]){i = i-1;}
                            }
                        }
                        // change elements
                        int oldfirstone = next_sire_mate_column[change[0]];
                        next_sire_mate_column[change[0]] = next_sire_mate_column[change[1]];
                        next_sire_mate_column[change[1]] = oldfirstone;
                        // compute previous one
                        double cost_previous = 0.0;
                        for(int i = 0; i < sire_mate_column.size(); i++){cost_previous += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                        // compute new one
                        double cost_current = 0.0;
                        for(int i = 0; i < sire_mate_column.size(); i++){cost_current += mate_value_matrix[(i*malegametecount)+next_sire_mate_column[i]];}
                        // Used to determine if you want to change it it is worse
                        double sa = exp((-cost_previous-cost_current)/temperature);
                        // check the boolean condition
                        if(cost_current < cost_previous)
                        {
                            for(int i = 0; i < sire_mate_column.size(); i++){sire_mate_column[i] = next_sire_mate_column[i];}
                        }
                        else
                        {
                            std::uniform_real_distribution<double> distribution16(0,1);
                            double sample = distribution16(gen);
                            if(sample < sa)
                            {
                                for(int i = 0; i < sire_mate_column.size(); i++)
                                {
                                    sire_mate_column[i] = next_sire_mate_column[i];
                                }
                            }
                        }
                        temperature = temperature * alpha;
                    }
                    mean = 0.0; min = 1.0; max = 0.0;
                    for(int i = 0; i < maleUni.size(); i++)
                    {
                        double temp = mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];
                        coancestoryvalues[i] = temp;
                        mean += temp;
                        if(temp < min){min = temp;}
                        if(temp > max){max = temp;}
                    }
                    logfile << "       - Inbreeding Level after mating strategy: " << endl;
                    logfile << "            - Initial Mean (Min - Max) Coancestory: " << mean / double(maleUni.size()) << " (" << min << " - " << max << ")." << endl;
                    delete [] mate_value_matrix;
                }
                if(Mating == "random" || Mating == "random5" || Mating == "random25" || Mating == "random125" || Mating == "minPedigree")
                {
                    /* the order of the sub relationship will be that same as the order of the parentID vector; so figure where at in sub relationship now */
                    for(int i = 0; i < maleid.size(); i++)
                    {
                        int j = 0;
                        while(j < parentID.size())
                        {
                            if(maleid[i] == parentID[j]){maleA[i] = j; break;}
                            if(maleid[i] != parentID[j]){j++;}
                            if(j == (parentID.size())){cout << "Couldn't Find Animal in Mating Design " << j << endl; exit (EXIT_FAILURE);}
                        }
                    }
                    for(int i = 0; i < femaleid.size(); i++)
                    {
                        int j = 0;
                        while(j < parentID.size())
                        {
                            if(femaleid[i] == parentID[j]){femaleA[i] = j; break;}
                            if(femaleid[i] != parentID[j]){j++;}
                            if(j == (parentID.size())){cout << "Couldn't Find Animal in Mating Design " << j << endl; exit (EXIT_FAILURE);}
                        }
                    }
                    /* Create Relationship Matrix */
                    double* subsetrelationship = new double[parentID.size()*parentID.size()];
                    pedigree_relationship(Pheno_Pedigree_File,parentID, subsetrelationship);
                    /* Once relationships are tabulated between parents and fill mate allocation matrix (sire by dams)*/
                    for(int i = 0; i < malegametecount; i++)
                    {
                        for(int j = 0; j < femalegametecount; j++)
                        {
                            mate_value_matrix[(i*malegametecount)+j] = subsetrelationship[(maleA[i]*parentID.size())+femaleA[j]];
                        }
                    }
                    /* randomly mate male and female gametes; copy male uniform gamete with row then randomly sort */
                    vector < double > copy_male_uni;
                    for(int i = 0; i < maleUni.size(); i++){copy_male_uni.push_back(maleUni[i]); sire_mate_column.push_back(i);}
                    double temp_uniform; int temp_id;
                    for(int i = 0; i < maleUni.size()-1; i++)
                    {
                        for(int j=i+1; j < maleUni.size(); j++)
                        {
                            if(copy_male_uni[i] > copy_male_uni[j])
                            {
                                /* put i values in temp variables */
                                temp_id = sire_mate_column[i]; temp_uniform = copy_male_uni[i];
                                /* swap lines */
                                sire_mate_column[i] = sire_mate_column[j]; copy_male_uni[i] = copy_male_uni[j];
                                /* put temp values in 1 backward */
                                sire_mate_column[j] = temp_id; copy_male_uni[j] = temp_uniform;
                            }
                        }
                    }
                    int greater125 = 0; int greater25 = 0; int greater5 = 0; double initial_inbreeding;
                    double mean = 0.0; double min = 1.0; double max = 0.0;
                    vector < double > coancestoryvalues(maleUni.size(), 0.0);
                    for(int i = 0; i < maleUni.size(); i++)
                    {
                        double temp = mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];
                        coancestoryvalues[i] = temp;
                        mean += temp;
                        if(temp < min){min = temp;}
                        if(temp > max){max = temp;}
                        if(temp >= 0.125){greater125++;}
                        if(temp >= 0.25){greater25++;}
                        if(temp >= 0.50){greater5++;}
                    }
                    logfile << "       - Number of male and female gametes: " << malegametecount << " " << femalegametecount << endl;
                    logfile << "       - Inbreeding Level prior to mating strategy: " << endl;
                    logfile << "            - Initial Mean (Min - Max) Coancestory: " << mean / double(maleUni.size()) << " (" << min << " - " << max << ")." << endl;
                    logfile << "            - Number of matings > 0.125 relationship: " << greater125 << "." << endl;
                    logfile << "            - Number of matings > 0.25 relationship: " << greater25 << "." << endl;
                    logfile << "            - Number of matings > 0.50 relationship: " << greater5 << "." << endl;
                    /* Depending on which matings are not allowed you set anything below the cutoff to 0 */
                    if(Mating == "random")
                    {
                        logfile << "       - All Matings Allowed." << endl;
                        for(int i = 0; i < (malegametecount * femalegametecount); i++){mate_value_matrix[i] = 0;} /* Set all to zero */
                        initial_inbreeding = 0.0;
                        for(int i = 0; i < maleUni.size(); i++){initial_inbreeding += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                    }
                    if(Mating == "random5")
                    {
                        logfile << "       - Only Matings less than 0.50 percent related allowed to mate." << endl;
                        for(int i = 0; i < (malegametecount * femalegametecount); i++)
                        {
                            if(mate_value_matrix[i] < 0.50){mate_value_matrix[i] = 0;} /* Set all to zero less than 0.5 */
                        }
                        initial_inbreeding = 0.0;
                        for(int i = 0; i < maleUni.size(); i++){initial_inbreeding += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                    }
                    if(Mating == "random25")
                    {
                        logfile << "       - Only Matings less than 0.25 percent related allowed to mate." << endl;
                        for(int i = 0; i < (malegametecount * femalegametecount); i++)
                        {
                            if(mate_value_matrix[i] < 0.25){mate_value_matrix[i] = 0;} /* Set all to zero less than 0.25 */
                        }
                        initial_inbreeding = 0.0;
                        for(int i = 0; i < maleUni.size(); i++){initial_inbreeding += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                    }
                    if(Mating == "random125")
                    {
                        logfile << "       - Only Matings less than 0.125 percent related allowed to mate." << endl;
                        for(int i = 0; i < (malegametecount * femalegametecount); i++)
                        {
                            if(mate_value_matrix[i] < 0.125){mate_value_matrix[i] = 0;} /* Set all to zero less than 0.125 */
                        }
                        initial_inbreeding = 0.0;
                        for(int i = 0; i < maleUni.size(); i++){initial_inbreeding += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                    }
                    if(Mating == "minPedigree")
                    {
                        logfile << "       - Minimize coancestries based on Pedigree information." << endl;
                        initial_inbreeding = 0.0;
                        for(int i = 0; i < maleUni.size(); i++){initial_inbreeding += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                    }
                    if(initial_inbreeding > 0.0)
                    {
                        double temperature = 10000.0;
                        double epsilon = 0.001;
                        double alpha = 0.9999;
                        /* Simulated Annealing Algorithm */
                        while(temperature > epsilon)
                        {
                            vector < int > next_sire_mate_column;
                            for(int i = 0; i < sire_mate_column.size(); i++)
                            {
                                next_sire_mate_column.push_back(sire_mate_column[i]);
                            }
                            /* Randomly swap two elements */
                            int change[2];
                            for(int i = 0; i < 2; i++)
                            {
                                std::uniform_real_distribution<double> distribution15(0,sire_mate_column.size());
                                change[i] = distribution15(gen);
                                if(i == 1)
                                {
                                    if(change[0] == change[1]){i = i-1;}
                                }
                            }
                            // change elements
                            int oldfirstone = next_sire_mate_column[change[0]];
                            next_sire_mate_column[change[0]] = next_sire_mate_column[change[1]];
                            next_sire_mate_column[change[1]] = oldfirstone;
                            // compute previous one
                            double cost_previous = 0.0;
                            for(int i = 0; i < sire_mate_column.size(); i++){cost_previous += mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];}
                            // compute new one
                            double cost_current = 0.0;
                            for(int i = 0; i < sire_mate_column.size(); i++){cost_current += mate_value_matrix[(i*malegametecount)+next_sire_mate_column[i]];}
                            // Used to determine if you want to change it it is worse
                            double sa = exp((-cost_previous-cost_current)/temperature);
                            // check the boolean condition
                            if(cost_current < cost_previous)
                            {
                                for(int i = 0; i < sire_mate_column.size(); i++){sire_mate_column[i] = next_sire_mate_column[i];}
                            }
                            else
                            {
                                std::uniform_real_distribution<double> distribution16(0,1);
                                double sample = distribution16(gen);
                                if(sample < sa)
                                {
                                    for(int i = 0; i < sire_mate_column.size(); i++)
                                    {
                                        sire_mate_column[i] = next_sire_mate_column[i];
                                    }
                                }
                            }
                            temperature = temperature * alpha;
                        }
                    }
                    /* See how mating pattern did */
                    for(int i = 0; i < malegametecount; i++)
                    {
                        for(int j = 0; j < femalegametecount; j++)
                        {
                            mate_value_matrix[(i*malegametecount)+j] = subsetrelationship[(maleA[i]*parentID.size())+femaleA[j]];
                        }
                    }
                    greater125 = 0; greater25 = 0; greater5 = 0;
                    mean = 0.0; min = 1.0; max = 0.0;
                    for(int i = 0; i < maleUni.size(); i++)
                    {
                        double temp = mate_value_matrix[(i*malegametecount)+sire_mate_column[i]];
                        coancestoryvalues[i] = temp;
                        mean += temp;
                        if(temp < min){min = temp;}
                        if(temp > max){max = temp;}
                        if(temp >= 0.125){greater125++;}
                        if(temp >= 0.25){greater25++;}
                        if(temp >= 0.50){greater5++;}
                    }
                    logfile << "       - Inbreeding Level after mating strategy: " << endl;
                    logfile << "            - Initial Mean (Min - Max) Coancestory: " << mean / double(maleUni.size()) << " (" << min << " - " << max << ")." << endl;
                    logfile << "            - Number of matings > 0.125 relationship: " << greater125 << "." << endl;
                    logfile << "            - Number of matings > 0.25 relationship: " << greater25 << "." << endl;
                    logfile << "            - Number of matings > 0.50 relationship: " << greater5 << "." << endl;
                    delete [] subsetrelationship;
                    delete [] mate_value_matrix;
                }
                /* Increase age of parents by one */
                for(int i = 0; i < population.size(); i++){population[i].UpdateAge();}
                /* loop across each row in mate allocation matrix, which is total number of matings */
                for(int i = 0; i < maleUni.size(); i++)
                {
                    string maleGame = malegamete[i];
                    string femaleGame = femalegamete[sire_mate_column[i]];
                    int SNP = maleGame.size();
                    /* Convert to Genotypes form */
                    vector < int > fullhomo1(SNP,0);            /* declare array for homologue 1 */
                    vector < int > fullhomo2(SNP,0);            /* declare array for homologue 1 */
                    vector < int > Geno(SNP,0);                 /* declare genotype array */
                    /* Convert Genotype string to genotype array */
                    for (int m = 0; m < SNP; m++){fullhomo1[m] = maleGame[m] - 48; fullhomo2[m] = femaleGame[m] - 48;}
                    // From two haplotypes create genotypes: 0 = a1,a1; 2 = a2,a2; 3 = a1,a2; 4 = a2,a1
                    for (int m = 0; m < SNP; m++)
                    {
                        if(fullhomo1[m] == 1 & fullhomo2[m] == 1){Geno[m] = 0;}             /* genotype a1a1 */
                        if(fullhomo1[m] == 2 & fullhomo2[m] == 2){Geno[m] = 2;}             /* genotype a2a2 */
                        if(fullhomo1[m] == 1 & fullhomo2[m] == 2){Geno[m] = 3;}             /* genotype a1a2 */
                        if(fullhomo1[m] == 2 & fullhomo2[m] == 1){Geno[m] = 4;}             /* genotype a2a1 */
                    }
                    /* Split off into Marker and QTL based on updated index */
                    /* MarkerGeno array contains Markers and QTL therefore need to split them off based on index arrays that were created previously */
                    vector < int > MarkerGenotypes(NUMBERMARKERS,0);        /* Marker Genotypes; Will always be of this size */
                    vector < int > QTLGenotypes(QTL_Index.size(),0);        /* QTL Genotypes */
                    m_counter = 0;                                          /* Counter to keep track where at in Marker Index */
                    qtl_counter = 0;                                        /* Counter to keep track where at in QTL Index */
                    /* Fill Genotype Array */
                    for(int j = 0; j < (TotalQTL + TotalMarker + CounterMutationIndex); j++) /* Place Genotypes based on Index value */
                    {
                        if(m_counter < MarkerIndex.size())                          /* ensures doesn't go over and cause valgrind error */
                        {
                            if(j == MarkerIndex[m_counter]){MarkerGenotypes[m_counter] = Geno[j]; m_counter++;}
                        }
                        if(qtl_counter < QTL_Index.size())                           /* ensures doesn't go over and cause valgrind error */
                        {
                            if(j == QTL_Index[qtl_counter]){QTLGenotypes[qtl_counter] = Geno[j]; qtl_counter++;}
                        }
                    }
                    Geno.clear();
                    /* Before you put the individual in the founder population need to determine if it dies */
                    /* Starts of as a viability of 1.0 */
                    double relativeviability = 1.0;                         /* represents the multiplicative fitness effect across lethal and sub-lethal alleles */
                    for(int j = 0; j < QTL_IndCounter; j++)
                    {
                        if(QTL_Type[j] == 3 || QTL_Type[j] == 4 || QTL_Type[j] == 5)            /* Fitness QTL */
                        {
                            if(QTLGenotypes[j] == QTL_Allele[j]){relativeviability = relativeviability * (1-(QTL_Add_Fit[j]));}
                            if(QTLGenotypes[j] > 2){relativeviability = relativeviability * (1-(QTL_Dom_Fit[j] * QTL_Add_Fit[j]));}
                        }
                    }
                    /* now take a draw from a uniform and if less than relativeviability than survives if greater than dead */
                    std::uniform_real_distribution<double> distribution5(0,1);
                    double draw = distribution5(gen);
                    if(draw > relativeviability)                                                    /* Animal Died due to a low fitness */
                    {
                        /* Need to add one to dead progeny for sire and dam */
                        int j = 0;
                        while(j < population.size())
                        {
                            if(population[j].getID() == maleid[i]){population[j].Update_Dead(); break;}
                            j++;
                        }
                        j = 0;
                        while(j < population.size())
                        {
                            if(population[j].getID() == femaleid[sire_mate_column[i]]){population[j].Update_Dead(); break;}
                            j++;
                        }
                        string Qf;
                        stringstream strStreamQf (stringstream::in | stringstream::out);
                        for (int j=0; j < qtl_counter; j++){if(QTL_Type[j] == 3 || QTL_Type[j] == 4 || QTL_Type[j] == 5){strStreamQf << QTLGenotypes[j];}}
                        string QF = strStreamQf.str();
                        std::ofstream output1(lowfitnesspath, std::ios_base::app | std::ios_base::out);
                        output1 << maleid[i] << " " << femaleid[sire_mate_column[i]] << " " << relativeviability << " " <<  QF << endl;
                        LethalFounder++;
                    }
                    if(draw < relativeviability)                                                    /* Animal Survived */
                    {
                        //////////////////////////////////////////////////////////////////////////
                        // Step 8: Create founder file that has everything set for Animal Class //
                        //////////////////////////////////////////////////////////////////////////
                        // Set up paramters for Animal Class
                        /* Declare Variables */
                        double GenotypicValue = 0.0;                                       /* Stores Genotypic value; resets to zero for each line */
                        double BreedingValue = 0.0;                                        /* Stores Breeding Value; resets to zero for each line */
                        double DominanceDeviation = 0.0;                                   /* Stores Dominance Deviation; resets to zero for each line */
                        double Residual = 0.0;                                             /* Stores Residual Value; resets to zero for each line */
                        double Phenotype = 0.0;                                            /* Stores Phenotype; resets to zero for each line */
                        double Homoz = 0.0;                                                /* Stores homozygosity based on marker information */
                        double DiagGenoInb = 0.0;                                          /* Diagonal of Genomic Relationship Matrix */
                        double sex;                                                        /* draw from uniform to determine sex */
                        int Sex;                                                           /* Sex of the animal 0 is male 1 is female */
                        double residvar = 1 - (Variance_Additiveh2 + Variance_Dominanceh2);/* Residual Variance; Total Variance equals 1 */
                        residvar = sqrt(residvar);                                         /* random number generator need standard deviation */
                        /* Determine Sex of the animal based on draw from uniform distribution; if sex < 0.5 sex is 0 if sex >= 0.5 */
                        std::uniform_real_distribution<double> distribution5(0,1);
                        sex = distribution5(gen);
                        if(sex < 0.5){Sex = 0;}         /* Male */
                        if(sex >= 0.5){Sex = 1;}        /* Female */
                        /* Calculate Genotypic Value */
                        for(int j = 0; j < qtl_counter; j++)
                        {
                            if(QTL_Type[j] == 2 || QTL_Type[j] == 3)            /* Quantitative QTL */
                            {
                                int tempgeno;
                                if(QTLGenotypes[j] == 0 || QTLGenotypes[j] == 2){tempgeno = QTLGenotypes[j];}
                                if(QTLGenotypes[j] == 3 || QTLGenotypes[j] == 4){tempgeno = 1;}
                                /* Breeding value is only a function of additive effects */
                                BreedingValue += tempgeno * QTL_Add_Quan[j];
                                if(tempgeno != 1){GenotypicValue += tempgeno * QTL_Add_Quan[j];}     /* Not a heterozygote so only a function of additive */
                                if(tempgeno == 1)
                                {
                                    GenotypicValue += (tempgeno * QTL_Add_Quan[j]) + QTL_Dom_Quan[j];    /* Heterozygote so need to include add and dom */
                                    DominanceDeviation += QTL_Dom_Quan[j];
                                }
                            }
                        }
                        /* Calculate Homozygosity only in the MARKERS */
                        for(int j = 0; j < m_counter; j++)
                        {
                            if(MarkerGenotypes[j] == 0 || MarkerGenotypes[j] == 2){Homoz += 1;}
                            if(MarkerGenotypes[j] == 3 || MarkerGenotypes[j] == 4){Homoz += 0;}
                        }
                        Homoz = (Homoz/(m_counter));
                        /* Count number of homozygous fitness loci */
                        int homozygouscount_lethal = 0; int homozygouscount_sublethal = 0;
                        int heterzygouscount_lethal = 0; int heterzygouscount_sublethal = 0; double lethalequivalent = 0.0;
                        for(int j = 0; j < qtl_counter; j++)
                        {
                            if(QTL_Type[j] == 3 || QTL_Type[j] == 4 || QTL_Type[j] == 5)            /* Fitness QTL */
                            {
                                if(QTLGenotypes[j] == QTL_Allele[j])
                                {
                                    if(QTL_Type[j] == 4){homozygouscount_lethal += 1;}
                                    if(QTL_Type[j] == 3 || QTL_Type[j] == 5){homozygouscount_sublethal += 1;}
                                    lethalequivalent += QTL_Add_Fit[j];
                                }
                                if(QTLGenotypes[j] > 2)
                                {
                                    if(QTL_Type[j] == 4){heterzygouscount_lethal += 1;}
                                    if(QTL_Type[j] == 3 || QTL_Type[j] == 5){heterzygouscount_sublethal += 1;}
                                    lethalequivalent += QTL_Add_Fit[j];
                                }
                            }
                        }
                        float S = 0.0;
                        for(int j = 0; j < m_counter; j++)
                        {
                            double temp = MarkerGenotypes[j];
                            if(temp == 3 || temp == 4){temp = 1;}
                            temp = (temp - 1) - (2 * (founderfreq[j] - 0.5));           /* Convert it to - 1 0 1 and then put it into Z format */
                            S += temp * temp;                                           /* Multipy then add to sum */
                        }
                        double diagInb = S / scale;
                        /* Subtract off mean Genotypic Value in Founder Generation */
                        GenotypicValue = GenotypicValue - BaseGenGV;
                        BreedingValue = BreedingValue - BaseGenBV;
                        DominanceDeviation = DominanceDeviation - BaseGenDD;
                        /* put marker, qtl and fitness into string to store */
                        stringstream strStreamM (stringstream::in | stringstream::out);
                        for (int j=0; j < m_counter; j++){strStreamM << MarkerGenotypes[j];}
                        string MA = strStreamM.str();
                        stringstream strStreamQt (stringstream::in | stringstream::out);
                        for (int j=0; j < qtl_counter; j++){strStreamQt << QTLGenotypes[j];}
                        string QT = strStreamQt.str();
                        /* Sample from standard normal to generate environmental effect */
                        std::normal_distribution<double> distribution6(0.0,residvar);
                        Residual = distribution6(gen);
                        Phenotype = GenotypicValue + Residual;
                        double rndselection = distribution5(gen);
                        double rndculling = distribution5(gen);
                        /* ID S D Sex Gen Age Prog Mts DdPrg RndSel RndCul Ped_F Gen_F H1F H2F H3F FitHomo FitHeter LethEqv Homozy EBV Acc P GV R M QN QF PH MH*/
                        Animal animal(StartID,maleid[i],femaleid[sire_mate_column[i]],Sex,Gen,1,0,0,0,rndselection,rndculling,0.0,diagInb,0.0,0.0,0.0,homozygouscount_lethal, heterzygouscount_lethal, homozygouscount_sublethal, heterzygouscount_sublethal,lethalequivalent,Homoz,0.0,0.0,Phenotype,relativeviability,GenotypicValue,BreedingValue,DominanceDeviation,Residual,MA,QT,"","","");
                        /* Then place given animal object in population to store */
                        population.push_back(animal);
                        StartID++;                              /* Increment ID by one for next individual */
                    }
                }
                logfile << "       - Number of Progeny that Died due to fitness: " << LethalFounder << endl;
                NumDeadFitness[Gen] = LethalFounder;
                time_t end_block3 = time(0);
                logfile << "   Finished " << Mating << " mating (Time: " << difftime(end_block3,start_block3) << " seconds)." << endl << endl;
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Update Haplotype Library based on new animals and compute diagonals of relationship matrix                //
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                logfile << "    Begin Creating Haplotype Library and assigning haplotypes IDs to individuals: " << endl;
                time_t start_block4 = time(0);
                vector < string > AnimalPatHap;     /* Stores haplotype ID's in a string */
                vector < string > AnimalMatHap;     /* Stores haplotype ID's in a string */
                for(int i = 0; i < haplib.size(); i++)
                {
                    vector < string > haplotypes;
                    /* Unstring haplotypes, seperated by "_" */
                    string temphapstring = haplib[i].getHaplo();
                    string quit = "NO";
                    while(quit != "YES")
                    {
                        size_t pos = temphapstring.find("_",0);
                        if(pos > 0)                             /* hasn't reached last one yet */
                        {
                            haplotypes.push_back(temphapstring.substr(0,pos));
                            temphapstring.erase(0, pos + 1);
                        }
                        if(pos == std::string::npos)            /* has reached last one so now save last one and kill while loop */
                        {
                            quit = "YES";
                        }
                    }
                    int counterhapind = 0;                                                                              /* determine where at in haplotype ID */
                    for(int j = 0; j < population.size(); j++)
                    {
                        if(population[j].getAge() == 1)
                        {
                            if(i == 0){AnimalPatHap.push_back(""); AnimalMatHap.push_back("");}
                            AnimalPatHap.push_back(""); AnimalMatHap.push_back("");
                            string temp = (population[j].getMarker()).substr(haplib[i].getStart(),haplotypesize);       /* Grab specific haplotype */
                            string homo1 = temp;                                                                        /* Paternal haplotypes */
                            string homo2 = temp;                                                                        /* Maternal haplotypes */
                            for(int g = 0; g < temp.size(); g++)
                            {
                                if(homo1[g] == '0'){homo1[g] = '1'; homo2[g] = '1';}
                                if(homo1[g] == '2'){homo1[g] = '2'; homo2[g] = '2';}
                                if(homo1[g] == '3'){homo1[g] = '1'; homo2[g] = '2';}
                                if(homo1[g] == '4'){homo1[g] = '2'; homo2[g] = '1';}
                            }
                            /* Loop across two gametes and see if unique if so put in haplotype library */
                            for(int g = 0; g < 2; g++)
                            {
                                string temp;
                                if(g == 0){temp = homo1;}
                                if(g == 1){temp = homo2;}
                                int num = 0;                                            /* has to not match up with all unique haplotypes before added */
                                if(haplotypes.size() == 0){haplotypes.push_back(temp);} /* Haplotype library will be empty for first individual */
                                if(haplotypes.size() > 0)
                                {
                                    for(int h = 0; h < haplotypes.size(); h++)
                                    {
                                        if(temp.compare(haplotypes[h]) != 0){num++;}
                                        if(temp.compare(haplotypes[h]) == 0 && g == 0)  /* paste haplotype ID to paternal haplotype string */
                                        {
                                            if(i <= haplib.size() - 2)
                                            {
                                                std::ostringstream s;
                                                s << AnimalPatHap[counterhapind] << h << "_";
                                                AnimalPatHap[counterhapind] = s.str();
                                            }
                                            if(i == haplib.size() - 1)
                                            {
                                                std::ostringstream s;
                                                s << AnimalPatHap[counterhapind] << h;
                                                AnimalPatHap[counterhapind] = s.str();
                                            }
                                        }
                                        if(temp.compare(haplotypes[h]) == 0 && g == 1)  /* paste haplotype ID to maternal haplotype string */
                                        {
                                            if(i <= haplib.size() - 2)
                                            {
                                                std::ostringstream s;
                                                s << AnimalMatHap[counterhapind] << h << "_";
                                                AnimalMatHap[counterhapind]=s.str();
                                            }
                                            if(i == haplib.size() - 1)
                                            {
                                                std::ostringstream s;
                                                s << AnimalMatHap[counterhapind] << h;
                                                AnimalMatHap[counterhapind]=s.str();
                                            }
                                        }
                                    }
                                }
                                if(num == haplotypes.size())
                                {
                                    haplotypes.push_back(temp);                         /* If number not match = size of hapLibary then add */
                                    if(g == 0)                                          /* paste haplotype ID to paternal haplotype string */
                                    {
                                        if(i <= haplib.size() - 2)
                                        {
                                            std::ostringstream s;
                                            s << AnimalPatHap[counterhapind] << (haplotypes.size()-1) << "_";
                                            AnimalPatHap[counterhapind] = s.str();
                                        }
                                        if(i == haplib.size() - 1)
                                        {
                                            std::ostringstream s;
                                            s << AnimalPatHap[counterhapind] << (haplotypes.size()-1);
                                            AnimalPatHap[counterhapind] = s.str();
                                        }
                                    }
                                    if(g == 1)                                          /* paste haplotype ID to maternal haplotype string */
                                    {
                                        if(i <= haplib.size() - 2)
                                        {
                                            std::ostringstream s;
                                            s << AnimalMatHap[counterhapind] << (haplotypes.size()-1) << "_";
                                            AnimalMatHap[counterhapind] = s.str();
                                        }
                                        if(i == haplib.size() - 1)
                                        {
                                            std::ostringstream s;
                                            s << AnimalMatHap[counterhapind] << (haplotypes.size()-1);
                                            AnimalMatHap[counterhapind] = s.str();
                                        }
                                    }
                                }
                            }   /* Close loop that loops through twice, once for each gamete */
                            /* Finished looping across both haplotypes. Either is new or has already been their so can create portion of each haplotype matrix */
                            /**********************/
                            /* Haplotype 1 Matrix */
                            /**********************/
                            float sum=0.0;
                            double hvalue = 0.0;
                            for(int g = 0; g < homo1.size(); g++)
                            {
                                sum += abs(homo1[g] - homo2[g]);
                                if(g == homo1.size() - 1){hvalue = (1 - (sum/homo1.size())) + 1;}
                            }
                            population[j].AccumulateH1(hvalue);                        /* Add to diagonal of population */
                            /**********************/
                            /* Haplotype 2 Matrix */
                            /**********************/
                            hvalue = 0.0;
                            int match[homo1.size()];                                    /* matrix that has 1 to match and 0 if not match */
                            for(int g = 0; g < homo1.size(); g++)
                            {
                                match[g] = 1 - abs(homo1[g] - homo2[g]);
                            }
                            double sumGlobal = 0;
                            double sumh2=0;
                            for(int g = 0; g < homo1.size(); g++)
                            {
                                if(match[g] < 1)
                                {
                                    sumGlobal = sumGlobal + sumh2*sumh2;
                                    sumh2 = 0;
                                } else {
                                    sumh2 = sumh2 + 1;
                                }
                            }
                            /* don't need to divide by 2 because is 1 + 1 + sqrt(Sum/length) + sqrt(Sum/length) */
                            hvalue = 1 + sqrt((sumGlobal + (sumh2 * sumh2)) / (homo1.size() * homo1.size()));
                            population[j].AccumulateH2(hvalue);                        /* Add to diagonal of population */
                            /**********************/
                            /* Haplotype 3 Matrix */
                            /**********************/
                            double sumh3 = 0.0;
                            hvalue = 0.0;
                            for(int g = 0; g < homo1.size(); g++)
                            {
                                sumh3 += match[g];
                            }
                            if(sumh3 == homo1.size()){hvalue = 2.0;}
                            if(sumh3 != homo1.size()){hvalue = 1.0;}
                            population[j].AccumulateH3(hvalue);                        /* Add to diagonal of population */
                            /* Finished with 3 haplotype based methods */
                            /* Once reached last individual put all unique haplotypes into string with a "_" delimter to split them apart later */
                            if(j == population.size() -1)
                            {   string temp;
                                for(int h = 0; h < haplotypes.size(); h++)
                                {
                                    if(h == 0){temp = haplotypes[h];}
                                    if(h > 0){temp = temp + "_" + haplotypes[h];}
                                }
                                haplib[i].UpdateHaplotypes(temp);
                            }
                            /* Once get to last haplotype segment need to standardize by number of haplotype segments */
                            if(i == haplib.size() - 1)
                            {
                                double denom = haplib.size();
                                population[j].StandardizeH1(denom);
                                population[j].StandardizeH2(denom);
                                population[j].StandardizeH3(denom);
                                population[j].Update_PatHap(AnimalPatHap[counterhapind]);
                                population[j].Update_MatHap(AnimalMatHap[counterhapind]);
                            }
                            counterhapind++;
                        }
                    }
                }
                time_t end_block4 = time(0);
                logfile << "    Finished Creating Haplotype Library and assigning haplotypes IDs to individuals (Time: ";
                logfile << difftime(end_block4,start_block4) << " seconds)." << endl << endl;
                /////////////////////////////////////////
                // Culll a proportion of the parents   //
                /////////////////////////////////////////
                logfile << "   Begin " << Culling << " Culling: " << endl;
                time_t start_block5 = time(0);
                /* Regardless of the method if Sire and Dam Replacement is 1.0 then all animals are removed */
                if(SireReplacement == 1.0 && DamReplacement == 1.0)
                {
                    /* Output all parents due to non-overlapping generations */
                    /* Remove all animals that are older than 1 year of age */
                    /* Remove Unselected Animals from class object and then resize vector from population */
                    stringstream outputstring(stringstream::out);
                    stringstream outputstringgeno(stringstream::out);
                    int ROWS = population.size();                           /* Current Size of Population Class */
                    int i = 0;
                    while(i < ROWS)
                    {
                        while(1)
                        {
                            if(population[i].getAge() > 1)
                            {
                                /* Output info into master file with everything in it */
                                outputstring<<population[i].getID()<<" "<<population[i].getSire()<<" "<<population[i].getDam() <<" "<< population[i].getSex() << " ";
                                outputstring<<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                                outputstring<<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" ";
                                outputstring<<population[i].getHap1_F() <<" "<<population[i].getHap2_F()<<" "<< population[i].getHap3_F() <<" ";
                                outputstring<<population[i].getunfavhomolethal()<<" "<<population[i].getunfavheterolethal()<<" ";
                                outputstring<<population[i].getunfavhomosublethal()<<" ";
                                outputstring<<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                                outputstring<<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" ";
                                outputstring<<population[i].getAcc()<<" ";
                                outputstring<<population[i].getGenotypicValue()<<" ";
                                outputstring<<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                                if(OutputGeno == "yes" && Gen >= outputgeneration)
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
                    std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
                    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
                    /* output master geno file */
                    std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
                    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
                    logfile << "       -Non-overlapping Generations so all animals removed." << endl;
                    clock_t end_block5 = clock();
                    logfile << "   Finished " << Culling << " culling of parents (Time: ";
                    logfile << static_cast<double>( end_block5 - start_block5 )/CLOCKS_PER_SEC << " seconds)." << endl << endl;
                }
                /* Get rid of animals that will be culled regardless of method due to old age */
                if(SireReplacement < 1.0 || DamReplacement < 1.0)
                {
                    /* Automatically remove Animals that are at the maximum age then the remaining are used used for proportion culled */
                    int oldage = 0;                                         /* Counter for old age */
                    int ROWS = population.size();                           /* Current Size of Population Class */
                    int i = 0;
                    stringstream outputstring(stringstream::out);
                    stringstream outputstringgeno(stringstream::out);
                    while(i < ROWS)
                    {
                        while(1)
                        {
                            if(population[i].getAge() > MaximumAge)
                            {
                                /* Output info into master file with everything in it */
                                outputstring<<population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" "<< population[i].getSex() << " ";
                                outputstring<<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                                outputstring<<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" ";
                                outputstring<< population[i].getHap1_F() <<" ";
                                outputstring<<population[i].getHap2_F() <<" "<< population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                                outputstring<<population[i].getunfavheterolethal()<<" "<<population[i].getunfavhomosublethal()<<" ";
                                outputstring<<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                                outputstring<<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" ";
                                outputstring<<population[i].getAcc()<<" ";
                                outputstring<<population[i].getGenotypicValue()<<" ";
                                outputstring<<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                                if(OutputGeno == "yes" && Gen >= outputgeneration)
                                {
                                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                                }
                                population.erase(population.begin()+i);     /* Delete Animal from population */
                                ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                                oldage++;
                                break;
                            }
                            if(population[i].getAge() <= MaximumAge){i++; break;}                   /* since kept animal move to next slot */
                        }
                    }
                    /* output master df file */
                    std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
                    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
                    /* output master geno file */
                    std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
                    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
                    logfile << "       - Culled " << oldage << " Animals Due To Old Age. (New Population Size: " << population.size() <<")" << endl;
                    double MaleCutOff = 0.0;                                /* Value cutoff for males */
                    double FemaleCutOff = 0.0;                              /* Value cutoff for females */
                    int male = 0;                                           /* number of male animals that are of age 1 */
                    int female= 0;                                          /* number of female animals that are of age 1 */
                    vector < double > MaleValue;                            /* vector to hold male random values that are of age greater than 1 (male is 0)*/
                    vector < double > FemaleValue;                          /* veector to hold female random values that are of age greater than 1 (female is 1)*/
                    if(Culling == "random")
                    {
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getRndCulling()); male++;}
                            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getRndCulling()); female++;}
                        }
                    }
                    if(Culling == "phenotype")
                    {
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getPhenotype()); male++;}
                            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getPhenotype()); female++;}
                        }
                    }
                    if(Culling == "true_bv")
                    {
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getGenotypicValue()); male++;}
                            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getGenotypicValue());female++;}
                        }
                    }
                    if(Culling == "ebv")
                    {
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getSex() == 0 && population[i].getAge() > 1){MaleValue.push_back(population[i].getEBV()); male++;}
                            if(population[i].getSex() == 1 && population[i].getAge() > 1){FemaleValue.push_back(population[i].getEBV()); female++;}
                        }
                    }
                    /* Array with correct value based on how selection is to proceed created now sort */
                    if(SelectionDir == "low" || Culling == "random")                       /* sort lowest to highest */
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
                    if(SelectionDir == "high" && Culling != "random")                      /* sort highest to lowest */
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
                    logfile << "       - Number Male Parents prior to culling: " << male << "." <<  endl;
                    logfile << "       - Number Female Parents prior to culling: " << female << "." << endl;
                    int malepos;
                    int femalepos;
                    /* add 0.5 due to rounding errors accumulating and sometimes will give one less i.e. if put replacement rate as 0.8 */
                    if(male > (SIRES * (1 - SireReplacement))){malepos = SIRES * (1 - SireReplacement) + 0.5;}  /* Grabs Position based on percentile in Males */
                    if(male <= (SIRES * (1 - SireReplacement))){malepos = male;}                                /* Keep all animals and keeps more progeny */
                    if(female > (DAMS * (1 - DamReplacement))){femalepos = DAMS * (1 - DamReplacement) + 0.5;}    /* Grabs Position based on percentile in Females */
                    if(female <= (DAMS * (1 - DamReplacement))){femalepos = female;}                        /* Keep all animmals and keeps more progeny */
                    MaleCutOff =  MaleValue[malepos - 1];          /* Uniform Value cutoff for males (anything greater then remove) */
                    FemaleCutOff = FemaleValue[femalepos - 1];     /* Uniform Value cutoff for females (anything greater then remove) */
                    /* Remove Culled Animals from class object and then resize vector from population */
                    male = 0;
                    female = 0;
                    ROWS = population.size();                           /* Current Size of Population Class */
                    i = 0;
                    string Action;                                          /* Based on Selection used will result in an action that is common to all */
                    
                    
                    while(i < ROWS)
                    {
                        while(1)
                        {
                            if(Culling == "random")
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getRndCulling()>MaleCutOff){Action = "RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getRndCulling()<=MaleCutOff){Action = "KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getRndCulling()>FemaleCutOff){Action = "RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getRndCulling()<=FemaleCutOff){Action = "KP_Female";break;}
                                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
                            }
                            if(Culling == "phenotype" && SelectionDir == "low") /* remove high animals */
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()>MaleCutOff){Action = "RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()<=MaleCutOff){Action = "KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()>FemaleCutOff){Action = "RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()<=FemaleCutOff){Action = "KP_Female";break;}
                                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
                            }
                            if(Culling == "phenotype" && SelectionDir == "high")
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()<MaleCutOff){Action = "RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getPhenotype()>=MaleCutOff){Action = "KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()<FemaleCutOff){Action = "RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getPhenotype()>=FemaleCutOff){Action = "KP_Female";break;}
                                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
                            }
                            if(Culling == "true_bv" && SelectionDir == "low")
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()>MaleCutOff){Action="RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()<=MaleCutOff){Action="KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()>FemaleCutOff){Action="RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()<=FemaleCutOff){Action="KP_Female";break;}
                                if(population[i].getAge() == 1){Action = "Young_Animal"; break;}
                            }
                            if(Culling == "true_bv" && SelectionDir == "high")
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()<MaleCutOff){Action="RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getGenotypicValue()>=MaleCutOff){Action="KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()<FemaleCutOff){Action="RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getGenotypicValue()>=FemaleCutOff){Action="KP_Female";break;}
                                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
                            }
                            if(Culling == "ebv" && SelectionDir == "low")
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()>MaleCutOff){Action="RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()<=MaleCutOff){Action="KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()>FemaleCutOff){Action="RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()<=FemaleCutOff){Action="KP_Female";break;}
                                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
                            }
                            if(Culling == "ebv" && SelectionDir == "high")
                            {
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()<MaleCutOff){Action="RM_Male";break;}
                                if(population[i].getSex()==0 && population[i].getAge()>1 && population[i].getEBV()>=MaleCutOff){Action="KP_Male";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()<FemaleCutOff){Action="RM_Female";break;}
                                if(population[i].getSex()==1 && population[i].getAge()>1 && population[i].getEBV()>=FemaleCutOff){Action="KP_Female";break;}
                                if(population[i].getAge()==1){Action = "Young_Animal"; break;}
                            }
                        }
                        if(Action == "RM_Male")
                        {
                            /* Output info into master file with everything in it */
                            outputstring<<population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" "<< population[i].getSex() << " ";
                            outputstring<<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                            outputstring<<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" ";
                            outputstring<< population[i].getHap1_F() <<" ";
                            outputstring<<population[i].getHap2_F() <<" "<< population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                            outputstring<<population[i].getunfavheterolethal()<<" "<<population[i].getunfavhomosublethal()<<" ";
                            outputstring<<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                            outputstring<<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" ";
                            outputstring<<population[i].getAcc()<<" ";
                            outputstring<<population[i].getGenotypicValue()<<" ";
                            outputstring<<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                            if(OutputGeno == "yes" && Gen >= outputgeneration)
                            {
                                outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                            }
                            population.erase(population.begin()+i);     /* Delete Animal from population */
                            ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                        }
                        if(Action == "KP_Male"){male++; i++;}
                        if(Action == "RM_Female")
                        {
                            /* Output info into master file with everything in it */
                            outputstring<<population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" "<< population[i].getSex() << " ";
                            outputstring<<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                            outputstring<<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" ";
                            outputstring<< population[i].getHap1_F() <<" ";
                            outputstring<<population[i].getHap2_F() <<" "<< population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                            outputstring<<population[i].getunfavheterolethal()<<" "<<population[i].getunfavhomosublethal()<<" ";
                            outputstring<<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                            outputstring<<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" ";
                            outputstring<<population[i].getAcc()<<" ";
                            outputstring<<population[i].getGenotypicValue()<<" ";
                            outputstring<<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                            if(OutputGeno == "yes" && Gen >= outputgeneration)
                            {
                                outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl; outputnum++;
                            }
                            population.erase(population.begin()+i);     /* Delete Animal from population */
                            ROWS = ROWS -1;                             /* Reduce size of population so i stays the same */
                        }
                        if(Action == "KP_Female"){female++; i++;}
                        if(Action == "Young_Animal"){i++;}
                    }
                    /* output master df file */
                    output3 << outputstring.str(); outputstring.str("");  outputstring.clear();
                    /* output master geno file */
                    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
                    /* Need to give old animals that were kept a new culling deviate because if they randomly get a small one will stick around */
                    if(Culling == "random")
                    {
                        std::uniform_real_distribution<double> distribution5(0,1);
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getAge() > 1)
                            {
                                double temp = distribution5(gen);
                                population[i].UpdateRndCulling(temp);
                            }
                        }
                    }
                    logfile << "       - Number Male parents after culling: " << male << "." <<  endl;
                    logfile << "       - Number Female parents after culling: " << female << "." << endl;
                    logfile << "       - Size of population after culling: " << population.size() << endl;
                    time_t end_block5 = time(0);
                    logfile << "   Finished " << Culling << " culling of parents (Time: " << difftime(end_block5,start_block5) << " seconds)." << endl << endl;
                }
                /* Calculated expected heterozygosity in progeny */
                double expectedhet = 0.0; vector < string > population_marker;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1){population_marker.push_back(population[i].getMarker());}
                }
                double* tempfreqexphet = new double[population_marker[0].size()];               /* Array that holds SNP frequencies that were declared as Markers*/
                frequency_calc(population_marker, tempfreqexphet);                              /* Function to calculate snp frequency */
                for(int i = 0; i < population_marker[0].size(); i++)
                {
                    expectedhet += (1 - ((tempfreqexphet[i]*tempfreqexphet[i]) + ((1-tempfreqexphet[i])*(1-tempfreqexphet[i]))));
                }
                expectedhet /= double(population_marker[0].size());
                population_marker.clear(); delete [] tempfreqexphet;
                ExpectedHeter[Gen] = expectedhet;
                logfile << "   Expected Heterozygosity in progeny calculated: " << ExpectedHeter[Gen] << "." << endl << endl;
                /* Calculate Frequencies */
                vector < string > genfreqgeno;
                for(int i = 0; i < population.size(); i++){genfreqgeno.push_back(population[i].getQTL());}
                double* tempgenfreq = new double[genfreqgeno[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
                frequency_calc(genfreqgeno, tempgenfreq);                              /* Function to calculate snp frequency */
                /* Calculate additive and dominance variance based on allele frequencies in the current generation */
                double tempva = 0.0;
                double tempvd = 0.0;
                for(int i = 0; i < (population[0].getQTL()).size(); i++)
                {
                    if(QTL_Type[i] == 2 || QTL_Type[i] == 3)
                    {
                        tempva += 2*tempgenfreq[i]*(1-tempgenfreq[i])*((QTL_Add_Quan[i]+(QTL_Dom_Quan[i]*((1-tempgenfreq[i])-tempgenfreq[i])))*(QTL_Add_Quan[i]+(QTL_Dom_Quan[i]*((1-tempgenfreq[i])-tempgenfreq[i]))));
                        tempvd += (2 * tempgenfreq[i] * (1-tempgenfreq[i]) * QTL_Dom_Quan[i]) * (2 * tempgenfreq[i] * (1-tempgenfreq[i]) * QTL_Dom_Quan[i]);
                    }
                }
                AdditiveVar[Gen] = tempva;
                DominanceVar[Gen] = tempvd;
                /* update frequency */
                for(int j = 0; j < (population[0].getQTL()).size(); j++)                                    /* Loops across old QTL until map position matches up */
                {
                    string currentType;
                    if(QTL_Type[j] == 2)
                    {
                        currentType = "2"; int i = 0;
                        while(1)
                        {
                            double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                            string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                            if(previousLoc == QTL_MapPosition[j] && previousType == currentType)
                            {
                                stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                                string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                            }
                            i++;
                        }
                    }
                    if(QTL_Type[j] == 4)
                    {
                        currentType = "4"; int i = 0;
                        while(1)
                        {
                            double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                            string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                            if(previousLoc == QTL_MapPosition[j] && previousType == currentType)
                            {
                                stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                                string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                            }
                            i++;
                        }
                    }
                    if(QTL_Type[j] == 5)
                    {
                        currentType = "5"; int i = 0;
                        while(1)
                        {
                            double previousLoc = population_QTL[i].getLocation();           /* matches up and append current frequency to previous ones */
                            string previousType = population_QTL[i].getType();              /* grabs type which should mat up with current type */
                            if(previousLoc == QTL_MapPosition[j] && previousType == currentType)
                            {
                                stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                                string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                            }
                            i++;
                        }
                    }
                    if(QTL_Type[j] == 3)
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
                                if(previousLoc == QTL_MapPosition[j] && previousType == currentType)
                                {
                                    stringstream strStreamcurrFreq (stringstream::in | stringstream::out); strStreamcurrFreq << tempgenfreq[j];
                                    string currentFreq = strStreamcurrFreq.str(); population_QTL[i].UpdateFreq(currentFreq); break;
                                }
                                i++;
                            }
                        }
                    }
                }
                delete [] tempgenfreq;
                logfile << "   QTL frequency updated." << endl << endl;
                /* Add New Mutations to population_QTL */
                /* Save to a file that will copy over old one */
                ofstream output10;
                output10.open (qtl_class_object);
                output10 << "Location Additive_Selective Dominance Type Gen Freq" << endl;
                for(int i = 0; i < population_QTL.size(); i++)
                {
                    output10 << population_QTL[i].getLocation() << " " << population_QTL[i].getAdditiveEffect() << " ";
                    output10 << population_QTL[i].getDominanceEffect() << " " << population_QTL[i].getType() << " ";
                    output10 << population_QTL[i].getGenOccured() << " " << population_QTL[i].getFreq() << endl;
                }
                output10.close();
                if(Create_LD_Decay == "yes")
                {
                    /* Vector of string of markers */
                    vector < string > markergenotypes;
                    for(int i = 0; i < population.size(); i++){markergenotypes.push_back(population[i].getMarker());}
                    ld_decay_estimator(LD_Decay_File,Marker_Map,"no",markergenotypes);      /* Function to calculate ld decay */
                    markergenotypes.clear();
                }
                time_t intend_time = time(0);
                cout << "   - Finished Generation " << Gen << " (Took: ";
                cout << difftime(intend_time,intbegin_time) << " seconds)" << endl;
            }
            cout << "Finished Simulating Generations" << endl;
            /* Clear old Simulation Files */
            fstream checkmasterdataframe; checkmasterdataframe.open(Master_DataFrame_path, std::fstream::out | std::fstream::trunc); checkmasterdataframe.close();
            /* if on last generation output all population but before update Inbreeding values and Calculate EBV's
             /* Output animals that are of age 1 into pheno_pedigree to use for pedigree relationship */
            /* That way when you read them back in to create relationship matrix don't need to order them */
            TotalOldAnimalNumber = TotalAnimalNumber;          /* Size of old animal matrix */
            using Eigen::MatrixXd; using Eigen::VectorXd;
            VectorXd lambda(1);                                                 /* Declare scalar vector for alpha */
            lambda(0) = (1 - Variance_Additiveh2) / Variance_Additiveh2;        /* Shrinkage Factor for MME */
            vector < double > Phenotype;                                        /* Vector of phenotypes */
            stringstream outputstringpedigree(stringstream::out);
            stringstream outputstringgenomic(stringstream::out); int outputnum = 0;
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getAge() == 1)
                {
                    /* For pedigree */
                    outputstringpedigree << population[i].getID() << " " << population[i].getSire() << " " << population[i].getDam() << " ";
                    outputstringpedigree << population[i].getPhenotype() << endl;
                    /* For Genomic */
                    outputstringgenomic << population[i].getID() << " " << population[i].getPhenotype() << " " << population[i].getMarker() << " ";
                    outputstringgenomic << population[i].getPatHapl() << " " << population[i].getMatHapl() <<endl;
                    TotalAnimalNumber++;                                /* to keep track of number of animals */
                    outputnum++;
                }
                if(outputnum % 1000 == 0)
                {
                    /* output pheno pedigree file */
                    std::ofstream output1(Pheno_Pedigree_File, std::ios_base::app | std::ios_base::out);
                    output1 << outputstringpedigree.str(); outputstringpedigree.str(""); outputstringpedigree.clear();
                    /* output pheno genomic file */
                    std::ofstream output2(Pheno_GMatrix_File, std::ios_base::app | std::ios_base::out);
                    output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
                }
            }
            /* output pheno pedigree file */
            std::ofstream output1(Pheno_Pedigree_File, std::ios_base::app | std::ios_base::out);
            output1 << outputstringpedigree.str(); outputstringpedigree.str(""); outputstringpedigree.clear();
            /* output pheno genomic file */
            std::ofstream output2(Pheno_GMatrix_File, std::ios_base::app | std::ios_base::out);
            output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
            clock_t start;
            clock_t end;
            vector < double > estimatedsolutions((TotalAnimalNumber + 1),0);                                            /* Initialize to zero */
            if(scalefactaddh2 != 0)
            {
                time_t start_block = time(0);
                cout << "Estimate Final Breeding Values For Last Generation" << endl;
                logfile << "Generate Estimated Breeding Values based on " << EBV_Calc << " information for last generation:" << endl;
                logfile << "   - Size of Relationship Matrix : " << TotalAnimalNumber << " X " << TotalAnimalNumber << " ." << endl;
                vector < int > animal(TotalAnimalNumber,0);
                MatrixXd Relationship(TotalAnimalNumber,TotalAnimalNumber);
                MatrixXd Relationshipinv(TotalAnimalNumber,TotalAnimalNumber);
                if(EBV_Calc == "h1" || EBV_Calc == "h2" || EBV_Calc == "ROH" || EBV_Calc == "genomic")
                {
                    if(EBV_Calc == "h1" || EBV_Calc == "h2" || EBV_Calc == "ROH")
                    {
                        logfile << "           - Begin Constructing Genomic Relationship Matrix." << endl;
                        /* Initialize Relationship Matrix as 0.0 */
                        for(int ind1 = 0; ind1 < TotalAnimalNumber; ind1++)
                        {
                            for(int ind2 = 0; ind2 < TotalAnimalNumber; ind2++){Relationship(ind1,ind2) = 0.0;}
                        }
                        /* Before you start to make h_matrix for each haplotype first create a 2-dimensional vector with haplotype id */
                        /* This way you don't have to repeat this step for each haplotype */
                        start = time(0);
                        vector < vector < int > > PaternalHaplotypeIDs;
                        vector < vector < int > > MaternalHaplotypeIDs;
                        /* read in all animals haplotype ID's; Don't need to really worry about this getting big */
                        int linenumber = 0;
                        string line;
                        ifstream infile2;
                        infile2.open(Pheno_GMatrix_File);
                        if(infile2.fail()){cout << "Error Opening File To Create Genomic Relationship Matrix!\n"; exit (EXIT_FAILURE);}
                        while (getline(infile2,line))
                        {
                            size_t pos = line.find(" ", 0); animal[linenumber] = (std::stoi(line.substr(0,pos))); line.erase(0, pos + 1);
                            pos = line.find(" ",0); Phenotype.push_back(std::stod(line.substr(0,pos))); line.erase(0,pos + 1);
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
                        /* Animal Haplotype ID's have been put in 2-D vector to grab */
                        MatrixXd OldRelationship(TotalOldAnimalNumber,TotalOldAnimalNumber);        /* Used to store old animal G */
                        Eigen::read_binary(BinaryG_Matrix_File.c_str(),OldRelationship);            /* Read in old G */
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
                                    if(EBV_Calc == "h1")
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
                                    if(EBV_Calc == "h2")
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
                                    if(EBV_Calc == "ROH")
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
                                        if(i == haplib.size() - 1)
                                        {
                                            VectorXd den(1);                                                 /* Scale Relationship Matrix */
                                            den(0) = haplib.size();
                                            Relationship(ind1,ind2) = Relationship(ind1,ind2) / den(0);
                                            if(ind1 == ind2){Relationship(ind2,ind1) = Relationship(ind2,ind1) + 1e-5;}
                                            if(ind1 != ind2){Relationship(ind2,ind1) = Relationship(ind2,ind1) / den(0);}
                                        }
                                    }
                                } /* Finish loop across ind2 */
                            } /* Finish loop across ind1 */
                        } /* Loop across haplotypes */
                        end = time(0);
                        logfile << "           - Finished constructing Genomic Relationship Matrix. " << endl;
                        logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                        Eigen::write_binary(BinaryG_Matrix_File.c_str(),Relationship);  /* Output Relationship Matrix into Binary */
                    }
                    if(EBV_Calc == "genomic")
                    {
                        logfile << "           - Begin Constructing Genomic Relationship Matrix." << endl;
                        start = time(0);
                        MatrixXd OldRelationship(TotalOldAnimalNumber,TotalOldAnimalNumber);
                        Eigen::read_binary(BinaryG_Matrix_File.c_str(),OldRelationship);
                        /* Fill Relationship matrix with previously calcuated cells */
                        for(int i = 0; i < TotalOldAnimalNumber; i++)
                        {
                            for(int j= 0; j < TotalOldAnimalNumber; j++){Relationship(i,j) = OldRelationship(i,j);}
                        }
                        OldRelationship.resize(0,0);
                        vector < string > newanimalgeno;
                        for(int ind = 0; ind < population.size(); ind++)
                        {
                            if(population[ind].getAge() == 1){newanimalgeno.push_back(population[ind].getMarker());}
                        }
                        /* Figure out number of new animals */
                        int newanimals = 0;
                        for(int i = 0; i < population.size(); i++){if(population[i].getAge() == 1){newanimals++;}}
                        for(int i = 0; i < (TotalOldAnimalNumber+newanimals); i++){Phenotype.push_back(0);}
                        double *_grm_mkl_12 = new double[TotalOldAnimalNumber*newanimals];      /* Allocate Memory for G12 */
                        double *_grm_mkl_22 = new double[newanimals*newanimals];                /* Allocate Memory for G22 */
                        grm_prevgrm(M,Pheno_GMatrix_File,newanimalgeno,_grm_mkl_12,_grm_mkl_22,scale,animal,Phenotype);
                        newanimalgeno.clear();
                        /* Fill G12 into eigen relationship matrix */
                        for(int i = 0; i < TotalOldAnimalNumber; i++)
                        {
                            int startofinner = 0;
                            for(int j = TotalOldAnimalNumber; j < (TotalOldAnimalNumber+newanimals); j++)
                            {
                                Relationship(i,j) = Relationship(j,i) = _grm_mkl_12[(i*newanimals)+startofinner]; startofinner++;
                            }
                        }
                        delete [] _grm_mkl_12;
                        /* Fill G22 into eigen relationship matrix */
                        int newanimalLocation_i = TotalOldAnimalNumber;
                        for(int i = 0; i < newanimals; i++)
                        {
                            int newanimalLocation_j = TotalOldAnimalNumber;
                            for(int j = 0; j < newanimals; j++)
                            {
                                Relationship(newanimalLocation_i,newanimalLocation_j) = _grm_mkl_22[(i*newanimals)+j];
                                if(i == j){Relationship(newanimalLocation_i,newanimalLocation_j) = Relationship(newanimalLocation_i,newanimalLocation_j) + 1e-5;}
                                newanimalLocation_j++;
                            }
                            newanimalLocation_i++;
                        }
                        delete [] _grm_mkl_22;
                        end = time(0);
                        logfile << "           - Finished constructing Genomic Relationship Matrix. " << endl;
                        logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                        Eigen::write_binary(BinaryG_Matrix_File.c_str(),Relationship);  /* Output Genomic Relationship Matrix into Binary for */
                    }
                    logfile << "           - Begin Constructing Genomic Relationship Inverse (Using " << Geno_Inverse << ")." << endl;
                    start = time(0);
                    if(Geno_Inverse == "recursion")
                    {
                        // Need to fill mg, pg and Relationshipinv first with old animals
                        MatrixXd Old_m(TotalOldAnimalNumber,1);                                 /* Used to store old animals for matrix m */
                        MatrixXd Old_p(TotalOldAnimalNumber,TotalOldAnimalNumber);              /* Used to store old animals for matrix p */
                        MatrixXd Old_Ginv(TotalOldAnimalNumber,TotalOldAnimalNumber);           /* Used to store old animals for matrix Ginv */
                        // Matrices for current generation
                        MatrixXd mg(TotalAnimalNumber,1);                                       /* m vector in Misztal et al. (2014) */
                        MatrixXd pg(TotalAnimalNumber,TotalAnimalNumber);                       /* p matrix in Misztal et al. (2014) */
                        /* Set matrices to zero */
                        Relationshipinv = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                        pg = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                        for(int i = 0; i < TotalAnimalNumber; i++){Relationshipinv(i,i) = 0; pg(i,i) = 0; mg(i,0) = 0;}
                        /* Fill old matrices */
                        Eigen::read_binary(Binarym_Matrix_File.c_str(),Old_m); Eigen::read_binary(Binaryp_Matrix_File.c_str(),Old_p);
                        Eigen::read_binary(BinaryGinv_Matrix_File.c_str(),Old_Ginv);
                        /* Fill full matrices with already computed animals */
                        for(int i = 0; i < TotalOldAnimalNumber; i++)
                        {
                            for(int j = 0; j < TotalOldAnimalNumber; j++){pg(i,j) = Old_p(i,j); Relationshipinv(i,j) = Old_Ginv(i,j);}
                            mg(i,0) = Old_m(i,0);
                        }
                        logfile << "               - Filled Inverse Relationship Matrix with old animals." << endl;
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
                        logfile << "               - Filled Inverse Relationship Matrix for new animals." << endl;
                    }
                    if(Geno_Inverse == "cholesky")
                    {
                        MatrixXd LINV_Old(TotalOldAnimalNumber,TotalOldAnimalNumber);               /* Used to store old animals for Linv */
                        MatrixXd GINV_Old(TotalOldAnimalNumber,TotalOldAnimalNumber);               /* Used to store old animals for Ginv */
                        Eigen::read_binary(BinaryLinv_Matrix_File.c_str(),LINV_Old); Eigen::read_binary(BinaryGinv_Matrix_File.c_str(),GINV_Old);
                        /* Parameters that are used for mkl functions */
                        unsigned long i_p = 0, j_p = 0;
                        const long long int newanm = (TotalAnimalNumber - TotalOldAnimalNumber);    /* Number of new animals */
                        const long long int oldanm = TotalOldAnimalNumber;                          /* Number of old animals */
                        const long long int length = int(newanm) * int(newanm);
                        const double beta = double(-1.0); const double alpha = double(1.0); const long long int increment = int(1);
                        long long int info = 0; const long long int int_n =(int)newanm; const char diag = 'N'; char lower='L';
                        /* Create Linv to save for next generation */
                        MatrixXd LINV(TotalAnimalNumber,TotalAnimalNumber);                 /* need to save for next generation */
                        LINV = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                        for(i_p = 0; i_p < (newanm + oldanm); i_p++){LINV(i_p,i_p) = 0.0;}
                        double *G11inv = new double[oldanm*oldanm];                             /* old G-inv */
                        #pragma omp parallel for private(j_p)
                        for(i_p=0; i_p < oldanm; i_p++)
                        {
                            for(j_p=0; j_p < oldanm; j_p++){G11inv[(i_p*oldanm)+j_p] = GINV_Old(i_p,j_p);}
                        }
                        GINV_Old.resize(0,0);
                        double *L11inv = new double[oldanm*oldanm];                             /* old L-inv */
                        #pragma omp parallel for private(j_p)
                        for(i_p=0; i_p < oldanm; i_p++)
                        {
                            for(j_p=0; j_p < oldanm; j_p++){L11inv[(i_p*oldanm)+j_p] = LINV_Old(i_p,j_p);}
                        }
                        /* Save for next generation */
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
                        /* Delete Matrices */
                        delete [] G22invnew; delete [] G11inv; delete [] L11inv; delete [] G21; delete [] G22; delete [] L22inv; delete [] L21Inv;
                        LINV.resize(0,0);
                    }
                    end = time(0);
                    logfile << "           - Finished constructing Genomic Relationship Inverse." << endl;
                    logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                    /* Still need to calculate pedigree inbreeding so do it now */
                    double *f_ped = new double[TotalAnimalNumber];
                    pedigree_inbreeding(Pheno_Pedigree_File,f_ped);                /* Function that calculates inbreeding */
                    /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
                    for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
                    {
                        int j = 0;                                                                  /* Counter for population spot */
                        while(1)
                        {
                            if(population[i].getID() == animal[j]){double temp = f_ped[j]; population[i].UpdateInb(temp); break;}
                            j++;                                                                    /* Loop across until animal has been found */
                        }
                    }
                    delete[] f_ped;
                }
                if(EBV_Calc == "pedigree")
                {
                    logfile << "   - Begin Constructing A Inverse." << endl;
                    start = time(0);
                    vector < int > sire(TotalAnimalNumber,0); vector < int > dam(TotalAnimalNumber,0);
                    /*read in Pheno_Pedigree file and store in a vector to determine how many animals their are */
                    int linenumber = 0;                                                         /* Counter to determine where at in pedigree index's */
                    string line;
                    ifstream infile2;
                    infile2.open(Pheno_Pedigree_File);                                          /* This file has all animals in it */
                    if(infile2.fail()){cout << "Error Opening File To Create Pedigree Relationship Matrix\n";}
                    while (getline(infile2,line))
                    {
                        /* Fill each array with correct number already in order so don't need to order */
                        size_t pos = line.find(" ",0); animal[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);       /* Grab Animal ID */
                        pos = line.find(" ",0); sire[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                /* Grab Sire ID */
                        pos = line.find(" ",0); dam[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                 /* Grab Dam ID */
                        Phenotype.push_back(stod(line));                                                                            /* Grab Phenotype */
                        linenumber++;
                    }
                    vector < double > f_ainv(TotalAnimalNumber * TotalAnimalNumber,0);
                    vector < double > f_ped(TotalAnimalNumber,0);
                    pedigree_inverse(animal,sire,dam,f_ainv,f_ped);                    /* Function that calculates A-inverse */
                    /* Fill eigen matrix of A inverse*/
                    Relationshipinv = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);
                    for(int i = 0; i < TotalAnimalNumber; i++){Relationshipinv(i,i) = 0;}
                    for(int i = 0; i < TotalAnimalNumber; i++)
                    {
                        for(int j = 0; j < TotalAnimalNumber; j++){Relationshipinv(i,j) = f_ainv[(i*TotalAnimalNumber)+j];}
                    }
                    end = time(0);
                    logfile << "       - Finished Constructing Ainverse created.\n" << "               - Took: " << difftime(end,start) << " seconds." << endl;
                    /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
                    for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
                    {
                        int j = 0;                                                                  /* Counter for population spot */
                        while(1)
                        {
                            if(population[i].getID() == animal[j]){double temp = f_ped[j]; population[i].UpdateInb(temp); break;}
                            j++;                                                                    /* Loop across until animal has been found */
                        }
                    }
                }
                logfile << "   - Begin Solving for equations using " << Solver << " method." << endl;
                start = time(0);
                ////////////////////////
                ///    Create MME    ///
                ////////////////////////
                MatrixXd C22(TotalAnimalNumber,TotalAnimalNumber);                              /* Declare C22 */
                MatrixXd Z(TotalAnimalNumber,TotalAnimalNumber);                                /* Declare Z */
                Z = MatrixXd::Identity(TotalAnimalNumber,TotalAnimalNumber);                    /* Z is same as Z'Z (i.e. diagonals of 1's) */
                C22 = Z + (Relationshipinv * lambda(0));                                        /* C22 is Z'Z + (Ainv * lambda) */
                Relationshipinv.resize(0,0);
                Z.resize(0,0);
                ///////////////////////////////////////////////////////////
                /////    LHS and RHS for direct or pcg function       /////
                ///////////////////////////////////////////////////////////
                int LHSsize = TotalAnimalNumber + 1;
                double* LHSarray = new double[LHSsize*LHSsize];                                 /* LHS dimension: animal + intercept by animal + intercept */
                LHSarray[0] = TotalAnimalNumber;                                                /* C11 is number of observations */
                for(int i = 1; i < LHSsize; i++)
                {
                    LHSarray[(0*LHSsize)+i] = 1; LHSarray[(i*LHSsize)+0] = 1;                   /* C12 & C21 total number of phenotypes per animal (i.e. 1) */
                }
                for(int i = 0; i < TotalAnimalNumber; i++)
                {
                    for(int j=0; j < TotalAnimalNumber; j++){LHSarray[((i+1)*LHSsize)+(j+1)] = C22(i,j);}   /* Fill LHS with C22 matrix */
                }
                double* RHSarray = new double[LHSsize];                                                     /* RHS dimension: animal + intercept by 1 */
                for(int i = 0; i < LHSsize; i++){RHSarray[i] = 0;}
                for(int i = 0; i < TotalAnimalNumber; i++){RHSarray[0] += Phenotype[i];}                    /* row 1 of RHS is sum of observations */
                for(int i = 0; i < TotalAnimalNumber; i++){RHSarray[i+1] = Phenotype[i];}                   /* Copy phenotypes to RHS */
                logfile << "        - RHS created, Dimension (" << TotalAnimalNumber + 1 << " X " << 1 << ")." << endl;
                logfile << "        - LHS created, Dimension (" << TotalAnimalNumber + 1 << " X " << TotalAnimalNumber + 1 << ")." << endl;
                if(Solver == "direct")                                                          /* Solve equations using direct inversion */
                {
                    logfile << "           - Starting " << Solver << "." << endl;
                    start = time(0);
                    direct_solver(LHSarray,RHSarray,estimatedsolutions,LHSsize);
                    end = time(0);
                    logfile << "       - Finished Solving Equations created." << endl;
                    logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                }
                if(Solver == "pcg")                                                                         /* Solve equations using pcg */
                {
                    logfile << "           - Starting " << Solver << "." << endl;
                    start = time(0);
                    int* solvedatiteration = new int[1]; solvedatiteration[0] = 0;
                    pcg_solver(LHSarray,RHSarray,estimatedsolutions,LHSsize,solvedatiteration);
                    end = time(0);
                    logfile << "           - PCG converged at iteration " << solvedatiteration[0] << "." << endl;
                    logfile << "       - Finished Solving Equations created." << endl;
                    logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
                    delete[] solvedatiteration;
                }
                delete[] LHSarray; delete[] RHSarray;
                /* Update Animal Class with EBV's and associated Accuracy */
                for(int i = 0; i < population.size(); i++)
                {
                    int j = 0;                                                  /* Counter for population spot */
                    while(1)
                    {
                        if(population[i].getID() == animal[j])
                        {
                            population[i].Update_EBV(estimatedsolutions[j+1]);
                            break;
                        }
                        j++;
                    }
                }
                time_t end_block = time(0);
                logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
                animal.clear();
            }
            if(scalefactaddh2 == 0)
            {
                vector < int > animal(TotalAnimalNumber,0);                         /* Array to store Animal IDs */
                vector < int > sire(TotalAnimalNumber,0);                           /* Array to store Sire IDs */
                vector < int > dam(TotalAnimalNumber,0);                            /* Array to store Dam IDs */
                int linenumber = 0;                                                                 /* Counter to determine where at in pedigree index's */
                string line;
                ifstream infile2;
                infile2.open(Pheno_Pedigree_File);                                                  /* This file has all animals in it */
                if(infile2.fail()){cout << "Error Opening File To Make Pedigree Relationship Matrix\n"; exit (EXIT_FAILURE);}
                while (getline(infile2,line))
                {
                    /* Fill each array with correct number already in order so don't need to order */
                    size_t pos = line.find(" ",0); animal[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);       /* Grab Animal ID */
                    pos = line.find(" ",0); sire[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                /* Grab Sire ID */
                    pos = line.find(" ",0); dam[linenumber] = stoi(line.substr(0,pos)); line.erase(0, pos + 1);                 /* Grab Dam ID */
                    linenumber++;
                }
                double *f_qs_u = new double[TotalAnimalNumber];
                pedigree_inbreeding(Pheno_Pedigree_File,f_qs_u);                /* Function that calculates inbreeding */
                /* All animals of age 1 haven't had inbreeding updated so need to update real inbreeding value */
                for(int i = 0; i < population.size(); i++)                                      /* Loop across individuals in population */
                {
                    int j = 0;                                                                  /* Counter for population spot */
                    while(1)
                    {
                        if(population[i].getID() == animal[j])
                        {
                            double temp = f_qs_u[j] - 1; population[i].UpdateInb(temp); break;
                        }
                        j++;                                                                    /* Loop across until animal has been found */
                    }
                }
                delete[] f_qs_u; animal.clear(); sire.clear(); dam.clear();
            }
            cout << "Generating Master Dataframe and Summmary Statistics." << endl;
            start = time(0);
            logfile << "   Creating Master File." << endl;
            stringstream outputstring(stringstream::out);
            stringstream outputstringgeno(stringstream::out); int outputnumpart1 = 0;
            for(int i = 0; i < population.size(); i++)
            {
                /* Output info into master file with everything in it */
                outputstring<<population[i].getID() <<" "<< population[i].getSire() <<" "<< population[i].getDam() <<" "<< population[i].getSex() << " ";
                outputstring<<population[i].getGeneration() <<" "<< population[i].getAge() <<" "<< population[i].getProgeny() <<" ";
                outputstring<<population[i].getDead() <<" "<< population[i].getPed_F() <<" "<< population[i].getGen_F() <<" "<<population[i].getHap1_F() <<" ";
                outputstring<<population[i].getHap2_F() <<" "<< population[i].getHap3_F() <<" "<< population[i].getunfavhomolethal() <<" ";
                outputstring<<population[i].getunfavheterolethal()<<" "<<population[i].getunfavhomosublethal()<<" ";
                outputstring<<population[i].getunfavheterosublethal()<<" "<<population[i].getlethalequiv() <<" "<< population[i].getHomozy() <<" ";
                outputstring<<population[i].getFitness()<<" "<<population[i].getPhenotype()<<" "<<population[i].getEBV()<<" "<<population[i].getAcc()<<" ";
                outputstring<<population[i].getGenotypicValue()<<" ";
                outputstring<<population[i].getBreedingValue() <<" "<< population[i].getDominanceDeviation() <<" "<< population[i].getResidual() << endl;
                if(OutputGeno == "yes" && GENERATIONS >= outputgeneration)
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() <<" "<< endl; outputnumpart1++;
                }
                if(outputnumpart1 % 1000 == 0)
                {
                    /* output master df file */
                    std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
                    output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
                    /* output master geno file */
                    std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
                    output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
                }
            }
            /* output master df file */
            std::ofstream output3(Master_DF_File, std::ios_base::app | std::ios_base::out);
            output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
            /* output master geno file */
            std::ofstream output4(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
            output4 << outputstringgeno.str(); outputstringgeno.str(""); outputstringgeno.clear();
            population.clear();
            /* Update Master_DF_File to  */
            /* Save animal ID and generation in a 2-D vector */
            vector < int > ID_Gen;
            outputstring << "ID Sire Dam Sex Gen Age Progeny Dead Ped_F Gen_F Hap1_F Hap2_F Hap3_F Homolethal Heterlethal ";
            outputstring << "Homosublethal Hetersublethal Letequiv Homozy Fitness Phen EBV Acc GV BV DD R\n";
            int linenumbera = 0; int outputnumpart2 = 0;
            ifstream infile3;
            infile3.open(Master_DF_File);
            if(infile3.fail()){cout << "Error Opening File With Animal Information!\n"; exit (EXIT_FAILURE);}
            while (getline(infile3,line))
            {
                vector <string> lineVar;
                vector < double > temp;
                for(int i = 0; i < 27; i++)
                {
                    if(i <= 25)
                    {
                        size_t pos = line.find(" ",0);
                        lineVar.push_back(line.substr(0,pos));
                        line.erase(0, pos + 1);
                    }
                    if(i == 26){lineVar.push_back(line);}
                }
                int templine = stoi(lineVar[0]);
                ID_Gen.push_back(stoi(lineVar[4]));
                outputstring << lineVar[0] << " " <<  lineVar[1] << " " <<  lineVar[2] << " " <<  lineVar[3] << " " << lineVar[4] << " ";
                outputstring << lineVar[5] << " " <<  lineVar[6] << " " <<  lineVar[7] << " " <<  lineVar[8] << " " << lineVar[9] << " ";
                outputstring << lineVar[10] << " " << lineVar[11] << " " << lineVar[12] << " " << lineVar[13] << " " << lineVar[14] << " ";
                outputstring << lineVar[15] << " " << lineVar[16] << " " << lineVar[17] << " " << lineVar[18] << " " << lineVar[19] << " ";
                outputstring << lineVar[20] << " " << estimatedsolutions[(templine-1)] << " ";
                outputstring <<  lineVar[22] << " " << lineVar[23] << " " << lineVar[24] << " " << lineVar[25] << " " << lineVar[26] << endl;
                lineVar.clear(); outputnumpart2++;
                if(outputnumpart2 % 1000 == 0)
                {
                    /* output master df file */
                    std::ofstream output20(Master_DataFrame_path, std::ios_base::app | std::ios_base::out);
                    output20 << outputstring.str(); outputstring.str(""); outputstring.clear();
                }
            }
            std::ofstream output20(Master_DataFrame_path, std::ios_base::app | std::ios_base::out);
            output20 << outputstring.str(); outputstring.str(""); outputstring.clear();
            end = time(0);
            logfile << "   Finished Creating Master File." << endl;
            logfile << "               - Took: " << difftime(end,start) << " seconds." << endl;
            /* Generate QTL summary Stats */
            int totalgroups = GENERATIONS + 1;
            generatesummaryqtl(Pheno_GMatrix_File, qtl_class_object, Summary_QTL_path,totalgroups,ID_Gen,AdditiveVar,DominanceVar,NumDeadFitness);
            generatessummarydf(Master_DataFrame_path,Summary_DF_path,totalgroups,ExpectedHeter);
            /* Make location be chr and pos instead of in current format to make it easier for user. */
            vector <string> numbers;
            line;
            ifstream infileqtlreformat;
            infileqtlreformat.open(qtl_class_object.c_str());
            if(infileqtlreformat.fail()){cout << "Error Opening File\n";}
            while (getline(infileqtlreformat,line)){numbers.push_back(line);}
            vector < int > qtlout_chr((numbers.size()-1),0);
            vector < int > qtlout_pos((numbers.size()-1),0.0);
            vector < string > restofit((numbers.size()-1),"");
            for(int i = 1; i < numbers.size(); i++)
            {
                size_t pos =  numbers[i].find(" ",0); string temp =  numbers[i].substr(0,pos); numbers[i].erase(0, pos + 1);
                qtlout_chr[i-1] = stoi(temp.c_str());
                /* convert to nuclotides */
                qtlout_pos[i-1] = (stof(temp.c_str()) - qtlout_chr[i-1]) * ChrLength[qtlout_chr[i-1]-1];
                restofit[i-1] = numbers[i];
            }
            ofstream output10;
            output10.open (qtl_class_object);
            output10 << "Chr Pos Additive_Selective Dominance Type Gen Freq" << endl;
            for(int i = 1; i < numbers.size(); i++)
            {
                output10 << qtlout_chr[i-1] << " " << qtlout_pos[i-1] << " " << restofit[i-1] << endl;
            }
            output10.close();
            /* Remove Master_DF_File since it is already in Master_DataFrame */
            string removedf = "rm " + Master_DF_File;
            system(removedf.c_str());
            delete [] M;                                    /* Now can delete the M matrix once simualtion is done */
            delete [] founderfreq;                          /* Now can delete founder frequencies once simualtion is done */
            time_t repend_time = time(0);
            if(replicates > 1)
            {
                cout.setf(ios::fixed);
                cout << setprecision(2) << endl << "Replicates " << reps + 1 << " has completed normally (Took: ";
                cout << difftime(repend_time,repbegin_time) / 60 << " minutes)" << endl << endl;
                cout.unsetf(ios::fixed);
            }
        }
        /* If you have multiple replicates create a new directory within this folder to store them and just attach seed afterwards */
        if(replicates > 1)
        {
            if(reps == 0)
            {
                /* First delete replicates folder if exists */
                string systemcall = "rm -rf " + path + "/" + outputfolder + "/replicates || true";
                system(systemcall.c_str());
                systemcall = "mkdir " + path + "/" + outputfolder + "/replicates";
                system(systemcall.c_str());
            }
            /* make seednumber a string */
            stringstream stringseed; stringseed << seednumber; string stringseednumber = stringseed.str();
            string systemcall = "mv " + logfileloc + " " + path + "/" + outputfolder + "/replicates/" + "log_file_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + lowfitnesspath + " " + path + "/" + outputfolder + "/replicates/" + "Low_Fitness_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + Master_DataFrame_path + " " + path + "/" + outputfolder + "/replicates/" + "Master_DataFrame_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + Master_Genotype_File + " " + path + "/" + outputfolder + "/replicates/" + "Master_Genotypes_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + qtl_class_object + " " + path + "/" + outputfolder + "/replicates/" + "QTL_new_old_Class_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + Marker_Map + " " + path + "/" + outputfolder + "/replicates/" + "Marker_Map_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + Summary_QTL_path + " " + path + "/" + outputfolder + "/replicates/" + "Summary_Statistics_QTL_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + Summary_DF_path + "_Inbreeding " + path + "/" + outputfolder + "/replicates/" + "Summary_Statistics_DataFrame_Inbreeding_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv " + Summary_DF_path + "_Performance " + path + "/" + outputfolder + "/replicates/" + "Summary_Statistics_DataFrame_Performance_" + stringseednumber;
            system(systemcall.c_str());
        }
    }
    free(cwd);
    mkl_free_buffers();
    mkl_thread_free_buffers();
    time_t fullend_time = time(0);
    cout.setf(ios::fixed);
    cout << setprecision(2) << "Simulation has completed normally (Took: " << difftime(fullend_time,fullbegin_time) / 60 << " minutes)" << endl << endl;
    cout.unsetf(ios::fixed);
}
