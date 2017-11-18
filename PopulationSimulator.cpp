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
#include <mkl.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
#include <tuple>

#include "Animal.h"
#include "HaplofinderClasses.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"
#include "Genome_ROH.h"

using namespace std;

/**********************************************/
/* Functions from EBV_Functions.cpp           */
/**********************************************/
/* setup either blup or marker effects prediction of ebv */
void Generate_BLUP_EBV(parameters &SimParameters,vector <Animal> &population, vector <double> &estimatedsolutions, vector <double> &trueaccuracy,ostream& logfileloc,vector <int> &trainanimals,int TotalAnimalNumber, int TotalOldAnimalNumber, int Gen,string Pheno_Pedigree_File,string GenotypeStatus_path,string Pheno_GMatrix_File,double* M, float scale,string BinaryG_Matrix_File,string Binarym_Matrix_File, string Binaryp_Matrix_File, string BinaryGinv_Matrix_File,string BinaryLinv_Matrix_File,vector < hapLibrary > &haplib);
void bayesianestimates(parameters &SimParameters,vector <Animal> &population,string Master_DF_File, string Pheno_GMatrix_File, string Pheno_Pedigree_File,int Gen ,vector <double> &estimatedsolutions,string Bayes_MCMC_Samples,string Bayes_PosteriorMeans, ostream& logfileloc);
/* summary stats functions */
void getamax (parameters &SimParameters,vector <Animal> &population,string Master_DF_File,string Pheno_Pedigree_File,string Amax_Output);
void trainrefcor(parameters &SimParameters,vector <Animal> &population,string Correlation_Output, int Gen);
void WindowVariance(parameters &SimParameters,vector <Animal> &population,vector < QTL_new_old > &population_QTL,string foundergen ,string Windowadditive_Output, string Windowdominance_Output);
void Inbreeding_Pedigree(vector <Animal> &population,string Pheno_Pedigree_File);
/* ssgblup genotype status functions */
void updateanimalgenostatus(parameters &SimParameters,vector <Animal> &population,string GenotypeStatus_path);
void breedingpopulationgenotyped(vector <Animal> &population, ostream& logfileloc);
int getTotalGenotypeCount(string GenotypeStatus_path);

/**********************************************/
/* Functions from ParameterClass.cpp          */
/**********************************************/
void read_generate_parameters(parameters &SimParameters, string parameterfile, string &logfilestring, string &logfilestringa);

/**********************************************/
/* Functions from Genome_ROH.cpp              */
/**********************************************/
void Genome_ROH_Summary(parameters &SimParameters, vector <Animal> &population,string Marker_Map, string Summary_ROHGenome_Length, string Summary_ROHGenome_Freq, int Gen, ostream& logfileloc);
void Proportion_ROH(parameters &SimParameters, vector <Animal> &population,string Marker_Map, ostream& logfileloc);
void ld_decay_estimator(string outputfile, string mapfile, string lineone, vector < string > const &genotypes);
void qtlld_decay_estimator(parameters &SimParameters, vector <Animal> &population, vector < QTL_new_old > &population_QTL,string Marker_Map,string foundergen, string QTL_LD_Decay_File, string Phase_Persistance_File, string Phase_Persistance_Outfile);

/**********************************************/
/* Functions from HaplofinderClasses.cpp      */
/**********************************************/
void ReadGenoDiverMapFile_Index(string Marker_Map, vector < int > &chr, vector < int > &position, vector < int > &index, vector < CHR_Index> &chr_index,ostream& logfileloc);
void ReadGenoDiverPhenoGenoFile(vector <Animal> &population, string Master_DF_File, string Pheno_GMatrix_File,vector <int> const &traingeneration,vector <string> &id, vector <double> &pheno, vector <double> &trueebv, vector<int> &phenogenorownumber, vector <string> &genotype, vector <string> &genotypeID);
void simulationlambda(vector <double> const &pheno, vector <double> const &trueebv, vector <double> &lambda);
void subtractmean(vector <double> &pheno);
void GenerateAinvGenoDiver(string pedigreefile,vector < string > const &uniqueID,vector < string > const &id, double* Relationshipinv_mkl);
void GenerateLHSRed(vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, double* Relationshipinv_mkl, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A, vector <string> id, vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A, int dim_lhs, vector <double> const &lambda);
void updateLHSinv(vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector <string> uniqueID, vector <int> const &sub_genotype,float * LHSinvupdated,int dim_lhs, int upddim_lhs, float * solutions, vector < double > const &pheno);
void estimateROHeffect(float * LHSinvupdated,float * solutions,int upddim_lhs,vector < string > const &factor_red, vector < int > const &zero_columns_red, vector <double> &LSM, vector <double> &T_stat, double resvar,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass);
double phenocutoff(vector <CHR_Index> chr_index,int null_samples,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc,ostream& logfile);
void Step1(vector < Unfavorable_Regions_sub > regions_sub, double phenotype_cutoff, string unfav_direc, int chromo, vector <CHR_Index> chr_index,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno, vector <string> const &id,vector<int> const &chr, vector<int> const &position, vector<int> const &index,vector < Unfavorable_Regions > &regions);
void Step2(vector < Unfavorable_Regions > &regions,int min_Phenotypes,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc,double one_sided_t,double phenotype_cutoff,ostream& logfile);
void Step3(vector < Unfavorable_Regions > &regions,vector <double> const &pheno,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector < string > const &id);
void replaceAll(string& str, const string& from, const string& to);
void calculate_IIL(vector <double> &inbreedingload, vector <string> const &progenygenotype,vector <Unfavorable_Regions> &trainregions,string unfav_direc);
bool sortByLength(const Unfavorable_Regions &alength, const Unfavorable_Regions &blength){return alength.Length_R > blength.Length_R;}
void calculatecorrelation(vector <double> &inbreedingload, vector <double> &progenyphenotype,vector <double> &progenytgv, vector <double> &progenytbv, vector <double> &progenytdd, vector < double > &outcorrelations);

/******************************************/
/* Functions from MatingDesignClasses.cpp */
/******************************************/
void agedistribution(vector <Animal> &population,vector <int> &MF_AgeClass, vector <int> &M_AgeClass, vector <int> &MF_AgeID);
void betadistributionmates(vector <Animal> &population,parameters &SimParameters, int M_NumberClassg0, vector <double> const &number, vector <int> const &M_AgeClass,vector <int> const &MF_AgeID);
void outputlogsummary(vector <Animal> &population, vector<int> &M_AgeClass, vector<int> &MF_AgeID,ostream& logfileloc);
string choosematingscenario(parameters &SimParameters, string tempselectionvector);
void generatematingpairs(vector <MatingClass> &matingindividuals, vector <Animal> &population, vector < hapLibrary > &haplib,parameters &SimParameters, string matingscenario, string Pheno_Pedigree_File, double* M,float scale, ostream& logfileloc);
void indexmatingdesign(vector <MatingClass> &matingindividuals, vector <Animal> &population, vector < hapLibrary > &haplib, parameters &SimParameters, string Pheno_Pedigree_File, double* M,float scale,ostream& logfileloc);
void updatematingindex(vector <MatingClass> &matingindividuals, vector <Animal> &population);

/************************************************/
/* Functions from SelectionCullingFunctions.cpp */
/************************************************/
void truncationselection(vector <Animal> &population,parameters &SimParameters,string tempselectionscen,int Gen,string Master_DF_File, string Master_Genotype_File, ostream& logfileloc);
void discretegenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,string Master_DF_File, string Master_Genotype_File, ostream& logfileloc);
void overlappinggenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,string Master_DF_File,string Master_Genotype_File,ostream& logfileloc);
void optimalcontributionselection(vector <Animal> &population,vector <MatingClass> &matingindividuals,vector < hapLibrary > &haplib,parameters &SimParameters,string tempselectionscen,string Pheno_Pedigree_File,double* M, float scale,string Master_DF_File,string Master_Genotype_File,int Gen,ostream& logfileloc);
void breedingagedistribution(vector <Animal> &population,parameters &SimParameters,ostream& logfileloc);

/*******************************************/
/* Functions from Simulation_Functions.cpp */
/*******************************************/
void pedigree_inverse(vector <int> const &f_anim, vector <int> const &f_sire, vector <int> const &f_dam, vector<double> &output,vector < double > &output_f);
void pedigree_inbreeding(string phenotypefile, double* output_f);
double lethal_pedigree_inbreeding(string phenotypefile, int tempsireid, int tempdamid);
void frequency_calc(vector < string > const &genotypes, double* output_freq);
void grm_noprevgrm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);
void grm_prevgrm(double* input_m, string genofile, vector < string > const &newgenotypes, double* output_grm12, double* output_grm22, float scaler,vector < int > &animalvector, vector < double > &phenotypevector);
void generatesummaryqtl(string inputfilehap, string inputfileqtl, string outputfile, int generations,vector < int > const &idgeneration, vector < double > const &tempaddvar, vector < double > const &tempdomvar,  vector < int > const &tempdeadfit);
void generatessummarydf(string inputfilehap, string outputfile, int generations, vector < double > const &tempexphet);
void pcg_solver(double* lhs, double* rhs, vector < double > &solutionsa, int dimen, int* solvediter);
void direct_solver(double* lhs, double* rhs, vector < double > &solutionsa, int dimen);
void pedigree_relationship(string phenotypefile, vector <int> const &parent_id, double* output_subrelationship);
void LinearProgramming(double* matingvalue_matrix, string direction, vector <int> &mate_column);
void sslr_mating(double* matingvalue_matrix, string direction, vector <int> &mate_column);
void generate2trindex(double* mate_matrix1, double* mate_matrix2, double* mate_index_matrix, vector < double > const &indexprop, int dim, vector < int > const &rowindex, vector <double> &returnweights);

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

int main(int argc, char* argv[])
{
    std::setprecision(10);
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
    parameters SimParameters; string logfilestring, logfilestringa;
    read_generate_parameters(SimParameters,paramterfile,logfilestring,logfilestringa);
    /* Need to remove all the files within GenoDiverFiles */
    if(SimParameters.getStartSim() == "sequence")
    {
        string rmfolder = "rm -rf ./" + SimParameters.getOutputFold();
        system(rmfolder.c_str());
        rmfolder = "mkdir " + SimParameters.getOutputFold();
        system(rmfolder.c_str());
    }
    if(SimParameters.getStartSim() == "founder")
    {
        std::string x = path + "/" + SimParameters.getOutputFold() + "/";
        const char *folderr = x.c_str();
        struct stat sb;
        if (stat(folderr, &sb) == 0 && S_ISDIR(sb.st_mode)){}
        else
        {
            cout << endl << " FOLDER DOESN'T EXIST. CHANGE TO 'START: sequence' TO FILL FOLDER IF USING FOUNDER AS START!\n" << endl;
        }
    }
    /* Files to drop things in */
    string logfileloc = path + "/" + SimParameters.getOutputFold() + "/log_file.txt";
    string lowfitnesspath = path + "/" + SimParameters.getOutputFold() + "/Low_Fitness";
    string snpfreqfileloc = path + "/" + SimParameters.getOutputFold() + "/SNPFreq";
    string foundergenofileloc = path + "/" + SimParameters.getOutputFold() + "/FounderGenotypes";
    string qtl_class_object = path + "/" + SimParameters.getOutputFold() + "/QTL_new_old_Class";
    string Pheno_Pedigree_File = path + "/" + SimParameters.getOutputFold() + "/Pheno_Pedigree";
    string Pheno_GMatrix_File = path + "/" + SimParameters.getOutputFold() + "/Pheno_GMatrix";
    string Master_DF_File = path + "/" + SimParameters.getOutputFold() + "/Master_DF";
    string Master_Genotype_File = path + "/" + SimParameters.getOutputFold() + "/Master_Genotypes";
    string BinaryG_Matrix_File = path + "/" + SimParameters.getOutputFold() + "/G_Matrix";
    string Binarym_Matrix_File = path + "/" + SimParameters.getOutputFold() + "/m_Matrix";
    string Binaryp_Matrix_File = path + "/" + SimParameters.getOutputFold() + "/p_Matrix";
    string BinaryLinv_Matrix_File = path + "/" + SimParameters.getOutputFold() + "/Linv_Matrix";
    string BinaryGinv_Matrix_File = path + "/" + SimParameters.getOutputFold() + "/Ginv_Matrix";
    string Marker_Map = path + "/" + SimParameters.getOutputFold() + "/Marker_Map";
    string Master_DataFrame_path = path + "/" + SimParameters.getOutputFold() + "/Master_DataFrame";
    string Summary_QTL_path = path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_QTL";
    string Summary_DF_path = path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_DataFrame";
    string GenotypeStatus_path = path + "/" + SimParameters.getOutputFold() + "/Animal_Genotype_Status";
    /* If doing LD based summary statistics will drop in these files */
    string LD_Decay_File = path + "/" + SimParameters.getOutputFold() + "/LD_Decay";
    string QTL_LD_Decay_File = path + "/" + SimParameters.getOutputFold() + "/QTL_LD_Decay";
    string Phase_Persistance_File = path + "/" + SimParameters.getOutputFold() + "/Phase_Persistance";
    string Phase_Persistance_Outfile = path + "/" + SimParameters.getOutputFold() + "/Phase_Persistance_Generation";
    /* if doing haplotype finder will drop in this file summary statistics */
    string Summary_Haplofinder = path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_Haplofinder";
    /* if doing ROH genome summary will drop in this file with summary statistics of frequency and length */
    string Summary_ROHGenome_Freq = path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_ROH_Freq";
    string Summary_ROHGenome_Length = path + "/" + SimParameters.getOutputFold() + "/Summary_Statistics_ROH_Length";
    /* if doing bayesian ebv based prediction save mcmc samples */
    string Bayes_MCMC_Samples = path + "/" + SimParameters.getOutputFold() + "/bayes_mcmc_samples";
    string Bayes_PosteriorMeans = path + "/" + SimParameters.getOutputFold() + "/bayes_posteriormeans";
    /* if want Amax by generation from most recent generation save to AmaxGeneration */
    string Amax_Output = path + "/" + SimParameters.getOutputFold() + "/AmaxGeneration";
    string Correlation_Output = path + "/" + SimParameters.getOutputFold() + "/ProgenyParentCorrelationGeneration";
    /* if want window true additive and dominance variance across genome */
    string Windowadditive_Output = path + "/" + SimParameters.getOutputFold() + "/WindowAdditiveVariance";
    string Windowdominance_Output = path + "/" + SimParameters.getOutputFold() + "/WindowDominanceVariance";
    /* Ensure removes files that aren't always created so wont' confuse users */
    string command = "rm -rf " + LD_Decay_File + " || true"; system(command.c_str());
    command = "rm -rf " + QTL_LD_Decay_File + " || true"; system(command.c_str());
    command = "rm -rf " + Phase_Persistance_File + " || true"; system(command.c_str());
    command = "rm -rf " + Phase_Persistance_Outfile + " || true"; system(command.c_str());
    command = "rm -rf " + Summary_Haplofinder + " || true"; system(command.c_str());
    command = "rm -rf " + Summary_ROHGenome_Freq + " || true"; system(command.c_str());
    command = "rm -rf " + Summary_ROHGenome_Length + " || true"; system(command.c_str());
    command = "rm -rf " + Bayes_MCMC_Samples + " || true"; system(command.c_str());
    command = "rm -rf " + Bayes_PosteriorMeans + " || true"; system(command.c_str());
    command = "rm -rf " + Amax_Output + " || true"; system(command.c_str());
    command = "rm -rf " + Correlation_Output + " || true"; system(command.c_str());
    command = "rm -rf " + Windowadditive_Output + " || true"; system(command.c_str());
    command = "rm -rf " + Windowdominance_Output + " || true"; system(command.c_str());
    command = "rm -rf " + GenotypeStatus_path + " || true"; system(command.c_str());
    /* Number of threads */
    int nthread = SimParameters.getThread();
    omp_set_num_threads(nthread);
    mkl_set_num_threads_local(nthread);
    /* loop across replicates */
    for(int reps = 0; reps < SimParameters.getReplicates(); reps++)
    {
        /* CHange when implement multiple type of genomic matrices */
        if(reps == 0)
        {
            /* Remove previous replicates folder */
            string systemcall = "rm -rf " + path + "/" + SimParameters.getOutputFold() + "/replicates || true";
            system(systemcall.c_str());
        }
        time_t repbegin_time = time(0);
        if(SimParameters.getReplicates() > 1)
        {
            cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~ Starting Replicate " << reps + 1 << " ~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        }
        /* add seed by one to generate new replicates and reprint out parameter file with updated logfile */
        if(reps > 0)
        {
            SimParameters.UpdateSeed(SimParameters.getSeed()+1); SimParameters.UpdateStartSim("founder");
            stringstream s1; s1 << SimParameters.getSeed(); string tempvar = s1.str();
        }
        /* Deletes files from Previous Simulation */
        fstream checklog; checklog.open(logfileloc, std::fstream::out | std::fstream::trunc); checklog.close();
        fstream checkmarkermap; checkmarkermap.open(Marker_Map, std::fstream::out | std::fstream::trunc); checkmarkermap.close();
        fstream checkphenoped; checkphenoped.open(Pheno_Pedigree_File, std::fstream::out | std::fstream::trunc); checkphenoped.close();
        fstream checkphenogen; checkphenogen.open(Pheno_GMatrix_File, std::fstream::out | std::fstream::trunc); checkphenogen.close();
        fstream checkmasterdf; checkmasterdf.open(Master_DF_File, std::fstream::out | std::fstream::trunc); checkmasterdf.close();
        fstream checkmastergeno; checkmastergeno.open(Master_Genotype_File, std::fstream::out | std::fstream::trunc); checkmastergeno.close();
        std::ofstream logfile(logfileloc, std::ios_base::out);               /* open log file to output verbage throughout code */
        if(SimParameters.getstartgen() != -5 && SimParameters.gettraingen() != -5)
        {
            fstream checkhaplosummary; checkhaplosummary.open(Summary_Haplofinder, std::fstream::out | std::fstream::trunc); checkhaplosummary.close();
        }
        logfile << logfilestringa << "        - Simulation Started at:\t\t\t\t\t\t\t\t\t'" << SimParameters.getStartSim() << endl;
        logfile << "        - Seed Number:\t\t\t\t\t\t\t\t\t\t\t'" << SimParameters.getSeed() << "'" << endl << logfilestring << endl;
        /* Variables that need initialized in order to ensure global inheritance */
        using Eigen::MatrixXd; using Eigen::VectorXd;
        /* checks to see if want dominance or add + dominance variance to 0 */
        double scalefactaddh2, scalefactpdomh2;
        if(SimParameters.getVarDom() == 0.0){scalefactpdomh2 = 0.0;}                        /* scaling factor for dominance set to 0 */
        if(SimParameters.getVarDom() > 0.0){scalefactpdomh2 = 1.0;}                         /* scaling factor for dominance set to 1.0 and will change */
        if(SimParameters.getVarAdd() == 0.0){scalefactaddh2 = 0.0;}                         /* scaling factor for additive set to 0 */
        if(SimParameters.getVarAdd() > 0.0){scalefactaddh2 = 1.0;}                          /* scaling factor for additive set to 1.0 and will change*/
        string SNPFiles[SimParameters.getChr()];                                            /* Stores names of SNP Files */
        string MapFiles[SimParameters.getChr()];                                            /* Stores names of MAP Files */
        int ChrSNPLength [SimParameters.getChr()];                                          /* number of SNP within each chromosome */
        /* Marker Information */
        int NUMBERMARKERS = 0;
        /* Figure out number of markers by adding up across chromosome */
        for (int i = 0; i < SimParameters.getChr(); i++){NUMBERMARKERS += (SimParameters.get_Marker_chr())[i];}
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
        int markerperChr[SimParameters.getChr()];                                                       /* Number of Markers for each chromosome */
        int qtlperChr[SimParameters.getChr()];                                                          /* Number of QTL per chromsome */
        vector < int > NumDeadFitness((SimParameters.getGener() + 1),0);                /* Number of dead due to fitness by generation */
        vector < double > AdditiveVar((SimParameters.getGener() + 1),0.0);              /* sum of 2pqa^2 */
        vector < double > DominanceVar((SimParameters.getGener() + 1),0.0);             /* sum of (2pqd)+^2 */
        vector < double > ExpectedHeter((SimParameters.getGener() + 1),0.0);            /* Expected Heterozygosity: (1 - p^2 - q^2) / markers */
        vector < QTL_new_old > population_QTL;                                          /* Hold in a vector of QTL_new_old Objects */
        vector < Animal > population;                                                   /* Hold in a vector of Animal Objects */
        vector < hapLibrary > haplib;                                                   /* Vector of haplotype library objects */
        vector < string > leftfitnessstring;                                            /* Vector that saves low fitness animal summary stats */
        vector < string > rightfitnessstring;                                           /* Vector that saves low fitness animal summary stats */
        vector < string > markerlowfitness;                                             /* Marker for low fitness animal */
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        /************************                        Unfavorable Haplotype Finder Variables                         ************************/
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        vector <Unfavorable_Regions> trainregions;                        /* vector of objects to store everything about unfavorable regions */
        /* Figure out unfavorable direction */
        string unfav_direc; int retraingeneration = (SimParameters.getstartgen() + SimParameters.getGenfoundsel());
        if(SimParameters.getSelectionDir() == "high"){unfav_direc = "low";}
        if(SimParameters.getSelectionDir() == "low"){unfav_direc = "high";}
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        /************************                                  Mating Design Variables                              ************************/
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        vector <MatingClass> matingindividuals;
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        /************************                                       Begin Simulation                                ************************/
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        const clock_t intbegin_time = clock();
        logfile << endl;
        if(SimParameters.getStartSim() == "sequence")
        {
            logfile << "============================================================\n";
            logfile << "===\t MaCS Part of Program (Chen et al. 2009) \t====\n";
            logfile << "============================================================\n";
            logfile << " Begin Generating Sequence Information: " << endl;
            if(SimParameters.getne_founder() == -5 && SimParameters.getNe_spec() != "")
            {
                if(SimParameters.getNe_spec() == "CustomNe")
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
                    float ScaledMutation = 4 * customNe * SimParameters.getu();
                    logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 50 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                           /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                      /* string for Scaled Recombination */
                    stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                       /* string for Number Haplotypes */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                        stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();         /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part2+part6+part7;
                        if(i == 0)
                        {
                            logfile << "    - Line used to call MaCS: " << endl;
                            logfile << "    " << command << endl;
                        }
                        system(command.c_str());
                        system("rm -rf haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail -n +6 file1.txt > Intermediate.txt");
                        part1 = "tail -n +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();                                  /* Convert i loop to string chromosome number */
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str()); /* Store name of genotype file */
                        system("head -n 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                        command = part1 + mapfile;  MapFiles[i] = mapfile; system(command.c_str());             /* Store name of map file */
                        part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                        part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                        command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(SimParameters.getNe_spec() == "Ne70")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 70 * SimParameters.getu();
                    logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 70 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                           /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                      /* string for Scaled Recombination */
                    stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                       /* string for Number Haplotypes */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                        stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 0.18 0.71 -eN 0.36 1.43 -eN 0.54 2.14 -eN 0.71 2.86 -eN 0.89 3.57 -eN 1.07 4.29 -eN 1.25 5.00 -eN 1.43 5.71";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part7;
                        if(i == 0)
                        {
                            logfile << "    - Line used to call MaCS: " << endl;
                            logfile << "    " << command << endl;
                        }
                        system(command.c_str());
                        system("rm -f haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail -n +6 file1.txt > Intermediate.txt");
                        part1 = "tail -n +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str()); /* Store name of genotype file */
                        system("head -n 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());              /* Store name of map file */
                        part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                        part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to Output Directory */
                        part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                        command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(SimParameters.getNe_spec() == "Ne100_Scen1")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 100 * SimParameters.getu();
                    logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 100 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                        stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 0.06 2.0 -eN 0.13 3.0 -eN 0.25 5.0 -eN 0.50 7.0 -eN 0.75 9.0 -eN 1.00 11.0 -eN 1.25 12.5 -eN 1.50 13.0 ";
                        string part6a = "-eN 1.75 13.5 -eN 2.00 14.0 -eN 2.25 14.5 -eN 2.50 15.0 -eN 5.00 20.0 -eN 7.50 25.0 -eN 10.00 30.0 -eN 12.50 35.0 ";
                        string part6b = "-eN 15.00 40.0 -eN 17.50 45.0 -eN 20.00 50.0 -eN 22.50 55.0 -eN 25.00 60.0 -eN 50.00 70.0 -eN 100.00 80.0 -eN 150.00 90.0 ";
                        string part6c = "-eN 200.00 100.0 -eN 250.00 120.0 -eN 500.00 200.0 -eN 1000.00 400.0 -eN 1500.00 600.0 -eN 2000.00 800.0 -eN 2500.00 1000.0";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part6c+part7;
                        if(i == 0)
                        {
                            logfile << "    - Line used to call MaCS: " << endl;
                            logfile << "    " << command << endl;
                        }
                        system(command.c_str());
                        system("rm -rf haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail -n +6 file1.txt > Intermediate.txt");
                        part1 = "tail -n +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                        system("head -n 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                        part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                        part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                        command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(SimParameters.getNe_spec() == "Ne100_Scen2")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 100 * SimParameters.getu();
                    logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 100 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                        stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 50.00 200.0 -eN 75.00 300.0 -eN 100.00 400.0 -eN 125.00 500.0 -eN 150.00 600.0 -eN 175.00 700.0 -eN 200.00 800.0 ";
                        string part6a = "-eN 225.00 900.0 -eN 250.00 1000.0 -eN 275.00 2000.0 -eN 300.00 3000.0 -eN 325.00 4000.0 -eN 350.00 5000.0 ";
                        string part6b = "-eN 375.00 6000.0 -eN 400.00 7000.0 -eN 425.00 8000.0 -eN 450.00 9000.0 -eN 475.00 10000.0";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part7;
                        if(i == 0)
                        {
                            logfile << "    - Line used to call MaCS: " << endl;
                            logfile << "    " << command << endl;
                        }
                        system(command.c_str());
                        system("rm -rf haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail -n +6 file1.txt > Intermediate.txt");
                        part1 = "tail -n +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                        system("head -n 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                        part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                        part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                        command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(SimParameters.getNe_spec() == "Ne250")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 250 * SimParameters.getu();
                    logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 250 * 1.0e-8; logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                        stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6 = " -eN 1 4 -eN 2 8 -eN 3 10 -eN 4 12 -eN 5 14 -eN 6 16 -eN 7 18 -eN 8 20 -eN 9 22 -eN 10 24 -eN 20 28 ";
                        string part6a = "-eN 40 32 -eN 60 36 -eN 80 40 -eN 100 48 -eN 200 80 -eN 400 160 -eN 600 240 -eN 800 320 -eN 1000 400";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part7;
                        if(i == 0)
                        {
                            logfile << "    - Line used to call MaCS: " << endl;
                            logfile << "    " << command << endl;
                        }
                        system(command.c_str());
                        system("rm -rf haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail -n +6 file1.txt > Intermediate.txt");
                        part1 = "tail -n +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                        system("head -n 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                        part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                        part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                        command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
                if(SimParameters.getNe_spec() == "Ne1000")
                {
                    /* Need to first initialize paramters for macs */
                    float ScaledMutation = 4 * 1000 * SimParameters.getu();
                    logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                    float ScaledRecombination = 4 * 1000 * 1.0e-8;
                    logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                    /* Convert Every Value to a string in order to make string */
                    stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                    stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                    stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                    /* string for Number Haplotypes */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                        stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();                   /* Convert seed number to a string */
                        /* Part 1 run the macs simulation program and output it into ms form */
                        string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                        string part6  = " -eN 0.50 2.00 -eN 0.75 2.50 -eN 1.00 3.00 -eN 1.25 3.20 -eN 1.50 3.50 -eN 1.75 3.80 -eN 2.00 4.00 -eN 2.25 4.20 ";
                        string part6a = "-eN 2.50 4.50 -eN 5.00 5.46 -eN 10.00 7.37 -eN 15.00 9.28 -eN 20.00 11.19 -eN 25.00 13.10 -eN 50.00 22.66 ";
                        string part6b = "-eN 100.00 41.77 -eN 150.00 60.89 -eN 200.00 80.00";
                        string part7 = " -h 1e2 2>debug.txt | ./msformatter > file1.txt";
                        string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6+part6a+part6b+part7;
                        if(i == 0)
                        {
                            logfile << "    - Line used to call MaCS: " << endl;
                            logfile << "    " << command << endl;
                        }
                        system(command.c_str());
                        system("rm -rf haplo* tree.* debug.txt");
                        /* Part 2 put into right format */
                        system("tail -n +6 file1.txt > Intermediate.txt");
                        part1 = "tail -n +2 Intermediate.txt > ";
                        stringstream ss; ss << (i + 1); string str = ss.str();
                        part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                        string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                        system("head -n 1 Intermediate.txt > TempMap.txt");
                        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                        command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                        part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                        part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                        part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                        system("rm Intermediate.txt file1.txt TempMap.txt");
                        /* need to move files to output Directory */
                        part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                        command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                        command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                        logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                    }
                }
            }
            if(SimParameters.getne_founder() != -5 && SimParameters.getNe_spec() == "")
            {
                /* Need to first initialize paramters for macs */
                float ScaledMutation = 4 * SimParameters.getne_founder() * SimParameters.getu();
                logfile << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
                float ScaledRecombination = 4 * SimParameters.getne_founder() * 1.0e-8;
                logfile << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
                /* Convert Every Value to a string in order to make string */
                stringstream s1; s1 << ScaledMutation; string scalmut = s1.str();                        /* string for Scaled Mutation */
                stringstream s2; s2 << ScaledRecombination; string scalrec = s2.str();                   /* string for Scaled Recombination */
                stringstream s3; s3 << SimParameters.getfnd_haplo(); string foundhap = s3.str();                    /* string for Number Haplotypes */
                for(int i = 0; i < SimParameters.getChr(); i++)
                {
                    stringstream s4; s4 << (SimParameters.get_ChrLength())[i]; string SizeChr = s4.str();   /* Convert chromosome length to a string */
                    stringstream s5; s5 << SimParameters.getSeed()+i; string macsseed = s5.str();                   /* Convert seed number to a string */
                    /* Part 1 run the macs simulation program and output it into ms form */
                    string part1 = "./macs "; string part2 = " "; string part3 = " -t "; string part4 = " -r "; string part5 = " -s ";
                    string part6 = " -h 1e3 2>debug.txt | ./msformatter > file1.txt";
                    string command = part1+foundhap+part2+SizeChr+part3+scalmut+part4+scalrec+part5+macsseed+part6;
                    if(i == 0)
                    {
                        logfile << "    - Line used to call MaCS: " << endl;
                        logfile << "    " << command << endl;
                    }
                    system(command.c_str());
                    system("rm -rf haplo* tree.* debug.txt");
                    /* Part 2 put into right format */
                    system("tail -n +6 file1.txt > Intermediate.txt");
                    part1 = "tail -n +2 Intermediate.txt > ";
                    stringstream ss; ss << (i + 1); string str = ss.str();
                    part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                    string genofile = part2 + str + part3; SNPFiles[i] = genofile; system(command.c_str());     /* Store name of genotype file */
                    system("head -n 1 Intermediate.txt > TempMap.txt");
                    part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                    command = part1 + mapfile; MapFiles[i] = mapfile; system(command.c_str());                  /* Store name of map file */
                    part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                    part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                    part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                    part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                    part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                    part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                    system("rm Intermediate.txt file1.txt TempMap.txt");
                    /* need to move files to output Directory */
                    part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                    command = part1 + SNPFiles[i] + part2 + SNPFiles[i]; system(command.c_str());
                    command = part1 + MapFiles[i] + part2 + MapFiles[i]; system(command.c_str());
                    logfile << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
                }
            }
            logfile << " Finished Generating Sequence Information: " << endl << endl;
            system("rm -rf ./nul || true");
        }
        if(SimParameters.getStartSim() != "sequence")
        {
            logfile << "============================================================\n";
            logfile << "===\t MaCS Part of Program (Chen et al. 2009) \t====\n";
            logfile << "============================================================\n";
            logfile << "    - File already exist do not need to create sequence information." << endl;
            logfile << "    - Need to ensure that parameters related to sequence information " << endl;
            logfile << "    - from previous simulation are what you wanted!!" << endl << endl;
            /* even though files already exist need to get names of files */
            for(int i = 0; i < SimParameters.getChr(); i++)
            {
                /* For the SNP and Map files */
                stringstream ss; ss<<(i + 1); string str=ss.str(); string part2="CH";
                string part3="SNP.txt"; string genofile=part2+str+part3; SNPFiles[i] = genofile;
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; MapFiles[i] = mapfile;
            }
        }
        if(SimParameters.getStartSim() == "founder" || SimParameters.getStartSim() == "sequence")
        {
            time_t t_start = time(0);
            logfile << "====================================================================\n";
            logfile << "===\t GenoDiver Part of Program (Howard et al. 2016) \t====\n";
            logfile << "====================================================================\n";
            logfile << " Begin Generating Founder Individual Genotypes: " << endl;
            mt19937 gen(SimParameters.getSeed());                                               /* Generate random number to start with */
            vector <double> SNPFreq;
            vector <string> FounderIndividuals((SimParameters.getfoundermale()+SimParameters.getfounderfemale()),"");   /* Founder haplotype across chr */
            vector <string> haplorowids((SimParameters.getfoundermale()+SimParameters.getfounderfemale()),"");  /* Rows that genotypes were from */
            for(int i = 0; i < SimParameters.getChr(); i++)                                     /* Loop through chr and grap haplotypes */
            {
                /* Read in all available haplotype for a particular chromosome */
                string snppath = path + "/" + SimParameters.getOutputFold() + "/" + SNPFiles[i];             /* directory where file is read in */
                /* Read in Haplotypes */
                vector <string> Haplotypes; string gam;
                ifstream infile; infile.open(snppath);
                if (infile.fail()){logfile << "Error Opening MaCS SNP File. Check Parameter File!\n"; exit (EXIT_FAILURE);}
                while (infile >> gam){Haplotypes.push_back(gam);}
                /* need to push_back GenoSum when start a new chromosome */
                ChrSNPLength[i] = Haplotypes[0].size();
                if(Haplotypes.size() < SimParameters.getfnd_haplo())
                {
                    cout << endl << "The number of haplotypes from a previous scenario has changed. Start from sequence!!" << endl; exit (EXIT_FAILURE);
                }
                logfile << "    - Number of SNP for Chromosome " << i + 1 << ": " << ChrSNPLength[i] << " SNP." << endl;
                if((ChrSNPLength[i]*0.95) < (SimParameters.get_Marker_chr())[i])
                {
                    logfile << endl << " Number of Markers too high for chromosome " << i + 1 << ". Alter parameterfile. " << endl; exit (EXIT_FAILURE);
                }
                /* Initialize all frequency counts to zero */
                int GenoFreqStart = SNPFreq.size();                               /* Index to tell you where you should start summing allele count */
                for(int g = 0; g < Haplotypes[0].size(); g++){SNPFreq.push_back(0);}    /* Starts at zero for each new snp */
                vector < int > withinchrhaplotype_used(SimParameters.getfnd_haplo(),0); /* Only a single haplotype is used across animals with chr */
                /* First take 1000 samples to determine distribution of marker homozygosity */
                vector < double > samplehomozygosity(1000,0.0);
                for(int sample = 0; sample < 1000; sample++)
                {
                    vector < int > rowstograb(2,-5);
                    /* Randomly grab to haplotypes */
                    for (int j = 0; j < 2; j++)                                                 /* Loop through 2x to get two haplotypes */
                    {
                        while(1)
                        {
                            std::uniform_real_distribution <double> distribution(0,1);          /* Generate sample */
                            double temp = (distribution(gen) * (SimParameters.getfnd_haplo()-1)); int tempa = (temp + 0.5);
                            if(j == 0){rowstograb[j] = tempa; break;}
                            if(j == 1){if(tempa != rowstograb[0]){rowstograb[j] = tempa; break;}}
                        }
                    }
                    string fullhomologo1 = Haplotypes[rowstograb[0]];            /* Paternal Haplotype */
                    string fullhomologo2 = Haplotypes[rowstograb[1]];            /* Maternal Haplotype */
                    double homozygosity = 0.0;
                    for(int g = 0; g < fullhomologo1.size(); g++)
                    {
                        int temp1 = fullhomologo1[g] - 48; int temp2 = fullhomologo2[g] - 48;
                        int tempvar = 0;
                        if(temp1 == 1 && temp2 == 1){homozygosity += 1;}                /* add genotype a1a1 to string then to genosum count */
                        if(temp1 == 2 && temp2 == 2){homozygosity += 1;}                /* add genotype a2a2 to string then to genosum count */
                    }
                    homozygosity = homozygosity / double(fullhomologo1.size()); samplehomozygosity[sample] = homozygosity;
                }
                /* Figure Out Mean */
                double samplemean = 0.0;
                for(int i = 0; i < samplehomozygosity.size(); i++){samplemean += samplehomozygosity[i];}
                samplemean = samplemean / double(samplehomozygosity.size());
                /* Figure out SD */
                double samplesd = 0.0;
                for(int i = 0; i < samplehomozygosity.size(); i++){samplesd += ((samplehomozygosity[i]-samplemean) * (samplehomozygosity[i]-samplemean));}
                samplesd = sqrt(samplesd / double(samplehomozygosity.size() - 1));
                /* First sets the miniumum at 0.90 and increases by 0.01 every 250 samples rejected */
                double minimimumhomozyg = samplemean - samplesd; double maximumhomozyg = samplemean + samplesd; int samplestaken = 0;
                for(int numfoun = 0; numfoun < (SimParameters.getfoundermale()+SimParameters.getfounderfemale()); numfoun++)
                {
                    string kill = "NO";
                    while(kill == "NO")
                    {
                        vector < int > rowstograb(2,-5);
                        /* Randomly grab to haplotypes */
                        for (int j = 0; j < 2; j++)                                                 /* Loop through 2x to get two haplotypes */
                        {
                            while(1)
                            {
                                std::uniform_real_distribution <double> distribution(0,1);          /* Generate sample */
                                double temp = (distribution(gen) * (SimParameters.getfnd_haplo()-1));
                                int tempa = (temp + 0.5);
                                if(withinchrhaplotype_used[tempa] == 0)                             /* first checks to make sure hasn't been used */
                                {
                                    tempa += 1;                  /* index go from 0 to founderhaplotypes -1; need to add one to get back to line numbers*/
                                    rowstograb[j] = tempa; break;
                                }
                            }
                        }
                        /* First check to see if passes threshold */
                        string fullhomologo1 = Haplotypes[(rowstograb[0]) - 1];            /* Paternal Haplotype */
                        string fullhomologo2 = Haplotypes[(rowstograb[1]) - 1];            /* Maternal Haplotype */
                        double homozygosity = 0.0;
                        for(int g = 0; g < fullhomologo1.size(); g++)
                        {
                            int temp1 = fullhomologo1[g] - 48; int temp2 = fullhomologo2[g] - 48;
                            if(temp1 == 1 && temp2 == 1){homozygosity += 1;}                /* add genotype a1a1 to string then to genosum count */
                            if(temp1 == 2 && temp2 == 2){homozygosity += 1;}                /* add genotype a2a2 to string then to genosum count */
                        }
                        homozygosity = homozygosity / double(fullhomologo1.size());
                        if(homozygosity < minimimumhomozyg || homozygosity > maximumhomozyg){samplestaken += 1;}     /* Didn't pass re-sample */
                        if(homozygosity >= minimimumhomozyg && homozygosity <= maximumhomozyg)                       /* Did pass so keep */
                        {
                            withinchrhaplotype_used[rowstograb[0]-1] = 1; withinchrhaplotype_used[rowstograb[1]-1] = 1;
                            /* Save row numbers */
                            stringstream s1; s1 << rowstograb[0]; string tempvar1 = s1.str();
                            stringstream s2; s2 << rowstograb[1]; string tempvar2 = s2.str();
                            if(i == 0){haplorowids[numfoun] = haplorowids[numfoun] + tempvar1 + "_" + tempvar2;}
                            if(i > 0){haplorowids[numfoun] = haplorowids[numfoun] + "_" + tempvar1 + "_" + tempvar2;}
                            /* Save as a genotype string */
                            string fullhomologo1 = Haplotypes[(rowstograb[0]) - 1];                     /* Paternal Haplotype */
                            string fullhomologo2 = Haplotypes[(rowstograb[1]) - 1];                     /* Maternal Haplotype */
                            stringstream strStreamM (stringstream::in | stringstream::out);             /* Used to put genotype into string */
                            int start = GenoFreqStart;                                                  /* where to start for each founder */
                            for(int g = 0; g < fullhomologo1.size(); g++)
                            {
                                int temp1 = fullhomologo1[g] - 48; int temp2 = fullhomologo2[g] - 48;
                                if(temp1 == 1 && temp2 == 1){strStreamM << 0; SNPFreq[start] += 0;} /* add genotype a1a1 to string and genosum count */
                                if(temp1 == 2 && temp2 == 2){strStreamM << 2; SNPFreq[start] += 2;} /* add genotype a2a2 to string and genosum count */
                                if(temp1 == 1 && temp2 == 2){strStreamM << 3; SNPFreq[start] += 1;} /* add genotype a1a2 to string and genosum count */
                                if(temp1 == 2 && temp2 == 1){strStreamM << 4; SNPFreq[start] += 1;} /* add genotype a1a2 to string and genosum count */
                                start++;
                            }
                            string Genotype = strStreamM.str();
                            FounderIndividuals[numfoun] = FounderIndividuals[numfoun] + Genotype;
                            kill = "YES";
                        }
                        if(samplestaken > 2500){minimimumhomozyg -= samplesd; maximumhomozyg += samplesd; samplestaken = 0;}
                    }
                }
            }
            logfile << "    - Size of Founder Sequence: " << FounderIndividuals[0].size() << " SNP." << endl;
            /* Founder Sequence Genotypes Created */
            int TotalSNP = SNPFreq.size();
            // Calculate Frequency and output //
            #pragma omp parallel for
            for(int i = 0; i < SNPFreq.size(); i++)
            {
                SNPFreq[i] = SNPFreq[i] / double(2 * (SimParameters.getfoundermale()+SimParameters.getfounderfemale()));
            }
            stringstream outputstringfreq(stringstream::out);
            for(int i = 0; i < SNPFreq.size(); i++)
            {
                if(i == 0){outputstringfreq << SNPFreq[i];}
                if(i > 0){outputstringfreq << " " << SNPFreq[i];}
            }
            ofstream output15; output15.open (snpfreqfileloc);
            output15 << outputstringfreq.str(); outputstringfreq.str(""); outputstringfreq.clear();
            // Part 4: Output into FounderFile
            ofstream output16; output16.open (foundergenofileloc);
            for(int i = 0; i < (SimParameters.getfoundermale()+SimParameters.getfounderfemale()); i++)
            {
                output16 << haplorowids[i] << " " << FounderIndividuals[i] << endl;
            }
            output16.close();
            /* Delete vectors and linenumber array to conserve memory */
            logfile << " Finished Generating Founder Individual Genotypes (Founders = " << FounderIndividuals.size() << ")." << endl << endl;
            FounderIndividuals.clear(); haplorowids.clear();
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
            int SIZEDF = 0;
            for(int i = 0; i < SimParameters.getChr(); i++)
            {
                SIZEDF += (SimParameters.get_Marker_chr())[i] + (SimParameters.get_QTL_chr())[i];
                SIZEDF += (SimParameters.get_FTL_lethal_chr())[i] + (SimParameters.get_FTL_sublethal_chr())[i];
            }
            vector < double > FullMapPos(SIZEDF,0.0);                       /* Marker Map for Marker + QTL */
            vector < int > FullColNum(SIZEDF,0);                            /* Column number for SNP that were kept */
            vector < double > FullSNPFreq(SIZEDF,0.0);                      /* SNP Frequency in Full Set of SNP */
            vector < int > FullQTL_Mark(SIZEDF,0);                          /* Indicator of whether SNP was a (Marker, QTL, or QTLFitness) */
            vector < double > FullAddEffectQuan(SIZEDF,0.0);                /* Array of additive_quan effect */
            vector < double > FullDomEffectQuan(SIZEDF,0.0);                /* Array of dominance_quan effect */
            vector < double > FullAddEffectFit(SIZEDF,0.0);                 /* Array of additive_quan effect */
            vector < double > FullDomEffectFit(SIZEDF,0.0);                 /* Array of dominance_quan effect */
            int TotalSNPCounter = 0;                                        /* Counter to set where SNP is placed FullMapPos, FullColNum, FullQTL_Mark */
            int TotalFreqCounter = 0;                                       /* Counter to set where SNP is placed in Freq File for QLT Location */
            int TotalFreqCounter1 = 0;                                      /* Counter to set where SNP is placed in Freq File for Marker Location */
            int firstsnpfreq = 0;                                           /* first snp that pertains to a given chromosome */
            /* Vectors used within each chromosome set to nothing now and then within increase to desired size and when done with clear to start fresh */
            vector < double > mappos;                                       /* Map position for a given chromosome;set length last SNP map */
            vector <double> Add_Array_Fit; vector <double> Add_Array_Quan;  /* array that holds additive effects for each SNP */
            vector <double> Dom_Array_Fit; vector <double> Dom_Array_Quan;  /* array that holds dominance effects for each SNP */
            vector < int > QTL;                                             /* array to hold whether SNP is QTL or can be used as a marker */
            vector < int > markerlocation;                                  /* at what number is marker suppose to be at */
            vector < double > mapmark;                                      /* array with Marker Map for Test Genotype */
            vector < int > colnum;                                          /* array with Column Number for genotypes that were kept to grab SNP Later */
            vector < double > MarkerQTLFreq;                                /* array with frequency's for marker and QTL SNP */
            vector < int > QTL_Mark;                                        /* array of whether SNP is QTL or Marker */
            vector < double > addMark_Fit; vector < double > addMark_Quan;  /* array of additive effects for each SNP */
            vector < double > domMark_Fit; vector < double > domMark_Quan;  /* array of dominance effects for each SNP */
            /* Indicator for type of trait: 1 = marker; 2 = QTLQuanti; 3 = QTLQuant_QTLFitness; 4 = QTLlethal; 5 =QTLsublethal */
            vector < double > covar_add_fitness;                            /* used later on to generate covariance */
            vector < double > covar_add;                                    /* used later on to generate covariance */
            for(int c = 0; c < SimParameters.getChr(); c++)
            {
                logfile << "   - Chromosome " << c + 1 << ":" << endl;
                int endspot = ChrSNPLength[c];                              /* Determines number of SNP to have in order to get end position */
                vector < double > mappos;                           /* Map position for a given chromosome; set length by last SNP map position number */
                string mapfilepath = path + "/" + SimParameters.getOutputFold() + "/" + MapFiles[c];
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
                    Add_Array_Fit.push_back(0.0); Dom_Array_Fit.push_back(0.0); Add_Array_Quan.push_back(0.0);
                    Dom_Array_Quan.push_back(0.0); QTL.push_back(0.0);
                }
                /* use uniform distribution to randomly place QTL along genome for quantitative trait */
                int qtlcountertotal = 0; int numbertimesqtlcycled = 0;
                while(qtlcountertotal < (SimParameters.get_QTL_chr())[c])
                {
                    double pltrpic = 0.0;
                    std::uniform_real_distribution <double> distribution(0,1);
                    pltrpic = distribution(gen);
                    if(pltrpic < SimParameters.getproppleitropic())        /* Has pleitrophic effect need find snp below MAF for fitness */
                    {
                        string killwithin = "NO";
                        while(killwithin == "NO")
                        {
                            std::uniform_real_distribution <double> distribution(0,1);
                            int indexid = (distribution(gen)) * endspot;
                            if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid + firstsnpfreq]<SimParameters.getUpThrMAFSFit() && MAF[indexid + firstsnpfreq]>0.03)
                            {
                                string keep = "yes"; int search = 1;
                                while(1)
                                {
                                    stringstream s4; s4 << (SimParameters.get_ChrLength())[c]; string SizeChr = s4.str(); /* string chromosome length */
                                    //cout << QTL[indexid-search] << " " << int(mappos[indexid-search] * ChrLength[c]) << " -- ";
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])!=int(mappos[indexid-search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        break;
                                    }
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])==int(mappos[indexid-search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        if(QTL[indexid-search] != 0){keep = "no"; break;}
                                        if(QTL[indexid-search] == 0){search++;}
                                    }
                                }
                                //cout << "1=" << QTL[indexid] << " " << int(mappos[indexid] * ChrLength[c]) << " -- ";
                                search = 1;
                                while(1)
                                {
                                    //cout << QTL[indexid+search] << " " << int(mappos[indexid+search] * ChrLength[c]) << " -- ";
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])!=int(mappos[indexid+search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        break;
                                    }
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])==int(mappos[indexid+search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        if(QTL[indexid+search] != 0){keep = "no"; break;}
                                        if(QTL[indexid+search] == 0){search++;}
                                    }
                                }
                                if(keep == "yes")
                                {
                                    QTL[indexid] = 3; qtlcountertotal++; killwithin = "YES";
                                }
                            }
                            numbertimesqtlcycled++;
                            if(numbertimesqtlcycled > (1000 * endspot))
                            {
                                logfile << endl << "Couldn't find enough quant+fitness that pass MAF threshold! increase fitness MAF or decrease QTL Number!" << endl;
                                exit (EXIT_FAILURE);
                            }
                        }
                    }
                    if(pltrpic >= SimParameters.getproppleitropic())     /* No pleitrophic effect only need to sample quantitative */
                    {
                        string killwithin = "NO";
                        while(killwithin == "NO")
                        {
                            std::uniform_real_distribution <double> distribution(0,1);
                            int indexid = (distribution(gen)) * endspot;
                            if(QTL[indexid]==0&&mappos[indexid]>0.0001&&mappos[indexid]<0.9999&&MAF[indexid+firstsnpfreq] > (SimParameters.getThresholdMAFQTL()))
                            {
                                string keep = "yes"; int search = 1;
                                while(1)
                                {
                                    //cout << QTL[indexid-search] << " " << int(mappos[indexid-search] * ChrLength[c]) << " -- ";
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])!=int(mappos[indexid-search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        break;
                                    }
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])==int(mappos[indexid-search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        if(QTL[indexid-search] != 0){keep = "no"; break;}
                                        if(QTL[indexid-search] == 0){search++;}
                                    }
                                }
                                //cout << "1=" << QTL[indexid] << " " << int(mappos[indexid] * ChrLength[c]) << " -- ";
                                search = 1;
                                while(1)
                                {
                                    //cout << QTL[indexid+search] << " " << int(mappos[indexid+search] * ChrLength[c]) << " -- ";
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])!=int(mappos[indexid+search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        break;
                                    }
                                    if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c])==int(mappos[indexid+search]*(SimParameters.get_ChrLength())[c]))
                                    {
                                        if(QTL[indexid+search] != 0){keep = "no"; break;}
                                        if(QTL[indexid+search] == 0){search++;}
                                    }
                                }
                                if(keep == "yes")
                                {
                                    QTL[indexid] = 2; qtlcountertotal++; killwithin = "YES";
                                }
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
                /* if randomly samples will make it difficult to find specific set of lethal QTL */
                vector < int > passMAFthreshold;
                for(int i = 0; i < endspot; i++)
                {
                    if(QTL[i]==0 && mappos[i]>0.0001 && mappos[i]<0.9999 && MAF[i+firstsnpfreq]<(SimParameters.getUpThrMAFLFit()) && MAF[i+firstsnpfreq] > (SimParameters.getUpThrMAFLFit()-SimParameters.getRangeMAFLFit()))
                    {
                        passMAFthreshold.push_back(i);
                        //cout << mappos[i] << "-" << MAF[i+firstsnpfreq] << "   ";
                    }
                }
                //cout << endl;
                if(passMAFthreshold.size() < (SimParameters.get_FTL_lethal_chr())[c])
                {
                    logfile << endl << "Couldn't find enough lethal that pass MAF threshold! Increase lethal MAF or decrease QTL Number!" << endl;
                    exit (EXIT_FAILURE);
                }
                qtlcountertotal = 0;
                while(qtlcountertotal < (SimParameters.get_FTL_lethal_chr())[c])
                {
                    std::uniform_real_distribution <double> distribution(0,1);
                    int indexid = (distribution(gen)) * passMAFthreshold.size();
                    if(QTL[passMAFthreshold[indexid]] == 0)
                    {
                        string keep = "yes"; int search = 1;
                        while(1)
                        {
                            //cout << QTL[passMAFthreshold[indexid]-search]<<" "<<int(mappos[passMAFthreshold[indexid]-search]*ChrLength[c]) << " -- ";
                            if(int(mappos[passMAFthreshold[indexid]]*(SimParameters.get_ChrLength())[c]) != int(mappos[passMAFthreshold[indexid]-search]*(SimParameters.get_ChrLength())[c]))
                            {
                                break;
                            }
                            if(int(mappos[passMAFthreshold[indexid]]*(SimParameters.get_ChrLength())[c]) == int(mappos[passMAFthreshold[indexid]-search] * (SimParameters.get_ChrLength())[c]))
                            {
                                if(QTL[passMAFthreshold[indexid]-search] != 0){keep = "no"; break;}
                                if(QTL[passMAFthreshold[indexid]-search] == 0){search++;}
                            }
                        }
                        //cout << "1=" << QTL[passMAFthreshold[indexid]] << " " << int(mappos[passMAFthreshold[indexid]] * ChrLength[c]) << " -- ";
                        search = 1;
                        while(1)
                        {
                            //cout << QTL[passMAFthreshold[indexid]+search]<<" "<<int(mappos[passMAFthreshold[indexid]+search]*ChrLength[c]) << " -- ";
                            if(int(mappos[passMAFthreshold[indexid]]*(SimParameters.get_ChrLength())[c]) != int(mappos[passMAFthreshold[indexid]+search]*(SimParameters.get_ChrLength())[c]))
                            {
                                break;
                            }
                            if(int(mappos[passMAFthreshold[indexid]]*(SimParameters.get_ChrLength())[c]) == int(mappos[passMAFthreshold[indexid]+search]*(SimParameters.get_ChrLength())[c]))
                            {
                                if(QTL[passMAFthreshold[indexid]+search] != 0){keep = "no"; break;}
                                if(QTL[passMAFthreshold[indexid]+search] == 0){search++;}
                            }
                        }
                        if(keep == "yes")
                        {
                            QTL[passMAFthreshold[indexid]] = 4; qtlcountertotal++; //cout << "Got to End" << endl;
                        }
                    }
                }
                passMAFthreshold.clear();
                 /* use uniform distribution to randomly place QTL along genome for sub-Lethal Fitness */
                qtlcountertotal = 0; numbertimesqtlcycled = 0;
                while(qtlcountertotal < (SimParameters.get_FTL_sublethal_chr())[c])
                {
                    std::uniform_real_distribution <double> distribution(0,1);
                    int indexid = (distribution(gen)) * endspot;
                    if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid + firstsnpfreq]<SimParameters.getUpThrMAFSFit() && MAF[indexid + firstsnpfreq]>0.01)
                    {
                        string keep = "yes"; int search = 1;
                        while(1)
                        {
                            //cout << QTL[indexid-search] << " " << int(mappos[indexid-search]*ChrLength[c]) << " -- ";
                            if(int(mappos[indexid]*(SimParameters.get_ChrLength())[c]) != int(mappos[indexid-search]*(SimParameters.get_ChrLength())[c]))
                            {
                                break;
                            }
                            if(int(mappos[indexid] * (SimParameters.get_ChrLength())[c]) == int(mappos[indexid-search] * (SimParameters.get_ChrLength())[c]))
                            {
                                if(QTL[indexid-search] != 0){keep = "no"; break;}
                                if(QTL[indexid-search] == 0){search++;}
                            }
                        }
                        //cout << "1=" << QTL[indexid] << " " << int(mappos[indexid]*ChrLength[c]) << " -- ";
                        search = 1;
                        while(1)
                        {
                            //cout << QTL[indexid+search] << " " << int(mappos[indexid+search]*ChrLength[c]) << " -- ";
                            if(int(mappos[indexid] * (SimParameters.get_ChrLength())[c]) != int(mappos[indexid+search] * (SimParameters.get_ChrLength())[c]))
                            {
                                break;
                            }
                            if(int(mappos[indexid] * (SimParameters.get_ChrLength())[c]) == int(mappos[indexid+search] * (SimParameters.get_ChrLength())[c]))
                            {
                                if(QTL[indexid+search] != 0){keep = "no"; break;}
                                if(QTL[indexid+search] == 0){search++;}
                            }
                        }
                        if(keep == "yes"){QTL[indexid] = 5; qtlcountertotal++;}
                    }
                    numbertimesqtlcycled++;
                    if(numbertimesqtlcycled > (100 * endspot))
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
                        /******* QTL Additive Effect (Gamma *******/
                        std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape(),SimParameters.getGamma_Scale());
                        Add_Array_Quan[i] = distribution1(gen);
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                        std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                        double temph = distribution2(gen);
                        Dom_Array_Quan[i] = (Add_Array_Quan[i] * temph);
                        /* Determine sign of additive effect */
                        /* 1 tells you range so can sample from 0 to 1 and 0.5 is the frequency */
                        std::binomial_distribution<int> distribution3(1,0.5);
                        int signadd = distribution3(gen);
                        if(signadd == 1){Add_Array_Quan[i] = Add_Array_Quan[i] * -1;}           /*Assign negative effect with a 50% probability */
                        /* set fitness ones to 0 */
                        Add_Array_Fit[i] = 0; Dom_Array_Fit[i] = 0;
                    }
                    if(QTL[i] == 3)                 /* Tagged as quantitative QTL with a relationship to fitness */
                    {
                        /******* QTL Additive Effect ********/
                        /* Utilizing a Trivariate Reduction */
                        double sub = SimParameters.getgencorr() * sqrt(SimParameters.getGamma_Shape()*SimParameters.getGamma_Shape_SubLethal());
                        std::gamma_distribution <double> distribution1((SimParameters.getGamma_Shape()-sub),1);            /* QTL generated from a gamma */
                        double temp = distribution1(gen);
                        std::gamma_distribution <double> distribution2(sub,1);                          /* Covariance part */
                        double tempa = distribution2(gen);
                        Add_Array_Quan[i] = SimParameters.getGamma_Scale() * (temp + tempa);
                        covar_add_fitness.push_back(tempa);
                        /*******   QTL Fit S effect  *******/               /* Generate effect after standardize add and dominance effect */
                        Add_Array_Fit[i] = -5; Dom_Array_Fit[i] = -5;       /* Set it to -5 first to use as a flag since # has to be greater than 0 */
                        /******* QTL Additive Effect *******/
                        /* 1 tells you range so can sample from 0 to 1 and 0.5 is the frequency */
                        //std::binomial_distribution<int> distribution4(1,0.5);
                        //int signadd = distribution4(gen);
                        //if(signadd == 1){Add_Effect_Quan[i] = Add_Effect_Quan[i] * -1;}       /*Assign negative effect with a 50% probability */
                        covar_add.push_back(Add_Array_Quan[i]);
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                        std::normal_distribution<double> distribution5(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                        double temph = distribution5(gen);
                        Dom_Array_Quan[i] = abs(Add_Array_Quan[i]) * temph;
                    }
                    if(QTL[i] == 4)                 /* Tagged as lethal fitness QTL with no relationship to quantitative */
                    {
                        /*******     QTL s effect (i.e. selection coeffecient)   *******/
                        std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape_Lethal(),SimParameters.getGamma_Scale_Lethal());
                        Add_Array_Fit[i] = distribution1(gen);
                        /******      QTL h Effect (i.e. degree of dominance)     *******/
                        /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                        std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom_Lethal(),SimParameters.getNormal_varRelDom_Lethal());
                        double temph = distribution2(gen);
                        Dom_Array_Fit[i] = abs(temph);
                        /* Set quantitative additive and dominance to 0 */
                        Add_Array_Quan[i] = 0; Dom_Array_Quan[i] = 0;
                    }
                    if(QTL[i] == 5)                 /* Tagged as sub-lethal fitness QTL with no relationship to quantitative */
                    {
                        /*******     QTL S effect (i.e. selection coeffecient)   *******/
                        std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape_SubLethal(),SimParameters.getGamma_Scale_SubLethal());
                        Add_Array_Fit[i] = distribution1(gen);
                        /******      QTL h Effect (i.e. degree of dominance)     *******/
                        /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                        std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom_SubLethal(),SimParameters.getNormal_varRelDom_SubLethal());
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
                markerperChr[c] = (SimParameters.get_Marker_chr())[c];
                ///////////////////////////////////////////////////////////////////////
                // Step 2: Construct Marker file based on given Marker MAF threshold //
                ///////////////////////////////////////////////////////////////////////
                /* Create Genotypes that has QTL and Markers */
                int markertokeep = markerperChr[c] + qtlperChr[c];  /* Determine length of array based on markers to keep plus number of actual QTL */
                for(int i = 0; i < markertokeep; i++)
                {
                    mapmark.push_back(0.0); colnum.push_back(0); MarkerQTLFreq.push_back(0.0); QTL_Mark.push_back(0);
                    addMark_Fit.push_back(0.0); domMark_Fit.push_back(0.0); addMark_Quan.push_back(0.0); domMark_Quan.push_back(0.0);
                }
                /* use uniform distribution to randomly place markers along genome */
                int markercountertotal = 0; int numbertimescycled = 0;
                while(markercountertotal < (SimParameters.get_Marker_chr())[c])
                {
                    std::uniform_real_distribution <double> distribution(0,1);
                    /* randomly sets index location */
                    int indexid = (distribution(gen)) * endspot;
                    if(QTL[indexid]==0 && mappos[indexid]>0.0001 && mappos[indexid]<0.9999 && MAF[indexid+firstsnpfreq]>=(SimParameters.getThresholdMAFMark()))
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
            /* Split off SNP that are markers and QTL for either Quantitative, Fitness or both by creating an index from 1 to number of SNP */
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
            for(int i = 0; i < SimParameters.getChr(); i++)
            {
                double chr = i + 1;
                for(int j = 0; j < Marker_IndCounter; j++)
                {
                    if(MarkerMapPosition[j] >= i + 1 && MarkerMapPosition[j] < i + 2)
                    {
                        MarkerMapPosition_MB[j] = (MarkerMapPosition[j] - chr) * (SimParameters.get_ChrLength())[i]; MarkerMapPosition_CHR[j] =  chr;
                    }
                }
            }
            ofstream output21;
            output21.open (Marker_Map);
            output21 << "chr pos" << endl;
            for(int i = 0; i < Marker_IndCounter; i++){output21 << MarkerMapPosition_CHR[i] << " " << MarkerMapPosition_MB[i] << endl;}
            output21.close();
            /* If doing ROH genome summary create a file with the appropriate row heading for frequency and length */
            if(SimParameters.getmblengthroh() > 0)
            {
                ofstream outputrohfreq;
                outputrohfreq.open(Summary_ROHGenome_Freq);
                outputrohfreq << "chr pos" << endl;
                for(int i = 0; i < Marker_IndCounter; i++){outputrohfreq << MarkerMapPosition_CHR[i] << " " << MarkerMapPosition_MB[i] << endl;}
                outputrohfreq.close();
                ofstream outputrohlength;
                outputrohlength.open(Summary_ROHGenome_Length);
                outputrohlength << "chr pos" << endl;
                for(int i = 0; i < Marker_IndCounter; i++){outputrohlength << MarkerMapPosition_CHR[i] << " " << MarkerMapPosition_MB[i] << endl;}
                outputrohlength.close();
            }
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
            //////////////////////////////////////////////////////////////
            // Now for each founder generate marker panel and qtl panel //
            //////////////////////////////////////////////////////////////
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
                for(int i = 0; i < SimParameters.getChr(); i++)
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
                    if(i != SimParameters.getChr())
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
            logfile << "   - Check Structure of Founder Individuals." << endl;
            /* Create Genomic Relationship Matrix to Remove Structure */
            /* calculate frequency */
            double* foundergmatrixfreq = new double[founder_snp[0].size()];     /* Array that holds SNP that were declared as Markers and QTL */
            frequency_calc(founder_snp, foundergmatrixfreq);                    /* Function to calculate snp frequency */
            /* M matrix for founders */
            double* Mfounder = new double[3*founder_snp[0].size()];             /* M matrix used to calculate GRM */
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < founder_snp[0].size(); j++){Mfounder[(i*founder_snp[0].size())+j] = i - (2 * foundergmatrixfreq[j]);}
            }
            /* Calculate Scale */
            float founderscale = 0;
            for (int j=0; j < founder_snp[0].size(); j++){founderscale += (1 - foundergmatrixfreq[j]) * foundergmatrixfreq[j];}
            founderscale = founderscale * 2;
            /* Generate G Matrix */
            double* foundergmatrix = new double[founder_snp.size()*founder_snp.size()];
            for(int i = 0; i < (founder_snp.size()*founder_snp.size()); i++){foundergmatrix[i] = 0.0;}
            grm_noprevgrm(Mfounder,founder_snp,foundergmatrix,founderscale);
            double mean_offdiagonals = 0.0; int numberoffdiagonals = 0;
            double sd_offdiagonals = 0.0;
            vector < int > extremerelationshipcount (founder_snp.size(),0);
            for(int i = 0; i < founder_snp.size(); i++)
            {
                for(int j = i; j < founder_snp.size(); j++)
                {
                    if(i != j)
                    {
                        mean_offdiagonals += foundergmatrix[(i*founder_snp.size()) + j];
                        numberoffdiagonals += 1;
                        if(foundergmatrix[(i*founder_snp.size()) + j] > 0.75)
                        {
                            extremerelationshipcount[i] += 1; extremerelationshipcount[j] += 1;
                        }
                    }
                }
            }
            mean_offdiagonals = mean_offdiagonals / double(numberoffdiagonals);
            for(int i = 0; i < founder_snp.size(); i++)
            {
                for(int j = i; j < founder_snp.size(); j++)
                {
                    if(i != j)
                    {
                        sd_offdiagonals += ((foundergmatrix[(i*founder_snp.size()) + j]-mean_offdiagonals)*(foundergmatrix[(i*founder_snp.size()) + j]-mean_offdiagonals));
                    }
                }
            }
            sd_offdiagonals = sd_offdiagonals / double(numberoffdiagonals-1);
            logfile << "        - Mean (sd) of off-diagonal founder GRM: " << mean_offdiagonals << " (" << sd_offdiagonals << ")" << endl;
            delete [] foundergmatrixfreq; delete [] Mfounder; delete [] foundergmatrix;
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
            /* Variance Additive = 2pq[a + d(q-p)]^2; Variance Dominance = (2pqd)^2; One depends on the other so therefore do optimization technique */
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
                if(abs(SimParameters.getVarAdd() - tempva) < 0.004){scalefactaddh2 = scalefactaddh2;}           /* Add Var Within Window (+/-) don't change */
                if(SimParameters.getVarAdd() - tempva > 0.004){scalefactaddh2 = scalefactaddh2 + 0.0001;}       /* Add Var Smaller than desired increase */
                if(SimParameters.getVarAdd() - tempva < -0.004){scalefactaddh2 = scalefactaddh2 - 0.0001;}      /* Add Var Bigger than desired decrease */
                if(abs(SimParameters.getVarDom() - tempvd) < 0.004){scalefactpdomh2 = scalefactpdomh2;}         /* Dom Var Within Window (+/-) don't change */
                if(SimParameters.getVarDom() - tempvd > 0.004){scalefactpdomh2 = scalefactpdomh2 + 0.0001;}     /* Dom Var Smaller than desired increase */
                if(SimParameters.getVarDom() - tempvd < -0.004){scalefactpdomh2 = scalefactpdomh2 - 0.0001;}    /* Dom Var Bigger than desired decrease */
                if(abs(SimParameters.getVarAdd() - tempva) < 0.004 && abs(SimParameters.getVarDom() - tempvd) < 0.004)  /* Dom & Add Var within window */
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
            if(SimParameters.getproppleitropic() > 0)
            {
                logfile << "   - Generate correlation (rank) between additive effects of quantitative and fitness." << endl;
                double cor = 0.0;
                while(SimParameters.getgencorr() - cor >= 0.015)
                {
                    vector < double > quant_rank; vector < double > fitness_rank;
                    int covarcount = 0;
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if(QTL_Type[i] == 3)
                        {
                            double sub = (-1 * SimParameters.getgencorr()) * sqrt(SimParameters.getGamma_Shape()*SimParameters.getGamma_Shape_SubLethal());
                            /******   QTL fit s Effect ********/
                            std::gamma_distribution <double> distribution2((SimParameters.getGamma_Shape_SubLethal()-sub),1);
                            double temp = distribution2(gen);                /* Raw Fitness Value need to transform to relative fitness */
                            QTL_Add_Fit[i] = SimParameters.getGamma_Scale_SubLethal() * (temp + (covar_add_fitness[covarcount]*scalefactaddh2));
                            /******   QTL Fit h Effect  *******/
                            /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                            std::normal_distribution<double> distribution3(SimParameters.getNormal_meanRelDom_SubLethal(),SimParameters.getNormal_varRelDom_SubLethal());
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
            double meanfreq_leth = 0.0; double meanfreq_sublethal = 0.0;
            for(int i = 0; i < QTL_IndCounter; i++)
            {
                if(QTL_Type[i] == 4)            /* Lethal */
                {
                    meanlethal_sel += QTL_Add_Fit[i]; meanlethal_dom += QTL_Dom_Fit[i]; numberlethal_sel += 1;
                    if(QTL_Freq[i] < 0.5){meanfreq_leth += QTL_Freq[i];}
                    if(QTL_Freq[i] > 0.5){meanfreq_leth += (1 - QTL_Freq[i]);}
                }
                if(QTL_Type[i] == 3 || QTL_Type[i] == 5)
                {
                    meansublethal_sel += QTL_Add_Fit[i]; meansublethal_dom += QTL_Dom_Fit[i]; numbersublethal_sel += 1;
                    if(QTL_Freq[i] < 0.5){meanfreq_sublethal += QTL_Freq[i];}
                    if(QTL_Freq[i] > 0.5){meanfreq_sublethal += (1 - QTL_Freq[i]);}
                }
            }
            if(numbersublethal_sel > 0 && SimParameters.getproppleitropic() > 0)
            {
                meansublethal_sel = meansublethal_sel / double(numbersublethal_sel);
                meansublethal_dom = meansublethal_dom / double(numbersublethal_sel);
                meanfreq_sublethal = meanfreq_sublethal / double(numbersublethal_sel);
                logfile << "   - Fitness Sub-Lethal Allele: " << endl;
                logfile << "        - Mean Frequency: " << meanfreq_sublethal << endl;
                logfile << "        - Mean Selection Coefficient: " << meansublethal_sel << endl;
                logfile << "        - Mean Degree of Dominance: " << meansublethal_dom << endl;
            }
            if(numbersublethal_sel > 0 && SimParameters.getproppleitropic() == 0)
            {
                meansublethal_sel = meansublethal_sel / double(numbersublethal_sel);
                meansublethal_dom = meansublethal_dom / double(numbersublethal_sel);
                meanfreq_sublethal = meanfreq_sublethal / double(numbersublethal_sel);
                logfile << "   - Fitness Sub-Lethal Allele: " << endl;
                logfile << "        - Mean Frequency: " << meanfreq_sublethal << endl;
                logfile << "        - Mean Selection Coefficient: " << meansublethal_sel << endl;
                logfile << "        - Mean Degree of Dominance: " << meansublethal_dom << endl;
            }
            if(numbersublethal_sel == 0)
            {
                logfile << "   - No Sub-Lethal Fitness Mutations; No summary statistics on frequency and allelic effects." << endl;
            }
            if(numberlethal_sel > 0)
            {
                meanlethal_sel = meanlethal_sel / double(numberlethal_sel);
                meanlethal_dom = meanlethal_dom / double(numberlethal_sel);
                meanfreq_leth = meanfreq_leth / double(numberlethal_sel);
                logfile << "   - Fitness Lethal Allele: " << endl;
                logfile << "        - Mean Frequency: " << meanfreq_leth << endl;
                logfile << "        - Mean Selection Coefficient: " << meanlethal_sel << endl;
                logfile << "        - Mean Degree of Dominance: " << meanlethal_dom << endl;
            }
            if(numberlethal_sel == 0)
            {
                logfile << "   - No Lethal Fitness Mutations; No summary statistics on frequency and allelic effects." << endl;
            }
            //////////////////////////////////////////////////////////////////////
            // Step 5: Save Founder mutation in QTL_new_old vector class object //
            //////////////////////////////////////////////////////////////////////
            /* Need to add both Quantitative QTL and Fitness QTL to class */
            for(int i = 0; i < QTL_IndCounter; i++)
            {
                if(QTL_Type[i] == 4 || QTL_Type[i] == 5)            /* Fitness QTL */
                {
                    stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<QTL_Freq[i];
                    string stringfreq=strStreamtempfreq.str();
                    stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<QTL_Type[i];
                    string stringtype=strStreamtemptype.str();
                    QTL_new_old tempa(QTL_MapPosition[i], QTL_Add_Fit[i], QTL_Dom_Fit[i],stringtype, 0, stringfreq, "");
                    population_QTL.push_back(tempa);
                }
                if(QTL_Type[i] == 2)                                /* Quantitative QTL */
                {
                    stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<QTL_Freq[i];
                    string stringfreq=strStreamtempfreq.str();
                    stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<QTL_Type[i];
                    string stringtype=strStreamtemptype.str();
                    QTL_new_old tempa(QTL_MapPosition[i], QTL_Add_Quan[i], QTL_Dom_Quan[i],stringtype, 0, stringfreq,"");
                    population_QTL.push_back(tempa);
                }
                if(QTL_Type[i] == 3)                                /* Fitness + Quantitative QTL */
                {
                    /* Quantitative one first */
                    stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<QTL_Freq[i];
                    string stringfreq=strStreamtempfreq.str();
                    stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<"2";
                    string stringtype=strStreamtemptype.str();
                    QTL_new_old tempa(QTL_MapPosition[i], QTL_Add_Quan[i], QTL_Dom_Quan[i],stringtype, 0, stringfreq,"");
                    population_QTL.push_back(tempa);
                    /* Fitness one */
                    stringstream strStreamtemptypea (stringstream::in | stringstream::out); strStreamtemptypea<<"5";
                    string stringtypea=strStreamtemptypea.str();
                    QTL_new_old tempb(QTL_MapPosition[i], QTL_Add_Fit[i], QTL_Dom_Fit[i],stringtypea, 0, stringfreq,"");
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
            for(int i = 0; i < SimParameters.getChr(); i++)
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
                    if(end == 0 && start == 0){end = end + (SimParameters.gethaplo_size()-1);}
                    if(end > 0 && start > 0){end = end + SimParameters.gethaplo_size();}
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
            /* Now ensure correct number of male and female founder individuals */
            vector < int > male_female_status (founder_snp.size(),-5);
            /* randomly declare males */
            for(int i = 0; i < SimParameters.getfoundermale(); i++)
            {
                while(1)
                {
                    std::uniform_real_distribution <double> distribution(0,1);          /* Generate sample */
                    double temp = (distribution(gen) * (male_female_status.size()-1));
                    int tempa = (temp + 0.5);
                    /* first checks to make sure hasn't been used */
                    if(male_female_status[tempa] == -5){male_female_status[tempa] = 0; break;}  /* Male */
                }
            }
            /* randomly declare females */
            for(int i = 0; i < SimParameters.getfounderfemale(); i++)
            {
                while(1)
                {
                    std::uniform_real_distribution <double> distribution(0,1);          /* Generate sample */
                    double temp = (distribution(gen) * (male_female_status.size()-1));
                    int tempa = (temp + 0.5);
                    /* first checks to make sure hasn't been used */
                    if(male_female_status[tempa] == -5){male_female_status[tempa] = 1; break;}  /* Female */
                }
            }
            /* Double Check to make sure line up */
            int nummales = 0; int numfemales = 0;
            for(int i = 0; i < male_female_status.size(); i++)
            {
                if(male_female_status[i] == 0){nummales += 1;}
                if(male_female_status[i] == 1){numfemales += 1;}
            }
            if(nummales != SimParameters.getfoundermale() || numfemales != SimParameters.getfounderfemale())
            {
                cout << endl << "Error in declares sex in male and females!!" << endl; exit (EXIT_FAILURE);
            }
            /* Deletes Previous Simulation founder low fitness individuals */
            fstream checklowfitness; checklowfitness.open(lowfitnesspath, std::fstream::out | std::fstream::trunc); checklowfitness.close();
            /* add first line as column ID's */
            std::ofstream outputlow(lowfitnesspath, std::ios_base::app | std::ios_base::out);
            outputlow << "Sire Dam Gen Ped_F Gen_F Hap3_F Homozy Homolethal Heterlethal Homosublethal Hetersublethal Letequiv Fitness ";
            outputlow << "GV BV DD QTL_Fitness" << endl;
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
                double relativeviability = 1.0;                 /* represents the multiplicative fitness effect across lethal and sub-lethal alleles */
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
                    stringstream strStreamQf (stringstream::in | stringstream::out);
                    stringstream strStreamleftfit (stringstream::in | stringstream::out);
                    strStreamleftfit << "0" << " " << "0" << " " << "0" << " ";
                    leftfitnessstring.push_back(strStreamleftfit.str());
                    /* Fitness Summary Statistics */
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
                    //cout << homozygouscount_lethal << " " << heterzygouscount_lethal << " ";
                    //cout << homozygouscount_sublethal << " " << heterzygouscount_sublethal << " " << lethalequivalent << endl;
                    /* Quantititative Summary Statistics */
                    double GenotypicValue = 0.0; double BreedingValue = 0.0; double DominanceDeviation = 0.0; double Homoz = 0.0;
                    /* Calculate Genotypic Value */
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if(QTL_Type[i] == 2 || QTL_Type[i] == 3)            /* Quantitative QTL */
                        {
                            int tempgeno;
                            if(QTLGenotypes[i] == 0 || QTLGenotypes[i] == 2){tempgeno = QTLGenotypes[i];}
                            if(QTLGenotypes[i] == 3 || QTLGenotypes[i] == 4){tempgeno = 1;}
                            /* Breeding value is only a function of additive effects */
                            BreedingValue += tempgeno * double(QTL_Add_Quan[i]);
                            if(tempgeno != 1){GenotypicValue += tempgeno * double(QTL_Add_Quan[i]);}     /* Not a heterozygote; function of additive */
                            if(tempgeno == 1)
                            {
                                GenotypicValue += (tempgeno * QTL_Add_Quan[i]) + QTL_Dom_Quan[i];    /* Heterozygote so need to include add and dom */
                                DominanceDeviation += QTL_Dom_Quan[i];
                            }
                        }
                    }
                    /* Calculate Homozygosity only in the MARKERS */
                    for(int i=0; i < Marker_IndCounter; i++){if(MarkerGenotypes[i] == 0 || MarkerGenotypes[i] == 2){Homoz += 1;}}
                    Homoz = (Homoz/double(Marker_IndCounter));
                    //cout << GenotypicValue << " " << BreedingValue << " " << DominanceDeviation << " " << Homoz << endl;
                    strStreamQf << Homoz << " " << homozygouscount_lethal << " " << heterzygouscount_lethal << " ";
                    strStreamQf << homozygouscount_sublethal << " " << heterzygouscount_sublethal << " ";
                    strStreamQf << lethalequivalent << " " << relativeviability << " ";
                    strStreamQf << GenotypicValue << " " << BreedingValue << " " << DominanceDeviation << " ";
                    for (int i=0; i < QTL_IndCounter; i++){if(QTL_Type[i] == 3 || QTL_Type[i] == 4 || QTL_Type[i] == 5){strStreamQf << QTLGenotypes[i];}}
                    rightfitnessstring.push_back(strStreamQf.str());
                    //cout << leftfitnessstring[leftfitnessstring.size()-1] << " " << rightfitnessstring[rightfitnessstring.size()-1] << endl;
                    stringstream strStreamM (stringstream::in | stringstream::out);
                    for (int i=0; i < Marker_IndCounter; i++){strStreamM << MarkerGenotypes[i];} markerlowfitness.push_back(strStreamM.str());
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
                    int Sex;                                                                /* Sex of the animal 0 is male 1 is female */
                    double residvar = 1 - (SimParameters.getVarAdd() + SimParameters.getVarDom());  /* Residual Variance; Total Variance equals 1 */
                    residvar = sqrt(residvar);                                              /* random number generator need standard deviation */
                    /* Determine Sex of the animal based on draw from uniform distribution; if sex < 0.5 sex is 0 if sex >= 0.5 */
                    Sex = male_female_status[i];
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
                    Animal animal(ind,0,0,Sex,0,1,0,0,0,rndselection,rndculling,0.0,0.0,0.0,0.0,0.0,homozygouscount_lethal, heterzygouscount_lethal, homozygouscount_sublethal, heterzygouscount_sublethal,lethalequivalent,Homoz,0.0,0.0,Phenotype,relativeviability,GenotypicValue,BreedingValue,DominanceDeviation,Residual,MA,QT,"","","",0.0,"");
                    /* Then place given animal object in population to store */
                    population.push_back(animal);
                    ind++;                              /* Increment Animal ID by one for next individual */
                }
                delete [] MarkerGenotypes; delete [] QTLGenotypes;
            }
            male_female_status.clear();
            /* Determine whether animal is genotyped or not */
            for(int i = 0; i < population.size(); i++)
            {
                /* pblup implies no animals are genotyped */
                if(SimParameters.getEBV_Calc() == "pblup"){population[i].UpdateGenoStatus("No");}
                /* gblup, rohblup or bayes implies all animals are genotyped with (i.e. "Full") */
                if(SimParameters.getEBV_Calc() == "gblup" || SimParameters.getEBV_Calc() == "rohblup" || SimParameters.getEBV_Calc() == "bayes")
                {
                    population[i].UpdateGenoStatus("Full");
                }
                /* ssgblup implies portion of animals genotyped. Therefore some will have "No" and others will have either "Full" or "Reduced" */
                if(SimParameters.getEBV_Calc() == "ssgblup")
                {
                    /* initialize to not genotyped; can change depending on the genotype strategy */
                    population[i].UpdateGenoStatus("No");
                }
            }
            /* Put newly created progeny in GenoStatus file; Save as a continuous string and then output */
            stringstream outputstringgenostatusfound(stringstream::out);
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].getAge() == 1){outputstringgenostatusfound << population[i].getID() << " " << population[i].getGenoStatus() << endl;}
            }
            /* output genostatus file */
            std::ofstream outputfounder(GenotypeStatus_path.c_str(), std::ios_base::app | std::ios_base::out);
            outputfounder << outputstringgenostatusfound.str(); outputstringgenostatusfound.str(""); outputstringgenostatusfound.clear();
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
                    string homo1 =  (population[j].getMarker()).substr(haplib[i].getStart(),SimParameters.gethaplo_size());     /* Paternal haplotypes */
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
                    string homo1 =  (population[j].getMarker()).substr(haplib[i].getStart(),SimParameters.gethaplo_size());     /* Paternal haplotypes */
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
                        if(g == homo1.size() - 1){hvalue = (1 - (sum/homo1.size())) + 1;}   /* don't need to divide by 2 because is 1+1+(1-sum)+(1-sum) */
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
            if(SimParameters.getLDDecay() == "yes")                            /* Estimate LD Decay */
            {
                time_t intbegin_time = time(0);
                logfile << "Generate Genome Summary Statistics: " << endl;
                /* Clear previous simulation */
                fstream checkldfile; checkldfile.open(LD_Decay_File.c_str(), std::fstream::out | std::fstream::trunc); checkldfile.close();
                /* Vector of string of markers */
                vector < string > markergenotypes;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1){markergenotypes.push_back(population[i].getMarker());}
                }
                ld_decay_estimator(LD_Decay_File,Marker_Map,"yes",markergenotypes);      /* Function to calculate ld decay */
                logfile << "    - Genome-wide marker LD decay." << endl;
                markergenotypes.clear();
                /* Clear previous simulation */
                fstream checkqtlldfile; checkqtlldfile.open(QTL_LD_Decay_File.c_str(), std::fstream::out | std::fstream::trunc); checkqtlldfile.close();
                string foundergen = "yes";
                /* Intialize outputfile */
                ofstream outputphase;
                outputphase.open(Phase_Persistance_Outfile.c_str());
                outputphase << "Generation PhaseCorrelation...." << endl;
                outputphase.close();
                /* Generate founder qtl lddecay */
                qtlld_decay_estimator(SimParameters,population,population_QTL,Marker_Map,foundergen,QTL_LD_Decay_File,Phase_Persistance_File,Phase_Persistance_Outfile);
                logfile << "    - QTL LD decay and Phase Persistance." << endl;
                time_t intend_time = time(0);
                logfile << "Finished Generating Genome Summary Statistics (Time: " << difftime(intend_time,intbegin_time) << " seconds)."<< endl << endl;
            }
            if(SimParameters.getOutputWindowVariance() == "yes")
            {
                string foundergen = "yes";
                WindowVariance(SimParameters,population,population_QTL,foundergen,Windowadditive_Output,Windowdominance_Output);
                logfile << "Generating Additive and Dominance Window Variance."<< endl << endl;
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
            sftrabbit::beta_distribution<> beta(SimParameters.getBetaDist_alpha(),SimParameters.getBetaDist_beta());     /* Beta Distribution */
            for (int i = 0; i < n; ++i){number[i] = beta(gen); ++p[int(nintervals*number[i])];}                  /* Get sample and put count in interval */
            /* Plots disribution based on alpha and beta */
            logfile << "Beta distribution (" << SimParameters.getBetaDist_alpha() << "," << SimParameters.getBetaDist_beta() << "):" << endl;
            for (int i=0; i<nintervals; ++i)
            {
                logfile << float(i)/nintervals << "-" << float(i+1)/nintervals << ": " << "\t" << std::string(p[i]*nstars/n,'*') << std::endl;
            }
            sort(number.begin(),number.end());
            logfile << "\nAllele Frequencies in Founder \n";
            /* Allele frequencies should be from a population that has been unselected; get allele frequency from unselected base population */
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
            double* M = new double[3*tempgenofreqcalc[0].size()];       /* Create M which is of dimension 3 by number of markers */
            float scale = 0;                                            /* Calculate Scale */
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < tempgenofreqcalc[0].size(); j++)
                {
                    M[(i*tempgenofreqcalc[0].size())+j] = i - (2 * founderfreq[j]);
                }
            }
            for (int j=0; j < tempgenofreqcalc[0].size(); j++){scale += (1 - founderfreq[j]) * founderfreq[j];}
            scale = scale * 2;
            logfile << "   - M Matrix Calculated which is based on frequencies from founder generation. " << endl;
            logfile << "   - Scale Factor used to construct G based on founder frequency: " << scale << endl << endl;
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
            /* Now generate inbreeding metrics for individuals that died and output */
            if(rightfitnessstring.size() > 0)
            {
                for(int i = 0; i < markerlowfitness.size(); i++)
                {
                    stringstream strStreammiddlefit (stringstream::in | stringstream::out);
                    strStreammiddlefit << "0" << " ";
                    string Geno = markerlowfitness[i];                              /* Grab Genotype for dead Individual i */
                    float S = 0.0;
                    for(int k = 0; k < Geno.size(); k++)                            /* Loop Across Markers to create cell within G */
                    {
                        double temp = Geno[k] - 48;                                 /* Create Z for a given SNP */
                        if(temp == 3 || temp == 4){temp = 1;}                       /* Convert 3 or 4 to 1 */
                        temp = (temp - 1) - (2 * (founderfreq[k] - 0.5));           /* Convert it to - 1 0 1 and then put it into Z format */
                        S += temp * temp;                                           /* Multipy then add to sum */
                    }
                    strStreammiddlefit << (S / double(scale)) << " ";
                    double sumhaplotypes = 0.0;
                    for(int j = 0; j < haplib.size(); j++)
                    {
                        string homo1 = (markerlowfitness[i]).substr(haplib[j].getStart(),SimParameters.gethaplo_size()); string homo2 = homo1;
                        for(int g = 0; g < homo1.size(); g++)
                        {
                            if(homo1[g] == '0'){homo1[g] = '1';} if(homo2[g] == '0'){homo2[g] = '1';}               /* a1a1 genotype */
                            if(homo1[g] == '2'){homo1[g] = '2';} if(homo2[g] == '2'){homo2[g] = '2';}               /* a2a2 genotype */
                            if(homo1[g] == '3'){homo1[g] = '1';} if(homo2[g] == '3'){homo2[g] = '2';}               /* a1a2 genotype */
                            if(homo1[g] == '4'){homo1[g] = '2';} if(homo2[g] == '4'){homo2[g] = '1';}               /* a2a1 genotype */
                        }
                        vector < int > match(homo1.size(),-5);
                        for(int g = 0; g < homo1.size(); g++){match[g] = 1 - abs(homo1[g] - homo2[g]);} /* Create match vector */
                        double sumh3 = 0.0; double hvalue = 0.0;
                        for(int g = 0; g < homo1.size(); g++){sumh3 += match[g];}
                        if(sumh3 == homo1.size()){hvalue = 2.0;}
                        if(sumh3 != homo1.size()){hvalue = 1.0;}
                        sumhaplotypes += hvalue;
                    }
                    strStreammiddlefit << (sumhaplotypes/double(haplib.size())) << " ";
                    std::ofstream outputlow(lowfitnesspath, std::ios_base::app | std::ios_base::out);
                    outputlow << leftfitnessstring[i] << (strStreammiddlefit.str()) <<  rightfitnessstring[i] << endl;
                }
            }
            leftfitnessstring.clear(); rightfitnessstring.clear(); markerlowfitness.clear();
            tempgenofreqcalc.clear();
            int TotalAnimalNumber = 0;                  /* Counter to determine how many animals their are for full matrix sizes */
            int TotalOldAnimalNumber = 0;               /* Counter to determine size of old animal matrix */
            std::ofstream outputmastgeno(Master_Genotype_File, std::ios_base::app | std::ios_base::out);
            outputmastgeno << "ID Marker QTL" << endl;
            time_t t_end = time(0);
            cout << "Constructed Trait Architecture and Founder Genomes. (Took: " << difftime(t_end,t_start) << " seconds)" << endl << endl;
            /* If doing Bayes ebv prediction create a new file and add a flag to get prior's based on h2 */
            if(SimParameters.getEBV_Calc() == "bayes")
            {
                fstream checkbayes; checkbayes.open(Bayes_PosteriorMeans.c_str(), std::fstream::out | std::fstream::trunc); checkbayes.close();
                std::ofstream flagbayes(Bayes_PosteriorMeans.c_str(), std::ios_base::app | std::ios_base::out);
                flagbayes<< "GeneratePriorH2" << endl;
            }
            ////////////////////////////////////////////////////////////////////////////////////
            //////      Loop through based on Number of Generations you want simulated    //////
            ////////////////////////////////////////////////////////////////////////////////////
            cout << "Begin Simulating Generations:" << endl;
            /* Create a vector of string that describe what selection is based on.  *
            /* Currently just change how founders are selected later change after n generation */
            vector < string > SelectionVector((SimParameters.getGener()),"");
            vector < string > EBVCalculationVector((SimParameters.getGener()),"");
            for(int i = 0; i < (SimParameters.getGener()); i++)
            {
                if(i < SimParameters.getGenfoundsel())
                {
                    SelectionVector[i] = SimParameters.getfounderselect();
                    EBVCalculationVector[i] = "NO";
                }
                if(i >= SimParameters.getGenfoundsel())
                {
                    SelectionVector[i] = SimParameters.getSelection();
                    
                    if(SimParameters.getEBV_Calc()!="SKIP")
                    {
                        EBVCalculationVector[i] = "YES";
                    } else {EBVCalculationVector[i] = "NO";}
                }
            }
            //for(int i = 0; i < SelectionVector.size(); i++){cout << SelectionVector[i] << " " << EBVCalculationVector[i] << endl;}
            for(int Gen = 1; Gen < (SimParameters.getGener() + 1); Gen++)
            {
                time_t intbegin_time = time(0);
                if(Gen > 1){TotalOldAnimalNumber = TotalAnimalNumber;}                              /* Size of old animal matrix */
                logfile << "------ Begin Generation " << Gen << " -------- " << endl;
                /* Output animals that are of age 1 into pheno_pedigree and Pheno_Gmatrix to use for relationships */
                /* That way when you read them back in to create relationship matrix don't need to order them */
                /* Save as a continuous string and then output */
                stringstream outputstringpedigree(stringstream::out);
                stringstream outputstringgenomic(stringstream::out); int outputnumpedgen = 0;
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
                    if(outputnumpedgen % 100 == 0)
                    {
                        /* Don't need to worry about pedigree file getting big */
                        /* output pheno genomic file */
                        std::ofstream output2(Pheno_GMatrix_File.c_str(), std::ios_base::app | std::ios_base::out);
                        output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
                    }
                }
                /* output pheno pedigree file */
                std::ofstream output1(Pheno_Pedigree_File.c_str(), std::ios_base::app | std::ios_base::out);
                output1 << outputstringpedigree.str(); outputstringpedigree.str(""); outputstringpedigree.clear();
                /* output pheno genomic file */
                std::ofstream output2(Pheno_GMatrix_File.c_str(), std::ios_base::app | std::ios_base::out);
                output2 << outputstringgenomic.str(); outputstringgenomic.str(""); outputstringgenomic.clear();
                /* If have ROH genome summary as an option; do proportion of genome in ROH for reach individual */
                if(SimParameters.getmblengthroh() != -5){Proportion_ROH(SimParameters,population,Marker_Map,logfile);}
                /* If want to identify haplotypes do it now */
                if(SimParameters.getstartgen() != -5 && Gen < (SimParameters.getstartgen() + SimParameters.getGenfoundsel()))
                {
                    logfile << "   Not enough generations completed to begin Haplotype Finder Algorithm" << endl << endl;
                }
                if(SimParameters.getstartgen() != -5 && Gen > (SimParameters.getstartgen() + SimParameters.getGenfoundsel()) && Gen != retraingeneration)
                {
                    logfile << "   Not a retraining generation for Haplotype Finder Algorithm" << endl << endl;
                }
                if(SimParameters.getstartgen() != -5 && Gen == retraingeneration)
                {
                    logfile << "   Begin to Identify Haplotypes in ROH associated with " << unfav_direc << " phenotypes." << endl;
                    trainregions.clear();                           /* First clear old haplotypes to start fresh */
                    /* Figure out training Generation */
                    vector < int > traingeneration;
                    for(int i = Gen-1; i > (Gen - SimParameters.gettraingen()-1); i--){traingeneration.push_back(i);}
                    //for(int i = 0; i < traingeneration.size(); i++){cout << traingeneration[i] << endl;}
                    /* Default Parameters */
                    double minimum_freq = 0.0075;
                    int null_samples = 250;
                    vector <int> width {50,45,40,35,30,25,20};
                    double one_sided_t = 2.326;
                    double phenotype_cutoff;
                    double residualvariance;
                    /* Vectors used in Program */
                    vector < int > haplo_chr;                                   /* stores chromosome in vector */
                    vector < int > haplo_position;                              /* stores position */
                    vector < int > haplo_index;                                 /* used to grab chromosome */
                    vector < CHR_Index > haplo_chr_index;                       /* Class to store chromosomal information */
                    vector < string > id;                                       /* ID */
                    vector < double > pheno;                                    /* Phenotype */
                    vector < double > trueebv;                                  /* Used to calculate lambda in MMME */
                    vector < int > phenogenorownumber;                          /* match to genotype row */
                    /* depending on how many fixed effect parameters their are and at what point need to make different sized matrices */
                    vector < vector < string > > FIXED_CLASS(0,vector <string>(0)); /* Stores Classification Fixed Effects */
                    vector < vector < string > > uniqueclass(0,vector <string>(0)); /* Number of levels within each classification variable */
                    vector < vector < double > > FIXED_COV(0,vector <double>(0));   /* Stores Covariate Fixed Effects */
                    vector < double > MeanPerCovClass(0,0.0);                       /* mean for each covariate */
                    vector < string > genotype;                         /* Genotype String */
                    vector < string > genotypeID;                       /* ID pertaining to Genotype String */
                    vector < string > uniqueID;                         /* Sets the rows and cols up for ZtZ */
                    vector <int> X_i; vector <int> X_j; vector <double> X_A;                    /* Used to store X in sparse ija format */
                    vector <int> ZW_i; vector <int> ZW_j; vector <double> ZW_A;                 /* Used to store ZW in sparse ija format */
                    vector <int> LHSred_i; vector <int> LHSred_j; vector <double> LHSred_A;     /* Used to store LHS in sparse symmetric ija format */
                    vector < double > lambda(2,0.0);
                    ReadGenoDiverMapFile_Index(Marker_Map,haplo_chr,haplo_position,haplo_index,haplo_chr_index,logfile);      /* fill map vectors */
                    ReadGenoDiverPhenoGenoFile(population,Master_DF_File,Pheno_GMatrix_File,traingeneration,id,pheno,trueebv,phenogenorownumber,genotype,genotypeID);
                    logfile << "      - Training Population Size " << id.size() << "." << endl;
                    simulationlambda(pheno,trueebv,lambda);
                    residualvariance = lambda[0]; lambda[0] = lambda[0] / double(lambda[1]); lambda[1] = 0.0;
                    subtractmean(pheno);                                /* Subtract off mean */
                    time_t fullped_begin_time = time(0);
                    for(int i = 0; i < genotypeID.size(); i++){uniqueID.push_back(genotypeID[i]);}
                    double* Relationshipinv_mkl = new double[uniqueID.size()*uniqueID.size()];
                    for(int i = 0; i < (uniqueID.size()*uniqueID.size()); i++){Relationshipinv_mkl[i] = 0.0;}
                    GenerateAinvGenoDiver(Pheno_Pedigree_File,uniqueID,id,Relationshipinv_mkl);            /* Generate Ainv for subset of animals */
                    time_t fullped_end_time = time(0);
                    logfile <<"      - Ainv: "<<uniqueID.size()<<" "<<uniqueID.size()<<" ("<<difftime(fullped_end_time,fullped_begin_time)<<" seconds)"<<endl;
                    time_t fulllhs_begin_time = time(0);
                    /* X is just a column of 1's since mean is only fixed effect */
                    int dimension = 1; int dim_lhs = dimension+uniqueID.size();
                    for(int i = 0; i < pheno.size(); i++){X_i.push_back(X_j.size()); X_j.push_back(0); X_A.push_back(1);}
                    /*** Generate LHS based on reduced Model ***/
                    /* Now know the dimension of LHS so fill LHS with appropriate columns */
                    GenerateLHSRed(X_i,X_j,X_A,dimension,Relationshipinv_mkl,ZW_i,ZW_j,ZW_A,id,uniqueID,LHSred_i,LHSred_j,LHSred_A,dim_lhs,lambda);
                    time_t fulllhs_end_time = time(0);
                    logfile<<"      - LHS reduced model: "<<dim_lhs<<"-"<<dim_lhs<<" ("<<difftime(fulllhs_end_time,fulllhs_begin_time)<<" seconds)"<<endl;
                    delete [] Relationshipinv_mkl;
                    fulllhs_end_time = time(0);
                    /***                  Generate vector add zero for contrasts                  ***/
                    vector < string > factor_red; vector < int > zero_columns_red;
                    factor_red.push_back("int"); zero_columns_red.push_back(1);
                    for(int i = 0; i < uniqueID.size(); i++){zero_columns_red.push_back(1); factor_red.push_back("Random");}
                    //cout << zero_columns_red.size() << " " << factor_red.size() << endl;
                    int min_Phenotypes = minimum_freq * id.size() + 0.5;  /* Determines Minimum Number and rounds up correctly */
                    phenotype_cutoff = phenocutoff(haplo_chr_index,null_samples,min_Phenotypes,width,genotype,phenogenorownumber,pheno,dim_lhs,X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,uniqueID,LHSred_i,LHSred_j,LHSred_A,factor_red,zero_columns_red,residualvariance,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass,unfav_direc,logfile);
                    fulllhs_end_time = time(0);
                    logfile<<"      - Minimum phenotype cutoff: "<<phenotype_cutoff<<" ("<<difftime(fulllhs_end_time,fulllhs_begin_time)<<" seconds)"<<endl;
                    logfile<<"      - Begin Looping Across Chromosomes: " << endl;
                    for(int chromo = 0; chromo < haplo_chr_index.size(); chromo++)
                    {
                        time_t chr_begin_time = time(0);
                        vector < Unfavorable_Regions > regions;                         /* vector of objects to store everything about unfavorable region */
                        vector < Unfavorable_Regions_sub > regions_sub;                 /* vector of objects to store everything about unfavorable region */
                        Step1(regions_sub,phenotype_cutoff,unfav_direc,chromo,haplo_chr_index,min_Phenotypes,width,genotype,phenogenorownumber,pheno,id,haplo_chr,haplo_position,haplo_index,regions);
                        Step2(regions,min_Phenotypes,genotype,phenogenorownumber,pheno,dim_lhs,X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,uniqueID,LHSred_i,LHSred_j,LHSred_A,factor_red,zero_columns_red,residualvariance,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass,unfav_direc,one_sided_t,phenotype_cutoff,logfile);
                        Step3(regions,pheno,genotype,phenogenorownumber,id);
                        for(int i = 0; i < regions.size(); i++)
                        {
                            Unfavorable_Regions region_temp(regions[i].getChr_R(),regions[i].getStPos_R(),regions[i].getEnPos_R(),regions[i].getStartIndex_R(),regions[i].getEndIndex_R(),regions[i].getHaplotype_R(),regions[i].getLength_R(),regions[i].getRawPheno_R(),regions[i].getEffect(),regions[i].getLSM_R(), regions[i].gettval());
                            trainregions.push_back(region_temp);
                        }
                        regions.clear(); regions_sub.clear();
                        time_t chr_end_time = time(0);
                        logfile<<"          - Finished Chromosome: "<<chromo+1<<" ("<<difftime(chr_end_time,chr_begin_time)<<" seconds)"<<endl;
                    }
                    logfile<<"      - Finished Looping Across Chromosomes (Regions " << trainregions.size() << "." << endl;
                    //for(int i = 0; i < trainregions.size(); i++)
                    //{
                    //    cout << trainregions[i].getChr_R() << " " << trainregions[i].getStPos_R() << " " << trainregions[i].getEnPos_R() << " ";
                    //    cout << trainregions[i].getStartIndex_R()<< " " <<trainregions[i].getEndIndex_R()<< " " <<trainregions[i].getHaplotype_R()<<" ";
                    //    cout << trainregions[i].getLength_R() << " " << trainregions[i].getRawPheno_R() << " " << trainregions[i].getEffect() << " ";
                    //    cout << trainregions[i].getLSM_R() << " " << trainregions[i].gettval() << endl;
                    //}
                    //cout << endl << endl;
                    sort(trainregions.begin(),trainregions.end(),sortByLength);
                    haplo_chr.clear(); haplo_position.clear(); haplo_index.clear(); haplo_chr_index.clear(); id.clear(); pheno.clear();
                    trueebv.clear(); phenogenorownumber.clear(); genotype.clear(); genotypeID.clear(); uniqueID.clear(); X_i.clear(); X_j.clear();
                    X_A.clear(); ZW_i.clear(); ZW_j.clear(); ZW_A.clear(); LHSred_i.clear(); LHSred_j.clear(); LHSred_A.clear(); lambda.clear();
                    logfile << "   Finished Identify Unfavorable Haplotypes." << endl << endl;
                }
                time_t start_block = time(0); time_t start; time_t end;
                /* Get ID for last animal; Start ID represent the first ID that will be used for first new animal */
                int StartID = population.size();
                StartID = population[StartID - 1].getID() + 1;
                vector < int > trainanimals;
                if(scalefactaddh2!=0 && EBVCalculationVector[Gen-1]=="YES" && SimParameters.getEBV_Calc()!="bayes")
                {
                    logfile << "   Generate Estimated Breeding Values based on " << SimParameters.getEBV_Calc() << " method:" << endl;
                    vector <double> estimatedsolutions; vector < double > trueaccuracy;
                    Generate_BLUP_EBV(SimParameters,population,estimatedsolutions,trueaccuracy,logfile,trainanimals,TotalAnimalNumber,TotalOldAnimalNumber,Gen,Pheno_Pedigree_File,GenotypeStatus_path,Pheno_GMatrix_File,M,scale,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,BinaryLinv_Matrix_File,haplib);
                }
                if(scalefactaddh2!=0 && EBVCalculationVector[Gen-1]=="YES" && SimParameters.getEBV_Calc()=="bayes")
                {
                    logfile << "   Generate Estimated Breeding Values utilizing bayesian regression methods." << endl;
                    logfile << "       - Bayesian Regression Method: " << SimParameters.getmethod() << endl;
                    /* Generate Breeding Values based on Bayesian regression models (i.e. Bayes A, B, C or RR) */
                    vector < double > estimatedsolutions;
                    bayesianestimates(SimParameters,population,Master_DF_File,Pheno_GMatrix_File,Pheno_Pedigree_File,Gen,estimatedsolutions,Bayes_MCMC_Samples,Bayes_PosteriorMeans,logfile);
                    estimatedsolutions.clear();
                    //for(int i = 0; i < population.size(); i++){cout << population[i].getEBV() << " ";}
                    //cout << endl;
                    Inbreeding_Pedigree(population,Pheno_Pedigree_File);
                }
                time_t end_block = time(0);
                logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
                if(scalefactaddh2 == 0 || EBVCalculationVector[Gen-1]=="NO")
                {
                    Inbreeding_Pedigree(population,Pheno_Pedigree_File);
                }
                if(SimParameters.getOutputTrainReference() == "yes")
                {
                    if(scalefactaddh2 != 0 && EBVCalculationVector[Gen-1]=="YES" && Gen > 1)
                    {
                        trainrefcor(SimParameters,population,Correlation_Output,Gen);
                    }
                }
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                   Start of Selection Functions                                  //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                matingindividuals.clear();                                   /* Make mate individual class starts out as empty */
                logfile << "   Begin " << SelectionVector[Gen-1] << " Selection of offspring: " << endl;
                time_t start_block1 = time(0);
                string tempselectionscen = SelectionVector[Gen-1];
                /* truncation selection variables */
                if(tempselectionscen=="random" || tempselectionscen=="phenotype" || tempselectionscen=="true_bv" || tempselectionscen=="ebv")
                {
                    truncationselection(population,SimParameters,tempselectionscen,Gen,Master_DF_File,Master_Genotype_File,logfile);
                }
                /* optimal contribution selection */
                if(tempselectionscen == "ocs")
                {
                    optimalcontributionselection(population,matingindividuals,haplib,SimParameters,tempselectionscen,Pheno_Pedigree_File,M,scale,Master_DF_File,Master_Genotype_File,Gen,logfile);
                }
                time_t end_block1 = time(0);
                logfile << "   Finished " << SelectionVector[Gen-1]<< " Selection of parents (Time: " << difftime(end_block1,start_block1);
                logfile << " seconds)." << endl << endl;
                /* If doing the ssgblup option now update who is genotyped or not */
                //if(SimParameters.getEBV_Calc() == "ssgblup")
                //{
                //    if(Gen >= SimParameters.getGenoGeneration()){updateanimalgenostatus(SimParameters,population,GenotypeStatus_path);}
                //}
                logfile << "   Male and Female Age Distribution: " << endl;
                breedingagedistribution(population,SimParameters,logfile);
                logfile << endl;
                //if(SelectionVector[Gen-1] != "random"){exit (EXIT_FAILURE);}
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                 Start of Mating Design Functions                                //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                logfile << "   Begin Generating Mating Design Based On Parental Generation." << endl;
                time_t start_mateblock = time(0);
                if(SelectionVector[Gen-1] != "ocs")
                {
                    /* Figure out age distribution */
                    vector <int> MF_AgeClass;                                    /* Number of animals in total for both male and female */
                    vector <int> M_AgeClass;                                     /* Number of Sires in a given age class */
                    vector <int> MF_AgeID;                                       /* ID for age class; (can skip age classes based on culling */
                    agedistribution(population,MF_AgeClass,M_AgeClass,MF_AgeID); /* Figure out Age Distribution */
                    int M_NumberClassg0 = 0;                                     /* Number of male age groups greater than 0 */
                    for(int i = 0; i < M_AgeClass.size(); i++){if(M_AgeClass[i] > 0){M_NumberClassg0++;}}
                    vector <double> CountSireMateClass(M_AgeClass.size(),0);
                    if(SimParameters.getmaxsireprop()== -5 || SelectionVector[Gen-1] == "random")     /* Sire distributed equally across dams */
                    {
                        /* Alpha and Beta of Beta Distribution both equal 1.0 */
                        if(SimParameters.getBetaDist_alpha() == 1.0 && SimParameters.getBetaDist_beta() == 1.0)
                        {
                            int temp = SimParameters.getDams() / double(SimParameters.getSires());
                            for(int i = 0; i < population.size(); i++){if(population[i].getSex() == 0){population[i].UpdateMatings(temp);}}
                        }
                        if((SimParameters.getBetaDist_alpha() != 1.0 || SimParameters.getBetaDist_beta() != 1.0))
                        {
                            if(M_NumberClassg0 == 1) /* This happens first generation */
                            {
                                int temp = SimParameters.getDams() / double(SimParameters.getSires());
                                for(int i = 0; i < population.size(); i++){if(population[i].getSex() == 0){population[i].UpdateMatings(temp);}}
                            }
                            /* more than one so need to first give proportion to parity 1 and remaining based of Beta Distribution */
                            if(M_NumberClassg0 > 1){betadistributionmates(population,SimParameters,M_NumberClassg0,number,M_AgeClass,MF_AgeID);}
                        }
                        /* Loop through females set to 0 and give them a mating of 1 */
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getSex() == 1){population[i].ZeroOutMatings(); population[i].UpdateMatings(1);}
                        }
                        outputlogsummary(population,M_AgeClass,MF_AgeID,logfile);
                        /* Now Figure out mating pairs based on a given mating design */
                        string matingscenario; string tempselectionvector = SelectionVector[Gen-1];
                        matingscenario = choosematingscenario(SimParameters,tempselectionvector);    /* Based on two variable decides which scenario */
                        /* Used to identify mating mating pairs based on a specified scenario*/
                        generatematingpairs(matingindividuals,population,haplib,SimParameters,matingscenario,Pheno_Pedigree_File,M,scale,logfile);
                    }
                    /* Sire not distributed equally across dams */
                    if(SimParameters.getmaxsireprop()!= -5 && SimParameters.getMating()=="index" && SelectionVector[Gen-1] != "random")
                    {
                        logfile << "       - Mating design based on index values which results in sire not being used equally across animals: " << endl;
                        indexmatingdesign(matingindividuals,population,haplib,SimParameters,Pheno_Pedigree_File,M,scale,logfile);
                        updatematingindex(matingindividuals,population);
                    }
                }
                if(SelectionVector[Gen-1] == "ocs")
                {
                    logfile << "       - Mating design based on random mating of selected mates from optimal contribution." << endl;
                }
                time_t end_mateblock = time(0);
                logfile << "   Finished Generating Mating Design Based On Parental Generation. (Time: " << difftime(end_mateblock,start_mateblock);
                logfile << " seconds)." << endl << endl;
                //if(SelectionVector[Gen-1] != "random"){exit (EXIT_FAILURE);}
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
                for(int i = 0; i < SimParameters.getChr(); i++)
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
                /* Find max number of SNP within a chromosome */
                for(int i = 1; i < SimParameters.getChr(); i++){if(ChrSNPLength[i] > col){col = ChrSNPLength[i];}}
                // store in 2-D vector using this may get large so need to store dynamically
                vector < vector < double > > SNPSeqPos;
                /* read in each chromsome files */
                for(int i = 0; i < SimParameters.getChr(); i++)
                {
                    vector < double > temp;
                    string mapfilepath = path + "/" + SimParameters.getOutputFold() + "/" + MapFiles[i];
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
                int gametesfemale = SimParameters.getOffspring();                         /* female can only be mated to one sire */
                logfile << "       - Number of offspring per mating pair: " << SimParameters.getOffspring() << "." << endl;
                /* Array to update old mutations allele frequency */
                vector < double > SNPFreqGen;
                for(int i = 0; i < (TotalQTL+TotalMarker); i++){SNPFreqGen.push_back(0);}     /* declare array to calculate QTL frequencies in parents */
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
                        for(int off = 0; off < SimParameters.getOffspring(); off++)
                        {
                            AnimGam_Dev.push_back(temp_dev);                        /* Store Uniform Deviate */
                            population[a].UpdateProgeny();                          /* Adds one to to progeny number for animal */
                            int genocounter = 0;                                    /* Counter for where to start next chromosome in full geno file */
                            vector < int > fullgamete((TotalQTL + TotalMarker),0);
                            AnimGam_ID.push_back(TEMPID);                           /* Store Animal ID */
                            AnimGam_GamID.push_back(g + 1);                         /* Store gamete number */
                            AnimGam_Sex.push_back(TEMPSEX);                         /* Store Sex of Animal */
                            /** Loop across chromosomes: Simulate mutations then generate gamete for each chromosome **/
                            for(int c = 0; c < SimParameters.getChr(); c++)
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
                                std::poisson_distribution<int>distribution(SimParameters.getu()*(SimParameters.get_ChrLength())[c]);
                                Nmb_Mutations = distribution(gen);
                                /* Proportion that can be QTL */
                                Nmb_Mutations = (double(Nmb_Mutations) * double(SimParameters.getPropQTL()));        /* integer */
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
                                        if((SimParameters.get_QTL_chr())[c]>0 && (SimParameters.get_FTL_lethal_chr())[c]>0 && (SimParameters.get_FTL_sublethal_chr())[c]>0)
                                        {
                                            if(temptype < 0.333){MutationType.push_back(2);}                                    /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.333 && temptype < 0.666){MutationType.push_back(4);}               /* Mutation is a Lethal Fitness QTL */
                                            if(temptype >= 0.666){MutationType.push_back(5);}                                   /* Mutation is a subLethal Fitness QTL */
                                        }
                                        if((SimParameters.get_QTL_chr())[c] > 0 && (SimParameters.get_FTL_lethal_chr())[c] == 0 && (SimParameters.get_FTL_sublethal_chr())[c] > 0)
                                        {
                                            if(temptype < 0.5){MutationType.push_back(2);}                                    /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.5){MutationType.push_back(5);}                                   /* Mutation is a subLethal Fitness QTL */
                                        }
                                        if((SimParameters.get_QTL_chr())[c] > 0 && (SimParameters.get_FTL_lethal_chr())[c] > 0 && (SimParameters.get_FTL_sublethal_chr())[c] == 0)
                                        {
                                            if(temptype < 0.5){MutationType.push_back(2);}                                    /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.5){MutationType.push_back(4);}                                   /* Mutation is a Lethal Fitness QTL */
                                        }
                                        if((SimParameters.get_QTL_chr())[c] > 0 && (SimParameters.get_FTL_lethal_chr())[c] == 0 && (SimParameters.get_FTL_sublethal_chr())[c] == 0)
                                        {
                                            MutationType.push_back(2);
                                        }
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
                                            /******* QTL Additive Effect (Gamma) *******/
                                            std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape(),SimParameters.getGamma_Scale());
                                            MutationAdd_quan.push_back(distribution1(gen));
                                            /****** QTL Dominance Effect *******/
                                            /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                                            std::normal_distribution<double>distribution2(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
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
                                            std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape_Lethal(),SimParameters.getGamma_Scale_Lethal());
                                            MutationAdd_fit.push_back(distribution1(gen));
                                            /******      QTL h Effect (i.e. degree of dominance)     *******/
                                            /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                                            std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom_Lethal(),SimParameters.getNormal_varRelDom_Lethal());
                                            double temph = distribution2(gen);
                                            MutationDom_fit.push_back(abs(temph));
                                            MutationAdd_quan.push_back(0.0); MutationDom_quan.push_back(0.0);
                                        }
                                        if(MutationType[CounterMutationIndex] == 5)
                                        {
                                            /*******     QTL s effect (i.e. selection coeffecient)   *******/
                                            std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape_SubLethal(),SimParameters.getGamma_Scale_SubLethal());
                                            MutationAdd_fit.push_back(distribution1(gen));
                                            /******      QTL h Effect (i.e. degree of dominance)     *******/
                                            /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                                            std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom_SubLethal(),SimParameters.getNormal_varRelDom_SubLethal());
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
                                double ugamete;                 /* random uniform[0,1] derivate  */
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
                                    if(SimParameters.getRecombDis() == "Uniform")
                                    {
                                        /* For each recombination event sample to get location */
                                        for(int countCx = 0; countCx < nCx; ++countCx)
                                        {
                                            std::uniform_real_distribution<double> distribution7(0,1.0);
                                            lCx[countCx] = distribution7(gen);
                                        }
                                    }
                                    if(SimParameters.getRecombDis() == "Beta")
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
                                ugamete = distribution8(gen);
                                /*  if ugamete < 0.5 point to homologe1 else to homologe2 */
                                if(ugamete < 0.5){currentHomologe_ptr = homo1; pointingAt = 1;}
                                if(ugamete >= 0.5){currentHomologe_ptr = homo2; pointingAt = 2;}
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
                for(int i = 0; i < SimParameters.getChr(); i++){SNPSeqPos[i].clear();}
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
                            QTL_new_old tempa(MutationLoc[i], MutationAdd_quan[i], MutationDom_quan[i], temp, Gen, stringfreq,"");
                            population_QTL.push_back(tempa);
                        }
                        if(MutationType[i] == 4 || MutationType[i] == 5)
                        {
                            QTL_new_old tempa(MutationLoc[i], MutationAdd_fit[i], MutationDom_fit[i], temp, Gen, stringfreq,"");
                            population_QTL.push_back(tempa);
                        }
                    }
                    logfile << "       - New QTL's Added to QTL class object (Total: " << population_QTL.size() << ")." << endl;
                    /* Update number of qtl and markers per chromosome */
                    for(int i = 0; i < SimParameters.getChr(); i++)
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
                //if(SelectionVector[Gen-1] != "random"){exit (EXIT_FAILURE);}
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////                       Generate Progeny For Next Generation                      ///////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
                logfile << "   Begin generating offspring from parental gametes and mating design: " << endl;
                time_t start_offspring = time(0);
                int LethalFounder = 0;                                  /* Offspring that didn't survive */
                /* Increase age of parents by one */
                for(int i = 0; i < population.size(); i++){population[i].UpdateAge();}
                vector < int > gametefound(AnimGam_ID.size(),0); int matingpairs = 1;
                for(int gamete = 0; gamete < AnimGam_ID.size(); gamete++)
                {
                    if(gametefound[gamete] == 0)                    /* gamete hasn't found mating pair yet */
                    {
                        /* find animal in matingindividuals class */
                        int searchingindex = 0;
                        while(searchingindex < matingindividuals.size())
                        {
                            if(AnimGam_ID[gamete] == matingindividuals[searchingindex].getID_MC() && gametefound[gamete] == 0)
                            {
                                int mate = 0; /* starts off at first mate id and if both -5 moves on to next one until reach end */
                                while(mate < (matingindividuals[searchingindex].get_mateIDs()).size())
                                {
                                    int searchingindexmate = 0;
                                    while(1)
                                    {
                                        
                                        if(AnimGam_ID[searchingindexmate] == (matingindividuals[searchingindex].get_mateIDs())[mate] && gametefound[searchingindexmate] == 0)
                                        {
                                            //cout << gamete<< " " <<AnimGam_ID[gamete] << " " << AnimGam_GamID[gamete] << " " << AnimGam_Sex[gamete] << " ";
                                            //cout << gametefound[gamete] << " -- " << matingindividuals[searchingindex].getID_MC() << " ";
                                            //cout << (matingindividuals[searchingindex].get_mateIDs())[mate] << " -- ";
                                            //cout << searchingindexmate << " " << AnimGam_ID[searchingindexmate] << " " << AnimGam_GamID[gamete] << " ";
                                            //cout << AnimGam_Sex[searchingindexmate] << " " << gametefound[searchingindexmate] << " " << matingpairs << endl;
                                            /* Now generate individuals */
                                            string maleGame, femaleGame;
                                            int tempsireid, tempdamid;
                                            if(AnimGam_Sex[gamete] == 0){maleGame = AnimGam_Gam[gamete]; tempsireid = AnimGam_ID[gamete];}
                                            if(AnimGam_Sex[gamete] == 1){femaleGame = AnimGam_Gam[gamete]; tempdamid = AnimGam_ID[gamete];}
                                            if(AnimGam_Sex[searchingindexmate] == 0)
                                            {
                                                maleGame = AnimGam_Gam[searchingindexmate]; tempsireid = AnimGam_ID[searchingindexmate];
                                            }
                                            if(AnimGam_Sex[searchingindexmate] == 1)
                                            {
                                                femaleGame = AnimGam_Gam[searchingindexmate]; tempdamid = AnimGam_ID[searchingindexmate];
                                            }
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
                                            /* MarkerGeno array contains Markers and QTL therefore need to split them off based on index arrays */
                                            vector < int > MarkerGenotypes(NUMBERMARKERS,0);        /* Marker Genotypes; Will always be of this size */
                                            vector < int > QTLGenotypes(QTL_Index.size(),0);        /* QTL Genotypes */
                                            m_counter = 0;                                          /* Counter to keep track where at in Marker Index */
                                            qtl_counter = 0;                                        /* Counter to keep track where at in QTL Index */
                                            /* Fill Genotype Array */
                                            for(int j = 0; j < (TotalQTL + TotalMarker + CounterMutationIndex); j++) /* Place Genotypes based on Index value */
                                            {
                                                if(m_counter < MarkerIndex.size())                  /* ensures doesn't go over and cause valgrind error */
                                                {
                                                    if(j == MarkerIndex[m_counter]){MarkerGenotypes[m_counter] = Geno[j]; m_counter++;}
                                                }
                                                if(qtl_counter < QTL_Index.size())                  /* ensures doesn't go over and cause valgrind error */
                                                {
                                                    if(j == QTL_Index[qtl_counter]){QTLGenotypes[qtl_counter] = Geno[j]; qtl_counter++;}
                                                }
                                            }
                                            Geno.clear();
                                            /* Before you put the individual in the founder population need to determine if it dies */
                                            /* represents the multiplicative fitness effect across lethal and sub-lethal alleles */
                                            double relativeviability = 1.0;             /* Starts of as a viability of 1.0 */
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
                                            if(draw > relativeviability)                                    /* Animal Died due to a low fitness */
                                            {
                                                
                                                /* Variables that need calculated */
                                                int homozygouscount_lethal=0; int homozygouscount_sublethal=0;
                                                int heterzygouscount_lethal=0; int heterzygouscount_sublethal=0;
                                                double lethalequivalent = 0.0; double GenotypicValue = 0.0; double genomic_f = 0.0; double pedigree_f;
                                                double BreedingValue = 0.0; double DominanceDeviation = 0.0; double Homoz = 0.0; double roh_f = 0.0;
                                                /* Need to add one to dead progeny for sire and dam */
                                                int j = 0;
                                                while(j < population.size())
                                                {
                                                    if(population[j].getID() == tempsireid){population[j].Update_Dead(); break;}
                                                    j++;
                                                }
                                                j = 0;
                                                while(j < population.size())
                                                {
                                                    if(population[j].getID() == tempdamid){population[j].Update_Dead(); break;}
                                                    j++;
                                                }
                                                /* Fitness Summary Statistics */
                                                for (int j = 0; j < qtl_counter; j++)
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
                                                /* Quantititative Summary Statistics */
                                                for (int j = 0; j < qtl_counter; j++)
                                                {
                                                    if(QTL_Type[j] == 2 || QTL_Type[j] == 3)            /* Quantitative QTL */
                                                    {
                                                        int tempgeno;
                                                        if(QTLGenotypes[j] == 0 || QTLGenotypes[j] == 2){tempgeno = QTLGenotypes[j];}
                                                        if(QTLGenotypes[j] == 3 || QTLGenotypes[j] == 4){tempgeno = 1;}
                                                        BreedingValue += tempgeno * double(QTL_Add_Quan[j]);
                                                        if(tempgeno != 1){GenotypicValue += tempgeno * double(QTL_Add_Quan[j]);}
                                                        if(tempgeno == 1)
                                                        {
                                                            GenotypicValue += (tempgeno * QTL_Add_Quan[j]) + QTL_Dom_Quan[j];
                                                            DominanceDeviation += QTL_Dom_Quan[j];
                                                        }
                                                    }
                                                }
                                                /* Calculate Homozygosity only in the MARKERS */
                                                for(int j = 0; j < m_counter; j++){if(MarkerGenotypes[j] == 0 || MarkerGenotypes[j] == 2){Homoz += 1;}}
                                                Homoz = (Homoz/(m_counter));
                                                /* Calculate Genomic Inbreeding */
                                                float S = 0.0;
                                                for(int k = 0; k < m_counter; k++)
                                                {
                                                    double temp = MarkerGenotypes[k];                           /* Create Z for a given SNP */
                                                    if(temp == 3 || temp == 4){temp = 1;}                       /* Convert 3 or 4 to 1 */
                                                    temp = (temp - 1) - (2 * (founderfreq[k] - 0.5));
                                                    S += temp * temp;                                           /* Multipy then add to sum */
                                                }
                                                genomic_f = (S / double(scale));
                                                /* Calculate pedigree Inbreeding */
                                                pedigree_f = lethal_pedigree_inbreeding(Pheno_Pedigree_File,tempsireid,tempdamid);
                                                /* Calculate ROH Inbreeding */
                                                stringstream strStreamM (stringstream::in | stringstream::out);
                                                for (int j=0; j < m_counter; j++){strStreamM << MarkerGenotypes[j];}
                                                string markerlowfitness = strStreamM.str();
                                                double sumhaplotypes = 0.0;
                                                for(int j = 0; j < haplib.size(); j++)
                                                {
                                                    string homo1 = markerlowfitness.substr(haplib[j].getStart(),SimParameters.gethaplo_size());
                                                    string homo2 = homo1;
                                                    for(int g = 0; g < homo1.size(); g++)
                                                    {
                                                        if(homo1[g] == '0'){homo1[g] = '1';} if(homo2[g] == '0'){homo2[g] = '1';}   /* a1a1 genotype */
                                                        if(homo1[g] == '2'){homo1[g] = '2';} if(homo2[g] == '2'){homo2[g] = '2';}   /* a2a2 genotype */
                                                        if(homo1[g] == '3'){homo1[g] = '1';} if(homo2[g] == '3'){homo2[g] = '2';}   /* a1a2 genotype */
                                                        if(homo1[g] == '4'){homo1[g] = '2';} if(homo2[g] == '4'){homo2[g] = '1';}   /* a2a1 genotype */
                                                    }
                                                    vector < int > match(homo1.size(),-5);
                                                    for(int g = 0; g < homo1.size(); g++){match[g] = 1 - abs(homo1[g] - homo2[g]);} /* Create match vector */
                                                    double sumh3 = 0.0; double hvalue = 0.0;
                                                    for(int g = 0; g < homo1.size(); g++){sumh3 += match[g];}
                                                    if(sumh3 == homo1.size()){hvalue = 2.0;}
                                                    if(sumh3 != homo1.size()){hvalue = 1.0;}
                                                    roh_f += hvalue;
                                                }
                                                roh_f = (roh_f/double(haplib.size()));
                                                /* grab fitness trait loci */
                                                stringstream strStreamQf (stringstream::in | stringstream::out);
                                                for(int j=0; j < qtl_counter; j++)
                                                {
                                                    if(QTL_Type[j] == 3 || QTL_Type[j] == 4 || QTL_Type[j] == 5){strStreamQf << QTLGenotypes[j];}
                                                }

                                                std::ofstream output1(lowfitnesspath, std::ios_base::app | std::ios_base::out);
                                                output1 << tempsireid << " " << tempdamid << " " << Gen << " " << pedigree_f << " " << genomic_f << " ";
                                                output1 << roh_f << " " << Homoz << " " << homozygouscount_lethal << " " << heterzygouscount_lethal << " ";
                                                output1 << homozygouscount_sublethal << " " << heterzygouscount_sublethal << " ";
                                                output1 << lethalequivalent << " " << relativeviability << " ";
                                                output1 << GenotypicValue << " " << BreedingValue << " " << DominanceDeviation << " " << strStreamQf.str();
                                                output1 << endl;
                                                LethalFounder++;
                                            }
                                            if(draw < relativeviability)                                                    /* Animal Survived */
                                            {
                                                //////////////////////////////////////////////////////////////////////////
                                                // Step 8: Create founder file that has everything set for Animal Class //
                                                //////////////////////////////////////////////////////////////////////////
                                                // Set up paramters for Animal Class
                                                /* Declare Variables */
                                                double GenotypicValue = 0.0;                /* Stores Genotypic value; resets to zero for each line */
                                                double BreedingValue = 0.0;                 /* Stores Breeding Value; resets to zero for each line */
                                                double DominanceDeviation = 0.0;            /* Stores Dominance Deviation; resets to zero for each line */
                                                double Residual = 0.0;                      /* Stores Residual Value; resets to zero for each line */
                                                double Phenotype = 0.0;                     /* Stores Phenotype; resets to zero for each line */
                                                double Homoz = 0.0;                         /* Stores homozygosity based on marker information */
                                                double DiagGenoInb = 0.0;                   /* Diagonal of Genomic Relationship Matrix */
                                                double sex;                                 /* draw from uniform to determine sex */
                                                int Sex;                                    /* Sex of the animal 0 is male 1 is female */
                                                /* Residual Variance; Total Variance equals 1 */
                                                double residvar = 1-(SimParameters.getVarAdd()+SimParameters.getVarDom());
                                                residvar = sqrt(residvar);                  /* random number generator need standard deviation */
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
                                                        /* Not a heterozygote so only a function of additive */
                                                        if(tempgeno != 1){GenotypicValue += tempgeno * QTL_Add_Quan[j];}
                                                        if(tempgeno == 1) /* Heterozygote so need to include add and dom */
                                                        {
                                                            GenotypicValue += (tempgeno * QTL_Add_Quan[j]) + QTL_Dom_Quan[j];
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
                                                Animal animal(StartID,tempsireid,tempdamid,Sex,Gen,1,0,0,0,rndselection,rndculling,0.0,diagInb,0.0,0.0,0.0,homozygouscount_lethal, heterzygouscount_lethal, homozygouscount_sublethal, heterzygouscount_sublethal,lethalequivalent,Homoz,0.0,0.0,Phenotype,relativeviability,GenotypicValue,BreedingValue,DominanceDeviation,Residual,MA,QT,"","","",0.0,"");
                                                /* Then place given animal object in population to store */
                                                population.push_back(animal);
                                                StartID++;                              /* Increment ID by one for next individual */
                                            }
                                            gametefound[searchingindexmate] =-5; gametefound[gamete] =-5; matingpairs++;
                                            mate = (matingindividuals[searchingindex].get_mateIDs()).size();
                                            break;
                                        }
                                        if(AnimGam_ID[searchingindexmate] != (matingindividuals[searchingindex].get_mateIDs())[mate] || gametefound[searchingindexmate] == -5)
                                        {
                                            searchingindexmate++;
                                        }
                                        if(searchingindexmate >= gametefound.size()){break;}
                                    }
                                    mate++;
                                }
                            }
                            if(AnimGam_ID[gamete] != matingindividuals[searchingindex].getID_MC() || gametefound[gamete] == -5){searchingindex++;}
                        }
                    }
                }
                /* Determine whether animal is genotyped or not */
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1)
                    {
                        /* pblup implies no animals are genotyped */
                        if(SimParameters.getEBV_Calc() == "pblup"){population[i].UpdateGenoStatus("No");}
                        /* gblup, rohblup or bayes implies all animals are genotyped with (i.e. "Full") */
                        if(SimParameters.getEBV_Calc() == "gblup" || SimParameters.getEBV_Calc() == "rohblup" || SimParameters.getEBV_Calc() == "bayes")
                        {
                            population[i].UpdateGenoStatus("Full");
                        }
                        /* ssgblup implies portion of animals may have been genotyped. Some will have "No" and others will have either "Full" or "Reduced" */
                        if(SimParameters.getEBV_Calc() == "ssgblup")
                        {
                            /* initialize to not genotyped; can change depending on the genotype strategy */
                            population[i].UpdateGenoStatus("No");
                        }
                    }
                }
                logfile << "       - Number of Progeny that Died due to fitness: " << LethalFounder << endl;
                logfile << "       - Size of population after progeny generated: " << population.size() << endl;
                NumDeadFitness[Gen] = LethalFounder;
                time_t end_offspring = time(0);
                logfile << "   Finished generating offspring from parental gametes and mating design: (Time: ";
                logfile << difftime(end_offspring,start_offspring) << " seconds)." << endl << endl;
                /* Put newly created progeny in GenoStatus file; Save as a continuous string and then output */
                stringstream outputstringgenostatus(stringstream::out);
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1){outputstringgenostatus << population[i].getID() << " " << population[i].getGenoStatus() << endl;}
                }
                /* output genostatus file */
                std::ofstream outputg(GenotypeStatus_path.c_str(), std::ios_base::app | std::ios_base::out);
                outputg << outputstringgenostatus.str(); outputstringgenostatus.str(""); outputstringgenostatus.clear();
                /* If doing ssgblup see if animal genotype status needs to get updated */
                if(SimParameters.getEBV_Calc() == "ssgblup")
                {
                    if(Gen > SimParameters.getGenoGeneration()){updateanimalgenostatus(SimParameters,population,GenotypeStatus_path);}
                }
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Update Haplotype Library based on new animals and compute diagonals of relationship matrix                //
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                logfile << "   Begin Creating Haplotype Library and assigning haplotypes IDs to individuals: " << endl;
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
                            string temp = (population[j].getMarker()).substr(haplib[i].getStart(),SimParameters.gethaplo_size());       /* Grab specific haplotype */
                            string homo1 = temp;                                                  /* Paternal haplotypes */
                            string homo2 = temp;                                                  /* Maternal haplotypes */
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
                logfile << "   Finished Creating Haplotype Library and assigning haplotypes IDs to individuals (Time: ";
                logfile << difftime(end_block4,start_block4) << " seconds)." << endl << endl;
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                    Start of Culling Functions                                   //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                time_t start_block5 = time(0);
                string tempcullingscen = SelectionVector[Gen-1];
                if(SelectionVector[Gen-1] == "random"){logfile<<"   Begin "<<SelectionVector[Gen-1]<<" Culling: "<<endl;}
                if(SelectionVector[Gen-1] != "random"){logfile << "   Begin " << SimParameters.getCulling() << " Culling: " << endl;}
                if(SimParameters.getSireRepl() == 1.0 && SimParameters.getDamRepl() == 1.0)
                {
                    discretegenerations(population,SimParameters,tempcullingscen,Gen,Master_DF_File,Master_Genotype_File,logfile);
                }
                if(SimParameters.getSireRepl() < 1.0 || SimParameters.getDamRepl() < 1.0)
                {
                    overlappinggenerations(population,SimParameters,tempcullingscen,Gen,Master_DF_File,Master_Genotype_File,logfile);
                    /* Need to give old animals that were kept a new culling deviate because if they randomly get a small one will stick around */
                    if(SimParameters.getCulling() == "random" || tempcullingscen != "random")
                    {
                        std::uniform_real_distribution<double> distribution5(0,1);
                        for(int i = 0; i < population.size(); i++)
                        {
                            if(population[i].getAge() > 1){double temp = distribution5(gen); population[i].UpdateRndCulling(temp);}
                        }
                    }
                }
                time_t end_block5 = time(0);
                if(SelectionVector[Gen-1] == "random")
                {
                    logfile << "   Finished " << SelectionVector[Gen-1] << " culling of parents (Time: " << difftime(end_block5,start_block5);
                    logfile << " seconds)." << endl << endl;
                }
                if(SelectionVector[Gen-1] != "random")
                {
                    logfile<<"   Finished " << SimParameters.getCulling() << " culling of parents (Time: " << difftime(end_block5,start_block5) << " seconds).";
                    logfile << endl << endl;
                }
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                    Housekeeping Functions                                       //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                /* Calculated expected heterozygosity in progeny */
                double expectedhet = 0.0; vector < string > population_marker;
                for(int i = 0; i < population.size(); i++)
                {
                    if(population[i].getAge() == 1){population_marker.push_back(population[i].getMarker());}
                }
                double* tempfreqexphet = new double[population_marker[0].size()];   /* Array that holds SNP frequencies that were declared as Markers*/
                frequency_calc(population_marker, tempfreqexphet);                  /* Function to calculate snp frequency */
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
                for(int j = 0; j < (population[0].getQTL()).size(); j++)                /* Loops across old QTL until map position matches up */
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
                if(SimParameters.getLDDecay() == "yes")
                {
                    logfile << "   Generate Genome Summary Statistics: " << endl;
                    /* Vector of string of markers */
                    vector < string > markergenotypes;
                    for(int i = 0; i < population.size(); i++)
                    {
                        if(population[i].getAge() == 1){markergenotypes.push_back(population[i].getMarker());}
                    }
                    ld_decay_estimator(LD_Decay_File,Marker_Map,"no",markergenotypes);      /* Function to calculate ld decay */
                    logfile << "       - Genome-wide marker LD decay." << endl;
                    markergenotypes.clear();
                    string foundergen = "no";
                    qtlld_decay_estimator(SimParameters,population,population_QTL,Marker_Map,foundergen,QTL_LD_Decay_File,Phase_Persistance_File,Phase_Persistance_Outfile);
                    logfile << "       - QTL LD decay and Phase Persistance." << endl;
                    time_t intend_time = time(0);
                    logfile<<"   Finished Generating Genome Summary Statistics (Time: "<<difftime(intend_time,intbegin_time)<<" seconds)."<< endl<<endl;
                }
                if(trainregions.size() > 0)
                {
                    vector < double > progenyphenotype; vector < double > progenytgv; vector < double > progenytbv; vector < double > progenytdd;
                    vector < string > progenygenotype; vector < double > inbreedingload; vector < double > outcorrelations(4,0);
                    for(int i = 0; i < population.size(); i++)
                    {
                        if(population[i].getAge() == 1)
                        {
                            progenyphenotype.push_back(population[i].getPhenotype()); progenytgv.push_back(population[i].getGenotypicValue());
                            progenytbv.push_back(population[i].getBreedingValue()); progenytdd.push_back(population[i].getDominanceDeviation());
                            progenygenotype.push_back(population[i].getMarker()); inbreedingload.push_back(0.0);
                        }
                    }
                    calculate_IIL(inbreedingload,progenygenotype,trainregions,unfav_direc);     /* Calculate inbreeding load based on training regions */
                    calculatecorrelation(inbreedingload,progenyphenotype,progenytgv,progenytbv,progenytdd,outcorrelations); /* Calculate Correlation */
                    logfile << "   Correlation between phenotype and haplofinder inbreeding load " << outcorrelations[0] << "." << endl << endl;
                    std::ofstream outsummaryhap(Summary_Haplofinder, std::ios_base::app | std::ios_base::out);
                    outsummaryhap<<Gen<<" "<<outcorrelations[0]<<" "<<outcorrelations[1]<<" "<<outcorrelations[2]<<" "<<outcorrelations[3]<<endl;
                }
                for(int i = 0; i < (SimParameters.get_rohgeneration()).size(); i++)
                {
                    if(Gen == (SimParameters.get_rohgeneration())[i])
                    {
                        time_t start_roh = time(0);
                        logfile << "   Generated summary statistics of ROH levels across the genome.";
                        Genome_ROH_Summary(SimParameters,population,Marker_Map,Summary_ROHGenome_Length,Summary_ROHGenome_Freq,Gen,logfile);
                        time_t end_roh = time(0);
                        logfile << " (Time: " << difftime(end_roh,start_roh) << " seconds)." << endl << endl;
                    }
                }
                if(SimParameters.getOutputTrainReference() == "yes")
                {
                    if(Gen == 1)
                    {
                        ofstream outputamax;
                        outputamax.open(Amax_Output.c_str());
                        outputamax << "Generation AvgAmax...." << endl;
                        outputamax.close();
                        if(SimParameters.getEBV_Calc() != "SKIP")
                        {
                            ofstream outputparentprogenycor;
                            outputparentprogenycor.open(Correlation_Output.c_str());
                            outputparentprogenycor << "Generation Cor_Parent_TGV Cor_Parent_TBV Cor_Parent_TDD Bias_Parent_TBV Cor_Progeny_TGV Cor_Progeny_TBV Cor_Progeny_TDD Bias_Progeny_TBV" << endl;
                            outputparentprogenycor.close();
                        }
                    }
                    getamax(SimParameters,population,Master_DF_File,Pheno_Pedigree_File,Amax_Output);
                    logfile << "   Generated Amax between recent generation and previous generations." << endl << endl;
                }
                if(SimParameters.getOutputWindowVariance() == "yes")
                {
                    string foundergen = "no";
                    WindowVariance(SimParameters,population,population_QTL,foundergen,Windowadditive_Output,Windowdominance_Output);
                    logfile << "   Generating Additive and Dominance Window Variance."<< endl << endl;
                }
                if(SimParameters.getEBV_Calc() == "ssgblup"){breedingpopulationgenotyped(population,logfile);}
                logfile << endl;
                time_t intend_time = time(0);
                cout << "   - Finished Generation " << Gen << " (Took: ";
                cout << difftime(intend_time,intbegin_time) << " seconds)" << endl;
            }
            cout << "Finished Simulating Generations" << endl;
            logfile << "------ Finished Simulating Generations --------" << endl;
            /* Clear old Simulation Files */
            fstream checkmasterdataframe;
            checkmasterdataframe.open(Master_DataFrame_path, std::fstream::out | std::fstream::trunc);
            checkmasterdataframe.close();
            /* if on last generation output all population but before update Inbreeding values and Calculate EBV's
             /* Output animals that are of age 1 into pheno_pedigree to use for pedigree relationship */
            /* That way when you read them back in to create relationship matrix don't need to order them */
            TotalOldAnimalNumber = TotalAnimalNumber;          /* Size of old animal matrix */
            using Eigen::MatrixXd; using Eigen::VectorXd;
            VectorXd lambda(1);                                 /* Declare scalar vector for alpha */
            double scalinglambda = (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd()); /* Shrinkage Factor for MME */
            lambda(0) = (1 - SimParameters.getVarAdd()) / double(SimParameters.getVarAdd());        /* Shrinkage Factor for MME */
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
            vector < double > estimatedsolutions((TotalAnimalNumber + 1),0); vector < double > trueaccuracy((TotalAnimalNumber),0);
            vector < int > trainanimals;
            /* Initialize to zero */
            if(scalefactaddh2 != 0 && SimParameters.getEBV_Calc() != "bayes" && SimParameters.getEBV_Calc()!="SKIP")
            {
                time_t start_block = time(0);
                cout << "Estimate Final Breeding Values For Last Generation" << endl;
                logfile<<"   Generate EBV based on "<< SimParameters.getEBV_Calc()<<" information for last generation:"<<endl;
                int Gen = SimParameters.getGener();
                Generate_BLUP_EBV(SimParameters,population,estimatedsolutions,trueaccuracy,logfile,trainanimals,TotalAnimalNumber,TotalOldAnimalNumber,Gen,Pheno_Pedigree_File,GenotypeStatus_path,Pheno_GMatrix_File,M,scale,BinaryG_Matrix_File,Binarym_Matrix_File,Binaryp_Matrix_File,BinaryGinv_Matrix_File,BinaryLinv_Matrix_File,haplib);
                time_t end_block = time(0);
                logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
            }
            if(scalefactaddh2 == 0 || (scalefactaddh2 != 0 && SimParameters.getEBV_Calc() == "bayes") || SimParameters.getEBV_Calc()=="SKIP")
            {
                Inbreeding_Pedigree(population,Pheno_Pedigree_File);
            }
            if(SimParameters.getOutputTrainReference() == "yes")
            {
                if(scalefactaddh2 != 0 && SimParameters.getEBV_Calc()!="SKIP")
                {
                    int tempgen = SimParameters.getGener()+1;
                    trainrefcor(SimParameters,population,Correlation_Output,tempgen);
                }
            }
            cout << "   Generating Master Dataframe and Summmary Statistics." << endl;
            /* If have ROH genome summary as an option; do proportion of genome in ROH for reach individual */
            if(SimParameters.getmblengthroh() != -5){Proportion_ROH(SimParameters,population,Marker_Map,logfile);}
            logfile << "   Creating Master File." << endl;
            stringstream outputstring(stringstream::out);
            stringstream outputstringgeno(stringstream::out); int outputnumpart1 = 0;
            for(int i = 0; i < population.size(); i++)
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
                if(SimParameters.getOutputGeno() == "yes" && SimParameters.getGener() >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() <<" "<< endl;
                    outputnumpart1++;
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
            outputstring << "Homosublethal Hetersublethal Letequiv Homozy PropROH Fitness Phen EBV Acc GV BV DD R\n";
            int linenumbera = 0; int outputnumpart2 = 0;
            ifstream infile3;
            infile3.open(Master_DF_File);
            if(infile3.fail()){cout << "Error Opening File With Animal Information!\n"; exit (EXIT_FAILURE);}
            while (getline(infile3,line))
            {
                vector <string> lineVar;
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
                int templine = stoi(lineVar[0]);
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
                            outputstring << trueaccuracy[(templine-1)] << " ";
                        } else {outputstring << 0.0 << " ";}
                    }
                    if(SimParameters.getEBV_Calc()=="gblup")
                    {
                        outputstring << estimatedsolutions[templine] << " ";
                        if(SimParameters.getSolver() == "direct" && SimParameters.getConstructGFreq()=="founder"){
                            outputstring << trueaccuracy[(templine-1)] << " ";
                        } else {outputstring << 0.0 << " ";}
                    }
                    if(SimParameters.getEBV_Calc()=="rohblup")
                    {
                        outputstring << estimatedsolutions[templine] << " ";
                        if(SimParameters.getSolver() == "direct"){
                            outputstring << trueaccuracy[(templine-1)] << " ";
                        } else {outputstring << 0.0 << " ";}
                    }
                    if(SimParameters.getEBV_Calc()=="ssgblup")
                    {
                        outputstring << estimatedsolutions[templine] << " ";
                        if(SimParameters.getSolver() == "direct"){
                            outputstring << trueaccuracy[(templine-1)] << " ";
                        } else {outputstring << 0.0 << " ";}
                    }
                    
                } else {outputstring << 0.0 << " " << 0.0 << " ";}
                outputstring << lineVar[24] << " " << lineVar[25] << " " << lineVar[26] << " " << lineVar[27] << endl;
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
            /* Generate QTL summary Stats */
            int totalgroups = SimParameters.getGener() + 1;
            generatesummaryqtl(Pheno_GMatrix_File, qtl_class_object, Summary_QTL_path,totalgroups,ID_Gen,AdditiveVar,DominanceVar,NumDeadFitness);
            generatessummarydf(Master_DataFrame_path,Summary_DF_path,totalgroups,ExpectedHeter);
            /* Make location be chr and pos instead of in current format to make it easier for user. */
            vector <string> numbers;
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
                qtlout_pos[i-1] = (stof(temp.c_str())-qtlout_chr[i-1])*(SimParameters.get_ChrLength())[qtlout_chr[i-1]-1]; /* convert to nuclotides */
                restofit[i-1] = numbers[i];
            }
            ofstream output10;
            output10.open (qtl_class_object);
            output10 << "Chr Pos Additive_Selective Dominance Type Gen Freq" << endl;
            for(int i = 1; i < numbers.size(); i++){output10 << qtlout_chr[i-1] << " " << qtlout_pos[i-1] << " " << restofit[i-1] << endl;}
            output10.close();
            /* Remove Master_DF_File since it is already in Master_DataFrame */
            string removedf = "rm " + Master_DF_File;
            system(removedf.c_str());
            delete [] M;                                    /* Now can delete the M matrix once simualtion is done */
            delete [] founderfreq;                          /* Now can delete founder frequencies once simualtion is done */
            time_t repend_time = time(0);
            if(SimParameters.getReplicates() > 1)
            {
                cout.setf(ios::fixed);
                cout << setprecision(2) << endl << "Replicate " << reps + 1 << " has completed normally (Took: ";
                cout << difftime(repend_time,repbegin_time) / 60 << " minutes)" << endl << endl;
                cout.unsetf(ios::fixed);
            }
        }
        if(SimParameters.getEBV_Calc() == "ssgblup")
        {
            int numgenoanimals = getTotalGenotypeCount(GenotypeStatus_path);
            logfile<<"   Number of animals genotyped in simulation: " << numgenoanimals << endl;
        }
        /* Remove files that aren't needed */
        command = "rm -rf " + GenotypeStatus_path + " || true"; system(command.c_str());
        /* If you have multiple replicates create a new directory within this folder to store them and just attach seed afterwards */
        if(SimParameters.getReplicates() > 1)
        {
            if(reps == 0)
            {
                /* First delete replicates folder if exists */
                string systemcall = "rm -rf " + path + "/" + SimParameters.getOutputFold() + "/replicates || true";
                system(systemcall.c_str());
                systemcall = "mkdir " + path + "/" + SimParameters.getOutputFold() + "/replicates";
                system(systemcall.c_str());
            }
            /* make seed a string */
            stringstream stringseed; stringseed << SimParameters.getSeed(); string stringseednumber = stringseed.str();
            string systemcall = "mv " + logfileloc + " " + path + "/" + SimParameters.getOutputFold() + "/replicates/" + "log_file_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+lowfitnesspath+" " + path + "/" + SimParameters.getOutputFold() + "/replicates/" + "Low_Fitness_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+Master_DataFrame_path+" " + path + "/" + SimParameters.getOutputFold() + "/replicates/" + "Master_DataFrame_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+Master_Genotype_File+" " + path + "/" + SimParameters.getOutputFold() + "/replicates/" + "Master_Genotypes_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+qtl_class_object+ " " + path + "/" + SimParameters.getOutputFold() + "/replicates/" + "QTL_new_old_Class_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+Marker_Map + " "+path + "/" + SimParameters.getOutputFold() + "/replicates/" + "Marker_Map_" + stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+Summary_QTL_path+" " + path + "/" + SimParameters.getOutputFold() + "/replicates/" + "Summary_Statistics_QTL_"+stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+Summary_DF_path+"_Inbreeding "+path+"/"+SimParameters.getOutputFold()+"/replicates/"+"Summary_Statistics_DataFrame_Inbreeding_"+stringseednumber;
            system(systemcall.c_str());
            systemcall = "mv "+Summary_DF_path+"_Performance "+path+"/"+SimParameters.getOutputFold()+"/replicates/"+"Summary_Statistics_DataFrame_Performance_"+stringseednumber;
            system(systemcall.c_str());
            if(trainregions.size() > 0)
            {
                systemcall = "mv "+Summary_Haplofinder + " " +path+"/"+SimParameters.getOutputFold()+"/replicates/"+"Summary_Haplofinder_"+stringseednumber;
                system(systemcall.c_str());
            }
            if((SimParameters.get_rohgeneration()).size() > 0)
            {
                systemcall = "mv "+Summary_ROHGenome_Freq + " " +path+"/"+SimParameters.getOutputFold()+"/replicates/"+"Summary_ROHGenome_Freq_"+stringseednumber;
                system(systemcall.c_str());
                systemcall = "mv "+Summary_ROHGenome_Length + " " +path+"/"+SimParameters.getOutputFold()+"/replicates/"+"ROHGenome_Length_"+stringseednumber;
                system(systemcall.c_str());
            }
            if(SimParameters.getOutputTrainReference() == "yes")
            {
                systemcall = "mv "+Amax_Output + " " +path+"/"+SimParameters.getOutputFold()+"/replicates/"+"AmaxGeneration_"+stringseednumber;
                system(systemcall.c_str());
                systemcall = "mv "+Correlation_Output + " " +path+"/"+SimParameters.getOutputFold()+"/replicates/"+"ProgenyParentCorrelationGeneration_"+stringseednumber;
                system(systemcall.c_str());
            }
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
