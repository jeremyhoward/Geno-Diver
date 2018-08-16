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
#include "mkl_spblas.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <tuple>
#include <map>
#include <iterator>
#include "zfstream.h"

using namespace std;

/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Modules within Program  **************************************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
#include "HaplofinderClasses.h"
#include "Animal.h"
#include "MatingDesignClasses.h"
#include "ParameterClass.h"
#include "OutputFiles.h"
#include "Global_Population.h"
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from ParameterClass.cpp          *******************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void read_generate_parameters(parameters &SimParameters, string parameterfile, string &logfilestring, string &logfilestringa);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from Global_Population.cpp       *******************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void initializevectors(globalpopvar &Population1, parameters &SimParameters);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/*  Class that stores all outputfile locations and generates output files ***************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void GenerateOutputFiles(parameters &SimParameters, outputfiles &OUTPUTFILES,string path);
void DeletePreviousSimulation(outputfiles &OUTPUTFILES,parameters &SimParameters,string logfileloc);
void ExpectedHeterozygosity(vector <Animal> &population, globalpopvar &Population1, int Gen,ostream& logfileloc);
void UpdateFrequency_GenVar(parameters &SimParameters, vector <Animal> &population, globalpopvar &Population1, int Gen, vector < QTL_new_old > &population_QTL,outputfiles &OUTPUTFILES,ostream& logfileloc);
void LD_Option(parameters &SimParameters, vector <Animal> &population, vector <QTL_new_old> &population_QTL,outputfiles &OUTPUTFILES,string foundergen,ostream& logfileloc);
void Haplofinder_Option(parameters &SimParameters, vector <Animal> &population,vector <Unfavorable_Regions> &trainregions,string unfav_direc,int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc);
void ROH_Option(parameters &SimParameters,vector <Animal> &population, int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc);
int OutputPedigree_GenomicEBV(outputfiles &OUTPUTFILES, vector <Animal> &population,int TotalAnimalNumber, parameters &SimParameters);
/* Generate Output Files */
void GenerateMaster_DataFrame(parameters &SimParameters, outputfiles &OUTPUTFILES,globalpopvar &Population1, vector<double> &estimatedsolutions,vector<double> &trueaccuracy,vector < int > &ID_Gen);
void generatesummaryqtl(parameters &SimParameters, outputfiles &OUTPUTFILES,vector < int > const &idgeneration,globalpopvar &Population1);
void generatessummarydf(parameters &SimParameters, outputfiles &OUTPUTFILES, globalpopvar &Population1);
void generateqtlfile(parameters &SimParameters, outputfiles &OUTPUTFILES);
void CleanUpSimulation(outputfiles &OUTPUTFILES);
void SaveReplicates (int reps,parameters &SimParameters, outputfiles &OUTPUTFILES,string logfileloc, string path);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/*  Setup Genome  ***********************************************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void RunMaCS(parameters &SimParameters,globalpopvar &Population1,ostream& logfileloc);
void NoMaCSUpdateFiles(parameters &SimParameters,globalpopvar &Population1,ostream& logfileloc);
void generatemapfile(globalpopvar &Population1, parameters &SimParameters,outputfiles &OUTPUTFILES);
void generatefounderanimals(globalpopvar &Population1, parameters &SimParameters,outputfiles &OUTPUTFILES,ostream& logfileloc,vector <int> &FullColNum,vector <string> &founder_qtl, vector <string> &founder_markers);
void calculatetraitcorrelation(globalpopvar &Population1,parameters &SimParameters,ostream& logfileloc);
void Scale_Quantitative(globalpopvar &Population1, parameters &SimParameters,vector <string> &founder_qtl,ostream& logfileloc);
void SummarizeFitness(globalpopvar &Population1,outputfiles &OUTPUTFILES,parameters &SimParameters,ostream& logfileloc);
void AddToQTLClass(globalpopvar &Population1,outputfiles &OUTPUTFILES,vector < QTL_new_old > &population_QTL,parameters &SimParameters,ostream& logfileloc);
void GenerateHapLibrary(globalpopvar &Population1,parameters &SimParameters,vector < hapLibrary > &haplib);
void traitcorrelation(vector <Animal> &population, ostream& logfileloc);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from Genome_ROH.cpp              *******************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void Genome_ROH_Summary(parameters &SimParameters, outputfiles &OUTPUTFILES, vector <Animal> &population,int Gen, ostream& logfileloc);
void Proportion_ROH(parameters &SimParameters, vector <Animal> &population,outputfiles &OUTPUTFILES, ostream& logfileloc);
void ld_decay_estimator(outputfiles &OUTPUTFILES, vector <Animal> &population, string lineone);
void qtlld_decay_estimator(parameters &SimParameters, vector <Animal> &population, vector <QTL_new_old> &population_QTL,outputfiles &OUTPUTFILES,string foundergen);
void WindowVariance(parameters &SimParameters,vector <Animal> &population,vector < QTL_new_old > &population_QTL,string foundergen ,outputfiles &OUTPUTFILES);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from MatingDesignClasses.cpp ***********************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void agedistribution(vector <Animal> &population,vector <int> &MF_AgeClass, vector <int> &M_AgeClass, vector <int> &MF_AgeID);
void betadistributionmates(vector <Animal> &population,parameters &SimParameters, int M_NumberClassg0, vector <double> const &number, vector <int> const &M_AgeClass,vector <int> const &MF_AgeID);
void outputlogsummary(vector <Animal> &population, vector<int> &M_AgeClass, vector<int> &MF_AgeID,ostream& logfileloc);
string choosematingscenario(parameters &SimParameters, string tempselectionvector);
void generatematingpairs(vector <MatingClass> &matingindividuals, vector <Animal> &population, vector < hapLibrary > &haplib,parameters &SimParameters, string matingscenario, outputfiles &OUTPUTFILES, double* M,float scale, ostream& logfileloc);
void indexmatingdesign(vector <MatingClass> &matingindividuals, vector <Animal> &population, vector < hapLibrary > &haplib, parameters &SimParameters, outputfiles &OUTPUTFILES , double* M,float scale,ostream& logfileloc);
void updatematingindex(vector <MatingClass> &matingindividuals, vector <Animal> &population);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from SelectionCullingFunctions.cpp *****************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void breedingagedistribution(vector <Animal> &population,parameters &SimParameters,ostream& logfileloc);
void truncationselection(vector <Animal> &population,parameters &SimParameters,string tempselectionscen,int Gen,outputfiles &OUTPUTFILES,globalpopvar &Population1,ostream& logfileloc);
void optimalcontributionselection(vector <Animal> &population,vector <MatingClass> &matingindividuals,vector < hapLibrary > &haplib,parameters &SimParameters,string tempselectionscen,double* M, float scale,outputfiles &OUTPUTFILES,int Gen,ostream& logfileloc);
void discretegenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc);
void overlappinggenerations(vector <Animal> &population,parameters &SimParameters,string tempcullingscen,int Gen,outputfiles &OUTPUTFILES,ostream& logfileloc);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from EBV_Functions.cpp  ****************************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void Generate_BLUP_EBV(parameters &SimParameters,vector <Animal> &population, vector <double> &estimatedsolutions, vector <double> &trueaccuracy,ostream& logfileloc,vector <int> &trainanimals,int TotalAnimalNumber, int TotalOldAnimalNumber, int Gen,double* M, float scale,vector < hapLibrary > &haplib,outputfiles &OUTPUTFILES);
void bayesianestimates(parameters &SimParameters,vector <Animal> &population,int Gen ,vector <double> &estimatedsolutions,outputfiles &OUTPUTFILES,ostream& logfileloc);
void getamax(parameters &SimParameters,vector <Animal> &population,outputfiles &OUTPUTFILES);
void trainrefcor(parameters &SimParameters,vector <Animal> &population,outputfiles &OUTPUTFILES, int Gen);
void Inbreeding_Pedigree(vector <Animal> &population,outputfiles &OUTPUTFILES);
void updateanimalgenostatus(parameters &SimParameters,vector <Animal> &population,outputfiles &OUTPUTFILES);
void breedingpopulationgenotyped(vector <Animal> &population, ostream& logfileloc);
void getGenotypeCountGeneration(outputfiles &OUTPUTFILES,ostream& logfileloc);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from HaplofinderClasses.cpp  ***********************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void EnterHaplotypeFinder(parameters &SimParameters,vector <Unfavorable_Regions> &trainregions,vector <Animal> &population,int Gen,int retraingeneration,string unfav_direc,outputfiles &OUTPUTFILES,ostream& logfileloc);
void calculate_IIL(vector <double> &inbreedingload, vector <string> const &progenygenotype,vector <Unfavorable_Regions> &trainregions,string unfav_direc);
void calculatecorrelation(vector <double> &inbreedingload, vector <double> &progenyphenotype,vector <double> &progenytgv, vector <double> &progenytbv, vector <double> &progenytdd, vector < double > &outcorrelations);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from Animal_Functions.cpp **************************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
void GenerateBaseValues(vector <Animal> &population,vector <double> &meangv, vector <double> &meanbv, vector <double> &meandd);
void GenerateGenerationInterval(vector <Animal> &population,globalpopvar &Population1,parameters &SimParameters,int Gen, ostream& logfileloc);
void GenerateBVIndex(vector <Animal> &population,globalpopvar &Population1,parameters &SimParameters,int Gen, ostream& logfileloc);
void UpdateTBV_TDD_Statistical(globalpopvar &Population1, vector <Animal> &population,parameters &SimParameters);
void Update_selcand_PA(vector <Animal> &population);
void UpdateGenoPhenoStatus(globalpopvar &Population1, vector <Animal> &population,parameters &SimParameters,outputfiles &OUTPUTFILES, int Gen, string stage, vector < string > SelectionVector = vector<string>(0));

/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Functions from Simulation_Functions.cpp **********************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
double lethal_pedigree_inbreeding(outputfiles &OUTPUTFILES, int tempsireid, int tempdamid);
void frequency_calc(vector < string > const &genotypes, double* output_freq);
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Function to create Beta Distribution. Random library does not produce it; stems from a gamma. ****************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
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
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
/* Start of Simulation ******************************************************************************************************************************/
/****************************************************************************************************************************************************/
/****************************************************************************************************************************************************/
int main(int argc, char* argv[])
{
    std::setprecision(10); using Eigen::MatrixXd; using Eigen::VectorXd;
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
    cout<<"###########  _-~~   -~~ _                               ,' #############\n";
    cout<<"########### `-_         _                              ; ###############\n";
    cout<<"#############  ~~----~~~   ;                          ; ################\n";
    cout<<"###############  /          ;                         ; ################\n";
    cout<<"#############  /             ;                      ; ##################\n";
    cout<<"###########  /                `                    ; ###################\n";
    cout<<"#########  /                                      ; ####################\n";
    cout<<"#######                                            #####################\n";
    cout<<"########################################################################\n";
    cout<<"------------------------------------------------------------------------\n";
    cout<<"- GENO-DIVER (V3)                                                      -\n";
    cout<<"- Complex Genomic Simulator                                            -\n";
    cout<<"- Authors: Jeremy T. Howard, Christian Maltecca, Francesco Tiezzi,     -\n";
    cout<<"-          Jennie E. Pryce and Matthew L. Spangler                     -\n";
    cout<<"- Date: July 2018                                                      -\n";
    cout<<"------------------------------------------------------------------------\n";
    cout<<"- For any questions/bugs e-mail jeremy.howard06@gmail.com.             -\n";
    cout<<"- This program is free software: you can redistribute it and/or modify -\n";
    cout<<"- it under the terms of the GNU General Public License as published by -\n";
    cout<<"- the Free Software Foundation, either version 3 of the License, or    -\n";
    cout<<"- (at your option) any later version.                                  -\n";
    cout<<"------------------------------------------------------------------------\n";
    /* Figure out where you are currently at then just append to string */
    char * cwd; cwd = (char*) malloc( FILENAME_MAX * sizeof(char)); getcwd(cwd,FILENAME_MAX); string path(cwd);
    /* Grab name of parameter file and initialize parameters */
    if(argc != 2){cout << "Program ended due to a parameter file not given!" << endl; exit (EXIT_FAILURE);}
    string paramterfile = argv[1];
    parameters SimParameters;       /* Initialize parameter class */
    string logfilestring, logfilestringa;
    read_generate_parameters(SimParameters,paramterfile,logfilestring,logfilestringa);      /* Figure out parameters */
    /* Need to remove all the files within GenoDiverFiles */
    if(SimParameters.getStartSim() == "sequence")
    {
        string rmfolder = "rm -rf ./" + SimParameters.getOutputFold(); system(rmfolder.c_str());
        rmfolder = "mkdir " + SimParameters.getOutputFold(); system(rmfolder.c_str());
    }
    if(SimParameters.getStartSim() == "founder")
    {
        std::string x = path + "/" + SimParameters.getOutputFold() + "/";
        const char *folderr = x.c_str();
        struct stat sb;
        if (stat(folderr, &sb) == 0 && S_ISDIR(sb.st_mode)){
        }else{cout << endl << " FOLDER DOESN'T EXIST. CHANGE TO 'START: sequence' TO FILL FOLDER IF USING FOUNDER AS START!\n" << endl;}
    }
    outputfiles OUTPUTFILES; /* Initialize output file class */
    GenerateOutputFiles(SimParameters, OUTPUTFILES, path);
    string logfileloc = path + "/" + SimParameters.getOutputFold() + "/log_file.txt";
    int nthread = SimParameters.getThread(); omp_set_num_threads(nthread); mkl_set_num_threads_local(nthread); /* Number of threads */
    /**************************/
    /* loop across replicates */
    /**************************/
    for(int reps = 0; reps < SimParameters.getReplicates(); reps++)
    {
        if(reps == 0)       /* Remove previous replicates folder */
        {
            string systemcall = "rm -rf " + path + "/" + SimParameters.getOutputFold() + "/replicates || true"; system(systemcall.c_str());
        }
        time_t repbegin_time = time(0);
        if(SimParameters.getReplicates() > 1){cout << endl << "~~~~~~~~~~~~~\t Starting Replicate " << reps + 1 << " \t~~~~~~~~~~~~~" << endl;}
        /* add seed by one to generate new replicates and reprint out parameter file with updated logfile */
        if(reps > 0)
        {
            SimParameters.UpdateSeed(SimParameters.getSeed()+1); SimParameters.UpdateStartSim("founder");
            stringstream s1; s1 << SimParameters.getSeed(); string tempvar = s1.str();
        }
        DeletePreviousSimulation(OUTPUTFILES,SimParameters,logfileloc); /* Deletes files from Previous Simulation */
        std::ofstream logfile(logfileloc, std::ios_base::out);               /* open log file to output verbage throughout code */
        if(SimParameters.getstartgen() != -5 && SimParameters.gettraingen() != -5)
        {
            fstream checkhaplosummary;
            checkhaplosummary.open(OUTPUTFILES.getloc_Summary_Haplofinder(), std::fstream::out | std::fstream::trunc); checkhaplosummary.close();
        }
        logfile << logfilestringa << "        - Simulation Started at:\t\t\t\t\t\t\t\t\t'" << SimParameters.getStartSim() << endl;
        logfile << "        - Seed Number:\t\t\t\t\t\t\t\t\t\t\t'" << SimParameters.getSeed() << "'" << endl << logfilestring << endl;
        /* Initialize global variabes for a given population */
        globalpopvar Population1;                                                       /* Initialize global variable class */
        initializevectors(Population1, SimParameters);                                  /* Initialize vector sizes */
        int Marker_IndCounter = 0;                                                      /* Counter to determine where you are at in Marker index array */
        int QTL_IndCounter = 0;                                                         /* Counter to determine where you are at in QTL index array */
        vector < Animal > population;                                                   /* Hold in a vector of Animal Objects */
        vector < QTL_new_old > population_QTL;                                          /* Hold in a vector of QTL_new_old Objects */
        vector < hapLibrary > haplib;                                                   /* Vector of haplotype library objects */
        vector <MatingClass> matingindividuals;                                         /* Mating Design Variables */
        vector <Unfavorable_Regions> trainregions;                                      /* Unfavorable Haplotype Finder Class Vector */
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        /************************                        Unfavorable Haplotype Finder Variables                         ************************/
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        string unfav_direc; int retraingeneration = (SimParameters.getstartgen() + SimParameters.getGenfoundsel());
        if(SimParameters.getSelectionDir() == "high"){unfav_direc = "low";}
        if(SimParameters.getSelectionDir() == "low"){unfav_direc = "high";}
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        /************************                                       Begin Simulation                                ************************/
        /***************************************************************************************************************************************/
        /***************************************************************************************************************************************/
        const clock_t intbegin_time = clock();
        logfile << endl;
        if(SimParameters.getStartSim() == "sequence"){RunMaCS(SimParameters,Population1,logfile);}
        if(SimParameters.getStartSim() != "sequence"){NoMaCSUpdateFiles(SimParameters,Population1,logfile);}
        if(SimParameters.getStartSim() == "founder" || SimParameters.getStartSim() == "sequence")
        {
            /**** Used to save animals that died as one string */
            vector < string > leftfitnessstring;                                            /* Vector that saves low fitness animal summary stats */
            vector < string > rightfitnessstring;                                           /* Vector that saves low fitness animal summary stats */
            vector < vector < double > > quantvalues;                                       /* Quantitative Values */
            vector < string > markerlowfitness;                                             /* Marker for low fitness animal */
            /*****************************************************/
            /**** Begin Forward-In-Time Simulation Techniques ****/
            /*****************************************************/
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
                string snppath = path + "/" + SimParameters.getOutputFold() + "/" + (Population1.get_snpfiles())[i]; /* directory where file is read in */
                /* Read in Haplotypes */
                vector <string> Haplotypes; string gam;
                ifstream infile; infile.open(snppath);
                if (infile.fail()){logfile << "Error Opening MaCS SNP File. Check Parameter File!\n"; exit (EXIT_FAILURE);}
                while (infile >> gam){Haplotypes.push_back(gam);}
                /* need to push_back GenoSum when start a new chromosome */
                Population1.update_chrsnplength(i, Haplotypes[0].size());
                if(Haplotypes.size() < SimParameters.getfnd_haplo())
                {
                    cout<<endl<<"Haplotype # from a previous scenario has changed. Start from sequence!!\n"<<endl; exit (EXIT_FAILURE);
                }
                logfile << "    - Number of SNP for Chromosome " << i + 1 << ": " << (Population1.get_chrsnplength())[i] << " SNP." << endl;
                if(((Population1.get_chrsnplength())[i]*0.95) < (SimParameters.get_Marker_chr())[i])
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
                    for (int j = 0; j < 2; j++)                                                 /* Randomly grab to 2 haplotypes without replacement */
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
            ofstream output15; output15.open(OUTPUTFILES.getloc_snpfreqfile().c_str());
            output15 << outputstringfreq.str(); outputstringfreq.str(""); outputstringfreq.clear();
            output15.close();
            // Part 4: Output into FounderFile
            ofstream output16; output16.open (OUTPUTFILES.getloc_foundergenofile().c_str());
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
            /* 2D-Array of additive_quan effect */
            vector < vector < double>> FullAddEffectQuan(SIZEDF,vector<double>(SimParameters.getnumbertraits(),0.0));
            /* 2D-Array of dominance_quan effect */
            vector < vector < double>> FullDomEffectQuan(SIZEDF,vector<double>(SimParameters.getnumbertraits(),0.0));
            vector < double > FullAddEffectFit(SIZEDF,0.0);                 /* Array of selection coefficent fitness effect */
            vector < double > FullDomEffectFit(SIZEDF,0.0);                 /* Array of degree of dominance fitness effect */
            int TotalSNPCounter = 0;                                        /* Counter to set where SNP is placed FullMapPos, FullColNum, FullQTL_Mark */
            int TotalFreqCounter = 0;                                       /* Counter to set where SNP is placed in Freq File for QLT Location */
            int TotalFreqCounter1 = 0;                                      /* Counter to set where SNP is placed in Freq File for Marker Location */
            int firstsnpfreq = 0;                                           /* first snp that pertains to a given chromosome */
            /* Vectors used within each chromosome set to nothing now and then within increase to desired size and when done with clear to start fresh */
            vector < double > mappos;                                       /* Map position for a given chromosome;set length last SNP map */
            vector <double> Add_Array_Fit;                                  /* Array of selection coefficent fitness effect */
            vector <double> Dom_Array_Fit;                                  /* Array of degree of dominance fitness effect */
            vector < vector < double>> Add_Array_Quan;                      /* 2D-Array of additive_quan effect */
            vector < vector < double>> Dom_Array_Quan;                      /* 2D-Array of dominance_quan effect */
            vector < int > QTL;                                             /* array to hold whether SNP is QTL or can be used as a marker */
            vector < int > markerlocation;                                  /* at what number is marker suppose to be at */
            vector < double > mapmark;                                      /* array with Marker Map for Test Genotype */
            vector < int > colnum;                                          /* array with Column Number for genotypes that were kept to grab SNP Later */
            vector < double > MarkerQTLFreq;                                /* array with frequency's for marker and QTL SNP */
            vector < int > QTL_Mark;                                        /* array of whether SNP is QTL or Marker */
            vector < double > addMark_Fit;                                  /* Array of selection coefficent fitness effect */
            vector < double > domMark_Fit;                                  /* Array of degree of dominance fitness effect */
            vector < vector < double>> addMark_Quan;                        /* 2D-Array of additive_quan effect */
            vector < vector < double>> domMark_Quan;                        /* 2D-Array of dominance_quan effect */
            /* Indicator for type of trait: 1 = marker; 2 = QTLQuanti; 3 = QTLQuant_QTLFitness; 4 = QTLlethal; 5 =QTLsublethal */
            vector < double > covar_add_fitness;                            /* used later on to generate covariance */
            vector < double > covar_add;                                    /* used later on to generate covariance */
            for(int c = 0; c < SimParameters.getChr(); c++)
            {
                logfile << "   - Chromosome " << c + 1 << ":" << endl;
                int endspot = (Population1.get_chrsnplength())[c];  /* Determines number of SNP to have in order to get end position */
                vector < double > mappos;                           /* Map position for a given chromosome; set length by last SNP map position number */
                string mapfilepath = path + "/" + SimParameters.getOutputFold() + "/" + (Population1.get_mapfiles())[c];
                /* READ map file for a given chromsome *************/
                ifstream infile1; infile1.open(mapfilepath);
                if(infile1.fail()){cout << "Error Opening MaCS Map File. Check ParameterFile!\n"; exit (EXIT_FAILURE);}
                for(int i = 0; i < endspot; i++){double temp; infile1 >> temp; mappos.push_back(temp);}
                infile1.close();
                ///////////////////////////////////
                ///// Step 1: Sample locations ////
                ///////////////////////////////////
                double lenChr = mappos[endspot - 1 ];               /* length of chromosome used when sampling from uniform distribution to get QTL location*/
                /* Fill vectors to zero */
                for(int i = 0; i < endspot; i++)
                {
                    vector < double > tempvect(SimParameters.getnumbertraits(),0.0);
                    Add_Array_Fit.push_back(0.0); Dom_Array_Fit.push_back(0.0); QTL.push_back(0.0);
                    Add_Array_Quan.push_back(tempvect); Dom_Array_Quan.push_back(tempvect);
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
                /* tabulate number of QTL for each chromosome may be less than what simulated due to two being within two markers*/
                int simulatedQTL = 0;
                for(int i = 0; i < endspot; i++){if(QTL[i] >= 2){simulatedQTL += 1;}}
                Population1.update_qtlperchr(c,simulatedQTL);
                Population1.update_markerperchr(c,(SimParameters.get_Marker_chr())[c]);
                ///////////////////////////////////////////////////////////////////////
                // Step 2: Construct Marker file based on given Marker MAF threshold //
                ///////////////////////////////////////////////////////////////////////
                /* Create Genotypes that has QTL and Markers */
                /* Determine length of array based on markers to keep plus number of actual QTL */
                int markertokeep = Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c];
                for(int i = 0; i < markertokeep; i++)
                {
                    vector < double > tempvect(SimParameters.getnumbertraits(),0.0);
                    mapmark.push_back(0.0); colnum.push_back(0); MarkerQTLFreq.push_back(0.0); QTL_Mark.push_back(0);
                    addMark_Fit.push_back(0.0); domMark_Fit.push_back(0.0);
                    addMark_Quan.push_back(tempvect); domMark_Quan.push_back(tempvect);
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
                        QTL_Mark[markloc] = 1; addMark_Fit[markloc] = 0; domMark_Fit[markloc] = 0;
                        addMark_Quan[markloc][0] = 0; domMark_Quan[markloc][0] = 0;
                        if(SimParameters.getnumbertraits() == 2){addMark_Quan[markloc][1] = 0.0; domMark_Quan[markloc][1] = 0.0;}
                        markloc++;   /* Increase position by one in mapMark and colnum and go to next declared marker */
                    }
                    if(QTL[i] > 1)
                    {
                        mapmark[markloc] = mappos[i]; colnum[markloc] = i; MarkerQTLFreq[markloc] = SNPFreq[i + firstsnpfreq];
                        QTL_Mark[markloc] = QTL[i]; addMark_Fit[markloc]=Add_Array_Fit[i]; domMark_Fit[markloc]=Dom_Array_Fit[i];
                        addMark_Quan[markloc][0]=Add_Array_Quan[i][0]; domMark_Quan[markloc][0]=Dom_Array_Quan[i][0];
                        if(SimParameters.getnumbertraits() == 2)
                        {
                            addMark_Quan[markloc][1] = Add_Array_Quan[i][1]; domMark_Quan[markloc][1] = Dom_Array_Quan[i][1];
                        }
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
                Population1.update_markerperchr(c,test_mark);
                logfile << "        - Number Markers: " << Population1.get_markerperchr()[c] << "." << endl;
                logfile << "        - Number of Quantitative QTL: " << test_QTL_Quan << "." << endl;
                logfile << "        - Number of Quantitative QTL with Pleiotropic effects on Fitness: " << test_QTL_Quan_Fitness << endl;
                logfile << "        - Number of Fitness QTL: " << test_QTL_Leth + test_QTL_SubLeth << "." << endl;
                markertokeep = totalkeep;
                /* Copy to full set */
                if(c == 0)
                {
                    int j = 0;                                                  /* where at in arrays for each chromosome */
                    for(int i = 0 ; i < markertokeep; i++)                      /* first chromsome would start at first postion until markertokeep */
                    {
                        FullMapPos[i] = mapmark[j] + c + 1; FullColNum[i] = colnum[j]; FullSNPFreq[i] = MarkerQTLFreq[j];
                        FullQTL_Mark[i] = QTL_Mark[j]; FullAddEffectFit[i]=addMark_Fit[j]; FullDomEffectFit[i]=domMark_Fit[j];
                        FullAddEffectQuan[i][0]=addMark_Quan[j][0]; FullDomEffectQuan[i][0]=domMark_Quan[j][0];
                        if(SimParameters.getnumbertraits() == 2){FullAddEffectQuan[i][1]=addMark_Quan[j][1]; FullDomEffectQuan[i][1]=domMark_Quan[j][1];}
                        j++;
                    }
                    TotalSNPCounter = markertokeep;
                }
                if(c > 0)
                {
                    int j = 0;
                    for(int i = TotalSNPCounter; i < (TotalSNPCounter + markertokeep); i++)
                    {
                        FullMapPos[i] = mapmark[j] + c + 1; FullColNum[i] = colnum[j]; FullSNPFreq[i] = MarkerQTLFreq[j]; FullQTL_Mark[i] = QTL_Mark[j];
                        FullAddEffectFit[i]=addMark_Fit[j]; FullDomEffectFit[i]=domMark_Fit[j];
                        FullAddEffectQuan[i][0]=addMark_Quan[j][0]; FullDomEffectQuan[i][0]=domMark_Quan[j][0];
                        if(SimParameters.getnumbertraits() == 2)
                        {
                            FullAddEffectQuan[i][1]=addMark_Quan[j][1]; FullDomEffectQuan[i][1]=domMark_Quan[j][1];
                        }
                        j++;
                    }
                    TotalSNPCounter = TotalSNPCounter + markertokeep;           /* Updates where should begin to fill for next chromosome */
                }
                firstsnpfreq += endspot;
                mappos.clear(); QTL.clear(); markerlocation.clear(); mapmark.clear(); colnum.clear();
                MarkerQTLFreq.clear(); QTL_Mark.clear(); Add_Array_Fit.clear(); Dom_Array_Fit.clear();
                for(int i = 0; i < endspot; i++){Add_Array_Quan[i].clear(); Dom_Array_Quan[i].clear();}
                for(int i = 0; i < markertokeep; i++){addMark_Quan[i].clear(); domMark_Quan[i].clear();}
                Add_Array_Quan.clear(); Dom_Array_Quan.clear(); addMark_Quan.clear();
                domMark_Quan.clear(); addMark_Fit.clear(); domMark_Fit.clear();
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
                if(FullQTL_Mark[i] == 1){Population1.add_markerindex(i); Population1.add_markermapposition(FullMapPos[i]); Marker_IndCounter++;}
                if(FullQTL_Mark[i] > 1)     /* If is a qtl */
                {
                    Population1.update_qtlindex(QTL_IndCounter,i); Population1.update_mapposition(QTL_IndCounter,FullMapPos[i]);
                    Population1.update_qtl_type(QTL_IndCounter,FullQTL_Mark[i]); Population1.update_qtl_freq(QTL_IndCounter,FullSNPFreq[i]);
                    /* Indicator for which homozygote impacted */
                    if((Population1.get_qtl_freq())[QTL_IndCounter] > 0.5){Population1.update_qtl_allele(QTL_IndCounter,0);}
                    if((Population1.get_qtl_freq())[QTL_IndCounter] < 0.5){Population1.update_qtl_allele(QTL_IndCounter,2);}
                    Population1.update_qtl_add_quan(QTL_IndCounter,0,FullAddEffectQuan[i][0]);
                    Population1.update_qtl_dom_quan(QTL_IndCounter,0,FullDomEffectQuan[i][0]);
                    if(SimParameters.getnumbertraits() == 2)
                    {
                        Population1.update_qtl_add_quan(QTL_IndCounter,1,FullAddEffectQuan[i][1]);
                        Population1.update_qtl_dom_quan(QTL_IndCounter,1,FullDomEffectQuan[i][1]);
                    }
                    Population1.update_qtl_add_fit(QTL_IndCounter,FullAddEffectFit[i]);
                    Population1.update_qtl_dom_fit(QTL_IndCounter,FullDomEffectFit[i]);
                    QTL_IndCounter++;                                                                   /* Move to next cell of array */
                }
            }
            generatemapfile(Population1,SimParameters,OUTPUTFILES); /* Generate Marker Map File and Output; If doing ROH also generate */
            ////////////////////////////////////////////////////////////////////////////////////////
            // Generate Founder Animal Marker+QTL/FTL string; then split off into Marker and QTL ///
            ////////////////////////////////////////////////////////////////////////////////////////
            vector <string > founder_qtl; vector <string> founder_markers;
            generatefounderanimals(Population1,SimParameters,OUTPUTFILES,logfile,FullColNum,founder_qtl,founder_markers);
            ///////////////////////////////
            // Generate QTL/FTL Effects ///
            ///////////////////////////////
            for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
            {
                if((Population1.get_qtl_type())[i] == 2)                    /* Tagged as quantitative QTL with no relationship to fitness */
                {
                    if(SimParameters.getnumbertraits() == 1)
                    {
                        /******* QTL Additive Effect (Gamma *******/
                        std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape(),SimParameters.getGamma_Scale());
                        double tempa = distribution1(gen);
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                        std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                        double temph = distribution2(gen);
                        Population1.update_qtl_dom_quan(i,0,double (tempa*temph));
                        std::binomial_distribution<int> distribution3(1,0.5);
                        int signadd = distribution3(gen);
                        if(signadd == 1){tempa = tempa * -1;}           /*Assign negative effect with a 50% probability */
                        Population1.update_qtl_add_quan(i,0,tempa);
                        /* set fitness ones to 0 */
                        Population1.update_qtl_add_fit(i,0.0);
                        Population1.update_qtl_dom_fit(i,0.0);
                    }
                    if(SimParameters.getnumbertraits() == 2)
                    {
                        /******* QTL Additive Effect ********/
                        /*          Generate three random gamma's with same Scale           */
                        /* x1 common to both; x2 common to trait one; x3 common to trait 3; */
                        /* Incommon shape: fullshape * correlation */
                        double shapeX1 = SimParameters.getGamma_Shape() * SimParameters.get_Var_Additive()[1];
                        /* Not incommon shape: fullshape - Incommon shape */
                        double shapeX2_X3 = SimParameters.getGamma_Shape() - shapeX1;
                        /* use binomial to figure out whether effect negative or positive */
                        std::binomial_distribution<int> distribution1(1,0.5);
                        /* Generate x1 */
                        std::gamma_distribution <double> distribution2(shapeX1,SimParameters.getGamma_Scale()); /* X1 for incommon portion*/
                        double Y1 = distribution2(gen);
                        int signadd = distribution1(gen);
                        if(signadd == 1){Y1 = Y1 * -1;}                                     /*Assign negative effect with a 50% probability */
                        /* Generate x2 */
                        std::gamma_distribution <double> distribution3(shapeX2_X3,SimParameters.getGamma_Scale()); /* X1 for incommon portion*/
                        double Y2 = distribution3(gen);
                        signadd = distribution1(gen);
                        if(signadd == 1){Y2 = Y2 * -1;}                                     /*Assign negative effect with a 50% probability */
                        /* Generate x2 */
                        double Y3 = distribution3(gen);
                        signadd = distribution1(gen);
                        if(signadd == 1){Y3 = Y3 * -1;}                                     /*Assign negative effect with a 50% probability */
                        Population1.update_qtl_add_quan(i,0,(Y1 + Y2));
                        Population1.update_qtl_add_quan(i,1,(Y1 + Y3));
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                        std::normal_distribution<double> distribution4(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                        double temph = distribution4(gen);
                        Population1.update_qtl_dom_quan(i,0,(abs((Y1 + Y2))*temph));
                        temph = distribution3(gen);
                        Population1.update_qtl_dom_quan(i,1,(abs((Y1 + Y3))*temph));
                        /* set fitness ones to 0 */
                        Population1.update_qtl_add_fit(i,0.0);
                        Population1.update_qtl_dom_fit(i,0.0);
                    }
                }
                if((Population1.get_qtl_type())[i] == 3)                 /* Tagged as quantitative QTL with a relationship to fitness */
                {
                    /******* QTL Additive Effect ********/
                    /* Utilizing a Trivariate Reduction */
                    double sub = SimParameters.getgencorr() * sqrt(SimParameters.getGamma_Shape()*SimParameters.getGamma_Shape_SubLethal());
                    std::gamma_distribution <double> distribution1((SimParameters.getGamma_Shape()-sub),1);            /* QTL generated from a gamma */
                    double temp = distribution1(gen);
                    std::gamma_distribution <double> distribution2(sub,1);                          /* Covariance part */
                    double tempa = distribution2(gen);
                    Population1.update_qtl_add_quan(i,0,(SimParameters.getGamma_Scale() * (temp + tempa)));
                    covar_add_fitness.push_back(tempa);
                    /*******   QTL Fit S effect  *******/               /* Generate effect after standardize add and dominance effect */
                    Population1.update_qtl_add_fit(i,-5);               /* Set it to -5 first to use as a flag since # has to be greater than 0 */
                    Population1.update_qtl_dom_fit(i,-5);
                    covar_add.push_back(Population1.get_qtl_add_quan(i,0));
                    /****** QTL Dominance Effect *******/
                    /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                    std::normal_distribution<double> distribution5(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                    double temph = distribution5(gen);
                    Population1.update_qtl_dom_quan(i,0,(abs(Population1.get_qtl_add_quan(i,0)) * temph));
                    if(SimParameters.getnumbertraits() == 2)
                    {
                        /******* QTL Additive Effect (Gamma *******/
                        std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape(),SimParameters.getGamma_Scale());
                        double tempa = distribution1(gen);
                        /****** QTL Dominance Effect *******/
                        /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                        std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                        double temph = distribution2(gen);
                        Population1.update_qtl_dom_quan(i,1,double (tempa*temph));
                        std::binomial_distribution<int> distribution3(1,0.5);
                        int signadd = distribution3(gen);
                        if(signadd == 1){tempa = tempa * -1;}           /*Assign negative effect with a 50% probability */
                        Population1.update_qtl_add_quan(i,1,tempa);
                    }
                }
                if((Population1.get_qtl_type())[i] == 4)                 /* Tagged as lethal fitness QTL with no relationship to quantitative */
                {
                    /*******     QTL s effect (i.e. selection coeffecient)   *******/
                    std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape_Lethal(),SimParameters.getGamma_Scale_Lethal());
                    Population1.update_qtl_add_fit(i,distribution1(gen));
                    /******      QTL h Effect (i.e. degree of dominance)     *******/
                    /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                    std::normal_distribution<double> distribution2(SimParameters.getNormal_meanRelDom_Lethal(),SimParameters.getNormal_varRelDom_Lethal());
                    double temph = distribution2(gen);
                    Population1.update_qtl_dom_fit(i,abs(temph));
                    /* Set quantitative additive and dominance to 0 */
                    Population1.update_qtl_add_quan(i,0,0.0); Population1.update_qtl_dom_quan(i,0,0.0);
                    if(SimParameters.getnumbertraits() == 2){Population1.update_qtl_add_quan(i,1,0.0); Population1.update_qtl_dom_quan(i,1,0.0);}
                }
                if((Population1.get_qtl_type())[i] == 5)                 /* Tagged as sub-lethal fitness QTL with no relationship to quantitative */
                {
                    /*******     QTL S effect (i.e. selection coeffecient)   *******/
                    std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape_SubLethal(),SimParameters.getGamma_Scale_SubLethal());
                    Population1.update_qtl_add_fit(i,distribution1(gen));
                    /******      QTL h Effect (i.e. degree of dominance)     *******/
                    /* relative dominance degrees simulated; this sets heterozygote genotype to s*h */
                    std::normal_distribution <double> distribution2(SimParameters.getNormal_meanRelDom_SubLethal(),SimParameters.getNormal_varRelDom_SubLethal());
                    double temph = distribution2(gen);
                    Population1.update_qtl_dom_fit(i,abs(temph));
                    /* Set quantitative additive and dominance to 0 */
                    Population1.update_qtl_add_quan(i,0,0.0); Population1.update_qtl_dom_quan(i,0,0.0);
                    if(SimParameters.getnumbertraits() == 2){Population1.update_qtl_add_quan(i,1,0.0); Population1.update_qtl_dom_quan(i,1,0.0);}
                }
                //if((Population1.get_qtl_type())[i] != 0)
                //{
                //    cout <<(Population1.get_qtlindex())[i]<<" "<<(Population1.get_qtl_mapposition())[i]<<" "<<(Population1.get_qtl_type())[i]<<" ";
                //    cout <<(Population1.get_qtl_freq())[i]<<" "<<(Population1.get_qtl_allele())[i]<<" ";
                //    for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                //    {
                //        cout <<Population1.get_qtl_add_quan(i,j)<<" "<<Population1.get_qtl_dom_quan(i,j)<<" ";
                //    }
                //    cout <<(Population1.get_qtl_add_fit())[i]<<" "<<(Population1.get_qtl_dom_fit())[i]<<endl;
                //}
            }
            /* When create covariance between quantitative and fitness some may fall out therefore need to make sure they are lined up */
            if(covar_add_fitness.size() > 0)
            {
                int counteradd_cov = 0;
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 3)
                    {
                        if(Population1.get_qtl_add_quan(i,0) != covar_add[counteradd_cov])
                        {
                            covar_add.erase(covar_add.begin()+counteradd_cov); covar_add_fitness.erase(covar_add_fitness.begin()+counteradd_cov);
                        }
                        if(Population1.get_qtl_add_quan(i,0) == covar_add[counteradd_cov]){counteradd_cov++;}
                    }
                }
                covar_add.clear();
            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Step 4: Scale Additive & Dominance Effects; Generate Covarance between fitness and quantitative traits  //
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            Scale_Quantitative(Population1,SimParameters,founder_qtl,logfile); /* Scale effects */
            if(SimParameters.getnumbertraits() == 2){calculatetraitcorrelation(Population1,SimParameters,logfile);}
            if(SimParameters.getproppleitropic() > 0)
            {
                logfile << "   - Generate correlation (rank) between additive effects of quantitative and fitness." << endl;
                double cor = 0.0;
                while(SimParameters.getgencorr() - cor >= 0.015)
                {
                    vector < double > quant_rank; vector < double > fitness_rank;
                    int covarcount = 0;
                    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                    {
                        if((Population1.get_qtl_type())[i] == 3)
                        {
                            double sub = (-1 * SimParameters.getgencorr()) * sqrt(SimParameters.getGamma_Shape()*SimParameters.getGamma_Shape_SubLethal());
                            /******   QTL fit s Effect ********/
                            std::gamma_distribution <double> distribution2((SimParameters.getGamma_Shape_SubLethal()-sub),1);
                            double temp = distribution2(gen);                /* Raw Fitness Value need to transform to relative fitness */
                            double temp1 = SimParameters.getGamma_Scale_SubLethal()*(temp+(covar_add_fitness[covarcount]*(Population1.get_qtlscalefactadd())[0]));
                            Population1.update_qtl_add_fit(i,temp1);
                            /******   QTL Fit h Effect  *******/
                            /* relative dominance degrees simulated than multiply |Additive| * dominance degrees */
                            std::normal_distribution<double> distribution3(SimParameters.getNormal_meanRelDom_SubLethal(),SimParameters.getNormal_varRelDom_SubLethal());
                            double temph = distribution3(gen);
                            Population1.update_qtl_dom_fit(i,abs(temph));
                            quant_rank.push_back(Population1.get_qtl_add_quan(i,0)); fitness_rank.push_back((Population1.get_qtl_add_fit())[i]); covarcount++;
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
            SummarizeFitness(Population1,OUTPUTFILES,SimParameters,logfile);            /* Summaryize Fitness Mutations */
            AddToQTLClass(Population1,OUTPUTFILES,population_QTL,SimParameters,logfile);              /* Save Founder mutation in QTL_new_old class */
            logfile << "Finished Constructing Trait Architecture." << endl << endl;
            GenerateHapLibrary(Population1,SimParameters,haplib);                       /* Generate hapLibrary */
            /////////////////////////////////////////////////////////////////////////////////////
            // Step 6: Make Marker Panel and QTL Genotypes place into vector of Animal Objects //
            /////////////////////////////////////////////////////////////////////////////////////
            logfile << "Begin Creating Founder Population: " << endl;
            /* First initialzize QTLFreq_AcrossGen size in order to add genotypes count across animals to it */
            for(int i = 0; i < (Population1.get_qtl_type()).size(); i++){Population1.add_QTLFreq_AcrossGen(-5.0);}
            /* Now ensure correct number of male and female founder individuals */
            vector < int > male_female_status (founder_qtl.size(),-5);
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
                cout << endl << "Error in declaring sex in male and females!!" << endl; exit (EXIT_FAILURE);
            }
            int ind = 1;                                                    /* ID of animal increments by one for each line */
            int LethalFounder = 0;                                          /* number of animals dead in founder */
            Eigen::MatrixXd cholmultrait(SimParameters.getnumbertraits(),SimParameters.getnumbertraits());
            if(SimParameters.getnumbertraits() == 2)
            {
                Eigen::MatrixXd cov(2,2);
                cov << (SimParameters.get_Var_Residual())[0],((SimParameters.get_Var_Residual())[1]*sqrt((SimParameters.get_Var_Residual())[0])*sqrt((SimParameters.get_Var_Residual())[2])),((SimParameters.get_Var_Residual())[1]*sqrt((SimParameters.get_Var_Residual())[0])*sqrt((SimParameters.get_Var_Residual())[2])),(SimParameters.get_Var_Residual())[2];
                Eigen::LLT<Eigen::MatrixXd> cholSolver(cov);
                if (cholSolver.info()==Eigen::Success){
                    cholmultrait = cholSolver.matrixL();
                } else {cout << "Failed computing the Cholesky decomposition." << endl; exit (EXIT_FAILURE);}
                cov.resize(0,0);
            }
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
                    if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 4 || (Population1.get_qtl_type())[i] == 5)
                    {
                        if(QTLGenotypes[i] == (Population1.get_qtl_allele())[i])
                        {
                            relativeviability = relativeviability * (1-(Population1.get_qtl_add_fit())[i]);
                        }
                        if(QTLGenotypes[i] > 2)
                        {
                            relativeviability = relativeviability * (1-((Population1.get_qtl_add_fit())[i]*(Population1.get_qtl_dom_fit())[i]));
                        }
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
                    double lethalequivalent = 0.0; double Homoz = 0.0;
                    vector < double > GenotypicValuevec(SimParameters.getnumbertraits(),0.0);       /* Stores vector of Genotypic value */
                    vector < double > BreedingValuevec(SimParameters.getnumbertraits(),0.0);        /* Stores vector of Breeding Value */
                    vector < double > DominanceDeviationvec(SimParameters.getnumbertraits(),0.0);   /* Stores vector of Dominance Deviation */
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 4 || (Population1.get_qtl_type())[i] == 5)
                        {
                            if(QTLGenotypes[i] == (Population1.get_qtl_allele())[i])
                            {
                                if((Population1.get_qtl_type())[i] == 4){homozygouscount_lethal += 1;}
                                if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 5){homozygouscount_sublethal += 1;}
                                lethalequivalent += (Population1.get_qtl_add_fit())[i];
                            }
                            if(QTLGenotypes[i] > 2)
                            {
                                if((Population1.get_qtl_type())[i] == 4){heterzygouscount_lethal += 1;}
                                if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 5){heterzygouscount_sublethal += 1;}
                                lethalequivalent += (Population1.get_qtl_add_fit())[i];
                            }
                        }
                    }
                    //cout << homozygouscount_lethal << " " << heterzygouscount_lethal << " ";
                    //cout << homozygouscount_sublethal << " " << heterzygouscount_sublethal << " " << lethalequivalent << endl;
                    /* Quantititative Summary Statistics */
                    for(int i = 0; i < QTL_IndCounter; i++)                                                 /* Calculate Genotypic Value */
                    {
                        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)            /* Quantitative QTL */
                        {
                            int tempgeno;
                            if(QTLGenotypes[i] == 0 || QTLGenotypes[i] == 2){tempgeno = QTLGenotypes[i];}
                            if(QTLGenotypes[i] == 3 || QTLGenotypes[i] == 4){tempgeno = 1;}
                            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                            {
                                /* Breeding value is only a function of additive effects */
                                BreedingValuevec[j] += tempgeno * double(Population1.get_qtl_add_quan(i,j));
                                /* Not a heterozygote; function of additive */
                                if(tempgeno != 1){GenotypicValuevec[j] += tempgeno * double(Population1.get_qtl_add_quan(i,j));}
                                if(tempgeno == 1)
                                {
                                    /* Heterozygote so need to include add and dom */
                                    GenotypicValuevec[j] += (tempgeno * Population1.get_qtl_add_quan(i,j)) + Population1.get_qtl_dom_quan(i,j);
                                    DominanceDeviationvec[j] += Population1.get_qtl_dom_quan(i,j);
                                }
                            }
                        }
                    }
                    vector < double > temp;
                    for(int i = 0; i < SimParameters.getnumbertraits(); i++)
                    {
                        temp.push_back(GenotypicValuevec[i]); temp.push_back(BreedingValuevec[i]); temp.push_back(DominanceDeviationvec[i]);
                    }
                    quantvalues.push_back(temp);
                    /* Calculate Homozygosity only in the MARKERS */
                    for(int i=0; i < Marker_IndCounter; i++){if(MarkerGenotypes[i] == 0 || MarkerGenotypes[i] == 2){Homoz += 1;}}
                    Homoz = (Homoz/double(Marker_IndCounter));
                    strStreamQf << Homoz << " " << homozygouscount_lethal << " " << heterzygouscount_lethal << " ";
                    strStreamQf << homozygouscount_sublethal << " " << heterzygouscount_sublethal << " ";
                    strStreamQf << lethalequivalent << " " << relativeviability << " ";
                    for (int i=0; i < QTL_IndCounter; i++)
                    {
                        if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 4 || (Population1.get_qtl_type())[i] == 5)
                        {
                            strStreamQf << QTLGenotypes[i];
                        }
                    }
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
                    vector < double > Phenotypevec(SimParameters.getnumbertraits(),0.0);            /* Stores vector of Phenotypes */
                    vector < double > GenotypicValuevec(SimParameters.getnumbertraits(),0.0);       /* Stores vector of Genotypic value */
                    vector < double > BreedingValuevec(SimParameters.getnumbertraits(),0.0);        /* Stores vector of Breeding Value */
                    vector < double > DominanceDeviationvec(SimParameters.getnumbertraits(),0.0);   /* Stores vector of Dominance Deviation */
                    vector < double > Residualvec(SimParameters.getnumbertraits(),0.0);             /* Stores vector of Residual Value */
                    double Homoz = 0.0;                                                     /* Stores homozygosity based on marker information */
                    int Sex;                                                                /* Sex of the animal 0 is male 1 is female */
                    if(SimParameters.getnumbertraits() == 1)
                    {
                        double residvar = sqrt((SimParameters.get_Var_Residual())[0]);                    /* random number generator need standard deviation */
                        std::normal_distribution<double> distribution6(0.0,residvar);
                        Residualvec[0] = distribution6(gen);
                    }
                    if(SimParameters.getnumbertraits() == 2)                                 /* residual from a multivariate normal */
                    {
                        VectorXd standardnormals(2);
                        std::normal_distribution<double> distributionstandnormal(0.0,1);
                        standardnormals(0) = distributionstandnormal(gen);
                        standardnormals(1) = distributionstandnormal(gen);
                        standardnormals = cholmultrait * standardnormals;
                        Residualvec[0] = standardnormals(0);
                        Residualvec[1] = standardnormals(1);
                        standardnormals.resize(0);
                    }
                    /* Determine Sex of the animal based on draw from uniform distribution; if sex < 0.5 sex is 0 if sex >= 0.5 */
                    Sex = male_female_status[i];
                    /* Add to genotype count across generations */
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        int tempgeno;
                        if(QTLGenotypes[i] == 0 || QTLGenotypes[i] == 2){tempgeno = QTLGenotypes[i];}
                        if(QTLGenotypes[i] == 3 || QTLGenotypes[i] == 4){tempgeno = 1;}
                        if((Population1.get_QTLFreq_AcrossGen())[i] == -5.0){Population1.update_QTLFreq_AcrossGen(i,5.0);}
                        Population1.update_QTLFreq_AcrossGen(i,tempgeno);
                    }
                    Population1.UpdateQTLFreq_Number(1);
                    /* Calculate Genotypic Value */
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)            /* Quantitative QTL */
                        {
                            int tempgeno;
                            if(QTLGenotypes[i] == 0 || QTLGenotypes[i] == 2){tempgeno = QTLGenotypes[i];}
                            if(QTLGenotypes[i] == 3 || QTLGenotypes[i] == 4){tempgeno = 1;}
                            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                            {
                                /* Breeding value is only a function of additive effects */
                                BreedingValuevec[j] += tempgeno * double(Population1.get_qtl_add_quan(i,j));
                                /* Not a heterozygote; function of additive */
                                if(tempgeno != 1){GenotypicValuevec[j] += tempgeno * double(Population1.get_qtl_add_quan(i,j));}
                                if(tempgeno == 1)
                                {
                                    /* Heterozygote so need to include add and dom */
                                    GenotypicValuevec[j] += (tempgeno * Population1.get_qtl_add_quan(i,j)) + Population1.get_qtl_dom_quan(i,j);
                                    DominanceDeviationvec[j] += Population1.get_qtl_dom_quan(i,j);
                                }
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
                    int homozygouscount_lethal=0; int homozygouscount_sublethal=0; int heterzygouscount_lethal=0; int heterzygouscount_sublethal=0;
                    double lethalequivalent = 0.0;
                    for(int i = 0; i < QTL_IndCounter; i++)
                    {
                        if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 4 || (Population1.get_qtl_type())[i] == 5)
                        {
                            if(QTLGenotypes[i] == (Population1.get_qtl_allele())[i])
                            {
                                if((Population1.get_qtl_type())[i] == 4){homozygouscount_lethal += 1;}
                                if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 5){homozygouscount_sublethal += 1;}
                                lethalequivalent += (Population1.get_qtl_add_fit())[i];
                            }
                            if(QTLGenotypes[i] > 2)
                            {
                                if((Population1.get_qtl_type())[i] == 4){heterzygouscount_lethal += 1;}
                                if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 5){heterzygouscount_sublethal += 1;}
                                lethalequivalent += (Population1.get_qtl_add_fit())[i];
                            }
                        }
                    }
                    /* put marker, qtl and fitness into string to store */
                    stringstream strStreamM (stringstream::in | stringstream::out);
                    for (int i=0; i < Marker_IndCounter; i++){strStreamM << MarkerGenotypes[i];} string MA = strStreamM.str();
                    stringstream strStreamQt (stringstream::in | stringstream::out);
                    for (int i=0; i < QTL_IndCounter; i++){strStreamQt << QTLGenotypes[i];} string QT = strStreamQt.str();
                    if(SimParameters.getnumbertraits() == 1){Phenotypevec[0] = GenotypicValuevec[0] + Residualvec[0];}
                    if(SimParameters.getnumbertraits() == 2)
                    {
                        Phenotypevec[0] = GenotypicValuevec[0] + Residualvec[0];
                        Phenotypevec[1] = GenotypicValuevec[1] + Residualvec[1];
                    }
                    //cout << Phenotypevec[0] << " " << Phenotypevec[1] << endl;
                    //cout << GenotypicValuevec[0] << " " << GenotypicValuevec[1] << endl;
                    //cout << BreedingValuevec[0] << " " << BreedingValuevec[1] << endl;
                    //cout << DominanceDeviationvec[0] << " " << DominanceDeviationvec[1] << endl;
                    //cout << Residualvec[0] << " " << Residualvec[1] << endl;
                    double rndselection = distribution5(gen);
                    double rndculling = distribution5(gen);
                    double rndphen1 = distribution5(gen);
                    double rndphen2 = distribution5(gen);
                    double rndgeno = distribution5(gen);
                    Animal animal(ind,0,0,0,0,Sex,0,1,0,0,0,rndselection,rndculling,rndphen1,rndphen2,rndgeno,0.0,0.0,0.0,0.0,0.0,homozygouscount_lethal, heterzygouscount_lethal, homozygouscount_sublethal, heterzygouscount_sublethal,lethalequivalent,Homoz,relativeviability,MA,QT,"","","",0.0,"",Phenotypevec,vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0),GenotypicValuevec,BreedingValuevec,DominanceDeviationvec,Residualvec,0.0,0.0,vector<double>(SimParameters.getnumbertraits()),vector<double>(SimParameters.getnumbertraits()),vector<std::string>(SimParameters.getnumbertraits()),"selcand");
                    /* Then place given animal object in population to store */
                    population.push_back(animal);
                    ind++;                              /* Increment Animal ID by one for next individual */
                }
                delete [] MarkerGenotypes; delete [] QTLGenotypes;
            }
            //for(int i = 0; i < population.size(); i++)
            //for(int i = 0; i < 15; i++)
            //{
            //    //cout << population[i].getID() << " " << (population[i].get_Phenvect())[0] << " " << (population[i].get_GVvect())[0] << " ";
            //    //cout << (population[i].get_BVvect())[0] << " " << (population[i].get_DDvect())[0] << " " << (population[i].get_Rvect())[0] << " ";
            //    //cout << (population[i].get_BVvectFalc())[0] << " " << (population[i].get_DDvectFalc())[0] << endl;
            //    //cout << population[i].getID() << " " << (population[i].get_Phenvect())[0] << " " << (population[i].get_Phenvect())[1] << " ";
            //    //cout << (population[i].get_GVvect())[0] << " " << (population[i].get_GVvect())[1] << " " << (population[i].get_BVvect())[0] << " ";
            //    //cout << (population[i].get_BVvect())[1] << " " << (population[i].get_DDvect())[0] << " " << (population[i].get_DDvect())[1] << " ";
            //    //cout << (population[i].get_Rvect())[0] << " " << (population[i].get_Rvect())[1] << endl;
            //}
            UpdateTBV_TDD_Statistical(Population1,population,SimParameters);
            if(SimParameters.getnumbertraits() == 2){traitcorrelation(population,logfile);}
            male_female_status.clear();
            UpdateGenoPhenoStatus(Population1,population,SimParameters,OUTPUTFILES,0,"outputselectioncand"); /* Update PhenoGenoStatus for animals */
            logfile << "   Number of Founder's that Died due to fitness: " << LethalFounder << endl;
            Population1.update_numdeadfitness(0,LethalFounder);
            ExpectedHeterozygosity(population,Population1,0,logfile);    /* Calculate expected heterozygosity in progeny */
            logfile << "Finished Creating Founder Population (Size: " << population.size() << ")." << endl << endl;
            // Compute mean genotypic value in founder generation
            vector <double> BaseGenGV_vec(SimParameters.getnumbertraits(),0.0);
            vector <double> BaseGenBV_vec(SimParameters.getnumbertraits(),0.0);
            vector <double> BaseGenDD_vec(SimParameters.getnumbertraits(),0.0);
            for(int i = 0; i < population.size(); i++)
            {
                for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                {
                    BaseGenGV_vec[j] += (population[i].get_GVvect())[j];
                    BaseGenBV_vec[j] += (population[i].get_BVvect())[j];
                    BaseGenDD_vec[j] += (population[i].get_DDvect())[j];
                    if(i == population.size() -1)
                    {
                        BaseGenGV_vec[j] /= population.size(); BaseGenBV_vec[j] /= population.size(); BaseGenDD_vec[j] /= population.size();
                    }
                }
            }
            GenerateBaseValues(population,BaseGenGV_vec,BaseGenBV_vec,BaseGenDD_vec);
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
            if(SimParameters.getLDDecay() == "yes"){LD_Option(SimParameters,population,population_QTL,OUTPUTFILES,"yes",logfile);}
            if(SimParameters.getOutputWindowVariance() == "yes")
            {
                string foundergen = "yes";
                WindowVariance(SimParameters,population,population_QTL,foundergen,OUTPUTFILES);
                logfile << "Generating Additive and Dominance Window Variance."<< endl << endl;
            }
            //////////////////////////////////////////////////////////////////////////
            // Create Distribution of Mating Pairs to Draw From                     //
            //////////////////////////////////////////////////////////////////////////
            const int n=10000;                          /* number of samples to draw */
            const int nstars=100;                      /* maximum number of stars to distribute for plot */
            const int nintervals=20;                   /* number of intervals for plot */
            vector < double > number(10000,0.0);        /* stores samples */
            int p[nintervals] = {};                    /* stores number in each interval */
            sftrabbit::beta_distribution<> beta(SimParameters.getBetaDist_alpha(),SimParameters.getBetaDist_beta());     /* Beta Distribution */
            for (int i = 0; i < n; ++i){number[i] = beta(gen); ++p[int(nintervals*number[i])];}                  /* Get sample and put count in interval */
            sort(number.begin(),number.end());
            if(SimParameters.getBetaDist_alpha() != 1.0 || SimParameters.getBetaDist_beta() != 1.0)
            {
                /* Plot distribution of mating pairs as an animal gets older */
                logfile << "Distribution of Mating Pairs \n";
                logfile << "Beta distribution (" << SimParameters.getBetaDist_alpha() << "," << SimParameters.getBetaDist_beta() << "):" << endl;
                for (int i=0; i<nintervals; ++i)
                {
                    logfile << float(i)/nintervals << "-" << float(i+1)/nintervals << ": " << "\t" << std::string(p[i]*nstars/n,'*') << std::endl;
                }
            }
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
                for(int j = 0; j < tempgenofreqcalc[0].size(); j++){M[(i*tempgenofreqcalc[0].size())+j] = i - (2 * founderfreq[j]);}
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
                    stringstream strStreamend (stringstream::in | stringstream::out);
                    if(SimParameters.getnumbertraits() == 1)
                    {
                        strStreamend << " " << quantvalues[i][0] - BaseGenGV_vec[0];
                        strStreamend << " " << quantvalues[i][1] - BaseGenBV_vec[0];
                        strStreamend << " " << quantvalues[i][2] - BaseGenDD_vec[0];
                    }
                    if(SimParameters.getnumbertraits() == 2)
                    {
                        strStreamend << " " << quantvalues[i][0] - BaseGenGV_vec[0];
                        strStreamend << " " << quantvalues[i][2] - BaseGenBV_vec[0];
                        strStreamend << " " << quantvalues[i][4] - BaseGenDD_vec[0];
                        strStreamend << " " << quantvalues[i][1] - BaseGenGV_vec[1];
                        strStreamend << " " << quantvalues[i][3] - BaseGenBV_vec[1];
                        strStreamend << " " << quantvalues[i][5] - BaseGenDD_vec[1];
                    }
                    //cout << leftfitnessstring[i] << (strStreammiddlefit.str()) << rightfitnessstring[i] << strStreamend.str() << endl;
                    std::ofstream outputlow(OUTPUTFILES.getloc_lowfitnesspath().c_str(), std::ios_base::app | std::ios_base::out);
                    outputlow << leftfitnessstring[i] << (strStreammiddlefit.str()) << rightfitnessstring[i] << strStreamend.str() << endl;
                }
            }
            leftfitnessstring.clear(); rightfitnessstring.clear(); markerlowfitness.clear();
            for(int i = 0; i < quantvalues.size(); i++){quantvalues[i].clear();}
            quantvalues.clear(); tempgenofreqcalc.clear();
            int TotalAnimalNumber = 0;                  /* Counter to determine how many animals their are for full matrix sizes */
            int TotalOldAnimalNumber = 0;               /* Counter to determine size of old animal matrix */
            /* Generate header for zipped master genotype file */
            gzofstream zippedgeno;
            zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
            if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
            zippedgeno << "ID Marker QTL" << endl;
            zippedgeno.close();
            //std::ofstream outputmastgeno(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
            //outputmastgeno << "ID Marker QTL" << endl;
            time_t t_end = time(0);
            cout << "Constructed Trait Architecture and Founder Genomes. (Took: " << difftime(t_end,t_start) << " seconds)" << endl << endl;
            /* If doing Bayes ebv prediction create a new file and add a flag to get prior's based on h2 */
            if(SimParameters.getEBV_Calc() == "bayes")
            {
                fstream checkbayes;
                checkbayes.open(OUTPUTFILES.getloc_Bayes_PosteriorMeans().c_str(), std::fstream::out | std::fstream::trunc); checkbayes.close();
                std::ofstream flagbayes(OUTPUTFILES.getloc_Bayes_PosteriorMeans().c_str(), std::ios_base::app | std::ios_base::out);
                flagbayes<< "GeneratePriorH2" << endl;
            }
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////                   Start Looping through generations based on forward-in-time simulation techniques                         ////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            cout << "Begin Simulating Generations:" << endl;
            vector < string > tempqtlfreqcalc;
            for(int i = 0; i < population.size(); i++){tempqtlfreqcalc.push_back(population[i].getQTL());}
            double* qtlfreq = new double[tempqtlfreqcalc[0].size()];               /* Array that holds SNP that were declared as Markers and QTL */
            frequency_calc(tempqtlfreqcalc, qtlfreq);                              /* Function to calculate snp frequency */
            /* Create a vector of string that describe what selection is based on.  *
            /* Currently just change how founders are selected later change after n generation */
            vector < string > SelectionVector((SimParameters.getGener()),"");
            vector < string > EBVCalculationVector((SimParameters.getGener()),"");
            vector < string > Indexsd_calc((SimParameters.getGener()),"NA"); string alreadychecked = "no";
            for(int i = 0; i < (SimParameters.getGener()); i++)
            {
                if(i < SimParameters.getGenfoundsel())
                {
                    SelectionVector[i] = SimParameters.getfounderselect(); EBVCalculationVector[i] = "NO";
                    if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){Indexsd_calc[i] = "NA";}
                }
                if(i >= SimParameters.getGenfoundsel())
                {
                    if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
                    {
                        if(alreadychecked == "yes"){Indexsd_calc[i] = "NO";}
                        if(alreadychecked == "no"){Indexsd_calc[i] = "YES"; alreadychecked = "yes";}
                    }
                    SelectionVector[i] = SimParameters.getSelection();
                    if(SimParameters.getEBV_Calc()!="SKIP"){EBVCalculationVector[i] = "YES";
                    } else {EBVCalculationVector[i] = "NO";}
                }
            }
            SelectionVector.push_back(SelectionVector[SelectionVector.size()-1]); /* Ensures EBV are calculated last generation */
            //for(int i = 0; i < SelectionVector.size(); i++){cout<<SelectionVector[i]<<" "<<EBVCalculationVector[i]<<" "<<Indexsd_calc[i]<<endl;}
            /**************************************************************/
            /* Output founder animals to Pheno_Pedigree and GMatrix Files */
            /**************************************************************/
            TotalOldAnimalNumber = TotalAnimalNumber;           /* Size of old animal matrix */
            TotalAnimalNumber = OutputPedigree_GenomicEBV(OUTPUTFILES,population,TotalAnimalNumber,SimParameters);
            for(int Gen = 1; Gen < (SimParameters.getGener() + 1); Gen++)
            {
                time_t intbegin_time = time(0);
                logfile << "------ Begin Generation " << Gen << " -------- " << endl;
                if(SimParameters.getOutputTrainReference() == "yes" && Gen == 1)            /* add header to output correlation file */
                {
                    ofstream outputamax; outputamax.open(OUTPUTFILES.getloc_Amax_Output() .c_str());
                    outputamax << "Generation AvgAmax...." << endl;
                    outputamax.close();
                    ofstream outputparentprogenycor; outputparentprogenycor.open(OUTPUTFILES.getloc_TraitReference_Output().c_str());
                    if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv"){
                        outputparentprogenycor << "ID T1_EBV T1_TBV T2_EBV T2_TBV Generation Group" << endl;
                    } else{outputparentprogenycor << "ID EBV TBV Generation Group" << endl;}
                    outputparentprogenycor.close();
                }
                /* If have ROH genome summary as an option; do proportion of genome in ROH for reach individual */
                if(SimParameters.getmblengthroh() != -5){Proportion_ROH(SimParameters,population,OUTPUTFILES,logfile);}
                /* If want to identify haplotypes do it now */
                if(SimParameters.getstartgen() != -5)
                {
                    EnterHaplotypeFinder(SimParameters,trainregions,population,Gen,retraingeneration,unfav_direc,OUTPUTFILES,logfile);
                }
                time_t start_block = time(0); time_t start; time_t end;
                /* Get ID for last animal; Start ID represent the first ID that will be used for first new animal */
                int StartID = population.size();
                StartID = population[StartID - 1].getID() + 1;
                vector < int > trainanimals;
                /* Prior to estimating breeding values update phenostatus */
                UpdateGenoPhenoStatus(Population1,population,SimParameters,OUTPUTFILES,Gen,"preebvcalc",SelectionVector);
                if((Population1.get_qtlscalefactadd())[0]!=0 && EBVCalculationVector[Gen-1]=="YES" && SimParameters.getEBV_Calc()!="bayes")
                {
                    logfile << "   Generate Estimated Breeding Values based on " << SimParameters.getEBV_Calc() << " method:" << endl;
                    vector <double> estimatedsolutions; vector < double > trueaccuracy;
                    Generate_BLUP_EBV(SimParameters,population,estimatedsolutions,trueaccuracy,logfile,trainanimals,TotalAnimalNumber,TotalOldAnimalNumber,Gen,M,scale,haplib,OUTPUTFILES);
                }
                if((Population1.get_qtlscalefactadd())[0]!=0 && EBVCalculationVector[Gen-1]=="YES" && SimParameters.getEBV_Calc()=="bayes")
                {
                    logfile << "   Generate Estimated Breeding Values utilizing bayesian regression methods." << endl;
                    logfile << "       - Bayesian Regression Method: " << SimParameters.getmethod() << endl;
                    /* Generate Breeding Values based on Bayesian regression models (i.e. Bayes A, B, C or RR) */
                    vector < double > estimatedsolutions;
                    bayesianestimates(SimParameters,population,Gen,estimatedsolutions,OUTPUTFILES,logfile);
                    estimatedsolutions.clear();
                    Inbreeding_Pedigree(population,OUTPUTFILES);
                }
                time_t end_block = time(0);
                if((Population1.get_qtlscalefactadd())[0]!=0 && EBVCalculationVector[Gen-1]=="YES")
                {
                    logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
                }
                //if(Gen == 7){exit (EXIT_FAILURE);}
                if((Population1.get_qtlscalefactadd())[0] == 0 || EBVCalculationVector[Gen-1]=="NO"){Inbreeding_Pedigree(population,OUTPUTFILES);}
                if(SimParameters.getOutputTrainReference() == "yes")
                {
                    if((Population1.get_qtlscalefactadd())[0] != 0 && EBVCalculationVector[Gen-1]=="YES" && Gen > 1)
                    {
                        trainrefcor(SimParameters,population,OUTPUTFILES,Gen);
                    }
                }
                for(int i = 0; i < population.size(); i++) /* Update Stage of Animal */
                {
                    if(population[i].getAge() == 1){population[i].UpdateAnimalStage("selcandebv");}
                }
                /* After estimating breeding values update phenostatus */
                UpdateGenoPhenoStatus(Population1,population,SimParameters,OUTPUTFILES,Gen,"postebvcalc",SelectionVector);
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                   Start of Selection Functions                                  //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                matingindividuals.clear();                                   /* Make mate individual class starts out as empty */
                logfile << "   Begin " << SelectionVector[Gen-1] << " Selection of offspring: " << endl;
                time_t start_block1 = time(0);
                string tempselectionscen = SelectionVector[Gen-1];
                if(Indexsd_calc[Gen-1] != "NA")
                {
                    if(Indexsd_calc[Gen-1] == "YES"){GenerateBVIndex(population,Population1,SimParameters,1,logfile);}
                    if(Indexsd_calc[Gen-1] == "NO"){GenerateBVIndex(population,Population1,SimParameters,2,logfile);}
                }
                /* truncation selection variables */
                if(tempselectionscen=="random" || tempselectionscen=="phenotype" || tempselectionscen=="tbv" || tempselectionscen=="ebv" || tempselectionscen=="index_tbv" || tempselectionscen=="index_ebv")
                {
                    truncationselection(population,SimParameters,tempselectionscen,Gen,OUTPUTFILES,Population1,logfile);
                }
                /* optimal contribution selection */
                if(tempselectionscen == "ocs")
                {
                    optimalcontributionselection(population,matingindividuals,haplib,SimParameters,tempselectionscen,M,scale,OUTPUTFILES,Gen,logfile);
                }
                time_t end_block1 = time(0);
                logfile <<"   Finished "<<SelectionVector[Gen-1]<<" Selection of parents (Time: "<<difftime(end_block1,start_block1)<<" seconds).\n\n";
                logfile <<"   Male and Female Age Distribution: " << endl;
                breedingagedistribution(population,SimParameters,logfile);
                logfile << endl;
                for(int i = 0; i < population.size(); i++) /* Update Stage of Animal */
                {
                    if(population[i].getAge() == 1){population[i].UpdateAnimalStage("parent");}
                }
                /* After Selection Animals update phenostatus */
                UpdateGenoPhenoStatus(Population1,population,SimParameters,OUTPUTFILES,Gen,"postselcalc",SelectionVector);
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                 Start of Mating Design Functions                                //////////////
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
                        generatematingpairs(matingindividuals,population,haplib,SimParameters,matingscenario,OUTPUTFILES,M,scale,logfile);
                    }
                    /* Sire not distributed equally across dams */
                    if(SimParameters.getmaxsireprop()!= -5 && SimParameters.getMating()=="index" && SelectionVector[Gen-1] != "random")
                    {
                        logfile << "       - Mating design based on index values which results in sire not being used equally across animals: " << endl;
                        indexmatingdesign(matingindividuals,population,haplib,SimParameters,OUTPUTFILES,M,scale,logfile);
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
                    TotalQTL += Population1.get_qtlperchr()[i];
                    TotalMarker += Population1.get_markerperchr()[i];
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
                    if(m_counter < (Population1.get_markerindex()).size())                          /* ensures doesn't go over and cause valgrind error */
                    {
                        if((Population1.get_markerindex())[m_counter] == i)
                        {
                            FullMap[i] = (Population1.get_markermapposition())[m_counter]; m_counter++;     /* If is marker put in full map */
                        }
                    }
                    if(qtl_counter < (Population1.get_qtlindex()).size())                           /* ensures doesn't go over and cause valgrind error */
                    {
                        if((Population1.get_qtlindex())[qtl_counter] == i)
                        {
                            FullMap[i] = (Population1.get_qtl_mapposition())[qtl_counter]; qtl_counter++;   /* If is quant QTL put in full map */
                        }
                    }
                }
                /* Put Sequence SNP map information in 2-dim array so only have to read in once; row = numchr col= max number of SNP */
                int col = (Population1.get_chrsnplength())[0];                             /* intitialize to figure out max column number */
                /* Find max number of SNP within a chromosome */
                for(int i = 1; i < SimParameters.getChr(); i++){if((Population1.get_chrsnplength())[i] > col){col = (Population1.get_chrsnplength())[i];}}
                // store in 2-D vector using this may get large so need to store dynamically
                vector < vector < double > > SNPSeqPos;
                /* read in each chromsome files */
                for(int i = 0; i < SimParameters.getChr(); i++)
                {
                    vector < double > temp;
                    string mapfilepath = path + "/" + SimParameters.getOutputFold() + "/" + (Population1.get_mapfiles())[i];
                    ifstream infile8;
                    infile8.open(mapfilepath);
                    if (infile8.fail()){cout << "Error Opening MaCS Map File\n"; exit (EXIT_FAILURE);}
                    for(int j = 0; j < (Population1.get_chrsnplength())[i]; j++)
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
                vector < vector < double > > MutationAdd_quan;                  /* Additive effect of allele quantitative */
                vector < vector < double > > MutationDom_quan;                  /* Dominance effect of mutant allele quantitative */
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
                    //cout << a << " " << population[a].getMatings() << endl;
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
                    for(int i = 0; i < qtl_counter; i++){qtl_geno[i] = qtl_qn[i] - 48;}             /* QTL genotypes */
                    m_counter = 0;                                                                  /* where at in marker array */
                    qtl_counter = 0;                                                                /* where at in quantitative qtl array */
                    /* Combine them into marker and QTL array for crossovers based on index value */
                    for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                    {
                        if(m_counter < (Population1.get_markerindex()).size())                      /* ensures doesn't go over and cause valgrind error */
                        {
                            if((Population1.get_markerindex())[m_counter] == i){geno[i]=markgeno[m_counter]; m_counter++;} /* If is marker put in geno */
                        }
                        if(qtl_counter < (Population1.get_qtlindex()).size())                       /* ensures doesn't go over and cause valgrind error */
                        {
                            if((Population1.get_qtlindex())[qtl_counter] == i){geno[i]=qtl_geno[qtl_counter]; qtl_counter++;} /* If is QTL put in geno */
                        }
                    }
                    /* but back into string to make storing easier then once gamete formation complete add back in mutations */
                    string parentGENO;
                    stringstream strStream (stringstream::in | stringstream::out);
                    for (int i=0; i < (TotalQTL + TotalMarker); i++){strStream << geno[i];}
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
                                //cout << mat << "(" <<Nmb_Mutations<<" ";
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
                                            if(temptype < 0.333){MutationType.push_back(2);}                       /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.333 && temptype < 0.666){MutationType.push_back(4);}  /* Mutation is a Lethal Fitness QTL */
                                            if(temptype >= 0.666){MutationType.push_back(5);}                      /* Mutation is a subLethal Fitness QTL */
                                        }
                                        if((SimParameters.get_QTL_chr())[c] > 0 && (SimParameters.get_FTL_lethal_chr())[c] == 0 && (SimParameters.get_FTL_sublethal_chr())[c] > 0)
                                        {
                                            if(temptype < 0.5){MutationType.push_back(2);}                         /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.5){MutationType.push_back(5);}                        /* Mutation is a subLethal Fitness QTL */
                                        }
                                        if((SimParameters.get_QTL_chr())[c] > 0 && (SimParameters.get_FTL_lethal_chr())[c] > 0 && (SimParameters.get_FTL_sublethal_chr())[c] == 0)
                                        {
                                            if(temptype < 0.5){MutationType.push_back(2);}                         /* Mutation is a Quantitative QTL */
                                            if(temptype >= 0.5){MutationType.push_back(4);}                        /* Mutation is a Lethal Fitness QTL */
                                        }
                                        if((SimParameters.get_QTL_chr())[c] > 0 && (SimParameters.get_FTL_lethal_chr())[c] == 0 && (SimParameters.get_FTL_sublethal_chr())[c] == 0)
                                        {
                                            MutationType.push_back(2);
                                        }
                                        MutationLoc.push_back(distribution1(gen) + c + 1);
                                        for(int j = 0; j < (Population1.get_chrsnplength())[c]; j++)
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
                                            if(SimParameters.getnumbertraits() == 1)
                                            {
                                                vector < double> tempormut;
                                                /******* QTL Additive Effect (Gamma *******/
                                                std::gamma_distribution <double> distribution1(SimParameters.getGamma_Shape(),SimParameters.getGamma_Scale());
                                                tempormut.push_back(distribution1(gen)); MutationAdd_quan.push_back(tempormut); tempormut.clear();
                                                /****** QTL Dominance Effect *******/
                                                /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                                                std::normal_distribution<double>distribution2(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                                                double temph = distribution2(gen); temph = temph * MutationAdd_quan[CounterMutationIndex][0];
                                                tempormut.push_back(temph); MutationDom_quan.push_back(tempormut); tempormut.clear();
                                                /* Determine sign of additive effect */
                                                /* 1 tells you range so can sample from 0 to 1 and 0.5 is the frequency */
                                                std::binomial_distribution<int> distribution3(1,0.5);
                                                int signadd = distribution3(gen);
                                                if(signadd == 1){MutationAdd_quan[CounterMutationIndex][0] = MutationAdd_quan[CounterMutationIndex][0] * -1;}
                                                /* Scale effects based on Founder Scale Factor */
                                                /* Scaling based on the original set of effects should put it close to the effect sizes of the founder QTL */
                                                MutationAdd_quan[CounterMutationIndex][0] = MutationAdd_quan[CounterMutationIndex][0]*(Population1.get_qtlscalefactadd())[0];
                                                MutationDom_quan[CounterMutationIndex][0] = MutationDom_quan[CounterMutationIndex][0]*(Population1.get_qtlscalefactdom())[0];
                                                MutationAdd_fit.push_back(0.0); MutationDom_fit.push_back(0.0);
                                            }
                                            if(SimParameters.getnumbertraits() == 2)
                                            {
                                                vector < double> tempormut;
                                                /*******                QTL Additive Effect                  ********/
                                                /*          Generate three random gamma's with same Scale           */
                                                /* x1 common to both; x2 common to trait one; x3 common to trait 3; */
                                                /* Incommon shape: fullshape * correlation */
                                                double shapeX1 = SimParameters.getGamma_Shape() * SimParameters.get_Var_Additive()[1];
                                                /* Not incommon shape: fullshape - Incommon shape */
                                                double shapeX2_X3 = SimParameters.getGamma_Shape() - shapeX1;
                                                /* use binomial to figure out whether effect negative or positive */
                                                std::binomial_distribution<int> distribution1(1,0.5);
                                                /* Generate x1 */
                                                /* X1 for incommon portion*/
                                                std::gamma_distribution <double> distribution2(shapeX1,SimParameters.getGamma_Scale());
                                                double Y1 = distribution2(gen);
                                                int signadd = distribution1(gen);
                                                /*Assign negative effect with a 50% probability */
                                                if(signadd == 1){Y1 = Y1 * -1;}
                                                /* Generate x2 */
                                                std::gamma_distribution <double> distribution3(shapeX2_X3,SimParameters.getGamma_Scale());
                                                double Y2 = distribution3(gen);
                                                signadd = distribution1(gen);
                                                /*Assign negative effect with a 50% probability */
                                                if(signadd == 1){Y2 = Y2 * -1;}
                                                /* Generate x3 */
                                                double Y3 = distribution3(gen);
                                                signadd = distribution1(gen);
                                                /*Assign negative effect with a 50% probability */
                                                if(signadd == 1){Y3 = Y3 * -1;}
                                                tempormut.push_back((Y1 + Y2)); tempormut.push_back((Y1 + Y3));
                                                MutationAdd_quan.push_back(tempormut); tempormut.clear();
                                                /****** QTL Dominance Effect *******/
                                                /* relative dominance degrees simulated than multiply Additive * dominance degrees */
                                                std::normal_distribution<double> distribution4(SimParameters.getNormal_meanRelDom(),SimParameters.getNormal_varRelDom());
                                                double temph = distribution4(gen);
                                                temph = temph * (abs((Y1 + Y2))); tempormut.push_back(temph);
                                                temph = distribution3(gen);
                                                temph = temph * (abs((Y1 + Y3))); tempormut.push_back(temph);
                                                MutationDom_quan.push_back(tempormut); tempormut.clear();
                                                /* Scale effects based on Founder Scale Factor */
                                                /* Scaling based on the original set of effects should put it close to the effect sizes of the founder QTL */
                                                MutationAdd_quan[CounterMutationIndex][0] = MutationAdd_quan[CounterMutationIndex][0]*(Population1.get_qtlscalefactadd())[0];
                                                MutationAdd_quan[CounterMutationIndex][1] = MutationAdd_quan[CounterMutationIndex][1]*(Population1.get_qtlscalefactadd())[1];
                                                MutationDom_quan[CounterMutationIndex][0] = MutationDom_quan[CounterMutationIndex][0]*(Population1.get_qtlscalefactdom())[0];
                                                MutationDom_quan[CounterMutationIndex][1] = MutationDom_quan[CounterMutationIndex][1]*(Population1.get_qtlscalefactdom())[1];
                                                MutationAdd_fit.push_back(0.0); MutationDom_fit.push_back(0.0);
                                            }
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
                                            vector < double> tempormut(SimParameters.getnumbertraits(),0.0);
                                            MutationAdd_quan.push_back(tempormut); MutationDom_quan.push_back(tempormut);
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
                                            vector < double> tempormut(SimParameters.getnumbertraits(),0.0);
                                            MutationAdd_quan.push_back(tempormut); MutationDom_quan.push_back(tempormut);
                                        }
                                        CounterMutationIndex++;
                                    }
                                }
                                ///////////////////////////////////////////////////////////////////////
                                /// Create gamete based on markers and QTL (not including mutation) ///
                                ///////////////////////////////////////////////////////////////////////
                                //cout << "Got Here ";
                                int homo1[(Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c])];
                                int homo2[(Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c])];
                                double mappos[(Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c])];
                                int newhaplotype[(Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c])];
                                /* Copy current chromosome haplotypes int temp homo1 and homo2 and their associated map position */
                                int spot = 0;                                       /* indicator to determine where at in temporary arrays */
                                for(int i = genocounter; i < (genocounter + (Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c])); i++)
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
                                if(nCx > 0){
                                    position_dummy = 0;}
                                else {position_dummy = 1;}
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
                                        if(numNextCx > nCx){break;}
                                        /* If the number of Loci is equal to countSNP break then recombination happend at last one can't be observed */
                                        if(countSNP > ((Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c]) - 1))
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
                                            if(numNextCx > nCx){break;}
                                            /* Change the location of next SNP */
                                            locNextCx = lCx[numNextCx -1];
                                            /* if the mapPos of the SNP is also larger than the next Cx  location then decrement countSNP. Outside this */
                                            /* if block countSNP will incremented again, so that countSNP stays the same, until the mapPos of countSNP */
                                            /* is not larger than the next Cx anymore. This assures that multiple Cx between two SNP are detected. */
                                            if(countSNP <= ((Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c]) - 1))
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
                                        if((numNextCx > nCx) || (*(SNPaftCx+numNextCx-1) > (Population1.get_markerperchr()[c]+Population1.get_qtlperchr()[c])))
                                        {
                                            while(countSNP <= ((Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c]) - 1))
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
                                    for(countSNP = 0; countSNP < (Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c]); ++countSNP)
                                    {
                                        *(newhaplotype + countSNP) = *(currentHomologe_ptr + countSNP);
                                    }
                                }
                                /* Done with current chromosome copy gamete onto full gamete */
                                int j = 0;                                                      /* j is a counter for location of gamete */
                                /* Copy to full new gamete */
                                for(int i = genocounter; i < (genocounter + (Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c])); i++)
                                {
                                    fullgamete[i] = newhaplotype[j];
                                    j++;
                                } /* close for loop */
                                /* Update position of where you are at in full gamete */
                                genocounter = genocounter + Population1.get_markerperchr()[c] + Population1.get_qtlperchr()[c];
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
                            QTL_new_old tempa(MutationLoc[i],vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0), MutationAdd_quan[i][0], MutationDom_quan[i][0], temp, Gen, stringfreq,"");
                            tempa.update_Additivevect(0,MutationAdd_quan[i][0]);
                            tempa.update_Dominancevect(0,MutationDom_quan[i][0]);
                            if(SimParameters.getnumbertraits() == 2)        /* put second trait as 0.0 */
                            {
                                tempa.update_Additivevect(1,MutationAdd_quan[i][1]);
                                tempa.update_Dominancevect(1,MutationDom_quan[i][1]);
                            }
                            population_QTL.push_back(tempa);
                        }
                        if(MutationType[i] == 4 || MutationType[i] == 5)
                        {
                            QTL_new_old tempa(MutationLoc[i],vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0), MutationAdd_fit[i], MutationDom_fit[i], temp, Gen, stringfreq,"");
                            population_QTL.push_back(tempa);
                        }
                    }
                    //for(int i = 0; i < population_QTL.size(); i++)
                    //{
                    //    cout << population_QTL[i].getLocation() << " ";
                    //    for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                    //    {
                    //        cout << (population_QTL[i].get_Additivevect())[j] << " " << (population_QTL[i].get_Dominancevect())[j] << " ";
                    //    }
                    //    cout << population_QTL[i].getType() << " ";
                    //    cout << population_QTL[i].getGenOccured() << " " << population_QTL[i].getFreq() << endl;
                    //}
                    logfile << "       - New QTL's Added to QTL class object (Total: " << population_QTL.size() << ")." << endl;
                    /* Update number of qtl and markers per chromosome */
                    for(int i = 0; i < SimParameters.getChr(); i++)
                    {
                        int numbqtl = 0;
                        vector < double > allqtllocations;
                        for(int j = 0; j < population_QTL.size(); j++){allqtllocations.push_back(population_QTL[j].getLocation());}
                        /* Remove duplicates if simulating mutations affecting fitness and quantitiative*/
                        allqtllocations.erase(unique(allqtllocations.begin(),allqtllocations.end()),allqtllocations.end());
                        /* Location is in double (1.2345) and when convert to integer will always round down so can get Chr */
                        for(int j = 0; j < allqtllocations.size(); j++)
                        {
                            int temp = allqtllocations[j]; if(temp == i + 1){numbqtl += 1;}
                        }
                        Population1.update_qtlperchr(i,numbqtl);
                    }
                    /////////////////////////////////////////////////////////
                    /// Part8: Reconstruct Index and Update Map           ///
                    /////////////////////////////////////////////////////////
                    /* put all old qtl effects and new mutations into Old_* */
                    vector <double> Old_mappos;     /* position across markers and all qtl */
                    vector <int> Old_newmutat;      /* whether it is a new mutation or not */
                    vector <int> Old_type;          /* (1 = marker,2 = quant QTL,3 = quant + fitness QTL, 4 = lethal fitness QTL, 5 = sublethal fitness QTL) */
                    vector < vector <double>> Old_Add_Quan;   /* Old additive effect quantitative */
                    vector < vector <double>> Old_Dom_Quan;   /* Old dominance effect quantitative */
                    vector <double> Old_Add_Fit;    /* Old additive effect fitness */
                    vector < double > Old_Dom_Fit;  /* Old additive effect fitness */
                    vector < int > Old_QTL_Allele;  /* Old unfavorable allele for fitness */
                    /******************************************************/
                    /*           Old QTL and Marker locations             */
                    /******************************************************/
                    m_counter = 0; qtl_counter = 0;
                    for(int i = 0; i < (TotalQTL + TotalMarker); i++)
                    {
                        if(m_counter < (Population1.get_markerindex()).size())                          /* ensures doesn't go over and cause valgrind error */
                        {
                            if((Population1.get_markerindex())[m_counter] == i)
                            {
                                Old_mappos.push_back((Population1.get_markermapposition())[m_counter]);
                                Old_newmutat.push_back(0); Old_type.push_back(1);
                                vector < double> tempormut(SimParameters.getnumbertraits(),-5.0);
                                Old_Add_Quan.push_back(tempormut); Old_Dom_Quan.push_back(tempormut);
                                Old_Add_Fit.push_back(-5.0); Old_Dom_Fit.push_back(-5.0); Old_QTL_Allele.push_back(-5); m_counter++;
                            }
                        }
                        if(qtl_counter < (Population1.get_qtlindex()).size())                           /* ensures doesn't go over and cause valgrind error */
                        {
                            if((Population1.get_qtlindex())[qtl_counter] == i)
                            {
                                Old_mappos.push_back((Population1.get_qtl_mapposition())[qtl_counter]);
                                Old_newmutat.push_back(0);
                                Old_type.push_back((Population1.get_qtl_type())[qtl_counter]);
                                
                                vector < double> tempormutadd; vector < double> tempormutdom;
                                for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                                {
                                    tempormutadd.push_back(Population1.get_qtl_add_quan(qtl_counter,j));
                                    tempormutdom.push_back(Population1.get_qtl_dom_quan(qtl_counter,j));
                                }
                                Old_Add_Quan.push_back(tempormutadd); Old_Dom_Quan.push_back(tempormutdom);
                                Old_Add_Fit.push_back((Population1.get_qtl_add_fit())[qtl_counter]);
                                Old_Dom_Fit.push_back((Population1.get_qtl_dom_fit())[qtl_counter]);
                                Old_QTL_Allele.push_back((Population1.get_qtl_allele())[qtl_counter]); qtl_counter++;
                            }
                        }
                    }
                    /******************************************************/
                    /*                  New Mutations                     */
                    /******************************************************/
                    for(int i = 0; i < CounterMutationIndex; i++)
                    {
                        Old_mappos.push_back(MutationLoc[i]); Old_newmutat.push_back(1); Old_type.push_back(MutationType[i]);
                        vector < double> tempormutadd; vector < double> tempormutdom;
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            tempormutadd.push_back(MutationAdd_quan[i][j]);
                            tempormutdom.push_back(MutationDom_quan[i][j]);
                        }
                        Old_Add_Quan.push_back(tempormutadd); Old_Dom_Quan.push_back(tempormutdom);
                        Old_Add_Fit.push_back(MutationAdd_fit[i]); Old_Dom_Fit.push_back(MutationDom_fit[i]); Old_QTL_Allele.push_back(2); qtl_counter++;
                    }
                    //for(int i = 0; i < Old_mappos.size(); i++)
                    //{
                    //    cout <<Old_mappos[i] <<" "<<Old_newmutat[i]<<" "<<Old_type[i]<<" ";
                    //    for(int j = 0; j < SimParameters.getnumbertraits(); j++){cout << Old_Add_Quan[i][j]<<" "<<Old_Dom_Quan[i][j]<<" ";}
                    //    cout <<Old_Add_Fit[i]<<" "<<Old_Dom_Fit[i]<<" "<<Old_QTL_Allele[i]<<"   +   ";
                    //}
                    //cout << endl << endl;
                    //cout << Old_mappos.size() << endl;
                    /* Store position and index in a map, for fast finding */
                    map <double,int> newmap_snppos;
                    for(int i = 0; i < Old_mappos.size(); i++){newmap_snppos.insert(pair <double,int> (Old_mappos[i],i));}
                    //cout << newmap_snppos.size() << endl;
                    //map<double,int> :: iterator itra;
                    //for(itra = newmap_snppos.begin(); itra != newmap_snppos.end(); ++itra){cout << itra ->first << " " << itra ->second << "   +   ";}
                    //cout << endl << endl;
                    /**************************************************************************************/
                    /* Zero out old QTL arrays and Marker Index to ensure something doesn't get messed up */
                    /**************************************************************************************/
                    for(int i = 0; i < (Population1.get_markerindex()).size(); i++)
                    {
                        Population1.update_markerindex(i,0); Population1.update_mapposition(i,0.0);
                    }
                    /* Zero out all QTL arrays to ensure doesn't mess things up */
                    Population1.clear_qtl_type();
                    for(int i = 0; i < 5000; i++)
                    {
                        Population1.update_qtlindex(i,0); Population1.update_mapposition(i,0.0); Population1.add_qtl_type(0);
                        Population1.update_qtl_allele(i,0);
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            Population1.update_qtl_add_quan(i,j,0.0); Population1.update_qtl_dom_quan(i,j,0.0);
                        }
                        Population1.update_qtl_add_fit(i,0.0); Population1.update_qtl_dom_fit(i,0.0);
                    }
                    vector < int > New_QTLMarker(newmap_snppos.size(),0); /* Used to figure out where new mutations occurred within new map */
                    /**************************************/
                    /* Update marker and QTL Data vectors */
                    /**************************************/
                    //for(int i = 0; i < Old_type.size(); i++){cout << Old_type[i] << " ";}
                    //cout << endl;
                    map<double,int> :: iterator itr; int indexnum = 0; m_counter = 0; qtl_counter = 0;
                    for(itr = newmap_snppos.begin(); itr != newmap_snppos.end(); ++itr)
                    {
                        if(Old_type[itr ->second] == 1)                   /* Marker */
                        {
                            /* Should all be -5; if not exit out */
                            if(Old_Add_Quan[itr ->second][0] != -5.0 && Old_Dom_Quan[itr ->second][0] != -5.0 && Old_Add_Fit[itr ->second] != -5.0 && Old_Dom_Fit[itr ->second] != -5.0 && Old_QTL_Allele[itr ->second] != -5)
                            {
                                cout << "Shouldn't Be here 1" << endl; exit (EXIT_FAILURE);
                            }
                            Population1.update_markerindex(m_counter,indexnum);
                            Population1.update_markermapposition(m_counter,Old_mappos[itr ->second]); m_counter++;
                        }
                        if(Old_type[itr ->second] > 1)                   /* QTL */
                        {
                            /* Shouldn't be -5; if not exit out */
                            if(Old_Add_Quan[itr ->second][0] == -5.0 && Old_Dom_Quan[itr ->second][0] == -5.0 && Old_Add_Fit[itr ->second] == -5.0 && Old_Dom_Fit[itr ->second] == -5.0 && Old_QTL_Allele[itr ->second] == -5)
                            {
                                cout << "Shouldn't Be here 2" << endl; exit (EXIT_FAILURE);
                            }
                            Population1.update_qtlindex(qtl_counter,indexnum);
                            Population1.update_mapposition(qtl_counter,Old_mappos[itr ->second]);
                            Population1.update_qtl_type(qtl_counter,Old_type[itr ->second]);
                            Population1.update_qtl_allele(qtl_counter,Old_QTL_Allele[itr ->second]);
                            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                            {
                                Population1.update_qtl_add_quan(qtl_counter,j,Old_Add_Quan[itr ->second][j]);
                                Population1.update_qtl_dom_quan(qtl_counter,j,Old_Dom_Quan[itr ->second][j]);
                            }
                            Population1.update_qtl_add_fit(qtl_counter,Old_Add_Fit[itr ->second]);
                            Population1.update_qtl_dom_fit(qtl_counter,Old_Dom_Fit[itr ->second]);
                            qtl_counter++;
                        }
                        New_QTLMarker[indexnum] = Old_newmutat[itr ->second];
                        indexnum++;
                        //cout << itr ->first << " " << itr ->second << " --- " << Old_mappos[itr ->second] << " " << Old_type[itr ->second] << endl;
                    }
                    /* Need to order mutation by location in genome */
                    int temp_mutAnim; int temp_mutGame; int temp_mutType; double temp_mutLoc; int temp_mutChr;
                    for(int i = 0; i < CounterMutationIndex-1; i++)
                    {
                        for(int j=i+1; j < CounterMutationIndex; j++)
                        {
                            if(MutationLoc[i] > MutationLoc[j])
                            {
                                /* put i values in temp variables */
                                temp_mutAnim=MutationAnim[i]; temp_mutGame=MutationGamete[i]; temp_mutType=MutationType[i];
                                temp_mutLoc=MutationLoc[i]; temp_mutChr=MutationChr[i];
                                /* swap lines */
                                MutationAnim[i]=MutationAnim[j]; MutationGamete[i]=MutationGamete[j]; MutationType[i]=MutationType[j];
                                MutationLoc[i]=MutationLoc[j]; MutationChr[i]=MutationChr[j];
                                /* put temp values in 1 backward */
                                MutationAnim[j]=temp_mutAnim; MutationGamete[j]=temp_mutGame; MutationType[j]=temp_mutType;
                                MutationLoc[j]=temp_mutLoc; MutationChr[j]=temp_mutChr;
                            }
                        }
                    }
                    //cout << MutationAnim.size() << endl;
                    //for(int i = 0; i < MutationAnim.size(); i++)
                    //{
                    //    cout << i << " " << MutationAnim[i] <<" "<< MutationGamete[i] << " " << MutationType[i] <<" ";
                    //   cout <<MutationLoc[i]<<" "<< MutationChr[i]<<" "<<endl;
                    //}
                    //cout << endl << endl;
                    logfile << "       - Updated Index and Map Positions (New Index Size: " << qtl_counter << ")." << endl;
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
                        int MutationCounter = 0;                                                        /* Keep track of which mutation you are adding */
                        int OldGameteCounter = 0;                                                   /* Keep track of where you are at in old gamete */
                        for(int i = 0; i < (TotalQTL + TotalMarker + CounterMutationIndex); i++)
                        {
                            /* SNP was a marker everyone in current generation has it */
                            if(New_QTLMarker[i] == 0){newhaplotype[i] = oldhaplotype[OldGameteCounter]; OldGameteCounter++;}
                            /* New Mutation was created therefore only one animal has it */
                            if(New_QTLMarker[i] == 1)
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
                        int MutationCounter = 0;                                                        /* Set to first mutation event */
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
                            if(New_QTLMarker[j] == 1){genomut[j] = 0;}       /* New Mutation only in gamete has it there for all animals are genotype 0 (11) */
                        }
                        /* Put back into QTL string */
                        int* updQTL = new int[5000];                                        /* QTL Genotypes */
                        qtl_counter = 0;                                                    /* Counter to keep track where at in quantitative QTL Index */
                        /* Loop through MarkerGenotypes array and place Genotypes based on Index value */
                        for(int j = 0; j < (TotalQTL + TotalMarker + CounterMutationIndex); j++)
                        {
                            if(j == (Population1.get_qtlindex())[qtl_counter]){updQTL[qtl_counter] = genomut[j]; qtl_counter++;}
                        }
                        /* Put back in string and update in animal class object */
                        stringstream strStreamQn (stringstream::in | stringstream::out);
                        for(int j=0; j < (qtl_counter); j++){strStreamQn << updQTL[j];}
                        string QTn = strStreamQn.str();
                        population[i].UpdateQTLGenotype(QTn);
                        delete [] updQTL;
                    }
                    logfile << "       - Updated Parent Genotypes with Mutations." << endl;
                    Old_mappos.clear(); Old_newmutat.clear(); Old_type.clear(); Old_Add_Quan.clear(); Old_Dom_Quan.clear(); Old_Add_Fit.clear();
                    Old_Dom_Fit.clear(); Old_QTL_Allele.clear(); newmap_snppos.clear(); New_QTLMarker.clear();
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
                                            //cout << AnimGam_ID[gamete] << " " << AnimGam_GamID[gamete] << " " << AnimGam_Sex[gamete] << " ";
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
                                            vector <int> MarkerGenotypes(Population1.getfullmarkernum(),0); /* Marker Genotypes; Will always be of this size */
                                            vector <int> QTLGenotypes((Population1.get_qtlindex()).size(),0);        /* QTL Genotypes */
                                            m_counter = 0;                                          /* Counter to keep track where at in Marker Index */
                                            qtl_counter = 0;                                        /* Counter to keep track where at in QTL Index */
                                            /* Fill Genotype Array */
                                            for(int j = 0; j < (TotalQTL + TotalMarker + CounterMutationIndex); j++) /* Place Genotypes based on Index value */
                                            {
                                                if(m_counter < (Population1.get_markerindex()).size())  /* ensures doesn't go over and cause valgrind error */
                                                {
                                                    if(j == (Population1.get_markerindex())[m_counter]){MarkerGenotypes[m_counter] = Geno[j]; m_counter++;}
                                                }
                                                if(qtl_counter < (Population1.get_qtlindex()).size())    /* ensures doesn't go over and cause valgrind error */
                                                {
                                                    if(j == (Population1.get_qtlindex())[qtl_counter]){QTLGenotypes[qtl_counter] = Geno[j]; qtl_counter++;}
                                                }
                                            }
                                            Geno.clear();
                                            /* Before you put the individual in the founder population need to determine if it dies */
                                            /* represents the multiplicative fitness effect across lethal and sub-lethal alleles */
                                            double relativeviability = 1.0;             /* Starts of as a viability of 1.0 */
                                            for(int j = 0; j < QTL_IndCounter; j++)
                                            {
                                                if((Population1.get_qtl_type())[j]==3 || (Population1.get_qtl_type())[j]==4 || (Population1.get_qtl_type())[j]==5)
                                                {
                                                    if(QTLGenotypes[j] == (Population1.get_qtl_allele())[j])
                                                    {
                                                        relativeviability = relativeviability * (1-((Population1.get_qtl_add_fit())[j]));
                                                    }
                                                    if(QTLGenotypes[j] > 2)
                                                    {
                                                        relativeviability = relativeviability*(1-((Population1.get_qtl_dom_fit())[j]*(Population1.get_qtl_add_fit())[j]));
                                                    }
                                                }
                                            }
                                            /* now take a draw from a uniform and if less than relativeviability than survives if greater than dead */
                                            std::uniform_real_distribution<double> distribution5(0,1);
                                            double draw = distribution5(gen);
                                            if(draw > relativeviability)                                    /* Animal Died due to a low fitness */
                                            {
                                                /* Variables that need calculated */
                                                int homozygouscount_lethal=0; int homozygouscount_sublethal=0; double lethalequivalent = 0.0;
                                                int heterzygouscount_lethal=0; int heterzygouscount_sublethal=0;
                                                double genomic_f = 0.0; double pedigree_f;  double Homoz = 0.0; double roh_f = 0.0;
                                                vector <double> GenotypicValuevec(SimParameters.getnumbertraits(),0.0);  /* Vector of Genotypic value */
                                                vector <double> BreedingValuevec(SimParameters.getnumbertraits(),0.0);   /* Vector of Breeding Value */
                                                vector <double> DominanceDeviationvec(SimParameters.getnumbertraits(),0.0);/* Vector of Dominance Deviation */
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
                                                    if((Population1.get_qtl_type())[j]==3 || (Population1.get_qtl_type())[j]==4 || (Population1.get_qtl_type())[j]==5)
                                                    {
                                                        if(QTLGenotypes[j] == (Population1.get_qtl_allele())[j])
                                                        {
                                                            if((Population1.get_qtl_type())[j] == 4){homozygouscount_lethal += 1;}
                                                            if((Population1.get_qtl_type())[j] == 3 || (Population1.get_qtl_type())[j] == 5)
                                                            {
                                                                homozygouscount_sublethal += 1;
                                                            }
                                                            lethalequivalent += (Population1.get_qtl_add_fit())[j];
                                                        }
                                                        if(QTLGenotypes[j] > 2)
                                                        {
                                                            if((Population1.get_qtl_type())[j] == 4){heterzygouscount_lethal += 1;}
                                                            if((Population1.get_qtl_type())[j] == 3 || (Population1.get_qtl_type())[j] == 5)
                                                            {
                                                                heterzygouscount_sublethal += 1;
                                                            }
                                                            lethalequivalent += (Population1.get_qtl_add_fit())[j];
                                                        }
                                                    }
                                                }
                                                /* Quantititative Summary Statistics */
                                                for (int j = 0; j < (Population1.get_qtl_type()).size(); j++)
                                                {
                                                    if((Population1.get_qtl_type())[j] == 2 || (Population1.get_qtl_type())[j] == 3)    /* Quantitative QTL */
                                                    {
                                                        int tempgeno;
                                                        if(QTLGenotypes[j] == 0 || QTLGenotypes[j] == 2){tempgeno = QTLGenotypes[j];}
                                                        if(QTLGenotypes[j] == 3 || QTLGenotypes[j] == 4){tempgeno = 1;}
                                                        for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                                                        {
                                                            /* Breeding value is only a function of additive effects */
                                                            BreedingValuevec[k] += tempgeno * double(Population1.get_qtl_add_quan(j,k));
                                                            /* Not a heterozygote; function of additive */
                                                            if(tempgeno != 1){GenotypicValuevec[k] += tempgeno * double(Population1.get_qtl_add_quan(j,k));}
                                                            if(tempgeno == 1)
                                                            {
                                                                /* Heterozygote so need to include add and dom */
                                                                GenotypicValuevec[k] += (tempgeno * Population1.get_qtl_add_quan(j,k)) + Population1.get_qtl_dom_quan(j,k);
                                                                DominanceDeviationvec[k] += Population1.get_qtl_dom_quan(j,k);
                                                            }
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
                                                pedigree_f = lethal_pedigree_inbreeding(OUTPUTFILES,tempsireid,tempdamid);
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
                                                    if((Population1.get_qtl_type())[j] == 3 || (Population1.get_qtl_type())[j] == 4 || (Population1.get_qtl_type())[j] == 5)
                                                    {
                                                        strStreamQf << QTLGenotypes[j];
                                                    }
                                                }
                                                std::ofstream output1(OUTPUTFILES.getloc_lowfitnesspath().c_str(), std::ios_base::app | std::ios_base::out);
                                                output1 << tempsireid << " " << tempdamid << " " << Gen << " " << pedigree_f << " " << genomic_f << " ";
                                                output1 << roh_f << " " << Homoz << " " << homozygouscount_lethal << " " << heterzygouscount_lethal << " ";
                                                output1 << homozygouscount_sublethal << " " << heterzygouscount_sublethal << " ";
                                                output1 << lethalequivalent << " " << relativeviability << " " << strStreamQf.str();
                                                if(SimParameters.getnumbertraits() == 1)
                                                {
                                                    output1 << " " << GenotypicValuevec[0] - BaseGenGV_vec[0];
                                                    output1 << " " << BreedingValuevec[0] - BaseGenBV_vec[0];
                                                    output1 << " " << DominanceDeviationvec[0] - BaseGenDD_vec[0];
                                                }
                                                if(SimParameters.getnumbertraits() == 2)
                                                {
                                                    output1 << " " << GenotypicValuevec[0] - BaseGenGV_vec[0];
                                                    output1 << " " << BreedingValuevec[0] - BaseGenBV_vec[0];
                                                    output1 << " " << DominanceDeviationvec[0] - BaseGenDD_vec[0];
                                                    output1 << " " << GenotypicValuevec[1] - BaseGenGV_vec[1];
                                                    output1 << " " << BreedingValuevec[1] - BaseGenBV_vec[1];
                                                    output1 << " " << DominanceDeviationvec[1] - BaseGenDD_vec[1];
                                                }
                                                output1 << endl;
                                                LethalFounder++;
                                            }
                                            if(draw < relativeviability)                                                    /* Animal Survived */
                                            {
                                                //////////////////////////////////////////////////////////////////////////
                                                // Step 7: Create founder file that has everything set for Animal Class //
                                                //////////////////////////////////////////////////////////////////////////
                                                /* Declare Variables */
                                                vector < double > Phenotypevec(SimParameters.getnumbertraits(),0.0);          /* Vector of Phenotypes */
                                                vector < double > GenotypicValuevec(SimParameters.getnumbertraits(),0.0);     /* Vector of Genotypic value */
                                                vector < double > BreedingValuevec(SimParameters.getnumbertraits(),0.0);      /* Vector of Breeding Value */
                                                vector < double > DominanceDeviationvec(SimParameters.getnumbertraits(),0.0); /* Vector of Dominance Deviation */
                                                vector < double > Residualvec(SimParameters.getnumbertraits(),0.0);           /* Vector of Residual Value */
                                                double Homoz = 0.0;                                     /* Stores homozygosity based on marker information */
                                                int Sex;                                                /* Sex of the animal 0 is male 1 is female */
                                                if(SimParameters.getnumbertraits() == 1)
                                                {
                                                    double residvar = sqrt((SimParameters.get_Var_Residual())[0]); /* random residual need standard deviation */
                                                    std::normal_distribution<double> distribution6(0.0,residvar);
                                                    Residualvec[0] = distribution6(gen);
                                                }
                                                if(SimParameters.getnumbertraits() == 2)                                 /* residual from a multivariate normal */
                                                {
                                                    VectorXd standardnormals(2);
                                                    std::normal_distribution<double> distributionstandnormal(0.0,1);
                                                    standardnormals(0) = distributionstandnormal(gen);
                                                    standardnormals(1) = distributionstandnormal(gen);
                                                    standardnormals = cholmultrait * standardnormals;
                                                    Residualvec[0] = standardnormals(0);
                                                    Residualvec[1] = standardnormals(1);
                                                    standardnormals.resize(0);
                                                }
                                                /* Determine Sex of the animal based on draw from uniform distribution; if sex < 0.5 sex is 0 if sex >= 0.5 */
                                                std::uniform_real_distribution<double> distribution5(0,1);
                                                double sex = distribution5(gen);
                                                if(sex < 0.5){Sex = 0;}         /* Male */
                                                if(sex >= 0.5){Sex = 1;}        /* Female */
                                                /* Add to genotype count across generations */
                                                for(int i = 0; i < QTL_IndCounter; i++)
                                                {
                                                    int tempgeno;
                                                    if(QTLGenotypes[i] == 0 || QTLGenotypes[i] == 2){tempgeno = QTLGenotypes[i];}
                                                    if(QTLGenotypes[i] == 3 || QTLGenotypes[i] == 4){tempgeno = 1;}
                                                    if((Population1.get_QTLFreq_AcrossGen())[i] == -5.0){Population1.update_QTLFreq_AcrossGen(i,5.0);}
                                                    Population1.update_QTLFreq_AcrossGen(i,tempgeno);
                                                }
                                                Population1.UpdateQTLFreq_Number(1);
                                                /* Quantititative Summary Statistics */
                                                for (int j = 0; j < (Population1.get_qtl_type()).size(); j++)
                                                {
                                                    if((Population1.get_qtl_type())[j] == 2 || (Population1.get_qtl_type())[j] == 3)    /* Quantitative QTL */
                                                    {
                                                        int tempgeno;
                                                        if(QTLGenotypes[j] == 0 || QTLGenotypes[j] == 2){tempgeno = QTLGenotypes[j];}
                                                        if(QTLGenotypes[j] == 3 || QTLGenotypes[j] == 4){tempgeno = 1;}
                                                        for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                                                        {
                                                            /* Breeding value is only a function of additive effects */
                                                            BreedingValuevec[k] += tempgeno * double(Population1.get_qtl_add_quan(j,k));
                                                            /* Not a heterozygote; function of additive */
                                                            if(tempgeno != 1){GenotypicValuevec[k] += tempgeno * double(Population1.get_qtl_add_quan(j,k));}
                                                            if(tempgeno == 1)
                                                            {
                                                                /* Heterozygote so need to include add and dom */
                                                                GenotypicValuevec[k] += (tempgeno * Population1.get_qtl_add_quan(j,k)) + Population1.get_qtl_dom_quan(j,k);
                                                                DominanceDeviationvec[k] += Population1.get_qtl_dom_quan(j,k);
                                                            }
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
                                                    if((Population1.get_qtl_type())[j] == 3 || (Population1.get_qtl_type())[j] == 4 || (Population1.get_qtl_type())[j] == 5)
                                                    {
                                                        if(QTLGenotypes[j] == (Population1.get_qtl_allele())[j])
                                                        {
                                                            if((Population1.get_qtl_type())[j] == 4){homozygouscount_lethal += 1;}
                                                            if((Population1.get_qtl_type())[j] == 3 || (Population1.get_qtl_type())[j] == 5)
                                                            {
                                                                homozygouscount_sublethal += 1;
                                                            }
                                                            lethalequivalent += (Population1.get_qtl_add_fit())[j];
                                                        }
                                                        if(QTLGenotypes[j] > 2)
                                                        {
                                                            if((Population1.get_qtl_type())[j] == 4){heterzygouscount_lethal += 1;}
                                                            if((Population1.get_qtl_type())[j] == 3 || (Population1.get_qtl_type())[j] == 5)
                                                            {
                                                                heterzygouscount_sublethal += 1;
                                                            }
                                                            lethalequivalent += (Population1.get_qtl_add_fit())[j];
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
                                                if(SimParameters.getnumbertraits() == 1)
                                                {
                                                    GenotypicValuevec[0] -= BaseGenGV_vec[0];
                                                    BreedingValuevec[0] -= BaseGenBV_vec[0];
                                                    DominanceDeviationvec[0] -= BaseGenDD_vec[0];
                                                }
                                                if(SimParameters.getnumbertraits() == 2)
                                                {
                                                    GenotypicValuevec[0] -= BaseGenGV_vec[0];
                                                    BreedingValuevec[0] -= BaseGenBV_vec[0];
                                                    DominanceDeviationvec[0] -= BaseGenDD_vec[0];
                                                    GenotypicValuevec[1] -= BaseGenGV_vec[1];
                                                    BreedingValuevec[1] -= BaseGenBV_vec[1];
                                                    DominanceDeviationvec[1] -= BaseGenDD_vec[1];
                                                }
                                                /* put marker, qtl and fitness into string to store */
                                                stringstream strStreamM (stringstream::in | stringstream::out);
                                                for (int j=0; j < m_counter; j++){strStreamM << MarkerGenotypes[j];}
                                                string MA = strStreamM.str();
                                                stringstream strStreamQt (stringstream::in | stringstream::out);
                                                for (int j=0; j < qtl_counter; j++){strStreamQt << QTLGenotypes[j];}
                                                string QT = strStreamQt.str();
                                                if(SimParameters.getnumbertraits() == 1){Phenotypevec[0] = GenotypicValuevec[0] + Residualvec[0];}
                                                if(SimParameters.getnumbertraits() == 2)
                                                {
                                                    Phenotypevec[0] = GenotypicValuevec[0] + Residualvec[0];
                                                    Phenotypevec[1] = GenotypicValuevec[1] + Residualvec[1];
                                                }
                                                //cout << GenotypicValuevec[0] << " " << GenotypicValuevec[1] << endl;
                                                //cout << BreedingValuevec[0] << " " << BreedingValuevec[1] << endl;
                                                //cout << DominanceDeviationvec[0] << " " << DominanceDeviationvec[1] << endl;
                                                //cout << Residualvec[0] << " " << Residualvec[1] << endl;
                                                double rndselection = distribution5(gen);
                                                double rndculling = distribution5(gen);
                                                double rndphen1 = distribution5(gen);
                                                double rndphen2 = distribution5(gen);
                                                double rndgeno = distribution5(gen);
                                                /* Find age of tempsire and tempdamid */
                                                int search = 0; int sireage, damage;
                                                while(1)
                                                {
                                                    if(tempsireid == population[search].getID()){sireage = (population[search].getAge()-1); break;}
                                                    if(tempsireid != population[search].getID()){search++;}
                                                    if(search >= population.size()){cout << "Shouldn't be here(Age Sire)!" << endl; exit (EXIT_FAILURE);}
                                                }
                                                search = 0;
                                                while(1)
                                                {
                                                    if(tempdamid == population[search].getID()){damage = (population[search].getAge()-1); break;}
                                                    if(tempdamid != population[search].getID()){search++;}
                                                    if(search >= population.size()){cout << "Shouldn't be here(Age Dam)!" << endl; exit (EXIT_FAILURE);}
                                                }
                                                Animal animal(StartID,tempsireid,sireage,tempdamid,damage,Sex,Gen,1,0,0,0,rndselection,rndculling,rndphen1,rndphen2,rndgeno,0.0,diagInb,0.0,0.0,0.0,homozygouscount_lethal, heterzygouscount_lethal, homozygouscount_sublethal,heterzygouscount_sublethal,lethalequivalent,Homoz,relativeviability,MA,QT,"","","",0.0,"",Phenotypevec,vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0),GenotypicValuevec,BreedingValuevec,DominanceDeviationvec,Residualvec,0.0,0.0,vector<double>(SimParameters.getnumbertraits()),vector<double>(SimParameters.getnumbertraits()),vector<std::string>(SimParameters.getnumbertraits()),"selcand");
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
                UpdateTBV_TDD_Statistical(Population1,population,SimParameters);                /* Update TBV and TDD for a given generation */
                logfile << "       - Number of Progeny that Died due to fitness: " << LethalFounder << endl;
                logfile << "       - Size of population after progeny generated: " << population.size() << endl;
                Population1.update_numdeadfitness(Gen,LethalFounder);
                time_t end_offspring = time(0);
                logfile << "   Finished generating offspring from parental gametes and mating design: (Time: ";
                logfile << difftime(end_offspring,start_offspring) << " seconds)." << endl << endl;
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Update Haplotype Library based on new animals and compute diagonals of relationship matrix                //
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                time_t start_block4 = time(0);
                vector < vector < string >> lib_haplotypes(haplib.size(),vector<string>(0)); /* Stores unique haplotypes across all libraries */
                string temphapstring;
                for(int i = 0; i < haplib.size(); i++)
                {
                    temphapstring = haplib[i].getHaplo();
                    string quit = "NO";
                    while(quit != "YES")
                    {
                        size_t pos = temphapstring.find("_",0);
                        /* hasn't reached last one yet */
                        if(pos > 0){lib_haplotypes[i].push_back(temphapstring.substr(0,pos)); temphapstring.erase(0, pos + 1);}
                        if(pos == std::string::npos){quit = "YES";}     /* has reached last one so now save last one and kill while loop */
                    }
                    //cout << lib_haplotypes[i].size() << endl;
                    //for(int j = 0; j < lib_haplotypes[i].size(); j++){cout << lib_haplotypes[i][j] << endl;}
                }
                int counterhapind = 0;
                for(int i = 0; i < population.size(); i++)
                {
                    string homo1, homo2, temp;
                    if(population[i].getAge() == 1)
                    {
                        vector < int > temppat(lib_haplotypes.size());
                        vector < int > tempmat(lib_haplotypes.size());
                        for(int j = 0; j < lib_haplotypes.size(); j++)
                        {
                            /* Grab specific haplotype */
                            temp = (population[i].getMarker()).substr(haplib[j].getStart(),SimParameters.gethaplo_size());
                            homo1 = homo2 = temp;                                                  /* Paternal (1) and Maternal (2)  */
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
                                if(g == 0){temp = homo1;
                                } else {temp = homo2;}
                                int num = 0;
                                while(1)
                                {
                                    if(num >= (lib_haplotypes[j].size()-1)){
                                        lib_haplotypes[j].push_back(temp);  /* If number not match = size of hapLibary then add */
                                        if(g == 0){temppat[j] = num; break;
                                        } else {tempmat[j] = num; break;}
                                    } else if(temp.compare(lib_haplotypes[j][num]) != 0){
                                        num++;
                                    } else if(temp.compare(lib_haplotypes[j][num]) == 0 && g == 0){
                                        temppat[j] = num; break;
                                    } else if(temp.compare(lib_haplotypes[j][num]) == 0 && g == 1){
                                        tempmat[j] = num; break;
                                    } else {
                                        cout << "Error when calculating haplotyhpes!" << endl; exit (EXIT_FAILURE);
                                    }
                                }
                            }
                            /* Finished looping across both haplotypes. Either is new or has already been their so can create portion of each haplotype matrix */
                            float sum1 = 0.0;                                 /* Haplotype 1 Matrix */
                            int match[homo1.size()];                          /* Haplotype 2 matrix that has 1 to match and 0 if not match */
                            double sumh3 = 0.0;                              /* Haplotype 3 Matrix */
                            double hvalue1 = 0.0;                             /* Haplotype 1 Matrix */
                            double hvalue2 = 0.0;                            /* Haplotype 2 Matrix */
                            double hvalue3 = 0.0;
                            for(int g = 0; g < homo1.size(); g++)
                            {
                                sum1 += abs(homo1[g] - homo2[g]);               /* H1 */
                                match[g] = 1 - abs(homo1[g] - homo2[g]);        /* H2 */
                                sumh3 += match[g];                              /* H3 */
                                if(g == homo1.size() - 1)
                                {
                                    hvalue1 = (1 - (sum1/homo1.size())) + 1;
                                    if(sumh3 == homo1.size()){hvalue3 = 2.0;}
                                    if(sumh3 != homo1.size()){hvalue3 = 1.0;}
                                }
                            }
                            double sumGlobal2 = 0;                          /* Haplotype 2 Matrix */
                            double sumh2=0;                                 /* Haplotype 2 Matrix */
                            for(int g = 0; g < homo1.size(); g++)
                            {
                                if(match[g] < 1)
                                {
                                    sumGlobal2 = sumGlobal2 + sumh2*sumh2; sumh2 = 0;
                                } else {
                                    sumh2 = sumh2 + 1;
                                }
                            }
                            /* don't need to divide by 2 because is 1 + 1 + sqrt(Sum/length) + sqrt(Sum/length) */
                            hvalue2 = 1 + sqrt((sumGlobal2 + (sumh2 * sumh2)) / (homo1.size() * homo1.size()));
                            population[i].AccumulateH1(hvalue1);                        /* Add to diagonal of population */
                            population[i].AccumulateH2(hvalue2);                        /* Add to diagonal of population */
                            population[i].AccumulateH3(hvalue3);                        /* Add to diagonal of population */
                        }
                        std::ostringstream pat; pat << temppat[0];
                        for(int j = 1; j < lib_haplotypes.size(); j++){pat << "_" << temppat[j];}
                        std::ostringstream mat; mat << tempmat[0];
                        for(int j = 1; j < lib_haplotypes.size(); j++){mat << "_" << tempmat[j];}
                        /* Update everything now */
                        population[i].StandardizeH1(haplib.size()); population[i].StandardizeH2(haplib.size()); population[i].StandardizeH3(haplib.size());
                        population[i].Update_PatHap(pat.str()); population[i].Update_MatHap(mat.str());
                        //cout << "P'" << population[i].getPatHapl() << "'" << endl;
                        //cout << "M'" << population[i].getMatHapl() << "'" << endl;
                        //cout << population[i].getHap1_F() << " " << population[i].getHap2_F() << " ";
                        //cout << population[i].getHap3_F() << endl;
                        counterhapind++;
                    }
                }
                /* Once reached last individual put all unique haplotypes into string with a "_" delimter to split them apart later */
                for(int i = 0; i < lib_haplotypes.size(); i++)
                {
                    temphapstring = ""; temphapstring = lib_haplotypes[i][0];
                    for(int j = 1; j < lib_haplotypes[i].size(); j++){temphapstring = temphapstring + "_" + lib_haplotypes[i][j];}
                    haplib[i].UpdateHaplotypes(temphapstring);
                }
                time_t end_block4 = time(0);
                logfile << "   Created haplotype library and assigned haplotypes IDs to individuals (Time: ";
                logfile << difftime(end_block4,start_block4) << " seconds)." << endl << endl;
                /* Put newly created progeny in pheno and g-matrix file */
                TotalOldAnimalNumber = TotalAnimalNumber;           /* Size of old animal matrix */
                TotalAnimalNumber = OutputPedigree_GenomicEBV(OUTPUTFILES,population,TotalAnimalNumber,SimParameters);
                Update_selcand_PA(population);                      /* Update 'selcand' ebv with parent average */
                if(Indexsd_calc[Gen-1] != "NA")
                {
                    if(Indexsd_calc[Gen-1] == "YES"){GenerateBVIndex(population,Population1,SimParameters,1,logfile);}
                    if(Indexsd_calc[Gen-1] == "NO"){GenerateBVIndex(population,Population1,SimParameters,2,logfile);}
                }
                /* After Generating New Progeny update phenostatus and append ot file */
                UpdateGenoPhenoStatus(Population1,population,SimParameters,OUTPUTFILES,Gen,"outputselectioncand",SelectionVector);
                GenerateGenerationInterval(population,Population1,SimParameters,Gen,logfile);
                /************************************************************************************************************************************/
                /* Determine whether to generate interim EBV's (offspring PA and parents now have phenotypic information if occured after selection */
                /************************************************************************************************************************************/
                if(SimParameters.getInterim_EBV() == "before_culling" && (SelectionVector[Gen]=="ebv" || SelectionVector[Gen]=="index_ebv"))
                {
                    /*****************************************************************************************************************************/
                    /* If doing the ebv based phenotype and/or genotype strategy need to calculate ebvs prior to determining whether an animals  */
                    /* is phenotyped or not. Therefore you can make a decision on whether to genotype/phenotype based on parent average (parents */
                    /* can now be genotyped) or updated ebv from trait one.                                                                      */
                    /*****************************************************************************************************************************/
                    time_t start_block = time(0);
                    logfile << "   Generate Estimated Breeding Values Prior to Culling Parents based on ";
                    logfile << SimParameters.getEBV_Calc() << " method:" << endl;
                    vector <double> estimatedsolutions; vector < double > trueaccuracy;
                    Generate_BLUP_EBV(SimParameters,population,estimatedsolutions,trueaccuracy,logfile,trainanimals,TotalAnimalNumber,TotalOldAnimalNumber,Gen,M,scale,haplib,OUTPUTFILES);
                    time_t end_block = time(0);
                    logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
                    if(Indexsd_calc[Gen-1] != "NA")
                    {
                        if(Indexsd_calc[Gen-1] == "YES"){GenerateBVIndex(population,Population1,SimParameters,1,logfile);}
                        if(Indexsd_calc[Gen-1] == "NO"){GenerateBVIndex(population,Population1,SimParameters,2,logfile);}
                    }
                }
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                    Start of Culling Functions                                   //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                time_t start5 = time(0);
                string tempcullingscen = SelectionVector[Gen-1];
                if(SelectionVector[Gen-1] == "random"){logfile<<"   Begin "<<SelectionVector[Gen-1]<<" Culling: "<<endl;}
                if(SelectionVector[Gen-1] != "random"){logfile << "   Begin " << SimParameters.getCulling() << " Culling: " << endl;}
                if(SimParameters.getSireRepl()==1.0 && SimParameters.getDamRepl()==1.0)
                {
                    discretegenerations(population,SimParameters,tempcullingscen,Gen,OUTPUTFILES,logfile);
                }
                if(SimParameters.getSireRepl()<1.0 || SimParameters.getDamRepl()<1.0)
                {
                    overlappinggenerations(population,SimParameters,tempcullingscen,Gen,OUTPUTFILES,logfile);
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
                time_t end5 = time(0);
                if(SelectionVector[Gen-1]=="random")
                {
                    logfile<<"   Finished "<<SelectionVector[Gen-1]<<" culling parents (Time: "<<difftime(end5,start5)<<" sec).\n\n";
                }
                if(SelectionVector[Gen-1] != "random")
                {
                    logfile<<"   Finished "<<SimParameters.getCulling()<<" culling parents (Time: "<<difftime(end5,start5)<<" sec).\n\n";
                }
                /************************************************************************************************************************************/
                /* Determine whether to generate interim EBV's (offspring PA and parents now have phenotypic information if occured after selection */
                /************************************************************************************************************************************/
                if(SimParameters.getInterim_EBV() == "after_culling" && (SelectionVector[Gen]=="ebv" || SelectionVector[Gen]=="index_ebv"))
                {
                    //for(int i = 0; i < 500; i++)
                    //{
                    //    if(population[i].getAnimalStage() == "selcand")
                    //    {
                    //        cout<<population[i].getID()<<" "<< population[i].getDam()<<" "<<population[i].getSire()<<" ";
                    //        cout<<(population[i].get_PhenStatus())[0]<< " " << (population[i].get_EBVvect())[0]<<endl;
                    //    }
                    //}
                    //cout << "________" << endl;
                    /*****************************************************************************************************************************/
                    /* If doing the ebv based phenotype and/or genotype strategy need to calculate ebvs prior to determining whether an animals  */
                    /* is phenotyped or not. Therefore you can make a decision on whether to genotype/phenotype based on parent average (parents */
                    /* can now be genotyped) or updated ebv from trait one.                                                                      */
                    /*****************************************************************************************************************************/
                    time_t start_block = time(0);
                    logfile << "   Generate Estimated Breeding Values After Culling Parents based on ";
                    logfile << SimParameters.getEBV_Calc() << " method:" << endl;
                    vector <double> estimatedsolutions; vector < double > trueaccuracy;
                    Generate_BLUP_EBV(SimParameters,population,estimatedsolutions,trueaccuracy,logfile,trainanimals,TotalAnimalNumber,TotalOldAnimalNumber,Gen,M,scale,haplib,OUTPUTFILES);
                    time_t end_block = time(0);
                    logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
                    if(Indexsd_calc[Gen-1] != "NA")
                    {
                        if(Indexsd_calc[Gen-1] == "YES"){GenerateBVIndex(population,Population1,SimParameters,1,logfile);}
                        if(Indexsd_calc[Gen-1] == "NO"){GenerateBVIndex(population,Population1,SimParameters,2,logfile);}
                    }
                    //for(int i = 0; i < 500; i++)
                    //{
                    //    if(population[i].getAnimalStage() == "selcand")
                    //    {
                    //        cout<<population[i].getID()<<" "<< population[i].getDam()<<" "<<population[i].getSire()<<" ";
                    //        cout<<(population[i].get_PhenStatus())[0]<< " " << (population[i].get_EBVvect())[0]<<endl;
                    //    }
                    //}
                }
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////                                    Housekeeping Functions                                       //////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ExpectedHeterozygosity(population,Population1,Gen,logfile);         /* Calculate expected heterozygosity in progeny */
                UpdateFrequency_GenVar(SimParameters,population,Population1,Gen,population_QTL,OUTPUTFILES,logfile);    /* Update Frequency and Va and Vd */
                if(SimParameters.getLDDecay() == "yes"){LD_Option(SimParameters,population,population_QTL,OUTPUTFILES,"no",logfile);}
                if(SimParameters.getstartgen() != -5){Haplofinder_Option(SimParameters,population,trainregions,unfav_direc,Gen,OUTPUTFILES,logfile);}
                if(SimParameters.getmblengthroh() != -5){ROH_Option(SimParameters,population,Gen,OUTPUTFILES,logfile);}
                if(SimParameters.getOutputTrainReference() == "yes")
                {
                    getamax(SimParameters,population,OUTPUTFILES);
                    logfile << "   Generated Amax between recent generation and previous generations." << endl << endl;
                }
                if(SimParameters.getOutputWindowVariance() == "yes")
                {
                    string foundergen = "no";
                    WindowVariance(SimParameters,population,population_QTL,foundergen,OUTPUTFILES);
                    logfile << "   Generating Additive and Dominance Window Variance."<< endl << endl;
                }
                logfile << endl;
                time_t intend_time = time(0);
                cout <<"   - Finished Generation "<<Gen<<" (Took: "<<difftime(intend_time,intbegin_time)<<" seconds)"<<endl;
            }
            cout << "Finished Simulating Generations" << endl;
            logfile << "------ Finished Simulating Generations --------" << endl;
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////                   Finished Looping through generations based on forward-in-time simulation techniques                      ////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /* If on last generation output all population but before update Inbreeding values and Calculate EBV's. */
            using Eigen::MatrixXd; using Eigen::VectorXd;
            clock_t start; clock_t end;
            vector < double > Phenotype; vector <double> estimatedsolutions; vector <double> trueaccuracy; vector <int> trainanimals;
            /* Prior to estimating breeding values update phenostatus */
            UpdateGenoPhenoStatus(Population1,population,SimParameters,OUTPUTFILES,SimParameters.getGener()-1,"preebvcalc",SelectionVector);
            /* Initialize to zero */
            if((Population1.get_qtlscalefactadd())[0] != 0 && SimParameters.getEBV_Calc() != "bayes" && SimParameters.getEBV_Calc()!="SKIP")
            {
                time_t start_block = time(0);
                cout << "Estimate Final Breeding Values For Last Generation" << endl;
                logfile<<"   Generate EBV based on "<< SimParameters.getEBV_Calc()<<" information for last generation:"<<endl;
                int Gen = SimParameters.getGener();
                Generate_BLUP_EBV(SimParameters,population,estimatedsolutions,trueaccuracy,logfile,trainanimals,TotalAnimalNumber,TotalOldAnimalNumber,Gen,M,scale,haplib,OUTPUTFILES);
                time_t end_block = time(0);
                logfile << "   Finished Estimating Breeding Values (Time: " << difftime(end_block,start_block) << " seconds)."<< endl << endl;
            }
            if((Population1.get_qtlscalefactadd())[0]==0 || ((Population1.get_qtlscalefactadd())[0]!=0 && SimParameters.getEBV_Calc()=="bayes") || SimParameters.getEBV_Calc()=="SKIP")
            {
                Inbreeding_Pedigree(population,OUTPUTFILES);
            }
            if(SimParameters.getOutputTrainReference() == "yes")
            {
                if((Population1.get_qtlscalefactadd())[0] != 0 && SimParameters.getEBV_Calc()!="SKIP")
                {
                    int tempgen = SimParameters.getGener()+1; trainrefcor(SimParameters,population,OUTPUTFILES,tempgen);
                }
            }
            cout << "   Generating Master Dataframe and Summmary Statistics." << endl;
            /* If have ROH genome summary as an option; do proportion of genome in ROH for reach individual */
            if(SimParameters.getmblengthroh() != -5){Proportion_ROH(SimParameters,population,OUTPUTFILES,logfile);}
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
                if(SimParameters.getOutputGeno() == "yes" && SimParameters.getGener() >= SimParameters.getoutputgeneration())
                {
                    outputstringgeno << population[i].getID() <<" "<< population[i].getMarker() <<" "<< population[i].getQTL() << endl;
                    outputnumpart1++;
                }
                if(outputnumpart1 % 1000 == 0)
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
            }
            /* output master df file */
            std::ofstream output3(OUTPUTFILES.getloc_Master_DF().c_str(), std::ios_base::app | std::ios_base::out);
            output3 << outputstring.str(); outputstring.str(""); outputstring.clear();
            /* output master geno file */
            //std::ofstream output4(OUTPUTFILES.getloc_Master_Genotype().c_str(), std::ios_base::app | std::ios_base::out);
            //output4 << outputstringgeno.str();
            zippedgeno.open(OUTPUTFILES.getloc_Master_Genotype_zip().c_str(),std::ios_base::app);
            if(!zippedgeno.is_open()){cout << endl << "Error can't open zipped genotyped file." << endl; exit (EXIT_FAILURE);}
            zippedgeno << outputstringgeno.str();
            zippedgeno.close(); outputstringgeno.str(""); outputstringgeno.clear();
            population.clear();
            /*********************************************************/
            /* Generate Summary Statistics and Clean up Output Files */
            /*********************************************************/
            vector < int > ID_Gen;
            GenerateMaster_DataFrame(SimParameters,OUTPUTFILES,Population1,estimatedsolutions,trueaccuracy,ID_Gen);
            generatesummaryqtl(SimParameters,OUTPUTFILES,ID_Gen,Population1);
            generatessummarydf(SimParameters,OUTPUTFILES,Population1);
            generateqtlfile(SimParameters,OUTPUTFILES);
            CleanUpSimulation(OUTPUTFILES);         /* Clean Up Files Not Needed In Folder*/
            delete [] M; delete [] founderfreq; estimatedsolutions.clear(); trueaccuracy.clear();     /* Clean up vectors */
            logfile << "   Created Master File." << endl << endl;
            time_t repend_time = time(0);
            if(SimParameters.getReplicates() > 1)
            {
                cout.setf(ios::fixed);
                cout << setprecision(2) << endl << "Replicate " << reps + 1 << " has completed normally (Took: ";
                cout << difftime(repend_time,repbegin_time) / 60 << " minutes)" << endl << endl;
                cout.unsetf(ios::fixed);
            }
        }
        if(SimParameters.getEBV_Calc() == "ssgblup"){getGenotypeCountGeneration(OUTPUTFILES,logfile);}
        /* If you have multiple replicates create a new directory within this folder to store them and just attach seed afterwards */
        if(SimParameters.getReplicates() > 1){SaveReplicates (reps,SimParameters,OUTPUTFILES,logfileloc,path);}
    }
    free(cwd); mkl_free_buffers(); mkl_thread_free_buffers();
    time_t fullend_time = time(0);
    cout.setf(ios::fixed);
    cout << setprecision(2) << "Simulation has completed normally (Took: " << difftime(fullend_time,fullbegin_time) / 60 << " minutes)" << endl << endl;
    cout.unsetf(ios::fixed);
}
