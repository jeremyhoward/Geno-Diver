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

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Class Functions       ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
parameters::parameters(){}
parameters::~parameters(){}
void parameters::UpdateSeed(int temp){seednumber = temp;}
void parameters::UpdateStartSim(std::string temp){StartSim = temp;}
void parameters::UpdateOutputFold(std::string temp){outputfolder = temp;}
void parameters::UpdateThread(int temp){nthread = temp;}
void parameters::UpdateReplicates(int temp){replicates = temp;}
void parameters::UpdateChr(int temp){Chr = temp;}
void parameters::UpdateThresholdMAFMark(double temp){ThresholdMAFMark = temp;}
void parameters::UpdateThresholdMAFQTL(double temp){ThresholdMAFQTL = temp;}
void parameters::UpdateUpThrMAFLFit(double temp){LowerThresholdMAFFitnesslethal = temp;}
void parameters::UpdateRangeMAFLFit(double temp){maxdifferencelethal = temp;}
void parameters::UpdateUpThrMAFSFit(double temp){LowerThresholdMAFFitnesssublethal = temp;}
void parameters::Updatehaplo_size(double temp){haplo_size = temp;}
void parameters::UpdateRecombDis(std::string temp){Recomb_Distribution = temp;}
void parameters::Updateu(float temp){u = temp;}
void parameters::UpdatePropQTL(double temp){PropQTL = temp;}
void parameters::UpdateGamma_Shape(double temp){Gamma_Shape = temp;}
void parameters::UpdateGamma_Scale(double temp){Gamma_Scale = temp;}
void parameters::UpdateNormal_meanRelDom(double temp){Normal_meanRelDom = temp;}
void parameters::UpdateNormal_varRelDom(double temp){Normal_varRelDom= temp;}
void parameters::UpdateQuantParam(std::string temp){QuantParam = temp;}
void parameters::UpdateGamma_Shape_Lethal(double temp){Gamma_Shape_Lethal = temp;}
void parameters::UpdateGamma_Scale_Lethal(double temp){Gamma_Scale_Lethal = temp;}
void parameters::UpdateNormal_meanRelDom_Lethal(double temp){Normal_meanRelDom_Lethal = temp;}
void parameters::UpdateNormal_varRelDom_Lethal(double temp){Normal_varRelDom_Lethal = temp;}
void parameters::UpdateGamma_Shape_SubLethal(double temp){Gamma_Shape_SubLethal = temp;}
void parameters::UpdateGamma_Scale_SubLethal(double temp){Gamma_Scale_SubLethal = temp;}
void parameters::UpdateNormal_meanRelDom_SubLethal(double temp){Normal_meanRelDom_SubLethal = temp;}
void parameters::UpdateNormal_varRelDom_SubLethal(double temp){Normal_varRelDom_SubLethal = temp;}
void parameters::Updateproppleitropic(double temp){proppleitropic = temp;}
void parameters::Updategencorr(double temp){genetic_correlation = temp;}
void parameters::UpdateNe_spec(std::string temp){Ne_spec = temp;}
void parameters::Updatene_founder(double temp){ne_founder = temp;}
void parameters::Updatefoundermale(int temp){foundermale = temp;}
void parameters::Updatefounderfemale(int temp){founderfemale = temp;}
void parameters::Updatefounderselect(std::string temp){founderselect = temp;}
void parameters::UpdateGenfoundsel(int temp){generationsfounderselect = temp;}
void parameters::Updatefnd_haplo(int temp){fnd_haplo = temp;}
void parameters::Updatenumbertraits(int temp){numberoftraits = temp;}
void parameters::UpdateGener(int temp){GENERATIONS = temp;}
void parameters::UpdateSires(int temp){SIRES = temp;}
void parameters::UpdateSireRepl(double temp){SireReplacement = temp;}
void parameters::UpdateDams(int temp){DAMS = temp;}
void parameters::UpdateDamRepl(double temp){DamReplacement = temp;}
void parameters::UpdateOffspring(int temp){OffspringPerMating = temp;}
void parameters::UpdateSelection(std::string temp){Selection = temp;}
void parameters::UpdateSelectionDir(std::string temp){SelectionDir = temp;}
void parameters::Updatemaxmating(int temp){maxmating = temp;}
void parameters::UpdateEBV_Calc(std::string temp){EBV_Calc = temp;}
void parameters::UpdateCulling(std::string temp){Culling = temp;}
void parameters::UpdateMaxAge(int temp){MaximumAge = temp;}
void parameters::UpdateMating(std::string temp){Mating = temp;}
void parameters::UpdateMatingAlg(std::string temp){mating_algorithm = temp;}
void parameters::UpdateBetaDist_alpha(double temp){BetaDist_alpha = temp;}
void parameters::UpdateBetaDist_beta(double temp){BetaDist_beta = temp;}
void parameters::Updatemaxsireprop(double temp){maxsireproportion = temp;}
void parameters::UpdateLDDecay(std::string temp){Create_LD_Decay = temp;}
void parameters::UpdateOutputGeno(std::string temp){OutputGeno = temp;}
void parameters::Updateoutputgeneration(int temp){outputgeneration = temp;}
void parameters::Updatestartgen(int temp){startgen = temp;}
void parameters::Updatetraingen(int temp){traingen = temp;}
void parameters::Updateretrain(int temp){retrain = temp;}
void parameters::Updateocsrelat(std::string temp){ocsrelationship = temp;}
void parameters::Updateocs_optimize(std::string temp){ocs_optimize = temp;}
void parameters::Updateocs_w_merit(double temp){ocs_w_merit = temp;}
void parameters::Updateocs_w_rel(double temp){ocs_w_rel = temp;}
void parameters::UpdatenEVAgen(double temp){nEVAgen = temp;}
void parameters::UpdatenEVApop(double temp){nEVApop = temp;}
void parameters::Updatemblengthroh(int temp){MbLenthCutoff = temp;}
void parameters::Updategenmafcutoff(double temp){genomicmafcutoff = temp;}
void parameters::Updategetgenmafdir(std::string temp){genomicmafdirection = temp;}
void parameters::Updatemethod(std::string temp){method = temp;}
void parameters::Updatenumiter(int temp){numiter = temp;}
void parameters::Updateburnin(int temp){burnin = temp;}
void parameters::Updatethin(int temp){thin = temp;}
void parameters::Updatepie_f(std::string temp){pie_f = temp;}
void parameters::Updateinitpi(double temp){initpi = temp;}
void parameters::Updatereferencegenerations(int temp){referencegenerations = temp;}
void parameters::UpdateSolver(std::string temp){Solver = temp;}
void parameters::UpdateGeno_Inverse(std::string temp){Geno_Inverse = temp;}
void parameters::Updatereferencegenblup(int temp){referencegenerationsblup = temp;}
void parameters::UpdateConstructG(std::string temp){ConstructG=temp;}
void parameters::UpdateConstructGFreq(std::string temp){ConstructGFreq=temp;}
void parameters::UpdateOutputTrainReference(std::string temp){OutputTrainReference=temp;}
void parameters::Updatenumiterstat(int temp){numiterstat = temp;}
void parameters::Updateburninstat(int temp){burninstat = temp;}
void parameters::UpdateOutputWindowVariance(std::string temp){OutputWindowVariance = temp;}
void parameters::UpdateGenoGeneration(int temp){GenoGeneration = temp;}
void parameters::UpdateMalePropGenotype(double temp){MalePropGenotype = temp;}
void parameters::UpdateMaleWhoGenotype(std::string temp){MaleWhoGenotype = temp;}
void parameters::UpdateFemalePropGenotype(double temp){FemalePropGenotype = temp;}
void parameters::UpdateFemaleWhoGenotype(std::string temp){FemaleWhoGenotype = temp;}
void parameters::UpdateGenotypePortionofDistribution(std::string temp){GenotypePortionofDistribution = temp;}
void parameters::UpdateInterim_EBV(std::string temp){Interim_EBV = temp;}
void parameters::UpdateImputationFile(std::string temp){ImputationFile = temp;}


////////////////////////////////////////////////////////////////////////////////////
///////////        Read in Parameter File and Fill Parameter Class       ///////////
////////////////////////////////////////////////////////////////////////////////////
void read_generate_parameters(parameters &SimParameters, string parameterfile, string &logfilestring, string &logfilestringa)
{
    /* read parameters file */
    vector <string> parm;
    string parline;
    ifstream parfile;
    parfile.open(parameterfile);
    if(parfile.fail()){cout << "Parameter file not found. Check log file." << endl; exit (EXIT_FAILURE);}
    while (getline(parfile,parline)){parm.push_back(parline);} /* Stores in vector and each new line push back to next space */
    logfilestringa=logfilestringa + "===============================================\n";
    logfilestringa=logfilestringa + "==        Read in Parameters from file       ==\n";
    logfilestringa=logfilestringa + "===============================================\n";
    logfilestringa=logfilestringa + "Name of parameter file was: '"+parameterfile+"'\n";
    logfilestringa=logfilestringa + "Parameters Specified in Paramter File: \n";
    logfilestringa=logfilestringa + "    - Initializing Simulation:\n";
    ///////////////////////////////////////////////////////////////
    ///////////        General Starting Parameters      ///////////
    ///////////////////////////////////////////////////////////////
    int search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("START:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            SimParameters.UpdateStartSim(parm[search]); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'START:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SimParameters.getStartSim() != "sequence" && SimParameters.getStartSim() != "founder")
    {
        cout << endl << "START (" << SimParameters.getStartSim() << ") didn't equal sequence or founder! Check parameter file!" << endl;
        exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("SEED:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            SimParameters.UpdateSeed(atoi(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size()){SimParameters.UpdateSeed(time(0));}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("OUTPUTFOLDER:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            SimParameters.UpdateOutputFold(parm[search].c_str());
            logfilestring = logfilestring + "        - Simulation Files Sent to:\t\t\t\t\t\t\t\t\t'" + SimParameters.getOutputFold() + "'\n"; break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateOutputFold("GenoDiverFiles");
            logfilestring = logfilestring + "        - Simulation Files Sent to:\t\t\t\t\t\t\t\t\t'" + SimParameters.getOutputFold() + "'\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("NTHREAD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring + "        - Number of threads:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateThread(atoi(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateThread(1); logfilestring = logfilestring + "        - Number of threads:\t\t\t\t\t\t\t\t\t\t'1' (Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("NREP:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring + "        - Number of replicates:\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateReplicates(atoi(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateReplicates(1); logfilestring = logfilestring + "        - Number of replicates:\t\t\t\t\t\t\t\t\t'1'\n"; break;
        }
    }
    ////////////////////////////////////////////////////////////////
    ///////////        Genome and Marker Parameters      ///////////
    ////////////////////////////////////////////////////////////////
    logfilestring = logfilestring + "    - Genome and Marker:\n";
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("CHR:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring + "        - Number of Chromosomes:\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateChr(atoi(parm[search].c_str())); break;
        }
        search++; if(search > parm.size()){cout << endl << "Couldn't find 'CHR:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("CHR_LENGTH:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            for(int i = 0; i < solvervariables.size(); i++)
            {
                SimParameters.add_ToChrLength(atof(solvervariables[i].c_str())*1000000);
            }
            if(solvervariables.size() != SimParameters.getChr())
            {
                cout << endl << "CHR_LENGTH " << solvervariables.size() << " doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);
            }
            break;
        }
        search++;
        if(search >= parm.size()){cout << endl << "Couldn't find 'CHR_LENGTH:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    /* Check to ensure isn't 0 which can happen if leave a blank when you give one less */
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        if((SimParameters.get_ChrLength())[i] == 0){cout << endl << "Incorrect Chromosome Length of 0!" << endl; exit (EXIT_FAILURE);}
    }
    logfilestring = logfilestring + "        - Length of Chromosome (Mb): \n";
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        stringstream s1; s1 << int((SimParameters.get_ChrLength())[i] / double(1000000)); string tempvar = s1.str();
        stringstream s2; s2 << i + 1; string tempvara = s2.str();
        logfilestring = logfilestring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("NUM_MARK:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            for(int i = 0; i < solvervariables.size(); i++){SimParameters.add_ToMarker_chr(atoi(solvervariables[i].c_str()));}
            if(solvervariables.size() != SimParameters.getChr())
            {
                cout << endl << "NUM_MARK " << solvervariables.size() << " doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);
            }
            break;
        }
        search++;
        if(search >= parm.size()){cout << endl << "Couldn't find 'NUM_MARK: ' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        if((SimParameters.get_Marker_chr())[i] == 0){cout << endl << "Incorrect Number of Marker value of 0!" << endl; exit (EXIT_FAILURE);}
    }
    logfilestring = logfilestring + "        - Number of Markers: \n";
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        stringstream s1; s1 << (SimParameters.get_Marker_chr())[i]; string tempvar = s1.str();
        stringstream s2; s2 << i + 1; string tempvara = s2.str();
        logfilestring = logfilestring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MARKER_MAF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring + "        - Minor Allele Frequency Threshold for Markers:\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateThresholdMAFMark(atof(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateThresholdMAFMark(0.05);
            logfilestring = logfilestring + "        - Minor Allele Frequency Threshold for Markers:\t\t\t\t\t\t'0.05' (Default)\n"; break;
        }
    }
    if(SimParameters.getThresholdMAFMark() >= 0.5)
    {
        cout << endl << "Marker MAF threshold can't be greater than 0.5! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("QTL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            for(int i = 0; i < solvervariables.size(); i++){SimParameters.add_ToQTL_chr(atoi(solvervariables[i].c_str()));}
            break;
        }
        search++;
        if(search >= parm.size())
        {
            for(int i = 0; i < SimParameters.getChr(); i++){SimParameters.add_ToQTL_chr(0);}
            break;
        }
    }
    /* Double check to see if given QTL mutations must be same size as number of chromosomes */
    int numbmutationsqtl = 0;
    for(int i = 0; i < SimParameters.getChr(); i++){numbmutationsqtl += (SimParameters.get_QTL_chr())[i];}
    if(numbmutationsqtl > 0 && (SimParameters.get_QTL_chr()).size() != SimParameters.getChr())
    {
        cout << endl << "Number of QTL doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);
    }
    if(numbmutationsqtl == 0){logfilestring = logfilestring + "        - No Quantitative Trait Mutations Segregating.\n";}
    if(numbmutationsqtl != 0)
    {
        logfilestring = logfilestring + "        - Number of Quantitative Trait Mutations: \n";
        for(int i = 0; i < (SimParameters.get_QTL_chr()).size(); i++)
        {
            stringstream s1; s1 << (SimParameters.get_QTL_chr())[i]; string tempvar = s1.str();
            stringstream s2; s2 << i + 1; string tempvara = s2.str();
            logfilestring = logfilestring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("QUANTITATIVE_MAF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring + "        - Minor Allele Frequency Threshold for Quantitative Trait QTL's:\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateThresholdMAFQTL(atof(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateThresholdMAFQTL(0.01);
            logfilestring = logfilestring + "        - Minor Allele Frequency Threshold for Quantitative Trait QTL's:\t\t\t\t'0.01' (Default)\n"; break;
        }
    }
    if(SimParameters.getThresholdMAFQTL() >= 0.5)
    {
        cout << endl << "Quantitative MAF threshold can't be greater than 0.5! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0; string templethal, tempsublethal;
    while(1)
    {
        size_t fnd = parm[search].find("FITNESS_MAF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
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
            if(solvervariables.size() < 2 || solvervariables.size() > 3)
            {
                cout  << endl << "        - Should be 2 or 3 values not " << solvervariables.size() << " for fitness MAF!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables.size() == 2)
            {
                templethal = "        - Upper Allele Frequency Threshold for Fitness Lethal QTL's:\t\t\t\t\t'" + solvervariables[0] + "'\n";
                tempsublethal= "        - Upper Allele Frequency Threshold for Fitness Sub Lethal QTL's:\t\t\t\t'"+solvervariables[1]+"'\n";
                templethal=templethal+"        - Maximum difference from upper frequency threshold for Fitness Lethal QTL's:\t\t'0.01' (Default)\n";
                SimParameters.UpdateUpThrMAFLFit(atof(solvervariables[0].c_str()));
                SimParameters.UpdateUpThrMAFSFit(atof(solvervariables[1].c_str()));
                SimParameters.UpdateRangeMAFLFit(0.01);
                 break;
            }
            if(solvervariables.size() == 3)
            {
                templethal = "        - Upper Allele Frequency Threshold for Fitness Lethal Loci:\t\t\t\t\t'" + solvervariables[0] + "'\n";
                tempsublethal= "        - Upper Allele Frequency Threshold for Fitness Sub Lethal Loci:\t\t\t\t'"+solvervariables[1]+"'\n";
                templethal=templethal+"        - Maximum difference from upper frequency threshold for Fitness Lethal Loci:\t\t'"+solvervariables[2]+"'\n";
                SimParameters.UpdateUpThrMAFLFit(atof(solvervariables[0].c_str()));
                SimParameters.UpdateUpThrMAFSFit(atof(solvervariables[1].c_str()));
                SimParameters.UpdateRangeMAFLFit(atof(solvervariables[2].c_str())); break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateUpThrMAFLFit(0.02); SimParameters.UpdateUpThrMAFSFit(0.08); SimParameters.UpdateRangeMAFLFit(0.01);
            templethal = "        - Upper Allele Frequency Threshold for Fitness Lethal Loci:\t\t\t\t\t'0.02' (Default)\n";
            tempsublethal = "        - Upper Allele Frequency Threshold for Fitness Sub Lethal Loci:\t\t\t\t'0.08' (Default)\n";
            templethal=templethal+"        - Maximum difference from upper frequency threshold for Fitness Lethal Loci:\t\t'0.01' (Default)\n";
            break;
        }
    }
    if(SimParameters.getUpThrMAFLFit() >= 0.5 || SimParameters.getUpThrMAFSFit() >= 0.5)
    {
        cout << endl << "Fitness MAF upper threshold can't be greater than 0.5! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    if((SimParameters.getUpThrMAFLFit() - SimParameters.getRangeMAFLFit()) < 0)
    {
        cout << endl << "Fitness MAF range can't be less than than 0.0! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("FIT_LETHAL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            for(int i = 0; i < solvervariables.size(); i++){SimParameters.add_ToFTL_lethal_chr(atoi(solvervariables[i].c_str()));}
            break;
        }
        search++;
        if(search >= parm.size())
        {
            for(int i = 0; i < SimParameters.getChr(); i++){SimParameters.add_ToFTL_lethal_chr(0);}
            break;
        }
    }
    /* Double check to see if given Lethal FTL mutations must be same size as number of chromosomes */
    int numbmutationslftl = 0;
    for(int i = 0; i < SimParameters.getChr(); i++){numbmutationslftl += (SimParameters.get_FTL_lethal_chr())[i];}
    if(numbmutationslftl > 0 && (SimParameters.get_FTL_lethal_chr()).size() != SimParameters.getChr())
    {
        cout << endl << "Number of Lethal FTL doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);
    }
    if(numbmutationslftl == 0){logfilestring = logfilestring + "        - No Lethal Fitness Mutations Segregating.\n";}
    if(numbmutationslftl != 0)
    {
        logfilestring = logfilestring + "        - Number of Lethal Fitness Mutations: \n";
        for(int i = 0; i < (SimParameters.get_QTL_chr()).size(); i++)
        {
            stringstream s1; s1 << (SimParameters.get_FTL_lethal_chr())[i]; string tempvar = s1.str();
            stringstream s2; s2 << i + 1; string tempvara = s2.str();
            logfilestring = logfilestring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
        }
        logfilestring = logfilestring + templethal;
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("FIT_SUBLETHAL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            for(int i = 0; i < solvervariables.size(); i++){SimParameters.add_ToFTL_sublethal_chr(atoi(solvervariables[i].c_str()));}
            break;
        }
        search++;
        if(search >= parm.size())
        {
            for(int i = 0; i < SimParameters.getChr(); i++){SimParameters.add_ToFTL_sublethal_chr(0);}
            break;
        }
    }
    /* Double check to see if given QTL mutations must be same size as number of chromosomes */
    int numbmutationssftl = 0;
    for(int i = 0; i < SimParameters.getChr(); i++){numbmutationssftl += (SimParameters.get_FTL_sublethal_chr())[i];}
    if(numbmutationssftl > 0 && (SimParameters.get_FTL_sublethal_chr()).size() != SimParameters.getChr())
    {
        cout << endl << "Number of Sub-Lethal FTL  doesn't correspond to CHR number!" << endl; exit (EXIT_FAILURE);
    }
    if(numbmutationssftl == 0){logfilestring = logfilestring + "        - No Sub-Lethal Fitness Mutations Segregating.\n";}
    if(numbmutationssftl != 0)
    {
        logfilestring = logfilestring + "        - Number of Sub-Lethal Fitness Mutations: \n";
        for(int i = 0; i < (SimParameters.get_QTL_chr()).size(); i++)
        {
            stringstream s1; s1 << (SimParameters.get_FTL_sublethal_chr())[i]; string tempvar = s1.str();
            stringstream s2; s2 << i + 1; string tempvara = s2.str();
            logfilestring = logfilestring + "            \t\t\t\t\t\t\t\t\t\t\t\tChr " + tempvara + ":'" + tempvar + "'\n";
        }
        logfilestring = logfilestring+tempsublethal;
    }
    if((numbmutationsqtl+numbmutationslftl+numbmutationssftl) > 5000)
    {
        cout << endl << "Cannot have greater than 5000 QTL! Check parameter file." << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("HAPLOTYPE_SIZE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring+"        - SNP Haplotype Size:\t\t\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.Updatehaplo_size(atoi(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.Updatehaplo_size(50); logfilestring = logfilestring+"        - SNP Haplotype Size:\t\t\t\t\t\t\t\t\t\t'50' (Default)\n";break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("RECOMBINATION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring=logfilestring+"        - Recombination Position generated from the following distribution:\t\t\t\t'"+parm[search]+"'\n";
            SimParameters.UpdateRecombDis(parm[search]); break;
        }
        search++;
        if(search >= parm.size())
        {
            logfilestring=logfilestring+"        - Recombination Position generated from the following distribution:\t\t\t\t'Uniform' (Default)\n";
            SimParameters.UpdateRecombDis("Uniform"); break;
        }
    }
    if(SimParameters.getRecombDis() != "Uniform" && SimParameters.getRecombDis() != "Beta")
    {
        cout << endl << "RECOMBINATION (" << SimParameters.getRecombDis() << ") didn't equal Uniform or Beta! Check parameter file!" << endl;
        exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MUTATION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for MUTATION!" << endl; exit (EXIT_FAILURE);}
            logfilestring=logfilestring+"        - Mutation Rate:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            logfilestring=logfilestring+"        - Proportion of mutations that can be QTL:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.Updateu(strtod(solvervariables[0].c_str(),NULL));
            SimParameters.UpdatePropQTL(atof(solvervariables[1].c_str())); break;
        }
        search++; if(search >= parm.size())
        {
            SimParameters.Updateu(2.5e-8); SimParameters.UpdatePropQTL(0.0);
            logfilestring=logfilestring+"        - Mutation Rate:\t\t\t\t\t\t\t\t\t\t'2.5e-8' (Default)\n";
            logfilestring=logfilestring+"        - Proportion of mutations that can be QTL:\t\t\t\t\t\t\t'0.0' (Default)\n"; break;
        }
    }
    ///////////////////////////////////////////////////////////////////
    ///////////        QTL/FTL Distribution Parameters      ///////////
    ///////////////////////////////////////////////////////////////////
    logfilestring=logfilestring+"    - QTL Effects: \n";
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("ADD_QUAN:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for ADD_QUAN!" << endl; exit (EXIT_FAILURE);}
            logfilestring=logfilestring+"        - Gamma Shape: Additive Quantitative Trait:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            logfilestring=logfilestring+ "        - Gamma Scale: Additive Quantitative Trait:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateGamma_Shape(atof(solvervariables[0].c_str()));
            SimParameters.UpdateGamma_Scale(atof(solvervariables[1].c_str()));  break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateGamma_Shape(0.4); SimParameters.UpdateGamma_Scale(1.66);
            logfilestring=logfilestring+"        - Gamma Shape: Additive Quantitative Trait:\t\t\t\t\t\t\t'0.4' (Default)\n";
            logfilestring=logfilestring+"        - Gamma Scale: Additive Quantitative Trait:\t\t\t\t\t\t\t'1.66' (Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("DOM_QUAN:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for DOM_QUAN!" << endl; exit (EXIT_FAILURE);}
            logfilestring=logfilestring+"        - Normal Mean: Dominance Quantitative Trait:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            logfilestring=logfilestring+"        - Normal SD: Dominance Quantitative Trait:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateNormal_meanRelDom(atof(solvervariables[0].c_str()));
            SimParameters.UpdateNormal_varRelDom(atof(solvervariables[1].c_str()));  break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateNormal_meanRelDom(0.1);  SimParameters.UpdateNormal_varRelDom(0.2);
            logfilestring=logfilestring+"        - Normal Mean: Dominance Quantitative Trait:\t\t\t\t\t\t\t'0.1' (Default)\n";
            logfilestring=logfilestring+"        - Normal SD: Dominance Quantitative Trait:\t\t\t\t\t\t\t'0.2' (Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("QUANTITATIVE_PARAMETERIZATION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 1)
            {
                cout << endl << "        - Should be one parameter for QUANTITATIVE_PARAMETERIZATION!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[0] != "biological" && solvervariables[0] != "statistical")
            {
                cout << endl << "        - Wrong parameter for QUANTITATIVE_PARAMETERIZATION. Check Manual." << endl; exit (EXIT_FAILURE);
            }
            logfilestring=logfilestring+"        - True breeding values and dominance deviation parameterized based on: \t\t\t'";
            logfilestring=logfilestring+solvervariables[0] + "' model. \n";
            SimParameters.UpdateQuantParam(solvervariables[0]); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateQuantParam("biological");
            logfilestring=logfilestring+"        - True breeding values and dominance deviation parameterized based on: \t\t\t'";
            logfilestring=logfilestring+"biological' model. (Default)\n"; break;
        }
    }
    search = 0; string templethalqtl;
    while(1)
    {
        size_t fnd = parm[search].find("LTHA:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for LTHA!" << endl; exit (EXIT_FAILURE);}
            templethalqtl = "        - Gamma Shape: S value Fitness Lethal:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
            templethalqtl = templethalqtl + "        - Gamma Scale: S value Fitness Lethal:\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
            SimParameters.UpdateGamma_Shape_Lethal(atof(solvervariables[0].c_str()));
            SimParameters.UpdateGamma_Scale_Lethal(atof(solvervariables[1].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateGamma_Shape_Lethal(900.0); SimParameters.UpdateGamma_Scale_Lethal(0.001);
            templethalqtl = templethalqtl+"        - Gamma Shape: S value Fitness Lethal:\t\t\t\t\t\t\t'900.0' (Default)\n";
            templethalqtl = templethalqtl+"        - Gamma Scale: S value Fitness Lethal:\t\t\t\t\t\t\t'0.001' (Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("LTHD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for LTHD!" << endl; exit (EXIT_FAILURE);}
            templethalqtl = templethalqtl+"        - Normal Mean: Dominance Fitness Lethal:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            templethalqtl = templethalqtl+"        - Normal Variance: Dominance Fitness Lethal:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateNormal_meanRelDom_Lethal(atof(solvervariables[0].c_str()));
            SimParameters.UpdateNormal_varRelDom_Lethal(atof(solvervariables[1].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateNormal_meanRelDom_Lethal(0.001); SimParameters.UpdateNormal_varRelDom_Lethal(0.0001);
            templethalqtl = templethalqtl+"        - Normal Mean: Dominance Fitness Lethal:\t\t\t\t\t\t\t'0.001' (Default)\n";
            templethalqtl = templethalqtl+"        - Normal Variance: Dominance Fitness Lethal:\t\t\t\t\t\t\t'0.0001' (Default)\n"; break;
        }
    }
    if(numbmutationslftl != 0){logfilestring = logfilestring + templethalqtl;}
    search = 0; string tempsublethalqtl;
    while(1)
    {
        size_t fnd = parm[search].find("SUBA:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for SUBA!" << endl; exit (EXIT_FAILURE);}
            tempsublethalqtl="        - Gamma Shape: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            tempsublethalqtl=tempsublethalqtl+"        - Gamma Scale: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateGamma_Shape_SubLethal(atof(solvervariables[0].c_str()));
            SimParameters.UpdateGamma_Scale_SubLethal(atof(solvervariables[1].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateGamma_Shape_SubLethal(0.1); SimParameters.UpdateGamma_Scale_SubLethal(0.2);
            tempsublethalqtl=tempsublethalqtl+"        - Gamma Shape: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'0.1' (Default)\n";
            tempsublethalqtl=tempsublethalqtl+"        - Gamma Scale: S value Fitness Sub-Lethal:\t\t\t\t\t\t\t'0.2' (Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("SUBD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for SUBD!" << endl; exit (EXIT_FAILURE);}
            tempsublethalqtl=tempsublethalqtl+"        - Normal Mean: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            tempsublethalqtl=tempsublethalqtl+"        - Normal Variance: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateNormal_meanRelDom_SubLethal(atof(solvervariables[0].c_str()));
            SimParameters.UpdateNormal_varRelDom_SubLethal(atof(solvervariables[1].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateNormal_meanRelDom_SubLethal(0.3); SimParameters.UpdateNormal_varRelDom_SubLethal(0.1);
            tempsublethalqtl=tempsublethalqtl+"        - Normal Mean: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t\t'0.3' (Default)\n";
            tempsublethalqtl=tempsublethalqtl+"        - Normal Variance: Dominance Fitness Sub-Lethal:\t\t\t\t\t\t'0.1' (Default)\n"; break;
        }
    }
    if(numbmutationssftl != 0){logfilestring = logfilestring + tempsublethalqtl;}
    search = 0; string tempcovar;
    while(1)
    {
        size_t fnd = parm[search].find("COVAR:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for COVAR!" << endl; exit (EXIT_FAILURE);}
            tempcovar="        - Proportion of QTL with Pleiotropic Fitness Effects :\t\t\t\t\t'" + solvervariables[0] + "'\n";
            tempcovar=tempcovar+ "        - Correlation (Rank) between QTL and Fitness Effects:\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.Updateproppleitropic(atof(solvervariables[0].c_str()));
            SimParameters.Updategencorr(atof(solvervariables[1].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.Updateproppleitropic(0); SimParameters.Updategencorr(0);
            tempcovar=tempcovar+ "        - Proportion of QTL with Pleiotropic Fitness Effects:\t\t\t\t\t'0.0' (Default)\n";
            tempcovar=tempcovar+ "        - Additive Genetic Correlation between QTL and Fitness Effects:\t\t\t\t'0.0' (Default)\n"; break;
        }
    }
    if(numbmutationssftl != 0){logfilestring = logfilestring + tempcovar;}
    if(SimParameters.getgencorr() < 0)
    {
        cout << "        - Correlation can't be negative due to how effects are constructed. Alter Favorable direction!" << endl; exit (EXIT_FAILURE);
    }
    if(numbmutationssftl == 0 && (SimParameters.getproppleitropic() > 0.0 || SimParameters.getgencorr() != 0.0))
    {
        cout << endl << "Cannot have a genetic correlation between fitness and quantitative QTL when no sublethal QTL!" << endl;
        exit (EXIT_FAILURE);
    }
    ///////////////////////////////////////////////////////////////////
    ///////////             Population Parameters           ///////////
    ///////////////////////////////////////////////////////////////////
    logfilestring = logfilestring+"    -Founder Population Parameters:\n";
    search = 0; int ne_founder = -10;
    while(1)
    {
        size_t fnd = parm[search].find("FOUNDER_Effective_Size:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            if(parm[search]=="Ne70"||parm[search]=="Ne100_Scen1"||parm[search]=="Ne100_Scen2"||parm[search]=="Ne250"||parm[search]=="Ne1000")
            {
                SimParameters.UpdateNe_spec(parm[search]); SimParameters.Updatene_founder(-5); break;
            }
            if(parm[search]=="CustomNe")
            {
                SimParameters.UpdateNe_spec(parm[search]); SimParameters.Updatene_founder(-5); break;
            }
            if(parm[search]!="Ne70"||parm[search]!="Ne100_Scen1"||parm[search]!="Ne100_Scen2"||parm[search]!="Ne250"||parm[search]!="Ne1000"||parm[search]!="CustomNe")
            {
                SimParameters.UpdateNe_spec(""); SimParameters.Updatene_founder(atoi(parm[search].c_str())); break;
            }
        }
        search++; if(search >= parm.size()){cout<<endl<<"Couldn't find 'FOUNDER_Effective_Size:' variable in parameter file!"<<endl;exit(EXIT_FAILURE);}
    }
    /* will exit if 0 because can't be zero and if give a letter value will default to zero */
    if(SimParameters.getne_founder() == -10 || SimParameters.getne_founder() == 0)
    {
        cout << endl << "        - Wrong parameter for Founder Effective Population Size. Check Manual." << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getne_founder() != -5 && SimParameters.getNe_spec() == "")
    {
        stringstream s1; s1 << SimParameters.getne_founder(); string tempvar = s1.str();
        logfilestring= logfilestring+"        - Effective Population Size in Founders:\t\t\t\t\t\t\t'" + tempvar + "'\n";
    }
    if(SimParameters.getne_founder() == -5 && SimParameters.getNe_spec() != "")
    {
        logfilestring= logfilestring+"        - Effective Population Modeled from the scenario:\t\t\t\t\t\t'" + SimParameters.getNe_spec() + "'\n";
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MALE_FEMALE_FOUNDER:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(6,"");
            for(int i = 0; i < 6; i++)
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
            if(solvervariables.size() != 2 && solvervariables.size() != 3 && solvervariables.size() != 4)
            {
                cout << endl << "Incorrect number of parameters for 'MALE_FEMALE_FOUNDER:'! Check Manual." << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size() == 2)
            {
                SimParameters.Updatefoundermale(atoi(solvervariables[0].c_str()));
                SimParameters.Updatefounderfemale(atoi(solvervariables[1].c_str()));
                SimParameters.Updatefounderselect("random"); SimParameters.UpdateGenfoundsel(1);
                logfilestring=logfilestring+"        - Number of Male Founder Individuals:\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring=logfilestring+"        - Number of Female Founder Individuals:\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring=logfilestring+"        - Founder Individuals selected based on:\t\t\t\t\t\t\t'Random' (Default)\n";
                logfilestring=logfilestring+"        - Generations Founder Individuals randomly selected for:\t\t\t\t'1' (Default)\n";
            }
            if(solvervariables.size() == 3)
            {
                SimParameters.Updatefoundermale(atoi(solvervariables[0].c_str()));
                SimParameters.Updatefounderfemale(atoi(solvervariables[1].c_str()));
                SimParameters.Updatefounderselect(solvervariables[2]); SimParameters.UpdateGenfoundsel(1);
                logfilestring=logfilestring+"        - Number of Male Founder Individuals:\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring=logfilestring+"        - Number of Female Founder Individuals:\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring=logfilestring+"        - Founder Individuals selected based on:\t\t\t\t\t\t\t'"+SimParameters.getfounderselect()+"'\n";
                logfilestring=logfilestring+"        - Generations Founder Individuals "+SimParameters.getfounderselect();
                logfilestring=logfilestring+" selected for:\t\t\t\t\t'1' (Default)\n";
            }
            if(solvervariables.size() == 4)
            {
                SimParameters.Updatefoundermale(atoi(solvervariables[0].c_str()));
                SimParameters.Updatefounderfemale(atoi(solvervariables[1].c_str()));
                SimParameters.Updatefounderselect(solvervariables[2]);
                SimParameters.UpdateGenfoundsel(atoi(solvervariables[3].c_str()));
                logfilestring=logfilestring+"        - Number of Male Founder Individuals:\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring=logfilestring+"        - Number of Female Founder Individuals:\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring=logfilestring+"        - Founder Individuals selected based on:\t\t\t\t\t\t\t'"+SimParameters.getfounderselect()+"'\n";
                logfilestring=logfilestring+"        - Generations Founder Individuals "+SimParameters.getfounderselect()+" selected for:\t\t\t\t\t'";
                logfilestring=logfilestring+solvervariables[3]+"'\n";
            }
            break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'MALE_FEMALE_FOUNDER:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SimParameters.getfounderselect() != "random")
    {
        cout << endl << " Founder SELECTION ("<<SimParameters.getfounderselect()<<") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("FOUNDER_HAPLOTYPES:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            SimParameters.Updatefnd_haplo(atoi(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.Updatefnd_haplo(2*(SimParameters.getfoundermale()+SimParameters.getfounderfemale()+200)); break;
        }
    }
    stringstream s1founderhap; s1founderhap << SimParameters.getfnd_haplo(); string s1founderhaptemp = s1founderhap.str();
    logfilestring=logfilestring+"        - Founder Haplotypes:\t\t\t\t\t\t\t\t\t\t'" + s1founderhaptemp + "'\n";
    if((SimParameters.getfoundermale() + SimParameters.getfounderfemale()) > (SimParameters.getfnd_haplo()/2))
    {
        cout << endl << "Not enough haplotypes to generate founder individuals. Check Manual!" << endl; exit (EXIT_FAILURE);
    }
    logfilestring = logfilestring+"    -Variance Components:\n";
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VARIANCE_A:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 1 && solvervariables.size() != 3)
            {
                cout << endl << "Incorrent number of (co)variances. Needs to be 1 or 3!!" << endl; exit(EXIT_FAILURE);
            }
            if(solvervariables.size() == 1)
            {
                SimParameters.add_Var_Additive(atof(solvervariables[0].c_str()));
                logfilestring = logfilestring+"        - Variance due to additive gene action:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n"; break;
            }
            if(solvervariables.size() == 3)
            {
                for(int i = 0; i < solvervariables.size(); i++){SimParameters.add_Var_Additive(atof(solvervariables[i].c_str()));}
                logfilestring = logfilestring+"        - Variance due to additive gene action (Trait 1):\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                logfilestring = logfilestring+"        - Variance due to additive gene action (Trait 2):\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                logfilestring = logfilestring+"        - Correlation between additive gene action for Trait 1 and 2:\t\t\t\t'" + solvervariables[1] + "'\n";
                break;
            }
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'VARIANCE_A:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VARIANCE_D:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 1 && solvervariables.size() != 2)
            {
                cout << endl << "Incorrent number of (co)variances. Needs to be 1 or 2!!" << endl; exit(EXIT_FAILURE);
            }
            if(solvervariables.size() == 1)
            {
                SimParameters.add_Var_Dominance(atof(solvervariables[0].c_str()));
                logfilestring = logfilestring+"        - Variance due to dominant gene action:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n"; break;
            }
            if(solvervariables.size() == 2)
            {
                SimParameters.add_Var_Dominance(atof(solvervariables[0].c_str()));
                SimParameters.add_Var_Dominance(0.0);
                SimParameters.add_Var_Dominance(atof(solvervariables[1].c_str()));
                logfilestring = logfilestring+"        - Variance due to dominance gene action (Trait 1):\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                logfilestring = logfilestring+"        - Variance due to dominance gene action (Trait 2):\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            if((SimParameters.get_Var_Additive()).size() == 1)
            {
                SimParameters.add_Var_Dominance(0.0);
                logfilestring = logfilestring+"        - Variance due to dominant gene action:\t\t\t\t\t\t\t'0.00' (Default)\n"; break;
            }
            if((SimParameters.get_Var_Additive()).size() == 3)
            {
                SimParameters.add_Var_Dominance(0.0); SimParameters.add_Var_Dominance(0.0); SimParameters.add_Var_Dominance(0.0);
                logfilestring = logfilestring+"        - Variance due to dominance gene action (Trait 1):\t\t\t\t\t\t'0.00' (Default)\n";
                logfilestring = logfilestring+"        - Variance due to dominance gene action (Trait 2):\t\t\t\t\t\t'0.00' (Default)\n"; break;
            }
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VARIANCE_R:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(31,"");
            for(int i = 0; i < 31; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 31;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 1 && solvervariables.size() != 3)
            {
                cout << endl << "Incorrent number of (co)variances. Needs to be 1 or 3!!" << endl; exit(EXIT_FAILURE);
            }
            if(solvervariables.size() == 1)
            {
                SimParameters.add_Var_Residual(atof(solvervariables[0].c_str()));
                logfilestring = logfilestring+"        - Variance due to environment:\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n"; break;
            }
            if(solvervariables.size() == 3)
            {
                for(int i = 0; i < solvervariables.size(); i++){SimParameters.add_Var_Residual(atof(solvervariables[i].c_str()));}
                logfilestring = logfilestring+"        - Residual Variance (Trait 1):\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                logfilestring = logfilestring+"        - Residual Variance (Trait 2):\t\t\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                logfilestring = logfilestring+"        - Residual Correlation for Trait 1 and 2:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            if((SimParameters.get_Var_Additive()).size() == 1)
            {
                double resvar = 1.0 - ((SimParameters.get_Var_Additive())[0] + (SimParameters.get_Var_Dominance())[0]);
                stringstream s2; s2 << resvar;
                logfilestring = logfilestring+"        - Variance due to environment:\t\t\t\t\t\t\t\t'" + s2.str() + "' (Default)\n";
                SimParameters.add_Var_Residual(resvar); break;
            }
            if((SimParameters.get_Var_Additive()).size() == 3)
            {
                double resvar1 = 1.0 - ((SimParameters.get_Var_Additive())[0] + (SimParameters.get_Var_Dominance())[0]);
                stringstream s2; s2 << resvar1;
                double resvar2 = 1.0 - ((SimParameters.get_Var_Additive())[2] + (SimParameters.get_Var_Dominance())[2]);
                stringstream s3; s3 << resvar2;
                logfilestring = logfilestring+"        - Variance due to environment (Trait 1):\t\t\t\t\t\t\t'" + s2.str() + "' (Default)\n";
                logfilestring = logfilestring+"        - Variance due to environment (Trait 2):\t\t\t\t\t\t\t'" + s3.str() + "' (Default)\n";
                logfilestring = logfilestring+"        - Residual Correlation for Trait 1 and 2:\t\t\t\t\t\t\t'0.00' (Default)\n";
                SimParameters.add_Var_Residual(resvar1); SimParameters.add_Var_Residual(0.0); SimParameters.add_Var_Residual(resvar2); break;
            }
        }
    }
    if((SimParameters.get_Var_Additive()).size() == 1 && (SimParameters.get_Var_Dominance()).size() == 1 && (SimParameters.get_Var_Residual()).size()==1)
    {
        SimParameters.Updatenumbertraits(1);
        logfilestring = logfilestring+"        - Number of quantitative traits simulated:\t\t\t\t\t\t\t'1'\n";
    } else if ((SimParameters.get_Var_Additive()).size()==3 && (SimParameters.get_Var_Dominance()).size()==3 && (SimParameters.get_Var_Residual()).size()==3) {
        SimParameters.Updatenumbertraits(2);
        logfilestring = logfilestring+"        - Number of quantitative traits simulated:\t\t\t\t\t\t\t'2'\n";
    } else {
        cout << "Incorrect Parameter File for Additive, Dominance or Residual Variance. Check parameter file!!" << endl; exit (EXIT_FAILURE);
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    ///////////             Selection and Culling Parameters Parameters           ///////////
    /////////////////////////////////////////////////////////////////////////////////////////
    logfilestring = logfilestring+"    -Recent Population, Selection and Culling Parameters:\n";
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("GENERATIONS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring = logfilestring+"        - Number of Generations to Simulate:\t\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateGener(atoi(parm[search].c_str())); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'GENERATIONS:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SimParameters.getGener() <= SimParameters.getGenfoundsel())
    {
        cout << endl << "Number of Generations simulated is less than founder population selection scenario!! Check Manual" <<endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("INDIVIDUALS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(9,"");
            for(int i = 0; i < 9; i++)
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
            if(solvervariables.size() != 4){cout << endl << "Incorrect number of parameters for 'INDIVIDUALS:'! Check Manual." << endl; exit (EXIT_FAILURE);}
            if(solvervariables.size() == 4)
            {
                SimParameters.UpdateSires(atoi(solvervariables[0].c_str())); SimParameters.UpdateSireRepl(atof(solvervariables[1].c_str()));
                SimParameters.UpdateDams(atoi(solvervariables[2].c_str())); SimParameters.UpdateDamRepl(atof(solvervariables[3].c_str()));
                logfilestring=logfilestring+"        - Number of Males in Population per Generation:\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                logfilestring=logfilestring+"        - Sire Replacement Rate:\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                logfilestring=logfilestring+"        - Number of Females in Population per Generation:\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                logfilestring=logfilestring+"        - Dam Replacement Rate:\t\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n"; break;
            }
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'INDIVIDUALS:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SimParameters.getSireRepl() <= 0 || SimParameters.getSireRepl() > 1.0)
    {
        cout << endl << "Replacement rate outside of range (0.0 - 1.0). Check Parameterfile" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getDamRepl() <= 0 || SimParameters.getDamRepl() > 1.0)
    {
        cout << endl << "Replacement rate outside of range (0.0 - 1.0). Check Parameterfile" << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PROGENY:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring=logfilestring+"        - Number of offspring for each mating pair:\t\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.UpdateOffspring(atoi(parm[search].c_str())); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'PROGENY:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0; string selectionstring;
    while(1)
    {
        size_t fnd = parm[search].find("SELECTION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(6,"");            /* expect 2 */
            for(int i = 0; i < 6; i++)
            {
                size_t pos = parm[search].find(" ",0);
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 2){cout << endl << "Wrong number of parameters for 'SELECTION:' parameter." << endl; exit (EXIT_FAILURE);}
            if(solvervariables.size() == 2)
            {
                SimParameters.UpdateSelection(solvervariables[0]); SimParameters.UpdateSelectionDir(solvervariables[1]);
                logfilestring=logfilestring+"        - Selection Criteria:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                logfilestring=logfilestring+"        - Selection Direction:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n"; break;
            }
        }
        search++;
        if(search >= parm.size()){cout << endl << "Couldn't find 'SELECTION:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SimParameters.getSelection() != "random" && SimParameters.getSelection() != "phenotype" && SimParameters.getSelection() != "tbv" && SimParameters.getSelection() != "ebv" && SimParameters.getSelection() != "ocs" && SimParameters.getSelection() != "index_tbv" && SimParameters.getSelection() != "index_ebv")
    {
        cout << endl << "SELECTION (" << SimParameters.getSelection() << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelectionDir() != "low" && SimParameters.getSelectionDir() != "high")
    {
        cout << endl << "SELECTIONDIR (" << SimParameters.getSelectionDir() << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    /* If chose index option has to be two traits that are being generated */
    if((SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv") && SimParameters.getnumbertraits() != 2)
    {
        cout << endl << "Need to simulation two trait to have 'index_tbv' or 'index_ebv' selection." << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("INDEX_PROPORTIONS:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getSelection() != "index_tbv" && SimParameters.getSelection() != "index_ebv")
            {
                cout << endl << "Didn't choose 'index_tbv' or 'index_ebv'. Don't need 'INDEX_PROPORTIONS' parameter." << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(6,"");            /* expect 2 */
            for(int i = 0; i < 6; i++)
            {
                size_t pos = parm[search].find(" ",0);
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
            if(solvervariables.size() != 2){cout << endl << "Need to have two weights (i.e. one for each trait)" << endl; exit (EXIT_FAILURE);}
            if(((atof(solvervariables[0].c_str()) + (atof(solvervariables[1].c_str()))) != 1.0))
            {
                cout << endl << "Sum of index weights doesn't equal one!!" << endl; exit (EXIT_FAILURE);
            }
            SimParameters.add_IndexWeights(atof(solvervariables[0].c_str())); SimParameters.add_IndexWeights(atof(solvervariables[1].c_str()));
            logfilestring = logfilestring+"        - Index Weight for Trait 1:\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            logfilestring = logfilestring+"        - Index Weight for Trait 2:\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";break;
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getSelection() == "index_tbv" || SimParameters.getSelection() == "index_ebv")
            {
                cout << endl << "You chose 'index_tbv' or 'index_ebv' option. Need to 'INDEX_PROPORTIONS' parameter." << endl; exit (EXIT_FAILURE);
            }
            break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PHENOTYPE_STRATEGY:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getSelection() != "ebv")
            {
                cout << endl << "Need to do 'ebv' selection for 'PHENOTYPE_STRATEGY' option. Change parameter file accordingly!" << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(12,"");
            for(int i = 0; i < 12; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 4 && solvervariables.size() != 5)
            {
                cout<<endl<<"'PHENOTYPE_STRATEGY:' wrong number of parameters. Look at manual."<<endl; exit(EXIT_FAILURE);
            }
            if(SimParameters.getnumbertraits() == 2 && SimParameters.getSelection() != "ebv")
            {
                cout <<endl<<"Doing more than one trait so need to have 'PHENOTYPE_STRATEGY1:' and 'PHENOTYPE_STRATEGY2:'!. Check parameter file!" << endl;
            }
            if(atof(solvervariables[0].c_str()) < 0.0 || atof(solvervariables[0].c_str()) > 1.0)
            {
                cout << endl << "PHENOTYPE_STRATEGY (" << solvervariables[0] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[2].c_str()) < 0.0 || atof(solvervariables[2].c_str()) > 1.0)
            {
                cout << endl << "PHENOTYPE_STRATEGY (" << solvervariables[2] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[1] != "pheno_atselection" && solvervariables[1] != "pheno_afterselection" && solvervariables[1] != "pheno_parents" && solvervariables[1] != "random_atselection" && solvervariables[1] != "random_afterselection" && solvervariables[1] != "random_parents" && solvervariables[1] != "ebv_atselection" && solvervariables[1] != "ebv_afterselection" && solvervariables[1] != "ebv_parents" && solvervariables[1] != "litterrandom_atselection" && solvervariables[1] != "litterrandom_afterselection")
            {
                cout << endl << "PHENOTYPE_STRATEGY (" << solvervariables[1] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[3] != "pheno_atselection" && solvervariables[3] != "pheno_afterselection" && solvervariables[3] != "pheno_parents" &&               solvervariables[3] != "random_atselection" && solvervariables[3] != "random_afterselection" && solvervariables[3] != "random_parents" &&               solvervariables[3] != "ebv_atselection" && solvervariables[3] != "ebv_afterselection" && solvervariables[3] != "ebv_parents" &&               solvervariables[3] != "litterrandom_atselection" && solvervariables[3] != "litterrandom_afterselection")
            {
                cout << endl << "PHENOTYPE_STRATEGY (" << solvervariables[3] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==5 && (solvervariables[3]!="ebv_atselection" && solvervariables[3]!="ebv_afterselection" && solvervariables[3]!="ebv_parents" && solvervariables[1]!="ebv_atselection" && solvervariables[1]!="ebv_afterselection" && solvervariables[1]!="ebv_parents"))
            {
                cout << endl << "PHENOTYPE_STRATEGY (" << solvervariables[4] << ") is only needed if doing ebv* based phenotype strategy" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==4)
            {
                logfilestring = logfilestring+"        - Phenotyping strategy parameters\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'"+solvervariables[2]+"'\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                if(solvervariables[3]=="ebv_atselection" || solvervariables[3]=="ebv_afterselection" || solvervariables[3]=="ebv_parents" || solvervariables[1] =="ebv_atselection" || solvervariables[1] =="ebv_afterselection" || solvervariables[1]=="ebv_parents")
                {
                    logfilestring = logfilestring+"             - Portion of ebv distribution animals are phenotyped from: \t\t\t\t'";
                    logfilestring = logfilestring+SimParameters.getSelectionDir()+"' (Default)\n";
                }
                SimParameters.add_MalePropPhenotype_vec(atof(solvervariables[0].c_str()));
                SimParameters.add_MaleWhoPhenotype_vec(solvervariables[1]);
                SimParameters.add_FemalePropPhenotype_vec(atof(solvervariables[2].c_str()));
                SimParameters.add_FemaleWhoPhenotype_vec(solvervariables[3]);
                SimParameters.add_PortionofDistribution_vec(SimParameters.getSelectionDir()); break;
            }
            if(solvervariables.size()==5)
            {
                logfilestring = logfilestring+"        - Phenotyping strategy parameters\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'"+solvervariables[2]+"'\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                logfilestring = logfilestring+"             - Portion of ebv distribution animals are phenotyped from: \t\t\t\t'"+solvervariables[4]+"'\n";
                SimParameters.add_MalePropPhenotype_vec(atof(solvervariables[0].c_str()));
                SimParameters.add_MaleWhoPhenotype_vec(solvervariables[1]);
                SimParameters.add_FemalePropPhenotype_vec(atof(solvervariables[2].c_str()));
                SimParameters.add_FemaleWhoPhenotype_vec(solvervariables[3]);
                SimParameters.add_PortionofDistribution_vec(solvervariables[4]); break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getSelection() == "ebv" || SimParameters.getSelection() == "random" || SimParameters.getSelection() == "phenotype" || SimParameters.getSelection() == "tbv")
            {
                SimParameters.add_MalePropPhenotype_vec(1.0);
                SimParameters.add_MaleWhoPhenotype_vec("pheno_atselection");
                SimParameters.add_FemalePropPhenotype_vec(1.0);
                SimParameters.add_FemaleWhoPhenotype_vec("pheno_atselection");
                SimParameters.add_PortionofDistribution_vec(SimParameters.getSelectionDir());
                logfilestring = logfilestring+"        - Phenotyping strategy parameters\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'1.0' (Default)\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'pheno_atselection' (Default)\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'1.0' (Default)\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'pheno_atselection' (Default)\n"; break;
            } else {break;}
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PHENOTYPE_STRATEGY1:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getSelection() != "index_ebv")
            {
                cout << endl << "Need to do 'index_ebv' selection for this option. Change parameter file accordingly!" << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(12,"");
            for(int i = 0; i < 12; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 4 && solvervariables.size() != 5)
            {
                cout<<endl<<"'PHENOTYPE_STRATEGY1:' wrong number of parameters. Look at manual."<<endl; exit(EXIT_FAILURE);
            }
            if(atof(solvervariables[0].c_str()) < 0.0 || atof(solvervariables[0].c_str()) > 1.0)
            {
                cout << endl << "PHENOTYPE_STRATEGY1 (" << solvervariables[0] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[2].c_str()) < 0.0 || atof(solvervariables[2].c_str()) > 1.0)
            {
                cout << endl << "PHENOTYPE_STRATEGY1 (" << solvervariables[2] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[1] != "pheno_atselection" && solvervariables[1] != "pheno_afterselection" && solvervariables[1] != "pheno_parents" &&               solvervariables[1] != "random_atselection" && solvervariables[1] != "random_afterselection" && solvervariables[1] != "random_parents" &&               solvervariables[1] != "ebv_atselection" && solvervariables[1] != "ebv_afterselection" && solvervariables[1] != "ebv_parents" &&               solvervariables[1] != "litterrandom_atselection" && solvervariables[1] != "litterrandom_afterselection")
            {
                cout << endl << "PHENOTYPE_STRATEGY1 (" << solvervariables[1] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[3] != "pheno_atselection" && solvervariables[3] != "pheno_afterselection" && solvervariables[3] != "pheno_parents" &&               solvervariables[3] != "random_atselection" && solvervariables[3] != "random_afterselection" && solvervariables[3] != "random_parents" &&               solvervariables[3] != "ebv_atselection" && solvervariables[3] != "ebv_afterselection" && solvervariables[3] != "ebv_parents" &&               solvervariables[3] != "litterrandom_atselection" && solvervariables[3] != "litterrandom_afterselection")
            {
                cout << endl << "PHENOTYPE_STRATEGY1 (" << solvervariables[3] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==5 && (solvervariables[3]!="ebv_atselection" && solvervariables[3]!="ebv_afterselection" && solvervariables[3]!="ebv_parents" && solvervariables[1]!="ebv_atselection" && solvervariables[1]!="ebv_afterselection" && solvervariables[1]!="ebv_parents"))
            {
                cout << endl << "PHENOTYPE_STRATEGY1 (" << solvervariables[4] << ") is only needed if doing ebv* based phenotype strategy" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==4)
            {
                logfilestring = logfilestring+"        - Phenotyping strategy parameter for Trait 1\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'"+solvervariables[2]+"'\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                if(solvervariables[3]=="ebv_atselection" || solvervariables[3]=="ebv_afterselection" || solvervariables[3]=="ebv_parents" || solvervariables[1] =="ebv_atselection" || solvervariables[1] =="ebv_afterselection" || solvervariables[1]=="ebv_parents")
                {
                    logfilestring = logfilestring+"             - Portion of ebv distribution animals are phenotyped from: \t\t\t\t'";
                    logfilestring = logfilestring+SimParameters.getSelectionDir()+"' (Default)\n";
                }
                SimParameters.add_MalePropPhenotype_vec(atof(solvervariables[0].c_str()));
                SimParameters.add_MaleWhoPhenotype_vec(solvervariables[1]);
                SimParameters.add_FemalePropPhenotype_vec(atof(solvervariables[2].c_str()));
                SimParameters.add_FemaleWhoPhenotype_vec(solvervariables[3]);
                SimParameters.add_PortionofDistribution_vec(SimParameters.getSelectionDir()); break;
            }
            if(solvervariables.size()==5)
            {
                logfilestring = logfilestring+"        - Phenotyping strategy parameter for Trait 1\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'"+solvervariables[2]+"'\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                logfilestring = logfilestring+"             - Portion of ebv distribution animals are phenotyped from: \t\t\t\t'"+solvervariables[4]+"'\n";
                SimParameters.add_MalePropPhenotype_vec(atof(solvervariables[0].c_str()));
                SimParameters.add_MaleWhoPhenotype_vec(solvervariables[1]);
                SimParameters.add_FemalePropPhenotype_vec(atof(solvervariables[2].c_str()));
                SimParameters.add_FemaleWhoPhenotype_vec(solvervariables[3]);
                SimParameters.add_PortionofDistribution_vec(solvervariables[4]); break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "index_tbv")
            {
                SimParameters.add_MalePropPhenotype_vec(1.0);
                SimParameters.add_MaleWhoPhenotype_vec("pheno_atselection");
                SimParameters.add_FemalePropPhenotype_vec(1.0);
                SimParameters.add_FemaleWhoPhenotype_vec("pheno_atselection");
                SimParameters.add_PortionofDistribution_vec(SimParameters.getSelectionDir());
                logfilestring = logfilestring+"        - Phenotyping strategy parameter for Trait 1\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'1.0' (Default)\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'pheno_atselection' (Default)\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'1.0' (Default)\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'pheno_atselection' (Default)\n"; break;
            } else {break;}
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PHENOTYPE_STRATEGY2:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getSelection() != "index_ebv")
            {
                cout << endl << "Need to do 'index_ebv' selection for this option. Change parameter file accordingly!" << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(12,"");
            for(int i = 0; i < 12; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 4 && solvervariables.size() != 5)
            {
                cout<<endl<<"'PHENOTYPE_STRATEGY2:' wrong number of parameters. Look at manual."<<endl; exit(EXIT_FAILURE);
            }
            if(atof(solvervariables[0].c_str()) < 0.0 || atof(solvervariables[0].c_str()) > 1.0)
            {
                cout << endl << "PHENOTYPE_STRATEGY2 (" << solvervariables[0] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[2].c_str()) < 0.0 || atof(solvervariables[2].c_str()) > 1.0)
            {
                cout << endl << "PHENOTYPE_STRATEGY2 (" << solvervariables[2] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[1] != "pheno_atselection" && solvervariables[1] != "pheno_afterselection" && solvervariables[1] != "pheno_parents" &&               solvervariables[1] != "random_atselection" && solvervariables[1] != "random_afterselection" && solvervariables[1] != "random_parents" &&               solvervariables[1] != "ebv_atselection" && solvervariables[1] != "ebv_afterselection" && solvervariables[1] != "ebv_parents" &&               solvervariables[1] != "litterrandom_atselection" && solvervariables[1] != "litterrandom_afterselection")
            {
                cout << endl << "PHENOTYPE_STRATEGY2 (" << solvervariables[1] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[3] != "pheno_atselection" && solvervariables[3] != "pheno_afterselection" && solvervariables[3] != "pheno_parents" &&               solvervariables[3] != "random_atselection" && solvervariables[3] != "random_afterselection" && solvervariables[3] != "random_parents" &&               solvervariables[3] != "ebv_atselection" && solvervariables[3] != "ebv_afterselection" && solvervariables[3] != "ebv_parents" &&               solvervariables[3] != "litterrandom_atselection" && solvervariables[3] != "litterrandom_afterselection")
            {
                cout << endl << "PHENOTYPE_STRATEGY2 (" << solvervariables[3] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==5 && (solvervariables[3]!="ebv_atselection" && solvervariables[3]!="ebv_afterselection" && solvervariables[3]!="ebv_parents" && solvervariables[1]!="ebv_atselection" && solvervariables[1]!="ebv_afterselection" && solvervariables[1]!="ebv_parents"))
            {
                cout << endl << "PHENOTYPE_STRATEGY2 (" << solvervariables[4] << ") is only needed if doing ebv* based phenotype strategy" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==4)
            {
                logfilestring = logfilestring+"        - Phenotyping strategy parameter for Trait 2\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'"+solvervariables[2]+"'\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                if(solvervariables[3]=="ebv_atselection" || solvervariables[3]=="ebv_afterselection" || solvervariables[3]=="ebv_parents" || solvervariables[1] =="ebv_atselection" || solvervariables[1] =="ebv_afterselection" || solvervariables[1]=="ebv_parents")
                {
                    logfilestring = logfilestring+"             - Portion of ebv distribution animals are phenotyped from: \t\t\t\t'";
                    logfilestring = logfilestring+SimParameters.getSelectionDir()+"' (Default)\n";
                }
                SimParameters.add_MalePropPhenotype_vec(atof(solvervariables[0].c_str()));
                SimParameters.add_MaleWhoPhenotype_vec(solvervariables[1]);
                SimParameters.add_FemalePropPhenotype_vec(atof(solvervariables[2].c_str()));
                SimParameters.add_FemaleWhoPhenotype_vec(solvervariables[3]);
                SimParameters.add_PortionofDistribution_vec(SimParameters.getSelectionDir()); break;
            }
            if(solvervariables.size()==5)
            {
                logfilestring = logfilestring+"        - Phenotyping strategy parameter for Trait 2\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'"+solvervariables[2]+"'\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                logfilestring = logfilestring+"             - Portion of ebv distribution animals are phenotyped from: \t\t\t\t'"+solvervariables[4]+"'\n";
                SimParameters.add_MalePropPhenotype_vec(atof(solvervariables[0].c_str()));
                SimParameters.add_MaleWhoPhenotype_vec(solvervariables[1]);
                SimParameters.add_FemalePropPhenotype_vec(atof(solvervariables[2].c_str()));
                SimParameters.add_FemaleWhoPhenotype_vec(solvervariables[3]);
                SimParameters.add_PortionofDistribution_vec(solvervariables[4]); break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "index_tbv")
            {
                SimParameters.add_MalePropPhenotype_vec(1.0);
                SimParameters.add_MaleWhoPhenotype_vec("pheno_atselection");
                SimParameters.add_FemalePropPhenotype_vec(1.0);
                SimParameters.add_FemaleWhoPhenotype_vec("pheno_atselection");
                SimParameters.add_PortionofDistribution_vec(SimParameters.getSelectionDir());
                logfilestring = logfilestring+"         - Phenotyping strategy parameter for Trait 2\n";
                logfilestring = logfilestring+"             - Proportion of males phenotyped:\t\t\t\t\t\t\t'1.0' (Default)\n";
                logfilestring = logfilestring+"             - Male phenotyping strategy:\t\t\t\t\t\t\t\t'pheno_atselection' (Default)\n";
                logfilestring = logfilestring+"             - Proportion of females phenotyped:\t\t\t\t\t\t\t'1.0' (Default)\n";
                logfilestring = logfilestring+"             - Female phenotyping strategy:\t\t\t\t\t\t\t\t'pheno_atselection' (Default)\n"; break;
            } else {break;}
        }
    }
    /* Loop through number of traits to make sure phenotype strategy makes sense */
    for(int trait = 0; trait < (SimParameters.get_MalePropPhenotype_vec()).size(); trait++)
    {
        if(SimParameters.get_PortionofDistribution_vec()[trait] != "high" && SimParameters.get_PortionofDistribution_vec()[trait] != "low" && SimParameters.get_PortionofDistribution_vec()[trait] != "tails")
        {
            cout <<endl<<"Parameter for deciding which portion of ebv distribution to decide whether an animals is phenotyped is not an option (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MalePropPhenotype_vec())[trait] == 0.0 && (SimParameters.get_FemalePropPhenotype_vec())[trait] == 0.0)
        {
            cout <<endl<<"No Phenotypes are being generated (Trait" << trait << ") Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        /******************************************************************/
        /*****        Doing pheno_* based phenotype strategy         ******/
        /******************************************************************/
        /* if do pheno_atselection strategy has to be a value of 0.0 or 1.0 for a given sex and can't have 0.0 across both sexes */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "pheno_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] > 0.0 && (SimParameters.get_MalePropPhenotype_vec())[trait] < 1.0)
        {
            cout <<endl<<"When doing 'pheno_atselection' proportion has to be 0.0 or 1.0 (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "pheno_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] > 0.0 && (SimParameters.get_FemalePropPhenotype_vec())[trait] < 1.0)
        {
            cout <<endl<<"When doing 'pheno_atselection' proportion has to be 0.0 or 1.0 (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        /* if do pheno_afterselection strategy has to be a value of 0.0 or 1.0 for a given sex and can't have 0.0 across both sexes */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "pheno_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] > 0.0 && (SimParameters.get_MalePropPhenotype_vec())[trait] < 1.0)
        {
            cout <<endl<<"When doing 'pheno_afterselection' proportion has to be 0.0 or 1.0 (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "pheno_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] > 0.0 && (SimParameters.get_FemalePropPhenotype_vec())[trait] < 1.0)
        {
            cout <<endl<<"When doing 'pheno_afterselection' proportion has to be 0.0 or 1.0 (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "pheno_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait" << trait << ")!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "pheno_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait" << trait << ")!"<<endl; exit (EXIT_FAILURE);
        }
        /* if do pheno_parents strategy has to be a value of 0.0 or 1.0 for a given sex and can't have 0.0 across both sexes */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "pheno_parents" && (SimParameters.get_MalePropPhenotype_vec())[trait] > 0.0 && (SimParameters.get_MalePropPhenotype_vec())[trait] < 1.0)
        {
            cout <<endl<<"When doing 'pheno_parents' proportion has to be 0.0 or 1.0 (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "pheno_parents" && (SimParameters.get_FemalePropPhenotype_vec())[trait] > 0.0 && (SimParameters.get_FemalePropPhenotype_vec())[trait] < 1.0)
        {
            cout <<endl<<"When doing 'pheno_parents' proportion has to be 0.0 or 1.0 (Trait" << trait << "). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "pheno_parents" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "pheno_parents" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        /******************************************************************/
        /*****        Doing random_* based phenotype strategy        ******/
        /******************************************************************/
        /* if do random_atselection strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'random_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'random_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'random_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'random_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        /* if do random_afterselection strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        /* if do random_parents strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_parents" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0  (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_parents" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0  (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_parents" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_parents" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'random_afterselection' proportion has to be > 0.0 and < 1.0  (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "random_parents" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "random_parents" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        /******************************************************************/
        /*****        Doing ebv_* based phenotype strategy         ******/
        /******************************************************************/
        /* if do ebv_atselection strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'ebv_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'ebv_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'ebv_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'ebv_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl; exit (EXIT_FAILURE);
        }
        /* if do ebv_afterselection strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        /* if do ebv_parents strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_parents" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_parents" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_parents" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_parents" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'ebv_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "ebv_parents" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "ebv_parents" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        /******************************************************************/
        /*****     Doing litterrandom_* based phenotype strategy     ******/
        /******************************************************************/
        if(SimParameters.getOffspring() == 1 && ((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_atselection" || (SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" || (SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_atselection" || (SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection"))
        {
            cout<<"\n This phenotyping option is only available for litter based populations ,therefore need to use 'litter' based option (Trait"<<trait<<").\n";
            exit (EXIT_FAILURE);
        }
        /* if do litterrandom_atselection strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout<<endl<<"When doing 'litterrandom_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_atselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'litterrandom_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'litterrandom_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_atselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'litterrandom_atselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        /* if do ebv_afterselection strategy has to be a value greater 0.0 and less than 1.0 for a given sex */
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'litterrandom_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" && (SimParameters.get_MalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'litterrandom_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] <= 0.0)
        {
            cout <<endl<<"When doing 'litterrandom_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" && (SimParameters.get_FemalePropPhenotype_vec())[trait] >= 1.0)
        {
            cout <<endl<<"When doing 'litterrandom_afterselection' proportion has to be > 0.0 and < 1.0 (Trait"<<trait<<"). Look at Manual!!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_MaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
        if((SimParameters.get_FemaleWhoPhenotype_vec())[trait] == "litterrandom_afterselection" && SimParameters.getGenfoundsel() == 0)
        {
            cout <<endl<<"Need to start off with at least 1 generation of random selection if progeny doesn't have phenotype (Trait"<<trait<<")!"<<endl;
            exit (EXIT_FAILURE);
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MAXFULLSIB:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            logfilestring=logfilestring+"        - Maximum Number of Full-Sibling taken per family:\t\t\t\t\t\t'" + parm[search] + "'\n";
            SimParameters.Updatemaxmating(atoi(parm[search].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.Updatemaxmating(SimParameters.getOffspring()); stringstream s1; s1 << SimParameters.getmaxmating(); string tempvar = s1.str();
            logfilestring=logfilestring+"        - Maximum Number of Full-Sibling taken per family:\t\t\t\t\t\t'" + tempvar + "'(Default)\n"; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("EBV_METHOD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
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
            if(solvervariables.size() != 1)
            {
                cout << endl << "'EBV_METHOD' option only needs one parameter! Look at user manual!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[0] != "pblup" && solvervariables[0] != "gblup" && solvervariables[0] != "ssgblup" && solvervariables[0] != "rohblup" && solvervariables[0] != "bayes")
            {
                cout<<endl<<"SOLVER_INVERSE first option ("<<solvervariables[0]<<") isn't an option! Check parameter file!"<<endl;exit (EXIT_FAILURE);
            }
            SimParameters.UpdateEBV_Calc(solvervariables[0]);
            logfilestring=logfilestring+"        - EBV estimated by:\t\t\t\t\t\t\t\t\t\t'" + SimParameters.getEBV_Calc() + "'\n";
            break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateEBV_Calc("SKIP");
            logfilestring=logfilestring+"        - EBV are not estimated each generation.\n"; break;
        }
    }
    if((SimParameters.getSelection() == "ebv" || SimParameters.getSelection() == "index_ebv" || SimParameters.getSelection() == "ocs") && SimParameters.getEBV_Calc() == "SKIP")
    {
        cout << endl << "Need to have 'EBV_METHOD:' option if selecting animals based on ebv, index_ebv or using ocs." << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("BLUP_OPTIONS:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getEBV_Calc() == "bayes")
            {
                cout << endl << "Chosen bayesian regression approach. Don't need 'BLUP_OPTIONS' parameter." << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            string kill = "NO";        /* Used to exit out of not in right order or not enough parameters */
            while(kill == "NO")
            {
                if(solvervariables.size() == 3)
                {
                    if(SimParameters.getEBV_Calc() == "pblup")
                    {
                        cout << endl << "Chosen pedigree based relationship matrix only need first two options." << endl; exit (EXIT_FAILURE);
                    }
                    /* First check to see if in right order */
                    if(atoi(solvervariables[0].c_str()) <= 0 || atoi(solvervariables[0].c_str()) > SimParameters.getGener())
                    {
                        cout<<endl<<"BLUP_OPTIONS first parameter ("<<solvervariables[0]<<") isn't feasible! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    if(solvervariables[1] != "direct" && solvervariables[1] != "pcg")
                    {
                        cout<<endl<<"BLUP_OPTIONS second parameter ("<<solvervariables[1]<<") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    if(solvervariables[2] != "cholesky" && solvervariables[2] != "recursion")
                    {
                        cout <<endl<<"BLUP_OPTIONS third parameter ("<<solvervariables[2]<<") isn't an option! Check parameter file!"<<endl;
                        exit (EXIT_FAILURE);
                    }
                    /* if didn't exit declare variables */
                    SimParameters.Updatereferencegenblup(atoi(solvervariables[0].c_str())); SimParameters.UpdateSolver(solvervariables[1]);
                    SimParameters.UpdateGeno_Inverse(solvervariables[2]);
                    logfilestring=logfilestring+"        - Generations until animals are truncated:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                    logfilestring=logfilestring+"        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                    logfilestring=logfilestring+"        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                    kill = "YES";
                }
                if(solvervariables.size() == 2)
                {
                    /* First check to see if in right order */
                    if(atoi(solvervariables[0].c_str()) <= 0 || atoi(solvervariables[0].c_str()) > SimParameters.getGener())
                    {
                        cout<<endl<<"BLUP_OPTIONS first parameter ("<<solvervariables[0]<<") isn't feasible! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    if(solvervariables[1] != "direct" && solvervariables[1] != "pcg")
                    {
                        cout<<endl<<"BLUP_OPTIONS second parameter ("<<solvervariables[1]<<") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    /* if didn't exit declare variables */
                    SimParameters.Updatereferencegenblup(atoi(solvervariables[0].c_str())); SimParameters.UpdateSolver(solvervariables[1]);
                    SimParameters.UpdateGeno_Inverse("cholesky");
                    logfilestring=logfilestring+"        - Generations until animals are truncated:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                    logfilestring=logfilestring+"        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                    if(SimParameters.getEBV_Calc() != "pblup")
                    {
                        logfilestring=logfilestring+"        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'cholesky' (Default).\n";
                    }
                    kill = "YES";
                }
                if(solvervariables.size() == 1)
                {
                    /* First check to see if in right order */
                    if(atoi(solvervariables[0].c_str()) <= 0 || atoi(solvervariables[0].c_str()) > SimParameters.getGener())
                    {
                        cout<<endl<<"BLUP_OPTIONS first parameter ("<<solvervariables[0]<<") isn't feasible! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    SimParameters.Updatereferencegenblup(atoi(solvervariables[0].c_str()));
                    SimParameters.UpdateSolver("pcg"); SimParameters.UpdateGeno_Inverse("cholesky");
                    logfilestring=logfilestring+"        - Generations until animals are truncated:\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                    logfilestring=logfilestring+"        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'pcg' (Default)'\n";
                    if(SimParameters.getEBV_Calc() != "pblup")
                    {
                        logfilestring=logfilestring+"        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'cholesky' (Default).\n";
                    }
                    kill = "YES";
                }
                if(kill == "NO"){cout << endl << "'BLUP_OPTIONS:' variable incorrect check user manual!" << endl; exit (EXIT_FAILURE);}
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getEBV_Calc() != "bayes" && SimParameters.getEBV_Calc() != "SKIP")
            {
                SimParameters.Updatereferencegenblup(SimParameters.getGener());
                SimParameters.UpdateSolver("pcg"); SimParameters.UpdateGeno_Inverse("cholesky");
                stringstream s1; s1 << SimParameters.getreferencegenblup(); string tempvar = s1.str();
                logfilestring=logfilestring+"        - Generations until animals are truncated:\t\t\t\t\t\t\t'" + tempvar + "'\n";
                logfilestring=logfilestring+"        - EBV solved by:\t\t\t\t\t\t\t\t\t\t'" + SimParameters.getSolver() + "' (Default)\n";
                if(SimParameters.getEBV_Calc() != "pblup")
                {
                    logfilestring=logfilestring+"        - Genomic inverse calculated by:\t\t\t\t\t\t\t\t'cholesky' (Default).\n";
                }
            }
            break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("CULLING:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for CULLING!" << endl; exit (EXIT_FAILURE);}
            logfilestring=logfilestring+"        - Culling Criteria:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            logfilestring=logfilestring+"        - Maximum Age Parents can Remain in Population:\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateCulling(solvervariables[0]);
            SimParameters.UpdateMaxAge(atoi(solvervariables[1].c_str())); break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'CULLING:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    if(SimParameters.getCulling() != "random" && SimParameters.getCulling() != "phenotype" && SimParameters.getCulling() != "tbv" && SimParameters.getCulling() != "ebv" && SimParameters.getCulling() != "index_tbv" && SimParameters.getCulling() != "index_ebv")
    {
        cout << endl << "CULLING (" << SimParameters.getCulling()  << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
    }
    /* ensure selection are culling criteria are the same */
    if(SimParameters.getSelection() == "random" && SimParameters.getCulling() != "random")
    {
        cout << endl << "Selection and Culling Criteria are different!!" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelection() == "phenotype" && SimParameters.getCulling() != "phenotype")
    {
        cout << endl << "Selection and Culling Criteria are different!!" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelection() == "tbv" && SimParameters.getCulling() != "tbv")
    {
        cout << endl << "Selection and Culling Criteria are different!!" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelection() == "ebv" && SimParameters.getCulling() != "ebv")
    {
        cout << endl << "Selection and Culling Criteria are different!!" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelection() == "index_tbv" && SimParameters.getCulling() != "index_tbv")
    {
        cout << endl << "Selection and Culling Criteria are different!!" << endl; exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelection() == "index_ebv" && SimParameters.getCulling() != "index_ebv")
    {
        cout << endl << "Selection and Culling Criteria are different!!" << endl; exit (EXIT_FAILURE);
    }
    /**********************************************************/
    /* To generate particular phenotype and genotype scenario */
    /**********************************************************/
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("INTERIM_EBV:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            if(parm[search] != "no" && parm[search] != "before_culling" && parm[search] != "after_culling")
            {
                cout << endl << "This option (" <<parm[search] << ") for the parameter. Look at user manual!" << endl; exit (EXIT_FAILURE);
            }
            SimParameters.UpdateInterim_EBV(parm[search]);
            logfilestring=logfilestring+"        - Interim EBV are Calculated:\t\t\t\t\t\t\t\t\t'"+parm[search]+"'\n"; break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateInterim_EBV("no"); logfilestring=logfilestring+"        - Interim EBV are Calculated:\t\t\t\t\t\t\t\t\t'no' (Default)\n";
            break;
        }
    }
    ///////////////////////////////////////////////////////////////
    ///////////             Mating Parameters           ///////////
    ///////////////////////////////////////////////////////////////
    logfilestring = logfilestring+"    - Mating Scenarios:\n";
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MATING:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(8,"");
            for(int i = 0; i < 8; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            string kill = "NO";        /* Used to exit out of not in right order or not enough parameters */
            while(kill == "NO")
            {
                if(solvervariables.size() > 1 && solvervariables[0]!="random" && solvervariables[0]!="pos_assort" && solvervariables[0]!="neg_assort")
                {
                    logfilestring=logfilestring+"        - Mating Criteria:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                    logfilestring=logfilestring+"        - Mating Algorithm:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                    SimParameters.UpdateMating(solvervariables[0]); SimParameters.UpdateMatingAlg(solvervariables[1]); kill = "YES";
                }
                if(solvervariables.size()==1 && solvervariables[0]!="random" && solvervariables[0]!="pos_assort" && solvervariables[0]!="neg_assort")
                {
                    cout << endl << "Need to specify mating algorithm! Look at Manual." << endl; exit (EXIT_FAILURE);
                }
                if(solvervariables.size()==1 && (solvervariables[0]=="random" || solvervariables[0]=="pos_assort" || solvervariables[0]=="pos_assort"))
                {
                    logfilestring=logfilestring+"        - Mating Criteria:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
                    SimParameters.UpdateMating(solvervariables[0]); SimParameters.UpdateMatingAlg("sslr"); kill = "YES";
                }
                if(kill == "NO"){cout << endl << "'MATING:' variable incorrect check user manual!" << endl; exit (EXIT_FAILURE);}
            }
            break;
        }
        search++; if(search >= parm.size()){cout << endl << "Couldn't find 'MATING:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    /* Check to ensure mating desing is correct */
    if(SimParameters.getMating()!="random" && SimParameters.getMating()!="random5" && SimParameters.getMating()!="random25" && SimParameters.getMating()!="random125" && SimParameters.getMating()!="minPedigree" && SimParameters.getMating()!="minGenomic" && SimParameters.getMating()!="minROH" && SimParameters.getMating()!="pos_assort" && SimParameters.getMating()!="neg_assort" && SimParameters.getMating()!="index" && SimParameters.getMating()!="minGenomic_maf")
    {
        cout << endl << "MATING: first parameter (" << SimParameters.getMating() << ") isn't an option! Check parameter file!" << endl;
        exit (EXIT_FAILURE);
    }
    /* Check optimization function */
    if(SimParameters.getMatingAlg() != "simu_anneal" && SimParameters.getMatingAlg() != "linear_prog" && SimParameters.getMatingAlg() != "sslr" && SimParameters.getMatingAlg() != "gp")
    {
        cout << endl << "MATING: second parameter (" << SimParameters.getMatingAlg() << ") isn't an option! Check parameter file!" << endl;
        exit (EXIT_FAILURE);
    }
    if(SimParameters.getSelection() == "ocs" && SimParameters.getMating()!="random")
    {
        cout << endl << "MATING: first parameter (" << SimParameters.getMating() << ") has to be random with optimum contribution selection." << endl;
        exit (EXIT_FAILURE);
    }
    /* If index mating can only do sslr and gp at the current time */
    if(SimParameters.getMating() == "index" && SimParameters.getMatingAlg() != "sslr" && SimParameters.getMatingAlg() != "gp")
    {
        cout << endl << "MATING: second parameter (" << SimParameters.getMatingAlg() << ") not currently available with index mating" << endl;
        exit (EXIT_FAILURE);
    }
    if(SimParameters.getMating() != "index" && SimParameters.getMatingAlg() == "gp")
    {
        cout <<endl<<"MATING: second parameter ("<<SimParameters.getMatingAlg() << ") not currently available with ";
        cout << SimParameters.getMating() << "!" << endl;
        exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("G_OPTIONS:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getEBV_Calc() != "gblup")
            {
                cout << endl << "Didn't choose 'gblup' ebv calculation method. Don't need 'G_OPTIONS' parameter." << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 2)
            {
                cout << endl << "Incorrect number of parameters for 'G_OPTIONS'. Needs to be 2. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            /* First check to see if in right order */
            if(solvervariables[0] != "VanRaden")
            {
                cout<<endl<<"G_OPTIONS first parameter ("<<solvervariables[0]<<") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[1] != "founder" && solvervariables[1] != "observed")
            {
                cout<<endl<<"G_OPTIONS second parameter ("<<solvervariables[1]<<") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            SimParameters.UpdateConstructG(solvervariables[0]) ; SimParameters.UpdateConstructGFreq(solvervariables[1]);
            logfilestring=logfilestring+"        - Method to construct genomic relationship:\t\t\t\t\t\t\t'"+SimParameters.getConstructG()+"'.\n";
            logfilestring=logfilestring+"        - Frequencies used to construct genomic relationship:\t\t\t\t\t'"+SimParameters.getConstructGFreq()+"'.\n";
            break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateConstructG("VanRaden") ; SimParameters.UpdateConstructGFreq("founder");
            if(SimParameters.getEBV_Calc() == "gblup")
            {
                logfilestring=logfilestring+"        - Method to construct genomic relationship:\t\t\t\t\t\t\t'VanRaden' (Default).\n";
                logfilestring=logfilestring+"        - Frequencies used to construct genomic relationship:\t\t\t\t\t'founder' (Default).\n";
            }
            break;
        }
    }
    search = 0; string tempindexmate;
    while(1)
    {
        size_t fnd = parm[search].find("MATING_INDEX:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getMating() != "index")
            {
                cout << endl << "This option is only available with index mating design" << endl; exit (EXIT_FAILURE);
            }
            tempindexmate = "        - index Mating option chosen:\n";
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(14,"");
            for(int i = 0; i < 14; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            string kill = "NO";        /* Used to exit out of not in right order or not enough parameters */
            while(kill == "NO")
            {
                if(solvervariables.size() != 3 && solvervariables.size() != 5)
                {
                    cout << endl << "Wrong number of parameters for 'MATING_INDEX:' parameter!! Check User Manual." << endl; exit (EXIT_FAILURE);
                }
                if(solvervariables.size() == 3)
                {
                    tempindexmate=tempindexmate+"               - Maxiumum proportion a sire can be mated to population \t\t\t\t\t'"+solvervariables[0]+"'\n";
                    SimParameters.Updatemaxsireprop(atof((solvervariables[0].c_str())));
                    if(solvervariables[1] != SimParameters.getSelection())
                    {
                        cout << endl << "At the current time selection parameter has to equal genetic value in index mating design" << endl;
                        exit (EXIT_FAILURE);
                    }
                    tempindexmate=tempindexmate+"               - Index option chosen comprised of only " + solvervariables[1] + " (1.0)\n";
                    SimParameters.add_Toindexparameters(solvervariables[1]); SimParameters.add_Toindexweights(atof((solvervariables[2].c_str())));
                    //cout << (SimParameters.get_indexparameters()).size() << " " << (SimParameters.get_indexweights()).size() << endl;
                    //cout << (SimParameters.get_indexparameters())[0] << " " << (SimParameters.get_indexweights())[0] << endl;
                    kill = "YES";
                }
                if(solvervariables.size() == 5)
                {
                    tempindexmate=tempindexmate+"               - Maxiumum proportion a sire can be mated to population \t\t\t\t\t'"+solvervariables[0]+"'\n";
                    SimParameters.Updatemaxsireprop(atof((solvervariables[0].c_str())));
                    if(solvervariables[1] != SimParameters.getSelection())
                    {
                        cout << endl << "At the current time selection parameter has to equal genetic value in index mating design" << endl;
                        exit (EXIT_FAILURE);
                    }
                    SimParameters.add_Toindexparameters(solvervariables[1]); SimParameters.add_Toindexweights(atof((solvervariables[2].c_str())));
                    tempindexmate=tempindexmate+"               - Index "+solvervariables[1]+" proportion = "+solvervariables[2]+"\n";
                    if(solvervariables[3] != "pedigree" && solvervariables[3] != "genomic" && solvervariables[3] != "ROH")
                    {
                        cout <<endl<<"'MATING_INDEX:' inbreeding option ("<<solvervariables[3]<<") isn't an option! Check parameter file!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    //cout << (SimParameters.get_indexparameters()).size() << " " << (SimParameters.get_indexweights()).size() << endl;
                    SimParameters.add_Toindexparameters(solvervariables[3]); SimParameters.add_Toindexweights(atof((solvervariables[4].c_str())));
                    tempindexmate=tempindexmate+"               - Index "+solvervariables[3]+" proportion = "+solvervariables[4]+"\n";
                    //cout << (SimParameters.get_indexparameters()).size() << " " << (SimParameters.get_indexweights()).size() << endl;
                    //cout << (SimParameters.get_indexparameters())[1] << " " << (SimParameters.get_indexweights())[1] << endl;
                    kill = "YES";
                }
                if(kill == "NO"){cout << endl << "'MATING_INDEX:' variable incorrect check user manual!" << endl; exit (EXIT_FAILURE);}
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getMating() == "index")
            {
                cout << endl << "'MATING_INDEX:' option has to be included with index mating design" << endl; exit (EXIT_FAILURE);
            }
            SimParameters.Updatemaxsireprop(-5); break;
        }
    }
    if(SimParameters.getMating() == "index"){logfilestring=logfilestring+tempindexmate;}
    search = 0; string tempgenomicmaf;
    while(1)
    {
        size_t fnd = parm[search].find("GENOMIC_MAF:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getMating() != "minGenomic_maf" && SimParameters.getSelection() != "ocs")
            {
                cout << endl << "This option is only available with minGenomic_maf mating design or optimal contribution selection" << endl;
                exit (EXIT_FAILURE);
            }
            tempgenomicmaf = "        - minGenomic_maf Mating option chosen:\n";
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 2)
            {
                cout << endl << "Wrong number of parameters for 'GENOMIC_MAF' option. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            if(atof((solvervariables[0].c_str())) < 0.0 && atof((solvervariables[0].c_str())) > 0.5)
            {
                cout << endl << "MAF out of range for first parameter in 'GENOMIC_MAF' option. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[1] != "above" && solvervariables[1] != "below")
            {
                cout << endl << "Not an option for second parameter in 'GENOMIC_MAF' option. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            SimParameters.Updategenmafcutoff(atof((solvervariables[0].c_str())));
            SimParameters.Updategetgenmafdir(solvervariables[1]);
            if(SimParameters.getgenmafdir() == "above")
            {
                tempgenomicmaf = tempgenomicmaf + "             - Genomic relationship created based on SNP with MAF above " + solvervariables[0] + "\n";
            }
            if(SimParameters.getgenmafdir() == "below")
            {
                tempgenomicmaf = tempgenomicmaf + "             - Genomic relationship created based on SNP with MAF below " + solvervariables[0] + "\n";
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getMating() == "minGenomic_maf")
            {
                cout << endl << "'GENOMIC_MAF:' option has to be included with minGenomic_maf mating design" << endl;
                exit (EXIT_FAILURE);
            }
            SimParameters.Updategenmafcutoff(-5); SimParameters.Updategetgenmafdir(""); break;
        }
    }
    if(SimParameters.getMating() == "minGenomic_maf"){logfilestring=logfilestring+tempgenomicmaf;}
    search = 0; string tempparitymate;
    while(1)
    {
        size_t fnd = parm[search].find("PARITY_MATES_DIST:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getMating() == "index")
            {
                cout << endl << "This option isn't available with index mating design" << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() != 2){cout << endl << "        - Should be two parameters for PARITY_MATES_DIST!" << endl; exit (EXIT_FAILURE);}
            tempparitymate="        - Beta Alpha - distribution of mating pairs by age:\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            tempparitymate=tempparitymate+"        - Beta Beta - distribution of mating pairs by age:\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            SimParameters.UpdateBetaDist_alpha(atof(solvervariables[0].c_str()));
            SimParameters.UpdateBetaDist_beta(atof(solvervariables[1].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateBetaDist_alpha(1.0); SimParameters.UpdateBetaDist_alpha(1.0);
            tempparitymate=tempparitymate+"        - Beta Alpha - distribution of mating pairs by age:\t\t\t\t\t\t'1.0' (Default)\n";
            tempparitymate=tempparitymate+"        - Beta Beta - distribution of mating pairs by age:\t\t\t\t\t\t'1.0' (Default)\n"; break;
        }
    }
    
    
    if(SimParameters.getMating() != "index"){logfilestring=logfilestring+tempparitymate;}
    logfilestring=logfilestring+"    - Output Options:\n";
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("OUTPUT_LD:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); SimParameters.UpdateLDDecay(parm[search]);
            logfilestring=logfilestring+"        - LD Parameters by Generation Created:\t\t\t\t\t\t\t'" + parm[search] + "'\n"; break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateLDDecay("no");
            logfilestring=logfilestring+"        - LD Parameters by Generation Not Created:\t\t\t\t\t\t\t'no' (Default)\n"; break;
        }
    }
    if(SimParameters.getLDDecay() != "yes" && SimParameters.getLDDecay() != "no")
    {
        cout << endl << "'OUTPUT_LD' parameter given is not an option!! Check Manual." << endl; exit (EXIT_FAILURE);
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("GENOTYPES:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(5,"");
            for(int i = 0; i < 5; i++)
            {
                size_t pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 5;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            if(solvervariables.size() > 2)
            {
                cout << endl << "Wrong number of parameters for 'GENOTYPES:' parameter!! Check User Manual." << endl; exit (EXIT_FAILURE);
            }
            SimParameters.UpdateOutputGeno(solvervariables[0]);
            if(SimParameters.getOutputGeno() == "no")
            {
                SimParameters.Updateoutputgeneration(0);
                logfilestring=logfilestring+"        - No genotypes are kept and placed in file.\n";
            }
            if(SimParameters.getOutputGeno() == "yes" && solvervariables.size() == 1)
            {
                SimParameters.Updateoutputgeneration(0);
                logfilestring=logfilestring+"        - All genotypes are kept and placed in file.\n";
            }
            if(SimParameters.getOutputGeno() == "yes" && solvervariables.size() > 1)
            {
                if((atoi(solvervariables[1].c_str())) > SimParameters.getGenfoundsel())
                {
                    cout << endl << "Generation to save genotypes is greater than generations simulation!!" << endl; exit (EXIT_FAILURE);
                }
                logfilestring=logfilestring+ "        - Genotypes after generation " + solvervariables[1] + " are kept and placed in file.\n";
                SimParameters.Updateoutputgeneration(atoi(solvervariables[1].c_str()));
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateOutputGeno("yes"); SimParameters.Updateoutputgeneration(0);
            logfilestring=logfilestring+"        - All genotypes are kept and placed in file. (Default)\n";break;
        }
    }
    search = 0; string rohsummary;
    while(1)
    {
        size_t fnd = parm[search].find("GENOME_ROH:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables((SimParameters.getGener()+5),"");
            for(int i = 0; i < solvervariables.size(); i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            for(int i = 1; i < solvervariables.size(); i++)
            {
                if(atoi(solvervariables[i].c_str()) > SimParameters.getGener())
                {
                    cout << endl << "ROH summary generation greater than generations simulated!! Check Manual." << endl; exit (EXIT_FAILURE);
                }
            }
            SimParameters.Updatemblengthroh(atoi(solvervariables[0].c_str()));
            for(int i = 1; i < solvervariables.size(); i++)
            {
                SimParameters.add_Torohgeneration(atoi(solvervariables[i].c_str()));
            }
            rohsummary = "        - ROH Genome Summary Options:\n";
            rohsummary = rohsummary + "             - ROH Cutoff Length:\t\t\t\t\t\t\t\t\t'" + solvervariables[0] + "'\n";
            rohsummary = rohsummary + "                 - Generations to Save:\n";
            for(int i = 0; i < (SimParameters.get_rohgeneration()).size(); i++)
            {
                stringstream s1; s1 << (SimParameters.get_rohgeneration())[i]; string tempvar = s1.str();
                rohsummary = rohsummary + "                     - Generation " + tempvar + "\n";
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.Updatemblengthroh(-5); break;
        }
    }
    if(SimParameters.getmblengthroh() > 0){logfilestring=logfilestring+rohsummary;}
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("TRAINREFER_STATS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            SimParameters.UpdateOutputTrainReference(parm[search]);
            logfilestring=logfilestring+"        - Summary Statistics based on progeny/parents and previous generations:\t\t\t'" + parm[search] + "'\n";
            break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateOutputTrainReference("no");
            logfilestring=logfilestring+"        - Summary Statistics based on progeny/parents and previous generations:\t\t\t'no' (Default).\n"; break;
        }
    }
    if(SimParameters.getOutputTrainReference() != "yes" && SimParameters.getOutputTrainReference() != "no")
    {
        cout << endl << "'TRAINREFER_STATS:' parameter given is not an option!! Check Manual." << endl; exit (EXIT_FAILURE);
    }
    
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("WINDOWQTLVAR:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            SimParameters.UpdateOutputWindowVariance(parm[search]);
            logfilestring=logfilestring+"        - Summary Statistics on additive and dominance variance across genome::\t\t\t'" + parm[search] + "'\n";
            break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.UpdateOutputWindowVariance("no");
            logfilestring=logfilestring+"        - Summary Statistics on additive and dominance variance across genome::\t\t\t'no' (Default).\n"; break;
        }
    }
    if(SimParameters.getOutputWindowVariance() != "yes" && SimParameters.getOutputWindowVariance() != "no")
    {
        cout << endl << "'WINDOWQTLVAR:' parameter given is not an option!! Check Manual." << endl; exit (EXIT_FAILURE);
    }
    search = 0; string temphaplofinder;
    while(1)
    {
        size_t fnd = parm[search].find("HAPLOFINDER:");
        if(fnd!=std::string::npos)
        {
            temphaplofinder = "    - Haplotype Finder Parameters:\n";
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(8,"");
            for(int i = 0; i < 8; i++)
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
            if(solvervariables.size() != 3)
            {
                cout << endl << "Wrong number of parameters for 'HAPLOFINDER:' parameter!! Check User Manual." << endl; exit (EXIT_FAILURE);
            }
            temphaplofinder=temphaplofinder+"        - Generation start to Identify Unfavorable Haplotypes:\t\t\t\t\t'" + solvervariables[0] + "'\n";
            temphaplofinder=temphaplofinder+"        - Training Generations:\t\t\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
            temphaplofinder=temphaplofinder+"        - Number of Generations Before Re-training:\t\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
            SimParameters.Updatestartgen(atoi(solvervariables[0].c_str()));
            SimParameters.Updatetraingen(atoi(solvervariables[1].c_str()));
            SimParameters.Updateretrain(atoi(solvervariables[2].c_str())); break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.Updatestartgen(-5); SimParameters.Updatetraingen(-5); SimParameters.Updateretrain(-5); break;
        }
    }
    if(SimParameters.getstartgen() > 0){logfilestring=logfilestring+temphaplofinder;}
    search = 0; string tempocs;
    while(1)
    {
        size_t fnd = parm[search].find("OCS_PARAMETERS:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getSelection() != "ocs")
            {
                cout << endl << "Need to to do optimal contribution selection. Change 'SELECTION:' parameter!" << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(7,"");
            for(int i = 0; i < 7; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 5 && solvervariables.size() != 6)
            {
                cout << endl << "'OCS_PARAMETERS:' wrong number of parameters. Look at manual." << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[0] != "pedigree" && solvervariables[0] != "genomic" && solvervariables[0] != "ROH" && solvervariables[0] != "genomicmaf")
            {
                cout << endl << "OCS Relationship (" << solvervariables[0] << ") isn't an option! Check parameter file!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[1] != "penalty" && solvervariables[1] != "constraint")
            {
                cout << endl << "Wrong optmization parameter for ocs." << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[1] == "penalty" && solvervariables.size() != 6)
            {
                cout << endl << "Wrong number of parameters given for ocs based on penalty. Should be 6!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[1] == "constraint" && solvervariables.size() != 5)
            {
                cout << endl << "Wrong number of parameters given for ocs based on constraint. Should be 5!" << endl; exit (EXIT_FAILURE);
            }
            tempocs = "    - Optimum Contribution Selection Parameters\n";
            if(solvervariables[1] == "penalty" && solvervariables.size() == 6)
            {
                tempocs = tempocs+"        - Optimum Contribution relationship matrix:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                tempocs = tempocs+"        - Optimum Contribution Optimization Function:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                tempocs = tempocs+"        - Weight on Genetic Merit:\t\t\t\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                tempocs = tempocs+"        - Penalty on Genetic Merit:\t\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                tempocs = tempocs+"        - EVA number of generations:\t\t\t\t\t\t\t\t\t'" + solvervariables[4] + "'\n";
                tempocs = tempocs+"        - EVA population size:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[5] + "'\n";
                SimParameters.Updateocsrelat(solvervariables[0]); SimParameters.Updateocs_optimize(solvervariables[1]);
                SimParameters.Updateocs_w_merit(atof(solvervariables[2].c_str())); SimParameters.Updateocs_w_rel(atof(solvervariables[3].c_str()));
                SimParameters.UpdatenEVAgen(atof(solvervariables[4].c_str())); SimParameters.UpdatenEVApop(atof(solvervariables[5].c_str())); break;
            }
            if(solvervariables[1] == "constraint" && solvervariables.size() == 5)
            {
                tempocs = tempocs+"        - Optimum Contribution relationship matrix:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n";
                tempocs = tempocs+"        - Optimum Contribution Optimization Function:\t\t\t\t\t\t\t'" + solvervariables[1] + "'\n";
                tempocs = tempocs+"        - Inbreeding constrained to:\t\t\t\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                tempocs = tempocs+"        - EVA number of generations:\t\t\t\t\t\t\t\t\t'" + solvervariables[3] + "'\n";
                tempocs = tempocs+"        - EVA population size:\t\t\t\t\t\t\t\t\t\t'" + solvervariables[4] + "'\n";
                SimParameters.Updateocsrelat(solvervariables[0]); SimParameters.Updateocs_optimize(solvervariables[1]);
                SimParameters.Updateocs_w_merit(-5); SimParameters.Updateocs_w_rel(atof(solvervariables[2].c_str()));
                SimParameters.UpdatenEVAgen(atof(solvervariables[3].c_str())); SimParameters.UpdatenEVApop(atof(solvervariables[4].c_str())); break;
            }
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getSelection() == "ocs")
            {
                cout << endl << "'OCS_PARAMETERS:' option has to be included with ocs selection" << endl; exit (EXIT_FAILURE);
            }
            SimParameters.Updateocsrelat(""); SimParameters.Updateocs_optimize(""); SimParameters.Updateocs_w_merit(-5.0);
            SimParameters.Updateocs_w_rel(-99.0); SimParameters.UpdatenEVAgen(-5.0); SimParameters.UpdatenEVApop(-5.0); break;
        }
    }
    if(SimParameters.getSelection() == "ocs"){logfilestring=logfilestring+tempocs;}
    if(SimParameters.getocsrelat() != "genomicmaf" && SimParameters.getgenmafcutoff() != -5 && SimParameters.getgenmafdir() != "")
    {
        cout << endl << "'GENOMIC_MAF:' option needs to be removed since ocs based on genomicmaf is not an option!!" << endl;
        exit (EXIT_FAILURE);
    }
    if(SimParameters.getocsrelat() == "genomicmaf" && SimParameters.getgenmafcutoff() == -5 && SimParameters.getgenmafdir() == "")
    {
        cout << endl << "'GENOMIC_MAF:' option has to be included with ocs based on genomicmaf!!" << endl;
        exit (EXIT_FAILURE);
    }
    if(SimParameters.getocsrelat() == "genomicmaf")
    {
        if(SimParameters.getgenmafdir() == "above")
        {
            stringstream s1; s1 << SimParameters.getgenmafcutoff(); string tempvar = s1.str();
            logfilestring=logfilestring+"        - Genomic relationship created based on SNP with MAF above "+tempvar+ "\n";
        }
        if(SimParameters.getgenmafdir() == "below")
        {
            stringstream s1; s1 << SimParameters.getgenmafcutoff(); string tempvar = s1.str();
            logfilestring=logfilestring+"        - Genomic relationship created based on SNP with MAF below "+tempvar+ "\n";
        }
    }
    search = 0; string tempbayesoptions;
    while(1)
    {
        size_t fnd = parm[search].find("BAYESOPTIONS:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getEBV_Calc() != "bayes")
            {
                cout << endl << "This option is only available with bayes ebv selection design" << endl;
                exit (EXIT_FAILURE);
            }
            tempbayesoptions = "    - Bayes EBV option:\n";
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(10,"");
            for(int i = 0; i < 10; i++)
            {
                pos = parm[search].find(" ",0);
                solvervariables[i] = parm[search].substr(0,pos);
                solvervariables[i].erase(remove(solvervariables[i].begin(), solvervariables[i].end(), ' '),solvervariables[i].end());
                if(pos != std::string::npos){parm[search].erase(0, pos + 1);}
                if(pos == std::string::npos){parm[search].clear(); i = 10;}
            }
            int start = 0;
            while(start < solvervariables.size())
            {
                if(solvervariables[start] == ""){solvervariables.erase(solvervariables.begin()+start);}
                if(solvervariables[start] != ""){start++;}
            }
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables[0]!="BayesA" && solvervariables[0]!="BayesB" && solvervariables[0]!="BayesC" && solvervariables[0]!="BayesRidgeRegression")
            {
                cout<<endl<<"'"<<solvervariables[0]<<"' is not a Bayes Method option. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[0] == "BayesA" && (solvervariables.size() != 6 && solvervariables.size() != 7))
            {
                cout<<endl<<"Bayes A requires 6 or 7 variables to be declared. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[0] == "BayesB" && (solvervariables.size() != 8 && solvervariables.size() != 9))
            {
                cout<<endl<<"Bayes B requires 8 or 9 variables to be declared. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[0] == "BayesRidgeRegression" && (solvervariables.size() != 6 && solvervariables.size() != 7))
            {
                cout<<endl<<"BayesRidgeRegression requires 6 or 7 variables to be declared. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            if(solvervariables[0] == "BayesC" && (solvervariables.size() != 8 && solvervariables.size() != 9))
            {
                cout<<endl<<"Bayes C requires 8 or 9 variables to be declared. Look at user manual!!" << endl; exit (EXIT_FAILURE);
            }
            /* parse out into parameters */
            if(solvervariables[0] == "BayesA" && (solvervariables.size() == 6 || solvervariables.size() == 7))
            {
                SimParameters.Updatemethod(solvervariables[0]);
                SimParameters.Updatenumiter(atoi(solvervariables[1].c_str()));
                SimParameters.Updateburnin(atoi(solvervariables[2].c_str()));
                SimParameters.Updatenumiterstat(atoi(solvervariables[3].c_str()));
                SimParameters.Updateburninstat(atoi(solvervariables[4].c_str()));
                SimParameters.Updatethin(atoi(solvervariables[5].c_str()));
                SimParameters.Updatepie_f("NA");
                SimParameters.Updateinitpi(1.0);
                if(solvervariables.size() == 6){SimParameters.Updatereferencegenerations(SimParameters.getGener());}
                if(solvervariables.size() == 7){SimParameters.Updatereferencegenerations(atoi(solvervariables[6].c_str()));}
                tempbayesoptions = tempbayesoptions+"        - Bayesian Method:\t\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Initial:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Initial:\t\t\t\t\t\t\t\t\t'"+solvervariables[2]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Stationay:\t\t\t\t\t\t\t\t'"+solvervariables[3]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Stationay:\t\t\t\t\t\t\t\t\t'"+solvervariables[4]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Thinning Rate:\t\t\t\t\t\t\t\t\t'"+solvervariables[5]+"'.\n";
                stringstream referencesize; referencesize << SimParameters.getreferencegenerations(); string refsize = referencesize.str();
                tempbayesoptions = tempbayesoptions+"        - Generations in Reference:\t\t\t\t\t\t\t\t'"+refsize+"'.\n";
            }
            if(solvervariables[0] == "BayesB" && (solvervariables.size() == 8 || solvervariables.size() == 9))
            {
                SimParameters.Updatemethod(solvervariables[0]);
                SimParameters.Updatenumiter(atoi(solvervariables[1].c_str()));
                SimParameters.Updateburnin(atoi(solvervariables[2].c_str()));
                SimParameters.Updatenumiterstat(atoi(solvervariables[3].c_str()));
                SimParameters.Updateburninstat(atoi(solvervariables[4].c_str()));
                SimParameters.Updatethin(atoi(solvervariables[5].c_str()));
                SimParameters.Updatepie_f(solvervariables[6]);
                if(SimParameters.getpie_f() != "fix" && SimParameters.getpie_f() != "estimate")
                {
                    cout << endl <<"Incorrect option for 7th parameter (" << SimParameters.getpie_f() << "). Look at user manual!" << endl;
                    exit (EXIT_FAILURE);
                }
                SimParameters.Updateinitpi(atof((solvervariables[7].c_str())));
                if(solvervariables.size() == 8){SimParameters.Updatereferencegenerations(SimParameters.getGener());}
                if(solvervariables.size() == 9){SimParameters.Updatereferencegenerations(atoi(solvervariables[8].c_str()));}
                tempbayesoptions = tempbayesoptions+"        - Bayesian Method:\t\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Initial:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Initial:\t\t\t\t\t\t\t\t\t'"+solvervariables[2]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Stationay:\t\t\t\t\t\t\t\t'"+solvervariables[3]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Stationay:\t\t\t\t\t\t\t\t\t'"+solvervariables[4]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Thinning Rate:\t\t\t\t\t\t\t\t\t'"+solvervariables[5]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - Fix or estimate pie:\t\t\t\t\t\t\t\t\t'"+solvervariables[6]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - Pie value:\t\t\t\t\t\t\t\t\t\t'"+solvervariables[7]+"'.\n";
                stringstream referencesize; referencesize << SimParameters.getreferencegenerations(); string refsize = referencesize.str();
                tempbayesoptions = tempbayesoptions+"        - Generations in Reference:\t\t\t\t\t\t\t\t'"+refsize+"'.\n";
            }
            /* parse out into parameters */
            if(solvervariables[0] == "BayesRidgeRegression" && (solvervariables.size() == 6 || solvervariables.size() == 7))
            {
                SimParameters.Updatemethod(solvervariables[0]);
                SimParameters.Updatenumiter(atoi(solvervariables[1].c_str()));
                SimParameters.Updateburnin(atoi(solvervariables[2].c_str()));
                SimParameters.Updatenumiterstat(atoi(solvervariables[3].c_str()));
                SimParameters.Updateburninstat(atoi(solvervariables[4].c_str()));
                SimParameters.Updatethin(atoi(solvervariables[5].c_str()));
                SimParameters.Updatepie_f("NA");
                SimParameters.Updateinitpi(1.0);
                if(solvervariables.size() == 6){SimParameters.Updatereferencegenerations(SimParameters.getGener());}
                if(solvervariables.size() == 7){SimParameters.Updatereferencegenerations(atoi(solvervariables[6].c_str()));}
                tempbayesoptions = tempbayesoptions+"        - Bayesian Method:\t\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Initial:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Initial:\t\t\t\t\t\t\t\t\t'"+solvervariables[2]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Stationay:\t\t\t\t\t\t\t\t'"+solvervariables[3]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Stationay:\t\t\t\t\t\t\t\t\t'"+solvervariables[4]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Thinning Rate:\t\t\t\t\t\t\t\t\t'"+solvervariables[5]+"'.\n";
                stringstream referencesize; referencesize << SimParameters.getreferencegenerations(); string refsize = referencesize.str();
                tempbayesoptions = tempbayesoptions+"        - Generations in Reference:\t\t\t\t\t\t\t\t'"+refsize+"'.\n";
            }
            if(solvervariables[0] == "BayesC" && (solvervariables.size() == 8 || solvervariables.size() == 9))
            {
                SimParameters.Updatemethod(solvervariables[0]);
                SimParameters.Updatenumiter(atoi(solvervariables[1].c_str()));
                SimParameters.Updateburnin(atoi(solvervariables[2].c_str()));
                SimParameters.Updatenumiterstat(atoi(solvervariables[3].c_str()));
                SimParameters.Updateburninstat(atoi(solvervariables[4].c_str()));
                SimParameters.Updatethin(atoi(solvervariables[5].c_str()));
                SimParameters.Updatepie_f(solvervariables[6]);
                if(SimParameters.getpie_f() != "fix" && SimParameters.getpie_f() != "estimate")
                {
                    cout << endl <<"Incorrect option for 7th parameter (" << SimParameters.getpie_f() << "). Look at user manual!" << endl;
                    exit (EXIT_FAILURE);
                }
                SimParameters.Updateinitpi(atof((solvervariables[7].c_str())));
                if(solvervariables.size() == 8){SimParameters.Updatereferencegenerations(SimParameters.getGener());}
                if(solvervariables.size() == 9){SimParameters.Updatereferencegenerations(atoi(solvervariables[8].c_str()));}
                tempbayesoptions = tempbayesoptions+"        - Bayesian Method:\t\t\t\t\t\t\t\t\t'"+solvervariables[0]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Initial:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Initial:\t\t\t\t\t\t\t\t\t'"+solvervariables[2]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Iterations Stationay:\t\t\t\t\t\t\t\t'"+solvervariables[3]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Burn-in Stationay:\t\t\t\t\t\t\t\t\t'"+solvervariables[4]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - MCMC Thinning Rate:\t\t\t\t\t\t\t\t\t'"+solvervariables[5]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - Fix or estimate pie:\t\t\t\t\t\t\t\t\t'"+solvervariables[6]+"'.\n";
                tempbayesoptions = tempbayesoptions+"        - Pie value:\t\t\t\t\t\t\t\t\t\t'"+solvervariables[7]+"'.\n";
                stringstream referencesize; referencesize << SimParameters.getreferencegenerations(); string refsize = referencesize.str();
                tempbayesoptions = tempbayesoptions+"        - Generations in Reference:\t\t\t\t\t\t\t\t'"+refsize+"'.\n";
            }
            break;
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getEBV_Calc() == "bayes")
            {
                cout << endl << "'BAYESOPTIONS:' option has to be included with bayes ebv selection design" << endl;
                exit (EXIT_FAILURE);
            }
            SimParameters.Updatemethod("NA"); SimParameters.Updatenumiter(-5); SimParameters.Updateburnin(-5);
            SimParameters.Updateburnin(-5); SimParameters.Updatenumiterstat(-5); SimParameters.Updatethin(-5);
            SimParameters.Updatepie_f("NA"); SimParameters.Updateinitpi(-5); break;
        }
    }
    if(SimParameters.getEBV_Calc() == "bayes"){logfilestring=logfilestring+tempbayesoptions;}
    /******************************************/
    /******************************************/
    /*      Genotyping Strategy               */
    /******************************************/
    /******************************************/
    /****************************************************************************************************************************************/
    /****************************************************************************************************************************************/
    /** Current Options:                                                                                                                    */
    /** - 'parents': Only genotype selected parents.                                                                                        */
    /** - 'offspring': Only genotype all offspring.                                                                                         */
    /** - 'parents_offspring': Genotype all parents and offspring.                                                                          */
    /** - 'random': randomly genotype a certain percentage of progeny across litters if more than one progeny.                              */
    /** - 'parents_random': genotype parents and randomly genotype a certain percentage of progeny across litters if more than one progeny. */
    /** - 'ebv_parentavg': genotype progeny based on parent average (i.e. progeny = 1/2(EBV_sire + EBV_dam). Only if progeny = 1.           */
    /** - 'parents_ebv_parentavg': genotype parents and progeny based on parent average. Only if progeny = 1.                               */
    /** - 'litter_random': randomly genotype progeny within each litter. Only if progeny > 1.                                               */
    /** - 'litter_parents_random': genotype parents and randomly genotype progeny within each litter. Only if progeny > 1.                  */
    /** - 'ebv': genotype progeny based on breeding values prior to add genotype information on progeny. (i.e. only pedigree based)         */
    /** - 'parent_ebv': genotype progeny based on breeding values prior to add genotype information on progeny.                             */
    /****************************************************************************************************************************************/
    /****************************************************************************************************************************************/
    search = 0; string tempgenostrat;
    while(1)
    {
        size_t fnd = parm[search].find("GENOTYPE_STRATEGY:");
        if(fnd!=std::string::npos)
        {
            if((SimParameters.getSelection() != "ebv" && SimParameters.getSelection() != "index_ebv") || SimParameters.getEBV_Calc() != "ssgblup")
            {
                cout << endl << "Need to do 'ebv' selection with a 'ssgblup' prediction option for this option. Change parameter";
                cout << " file accordingly!" << endl; exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(7,"");
            for(int i = 0; i < 7; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            if(solvervariables.size() != 5 && solvervariables.size() != 6)
            {
                cout<<endl<<"'GENOTYPE_STRATEGY:' wrong number of parameters. Look at manual."<<endl; exit(EXIT_FAILURE);
            }
            if(atoi(solvervariables[0].c_str()) < 0 || atoi(solvervariables[0].c_str()) > SimParameters.getGener())
            {
                cout << endl << "GENOTYPE_STRATEGY (" << solvervariables[0] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            /* needs to be greater than number of random generations of founder selection + 2 in order to make G and A compatible */
            if(atoi(solvervariables[0].c_str()) < (SimParameters.getGenfoundsel() +2))
            {
                cout << endl << "In order to make G and A more compatible need to add more than generations of pedigree-based selection!" << endl;
                exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[1].c_str()) < 0.0 || atof(solvervariables[1].c_str()) > 1.0)
            {
                cout << endl << "GENOTYPE_STRATEGY (" << solvervariables[1] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[3].c_str()) < 0.0 || atof(solvervariables[3].c_str()) > 1.0)
            {
                cout << endl << "GENOTYPE_STRATEGY (" << solvervariables[3] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[2] != "parents" && solvervariables[2] != "offspring" && solvervariables[2] != "parents_offspring" && solvervariables[2] != "random" && solvervariables[2] != "parents_random" && solvervariables[2] != "litter_random" && solvervariables[2] != "litter_parents_random" && solvervariables[2] != "ebv" && solvervariables[2] != "parents_ebv")
            {
                cout << endl << "GENOTYPE_STRATEGY (" << solvervariables[2] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables[4] != "parents" && solvervariables[4] != "offspring" && solvervariables[4] != "parents_offspring" && solvervariables[4] != "random" && solvervariables[4] != "parents_random" && solvervariables[4] != "litter_random" && solvervariables[4] != "litter_parents_random" && solvervariables[4] != "ebv" && solvervariables[4] != "parents_ebv")
            {
                cout << endl << "GENOTYPE_STRATEGY (" << solvervariables[4] << ") isn't an option! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(SimParameters.getOffspring() > 1 && (solvervariables[2] == "litter_random" || solvervariables[2] == "litter_parents_random"))
            {
                double temp = int(double(atof(solvervariables[1].c_str()) * SimParameters.getOffspring())+0.5) / double(SimParameters.getOffspring());
                stringstream s1; s1 << temp; solvervariables[1] = s1.str();
                
            }
            if(SimParameters.getOffspring() > 1 && (solvervariables[4] == "litter_random" || solvervariables[4] == "litter_parents_random"))
            {
                double temp = int(double(atof(solvervariables[3].c_str()) * SimParameters.getOffspring())+0.5) / double(SimParameters.getOffspring());
                stringstream s1; s1 << temp; solvervariables[3] = s1.str();
            }
            if(solvervariables.size()==6 && solvervariables[2]!="parents_ebv" && solvervariables[2]!="ebv" && solvervariables[4]!="parents_ebv" && solvervariables[4]!="ebv")
            {
                cout << endl << "GENOTYPE_STRATEGY (" << solvervariables[5] << ") is only needed if doing ebv* based genotype strategy" << endl;
                exit (EXIT_FAILURE);
            }
            if(solvervariables.size()==5)
            {
                tempgenostrat = "    - Genotyping strategy parameters\n";
                tempgenostrat = tempgenostrat+"        - Generation at which animals will start to be genotyped:\t\t\t\t\t'"+solvervariables[0]+"'\n";
                tempgenostrat = tempgenostrat+"        - Proportion of males genotyped:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                tempgenostrat = tempgenostrat+"        - Male genotyping strategy:\t\t\t\t\t\t\t\t\t'" + solvervariables[2] + "'\n";
                tempgenostrat = tempgenostrat+"        - Proportion of females genotyped:\t\t\t\t\t\t\t\t'"+solvervariables[3]+"'\n";
                tempgenostrat = tempgenostrat+"        - Female genotyping strategy:\t\t\t\t\t\t\t\t\t'" + solvervariables[4] + "'\n";
                if(solvervariables[2]=="parents_ebv" || solvervariables[2]=="ebv" || solvervariables[4]=="parents_ebv" || solvervariables[4]=="ebv")
                {
                    tempgenostrat = tempgenostrat+"        - Portion of ebv distribution animals are genotyped from: \t\t\t\t\t'";
                    tempgenostrat = tempgenostrat+SimParameters.getSelectionDir()+"' (Default)\n";
                }
                SimParameters.UpdateGenoGeneration(atoi(solvervariables[0].c_str()));
                SimParameters.UpdateMalePropGenotype(atof(solvervariables[1].c_str()));
                SimParameters.UpdateMaleWhoGenotype(solvervariables[2]);
                SimParameters.UpdateFemalePropGenotype(atof(solvervariables[3].c_str()));
                SimParameters.UpdateFemaleWhoGenotype(solvervariables[4]);
                SimParameters.UpdateGenotypePortionofDistribution(SimParameters.getSelectionDir()); break;
            }
            if(solvervariables.size()==6)
            {
                tempgenostrat = "    - Genotyping strategy parameters\n";
                tempgenostrat = tempgenostrat+"        - Generation at which animals will start to be genotyped:\t\t\t\t\t'"+solvervariables[0]+"'\n";
                tempgenostrat = tempgenostrat+"        - Proportion of males genotyped:\t\t\t\t\t\t\t\t'"+solvervariables[1]+"'\n";
                tempgenostrat = tempgenostrat+"        - Male genotyping strategy:\t\t\t\t\t\t\t\t\t'" + solvervariables[2]+"'\n";
                tempgenostrat = tempgenostrat+"        - Proportion of females genotyped:\t\t\t\t\t\t\t\t'"+solvervariables[3]+"'\n";
                tempgenostrat = tempgenostrat+"        - Female genotyping strategy:\t\t\t\t\t\t\t\t\t'" + solvervariables[4]+"'\n";
                tempgenostrat = tempgenostrat+"        - Portion of ebv distribution animals are genotyped from: \t\t\t\t\t'"+solvervariables[5]+"'\n";
                SimParameters.UpdateGenoGeneration(atoi(solvervariables[0].c_str()));
                SimParameters.UpdateMalePropGenotype(atof(solvervariables[1].c_str()));
                SimParameters.UpdateMaleWhoGenotype(solvervariables[2]);
                SimParameters.UpdateFemalePropGenotype(atof(solvervariables[3].c_str()));
                SimParameters.UpdateFemaleWhoGenotype(solvervariables[4]);
                SimParameters.UpdateGenotypePortionofDistribution(solvervariables[5]); break;
            }
                
        }
        search++;
        if(search >= parm.size())
        {
            if(SimParameters.getSelection() == "ebv" && SimParameters.getEBV_Calc() == "ssgblup")
            {
                cout << endl << "You are doing 'ebv' selection with a 'ssgblup' prediction option so you need to include the ";
                cout << "'GENOTYPE_STRATEGY:' parameter option." << endl; exit (EXIT_FAILURE);
            }
            SimParameters.UpdateGenoGeneration(-99);
            SimParameters.UpdateMalePropGenotype(0.0);
            SimParameters.UpdateMaleWhoGenotype("");
            SimParameters.UpdateFemalePropGenotype(0.0);
            SimParameters.UpdateFemaleWhoGenotype("");
            SimParameters.UpdateGenotypePortionofDistribution(""); break;
        }
    }
    if(SimParameters.getMaleWhoGenotype() != ""){logfilestring=logfilestring+tempgenostrat;}
    /* Check to see if genotype portion is correct */
    if(SimParameters.getGenotypePortionofDistribution() != "high" && SimParameters.getGenotypePortionofDistribution() != "low" && SimParameters.getGenotypePortionofDistribution() != "tails" && SimParameters.getGenotypePortionofDistribution() != "")
    {
        cout <<endl<<"Parameter for deciding which portion of ebv distribution to decide if an animal is genotyped. Look at Manual!!"<<endl;
        exit (EXIT_FAILURE);
    }
    /* Genotype parents only */
    if(SimParameters.getMaleWhoGenotype() == "parents" && SimParameters.getMalePropGenotype() != 1.0)
    {
        cout << "\n When doing 'parent' option for male genotyping need male proportion genotyped to be '1.0'.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "parents" && SimParameters.getFemalePropGenotype() != 1.0)
    {
        cout << "\n When doing 'parent' option for female genotyping need female proportion genotyped to be '1.0'.\n"; exit (EXIT_FAILURE);
    }
    /* Genotype offspring only */
    if(SimParameters.getMaleWhoGenotype() == "offspring" && SimParameters.getMalePropGenotype() != 1.0)
    {
        cout << "\n When doing 'offspring' option for male genotyping need male proportion genotyped to be '1.0'.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "offspring" && SimParameters.getFemalePropGenotype() != 1.0)
    {
        cout << "\n When doing 'offspring' option for female genotyping need female proportion genotyped to be '1.0'.\n"; exit (EXIT_FAILURE);
    }
    /* Genotype parents and offspring only */
    if(SimParameters.getMaleWhoGenotype() == "parents_offspring" && SimParameters.getMalePropGenotype() != 1.0)
    {
        cout << "\n When doing 'parents_offspring' option for male genotyping need male proportion genotyped to be '1.0'.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "parents_offspring" && SimParameters.getFemalePropGenotype() != 1.0)
    {
        cout << "\n When doing 'parents_offspring' option for female genotyping need female proportion genotyped to be '1.0'.\n"; exit (EXIT_FAILURE);
    }
    /* Genotype offspring randomly only */
    if(SimParameters.getMaleWhoGenotype() == "random" && SimParameters.getMalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all male selection candidates, therefore need to use 'offspring' option.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "random" && SimParameters.getFemalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all female selection candidates, therefore need to use 'offspring' option.\n"; exit (EXIT_FAILURE);
    }
    /* Genotype all parents and offspring randomly only */
    if(SimParameters.getMaleWhoGenotype() == "parents_random" && SimParameters.getMalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all male selection candidates, therefore need to use 'parents_offspring' option.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "parents_random" && SimParameters.getFemalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all female selection candidates, therefore need to use 'parents_offspring' option.\n"; exit (EXIT_FAILURE);
    }
    /* Genotype offspring based on ebv; generate preliminary blup first */
    if(SimParameters.getMaleWhoGenotype() == "ebv" && SimParameters.getMalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all male selection candidates, therefore need to use 'offspring' option.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "ebv" && SimParameters.getFemalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all female selection candidates, therefore need to use 'offspring' option.\n"; exit (EXIT_FAILURE);
    }
    /* Genotype all parents and offspring based on ebv; generate preliminary blup first */
    if(SimParameters.getMaleWhoGenotype() == "parents_ebv" && SimParameters.getMalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all male selection candidates, therefore need to use 'parents_offspring' option.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "parents_ebv" && SimParameters.getFemalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all female selection candidates, therefore need to use 'parents_offspring' option.\n"; exit (EXIT_FAILURE);
    }
    /**************************************************************************/
    /* can only do litter_random for offspring or parent when offspring is greater than one */
    /**************************************************************************/
    if(SimParameters.getOffspring() == 1 && (SimParameters.getMaleWhoGenotype() == "litter_random" || SimParameters.getFemaleWhoGenotype() == "litter_random" || SimParameters.getMaleWhoGenotype() == "litter_parents_random" || SimParameters.getFemaleWhoGenotype() == "litter_parents_random"))
    {
        cout << "\n This genotyping option is only available for litter based populations ,therefore need to use 'litter' based option.\n";
        exit (EXIT_FAILURE);
    }
    /* Genotype all parents and offspring based on parentavg */
    if(SimParameters.getMaleWhoGenotype() == "litter_random" && SimParameters.getMalePropGenotype() == 0.0)
    {
        cout << "\n Genotyping no male selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getMaleWhoGenotype() == "litter_random" && SimParameters.getMalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all male selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "litter_random" && SimParameters.getFemalePropGenotype() == 0.0)
    {
        cout << "\n Genotyping no female selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "litter_random" && SimParameters.getFemalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all female selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getMaleWhoGenotype() == "litter_parents_random" && SimParameters.getMalePropGenotype() == 0.0)
    {
        cout << "\n Genotyping no male selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getMaleWhoGenotype() == "litter_parents_random" && SimParameters.getMalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all male selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "litter_parents_random" && SimParameters.getFemalePropGenotype() == 0.0)
    {
        cout << "\n Genotyping no female selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    if(SimParameters.getFemaleWhoGenotype() == "litter_parents_random" && SimParameters.getFemalePropGenotype() == 1.0)
    {
        cout << "\n Genotyping all female selection candidates, alter parameter file.\n"; exit (EXIT_FAILURE);
    }
    /************************/
    /** Blending A22 and G **/
    /************************/
    search = 0; string tempblend;
    while(1)
    {
        size_t fnd = parm[search].find("BLENDING_GA22:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getEBV_Calc() != "ssgblup")
            {
                cout << endl << "Need to do selection with a 'ssgblup' prediction option for this option. Change parameter file accordingly!" << endl;
                exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(7,"");
            for(int i = 0; i < 7; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            /* needs to be greater than number of random generations of founder selection + 2 in order to make G and A compatible */
            if(solvervariables.size() != 2)
            {
                cout << endl << "The 'BLENDING_GA22' parameter requires two values!! Check parameter file!" << endl; exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[0].c_str()) < 0.0 || atof(solvervariables[0].c_str()) > 1.0)
            {
                cout << endl << "BLENDING_GA22 (" << solvervariables[0] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if(atof(solvervariables[1].c_str()) < 0.0 || atof(solvervariables[1].c_str()) > 1.0)
            {
                cout << endl << "BLENDING_GA22 (" << solvervariables[1] << ") isn't possible! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            if((atof(solvervariables[0].c_str()) + atof(solvervariables[1].c_str())) != 1.0)
            {
                cout << endl << "BLENDING_GA22 parameters do not sum to 1.0! Check parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            SimParameters.add_Toblending_G_A22(atof(solvervariables[0].c_str()));
            SimParameters.add_Toblending_G_A22(atof(solvervariables[1].c_str()));
            tempblend = "    - Genomic matrix blending factors:\t\t\t\t\t\t\t\t'G("+solvervariables[0]+") + A22("+solvervariables[1]+")'\n"; break;
        }
        search++;
        if(search >= parm.size())
        {
            SimParameters.add_Toblending_G_A22(0.95); SimParameters.add_Toblending_G_A22(0.05);
            tempblend = "    - Genomic matrix blending factors:\t\t\t\t\t\t\t\t'G(0.95) + A22(0.05)' (Default)\n"; break;
        }
    }
    if(SimParameters.getEBV_Calc() == "ssgblup"){logfilestring=logfilestring+tempblend;}
    /****************/
    /** Imputation **/
    /****************/
    search = 0; string tempimpute;
    while(1)
    {
        size_t fnd = parm[search].find("IMPUTATION:");
        if(fnd!=std::string::npos)
        {
            if(SimParameters.getEBV_Calc() != "ssgblup")
            {
                cout << endl << "Need to do selection with a 'ssgblup' prediction option for this option. Change parameter file!" << endl;
                exit (EXIT_FAILURE);
            }
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 1);
            vector < string > solvervariables(7,"");
            for(int i = 0; i < 7; i++)
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
            //for(int i = 0; i < solvervariables.size(); i++){cout << solvervariables[i] << endl;}
            /* needs to be greater than number of random generations of founder selection + 2 in order to make G and A compatible */
            if(solvervariables.size() != 1)
            {
                cout << endl << "The 'IMPUTATION' parameter requires only one value!! Check parameter file!" << endl; exit (EXIT_FAILURE);
            }
            SimParameters.UpdateImputationFile(solvervariables[0]);
            tempimpute = "    - Perform Imputation using the following script:\t\t\t\t\t\t\t'"+solvervariables[0]+"'\n"; break;
        }
        search++;
        if(search >= parm.size()){break;}
    }
    if(SimParameters.getImputationFile() != "nofile"){logfilestring=logfilestring+tempimpute;}
}
