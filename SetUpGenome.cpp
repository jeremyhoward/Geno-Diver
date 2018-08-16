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
#include "OutputFiles.h"
#include "Global_Population.h"

/* From EBV_Functions */
void VanRaden_grm(double* input_m, vector < string > const &genotypes, double* output_grm, float scaler);


using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////        Functions to Generate or Setup Sequence Generation Using MaCS               ////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
////                    Run Macs                      ////
//////////////////////////////////////////////////////////
void RunMaCS(parameters &SimParameters,globalpopvar &Population1,ostream& logfileloc)
{
    logfileloc << "============================================================\n";
    logfileloc << "===\t MaCS Part of Program (Chen et al. 2009) \t====\n";
    logfileloc << "============================================================\n";
    logfileloc << " Begin Generating Sequence Information: " << endl;
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
            logfileloc << "    - Ne that was read in: " << "'" << customNe << "'." << endl;
            float ScaledMutation = 4 * customNe * SimParameters.getu();
            logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
            float ScaledRecombination = 4 * 50 * 1.0e-8;
            logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                    logfileloc << "    - Line used to call MaCS: " << endl;
                    logfileloc << "    " << command << endl;
                }
                system(command.c_str());
                system("rm -rf haplo* tree.* debug.txt");
                /* Part 2 put into right format */
                system("tail -n +6 file1.txt > Intermediate.txt");
                part1 = "tail -n +2 Intermediate.txt > ";
                stringstream ss; ss << (i + 1); string str = ss.str();                                  /* Convert i loop to string chromosome number */
                part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str()); /* Store name of genotype file */
                system("head -n 1 Intermediate.txt > TempMap.txt");
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());             /* Store name of map file */
                part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                system("rm Intermediate.txt file1.txt TempMap.txt");
                /* need to move files to output Directory */
                part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
                command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
                logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
            }
        }
        if(SimParameters.getNe_spec() == "Ne70")
        {
            /* Need to first initialize paramters for macs */
            float ScaledMutation = 4 * 70 * SimParameters.getu();
            logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
            float ScaledRecombination = 4 * 70 * 1.0e-8;
            logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                    logfileloc << "    - Line used to call MaCS: " << endl;
                    logfileloc << "    " << command << endl;
                }
                system(command.c_str());
                system("rm -f haplo* tree.* debug.txt");
                /* Part 2 put into right format */
                system("tail -n +6 file1.txt > Intermediate.txt");
                part1 = "tail -n +2 Intermediate.txt > ";
                stringstream ss; ss << (i + 1); string str = ss.str();
                part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str()); /* Store name of genotype file */
                system("head -n 1 Intermediate.txt > TempMap.txt");
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());              /* Store name of map file */
                part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                system("rm Intermediate.txt file1.txt TempMap.txt");
                /* need to move files to Output Directory */
                part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
                command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
                logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
            }
        }
        if(SimParameters.getNe_spec() == "Ne100_Scen1")
        {
            /* Need to first initialize paramters for macs */
            float ScaledMutation = 4 * 100 * SimParameters.getu();
            logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
            float ScaledRecombination = 4 * 100 * 1.0e-8;
            logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                    logfileloc << "    - Line used to call MaCS: " << endl;
                    logfileloc << "    " << command << endl;
                }
                system(command.c_str());
                system("rm -rf haplo* tree.* debug.txt");
                /* Part 2 put into right format */
                system("tail -n +6 file1.txt > Intermediate.txt");
                part1 = "tail -n +2 Intermediate.txt > ";
                stringstream ss; ss << (i + 1); string str = ss.str();
                part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str());     /* Store name of genotype file */
                system("head -n 1 Intermediate.txt > TempMap.txt");
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());                  /* Store name of map file */
                part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                system("rm Intermediate.txt file1.txt TempMap.txt");
                /* need to move files to output Directory */
                part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
                command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
                logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
            }
        }
        if(SimParameters.getNe_spec() == "Ne100_Scen2")
        {
            /* Need to first initialize paramters for macs */
            float ScaledMutation = 4 * 100 * SimParameters.getu();
            logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
            float ScaledRecombination = 4 * 100 * 1.0e-8;
            logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                    logfileloc << "    - Line used to call MaCS: " << endl;
                    logfileloc << "    " << command << endl;
                }
                system(command.c_str());
                system("rm -rf haplo* tree.* debug.txt");
                /* Part 2 put into right format */
                system("tail -n +6 file1.txt > Intermediate.txt");
                part1 = "tail -n +2 Intermediate.txt > ";
                stringstream ss; ss << (i + 1); string str = ss.str();
                part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str());     /* Store name of genotype file */
                system("head -n 1 Intermediate.txt > TempMap.txt");
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());                  /* Store name of map file */
                part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                system("rm Intermediate.txt file1.txt TempMap.txt");
                /* need to move files to output Directory */
                part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
                command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
                logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
            }
        }
        if(SimParameters.getNe_spec() == "Ne250")
        {
            /* Need to first initialize paramters for macs */
            float ScaledMutation = 4 * 250 * SimParameters.getu();
            logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
            float ScaledRecombination = 4 * 250 * 1.0e-8;
            logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                    logfileloc << "    - Line used to call MaCS: " << endl;
                    logfileloc << "    " << command << endl;
                }
                system(command.c_str());
                system("rm -rf haplo* tree.* debug.txt");
                /* Part 2 put into right format */
                system("tail -n +6 file1.txt > Intermediate.txt");
                part1 = "tail -n +2 Intermediate.txt > ";
                stringstream ss; ss << (i + 1); string str = ss.str();
                part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str());     /* Store name of genotype file */
                system("head -n 1 Intermediate.txt > TempMap.txt");
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());                  /* Store name of map file */
                part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                system("rm Intermediate.txt file1.txt TempMap.txt");
                /* need to move files to output Directory */
                part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
                command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
                logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
            }
        }
        if(SimParameters.getNe_spec() == "Ne1000")
        {
            /* Need to first initialize paramters for macs */
            float ScaledMutation = 4 * 1000 * SimParameters.getu();
            logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
            float ScaledRecombination = 4 * 1000 * 1.0e-8;
            logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                    logfileloc << "    - Line used to call MaCS: " << endl;
                    logfileloc << "    " << command << endl;
                }
                system(command.c_str());
                system("rm -rf haplo* tree.* debug.txt");
                /* Part 2 put into right format */
                system("tail -n +6 file1.txt > Intermediate.txt");
                part1 = "tail -n +2 Intermediate.txt > ";
                stringstream ss; ss << (i + 1); string str = ss.str();
                part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
                string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str());     /* Store name of genotype file */
                system("head -n 1 Intermediate.txt > TempMap.txt");
                part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
                command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());                  /* Store name of map file */
                part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
                part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
                part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
                system("rm Intermediate.txt file1.txt TempMap.txt");
                /* need to move files to output Directory */
                part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
                command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
                command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
                logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
            }
        }
    }
    if(SimParameters.getne_founder() != -5 && SimParameters.getNe_spec() == "")
    {
        /* Need to first initialize paramters for macs */
        float ScaledMutation = 4 * SimParameters.getne_founder() * SimParameters.getu();
        logfileloc << "    - Scaled Mutation Rate used for Macs: " << ScaledMutation << "." << endl;
        float ScaledRecombination = 4 * SimParameters.getne_founder() * 1.0e-8;
        logfileloc << "    - Scaled Recombination Rate for Macs: " << ScaledRecombination  << "." << endl;
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
                logfileloc << "    - Line used to call MaCS: " << endl;
                logfileloc << "    " << command << endl;
            }
            system(command.c_str());
            system("rm -rf haplo* tree.* debug.txt");
            /* Part 2 put into right format */
            system("tail -n +6 file1.txt > Intermediate.txt");
            part1 = "tail -n +2 Intermediate.txt > ";
            stringstream ss; ss << (i + 1); string str = ss.str();
            part2 = "CH"; part3 = "SNP.txt"; command = part1 + part2 + str + part3;
            string genofile = part2 + str + part3; Population1.update_snpfiles(i,genofile); system(command.c_str());     /* Store name of genotype file */
            system("head -n 1 Intermediate.txt > TempMap.txt");
            part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; part1 = "tail -c +11 TempMap.txt > ";
            command = part1 + mapfile; Population1.update_mapfiles(i,mapfile); system(command.c_str());                  /* Store name of map file */
            part1 = "sed -e '1s/^.//' "; part2 = " > copyfile"; command = part1+mapfile+part2; system(command.c_str());
            part1 = "mv ./copyfile ./"; command = part1 +mapfile; system(command.c_str());
            part1 = "sed -e 's/1/2/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
            part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
            part1 = "sed -e 's/0/1/g' "; part2 = " > copyfile"; command = part1+genofile+part2; system(command.c_str());
            part1 = "mv ./copyfile ./"; command = part1 +genofile; system(command.c_str());
            system("rm Intermediate.txt file1.txt TempMap.txt");
            /* need to move files to output Directory */
            part1 = "mv ./"; part2 = " ./" + SimParameters.getOutputFold() + "/";
            command = part1 + (Population1.get_snpfiles())[i] + part2 + (Population1.get_snpfiles())[i]; system(command.c_str());
            command = part1 + (Population1.get_mapfiles())[i] + part2 + (Population1.get_mapfiles())[i]; system(command.c_str());
            logfileloc << "    - Finished Generating Sequence Information for Chromosome " << i + 1 << endl;
        }
    }
    logfileloc << " Finished Generating Sequence Information: " << endl << endl;
    system("rm -rf ./nul || true");
}
//////////////////////////////////////////////////////////
////  Figure out what map and snp files were called   ////
////  if not generation new sequence information      ////
//////////////////////////////////////////////////////////
void NoMaCSUpdateFiles(parameters &SimParameters,globalpopvar &Population1,ostream& logfileloc)
{
    logfileloc << "============================================================\n";
    logfileloc << "===\t MaCS Part of Program (Chen et al. 2009) \t====\n";
    logfileloc << "============================================================\n";
    logfileloc << "    - File already exist do not need to create sequence information." << endl;
    logfileloc <<"     - Need to ensure that parameters related to sequence information \n     - from previous simulation are what you wanted!!\n\n";
    /* even though files already exist need to get names of files for the SNP and Map files */
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        stringstream ss; ss<<(i + 1); string str=ss.str(); string part2="CH";
        string part3="SNP.txt"; string genofile=part2+str+part3; Population1.update_snpfiles(i,genofile);
        part2 = "Map"; part3 = ".txt"; string mapfile = part2 + str + part3; Population1.update_mapfiles(i,mapfile);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////               Functions to Setup Genome for Founder Population                     ////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////
// Generate Marker Map and Output ///
/////////////////////////////////////
void generatemapfile(globalpopvar &Population1, parameters &SimParameters,outputfiles &OUTPUTFILES)
{
    for(int i = 0; i < (Population1.get_markerindex()).size(); i++){Population1.add_marker_chr(0); Population1.add_marker_posmb(0);}
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        double chr = i + 1;
        for(int j = 0; j < (Population1.get_markerindex()).size(); j++)
        {
            if((Population1.get_markermapposition())[j] >= i + 1 && (Population1.get_markermapposition())[j] < i + 2)
            {
                Population1.update_marker_chr(j,chr);
                Population1.update_marker_posmb(j,(((Population1.get_markermapposition())[j] - chr) * (SimParameters.get_ChrLength())[i]));
            }
        }
    }
    /************************************/
    /** Generate Marker Map and Output **/
    /************************************/
    ofstream output21; output21.open (OUTPUTFILES.getloc_Marker_Map().c_str());
    output21 << "chr pos" << endl;
    for(int i = 0; i < (Population1.get_markerindex()).size(); i++)
    {
        output21 << (Population1.get_marker_chr())[i] << " " << (Population1.get_marker_posmb())[i] << endl;
    }
    output21.close();
    if(SimParameters.getmblengthroh() > 0)
    {
        ofstream outputrohfreq;
        outputrohfreq.open(OUTPUTFILES.getloc_Summary_ROHGenome_Freq().c_str());
        outputrohfreq << "chr pos" << endl;
        for(int i = 0; i < (Population1.get_markerindex()).size(); i++)
        {
            outputrohfreq << (Population1.get_marker_chr())[i] << " " << (Population1.get_marker_posmb())[i] << endl;
        }
        outputrohfreq.close();
        ofstream outputrohlength;
        outputrohlength.open(OUTPUTFILES.getloc_Summary_ROHGenome_Length().c_str());
        outputrohlength << "chr pos" << endl;
        for(int i = 0; i < (Population1.get_markerindex()).size(); i++)
        {
            outputrohlength << (Population1.get_marker_chr())[i] << " " << (Population1.get_marker_posmb())[i] << endl;
        }
        outputrohlength.close();
    }
}
////////////////////////////////////////////////////
// Generate Founder Animal Marker+QTL/FTL string ///
////////////////////////////////////////////////////
void generatefounderanimals(globalpopvar &Population1, parameters &SimParameters,outputfiles &OUTPUTFILES,ostream& logfileloc,vector <int> &FullColNum,vector <string> &founder_qtl, vector <string> &founder_markers)
{
    vector < string > founder_snp;
    /* Figure out Marker Number */
    int num_markqtl = Population1.getfullmarkernum(); int num_qtl = 0;
    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
    {
        if((Population1.get_qtl_type())[i] != 0){num_markqtl++; num_qtl++;}
    }
    /* Read in founder genotypes and grab markes and QTL/FTL */
    string line;
    ifstream infilefounder;
    infilefounder.open(OUTPUTFILES.getloc_foundergenofile().c_str());
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
        for(int i = 0; i < TotalSNP; i++)
        {
            int temp = geno[i] - 48; Genotype[i]= temp;         /* ASCI value is 48 for 0; and ASCI value for 0 to 9 is 48 to 57 */
        }
        int* MarkerGeno = new int[num_markqtl];                 /* Array that holds SNP that were declared as Markers and QTL */
        int markercounter = 0;                                  /* counter for where we are at in SNPLoc */
        int bigmarkercounter = 0;                               /* counter to determine where chromsome ends and begins */
        for(int i = 0; i < SimParameters.getChr(); i++)
        {
            /* fill in temp matrix where column number from SNPLoc should match up within each chromosome */
            vector < int > temp((Population1.get_chrsnplength())[i],0);
            int k = 0;
            /* grab genotype from right chromosome and go to next temp position */
            for(int j = bigmarkercounter; j < (bigmarkercounter + (Population1.get_chrsnplength())[i]); j++){temp[k] = Genotype[j]; k++;}
            for(int j = 0; j < (Population1.get_chrsnplength())[i]; j++)
            {
                if(markercounter < num_markqtl)
                {
                    /* if colnumber lines up with big file then it is a marker or QTL */
                    if(FullColNum[markercounter] == j){MarkerGeno[markercounter] = temp[j]; markercounter++;}
                }
            }
            /* Fix it so it kills once reached TotalSNPCounter */
            /* determine where next SNP should begin next in in large genotype file */
            if(i != SimParameters.getChr())
            {
                int temp = 0;
                for(int chr = 0; chr < i + 1; chr++){temp = (Population1.get_chrsnplength())[chr] + temp;}
                bigmarkercounter = temp;
            }
        }
        /* Put into string */
        stringstream strStreamgeno (stringstream::in | stringstream::out);
        for (int i=0; i < num_markqtl; i++){strStreamgeno << MarkerGeno[i];}
        geno = strStreamgeno.str();
        delete [] Genotype; delete [] MarkerGeno;
        founder_snp.push_back(geno);
    }
    infilefounder.close();
    logfileloc << "   - Check Structure of Founder Individuals." << endl;
    /* calculate frequency */
    vector < double > foundergmatrixfreq (founder_snp[0].size(),0.0);
    for(int i = 0; i < founder_snp.size(); i++)
    {
        string geno = founder_snp[i];
        for(int j = 0; j < geno.size(); j++)
        {
            int temp = geno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            foundergmatrixfreq[j] += temp;
        }
    }
    for(int i = 0; i < foundergmatrixfreq.size(); i++){foundergmatrixfreq[i] = foundergmatrixfreq[i] / (2 * founder_snp.size());}
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
    VanRaden_grm(Mfounder,founder_snp,foundergmatrix,founderscale);
    double mean_offdiagonals = 0.0; int numberoffdiagonals = 0;
    double sd_offdiagonals = 0.0;
    double max_offdiagonal = foundergmatrix[(0*founder_snp.size())+1];
    double min_offdiagonal = foundergmatrix[(0*founder_snp.size())+1];
    vector < int > extremerelationshipcount (founder_snp.size(),0);
    for(int i = 0; i < founder_snp.size(); i++)
    {
        for(int j = i; j < founder_snp.size(); j++)
        {
            if(i != j)
            {
                if(foundergmatrix[(i*founder_snp.size()) + j] > max_offdiagonal){max_offdiagonal = foundergmatrix[(i*founder_snp.size()) + j];}
                if(foundergmatrix[(i*founder_snp.size()) + j] < min_offdiagonal){min_offdiagonal = foundergmatrix[(i*founder_snp.size()) + j];}
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
    logfileloc << "        - Mean (sd) of off-diagonal founder GRM: " << mean_offdiagonals << " (" << sd_offdiagonals << ")" << endl;
    logfileloc << "        - Maximum off-diagonal: " << max_offdiagonal << "; Minimum off-diagonal: " << min_offdiagonal << endl;
    foundergmatrixfreq.clear(); delete [] Mfounder; delete [] foundergmatrix;
    /* put into founder qtl and markers then start selecting */
    for(int ind = 0; ind < founder_snp.size(); ind++)
    {
        string snp = founder_snp[ind];
        vector < int > MarkerGeno(snp.size(),0);
        for(int i = 0; i < snp.size(); i++){MarkerGeno[i] = snp[i] - 48;}   /* Put into vector */
        int* MarkerGenotypes = new int[Population1.getfullmarkernum()];     /* Marker Genotypes */
        int* QTLGenotypes = new int[num_qtl];                               /* QTL Genotypes */
        int Marker_IndCounter = 0;                                          /* Counter to determine where you are at in Marker index array */
        int QTL_IndCounter = 0;                                             /* Counter to determine where you are at in QTL index array */
        /* Fill Genotype Array */
        for(int i = 0; i < MarkerGeno.size(); i++)        /* Loop through MarkerGenotypes array & place Geno based on Index value */
        {
            if(Marker_IndCounter < (Population1.get_markerindex()).size())  /* ensures doesn't go over and cause valgrind error */
            {
                if(i == (Population1.get_markerindex())[Marker_IndCounter])
                {
                    MarkerGenotypes[Marker_IndCounter] = MarkerGeno[i]; Marker_IndCounter++;
                }
            }
            if(QTL_IndCounter < (Population1.get_qtlindex()).size())        /* ensures doesn't go over and cause valgrind error */
            {
                if(i == (Population1.get_qtlindex())[QTL_IndCounter])
                {
                    QTLGenotypes[QTL_IndCounter] = MarkerGeno[i]; QTL_IndCounter++;
                }
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
    vector < double > founderqtlfreq(founder_qtl[0].size(),0.0);
    for(int i = 0; i < founder_qtl.size(); i++)
    {
        string geno = founder_qtl[i];
        for(int j = 0; j < geno.size(); j++)
        {
            int temp = geno[j] - 48;
            if(temp == 3 || temp == 4){temp = 1;}
            founderqtlfreq[j] += temp;
        }
    }
    for(int i = 0; i < founderqtlfreq.size(); i++){founderqtlfreq[i] = founderqtlfreq[i] / (2 * founder_qtl.size());}
    for(int i = 0; i < founderqtlfreq.size(); i++){Population1.update_qtl_freq(i,founderqtlfreq[i]);}
    founderqtlfreq.clear();
}
/////////////////////////////////////////////////////////////////////////////////
////            Calculate Correlation Between Traits QTL Effects             ////
/////////////////////////////////////////////////////////////////////////////////
void calculatetraitcorrelation(globalpopvar &Population1,parameters &SimParameters,ostream& logfileloc)
{
    vector <double> mean_add(2,0.0); vector <double> var_add(2,0.0); double covar_add = 0.0;
    vector <double> mean_dom(2,0.0); vector <double> var_dom(2,0.0); double covar_dom = 0.0;
    int number = 0;
    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
    {
        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
        {
            mean_add[0] += Population1.get_qtl_add_quan(i,0); mean_add[1] += Population1.get_qtl_add_quan(i,1);
            if((SimParameters.get_Var_Dominance())[0] != 0.0 && (SimParameters.get_Var_Dominance())[1] != 0.0)
            {
                mean_dom[0] += Population1.get_qtl_dom_quan(i,0); mean_dom[1] += Population1.get_qtl_dom_quan(i,1);
            }
            number++;
        }
    }
    mean_add[0] /= double(number); mean_add[1] /= double(number);
    if((SimParameters.get_Var_Dominance())[0] != 0.0 && (SimParameters.get_Var_Dominance())[1] != 0.0)
    {
        mean_dom[0] /= double(number); mean_dom[1] /= double(number);
    }
    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
    {
        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
        {
            var_add[0] += (Population1.get_qtl_add_quan(i,0)-mean_add[0])*(Population1.get_qtl_add_quan(i,0)-mean_add[0]);
            var_add[1] += (Population1.get_qtl_add_quan(i,1)-mean_add[1])*(Population1.get_qtl_add_quan(i,1)-mean_add[1]);
            covar_add += ((Population1.get_qtl_add_quan(i,0)-mean_add[0])*(Population1.get_qtl_add_quan(i,1)-mean_add[1]));
            if((SimParameters.get_Var_Dominance())[0] != 0.0 && (SimParameters.get_Var_Dominance())[1] != 0.0)
            {
                var_dom[0] += (Population1.get_qtl_dom_quan(i,0)-mean_dom[0])*(Population1.get_qtl_dom_quan(i,0)-mean_dom[0]);
                var_dom[1] += (Population1.get_qtl_dom_quan(i,1)-mean_dom[0])*(Population1.get_qtl_dom_quan(i,1)-mean_dom[0]);
                covar_dom += ((Population1.get_qtl_dom_quan(i,0)-mean_dom[0])*(Population1.get_qtl_dom_quan(i,1)-mean_dom[0]));
            }
        }
    }
    var_add[0] /= double(number-1); var_add[1] /= double(number-1);
    covar_add = (covar_add / sqrt(var_add[0]*var_add[1])) / double(number-1);
    if((SimParameters.get_Var_Dominance())[0] != 0.0 && (SimParameters.get_Var_Dominance())[1] != 0.0)
    {
        var_dom[0] /= double(number-1); var_dom[1] /= double(number-1);
        covar_dom = (covar_dom / sqrt(var_dom[0]*var_dom[1])) / double(number-1);
        
    }
    logfileloc << "   - Correlation between additive effects of quantitative trait 1 and 2: " << covar_add << endl;
    if((SimParameters.get_Var_Dominance())[0] != 0.0 && (SimParameters.get_Var_Dominance())[1] != 0.0)
    {
        logfileloc << "   - Correlation between dominance effects of quantitative trait 1 and 2: " << covar_dom << endl;
    }
}
//////////////////////////////////////////////////////////
////            Scale Quantitative Traits             ////
//////////////////////////////////////////////////////////
/* Depending on the parameterization 'biological' or 'statistical' scale additive and dominance effects */
/* Variance Additive = 2pq[a + d(q-p)]^2; Variance Dominance = (2pqd)^2; One depends on the other so therefore do optimization technique */
void Scale_Quantitative(globalpopvar &Population1, parameters &SimParameters,vector <string> &founder_qtl,ostream& logfileloc)
{
    logfileloc << "   - Scale additive and dominance effects." << endl;
    if(SimParameters.getQuantParam() == "biological")
    {
        if((SimParameters.get_Var_Additive()).size() == 1)
        {
            /* put into 2-d vector of qtl genotypes */
            std::vector<std::vector<int> > qtlgenotypes(founder_qtl.size(),std::vector<int>(founder_qtl[0].size()));
            for(int i = 0; i < founder_qtl.size(); i++)
            {
                string geno = founder_qtl[i];
                for(int j = 0; j < geno.size(); j++)
                {
                    int temp = geno[j] - 48;
                    if(temp > 2){temp = 1;}
                    qtlgenotypes[i][j] = temp;
                }
            }
            double obsh2 = 0.0;               /* current iterations h2 for additive */
            double obsph2 = 0.0;              /* current iterations h2 for dominance */
            int quit = 1;                     /* won't quit until = 0 */
            vector <double> tempadd((Population1.get_qtl_type()).size(),0.0);       /* Temporary array to store scaled additive effects */
            vector <double> tempdom((Population1.get_qtl_type()).size(),0.0);       /* Temporary array to store scaled dominance effects */
            vector <double> temptbv(qtlgenotypes.size(),0.0);                       /* Temporary array to store tbv */
            vector <double> temptdd(qtlgenotypes.size(),0.0);                       /* Temporary array to store tdd */
            int interationnumber = 0;
            while(quit != 0)
            {
                for(int i = 0; i < temptbv.size(); i++){temptbv[i] = 0.0; temptdd[i] = 0.0;}
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        /* based on current, scale get temp additive effect */
                        tempadd[i] = Population1.get_qtl_add_quan(i,0)*(Population1.get_qtlscalefactadd())[0];
                        /* based on current, scale to get temp dom effect */
                        if((Population1.get_qtlscalefactdom())[0] == 0.0){tempdom[i] = 0;}
                        if((Population1.get_qtlscalefactdom())[0] > 0.0)
                        {
                            tempdom[i] = Population1.get_qtl_dom_quan(i,0)*(Population1.get_qtlscalefactdom())[0];
                        }
                    }
                }
                /* Calculate tbv and tdd for current marker estimates */
                for(int i = 0; i < qtlgenotypes.size(); i++)
                {
                    for(int j = 0; j < qtlgenotypes[0].size(); j++)
                    {
                        if((Population1.get_qtl_type())[j] == 2 || (Population1.get_qtl_type())[j] == 3)            /* Quantitative QTL */
                        {
                            if(qtlgenotypes[i][j] != 1){temptbv[i] += (qtlgenotypes[i][j] * double(tempadd[j]));}
                            if(qtlgenotypes[i][j] == 1)
                            {
                                temptbv[i] += (qtlgenotypes[i][j] * double(tempadd[j])); temptdd[i] += tempdom[j];
                            }
                        }
                    }
                }
                /* Calculate variance in tbv and tdd */
                double meantbv = 0.0; double meantdd = 0.0;
                double tempva = 0.0; double tempvd = 0.0;
                for(int i = 0; i < temptbv.size(); i++){meantbv += temptbv[i]; meantdd += temptdd[i];}
                meantbv /= double(temptbv.size());
                if(meantdd > 0.0){meantdd /= double(temptdd.size());}
                for(int i = 0; i < temptbv.size(); i++)
                {
                    tempva += (temptbv[i]-meantbv)*(temptbv[i]-meantbv);
                    if(meantdd > 0.0){tempvd += (temptdd[i]-meantdd)*(temptdd[i]-meantdd);}
                }
                tempva /= double(temptbv.size()-1);
                if(meantdd > 0.0){tempvd /= double(temptdd.size()-1);}
                /* Depeding on where at change */
                /* Add Var Within Window (+/-) don't change */
                /* Add Var Smaller than desired increase */
                if((SimParameters.get_Var_Additive())[0] - tempva > 0.004)
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]+0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                /* Add Var Bigger than desired decrease */
                if((SimParameters.get_Var_Additive())[0] - tempva < -0.004)
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]-0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                /* Dom Var Within Window (+/-) don't change */
                /* Dom Var Smaller than desired increase */
                if((SimParameters.get_Var_Dominance())[0] - tempvd > 0.004)
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]+0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                /* Dom Var Bigger than desired decrease */
                if((SimParameters.get_Var_Dominance())[0] - tempvd < -0.004)
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]-0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                /* Dom & Add Var within window */
                if(abs((SimParameters.get_Var_Additive())[0] - tempva) < 0.004 && abs((SimParameters.get_Var_Dominance())[0] - tempvd) < 0.004)
                {
                    quit = 0;
                    logfileloc << "   - Effects Centered and Scaled:" << endl;
                    logfileloc << "       - Additive Variance in Founders: " << tempva << "." << endl;
                    logfileloc << "       - Dominance Variance in Founders: " << tempvd << "." << endl;
                    logfileloc << "       - Scale factor for additive effects: " << (Population1.get_qtlscalefactadd())[0] << "." << endl;
                    logfileloc << "       - Scale factor for dominance effects: " << (Population1.get_qtlscalefactdom())[0] << "." << endl;
                    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                    {
                        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                        {
                            double temp = Population1.get_qtl_add_quan(i,0) * (Population1.get_qtlscalefactadd())[0];
                            Population1.update_qtl_add_quan(i,0,temp);
                            temp = Population1.get_qtl_dom_quan(i,0) * (Population1.get_qtlscalefactdom())[0];
                            if((Population1.get_qtlscalefactdom())[0] == 0.0){temp = 0.0;}
                            Population1.update_qtl_dom_quan(i,0,temp);
                        }
                    }
                    Population1.update_qtl_additivevar(0,0,tempva);
                    Population1.update_qtl_dominancevar(0,0,tempvd);
                }
                tempadd.clear(); tempdom.clear(); interationnumber++;
                //cout << interationnumber << " " << " " << tempva << " " << (Population1.get_qtlscalefactadd())[0] << " ";
                //cout << tempvd << " " << (Population1.get_qtlscalefactdom())[0] << endl << endl;;
                //if(interationnumber > 2000){exit (EXIT_FAILURE);}
            }
            for(int i = 0; i < qtlgenotypes.size(); i++){qtlgenotypes[i].clear();}
            qtlgenotypes.clear(); temptbv.clear(); temptdd.clear();
            if((Population1.get_qtlscalefactadd())[0] == 0 && (Population1.get_qtlscalefactdom())[0] == 0)
            {
                logfileloc << "   - Dominance and Additive Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
            }
            if((Population1.get_qtlscalefactadd())[0] > 0)
            {
                /* print number of QTL that display partial or over-dominance */
                int partDom = 0; int overDom = 0; int negativesign = 0; int positivesign = 0;
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        if(abs(Population1.get_qtl_dom_quan(i,0)) < abs(Population1.get_qtl_add_quan(i,0))){partDom += 1;}
                        if(abs(Population1.get_qtl_dom_quan(i,0)) > abs(Population1.get_qtl_add_quan(i,0))){overDom += 1;}
                        if(Population1.get_qtl_dom_quan(i,0) > 0){positivesign += 1;}
                        if(Population1.get_qtl_dom_quan(i,0) < 0){negativesign += 1;}
                    }
                }
                if((Population1.get_qtlscalefactdom())[0] > 0)
                {
                    logfileloc << "   -After Centering and Scaling Number of QTL with: " << endl;
                    logfileloc << "       - Partial-Dominance: " << partDom << "." << endl;
                    logfileloc << "       - Over-Dominance: " << overDom << "." << endl;
                    logfileloc << "       - Negative Sign: " << negativesign << "." << endl;
                    logfileloc << "       - Positive Sign: " << positivesign << "." << endl;
                }
                if((Population1.get_qtlscalefactdom())[0] == 0)
                {
                    logfileloc << "   - Dominance Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
                }
            }
        }
        if((SimParameters.get_Var_Additive()).size() == 3)
        {
            /* put into 2-d vector of qtl genotypes */
            std::vector<std::vector<int> > qtlgenotypes(founder_qtl.size(),std::vector<int>(founder_qtl[0].size()));
            for(int i = 0; i < founder_qtl.size(); i++)
            {
                string geno = founder_qtl[i];
                for(int j = 0; j < geno.size(); j++)
                {
                    int temp = geno[j] - 48;
                    if(temp > 2){temp = 1;}
                    qtlgenotypes[i][j] = temp;
                }
            }
            int quit = 1;                           /* won't quit until = 0 */
            /* Temporary array to store scaled additive and dominnace effects */
            vector < vector < double>> tempadd((Population1.get_qtl_type()).size(),vector<double>(SimParameters.getnumbertraits(),0.0));
            vector < vector < double>> tempdom((Population1.get_qtl_type()).size(),vector<double>(SimParameters.getnumbertraits(),0.0));
            vector < vector < double>> temptbv(qtlgenotypes.size(),vector<double>(SimParameters.getnumbertraits(),0.0)); /* Temporary array to store tbv */
            vector < vector < double>> temptdd(qtlgenotypes.size(),vector<double>(SimParameters.getnumbertraits(),0.0)); /* Temporary array to store tdd */
            int interationnumber = 0;
            while(quit != 0)
            {
                for(int i = 0; i < temptbv.size(); i++)
                {
                    for(int j = 0; j < SimParameters.getnumbertraits(); j++){temptbv[i][j] = 0.0; temptdd[i][j] = 0.0;}
                }
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            tempadd[i][j] = Population1.get_qtl_add_quan(i,j)*(Population1.get_qtlscalefactadd())[j];
                            if((Population1.get_qtlscalefactdom())[j] == 0.0){tempdom[i][j] = 0;}
                            if((Population1.get_qtlscalefactdom())[j] > 0.0)
                            {
                                tempdom[i][j] = Population1.get_qtl_dom_quan(i,j)*(Population1.get_qtlscalefactdom())[j];
                            }
                        }
                    }
                }
                /* Calculate tbv and tdd for current marker estimates */
                for(int i = 0; i < qtlgenotypes.size(); i++)
                {
                    for(int j = 0; j < qtlgenotypes[0].size(); j++)
                    {
                        if((Population1.get_qtl_type())[j] == 2 || (Population1.get_qtl_type())[j] == 3)            /* Quantitative QTL */
                        {
                            for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                            {
                                if(qtlgenotypes[i][j] != 1){temptbv[i][k] += (qtlgenotypes[i][j] * double(tempadd[j][k]));}
                                if(qtlgenotypes[i][j] == 1)
                                {
                                    temptbv[i][k] += (qtlgenotypes[i][j] * double(tempadd[j][k])); temptdd[i][k] += tempdom[j][k];
                                }
                            }
                        }
                    }
                }
                vector <double> mean_dom(2,0.0); vector <double>  mean_add(2,0.0);
                vector <double> var_dom(2,0.0); vector <double>  var_add(2,0.0);
                /* Calculate variance in tbv and tdd */
                for(int i = 0; i < temptbv.size(); i++)
                {
                    for(int k = 0; k < SimParameters.getnumbertraits(); k++){mean_add[k] += temptbv[i][k]; mean_dom[k] += temptdd[i][k];}
                }
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    mean_add[k] /= double(temptbv.size());
                    if(mean_dom[k] > 0.0){mean_dom[k] /= double(temptdd.size());}
                }
                for(int i = 0; i < temptbv.size(); i++)
                {
                    for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                    {
                        var_add[k] += (temptbv[i][k]-mean_add[k])*(temptbv[i][k]-mean_add[k]);
                        if(mean_dom[k] > 0.0){var_dom[k] += (temptdd[i][k]-mean_dom[k])*(temptdd[i][k]-mean_dom[k]);}
                    }
                }
                for(int k = 0; k < SimParameters.getnumbertraits(); k++)
                {
                    var_add[k] /= double(temptbv.size()-1);
                    if(mean_dom[k] > 0.0){var_dom[k] /= double(temptdd.size()-1);}
                }
                //for(int k = 0; k < SimParameters.getnumbertraits(); k++){cout << var_add[k] << " " << var_dom[k] << endl;}
                /* Depeding on where at change */
                /* Add Var Within Window (+/-) don't change */
                if((SimParameters.get_Var_Additive())[0] - var_add[0] > 0.004)          /* Add Var Smaller than desired increase Trait 1*/
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]+0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                if((SimParameters.get_Var_Additive())[0] - var_add[0] < -0.004)         /* Add Var Bigger than desired decrease Trait 1*/
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]-0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                if((SimParameters.get_Var_Additive())[2] - var_add[1] > 0.004)          /* Add Var Smaller than desired increase Trait 2 */
                {
                    double temp = ((Population1.get_qtlscalefactadd())[1]+0.0001);
                    Population1.update_qtlscalefactadd(1,temp);
                }
                if((SimParameters.get_Var_Additive())[2] - var_add[1] < -0.004)         /* Add Var Bigger than desired decrease Trait 2 */
                {
                    double temp = ((Population1.get_qtlscalefactadd())[1]-0.0001);
                    Population1.update_qtlscalefactadd(1,temp);
                }
                /* Dom Var Within Window (+/-) don't change */
                if((SimParameters.get_Var_Dominance())[0] - var_dom[0] > 0.004)          /* Dom Var Smaller than desired increase Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]+0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                if((SimParameters.get_Var_Dominance())[0] - var_dom[0] < -0.004)         /* Dom Var Bigger than desired decrease Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]-0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                /* Dom Var Within Window (+/-) don't change */
                if((SimParameters.get_Var_Dominance())[2] - var_dom[1] > 0.004)          /* Dom Var Smaller than desired increase Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[1]+0.0001);
                    Population1.update_qtlscalefactdom(1,temp);
                }
                if((SimParameters.get_Var_Dominance())[2] - var_dom[1] < -0.004)         /* Dom Var Bigger than desired decrease Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[1]-0.0001);
                    Population1.update_qtlscalefactdom(1,temp);
                }
                if(abs((SimParameters.get_Var_Additive())[0] - var_add[0]) < 0.004 && abs((SimParameters.get_Var_Additive())[2] - var_add[1]) < 0.004 && abs((SimParameters.get_Var_Dominance())[0] - var_dom[0]) < 0.004 && abs((SimParameters.get_Var_Dominance())[2] - var_dom[1]) < 0.004)
                {
                    quit = 0;
                    logfileloc << "   - Effects Centered and Scaled:" << endl;
                    logfileloc << "       - Trait 1:" << endl;
                    logfileloc << "       - Additive Variance in Founders: " << var_add[0] << "." << endl;
                    logfileloc << "       - Dominance Variance in Founders: " << var_dom[0] << "." << endl;
                    logfileloc << "       - Scale factor for additive effects: " << (Population1.get_qtlscalefactadd())[0] << "." << endl;
                    logfileloc << "       - Scale factor for dominance effects: " << (Population1.get_qtlscalefactdom())[0] << "." << endl;
                    logfileloc << "       - Trait 2:" << endl;
                    logfileloc << "       - Additive Variance in Founders: " << var_add[1] << "." << endl;
                    logfileloc << "       - Dominance Variance in Founders: " << var_dom[1] << "." << endl;
                    logfileloc << "       - Scale factor for additive effects: " << (Population1.get_qtlscalefactadd())[1] << "." << endl;
                    logfileloc << "       - Scale factor for dominance effects: " << (Population1.get_qtlscalefactdom())[1] << "." << endl;
                    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                    {
                        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                        {
                            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                            {
                                double temp = Population1.get_qtl_add_quan(i,j) * (Population1.get_qtlscalefactadd())[j];
                                Population1.update_qtl_add_quan(i,j,temp);
                                temp = Population1.get_qtl_dom_quan(i,j) * (Population1.get_qtlscalefactdom())[j];
                                if((Population1.get_qtlscalefactdom())[j] == 0.0){temp = 0.0;}
                                Population1.update_qtl_dom_quan(i,j,temp);
                            }
                        }
                    }
                    for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                    {
                        Population1.update_qtl_additivevar(0,j,var_add[j]);
                        Population1.update_qtl_dominancevar(0,j,var_dom[j]);
                    }
                }
                //if(interationnumber % 1 == 0)
                //{
                //    cout << interationnumber << ": " << var_add[0] << " " << var_add[1] << "   ---    ";
                //    cout << var_dom[0] << " " << var_dom[1] << " " << (Population1.get_qtlscalefactadd())[0] << " ";
                //   cout << (Population1.get_qtlscalefactadd())[1] << endl << endl;
                //}
                //if(interationnumber > 2000){exit (EXIT_FAILURE);}
                interationnumber++;
            }
            if((Population1.get_qtlscalefactadd())[0] == 0 && (Population1.get_qtlscalefactdom())[0] == 0)
            {
                logfileloc << "   - Dominance and Additive Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
            }
            if((Population1.get_qtlscalefactadd())[0] > 0)
            {
                /* print number of QTL that display partial or over-dominance */
                vector <int> partDom(SimParameters.getnumbertraits(),0);
                vector <int> overDom(SimParameters.getnumbertraits(),0);
                vector <int> negativesign(SimParameters.getnumbertraits(),0);
                vector <int> positivesign(SimParameters.getnumbertraits(),0);
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            if(abs(Population1.get_qtl_dom_quan(i,j)) < abs(Population1.get_qtl_add_quan(i,j))){partDom[j] += 1;}
                            if(abs(Population1.get_qtl_dom_quan(i,j)) > abs(Population1.get_qtl_add_quan(i,j))){overDom[j] += 1;}
                            if(Population1.get_qtl_dom_quan(i,j) > 0){positivesign[j] += 1;}
                            if(Population1.get_qtl_dom_quan(i,j) < 0){negativesign[j] += 1;}
                        }
                    }
                }
                if((Population1.get_qtlscalefactdom())[0] > 0)
                {
                    for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                    {
                        logfileloc << "   -After Centering and Scaling for Trait 1 Number of QTL with: " << endl;
                        logfileloc << "       - Partial-Dominance: " << partDom[j] << "." << endl;
                        logfileloc << "       - Over-Dominance: " << overDom[j] << "." << endl;
                        logfileloc << "       - Negative Sign: " << negativesign[j] << "." << endl;
                        logfileloc << "       - Positive Sign: " << positivesign[j] << "." << endl;
                    }
                    //logfileloc << "   -Correlation between additive effects Trait 1 and 2: " << correlation[0] << "." << endl;
                    //logfileloc << "   -Correlation between dominance effects Trait 1 and 2: " << correlation[1] << "." << endl;
                }
                if((Population1.get_qtlscalefactdom())[0] == 0)
                {
                    logfileloc << "   - Dominance Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
                }
            }
        }
    }
    if(SimParameters.getQuantParam() == "statistical")
    {
        if((SimParameters.get_Var_Additive()).size() == 1)
        {
            double obsh2 = 0.0;               /* current iterations h2 for additive */
            double obsph2 = 0.0;              /* current iterations h2 for dominance */
            int quit = 1;                     /* won't quit until = 0 */
            vector < double > tempadd((Population1.get_qtl_type()).size(),0.0);     /* Temporary array to store scaled additive effects */
            vector < double > tempdom((Population1.get_qtl_type()).size(),0.0);     /* Temporary array to store scaled dominance effects */
            int interationnumber = 0;
            while(quit != 0)
            {
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        /* based on current, scale get temp additive effect */
                        tempadd[i] = Population1.get_qtl_add_quan(i,0)*(Population1.get_qtlscalefactadd())[0];
                        /* based on current, scale to get temp dom effect */
                        if((Population1.get_qtlscalefactdom())[0] == 0.0){tempdom[i] = 0;}
                        if((Population1.get_qtlscalefactdom())[0] > 0.0)
                        {
                            tempdom[i] = Population1.get_qtl_dom_quan(i,0)*(Population1.get_qtlscalefactdom())[0];
                        }
                    }
                }
                double tempva = 0; double tempvd = 0;           /* Variance in additive or dominance based on current iterations scale factor */
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        tempva += 2 * (Population1.get_qtl_freq())[i] * (1 - (Population1.get_qtl_freq())[i]) * ((tempadd[i]+(tempdom[i] * ((1 - (Population1.get_qtl_freq())[i]) - (Population1.get_qtl_freq())[i]))) * (tempadd[i]+(tempdom[i] * ((1 - (Population1.get_qtl_freq())[i]) - (Population1.get_qtl_freq())[i]))));
                        tempvd += ((2 * (Population1.get_qtl_freq())[i] * (1 - (Population1.get_qtl_freq())[i]) * tempdom[i]) * (2 * (Population1.get_qtl_freq())[i] * (1 - (Population1.get_qtl_freq())[i]) * tempdom[i]));
                    }
                }
                /* Depeding on where at change */
                /* Add Var Within Window (+/-) don't change */
                /* Add Var Smaller than desired increase */
                if((SimParameters.get_Var_Additive())[0] - tempva > 0.004)
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]+0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                /* Add Var Bigger than desired decrease */
                if((SimParameters.get_Var_Additive())[0] - tempva < -0.004)
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]-0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                /* Dom Var Within Window (+/-) don't change */
                /* Dom Var Smaller than desired increase */
                if((SimParameters.get_Var_Dominance())[0] - tempvd > 0.004)
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]+0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                /* Dom Var Bigger than desired decrease */
                if((SimParameters.get_Var_Dominance())[0] - tempvd < -0.004)
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]-0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                /* Dom & Add Var within window */
                if(abs((SimParameters.get_Var_Additive())[0] - tempva) < 0.004 && abs((SimParameters.get_Var_Dominance())[0] - tempvd) < 0.004)
                {
                    quit = 0;
                    logfileloc << "   - Effects Centered and Scaled:" << endl;
                    logfileloc << "       - Additive Variance in Founders: " << tempva << "." << endl;
                    logfileloc << "       - Dominance Variance in Founders: " << tempvd << "." << endl;
                    logfileloc << "       - Scale factor for additive effects: " << (Population1.get_qtlscalefactadd())[0] << "." << endl;
                    logfileloc << "       - Scale factor for dominance effects: " << (Population1.get_qtlscalefactdom())[0] << "." << endl;
                    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                    {
                        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                        {
                            double temp = Population1.get_qtl_add_quan(i,0) * (Population1.get_qtlscalefactadd())[0];
                            Population1.update_qtl_add_quan(i,0,temp);
                            temp = Population1.get_qtl_dom_quan(i,0) * (Population1.get_qtlscalefactdom())[0];
                            if((Population1.get_qtlscalefactdom())[0] == 0.0){temp = 0.0;}
                            Population1.update_qtl_dom_quan(i,0,temp);
                        }
                    }
                    Population1.update_qtl_additivevar(0,0,tempva);
                    Population1.update_qtl_dominancevar(0,0,tempvd);
                }
                tempadd.clear(); tempdom.clear(); interationnumber++;
                //cout << interationnumber << " " << " " << tempva << " " << (Population1.get_qtlscalefactadd())[0] << " ";
                //cout << tempvd << " " << (Population1.get_qtlscalefactdom())[0] << endl << endl;;
                //if(interationnumber > 10000){exit (EXIT_FAILURE);}
            }
            if((Population1.get_qtlscalefactadd())[0] == 0 && (Population1.get_qtlscalefactdom())[0] == 0)
            {
                logfileloc << "   - Dominance and Additive Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
            }
            if((Population1.get_qtlscalefactadd())[0] > 0)
            {
                /* print number of QTL that display partial or over-dominance */
                int partDom = 0; int overDom = 0; int negativesign = 0; int positivesign = 0;
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        if(abs(Population1.get_qtl_dom_quan(i,0)) < abs(Population1.get_qtl_add_quan(i,0))){partDom += 1;}
                        if(abs(Population1.get_qtl_dom_quan(i,0)) > abs(Population1.get_qtl_add_quan(i,0))){overDom += 1;}
                        if(Population1.get_qtl_dom_quan(i,0) > 0){positivesign += 1;}
                        if(Population1.get_qtl_dom_quan(i,0) < 0){negativesign += 1;}
                    }
                }
                if((Population1.get_qtlscalefactdom())[0] > 0)
                {
                    logfileloc << "   -After Centering and Scaling Number of QTL with: " << endl;
                    logfileloc << "       - Partial-Dominance: " << partDom << "." << endl;
                    logfileloc << "       - Over-Dominance: " << overDom << "." << endl;
                    logfileloc << "       - Negative Sign: " << negativesign << "." << endl;
                    logfileloc << "       - Positive Sign: " << positivesign << "." << endl;
                }
                if((Population1.get_qtlscalefactdom())[0] == 0)
                {
                    logfileloc << "   - Dominance Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
                }
            }
        }
        if((SimParameters.get_Var_Additive()).size() == 3)
        {
            int quit = 1;                           /* won't quit until = 0 */
            /* Temporary array to store scaled additive and dominnace effects */
            vector < vector < double>> tempadd((Population1.get_qtl_type()).size(),vector<double>(SimParameters.getnumbertraits(),0.0));
            vector < vector < double>> tempdom((Population1.get_qtl_type()).size(),vector<double>(SimParameters.getnumbertraits(),0.0));
            int interationnumber = 0;
            while(quit != 0)
            {
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            tempadd[i][j] = Population1.get_qtl_add_quan(i,j)*(Population1.get_qtlscalefactadd())[j];
                            if((Population1.get_qtlscalefactdom())[j] == 0.0){tempdom[i][j] = 0;}
                            if((Population1.get_qtlscalefactdom())[j] > 0.0)
                            {
                                tempdom[i][j] = Population1.get_qtl_dom_quan(i,j)*(Population1.get_qtlscalefactdom())[j];
                            }
                        }
                    }
                }
                vector <double> var_dom(2,0.0); vector <double>  var_add(2,0.0);
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            var_add[j] += 2 * (Population1.get_qtl_freq())[i] * (1 - (Population1.get_qtl_freq())[i]) * ((tempadd[i][j]+(tempdom[i][j] * ((1 - (Population1.get_qtl_freq())[i]) - (Population1.get_qtl_freq())[i]))) * (tempadd[i][j]+(tempdom[i][j] * ((1 - (Population1.get_qtl_freq())[i]) - (Population1.get_qtl_freq())[i]))));
                            var_dom[j] += ((2 * (Population1.get_qtl_freq())[i] * (1 - (Population1.get_qtl_freq())[i]) * tempdom[i][j]) * (2 * (Population1.get_qtl_freq())[i] * (1 - (Population1.get_qtl_freq())[i]) * tempdom[i][j]));
                        }
                    }
                }
                /* Depeding on where at change */
                /* Add Var Within Window (+/-) don't change */
                if((SimParameters.get_Var_Additive())[0] - var_add[0] > 0.004)          /* Add Var Smaller than desired increase Trait 1*/
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]+0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                if((SimParameters.get_Var_Additive())[0] - var_add[0] < -0.004)         /* Add Var Bigger than desired decrease Trait 1*/
                {
                    double temp = ((Population1.get_qtlscalefactadd())[0]-0.0001);
                    Population1.update_qtlscalefactadd(0,temp);
                }
                if((SimParameters.get_Var_Additive())[2] - var_add[1] > 0.004)          /* Add Var Smaller than desired increase Trait 2 */
                {
                    double temp = ((Population1.get_qtlscalefactadd())[1]+0.0001);
                    Population1.update_qtlscalefactadd(1,temp);
                }
                if((SimParameters.get_Var_Additive())[2] - var_add[1] < -0.004)         /* Add Var Bigger than desired decrease Trait 2 */
                {
                    double temp = ((Population1.get_qtlscalefactadd())[1]-0.0001);
                    Population1.update_qtlscalefactadd(1,temp);
                }
                /* Dom Var Within Window (+/-) don't change */
                if((SimParameters.get_Var_Dominance())[0] - var_dom[0] > 0.004)          /* Dom Var Smaller than desired increase Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]+0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                if((SimParameters.get_Var_Dominance())[0] - var_dom[0] < -0.004)         /* Dom Var Bigger than desired decrease Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[0]-0.0001);
                    Population1.update_qtlscalefactdom(0,temp);
                }
                /* Dom Var Within Window (+/-) don't change */
                if((SimParameters.get_Var_Dominance())[2] - var_dom[1] > 0.004)          /* Dom Var Smaller than desired increase Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[1]+0.0001);
                    Population1.update_qtlscalefactdom(1,temp);
                }
                if((SimParameters.get_Var_Dominance())[2] - var_dom[1] < -0.004)         /* Dom Var Bigger than desired decrease Trait 1 */
                {
                    double temp = ((Population1.get_qtlscalefactdom())[1]-0.0001);
                    Population1.update_qtlscalefactdom(1,temp);
                }
                if(abs((SimParameters.get_Var_Additive())[0] - var_add[0]) < 0.004 && abs((SimParameters.get_Var_Additive())[2] - var_add[1]) < 0.004 && abs((SimParameters.get_Var_Dominance())[0] - var_dom[0]) < 0.004 && abs((SimParameters.get_Var_Dominance())[2] - var_dom[1]) < 0.004)
                {
                    quit = 0;
                    logfileloc << "   - Effects Centered and Scaled:" << endl;
                    logfileloc << "       - Trait 1:" << endl;
                    logfileloc << "       - Additive Variance in Founders: " << var_add[0] << "." << endl;
                    logfileloc << "       - Dominance Variance in Founders: " << var_dom[0] << "." << endl;
                    logfileloc << "       - Scale factor for additive effects: " << (Population1.get_qtlscalefactadd())[0] << "." << endl;
                    logfileloc << "       - Scale factor for dominance effects: " << (Population1.get_qtlscalefactdom())[0] << "." << endl;
                    logfileloc << "       - Trait 2:" << endl;
                    logfileloc << "       - Additive Variance in Founders: " << var_add[1] << "." << endl;
                    logfileloc << "       - Dominance Variance in Founders: " << var_dom[1] << "." << endl;
                    logfileloc << "       - Scale factor for additive effects: " << (Population1.get_qtlscalefactadd())[1] << "." << endl;
                    logfileloc << "       - Scale factor for dominance effects: " << (Population1.get_qtlscalefactdom())[1] << "." << endl;
                    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                    {
                        if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                        {
                            for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                            {
                                double temp = Population1.get_qtl_add_quan(i,j) * (Population1.get_qtlscalefactadd())[j];
                                Population1.update_qtl_add_quan(i,j,temp);
                                temp = Population1.get_qtl_dom_quan(i,j) * (Population1.get_qtlscalefactdom())[j];
                                if((Population1.get_qtlscalefactdom())[j] == 0.0){temp = 0.0;}
                                Population1.update_qtl_dom_quan(i,j,temp);
                            }
                        }
                    }
                    for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                    {
                        Population1.update_qtl_additivevar(0,j,var_add[j]);
                        Population1.update_qtl_dominancevar(0,j,var_dom[j]);
                    }
                }
                //if(interationnumber % 100 == 0)
                //{
                //    cout << interationnumber << ": " << var_add[0] << " " << var_add[1] << "   ---    ";
                //    cout << var_dom[0] << " " << var_dom[1] << " " << (Population1.get_qtlscalefactadd())[0] << " ";
                //    cout << (Population1.get_qtlscalefactadd())[1] << endl;
                //}
                //if(interationnumber > 2500){exit (EXIT_FAILURE);}
                interationnumber++;
            }
            if((Population1.get_qtlscalefactadd())[0] == 0 && (Population1.get_qtlscalefactdom())[0] == 0)
            {
                logfileloc << "   - Dominance and Additive Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
            }
            if((Population1.get_qtlscalefactadd())[0] > 0)
            {
                /* print number of QTL that display partial or over-dominance */
                vector <int> partDom(SimParameters.getnumbertraits(),0);
                vector <int> overDom(SimParameters.getnumbertraits(),0);
                vector <int> negativesign(SimParameters.getnumbertraits(),0);
                vector <int> positivesign(SimParameters.getnumbertraits(),0);
                for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
                {
                    if((Population1.get_qtl_type())[i] == 2 || (Population1.get_qtl_type())[i] == 3)
                    {
                        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                        {
                            if(abs(Population1.get_qtl_dom_quan(i,j)) < abs(Population1.get_qtl_add_quan(i,j))){partDom[j] += 1;}
                            if(abs(Population1.get_qtl_dom_quan(i,j)) > abs(Population1.get_qtl_add_quan(i,j))){overDom[j] += 1;}
                            if(Population1.get_qtl_dom_quan(i,j) > 0){positivesign[j] += 1;}
                            if(Population1.get_qtl_dom_quan(i,j) < 0){negativesign[j] += 1;}
                        }
                    }
                }
                if((Population1.get_qtlscalefactdom())[0] > 0)
                {
                    for(int j = 0; j < SimParameters.getnumbertraits(); j++)
                    {
                        logfileloc << "   -After Centering and Scaling for Trait 1 Number of QTL with: " << endl;
                        logfileloc << "       - Partial-Dominance: " << partDom[j] << "." << endl;
                        logfileloc << "       - Over-Dominance: " << overDom[j] << "." << endl;
                        logfileloc << "       - Negative Sign: " << negativesign[j] << "." << endl;
                        logfileloc << "       - Positive Sign: " << positivesign[j] << "." << endl;
                    }
                    //logfileloc << "   -Correlation between additive effects Trait 1 and 2: " << correlation[0] << "." << endl;
                    //logfileloc << "   -Correlation between dominance effects Trait 1 and 2: " << correlation[1] << "." << endl;
                }
                if((Population1.get_qtlscalefactdom())[0] == 0)
                {
                    logfileloc << "   - Dominance Variance is 0; No summary statistics on proportion +/- or degree of dominance." << endl;
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////
// Summarize Fitness Mutations                              //
//////////////////////////////////////////////////////////////
void SummarizeFitness(globalpopvar &Population1,outputfiles &OUTPUTFILES,parameters &SimParameters,ostream& logfileloc)
{
    int num_qtl = 0;
    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
    {
        if((Population1.get_qtl_type())[i] != 0){num_qtl++;}
    }
    /* generate mean selection and dominance coeffecint for fitness traits */
    double meanlethal_sel = 0.0; double meanlethal_dom = 0.0; int numberlethal_sel = 0;
    double meansublethal_sel = 0.0; double meansublethal_dom = 0.0; int numbersublethal_sel = 0;
    double meanfreq_leth = 0.0; double meanfreq_sublethal = 0.0;
    for(int i = 0; i < num_qtl; i++)
    {
        if((Population1.get_qtl_type())[i] == 4)            /* Lethal */
        {
            meanlethal_sel += (Population1.get_qtl_add_fit())[i];
            meanlethal_dom += (Population1.get_qtl_dom_fit())[i]; numberlethal_sel += 1;
            
            if((Population1.get_qtl_freq())[i] < 0.5){meanfreq_leth += (Population1.get_qtl_freq())[i];}
            if((Population1.get_qtl_freq())[i] > 0.5){meanfreq_leth += (1 - (Population1.get_qtl_freq())[i]);}
        }
        if((Population1.get_qtl_type())[i] == 3 || (Population1.get_qtl_type())[i] == 5)    /* Sub-Lethal */
        {
            meansublethal_sel += (Population1.get_qtl_add_fit())[i];
            meansublethal_dom += (Population1.get_qtl_dom_fit())[i]; numbersublethal_sel += 1;
            if((Population1.get_qtl_freq())[i] < 0.5){meanfreq_sublethal += (Population1.get_qtl_freq())[i];}
            if((Population1.get_qtl_freq())[i] > 0.5){meanfreq_sublethal += (1 - (Population1.get_qtl_freq())[i]);}
        }
    }
    if(numbersublethal_sel > 0 && SimParameters.getproppleitropic() > 0)
    {
        meansublethal_sel = meansublethal_sel / double(numbersublethal_sel);
        meansublethal_dom = meansublethal_dom / double(numbersublethal_sel);
        meanfreq_sublethal = meanfreq_sublethal / double(numbersublethal_sel);
        logfileloc << "   - Fitness Sub-Lethal Allele: " << endl;
        logfileloc << "        - Mean Frequency: " << meanfreq_sublethal << endl;
        logfileloc << "        - Mean Selection Coefficient: " << meansublethal_sel << endl;
        logfileloc << "        - Mean Degree of Dominance: " << meansublethal_dom << endl;
    }
    if(numbersublethal_sel > 0 && SimParameters.getproppleitropic() == 0)
    {
        meansublethal_sel = meansublethal_sel / double(numbersublethal_sel);
        meansublethal_dom = meansublethal_dom / double(numbersublethal_sel);
        meanfreq_sublethal = meanfreq_sublethal / double(numbersublethal_sel);
        logfileloc << "   - Fitness Sub-Lethal Allele: " << endl;
        logfileloc << "        - Mean Frequency: " << meanfreq_sublethal << endl;
        logfileloc << "        - Mean Selection Coefficient: " << meansublethal_sel << endl;
        logfileloc << "        - Mean Degree of Dominance: " << meansublethal_dom << endl;
    }
    if(numbersublethal_sel == 0)
    {
        logfileloc << "   - No Sub-Lethal Fitness Mutations; No summary statistics on frequency and allelic effects." << endl;
    }
    if(numberlethal_sel > 0)
    {
        meanlethal_sel = meanlethal_sel / double(numberlethal_sel);
        meanlethal_dom = meanlethal_dom / double(numberlethal_sel);
        meanfreq_leth = meanfreq_leth / double(numberlethal_sel);
        logfileloc << "   - Fitness Lethal Allele: " << endl;
        logfileloc << "        - Mean Frequency: " << meanfreq_leth << endl;
        logfileloc << "        - Mean Selection Coefficient: " << meanlethal_sel << endl;
        logfileloc << "        - Mean Degree of Dominance: " << meanlethal_dom << endl;
    }
    if(numberlethal_sel == 0)
    {
        logfileloc << "   - No Lethal Fitness Mutations; No summary statistics on frequency and allelic effects." << endl;
    }
}
//////////////////////////////////////////////////////////////
// Save Founder mutation in QTL_new_old vector class object //
//////////////////////////////////////////////////////////////
void AddToQTLClass(globalpopvar &Population1,outputfiles &OUTPUTFILES,vector < QTL_new_old > &population_QTL,parameters &SimParameters,ostream& logfileloc)
{
    int num_qtl = 0;
    for(int i = 0; i < (Population1.get_qtl_type()).size(); i++)
    {
        if((Population1.get_qtl_type())[i] != 0){num_qtl++;}
    }
    for(int i = 0; i < num_qtl; i++)
    {
        if((Population1.get_qtl_type())[i] == 4 || (Population1.get_qtl_type())[i] == 5)            /* Fitness QTL */
        {
            stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<(Population1.get_qtl_freq())[i];
            string stringfreq=strStreamtempfreq.str();
            stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<(Population1.get_qtl_type())[i];
            string stringtype=strStreamtemptype.str();
            QTL_new_old tempa((Population1.get_qtl_mapposition())[i],vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0),(Population1.get_qtl_add_fit())[i],(Population1.get_qtl_dom_fit())[i],stringtype, 0, stringfreq, "");
            /* add fitness effects to vector */
            tempa.update_Additivevect(0,(Population1.get_qtl_add_fit())[i]); tempa.update_Dominancevect(0,(Population1.get_qtl_dom_fit())[i]);
            if(SimParameters.getnumbertraits() == 2)        /* put second trait as 0.0 */
            {
                tempa.update_Additivevect(1,0.0); tempa.update_Dominancevect(1,0.0);
            }
            population_QTL.push_back(tempa);
        }
        if((Population1.get_qtl_type())[i] == 2)                                /* Quantitative QTL */
        {
            stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<(Population1.get_qtl_freq())[i];
            string stringfreq=strStreamtempfreq.str();
            stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<(Population1.get_qtl_type())[i];
            string stringtype=strStreamtemptype.str();
            QTL_new_old tempa((Population1.get_qtl_mapposition())[i],vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0),Population1.get_qtl_add_quan(i,0),Population1.get_qtl_dom_quan(i,0),stringtype,0, stringfreq,"");
            tempa.update_Additivevect(0,Population1.get_qtl_add_quan(i,0)); tempa.update_Dominancevect(0,Population1.get_qtl_dom_quan(i,0));
            if(SimParameters.getnumbertraits() == 2)        /* put second trait as 0.0 */
            {
                tempa.update_Additivevect(1,Population1.get_qtl_add_quan(i,1)); tempa.update_Dominancevect(1,Population1.get_qtl_dom_quan(i,1));
            }
            population_QTL.push_back(tempa);
        }
        if((Population1.get_qtl_type())[i] == 3)                                /* Fitness + Quantitative QTL */
        {
            /* Quantitative one first */
            stringstream strStreamtempfreq (stringstream::in | stringstream::out); strStreamtempfreq<<(Population1.get_qtl_freq())[i];
            string stringfreq=strStreamtempfreq.str();
            stringstream strStreamtemptype (stringstream::in | stringstream::out); strStreamtemptype<<"2";
            string stringtype=strStreamtemptype.str();
            QTL_new_old tempa((Population1.get_qtl_mapposition())[i],vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0),Population1.get_qtl_add_quan(i,0),Population1.get_qtl_dom_quan(i,0),stringtype,0, stringfreq,"");
            tempa.update_Additivevect(0,Population1.get_qtl_add_quan(i,0)); tempa.update_Dominancevect(0,Population1.get_qtl_dom_quan(i,0));
            if(SimParameters.getnumbertraits() == 2)        /* put second trait as 0.0 */
            {
                tempa.update_Additivevect(1,Population1.get_qtl_add_quan(i,1)); tempa.update_Dominancevect(1,Population1.get_qtl_dom_quan(i,1));
            }
            population_QTL.push_back(tempa);
            /* Fitness one */
            stringstream strStreamtemptypea (stringstream::in | stringstream::out); strStreamtemptypea<<"5";
            string stringtypea=strStreamtemptypea.str();
            QTL_new_old tempb((Population1.get_qtl_mapposition())[i],vector<double>(SimParameters.getnumbertraits(),0.0),vector<double>(SimParameters.getnumbertraits(),0.0),(Population1.get_qtl_add_fit())[i],(Population1.get_qtl_dom_fit())[i],stringtypea, 0, stringfreq, "");
            tempb.update_Additivevect(0,(Population1.get_qtl_add_fit())[i]); tempb.update_Dominancevect(0,(Population1.get_qtl_dom_fit())[i]);
            if(SimParameters.getnumbertraits() == 2)        /* put second trait as 0.0 */
            {
                tempb.update_Additivevect(1,0.0); tempa.update_Dominancevect(1,0.0);
            }
            population_QTL.push_back(tempb);
        }
    }
    /* Deletes Previous Simulation */
    fstream checkqtl; checkqtl.open(OUTPUTFILES.getloc_qtl_class_object().c_str(), std::fstream::out | std::fstream::trunc); checkqtl.close();
    /* Save to a file that will continually get updated based on new mutation and frequencies */
    ofstream output;
    output.open (OUTPUTFILES.getloc_qtl_class_object().c_str());
    for(int i = 0; i < population_QTL.size(); i++)
    {
        output << population_QTL[i].getLocation() << " ";
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            output << (population_QTL[i].get_Additivevect())[j] << " " << (population_QTL[i].get_Dominancevect())[j] << " ";
        }
        output << population_QTL[i].getType() << " ";
        output << population_QTL[i].getGenOccured() << " " << population_QTL[i].getFreq() << endl;
    }
    output.close();
    logfileloc << "   - Copied Founder Mutations to QTL class object." << endl;
}
//////////////////////////////////////////////////////////
//  Generate hapLibrary and tabulate unique haplotypes  //
//////////////////////////////////////////////////////////
void GenerateHapLibrary(globalpopvar &Population1,parameters &SimParameters,vector < hapLibrary > &haplib)
{
    vector < int > hapChr(Population1.getfullmarkernum(),0);             /* Stores the chromosome number for a given marker */
    vector < int > hapNum(Population1.getfullmarkernum(),0);             /* Number of SNP based on string of markers */
    #pragma omp parallel for
    for(int i = 0; i < Population1.getfullmarkernum(); i++)
    {
        hapChr[i] = (Population1.get_markermapposition())[i]; hapNum[i] = i; /* When converted to integer will always round down */
    }
    /* Create index values to grab chunks from */
    int haplotindex = 1;                                /* Initialize haplotype id */
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        vector < int > tempchr;                         /* stores chromosome in vector */
        vector < int > tempnum;                         /* stores index of where it is at */
        int j = 0;
        /* Grab Correct Chromosome and store in three vectors */
        while(j < (Population1.getfullmarkernum()))
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
}

/////////////////////////////////////////////////////////////////////////////////
////            Calculate Correlation Between Traits (sum of effects)        ////
/////////////////////////////////////////////////////////////////////////////////
void traitcorrelation(vector <Animal> &population, ostream& logfileloc)
{
    vector < double > correlations(5,0.0);
    /* Phenotype */
    vector <double> mean(2,0.0); vector <double> var(2,0.0); double covar = 0.0;
    for(int i = 0; i < population.size(); i++)
    {
        mean[0] += (population[i].get_Phenvect())[0]; mean[1] += (population[i].get_Phenvect())[1];
    }
    mean[0] /= population.size(); mean[1] /= population.size();
    for(int i = 0; i < population.size(); i++)
    {
        var[0] += (((population[i].get_Phenvect())[0]-mean[0])*((population[i].get_Phenvect())[0]-mean[0]));
        var[1] += (((population[i].get_Phenvect())[1]-mean[1])*((population[i].get_Phenvect())[1]-mean[1]));
        covar += (((population[i].get_Phenvect())[0]-mean[0])*((population[i].get_Phenvect())[1]-mean[1]));
    }
    var[0] /= double(population.size()-1); var[1] /= double(population.size()-1);
    correlations[0] = (covar / sqrt(var[0]*var[1])) / double(population.size()-1);
    /* GV */
    mean[0] = 0.0; mean[1] = 0.0; var[0] = 0.0; var[1] = 0.0; covar = 0.0;
    for(int i = 0; i < population.size(); i++)
    {
        mean[0] += (population[i].get_GVvect())[0]; mean[1] += (population[i].get_GVvect())[1];
    }
    mean[0] /= population.size(); mean[1] /= population.size();
    for(int i = 0; i < population.size(); i++)
    {
        var[0] += (((population[i].get_GVvect())[0]-mean[0])*((population[i].get_GVvect())[0]-mean[0]));
        var[1] += (((population[i].get_GVvect())[1]-mean[1])*((population[i].get_GVvect())[1]-mean[1]));
        covar += (((population[i].get_GVvect())[0]-mean[0])*((population[i].get_GVvect())[1]-mean[1]));
    }
    var[0] /= double(population.size()-1); var[1] /= double(population.size()-1);
    correlations[1] = (covar / sqrt(var[0]*var[1])) / double(population.size()-1);
    /* BV */
    mean[0] = 0.0; mean[1] = 0.0; var[0] = 0.0; var[1] = 0.0; covar = 0.0;
    for(int i = 0; i < population.size(); i++)
    {
        mean[0] += (population[i].get_BVvect())[0]; mean[1] += (population[i].get_BVvect())[1];
    }
    mean[0] /= population.size(); mean[1] /= population.size();
    for(int i = 0; i < population.size(); i++)
    {
        var[0] += (((population[i].get_BVvect())[0]-mean[0])*((population[i].get_BVvect())[0]-mean[0]));
        var[1] += (((population[i].get_BVvect())[1]-mean[1])*((population[i].get_BVvect())[1]-mean[1]));
        covar += (((population[i].get_BVvect())[0]-mean[0])*((population[i].get_BVvect())[1]-mean[1]));
    }
    var[0] /= double(population.size()-1); var[1] /= double(population.size()-1);
    correlations[2] = (covar / sqrt(var[0]*var[1])) / double(population.size()-1);
    /* DD */
    mean[0] = 0.0; mean[1] = 0.0; var[0] = 0.0; var[1] = 0.0; covar = 0.0;
    for(int i = 0; i < population.size(); i++)
    {
        mean[0] += (population[i].get_DDvect())[0]; mean[1] += (population[i].get_DDvect())[1];
    }
    mean[0] /= population.size(); mean[1] /= population.size();
    for(int i = 0; i < population.size(); i++)
    {
        var[0] += (((population[i].get_DDvect())[0]-mean[0])*((population[i].get_DDvect())[0]-mean[0]));
        var[1] += (((population[i].get_DDvect())[1]-mean[1])*((population[i].get_DDvect())[1]-mean[1]));
        covar += (((population[i].get_DDvect())[0]-mean[0])*((population[i].get_DDvect())[1]-mean[1]));
    }
    var[0] /= double(population.size()-1); var[1] /= double(population.size()-1);
    correlations[3] = (covar / sqrt(var[0]*var[1])) / double(population.size()-1);
    /* Res */
    mean[0] = 0.0; mean[1] = 0.0; var[0] = 0.0; var[1] = 0.0; covar = 0.0;
    for(int i = 0; i < population.size(); i++)
    {
        mean[0] += (population[i].get_Rvect())[0]; mean[1] += (population[i].get_Rvect())[1];
    }
    mean[0] /= population.size(); mean[1] /= population.size();
    for(int i = 0; i < population.size(); i++)
    {
        var[0] += (((population[i].get_Rvect())[0]-mean[0])*((population[i].get_Rvect())[0]-mean[0]));
        var[1] += (((population[i].get_Rvect())[1]-mean[1])*((population[i].get_Rvect())[1]-mean[1]));
        covar += (((population[i].get_Rvect())[0]-mean[0])*((population[i].get_Rvect())[1]-mean[1]));
    }
    var[0] /= double(population.size()-1); var[1] /= double(population.size()-1);
    correlations[4] = (covar / sqrt(var[0]*var[1])) / double(population.size()-1);
    logfileloc << "   - Correlation Between Quantitiative Traits: " << endl;
    logfileloc << "        - Phenotype: " << correlations[0] << "." << endl;
    logfileloc << "        - Genotypic Value: " << correlations[1] << "." << endl;
    logfileloc << "        - Breeding Value: " << correlations[2] << "." << endl;
    logfileloc << "        - Dominance Deviation: " << correlations[3] << "." << endl;
    logfileloc << "        - Residual: " << correlations[4] << "." << endl;
}


