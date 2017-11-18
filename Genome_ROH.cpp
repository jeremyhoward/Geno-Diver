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

#include "Animal.h"
#include "ParameterClass.h"
#include "Genome_ROH.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////        Functions that generate ROH based diversity metric across the genome and LD decay surrounding QTL           ////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////        ROH Functions       ////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////

// Class to store index start and stop site ##
// Constructors ROH_Index
ROH_Index::ROH_Index(){Chromosome = 0;  StartPosition = 0; EndPosition = 0; StartIndex = 0; EndIndex = 0; NumberSNP = 0;}
ROH_Index::ROH_Index(int chr, int stpos, int enpos, int stind, int enind, int numsnp)
{
    Chromosome = chr;  StartPosition = stpos; EndPosition = enpos;
    StartIndex = stind; EndIndex = enind; NumberSNP = numsnp;
}
// Constructors Ani_ROH
Ani_ROH::Ani_ROH(){ROHChrome = ""; ROHStart = ""; ROHEnd = ""; Distance = "";}
Ani_ROH::Ani_ROH(std::string rohchrome, std::string ws, std::string we, std::string dist_MB)
{
    ROHChrome = rohchrome; ROHStart = ws; ROHEnd = we; Distance = dist_MB;
}
// Destructors
ROH_Index::~ROH_Index(){}                   /* ROH_Index*/
Ani_ROH::~Ani_ROH(){}                       /* Ani_ROH */


void Genome_ROH_Summary(parameters &SimParameters, vector <Animal> &population,string Marker_Map, string Summary_ROHGenome_Length, string Summary_ROHGenome_Freq, int Gen, ostream& logfileloc)
{
    /* Read in map file don't need to know how many SNP are in the file */
    vector <string> numbers;
    /* Import file and put each row into a vector */
    string line;
    ifstream infile;
    infile.open(Marker_Map.c_str());
    if(infile.fail()){cout << "Error Opening map File\n"; exit (EXIT_FAILURE);}
    while (getline(infile,line)){numbers.push_back(line);}        /* Stores in vector and each new line push back to next space */
    vector < int > chr;         /* stores chromosome in vector */
    vector < int > position;    /* stores position */
    vector < int > index;       /* used for when grabs size of 4Mb */
    for(int i = 1; i < numbers.size(); i++)
    {
        string temp = numbers[i];
        size_t pos = temp.find(" ", 0); string tempa = temp.substr(0,pos); chr.push_back(atoi(tempa.c_str())); /* Grab Chr */
        temp.erase(0, pos+1); position.push_back(atoi(temp.c_str()));
        index.push_back(i-1);
    }
    numbers.clear();                                    // clear vector that holds each row
    vector<ROH_Index> roh_index;
    /* Create index to grab correct columns from genotype file when constructing ROH and Autozygosity*/
    for(int i = 0; i < chr.size(); i++)
    {
        int j = i;
        while(1)
        {
            if(chr[i] == chr[j])
            {
                int temp = position[j] - position[i];
                if(temp > (SimParameters.getmblengthroh()*1000000))
                {
                    int numsnp = index[j] - index[i] + 1;
                    ROH_Index roh_temp(chr[i],position[i],position[j],index[i],index[j],numsnp);
                    roh_index.push_back(roh_temp);                  /* store in vector of roh_index objects */
                    break;
                }
                if(temp <= (SimParameters.getmblengthroh()*1000000)){j++;}
            }
            if(chr[i] != chr[j]){break;}
        }
    }
    vector < double > autozygosityfreq(chr.size(),0);               /* Hold count of SNP being in ROH */
    vector < vector < int >> autozygositylength;                    /* Hold length of SNP if in ROH */
    for(int i = 0; i < autozygosityfreq.size(); i++){autozygositylength.push_back(vector<int>(0));}
    int individuals = 0;
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            string temp = population[i].getMarker();
            std::replace(temp.begin(),temp.end(),'3','1');                                      /* Convert heterozygotes to 1 */
            std::replace(temp.begin(),temp.end(),'4','1');                              /* Convert heterozygotes to 1 */
            vector < double > roh_status(roh_index.size(),5);                           /* Vector to store ROH status */
            vector < int > auto_status(temp.size(),5);                                  /* Vector to store autozygosity genotypes */
            /* put in 3 vectors that keep track of chr start and end in Mb for ROH in order to calculate ROH based inbreeding for individual */
            vector < int > indchr_roh; vector < int > indstr_roh; vector < int > indend_roh; vector < int > rohlength;
            for(int i = 0; i < roh_index.size(); i++)                                           /* loop across roh indexes */
            {
                /* check to see if a 1 exists; if so then not a ROH */
                size_t found =  (temp.substr(roh_index[i].getStInd(),roh_index[i].getNumSNP())).find("1");
                /* if found than isn't an ROH  */
                if(found != string::npos){roh_status[i] = 0;}
                /* if not found than is in an ROH  */
                if (found == string::npos){roh_status[i] = 1;}
                /* save regions that were in ROH to calculate proportion ROH */
                if(roh_status[i] == 1)
                {
                    indchr_roh.push_back(roh_index[i].getChr());
                    indstr_roh.push_back(roh_index[i].getStPos());
                    indend_roh.push_back(roh_index[i].getEnPos());
                }
                for(int j = roh_index[i].getStInd(); j < roh_index[i].getEnInd() + 1; j++)  /* fill autozygosity matrix */
                {
                    if(auto_status[j] == 0 && roh_status[i] == 1)          /* has been changed to a not in an ROH but is now in one */
                    {
                        auto_status[j] = 1;                                /* SNP is within an ROH */
                    }
                    if(auto_status[j] == 5 && roh_status[i] == 0)          /* has not been changed yet and is not in ROH */
                    {
                        auto_status[j] = 0;                                /* SNP is not within an ROH */
                    }
                    if(auto_status[j] == 5 && roh_status[i] == 1)          /* has not been changed yet and is in an ROH */
                    {
                        auto_status[j] = 1;                                /* SNP is within an ROH */
                    }
                    /* once been tagged as being in an ROH can't go back to not being in an ROH */
                }
            }
            //for(int i = 0; i < chr.size(); i++){cout << NewAnimalROH[i];}
            //cout << endl << endl;
            //for(int i = 0; i < chr.size(); i++){cout << auto_status[i];}
            //cout << endl << endl;
            //for(int i = 0; i < indchr_roh.size(); i++){cout << indchr_roh[i] << " " << indstr_roh[i] << " " << indend_roh[i] << endl;}
            int vector_size = indstr_roh.size();
            if(indstr_roh.size() > 1)
            {
                string stop = "GO";
                int i = 1;
                while(stop == "GO")
                {
                    /* If current ROH is within previous one then remove and replace end with current row */
                    if(indchr_roh[i - 1] == indchr_roh[i] && indstr_roh[i] >= indstr_roh[i-1] && indstr_roh[i] <= indend_roh[i-1])
                    {
                        indend_roh[i-1] = indend_roh[i];            /* replace row before it with end position */
                        indchr_roh.erase(indchr_roh.begin() + i);   /* delete that row from indchr_roh */
                        indstr_roh.erase(indstr_roh.begin() + i);   /* delete that row from indchr_roh */
                        indend_roh.erase(indend_roh.begin() + i);   /* delete that row from indchr_roh */
                        vector_size = vector_size - 1;
                    }
                    /* Not within each other so skip and go onto next one */
                    if(indchr_roh[i - 1] != indchr_roh[i] || indstr_roh[i] < indstr_roh[i-1] || indstr_roh[i] > indend_roh[i-1])
                    {
                        i = i + 1;
                    }
                    /* Once it reaches the last one stop */
                    if(i == vector_size){stop = "Kill";}
                }
            }
            for(int i = 0; i < autozygosityfreq.size(); i++)
            {
                if(auto_status[i] != 5){autozygosityfreq[i] += auto_status[i];}
                if(auto_status[i] == 5){autozygosityfreq[i] = -5;}
            }
            //for(int i = 0; i < indchr_roh.size(); i++){cout << indchr_roh[i] << " " << indstr_roh[i] << " " << indend_roh[i] << endl;}
            /* Fill in length of roh; if autosnp within window put length and add it to number of times */
            for(int i = 0; i < indchr_roh.size(); i++)
            {
                for(int j = 0; j < chr.size(); j++)
                {
                    if(indchr_roh[i] == chr[j])
                    {
                        if(position[j] >= indstr_roh[i] && position[j] <= indend_roh[i])
                        {
                            if(auto_status[j] != 5){autozygositylength[j].push_back(indend_roh[i] - indstr_roh[i]);}
                        }
                    }
                }
            }
            individuals++;
        }
    }
    /* Calculate Frequency and median length */
    for(int i = 0; i < autozygosityfreq.size(); i++)
    {
        if(autozygosityfreq[i] != -5){autozygosityfreq[i] = autozygosityfreq[i] / double(individuals);}
    }
    vector < string > automean_median(autozygosityfreq.size(),"");
    for(int i = 0; i < autozygositylength.size(); i++)
    {
        if(autozygositylength[i].size() > 0)
        {
            vector < int > temp(autozygositylength[i].size(),0); double sum = 0.0;
            for(int j = 0; j < autozygositylength[i].size(); j++){temp[j] = autozygositylength[i][j]; sum += double(temp[j]);}
            stringstream s1; s1 << int((sum / autozygositylength[i].size())+0.5); string tempmean = s1.str();
            sort(temp.begin(),temp.end());
            string tempmedian;
            if(autozygositylength[i].size() % 2 != 0){stringstream s2; s2 << temp[int((autozygositylength[i].size()/2)+0.5)]; tempmedian = s2.str();}
            if(autozygositylength[i].size() % 2 == 0)
            {
                int tempmedianvalue = (temp[int((autozygositylength[i].size() / 2)-1)] + temp[int((autozygositylength[i].size()/2))]) / 2;
                stringstream s2; s2 << tempmedianvalue; tempmedian = s2.str();
                automean_median[i] = tempmean + "_" + tempmedian;
            }
            automean_median[i] = tempmean + "_" + tempmedian;
        }
        if(autozygositylength[i].size() == 0)
        {
            stringstream s2; s2 << -5; string tempmedian = s2.str();
            automean_median[i] = tempmedian + "_" + tempmedian;
        }
        //cout << autozygositylength[i].size() << " " << autozygosityfreq[i] << " " << automean_median[i] << endl;
    }
    /* Once finished save in 2-D vector that stores everything */
    /* Update ROH genome length summary file */
    numbers.clear();
    /* Import file and put each row into a vector */
    ifstream infile1;
    infile1.open(Summary_ROHGenome_Length.c_str());
    if(infile1.fail()){cout << "Error Opening map File\n"; exit (EXIT_FAILURE);}
    while (getline(infile1,line)){numbers.push_back(line);}        /* Stores in vector and each new line push back to next space */
    /* now update numbers with new generation and then output back out */
    for(int i = 0; i < numbers.size(); i++)
    {
        if(i == 0)
        {
            stringstream s2; s2 << Gen; string tempgen = s2.str();
            string outstring = numbers[i] + " Gen" + tempgen;
            numbers[i] = outstring;
        }
        if(i > 0){string outstring = numbers[i] + " " + automean_median[i-1]; numbers[i] = outstring;}
    }
    fstream clearlength; clearlength.open(Summary_ROHGenome_Length.c_str(), std::fstream::out | std::fstream::trunc); clearlength.close();
    ofstream outputrohlength;
    outputrohlength.open(Summary_ROHGenome_Length.c_str());
    for(int i = 0; i < numbers.size(); i++){outputrohlength << numbers[i] << endl;}
    outputrohlength.close();
    numbers.clear();
    /* Import file and put each row into a vector */
    ifstream infile2;
    infile2.open(Summary_ROHGenome_Freq.c_str());
    if(infile2.fail()){cout << "Error Opening map File\n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line)){numbers.push_back(line);}        /* Stores in vector and each new line push back to next space */
    /* now update numbers with new generation and then output back out */
    for(int i = 0; i < numbers.size(); i++)
    {
        if(i == 0)
        {
            stringstream s2; s2 << Gen; string tempgen = s2.str();
            string outstring = numbers[i] + " Gen" + tempgen;
            numbers[i] = outstring;
        }
        if(i > 0)
        {
            stringstream s2; s2 << autozygosityfreq[i-1]; string tempgen = s2.str();
            string outstring = numbers[i] + " " + tempgen; numbers[i] = outstring;
        }
    }
    fstream clearfreq; clearfreq.open(Summary_ROHGenome_Freq.c_str(), std::fstream::out | std::fstream::trunc); clearfreq.close();
    ofstream outputrohfreq;
    outputrohfreq.open(Summary_ROHGenome_Freq.c_str());
    for(int i = 0; i < numbers.size(); i++){outputrohfreq << numbers[i] << endl;}
    outputrohfreq.close();
}

void Proportion_ROH(parameters &SimParameters, vector <Animal> &population,string Marker_Map, ostream& logfileloc)
{
    /* Read in map file don't need to know how many SNP are in the file */
    vector <string> numbers;
    /* Import file and put each row into a vector */
    string line;
    ifstream infile;
    infile.open(Marker_Map.c_str());
    if(infile.fail()){cout << "Error Opening map File\n"; exit (EXIT_FAILURE);}
    while (getline(infile,line)){numbers.push_back(line);}        /* Stores in vector and each new line push back to next space */
    vector < int > chr;         /* stores chromosome in vector */
    vector < int > position;    /* stores position */
    vector < int > index;       /* used for when grabs size of 4Mb */
    for(int i = 1; i < numbers.size(); i++)
    {
        string temp = numbers[i];
        size_t pos = temp.find(" ", 0); string tempa = temp.substr(0,pos); chr.push_back(atoi(tempa.c_str())); /* Grab Chr */
        temp.erase(0, pos+1); position.push_back(atoi(temp.c_str()));
        index.push_back(i-1);
    }
    numbers.clear();                                    // clear vector that holds each row
    vector<ROH_Index> roh_index;
    /* Create index to grab correct columns from genotype file when constructing ROH and Autozygosity*/
    for(int i = 0; i < chr.size(); i++)
    {
        int j = i;
        while(1)
        {
            if(chr[i] == chr[j])
            {
                int temp = position[j] - position[i];
                if(temp > (SimParameters.getmblengthroh()*1000000))
                {
                    int numsnp = index[j] - index[i] + 1;
                    ROH_Index roh_temp(chr[i],position[i],position[j],index[i],index[j],numsnp);
                    roh_index.push_back(roh_temp);                  /* store in vector of roh_index objects */
                    break;
                }
                if(temp <= (SimParameters.getmblengthroh()*1000000)){j++;}
            }
            if(chr[i] != chr[j]){break;}
        }
    }
    /* Figure out full genome length */
    double genome_length = 0.0;
    for(int i = 0; i < SimParameters.getChr(); i++){genome_length += (SimParameters.get_ChrLength())[i];}
    #pragma omp parralel for
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            string temp = population[i].getMarker();
            std::replace(temp.begin(),temp.end(),'3','1');                              /* Convert heterozygotes to 1 */
            std::replace(temp.begin(),temp.end(),'4','1');                              /* Convert heterozygotes to 1 */
            vector < int > indchr_roh; vector < int > indstr_roh; vector < int > indend_roh; vector < int > rohlength;
            for(int i = 0; i < roh_index.size(); i++)                                           /* loop across roh indexes */
            {
                /* check to see if a 1 exists; if so then not a ROH */
                size_t found =  (temp.substr(roh_index[i].getStInd(),roh_index[i].getNumSNP())).find("1");
                /* if not found than is in an ROH  */
                if (found == string::npos)
                {
                    indchr_roh.push_back(roh_index[i].getChr());
                    indstr_roh.push_back(roh_index[i].getStPos());
                    indend_roh.push_back(roh_index[i].getEnPos());
                }
            }
            int vector_size = indstr_roh.size();
            if(indstr_roh.size() > 1)
            {
                string stop = "GO";
                int i = 1;
                while(stop == "GO")
                {
                    /* If current ROH is within previous one then remove and replace end with current row */
                    if(indchr_roh[i - 1] == indchr_roh[i] && indstr_roh[i] >= indstr_roh[i-1] && indstr_roh[i] <= indend_roh[i-1])
                    {
                        indend_roh[i-1] = indend_roh[i];            /* replace row before it with end position */
                        indchr_roh.erase(indchr_roh.begin() + i);   /* delete that row from indchr_roh */
                        indstr_roh.erase(indstr_roh.begin() + i);   /* delete that row from indchr_roh */
                        indend_roh.erase(indend_roh.begin() + i);   /* delete that row from indchr_roh */
                        vector_size = vector_size - 1;
                    }
                    /* Not within each other so skip and go onto next one */
                    if(indchr_roh[i - 1] != indchr_roh[i] || indstr_roh[i] < indstr_roh[i-1] || indstr_roh[i] > indend_roh[i-1])
                    {
                        i = i + 1;
                    }
                    /* Once it reaches the last one stop */
                    if(i == vector_size){stop = "Kill";}
                }
            }
            double total_length = 0.0;
            for(int i = 0; i < indchr_roh.size(); i++){total_length += double(indend_roh[i] - indstr_roh[i]);}
            population[i].UpdatepropROH(total_length / double(genome_length));
        }
    }
}
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////     LD-Decay Functions     ////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
/****************************************************************/
/* Calculate LD decay by binning into windows across the genome */
/****************************************************************/
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
            if(samplingindex.size() > 10)
            {
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
/******************************************/
/* Calculate LD decay for each QTL or FTL */
/******************************************/
void qtlld_decay_estimator(parameters &SimParameters, vector <Animal> &population, vector < QTL_new_old > &population_QTL,string Marker_Map,string foundergen, string QTL_LD_Decay_File, string Phase_Persistance_File, string Phase_Persistance_Outfile)
{
    //time_t fullbegin_time = time(0);
    /* Read in Marker File */
    /* read in mb and chr genotype information for markers */
    vector < int > markerpositionMb;
    vector < int > markerchromosome;
    int linenumber = 0;
    ifstream infile1; string line;
    infile1.open(Marker_Map.c_str());
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
    /* Previous Generation R value */
    vector <int> old_QTL_outrfile; vector < int > old_diff_outrfile; vector < int > old_class_rfile;
    vector < vector < double >> old_d_outrfile;
    if(foundergen != "yes")
    {
        vector < string > number;
        ifstream infile2;
        infile2.open(Phase_Persistance_File.c_str());
        if(infile2.fail()){cout << "Error Opening File\n";}
        while (getline(infile2,line)){number.push_back(line);}
        if(number.size() > 0)
        {
            for(int i = 0; i < number.size(); i++)
            {
                size_t pos = number[i].find(" ", 0); old_QTL_outrfile.push_back((std::stoi(number[i].substr(0,pos)))); number[i].erase(0, pos + 1);
                pos = number[i].find(" ", 0); old_diff_outrfile.push_back((std::stoi(number[i].substr(0,pos)))); number[i].erase(0, pos + 1);
                pos = number[i].find(" ", 0); old_class_rfile.push_back((std::stoi(number[i].substr(0,pos)))); number[i].erase(0, pos + 1);
                /* loop through and grab d for a previous generation */
                vector <double> temp;
                for(int j = 0; j < SimParameters.getGener(); j++)
                {
                    pos = number[i].find(" ",0);
                    temp.push_back(atof(number[i].substr(0,pos).c_str()));
                    if(pos != std::string::npos){number[i].erase(0, pos + 1);}
                    if(pos == std::string::npos){number[i].clear(); j = SimParameters.getGener();}
                }
                old_d_outrfile.push_back(temp);
            }
        }
    }
    /* Read in animals and fill vectors to compute average r2 and persitance of phase */
    vector <int> animal;
    vector <int> generation;
    vector <string> qtlgeno;
    vector <string> markergeno;
    /***********************************************/
    /* Grab the animals that are currently progeny */
    /***********************************************/
    for(int i = 0; i < population.size(); i++)
    {
        if(population[i].getAge() == 1)
        {
            animal.push_back(population[i].getID()); generation.push_back(population[i].getGeneration());
            qtlgeno.push_back(population[i].getQTL()); markergeno.push_back(population[i].getMarker());
        }
    }
    //for(int i = 0; i < animal.size(); i++){cout << animal[i] << " " << generation[i] << "    +++    ";}
    /* Loop across most recent generations; Do LD within generation and phase across generations */
    int n = animal.size(); int mqtl = qtlgeno[0].size(); int mark = markergeno[0].size(); int animalindex;
    int * QTL = new int[n*mqtl];
    int * Marker = new int[n*mark];
    string tempgeno;
    for(int i = 0; i < generation.size(); i++)
    {
        tempgeno = qtlgeno[i];
        //if(i == 0){cout << tempgeno << endl;}
        for(int j = 0; j < mqtl; j++){int snp = tempgeno[j] - 48; QTL[(i*mqtl)+j] = snp;}
        tempgeno = markergeno[i];
        for(int j = 0; j < mark; j++){int snp = tempgeno[j] - 48; Marker[(i*mark)+j] = snp;}
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << QTL[(i*mqtl)+j] << " ";}
    //    cout << endl;
    //}
    //cout << endl;
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Marker[(i*mark)+j] << " ";}
    //    cout << endl;
    //}
    //cout << endl;
    /* Loop across QTL now */
    double hap11, hap12, hap21, hap22, freqsnp1, freqsnp2, sum1, sum2, sum3, sum4, sum5;
    int numb1, numb2, numb3, numb4, numb5;
    /* Used to calculate persistance */
    vector < int > QTL_outrfile; vector < int > diff_outrfile; vector < double > d_outrfile; vector < int > group_outrfile;
    for(int q = 0; q < population_QTL.size(); q++)
    {
        int qtlchr = int(population_QTL[q].getLocation());
        double qtlpos = population_QTL[q].getLocation() - qtlchr;
        int qtlposmb = qtlpos * (SimParameters.get_ChrLength())[qtlchr-1];
        //cout << "QTL " << q << ": " << population_QTL[q].getLocation() << " " << (SimParameters.get_ChrLength())[qtlchr-1] << " ";
        //cout << population_QTL[q].getFreq() << endl;
        //cout << qtlchr << " " << qtlpos << " " << qtlposmb << " " << endl;
        int leftend = qtlposmb - 2500000;
        if(leftend < 0){leftend = 0;}
        int rightend = qtlposmb + 2500000;
        if(rightend > (SimParameters.get_ChrLength())[qtlchr-1]){rightend = (SimParameters.get_ChrLength())[qtlchr-1];}
        //cout << leftend << " " << rightend << " ";
        /* Grab Markers 5 Mb to right and left */
        vector <int> indextograb; vector <double> r2value; vector <double> rvalue; vector <int> lowfreq; vector <int> distanceqtl;
        /* Start at beginning of chromosome */
        int searchloc = 0; int currentchr = 1;
        while(qtlchr > currentchr){searchloc += (SimParameters.get_Marker_chr())[qtlchr-1]; currentchr++;}
        //cout << searchloc << " ";
        //cout << searchloc << endl;
        while(1)
        {
            if(markerchromosome[searchloc] == qtlchr && markerpositionMb[searchloc] > leftend && markerpositionMb[searchloc] < rightend)
            {
                indextograb.push_back(searchloc); r2value.push_back(-5); rvalue.push_back(-5);
                distanceqtl.push_back(markerpositionMb[searchloc]-qtlposmb);
            }
            searchloc++;
            if(markerchromosome[searchloc] != qtlchr){break;}
            if(markerpositionMb[searchloc] > rightend){break;}
        }
        //cout << indextograb.size() << endl;
        //for(int check = 0; check < indextograb.size(); check++){cout << indextograb[check] << " " << distanceqtl[check] << "   +   ";}
        for(int markind = 0 ; markind < indextograb.size(); markind++)
        {
            hap11 = 0; hap12 = 0; hap21 = 0; hap22 = 0; freqsnp1 = 0; freqsnp2 = 0;
            for(int j = 0; j < n; j++)
            {
                /* add to haplotype frequencies */
                while(1)
                {
                    if(QTL[(j*mqtl)+q] == 0 && Marker[(j*mark)+indextograb[markind]] == 0){hap11 += 2; break;}
                    if(QTL[(j*mqtl)+q] == 0 && Marker[(j*mark)+indextograb[markind]] == 2){hap12 += 2; freqsnp2 += 2; break;}
                    if(QTL[(j*mqtl)+q] == 0 && Marker[(j*mark)+indextograb[markind]] == 3){hap11 += 1; hap12 += 1; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 0 && Marker[(j*mark)+indextograb[markind]] == 4){hap11 += 1; hap12 += 1; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 2 && Marker[(j*mark)+indextograb[markind]] == 0){hap21 += 2; freqsnp1 += 2; break;}
                    if(QTL[(j*mqtl)+q] == 2 && Marker[(j*mark)+indextograb[markind]] == 2){hap22 += 2; freqsnp1 += 2; freqsnp2 += 2; break;}
                    if(QTL[(j*mqtl)+q] == 2 && Marker[(j*mark)+indextograb[markind]] == 3){hap22 += 1; hap21 += 1; freqsnp1 += 2; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 2 && Marker[(j*mark)+indextograb[markind]] == 4){hap22 += 1; hap21 += 1; freqsnp1 += 2; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 3 && Marker[(j*mark)+indextograb[markind]] == 0){hap11 += 1; hap21 += 1; freqsnp1 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 3 && Marker[(j*mark)+indextograb[markind]] == 2){hap12 += 1; hap22 += 1; freqsnp1 += 1; freqsnp2 += 2; break;}
                    if(QTL[(j*mqtl)+q] == 3 && Marker[(j*mark)+indextograb[markind]] == 3){hap11 += 1; hap22 += 1; freqsnp1 += 1; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 3 && Marker[(j*mark)+indextograb[markind]] == 4){hap12 += 1; hap21 += 1; freqsnp1 += 1; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 4 && Marker[(j*mark)+indextograb[markind]] == 0){hap11 += 1; hap21 += 1; freqsnp1 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 4 && Marker[(j*mark)+indextograb[markind]] == 2){hap12 += 1; hap22 += 1; freqsnp1 += 1; freqsnp2 += 2; break;}
                    if(QTL[(j*mqtl)+q] == 4 && Marker[(j*mark)+indextograb[markind]] == 3){hap12 += 1; hap21 += 1; freqsnp1 += 1; freqsnp2 += 1; break;}
                    if(QTL[(j*mqtl)+q] == 4 && Marker[(j*mark)+indextograb[markind]] == 4){hap22 += 1; hap11 += 1; freqsnp1 += 1; freqsnp2 += 1; break;}
                }
            }
            /* Get frequencies */
            hap11 /= double(2*n); hap12 /= double(2 * n); hap21 /= double(2*n); hap22 /= double(2 * n);
            freqsnp1 /= double(2*n); freqsnp2 /= double(2 * n);
            if(hap11 != 0 && hap12 != 0 && hap21 != 0 && hap22 != 0 && (freqsnp1 > 0.0 && freqsnp1 < 1.0) && (freqsnp2 > 0.0 && freqsnp2 < 1.0))
            {
                //cout << distanceqtl[markind] << " " << markerpositionMb[indextograb[markind]] << " - ";
                //cout << "HapFreq: " << hap11 << " " << hap12 << " " << hap21 << " " << hap22 << " ";
                //cout << " SNPFreq: " << freqsnp1 << " " << freqsnp2 << " ";
                double D = ((hap11*hap22 ) - (hap12*hap21)) * ((hap11*hap22 ) - (hap12*hap21));
                double den = (freqsnp1*(1-freqsnp1)) *(freqsnp2*(1-freqsnp2));
                r2value[markind] = (D / den);
                rvalue[markind] = ((hap11*hap22 ) - (hap12*hap21)) / double(sqrt((freqsnp1*(1-freqsnp1)) *(freqsnp2*(1-freqsnp2))));
                //cout << "  D r2 r:" << ((hap11*hap22 ) - (hap12*hap21)) << " " << r2value[markind] << " " << rvalue[markind] << endl;
                //exit (EXIT_FAILURE);
                
            } else {r2value[markind] = -5; rvalue[markind] = -5;}
            //cout << " R2 Values: " << r2value[markind] << endl;
            QTL_outrfile.push_back(qtlposmb); diff_outrfile.push_back(distanceqtl[markind]);
            d_outrfile.push_back(rvalue[markind]);
        }
        numb1 = 0; sum1 = 0; numb2 = 0; sum2 = 0; numb3 = 0; sum3 = 0; numb4 = 0; sum4 = 0; numb5 = 0; sum5 = 0;
        for(int markind = 0 ; markind < indextograb.size(); markind++)
        {
            if(r2value[markind] != -5)
            {
                if((abs(distanceqtl[markind]) < 500000))
                {
                    numb1++; sum1 += r2value[markind]; group_outrfile.push_back(0);
                }
                if((abs(distanceqtl[markind]) >= 500000) && (abs(distanceqtl[markind]) < 1000000))
                {
                    numb2++; sum2 += r2value[markind]; group_outrfile.push_back(1);
                }
                if((abs(distanceqtl[markind]) >= 1000000) && (abs(distanceqtl[markind]) < 1500000))
                {
                    numb3++; sum3 += r2value[markind]; group_outrfile.push_back(2);
                }
                if((abs(distanceqtl[markind]) >= 1500000) && (abs(distanceqtl[markind]) < 2000000))
                {
                    numb4++; sum4 += r2value[markind]; group_outrfile.push_back(3);
                }
                if((abs(distanceqtl[markind]) >= 2000000) && (abs(distanceqtl[markind]) < 2500000))
                {
                    numb5++; sum5 += r2value[markind]; group_outrfile.push_back(4);
                }
            }
            if(r2value[markind] == -5){group_outrfile.push_back(5);}
        }
        //cout << QTL_outrfile.size() << " " << diff_outrfile.size() << " " << d_outrfile.size() << " " << group_outrfile.size() << endl;
        //for(int check = 0; check < QTL_outrfile.size(); check++)
        //{
        //    cout << QTL_outrfile[check] << " " << diff_outrfile[check] << " " << d_outrfile[check] << " " << group_outrfile[check] << endl;
        //}
        //exit (EXIT_FAILURE);
        //cout << sum1 << " " << sum2 << " " << sum3 << " " << sum4 << " " << sum5 << endl;
        if(numb1 > 0)
        {
            sum1 = sum1/double(numb1);
        } else(sum1 = -5);
        if(numb2 > 0)
        {
            sum2 = sum2/double(numb2);
        } else(sum2 = -5);
        if(numb3 > 0)
        {
            sum3 = sum3/double(numb3);
        } else(sum3 = -5);
        if(numb4 > 0)
        {
            sum4 = sum4/double(numb4);
        } else(sum4 = -5);
        if(numb5 > 0)
        {
            sum5 = sum5/double(numb5);
        } else(sum5 = -5);
        //cout << sum1 << " " << sum2 << " " << sum3 << " " << sum4 << " " << sum5 << endl;
        stringstream strStreamLD (stringstream::in | stringstream::out);
        strStreamLD << sum1; strStreamLD << ":"; strStreamLD << sum2; strStreamLD << ":"; strStreamLD << sum3; strStreamLD << ":";
        strStreamLD << sum4; strStreamLD << ":"; strStreamLD << sum5;
        string LDstring = strStreamLD.str();
        //cout << LDstring << endl;
        //cout << "'" << population_QTL[q].getLDDecay() << "'" << endl;
        if(foundergen == "no"){population_QTL[q].UpdateLDDecay(LDstring);}
        if(foundergen == "yes"){population_QTL[q].FounderLDDecay(LDstring);}
        //cout << "'" << population_QTL[q].getLDDecay() << "'" << endl;
        //exit (EXIT_FAILURE);
        //if(i >= 100){exit (EXIT_FAILURE);}
    }
    //time_t fullend_time = time(0);
    //cout << "Took: " << difftime(fullend_time,fullbegin_time) << " seconds" << endl << endl;
    delete [] QTL; delete [] Marker;
    ofstream output;
    output.open (QTL_LD_Decay_File.c_str());
    output << "Chr Pos R2" << endl;
    for(int i = 0; i < population_QTL.size(); i++)
    {
        /* Grab QTL and get position Mb */
        int qtlchr = int(population_QTL[i].getLocation());
        double qtlpos = population_QTL[i].getLocation() - qtlchr;
        int qtlposmb = qtlpos * (SimParameters.get_ChrLength())[qtlchr-1];
        output << qtlchr << " " << qtlposmb << " " << population_QTL[i].getLDDecay() << endl;
    }
    output.close();
    if(foundergen != "yes")
    {
        /* First loop through and see if some of the markers have not passed threshold in most recent generation */
        for(int i = 0; i < group_outrfile.size(); i++)
        {
            if(d_outrfile[i] == -5 && old_class_rfile[i] != 5){old_class_rfile[i] = 5;}
        }
        int previousgen = old_d_outrfile[0].size();
        vector < vector <double>> meanr(5,std::vector<double>(previousgen+1,0.0));
        vector < vector <double>> number(5,std::vector<double>(previousgen+1,0.0));
        vector < vector <double>> variance(5,std::vector<double>(previousgen+1,0.0));
        vector < vector <double>> covariance(5,std::vector<double>(previousgen,0.0));
        for(int i = 0; i < old_QTL_outrfile.size(); i++)
        {
            if(old_class_rfile[i] != 5)
            {
                if(old_QTL_outrfile[i] != QTL_outrfile[i] && old_diff_outrfile[i] != diff_outrfile[i])
                {
                    cout << old_QTL_outrfile[i] << " " << QTL_outrfile[i] << endl;
                    cout << old_diff_outrfile[i] << " " << diff_outrfile[i] << endl;
                    
                    cout << endl; cout << "Messed up line 819 " << endl; exit (EXIT_FAILURE);
                }
                //cout << old_class_rfile[i] << endl;
                for(int j = 0; j < old_d_outrfile[i].size(); j++)
                {
                    meanr[old_class_rfile[i]][j] += old_d_outrfile[i][j]; number[old_class_rfile[i]][j] += 1;
                    //cout << old_d_outrfile[i][j] << " ";
                }
                meanr[old_class_rfile[i]][old_d_outrfile[i].size()] += d_outrfile[i];
                number[old_class_rfile[i]][old_d_outrfile[i].size()] += 1;
                //cout << endl;
                //cout << d_outrfile[i] << endl;
            }
            //if(old_class_rfile[i] != 5)
            //{
            //    cout << i << endl;
            //    for(int check = 0; check < 5; check++)
            //    {
            //        for(int checka = 0; checka < (previousgen+1); checka++){cout << meanr[check][checka] << " ";}
            //        cout << endl;
            //    }
            //    cout << endl;
            //    for(int check = 0; check < 5; check++)
            //    {
            //        for(int checka = 0; checka < (previousgen+1); checka++){cout << number[check][checka] << " ";}
            //        cout << endl;
            //    }
            //    if(i > 100){exit (EXIT_FAILURE);}
            //}
        }
        /* Calculate Mean for each one */
        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < (previousgen+1); j++){meanr[i][j] /= number[i][j];}
        }
        //for(int check = 0; check < 5; check++)
        //{
        //    for(int checka = 0; checka < (previousgen+1); checka++){cout << meanr[check][checka] << " ";}
        //    cout << endl;
        //}
        //cout << endl;
        //for(int check = 0; check < 5; check++)
        //{
        //    for(int checka = 0; checka < (previousgen+1); checka++){cout << number[check][checka] << " ";}
        //    cout << endl;
        //}
        /* Calculate Variance and covariance for each class */
        for(int i = 0; i < old_QTL_outrfile.size(); i++)
        {
            if(old_class_rfile[i] != 5)
            {
                if(old_QTL_outrfile[i] != QTL_outrfile[i] && old_diff_outrfile[i] != diff_outrfile[i])
                {
                    cout << endl; cout << "Messed up line 819 " << endl; exit (EXIT_FAILURE);
                }
                //cout << old_class_rfile[i] << endl;
                for(int j = 0; j < old_d_outrfile[i].size(); j++)
                {
                    variance[old_class_rfile[i]][j] += (old_d_outrfile[i][j]-meanr[old_class_rfile[i]][j])*(old_d_outrfile[i][j]-meanr[old_class_rfile[i]][j]);
                    covariance[old_class_rfile[i]][j] += (old_d_outrfile[i][j]-meanr[old_class_rfile[i]][j])*(d_outrfile[i]-meanr[old_class_rfile[i]][old_d_outrfile[i].size()]);
                    //cout << old_d_outrfile[i][j] << " ";
                }
                variance[old_class_rfile[i]][old_d_outrfile[i].size()] += (d_outrfile[i]-meanr[old_class_rfile[i]][old_d_outrfile[i].size()])*(d_outrfile[i]-meanr[old_class_rfile[i]][old_d_outrfile[i].size()]);
                //cout << endl;
                //cout << d_outrfile[i] << endl;
            }
            //if(old_class_rfile[i] != 5)
            //{
            //    cout << i << endl;
            //    for(int check = 0; check < 5; check++)
            //    {
            //        for(int checka = 0; checka < (previousgen+1); checka++){cout << meanr[check][checka] << " ";}
            //        cout << endl;
            //    }
            //    cout << endl;
            //    for(int check = 0; check < 5; check++)
            //    {
            //        for(int checka = 0; checka < (previousgen+1); checka++){cout << variance[check][checka] << " ";}
            //        cout << endl;
            //    }
            //    cout << endl;
            //    for(int check = 0; check < 5; check++)
            //    {
            //        for(int checka = 0; checka < (previousgen); checka++){cout << covariance[check][checka] << " ";}
            //        cout << endl;
            //    }
            //    cout << endl;
            //    for(int check = 0; check < 5; check++)
            //    {
            //        for(int checka = 0; checka < (previousgen+1); checka++){cout << number[check][checka] << " ";}
            //        cout << endl;
            //    }
            //    //exit (EXIT_FAILURE);
            //    //if(i > 50){exit (EXIT_FAILURE);}
            //}
        }
        /* Calculate sd for each one */
        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < (previousgen+1); j++){variance[i][j] = sqrt(variance[i][j] / double(number[i][j]-1));}
        }
        /* Calculate covariance for each one */
        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < (previousgen); j++){covariance[i][j] /=  double(number[i][j]-1);}
        }
        //for(int check = 0; check < 5; check++)
        //{
        //    for(int checka = 0; checka < (previousgen+1); checka++){cout << meanr[check][checka] << " ";}
        //    cout << endl;
        //}
        //cout << endl;
        //for(int check = 0; check < 5; check++)
        //{
        //    for(int checka = 0; checka < (previousgen+1); checka++){cout << variance[check][checka] << " ";}
        //    cout << endl;
        //}
        //cout << endl;
        //for(int check = 0; check < 5; check++)
        //{
        //    for(int checka = 0; checka < (previousgen); checka++){cout << covariance[check][checka] << " ";}
        //    cout << endl;
        //}
        //cout << endl;
        //for(int check = 0; check < 5; check++)
        //{
        //    for(int checka = 0; checka < (previousgen+1); checka++){cout << number[check][checka] << " ";}
        //    cout << endl;
        //}
        //for(int i = 0; i < 5; i++)
        //{
        //    for(int j = 0; j < (previousgen); j++)
        //    {
        //        cout << covariance[i][j] / double(variance[i][j]*variance[i][previousgen]) << " ";
        //    }
        //    cout << endl;
        //}
        /* Calcuate phase correlation between generations */
        vector < string > acrossgenerationPhaseCor(SimParameters.getGener(),"-");
        for(int i = 0; i < previousgen; i++)
        {
            stringstream strStreamPhaseCor (stringstream::in | stringstream::out);
            for(int j = 0; j < 5; j++)
            {
                if (j == 0)
                {
                    strStreamPhaseCor << covariance[j][i] / double(variance[j][i]*variance[j][previousgen]);
                } else {
                    strStreamPhaseCor << ":" << covariance[j][i] / double(variance[j][i]*variance[j][previousgen]);
                }
            }
            //cout << strStreamPhaseCor.str() << endl;
            acrossgenerationPhaseCor[i] = strStreamPhaseCor.str();
        }
        std::ofstream outPhasePers(Phase_Persistance_Outfile.c_str(), std::ios_base::app | std::ios_base::out);
        outPhasePers << generation[0];
        for(int i = 0; i < acrossgenerationPhaseCor.size(); i++){outPhasePers << " " << acrossgenerationPhaseCor[i];}
        outPhasePers << endl;
        ofstream output1;
        output1.open (Phase_Persistance_File.c_str());
        for(int i = 0; i < old_QTL_outrfile.size(); i++)
        {
            output1 << old_QTL_outrfile[i] << " " << old_diff_outrfile[i] << " " << old_class_rfile[i] << " ";
            for(int j = 0; j < old_d_outrfile[i].size(); j++){output1 << old_d_outrfile[i][j] << " ";}
            output1 << d_outrfile[i] << endl;
        }
        output1.close();
    }
    if(foundergen == "yes")
    {
        //cout << QTL_outrfile.size() << " " << diff_outrfile.size() << " " << d_outrfile.size() << endl;
        ofstream output1;
        output1.open (Phase_Persistance_File.c_str());
        for(int i = 0; i < QTL_outrfile.size(); i++)
        {
            output1 << QTL_outrfile[i] << " " << diff_outrfile[i] << " " << group_outrfile[i] << " " << d_outrfile[i] << endl;
        }
        output1.close();
    }
}

