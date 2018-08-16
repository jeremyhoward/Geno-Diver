#include <iostream>
#include <vector>


#ifndef Global_Population_H_
#define Global_Population_H_

#include "Animal.h"
#include "ParameterClass.h"


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// Class to parameters used in the simulation program //
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
class globalpopvar
{
private:
    std::vector<double> qtlscalefactadd = std::vector<double>(0);       /* Scale factor for additive QTL variance */
    std::vector<double> qtlscalefactdom = std::vector<double>(0);       /* Scale factor for dominance QTL variance */
    std::vector<std::string> snpfiles = std::vector<std::string>(0);    /* Name of sequence snp file for a chromosome */
    std::vector<std::string> mapfiles = std::vector<std::string>(0);    /* Name of sequence map file for a chromosome */
    std::vector<int> chrsnplength = std::vector<int>(0);                /* Number of SNP within each chromosome */
    int fullmarkernum = 0;                                              /* Markers across all chromosomes */
    std::vector<int> markerindex = std::vector<int>(0);                 /* Index of markers in combined set */
    std::vector<double> markermapposition = std::vector<double>(0);     /* Marker map position (i.e 1.234) */
    std::vector<int> qtlindex = std::vector<int>(0);                    /* Index of qtl/ftl in combined set */
    std::vector<double> qtl_mapposition = std::vector<double>(0);       /* Qtl map position (i.e 1.234) */
    std::vector<int> qtl_type = std::vector<int>(0);                    /* Indicator: quantitative, lethal, sublethal or both */
    std::vector<int> qtl_allele = std::vector<int>(0);                  /* Which allele it is referring to */
    std::vector<double> qtl_freq = std::vector<double>(0);              /* Freq of QTL/FTL */
    std::vector <std::vector<double>> qtl_add_quan ;                    /* Additive Effect quantitative; column for each trait */
    std::vector<std::vector<double>> qtl_dom_quan;                      /* Dominance Effect quantitative; column for each trait */
    std::vector<double> qtl_add_fit = std::vector<double>(0);           /* Selection coefficent fitness */
    std::vector<double> qtl_dom_fit = std::vector<double>(0);           /* Degree of dominance fitness */
    std::vector<int> markerperchr = std::vector<int>(0);                /* Number of Markers for each chromosome */
    std::vector<int> qtlperchr = std::vector<int>(0);                   /* Number of QTL for each chromosome */
    std::vector<int> numdeadfitness = std::vector<int>(0);              /* Number of dead due to fitness by generation */
    std::vector<double> expectedheter = std::vector<double>(0);         /* Marker Expected Heterozygosity: (1 - p^2 - q^2) */
    std::vector<std::vector<double>> additivevar;                       /* Additive variance by Generation (sum of 2pqa^2); column for each trait */
    std::vector<std::vector<double>> dominancevar;                      /* Dominance variance by Generation (sum of (2pqd)^2); column for each trait */
    /* Standard deviation in EBV and TBV at Gen 0; used to standardize index traits */
    std::vector<double> sdGen0_EBV = std::vector<double>(0);            /* Standard Deviation EBV Founder Generation */
    std::vector<double> sdGen0_TBV = std::vector<double>(0);            /* Standard Deviation TBV Founder Generation */
    /* Delta G Summary Statistics */
    std::vector<double> accuracydeltaG;                                 /* accuracy of selection (i.e. for blup based values) */
    std::vector<std::vector<double>> Intensity_Males;                   /* Selection Intensity for Males */
    std::vector<std::vector<double>> Intensity_Females;                 /* Selection Intensity for Females */
    std::vector<std::vector<double>> GenInterval_Males;                 /* Generation Interval Males */
    std::vector<std::vector<double>> GenInterval_Females;               /* Generation Interval Females */
    /* Used to Generate Marker Map */
    std::vector<int> marker_chr = std::vector<int>(0);                  /* Marker file chromosome */
    std::vector<int> marker_posmb = std::vector<int>(0);                /* Marker file position */
    /* Across generations overall sum of genotypes*/
    std::vector<double> QTLFreq_AcrossGen = std::vector<double>(0);     /* Freq of QTL/FTL across generations */
    int QTLFreq_Number = 0;                                             /* Number of individuals across generations */
public:
    globalpopvar();
    ~globalpopvar();
    /* Scale factor for additive QTL variance */
    const std::vector <double>& get_qtlscalefactadd() const{return qtlscalefactadd;}
    void add_qtlscalefactadd(double x){qtlscalefactadd.push_back(x);}
    void update_qtlscalefactadd(int i, double x);
    void clear_qtlscalefactadd(){qtlscalefactadd.clear();}
    /* Scale factor for dominance QTL variance */
    const std::vector <double>& get_qtlscalefactdom() const{return qtlscalefactdom;}
    void add_qtlscalefactdom(double x){qtlscalefactdom.push_back(x);}
    void update_qtlscalefactdom(int i, double x);
    void clear_qtlscalefactdom(){qtlscalefactdom.clear();}
    /* Name of sequence snp file for a chromosome */
    const std::vector <std::string>& get_snpfiles() const{return snpfiles;}
    void add_snpfiles(std::string x){snpfiles.push_back(x);}
    void update_snpfiles(int i, std::string x);
    void clear_snpfiles(){snpfiles.clear();}
    /* Name of sequence map file for a chromosome */
    const std::vector <std::string>& get_mapfiles() const{return mapfiles;}
    void add_mapfiles(std::string x){mapfiles.push_back(x);}
    void update_mapfiles(int i, std::string x);
    void clear_mapfiles(){mapfiles.clear();}
    /* Number of SNP within each chromosome */
    const std::vector <int>& get_chrsnplength() const{return chrsnplength;}
    void add_chrsnplength(int x){chrsnplength.push_back(x);}
    void update_chrsnplength(int i, int x);
    void clear_chrsnplength(){chrsnplength.clear();}
    /* Markers across all chromosomes */
    int getfullmarkernum(){return fullmarkernum;}
    void Updatefullmarkernum(int temp);
    /* Index of markers in combined set */
    const std::vector <int>& get_markerindex() const{return markerindex;}
    void add_markerindex(int x){markerindex.push_back(x);}
    void update_markerindex(int i, int x);
    void clear_markerindex(){markerindex.clear();}
    /* Marker map position (i.e 1.234) */
    const std::vector <double>& get_markermapposition() const{return markermapposition;}
    void add_markermapposition(double x){markermapposition.push_back(x);}
    void update_markermapposition(int i, double x);
    void clear_markermapposition(){markermapposition.clear();}
    /* Index of qtl/ftl in combined set */
    const std::vector <int>& get_qtlindex() const{return qtlindex;}
    void add_qtlindex(int x){qtlindex.push_back(x);}
    void update_qtlindex(int i, int x);
    void clear_qtlindex(){qtlindex.clear();}
    /* Qtl map position (i.e 1.234) */
    const std::vector <double>& get_qtl_mapposition() const{return qtl_mapposition;}
    void add_qtl_mapposition(double x){qtl_mapposition.push_back(x);}
    void update_mapposition(int i, double x);
    void clear_qtl_mapposition(){qtl_mapposition.clear();}
    /* Indicator: quantitative, lethal, sublethal or both */
    const std::vector <int>& get_qtl_type() const{return qtl_type;}
    void add_qtl_type(int x){qtl_type.push_back(x);}
    void update_qtl_type(int i, int x);
    void clear_qtl_type(){qtl_type.clear();}
    /* Which allele it is referring to */
    const std::vector <int>& get_qtl_allele() const{return qtl_allele;}
    void add_qtl_allele(int x){qtl_allele.push_back(x);}
    void update_qtl_allele(int i, int x);
    void clear_qtl_allele(){qtl_allele.clear();}
    /* Freq of QTL/FTL */
    const std::vector <double>& get_qtl_freq() const{return qtl_freq;}
    void add_qtl_freq(double x){qtl_freq.push_back(x);}
    void update_qtl_freq(int i, double x);
    void clear_qtl_freq(){qtl_freq.clear();}
    /* Additive Effect quantitative; column for each trait */
    void addrow_qtl_add_quan(){qtl_add_quan.push_back(std::vector<double>(0));}
    void addcol_qtl_add_quan(int i){qtl_add_quan[i].push_back(0);}
    void update_qtl_add_quan(int i, int j,double temp){qtl_add_quan[i][j] = temp;}
    double get_qtl_add_quan(int i, int j){return qtl_add_quan[i][j];}
    int getrow_qtl_add_quan(){return qtl_add_quan.size();}
    int getcol_qtl_add_quan(){return qtl_add_quan[0].size();}
    void clear_qtl_add_quan(){qtl_add_quan.clear();}
    /* Dominance Effect quantitative; column for each trait */
    void addrow_qtl_dom_quan(){qtl_dom_quan.push_back(std::vector<double>(0));}
    void addcol_qtl_dom_quan(int i){qtl_dom_quan[i].push_back(0);}
    void update_qtl_dom_quan(int i, int j,double temp){qtl_dom_quan[i][j] = temp;}
    double get_qtl_dom_quan(int i, int j){return qtl_dom_quan[i][j];}
    int getrow_qtl_dom_quan(){return qtl_dom_quan.size();}
    int getcol_qtl_dom_quan(){return qtl_dom_quan[0].size();}
    void clear_qtl_dom_quan(){qtl_dom_quan.clear();}
    /* Selection coefficent fitness */
    const std::vector <double>& get_qtl_add_fit() const{return qtl_add_fit;}
    void add_qtl_add_fit(double x){qtl_add_fit.push_back(x);}
    void update_qtl_add_fit(int i, double x);
    void clear_qtl_add_fit(){qtl_add_fit.clear();}
    /* Degree of dominance fitness */
    const std::vector <double>& get_qtl_dom_fit() const{return qtl_dom_fit;}
    void add_qtl_dom_fit(double x){qtl_dom_fit.push_back(x);}
    void update_qtl_dom_fit(int i, double x);
    void clear_qtl_dom_fit(){qtl_dom_fit.clear();}
    /* Number of Markers for each chromosome */
    const std::vector <int>& get_markerperchr() const{return markerperchr;}
    void add_markerperchr(int x){markerperchr.push_back(x);}
    void update_markerperchr(int i, int x);
    void clear_markerperchr(){markerperchr.clear();}
    /* Number of QTL for each chromosome */
    const std::vector <int>& get_qtlperchr() const{return qtlperchr;}
    void add_qtlperchr(int x){qtlperchr.push_back(x);}
    void update_qtlperchr(int i, int x);
    void clear_qtlperchr(){qtlperchr.clear();}
    /* Number of dead due to fitness by generation */
    const std::vector <int>& get_numdeadfitness() const{return numdeadfitness;}
    void add_numdeadfitness(int x){numdeadfitness.push_back(x);}
    void update_numdeadfitness(int i, int x);
    void clear_numdeadfitness(){numdeadfitness.clear();}
    /* Marker Expected Heterozygosity: (1 - p^2 - q^2) */
    const std::vector <double>& get_expectedheter() const{return expectedheter;}
    void add_expectedheter(double x){expectedheter.push_back(x);}
    void update_expectedheter(int i, double x);
    void clear_expectedheter(){expectedheter.clear();}
    /* Standard Deviation EBV Founder Generation */
    const std::vector <double>& get_sdGen0_EBV() const{return sdGen0_EBV;}
    void add_sdGen0_EBV(double x){sdGen0_EBV.push_back(x);}
    void update_sdGen0_EBV(int i, double x);
    void clear_sdGen0_EBV(){sdGen0_EBV.clear();}
    /* Standard Deviation TBV Founder Generation */
    const std::vector <double>& get_sdGen0_TBV() const{return sdGen0_TBV;}
    void add_sdGen0_TBV(double x){sdGen0_TBV.push_back(x);}
    void update_sdGen0_TBV(int i, double x);
    void clear_sdGen0_TBV(){sdGen0_TBV.clear();}
    /* Additive variance by Generation (sum of 2pqa^2); column for each trait */
    void addrow_qtl_additivevar(){additivevar.push_back(std::vector<double>(0));}
    void addcol_qtl_additivevar(int i){additivevar[i].push_back(0);}
    void update_qtl_additivevar(int i, int j,double temp){additivevar[i][j] = temp;}
    double get_qtl_additivevar(int i, int j){return additivevar[i][j];}
    int getrow_qtl_additivevar(){return additivevar.size();}
    int getcol_qtl_additivevar(){return additivevar[0].size();}
    void clear_qtl_additivevar(){additivevar.clear();}
    /* Dominance variance by Generation (sum of (2pqd)^2); column for each trait */
    void addrow_qtl_dominancevar(){dominancevar.push_back(std::vector<double>(0));}
    void addcol_qtl_dominancevar(int i){dominancevar[i].push_back(0);}
    void update_qtl_dominancevar(int i, int j,double temp){dominancevar[i][j] = temp;}
    double get_qtl_dominancevar(int i, int j){return dominancevar[i][j];}
    int getrow_qtl_dominancevar(){return dominancevar.size();}
    int getcol_qtl_dominancevar(){return dominancevar[0].size();}
    void clear_qtl_dominancevar(){dominancevar.clear();}
    /* accuracy of selection (i.e. for blup based values) */
    const std::vector <double>& get_accuracydeltaG() const{return accuracydeltaG;}
    void add_accuracydeltaG(double x){accuracydeltaG.push_back(x);}
    void update_accuracydeltaG(int i, double x);
    void clear_accuracydeltaG(){accuracydeltaG.clear();}
    /* Selection Intensity for Males */
    void addrow_Intensity_Males(){Intensity_Males.push_back(std::vector<double>(0));}
    void addcol_Intensity_Males(int i){Intensity_Males[i].push_back(0);}
    void update_Intensity_Males(int i, int j,double temp){Intensity_Males[i][j] = temp;}
    double get_Intensity_Males(int i, int j){return Intensity_Males[i][j];}
    int getrow_Intensity_Males(){return Intensity_Males.size();}
    int getcol_Intensity_Males(){return Intensity_Males[0].size();}
    void clear_Intensity_Males(){Intensity_Males.clear();}
    /* Selection Intensity for Females */
    void addrow_Intensity_Females(){Intensity_Females.push_back(std::vector<double>(0));}
    void addcol_Intensity_Females(int i){Intensity_Females[i].push_back(0);}
    void update_Intensity_Females(int i, int j,double temp){Intensity_Females[i][j] = temp;}
    double get_Intensity_Females(int i, int j){return Intensity_Females[i][j];}
    int getrow_Intensity_Females(){return Intensity_Females.size();}
    int getcol_Intensity_Females(){return Intensity_Females[0].size();}
    void clear_Intensity_Females(){Intensity_Females.clear();}
    /* Generation Interval Males */
    void addrow_GenInterval_Males(){GenInterval_Males.push_back(std::vector<double>(0));}
    void addcol_GenInterval_Males(int i){GenInterval_Males[i].push_back(0);}
    void update_GenInterval_Males(int i, int j,double temp){GenInterval_Males[i][j] = temp;}
    double get_GenInterval_Males(int i, int j){return GenInterval_Males[i][j];}
    int getrow_GenInterval_Males(){return GenInterval_Males.size();}
    int getcol_GenInterval_Males(){return GenInterval_Males[0].size();}
    void clear_GenInterval_Males(){GenInterval_Males.clear();}
    /* Generation Interval Females */
    void addrow_GenInterval_Females(){GenInterval_Females.push_back(std::vector<double>(0));}
    void addcol_GenInterval_Females(int i){GenInterval_Females[i].push_back(0);}
    void update_GenInterval_Females(int i, int j,double temp){GenInterval_Females[i][j] = temp;}
    double get_GenInterval_Females(int i, int j){return GenInterval_Females[i][j];}
    int getrow_GenInterval_Females(){return GenInterval_Females.size();}
    int getcol_GenInterval_Females(){return GenInterval_Females[0].size();}
    void clear_GenInterval_Females(){GenInterval_Females.clear();}
    /* Marker file chromosome */
    const std::vector <int>& get_marker_chr() const{return marker_chr;}
    void add_marker_chr(int x){marker_chr.push_back(x);}
    void update_marker_chr(int i, double x);
    void clear_marker_chr(){marker_chr.clear();}
    /* Marker file position */
    const std::vector <int>& get_marker_posmb() const{return marker_posmb;}
    void add_marker_posmb(int x){marker_posmb.push_back(x);}
    void update_marker_posmb(int i, double x);
    void clear_marker_posmb(){marker_posmb.clear();}
    /* Freq of QTL/FTL across generation */
    const std::vector <double>& get_QTLFreq_AcrossGen() const{return QTLFreq_AcrossGen;}
    void add_QTLFreq_AcrossGen(double x){QTLFreq_AcrossGen.push_back(x);}
    void update_QTLFreq_AcrossGen(int i, double x);
    void clear_QTLFreq_AcrossGen(){QTLFreq_AcrossGen.clear();}
    /* Number of individuals across generations */
    int getQTLFreq_Number(){return QTLFreq_Number;}
    void UpdateQTLFreq_Number(int temp);
};

#endif
    
