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
#include "Global_Population.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Class Functions       ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
globalpopvar::globalpopvar(){}
globalpopvar::~globalpopvar(){}
void globalpopvar::update_qtlscalefactadd(int i, double x){qtlscalefactadd[i] = x;}
void globalpopvar::update_qtlscalefactdom(int i, double x){qtlscalefactdom[i] = x;}
void globalpopvar::update_snpfiles(int i, std::string x){snpfiles[i] = x;}
void globalpopvar::update_mapfiles(int i, std::string x){mapfiles[i] = x;}
void globalpopvar::update_chrsnplength(int i, int x){chrsnplength[i] = x;}
void globalpopvar::Updatefullmarkernum(int temp){fullmarkernum = temp;}
void globalpopvar::update_markerindex(int i, int x){markerindex[i] = x;}
void globalpopvar::update_markermapposition(int i, double x){markermapposition[i] = x;}
void globalpopvar::update_qtlindex(int i, int x){qtlindex[i] = x;}
void globalpopvar::update_mapposition(int i, double x){qtl_mapposition[i] = x;}
void globalpopvar::update_qtl_type(int i, int x){qtl_type[i] = x;}
void globalpopvar::update_qtl_allele(int i, int x){qtl_allele[i] = x;}
void globalpopvar::update_qtl_freq(int i, double x){qtl_freq[i] = x;}
void globalpopvar::update_qtl_add_fit(int i, double x){qtl_add_fit[i] = x;}
void globalpopvar::update_qtl_dom_fit(int i, double x){qtl_dom_fit[i] = x;}
void globalpopvar::update_markerperchr(int i, int x){markerperchr[i] = x;}
void globalpopvar::update_qtlperchr(int i, int x){qtlperchr[i] = x;}
void globalpopvar::update_numdeadfitness(int i, int x){numdeadfitness[i] = x;}
void globalpopvar::update_expectedheter(int i, double x){expectedheter[i] = x;}
void globalpopvar::update_marker_chr(int i, double x){marker_chr[i] = x;}
void globalpopvar::update_marker_posmb(int i, double x){marker_posmb[i] = x;}
void globalpopvar::update_accuracydeltaG(int i, double x){accuracydeltaG[i] = x;}
void globalpopvar::update_sdGen0_EBV(int i, double x){sdGen0_EBV[i] = x;}
void globalpopvar::update_sdGen0_TBV(int i, double x){sdGen0_TBV[i] = x;}
void globalpopvar::update_QTLFreq_AcrossGen(int i, double x){QTLFreq_AcrossGen[i] += x;}
void globalpopvar::UpdateQTLFreq_Number(int temp){QTLFreq_Number += temp;}

void initializevectors(globalpopvar &Population1, parameters &SimParameters)
{
    if((SimParameters.get_Var_Dominance())[0] == 0.0){Population1.add_qtlscalefactdom(0.0);}    /* scaling factor for dominance set to 0 */
    if((SimParameters.get_Var_Dominance())[0] > 0.0){Population1.add_qtlscalefactdom(1.0);}     /* scaling factor for dominance set to 1.0 and will change */
    if((SimParameters.get_Var_Additive())[0] == 0.0){Population1.add_qtlscalefactadd(0.0);}     /* scaling factor for additive set to 0 */
    if((SimParameters.get_Var_Additive())[0] > 0.0){Population1.add_qtlscalefactadd(1.0);}      /* scaling factor for additive set to 1.0 and will change */
    if(SimParameters.getnumbertraits() == 2)
    {
        if((SimParameters.get_Var_Dominance())[2] == 0.0){Population1.add_qtlscalefactdom(0.0);}    /* scaling factor for dominance set to 0 */
        if((SimParameters.get_Var_Dominance())[2] > 0.0){Population1.add_qtlscalefactdom(1.0);}     /* scaling factor for dominance set to 1.0 and will change */
        if((SimParameters.get_Var_Additive())[2] == 0.0){Population1.add_qtlscalefactadd(0.0);}     /* scaling factor for additive set to 0 */
        if((SimParameters.get_Var_Additive())[2] > 0.0){Population1.add_qtlscalefactadd(1.0);}      /* scaling factor for additive set to 1.0 and will change */
    }
    for(int i = 0; i < SimParameters.getChr(); i++)
    {
        Population1.add_snpfiles(""); Population1.add_mapfiles(""); Population1.add_chrsnplength(0);
        Population1.add_markerperchr(0); Population1.add_qtlperchr(0);
    }
    int tempmark = 0;
    for(int i = 0; i < SimParameters.getChr(); i++){tempmark += (SimParameters.get_Marker_chr())[i];}
    Population1.Updatefullmarkernum(tempmark);
    for(int i = 0; i < 5000; i++)
    {
        Population1.add_qtlindex(0); Population1.add_qtl_mapposition(0.0); Population1.add_qtl_type(0); Population1.add_qtl_allele(0);
        Population1.add_qtl_freq(0); Population1.addrow_qtl_add_quan(); Population1.addrow_qtl_dom_quan();
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            Population1.addcol_qtl_add_quan(i); Population1.addcol_qtl_dom_quan(i);
        }
        Population1.add_qtl_add_fit(0.0); Population1.add_qtl_dom_fit(0.0);
    }
    for(int i = 0; i < (SimParameters.getGener()+1); i++)
    {
        Population1.add_numdeadfitness(0); Population1.add_expectedheter(0.0); Population1.add_accuracydeltaG(0.0);
        Population1.addrow_qtl_additivevar(); Population1.addrow_qtl_dominancevar();
        Population1.addrow_Intensity_Males(); Population1.addrow_Intensity_Females();
        Population1.addrow_GenInterval_Males(); Population1.addrow_GenInterval_Females();
        for(int j = 0; j < SimParameters.getnumbertraits(); j++)
        {
            Population1.addcol_qtl_additivevar(i); Population1.addcol_qtl_dominancevar(i);
            Population1.addcol_Intensity_Males(i); Population1.addcol_Intensity_Females(i);
            Population1.addcol_GenInterval_Males(i); Population1.addcol_GenInterval_Females(i);
        }
    }
    for(int i = 0; i < SimParameters.getnumbertraits(); i++){Population1.add_sdGen0_EBV(0.0); Population1.add_sdGen0_TBV(0.0);}
    //cout << (Population1.get_qtlscalefactadd()).size() << " " << (Population1.get_qtlscalefactadd())[0] << endl;
    //cout << (Population1.get_qtlscalefactdom()).size() << " " << (Population1.get_qtlscalefactdom())[0] << endl;
    //cout << (Population1.get_snpfiles()).size() << endl;
    //cout << (Population1.get_mapfiles()).size() << endl;
    //cout << (Population1.get_chrsnplength()).size() << endl;
    //cout << Population1.getfullmarkernum() << endl;
    //cout << (Population1.get_markerindex()).size() << endl;
    //cout << (Population1.get_markermapposition()).size() << endl;
    //cout << (Population1.get_qtlindex()).size() << endl;
    //cout << (Population1.get_qtl_mapposition()).size() << endl;
    //cout << (Population1.get_qtl_type()).size() << endl;
    //cout << (Population1.get_qtl_allele()).size() << endl;
    //cout << (Population1.get_qtl_freq()).size() << endl;
    //cout << (Population1.getrow_qtl_add_quan()) << " " << (Population1.getcol_qtl_add_quan()) << endl;
    //cout << (Population1.getrow_qtl_dom_quan()) << " " << (Population1.getcol_qtl_dom_quan()) << endl;
    //cout << (Population1.get_markerperchr()).size() << endl;
    //cout << (Population1.get_qtlperchr()).size() << endl;
    //cout << (Population1.get_numdeadfitness()).size() << endl;
    //cout << (Population1.get_expectedheter()).size() << endl;
    //cout << (Population1.getrow_qtl_additivevar()) << " " << (Population1.getcol_qtl_additivevar()) << endl;
    //cout << (Population1.getrow_qtl_dominancevar()) << " " << (Population1.getcol_qtl_dominancevar()) << endl;
    //cout << (Population1.getrow_Intensity_Females()) << " " << (Population1.getrow_Intensity_Males()) << endl;
    //cout << (Population1.getrow_GenInterval_Females()) << " " << (Population1.getrow_GenInterval_Males()) << endl;
    //(Population1.get_markerindex()).size() << " " << (Population1.get_markerindex()).[i] << " " << Population1.update_markerindex(i,x)
    //(Population1.get_markermapposition()).size() << " " << (Population1.get_markermapposition()).[i] << " " << Population1.update_mapposition(i,x)
    //Population1.update_markerperchr(c,test_mark) << " " << Population1.get_markerperchr()[c]
    //Population1.update_qtlperchr(c,test_mark) << " " << Population1.get_qtlperchr()[c]
    //(Population1.get_qtlindex()).size()<< " " <<(Population1.get_qtl_mapposition()).size()
    //Population1.get_qtl_add_quan(i,0)<< " " <<Population1.update_qtl_add_quan(int i, int j,double temp)
    //Population1.get_qtl_dom_quan(i,0)<< " " <<Population1.update_qtl_dom_quan(int i, int j,double temp)
    //Population1.get_qtl_add_fit())[QTL_IndCounter]<< " " <<Population1.update_qtl_add_fit(i,x)
    //(Population1.get_qtl_dom_fit())[QTL_IndCounter]<< " " <<Population1.update_qtl_dom_fit(i,x)
    //cout << (Population1.get_qtlindex())[QTL_IndCounter] << " " << (Population1.get_qtl_mapposition())[QTL_IndCounter] << " ";
    //cout << (Population1.get_qtl_type())[QTL_IndCounter] << " " << (Population1.get_qtl_freq())[QTL_IndCounter] << " ";
    //cout << (Population1.get_qtl_allele())[QTL_IndCounter] << " " << Population1.get_qtl_add_quan(QTL_IndCounter,0) << " ";
    //cout << Population1.get_qtl_dom_quan(QTL_IndCounter,0) << " " << (Population1.get_qtl_add_fit())[QTL_IndCounter] << " ";
    //cout << (Population1.get_qtl_dom_fit())[QTL_IndCounter] << endl;
}
