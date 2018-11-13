#include <iostream>
#include <vector>
#include <string>
#ifndef OutputFiles_H_
#define OutputFiles_H_

using namespace std;
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// Class for OutputFiles used in the simulation program //
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
class outputfiles
{
private:
    string loc_lowfitnesspath  = "";        /* Stores all the animals that died */
    string loc_snpfreqfile = "";            /* Stores Frequency file for sequence information */
    string loc_foundergenofile = "";        /* Stores sequence file for founder genotypes */
    string loc_qtl_class_object = "";       /* All qtl/ftl information */
    string loc_Pheno_Pedigree = "";         /* Stores pedigree relationship input */
    string loc_Pheno_GMatrix = "";          /* Stores genomic relationship input (deletes at end of program)*/
    string loc_Temporary_EBV = "";          /* Stores current ebvs for all animals (deletes at end of program)*/
    string loc_Pheno_GMatrixImp = "";       /* Stores genomic relationship input for imputated matrix (deletes at end of program)*/
    string loc_Master_DF = "";              /* Stores full data for an animal */
    string loc_Master_Genotype = "";        /* Stores full genotype for an animal (deletes at end of program)*/
    string loc_Master_Genotype_zip = "";    /* Stores full genotype for an animal in compressed format */
    string loc_BinaryG_Matrix = "";         /* Stores genomic relationship in binary format */
    string loc_Binarym_Matrix = "";         /* Stores genomic relationship inverse matrix */
    string loc_Binaryp_Matrix = "";         /* Stores genomic relationship inverse matrix */
    string loc_BinaryLinv_Matrix = "";      /* Stores genomic relationship inverse matrix */
    string loc_BinaryGinv_Matrix = "";      /* Stores genomic relationship inverse matrix */
    string loc_Marker_Map = "";             /* Stores marker file */
    string loc_Master_DataFrame = "";       /* Stores final dataframe file across all animals */
    string loc_Summary_QTL = "";            /* Stores summary qtl statistics */
    string loc_Summary_DF = "";             /* Stores summary statistics for an animal */
    string loc_GenotypeStatus = "";         /* Stores genotype status for an animal */
    string loc_ExpRealDeltaG = "";          /* Stores expected and realized DeltaG */
    /* If doing LD based summary statistics will drop in these files */
    string loc_LD_Decay = "";               /* LD Decay across the genome */
    string loc_QTL_LD_Decay = "";           /* QTL LD Decay across the genome */
    string loc_Phase_Persistance = "";          /* Phase Persistance */
    string loc_Phase_Persistance_Outfile = "";  /* Phase Persistance */
    /* if doing haplotype finder will drop in this file summary statistics */
    string loc_Summary_Haplofinder = "";    /* Haplofinder Summary */
    /* if doing ROH genome summary will drop in this file with summary statistics of frequency and length */
    string loc_Summary_ROHGenome_Freq = "";     /* ROH Frequency across Genome */
    string loc_Summary_ROHGenome_Length = "";   /* ROH length across Genome */
    /* if doing bayesian ebv based prediction save mcmc samples */
    string loc_Bayes_MCMC_Samples = "";     /* Bayes MCMC Samples */
    string loc_Bayes_PosteriorMeans = "";   /* Bayes Posterior Means for a given generation */
    /* if want Amax by generation from most recent generation save to AmaxGeneration */
    string loc_Amax_Output = "";            /* Amax across generations */
    string loc_TraitReference_Output = "";  /* Output ebv each generation so can split into geno versus non-genotyped */
    /* if want window true additive and dominance variance across genome */
    string loc_Windowadditive_Output = "";  /* Window additive Output */
    string loc_Windowdominance_Output = ""; /* Window dominance Output */
public:
    outputfiles();
    ~outputfiles();
    /* loc_lowfitnesspath */
    std::string getloc_lowfitnesspath(){return loc_lowfitnesspath;}
    void Updateloc_lowfitnesspath(std::string temp);
    /* loc_snpfreqfile */
    std::string getloc_snpfreqfile(){return loc_snpfreqfile;}
    void Updateloc_snpfreqfile(std::string temp);
    /* loc_foundergenofile */
    std::string getloc_foundergenofile(){return loc_foundergenofile;}
    void Updateloc_foundergenofile(std::string temp);
    /* loc_qtl_class_object */
    std::string getloc_qtl_class_object(){return loc_qtl_class_object;}
    void Updateloc_qtl_class_object(std::string temp);
    /* loc_Pheno_Pedigree */
    std::string getloc_Pheno_Pedigree(){return loc_Pheno_Pedigree;}
    void Updateloc_Pheno_Pedigreet(std::string temp);
    /* loc_Pheno_GMatrix */
    std::string getloc_Pheno_GMatrix(){return loc_Pheno_GMatrix;}
    void Updateloc_Pheno_GMatrix(std::string temp);
    /* loc_Temporary_EBV */
    std::string getloc_Temporary_EBV(){return loc_Temporary_EBV;}
    void Updateloc_Temporary_EBV(std::string temp);
    /* loc_Pheno_GMatrixImp */
    std::string getloc_Pheno_GMatrixImp(){return loc_Pheno_GMatrixImp;}
    void Updateloc_Pheno_GMatrixImp(std::string temp);
    /* loc_Master_DF */
    std::string getloc_Master_DF(){return loc_Master_DF;}
    void Updateloc_Master_DF(std::string temp);
    /* loc_Master_Genotype */
    std::string getloc_Master_Genotype(){return loc_Master_Genotype;}
    void Updateloc_Master_Genotype(std::string temp);
    std::string getloc_Master_Genotype_zip(){return loc_Master_Genotype_zip;}
    void Updateloc_Master_Genotype_zip(std::string temp);
    /* loc_BinaryG_Matrix */
    std::string getloc_BinaryG_Matrix(){return loc_BinaryG_Matrix;}
    void Updateloc_BinaryG_Matrix(std::string temp);
    /* loc_Binarym_Matrix */
    std::string getloc_Binarym_Matrix(){return loc_Binarym_Matrix;}
    void Updateloc_Binarym_Matrix(std::string temp);
    /* loc_Binaryp_Matrix */
    std::string getloc_Binaryp_Matrix(){return loc_Binaryp_Matrix;}
    void Updateloc_Binaryp_Matrix(std::string temp);
    /* loc_BinaryLinv_Matrix */
    std::string getloc_BinaryLinv_Matrix(){return loc_BinaryLinv_Matrix;}
    void Updateloc_BinaryLinv_Matrix(std::string temp);
    /* loc_BinaryGinv_Matrix */
    std::string getloc_BinaryGinv_Matrix(){return loc_BinaryGinv_Matrix;}
    void Updateloc_BinaryGinv_Matrix(std::string temp);
    /* loc_Marker_Map */
    std::string getloc_Marker_Map(){return loc_Marker_Map;}
    void Updateloc_Marker_Map(std::string temp);
    /* loc_Master_DataFrame */
    std::string getloc_Master_DataFrame(){return loc_Master_DataFrame;}
    void Updateloc_Master_DataFrame(std::string temp);
    /* loc_Summary_QTL */
    std::string getloc_Summary_QTL(){return loc_Summary_QTL;}
    void Updateloc_Summary_QTL(std::string temp);
    /* loc_Summary_DF */
    std::string getloc_Summary_DF(){return loc_Summary_DF;}
    void Updateloc_Summary_DF(std::string temp);
    /* loc_GenotypeStatus */
    std::string getloc_GenotypeStatus(){return loc_GenotypeStatus;}
    void Updateloc_GenotypeStatus(std::string temp);
    /* Stores expected and realized DeltaG */
    std::string getloc_ExpRealDeltaG(){return loc_ExpRealDeltaG;}
    void Updateloc_ExpRealDeltaG(std::string temp);
    /* loc_LD_Decay */
    std::string getloc_LD_Decay(){return loc_LD_Decay;}
    void Updateloc_LD_Decay(std::string temp);
    /* loc_QTL_LD_Decay */
    std::string getloc_QTL_LD_Decay(){return loc_QTL_LD_Decay;}
    void Updateloc_QTL_LD_Decay(std::string temp);
    /* loc_Phase_Persistance */
    std::string getloc_Phase_Persistance(){return loc_Phase_Persistance;}
    void Updateloc_Phase_Persistance(std::string temp);
    /* loc_LD_Decay */
    std::string getloc_Phase_Persistance_Outfile(){return loc_Phase_Persistance_Outfile;}
    void Updateloc_Phase_Persistance_Outfile(std::string temp);
    /* loc_Summary_Haplofinder */
    std::string getloc_Summary_Haplofinder(){return loc_Summary_Haplofinder;}
    void Updateloc_Summary_Haplofinder(std::string temp);
    /* loc_Summary_ROHGenome_Freq */
    std::string getloc_Summary_ROHGenome_Freq(){return loc_Summary_ROHGenome_Freq;}
    void Updateloc_Summary_ROHGenome_Freq(std::string temp);
    /* loc_Summary_ROHGenome_Length */
    std::string getloc_Summary_ROHGenome_Length(){return loc_Summary_ROHGenome_Length;}
    void Updateloc_Summary_ROHGenome_Length(std::string temp);
    /* loc_Bayes_MCMC_Samples */
    std::string getloc_Bayes_MCMC_Samples(){return loc_Bayes_MCMC_Samples;}
    void Updateloc_Bayes_MCMC_Samples(std::string temp);
    /* loc_Bayes_PosteriorMeans */
    std::string getloc_Bayes_PosteriorMeans(){return loc_Bayes_PosteriorMeans;}
    void Updateloc_Bayes_PosteriorMeans(std::string temp);
    /* loc_Amax_Output */
    std::string getloc_Amax_Output(){return loc_Amax_Output;}
    void Updateloc_Amax_Output(std::string temp);
    /* Output ebv each generation so can split into geno versus non-genotyped */
    std::string getloc_TraitReference_Output(){return loc_TraitReference_Output;}
    void Updateloc_TraitReference_Output(std::string temp);
    /* loc_Windowadditive_Output */
    std::string getloc_Windowadditive_Output(){return loc_Windowadditive_Output;}
    void Updateloc_Windowadditive_Output(std::string temp);
    /* loc_Windowdominance_Output */
    std::string getloc_Windowdominance_Output(){return loc_Windowdominance_Output;}
    void Updateloc_Windowdominance_Output(std::string temp);
};

#endif
