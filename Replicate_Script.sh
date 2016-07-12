#!/bin/bash

## Delete old parameter file ##
rm -rf ./parameterfile.txt || true
## Create Parameter File to Generate New Sequence information ##
echo "-----   Parameter file for Geno-Driver   -----" >> parameterfile.txt
echo "-----|          Starting Point          |-----" >> parameterfile.txt
echo "START: sequence" >> parameterfile.txt
echo "OUTPUTFOLDER: GenoDriverFiles" >> parameterfile.txt
echo "SEED: 1500" >> parameterfile.txt
echo "-----|   Genome and Marker Information  |-----" >> parameterfile.txt
echo "CHR: 3" >> parameterfile.txt
echo "CHR_LENGTH: 150 150 150" >> parameterfile.txt
echo "QTL: 50" >> parameterfile.txt
echo "FIT_LETHAL: 0" >> parameterfile.txt
echo "FIT_SUBLETHAL: 0" >> parameterfile.txt
echo "NUM_MARK: 4000 4000 4000" >> parameterfile.txt
echo "MARKER_MAF: 0.10" >> parameterfile.txt
echo "-----|    Population Characteristics    |-----" >> parameterfile.txt
echo "FOUNDER_Effective_Size: Ne70" >> parameterfile.txt
echo "VARIANCE_A: 0.35" >> parameterfile.txt
echo "VARIANCE_D: 0.05" >> parameterfile.txt
echo "-----|   Selection and Mating Parameters|-----" >> parameterfile.txt
echo "GENERATIONS: 10" >> parameterfile.txt
echo "INDIVIDUALS: 50 0.2 600 0.2" >> parameterfile.txt
echo "PROGENY: 1" >> parameterfile.txt
echo "SELECTION: ebv high" >> parameterfile.txt
echo "SOLVER_INVERSE: pedigree pcg cholesky" >> parameterfile.txt
echo "MATING: random5" >> parameterfile.txt
echo "CULLING: ebv 5" >> parameterfile.txt
## run Geno Driver ##
./GenoDiver << BLK
parameterfile.txt
BLK

## Move Files to Permanent Folder ##
mv ./GenoDriverFiles/Marker_Map /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/Marker_Map_1500
mv ./Master_DataFrame /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/Master_DataFrame_1500
gzip Master_Genotypes
mv ./Master_Genotypes.gz /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/Master_Genotypes_1500.gz
mv ./GenoDriverFiles/QTL_new_old_Class /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/QTL_new_old_Class_1500
mv ./GenoDriverFiles/log_file.txt /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/log_file_1500

## Now Loop Through and use sequence information generated ##
for i in {1501..1509}
do
## Delete old parameter file ##
rm -rf ./parameterfile.txt || true
## Create Parameter File ##
echo "-----   Parameter file for Geno-Driver   -----" >> parameterfile.txt
echo "-----|          Starting Point          |-----" >> parameterfile.txt
echo "START: sequence" >> parameterfile.txt
echo "OUTPUTFOLDER: GenoDriverFiles" >> parameterfile.txt
echo "SEED: $i" >> parameterfile.txt
echo "-----|   Genome and Marker Information  |-----" >> parameterfile.txt
echo "CHR: 3" >> parameterfile.txt
echo "CHR_LENGTH: 150 150 150" >> parameterfile.txt
echo "QTL: 50" >> parameterfile.txt
echo "FIT_LETHAL: 0" >> parameterfile.txt
echo "FIT_SUBLETHAL: 0" >> parameterfile.txt
echo "NUM_MARK: 4000 4000 4000" >> parameterfile.txt
echo "MARKER_MAF: 0.10" >> parameterfile.txt
echo "-----|    Population Characteristics    |-----" >> parameterfile.txt
echo "FOUNDER_Effective_Size: Ne70" >> parameterfile.txt
echo "VARIANCE_A: 0.35" >> parameterfile.txt
echo "VARIANCE_D: 0.05" >> parameterfile.txt
echo "-----|   Selection and Mating Parameters|-----" >> parameterfile.txt
echo "GENERATIONS: 10" >> parameterfile.txt
echo "INDIVIDUALS: 50 0.2 600 0.2" >> parameterfile.txt
echo "PROGENY: 1" >> parameterfile.txt
echo "SELECTION: ebv high" >> parameterfile.txt
echo "SOLVER_INVERSE: pedigree pcg cholesky" >> parameterfile.txt
echo "MATING: random5" >> parameterfile.txt
echo "CULLING: ebv 5" >> parameterfile.txt

## run Geno Driver ##
./GenoDiver << BLK
parameterfile.txt
BLK
## Move Files to Permanent Folder ##
mv ./GenoDriverFiles/Marker_Map /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/Marker_Map_$i
mv ./Master_DataFrame /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/Master_DataFrame_$i
gzip Master_Genotypes
mv ./Master_Genotypes.gz /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/Master_Genotypes_$i.gz
mv ./GenoDriverFiles/QTL_new_old_Class /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/QTL_new_old_Class_$i
mv ./GenoDriverFiles/log_file.txt /home/jthoward/PIG_ROH/Haplofinder/simulation/datafiles3/LD_very_very_high/log_file_$i

done

