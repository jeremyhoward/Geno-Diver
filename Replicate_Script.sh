#!/bin/bash

mkdir replicates

## Delete old parameter file ##
rm -rf ./parameterfile.txt || true
## Create Parameter File to Generate New Sequence information ##
echo "-----   Parameter file for Geno-Driver   -----" >> parameterfile.txt
echo "-----|          Starting Point          |-----" >> parameterfile.txt
echo "START: sequence" >> parameterfile.txt
echo "OUTPUTFOLDER: GenoDiverFiles" >> parameterfile.txt
echo "SEED: 1500" >> parameterfile.txt
echo "-----|   Genome and Marker Information  |-----" >> parameterfile.txt
echo "CHR: 3" >> parameterfile.txt
echo "CHR_LENGTH: 150 150 150" >> parameterfile.txt
echo "QTL: 50" >> parameterfile.txt
echo "FIT_LETHAL: 0" >> parameterfile.txt
echo "FIT_SUBLETHAL: 0" >> parameterfile.txt
echo "NUM_MARK: 4000 4000 4000" >> parameterfile.txt
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
mv ./GenoDiverFiles/Marker_Map ./replicates/Marker_Map_1500
mv ./GenoDiverFiles/Master_DataFrame ./replicates/Master_DataFrame_1500
gzip ./GenoDiverFiles/Master_Genotypes
mv ./GenoDiverFiles/Master_Genotypes.gz ./replicates/Master_Genotypes_1500.gz
mv ./GenoDiverFiles/QTL_new_old_Class ./replicates/QTL_new_old_Class_1500
mv ./GenoDiverFiles/log_file.txt ./replicates/log_file_1500

## Now Loop Through and use sequence information generated ##
sed -i '/START: sequence/c\START: founder' parameterfile.txt

## Loop across replicates ##
let count=1500
for i in {1501..1502}
do

sed -i -e 's/SEED: '"$count"'/\SEED: '"$i"'/g' parameterfile.txt
((count++))
#####################
## run Geno Driver ##
#####################
./GenoDiver << BLK
parameterfile.txt
BLK
####################################
## Move Files to Permanent Folder ##
####################################
mv ./GenoDiverFiles/Marker_Map ./replicates/Marker_Map_$i
mv ./GenoDiverFiles/Master_DataFrame ./replicates/Master_DataFrame_$i
gzip ./GenoDiverFiles/Master_Genotypes
mv ./GenoDiverFiles/Master_Genotypes.gz ./replicates/Master_Genotypes_$i.gz
mv ./GenoDiverFiles/QTL_new_old_Class ./replicates/QTL_new_old_Class_$i
mv ./GenoDiverFiles/log_file.txt ./replicates/log_file_$i

done