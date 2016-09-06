## Delete old parameter file and director to place files if their##
rm -rf ./Example4.txt || true
rm -rf ./Example4_Output || true
mkdir Example4_Output

################################################################
## Create Parameter File to Generate New Sequence information ##
################################################################
echo "−−−−−−−|     Example 1 Parameter File  |−−−−−−−" >> Example4.txt
echo "−−−−−−−|       Starting Parameters     |−−−−−−−" >> Example4.txt
echo "START: sequence" >> Example4.txt
echo "SEED: 1500" >> Example4.txt
echo "THREAD: 4" >> Example4.txt
echo "−−−−−−−| Genome and Marker Information |−−−−−−−" >> Example4.txt
echo "CHR: 3" >> Example4.txt
echo "CHR_LENGTH: 150 150 150" >> Example4.txt
echo "NUM_MARK: 4000 4000 4000" >> Example4.txt
echo "QTL: 50" >> Example4.txt
echo "−−−−−−−|     Population Parameters     |−−−−−−−" >> Example4.txt
echo "FOUNDER_Effective_Size: Ne70" >> Example4.txt
echo "VARIANCE_A: 0.35" >> Example4.txt
echo "VARIANCE_D: 0.05" >> Example4.txt
echo "−−−−−−−| Selection and Mating Parameters |−−−−−−−" >> Example4.txt
echo "GENERATIONS: 10" >> Example4.txt
echo "INDIVIDUALS: 50 0.2 600 0.2" >> Example4.txt
echo "PROGENY: 1" >> Example4.txt
echo "SELECTION: ebv high" >> Example4.txt
echo "SOLVER_INVERSE: pedigree pcg cholesky" >> Example4.txt
echo "MATING: random" >> Example4.txt
echo "CULLING: ebv 5" >> Example4.txt
## run Geno Driver ##
./GenoDiver << BLK
Example4.txt
BLK
## Move inbreeding folder to permanent location
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random

## Now do random 5 and start with founder
sed -i '/START: sequence/c\START: founder' Example4.txt
sed -i '/MATING: random/c\MATING: random5' Example4.txt
## run program
./GenoDiver << BLK
Example4.txt
BLK
## move to seperate folder
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random5

## Now do random 25 
sed -i '/MATING: random5/c\MATING: random25' Example4.txt
./GenoDiver << BLK
Example4.txt
BLK
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random25

## Now do random 125
sed -i '/MATING: random25/c\MATING: random125' Example4.txt
./GenoDiver << BLK
Example4.txt
BLK
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random125

## Now do minimize pedigree
sed -i '/MATING: random125/c\MATING: minPedigree' Example4.txt
./GenoDiver << BLK
Example4.txt
BLK
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_pedigree

## Now do minimize genomic
sed -i '/MATING: minPedigree/c\MATING: minGenomic' Example4.txt
./GenoDiver << BLK
Example4.txt
BLK
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_genomic

## Now do minimize genomic
sed -i '/MATING: minGenomic/c\MATING: minROH' Example4.txt
./GenoDiver << BLK
Example4.txt
BLK
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_ROH




