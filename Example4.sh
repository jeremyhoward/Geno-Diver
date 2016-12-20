## Delete old parameter file and director to place files if their##
rm -rf ./Example4_Output || true
mkdir Example4_Output

## run Geno Driver using the initial parameter file ##
./GenoDiver Example4.txt
## Move inbreeding folder to permanent location
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random

## Now do random 5 and start with founder
sed -i '/START: sequence/c\START: founder' Example4.txt
sed -i '/MATING: random/c\MATING: random5' Example4.txt
./GenoDiver Example4.txt
## move to seperate folder
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random5

## Now do random 25 
sed -i '/MATING: random5/c\MATING: random25' Example4.txt
./GenoDiver Example4.txt
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random25

## Now do random 125
sed -i '/MATING: random25/c\MATING: random125' Example4.txt
./GenoDiver Example4.txt
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_random125

## Now do minimize pedigree
sed -i '/MATING: random125/c\MATING: minPedigree' Example4.txt
./GenoDiver Example4.txt
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_pedigree

## Now do minimize genomic
sed -i '/MATING: minPedigree/c\MATING: minGenomic' Example4.txt
./GenoDiver Example4.txt
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_genomic

## Now do minimize genomic
sed -i '/MATING: minGenomic/c\MATING: minROH' Example4.txt
./GenoDiver Example4.txt
mv ./GenoDiverFiles/Summary_Statistics_DataFrame_Inbreeding ./Example4_Output/inbreeding_ROH

# Change back to original Example4.txt parameter file
sed -i '/MATING: minROH/c\MATING: random' Example4.txt


