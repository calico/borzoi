#!/bin/bash

# download example data from ENCODE (ENCSR000AEL - K562 RNA-seq); 2 replicates

# define ENCODE ID
ENC_ID='ENCSR000AEL'

# define remote urls
URL_P_REP1='https://www.encodeproject.org/files/ENCFF980ZHM/@@download/ENCFF980ZHM.bigWig'
URL_M_REP1='https://www.encodeproject.org/files/ENCFF533LJF/@@download/ENCFF533LJF.bigWig'

URL_P_REP2='https://www.encodeproject.org/files/ENCFF335LVS/@@download/ENCFF335LVS.bigWig'
URL_M_REP2='https://www.encodeproject.org/files/ENCFF257NOL/@@download/ENCFF257NOL.bigWig'

# define ENCODE file IDs
FILE_P_REP1='ENCFF980ZHM'
FILE_M_REP1='ENCFF533LJF'

FILE_P_REP2='ENCFF335LVS'
FILE_M_REP2='ENCFF257NOL'

# create folder for bigwig files
mkdir -p "human/rna/encode/$ENC_ID/rep1"
mkdir -p "human/rna/encode/$ENC_ID/rep2"


# download bigwig files; rep1
if [ -f "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1.bigWig" ]; then
  echo "example RNA-seq data already downloaded (rep 1)."
else
  wget $URL_P_REP1 -O "human/rna/encode/$ENC_ID/rep1/$FILE_P_REP1.bigWig"
  wget $URL_M_REP1 -O "human/rna/encode/$ENC_ID/rep1/$FILE_M_REP1.bigWig"
fi

# download bigwig files; rep2
if [ -f "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2.bigWig" ]; then
  echo "example RNA-seq data already downloaded (rep 2)."
else
  wget $URL_P_REP2 -O "human/rna/encode/$ENC_ID/rep2/$FILE_P_REP2.bigWig"
  wget $URL_M_REP2 -O "human/rna/encode/$ENC_ID/rep2/$FILE_M_REP2.bigWig"
fi
