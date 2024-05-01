#!/bin/bash

folders=(
    "/gpfs/data/ranganathlab/symile_tmp/20240421_132103_7385"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_132441_4945"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_134302_9434"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_152503_0405"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_152523_9052"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_172634_4873"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_172654_1555"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_183428_1528"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_192728_1015"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_192728_0211"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_204252_7346"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_212851_2404"
    "/gpfs/data/ranganathlab/symile_tmp/20240422_121618_4413"
    "/gpfs/data/ranganathlab/symile_tmp/20240422_020001_0961"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_215205_7219"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_225159_3035"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_233259_8398"
    "/gpfs/data/ranganathlab/symile_tmp/20240421_235158_9877"
    "/gpfs/data/ranganathlab/symile_tmp/20240422_010100_9123"
    "/gpfs/data/ranganathlab/symile_tmp/20240422_013602_9292"
)

for folder in "${folders[@]}"; do
  mv "$folder" /gpfs/scratch/as16583/ckpts/risk_prediction
done