#!/bin/bash

# set these variables before running the script
LOCAL_BORZOI_PATH="/home/jlinder/borzoi"
LOCAL_CONDA_PATH="/home/jlinder/anaconda3/etc/profile.d/conda.sh"

# create env_vars sh scripts in local conda env
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

file_vars_act="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
if ! [ -e $file_vars_act ]; then
    echo '#!/bin/sh' > $file_vars_act
fi

file_vars_deact="$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
if ! [ -e $file_vars_deact ]; then
    echo '#!/bin/sh' > $file_vars_deact
fi

# append env variable exports to /activate.d/env_vars.sh
echo "export BORZOI_DIR=$LOCAL_BORZOI_PATH" >> $file_vars_act
echo 'export PATH=$BORZOI_DIR/src/scripts:$PATH' >> $file_vars_act
echo 'export PYTHONPATH=$BORZOI_DIR/src/scripts:$PYTHONPATH' >> $file_vars_act

echo 'export BORZOI_HG38=$BORZOI_DIR/examples/hg38' >> $file_vars_act
echo 'export BORZOI_MM10=$BORZOI_DIR/examples/mm10' >> $file_vars_act

echo "export BORZOI_CONDA=$LOCAL_CONDA_PATH" >> $file_vars_act

# append env variable unsets to /deactivate.d/env_vars.sh
echo 'unset BORZOI_DIR' >> $file_vars_deact
echo 'unset BORZOI_HG38' >> $file_vars_deact
echo 'unset BORZOI_MM10' >> $file_vars_deact
echo 'unset BORZOI_CONDA' >> $file_vars_deact

# finally activate env variables
source $file_vars_act
