## Data processing & Training

Processing of the ENCODE, GTEx, FANTOM5, and CATlas training data is done through the Makefile in this folder. It requires a number of auxiliary files (e.g. genome alignments and lists of unmappable regions), which can be downloaded from the Borzoi training data bucket [here](https://storage.googleapis.com/borzoi-paper/data/) (GCP).<br/>
<br/>
The Makefile relies on the script 'basenji_data.py' from the [basenji repository](https://github.com/calico/basenji/blob/master/bin/basenji_data.py), which in turn calls the scripts 'basenji_data_read.py' and 'basenji_data_write.py' from the same repo, in order to (1) read coverage data (from bigwig-like files) along with a matched segment from a fasta genome file, and (2) write the (one-hot coded) sequence along with coverage values into compressed TF records.<br/>
<br/>
*Notes*: 
- The attached Makefile shows the exact commands used to call basenji_data.py and other related scripts to create the specific training data for the published model.
- The script(s) take as input a fasta genome file, optional blacklist+unmappable region files, as well as a .txt file where each row points to a bigwig coverage file location (see for [this file](https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt)).<br/>
<br/>
The model training is done through the script 'hound_train.py' from the [baskerville repository](https://github.com/calico/baskerville/blob/main/src/baskerville/scripts/hound_train.py). Most of the training parameters are set through a .json file that is supplied to the script. The published model's parameter file can be found [here](https://storage.googleapis.com/seqnn-share/borzoi/params.json).<br/>
