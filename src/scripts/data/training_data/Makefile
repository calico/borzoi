FASTA_HUMAN=$$HG38/assembly/ucsc/hg38.ml.fa
GAPS_HUMAN=$$HG38/assembly/ucsc/hg38_gaps.bed
UMAP_HUMAN=$$HG38//mappability/umap_k36_t10_l32.bed
BLACK_HUMAN=$$HG38/blacklist/blacklist_hg38_all.bed

FASTA_MOUSE=$$MM10/assembly/ucsc/mm10.ml.fa
GAPS_MOUSE=$$MM10/assembly/ucsc/mm10_gaps.bed
UMAP_MOUSE=$$MM10//mappability/umap_k36_t10_l32.bed
BLACK_MOUSE=$$MM10/blacklist/blacklist_mm10_all.bed

ALIGN=$$HG38/align/hg38.mm10.syn.net.gz

OUT=/scratch3/drk/seqnn/data/v9

# LENGTH=393216
# TSTRIDE=43691 # (393216-2*131072)/3
# CROP=131072

LENGTH=524288
TSTRIDE=49173 # (524288-2*163840)/4 + 21
CROP=163840
WIDTH=32
FOLDS=8

AOPTS=--break 2097152 -c $(CROP) --nf 524288 --no 393216 -l $(LENGTH) --stride $(TSTRIDE) -f $(FOLDS) --umap_t 0.5 -w $(WIDTH)
DOPTS=-c $(CROP) -d 2 -f $(FOLDS) -l $(LENGTH) -p 64 -r 16 --umap_clip 0.5 -w $(WIDTH)


all: $(OUT)/hg38/tfrecords/train-0.tfr $(OUT)/mm10/tfrecords/train-0.tfr

umap_human.bed:
	cat $(UMAP_HUMAN) $(BLACK_HUMAN) | awk 'BEGIN {OFS="\t"} {print $$1, $$2, $$3}' | bedtools sort -i - | bedtools merge -i - > umap_human.bed

umap_mouse.bed:
	cat $(UMAP_MOUSE) $(BLACK_MOUSE) | awk 'BEGIN {OFS="\t"} {print $$1, $$2, $$3}' | bedtools sort -i - | bedtools merge -i - > umap_mouse.bed

targets_human.txt targets_mouse.txt:
	./make_targets.py

$(OUT)/hg38/sequences.bed $(OUT)/mm10/sequences.bed: umap_human.bed umap_mouse.bed
	basenji_data_align.py -a hg38,mm10 -g $(GAPS_HUMAN),$(GAPS_MOUSE) -u umap_human.bed,umap_mouse.bed $(AOPTS) -o $(OUT) $(ALIGN) $(FASTA_HUMAN),$(FASTA_MOUSE)

$(OUT)/hg38/tfrecords/train-0.tfr: $(OUT)/hg38/sequences.bed targets_human.txt
	basenji_data.py --restart $(DOPTS) -b $(BLACK_HUMAN) -o $(OUT)/hg38 $(FASTA_HUMAN) -u umap_human.bed targets_human.txt

$(OUT)/mm10/tfrecords/train-0.tfr: $(OUT)/mm10/sequences.bed targets_mouse.txt
	basenji_data.py --restart $(DOPTS) -b $(BLACK_MOUSE) -o $(OUT)/mm10 $(FASTA_MOUSE) -u umap_mouse.bed targets_mouse.txt
