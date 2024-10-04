FASTA_HUMAN=$$BORZOI_HG38/assembly/ucsc/hg38.ml.fa
GAPS_HUMAN=$$BORZOI_HG38/assembly/ucsc/hg38_gaps.bed
UMAP_HUMAN=$$BORZOI_HG38/mappability/umap_k36_t10_l32.bed
BLACK_HUMAN=$$BORZOI_HG38/blacklist/blacklist_hg38_all.bed

FASTA_MOUSE=$$BORZOI_MM10/assembly/ucsc/mm10.ml.fa
GAPS_MOUSE=$$BORZOI_MM10/assembly/ucsc/mm10_gaps.bed
UMAP_MOUSE=$$BORZOI_MM10/mappability/umap_k36_t10_l32.bed
BLACK_MOUSE=$$BORZOI_MM10/blacklist/blacklist_mm10_all.bed

ALIGN=$$BORZOI_HG38/align/hg38.mm10.syn.net.gz

OUT=data

# mini borzoi configuration
LENGTH=393216
TSTRIDE=43691 # (393216-2*131072)/3
CROP=98304
WIDTH=32
FOLDS=8

AOPTS=--break 2097152 -c $(CROP) --nf 524288 --no 393216 -l $(LENGTH) --stride $(TSTRIDE) -f $(FOLDS) --umap_t 0.5 -w $(WIDTH)
DOPTS=-c $(CROP) -d 2 -f $(FOLDS) -l $(LENGTH) -p 64 -r 16 --umap_clip 0.5 -w $(WIDTH) --transform_old

all: $(OUT)/hg38/tfrecords/train-0.tfr # $(OUT)/mm10/tfrecords/train-0.tfr

umap_human.bed:
	cat $(UMAP_HUMAN) $(BLACK_HUMAN) | awk 'BEGIN {OFS="\t"} {print $$1, $$2, $$3}' | bedtools sort -i - | bedtools merge -i - > umap_human.bed

umap_mouse.bed:
	cat $(UMAP_MOUSE) $(BLACK_MOUSE) | awk 'BEGIN {OFS="\t"} {print $$1, $$2, $$3}' | bedtools sort -i - | bedtools merge -i - > umap_mouse.bed

# targets file is already generated in this example
#targets_human.txt targets_mouse.txt:
#	./make_targets.py

$(OUT)/hg38/sequences.bed $(OUT)/mm10/sequences.bed: umap_human.bed umap_mouse.bed
	hound_data_align.py -a hg38,mm10 -g $(GAPS_HUMAN),$(GAPS_MOUSE) -u umap_human.bed,umap_mouse.bed $(AOPTS) -o $(OUT) $(ALIGN) $(FASTA_HUMAN),$(FASTA_MOUSE)

$(OUT)/hg38/tfrecords/train-0.tfr: $(OUT)/hg38/sequences.bed targets_human.txt
	hound_data.py --restart $(DOPTS) -b $(BLACK_HUMAN) -o $(OUT)/hg38 $(FASTA_HUMAN) -u umap_human.bed targets_human.txt

# no mouse data in this example
#$(OUT)/mm10/tfrecords/train-0.tfr: $(OUT)/mm10/sequences.bed targets_mouse.txt
#	hound_data.py --restart $(DOPTS) -b $(BLACK_MOUSE) -o $(OUT)/mm10 $(FASTA_MOUSE) -u umap_mouse.bed targets_mouse.txt
