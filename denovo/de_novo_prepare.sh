###########################################
## DE NOVO DRAFT PROTOCOL - PREP
# Reqs.: ninja_filter (ninja_filter_linux)
###########################################

INPUTDIR=CflaA_raw_fastaq
TOO_SHORT=355
TOO_LONG=370

# 1. SHI7 without quality control. (optionally, learn beforehand for adaptor determination)
shi7.py -i $INPUTDIR -o shi7 --adaptor TruSeq2 --allow_outies F --filter_qual 1 --trim_qual 1 -s _,1

# 2. Filter for and remove primers. 
grep -B1 --no-group-separator 'GC.ATGGATGAGCA.*TTC.ACTTC.GT.G' shi7/combined_seqs.fna | sed 's/.*GC.ATGGATGAGCA//' | sed 's/TTC.ACTTC.GT.G.*//' > combined_seqs_np.fna
## depending on protocol, they may be different. V4: GGACTAC..GGGT.TCTAAT to GTG.CAGC.GCCGCGGTAA

# 3. Filter for desired amplicon length(s)
  #(optional: check lengths):
  awk '!(NR % 2) {print length}' combined_seqs_np.fna | sort | uniq -c | sort -rh | head -50
awk '{if (!(NR % 2) && length > 355 && length < 370) {print x; print $0}; x=$0}' combined_seqs_np.fna > manicured.fna

# 4. Create some "seed sets" to determine proper ref set cutoff (refine if desired)
for i in 002 003 005 010 025 050 100; do ninja_filter_linux manicured.fna D$i D $i && rm D$i.db; done
