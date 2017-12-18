###############################################################
## DE NOVO DRAFT PROTOCOL - REPSET
# Reqs.: burst (burst12), embalmulate, akronymer (akmer94b)
# Makes alignments.b6, otu_table.txt, rep_set.fna, rep_set.tre
###############################################################

PERC_ALIGN=95
SIMILARITY=0.98
OMP_NUM_THREADS=136


# 1. Store original number of sequences
D1OTUS=$(($(wc -l manicured.fna | tr -s ' ' | cut -d' ' -f1)/2))

# 2. Determine appropriate threshold
for f in D*_filt.fa; do 
  DBNAME=${f/_filt.fa/}; 
  burst12 -r "$DBNAME"_filt.fa -d -o "$DBNAME.edb" -a "$DBNAME.acc" -t $OMP_NUM_THREADS;
  burst12 -o /dev/null -r "$DBNAME.edb" && burst12 -o /dev/null -a "$DBNAME.acc";
  rm "$DBNAME.edb" && rm "$DBNAME.acc";
  burst12 -r "$DBNAME".edx -a "$DBNAME".acx -q manicured.fna -i $SIMILARITY -o "$DBNAME".b6 -t $OMP_NUM_THREADS; 
done

  #(optional: take a look!)
  wc -l *.b6 | grep -v "total" | sed 's/^ *//' | sort -h; echo "$D1OTUS Original"

# 3. Align to candidate rep sets; pick best
REPSET=
SZLST=($(wc -l *.b6 | grep -v " total" | sort -h | sed 's/ *//' | cut -d' ' -f1))
FLST=($(wc -l *.b6 | grep -v " total" | sort -h | sed 's/ *//' | cut -d' ' -f2))
for i in "${!SZLST[@]}"; do 
  if ((${SZLST[$i]} > $D1OTUS*$PERC_ALIGN/100)); then 
    REPSET=${FLST[$i]}; echo "Rep set is $REPSET with $((${SZLST[$i]}/2)) OTUs";
    break; 
  fi; 
done

cp $REPSET alignments.b6
cp ${REPSET/.b6/_filt.fa} rep_set.fna
embalmulate alignments.b6 otu_table.txt

# 4. Build phylogenetic tree out of rep set
akmer94b rep_set.fna rep_set.tre 5 ADJ DIRECT TREE

## DONE!!

# Analysis (optional)
#burst12 -r D100.edx -a D100.acx -q rep_set.fna -i 0.98 -o self_D100.b6 -m FORAGE -t $OMP_NUM_THREADS
#sort -n -k 2,2 -s self_D100.b6 | sort -n -s -k 3,3 | sort -n -s -k 1,1 > self_D100_srt.b6
#cut -f1 alignments.b6 | cut -f1 -d_ | sort | uniq -c | sort -hr
#cut -f2 alignments.b6 | sort | uniq -c | sort -hr
