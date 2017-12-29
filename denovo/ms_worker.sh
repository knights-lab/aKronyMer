#####################################################################
## DE NOVO DRAFT PROTOCOL - REPSET
# Reqs.: burst (burst12)
# Produce a similarity report with a given similarity; use in loop!
# Usage: ms_worker.sh <input.fna> <similarity> <threshold> <threads>
#####################################################################

FASTA=$1
SIM=$2
PERC_ALIGN=$3
OMP_NUM_THREADS=$4

# 1. Store original number of sequences
D1OTUS=$(($(wc -l $FASTA | tr -s ' ' | cut -d' ' -f1)/2))

# 2. Determine appropriate threshold
for f in D*_filt.fa; do 
  DBNAME=${f/_filt.fa/}; 
  burst12 -r "$DBNAME"_filt.fa -d -o "$DBNAME$SIM.edb" -a "$DBNAME$SIM.acc" -t $OMP_NUM_THREADS;
  burst12 -o /dev/null -r "$DBNAME$SIM.edb" && burst12 -o /dev/null -a "$DBNAME$SIM.acc";
  rm "$DBNAME$SIM.edb" && rm "$DBNAME$SIM.acc";
  burst12 -r "$DBNAME$SIM".edx -a "$DBNAME$SIM".acx -q $FASTA -i $SIM -o "$DBNAME-$SIM".b6 -t $OMP_NUM_THREADS; 
  rm "$DBNAME$SIM.edx" && rm "$DBNAME$SIM.acx";
done

# Report results
echo "Similarity: $SIM" > report_"$SIM".txt
wc -l *"-$SIM".b6 | grep -v "total" | sed 's/^ *//' | sort -h >> report_"$SIM".txt; 
echo "$D1OTUS Original" >> report_"$SIM".txt
echo "$(($D1OTUS*$PERC_ALIGN/100)) [Threshold]" >> report_"$SIM".txt
echo "" >> report_"$SIM".txt
rm *"-$SIM".b6 
