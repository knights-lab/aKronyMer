##########################################################################
## DE NOVO DRAFT PROTOCOL - REPSET
# Reqs.: burst (burst12), ninja_filter (ninja_filter_linux), ms_worker.sh
# Makes a rep set report with various duplication/similarity thresholds
# Usage: bash do_multisim.sh <manicured.fna>
# Tip: invoke in a clean, writable working directory
#########################################################################

FASTA=$1
PERC_ALIGN=95
THREADS=8

echo "Preparing rep sets..."
for D in 002 003 005 010 025 050 100; do ninja_filter_linux $FASTA D$D D $D &> /dev/null && rm D$D.db & done; wait
echo "Aligning rep sets..."
for S in 0.9900 0.9875 0.9850 0.9825 0.9800 0.9775 0.9750 0.9725 0.9700; do bash ms_worker.sh $FASTA $S $PERC_ALIGN $THREADS $S &> /dev/null & done; wait
cat report_* > clus_report.txt
rm report_* *_filt.fa
#cat clus_report.txt
echo "Done. Report saved as clus_report.txt"
