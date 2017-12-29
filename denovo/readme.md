## Instructions for aKronyMer de novo go here!

# aKronyMer de novo is a fast (~60 second) de novo pipeline operating on amplicon data. 
The goal is to go from sequences to rep set to clusters/ASVs to phylogeny to results in seconds.

Basically, edit and run these, in order:
1. bash de_novo_prepare.sh
2. bash de_novo_calculate.sh

Optionally, for quick rep set analysis, you can just start with a manicured fasta file directly (fewer dependencies and no editing required as long as the burst12 and ninja_filter_linux dependencies are in PATH and ms_worker.sh is in the calling directory):

bash do_multisim.sh myFasta.fna

A report will be generated showing the effects of the two main thresholds, D (deduplication) and S (similarity), in your dataset. 
