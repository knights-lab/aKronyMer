# aKronyMer Documentation

## Overview

aKronyMer is a powerful and versatile tool for sequence comparison and analysis. It uses k-mer based approaches to compute distances between sequences, create phylogenetic trees, perform set clustering, and more. aKronyMer is particularly useful for large-scale genomic and metagenomic studies, offering various distance metrics and output formats.

## Version

This documentation covers aKronyMer version 1.00+.

## Features

1. Fast k-mer based sequence comparison
2. Multiple distance calculation methods (ANI-like, phylogenetic, etc.)
3. Tree construction
4. Set clustering for representative sequence selection
5. Support for reverse complement comparisons
6. Genome complexity adjustment
7. Query vs. reference alignment mode
8. Heuristic acceleration for large datasets
9. Multithreading support

## Usage

```
aKronyMer inseqs.lin.fna output [K] [Q queries.lin.fna [MIN2]] [HEUR[0-9]] [ANI] [CHANCE] [GC] [ADJ] [GLOBAL/DIRECT/LOCAL] [TREE/SETCLUSTER] [RC] [NOCUT]
```

## Input Requirements

- Input sequences must be in linearized FASTA format (one line header, one line sequence)
- Maximum sequence length: 2GB (can be extended to 4GB for human genomes)
- Supports DNA sequences only (protein support may be added in future versions)
- For metagenome samples, joining sequences with N's is recommended

## Basic Parameters

1. `inseqs.lin.fna`: Input file in linearized FASTA format
   - Can contain marker genes, short reads, whole genomes, or metagenomic samples
   - Must be preprocessed using the `lingenome` tool (see File Format Details section)

2. `output`: Output file (format depends on other options)
   - Will be a distance matrix, Newick tree, or set clustering results depending on other options

3. `K`: K-mer size (optional, auto-selected if not specified)
   - Shorter k-mers allow more distant comparisons, longer ones provide more specificity
   - If not specified, the program attempts to auto-select based on sequence length distribution
   - Recommended values:
     - Phage genomes: K = 10
     - Bacterial genomes: K = 13
     - Metagenomes or raw (R1) shallow metagenome samples: K = 14
     - Amplicons (16S, ITS, etc.): K = 7 (omit RC option)

## Modes and Options

- `Q queries.lin.fna [MIN2]`: Query vs. reference alignment mode
  - Converts aKronyMer to a query-vs-reference alignment paradigm
  - Incompatible with TREE and SETCLUSTER options
  - Reports distances from queries to references
  - If MIN2 is specified, only the top 2 hits by minimum distance are returned

- `HEUR[0-9]`: Use heuristics for faster processing (0-9 for strength)
  - 0 is equivalent to omitting the parameter (no heuristics)
  - Higher numbers trade accuracy for speed
  - Recommended for very large datasets (> 100,000 input sequences)
  - HEUR4 is generally reasonable for large amplicon datasets
  - HEUR9 may be valid for genomic datasets

- `ANI`: Use Average Nucleotide Identity-like distance metric
  - Attempts to simulate results of conventional sequence alignment
  - Slightly less accurate for shorter sequences (< 1000nt)

- `CHANCE`: Probabilistic correction for ANI (recommended with ANI)
  - Works in tandem with ANI to probabilistically correct the ANI
  - Accounts for k-mer size, k-mer space saturation, and expected distance given random sequences
  - Highly recommended when using ANI

- `GC`: Genome complexity adjustment
  - Uses machine learning inference to account for base composition
  - Leads to more accurate determination of baseline alignment probability
  - Recommended for more accurate distances

- `ADJ`: Perform Jukes-Cantor or GTR-like phylogenetic site substitution rate correction
  - Recommended when running on genes or genomes
  - May be useful when computing distances between samples if the tree is used
  - Not recommended when ANI and CHANCE are used together

- `GLOBAL/DIRECT/LOCAL`: Type of distance to compute
  - GLOBAL: Uses a global distance, penalizing distance relative to the longer sequence
  - DIRECT: Most advanced distance for trees, accounting for proportion of joint k-mer space and entropy
  - LOCAL: Allows short sequences to achieve low distances to much longer sequences (default for non-ANI)

- `TREE`: Output Newick-formatted tree instead of distance matrix
  - If omitted, default output is a lower-left-triangular distance matrix

- `SETCLUSTER`: Perform set clustering and output representative sequences
  - Outputs a list of representative genomes, ordered by most relevant new genetic content
  - Continues until 0.001 information gain is achieved (or until NOCUT is used)

- `RC`: Consider reverse complements of k-mers
  - Useful for genomes with contigs that are not uniformly oriented
  - Recommended for comparing raw metagenomes or genes/genomes with mobile elements

- `NOCUT`: For SETCLUSTER, continue until no new k-mer content is added
  - Overrides the default 0.001 information gain cutoff in SETCLUSTER mode

## Output Formats

1. Distance Matrix: Tab-delimited lower-left triangle format
   - Readable in R using: `as.dist(read.delim('output',row=1))`

2. Newick Tree: When TREE option is specified

3. Set Clustering: List of representative genomes with statistics
   - Includes running total of k-mer space coverage and uniqueness measures

## Performance Considerations

- Use the number of non-hyperthreaded cores for optimal performance
- Adjust K value based on input sequence lengths and desired specificity
- Use heuristics (HEUR option) for very large datasets (>100,000 sequences)
- If k-mer space becomes saturated (Density > 0.25), try increasing the K value

## Examples

1. Creating a tree for genomes:
   ```
   aKronyMer AllGenomes.fasta AllGenomes.tre 13 ADJ DIRECT TREE RC
   ```

2. Creating an ANI distance matrix:
   ```
   aKronyMer AllGenomes.fasta AllGenomes.dm 13 ANI CHANCE GC RC
   ```

3. Creating a glocal ANI distance matrix:
   ```
   aKronyMer AllGenomes.fasta AllGenomes.dm 13 ANI CHANCE GC LOCAL RC
   ```

4. Set clustering for bacterial genomes:
   ```
   aKronyMer AllGenomes.fasta AllGenomes.sc 14 RC SETCLUSTER NOCUT
   ```

5. Query vs. reference alignment:
   ```
   aKronyMer ReferenceGenomes.fasta QueryDists.tsv 13 Q Queries.fasta ANI CHANCE GC RC
   ```

6. Query vs. reference alignment, returning only top 2 hits:
   ```
   aKronyMer ReferenceGenomes.fasta QueryDists.tsv 13 Q Queries.fasta MIN2 ANI CHANCE GC RC
   ```

7. Creating a tree for huge amplicon datasets (millions of reads):
   ```
   aKronyMer AmpliconData.fasta AmpliconTree.tre 5 HEUR4 TREE
   ```

## Limitations

- Currently supports DNA sequences only
- Maximum sequence length of 2GB (4GB with modifications)
- Heuristics may trade accuracy for speed

## Detailed Algorithm Description

aKronyMer uses a k-mer based approach for sequence comparison. It converts sequences into k-mer profiles and compares these profiles to calculate distances between sequences. The process involves:

1. K-mer extraction and counting
2. Profile creation and comparison
3. Distance calculation using various metrics
4. Optional corrections and adjustments (GC content, CHANCE, etc.)

## Distance Metrics

### ANI-like Distance

When the ANI option is used, aKronyMer calculates a distance metric similar to Average Nucleotide Identity. The calculation involves several steps:

1. Calculate the Jaccard-like similarity: `sim = (matching k-mers) / (total unique k-mers)`
2. Apply a log transformation: `logSim = log(2 * sim / (1 + sim)) / K`
3. Calculate the initial distance: `dist = 1 - (1 + logSim)`
4. If ADJ option is used, apply further correction: `adjDist = -0.75 * log(1 - (4/3) * dist)`

This process accounts for the sampling nature of k-mers and attempts to correct for multiple substitutions when the ADJ option is used.

### Phylogenetic Distance

When the ADJ option is used, aKronyMer applies a Jukes-Cantor or GTR-like correction to the distances. This is recommended for genes or genomes to account for multiple substitutions.

### DIRECT Distance

This is a probabilistic distance that accounts for the proportion of the joint k-mer space occupied by a pair of sequences, penalized by the increase in entropy of the resulting pseudoalignment. It's most useful in combination with the ADJ option.

## Genome Complexity Adjustment (GC)

The GC option uses machine learning inference to account for base composition when performing distance adjustments. This leads to more accurate determination of baseline alignment probability and more accurate distances.

## CHANCE Correction

The CHANCE option works with ANI to probabilistically correct the ANI based on k-mer size, k-mer space saturation, and expected distance given random sequences. It's highly recommended when using ANI.

## Heuristics

Heuristics can be enabled with the HEUR option, followed by a number from 0 to 9 indicating the strength. Higher numbers trade accuracy for speed. The heuristic works by sampling smaller portions of the original sequences for comparison.

## Set Clustering Algorithm

When SETCLUSTER is specified, aKronyMer performs a greedy set coverage algorithm:

1. Start with the sequence having the most unique k-mers
2. Iteratively add sequences that contribute the most new k-mers
3. Continue until a cutoff is reached (0.001 information gain or no new k-mers if NOCUT is specified)

## Memory Usage and Optimizations

aKronyMer uses several optimization techniques:

1. SIMD instructions for faster k-mer counting and comparison
2. Multithreading for parallel processing of sequences
3. Memory-efficient data structures for k-mer storage

## Advanced Usage Notes

### K-mer Saturation

The "Density" or "K_saturation" output line indicates k-mer space saturation. Values > 0.25 may limit similarity detection accuracy.

### Reverse Complement (RC)

The RC option is useful for:
- Genomes with contigs that are not uniformly oriented
- Comparing raw metagenomes
- Analyzing genes or genomes with mobile elements that may have been reoriented during evolution

### Query vs. Reference Mode

In this mode (Q option), aKronyMer compares each query sequence against all reference sequences. The MIN2 suboption restricts output to only the top 2 hits for each query.

## File Format Details

### Linearized FASTA

A modified FASTA format where each sequence entry consists of exactly two lines:
1. Header line (starts with '>')
2. Sequence line (no line breaks within the sequence)

This format can be created using the `lingenome` tool:

```
lingenome myFolderOfFastaGenomes AllGenomes.fasta HEADFIX FILENAME
```

## Future Directions

1. Better documentation
2. Adjustable weight for phylogenetic distance adjustment
3. Differential k-mer weights based on relative abundance
4. Protein/arbitrary sequence support
5. Further tuning of distance metrics
6. Improved SETCLUSTER objective function
7. Revamped command-line structure (non-positional arguments)

## Troubleshooting

1. If k-mer space becomes saturated (Density > 0.25), try increasing the K value
2. For very large datasets, start with a higher heuristic level (e.g., HEUR4) and adjust as needed
3. When comparing sequences of vastly different lengths, consider using the LOCAL distance option

By understanding these features and algorithms, users can fine-tune aKronyMer for optimal performance across a wide range of comparative genomics applications.
