#Author: K.T. Shreya Parthasarathi
#Script: To get the count of each 10-mer in each strain 

setwd ('path/to/all_files')
library(seqinr)
#BiocManager::install("Biostrings")
    

library(Biostrings)

dat = read.table("unique_tenmers.txt", header = T)
fasta_file = readDNAStringSet("sequences_final.fasta")
head(fasta_file)
pattern = dat$tenmers
pattern = as.character(pattern)
head(pattern)
class(pattern)

dict = PDict(pattern, max.mismatch = 0)
#seq = DNAStringSet(unlist(fasta_file))
result = vcountPDict(dict, fasta_file)
class(result)
head(result)
result2 = as.data.frame(result)
head(result2)

colnames(result2) = names(fasta_file)
result3 = cbind(tenmers = dat$tenmers, result2)

dim(result3)
head(result3)
class(result3)

#install.packages("data.table")
library('data.table')
fwrite(result3, "final_sequence_kmer_count_021324.csv")


