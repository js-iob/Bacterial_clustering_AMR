#Author: K. T. Shreya Parthasarathi
#Script: Calculate various features of the fasta sequences

from Bio import SeqIO

FastaFile = open("sequences_final_021324.fasta", 'r')
features = open ("seq_features.csv", 'w')
features.write('header' + '\t' + 'seqLen' + '\t' + 'gc_content' + '\t' + 'A' + '\t' + 'T' + '\t' + 'G' + '\t' + 'C' + '\n')

for rec in SeqIO.parse(FastaFile, 'fasta'):
    name = rec.id
    seq = rec.seq.upper()
    seqLen = len(rec)
    print(name + '\t' + str(seqLen))
    gc_content = ((seq.count('G') + seq.count('C')) * 100)/(seqLen)
    a = seq.count('A')
    t = seq.count('T')
    g = seq.count ('G')
    c = seq.count ('C')
    features.write (str(name) + '\t' + str(seqLen) + '\t' + str(gc_content) + '\t' + str(a) + '\t' + str(t) + '\t' + str(g) + '\t' + str(c) + '\n')

FastaFile.close()
features.close()



