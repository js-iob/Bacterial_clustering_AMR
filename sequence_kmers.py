#Author: K. T. Shreya Parthasarathi
#Script: To split the fasta sequences into kmers of length 10

seq = str()
sequence = []
length = []

with open('sequences_final_021324.fasta') as file:
    for line in file:
        line = line.rstrip()
        if not line.startswith('>'):
            seq = seq + line
        else:
            sequence.append(seq)
            length.append(len(seq))
            seq = ''
file.close()
print ('Done step1')

#Generate k-mers
kmer = []
kmers = set(kmer)
for seq in sequence:
    for i in range(0, (len(seq)-1)):
        k = seq[i:i+10]
        if k not in kmers:
            kmers.add(k)
list_kmers = list(kmers)
print ('Done step2')

#Extract tenmers
ten_mer = []
for i in list_kmers:
    if len(i) == 10:
        ten_mer.append(i)
print ('Done step3')

#Remove Ns bases
ten_mer_base = []
for i in ten_mer:
    i = i.upper()
    if 'N' not in i:
        ten_mer_base.append(i)
print ('Done step4')

out = open('all_seq_tenmenrs_021324.txt', 'w')
for i in ten_mer_base:
    out.write(i + '\n')
out.close()
print ('Done step5')




            
