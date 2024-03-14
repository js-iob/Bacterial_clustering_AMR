#Author: K. T. Shreya Parthasarathi
#Script: To remove the 10-mers with bases other than 'A', 'T', 'G', 'C'

tenmers = list()
non_bases = list()
file = open('all_seq_tenmers_021324.txt')
out = open('tenmers.txt', 'w')
matches = ["B", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "U", "V", "W", "X", "Y", "Z"]

for line in file:
    line = line.rstrip()
    if any(x in line for x in matches):
        non_bases.append(line)
    else:
        out.write(line+'\n')
        tenmers.append(line)

out.close()
		

