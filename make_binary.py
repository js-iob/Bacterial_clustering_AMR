#Author: K. T. Shreya PArthasarathi
#Script: To make a binary matrix from the complete count matrix obtained



import pandas as pd
import numpy as np

file = pd.read_csv('final_sequence_kmer_count_021324.csv', sep = ',')
#print (file.head())

df = file.loc[: , file.columns != 'tenmers']
#print(df)

df.replace(0, np.nan, inplace = True)

kmers = file['tenmers']
#print (kmers)

#print (df.notnull())
binary =((df.notnull()).astype('int'))


final_binary = binary.join(kmers)
#print (final_binary)


cols = list(final_binary.columns)
#print (cols)
cols = [cols[-1]] + cols[:-1]
final_binary = final_binary[cols]
#print (final_binary)

final_binary.to_csv('final_sequence_kmer_binary_021324_2.csv', index= False)



