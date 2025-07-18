from Bio import SeqIO
import numpy as np
import re
from scipy.integrate import lebedev_rule
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

for sequences in SeqIO.parse('/home/raza/ML/DNA_seq_ML/example_dna.fa', 'fasta'):
    print(sequences.id)
    print(sequences.description)
    print(sequences.seq)

def string_to_array(seq_string):
    seq_string = seq_string.lower()
    seq_string = re.sub('[^atgc]','n', seq_string)
    seq_string = np.array(list(seq_string))
    return seq_string

# create a label encoder with 'acgtn' alphabet
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))

def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0 ] = 0.25 #A
    float_encoded[float_encoded == 1 ] = 0.50 #C
    float_encoded[float_encoded == 2 ] = 0.75 #G
    float_encoded[float_encoded == 3 ] = 1.00   #T
    float_encoded[float_encoded == 4 ] = 0.00 #N
    return float_encoded

#lets try it out a simple short sequences

seq_test = 'TTCAGCCAGTG'
float=ordinal_encoder(string_to_array(seq_test))
print(float)

def one_hot_encoder(seq_string):
    int_encoded = label_encoder.transform(seq_string)
    onehot_encoder = OneHotEncoder(sparse_output=False,dtype=int)
    int_encoded = int_encoded.reshape(len(int_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    onehot_encoded = np.delete(onehot_encoded,-1,1)
    return onehot_encoded

seq_test = 'GAATTCTCGAA'
array=one_hot_encoder(string_to_array(seq_test))
print(array)

def Kmers_funct(seq, size):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]


#So let’s try it out with a simple sequence:
mySeq = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
KMERS = Kmers_funct(mySeq, size=7)
print(KMERS)

words = Kmers_funct(mySeq, size=6)
joined_sentence = ' '.join(words)
print(joined_sentence)


mySeq1 = 'TCTCACACATGTGCCAATCACTGTCACCC'
mySeq2 = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
sentence1 = ' '.join(Kmers_funct(mySeq1, size=6))
print(sentence1)
sentence2 = ' '.join(Kmers_funct(mySeq2, size=6))
print(sentence2)

#Creating the Bag of Words model:
cv = CountVectorizer()
X = cv.fit_transform([joined_sentence, sentence1, sentence2]).toarray()
print(X)
