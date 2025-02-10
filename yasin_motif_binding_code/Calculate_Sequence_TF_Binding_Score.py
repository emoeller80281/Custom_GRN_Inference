#!/usr/bin/python
import sys
import math
import random

pwm_file=sys.argv[1]
seq_file=sys.argv[2]
output_score_file=sys.argv[3]
log_file=sys.argv[4]

PWM=open(pwm_file, 'r')

SEQ=open(seq_file,'r')
OUT=open(output_score_file,'w')
LOG=open(log_file,'w')

print "pwm_file: " + pwm_file
print "seq_file: " + seq_file
print "output_score_file: " + output_score_file


#seq=SEQ.readline()
#seq=seq[:-1]

def score(sequ,motif):
	sequ=sequ.upper()
	Code={'A':0,'C':1,'G':2,'T':3}
	s=0
	for i in range(len(sequ)):
		if sequ[i]=='N':
			ii=random.randint(0,3)
		else:
			#print(sequ[i] + '\n')
			ii=Code[sequ[i]]

		b=float(motif[ii][i])		
		s=s+b

	es=math.pow(2, s)

	return(es)





#The matrix has 4 columns and is now to be transposed
M = []
l = 0
for line in PWM:
   a = line[:-1]
   M.append(a)
   l = l+1
#print M
#print l

Matrix = [[0 for i in range(l)] for j in range(4)]
for i in range(l):
	r = M[i].split('\t')
	Matrix[0][i] = r[0]
	Matrix[1][i] = r[1]
	Matrix[2][i] = r[2]
	Matrix[3][i] = r[3]

#print Matrix
#print 'Starting to calculate scores.'	

#norm_factor = 1000000
norm_factor = 1

def get_cumulative_seq_score(seq, Matrix):

	Len_seq=len(seq)
	random.seed(Len_seq) # for ambiguous reads

	Len_motif=len(Matrix[0])
	cumulative_score=0

	#norm_factor = float(Len_seq-Len_motif+1)


	for i in range(Len_seq-Len_motif+1): 
		s1 = score(seq[i:i+Len_motif],Matrix)
		new_score = float( s1 ) 
		new_score_norm = new_score / norm_factor
		cumulative_score = cumulative_score + new_score_norm

		if(i == 950 or i % 1000000 == 0) :
			LOG=open(log_file,'a')
			print >> LOG, (i/1000000), cumulative_score   
			LOG.close()		

	return(cumulative_score)

 
print >> LOG, 'Processing ', pwm_file   
print >> LOG, 'Scaling factor:', str(norm_factor)   
LOG.close()


seq_line = SEQ.readline() #read the sequence name (coord)

while seq_line:
	seq_name = seq_line[1:-1]

	seq_line = SEQ.readline() # read the seq
	seq=seq_line[:-1]
	cumulative_score = get_cumulative_seq_score(seq, Matrix)
	print >> OUT, seq_name, cumulative_score, str(len(seq))
	seq_line = SEQ.readline()	#read the  sequence name (coord)



#print cumulative_score




