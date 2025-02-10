use File::Basename;

#$cell = $ARGV[0];
#@all_pwm_matrices = glob '/mnt/isilon/cbmi/tan_lab/uzuny/motifs/cisbp_human_direct/PSSM_longest_motif/*.matrix';
#$enhancer_file="/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Data/enhancers_mid1K_$cell.fa";
#$enhancer_file="/mnt/isilon/cbmi/tan_lab/uzuny/genomes/mm9/chr1.labeled.fa";
#$seq_dir = "/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Data/enhancer_sequences/$cell/";
#$sequence_score_dir = "/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Data/enhancer_scores/$cell/";
#$job_dir="/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Scripts/calc_jobs/$cell/";
#$submission_file = "/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Scripts/submit_jobs/submit_enhancer_score_jobs.$cell.sh";


$pwm_dir = $ARGV[0];
$sequence_file = $ARGV[1];
$sequence_score_dir = $ARGV[2];
$job_dir = $ARGV[3];
$submission_file = $ARGV[4];

print "PWM dir: $pwm_dir \n";
print "sequence_file: $sequence_file \n";
print "sequence_score_dir: $sequence_score_dir \n";
print "job_dir: $job_dir \n";
print "submission_file: $submission_file \n";

print "*****************\n";
#@all_pwm_matrices = glob '/mnt/isilon/cbmi/tan_lab/uzuny/motifs/cisbp_human_direct/PSSM_longest_motif/*.matrix';
#$enhancer_file="/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Data/enhancers_mid1K_$cell.fa";
#$seq_dir = "/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Data/enhancer_sequences/$cell/";
#$sequence_score_dir = "/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Data/enhancer_scores/$cell/";
#$job_dir="/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Scripts/calc_jobs/$cell/";
#$submission_file = "/mnt/isilon/cbmi/tan_lab/uzuny/t1d/trn/ruimin/Yasin/T1D/Scripts/submit_jobs/submit_enhancer_score_jobs.$cell.sh";

@all_pwm_matrices = glob "$pwm_dir/*.matrix";

#print "Clearing old files. \n";

#system("rm -r $job_dir");
system("mkdir -p $job_dir");

#print "Cleared old job files. \n";



#system("rm -r $sequence_score_dir");
system("mkdir -p $sequence_score_dir");

#print "Cleared old score files. \n";


open(SUBMISSION, ">$submission_file") or die("Cannot open '$submission_file' \n !");



print("Reading PWM matrices ...\n");
$counter = 0;
while($all_pwm_matrices[$counter]){
	$pwm_file = $all_pwm_matrices[$counter];
   $counter++;

	$motif = basename($pwm_file);
   print("Reading $motif \n");
	$motif1 = $motif ;
   $motif1 =~ s/@/-/;

   $job_file = "$job_dir/sequence_score_$motif1.sh"; 
   $log_file = "$job_dir/sequence_score_$motif1.log"; 
   $py_log_file = "$job_dir/sequence_score_$motif1.py_log";    
   $sequence_score_file = $sequence_score_dir.$motif.".sequence_scores";
   #$enhancer_score_last_file = $sequence_score_dir.$motif.".enhancer_scores.last";

	open(JOB, ">$job_file");

   print JOB "source ~/.bashrc\n";
	print JOB "rm -f $sequence_score_file\n";

	print JOB "/usr/bin/python Calculate_Sequence_TF_Binding_Score.py $pwm_file $sequence_file $sequence_score_file $py_log_file \n";


   print JOB "echo Processed $pwm_file \n";
   close JOB;
  
   print SUBMISSION "qsub -l h_vmem=8G -e $log_file -o $log_file $job_file \n";
   #print SUBMISSION "sleep 1\n";
   print SUBMISSION "echo Submitted $job_file .\n\n";


}

print("$counter PWM matrices are read ...\n*************\n");

close(SUBMISSION);
print("Jobs are ready to submit in file: \n $submission_file \n");




  
