cd /nfs/Lab_Space/yuzun/diabetes/disruption/scripts/
tf_list_file=/nfs/Lab_Space/yuzun/motif_cisbp/TF_list.txt
meme_dir=/nfs/Lab_Space/yuzun/motif_cisbp/meme/
ppm_dir=/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_Cisbp/ppm/
pwm_dir=/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_Cisbp/pwm/
pwm_wo_letter_dir=/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_Cisbp/pwm_wo_letter/

perl step_3_1_meme2ppm.pl $tf_list_file $meme_dir $ppm_dir
perl step_3_2_ppm2pwm.pl $tf_list_file $ppm_dir $pwm_dir
perl step_3_3_pwm_wo_letter.pl $tf_list_file $pwm_dir $pwm_wo_letter_dir

