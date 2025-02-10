#this code is used to modify the pwm matrix 

#my $tf_names_file = '/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_1773/TF.txt';
my $tf_names_file  = $ARGV[0];
my $pwm_dir  = $ARGV[1];
my $pwm_wo_letter_dir  = $ARGV[2];


unless (open (TF_FILE,$tf_names_file)){ die ("cannot open input file: $tf_names_file  \n"); }  
 

while(my $line=<TF_FILE>)
{
	chomp($line);
	print $line."\n";


	my @array2 = split(/\s+/,$line);
    $inFile = $array2[0];
    $inFile =~ s/.meme/.pwm/;
	my $file_path = $pwm_dir.$inFile;

	unless (open (MYFILE2,$file_path)){ die ("cannot open input file : $file_path\n"); }  
	
    $outputFileName = $array2[0];
    $outputFileName =~ s/.meme/.pwm_wo_letter/;
	my $output_file_path = $pwm_wo_letter_dir.$outputFileName;


	open (OUTFILE, ">$output_file_path") || die("Cannot open file: $output_file_path\n"); 



	while(my $line1=<MYFILE2>)
	{
		chomp($line1);
		my @array = split(/\s/,$line1);

		my $tempstring = '';

		for(my $i=1;$i<scalar(@array);$i++)
		{
			$tempstring = $tempstring." ".$array[$i];
		}

		print OUTFILE $tempstring."\n";

	}

	close(MYFILE2);
	close(OUTFILE);


}
