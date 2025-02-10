

#my $tf_names_file  = "/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_1773/TF.txt";
# Open the TF names
my $tf_names_file  = $ARGV[0];

my $meme_dir = $ARGV[1];
my $ppm_dir  = $ARGV[2];

 #open OUTFILE,">TF_motif" or warn"can't open the file";

unless (open (TF_FILE, $tf_names_file)){ die ("cannot open input file : $tf_names_file\n"); }

# Iterate through each line of the TF file
while (my $line=<TF_FILE>){

	# Read the motif information in to an array
	chomp($line);
	print $line."\n";
	my @array = split(/\s+/,$line);

    #my $motif_file = '/nfs/Lab_Space/bihe/Motif/motif/'.$array[0];
	my $motif_file = $meme_dir.$array[0];


	unless (open (MOTIF_FILE, $motif_file)){ die ("cannot open input file: $motif_file \n"); }

	#my $output_file = '/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_1773/ppm/'.$array[0].".matrix";
	
    $outputFileName = $array[0];
    $outputFileName =~ s/.meme/.ppm/;
	my $output_file = $ppm_dir.$outputFileName;

	unless (open OUTFILE,">$output_file"){ die ("cannot open output file : $output_file\n"); }

	my $count = 0;
	my $count2 = 0;

	my $tempstring1 = "A"." "."|";
	my $tempstring2 = "C"." "."|";
	my $tempstring3 = "G"." "."|";
	my $tempstring4 = "T"." "."|";

	while (my $line1=<MOTIF_FILE>){

		chomp($line1);
		if($line1 =~ /letter-probability/)
		{
			
			$count = 1;
		}
		if($count>0)
		{
			$count2 = $count2 + 1;
		}
		if($count2>1)
		{
			#print $line1."\n";
			$line1 =~ s/^\s+//;
			my @temparray = split(/\s+/,$line1);
			$tempstring1 =  $tempstring1." ".$temparray[0];
			$tempstring2 =  $tempstring2." ".$temparray[1];
			$tempstring3 =  $tempstring3." ".$temparray[2];
			$tempstring4 =  $tempstring4." ".$temparray[3];
		} 

	}

	#print  $tempstring1."\n";
	#print  $tempstring2."\n";
	#print  $tempstring3."\n";
	#print  $tempstring4;
	#print  "\n*********\n";

	print OUTFILE $tempstring1."\n";
	print OUTFILE $tempstring2."\n";
	print OUTFILE $tempstring3."\n";
	print OUTFILE $tempstring4;


}
