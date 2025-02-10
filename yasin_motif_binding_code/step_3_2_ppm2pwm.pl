
#this is the code for generate the pwm from ppm 

#my $tf_names_file = '/nfs/Lab_Space/yuzun/diabetes/disruption/data/TF_1773/TF.txt';
my $tf_names_file  = $ARGV[0];
my $ppm_dir  = $ARGV[1];
my $pwm_dir  = $ARGV[2];

unless (open (tf_names_file,$tf_names_file)){ die ("Cannot open tf list file: $tf_names_file\n"); }  
my $c= 0.21;
my $at= 0.29;

while(my $line=<tf_names_file>)
{
	chomp($line);
	my @array2 = split(/\s+/,$line);
    
    $inFile = $array2[0];
    $inFile =~ s/.meme/.ppm/;
	my $file_name = $ppm_dir.$inFile;
	print $file_name."\n";
	unless (open (PPM_FILE,$file_name)){ die ("Cannot open input file: $file_name\n"); }


    $outputFileName = $array2[0];
    $outputFileName =~ s/.meme/.pwm/;
	my $output_file = $pwm_dir.$outputFileName;

	open (OUTFILE, ">$output_file") || die("Cannot open output file: $output_file\n"); 

	while(my $line1=<PPM_FILE>)
	{
		chomp($line1);
		my @array = split(/\s/,$line1);
		my $tempstring = $array[0];
        
		print $line1."\n";
		if(($array[0] =~ /A/)||($array[0] =~ /T/))
		{
			for(my $i=2;$i<scalar(@array);$i++)
			{
				if(($array[$i] == 0)||($array[$i] eq '0.000000')||($array[$i] eq '0.0')||($array[$i] eq '0'))
				{
					$tempstring = $tempstring." "."0";
				}
				else
				{
					my $tempvalue = $array[$i]/$at;
					print $i."\t".$array[$i]."\t".$tempvalue."\n";
					$tempvalue = log($tempvalue)/log(2);
					print $array[$i]."\t".$tempvalue."\n";
					$tempstring = $tempstring." ".$tempvalue;
				}
			}

		}
		else
		{
			for(my $i=2;$i<scalar(@array);$i++)
			{
					if(($array[$i] == 0)||($array[$i] eq '0.000000')||($array[$i] eq '0.0')||($array[$i] eq '0'))
					{
					   $tempstring = $tempstring." "."0";
					}
					else
					{
						my $tempvalue = $array[$i]/$gc;
						$tempvalue = log($tempvalue)/log(2);
						$tempstring = $tempstring." ".$tempvalue;
					}
			}

		}

		print OUTFILE $tempstring."\n";

	}#while(my $line1=<PPM_FILE>)

}#while(my $line=<tf_names_file>)



