
## Homer
### Installing Homer

Download Homer from `http://homer.ucsd.edu/homer/configureHomer.pl` into the 'Homer' directory.

In the terminal, navigate to the 'Homer' directory.

Run `perl ./configureHomer.pl -install`.

Add Homer to your system path using `PATH=$PATH:<Homer absolute path>/.//bin/`.

Install Homer's human genome using `perl ./configureHomer.pl -install hg38`.

OR

Install Homer's mouse genome using `perl ./configureHomer.pl -install mm10`.