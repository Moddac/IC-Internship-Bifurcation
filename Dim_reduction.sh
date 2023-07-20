cd ~/Practice
venvIPS
python Dim_reduction.py\
	--scheme "cumulant"\
	--noise "OU"\
	--alpha 1.0\
	--theta 4.0\
	--sigma_m 0.0\
	--sigma_start 1.6\
	--sigma_end 2.0\
	--N_sigma 100\
	--N 16\
	--epsilon 0.001\
	--name "Test_OU_fail.json"\
	--delete_file\
	