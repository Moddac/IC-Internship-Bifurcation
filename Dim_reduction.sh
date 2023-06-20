cd ~/Practice
venvIPS
python Dim_reduction.py\
       	--scheme "moment"\
	--alpha 1\
	--theta 4\
	--sigma_m 0.8\
	--sigma_start 2.0\
	--sigma_end 3.0\
	--N_sigma 50\
	--N 4 8\
	--name "Test_OU_matrix.json"\
	--delete_file
