cd ~/Practice
venvIPS
python Dim_reduction.py\
       	--scheme "cumulant"\
	--noise "OU"\
	--alpha 1\
	--theta 4\
	--sigma_m 0.0\
	--sigma_start 1.7\
	--sigma_end 2.0\
	--N_sigma 20\
	--N 6\
	--name "Test_OU_eps_k.json"\
	--delete_file\
	--epsilon 0.001
