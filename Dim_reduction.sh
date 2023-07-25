cd ~/Practice
venvIPS
python Dim_reduction.py\
	--scheme "cumulant"\
	--noise "OU"\
	--alpha 1.0\
	--theta 4.0\
	--sigma_m 0.0\
	--gamma 100.0\
	--sigma_start 1.5\
	--sigma_end 2.0\
	--N_sigma 10\
	--N 4\
	--epsilon 0.01\
	--stop_time 600\
	--name "Test_OU_inertia.json" \
	--delete_file \
	