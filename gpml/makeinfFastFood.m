me = mfilename; % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me)); % where am I located
cd(mydir)
%this_files_location = 'D:\work\studies\master2\lectures\Dropbox\4. Semester\workspace\src\libgp-1\gpml';
files = '../src/fic_gp.cc ../src/abstract_gp.cc ../src/basis_functions/basisf_factory.cc ../src/basis_functions/bf_multi_scale.cc ../src/gp_utils.cc ../src/sampleset.cc ../src/cov.cc ../src/cov_factory.cc ../src/cov_linear_ard.cc ../src/cov_linear_one.cc ../src/cov_matern3_iso.cc ../src/cov_matern5_iso.cc ../src/cov_noise.cc ../src/cov_periodic_matern3_iso.cc ../src/cov_rq_iso.cc ../src/cov_se_ard.cc ../src/cov_se_iso.cc ../src/cov_prod.cc ../src/cov_sum.cc ../src/cov_periodic.cc ../src/input_dim_filter.cc ../src/gp_deg.cc ../spiral_wht/spiral_wht.c ../spiral_wht/s_*.c ../src/basis_functions/bf_fast_food.cc ';%../spiral_wht/transpose.c ../spiral_wht/transpose_stride.c ../spiral_wht/wht_trees.c ';%../spiral_wht/p_transpose.c';
%eval(['mex -O infFastFoodmex.cc -I../include -I../eigen3 -I../spiral_wht', files]);
eval(['mex -O infFastFoodmex.cc -I../include -I../spiral_wht -I/usr/include/eigen3 ', files]);
%eval('mex -O infFastFoodmex.cc -I../include -I../spiral_wht -I/usr/include/eigen3 -l../../build/libgp.a');
% ../src/basis_functions/*.cc ../src/*.cc');
