me = mfilename; % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me)); % where am I located
cd(mydir)
BESTTREE = '''"besttrees.default"''';
files = '../src/gp_fic.cc ../src/abstract_gp.cc ../src/basis_functions/basisf_factory.cc ../src/basis_functions/bf_multi_scale.cc ../src/gp_utils.cc ../src/sampleset.cc ../src/cov.cc ../src/cov_factory.cc ../src/cov_linear_ard.cc ../src/cov_linear_one.cc ../src/cov_matern3_iso.cc ../src/cov_matern5_iso.cc ../src/cov_noise.cc ../src/cov_periodic_matern3_iso.cc ../src/cov_rq_iso.cc ../src/cov_se_ard.cc ../src/cov_se_iso.cc ../src/cov_prod.cc ../src/cov_sum.cc ../src/cov_periodic.cc ../src/input_dim_filter.cc ../src/gp_deg.cc ../src/basis_functions/bf_fast_food.cc ../src/basis_functions/bf_solin.cc ';%../spiral_wht/transpose.c ../spiral_wht/transpose_stride.c ';%../spiral_wht/p_transpose.c';
%eval(['mex -O infFastFoodmex.cc -I../include -I../eigen3 -I../spiral_wht', files]);
files1 = [files, '../spiral_wht/spiral_wht.c ../spiral_wht/s_*.c ../spiral_wht/wht_trees.c '];
%eval(['mex -O infFastFoodmex.cc -I../include -I../spiral_wht -I/usr/include/eigen3 ', files1, ' DBESTTREE = ', BESTTREE]);

%eval(['mex -O infFastFoodmex.cc -Llibwht.a -I../include -I../spiral_wht -I/usr/include/eigen3 ', files]);
eval(['mex -O bfmex.cc libwht.a -DBUILD_FAST_FOOD -I../include -I../spiral_wht -I/usr/include/eigen3 ', files]);
%TODO: one of the reasons this line fails could be a 32 / 64 bit issue
%files='../../build/libgp.a';
%eval(['mex -O bfmex.cc -I../include -I/usr/include/eigen3 -I../spiral_wht ', files]);

