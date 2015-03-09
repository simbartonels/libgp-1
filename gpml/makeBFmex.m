me = mfilename; % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me)); % where am I located
cd(mydir)
files = '../src/gp_fic.cc ../src/abstract_gp.cc ../src/basis_functions/basisf_factory.cc ../src/basis_functions/bf_multi_scale.cc  ../src/basis_functions/bf_fic.cc ../src/gp_utils.cc ../src/sampleset.cc ../src/cov.cc ../src/cov_factory.cc ../src/cov_linear_ard.cc ../src/cov_linear_one.cc ../src/cov_matern3_iso.cc ../src/cov_matern5_iso.cc ../src/cov_noise.cc ../src/cov_periodic_matern3_iso.cc ../src/cov_rq_iso.cc ../src/cov_se_ard.cc ../src/cov_se_iso.cc ../src/cov_prod.cc ../src/cov_sum.cc ../src/cov_periodic.cc ../src/input_dim_filter.cc ../src/gp_deg.cc ../src/basis_functions/bf_solin.cc ';
spiral_compiled = exist('../spiral_wht/transpose.h', 'file') == 2;
%TODO: make Eigen include relative...
if spiral_compiled
    files = [files  '../src/basis_functions/bf_fast_food.cc'];
    eval(['mex -O bfmex.cc libwht.a -DBUILD_FAST_FOOD -I../include -I../spiral_wht -I/usr/include/eigen3 ', files]);
else
    %FastFoodFree version
    eval(['mex -O bfmex.cc -I../include -I../eigen3 ', files]);
end
