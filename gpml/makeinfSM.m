eval('mex -O infSMmex.cc -I../include -I../eigen3 ../src/basis_functions/*.cc ../src/*.cc');
%../src/fic_gp.cc ../src/abstract_gp.cc ../src/basis_functions/basisf_factory.cc ../src/cov.cc ../src/sampleset.cc ../src/cov_factory.cc ../src/cov_noise.cc ../src/cov_se_ard.cc ../src/basis_functions/bf_multi_scale.cc ../src/gp_utils.cc');
%-L../src');
