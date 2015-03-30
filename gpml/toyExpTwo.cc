#include <cmath>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>
#include <mex.h>
#include "cov.h"
#include "cov_factory.h"
#include "gp_utils.h"
#include "gp_deg.h"
#include "gp_multiscale_optimized.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 5 || nlhs != 2) /* check the input */
		mexErrMsgTxt("Usage: [e1, e2] = toy_exp2(seeds, n, D, M1, M2)");
	size_t n = (size_t) mxGetScalar(prhs[1]);
	size_t D = (size_t) mxGetScalar(prhs[2]);;
	size_t s = mxGetN(prhs[0]);;
	size_t M1 = (size_t) mxGetScalar(prhs[3]);;
	size_t M = (size_t) mxGetScalar(prhs[4]);;
	Eigen::VectorXd e1(s);
	e1.setZero();
	Eigen::VectorXd e2(s);
	e2.setZero();
	Eigen::Map<Eigen::VectorXd> seeds(mxGetPr(prhs[0]), s);
	libgp::CovFactory cf;
	libgp::CovarianceFunction * ardse = cf.create(D, "CovSum ( CovSEard, CovNoise)");
	ardse->init(D);
	Eigen::VectorXd p_se(ardse->get_param_dim());
	p_se.fill(0);
	ardse->set_loghyper(p_se);
	srand(seeds(0));
	srand48(seeds(0));
	Eigen::MatrixXd X(2*n, D);
	mexPrintf("Sampling input data.\n");
	Eigen::VectorXd y(2*n);
	for (size_t j = 0; j < n; j++) {
		for (size_t d = 0; d < D; d++) {
			X(j, d) = libgp::Utils::randn();
			//Xstar
			X(n+j, d) = libgp::Utils::randn();
		}
	}
	mexPrintf("Sampling function.\n");
	Eigen::VectorXd y = ardse->draw_random_sample(X);

	std::cout << "last testing target: " << y.tail(0) << std::endl;

	Eigen::VectorXd p(D+2);
	p.head(D+1) = p_se;
	p(D+1) = -1; //0 noise

	libgp::OptMultiScaleGaussianProcess multiscale = libgp::OptMultiScaleGaussianProcess(D,
					"CovSum ( CovSEard, CovNoise)", M, "SparseMultiScaleGP");
	for(size_t j = 0; j < n; j++)
		multiscale.add_pattern(X.row(j), y(j));

	Eigen::VectorXd ms_params(2 * M * D + D + 2);
	ms_params.head(D) = p.head(D) * n / M;
	ms_params.segment(D, M * D).fill(((double) M) / n);

	for (size_t j = 0; j < s; j++) {
		srand(seeds(j));
		srand48(seeds(j));
		mexPrintf("Initializing Gaussian processes.\n");
		libgp::DegGaussianProcess fastfood = libgp::DegGaussianProcess(D,
				"CovSum ( CovSEard, CovNoise)", M1, "FastFood", seeds(j));
		for(size_t k = 0; k < n; k++)
			fastfood.add_pattern(X.row(k), y(k));
		fastfood.covf().set_loghyper(p);
		//not a good idea => multiscale would get X as inducing inputs!
//		srand(seeds(j));
//		srand48(seeds(j));
		for (size_t d = 0; d < M*D; d++) {
			ms_params(M * D + D + d) = libgp::Utils::randn();
		}
		ms_params.tail(2) = p.tail(2);
		multiscale.covf().set_loghyper(ms_params);
		mexPrintf("Calculating error.\n");
		for(size_t k = 0; k < n; k++){
			double pred = (fastfood.f(X.row(n+k)) - y(n+k));
			e1(j) += pred * pred / n;
			pred = (multiscale.f(X.row(n+k)) - y(n+k));
			e2(j) += pred * pred / n;
		}
	}
	plhs[0] = mxCreateDoubleMatrix(s, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), s) = e1;
	plhs[1] = mxCreateDoubleMatrix(s, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[1]), s) = e2;
}
