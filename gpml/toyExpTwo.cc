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
#include "gp.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 4 || nlhs != 2) /* check the input */
		mexErrMsgTxt("Usage: [e1, e2] = toy_exp2(seeds, X, U, M)");
	size_t n = mxGetM(prhs[1]);
	size_t D = mxGetN(prhs[1]);
	size_t s = mxGetN(prhs[0]);
	size_t M1 = (size_t) mxGetScalar(prhs[3]);
	size_t M = mxGetM(prhs[2]) / D;
	Eigen::VectorXd e1(s);
	e1.setZero();
	Eigen::VectorXd e2(s);
	e2.setZero();
	Eigen::Map<const Eigen::VectorXd> seeds(mxGetPr(prhs[0]), s);
	libgp::CovFactory cf;
	libgp::CovarianceFunction * ardse = cf.create(D, "CovSum ( CovSEard, CovNoise)");
	ardse->init(D);
	Eigen::VectorXd p_se(ardse->get_param_dim());
	p_se.fill(0);
	ardse->set_loghyper(p_se);
	srand(seeds(0));
	srand48(seeds(0));
	Eigen::Map<const Eigen::MatrixXd> X(mxGetPr(prhs[1]), n, D);
	mexPrintf("Sampling input data.\n");
	Eigen::MatrixXd Xtest(n, D);
	for (size_t j = 0; j < n; j++) {
		for (size_t d = 0; d < D; d++) {
			Xtest(j, d) = libgp::Utils::randn();
		}
	}
	mexPrintf("Sampling function.\n");
	Eigen::VectorXd y = ardse->draw_random_sample(X);
	std::cout << "last training target: " << y.tail(1) << std::endl;

	Eigen::VectorXd p(D+2);
	p.head(D+1) = p_se;
	p(D+1) = 1e-25; //log(0.0); //0 noise

	libgp::GaussianProcess gp = libgp::GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
	gp.covf().set_loghyper(p);
	libgp::OptMultiScaleGaussianProcess multiscale = libgp::OptMultiScaleGaussianProcess(D,
					"CovSum ( CovSEard, CovNoise)", M, "SparseMultiScaleGP");
	for(size_t j = 0; j < n; j++){
		gp.add_pattern(X.row(j), y(j));
		multiscale.add_pattern(X.row(j), y(j));
	}
	Eigen::VectorXd ms_params(2 * M * D + D + 2);
	ms_params.head(D).fill(log(M) - log(n));
	ms_params.segment(D, M * D).fill(log(n) - log(M));
	Eigen::Map<const Eigen::VectorXd> U(mxGetPr(prhs[2]), M*D);
	ms_params.segment(M*D+D, M*D) = U;
	ms_params.tail(2) = p.tail(2);
	multiscale.covf().set_loghyper(ms_params);
	Eigen::VectorXd ytest(n);
	for(size_t k = 0; k < n; k++){
		double target = gp.f(Xtest.row(k));
		ytest(k) = target;
		//TODO: standardize
		double pred = (multiscale.f(Xtest.row(k)) - target);
		e2(0) += pred * pred / n;
	}
//	std::cout << ytest.transpose() << std::endl;
	double testvar = (ytest.array() - ytest.mean()).square().sum() / n + 1e-50;
	std::cout << "Test target variance: " << testvar << std::endl;
	for (size_t j = 0; j < s; j++) {
		srand(seeds(j));
		srand48(seeds(j));
		mexPrintf("Initializing Gaussian processes.\n");
		libgp::DegGaussianProcess fastfood = libgp::DegGaussianProcess(D,
				"CovSum ( CovSEard, CovNoise)", M1, "FastFood", seeds(j));
		for(size_t k = 0; k < n; k++)
			fastfood.add_pattern(X.row(k), y(k));
		fastfood.covf().set_loghyper(p);
		mexPrintf("Calculating error.\n");
		for(size_t k = 0; k < n; k++){
			double pred = (fastfood.f(Xtest.row(k)) - ytest(k));
			e1(j) += pred * pred / n;
		}
	}

	e1 /= testvar;
	e2 /= testvar;
	plhs[0] = mxCreateDoubleMatrix(s, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), s) = e1;
	plhs[1] = mxCreateDoubleMatrix(s, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[1]), s) = e2;
}
