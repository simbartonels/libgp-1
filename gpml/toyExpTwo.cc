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
#include "basis_functions/bf_multi_scale.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 7 || nlhs < 2) /* check the input */
		mexErrMsgTxt("Usage: [e1, e2, y] = toy_exp2(seeds, X, paramsMS, M, log(sn2), Xtest, c");
	size_t n = mxGetM(prhs[1]);
	size_t D = mxGetN(prhs[1]);
	size_t s = mxGetN(prhs[0]);
	size_t M1 = (size_t) mxGetScalar(prhs[3]);
	size_t M = (mxGetM(prhs[2]) - D - 2) / D / 2;
	Eigen::MatrixXd c(1, 1);
	c(0, 0) = mxGetScalar(prhs[6]);
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
	p_se(D+1) = mxGetScalar(prhs[4]);//1e-25; //log(0.0); //0 noise
	ardse->set_loghyper(p_se);
	srand(seeds(0));
	srand48(seeds(0));
	Eigen::Map<const Eigen::MatrixXd> X(mxGetPr(prhs[1]), n, D);
	Eigen::Map<const Eigen::MatrixXd> Xtest(mxGetPr(prhs[5]), n, D);
	mexPrintf("Sampling function.\n");
	Eigen::VectorXd y = ardse->draw_random_sample(X);
	std::cout << "last training target: " << y.tail(1) << std::endl;
	//standardize dataset
	y.array() -= y.mean();
	y.array() /= y.array().square().sum() / y.size();

	Eigen::VectorXd p = p_se;

	libgp::GaussianProcess gp = libgp::GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
	gp.covf().set_loghyper(p);
	libgp::OptMultiScaleGaussianProcess multiscale = libgp::OptMultiScaleGaussianProcess(D,
					"CovSum ( CovSEard, CovNoise)", M, "SparseMultiScaleGP");
	((libgp::MultiScale *) &(multiscale.covf()))->setExtraParameters(c);
	for(size_t j = 0; j < n; j++){
		gp.add_pattern(X.row(j), y(j));
		multiscale.add_pattern(X.row(j), y(j));
	}
//	Eigen::VectorXd ms_params(2 * M * D + D + 2);
//	ms_params.head(D).fill(log(M) - log(n));
//	ms_params.segment(D, M * D).fill(log(n) - log(M) - log(2));
//	Eigen::Map<const Eigen::VectorXd> U(mxGetPr(prhs[2]), M*D);
//	ms_params.segment(M*D+D, M*D) = U;
//	ms_params.tail(2) = p.tail(2);
//	std::cout << multiscale.covf().get_loghyper().transpose() << std::endl;
	Eigen::Map<const Eigen::VectorXd> ms_params(mxGetPr(prhs[2]), 2*M*D+D+2);
//	std::cout << M << std::endl;
//	std::cout << ms_params.transpose() << std::endl;
	multiscale.covf().set_loghyper(ms_params);
	Eigen::VectorXd ytest(n);
	for(size_t k = 0; k < n; k++){
		double target = gp.f(Xtest.row(k));
		ytest(k) = target;
		//TODO: standardize
		double pred = (multiscale.f(Xtest.row(k)) - target);
		e2(0) += pred * pred / n;
	}

	std::cout << "ytest.tail(1): " << ytest.tail(1) << std::endl;
	if(!ytest.allFinite())
		mexErrMsgTxt("Test data contains NaN or Inf!");
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
	if(nlhs > 2){
		plhs[2] = mxCreateDoubleMatrix(ytest.size(), 1, mxREAL);
		Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[2]), ytest.size()) = ytest;
	}
}
