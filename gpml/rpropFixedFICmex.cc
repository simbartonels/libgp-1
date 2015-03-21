#include "rprop.h"
#include "basis_functions/bf_fic_fixed.h"

#include "util/util.cc"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>

#define P_HYP 5
#define P_M 6
#define P_U 7

std::stringstream ss;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	libgp::AbstractGaussianProcess * gp;
	size_t seed;
	size_t iters;
	size_t D;
	size_t n;
	size_t n2;
	size_t p;
	int buflen;
	int status;
	char * gp_name_buf;
	char * cov_name_buf;

	if (nlhs != 5 || nrhs != 8) /* check the input */
		mexErrMsgTxt(
				"Usage: [times, theta_over_time, meanY, varY, nlZ] = rpropmex(seed, iters, X, y, Xtest, unwrap(hyp), M, U)");
	seed = (size_t) mxGetScalar(prhs[0]);
	iters = (size_t) mxGetScalar(prhs[1]);
	n = mxGetM(prhs[2]);
	D = mxGetN(prhs[2]);

	size_t M = mxGetScalar(prhs[P_M]);
	gp = constructGP("FIC", D, "CovSum(CovSEard, CovNoise)", M, "FICfixed", seed);
	if(mxGetM(prhs[P_U]) != M || mxGetN(prhs[P_U]) != D){
		std::stringstream ss;
		ss << "rpropmex: The size of the inducing points matrix (" << mxGetM(prhs[P_U]) << ", "
			<< mxGetN(prhs[P_U]) <<") does not match M=" << M << " and D=" << D;
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	Eigen::Map<const Eigen::MatrixXd> U(mxGetPr(prhs[P_U]), M, D);
	((libgp::FICfixed * ) &(gp->covf()))->setU(U);
	std::cout << "rpropmex: GP instantiated." << std::endl;
	Eigen::Map<const Eigen::MatrixXd> X(mxGetPr(prhs[2]), n,
			D);
	if (mxGetM(prhs[3]) != n) {
		std::stringstream ss;
		ss << "rpropmex: The number of training locations " << n
				<< " does not match the number of training observations "
				<< mxGetM(prhs[3]);
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	Eigen::Map<const Eigen::VectorXd> y(mxGetPr(prhs[3]), n);
	for (size_t i = 0; i < n; i++)
		gp->add_pattern(X.row(i), y(i));

	if (D != mxGetN(prhs[4])) {
		std::stringstream ss;
		ss << "rpropmex: The dimension of training locations " << D
				<< " does not match the dimension of test locations "
				<< mxGetN(prhs[4]);
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	size_t test_n = mxGetM(prhs[4]);
	Eigen::Map<const Eigen::MatrixXd> testX(mxGetPr(prhs[4]),
			test_n, D);
	std::cout << "rpropmex: Data sets transferred." << std::endl;
	p = mxGetM(prhs[P_HYP]);
	if (p != gp->covf().get_param_dim()) {
		std::stringstream ss;
		ss << "The length of the initial parameter vector " << p
				<< " does not match the number of parameters of the covariance function: "
				<< gp->covf().get_param_dim();
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	Eigen::Map<const Eigen::VectorXd> params(mxGetPr(prhs[P_HYP]), p);

//	mexPrintf("rpropmex: Initializating GP with hyper-parameters.\n");
	gp->covf().set_loghyper(params);
//	mexPrintf("rpropmex: GP initialization complete. Starting hyper-parameter optimization.\n");
	libgp::RProp rprop;
	rprop.init(1e-12);
	Eigen::VectorXd times(iters);
	times.fill(-1);
	Eigen::MatrixXd theta_over_time(p, iters);
	Eigen::MatrixXd meanY(test_n, iters);
	Eigen::MatrixXd varY(test_n, iters);
	Eigen::VectorXd nlZ(iters);
	std::cout << "starting optimization" << std::endl;
	rprop.maximize(gp, testX, times, theta_over_time, meanY, varY, nlZ);
	plhs[0] = mxCreateDoubleMatrix(iters, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), iters) = times;
	plhs[1] = mxCreateDoubleMatrix(p, iters, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), p, iters) = theta_over_time;
	plhs[2] = mxCreateDoubleMatrix(test_n, iters, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[2]), test_n, iters) = meanY;
	plhs[3] = mxCreateDoubleMatrix(test_n, iters, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[3]), test_n, iters) = varY;
	plhs[4] = mxCreateDoubleMatrix(iters, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[4]), iters) = nlZ;

	delete (gp);
	mexPrintf("Hyper-parameter optimization finished.\n");
}

