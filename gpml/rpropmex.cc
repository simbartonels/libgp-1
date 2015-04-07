#include "abstract_gp.h"
#include "gp.h"
#include "gp_deg.h"
#include "gp_fic.h"
#include "gp_solin.h"
#include "rprop.h"

#include "mex.h"
#include "util/util.cc"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>

#define P_GP_NAME 5
#define P_COV_NAME 6
#define P_HYP 7
#define P_M 8
#define P_BF_NAME 9
#define P_CAP_TIME 10
#define P_EXTRA 11

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

	if (nlhs != 6 || nrhs < 8) /* check the input */
		mexErrMsgTxt(
				"Usage: [times, theta_over_time, meanY, varY, nlZ, grad_norms] = rpropmex(seed, iters, X, y, Xtest, gpName, covName, unwrap(hyp), M, bfName, capTime, extraParameters)");
	seed = (size_t) mxGetScalar(prhs[0]);
	n = mxGetM(prhs[2]);
	D = mxGetN(prhs[2]);

	buflen = getBufferLength(prhs, P_GP_NAME);
	gp_name_buf = (char *) mxCalloc(buflen, sizeof(char));
	status = mxGetString(prhs[P_GP_NAME], gp_name_buf, buflen);
	if (!checkStatus(status, "GP name"))
		return;
	std::string gp_name(gp_name_buf);

	buflen = getBufferLength(prhs, P_COV_NAME);
	cov_name_buf = (char *) mxCalloc(buflen, sizeof(char));
	status = mxGetString(prhs[P_COV_NAME], cov_name_buf, buflen);
	if (!checkStatus(status, "Covariance function name"))
		return;
	std::string cov_name(cov_name_buf);
	if (gp_name.compare("full") == 0) {
		gp = new libgp::GaussianProcess(D, cov_name);
	} else {
		if (nrhs < 9) {
			mexErrMsgTxt(
					"Usage: [times, theta_over_time, meanY, varY, nlZ, trainMean] = rpropmex(seed, iters, X, y, Xtest, gpName, covName, unwrap(hyp), M, bfName)");
			return;
		}
		char * bf_name_buf;
		size_t M = mxGetScalar(prhs[P_M]);
		buflen = getBufferLength(prhs, P_BF_NAME);
		bf_name_buf = (char *) mxCalloc(buflen, sizeof(char));
		status = mxGetString(prhs[P_BF_NAME], bf_name_buf, buflen);
		if (!checkStatus(status, "Basis function name"))
			return;
		std::string bf_name(bf_name_buf);
		gp = constructGP(gp_name, D, cov_name, M, bf_name, seed);
		mxFree(bf_name_buf);
		if(nrhs > P_EXTRA){
			Eigen::Map<const Eigen::MatrixXd> extra(mxGetPr(prhs[P_EXTRA]), mxGetM(prhs[P_EXTRA]), mxGetN(prhs[P_EXTRA]));
			((libgp::IBasisFunction *) &(gp->covf()))->setExtraParameters(extra);
		}
	}
	mxFree(gp_name_buf);
	mxFree(cov_name_buf);
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

	double capTime = mxGetScalar(prhs[P_CAP_TIME]);

//	mexPrintf("rpropmex: Initializating GP with hyper-parameters.\n");
	gp->covf().set_loghyper(params);
//	mexPrintf("rpropmex: GP initialization complete. Starting hyper-parameter optimization.\n");
	libgp::RProp rprop;
	rprop.init(1e-4);

	iters = (size_t) mxGetScalar(prhs[1]);
	iters = iters + 1; //for the initial values
	Eigen::VectorXd times(iters);
	times.fill(-1);
	Eigen::MatrixXd theta_over_time(p, iters);
	Eigen::MatrixXd meanY(test_n, iters);
	Eigen::MatrixXd varY(test_n, iters);
	Eigen::VectorXd nlZ(iters);
	Eigen::VectorXd grad_norms(iters);
	std::cout << "starting optimization" << std::endl;
	rprop.maximize(gp, testX, capTime, times, theta_over_time, meanY, varY, nlZ, grad_norms);
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
	plhs[5] = mxCreateDoubleMatrix(iters, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[5]), iters) = grad_norms;

	delete (gp);
	mexPrintf("Hyper-parameter optimization finished.\n");
}

