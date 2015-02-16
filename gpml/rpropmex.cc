#include "abstract_gp.h"
#include "gp.h"
#include "gp_deg.h"
#include "gp_fic.h"
#include "rprop.h"

#include "mex.h"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>

std::stringstream ss;

/**
 * This function performs consistency checks on the input string and retuns the length of the string.
 */
int getBufferLength(const mxArray *prhs[], size_t param_number) {
	/* Input must be a string. */
	if (!mxIsChar(prhs[param_number]))
		mexErrMsgTxt("GP name must be a string.");

	/* Input must be a row vector. */
	if (mxGetM(prhs[param_number]) != 1)
		mexErrMsgTxt("GP name must be a row vector.");

	/* Get the length of the basis function name. */
	return mxGetN(prhs[param_number]) + 1;
}

bool checkStatus(int status, const std::string & varName) {
	if (status != 0) {
		std::stringstream ss;
		ss << "rpropmex: Could not read " << varName << ". Status: " << status;
		mexErrMsgTxt(ss.str().c_str());
		return false;
	}
	return true;
}

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

	if (nrhs < 7 || nlhs > 2) /* check the input */
		mexErrMsgTxt(
				"Usage: [hyp, theta_over_time] = rpropmex(seed, iters, X, y, gpName, covName, unwrap(hyp), M, bfName)");
	seed = (size_t) mxGetScalar(prhs[0]);
	iters = (size_t) mxGetScalar(prhs[1]);
	n = mxGetM(prhs[2]);
	D = mxGetN(prhs[2]);

	buflen = getBufferLength(prhs, 4);
	gp_name_buf = (char *) mxCalloc(buflen, sizeof(char));
	status = mxGetString(prhs[4], gp_name_buf, buflen);
	if (!checkStatus(status, "GP name"))
		return;
	std::string gp_name(gp_name_buf);

	buflen = getBufferLength(prhs, 5);
	cov_name_buf = (char *) mxCalloc(buflen, sizeof(char));
	status = mxGetString(prhs[5], cov_name_buf, buflen);
	if (!checkStatus(status, "Covariance function name"))
		return;
	std::string cov_name(cov_name_buf);
	if (gp_name.compare("full") == 0) {
		gp = new libgp::GaussianProcess(D, cov_name);
	} else {
		if (nrhs < 9) {
			mexErrMsgTxt(
					"Usage: [hyp, theta_over_time] = rpropmex(seed, iters, X, y, gpName, covName, unwrap(hyp), M, bfName)");
			return;
		}
		char * bf_name_buf;
		size_t M = mxGetScalar(prhs[7]);
		if (M <= 0) {
			mexErrMsgTxt("rpropmex: M must be greater 0!");
			return;
		}
		buflen = getBufferLength(prhs, 8);
		bf_name_buf = (char *) mxCalloc(buflen, sizeof(char));
		status = mxGetString(prhs[8], bf_name_buf, buflen);
		if (!checkStatus(status, "Basis function name"))
			return;
		std::string bf_name(bf_name_buf);
		if (gp_name.compare("degenerate") == 0) {
			gp = new libgp::DegGaussianProcess(D, cov_name, M, bf_name, seed);
		} else if (gp_name.compare("FIC") == 0) {
			gp = new libgp::FICGaussianProcess(D, cov_name, M, bf_name);
		} else {
			std::stringstream ss;
			ss << "rpropmex: The GP name " << gp_name
					<< " is unknown. Options are full, degenerate and FIC.";
			mexErrMsgTxt(ss.str().c_str());
			return;
		}
		mxFree(bf_name_buf);
	}
	mxFree(gp_name_buf);
	mxFree(cov_name_buf);

	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[2]), n,
			D);
	if (mxGetM(prhs[3]) != n) {
		std::stringstream ss;
		ss << "rpropmex: The number of training locations " << n
				<< " does not match the number of training observations " << mxGetM(prhs[3]);
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]), n);
	for (size_t i = 0; i < n; i++)
		gp->add_pattern(X.row(i), y(i));

	p = mxGetM(prhs[6]);
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[6]),
			p);
	if (p != gp->covf().get_param_dim()) {
		std::stringstream ss;
		ss << "The length of the initial parameter vector "
				<< params.transpose()
				<< " does not match the number of parameters of the covariance function: "
				<< gp->covf().get_param_dim();
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	gp->covf().set_loghyper(params);
	libgp::RProp rprop;
	if (nlhs == 2) {
		plhs[1] = mxCreateDoubleMatrix(p + 1, iters, mxREAL);
		Eigen::MatrixXd theta_over_time(p + 1, iters);
		rprop.maximize(gp, theta_over_time);
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), p + 1, iters) =
				theta_over_time;
	} else if (nlhs == 1) {
		rprop.maximize(gp, iters, false);
	}
	plhs[0] = mxCreateDoubleMatrix(p, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), p) =
			gp->covf().get_loghyper();
	delete (gp);
	mexPrintf("Hyper-parameter optimization finished.");
}

