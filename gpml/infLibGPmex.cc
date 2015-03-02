#include "abstract_gp.h"
#include "gp_solin.h"
#include "gp_deg.h"
#include "gp_fic.h"
#include "gp.h"

#include "mex.h"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>

#include <sstream>

#define P_X 0
#define P_y 1
#define P_Xtest 2
#define P_GP_NAME 3
#define P_COV_NAME 4
#define P_HYP 5
#define P_M 6
#define P_BF_NAME 7
#define USAGE "Usage: [alpha, L, nlZ, mF, s2F] = infLibGPmex(X, y, Xtest, gpName, covName, unwrap(hyp), M, bfName)"

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
		ss << "inflibgp: Could not read " << varName << ". Status: " << status;
		mexErrMsgTxt(ss.str().c_str());
		return false;
	}
	return true;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	libgp::AbstractGaussianProcess * gp;
	size_t D;
	size_t n;
	size_t M;
	size_t n2;
	size_t p;
	int buflen;
	int status;
	char * gp_name_buf;
	char * cov_name_buf;
	if (nlhs > 5 || nrhs < 6) /* check the input */
		mexErrMsgTxt(USAGE);
	n = mxGetM(prhs[P_X]);
	//M will be overwritten later. It's just easier to define the output that way.
	M = n;
	D = mxGetN(prhs[P_X]);

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
		if (nrhs < P_BF_NAME + 1) {
			mexErrMsgTxt(USAGE);
			return;
		}
		char * bf_name_buf;
		M = mxGetScalar(prhs[P_M]);
		if (M <= 0) {
			mexErrMsgTxt("rpropmex: M must be greater 0!");
			return;
		}
		buflen = getBufferLength(prhs, P_BF_NAME);
		bf_name_buf = (char *) mxCalloc(buflen, sizeof(char));
		status = mxGetString(prhs[P_BF_NAME], bf_name_buf, buflen);
		if (!checkStatus(status, "Basis function name"))
			return;
		std::string bf_name(bf_name_buf);
		if (gp_name.compare("degenerate") == 0) {
			//TODO: do we need a seed as argument?
			gp = new libgp::DegGaussianProcess(D, cov_name, M, bf_name, 0);
		} else if (gp_name.compare("FIC") == 0) {
			gp = new libgp::FICGaussianProcess(D, cov_name, M, bf_name);
		} else if (gp_name.compare("Solin") == 0) {
			gp = new libgp::SolinGaussianProcess(D, cov_name, M, bf_name);
		} else {
			std::stringstream ss;
			ss << "rpropmex: The GP name " << gp_name
					<< " is unknown. Options are full, degenerate, Solin and FIC.";
			mexErrMsgTxt(ss.str().c_str());
			return;
		}
		mxFree(bf_name_buf);
	}
	mxFree(gp_name_buf);
	mxFree(cov_name_buf);

	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[P_X]), n,
			D);
	if (mxGetM(prhs[P_y]) != n) {
		std::stringstream ss;
		ss << "inflibgp: The number of training locations " << n
				<< " does not match the number of training observations "
				<< mxGetM(prhs[P_y]);
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[P_y]), n);
	for (size_t i = 0; i < n; i++)
		gp->add_pattern(X.row(i), y(i));

	if (D != mxGetN(prhs[P_Xtest])) {
		std::stringstream ss;
		ss << "inflibgp: The dimension of training locations " << D
				<< " does not match the dimension of test locations "
				<< mxGetN(prhs[P_Xtest]);
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	size_t test_n = mxGetM(prhs[P_Xtest]);
	Eigen::MatrixXd testX = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[P_Xtest]),
			test_n, D);

	p = mxGetM(prhs[P_HYP]);
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(
			mxGetPr(prhs[P_HYP]), p);
	if (p != gp->covf().get_param_dim()) {
		std::stringstream ss;
		ss << "The length of the initial parameter vector "
				<< params.transpose()
				<< " does not match the number of parameters of the covariance function: "
				<< gp->covf().get_param_dim();
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	mexPrintf("inflibgp: Initializating GP.\n");
	gp->covf().set_loghyper(params);
	mexPrintf(
			"inflibgp: GP initialization complete.\n");
	Eigen::VectorXd meanY(test_n);
	Eigen::VectorXd varY(test_n);
	for(size_t i = 0; i < test_n; i++){
		meanY(i) = gp->f(testX.row(i));
		varY(i) = gp->var(testX.row(i));
	}

	plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL); /* allocate space for output */
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), M) = gp->getAlpha();
	plhs[1] = mxCreateDoubleMatrix(M, M, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), M, M) = gp->getL();
	double nlZ = -gp->log_likelihood();
	plhs[2] = mxCreateDoubleScalar(nlZ);
	plhs[3] = mxCreateDoubleMatrix(test_n, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[3]), test_n) = meanY;
	plhs[4] = mxCreateDoubleMatrix(test_n, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[4]), test_n) = varY;

	delete gp;
}

