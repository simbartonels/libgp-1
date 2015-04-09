#include "mex.h"
#include "basis_functions/basisf_factory.h"
#include "basis_functions/IBasisFunction.h"
#include "cov.h"
#include "cov_factory.h"
#include "abstract_gp.h"
#include "gp_solin.h"
#include "gp_deg.h"
#include "gp_fic.h"
#include "gp_fic_optimized.h"
#include "gp_multiscale_optimized.h"
#include "gp.h"

#if defined(WIN32) || defined(WIN64) || defined(_WIN32) || defined(_WIN64)
#include "winsock2.h"
#else
#include <sys/time.h>
#endif

double tic() {
#if defined(WIN32) || defined(WIN64) || defined(_WIN32) || defined(_WIN64)
	return GetTickCount() / 1000.0;
#else
	struct timeval now;
	gettimeofday(&now, NULL);
	double time_in_seconds = now.tv_sec + now.tv_usec / 1000000.0;
	return time_in_seconds;
#endif
}

/**
 * This function performs consistency checks on the input string and retuns the length of the string.
 */
int getBufferLength(const mxArray *prhs[], size_t param_number) {
	/* Input must be a string. */
	if (!mxIsChar(prhs[param_number])){
		std::stringstream ss;
		ss << "Input " << param_number << "must be a string.";
		mexErrMsgTxt(ss.str().c_str());
	}
	/* Input must be a row vector. */
	if (mxGetM(prhs[param_number]) != 1){
		std::stringstream ss;
		ss << "Input " << param_number << "must be a row vector.";
		mexErrMsgTxt(ss.str().c_str());

	}

	/* Get the length of the basis function name. */
	return mxGetN(prhs[param_number]) + 1;
}

bool checkStatus(int status, const std::string & varName) {
	if (status != 0) {
		std::stringstream ss;
		ss << "Could not read " << varName << ". Status: " << status;
		mexErrMsgTxt(ss.str().c_str());
		return false;
	}
	return true;
}

libgp::AbstractGaussianProcess * constructGP(const std::string & gp_name, size_t D,
		const std::string & cov_name, size_t M, const std::string & bf_name,
		size_t seed) {
	libgp::AbstractGaussianProcess * gp;
	if (M <= 0) {
		mexErrMsgTxt("M must be greater 0!");
//			return;
	}
	if (gp_name.compare("degenerate") == 0) {
		gp = new libgp::DegGaussianProcess(D, cov_name, M, bf_name, seed);
	} else if (gp_name.compare("FIC") == 0) {
		gp = new libgp::FICGaussianProcess(D, cov_name, M, bf_name);
	}else if (gp_name.compare("OptFIC") == 0) {
		gp = new libgp::OptFICGaussianProcess(D, cov_name, M, bf_name);
	}else if (gp_name.compare("OptMultiscale") == 0) {
		gp = new libgp::OptMultiScaleGaussianProcess(D, cov_name, M, bf_name);
	} else if (gp_name.compare("Solin") == 0) {
		gp = new libgp::SolinGaussianProcess(D, cov_name, M);
	} else {
		std::stringstream ss;
		ss << "The GP name " << gp_name
				<< " is unknown. Options are full, degenerate, Solin and FIC.";
		mexErrMsgTxt(ss.str().c_str());
//			return;
	}
	return gp;
}

/**
 * Function that initializes a basis function.
 */
libgp::IBasisFunction * bfmex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/* Input must be a string. */
	if (!mxIsChar(prhs[0]))
		mexErrMsgTxt("Basis function name must be a string.");

	/* Input must be a row vector. */
	if (mxGetM(prhs[0]) != 1)
		mexErrMsgTxt("Basis function name must be a row vector.");

	/* Get the length of the basis function name. */
	int buflen = mxGetN(prhs[0]) + 1;

	/* Allocate memory for basis function name. */
	char * input_buf = (char *) mxCalloc(buflen, sizeof(char));

	/* Copy the string data from prhs[0] into a C string
	 * input_buf. If the string array contains several rows,
	 * they are copied, one column at a time, into one long
	 * string array. */
	int status = mxGetString(prhs[0], input_buf, buflen);
	checkStatus(status, "Basis Function Name");
	std::string bf_name(input_buf);
	size_t seed = (size_t) mxGetScalar(prhs[1]);
	size_t M = (size_t) mxGetScalar(prhs[2]);
	if (M == 0) {
		mexErrMsgTxt("bfmex: M must be greater 0!");
		return 0;
	}
	size_t D = (size_t) mxGetScalar(prhs[4]);
	libgp::CovFactory cfactory;
		libgp::CovarianceFunction * ardse = cfactory.create(D,
			"CovSum ( CovSEard, CovNoise)");
	libgp::BasisFFactory bfactory;
	libgp::IBasisFunction * bf = bfactory.createBasisFunction(bf_name, M, ardse, seed);
	mxFree(input_buf);
	size_t p = mxGetM(prhs[3]);
	if(p != bf->get_param_dim()){
		std::stringstream ss;
		ss << "The desired basis function " << bf->to_string() << " requires " << bf->get_param_dim()
				<< " parameters but received only " << p;
		mexErrMsgTxt(ss.str().c_str());
		return 0;
	}
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]),
			p);
	bf->set_loghyper(params);
	return bf;
}
