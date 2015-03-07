#include "mex.h"

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
