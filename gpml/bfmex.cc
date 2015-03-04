#include "basis_functions/basisf_factory.h"
#include "basis_functions/IBasisFunction.h"
#include "cov.h"
#include "cov_factory.h"

#include "mex.h"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>

std::stringstream ss;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	libgp::BasisFFactory bfactory;
	libgp::IBasisFunction * bf;
	size_t seed;
	size_t M;
	size_t D;
	size_t n;
	size_t n2;
	size_t p;
	libgp::CovFactory cfactory;
	if (nrhs < 5 || nlhs != 1) /* check the input */
		mexErrMsgTxt("Usage: k = bfmex(bf_name, seed, M, unwrap(hyp), D, z)");

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

	if (status != 0) {
		//something went wrong
	}
	std::string bf_name(input_buf);
//	std::cout << "bfmex: Using basis function: " << bf_name << std::endl;
	seed = (size_t) mxGetScalar(prhs[1]);
	M = (size_t) mxGetScalar(prhs[2]);
	if (M == 0) {
		mexErrMsgTxt("bfmex: M must be greater 0!");
		return;
	}
	D = (size_t) mxGetScalar(prhs[4]);
	libgp::CovarianceFunction * ardse = cfactory.create(D,
			"CovSum ( CovSEard, CovNoise)");
	bf = bfactory.createBasisFunction(bf_name, M, ardse, seed);
	mxFree(input_buf);
	p = mxGetM(prhs[3]);
	if(p != bf->get_param_dim()){
		std::stringstream ss;
		ss << "The desired basis function " << bf->to_string() << " requires " << bf->get_param_dim()
				<< " parameters but received only " << p;
		mexErrMsgTxt(ss.str().c_str());
		return;
	}
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]),
			p);
	bf->set_loghyper(params);

	if (nrhs >= 6) {
		//compute basis function
		n = mxGetM(prhs[5]);
		if(D != mxGetN(prhs[5])){
			mexErrMsgTxt("The 2nd dimension of X disagrees with D. Aborting! Call without arguments to see the correct usage.");
			return;
		}

		Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[5]),
				n, D);

		plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL); /* allocate space for output */

		Eigen::MatrixXd Phi(M, n);
		for (size_t i = 0; i < n; i++)
			Phi.col(i) = bf->computeBasisFunctionVector(X.row(i));

		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = Phi;
	}
	else {
		//return Sigma
		plhs[0] = mxCreateDoubleMatrix(M, M, mxREAL);
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, M) = bf->getSigma();
	}
}

