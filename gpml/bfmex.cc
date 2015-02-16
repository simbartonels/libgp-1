#include "basis_functions/basisf_factory.h"
#include "basis_functions/IBasisFunction.h"
#include "cov.h"
#include "cov_factory.h"

#include "mex.h"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>

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
		mexErrMsgTxt("Usage: k = bfmex(bf_name, seed, M, unwrap(hyp), z, i)");

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
	if(M == 0){
		std::cout << "bfmex: M must be greater 0!" << std::endl;
		exit(-1);
	}
	p = mxGetM(prhs[3]);
	n = mxGetM(prhs[4]);
	D = mxGetN(prhs[4]);
	libgp::CovarianceFunction * ardse = cfactory.create(D,
			"CovSum ( CovSEard, CovNoise)");
	bf = bfactory.createBasisFunction(bf_name, M, ardse, seed);
	mxFree(input_buf);
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]),
			p);
//	std::cout << "bf_multi_scale: params" << std::endl << params << std::endl;

	bf->set_loghyper(params);

	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[4]), n,
			D);

	plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL); /* allocate space for output */

	Eigen::MatrixXd Phi(M, n);
	for (size_t i = 0; i < n; i++)
		Phi.col(i) = bf->computeBasisFunctionVector(X.row(i));

	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = Phi;
	if (nrhs > 5) {
		//we want the gradients
		std::cout << "gradients not implemented" << std::endl;
		Phi.setZero();
//		for (size_t i = 0; i < n; i++)
//			Phi.col(i) = bf->gradBasisFunction(X.col(i));
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = Phi;
	}
}

