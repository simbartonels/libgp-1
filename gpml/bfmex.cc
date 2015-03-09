#include "util/util.cc"

#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>

std::stringstream ss;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs < 5 || nlhs != 1) /* check the input */
		mexErrMsgTxt("Usage: k = bfmex(bf_name, seed, M, unwrap(hyp), D, z)");
	libgp::IBasisFunction * bf = bfmex(nlhs, plhs, nrhs, prhs);

	size_t M = bf->getNumberOfBasisFunctions();
	size_t D = bf->get_input_dim();
	if (nrhs >= 6) {
		//compute basis function
		size_t n = mxGetM(prhs[5]);
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
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, M) = bf->getInverseOfSigma();
	}
}

