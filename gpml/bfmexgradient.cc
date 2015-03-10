#include "util/util.cc"

#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>

std::stringstream ss;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs < 6 || nlhs != 1) /* check the input */
		mexErrMsgTxt("Usage: k = bfmex(bf_name, seed, M, unwrap(hyp), D, di, z)");
	libgp::IBasisFunction * bf = bfmex(nlhs, plhs, nrhs, prhs);

	size_t M = bf->getNumberOfBasisFunctions();
	size_t D = bf->get_input_dim();
	size_t di = (size_t) mxGetScalar(prhs[5]);
	if (nrhs >= 7) {
		//compute basis function
		size_t n = mxGetM(prhs[6]);
		if(D != mxGetN(prhs[6])){
			mexErrMsgTxt("The 2nd dimension of X disagrees with D. Aborting! Call without arguments to see the correct usage.");
			return;
		}

		Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[6]),
				n, D);

		plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL); /* allocate space for output */

		Eigen::MatrixXd Phi(M, n);
		for (size_t i = 0; i < n; i++){
			Eigen::VectorXd phi = bf->computeBasisFunctionVector(X.row(i));
			Eigen::VectorXd grad(M);
			grad.setZero();
			bf->gradBasisFunction(X.row(i), phi, di, grad);
			Phi.col(i) = grad;
		}
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = Phi;
	}
	else {
		//return Sigma
		plhs[0] = mxCreateDoubleMatrix(M, M, mxREAL);
		Eigen::MatrixXd diSigmadp(M, M);
		diSigmadp.setZero();
		bf->gradiSigma(di, diSigmadp);
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, M) = diSigmadp;
	}
}

