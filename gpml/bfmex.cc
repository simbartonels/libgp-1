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
	std::string bf_name;
	size_t seed;
	size_t M;
	size_t D;
	size_t n;
	size_t n2;
	size_t p;
	libgp::CovFactory cfactory;
	if (nrhs < 5 || nlhs != 1) /* check the input */
		mexErrMsgTxt("Usage: k = bfmex(bf_name, seed, M, unwrap(hyp), z, i)");
	bf_name = (std::string) mxGetString(prhs[0]);
	seed = (size_t) mxGetScalar(prhs[1]);
	M = (size_t) mxGetScalar(prhs[2]);
	p = mxGetM(prhs[3]);
	n = mxGetM(prhs[4]);
	D = mxGetN(prhs[4]);
	libgp::CovarianceFunction * ardse = cfactory.create(D,
			"CovSum ( CovSEard, CovNoise)");
	bf = bfactory.createBasisFunction(bf_name, M, ardse, seed);
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]),
			p);
//	std::cout << "bf_multi_scale: params" << std::endl << params << std::endl;

	bf->set_loghyper(params);

	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[4]), n,
			D);
	if (nrhs == 5) {
		plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL); /* allocate space for output */
		Eigen::MatrixXd Phi(M, n);
		for (size_t i = 0; i < n; i++)
			Phi.col(i) = bf->computeBasisFunctionVector(X.col(i));
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = Phi;
	} else {
		//we want the gradients as well
		std::cout << "gradients not implemented" << std::endl;
		plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL); /* allocate space for output */
		Eigen::MatrixXd Phi(M, n);
		Phi.setZero();
//		for (size_t i = 0; i < n; i++)
//			Phi.col(i) = bf->gradBasisFunction(X.col(i));
		Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = Phi;
	}
}

