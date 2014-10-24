/* solve_chol - solve a linear system A*X = B using the cholesky factorization
 of A (where A is square, symmetric and positive definite.

 Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-12-16. */

#include "mex.h"
#include <math.h>
#include <string.h>
#include "fic_gp.h"
#include <Eigen/Dense>
#include <iostream>

//extern int dpotrs_(char *, long *, long *, double *, long *, double *, long *, long *);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	size_t M;
	size_t D;
	size_t n;
	size_t p;
	if (nrhs != 4 || nlhs != 2) /* check the input */
		mexErrMsgTxt("Usage: [alpha, L] = infSMmex(M, unwrap(hyp), x, y)");
	M = (size_t) mxGetScalar(prhs[0]);
	n = mxGetM(prhs[2]);
	D = mxGetN(prhs[2]);
	std::cout << "input dimensionality: " << D << std::endl;
	std::cout << "number of basis function: " << M << std::endl;
	std::cout << "number of points: " << n << std::endl;
	p = mxGetM(prhs[1]);
	libgp::FICGaussianProcess gp(D, "CovSum ( CovSEiso, CovNoise)", M,
			"SparseMultiScaleGP");
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[1]),
			p);
//	std::cout << "bf_multi_scale: params" << std::endl << params << std::endl;

	gp.covf().set_loghyper(params);

	Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]), n);
	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[1]), n,
			D);
//	std::cout << "infSMmex: X:" << std::endl << X << std::endl;

	for (size_t i=0; i < n; i++) {
		gp.add_pattern(X.row(i), y(i));
	}

	plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL); /* allocate space for output */
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = gp.getAlpha();
	plhs[1] = mxCreateDoubleMatrix(M, M, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), M, M) = gp.getL();
}

