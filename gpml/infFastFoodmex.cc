/* solve_chol - solve a linear system A*X = B using the cholesky factorization
 of A (where A is square, symmetric and positive definite.

 Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-12-16. */

#include "mex.h"
#include <math.h>
#include <string.h>
#include "basis_functions/bf_fast_food.h"
#include "gp_deg.h"
#include <Eigen/Dense>
#include <iostream>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	//TODO: create abstract class
	size_t M;
	size_t D;
	size_t n;
	size_t p;
	if (nrhs != 4 || nlhs < 2){ /* check the input */
		mexErrMsgTxt(
						"Usage: [alpha, L, nlZ, dnlZ] = infFastFoodmex(M, unwrap(hyp), x, y)");
		mexErrMsgTxt(
						"Usage: [alpha, L, nlZ, s, g, pi, b] = infFastFoodmex(M, unwrap(hyp), x, y)");
	}
	M = (size_t) mxGetScalar(prhs[0]);
	n = mxGetM(prhs[2]);
	D = mxGetN(prhs[2]);
	std::cout << "input dimensionality: " << D << std::endl;
	std::cout << "number of basis function: " << M << std::endl;
	std::cout << "number of points: " << n << std::endl;
	p = mxGetM(prhs[1]);
	libgp::DegGaussianProcess gp(D, "CovSum ( CovSEard, CovNoise)", M,
			"FastFood");
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[1]),
			p);

	gp.covf().set_loghyper(params);

	Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]), n);
	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[2]), n,
			D);

	for (size_t i = 0; i < n; i++) {
		gp.add_pattern(X.row(i), y(i));
	}

	plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL); /* allocate space for output */
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), M) = gp.getAlpha();
	plhs[1] = mxCreateDoubleMatrix(M, M, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), M, M) = gp.getL();
	if (nlhs >= 3) {
		double nlZ = gp.log_likelihood();
		plhs[2] = mxCreateDoubleScalar(nlZ);
		if (nlhs == 4) {
			std::cout << "infFastFood: computing gradients" << std::endl;
			Eigen::VectorXd grads = gp.log_likelihood_gradient();
			size_t params = grads.size();
			plhs[3] = mxCreateDoubleMatrix(params, 1, mxREAL); /* allocate space for output */
			Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[3]), params) = grads;
		} else if (nlhs >= 5) {
			std::cout << "infFastFood: gathering data..." << std::endl;
			M = floor(M/2/D);
			int out;
			std::frexp(D - 1, &out);
			D = pow(2, out);
			libgp::FastFood * bf = (libgp::FastFood *) &gp.covf();
			plhs[3] = mxCreateDoubleMatrix(M, D, mxREAL);
			plhs[4] = mxCreateDoubleMatrix(M, D, mxREAL);
			plhs[5] = mxCreateDoubleMatrix(M, D, mxREAL);
			plhs[6] = mxCreateDoubleMatrix(M, D, mxREAL);
			//TODO: refactor
			int temp = M;
			M = D;
			D = temp;
			Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[3]), D, M) =
					bf->getS();
			Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[4]), D, M) =
					bf->getG();
			Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[5]), D, M) =
					bf->getPI();
			Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[6]), D, M) =
					bf->getB();
			std::cout << "infFastFood: done." << std::endl;
		}
	}
}

