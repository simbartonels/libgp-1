//TODO: copy&paste code! write a base class!
#include "gp_deg.h"

#include "mex.h"
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>

//extern int dpotrs_(char *, long *, long *, double *, long *, double *, long *, long *);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	size_t M;
	size_t D;
	size_t n;
	size_t p;
	if (nrhs != 4 || nlhs < 2) /* check the input */
		mexErrMsgTxt("Usage: [alpha, L, nlZ, dnlZ] = infSolinmex(M, unwrap(hyp), x, y)");
	M = (size_t) mxGetScalar(prhs[0]);
	n = mxGetM(prhs[2]);
	D = mxGetN(prhs[2]);
	//std::cout << "input dimensionality: " << D << std::endl;
	//std::cout << "number of basis function: " << M << std::endl;
	//std::cout << "number of points: " << n << std::endl;
	p = mxGetM(prhs[1]);
	libgp::DegGaussianProcess gp(D, "CovSum ( CovSEard, CovNoise)", M,
			"Solin");
	Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[1]),
			p);
//	std::cout << "bf_multi_scale: params" << std::endl << params << std::endl;

	gp.covf().set_loghyper(params);

	Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]), n);
	Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[2]), n,
			D);

	for (size_t i=0; i < n; i++) {
		gp.add_pattern(X.row(i), y(i));
	}

	plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL); /* allocate space for output */
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), M) = gp.getAlpha();
	plhs[1] = mxCreateDoubleMatrix(M, M, mxREAL);
	Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), M, M) = gp.getL();
	if(nlhs >= 3){
		double nlZ = gp.log_likelihood();
		plhs[2] = mxCreateDoubleScalar(nlZ);
		if(nlhs >= 4){
			Eigen::VectorXd grads = gp.log_likelihood_gradient();
			size_t params = grads.size();
			plhs[3] = mxCreateDoubleMatrix(params, 1, mxREAL); /* allocate space for output */
			Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[3]), params) = grads;
		}
	}
}

