#include <cmath>
#include <string.h>
#include <Eigen/Dense>
#include <iostream>
#include <mex.h>
#include "cov.h"
#include "cov_factory.h"
#include "gp_utils.h"
#include "gp_deg.h"
#include "gp_multiscale_optimized.h"

namespace libgp {

std::stringstream ss;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 5 || nlhs != 2) /* check the input */
		mexErrMsgTxt("Usage: [e1, e2] = toy_exp2(seeds, n, D, M1, M2)");
	size_t n = (size_t) mxGetScalar(prhs[1]);
	size_t D = (size_t) mxGetScalar(prhs[2]);;
	size_t s = mxGetN(prhs[0]);;
	size_t M1 = (size_t) mxGetScalar(prhs[3]);;
	size_t M = (size_t) mxGetScalar(prhs[4]);;
	Eigen::VectorXd e1(s);
	Eigen::VectorXd e2(s);
	Eigen::Map<Eigen::VectorXd> seeds(mxGetPr(prhs[0]), s);
	CovFactory cf;
	CovarianceFunction * ardse = cf.create(D, "CovSum ( CovSEard, CovNoise)");
	ardse->init(D);
	Eigen::VectorXd p(ardse->get_param_dim());
	p.fill(0);
	p.tail(0).fill(-log(0.0)); //0 noise
	ardse->set_loghyper(p);
	srand(seed(0));
	srand48(seed(0));
	Eigen::MatrixXd X(n, D);
	Eigen::MatrixXd Xstar(n, D);
	for (size_t j = 0; j < n; j++) {
		for (size_t d = 0; d < D; d++) {
			X(j, d) = Utils::randn();
			Xstar(j, d) = Utils::randn();
		}
	}
	y = ardse->draw_random_sample(X);
	for (size_t j = 0; j < s; j++) {
		srand(seed(j));
		srand48(seed(j));
		DegGaussianProcess fastfood = DegGaussianProcess(D,
				"CovSum ( CovSEard, CovNoise)", M1, "FastFood");
		fastfood.covf().set_loghyper(p);
		srand(seed(j));
		srand48(seed(j));
		OptFICGaussianProcess multiscale = OptFICGaussianProcess(D,
				"CovSum ( CovSEard, CovNoise)", M, "SparseMultiScaleGP");
		Eigen::VectorXd ms_params(2 * M * D + D + 2);
		ms_params.head(D) = p.head(D) * n / M;
		ms_params.segment(D, M * D).fill(((double) M) / n);
		for (size_t d = 0; d < D; d++) {
			ms_params(M * D + D + d) = Utils::randn();
		}
		ms_params.tail(2) = p.tail(2);
		multiscale.covf().set_loghyper(p);
		for(size_t k = 0; k < n; k++){
			double pred = (fastfood.f(X.row(k)) - y(k));
			e1(j) += pred * pred / n;
			pred = (multiscale.f(X.row(k)) - y(k));
			e2(j) += pred * pred / n;
		}
	}
	plhs[0] = mxCreateDoubleMatrix(s, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), s) = e1;
	plhs[1] = mxCreateDoubleMatrix(s, 1, mxREAL);
	Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[1]), s) = e2;
}

}
