// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <gp_fic.h>
#include "gp.h"
#include "gp_deg.h"
#include "gp_solin.h"
#include "gp_utils.h"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

/**
 * Tests if mean and variance derivatives are computed correctly.
 */
void genericPredictionGradientTest(libgp::AbstractGaussianProcess * gp,
		size_t input_dim) {
	size_t param_dim = gp->covf().get_param_dim();
	Eigen::VectorXd params(param_dim);
	params.setRandom();
	gp->covf().set_loghyper(params);
	size_t n = 500;
	Eigen::MatrixXd X(n, input_dim);
	X.setRandom();
//	  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
	Eigen::VectorXd y(n);
	y.setRandom();
	for (size_t i = 0; i < n; ++i) {
		double x[input_dim];
		for (int j = 0; j < input_dim; ++j)
			x[j] = X(i, j);
		gp->add_pattern(x, y(i));
	}

	double e = 1e-4;

	Eigen::VectorXd x(input_dim);
	x.setRandom();
	Eigen::VectorXd grad_mean(input_dim);
	Eigen::VectorXd grad_var(input_dim);
	gp->grad_f(x, grad_mean);
	gp->grad_var(x, grad_var);

	for (int i = 0; i < input_dim; i++) {
		double theta = x(i);
		x(i) = theta - e;
		double j1 = gp->f(x);
		double k1 = gp->var(x);
		x(i) = theta + e;
		double j2 = gp->f(x);
		double k2 = gp->var(x);
		x(i) = theta;
		ASSERT_NEAR((j2-j1)/(2*e), grad_mean(i), 1e-5)<< "parameter number: " << i;
		ASSERT_NEAR((k2-k1)/(2*e), grad_var(i), 1e-5)<< "parameter number: " << i;
	}

	delete gp;
}

//TEST(DerivativeTest, CheckGradientsFullGP) {
//	int input_dim = 3;
//	libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim,
//			"CovSum ( CovSEiso, CovNoise)");
//	genericPredictionGradientTest(gp, input_dim);
//}

TEST(DerivativeTest, CheckGradientsFICGP) {
	int input_dim = 3;
	libgp::FICGaussianProcess * gp = new libgp::FICGaussianProcess(input_dim,
			"CovSum ( CovSEard, CovNoise)", 20, "SparseMultiScaleGP");
	genericPredictionGradientTest(gp, input_dim);
}

#ifdef BUILD_FAST_FOOD
TEST(DerivativeTest, CheckGradientsDegGPFastFood) {
	int input_dim = 3;
	libgp::DegGaussianProcess * gp = new libgp::DegGaussianProcess(input_dim,
			"CovSum ( CovSEard, CovNoise)", 20, "FastFood");
	genericPredictionGradientTest(gp, input_dim);
}
#endif
//
//TEST(DerivativeTest, CheckGradientsDegGP) {
//	int input_dim = 3;
//	libgp::DegGaussianProcess * gp = new libgp::DegGaussianProcess(input_dim,
//			"CovSum ( CovSEard, CovNoise)", 20, "Solin");
//	genericPredictionGradientTest(gp, input_dim);
//}
//
//TEST(DerivativeTest, CheckGradientsSolinGP) {
//	int input_dim = 3;
//	libgp::SolinGaussianProcess * gp = new libgp::SolinGaussianProcess(
//			input_dim, "CovSum ( CovSEard, CovNoise)", 20, "Solin");
//	genericPredictionGradientTest(gp, input_dim);
//}
