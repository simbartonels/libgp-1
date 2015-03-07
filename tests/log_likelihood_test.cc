// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <gp_fic.h>
#include "gp.h"
#include "gp_deg.h"
#include "gp_solin.h"
#include "gp_utils.h"
#ifdef BUILD_BENCHMARK
#include "gp_fic_naive.h"
#endif
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>


void genericGradientTest(libgp::AbstractGaussianProcess * gp, size_t input_dim){
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
	  for(size_t i = 0; i < n; ++i) {
	    gp->add_pattern(X.row(i), y(i));
	  }

	  double e = 1e-4;

	  Eigen::VectorXd grad = gp->log_likelihood_gradient();

	  for (int i=0; i<param_dim; i++) {
	    double theta = params(i);
	    params(i) = theta - e;
	    gp->covf().set_loghyper(params);
	    double j1 = gp->log_likelihood();
	    params(i) = theta + e;
	    gp->covf().set_loghyper(params);
	    double j2 = gp->log_likelihood();
	    params(i) = theta;
	    ASSERT_NEAR((j2-j1)/(2*e), grad(i), 1e-5) << "parameter number: " << i;
	  }

	  delete gp;
}

TEST(LogLikelihoodTest, CheckGradientsFullGP)
{
  int input_dim = 3;
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEiso, CovNoise)");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsFICGP)
{
  int input_dim = 3;
  libgp::FICGaussianProcess * gp = new libgp::FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", 20, "SparseMultiScaleGP");
  genericGradientTest(gp, input_dim);
}

#ifdef BUILD_FAST_FOOD
TEST(LogLikelihoodTest, CheckGradientsDegGPFastFood)
{
  int input_dim = 3;
  libgp::DegGaussianProcess * gp = new libgp::DegGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", 20, "FastFood");
  genericGradientTest(gp, input_dim);
}
#endif

TEST(LogLikelihoodTest, CheckGradientsDegGP)
{
  int input_dim = 3;
  libgp::DegGaussianProcess * gp = new libgp::DegGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", 20, "Solin");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsSolinGP)
{
  int input_dim = 3;
  libgp::SolinGaussianProcess * gp = new libgp::SolinGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", 20);
  genericGradientTest(gp, input_dim);
}
