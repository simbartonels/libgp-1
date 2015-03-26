// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <gp_fic.h>
#include "gp.h"
#include "gp_deg.h"
#include "gp_solin.h"
#include "gp_utils.h"
#include "gp_fic_optimized.h"
#include "gp_multiscale_optimized.h"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

static int input_dim = 2;
static int M = 40;
void genericGradientTest(libgp::AbstractGaussianProcess * gp, size_t input_dim){
	  size_t param_dim = gp->covf().get_param_dim();
	  Eigen::VectorXd params(param_dim);
	  params.setRandom();
	  gp->covf().set_loghyper(params);
	  size_t n = 1000;
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
	    double dist = (j2-j1)/(2*e);
	    dist = std::fabs(dist-grad(i))/(std::fabs(dist) + 1e-50);
	    EXPECT_NEAR(dist, 0., 1e-3) << "parameter number: " << i;
	  }

	  delete gp;
}

TEST(LogLikelihoodTest, CheckGradientsFullGP)
{

  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEiso, CovNoise)");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsMultiScaleGP)
{

  libgp::FICGaussianProcess * gp = new libgp::FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "SparseMultiScaleGP");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsMultiScaleGPOptimized)
{

  libgp::OptMultiScaleGaussianProcess * gp = new libgp::OptMultiScaleGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "SparseMultiScaleGP");
  genericGradientTest(gp, input_dim);
}


TEST(LogLikelihoodTest, CheckGradientsFICGP)
{

  libgp::FICGaussianProcess * gp = new libgp::FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "FIC");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsFICGPOptimized)
{

  libgp::OptFICGaussianProcess * gp = new libgp::OptFICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "FIC");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsFICfixedGP)
{

  libgp::FICGaussianProcess * gp = new libgp::FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "FICfixed");
  genericGradientTest(gp, input_dim);
}

#ifdef BUILD_FAST_FOOD
TEST(LogLikelihoodTest, CheckGradientsDegGPFastFood)
{

  libgp::DegGaussianProcess * gp = new libgp::DegGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "FastFood");
  genericGradientTest(gp, input_dim);
}
#endif

TEST(LogLikelihoodTest, CheckGradientsDegGP)
{

  libgp::DegGaussianProcess * gp = new libgp::DegGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M, "Solin");
  genericGradientTest(gp, input_dim);
}

TEST(LogLikelihoodTest, CheckGradientsSolinGP)
{

  libgp::SolinGaussianProcess * gp = new libgp::SolinGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M);
  genericGradientTest(gp, input_dim);
}
