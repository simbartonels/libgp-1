// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "sampleset.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

class GradientTest : public TestWithParam<std::string> {
  protected:
    virtual void SetUp() {
      n = 3;
      e = 1e-8;
      covf = factory.create(n, GetParam());
      param_dim = covf->get_param_dim();
      params = Eigen::VectorXd::Random(param_dim);
      x1 = Eigen::VectorXd::Random(n);
      x2 = Eigen::VectorXd::Random(n);
      covf->set_loghyper(params);

    }
    virtual void TearDown() {
      delete covf;
    }
    int n, param_dim;
    libgp::CovFactory factory;
    libgp::CovarianceFunction * covf;
    double e;
    Eigen::VectorXd params;
    Eigen::VectorXd x1;
    Eigen::VectorXd x2;

    Eigen::VectorXd gradient() {
      Eigen::VectorXd grad(param_dim);
      covf->grad(x1, x2, grad);
      return grad;
    }

   double gradient_input_d(size_t i){
	return covf->grad_input_d(x1(i), x2(i), covf->get(x1, x2), i);
   }

	Eigen::VectorXd gradient_input() {
		Eigen::VectorXd grad(x1.size());
		covf->grad_input(x1, x2, grad);
		return grad;
	}

	Eigen::VectorXd gradient_input_diag() {
		Eigen::VectorXd grad(x1.size());
		covf->grad_input(x1, x1, grad);
		return grad;
	}

    double numerical_gradient(int i) {
      double theta = params(i);
      params(i) = theta - e;
      covf->set_loghyper(params);
      double j1 = covf->get(x1, x2);
      params(i) = theta + e;
      covf->set_loghyper(params);
      double j2 = covf->get(x1, x2);
      params(i) = theta;
      return (j2-j1)/(2*e);
    }

	double numerical_gradient_input(size_t i){
		double theta = x1(i);
		x1(i) = theta - e;
		double j1 = covf->get(x1, x2);
		x1(i) = theta + e;
		double j2 = covf->get(x1, x2);
		x1(i) = theta;
		return (j2 - j1)/2/e;
	}

	double numerical_gradient_input_diag(size_t i){
		double theta = x1(i);
		x1(i) = theta - e;
		double j1 = covf->get(x1, x1);
		x1(i) = theta + e;
		double j2 = covf->get(x1, x1);
		x1(i) = theta;
		return (j2 - j1)/2/e;
	}
};

TEST_P(GradientTest, EqualToNumerical) {
  Eigen::VectorXd grad = gradient();
  for (int i=0; i<param_dim; ++i) {
    double num_grad = numerical_gradient(i);
    if (num_grad == 0.0) EXPECT_NEAR(grad(i), 0.0, 1e-2);
    else EXPECT_NEAR((num_grad-grad(i))/num_grad, 0.0, 1e-2);
  }
}

TEST_P(GradientTest, EqualToNumerical_input) {
	Eigen::VectorXd grad = gradient_input();
	if(grad.isZero(1e-50))
		return; //not implemented
	for (int i = 0; i < grad.size(); ++i) {
		double num_grad = numerical_gradient_input(i);
                double grad_d = gradient_input_d(i);
                EXPECT_NEAR(grad_d, grad(i), 1e-15);
		if (grad(i) == 0.0) {
			EXPECT_NEAR(num_grad, 0.0, 1e-2)<< "Parameter number: " << i
			<< std::endl << "numerical gradient: " << num_grad;
		}
		else {
			EXPECT_NEAR((num_grad-grad(i))/grad(i), 0.0, 1e-2) << "Parameter number: " << i
			<< std::endl << "numerical gradient: " << num_grad
			<< std::endl << "computed gradient: " << grad(i);
		}
	}
}

//TEST_P(GradientTest, DiagEqualToNumerical_input) {
//	Eigen::VectorXd grad = gradient_input_diag();
//	for (int i = 0; i < grad.size(); ++i) {
//		double num_grad = numerical_gradient_input_diag(i);
//
//		if (grad(i) == 0.0) {
//			ASSERT_NEAR(num_grad, 0.0, 1e-2)<< "Parameter number: " << i
//			<< std::endl << "numerical gradient: " << num_grad;
//		}
//		else {
//			ASSERT_NEAR((num_grad-grad(i))/grad(i), 0.0, 1e-2) << "Parameter number: " << i
//			<< std::endl << "numerical gradient: " << num_grad;
//		}
//	}
//}

INSTANTIATE_TEST_CASE_P(CovarianceFunction, GradientTest, Values(
          "CovLinearard",
          "CovLinearone",
          "CovMatern3iso",
          "CovMatern5iso",
          "CovNoise",
          "CovProd(CovSEiso, CovMatern3iso)",
          "CovRQiso",
          "CovSEard",
          "CovSEiso",
          "CovSum(CovSEiso, CovNoise)",
          "CovSum(CovLinearard, CovNoise)",
          "InputDimFilter(1/CovSEiso)",
          "InputDimFilter(0/CovSum(CovSEiso, CovNoise))"
          ));

TEST(FilterTest, EqualToVector) {



}


#else

// Google Test may not support value-parameterized tests with some
// compilers. If we use conditional compilation to compile out all
// code referring to the gtest_main library, MSVC linker will not link
// that library at all and consequently complain about missing entry
// point defined in that library (fatal error LNK1561: entry point
// must be defined). This dummy test keeps gtest_main linked in.
TEST(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
