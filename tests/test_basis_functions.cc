// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.
#include "cov_factory.h"
#include "cov.h"
#include "basis_functions/basisf_factory.h"
#include "basis_functions/IBasisFunction.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

class BFGradientTest: public TestWithParam<std::string> {
protected:
	virtual void SetUp() {
		n = 3;
		M = 20;
		e = 1e-8;
		wrappedCovarianceFunction = covFactory.create(n,
				"CovSum ( CovSEiso, CovNoise)");
		covf = factory.createBasisFunction(GetParam(), M,
				wrappedCovarianceFunction);
		param_dim = covf->get_param_dim();
		params = Eigen::VectorXd::Random(param_dim);
		x1 = Eigen::VectorXd::Random(n);
		x2 = Eigen::VectorXd::Random(n);
		covf->set_loghyper(params);
	}
	virtual void TearDown() {
		delete covf;
		delete wrappedCovarianceFunction;
	}
	int n, param_dim;
	size_t M;
	libgp::CovFactory covFactory;
	libgp::BasisFFactory factory;
	libgp::CovarianceFunction * wrappedCovarianceFunction;
	libgp::IBasisFunction * covf;
	double e;
	Eigen::VectorXd params;
	Eigen::VectorXd x1;
	Eigen::VectorXd x2;
	Eigen::VectorXd gradient() {
		Eigen::VectorXd grad(param_dim);
		covf->grad(x1, x2, grad);
		return grad;
	}
	double numerical_gradient(int i) {
		double theta = params(i);
		params(i) = theta - e;
		covf->set_loghyper(params);
		double j1 = covf->getWrappedKernelValue(x1, x2);
		params(i) = theta + e;
		covf->set_loghyper(params);
		double j2 = covf->getWrappedKernelValue(x1, x2);
		params(i) = theta;
		return (j2 - j1) / (2 * e);
	}

    Eigen::MatrixXd numerical_weight_prior_gradient(size_t i){
        double theta = params(i);
        params(i) = theta - e;
        covf->set_loghyper(params);
        Eigen::MatrixXd j1 = covf->getInverseWeightPrior();
        params(i) = theta + e;
        covf->set_loghyper(params);
        Eigen::MatrixXd j2 = covf->getInverseWeightPrior();
        params(i) = theta;
        return ((j2.array()-j1.array())/(2*e)).matrix();
    }

	Eigen::VectorXd numerical_basis_function_gradient(size_t i) {
		double theta = params(i);
		params(i) = theta - e;
		covf->set_loghyper(params);
		Eigen::VectorXd j1 = covf->computeBasisFunctionVector(x1);
		params(i) = theta + e;
		covf->set_loghyper(params);
		Eigen::MatrixXd j2 = covf->computeBasisFunctionVector(x1);
		params(i) = theta;
		return ((j2.array() - j1.array()) / (2 * e)).matrix();
	}
};

TEST_P(BFGradientTest, EqualToNumerical) {
	Eigen::VectorXd grad = gradient();
	for (int i = 0; i < param_dim; ++i) {
		double num_grad = numerical_gradient(i);
		if (grad(i) == 0.0) {
			ASSERT_NEAR(num_grad, 0.0, 1e-2)<< "Parameter number: " << i
			<< std::endl << "numerical gradient: " << num_grad;
		}
		else {
			ASSERT_NEAR((num_grad-grad(i))/grad(i), 0.0, 1e-2) << "Parameter number: " << i
			<< std::endl << "numerical gradient: " << num_grad;
		}
	}
}

TEST_P(BFGradientTest, InverseWeightPriorEqualToNumerical) {
	size_t M = covf->getNumberOfBasisFunctions();
  Eigen::MatrixXd grad(M, M);
  for (int i=0; i<param_dim; ++i) {
	covf->gradInverseWeightPrior(i, grad);
	Eigen::MatrixXd numeric_gradient = numerical_weight_prior_gradient(i);
	for(size_t j=0; j < M; j++){
		for(size_t k = 0; k < M; k++){
			if (grad(j, k) == 0.0) ASSERT_NEAR(numeric_gradient(j, k), 0.0, 1e-2);
			else ASSERT_NEAR((numeric_gradient(j, k)-grad(j, k))/grad(j, k), 0.0, 1e-2);
		}
	}
  }
}

TEST_P(BFGradientTest, BasisFunctionEqualToNumerical) {
	size_t M = covf->getNumberOfBasisFunctions();
	Eigen::VectorXd phi = covf->computeBasisFunctionVector(x1);
	Eigen::VectorXd grad(M);
	for (int i = 0; i < param_dim; i++) {
		covf->gradBasisFunction(x1, phi, i, grad);
		Eigen::VectorXd numeric_gradient = numerical_basis_function_gradient(i);
		for (size_t j = 0; j < M; j++) {
			if (grad(j) == 0.0) {
				ASSERT_NEAR(numeric_gradient(j), 0.0, 1e-2)<< "parameter: " << i<< std::endl
						<< "numerical gradient: " << numeric_gradient(j);
			}
			else {
				ASSERT_NEAR((numeric_gradient(j)-grad(j))/grad(j), 0.0, 1e-2)<< "parameter: "
						<< i<< std::endl << "numerical gradient: " << numeric_gradient(j);
			}
		}
	}
}

INSTANTIATE_TEST_CASE_P(BasisFunction, BFGradientTest,
		Values("SparseMultiScaleGP", "SparseMultiScaleGP"));

#else

// Google Test may not support value-parameterized tests with some
// compilers. If we use conditional compilation to compile out all
// code referring to the gtest_main library, MSVC linker will not link
// that library at all and consequently complain about missing entry
// point defined in that library (fatal error LNK1561: entry point
// must be defined). This dummy test keeps gtest_main linked in.
TEST(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
