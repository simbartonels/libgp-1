// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.
#include "cov_factory.h"
#include "cov.h"
#include "basis_functions/basisf_factory.h"
#include "basis_functions/IBasisFunction.h"
#include "sampleset.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

class BFGradientTest: public TestWithParam<std::string> {
protected:
	virtual void SetUp() {
		n = 3;
		M = 12;
		e = 1e-8;
		wrappedCovarianceFunction = covFactory.create(n,
				"CovSum ( CovSEard, CovNoise)");
		covf = factory.createBasisFunction(GetParam(), M,
				wrappedCovarianceFunction);
		param_dim = covf->get_param_dim();
		params = Eigen::VectorXd::Random(param_dim);
		x1 = Eigen::VectorXd::Random(n);
		x2 = Eigen::VectorXd::Random(n);
		covf->set_loghyper(params);
		sampleset = new libgp::SampleSet(n);
		sampleset->add(x1, 0.);
		grad.resize(1);
		k.resize(1);
		k(0) = covf->getWrappedKernelValue(x1, x1);
		phix1 = Eigen::MatrixXd(M, sampleset->size());
		for (size_t i = 0; i < sampleset->size(); i++)
			phix1.col(i) = covf->computeBasisFunctionVector(sampleset->x(i));
	}
	virtual void TearDown() {
		delete covf;
		delete wrappedCovarianceFunction;
		delete sampleset;
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
	Eigen::VectorXd grad;
	Eigen::VectorXd k;
	Eigen::MatrixXd phix1;
	libgp::SampleSet * sampleset;

	Eigen::VectorXd gradient() {
		Eigen::VectorXd grad(param_dim);
		covf->grad(x1, x2, grad);
		return grad;
	}

	Eigen::VectorXd gradient_diag() {
		Eigen::VectorXd grad(param_dim);
		covf->grad(x1, x1, grad);
		return grad;
	}

	double gradient_diag(size_t p) {
		covf->gradDiagWrapped(sampleset, k, p, grad);
		//grad is a vector of size n=1
		return grad(0);
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

	double numerical_gradient_input(size_t i) {
		double theta = x1(i);
		x1(i) = theta - e;
		double j1 = covf->get(x1, x2);
		x1(i) = theta + e;
		double j2 = covf->get(x1, x2);
		x1(i) = theta;
		return (j2 - j1) / 2 / e;
	}

	double numerical_gradient_input_diag(size_t i) {
		double theta = x1(i);
		x1(i) = theta - e;
		double j1 = covf->get(x1, x1);
		x1(i) = theta + e;
		double j2 = covf->get(x1, x1);
		x1(i) = theta;
		return (j2 - j1) / 2 / e;
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

	double numerical_gradient_diag(int i) {
		double theta = params(i);
		params(i) = theta - e;
		covf->set_loghyper(params);
		double j1 = covf->getWrappedKernelValue(x1, x1);
		params(i) = theta + e;
		covf->set_loghyper(params);
		double j2 = covf->getWrappedKernelValue(x1, x1);
		params(i) = theta;
		return (j2 - j1) / (2 * e);
	}

	Eigen::MatrixXd numerical_gradient_of_isigma(size_t i) {
		double theta = params(i);
		params(i) = theta - e;
		covf->set_loghyper(params);
		Eigen::MatrixXd j1 = covf->getInverseOfSigma();
		params(i) = theta + e;
		covf->set_loghyper(params);
		Eigen::MatrixXd j2 = covf->getInverseOfSigma();
		params(i) = theta;
		return ((j2.array() - j1.array()) / (2 * e)).matrix();
	}

	Eigen::VectorXd numerical_basis_function_gradient(size_t i) {
		double theta = params(i);
		params(i) = theta - e;
		covf->set_loghyper(params);
		Eigen::VectorXd j1 = covf->computeBasisFunctionVector(x1);
		params(i) = theta + e;
		covf->set_loghyper(params);
		Eigen::VectorXd j2 = covf->computeBasisFunctionVector(x1);
//		std::cout << "j1 - j2:" << ((j1-j2)/2/e).transpose().array() << std::endl;
		params(i) = theta;
		return (j2 - j1) / 2 / e;
	}

	Eigen::VectorXd numerical_gradient_k(size_t i) {
		double theta = x1(i);
		Eigen::VectorXd j1(x1.size());
		Eigen::VectorXd j2(x1.size());
		x1(i) = theta - e;
		j1 = covf->computeBasisFunctionVector(x1);
		x1(i) = theta + e;
		j2 = covf->computeBasisFunctionVector(x1);
		x1(i) = theta;
		return (j2 - j1) / (2 * e);
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
			<< std::endl << "numerical gradient: " << num_grad
			<< std::endl << "computed gradient: " << grad(i);
		}
	}
}

//TODO: refactor. copy&paste code!
TEST_P(BFGradientTest, DiagEqualToNumerical) {
	Eigen::VectorXd grad = gradient_diag();
	for (int i = 0; i < param_dim; ++i) {
		double num_grad = numerical_gradient_diag(i);
		double comp_grad2 = gradient_diag(i);

		if (grad(i) == 0.0) {
			ASSERT_NEAR(grad(i), comp_grad2, 1e-7)<< "Parameter number: " << i;
			ASSERT_NEAR(num_grad, 0.0, 1e-2)<< "Parameter number: " << i
			<< std::endl << "numerical gradient: " << num_grad;
		}
		else {
			ASSERT_NEAR((grad(i) - comp_grad2)/grad(i), 0.0, 1e-7) << "Parameter number: " << i;
			ASSERT_NEAR((num_grad-grad(i))/grad(i), 0.0, 1e-2) << "Parameter number: " << i
			<< std::endl << "numerical gradient: " << num_grad;
		}

	}
}

TEST_P(BFGradientTest, dkdx) {
	size_t M = covf->getNumberOfBasisFunctions();
	Eigen::MatrixXd grad(n, M);
	grad.setZero();
	covf->compute_dkdx(x1, phix1, sampleset, grad);
	for (int i = 0; i < n; ++i) {
		Eigen::VectorXd num_grad = numerical_gradient_k(i);
		for (size_t j = 0; j < M; j++) {
			if (num_grad(j) == 0.0)
				ASSERT_NEAR(grad(i, j), 0.0, 1e-2);
			else
				ASSERT_NEAR((num_grad(j)-grad(i, j))/num_grad(j), 0.0, 1e-2)<< "dimension: " << i
				<< std::endl << "numerical gradient: " << num_grad(j)
				<< std::endl << "computed gradient: " << grad(i, j)
				<< std::endl << "basis function: " << j << std::endl;
			}
		}
	}

TEST_P(BFGradientTest, GradientOfiSigmaEqualToNumerical) {
	size_t M = covf->getNumberOfBasisFunctions();
	Eigen::MatrixXd grad(M, M);
	grad.setZero();
	for (int i = 0; i < param_dim; ++i) {
		covf->gradiSigma(i, grad);
		Eigen::MatrixXd numeric_gradient = numerical_gradient_of_isigma(i);
		for (size_t j = 0; j < M; j++) {
			for (size_t k = 0; k < M; k++) {
				if (grad(j, k) == 0.0)
					ASSERT_NEAR(numeric_gradient(j, k), 0.0, 1e-2)<< "Parameter number: " << i << std::endl
					 << "index: " << j << "," << k;
				else
					ASSERT_NEAR((numeric_gradient(j, k)-grad(j, k))/grad(j, k), 0.0, 1e-2)<< "Parameter number: " << i
					<< std::endl << "numerical gradient: " << numeric_gradient(j, k)
					<< std::endl << "computed gradient: " << grad(j, k)
					<< std::endl << "index: " << j << "," << k;
				}
			}
		}
	}

TEST_P(BFGradientTest, BasisFunctionEqualToNumerical) {
	size_t M = covf->getNumberOfBasisFunctions();
	Eigen::VectorXd phi = covf->computeBasisFunctionVector(x1);
	Eigen::VectorXd grad(M);
	grad.setZero();
	for (int i = 0; i < param_dim; i++) {
		covf->gradBasisFunction(x1, phi, i, grad);
		Eigen::VectorXd numeric_gradient = numerical_basis_function_gradient(i);
		for (size_t j = 0; j < M; j++) {
			if (grad(j) == 0.0) {
				ASSERT_NEAR(numeric_gradient(j), 0.0, 1e-2)<< "parameter: " << i<< std::endl
				<< "numerical gradient: " << numeric_gradient(j) << std::endl << "m: " << j;
			}
			else {
				ASSERT_NEAR((numeric_gradient(j)-grad(j))/grad(j), 0.0, 1e-2)<< "parameter: "
				<< i<< std::endl << "numerical gradient: " << numeric_gradient(j) << std::endl << "m: " << j;
			}
		}
	}
}

TEST_P(BFGradientTest, LogDeterminantCorrect) {
	//det = 0.5*log|Sigma|
	double det = covf->getLogDeterminantOfSigma();
	double nan_check = det;
	if(nan_check != det){
		FAIL() << "Log Determinant is NaN!";
	}
	double det2 = -log(covf->getCholeskyOfInvertedSigma().diagonal().prod());

	if (covf->sigmaIsDiagonal()) {
		double det_true = log(covf->getSigma().diagonal().prod()) / 2;
		//make sure the test values agree
		ASSERT_NEAR(det_true, det2, 1e-5);

		ASSERT_NEAR(det, det_true, 1e-5);
	}

	ASSERT_NEAR(det, det2, 1e-5);
}

TEST_P(BFGradientTest, CholeskyCorrect) {
	Eigen::MatrixXd iSigma = covf->getInverseOfSigma();
	if(!iSigma.allFinite())
		FAIL() << "iSigma contains Inf or NaN!";
	Eigen::MatrixXd L = covf->getCholeskyOfInvertedSigma();
	if(!L.allFinite())
		FAIL() << "Cholesky contains Inf or NaN!";
	L.array() = (iSigma - L * L.transpose()).array().abs();
	L.array() = L.array() / (iSigma.array() + 1e-15);
	ASSERT_NEAR(L.maxCoeff(), 0, 1e-15)<< "iSigma: " << std::endl << iSigma << std::endl
	<< "diff: " << std::endl << L << std::endl;
}

TEST_P(BFGradientTest, InverseCorrect) {
	Eigen::MatrixXd Sigma = covf->getSigma();
	if(!Sigma.allFinite())
			FAIL() << "Sigma contains Inf or NaN!";
	Eigen::MatrixXd iSigma = covf->getInverseOfSigma();
	Eigen::MatrixXd shouldBeI = iSigma * Sigma;
	shouldBeI.diagonal().array() -= 1;
	ASSERT_TRUE(shouldBeI.isZero(1e-10));
}

//TEST_P(GradientTest, EqualToNumerical_input) {
//	Eigen::VectorXd grad = gradient_input();
//	for (int i = 0; i < grad.size(); ++i) {
//		double num_grad = numerical_gradient_input(i);
//		if (grad(i) == 0.0) {
//			ASSERT_NEAR(num_grad, 0.0, 1e-2)<< "Parameter number: " << i
//			<< std::endl << "numerical gradient: " << num_grad;
//		}
//		else {
//			ASSERT_NEAR((num_grad-grad(i))/grad(i), 0.0, 1e-2) << "Parameter number: " << i
//			<< std::endl << "numerical gradient: " << num_grad
//			<< std::endl << "computed gradient: " << grad(i);
//		}
//	}
//}

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

#ifdef BUILD_FAST_FOOD
INSTANTIATE_TEST_CASE_P(BasisFunction, BFGradientTest,
		Values("SparseMultiScaleGP", "Solin", "FastFood", "FIC"));
#else
INSTANTIATE_TEST_CASE_P(BasisFunction, BFGradientTest,
		Values("SparseMultiScaleGP", "Solin", "FIC"));
#endif

#else

// Google Test may not support value-parameterized tests with some
// compilers. If we use conditional compilation to compile out all
// code referring to the gtest_main library, MSVC linker will not link
// that library at all and consequently complain about missing entry
// point defined in that library (fatal error LNK1561: entry point
// must be defined). This dummy test keeps gtest_main linked in.
TEST(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
