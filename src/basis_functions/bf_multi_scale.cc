// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_multi_scale.h"

#include "cov_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"

#include <cmath>

namespace libgp {

bool MultiScale::real_init() {
	//TODO: signal that Multiscale ignores the covariance function!
	CovFactory f;
	CovarianceFunction * expectedCov;
	expectedCov = f.create(input_dim, "CovSum ( CovSEiso, CovNoise)");
	if (cov->to_string() != expectedCov->to_string()) {
		//TODO: signal reason for error!
		return false;
	}

	loghyper.resize(get_param_dim());
	LUpsi.resize(M, M);
	iUpsi.resize(M, M);
	U.resize(M, input_dim);
	Uell.resize(M, input_dim);
	ell.resize(input_dim);
	return true;
}

Eigen::VectorXd MultiScale::computeBasisFunctionVector(
		const Eigen::VectorXd & x) {
	//FIXME: something's wrong here. U or Uell?
	Eigen::VectorXd uvx(M);
	for(size_t i = 0; i < M; i++){
		uvx(i) = g(x, U.row(i), Uell.row(i));
	}
	return uvx;
}

Eigen::MatrixXd MultiScale::getCholeskyOfInverseWeightPrior() {
	return LUpsi;
}

Eigen::MatrixXd MultiScale::getWeightPrior() {
	return iUpsi;
}

double MultiScale::getWrappedKernelValue(const Eigen::VectorXd &x1,
		const Eigen::VectorXd &x2) {
	return c * g(x1, x2, ell);
}

void MultiScale::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
	//TODO: implement
}

void MultiScale::set_loghyper(const Eigen::VectorXd& p) {
	CovarianceFunction::set_loghyper(p);

	for (size_t i = 0; i < input_dim; i++)
		ell(i) = exp(loghyper(i));
	size_t idx = input_dim;
	for (size_t d = 0; d < input_dim; d++) {
		for (size_t m = 0; m < M; m++) {
			/*
			 * For robustness half of the length scales is added to the
			 * inducing length scales.
			 */
			Uell(m, d) = exp(loghyper(idx)) + ell(d) / 2;
			U(m, d) = loghyper(idx + M * input_dim);
			idx++;
		}
	}

//	std::cout << "bf_multi_scale: U" << std::endl << U << std::endl;
//	std::cout << "bf_multi_scale: Uell" << std::endl << Uell << std::endl;

	c = exp(loghyper(2 * M * input_dim + input_dim));

	sn2 = exp(2 * loghyper(2 * M * input_dim + input_dim + 1));
	initializeMatrices();
}

void MultiScale::initializeMatrices(){
	/**
	 * Initializes the matrices LUpsi and iUpsi.
	 */
	for(size_t i = 0; i < M; i++){
		Eigen::VectorXd vi = U.row(i);
		Eigen::VectorXd s = Uell.row(i) - ell.transpose();


		for(size_t j = 0; j <= i; j++){
			LUpsi(i, j) = g(vi, U.row(j), s.transpose()+Uell.row(j));
		}
	}
	LUpsi = LUpsi/c;

	LUpsi.topLeftCorner(M, M) = LUpsi.topLeftCorner(M, M).selfadjointView<Eigen::Lower>().llt().matrixL();

	iUpsi = LUpsi.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(LUpsi.Identity(M, M));
	LUpsi.topLeftCorner(M, M).triangularView<Eigen::Lower>().adjoint().solveInPlace(iUpsi);
}

void MultiScale::set_loghyper(const double p[]) {
	CovarianceFunction::set_loghyper(p);
}

size_t MultiScale::get_param_dim() {
	//1 for length scale
	//1 for noise
	//input_dim length scales
	//M*input_dim inducing inputs
	//M*input_dim corresponding length scales
	return 2 * M * input_dim + input_dim + 1 + 1;
}

Eigen::VectorXd MultiScale::get_loghyper() {
	return loghyper;
}

std::string MultiScale::to_string() {
	return "MultiScale";
}

double MultiScale::g(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const Eigen::VectorXd& sigma) {
	//TODO: can we make this numerically more stable?
	Eigen::VectorXd delta = x1 - x2;
	double z = delta.cwiseQuotient(sigma).transpose() * delta;
	z = exp(-0.5 * z);
	double p = 1;
	for(size_t i = 0; i < input_dim; i++){
		p = 2 * M_PI * p * sigma(i);
	}
	p = sqrt(p);
	return z/p;
}

}
