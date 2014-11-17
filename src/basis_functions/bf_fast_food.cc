// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_fast_food.h"
#include "spiral_wht.h"

#include "cov_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"

#include <cmath>

namespace libgp {
Eigen::VectorXd libgp::FastFood::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {
	Eigen::VectorXd retval(1);
	return retval;
}

Eigen::MatrixXd libgp::FastFood::getInverseWeightPrior() {
	Eigen::MatrixXd retval(1, 1);
	return retval;
}

Eigen::MatrixXd libgp::FastFood::getCholeskyOfInverseWeightPrior() {
	Eigen::MatrixXd retval(1, 1);
	return retval;
}

Eigen::MatrixXd libgp::FastFood::getWeightPrior() {
	Eigen::MatrixXd retval(1, 1);
	return retval;
}

void libgp::FastFood::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
}

void libgp::FastFood::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
}

void libgp::FastFood::gradBasisFunction(const Eigen::VectorXd& x,
		const Eigen::VectorXd& phi, size_t p, Eigen::VectorXd& grad) {
}

void libgp::FastFood::gradInverseWeightPrior(size_t p,
		Eigen::MatrixXd & diSigmadp) {
}

void libgp::FastFood::set_loghyper(const Eigen::VectorXd& p) {
}

void libgp::FastFood::set_loghyper(const double p[]) {
}

std::string libgp::FastFood::to_string() {
	return "FastFood";
}

bool libgp::FastFood::real_init() {
	return false;
}
}
