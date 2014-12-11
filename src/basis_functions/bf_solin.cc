// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "bf_solin.h"

namespace libgp{

libgp::Solin::~Solin() {
}

Eigen::VectorXd libgp::Solin::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {

}

Eigen::MatrixXd libgp::Solin::getInverseOfSigma() {
}

Eigen::MatrixXd libgp::Solin::getCholeskyOfInvertedSigma() {
}

Eigen::MatrixXd libgp::Solin::getSigma() {
}

double libgp::Solin::getLogDeterminantOfSigma() {
}

void libgp::Solin::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
}

void libgp::Solin::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
}

void libgp::Solin::gradBasisFunction(const Eigen::VectorXd& x,
		const Eigen::VectorXd& phi, size_t p, Eigen::VectorXd& grad) {
}

bool libgp::Solin::gradBasisFunctionIsNull(size_t p) {
}

void libgp::Solin::gradiSigma(size_t p, Eigen::MatrixXd& diSigmadp) {
}

bool libgp::Solin::gradiSigmaIsNull(size_t p) {
}

std::string libgp::Solin::to_string() {
	return "Solin";
}

void libgp::Solin::log_hyper_updated(const Eigen::VectorXd& p) {

}

bool libgp::Solin::real_init() {

	return false;
}

}
