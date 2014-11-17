// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_deg.h"

#include "cov_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"

#include <cmath>

namespace libgp {

libgp::DegGaussianProcess::DegGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) {
	BasisFFactory factory;
	//wrap initialized covariance function with basis function
	cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	cf->loghyper_changed = 0;
	bf = (IBasisFunction *) cf;
	M = bf->getNumberOfBasisFunctions();
	alpha.resize(M);
	L.resize(M, M);
}

libgp::DegGaussianProcess::~DegGaussianProcess() {
}

double libgp::DegGaussianProcess::var_impl(const Eigen::VectorXd x_star) {
}

double libgp::DegGaussianProcess::log_likelihood_impl() {
}

Eigen::VectorXd libgp::DegGaussianProcess::log_likelihood_gradient_impl() {
	Eigen::MatrixXd M;
	if(M.isd)
}

void libgp::DegGaussianProcess::update_k_star(const Eigen::VectorXd& x_star) {

}

void libgp::DegGaussianProcess::update_alpha() {
}

void libgp::DegGaussianProcess::computeCholesky() {
}

void libgp::DegGaussianProcess::updateCholesky(const double x[], double y) {
}

}
