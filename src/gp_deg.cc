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

const double log2pi = log(2 * M_PI);

libgp::DegGaussianProcess::DegGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) {
	BasisFFactory factory;
	//wrap initialized covariance function with basis function
	cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	cf->loghyper_changed = 0;
	bf = (IBasisFunction *) cf;
	log_noise = bf->getLogNoise();
	squared_noise = exp(2*log_noise);
	M = bf->getNumberOfBasisFunctions();
	alpha.resize(M);
	Phiy.resize(M);
	L.resize(M, M);
	k_star.resize(M);
	temp.resize(M);
}

libgp::DegGaussianProcess::~DegGaussianProcess() {
}

double libgp::DegGaussianProcess::var_impl(const Eigen::VectorXd x_star) {
	temp = L.triangularView<Eigen::Lower>().solve(k_star);
	return squared_noise * (temp.transpose()*temp);
}

double libgp::DegGaussianProcess::log_likelihood_impl() {
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	//TODO: this needs to be evaluated only once!
	double yy = y.transpose() * y;
	double PhiyAlpha = Phiy.transpose()*alpha;
	double halfLogDetA = 0;
	double halfLogDetSigma = bf->getLogDeterminantOfWeightPrior();
	for(size_t j = 0; j < M; j++){
		halfLogDetA+=log(L(j, j));
	}
	return (yy + PhiyAlpha)/squared_noise/2
			+halfLogDetA+halfLogDetSigma
			+(y.size()-M)*log_noise
			+y.size()*log2pi;
}

Eigen::VectorXd libgp::DegGaussianProcess::log_likelihood_gradient_impl() {
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	for(size_t i = 0; i < num_params - 1; i++){

	}

}

void libgp::DegGaussianProcess::update_k_star(const Eigen::VectorXd& x_star) {
	k_star = bf->computeBasisFunctionVector(x_star);
}

void libgp::DegGaussianProcess::update_alpha() {
	//TODO: this step can be simplified for Solin!
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	Phiy = Phi*y;
	alpha = L.triangularView<Eigen::Lower>().solve(Phiy);
	//TODO: correct?!
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);
}

void libgp::DegGaussianProcess::computeCholesky() {
	//TODO: this step can be simplified for Solin!
	size_t n = sampleset->size();
	if (n > Phi.rows())
		Phi.resize(M, n);
	for (size_t i = 0; i < n; i++)
		Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
	L = (Phi * Phi.transpose() + squared_noise * bf->getInverseWeightPrior()).selfAdjointView<Eigen::Lower>().llt().matrixL();
}

void libgp::DegGaussianProcess::updateCholesky(const double x[], double y) {
	//Do nothing and just recompute everything.
	//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

}
