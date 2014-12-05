// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_deg.h"

#include "basis_functions/basisf_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"

#include <cmath>

namespace libgp {

const double log2pi = log(2 * M_PI);

libgp::DegGaussianProcess::DegGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) :
		AbstractGaussianProcess(input_dim, covf_def) {
	BasisFFactory factory;
	//wrap initialized covariance function with basis function
	cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	cf->loghyper_changed = 0;
	bf = (IBasisFunction *) cf;
	log_noise = bf->getLogNoise();
	squared_noise = exp(2 * log_noise);
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
	return squared_noise * temp.squaredNorm();
}

double libgp::DegGaussianProcess::log_likelihood_impl() {
	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	//TODO: this needs to be evaluated only once!
	double yy = y.squaredNorm();
	assert(yy == y.transpose() * y);
	double PhiyAlpha = Phiy.transpose() * alpha;
	double halfLogDetA = 0;
	double halfLogDetSigma = bf->getLogDeterminantOfWeightPrior();
	for (size_t j = 0; j < M; j++) {
		halfLogDetA += log(L(j, j));
	}
	return (yy + PhiyAlpha) / squared_noise / 2 + halfLogDetA + halfLogDetSigma
			+ (n - M) * log_noise + n * log2pi;
}

Eigen::VectorXd libgp::DegGaussianProcess::log_likelihood_gradient_impl() {
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	size_t n = sampleset->size();
	//TODO: move these allocations outside the function?
	Eigen::MatrixXd dSigma(M, M);
	Eigen::MatrixXd dPhidi(M, n);
	Eigen::VectorXd t(M);
	//TODO: here we have a few steps that aren't necessary for Solin!
	Eigen::VectorXd phi_alpha_plus_y = Phi.transpose() * alpha + y;
	Eigen::VectorXd sigma_alpha = bf->getInverseWeightPrior().transpose()
			* alpha;
	Eigen::MatrixXd iAPhi = L.triangularView<Eigen::Lower>().solve(Phi);
	Eigen::MatrixXd Gamma = L.triangularView<Eigen::Lower>().solve(
			bf->getInverseWeightPrior());
	Gamma = Gamma.transpose() * Gamma;
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(iAPhi);
	for (size_t i = 0; i < num_params - 1; i++) {
		//let's start with dA
		int dPhiInfo = bf->gradBasisFunctionInfo(i);
		if (dPhiInfo != bf->IBF_MATRIX_INFO_NULL) {
			for (size_t j = 0; j < n; j++) {
				//TODO: this has a lot of optimization potential especially for fast food
				//when vectorizing this
				bf->gradBasisFunction(sampleset->x(j), Phi.col(j), i, t);
				//TODO: it is actually quite bad that we need to copy here
				//could it be faster using .data()?
				dPhidi.col(j) = t;
			}
			//the first sum() call is a workaround for Eigen not recognizing the result to be a scalar
			gradient(i) = ((alpha.transpose() * dPhidi * phi_alpha_plus_y).sum()/squared_noise
			//now d|A|
					+ iAPhi.cwiseProduct(dPhidi).sum());
		}
		//now the dSigma parts
		int dSigmaInfo = bf->gradWeightPriorInfo(i);
		if (dSigmaInfo != bf->IBF_MATRIX_INFO_NULL) {
			/**
			 * TODO: If the inverse weight prior is known to be diagonal the trace computations can
			 * be sped up using the relation between trace and eigenvalues.
			 * Maybe use macros instead of functions to keep this function readable.
			 * I.e.:
			 * if(weight_prior is diagonal){
			 * 		for(i=0; i < num_params)
			 * 			dPhiMacro
			 * 			...
			 * 	else {
			 * 		for(i=0; i < num_params)
			 * 			dPhiMacro
			 * 			...
			 * 	}
			 */
			bf->gradWeightPrior(i, dSigma);
			//these are from Sigma and |Sigma| from the derivation of A
			gradient(i) -= squared_noise
					* (sigma_alpha.transpose() * dSigma * sigma_alpha
							+ Gamma.cwiseProduct(dSigma).sum())
			//and last but not least d|Sigma|
					+ dSigma.cwiseProduct(bf->getInverseWeightPrior()).sum();
		}

	}
	return gradient;
}

void libgp::DegGaussianProcess::update_k_star(const Eigen::VectorXd& x_star) {
	k_star = bf->computeBasisFunctionVector(x_star);
}

void libgp::DegGaussianProcess::update_alpha() {
	std::cout << "deg_gp: computing alpha" << std::endl;
	squared_noise = exp(2*bf->getLogNoise());
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	Phiy = Phi * y;
	alpha = L.triangularView<Eigen::Lower>().solve(Phiy);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);
	std::cout << "deg_gp: done" << std::endl;
}

void libgp::DegGaussianProcess::computeCholesky() {
//TODO: this step can be simplified for Solin! (Phi is constant)
	std::cout << "deg_gp: computing Phi ... " << std::endl;
	size_t n = sampleset->size();
	if (n > Phi.rows())
		Phi.resize(M, n);
	for (size_t i = 0; i < n; i++)
		Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
	std::cout << "deg_gp: done" << std::endl;
	std::cout << "deg_gp: computing Cholesky ... " << std::endl;
	//L = (Phi * Phi.transpose() + squared_noise * bf->getInverseWeightPrior());
	L = (Phi * Phi.transpose() + squared_noise * bf->getInverseWeightPrior()).selfadjointView<
					Eigen::Lower>().llt().matrixL();
	std::cout << "deg_gp: done" << std::endl;
}

void libgp::DegGaussianProcess::updateCholesky(const double x[], double y) {
//Do nothing and just recompute everything.
//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

}
