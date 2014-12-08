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

const double log2piOver2 = log(2 * M_PI) / 2;

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
	double halfLogDetA = 0;
	double halfLogDetSigma = bf->getLogDeterminantOfWeightPrior();
	for (size_t j = 0; j < M; j++) {
		halfLogDetA += log(L(j, j));
	}
	double llh = (yy - PhiyAlpha) / squared_noise / 2 + halfLogDetA
			+ halfLogDetSigma + (n - M) * log_noise + n * log2piOver2;
	return llh;
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
	Eigen::VectorXd phi_alpha_minus_y = Phi.transpose() * alpha - y;
	Eigen::VectorXd iSigma_alpha = bf->getInverseWeightPrior().transpose()
			* alpha;
	Eigen::MatrixXd iAPhi = L.triangularView<Eigen::Lower>().solve(Phi);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(iAPhi);
	Eigen::MatrixXd Gamma = L.triangularView<Eigen::Lower>().solve(
			bf->getInverseWeightPrior());
	//TODO: can this be more efficient?
	Gamma = Gamma.transpose() * Gamma;
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
			/*
			 * dividing by squared noise here is more efficient since alpha and phialpha-y have
			 * more entries than this for loop has iterations.
			 */
			gradient(i) =
					((alpha.transpose() * dPhidi * phi_alpha_minus_y).sum()
							/ squared_noise
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
			//these are the Sigma and |Sigma| parts from the derivatives of A and |A|
			gradient(i) -= (
			//no multiplication with sn2 since it cancels
			(iSigma_alpha.transpose() * dSigma * iSigma_alpha
					+ squared_noise * Gamma.cwiseProduct(dSigma).sum())
			//and last but not least d log(|Sigma|)
					- dSigma.cwiseProduct(bf->getInverseWeightPrior()).sum())
					/ 2;
		}
		//noise gradient
		//TODO: implement (efficiently)
		double triAiSigma = 0;
		//TODO: can be more efficient if the weight prior is diagonal!
		gradient(num_params - 1) = yy / squared_noise
				- PhiyAlpha / squared_noise
				- squared_noise * alpha.transpose() * iSigma_alpha
				- squared_noise * triAiSigma - n + M;
	}
	return gradient;
}

void libgp::DegGaussianProcess::update_k_star(const Eigen::VectorXd& x_star) {
	k_star = bf->computeBasisFunctionVector(x_star);
}

void libgp::DegGaussianProcess::update_alpha() {
	std::cout << "deg_gp: computing alpha" << std::endl;
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	Phiy = Phi * y;
	alpha = L.triangularView<Eigen::Lower>().solve(Phiy);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);
	std::cout << "deg_gp: done" << std::endl;

	//this is stuff we need for the computation of the likelihood
	//TODO: What if the user is not interested in doing that?
	std::cout << "deg_gp: preparing computation of log-likelihood" << std::endl;
	yy = y.squaredNorm();
	assert(yy == y.transpose() * y);
	PhiyAlpha = Phiy.transpose() * alpha;
	std::cout << "deg_gp: done" << std::endl;
}

void libgp::DegGaussianProcess::computeCholesky() {
	log_noise = bf->getLogNoise();
	squared_noise = exp(2 * log_noise);
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
	L =
			(Phi * Phi.transpose() + squared_noise * bf->getInverseWeightPrior()).selfadjointView<
					Eigen::Lower>().llt().matrixL();
	std::cout << "deg_gp: done" << std::endl;
}

void libgp::DegGaussianProcess::updateCholesky(const double x[], double y) {
//Do nothing and just recompute everything.
//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

}
