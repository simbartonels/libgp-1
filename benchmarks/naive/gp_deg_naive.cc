// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/basisf_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"

#include <cmath>
#include "gp_deg_naive.h"

namespace libgp {

const double log2piOver2 = log(2 * M_PI) / 2;

libgp::DegGaussianProcessNaive::DegGaussianProcessNaive(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) :
		AbstractGaussianProcess(input_dim, covf_def) {
	BasisFFactory factory;
	//wrap initialized covariance function with basis function
	cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	cf->loghyper_changed = 0;
	bf = (IBasisFunction *) cf;
	sigmaIsDiagonal = bf->sigmaIsDiagonal();
	log_noise = bf->getLogNoise();
	squared_noise = exp(2 * log_noise);
	M = bf->getNumberOfBasisFunctions();
	alpha.resize(M);
	Phiy.resize(M);
	L.resize(M, M);
	k_star.resize(M);
	temp.resize(M);

	diSigma.resize(M, M);
	diSigma.setZero();
	dPhidi.resize(M, 1);
	dPhidi.setZero();

}

libgp::DegGaussianProcessNaive::~DegGaussianProcessNaive() {
}

double libgp::DegGaussianProcessNaive::var_impl(const Eigen::VectorXd &x_star) {
	temp = L.triangularView<Eigen::Lower>().solve(k_star);
	return squared_noise * temp.squaredNorm();
}

double libgp::DegGaussianProcessNaive::log_likelihood_impl() {
	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	double halfLogDetA = 0;
	double halfLogDetSigma = bf->getLogDeterminantOfSigma();
	//TODO: does this call work?
//	halfLogDetA = L.diagonal().log().sum();
	for (size_t j = 0; j < M; j++) {
		halfLogDetA += log(L(j, j));
	}
	double llh = (yy - PhiyAlpha) / squared_noise / 2 + halfLogDetA
			+ halfLogDetSigma + (n - M) * log_noise + n * log2piOver2;
	return llh;
}

Eigen::VectorXd libgp::DegGaussianProcessNaive::log_likelihood_gradient_impl() {
	//TODO: refactor, this method is too long.
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	size_t n = sampleset->size();
	if (n > dPhidi.cols()) {
		dPhidi.resize(M, n);
		dPhidi.setZero();
	}
	//TODO: here we have a few steps that aren't necessary for Solin!
	Eigen::VectorXd phi_alpha_minus_y = Phi.transpose() * alpha - y;
	//TODO: move allocations to constructor
	Eigen::VectorXd t(M);
	Eigen::MatrixXd Gamma(M, M);
	Eigen::MatrixXd iAPhi = L.triangularView<Eigen::Lower>().solve(Phi);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(iAPhi);

	//TODO: is it possible to speed this up if sigma is diagonal?
	//TODO: it is certainly faster to precompute this
	//on the other hand: is there not going to be another hyper-parameter update after this call anyway?
	//Will be A^-1
	Gamma.setIdentity();
	L.triangularView<Eigen::Lower>().solveInPlace(Gamma);
	Gamma = Gamma.transpose() * Gamma;
	for (size_t i = 0; i < num_params - 1; i++) {
		//let's start with dA
		if (!bf->gradBasisFunctionIsNull(i)) {
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
		if (!bf->gradiSigmaIsNull(i)) {
			bf->gradiSigma(i, diSigma);
			/*
			 * Hopefully the compiler can optimize the following branch, i.e. use that
			 * sigmaIsDiagonal does not change.
			 */
			if (sigmaIsDiagonal) {
				//these are the Sigma and |Sigma| parts from the derivatives of A and |A|
				//we divert from the thesis here since we have the gradient of Sigma^-1
				gradient(i) += (
				//no multiplication with sn2 since it cancels
				(alpha.transpose() * diSigma.diagonal().cwiseProduct(alpha)
						+ squared_noise
								* Gamma.diagonal().cwiseProduct(
										diSigma.diagonal()).sum())
				//and last but not least d log(|Sigma|)
						- diSigma.diagonal().cwiseProduct(
								bf->getSigma().diagonal()).sum()) / 2;
			} else {
				//these are the Sigma and |Sigma| parts from the derivatives of A and |A|
				//we divert from the thesis here since we have the gradient of Sigma^-1
				gradient(i) += (
				//no multiplication with sn2 since it cancels
				(alpha.transpose() * diSigma * alpha
						+ squared_noise * Gamma.cwiseProduct(diSigma).sum())
				//and last but not least d log(|Sigma|)
						- diSigma.cwiseProduct(bf->getSigma()).sum()) / 2;
			}
		}
	}
	//noise gradient
	//TODO: this stuff is parameter-independent! move it to compute alpha?
	double tr_iAiSigma;
	double alpha_iSigma_alpha;
	if (sigmaIsDiagonal) {
		tr_iAiSigma = Gamma.diagonal().cwiseProduct(
				bf->getInverseOfSigma().diagonal()).sum();
		alpha_iSigma_alpha = (alpha.array().square() * bf->getInverseOfSigma().diagonal().array()).sum();
	} else {
		tr_iAiSigma = Gamma.cwiseProduct(bf->getInverseOfSigma()).sum();
		alpha_iSigma_alpha = alpha.transpose() * bf->getInverseOfSigma() * alpha;
	}
	gradient(num_params - 1) = -(yy / squared_noise - PhiyAlpha / squared_noise
			- alpha_iSigma_alpha
			- squared_noise * tr_iAiSigma - n + M);
	return gradient;
}

void libgp::DegGaussianProcessNaive::update_k_star(const Eigen::VectorXd& x_star) {
	k_star = bf->computeBasisFunctionVector(x_star);
}

void libgp::DegGaussianProcessNaive::update_alpha() {
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	Phiy = Phi * y;
	alpha = L.triangularView<Eigen::Lower>().solve(Phiy);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);

	//this is stuff we need for the computation of the likelihood
	//TODO: What if the user is not interested in doing that?
	yy = y.squaredNorm();
	assert(yy == y.transpose() * y);
	PhiyAlpha = Phiy.transpose() * alpha;
}

void libgp::DegGaussianProcessNaive::computeCholesky() {
	log_noise = bf->getLogNoise();
	squared_noise = exp(2 * log_noise);
//TODO: this step can be simplified for Solin! (Phi is constant)
	//the best solution is probably to create gp_solin that inherits from gp_deg
	size_t n = sampleset->size();
	if (n > Phi.cols())
		Phi.resize(M, n);
	for (size_t i = 0; i < n; i++){
		Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
	}
	//TODO: find the fastest way to compute phi*phi'
	//TODO: for which ever reason the lines below can not be put in one line
	L = Phi * Phi.transpose();
	L += squared_noise * bf->getInverseOfSigma();
	L = L.selfadjointView<Eigen::Lower>().llt().matrixL();
//	L =
//			(Phi * Phi.transpose() + squared_noise * bf->getInverseOfSigma()).selfadjointView<
//					Eigen::Lower>().llt().matrixL();
}

void libgp::DegGaussianProcessNaive::updateCholesky(const double x[], double y) {
//Do nothing and just recompute everything.
//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

}