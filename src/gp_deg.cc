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
	recompute_yy = true;
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
	Gamma.resize(M, M);

	diSigma.resize(M, M);
	diSigma.setZero();
	dPhidi.resize(M, 1);
	dPhidi.setZero();

}

libgp::DegGaussianProcess::~DegGaussianProcess() {
}

double libgp::DegGaussianProcess::var_impl(const Eigen::VectorXd &x_star) {
	temp = L.triangularView<Eigen::Lower>().solve(k_star);
	return squared_noise * temp.squaredNorm();
}

double libgp::DegGaussianProcess::log_likelihood_impl() {
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);

	double halfLogDetSigma = bf->getLogDeterminantOfSigma();
	double halfLogDetA = L.diagonal().array().log().sum();
	double llh = (yy - PhiyAlpha) / squared_noise / 2 + halfLogDetA
			+ halfLogDetSigma + (n - M) * log_noise + n * log2piOver2;
	return llh;
}

void inline DegGaussianProcess::llh_setup_other(){
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	if (n > dPhidi.cols()) {
		dPhidi.resize(M, n);
		phi_alpha_minus_y.resize(n);
		iAPhi.resize(M, n);
	}
	phi_alpha_minus_y = Phi.transpose() * alpha - y;
	iAPhi = Gamma.selfadjointView<Eigen::Lower>() * Phi;
}

void inline DegGaussianProcess::llh_setup_Gamma(){
	//we misuse diSigma as temporary variable here
	diSigma.setIdentity();
	L.triangularView<Eigen::Lower>().solveInPlace(diSigma);
	Gamma.setZero();
	Gamma.selfadjointView<Eigen::Lower>().rankUpdate(diSigma.transpose());
	//TODO: would be nice to avoid this copy
	Gamma.triangularView<Eigen::StrictlyUpper>() = Gamma.triangularView<Eigen::StrictlyLower>().transpose();
	//Gamma is now A^-1
	//is it possible avoid computing Gamma if sigma is diagonal? not when looking at Solin's implementation

	llh_setup_other();

	diSigma.setZero();
	dPhidi.setZero();
}

Eigen::VectorXd libgp::DegGaussianProcess::log_likelihood_gradient_impl() {
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());

	llh_setup_Gamma(); //also calls llh_setup_other().

	for (size_t i = 0; i < num_params - 1; i++) {
		//let's start with dA
		if (!bf->gradBasisFunctionIsNull(i)) {
			bf->gradBasisFunction(sampleset, Phi, i, dPhidi);
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
			gradient(i) += getSigmaGradient(i);
		}
	}

	gradient(num_params - 1) = getNoiseGradient();
	return gradient;
}

inline double DegGaussianProcess::getSigmaGradient(size_t i){
	bf->gradiSigma(i, diSigma);
	/*
	 * Hopefully the compiler can optimize the following branch, i.e. use that
	 * sigmaIsDiagonal does not change.
	 */
	if (sigmaIsDiagonal) {
		//these are the Sigma and |Sigma| parts from the derivatives of A and |A|
		//we divert from the thesis here since we have the gradient of Sigma^-1
		return (
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
		return (
		//no multiplication with sn2 since it cancels
		(alpha.transpose() * diSigma * alpha
				+ squared_noise * diSigma.cwiseProduct(Gamma).sum())
		//and last but not least d log(|Sigma|)
				- diSigma.cwiseProduct(bf->getSigma()).sum()) / 2;
	}
}

inline double DegGaussianProcess::getNoiseGradient(){
	//noise gradient
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
	return -(yy / squared_noise - PhiyAlpha / squared_noise
			- alpha_iSigma_alpha
			- squared_noise * tr_iAiSigma - n + M);
}

void libgp::DegGaussianProcess::update_k_star(const Eigen::VectorXd& x_star) {
	k_star = bf->computeBasisFunctionVector(x_star);
}

void libgp::DegGaussianProcess::update_alpha() {
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
	Phiy = Phi * y;
	alpha = L.triangularView<Eigen::Lower>().solve(Phiy);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);

	//this is stuff needed only in the computation of llh and gradient
	//but it MUST NOT be moved to llh
	//if somebody calls only gradient without llh that would give wrong results
	if(recompute_yy){
		yy = y.squaredNorm();
		recompute_yy = false;
	}
	PhiyAlpha = Phiy.transpose() * alpha;
}

void libgp::DegGaussianProcess::computeCholesky() {
	update_internal_variables();

	for (size_t i = 0; i < n; i++)
		Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));

	L.triangularView<Eigen::Lower>().setZero();
	L.selfadjointView<Eigen::Lower>().rankUpdate(Phi);
	if(sigmaIsDiagonal)
		L.diagonal() += squared_noise * bf->getInverseOfSigma().diagonal();
	else
		L += squared_noise * bf->getInverseOfSigma();
	L.triangularView<Eigen::Lower>() = L.selfadjointView<Eigen::Lower>().llt().matrixL();
}

void DegGaussianProcess::update_internal_variables(){
	log_noise = bf->getLogNoise();
	squared_noise = exp(2 * log_noise);
	n = sampleset->size();
	if (n > Phi.cols())
		Phi.resize(M, n);
}

void libgp::DegGaussianProcess::updateCholesky(const double x[], double y) {
//Do nothing and just recompute everything.
//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
	recompute_yy = true;
}

}
