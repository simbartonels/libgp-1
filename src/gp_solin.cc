// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_solin.h"
namespace libgp {

libgp::SolinGaussianProcess::SolinGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) :
		DegGaussianProcess(input_dim, covf_def, num_basisf, basisf_def) {
	newDataPoints = false;
	PhiPhi.resize(M, M);
}

libgp::SolinGaussianProcess::~SolinGaussianProcess() {
}

void libgp::SolinGaussianProcess::updateCholesky(const double x[], double y) {
	newDataPoints = true;
}

void libgp::SolinGaussianProcess::computeCholesky() {
	if (newDataPoints) {
		newDataPoints = false;
		size_t n = sampleset->size();
		if (n > Phi.cols())
			Phi.resize(M, n);
		for (size_t i = 0; i < n; i++)
			Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
		PhiPhi.setZero();
		PhiPhi.selfadjointView<Eigen::Lower>().rankUpdate(Phi);
	}
	log_noise = bf->getLogNoise();
	squared_noise = exp(2 * log_noise);

	L.triangularView<Eigen::Lower>() = PhiPhi.triangularView<Eigen::Lower>();
	//TODO: unfair advantage over fast food
	L.diagonal() += squared_noise * bf->getInverseOfSigma().diagonal();
	L.triangularView<Eigen::Lower>() =
			L.selfadjointView<Eigen::Lower>().llt().matrixL();
}

inline void libgp::SolinGaussianProcess::llh_setup() {
	//do nothing
	size_t n = sampleset->size();
	if (n > dPhidi.cols()) {
		dPhidi.resize(M, n);
	}
}

}
