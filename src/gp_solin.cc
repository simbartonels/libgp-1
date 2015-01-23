// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_solin.h"
namespace libgp {

libgp::SolinGaussianProcess::SolinGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) :
		DegGaussianProcess(input_dim, covf_def, num_basisf, basisf_def) {
	newDataPoints = true;
	PhiPhi.resize(M, M);
}

libgp::SolinGaussianProcess::~SolinGaussianProcess() {
}

void libgp::SolinGaussianProcess::updateCholesky(const double x[], double y) {
	newDataPoints = true;
	//recompute_yy is updated in the parent method
	DegGaussianProcess::updateCholesky(x, y);
}

void libgp::SolinGaussianProcess::update_alpha() {
	if(recompute_yy){
		const std::vector<double>& targets = sampleset->y();
		Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
		Phiy = Phi * y;
		yy = y.squaredNorm();
		recompute_yy = false;
	}
	alpha = L.triangularView<Eigen::Lower>().solve(Phiy);
	L.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);
	PhiyAlpha = Phiy.transpose() * alpha;
}

void libgp::SolinGaussianProcess::computeCholesky() {
	DegGaussianProcess::update_internal_variables();
	if (newDataPoints) {
		newDataPoints = false;
		if (n > Phi.cols())
			Phi.resize(M, n);
		for (size_t i = 0; i < n; i++)
			Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
		PhiPhi.setZero();
		PhiPhi.selfadjointView<Eigen::Lower>().rankUpdate(Phi);
	}

	L.triangularView<Eigen::Lower>() = PhiPhi.triangularView<Eigen::Lower>();
	L.diagonal() += squared_noise * bf->getInverseOfSigma().diagonal();
	L.triangularView<Eigen::Lower>() =
			L.selfadjointView<Eigen::Lower>().llt().matrixL();
}

inline void libgp::SolinGaussianProcess::llh_setup_other() {
	//do basically nothing
	if (n > dPhidi.cols())
		dPhidi.resize(M, n);
}

}
