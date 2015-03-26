// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.
#include "gp_multiscale_optimized.h"
#include "basis_functions/bf_multi_scale.h"
namespace libgp {

OptMultiScaleGaussianProcess::OptMultiScaleGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) :
		FICGaussianProcess(input_dim, covf_def, num_basisf, basisf_def) {
	optimize = false;
	temp_input_dim.resize(input_dim);
	dkuui.resize(M);
}
;

double OptMultiScaleGaussianProcess::grad_basis_function(size_t i,
		bool gradBasisFunctionIsNull, bool gradiSigmaIsNull) {
	double wdKuial;
	if (optimize) {
//		double wdKuial_target = FICGaussianProcess::grad_basis_function(i,
//			gradBasisFunctionIsNull, gradiSigmaIsNull);
		RWdg.setZero();
		if (!gradBasisFunctionIsNull) {
			size_t n = sampleset->size();
			if (dkui.size() < n)
				dkui.resize(n);
			((MultiScale *) bf)->gradBasisFunctionVector(sampleset, Phi, i,
					dkui);
			v = 2 * dkui;
			wdKuial = w(m) * (v.array() * al.array()).sum(); //O(n)
			RWdg.row(m) = v.transpose() * Wdg.transpose();
			v.array() *= -B.row(m).transpose().array();

//			wdKuial_target = std::fabs(wdKuial_target - wdKuial)/(std::fabs(wdKuial_target) + 1e-50);
//			assert(wdKuial_target < 1e-5);
		} else {
			v.setZero();
			wdKuial = 0;
		}
		if (!gradiSigmaIsNull) {
			v += B.row(m).cwiseProduct(dkuui.transpose() * B);
			RWdg.row(m) -= dkuui.transpose() * BWdg;
			for (size_t j = 0; j < M; j++) {
				v += B.row(j).cwiseProduct(dkuui(j) * B.row(m));
				RWdg.row(j) -= dkuui(j) * BWdg.row(m);
			}
//			bf->gradiSigma(i, dKuui);
//			Eigen::MatrixXd R_target = -dKuui * B;
//			if(!gradBasisFunctionIsNull)
//				R_target += 2*dKui;
//			double dist = ((R_target.array()-R.array())/R_target.array()+1e-50).abs().maxCoeff();
//			assert(dist < 1e-5);
		}
	} else {
		wdKuial = FICGaussianProcess::grad_basis_function(i,
				gradBasisFunctionIsNull, gradiSigmaIsNull);
	}
	return wdKuial;
}

double OptMultiScaleGaussianProcess::grad_isigma(size_t p,
		bool gradiSigmaIsNull) {
	double wdKuuiw;
	size_t bf_params_size = bf->get_param_dim();
	size_t cov_params_size = bf_params_size - M * input_dim;
	if (p >= input_dim && p < 2 * M * input_dim + input_dim) {
		optimize = true;
		m = (p - input_dim) % M;
		((MultiScale *) bf)->gradiSigmaVector(p, m, dkuui);
		/*
		 * This step is needed to assume dKuui = A + B where
		 * A[i,j] = \delta_{im} dkuui[j]
		 * B[i,j] = \delta_{jm} dkuui[i]
		 */
		//wdKuuiw = non-zero entry of w^T * B:
		wdKuuiw = (w.array() * dkuui.array()).sum();
		//dkuui = non-zero entries of w^T * A
//		dkuui *= w(m);
		//dkuui = w^T * (A + B)
//		dkuui(m)+=wdKuuiw;
		//wdKuui = w^T * (A * B)* w
//		wdKuuiw = (dkuui.array() * w.array()).sum();
//		wdKuuiw = (w(m) * dkuui.array() * w.array()).sum() + wdKuuiw * w(m);
		wdKuuiw = w(m) * ((dkuui.array() * w.array()).sum() + wdKuuiw);
//		double dist = FICGaussianProcess::grad_isigma(p,
//				gradiSigmaIsNull);
//		dist = std::fabs((wdKuuiw - dist)/(dist + 1e-50));
//		assert(dist < 1e-5);
	} else {
		optimize = false;
		wdKuuiw = FICGaussianProcess::grad_isigma(p, gradiSigmaIsNull);
	}
	return wdKuuiw;
}
}
