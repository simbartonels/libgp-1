// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.
#include "gp_fic_optimized.h"
#include "basis_functions/bf_fic.h"
namespace libgp {

OptFICGaussianProcess::OptFICGaussianProcess(size_t input_dim,
		std::string covf_def, size_t num_basisf, std::string basisf_def) :
		FICGaussianProcess(input_dim, covf_def, num_basisf, basisf_def) {
	optimize = false;
	temp_input_dim.resize(input_dim);
	dkuui.resize(M);
}
;

double OptFICGaussianProcess::grad_basis_function(size_t i,
		bool gradBasisFunctionIsNull, bool gradiSigmaIsNull) {
	double wdKuial;
	if (optimize) {
		RWdg.setZero();
		//TODO: remove
		R.setZero();
		if (!gradBasisFunctionIsNull) {
			size_t n = sampleset->size();
			if(dkui.size() < n)
				dkui.resize(n);
			Eigen::Map<const Eigen::MatrixXd> U(((FIC *) bf)->U.data(), M, input_dim);
			for (size_t j = 0; j < n; j++) {
				bf->cov->grad_input(U.row(m), sampleset->x(j), temp_input_dim);
				dkui(j) = temp_input_dim(d);
			}
			wdKuial = 2 * w(m) * (dkui.array() * al.array()).sum(); //O(n)
					//R = 2*dKui-dKuui*B;
			//TODO: remove
			R.row(m) = 2 * dkui.transpose();
			v = 2 * dkui;
			RWdg.row(m) = v.transpose() * Wdg.transpose();
		} else {
			wdKuial = 0;
			v.setZero();
		}
		if (!gradiSigmaIsNull) {
			R.row(m) -= dkuui.transpose() * B;
			RWdg.row(m) -= dkuui.transpose() * BWdg;
			for(size_t j=0; j < M; j++){
				R.row(j) -= dkuui(j) * B.row(m);
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
	//TODO: optimize this!
	v = -R.cwiseProduct(B).colwise().sum();
	return wdKuial;
}

double OptFICGaussianProcess::grad_isigma(size_t p, bool gradiSigmaIsNull) {
	double wdKuuiw;
	size_t bf_params_size = bf->get_param_dim();
	size_t cov_params_size = bf_params_size - M * input_dim;
	Eigen::Map<const Eigen::MatrixXd> U(((FIC *) bf)->U.data(), M, input_dim);
	if (p >= cov_params_size - 1 && p < bf_params_size - 1) {
		optimize = true;
		m = (p - cov_params_size + 1) % M;
		d = (p - cov_params_size + 1 - m) / M;
		for (size_t i = 0; i < M; i++) {
			(bf->cov)->grad_input(U.row(m), U.row(i), temp_input_dim);
			dkuui(i) = temp_input_dim(d);
		}
		/*
		 * This step is needed to assume dKuui = A + B where
		 * A[i,j] = \delta_{im} dkuui[j]
		 * B[i,j] = \delta_{jm} dkuui[i]
		 */
		dkuui(m) /= 2;
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
