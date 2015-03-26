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
	RWdg.resize(M, M);
}
;

double OptMultiScaleGaussianProcess::grad_basis_function(size_t i,
		bool gradBasisFunctionIsNull, bool gradiSigmaIsNull) {
	double wdKuial;
	if (optimize) {
		if (!gradBasisFunctionIsNull) {
			bf->gradBasisFunction(sampleset, Phi, i, dKui);
			wdKuial = 2 * (w.transpose() * dKui * al).sum(); //O(Mn)
					//R = 2*dKui-dKuui*B;
			R = 2 * dKui;
		} else {
			R.setZero();
			wdKuial = 0;
		}
		if (!gradiSigmaIsNull) {
			R.row(m) -= dkuui.transpose() * B;
			for(size_t j=0; j < M; j++)
				R.row(j) -= dkuui(j) * B.row(m);
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

double OptMultiScaleGaussianProcess::grad_isigma(size_t p, bool gradiSigmaIsNull) {
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

Eigen::VectorXd OptMultiScaleGaussianProcess::log_likelihood_gradient_impl() {
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	log_likelihood_gradient_precomputations();
	for (size_t i = 0; i < num_params; i++) {
//      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);
		bool gradiSigmaIsNull = bf->gradiSigmaIsNull(i);
		bool gradBasisFunctionIsNull = bf->gradBasisFunctionIsNull(i);

		double wdKuuiw = grad_isigma(i, gradiSigmaIsNull);

		double wdKuial = grad_basis_function(i, gradBasisFunctionIsNull, gradiSigmaIsNull);

		double ddiagK_idg;
		if (!bf->gradDiagWrappedIsNull(i)) {
			bf->gradDiagWrapped(sampleset, k, i, ddiagK);
			ddiagK_idg = ddiagK.cwiseQuotient(dg).sum();
			// v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
			v = ddiagK.transpose() - R.cwiseProduct(B).colwise().sum();
		} else {
			ddiagK_idg = 0;
			v = -R.cwiseProduct(B).colwise().sum();
		}

		if(optimize){
			if(m == 1)
				tempM = R.row(2) * Wdg.transpose();
			else
				tempM = R.row(1) * Wdg.transpose();
			dkuui = R.row(m) * Wdg.transpose();
			for(size_t j = 0; j < M; j++)
				RWdg.row(j) = tempM.transpose();
			RWdg.row(m)  = dkuui.transpose();
		}
		else
			RWdg = R * Wdg.transpose();

		gradient(i) = ddiagK_idg
				+ wdKuuiw - wdKuial
				- (v.array() * alSqrd.array()).sum()
				- WdgSum.cwiseProduct(v).sum()
				- (RWdg).cwiseProduct(BWdg).sum(); //O(M^2n)
	}
	gradient /= 2;
	//noise gradient included in the loop above
	return -gradient;
}
}
