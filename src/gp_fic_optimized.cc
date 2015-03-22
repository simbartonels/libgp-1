// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.
#include "gp_fic_optimized.h"
#include "basis_functions/bf_fic.h"
namespace libgp {

OptFICGaussianProcess::OptFICGaussianProcess(size_t input_dim, std::string covf_def,
		size_t num_basisf, std::string basisf_def) : FICGaussianProcess(input_dim, covf_def, num_basisf, basisf_def){
	optimize = false;
	temp_input_dim.resize(input_dim);
	dkuui.resize(M);
};


double OptFICGaussianProcess::grad_basis_function(size_t i,
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
		if (!gradiSigmaIsNull){
			//we misuse v as temporary variable here
			//TODO: this could be wrong
			std::cout << "v: " << v.size() << " B: " << B.rows() << ", " << B.cols() << std::endl;
			v = dkuui.transpose() * B;
			//v = B * dkuui;
			std::cout << "R: " << R.rows() << ", " << R.cols() << std::endl;
			R.row(m) -= v.transpose();
			std::cout << "okay" << std::endl;
			R.col(m).array() -= dkuui.array() * B.col(m).array();
			std::cout << " the problem is not here " << std::endl;
		}
	} else {
		wdKuial = FICGaussianProcess::grad_basis_function(i,
				gradBasisFunctionIsNull, gradiSigmaIsNull);
	}
	return wdKuial;
}

double OptFICGaussianProcess::grad_isigma(size_t p, bool gradiSigmaIsNull) {
	double wdKuuiw;
	size_t bf_params_size = bf->get_param_dim();
	size_t cov_params_size = bf_params_size - M * input_dim;
	//TODO: it would be better to make this const...
	Eigen::Map<const Eigen::MatrixXd> U(((FIC *) bf)->U.data(), M, input_dim);
	if (p >= cov_params_size - 1 && p < bf_params_size - 1) {
		optimize = true;
//			diSigmadp.setZero();
		m = (p - cov_params_size + 1) % M;
		d = (p - cov_params_size + 1 - m) / M;
		for (size_t i = 0; i < M; i++) {
			(bf->cov)->grad_input(U.row(m), U.row(i), temp_input_dim);
			dkuui(i) = temp_input_dim(d);
		}
		dkuui(m) /= 2;
		wdKuuiw = 2 * (w.array().square() * dkuui.array()).sum();
		//TODO: remove
		assert(wdKuuiw == FICGaussianProcess::grad_isigma(p, gradiSigmaIsNull));
	} else {
		optimize = false;
		wdKuuiw = FICGaussianProcess::grad_isigma(p, gradiSigmaIsNull);
	}
	return wdKuuiw;
}
}
