#include "basis_functions/bf_FIC_fixed.h"
namespace libgp {

libgp::FICfixed::FICfixed() {
}

libgp::FICfixed::~FICfixed() {
}

void FICfixed::setU(const Eigen::MatrixXd & U){
	this->U = U;
	U_initialized = true;
}

void libgp::FICfixed::gradBasisFunction(SampleSet* sampleSet,
		const Eigen::MatrixXd& Phi, size_t p, Eigen::MatrixXd& Grad) {
	if (p < cov_params_size - 1) {
		for (size_t i = 0; i < sampleSet->size(); i++) {
			for (size_t j = 0; j < M; j++) {
				cov->grad(sampleSet->x(i), U.row(j), temp_cov_params_size);
				Grad(j, i) = temp_cov_params_size(p);
			}
		}
	} else {
		//noise gradient is 0
		Grad.setZero();
	}
}

void libgp::FICfixed::gradiSigma(size_t p, Eigen::MatrixXd& diSigmadp) {
	if (p < cov_params_size - 1) {
		//this could be a bit faster for the length scale gradient but since we iterate only over M...
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j <= i; j++) {
				cov->grad(U.row(i), U.row(j), temp_cov_params_size);
				diSigmadp(i, j) = temp_cov_params_size(p);
			}
		}
		diSigmadp = diSigmadp.selfadjointView<Eigen::Lower>();
	} else {
		//TODO: is this correct? tests fail occasionally
		diSigmadp.setZero();
		diSigmadp.diagonal().fill(2 * snu2);
	}
}

std::string libgp::FICfixed::to_string() {
	return "FICfixed";
}

void libgp::FICfixed::log_hyper_updated(const Eigen::VectorXd& p) {
	if(!U_initialized){
		std::cerr << "ERROR: The inducing points have not been initialized! ACTION: Setting U random!"
		<<std::endl;
		U.setRandom();
		U_initialized = true;
	}
	cov_params.head(cov_params_size - 1) = loghyper.head(cov_params_size - 1);
	//TODO: strong assumption that noise parameter is the last
	cov_params.tail(1) = loghyper.tail(1);
	cov->set_loghyper(cov_params);
	double sn2 = exp(2 * loghyper(loghyper.size() - 1));
	snu2 = 1e-6 * sn2;
	size_t idx = 0;
	for (size_t m = 0; m < M; m++) {
		for (size_t m2 = 0; m2 < m; m2++)
			iSigma(m, m2) = cov->get(U.row(m), U.row(m2));
		//is there noise on the diagonal?! seems not (seems like Eigen performs two copy operations!)
		iSigma(m, m) = cov->get(U.row(m), U.row(m)) + snu2; //-sn2;
	}
	iSigma = iSigma.selfadjointView<Eigen::Lower>();
	choliSigma = iSigma.selfadjointView<Eigen::Lower>().llt().matrixL();
	Sigma = choliSigma.triangularView<Eigen::Lower>().solve(
			Eigen::MatrixXd::Identity(M, M));
	choliSigma.transpose().triangularView<Eigen::Upper>().solveInPlace(Sigma);
	half_log_det_sigma = -choliSigma.diagonal().array().log().sum();
}

size_t libgp::FICfixed::get_param_dim_without_noise(size_t input_dim,
		size_t num_basis_functions) {
	//-1 for the noise.
	return cov->get_param_dim() - 1;
}
}
