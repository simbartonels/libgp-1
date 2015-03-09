#include "basis_functions/bf_fic.h"
namespace libgp {

libgp::FIC::FIC() {
}

libgp::FIC::~FIC() {
}

Eigen::VectorXd libgp::FIC::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {
	Eigen::VectorXd kx(M);
	for (size_t m = 0; m < M; m++)
		kx(m) = cov->get(x, U.row(m));
	return kx;
}

const Eigen::MatrixXd& libgp::FIC::getInverseOfSigma() {
	return iSigma;
}

const Eigen::MatrixXd& libgp::FIC::getCholeskyOfInvertedSigma() {
	return choliSigma;
}

const Eigen::MatrixXd& libgp::FIC::getSigma() {
	return Sigma;
}

double libgp::FIC::getLogDeterminantOfSigma() {
	return half_log_det_sigma;
}

void libgp::FIC::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
	//TODO: highly inefficient
	//TODO: remember that noise gradient is somewhere else!
	Eigen::VectorXd grad2(cov_params.size());
	cov->grad(x1, x2, grad2);
	grad.head(cov_params.size()) = grad2;
	grad.tail(grad.size()-cov_params.size()).setZero();
//	grad.head(cov_params.size() - 1) = grad2.head(cov_params.size() - 1);
//	grad.tail(grad.size() - cov_params.size() - 1).setZero();
//	grad(loghyper.size() - 1) = grad2(cov_params.size() - 1);
}

//void gradDiagWrapped(SampleSet * sampleset, const Eigen::VectorXd & diagK, size_t parameter, Eigen::VectorXd & gradient){
//	//TODO: should I implement this? it can be much more efficient for stationary kernels
//}

bool libgp::FIC::gradDiagWrappedIsNull(size_t parameter) {
	return (parameter >= cov_params.size());
}

void libgp::FIC::gradBasisFunction(SampleSet* sampleSet,
		const Eigen::MatrixXd& Phi, size_t p, Eigen::MatrixXd& Grad) {
	//TODO: highly inefficient
	if (p < cov_params.size() - 1) {
		Eigen::VectorXd grad(cov_params.size());
		for (size_t i = 0; i < sampleSet->size(); i++) {
			for (size_t j = 0; j < M; j++) {
				cov->grad(sampleSet->x(i), U.row(j), grad);
				Grad(j, i) = grad(p);
			}
			//					}
		}
	} else if (p < loghyper.size() - 1) {
		Eigen::VectorXd grad(input_dim);
		Grad.setZero();
		size_t m = (p - cov_params.size() + 1) % M;
		size_t d = (p - cov_params.size() + 1 - m) / M;
		for (size_t i = 0; i < sampleSet->size(); i++) {
			cov->grad_input(U.row(m), sampleSet->x(i), grad);
			Grad(m, i) = grad(d);
		}
		return;
	} else {
		//noise gradient is 0
		Grad.setZero();
	}
}

bool libgp::FIC::gradBasisFunctionIsNull(size_t p) {
	return (p >= cov_params.size());
}

void libgp::FIC::gradiSigma(size_t p, Eigen::MatrixXd& diSigmadp) {
	//TODO: highly inefficient
	if (p < cov_params.size() - 1) {
		Eigen::VectorXd grad(cov_params.size());
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j <= i; j++) {
				cov->grad(U.row(i), U.row(j), grad);
				diSigmadp(i, j) = grad(p);
			}
		}
		//TODO: can we avoid this copy?
		diSigmadp = diSigmadp.selfadjointView<Eigen::Lower>();
	} else if (p < loghyper.size() - 1) {
		//TODO: use that FIC uses only lower half (i.e. dSigma is assumed self-adjoint)
		diSigmadp.setZero();
		Eigen::VectorXd grad(input_dim);
		size_t m = (p - cov_params.size() + 1) % M;
		size_t d = (p - cov_params.size() + 1 - m) / M;
		for (size_t i = 0; i < M; i++) {
			cov->grad_input(U.row(m), U.row(i), grad);
			diSigmadp(i, m) = grad(d);
		}
		diSigmadp.row(m).array() = diSigmadp.col(m).transpose().array();
	} else {
		//TODO: inducing noise gradient
		diSigmadp.setZero();
		diSigmadp.diagonal().fill(2 * snu2);
	}
}

bool libgp::FIC::gradiSigmaIsNull(size_t p) {
	return p < cov_params.size();
}

std::string libgp::FIC::to_string() {
	return "FIC";
}

std::string FIC::pretty_print_parameters() {
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	std::stringstream ss;
	ss << loghyper.format(CleanFmt);
	return ss.str();
}

void libgp::FIC::log_hyper_updated(const Eigen::VectorXd& p) {
	//TODO: probably already done in IBasisFunction
//	loghyper = p;
	size_t cov_params_size = cov_params.size();
	//exchange noise and last parameter
//	loghyper.tail(loghyper.size() - cov_params_size) = p.segment(cov_params_size-1, loghyper.size() - cov_params_size);
//	std::cout << "FIC: loghyper" << std::endl;
//	std::cout << loghyper.transpose() << std::endl;
//	loghyper.segment(cov_params_size, p.size() - cov_params_size - 1) =
//			p.segment(cov_params_size - 1, p.size() - cov_params_size);
//	loghyper(cov_params_size - 1) = p(p.size() - 1);
//	std::cout << loghyper.transpose() << std::endl;

	cov_params.head(cov_params_size - 1) = loghyper.head(cov_params_size - 1);
	//TODO: strong assumption that noise parameter is the last
	cov_params.tail(1) = loghyper.tail(1);
	cov->set_loghyper(cov_params);
	snu2 = 1e-6 * exp(2 * loghyper(loghyper.size() - 1));
	size_t idx = 0;
	/*
	 * The loop needs to be like this as to read the parameters in the right order.
	 * Also for consistency to bf_multi_scale.
	 */
	for (size_t d = 0; d < input_dim; d++) {
		for (size_t m = 0; m < M; m++) {
			//-1 for the noise
			U(m, d) = loghyper(idx + cov_params_size - 1);
			idx++;
		}
	}
	for (size_t m = 0; m < M; m++) {
		for (size_t m2 = 0; m2 < m; m2++)
			//TODO: add inducing input noise and adapt gradients accordingly
			iSigma(m, m2) = cov->get(U.row(m), U.row(m2));
		iSigma(m, m) = cov->get(U.row(m), U.row(m)) + snu2;
	}
	//TODO: is it possible to avoid that? Can we use views in general?
	iSigma = iSigma.selfadjointView<Eigen::Lower>();
	choliSigma = iSigma.selfadjointView<Eigen::Lower>().llt().matrixL();
	Sigma = choliSigma.triangularView<Eigen::Lower>().solve(
			Eigen::MatrixXd::Identity(M, M));
	choliSigma.transpose().triangularView<Eigen::Upper>().solveInPlace(Sigma);
	half_log_det_sigma = -choliSigma.diagonal().array().log().sum();
}

bool libgp::FIC::real_init() {
	//TODO: somehow check that provided covariance function has a noise parameter!
	U.resize(M, input_dim);
	Sigma.resize(M, M);
	iSigma.resize(M, M);
	iSigma.setZero();
	choliSigma.resize(M, M);
	cov_params.resize(cov->get_param_dim());
	return true;
}

size_t libgp::FIC::get_param_dim_without_noise(size_t input_dim,
		size_t num_basis_functions) {
	//-1 for the noise.
	return input_dim * num_basis_functions + cov->get_param_dim() - 1;
}
}
