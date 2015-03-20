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
	cov->grad(x1, x2, temp_cov_params_size);
	grad.setZero();
	grad.head(cov_params.size() - 1) = temp_cov_params_size.head(
			cov_params.size() - 1);
	grad(loghyper.size() - 1) = temp_cov_params_size(cov_params.size() - 1);
}

void FIC::gradDiagWrapped(SampleSet * sampleset, const Eigen::VectorXd & diagK,
		size_t parameter, Eigen::VectorXd & gradient) {
	/**
	 * TODO: TO WHOMEVER ADAPTS THIS FUNCTION:
	 * 1) Remove this function or adapt a more efficient of what the method in IBasisFunction.
	 * 2) Note the function below: remove the if branch.
	 */
	if (parameter == input_dim || parameter == loghyper.size() - 1) {
		//amplitude gradient or noise gradient
		gradient.fill(2 * exp(2 * loghyper(parameter)));
	} else {
		gradient.setZero();
	}
}

bool libgp::FIC::gradDiagWrappedIsNull(size_t parameter) {
	//TODO: remove
	if (parameter < cov_params_size - 2)
		return true;
	return (parameter >= cov_params_size - 1
			&& parameter < loghyper.size() - 1);
}

void libgp::FIC::gradBasisFunction(SampleSet* sampleSet,
		const Eigen::MatrixXd& Phi, size_t p, Eigen::MatrixXd& Grad) {
	if (p < cov_params_size - 1) {
		for (size_t i = 0; i < sampleSet->size(); i++) {
			for (size_t j = 0; j < M; j++) {
				cov->grad(sampleSet->x(i), U.row(j), temp_cov_params_size);
				Grad(j, i) = temp_cov_params_size(p);
			}
		}
	} else if (p < loghyper.size() - 1) {
		Grad.setZero();
		size_t m = (p - cov_params.size() + 1) % M;
		size_t d = (p - cov_params.size() + 1 - m) / M;
		for (size_t i = 0; i < sampleSet->size(); i++) {
			cov->grad_input(U.row(m), sampleSet->x(i), temp_input_dim);
			Grad(m, i) = temp_input_dim(d);
		}
		return;
	} else {
		//noise gradient is 0
		Grad.setZero();
	}
}

bool libgp::FIC::gradBasisFunctionIsNull(size_t p) {
	return (p == loghyper.size() - 1);
}

void libgp::FIC::gradiSigma(size_t p, Eigen::MatrixXd& diSigmadp) {
	if (p < cov_params_size - 1) {
		//this could be a bit faster for the length scale gradient but since we iterate only over M...
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j <= i; j++) {
				cov->grad(U.row(i), U.row(j), temp_cov_params_size);
				diSigmadp(i, j) = temp_cov_params_size(p);
			}
		}
		diSigmadp = diSigmadp.selfadjointView<Eigen::Lower>();
	} else if (p < loghyper.size() - 1) {
		diSigmadp.setZero();
		size_t m = (p - cov_params.size() + 1) % M;
		size_t d = (p - cov_params.size() + 1 - m) / M;
		for (size_t i = 0; i < M; i++) {
			cov->grad_input(U.row(m), U.row(i), temp_input_dim);
			diSigmadp(i, m) = temp_input_dim(d);
		}
		diSigmadp.row(m).array() = diSigmadp.col(m).transpose().array();
	} else {
		//TODO: is this correct? tests fail occasionally
		diSigmadp.setZero();
		diSigmadp.diagonal().fill(2 * snu2);
	}
}

bool libgp::FIC::gradiSigmaIsNull(size_t p) {
	return false;
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
	cov_params.head(cov_params_size - 1) = loghyper.head(cov_params_size - 1);
	//TODO: strong assumption that noise parameter is the last
	cov_params.tail(1) = loghyper.tail(1);
	cov->set_loghyper(cov_params);
	double sn2 = exp(2 * loghyper(loghyper.size() - 1));
	snu2 = 1e-6 * sn2;
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

bool libgp::FIC::real_init() {
	if (cov->to_string().compare("CovSum(CovSEard, CovNoise)") != 0) {
		std::cerr
				<< "BF_FIC: Currently supporting only 'CovSum(CovSEard, CovNoise)'. "
				<< std::endl;
		std::cerr << "You tried '" << cov->to_string() << "'" << std::endl;
		std::cerr
				<< "BF_FIC: To change this look into the function gradDiagWrapped."
				<< std::endl;
		return false;
	}
	U.resize(M, input_dim);
	Sigma.resize(M, M);
	iSigma.resize(M, M);
	iSigma.setZero();
	choliSigma.resize(M, M);
	cov_params_size = cov->get_param_dim();
	cov_params.resize(cov_params_size);
	temp_cov_params_size.resize(cov_params.size());
	temp_input_dim.resize(input_dim);
	return true;
}

size_t libgp::FIC::get_param_dim_without_noise(size_t input_dim,
		size_t num_basis_functions) {
	//-1 for the noise.
	return input_dim * num_basis_functions + cov->get_param_dim() - 1;
}
}
