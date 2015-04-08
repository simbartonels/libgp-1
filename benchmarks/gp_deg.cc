#include <cmath>

#include "gp_deg.h"
#include "naive/gp_deg_naive.h"
#include "gp_solin.h"
#include "gp_utils.h"
#include "util/time_call.h"
#include "basis_functions/bf_solin.h"

using namespace libgp;
static DegGaussianProcess * gp;

static SolinGaussianProcess * gpsolin;

static DegGaussianProcessNaive * gpnaive;

void f1() {
	gp->log_likelihood();
}

void f2() {
	gpnaive->log_likelihood();
}

void testSpeedOfLogLikelihood() {
	double fv = gp->log_likelihood();
	double nv = gpnaive->log_likelihood();
	std::cout << "fv: " << fv << std::endl;
	std::cout << "nv: " << nv << std::endl;
	assert(std::abs(fv - nv) / nv < 1e-5);

	std::cout << "speed of llh computation:" << std::endl;
	compare_time(f1, f2, 5);
}

void compChol_fast() {
	gp->covf().loghyper_changed = true;
	gp->getL();
}

void compChol_base() {
	gpnaive->covf().loghyper_changed = true;
	gpnaive->getL();
}

void testSpeedOfCholeskyComputation2() {
	compare_time(compChol_base, compChol_fast, 10);
}

void llhGradBaseline() {
	gpnaive->log_likelihood_gradient();
}

void llhGradFast() {
	gp->log_likelihood_gradient();
}

void compCholSolin() {
	gpsolin->covf().loghyper_changed = true;
	gpsolin->getL();
}

void llhGradSolin() {
	gpsolin->log_likelihood_gradient();
}

void compareSolinAndDeg() {
	compare_time(compChol_fast, compCholSolin, 10);
	compare_time(llhGradFast, llhGradSolin, 10);
}

void testReSeeding() {
	size_t M = 100;
	Eigen::VectorXd x1(M);
	Eigen::VectorXd x2(M);
	size_t seed = (size_t) time(0);
	srand(seed);
	x1.setRandom();
	x2.setRandom();
	assert(!(x1 - x2).isZero(1e-5));
	std::cout << x1.transpose() << std::endl;
	srand(seed);
	x2.setRandom();
	assert((x1 - x2).isZero(1e-50));
	std::cout << x2.transpose() << std::endl;

}

#define USED_GP "FastFood"

void measureBFcomputationTime() {
	size_t num_execs = 100;
	size_t trials = num_execs;
	std::cout << "Choose D: ";
	size_t D;
	std::cin >> D;
	std::cout << "Choose n: ";
	size_t n;
	std::cin >> n;
	Eigen::VectorXd grad(D);
	Eigen::VectorXd x(D);
	x.setRandom();
	x(0) = 1;
	Eigen::MatrixXd X(n, D);
	X.setRandom();
	Eigen::VectorXd y(n);
	y.setRandom();

	while (true) {
		std::cout << "Choose M: ";
		size_t M;
		std::cin >> M;
		std::cout << "initializing GP using approximation" << USED_GP
				<< std::endl;
		gp = new DegGaussianProcess(D, "CovSum ( CovSEard, CovNoise)", M, USED_GP);
		Eigen::VectorXd params(gp->covf().get_param_dim());
		params.setRandom();
		for (int i = 0; i < n; ++i) {
			gp->add_pattern(X.row(i), y(i));
		}
		Eigen::VectorXd lstretcher(D);
		lstretcher.fill(3);
		gp->add_pattern(lstretcher, 0);
		gp->covf().set_loghyper(params);
		gp->log_likelihood();
		std::cout << "done" << std::endl;
		double tic = -log(0.);
		for (size_t j = 0; j < trials; j++) {
			std::cout << j << "/" << trials << ": " << tic << std::endl;
			stop_watch();
			for (size_t i = 0; i < num_execs; i++) {
				gp->f(x);
				gp->var(x);
				x(0) = -x(0);
			}
			double temp = stop_watch() / num_execs;
			if(temp < tic)
				tic = temp;
		}
		std::cout << "time: " << tic << std::endl;
//    		std::cout << "diag(L): " << gp->getL().diagonal().transpose() << std::endl;
		std::cout << "f(x): " << gp->f(x) << std::endl;
		Eigen::VectorXd bf(M);
		bf = ((IBasisFunction *) &(gp->covf()))->computeBasisFunctionVector(x);
		std::cout << "phi(x).tail(10): " <<  bf.tail(10).transpose() << std::endl;
		delete gp;
	}
}

int main(int argc, char const *argv[]) {
	measureBFcomputationTime();
//	testReSeeding();
	return 0;
	size_t input_dim = 2;
	size_t M = 2048;
	size_t n = 5000;
	//fast food is not such a good idea since there are some random effects
	gp = new DegGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
			"Solin");
	gpsolin = new SolinGaussianProcess(input_dim,
			"CovSum ( CovSEard, CovNoise)", M);

	// initialize hyper parameter vector
	Eigen::VectorXd params(gp->covf().get_param_dim());
	params.setRandom();
	// set parameters of covariance function
	gp->covf().set_loghyper(params);

	gpnaive = new DegGaussianProcessNaive(input_dim,
			"CovSum ( CovSEard, CovNoise)", M, "Solin");
	gpnaive->covf().set_loghyper(params);

	Eigen::MatrixXd X(n, input_dim);
	X.setRandom();
	Eigen::VectorXd y(n);
	y.setRandom();
	// add training patterns
	for (int i = 0; i < n; ++i) {
		//double x[] = { drand48() * 4 - 2, drand48() * 4 - 2 };
		//double y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
		//gp->add_pattern(x, y);
		//gpnaive->add_pattern(x, y);
		gpsolin->add_pattern(X.row(i), y(i));
	}
	//gp->log_likelihood();
	//gpnaive->log_likelihood();
	gpsolin->log_likelihood();
//	testSpeedOfLogLikelihood();
//	testSpeedOfCholeskyComputation2();
//	compare_time(llhGradBaseline, llhGradFast, 15);
//	compareSolinAndDeg();
	libgp::Solin * bf = (libgp::Solin *) &(gpsolin->covf());
	std::cout << gpsolin->getL().diagonal().transpose() << std::endl;
	std::cout << "phi(X(0)): " << gpsolin->f(X.row(0)) << std::endl;
	std::cout << bf->computeBasisFunctionVector(X.row(0)).transpose()
			<< std::endl;
	std::cout << "Sigma: " << bf->getSigma().diagonal().transpose()
			<< std::endl;
	std::cout << "log(|Sigma|): " << bf->getLogDeterminantOfSigma()
			<< std::endl;
	std::cout << gpsolin->getL().diagonal().array().log().sum() << std::endl;
	std::cout << log(gpsolin->getL().diagonal().array().prod()) << std::endl;
	std::cout << "starting measurements" << std::endl;
	stop_watch();
	std::cout << gpsolin->log_likelihood() << std::endl;
	std::cout << stop_watch() << std::endl;
	std::cout << gpsolin->log_likelihood() << std::endl;
	std::cout << stop_watch() << std::endl;

	delete gp;
	delete gpnaive;
	return 0;
}
