#include <cmath>

#include "gp_deg.h"
#include "naive/gp_deg_naive.h"
#include "gp_utils.h"
#include "util/time_call.h"

using namespace libgp;
    static DegGaussianProcess * gp;

    static DegGaussianProcessNaive * gpnaive;

    void f1(){
    	gp->log_likelihood();
    }

    void f2(){
    	gpnaive->log_likelihood();
    }

    void testSpeedOfLogLikelihood(){
    	double fv = gp->log_likelihood();
    	double nv = gpnaive->log_likelihood();
    	std::cout << "fv: " << fv << std::endl;
    	std::cout << "nv: " << nv << std::endl;
    	assert(std::abs(fv-nv)/nv <1e-5);

    	std::cout << "speed of llh computation:" << std::endl;
    	compare_time(f1, f2, 5);
    }

    void compChol_fast(){
    	gp->covf().loghyper_changed = true;
    	gp->getL();
    }

    void compChol_base(){
    	gpnaive->covf().loghyper_changed = true;
    	gpnaive->getL();
    }

    void testSpeedOfCholeskyComputation2(){
    	compare_time(compChol_base, compChol_fast, 10);
    }

    void llhGradBaseline(){
    	gpnaive->log_likelihood_gradient();
    }

    void llhGradFast(){
    	gp->log_likelihood_gradient();
    }

int main(int argc, char const *argv[]) {
	size_t input_dim = 3;
	size_t M = 200;
	size_t n = 2000;
	//fast food is not such a good idea since there are some random effects
	gp = new DegGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
			"Solin");

	// initialize hyper parameter vector
	Eigen::VectorXd params(gp->covf().get_param_dim());
	params.setRandom();
	// set parameters of covariance function
	gp->covf().set_loghyper(params);

	gpnaive = new DegGaussianProcessNaive(input_dim, "CovSum ( CovSEard, CovNoise)",
			M, "Solin");
	gpnaive->covf().set_loghyper(params);

	// add training patterns
	for (int i = 0; i < n; ++i) {
		double x[] = { drand48() * 4 - 2, drand48() * 4 - 2 };
		double y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
		gp->add_pattern(x, y);
		gpnaive->add_pattern(x, y);
	}

//	testSpeedOfLogLikelihood();
//	testSpeedOfCholeskyComputation2();
	Eigen::VectorXd grad = gpnaive->log_likelihood_gradient();
	grad.array()=(grad.array()-gp->log_likelihood_gradient().array())/grad.array();
	std::cout << grad << std::endl;
	assert(grad.array().abs().maxCoeff() < 1e-5);
	compare_time(llhGradBaseline, llhGradFast, 15);

	delete gp;
	delete gpnaive;
	return 0;
}
