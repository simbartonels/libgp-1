#include <cmath>

#include "gp.h"
#include "gp_utils.h"
#include "util/time_call.h"



using namespace libgp;
GaussianProcess * gp;

    void measureBFcomputationTime() {
    	size_t num_execs = 100;
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
    		gp = new GaussianProcess(D, "CovSum ( CovSEard, CovNoise)");
    		for (int i = 0; i < M; ++i) {
    			gp->add_pattern(X.row(i), y(i));
    		}
    		Eigen::VectorXd params(gp->covf().get_param_dim());
    		params.setRandom();
    		gp->covf().set_loghyper(params);
    		gp->log_likelihood();
    		std::cout << "done" << std::endl;
    		double min = -log(0.0);
    		for(size_t j = 0; j < num_execs; j++){
				stop_watch();
				for (size_t i = 0; i < num_execs; i++) {
					gp->f(x);
					gp->var(x);
					x(0) = -x(0);
				}
	    		double tic = stop_watch() / num_execs;
	    		if(tic < min)
				min = tic;
	    		std::cout << j << "/" << num_execs << ": " << min << std::endl;
    		}
    		delete gp;
    	}
    }

int main(int argc, char const *argv[]) {
	measureBFcomputationTime();
	return 0;
}
