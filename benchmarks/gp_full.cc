#include <cmath>

#include "gp.h"
#include "gp_utils.h"
#include "util/time_call.h"



using namespace libgp;
GaussianProcess * gp;

    void measureBFcomputationTime() {
    	size_t D = 2;
    	size_t n = 2000;
    	size_t num_execs = 100;
    	std::cout << "D is " << D << std::endl;
    	Eigen::VectorXd grad(D);
    	Eigen::VectorXd x(D);
    	x.setRandom();
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
    		gp->log_likelihood();
    		std::cout << "done" << std::endl;
    		stop_watch();
    		for (size_t i = 0; i < num_execs; i++) {
    			gp->f(x);
    			gp->var(x);
    		}
    		double tic = stop_watch() / num_execs;
    		std::cout << tic << std::endl;
    	}
    }

int main(int argc, char const *argv[]) {
	measureBFcomputationTime();
	return 0;
}
