#include "util/time_call.h"
#include <Eigen/Dense>

Eigen::MatrixXd Mat1;
Eigen::MatrixXd Mat2;
Eigen::MatrixXd Target;

void mult1(){
	Target = Mat1*Mat1.transpose();
}

void mult1_alternative(){
	size_t n = 500;
	for(size_t i=0; i < n; i++){
		for(size_t j=0; j < i; j++){
			Target(i, j) = Mat1.row(i) * Mat1.row(j).transpose();
			Target(j, i) = Target(i, j);
		}
		Target(i, i) = Mat1.row(i).norm();
	}
}

int main(int argc, char const *argv[]) {
	Mat1.resize(500, 3000);
	Mat1.setRandom();
//	Mat2.resize(3000, 500);
//	Mat2.setRandom();
	Target.resize(500, 500);
	compare_time(mult1, mult1_alternative, 2);
}



