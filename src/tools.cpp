#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
			      const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  
  // check for errors in estimations or ground truth vector size
  if (estimations.size() == 0 || (estimations.size() != ground_truth.size())) {
    std::cout << "Error";
    return rmse;
  }

  // calculate the residuals
  for (int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the square root
  rmse = rmse.array().sqrt();

  return rmse;
}
