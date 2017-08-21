#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() { 
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 13;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
  Complete the initialization. 
  */

  // intialise to false 
  is_initialized_ = false;

  // initial state and augmented dimension
  n_x_ = 5;
  n_aug_ = 7;

  // initialize the lambda factor
  lambda_ = 3 - n_x_;

  // predicted sigma points
  n_sig_ = 2 * n_aug_ + 1;
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  
  // initialize weights
  weights_ = VectorXd(n_sig_);

  P_ << 1, 0, 0,    0,    0,
        0, 1, 0,    0,    0,
        0, 0, 1000, 0,    0,
        0, 0, 0,    1,    0,
        0, 0, 0,    0,    1; 

  previous_timestamp_ = 0;
}

UKF::~UKF() {}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**

  Complete this function
  */

  long long timestamp = meas_package.timestamp_;
  
  if (!is_initialized_) {
    previous_timestamp_ = timestamp;
    
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      initialize and convert polar to cartesian coordinates.
       */

	  float rho = meas_package.raw_measurements_(0);
	  float phi = meas_package.raw_measurements_(1);
	  float rho_dot = meas_package.raw_measurements_(2);
	  
	  float px = rho * cos(phi);
	  float py = rho * sin(phi);
	  float v = rho_dot;
      
     	  x_ << px, py, v, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
       initialize state.
       */
      x_ << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1),
            0,
            0,
            0;
    }
    
    is_initialized_ = true;
    return;
  }
  

  // calculate time delta.
  double dt = (timestamp - previous_timestamp_) / 1000000.;  
  previous_timestamp_ = timestamp;
  
  Prediction(dt);


  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  } 
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // generate sigma points section
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);

  x_aug.fill(0);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  MatrixXd Q = MatrixXd(2, 2);
  Q << pow(std_a_, 2), 0,
       0, pow(std_yawdd_, 2);
   
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  
  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) 
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  
  // predict sigma points

  for (int i = 0; i< n_sig_; i++)
  {

    VectorXd sigpt = Xsig_aug.col(i);

    // extract the values for reference purposes in next steps
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);


    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma points
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;

  }

  // set weights
  for (int i = 0; i < n_sig_; i++) {
    if (i == 0) {
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    } else {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }
  }  

  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) -= (2 * M_PI) * floor((x_diff(3) + M_PI) / (2 * M_PI));
    
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  
  MatrixXd z = meas_package.raw_measurements_;
  int n_z = 2;
  
  MatrixXd H = MatrixXd(n_z, n_x_);
  MatrixXd R = MatrixXd(n_z, n_z);

  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
  R << pow(std_laspx_, 2), 0,
       0, std_laspy_;
  
  MatrixXd Ht = H.transpose();
  VectorXd z_pred = H * x_;
  VectorXd z_diff = z - z_pred;
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  
  // new estimate
  x_ = x_ + K * z_diff;
  long size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  P_ = (I - K * H) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the radar NIS.
  */
  
  int n_z = 3;
  
  // measurement sigma points matrix
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  // mean prediction
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd z = meas_package.raw_measurements_;
  
  z_pred.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    // transform sigma points into measurement space
    VectorXd sigpt = Xsig_pred_.col(i);
    double px   = sigpt(0);
    double py   = sigpt(1);
    double v    = sigpt(2);
    double yaw  = sigpt(3);

    // no div by 0
    float eps = 0.00001;
    if (fabs(px) < eps) {
      px = eps;
    }
    if (fabs(py) < eps) {
      py = eps;
    }

    // measurement model
    double rho = sqrt((px*px) + (py*py));
    double phi = atan2(py, px);
    double r_dot = v * (px * cos(yaw) + py * sin(yaw)) / rho;
    
    Zsig.col(i) << rho, phi, r_dot;
    
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  // measurement covariance matrix S

  // radar noise covariance
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radr_*std_radr_;

  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    // residual
    VectorXd z_diff_residual = Zsig.col(i) - z_pred;

    // angle normalization
    S += weights_(i) * z_diff_residual * z_diff_residual.transpose();
  }

  // add noise
  S += R;

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    
    // state difference
    MatrixXd xdiff_ = Xsig_pred_.col(i) - x_;

    // residual
    MatrixXd zdiff = Zsig.col(i) - z_pred;

    // angle normalization
    while (xdiff_(3)> M_PI) xdiff_(3) -= 2.*M_PI;
    while (xdiff_(3)<-M_PI) xdiff_(3) += 2.*M_PI;

    Tc += weights_(i) * xdiff_ * zdiff.transpose();
  }
  
  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff_residual2 = z - z_pred;

  //angle normalization
  while (z_diff_residual2(1)> M_PI) z_diff_residual2(1) -= 2.*M_PI;
  while (z_diff_residual2(1)<-M_PI) z_diff_residual2(1) += 2.*M_PI;
  
  // update state mean and covariance matrix
  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();
}
