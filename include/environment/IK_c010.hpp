//
// Created by joonho on 10.06.19.
//

#ifndef OLD_RAI_IK_HPP
#define OLD_RAI_IK_HPP

#include <Eigen/Core>

class InverseKinematics {

 public:
  InverseKinematics() {

    ///Hard code all the shits
    PositionBaseToHipInBaseFrame.resize(4);
    positionHipToThighInHipFrame.resize(4);
    positionThighToShankInThighFrame.resize(4);
    positionShankToFootInShankFrame.resize(4);
    PositionBaseToHAACenterInBaseFrame.resize(4);

    PositionBaseToHipInBaseFrame[0] << 0.3, 0.104, 0.0;
    PositionBaseToHipInBaseFrame[1] << 0.3, -0.104, 0.0;
    PositionBaseToHipInBaseFrame[2] << -0.3, 0.104, 0.0;
    PositionBaseToHipInBaseFrame[3] << -0.3, -0.104, 0.0;

    positionHipToThighInHipFrame[0] << 0.06, 0.0701, -0.0;
    positionHipToThighInHipFrame[1] << 0.06, -0.0701, -0.0;
    positionHipToThighInHipFrame[2] << -0.06, 0.0701, -0.0;
    positionHipToThighInHipFrame[3] << -0.06, -0.0701, -0.0;

    positionThighToShankInThighFrame[0] << 0.0, 0.086887, -0.282627;
    positionThighToShankInThighFrame[1] << 0.0, -0.086887, -0.282627;
    positionThighToShankInThighFrame[2] << 0.0, 0.086887, -0.282627;
    positionThighToShankInThighFrame[3] << 0.0, -0.086887, -0.282627;

    positionShankToFootInShankFrame[0] << 0.08748, -0.006, -0.342504;
    positionShankToFootInShankFrame[1] << 0.08748, 0.006, -0.342504;
    positionShankToFootInShankFrame[2] << -0.08748, -0.006, -0.342504;
    positionShankToFootInShankFrame[3] << -0.08748, 0.006, -0.342504;

    for (size_t i = 0; i < 4; i++) {
      PositionBaseToHAACenterInBaseFrame[i] = PositionBaseToHipInBaseFrame[i];
      PositionBaseToHAACenterInBaseFrame[i][0] += positionHipToThighInHipFrame[i][0];
      hfe_to_foot_y_offset_[i] = positionThighToShankInThighFrame[i][1];
      hfe_to_foot_y_offset_[i] += positionShankToFootInShankFrame[i][1];

      haa_to_foot_y_offset_[i] = hfe_to_foot_y_offset_[i];
      haa_to_foot_y_offset_[i] += positionHipToThighInHipFrame[i][1];

    }

    a0_ = std::sqrt(positionHipToThighInHipFrame[0][1] * positionHipToThighInHipFrame[0][1]
                        + positionHipToThighInHipFrame[0][2] * positionHipToThighInHipFrame[0][2]);
    haa_offset_ = std::abs(std::atan2(positionHipToThighInHipFrame[0][2], positionHipToThighInHipFrame[0][1]));

    a1_squared_ = positionThighToShankInThighFrame[0][0] * positionThighToShankInThighFrame[0][0]
        + positionThighToShankInThighFrame[0][2] * positionThighToShankInThighFrame[0][2];
    a2_squared_ = positionShankToFootInShankFrame[0][0] * positionShankToFootInShankFrame[0][0]
        + positionShankToFootInShankFrame[0][2] * positionShankToFootInShankFrame[0][2];

    minReach_SP = std::abs(sqrt(a1_squared_) - sqrt(a2_squared_)) + 0.1;
    maxReach_SP = sqrt(a1_squared_) + sqrt(a2_squared_) - 0.05;
//    maxReach_SP = sqrt(a1_squared_) + sqrt(a2_squared_) - std::abs(positionHipToThighInHipFrame[0][1]);
    minReach = std::sqrt(haa_to_foot_y_offset_[0] * haa_to_foot_y_offset_[0] + minReach_SP * minReach_SP);
    maxReach = sqrt(haa_to_foot_y_offset_[0] * haa_to_foot_y_offset_[0] + maxReach_SP * maxReach_SP);
    KFEOffset_ = std::abs(std::atan(positionShankToFootInShankFrame[0][0] / positionShankToFootInShankFrame[0][2]));
  }

  ~InverseKinematics() = default;

  inline bool IKSagittal(
      Eigen::Vector3d &legJoints,
      const Eigen::Vector3d &positionBaseToFootInBaseFrame,
      size_t limb) {

    Eigen::Vector3d
        positionHAAToFootInBaseFrame = positionBaseToFootInBaseFrame - PositionBaseToHAACenterInBaseFrame[limb];

    const double d = haa_to_foot_y_offset_[limb];
    const double dSquared = d * d;

    ///Rescaling target
    double reach = positionHAAToFootInBaseFrame.norm();
    if (reach > maxReach) {
      positionHAAToFootInBaseFrame /= reach;
      positionHAAToFootInBaseFrame *= maxReach;
    } else if (reach < minReach) {
      positionHAAToFootInBaseFrame /= reach;
      positionHAAToFootInBaseFrame *= minReach;
    }
    double positionYzSquared = positionHAAToFootInBaseFrame.tail(2).squaredNorm();

    if (positionYzSquared < dSquared) {

      positionHAAToFootInBaseFrame.tail(2) /= std::sqrt(positionYzSquared);
      positionHAAToFootInBaseFrame.tail(2) *= (std::abs(d) + 0.01);

      if (positionHAAToFootInBaseFrame[0] > maxReach_SP) {
        positionHAAToFootInBaseFrame[0] /= std::abs(positionHAAToFootInBaseFrame[0]);
        positionHAAToFootInBaseFrame[0] *= maxReach_SP;
      }
      positionYzSquared = positionHAAToFootInBaseFrame.tail(2).squaredNorm();
    }

    //compute HAA angle
    double rSquared = positionYzSquared - dSquared;
    const double r = std::sqrt(rSquared);
    const double delta = std::atan2(positionHAAToFootInBaseFrame.y(),
                                    -positionHAAToFootInBaseFrame.z());
    const double beta = std::atan2(r, d);
    const double qHAA = beta + delta - M_PI_2;
    legJoints[0] = qHAA;

    ///simplification for anymal
    const double l_squared = (rSquared + positionHAAToFootInBaseFrame[0] * positionHAAToFootInBaseFrame[0]);
    const double phi1 = std::acos((a1_squared_ + l_squared - a2_squared_) * 0.5 / sqrt(a1_squared_ * l_squared));
    const double phi2 = std::acos((a2_squared_ + l_squared - a1_squared_) * 0.5 / sqrt(a2_squared_ * l_squared));

    double qKFE = phi1 + phi2 - KFEOffset_;

    if (limb < 2) {
      qKFE *= -1.0;
    }
    legJoints[2] = qKFE;

    double theta_prime = atan2(positionHAAToFootInBaseFrame[0], r);
    double qHFE = phi1 - theta_prime;

    if (limb > 1) {
      qHFE = -phi1 - theta_prime;
    }
    legJoints[1] = qHFE;
    return true;
  }

  double aKFE_[4];
  double bKFE_[4];
  double cKFE_[4];
  double haa_to_foot_y_offset_[4];
  double hfe_to_foot_y_offset_[4];

  double a0_;
  double haa_offset_;

  double a1_squared_;
  double a2_squared_;
  double KFEOffset_;
  double minReach;
  double maxReach;
  double minReach_SP;
  double maxReach_SP;

  std::vector<Eigen::Vector3d> positionHipToThighInHipFrame;
  std::vector<Eigen::Vector3d> positionThighToShankInThighFrame;
  std::vector<Eigen::Vector3d> positionShankToFootInShankFrame;
  std::vector<Eigen::Vector3d> PositionBaseToHipInBaseFrame;
  std::vector<Eigen::Vector3d> PositionBaseToHAACenterInBaseFrame;

};

#endif //OLD_RAI_IK_HPP
