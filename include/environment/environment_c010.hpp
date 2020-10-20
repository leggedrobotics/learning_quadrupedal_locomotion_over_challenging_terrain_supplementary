#pragma once

// system inclusion
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h>

#include <common/SimpleMLPLayer.hpp>
#include <common/math.hpp>
#include <common/RandomNumberGenerator.hpp>
#include <common/message_macros.hpp>
#include "IK_c010.hpp"

// simulator
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/OgreVis.hpp"

//visualizer
#include "visualizer/visSetupCallback.hpp"
#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/guiState.hpp"

namespace Env {
enum TerrainType {
  Flat_,
  Hills,
  Steps,
  Stairs,
  SingleStep,
  UniformSlope
};

enum ActionType {
  EE = 0,
  JOINT
};

enum CommandMode {
  RANDOM = 0,
  FIXED_DIR,
  STRAIGHT,
  STOP,
  ZERO,
  NOZERO
};

constexpr int sampleN = 36;
constexpr int ObservationDim = 60;
constexpr int ActionDim = 16;
constexpr int StateDim = 133;
constexpr int PrivilegedStateDim = StateDim + sampleN + 4 + 12 + 12 + 4 + 8 + 3; //
constexpr int JointHistoryLength = 128;

class blind_locomotion {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<double, 19, 1> GeneralizedCoordinate;
  typedef Eigen::Matrix<double, 18, 1> GeneralizedVelocities;

  typedef Eigen::Matrix<double, 12, 1> Vector12d;
  typedef Eigen::Matrix<double, 18, 1> Vector18d;

  typedef Eigen::Matrix<float, StateDim, 1> State;
  typedef Eigen::Matrix<float, ActionDim, 1> Action;
  typedef Eigen::Matrix<float, ObservationDim, 1> Observation;

  blind_locomotion() = delete;

  explicit blind_locomotion(bool visualize = false,
                            int instance = 0,
                            std::string urdf_path = "",
                            std::string actuator_path = "") :
      vis_on_(visualize),
      vis_ready_(false),
      vid_on_(false),
      actuator_A_({32, 32}),
      instance_(instance) {

    actuator_A_.updateParamFromTxt(actuator_path);

    control_dt_ = 0.01;
    terrainparams_.setZero();

    footPositionOffset_ <<
                        0.3 + 0.1, 0.2, h0_,
        0.3 + 0.1, -0.2, h0_,
        -0.3 - 0.1, 0.2, h0_,
        -0.3 - 0.1, -0.2, h0_;

    Eigen::Vector3d sol;
    for (int i = 0; i < 4; i++) {
      IK_.IKSagittal(sol, footPositionOffset_.segment(3 * i, 3).template cast<double>(), i);
      jointNominalConfig_.segment(3 * i, 3) = sol;
    }

    q_.setZero(19);
    u_.setZero(18);

    q0.resize(19);
    q0 << 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, jointNominalConfig_;
    u0.setZero(18);

    q_initialNoiseScale.setZero(19);
    u_initialNoiseScale.setZero(18);

    command_.setZero();
    goalPosition_[0] = 0.0;
    goalPosition_[1] = 0.0;
    disturbance_ = {0.0, 0.0, 0.0};
    tau_.setZero();

    observationOffset_ <<
                       Eigen::VectorXf::Constant(3, 0.0),
        0.0, 0.0, 1.0, /// gravity axis
        Eigen::VectorXf::Constant(6, 0.0),
        jointNominalConfig_.template cast<float>(),
        Eigen::VectorXf::Constant(12, 0.0),
        Eigen::VectorXf::Constant(12, 0.0),
        Eigen::VectorXf::Constant(8, 0.0),
        Eigen::VectorXf::Constant(4, 0.0);

    observationScale_ <<
                      1.5, 1.5, 1.5, /// command
        5.0, 5.0, 5.0, /// gravity axis
        Eigen::VectorXf::Constant(3, 2.0),
        Eigen::VectorXf::Constant(3, 2.0),
        Eigen::VectorXf::Constant(12, 2.0), /// joint position
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        6.5, 4.5, 3.5,
        6.5, 4.5, 3.5,
        6.5, 4.5, 3.5,
        6.5, 4.5, 3.5,
        Eigen::VectorXf::Constant(8, 1.5),
        Eigen::VectorXf::Constant(4, 2.0 / freqScale_);

    /// state params
    stateOffset_ << 0.0, 0.0, 0.0, /// command
        0.0, 0.0, 1.0, /// gravity axis
        Eigen::VectorXf::Constant(6, 0.0), /// body lin/ang vel
        jointNominalConfig_.template cast<float>(), /// joint position
        Eigen::VectorXf::Constant(12, 0.0),
        Eigen::VectorXf::Constant(12, 0.0),
        Eigen::VectorXf::Constant(4, 0.0), //52
        Eigen::VectorXf::Constant(8, 0.0), // 60
        Eigen::VectorXf::Constant(24, 0.0), // 84
        Eigen::VectorXf::Constant(24, 0.0), // 108
        jointNominalConfig_.template cast<float>(), /// joint position
        jointNominalConfig_.template cast<float>(), /// joint position
        0.0; // 132

    stateScale_ << 1.5, 1.5, 1.5, /// command
        5.0, 5.0, 5.0, /// gravity axis
        Eigen::VectorXf::Constant(3, 2.0),
        Eigen::VectorXf::Constant(3, 2.0),
        Eigen::VectorXf::Constant(12, 2.0), /// joint angles
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        6.5, 4.5, 3.5,
        6.5, 4.5, 3.5,
        6.5, 4.5, 3.5,
        6.5, 4.5, 3.5,
        Eigen::VectorXf::Constant(4, 2.0 / freqScale_),
        Eigen::VectorXf::Constant(8, 1.5),
        Eigen::VectorXf::Constant(24, 5.0), /// joint position errors
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        0.5, 0.4, 0.3,
        Eigen::VectorXf::Constant(12, 2.0), /// prev. action
        Eigen::VectorXf::Constant(12, 2.0),
        2.0 / freqScale_;

    double scale1 = 0.1, scale2 = 0.025;
    /// action params
    actionScale_ <<
                 Eigen::VectorXf::Constant(4, 0.5 * freqScale_),
        Eigen::VectorXf::Constant(2, scale1), scale2,
        Eigen::VectorXf::Constant(2, scale1), scale2,
        Eigen::VectorXf::Constant(2, scale1), scale2,
        Eigen::VectorXf::Constant(2, scale1), scale2;

    actionOffset_ << Eigen::VectorXf::Constant(16, 0.0);

    for (size_t i = 0; i < 4; i++) {
      clearance_[i] = 0.2;
    }

    previousAction_ = actionOffset_.template cast<float>();
    /// env setup & visualization
    realTimeRatio_ = 1.0;
    env_ = new raisim::World;

    if (vis_on_) {
      auto vis = raisim::OgreVis::get();
      vis->setWorld(env_);
      vis->setWindowSize(650, 800);

      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);

      /// starts visualizer thread
      vis->initApp();
      vis->setDesiredFPS(30.0);
    }

    anymal_ = env_->addArticulatedSystem(urdf_path);
    env_->setTimeStep(simulation_dt_);
    gravity_.e() << 0, 0, -9.81;
    env_->setGravity(gravity_);

    /// terrain
    terrainProp_.xSize = 10.0;
    terrainProp_.ySize = 10.0;
    terrainProp_.xSamples = terrainProp_.xSize / gridSize_;
    terrainProp_.ySamples = terrainProp_.ySize / gridSize_;

    terrainProp_.fractalOctaves = 1;
    terrainProp_.frequency = 0.2; ///
    terrainProp_.frequency = 0.2; ///
    terrainProp_.fractalLacunarity = 3.0;
    terrainProp_.fractalGain = 0.1;
    terrainGenerator_.getTerrainProp() = terrainProp_;

    board_ = env_->addGround(0.0, "terrain");
    heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples, 0.0);

    /// env visual
    if (vis_on_) {
      auto vis = raisim::OgreVis::get();
      anymalVisual_ = vis->createGraphicalObject(anymal_, "ANYmal");
      auto groundVisual = vis->createGraphicalObject(board_, 20, "ground", "white_smoke");

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-1.2), Ogre::Radian(-1.35), 4.0, true);
    }

    for (int i = 0; i < anymalVisual_->size(); i++) {
      WARN(anymalVisual_->at(i).graphics->getAttachedObject(0)->getName())
    }

    /// task options
    steps_ = 0;
    command_ << 0.0, 0.0, 0.0;
    commandDirection_ = 0.0;

    uPrev_.setZero(18);
    acc_.setZero(18);
    netFootContacts_.resize(4);
    netFootContacts_b.resize(4);
    netFootContactVels_.resize(4);
    FootContactNums_.resize(4);

    /// noise peoperty
    q_initialNoiseScale.setConstant(0.4);
    q_initialNoiseScale.segment(0, 3) << 0.00, 0.00, 0.03;

    u_initialNoiseScale.setConstant(2.0);
    u_initialNoiseScale.segment(3, 3).setConstant(1.);
    u_initialNoiseScale.head(3).setConstant(0.4);

    numContact_ = 0;
    jointVelHist_.setZero();
    jointPosHist_.setZero();
    torqueHist_.setZero();

    for (int i = 0; i < 4; i++) {
      footPos_.push_back(anymal_->getCollisionBodies()[4 * i + 5].posOffset);
      footR_[i] = anymal_->getVisColOb()[4 * i + 5].visShapeParam[0];
      footNames_[i] = "foot";
      footNames_[i] += std::to_string(i);
      anymal_->getCollisionBodies()[4 * i + 4].setMaterial(footNames_[i]);
      env_->setMaterialPairProp("terrain", footNames_[i], 0.9, 0.0, 0.0);
    }
    env_->setERP(0.1, 0.0);

    footAngVel_W.resize(4);
    footPos_W.resize(4);
    footPos_b.resize(4);
    footPos_b_prev.resize(4);
    footVel_W.resize(4);
    footVel_projected.resize(4);
    footPos_Target.resize(4);
    prevfootPos_Target.resize(4);
    prevfootPos_Target2.resize(4);

    footNormal_.resize(4);
    footNormal_b.resize(4);
    badlyConditioned_ = false;
    footContactPositions_W_.resize(4);
    footContactPositions_W_Noisy.resize(4);
    footNormals_W_.resize(4);

    /// initialize ANYmal
    anymal_->setGeneralizedCoordinate(q0);
    anymal_->setGeneralizedVelocity(u0);
    anymal_->setGeneralizedForce(tau_);

    /// collect joint positions, collision geometry
    defaultJointPositions_.resize(13);
    defaultBodyMasses_.resize(13);

    for (int i = 0; i < 13; i++) {
      defaultJointPositions_[i] = anymal_->getJointPos_P()[i].e();
      defaultBodyMasses_[i] = anymal_->getMass(i);
    }

    int cnt = 0;
    COMPosition_ = anymal_->getLinkCOM()[0].e();
  }

  ~blind_locomotion() {
    if (vis_on_) {
      raisim::OgreVis::get()->closeApp();
    }
    delete env_;
  }

  inline void comprehendContacts() {
    numContact_ = anymal_->getContacts().size();
    numFootContact_ = 0;
    numBaseContact_ = 0;
    numShankContact_ = 0;
    numThighContact_ = 0;
    numInternalContact_ = 0;

    for (int k = 0; k < 4; k++) {
      footContactState_[k] = false;
      shankContacts_[k] = false;
      thighContacts_[k] = false;

      netFootContacts_[k].setZero();
      netFootContacts_b[k].setZero();
      netFootContactVels_[k].setZero();
      FootContactNums_[k] = 0;
      footNormal_[k] << 0.0, 0.0, 1.0;
      footNormal_b[k] << 0.0, 0.0, 1.0;
    }
    netContacts_.setZero();

    raisim::Vec<3> vec3;

    //position of the feet
    for (int k = 0; k < 4; k++) {
      int footID = 3 * k + 3;
      anymal_->getPosition(footID, footPos_[k], footPos_W[k]);
      anymal_->getVelocity(footID, footPos_[k], footVel_W[k]);
      anymal_->getAngularVelocity(footID, footAngVel_W[k]);
    }
    anymal_->getPosition(0, vec3);

    //Classify foot contact
    if (numContact_ > 0) {
      for (int k = 0; k < numContact_; k++) {
        if (!anymal_->getContacts()[k].skip() && anymal_->getContacts()[k].getPairObjectIndex() > 0) {

          int idx = anymal_->getContacts()[k].getlocalBodyIndex();
          netContacts_ += anymal_->getContacts()[k].getImpulse()->e();
          if (idx == 0) {
            numBaseContact_++;
          } else if (idx == 3 || idx == 6 || idx == 9 || idx == 12) {
            int fid = idx / 3 - 1;
            double err = (footPos_W[fid].e() - anymal_->getContacts()[k].getPosition().e()).norm();

            if (err < 0.035) {
              netFootContacts_[fid] +=
                  (anymal_->getContacts()[k].getContactFrame().e() * anymal_->getContacts()[k].getImpulse()->e());

              anymal_->getContactPointVel(k, vec3);
              netFootContactVels_[fid] += vec3.e();
              footNormal_[fid] += anymal_->getContacts()[k].getNormal().e();
              FootContactNums_[fid]++;
              footContactState_[fid] = true;
              numFootContact_++;
            } else {
              numShankContact_++;
              shankContacts_[fid] = true;
            }

          } else if (idx == 2 || idx == 5 || idx == 8 || idx == 11) {
            int fid = idx / 3;
            numThighContact_++;
            thighContacts_[fid] = true;
          }
        } else {
          numInternalContact_++;
        }
      }
    }

    netContacts_ /= simulation_dt_;
    netContacts_b = R_b_.transpose() * netContacts_;

    for (size_t i = 0; i < 4; i++) {
      if (FootContactNums_[i] > 0) {
        netFootContactVels_[i] /= FootContactNums_[i];
        netFootContacts_[i] /= simulation_dt_;
        netFootContacts_[i] = netFootContacts_[i].array().min(200.0); // For stability
        netFootContacts_[i] = netFootContacts_[i].array().max(-200.0); // For stability
        footNormal_[i].normalize();
        footNormal_b[i] = R_b_.transpose() * footNormal_[i];
        netFootContacts_b[i] = R_b_.transpose() * netFootContacts_[i];
        double scale = footNormal_[i].dot(footVel_W[i].e());
        footVel_projected[i] = footVel_W[i].e() - scale * footNormal_[i];
      } else {
        footVel_projected[i].setZero();
      }
    }

    double headingAngle = std::atan2(R_b_.col(0)[1], R_b_.col(0)[0]);

    Eigen::Vector2d x_, y_;
    x_ = footMargin_ * (R_b_ * xHorizontal_).head(2);
    y_ = footMargin_ * (R_b_ * yHorizontal_).head(2);

    double Angle = 2.0 * M_PI / (double) (Fdxs_.size() - 1);
    Fdxs_.setConstant(0.0);
    Fdys_.setConstant(0.0);

    for (size_t i = 1; i < Fdxs_.size(); i++) {
      const float Cos = std::cos((i - 1) * Angle);
      const float Sin = std::sin((i - 1) * Angle);
      Fdxs_.data()[i] += (Cos * x_[0]);
      Fdxs_.data()[i] += (Sin * y_[0]);

      Fdys_.data()[i] += (Cos * x_[1]);
      Fdys_.data()[i] += (Sin * y_[1]);
    }

    if (terrainType_ != Flat_) {
      int dataID = 0, dataID2 = 0;
      for (size_t fid = 0; fid < 4; fid++) {
        for (size_t k = 0; k < 9; k++) {
          FHs_.data()[dataID++] =
              terrain_->getHeight(footPos_W[fid][0] + Fdxs_.data()[k], footPos_W[fid][1] + Fdys_.data()[k]);
        }
      }
    } else {
      FHs_.setZero();
    }
  }

  void updateHistory() {
    Eigen::Matrix<double, 4, 1> quat = q_.template segment<4>(3);
    Eigen::Matrix<double, 3, 3> R_b = Math::MathFunc::quatToRotMat(quat);

    Eigen::Matrix<double, 4, 1> quat2;
    quat2 = Math::MathFunc::boxplusI_Frame(quat, e_g_bias_);
    Eigen::Matrix<double, 3, 3> R_b2 = Math::MathFunc::quatToRotMat(quat2);

    Eigen::Matrix<double, 3, 1> bodyVel = R_b2.transpose() * u_.template segment<3>(0);
    Eigen::Matrix<double, 3, 1> bodyAngVel = R_b2.transpose() * u_.template segment<3>(3);
    Eigen::Matrix<double, 12, 1> jointVel = u_.template segment<12>(6);

    Observation observation_unscaled;
    Observation observation_scaled;

    observation_unscaled.head(3) = command_.template cast<float>();
    observation_unscaled.template segment<3>(3) = R_b2.row(2).transpose().template cast<float>();
    observation_unscaled.template segment<3>(6) = bodyVel.template cast<float>();
    observation_unscaled.template segment<3>(9) = bodyAngVel.template cast<float>();
    observation_unscaled.template segment<12>(12) = q_.tail(12).template cast<float>(); /// position
    observation_unscaled.template segment<12>(24) = jointVel.template cast<float>();
    observation_unscaled.template segment<12>(36) = (jointPositionTarget_ - q_.tail(12)).template cast<float>();

    for (size_t i = 0; i < 4; i++) {
      observation_unscaled[48 + 2 * i] = std::sin(pi_[i]);
      observation_unscaled[49 + 2 * i] = std::cos(pi_[i]);
      observation_unscaled[56 + i] = piD_[i];
    }

    /// noisify body vel
    for (int i = 6; i < 9; i++)
      observation_unscaled[i] += rn_.sampleUniform() * 0.05;

    /// noisify body angvel
    for (int i = 9; i < 12; i++)
      observation_unscaled[i] += rn_.sampleUniform() * 0.1;

    /// noisify joint position
    for (int i = 12; i < 24; i++)
      observation_unscaled[i] += rn_.sampleUniform() * 0.01;

    /// noisify joint vel
    for (int i = 24; i < 36; i++)
      observation_unscaled[i] += rn_.sampleUniform() * 1.;

    /// noisify joint position
    for (int i = 36; i < 48; i++)
      observation_unscaled[i] += rn_.sampleUniform() * 0.01;

    observation_scaled = (observation_unscaled - observationOffset_).cwiseProduct(observationScale_);

    Eigen::Matrix<float, -1, -1> temp = historyBuffer_;
    historyBuffer_.block(0, 1, ObservationDim, JointHistoryLength - 1) =
        temp.block(0, 0, ObservationDim, JointHistoryLength - 1);
    historyBuffer_.col(0) = observation_scaled;
  }

  void getHistory(Eigen::Matrix<float, -1, -1> &out, const size_t &nums) {
    out = historyBuffer_.block(0, 0, ObservationDim, nums);
  };

  virtual void integrate() {
    bool terminate = false;
    size_t decimation = (size_t) (control_dt_ / simulation_dt_);
    double stepSize = 1.0 / decimation;

    Eigen::Vector3d sol;
    Eigen::Vector3d target;
    Eigen::Vector3d heightOffset;
    sol.setZero();

    double a = 0.5;
    for (size_t i = 0; i < decimation; i++) {
      if (controlCounter_ == decimation) {
        controlCounter_ = 0;
      }

      for (size_t j = 0; j < 4; j++) {
        ///Interpolation
        target = footPos_Target[j];
        IK_.IKSagittal(sol, target, j);
        jointPositionTarget_.segment<3>(3 * j) = sol;
      }

      Eigen::Matrix<float, JointHistoryLength * 12 - 12, 1> temp;
      temp = jointVelHist_.tail(JointHistoryLength * 12 - 12);
      jointVelHist_.head(JointHistoryLength * 12 - 12) = temp;
      jointVelHist_.tail(12) = u_.tail(12).template cast<float>();

      temp = jointPosHist_.tail(JointHistoryLength * 12 - 12);
      jointPosHist_.head(JointHistoryLength * 12 - 12) = temp;
      jointPosHist_.tail(12) = (jointPositionTarget_ - q_.tail(12)).template cast<float>();

      Eigen::Matrix<double, 6, 1> seaInput;
      Eigen::Matrix<double, 8, 1> seaInput2;

      for (int actId = 0; actId < 12; actId++) {
        seaInput[0] = (jointVelHist_(actId + (JointHistoryLength - 9) * 12)) * 0.4;
        seaInput[1] = (jointVelHist_(actId + (JointHistoryLength - 4) * 12)) * 0.4;
        seaInput[2] = (jointVelHist_(actId + (JointHistoryLength - 1) * 12)) * 0.4;
        seaInput[3] = (jointPosHist_(actId + (JointHistoryLength - 9) * 12)) * 3.0;
        seaInput[4] = (jointPosHist_(actId + (JointHistoryLength - 4) * 12)) * 3.0;
        seaInput[5] = (jointPosHist_(actId + (JointHistoryLength - 1) * 12)) * 3.0;
        tau_(6 + actId) = actuator_A_.forward(seaInput)[0] * 20.0;
      }

      tau_.head(6).setZero();
      integrateOneTimeStep();

      if (steps_ % ObservationStride_ == 0) {
        updateHistory();
      }

      t_ += dt_;
      steps_++;
      controlCounter_++;
    }
    updateVisual();
  }

  void setActionType(ActionType in) {
    actionType_ = in;
    if (actionType_ == ActionType::EE) {
      double amp1 = 0.5 * freqScale_;
      double scale0 = 0.03, scale1 = 0.1, scale2 = 0.025;

      actionScale_ <<
                   amp1, amp1, amp1, amp1,//    footPositionOffset_ <<
          Eigen::Matrix<float, -1, 1>::Constant(2, scale1), scale2,
          Eigen::Matrix<float, -1, 1>::Constant(2, scale1), scale2,
          Eigen::Matrix<float, -1, 1>::Constant(2, scale1), scale2,
          Eigen::Matrix<float, -1, 1>::Constant(2, scale1), scale2;
    } else {
      double amp1 = 0.5 * freqScale_;

      actionScale_ <<
                   amp1, amp1, amp1, amp1,//    footPositionOffset_ <<
          Eigen::Matrix<float, -1, 1>::Constant(12, 0.2);
    }
  }

  void setCommandMode(CommandMode in) {
    commandMode_ = in;
  }

  void setFootFriction(int idx, double c_f) {
    footFriction_[idx] = c_f;
    env_->setMaterialPairProp("terrain", footNames_[idx], footFriction_[idx], 0.0, 0.0);
  }

  inline void updateVisual() {
    if (vis_on_) {
      auto vis = raisim::OgreVis::get();
      /// visualize contact


      for (int i = 0; i < 4; i++) {
        auto foot0 = vis->getSceneManager()->getEntity(
            anymalVisual_->at(26 + 8 * i).graphics->getAttachedObject(0)->getName());
        auto foot1 = vis->getSceneManager()->getEntity(
            anymalVisual_->at(25 + 8 * i).graphics->getAttachedObject(0)->getName());

        if (footContactState_[i]) {
          foot0->setMaterialName("blueEmit");
          foot1->setMaterialName("blueEmit");
        } else {
          foot0->setMaterialName("black");
          foot1->setMaterialName("black");
        }
//
//        if(footFriction_[i] < 0.5){
//          foot0->setMaterialName("green");
//          foot1->setMaterialName("green");
//        }
      }

      /// info
      std::ostringstream out;
      out << std::fixed << std::setprecision(2);

      Eigen::Vector3d linearSpeed = (R_b_.transpose() * u_.segment<3>(0));
      Eigen::Vector3d bodyAngVel = R_b_.transpose() * u_.template segment<3>(3);

      out << "Vel: ["
          << linearSpeed[0] << ", "
          << linearSpeed[1] << ", "
          << linearSpeed[2] << "], [" << bodyAngVel[2] << "]\n";
      out << "Terrain: [" << terrainparams_.transpose() << "]\n";
      out << "Friction: ["
          << footFriction_[0] << ", "
          << footFriction_[1] << ", "
          << footFriction_[2] << ", " << footFriction_[3] << "]\n";

      if (disturbance_on_) {
        out << "disturb " << disturbance_[0] << ", " << disturbance_[1] << ", " << disturbance_[2];
      }

      raisim::gui::VisualizerString = out.str();
      raisim::gui::phases[0] = pi_[0];
      raisim::gui::phases[1] = pi_[1];
      raisim::gui::phases[2] = pi_[2];
      raisim::gui::phases[3] = pi_[3];

      /// Modify Arrows
      auto &list = raisim::OgreVis::get()->getVisualObjectList();
      Eigen::Vector3d temp;
      raisim::Vec<3> temp2;
      raisim::Mat<3, 3> orientation;
      temp << q_[0], q_[1], q_[2] + 0.5;
      {
        auto &arrow = list["command_arrow1"];
        if (command_.head(2).norm() > 0.0) {
          temp2.e() << command_[0], command_[1], 0.0;
          temp2.e() = R_b_ * temp2.e();
          temp2.e().normalize();

          raisim::zaxisToRotMat(temp2, arrow.rotationOffset);
          arrow.offset.e() = temp;
        }
        double scale = command_.head(2).norm();
        arrow.scale[0] = scale * 0.2;
        arrow.scale[1] = scale * 0.2;
        arrow.scale[2] = scale * 0.5;
      }
      {
        auto &arrow = list["command_arrow2"];
        if (command_[2] != 0) {
          temp2.e() << 0.0, 0.0, command_[2];
          temp2.e() = R_b_ * temp2.e();
          temp2.e().normalize();
          raisim::zaxisToRotMat(temp2, arrow.rotationOffset);
          arrow.offset.e() = temp;
        }
        double scale = std::abs(command_[2]);
        arrow.scale[0] = scale * 0.4;
        arrow.scale[1] = scale * 0.4;
        arrow.scale[2] = scale * 0.4;
      }
      {
        auto &arrow = list["disturb_arrow"];
        if (disturbance_on_) {
          temp2.e() = disturbance_.e();
          temp2.e().normalize();

        } else {
          temp2.e() << 0.0, 0.0, 1.0;
        }
        raisim::zaxisToRotMat(temp2, arrow.rotationOffset);

        arrow.offset.e() = temp;
        double scale = disturbance_.e().norm();
        arrow.scale[0] = scale * 0.005;
        arrow.scale[1] = scale * 0.005;
        arrow.scale[2] = scale * 0.01;
      }
      {
        int pos = 0;
        for (size_t i = 0; i < 4; i++) {
          for (size_t j = 0; j < 9; j++) {
            std::string name = "footScan_";
            name += std::to_string(pos);
            auto &dot = list[name];
            dot.offset[0] = footPos_W[i][0] + Fdxs_.data()[j];
            dot.offset[1] = footPos_W[i][1] + Fdys_.data()[j];
            dot.offset[2] = FHs_.data()[pos];
            pos++;
          }
        }
      }

      if (raisim::gui::reinit) {
        init();
        raisim::gui::reinit = false;
      }
      if (raisim::gui::sampleCommand) {
        sampleCommand();
        raisim::gui::sampleCommand = false;
      }
      if (raisim::gui::zeroCommand) {
        command_.setZero();
        raisim::gui::zeroCommand = false;
      }
      if (raisim::gui::disturb) {
        if (!disturbance_on_) {
          double ang = rn_.sampleUniform() * M_PI;
          disturbance_[0] = 50.0 * std::cos(ang);
          disturbance_[1] = 50.0 * std::sin(ang);
          disturbance_[2] = 20.0 * rn_.sampleUniform();
        }
        disturbance_on_ = (!disturbance_on_);
        raisim::gui::disturb = false;
      }
      for (int i = 0; i < 4; i++) {
        if (pi_[i] < 0.0) {
          desiredContactState_[i] = true;
        } else {
          desiredContactState_[i] = false;
        }
      }
      raisim::gui::gaitLogger.appendContactStates(footContactState_);
      raisim::gui::gaitLogger2.appendContactStates(desiredContactState_);
    }
  }

  void updateAction(const Action &action_t) {

    if (isnan(action_t.norm()) || isinf(action_t.norm())) {
      badlyConditioned_ = true;
    }

    for (size_t i = 0; i < 4; i++) {
      prevfootPos_Target2[i] = prevfootPos_Target[i];
      prevfootPos_Target[i] = footPos_Target[i];
    }

    scaledAction_ = action_t.cwiseProduct(actionScale_) + actionOffset_;

    for (size_t i = 0; i < 4; i++) {
      piD_[i] = scaledAction_[i] + baseFreq_;
    }


    /// update Target
    yHorizontal_ << 0.0, e_g_noisy_[2], -e_g_noisy_[1]; // e_g cross 1,0,0
    yHorizontal_.normalize();
    xHorizontal_ = yHorizontal_.cross(e_g_noisy_); // e_g cross y_;
    xHorizontal_.normalize();
    size_t decimation = (size_t) (this->control_dt_ / simulation_dt_);

    for (size_t j = 0; j < 4; j++) {
      pi_[j] += piD_[j] * decimation;
      pi_[j] = anglemod(pi_[j]);
    }
    Eigen::Vector3d sol;

    for (size_t j = 0; j < 4; j++) {
      double dh = 0.0;
      if (pi_[j] > 0.0) {
        double t = pi_[j] / M_PI_2;
        if (t < 1.0) {
          double t2 = t * t;
          double t3 = t2 * t;
          dh = (-2 * t3 + 3 * t2);
        } else {
          t = t - 1;
          double t2 = t * t;
          double t3 = t2 * t;
          dh = (2 * t3 - 3 * t2 + 1.0);
        }
        dh *= clearance_[j];
      }

      footPos_Target[j][0] = footPositionOffset_.tail(12)[3 * j];
      footPos_Target[j][1] = footPositionOffset_.tail(12)[3 * j + 1];
      footPos_Target[j][2] = 0.0;

      footPos_Target[j] += e_g_noisy_ * (footPositionOffset_.tail(12)[3 * j + 2] + dh);
      footPos_Target[j] += e_g_noisy_ * scaledAction_.tail(12)[3 * j + 2];
      footPos_Target[j] += xHorizontal_ * scaledAction_.tail(12)[3 * j];
      footPos_Target[j] += yHorizontal_ * scaledAction_.tail(12)[3 * j + 1];
    }

    t_ = 0.0;
  }

  void addBaseMass(double in) {
    anymal_->getMass()[0] = defaultBodyMasses_[0] + in;
    anymal_->updateMassInfo();
  }

  void seed(int seed) {
    seed_ = seed;
    rn_.seed(seed_);
  }

  void setF(double low, double mid) {
    for (int i = 0; i < 4; i++) {
      footFriction_[i] = std::max(mid + 0.1 * rn_.sampleNormal(), low);
      env_->setMaterialPairProp("terrain", footNames_[i], footFriction_[i], 0.0, 0.0);
    }
    env_->setDefaultMaterial(0.1 + rn_.sampleUniform01() * 0.4, 0.0, 0.0);
  }

  void generateTerrain(Eigen::Matrix<double, 3, 1> &params) {

    if (terrainType_ != Flat_) {
      if (vis_on_) {
        raisim::OgreVis::get()->remove(terrain_);
      }
      env_->removeObject(terrain_);
    }

    double pixelSize_ = 0.02;

    if (taskIndex_ == 0) {
      terrainType_ = TerrainType::Hills;
      setF(0.5, 0.9);
    } else if (taskIndex_ == 1) {
      terrainType_ = TerrainType::Steps;
      setF(0.5, 0.9);
    } else if (taskIndex_ == 2) {
      terrainType_ = TerrainType::Stairs;
      setF(0.5, 0.8);
    } else if (taskIndex_ == 3) {
      terrainType_ = TerrainType::Hills;
      setF(0.1, 0.2);
    } else if (taskIndex_ == 4) {
      terrainType_ = TerrainType::Steps;
      setF(0.1, 0.2);
    } else if (taskIndex_ == 5) {
      terrainType_ = TerrainType::Hills;
      setF(0.2, 0.7);
    } else if (taskIndex_ == 6) {
      terrainType_ = TerrainType::Steps;
      setF(0.2, 0.7);
    } else if (taskIndex_ == 7) {
      terrainType_ = TerrainType::Stairs;
      setF(0.2, 0.7);
    } else if (taskIndex_ == 8) {
      terrainType_ = TerrainType::UniformSlope;
      setF(0.2, 0.7);
    };

    if (terrainType_ == TerrainType::Stairs) {
      terrainProp_.xSize = 4.0;
      terrainProp_.ySize = 8.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      double stepLength = params[1] + 0.1;
      double stepHeight = params[2] + 0.05;

      int N = (int) (stepLength / pixelSize_);
      int mid0 = 0.5 * terrainProp_.ySamples - (int) (0.5 / pixelSize_);
      int mid1 = 0.5 * terrainProp_.ySamples + (int) (0.7 / pixelSize_);
      double stepStart = 0.0;
      double max = 0.0;
      int cnt = 0;
      bool chamfer = false;
      for (int y = 0; y < mid0; y++) {
        if (cnt == N) {
          stepStart = max;
          cnt = 0;
        }
        if (cnt == 0 && rn_.sampleUniform01() < 0.5) chamfer = true;
        else chamfer = false;
        for (int x = 0; x < terrainProp_.xSamples; x++) {
          size_t idx = y * terrainProp_.xSamples + x;
          max = stepStart + stepHeight;
          heights_[idx] = max;
          if (chamfer) heights_[idx] -= gridSize_;
        }
        cnt++;
      }

      for (int y = mid0; y < mid1; y++) {
        for (int x = 0; x < terrainProp_.xSamples; x++) {
          size_t idx = y * terrainProp_.xSamples + x;
          heights_[idx] = max;
        }
      }

      cnt = N;
      for (int y = mid1; y < terrainProp_.ySamples; y++) {
        if (cnt == N) {
          stepStart = max;
          cnt = 0;
        }
        if (cnt == 0 && rn_.sampleUniform01() < 0.5) chamfer = true;
        else chamfer = false;
        for (int x = 0; x < terrainProp_.xSamples; x++) {
          size_t idx = y * terrainProp_.xSamples + x;
          max = stepStart + stepHeight;
          heights_[idx] = max;
          if (chamfer) heights_[idx] -= gridSize_;

        }
        cnt++;
      }
    } else if (terrainType_ == TerrainType::Steps) {
      pixelSize_ = 0.02;
      terrainProp_.xSize = 6.0;
      terrainProp_.ySize = 6.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      double stepSize = params[1] + 0.2;
      double stepHeight = params[2] + 0.05;
      int xNum = terrainProp_.xSize / stepSize;
      int yNum = terrainProp_.ySize / stepSize;
      int gridWidth_ = stepSize / pixelSize_;

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);

      mapMat.setZero();
      ///steps
      for (size_t i = 0; i < xNum; i++) {
        for (size_t j = 0; j < yNum; j++) {
          double h = rn_.sampleUniform01() * stepHeight + 0.5;

          mapMat.block(gridWidth_ * i, gridWidth_ * j, gridWidth_, gridWidth_).setConstant(h);
        }
      }

    } else if (terrainType_ == TerrainType::SingleStep) {

      terrainProp_.xSize = 8.0;
      terrainProp_.ySize = 8.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      double stepHeight = params[0] + 0.5;

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);

      mapMat.setConstant(stepHeight);
      int end_idx = 0.6 * terrainProp_.ySamples;
      for (int y = 0; y < end_idx; y++) {
        for (int x = 0; x < terrainProp_.xSamples; x++) {
          size_t idx = y * terrainProp_.xSamples + x;
          if (y == end_idx - 1) {
//            heights_[idx] -= pixelSize_;
          } else {
            heights_[idx] = 0.5;
          }
        }
      }

    } else if (terrainType_ == TerrainType::Hills) {
      pixelSize_ = 0.15;

      terrainProp_.xSize = 8.0;
      terrainProp_.ySize = 8.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      terrainProp_.fractalOctaves = 1;
      terrainProp_.frequency = 0.1 + params[1]; ///

      if (seed_ == -1) {
        terrainGenerator_.setSeed(rn_.intRand(0, 100));
      } else {
        terrainGenerator_.setSeed(seed_);
      }

      terrainGenerator_.getTerrainProp() = terrainProp_;

      heights_ = terrainGenerator_.generatePerlinFractalTerrain();
      for (size_t i = 0; i < heights_.size(); i++) {
        heights_[i] += 1.0;
      }

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);
      mapMat *= params[2] + 0.2;

      for (size_t idx = 0; idx < heights_.size(); idx++) {
        heights_[idx] += (params[0]) * rn_.sampleUniform01();
      }
    } else if (terrainType_ == TerrainType::UniformSlope) {
      pixelSize_ = 0.02;
      terrainProp_.xSize = 8.0;
      terrainProp_.ySize = 8.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;
      ///temp
      double Hills = params[0];
      double dh = std::tan(Hills) * pixelSize_;

      for (int y = 0; y < terrainProp_.ySamples; y++) {
        for (int x = 0; x < terrainProp_.xSamples; x++) {
          size_t idx = y * terrainProp_.xSamples + x;
          heights_[idx] = dh * y + 5.0;
        }
      }
    } else {
      terrainProp_.xSize = 8.0;
      terrainProp_.ySize = 8.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      terrainProp_.fractalOctaves = 1;
      terrainProp_.frequency = 0.2 + params[2]; ///

      if (seed_ == -1) {
        terrainGenerator_.setSeed(rn_.intRand(0, 100));
      } else {
        terrainGenerator_.setSeed(seed_);
      }

      terrainGenerator_.getTerrainProp() = terrainProp_;

      heights_ = terrainGenerator_.generatePerlinFractalTerrain();

      for (size_t i = 0; i < heights_.size(); i++) {
        heights_[i] += 1.0;
      }

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);
      mapMat *= params[3] + 0.2;

      ///steps
      double stepSize = params[1] + 0.2;
      int xNum = terrainProp_.xSize / stepSize;
      int yNum = terrainProp_.ySize / stepSize;
      int gridWidth_ = stepSize / pixelSize_;

      for (size_t i = 0; i < xNum; i++) {
        for (size_t j = 0; j < yNum; j++) {
          double h = mapMat.block(gridWidth_ * i, gridWidth_ * j, gridWidth_, gridWidth_).mean();

          if (rn_.sampleUniform01() < 0.5) {
            mapMat.block(gridWidth_ * i, gridWidth_ * j, gridWidth_, gridWidth_).setConstant(h);
          }
        }
      }
    }
    Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                     terrainProp_.xSamples,
                                                     terrainProp_.ySamples);

    terrain_ = env_->addHeightMap(terrainProp_.xSamples,
                                  terrainProp_.ySamples,
                                  terrainProp_.xSize,
                                  terrainProp_.ySize, 0.0, 0.0, heights_, "terrain");
    if (vis_on_) {
      raisim::OgreVis::get()->createGraphicalObject(terrain_, "height_map", "gray");
    }
  }

  void updateCommand() {
    if (commandMode_ == CommandMode::FIXED_DIR) {
      return;
    }
    if (command_.head(2).norm() != 0.0) {
      double x_dist = goalPosition_[0] - q_[0];
      double y_dist = goalPosition_[1] - q_[1];
      if (std::sqrt(x_dist * x_dist + y_dist * y_dist) < 0.5) {
        sampleCommand();
      }
      commandDirection_ = std::atan2(y_dist, x_dist);
      Eigen::Vector3d heading;
      heading = R_b_.col(0);
      double headingAngle = std::atan2(heading[1], heading[0]);
      commandDirection_bodyframe_ = commandDirection_ - headingAngle;
      commandDirection_bodyframe_ = anglemod(commandDirection_bodyframe_);
      command_[0] = std::cos(commandDirection_bodyframe_);
      command_[1] = std::sin(commandDirection_bodyframe_);
    }

    if (stopMode_) {
      baseFreq_ = 0.0;
    } else {
      baseFreq_ = 1.3 * freqScale_;
    }
  }

  inline void sampleGoal() {
    goalPosition_[0] = q_[0];
    goalPosition_[1] = q_[1];

    if (terrainType_ == TerrainType::Stairs) {
      goalPosition_[0] += terrainProp_.xSize * 0.1 * rn_.sampleUniform();
      if (q_[1] < 0.3) {
        goalPosition_[1] = terrainProp_.ySize * (0.3 + 0.2 * rn_.sampleUniform01());
      } else {
        goalPosition_[1] = -terrainProp_.ySize * (0.3 + 0.2 * rn_.sampleUniform01());
      }
    } else {
      goalPosition_[0] += terrainProp_.xSize * 0.4 * rn_.sampleUniform();
      goalPosition_[1] += terrainProp_.ySize * 0.4 * rn_.sampleUniform();

      if (commandMode_ == CommandMode::STRAIGHT) {
        goalPosition_[0] = q_[0] + terrainProp_.xSize * 0.1 * rn_.sampleUniform();
        goalPosition_[1] = q_[1] + terrainProp_.ySize * 0.4;
      }
    }

    goalPosition_[0] = std::min(0.5 * terrainProp_.xSize - 1.0, goalPosition_[0]);
    goalPosition_[0] = std::max(-0.5 * terrainProp_.xSize + 1.0, goalPosition_[0]);
    goalPosition_[1] = std::min(0.5 * terrainProp_.ySize - 1.0, goalPosition_[1]);
    goalPosition_[1] = std::max(-0.5 * terrainProp_.ySize + 1.0, goalPosition_[1]);
  }

  void sampleCommand() {
    if (commandMode_ == CommandMode::FIXED_DIR) {
      goalPosition_[0] = 0.0;
      goalPosition_[1] = 0.0;
      command_ << 1.0, 0.0, 0.0;
      return;
    } else if (commandMode_ == CommandMode::ZERO) {
      command_.setZero();
      goalPosition_[0] = 10.0;
      goalPosition_[1] = 10.0;
      return;
    }
    sampleGoal();
    commandDirection_ = std::atan2(goalPosition_[1] - q_[1], goalPosition_[0] - q_[0]);

    Eigen::Vector3d heading;
    heading = R_b_.col(0);
    double headingAngle = std::atan2(heading[1], heading[0]);
    commandDirection_bodyframe_ = commandDirection_ - headingAngle;
    commandDirection_bodyframe_ = anglemod(commandDirection_bodyframe_);

    command_[0] = std::cos(commandDirection_bodyframe_);
    command_[1] = std::sin(commandDirection_bodyframe_);
    command_[2] = 0.0;

    if (commandMode_ != CommandMode::STRAIGHT) {
      command_[2] = 1.0 - 2.0 * rn_.intRand(0, 1);
      command_[2] *= rn_.sampleUniform01();
      if ((commandMode_ != CommandMode::NOZERO) && (rn_.sampleUniform01() > 0.8)) {
        command_.head(2).setZero();
      }
    }

    if (terrainType_ == TerrainType::Stairs) {
      if (rn_.sampleUniform01() < 0.5) {
        command_[2] = 0.0;
      }
    }
  }

  void init() {
    baseFreq_ = 1.3 * freqScale_;

    numContact_ = 0;
    numFootContact_ = 0;
    numShankContact_ = 0;
    numThighContact_ = 0;
    numBaseContact_ = 0;
    controlCounter_ = 0;

    for (int i = 0; i < 4; i++) {
      footContactState_[i] = false;
      shankContacts_[i] = false;
      thighContacts_[i] = false;
    }

    steps_ = 0;
    tau_.setZero();
    badlyConditioned_ = false;

    Eigen::Matrix<double, 3, 1> terrainNormal_;
    Eigen::Matrix<double, 3, 1> a1_;
    Eigen::Matrix<double, 3, 1> a2_;
    q_ = q0;

    q_[0] = 0.1 * terrainProp_.xSize * rn_.sampleUniform();
    q_[1] = 0.1 * terrainProp_.ySize * rn_.sampleUniform();

    if (terrainType_ != TerrainType::Flat_) {
      a1_[0] = 2.0;
      a1_[1] = 0.0;
      a1_[2] = terrain_->getHeight(q_[0] + 1.0, q_[1]) - terrain_->getHeight(q_[0] - 1.0, q_[1]);
      a2_[0] = 0.0;
      a2_[1] = 2.0;
      a2_[2] = terrain_->getHeight(q_[0], q_[1] + 1.0) - terrain_->getHeight(q_[0], q_[1] - 1.0);
      a1_.normalize();
      a2_.normalize();
      terrainNormal_ = a1_.cross(a2_);

      R_b_.col(0) = a1_;
      R_b_.col(1) = a2_;
      R_b_.col(2) = terrainNormal_;

      Eigen::Matrix<double, 4, 1> quat;
      quat = Math::MathFunc::rotMatToQuat(R_b_);
      double angle = M_PI * rn_.sampleUniform();

      if (terrainType_ == TerrainType::Stairs) {
        angle = M_PI * 0.5 - M_PI * rn_.intRand(0, 1);
      }
      if (commandMode_ == CommandMode::STRAIGHT || commandMode_ == CommandMode::FIXED_DIR) {
        angle = M_PI * 0.5;
      }
      q_.template segment<4>(3) = Math::MathFunc::rotateQuatByAngleAxis(quat, angle, terrainNormal_);
    } else {
      Eigen::Vector3d heading;
      heading(0) = 0.05 * rn_.sampleUniform() * noiseFtr_;
      heading(1) = 0.05 * rn_.sampleUniform() * noiseFtr_;
      heading(2) = 1.0;
      heading.normalize();
      double angle = M_PI * rn_.sampleUniform();
      double sin = std::sin(angle / 2.0);
      q_.template segment<3>(4) = heading * sin;
      q_(3) = std::cos(angle / 2.0);
    }

    for (int i = 0; i < 18; i++) {
      u_(i) = u_initialNoiseScale(i) * rn_.sampleUniform() * noiseFtr_; // sample uniform
    }

    for (size_t i = 0; i < 4; i++) {
      pi_[i] = 2.0 * M_PI * rn_.sampleUniform01();
      pi_[i] = anglemod(pi_[i]);
    }

    for (size_t i = 0; i < 4; i++) {
      footNormals_W_[i] << 0.0, 0.0, 1.0;
      footContactPositions_W_[i][2] = -10.0;
    }

    for (size_t j = 0; j < 4; j++) {
      Eigen::Vector3d target, sol;
      target[0] = actionOffset_.tail(12)[3 * j];
      target[1] = actionOffset_.tail(12)[3 * j + 1];
      target[2] = -0.5 + std::max(actionOffset_[0] * std::sin(pi_[j]), 0.0);
      footPos_Target[j] = target;

      IK_.IKSagittal(sol, target, j);
    }

    for (int i = 7; i < 19; i++) {
      q_(i) += q_initialNoiseScale(i) * rn_.sampleUniform(); // sample uniform
    }

    /// decide initial height

    anymal_->setGeneralizedCoordinate(q_);
    anymal_->getPosition(3, footPos_[0], footPos_W[0]);
    anymal_->getPosition(6, footPos_[1], footPos_W[1]);
    anymal_->getPosition(9, footPos_[2], footPos_W[2]);
    anymal_->getPosition(12, footPos_[3], footPos_W[3]);

    if (terrainType_ != Flat_) {
      double minFootGap = footPos_W[0][2] - terrain_->getHeight(footPos_W[0][0], footPos_W[0][1]);
      int idx = 0;
      for (int i = 1; i < 4; i++) {
        double gap = footPos_W[i][2] - terrain_->getHeight(footPos_W[i][0], footPos_W[i][1]);
        if (gap < minFootGap) {
          minFootGap = gap;
          idx = i;
        }
      }

      q_(2) -= minFootGap;
    } else {
      double minFootGap = footPos_W[0][2];
      int idx = 0;
      for (int i = 1; i < 4; i++) {
        double gap = footPos_W[i][2];
        if (gap < minFootGap) {
          minFootGap = gap;
          idx = i;
        }
      }
      q_(2) -= minFootGap;
    }

    uPrev_ = u_;
    acc_.setZero();

    anymal_->setGeneralizedCoordinate(q_);
    anymal_->setGeneralizedVelocity(u_);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(anymal_->getDOF()));

    env_->integrate1();
    anymal_->setGeneralizedForce(tau_);
    env_->integrate2();

    q_ = anymal_->getGeneralizedCoordinate().e();
    u_ = anymal_->getGeneralizedVelocity().e();

    Eigen::Matrix<double, 4, 1> quat = q_.segment<4>(3);
    R_b_ = Math::MathFunc::quatToRotMat(quat);
    e_g_ = R_b_.row(2).transpose();
    Eigen::Matrix<double, 4, 1> quat2;
    Eigen::Matrix<double, 3, 1> axis;
    axis << rn_.sampleUniform01(), rn_.sampleUniform01(), rn_.sampleUniform01();
    axis.normalize();
    e_g_bias_ = 0.1 * rn_.sampleUniform() * axis;
    quat2 = Math::MathFunc::boxplusI_Frame(quat, e_g_bias_);
    R_b_noisy = Math::MathFunc::quatToRotMat(quat2);
    e_g_noisy_ = R_b_noisy.row(2).transpose();

    yHorizontal_ << 0.0, e_g_noisy_[2], -e_g_noisy_[1]; // e_g cross 1,0,0
    yHorizontal_.normalize();
    xHorizontal_ = yHorizontal_.cross(e_g_noisy_); // e_g cross y_;
    xHorizontal_.normalize();

    comprehendContacts();

    jointPositionTarget_ = q_.tail(12);
    previousjointPositionTarget_ = jointPositionTarget_;
    previousjointPositionTarget2_ = jointPositionTarget_;

    previousAction_ = actionOffset_.template cast<float>();
    scaledAction_ = actionOffset_.template cast<float>();

    jointPosHist_.setZero();
    historyBuffer_.setZero();

    for (int i = 0; i < JointHistoryLength; i++) {
      jointVelHist_.segment(i * 12, 12) = u_.tail(12).template cast<float>();
    }

    dt_ = simulation_dt_ / control_dt_;
  }

  void getState(Eigen::Matrix<float, -1, 1> &state) {
    updateCommand();

    State temp;
    conversion_GeneralizedState2LearningState(temp, q_, u_);

    state = temp;
    /// noisify body vel
    for (int i = 6; i < 9; i++)
      state[i] += rn_.sampleUniform() * 0.1 * stateScale_[i];

    /// noisify body angvel
    for (int i = 9; i < 12; i++)
      state[i] += rn_.sampleUniform() * 0.2 * stateScale_[i];

    /// noisify joint position
    for (int i = 12; i < 24; i++)
      state[i] += rn_.sampleUniform() * 0.01 * stateScale_[i];

    /// noisify joint vel
    for (int i = 24; i < 36; i++)
      state[i] += rn_.sampleUniform() * 1.5 * stateScale_[i];

    /// noisify joint position
    for (int i = 36; i < 48; i++)
      state[i] += rn_.sampleUniform() * 0.01 * stateScale_[i];

    for (int i = 84; i < 108; i++)
      state[i] += rn_.sampleUniform() * 1.5 * stateScale_[i];
    for (int i = 60; i < 84; i++)
      state[i] += rn_.sampleUniform() * 0.01 * stateScale_[i];

    if (isnan(state.norm()) || isinf(state.norm())) {
      badlyConditioned_ = true;
    }
  }

  void getPriviligedState(Eigen::Matrix<float, -1, -1> &state) {
    updateCommand();

    State temp;
    state.resize(PrivilegedStateDim, 1);
    state.setZero();

    conversion_GeneralizedState2LearningState(temp, q_, u_);
    state.col(0).head(StateDim) = temp;

//return;
    int pos = StateDim;

    int num = sampleN / 4;
    int pos2 = 0;

    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < num; j++) {
        state(pos + pos2, 0) = footPos_W[i][2] - FHs_.data()[pos2];
        state(pos + pos2, 0) = std::max(std::min(state(pos + pos2, 0), 0.25f), -0.25f);
        state(pos + pos2, 0) -= 0.05;
        state(pos + pos2, 0) *= 10.0;
        pos2++;
      }
    }

    pos += sampleN;
    for (size_t i = 0; i < 4; i++) {
      state(pos + i, 0) = (footContactState_[i] - 0.5) * 3.0;
    }
    pos += 4;

    for (size_t i = 0; i < 4; i++) {
      int start_idx = 3 * i + pos;
      state.col(0).template segment<3>(start_idx) = netFootContacts_b[i].template cast<float>();
      state(start_idx + 2, 0) -= 80.0;
      state(start_idx, 0) = std::max(std::min(state(start_idx, 0), 50.0f), -50.0f);
      state(start_idx + 1, 0) = std::max(std::min(state(start_idx + 1, 0), 50.0f), -50.0f);
      state(start_idx + 2, 0) = std::max(std::min(state(start_idx + 2, 0), 100.0f), -100.0f);
      state(start_idx, 0) *= 0.01;
      state(start_idx + 1, 0) *= 0.01;
      state(start_idx + 2, 0) *= 0.02;
    }
    pos += 12;

    for (size_t i = 0; i < 4; i++) {
      int start_idx = 3 * i + pos;
      state.col(0).template segment<3>(start_idx) = footNormal_b[i].template cast<float>();
      state(start_idx, 0) *= 5.0;
      state(start_idx + 1, 0) *= 5.0;
      state(start_idx + 2, 0) -= 1.0;
      state(start_idx + 2, 0) *= 20.0;
    }
    pos += 12;

    for (size_t i = 0; i < 4; i++) {
      state(pos + i, 0) = (footFriction_[i] - 0.6) * 2.0;
    }
    pos += 4;

    for (size_t i = 0; i < 4; i++) {
      state(pos, 0) = (thighContacts_[i] - 0.5) * 2.0;
      pos++;
      state(pos, 0) = (shankContacts_[i] - 0.5) * 2.0;
      pos++;
    }

    Eigen::Matrix<double, 3, 1> disturbance_in_base;
    disturbance_in_base = R_b_.transpose() * disturbance_.e();
    disturbance_in_base *= 0.1;

    state(pos++, 0) = disturbance_in_base[0];
    state(pos++, 0) = disturbance_in_base[1];
    state(pos++, 0) = disturbance_in_base[2];
  }

  Eigen::VectorXd getGeneralizedState() { return q_; }

  Eigen::VectorXd getGeneralizedVelocity() { return u_; }

  Eigen::VectorXd getBaseVelocity() {
    Eigen::Matrix<double, 6, 1> output;
    output.segment(0, 3) = (R_b_.transpose() * u_.head(3));
    output.segment(3, 3) = (R_b_.transpose() * u_.segment<3>(3));
    return output;
  }

  void setRealTimeFactor(double fctr) {
    realTimeRatio_ = fctr;
  }

  inline void integrateOneTimeStep() {

    env_->integrate1();
    anymal_->setGeneralizedForce(tau_);
    env_->integrate2();

    Eigen::VectorXd q_temp = anymal_->getGeneralizedCoordinate().e();
    Eigen::VectorXd u_temp = anymal_->getGeneralizedVelocity().e();

    if (!badlyConditioned_) {
      q_ = q_temp;
      u_ = u_temp;
    }

    Eigen::Matrix<double, 4, 1> quat = q_.segment<4>(3);
    R_b_ = Math::MathFunc::quatToRotMat(quat);
    e_g_ = R_b_.row(2).transpose();

    if (disturbance_on_) {
      anymal_->setExternalForce(0,
                                disturbance_);

    }

    visDecimation_ = size_t(1. / (30.0 * 0.0025)) * realTimeRatio_;

    if (vis_on_) {
      if (visCounter_ % visDecimation_ == 0) {
        raisim::OgreVis::get()->renderOneFrame();
        visCounter_ = 0;
      }
      visCounter_++;
    }

    comprehendContacts();
  }

  virtual void startRecordingVideo(std::string path) {
    if (vis_on_) {
      raisim::OgreVis::get()->showWindow();
      raisim::OgreVis::get()->startRecordingVideo(path);
      vid_on_ = true;
    }
  }

  virtual void endRecordingVideo() {
    if (vid_on_) {
      raisim::OgreVis::get()->stopRecordingVideoAndSave();
      raisim::OgreVis::get()->hideWindow();
    }
    vid_on_ = false;
  }

  void updateTask(Eigen::Matrix<float, -1, 1> &input) {
    bool same = true;
    taskIndex_ = input[0];
    for (size_t i = 0; i < terrainparams_.size(); i++) {
      if (terrainparams_[i] != input[i + 1]) {
        same = false;
        terrainparams_[i] = input[i + 1];
      }
    }

    double disturbance_norm = 0.0;
    if (input.size() < 5) {
      disturbance_norm = rn_.sampleUniform() * 120;
      if (rn_.sampleUniform() < 0.0) disturbance_norm = 0.0;

    } else {
      disturbance_norm = input[4];
    }

    if (disturbance_on_) {
      double az = M_PI * rn_.sampleUniform();
      double el = M_PI_2 * rn_.sampleUniform();
      disturbance_[0] = std::cos(el) * std::cos(az);
      disturbance_[1] = std::cos(el) * std::sin(az);
      disturbance_[2] = std::sin(el);

      disturbance_ *= disturbance_norm;
    } else {
      disturbance_.e().setZero();
    }

    if (!same || terrainType_ == Flat_) {
      generateTerrain(terrainparams_);
    }
  }

 private:

  inline double anglediff(double target, double source) {
    //https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return anglemod(target - source);
  }

  inline double anglemod(double a) {
    return wrapAngle((a + M_PI)) - M_PI;
  }

  inline double wrapAngle(double a) {
    double twopi = 2.0 * M_PI;
    return a - twopi * fastfloor(a / twopi);
  }

  inline int fastfloor(double a) {
    int i = int(a);
    if (i > a) i--;
    return i;
  }

/**
      current state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       gravity vector                                              n =  3, si =   3
       *       body Linear velocities,                                     n =  3, si =   6
       *       body Angular velocities,                                    n =  3, si =   9
       *       joint position                                              n = 12, si =  12
       *       joint velocity                                              n = 12, si =  24
       *       joint position err                                          n = 12, si =  36
       *       previous action                                             n = 16, si =  48
       *       contact state                                               n =  4, si =  64
       *       phases                                                      n =  8, si =  68
       *       ]
*/
  inline void conversion_GeneralizedState2LearningState(State &state,
                                                        const Eigen::Matrix<double, -1, 1> &q,
                                                        const Eigen::Matrix<double, -1, 1> &u) {

    Eigen::Matrix<double, 4, 1> quat = q_.segment<4>(3);
    R_b_ = Math::MathFunc::quatToRotMat(quat);
    e_g_ = R_b_.row(2).transpose();

    Eigen::Matrix<double, 4, 1> quat2;
    quat2 = Math::MathFunc::boxplusI_Frame(quat, e_g_bias_);
    R_b_noisy = Math::MathFunc::quatToRotMat(quat2);
    e_g_noisy_ = R_b_noisy.row(2).transpose();

    double r, p, y;
    Math::MathFunc::QuattoEuler(quat, r, p, y);

    State state_unscaled;

    /// velocity in body coordinate
    Eigen::Matrix<double, 3, 1> bodyVel = R_b_.transpose() * u.template segment<3>(0);
    Eigen::Matrix<double, 3, 1> bodyAngVel = R_b_.transpose() * u.template segment<3>(3);
    Eigen::Matrix<double, 12, 1> jointVel = u.template segment<12>(6);

    state_unscaled.head(3) = command_.template cast<float>();

    state_unscaled.template segment<3>(3) = e_g_noisy_.template cast<float>();
    state_unscaled.template segment<3>(6) = bodyVel.template cast<float>();
    state_unscaled.template segment<3>(9) = bodyAngVel.template cast<float>();
    state_unscaled.template segment<12>(12) = q.template segment<12>(7).
        template cast<float>(); /// position
    state_unscaled.template segment<12>(24) = jointVel.template cast<float>();

    state_unscaled.template segment<12>(36) =
        jointPosHist_.template segment<12>((JointHistoryLength - 1) * 12);

    for (size_t i = 0; i < 4; i++) {
      state_unscaled[48 + i] = piD_[i];
    }

    int pos = 52;
    for (size_t i = 0; i < 4; i++) {
      state_unscaled[pos + 2 * i] = std::sin(pi_[i]);
      state_unscaled[pos + 2 * i + 1] = std::cos(pi_[i]);
    }
    pos += 8;

    state_unscaled.template segment<12>(pos) =
        jointPosHist_.template segment<12>((JointHistoryLength - 4) * 12);
    pos += 12;

    state_unscaled.template segment<12>(pos) =
        jointPosHist_.template segment<12>((JointHistoryLength - 9) * 12);
    pos += 12;

    state_unscaled.template segment<12>(pos) =
        jointVelHist_.template segment<12>((JointHistoryLength - 4) * 12);
    pos += 12;

    state_unscaled.template segment<12>(pos) =
        jointVelHist_.template segment<12>((JointHistoryLength - 9) * 12);
    pos += 12;

//    for (size_t i = 0; i < 4; i++) {
//      state_unscaled.template segment<3>(pos) = footPos_Target[i].template cast<float>();
//      pos += 3;
//    }
//
//    for (size_t i = 0; i < 4; i++) {
//      state_unscaled.template segment<3>(pos) = prevfootPos_Target[i].template cast<float>();
//      pos += 3;
//    }
    state_unscaled.template segment<12>(pos) = jointPositionTarget_.template cast<float>();
    pos += 12;

    state_unscaled.template segment<12>(pos) = previousjointPositionTarget_.template cast<float>();
    pos += 12;

    state_unscaled[pos] = baseFreq_;
    pos++;

    state = (state_unscaled - stateOffset_).cwiseProduct(stateScale_);
  }

/// task spec
 public:
  ActionType actionType_ = ActionType::EE;
  CommandMode commandMode_ = CommandMode::RANDOM;

  int instance_;
  raisim::Vec<3> disturbance_;
  bool disturbance_on_ = false;

  double pi_[4];
  double piD_[4];
  double clearance_[4];
  double h0_ = -0.55;
  Eigen::VectorXd uPrev_;
  Eigen::VectorXd acc_;
  int steps_ = 0;

/// sim
  raisim::ArticulatedSystem *anymal_ = nullptr;
  std::vector<raisim::GraphicObject> *anymalVisual_ = nullptr;
  raisim::Ground *board_;

///terrain
  std::vector<double> heights_;
  raisim::HeightMap *terrain_;
  raisim::TerrainProperties terrainProp_;
  raisim::TerrainGenerator terrainGenerator_;
  Eigen::Matrix<double, 3, 1> terrainparams_;
  int taskIndex_ = 0;
  TerrainType terrainType_ = TerrainType::Flat_;
  double gridSize_ = 0.025;

  raisim::World *env_;
  int terrainKey_, robotKey_, slipperyKey_;
  raisim::Vec<3> gravity_;

  rai::FuncApprox::MLP_fullyconnected<double, 6, 1, rai::FuncApprox::ActivationType::softsign> actuator_A_;

  Eigen::VectorXd u_, u_initialNoiseScale, u0;
  Eigen::VectorXd q_, q_initialNoiseScale, q0;

  Vector18d tau_;
  Eigen::Matrix<double, 3, 3> R_b_, R_b_noisy;
  Eigen::Matrix<double, 3, 1> e_g_, e_g_noisy_, e_g_bias_;

  Vector12d jointNominalConfig_;
  Eigen::Matrix<double, 3, 1> xHorizontal_, yHorizontal_; //in body frame

  const double simulation_dt_ = 0.0025;
  double control_dt_ = 0.01;
  size_t controlCounter_ = 0;

// innerStates
  std::vector<Eigen::Matrix<double, 3, 1>> defaultJointPositions_;
  std::vector<double> defaultBodyMasses_;

  std::vector<raisim::Vec<3>> footPos_;
  std::vector<raisim::Vec<3>> footPos_W;
  std::vector<raisim::Vec<3>> footPos_b;
  std::vector<raisim::Vec<3>> footPos_b_prev;

  std::vector<Eigen::Vector3d> footPos_Target;
  std::vector<Eigen::Vector3d> prevfootPos_Target;
  std::vector<Eigen::Vector3d> prevfootPos_Target2;

  std::vector<raisim::Vec<3>> footVel_W;
  std::vector<Eigen::Matrix<double, 3, 1>> footVel_projected;
  std::vector<raisim::Vec<3>> footAngVel_W;
  double footR_[4];
  std::string footNames_[4];

  std::vector<Eigen::Matrix<double, 3, 1>> footNormal_;
  std::vector<Eigen::Matrix<double, 3, 1>> footNormal_b;
  Eigen::Matrix<double, 3, 1> COMPosition_;

// history buffers
  Eigen::Matrix<float, 12 * JointHistoryLength, 1> jointVelHist_, jointPosHist_;
  Eigen::Matrix<float, 12 * JointHistoryLength, 1> torqueHist_;
  Eigen::Matrix<float, ObservationDim, JointHistoryLength> historyBuffer_;

// Buffers for contact states
  std::array<bool, 4> footContactState_;
  std::array<bool, 4> desiredContactState_;

  std::array<bool, 4> shankContacts_;
  std::array<bool, 4> thighContacts_;
  std::array<double, 4> footFriction_;

  size_t numContact_;
  size_t numFootContact_;
  size_t numShankContact_;
  size_t numThighContact_;
  size_t numBaseContact_;
  size_t numInternalContact_;
  std::vector<Eigen::Vector3d> netFootContacts_;
  std::vector<Eigen::Vector3d> netFootContacts_b;
  std::vector<Eigen::Vector3d> netFootContactVels_;
  Eigen::Vector3d netContacts_;
  Eigen::Vector3d netContacts_b;
  std::vector<size_t> FootContactNums_;

// Check for divergence
  bool badlyConditioned_ = false;

// Visualize
  double realTimeRatio_;
  bool vis_on_ = false;
  bool vis_ready_;
  bool vid_on_;

 public:
  int ObservationStride_ = 8; //collected in 100 Hz

  Eigen::Matrix<double, 3, 1> command_;
  bool stopMode_ = false;
  double velThres_ = 0.2;
  double goalPosition_[2];
  double commandDirection_;
  double commandDirection_bodyframe_;

  State state0_;
  State stateOffset_;
  State stateScale_;
  Action actionOffset_;
  Action actionScale_;
  Action scaledAction_;

  Observation observationScale_;
  Observation observationOffset_;

  double freqScale_ = 0.0025 * 2.0 * M_PI; // 1 Hz 0.0157
  double baseFreq_;
  Eigen::Matrix<float, 12, 1> footPositionOffset_;
  Eigen::Matrix<double, 12, 1> jointPositionTarget_;
  Eigen::Matrix<double, 12, 1> previousjointPositionTarget_;
  Eigen::Matrix<double, 12, 1> previousjointPositionTarget2_;

  std::vector<Eigen::Matrix<double, 3, 1>> footContactPositions_W_;
  std::vector<Eigen::Matrix<double, 3, 1>> footContactPositions_W_Noisy;
  std::vector<Eigen::Matrix<double, 3, 1>> footNormals_W_;

  Eigen::Matrix<float, 9, 1> Fdxs_;
  Eigen::Matrix<float, 9, 1> Fdys_;
  Eigen::Matrix<float, sampleN, 1> FHs_;

  double footMargin_ = 0.13;
  double t_; // [0, 1]
  double dt_; // [0, 1]
  Action previousAction_;

  size_t visCounter_ = 0;
  size_t visDecimation_ = size_t(1. / (30.0 * 0.0025));

  double noiseFtr_ = 0.2;

  RandomNumberGenerator<float> rn_;
  int seed_ = -1;
  InverseKinematics IK_;
};
}//namespace env
