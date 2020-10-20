//
// Created by joonho on 7/28/20.
//

#include <environment/environment_c010.hpp>
#include <graph/Policy.hpp>


int main(int argc, char *argv[]) {

  ///Hard-coded. TODO: clean it up
  std::string rsc_path = "/home/jolee/raisimws/learning_locomotion_over_challening_terrain_supplementary/rsc";
  std::string policy_name = "tcn100";
  int history_len = 100; // TODO: move to controller config.


  std::string urdf_path = rsc_path + "/robot/c010/urdf/anymal_minimal.urdf";
  std::string actuator_path = rsc_path + "/actuator/c010/seaModel_A.txt";
  std::string network_path = rsc_path + "/controller/c010/" + policy_name + "/graph.pb";
  std::string param_path = rsc_path + "/controller/c010/"+ policy_name + "/param.txt";
  bool teacher_policy = false;
  if(policy_name == "teacher") teacher_policy=true;

  raisim::World::setActivationKey("/home/jolee/raisimws/install/activation.raisim");

  Env::blind_locomotion sim(true, 0, urdf_path, actuator_path);
  Policy<Env::ActionDim> policy;

  if (teacher_policy) {
    policy.load(network_path,
                param_path,
                Env::StateDim,
                Env::PrivilegedStateDim);
  }else{
    policy.load(network_path,
                param_path,
                Env::StateDim,
                Env::ObservationDim,
                history_len);
  }

  Eigen::Matrix<float, -1, -1> state2;
  Eigen::Matrix<float, -1, 1> state;
  Eigen::Matrix<float, Env::ActionDim, 1> action;

  /// set terrain properties
  Eigen::Matrix<float, -1, 1> task_params(4);
  task_params << 0.0, 0.05, 0.5, 0.5;
  sim.updateTask(task_params);

  sim.init();
  /// simulate for 30 seconds.
  for (int i = 0; i < 1500; i++) {
    sim.integrate();
    if (policy.teacher_) {
      sim.getPriviligedState(state2);
    } else {
      sim.getHistory(state2, history_len);
    }
    sim.getState(state);

    policy.updateStateBuffer(state);
    policy.updateStateBuffer2(state2);
    policy.getAction(action);
    sim.updateAction(action);
  }
  return 0;
}