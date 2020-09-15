//
// Created by joonho on 7/28/20.
//

#include <environment/environment.hpp>
#include <graph/Policy.hpp>

constexpr int history_len = 100; // todo: move to controller config.

int main(int argc, char *argv[]) {

  std::string urdf_path = "/home/joonho/oldws/blind_sandbox/rsc/robot/chimera/urdf/anymal_minimal.urdf";
  std::string actuator_path = "/home/joonho/oldws/blind_sandbox/rsc/actuator/C100/seaModel_2500.txt";
  std::string network_path = "/home/joonho/oldws/blind_sandbox/rsc/controller/graph.pb";
  std::string param_path = "/home/joonho/oldws/blind_sandbox/rsc/controller/param.txt";

  Env::Chimera_blind sim(true, 0, urdf_path, actuator_path);
  Policy<Env::ActionDim> policy(network_path,
                                param_path,
                                Env::StateDim,
                                Env::ObservationDim,
                                history_len);

  Eigen::Matrix<float, -1, 1> task_params(4);
  task_params << 0.0, 0.05, 0.5, 0.5;
  sim.updateTask(task_params);

  Eigen::Matrix<float, -1, -1> history;
  Eigen::Matrix<float, -1, 1> state;
  Eigen::Matrix<float, Env::ActionDim, 1> action;

  sim.init();
  for (int i = 0; i < 500; i++) {
    sim.integrate();
    sim.getHistory(history, history_len);
    sim.getState(state);

    policy.updateStateBuffer(state);
    policy.updateHistoryBuffer(history);
    policy.getAction(action);
    sim.updateAction(action);
  }
  return 0;
}