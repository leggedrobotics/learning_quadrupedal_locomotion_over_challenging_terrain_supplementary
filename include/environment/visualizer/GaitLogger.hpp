
#ifndef _GAIT_LOGGER_HPP
#define _GAIT_LOGGER_HPP

#include <unordered_map>
#include <vector>
#include <Eigen/Core>

namespace raisim {

class GaitLogger {

 public:
  GaitLogger() = default;

  void init(int size = 0) {
    clean();
    if(size != 0){
      for(size_t i = 0; i < 4; i++) {
        contactStates_[i].reserve(size);
      }
      phases_.reserve(size);
    }
  }

  void clean() {
    for(size_t i = 0; i < 4; i++) {
      contactStates_[i].clear();
    }
    phases_.clear();
  }

  void appendContactStates(std::array<bool,4> & in){
    for(size_t i = 0; i < 4; i++){
      contactStates_[i].push_back((float)in[i]);
    }
  }

//  void appendPhases(std::array<double,4> & in){
//    phases_.push_back(in);
//  }

  std::vector<float> contactStates_[4];
  std::vector<std::array<float, 4>> phases_;
};

};
#endif //_RAISIM_GYM_REWARDLOGGER_HPP
