// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_SIM_NODE_H_
#define HURO_SIM_NODE_H_

#include <huro/params.h>

#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
// #include <sensor_msgs/msg/joy.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/motor_cmd.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

namespace huro {
template <typename LowCmdMsg, typename LowStateMsg, typename OdometryMsg,
          size_t N_MOTORS>
class SimNode : public rclcpp::Node {
public:
  SimNode(Params params) : Node("sim_node"), params_(params) {
    std::string ls_topic = params_.lowstate_topic_name;
    std::string odom_topic = params_.odom_topic_name;

    std::string xml_path =
        ament_index_cpp::get_package_share_directory("huro") +
        "/resources/description_files/xml/" + params_.xml_filename;

    // Load robot description in MuJoCo
    mj_model_ = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
    if (!mj_model_) {
      std::string error_msg = "Mujoco XML Model Loading. The XML path is: \n";
      RCLCPP_ERROR_STREAM(this->get_logger(), error_msg << xml_path);
    }
    mj_data_ = mj_makeData(mj_model_);

    // Initialise robot base position
    for (size_t i = 0; i < 3; ++i) {
      mj_data_->qpos[i] = params_.init_base_pos[i];
    }

    // Initialize buffers
    for (size_t i = 0; i < N_MOTORS; ++i) {
      motor_mode_[i] = 0;
      q_des_[i] = 0.0;
      qdot_des_[i] = 0.0;
      tau_ff_[i] = 0.0;
      kp_[i] = 0.0;
      kd_[i] = 0.0;
    }

    lowstate_pub_ = this->create_publisher<LowStateMsg>(ls_topic, 10);
    odom_pub_ = this->create_publisher<OdometryMsg>(odom_topic, 10);
    clock_pub_ =
        this->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);

    lowmcd_sub_ = this->create_subscription<LowCmdMsg>(
        params_.lowcmd_topic_name, 10,
        std::bind(&SimNode::LowCmdHandler, this, std::placeholders::_1));

    // joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
    //     "/joy", 10,
    //     std::bind(&SimNode::JoyHandler, this, std::placeholders::_1));

    // 500Hz control loop
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(params_.sim_dt_ms),
                                std::bind(&SimNode::Step, this));

    // Running time count
    time_s_ = 0;
    mj_step(mj_model_, mj_data_);
  }

  ~SimNode() {}

protected:
  void Step() {
    time_s_ += params_.sim_dt_ms / 1000.0;

    // Calculate control
    for (size_t i = 0; i < N_MOTORS; ++i) {
      mjtNum q_e = q_des_[i] - mj_data_->qpos[7 + i];
      mjtNum qdot_e = qdot_des_[i] - mj_data_->qvel[6 + i];

      mj_data_->ctrl[i] =
          motor_mode_[i] * (kp_[i] * q_e + kd_[i] * qdot_e + tau_ff_[i]);
      // mj_data_->ctrl[i] = 0.0;
    }

    // Step the simulation
    mj_step(mj_model_, mj_data_);

    // Publish the new state
    // OdomMsg: world frame base position and linear velocity
    // LowStateMsg: bodyframe base orientation and angular velocity and
    // joint state
    OdometryMsg odom_msg = GenerateOdometryMsg();
    LowStateMsg lowstate_msg = GenerateLowStateMsg();

    odom_pub_->publish(odom_msg);
    lowstate_pub_->publish(lowstate_msg);

    // Publish clock for use_sim_time synchronization
    rosgraph_msgs::msg::Clock clock_msg;
    int32_t sec = static_cast<int32_t>(time_s_);
    double frac = time_s_ - static_cast<double>(sec);
    clock_msg.clock.sec = sec;
    clock_msg.clock.nanosec = static_cast<uint32_t>(frac * 1e9);
    clock_pub_->publish(clock_msg);
  }

  void LowCmdHandler(std::shared_ptr<LowCmdMsg> message) {
    // Not used in simulation, also breaks go2 api comaptibility
    // mode_machine = (int)message->mode_machine;

    for (size_t i = 0; i < N_MOTORS; ++i) {
      motor_mode_[i] = static_cast<int>(message->motor_cmd[i].mode);
      q_des_[i] = static_cast<mjtNum>(message->motor_cmd[i].q);
      qdot_des_[i] = static_cast<mjtNum>(message->motor_cmd[i].dq);
      tau_ff_[i] = static_cast<mjtNum>(message->motor_cmd[i].tau);
      kp_[i] = static_cast<mjtNum>(message->motor_cmd[i].kp);
      kd_[i] = static_cast<mjtNum>(message->motor_cmd[i].kd);
    }
  }

  // void JoyHandler(std::shared_ptr<sensor_msgs::msg::Joy> message) {}

  OdometryMsg GenerateOdometryMsg() const {
    OdometryMsg odom;

    odom.position[0] = static_cast<float>(mj_data_->qpos[0]);
    odom.position[1] = static_cast<float>(mj_data_->qpos[1]);
    odom.position[2] = static_cast<float>(mj_data_->qpos[2]);

    odom.velocity[0] = static_cast<float>(mj_data_->qvel[0]);
    odom.velocity[1] = static_cast<float>(mj_data_->qvel[1]);
    odom.velocity[2] = static_cast<float>(mj_data_->qvel[2]);

    float qw = static_cast<float>(mj_data_->qpos[3]);
    float qx = static_cast<float>(mj_data_->qpos[4]);
    float qy = static_cast<float>(mj_data_->qpos[5]);
    float qz = static_cast<float>(mj_data_->qpos[6]);
    odom.imu_state.quaternion[0] = qw;
    odom.imu_state.quaternion[1] = qx;
    odom.imu_state.quaternion[2] = qy;
    odom.imu_state.quaternion[3] = qz;

    // Angular velocity
    float omegax = static_cast<float>(mj_data_->qvel[3]);
    float omegay = static_cast<float>(mj_data_->qvel[4]);
    float omegaz = static_cast<float>(mj_data_->qvel[5]);
    odom.imu_state.gyroscope[0] = omegax;
    odom.imu_state.gyroscope[1] = omegay;
    odom.imu_state.gyroscope[2] = omegaz;

    return odom;
  }
  LowStateMsg GenerateLowStateMsg() const {
    LowStateMsg lowstate;

    // Rotation
    float qx = static_cast<float>(mj_data_->qpos[3]);
    float qy = static_cast<float>(mj_data_->qpos[4]);
    float qz = static_cast<float>(mj_data_->qpos[5]);
    float qw = static_cast<float>(mj_data_->qpos[6]);
    lowstate.imu_state.quaternion[0] = qx;
    lowstate.imu_state.quaternion[1] = qy;
    lowstate.imu_state.quaternion[2] = qz;
    lowstate.imu_state.quaternion[3] = qw;

    // angular velocity
    float omegax = static_cast<float>(mj_data_->qvel[3]);
    float omegay = static_cast<float>(mj_data_->qvel[4]);
    float omegaz = static_cast<float>(mj_data_->qvel[5]);
    lowstate.imu_state.gyroscope[0] = omegax;
    lowstate.imu_state.gyroscope[1] = omegay;
    lowstate.imu_state.gyroscope[2] = omegaz;

    // Motor States
    for (size_t i = 0; i < N_MOTORS; ++i) {
      float q = static_cast<float>(mj_data_->qpos[7 + i]);
      float qdot = static_cast<float>(mj_data_->qvel[6 + i]);
      float qddot = static_cast<float>(mj_data_->qacc[6 + i]);

      lowstate.motor_state[i].q = q;
      lowstate.motor_state[i].dq = qdot;
      lowstate.motor_state[i].ddq = qddot;
    }

    // Foot contact forces from MuJoCo contact data for go2
    if constexpr (N_MOTORS == 12) {
      for (size_t i = 0; i < 4; ++i) {
        lowstate.foot_force[i] = 0;
        lowstate.foot_force_est[i] = 0;
      }

      // Get foot geom IDs for Go2 (collisions geometries)
      std::vector<std::string> foot_geom_names = {"FL", "FR", "RL", "RR"};

      // Sum contact forces for each foot
      for (int i = 0; i < mj_data_->ncon; ++i) {
        const mjContact &con = mj_data_->contact[i];

        // Check contact
        for (size_t foot_idx = 0; foot_idx < foot_geom_names.size();
             ++foot_idx) {
          int foot_geom_id = mj_name2id(mj_model_, mjOBJ_GEOM,
                                        foot_geom_names[foot_idx].c_str());

          if (con.geom1 == foot_geom_id || con.geom2 == foot_geom_id) {
            // Get contact force in world frame
            mjtNum contact_force[6];
            mj_contactForce(mj_model_, mj_data_, i, contact_force);

            // Sum normal force magnitude (contact_force[0] is normal force in
            // contact frame)
            float normal_force = static_cast<float>(std::abs(contact_force[0]));
            lowstate.foot_force[foot_idx] += static_cast<int16_t>(normal_force);
            lowstate.foot_force_est[foot_idx] +=
                static_cast<int16_t>(normal_force);
          }
        }
      }
    }

    return lowstate;
  }

protected:
  Params params_;

  double time_s_; // Running time count (in seconds)
  int mode_machine;

  std::shared_ptr<rclcpp::Publisher<LowStateMsg>> lowstate_pub_;
  std::shared_ptr<rclcpp::Publisher<OdometryMsg>> odom_pub_;
  std::shared_ptr<rclcpp::Publisher<rosgraph_msgs::msg::Clock>> clock_pub_;
  std::shared_ptr<rclcpp::Subscription<LowCmdMsg>> lowmcd_sub_;
  // std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Joy>> joy_sub_;
  std::shared_ptr<rclcpp::TimerBase> timer_;

  mjModel *mj_model_;
  mjData *mj_data_;

  std::array<int, N_MOTORS> motor_mode_;
  std::array<double, N_MOTORS> q_des_;
  std::array<double, N_MOTORS> qdot_des_;
  std::array<double, N_MOTORS> tau_ff_;
  std::array<double, N_MOTORS> kp_;
  std::array<double, N_MOTORS> kd_;
};
} // namespace huro
#endif // HURO_SIM_NODE_H_
