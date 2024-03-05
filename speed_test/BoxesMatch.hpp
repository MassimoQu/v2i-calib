#include <Eigen/Dense>
#include <vector>
#include <map>
#include <algorithm> // For std::sort, std::transform
#include <numeric> // For std::iota
#include <iostream>

#include "hungarian.hpp"
#include "IoU_utils.hpp"

typedef Eigen::Matrix<double, 8, 3> Box3D;

class BoxObject {
    public:
        BoxObject(const Box3D& box3d_8_3, const std::string& type) : box3d_8_3(box3d_8_3), type(type) {}

        Box3D box3d_8_3;
        std::string type;

        std::string get_type() const {
            return type;
        }

        Box3D get_box3d_8_3() const {
            return box3d_8_3;
        }
};

// extrinsic convert
std::pair<Eigen::Matrix3d, Eigen::Vector3d> convert_T_to_Rt(const Eigen::Matrix4d& T);

Eigen::Matrix4d convert_Rt_to_T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

// implement T to 3dbox
Eigen::MatrixXd implement_R_t_points_n_3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::MatrixXd& points);

Box3D implement_R_t_3dbox_n_8_3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Box3D& box3d_8_3);

std::vector<BoxObject> implement_R_t_3dbox_object_list(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const std::vector<BoxObject>& box_object_list);

std::vector<BoxObject> implement_T_3dbox_object_list(const Eigen::Matrix4d& T, const std::vector<BoxObject>& box_object_list);

// get extrinsic from 3dbox
Eigen::MatrixXd subtractMean(const Eigen::MatrixXd& points);

Eigen::Matrix4d get_extrinsic_from_two_3dbox_object(const BoxObject& boxObject1, const BoxObject& boxObject2);

// Equivalent to get_Yscore in Python
double getYscore(const std::vector<BoxObject>& infraBoxes, const std::vector<BoxObject>& vehicleBoxes);

std::vector<std::pair<std::pair<int, int>, double>> getMatchesWithScore(const std::vector<BoxObject>& infraBoxes, const std::vector<BoxObject>& vehicleBoxes);