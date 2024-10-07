#include <Eigen/Dense>
#include <vector>
#include <map>
#include <algorithm> // For std::sort, std::transform
#include <numeric> // For std::iota
#include <iostream>

#include "hungarian.hpp"
#include "IoU_utils.hpp"
#include "BoxesMatch.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // Provides py::scoped_interpreter and py::exec
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Automatic conversion of STL containers

namespace py = pybind11;

// extrinsic convert
std::pair<Eigen::Matrix3d, Eigen::Vector3d> convert_T_to_Rt(const Eigen::Matrix4d& T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    return {R, t};
}

Eigen::Matrix4d convert_Rt_to_T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}


// implement T to 3dbox
// Eigen::MatrixXd implement_R_t_points_n_3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::MatrixXd& points) {
//     Eigen::MatrixXd points_transposed = points.transpose();
//     Eigen::MatrixXd converted_points = (R * points_transposed).colwise() + t;
//     return converted_points.transpose();
// }

Box3D implement_R_t_points_8_3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::MatrixXd& points) {
    // Ensure the input 'points' matrix has the correct size
    assert(points.rows() == 8 && points.cols() == 3 && "The input points matrix must be of size 8x3.");

    // Perform the transformation
    Eigen::MatrixXd points_transposed = points.transpose();
    Eigen::MatrixXd converted_points = (R * points_transposed).colwise() + t;

    // Cast the result to a fixed-size matrix (8x3)
    Eigen::Matrix<double, 8, 3> result = converted_points.transpose().cast<double>();

    return result;
}


Box3D implement_R_t_3dbox_n_8_3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Box3D& box3d_8_3) {
    return implement_R_t_points_8_3(R, t, box3d_8_3);
}

std::vector<BoxObject> implement_R_t_3dbox_object_list(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const std::vector<BoxObject>& boxObjectList) {
    std::vector<BoxObject> convertedBoxObjectList;
    for (const auto& boxObject : boxObjectList) {
        Box3D convertedBox3D = implement_R_t_points_8_3(R, t, boxObject.get_box3d_8_3());
        convertedBoxObjectList.emplace_back(convertedBox3D, boxObject.get_type());
    }
    return convertedBoxObjectList;
}

// std::vector<BoxObject> implement_R_t_3dbox_object_list(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const std::vector<BoxObject>& box_object_list) {
//     std::vector<BoxObject> converted_box3d_list;
//     std::transform(box_object_list.begin(), box_object_list.end(), std::back_inserter(converted_box3d_list), [&](const BoxObject& box_object) {
//         return implement_R_t_3dbox_n_8_3(R, t, box_object.get_box3d_8_3());
//     });
//     return converted_box3d_list;
// }

std::vector<BoxObject> implement_T_3dbox_object_list(const Eigen::Matrix4d& T, const std::vector<BoxObject>& box_object_list) {
    // auto [R, t] = convert_T_to_Rt(T);
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> Rt = convert_T_to_Rt(T);
    Eigen::Matrix3d R = Rt.first;
    Eigen::Vector3d t = Rt.second;
    return implement_R_t_3dbox_object_list(R, t, box_object_list);
}

Eigen::Matrix4d get_extrinsic_from_two_3dbox_object(const BoxObject& boxObject1, const BoxObject& boxObject2) {
    Eigen::MatrixXd points1 = boxObject1.get_box3d_8_3();
    Eigen::MatrixXd points2 = boxObject2.get_box3d_8_3();

    Eigen::Vector3d centroid1 = points1.colwise().mean();
    Eigen::Vector3d centroid2 = points2.colwise().mean();

    points1 = points1.rowwise() - centroid1.transpose();
    points2 =  points2.rowwise() - centroid2.transpose();
    // std::cout<<"points1: "<<std::endl;
    // std::cout<<points1<<std::endl;
    // std::cout<<"points2: "<<std::endl;
    // std::cout<<points2<<std::endl;

    Eigen::Matrix3d H = points1.transpose() * points2;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV); // 这个策略和python的不太一样,具体是Vt
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d Vt = svd.matrixV();
    // std::cout<<"U: "<<std::endl;
    // std::cout<<U<<std::endl;
    // std::cout<<"Vt: "<<std::endl;
    // std::cout<<Vt<<std::endl;

    Eigen::Matrix3d R = Vt.transpose() * U.transpose();
    if (R.determinant() < 0) {
        Vt.row(2) *= -1;
        R = Vt.transpose() * U.transpose();
    }

    
    Eigen::Vector3d t = -(R * centroid2) + centroid1;

    // std::cout<<"centroid1: "<<std::endl;
    // std::cout<<centroid1<<std::endl;
    // std::cout<<"centroid2: "<<std::endl;    
    // std::cout<<centroid2<<std::endl;

    // std::cout<<"R: "<<std::endl;
    // std::cout<<R<<std::endl;
    // std::cout<<"t: "<<std::endl;
    // std::cout<<t<<std::endl;

    return convert_Rt_to_T(R, t);
}

// Equivalent to get_Yscore in Python
double getYscore(const std::vector<BoxObject>& infraBoxes, const std::vector<BoxObject>& vehicleBoxes) {
    std::map<std::pair<int, int>, double> correspondingIoUDict;
    double Y = -1.0;

    for (int i = 0; i < infraBoxes.size(); ++i) {
        for (int j = 0; j < vehicleBoxes.size(); ++j) {
            if (infraBoxes[i].get_type() == vehicleBoxes[j].get_type()) {
                double box3dIoUScore = calculate3DIoU(infraBoxes[i].get_box3d_8_3(), vehicleBoxes[j].get_box3d_8_3());
                if (box3dIoUScore > 0) {
                    correspondingIoUDict[std::make_pair(i, j)] = box3dIoUScore;
                }
            }
        }
    }

    if (!correspondingIoUDict.empty()) {
        Y = std::accumulate(begin(correspondingIoUDict), end(correspondingIoUDict), 0.0, 
                            [](double value, const std::map<std::pair<int, int>, double>::value_type& p){ return value + p.second; });
    }

    int totalNum = std::min(infraBoxes.size(), vehicleBoxes.size());
    return totalNum > 0 && Y > 0 ? Y / totalNum : 0.0;
}


std::vector<std::pair<std::pair<int, int>, double>> getMatchesWithScore(const std::vector<BoxObject>& infraBoxes, const std::vector<BoxObject>& vehicleBoxes) {
    int infraNodeNum = infraBoxes.size();
    int vehicleNodeNum = vehicleBoxes.size();

    // Calculate KP
    Eigen::MatrixXd KP = Eigen::MatrixXd::Zero(infraNodeNum, vehicleNodeNum);
    for (int i = 0; i < infraNodeNum; ++i) {
        for (int j = 0; j < vehicleNodeNum; ++j) {
            if (infraBoxes[i].get_type() != vehicleBoxes[j].get_type()) {
                continue;
            }
            Eigen::Matrix4d T = get_extrinsic_from_two_3dbox_object(infraBoxes[i], vehicleBoxes[j]);

            // std::cout<<"T: "<<i<<std::endl;
            // std::cout<<T<<std::endl;

            // std::cout<<"infraBoxes[i].get_box3d_8_3(): "<<std::endl;
            // std::cout<<infraBoxes[i].get_box3d_8_3()<<std::endl;
            // std::cout<<"vehicleBoxes[j].get_box3d_8_3(): "<<std::endl;
            // std::cout<<vehicleBoxes[j].get_box3d_8_3()<<std::endl;
            std::vector<BoxObject> convertedInfraBoxes = implement_T_3dbox_object_list(T, infraBoxes);
            // std::cout<<"convertedInfraBoxes[i].get_box3d_8_3(): "<<std::endl;
            // std::cout<<convertedInfraBoxes[i].get_box3d_8_3()<<std::endl;

            KP(i, j) = getYscore(convertedInfraBoxes, vehicleBoxes) * 100;
            // if (i >= 0 && j >= 0) break;
        }
        // if (i >= 0 ) break;
    }

    // print KP
    // std::cout<<"KP: "<<std::endl;
    // std::cout<<KP<<std::endl;

    // Normalize KP
    int maxVal = KP.maxCoeff();
    int minVal = KP.minCoeff();
    for (int i = 0; i < infraNodeNum; ++i) {
        for (int j = 0; j < vehicleNodeNum; ++j) {
            if (KP(i, j) != 0) {
                KP(i, j) = (KP(i, j) - minVal) / (maxVal - minVal) * 10;
            }
        }
    }

    // print KP
    // std::cout<<"normalized KP: "<<std::endl;
    // std::cout<<KP<<std::endl;

    // Calculate matches using Hungarian algorithm
    Eigen::VectorXi resultVector(KP.rows());
    resultVector.fill(0);
    findMatching(KP, resultVector);

    // std::cout<<"resultVector: "<<std::endl;
    // std::cout<<resultVector<<std::endl;

    // Construct matches and score dictionary
    std::vector<std::pair<std::pair<int, int>, double>> matchesScoreVector;
    for (int i = 0; i < resultVector.size(); ++i) {
        if (resultVector(i) >= 0 && resultVector(i) < vehicleNodeNum && KP(i, resultVector(i)) != 0) {
            matchesScoreVector.emplace_back(std::make_pair(std::make_pair(i, resultVector(i)), KP(i, resultVector(i))));
        }
    }

    // std::cout<<"matchesScoreVector: "<<std::endl;
    // int cnt = 0;
    // for (const auto& match : matchesScoreVector) {
    //     std::cout << "match: " << cnt  << "," << match.first << ", Score: " << match.second << std::endl;
    //     cnt++;
    // }
    // std::cout<<"matchesScoreVector.size(): "<<matchesScoreVector.size()<<std::endl;

    // Sort matches by score
    std::sort(matchesScoreVector.begin(), matchesScoreVector.end(), [](const std::pair<std::pair<int, int>, double>& a, const std::pair<std::pair<int, int>, double>& b) {
        return a.second > b.second;
    });

    return matchesScoreVector;
}


// Helper function to convert a single BBox3d Python object to a BoxObject C++ object
BoxObject convertBBox3dToBoxObject(const py::handle& bbox3d_py) {
    // Extract bbox_type (string) from the Python object
    std::string type = bbox3d_py.attr("bbox_type").cast<std::string>();

    // Extract bbox3d_8_3 (ndarray) from the Python object and convert it to Eigen::Matrix<double, 8, 3>
    py::array_t<double> bbox3d_8_3_py = bbox3d_py.attr("bbox3d_8_3").cast<py::array_t<double>>();
    // Eigen::Matrix<double, 8, 3> bbox3d_8_3_cpp = Eigen::Map<const Eigen::Matrix<double, 8, 3, Eigen::RowMajor>>(bbox3d_8_3_py.data());

    // Assuming the numpy array is 2D with shape (8, 3) and in C-contiguous layout
    auto bbox3d_8_3_cpp = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        bbox3d_8_3_py.data(), bbox3d_8_3_py.shape(0), bbox3d_8_3_py.shape(1));

    // Construct and return the C++ BoxObject
    return BoxObject(bbox3d_8_3_cpp, type);
}


int main() {

    // std::cout<<"Start"<<std::endl;

    // Start the Python interpreter
    py::scoped_interpreter guard{};
    // std::cout<<"start interpreter"<<std::endl;

    // Ensure the directory containing CooperativeReader.py is in the Python path
    py::exec(R"(
import sys
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/reader')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/utils')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/corresponding')
try:
    import CooperativeReader
    print("Module imported successfully")
except Exception as e:
    print(f"Failed to import module: {e}")

)");
    // std::cout<<"add path"<<std::endl;

    // Import the Python module containing the CooperativeReader class
    py::module cooperative_module = py::module::import("CooperativeReader");
    // std::cout<<"import module"<<std::endl;

    // Initialize the CooperativeReader class
    py::object cooperative_reader = cooperative_module.attr("CooperativeReader")("007489", "000289", "/mnt/c/Users/10612/Downloads/cooperative_data");
    // std::cout<<"init reader"<<std::endl;

    // Call the method and get the return value as a py::tuple
    py::tuple boxes_tuple = cooperative_reader.attr("get_cooperative_infra_vehicle_boxes_object_list")();
    // Ensure that the returned object is indeed a tuple and has exactly two elements
    if (!boxes_tuple || boxes_tuple.size() != 2) {
        throw std::runtime_error("Expected a tuple of size 2");
    }
    // Extract infraBoxesList and vehicleBoxesList from the tuple
    py::list infraBoxesList = boxes_tuple[0].cast<py::list>();
    py::list vehicleBoxesList = boxes_tuple[1].cast<py::list>();
    // std::cout<<"get infraBoxesList & vehicleBoxesList"<<std::endl;
    // std::cout<<"infraBoxesList:"<<std::endl;
    // for (const auto& bbox3d_py : infraBoxesList) {
    //     std::cout << bbox3d_py.attr("bbox_type").cast<std::string>() << std::endl;
    //     std::cout << bbox3d_py.attr("bbox3d_8_3").cast<py::array_t<double>>() << std::endl;
    // }

    // You need to convert the Python lists to C++ vectors before calling the function
    std::vector<BoxObject> infraBoxesCpp, vehicleBoxesCpp;

    // Convert Python lists to C++ vectors...
    for (const auto& bbox3d_py : infraBoxesList) {
        infraBoxesCpp.push_back(convertBBox3dToBoxObject(bbox3d_py));
    }
    for (const auto& bbox3d_py : vehicleBoxesList) {
        vehicleBoxesCpp.push_back(convertBBox3dToBoxObject(bbox3d_py));
    }

    std::cout<<"infraBoxesCpp.size(): "<<infraBoxesCpp.size()<<std::endl;
    std::cout<<"vehicleBoxesCpp.size(): "<<vehicleBoxesCpp.size()<<std::endl;

    // Call the C++ function with the converted vectors
    auto matches = getMatchesWithScore(infraBoxesCpp, vehicleBoxesCpp);

    // Print the result
    // for (const auto& match : matches) {
    //     std::cout << "match: " << match.first << ", Score: " << match.second << std::endl;
    // }

    return 0;
}
