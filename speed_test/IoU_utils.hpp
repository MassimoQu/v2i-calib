#ifndef BOX3DIOU_HPP
#define BOX3DIOU_HPP

#include <Eigen/Dense>
#include <vector>

typedef Eigen::Vector2f Point;

// Check if a point is inside the clip polygon
bool inside(const Point& p, const Point& cp1, const Point& cp2);

// Compute the intersection point of two line segments
Point computeIntersection(const Point& cp1, const Point& cp2, const Point& s, const Point& e);

// Clip a polygon with another polygon
std::vector<Point> polygonClip(const std::vector<Point>& subjectPolygon, const std::vector<Point>& clipPolygon);

// Calculate the area of a polygon
double polyArea(const std::vector<Point>& points);

// Calculate the volume of a 3D box
double box3dVol(const Eigen::Matrix<double, 8, 3>& corners);

// Calculate the 3D IoU and BEV (Bird's Eye View) IoU between two 3D boxes
std::pair<double, double> box3dIoU(const Eigen::Matrix<double, 8, 3>& corners1, const Eigen::Matrix<double, 8, 3>& corners2);

double calculate3DIoU(const Eigen::Matrix<double, 8, 3>& corners1, const Eigen::Matrix<double, 8, 3>& corners2);

#endif // BOX3DIOU_HPP

