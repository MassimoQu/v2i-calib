#include <Eigen/Dense>
#include <vector>
#include "IoU_utils.hpp"

bool inside(const Point& p, const Point& cp1, const Point& cp2) {
    return (cp2.x() - cp1.x()) * (p.y() - cp1.y()) > (cp2.y() - cp1.y()) * (p.x() - cp1.x());
}

Point computeIntersection(const Point& cp1, const Point& cp2, const Point& s, const Point& e) {
    Point dc = cp1 - cp2;
    Point dp = s - e;
    double n1 = cp1.x() * cp2.y() - cp1.y() * cp2.x();
    double n2 = s.x() * e.y() - s.y() * e.x();
    double n3 = 1.0f / (dc.x() * dp.y() - dc.y() * dp.x());
    return Point((n1 * dp.x() - n2 * dc.x()) * n3, (n1 * dp.y() - n2 * dc.y()) * n3);
}

std::vector<Point> polygonClip(const std::vector<Point>& subjectPolygon, const std::vector<Point>& clipPolygon) {
    std::vector<Point> outputList = subjectPolygon;
    Point cp1 = clipPolygon.back();

    for (const auto& cp2 : clipPolygon) {
        std::vector<Point> inputList = outputList;
        outputList.clear();
        Point s = inputList.back();

        for (const auto& e : inputList) {
            if (inside(e, cp1, cp2)) {
                if (!inside(s, cp1, cp2)) {
                    outputList.push_back(computeIntersection(cp1, cp2, s, e));
                }
                outputList.push_back(e);
            } else if (inside(s, cp1, cp2)) {
                outputList.push_back(computeIntersection(cp1, cp2, s, e));
            }
            s = e;
        }
        cp1 = cp2;
        if (outputList.empty()) {
            break;
        }
    }
    return outputList;
}

double polyArea(const std::vector<Point>& points) {
    double area = 0.0f;
    int j = points.size() - 1;
    for (int i = 0; i < points.size(); i++) {
        area += (points[j].x() + points[i].x()) * (points[j].y() - points[i].y());
        j = i;
    }
    return std::abs(area / 2.0f);
}

double box3dVol(const Eigen::Matrix<double, 8, 3>& corners) {
    double a = (corners.row(0) - corners.row(1)).norm();
    double b = (corners.row(1) - corners.row(2)).norm();
    double c = (corners.row(0) - corners.row(4)).norm();
    return a * b * c;
}

std::pair<double, double> box3dIoU(const Eigen::Matrix<double, 8, 3>& corners1, const Eigen::Matrix<double, 8, 3>& corners2) {
    std::vector<Point> rect1 = {{corners1(3, 0), corners1(3, 1)},
                                {corners1(2, 0), corners1(2, 1)},
                                {corners1(1, 0), corners1(1, 1)},
                                {corners1(0, 0), corners1(0, 1)}};
    std::vector<Point> rect2 = {{corners2(3, 0), corners2(3, 1)},
                                {corners2(2, 0), corners2(2, 1)},
                                {corners2(1, 0), corners2(1, 1)},
                                {corners2(0, 0), corners2(0, 1)}};

    double area1 = polyArea(rect1);
    double area2 = polyArea(rect2);

    std::vector<Point> inter = polygonClip(rect1, rect2);
    double interArea = polyArea(inter);

    double iou2d = interArea / (area1 + area2 - interArea);

    double ymax = std::min(corners1(4, 2), corners2(4, 2));
    double ymin = std::max(corners1(0, 2), corners2(0, 2));

    double interVol = interArea * std::max(0.0, ymax - ymin);
    double vol1 = box3dVol(corners1);
    double vol2 = box3dVol(corners2);

    double iou = interVol / (vol1 + vol2 - interVol);
    return {iou, iou2d};
}

double calculate3DIoU(const Eigen::Matrix<double, 8, 3>& corners1, const Eigen::Matrix<double, 8, 3>& corners2) {
    // auto [iou3d, _] = box3dIoU(corners1, corners2);
    std::pair<double, double> iou3d_iou2d = box3dIoU(corners1, corners2);
    double iou3d = iou3d_iou2d.first;
    return iou3d;
}
