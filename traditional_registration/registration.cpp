#include <string>
#include <ctime>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


void print4x4Matrix(const Eigen::Matrix4d &matrix) {
    printf("[[%f %f %f %f]\n[%f %f %f %f]\n[%f %f %f %f]\n[%f %f %f %f]]\n",
           matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3),
           matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3),
           matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3),
           matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3));
}


void pointsToCloud(const double *points,
                   const size_t size,
                   PointCloudT &cloud) {
    cloud.width = size;
    cloud.height = 1;
    cloud.resize(size);
    size_t index = 0;
    for (auto &it : cloud) {
        it.x = (float) points[index];
        it.y = (float) points[index + 1];
        it.z = (float) points[index + 2];
        index += 3;
    }
}


bool next_iteration = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event) {
    if (event.getKeySym() == "space" && event.keyDown())
        next_iteration = true;
}


template<class T>
Eigen::Matrix4d plot(T &algorithm,
         PointCloudT::Ptr &source,
         const PointCloudT::ConstPtr &target) {

    int iteration = algorithm.getMaximumIterations();
    algorithm.align(*source);
    algorithm.setMaximumIterations(1);
    Eigen::Matrix4d transformation = algorithm.getFinalTransformation().template cast<double>();

    pcl::visualization::PCLVisualizer viewer("Registration");

    int v1(0);
    viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_in_color_h(target, 255, 255, 255);
    viewer.addPointCloud(target, target_in_color_h, "target", v1);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_in_color_h(source, 180, 20, 20);
    viewer.addPointCloud(source, source_in_color_h, "source", v1);

    viewer.addText("White: Target point cloud\nRed: Source point cloud", 10, 15, 16, 1.0, 1.0, 1.0, "info", v1);

    std::stringstream ss;
    ss << "Iteration: " << iteration << "\nScore: " << algorithm.getFitnessScore();
    std::string text = ss.str();
    viewer.addText(text, 10, 65, 16, 1.0, 1.0, 1.0, "text", v1);

    viewer.setBackgroundColor(0.0, 0.0, 0.0, v1);
    viewer.setCameraPosition(300, 300, 300, 0, 0, 0, 0, 0, 1);
    viewer.setSize(600, 600);
    viewer.addCoordinateSystem(1.0);
    viewer.registerKeyboardCallback(&keyboardEventOccurred);

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        if (next_iteration) {
            algorithm.align(*source);
            iteration++;
            if (algorithm.hasConverged()) {
                std::cout << "\nTransformation " << iteration << ": Source -> Target" << std::endl;
                transformation *= algorithm.getFinalTransformation().template cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
                print4x4Matrix(transformation);  // Print the transformation between original pose and current pose

                ss.str("");
                ss << "Iteration: " << iteration << "\nScore: " << algorithm.getFitnessScore();
                text = ss.str();
                viewer.updateText(text, 10, 65, 16, 1.0, 1.0, 1.0, "text");

                viewer.updatePointCloud(source, source_in_color_h, "source");
            } else {
                PCL_ERROR ("\nICP has not converged.\n");
            }
        } next_iteration = false;
    }
    return transformation;
}


void uniformSubsampling(const PointCloudT::ConstPtr &input,
                        PointCloudT &output,
                        double radius = 0.1) {
    pcl::UniformSampling<PointT> us;
    us.setInputCloud(input);
    us.setRadiusSearch(radius);
    us.filter(output);
}


extern "C" double ndt(const double *sourcePoints,
                   const size_t sourceSize,
                   const double *targetPoints,
                   const size_t targetSize,
                   double *transformation,
                   int nrIterations = 25,
                   double distanceThreshold = 1.0,
                   double epsilon = 0.01,
                   double inlierThreshold = 0.05,
                   double downsample = 0,
                   bool visualize = false,
                   float resolution = 1.0,
                   double stepSize = 0.1,
                   float voxelize = 0) {

    PointCloudT::Ptr sourceCloud(new PointCloudT);
    PointCloudT::Ptr coarseSourceCloud(new PointCloudT);
    PointCloudT::Ptr voxelizedSourceCloud(new PointCloudT);
    PointCloudT::Ptr finalSourceCloud(new PointCloudT);
    pointsToCloud(sourcePoints, sourceSize, *sourceCloud);
    *finalSourceCloud = *sourceCloud;

    PointCloudT::Ptr targetCloud(new PointCloudT);
    PointCloudT::Ptr coarseTargetCloud(new PointCloudT);
    PointCloudT::Ptr finalTargetCloud(new PointCloudT);
    pointsToCloud(targetPoints, targetSize, *targetCloud);
    *finalTargetCloud = *targetCloud;

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setMaximumIterations(nrIterations);
    ndt.setMaxCorrespondenceDistance(distanceThreshold);
    ndt.setTransformationEpsilon(epsilon);
    ndt.setRANSACOutlierRejectionThreshold(inlierThreshold);
    //ndt.setRANSACIterations();
    ndt.setResolution(resolution);
    ndt.setStepSize(stepSize);

    if (downsample > 0) {
        uniformSubsampling(sourceCloud, *coarseSourceCloud, downsample);
        *finalSourceCloud = *coarseSourceCloud;

        uniformSubsampling(targetCloud, *coarseTargetCloud, downsample);
        *finalTargetCloud = *coarseTargetCloud;
    }
    if (voxelize > 0) {
        pcl::ApproximateVoxelGrid<PointT> voxelFilter;
        voxelFilter.setLeafSize(voxelize, voxelize, voxelize);
        voxelFilter.setInputCloud(finalSourceCloud);
        voxelFilter.filter(*voxelizedSourceCloud);
        *finalSourceCloud = *voxelizedSourceCloud;
    }

    ndt.setInputSource(finalSourceCloud);
    ndt.setInputTarget(finalTargetCloud);

    Eigen::Matrix4d finalTransform = Eigen::Matrix4d::Identity();
    if (visualize) {
        finalTransform = plot<pcl::NormalDistributionsTransform<PointT, PointT>>(ndt, finalSourceCloud, finalTargetCloud);
    } else {
        ndt.align(*finalSourceCloud);
        finalTransform = ndt.getFinalTransformation().cast<double>();
    }

    if (ndt.hasConverged()) {
        transformation[0] = finalTransform(0, 0);
        transformation[1] = finalTransform(0, 1);
        transformation[2] = finalTransform(0, 2);
        transformation[3] = finalTransform(0, 3);

        transformation[4] = finalTransform(1, 0);
        transformation[5] = finalTransform(1, 1);
        transformation[6] = finalTransform(1, 2);
        transformation[7] = finalTransform(1, 3);

        transformation[8] = finalTransform(2, 0);
        transformation[9] = finalTransform(2, 1);
        transformation[10] = finalTransform(2, 2);
        transformation[11] = finalTransform(2, 3);
        return ndt.getFitnessScore();
    }
    return (-1);
}


extern "C" double icp(const double *sourcePoints,
                   const size_t sourceSize,
                   const double *targetPoints,
                   const size_t targetSize,
                   double *transformation,
                   int nrIterations = 25,
                   double distanceThreshold = 1.0,
                   double epsilon = 0.01,
                   double inlierThreshold = 0.05,
                   double downsample = 0,
                   bool visualize = false) {

    PointCloudT::Ptr sourceCloud(new PointCloudT);
    PointCloudT::Ptr coarseSourceCloud(new PointCloudT);
    PointCloudT::Ptr finalSourceCloud(new PointCloudT);
    pointsToCloud(sourcePoints, sourceSize, *sourceCloud);
    *finalSourceCloud = *sourceCloud;

    PointCloudT::Ptr targetCloud(new PointCloudT);
    PointCloudT::Ptr coarseTargetCloud(new PointCloudT);
    PointCloudT::Ptr finalTargetCloud(new PointCloudT);
    pointsToCloud(targetPoints, targetSize, *targetCloud);
    *finalTargetCloud = *targetCloud;

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(nrIterations);
    icp.setMaxCorrespondenceDistance(distanceThreshold);
    icp.setTransformationEpsilon(epsilon);
    icp.setRANSACOutlierRejectionThreshold(inlierThreshold);

    if (downsample > 0) {
        uniformSubsampling(sourceCloud, *coarseSourceCloud, downsample);
        *finalSourceCloud = *coarseSourceCloud;

        uniformSubsampling(targetCloud, *coarseTargetCloud, downsample);
        *finalTargetCloud = *coarseTargetCloud;
    }

    icp.setInputSource(finalSourceCloud);
    icp.setInputTarget(finalTargetCloud);

    Eigen::Matrix4d finalTransform = Eigen::Matrix4d::Identity();
    if (visualize) {
        finalTransform = plot<pcl::IterativeClosestPoint<PointT, PointT>>(icp, finalSourceCloud, finalTargetCloud);
    } else {
        icp.align(*finalSourceCloud);
        finalTransform = icp.getFinalTransformation().cast<double>();
    }

    if (icp.hasConverged()) {
        transformation[0] = finalTransform(0, 0);
        transformation[1] = finalTransform(0, 1);
        transformation[2] = finalTransform(0, 2);
        transformation[3] = finalTransform(0, 3);

        transformation[4] = finalTransform(1, 0);
        transformation[5] = finalTransform(1, 1);
        transformation[6] = finalTransform(1, 2);
        transformation[7] = finalTransform(1, 3);

        transformation[8] = finalTransform(2, 0);
        transformation[9] = finalTransform(2, 1);
        transformation[10] = finalTransform(2, 2);
        transformation[11] = finalTransform(2, 3);
        return icp.getFitnessScore();
    }
    return (-1);
}
