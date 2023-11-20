import mayavi.mlab as mlab
# import open3d as mlab

class PointCloudVisualizer():
    
    def __init__(self) -> None:
        pass


    def draw_pointclouds(self, fig, pointclouds, colors=None):
        for pointcloud, color in zip(pointclouds, colors):
            mlab.points3d(
                pointcloud[:, 0],
                pointcloud[:, 1],
                pointcloud[:, 2],
                pointcloud[:, 2],
                mode="point",
                colormap="spectral",
                color=color,
                figure=fig,
            )

    def plot_pointclouds(self, pointclouds, colors):
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        self.draw_pointclouds(fig, pointclouds, colors=colors)
        mlab.show()


    