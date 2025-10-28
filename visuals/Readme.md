<!-- ## Visualization Results -->

<!-- <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px;">
  <div style="flex: 1; min-width: 300px;">
    <h3>Calibration Quality Comparison</h3>
    <video controls width="100%" poster="thumbnail_merged.jpg">
      <source src="merged_output.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p style="margin-top: 10px; font-size: 0.95rem;">
      This visualization `visuals/merged_output.mp4` compares the bounding boxes (obtained via PointPillars on DAIR-V2X) 
      after registration using extrinsic parameters from V2I-Calib++ (left) versus the 
      official DAIR dataset parameters (right). The point cloud overlay demonstrates the 
      improved alignment accuracy achieved by our method.
    </p>
  </div>
</div>
<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px;">
  <div style="flex: 1; min-width: 300px;">
    <h3>Latency Scenario Performance</h3>
    <video controls width="100%" poster="thumbnail_delay.jpg">
      <source src="delay_scene.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p style="margin-top: 10px; font-size: 0.95rem;">
      (`visuals/delay_scene.mp4`) Extrinsic parameters results from V2I-Calib++ using PointPillars detection boxes under varying latency conditions. We observed an interesting phenomenon: instead of a cliff-like performance decline with increasing latency, the results exhibited complementary trade-offs across different frames. Analysis suggests that at low latency, occlusion-induced detection box invisibility in certain frame pairs adversely affected registration quality. As latency increased, temporal shifting alleviated local occlusion patterns, improving detection in these frames. Concurrently, our method's inherent tolerance to temporal asynchrony enhanced performance in previously suboptimal low-latency scenarios. The visualization highlights our method's robustness in asynchronous real-world conditions.
  </div>
</div> -->


## Visualization Results

### Calibration Quality Comparison

[![Calibration Quality Comparison Video](thumbnail_merged.png)](https://github.com/MassimoQu/v2i-calib/blob/main/visuals/merged_output.mp4)

This visualization (`visuals/merged_output.mp4`) compares the bounding boxes (obtained via PointPillars on DAIR-V2X) after registration using extrinsic parameters from V2I-Calib++ (left) versus the official DAIR dataset parameters (right). The point cloud overlay demonstrates the improved alignment accuracy achieved by our method.

---

### Latency Scenario Performance

[![Latency Scenario Performance Video](thumbnail_delay.png)](https://github.com/MassimoQu/v2i-calib/blob/main/visuals/delay_scene.mp4)

(`visuals/delay_scene.mp4`) Extrinsic parameters results from V2I-Calib++ using PointPillars detection boxes under varying latency conditions. We observed an interesting phenomenon: instead of a cliff-like performance decline with increasing latency, the results exhibited complementary trade-offs across different frames. Analysis suggests that at low latency, occlusion-induced detection box invisibility in certain frame pairs adversely affected registration quality. As latency increased, temporal shifting alleviated local occlusion patterns, improving detection in these frames. Concurrently, our method's inherent tolerance to temporal asynchrony enhanced performance in previously suboptimal low-latency scenarios. The visualization highlights our method's robustness in asynchronous real-world conditions.