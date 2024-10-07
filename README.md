# V2I-CALIB/V2I-CALIB++: An Object-Level, Real-Time Point Cloud Global Registration Framework for V2I/V2X Applications

<div align="center">
    <img src="./static/images/V2I-CALIB++_workflow.png" alt="V2I-CALIB++_Workflow" width="88%">
  </div>

<h3 align="center">
  <a href="https://arxiv.org/abs/2204.05575">V2X-Calib arXiv</a> 
</h3>

## Table of Contents:
1. [News](#news)
2. [Experimental Comparison](#experimental-comparison)
3. [Getting Started](#getting-started)
4. [Acknowledgement](#acknowledgment)
5. [Citation](#citation)


## News
* [2024/09/13] V2I-CALIB++ is comming soon!
* [2024/06/30] V2I-CALIB is accepted by IROS 2024!

## Experimental Comparison

We conducted experiments comparing V2I-Calib and V2I-Calib++ against well-performed point cloud Global Registration methods, using two widely recognized V2X datasets: DAIR-V2X and V2X-Sim. The results are as follows.

* <a href="https://github.com/ai4ce/V2X-Sim">V2X-Sim</a> (Homologous LiDARs)
    <div align="left">
    <table>
        <tr align="center">
            <td rowspan="2">Method</td>
            <td rowspan="2">RRE(°)</td>
            <td rowspan="2">RTE(m)</td>
            <td colspan="2" align="center">Success Rate(%)</td>
            <td rowspan="2">Time (s)</td>
        </tr>
        <tr align="center">
            <td>@1</td>
            <td>@2</td>
        </tr>
        <tr align="center">
            <td><a href="https://github.com/isl-org/FastGlobalRegistration">FGR</a></td>
            <td>0.69</td>
            <td>0.16</td>
            <td>78.64</td>
            <td>95.15</td>
            <td>0.92</td>
        </tr>
        <tr align="center">
            <td><a href="https://github.com/url-kaist/Quatro">Quatro</a></td>
            <td>0.17</td>
            <td>0.18</td>
            <td>96.40</td>
            <td>98.20</td>
            <td>0.83</td>
        </tr>
        <tr align="center">
            <td><a href="https://github.com/MIT-SPARK/TEASER-plusplus">Teaser++</a></td>
            <td>0.77</td>
            <td>0.17</td>
            <td>76.70</td>
            <td>94.17</td>
            <td>0.91</td>
        </tr>
        <tr align="center">
            <td>V2I-Calib(Ours)</td>
            <td>0.06</td>
            <td>0.03</td>
            <td>93.26</td>
            <td>95.48</td>
            <td>0.37</td>
        </tr>
        <tr align="center">
            <td><strong>V2I-Calib++(Ours)</strong></td>
            <td><strong>0.01</strong></td>
            <td><strong>0.01</strong></td>
            <td><strong>96.80</strong></td>
            <td><strong>98.31</strong></td>
            <td><strong>0.13</strong></td>
        </tr>
    </table>
    </div>
    <h6>* Note: @λ indicates the threshold for success rate</h6>
* <a href="https://github.com/AIR-THU/DAIR-V2X">DAIR-V2X</a> (Heterogeneous LiDARs)
    <div align="left">
    <table>
        <tr align="center">
            <td rowspan="2">Method</td>
            <td rowspan="2">RRE(°)</td>
            <td rowspan="2">RTE(m)</td>
            <td colspan="2" align="center">Success Rate(%)</td>
            <td rowspan="2">Time (s)</td>
        </tr>
        <tr align="center">
            <td>@1</td>
            <td>@2</td>
        </tr>
        <tr align="center">
            <td><a href="https://github.com/isl-org/FastGlobalRegistration">FGR</a> </td>
            <td>1.71</td>
            <td>1.61</td>
            <td>22.11</td>
            <td>62.81</td>
            <td>25.50</td>
        </tr>
        <tr align="center">
            <td><a href="https://github.com/url-kaist/Quatro">Quatro</a></td>
            <td>1.46</td>
            <td>1.49</td>
            <td>19.90</td>
            <td>69.90</td>
            <td>24.52</td>
        </tr>
        <tr align="center">
            <td><a href="https://github.com/MIT-SPARK/TEASER-plusplus">Teaser++</a></td>
            <td>1.83</td>
            <td>1.67</td>
            <td>20.30</td>
            <td>58.91</td>
            <td>24.33</td>
        </tr>
        <tr align="center">
            <td>V2I-Calib(Ours)</td>
            <td>1.25</td>
            <td>1.32</td>
            <td>42.81</td>
            <td>76.04</td>
            <td>0.34</td>
        </tr>
        <tr align="center">
            <td><strong>V2I-Calib++(Ours)</strong></td>
            <td><strong>1.23</strong></td>
            <td><strong>1.16</strong></td>
            <td><strong>51.40</strong></td>
            <td><strong>84.58</strong></td>
            <td><strong>0.12</strong></td>
        </tr>
    </table>
    </div>
    <h6>* Note: @λ indicates the threshold for success rate</h6>


## Getting Started

#TODO

## Acknowledgment

This project is not possible without the following codebases.
* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
* [LiDAR-Registration-Benchmark](https://github.com/HKUST-Aerial-Robotics/LiDAR-Registration-Benchmark)


## Citation

If you find our work or this repo useful, please cite:
```
@article{qu2024v2i,
  title={V2I-Calib: A Novel Calibration Approach for Collaborative Vehicle and Infrastructure LiDAR Systems},
  author={Qu, Qianxin and Xiong, Yijin and Wu, Xin and Li, Hanyu and Guo, Shichun},
  journal={arXiv preprint arXiv:2407.10195},
  year={2024}
}
```
