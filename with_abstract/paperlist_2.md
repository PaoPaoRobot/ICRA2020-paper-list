
# International Conference on Robotics and Automation 2020
 
Welcome to ICRA 2020, the 2020 IEEE International Conference on Robotics and Automation.

This list is edited by [PaopaoRobot, 泡泡机器人](https://github.com/PaoPaoRobot) , the Chinese academic nonprofit organization. Recently we will classify these papers by topics. Welcome to follow our github and our WeChat Public Platform Account ( [paopaorobot_slam](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=100000102&idx=1&sn=0a8a831a4f2c18443dbf436ef5d5ff8c&chksm=6c10bf625b6736748c9612879e166e510f1fe301b72ed5c5d7ecdd0f40726c5d757e975f37af&mpshare=1&scene=1&srcid=0530KxSLjUE9I38yLgfO2nVm&pass_ticket=0aB5tcjeTfmcl9u0eSVzN4Ag4tkpM2RjRFH8DG9vylE%3D#rd) ). Of course, you could contact with [daiwei.song@outlook.com](mailto://daiwei.song@outlook.com)



## SLAM 

- Real-Time Graph-Based SLAM with Occupancy Normal Distributions Transforms

    Author: Schulz, Cornelia | University of Tübingen
    Author: Zell, Andreas | University of Tübingen
 
    keyword: SLAM; Performance Evaluation and Benchmarking; Software, Middleware and Programming Environments

    Abstract : Simultaneous Localization and Mapping (SLAM) is one of the basic problems in mobile robotics. While most approaches are based on occupancy grid maps, Normal Distributions Transforms (NDT) and mixtures like Occupancy Normal Distribution Transforms (ONDT) have been shown to represent sensor measurements more accurately. In this work, we slightly re-formulate the (O)NDT matching function such that it becomes a least squares problem that can be solved with various robust numerical and analytical non-linear optimizers. Further, we propose a novel global (O)NDT scan matcher for loop closure. In our evaluation, our NDT and ONDT methods are able to outperform the occupancy grid map based ones we adopted from Google's Cartographer implementation.

- Spatio-Temporal Non-Rigid Registration of 3D Point Clouds of Plants

    Author: Chebrolu, Nived | University of Bonn
    Author: L�be, Thomas | University of Bonn
    Author: Stachniss, Cyrill | University of Bonn
 
    keyword: SLAM; Robotics in Agriculture and Forestry

    Abstract : Analyzing sensor data of plants and monitoring plant performance is a central element in different agricultural robotics applications. In plant science, phenotyping refers to analyzing plant traits for monitoring growth, for describing plant properties, or characterizing the plant's overall performance. It plays a critical role in the agricultural tasks and in plant breeding. Recently, there is a rising interest in using 3D data obtained from laser scanners and 3D cameras to develop automated non-intrusive techniques for estimating plant traits. In this paper, we address the problem of registering 3D point clouds of the plants over time, which is a backbone of applications interested in tracking spatio-temporal traits of individual plants. Registering plants over time is challenging due to its changing topology, anisotropic growth, and non-rigid motion in between scans. We propose a novel approach that exploits the skeletal structure of the plant and determines correspondences over time and drives the registration process. Our approach explicitly accounts for the non-rigidity and the growth of the plant over time in the registration. We tested our approach on a challenging dataset acquired over the course of two weeks and successfully registered the 3D plant point clouds recorded with a laser scanner forming a basis for developing systems for automated temporal plant-trait analysis.

- Uncertainty-Based Adaptive Sensor Fusion for Visual-Inertial Odometry under Various Motion Characteristics

    Author: Nakashima, Ryo | Toshiba Corporation
    Author: Seki, Akihito | Toshiba Corporation
 
    keyword: SLAM; Sensor Fusion; Localization

    Abstract : We propose an uncertainty-based sensor fusion framework for visual-inertial odometry, which is the task of estimating relative motion using images and measurements from inertial measurement units. Visual-inertial odometry enables robust and scale-aware estimation of motion by incorporating sensor states, such as metric scale, velocity, and the direction of gravity, into the estimation. However, the observability of the states depends on sensor motion. For example, if the sensor moves in a constant velocity, scale and velocity cannot be observed from inertial measurements. Under these degenerate motions, existing methods may produce inaccurate results because they incorporate erroneous states estimated from non-informative inertial measurements. Our proposed framework is able to avoid this situation by adaptively switching estimation modes, which represents the states that should be incorporated, based on their uncertainties. These uncertainties can be obtained at a small computational cost by reusing the Jacobian matrices computed in bundle adjustment. Our approach consistently outperformed conventional sensor fusion in datasets with different motion characteristics, namely, the KITTI odometry dataset recorded by a ground vehicle and the EuRoC MAV dataset captured from a micro aerial vehicle.

- Loam_livox: A Fast, Robust, High-Precision LiDAR Odometry and Mapping Package for LiDARs of Small FoV

    Author: Lin, Jiarong | The University of Hong Kong
    Author: Zhang, Fu | University of Hong Kong
 
    keyword: SLAM; Localization; Mapping

    Abstract : LiDAR odometry and mapping (LOAM) has been playing an important role in autonomous vehicles, due to its ability to simultaneously localize the robot's pose and build high-precision, high-resolution maps of the surrounding environment. This enables autonomous navigation and safe path planning of autonomous vehicles. In this paper, we present a robust, real-time LOAM algorithm for LiDARs with small FoV and irregular samplings. By taking effort on both front-end and back-end, we address several fundamental challenges arising from such LiDARs, and achieve better performance in both precision and efficiency compared to existing baselines. To share our findings and to make contributions to the community, we open source our codes on Github

- Active SLAM Using 3D Submap Saliency for Underwater Volumetric Exploration

    Author: Suresh, Sudharshan | Carnegie Mellon University
    Author: Sodhi, Paloma | Carnegie Mellon University
    Author: Mangelson, Joshua | Brigham Young University
    Author: Wettergreen, David | Carnegie Mellon University
    Author: Kaess, Michael | Carnegie Mellon University
 
    keyword: SLAM; Marine Robotics; Motion and Path Planning

    Abstract : In this paper, we present an active SLAM framework for volumetric exploration of 3D underwater environments with multibeam sonar. Recent work in integrated SLAM and planning performs localization while maintaining volumetric free-space information. However, an absence of informative loop closures can lead to imperfect maps, and therefore unsafe behavior. To solve this, we propose a navigation policy that reduces vehicle pose uncertainty by balancing between volumetric exploration and revisitation. To identify locations to revisit, we build a 3D visual dictionary from real-world sonar data and compute a metric of submap saliency. Revisit actions are chosen based on propagated pose uncertainty and sensor information gain. Loop closures are integrated as constraints in our pose-graph SLAM formulation and these deform the global occupancy grid map. We evaluate our performance in simulation and real-world experiments, and highlight the advantages over an uncertainty-agnostic framework.

- Are We Ready for Service Robots? the OpenLORIS-Scene Datasets for Lifelong SLAM

    Author: Shi, Xuesong | Intel
    Author: Li, Dongjiang | Beijing Jiaotong University
    Author: Zhao, Pengpeng | Beihang University
    Author: Tian, Qinbin | Beijing Jiaotong University
    Author: Tian, YuXin | Beihang Universuty
    Author: Long, Qiwei | Beijingjiaotong University
    Author: Zhu, Chunhao | Beijing Jiaotong University
    Author: Song, Jingwei | Beijing Jiaotong University
    Author: Qiao, Fei | Tsinghua University
    Author: Song, Le | Gaussian Robotics
    Author: Guo, Yangquan | Gaussian Robotics
    Author: Wang, Zhigang | Intel Labs
    Author: Zhang, Yimin | Intel Corporation
    Author: Qin, Baoxing | NUS
    Author: Yang, Wei | Beijing Jiaotong University, School of Electronic and Information
    Author: Wang, Fangshi | Beijing Jiaotong University
    Author: Chan, Rosa H. M. | City University of Hong Kong
    Author: She, Qi | Intel Labs
 
    keyword: SLAM; Localization; Service Robots

    Abstract : Service robots should be able to operate autonomously in dynamic and daily changing environments over an extended period of time. While Simultaneous Localization And Mapping (SLAM) is one of the most fundamental problems for robotic autonomy, most existing SLAM works are evaluated with data sequences that are recorded in a short period of time. In real-world deployment, there can be out-of-sight scene changes caused by both natural factors and human activities. For example, in home scenarios, most objects may be movable, replaceable or deformable, and the visual features of the same place may be significantly different in some successive days. Such out-of-sight dynamics pose great challenges to the robustness of pose estimation, and hence a robot's long-term deployment and operation. To differentiate the forementioned problem from the conventional works which are usually evaluated in a static setting in a single run, the term textit{lifelong SLAM} is used here to address SLAM problems in an ever-changing environment over a long period of time. To accelerate lifelong SLAM research, we release the OpenLORIS-Scene datasets. The data are collected in real-world indoor scenes, for multiple times in each place to include scene changes in real life. We also design benchmarking metrics for lifelong SLAM, with which the robustness and accuracy of pose estimation are evaluated separately. The datasets and benchmark are available online at https://lifelong-robotic-vision.github.io


- Intensity Scan Context: Coding Intensity and Geometry Relations for Loop Closure Detection

    Author: Wang, Han | Nanyang Technological University
    Author: Wang, Chen | Carnegie Mellon University
    Author: Xie, Lihua | NanyangTechnological University
 
    keyword: SLAM; Recognition; Factory Automation

    Abstract : Loop closure detection is an essential and challenging problem in simultaneous localization and mapping (SLAM). It is often tackled with light detection and ranging (LiDAR) sensor due to its view-point and illumination invariant properties. Existing works on 3D loop closure detection often leverage on matching of local or global geometrical-only descriptors which discard intensity reading. In this paper we explore the intensity property from LiDAR scan and show that it can be effective for place recognition. We propose a novel global descriptor, intensity scan context (ISC), that explores both geometry and intensity characteristics. To improve the efficiency for loop closure detection, an efficient two-stage hierarchical re-identification process is proposed, including binary-operation based fast geometric relation retrieval and intensity structure re-identification. Thorough experiments including both local experiment and public datasets test have been conducted to evaluate the performance of the proposed method. Our method achieves better recall rate and recall precision than existing geometric-only methods.

- TextSLAM: Visual SLAM with Planar Text Features

    Author: Li, Boying | Shanghai Jiao Tong University
    Author: Zou, Danping | Shanghai Jiao Ton University
    Author: Sartori, Daniele | Shanghai Jiao Tong University
    Author: Pei, Ling | Shanghai Jiao Tong University
    Author: Yu, Wenxian | Shanghai Jiao Tong University
 
    keyword: SLAM; Visual-Based Navigation; Mapping

    Abstract : We propose to integrate text objects in man-made scenes tightly into the visual SLAM pipeline. The key idea of our novel text-based visual SLAM is to treat each detected text as a planar feature which is rich of textures and semantic meanings. The text feature is compactly represented by three parameters and integrated into visual SLAM by adopting the illumination-invariant photometric error. We also describe important details involved in implementing a full pipeline of text-based visual SLAM. To our best knowledge, this is the first visual SLAM method tightly coupled with the text features. We tested our method in both indoor and outdoor environments. The results show that with text features, the visual SLAM system becomes more robust and produces much more accurate 3D text maps that could be useful for navigation and scene understanding in robotic or augmented reality applications.

- FlowNorm: A Learning-Based Method for Increasing Convergence Range of Direct Alignment

    Author: Wang, Ke | Hong Kong University of Science and Technology
    Author: Wang, Kaixuan | Hong Kong University of Science and Technology
    Author: Shen, Shaojie | Hong Kong University of Science and Technology
 
    keyword: SLAM; Localization

    Abstract : Many works have been proposed to estimate camera poses by directly minimizing photometric error. However, due to the non-convex property of the direct alignment, proper initialization is still required for these methods. Many robust norm (e.g. Huber norm) have been proposed to deal with the outlier terms caused by incorrect initializations. These robust norms are solely defined on the magnitude of each error terms. In this paper, we propose a novel robust norm, named FlowNorm, that exploit the information from both the local error term and the global image registration information. While the local information is defined on patch alignments, the global information is estimated using a learning-based network. Using both the local and global information, we achieve a large convergence range that images can be aligned given large view angle changes or small overlaps. We further demonstrate the usability of the proposed robust norm by integrating it into direct methods, DSO and BA-Net, and generate more robust and accurate results in real-time.

- Redesigning SLAM for Arbitrary Multi-Camera Systems

    Author: Kuo, Juichung | ETH Zurich
    Author: Muglikar, Manasi | University of Zurich
    Author: Zhang, Zichao | Robotics and Perception Group, University of Zurich
    Author: Scaramuzza, Davide | University of Zurich
 
    keyword: SLAM; Localization; Visual-Based Navigation

    Abstract : Adding more cameras to SLAM systems improves robustness and accuracy but complicates the design of the visual front-end significantly. Thus, most systems in the literature are tailored for specific camera configurations. In this work, we aim at an adaptive SLAM system that works for arbitrary multi-camera setups. To this end, we revisit several common building blocks in visual SLAM. In particular, we propose an adaptive initialization scheme, a sensor-agnostic, information-theoretic keyframe selection algorithm, and a scalable voxel-based map. These techniques make little assumption about the actual camera setups and prefer theoretically grounded methods over heuristics. We adapt a state-of-the-art visual-inertial odometry with these modifications, and experimental results show that the modified pipeline can adapt to a wide range of camera setups (e.g., 2 to 6 cameras in one experiment) without the need of sensor-specific modifications or tuning.

- Dynamic SLAM: The Need for Speed

    Author: Henein, Mina | Australian National University
    Author: Zhang, Jun | Australian National University
    Author: Mahony, Robert | Australian National University
    Author: Ila, Viorela | The University of Sydney
 
    keyword: SLAM; Mapping

    Abstract : The static world assumption is standard in most simultaneous localisation and mapping (SLAM) algorithms. Increased deployment of autonomous systems to unstructured dynamic environments is driving a need to identify moving objects and estimate their velocity in real-time. Most existing SLAM based approaches rely on a database of 3D models of objects or impose significant motion constraints. In this paper, we propose a new feature-based, model-free, object-aware dynamic SLAM algorithm that exploits semantic segmentation to allow estimation of motion of rigid objects in a scene without the need to estimate the object poses or have any prior knowledge of their 3D models. The algorithm generates a map of dynamic and static structure and has the ability to extract velocities of rigid moving objects in the scene. Its performance is demonstrated on simulated, synthetic and real-world datasets.

- GradSLAM: Dense SLAM Meets Automatic Differentiation

    Author: Jatavallabhula, Krishna | Mila, Universite De Montreal
    Author: Iyer, Ganesh | International Institute of Information Technology, Hyderabad
    Author: Paull, Liam | Université De Montr�al
 
    keyword: SLAM; Mapping; Deep Learning in Robotics and Automation

    Abstract : The question of representation is central in the context of dense simultaneous localization and mapping (SLAM). Newer learning-based approaches have the potential to leverage data or task performance to directly inform the choice of representation. However, blending representation learning approaches with ``classical" SLAM systems has remained an open question, because of the highly modular and complex nature of classical SLAM systems. In robotics, the task of a SLAM system is often to transform raw sensor inputs to a distribution of the state(s) of the robot and the environment. If this transformation (SLAM) were expressible as a differentiable function, we could leverage task-based error signals over the outputs of this function to learn representations that optimize task performance. The challenge---however---is that several components of a typical dense SLAM system are non-differentiable. In this work, we propose a novel way of unifying gradient-based learning and SLAM. We propose the term gradSLAM, a methodology for posing SLAM systems as differentiable computational graphs. We demonstrate how to design differentiable trust-region optimizers, surface measurement and fusion schemes, as well as differentiate over rays, without sacrificing performance. This amalgamation of dense SLAM with computational graphs enables us to backprop all the way from 3D maps to 2D pixels, opening up new possibilities in gradient-based learning for SLAM.



- Long-Term Place Recognition through Worst-Case Graph Matching to Integrate Landmark Appearances and Spatial Relationships

    Author: Gao, Peng | Colorado School of Mines
    Author: Zhang, Hao | Colorado School of Mines
 
    keyword: SLAM; Localization

    Abstract : Place recognition is an important component for simultaneously localization and mapping in a variety of robotics applications. Recently, several approaches using landmark information to represent a place showed promising performance to address long-term environment changes. However, previous approaches do not explicitly consider changes of the landmarks, i,e., old landmarks may disappear and new ones often appear over time. In addition, representations used in these approaches to represent landmarks are limited, based upon visual or spatial cues only. In this paper, we introduce a novel worst-case graph matching approach that integrates spatial relationships of landmarks with their appearances for long-term place recognition. Our method designs a graph representation to encode distance and angular spatial relationships as well as visual appearances of landmarks in order to represent a place. Then, we formulate place recognition as a graph matching problem under the worst-case scenario. Our approach matches places by computing the similarities of distance and angular spatial relationships of the landmarks that have the least similar appearances (i.e., worst-case). If the worst appearance similarity of landmarks is small, two places are identified to be not the same, even though their graph representations have high spatial relationship similarities. We evaluate our approach over two public benchmark datasets for long-term place recognition, including St. Lucia and CMU-VL. Th

- Linear RGB-D SLAM for Atlanta World

    Author: Joo, Kyungdon | Korea Advanced Institute of Science and Technology (KAIST)
    Author: Oh, Tae-Hyun | MIT
    Author: Rameau, Francois | KAIST, RCV Lab
    Author: Bazin, Jean-Charles | KAIST
    Author: Kweon, In So | KAIST
 
    keyword: SLAM; RGB-D Perception; Localization

    Abstract : We present a new linear method for RGB-D based simultaneous localization and mapping (SLAM). Compared to existing techniques relying on the Manhattan world assumption defined by three orthogonal directions, our approach is designed for the more general scenario of the Atlanta world. It consists of a vertical direction and a set of horizontal directions orthogonal to the vertical direction and thus can represent a wider range of scenes. Our approach leverages the structural regularity of the Atlanta world to decouple the non-linearity of camera pose estimations. This allows us separately to estimate the camera rotation and then the translation, which bypasses the inherent non-linearity of traditional SLAM techniques. To this end, we introduce a novel tracking-by-detection scheme to estimate the underlying scene structure by Atlanta representation. Thereby, we propose an Atlanta frame-aware linear SLAM framework which jointly estimates the camera motion and a planar map supporting the Atlanta structure through a linear Kalman filter. Evaluations on both synthetic and real datasets demonstrate that our approach provides favorable performance compared to existing state-of-the-art methods while extending their working range to the Atlanta world.

- Stereo Visual Inertial Odometry with Online Baseline Calibration

    Author: Fan, Yunfei | Meituan-Dianping Group
    Author: Wang, Ruofu | University of Southern California
    Author: Mao, Yinian | Meituan-Dianping Group
 
    keyword: SLAM; Sensor Fusion

    Abstract : Stereo-vision devices have rigorous requirements for extrinsic parameter calibration. In Stereo Visual Inertial Odometry (VIO), inaccuracy in or changes to camera extrinsic parameters may lead to serious degradation in estimation performance. In this manuscript, we propose an online calibration method for stereo VIO extrinsic parameters correction. In particular, we focus on Multi-State Constraint Kalman Filter (MSCKF [1]) framework to implement our method. The key component is to formulate stereo extrinsic parameters as part of the state variables and model the Jacobian of feature reprojection error with respect to stereo extrinsic parameters as sub-block of update Jacobian. Therefore we can estimate stereo extrinsic parameters simultaneously with inertial measurement unit (IMU) states and camera poses. Experiments on EuRoC dataset and real-world outdoor dataset demonstrate that the proposed algorithm produce higher positioning accuracy than the original S-MSCKF [2], and the noise of camera extrinsic parameters are self-corrected within the system.

- Lidar-Monocular Visual Odometry Using Point and Line Features

    Author: Huang, Shi-Sheng | Tsinghua University, Beijing
    Author: Ma, Zeyu | Tsinghua University
    Author: Mu, Tai-Jiang | Tsinghua University, Beijing, PRC
    Author: Fu, Hongbo | City University of Hong Kong
    Author: Hu, Shi-Min | Tsinghua University
 
    keyword: SLAM; Visual-Based Navigation; Sensor Fusion

    Abstract : We introduce a novel lidar-monocular visual odometry approach using point and line features. Compared to previous point-only based lidar-visual odometry, our approach leverages more environment structure information by introducing both point and line features into pose estimation. We provide a robust method for point and line depth extraction, and formulate the extracted depth as prior factors for point-line bundle adjustment. This method greatly reduces the features' 3D ambiguity and thus improves the pose estimation accuracy. Besides, we also provide a purely visual motion tracking method and a novel scale correction scheme, leading to an efficient lidar-monocular visual odometry system with high accuracy. The evaluations on the public KITTI odometry benchmark show that our technique achieves more accurate pose estimation than the state-of-the-art approaches, and is sometimes even better than those leveraging semantic information.

- Probabilistic Data Association Via Mixture Models for Robust Semantic SLAM

    Author: Doherty, Kevin | Massachusetts Institute of Technology
    Author: Baxter, David | Massachusetts Institute of Technology
    Author: Schneeweiss, Edward | University of Massachusetts Amherst
    Author: Leonard, John | MIT
 
    keyword: SLAM; Visual-Based Navigation; Semantic Scene Understanding

    Abstract : Modern robotic systems sense the environment geometrically, through sensors like cameras, lidar, and sonar, as well as semantically, often through visual models learned from data, such as object detectors. We aim to develop robots that can use all of these sources of information for reliable navigation, but each is corrupted by noise. Rather than assume that object	detection will eventually achieve near perfect performance across the lifetime of a robot, in this work we represent and cope with the semantic and geometric uncertainty inherent in object detection methods. Specifically, we model data association ambiguity, which is typically non-Gaussian, in a way that is amenable to solution within the common nonlinear Gaussian formulation of simultaneous localization and mapping (SLAM). We do so by eliminating data association variables from the inference process through max-marginalization, preserving standard Gaussian posterior assumptions. The result is a	max-mixture-type model that accounts for multiple data association hypotheses. We provide experimental results on indoor	and outdoor semantic navigation tasks with noisy odometry and object detection and find that the ability of the proposed approach to represent multiple hypotheses, including the �null'' hypothesis, gives substantial robustness advantages in comparison to alternative semantic SLAM approaches.

- Closed-Loop Benchmarking of Stereo Visual-Inertial SLAM Systems: Understanding the Impact of Drift and Latency on Tracking Accuracy

    Author: Zhao, Yipu | Facebook Inc
    Author: Smith, Justin | Georgia Institute of Technology
    Author: Karumanchi, Sambhu Harimanas | National Institute of Technology Karnataka, Surathkal
    Author: Vela, Patricio | Georgia Institute of Technology
 
    keyword: SLAM; Performance Evaluation and Benchmarking; Localization

    Abstract : Visual-inertial SLAM is essential for robot navigation in GPS-denied environments, e.g. indoor, underground. Conventionally, the performance of visual-inertial SLAM is evaluated with open-loop analysis, with a focus on the drift level of SLAM systems. In this paper, we raise the question on the importance of visual estimation latency in closed-loop navigation tasks, such as accurate trajectory tracking. To understand the impact of both drift and latency on visual-inertial SLAM systems, a closed-loop benchmarking simulation is conducted, where a robot is commanded to follow a desired trajectory using the feedback from visual-inertial estimation. By extensively evaluating the trajectory tracking performance of representative state-of-the-art visual-inertial SLAM systems, we reveal the importance of latency reduction in visual estimation module of these systems. The findings suggest directions of future improvements for visual-inertial SLAM.

- Metrically-Scaled Monocular SLAM Using Learned Scale Factors

    Author: Greene, William Nicholas | Massachusetts Institute of Technology
    Author: Roy, Nicholas | Massachusetts Institute of Technology
 
    keyword: SLAM; Visual-Based Navigation; Deep Learning in Robotics and Automation

    Abstract : We propose an efficient method for monocular simultaneous localization and mapping (SLAM) that is capable of estimating metrically-scaled motion without additional sensors or hardware acceleration by integrating metric depth predictions from a neural network into a geometric SLAM factor graph. Unlike learned end-to-end SLAM systems, ours does not ignore the relative geometry directly observable in the images. Unlike existing learned depth estimation approaches, ours leverages the insight that when used to estimate scale, learned depth predictions need only be coarse in image space. This allows us to shrink our network to the point that performing inference on a standard CPU becomes computationally tractable.<p>We make several improvements to our network architecture and training procedure to address the lack of depth observability when using coarse images, which allows us to estimate spatially coarse, but depth-accurate predictions in only 30 ms per frame without GPU acceleration. At runtime we incorporate the learned metric data as unary scale factors in a Sim(3) pose graph. Our method is able to generate accurate, scaled poses without additional sensors, hardware accelerators, or special maneuvers and does not ignore or corrupt the observable epipolar geometry. We show compelling results on the KITTI benchmark dataset in addition to real-world experiments with a handheld camera.

- Inertial-Only Optimization for Visual-Inertial Initialization

    Author: Campos, Carlos | Universidad De Zaragoza
    Author: Montiel, J.M.M | I3A. Universidad De Zaragoza
    Author: Tardos, Juan D. | Universidad De Zaragoza
 
    keyword: SLAM; Mapping; Localization

    Abstract : We formulate for the first time visual-inertial initialization as an optimal estimation problem, in the sense of maximum-a-posteriori (MAP) estimation. This allows us to properly take into account IMU measurement uncertainty, which was neglected in previous methods that either solved sets of algebraic equations, or minimized ad-hoc cost functions using least squares. Our exhaustive initialization tests on EuRoC dataset show that our proposal largely outperforms the best methods in the literature, being able to initialize in less than 4 seconds in almost any point of the trajectory, with a scale error of 5.3% on average. This initialization has been integrated into ORB-SLAM Visual-Inertial boosting its robustness and efficiency while maintaining its excellent accuracy.

- Hierarchical Quadtree Feature Optical Flow Tracking Based Sparse Pose-Graph Visual-Inertial SLAM

    Author: Xie, Hongle | Shanghai Jiao Tong University
    Author: Chen, Weidong | Shanghai Jiao Tong University
    Author: Wang, Jingchuan | Shanghai Jiao Tong University
    Author: Wang, Hesheng | Shanghai Jiao Tong University
 
    keyword: SLAM; Localization; Sensor Fusion

    Abstract : Accurate, robust and real-time localization under constrained-resources is a critical problem to be solved. In this paper, we present a new sparse pose-graph visual-inertial SLAM (SPVIS). Unlike the existing methods that are costly to deal with a large number of redundant features and 3D map points, which are inefficient for improving positioning accuracy, we focus on the concise visual cues for high-precision pose estimating. We propose a novel hierarchical quadtree based optical flow tracking algorithm, it achieves high accuracy and robustness within very few concise features, which is only about one fifth features of the state-of-the-art visual-inertial SLAM algorithms. Benefiting from the efficient optical flow tracking, our sparse pose-graph optimization time cost achieves bounded complexity. By selecting and optimizing the informative features in sliding window and local VIO, the computational complexity is bounded, it achieves low time cost in long-term operation. We compare with the state-of-the-art VIO/VI-SLAM systems on the challenging public datasets by the embedded platform without GPUs, the results effectively verify that the proposed method has better real-time performance and localization accuracy.

- Keypoint Description by Descriptor Fusion Using Autoencoders

    Author: Dai, Zhuang | Guangdong University of Technology
    Author: Huang, Xinghong | Guangdong University of Technology
    Author: Chen, Weinan | The Chinese University of Hong Kong
    Author: Chen, Chuangbin | Guangdong University of Technology
    Author: He, Li | Guangdong University of Technology
    Author: Wen, Shuhuan | Yanshan University
    Author: Zhang, Hong | University of Alberta
 
    keyword: SLAM; Localization; Visual Learning

    Abstract : Keypoint matching is an important operation in computer vision and its applications such as visual simultaneous localization and mapping (SLAM) in robotics. This matching operation heavily depends on the descriptors of the keypoints, and it must be performed reliably when images undergo conditional changes such as those in illumination and viewpoint. In this paper, a descriptor fusion model (DFM) is proposed to create a robust keypoint descriptor by fusing CNN-based descriptors using autoencoders. Our DFM architecture can be adapted to either trained or pre-trained CNN models. Based on the performance of existing CNN descriptors, we choose HardNet and DenseNet169 as representatives of trained and pre-trained descriptors. Our proposed DFM is evaluated on the latest benchmark datasets in computer vision with challenging conditional changes. The experimental results show that DFM is able to achieve state-of-the-art performance, with the mean mAP that is 6.34% and 6.42% higher than HardNet and DenseNet169, respectively.

- Towards Noise Resilient SLAM

    Author: Thyagharajan, Anirud | Intel Labs
    Author: Mandal, Dipan | Intel Labs
    Author: Ji Omer, Om | Intel Labs
    Author: Subramoney, Sreenivas | Intel Labs
 
    keyword: SLAM; RGB-D Perception; Visual Tracking

    Abstract : Sparse-indirect SLAM systems have been dominantly popular due to their computational efficiency and photometric invariance properties. Depth sensors are critical to SLAM frameworks for providing scale information to the 3D world, yet known to be plagued by a wide variety of noise sources, possessing lateral and axial components. In this work, we demonstrate the detrimental impact of these depth noise components on the performance of the state-of-the-art sparse-indirect SLAM system (ORB-SLAM2). We propose (i) Map-Point Consensus based Outlier Rejection (MC-OR) to counter lateral noise, and (ii) Adaptive Virtual Camera (AVC) to combat axial noise accurately. MC-OR utilizes consensus information between multiple sightings of the same landmark to disambiguate noisy depth and filter it out before pose optimization. In AVC, we introduce an error vector as an accurate representation of the axial depth error. We additionally propose an adaptive algorithm to find the virtual camera location for projecting the error used in the objective function of the pose optimization. Our techniques work equally well for stereo image pairs and RGB-D input directly used by sparse-indirect SLAM systems. Our methods were tested on the TUM (RGB-D) and EuRoC (stereo) datasets and we show that they outperform existing state-of-the-art ORB-SLAM2 by 2-3x, especially in sequences critically affected by depth noise.

- LAMP: Large-Scale Autonomous Mapping and Positioning for Exploration of Perceptually-Degraded Subterranean Environments

    Author: Ebadi, Kamak | Santa Clara University
    Author: Chang, Yun | MIT
    Author: Palieri, Matteo | Polytechnic University of Bari
    Author: Stephens, Alex | University of Sydney
    Author: Hatteland, Alexander Haugland | ETH University
    Author: Heiden, Eric | University of Southern California
    Author: Thakur, Abhishek | Aptiv Tro
    Author: Morrell, Benjamin | The University of Sydney
    Author: Carlone, Luca | Massachusetts Institute of Technology
    Author: Agha-mohammadi, Ali-akbar | NASA-JPL, Caltech
    Author: Wood, Sally | Santa Clara University
    Author: Funabiki, Nobuhiro | University of Tokyo
 
    keyword: SLAM; Multi-Robot Systems; Search and Rescue Robots

    Abstract : Simultaneous Localization and Mapping (SLAM) in large-scale, unknown, and complex subterranean environments is a challenging problem. Sensors have to operate in off-nominal conditions; uneven and slippery terrains make wheel odometry inaccurate, while long corridors without salient features make exteroceptive sensing ambiguous and prone to drift; finally, spurious loop closures that are frequent in environments with repetitive appearance, such as tunnels and mines, could result in a significant distortion of the entire map. These challenges are in stark contrast with the need to build highly-accurate 3D maps to support a wide variety of applications, ranging from disaster response to the exploration of underground extraterrestrial worlds. This paper reports on the implementation and testing of a lidar-based multi-robot SLAM system developed in the context of the DARPA Subterranean Challenge. We present a system architecture to enhance subterranean operation, including an accurate lidar-based front-end, and a flexible and robust back-end that automatically rejects outlying loop closures. We present an extensive evaluation in large-scale, challenging subterranean environments, including the results obtained in the Tunnel Circuit of the DARPA Subterranean Challenge. Finally, we discuss potential improvements, limitations of the state of the art, and future research directions.


- Modeling Semi-Static Scenes with Persistence Filtering in Visual SLAM

    Author: Hashemifar, Zakieh | Zoox, University at Buffalo
    Author: Dantu, Karthik | University of Buffalo
 
    keyword: SLAM; Mapping; RGB-D Perception

    Abstract : Many existing SLAM approaches rely on the assumption of static environments for accurate performance. However, several robot applications require them to traverse repeatedly in semi-static or dynamic environments. There has been some recent research interest in designing persistence filters to reason about persistence in such scenarios. Our goal in this work is to incorporate such persistence reasoning in visual SLAM. To this end, we incorporate persistence filters [1] into ORB-SLAM, a well-known visual SLAM algorithm. We observe that the simple integration of their proposal results in inefficient persistence reasoning. Through a series of modifications and using two locally collected datasets, we demonstrate the utility of such persistence filtering as well as our customizations in ORB- SLAM. Overall, incorporating persistence filtering could result in a significant reduction in map size (about 30% in the best case) and a corresponding reduction in run-time while retaining similar accuracy to methods that use much larger maps.

- Broadcast Your Weaknesses: Cooperative Active Pose-Graph SLAM for Multiple Robots

    Author: Chen, Yongbo | University of Technology, Sydney
    Author: Zhao, Liang | University of Technology Sydney
    Author: Lee, Ki Myung Brian | University of Technology Sydney
    Author: Yoo, Chanyeol | University of Technology Sydney
    Author: Huang, Shoudong | University of Technology, Sydney
    Author: Fitch, Robert | University of Technology Sydney
 
    keyword: SLAM; Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems

    Abstract : In this paper, we propose a low-cost, high-efficiency framework for cooperative active pose-graph simultaneous localization and mapping (SLAM) for multiple robots in three-dimensional (3D) environments based on graph topology. Based on the selection of weak connections in pose graphs, this method aims to find the best trajectories for optimal information exchange to repair these weaknesses opportunistically when robots move near them. Based on tree-connectivity, which is greatly related to the D-optimality metric of the Fisher information matrix (FIM), we explore the relationship between measurement (edge) selection and pose-measurement (node-edge) selection, which often occurs in active SLAM, in terms of information increment. The measurement selection problem is formulated as a submodular optimization problem and solved by an exhaustive method using rank-1 updates. We decide which robot takes the selected measurements through a bidding framework where each robot computes its predicted cost. Finally, based on a novel continuous trajectory optimization method, these additional measurements collected by the winning robot are sent to the requesting robot to strengthen its pose graph. In simulations and experiments, we validate our approach by comparing against existing methods. Further, we demonstrate online communication based on offline planning results using two unmanned aerial vehicles (UAVs).

- FlowFusion: Dynamic Dense RGB-D SLAM Based on Optical Flow

    Author: Zhang, Tianwei | The University of Tokyo
    Author: Zhang, Huayan | Beijing University of Civil Engineering and Architecture
    Author: Li, Yang | The University of Tokyo
    Author: Nakamura, Yoshihiko | University of Tokyo
    Author: Zhang, Lei | Beijing University of Civil Engineering and Architecture
 
    keyword: Inventory Management; Logistics

    Abstract : Dynamic environments are challenging for visual SLAM since the moving objects occlude the static environment features and lead to wrong camera motion estimation. In this paper, we present a novel dense RGB-D SLAM solution that simultaneously accomplishes the dynamic/static segmentation and camera ego-motion estimation as well as the static background reconstructions. Our novelty is using optical flow residuals to highlight the dynamic semantics in the RGB-D point clouds and provide more accurate and efficient dynamic/static segmentation for camera tracking and background reconstructions. Dense reconstruction results on both public datasets and the real dynamic environments indicate the proposed approach achieved accurate and efficient performances in both dynamic and static environment compared to state-of-the-art approaches.

- Efficient Algorithms for Maximum Consensus Robust Fitting (I)

    Author: Wen, Fei | Shanghai Jiao Tong University
    Author: Ying, Rendong | Shanghai Jiao Tong University
    Author: Gong, Zheng | Shanghai Jiao Tong University
    Author: Liu, Peilin | Shanghai Jiao Tong Universit
 
    keyword: Probability and Statistical Methods; Visual-Based Navigation; SLAM

    Abstract : Maximum consensus robust fitting is a fundamental problem in many computer vision applications, such as vision-based robotic navigation and mapping. While exact search algorithms are computationally demanding, randomized algorithms are cheap but the solution quality is not guaranteed. Deterministic algorithms fill the gap between these two kinds of algorithms, which have better solution quality than randomized algorithms while be much faster than exact algorithms. In this paper, we develop two highly-efficient deterministic algorithms based on the alternating direction method of multipliers (ADMM) and proximal block coordinate descent (BCD) frameworks. Particularly, the proposed BCD algorithm is guaranteed convergent. Further, on the slack variable in the BCD algorithm, which indicates the inliers and outliers, we establish some meaningful properties, such as support convergence within finite iterations and convergence to restricted strictly local minimizer. Compared with state-of-the-art algorithms, the new algorithms with initialization from a randomized or convex relaxed algorithm can achieve improved solution quality while being much more efficient (e.g., more than an order of magnitude faster).<p>An application of the new ADMM algorithm in simultaneous localization and mapping (SLAM) has also been provided to demonstrate its effectiveness. Code for reproducing the results is available at https://github.com/FWen/emc.git.

- MulRan: Multimodal Range Dataset for Urban Place Recognition

    Author: Kim, Giseop | KAIST(Korea Advanced Institute of Science and Technology)
    Author: Park, Yeong Sang | KAIST
    Author: Cho, Younghun | KAIST
    Author: Jeong, Jinyong | KAIST
    Author: Kim, Ayoung | Korea Advanced Institute of Science Technology
 
    keyword: Range Sensing; Localization; SLAM

    Abstract : This paper introduces a multimodal range dataset namely for radio detection and ranging (radar) and light detection and ranging (LiDAR) specifically targeting the urban environment. By extending our workshop paper [1] to a larger scale, this dataset focuses on the range sensor-based place recognition and provides 6D baseline trajectories of a vehicle for place recognition ground truth. Provided radar data support both raw-level and image-format data, including a set of time-stamped 1D intensity arrays and 360 &#9702; polar images, respectively. In doing so, we provide flexibility between raw data and image data depending on the purpose of the research. Unlike existing datasets, our focus is at capturing both temporal and structural diversities for range-based place recognition research. For evaluation, we applied and validated that our previous location descriptor and its search algorithm [2] are highly effective for radar place recognition method. Furthermore, the result shows that radar-based place recognition outperforms LiDAR-based one exploiting its longer-range measurements. The dataset is available from https://sites.google.com/view/mulran-pr

- GPO: Global Plane Optimization for Fast and Accurate Monocular SLAM Initialization

    Author: Du, Sicong | Institute of Automation&#65292;Chinese Academy of Sciences
    Author: Guo, Hengkai | ByteDance AI Lab
    Author: Chen, Yao | ByteDance Inc
    Author: Woods, Yilun | Institute of Automation&#65292;Chinese Academy of Sciences
    Author: Meng, Xiangbing | Institute of Automation&#65292;Chinese Academy of Sciences
    Author: Wen, Linfu | ByteDance AI Lab
    Author: Wang, Feiyue | Institute of Automation, Chinese Academy of Sciences
 
    keyword: SLAM; Localization

    Abstract : Initialization is essential to monocular Simultaneous Localization and Mapping (SLAM) problems. This paper focuses on a novel initialization method for monocular SLAM based on planar features. The algorithm starts by homography estimation in a sliding window. It then proceeds to a global plane optimization (GPO) to obtain camera poses and the plane normal. 3D points can be recovered using planar constraints without triangulation. The proposed method fully exploits the plane information from multiple frames and avoids the ambiguities in homography decomposition. We validate our algorithm on the collected chessboard dataset against baseline implementations and present extensive analysis. Experimental results show that our method outperforms the &#64257;ne-tuned baselines in both accuracy and real-time.

- Large-Scale Volumetric Scene Reconstruction Using LiDAR

    Author: K�hner, Tilman | FZI Forschungszentrum Informatik
    Author: K�mmerle, Julius | FZI Forschungszentrum Informatik
 
    keyword: Range Sensing; Mapping; SLAM

    Abstract : Large-scale 3D scene reconstruction is an important task in autonomous driving and other robotics applications as having an accurate representation of the environment is necessary to safely interact with it. Reconstructions are used for numerous tasks ranging from localization and mapping to planning. In robotics, volumetric depth fusion is the method of choice for indoor applications since the emergence of commodity RGB-D cameras due to its robustness and high reconstruction quality. In this work we present an approach for volumetric depth fusion using LiDAR sensors as they are common on most autonomous cars. We present a framework for large-scale mapping of urban areas considering loop closures. Our method creates a meshed representation of an urban area from recordings over a distance of 3.7 km with a high level of detail on consumer graphics hardware in several minutes. The whole process is fully automated and does not need any user interference. We quantitatively evaluate our results from a real world application. Also, we investigate the effects of the sensor model that we assume on reconstruction quality by using synthetic data.

- Topological Mapping for Manhattan-Like Repetitive Environments

    Author: Puligilla, Sai Shubodh | IIIT Hyderabad
    Author: Tourani, Satyajit | IIIT Hyderabad
    Author: Vaidya, Tushar | IIIT Hyderabad
    Author: Singh Parihar, Udit | IIIT Hyderabad
    Author: Sarvadevabhatla, Ravi Kiran | IIIT Hyderabad
    Author: Krishna, Madhava | IIIT Hyderabad
 
    keyword: SLAM; Mapping; Computer Vision for Automation

    Abstract : We showcase a topological mapping framework for a challenging indoor warehouse setting. At the most     Abstract level,	the warehouse	is represented as a Topological Graphwhere the nodes of the graph represent a particular warehouse topological construct (e.g. rackspace, corridor) and the edges denote the existence of	 a path between two neighbouring nodes or topologies. At	the intermediate level,	the map is represented as a Manhattan Graph where the nodes and edges are characterized by Manhattan properties and as a Pose Graphat the lower-most level of detail. The topological constructs are learned	 via a Deep	 Convolutional	 Network while the relational properties between topological instances are learnt via	a Siamese-style Neural Network. In	the paper, we show that maintaining     Abstractions such	as Topological Graph and manhattan Graph help in recovering an	accurate Pose	Graphstarting from a highly erroneous and unoptimized Pose Graph. We show how this is achieved by embedding topological and manhattan relations, as well as Manhattan Graph, aided loop closure relations	as constraints in the backend Pose Graph optimization framework. The recovery of near ground-truth Pose	Graph on real-world indoor warehouse scenes vindicate the efficacy of the proposed framework.

-  Structure-Aware COP-SLAM

    Author: Li, Liang | Eindhoven University of Technology
    Author: Dubbelman, Gijs | Eindhoven University of Technology

- Robust RGB-D Camera Tracking Using Optimal Key-Frame Selection

    Author: Han, Kyung Min | Ewha Woman's Univeristy
    Author: Kim, Young J. | Ewha Womans University
 
    keyword: Mapping; RGB-D Perception; SLAM

    Abstract : We propose a novel RGB-D camera tracking system that robustly reconstructs hand-held RGB-D camera sequences. The robustness of our system is achieved by two independent features of our method: adaptive visual odometry (VO) and integer programming-based key-frame selection. Our VO method adaptively interpolates the camera motion results of the direct VO and the iterative closed point to yield more optimal results than existing methods such as Elastic-Fusion. Moreover, our keyframe selection method locates globally optimum key-frames using a comprehensive objective function in a deterministic manner rather than heuristic or experience-based rules that prior methods mostly rely on. As a result, our method can complete reconstruction even if the camera fails to be tracked due to discontinuous camera motions, such as kidnap events, when conventional systems need to backtrack the scene. We validated our tracking system on 25 TUM benchmark sequences against state-of-the-art works, such as ORBSLAM2, Elastic-Fusion, and DVO SLAM, and experimentally showed that our method has smaller and more robust camera trajectory errors than these systems.


- Voxgraph: Globally Consistent, Volumetric Mapping Using Signed Distance Function Submaps

    Author: Reijgwart, Victor | ETH Zurich
    Author: Millane, Alexander James | ETH Zurich
    Author: Oleynikova, Helen | Microsoft
    Author: Siegwart, Roland | ETH Zurich
    Author: Cadena Lerma, Cesar | ETH Zurich
    Author: Nieto, Juan | ETH Zurich
 
    keyword: SLAM; Aerial Systems: Perception and Autonomy; Mapping

    Abstract : Globally consistent dense maps are a key requirement for long-term robot navigation in complex environments. While previous works have addressed the challenges of dense mapping and global consistency, most require more computational resources than may be available on-board small robots. We propose a framework that creates globally consistent volumetric maps on a CPU and is lightweight enough to run on computationally constrained platforms.<p>Our approach represents the environment as a collection of overlapping SDF submaps, and maintains global consistency by computing an optimal alignment of the submap collection. By exploiting the underlying SDF representation, we generate correspondence-free constraints between submap pairs that are computationally efficient enough to optimize the global problem each time a new submap is added. </p><p>We deploy the proposed system on a hexacopter MAV with an Intel i7-8650U CPU in two realistic scenarios: mapping a large-scale area using a 3D LiDAR, and mapping an industrial space using a RGB-D camera. In the large-scale outdoor experiments, the system optimizes a 120x80m map in less than 4s and produces absolute trajectory RMSEs of less than 1m over 400m trajectories. Our complete system, called voxgraph, will be available open source.

- DeepFactors: Real-Time Probabilistic Dense Monocular SLAM

    Author: Czarnowski, Jan | Imperial College London
    Author: Laidlow, Tristan | Imperial College London
    Author: Clark, Ronald | Imperial College London
    Author: Davison, Andrew J | Imperial College London
 
    keyword: SLAM; Deep Learning in Robotics and Automation; Mapping

    Abstract : The ability to estimate rich geometry and camera motion from monocular imagery is fundamental to future interactive robotics and augmented reality applications. Different approaches have been proposed that vary in scene geometry representation (sparse landmarks, dense maps), the consistency metric used for optimising the multi-view problem, and the use of learned priors. We present a SLAM system that unifies these methods in a probabilistic framework while still maintaining real-time performance. This is achieved through the use of a learned compact depth map representation and reformulating three different types of errors: photometric, reprojection and geometric, which we make use of within standard factor graph software. We evaluate our system on trajectory estimation and depth reconstruction on real-world sequences and present various examples of estimated dense geometry.

- DOOR-SLAM: Distributed, Online, and Outlier Resilient SLAM for Robotic Teams

    Author: Lajoie, Pierre-Yves | École Polytechnique De Montr�al
    Author: Ramtoula, Benjamin | École Polytechnique De Montr�al, École Polytechnique Fédérale De
    Author: Chang, Yun | MIT
    Author: Carlone, Luca | Massachusetts Institute of Technology
    Author: Beltrame, Giovanni | Ecole Polytechnique De Montreal
 
    keyword: SLAM; Multi-Robot Systems; Localization

    Abstract : To achieve collaborative tasks, robots in a team need to have a shared understanding of the environment and their location within it. Distributed Simultaneous Localization and Mapping (SLAM) offers a practical solution to localize the robots without relying on an external positioning system (e.g. GPS) and with minimal information exchange. Unfortunately, current distributed SLAM systems are vulnerable to perception outliers and therefore tend to use very conservative parameters for inter-robot place recognition. However, being too conservative comes at the cost of rejecting many valid loop-closure candidates, which results in less accurate trajectory estimates. This paper introduces DOOR-SLAM, a fully distributed SLAM system with an outlier rejection mechanism that can work with less conservative parameters. DOOR-SLAM is based on peer-to-peer communication and does not require full connectivity among the robots. DOOR-SLAM includes two key modules: a pose graph optimizer combined with a distributed pairwise consistent measurement set maximization algorithm to reject spurious inter-robot loop closures; and a distributed SLAM front-end that detects inter-robot loop closures without exchanging raw sensor data. The system has been evaluated in simulations, benchmarking datasets, and field experiments, including tests in GPS-denied subterranean environments.

- Windowed Bundle Adjustment Framework for Unsupervised Learning of Monocular Depth Estimation with U-Net Extension and Clip Loss

    Author: Zhou, Lipu | Carnegie Mellon University
    Author: Kaess, Michael | Carnegie Mellon University
 
    keyword: SLAM; Mapping

    Abstract : This paper presents a self-supervised framework for learning depth from monocular videos. In particular, the main contributions of this paper include: (1) We present a windowed bundle adjustment framework to train the network. Compared to most previous works that only consider constraints from consecutive frames, our framework increases the camera baseline and introduces more constraints to avoid overfitting. (2) We extend the widely used U-Net architecture by applying a Spatial Pyramid Net (SPN) and a Super Resolution Net (SRN). The SPN fuses information from an image spatial pyramid for the depth estimation, which addresses the context information attenuation problem of the original U-Net. The SRN learns to estimate a high resolution depth map from a low resolution image, which can benefit the recovery of details. (3) We adopt a clip loss function to handle moving objects and occlusions that were solved by designing complicated network or requiring extra information (such as segmentation mask [1]) in previous works. Experimental results show that our algorithm provides state-of-the-art results on the KITTI benchmark.

- StructVIO : Visual-Inertial Odometry with Structural Regularity of Man-Made Environments (I)

    Author: Zou, Danping | Shanghai Jiao Ton University
    Author: Wu, Yuanxin | Shanghai Jiao Tong University
    Author: Pei, Ling | Shanghai Jiao Tong University
    Author: Ling, Haibin | Temple University
    Author: Yu, Wenxian | Shanghai Jiao Tong University
 
    keyword: SLAM; Localization; Visual-Based Navigation

    Abstract : In this paper, we propose a novel visual-inertial odometry (VIO) approach that adopts structural regularity in man-made environments. Instead of using Manhattan world assumption, we use Atlanta world model to describe such regularity. An Atlanta world is a world that contains multiple local Manhattan worlds with different heading directions. Each local Manhattan world is detected on the fly, and their headings are gradually refined by the state estimator when new observations are received. With full exploration of structural lines that aligned with each local Manhattan worlds, our VIO method becomes more accurate and robust, as well as more flexible to different kinds of complex man-made environments. Through benchmark tests and real-world tests, the results show that the proposed approach outperforms existing visual-inertial systems in large-scale man-made environments.


- Flow-Motion and Depth Network for Monocular Stereo and Beyond

    Author: Wang, Kaixuan | Hong Kong University of Science and Technology
    Author: Shen, Shaojie | Hong Kong University of Science and Technology
 
    keyword: SLAM; Visual Learning; Aerial Systems: Perception and Autonomy

    Abstract : We propose a learning-based method that solves monocular stereo and can be extended to fuse depth information from multiple target frames. Given two unconstrained images from a monocular camera with known intrinsic calibration, our network estimates relative camera poses and the depth map of the source image. The core contribution of the proposed method is threefold. First, a network is tailored for static scenes that jointly estimates the optical flow and camera motion. By the joint estimation, the optical flow search space is gradually reduced resulting in an efficient and accurate flow estimation. Second, a novel triangulation layer is proposed to encode the estimated optical flow and camera motion while avoiding common numerical issues caused by epipolar. Third, beyond two-view depth estimation, we further extend the above networks to fuse depth information from multiple target images and estimate the depth map of the source image. To further benefit the research community, we introduce tools to generate photorealistic structure-from-motion datasets such that deep networks can be well trained and evaluated. The proposed method is compared with previous methods and achieves state-of-the-art results within less time. Images from real-world applications and Google Earth are used to demonstrate the generalization ability of the method.

- Online LiDAR-SLAM for Legged Robots with Robust Registration and Deep-Learned Loop Closure

    Author: Ramezani, Milad | University of Oxford
    Author: Tinchev, Georgi | University of Oxford
    Author: Iuganov, Egor | University of Oxford
    Author: Fallon, Maurice | University of Oxford
 
    keyword: SLAM; Legged Robots; Deep Learning in Robotics and Automation

    Abstract : In this paper, we present a 3D factor-graph LiDAR-SLAM system which incorporates a state-of-the-art deeply learned feature-based loop closure detector to enable a legged robot to localize and map in industrial environments. Point clouds are accumulated using an inertial-kinematic state estimator before being aligned using ICP registration. To close loops we use a loop proposal mechanism which matches individual segments between clouds. We trained a descriptor offline to match these segments. The efficiency of our method comes from carefully designing the network architecture to minimize the number of parameters such that this deep learning method can be deployed in real-time using only the CPU of a legged robot, a major contribution of this work. The set of odometry and loop closure factors are updated using pose graph optimization. Finally we present an efficient risk alignment prediction method which verifies the reliability of the registrations. Experimental results at an industrial facility demonstrated the robustness and flexibility of our system, including autonomous following paths derived from the SLAM map.

- Hybrid Camera Pose Estimation with Online Partitioning for SLAM

    Author: Li, Xinyi | Temple University
    Author: Ling, Haibin | Temple University
 
    keyword: SLAM; Localization; Visual-Based Navigation

    Abstract : This paper presents a hybrid real-time camera pose estimation framework with a novel partitioning scheme and introduces motion averaging to monocular Simultaneous Localization and Mapping (SLAM) systems. Breaking through the limitations of fixed-size temporal partitioning in most conventional SLAM pipelines, the proposed approach significantly improves the accuracy of local bundle adjustment by gathering spatially-strongly-connected cameras into each block. With the dynamic initialization using intermediate computation values, our proposed self-adaptive Levenberg-Marquardt solver achieves a quadratic convergence rate to further enhance the efficiency of the local optimization. Moreover, the dense data association between blocks by virtue of our co-visibility-based partitioning enables us to explore and implement motion averaging to efficiently align the blocks globally, updating camera motion estimations on-the-fly. Experiment results on benchmarks convincingly demonstrate the practicality and robustness of our proposed approach by outperforming conventional SLAM/SfM systems by orders of magnitude.

- Analysis of Minima for Geodesic and Chordal Cost for a Minimal 2D Pose-Graph SLAM Problem

    Author: Kong, Felix Honglim | The University of Technology Sydney
    Author: Zhao, Jiaheng | University of Technology Sydney
    Author: Zhao, Liang | University of Technology Sydney
    Author: Huang, Shoudong | University of Technology, Sydney
 
    keyword: SLAM

    Abstract : In this paper, we show that for a minimal 2D pose-graph SLAM problem, even in the ideal case of perfect measurements and spherical covariance, using geodesic distance (in 2D, the ``wrap function'') to compare angles results in multiple suboptimal local minima. We numerically estimate regions of attraction to these local minima for some examples, give evidence to show that they are of nonzero measure, and that these regions grow in size as noise is added. In contrast, under the same assumptions, we show that the chordal distance representation of angle error has a unique minimum up to periodicity. For chordal cost, we find that initial conditions failing to converge to the global minimum are far fewer, fail because of numerical issues, and do not seem to grow with noise in our examples.

- Voxel Map for Visual SLAM

    Author: Muglikar,, Manasi | University of Zurich
    Author: Zhang, Zichao | Robotics and Perception Group, University of Zurich
    Author: Scaramuzza, Davide | University of Zurich
 
    keyword: SLAM; Mapping; Localization

    Abstract : In modern visual SLAM systems, it is a standard practice to retrieve potential candidate map points from overlapping keyframes for further feature matching or direct tracking. In this work, we argue that keyframes are not the optimal choice for this task, due to several inherent limitations, such as weak geometric reasoning and poor scalability. We propose a voxel-map representation to ef&#64257;ciently retrieve map points for visual SLAM. In particular, we organize the map points in a regular voxel grid. Visible points from a camera pose are queried by sampling the camera frustum in a raycasting manner, which can be done in constant time using an ef&#64257;cient voxel hashing method. Compared with keyframes, the retrieved points using our method are geometrically guaranteed to fall in the camera &#64257;eld-of-view, and occluded points can be identi&#64257;ed and removed to a certain extend. This method also naturally scales up to large scenes and complicated multicamera con&#64257;gurations. Experimental results show that our voxel map representation is as ef&#64257;cient as a keyframe map with 5 keyframes and provides signi&#64257;cantly higher localization accuracy (average 46% improvement in RMSE) on the EuRoC dataset. The proposed voxel-map representation is a general approach to a fundamental functionality in visual SLAM and widely applicable

## Deep Learning in Robotics and Automation 

- Learning 3D-Aware Egocentric Spatial-Temporal Interaction Via Graph Convolutional Networks

    Author: Li, Chengxi | Purdue University
    Author: Meng, Yue | IBM T. J. Watson Research Center
    Author: Chan, Stanley | Purdue University
    Author: Chen, Yi-Ting | Honda Research Institute USA
 
    keyword: Deep Learning in Robotics and Automation; Semantic Scene Understanding; Computer Vision for Transportation

    Abstract : To enable intelligent automated driving systems, a promising strategy is to understand how human drives and interacts with road users in complicated driving situations. In this paper, we propose a 3D-aware egocentric spatial-temporal interaction framework for automated driving applications. Graph convolution networks (GCN) is devised for interaction modeling. We introduce three novel concepts into GCN. First, we decompose egocentric interactions into ego-thing and egostuff interaction, modeled by two GCNs. In both GCNs, ego nodes are introduced to encode the interaction between thing objects (e.g., car and pedestrian), and interaction between stuff objects (e.g., lane marking and traffic light). Second, objects' 3D locations are explicitly incorporated into GCN to better model egocentric interactions. Third, to implement ego-stuff interaction in GCN, we propose a MaskAlign operation to extract features for irregular objects. We validate the proposed framework on tactical driver behavior recognition. Extensive experiments are conducted using Honda Research Institute Driving Dataset, the largest dataset with diverse tactical driver behavior annotations. Our framework demonstrates substantial performance boost over baselines on the two experimental settings by 3.9% and 6.0%, respectively. Furthermore, we visualize the learned affinity matrices, which encode ego-thing and ego-stuff interactions, to showcase the proposed framework can capture interactions effectively.

- C-3PO: Cyclic-Three-Phase Optimization for Human-Robot Motion Retargeting Based on Reinforcement Learning

    Author: Kim, Taewoo | University of Science and Technology
    Author: Lee, Joo-Haeng | ETRI
 
    keyword: Deep Learning in Robotics and Automation; Social Human-Robot Interaction; Gesture, Posture and Facial Expressions

    Abstract : Motion retargeting between heterogeneous polymorphs with different sizes and kinematic configurations requires a comprehensive knowledge of (inverse) kinematics. Moreover, it is non-trivial to provide a kinematic independent general solution. In this study, we developed a cyclic three-phase optimization method based on deep reinforcement learning for human-robot motion retargeting. The motion retargeting learning is performed using refined data in a latent space by the cyclic and filtering paths of our method. In addition, the humanin-the-loop based three-phase approach provides a framework for the improvement of the motion retargeting policy by both quantitative and qualitative manners. Using the proposed C-3PO method, we were successfully able to learn the motion retargeting skill between the human skeleton and motion of the multiple robots such as NAO, Pepper, Baxter and C-3PO.

- AP-MTL: Attention Pruned Multi-Task Learning Model for Real-Time Instrument Detection and Segmentation in Robot-Assisted Surgery

    Author: Islam, Mobarakol | National University of Singapore
    Author: Vs, Vibashan | National Institute of Technology
    Author: Ren, Hongliang | Faculty of Engineering, National University of Singapore
 
    keyword: Deep Learning in Robotics and Automation; Medical Robots and Systems; Surgical Robotics: Laparoscopy

    Abstract : Surgical scene understanding and multi-tasking learning are crucial for image-guided robotic surgery. Training a real-time robotic system for the detection and segmentation of high-resolution images provides a challenging problem with the limited computational resource. The perception drawn can be applied in effective real-time feedback, surgical skill assessment, and human-robot collaborative surgeries to enhance surgical outcomes. For this purpose, we develop a novel end-to-end trainable real-time Multi-Task Learning (MTL) model with weight-shared encoder and task-aware detection and segmentation decoders. Optimization of multiple tasks at the same convergence point is vital and present a complex problem. Thus, we propose an asynchronous task-aware optimization (ATO) technique to calculate task-oriented gradients and train the decoders independently. Moreover, MTL models are always computationally expensive, which hinder real-time applications. To address this challenge, we introduce a global attention dynamic pruning (GADP) by removing less significant and sparse parameters. We further design a skip squeeze and excitation (SE) module, which suppresses weak features, excites significant features and performs dynamic spatial and channel-wise feature re-calibration. Validating on the robotic instrument segmentation dataset of MICCAI endoscopic vision challenge, our model significantly outperforms state-of-the-art segmentation and detection models, including best-performed mod

- Automatic Gesture Recognition in Robot-Assisted Surgery with Reinforcement Learning and Tree Search

    Author: Gao, Xiaojie | The Chinese University of Hong Kong
    Author: Jin, Yueming | The Chinese University of Hong Kong
    Author: Dou, Qi | The Chinese University of Hong Kong
    Author: Heng, Pheng Ann | The Chinese University of Hong Kong
 
    keyword: Computer Vision for Medical Robotics; Deep Learning in Robotics and Automation; Recognition

    Abstract : Automatic surgical gesture recognition is fundamental for improving intelligence in robot-assisted surgery, such as conducting complicated tasks of surgery surveillance and skill evaluation. However, current methods treat each frame individually and produce the outcomes without effective consideration on future information. In this paper, we propose a framework based on reinforcement learning and tree search for joint surgical gesture segmentation and classification. An agent is trained to segment and classify the surgical video in a human-like manner whose direct decisions are re-considered by tree search appropriately. Our proposed tree search algorithm unites the outputs from two designed neural networks, i.e., policy and value network. With the integration of complementary information from distinct models, our framework is able to achieve the better performance than baseline methods using either of the neural networks. For an overall evaluation, our developed approach consistently outperforms the existing methods on the suturing task of JIGSAWS dataset in terms of accuracy, edit score and F1 score. Our study highlights the utilization of tree search to refine actions in reinforcement learning framework for surgical robotic applications.

- Towards Privacy-Preserving Ego-Motion Estimation Using an Extremely Low-Resolution Camera

    Author: Shariati, Armon | University of Pennsylvania
    Author: Holz, Christian | ETH Zurich
    Author: Sinha, Sudipta | Microsoft Research
 
    keyword: Deep Learning in Robotics and Automation; SLAM; Human-Centered Robotics

    Abstract : Ego-motion estimation is a core task in robotic systems as well as in augmented and virtual reality applications. It is often solved using visual-inertial odometry, which involves using one or more emph{always-on} cameras on mobile robots and wearable devices. As consumers increasingly use such devices in their homes and workplaces, which are filled with sensitive details, the role of privacy in such camera-based approaches is of ever increasing importance.<p>In this paper, we introduce the first solution to perform emph{privacy-preserving} ego-motion estimation. We recover camera ego-motion from an extremely low-resolution monocular camera by estimating dense optical flow at a higher spatial resolution (i.e., 4x super resolution). We propose textit{SRFNet} for directly estimating Super-Resolved Flow, a novel convolutional neural network model that is trained in a supervised setting using ground-truth optical flow. We also present a weakly supervised approach for training a variant of SRFNet on real videos where ground truth flow is unavailable. On image pairs with known relative camera orientations, we use SRFNet to predict the auto-epipolar flow that arises from pure camera translation, from which we robustly estimate the camera translation direction. We evaluate our super-resolved optical flow estimates and camera translation direction estimates on the Sintel and KITTI odometry datasets, where our methods outperform several baselines. Our results indicate that robust ego

- ACNN: A Full Resolution DCNN for Medical Image Segmentation

    Author: Zhou, Xiao-Yun | Imperial College London
    Author: Zheng, Jian-Qing | University of Oxford
    Author: Li, Peichao | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Medical Robotics; Medical Robots and Systems

    Abstract : Deep Convolutional Neural Networks (DCNNs) are used extensively in medical image segmentation and hence 3D navigation for robot-assisted Minimally Invasive Surgeries (MISs). However, current DCNNs usually use down sampling layers for increasing the receptive field and gaining     Abstract semantic information. These down sampling layers decrease the spatial dimension of feature maps, which can be detrimental to image segmentation. Atrous convolution is an alternative for the down sampling layer. It increases the receptive field whilst maintains the spatial dimension of feature maps. In this paper, a method for effective atrous rate setting is proposed to achieve the largest and fully-covered receptive field with a minimum number of atrous convolutional layers. Furthermore, a new and full resolution DCNN - Atrous Convolutional Neural Network (ACNN), which incorporates cascaded atrous II-blocks, residual learning and Instance Normalization (IN) is proposed. Application results of the proposed ACNN to Magnetic Resonance Imaging (MRI) and Computed Tomography (CT) image segmentation demonstrate that the proposed ACNN can achieve higher segmentation Intersection over Unions (IoUs) than U-Net and Deeplabv3+, but with reduced trainable parameters.

- RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, &amp; New Methods

    Author: Herath, Sachini | Simon Fraser University
    Author: Yan, Hang | Washington University in St. Louis
    Author: Furakawa, Yasutaka | Simon Fraser University
 
    keyword: Deep Learning in Robotics and Automation; Human Detection and Tracking; Sensor Fusion

    Abstract : This paper sets a new foundation for data-driven inertial navigation research, where the task is the estimation of horizontal positions and heading direction of a moving subject from a sequence of IMU sensor measurements from a phone. In contrast to existing methods, our method can handle varying phone orientations and placements.<p>More concretely, the paper presents 1) a new benchmark containing more than 40 hours of IMU sensor data from 100 human subjects with ground-truth 3D trajectories under natural human motions; 2) novel neural inertial navigation architectures, making significant improvements for challenging motion cases; and 3) qualitative and quantitative evaluations of the competing methods over three inertial navigation benchmarks. We share the code and data to promote further research (http://ronin.cs.sfu.ca).

- Segmenting 2K-Videos at 36.5 FPS with 24.3 GFLOPs: Accurate and Lightweight Realtime Semantic Segmentation Network

    Author: Oh, Dokwan | Samsung Electronics
    Author: Ji, Daehyun | Samsung Advanced Institute of Technology
    Author: Jang, Cheolhun | Samsung Advanced Institute of Technology
    Author: Hyun, Yoonsuk | Samsung Electronics
    Author: Bae, Hong S. | SAMSUNG
    Author: Hwang, Sung Ju | KAIST
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization; AI-Based Methods

    Abstract : We propose a fast and lightweight end-to-end convolutional network architecture for real-time segmentation of high resolution videos, NfS-SegNet, that can segment 2K-videos at 36.5 FPS with 24.3 GFLOPS. This speed and computation-efficiency is due to following reasons: 1) The encoder network, NfS-Net, is optimized for speed with simple building blocks without memory-heavy operations such as depthwise convolutions, and outperforms state-of-the-art lightweight CNN architectures such as SqueezeNet [2], MobileNet v1 [3] &amp; v2 [4] and ShuffleNet v1 [5] &amp; v2 [6] on image classification with significantly higher speed. 2) The NfS-SegNet has an asymmetric architecture with deeper encoder and shallow decoder, whose design is based on our empirical finding that the decoder is the main bottleneck in computation with relatively small contribution to the final performance. 3) Our novel uncertainty-aware knowledge distillation method guides the teacher model to focus its knowledge transfer on the most difficult image regions. We validate the performance of NfS-SegNet with the CITYSCAPE [1] benchmark, on which it achieves state-of-the-art performance among lightweight segmentation models in terms of both accuracy and speed.

- Temporally Consistent Horizon Lines

    Author: Kluger, Florian | University of Hannover
    Author: Ackermann, Hanno | Institute of Information Processing, Leibniz Universitét Hannove
    Author: Yang, Michael Ying | University of Twente
    Author: Rosenhahn, Bodo | Institute of Information Processing, Leibniz Universitét Hannove
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Other Robotic Applications; Computer Vision for Transportation

    Abstract : The horizon line is an important geometric feature for many image processing and scene understanding tasks in computer vision. For instance, in navigation of autonomous vehicles or driver assistance, it can be used to improve 3D reconstruction as well as for semantic interpretation of dynamic environments. While both algorithms and datasets exist for single images, the problem of horizon line estimation from video sequences has not gained attention. In this paper, we show how convolutional neural networks are able to utilise the temporal consistency imposed by video sequences in order to increase the accuracy and reduce the variance of horizon line estimates. A novel CNN architecture with an improved residual convolutional LSTM is presented for temporally consistent horizon line estimation. We propose an adaptive loss function that ensures stable training as well as accurate results. Furthermore, we introduce an extension of the KITTI dataset which contains precise horizon line labels for 43699 images across 72 video sequences. A comprehensive evaluation shows that the proposed approach consistently achieves superior performance compared with existing methods.

- Full-Scale Continuous Synthetic Sonar Data Generation with Markov Conditional Generative Adversarial Networks

    Author: Jegorova, Marija | University of Edinburgh
    Author: Karjalainen, Antti Ilari | SeeByte Ltd
    Author: Vazquez-Diosdado, Jose | SeeByte Ltd
    Author: Hospedales, Timothy | Queen Mary University of London
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization; Simulation and Animation

    Abstract : Deployment and operation of autonomous underwater vehicles is expensive and time-consuming. High-quality realistic sonar data simulation could be of benefit to multiple applications, including training of human operators for post-mission analysis, as well as tuning and validation of autonomous target recognition (ATR) systems for underwater vehicles. Producing realistic synthetic sonar imagery is a challenging problem as the model has to account for specific artefacts of real acoustic sensors, vehicle altitude, and a variety of environmental factors. We propose a novel method for generating realistic-looking sonar side-scans of full-length missions, called Markov Conditional pix2pix (MC-pix2pix). Quantitative assessment results confirm that the quality of the produced data is almost indistinguishable from real. Furthermore, we show that bootstrapping ATR systems with MC-pix2pix data can improve the performance. Synthetic data is generated 18 times faster than real acquisition speed, with full user control over the topography of the generated data.

- Real-Time Soft Body 3D Proprioception Via Deep Vision-Based Sensing

    Author: Wang, Ruoyu | New York University
    Author: Wang, Shiheng | New York University
    Author: Du, Songyu | New York University
    Author: Xiao, Erdong | New York University
    Author: Yuan, Wenzhen | Carnegie Mellon University
    Author: Feng, Chen | New York University
 
    keyword: Deep Learning in Robotics and Automation; Modeling, Control, and Learning for Soft Robots; Computer Vision for Automation

    Abstract : Soft bodies made from flexible and deformable materials are popular in many robotics applications, but their proprioceptive sensing has been a long-standing challenge. In other words, there has hardly been a method to measure and model the high-dimensional 3D shapes of soft bodies with internal sensors. We propose a framework to measure the high-resolution 3D shapes of soft bodies in real-time with embedded cameras. The cameras capture visual patterns inside a soft body, and a convolutional neural network (CNN) produces a latent code representing the deformation state, which can then be used to reconstruct the body's 3D shape using another neural network. We test the framework on various soft bodies, such as a Baymax-shaped toy, a latex balloon, and some soft robot fingers, and achieve real-time computation (&lt;2.5ms/frame) for robust shape estimation with high precision (&lt;1% relative error) and high resolution. We believe the method could be applied to soft robotics and human-robot interaction for proprioceptive shape sensing. Our code is available at: https://ai4ce.github.io/DeepSoRo/.

- A General Framework for Uncertainty Estimation in Deep Learning

    Author: Loquercio, Antonio | UZH, University of Zurich
    Author: Segu, Mattia | ETH Zurich
    Author: Scaramuzza, Davide | University of Zurich
 
    keyword: Deep Learning in Robotics and Automation; AI-Based Methods; Autonomous Agents

    Abstract : Neural networks predictions are unreliable when the input sample is out of the training distribution or corrupted by noise. Being able to detect such failures automatically is fundamental to integrate deep learning algorithms into robotics. Current approaches for uncertainty estimation of neural networks require changes to the network and optimization process, typically ignore prior knowledge about the data, and tend to make over-simplifying assumptions which underestimate uncertainty. To address these limitations, we propose a novel framework for uncertainty estimation. Based on Bayesian belief networks and Monte-Carlo sampling, our framework not only fully models the different sources of prediction uncertainty, but also incorporates prior data information, e.g. sensor noise. We show theoretically that this gives us the ability to capture uncertainty better than existing methods. In addition, our framework has several desirable properties: (i) it is agnostic to the network architecture and task; (ii) it does not require changes in the optimization process; (iii) it can be applied to already trained architectures. We thoroughly validate the proposed framework through extensive experiments on both computer vision and control tasks, where we outperform previous methods by up to 23% in accuracy.


- Learning Local Behavioral Sequences to Better Infer Non-Local Properties in Real Multi-Robot Systems

    Author: Choi, Taeyeong | Arizona State University
    Author: Kang, Sehyeok | Arizona State University
    Author: Pavlic, Theodore | Arizona State University
 
    keyword: Deep Learning in Robotics and Automation; Recognition; Localization

    Abstract : When members of a multi-robot team follow regular motion rules sensitive to robots and other environmental factors within sensing range, the team itself may become an informational fabric for gaining situational awareness without explicit signalling among robots. In our previous work [1], we used machine learning to develop a scalable module, trained only on data from 3-robot teams, that could predict the positions of all robots in larger multi-robot teams based only on observations of the movement of a robot's nearest neighbor. Not only was this approach scalable from 3-to-many robots, but it did not require knowledge of the control laws of the robots under observation, as would a traditional observer-based approach. However, performance was only tested in simulation and could only be a substitute for explicit communication for short periods of time or in cases of very low sensing noise. In this work, we apply more sophisticated machine learning methods to data from a physically realized robotic team to develop Remote Teammate Localization (ReTLo) modules that can be used in realistic environments. To be specific, we adopt Long�Short-Term�Memory (LSTM) [2] to learn the evolution of behaviors in a modular team, which has the effect of greatly reducing errors from regression outcomes. In contrast with our previous work in simulation, all of the experiments conducted in this work were conducted on the Thymio physical, two-wheeled robotic platform.

- Unsupervised Geometry-Aware Deep LiDAR Odometry

    Author: Cho, Younggun | Korea Advanced Institute of Science and Technology
    Author: Kim, Giseop | KAIST(Korea Advanced Institute of Science and Technology)
    Author: Kim, Ayoung | Korea Advanced Institute of Science Technology
 
    keyword: Deep Learning in Robotics and Automation; SLAM; Range Sensing

    Abstract : Learning-based ego-motion estimation approaches have recently drawn strong interest from researchers, mostly focusing on visual perception. A few learning-based approaches using Light Detection and Ranging (LiDAR) have been reported; however, they heavily rely on a supervised learning manner. Despite the meaningful performance of these approaches, supervised training requires ground-truth pose labels, which is the bottleneck for real-world applications. Differing from these approaches, we focus on unsupervised learning for LiDAR odometry (LO) without trainable labels. Achieving trainable LO in an unsupervised manner, we introduce the uncertainty-aware loss with geometric confidence, thereby allowing the reliability of the proposed pipeline. Evaluation on the KITTI, Complex Urban, and Oxford RobotCar datasets demonstrate the prominent performance of the proposed method compared to conventional model-based methods. The proposed method shows a comparable result against SuMa (in KITTI), LeGO-LOAM (in Complex Urban), and Stereo-VO (in Oxford RobotCar). The video and extra-information of the paper are described in https://sites.google.com/view/deeplo.

- SA-Net: Robust State-Action Recognition for Learning from Observations

    Author: Soans, Nihal | University of Georgia
    Author: Asali, Ehsan | University of Georgia
    Author: Hong, Yi | University of Georgia
    Author: Doshi, Prashant | University of Georgia
 
    keyword: Deep Learning in Robotics and Automation; Learning from Demonstration; RGB-D Perception

    Abstract : Learning from observation (LfO) offers a new paradigm for transferring task behavior to robots. LfO requires the robot to observe the task being performed and decompose the sensed streaming data into sequences of state-action pairs, which are then input to LfO methods. Thus, recognizing the state-action pairs correctly and quickly in sensed data is a crucial prerequisite. We present SA-Net a deep neural network architecture that recognizes state-action pairs from RGB-D data streams. SA-Net performs well in two replicated robotic applications of LfO -- one involving mobile ground robots and another involving a robotic manipulator -- which demonstrates that the architecture could generalize well to differing contexts. Comprehensive evaluations including deployment on a physical robot show that SA-Net significantly improves on the accuracy of the previous methods under various conditions.

- A Generative Approach for Socially Compliant Navigation

    Author: Tsai, Chieh-En | Carnegie Mellon University
    Author: Oh, Jean | Carnegie Mellon University
 
    keyword: Deep Learning in Robotics and Automation; Social Human-Robot Interaction; Human-Centered Robotics

    Abstract : Robots navigating in human crowds need to optimize their paths not only for their task performance but also for their compliance to social norms. One of the key challenges in this context is the lack of standard metrics for evaluating and optimizing a socially compliant behavior. Existing works in social navigation can be grouped according to the differences in their optimization objectives. For instance, the reinforcement learning approaches tend to optimize on the comfort aspect of the socially compliant navigation, whereas the inverse reinforcement learning approaches are designed to achieve natural behavior. In this paper, we propose NaviGAN, a generative navigation algorithm that jointly optimizes both of the comfort and naturalness aspects. Our approach is designed as an adversarial training framework that can learn to generate a navigation path that is both optimized for achieving a goal and for complying with latent social rules. A set of experiments has been carried out on multiple datasets to demonstrate the strengths of the proposed approach quantitatively. We also perform extensive experiments using a physical robot in a real-world environment to qualitatively evaluate the trained social navigation behavior. The video recordings of the robot experiments can be found in the link: https://youtu.be/61blDymjCpw.

- Scalable Multi-Task Imitation Learning with Autonomous Improvement

    Author: Singh, Avi | UC Berkeley
    Author: Jang, Eric | Google
    Author: Irpan, Alexander | Google
    Author: Kappler, Daniel | X (Google)
    Author: Dalal, Murtaza | University of California Berkeley
    Author: Levine, Sergey | UC Berkeley
    Author: Khansari, Mohi | Google X
    Author: Finn, Chelsea | Stanford University
 
    keyword: Deep Learning in Robotics and Automation

    Abstract : While robot learning has demonstrated promising results for enabling robots to automatically acquire new skills, a critical challenge in deploying learning-based systems is scale: acquiring enough data for the robot to effectively generalize broadly. Imitation learning, in particular, has remained a stable and powerful approach for robot learning, but critically relies on expert operators for data collection. In this work, we target this challenge, aiming to build an imitation learning system that can continuously improve through autonomous data collection, while simultaneously avoiding the explicit use of reinforcement learning, to maintain the stability, simplicity, and scalability of supervised imitation. To accomplish this, we cast the problem of imitation with autonomous improvement into a multi-task setting. We utilize the insight that, in a multi-task setting, a failed attempt at one task might represent a successful attempt at another task. This allows us to leverage the robot's own trials as demonstrations for tasks other than the one that the robot actually attempted. In contrast to prior imitation learning approaches, our method can autonomously collect data with sparse supervision for continuous improvement, and in contrast to reinforcement learning algorithms, our method can effectively improve from sparse, task-agnostic reward signals.

- Motion2Vec: Semi-Supervised Representation Learning from Surgical Videos

    Author: Tanwani, Ajay Kumar | UC Berkeley
    Author: Sermanet, Pierre | Google
    Author: Yan, Andy | UC Berkeley
    Author: Anand, Raghav | UC Berkeley
    Author: Phielipp, Mariano | Intel Corporation
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Deep Learning in Robotics and Automation; Learning from Demonstration; Learning and Adaptive Systems

    Abstract : Learning meaningful visual representations in an embedding space can facilitate generalization in downstream tasks such as action segmentation and imitation. In this paper, we learn a motion-centric representation of surgical video demonstrations by grouping them into action segments/sub-goals/options in a semi-supervised manner. We present Motion2Vec, an algorithm that learns a deep embedding feature space from video observations by minimizing a metric learning loss in a Siamese network: images from the same action segment are pulled together while pushed away from randomly sampled images of other segments. The embeddings are iteratively segmented with a recurrent neural network for a given parametrization of the embedding space after pre-training the Siamese network. We only use a small set of labeled video segments to semantically align the embedding space and assign pseudo-labels to the remaining unlabeled data by inference on the learned model parameters. We demonstrate the use of this representation to imitate surgical suturing kinematic motions from publicly available videos of the JIGSAWS dataset. Results give 85.5% segmentation accuracy on average suggesting performance improvement over several state-of-the-art baselines, while kinematic pose imitation gives 0.94 centimeter error in position per observation on the test set. Videos, code and data are available at: https://sites.google.com/view/motion2vec


- PointAtrousGraph: Deep Hierarchical Encoder-Decoder with Point Atrous Convolution for Unorganized 3D Points

    Author: Pan, Liang | National University of Singapore
    Author: Chew, Chee Meng | National University of Singapore
    Author: Lee, Gim Hee | National University of Singapore
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Other Robotic Applications; Semantic Scene Understanding

    Abstract : Motivated by the success of encoding multi-scale contextual information for image analysis, we propose our PointAtrousGraph (PAG) - a deep permutation-invariant hierarchical encoder-decoder for efficiently exploiting multi-scale edge features in point clouds. Our PAG is constructed by several novel modules, such as Point Atrous Convolution (PAC), Edge-preserved Pooling (EP) and Edge-preserved Unpooling (EU). Similar with atrous convolution, our PAC can effectively enlarge receptive fields of filters and thus densely learn multi-scale point features. Following the idea of non-overlapping max-pooling operations, we propose our EP to preserve critical edge features during subsampling. Correspondingly, our EU modules gradually recover spatial information for edge features. In addition, we introduce chained skip subsampling/upsampling modules that directly propagate edge features to the final stage. Particularly, our proposed auxiliary loss functions can further improve our performance. Experimental results show that our PAG outperform previous state-of-the-art methods on various 3D semantic perception applications.

- Learning Error Models for Graph SLAM

    Author: Reymann, Christophe | LAAS-CNRS
    Author: Lacroix, Simon | LAAS/CNRS
 
    keyword: Deep Learning in Robotics and Automation; SLAM; Mapping

    Abstract : Following recent developments, this paper investigates the possibility to predict uncertainty models for monocular graph SLAM using topological features of the problem. An architecture to learn relative (i.e. inter-keyframe) uncertainty models using the resistance distance in the covisibility graph is presented. The proposed architecture is applied to simulated UAV coverage path planning trajectories and an analysis of the approaches strengths and shortcomings is provided.

- SMArT: Training Shallow Memory-Aware Transformers for Robotic Explainability

    Author: Cornia, Marcella | University of Modena and Reggio Emilia
    Author: Baraldi, Lorenzo | Université Degli Studi Di Modena E Reggio Emilia
    Author: Cucchiara, Rita | Université Degli Studi Di Modena E Reggio Emilia
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization

    Abstract : The ability to generate natural language explanations conditioned on the visual perception is a crucial step towards autonomous agents which can explain themselves and communicate with humans. While the research efforts in image and video captioning are giving promising results, this is often done at the expense of the computational requirements of the approaches, limiting their applicability to real contexts. In this paper, we propose a fully-attentive captioning algorithm which can provide state-of-the-art performances on language generation while restricting its computational demands. Our model is inspired by the Transformer model and employs only two Transformer layers in the encoding and decoding stages. Further, it incorporates a novel memory-aware encoding of image regions. Experiments demonstrate that our approach achieves competitive results in terms of caption quality while featuring reduced computational demands. Further, to evaluate its applicability on autonomous agents, we conduct experiments on simulated scenes taken from the perspective of domestic robots.

- A 3D-Deep-Learning-Based Augmented Reality Calibration Method for Robotic Environments Using Depth Sensor Data

    Author: K�stner, Linh | Technische Universitét Berlin
    Author: Frasineanu, Vlad-Catalin | TU Berlin
    Author: Lambrecht, Jens | Technische Universitét Berlin
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization; AI-Based Methods

    Abstract : Augmented Reality and mobile robots are gaining increased attention within industries due to the high potential to make processes cost and time efficient. To facilitate augmented reality, a calibration between the Augmented Reality device and the environment is necessary. This is a challenge when dealing with mobile robots due to the mobility of all entities making the environment dynamic. On this account, we propose a novel approach to calibrate Augmented Reality devices using 3D depth sensor data. We use the depth camera of a Head Mounted Augmented Reality Device, the Microsoft Hololens, for deep learning-based calibration. Therefore, we modified a neural network based on the recently published VoteNet architecture which works directly on raw point cloud input observed by the Hololens. We achieve satisfying results and eliminate external tools like markers, thus enabling a more intuitive and flexible work flow for Augmented Reality integration. The results are adaptable to work with all depth cameras and are promising for further research. Furthermore, we introduce an open source 3D point cloud labeling tool, which is to our knowledge the first open source tool for labeling raw point cloud data.

- Adversarial Feature Training for Generalizable Robotic Visuomotor Control

    Author: Chen, Xi | KTH
    Author: Ghadirzadeh, Ali | KTH Royal Institute of Technology, Aalto University
    Author: Bj�rkman, M�rten | KTH
    Author: Jensfelt, Patric | KTH - Royal Institute of Technology
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation

    Abstract : Deep reinforcement learning (RL) has enabled training action-selection policies, end-to-end, by learning a function which maps image pixels to action outputs. However, it's application to visuomotor robotic policy training has been limited because of the challenge of large-scale data collection when working with physical hardware. A suitable visuomotor policy should perform well not just for the task-setup it has been trained for, but also for all varieties of the task, including novel objects at different viewpoints surrounded by task-irrelevant objects. However, it is impractical for a robotic setup to sufficiently collect interactive samples in a RL framework to generalize well to novel aspects of a task. <p>In this work, we demonstrate that by using adversarial training for domain transfer, it is possible to train visuomotor policies based on RL frameworks, and then transfer the acquired policy to other novel task domains. We propose to leverage the deep RL capabilities to learn complex visuomotor skills for uncomplicated task setups, and then exploit transfer learning to generalize to new task domains provided only still images of the task in the target domain. We evaluate our method on two real robotic tasks, picking and pouring, and compare it to a number of prior works, demonstrating its superiority.

- Efficient Bimanual Manipulation Using Learned Task Schemas

    Author: Chitnis, Rohan | Massachusetts Institute of Technology
    Author: Tulsiani, Shubham | Facebook AI Research
    Author: Gupta, Saurabh | UIUC
    Author: Gupta, Abhinav | Carnegie Mellon University
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; Dual Arm Manipulation

    Abstract : We address the problem of effectively composing skills to solve sparse-reward tasks in the real world. Given a set of parameterized skills (such as exerting a force or doing a top grasp at a location), our goal is to learn policies that invoke these skills to efficiently solve such tasks. Our insight is that for many tasks, the learning process can be decomposed into learning a state-independent task schema (a sequence of skills to execute) and a policy to choose the parameterizations of the skills in a state-dependent manner. For such tasks, we show that explicitly modeling the schema's state-independence can yield significant improvements in sample efficiency for model-free reinforcement learning algorithms. Furthermore, these schemas can be transferred to solve related tasks, by simply re-learning the parameterizations with which the skills are invoked. We find that doing so enables learning to solve sparse-reward tasks on real-world robotic systems very efficiently. We validate our approach experimentally over a suite of robotic bimanual manipulation tasks, both in simulation and on real hardware. See videos at http://tinyurl.com/chitnis-schema.

- BayesOD: A Bayesian Approach for Uncertainty Estimation in Deep Object Detectors

    Author: Harakeh, Ali | University of Toronto
    Author: Smart, Michael | University of Waterloo
    Author: Waslander, Steven Lake | University of Toronto
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization; Probability and Statistical Methods

    Abstract : When incorporating deep neural networks into robotic systems, a major challenge is the lack of uncertainty measures associated with their output predictions. Methods for uncertainty estimation in the output of deep object detectors (DNNs) have been proposed in recent works, but have had limited success due to 1) information loss at the detectors non-maximum suppression (NMS) stage, and 2) failure to take into account the multitask, many-to-one nature of anchor-based object detection. To that end, we introduce BayesOD, an uncertainty estimation approach that reformulates the standard object detector inference and Non-Maximum suppression components from a Bayesian perspective. Experiments performed on four common object detection datasets show that BayesOD provides uncertainty estimates that are better correlated with the accuracy of detections, manifesting as a significant reduction of 9.77%-13.13% on the minimum Gaussian uncertainty error metric and a reduction of 1.63%-5.23% on the minimum Categorical uncertainty error metric. Code will be released at https://github.com/asharakeh/bayes-od-rc.

- Learning Object Placements for Relational Instructions by Hallucinating Scene Representations

    Author: Mees, Oier | Albert-Ludwigs-Universitét
    Author: Emek, Alp | University of Freiburg
    Author: Vertens, Johan | University of Freiburg
    Author: Burgard, Wolfram | Toyota Research Institute
 
    keyword: Deep Learning in Robotics and Automation; Cognitive Human-Robot Interaction; Computer Vision for Automation

    Abstract : Human-centered environments contain a wide variety of spatial relations between everyday objects. For autonomous robots to interact with humans effectively in such environments, they should be able to reason about the best way to place objects in order to follow natural language instructions based on spatial relations. In this work, we present a convolutional neural network for estimating pixelwise object placement probabilities for a set of spatial relations from a single input image. During training, our network receives the learning signal by classifying hallucinated high-level scene representations as an auxiliary task. Unlike previous approaches, our method does not require ground truth data for the pixelwise relational probabilities or 3D models of the objects, which significantly expands the applicability in practical robotics scenarios. Our results, based on both real-world data and human-robot experiments, demonstrate the effectiveness of our method in reasoning about the best way to place objects to reproduce a spatial relation. Videos of our experiments can be found at https://youtu.be/zaZkHTWFMKM

- FADNet: A Fast and Accurate Network for Disparity Estimation

    Author: Wang, Qiang | Hong Kong Baptist University
    Author: Shi, Shaohuai | Hong Kong Baptist University
    Author: Zheng, Shizhen | HKBU
    Author: Zhao, Kaiyong | Hong Hong Baptist University
    Author: Chu, Xiaowen | Hong Kong Baptist University
 
    keyword: Deep Learning in Robotics and Automation; RGB-D Perception; Visual Learning

    Abstract : Deep neural networks (DNNs) have achieved great success in the area of computer vision. The disparity estimation problem tends to be addressed by DNNs which achieve much better prediction accuracy in stereo matching than traditional hand-crafted feature based methods. On one hand, however, the designed DNNs require significant memory and computation resources to accurately predict the disparity, especially for those 3D convolution based networks, which makes it difficult for deployment in real-time applications. On the other hand, existing computation-efficient networks lack expression capability in large-scale datasets so that they cannot make an accurate prediction in many scenarios. To this end, we propose an efficient and accurate deep network for disparity estimation named FADNet with three main features: 1) It exploits efficient 2D based correlation layers with stacked blocks to preserve fast computation; 2) It combines the residual structures to make the deeper model easier to learn; 3) It contains multi-scale predictions so as to exploit a multi-scale weight scheduling training technique to improve the accuracy. We conduct experiments to demonstrate the effectiveness of FADNet on two popular datasets, Scene Flow and KITTI 2015. Experimental results show that FADNet achieves state-of-the-art prediction accuracy, and runs at a significant order of magnitude faster speed than existing 3D models. The codes of FADNet are available at https://github.com/HKBU-HPML/FADNet.

- Training Adversarial Agents to Exploit Weaknesses in Deep Control Policies

    Author: Kuutti, Sampo | University of Surrey
    Author: Fallah, Saber | University of Surrey
    Author: Bowden, Richard | University of Surrey
 
    keyword: Deep Learning in Robotics and Automation; Autonomous Agents; Collision Avoidance

    Abstract : Deep learning has become an increasingly com- mon technique for various control problems, such as robotic arm manipulation, robot navigation, and autonomous vehicles. However, the downside of using deep neural networks to learn control policies is their opaque nature and the difficulties of validating their safety. As the networks used to obtain state- of-the-art results become increasingly deep and complex, the rules they have learned and how they operate become more challenging to understand. This presents an issue, since in safety-critical applications the safety of the control policy must be ensured to a high confidence level. In this paper, we propose an automated black box testing framework based on adversarial reinforcement learning. The technique uses an adversarial agent, whose goal is to degrade the performance of the target model under test. We test the approach on an autonomous vehicle problem, by training an adversarial reinforcement learning agent, which aims to cause a deep neural network- driven autonomous vehicle to collide. Two neural networks trained for autonomous driving are compared, and the results from the testing are used to compare the robustness of their learned control policies. We show that the proposed framework is able to find weaknesses in both control policies that were not evident during online testing and therefore, demonstrate a significant benefit over manual testing methods.

- TRASS: Time Reversal As Self-Supervision

    Author: Nair, Suraj | Stanford University
    Author: Babaeizadeh, Mohammad | UIUC
    Author: Finn, Chelsea | Stanford University
    Author: Levine, Sergey | UC Berkeley
    Author: Kumar, Vikash | Google-Brain
 
    keyword: Deep Learning in Robotics and Automation

    Abstract : A longstanding challenge in robot learning for manipulation tasks has been the ability to generalize to varying initial conditions, diverse objects, and changing objectives. Learning based approaches have shown promise in producing robust policies, but require heavy supervision and large number of environment interactions, especially from visual inputs. We propose a novel self-supervision technique that uses time-reversal to provide high level supervision to reach goals. In particular, we introduce the time-reversal model (TRM), a self-supervised model which explores outward from a set of goal states and learns to predict these trajectories in reverse. This provides a high level plan towards goals, allowing us to learn complex manipulation tasks with no demonstrations or exploration at test time. We test our method on the domain of assembly, specifically the mating of tetris-style block pairs. Using our method operating atop visual model predictive control, we are able to assemble tetris blocks on a KuKa IIWA-7 using only uncalibrated RGB camera input, and generalize to unseen block pairs. Project's-page: https://sites.google.com/view/time-reversal

- Federated Imitation Learning: A Novel Framework for Cloud Robotic Systems with Heterogeneous Sensor Data

    Author: Liu, Boyi | Chinese Academy of Sciences
    Author: Wang, Lujia | Shenzhen Institutes of Advanced Technology
    Author: Liu, Ming | Hong Kong University of Science and Technology
    Author: Xu, Cheng-Zhong | University of Macau
 
    keyword: Big Data in Robotics and Automation; Deep Learning in Robotics and Automation; Motion and Path Planning

    Abstract : Humans are capable of learning a new behavior by observing others to perform the skill. Similarly, robots can also implement this by imitation learning. Furthermore, if with external guidance, humans can master the new behavior more efficiently. So, how can robots achieve this? To address the issue, we present a novel framework named FIL. It provides a heterogeneous knowledge fusion mechanism for cloud robotic systems. Then, a knowledge fusion algorithm in FIL is proposed. It enables the cloud to fuse heterogeneous knowledge from local robots and generate guide models for robots with service requests. After that, we introduce a knowledge transfer scheme to facilitate local robots acquiring knowledge from the cloud. With FIL, a robot is capable of utilizing knowledge from other robots to increase its imitation learning in accuracy and efficiency. Compared with transfer learning and meta-learning, FIL is more suitable to be deployed in cloud robotic systems. Finally, we conduct experiments of a self-driving task for robots (cars). The experimental results demonstrate that the shared model generated by FIL increases imitation learning efficiency of local robots in cloud robotic systems.


- Uncertainty Quantification with Statistical Guarantees in End-To-End Autonomous Driving Control

    Author: Michelmore, Rhiannon | University of Oxford
    Author: Wicker, Matthew | Oxford University
    Author: Laurenti, Luca | Oxford University
    Author: Cardelli, Luca | Oxford University
    Author: Gal, Yarin | University of Cambridge
    Author: Kwiatkowska, kwiatkowska_floc18_4YQkwtC9 | University of Oxford
 
    keyword: Probability and Statistical Methods; Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation

    Abstract : Deep neural network controllers for autonomous driving have recently benefited from significant performance improvements, and have begun deployment in the real world. Prior to their widespread adoption, safety guarantees are needed on the controller behaviour that properly take account of the uncertainty within the model as well as sensor noise. Bayesian neural networks, which assume a prior over the weights, have been shown capable of producing such uncertainty measures, but properties surrounding their safety have not yet been quantified for use in autonomous driving scenarios. In this paper, we develop a framework based on a state-of-the-art simulator for evaluating end-to-end Bayesian controllers. In addition to computing pointwise uncertainty measures that can be computed in real time and with statistical guarantees, we also provide a method for estimating the probability that, given a scenario, the controller keeps the car safe within a finite horizon. We experimentally evaluate the quality of uncertainty computation by several Bayesian inference methods in different scenarios and show how the uncertainty measures can be combined and calibrated for use in collision avoidance. Our results suggest that uncertainty estimates can greatly aid decision making in autonomous driving.

- Autonomously Navigating a Surgical Tool Inside the Eye by Learning from Demonstration

    Author: Kim, Ji Woong | Johns Hopkins University
    Author: He, Changyan | Beihang University
    Author: Urias, Muller | Wilmer Eye Institute
    Author: Gehlbach, Peter | Johns Hopkins Medical Institute
    Author: Hager, Gregory | Johns Hopkins University
    Author: Iordachita, Ioan Iulian | Johns Hopkins University
    Author: Kobilarov, Marin | Johns Hopkins University
 
    keyword: Deep Learning in Robotics and Automation; Surgical Robotics: Planning; Visual Servoing

    Abstract : A fundamental challenge in retinal surgery is safely navigating a surgical tool to a desired goal position on the retinal surface while avoiding damage to surrounding tissues, a procedure that typically requires tens-of-microns accuracy. In practice, the surgeon relies on depth-estimation skills to localize the tool-tip with respect to the retina in order to perform the tool-navigation task, which can be prone to human error. To alleviate such uncertainty, prior work has introduced ways to assist the surgeon by estimating the tool-tip distance to the retina and providing haptic or auditory feedback. However, automating the tool-navigation task itself remains unsolved and largely unexplored. Such a capability, if reliably automated, could serve as a building block to streamline complex procedures and reduce the chance for tissue damage. Towards this end, we propose to automate the tool-navigation task by mimicking the perception-action feedback loop of a demonstrated expert trajectory. Specifically, a deep network is trained to imitate expert trajectories toward various locations on the retina based on recorded visual servoing to a given goal specified by the user. The proposed autonomous navigation system is evaluated in simulation and in physical experiments using a silicone eye phantom. We show that the network can reliably navigate a needle surgical tool to various desired locations within 137 �m accuracy in physical experiments and 94 �m in simulation on average.

- Learn-To-Recover: Retrofitting UAVs with Reinforcement Learning-Assisted Flight Control under Cyber-Physical Attacks

    Author: Fei, Fan | Purdue University
    Author: Tu, Zhan | Purdue University
    Author: Xu, Dongyan | Purdue University
    Author: Deng, Xinyan | Purdue University
 
    keyword: Deep Learning in Robotics and Automation; Failure Detection and Recovery; AI-Based Methods

    Abstract : In this paper, we present a generic fault-tolerant control (FTC) strategy via reinforcement learning (RL). We demonstrate the effectiveness of this method on quadcopter unmanned aerial vehicles (UAVs). The fault-tolerant control policy is trained to handle actuator and sensor fault/attack. Unlike traditional FTC, this policy does not require fault detection and diagnosis (FDD) nor tailoring the controller for specific attack scenarios. Instead, the policy is running simultaneously alongside the stabilizing controller without the need for on-detection activation. The effectiveness of the policy is compared with traditional active and passive FTC strategies against actuator and sensor faults. We compare their performance in position control tasks via simulation and experiments on quadcopters. The result shows that the strategy can effectively tolerate different types of attacks/faults and maintain the vehicle's position, outperforming the other two methods.

- Model-Based Reinforcement Learning for Physical Systems without Velocity and Acceleration Measurements

    Author: Romeres, Diego | Mitsubishi Electric Research Laboratories
    Author: Dalla Libera, Alberto | University of Padova
    Author: Jha, Devesh | Mitsubishi Electric Research Laboratories
    Author: Yerazunis, William | Mitsubishi Electric Research Laboratory
    Author: Nikovski, Daniel | MERL
 
    keyword: Model Learning for Control; Dynamics

    Abstract : In this paper, we propose a derivative-free model learning framework for Reinforcement Learning (RL) algorithms based on Gaussian Process Regression (GPR). In many mechanical systems, only positions can be measured by the sensing instruments. Then, instead of representing the system state as suggested by the physics with a collection of positions, velocities, and accelerations, we define the state as the set of past position measurements. However, the equation of motions derived by physical first principles cannot be directly applied in this framework, being functions of velocities and accelerations. For this reason, we introduce a novel derivative-free physically-inspired kernel, which can be easily combined with nonparametric derivative-free Gaussian Process models. Tests performed on two real platforms show that the considered state definition combined with the proposed model improves estimation performance and data-efficiency w.r.t. traditional models based on GPR. Finally, we validate the proposed framework by solving two RL control problems for two real robotic systems.

- Towards the Probabilistic Fusion of Learned Priors into Standard Pipelines for 3D Reconstruction

    Author: Laidlow, Tristan | Imperial College London
    Author: Czarnowski, Jan | Imperial College London
    Author: Nicastro, Andrea | Imperial College London
    Author: Clark, Ronald | Imperial College London
    Author: Leutenegger, Stefan | Imperial College London
 
    keyword: Deep Learning in Robotics and Automation; Mapping

    Abstract : The best way to combine the results of deep learning with standard 3D reconstruction pipelines remains an open problem. While systems that pass the output of traditional multi-view stereo approaches to a network for regularisation currently seem to get the best results, it may be preferable to treat deep neural networks as separate components whose results can be probabilistically fused into geometry-based systems. Unfortunately, the error models required to do this are not well understood. Recently, a few systems have achieved good results by having their networks predict probability distributions rather than single values. We propose using this approach to fuse a learned single-view depth prior into a standard 3D reconstruction system.<p>Our system is capable of incrementally producing dense depth maps for a set of keyframes. We train a deep neural network to predict discrete, nonparametric probability distributions for the depth of each pixel from a single image. We then fuse this ``probability volume'' with another probability volume based on photometric consistency. We argue that combining these two sources will result in a volume that is better conditioned. To extract depth maps from the volume, we minimise a cost function that includes a regularisation term based on network predicted surface normals and boundaries. Through a series of experiments, we demonstrate that each of these components improves the overall performance of the system.

- Learning Natural Locomotion Behaviors for Humanoid Robots Using Human Bias

    Author: Yang, Chuanyu | University of Edinburgh
    Author: Yuan, Kai | University of Edinburgh
    Author: Heng, Shuai | Harbin Institute of Technology
    Author: Komura, Taku | University of Edinburgh
    Author: Li, Zhibin | University of Edinburgh
 
    keyword: Deep Learning in Robotics and Automation; Humanoid and Bipedal Locomotion; Learning from Demonstration

    Abstract : This paper presents a new learning framework that leverages the knowledge from imitation learning, deep reinforcement learning, and control theories to achieve human-style locomotion that is natural, dynamic, and robust for humanoids. We proposed novel approaches to introduce human bias, ie motion capture data and a special Multi-Expert network structure. We used the Multi-Expert network structure to smoothly blend behavioral features, and used the augmented reward design for the task and imitation rewards. Our reward design is more composable, tunable, and explainable by using fundamental concepts from conventional humanoid control. We rigorously validated and benchmarked the learning framework which consistently produced robust locomotion behaviors in various test scenarios. Further, we demonstrated the capability of learning robust and versatile policies in the presence of disturbances, such as terrain irregularities and external pushes.

- Aggressive Online Control of a Quadrotor Via Deep Network Representations of Optimality Principles

    Author: Li, Shuo | TU Delft
    Author: Ozturk, Ekin | European Space Agency
    Author: De Wagter, Christophe | Delft University of Technology
    Author: de Croon, Guido | TU Delft / ESA
    Author: Izzo, Dario | European Space Agency
 
    keyword: Deep Learning in Robotics and Automation; Optimization and Optimal Control; Aerial Systems: Mechanics and Control

    Abstract : Optimal control holds great potential to improve a variety of robotic applications. The application of optimal control on-board limited platforms has been severely hindered by the large computational requirements of current state of the art implementations. In this work, we make use of a deep neural network to directly map the robot states to control actions. The network is trained offline to imitate the optimal control computed by a time consuming direct nonlinear method. A mixture of time optimality and power optimality is considered with a continuation parameter used to select the predominance of each objective. We apply our networks (termed G&amp;CNets) to aggressive quadrotor control, first in simulation and then in the real world. We give insight into the factors that influence the �reality gap� between the quadrotor model used by the offline optimal control method and the real quadrotor. Furthermore, we explain how we set up the model and the control structure on-board of the real quadrotor to successfully close this gap and perform time-optimal maneuvers in the real world. Finally, G&amp;CNet's performance is compared to state-of- the-art differential-flatness-based optimal control methods. We show, in the experiments, that G&amp;CNets lead to significantly faster trajectory execution due to, in part, the less restrictive nature of the allowed state-to-input mappings.

- Visual Object Search by Learning Spatial Context

    Author: Druon, Raphael | Paul Sabatier University
    Author: Yoshiyasu, Yusuke | CNRS-AIST JRL
    Author: Kanezaki, Asako | National Institute of Advanced Industrial Science and Technology
    Author: Watt, Alassane Mamadou | CentraleSupelec
 
    keyword: Deep Learning in Robotics and Automation; Visual-Based Navigation; Autonomous Agents

    Abstract : We present a visual navigation approach that uses context information to navigate an agent to find and reach a target object. To learn context from the objects present in the scene, we transform visual information into an intermediate representation called context grid which essentially represents how much the object at the location is semantically similar to the target object. As this representation can encode the target object and other objects together, it allows us to navigate an agent in a human-inspired way: the agent will go to the likely place by seeing surrounding context objects in the beginning when the target is not visible and, once the target object comes into sight, it will reach the target quickly. Since context grid does not directly contain visual or semantic feature values that change according to introductions of new objects, such as new instances of the same object with different appearance or an object from a slightly different class, our navigation model generalizes well to unseen scenes/objects. Experimental results show that our approach outperforms previous approaches in navigating in unseen scenes, especially for broad scenes. We also evaluated human performances in the target-driven navigation task and compared with machine learning based navigation approaches including this work.

- Salient View Selection for Visual Recognition of Industrial Components

    Author: Kim, Seong-heum | Korea Electronics Technology Institute (KETI)
    Author: Choe, Gyeongmin | KAIST
    Author: Park, Min-Gyu | KETI
    Author: Kweon, In So | KAIST
 
    keyword: Deep Learning in Robotics and Automation; Big Data in Robotics and Automation; Computer Vision for Manufacturing

    Abstract : We introduce a new method to find a salient viewpoint with a deep representation, according to ease of semantic segmentation. The main idea in our segmentation network is to utilize the multipath network with informative two views. In order to collect training samples, we assume all the information of designed components and even error tolerances are available. Before installing the actual camera layout, we simulate different model descriptions in a physically correct way and determine the best viewing parameters to retrieve a correct instance model from an established database. By selecting the salient viewpoint, we better understand fine-grained shape variations with specular materials. From the fixed top-view, our system initially predicts a 3-DoF pose of a testing object in a data-driven way, and precisely align the model with a refined semantic mask. Under various conditions of our system setup, the presented method is experimentally validated. A robotic assembly task with our vision solution is also successfully demonstrated.

- Low to High Dimensional Modality Reconstruction Using Aggregated Fields of View

    Author: Gunasekar, Kausic | Arizona State University
    Author: Qiu, Qiang | Duke Univeristy
    Author: Yang, Yezhou | Arizona State University
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Other Robotic Applications; Robot Safety

    Abstract : Real world robotics systems today deal with data from a multitude of modalities, especially for tasks such as navigation and recognition. The performance of those systems can drastically degrade when one or more modalities become inaccessible, due to factors such as sensors' malfunctions or adverse environments. Here, we argue modality hallucination as one effective way to ensure consistent modality availability and thereby reduce unfavorable consequences. While hallucinating data from a modality with richer information, e.g., RGB to depth, has been researched extensively, we investigate the more challenging low-to-high modality hallucination with interesting use cases in robotics and autonomous systems. We present a novel hallucination architecture that aggregates information from multiple fields of view of the local neighborhood to recover the lost information from the extant modality. The process is implemented by capturing a non-linear mapping between the data modalities and the learned mapping is used to aid the extant modality to mitigate the risk posed to the system in the adverse scenarios which involve modality loss. We also conduct extensive classification and segmentation experiments on UW-RGBD and NYUD datasets and demonstrate that hallucination indeed allays the negative effects of the modality loss.

- Learning Fast Adaptation with Meta Strategy Optimization

    Author: Yu, Wenhao | Georgia Institute of Technology
    Author: Tan, Jie | Google
    Author: Bai, Yunfei | Google X
    Author: Coumans, Erwin | Google Inc
    Author: Ha, Sehoon | Google Brain
 
    keyword: Deep Learning in Robotics and Automation; Learning and Adaptive Systems; Legged Robots

    Abstract : The ability to walk in new scenarios is a key milestone on the path toward real-world applications of legged robots. In this work, we introduce Meta Strategy Optimization, a meta-learning algorithm for training policies with latent variable inputs that can quickly adapt to new scenarios with a handful of trials in the target environment. The key idea behind MSO is to expose the same adaptation process, Strategy Optimization (SO), to both the training and testing phases. This allows MSO to effectively learn locomotion skills as well as a latent space that is suitable for fast adaptation. We evaluate our method on a real quadruped robot and demonstrate successful adaptation in various scenarios, including sim-to-real transfer, walking with a weakened motor, or climbing up a slope. Furthermore, we quantitatively analyze the generalization capability of the trained policy in simulated environments. Both real and simulated experiments show that our method outperforms previous methods in adaptation to novel tasks.

- Deep Neural Network Approach in Robot Tool Dynamics Identification for Bilateral Teleoperation

    Author: Su, Hang | Politecnico Di Milano
    Author: Qi, Wen | Politecnico Di Milano
    Author: Yang, Chenguang | University of the West of England
    Author: Sandoval, Juan Sebasti�n | Université D'Orl�ans
    Author: Ferrigno, Giancarlo | Politecnico Di Milano
    Author: De Momi, Elena | Politecnico Di Milano
 
    keyword: Deep Learning in Robotics and Automation; Force and Tactile Sensing; Telerobotics and Teleoperation

    Abstract : For bilateral teleoperation, the haptic feedback demands the availability of accurate force information transmitted from the remote site. Nevertheless, due to the limitation of the size, the force sensor is usually attached outside of the patient's abdominal cavity for the surgical operation. Hence, it measures not only the interaction forces on the surgical tip but also the surgical tool dynamics. In this paper, a model-free based deep convolutional neural network (DCNN) structure is proposed for the tool dynamics identification, which features fast computation and noise robustness. After the tool dynamics identification using DCNN, the calibration is performed, and the bilateral teleoperation is demonstrated to verify the proposed method. The comparison results prove that the proposed DCNN model promises prominent performance than other methods. Low computational time (0.0031 seconds) is ensured by the rectified linear unit (ReLU) function, and the DCNN approach provides superior accuracy for predicting the noised dynamics force and enable its feasibility for bilateral teleoperation.
- Learning Matchable Image Transformations for Long-Term Metric Visual Localization

    Author: Clement, Lee | University of Toronto
    Author: Gridseth, Mona | University of Toronto
    Author: Tomasi, Justin | University of Toronto
    Author: Kelly, Jonathan | University of Toronto
 
    keyword: Deep Learning in Robotics and Automation; Visual Learning; Visual-Based Navigation

    Abstract : Long-term metric self-localization is an essential capability of autonomous mobile robots, but remains challenging for vision-based systems due to appearance changes caused by lighting, weather, or seasonal variations. While experience-based mapping has proven to be an effective technique for bridging the 'appearance gap,' the number of experiences required for reliable metric localization over days or months can be very large, and methods for reducing the necessary number of experiences are needed for this approach to scale. Taking inspiration from color constancy theory, we learn a nonlinear RGB-to-grayscale mapping that explicitly maximizes the number of inlier feature matches for images captured under different lighting and weather conditions, and use it as a pre-processing step in a conventional single-experience localization pipeline to improve its robustness to appearance change. We train this mapping by approximating the target non-differentiable localization pipeline with a deep neural network, and find that incorporating a learned low-dimensional context feature can further improve cross-appearance feature matching. Using synthetic and real-world datasets, we demonstrate substantial improvements in localization performance across day-night cycles, enabling continuous metric localization over a 30-hour period using a single mapping experience, and allowing experience-based localization to scale to long deployments with dramatically reduced data requirements.

- OriNet: Robust 3-D Orientation Estimation with a Single Particular IMU

    Author: Abolfazli Esfahani, Mahdi | Nanyang Technologicl University
    Author: Wang, Han | Nanyang Technological University
    Author: Wu, Keyu | Nanyang Technological University
    Author: Yuan, Shenghai | Nanyang Technological University
 
    keyword: Deep Learning in Robotics and Automation; Intelligent Transportation Systems; AI-Based Methods

    Abstract : Estimating the robot's heading is a crucial requirement in odometry systems which are attempting to estimate the movement trajectory of a robot. Small errors in the orientation estimation result in a significant difference between the estimated and real trajectory, and failure of the odometry system. The odometry problem becomes much more complicated for micro flying robots since they cannot carry massive sensors. In this manner, they should benefit from the small size and low-cost sensors, such as IMU, to solve the odometry problem, and industries always look for such solutions. However, IMU suffers from bias and measurement noise, which makes the problem of position and orientation estimation challenging to be solved by a single IMU. While there are numerous studies on the fusion of IMU with other sensors, this study illustrates the power of the first deep learning framework for estimating the full 3D orientation of the flying robots (as yaw, pitch, and roll in quaternion coordinates) accurately with the presence of a single IMU. A particular IMU should be utilized during the training and testing of the proposed system. Besides, a method based on the Genetic Algorithm is introduced to measure the IMU bias in each execution. The results show that the proposed method improved the flying robots' ability to estimate their orientation displacement by approximately 80% with the presence of a single particular IMU.

- Learning Densities in Feature Space for Reliable Segmentation of Indoor Scenes

    Author: Marchal, Nicolas Paul | ETH Zurich
    Author: Moraldo, Charlotte | ETH Zurich
    Author: Blum, Hermann | ETH Zurich
    Author: Siegwart, Roland | ETH Zurich
    Author: Cadena Lerma, Cesar | ETH Zurich
    Author: Gawel, Abel Roman | ETH Zurich
 
    keyword: Deep Learning in Robotics and Automation; Semantic Scene Understanding; Visual Learning

    Abstract : Deep learning has enabled remarkable advances in scene understanding, particularly in semantic segmentation tasks. Yet, current state of the art approaches are limited to a closed set of classes, and fail when facing novel elements, also known as out of distribution (OoD) data. This is a problem as autonomous agents will inevitably come across a wide range of objects, all of which cannot be included during training. We propose a novel method to distinguish any object (foreground) from empty building structure (background) in indoor environments. We use normalizing flow to estimate the probability distribution of high-dimensional background descriptors. Foreground objects are therefore detected as areas in an image for which the descriptors are unlikely given the background distribution. As our method does not explicitly learn the representation of individual objects, its performance generalizes well outside of the training examples. Our model results in an innovative solution to reliably segment foreground from background in indoor scenes, which opens the way to a safer deployment of robots in human environments.

- A Multimodal Target-Source Classifier with Attention Branches to Understand Ambiguous Instructions for Fetching Daily Objects

    Author: Magassouba, Aly | NICT
    Author: Sugiura, Komei | National Institute of Information and Communications Tech
    Author: Kawai, Hisashi | National Institute of Information and Communications Technology
 
    keyword: Deep Learning in Robotics and Automation; Domestic Robots

    Abstract : In this study,	we focus on multimodal language understanding for fetching instructions in domestic service robots context. This task consists of predicting a target object as instructed by the user given an image and an unstructured sentence, such as ``Bring me the yellow box (from the wooden cabinet) .'' This is challenging because of the ambiguity of natural language, i.e., the relevant information may be missing or there might be several candidates. To solve such a task, we propose the multimodal target-source classifier model with attention branch (MTCM-AB), which is an extension of the MTCM. Our methodology uses the attention branch network (ABN) to develop a multimodal attention mechanism based on linguistic and visual inputs. Experimental validation using a standard dataset showed that the MTCM-AB outperformed both state-of-the-art methods and MTCM. In particular the MTCM-AB accuracy on average was 90.1% while human performance was 90.3% on PFN-PIC dataset.

- On-Board Deep-Learning-Based Unmanned Aerial Vehicle Fault Cause Detection and Identification

    Author: Sadhu, Vidyasagar | Rutgers University
    Author: Zonouz, Saman | Rutgers University
    Author: Pompili, Dario | Rutgers University
 
    keyword: Deep Learning in Robotics and Automation

    Abstract : With the increase in use of Unmanned Aerial Vehicles (UAVs)/drones, it is important to detect and identify causes of failure in real time for proper recovery from a potential crash-like scenario or post incident forensics analysis. The cause of crash could be either a fault in the sensor/actuator system, a physical damage/attack, or a cyber attack on the drone's software. In this paper, we propose novel architectures based on deep Convolutional and Long Short-Term Memory Neural Networks (CNNs and LSTMs) to detect (via Autoencoder) and classify drone mis-operations based on real-time sensor data. The proposed architectures are able to learn high-level features automatically from the raw sensor data and learn the spatial and temporal dynamics in the sensor data. We validate the proposed deep-learning architectures via simulations and real-world experiments on a drone. Empirical results show that our solution is able to detect (with over 90% accuracy) and classify various types of drone mis-operations (with about 99% accuracy (simulation data) and upto 85% accuracy (experimental data)).

- Learning One-Shot Imitation from Humans without Humans

    Author: Bonardi, Alessandro | Imperial College London
    Author: James, Stephen | Imperial College London
    Author: Davison, Andrew J | Imperial College London
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; Learning from Demonstration

    Abstract : Humans can naturally learn to execute a new task by seeing it performed by other individuals once, and then reproduce it in a variety of configurations. Endowing robots with this ability of imitating humans from third person is a very immediate and natural way of teaching new tasks. Only recently, through meta-learning, there have been successful attempts to one-shot imitation learning from humans; however, these approaches require a lot of human resources to collect the data in the real world to train the robot. But is there a way to remove the need for real world human demonstrations during training? We show that with Task-Embedded Control Networks, we can infer control polices by embedding human demonstrations that can condition a control policy and achieve one-shot imitation learning. Importantly, we do not use a real human arm to supply demonstrations during training, but instead leverage domain randomisation in an application that has not been seen before: sim-to-real transfer on humans. Upon evaluating our approach on pushing and placing tasks in both simulation and in the real world, we show that in comparison to a system that was trained on real-world data we are able to achieve similar results by utilising only simulation data.


- Adversarial Skill Networks: Unsupervised Robot Skill Learning from Video

    Author: Mees, Oier | Albert-Ludwigs-Universitét
    Author: Merklinger, Markus | Uni Freiburg
    Author: Kalweit, Gabriel | University of Freiburg
    Author: Burgard, Wolfram | Toyota Research Institute
 
    keyword: Deep Learning in Robotics and Automation; Visual Learning; Computer Vision for Automation

    Abstract : Key challenges for the deployment of reinforcement learning (RL) agents in the real world are the discovery, representation and reuse of skills in the absence of a reward function. To this end, we propose a novel approach to learn a task-agnostic skill embedding space from unlabeled multi-view videos. Our method learns a general skill embedding independently from the task context by using an adversarial loss. We combine a metric learning loss, which utilizes temporal video coherence to learn a state representation, with an entropy regularized adversarial skill-transfer loss. The metric learning loss learns a disentangled representation by attracting simultaneous viewpoints of the same observations and repelling visually similar frames from temporal neighbors. The adversarial skill-transfer loss enhances re-usability of learned skill embeddings over multiple task domains. We show that the learned embedding enables training of continuous control policies to solve novel tasks that require the interpolation of previously seen skills. Our extensive evaluation with both simulation and real world data demonstrates the effectiveness of our method in learning transferable skills from unlabeled interaction videos and composing them for new tasks. Code, pretrained models and dataset are available at http://robotskills.cs. uni-freiburg.de

- Event-Based Angular Velocity Regression with Spiking Networks

    Author: Gehrig, Mathias | University of Zurich
    Author: Shrestha, Sumit Bam | Temasek Laboratories @ National University of Singapore
    Author: Mouritzen, Daniel S. | University of Zurich and ETH Zurich
    Author: Scaramuzza, Davide | University of Zurich
 
    keyword: Deep Learning in Robotics and Automation; Visual Learning; Probability and Statistical Methods

    Abstract : Spiking Neural Networks (SNNs) are bio-inspired networks that process information conveyed as temporal spikes rather than numeric values. Due to their spike-based computational model, SNNs can process output from event-based, asynchronous sensors without any pre-processing at extremely lower power unlike standard artificial neural networks. Yet, SNNs have not enjoyed the same rise of popularity as artificial neural networks. This not only stems from the fact that their input format is rather unconventional but also due to the challenges in training spiking networks. Despite their temporal nature and recent algorithmic advances, they have been mostly evaluated on classification problems. We propose, for the first time, a temporal regression problem of numerical values given events from an event-camera. We specifically investigate the prediction of the 3-DOF angular velocity of a rotating event-camera with an SNN. The difficulty of this problem arises from the prediction of angular velocities continuously in time directly from irregular, asynchronous event-based input. To assess the performance of SNNs on this task, we introduce a large-scale synthetic dataset generated from real-world panoramic images and show that we can successfully train an SNN to perform angular velocity regression.

- Visual Odometry Revisited: What Should Be Learnt?

    Author: Zhan, Huangying | The University of Adelaide
    Author: Weerasekera, Chamara Saroj | The University of Adelaide
    Author: Bian, Jiawang | University of Adelaide
    Author: Reid, Ian | University of Adelaide
 
    keyword: Deep Learning in Robotics and Automation; Visual-Based Navigation; Localization

    Abstract : In this work we present a monocular visual odometry (VO) algorithm which leverages geometry-based methods and deep learning. Most existing VO/SLAM systems with superior performance are based on geometry and have to be carefully designed for different application scenarios. Moreover, most monocular systems suffer from scale-drift issue. Some recent deep learning works learn VO in an end-to-end manner but the performance of these deep systems is still not comparable to geometry-based methods. In this work, we revisit the basics of VO and explore the right way for integrating deep learning with epipolar geometry and Perspective-n-Point (PnP) method. Specifically, we train two convolutional neural networks (CNNs) for estimating single-view depths and two-view optical flows as intermediate outputs. With the deep predictions, we design a simple but robust frame-to-frame VO algorithm (DF-VO) which outperforms pure deep learning-based and geometry-based methods. More importantly, our system does not suffer from the scale-drift issue being aided by a scale consistent single-view depth CNN. Extensive experiments on KITTI dataset shows the robustness of our system and a detailed ablation study shows the effect of different factors in our system. Code is available at here: DF-VO.

- 3D Scene Geometry-Aware Constraint for Camera Localization with Deep Learning

    Author: Tian, Mi | Meituan-Dianping Group
    Author: Nie, Qiong | Meituan-Dianping Group
    Author: Shen, Hao | Meituan-Dianping Group
 
    keyword: Deep Learning in Robotics and Automation; Localization

    Abstract : Camera localization is a fundamental and key component of autonomous driving vehicles and mobile robots to localize themselves globally for further environment perception, path planning and motion control. Recently end-to-end approaches based on convolutional neural network have been much studied to achieve or even exceed 3D-geometry based traditional methods. In this work, we propose a compact network for absolute camera pose regression. Inspired from those traditional methods, a 3D scene geometry-aware constraint is also introduced by exploiting all available information including motion, depth and image contents. We add this constraint as a regularization term to our proposed network by defining a pixel-level photometric loss and an image-level structural similarity loss. To benchmark our method, different challenging scenes including indoor and outdoor environment are tested with our proposed approach and state-of-the-arts. And the experimental results demonstrate significant performance improvement of our method on both prediction accuracy and convergence efficiency.

- ACDER: Augmented Curiosity-Driven Experience Replay

    Author: Li, Boyao | Institute of Automation, Chinese Academy of Sciences
    Author: Lu, Tao | The Hi-Tech Innovation Engineering Center
    Author: Li, Jiayi | Institute of Automation, Chinese Academy of Sciences
    Author: Lu, Ning | Institute of Automation, Chinese Academy of Sciences
    Author: Cai, Yinghao | Institute of Automation, Chinese Academy of Sciences
    Author: Wang, Shuo | Chinese Academy of Sciences
 
    keyword: Deep Learning in Robotics and Automation; AI-Based Methods

    Abstract : Exploration in environments with sparse feedback remains a challenging research problem in reinforcement learning (RL). When the RL agent explores the environment randomly, it results in low exploration efficiency, especially in robotic manipulation tasks with high dimensional continuous state and action space. In this paper, we propose a novel method, called Augmented Curiosity-Driven Experience Replay (ACDER), which leverages (i) a new goal-oriented curiosity-driven exploration to encourage the agent to pursue novel and task-relevant states more purposefully and (ii) the dynamic initial states selection as an automatic exploratory curriculum to further improve the sample-efficiency. Our approach complements Hindsight Experience Replay (HER) by introducing a new way to pursue valuable states. Experiments conducted on four challenging robotic manipulation tasks with binary rewards, including Reach, Push, Pick&amp;Place and Multi-step Push. The empirical results show that our proposed method significantly outperforms existing methods in the first three basic tasks and also achieves satisfactory performance in multi-step robotic task learning.

- TrueRMA: Learning Fast and Smooth Robot Trajectories with Recursive Midpoint Adaptations in Cartesian Space

    Author: Kiemel, Jonas | Karlsruhe Institute of Technology
    Author: Mei�ner, Pascal | Karlsruhe Institute of Technology
    Author: Kroeger, Torsten | Karlsruher Institut F�r Technologie (KIT)
 
    keyword: Deep Learning in Robotics and Automation; Big Data in Robotics and Automation

    Abstract : We present TrueRMA, a data-efficient, model-free method to learn cost-optimized robot trajectories over a wide range of starting points and endpoints. The key idea is to calculate trajectory waypoints in Cartesian space by recursively predicting orthogonal adaptations relative to the midpoints of straight lines. We generate a differentiable path by adding circular blends around the waypoints, calculate the corresponding joint positions with an inverse kinematics solver and calculate a time-optimal parameterization considering velocity and acceleration limits. During training, the trajectory is executed in a physics simulator and costs are assigned according to a user-specified cost function which is not required to be differentiable. Given a starting point and an endpoint as input, a neural network is trained to predict midpoint adaptations that minimize the cost of the resulting trajectory via reinforcement learning. We successfully train a KUKA iiwa robot to keep a ball on a plate while moving between specified points and compare the performance of TrueRMA against two baselines. The results show that our method requires less training data to learn the task while generating shorter and faster trajectories.

## Motion and Path Planning

- Hyperproperties for Robotics: Planning Via HyperLTL

    Author: Wang, Yu | Duke University
    Author: Siddhartha, Nalluri | Duke University
    Author: Pajic, Miroslav | Duke University
 
    keyword: Formal Methods in Robotics and Automation; Motion and Path Planning; Robot Safety

    Abstract : There is a growing interest on formal methods-based robotic planning for temporal logic objectives. In this work, we extend the scope of existing synthesis methods to hyper-temporal logics. We are motivated by the fact that important planning objectives, such as optimality, robustness, and privacy, (maybe implicitly) involve the interrelation between multiple paths. Such objectives are thus hyperproperties, and cannot be expressed with usual temporal logics like the linear temporal logic (LTL). We show that such hyperproperties can be expressed by HyperLTL, an extension of LTL to multiple paths. To handle the complexity of planning with HyperLTL specifications, we introduce a symbolic approach for synthesizing planning strategies on discrete transition systems. Our planning method is evaluated on several case studies.

-     Abstractions for Computing All Robotic Sensors That Suffice to Solve a Planning Problem

    Author: Zhang, Yulin | Texas A&amp;M University
    Author: Shell, Dylan | Texas A&amp;M University
 
    keyword: Formal Methods in Robotics and Automation; Reactive and Sensor-Based Planning

    Abstract : Whether a robot can perform some specific task depends on several aspects, including the robot's sensors and the plans it possesses. We are interested in search algorithms that treat plans and sensor designs jointly, yielding solutions---i.e., plan and sensor characterization pairs---if and only if they exist. Such algorithms can help roboticists explore the space of sensors to aid in making design trade-offs. Generalizing prior work where sensors are modeled     Abstractly as sensor maps on p-graphs, the present paper increases the potential sensors which can be sought significantly. But doing so enlarges a problem currently on the outer limits of being considered tractable. Toward taming this complexity, two contributions are made: (1) we show how to represent the search space for this more general problem and describe data structures that enable whole sets of sensors to be summarized via a single special representative; (2) we give a means by which other structure (either task domain knowledge, sensor technology or fabrication constraints) can be incorporated to reduce the sets to be enumerated. These lead to algorithms that we have implemented and which suffice to solve particular problem instances, albeit only of small scale. Nevertheless, the algorithm aids in helping understand what attributes sensors must possess and what information they must provide in order to ensure a robot can achieve its goals despite non-determinism.	

- T* : A Heuristic Search Based Path Planning Algorithm for Temporal Logic Specifications

    Author: Khalidi, Danish | IIT Kanpur
    Author: Gujarathi, Dhaval | SAP Labs, India
    Author: Saha, Indranil | IIT Kanpur
 
    keyword: Formal Methods in Robotics and Automation; Motion and Path Planning; Task Planning

    Abstract : The fundamental path planning problem for a mobile robot involves generating a trajectory for point-to-point navigation while avoiding obstacles. Heuristic-based search algorithms like A* have been shown to be efficient in solving such planning problems. Recently, there has been an increased interest in specifying complex path planning problem using temporal logic. In the state-of-the-art algorithm, the temporal logic path planning problem is reduced to a graph search problem, and Dijkstra's shortest path algorithm is used to compute the optimal trajectory satisfying the specification.<p>The A* algorithm, when used with an appropriate heuristic for the distance from the destination, can generate an optimal path in a graph more efficiently than Dijkstra's shortest path algorithm. The primary challenge for using A* algorithm in temporal logic path planning is that there is no notion of a single destination state for the robot. We present a novel path planning algorithm T* that uses the A* search procedure opportunistically to generate an optimal trajectory satisfying a temporal logic query. Our experimental results demonstrate that T* achieves an order of magnitude improvement over the state-of-the-art algorithm to solve many temporal logic path planning problems in 2-D as well as 3-D workspaces.

- Global/local Motion Planning Based on Dynamic Trajectory Reconfiguration and Dynamical Systems for Autonomous Surgical Robots

    Author: Sayols, Narcis | Universitat Politecnica De Catalunya
    Author: Sozzi, Alessio | University of Ferrara
    Author: Piccinelli, Nicola | University of Verona
    Author: Hernansanz, Albert | UPC (Universitat Politecnicade Catalunya)
    Author: Casals, Alicia | UniversitatPolitècnica De Catalunya, Barcelona Tech
    Author: Bonfe, Marcello | University of Ferrara
    Author: Muradore, Riccardo | University of Verona
 
    keyword: Collision Avoidance; Motion and Path Planning; Surgical Robotics: Laparoscopy

    Abstract : This paper addresses the generation of collision-free trajectories for the autonomous execution of assistive tasks in Robotic Minimally Invasive Surgery (R-MIS). The proposed approach takes into account geometric constraints related to the desired task, like for example the direction to approach the final target and the presence of moving obstacles. The developed motion planner is structured as a two-layer architecture: a global level computes smooth spline-based trajectories that are continuously updated using virtual potential fields; a local level, exploiting Dynamical Systems based obstacle avoidance, ensures collision free connections among the spline control points. The proposed architecture is validated in a realistic surgical scenario.

- Deep Imitative Reinforcement Learning for Temporal Logic Robot Motion Planning with Noisy Semantic Observations

    Author: Gao, Qitong | Duke University
    Author: Pajic, Miroslav | Duke University
    Author: Zavlanos, Michael M. | Duke University
 
    keyword: Formal Methods in Robotics and Automation; Deep Learning in Robotics and Automation

    Abstract : In this paper, we propose a Deep Imitative Q-learning (DIQL) method to synthesize control policies for mobile robots that need to satisfy Linear Temporal Logic (LTL) specifications using noisy semantic observations of their surroundings. The robot sensing error is modeled using probabilistic labels defined over the states of a Labeled Transition System (LTS) and the robot mobility is modeled using a Labeled Markov Decision Process (LMDP) with unknown transition probabilities. We use existing product-based model checkers (PMCs) as experts to guide the Q-learning algorithm to convergence. To the best of our knowledge, this is the first approach that models noise in semantic observations using probabilistic labeling functions and employs existing model checkers to provide suboptimal instructions to the Q-learning agent.

- Minimal 3D Dubins Path with Bounded Curvature and Pitch Angle

    Author: V�&#328;a, Petr | Czech Technical University in Prague
    Author: Alves Neto, Armando | Universidade Federal De Minas Gerais
    Author: Faigl, Jan | Czech Technical University in Prague
    Author: Guimar�es Macharet, Douglas | Universidade Federal De Minas Gerais
 
    keyword: Motion and Path Planning; Nonholonomic Motion Planning; Aerial Systems: Applications

    Abstract : In this paper, we address the problem of finding cost-efficient three-dimensional paths that satisfy the maximum allowed curvature and the pitch angle of the vehicle. For any given initial and final configurations, the problem is decoupled into finding the horizontal and vertical parts of the path separately. Although the individual paths are modeled as two-dimensional Dubins curves using closed-form solutions, the final 3D path is constructed using the proposed local optimization to find a cost-efficient solution. Moreover, based on the decoupled approach, we provide a lower bound estimation of the optimal path that enables us to determine the quality of the found heuristic solution. The proposed solution has been evaluated using existing benchmark instances and compared with state-of-the-art approaches. Based on the reported results and lower bounds, the proposed approach provides paths close to the optimal solution while the computational requirements are in hundreds of microseconds. Besides, the proposed method provides paths with fewer turns than others, which make them easier to be followed by the vehicle's controller.

- Adaptively Informed Trees (AIT*): Fast Asymptotically Optimal Path Planning through Adaptive Heuristics

    Author: Strub, Marlin Polo | University of Oxford
    Author: Gammell, Jonathan | University of Oxford
 
    keyword: Motion and Path Planning; Autonomous Vehicle Navigation; Space Robotics and Automation

    Abstract : Informed sampling-based planning algorithms exploit problem knowledge for better search performance. This knowledge is often expressed as heuristic estimates of solution cost and used to order the search. The practical improvement of this informed search depends on the accuracy of the heuristic.<p>Selecting an appropriate heuristic is difficult. Heuristics applicable to an entire problem domain are often simple to define and inexpensive to evaluate but may not be beneficial for a specific problem instance. Heuristics specific to a problem instance are often difficult to define or expensive to evaluate but can make the search itself trivial.</p><p>This paper presents Adaptively Informed Trees (AIT*), an almost-surely asymptotically optimal sampling-based planner based on BIT*. AIT* adapts its search to each problem instance by using an asymmetric bidirectional search to simultaneously estimate and exploit a problem-specific heuristic. This allows it to quickly find initial solutions and converge towards the optimum. AIT* solves the tested problems as fast as RRT-Connect while also converging towards the optimum.

- Informing Multi-Modal Planning with Synergistic Discrete Leads

    Author: Kingston, Zachary | Rice University
    Author: Wells, Andrew | Rice University
    Author: Moll, Mark | Rice University
    Author: Kavraki, Lydia | Rice University
 
    keyword: Motion and Path Planning; Manipulation Planning

    Abstract : Robotic manipulation problems are inherently continuous, but typically have underlying discrete structure, e.g., whether or not an object is grasped. This means many problems are multi-modal and in particular have a continuous infinity of modes. For example, in a pick-and-place manipulation domain, every grasp and placement of an object is a mode. Usually manipulation problems require the robot to transition into different modes, e.g., going from a mode with an object placed to another mode with the object grasped. To successfully find a manipulation plan, a planner must find a sequence of valid single-mode motions as well as valid transitions between these modes. Many manipulation planners have been proposed to solve tasks with multi-modal structure. However, these methods require mode-specific planners and fail to scale to very cluttered environments or to tasks that require long sequences of transitions. This paper presents a general layered planning approach to multi-modal planning that uses a discrete "lead" to bias search towards useful mode transitions. The difficulty of achieving specific mode transitions is captured online and used to bias search towards more promising sequences of modes. We demonstrate our planner on complex scenes and show that significant performance improvements are tied to both our discrete "lead" and our continuous representation.

- Hierarchical Coverage Path Planning in Complex 3D Environments

    Author: Cao, Chao | Carnegie Mellon University
    Author: Zhang, Ji | Carnegie Mellon University
    Author: Travers, Matthew | Carnegie Mellon University
    Author: Choset, Howie | Carnegie Mellon University
 
    keyword: Motion and Path Planning; Aerial Systems: Applications

    Abstract : State-of-the-art coverage planning methods perform well in simple environments but take an ineffectively long time to converge to an optimal solution in complex three-dimensional (3D) environments. As more structures are present in the same volume of workspace, these methods slow down as they spend more time searching for all of the nooks and crannies concealed in three-dimensional spaces. This work presents a method for coverage planning that employs a multi-resolution hierarchical framework to solve the problem at two different levels, producing much higher efficiency than the state-of-the-art. First, a high-level algorithm separates the environment into multiple subspaces at different resolutions and computes an order of the subspaces for traversal. Second, a low-level sampling-based algorithm solves for paths within the subspaces for detailed coverage. In experiments, we evaluate our method using real-world datasets from complex three-dimensional scenes. Our method finds paths that are constantly shorter and converges at least ten times faster than the state-of-the-art. Further, we show results of a physical experiment where a lightweight UAV follows the paths to realize the coverage.

- Perception-Aware Time Optimal Path Parameterization for Quadrotors

    Author: Spasojevic, Igor | MIT
    Author: Murali, Varun | Massachusetts Institute of Technology
    Author: Karaman, Sertac | Massachusetts Institute of Technology
 
    keyword: Motion and Path Planning; Aerial Systems: Perception and Autonomy; Visual-Based Navigation

    Abstract : The increasing popularity of quadrotors has given rise to a class of predominantly vision-driven vehicles. This paper addresses the problem of perception-aware time optimal path parametrization for quadrotors. Although many different choices of perceptual modalities are available, the low weight and power budgets of quadrotor systems makes a camera ideal for on-board navigation and estimation algorithms. However, this does come with a set of challenges. The limited field of view of the camera can restrict the visibility of salient regions in the environment, which dictates the necessity to consider perception and planning jointly. The main contribution of this paper is an efficient time optimal path parametrization algorithm for quadrotors with limited field of view constraints. We show in a simulation study that a state-of-the-art controller can track planned trajectories, and we validate the proposed algorithm on a quadrotor platform in experiments.

- Generating Visibility-Aware Trajectories for Cooperative and Proactive Motion Planning

    Author: Buckman, Noam | Massachusetts Institute of Technology
    Author: Pierson, Alyssa | Massachusetts Institute of Technology
    Author: Karaman, Sertac | Massachusetts Institute of Technology
    Author: Rus, Daniela | MIT
 
    keyword: Motion and Path Planning; Autonomous Vehicle Navigation; Cooperating Robots

    Abstract : The safety of an autonomous vehicle not only depends on its own perception of the world around it, but also on the perception and recognition from other vehicles. If an ego vehicle considers the uncertainty other vehicles have about itself, then by reducing the estimated uncertainty it can increase its safety. In this paper, we focus on how an ego vehicle plans its trajectories through the blind spots of other vehicles. We create visibility-aware planning, where the ego vehicle chooses its trajectories such that it reduces the perceived uncertainty other vehicles may have about the state of the ego vehicle. We present simulations of traffic and highway environments, where an ego vehicle must pass another vehicle, make a lane change, or traverse a partially-occluded intersection. Emergent behavior shows that when using visibility-aware planning, the ego vehicle spends less time in a blind spot, and may slow down before entering the blind spot so as to increase the likelihood other vehicles perceive the ego vehicle.

- An Obstacle-Interaction Planning Method for Navigation of Actuated Vine Robots

    Author: Selvaggio, Mario | Université Degli Studi Di Napoli Federico II
    Author: Ramirez, Luis Adrian | University of California, Irvine
    Author: Naclerio, Nicholas | University of California, Santa Barbara
    Author: Siciliano, Bruno | Univ. Napoli Federico II
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
 
    keyword: Motion and Path Planning; Modeling, Control, and Learning for Soft Robots

    Abstract : The field of soft robotics is grounded on the idea that, due to their inherent compliance, soft robots can safely interact with the environment. Thus, the development of effective planning and control pipelines for soft robots should incorporate reliable robot-environment interaction models. This strategy enables soft robots to effectively exploit contacts to autonomously navigate and accomplish tasks in the environment. However, for a class of soft robots, namely vine-inspired, tip-extending or "vine" robots, such interaction models and the resulting planning and control strategies do not exist. In this paper, we analyze the behavior of vine robots interacting with their environment and propose an obstacle-interaction model that characterizes the bending and wrinkling deformation induced by the environment. Starting from this, we devise a novel obstacle-interaction planning method for these robots. We show how obstacle interactions can be effectively leveraged to enlarge the set of reachable workspace for the robot tip, and verify our findings with both simulated and real experiments. Our work improves the capabilities of this new class of soft robot, helping to advance the field of soft robotics.

- A New Path Planning Architecture to Consider Motion Uncertainty in Natural Environment

    Author: Mizuno, Michihiro | The University of Tokyo
    Author: Kubota, Takashi | JAXA ISAS
 
    keyword: Motion and Path Planning; Field Robots; Space Robotics and Automation

    Abstract : This paper proposes a new path planning archi- tecture with consideration of motion uncertainty for wheeled robots in rough terrain. The proposed scheme uses particles to express the uncertainty propagation in the complicated environments constructed with various types of terrain. Also, RRT (Rapidly-exploring Random Tree) is expanded based on the uncertainty of each node in order to prevent increasing the accumulated position uncertainty. As a result, the generated path recudes the times of path-following and re-planning due to inaccurate localization. The effetiveness of the proposed method is evaluated in the simulation using the motion uncertainty models obtained by experiments. The results show that the proposed method decreases the position uncertainty while it keeps the probability to avoid collisions and to reach the goal area compared with conventional approaches.

- Revisiting the Asymptotic Optimality of RRT*

    Author: Solovey, Kiril | Stanford University
    Author: Janson, Lucas | Harvard University
    Author: Schmerling, Edward | Waymo
    Author: Frazzoli, Emilio | ETH Zurich
    Author: Pavone, Marco | Stanford University
 
    keyword: Motion and Path Planning

    Abstract : RRT* is one of the most widely used sampling-based algorithms for asymptotically-optimal motion planning. RRT* laid the foundations for optimality in motion planning as a whole, and inspired the development of numerous new algorithms in the field, many of which build upon RRT* itself. In this paper, we first identify a logical gap in the optimality proof of RRT*, which was developed in Karaman and Frazzoli (2011). Then, we present an alternative and mathematically-rigorous proof for asymptotic optimality. Our proof suggests that the connection radius used by RRT* should be increased from gamma left(frac{log n}{n}right)^{1/d} to gamma' left(frac{log n}{n}right)^{1/(d+1)} in order to account for the additional dimension of time that dictates the samples' ordering. Here gamma, gamma' are constants, and n, d are the number of samples and the dimension of the problem, respectively.

- Sample Complexity of Probabilistic Roadmaps Via Epsilon Nets

    Author: Tsao, Matthew | Stanford University
    Author: Solovey, Kiril | Stanford University
    Author: Pavone, Marco | Stanford University
 
    keyword: Motion and Path Planning

    Abstract : We study fundamental theoretical aspects of probabilistic roadmaps (PRM) in the finite time (non-asymptotic) regime. In particular, we investigate how completeness and optimality guarantees of the approach are influenced by the underlying deterministic sampling distribution X and connection radius r. We develop the notion of (delta,epsilon)-completeness of the parameters X, r, which indicates that for every motion-planning problem of clearance at least delta&gt;0, PRM using X, r returns a solution no longer than 1+epsilon times the shortest delta-clear path. Leveraging the concept of epsilon-nets, we characterize in terms of lower and upper bounds the number of samples needed to guarantee (delta,epsilon)-completeness. This is in contrast with previous work which mostly considered the asymptotic regime in which the number of samples tends to infinity. In practice, we propose a sampling distribution inspired by epsilon-nets that achieves nearly the same coverage as grids while using significantly fewer samples.

- Reinforcement Learning Based Manipulation Skill Transferring for Robot-Assisted Minimally Invasive Surgery

    Author: Su, Hang | Politecnico Di Milano
    Author: Hu, Yingbai | Technische Universitét M�nchen
    Author: Li, Zhijun | University of Science and Technology of China
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
    Author: Ferrigno, Giancarlo | Politecnico Di Milano
    Author: De Momi, Elena | Politecnico Di Milano
 
    keyword: Motion and Path Planning; Kinematics; Medical Robots and Systems

    Abstract : The complexity of surgical operation can be released significantly if surgical robots can learn the manipulation skills by imitation from complex tasks demonstrations such as puncture, suturing, and knotting, etc.. This paper proposes a reinforcement learning algorithm based manipulation skill transferring technique for robot-assisted Minimally Invasive Surgery by Teaching by Demonstration. It employed Gaussian mixture model and Gaussian mixture Regression based dynamic movement primitive to model the high-dimensional human-like manipulation skill after multiple demonstrations. Furthermore, this approach fascinates the learning and trial phase performed offline, which reduces the risks and cost for the practical surgical operation. Finally, it is demonstrated by transferring manipulation skills for reaching and puncture using a KUKA LWR4+ robot in a lab setup environment. The results show the effectiveness of the proposed approach for modelling and learning of human manipulation skill.

- Safe Mission Planning under Dynamical Uncertainties

    Author: Lu, Yimeng | ETH Zurich
    Author: Kamgarpour, Maryam | University of British Columbia
 
    keyword: Motion and Path Planning; Search and Rescue Robots; Formal Methods in Robotics and Automation

    Abstract : This paper considers safe robot mission planning in uncertain dynamical environments. This problem arises in applications such as surveillance, emergency rescue and autonomous driving. It is a challenging problem due to modeling and integrating dynamical uncertainties into a safe planning framework, and &#64257;nding a solution in a computationally tractable way. In this work, we &#64257;rst develop a probabilistic model for dynamical uncertainties. Then, we provide a framework to generate a path that maximizes safety for complex missions by incorporating the uncertainty model. We also devise a Monte Carlo method to obtain a safe path ef&#64257;ciently. Finally, we evaluate the performance of our approach and compare it to potential alternatives in several case studies.

- An Iterative Quadratic Method for General-Sum Differential Games with Feedback Linearizable Dynamics

    Author: Fridovich-Keil, David | University of California, Berkeley
    Author: Rubies Royo, Vicenc | UC Berkeley
    Author: Tomlin, Claire | UC Berkeley
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems; Motion and Path Planning

    Abstract : Iterative linear-quadratic (ILQ) methods are widely used in the nonlinear optimal control community. Recent work has applied similar methodology in the setting of multi-player general-sum differential games. Here, ILQ methods are capable of finding local equilibria in interactive motion planning problems in real-time. As in most iterative procedures, however, this approach can be sensitive to initial conditions and hyperparameter choices, which can result in poor computational performance or even unsafe trajectories. In this paper, we focus our attention on a broad class of dynamical systems which are feedback linearizable, and exploit this structure to improve both algorithmic reliability and runtime. We showcase our new algorithm in three distinct traffic scenarios, and observe that in practice our method converges significantly more often and more quickly than was possible without exploiting the feedback linearizable structure.

- Real-Time UAV Path Planning for Autonomous Urban Scene Reconstruction

    Author: Kuang, Qi | Beihang University
    Author: WuJinbo, WuJinbo | BUAA
    Author: Pan, Jia | University of Hong Kong
    Author: Zhou, Bin | Beihang University
 
    keyword: Motion and Path Planning; Computer Vision for Other Robotic Applications; Visual-Based Navigation

    Abstract : Unmanned aerial vehicles (UAVs) are frequently used for large-scale scene mapping and reconstruction. However, in most cases, drones are operated manually, which should be more effective and intelligent. In this article, we present a method of real-time UAV path planning for autonomous urban scene reconstruction. Considering the obstacles and time costs, we utilize the top view to generate the initial path. Then we estimate the building heights and take close-up pictures that reveal building details through a SLAM framework. To predict the coverage of the scene, we propose a novel method which combines information on reconstructed point clouds and possible coverage areas. The experimental results reveal that the reconstruction quality of our method is good enough. Our method is also more time-saving than the state-of-the-arts.

- A Fast Marching Gradient Sampling Strategy for Motion Planning Using an Informed Certificate Set
 
    Author: Shi, Shenglei | Huazhong University of Science and Technology
    Author: Chen, Jiankui | Huazhong University of Science and Technology
    Author: Xiong, Youlun | Huazhong University of Science and Technology
 
    keyword: Motion and Path Planning

    Abstract : We present a novel fast marching gradient sampling strategy to accelerate the convergence speed of sampling-based motion planning algorithms. This strategy is based on an informed certificate set which consists of the robot states with exact collision status as well as the minimum distance and the gradient to the nearest obstacle. The informed certificate set covers almost the whole planning space such that it contains rich information for the planner. The best quality point in this set is selected as the marching seed to guide the search graph move steadily to the goal set. Besides, the distance and gradient information of the marching seed helps to generate a new sample with almost sure collision status. When a feasible solution has been found, this set can construct the restricted subset that can truly improve current path quality. This marching gradient sampling strategy is applied to the RRT (Rapidly exploring random tree) and RRT* algorithms. Simulation experiments demonstrate that the convergence speed to a feasible solution or to the optimal solution is almost twice faster than that of the safety certificate algorithms.

- Privacy-Aware UAV Flights through Self-Configuring Motion Planning

    Author: Luo, Yixing | Peking University
    Author: Yu, Yijun | The Open University
    Author: Jin, Zhi | Peking University
    Author: Li, Yao | Zhejiang Sci-Tech University
    Author: Ding, Zuohua | Zhejiang Sci-Tech University
    Author: Zhou, Yuan | Nanyang Technological University
    Author: Liu, Yang | Nanyang Technological University
 
    keyword: Motion and Path Planning; Robust/Adaptive Control of Robotic Systems; Energy and Environment-Aware Automation

    Abstract : During flights, an unmanned aerial vehicle (UAV) may not be allowed to move across certain areas due to soft constraints such as privacy restrictions. Current methods on self-adaption focus mostly on motion planning such that the trajectory does not trespass predetermined restricted areas. When the environment is cluttered with uncertain obstacles, however, these motion planning algorithms are not flexible enough to find a trajectory that satisfies additional privacy-preserving requirements within a tight time budget during the flights. In this paper, we propose a privacy risk aware motion planning method through the reconfiguration of privacy-sensitive sensors. It minimises environmental impact by re-configuring the sensor during flight, while still guaranteeing the hard safety and energy constraints such as collision avoidance and timeliness. First, we formulate a model for assessing privacy risks of dynamically detected restricted areas. In case the UAV cannot find a feasible solution to satisfy both hard and soft constraints from the current configuration, our decision making method can then produce an optimal reconfiguration of the privacy-sensitive sensor with a more efficient trajectory. We evaluate the proposal through various simulations with different settings in a virtual environment and also validate the approach through real test flights on DJI Matrice 100 UAV.

- Improved C-Space Exploration and Path Planning for Robotic Manipulators Using Distance Information

    Author: Lacevic, Bakir | University of Sarajevo
    Author: Osmankovic, Dinko | Faculty of Electrical Engineering Sarajevo
 
    keyword: Motion and Path Planning

    Abstract : We present a simple method to quickly explore C-spaces of robotic manipulators and thus facilitate path planning. The method is based on a novel geometrical structure called generalized bur. It is a star-like tree, rooted at a given point in free C-space, with an arbitrary number of guaranteed collision-free edges computed using distance information from the workspace and simple forward kinematics. Generalized bur captures large portions of free C-space, enabling accelerated exploration. The workspace is assumed to be decomposable into a finite set of (possibly overlapping) convex obstacles. When plugged in a suitable RRT-like planning algorithm, generalized burs enable significant performance improvements, while at the same time enabling exact collision-free paths.

- Tuning-Free Contact-Implicit Trajectory Optimization

    Author: Onol, Aykut Ozgun | Northeastern University
    Author: Corcodel, Radu Ioan | Mitsubishi Electric Research Laboratories
    Author: Long, Philip | Irish Manufacturing Research
    Author: Padir, Taskin | Northeastern University
 
    keyword: Motion and Path Planning; Planning, Scheduling and Coordination; Optimization and Optimal Control

    Abstract : We present a contact-implicit trajectory optimization framework that can plan contact-interaction trajectories for different robot architectures and tasks using a trivial initial guess and without requiring any parameter tuning. This is achieved by using a relaxed contact model along with an automatic penalty adjustment loop for suppressing the relaxation. Moreover, the structure of the problem enables us to exploit the contact information implied by the use of relaxation in the previous iteration, such that the solution is explicitly improved with little computational overhead. We test the proposed approach in simulation experiments for non-prehensile manipulation using a 7-DOF arm and a mobile robot and for planar locomotion using a humanoid-like robot in zero gravity. The results demonstrate that our method provides an out-of-the-box solution with good performance for a wide range of applications.

- PPCPP: A Predator�Prey-Based Approach to Adaptive Coverage Path Planning (I)

    Author: Hassan, Mahdi | University of Technology, Sydney
    Author: Liu, Dikai | University of Technology, Sydney
 
    keyword: Motion and Path Planning; Learning and Adaptive Systems; Collision Avoidance

    Abstract : Most of the existing coverage path planning (CPP)algorithms do not have the capability of enabling a robot to handle unexpected changes in the coverage area of interest. Examples of unexpected changes include the sudden introduction of stationary or dynamic obstacles in the environment and change in the reachable area for coverage (e.g., due to imperfect base localization by an industrial robot). Thus, a novel adaptive CPP approach is developed that is efficient to respond to changes in real-time while aiming to achieve complete coverage with minimal cost. As part of the approach, a total reward function that incorporates three rewards is designed where the first reward is inspired by the predator�prey relation, the second reward is related to continuing motion in a straight direction, and the third reward is related to covering the boundary.The total reward function acts as a heuristic to guide the robot at each step. For a given map of an environment, model parameters are first tuned offline tominimize the path length while assuming no obstacles. It is shown that applying these learned parameters during real-time adaptive planning in the presence of obstacles will still result in a coverage path with a length close to the optimized path length.Many case studies with various scenarios are presented to validate the approach and to perform numerous comparisons.

- Advanced BIT* (ABIT*): Sampling-Based Planning with Advanced Graph-Search Techniques

    Author: Strub, Marlin Polo | University of Oxford
    Author: Gammell, Jonathan | University of Oxford
 
    keyword: Motion and Path Planning; Space Robotics and Automation; Autonomous Vehicle Navigation

    Abstract : Path planning is an active area of research essential for many applications in robotics. Popular techniques include graph-based searches and sampling-based planners. These approaches are powerful but have limitations.<p>This paper continues work to combine their strengths and mitigate their limitations using a unified planning paradigm. It does this by viewing the path planning problem as the two subproblems of search and approximation and using advanced graph-search techniques on a sampling-based approximation.</p><p>This perspective leads to Advanced BIT*. ABIT* combines truncated anytime graph-based searches, such as ATD*, with anytime almost-surely asymptotically optimal sampling-based planners, such as RRT*. This allows it to quickly find initial solutions and then converge towards the optimum in an anytime manner. ABIT* outperforms existing single-query, sampling-based planners on the tested problems in R<sup>4</sup> and R<sup>8</sup>, and was demonstrated on real-world problems with NASA/JPL-Caltech.

- Voxel-Based General Voronoi Diagram for Complex Data with Application on Motion Planning

    Author: Dorn, Sebastian | Daimler
    Author: Wolpert, Nicola | HFT Stuttgart
    Author: Sch�mer, Elmar | Mainz University
 
    keyword: Motion and Path Planning; Computational Geometry

    Abstract : One major challenge in Assembly Sequence Planning (ASP) for complex real-world CAD-scenarios is to find appropriate disassembly paths for all assembled parts. Such a path places demands on its length and clearance. In the past, it became apparent that planning the disassembly path based on the (approximate) General Voronoi Diagram (GVD) is a good approach to achieve these requirements. But for complex real-world data, every known solution for computing the GVD is either too slow or very memory consuming, even if only approximating the GVD. We present a new approach for computing the approximate GVD and demonstrate its practicability using a representative vehicle data set. We can calculate an approximation of the GVD within minutes and meet the accuracy requirement of some few millimeters for the subsequent path planning. This is achieved by voxelizing the surface with a common errorbounded GPU render approach. We then use an error-bounded wavefront propagation technique and combine it with a novel hash table-based data structure, the so-called Voronoi Voxel History (VVH). On top of the GVD, we present a novel approach for the creation of a General Voronoi Diagram Graph (GVDG) that leads to an extensive roadmap. This roadmap can be used to suggest appropriate disassembly paths for the later task of motion planning.

- Dynamic Movement Primitives for Moving Goals with Temporal Scaling Adaptation

    Author: Koutras, Leonidas | Aristotle University of Thessaloniki
    Author: Doulgeri, Zoe | Aristotle University of Thessaloniki
 
    keyword: Motion and Path Planning

    Abstract : In this work, we propose an augmentation to the Dynamic Movement Primitives (DMP) framework which allows the system to generalize to moving goals without the use of any known or approximation model for estimating the goal's motion. We aim to maintain the demonstrated velocity levels during the execution to the moving goal, generating motion profiles appropriate for human robot collaboration. The proposed method employs a modified version of a DMP, learned by a demonstration to a static goal, with adaptive temporal scaling in order to achieve reaching of the moving goal with the learned kinematic pattern. Only the current position and velocity of the goal are required. The goal's reaching error and its derivative is proved to converge to zero via contraction analysis. The theoretical results are verified by simulations and experiments on a KUKA LWR4+ robot.

- Navigating Discrete Difference Equation Governed WMR by Virtual Linear Leader Guided HMPC

    Author: Huang, Chao | Nanjing University
    Author: Chen, Xin | Nanjing University
    Author: Tang, Enyi | Nanjing University
    Author: He, Mengda | Teesside University
    Author: Bu, Lei | Nanjing University
    Author: Qin, Shengchao | Teesside University
    Author: Zeng, Yifeng | Teesside University
 
    keyword: Nonholonomic Motion Planning; Optimization and Optimal Control; Motion and Path Planning

    Abstract : In this paper, we revisit model predictive control (MPC), for the classical wheeled mobile robot (WMR) navigation problem. We prove that the reachable set based hierarchical MPC (HMPC), a state-of-the-art MPC, cannot handle WMR navigation in theory due to the non-existence of non-trivial linear system with an under-approximate reachable set of WMR. Nevertheless, we propose a virtual linear leader based MPC (VLL-MPC) to enable HMPC structure. Different from current HMPCs, we use a virtual linear system with an under-approximate path set rather than the traditional reachable set to guide the WMR. We provide a valid construction of the virtual linear leader. We prove the stability of VLL-MPC, and discuss its complexity. In the experiment, we demonstrate the advantage of VLL-MPC empirically by comparing it with NMPC, LMPC and anytime RRT* in several scenarios.

- Dispertio: Optimal Sampling for Safe Deterministic Sampling-Based Motion Planning

    Author: Palmieri, Luigi | Robert Bosch GmbH
    Author: Bruns, Leonard | KTH Royal Institute of Technology
    Author: Meurer, Michael | German Aerospace Center (DLR) and RWTH Aachen University (RWTH)
    Author: Arras, Kai Oliver | Bosch Research
 
    keyword: Nonholonomic Motion Planning; Motion and Path Planning; Robot Safety

    Abstract : A key challenge in robotics is the efficient generation of optimal robot motion with safety guarantees in cluttered environments. Recently, deterministic optimal sampling-based motion planners have been shown to achieve good performance towards this end, in particular in terms of planning efficiency, final solution cost, quality guarantees as well as non-probabilistic completeness. Yet their application is still limited to relatively simple systems (i.e., linear, holonomic, Euclidean state spaces). In this work, we extend this technique to the class of symmetric and optimal driftless systems by presenting Dispertio, an offline dispersion optimization technique for computing sampling sets, aware of differential constraints, for sampling-based robot motion planning. We prove that the approach, when combined with PRM*, is deterministically complete and retains asymptotic optimality. Furthermore, in our experiments we show that the proposed deterministic sampling technique outperforms several baselines and alternative methods in terms of planning efficiency and solution cost.

- Aggregation and Localization of Simple Robots in Curved Environments

    Author: Moan, Rachel | Winthrop University
    Author: Montano, Victor | University of Houston
    Author: Becker, Aaron | University of Houston
    Author: O'Kane, Jason | University of South Carolina
 
    keyword: Nonholonomic Motion Planning; Medical Robots and Systems; Localization

    Abstract : This paper is about the closely-related problems of localization and aggregation for extremely simple robots, for which the only available action is to move in a given direction as far as the geometry of the environment allows. Such problems may arise, for example, in biomedical applications, wherein a large group of tiny robots moves in response to a shared external stimulus. Specifically, we extend the prior work on these kinds of problems presenting two algorithms for localization in environments with curved (rather than polygonal) boundaries and under low-friction models of interaction with the environment boundaries. We present both simulations and physical demonstrations to validate the approach.

- Interpretable Run-Time Monitoring and Replanning for Safe Autonomous Systems Operations

    Author: Di Franco, Carmelo | University of Virginia
    Author: Bezzo, Nicola | University of Virginia
 
    keyword: Motion and Path Planning; Aerial Systems: Applications; Collision Avoidance

    Abstract : Autonomous robots, especially aerial vehicles, when subject to disturbances, uncertainties, and noises may experience variations from their desired states and deviations from the planned trajectory which may lead them into an unsafe state (e.g., a collision). It is thus necessary to monitor their states at run-time when operating in uncertain and cluttered environments and intervene to guarantee their and the surrounding's safety. While Reachability Analysis (RA) has been successfully used to provide safety guarantees, it doesn't provide explanations on why a system is predicted to be unsafe and what type of corrective actions to perform to change the decision. In this work we propose a novel approach for run-time monitoring that leverages a library of previously observed trajectories together with decision tree theory to predict if the system will be safe/unsafe and provide an explanation to understand the causes of the prediction. We design an interpretable monitor that checks at run-time if the vehicle may become unsafe and plan safe corrective actions if found unsafe. For each prediction, we provide a logical explanation -- a decision rule -- that includes information about the causes that lead to the predicted safety decision. The explanation also includes a set of counterfactual rules that shows what system variables may bring the system to the opposite safety decision, if changed. We leverage such an explanation to plan corrective actions that always keep the vehicle s

- An Efficient Sampling-Based Method for Online Informative Path Planning in Unknown Environments

    Author: Schmid, Lukas Maximilian | ETH Zurich
    Author: Pantic, Michael | ETH Zurich
    Author: Khanna, Raghav | ETH Zurich
    Author: Ott, Lionel | University of Sydney
    Author: Siegwart, Roland | ETH Zurich
    Author: Nieto, Juan | ETH Zurich
 
    keyword: Motion and Path Planning; Aerial Systems: Perception and Autonomy; Reactive and Sensor-Based Planning

    Abstract : The ability to plan informative paths online is essential to robot autonomy. In particular, sampling-based approaches are often used as they are capable of using arbitrary information gain formulations. However, they are prone to local minima, resulting in sub-optimal trajectories, and sometimes do not reach global coverage. In this paper, we present a new RRT*-inspired online informative path planning algorithm. Our method continuously expands a single tree of candidate trajectories and rewires segments to maintain the tree and refine intermediate trajectories. This allows the algorithm to achieve global coverage and maximize the utility of a path in a global context, using a single objective function. We demonstrate the algorithm's capabilities in the applications of autonomous indoor exploration as well as accurate Truncated Signed Distance Field (TSDF)-based 3D reconstruction on-board a Micro Aerial Vehicle (MAV). We study the impact of commonly used information gain and cost formulations in these scenarios and propose a novel TSDF-based 3D reconstruction gain and cost-utility formulation. Detailed evaluation in realistic simulation environments show that our approach outperforms state of the art methods in these tasks. Experiments on a real MAV demonstrate the ability of our method to robustly plan in real-time, exploring an indoor environment with on-board sensing and computation. We make our framework available for future research.

- Koopman Operator Method for Chance-Constrained Motion Primitive Planning

    Author: Gutow, Geordan | Georgia Institute of Technology
    Author: Rogers, Jonathan | Georgia Institute of Technology
 
    keyword: Motion and Path Planning; Collision Avoidance; Probability and Statistical Methods

    Abstract : The use of motion primitives to plan trajectories has received significant attention in the robotics literature. This work considers the application of motion primitives to path planning and obstacle avoidance problems in which the system is subject to significant parametric and/or initial condition uncertainty. In problems involving parametric uncertainty, optimal path planning is achieved by minimizing the expected value of a cost function subject to probabilistic (chance) constraints on vehicle-obstacle collisions. The Koopman operator provides an efficient means to compute expected values for systems under parametric uncertainty. In the context of motion planning, these include both the expected cost function and chance constraints. This work describes a maneuver-based planning method that leverages the Koopman operator to minimize an expected cost while satisfying user-imposed risk tolerances. The developed method is illustrated in two separate examples using a Dubins car model subject to parametric uncertainty in its dynamics or environment. Prediction of constraint violation probability is compared with a Monte Carlo method to demonstrate the advantages of the Koopman-based calculation.

- Robust Humanoid Contact Planning with Learned Zero and One-Step Capturability Prediction

    Author: Lin, Yu-Chi | University of Michigan
    Author: Righetti, Ludovic | New York University
    Author: Berenson, Dmitry | University of Michigan
 
    keyword: Motion and Path Planning; Humanoid and Bipedal Locomotion; Deep Learning in Robotics and Automation

    Abstract : Humanoid robots maintain balance and navigate by controlling the contact wrenches applied to the environment. While it is possible to plan dynamically-feasible motion that applies appropriate wrenches using existing methods, a humanoid may also be affected by external disturbances. Existing systems typically rely on controllers to reactively recover from disturbances. However, such controllers may fail when the robot cannot reach contacts capable of rejecting a given disturbance. In this paper, we propose a search-based footstep planner which aims to maximize the probability of the robot successfully reaching the goal without falling as a result of a disturbance. The planner considers not only the poses of the planned contact sequence, but also alternative contacts near the planned contact sequence that can be used to recover from external disturbances. Although this additional consideration significantly increases the computation load, we train neural networks to efficiently predict multi-contact zero-step and one-step capturability, which allows the planner to generate robust contact sequences efficiently. Our results show that our approach generates footstep sequences that are more robust to external disturbances than a conventional footstep planner in four challenging scenarios.

- Differential Flatness Based Path Planning with Direct Collocation on Hybrid Modes for a Quadrotor with a Cable-Suspended Payload

    Author: Zeng, Jun | University of California, Berkeley
    Author: Kotaru, Venkata Naga Prasanth | University of California Berkeley
    Author: Mueller, Mark Wilfried | University of California, Berkeley
    Author: Sreenath, Koushil | University of California, Berkeley
 
    keyword: Motion and Path Planning; Aerial Systems: Applications; Optimization and Optimal Control

    Abstract : Generating agile maneuvers for a quadrotor with a cable-suspended load is a challenging problem. State-of-the-art approaches often need significant computation time and complex parameter tuning. We use a coordinate-free geometric formulation and exploit a differential flatness based hybrid model of a quadrotor with a cable-suspended payload. We perform direct collocation on the differentially-flat hybrid system, and use complementarity constraints to avoid specifying hybrid mode sequences. The non-differentiable obstacle avoidance constraints are reformulated using dual variables, resulting in smooth constraints. We show that our approach has lower computational time than the state-of-the-art and guarantees feasibility of the trajectory with respect to both the system dynamics and input constraints without the need to tune lots of parameters. We validate our approach on a variety of tasks in both simulations and experiments, including navigation through waypoints and obstacle avoidance.

- A Real-Time Approach for Chance-Constrained Motion Planning with Dynamic Obstacles

    Author: Castillo-Lopez, Manuel | University of Luxembourg
    Author: Ludivig, Philippe | University of Luxembourg
    Author: Sajadi-Alamdari, Seyed Amin | University of Luxembourg
    Author: Sanchez-Lopez, Jose Luis | Interdisciplinary Center for Security, Reliability and Trust (Sn
    Author: Olivares-Mendez, Miguel A. | Interdisciplinary Centre for Security, Reliability and Trust - U
    Author: Voos, Holger | University of Luxembourg
 
    keyword: Motion and Path Planning; Collision Avoidance; Optimization and Optimal Control

    Abstract : Uncertain dynamic obstacles such as pedestrians or vehicles pose a major challenge for optimal robot navigation with safety guarantees. Previous work on motion planning have followed two main strategies to provide a safe bound on an obstacle's space: a polyhedron, such as a cuboid, or a nonlinear differentiable surface, such as an ellipsoid. The former approach relies on disjunctive programming which, in general, is N P-hard and its size grows exponentially with the number of obstacles. The latter approach needs to be linearized locally to find a tractable evaluation of the chance constraints, which reduces dramatically the remaining free space and leads to over-conservative trajectories or even unfeasibility. In this work, we present a hybrid approach that eludes the pitfalls of both strategies while maintaining the same safety guarantees. The key idea consists in obtaining safe differentiable bounds on the disjunctive chance constraints on the obstacles. The resulting nonlinear optimization problem is free of obstacle linearization and disjunctive programming, and therefore, it can be efficiently solved to meet fast real-time requirements with multiple obstacles. We validate our approach through mathematical proof, simulation and real experiments with an aerial robot using nonlinear model predictive control to avoid pedestrians.

- Learning When to Trust a Dynamics Model for Planning in Reduced State Spaces

    Author: McConachie, Dale Steven | University of Michigan, Ann Arbor
    Author: Power, Thomas | Robotics Institute, University of Michigan
    Author: Mitrano, Peter | University of Michigan
    Author: Berenson, Dmitry | University of Michigan
 
    keyword: Motion and Path Planning; Learning and Adaptive Systems

    Abstract : When the dynamics of a system are difficult to model and/or time-consuming to evaluate, such as in deformable object manipulation tasks, motion planning algorithms struggle to find feasible plans efficiently. Such problems are often reduced to state spaces where the dynamics are straightforward to model and evaluate. However, such reductions usually discard information about the system for the benefit of computational efficiency, leading to cases where the true and reduced dynamics disagree on the result of an action. This paper presents a formulation for planning in reduced state spaces that uses a classifier to bias the planner away from state-action pairs that are not reliably feasible under the true dynamics. We present a method to generate and label data to train such a classifier, as well as an application of our framework to rope manipulation, where we use a Virtual Elastic Band (VEB) approximation to the true dynamics. Our experiments with rope manipulation demonstrate that the classifier significantly improves the success rate of our RRT-based planner in several difficult scenarios which are designed to cause the VEB to produce incorrect predictions in key parts of the environment.

-  MIST: A Single-Query Path Planning Approach Using Memory and Information-Sharing Trees

    Author: Rakita, Daniel | University of Wisconsin-Madison
    Author: Mutlu, Bilge | University of Wisconsin-Madison
    Author: Gleicher, Michael | University of Wisconsin-Madison

- Fast Planning Over Roadmaps Via Selective Densification

    Author: Saund, Brad | University of Michigan
    Author: Berenson, Dmitry | University of Michigan
 
    keyword: Motion and Path Planning

    Abstract : We propose the Selective Densification method for fast motion planning through configuration space. We create a sequence of roadmaps by iteratively adding configurations. We organize these roadmaps into layers and add edges between identical configurations between layers. We find a path using best-first search, guided by our proposed estimate of remaining planning time. This estimate prefers to expand nodes closer to the goal and nodes on sparser layers.<p>We present proofs of the path quality and maximum depth of nodes expanded using our proposed graph and heuristic. We also present experiments comparing Selective Densification to bidirectional RRT-connect, as well as many graph search approaches. In difficult environments that require exploration on the dense layers we find Selective Densification finds solutions faster than all other approaches.

- Refined Analysis of Asymptotically-Optimal Kinodynamic Planning in the State-Cost Space

    Author: Kleinbort, Michal | Tel Aviv University
    Author: Granados, Edgar | Rutgers
    Author: Solovey, Kiril | Stanford University
    Author: Bonalli, Riccardo | Stanford University
    Author: Bekris, Kostas E. | Rutgers, the State University of New Jersey
    Author: Halperin, Dan | Tel Aviv University
 
    keyword: Motion and Path Planning; Optimization and Optimal Control

    Abstract : We present a novel analysis of AO-RRT: a tree-based planner for motion planning with kinodynamic constraints, originally described by Hauser and Zhou (AO-X, 2016). AO-RRT explores the state-cost space and has been shown to efficiently obtain high-quality solutions in practice without relying on the availability of a computationally-intensive two-point boundary-value solver. Our main contribution is an optimality proof for the single-tree version of the algorithm---a variant that was not analyzed before. Our proof only requires a mild and easily-verifiable set of assumptions on the problem and system: Lipschitz-continuity of the cost function and the dynamics. In particular, we prove that for any system satisfying these assumptions, any trajectory having a piecewise-constant control function and positive clearance from the obstacles can be approximated arbitrarily well by a trajectory found by AO-RRT. We also discuss practical aspects of AO-RRT and present experimental comparisons of variants of the algorithm.

- Polygon-Based Random Tree Search Planning for Variable Geometry Truss Robot

    Author: Park, Sumin | Seoul National University
    Author: Bae, Jangho | Seoul National University
    Author: Lee, Seohyeon | Hanyang University
    Author: Yim, Mark | University of Pennsylvania
    Author: Kim, Jongwon | Seoul National University
    Author: Seo, TaeWon | Hanyang University
 
    keyword: Motion and Path Planning; Cellular and Modular Robots

    Abstract : This paper proposes the use of a polygon-based random tree path planning algorithm for a variable geometry topology system (VGT). By combining a path planning algorithm and our previous non-impact locomotion algorithm, the proposed VGT system reaches a objective point. The proposed path planning algorithm provides desired set of support polygons with a modified rapid random tree algorithm. The algorithm can significantly reduce distortion of the VGT system while moving by limiting the deformation of the desired support polygon. With this algorithm feature, constraint violations of the system were significantly reduced when using a normal rapid random tree algorithm for path planning. The performance of the algorithm was validated using the simulation results.

- An Iterative Dynamic Programming Approach to the Multipoint Markov-Dubins Problem

    Author: Frego, Marco | University of Trento
    Author: Bevilacqua, Paolo | University of Trento
    Author: Saccon, Enrico | University of Trento
    Author: Palopoli, Luigi | University of Trento
    Author: Fontanelli, Daniele | University of Trento
 
    keyword: Motion and Path Planning; Nonholonomic Motion Planning; Computational Geometry

    Abstract : A new solution to the multipoint Markov-Dubins problem via Iterative Dynamic Programming is herein presented. The shortest path problem connecting a sequence of given points in the plane while maintaining angle continuity and bounded curvature is presented. As in the classic two points Dubins problem, the solution is a juxtaposition of line segments and circle arcs. This problem is relevant for the path planning of a non-holonomic robot, such as a wheeled vehicle. The proposed method is robust and computationally inexpensive with respect to existing solutions and is therefore suitable to be integrated as motion primitive into Dubins-based applications, e.g. orienteering problems or waypoint following robotics.

- GOMP: Grasp-Optimized Motion Planning for Bin Picking

    Author: Ichnowski, Jeffrey | UC Berkeley
    Author: Danielczuk, Michael | UC Berkeley
    Author: Xu, Jingyi | Technical University of Munich
    Author: Satish, Vishal | UC Berkeley
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Motion and Path Planning

    Abstract : Rapid and reliable robot bin picking is a critical challenge in automating warehouses, often measured in picks-per-hour (PPH). We explore increasing PPH using faster motions based on optimizing over a set of candidate grasps. The source of this set of grasps is two-fold: (1) grasp-analysis tools such as Dex-Net generate multiple candidate grasps, and (2) each of these grasps has a degree of freedom about which a robot gripper can rotate. In this paper, we present Grasp-Optimized Motion Planning (GOMP), an algorithm that speeds up the execution of a bin-picking robot's operations by incorporating robot dynamics and a set of candidate grasps produced by a grasp planner into an optimizing motion planner. We compute motions by optimizing with sequential quadratic programming (SQP) and iteratively updating trust regions to account for the non-convex nature of the problem. In our formulation, we constrain the motion to remain within the mechanical limits of the robot while avoiding obstacles. We further convert the problem to a time-minimization by repeatedly shorting a time horizon of a trajectory until the SQP is infeasible. In experiments with a UR5, GOMP achieves a speedup of 9x over a baseline planner.

- Motion Planning and Task Allocation for a Jumping Rover Team

    Author: Tan, Kai Chuen | Ohio State University
    Author: Jung, Myungjin | The Ohio State University
    Author: Shyu, Isaac | Ohio State University
    Author: Changhuang, Wan | The Ohio State University
    Author: Dai, Ran | The Ohio State University
 
    keyword: Motion and Path Planning; Hybrid Logical/Dynamical Planning and Verification; Planning, Scheduling and Coordination

    Abstract : This paper presents a cooperative robotic team composed of unmanned ground vehicles (UGVs) with hybrid operational modes to tackle the multiple traveling salesman problem (mTSP) with obstacles. The hybrid operational modes allow every UGV in the team to not only travel on a ground surface but also jump over obstacles. We name these UGVs jumping rovers. The jumping capability provides a flexible form of locomotion by leaping and landing on top of obstacles instead of navigating around obstacles. To solve the mTSP, an optimal path between any two objective points in an mTSP is determined by the optimized rapidly-exploring random tree method, named RRT*, and is further improved through a refined RRT* algorithm to find a smoother path between targets. We then formulate the mTSP as a mixed-integer linear programming (MILP) problem to search for the most cost-effective combination of paths for multiple UGVs. The effectiveness of the hybrid operational modes and optimized motion with assigned tasks is verified in an indoor, physical experimental environment using customized jumping rovers.

- Active 3D Modeling Via Online Multi-View Stereo

    Author: Song, Soohwan | KAIST
    Author: Kim, Daekyum | Korea Advanced Institute of Science and Technology
    Author: Jo, Sungho | Korea Advanced Institute of Science and Technology (KAIST)
 
    keyword: Motion and Path Planning; Aerial Systems: Perception and Autonomy; Computer Vision for Automation

    Abstract : Multi-view stereo (MVS) algorithms have been commonly used to model large-scale structures. When processing MVS, image acquisition is an important issue because its reconstruction quality depends heavily on the acquired images. Recently, an explore-then-exploit strategy has been used to acquire images for MVS. This method first constructs a coarse model by exploring an entire scene using a pre-allocated camera trajectory. Then, it rescans the unreconstructed regions from the coarse model. However, this strategy is inefficient because of the frequent overlap of the initial and rescanning trajectories. Furthermore, given the complete coverage of images, MVS algorithms do not guarantee an accurate reconstruction result.<p>In this study, we propose a novel view path-planning method based on an online MVS system. This method aims to incrementally construct the target three-dimensional (3D) model in real time. View paths are continually planned based on online feedbacks from the partially constructed model. The obtained paths fully cover low-quality surfaces while maximizing the reconstruction performance of MVS. Experimental results demonstrate that the proposed method can construct high quality 3D models with one exploration trial, without any rescanning trial as in the explore-then-exploit method.

- Reoriented Short-Cuts (RSC): An Adjustment Method for Locally Optimal Path Short-Cutting in High DoF Configuration Spaces

    Author: Holston, Alexander Christopher | Korean Advanced Institude of Science and Technology
    Author: Kim, Jong-Hwan | KAIST
 
    keyword: Motion and Path Planning

    Abstract : This paper presents Reoriented Short-Cuts (RSC): A modification of the traditional short-cut technique, allowing almost sure, single homotopy class, asymptotic convergence in high degree of freedom (DoF) problems. An additional Informed Gaussian Sampling (IGS) technique is also presented for convergence comparison. Traditionally, Short-Cut methods are used as a final technique to further optimize an initially found path. Typical short-cut methods fail as a single DoF may converge faster than the remaining, creating a zero-volume region between path segments and objects, halting further improvements. Previous attempts to solve this separate DoFs individually, drastically increasing collision checking computation. RSC and IGS control the shifting of the vertex to be Short-Cut, moving vertex positions by reorienting the line segments, removing the zero-volume convergence region. These methods are compared to similar strategies in a variety of problems including random worlds, and robot manipulation, to show the convergence across both translation and rotation oriented problems.

- Learning Resilient Behaviors for Navigation under Uncertainty Environments

    Author: Fan, Tingxiang | The University of Hong Kong
    Author: Long, Pinxin | Baidu Inc
    Author: Liu, Wenxi | Fuzhou University
    Author: Pan, Jia | University of Hong Kong
    Author: Yang, Ruigang | University of Kentucky
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : Deep reinforcement learning has great potential to acquire complex, adaptive behaviors for autonomous agents automatically. However, the underlying neural network polices have not been widely deployed in real-world applications, especially in these safety-critical tasks (e.g., autonomous driving). One of the reasons is that the learned policy cannot perform flexible and resilient behaviors as traditional methods to adapt to diverse environments. In this paper, we consider the problem that a mobile robot learns adaptive and resilient behaviors for navigating in unseen uncertain environments while avoiding collisions. We present a novel approach for uncertainty-aware navigation by introducing an uncertainty-aware predictor to model the environmental uncertainty, and we propose a novel uncertainty-aware navigation network to learn resilient behaviors in the prior unknown environments. To train the proposed uncertainty-aware network more stably and efficiently, we present the temperature decay training paradigm, which balances exploration and exploitation during the training process. Our experimental evaluation demonstrates that our approach can learn resilient behaviors in diverse environments and generate adaptive trajectories according to environmental uncertainties.

- Motion Planning Explorer: Visualizing Local Minima Using a Local-Minima Tree

    Author: Orthey, Andreas | University Stuttgart
    Author: Fr�sz, Benjamin | University of Stuttgart
    Author: Toussaint, Marc | University of Stuttgart
 
    keyword: Motion and Path Planning; Nonholonomic Motion Planning; Formal Methods in Robotics and Automation

    Abstract : Motion planning problems often have many local minima. Those minima are important to visualize to let a user guide, prevent or predict motions. Towards this goal, we develop the motion planning explorer, an algorithm to let users interactively explore a tree of local-minima. Following ideas from Morse theory, we define local minima as paths invariant under minimization of a cost functional. The localminima are grouped into a local-minima tree using lowerdimensional projections specified by a user. The user can then interactively explore the local-minima tree, thereby visualizing the problem structure and guide or prevent motions. We show the motion planning explorer to faithfully capture local minima in four realistic scenarios, both for holonomic and certain nonholonomic robots.

- Fog Robotics Algorithms for Distributed Motion Planning Using Lambda Serverless Computing

    Author: Ichnowski, Jeffrey | University of North Carolina at Chapel Hill
    Author: Lee, William | University of North Carolina - Chapel Hill
    Author: Murta, Victor | University of North Carolina at Chapel Hill
    Author: Paradis, Samuel | University of California, Berkeley
    Author: Alterovitz, Ron | University of North Carolina at Chapel Hill
    Author: Gonzalez, Joseph E. | UC Berkeley
    Author: Stoica, Ion | UC Berkeley
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Motion and Path Planning

    Abstract : For robots using motion planning algorithms such as RRT and RRT*, the computational load can vary by orders of magnitude as the complexity of the local environment changes. To adaptively provide such computation, we propose Fog Robotics algorithms in which cloud-based serverless lambda computing provides parallel computation on demand. To use this parallelism, we propose novel motion planning algorithms that scale effectively with an increasing number of serverless computers. However, given that the allocation of computing is typically bounded by both monetary and time constraints, we show how prior learning can be used to efficiently allocate resources at runtime. We demonstrate the algorithms and application of learned parallel allocation in both simulation and with the Fetch commercial mobile manipulator using Amazon Lambda to complete a sequence of sporadically computationally intensive motion planning tasks.

- Exploration of 3D Terrains Using Potential Fields with Elevation-Based Local Distortions

    Author: Maffei, Renan | Universidade Federal Do Rio Grande Do Sul
    Author: Praisler de Souza, Marcos | Universidade Federal Do Rio Grande Do Sul
    Author: Mantelli, Mathias Fassini | Federal University of Rio Grande Do Sul
    Author: Pittol, Diego | Federal University of Rio Grande Do Sul
    Author: Kolberg, Mariana | UFRGS
    Author: Jorge, Vitor | ITA
 
    keyword: Motion and Path Planning; Mapping

    Abstract : Mobile robots can be used in numerous outdoor tasks such as patrolling, delivery applications, and military. In order to deploy mobile robots in this kind of environment, where there are different challenges like slopes, elevations, or even holes, they should be able to detect such challenges and determine the best path to accomplish their tasks. In this paper, we are proposing an exploration approach based on potential fields with local distortions, in which we define preferences in uneven terrains to avoid high declivity regions without compromising the best path. The approach was implemented and tested in simulated environments, considering a ground robot embedded with two 2D LIDAR sensors, and the experiments demonstrated the efficiency of our method.

- R3T: Rapidly-Exploring Random Reachable Set Tree for Optimal Kinodynamic Planning of Nonlinear Hybrid Systems

    Author: Wu, Albert | Massachusetts Institute of Technology
    Author: Sadraddini, Sadra | MIT
    Author: Tedrake, Russ | Massachusetts Institute of Technology
 
    keyword: Motion and Path Planning; Hybrid Logical/Dynamical Planning and Verification; Optimization and Optimal Control

    Abstract : We introduce R3T, a reachability-based variant of the rapidly-exploring random tree (RRT) algorithm that is suitable for (optimal) kinodynamic planning in nonlinear and hybrid systems. We developed tools to approximate reachable sets using polytopes and perform sampling-based planning with them. This method has a unique advantage in hybrid systems: different dynamic modes in the reachable set can be explicitly represented using multiple polytopes. We prove that under mild assumptions, R3T is probabilistically complete in kinodynamic systems, and asymptotically optimal through rewiring. Moreover, R3T provides a formal verification method for reachability analysis of nonlinear systems. The advantages of R3T are demonstrated with case studies on nonlinear, hybrid, and contact-rich robotic systems.

- DeepSemanticHPPC: Hypothesis-Based Planning Over Uncertain Semantic Point Clouds

    Author: Han, Yutao | Cornell University
    Author: Lin, Hubert | Cornell University
    Author: Banfi, Jacopo | Cornell University
    Author: Bala, Kavita | Cornell University
    Author: Campbell, Mark | Cornell University
 
    keyword: Motion and Path Planning; Visual-Based Navigation; Semantic Scene Understanding

    Abstract : Planning in unstructured environments is challenging - it relies on sensing, perception, scene reconstruction, and reasoning about various uncertainties. We propose DeepSemanticHPPC, a novel uncertainty-aware hypothesis-based planner for unstructured environments. Our algorithmic pipeline consists of: a deep Bayesian neural network which segments surfaces with uncertainty estimates; a flexible point cloud scene representation; a next-best-view planner which minimizes the uncertainty of scene semantics using sparse visual measurements; and a hypothesis-based path planner that proposes multiple kinematically feasible paths with evolving safety confidences given next-best-view measurements. Our pipeline iteratively decreases semantic uncertainty along planned paths, filtering out unsafe paths with high confidence. We show that our framework plans safe paths in real-world environments where existing path planners typically fail.

- Balancing Actuation and Computing Energy in Motion Planning

    Author: Sudhakar, Soumya | Massachusetts Institute of Technology
    Author: Karaman, Sertac | Massachusetts Institute of Technology
    Author: Sze, Vivienne | Massachusetts Institute of Technology
 
    keyword: Motion and Path Planning

    Abstract : We study a novel class of motion planning problems, inspired by emerging low-energy robotic vehicles, such as insect-size flyers, chip-size satellites, and high-endurance autonomous blimps, for which the energy consumed by computing hardware during planning a path can be as large as the energy consumed by actuation hardware during the execution of the same path. We propose a new algorithm, called Compute Energy Included Motion Planning (CEIMP). CEIMP operates similarly to any other anytime planning algorithm, except it stops when it estimates further computing will require more computing energy than potential savings in actuation energy. We show that CEIMP has the same asymptotic computational complexity as existing sampling-based motion planning algorithms, such as PRM*. We also show that CEIMP outperforms the average baseline of using maximum computing resources in realistic computational experiments involving 10 floor plans from MIT buildings. In one representative experiment, CEIMP outperforms the average baseline 90.6% of the time when energy to compute one more second is equal to the energy to move one more meter, and 99.7% of the time when energy to compute one more second is equal to or greater than the energy to move 3 more meters.

- Posterior Sampling for Anytime Motion Planning on Graphs with Expensive-To-Evaluate Edges

    Author: Hou, Brian | University of Washington
    Author: Choudhury, Sanjiban | University of Washington
    Author: Lee, Gilwoo | University of Washington
    Author: Mandalika, Aditya | University of Washington
    Author: Srinivasa, Siddhartha | University of Washington
 
    keyword: Motion and Path Planning; Learning and Adaptive Systems

    Abstract : Collision checking is a computational bottleneck in motion planning, requiring lazy algorithms that explicitly reason about when to perform this computation. Optimism in the face of collision uncertainty minimizes the number of checks before finding the shortest path. However, this may take a prohibitively long time to compute, with no other feasible paths discovered during this period. For many real-time applications, we instead demand strong anytime performance, defined as minimizing the cumulative lengths of the feasible paths yielded over time. We introduce Posterior Sampling for Motion Planning (PSMP), an anytime lazy motion planning algorithm that leverages learned posteriors on edge collisions to quickly discover an initial feasible path and progressively yield shorter paths. PSMP obtains an expected regret bound of tilde{O}(sqrt{S A T}) and outperforms comparative baselines on a set of 2D and 7D planning problems.

## Aerial Systems: Mechanics and Control

- Model Reference Adaptive Control of Multirotor for Missions with Dynamic Change of Payloads During Flight

    Author: Maki, Toshiya | University of Tokyo
    Author: Zhao, Moju | The University of Tokyo
    Author: Shi, Fan | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Aerial Systems: Mechanics and Control; Robust/Adaptive Control of Robotic Systems

    Abstract : Carrying payloads in air is a major mission for multirotor aerial robot. However, the presence of payloads on multirotor aerial robot has a risk of degrading the performance of the flight controller. This concern becomes obvious especially when carrying objects not securely attached to the body or performing aerial manipulation. Therefore, controller with the ability to adapt itself to the effects of payloads on flight stability is needed. This paper proposes a novel nonlinear multiple-input and multiple-output (MIMO) model reference adaptive control (MRAC) system for attitude control of multirotor aerial robots which can dynamically compensate change in the position of center of gravity and inertia caused by payloads. Stability and robustness of the controller are experimentally confirmed in quadrotor and transformable multirotor, and experiments modeling practical applications are conducted for each aerial robot system, proving the utility of the controller.

- Adaptive Air Density Estimation for Precise Tracking Control and Accurate External Wrench Observation for Flying Robots

    Author: Maier, Moritz | German Aerospace Center (DLR)
    Author: Keppler, Manuel | German Aerospace Center (DLR)
    Author: Ott, Christian | German Aerospace Center (DLR)
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Aerial Systems: Mechanics and Control; Robust/Adaptive Control of Robotic Systems; Aerial Systems: Applications

    Abstract : Air density changes depending on the local atmosphere and affects the rotor thrust of flying robots. This effect has to be compensated by the flight controller in order to realize precise tracking of a desired trajectory. So far, the influence of the air density has been disregarded or only considered implicitly in the control of flying robots. In this work, a nonlinear adaptive control approach is presented. It explicitly considers the air density in the dynamical model and enables air density estimation and tracking control under changing atmospheric conditions and with added payload. Furthermore, the estimated air density is used to enhance the accuracy of a state-of-the-art external wrench estimator. The adaptive control approach is evaluated in simulations and experiments with a quadrocopter and a coaxial hexacopter.

- The Tiercel: A Novel Autonomous Micro Aerial Vehicle That Can Map the Environment by Flying into Obstacles

    Author: Mulgaonkar, Yash | University of Pennsylvania
    Author: Liu, Wenxin | University of Pennsylvania
    Author: Thakur, Dinesh | University of Pennsylvania
    Author: Daniilidis, Kostas | University of Pennsylvania
    Author: Taylor, Camillo Jose | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania
 
    keyword: Aerial Systems: Mechanics and Control; Aerial Systems: Perception and Autonomy; Biologically-Inspired Robots

    Abstract : Autonomous flight through unknown environments in the presence of obstacles is a challenging problem for micro aerial vehicles (MAVs). A majority of the current state-of-art research assumes obstacles as opaque objects that can be easily sensed by optical sensors such as cameras or LiDARs. However in indoor environments with glass walls and windows, or scenarios with smoke and dust, robots (even birds) have a difficult time navigating through the unknown space.<p>In this paper, we present the design of a new class of micro aerial vehicles that achieves autonomous navigation and are robust to collisions. In particular, we present the Tiercel MAV: a small, agile, light weight and collision-resilient robot powered by a cellphone grade CPU. Our design exploits contact to infer the presence of transparent or reflective obstacles like glass walls, integrating touch with visual perception for SLAM. The Tiercel is able to localize using visual-inertial odometry (VIO) running on board the robot with a single downward facing fisheye camera and an IMU. We show how our collision detector design and experimental set up enable us to characterize the impact of collisions on VIO. We further develop a planning strategy to enable the Tiercel to fly autonomously in an unknown space, sustaining collisions and creating a 2D map of the environment. Finally we demonstrate a swarm of three autonomous Tiercel robots safely navigating and colliding through an obstacle field to reach their objectives.

- Full-Pose Manipulation Control of a Cable-Suspended Load with Multiple UAVs under Uncertainties

    Author: Sanalitro, Dario | LAAS-CNRS
    Author: Savino, Heitor J | Laboratory for Analysis and Architecture of Systems
    Author: Tognon, Marco | LAAS-CNRS
    Author: Cortes, Juan | LAAS-CNRS
    Author: Franchi, Antonio | University of Twente
 
    keyword: Aerial Systems: Mechanics and Control; Multi-Robot Systems; Mobile Manipulation

    Abstract : In this work we propose an uncertainty-aware controller for the FlyCrane system, a statically rigid cable-suspended aerial manipulator using the minimum number of aerial robots and cables. The force closure property of the FlyCrane makes it ideal for applications where high precision is required and external disturbances should be compensated. The proposed control requires the knowledge of the nominal values of a minimum number of uncertain kinematic parameters, thus simplifying the identification process and the controller implementation. We propose an optimization-based tuning method of the control gains that ensures stability despite parameter uncertainty and maximizes the H-infinity performance. The validity of the proposed framework is shown through real experiments.

- Learning Pugachev's Cobra Maneuver for Tail-Sitter UAVs Using Acceleration Model
 
    Author: Xu, Wei | University of Hong Kong
    Author: Zhang, Fu | University of Hong Kong
 
    keyword: Aerial Systems: Mechanics and Control; Learning and Adaptive Systems; Robust/Adaptive Control of Robotic Systems

    Abstract : The Pugachev's cobra maneuver is a dramatic and demanding maneuver requiring the aircraft to fly at extremely high Angle of Attacks (AOA) where stalling occurs. This paper considers this maneuver on tail-sitter UAVs. We present a simple yet very effective feedback-iterative learning position control structure to regulate the altitude error and lateral displacement during the maneuver. Both the feedback controller and the iterative learning controller are based on the aircraft acceleration model, which is directly measurable by the onboard accelerometer. Moreover, the acceleration model leads to an extremely simple dynamic model that does not require any model identification in designing the position controller, greatly simplifying the implementation of the iterative learning control. Real-world outdoor flight experiments on the ``Hong Hu" UAV, an aerobatic yet efficient quadrotor tail-sitter UAV of small-size, are provided to show the effectiveness of the proposed controller.

- Adaptive Control of Variable-Pitch Propellers: Pursuing Minimum-Effort Operation

    Author: Henderson, Travis | CSE, UMN
    Author: Papanikolopoulos, Nikos | University of Minnesota
 
    keyword: Aerial Systems: Mechanics and Control; Aerial Systems: Applications

    Abstract : As Unmanned Aerial Vehicles (UAVs) become more commonly used in industry, their performance will continue to be challenged. A performance bottleneck that is crucial to overcome is the design of electric propulsion systems for UAVs that operate in disparate flight modes (e.g., hovering and forward-moving flight). While flight mode dissimilarity presents a fundamental design challenge for fixed-geometry propulsion systems, variable-geometry systems such as the Variable Pitch Propeller (VPP) ones are able to provide superior propulsion performance across a wide range of flight modes. This work builds on previous work by the     Authors and presents a VPP system control and estimation framework for safe near-optimal propulsion system behavior across the whole operation state space of any UAV. Multiple validations are presented to support the feasibility of the approach.


- Design and Control of a Variable Aerial Cable Towed System

    Author: Li, Zhen | Ecole Centrale Nantes
    Author: Erskine, Julian | Ecole Centrale De Nantes
    Author: Caro, Stéphane | CNRS/LS2N
    Author: Chriette, Abdelhamid | Ecole Centrale De Nantes
 
    keyword: Aerial Systems: Mechanics and Control; Tendon/Wire Mechanism; Parallel Robots

    Abstract : Aerial Cable Towed Systems (ACTS) are composed of several Unmanned Aerial Vehicles (UAVs) connected to a payload by cables. Compared to towing objects from individual aerial vehicles, an ACTS has significant advantages such as heavier payload capacity, modularity, and full control of the payload pose. They are however generally large with limited ability to meet geometric constraints while avoiding collisions between UAVs. This paper presents the modeling, performance analysis, design, and a proposed controller for a novel ACTS with variable cable lengths, named Variable Aerial Cable Towed System (VACTS).<p>Winches are embedded on the UAVs for actuating the cable lengths similar to a Cable-Driven Parallel Robot to increase the versatility of the ACTS. The general geometric, kinematic and dynamic models of the VACTS are derived, followed by the development of a centralized feedback linearization controller. The design is based on a wrench analysis of the VACTS, without constraining the cables to pass through the UAV center of mass, as in current works. Additionally, the performance of the VACTS and ACTS are compared showing that the added versatility comes at the cost of payload and configuration flexibility. A prototype confirms the feasibility of the system.

- Novel Model-Based Control Mixing Strategy for a Coaxial Push-Pull Multirotor

    Author: Chebbi, Jawhar | ISAE-SUPAERO &amp; Donecle
    Author: Defa�, Fran�ois | IUT Tarbes - GEII - Université De Toulouse
    Author: Briere, Yves | Université De Toulouse ISAE
    Author: Deruaz-Pepin, Alban | Donecle
 
    keyword: Aerial Systems: Mechanics and Control; Calibration and Identification; Force Control

    Abstract : A Coaxial push-pull multirotor is a Vertical Take- Off and Landing (VTOL) Unmanned Aerial Vehicle (UAV) having <i>2n</i> (<i>n &#8712; &#8469;<sup>*</sup></i>) rotors arranged in <i>n</i> blocks of two coaxial contra-rotating rotors. A model-based control allocation algorithm (mixer) for this architecture is proposed. The novelty of the approach lies in the fact that the coaxial aerodynamic interference occurring between the pairs of superimposed rotors is not neglected but rather nonlinear empiric models of the coaxial aerodynamic thrust and torque are used to build the mixer. Real flight experiments were conducted and the new approach showed promising results.

- Robust Quadcopter Control with Artificial Vector Fields

    Author: Rezende, Adriano | Universidade Federal De Minas Gerais
    Author: Gon�alves, Vinicius Mariano | UFMG
    Author: Dias Nunes, Arthur Henrique | Federal University of Minas Gerais
    Author: Pimenta, Luciano | Universidade Federal De Minas Gerais
 
    keyword: Aerial Systems: Mechanics and Control; Motion Control; Motion and Path Planning

    Abstract : This article presents a path tracking control strategy for a quadcopter to follow a time varying curve. The control is based on artificial vector fields. The construction of the field is based on a well known technique in the literature. Next, control laws are developed to impose the behavior of the vector field to a second order integrator model. Finally, control laws are developed to impose the dynamics of the controlled second order integrator to a quadcopter model, which assumes the thrust and the angular rates as input commands. Asymptotic convergence of the whole system is proved by showing that the individual systems in cascade connection are input-to-state stable. We also analyze the influence of norm-bounded disturbances in the control inputs to evaluate the robustness of the controller. We show that bounded disturbances originate limited deviations from the target curve. Simulations and a real robot experiment exemplify and validate the developed theory.

- Global Identification of the Propeller Gains and Dynamic Parameters of Quadrotors from Flight Data

    Author: Six, Damien | IIT
    Author: Briot, S�bastien | LS2N
    Author: Erskine, Julian | Ecole Centrale De Nantes
    Author: Chriette, Abdelhamid | Ecole Centrale De Nantes
 
    keyword: Aerial Systems: Mechanics and Control; Dynamics; Underactuated Robots

    Abstract : Several methods can be applied to estimate the propeller thrust and torque coefficients and dynamics parameters of quadrotor UAVs. These parameters are necessary for many controllers that have been proposed for these vehicles. However, these methods require the use of specific test benches, which do not well simulate real flight conditions.<p>In this paper, a new method is introduced which allows the identification of the propeller coefficients and dynamic parameters of a quadrotor in a single procedure. It is based on a Total-Least-Square identification technique, does not require any specific test bench and needs only a measurement of the mass of the quadrotor and a recording of data from a flight that can be performed manually by an operator.</p><p>Because the symmetries of classic quadrotors limit the performance of the algorithm, an extension of the procedure is proposed. Two types of flights are then used: one with the initial quadrotor and a second flight with an additional payload on the vehicle that modifies the mass distribution. This new procedure, which is validated experimentally, increases the performance of the identification and allows an estimation of all the relevant dynamic parameters of the quadrotor near hovering conditions.

- Gemini: A Compact yet Efficient Bi-Copter UAV for Indoor Applications

    Author: Qin, Youming | The University of Hong Kong
    Author: Xu, Wei | University of Hong Kong
    Author: Lee, Adrian | University of California, Davis
    Author: Zhang, Fu | University of Hong Kong
 
    keyword: Aerial Systems: Mechanics and Control; Aerial Systems: Applications

    Abstract : Quad-copters are the premier platform for data collection tasks, yet their ability to collect data in indoor narrow spaces is severely compromised due to their huge size when carrying heavy sensors. In this paper, we study a bi-copter UAV configuration that has similar levels of versatility and improves the compactness or efficiency at the same time. Such an arrangement allows for the preservation of propeller size, meaning that we can effectively reduce the horizontal width of the UAV while still maintains the same payload capacity. Furthermore, pitch, roll and yaw control can also be achieved through mechanically simple means as well, increasing reliability and precision. We also found that the Gemini platform is the most power-efficient yet practical solution for indoor applications among all the twelve common UAV configurations. This paper will detail the entire process of creating the platform from picking the ideal propeller through aerodynamic analysis, system design, optimization, implementation, control, and real flight tests that demonstrate its ability to function seamlessly.

- Direct Force Feedback Control and Online Multi-Task Optimization for Aerial Manipulators

    Author: Nava, Gabriele | Istituto Italiano Di Tecnologia
    Author: Sabl�, Quentin | LAAS-CNRS
    Author: Tognon, Marco | LAAS-CNRS
    Author: Pucci, Daniele | Italian Institute of Technology
    Author: Franchi, Antonio | University of Twente
 
    keyword: Aerial Systems: Mechanics and Control; Force Control; Optimization and Optimal Control

    Abstract : In this paper we present an optimization-based method for controlling aerial manipulators in physical contact with the environment. The multi-task control problem, which includes hybrid force-motion tasks, energetic tasks, and position/postural tasks, is recast as a quadratic programming problem with equality and inequality constraints, which is solved online. Thanks to this method, the aerial platform can be exploited at its best to perform the multi-objective tasks, with tunable priorities, while hard constraints such as contact maintenance, friction cones, joint limits, maximum and minimum propeller speeds are all respected. An on-board force/torque sensor mounted at the end effector is used in the feedback loop in order to cope with model inaccuracies and reject external disturbances. Real experiments with a multi-rotor platform and a multi-DoF lightweight manipulator demonstrate the applicability and effectiveness of the proposed approach in the real world.

- Nonlinear Vector-Projection Control for Agile Fixed-Wing Unmanned Aerial Vehicles

    Author: Hernandez Ramirez, Juan Carlos | McGill University
    Author: Nahon, Meyer | McGill University
 
    keyword: Aerial Systems: Mechanics and Control; Motion Control; Control Architectures and Programming

    Abstract : Agile fixed-wing aircraft integrate the efficient, high-speed capabilities of conventional fixed-wing platforms with the extreme maneuverability of rotorcraft. This work presents a nonlinear control strategy that harnesses these capabilities to enable autonomous flight through aggressive, time-constrained, three-dimensional trajectories. The cascaded control structure consists of two parts; an inner attitude control loop developed on the Special Orthornormal group that avoids singularities commonly associated with other parametrizations, and an outer position control loop that jointly determines the thrust command and attitude references by implementing a novel vector-projection algorithm. The objective of the algorithm is to decouple roll from the reference attitude to ensure that thrust and lift forces can always be pointed such that position errors converge to zero. The proposed control system represents a single, unified solution that remains effective throughout the aircraft's flight envelope, including aerobatic operation. Controller performance is verified through simulations and experimental flight tests; results show the unified control scheme is capable of performing a wide range of operations that would normally require multiple, single-purpose controllers, and their associated switching logic.

- Adaptive Nonlinear Control of Fixed-Wing VTOL with Airflow Vector Sensing

    Author: Shi, Xichen | California Institute of Technology
    Author: Spieler, Patrick | Caltech
    Author: Tang, Ellande | California Institute of Technology
    Author: Lupu, Elena-Sorina | California Institute of Technology
    Author: Tokumaru, Phillip | AeroVironment, Inc
    Author: Chung, Soon-Jo | Caltech
 
    keyword: Aerial Systems: Mechanics and Control; Robust/Adaptive Control of Robotic Systems; Learning and Adaptive Systems

    Abstract : Fixed-wing vertical take-off and landing (VTOL) aircraft pose a unique control challenge that stems from complex aerodynamic interactions between wings and rotors. Thus, accurate estimation of external forces is indispensable for achieving high performance flight. In this paper, we present a composite adaptive nonlinear tracking controller for a fixed-wing VTOL. The method employs online adaptation of linear force models, and generates accurate estimation for wing and rotor forces in real-time based on information from a three-dimensional airflow sensor. The controller is implemented on a custom-built fixed-wing VTOL, which shows improved velocity tracking and force prediction during the transition stage from hover to forward flight, compared to baseline flight controllers.

- The Reconfigurable Aerial Robotic Chain: Modeling and Control

    Author: Nguyen, Dinh Huan | University of Nevada, Reno
    Author: Dang, Tung | University of Nevada, Reno
    Author: Alexis, Kostas | University of Nevada, Reno
 
    keyword: Aerial Systems: Mechanics and Control

    Abstract : This paper overviews the system design, modeling and control of the Aerial Robotic Chain. This new design corresponds to a reconfigurable robotic system of systems consisting of multilinked micro aerial vehicles that presents the ability to cross narrow sections, morph its shape, ferry significant payloads, offer the potential of distributed sensing and processing, and enable system extendability. We present the system dynamics for any number of connected aerial vehicles, followed by the controller design involving a model predictive position control loop combined with multiple parallel angular controllers on SO(3). Evaluation studies both in simulation and through experiments based on our ARC-Alpha prototype are depicted and involve coordinated maneuvering and shape configuration to cross narrow windows.

- Direct Acceleration Feedback Control of Quadrotor Aerial Vehicles

    Author: Hamandi, Mahmoud | INSA Toulouse
    Author: Tognon, Marco | LAAS-CNRS
    Author: Franchi, Antonio | University of Twente
 
    keyword: Aerial Systems: Mechanics and Control; Robust/Adaptive Control of Robotic Systems

    Abstract : In this paper we propose to control a quadrotor through direct acceleration feedback. The proposed method, while simple in form, alleviates the need for accurate estimation of platform parameters such as mass and propeller effectiveness. In order to use efficaciously the noisy acceleration measurements in direct feedback, we propose a novel regression-based filter that exploits the knowledge on the commanded propeller speeds, and extracts smooth platform acceleration with minimal delay. Our tests show that the controller exhibits a few millimeter error when performing real world tasks with fast changing mass and effectiveness, e.g., in pick and place operation and in turbulent conditions. Finally, we benchmark the direct acceleration controller against the PID strategy and show the clear advantage of using high-frequency and low-latency acceleration measurements directly in the control feedback, especially in the case of low frequency position measurements that are typical for real outdoor conditions.

- Trajectory Tracking Nonlinear Model Predictive Control for an Overactuated MAV

    Author: Brunner, Maximilian | ETH Zurich
    Author: Bodie, Karen | ETH Zurich
    Author: Kamel, Mina | Autonomous Systems Lab, ETH Zurich
    Author: Pantic, Michael | ETH Zurich
    Author: Zhang, Weixuan | ETH Zurich
    Author: Nieto, Juan | ETH Zurich
    Author: Siegwart, Roland | ETH Zurich
 
    keyword: Aerial Systems: Mechanics and Control; Optimization and Optimal Control

    Abstract : This work presents a method to control omnidirectional micro aerial vehicles (OMAVs) for the tracking of 6-DoF trajectories in free space. A rigid body model based approach is applied in a receding horizon fashion to generate optimal wrench commands that can be constrained to meet limits given by the mechanical design and actuators of the platform. Allocation of optimal actuator commands is performed in a separate step. A disturbance observer estimates forces and torques that may arise from unmodeled dynamics or external disturbances and fuses them into the optimization to achieve offset-free tracking. Experiments on a fully overactuated MAV show the tracking performance and compare it against a classical PD-based controller.

- Optimal Oscillation Damping Control of Cable-Suspended Aerial Manipulator with a Single IMU Sensor

    Author: Sarkisov, Yuri | Skolkovo Institute of Science and Technology
    Author: Kim, Min Jun | DLR
    Author: Coelho, Andre | German Aerospace Center (DLR)
    Author: Tsetserukou, Dzmitry | Skolkovo Institute of Science and Technology
    Author: Ott, Christian | German Aerospace Center (DLR)
    Author: Kondak, Konstantin | German Aerospace Center
 
    keyword: Aerial Systems: Mechanics and Control; Aerial Systems: Applications

    Abstract : This paper presents a design of oscillation damp- ing control for the cable-Suspended Aerial Manipulator (SAM). The SAM is modeled as a double pendulum, and it can generate a body wrench as a control action. The main challenge is the fact that there is only one onboard IMU sensor which does not provide full information on the system state. To overcome this difficulty, we design a controller motivated by a simplified SAM model. The proposed controller is very simple yet robust to model uncertainties. Moreover, we propose a gain tuning rule by formulating the proposed controller in the form of output feedback linear quadratic regulation problem. Consequently, it is possible to quickly dampen oscillations with minimal energy consumption. The proposed approach is validated through simulations and experiments.


- Upset Recovery Control for Quadrotors Subjected to a Complete Rotor Failure from Large Initial Disturbances

    Author: Sun, Sihao | Delft University of Technology
    Author: Baert, Matthias | Technical University Delft
    Author: Strack van Schijndel, Bram Adriaan | Delft University of Technology
    Author: de Visser, Coen | TU Delft
 
    keyword: Aerial Systems: Mechanics and Control; Robot Safety; Dynamics

    Abstract : This study has developed a fault-tolerant controller that is able to recover a quadrotor from arbitrary initial orientations and angular velocities, despite the complete failure of a rotor. This cascaded control method includes a position/altitude controller, an almost-global convergence attitude controller, and a control allocation method based on quadratic programming. As a major novelty, a constraint of undesirable angular velocity is derived and fused into the control allocator, which significantly improves the recovery performance. For validation, we have conducted a set of Monte-Carlo simulation to test the reliability of the proposed method of recovering the quadrotor from arbitrary initial attitude/rate conditions. In addition, real-life flight tests have been performed. The results demonstrate that the post-failure quadrotor can recover after being casually tossed into the air.

- Identification and Evaluation of a Force Model for Multirotor UAVs

    Author: Letalenet, Alexandre | Sorbonne Université
    Author: Morin, Pascal | UPMC
 
    keyword: Aerial Systems: Mechanics and Control; Dynamics

    Abstract : This paper proposes a model identification method and evaluation of a force model for multirotor UAVs. The model incorporates propellers' aerodynamics derived from momentum and blade element theories, as well as aerodynamics of the UAV's structure and actuation dynamics. A two-steps identification approach of the model parameters is proposed. The model is identified and evaluated from outdoor experiments with flight speeds exceeding 10m/s.

- Preliminary Study of an Aerial Manipulator with Elastic Suspension

    Author: Yigit, Arda | University of Strasbourg
    Author: Grappe, Gustave | University of Strasbourg
    Author: Cuvillon, Loic | University of Strasbourg
    Author: Durand, Sylvain | INSA Strasbourg &amp; ICube
    Author: Gangloff, Jacques | University of Strasbourg
 
    keyword: Aerial Systems: Mechanics and Control; Visual Servoing; Flexible Robots

    Abstract : This paper presents a preliminary study of an Aerial Manipulator suspended by a spring to a robotic carrier. The suspended aerial manipulator is actuated by six pairs of contra-rotating propellers generating a 6-DoF wrench. Simulations show path following results using a computed torque (feedback linearization) control strategy. Active vibration canceling is validated experimentally on a first prototype.

- Towards Low-Latency High-Bandwidth Control of Quadrotors Using Event Cameras

    Author: Sugimoto, Rika | University of Zurich
    Author: Gehrig, Mathias | University of Zurich
    Author: Brescianini, Dario | University of Zurich
    Author: Scaramuzza, Davide | University of Zurich
 
    keyword: Aerial Systems: Mechanics and Control; Visual Servoing; Sensor-based Control

    Abstract : Event cameras are a promising candidate to enable high speed vision-based control due to their low sensor latency and high temporal resolution. However, purely event-based feedback has yet to be used in the control of drones. In this work, a first step towards implementing low-latency high-bandwidth control of quadrotors using event cameras is taken. In particular, this paper addresses the problem of one-dimensional attitude tracking using a dualcopter platform equipped with an event camera. The event-based state estimation consists of a modified Hough transform algorithm combined with a Kalman filter that outputs the roll angle and angular velocity of the dualcopter relative to a horizon marked by a black-and-white disk. The estimated state is processed by a proportional-derivative attitude control law that computes the rotor thrusts required to track the desired attitude. The proposed attitude tracking scheme shows promising results of event-camera-driven closed loop control: the state estimator performs with an update rate of 1 kHz and a latency determined to be 12 milliseconds, enabling attitude tracking at speeds of over 1600 degrees per second.

- Perception-Constrained and Motor-Level Nonlinear MPC for Both Underactuated and Tilted-Propeller UAVs

    Author: Jacquet, Martin | LAAS, CNRS
    Author: Corsini, Gianluca | LAAS-CNRS
    Author: Bicego, Davide | LAAS-CNRS
    Author: Franchi, Antonio | University of Twente
 
    keyword: Aerial Systems: Mechanics and Control; Motion Control; Aerial Systems: Perception and Autonomy

    Abstract : In this paper, we present a Perception-constrained Nonlinear Model Predictive Control (NMPC) framework for the real-time control of multi-rotor aerial vehicles. Our formulation considers both constraints from a perceptive sensor and realistic actuator limitations that are the rotor minimum and maximum speeds and accelerations. The formulation is meant to be generic and considers a large range of multi-rotor platforms (such as underactuated quadrotors or tilted-propellers hexarotors) since it does not rely on differential flatness for the dynamical equation, and a broad range of sensors, such as cameras, lidars, etc... The perceptive constraints are expressed to maintain visibility of a feature point in the sensor's field of view, while performing a reference maneuver. We demonstrate both in simulation and real experiments that our framework is able to exploit the full capabilities of the multi-rotor to achieve the motion under the aforementioned constraints, and control in real-time the platform at a motor-torque level, to avoid the use of an intermediate unconstrained trajectory tracker.

- Coordinate-Free Dynamics and Differential Flatness of a Class of 6DOF Aerial Manipulators

    Author: Welde, Jake | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania
 
    keyword: Aerial Systems: Mechanics and Control; Motion Control of Manipulators; Dynamics

    Abstract : In this work, we derive a coordinate-free formulation of the coupled dynamics of a class of 6DOF aerial manipulators consisting of an underactuated quadrotor equipped with a 2DOF articulated manipulator, and demonstrate that the system is differentially flat with respect to the end effector pose. In particular, we require the center of mass of the entire system to be fixed in the end effector frame, suggesting a reasonable mechanical design criterion. We make use of an inertial decoupling transformation to demonstrate differential flatness, allowing us to plan dynamically feasible trajectories for the system in the space of the 6DOF pose of the end effector, which is ideal for achieving precise manipulator tasks. Simulation results validate the flatness-based planning methodology for our dynamic model, and its usefulness is demonstrated in a simulated aerial videography task.

## Autonomous Driving 

- Goal Directed Occupancy Prediction for Lane Following Actors

    Author: Kaniarasu, Poornima | Uber Advanced Technology Group (Jan 11, 2016 - Dec 18, 2019)
    Author: Haynes, Galen Clark | Uber ATG
    Author: Marchetti-Bowick, Micol | Uber Advanced Technologies Group
 
    keyword: Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation; Intelligent Transportation Systems

    Abstract : Predicting the possible future behaviors of vehicles that drive on shared roads is a crucial task for safe autonomous driving. Many existing approaches to this problem strive to distill all possible vehicle behaviors into a simplified set of high-level actions. However, these action categories do not suffice to describe the full range of maneuvers possible in the complex road networks we encounter in the real world. To combat this deficiency, we propose a new method that leverages the mapped road topology to reason over possible goals and predict the future spatial occupancy of dynamic road actors. We show that our approach is able to accurately predict future occupancy that remains consistent with the mapped lane geometry and naturally captures multi-modality based on the local scene context while also not suffering from the mode collapse problem observed in prior work.

- Intent-Aware Pedestrian Prediction for Adaptive Crowd Navigation

    Author: Katyal, Kapil | Johns Hopkins University Applied Physics Lab
    Author: Hager, Gregory | Johns Hopkins University
    Author: Huang, Chien-Ming | Johns Hopkins University
 
    keyword: Autonomous Vehicle Navigation; Human Detection and Tracking; Deep Learning in Robotics and Automation

    Abstract : Mobile robots capable of navigating seamlessly and safely in pedestrian rich environments promise to bring robotic assistance closer to our daily lives. In this paper we draw on insights of how humans move in crowded spaces to explore how to recognize pedestrian navigation intent, how to predict pedestrian motion and how a robot may adapt its navigation policy dynamically when facing unexpected human movements. Our approach is to develop algorithms that replicate this behavior. We experimentally demonstrate the effectiveness of our prediction algorithm using real-world pedestrian datasets and achieve comparable or better prediction accuracy compared to several state-of-the-art approaches. Moreover, we show that confidence of pedestrian prediction can be used to adjust the risk of a navigation policy adaptively to afford the most comfortable level as measured by the frequency of personal space violation in comparison with baselines. Furthermore, our adaptive navigation policy is able to reduce the number of collisions by 43% in the presence of novel pedestrian motion not seen during training.

- Brno Urban Dataset - the New Data for Self-Driving Agents and Mapping Tasks

    Author: Ligocki, Adam | Brno University of Technology
    Author: Jelinek, Ales | Brno University of Technology
    Author: Zalud, Ludek | Brno University of Technology
 
    keyword: Autonomous Vehicle Navigation; Range Sensing; Mapping

    Abstract : Autonomous driving is a dynamically growing field of research, where quality and amount of experimental data is critical. Although several rich datasets are available these days, the demands of researchers and technical possibilities are evolving. Through this paper, we bring a new dataset recorded in Brno - Czech Republic. It offers data from four WUXGA cameras, two 3D LiDARs, inertial measurement unit, infrared camera and especially differential RTK GNSS receiver with centimetre accuracy which, to the best knowledge of the     Authors, is not available from any other public dataset so far. In addition, all the data are precisely timestamped with sub-millisecond precision to allow wider range of applications. At the time of publishing of this paper, recordings of more than 350 km of rides in varying environment are shared at: https://github.com/RoboticsBUT/Brno-Urban-Dataset.

- Efficient Uncertainty-Aware Decision-Making for Automated Driving Using Guided Branching

    Author: Zhang, Lu | The Hong Kong University of Science and Technology
    Author: Ding, Wenchao | Hong Kong University of Science and Technology
    Author: Chen, Jing | Hong Kong University of Science and Technology
    Author: Shen, Shaojie | Hong Kong University of Science and Technology
 
    keyword: Autonomous Vehicle Navigation; Intelligent Transportation Systems

    Abstract : Decision-making in dense traffic scenarios is challenging for automated vehicles (AVs) due to potentially stochastic behaviors of other traffic participants and perception uncertainties (e.g., tracking noise and prediction errors, etc.). Although the partially observable Markov decision process (POMDP) provides a systematic way to incorporate these uncertainties, it quickly becomes computationally intractable when scaled to the real-world large-size problem. In this paper, we present an efficient uncertainty-aware decision-making (EUDM) framework, which generates long-term lateral and longitudinal behaviors in complex driving environments in real-time. The computation complexity is controlled to an appropriate level by two novel techniques, namely, the domain-specific closed-loop policy tree (DCP-Tree) structure and conditional focused branching (CFB) mechanism. The key idea is utilizing domain-specific expert knowledge to guide the branching in both action and intention space. The proposed framework is validated using both onboard sensing data captured by a real vehicle and an interactive multi-agent simulation platform. We also release the code of our framework to accommodate benchmarking.

- Imitative Reinforcement Learning Fusing Vision and Pure Pursuit for Self-Driving

    Author: Peng, Mingxing | Sun Yat-Sen University
    Author: Gong, Zhihao | Sun Yat-Sen University
    Author: Sun, Chen | University of Waterloo
    Author: Chen, Long | Sun Yat-Sen University
    Author: Cao, Dongpu | University of Waterloo
 
    keyword: Autonomous Vehicle Navigation; Autonomous Agents; Deep Learning in Robotics and Automation

    Abstract : Autonomous urban driving navigation is still an open problem and has ample room for improvement in unknown complex environments and terrible weather conditions. In this paper, we propose a two-stage framework, called IPP-RL, to handle these problems. IPP means an Imitation learning method fusing visual information with the additional steering angle calculated by Pure-Pursuit (PP) method, and RL means using Reinforcement Learning for further training. In our IPP model, the visual information captured by camera can be compensated by the calculated steering angle, thus it could perform well under bad weather conditions. However, imitation learning performance is limited by the driving data severely. Thus we use a reinforcement learning method-Deep Deterministic Policy Gradient (DDPG)-in the second stage training, which shares the learned weights from pretrained IPP model. In this way, our IPP-RL can lower the dependency of imitation learning on demonstration data and solve the problem of low exploration efficiency caused by randomly initialized weights in reinforcement learning. Moreover, we design a more reasonable reward function and use the n-step return to update the critic-network in DDPG. Our experiments on CARLA driving benchmark demonstrate that our IPP-RL is robust to lousy weather conditions and shows remarkable generalization capability in unknown environments on navigation task.

- Adversarial Appearance Learning in Augmented Cityscapes for Pedestrian Recognition in Autonomous Driving

    Author: Savkin, Artem | TUM
    Author: Lapotre, Thomas | TUM
    Author: Strauss, Kevin | Technical Universtity of Munich
    Author: Akbar, Uzair | Technical University of Munich
    Author: Tombari, Federico | Technische Universitét M�nchen
 
    keyword: Autonomous Vehicle Navigation; Virtual Reality and Interfaces; Computer Vision for Transportation

    Abstract : In the autonomous driving area synthetic data is crucial for cover specific traffic scenarios which autonomous vehicle must handle. This data commonly introduces domain gap between synthetic and real domains. In this paper we deploy data augmentation to generate custom traffic scenarios with VRUs in order to improve pedestrian recognition. We provide a pipeline for augmentation of the Cityscapes dataset with virtual pedestrians. In order to improve augmentation realism of the pipeline we reveal a novel generative network architecture for adversarial learning of the data-set lighting conditions. We also evaluate our approach on the tasks of semantic and instance segmentation.


- A*3D Dataset: Towards Autonomous Driving in Challenging Environments

    Author: Pham, Quang-Hieu | Singapore University of Technology and Design (SUTD)
    Author: Sevestre, Pierre | CentraleSup�lec
    Author: Pahwa, Ramanpreet Singh | Institute for Infocomm Research, Singapore
    Author: Zhan, Huijing | Institute for Infocomm Research
    Author: Mustafa, Armin | University of Surrey
    Author: Pang, Chun Ho | Institute for Infocomm Research, A*STAR Research Entities
    Author: Chen, Yuda | Institute for Infocomm Research, A*STAR Research Entities
    Author: Chandrasekhar, Vijay | Institute for Infocomm Research
    Author: Lin, Jie | Institute for Infocomm Research
 
    keyword: Object Detection, Segmentation and Categorization; Performance Evaluation and Benchmarking; Deep Learning in Robotics and Automation

    Abstract : With the increasing global popularity of self-driving cars, there is an immediate need for challenging real-world datasets for benchmarking and training various computer vision tasks such as 3D object detection. Existing datasets either represent simple scenarios or provide only day-time data. In this paper, we introduce a new challenging A*3D dataset which consists of RGB images and LiDAR data with a significant diversity of scene, time, and weather. The dataset consists of high-density images (&#8776;10 times more than the pioneering KITTI dataset), heavy occlusions, a large number of night-time frames (&#8776;3 times the nuScenes dataset), addressing the gaps in the existing datasets to push the boundaries of tasks in autonomous driving research to more challenging highly diverse environments. The dataset contains 39K frames, 7 classes, and 230K 3D object annotations. An extensive 3D object detection benchmark evaluation on the A*3D dataset for various attributes such as high density, day-time/night-time, gives interesting insights into the advantages and limitations of training and testing 3D object detection in real-world setting.

- SegVoxelNet: Exploring Semantic Context and Depth-Aware Features for 3D Vehicle Detection from Point Cloud

    Author: Yi, Hongwei | Peking University
    Author: Shi, Shaoshuai | The Chinese University of Hong Kong
    Author: Ding, Mingyu | The University of Hong Kong
    Author: Sun, Jiankai | The Chinese University of Hong Kong
    Author: Xu, Kui | Tsinghua University
    Author: Zhou, Hui | Sensetime Group Limited
    Author: Wang, Zhe | SenseTime Group Limited
    Author: Li, Sheng | Peking University
    Author: Wang, Guoping | Peking University
 
    keyword: Object Detection, Segmentation and Categorization; Autonomous Vehicle Navigation; Autonomous Agents

    Abstract : 3D vehicle detection based on point cloud is a challenging task in real-world applications such as autonomous driving. Despite significant progress has been made, we observe two aspects to be further improved. First, the semantic context information in LiDAR is seldom explored in previous works, which may help identify ambiguous vehicles. Second, the distribution of point cloud on vehicles varies continuously with increasing depths, which may not be well modeled by a single model. In this work, we propose a unified model SegVoxelNet to address the above two problems. A semantic context encoder is proposed to leverage the free-of-charge semantic segmentation masks in the bird eye view. Suspicious regions could be highlighted while noisy regions are suppressed by this module. To better deal with vehicles at different depths, a novel depth-aware head is designed to explicitly model the distribution differences and each part of the depth-aware head is made to focus on its own target detection range. Extensive experiments on the KITTI dataset show that the proposed method outperforms the state-of-the-art alternatives in both accuracy and efficiency with point cloud as input only.

- Fine-Grained Driving Behavior Prediction Via Context-Aware Multi-Task Inverse Reinforcement Learning
 
    Author: Nishi, Kentaro | Yahoo Japan Corporation
    Author: Shimosaka, Masamichi | Tokyo Institute of Technology
 
    keyword: Big Data in Robotics and Automation; Learning from Demonstration; Motion and Path Planning

    Abstract : Research on advanced driver assistance systems for reducing risks to vulnerable road users (VRUs) has recently gained popularity because the traffic accident reduction rate for VRUs is still small. Dealing with unexpected VRU movements on residential roads requires proficient acceleration and deceleration. Although fine-grained prediction of driving behavior through inverse reinforcement learning (IRL) has been reported with promising results in recent years, learning of a precise model fails when driving strategies vary with contextual factors, i.e., weather, time of day, road width, and traffic direction. In this work, we propose a novel multi-task IRL approach with a multilinear reward function to incorporate contextual information into the model. This approach can provide precise long-term prediction of fine-grained driving behavior while adjusting to context. Experimental results using actual driving data over 141 km with various contexts and roads confirm the success of this approach in terms of predicting defensive driving strategy even in unknown situations.

- How to Keep HD Maps for Automated Driving up to Date

    Author: Pannen, David | BMW Group
    Author: Liebner, Martin | BMW Group
    Author: Hempel, Wolfgang | BMW Group
    Author: Burgard, Wolfram | Toyota Research Institute
 
    keyword: Mapping; Intelligent Transportation Systems; SLAM

    Abstract : The current state of the art in automotive high definition digital (HD) map generation based on dedicated mapping vehicles cannot reliably keep these maps up to date because of the low traversal frequencies. Anonymized data collected from the fleet of vehicles that is already on the road provides a huge potential to outperform such state of the art solutions in robustness, safety and up-to-dateness of the map while achieving comparable quality. We thus present a solution based on crowdsourced data to (i) detect changes in the map independent of the type of change, (ii) automatically trigger map update jobs for parts of the map, and (iii) create and integrate map patches to keep the map always up to date. The developed solution provides a crowdsourced up to date HD map to make reliable prior information on lane markings and road edges available to automated driving functions.

- Binary DAD-Net: Binarized Driveable Area Detection Network for Autonomous Driving

    Author: Frickenstein, Alexander | BMW Group
    Author: Vemparala, Manoj Rohit | BMW Group
    Author: Mayr, Jakob | BMW Group
    Author: Nagaraja, Naveen Shankar | BMW Group
    Author: Unger, Christian | BMW Group
    Author: Tombari, Federico | Technische Universitét M�nchen
    Author: Stechele, Walter | Technical University of Munich
 
    keyword: Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation

    Abstract : Drivable area detection is a key component for various applications in the field of autonomous driving (AD), such as ground-plane detection, obstacle detection and maneuver planning. Additionally, bulky and over-parameterized networks can be easily forgone and replaced with smaller networks for faster inference on embedded systems. The drivable area detection, posed as a two class segmentation task, can be efficiently modeled with slim binary networks. This paper proposes a novel textit{binarized drivable area detection network (binary DAD-Net)}, which uses only binary weights and activations in the encoder, the bottleneck, and the decoder part. The latent space of the bottleneck is efficiently increased (x32 -&gt; x16 downsampling) through binary dilated convolutions, learning more complex features. Along with automatically generated training data, the binary DAD-Net outperforms state-of-the-art semantic segmentation networks on public datasets. In comparison to a full-precision model, our approach has a x14.3 reduced compute complexity on an FPGA and it requires only 0.9MB memory resources. Therefore, commodity SIMD-based AD-hardware is capable of accelerating the binary DAD-Net.

- Learning Robust Control Policies for End-To-End Autonomous Driving from Data-Driven Simulation

    Author: Amini, Alexander | Massachusetts Institute of Technology
    Author: Gilitschenski, Igor | Massachusetts Institute of Technology
    Author: Phillips, Jacob | Massachusetts Institute of Technology
    Author: Moseyko, Julia | Massachusetts Institute of Technology
    Author: Banerjee, Rohan | Massachusetts Institute of Technology
    Author: Karaman, Sertac | Massachusetts Institute of Technology
    Author: Rus, Daniela | MIT
 
    keyword: Deep Learning in Robotics and Automation; Visual Learning; Computer Vision for Transportation

    Abstract : In this work, we present a data-driven simulation and training engine capable of learning end-to-end autonomous vehicle control policies using only sparse rewards. By leveraging real, human-collected trajectories through an environment, we render novel training data that allows virtual agents to drive along a continuum of new local trajectories consistent with the road appearance and semantics, each with a different view of the scene. We demonstrate the ability of policies learned within our simulator to generalize to and navigate in previously unseen real-world roads, without access to any human control labels during training. Our results validate the learned policy onboard a full-scale autonomous vehicle, including in previously un-encountered scenarios, such as new roads and novel, complex, near-crash situations. Our methods are scalable, leverage reinforcement learning, and apply broadly to situations requiring effective perception and robust operation in the physical world.


- FG-GMM-Based Interactive Behavior Estimation for Autonomous Driving Vehicles in Ramp Merging Control

    Author: Lyu, Yiwei | Carnegie Mellon University
    Author: Dong, Chiyu | DiDi Labs
    Author: Dolan, John M. | Carnegie Mellon University
 
    keyword: Intelligent Transportation Systems

    Abstract : Interactive behavior is important for autonomous driving vehicles, especially for scenarios like ramp merging which require significant social interaction between autonomous driving vehicles and human-driven cars. This paper enhances our previous Probabilistic Graphical Model (PGM) merging control model for the interactive behavior of autonomous driving vehicles. To better estimate the interactive behavior for autonomous driving cars, a Factor Graph (FG) is used to describe the dependency among observations and estimate other cars' intentions. Real trajectories are used to approximate the model instead of human-designed models or cost functions. Forgetting factors and a Gaussian Mixture Model (GMM) are also applied in the intention estimation process for stabilization, interpolation and smoothness. The advantage of the factor graph is that the relationship between its nodes can be described by self-defined functions, instead of probabilistic relationships as in PGM, giving more flexibility. Continuity of GMM also provides higher accuracy than the previous discrete speed transition model. The proposed method enhances the overall performance of intention estimation, in terms of collision rate and average distance between cars after merging, which means it is safer and more efficient.

- Cooperative Perception and Localization for Cooperative Driving

    Author: Miller, Aaron | Carnegie Mellon University
    Author: Rim, Kyungzun | Carnegie Mellon University
    Author: Chopra, Parth | Honda R&amp;D Americas, Inc
    Author: Kelkar, Paritosh | Honda R&amp;D Americas, Inc
    Author: Likhachev, Maxim | Carnegie Mellon University
 
    keyword: Intelligent Transportation Systems; Sensor Fusion; Multi-Robot Systems

    Abstract : Fully autonomous vehicles are expected to share the road with less advanced vehicles for a significant period of time. Furthermore, an increasing number of vehicles on the road are equipped with a variety of low-fidelity sensors which provide some perception and localization data, but not at a high enough quality for full autonomy. In this paper, we develop a perception and localization system that allows a vehicle with low-fidelity sensors to incorporate high-fidelity observations from a vehicle in front of it, allowing both vehicles to operate with full autonomy. The resulting system generates perception and localization information that is both low-noise in regions covered by high-fidelity sensors and avoids false negatives in areas only observed by low-fidelity sensors, while dealing with latency and dropout of the communication link between the two vehicles. At its core, the system uses a set of Extended Kalman filters which incorporate observations from both vehicles' sensors and extrapolate them using information about the road geometry. The perception and localization algorithms are evaluated both in simulation and on real vehicles as part of a full cooperative driving system.

- Learning to Drive Off Road on Smooth Terrain in Unstructured Environments Using an On-Board Camera and Sparse Aerial Images

    Author: Manderson, Travis | McGill University
    Author: Wapnick, Stefan | McGill University
    Author: Meger, David Paul | McGill University
    Author: Dudek, Gregory | McGill University
 
    keyword: Learning and Adaptive Systems

    Abstract : We present a method for learning to drive on smooth terrain while simultaneously avoiding collisions in challenging off-road and unstructured outdoor environments using only visual inputs. Our approach applies a hybrid model-based and model-free reinforcement learning method that is entirely self-supervised in labeling terrain roughness and collisions using on-board sensors. Notably, we provide both first-person and overhead aerial image inputs to our model. We find that the fusion of these complementary inputs improves planning foresight and makes the model robust to visual obstructions. Our results show the ability to generalize to environments with plentiful vegetation, various types of rock, and sandy trails. During evaluation, our policy attained 90% smooth terrain traversal and reduced the proportion of rough terrain driven over by 6.1 times compared to a model using only first-person imagery. Video and project details can be found at www.cim.mcgill.ca/mrl/offroad_driving/

- RoadTrack: Realtime Tracking of Road Agents in Dense and Heterogeneous Environments

    Author: Chandra, Rohan | University of Maryland
    Author: Bhattacharya, Uttaran | UMD College Park
    Author: Randhavane, Tanmay | UNC
    Author: Bera, Aniket | University of Maryland
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Intelligent Transportation Systems; Visual Tracking; Agent-Based Systems

    Abstract : We present a realtime tracking algorithm, RoadTrack, to track heterogeneous road-agents in dense traffic videos. Our approach is designed for traffic scenarios that consist of different road-agents such as pedestrians, two-wheelers, cars, buses, etc. sharing the road. We use the tracking-by-detection approach where we track a road-agent by matching the appearance or bounding box region in the current frame with the predicted bounding box region propagated from the previous frame. Roadtrack uses a novel motion model called the Simultaneous Collision Avoidance and Interaction (SimCAI) model to predict the motion of road-agents by modeling collision avoidance and interactions between the road-agents for the next frame. We demonstrate the advantage of RoadTrack on a dataset of dense traffic videos and observe an accuracy of 75.8% on this dataset, outperforming prior state-of-the-art tracking algorithms by at least 5.2%. RoadTrack operates in realtime at approximately 30 fps and is at least 4 times faster than prior tracking algorithms on standard tracking datasets.

- Cooperative Control of Heterogeneous Connected Vehicle Platoons: An Adaptive Leader-Following Approach

    Author: Hu, Junyan | The University of Manchester
    Author: Bhowmick, Parijat | University of Manchester
    Author: Arvin, Farshad | University of Manchester
    Author: Lanzon, Alexander | The University of Manchester
    Author: Lennox, Barry | The University of Manchester
 
    keyword: Intelligent Transportation Systems; Robust/Adaptive Control of Robotic Systems; Motion Control

    Abstract : Automatic cruise control of a platoon of multiple connected vehicles in an automated highway system has drawn significant attention of the control practitioners over the past two decades due to its ability to reduce traffic congestion problems, improve traffic throughput and enhance safety of highway traffic. This paper proposes a two-layer distributed control scheme to maintain the string stability of a heterogeneous and connected vehicle platoon moving in one dimension with constant spacing policy assuming constant velocity of the lead vehicle. A feedback linearization tool is applied first to transform the nonlinear vehicle dynamics into a linear heterogeneous state-space model and then a distributed adaptive control protocol has been designed to keep equal inter-vehicular spacing between any consecutive vehicles while maintaining a desired longitudinal velocity of the entire platoon. The proposed scheme utilizes only the neighbouring state information (i.e. relative distance, velocity and acceleration) and the leader is not required to communicate with each and every one of the following vehicles directly since the interaction topology of the vehicle platoon is designed to have a spanning tree rooted at the leader. Simulation results demonstrated the effectiveness of the proposed platoon control scheme. Moreover, the practical feasibility of the scheme was validated by hardware experiments with real robots.

- Semantic Segmentation with Unsupervised Domain Adaptation under Varying Weather Conditions for Autonomous Vehicles

    Author: Erkent, Ozgur | Inria
    Author: Laugier, Christian | INRIA
 
    keyword: Intelligent Transportation Systems; Semantic Scene Understanding; Learning and Adaptive Systems

    Abstract : Semantic information provides a valuable source for scene understanding around autonomous vehicles in order to plan their actions and make decisions; however, varying weather conditions reduce the accuracy of the semantic segmentation. We propose a method to adapt to varying weather conditions without supervision, namely without labeled data. We update the parameters of a deep neural network (DNN) model that is pre-trained on the known weather condition (source domain) to adapt it to the new weather conditions (target domain) without forgetting the segmentation in the known weather condition. Furthermore, we don't require the labels from the source domain during adaptation training. The parameters of the DNN are optimized to reduce the distance between the distribution of the features from the images of old and new weather conditions. To measure this distance, we propose three alternatives: W-GAN, GAN and maximum-mean discrepancy (MMD). We evaluate our method on various datasets with varying weather conditions. The results show that the accuracy of the semantic segmentation is improved for varying conditions after adaptation with the proposed method.

- Deep Merging: Vehicle Merging Controller Based on Deep Reinforcement Learning with Embedding Network

    Author: Ippei, Nishitani | Toyota Motor Corporation
    Author: Yang, Hao | McMaster University
    Author: Guo, Rui | Toyota InfoTechnology Center USA
    Author: Keshavamurthy, Shalini | Toyota North America
    Author: Oguchi, Kentaro | Toyota InfoTechnology Center, USA
 
    keyword: Deep Learning in Robotics and Automation; Autonomous Vehicle Navigation; AI-Based Methods

    Abstract : Vehicles at highway merging sections must make lane changes to join the highway. This lane change can generate congestion. To reduce congestion, vehicles should merge so as not to affect traffic flow as much as possible. In our study, we propose a vehicle controller called Deep Merging that uses deep reinforcement learning to improve the merging efficiency of vehicles while considering the impact on traffic flow. The system uses the images of a merging section as input to output the target vehicle speed. Moreover, an embedding network for estimating the controlled vehicle speed is introduced to the deep reinforcement learning network architecture to improve the learning efficiency. In order to show the effectiveness of the proposed method, the merging behavior and traffic conditions in several situations are verified by experiments using a traffic simulator. Through these experiments, it is confirmed that the proposed method enables controlled vehicles to effectively merge without adversely affecting to the traffic flow.

- Radar As a Teacher: Weakly Supervised Vehicle Detection Using Radar Labels

    Author: Chadwick, Simon | University of Oxford
    Author: Newman, Paul | Oxford University
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization

    Abstract : It has been demonstrated that the performance of an object detector degrades when it is used outside the domain of the data used to train it. However, obtaining training data for a new domain can be time consuming and expensive. In this work we demonstrate how a radar can be used to generate plentiful (but noisy) training data for image-based vehicle detection. We then show that the performance of a detector trained using the noisy labels can be considerably improved through a combination of noise-aware training techniques and relabelling of the training data using a second viewpoint. In our experiments, using our proposed process improves average precision by more than 17 percentage points when training from scratch and 10 percentage points when fine-tuning a pre-trained model.

- Robust Lane Detection with Binary Integer Optimization

    Author: Brandes, Kathleen | Massachusetts Institute of Technology
    Author: Wang, Allen | Massachusetts Institute of Technology
    Author: Shah, Rushina | Massachusetts Institute of Technology
 
    keyword: Autonomous Vehicle Navigation; Motion and Path Planning; Optimization and Optimal Control

    Abstract : Formula Student Driverless (FSD) is a competition where student teams compete to build an autonomous racecar. The main dynamic event in FSD is trackdrive, where the racecar traverses an unknown track whose boundaries are demarcated by cones. One challenge of the event is to determine the track boundaries based on cone locations in the presence of false positive cone detections, sharp turns and uncertain cone color information while traversing the track. In this work, we present a binary integer optimization that encapsulates this problem, along with taking into account competition rule specifications, such as cone spacing and track width. This optimization routine is implemented in simulation, and on an autonomous electric racecar. We present our approach, and analyze its latency, accuracy, and robustness to uncertain cone detections. This approach is used on-vehicle to solve the real-time boundary generation problem during the competition.

- A Synchronization Approach for Achieving Cooperative Adaptive Cruise Control Based Non-Stop Intersection Passing

    Author: Liu, Zhe | The Chinese University of Hong Kong
    Author: Wei, Huanshu | Chinese University of Hong Kong
    Author: Hu, Hanjiang | Shanghai Jiao Tong University
    Author: Suo, Chuanzhe | The Chinese University of Hong Kong
    Author: Wang, Hesheng | Shanghai Jiao Tong University
    Author: Li, Haoang | The Chinese University of Hong Kong
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Intelligent Transportation Systems; Autonomous Vehicle Navigation

    Abstract : Cooperative adaptive cruise control (CACC) of intelligent vehicles contributes to improving cruise control performance, reducing traffic congestion, saving energy and increasing traffic flow capacity. In this paper, we resolve the CACC problem from the viewpoint of synchronization control, our main idea is to introduce the spatial-temporal synchronization mechanism into vehicle platoon control to achieve the robust CACC and to further realize the non-stop intersection control. Firstly, by introducing the cross-coupling based space synchronization mechanism, a distributed control algorithm is presented to achieve the single-lane CACC in the presence of vehicle-to-vehicle (V2V) communications, which enables autonomous vehicles to track the desired platoon trajectory while synchronizing their longitudinal velocities to keeping the expected inter-vehicle distance. Secondly, by designing the enter-time scheduling mechanism (temporal synchronization), a high-level intersection control strategy is proposed to command vehicles to form a virtual platoon to pass through the intersection without stopping. Thirdly, a Lyapunov-based time-domain stability analysis approach is presented. Compared with the traditional string stability based approach, the proposed approach guarantees the global asymptotical convergence of the proposed CACC system. Experiments in the small-scale simulated system demonstrate the effectiveness of the proposed approach.

- Self-Supervised Linear Motion Deblurring

    Author: Liu, Peidong | ETH Zurich
    Author: Janai, Joel | Max Planck Institute for Intelligent Systems, Autonomous Vision
    Author: Pollefeys, Marc | ETH Zurich
    Author: Sattler, Torsten | Chalmers University of Technology
    Author: Geiger, Andreas | Max Planck Institute for Intelligent Systems, Tübingen
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Other Robotic Applications; Computer Vision for Automation

    Abstract : Motion blurry images challenge many computer vision algorithms, e.g., feature detection, motion estimation, or object recognition. Deep convolutional neural networks are state-of-the-art for image deblurring. However, obtaining training data with corresponding sharp and blurry image pairs can be difficult. In this paper, we present a differentiable reblur model for self-supervised motion deblurring, which enables the network to learn from real-world blurry image sequences without relying on sharp images for supervision. Our key insight is that motion cues obtained from consecutive images yield sufficient information to inform the deblurring task. We therefore formulate deblurring as an inverse rendering problem, taking into account the physical image formation process: we first predict two deblurred images from which we estimate the corresponding optical flow. Using these predictions, we re-render the blurred images and minimize the difference with respect to the original blurry inputs. We use both synthetic and real dataset for experimental evaluations. Our experiments demonstrate that self-supervised single image deblurring is really feasible and leads to visually compelling results.

- Urban Driving with Conditional Imitation Learning

    Author: Hawke, Jeffrey | Wayve
    Author: Shen, Richard | Wayve
    Author: Gurau, Corina | Oxford University
    Author: Sharma, Siddharth | Wayve Technologies
    Author: Reda, Daniele | University of British Columbia // Wayve
    Author: Nikolov, Nikolay | Imperial College London
    Author: Mazur, Przemys&#322;aw | Wayve Technologies
    Author: Micklethwaite, Sean David | Wayve
    Author: Shah, Amar | Wayve
    Author: Kendall, Alex | University of Cambridge
 
    keyword: Deep Learning in Robotics and Automation; AI-Based Methods; Computer Vision for Transportation

    Abstract : Hand-crafting generalised decision-making rules for real-world urban autonomous driving is hard. Alternatively, learning behaviour from easy-to-collect human driving demonstrations is appealing. Prior work has studied imitation learning (IL) for autonomous driving with a number of limitations. Examples include only performing lane-following rather than following a user-defined route, only using a single camera view or heavily cropped frames lacking state observability, only lateral (steering) control, but not longitudinal (speed) control and a lack of interaction with traffic. Importantly, the majority of such systems have been primarily evaluated in simulation - a simple domain, which lacks real-world complexities. Motivated by these challenges, we focus on learning representations of semantics, geometry and motion with computer vision for IL from human driving demonstrations. As our main contribution, we present an end-to-end conditional imitation learning approach, combining both lateral and longitudinal control on a real vehicle for following urban routes with simple traffic. We address inherent dataset bias by data balancing, training our final policy on approximately 30 hours of demonstrations gathered over six months. We evaluate our method on an autonomous vehicle by driving 35km of novel routes in European urban streets.


- Simulation-Based Reinforcement Learning for Real-World Autonomous Driving

    Author: Osi&#324;ski, B&#322;a&#380;ej | University of Warsaw, Deepsense.ai
    Author: Jakubowski, Adam | Deepsense.ai
    Author: Zi&#281;cina, Pawe&#322; | Deepsense.ai
    Author: Mi&#322;o&#347;, Piotr | Institute of Mathematics of the Polish Academy of Sciences, Deep
    Author: Galias, Christopher | Jagiellonian University, Deepsense.ai
    Author: Homoceanu, Silviu | Volkswagen AG
    Author: Michalewski, Henryk | University of Warsaw
 
    keyword: Autonomous Vehicle Navigation; Visual-Based Navigation; Deep Learning in Robotics and Automation

    Abstract : We use reinforcement learning in simulation to obtain a driving system controlling a full-size real-world vehicle. The driving policy takes RGB images from a single camera and their semantic segmentation as input. We use mostly synthetic data, with labelled real-world data appearing only in the training of the segmentation network.<p>Using reinforcement learning in simulation and synthetic data is motivated by lowering costs and engineering effort.</p><p>In real-world experiments we confirm that we achieved successful sim-to-real policy transfer. Based on the extensive evaluation, we analyze how design decisions about perception, control, and training impact the real-world performance.

- Driving Style Encoder: Situational Reward Adaptation for General-Purpose Planning in Automated Driving

    Author: Rosbach, Sascha | Volkswagen AG
    Author: James, Vinit | Volkswagen AG
    Author: Grossjohann, Simon | Volkswagen AG
    Author: Homoceanu, Silviu | Volkswagen AG
    Author: Li, Xing | Volkswagen AG
    Author: Roth, Stefan | TU Darmstadt
 
    keyword: Learning from Demonstration; Deep Learning in Robotics and Automation; Motion and Path Planning

    Abstract :  General-purpose planning algorithms for automated driving combine mission, behavior, and local motion planning. Such planning algorithms map features of the environment and driving kinematics into complex reward functions. To achieve this, planning experts often rely on linear reward functions. The specification and tuning of these reward functions is a tedious process and requires significant experience. Moreover, a manually designed linear reward function does not generalize across different driving situations. In this work, we propose a deep learning approach based on inverse reinforcement learning that generates situation-dependent reward functions. Our neural network provides a mapping between features and actions of sampled driving policies of a model-predictive control-based planner and predicts reward functions for upcoming planning cycles. In our evaluation, we compare the driving style of reward functions predicted by our deep network against clustered and linear reward functions. Our proposed deep learning approach outperforms clustered linear reward functions and is at par with linear reward functions with a-priori knowledge about the situation.

- Analysis and Prediction of Pedestrian Crosswalk Behavior During Automated Vehicle Interactions

    Author: Jayaraman, Suresh Kumaar | University of Michigan
    Author: Tilbury, Dawn | University of Michigan
    Author: Yang, X. Jessie | University of Michigan
    Author: Pradhan, Anuj | University of Massachusetts Amherst
    Author: Robert, Lionel | University of Michigan
 
    keyword: Autonomous Vehicle Navigation; Human-Centered Robotics; Human Detection and Tracking

    Abstract : For safe navigation around pedestrians, automated vehicles (AVs) need to plan their motion by accurately predicting pedestrians' trajectories over long time horizons. Current approaches to AV motion planning around crosswalks predict only for short time horizons (1-2 s) and are based on data from pedestrian interactions with human-driven vehicles (HDVs). In this paper, we develop a hybrid systems model that uses pedestrians' gap acceptance behavior and constant velocity dynamics for long-term pedestrian trajectory prediction when interacting with AVs. Results demonstrate the applicability of the model for long-term (&gt; 5 s) pedestrian trajectory prediction at crosswalks. Further, we compared measures of pedestrian crossing behaviors in the immersive virtual environment (when interacting with AVs) to that in the real world (results of published studies of pedestrians interacting with HDVs), and found similarities between the two. These similarities demonstrate the applicability of the hybrid model of AV interactions developed from an immersive virtual environment (IVE) for real-world scenarios for both AVs and HDVs.

- The Oxford Radar RobotCar Dataset: A Radar Extension to the Oxford RobotCar Dataset

    Author: Barnes, Dan | University of Oxford
    Author: Gadd, Matthew | University of Oxford
    Author: Murcutt, Paul | Oxford University
    Author: Newman, Paul | Oxford University
    Author: Posner, Ingmar | Oxford University
 
    keyword: Big Data in Robotics and Automation; Autonomous Vehicle Navigation; SLAM

    Abstract : In this paper we present The Oxford Radar RobotCar Dataset, a new dataset for researching scene understanding using Millimetre-Wave FMCW scanning radar data.The target application is autonomous vehicles where this modality is robust to environmental conditions such as fog, rain, snow, or lens flare, which typically challenge other sensor modalities such as vision and LIDAR.<p>The data were gathered in January 2019 over thirty-two traversals of a central Oxford route spanning a total of 280km of urban driving. It encompasses a variety of weather, traffic, and lighting conditions. This 4.7TB dataset consists of over 240000 scans from a Navtech CTS350-X radar and 2.4 million scans from two Velodyne HDL-32E 3D LIDARs; along with six cameras, two 2D LIDARs, and a GPS/INS receiver. In addition we release ground truth optimised radar odometry to provide an additional impetus to research in this domain. </p><p>The full dataset is available for download at: ori.ox.ac.uk/datasets/radar-robotcar-dataset

- Multi-Modal Experts Network for Autonomous Driving

    Author: Fang, Shihong | New York University
    Author: Choromanska, Anna | New York University Tandon School of Engineering
 
    keyword: Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation

    Abstract : End-to-end learning from sensory data has shown promising results in autonomous driving. While employing many sensors enhances world perception and should lead to more robust and reliable behavior of autonomous vehicles, it is challenging to train and deploy such network and at least two problems are encountered in the considered setting. The first one is the increase of computational complexity with the number of sensing devices. The other is the phenomena of network overfitting to the simplest and most informative input. We address both challenges with a novel, carefully tailored multi-modal experts network architecture and propose a multi-stage training procedure. The network contains a gating mechanism, which selects the most relevant input at each inference time step using a mixed discrete-continuous policy. We demonstrate the plausibility of the proposed approach on our 1/6 scale truck equipped with three cameras and one LiDAR.

- Spatiotemporal Relationship Reasoning for Pedestrian Intent Prediction

    Author: Liu, Bingbin | Stanford University
    Author: Adeli, Ehsan | Stanford University
    Author: Cao, Zhangjie | Stanford University
    Author: Lee, Kuan-Hui | Toyota Research Institute
    Author: Shenoi, Abhijeet | Stanford University
    Author: Gaidon, Adrien | Toyota Research Institute
    Author: Niebles, Juan Carlos | Stanford University
 
    keyword: Autonomous Vehicle Navigation; Visual-Based Navigation; Computer Vision for Transportation

    Abstract : Reasoning over visual data is a desirable capability for robotics applications. Such reasoning enables forecasting of the next events or actions in videos. In recent years, various models have been developed based on convolution operations for prediction or forecasting, but they lack the ability to reason over spatiotemporal data and infer the relationships of objects in the scene. In this paper, we present a framework based on graph convolution to uncover the spatiotemporal relationships in the scene for reasoning about pedestrian intent. A scene graph is built on top of segmented object instances within and across video frames. Pedestrian intent (future action of crossing or not-crossing) is crucial information for autonomous vehicles to navigate safely and smoothly. We approach the problem of intent prediction from two different perspectives and anticipate the intention-to-cross. We introduce a new dataset designed specifically for autonomous-driving scenarios in areas with dense pedestrian populations: the Stanford-TRI Intent Prediction (STIP) dataset. Our experiments on STIP and another benchmark dataset show that our graph modeling framework is able to predict the intention-to-cross of the pedestrians with an accuracy of 79.10% on STIP and 79.28% on Joint Attention for Autonomous Driving (JAAD) dataset up to one second earlier than when the actual crossing happens. These results outperform baseline and previous work. Refer to stip.stanford.edu for dataset and code.

- TunerCar: A Superoptimization Toolchain for Autonomous Racing

    Author: O'Kelly, Matthew | University of Pennsylvania
    Author: Zheng, Hongrui | University of Pennsylvania
    Author: Jain, Achin | University of Pennsylvania
    Author: Auckley, Joseph | University of Pennsylvania
    Author: Luong, Kim | University of Pennsylvania
    Author: Mangharam, Rahul | University of Pennsylvania
 
    keyword: Autonomous Vehicle Navigation; Motion and Path Planning; Education Robotics

    Abstract : TunerCar is a toolchain that jointly optimizes racing strategy, planning methods, control algorithms, and vehicle parameters for an autonomous racecar. In this paper, we detail the target hardware, software, simulators, and systems infrastructure for this toolchain. Our methodology employs a parallel implementation of CMA-ES which enables simulations to proceed 6 times faster than real-world rollouts. We show our approach can reduce the lap times in autonomous racing, given a fixed computational budget. For all tested tracks, our method provides the lowest lap time, and relative improvements in lap time between 7-21%. We demonstrate improvements over a naive random search method with equivalent computational budget of over 15 seconds/lap, and improvements over expert solutions of over 2 seconds/lap. We further compare the performance of our method against hand-tuned solutions submitted by over 30 international teams, comprised of graduate students working in the field of autonomous vehicles. Finally, we discuss the effectiveness of utilizing an online planning mechanism to reduce the reality gap between our simulation and actual tests.

- Risk Assessment and Planning with Bidirectional Reachability for Autonomous Driving

    Author: Yu, Ming-Yuan | University of Michigan
    Author: Johnson-Roberson, Matthew | University of Michigan
    Author: Vasudevan, Ram | University of Michigan
 
    keyword: Autonomous Vehicle Navigation; Collision Avoidance; Motion and Path Planning

    Abstract : Risk assessment to quantify the danger associated with taking a certain action is critical to navigating safely through crowded urban environments during autonomous driving. Risk assessment and subsequent planning is usually done by first tracking and predicting trajectories of other agents, such as vehicles and pedestrians, and then choosing an action to avoid future collisions. However, few existing risk assessment algorithms handle occlusion and other sensory limitations effectively. One either assesses the risk in the worst-case scenario and thus makes the ego vehicle overly conservative, or predicts as many hidden agents as possible and thus makes the computation intensive. This paper explores the possibility of efficient risk assessment under occlusion via both forward and backward reachability. The proposed algorithm can not only identify the location of risk-inducing factors, but can also be used during motion planning. The proposed method is evaluated on various four-way highly occluded intersections with up to five other vehicles in the scene. Compared with other risk assessment algorithms, the proposed method shows better efficiency, meaning that the ego vehicle reaches the goal at a higher speed. In addition, it also lowers the median collision rate by 7.5� when compared to state of the art techniques.

- MapLite: Autonomous Intersection Navigation without a Detailed Prior Map

    Author: Ort, Teddy | Massachusetts Institute of Technology
    Author: Jatavallabhula, Krishna | Mila, Universite De Montreal
    Author: Banerjee, Rohan | Massachusetts Institute of Technology
    Author: Gottipati, Sai Krishna | MILA, University of Montreal
    Author: Bhatt, Dhaivat | IIIT-Hyderabad
    Author: Gilitschenski, Igor | Massachusetts Institute of Technology
    Author: Paull, Liam | Université De Montr�al
    Author: Rus, Daniela | MIT
 
    keyword: Autonomous Vehicle Navigation; Intelligent Transportation Systems; Field Robots

    Abstract : In this work, we present MapLite: a one-click autonomous navigation system capable of piloting a vehicle to an arbitrary desired destination point given only a sparse publicly available topometric map (from OpenStreetMap). The onboard sensors are used to segment the road region and register the topometric map in order to fuse the high-level navigation goals with a variational path planner in the vehicle frame. This enables the system to plan trajectories that correctly navigate road intersections without the use of an external localization system such as GPS or a detailed prior map. Since the topometric maps already exist for the vast majority of roads, this solution greatly increases the geographical scope for autonomous mobility solutions. We implement MapLite on a full-scale autonomous vehicle and exhaustively test it on over 15km of road including over 100 autonomous intersection traversals. We further extend these results through simulated testing to validate the system on complex road junction topologies such as traffic circles.

- Game Theoretic Decision Making Based on Real Sensor Data for Autonomous Vehicles' Maneuvers in High Traffic

    Author: Garzon Oviedo, Mario | Delft University of Technology
    Author: Spalanzani, Anne | INRIA / Univ. Grenoble Alpes
 
    keyword: Autonomous Vehicle Navigation; Intelligent Transportation Systems; Cognitive Human-Robot Interaction

    Abstract : This paper presents an approach for implementing game theoretic decision making in combination with realistic sensory data input so as to allow an autonomous vehicle to perform manoeuvrers, such as lane change or merge in high traffic scenarios. The main novelty of this work, is the use of realistic sensory data input to obtain the observations as input of an iterative multi-player game in a realistic simulator. The game model allows to anticipate reactions of additional vehicles to the movements of the ego-vehicle without using any specific coordination or vehicle-to-vehicle communication. Moreover, direct information from the simulator, such as position or speed of the vehicles is also avoided. <p>The solution of the game is based on cognitive hierarchy reasoning and it uses Monte Carlo reinforcement learning in order to obtain a near-optimal policy towards a specific goal. Moreover, the game proposed is capable of solving different situations using a single policy. The system has been successfully tested and compared with previous techniques using a realistic hybrid simulator, where the ego-vehicle and its sensors are simulated on a 3D simulator and the additional vehicles' behaviour is obtained from a traffic simulator.

- Driving in Dense Traffic with Model-Free Reinforcement Learning

    Author: Saxena, Dhruv Mauria | The Robotics Institute, Carnegie Mellon University
    Author: Bae, Sangjae | University of California, Berkeley
    Author: Nakhaei, Alireza | Honda Research Institute USA
    Author: Fujimura, Kikuo | Honda Research Institute
    Author: Likhachev, Maxim | Carnegie Mellon University
 
    keyword: Autonomous Vehicle Navigation; Autonomous Agents; Sensor-based Control

    Abstract : Traditional planning and control methods could fail to find a feasible trajectory for an autonomous vehicle to execute amongst dense traffic on roads. This is because the obstacle-free volume in spacetime is very small in these scenarios for the vehicle to drive through. However, that does not mean the task is infeasible since human drivers are known to be able to drive amongst dense traffic by leveraging the cooperativeness of other drivers to open a gap. The traditional methods fail to take into account the fact that the actions taken by an agent affect the behaviour of other vehicles on the road. In this work, we rely on the ability of deep reinforcement learning to implicitly model such interactions and learn a continuous control policy over the action space of an autonomous vehicle. The application we consider requires our agent to negotiate and open a gap in the road in order to successfully merge or change lanes. Our policy learns to repeatedly probe into the target road lane while trying to find a safe spot to move in to. We compare against two model-predictive control-based algorithms and show that our policy outperforms them in simulation. As part of this work, we introduce a benchmark for driving in dense traffic for use by the community.

- Enhancing Game-Theoretic Autonomous Car Racing Using Control Barrier Functions

    Author: Notomista, Gennaro | Georgia Institute of Technology
    Author: Wang, Mingyu | Stanford University
    Author: Schwager, Mac | Stanford University
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Autonomous Vehicle Navigation; Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems

    Abstract : In this paper, we consider a two-player racing game, where an autonomous ego vehicle has to be controlled to race against an opponent vehicle, which is either autonomous or human-driven. The approach to control the ego vehicle is based on a Sensitivity-ENhanced NAsh equilibrium seeking (SENNA) method, which uses an iterated best response algorithm in order to optimize for a trajectory in a two-car racing game. This method exploits the interactions between the ego and the opponent vehicle that take place through a collision avoidance constraint. This game-theoretic control method hinges on the ego vehicle having an accurate model and correct knowledge of the state of the opponent vehicle. However, when an accurate model for the opponent vehicle is not available, or the estimation of its state is corrupted by noise, the performance of the approach might be compromised. For this reason, we augment the SENNA algorithm by enforcing Permissive RObust SafeTy (PROST) conditions using control barrier functions. The objective is to successfully overtake or to remain in the front of the opponent vehicle, even when the information about the latter is not fully available. The successful synergy between SENNA and PROST�antithetical to the notable rivalry between the two namesake Formula 1 drivers�is demonstrated through extensive simulated experiments.




- CMTS: An Conditional Multiple Trajectory Synthesizer for Generating Safety-Critical Driving Scenarios

    Author: Ding, Wenhao | Carnegie Mellon University
    Author: Xu, Mengdi | Carnegie Mellon University
    Author: Zhao, Ding | Carnegie Mellon University
 
    keyword: Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation; Intelligent Transportation Systems

    Abstract : Naturalistic driving trajectory generation is crucial for the development of autonomous driving algorithms. However, most of the data is collected in collision-free scenarios leading to the sparsity of the safety-critical cases. When considering safety, testing algorithms in near-miss scenarios that rarely show up in off-the-shelf datasets and are costly to accumulate is a vital part of the evaluation. As a remedy, we propose a safety-critical data synthesizing framework based on variational Bayesian methods and term it as Conditional Multiple Trajectory Synthesizer (CMTS). We extend a generative model to connect safe and collision driving data by representing their distribution in the latent space and use conditional probability to adapt to different maps. Sampling from the mixed distribution enables us to synthesize the safety-critical data not shown in the safe or collision datasets. Experimental results demonstrate that the generated dataset covers many different realistic scenarios, especially the near-misses. We conclude that the use of data generated by CMTS can improve the accuracy of trajectory predictions and autonomous vehicle safety.

- LiDAR Inertial Odometry Aided Robust LiDAR Localization System in Changing City Scenes

    Author: Ding, Wendong | Baidu
    Author: Hou, Shenhua | Baidu
    Author: Gao, Hang | Baidu
    Author: Wan, Guowei | Baidu
    Author: Song, Shiyu | Baidu
 
    keyword: Autonomous Vehicle Navigation; Localization

    Abstract : Environmental fluctuations pose crucial challenges to a localization system in autonomous driving. We present a robust LiDAR localization system that maintains its kinematic estimation in changing urban scenarios by using a dead reckoning solution implemented through a LiDAR inertial odometry. Our localization framework jointly uses information from complementary modalities such as global matching and LiDAR inertial odometry to achieve accurate and smooth localization estimation. To improve the performance of the LiDAR odometry, we incorporate inertial and LiDAR intensity cues into an occupancy grid based LiDAR odometry to enhance frame-to-frame motion and matching estimation. Multi-resolution occupancy grid is implemented yielding a coarse-to-fine approach to balance the odometry's precision and computational requirement. To fuse both the odometry and global matching results, we formulate a MAP estimation problem in a pose graph fusion framework that can be efficiently solved. An effective environmental change detection method is proposed that allows us to know exactly when and what portion of the map requires an update. We comprehensively validate the effectiveness of the proposed approaches using both the Apollo-SouthBay dataset and our internal dataset. The results confirm that our efforts lead to a more robust and accurate localization system, especially in dynamically changing urban scenarios.

- Dynamic Interaction-Aware Scene Understanding for Reinforcement Learning in Autonomous Driving

    Author: Huegle, Maria | University of Freiburg
    Author: Kalweit, Gabriel | University of Freiburg
    Author: Werling, Moritz | Karlsruhe Institute of Technology
    Author: Boedecker, Joschka | University of Freiburg
 
    keyword: Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation; Learning and Adaptive Systems

    Abstract : The common pipeline in autonomous driving systems is highly modular and includes a perception component which extracts lists of surrounding objects and passes these lists to a high-level decision component. In this case, leveraging the benefits of deep reinforcement learning for high-level decision making requires special architectures to deal with multiple variable-length sequences of different object types, such as vehicles, lanes or traffic signs. At the same time, the architecture has to be able to cover interactions between traffic participants in order to find the optimal action to be taken. In this work, we propose the novel Deep Scenes architecture, that can learn complex interaction-aware scene representations based on extensions of either 1) Deep Sets or 2) Graph Convolutional Networks. We present the Graph-Q and DeepScene-Q off-policy reinforcement learning algorithms, both outperforming state-of-the-art methods in evaluations with the publicly available traffic simulator SUMO.

- Interacting Vehicle Trajectory Prediction with Convolutional Recurrent Neural Networks

    Author: Mukherjee, Saptarshi | Heriot Watt University
    Author: Wang, Sen | Edinburgh Centre for Robotics, Heriot-Watt University
    Author: Wallace, Andrew M. | Heriot-Watt University
 
    keyword: Autonomous Vehicle Navigation; Intelligent Transportation Systems

    Abstract : Anticipating the future trajectories of surrounding vehicles is a crucial and challenging task in path planning for autonomous vehicles. In this paper, we propose a novel Convolutional Long Short Term Memory (Conv-LSTM) based neural network architecture to predict future positions of cars using several seconds of historical driving observations. This consists of three parts: 1) Interaction Learning to capture the effect of surrounding cars, 2) Temporal Learning to identify the dependency on past movements and 3) Motion Learning to convert the extracted features from these two modules into future positions. To continuously achieve accurate prediction,we introduce a novel feedback based prediction scheme where the current predicted positions of each car are leveraged to update future instance prediction, encapsulating the surrounding cars' effect on future motion. Experiments on two public datasets demonstrate that the proposed network architecture and feedback based prediction technique in combined effort can match or outperform the state-of-the-art methods for long-term trajectory prediction.

- Navigation Command Matching for Vision-Based Autonomous Driving

    Author: Pan, Yuxin | Xi'an Jiaotong University
    Author: Xue, Jianru | Xi'an Jiaotong University
    Author: Zhang, Pengfei | Xi'an Jiaotong University
    Author: Ouyang, Wanli | The University of Sydney
    Author: Fang, Jianwu | Xian Jiaotong University
    Author: Chen, Xingyu | Laboratory of Visual Cognitive Computing and Intelligent Vehicle
 
    keyword: Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation; Visual-Based Navigation

    Abstract : Learning an optimal policy for autonomous driving task to confront with complex environment is a long-studied challenge. Imitative reinforcement learning is accepted as a promising approach to learn a robust driving policy through expert demonstrations and interactions with environments. However, this model utilizes non-smooth rewards, which have a negative impact on matching between navigation commands and trajectory (state-action pairs), and degrade the generalizability of an agent. Smooth rewards are crucial to discriminate actions generated from sub-optimal policy. In this paper, we propose a navigation command matching (NCM) model to address this issue. There are two key components in NCM, 1) a matching measurer produces smooth navigation rewards that measure matching between navigation commands and trajectory; 2) attention-guided agent performs actions given states where salient regions in RGB images (i.e. roadsides, lane markings and dynamic obstacles) are highlighted to amplify their influence on the final model. We obtain navigation rewards and store transitions to replay buffer after an episode, so NCM is able to discriminate actions generated from sub-optimal policy. Experiments on CARLA driving benchmark show our proposed NCM outperforms previous state-of-the-art models on various tasks in terms of the percentage of successfully completed episodes. Moreover, our model improves generalizability of the agent and obtains good performance even in unseen scenarios.

- GraphRQI: Classifying Driver Behaviors Using Graph Spectrums

    Author: Chandra, Rohan | University of Maryland
    Author: Bhattacharya, Uttaran | UMD College Park
    Author: Mittal, Trisha | University of Maryland, College Park
    Author: Li, Xiaoyu | Cornell University
    Author: Bera, Aniket | University of Maryland
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Autonomous Vehicle Navigation; Agent-Based Systems

    Abstract : We present a novel algorithm to identify driver behaviors from road-agent trajectories. Our approach assumes that the road agents exhibit a range of driving traits, such as aggressive or conservative driving. Moreover, these traits affect the trajectories of nearby road-agents as well as the interactions between road-agents. We represent these inter-agent interactions using unweighted and undirected traffic graphs. Our algorithm classifies the driver behavior using a supervised learning algorithm by reducing the computation to the spectral analysis of the traffic graph. Moreover, we present a novel eigenvalue algorithm (GraphRQI) to compute the spectrum efficiently. We provide theoretical guarantees for the running time complexity of GraphRQI and show that it is faster than previous methods by 2X. We evaluate the classification accuracy of our approach on traffic videos and autonomous driving datasets corresponding to urban traffic. In practice, our GraphRQI achieves an accuracy improvement of up to 25% over prior driver behavior classification algorithms. We also use our classification algorithm to predict the future trajectories of road-agents.

## Localization

- ROI-Cloud: A Key Region Extraction Method for LiDAR Odometry and Localization

    Author: Zhou, Zhibo | Shanghai Jiao Tong University
    Author: Yang, Ming | Shanghai Jiao Tong University
    Author: Wang, Chunxiang | Shanghai Jiaotong University
    Author: Wang, Bing | Shanghai Jiao Tong University
 
    keyword: Localization; SLAM; Intelligent Transportation Systems

    Abstract :     Abstract�We present a novel key region extraction method of point cloud, ROI-cloud, for LiDAR odometry and localization with autonomous robots. Traditional methods process massive point cloud data in every region within the field of view. In dense urban environments, however, processing redundant and dynamic regions of point cloud is time-consuming and harmful to the results of matching algorithms. In this paper, a voxelized cube set, ROI-cloud, is proposed to solve this problem by exclusively reserving the regions of interest for better point set registration and pose estimation. 3D space is firstly voxelized into weighted cubes. The key idea is to update their weights continually and extract cubes with high importance as key regions. By extracting geometrical features of a LiDAR scan, the importance of each cube is evaluated as a new measurement. With the help of on-board IMU/odometry data as well as new measurements, the weights of cubes are updated recursively through Bayes filtering. Thus, dynamic and redundant point cloud inside cubes with low importance are discarded by means of Monte Carlo sampling. Our method is validated on various datasets, and results indicate that the ROI-cloud improves the existing method in both accuracy and speed.

- To Learn or Not to Learn: Visual Localization from Essential Matrices

    Author: Zhou, Qunjie | Technical University of Munich
    Author: Sattler, Torsten | Chalmers University of Technology
    Author: Pollefeys, Marc | ETH Zurich
    Author: Leal-Taixe, Laura | Technical University of Munich
 
    keyword: Localization; Visual-Based Navigation

    Abstract : Visual localization is the problem of estimating a camera within a scene and a key technology for autonomous robots. State-of-the-art approaches for accurate visual localization use scene-specific representations, resulting in the overhead of constructing these models when applying the techniques to new scenes. Recently, learned approaches based on relative pose estimation have been proposed, carrying the promise of easily adapting to new scenes. However, they are currently significantly less accurate than state-of-the-art approaches. In this paper, we are interested in analyzing this behavior. To this end, we propose a novel framework for visual localization from relative poses. Using a classical feature-based approach within this framework, we show state-of-the-art performance. Replacing the classical approach with learned alternatives at various levels, we then identify the reasons for why deep learned approaches do not perform well. Based on our analysis, we make recommendations for future work.

- Hierarchical Multi-Process Fusion for Visual Place Recognition

    Author: Hausler, Stephen | Queensland University of Technology
    Author: Milford, Michael J | Queensland University of Technology
 
    keyword: Localization; Visual-Based Navigation

    Abstract : Combining multiple complementary techniques together has long been regarded as a way to improve performance. In visual localization, multi-sensor fusion, multi-process fusion of a single sensing modality, and even combinations of different localization techniques have been shown to result in improved performance. However, merely fusing together different localization techniques does not account for the varying performance characteristics of different localization techniques. In this paper we present a novel, hierarchical localization system that explicitly benefits from three varying characteristics of localization techniques: the distribution of their localization hypotheses, their appearance- and viewpoint-invariant properties, and the resulting differences in where in an environment each system works well and fails. We show how two techniques deployed hierarchically work better than in parallel fusion, how combining two different techniques works better than two levels of a single technique, even when the single technique has superior individual performance, and develop two and three-tier hierarchical structures that progressively improve localization performance. Finally, we develop a stacked hierarchical framework where localization hypotheses from techniques with complementary characteristics are concatenated at each layer, significantly improving retention of the correct hypothesis through to the final localization stage.

- Camera Tracking in Lighting Adaptable Maps of Indoor Environments

    Author: Caselitz, Tim | University of Freiburg
    Author: Krawez, Michael | University of Freiburg
    Author: Sundram, Jugesh | Toyota Motor Europe NV/SA
    Author: Van Loock, Mark | Toyota Motor Europe NV/SA
    Author: Burgard, Wolfram | Toyota Research Institute
 
    keyword: Localization; Mapping; RGB-D Perception

    Abstract : Tracking the pose of a camera is at the core of visual localization methods used in many applications. As the observations of a camera are inherently affected by lighting, it has always been a challenge for these methods to cope with varying lighting conditions. Thus far, this issue has mainly been approached with the intent to increase robustness by choosing lighting invariant map representations. In contrast, our work aims at explicitly exploiting lighting effects for camera tracking. To achieve this, we propose a lighting adaptable map representation for indoor environments that allows real-time rendering of the scene illuminated by an arbitrary subset of the lamps contained in the model. Our method for estimating the light setting from the current camera observation enables us to adapt the model according to the lighting conditions present in the scene. As a result, lighting effects like cast shadows do no longer act as disturbances that demand robustness but rather as beneficial features when matching observations against the map. We leverage these capabilities in a direct dense camera tracking approach and demonstrate its performance in real-world experiments in scenes with varying lighting conditions.

- Fast, Compact and Highly Scalable Visual Place Recognition through Sequence-Based Matching of Overloaded Representations

    Author: Garg, Sourav | Queensland University of Technology
    Author: Milford, Michael J | Queensland University of Technology
 
    keyword: Localization; Visual-Based Navigation; Recognition

    Abstract : Visual place recognition algorithms trade off three key characteristics: their storage footprint, their computational requirements, and their resultant performance, often expressed in terms of recall rate. Significant prior work has investigated highly compact place representations, sub-linear computational scaling and sub-linear storage scaling techniques, but have always involved a significant compromise in one or more of these regards, and have only been demonstrated on relatively small datasets. In this paper we present a novel place recognition system which enables for the first time the combination of ultra-compact place representations, near sub-linear storage scaling and extremely lightweight compute requirements. Our approach exploits the inherently sequential nature of much spatial data in the robotics domain and inverts the typical target criteria, through intentionally coarse scalar quantization-based hashing that leads to more collisions but is resolved by sequence-based matching. For the first time, we show how effective place recognition rates can be achieved on a new very large 10 million place dataset, requiring only 8 bytes of storage per place and 37K unitary operations to achieve over 50% recall for matching a sequence of 100 frames, where a conventional state-of-the-art approach both consumes 1300 times more compute and fails catastrophically.

- Vision-Based Multi-MAV Localization with Anonymous Relative Measurements Using Coupled Probabilistic Data Association Filter

    Author: Nguyen, Ty | University of Pennsylvania
    Author: Mohta, Kartik | University of Pennsylvania
    Author: Taylor, Camillo Jose | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania
 
    keyword: Localization; Aerial Systems: Perception and Autonomy; Swarms

    Abstract : We address the localization of robots in a multi-MAV system where external infrastructure like GPS or motion capture systems may not be available. Our approach lends itself to implementation on platforms with several constraints on size, weight, and power (SWaP). Particularly, our framework fuses the onboard VIO with the anonymous, visual-based robot-to-robot detection to estimate all robot poses in one common frame, addressing three main challenges: 1) the initial configuration of the robot team is unknown, 2) the data association between each vision-based detection and robot targets is unknown, and 3) the vision-based detection yields false negatives, false positives, inaccurate, and provides noisy bearing, distance measurements of other robots. Our approach extends the Coupled Probabilistic Data Association Filter~cite{cpdaf} to cope with nonlinear measurements. We demonstrate the superior performance of our approach over a simple VIO-based method in a simulation with the measurement models statistically modeled using the real experimental data. We also show how onboard sensing, estimation, and control can be used for formation flight.



- UrbanLoco: A Full Sensor Suite Dataset for Mapping and Localization in Urban Scenes

    Author: Wen, Weisong | Hong Kong Polytechnic University
    Author: Zhou, Yiyang | University of California, Berkeley
    Author: Zhang, Guohao | The Hong Kong Polytechnic University
    Author: Fahandezh-Saadi, Saman | University of California, Berkeley
    Author: Bai, Xiwei | Hong Kong Polytechnic University
    Author: Zhan, Wei | Univeristy of California, Berkeley
    Author: Tomizuka, Masayoshi | University of California
    Author: Hsu, Li-ta | Hong Kong Polytechnic University
 
    keyword: Localization; Mapping; Performance Evaluation and Benchmarking

    Abstract : Mapping and localization is a critical module of autonomous driving, and significant achievements have been reached in this field. Beyond Global Navigation Satellite System (GNSS), research in point cloud registration, visual feature matching, and inertia navigation has greatly enhanced the accuracy and robustness of mapping and localization in different scenarios. However, highly urbanized scenes are still challenging: LIDAR- and camera-based methods perform poorly with numerous dynamic objects; the GNSS-based solutions experience signal loss and multipath problems; the inertia measurement units (IMU) suffer from drifting. Unfortunately, current public datasets either do not adequately address this urban challenge or do not provide enough sensor information related to mapping and localization. Here we present UrbanLoco: a mapping/localization dataset collected in highly-urbanized environments with a full sensor-suite. The dataset includes 13 trajectories collected in San Francisco and Hong Kong, covering a total length of over 40 kilometers. Our dataset includes a wide variety of urban terrains: urban canyons, bridges, tunnels, sharp turns, etc. More importantly, our dataset includes information from LIDAR, cameras, IMU, and GNSS receivers. Now the dataset is publicly available through the link in the footnote.

- Map As the Hidden Sensor: Fast Odometry-Based Global Localization

    Author: Peng, Cheng | Univerisyt of Minnesota, Twin Cities
    Author: Weikersdorfer, David | Technische Universitét M�nchen
 
    keyword: Localization; SLAM; Sensor Fusion

    Abstract : Accurate and robust global localization is essential to robotics applications. We propose a novel global localization method that employs the map traversability as a hidden observation. The resulting map-corrected odometry localization is able to provide an accurate belief tensor of the robot state. Our method can be used for blind robots in dark or highly reflective areas. In contrast to odometry drift in long-term, our method using only odometry and the map converges in the long-term. Our method can also be integrated with other sensors to boost the localization performance. The algorithm does not have any initial state assumption and tracks all possible robot states at all times. Therefore, our method is global and is robust in the event of ambiguous observations. We parallel each step of our algorithm such that it can be performed in real-time (up to sim 300 Hz) using GPU. We validate our algorithm in different publicly available floor-plans and show that it is able to converge to the ground truth fast while being robust to ambiguities.

- Joint Human Pose Estimation and Stereo 3D Localization

    Author: Deng, Wenlong | EPFL
    Author: Bertoni, Lorenzo | EPFL
    Author: Kreiss, Sven | EPFL
    Author: Alahi, Alexandre | EPFL
 
    keyword: Localization

    Abstract : We present a new end-to-end trainable Neural Network architecture for stereo imaging that jointly locates and estimates human body poses in 3D, particularly suitable for autonomous vehicles. Our contribution, referred to as Part Spatial Field (PSF), defines a 2D pose for each human in a stereo pair of images and uses a correlation layer with a composite field to associate each left-right pair of joints. Finally, we show that we can train our method with synthetic data only and test it on real-world images (textit{i.e.}, our method is domain invariant). We achieve state-of-the-art results for the 3D localization task on the challenging real-world KITTI dataset.

- Self-Supervised Deep Pose Corrections for Robust Visual Odometry

    Author: Wagstaff, Brandon | University of Toronto
    Author: Peretroukhin, Valentin | University of Toronto
    Author: Kelly, Jonathan | University of Toronto
 
    keyword: Localization; Deep Learning in Robotics and Automation; Visual-Based Navigation

    Abstract : We present a self-supervised deep pose correction (DPC) network that applies pose corrections to a visual odometry estimator to improve its accuracy. Instead of regressing inter-frame pose changes directly, we build on prior work that uses data-driven learning to regress pose corrections that account for systematic errors due to violations of modelling assumptions. Our self-supervised formulation removes any requirement for six-degrees-of-freedom ground truth and, in contrast to expectations, often improves overall navigation accuracy compared to a supervised approach. Through extensive experiments, we show that our self-supervised DPC network can significantly enhance the performance of classical monocular and stereo odometry estimators and substantially out-performs state-of-the-art learning-only approaches.

- Ultra-High-Accuracy Visual Marker for Indoor Precise Positioning

    Author: Tanaka, Hideyuki | National Institute of AIST
 
    keyword: Localization; Automation Technologies for Smart Cities; Service Robots

    Abstract : Indoor positioning technology is essential for indoor mobile robots and drones. However, there has never been a general-purpose technology or infrastructure that enables indoor positioning with an accuracy of less than 10 cm. We have developed an attitude measurement method using multiple dynamic moires with a lenticular lens and developed an ultra-high-accuracy visual marker with an attitude estimation error of less than 0.1 deg. We also developed a calculation method that minimizes the marker position error by reminimizing reprojection error using its good attitude accuracy. We proved that accurate local positioning with a position error of about 1 cm in a marker coordinate system is possible even when a marker is shot from a distance of 10 m. In addition, a demonstration test was performed in a public space, and it was shown that high-accuracy global positioning with a position error of about 10 cm is possible.

- Accurate Position Tracking with a Single UWB Anchor

    Author: Cao, Yanjun | École Polytechnique De Montr�al (Université De Montr�al)
    Author: Yang, Chenhao | University of Tuebingen
    Author: Li, Rui | Institute of Automation, Chinese Academy of Sciences
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
    Author: Beltrame, Giovanni | Ecole Polytechnique De Montreal
 
    keyword: Localization; Sensor Fusion

    Abstract :  Accurate localization and tracking are a fundamental requirement for robotic applications. Localization systems like GPS, optical tracking, simultaneous localization and mapping (SLAM) are used for daily life activities, research, and commercial applications.	Ultra-wideband (UWB) technology provides another venue to accurately localize devices both indoors and outdoors. In this paper, we study a localization solution with a single UWB anchor, instead of the traditional multi-anchor setup. Besides the challenge of a single UWB ranging source, the only other sensor we require is a low-cost 9 DoF inertial measurement unit (IMU). Under such a configuration, we propose continuous monitoring of UWB range changes to estimate the robot speed when moving on a line. Combining speed estimation with orientation estimation from the IMU sensor, the system becomes temporally observable. We use an Extended Kalman Filter (EKF) to estimate the pose of a robot. With our solution, we can effectively correct the accumulated error and maintain accurate tracking of a moving robot.

- Association-Free Multilateration Based on Times of Arrival

    Author: Frisch, Daniel | Karlsruhe Institute of Technology (KIT)
    Author: Hanebeck, Uwe D. | Karlsruhe Institute of Technology (KIT)
 
    keyword: Localization; Surveillance Systems; Sensor Fusion

    Abstract : Multilateration systems reconstruct the location of a target that transmits electromagnetic or acoustic signals. The employed measurements for localization are the times of arrival (TOAs) of the transmitted signal, measured by a number of spatially distributed receivers at known positions. We present a novel multilateration algorithm to localize multiple targets that transmit indistinguishable signals at unknown times. That is, each receiver measures merely a set of TOAs with no association to the targets. Our method does not need any prior information. Therefore, it can provide uncorrelated, static measurements to be introduced into a separate tracker subsequently, or an initialization routine for multi target trackers.

- Adversarial Feature Disentanglement for Place Recognition across Changing Appearance

    Author: Tang, Li | Zhejiang University
    Author: Wang, Yue | Zhejiang University
    Author: Luo, Qianhui | Zhejiang University
    Author: Ding, Xiaqing | Zhejiang University
    Author: Xiong, Rong | Zhejiang University
 
    keyword: Localization; Deep Learning in Robotics and Automation; SLAM

    Abstract :  When robots move autonomously for long-term, varied appearance such as the transition from day to night and seasonal variation brings challenges to visual place recognition. Defining an appearance condition (e.g. a season, a kind of weather) as a <i>domain</i>, we consider that the desired representation for place recognition (i) should be domain-unrelated so that images from different time can be matched regardless of varied appearance, (ii) should be learned in a self-supervised manner without the need of massive manually labeled data, and (iii) should be able to train among multiple domains in one model to keep limited model complexity. This paper sets to find domain-unrelated features across extremely changing appearance, which can be used as image descriptors to match between images collected at different conditions. We propose to use the adversarial network to disentangle domain-unrelated and domain-related features, which are named <i>place</i> and <i>appearance</i> features respectively. During training, only domain information is needed without requiring manually aligned image sequences. Experiments demonstrated that our method can disentangle place and appearance features in both toy case and images from the real world, and the place features are qualified in place recognition tasks under different appearance conditions. The proposed network is also adaptable to multiple domains without increasing model capacity and shows favorable generalization.

- A Fast and Accurate Solution for Pose Estimation from 3D Correspondences

    Author: Zhou, Lipu | Carnegie Mellon University
    Author: Wang, Shengze | Carnegie Mellon University
    Author: Kaess, Michael | Carnegie Mellon University
 
    keyword: Localization; SLAM; Mapping

    Abstract : Estimating pose from given 3D correspondences, including point-to-point, point-to-line and point-to-plane correspondences, is a fundamental task in computer vision with many applications. We present a fast and accurate solution for the least-squares problem of this task. Previous works mainly focus on studying the way to find the global minimizer of the leastsquares problem. However, existing works that show the ability to achieve the global minimizer are still unsuitable for real-time applications. Furthermore, as one of contributions of this paper, we prove that there exist ambiguous configurations for any number of lines and planes. These configurations have several solutions in theory, which makes the correct solution may come from a local minimizer when the data are with noise. Previous works based on convex optimization which is unable to find local minimizers do not work in the ambiguous configuration. Our algorithm is efficient and able to reveal local minimizers. We employ the Cayley-Gibbs-Rodriguez (CGR) parameterization of the rotation to derive a general rational cost for the three cases of 3D correspondences. The main contribution of this paper is to solve the first-order optimality conditions of the least-squares problem, which are of a complicated rational form. The central idea of our algorithm is to introduce some intermediate unknowns to simplify the problem. Extensive experimental results show that our algorithm is more stable than previous algorithms when

- Ground Texture Based Localization Using Compact Binary Descriptors

    Author: Schmid, Jan Fabian | Robert Bosch GmbH; Goethe University Frankfurt
    Author: Simon, Stephan F. | Robert Bosch GmbH
    Author: Mester, Rudolf | NTNU Trondheim
 
    keyword: Localization; Mapping; SLAM

    Abstract : Ground texture based localization is a promising approach to achieve high-accuracy positioning of vehicles. We present a self-contained method that can be used for global localization as well as for subsequent local localization updates, i.e. it allows a robot to localize without any knowledge of its current whereabouts, but it can also take advantage of a prior pose estimate to reduce computation time significantly. Our method is based on a novel matching strategy, which we call identity matching, that is based on compact binary feature descriptors. Identity matching treats pairs of features as matches only if their descriptors are identical. While other methods for global localization are faster to compute, our method reaches higher localization success rates, and can switch to local localization after the initial localization.

- Reliable Data Association for Feature-Based Vehicle Localization Using Geometric Hashing Methods

    Author: Hofstetter, Isabell | Mercedes-Benz AG
    Author: Sprunk, Michael | Daimler AG
    Author: Ries, Florian | Mercedes-Benz AG
    Author: Haueis, Martin | Mercedes Benz AG
 
    keyword: Localization; Recognition; Autonomous Vehicle Navigation

    Abstract : Reliable data association represents a main challenge of feature-based vehicle localization and is the key to integrity of localization. Independent of the type of features used, incorrect associations between detected and mapped features will provide erroneous position estimates. Only if the uniqueness of a local environment is represented by the features that are stored in the map, reliable and safe localization can be guaranteed. In this work, a new approach based on Geometric Hashing is introduced to the field of data association for feature-based vehicle localization. Without any Information on a prior position, the proposed method allows to efficiently search large map regions for plausible Feature associations. Therefore, odometry and GNSS-based inputs can be neglected, which eliminates the risk of error propagation and enables safe localization. The approach is demonstrated on approximately 10min of data recorded in an urban scenario. Cylindrical objects without distinctive descriptors, which were extracted from LiDAR data, serve as localization features. Experimental results both demonstrate the feasibility as well as limitations of the approach.

- Vehicle Localization Based on Visual Lane Marking and Topological Map Matching

    Author: Asghar, Rabbia | Inria
    Author: Garzon Oviedo, Mario | Inria
    Author: Lussereau, Jerome | INRIA
    Author: Laugier, Christian | INRIA
 
    keyword: Localization; Sensor Fusion; Autonomous Vehicle Navigation

    Abstract : Accurate and reliable localization is crucial to autonomous vehicle navigation and driver assistance systems. This paper presents a novel approach for online vehicle localization in a digital map. Two distinct map matching algorithms are proposed: i) Iterative Closest Point (ICP) based lane level map matching is performed with visual lane tracker and grid map ii) decision-rule based approach is used to perform topological map matching. Results of both the map matching algorithms are fused together with GPS and dead reckoning using Extended Kalman Filter to estimate vehicle's pose relative to the map. The proposed approach has been validated on real life conditions on an equipped vehicle. Detailed analysis of the experimental results show improved localization using the two aforementioned map matching algorithms.

- RISE: A Novel Indoor Visual Place Recogniser

    Author: Sanchez Belenguer, Carlos | Joint Research Centre (JRC) - European Commission
    Author: Wolfart, Erik | European Commission, Joint Research Centre (JRC), Institute For
    Author: Sequeira, Vitor | Joint Research Centre
 
    keyword: Localization; Deep Learning in Robotics and Automation; Recognition

    Abstract : This paper presents a new technique to solve the Indoor Visual Place Recognition problem from the Deep Learning perspective. It consists on an image retrieval approach supported by a novel image similarity metric. Our work uses a 3D laser sensor mounted on a backpack with a calibrated spherical camera i) to generate the data for training the deep neural network and ii) to build a database of geo-referenced images for an environment. The data collection stage is fully automatic and requires no user intervention for labelling. Thanks to the 3D laser measurements and the spherical panoramas, we can efficiently survey large indoor areas in a very short time. The underlying 3D data associated to the map allows us to define the similarity between two training images as the geometric overlap between the observed pixels. We exploit this similarity metric to effectively train a CNN that maps images into compact embeddings. The goal of the training is to ensure that the L2 distance between the embeddings associated to two images is small when they are observing the same place and large when they are observing different places. After the training, similarities between a query image and the geo-referenced images in the database are efficiently retrieved by performing a nearest neighbour search in the embeddings space.

- Beyond Photometric Consistency: Gradient-Based Dissimilarity for Improving Visual Odometry and Stereo Matching

    Author: Quenzel, Jan | University of Bonn
    Author: Rosu, Radu Alexandru | University of Bonn
    Author: L�be, Thomas | University of Bonn
    Author: Stachniss, Cyrill | University of Bonn
    Author: Behnke, Sven | University of Bonn
 
    keyword: Localization; Visual-Based Navigation

    Abstract : Pose estimation and map building are central ingredients of autonomous robots and typically rely on the registration of sensor data. In this paper, we investigate a new metric for registering images that builds upon on the idea of the photometric error. Our approach combines a gradient orientation-based metric with a magnitude-dependent scaling term. We integrate both into stereo estimation as well as visual odometry systems and show clear benefits for typical disparity and direct image registration tasks when using our proposed metric. Our experimental evaluation indicates that our metric leads to more robust and more accurate estimates of the scene depth as well as camera trajectory. Thus, the metric improves camera pose estimation and in turn the mapping capabilities of mobile robots. We believe that a series of existing visual odometry and visual SLAM systems can benefit from the findings reported in this paper.

- ICS: Incremental Constrained Smoothing for State Estimation

    Author: Sodhi, Paloma | Carnegie Mellon University
    Author: Choudhury, Sanjiban | University of Washington
    Author: Mangelson, Joshua | Brigham Young University
    Author: Kaess, Michael | Carnegie Mellon University
 
    keyword: Localization; SLAM; Optimization and Optimal Control

    Abstract : A robot operating in the world constantly receives information about its environment in the form of new measurements at every time step. Smoothing-based estimation methods seek to optimize for the most likely robot state estimate using all measurements up till the current time step. Existing methods solve for this smoothing objective efficiently by framing the problem as that of incremental unconstrained optimization. However, in many cases observed measurements and knowledge of the environment is better modeled as hard constraints derived from real-world physics or dynamics. A key challenge is that the new optimality conditions introduced by the hard constraints break the matrix structure needed for incremental factorization in these incremental optimization methods.<p>Our key insight is that if we leverage primal-dual methods, we can recover a matrix structure amenable to incremental factorization. We propose a framework ICS that combines a primal-dual method like the Augmented Lagrangian with an incremental Gauss Newton approach that reuses previously computed matrix factorizations. We evaluate ICS on a set of simulated and real-world problems involving equality constraints like object contact and inequality constraints like collision avoidance.

- Drone-Aided Localization in LoRa IoT Networks

    Author: Delafontaine, Victor Pierre Guy | Laboratory of Intelligent Systems (LIS), École Polytechnique F�d
    Author: Schiano, Fabrizio | Ecole Polytechnique Federale De Lausanne, EPFL
    Author: Cocco, Giuseppe | Pompeu Fabra University
    Author: Rusu, Alexandru | Swisscom
    Author: Floreano, Dario | Ecole Polytechnique Federal, Lausanne
 
    keyword: Localization; Aerial Systems: Perception and Autonomy; Sensor Networks

    Abstract : Besides being part of the Internet of Things (IoT), drones can play a relevant role in it as enablers. The 3D mobility of UAVs can be exploited to improve node localization in IoT networks for, e.g., search and rescue or goods localization and tracking. One of the widespread IoT communication technologies is Long Range Wide Area Network (LoRaWAN), which allows achieving long communication distances with low power. In this work, we present a drone-aided localization system for LoRa networks in which a UAV is used to improve the estimation of a node's location initially provided by the network. We characterize the relevant parameters of the communication system and use them to develop and test a search algorithm in a realistic simulated scenario. We then move to the full implementation of a real system in which a drone is seamlessly integrated into Swisscom's LoRa network. The drone coordinates with the network with a two-way exchange of information which results in an accurate and fully autonomous localization system. The results obtained in our field tests show a ten-fold improvement in localization precision with respect to the estimation provided by the fixed network. Up to our knowledge, this is the first time a UAV is successfully integrated in a LoRa network to improve its localization accuracy.

- A Fast and Practical Method of Indoor Localization for Resource-Constrained Devices with Limited Sensing

    Author: Wietrzykowski, Jan | Poznan University of Technology
    Author: Skrzypczynski, Piotr | Poznan University of Technology
 
    keyword: Localization; Probability and Statistical Methods; Sensor Fusion

    Abstract : We describe and experimentally demonstrate a practical method for indoor localization using measurements obtained from resource-constrained devices with limited sensing capabilities. We focus on handheld/mobile devices but the method can be useful for a variety of wearable devices. Our system works with sparse WiFi or image-based measurements, avoiding laborious site surveying for dense signal maps and runs in real-time. It uses Conditional Random Fields to infer the most probable sequence of agent positions from a known floor plan, dead reckoning and sparse absolute position estimates. Our solution leverages known topology of the environment by pre-computing allowed motion sequences of an agent, which are then used to constraint the motion inferred from the sensory data. The system is evaluated in a typical office building, demonstrating good accuracy and robustness to sparse, low-quality measurements.


- GN-Net: The Gauss-Newton Loss for Multi-Weather Relocalization

    Author: von Stumberg, Lukas | Technische Universitét M�nchen
    Author: Wenzel, Patrick | Technical University of Munich
    Author: Khan, Qadeer | Technical University of Munich
    Author: Cremers, Daniel | Technical University of Munich
 
    keyword: Localization; Visual Learning; SLAM

    Abstract : Direct SLAM methods have shown exceptional performance on odometry tasks. However, they are susceptible to dynamic lighting and weather changes while also suffering from a bad initialization on large baselines. To overcome this, we propose GN-Net: a network optimized with the novel Gauss Newton loss for training weather invariant deep features, tailored for direct image alignment. Our network can be trained with pixel correspondences between images taken from different sequences. Experiments on both simulated and real-world datasets demonstrate that our approach is more robust against bad initialization, variations in day-time, and weather changes thereby outperforming state-of-the-art direct and indirect methods. Furthermore, we release an evaluation benchmark for relocalization tracking against different types of weather. Our benchmark is available at https://vision.in.tum.de/gn-net.

- A Data-Driven Motion Prior for Continuous-Time Trajectory Estimation on SE(3)

    Author: Wong, Jeremy Nathan | University of Toronto
    Author: Yoon, David Juny | University of Toronto
    Author: Schoellig, Angela P. | University of Toronto
    Author: Barfoot, Timothy | University of Toronto
 
    keyword: Localization; SLAM

    Abstract : Simultaneous trajectory estimation and mapping (STEAM) is a method for continuous-time trajectory estimation in which the trajectory is represented as a Gaussian Process (GP). Previous formulations of STEAM used a GP prior that assumed either white-noise-on-acceleration (WNOA) or white-noise-on-jerk (WNOJ). However, previous work did not provide a principled way to choose the continuous-time motion prior or its parameters on a real robotic system. This paper derives a novel data-driven motion prior where ground truth trajectories of a moving robot are used to train a motion prior that better represents the robot's motion. In this approach, we use a prior where latent accelerations are represented as a GP with a Matern covariance function and draw a connection to the Singer acceleration model. We then formulate a variation of STEAM using this new prior. We train the WNOA, WNOJ, and our new latent-force prior and evaluate their performance in the context of both lidar localization and lidar odometry of a car driving along a 20 km route, where we show improved state estimates compared to the two previous formulations.

- Estimation with Fast Feature Selection in Robot Visual Navigation
 
    Author: Mousavi, Hossein K. | Lehigh University
    Author: Motee, Nader | Lehigh Universitty
 
    keyword: Localization; Visual-Based Navigation; Autonomous Vehicle Navigation

    Abstract : We consider the robot localization problem with sparse visual feature selection. The underlying key property is that contributions of trackable features (landmarks) appear linearly in the information matrix of the corresponding estimation problem. We utilize standard models for motion and vision system using a camera to formulate the feature selection problem over moving finite-time horizons. We propose a scalable randomized sampling algorithm to select more informative features to obtain a certain estimation quality. We provide probabilistic performance guarantees for our method. The time-complexity of our feature selection algorithm is linear in the number of candidate features, which is practically plausible and outperforms existing greedy methods that scale quadratically with the number of candidate features. Our numerical simulations confirm that not only the execution time of our proposed method is comparably less than that of the greedy method, but also the resulting estimation quality is very close to the greedy method.

- A Tightly Coupled VLC-Inertial Localization System by EKF

    Author: Liang, Qing | Hong Kong University of Science and Technology
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Service Robots; Sensor Fusion; Localization

    Abstract : Lightweight global localization is favorable by many resource-constrained platforms working in GPS-denied indoor environments, such as service robots and mobile devices. In recent years, visible light communication (VLC) has emerged as a promising technology that can support global positioning in buildings by reusing the widespread LED luminaries as artificial visual landmarks. In this paper, we propose a novel VLC/IMU integrated system with a tightly coupled formulation by an extended-Kalman filter (EKF) for robust VLC-inertial localization. By tightly fusing the inertial measurements with the visual measurements of LED fiducials, our EKF localizer can provide lightweight real-time accurate global pose estimates, even in LED-shortage situations. We further complete it with a 2-point global pose initialization method that loosely couples the two sensor measurements. We can hence bootstrap our system with two or more LED features observed in one camera frame. The proposed system and method are verified by extensive field experiments using dozens of self-made LED prototypes.

- Localization of Inspection Device Along Belt Conveyors with Multiple Branches Using Deep Neural Networks

    Author: Yasutomi, André Yuji | Hitachi Ltd
    Author: Enoki, Hideo | Hitachi Ltd
 
    keyword: Localization; Manufacturing, Maintenance and Supply Chains; Deep Learning in Robotics and Automation

    Abstract : Regular inspections of belt conveyors are required to prevent the damage of transported objects. Nevertheless, inspections can be troublesome for belt conveyors composed of a plurality of belt lines with multiple branches. To improve the inspection process, an inspection device composed of an inertial measurement unit (IMU) inside a transported object was introduced in our previous study jointly with an algorithm to detect anomalies in the joints of the belt lines. When belt conveying this device for inspection, however, it is required to not only detect the anomaly but also know its position. This study presents a novel method to estimate the position of the inspection device along the belt conveyor using a deep neural network (DNN). The DNN uses the IMU data to detect seven types of features (passage through five types of joints, device stoppage and regular transport), which, when matched to data on a belt conveyor position database, can correctly be translated into positions. Additionally, the proposed method enables the detection of changes of routes along the belt conveyor that occur when a belt line branches into two output streams. To enhance the DNN feature detections, two original algorithms for DNN output post-processing are also introduced. Experiments with a complex belt conveyor demonstrate this method can successfully detect the position of the inspected device more accurately and more cost-effectively than conventional methods.

- Localising PMDs through CNN Based Perception of Urban Streets

    Author: Jayasuriya, Maleen | University of Technology Sydney
    Author: Arukgoda, Janindu | University of Technology Sydney
    Author: Ranasinghe, Ravindra | University of Technology Sydney
    Author: Dissanayake, Gamini | University of Technology Sydney
 
    keyword: Localization; Visual-Based Navigation

    Abstract : The main contribution of this paper is a novel Extended Kalman Filter (EKF) based localisation scheme that fuses two complementary approaches to outdoor vision based localisation. This EKF is aided by a front end consisting of two Convolutional Neural Networks (CNNs) that provide the necessary perceptual information from camera images. The first approach involves a CNN based extraction of information corresponding to artefacts such as curbs, lane markings, and manhole covers to localise on a vector distance transform representation of a binary image of these ground surface boundaries. The second approach involves a CNN based detection of common environmental landmarks such as tree trunks and light poles, which are represented as point features on a sparse map. Utilising CNNs to obtain higher level information about the environment enables this framework to avoid the typical pitfalls of common vision based approaches that use low level hand crafted features for localisation. The EKF framework makes it possible to deal with false positives and missed detections that are inevitable in a practical CNN, to produce a location estimate together with its associated uncertainty. Experiments using a Personal Mobility Device (PMD) driven in typical suburban streets are presented to demonstrate the effectiveness of the proposed localiser.

- The Complex-Step Derivative Approximation on Matrix Lie Groups

    Author: Cossette, Charles Champagne | McGill University
    Author: Walsh, Alex | McGill University
    Author: Forbes, James Richard | McGill University
 
    keyword: Localization; Optimization and Optimal Control; SLAM

    Abstract : The complex-step derivative approximation is a numerical differentiation technique that can achieve analytical accuracy, to machine precision, with a single function evaluation. In this paper, the complex-step derivative approximation is extended to be compatible with elements of matrix Lie groups. As with the standard complex-step derivative, the method is still able to achieve analytical accuracy, up to machine precision, with a single function evaluation. Compared to a central-difference scheme, the proposed complex-step approach is shown to have superior accuracy. The approach is applied to two different pose estimation problems, and is able to recover the same results as an analytical method when available.

- Hybrid Localization Using Model and Learning-Based Methods: Fusion of Monte Carlo and E2E Localizations Via Importance Sampling

    Author: Akai, Naoki | Nagoya University
    Author: Hirayama, Takatsugu | Nagoya University
    Author: Murase, Hiroshi | Nagoya University
 
    keyword: Localization; Deep Learning in Robotics and Automation; Autonomous Vehicle Navigation

    Abstract : This paper proposes a hybrid localization method that fuses Monte Carlo localization (MCL) and convolutional neural network (CNN)-based end-to-end (E2E) localization. MCL is based on particle filter and requires proposal distributions to sample the particles. The proposal distribution is generally predicted using a motion model. However, because the motion model cannot handle unanticipated errors, the predicted distribution is sometimes inaccurate. The use of other ideal proposal distributions, such as the measurement model, can improve robustness against such unanticipated errors. This technique is called importance sampling (IS). However, it is difficult to sample the particles from such ideal distributions because they are not represented in the closed form. Recent works have proved that CNNs with dropout layers represent the posterior distributions over their outputs conditioned on the inputs and the CNN predictions are equivalent to sampling the outputs from the posterior. Therefore, the proposed method utilizes a CNN to sample the particles and fuses them with MCL via IS. Consequently, the advantages of both MCL and E2E localization can be simultaneously leveraged while preventing their disadvantages. Experiments demonstrate that the proposed method can smoothly estimate the robot pose, similar to the model-based method, and quickly re-localize it from the failures, similar to the learning-based method.

- Measurement Scheduling for Cooperative Localization in Resource-Constrained Conditions

    Author: Yan, Qi | Shanghai Jiao Tong University
    Author: Jiang, Li | Shanghai Jiao Tong University
    Author: Kia, Solmaz | Uinversity of California Irvine
 
    keyword: Localization; Networked Robots; Multi-Robot Systems

    Abstract : This paper studies the measurement scheduling problem for a group of N mobile robots moving on a flat surface that are performing cooperative localization (CL). We consider a scenario in which due to the limited on-board resources such as battery life and communication bandwidth only a given number of relative measurements per robot are allowed at observation and update stage. Optimal selection of which teammates a robot should take a relative measurement from such that the updated joint localization uncertainty of the team is minimized is an NP-hard problem. In this paper, we propose a suboptimal greedy approach that allows each robot to choose its landmark robots locally in polynomial time. Our method, unlike the known results in the literature, does not assume the full-observability of CL algorithm. Moreover, it does not require inter-robot communication at the scheduling stage. That is, there is no need for the robots to collaborate to carry out the landmark robot selections. We discuss the application of our method in the context of a state-of-the-art decentralized CL algorithm and demonstrate its effectiveness through numerical simulations. Even though our solution does not come with rigorous performance guarantees, its low computational cost along with no communication requirement makes it an appealing solution for operations with resource-constrained robots.

- Quantifying Robot Localization Safety: A New Integrity Monitoring Method for Fixed-Lag Smoothing

    Author: Abdul Hafez, Osama | Illinois Institute of Technology
    Author: Duenas Arana, Guillermo | Illinois Institute of Technology
    Author: Joerger, Mathieu | Virginia Tech
    Author: Spenko, Matthew | Illinois Institute of Technology
 
    keyword: Localization; Autonomous Vehicle Navigation; Probability and Statistical Methods

    Abstract : Localization safety, or integrity risk, is the probability of undetected localization failures and a common aviation performance metric used to verify a minimum accuracy requirement. As autonomous robots become more common, applying integrity risk metrics will be necessary to verify localization performance. This paper introduces a new method, solution separation, to quantify landmark-based mobile robot localization safety for fixed-lag smoothing estimators and compares it's computation time and fault detection capabilities to a chi-squared integrity monitoring method. Results show that solution separation is more computationally efficient and results in a tighter upper-bound on integrity risk when few measurements are included, which makes it the method of choice for lightweight, safety-critical applications such as UAVs. Conversely, chi-squared requires more computing resources but performs better when more measurements are included, making the method more appropriate for high performance computing platforms such as autonomous vehicles.

- Visual Localization with Google Earth Images for Robust Global Pose Estimation of UAVs

    Author: Patel, Bhavit | University of Toronto
    Author: Barfoot, Timothy | University of Toronto
    Author: Schoellig, Angela P. | University of Toronto
 
    keyword: Localization; Field Robots; Aerial Systems: Perception and Autonomy

    Abstract : We estimate the global pose of a multirotor UAV by visually localizing images captured during a flight with Google Earth images pre-rendered from known poses. We metrically localize real images with georeferenced rendered images using a dense mutual information technique to allow accurate global pose estimation in outdoor GPS-denied environments. We show the ability to consistently localize throughout a sunny summer day despite major lighting changes while demonstrating that a typical feature-based localizer struggles under the same conditions. Successful image registrations are used as measurements in a filtering framework to apply corrections to the pose estimated by a gimballed visual odometry pipeline. We achieve less than 1 metre and 1 degree RMSE on a 303 metre flight and less than 3 metres and 3 degrees RMSE on six 1132 metre flights as low as 36 metres above ground level conducted at different times of the day from sunrise to sunset.

- Relax and Recover: Guaranteed Range-Only Continuous Localization

    Author: Pacholska, Michalina | EPFL
    Author: D�mbgen, Frederike | EPFL
    Author: Scholefield, Adam | EPFL
 
    keyword: Localization; Range Sensing; Optimization and Optimal Control

    Abstract : Range-only localization has applications as diverse as underwater navigation, drone tracking and indoor localization. While the theoretical foundations of lateration�range-only localization for static points�are well understood, there is a lack of understanding when it comes to localizing a moving device. As most interesting applications in robotics involve moving objects, we study the theory of trajectory recovery. This problem has received a lot of attention; however, state-of-the-art methods are of a probabilistic or heuristic nature and not well suited for guaranteeing trajectory recovery. In this letter, we pose trajectory recovery as a quadratic problem and show that we can relax it to a linear form, which admits a closed-form solution. We provide necessary and sufficient recovery conditions and in particular show that trajectory recovery can be guaranteed when the number of measurements is proportional to the trajectory complexity. Finally, we apply our reconstruction algorithm to simulated and real-world data.

- SPRINT: Subgraph Place Recognition for Intelligent Transportation

    Author: Latif, Yasir | University of Adelaide
    Author: Doan, Anh-Dzung | The University of Adelaide
    Author: Chin, Tat-Jun | The University of Adelaide
    Author: Reid, Ian | University of Adelaide
 
    keyword: Localization; SLAM

    Abstract : Place recognition is an important problem in mobile robotics that allows a robot to localize itself using image data alone. Recent methods have shown good performance for place recognition under varying environmental conditions by exploiting sequential nature of the incoming data. Using k nearest neighbours based image retrieval as the backend, and exploiting the structure of the image acquisition process which introduces temporal relations between images in the database, the location of possible matches can be restricted to a subset of all the images seen so far. We show that when using Hidden Markov Models for inference, the original problem space can be restricted to a significantly smaller subspace by exploiting these properties of the problem, significantly reducing the inference time. This is important if we want to carry out place recognition over database containing millions of images. We show large scale experiments using publicly sourced data that show the computational performance of the proposed method under varying environmental conditions.

- OneShot Global Localization: Instant LiDAR-Visual Pose Estimation

    Author: Ratz, Sebastian | ETH Zurich, Sevensense Robotics AG
    Author: Dymczyk, Marcin Tomasz | ETH Zurich, Autonomous Systems Lab
    Author: Siegwart, Roland | ETH Zurich
    Author: Dub�, Renaud | ETH Zurich
 
    keyword: Localization; Deep Learning in Robotics and Automation; RGB-D Perception

    Abstract : Globally localizing in a given map is a crucial ability for robots to perform a wide range of autonomous navigation tasks. This paper presents OneShot - a global localization algorithm that uses only a single 3D LiDAR scan at a time, while outperforming approaches based on integrating a sequence of point clouds. Our approach, which does not require the robot to move, relies on learning-based descriptors of point cloud segments and computes the full 6 degree-of-freedom pose in a map. The segments are extracted from the current LiDAR scan and are matched against a database using the computed descriptors. Candidate matches are then verified with a geometric consistency test. We additionally present a strategy to further improve the performance of the segment descriptors by augmenting them with visual information provided by a camera. For this purpose, a custom-tailored neural network architecture is proposed. We demonstrate that our LiDAR-only approach outperforms a state-of-the-art baseline on a sequence of the KITTI dataset and also evaluate its performance on the challenging NCLT dataset. Finally, we show that fusing in visual information boosts segment retrieval rates by up to 26% compared to LiDAR-only description.

- Relocalization on Submaps: Multi-Session Mapping for Planetary Rovers Equipped with Stereo Cameras

    Author: Giubilato, Riccardo | University of Padova
    Author: Vayugundla, Mallikarjuna | DLR (German Aerospace Center)
    Author: Schuster, Martin J. | German Aerospace Center (DLR)
    Author: Stuerzl, Wolfgang | DLR, Institute of Robotics and Mechantronics
    Author: Wedler, Armin | DLR - German Aerospace Center
    Author: Triebel, Rudolph | German Aerospace Center (DLR)
    Author: Debei, Stefano | Université Degli Studi Di Padova
 
    keyword: Localization; Space Robotics and Automation; Mapping

    Abstract : To enable long term exploration of extreme environments such as planetary surfaces, heterogeneous robotic teams need the ability to localize themselves on previously built maps. While the Localization and Mapping problem for single sessions can be efficiently solved with many state of the art solutions, place recognition in natural environments still poses great challenges for the perception system of a robotic agent. In this paper we propose a relocalization pipeline which exploits both 3D and visual information from stereo cameras to detect matches across local point clouds of multiple SLAM sessions. Our solution is based on a Bag of Binary Words scheme where binarized SHOT descriptors are enriched with visual cues to recall in a fast and efficient way previously visited places. The proposed relocalization scheme is validated on challenging datasets captured using a planetary rover prototype on Mount Etna, designated as a Moon analogue environment

- DeepTIO: A Deep Thermal-Inertial Odometry with Visual Hallucination

    Author: Saputra, Muhamad Risqi U. | University of Oxford
    Author: Porto Buarque de Gusm�o, Pedro | University of Oxford
    Author: Lu, Chris Xiaoxuan | University of Oxford
    Author: Almalioglu, Yasin | The University of Oxford
    Author: Rosa, Stefano | University of Oxford
    Author: Chen, Changhao | University of Oxford
    Author: Wahlstrom, Johan | University of Oxford
    Author: Wang, Wei | University of Oxford
    Author: Markham, Andrew | Oxford University
    Author: Trigoni, Niki | University of Oxford
 
    keyword: Localization; Deep Learning in Robotics and Automation; Sensor Fusion

    Abstract : Visual odometry shows excellent performance in a wide range of environments. However, in visually-denied scenarios (e.g. heavy smoke or darkness), pose estimates degrade or even fail. Thermal cameras are commonly used for perception and inspection when the environment has low visibility. However, their use in odometry estimation is hampered by the lack of robust visual features. In part, this is as a result of the sensor measuring the ambient temperature profile rather than scene appearance and geometry. To overcome this issue, we propose a Deep Neural Network model for thermal-inertial odometry (DeepTIO) by incorporating a visual hallucination network to provide the thermal network with complementary information. The hallucination network is taught to predict fake visual features from thermal images by using Huber loss. We also employ selective fusion to attentively fuse the features from three different modalities, i.e thermal, hallucination, and inertial features. Extensive experiments are performed in hand-held and mobile robot data in benign and smoke-filled environments, showing the efficacy of the proposed model.

- RSL-Net: Localising in Satellite Images from a Radar on the Ground

    Author: Tang, Tim Yuqing | University of Oxford
    Author: De Martini, Daniele | University of Oxford
    Author: Barnes, Dan | University of Oxford
    Author: Newman, Paul | Oxford University
 
    keyword: Localization; Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation

    Abstract : This paper is about localising a vehicle in an overhead image using FMCW radar mounted on a ground vehicle. FMCW radar offers extraordinary promise and efficacy for vehicle localisation. It is impervious to all weather types and lighting conditions. However the complexity of the interactions between millimetre radar wave and the physical environment makes it a challenging domain. Infrastructure-free large-scale radar-based localisation is in its infancy. Typically here a map is built and suitable techniques, compatible with the nature of sensor, are brought to bear. In this work we eschew the need for a radar-based map; instead we simply use an overhead image -- a resource readily available everywhere. This paper introduces a method that not only naturally deals with the complexity of the signal type but does so in the context of cross modal processing.


- Kidnapped Radar: Topological Radar Localisation Using Rotationally-Invariant Metric Learning

    Author: Saftescu, Stefan | University of Oxford
    Author: Gadd, Matthew | University of Oxford
    Author: De Martini, Daniele | University of Oxford
    Author: Barnes, Dan | University of Oxford
    Author: Newman, Paul | Oxford University
 
    keyword: Localization; Mapping; Deep Learning in Robotics and Automation

    Abstract : This paper presents a system for robust, large-scale topological localisation using Frequency-Modulated Continuous-Wave (FMCW) scanning radar. We learn a metric space for embedding polar radar scans using CNN and VLAD architectures traditionally applied to the visual domain. However, we tailor the feature extraction for more suitability to the polar nature of radar scan formation using cylindrical convolutions, anti-aliasing blurring, and azimuth-wise max-pooling; all in order to bolster the rotational invariance. The enforced metric space is then used to encode a reference trajectory, serving as a map, which is queried for nearest neighbours (NN) for recognition of places at run-time. We demonstrate the performance of our topological localisation system over the course of many repeat forays using the largest radar-focused mobile autonomy dataset released to date, totalling 280 km of urban driving, a small portion of which we also use to learn the weights of the modified architecture. As this work represents a novel application for FMCW radar, we analyse the utility of the proposed method via a comprehensive set of metrics which provide insight into the efficacy when used in a realistic system, showing improved performance over the root architecture even in the face of random rotational perturbation.

- Global Visual Localization in LiDAR-Maps through Shared 2D-3D Embedding Space

    Author: Cattaneo, Daniele | University of Freiburg
    Author: Vaghi, Matteo | Université Degli Studi Di Milano - Bicocca
    Author: Fontana, Simone | Univ. of Milano Bicocca
    Author: Ballardini, Augusto Luis | Universidad De Alcal�
    Author: Sorrenti, Domenico G. | Université Di Milano - Bicocca
 
    keyword: Localization; Deep Learning in Robotics and Automation; Sensor Fusion

    Abstract : Global localization is an important and widely studied problem for many robotic applications. Place recognition approaches can be exploited to solve this task, <i>e.g.</i>, in the autonomous driving field. While most vision-based approaches match an image <i>w.r.t.</i> an image database, global visual localization within LiDAR-maps remains fairly unexplored, even though the path toward high definition 3D maps, produced mainly from LiDARs, is clear. In this work we leverage Deep Neural Network (DNN) approaches to create a shared embedding space between images and LiDAR-maps, allowing for image to 3D-LiDAR place recognition. We trained a 2D and a 3D DNN that create embeddings, respectively from images and from point clouds, that are close to each other whether they refer to the same place. An extensive experimental activity is presented to assess the effectiveness of the approach <i>w.r.t.</i> different learning paradigms, network architectures, and loss functions. All the evaluations have been performed using the Oxford Robotcar Dataset, which encompasses a wide range of weather and light conditions.

- Unsupervised Learning Methods for Visual Place Recognition in Discretely and Continuously Changing Environments

    Author: Schubert, Stefan | Chemnitz University of Technology
    Author: Neubert, Peer | Chemnitz University of Technology
    Author: Protzel, Peter | Chemnitz University of Technology
 
    keyword: Localization; SLAM; Visual-Based Navigation

    Abstract : Visual place recognition in changing environments is the problem of finding matchings between two sets of observations, a query set and a reference set, despite severe appearance changes. Recently, image comparison using CNN-based descriptors showed very promising results. However, existing experiments from the literature typically assume a single distinctive condition within each set (e.g., reference: day, query: night). We demonstrate that as soon as the conditions change within one set (e.g., reference: day,	query: traversal daytime-dusk-night-dawn), different places under the same condition can suddenly look more similar than same places under different conditions and state-of-the-art approaches like CNN-based descriptors fail. This paper discusses this practically very important problem of in-sequence condition changes and defines a hierarchy of problem setups from (1) no in-sequence changes, (2) discrete in-sequence changes, to (3) continuous in-sequence changes. We will experimentally evaluate the effect of these changes on two state-of-the-art CNN-descriptors. Our experiments emphasize the importance of statistical standardization of descriptors and shows its limitations in case of continuous changes. To address this practically most relevant setup, we investigate and experimentally evaluate the application of unsupervised learning methods using two available PCA-based approaches and propose a novel clustering-based extension of the statistical normalization.

- LOL: Lidar-Only Odometry and Localization in 3D Point Cloud Maps

    Author: Rozenberszki, D�vid | MTA SZTAKI
    Author: Majdik, Andras | Hungarian Academy of Sciences
 
    keyword: Localization; Range Sensing; Computer Vision for Transportation

    Abstract : In this paper we deal with the problem of odometry and localization for Lidar-equipped vehicles driving in urban environments, where a premade target map exists to localize against. In our problem formulation, to correct the accumulated drift of the Lidar-only odometry we apply a place recognition method to detect geometrically similar locations between the online 3D point cloud and the a priori offline map. In the proposed system, we integrate a state-of-the-art Lidar-only odometry algorithm with a recently proposed 3D point segment matching method by complementing their advantages. Also, we propose additional enhancements in order to reduce the number of false matches between the online point cloud and the target map, and to refine the position estimation error whenever a good match is detected. We demonstrate the utility of the proposed LOL system on several Kitti datasets of different lengths and environments, where the relocalization accuracy and the precision of the vehicle's trajectory were significantly improved in every case, while still being able to maintain real-time performance.

- Localising Faster: Efficient and Precise Lidar-Based Robot Localisation in Large-Scale Environments

    Author: Sun, Li | University of Oxford
    Author: Adolfsson, Daniel | Örebro University
    Author: Magnusson, Martin | Örebro University
    Author: Andreasson, Henrik | Örebro University
    Author: Posner, Ingmar | Oxford University
    Author: Duckett, Tom | University of Lincoln
 
    keyword: Localization; Deep Learning in Robotics and Automation

    Abstract : This paper proposes a novel approach for global localisation of mobile robots in large-scale environments. Our method leverages learning-based localisation and filtering-based localisation, to localise the robot efficiently and precisely through seeding Monte Carlo Localisation (MCL) with a deep-learned distribution. In particular, a fast localisation system rapidly estimates the 6-DOF pose through a deep-probabilistic model (Gaussian Process Regression with a deep kernel), then a precise recursive estimator refines the estimated robot pose according to the geometric alignment. More importantly, the Gaussian method (i.e. deep probabilistic localisation) and non-Gaussian method (i.e. MCL) can be integrated naturally via importance sampling. Consequently, the two systems can be integrated seamlessly and mutually benefit from each other. To verify the proposed framework, we provide a case study in large-scale localisation with a 3D lidar sensor. Our experiments on the Michigan NCLT long-term dataset show that the proposed method is able to localise the robot in 1.94 s on average (median of 0.8 s) with precision 0.75 m in a largescale environment of approximately 0.5 km2.

- Set-Membership State Estimation by Solving Data Association

    Author: Rohou, Simon | ENSTA Bretagne
    Author: Desrochers, Benoit | ENSTA-Bretagne
    Author: Jaulin, Luc | ENSTA-Bretagne
 
    keyword: Localization; Autonomous Vehicle Navigation; Marine Robotics

    Abstract : This paper deals with the localization problem of a robot in an environment made of indistinguishable landmarks, and assuming the initial position of the vehicle is unknown. This scenario is typically encountered in underwater applications for which landmarks such as rocks all look alike. Furthermore, the position of the robot may be lost during a diving phase, which obliges us to consider unknown initial position. We propose a deterministic approach to solve simultaneously the problems of data association and state estimation, without combinatorial explosion. The efficiency of the method is shown on an actual experiment involving an underwater robot and sonar data.

## Learning from Demonstration

- Benchmark for Skill Learning from Demonstration: Impact of User Experience, Task Complexity, and Start Configuration on Performance

    Author: Rana, Muhammad Asif | Georgia Institute of Technology
    Author: Chen, Daphne | Georgia Institute of Technology
    Author: Williams, Jacob | Georgia Institute of Technology
    Author: Chu, Vivian | Georgia Institute of Technology
    Author: Ahmadzadeh, S. Reza | University of Massachusetts Lowell
    Author: Chernova, Sonia | Georgia Institute of Technology
 
    keyword: Learning from Demonstration

    Abstract : We contribute a study benchmarking the performance of multiple motion-based learning from demonstration approaches. Given the number and diversity of existing methods, it is critical that comprehensive empirical studies be performed comparing the relative strengths of these techniques. In particular, we evaluate four approaches based on properties an end user may desire for real-world tasks. To perform this evaluation, we collected data from nine participants, across four manipulation tasks. The resulting demonstrations were used to train 180 task models and evaluated on 720 task reproductions on a physical robot. Our results detail how i) complexity of the task, ii) the expertise of the human demonstrator, and iii) the starting configuration of the robot affect task performance. The collected dataset of demonstrations, robot executions, and evaluations are publicly available. Research insights and guidelines are also provided to guide future research and deployment choices about these approaches.

- MPC-Net: A First Principles Guided Policy Search

    Author: Carius, Jan | ETH Zurich
    Author: Farshidian, Farbod | ETH Zurich
    Author: Hutter, Marco | ETH Zurich
 
    keyword: Learning from Demonstration; Legged Robots; Optimization and Optimal Control

    Abstract : We present an Imitation Learning approach for the control of dynamical systems with a known model. Our policy search method is guided by solutions from MPC. Typical policy search methods of this kind minimize a distance metric between the guiding demonstrations and the learned policy. Our loss function, however, corresponds to the minimization of the control Hamiltonian, which derives from the principle of optimality. Therefore, our algorithm directly attempts to solve the optimality conditions with a parameterized class of control laws. Additionally, the proposed loss function explicitly encodes the constraints of the optimal control problem and we provide numerical evidence that its minimization achieves improved constraint satisfaction. We train a mixture-of-expert neural network architecture for controlling a quadrupedal robot and show that this policy structure is well suited for such multimodal systems. The learned policy can successfully stabilize different gaits on the real walking robot from less than 10 min of demonstration data.

- Robot Programming without Coding

    Author: Lentini, Gianluca | University of Pisa
    Author: Grioli, Giorgio | Istituto Italiano Di Tecnologia
    Author: Catalano, Manuel Giuseppe | Istituto Italiano Di Tecnologia
    Author: Bicchi, Antonio | Université Di Pisa
 
    keyword: Learning from Demonstration; Natural Machine Motion; Learning and Adaptive Systems

    Abstract : An approach toward intuitive and easy robot programming, consists to transfer skills from humans to machines, through demonstration. A vast literature exists on learning from multiple demonstrations. This paper, on the other hand, tackles the problem of providing all needed information to execute a certain task by resorting to one single demonstration - hence, a problem closer to programming than to learning. We use wearable consumer devices - but no keyboard nor coding - as programming tools, to let the programmer tele-operate the robot, which in turn records the most salient features and affordances from the object, environment, robot, and human. To enable this goal we combine off-the-shelf soft-articulated robotic components with the framework of Dynamic Movement Primitives, which we contribute to extend to generalize human trajectories and impedance regulation skills. This framework enables to teach robot quickly and in a intuitive way without coding. Experimental tests have been performed on a dual-arm system composed by two 7-dofs collaborative robots equipped with anthropomorphic end-effectors. Experiments show the functionality of the framework and verify the effectiveness of the impedance extension.

- Learning Robust Task Priorities and Gains for Control of Redundant Robots

    Author: Penco, Luigi | INRIA
    Author: Mingo Hoffman, Enrico | Fondazione Istituto Italiano Di Tecnologia
    Author: Modugno, Valerio | Sapienza Université Di Roma
    Author: Gomes, Waldez | INRIA
    Author: Mouret, Jean-Baptiste | Inria
    Author: Ivaldi, Serena | INRIA
 
    keyword: Learning from Demonstration; Humanoid Robots

    Abstract : Generating complex movements in redundant robots like humanoids is usually done by means of multi-task controllers based on quadratic programming, where a multitude of tasks is organized according to strict or soft priorities. Time-consuming tuning and expertise are required to choose suitable task priorities, and to optimize their gains. Here, we automatically learn the controller configuration (soft and strict task priorities and Convergence Gains), looking for solutions that track a variety of desired task trajectories efficiently while preserving the robot's balance. We use multi-objective optimization to compare and choose among Pareto-optimal solutions that represent a trade-off of performance and robustness and can be transferred onto the real robot. We experimentally validate our method by learning a control configuration for the iCub humanoid, to perform different whole-body tasks, such as picking up objects, reaching and opening doors.

- Planning with Uncertain Specifications (PUnS)

    Author: Shah, Ankit Jayesh | Massachusetts Institute of Technology
    Author: Li, Shen | MIT
    Author: Shah, Julie A. | MIT
 
    keyword: Learning from Demonstration; AI-Based Methods

    Abstract : Reward engineering is crucial to high performance in reinforcement learning systems. Prior research into reward design has largely focused on Markovian functions representing the reward. While there has been research into expressing non-Markov rewards as linear temporal logic (LTL) formulas, this has focused on task specifications directly defined by the user. However, in many real-world applications, task specifications are ambiguous, and can only be expressed as a belief over LTL formulas. In this paper, we introduce planning with uncertain specifications (PUnS), a novel formulation that addresses the challenge posed by non-Markovian specifications expressed as beliefs over LTL formulas. We present four criteria that capture the semantics of satisfying a belief over specifications for different applications, and analyze the qualitative implications of these criteria within a synthetic domain. We demonstrate the existence of an equivalent Markov decision process (MDP) for any instance of PUnS. Finally, we demonstrate our approach on the real-world task of setting a dinner table automatically with a robot that inferred task specifications from human demonstrations.

- Predictive Modeling of Periodic Behavior for Human-Robot Symbiotic Walking

    Author: Clark, Geoffrey | ASU
    Author: Campbell, Joseph | Arizona State University
    Author: Rezayat sorkhabadi, Seyed Mostafa | Arizona State University
    Author: Zhang, Wenlong | Arizona State University
    Author: Ben Amor, Heni | Arizona State University
 
    keyword: Learning from Demonstration; Physical Human-Robot Interaction

    Abstract : We propose in this paper Periodic Interaction Primitives - a probabilistic framework that can be used to learn compact models of periodic behavior. Our approach extends existing formulations of Interaction Primitives to periodic movement regimes, i.e., walking. We show that this model is particularly well-suited for learning data-driven, customized models of human walking, which can then be used for generating predictions over future states or for inferring latent, biomechanical variables. We also demonstrate how the same framework can be used to learn controllers for a robotic prosthesis using an imitation learning approach. Results in experiments with human participants indicate that Periodic Interaction Primitives efficiently generate predictions and ankle angle control signals for a robotic prosthetic ankle, with MAE of 2.21&#9702;in 0.0008s per inference. Performance degrades gracefully in the presence of noise or sensor fall outs. Compared to alternatives, this algorithm functions 20 times faster and performed 4.5 times more accurately on test subjects.

- Adaptive Curriculum Generation from Demonstrations for Sim-To-Real Visuomotor Control

    Author: Hermann, Lukas | University of Freiburg
    Author: Argus, Maximilian | University of Freiburg
    Author: Eitel, Andreas | University of Freiburg
    Author: Amiranashvili, Artemij | University of Freiburg
    Author: Burgard, Wolfram | Toyota Research Institute
    Author: Brox, Thomas | University of Freiburg
 
    keyword: Learning from Demonstration; Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation

    Abstract : We propose Adaptive Curriculum Generation from demonstrations (ACGD) for reinforcement learning in the presence of sparse rewards. Rather than designing shaped reward functions, ACGD adaptively sets the appropriate task difficulty for the learner by controlling where to sample from the demonstration trajectories and which set of simulation parameters to use. We show that training vision-based control policies in simulation while gradually increasing the difficulty of the task via ACGD improves the policy transfer to the real world. The degree of domain randomization is also gradually increased through the task difficulty. We demonstrate zero-shot transfer for two real-world manipulation tasks: pick-and-stow and block stacking.

- Accept Synthetic Objects As Real: End-To-End Training of Attentive Deep Visuomotor Policies for Manipulation in Clutter

    Author: Abolghasemi, Pooya | University of Central Florida
    Author: B�l�ni, Ladislau | University of Central Florida
 
    keyword: Learning from Demonstration; Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation

    Abstract : Recent research demonstrated that it is feasible to end-to-end train multi-task deep visuomotor policies for robotic manipulation using variations of learning from demonstration (LfD) and reinforcement learning (RL). In this paper, we extend the capabilities of end-to-end LfD architectures to object manipulation in clutter. We start by introducing a data augmentation procedure called Accept Synthetic Objects as Real (ASOR). Using ASOR we develop two network architectures: implicit attention ASOR-IA and explicit attention ASOR-EA. Both architectures use the same training data (demonstrations in uncluttered environments) as previous approaches. Experimental results show that ASOR-IA and ASOR-EA succeed in a significant fraction of trials in cluttered environments where previous approaches never succeed. In addition, we find that both ASOR-IA and ASOR-EA outperform previous approaches even in uncluttered environments, with ASOR-EA performing better even in clutter compared to the previous best baseline in an uncluttered environment.

- A Probabilistic Framework for Imitating Human Race Driver Behavior

    Author: L�ckel, Stefan | TU Darmstadt, Porsche AG
    Author: Peters, Jan | Technische Universitét Darmstadt
    Author: van Vliet, Peter | Porsche AG
 
    keyword: Learning from Demonstration; Autonomous Vehicle Navigation

    Abstract : Understanding and modeling human driver behavior is crucial for advanced vehicle development. However, unique driving styles, inconsistent behavior, and complex decision processes render it a challenging task, and existing approaches often lack variability or robustness. To approach this problem, we propose Probabilistic Modeling of Driver behavior (ProMoD), a modular framework which splits the task of driver behavior modeling into multiple modules. A global target trajectory distribution is learned with Probabilistic Movement Primitives, clothoids are utilized for local path generation, and the corresponding choice of actions is performed by a neural network. Experiments in a simulated car racing setting show considerable advantages in imitation accuracy and robustness compared to other imitation learning algorithms. The modular architecture of the proposed framework facilitates straightforward extensibility in driving line adaptation and sequencing of multiple movement primitives for future research.

- Learning of Exception Strategies in Assembly Tasks

    Author: Nemec, Bojan | Jozef Stefan Institute
    Author: Simoni&#269;, Mihael | Jo�ef Stefan Institute
    Author: Ude, Ales | Jozef Stefan Institute
 
    keyword: Learning from Demonstration; Learning and Adaptive Systems; Intelligent and Flexible Manufacturing

    Abstract : Assembly tasks performed with a robot often fail due to unforeseen situations, regardless of the fact that we carefully learned and optimized the assembly policy. This problem is even more present in humanoid robots acting in an unstructured environment where it is not possible to anticipate all factors that might lead to the failure of the given task. In this work, we propose a concurrent LfD framework, which associates demonstrated exception strategies to the given context. Whenever a failure occurs, the proposed algorithm generalizes past experience regarding the current context and generates an appropriate policy that solves the assembly issue. For this purpose, we applied PCA on force/torque data, which generates low dimensional descriptor of the current context. The proposed framework was validated in a peg-in-hole (PiH) task using Franka-Emika Panda robot.

- A Framework for Learning from Demonstration with Minimal Human Effort

    Author: Rigter, Marc | University of Oxford
    Author: Lacerda, Bruno | University of Oxford
    Author: Hawes, Nick | University of Oxford
 
    keyword: Learning from Demonstration; Human-Centered Robotics; Telerobotics and Teleoperation

    Abstract : We consider robot learning in the context of shared autonomy, where control of the system can switch between a human teleoperator and autonomous control. In this setting we address reinforcement learning, and learning from demonstration, where there is a cost associated with human time. This cost represents the human time required to teleoperate the robot, or recover the robot from failures. For each episode, the agent must choose between requesting human teleoperation, or using one of its autonomous controllers. In our approach, we learn to predict the success probability for each controller, given the initial state of an episode. This is used in a contextual multi-armed bandit algorithm to choose the controller for the episode. Furthermore, an autonomous controller is learnt from demonstrations and reinforcement learning so that the system becomes increasingly autonomous with more experience. We show that our approach to controller selection reduces the human cost to perform two robotics tasks.

- Learning Constraints from Locally-Optimal Demonstrations under Cost Function Uncertainty

    Author: Chou, Glen | University of Michigan
    Author: Ozay, Necmiye | Univ. of Michigan
    Author: Berenson, Dmitry | University of Michigan
 
    keyword: Learning from Demonstration

    Abstract : We present an algorithm for learning parametric constraints from locally-optimal demonstrations, where the cost function being optimized is uncertain to the learner. Our method uses the Karush-Kuhn-Tucker (KKT) optimality conditions of the demonstrations within a mixed integer linear program (MILP) to learn constraints which are consistent with the local optimality of the demonstrations, by either using a known constraint parameterization or by incrementally growing a parameterization that is consistent with the demonstrations. We provide theoretical guarantees on the conservativeness of the recovered safe/unsafe sets and analyze the limits of constraint learnability when using locally-optimal demonstrations. We evaluate our method on high-dimensional constraints and systems by learning constraints for 7-DOF arm and quadrotor examples, show that it outperforms competing constraint-learning approaches, and can be effectively used to plan new constraint-satisfying trajectories in the environment.

- Gershgorin Loss Stabilizes the Recurrent Neural Network Compartment of an End-To-End Robot Learning Scheme

    Author: Lechner, Mathias | IST Austria
    Author: Hasani, Ramin | TU Wien
    Author: Rus, Daniela | MIT
    Author: Grosu, Radu | TU Wien
 
    keyword: Learning from Demonstration; Deep Learning in Robotics and Automation; Model Learning for Control

    Abstract : Traditional robotic control suits require profound task-specific knowledge for designing, building and testing control software. The rise of Deep Learning has enabled end-to-end solutions to be learned entirely from data, requiring minimal knowledge about the application area. We design a learning scheme to train end-to-end linear dynamical systems (LDS)s by gradient descent in imitation learning robotic domains. We introduce a new regularization loss component together with a learning algorithm that improves the stability of the learned autonomous system, by forcing the eigenvalues of the internal state updates of an LDS to be negative reals. We evaluate our approach on a series of real-life and simulated robotic experiments, in comparison to linear and nonlinear Recurrent Neural Network (RNN) architectures. Our results show that our stabilizing method significantly improves test performance of LDS, enabling such linear models to match the performance of contemporary nonlinear RNN architectures. A video of the obstacle avoidance performance of our method on a mobile robot, in unseen environments, compared to other methods can be viewed at https://youtu.be/mhEsCoNao5E.

- Mini-Batched Online Incremental Learning through Supervisory Teleoperation with Kinesthetic Coupling

    Author: Latifee, Hiba | Korea Advanced Institute of Science and Technology
    Author: Pervez, Affan | Technical University of Munich
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
    Author: Lee, Dongheui | Technical University of Munich
 
    keyword: Learning from Demonstration; Telerobotics and Teleoperation; Haptics and Haptic Interfaces

    Abstract : We propose an online incremental learning approach through teleoperation which allows an operator to partially modify a learned model, whenever it is necessary, during task execution. Compared to conventional incremental learning approaches, the proposed approach is applicable for teleoperation-based teaching and it needs only partial demonstration without any need to obstruct the task execution. Dynamic     Authority distribution and kinesthetic coupling between the operator and the agent helps the operator to correctly perceive the exact instance where modification needs to be asserted in the agent's behaviour online using partial trajectory. For this, we propose a variation of the Expectation-Maximization algorithm for updating original model through mini batches of the modified partial trajectory. The proposed approach reduces human workload and latency for a rhythmic peg-in-hole teleoperation task where online partial modification is required during the task operation.

- Recurrent Neural Network Control of a Hybrid Dynamical Transfemoral Prosthesis with EdgeDRNN Accelerator

    Author: Gao, Chang | University of Zurich and ETH Zurich
    Author: Gehlhar, Rachel | California Institute of Technology
    Author: Ames, Aaron | Caltech
    Author: Liu, Shih-Chii | University of Zurich and ETH Zurich
    Author: Delbruck, Tobi | Institute of Neuroinformatics, University of Zurich/ETH
 
    keyword: Learning from Demonstration; Prosthetics and Exoskeletons

    Abstract : Lower leg prostheses could improve the life quality of amputees by increasing comfort and reducing energy to locomote, but currently control methods are limited in modulating behaviors based upon the human's experience. This paper describes the first steps toward learning complex controllers for dynamical robotic assistive devices. We provide the first example of behavioral cloning to control a powered transfemoral prostheses using a Gated Recurrent Unit (GRU) based recurrent neural network (RNN) running on a custom hardware accelerator that exploits temporal sparsity. The RNN is trained on data collected from the original prosthesis controller. The RNN inference is realized by a novel EdgeDRNN accelerator in real-time. Experimental results show that the RNN can replace the nominal PD controller to realize end-to-end control of the AMPRO3 prosthetic leg walking on flat ground and unforeseen slopes with comparable tracking accuracy. EdgeDRNN computes the RNN about 240 times faster than real time, opening the possibility of running larger networks for more complex tasks in the future. Implementing an RNN on this real-time dynamical system with impacts sets the ground work to incorporate other learned elements of the human-prosthesis system into prosthesis control.

- Cross-Context Visual Imitation Learning from Demonstrations

    Author: Yang, Shuo | Shandong University
    Author: Zhang, Wei | Shandong University
    Author: Lu, Weizhi | Shandong University
    Author: Wang, Hesheng | Shanghai Jiao Tong University
    Author: Li, Yibin | Shandong University
 
    keyword: Learning from Demonstration; Model Learning for Control; Grasping

    Abstract : Imitation learning enables robots to learn a task by simply watching the demonstration of the task. Current imitation learning methods usually require the learner and demonstrator to occur in the same context. This limits their scalability to practical applications. In this paper, we propose a more general imitation learning method which allows the learner and the demonstrator to come from different contexts, such as different viewpoints, backgrounds, and object positions and appearances. Specifically, we design a robotic system consisting of three models: context translation model, depth prediction model and multi-modal inverse dynamics model. First, the context translation model translates the demonstration to the context of learner from a different context. Then combining the color observation and depth observation as inputs, the inverse model maps the multi-modal observations into actions to reproduce the demonstration, where the depth observation is provided by a depth prediction model. By performing the block stacking tasks both in simulation and real world, we prove the cross-context learning advantage of the proposed robotic system over other systems.

- Improving Generalisation in Learning Assistance by Demonstration for Smart Wheelchairs

    Author: Schettino, Vinicius | Imperial College London
    Author: Demiris, Yiannis | Imperial College London
 
    keyword: Learning from Demonstration; Physically Assistive Devices; Virtual Reality and Interfaces

    Abstract : Learning Assistance by Demonstration (LAD) is concerned with using demonstrations of a human agent to teach a robot how to assist another human. The concept has previously been used with smart wheelchairs to provide customised assistance to individuals with driving difficulties. A basic premise of this technique is that the learned assistive policy should be able to generalise to environments different than the ones used for training; but this has not been tested before. In this work we evaluate the assistive power and the generalisation capability of LAD using our custom teleoperation and learning system for smart wheelchairs, while seeking to improve it by experimenting with different combinations of dimensionality reduction techniques and machine learning models. Using Autoencoders to reduce the dimension of laser-scan data and a Gaussian Process as the learning model, we achieved a 23% improvement in prediction performance against the combination used by the latest work on the field. Using this model to assist a driver exposed to a simulated disability, we observed a 9.8% reduction in track completion times when compared to driving without assistance.

- Analyzing the Suitability of Cost Functions for Explaining and Imitating Human Driving Behavior Based on Inverse Reinforcement Learning

    Author: Naumann, Maximilian | FZI Research Center for Information Technology
    Author: Sun, Liting | University of California, Berkeley
    Author: Zhan, Wei | Univeristy of California, Berkeley
    Author: Tomizuka, Masayoshi | University of California
 
    keyword: Learning from Demonstration; Motion and Path Planning; Autonomous Vehicle Navigation

    Abstract : Autonomous vehicles are sharing the road with human drivers. In order to facilitate interactive driving and cooperative behavior in dense traffic, a thorough understanding and representation of other traffic participants' behavior are necessary. Cost functions (or reward functions) have been widely used to describe the behavior of human drivers since they can not only explicitly incorporate the rationality of human drivers and the theory of mind (TOM), but also share similarity with the motion planning problem of autonomous vehicles. Hence, more human-like driving behavior and comprehensible trajectories can be generated to enable safer interaction and cooperation. However, the selection of cost functions in different driving scenarios is not trivial, and there is no systematic summary and analysis for cost function selection and learning from a variety of driving scenarios. <p>In this work, we aim to investigate to what extent cost functions are suitable for explaining and imitating human driving behavior. Further, we focus on how cost functions differ from each other in different driving scenarios. Towards this goal, we first comprehensively review existing cost function structures in literature. Based on that, we point out required conditions for demonstrations to be suitable for inverse reinforcement learning (IRL). Finally, we use IRL to explore suitable features and learn representative cost functions from human driven trajectories in three different scenarios.

 


- A Linearly Constrained Nonparametric Framework for Imitation Learning

    Author: Huang, Yanlong | University of Leeds
    Author: Caldwell, Darwin G. | Istituto Italiano Di Tecnologia
 
    keyword: Learning from Demonstration

    Abstract : In recent years, a myriad of advanced results have been reported in the community of imitation learning, ranging from parametric to non-parametric, probabilistic to non-probabilistic and Bayesian to frequentist approaches. Meanwhile, ample applications (e.g., grasping tasks and human-robot collaborations) further show the applicability of imitation learning in a wide range of domains. While numerous literature is dedicated to the learning of human skills in unconstrained environments, the problem of learning constrained motor skills, however, has not received equal attention. In fact, constrained skills exist widely in robotic systems. For instance, when a robot is demanded to write letters on a board, its end-effector trajectory must comply with the plane constraint from the board. In this paper, we propose linearly constrained kernelized movement primitives (LC-KMP) to tackle the problem of imitation learning with linear constraints. Specifically, we propose to exploit the probabilistic properties of multiple demonstrations, and subsequently incorporate them into a linearly constrained optimization problem, which finally leads to a non-parametric solution. In addition, a connection between our framework and the classical model predictive control is provided. Several examples including simulated writing and locomotion tasks are presented to show the effectiveness of our framework.

- An Energy-Based Approach to Ensure the Stability of Learned Dynamical Systems

    Author: Saveriano, Matteo | University of Innsbruck
 
    keyword: Learning from Demonstration; Learning and Adaptive Systems

    Abstract : Non-linear dynamical systems represent a compact, flexible, and robust tool for real-time motion generation. The effectiveness on dynamical systems relies on their ability to accurately represent stable motions. Several approaches have been proposed to learn stable and accurate motions from demonstration. Some approaches work by separating accuracy and stability into two learning problems, which increases the number of open parameters and the overall training time. Alternative solutions exploit single-step learning but restrict the applicability to one regression technique. This paper presents a single-step approach to learn stable and accurate motions that work with any regression technique. The approach makes energy considerations on the learned dynamics to stabilize the system at run-time while introducing small deviations from the demonstrated motion. Since the initial value of the energy injected into the system affects the reproduction accuracy, it is estimated from training data using an efficient procedure. Experiments on a real robot and comparisons on a public benchmark show the effectiveness of the proposed approach.

- IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline Robot Manipulation Data

    Author: Mandlekar, Ajay Uday | Stanford University
    Author: Ramos, Fabio | University of Sydney, NVIDIA
    Author: Boots, Byron | University of Washington
    Author: Savarese, Silvio | Stanford University
    Author: Fei-Fei, Li | Stanford University
    Author: Garg, Animesh | University of Toronto
    Author: Fox, Dieter | University of Washington
 
    keyword: Learning from Demonstration; Deep Learning in Robotics and Automation; Learning and Adaptive Systems

    Abstract : Learning from offline task demonstrations is a problem of great interest in robotics. For simple short-horizon manipulation tasks with modest variation in task instances, offline learning from a small set of demonstrations can produce controllers that successfully solve the task. However, leveraging a fixed batch of data can be problematic for larger datasets and longer-horizon tasks with greater variation. The data can exhibit substantial diversity and consist of suboptimal solution approaches. In this paper, we propose Implicit Reinforcement without Interaction at Scale (IRIS), a novel framework for learning from large-scale demonstration datasets. IRIS factorizes the control problem into a goal-conditioned low-level controller that imitates short demonstration sequences and a high-level goal selection mechanism that sets goals for the low-level and selectively combines parts of suboptimal solutions leading to more successful task completions. We evaluate IRIS across three datasets, including the RoboTurk Cans dataset collected by humans via crowdsourcing, and show that performant policies can be learned from purely offline learning.

- Geometry-Aware Dynamic Movement Primitives

    Author: Abu-Dakka, Fares | Aalto University
    Author: Kyrki, Ville | Aalto University
 
    keyword: Learning from Demonstration; Motion Control of Manipulators; Industrial Robots

    Abstract : In many robot control problems, factors such as stiffness and damping matrices and manipulability ellipsoids are naturally represented as symmetric positive definite (SPD) matrices, which capture the specific geometric characteristics of those factors. Typical learned skill models such as dynamic movement primitives (DMPs) can not, however, be directly employed with quantities expressed as SPD matrices as they are limited to data in Euclidean space. In this paper, we propose a novel and mathematically principled framework that uses Riemannian metrics to reformulate DMPs such that the resulting formulation can operate with SPD data in the SPD manifold. Evaluation of the approach demonstrates that beneficial properties of DMPs such as change of the goal during operation apply also to the proposed formulation.

- Learning a Pile Loading Controller from Demonstrations

    Author: Yang, Wenyan | Tampere University
    Author: Strokina, Nataliya | Tampere University of Technology
    Author: Serbenyuk, Nikolay | Tampere University
    Author: Ghabcheloo, Reza | Tampere University of Technology
    Author: Kamarainen, Joni-Kristian | Tampere University of Technology
 
    keyword: Learning from Demonstration; Computer Vision for Automation; Robotics in Construction

    Abstract : This work introduces a learning-based pile loading controller for autonomous robotic wheel loaders. Controller parameters are learnt from a small number of demonstrations for which low level sensor (boom angle, bucket angle and hy- drostatic driving pressure), egocentric video frames and control signals are recorded. Application specific deep visual features are learnt from demonstrations using a Siamese network architecture and a combination of cross-entropy and contrastive loss. The controller is based on a Random Forest (RF) regressor that provides robustness against changes in field conditions (loading distance, soil type, weather and illumination). The controller is deployed to a real autonomous robotic wheel loader and it outperforms prior art with a clear margin.

- Learning Navigation Costs from Demonstration in Partially Observable Environments

    Author: Wang, Tianyu | University of California, San Diego
    Author: Dhiman, Vikas | University of Michigan
    Author: Atanasov, Nikolay | University of California, San Diego
 
    keyword: Learning from Demonstration; Autonomous Vehicle Navigation; Model Learning for Control

    Abstract : This paper focuses on inverse reinforcement learning (IRL) to enable safe and efficient autonomous navigation in unknown partially observable environments. The objective is to infer a cost function that explains expert-demonstrated navigation behavior while relying only on the observations and state-control trajectory used by the expert. We develop a cost function representation composed of two parts: a probabilistic occupancy encoder, with recurrent dependence on the observation sequence, and a cost encoder, defined over the occupancy features. The representation parameters are optimized by differentiating the error between demonstrated controls and a control policy computed from the cost encoder. Such differentiation is typically computed by dynamic programming through the value function over the whole state space. We observe that this is inefficient in large partially observable environments because most states are unexplored. Instead, we rely on a closed-form subgradient of the cost-to-go obtained only over a subset of promising states via an efficient motion-planning algorithm such as A* or RRT. Our experiments show that our model exceeds the accuracy of baseline IRL algorithms in robot navigation tasks, while substantially improving the efficiency of training and test-time inference.


## Medical Robots and Systems

- Design of a Percutaneous MRI-Guided Needle Robot with Soft Fluid-Driven Actuator

    Author: He, Zhuoliang | The University of Hong Kong
    Author: Dong, Ziyang | The University of Hong Kong
    Author: Fang, Ge | The University of Hong Kong
    Author: Ho, Justin Di-Lang | The University of Hong Kong
    Author: Cheung, Chim Lee | The University of Hong Kong
    Author: Chang, Hing-Chiu | The University of Hong Kong
    Author: Chong, Ching-Ning | The Chinese University of Hong Kong
    Author: Chan, Ying-Kuen | The Chinese University of Hong Kong
    Author: Chan, Tat-Ming | Prince of Wales Hospital
    Author: Kwok, Ka-Wai | The University of Hong Kong
 
    keyword: Medical Robots and Systems; Soft Robot Applications; Mechanism Design

    Abstract : Percutaneous ablation is a standard therapy for most cases of hepatocellular carcinoma (HCC), which is a general type of primary liver cancer. Magnetic resonance imaging (MRI) offers high-contrast images of soft tissue to monitor the ablation procedure. However, the success of MRI-guided ablation still depends on precise intra-tumor probe placement and skin insertion positioning, both of which require highly experienced operators, and can induce inter-operator variability in ablation results. In this letter, we present a semi-automated robotic system for MRI-guided percutaneous needle procedures. The compact and lightweight design enables the direct fixture of robot on the patient body and simultaneous needle targeting at multiple locations with several robots. Accurate (0.89 - 0.31 mm) needle navigation is achieved by incorporating soft fluid-driven actuators with feedback control and stiffness modulation capabilities. The 3D location of the needle guide is reconfirmed by wireless MR tracking coils. The performance of the robotic platform, such as stiffness, needle positioning accuracy and frequency response was experimentally evaluated. Negligible interference to MR imaging was also validated by an MR compatibility test.

- SCADE: Simultaneous Sensor Calibration and Deformation Estimation of FBG-Equipped Unmodeled Continuum Manipulators (I)

    Author: Alambeigi, Farshid | University of Texas at Austin
    Author: Aghajani Pedram, Sahba | University of California, Los Angeles
    Author: Jason L., Speyer | Department of Mechanical and Aerospace Engineering, University O
    Author: Rosen, Jacob | &#8203;University of California, Los Angeles
    Author: Iordachita, Ioan Iulian | Johns Hopkins University
    Author: Taylor, Russell H. | The Johns Hopkins University
    Author: Armand, Mehran | Johns Hopkins University Applied Physics Laboratory
 
    keyword: Medical Robots and Systems; Sensor Fusion; Flexible Robots

    Abstract : We present a novel stochastic algorithm called simultaneous sensor calibration and deformation estimation(SCADE) to address the problem of modeling deformation behavior of a generic continuum manipulator (CM) in free and obstructed environments. In SCADE, using a	novel mathematical formulation, we introduce a priori model-independent filtering algorithm to fuse the continuous and inaccurate	measurements of an embedded sensor (e.g., magnetic or	piezoelectric sensors) with an intermittent but accurate data of an external imaging system(e.g., optical	trackers or cameras). The	main motivation of this study is the crucial need of obtaining an accurate shape/position estimation of a CM utilized in a surgical intervention.	In these robotic procedures,	the CM is typically equipped with an embedded sensing unit while an external imaging modality (e.g.,ultrasound or a fluoroscopy machine) is also available in	the surgical site. The results of two different set of prior experiments in	free and obstructed environments were used to evaluate the efficacy of SCADE algorithm. The experiments was performed with a CM specifically designed for orthopaedic interventions and equipped with an inaccurate fiber Bragg grating (FBG) ESU and an overhead camera.

- Novel Optimization-Based Design and Surgical Evaluation of a Treaded Robotic Capsule Colonoscope (I)

    Author: Formosa, Gregory | University of Colorado at Boulder
    Author: Prendergast, Joseph Micah | University of Colorado at Boulder
    Author: Edmundowicz, Steven | University of Colorado Anschutz Medical Campus
    Author: Rentschler, Mark | University of Colorado at Boulder
 
    keyword: Medical Robots and Systems; Product Design, Development and Prototyping; Wheeled Robots

    Abstract : Robotic capsule endoscopes (RCEs) are being widely investigated to improve the state of various endoscopy procedures. This article presents the novel design of a multi-DOF sensor-enabled RCE for colonoscopies (Endoculus) and evaluates porcine in vivo and ex vivo performance. The novelty of the design includes a custom �double-worm� drive that removes axial gear forces while reducing radial moments, and the full parameterization of gear geometries allows for size minimization via an optimization routine over design constraints. Two independently controlled motors drive micro-pillared treads above and below the device allowing for two-degrees of freedom (2-DOF) skid-steering, even in a collapsed lumen. The Endoculus contains all functionality of a traditional endoscope: a camera, adjustable light emitting diodes (LEDs), channels for insufflation and irrigation, and a tool port for endoscopy instruments (e.g., forceps, snares, etc.). Additionally, the Endoculus carries an inertial measurement unit, magnetometer, motor encoders, and motor current sensors to aid in future autonomy strategies. Porcine surgical evaluation demonstrated locomotion up to 40 mm/s on the colon mucosa, 2-DOF steering, the ability to traverse haustral folds, and functionality of endoscopy tools. This platform will enable future validation of feedback control, localization, and mapping algorithms in the unconventional in vivo environment.

- Generative Localisation with Uncertainty Estimation through Video-CT Data for Bronchoscopic Biopsy

    Author: Zhao, Cheng | University of Oxford
    Author: Shen, Mali | The Hamlyn Centre for Robotic Surgery, Imperial College London
    Author: Sun, Li | University of Sheffield
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Medical Robots and Systems

    Abstract : Robot-assisted endobronchial intervention requires accurate localisation based on both intra- and pre-operative data. Most existing methods achieve this by registering 2D videos with 3D CT models according to a defined similarity metric with local features. Instead, we formulate the bronchoscopic localisation as a learning-based global localisation using deep neural networks. The proposed network consists of two generative architectures and one auxiliary learning component. The cycle generative architecture bridges the domain variance between the real bronchoscopic videos and virtual views derived from pre-operative CT data so that the proposed approach can be trained through a large number of generated virtual images but deployed through real images. The auxiliary learning architecture leverages complementary relative pose regression to constrain the search space, ensuring consistent global pose predictions. Most importantly, the uncertainty of each global pose is obtained through variational inference by sampling within the learned underlying probability distribution. Detailed validation results demonstrate the localisation accuracy with reasonable uncertainty achieved and its potential clinical value.

- Internet of Things (IoT)-Based Collaborative Control of a Redundant Manipulator for Teleoperated Minimally Invasive Surgeries

    Author: Su, Hang | Politecnico Di Milano
    Author: Ovur, Salih Ertug | Politecnico Di Milano
    Author: Li, Zhijun | University of Science and Technology of China
    Author: Hu, Yingbai | Technische Universitét M�nchen
    Author: Li, Jiehao | Beijing Institute of Technology
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
    Author: Ferrigno, Giancarlo | Politecnico Di Milano
    Author: De Momi, Elena | Politecnico Di Milano
 
    keyword: Medical Robots and Systems; Cognitive Human-Robot Interaction; Redundant Robots

    Abstract : In this paper, an Internet of Things-based human-robot collaborative control scheme is developed in Robot-assisted Minimally Invasive Surgery scenario. A hierarchical operational space formulation is designed to exploit the redundancies of the 7-DoFs redundant manipulator to handle multiple operational tasks based on their priority levels, such as guaranteeing a remote center of motion constraint and avoiding collision with a swivel motion without influencing the undergoing surgical operation. Furthermore, the concept of the Internet of Robotic Things is exploited to facilitate the best action of the robot in human-robot interaction. Instead of utilizing compliant swivel motion, HTC VIVE PRO controllers, used as the Internet of Things technology, is adopted to detect the collision. A virtual force is applied to the robot elbow, enabling a smooth swivel motion for human-robot interaction. The effectiveness of the proposed strategy is validated using experiments performed on a patient phantom in a lab setup environment, with a KUKA LWR4+ slave robot and a SIGMA 7 master manipulator. By comparison with previous works, the results show improved performances in terms of the accuracy of the RCM constraint and surgical tip.

- Design and Prototyping of a Bio-Inspired Kinematic Sensing Suit for the Shoulder Joint: Precursor to a Multi-DoF Shoulder Exosuit

    Author: Varghese, Rejin John | Imperial College London
    Author: Lo, Benny Ping Lai | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Medical Robots and Systems; Prosthetics and Exoskeletons; Biomimetics

    Abstract : Soft wearable robots represent a promising new design paradigm for rehabilitation and active assistance applications. Their compliant nature makes them ideal for complex joints, but intuitive control of these robots require robust and compliant sensing mechanisms. In this work, we introduce the sensing framework for a multiple degrees-of-freedom shoulder exosuit capable of sensing the kinematics of the joint. The proposed sensing system is inspired by the body's embodied kinematic sensing, and the organisation of muscles and muscle synergies responsible for shoulder movements. A motion-capture-based evaluation study of the developed framework confirmed conformance with the behaviour of the muscles that inspired its routing. This validation of the tendon-routing hypothesis allows for it to be extended to the actuation framework of the exosuit in the future. The sensor-to-joint-space mapping is based on multivariate multiple regression and derived using an Artificial Neural Network. Evaluation of the derived mapping achieved root mean square error of approximately 5.43<sup>o</sup> and approximately 3.65<sup>o</sup> for the azimuth and elevation joint angles measured over 29,500 frames (4+ minutes) of motion-capture data.

- LaryngoTORS: A Novel Cable-Driven Parallel Robotic System for Transoral Laser Phonosurgery

    Author: Zhao, Ming | Imperial College London
    Author: Oude Vrielink, Timo Joric Corman | Imperial College London
    Author: Kogkas, Alexandros | Imperial College London
    Author: Runciman, Mark | Imperial College London
    Author: Elson, Daniel | Imperial College London
    Author: Mylonas, George | Imperial College London
 
    keyword: Medical Robots and Systems

    Abstract : Transoral laser phonosurgery is a commonly used surgical procedure in which a laser beam is used to perform incision, ablation or photocoagulation of laryngeal tissues. Two techniques are commonly practised: free beam and fiber delivery. For free beam delivery, a laser scanner is integrated into a surgical microscope to provide an accurate laser scanning pattern. This approach can only be used under direct line of sight, which is uncomfortable for the surgeon during prolonged operations, the manipulability is poor and extensive training is required. In contrast, in the fiber delivery technique, a flexible fiber is used to transmit the laser beam and therefore does not require a direct line of sight. However, this can only achieve manual level accuracy, repeatability and velocity and does not allow for pattern scanning. This work presents the LaryngoTORS, a robotic system that aims at overcoming the current limitations of the two techniques by using a cable-driven parallel robotic mechanism for controlling the end tip of the laser fiber. The robotic mechanism is attached at the end of a curved laryngeal blade and allows generation of pattern or free-path scanning. Scanning paths have been performed, exhibiting a root-mean-square error of 0.054�0.028 mm at velocity 0.5 mm/s, with repeatability error of 0.027�0.020 mm. Ex vivo tests on chicken tissue have been carried out. The results have demonstrated the LaryngoTORS is able to overcome limitations of current clinical methods

- Online Disturbance Estimation for Improving Kinematic Accuracy in Continuum Manipulators

    Author: Campisano, Federico | Vanderbilt University
    Author: Remirez, Andria | Vanderbilt University
    Author: Cal�, Simone | University of Leeds
    Author: Chandler, James Henry | University of Leeds
    Author: Obstein, Keith | Vanderbilt University
    Author: Webster III, Robert James | Vanderbilt University
    Author: Valdastri, Pietro | University of Leeds
 
    keyword: Medical Robots and Systems; Modeling, Control, and Learning for Soft Robots; Flexible Robots

    Abstract : Continuum manipulators are flexible robots which undergo continuous deformation as they are actuated. To describe the elastic deformation of such robots, kinematic models have been developed and successfully applied to a large variety of designs and to various levels of constitutive stiffness. Independent of the design, kinematic models need to be calibrated to best describe the deformation of the manipulator. However, even after calibration, unmodeled effects such as friction, nonlinear elastic and/or spatially varying material properties as well as manufacturing imprecision reduce the accuracy of these models. In this paper, we present a method for improving the accuracy of kinematic models of continuum manipulators through the incorporation of orientation sensor feedback. We achieve this through the use of a ``disturbance wrench", which is used to compensate for these unmodeled effects, and is continuously estimated based on orientation sensor feedback as the robot moves through its workspace. The presented method is applied to the HydroJet, a waterjet-actuated soft continuum manipulator, and shows an average a 40% reduction in RMS position and orientation error in the two most common types of kinematic models for continuum manipulators, a Cosserat rod model and a pseudo-rigid body model.

- Permanent Magnet-Based Localization for Growing Robots in Medical Applications

    Author: Watson, Connor | University of California, San Diego
    Author: Morimoto, Tania | University of California San Diego
 
    keyword: Medical Robots and Systems; Surgical Robotics: Steerable Catheters/Needles; Modeling, Control, and Learning for Soft Robots

    Abstract : Growing robots that achieve locomotion by extending from their tip, are inherently compliant and can safely navigate through constrained environments that prove challenging for traditional robots. However, the same compliance and tip-extension mechanism that enables this ability, also leads directly to challenges in their shape estimation and control. In this paper, we present a low-cost, wireless, permanent magnet-based method for localizing the tip of these robots. A permanent magnet is placed at the robot tip, and an array of magneto-inductive sensors is used to measure the change in magnetic field as the robot moves through its workspace. We develop an approach to localization that combines analytical and machine learning techniques and show that it outperforms existing methods. We also measure the position error over a 500 mm x 500 mm workspace with different magnet sizes to show that this approach can accommodate growing robots of different scales. Lastly, we show that our localization method is suitable for tracking the tip of a growing robot by deploying a 12 mm robot through different, constrained environments. Our method achieves position and orientation errors of 3.0+/-1.1~mm and 6.5+/-5.4 degrees in the planar case and 4.3+/-2.3 mm, 3.9+/-3.0 degrees, and 3.8+/-3.5 degrees in the 5-DOF setting.

- An Ergonomic Shared Workspace Analysis Framework for the Optimal Placement of a Compact Master Control Console

    Author: Zhang, Dandan | Imperial College London
    Author: Liu, Jindong | Imperial College London
    Author: Gao, Anzhu | Shanghai Jiao Tong University
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Medical Robots and Systems; Surgical Robotics: Laparoscopy; Human Factors and Human-in-the-Loop

    Abstract : Master-Slave control is commonly used for Robot-Assisted Minimally Invasive Surgery (RAMIS). The configuration, as well as the placement of the master manipulators, can influence the remote control performance. An ergonomic shared workspace analysis framework is proposed in this paper. Combined with the workspace of the master manipulators and the human arms, the human-robot interaction workspace can be generated. The optimal master robot placement can be determined based on three criteria: 1) interaction workspace volume, 2) interaction workspace quality, and 3) intuitiveness for slave robot control. Experimental verification of the platform is conducted on a da Vinci Research Kit (dVRK). An in-house compact master manipulator (Hamlyn CRM) is used as the master robot and the da Vinci robot is used as the slave robot. Comparisons are made between with and without using design optimization to validate the effectiveness of ergonomic shared workspace analysis. Results indicate that the proposed ergonomic shared workspace analysis can improve the performance of teleoperation in terms of task completion time and the number of clutching required during operation.

- Virtual Fixture Assistance for Suturing in Robot-Aided Pediatric Endoscopic Surgery

    Author: Marques Marinho, Murilo | The University of Tokyo
    Author: Ishida, Hisashi | The University of Tokyo
    Author: Harada, Kanako | The University of Tokyo
    Author: Deie, Kyoichi | The University of Tokyo Hospital
    Author: Mitsuishi, Mamoru | The University of Tokyo
 
    keyword: Medical Robots and Systems; Kinematics; Collision Avoidance

    Abstract : The limited workspace in pediatric endoscopic surgery makes surgical suturing one of the most difficult tasks. During suturing, surgeons have to prevent collisions between instruments and also collisions with the surrounding tissues. Surgical robots have been shown to be effective in adult laparoscopy, but assistance for suturing in constrained workspaces has not been yet fully explored. In this letter, we propose guidance virtual fixtures to enhance the performance and the safety of suturing while generating the required task constraints using constrained optimization and Cartesian force feedback. We propose two guidance methods: looping virtual fixtures and a trajectory guidance cylinder, that are based on dynamic geometric elements. In simulations and experiments with a physical robot, we show that the proposed methods increase precision and safety in-vitro.

- Design, Modeling, and Control of a Compact SMA-Actuated MR-Conditional Steerable Neurosurgical Robot

    Author: Shao, Shicong | The Chinese University of Hong Kong
    Author: Sun, Botian | Peking University
    Author: Ding, Qingpeng | The Chinese University of Hong Kong
    Author: Yan, Wanquan | The Chinese University of Hong Kong
    Author: Zheng, Wenjia | The Chinese University of Hong Kong
    Author: Yan, Kim | The Chinese University of Hong Kong
    Author: Hong, Yilun | The Chinese University of Hong Kong
    Author: Cheng, Shing Shin | The Chinese University of Hong Kong
 
    keyword: Medical Robots and Systems; Mechanism Design; Tendon/Wire Mechanism

    Abstract : The paper presents a compact shape memory alloy (SMA)-actuated magnetic resonance (MR)-conditional neurosurgical robotic system. It consists of a 2-degree of freedom (DoF) cable-driven steerable end effector, an antagonistic SMA springs-based actuation module, and a quick-connect module, packaged into a single integrated device measuring 305 mm length - 76 mm diameter. The system is also highly adaptable, such that it could operate a cable-driven end effector up to a maximum of 4-DoFs and its length can be easily modified due to the acrylic plate-based modular construction. In addition to the kinematics of the robotic end effector and the SMA constitutive model, we also present the antagonistic SMA springs model under the known tension and cable displacement from the robotic end effector. We performed extensive characterization experiments to obtain SMA model parameters and integrated a feedforward component in our controller to achieve improved tracking of a sinusoidal reference up to 80� bending angle amplitude and 100 s period. Lastly, proof-of-concept robot demonstrations were performed in a gel phantom and in the MRI that confirmed the robot motion capability in the brain and MRI compatibility of the robot.

- Constrained-Space Optimization and Reinforcement Learning for Complex Tasks

    Author: Tsai, Ya-Yen | Imperial College London
    Author: Xiao, Bo | Imperial College London
    Author: Johns, Edward | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Medical Robots and Systems; Learning from Demonstration

    Abstract : Learning from Demonstration is increasingly used for transferring operator manipulation skills to robots. In practice, it is important to cater for limited data and imperfect human demonstrations, as well as underlying safety constraints. This paper presents a constrained-space optimization and reinforcement learning scheme for managing complex tasks. Through interactions within the constrained space, the reinforcement learning agent is trained to optimize the manipulation skills according to a defined reward function. After learning, the optimal policy is derived from the well-trained reinforcement learning agent, which is then implemented to guide the robot to conduct tasks that are similar to the experts' demonstrations. The effectiveness of the proposed method is verified with a robotic suturing task, demonstrating that the learned policy outperformed the experts' demonstrations in terms of the smoothness of the joint motion and end-effector trajectories, as well as the overall task completion time.

- Automatic Design of Compliant Surgical Forceps with Adaptive Grasping Functions

    Author: Sun, Yilun | Technical University of Munich
    Author: Liu, Yuqing | Technical University of Munich
    Author: Xu, Lingji | Technical University of Munich
    Author: Zou, Yunzhe | Technical Universty of Munich
    Author: Faragasso, Angela | The University of Tokyo
    Author: Lueth, Tim C. | Technical University of Munich
 
    keyword: Medical Robots and Systems; Mechanism Design; Surgical Robotics: Laparoscopy

    Abstract : In this paper, we present a novel method for achieving automatic design of compliant surgical forceps with adaptive grasping functions. Compliant forceps are much easier to assemble and sterilize than conventional rigid-joint forceps, hence their use is spreading from traditional open surgery to robot-assisted minimally invasive applications. However, many compliant forceps still perform stiff grasping, and thus can damage sensitive organs and tissues during the operation. Adaptive grasping function is therefore required for safe manipulation of vulnerable structures. Currently, it is difficult and time consuming to use empirical methods for designing adaptive compliant forceps for different surgical robotic applications. To cope with this problem, we developed a topology-optimization-based method able to synthesize adaptive compliant forceps automatically. Simulation and experimental tests were conducted to evaluate the adaptive grasping function of designed surgical forceps. The results demonstrated that the developed method greatly simplifies the design process and makes it possible to efficiently realize task-specific compliant forceps.

- A Parametric Grasping Methodology for Multi-Manual Interactions in Real-Time Dynamic Simulations

    Author: Munawar, Adnan | Johns Hopkins University
    Author: Srishankar, Nishan | Worcester Polytechnic Institute
    Author: Fichera, Loris | Worcester Polytechnic Institute
    Author: Fischer, Gregory Scott | Worcester Polytechnic Institute, WPI
 
    keyword: Medical Robots and Systems; Contact Modeling; Simulation and Animation

    Abstract : Interactive simulators are used in several important applications which include the training simulators for teleoperated robotic laparoscopic surgery. While state-of-art simulators are capable of rendering realistic visuals and accurate dynamics, grasping is often implemented using kinematic simplification techniques that prevent truly multi-manual manipulation, which is often an important requirement of the actual task. Realistic grasping and manipulation in simulation is a challenging problem due to the constraints imposed by the implementation of rigid-body dynamics and collision computation techniques in state-of-the-art physics libraries. We present a penalty based parametric approach to achieve multi-manual grasping and manipulation of complex objects at arbitrary postures in a real-time dynamic simulation. This approach is demonstrated by accomplishing multi-manual tasks modeled after realistic scenarios, which include the grasping and manipulation of a two-handed screwdriver task and the manipulation of a deformable thread.
- PCA-Based Visual Servoing Using Optical Coherence Tomography

    Author: Dahroug, Bassem | Femto-St
    Author: Tamadazte, Brahim | Univ. Bourgogne Franche-Comt�, CNRS
    Author: Andreff, Nicolas | Université De Franche Comt�
 
    keyword: Medical Robots and Systems; Computer Vision for Medical Robotics; Visual Servoing

    Abstract : This article deals with the development of a vision-based control law to achieve high-accuracy automatic six degrees of freedom (DoF) positioning tasks. The objective of this work is to be able to replace a biological sample under an optical device for a non-invasive depth examination at any given time (i.e., performing repetitive and accurate optical characterizations of the sample). The optical examination, also called optical biopsy, is performed thanks to an optical coherence tomography (OCT) system. The OCT device is used to perform a 3-dimensional optical biopsy, and as a sensor to control the robot motion during the repositioning process. The proposed visual servoing controller uses the 3D pose of the studied biological sample estimated directly from the C-scan OCT images using a Principal Component Analysis (PCA)~framework.<p>The proposed materials and methods were experimentally validated using a spectral-domain OCT and a 6-DoF robotic platform. The obtained results have demonstrated the pertinence of such methods which offer a positioning accuracy around 0.052pm0.03~mm (mean error pm standard deviation) for linear errors and 0.41pm0.16^circ for angular ones over a 8 times 9 times 3.5~mm^3 workspace.

- A Tele-Operated Microsurgical Forceps-Driver with a Variable Stiffness Haptic Feedback Master Device

    Author: Park, Sungwoo | Korea University, Korea Institute of Science and Technology
    Author: Jang, Namseon | Korea Institute of Science and Technology
    Author: Ihn, Yong Seok | Korea Institute of Science and Technology
    Author: Yang, Sungwook | Korea Institute of Science and Technology
    Author: Jeong, Jinwoo | Korea Institute of Science and Technology
    Author: Yim, Sehyuk | KIST
    Author: Oh, Sang-Rok | KIST
    Author: Kim, Keehoon | POSTECH, Pohang University of Science and Technology
    Author: Hwang, Donghyun | Korea Institute of Science and Technology
 
    keyword: Medical Robots and Systems; Haptics and Haptic Interfaces; Grippers and Other End-Effectors

    Abstract : We develop a surgical forceps-driver and a haptic feedback master device that could be applied to a tele-operated microsurgical robotic system for peripheral nerve surgery. Considering that the neurological damage could occur to the peripheral nerve when an excessive force is applied to the nerve tissue, we aim to implement the forceps-driver with precision gripping-force control and the master device capable of providing haptic feedback. For this aim, a high-precision tiny force sensor is fabricated and embedded into the forceps-driver for gripping-force measurement. And, we develop a novel master device handled by surgeon's thumb and forefinger. As a kind of a variable stiffness module, it functions to generate the forceps-drive operation command according to a surgeon's pinching motion and to display haptic feedback by varying the stiffness based on the gripping-force measured from the slave device. Thus, the surgeon can tele-operate the forceps-driver intuitively as well as feel the gripped-object by proprioceptive haptic feedback. In the performance evaluation experiments for the master-slave system, the gripping-force measurement capacity of the forceps-driver and the stiffness variation range of the master device are investigated as about 4 N with a resolution of 0.03 N and about 3.6 N/mm with a resolution of 0.025 N/mm, respectively.

- Hysteresis Compensator with Learning-Based Hybrid Joint Angle Estimation for Flexible Surgery Robots

    Author: Baek, DongHoon | KAIST
    Author: Seo, JuHwan | KAIST(Korea Advanced Institute of Science and Technology)
    Author: Kim, Joonhwan | Korea Advanced Institute of Science and Technology(KAIST)
    Author: Kwon, Dong-Soo | KAIST
 
    keyword: Medical Robots and Systems; Deep Learning in Robotics and Automation; Tendon/Wire Mechanism

    Abstract : Hysteresis causes difficulties in precisely controlling motion of flexible surgery robots and degrades the surgical performance. In order to reduce hysteresis, model-based feed-forward and feedback-based methods using endoscopic cameras have been suggested. However, model-based methods show limited performance when the sheath configuration is deformed. Although feedback-based methods maintain their performance regardless of the changing sheath configuration, these methods are limited in practical situations where the surgical instruments are obscured by surgical debris, such as blood and tissues. In this paper, a hysteresis compensation method using learning-based hybrid joint angle estimation (LBHJAE) is proposed to address both of these situations. This hybrid method combines image-based joint angle estimation (IBJAE) and kinematic-based joint angle estimation (KBJAE) using a Kalman filter. The proposed method can estimate an actual joint angle of a surgical instrument as well as reduce its hysteresis both in the face of partial obscuration and in different sheath configurations. We use a flexible surgery robot, K-FLEX, to evaluate our approach. The results indicate that the proposed method has effective performance in reducing hysteresis.

- Towards FBG-Based Shape Sensing for Micro-Scale and Meso-Scale Continuum Robots with Large Deflection

    Author: Chitalia, Yash | Georgia Institute of Technology
    Author: Deaton, Nancy Joanna | Georgia Institute of Technology
    Author: Jeong, Seokhwan | Georgia Institute of Technology
    Author: Rahman, Nahian | Georgia Institute of Technology
    Author: Desai, Jaydev P. | Georgia Institute of Technology
 
    keyword: Medical Robots and Systems; Surgical Robotics: Steerable Catheters/Needles; Mechanism Design

    Abstract : Endovascular and endoscopic surgical procedures require micro-scale and meso-scale continuum robotic tools to navigate complex anatomical structures. In numerous studies, fiber Bragg grating (FBG) based shape sensing has been used for measuring the deflection of continuum robots on larger scales, but has proved to be a challenge for micro-scale and meso-scale robots with large deflections. In this paper, we have developed a sensor by mounting an FBG fiber within a micromachined nitinol tube whose neutral axis is shifted to one side due to the machining. This shifting of the neutral axis allows the FBG core to experience compressive strain when the tube bends. The fabrication method of the sensor has been explicitly detailed and the sensor has been tested with two tendon-driven micro-scale and meso-scale continuum robots with outer diameters of 0.41 mm and 1.93 mm respectively. The compact sensor allows repeatable and reliable estimates of the shape of both scales of robots with minimal hysteresis. We propose an analytical model to derive the curvature of the robot joints from FBG fiber strain and a static model that relates joint curvature to the tendon force. Finally, as proof-of-concept, we demonstrate the feasibility of our sensor assembly by combining tendon force feedback and the FBG strain feedback to generate reliable estimates of joint angles for the meso-scale robot.

- Agile 3D-Navigation of a Helical Magnetic Swimmer

    Author: Julien, Leclerc | University of Houston
    Author: Zhao, Haoran | University of Houston
    Author: Bao, Daniel | University of Houston
    Author: Becker, Aaron | University of Houston
    Author: Ghosn, Mohamad | Houston Methodist DeBakey Heart and Vascular Center
    Author: Shah, Dipan J. | Houston Methodist DeBakey Heart &amp; Vascular Center
 
    keyword: Medical Robots and Systems

    Abstract : Rotating miniature magnetic swimmers are devices that could navigate within the bloodstream to access remote locations of the body and perform minimally invasive procedures. The rotational movement could be used, for example, to abrade a pulmonary embolus. Some regions, such as the heart, are challenging to navigate. Cardiac and respiratory motions of the heart combined with a fast and variable blood flow necessitate a highly agile swimmer. This swimmer should minimize contact with the walls of the blood vessels and the cardiac structures to mitigate the risk of complications. This paper presents experimental tests of a millimeter-scale magnetic helical swimmer navigating in a blood-mimicking solution and describes its turning capabilities. The step-out frequency and the position error were measured for different values of turn radius. The paper also introduces rapid movements that increase the swimmer's agility and demonstrates these experimentally on a complex 3D trajectory.

- Two Shank-Mounted IMUs-Based Gait Analysis and Classification for Neurological Disease Patients

    Author: Wang, Lei | Zhejiang University
    Author: Sun, Yun | The First Affiliated Hospital of Zhejiang University
    Author: Li, Qingguo | Queen's University
    Author: Liu, Tao | Zhejiang University
    Author: Yi, Jingang | Rutgers University
 
    keyword: Medical Robots and Systems; Human-Centered Automation

    Abstract : Automatic gait measurement and analysis is an enabling tool for intelligent healthcare and robotics-assisted rehabilitation. This paper proposes a novel two shank-mounted inertial measurement units (IMU)-based method on gait analysis and classification for three different neurological diseases. The IMU-based gait analysis and design aims to be applied in personal daily activities and environment for remote diagnosis and rehabilitation guidance. In the design, eight spatial-temporal and kinematic gait parameters are extracted from two shank-mounted IMUs. A support vector machine-based classifier is developed to classify four types of gait patterns with different neurological diseases (healthy control, peripheral neuropathy, post-stroke and Parkinson's disease). A total of 49 subjects are recruited and 93.9% of them are assigned to the right group after the leave-one-subject-out cross validation. The results demonstrate that the proposed IMU-based gait parameters and classifier are capable of differentiating the four types of gait patterns. The analysis and design have great potentials for use in clinical applications.

- An Open-Source Framework for Rapid Development of Interactive Soft-Body Simulations for Real-Time Training

    Author: Munawar, Adnan | Johns Hopkins University
    Author: Srishankar, Nishan | Worcester Polytechnic Institute
    Author: Fischer, Gregory Scott | Worcester Polytechnic Institute, WPI
 
    keyword: Medical Robots and Systems; Simulation and Animation; Software, Middleware and Programming Environments

    Abstract : We present an open-source framework that provides a low barrier to entry for real-time simulation, visualization, and interactive manipulation of user-specifiable soft-bodies, environments, and robots (using a human-readable front-end interface). The simulated soft-bodies can be interacted by a variety of input interface devices including commercially available haptic devices, game controllers, and the Master Tele-Manipulators (MTMs) of the da Vinci Research Kit (dVRK) with real-time haptic feedback. We propose this framework for carrying out multi-user training, user-studies, and improving the control strategies for manipulation problems. In this paper, we present the associated challenges to the development of such a framework and our proposed solutions. We also demonstrate the performance of this framework with examples of soft-body manipulation and interaction with various input devices.

- Towards 5-DoF Control of an Untethered Magnetic Millirobot Via MRI Gradient Coils

    Author: Erin, Onder | Carnegie Mellon University, Max Planck Institute
    Author: Antonelli, Dario | La Sapienza University of Rome
    Author: Tiryaki, Mehmet Efe | Max Plank Institute for Intelligent Systems
    Author: Sitti, Metin | Max-Planck Institute for Intelligent Systems
 
    keyword: Medical Robots and Systems; Micro/Nano Robots; Optimization and Optimal Control

    Abstract : Electromagnetic field gradients generated by magnetic resonance imaging (MRI) devices pave the way to power untethered magnetic robots remotely. This innovative use of MRI devices allows exerting magnetic pulling forces on untethered magnetic robots, which could be used for navigation, diagnosis, drug delivery and therapeutic procedures inside a human body. So far, MRI-powered untethered magnetic robots lack simultaneous position and orientation control inside three-dimensional (3D) fluids, and therefore, their control has been limited to 3-DoF position control. In this paper, we present a path-planning-based 5-DoF control algorithm to steer and control an MRI-powered untethered robot's position and orientation simultaneously in 3D workspaces in fluids. Even though the simulation results show that the proposed optimal controller can successfully control the robot for 5-DoF, in the experiments, we observe a reduced 5-DoF controllability due to the robot manufacturing errors, which result in pitch angle to remain at around the neutral pitching angle at the steady state. The proposed controller was evaluated to track four different paths (linear, planar-horizontal, planar-vertical and 3D paths) generated by 3D Bezier curves. The worst-case path-tracking error was observed for 3D path-following experiments. For this case, the position-tracking error was

- The ARMM System - Autonomous Steering of Magnetically-Actuated Catheters: Towards Endovascular Applications

    Author: Heunis, Christoff Marthinus | University of Twente
    Author: Wotte, Yannik P | University of Twente
    Author: Sikorski, Jakub | University of Twente
    Author: Phillips Furtado, Guilherme | University of Twente
    Author: Misra, Sarthak | University of Twente
 
    keyword: Medical Robots and Systems; Surgical Robotics: Steerable Catheters/Needles; Surgical Robotics: Planning

    Abstract : Positioning conventional endovascular catheters is not without risk, and there is a multitude of complications that are associated with their use in manual surgical interventions. By utilizing surgical manipulators, the efficacy of remote-controlled catheters can be investigated in vivo. However, technical challenges, such as the duration of catheterizations, accurate positioning at target sites, and consistent imaging of these catheters using nonhazardous modalities, still exist. In this paper, we propose the integration of multiple sub-systems in order to extend the clinical feasibility of an autonomous surgical system designed to address these challenges. The system handles the full synchronization of co-operating manipulators that both actuate a clinical tool. The experiments within this study are conducted within a clinically-relevant workspace and inside a gelatinous phantom that represents a life-size human torso. A catheter is positioned using magnetic actuation and proportional-integral (PI) control in conjunction with real-time ultrasound images. Our results indicate an average error between the tracked catheter tip and target positions of 2.09�0.49 mm. The median procedure time to reach targets is 32.6 s. We expect that our system will provide a step towards collaborative manipulators employing mobile electromagnets, and possibly improve autonomous catheterization procedures within endovascular surgeries.

- Automatic Normal Positioning of Robotic Ultrasound Probe Based Only on Confidence Map Optimization and Force Measurement

    Author: Jiang, Zhongliang | Technische Universitat Munchen
    Author: Grimm, Matthias | TU Munich
    Author: Zhou, Mingchuan | Technische Universitét M�nchen
    Author: Esteban, Javier | Technische Universitét M�nchen
    Author: Simson, Walter | Technical University Munich
    Author: Zahnd, Guillaume | TUM
    Author: Navab, Nassir | TU Munich
 
    keyword: Medical Robots and Systems; Force and Tactile Sensing

    Abstract : Acquiring good image quality is one of the main challenges for fully-automatic robot-assisted ultrasound systems (RUSS). The presented method aims at overcoming this challenge for orthopedic applications by optimizing the orientation of the robotic ultrasound (US) probe, i.e. aligning the US probe to the tissue's surface normal at the point of contact in order to improve sound propagation within the tissue. We first optimize the in-plane orientation of the probe by analyzing the confidence map of the US image. We then carry out a fan motion and analyze the resulting forces estimated from joint torques to align the central axis of the probe to the normal within the plane orthogonal to the initial image plane. This results in the final 3D alignment of the probe's main axis with the normal to the anatomical surface at the point of contact without using external sensors for surface reconstruction or localizing the point of contact in an anatomical atlas. The algorithm is evaluated both on a phantom and on human tissues. The mean absolute angular difference (�STD) between true and estimated normal on stationary phantom, forearm, upper arm and lower back was 3.1 �1.0, 3.7�1.7, 5.3�1.3 and 6.9�3.5. In comparison, six human operators obtained errors of 3.2�1.7 deg on the phantom. Hence the method is able to automatically position the probe normal to the scanned tissue at the point of contact and the point of contact and thus improve the quality of automatically acquired US images.

- A Semi-Autonomous Stereotactic Brain Biopsy Robot with Enhanced Safety

    Author: Minxin, Ye | The Chinese University of HongKong
    Author: Li, Weibing | The Chinese University of Hong Kong
    Author: Chan, Tat-Ming | Prince of Wales Hospital
    Author: Chiu, Philip, Wai-yan | Chinese University of Hong Kong
    Author: Li, Zheng | The Chinese University of Hong Kong
 
    keyword: Medical Robots and Systems; Surgical Robotics: Steerable Catheters/Needles; Kinematics

    Abstract : In stereotactic brain biopsy, operating the needle accurately and taking the biopsy specimen safely are two major challenges for ensuring the success of the surgical procedure. Considering this fact, surgical robots offering high accuracy and precision have been developed for neurosurgery including brain biopsy. Typical brain biopsy robots are only commanded to adjust the needle's pose before inserting the needle manually by a neurosurgeon. In the literature, there exists no robotic system that is competent to complete the needle insertion task autonomously. To move a step forward, a novel biopsy module for brain biopsy is first designed and fabricated in this paper. The biopsy module can be automated to complete a series of tasks such as inserting the needle, generating and controlling the aspiration pressure for specimen acquisition, and rotating the cannula for side-cutting. The biopsy module is further integrated with a cost-effective and lightweight UR5 robot and an optical tracking system to improve the autonomy, leading to a stereotactic neuronavigation system. Kinematic relationships of the involved elements are established via a calibration process. A quadratic programming based approach equipped with a virtual potential field method is implemented to safely control the robot with joint-limit avoidance and obstacle avoidance capabilities. The experimentation of the brain biopsy robot is performed, demonstrating that the developed robotic system has potential applicab

- Magnetically Steered Robotic Insertion of Cochlear-Implant Electrode Arrays: System Integration and First-In-Cadaver Results

    Author: Bruns, Trevor | Vanderbilt University
    Author: Riojas, Katherine | Vanderbilt University
    Author: Ropella, Dominick | Vanderbilt University
    Author: Cavilla, Matt | University of Utah
    Author: Petruska, Andrew J. | Colorado School of Mines
    Author: Freeman, Michael | Vanderbilt University Medical Center
    Author: Labadie, Robert F | Vanderbilt University
    Author: Abbott, Jake | University of Utah
    Author: Webster III, Robert James | Vanderbilt University
 
    keyword: Medical Robots and Systems

    Abstract : Cochlear-implant electrode arrays (EAs) must be inserted accurately and precisely to avoid damaging the delicate anatomical structures of the inner ear. It has previously been shown on the benchtop that using magnetic fields to steer magnet-tipped EAs during insertion reduces insertion forces, which correlate with insertion errors and damage to internal cochlear structures. This paper presents several advancements toward the goal of deploying magnetic steering of cochlear-implant EAs in the operating room. In particular, we integrate image guidance with patient-specific insertion vectors, we incorporate a new nonmagnetic insertion tool, and we use an electromagnetic source, which provides programmable control over the generated field. The electromagnet is safer than prior permanent-magnet approaches in two ways: it eliminates motion of the field source relative to the patient's head and creates a field-free source in the power-off state. Using this system, we demonstrate system feasibility by magnetically steering EAs into a cadaver cochlea for the first time. We show that magnetic steering decreases average insertion forces, in comparison to manual insertions and to image-guided robotic insertions alone.

- Magnetic Sensor Based Topographic Localization for Automatic Dislocation of Ingested Button Battery

    Author: Liu, Jialun | The University of Sheffield
    Author: Sugiyama, Hironari | Nagaoka University of Technology
    Author: Nakayama, Tadachika | Nagaoka University of Technology
    Author: Miyashita, Shuhei | University of Sheffield
 
    keyword: Medical Robots and Systems; Localization; Micro/Nano Robots

    Abstract : A button battery accidentally ingested by a toddler or small child can cause severe damage to the stomach within a short period of time. Once a battery lands on the surface of the esophagus or stomach, it can run a current in the tissue and induce a chemical reaction resulting in injury. Following our previous work where we presented an ingestible magnetic robot for button battery retrieval, this study presents a remotely achieved novel localization method of a button battery with commonly available magnetic sensors (Hall-effect sensors). By applying a direct magnetic field to the button battery using an electromagnetic coil, the battery is magnetized, and hence it becomes able to be sensed by Hall-effect sensors. Using a trilateration method, we were able to detect the locations of an LR44 button battery and other ferromagnetic materials at variable distances. Additional four electromagnetic coils were used to autonomously navigate a magnet-containing capsule to dislocate the battery from the affected site.

- A Fully Actuated Body-Mounted Robotic Assistant for MRI-Guided Low Back Pain Injection

    Author: Li, Gang | Johns Hopkins University
    Author: Patel, Niravkumar | Johs Hopkins University
    Author: Liu, Weiqiang | Johns Hopkins University
    Author: Wu, Di | Johns Hopkins University
    Author: Sharma, Karun | Sheikh Zayed Institute for Pediatric Surgical Innovation, Childr
    Author: Cleary, Kevin | Children's National Medical Center
    Author: Fritz, Jan | John Hopkins
    Author: Iordachita, Ioan Iulian | Johns Hopkins University
 
    keyword: Medical Robots and Systems; Mechanism Design

    Abstract : This paper reports the development of a fully actuated body-mounted robotic assistant for MRI-guided low back pain injection. The robot is designed with a 4-DOF needle alignment module and a 2-DOF remotely actuated needle driver module. The 6-DOF fully actuated robot can operate inside the scanner during imaging; hence, minimizing the need of moving the patient in or out of the scanner during the procedure, and thus potentially reducing the procedure time and streamlining the workflow. The robot is built with a lightweight and compact profile that could be mounted directly on the patient's lower back via straps; therefore, attenuating the effect of patient motion by moving with the patient. The novel remote actuation design of the needle driver module with beaded chain transmission can reduce the weight and profile on the patient, as well as minimize the imaging degradation caused by the actuation electronics. The free space positioning accuracy of the system was evaluated with an optical tracking system, demonstrating the mean absolute errors (MAE) of the tip position to be 0.99�0.46mm and orientation to be 0.99�0.65&#9702;. Qualitative imaging quality evaluation was performed on a human volunteer, revealing minimal visible image degradation that should not affect the procedure. The mounting stability of the system was assessed on a human volunteer, indicating the 3D position variation of target movement with respect to the robot frame to be less than 0.7mm.

- Fault Tolerant Control in Shape-Changing Internal Robots

    Author: Balasubramanian, Lavanya | University of Sheffield
    Author: Wray, Tom | University of Sheffield
    Author: Damian, Dana | University of Sheffield
 
    keyword: Medical Robots and Systems; Flexible Robots; Failure Detection and Recovery

    Abstract : It is known that the interior of the human body is one of the most adverse environments for a foreign body, such as an in vivo robot, and vice-versa. As robots operating in vivo are increasingly recognized for their capabilities and potential for improved therapies, it is important to ensure their safety, especially for long term treatments when little supervision can be provided. We introduce an implantable robot that is flexible, extendable and symmetric, thus changing shape and size. This design allows the implementation of an effective fault tolerant control, with features such as physical polling for fault diagnosis, retraction and redundancy-based control switching at fault. We demonstrate the fault-tolerant capabilities for an implantable robot that elongates tubular tissues by applying tension to the tissue. In benchtop tests, we show a reduction of the fault risks by at least 83%. The study provides a valuable methodology to enhance safety and efficacy of implantable and surgical robots, and thus to accelerate their adoption.

- Evaluation of Increasing Camera Baseline on Depth Perception in Surgical Robotics

    Author: Avinash, Apeksha | University of British Columbia
    Author: Abdelaal, Alaa Eldin | University of British Columbia
    Author: Salcudean, Septimiu E. | University of British Columbia
 
    keyword: Medical Robots and Systems; Telerobotics and Teleoperation; Performance Evaluation and Benchmarking

    Abstract : In this paper, we evaluate the effect of increasing camera baselines on depth perception in robot-assisted surgery. Restricted by the diameter of the surgical trocar through which they are inserted, current clinical stereo endoscopes have a fixed baseline of 5.5 mm. To overcome this restriction, we propose using a stereoscopic "pickup" camera with a side-firing design that allows for larger baselines. We conducted a user study with baselines of 10 mm, 15 mm, 20 mm, and 30 mm to evaluate the effect of increasing baseline on depth perception when used with the da Vinci surgical system. Subjects (N=28) were recruited and asked to rank differently sized poles, mounted at a distance of 200 mm from the cameras, according to their increasing order of height when viewed under different baseline conditions. The results showed that subjects performed better as the baseline was increased with the best performance at a 20 mm baseline. This preliminary proof-of-concept study shows that there is opportunity to improve depth perception in robot-assisted surgical systems with a change in endoscope design philosophy. In this paper, we present this change with our side-firing "pickup" camera and its flexible baseline design. Ultimately, this serves as the first step towards an adaptive baseline camera design that maximizes depth perception in surgery.

- Toward Autonomous Robotic Micro-Suturing Using Optical Coherence Tomography Calibration and Path Planning

    Author: Tian, Yuan | Duke University
    Author: Draelos, Mark | Duke University
    Author: Tang, Gao | University of Illinois at Urbana-Champaign
    Author: Qian, Ruobing | Duke University
    Author: Kuo, Anthony | Duke University
    Author: Izatt, Joseph | Duke University
    Author: Hauser, Kris | University of Illinois at Urbana-Champaign
 
    keyword: Medical Robots and Systems; Calibration and Identification

    Abstract : Robotic automation has the potential to assist human surgeons in performing suturing tasks in microsurgery, and in order to do so a robot must be able to guide a needle with sub-millimeter precision through soft tissue. This paper presents a robotic suturing system that uses 3D optical coherence tomography (OCT) system for imaging feedback. Calibration of the robot-OCT and robot-needle transforms, wound detection, keypoint identification, and path planning are all performed automatically. The calibration method handles pose uncertainty when the needle is grasped using a variant of iterative closest points. The path planner uses the identified wound shape to calculate needle entry and exit points to yield an evenly-matched wound shape after closure. Experiments on tissue phantoms and animal tissue demonstrate that the system can pass a suture needle through wounds with 0.194mm overall accuracy in achieving the planned entry and exit points.

- Improved Multiple Objects Tracking Based Autonomous Simultaneous Magnetic Actuation &amp; Localization for WCE

    Author: Xu, Yangxin | The Chinese University of Hong Kong
    Author: Li, Keyu | The Chinese University of Hong Kong
    Author: Zhao, Ziqi | The Chinese University of Hong Kong
    Author: Meng, Max Q.-H. | The Chinese University of Hong Kong
 
    keyword: Medical Robots and Systems

    Abstract : Wireless Capsule Endoscopy (WCE) has the advantage of reducing the invasiveness and pain of gastrointestinal examinations. In this work, we propose a system aimed at autonomously accelerating and locating the WCE inside the intestine for clinical applications. A rotating magnet controlled by a robotic arm is placed outside the patient's body to actuate the capsule with an internal magnetic ring, and the magnetic fields of the two sources are measured by an external sensor array. The original Multiple Objects Tracking method is improved by combining Normal Vector Fitting, Bezier Curve Gradient, and Spherical Linear Interpolation to estimate the 6-D pose of the WCE from a 5-D pose sequence. In order to close the actuation-localization loop, a strategy is presented to react to different states of the capsule. The proposed method is validated via experiments on phantoms as well as on animal intestines. The localization of the capsule shows an accuracy of 3.5mm in position and 9.4 degrees in orientation, and the average update frequency of the estimated 6-D pose reaches 25Hz.
- Towards Bimanual Vein Cannulation: Preliminary Study of a Bimanual Robotic System with a Dual Force Constraints Controller

    Author: He, Changyan | Beihang University
    Author: Ebrahimi, Ali | Johns Hopkins University
    Author: Yang, Emily | Johns Hopkins University
    Author: Urias, Muller | Wilmer Eye Institute
    Author: Yang, Yang | Beijing University of Aerosnautics and Astronautics
    Author: Gehlbach, Peter | Johns Hopkins Medical Institute
    Author: Iordachita, Ioan Iulian | Johns Hopkins University
 
    keyword: Medical Robots and Systems; Force Control; Motion Control of Manipulators

    Abstract : Retinal vein cannulation is a promising approach for treating retinal vein occlusion. The approach remains largely unexploited clinically due to surgeon limitations in detecting interaction forces between surgical tools and retinal tissue. In this paper, a dual force constraint controller for robot-assisted retinal surgery was presented to keep the tool-to-vessel forces and tool-to-sclera forces below prescribed thresholds. A cannulation tool and forceps with dual force-sensing capability were developed and used to measure force information fed into the robot controller, which was implemented on existing Steady Hand Eye Robot platforms. The robotic system facilitates retinal vein cannulation by allowing a user to grasp the target vessel with the forceps and then enter the vessel with the cannula. The system was evaluated on an eye phantom. The results showed that, while the eyeball was subjected to rotational disturbances, the proposed controller actuates the robotic manipulators to maintain the average tool-to-vessel force at 10.9 mN and 13.1 mN and the average tool-to-sclera force at 38.1 mN and 41.2 mN for the cannula and the forcpes, respectively. Such small tool-to-tissue forces are acceptable to avoid retinal tissue injury. Additionally, two clinicians participated in a preliminary user study of the bimanual cannulation demonstrating that the operation time and tool-to-tissue forces are significantly decreased with the robotic system as compared to freehand performance.

- Evaluation of a Combined Grip of Pinch and Power Grips in Manipulating a Master Manipulator

    Author: Jeong, Solmon | Tokyo Institute of Technology
    Author: Tadano, Kotaro | Tokyo Institute of Technology
 
    keyword: Medical Robots and Systems; Human-Centered Robotics

    Abstract : In conventional robotic surgery, the manipulating methods exhibit limitations that are strongly related to the advantages and disadvantages of a pinch grip and power grip. The context of this study is focused on the introduction of a combined grip to compensate for such restraints. In particular, this study proposed the combined-grip-handle scheme on a master manipulator. In this framework, the position of fingertips was designed to be adjustable in distance and direction to allow for a pinch grip motion around the holding axis of a power grip motion. A ring-bar experiment applying the master-slave scheme was conducted with the master manipulator under several manipulating conditions of the combined grip and the conventional gripping types. Results for using the combined grip demonstrated that the combined grip showed better performance on the positioning operation, compared with the conventional gripping types

- Contact Stability Analysis of Magnetically-Actuated Robotic Catheter under Surface Motion

    Author: Hao, Ran | Case Western Reserve University
    Author: Greigarn, Tipakorn | Case Western Reserve University
    Author: Cavusoglu, M. Cenk | Case Western Reserve University
 
    keyword: Medical Robots and Systems; Surgical Robotics: Steerable Catheters/Needles

    Abstract : Contact force quality is one of the most critical factors for safe and effective lesion formation during cardiac ablation. The contact force and contact stability plays important roles in determining the lesion size and creating a gap-free lesion. In this paper, the contact stability of a novel magnetic resonance imaging (MRI)-actuated robotic catheter under tissue surface motion is studied. The robotic catheter is modeled using a pseudo-rigid-body model, and the contact model under surface constraint is provided. Two contact force control schemes to improve the contact stability of the catheter under heart surface motions are proposed and their performance are evaluated in simulation.

- Fast and Accurate Intracorporeal Targeting through an Anatomical Orifice Exhibiting Unknown Behavior

    Author: Chalard, R�mi | Université Pierre Et Marie Curie (UPMC)
    Author: Reversat, David | Université Pierre Et Marie Curie
    Author: Morel, Guillaume | Sorbonne Université, CNRS, INSERM
    Author: Vitrani, Marie-Aude | Univ. Pierre Et Marie Curie - Paris6
 
    keyword: Medical Robots and Systems; Robust/Adaptive Control of Robotic Systems; Surgical Robotics: Laparoscopy

    Abstract : Surgery may involve precise instrument tip positioning in a minimally invasive way. During these operations, the instrument is inserted in the body through an orifice. The movements of the instrument are constrained by interaction forces arising at the orifice level. The physical constraints may drastically vary depending on the patient's anatomy. This introduces uncertainties that challenge the positioning task for a robot. Indeed, it raises an antagonism: On one side, the required precision appeals for a rigid behavior. On the other side, forces applied at the entry point should be limited, which requires softness. In this paper we choose to minimize forces at the orifice by using a passive ball joint wrist to manipulate the instrument. From a control perspective, this leads to consider the task as a 3 DOF wrist center positioning problem, whose softness can be achieved through conventional low impedance control. However, positioning the wrist center, even with a high static precision, does not allow to achieve a high precision of the instrument tip positioning when the orifice behavior is not known. To cope with this problem, we implement a controller that servos the tip position by commanding the wrist position. In order to deal with uncertainties, we exploit an adaptive control scheme that identifies in real-time the unknown mapping between the wrist velocity and the tip velocity. Both simulations and in vitro experimental results show the efficiency of the control law.

- Robotic Swarm Control for Precise and On-Demand Embolization

    Author: Luo, Mengxi | University of Toronto
    Author: Law, Junhui | University of Toronto
    Author: Wang, Xian | University of Toronto
    Author: Xin, Liming | University of Toronto
    Author: Shan, Guanqiao | University of Toronto
    Author: Badiwala, Mitesh | University of Toronto
    Author: Huang, Xi | The Hospital for Sick Children (SickKids)
    Author: Sun, Yu | University of Toronto
 
    keyword: Medical Robots and Systems; Micro/Nano Robots; Automation in Life Sciences: Biotechnology, Pharmaceutical and Health Care

    Abstract : Existing approaches for robotic control of magnetic swarms are not capable of generating magnetic aggregates precisely in an arbitrarily specified target region in a fluidic flow environment. Such a swarm control capability is demanded by medical applications such as clinical embolization (i.e., localized clogging of blood vessels). This paper presents a new magnetic swarm control strategy to generate aggregates only in a specified target region under fluidic flow. Within the target region, the magnetic field generates sufficiently large magnetic forces among magnetic particles to maintain the aggregates' integrity at the junctions of blood vessels. In contrast, unintended aggregates outside the target region are disassembled by fluidic shear. The aggregation control approach achieved a mean absolute error of 0.15 mm in positioning a target region and a mean absolute error of 0.30 mm in controlling the target region's radius. With thrombin coating, 1 �m magnetic particles were controlled to perform embolization both in vitro (using microfluidic channel networks) and ex vivo (using porcine tissue). Experiments proved the effectiveness of the swarm control technique for on-demand, targeted embolization.

- Bilateral Teleoperation Control of a Redundant Manipulator with an RCM Kinematic Constraint

    Author: Su, Hang | Politecnico Di Milano
    Author: Schmirander, Yunus | Politecnico Di Milano
    Author: Li, Zhijun | University of Science and Technology of China
    Author: Zhou, Xuanyi | Central South University
    Author: Ferrigno, Giancarlo | Politecnico Di Milano
    Author: De Momi, Elena | Politecnico Di Milano
 
    keyword: Medical Robots and Systems; Telerobotics and Teleoperation; Haptics and Haptic Interfaces

    Abstract : In this paper, a bilateral teleoperation control of a serial robot manipulator, which guarantees a Remote Center of Motion (RCM) constraint in its kinematic level, is developed. A two-layered approach based on the energy tank model is proposed to achieve haptic feedback on the end effector with a pedal switch. The redundancy of the manipulator is exploited to maintain the RCM constraint using the decoupled Cartesian Admittance Control. Transparency and stability of the proposed bilateral teleoperation are demonstrated using a KUKA LWR4+ serial robot and a Sigma 7 haptic manipulator with an RCM constraint in augmented reality. The results prove that the control can achieve not only the bilateral teleoperation but also maintain the RCM constraint.

## Legged Robots

- An Open Torque-Controlled Modular Robot Architecture for Legged Locomotion Research

    Author: Grimminger, Felix | Max Planck Institute for Intelligent Systems
    Author: Meduri, Avadesh | New York University
    Author: Khadiv, Majid | Max Planck Institute for Intelligent Systems
    Author: Viereck, Julian | Max Planck Institute for Intelligent Systems
    Author: W�thrich, Manuel | Max-Planck-Institute for Intelligent Systems
    Author: Naveau, Maximilien | LAAS/CNRS
    Author: Berenz, Vincent | Max Planck Institute for Intelligent Systems
    Author: Heim, Steve | Max Planck Institute for Intelligent Systems
    Author: Widmaier, Felix | Max-Planck Institute for Intelligent Systems
    Author: Flayols, Thomas | LAAS, CNRS
    Author: Fiene, Jonathan | Max Planck Institute for Intelligent Systems
    Author: Badri-Spröwitz, Alexander | Max Planck Institute for Intelligent Systems
    Author: Righetti, Ludovic | New York University
 
    keyword: Legged Robots; Compliance and Impedance Control; Mechanism Design

    Abstract : We present a new open-source torque-controlled legged robot system, with a low-cost and low-complexity actuator module at its core. It consists of a high-torque brushless DC motor and a low-gear-ratio transmission suitable for impedance and force control. We also present a novel foot contact sensor suitable for legged locomotion with hard impacts. A 2.2 kg quadruped robot with a large range of motion is assembled from eight identical actuator modules and four lower legs with foot contact sensors. Leveraging standard plastic 3D printing and off-the-shelf parts results in a lightweight and inexpensive robot, allowing for rapid distribution and duplication within the research community. We systematically characterize the achieved impedance at the foot in both static and dynamic scenarios, and measure a maximum dimensionless leg stiffness of 10.8 without active damping, which is comparable to the leg stiffness of a running human. Finally, to demonstrate the capabilities of the quadruped, we present a novel controller which combines feedforward contact forces computed from a kino-dynamic optimizer with impedance control of the center of mass and base orientation. The controller can regulate complex motions while being robust to environmental uncertainty.

- Passive Quadrupedal Gait Synchronization for Extra Robotic Legs Using a Dynamically Coupled Double Rimless Wheel Model

    Author: Gonzalez, Daniel | United States Military Academy at West Point
    Author: Asada, Harry | MIT
 
    keyword: Passive Walking; Physical Human-Robot Interaction; Dynamics

    Abstract : The Extra Robotic Legs (XRL) system is a robotic augmentation worn by a human operator consisting of two articulated robot legs that walk with the operator and help bear a heavy backpack payload. It is desirable for the Human-XRL quadruped system to walk with the rear legs lead the front by 25% of the gait period, minimizing the energy lost from foot impacts while maximizing balance stability. Unlike quadrupedal robots, the XRL cannot command the human's limbs to coordinate quadrupedal locomotion. Using a pair of Rimless Wheel models, it is shown that the systems coupled with a spring and damper converge to the desired 25% phase difference. A Poincare return map was generated using numerical simulation to examine the convergence properties to different coupler design parameters, and initial conditions. The Dynamically Coupled Double Rimless Wheel system was physically realized with a spring and dashpot chosen from the theoretical results, and initial experiments indicate that the desired synchronization properties may be achieved within several steps using this set of passive components alone.

- Optimal Fast Entrainment Waveform for Indirectly Controlled Limit Cycle Walker against External Disturbances

    Author: Li, Longchuan | Ritsumeikan University
    Author: Tokuda, Isao | Ritsumeikan University
    Author: Asano, Fumihiko | Japan Advanced Institute of Science and Technology
 
    keyword: Legged Robots; Underactuated Robots; Dynamics

    Abstract : After occasional perturbation, it is crucial to spontaneously control the limit cycle walking so that it quickly returns to its closed orbit in phase space. Otherwise, its stability can not be sufficiently guaranteed if the speed of recovery is slow while successive perturbation is applied. The accumulated deviation may eventually drive the phase outside the basin of attraction, leading to failure of the walking. In this sense, a control law that quickly recovers the disturbed phase before encountering the following perturbations is indispensable. With this consideration, here we analytically derive an optimal fast entrainment waveform that maximizes the speed of phase recovery based on phase reduction theory. Our theoretical method is numerically evaluated using a limit cycle walker, which is indirectly controlled by the oscillation of a wobbling mass via entrainment effect. The obtained waveform is used as the desired trajectory of the wobbling motion. The simulation results show that the waveform we derived achieves the best performance among all candidates. Our method helps to enhance the stability of limit cycle walking.

- GaitMesh: Controller-Aware Navigation Meshes for Long-Range Legged Locomotion Planning in Multi-Layered Environments

    Author: Brandao, Martim | University of Oxford
    Author: Aladag, Omer Burak | Sabanci University
    Author: Havoutis, Ioannis | University of Oxford
 
    keyword: Legged Robots; Autonomous Vehicle Navigation; Motion and Path Planning

    Abstract : Long-range locomotion planning is an important problem for the deployment of legged robots to real scenarios. Current methods used for legged locomotion planning often do not exploit the flexibility of legged robots, and do not scale well with environment size. In this paper we propose the use of navigation meshes for deployment in large-scale multi-floor sites. We leverage this representation to improve long-term locomotion plans in terms of success rates, path costs and reasoning about which gait-controller to use when. We show that NavMeshes have higher planning success rates than sampling-based planners, but are 400x faster to construct and at least 100x faster to plan with. The performance gap further increases when considering multi-floor environments. We present both a procedure for building controller-aware NavMeshes and a full navigation system that adapts to changes to the environment. We demonstrate the capabilities of the system in simulation experiments and in field trials at a real-world oil rig facility.

- Mechanical Shock Propagation Reduction in Robot Legs

    Author: Singh, Bajwa Roodra Pratap | Italian Institute of Technology
    Author: Featherstone, Roy | Istituto Italiano Di Tecnologia
 
    keyword: Legged Robots; Mechanism Design; Dynamics

    Abstract : This paper shows how the mass distribution in a robot's leg affects the propagation of mechanical shocks from the foot to the torso. An example is given of a leg design that propagates no shock at all; and a formula is given for the propagation of shock in a general robot leg, modelled as a chain of rigid bodies, assuming that the foot makes a point contact when it strikes the ground.

- Guided Constrained Policy Optimization for Dynamic Quadrupedal Robot Locomotion

    Author: Gangapurwala, Siddhant | University of Oxford
    Author: Mitchell, Alexander Luis | University of Oxford
    Author: Havoutis, Ioannis | University of Oxford
 
    keyword: Legged Robots; Deep Learning in Robotics and Automation; Control Architectures and Programming

    Abstract : Deep reinforcement learning (RL) uses model-free techniques to optimize task-specific control policies. Despite having emerged as a promising approach for complex problems, RL is still hard to use reliably for real world applications. Apart from challenges such as precise reward function tuning, inaccurate sensing and actuation, and non-deterministic response, existing RL methods do not guarantee behavior within required safety constraints that are crucial for real robot scenarios. In this regard, we introduce guided constrained policy optimization (GCPO), an RL framework based upon our implementation of constrained proximal policy optimization (CPPO) for tracking base velocity commands while following the defined constraints. We also introduce schemes which encourage state recovery into constrained regions in case of constraint violations. We present experimental results of our training method and test it on the real ANYmal quadruped robot. We compare our approach against the unconstrained RL method and show that guided constrained RL offers faster convergence close to the desired optimum resulting in an optimal, yet physically feasible, robotic control behavior without the need for precise reward function tuning.

- MPC-Based Controller with Terrain Insight for Dynamic Legged Locomotion

    Author: Villarreal Maga�a, Octavio Antonio | Istituto Italiano Di Tecnologia
    Author: Barasuol, Victor | Istituto Italiano Di Tecnologia
    Author: Wensing, Patrick M. | University of Notre Dame
    Author: Caldwell, Darwin G. | Istituto Italiano Di Tecnologia
    Author: Semini, Claudio | Istituto Italiano Di Tecnologia
 
    keyword: Legged Robots; Optimization and Optimal Control; Dynamics

    Abstract : We present a novel control strategy for dynamic legged locomotion in complex scenarios, that considers information about the morphology of the terrain in contexts when only on-board mapping and computation are available. The strategy is built on top of two main elements: first a contact sequence task that provides safe foothold locations based on a convolutional neural network to perform fast and continuous evaluation of the terrain in search of safe foothold locations; then a model predictive controller that considers the foothold locations given by the contact sequence task to optimize target ground reaction forces. We assess the performance of our strategy through simulations of the hydraulically actuated quadruped robot HyQReal traversing rough terrain under realistic on-board sensing conditions.

- An Adaptive Supervisory Control Approach to Dynamic Locomotion under Parametric Uncertainty

    Author: Chand, Prem | University of Delaware
    Author: Veer, Sushant | Princeton University
    Author: Poulakakis, Ioannis | University of Delaware
 
    keyword: Legged Robots; Robust/Adaptive Control of Robotic Systems; Humanoid and Bipedal Locomotion

    Abstract : This paper presents an adaptive control scheme for robotic systems that operate in the face of--potentially large--structured uncertainty. The proposed adaptive controller employs an on-line supervisor that utilizes logic-based switching among a finite set of controllers to identify uncertain parameters, and adapt the behavior of the system based on a current estimate of their value. To achieve this, the adaptive control approach in this paper combines on-line parameter estimation and feedback control while avoiding some of the inherent difficulties of classical adaptive control strategies. Furthermore, the proposed supervisory control architecture is modular as it relies on established `off-the-shelf' feedback control law and estimator design approaches, instead of customizing the overall design to the specific requirements of an adaptive control algorithm. We demonstrate the efficacy of the method on the problem of a dynamically-walking bipedal robot delivering a payload of unknown mass, and show that, by switching to the controller that is the `best' according to a current estimate of the uncertainty, the system maintains a low energy cost during its operation.

- Joint Space Position/Torque Hybrid Control of the Quadruped Robot for Locomotion and Push Reaction

    Author: Sim, Okkee | KAIST
    Author: Jeong, Hyobin | KAIST
    Author: Oh, Jaesung | KAIST
    Author: Lee, Moonyoung | Korea Advanced Institute of Science and Technology
    Author: Lee, Kang Kyu | KAIST Hubolab
    Author: Park, Hae-Won | Korea Advanced Institute of Science and Technology
    Author: Oh, Jun Ho | Korea Advanced Inst. of Sci. and Tech
 
    keyword: Legged Robots; Motion Control; Dynamics

    Abstract : This paper proposes a novel algorithm for joint space position/torque hybrid control of a mammal-type quadruped robot. With this control algorithm, the robot demonstrated both dynamic locomotion and push reaction abilities without the need for torque control in the ab/ad joints. Based on the tipping and slipping condition of the legged robot, we showed that reaction to a typical push in the horizontal direction does not require full contact-force-control in the frontal plane. Furthermore, we showed that position/torque hybrid control in Cartesian space is directly applicable to joint space hybrid control due to the joint configuration of the quadruped robot. We conducted experiments on our legged robot platform to verify the performance of our hybrid control algorithm. With this approach, the robot displayed stability while walking and reacting to external push disturbances.

- Improved Performance on Moving-Mass Hopping Robots with Parallel Elasticity

    Author: Ambrose, Eric | California Institute of Technology
    Author: Ames, Aaron | Caltech
 
    keyword: Legged Robots

    Abstract : Robotic Hopping is challenging from the perspective of both modeling the dynamics as well as the mechanical design due to the short period of ground contact in which to actuate on the world. Previous work has demonstrated stable hopping on a moving-mass robot, wherein a single spring was utilized below the body of the robot. This paper finds that the addition of a spring in parallel to the actuator greatly improves the performance of moving mass hopping robots. This is demonstrated through the design of a novel one-dimensional hopping robot. For this robot, a rigorous trajectory optimization method is developed using hybrid systems models with experimentally tuned parameters. Simulation results are used to study the effects of a parallel spring on energetic efficiency, stability, and hopping effort. We find that the double-spring model had 2.5x better energy efficiency than the single-spring model, and was able to hop using 40% less peak force from the actuator. Furthermore, the double-spring model produces stable hopping without the need for stabilizing controllers. These concepts are demonstrated experimentally on a novel hopping robot, wherein hop heights up to 40cm were achieved with comparable efficiency and stability.

- Vision Aided Dynamic Exploration of Unstructured Terrain with a Small-Scale Quadruped Robot

    Author: Kim, Donghyun | Massachusetts Institute of Technology
    Author: Carballo, Daniel | MIT
    Author: Di Carlo, Jared | Massachusetts Institute of Technology
    Author: Katz, Benjamin | Massachusetts Institute of Technology
    Author: Bledt, Gerardo | Massachusetts Institute of Technology (MIT)
    Author: Lim, Bryan Wei Tern | Massachusetts Institute of Technology
    Author: Kim, Sangbae | Massachusetts Institute of Technology
 
    keyword: Legged Robots; Autonomous Vehicle Navigation; Underactuated Robots

    Abstract : Legged robots have been highlighted as promising mobile platforms for disaster response and rescue scenar- ios because of their rough terrain locomotion capability. In cluttered environments, small robots are desirable as they can maneuver through small gaps, narrow paths, or tunnels. However small robots have their own set of difficulties such as limited space for sensors, limited obstacle clearance, and scaled- down walking speed. In this paper, we extensively address these difficulties via effective sensor integration and exploitation of dynamic locomotion and jumping. We integrate two Intel RealSense sensors into the MIT Mini-Cheetah, a 0.3 m tall, 9 kg quadruped robot. Simple and effective filtering and evaluation algorithms are used for foothold adjustment and obstacle avoidance. We showcase the exploration of highly irregular terrain using dynamic trotting and jumping with the small- scale, fully sensorized Mini-Cheetah quadruped robot.

- Reactive Support Polygon Adaptation for the Hybrid Legged-Wheeled CENTAURO Robot

    Author: Kamedula, Malgorzata | Istituto Italiano Di Tecnologia
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
 
    keyword: Legged Robots; Wheeled Robots; Motion Control

    Abstract : The need for robots operating in the real-world has sparked interest in hybrid locomotion that combines the versatile legged mobility with a simpler wheeled motion. The use of legged-wheeled robots in complex real-world scenarios requires controllers that will capitalise on this flexibility.<p>In this work, a reactive control scheme that exploits wheels steering and robot articulated legs is proposed to continuously adjust the robot support polygon (SP) in response to unknown disturbances. The designed cartesian-space control is expressed in the joint-space to account for the hardware limits. To tackle the non-holonomy in the joint-space model, the linear velocity/acceleration-based model is developed for the general legged-wheeled platform and applied to resolve the SP adaptation of a platform with steerable wheels. The proposed control is experimentally verified on the CENTAURO robot, demonstrating the SP adjustment when external disturbances are applied.


- Reliable Trajectories for Dynamic Quadrupeds Using Analytical Costs and Learned Initializations

    Author: Melon, Oliwier Aleksander | University of Oxford
    Author: Geisert, Mathieu | University of Oxford
    Author: Surovik, David | University of Oxford
    Author: Havoutis, Ioannis | University of Oxford
    Author: Fallon, Maurice | University of Oxford
 
    keyword: Legged Robots; Optimization and Optimal Control; Deep Learning in Robotics and Automation

    Abstract : Dynamic traversal of uneven terrain is a major objective in the field of legged robotics. The most recent model predictive control approaches for these systems can generate robust dynamic motion of short duration; however, planning over a longer time horizon may be necessary when navigating complex terrain. A recently-developed framework, Trajectory Optimization for Walking Robots (TOWR), computes such plans but does not guarantee their reliability on real platforms, under uncertainty and perturbations. We extend TOWR with analytical costs to generate trajectories that a state-of-the-art whole-body tracking controller can successfully execute. To reduce online computation time, we implement a learning-based scheme for initialization of the nonlinear program based on offline experience. The execution of trajectories as long as 16 footsteps and 5.5 s over different terrains by a real quadruped demonstrates the effectiveness of the approach on hardware. This work builds toward an online system which can efficiently and robustly replan dynamic trajectories.

- On the Hardware Feasibility of Nonlinear Trajectory Optimization for Legged Locomotion Based on a Simplified Dynamics

    Author: Bratta, Angelo | Istituto Italiano Di Tecnologia
    Author: Orsolino, Romeo | University of Oxford
    Author: Focchi, Michele | Fondazione Istituto Italiano Di Tecnologia
    Author: Barasuol, Victor | Istituto Italiano Di Tecnologia
    Author: Muscolo, Giovanni Gerardo | Politecnico Di Torino
    Author: Semini, Claudio | Istituto Italiano Di Tecnologia
 
    keyword: Legged Robots; Optimization and Optimal Control; Dynamics

    Abstract : Simplified models are useful to increase the com- putational efficiency of a motion planning algorithm, but their lack of accuracy have to be managed. We propose two feasibility constraints to be included in a Single Rigid Body Dynamics- based trajectory optimizer in order to obtain robust motions in challenging terrain. The first one finds an approximate relation- ship between joint-torque limits and admissible contact forces, without requiring the joint positions. The second one proposes a leg model to prevent leg collision with the environment. Such constraints have been included in a simplified nonlinear non- convex trajectory optimization problem. We demonstrate the feasibility of the resulting motion plans both in simulation and on the Hydraulically actuated Quadruped (HyQ) robot, considering experiments on an irregular terrain.

- Agile Legged-Wheeled Reconfigurable Navigation Planner Applied on the CENTAURO Robot

    Author: Raghavan, Vignesh Sushrutha | Istituto Italiano Di Tecnologia
    Author: Kanoulas, Dimitrios | University College London
    Author: Caldwell, Darwin G. | Istituto Italiano Di Tecnologia
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
 
    keyword: Motion and Path Planning; Visual-Based Navigation; Legged Robots

    Abstract : Hybrid legged-wheeled robots such as the CENTAURO, are capable of varying their footprint polygon to carry out various agile motions. This property can be advantageous for wheeled-only planning in cluttered spaces, which is our focus. In this paper, we present an improved algorithm that builds upon our previously introduced preliminary footprint varying A* planner, which was based on the rectangular symmetry of the foot support polygon. In particular, we introduce a Theta* based planner with trapezium-like search, which aims to further reduce the limitations imposed upon the wheeled-only navigation of the CENTAURO robot by the low-dimensional search space, maintaining the real-time computational efficiency. The method is tested on the simulated and real full-size CENTAURO robot in cluttered environments.

- Bounded Haptic Teleoperation of a Quadruped Robot's Foot Posture for Sensing and Manipulation

    Author: Xin, Guiyang | The University of Edinburgh
    Author: Smith, Joshua | University of Edinburgh
    Author: Rytz, David | Oxford
    Author: Wolfslag, Wouter | University of Edinburgh
    Author: Lin, Hsiu-Chin | McGIll University
    Author: Mistry, Michael | University of Edinburgh
 
    keyword: Legged Robots; Motion Control; Haptics and Haptic Interfaces

    Abstract : This paper presents a control framework to teleoperate a quadruped robot's foot for operator-guided haptic exploration of the environment. Since one leg of a quadruped robot typically only has 3 actuated degrees of freedom (DoFs), the torso is employed to assist foot posture control via a hierarchical whole-body controller. The foot and torso postures are controlled by two analytical Cartesian impedance controllers cascaded by a null space projector. The contact forces acting on supporting feet are optimized by quadratic programming (QP). The foot's Cartesian impedance controller may also estimate contact forces from trajectory tracking errors, and relay the force-feedback to the operator. A 7D haptic joystick, Sigma.7, transmits motion commands to the quadruped robot ANYmal, and renders the force feedback. Furthermore, the joystick's motion is bounded by mapping the foot's feasible force polytope constrained by the friction cones and torque limits in order to prevent the operator from driving the robot to slipping or falling over. Experimental results demonstrate the efficiency of the proposed framework.

- Pinbot: A Walking Robot with Locking Pin Arrays for Passive Adaptability to Rough Terrains

    Author: Noh, Seonghoon | Yale University
    Author: Dollar, Aaron | Yale University
 
    keyword: Legged Robots; Underactuated Robots; Mechanism Design

    Abstract : To date, many control strategies for legged robots have been proposed for stable locomotion over rough and unstructured terrains. However, these approaches require sensing information throughout locomotion, which may be noisy or unavailable at times. An alternative solution to rough terrain locomotion is a legged robot design that can passively adapt to the variations in the terrain without requiring knowledge of them. This paper presents one such solution in the design of a walking robot that employs pin array mechanisms to passively adapt to rough terrains. The pins are passively dropped over the terrain to conform to its variations and then locked to provide a statically stable stance. Locomotion is achieved with parallel four-bar linkages that swing forward the platforms in an alternating manner. Experimental evaluation of the robot demonstrates that the pin arrays enable legged locomotion over rough terrains under open-loop control.

- Planning for the Unexpected: Explicitly Optimizing Motions for Ground Uncertainty in Running

    Author: Green, Kevin | Oregon State University
    Author: Hatton, Ross | Oregon State University
    Author: Hurst, Jonathan | Oregon State University
 
    keyword: Legged Robots; Optimization and Optimal Control; Robust/Adaptive Control of Robotic Systems

    Abstract : We propose a method to generate actuation plans for a reduced order, dynamic model of bipedal running. This method explicitly enforces robustness to ground uncertainty. The plan generated is not a fixed body trajectory that is aggressively stabilized: instead, the plan interacts with the passive dynamics of the reduced order model to create emergent robustness. The goal is to create plans for legged robots that will be robust to imperfect perception of the environment, and to work with dynamics that are too complex to optimize in real-time. Working within this dynamic model of legged locomotion, we optimize a set of disturbance cases together with the nominal case, all with linked inputs. The input linking is nontrivial due to the hybrid dynamics of the running model but our solution is effective and has analytical gradients. The optimization procedure proposed is significantly slower than a standard trajectory optimization, but results in robust gaits that reject disturbances extremely effectively without any replanning required.
- On the Efficient Control of Series-Parallel Compliant Articulated Robots

    Author: Amara, Vishnu Dev | Istituto Italiano Di Tecnologia
    Author: Malzahn, J�rn | Istituto Italiano Di Tecnologia
    Author: Ren, Zeyu | Istituto Italiano Di Tecnologia
    Author: Roozing, Wesley | University of Twente
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
 
    keyword: Legged Robots; Energy and Environment-Aware Automation; Optimization and Optimal Control

    Abstract : The paper applies optimal control to investigate the efficacy of distinct torque distribution strategies on a redundant robot that combines the potential of asymmetric series-parallel compliant actuation branches and multi-articulation. An optimization based controller that can accommodate various quadratic cost functions to perform the non-trivial torque distribution among the dissimilar actuators is contrived. Three candidate criteria are composed and their performance during periodic squat motions are compared. From one set of experiments, it is learnt that by minimizing a criterion that takes into account the actuator hardware specifications, the gravity-driven phases can be lengthened. Thereby, the particular criterion results in slightly better performance than when adopting a strategy that maximizes the torque allocation to the higher efficiency actuators. Another set of experiments performed across a range of frequencies provide valuable insights such as the efficacy of maximum utilization of the highly-efficient but slow actuators decreases progressively at high frequencies.

- Preintegrated Velocity Bias Estimation to Overcome Contact Nonlinearities in Legged Robot Odometry

    Author: Wisth, David | University of Oxford
    Author: Camurri, Marco | University of Oxford
    Author: Fallon, Maurice | University of Oxford
 
    keyword: Legged Robots; Sensor Fusion; Visual-Based Navigation

    Abstract : In this paper, we present a novel factor graph formulation to estimate the pose and velocity of a quadruped robot on slippery and deformable terrains. The factor graph includes a new type of preintegrated velocity factor that incorporates velocity inputs from leg odometry. To accommodate for leg odometry drift, we extend the robot's state vector with a bias term for this preintegrated velocity factor. This term incorporates all the effects of unmodeled uncertainties at the contact point, such as slippery or deformable grounds and leg flexibility. The bias term can be accurately estimated thanks to the tight fusion of the preintegrated velocity factor with stereo vision and IMU factors, without which it would be unobservable. The system has been validated on several scenarios that involve dynamic motions of the ANYmal robot on loose rocks, slopes and muddy ground. We demonstrate a 26% improvement of relative pose error compared to our previous work and 52% compared to a state-of-the-art proprioceptive state estimator.

- Optimized Foothold Planning and Posture Searching for Energy-Efficient Quadruped Locomotion Over Challenging Terrains

    Author: Chen, Lu | Peng Cheng Laboratory (PCL) , Shenzhen, China
    Author: Ye, Shusheng | The Chinese University of Hong Kong (CUHK), Shenzhen, China
    Author: Sun, Caiming | The Chinese University of Hong Kong, Shenzhen
    Author: Zhang, Aidong | The Chinese University of Hong Kong, Shenzhen
    Author: Deng, Ganyu | The Chinese University of Hong Kong, Shenzhen
    Author: Liao, Tianjiao | The Chinese University of Hong Kong, Shenzhen
 
    keyword: Legged Robots; Energy and Environment-Aware Automation

    Abstract : Energy-efficient locomotion is of primary importance for legged robot to extend operation time in practical applications. This paper presents an approach to achieve energy-efficient locomotion for a quadrupedal robot walking over challenging terrains. Firstly, we optimize the nominal stance parameters based on the analysis of leg torque distribution. Secondly, we proposed the foothold planner and the center of gravity (COG) trajectory planner working together to guide the robot to place its standing legs in an energy-saving stance posture. We have validated the effectiveness of our method on a real quadrupedal robot in experiments including autonomously walking on plain ground and climbing stairs.

- Extracting Legged Locomotion Heuristics with Regularized Predictive Control

    Author: Bledt, Gerardo | Massachusetts Institute of Technology (MIT)
    Author: Kim, Sangbae | Massachusetts Institute of Technology
 
    keyword: Legged Robots

    Abstract : Optimization based predictive control is a powerful tool that has improved the ability of legged robots to execute dynamic maneuvers and traverse increasingly difficult terrains. However, it is often challenging and unintuitive to design meaningful cost functions and build high-fidelity models while adhering to timing restrictions. A novel framework to extract and design principled regularization heuristics for legged locomotion optimization control is presented. By allowing a simulation to fully explore the cost space offline, certain states and actions can be constrained or isolated. Data is fit with simple models relating the desired commands, optimal control actions, and robot states to identify new heuristic candidates. Basic parameter learning and adaptation laws are then applied to the models online. This method extracts simple, but powerful heuristics that can approximate complex dynamics and account for errors stemming from model simplifications and parameter uncertainty without the loss of physical intuition while generalizing the parameter tuning process. Results on the Mini Cheetah robot verify the increased capabilities due to the newly extracted heuristics without any modification to the controller structure or gains.

- Learning Generalizable Locomotion Skills with Hierarchical Reinforcement Learning

    Author: Li, Tianyu | Facebook
    Author: Lambert, Nathan | University of California, Berkeley
    Author: Calandra, Roberto | Facebook
    Author: Meier, Franziska | Facebook
    Author: Rai, Akshara | Facebook AI Research
 
    keyword: Legged Robots; AI-Based Methods; Model Learning for Control

    Abstract : Learning to locomote to arbitrary goals on hardware remains a challenging problem for reinforcement learning. In this paper, we present a hierarchical learning framework that improves sample-efficiency and generalizability of locomotion skills on real-world robots. Our approach divides the problem of goal-oriented locomotion into two sub-problems: learning diverse primitives skills, and using model-based planning to sequence these skills. We parametrize our primitives as cyclic movements, improving sample-efficiency of learning on a 18 degrees of freedom robot. Then, we learn coarse dynamics models over primitive cycles and use them in a model predictive control framework. This allows us to learn to walk to arbitrary goals up to 12m away, after about two hours of training from scratch on hardware. Our results on a Daisy hexapod hardware and simulation demonstrate the efficacy of our approach at reaching distant targets, in different environments and with sensory noise.

- SoRX: A Soft Pneumatic Hexapedal Robot to Traverse Rough, Steep, and Unstable Terrain

    Author: Liu, Zhichao | University of California, Riverside
    Author: Lu, Zhouyu | University of California, Riverside
    Author: Karydis, Konstantinos | University of California, Riverside
 
    keyword: Legged Robots; Soft Robot Applications; Soft Robot Materials and Design

    Abstract : Soft robotics technology creates new ways for legged robots to interact with and adapt to their environment. In this paper we develop i) a new 2-degree-of-freedom soft pneumatic actuator, and ii) a novel soft robotic hexapedal robot called SoRX that leverages the new actuators. Simulation and physical testing confirm that the proposed actuator can generate cyclic foot trajectories that are appropriate for legged locomotion. Consistent with other hexapedal robots (and animals), SoRX employs an alternating tripod gait to propel itself forward. Experiments reveal that SoRX can reach forward speeds of up to 0.44 body lengths per second, or equivalently 101 mm/s. With a size of 230 mm length, 140 mm width and 100 mm height, and weight of 650 grams, SoRX is among the fastest tethered soft pneumatically-actuated legged robots to date. The motion capabilities of SoRX are evaluated through five experiments: running, step climbing, and traversing rough terrain, steep terrain, and unstable terrain. Experimental results show that SoRX is able to operate over challenging terrains in open-loop control and by following the same alternating tripod gait across all experimental cases.

- Probe-Before-Step Walking Strategy for Multi-Legged Robots on Terrain with Risk of Collapse

    Author: Tennakoon, Eranda | Queensland University of Technology
    Author: Peynot, Thierry | Queensland University of Technology (QUT)
    Author: Roberts, Jonathan | Queensland University of Technology
    Author: Kottege, Navinda | CSIRO
 
    keyword: Legged Robots; Failure Detection and Recovery; Robot Safety

    Abstract : Multi-legged robots are effective at traversing rough terrain. However, terrains that include collapsible footholds (i.e. regions that can collapse when stepped on) remain a significant challenge, especially since such situations can be extremely difficult to anticipate using only exteroceptive sensing. State-of-the-art methods typically use various stabilisation techniques to regain balance and counter changing footholds. However, these methods are likely to fail if safe footholds are sparse and spread out or if the robot does not respond quickly enough after a foothold collapse. This paper presents a novel method for multi-legged robots to probe and test the terrain for collapses using its legs while walking. The proposed method improves on existing terrain probing approaches, and integrates the probing action into a walking cycle. A follow-the-leader strategy with a suitable gait and stance is presented and implemented on a hexapod robot. The proposed method is experimentally validated, demonstrating the robot can safely traverse terrain containing collapsible footholds.

- An Augmented Kinematic Model for the Cartesian Control of the Hybrid Wheeled-Legged Quadrupedal Robot CENTAURO

    Author: Laurenzi, Arturo | Istituto Italiano Di Tecnologia
    Author: Mingo Hoffman, Enrico | Fondazione Istituto Italiano Di Tecnologia
    Author: Parigi Polverini, Matteo | Istituto Italiano Di Tecnologia (IIT)
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
 
    keyword: Wheeled Robots; Motion Control; Legged Robots

    Abstract : This work deals with the kinematic control of Centauro, a highly redundant, hybrid wheeled-legged robot designed at Istituto Italiano di Tecnologia (IIT). Given its full wheeled mobility as allowed by its four independently steerable wheels, the choice of some local frame (in addition to the global world) is required in order to express tasks that are naturally defined in a robot-centric fashion. In this work, we show that trivially selecting such a frame as the robot trunk leads to sub-optimal results in terms of motion capabilities; as main contribution, we therefore propose a comparative analysis among three different choices of local frame, and demonstrate that in order to retain all advantages from the whole-body control domain, the kinematic model of the robot must be augmented with an additional virtual frame, which proves to be a useful choice of local frame, enabling e.g. the automatic adaptation of the trunk posture under constraint activation. The resulting Cartesian controller is finally validated by means of an extensive experimental session on the real hardware.

- Precision Robotic Leaping and Landing Using Stance-Phase Balance

    Author: Yim, Justin K. | University of California, Berkeley
    Author: Singh, Bajwa Roodra Pratap | Italian Institute of Technology
    Author: Wang, Eric K. | University of California, Berkeley
    Author: Featherstone, Roy | Istituto Italiano Di Tecnologia
    Author: Fearing, Ronald | University of California at Berkeley
 
    keyword: Legged Robots

    Abstract : Prior work has addressed the control of continuous jumping by setting touchdown angle in flight, but greater precision can be obtained by using liftoff angle in stance to direct individual leaps. We demonstrate targeted leaping and landing on a narrow foot with a small, single leg hopping robot, Salto-1P. Accurate and reliable leaping and landing are achieved by the combination of stance-phase balance control based on angular momentum, a launch trajectory that stabilizes the robot at a desired launch angle, and an approximate expression for selecting touchdown angle before landing. Furthermore, dynamic transitions between standing, hopping, and standing again are now possible in a robot with a narrow foot. We also present approximate bounds on acceptable velocity estimate and angle errors outside of which balanced landing is no longer possible. Compared to a prior Spring Loaded Inverted Pendulum (SLIP)-like gait, the jump distance standard deviation is reduced from 9.2 cm to 1.6 cm for particular jumps, now enabling precise jumps to narrow targets.

- STANCE: Locomotion Adaptation Over Soft Terrain (I)

    Author: Fahmi, Shamel | Istituto Italiano Di Tecnologia
    Author: Focchi, Michele | Fondazione Istituto Italiano Di Tecnologia
    Author: Radulescu, Andreea | Dyson Technology Ltd
    Author: Fink, Geoff | Istituto Italiano Di Tecnologia
    Author: Barasuol, Victor | Istituto Italiano Di Tecnologia
    Author: Semini, Claudio | Istituto Italiano Di Tecnologia
 
    keyword: Legged Robots; Compliance and Impedance Control; Optimization and Optimal Control

    Abstract : Whole-Body Control (WBC) has emerged as an important framework in locomotion control for legged robots. However, most WBC frameworks fail to generalize beyond rigid terrains. Legged locomotion over soft terrain is difficult due to the presence of unmodeled contact dynamics that WBCs do not account for. This introduces uncertainty in locomotion and affects the stability and performance of the system. In this paper, we propose a novel soft terrain adaptation algorithm called STANCE: Soft Terrain Adaptation and Compliance Estimation. STANCE consists of a WBC that exploits the knowledge of the terrain to generate an optimal solution that is contact consistent and an online terrain compliance estimator that provides the WBC with terrain knowledge. We validated STANCE both in simulation and experiment on the Hydraulically actuated Quadruped (HyQ) robot, and we compared it against the state of the art WBC. We demonstrated the capabilities of STANCE with multiple terrains of different compliances, aggressive maneuvers, different forward velocities, and external disturbances. STANCE allowed HyQ to adapt online to terrains with different compliances (rigid and soft) without pre-tuning. HyQ was able to successfully deal with the transition between different terrains and showed the ability to differentiate between compliances under each foot.

- Rolling in the Deep - Hybrid Locomotion for Wheeled-Legged Robots Using Online Trajectory Optimization

    Author: Bjelonic, Marko | ETH Zurich
    Author: Sekoor Lakshmana Sankar, Prajish Kumar | Technische Universiteit Delft
    Author: Bellicoso, C. Dario | ETH Zurich
    Author: Vallery, Heike | TU Delft
    Author: Hutter, Marco | ETH Zurich
 
    keyword: Legged Robots; Wheeled Robots; Optimization and Optimal Control

    Abstract : Wheeled-legged robots have the potential for highly agile and versatile locomotion. The combination of legs and wheels might be a solution for any real-world application requiring rapid, and long-distance mobility skills on challenging terrain. In this paper, we present an online trajectory optimization framework for wheeled quadrupedal robots capable of executing hybrid walking-driving locomotion strategies. By breaking down the optimization problem into a wheel and base trajectory planning, locomotion planning for high dimensional wheeled-legged robots becomes more tractable, can be solved in real-time on-board in a model predictive control fashion, and becomes robust against unpredicted disturbances. The reference motions are tracked by a hierarchical whole-body controller that sends torque commands to the robot. Our approach is verified on a quadrupedal robot with non-steerable wheels attached to its legs. The robot performs hybrid locomotion with a great variety of gait sequences on rough terrain. Besides, we validated the robotic platform at the Defense Advanced Research Projects Agency (DARPA) Subterranean Challenge, where the robot rapidly mapped, navigated and explored dynamic underground environments.







- Optimal Landing Strategy for Two-Mass Hopping Leg with Natural Dynamics

    Author: Lee, Chan | DGIST (Daegu Gyeongbuk Institute of Science and Technology)
    Author: Oh, Sehoon | DGIST (Daegu Gyeongbuk Institute of Science and Technology)
 
    keyword: Legged Robots; Compliance and Impedance Control; Humanoid and Bipedal Locomotion

    Abstract : It is necessary for a robotic leg to behave like a spring to realize a periodic hopping, since it can be efficient and does not require complicated control algorithm. However, the impact force makes the realization of periodic hopping more challenging. In this paper, an optimal landing strategy for a hopping leg is proposed, which can realize continuous hopping motion only by natural dynamics. The proposed strategy can reduce the foot landing velocity to zero and thus minimize the impact force. The formulation to derive the optimal condition is derived theoretically based on two-mass leg model, and its effectiveness is verified through various simulations and experiments using a series elastic actuator-driven robot leg.

- From Bipedal Walking to Quadrupedal Locomotion: Full-Body Dynamics Decomposition for Rapid Gait Generation

    Author: Ma, Wenlong | Caltech
    Author: Ames, Aaron | Caltech
 
    keyword: Legged Robots; Humanoid and Bipedal Locomotion; Optimization and Optimal Control

    Abstract : This paper systematically decomposes a quadrupedal robot into bipeds to rapidly generate walking gaits and then recomposes these gaits to obtain quadrupedal locomotion. We begin by decomposing the full-order, nonlinear and hybrid dynamics of a three-dimensional quadrupedal robot, including its continuous and discrete dynamics, into two bipedal systems that are subject to external forces. Using the hybrid zero dynamics (HZD) framework, gaits for these bipedal robots can be rapidly generated (on the order of seconds) along with corresponding controllers. The decomposition is achieved in such a way that the bipedal walking gaits and controllers can be composed to yield dynamic walking gaits for the original quadrupedal robot --- the result is the rapid generation of dynamic quadruped gaits utilizing the full-order dynamics. This methodology is demonstrated through the rapid generation (3.96 seconds on average) of four stepping-in-place gaits and one diagonally symmetric ambling gait at 0.35 m/s on a quadrupedal robot --- the Vision 60, with 36 state variables and 12 control inputs --- both in simulation and through outdoor experiments. This suggested a new approach for fast quadrupedal trajectory planning using full-body dynamics, without the need for empirical model simplification, wherein methods from dynamic bipedal walking can be directly applied to quadrupeds.

- Posture Control for a Low-Cost Commercially-Available Hexapod Robot

    Author: Tikam, Mayur | CSIR South Africa
    Author: Withey, Daniel | CSIR
    Author: Theron, Nicolaas Johannes | University of Pretoria
 
    keyword: Legged Robots; Force Control; Motion Control

    Abstract : Posture control for legged robots has been widely developed on custom-designed robotic platforms, with little work being done on commercially-available robots despite their potential as low-cost research platforms. This paper presents the implementation of a Walking Posture Control system on a commercially-available hexapod robot which utilizes low-cost joint actuators without torque control capabilities. The hierarchical control system employs Virtual Model Control with simple foot force distribution and a novel, position-based Foot Force Controller that enables direct force control during the leg's stance phase and active compliance control during the swing phase. Ground truth measurements in experimental tests, obtained with a Vicon motion capture system, demonstrate the improvement to posture made by the control system on uneven terrain, with the results comparing favorably to those obtained in similar tests on more sophisticated, custom-designed platforms.

- DeepGait: Planning and Control of Quadrupedal Gaits Using Deep Reinforcement Learning

    Author: Tsounis, Vassilios | ETH Zurich
    Author: Alge, Mitja | ETH Zurich
    Author: Lee, Joonho | ETH Zurich Robotic Systems Laboratory
    Author: Farshidian, Farbod | ETH Zurich
    Author: Hutter, Marco | ETH Zurich
 
    keyword: Legged Robots; Deep Learning in Robotics and Automation; Motion and Path Planning

    Abstract : This paper addresses the problem of legged locomotion in non-flat terrain. As legged robots such as quadrupeds are to be deployed in terrains with geometries which are difficult to model and predict, the need arises to equip them with the capability to generalize well to unforeseen situations. In this work, we propose a novel technique for training neural-network policies for terrain-aware locomotion, which combines state-of-the-art methods for model-based motion planning and reinforcement learning. Our approach is centered on formulating Markov decision processes using the evaluation of dynamic feasibility criteria in place of physical simulation. We thus employ policy-gradient methods to independently train policies which respectively plan and execute foothold and base motions in 3D environments using both proprioceptive and exteroceptive measurements. We apply our method within a challenging suite of simulated terrain scenarios which contain features such as narrow bridges, gaps and stepping-stones, and train policies which succeed in locomoting effectively in all cases.

- The Soft-Landing Problem: Minimizing Energy Loss by a Legged Robot Impacting Yielding Terrain

    Author: Lynch, Daniel | Northwestern University
    Author: Lynch, Kevin | Northwestern University
    Author: Umbanhowar, Paul | Northwestern University
 
    keyword: Legged Robots; Compliance and Impedance Control; Optimization and Optimal Control

    Abstract : Enabling robots to walk and run on yielding terrain is vital to endeavors ranging from disaster response to extraterrestrial exploration. While dynamic legged locomotion on rigid ground is challenging enough, yielding terrain presents additional challenges such as ground deformation which dissipates energy. In this paper, we examine the soft-landing problem: given some impact momentum, bring the robot to rest while minimizing foot penetration depth. To gain insight into properties of penetration depth-minimizing control policies, we formulate a constrained optimal control problem and obtain a bang-bang open-loop force profile. Motivated by examples from biology and recent advances in legged robotics, we also examine impedance-control solutions to the soft-landing problem. Through simulations and experiments, we find that optimal impedance reduces penetration depth nearly as much as the open-loop force profile, while remaining robust to model uncertainty. Lastly, we discuss the relevance of this work to minimum-cost-of-transport locomotion for several actuator design choices.

- A Computational Framework for Designing Skilled Legged-Wheeled Robots

    Author: Geilinger, Moritz | ETH Zurich
    Author: Winberg, Sebastian | ETH Zurich
    Author: Coros, Stelian | Carnegie Mellon University
 
    keyword: Legged Robots; Motion and Path Planning; Wheeled Robots

    Abstract : Legged-wheeled robots promise versatile, fast and efficient mobile capabilities. To unleash their full potential, however, such hybrid robots need to be designed in a way that promotes the rich, full-body motions required for novel locomotion modes. This paper discusses the computational framework we have used to create a new type of legged robot which, when equipped with different types of end-effectors, is capable of an array of interesting locomotory behaviors, including walking, roll-walking, roller-blading, and ice-skating. We show that this computational framework, which builds on a design system we recently introduced in the computer graphics community, can accurately predict the way in which different design decisions affect the robot's ability to move, thus serving as an important tool in engineering new types of mobile robots. We also propose a novel warm-starting method which leverages ideas from numerical continuation to drastically improve convergence rates for the trajectory optimization routine we employ to generate optimal motions.

## Multi-Robot Systems

- Optimal Perimeter Guarding with Heterogeneous Robot Teams: Complexity Analysis and Effective Algorithms

    Author: Feng, Si Wei | Rutgers University
    Author: Yu, Jingjin | Rutgers University
 
    keyword: Multi-Robot Systems; Optimization and Optimal Control; Surveillance Systems

    Abstract : We perform structural and algorithmic studies of significantly generalized versions of the optimal perimeter guarding (OPG) problem. As compared with the original OPG where robots are uniform, in this paper, many mobile robots with heterogeneous sensing capabilities are to be deployed to optimally guard a set of one-dimensional segments. Two complimentary formulations are investigated where one limits the number of available robots (OPGLR) and the other seeks to minimize the total deployment cost (OPGMC). In contrast to the original OPG which admits low-polynomial time solutions, both OPGLR and OPGMC are computationally intractable with OPGLR being strongly NP-hard. Nevertheless, we develop fairly scalable pseudo-polynomial time algorithms for practical, fixed-parameter subcase of OPGLR; we also develop pseudo-polynomial time algorithm for general OPGMC and polynomial time algorithm for the fixed-parameter OPGMC case. The applicability and effectiveness of selected algorithms are demonstrated through extensive numerical experiments.

- Spatial Scheduling of Informative Meetings for Multi-Agent Persistent Coverage

    Author: Haksar, Ravi N. | Stanford University
    Author: Trimpe, Sebastian | Max Planck Institute for Intelligent Systems
    Author: Schwager, Mac | Stanford University
 
    keyword: Multi-Robot Systems; Path Planning for Multiple Mobile Robots or Agents; Distributed Robot Systems

    Abstract : In this work, we develop a novel decentralized coordination algorithm for a team of autonomous unmanned aerial vehicles (UAVs) to surveil an aggressive forest wildfire. For dangerous environmental processes that occur over very large areas, like forest wildfires, multi-agent systems cannot rely on long-range communication networks. Therefore, our framework is formulated for very restrictive communication constraints: UAVs are only able to communicate when they are physically close to each other. To accommodate this constraint, the UAVs schedule a time and place to meet in the future to guarantee that they will be able to meet up again and share their belief of the wildfire state. In contrast with prior work, we allow for a discrete time, discrete space Markov model with a large state space as well as restrictive communication constraints. We demonstrate the effectiveness of our approach using simulations of a wildfire model that has 10^{298} total states.

- Simultaneous Policy and Discrete Communication Learning for Multi-Agent Cooperation

    Author: Freed, Benjamin | Carnegie Mellon University
    Author: Sartoretti, Guillaume Adrien | National University of Singapore (NUS)
    Author: Choset, Howie | Carnegie Mellon University
 
    keyword: Multi-Robot Systems; Deep Learning in Robotics and Automation; Distributed Robot Systems

    Abstract : Decentralized multi-agent reinforcement learning has been demonstrated to be an effective solution to large multi-agent control problems. However, agents typically can only make decisions based on local information, resulting in suboptimal performance in partially-observable settings. The addition of a communication channel overcomes this limitation by allowing agents to exchange information. Existing approaches, however, have required agent output size to scale exponentially with the number of message bits, and have been slow to converge to satisfactory policies due to the added difficulty of learning message selection. We propose an independent bitwise message policy parameterization that allows agent output size to scale linearly with information content. Additionally, we leverage aspects of the environment structure to derive a novel policy gradient estimator that is both unbiased and has a lower variance message gradient contribution than typical policy gradient estimators. We evaluate the impact of these two contributions on a collaborative multi-agent robot navigation problem, in which information must be exchanged among agents. We find that both significantly improve sample efficiency and result in improved final policies, and demonstrate the applicability of these techniques by deploying the learned policies on physical robots.

- Cooperative Team Strategies for Multi-Player Perimeter-Defense Games

    Author: Shishika, Daigo | University of Pennsylvania
    Author: Paulos, James | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania, School of Engineering and Applied Sc
 
    keyword: Multi-Robot Systems; Cooperating Robots

    Abstract : This paper studies a variant of the multi-player reach-avoid game played between intruders and defenders with applications to perimeter defense. The intruder team tries to score by sending as many intruders as possible to the target area, while the defender team tries to minimize this score by intercepting them. Finding the optimal strategies of the game is challenging due to the high dimensionality of the joint state space. Existing works have proposed approximation methods to reduce the design of the defense strategy into assignment problems, however, they suffer from either suboptimal defender performance or computational complexity. Based on a novel decomposition method, this paper proposes a scalable (polynomial-time) assignment algorithm that accommodates cooperative behaviors and outperforms the existing defense strategies. For a certain class of initial configurations, we derive the exact score by showing that the lower bound provided by the intruder team matches the upper bound provided by the defender team, which also proves the optimality of the team strategies.

- Multirobot Symmetric Formations for Gradient and Hessian Estimation with Application to Source Seeking (I)

    Author: Bri��n Arranz, Lara | CEA Tech
    Author: Renzaglia, Alessandro | INRIA
    Author: Schenato, Luca | University of Padova
 
    keyword: Cooperating Robots; Networked Robots; Marine Robotics

    Abstract : This paper deals with the problem of estimating in a collaborative way the gradient and the Hessian matrix of an unknown signal via noisy measurements collected by a group of robots. We propose symmetric formations with a reduced number of robots for both the two-dimensional (2-D) and the three-dimensional (3-D) cases, such that the gradient and Hessian of the signal are estimated at the center of the formation via simple computation on local quantities independently of the orientation of the formation. If only gradient information is required, the proposed formations are suit- able for mobile robots that need to move in circular motion. We also provide explicit bounds for the approximation error and for the noise perturbation that can be used to optimally scale the formation radius. Numerical simulations illustrate the performance of the proposed strategy for source seeking against alternative solutions available in the literature and show how Hessian estimation can provide faster convergence even in the presence of noisy measurements.

- Multi-Robot Path Deconfliction through Prioritization by Path Prospects

    Author: Wu, Wenying | University of Cambridge
    Author: Bhattacharya, Subhrajit | Lehigh University
    Author: Prorok, Amanda | University of Cambridge
 
    keyword: Multi-Robot Systems; Distributed Robot Systems; Path Planning for Multiple Mobile Robots or Agents

    Abstract : This work deals with the problem of planning conflict-free paths for mobile robots in cluttered environments. Since centralized, coupled planning algorithms are computationally intractable for large numbers of robots, we consider decoupled planning, in which robots plan their paths sequentially in order of priority. Choosing how to prioritize the robots is a key consideration. State-of-the-art prioritization heuristics, however, do not model the coupling between a robot's mobility and its environment. This is particularly relevant when prioritizing between robots with different degrees of mobility. In this paper, we propose a prioritization rule that can be computed online by each robot independently, and that provides consistent, conflict-free path plans. Our innovation is to formalize a robot's path prospects to reach its goal from its current location. To this end, we consider the number of homology classes of trajectories, which capture distinct prospects of paths for each robot. This measure is used as a prioritization rule, whenever any robots enter negotiation to deconflict path plans. We perform simulations with heterogeneous robot teams and compare our method to five benchmarks. Our method achieves the highest success rate, and strikes a good balance between makespan and flowtime objectives.

- Cooperative Aerial-Ground Multi-Robot System for Automated Construction Tasks

    Author: Krizmancic, Marko | University of Zagreb, Faculty of Electrical Engineering and Comp
    Author: Arbanas, Barbara | University of Zagreb, Faculty of Electrical Engineering and Comp
    Author: Petrovic, Tamara | Univ. of Zagreb
    Author: Petric, Frano | University of Zagreb, Faculty of Electrical Engineering and Comp
    Author: Bogdan, Stjepan | University of Zagreb
 
    keyword: Multi-Robot Systems; Planning, Scheduling and Coordination; Robotics in Construction

    Abstract : In this paper, we study a cooperative aerial-ground robotic team and its application to the task of automated construction. We propose a solution for planning and coordinating the mission of constructing a wall with a predefined structure for a heterogeneous system consisting of one mobile robot and up to three unmanned aerial vehicles. The wall consists of bricks of various weights and sizes, some of which need to be transported using multiple robots simultaneously. To that end, we use hierarchical task representation to specify interrelationships between mission subtasks and employ effective scheduling and coordination mechanism, inspired by Generalized Partial Global Planning. We evaluate the performance of the method under different optimization criteria and validate the solution in the realistic Gazebo simulation environment.

- A Connectivity-Prediction Algorithm and Its Application in Active Cooperative Localization for Multi-Robot Systems

    Author: Zhang, Liang | Harbin Institute of Technology, Swiss Federal Institute of Techn
    Author: Zhang, Zexu | Harbin Institute of Technology
    Author: Siegwart, Roland | ETH Zurich
    Author: Chung, Jen Jen | Eidgen�ssische Technische Hochschule Zurich
 
    keyword: Multi-Robot Systems; Path Planning for Multiple Mobile Robots or Agents; Localization

    Abstract : This paper presents a method for predicting the probability of future connectivity between mobile robots with range-limited communication. In particular, we focus on its application to active motion planning for cooperative localization (CL). The probability of connection is modeled by the distribution of quadratic forms in random normal variables and is computed by the infinite power series expansion theorem. A finite-term approximation is made to realize the computational feasibility and three more modifications are designed to handle the adverse impacts introduced by the omission of the higher order series terms. On the basis of this algorithm, an active and CL problem with leader-follower architecture is then reformulated into a Markov Decision Process (MDP) with a one-step planning horizon, and the optimal motion strategy is generated by minimizing the expected cost of the MDP. Extensive simulations and comparisons are presented to show the effectiveness and efficiency of both the proposed prediction algorithm and the MDP model.

- Multi-Agent Formation Control Based on Distributed Estimation with Prescribed Performance

    Author: Stamouli, Charis | NTUA
    Author: Bechlioulis, Charalampos | National Technical University of Athens
    Author: Kyriakopoulos, Kostas | National Technical Univ. of Athens
 
    keyword: Multi-Robot Systems; Distributed Robot Systems; Autonomous Agents

    Abstract : We consider the distributed simultaneous estimation and formation control problem for swarms of identical mobile agents with limited communication, sensing and computation capabilities. In particular, we develop a novel scalable algorithm that encodes the formation specifications of the swarm via geometric moment statistics, which are estimated by a distributed scheme with prescribed performance guarantees. Based on the locally available information, each agent calculates an estimate of the global formation statistics, which is then employed by its local motion controller, thus creating a feedback interconnection between the estimator and the controller. The proposed scheme guarantees convergence of the global formation statistics to the desired values, while decoupling the estimation performance from the control performance. Moreover, a minimum allowable inter-agent distance can be predetermined so that inter-agent collision avoidance is achieved. Finally, simulation paradigms are provided to validate the approach.

- Optimization-Based Distributed Flocking Control for Multiple Rigid Bodies

    Author: Ibuki, Tatsuya | Tokyo Institute of Technology
    Author: Wilson, Sean | Georgia Institute of Technology
    Author: Yamauchi, Junya | Tokyo Institute of Technology
    Author: Fujita, Masayuki | Tokyo Institute of Technology
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Multi-Robot Systems; Optimization and Optimal Control; Swarms

    Abstract : This paper considers distributed flocking control on the Special Euclidean group for networked rigid bodies. The method captures the three flocking rules proposed by Reynolds: cohesion; alignment; and separation. The proposed controller is based only on relative pose (position and attitude) information with respect to neighboring rigid bodies so that it can be implemented in a fully distributed manner using only local sensors. The flocking algorithm is moreover based on pose synchronization methods for the cohesion/alignment rules and achieves safe separation distances through the application of control barrier functions. The control input for each rigid body is chosen by solving a distributed optimization problem with constraints for pose synchronization and collision avoidance. Here, the inherent conflict between cohesion and separation is explicitly handled by relaxing the position synchronization constraint. The effectiveness of the proposed flocking algorithm is demonstrated via simulation and hardware experiments.

- Behavior Mixing with Minimum Global and Subgroup Connectivity Maintenance for Large-Scale Multi-Robot Systems

    Author: Luo, Wenhao | Carnegie Mellon University
    Author: Yi, Sha | Carnegie Mellon University
    Author: Sycara, Katia | Carnegie Mellon University
 
    keyword: Multi-Robot Systems; Networked Robots; Autonomous Agents

    Abstract : In many cases the multi-robot systems are desired to execute simultaneously multiple behaviors with different controllers, and sequences of behaviors in real time, which we call textit{behavior mixing}. Behavior mixing is accomplished when different subgroups of the overall robot team change their controllers to collectively achieve given tasks while maintaining connectivity within and across subgroups in one connected communication graph. In this paper, we present a provably minimum connectivity maintenance framework to ensure the subgroups and overall robot team stay connected at all times while providing the highest freedom for behavior mixing. In particular, we propose a real-time distributed Minimum Connectivity Constraint Spanning Tree (MCCST) algorithm to select the minimum inter-robot connectivity constraints preserving subgroup and global connectivity that are textit{least likely to be violated} by the original controllers. With the employed safety and connectivity barrier certificates for the activated connectivity constraints and collision avoidance, the behavior mixing controllers are thus minimally modified from the original controllers. We demonstrate the effectiveness and scalability of our approach via simulations of up to 100 robots with multiple behaviors.


- CAPRICORN: Communication Aware Place Recognition Using Interpretable Constellations of Objects in Robot Networks

    Author: Ramtoula, Benjamin | École Polytechnique De Montréal, École Polytechnique Fédérale De
    Author: de Azambuja, Ricardo | University of Plymouth
    Author: Beltrame, Giovanni | Ecole Polytechnique De Montreal
 
    keyword: Multi-Robot Systems; Distributed Robot Systems; SLAM

    Abstract : Using multiple robots for exploring and mapping environments can provide improved robustness and performance, but it can be difficult to implement. In particular, limited communication bandwidth is a considerable constraint when a robot needs to determine if it has visited a location that was previously explored by another robot, as it requires for robots to share descriptions of places they have visited. One way to compress this description is to use constellations, groups of 3D points that correspond to the estimate of a set of relative object positions. Constellations maintain the same pattern from different viewpoints and can be robust to illumination changes or dynamic elements. We present a method to extract from these constellations compact spatial and semantic descriptors of the objects in a scene. We use this representation in a 2-step decentralized loop closure verification: first, we distribute the compact semantic descriptors to determine which other robots might have seen scenes with similar objects; then we query matching robots with the full constellation to validate the match using geometric information. The proposed method requires less memory, is more interpretable than global image descriptors, and could be useful for other tasks and interactions with the environment. We validate our system's performance on a TUM RGB-D SLAM sequence and show its benefits in terms of bandwidth requirements.

- Online Planning for Quadrotor Teams in 3-D Workspaces Via Reachability Analysis on Invariant Geometric Trees

    Author: Desai, Arjav Ashesh | Carnegie Mellon University
    Author: Michael, Nathan | Carnegie Mellon University
 
    keyword: Multi-Robot Systems; Motion and Path Planning; Planning, Scheduling and Coordination

    Abstract : We consider the kinodynamic multi-robot planning problem in cluttered 3-D workspaces. Reachability analysis on position invariant geometric trees is leveraged to find kinodynamically feasible trajectories for the multi-robot team from potentially non-stationary initial states. The key contribution of our approach is that a collision-free geometric solution guarantees a kinodynamically feasible, safe solution without additional refinement. Simulation results with up-to 40 robots and hardware results with 5 robots suggest the viability of the proposed approach for online planning and replanning for large teams of aerial robots in cluttered 3-D workspaces.

- Decentralized Visual-Inertial-UWB Fusion for Relative State Estimation of Aerial Swarm

    Author: Xu, Hao | HKUST
    Author: Wang, Luqi | HKUST
    Author: Zhang, Yichen | The Hong Kong University of Science and Technology
    Author: Qiu, Kejie | The Hong Kong University of Science and Technology
    Author: Shen, Shaojie | Hong Kong University of Science and Technology
 
    keyword: Multi-Robot Systems; Aerial Systems: Perception and Autonomy; Swarms

    Abstract : The collaboration of unmanned aerial vehicles (UAVs) has become a popular research topic for its practicability in multiple scenarios. The collaboration of multiple UAVs, which is also known as aerial swarm is a highly complex system, which still lacks a state-of-art decentralized relative state estimation method. In this paper, we present a novel fully decentralized visual-inertial-UWB fusion framework for relative state estimation and demonstrate the practicability by performing extensive aerial swarm flight experiments. The comparison result with ground truth data from the motion capture system shows the centimeter-level precision which outperforms all the Ultra-WideBand (UWB) and even vision based method. The system is not limited by the field of view (FoV) of the camera or Global Positioning System (GPS), meanwhile on account of its estimation consistency, we believe that the proposed relative state estimation framework has the potential to be prevalently adopted by aerial swarm applications in different scenarios in multiple scales.

- Synthesis of a Time-Varying Communication Network by Robot Teams with Information Propagation Guarantees

    Author: Yu, Xi | University of Pennsylvania
    Author: Hsieh, M. Ani | University of Pennsylvania
 
    keyword: Multi-Robot Systems; Networked Robots; Distributed Robot Systems

    Abstract : We present a distributed control and coordination strategy to enable a swarm of mobile robots to form an intermittently connected communication network while monitoring an environment. In particular, we consider the scenario where robots are tasked to patrol a collection of perimeters within a workspace and are only able to communicate with one another when they move into each other's communication range as they move along their respective perimeters. We show how intermittent connectivity can be achieved as each robot synchronizes its speed with robots moving along neighboring perimeters. By ensuring future rendezvous between robot pairs, the team forms a time-varying communication network where information can be successfully transmitted between any pair of robots within some finite period of time. We show how the proposed strategy guarantees a tau-connected network for some finite tau&gt;0 and provide bounds on the time needed to propagate information throughout the network. Simulations are presented to show the feasibility of our strategy and the validity of our approach.

- DC-CAPT: Concurrent Assignment and Planning of Trajectories for Dubins Cars

    Author: Whitzer, Michael | University of Pennsylvania
    Author: Shishika, Daigo | University of Pennsylvania
    Author: Thakur, Dinesh | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania, School of Engineering and Applied Sc
    Author: Prorok, Amanda | University of Cambridge
 
    keyword: Multi-Robot Systems

    Abstract : We present an algorithm for the concurrent assignment and planning of collision-free trajectories (DC-CAPT) for robots whose kinematics can be modeled as Dubins cars, i.e., robots constrained in terms of their initial orientation and their minimum turning radius. Coupling the assignment and trajectory planning subproblems allows for a computationally tractable solution. This solution is guaranteed to be collision-free through the use of a single constraint: the start and goal locations have separation distance greater than some threshold. We derive this separation distance by extending a prior work that assumed holonomic robots. We demonstrate the validity of our approach, and show its efficacy through simulations and experiments where groups of robots executing Dubins curves travel to their assigned goal locations without collisions.

- An Adversarial Approach to Private Flocking in Mobile Robot Teams

    Author: Zheng, Hehui | University of Cambridge
    Author: Panerati, Jacopo | Polytechnique Montreal
    Author: Beltrame, Giovanni | Ecole Polytechnique De Montreal
    Author: Prorok, Amanda | University of Cambridge
 
    keyword: Multi-Robot Systems

    Abstract : Privacy is an important facet of defence against adversaries. In this letter, we introduce the problem of private flocking. We consider a team of mobile robots flocking in the presence of an adversary, who is able to observe all robots' trajectories, and who is interested in identifying the leader. We present a method that generates private flocking controllers that hide the identity of the leader robot. Our approach towards privacy leverages a data-driven adversarial co-optimization scheme. We design a mechanism that optimizes flocking control parameters, such that leader inference is hindered. As the flocking performance improves, we succes- sively train an adversarial discriminator that tries to infer the identity of the leader robot. To evaluate the performance of our co-optimization scheme, we investigate different classes of reference trajectories. Although it is reasonable to assume that there is an inherent trade-off between flocking performance and privacy, our results demonstrate that we are able to achieve high flocking performance and simultaneously reduce the risk of revealing the leader.


- Subspace Projectors for State-Constrained Multi-Robot Consensus

    Author: Morbidi, Fabio | Université De Picardie Jules Verne
 
    keyword: Multi-Robot Systems; Distributed Robot Systems; Cooperating Robots

    Abstract : In this paper, we study the state-constrained consensus problem and introduce a new family of distributed algorithms based on subspace projection methods which are simple to implement and which preserve, under some suitable conditions, the consensus value of the original discrete-time agreement protocol. The proposed theory is supported by extensive numerical experiments for the constrained 2D rendezvous of single-integrator robots.

- Multi-Agent Task Allocation Using Cross-Entropy Temporal Logic Optimization

    Author: Banks, Christopher | Georgia Institute of Technology
    Author: Wilson, Sean | Georgia Institute of Technology
    Author: Coogan, Samuel | Georgia Tech
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Multi-Robot Systems; Formal Methods in Robotics and Automation; Aerial Systems: Applications

    Abstract : In this paper, we propose a graph-based search method to optimally allocate tasks to a team of robots given a global task specification. In particular, we define these agents as discrete transition systems. In order to allocate tasks to the team of robots, we decompose finite linear temporal logic (LTL) specifications and consider agent specific cost functions. We propose to use the stochastic optimization technique, cross entropy, to optimize over this cost function. The multi-agent task allocation cross-entropy (MTAC-E) algorithm is developed to determine both when it is optimal to switch to a new agent to complete a task and minimize the costs associated with individual agent trajectories. The proposed algorithm is verified in simulation and experimental results are included.

- Adaptive Task Allocation for Heterogeneous Multi-Robot Teams with Evolving and Unknown Robot Capabilities

    Author: Emam, Yousef | Mr
    Author: Mayya, Siddharth | University of Pennsylvania
    Author: Notomista, Gennaro | Georgia Institute of Technology
    Author: Bohannon, Addison | CCDC Army Research Laboratory
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Multi-Robot Systems; Networked Robots; Learning and Adaptive Systems

    Abstract : For multi-robot teams with heterogeneous capabilities, typical task allocation methods assign tasks to robots based on the suitability of the robots to perform certain tasks as well as the requirements of the task itself. However, in real-world deployments of robot teams, the suitability of a robot might be unknown prior to deployment, or might vary due to changing environmental conditions. This paper presents an adaptive task allocation and task execution framework which allows individual robots to prioritize among tasks while explicitly taking into account their efficacy at performing the tasks---the parameters of which might be unknown before deployment and/or might vary over time. Such a specialization parameter---encoding the effectiveness of a given robot towards a task---is updated on-the-fly, allowing our algorithm to reassign tasks among robots with the aim of executing them. The developed framework requires no explicit model of the changing environment or of the unknown robot capabilities---it only takes into account the progress made by the robots at completing the tasks. Simulations and experiments demonstrate the efficacy of the proposed approach during variations in environmental conditions and when robot capabilities are unknown before deployment.

- Mobile Wireless Network Infrastructure on Demand

    Author: Mox, Daniel | University of Pennsylvania
    Author: Calvo-Fullana, Miguel | University of Pennsylvania
    Author: Gerasimenko, Mikhail | Tampere University
    Author: Fink, Jonathan | US Army Research Laborator
    Author: Kumar, Vijay | University of Pennsylvania
    Author: Ribeiro, Alejandro | University of Pennsylvania
 
    keyword: Multi-Robot Systems; Networked Robots

    Abstract : In this work, we introduce Mobile Wireless Infrastructure on Demand: a framework for providing wireless connectivity to multi-robot teams via autonomously reconfiguring ad-hoc networks. In many cases, previous multi-agent systems either assumed the availability of existing communication infrastructure or were required to create a network in addition to completing their objective. Instead our system explicitly assumes the responsibility of creating and sustaining a wireless network capable of satisfying end-to-end communication requirements of a team of agents, called the task team, performing an arbitrary objective. To accomplish this goal, we propose a joint optimization framework that alternates between finding optimal network routes to support data flows between the task agents and improving the performance of the network by repositioning a collection of mobile relay nodes referred to as the network team. We demonstrate our approach with simulations and experiments wherein wireless connectivity is provided to patrolling task agents.

- Monitoring Over the Long Term: Intermittent Deployment and Sensing Strategies for Multi-Robot Teams

    Author: Liu, Jun | Virginia Tech
    Author: Williams, Ryan | Virginia Polytechnic Institute and State University
 
    keyword: Multi-Robot Systems

    Abstract : In this paper, we formulate and solve the intermittent deployment problem, which yields strategies that couple when heterogeneous robots should sense an environmental process, with where a deployed team should sense in the environment. As a motivation, suppose that a spatiotemporal process is slowly evolving and must be monitored by a multi-robot team, e.g., UAVs monitoring pasturelands in a precision agriculture context. In such a case, an intermittent deployment strategy is necessary as persistent deployment or monitoring is not cost-efficient for a slowly evolving process. At the same time, the problem of where to sense once deployed must be solved as process observations yield useful feedback for determining effective future deployment and monitoring decisions. In this context, we model the environmental process to be monitored as a spatiotemporal Gaussian process with mutual information as a measurement criterion. To make the sensing resource-efficient, we demonstrate how to use matroid constraints to impose a diverse set of homogeneous and heterogeneous constraints. In addition, to reflect the cost-sensitive nature of real-world applications, we apply budgets on the cost of deployed heterogeneous robot teams. To solve the resulting problem, we exploit the theories of submodular optimization and matroids and present a greedy algorithm with bounds on sub-optimality. Finally, Monte Carlo simulations demonstrate the correctness of the proposed method.

- Multi-Robot Coordination for Estimation and Coverage of Unknown Spatial Fields

    Author: Benevento, Alessia | University of Salento
    Author: Santos, Mar�a | Georgia Institute of Technology
    Author: Notarstefano, Giuseppe | University of Bologna
    Author: Paynabar, Kamran | Georgia Tech
    Author: Bloch, Matthieu | Georgia Institute of Technology
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Multi-Robot Systems; Learning and Adaptive Systems; Optimization and Optimal Control

    Abstract : We present an algorithm for multi-robot coverage of an initially unknown spatial scalar field characterized by a density function, whereby a team of robots simultaneously estimates and optimizes its coverage of the density function over the domain. The proposed algorithm borrows powerful concepts from Bayesian Optimization with Gaussian Processes that, when combined with control laws to achieve centroidal Voronoi tessellation, give rise to an adaptive sequential sampling method to explore and cover the domain. The crux of the approach is to apply a control law using a surrogate function of the true density function, which is then successively refined as robots gather more samples for estimation. The performance of the algorithm is justified theoretically under slightly idealized assumptions, by demonstrating asymptotic no-regret with respect to the coverage obtained with a known density function. The performance is also evaluated in simulation and on the Robotarium with small teams of robots, confirming the good performance suggested by the theoretical analysis.

- Dense R-Robust Formations on Lattices
 
    Author: Guerrero-Bonilla, Luis | KTH Royal Institute of Technology
    Author: Salda�a, David | Lehigh University
    Author: Kumar, Vijay | University of Pennsylvania
 
    keyword: Multi-Robot Systems; Networked Robots; Cooperating Robots

    Abstract : Robot networks are susceptible to fail under the presence of malicious or defective robots. Resilient networks in the literature require high connectivity and large communication ranges, leading to high energy consumption in the communication network. This paper presents robot formations with guaranteed resiliency that use smaller communication ranges than previous results in the literature. The formations can be built on triangular and square lattices in the plane, and cubic lattices in the three-dimensional space. We support our theoretical framework with simulations.

- Optimizing Topologies for Probabilistically Secure Multi-Robot Systems

    Author: Wehbe, Remy | Virginia Tech
    Author: Williams, Ryan | Virginia Polytechnic Institute and State University
 
    keyword: Multi-Robot Systems; Networked Robots; Distributed Robot Systems

    Abstract : In this paper, we optimize the interaction graph of a multi-robot system (MRS) by maximizing its probability of security while requiring the MRS to have the fewest edges possible. Edges that represent robot interactions exist according to a probability distribution and security is defined using the control theoretic notion of left invertibility. To compute an optimal solution to our problem, we first start by reducing our problem to a variation of the rooted k-connections problem using three graph transformations. Then, we apply a weighted matroid intersection algorithm (WMIA) on matroids defined on the edge set of the interaction graph. Although the optimal solution can be found in polynomial time, MRSs are dynamic and their topologies may change faster than the rate at which the optimal security solution can be found. To cope with dynamic behavior, we present two heuristics that relax optimality but execute with much lower time complexity. Finally, we validate our results through Monte Carlo simulations.

- Efficient Communication in Large Multi-Robot Networks

    Author: Dutta, Ayan | University of North Florida
    Author: Ghosh, Anirban | University of North Florida
    Author: Sisley, Stephen | University of North Florida
    Author: Kreidl, Patrick | University of North Florida
 
    keyword: Multi-Robot Systems; Planning, Scheduling and Coordination

    Abstract : To achieve coordination in a multi-robot system, the robots typically resort to some form of communication among each other. In most of the multi-robot coordination frameworks, high-level coordination strategies are studied but "how" the ground-level communication takes place, is assumed to be taken care of by another program. In this paper, we study the communication routing problem for large multi-robot systems where the robots have limited communication ranges. The objective is to send a message from a robot to another in the network, routed through a low number of other robots. To this end, we propose a communication model between any pair of robots using peer-to-peer radio communication. Our proposed model is generic to any type of message and guarantees a low hop routing between any pair of robots in this network. These help the robots to exchange large messages (e.g., multispectral images) in a short amount of time. Results show that our proposed approach easily scales up to 1000 robots while drastically reducing the space complexity for maintaining the network information.

- CyPhyHouse: A Programming, Simulation, and Deployment Toolchain for Heterogeneous Distributed Coordination

    Author: Ghosh, Ritwika | University of Illinois at Urbana-Champaign
    Author: Jansch-Porto, Joao Paulo | University of Illinois at Urbana-Champaign
    Author: Hsieh, Chiao | University of Illinois at Urbana-Champaign
    Author: Gosse, Amelia | University of Illinois at Urbana-Champaign
    Author: Jiang, Minghao | University of Illinois at Urbana-Champaign
    Author: Taylor, Hebron | University of Illinois at Urbana-Champaign
    Author: Du, Peter | University of Illinois at Urbana Champaign
    Author: Mitra, Sayan | University of Ilinois, Urbana Champagne
    Author: Dullerud, Geir E. | University of Illinois
 
    keyword: Multi-Robot Systems; Autonomous Agents; Distributed Robot Systems

    Abstract : Programming languages, libraries, and development tools have transformed the application development processes for mobile computing and machine learning. This paper introduces CyPhyHouse---a toolchain that aims to provide similar programming, debugging, and deployment benefits for distributed mobile robotic applications. Users can develop hardware-agnostic, distributed applications using the high-level, event-driven Koord programming language, without requiring expertise in controller design or distributed network protocols. The modular, platform-independent middleware of CyPhyHouse implements these functionalities using standard algorithms for path planning (RRT), control (MPC), mutual exclusion, etc. A high-fidelity, scalable, multi-threaded simulator for Koord applications is developed to simulate the same application code for dozens of heterogeneous agents. The same compiled code can also be deployed on heterogeneous mobile platforms. The effectiveness of CyPhyHouse in improving the design cycles is explicitly illustrated in a robotic testbed through development, simulation, and deployment of a distributed task allocation application on in-house ground and aerial vehicles.

- Chance Constrained Simultaneous Path Planning and Task Assignment for Multiple Robots with Stochastic Path Costs

    Author: Yang, Fan | Stony Brook University
    Author: Chakraborty, Nilanjan | Stony Brook University
 
    keyword: Multi-Robot Systems; Optimization and Optimal Control; Planning, Scheduling and Coordination

    Abstract : We present a novel algorithm for simultaneous task assignment and path planning on a graph (or roadmap) with stochastic edge costs. In this problem, the initially unassigned robots and tasks are located at known positions in a roadmap. We want to assign a unique task to each robot and compute a path for the robot to go to its assigned task location. Given the mean and variance of travel cost of each edge, our goal is to develop algorithms that, with high probability, the total path cost of the robot team is below a minimum value in any realization of the stochastic travel costs. We formulate the problem as a chance-constrained simultaneous task assignment and path planning problem (CC-STAP). We prove that the optimal solution of CC-STAP can be obtained by solving a sequence of deterministic simultaneous task assignment and path planning problems in which the travel cost is a linear combination of mean and variance of the edge cost. We show that the deterministic problem can be solved in two steps. In the first step, robots compute the shortest paths to the task locations and in the second step, the robots solve a linear assignment problem with the costs obtained in the first step. We also propose a distributed algorithm that solves CC-STAP near-optimally. We present simulation results on randomly generated networks and data to demonstrate that our algorithm is scalable with the number of robots (or tasks) and the size of the network.

- Optimal Topology Selection for Stable Coordination of Asymmetrically Interacting Multi-Robot Systems

    Author: Mukherjee, Pratik | Virginia Polytechnic Institute and State University
    Author: Santilli, Matteo | Université Degli Studi Roma Tre
    Author: Gasparri, Andrea | Université Degli Studi Roma Tre
    Author: Williams, Ryan | Virginia Polytechnic Institute and State University
 
    keyword: Multi-Robot Systems; Optimization and Optimal Control

    Abstract : In this paper, we address the problem of optimal topology selection for stable coordination of multi-robot systems with asymmetric interactions. This problem arises naturally for multi-robot systems that	interact based on sensing, e.g., with limited field of view (FOV) cameras. From our previous efforts on motion control in such settings, we have shown that not all interaction topologies yield stable coordinated	motion	when asymmetry exists. At	the same time, not all robot-to-robot	interactions are of equal quality,	and thus we seek to optimize asymmetric interaction topologies subject to the constraint that the topology yields stable multi-robot motion. In this context, we formulate an optimal	topology selection problem (OTSP)	as a mixed integer semidefinite programming (MISDP) problem to compute optimal topologies that yield stable coordinated motion.Simulation results are provided to corroborate the effectiveness of the proposed OTSP formulation.


- Representing Multi-Robot Structure through Multimodal Graph Embedding for the Selection of Robot Teams

    Author: Reily, Brian | Colorado School of Mines
    Author: Reardon, Christopher M. | U.S. Army Research Laboratory
    Author: Zhang, Hao | Colorado School of Mines
 
    keyword: Multi-Robot Systems; Swarms

    Abstract : Multi-robot systems of increasing size and complexity are used to solve large-scale problems, such as area exploration and search and rescue. A key decision in human-robot teaming is dividing a multi-robot system into teams to address separate issues or to accomplish a task over a large area. In order to address the problem of selecting teams in a multi-robot system, we propose a new multimodal graph embedding method to construct a unified representation that fuses multiple information modalities to describe and divide a multi-robot system. The relationship modalities are encoded as directed graphs that can encode asymmetrical relationships, which are embedded into a unified representation for each robot. Then, the constructed multimodal representation is used to determine teams based upon unsupervised learning. We perform experiments to evaluate our approach on expert-defined team formations, large-scale simulated multi-robot systems, and a system of physical robots. Experimental results show that our method successfully decides correct teams based on the multifaceted internal structures describing multi-robot systems, and outperforms baseline methods based upon only one mode of information, as well as other graph embedding-based division methods.

- MAMS-A*: Multi-Agent Multi-Scale A*

    Author: Lim, Jaein | Georgia Institute of Technology
    Author: Tsiotras, Panagiotis | Georgia Tech
 
    keyword: Multi-Robot Systems; Motion and Path Planning

    Abstract : We present a multi-scale forward search algorithm for distributed agents to solve single-query shortest path planning problems. Each agent first builds a representation of its own search space of the common environment as a multi-resolution graph, it communicates with the other agents the result of its local search, and it uses received information from other agents to refine its own graph and update the local inconsistency conditions. As a result, all agents attain a common subgraph that includes a provably optimal path in the most informative graph available among all agents, if one exists, without necessarily communicating the entire graph. We prove the completeness and optimality of the proposed algorithm, and present numerical results supporting the advantages of the proposed approach.

- Connectivity Maintenance: Global and Optimized Approach through Control Barrier Functions

    Author: Capelli, Beatrice | University of Modena and Reggio Emilia
    Author: Sabattini, Lorenzo | University of Modena and Reggio Emilia
 
    keyword: Multi-Robot Systems; Networked Robots

    Abstract : Connectivity maintenance is an essential aspect to consider while controlling a multi-robot system. In general, a multi-robot system should be connected to obtain a certain common objective. Connectivity must be kept regardless of the control strategy or the objective of the multi-robot system. Two main methods exist for connectivity maintenance: keep the initial connections (local connectivity) or allow modifications to the initial connections, but always keeping the overall system connected (global connectivity). In this paper we present a method that allows, at the same time, to maintain global connectivity and to implement the desired control strategy (e.g., consensus, formation control, coverage), all in an optimized fashion. For this purpose, we defined and implemented a Control Barrier Function that can incorporate constraints and objectives. We provide a mathematical proof of the method, and we demonstrate its versatility with simulations of different applications.

- Controller Synthesis for Infinitesimally Shape-Similar Formations

    Author: Buckley, Ian | Georgia Institute of Technology
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Multi-Robot Systems; Networked Robots

    Abstract : The interplay between network topology and the interaction modalities of a multi-robot team fundamentally impact the types of formations that can be achieved. To explore the trade-offs between network structure and the sensing and communication capabilities of individual robots, this paper applies controller synthesis to formation control of infinitesimally shape-similar frameworks, for which maintaining the relative angles between robots ensures invariance of the framework to translation, rotation, and uniform scaling. Beginning with the development of a controller for the sole purpose of maintaining the formation, the controller-synthesis approach is introduced as a mechanism for incorporating user-designated objectives while ensuring that the formation is maintained. Both centralized and decentralized formulations of the synthesized controller are presented, the resulting sensing and communication requirements are discussed, and the method is demonstrated on a team of differential-drive robots.

- A Distributed Source Term Estimation Algorithm for Multi-Robot Systems

    Author: Rahbar, Faezeh | EPFL
    Author: Martinoli, Alcherio | EPFL
 
    keyword: Multi-Robot Systems; Autonomous Agents; Planning, Scheduling and Coordination

    Abstract : Finding sources of airborne chemicals with mobile sensing systems finds applications in safety, security, and emergency situations related to medical, domestic, and environmental domains. Given the often critical nature of all the applications, it is important to reduce the amount of time necessary to accomplish this task through intelligent systems and algorithms. In this paper, we extend a previously presented algorithm based on source term estimation for odor source localization for homogeneous multi-robot systems. By gradually increasing the level of coordination among multiple mobile robots, we study the benefits of a distributed system on reducing the amount of time and resources necessary to achieve the task at hand. The method has been evaluated systematically through high-fidelity simulations and in a wind tunnel emulating realistic and repeatable conditions in different coordination scenarios and with different number of robots.

- Weighted Buffered Voronoi Cells for Distributed Semi-Cooperative Behavior

    Author: Pierson, Alyssa | Massachusetts Institute of Technology
    Author: Schwarting, Wilko | Massachusetts Institute of Technology (MIT)
    Author: Karaman, Sertac | Massachusetts Institute of Technology
    Author: Rus, Daniela | MIT
 
    keyword: Multi-Robot Systems; Collision Avoidance; Path Planning for Multiple Mobile Robots or Agents

    Abstract : This paper introduces the Weighted Buffered Voronoi tessellation, which allows us to define distributed, semi-cooperative multi-agent navigation policies with guarantees on collision avoidance. We generate the Voronoi cells with dynamic weights that bias the boundary towards the agent with the lower relative weight while always maintaining a buffered distance between two agents. By incorporating agent weights, we can encode selfish or prioritized behavior among agents, where a more selfish agent will have a larger relative cell over less selfish agents. We consider this semi-cooperative since agents do not cooperate in symmetric ways. Furthermore, when all agents start in a collision-free configuration and plan their control actions within their cells, we prove that no agents will collide. Simulations demonstrate the performance of our algorithm for agents navigating to goal locations in a position-swapping game. We observe that agents with more egoistic weights consistently travel shorter paths to their goal than more altruistic agents.
- Collaborative Multi-Robot Localization in Natural Terrain

    Author: Wiktor, Adam | Stanford University
    Author: Rock, Stephen | Stanford
 
    keyword: Multi-Robot Systems; Localization; Sensor Fusion

    Abstract : This paper presents a novel filter architecture that allows a team of vehicles to collaboratively localize using Terrain Relative Navigation (TRN). The work explores several causes of measurement correlation that preclude the use of traditional estimators, and proposes an estimator structure that eliminates one source of measurement correlation while properly incorporating others through the use of Covariance Intersection. The result is a consistent estimator that is able to augment proven TRN techniques with multi-robot information, significantly improving localization for vehicles in uninformative terrain. The approach is demonstrated using field data from an Autonomous Underwater Vehicle (AUV) navigating with TRN in Monterey Bay and simulated inter-vehicle range measurements. In addition, a Monte Carlo simulation was used to quantify the algorithm's performance on one example mission. Monte Carlo results show that a vehicle operating in uninformative terrain has 62% lower localization error when fusing range measurements to two converged AUVs than it would using standard TRN.

- Multi-Robot Control Using Coverage Over Time-Varying Non-Convex Domains

    Author: Xu, Xiaotian | University of Maryland, College Park
    Author: Diaz-Mercado, Yancy | University of Maryland
 
    keyword: Multi-Robot Systems; Path Planning for Multiple Mobile Robots or Agents

    Abstract : This paper addresses the problem of a domain becoming non-convex while using coverage control of a multirobot system over time-varying domains. When the domain moves around in the workspace, its motion and the presence of obstacles might cause it to deform into some non-convex shape, and the robot team should act in a coordinating manner to maintain coverage. The proposed solution is based on a framework for constructing a diffeomorphism to transform a non-convex coverage problem into a convex one. A control law is developed to capture the effects of time variations (e.g., from a time-varying density, time-varying convex hull of the domain and time-varying diffeomorphism) in the system. Analytic expressions of each term in the control law are found for uniform density case. A simulation and robotic implementation are used to validate the proposed algorithm.

- Efficient Large-Scale Multi-Drone Delivery Using Transit Networks

    Author: Choudhury, Shushman | Stanford University
    Author: Solovey, Kiril | Stanford University
    Author: Kochenderfer, Mykel | Stanford University
    Author: Pavone, Marco | Stanford University
 
    keyword: Multi-Robot Systems; Intelligent Transportation Systems; Planning, Scheduling and Coordination

    Abstract : We consider the problem of controlling a large fleet of drones to deliver packages simultaneously across broad urban areas. To conserve energy, drones hop between public transit vehicles (e.g., buses and trams). We design a comprehensive algorithmic framework that strives to minimize the maximum time to complete any delivery. We address the multifaceted complexity of the problem through a two-layer approach. First, the upper layer assigns drones to package delivery sequences with a near-optimal polynomial-time task allocation algorithm. Then, the lower layer executes the allocation by periodically routing the fleet over the transit network while employing efficient bounded-suboptimal multi-agent pathfinding techniques tailored to our setting. Experiments demonstrate the efficiency of our approach on settings with up to 200 drones, 5000 packages, and transit networks with up to 8000 stops in San Francisco and Washington DC. Our results show that the framework computes solutions within a few seconds (up to 2 minutes at most) on commodity hardware, and that drones travel up to 450% of their flight range with public transit.

- Resilience in Multi-Robot Target Tracking through Reconfiguration

    Author: Ramachandran, Ragesh Kumar | University Southern California
    Author: Fronda, Nicole | University of Southern California
    Author: Sukhatme, Gaurav | University of Southern California
 
    keyword: Multi-Robot Systems; Cooperating Robots; Sensor Networks

    Abstract : We address the problem of maintaining resource availability in a networked multi-robot system performing distributed target tracking. In our model, robots are equipped with sensing and computational resources enabling them to track a target's position using a Distributed Kalman Filter (DKF). We use the trace of each robot's sensor measurement noise covariance matrix as a measure of sensing quality. When a robot's sensing quality deteriorates, the systems communication graph is modified by adding edges such that the robot with deteriorating sensor quality may share information with other robots to improve the team's target tracking ability. This computation is performed centrally and is designed to work without a large change in the number of active communication links. We propose two mixed integer semi-definite programming formulations (an �agent-centric� strategy and a �team-centric� strategy) to achieve this goal. We implement both formulations and a greedy strategy in simulation and show that the team- centric strategy outperforms the agent-centric and greedy strategies.

- Teleoperation of Multi-Robot Systems to Relax Topological Constraints

    Author: Sabattini, Lorenzo | University of Modena and Reggio Emilia
    Author: Capelli, Beatrice | University of Modena and Reggio Emilia
    Author: Fantuzzi, Cesare | Université Di Modena E Reggio Emilia
    Author: Secchi, Cristian | Univ. of Modena &amp; Reggio Emilia
 
    keyword: Multi-Robot Systems; Telerobotics and Teleoperation

    Abstract : Multi-robot systems are able to achieve common objectives exchanging information among each other. This is possible exploiting a communication structure, usually modeled as a graph, whose topological properties (such as connectivity) are very relevant in the overall performance of the multi-robot system. When considering mobile robots, such properties can change over time: robots are then controlled to preserve them, thus guaranteeing the possibility, for the overall system, to achieve its goals. This, however, implies limitations on the possible motion patterns of the robots, thus reducing the flexibility of the overall multi-robot system. In this paper we introduce teleoperation as a means to reduce these limitations, allowing temporary violations of topological properties, with the aim of increasing the flexibility of the multi-robot system.

- Eciton Robotica: Design and Algorithms for an Adaptive Self-Assembling Soft Robot Collective

    Author: Malley, Melinda | Harvard University
    Author: Haghighat, Bahar | EPFL
    Author: Houel, Lucie | Ecole Polytechnique Federale De Lausanne (EPFL),
    Author: Nagpal, Radhika | Harvard University
 
    keyword: Multi-Robot Systems; Swarms; Biologically-Inspired Robots

    Abstract : Social insects successfully create bridges, rafts, nests and other structures out of their own bodies and do so with no centralized control system, simply by following local rules. For example, while traversing rough terrain, army ants (genus Eciton) build bridges which grow and dissolve in response to local traffic. Because these self-assembled structures incorporate smart, flexible materials (i.e. ant bodies) and emerge from local behavior, the bridges are adaptive and dynamic. With the goal of realizing robotic collectives with similar features, we designed a hardware system, Eciton robotica, consisting of flexible robots that can climb over each other to assemble compliant structures and communicate locally using vibration. In simulation, we demonstrate self-assembly of structures: using only local rules and information, robots build and dissolve bridges in response to local traffic and varying terrain. Unlike previous self-assembling robotic systems that focused on lattice-based structures and predetermined shapes, our system takes a new approach where soft robots attach to create amorphous structures whose final self-assembled shape can adapt to the needs of the group.

## Modeling, Control, and Learning for Soft Robots

- Learning Robotic Assembly Tasks with Lower Dimensional Systems by Leveraging Softness and Environmental Constraints

    Author: Hamaya, Masashi | OMRON SINIC X Corporation
    Author: Lee, Robert | Australian Centre for Robotic Vision
    Author: Tanaka, Kazutoshi | OMRON SINIC X Corporation
    Author: von Drigalski, Felix Wolf Hans Erich | OMRON SINIC X Corporation
    Author: Nakashima, Chisato | OMRON Corp
    Author: Shibata, Yoshiya | OMRON Corpration
    Author: Ijiri, Yoshihisa | OMRON Corp
 
    keyword: Soft Robot Applications; Modeling, Control, and Learning for Soft Robots; Compliant Assembly

    Abstract : In this study, we present a novel control framework for assembly tasks with a soft robot. Typically, existing hard robots require high frequency controllers and precise force/torque sensors for assembly tasks. The resulting robot system is complex, entailing large amounts of engineering and maintenance. Physical softness allows the robot to interact with the environment easily. We expect soft robots to perform assembly tasks without the need for high frequency force/torque controllers and sensors. However, specific data-driven approaches are needed to deal with complex models involving nonlinearity and hysteresis. If we were to apply these approaches directly, we would be required to collect very large amounts of training data. To solve this problem, we argue that by leveraging softness and environmental constraints, a robot can complete tasks in lower dimensional state and action spaces, which could greatly facilitate the exploration of appropriate assembly skills. Then, we apply a highly efficient model-based reinforcement learning method to lower dimensional systems. To verify our method, we perform a simulation for peg-in-hole tasks. The results show that our method learns the appropriate skills faster than an approach that does not consider lower dimensional systems. Moreover, we demonstrate that our method works on a real robot equipped with a compliant module on the wrist.

- Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics Using NVIDIA CUDA

    Author: Austin, Jacob | Columbia University
    Author: Corrales-Fatou, Rafael | Imperial College London
    Author: Wyetzner, Sofia | Columbia University
    Author: Lipson, Hod | Columbia University
 
    keyword: Software, Middleware and Programming Environments; Modeling, Control, and Learning for Soft Robots; Simulation and Animation

    Abstract : While most robotics simulation libraries are built for low-dimensional and intrinsically serial tasks, soft-body and multi-agent robotics have created a demand for simulation environments that can model many interacting bodies in parallel. Despite the increasing interest in these fields, no existing simulation library addresses the challenge of providing a unified, highly-parallelized, GPU-accelerated interface for simulating large robotic systems. Titan is a versatile CUDA-based C++ robotics simulation library that employs a novel asynchronous computing model for GPU-accelerated simulations of robotics primitives. The innovative GPU architecture design permits simultaneous optimization and control on the CPU while the GPU runs asynchronously, enabling rapid topology optimization and reinforcement learning iterations. Kinematics are solved with a massively parallel integration scheme that incorporates constraints and environmental forces. We report dramatically improved performance over CPU baselines, simulating as many as 300 million primitive updates per second, while allowing flexibility for a wide range of research applications. We present several applications of Titan to high-performance simulations of soft-body and multi-agent robots.

- Motion Planning with Competency-Aware Transition Models for Underactuated Adaptive Hands

    Author: Sintov, Avishai | Tel-Aviv University
    Author: Kimmel, Andrew | Rutgers University
    Author: Bekris, Kostas E. | Rutgers, the State University of New Jersey
    Author: Boularias, Abdeslam | Rutgers University
 
    keyword: Modeling, Control, and Learning for Soft Robots; Dexterous Manipulation; Motion and Path Planning

    Abstract : Underactuated adaptive hands simplifying grasping tasks but it can be difficult to model their interactions with objects during in-hand manipulation. Learned data-driven models, however, have been recently shown to be efficient in motion planning and control of such hands. Still, the accuracy of the models is limited even with the addition of more data. This becomes important for long horizon predictions where errors are accumulated along the length of the path. Instead of throwing more data into learning the transition model, this work proposes to rather invest a portion of the training data in a {it critic} model. The critic is trained to estimate the error of the transition model given a state and a sequence of future actions, along with information of past actions. The critic is used to reformulate the cost function of an asymptotically optimal motion planner. Given the critic, the planner directs planned paths to less erroneous regions in the state space. The approach is evaluated against standard planning on simulated and real underactuated hands. The results show that it outperforms an alternative where all the available data is used for training the transition model, without a critic.

- Learning to Walk a Tripod Mobile Robot Using Nonlinear Soft Vibration Actuators with Entropy Adaptive Reinforcement Learning

    Author: Kim, Jae In | Seoul National University
    Author: Hong, Mineui | Seoul National University
    Author: Lee, Kyungjae | Seoul National University
    Author: Kim, DongWook | Seoul National University
    Author: Park, Yong-Lae | Seoul National University
    Author: Oh, Songhwai | Seoul National University
 
    keyword: Modeling, Control, and Learning for Soft Robots; Hydraulic/Pneumatic Actuators; Motion and Path Planning

    Abstract : Soft mobile robots have shown great potential in unstructured and confined environments by taking advantage of their excellent adaptability and high dexterity. However, there are several issues to be addressed in terms of actuating speed and controllability of soft robots. In this paper, a new vibration actuator is proposed using the nonlinear stiffness characteristic of the hyper-elastic material in order to make the actuator vibrate continuously, and an advanced soft mobile robot is presented which has a high degree of freedom of movement. However, since the dynamics model of a soft mobile robot is generally intractable, it is difficult to design a controller for the robot. In this regard, we present a method to train a controller, using<p>our novel reinforcement learning (RL) algorithm called adaptive soft actor-critic (ASAC). ASAC gradually reduces a parameter called the entropy temperature, which regulates the entropy of the control policy.</p><p>By doing so, the proposed method can narrow down the search space during the training, and reduce the duration of the demanding data collection processes in the real-world experiment. For the verification</p><p>of the robustness and controllability of our robot and RL algorithm, zig-zagging path tracking and obstacle avoidance experiments were conducted, and the robot successfully finished the missions with only an hour of training time.

- Time Generalization of Trajectories Learned on Articulated Soft Robots

    Author: Angelini, Franco | University of Pisa
    Author: Mengacci, Riccardo | Université Di Pisa
    Author: Della Santina, Cosimo | Massachusetts Institute of Technology
    Author: Catalano, Manuel Giuseppe | Istituto Italiano Di Tecnologia
    Author: Garabini, Manolo | Université Di Pisa
    Author: Bicchi, Antonio | Université Di Pisa
    Author: Grioli, Giorgio | Istituto Italiano Di Tecnologia
 
    keyword: Natural Machine Motion; Motion Control; Flexible Robots

    Abstract : To avoid feedback-related stiffening of articulated soft robots, a substantive feedforward contribution is crucial. However, obtaining reliable feedforward actions requires very accurate models, which are not always available for soft robots. Learning-based approaches are a promising solution to the problem. They proved to be an effective strategy achieving good tracking performance, while preserving the system intrinsic compliance. Nevertheless, learning methods require rich data sets, and issues of scalability and generalization still remain to be solved. This paper proposes a method to generalize learned control actions to execute a desired trajectory with different velocities - with the ultimate goal of making these learning-based architectures sample efficient. More specifically, we prove that the knowledge of how to execute a same trajectory at five different speeds is necessary and sufficient to execute the same trajectory at any velocity - without any knowledge of the model. We also give a simple constructive way to calculate this new feedforward action. The effectiveness of the proposed technique is validated in extensive simulation on a Baxter robot with soft springs playing a drum, and experimentally on a VSA double pendulum performing swinging motions.

- A Probabilistic Model-Based Online Learning Optimal Control Algorithm for Soft Pneumatic Actuators

    Author: Tang, ZhiQiang | The Chinese University of Hong Kong
    Author: Heung, Ho Lam | The Chinese University of Hong Kong
    Author: Tong, Kai Yu | The Chinese University of Hong Kong
    Author: Li, Zheng | The Chinese University of Hong Kong
 
    keyword: Modeling, Control, and Learning for Soft Robots; Model Learning for Control; Optimization and Optimal Control

    Abstract : Soft robots are increasingly being employed in different fields and various designs are created to satisfy relevant requirements. The wide ranges of design bring challenges to soft robotic control in that a unified control framework is difficult to derive. Traditional model-driven approaches for soft robots are usually design-specific which highly depend on specific design structures. Our approach to such challenges involves a probabilistic model that learns a mapping from the soft actuator states and controls to the next states. Then an optimal control policy is derived by minimizing a cost function based on the probabilistic model. We demonstrate the efficiency of our approach through simulations with parameter analysis and real-robot experiments involving three different designs of soft pneumatic actuators. Comparisons with previous model-based controllers are also provided to show advantages of the proposed method. Overall, this work provides a promising design-independent control approach for the soft robotics community.

- Rigid-Soft Interactive Learning for Robust Grasping

    Author: Yang, Linhan | Southern University of Science and Technology
    Author: Wan, Fang | Ancora Spring Inc
    Author: Wang, Haokun | Southern University of Science and Technology
    Author: Liu, Xiaobo | Southern University of Science and Technology
    Author: Liu, Yujia | The University of Hong Kong
    Author: Pan, Jia | University of Hong Kong
    Author: Song, Chaoyang | Southern University of Science and Technology
 
    keyword: Modeling, Control, and Learning for Soft Robots; Grasping; Multifingered Hands

    Abstract : Inspired by widely used soft fingers on grasping, we propose a method of rigid-soft interactive learning, aiming at reducing the time of data collection. In this paper, we classify the interaction categories into Rigid-Rigid, Rigid-Soft, Soft-Rigid according to the interaction surface between grippers and target objects. We find experimental evidence that the interaction types between grippers and target objects play an essential role in the learning methods. We use soft, stuffed toys for training, instead of everyday objects, to reduce the integration complexity and computational burden and exploit such rigid-soft interaction by changing the gripper fingers to the soft ones when dealing with rigid, daily-life items such as the Yale-CMU-Berkeley (YCB) objects. With a small data collection of 5K picking attempts in total, our results suggest that such Rigid-Soft and Soft-Rigid interactions are transferable. Moreover, the combination of different grasp types shows better performance on the grasping test. We achieve the best grasping performance at 97.5% for easy YCB objects and 81.3% for difficult YCB objects while using a precise grasp with a two-soft-finger gripper to collect training data and power grasp with a four-soft-finger gripper to test.

- Model and Data Based Approaches to the Control of Tensegrity Robots

    Author: Wang, Ran | Texas A&amp;M University
    Author: Goyal, Raman | Texas A&amp;M University
    Author: Chakravorty, Suman | Texas A&amp;M University
    Author: Skelton, Robert | Texas A&amp;M University
 
    keyword: Modeling, Control, and Learning for Soft Robots; Optimization and Optimal Control; Motion and Path Planning

    Abstract : This paper proposes two approaches to control the shape of the structure or the position of the end effector for a soft-robotic application. The first approach is a model based approach where the non-linear dynamics of the tensegrity system is used to regulate position, velocity and acceleration to the specified reference trajectory. The formulation uses state feedback to obtain the solution for the control (tension in the strings) as a linear programming problem. The other model-free approach is a novel decoupled data based control (D2C) which first optimizes a deterministic open-loop trajectory using a blackbox (no actual model) simulation model and then develops a linear quadratic regulator around the linearized open-loop trajectory. A 2-dimensional tensegrity robotic reacher is used to compare the results for both the approaches for a given cost function. The D2C approach is also used to study two more complex tensegrity examples whose dynamics is difficult to model analytically.

- Stiffness Imaging with a Continuum Appendage: Real-Time Shape and Tip Force Estimation from Base Load Readings

    Author: Sadati, Seyedmohammadhadi | King's College London
    Author: Shiva, Ali | King's College London
    Author: Herzig, Nicolas | University of Sheffield
    Author: Rucker, Caleb | University of Tennessee
    Author: Hauser, Helmut | University of Bristol
    Author: Walker, Ian | Clemson University
    Author: Bergeles, Christos | King's College London
    Author: Althoefer, Kaspar | Queen Mary University of London
    Author: Nanayakkara, Thrishantha | Imperial College London
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Robot Applications; Medical Robots and Systems

    Abstract : In this paper, we propose benefiting from load readings at the base of a continuum appendage for real-time forward integration of Cosserat rod model with application in configuration and tip load estimation. The application of this method is successfully tested for stiffness imaging of a soft tissue, using a 3-DOF hydraulically actuated braided continuum appendage. Multiple probing runs with different actuation pressures are used for mapping the tissue surface shape and directional linear stiffness, as well as detecting non-homogeneous regions, e.g. a hard nodule embedded in a soft silicon tissue phantom. Readings from a 6-axis force sensor at the tip is used for comparison and verification. As a result, the tip force is estimated with 0.016-0.037 N (7-20%) mean error in the probing and 0.02-0.1 N (6-12%) in the indentation direction, 0.17 mm (14%) mean error is achieved in estimating the surface profile, and 3.415 [N/m] (10-16%) mean error is observed in evaluating tissue directional stiffness, depending on the appendage actuation. We observed that if the appendage bends against the slider motion (toward the probing direction), it provides better horizontal stiffness estimation and better estimation in the perpendicular direction is achieved when it bends toward the slider motion (against the probing direction). In comparison with a rigid probe, &#8776; 10 times smaller stiffness and &#8776; 7 times larger mean standard deviation values were observed.

- Sim-To-Real Transfer Learning Approach for Tracking Multi-DOF Ankle Motions Using Soft Strain Sensors

    Author: Park, Hyunkyu | Korea Advanced Institute of Science and Technology
    Author: Cho, Junhwi | KAIST
    Author: Park, Junghoon | KAIST
    Author: Na, Youngjin | Sookmyung Women's University
    Author: Kim, Jung | KAIST
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Sensors and Actuators; Soft Robot Applications

    Abstract : A data-driven approach has recently been investigated for identifying human joint angles by means of soft strain sensors because of the corresponding modeling difficulty. However, this approach commonly incurs a high computational burden due to the voluminous amount of data required and the time-series-oriented network architecture. Moreover, the nature of soft sensors makes the problem worse due to the inherent nonlinearity and hysteresis of the material. In this study, we developed a novel wearable sensing brace design for measuring multiple degrees of freedom (DOFs) ankle motions to minimize hysteresis and to improve the measurement repeatability and developed a computationally efficient calibration method based on sim-to-real transfer learning. By attaching the soft sensors to shin links rather than directly to the ankle joint, the effects of external disturbances during joint motions were minimized. To calibrate the sensors to body motions, transfer learning was used based on the results from musculoskeletal simulation(OpenSim) and sensor data. The average tracking error for ankle motions using the proposed method was found to be 12.02� for five healthy subjects, while the direct deep neural network approach showed an error of 17.88�. The proposed method could be used to calibrate the soft sensors with 1000 times faster training spee d while maintaining comparable tracking accuracy with a smaller amount of data.

- Model-Based Pose Control of Inflatable Eversion Robot with Variable Stiffness

    Author: Ataka, Ahmad | Queen Mary University of London
    Author: Abrar, Taqi | Queen Mary University of London
    Author: Putzu, Fabrizio | Queen Mary University of London
    Author: Godaba, Hareesh | Queen Mary University of London
    Author: Althoefer, Kaspar | Queen Mary University of London
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Robot Applications; Motion Control

    Abstract : Plant-inspired inflatable eversion robots with their tip growing behaviour have recently emerged. Because they extend from the tip, eversion robots are particularly suitable for applications that require reaching into remote places through narrow openings. Besides, they can vary their structural stiffness. Despite these essential properties which make the eversion robot a promising candidate for applications involving cluttered environments and tight spaces, controlling their motion especially laterally has not been investigated in depth. In this paper, we present a new approach based on model-based kinematics to control the eversion robot's tip position and orientation. Our control approach is based on Euler-Bernoulli beam theory which takes into account the effect of the internal inflation pressure to model each robot bending segment for various conditions of structural stiffness. We determined the parameters of our bending model by performing a least-square technique based on the pressure-bending data acquired from an experimental study. The model is then used to develop a pose controller for the tip of our eversion robot. Experimental results show that the proposed control strategy is capable of guiding the tip of the eversion robot to reach a desired position and orientation whilst varying its structural stiffness.

- Learning to Control Reconfigurable Staged Soft Arms

    Author: Nicolai, Austin | Oregon State University
    Author: Olson, Gina | Oregon State University
    Author: Menguc, Yigit | Facebook Reality Labs
    Author: Hollinger, Geoffrey | Oregon State University
 
    keyword: Modeling, Control, and Learning for Soft Robots; Deep Learning in Robotics and Automation

    Abstract : In this work, we present a novel approach for modeling, and classifying between, the system load states introduced when constructing staged soft arm configurations. Through a two stage approach: (1) an LSTM calibration routine is used to identify the current load state then (2) a control input generation step combines a generalized quasistatic model with the learned load model. Our experiments show that accounting for system load allows us to more accurately control tapered arm configurations. We analyze the performance of our method using soft robotic actuators and show it is capable of classifying between different arm configurations at a rate greater than 95%. Additionally, our method is capable of reducing the end-effector error of quasistatic model only control to within 1 cm of our controller baseline.

- Open-Loop Position Control in Collaborative, Modular Variable-Stiffness-Link (VSL) Robots

    Author: Gandarias, Juan M. | University of Malaga
    Author: Wang, Yongjing | University College London
    Author: Stilli, Agostino | University College London
    Author: Garc�a-Cerezo, Alfonso | University of Malaga
    Author: Gomez de Gabriel, Jesus Manuel | Universidad De Malaga
    Author: Wurdemann, Helge Arne | University College London
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Robot Materials and Design; Deep Learning in Robotics and Automation

    Abstract : Collaborative robots open up new avenues in the field of industrial robotics and physical Human-Robot Interaction (pHRI) as they are suitable to work in close approximation with humans. The integration and control of variable stiffness elements allow inherently safe interaction: Apart from notable work on Variable Stiffness Actuators, the concept of Variable-Stiffness-Link (VSL) manipulators promises safety improvements in cases of unintentional physical collision. However, position control of these type of robotic manipulators is challenging for critical task-oriented motions. In this paper, we propose a hybrid, learning based kinematic modelling approach to improve the performance of traditional open-loop position controllers for a modular, collaborative VSL robot. We show that our approach improves the performance of traditional open-loop position controllers for robots with VSL and compensates for position errors, in particular, for lower stiffness values inside the links: Using our upgraded and modular robot, two experiments have been carried out to evaluate the behaviour of the robot during task-oriented motions. Results show that traditional model-based kinematics are not able to accurately control the position of the end-effector: the position error increases with higher loads and lower pressures inside the VSLs. On the other hand, we demonstrate that, using our approach, the VSL robot can outperform the position control compared to a robotic manipulator with 3D print

-  Control of a Silicone Soft Tripod Robot Via Uncertainty Compensation

    Author: Zheng, Gang | INRIA

- Control Oriented Modeling of Soft Robots: The Polynomial Curvature Case

    Author: Della Santina, Cosimo | Massachusetts Institute of Technology
    Author: Rus, Daniela | MIT
 
    keyword: Modeling, Control, and Learning for Soft Robots; Motion Control; Dynamics

    Abstract : The complex nature of soft robot dynamics calls for the development of models specifically tailored on the control application. In this paper we take a first step in this direction, by proposing a dynamic model for slender soft robots taking into account the fully infinite-dimensional dynamical structure of the system. We also contextually introduce a strategy to approximate this model at any level of detail through a finite dimensional system. First, we analyze the main mathematical properties of this model, in the case of lightweight and non lightweight soft robots. Then, we prove that using the constant term of curvature as control output produces a minimum phase system, in this way providing the theoretical support that existing curvature control techniques lack, and at the same time opening up to the use of advanced nonlinear control techniques. Finally, we propose a new controller, the PD-poly, which exploits information on high order deformations, to achieve zero steady state regulation error in presence of gravity and generic non constant curvature conditions.

- Modeling and Analysis of SMA Actuator Embedded in Stretchable Coolant Vascular Pursuing Artificial Muscles

    Author: Jeong, Jaeyeon | Korea Advanced Institute of Science Ane Technology
    Author: Park, Cheol Hoon | Korea Institute of Machinery &amp; Materials
    Author: Kyung, Ki-Uk | Korea Advanced Institute of Science &amp; Technology (KAIST)
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Sensors and Actuators

    Abstract : This paper proposes a muscle-like SMA (Shape Memory Alloy) actuator with an active cooling system for efficient response. An SMA coil spring is embedded into a stretchable coolant vascular for soften structure of robots. In order to design a flexible, lightweight, and fast-response soft actuator with the SMA coil spring and coolant circulation system, a modeling based approach has been conducted. Analysis of coolant effects has been conducted in aspects of heating speed, cooling speed, and energy consumption based on both theoretical and empirical studies. From thermomechanical and heat transfer model between SMA and coolant, the actuation times in the case of heating and cooling phase have been estimated. From experimental results, Mineral oil is selected as the optimal coolant, and the maximum actuation frequency was measured as 0.5Hz for 40% contraction lifting 1kg.

- Distributed Proprioception of 3D Configuration in Soft, Sensorized Robots Via Deep Learning

    Author: Truby, Ryan Landon | Massachusetts Institute of Technology
    Author: Della Santina, Cosimo | Massachusetts Institute of Technology
    Author: Rus, Daniela | MIT
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Sensors and Actuators; Soft Robot Materials and Design

    Abstract : Creating soft robots with sophisticated, autonomous capabilities requires these systems to possess reliable, on-line proprioception of 3D configuration through integrated soft sensors. We introduce a framework for predicting the 3D configuration of a soft robot via deep learning using feedback provided by a soft, proprioceptive sensor skin. Our framework introduces a kirigami-enabled strategy for rapidly sensorizing soft robots using off-the-shelf materials, a general kinematic description for soft robot geometry, and an investigation of neural network designs for predicting soft robot configuration. Even with hysteretic, non-monotonic feedback from the soft piezoresistive sensors, recurrent neural networks show potential for predicting our new kinematic parameters and, in turn, the soft robot's configuration. One trained neural network closely predicts steady-state configuration during operation, though complete dynamic behavior is not fully captured. We validate our methods on a soft robotic arm with 12 discrete actuators and 12 proprioceptive strain sensors. As an essential advance in soft robotic perception, we anticipate our framework will open new avenues towards closed loop control in soft robotics.


- Stable Tool-Use with Flexible Musculoskeletal Hands by Learning the Predictive Model of Sensor State Transition

    Author: Kawaharazuka, Kento | The University of Tokyo
    Author: Tsuzuki, Kei | University of Tokyo
    Author: Onitsuka, Moritaka | The University of Tokyo
    Author: Asano, Yuki | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Kawasaki, Koji | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Modeling, Control, and Learning for Soft Robots; Biomimetics; Tendon/Wire Mechanism

    Abstract : The flexible under-actuated musculoskeletal hand is superior in its adaptability and impact resistance. On the other hand, since the relationship between sensors and actuators cannot be uniquely determined, almost all its controls are based on feedforward controls. When grasping and using a tool, the contact state of the hand gradually changes due to the inertia of the tool or impact of action, and the initial contact state is hardly kept. In this study, we propose a system that trains the predictive network of sensor state transition using the actual robot sensor information, and keeps the initial contact state by a feedback control using the network. We conduct experiments of hammer hitting, vacuuming, and brooming, and verify the effectiveness of this study.

- Learning to Transfer Dynamic Models of Underactuated Soft Robotic Hands

    Author: Schramm, Liam | Rutgers University
    Author: Sintov, Avishai | Tel-Aviv University
    Author: Boularias, Abdeslam | Rutgers University
 
    keyword: Modeling, Control, and Learning for Soft Robots; Learning and Adaptive Systems; Model Learning for Control

    Abstract : Transfer learning is a popular approach to bypassing data limitations in one domain by leveraging data from another domain. This is especially useful in robotics, as it allows practitioners to reduce data collection with physical robots, which can be time-consuming and cause wear and tear. The most common way of doing this with neural networks is to take an existing neural network, and simply train it more with new data. However, we show that in some situations this can lead to significantly worse performance than simply using the transferred model without adaptation. We find that a major cause of these problems is that models trained on small amounts of data can have chaotic or divergent behavior in some regions. We derive an upper bound on the Lyapunov exponent of a trained transition model, and demonstrate two approaches that make use of this insight. Both show significant improvement over traditional fine-tuning. Experiments performed on real underactuated soft robotic hands clearly demonstrate the capability to transfer a dynamic model from one hand to another.

- Periodic Movement Learning in a Soft-Robotic Arm

    Author: Oikonomou, Paris | National Technical University of Athens (NTUA)
    Author: Khamassi, Mehdi | Cnrs / Upmc
    Author: Tzafestas, Costas S. | ICCS - Inst of Communication and Computer Systems
 
    keyword: Modeling, Control, and Learning for Soft Robots

    Abstract : In this paper we introduce a novel technique that aims to dynamically control a modular bio-inspired soft-robotic arm in order to perform cyclic rhythmic patterns. Oscillatory signals are produced at the actuator's level by a central pattern generator (CPG), resulting in the generation of a periodic motion by the robot's end-effector. The proposed controller is based on a model-free neurodynamic scheme and is assigned with the task of training a policy that computes the parameters of the CPG model which generates a trajectory with desired features. The proposed methodology is first evaluated with a simulation model, which successfully reproduces the trained targets. Then experiments are also conducted using the real robot. Both procedures validate the efficiency of the learning architecture to successfully complete these tasks.

- An Input Observer-Based Stiffness Estimation Approach for Flexible Robot Joints

    Author: Fagiolini, Adriano | University of Palermo
    Author: Trumic, Maja | University of Palermo
    Author: Jovanovic, Kosta | University of Belgrade, Serbia
 
    keyword: Modeling, Control, and Learning for Soft Robots; Physical Human-Robot Interaction; Flexible Robots

    Abstract : This paper addresses the stiffness estimation prob- lem for flexible robot joints, driven by variable stiffness actua- tors in antagonistic setups. Due to the difficulties of achieving consistent production of these actuators and the time-varying nature of their internal flexible elements, which are subject to plastic deformation over time, it is currently a challenge to precisely determine the total flexibility torque applied to a robot's joint and the corresponding joint stiffness. Herein, by considering the flexibility torque acting on each motor as an unknown signal and building upon Unknown Input Observer theory, a solution for electrically-driven actuators is proposed, which consists of a linear estimator requiring only knowledge about the positions of the joints and the motors as well as the drive's dynamic parameters. Beyond its linearity advantage, another appealing feature of the solution is the lack of need for torque and velocity sensors. The presented approach is first verified via simulations and then successfully tested on an experimental setup, comprising bidirectional antagonistic variable stiffness actuators.

- Fast Model-Based Contact Patch and Pose Estimation for Highly Deformable Dense-Geometry Tactile Sensors

    Author: Kuppuswamy, Naveen | Toyota Research Institute
    Author: Castro, Alejandro | Toyota Research Institute
    Author: Phillips-Grafflin, Calder | Toyota Research Institute
    Author: Alspach, Alex | Toyota Research Institute
    Author: Tedrake, Russ | Massachusetts Institute of Technology
 
    keyword: Modeling, Control, and Learning for Soft Robots; Perception for Grasping and Manipulation; Force and Tactile Sensing

    Abstract : Modeling deformable contact is a well-known problem in soft robotics and is particularly challenging for compliant interfaces that permit large deformations. We present a model for the behavior of a highly deformable dense geometry sensor in its interaction with objects; the forward model predicts the elastic deformation of a mesh given the pose and geometry of a contacting rigid object. We use this model to develop a fast approximation to solve the inverse problem: estimating the contact patch when the sensor is deformed by arbitrary objects. This inverse model can be easily identified through experiments and is formulated as a sparse Quadratic Program (QP) that can be solved efficiently online. The proposed model serves as the first stage of a pose estimation pipeline for robot manipulation. We demonstrate the proposed inverse model through real-time estimation of contact patches on a contact-rich manipulation problem in which oversized fingers screw a nut onto a bolt, and as part of a complete pipeline for pose-estimation and tracking based on the Iterative Closest Point (ICP) algorithm. Our results demonstrate a path towards realizing soft robots with highly compliant surfaces that perform complex real-world manipulation tasks.

- Mechanism and Model of a Soft Robot for Head Stabilization in Cancer Radiation Therapy

    Author: Ogunmolu, Olalekan | The University of Pennsylvania
    Author: Liu, Xinmin | University of Chicago
    Author: Gans, Nicholas (Nick) | University Texas at Arlington
    Author: Wiersma, Rodney | University of Chicago
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Robot Applications; Medical Robots and Systems

    Abstract : We present a parallel robot mechanism and the constitutive laws that govern the deformation of its constituent soft actuators. Our ultimate goal is the real-time motion-correction of a patient's head deviation from a target pose where the soft actuators control the position of the patient's cranial region on a treatment machine. We describe the mechanism, derive the stress-strain constitutive laws for the individual actuators and the inverse kinematics that prescribes a given deformation, and then present simulation results that validate our mathematical formulation. Our results demonstrate deformations consistent with our radially symmetric displacement formulation under a finite elastic deformation framework.

## Manipulation

- Grasping Unknown Objects by Coupling Deep Reinforcement Learning, Generative Adversarial Networks, and Visual Servoing

    Author: Pedersen, Ole-Magnus | NTNU - Norwegian University of Technology and Science
    Author: Misimi, Ekrem | SINTEF Ocean
    Author: Chaumette, Francois | Inria Rennes-Bretagne Atlantique
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; Visual Servoing

    Abstract : In this paper, we propose a novel approach for transferring a deep reinforcement learning (DRL) grasping agent from simulation to a real robot, without fine tuning in the real world. The approach utilises a CycleGAN to close the reality gap between the simulated and real environments, in a reverse real-to-sim manner, effectively "tricking" the agent into believing it is still in the simulator. Furthermore, a visual servoing (VS) grasping task is added to correct for inaccurate agent gripper pose estimations derived from deep learning. The proposed approach is evaluated by means of real grasping experiments, achieving a success rate of 83 % on previously seen objects, and the same success rate for previously unseen, semi-compliant objects. The robustness of the approach is demonstrated by comparing it with two baselines, DRL plus CycleGAN, and VS only. The results clearly show that our approach outperforms both baselines.

- Incorporating Motion Planning Feasibility Considerations During Task-Agent Assignment to Perform Complex Tasks Using Mobile-Manipulators

    Author: Kabir, Ariyan M | University of Southern California
    Author: Thakar, Shantanu | University of Southern California
    Author: Bhatt, Prahar | University of Southern California
    Author: Malhan, Rishi | University of Southern California
    Author: Rajendran, Pradeep | University of Southern California
    Author: Shah, Brual C. | University of Southern California
    Author: Gupta, Satyandra K. | University of Southern California
 
    keyword: Task Planning; Mobile Manipulation; Motion and Path Planning

    Abstract : Multi-arm mobile manipulators can be represented as a combination of multiple robotic agents from the perspective of task-assignment and motion planning. Depending upon the task, agents might collaborate or work independently. Integrating motion planning with task-agent assignment is a computationally slow process as infeasible assignments can only be detected through expensive motion planning queries. We present three speed-up techniques for addressing this problem- (1) spatial constraint checking using conservative surrogates for motion planners, (2) instantiating symbolic conditions for pruning infeasible assignments, and (3) efficiently caching and reusing previously generated motion plans. We show that the developed method is useful for real-world operations that require complex interaction and coordination among high-DOF robotic agents.

- Learning to Scaffold the Development of Robotic Manipulation Skills

    Author: Shao, Lin | Stanford University
    Author: Migimatsu, Toki | Stanford University
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Intelligent and Flexible Manufacturing; Learning and Adaptive Systems; Deep Learning in Robotics and Automation

    Abstract : Learning contact-rich, robotic manipulation skills is a challenging problem due to the high-dimensionality of the state and action space as well as uncertainty from noisy sensors and inaccurate motor control. To combat these factors and achieve more robust manipulation, humans are actively exploiting contact constraints in the environment. By adopting a similar strategy, robots can also achieve more robust manipulation. In this paper, we enable a robot to autonomously modify its environment and thereby discover how to ease manipulation skill learning. Specifically, we provide the robot with fixtures that it can freely place within the environment. These fixtures provide hard constraints that limit the outcome of robot actions. Thereby, they funnel uncertainty from perception and motor control and scaffold manipulation skill learning. We propose a learning system that consist of two learning loops. In the outer loop, the robot positions the fixture in the workspace. In the inner loop, the robot learns a manipulation skill and after a fixed number of episodes, returns the reward to the outer loop. Thereby, the robot is incentivised to place the fixture such that the inner loop quickly achieves a high reward. We demonstrate our framework both in simulation and the real world on three tasks: peg insertion, wrench manipulation and shallow-depth insertion. We show that manipulation skill learning is dramatically sped up through this way of scaffolding.

- Online Replanning in Belief Space for Partially Observable Task and Motion Problems

    Author: Garrett, Caelan | Massachusetts Institute of Technology
    Author: Paxton, Chris | NVIDIA Research
    Author: Lozano-Perez, Tomas | MIT
    Author: Kaelbling, Leslie | MIT
    Author: Fox, Dieter | University of Washington
 
    keyword: Task Planning; Manipulation Planning; Mobile Manipulation

    Abstract : To solve multi-step manipulation tasks in the real world, an autonomous robot must take actions to observe its environment and react to unexpected observations. This may require opening a drawer to observe its contents or moving an object out of the way to examine the space behind it. Upon receiving a new observation, the robot must update its belief about the world and compute a new plan of action. In this work, we present an online planning and execution system for robots faced with these challenges. We perform deterministic cost-sensitive planning in the space of hybrid belief states to select likely-to-succeed observation actions and continuous control actions. After execution and observation, we replan using our new state estimate. We initially enforce that planner reuses the structure of the unexecuted tail of the last plan. This both improves planning efficiency and ensures that the overall policy does not undo its progress towards achieving the goal. Our approach is able to efficiently solve partially observable problems both in simulation and in a real-world kitchen.

- An Automated Dynamic-Balancing-Inspection Scheme for Wheel Machining

    Author: Hao, Tieng | National Cheng Kung University
    Author: Li, Yu-Yong | National Cheng Kung University
    Author: Tseng, Kuang-Ping | National Cheng Kung University
    Author: Yang, Haw-Ching | National Kaohsiung Univ. of Sci. and Tech
    Author: Cheng, Fan-Tien | National Cheng Kung University
 
    keyword: Intelligent and Flexible Manufacturing

    Abstract : Wheel balance plays an important role in vehicle safety. The existing inspection method for wheel balance mainly relies on the off-machine measurement technique, which is time- and manpower-consuming as the worldwide requirement of the automated production system gradually increases. However, the multi-unbalance causes are difficult to identify due to complex machine structures; and the low signal-noise-ratio between wheel and machine vibration makes traditional handcrafted features difficult to detect wheel unbalance. To overcome these two problems, this paper proposes to a Dynamic-Balancing-Inspection (DBI) scheme which integrates steps of data collection, data preprocessing, ensemble average of Convolution Neural Network (CNN) based models with well-tailored filters and activation functions, to automatically uncover critical information from frequency data and provide a reliable total inspection method. The application of the wheel balance from a practical CNC-machine is adopted to illustrate the performance of the DBI approach.

- Faster Confined Space Manufacturing Teleoperation through Dynamic Autonomy with Task Dynamics Imitation Learning

    Author: Owan, Parker | University of Washington
    Author: Garbini, Joseph | U. of Washington
    Author: Devasia, Santosh | University of Washington
 
    keyword: Intelligent and Flexible Manufacturing; Human Performance Augmentation; Learning and Adaptive Systems

    Abstract : Confined space manufacturing tasks, such as cleaning pilot holes prior to installing fasteners during aircraft wing assembly, currently require human experts to be inside ergonomically-challenging environments. Small rapidly deployable robots can substantially improve manufacturing safety and productivity. However, relatively rapid full automation remains elusive due to high-level of uncertainty in the environment, lack of cost-effective programming for low volume production, and difficulty of deploying adequate number of sensors in the confined space. Moreover, currently, teleoperation with typical levels of training and limited transparency of hardware is too slow for manufacturing applications, requiring experts to spend more time for each task to achieve the same cleaning quality. In this context, the main contribution of this article is to reduce cycle times for remote manufacturing by learning statistical dynamic autonomy from higher quality expert demonstrations in an ideal offline scenario. During the task, to keep cycle times low, the dynamic autonomy imitates the faster expert demonstrations when certain, and employs the slower human teleoperation when uncertain. A user study (n=8) with an experimental robot platform shows that for the same cleaning quality, the dynamic autonomy reduces process completion time by 54.0% and human operator energy expenditure by 80.5% as compared with teleoperation without dynamic autonomy.

- Learning Precise 3D Manipulation from Multiple Uncalibrated Cameras

    Author: Akinola, Iretiayo | Columbia University
    Author: Varley, Jacob | Google
    Author: Kalashnikov, Dmitry | Google Brain
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; RGB-D Perception

    Abstract : In this work, we present an effective end-to-end learning approach for solving precision tasks that are 3D in nature. Our method learns to accomplish these tasks using multiple statically placed but uncalibrated RGB camera views without building an explicit 3D representation such as a pointcloud or voxel grid. This multi-camera approach achieves superior task performance on difficult stacking and insertion tasks compared to single-view baselines. Single view robotic agents struggle from occlusion and challenges in estimating relative poses between points of interest. Operating off explicit 3D representations from multiple depth sensors is often complicated by challenges in camera calibration, obtaining depth maps due to object properties such as reflective surfaces, and slower inference speeds over 3D representations compared to 2D images. Our use of static but uncalibrated cameras does not require camera-robot or camera-camera calibration and is robust to sensor dropout making it easy to setup and resilient to the loss of camera-views after deployment.

- Surfing on an Uncertain Edge: Precision Cutting of Soft Tissue Using Torque-Based Medium Classification

    Author: Straizys, Arturas | University of Edinburgh
    Author: Burke, Michael | University of Edinburgh
    Author: Ramamoorthy, Subramanian | The University of Edinburgh
 
    keyword: Learning and Adaptive Systems

    Abstract : Precision cutting of soft-tissue remains a challenging problem in robotics, due to the complex and unpredictable mechanical behaviour of tissue under manipulation. Here, we consider the challenge of cutting along the boundary between two soft mediums, a problem that is made extremely difficult due to visibility constraints, which means that the precise location of the cutting trajectory is typically unknown. This paper introduces a novel strategy to address this task, using a binary medium classifier trained using joint torque measurements, and a closed loop control law that relies on an error signal compactly encoded in the decision boundary of the classifier. We illustrate this on a grapefruit cutting task, successfully modulating a nominal trajectory fit using dynamic movement primitives to follow the boundary between grapefruit pulp and peel using torque based medium classification. Results show that this control strategy is successful in 72 % of attempts in contrast to control using a nominal trajectory, which only succeeds in 50 % of attempts.

- Dynamic Cloth Manipulation with Deep Reinforcement Learning

    Author: Jangir, Rishabh | Institut De Robòtica I Informàtica Industrial, CSIC-UPC
    Author: Aleny�, Guillem | CSIC-UPC
    Author: Torras, Carme | Csic - Upc
 
    keyword: Deep Learning in Robotics and Automation; Motion Control of Manipulators; Manipulation Planning

    Abstract : In this paper we present a Deep Reinforcement Learning approach to solve dynamic cloth manipulation tasks. Differing from the case of rigid objects, we stress that the followed trajectory (including speed and acceleration) has a decisive influence on the final state of cloth, which can greatly vary even if the positions reached by the grasped points are the same. We explore how goal positions for non-grasped points can be attained through learning adequate trajectories for the grasped points. Our approach uses few demonstrations to improve control policy learning, and a sparse reward approach to avoid engineering complex reward functions. Since perception of textiles is challenging, we also study different state representations to assess the minimum observation space required for learning to succeed. Finally, we compare different combinations of control policy encodings, demonstrations, and sparse reward learning techniques, and show that our proposed approach can learn dynamic cloth manipulation in an efficient way, i.e., using a reduced observation space, a few demonstrations, and a sparse reward.

- Learning to Combine Primitive Skills: A Step towards Versatile Robotic Manipulation

    Author: Strudel, Robin | INRIA Paris
    Author: Pashevich, Alexander | INRIA Grenoble Rhone-Alpes
    Author: Kalevatykh, Igor | INRIA
    Author: Laptev, Ivan | INRIA
    Author: Sivic, Josef | INRIA, Ecole Normale Supérieure, Paris, France
    Author: Schmid, Cordelia | Inria
 
    keyword: Deep Learning in Robotics and Automation; Learning and Adaptive Systems; Visual Learning

    Abstract : Manipulation tasks such as preparing a meal or assembling furniture remain highly challenging for robotics and vision. Traditional task and motion planning (TAMP) methods can solve complex tasks but require full state observability and are not adapted to dynamic scene changes. Recent learning methods can operate directly on visual inputs but typically require many demonstrations and/or task-specific reward engineering. In this work we aim to overcome previous limitations and propose a reinforcement learning (RL) approach to task planning that learns to combine primitive skills. First, compared to previous learning methods, our approach requires neither intermediate rewards nor complete task demonstrations during training. Second, we demonstrate the versatility of our vision-based task planning in challenging settings with temporary occlusions and dynamic scene changes. Third, we propose an efficient training of basic skills from few synthetic demonstrations by exploring recent CNN architectures and data augmentation. Notably, while all of our policies are learned on visual inputs in simulated environments, we demonstrate the successful transfer and high success rates when applying such policies to manipulation tasks on a real UR5 robotic arm.

- Learning to Assemble: Estimating 6D Poses for Robotic Object-Object Manipulation

    Author: Stevsic, Stefan | ETH Zurich
    Author: Christen, Sammy | ETH Zurich
    Author: Hilliges, Otmar | ETH Zurich
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; Computer Vision for Automation

    Abstract : In this paper we propose a robotic vision task with the goal of enabling robots to execute complex assembly tasks in unstructured environments using a camera as the primary sensing device. We formulate the task as an instance of 6D pose estimation of template geometries, to which manipulation objects should be connected. In contrast to the standard 6D pose estimation task, this requires reasoning about local geometry that is surrounded by arbitrary context, such as a power outlet embedded into a wall. We propose a deep learning based approach to solve this task alongside a novel dataset that will enable future work in this direction and can serve as a benchmark. We experimentally show that state-of-the-art 6D pose estimation methods alone are not sufficient to solve the task but that our training procedure significantly improves the performance of deep learning techniques in this context.

- Learning Affordance Space in Physical World for Vision-Based Robotic Object Manipulation

    Author: Wu, Huadong | Sun Yat-Sen University
    Author: Zhang, Zhanpeng | SenseTime Group Limited
    Author: Cheng, Hui | Sun Yat-Sen University
    Author: Yang, Kai | Sun Yat-Sen University
    Author: Liu, Jiaming | Sensetime
    Author: Guo, Ziying | Sun Yat-Sen University
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Automation; Perception for Grasping and Manipulation

    Abstract : What is a proper representation for objects in manipulation? What would human try to perceive when manipulating a new object in a new environment? In fact, instead of focusing on the texture and illumination, human can infer the "affordance" of the objects from vision. Here "affordance" describes the object's intrinsic property that affords a particular type of manipulation. In this work, we investigate whether such affordance can be learned by a deep neural network. In particular, we propose an Affordance Space Perception Network (ASPN) that takes an image as input and outputs an affordance map. Different from existing works that infer the pixel-wise probability affordance map in image space, our affordance is defined in the real world space, thus eliminates the need of hand-eye calibration. In addition, we extend the representation ability of affordance by defining it in a 3D affordance space and propose a novel training strategy to improve the performance. Trained purely with simulation data, ASPN can achieve significant performance in the real world. It is a task-agnostic framework and can handle different objects, scenes and viewpoints. Extensive real-world experiments demonstrate the accuracy and robustness of our approach. We achieve the success rates of 94.2% for singular-object pushing and 92.4% for multiple-object pushing. We achieve the success rates of 97.2% for singular-object grasping and 95.4% for multiple-object grasping, which outperform current SotA methods.
