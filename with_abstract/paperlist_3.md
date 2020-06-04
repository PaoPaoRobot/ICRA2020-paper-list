
# International Conference on Robotics and Automation 2020
 
Welcome to ICRA 2020, the 2020 IEEE International Conference on Robotics and Automation.

This list is edited by [PaopaoRobot, 泡泡机器人](https://github.com/PaoPaoRobot) , the Chinese academic nonprofit organization. Recently we will classify these papers by topics. Welcome to follow our github and our WeChat Public Platform Account ( [paopaorobot_slam](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=100000102&idx=1&sn=0a8a831a4f2c18443dbf436ef5d5ff8c&chksm=6c10bf625b6736748c9612879e166e510f1fe301b72ed5c5d7ecdd0f40726c5d757e975f37af&mpshare=1&scene=1&srcid=0530KxSLjUE9I38yLgfO2nVm&pass_ticket=0aB5tcjeTfmcl9u0eSVzN4Ag4tkpM2RjRFH8DG9vylE%3D#rd) ). Of course, you could contact with [daiwei.song@outlook.com](mailto://daiwei.song@outlook.com)

## Sensor Fusion

- LiStereo: Generate Dense Depth Maps from LIDAR and Stereo Imagery

    Author: Junming, Zhang | University of Michigan
    Author: Srinivasan Ramanagopal, Manikandasriram | University of Michigan
    Author: Vasudevan, Ram | University of Michigan
    Author: Johnson-Roberson, Matthew | University of Michigan
 
    keyword: Sensor Fusion; Computer Vision for Transportation; RGB-D Perception

    Abstract : An accurate depth map of the environment is critical to the safe operation of autonomous robots and vehicles. Currently, either light detection and ranging (LIDAR) or stereo matching algorithms are used to acquire such depth information. However, a high-resolution LIDAR is expensive and produces sparse depth map at large range; stereo matching algorithms are able to generate denser depth maps but are typically less accurate than LIDAR at long range. This paper combines these approaches together to generate high-quality dense depth maps. Unlike previous approaches that are trained using ground-truth labels, the proposed model adopts a self-supervised training process. Experiments show that the proposed method is able to generate high-quality dense depth maps and performs robustly even with low-resolution inputs. This shows the potential to reduce the cost by using LIDARs with lower resolution in concert with stereo systems while maintaining high resolution.

- Monocular Visual-Inertial Odometry in Low-Textured Environments with Smooth Gradients: A Fully Dense Direct Filtering Approach

    Author: Hardt-Stremayr, Alexander | Alpen-Adria-Universitét Klagenfurt
    Author: Weiss, Stephan | Universitét Klagenfurt
 
    keyword: Sensor Fusion; Visual-Based Navigation; Computer Vision for Other Robotic Applications

    Abstract : State of the art visual-inertial odometry approaches suffer from the requirement of high gradients and sufficient visual texture. Even direct photometric approaches select a subset of the image with high-gradient areas and ignore smooth gradients or generally low-textured areas. In this work, we show that taking all image information (i.e. every single pixel) enables visual-inertial odometry even on areas with very low texture and smooth gradients, inherently interpolating and estimating the scene with no texture based on its informative surrounding. This information propagation is only possible as we estimate all states and their uncertainties (robot pose, extrinsic sensor calibration, and scene depth) jointly in a fully dense filter framework. Our complexity reduction approach enables real-time execution despite the large size of the state vector. Compared to our previous basic feasibility study on this topic, this work includes higher order covariance propagation and improved state handling for a significant performance gain, thorough comparisons to state-of-the-art algorithms, larger mapping components with uncertainty, self-calibration capability, and real-data tests.

- Gated Recurrent Fusion to Learn Driving Behaviour from Temporal Multimodal Data

    Author: Lakshmi Narayanan, Athmanarayanan | Honda Research Institute
    Author: Siravuru, Avinash | Carnegie Mellon University
    Author: Dariush, Behzad | Honda Research Institute USA
 
    keyword: Sensor Fusion; Intelligent Transportation Systems; Computer Vision for Transportation

    Abstract : The Tactical Driver Behavior modeling problem requires an understanding of driver actions in complicated urban scenarios from rich multimodal signals including video, LiDAR and CAN signal data streams. However, the majority of deep learning research is focused either on learning the vehicle/environment state (sensor fusion) or the driver policy (from temporal data), but not both. Learning both tasks jointly offers the richest distillation of knowledge but presents challenges in the formulation and successful training. In this work, we propose promising first steps in this direction. Inspired by the gating mechanisms in Long Short-Term Memory units (LSTMs), we propose Gated Recurrent Fusion Units (GRFU) that learn fusion weighting and temporal weighting simultaneously. We demonstrate it's superior performance over multimodal and temporal baselines in supervised regression and classification tasks, all in the realm of autonomous navigation. On tactical driver behavior classification using Honda Driving Dataset (HDD), we report 10% improvement in mean Average Precision (mAP) score, and similarly, for steering angle regression on TORCS dataset, we note a 20% drop in Mean Squared Error (MSE) over the state-of-the-art.

- Cooperative Visual-Inertial Odometry: Analysis of Singularities, Degeneracies and Minimal Cases

    Author: Martinelli, Agostino | INRIA Grenoble-Rhone-Alpes
 
    keyword: Sensor Fusion; Visual-Based Navigation; Distributed Robot Systems

    Abstract : This paper provides an exhaustive analysis of all the singularities and minimal cases in cooperative visual-inertial odometry. Specifically, the case of two agents is analysed. As in the case of a single agent (addressed in [1]) and in the case of other computer vision problems, the key of the analysis is the establishment of an equivalence between the cooperative visual-inertial odometry problem and a Polynomial Equation System (PES). In the case of a single agent, the PES consists of linear equations and a single polynomial of second degree. In the case of two agents, the number of second degree equations becomes three and, also in this case, a complete analytic solution can be obtained [2]. The power of the analytic solution is twofold. From one side, it allows us to determine the state without the need of an initialization. From another side, it provides fundamental insights into all the structural properties of the problem. This paper focuses on this latter issue. Specifically, we obtain all the minimal cases and singularities depending on the number of camera images and the relative trajectory between the agents. The problem, when non singular, can have up to eight distinct solutions. The usefulness of this analysis is illustrated with simulations. In particular, we show quantitatively how the performance of the state estimation worsens near a singularity.

- A Lightweight and Accurate Localization Algorithm Using Multiple Inertial Measurement Units

    Author: Zhang, Ming | Alibaba Incorporation
    Author: Xu, Xiangyu | Alibaba Inc
    Author: Chen, Yiming | Alibaba Group
    Author: Li, Mingyang | Alibaba
 
    keyword: Sensor Fusion; Localization; SLAM

    Abstract : This paper proposes a novel inertial-aided localization approach by fusing information from multiple inertial measurement units (IMUs) and exteroceptive sensors. IMU is a low-cost motion sensor which provides measurements on angular velocity and gravity compensated linear acceleration of a moving platform, and widely used in modern localization systems. To date, most existing inertial-aided localization methods exploit only one single IMU. While the single-IMU localization yields acceptable accuracy and robustness for different use cases, the overall performance can be further improved by using multiple IMUs. To this end, we propose a lightweight and accurate algorithm for fusing measurements from multiple IMUs and exteroceptive sensors, which is able to obtain noticeable performance gain without incurring additional computational cost. To achieve this, we first probabilistically map measurements from all IMUs onto a virtual IMU. This step is performed by stochastic estimation with least-square estimators and probabilistic marginalization of inter-IMU rotational accelerations. Subsequently, the propagation model for both state and error state of the virtual IMU is also derived, which enables the use of the classical filter-based or optimization-based sensor fusion algorithms for localization. Finally, results from both simulation and real-world tests are provided, which demonstrate that the proposed algorithm outperforms competing algorithms by noticeable margins.

- Accelerating the Estimation of Metabolic Cost Using Signal Derivatives: Implications for Optimization and Evaluation of Wearable Robots (I)

    Author: Ingraham, Kimberly | University of Michigan
    Author: Rouse, Elliott | University of Michigan / (Google) X
    Author: Remy, C. David | University of Stuttgart
 
    keyword: Sensor Fusion; Rehabilitation Robotics; Optimization and Optimal Control

    Abstract : Body-in-the-loop optimization algorithms use real-time estimates of metabolic cost to iteratively tune the actuation profile of a robotic assistive device (e.g., an exoskeleton) to minimize the wearer's energetic cost. To translate these algorithms outside the laboratory environment, we need to obtain estimates of metabolic energy cost quickly, accurately, and with unobtrusive equipment. We have previously shown that we can estimate metabolic cost using physiological signals collected from portable, wearable sensors (e.g., accelerometers or EMG). However, these estimates are still dynamically delayed. Inspired by model-based estimation techniques, in this article we show that including signal derivatives in a black-box estimation process improves both the magnitude and speed of predicting instantaneous metabolic cost from wearable sensors when the input signals are dynamically delayed. These performance improvements were observed during the transient phase of an activity, while steady state performance remained unchanged. This article provides a practical foundation for improving the speed of predicting metabolic energy cost for wearable robotics applications.

- Deep Depth Fusion for Black, Transparent, Re&#64258;ective and Texture-Less Objects

    Author: Chai, Chun-Yu | National Chiao Tung University
    Author: Wu, Yu-Po | National Chiao Tung University
    Author: Tsao, Shiao-Li | National Chiao Tung University
 
    keyword: Sensor Fusion; RGB-D Perception; Perception for Grasping and Manipulation

    Abstract : Structured-light and stereo cameras, which are widely used to construct point clouds for robotic applications, have different limitations on estimating depth values. Structured-light cameras fail in black, transparent, and reflective objects, which influence the light path; stereo cameras fail in texture-less objects.<p>In this work, we propose a depth fusion model that complements these two types of methods to generate high-quality point clouds for short-range robotic applications. The model first determines the fusion weights from the two input depth images and then refines the fused depth using color features.</p><p>We construct a dataset containing the aforementioned challenging objects and report the performance of our proposed model. The results reveal that our method reduces the average L1 distance on depth prediction by 75% and 52% compared with the original depth output of the structured-light camera and the stereo model, respectively. A direct improvement on the Iterative Closest Point (ICP) algorithm can be achieved by using the refined depth images output from our method.

- LiDAR-Enhanced Structure-From-Motion

    Author: Zhen, Weikun | Carnegie Mellon University
    Author: Hu, Yaoyu | Carnegie Mellon University
    Author: Yu, Huai | Wuhan University
    Author: Scherer, Sebastian | Carnegie Mellon University
 
    keyword: Sensor Fusion; Range Sensing; Mapping

    Abstract : Although Structure-from-Motion (SfM) as a maturing technique has been widely used in many applications, state-of-the-art SfM algorithms are still not robust enough in certain situations. For example, images for inspection purposes are often taken in close distance to obtain detailed textures, which will result in less overlap between images and thus decrease the accuracy of estimated motion. In this paper, we propose a LiDAR-enhanced SfM pipeline that jointly processes data from a rotating LiDAR and a stereo camera pair to estimate sensor motions. We show that incorporating LiDAR helps to effectively reject falsely matched images and significantly reduce the motion drift in large scale environments. Experiments are conducted in different environments to test the performance of the proposed pipeline and comparison results with the state-of-the-art SfM algorithms are reported.

- Low Latency and Low-Level Sensor Fusion for Automotive Use-Cases

    Author: Pollach, Matthias | Technical University Munich
    Author: Schiegg, Felix | Technische Universitét M�nchen
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
 
    keyword: Sensor Fusion; Object Detection, Segmentation and Categorization; Sensor Networks

    Abstract : This work proposes a probabilistic low level automotive sensor fusion approach using LiDAR, RADAR and camera data. The method is stateless and directly operates on associated data from all sensor modalities. Tracking is not used, in order to reduce the object detection latency and create existence hypotheses per frame. The probabilistic fusion uses input from 3D and 2D space. An association method using a combination of overlap and distance metrics, avoiding the need for sensor synchronization is proposed. A Bayesian network executes the sensor fusion. The proposed approach is compared with a state of the art fusion system, which is using multiple sensors of the same modality and relies on tracking for object detection. Evaluation was done using low level sensor data recorded in an urban environment. The test results show that the low level sensor fusion reduces the object detection latency.

- Spatiotemporal Camera-LiDAR Calibration: A Targetless and Structureless Approach

    Author: Park, Chanoh | CSIRO, QUT
    Author: Moghadam, Peyman | CSIRO
    Author: Kim, Soohwan | CSIRO
    Author: Sridharan, Sridha | Queensland University of Technology
    Author: Fookes, Clinton | Queensland University of Technology
 
    keyword: Sensor Fusion; SLAM

    Abstract : The demand for multimodal sensing systems for robotics is growing due to the increase in robustness, reliability and accuracy offered by these systems. These systems also need to be spatially and temporally co-registered to be effective. In this paper, we propose a targetless and structureless spatiotemporal camera-LiDAR calibration method. Our method combines a closed-form solution with a modified structureless bundle adjustment where the coarse-to-fine approach does not require an initial guess on the spatiotemporal parameters. Also, as 3D features (structure) are calculated from triangulation only, there is no need to have a calibration target or to match 2D features with the 3D point cloud which provides flexibility in the calibration process and sensor configuration. We demonstrate the accuracy and robustness of the proposed method through both simulation and real data experiments using multiple sensor payload configurations mounted to hand-held, aerial and legged robot systems. Also, qualitative results are given in the form of a colorized point cloud visualization.

- Robot-Assisted and Wearable Sensor-Mediated Autonomous Gait Analysis

    Author: Zhang, Huanghe | Stevens Institute of Technology
    Author: Chen, Zhuo | Stevens Institute of Technology
    Author: Zanotto, Damiano | Stevens Institute of Technology
    Author: Guo, Yi | Stevens Institute of Technology
 
    keyword: Sensor Fusion; Physically Assistive Devices; Robot Companions

    Abstract : In this paper, we propose an autonomous gait analysis system consisting of a mobile robot and custom-engineered instrumented insoles. The robot is equipped with an on-board RGB-D sensor, the insoles feature inertial sensors and force sensitive resistors. This system is motivated by the need of a robot companion to engage older adults in walking exercises. Support vector regression (SVR) models were developed to extract accurate estimates of fundamental kinematic gait parameters (i.e., stride length, velocity, foot clearance, and step length), from data collected with the robot's on-board RGB-D sensor and with the instrumented insoles during straight walking and turning tasks. The accuracy of each model was validated against ground-truth data measured by an optical motion capture system with N=10 subjects. Results suggest that the combined use of wearable and robot's sensors yields more accurate gait estimates than either sub-system used independently. Additionally, SVR models are robust to inter-subject variability and type of walking task (i.e., straight walking vs. turning), thereby making it unnecessary to collect subject-specific or task-specific training data for the models. These findings indicate the potential of the synergistic use of autonomous mobile robots and wearable sensors for accurate out-of-the-lab gait analysis.

- Gaussian Process Preintegration for Inertial-Aided State Estimation

    Author: Le Gentil, Cedric | University of Technology Sydney
    Author: Vidal-Calleja, Teresa A. | University of Technology Sydney
    Author: Huang, Shoudong | University of Technology, Sydney
 
    keyword: Sensor Fusion; SLAM; Localization

    Abstract : In this paper, we present Gaussian Process Preintegration, a preintegration theory based on continuous representations of inertial measurements. A novel use of linear operators on Gaussian Process kernels is employed to generate the proposed Gaussian Preintegrated Measurements (GPMs). This formulation allows the analytical integration of inertial signals on any time interval. Consequently, GPMs are especially suited for asynchronous inertial-aided estimation frameworks. Unlike discrete preintegration approaches, the proposed method does not rely on any explicit motion-model and does not suffer from numerical integration noise. Additionally, we provide the analytical derivation of the Jacobians involved in the first-order expansion for postintegration bias and inter-sensor time-shift correction. We benchmarked the proposed method against existing preintegration methods on simulated data. Our experiments show that GPMs produce the most accurate results and their computation time allows close-to-real-time operations. We validated the suitability of GPMs for inertial-aided estimation by integrating them into a lidar-inertial localisation and mapping framework.

- A Code for Unscented Kalman Filtering on Manifolds (UKF-M)

    Author: Brossard, Martin | Mines ParisTech
    Author: Barrau, Axel | Safran
    Author: Bonnabel, Silvere | Mines ParisTech
 
    keyword: Sensor Fusion; Localization; Wheeled Robots

    Abstract : The present paper introduces a novel methodology for Unscented Kalman Filtering (UKF) on manifolds that extends previous work by the     Authors on UKF on Lie groups. Beyond filtering performance, the main interests of the approach are its versatility, as the method applies to numerous state estimation problems, and its simplicity of implementation for practitioners not being necessarily familiar with manifolds and Lie groups. We have developed the method on two independent open-source Python and Matlab frameworks we call UKF-M, for quickly implementing and testing the approach. The online repositories contain tutorials, documentation, and various relevant robotics examples that the user can readily reproduce and then adapt, for fast prototyping and benchmarking. The code is available at https://github.com/CAOR-MINES-ParisTech/ukfm.

- Efficient and Precise Sensor Fusion for Non-Linear Systems with Out-Of-Sequence Measurements by Example of Mobile Robotics

    Author: B�hmler, Pascal | Karlsruhe Institute of Technology
    Author: Dziedzitz, Paul Jonathan | Karlsruhe Institute of Technology
    Author: Hopfgarten, Patric | Karlsruher Institut F�r Technologie
    Author: Specker, Thomas | Robert Bosch GmbH
    Author: Lange, Ralph | Robert Bosch GmbH
 
    keyword: Sensor Fusion; Localization

    Abstract : For most applications in mobile robotics, precise state estimation is essential. Typically, the state estimation is based on the fusion of data from different sensors. In practice, these sensors differ in their characteristics and measurements are available to the sensor fusion algorithm only with delay. Based on a brief survey of sensor fusion approaches that consider delayed and out-of-sequence availability of measurements, suitable approaches for applications in mobile robotics are identified. In a consumer robot use-case, experiments show that the estimation is biased if delayed availability of measurements is not considered appropriately. However, if delays are considered in the fusion process, the estimation bias is reduced to almost zero and in consequence, the estimation performance is distinctly improved. Two computational favorable approximative methods are described and provide almost the same accuracy as - theoretically optimal - brute-force filter recalculation at much lower and well-distributed computational costs.

- UNO: Uncertainty-Aware Noisy-Or Multimodal Fusion for Unanticipated Input Degradation

    Author: Tian, Junjiao | Georgia Institute of Technology
    Author: Cheung, Wesley | Georgia Institute of Technology
    Author: Glaser, Nathaniel | Georgia Institute of Technology
    Author: Liu, Yen-Cheng | Georgia Tech
    Author: Kira, Zsolt | Georgia Institute of Technology
 
    keyword: Sensor Fusion; RGB-D Perception; Semantic Scene Understanding

    Abstract : The fusion of multiple sensor modalities, especially through deep learning architectures, has been an active area of study. However, an under-explored aspect of such work is whether the methods can be robust to degradation across their input modalities, especially when they must generalize to degradation not seen during training. In this work, we propose an uncertainty-aware fusion scheme to effectively fuse inputs that might suffer from a range of known and unknown degradation. Specifically, we analyze a number of uncertainty measures, each of which captures a different aspect of uncertainty, and we propose a novel way to fuse degraded inputs by scaling modality-specific output softmax probabilities. We additionally propose a novel data-dependent spatial temperature scaling method to complement these existing uncertainty measures. Finally, we integrate the uncertainty-scaled output from each modality using a probabilistic noisy-or fusion method. In a photo-realistic simulation environment (AirSim), we show that our method achieves significantly better results on a semantic segmentation task, as compared to state-of-art fusion architectures, on a range of degradation (e.g. fog, snow, frost, and various other types of noise), some of which are unknown during training.

- Intermittent GPS-Aided VIO: Online Initialization and Calibration

    Author: Lee, Woosik | University of Delaware
    Author: Eckenhoff, Kevin | University of Delaware
    Author: Geneva, Patrick | University of Delaware
    Author: Huang, Guoquan | University of Delaware
 
    keyword: Sensor Fusion; Localization; SLAM

    Abstract : In this paper, we present an efficient and robust GPS-aided visual inertial odometry (GPS-VIO) system that fuses IMU-camera data with intermittent GPS measurements. To perform sensor fusion, spatiotemporal sensor calibration and initialization of the transform between the sensor reference frames are required. We propose an online calibration method for both the GPS-IMU extrinsics and time offset as well as a reference frame initialization procedure that is robust to GPS sensor noise. In addition, we prove the existence of four unobservable directions of the GPS-VIO system when estimating in the VIO reference frame, and advocate a state transformation to the GPS reference frame for full observability. We extensively evaluate the proposed approach in Monte-Carlo simulations where we investigate the system's robustness to different levels of GPS noise and loss of GPS signal, and additionally study the hyper-parameters used in the initialization procedure. Finally, the proposed system is validated in a large-scale real-world experiment.

- A Mathematical Framework for IMU Error Propagation with Applications to Preintegration

    Author: Barrau, Axel | Safran
    Author: Bonnabel, Silvere | Mines ParisTech
 
    keyword: Sensor Fusion; Localization; SLAM

    Abstract : To fuse information from inertial measurement units (IMU) with other sensors one needs an accurate model for IMU error propagations in terms of position, velocity and orientation, a triplet we call extended pose.	In this paper we leverage a nontrivial result, namely log-linearity of inertial navigation equations based on the recently introduced Lie group SE_2(3), to	transpose the recent methodology of Barfoot and Furgale for associating uncertainty with poses (position, orientation) of SE(3) when using noisy wheel speeds, to the case of extended poses (position, velocity, orientation) of SE_2(3) when using noisy IMUs. Besides, our approach to extended poses combined with log-linearity property allows revisiting the theory of preintegration on manifolds and reaching a further theoretic level in this field. We show exact preintegration formulas that account for rotating earth, that is, centrifugal force and Coriolis effect,	may be derived as a byproduct.

- Radar-Inertial Ego-Velocity Estimation for Visually DegradedEnvironments

    Author: Kramer, Andrew | University of Colorado Boulder
    Author: Stahoviak, Carl | University of Colorado Boulder
    Author: Santamaria-Navarro, Angel | NASA Jet Propulsion Laboratory, Caltech
    Author: Heckman, Christoffer | University of Colorado at Boulder
    Author: Agha-mohammadi, Ali-akbar | NASA-JPL, Caltech
 
    keyword: Sensor Fusion; Aerial Systems: Perception and Autonomy; Field Robots

    Abstract : We present an approach for estimating the body-frame velocity of a mobile robot. We combine measurements from a millimeter-wave radar-on-a-chip sensor and an inertial measurement unit (IMU) in a batch optimization over a sliding window of recent measurements. The sensor suite employed is lightweight, low-power, and is invariant to ambient lighting conditions. This makes the proposed approach an attractive solution for platforms with limitations around payload and longevity, such as aerial vehicles conducting autonomous exploration in perceptually degraded operating conditions, including subterranean environments. We compare our radar-inertial velocity estimates to those from a visual-inertial (VI) approach. We show the accuracy of our method is comparable to VI in conditions favorable to VI, and far exceeds the accuracy of VI when conditions deteriorate.

- Observability Analysis of Flight State Estimation for UAVs and Experimental Validation

    Author: Huang, Peng | Technische Universitét Dresden
    Author: Meyr, Heinrich | Barkhausen Institut
    Author: D�rpinghaus, Meik | Technische Universitét Dresden
    Author: Fettweis, Gerhard | Technische Universitét Dresden
 
    keyword: Sensor Fusion; Autonomous Vehicle Navigation; Aerial Systems: Mechanics and Control

    Abstract : UAVs require reliable, cost-efficient onboard flight state estimation that achieves high accuracy and robustness to perturbation. We analyze a multi-sensor extended Kalman filter (EKF) based on the work by Leutenegger. The EKF uses measurements from a MEMS-based inertial system, static and dynamic pressure sensors as well as GPS. As opposed to other implementations we do not use a magnetic sensor because the weak magnetic field of the earth is subject to disturbances. Observability of the state is a necessary condition for the EKF to work. In this paper, we demonstrate that the system state is observable - which is in contrast to statements in the literature - if the random nature of the air mass is taken into account. Therefore, we carry out an in-depth observability analysis based on a singular value decomposition (SVD). The numerical SVD delivers a wealth of information regarding the observable (sub)spaces. We validated the theoretical findings based on sensor data recorded in test flights on a glider. Most importantly, we demonstrate that the EKF works. It is capable of absorbing large perturbations in the wind state variable converging to the undisturbed estimates.

- OpenVINS: A Research Platform for Visual-Inertial Estimation

    Author: Geneva, Patrick | University of Delaware
    Author: Eckenhoff, Kevin | University of Delaware
    Author: Lee, Woosik | University of Delaware
    Author: Yang, Yulin | University of Delaware
    Author: Huang, Guoquan | University of Delaware
 
    keyword: Sensor Fusion; Localization; SLAM

    Abstract : In this paper, we present an open platform, termed OpenVINS, for visual-inertial estimation research for both the academic community and practitioners from industry. The open sourced codebase provides a foundation for researchers and engineers to quickly start developing new capabilities for their visual-inertial systems. This codebase has out of the box support for commonly desired visual-inertial estimation features, which include: (i) on-manifold sliding window Kalman filter, (ii) online camera intrinsic and extrinsic calibration, (iii) camera to inertial sensor time offset calibration, (iv) SLAM landmarks with different representations and consistent First-Estimates Jacobian (FEJ) treatments, (v) modular type system for state management, (vi) extendable visual-inertial system simulator, and (vii) extensive toolbox for algorithm evaluation. Moreover, we have also focused on detailed documentation and theoretical derivations to support rapid development and research, which are greatly lacked in the current open sourced algorithms. Finally, we perform comprehensive validation of the proposed OpenVINS against state-of-the-art open sourced algorithms, showing its competing estimation performance.

- Decentralized Collaborative State Estimation for Aided Inertial Navigation

    Author: Jung, Roland | Alpen-Adria-Universitét Klagenfurt
    Author: Brommer, Christian | Alpen Adria University
    Author: Weiss, Stephan | Universitét Klagenfurt
 
    keyword: Sensor Fusion; Cooperating Robots; Localization

    Abstract : In this paper, we present a Quaternion-based Error-State Extended Kalman Filter (Q-ESEKF) based on IMU propagation with an extension for Collaborative State Estimation (CSE) and a communication complexity of O(1) (in terms of required communication links). Our approach combines a versatile filter formulation with the concept of CSE, allowing independent state estimation on each of the agents and at the same time leveraging and statistically maintaining interdependencies between agents, after joint measurements and communication (i.e. relative position measurements) occur. We discuss the development of the overall framework and the probabilistic (re-)initialization of the agent's states upon initial or recurring joint observations. Our approach is evaluated in a simulation framework on two prominent benchmark datasets in 3D.

- Analytic Combined IMU Integration (ACI^2) for Visual Inertial Navigation

    Author: Yang, Yulin | University of Delaware
    Author: Wisely Babu, Benzun Pious | Robert Bosch Research &amp; Technology Center, North America
    Author: Chen, Chuchu | University of Delaware
    Author: Huang, Guoquan | University of Delaware
    Author: Ren, Liu | Robert Bosch North America Research Technology Center
 
    keyword: Sensor Fusion; SLAM; Localization

    Abstract : Batch optimization based inertial measurement unit (IMU) and vision sensor fusion enables high rate localization for many robotics tasks. However, it remains a challenge to ensure that the batch optimization is computationally efficient while being consistent for high rate IMU measurements without marginalization. In this paper, we derive inspiration from maximum likelihood estimation with partial-fixed estimates to provide a unified approach for handing both IMU pre-integration and time-offset calibration. We present a novel modularized analytic combined IMU integrator (ACI^2) with derivations for integration, Jabcobians and covariances estimation. To simplify our derivation we also prove that the right Jacobians for Hamilton quaterions and SO(3) are equivalent. Finally, we also present a time offset calibrator that operates by fixing the linearization point for a given time offset. This reduces re-integration of the IMU measurements and thus improve efficiency. The proposed ACI^2 and time offset calibration is verified by intensive Monte-Carlo simulations generated from real world datasets. A proof-of-concept real world experiment is also conducted to verify the proposed ACI^2 estimator.

- Second-Order Kinematics for Floating-Base Robots Using the Redundant Acceleration Feedback of an Artificial Sensory Skin

    Author: Leboutet, Quentin | Technical University of Munich
    Author: Guadarrama-Olvera, Julio Rogelio | Technical University of Munich
    Author: Bergner, Florian | Technical University of Munich
    Author: Cheng, Gordon | Technical University of Munich
 
    keyword: Sensor Fusion; Sensor Networks; Kinematics

    Abstract : In this work, we propose a new estimation method for second-order kinematics for floating-base robots, based on highly redundant distributed inertial feedback. The linear acceleration of each robot link is measured at multiple points using a multimodal, self-configuring and self-calibrating artificial skin. The proposed algorithm is two-fold: i) the skin acceleration data is fused at the link level for state dimensionality reduction; ii) the estimated values are then fused limb-wise with data from the joint encoders and the main inertial measurement unit (IMU), using a Sigma-point Kalman filter. In this manner, it is possible to estimate the joint velocities and accelerations while avoiding the lag and noise amplification phenomena associated with conventional numerical derivation approaches. Experiments performed on the right arm and torso of a REEM-C humanoid robot, demonstrate the consistency of the proposed estimation method.

- Clock-Based Time Synchronization for an Event-Based Camera Dataset Acquisition Platform

    Author: Osadcuks, Vitalijs | Latvia University of Life Sciences and Technologies, Faculty Of
    Author: Pudzs, Mihails | Riga Technical University
    Author: Zujevs, Andrejs | Riga Technical University
    Author: Pecka, Aldis | Latvia University of Life Sciences and Technologies
    Author: Ardavs, Arturs | Riga Technical University
 
    keyword: Sensor Fusion; Neurorobotics; Robotics in Agriculture and Forestry

    Abstract : The Dynamic Visual Sensor is considered to be a next-generation vision sensor. Since event-based vision is in its early stage of development, a small number of datasets has been created during the last decade. Dataset creation is motivated by the need for real data from one or many sensors. Temporal accuracy of data in such datasets is crucially important since the events have high temporal resolution measured in microseconds and, during an algorithm evaluation task, such type of visual data is usually fused with data from other types of sensors. The main aim of our research is to achieve the most accurate possible time synchronization between an event camera, LIDAR, and ambient environment sensors during a session of data acquisition. All the mentioned sensors as well as a stereo and a monocular camera were installed on a mobile robotic platform. In this work, a time synchronization architecture and algorithm are proposed for time synchronization with an implementation example on a PIC32 microcontroller. The overall time synchronization approach is scalable for other sensors where there is a need for accurate time synchronization between many nodes. The evaluation results of the proposed solution are reported and discussed in the paper.

## Compliance and Impedance Control
- Hierarchical Impedance-Based Tracking Control of Kinematically Redundant Robots (I)

    Author: Dietrich, Alexander | German Aerospace Center (DLR)
    Author: Ott, Christian | German Aerospace Center (DLR)
 
    keyword: Compliance and Impedance Control; Redundant Robots; Force Control

    Abstract : The control of a robot in its task space is a standard approach nowadays. If the system is kinematically redundant with respect to this goal, one can even execute additional subtasks simultaneously. By utilizing null space projections, for example, the whole stack of tasks can be implemented within a strict task hierarchy following the order of priority. One of the most common methods to track multiple task-space trajectories at the same time is to feedback-linearize the system and dynamically decouple all involved subtasks, which finally yields exponential stability of the desired equilibrium. Here, we provide a hierarchical multi-objective controller for trajectory tracking that ensures both asymptotic stability of the equilibrium and a desired contact impedance at the same time. In contrast to the state of the art in prioritized multi-objective control, feedback of the external forces can be avoided and the natural inertia of the robot is preserved. The controller is evaluated in simulations and on a standard lightweight robot with torque interface. The approach is predestined for precise trajectory tracking where dedicated and robust physical-interaction compliance is crucial at the same time.

- Position-Based Impedance Control of a 2-DOF Compliant Manipulator for a Facade Cleaning Operation

    Author: Kim, Taegyun | Yeungnam University
    Author: Yoo, Sungkeun | Seoul National University
    Author: Kim, Hwa Soo | Kyonggi University
    Author: Seo, TaeWon | Hanyang University
 
    keyword: Compliance and Impedance Control; Force Control; Service Robots

    Abstract : This paper presents the design of a compliant manipulator using a series elastic actuator (SEA) and a mechanism for precisely measuring the force acting on the contact part of the manipulator without using a force sensor. It is important to maintain a constant contact force between the compliant manipulator and the wall in order to guarantee cleaning performance, and the ball screw mechanism is used to adapt to changes in the distance and the angle. Position-based impedance control is used to maintain a constant contact force when the manipulator interacts with the wall of the building, and the results confirm that the system stability is guaranteed when using SEA, regardless of the variation in the actual stiffness of the manipulator. The results of extensive experimentation using the test bench demonstrate the force tracking performance against various types of wall changes using the stiff wet-type cleaning manipulator. The results indicate that the stiffness of SEA affects the force tracking performance and system stability under the condition of the manipulator and environment interaction, and that the system stability and control performance can be improved by applying a robust force measurement mechanism to noise.

- Robust, Locally Guided Peg-In-Hole with Impedance-Controlled Robots

    Author: Nottensteiner, Korbinian | German Aerospace Center (DLR)
    Author: Stulp, Freek | DLR - Deutsches Zentrum F�r Luft Und Raumfahrt E.V
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Compliant Assembly; Force and Tactile Sensing; Reactive and Sensor-Based Planning

    Abstract : We present an approach for the autonomous, robust execution of peg-in-hole assembly tasks. We build on a sampling-based state estimation framework, in which samples are weighted according to their consistency with the position and joint torque measurements. The key idea is to reuse these samples in a motion generation step, where they are assigned a second task-specific weight. The algorithm thereby guides the peg towards the goal along the configuration space. An advantage of the approach is that the user only needs to provide: the geometry of the objects as mesh data, as well as a rough estimate of the object poses in the workspace, and a desired goal state. Another advantage is that the local, online nature of our algorithm leads to robust behavior under uncertainty. The approach is validated in the case of our robotic setup and under varying uncertainties for the classical peg-in-hole problem subject to two different geometries.

- Model-Free Friction Observers for Flexible Joint Robots with Torque Measurements (I)

    Author: Kim, Min Jun | DLR
    Author: Beck, Fabian | German Aerospace Center (DLR)
    Author: Ott, Christian | German Aerospace Center (DLR)
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Compliance and Impedance Control; Robust/Adaptive Control of Robotic Systems; Physical Human-Robot Interaction

    Abstract : This paper tackles a friction compensation problem without using a friction model. The unique feature of the proposed friction observer is that the nominal motor-side signal is fed back into the controller instead of the measured signal. By doing so, asymptotic stability and passivity of the controller are maintained. Another advantage of the proposed observer is that it provides a clear understanding for the stiction compensation which is hard to be captured in model-free approaches. This allows to design observers that do not overcompensate for the stiction. The proposed scheme is validated through simulations and experiments.

- Necessary and Sufficient Conditions for the Passivity of Impedance Rendering with Velocity-Sourced Series Elastic Actuation (I)

    Author: Tosun, Fatih Emre | Sabanci University
    Author: Patoglu, Volkan | Sabanci University
 
    keyword: Compliance and Impedance Control; Physical Human-Robot Interaction; Haptics and Haptic Interfaces

    Abstract : Series Elastic Actuation (SEA) has become prevalent in applications involving physical human-robot interaction as it provides considerable advantages over traditional stiff actuators in terms of stability robustness and force control fidelity. Several impedance control architectures have been proposed for SEA.<p>Among these alternatives, the cascaded controller with an inner- most velocity loop, an intermediate torque loop and an outer-most</p><p>impedance loop is particularly favoured for its simplicity, robust- ness, and performance. In this paper, we derive the necessary</p><p>and sufficient conditions for passively rendering null impedance and virtual springs with this cascade-controller architecture.</p><p>Based on the newly established conditions, we provide non- conservative passivity design guidelines to haptically display these</p><p>two impedance models, which serve as the basic building blocks of various virtual environments, while ensuring the safety of interaction. We also demonstrate the importance of including physical damping in the actuator model for deriving the passivity conditions, when integrators are utilized. In particular, we prove the unintuitive adversary effect of physical damping on the passivity of the system by noting that the damping term reduces the system Z-width as well as introducing an extra passivity constraint. Finally, we experimentally validate our theoretical results using a SEA brake pedal.

- Design of Spatial Admittance for Force-Guided Assembly of Polyhedral Parts in Single Point Frictional Contact

    Author: Huang, Shuguang | Marquette University
    Author: Schimmels, Joseph | Marquette University
 
    keyword: Compliant Assembly; Compliance and Impedance Control

    Abstract : This paper identifies conditions for designing the appropriate spatial admittance to achieve reliable force-guided assembly of polyhedral parts for cases in which a single frictional contact occurs between the two parts.	This work is an extension of previous work in which frictionless contact was considered. This paper presents a way to characterize friction without solving a set of complicated non-linear equations. We show that, by modifying the error reduction function and evaluating the function bounds associated with friction, the procedures developed for frictionless contact apply to the frictional cases. Thus, for bounded misalignments, if an admittance satisfies the misalignment-reducing conditions at a finite number of contact configurations, then the admittance will also satisfy the conditions at all intermediate configurations for any value of friction less than the specified upper bound.

- Model Predictive Impedance Control

    Author: Bednarczyk, Maciej | ICube Laboratory, University of Strasbourg, Strasbourg
    Author: Omran, Hassan | ICube Laboratory, University of Strasbourg, Strasbourg
    Author: Bayle, Bernard | University of Strasbourg
 
    keyword: Compliance and Impedance Control; Optimization and Optimal Control

    Abstract : Robots are more and more often designed in order to perform tasks in synergy with human operators. In this context, a current research focus for collaborative robotics lies in the design of high-performance control solutions, which ensure security in spite of unmodeled external forces. The present work provides a method based on Model Predictive Control (MPC) to allow compliant behavior when interacting with an environment, while respecting practical robotic constraints. The study shows in particular how to define the impedance control problem as a MPC problem. The approach is validated with an experimental setup including a collaborative robot. The obtained results emphasize the ability of this control strategy to solve constraints like speed, energy or jerk limits, which have a direct impact on the operator's security during human-robot compliant interactions.

- Kinematic Modeling and Compliance Modulation of Redundant Manipulators under Bracing Constraints

    Author: Johnston, Garrison | Vanderbilt University
    Author: Orekhov, Andrew | Vanderbilt University
    Author: Simaan, Nabil | Vanderbilt University
 
    keyword: Compliance and Impedance Control; Redundant Robots; Kinematics

    Abstract : Collaborative robots should ideally use low torque actuators for passive safety reasons. However, some applications require these collaborative robots to reach deep into confined spaces while assisting a human operator in physically demanding tasks. In this paper, we consider the use of in-situ collaborative robots (ISCRs) that balance the conflicting demands of passive safety dictating low torque actuation and the need to reach into deep confined spaces. We consider the judicious use of bracing as a possible solution to these conflicting demands and present a modeling framework that takes into account the constrained kinematics and the effect of bracing on the end-effector compliance. We then define a redundancy resolution framework that minimizes the directional compliance of the end-effector while maximizing end-effector dexterity. Kinematic simulation results show that the redundancy resolution strategy successfully decreases compliance and improves kinematic conditioning while satisfying the constraints imposed by the bracing task. Applications of this modeling framework can support future research on the choice of bracing locations and support the formation of an admittance control framework for collaborative control of ISCRs under bracing constraints. Such robots can benefit workers in the future by reducing the physiological burdens that contribute to musculoskeletal injury.

- Successive Stiffness Increment and Time Domain Passivity Approach for Stable High Bandwidth Control of Series Elastic Actuator

    Author: Lee, ChanIl | Korea Advanced Institute of Science and Technology
    Author: Kim, Do-Hyeong | KAIST
    Author: Singh, Harsimran | DLR German Aerospace Center
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
 
    keyword: Compliance and Impedance Control; Physical Human-Robot Interaction; Flexible Robots

    Abstract : For safe human-robot interaction, various type of flexible manipulators have been developed. Especially series elastic actuator (SEA) based manipulators have been getting huge attention since the elastic element of SEA prevents people from injury when undesirable collision happens. Moreover, it improves system durability by absorbing impact force, which could damage actuators. However, the elastic element inside SEA manipulator causes low system bandwidth which limits the speed performance of conventional impedance control approaches. To alleviate the low bandwidth issue of impedance controlled SEA while guaranteeing system stability, we implement Time Domain Passivity Approach (TDPA) and Successive Stiffness Increment (SSI) approach, which was invented in haptic and teleoperation domain. Impedance controlled SEA is reformulated as a two-port electrical circuit network for implementing TDPA. In addition, a pair of input and output power conjugate variable, dominating the system passivity is identified for implementing SSI approach. Experimental results showed that TDPA and SSI approach can render the stiffness of the impedance controller, which decides the bandwidth, upto 350 kN/m without any stability issue, while normal impedance controller only render upto 120 kN/m. Although both of the approaches significantly increased the bandwidth of the impedance controlled SEA, TDPA slightly outperformed in stability, and SSI outperformed in tracking.

- Arm-Hand Motion-Force Coordination for Physical Interactions with Non-Flat Surfaces Using Dynamical Systems: Toward Compliant Robotic Massage

    Author: Khoramshahi, Mahdi | EPFL
    Author: Henriks, Gustav | EPFL
    Author: Naef, Aileen | EPFL
    Author: Mirrazavi Salehian, Seyed Sina | EPFL
    Author: Kim, Joonyoung | Samsung Research
    Author: Billard, Aude | EPFL
 
    keyword: Compliance and Impedance Control; Reactive and Sensor-Based Planning; Manipulation Planning

    Abstract : Many manipulation tasks require coordinated motions for arm and fingers. Complexity increases when the task requires to control for the force at contact against a non-flat surface; This becomes even more challenging when this contact is done on a human. All these challenges are regrouped when one, for instance, massages a human limb. When massaging, the robotic arm is required to continuously adapt its orientation and distance to the limb while the robot fingers exert desired patterns of forces and motion on the skin surface. To address these challenges, we adopt a Dynamical System (DS) approach that offers a unified motion-force control approach and enables to easily coordinate multiple degrees of freedom. As each human limb may slightly differ, we learn a model of the surface using support vector regression (SVR) which enable us to obtain a distance-to-surface mapping. The gradient of this mapping, along with the DS, generates the desired motions for the interaction with the surface. A DS-based impedance control for the robotic fingers allows to control separately for force along the normal direction of the surface while moving in the tangential plane. We validate our approach using the KUKA IIWA robotic arm and Allegro robotic hand for massaging a mannequin arm covered with a skin-like material. Our results show the effectiveness of our approach; especially the robustness toward uncertainties for shape and the given location of the surface.

- A Control Scheme with a Novel DMP-Robot Coupling Achieving Compliance and Tracking Accuracy under Unknown Task Dynamics and Model Uncertainties

    Author: Vlachos, Konstantinos | Aristotle University of Thessaloniki
    Author: Doulgeri, Zoe | Aristotle University of Thessaloniki
 
    keyword: Compliance and Impedance Control; Motion Control

    Abstract : A control scheme consisting of a novel coupling of a DMP based virtual reference with a low stiffness controlled robot is proposed. The overall system is proved to achieve superior tracking of a DMP encoded trajectory and accurate target reaching with respect to the conventional scheme under the presence of constant and periodic disturbances owing to unknown task dynamics and robot model uncertainties. It further preserves the desired compliance under contact forces that may arise in human interventions and collisions. Results in simulations and experiments validate the theoretical findings.

- A Bio-Signal Enhanced Adaptive Impedance Controller for Lower Limb Exoskeleton

    Author: Xia, Linqing | Chinese Academy of Sciences
    Author: Feng, Yachun | Shenzhen Institutes of Advanced Technology, Chinese Academy of Sc
    Author: Chen, Fan | Shenzhen Institutes of Advanced Technology, Chinese Academy of S
    Author: Wu, Xinyu | CAS
 
    keyword: Compliance and Impedance Control; Wearable Robots; Robust/Adaptive Control of Robotic Systems

    Abstract : The problem of human-exoskeleton interaction with uncertain dynamical parameters remains an open-ended research area. It requires an elaborate control strategy design of the exoskeleton to accommodate complex and unpredictable human body movements. In this paper, we proposed a novel control approach for the lower limb exoskeleton to realize its task of assisting the human operator walking. The main challenge of this study was to determine the human lower extremity dynamics, such as the joint torque. For this purpose, we developed a neural network-based torque estimation method. It can predict the joint torques of humans with surface electromyogram signals (sEMG). Then an RBF neural network enhanced adaptive impedance controller is employed to ensure exoskeleton track desired motion trajectory of a human operator. Algorithm performance is evaluated with two healthy subjects and the rehabilitation lower-limb exoskeleton developed by Shenzhen Institutes of Advanced Technology (SIAT).

## Visual Learning
- Vid2Param: Modelling of Dynamics Parameters from Video

    Author: Asenov, Martin | The University of Edinburgh
    Author: Burke, Michael | University of Edinburgh
    Author: Angelov, Daniel | University of Edinburgh
    Author: Davchev, Todor Bozhinov | University of Edinburgh
    Author: Subr, Kartic | The University of Edinburgh
    Author: Ramamoorthy, Subramanian | The University of Edinburgh
 
    keyword: Visual Learning; Sensor-based Control; Motion and Path Planning

    Abstract : Videos provide a rich source of information, but it is generally hard to extract dynamical parameters of interest. Inferring those parameters from a video stream would be beneficial for physical reasoning. Robots performing tasks in dynamic environments would benefit greatly from understanding the underlying environment motion, in order to make future predictions and to synthesize effective control policies that use this inductive bias. Online physical reasoning is therefore a fundamental requirement for robust autonomous agents. When the dynamics involves multiple modes (due to contacts or interactions between objects) and sensing must proceed directly from a rich sensory stream such as video, then traditional methods for system identification may not be well suited. We propose an approach wherein fast parameter estimation can be achieved directly from video. We integrate a physically based dynamics model with a recurrent variational autoencoder, by introducing an additional loss to enforce desired constraints. The model, which we call Vid2Param, can be trained entirely in simulation, in an end-to-end manner with domain randomization, to perform online system identification, and make probabilistic forward predictions of parameters of interest. This enables the resulting model to encode parameters such as position, velocity, restitution, air drag and other physical properties of the system.

- Safe Robot Navigation Via Multi-Modal Anomaly Detection

    Author: Wellhausen, Lorenz | ETH Zurich
    Author: Ranftl, Rene | Intel
    Author: Hutter, Marco | ETH Zurich
 
    keyword: Visual Learning; Visual-Based Navigation; Deep Learning in Robotics and Automation

    Abstract : Navigation in natural outdoor environments requires a robust and reliable traversability classification method to handle the plethora of situations a robot can encounter. Binary classification algorithms perform well in their native domain but tend to provide overconfident predictions when presented with out-of-distribution samples, which can lead to catastrophic failure when navigating unknown environments. We propose to overcome this issue by using anomaly detection on multi-modal images for traversability classification, which is easily scalable by training in a self-supervised fashion from robot experience. In this work, we evaluate multiple anomaly detection methods with a combination of uni- and multi-modal images in their performance on data from different environmental conditions. Our results show that an approach using a feature extractor and normalizing flow with an input of RGB, depth and surface normals performs best. It achieves over 95% area under the ROC curve and is robust to out-of-distribution samples.

- MAVRIC: Morphology-Agnostic Visual Robotic Control

    Author: Yang, Brian | University of California, Berkeley
    Author: Jayaraman, Dinesh | Facebook AI Research and University of Pennsylvania
    Author: Berseth, Glen | University of British Columbia
    Author: Efros, Alexei A. | Carnegie Mellon University
    Author: Levine, Sergey | UC Berkeley
 
    keyword: Visual Learning; Visual Tracking; Visual Servoing

    Abstract : Existing approaches for visuomotor robotic control typically require characterizing the robot in advance by calibrating the camera or performing system identification. We propose MAVRIC, an approach that works with minimal prior knowledge of the robot's morphology, and requires only a camera view containing the robot and its environment and an unknown control interface. MAVRIC revolves around a mutual information-based method for self-recognition, which discovers visual "control points" on the robot body within a few seconds of exploratory interaction, and these control points in turn are then used for visual servoing. MAVRIC can control robots with imprecise actuation, no proprioceptive feedback, unknown morphologies including novel tools, unknown camera poses, and even unsteady handheld cameras. We demonstrate our method on visually-guided 3D point reaching, trajectory following, and robot-to-robot imitation. Project website: https://bit.ly/386j2fi.

- MFuseNet: Robust Depth Estimation with Learned Multiscopic Fusion

    Author: Yuan, Weihao | Hong Kong University of Science and Technology
    Author: Fan, Rui | The Hong Kong University of Science and Technology
    Author: Wang, Michael Yu | Hong Kong University of Science &amp; Technology
    Author: Chen, Qifeng | HKUST
 
    keyword: Visual Learning; Deep Learning in Robotics and Automation; Computer Vision for Automation

    Abstract : We design a multiscopic vision system that utilizes a low-cost monocular RGB camera to acquire accurate depth estimation. Unlike multi-view stereo with images captured at unconstrained camera poses, the proposed system controls the motion of a camera to capture a sequence of images in horizontally or vertically aligned positions with the same parallax. In this system, we propose a new heuristic method and a robust learning-based method to fuse multiple cost volumes between the reference image and its surrounding images. To obtain training data, we build a synthetic dataset with multiscopic images. The experiments on the real-world Middlebury dataset and real robot demonstration show that our multiscopic vision system outperforms traditional two-frame stereo matching methods in depth estimation. Our code and dataset are available at https://sites.google.com/view/multiscopic.

- Deceiving Image-To-Image Translation Networks for Autonomous Driving with Adversarial Perturbations

    Author: Wang, Lin | KAIST
    Author: Cho, Wonjune | KAIST
    Author: Yoon, Kuk-Jin | KAIST
 
    keyword: Visual Learning; Computer Vision for Automation; Deep Learning in Robotics and Automation

    Abstract : Deep neural networks (DNNs) have achieved impressive performance on handling computer vision problems, however, it has been found that DNNs are vulnerable to adversarial examples. For such reason, adversarial perturbations have been recently studied in several respects. However, most previous works have focused on image classification tasks, and it has never been studied regarding adversarial perturbations on Image-to-image (Im2Im) translation tasks, showing great success in handling paired and/or unpaired mapping problems in the field of autonomous driving and robotics. This paper examines different types of adversarial perturbations that can fool Im2Im frameworks for autonomous driving purposes. We propose both quasi-physical and digital adversarial perturbations that can make Im2Im models yield unexpected results. We then empirically analyze these perturbations and show that they generalize well under both paired for image synthesis and unpaired settings for style transfer. We also validate that there exist some perturbation thresholds over which the Im2Im mapping is disrupted or impossible. The existence of these perturbations reveals that there exist crucial weaknesses in Im2Im models. Lastly, we show that our methods illustrate how perturbations affect the quality of outputs, pioneering the improvement of the robustness of current SOTA networks for autonomous driving.

- Self-Supervised Learning of State Estimation for Manipulating Deformable Linear Objects

    Author: Yan, Mengyuan | Stanford University
    Author: Zhu, Yilin | Stanford University
    Author: Jin, Ning | Stanford University
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Visual Learning; Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation

    Abstract : We demonstrate model-based, visual robot manipulation of linear deformable objects. Our approach is based on a state-space representation of the physical system. This choice has multiple advantages, including the ease of incorporating physics priors in the dynamics model and perception model, and the ease of planning manipulation actions. In addition, physical states can naturally represent object instances of different appearances. Therefore, dynamics in the state space generalizes across visually different settings, in contrast to dynamics learned in pixel space or latent space, where such generalization is not guaranteed. Challenges in taking the state-space approach are the estimation of the high-dimensional state of a deformable object from raw images, where annotations are very expensive on real data, and finding a dynamics model that is both accurate, generalizable, and efficient to compute. We are the first to demonstrate self-supervised training of rope state estimation on real images. We propose a novel self-supervising learning objective, which is generalizable across a wide range of visual appearances. With estimated rope states, we train a fast and differentiable neural network dynamics model that encodes the physics of mass-spring systems. Our method has a higher accuracy in predicting future states compared to models in pixel space, while only using 3% of training data. Our approach also achieves more efficient manipulation, both in simulation and on real robot

- Self-Supervised Correspondence in Visuomotor Policy Learning

    Author: Florence, Peter | MIT
    Author: Manuelli, Lucas | Massachusetts Institute of Technology
    Author: Tedrake, Russ | Massachusetts Institute of Technology
 
    keyword: Visual Learning; Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation

    Abstract : In this paper we explore using self-supervised correspondence for improving the generalization performance and sample efficiency of visuomotor policy learning. Prior work has primarily used approaches such as autoencoding, pose-based losses, and end-to-end policy optimization in order to train the visual portion of visuomotor policies. We instead propose an approach using self-supervised dense visual correspondence training, and show this enables visuomotor policy learning with surprisingly high generalization performance with modest amounts of data: using imitation learning, we demonstrate extensive hardware validation on challenging manipulation tasks with as few as 50 demonstrations. Our learned policies can generalize across classes of objects, react to deformable object configurations, and manipulate textureless symmetrical objects in a variety of backgrounds, all with closed-loop, real-time vision-based policies. Simulated imitation learning experiments suggest that correspondence training offers sample complexity and generalization benefits compared to autoencoding and end-to-end training.

- Differentiable Mapping Networks: Learning Structured Map Representations for Sparse Visual Localization

    Author: Karkus, Peter | National University of Singapore
    Author: Angelova, Anelia | Google Research
    Author: Vanhoucke, Vincent | Google Research
    Author: Jonschkowski, Rico | Google
 
    keyword: Visual Learning; Deep Learning in Robotics and Automation

    Abstract : Mapping and localization, preferably from a small number of observations, are fundamental tasks in robotics. We address these tasks by combining spatial structure (differentiable mapping) and end-to-end learning in a novel neural network architecture: the Differentiable Mapping Network (DMN). The DMN constructs a spatially structured view-embedding map and uses it for subsequent visual localization with a particle filter. Since the DMN architecture is end-to-end differentiable, we can jointly learn the map representation and localization using gradient descent. We apply the DMN to sparse visual localization, where a robot needs to localize in a new environment with respect to a small number of images from known viewpoints. We evaluate the DMN using simulated environments and a challenging real-world Street View dataset. We find that the DMN learns effective map representations for visual localization. The benefit of spatial structure increases with larger environments, more viewpoints for mapping, and when training data is scarce. Project website: https://sites.google.com/view/differentiable-mapping.

- Attentive Task-Net: Self Supervised Task-Attention Network for Imitation Learning Using Video Demonstration

    Author: Ramachandruni, Kartik | TCS Innovation Labs
    Author: Vankadari, Madhu Babu | TCS
    Author: Majumder, Anima | Tata Consultancy Services
    Author: Dutta, Samrat | TCS Research and Innovation
    Author: Swagat, Kumar | Tata Consultancy Services
 
    keyword: Visual Learning; Learning and Adaptive Systems; Deep Learning in Robotics and Automation

    Abstract : This paper proposes an end-to-end self-supervised feature representation network named Attentive Task-Net or AT-Net for video-based task imitation. The proposed AT-Net incorporates a novel multi-level spatial attention module to highlight spatial features corresponding to the intended task demonstrated by the expert. The neural connections in AT-Net ensure the relevant information in the demonstration is amplified and the irrelevant information is suppressed while learning task-specific feature embeddings. This is achieved by a weighted combination of multiple intermediate feature maps of the input image at different stages of the CNN pipeline. The weights of the combination are given by the compatibility scores, predicted by the attention module for respective feature maps. The AT-Net is trained using a metric learning loss which aims to decrease the distance between the feature representations of concurrent frames from multiple view points and increase the distance between temporally consecutive frames. The AT-Net features are then used to formulate a reinforcement learning problem for task imitation. Through experiments on the publicly available Multi-view pouring dataset, it is demonstrated that the output of the attention module highlights the task-specific objects while suppressing the rest of the background. The efficacy of the proposed method is further validated by qualitative and quantitative comparison with a state-of-the-art technique along with ablations.

- OpenLORIS-Object: A Robotic Vision Dataset and Benchmark for Lifelong Deep Learning

    Author: She, Qi | Intel Labs
    Author: Feng, Fan | City University of Hong Kong
    Author: Hao, Xinyue | Beijing University of Posts and Telecommunications
    Author: Yang, Qihan | City University of Hong Kong
    Author: Lan, Chuanlin | Wuhan University
    Author: Lomonaco, Vincenzo | University of Bologna
    Author: Shi, Xuesong | Intel
    Author: Wang, Zhengwei | Dublin City University
    Author: Guo, Yao | Imperial College London
    Author: Zhang, Yimin | Intel Corporation
    Author: Qiao, Fei | Tsinghua University
    Author: Chan, Rosa H. M. | City University of Hong Kong
 
    keyword: Visual Learning; Learning and Adaptive Systems; RGB-D Perception

    Abstract : The recent breakthroughs in computer vision have benefited from the availability of large representative datasets (e.g. ImageNet and COCO) for training. Yet, robotic vision poses unique challenges for applying visual algorithms developed from these standard computer vision datasets due to their implicit assumption over non-varying distributions for a fixed set of tasks. Fully retraining models each time a new task becomes available is infeasible due to computational, storage and sometimes privacy issues, while naive incremental strategies have been shown to suffer from catastrophic forgetting. It is crucial for the robots to operate continuously under open-set and detrimental conditions with adaptive visual perceptual systems, where lifelong learning is a fundamental capability. However, very few datasets and benchmarks are available to evaluate and compare emerging techniques. To fill this gap, we provide a new lifelong robotic vision dataset (�OpenLORIS-Object�). The dataset embeds the challenges faced by a robot in the real-life application and provides new benchmarks for validating lifelong object recognition algorithms. Moreover, we have provided a testbed of 9 state-of-the-art lifelong learning algorithms. Each of them involves 48 tasks with 4 evaluation metrics over the OpenLORIS-Object dataset. The results demonstrate that the object recognition task in the ever-changing difficulty environments is not solved.

- Unsupervised Depth Completion from Visual Inertial Odometry

    Author: Wong, Alex | University of California Los Angeles
    Author: Fei, Xiaohan | University of California, Los Angeles
    Author: Tsuei, Stephanie | University of California, Los Angeles
    Author: Soatto, Stefano | University of California, Los Angeles
 
    keyword: Visual Learning; Sensor Fusion

    Abstract : We describe a method to infer dense depth from camera motion and sparse depth as estimated using a visual-inertial odometry system. Unlike other scenarios using point clouds from lidar or structured light sensors, we only have a few hundred to a few thousand points, insufficient to inform the topology of the scene. Counter to current trends of end-to-end learning, our method first constructs a piecewise planar scaffolding of the scene, and then uses it to infer dense depth using the image along with the sparse points. We use a predictive cross-modal criterion, akin to ``self-supervision,'' measuring photometric consistency across time, forward-backward pose consistency, and geometric compatibility with the sparse point cloud. We also present the first visual-inertial + depth dataset, which we hope will foster additional exploration into combining the complementary strengths of visual and inertial sensors. To compare our method to prior work, we adopt the unsupervised KITTI depth completion benchmark, where we achieve state-of-the-art performance with significantly fewer parameters.

- Geometric Pretraining for Monocular Depth Estimation

    Author: Wang, Kaixuan | Hong Kong University of Science and Technology
    Author: Chen, Yao | ByteDance Inc
    Author: Guo, Hengkai | ByteDance AI Lab
    Author: Wen, Linfu | ByteDance AI Lab
    Author: Shen, Shaojie | Hong Kong University of Science and Technology
 
    keyword: Visual Learning; Learning and Adaptive Systems; Deep Learning in Robotics and Automation

    Abstract : ImageNet-pretrained networks have been widely used in transfer learning for monocular depth estimation. These pretrained networks are trained with classification losses that only semantic information is exploited while spatial information is ignored. However, both semantic and spatial information is important for per-pixel depth estimation. In this paper, we design a novel self-supervised geometric pretraining task that is tailored for monocular depth estimation using uncalibrated videos. The designed task decouples the structure information from input videos by a simple yet effective conditional autoencoder-decoder structure. Using almost unlimited videos from the internet, networks are pretrained to capture a variety of structures of the scene and can be easily transferred to depth estimation tasks using calibrated images. Extensive experiments are used to demonstrate that the proposed geometric-pretrained networks perform better than ImageNet-pretrained networks in terms of accuracy, few-shot learning and generalization ability. With the same learning method, geometric-transferred networks achieve new state-of-the-art results by a large margin. Pretrained networks will be open source soon.

## Soft Sensors and Actuators

- Active Acoustic Contact Sensing for Soft Pneumatic Actuators

    Author: Zöller, Gabriel Donald | TU Berlin
    Author: Wall, Vincent | TU Berlin
    Author: Brock, Oliver | Technische Universitét Berlin
 
    keyword: Soft Sensors and Actuators; Force and Tactile Sensing

    Abstract : We present an active acoustic sensor that turns soft pneumatic actuators into contact sensors. The whole surface of the actuator becomes a sensor, rendering the question of where best to place a contact sensor unnecessary. At the same time, the compliance of the soft actuator remains unaffected. A small, embedded speaker emits a frequency sweep which travels through the actuator before it is recorded with an embedded microphone. The specific contact state of the actuator affects how the sound is modulated while traversing the structure. We learn to recognize these changes in the sound and map them to the corresponding contact locations. We demonstrate the method on the PneuFlex actuator. The active acoustic sensor achieves a classification rate of 93% and mean regression error of 3.7mm. It is robust against background noises and different objects. Finally, we test it on a Panda robot arm and show that it is unaffected by motor noises and other active sensors.

- PneuAct-II: Hybrid Manufactured Electromagnetically Stealth Pneumatic Stepper Actuator

    Author: Sojoodi Farimani, Foad | University of Twente
    Author: Mojarradi, Morteza | University of Twente
    Author: Hekman, Edsko E.G. | University of Twente
    Author: Misra, Sarthak | University of Twente
 
    keyword: Additive Manufacturing; Medical Robots and Systems; Hydraulic/Pneumatic Actuators

    Abstract : Additive manufacturing (AM) is one of the emerging production methodologies transforming the industrial landscape. However, application of the technology in fluidic power transmission and actuation is still limited. AM pneumatic stepper motors have been previously introduced to the field of image-guided surgical robotics, where their disposability and customizability are considered a significant advantage over conventional manufacturing. However, intrinsic dimensional limitations of AM parts and their poor surface quality affect mechanical performance. In this study, a novel design, PneuAct-II, is presented combining AM, subtractive machining, and off-the-shelf components to achieve higher mechanical performance and resolution. Moreover, a motor identification setup has been built to automatically measure different aspects of the PneuAct motors, including wear, friction, leakage, and stall behavior at various boundary conditions. The effects of input pressure, stepping frequency, signal-width, and external torque on the stall behavior of motors with different clearances are studied. A maximum torque of 0.39N.m at an input pressure of 6.5bar is achieved for a motor with a total volume of 90cm3, and a clearance of 156&#120583;m. A nominal resolution of 2.25� at full-pitch and 1.125� at half-pitch is accomplished. Both resolution and mechanical performance (667 N.m/bar.m3) outperform the state-of-the-art.

- A Bidirectional 3D-Printed Soft Pneumatic Actuator and Graphite-Based Flex Sensor for Versatile Grasping

    Author: Low, Jin Huat | National University of Singapore
    Author: Goh, Jing Yuan Aaron | National University of Singapore
    Author: Cheng, Nicholas | National University of Singapore
    Author: Khin, Phone May | National University of Singapore
    Author: Han, Qian Qian | National University of Singapore
    Author: Yeow, Chen-Hua | National University of Singapore
 
    keyword: Soft Robot Applications; Grippers and Other End-Effectors; Soft Sensors and Actuators

    Abstract : THIS paper presents a bidirectional 3D-printed soft pneumatic actuator that is capable of inward and outward bending. A direct 3D-printing approach is adopted to fabricate the actuator, which reduces fabrication complexity and allows for easy customization of actuator dimensions for various applications. To illustrate the applicability of the bidirectional actuators, four of these actuators were incorporated into a gripper system. A suite of various functional grasping tasks, such as packaging, assembly, and alignment tasks, were successfully conducted. It was observed that the unique bidirectional bending characteristic of the actuator allows the gripper to grasp objects with sizes up to 245% larger than its default grasping width. To complement the gripper system, a graphite-based flex sensor that is able to sense bending in two directions is developed to control the bidirectional actuators. A preliminary test was conducted successfully where the user controlled the gripper system to grasp, hold, and release an object using a glove with the sensors.

- A Proprioceptive Bellows (PB) Actuator with Position Feedback and Force Estimation

    Author: Zhou, Jianshu | The Univerisity of Hong Kong
    Author: Chen, Yonghua | The University of Hong Kong
    Author: Chen, Xiaojiao | The University of Hong Kong
    Author: Wang, Zheng | The University of Hong Kong
    Author: Li, Yunquan | The University of Hong Kong
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Soft Robot Applications; Soft Sensors and Actuators; Perception for Grasping and Manipulation

    Abstract : Soft robot is known for great safety in human-centered environments due to its inherent compliance. However, the compliance resulting from the soft continuum structure and viscoelastic material also induces challenges for sensing and control of soft robots. In this paper, we propose a proprioceptive soft actuator design approach based on 3D printed conductive material and 3D printed deformable structure, such as bellows. The conductive bellow exhibits effective resistance change and structural deformation, thus provides a promising solution for the challenge of deformable soft robots that need integrated actuation and sensing. The proposed proprioceptive bellow actuator (PB actuator) achieves effective position feedback and real-time output force estimation. Using a dedicated control logic of the pressure controller, the PB actuator can not only provide anticipated motion but also estimate the interactive force based on real-time position sensing and input pressure. The design, fabrication, modeling, control, and experimental validation of our proposed PB actuator are discussed in detail in this paper. The parameters of PB actuator are highly customizable depending on the intended applications. Based on the proposed PB actuator, two specialized grippers, T and Y gripper, are designed and prototyped to demonstrate the grasping force estimation capability. The proposed proprioceptive soft robotic approach provides a promising solution to design behavior steerable soft robots.

-  Automatic Design of Soft Dielectric Elastomer Actuators with Optimal Spatial Electric Fields (I)

    Author: Chen, Feifei | Shanghai Jiao Tong University
    Author: Liu, Kun | shanghai jiaotong university
    Author: Wang, Yiqiang | Technical University of Denmark
    Author: Zou, Jiang | Shanghai Jiao Tong University
    Author: Gu, Guoying | Shanghai Jiao Tong University
    Author: Zhu, Xiangyang | Shanghai Jiao Tong University

- Stochastic Control for Orientation and Transportation of Microscopic Objects Using Multiple Optically Driven Robotic Fingertips (I)

    Author: Ta, Quang Minh | Nanyang Technological University
    Author: Cheah, C. C. | Nanyang Technological University
 
    keyword: Robust/Adaptive Control of Robotic Systems; Dexterous Manipulation; Automation at Micro-Nano Scales

    Abstract : The effect of Brownian motion on maneuvering of micro-objects in a fluid medium is one of the fundamental differences between micro-manipulation and robotic manipulators in the physical world. Besides, due to the limitation of feasible sensors and actuators in micro-manipulation, current control techniques for manipulation of micro-objects or cells are mostly dependent on the physical properties of target micro-objects or cells. In this paper, we propose the first stochastic control technique to achieve simultaneous orientation and transportation of micro-objects with Brownian perturbations. Several micro-particles which are optically trapped and driven by laser beams are utilized as fingertips to first grasp a target micro-object. Cooperative control of robot-assisted stage and the fingertips is then performed to achieve the control objective, in which the target micro-object is transported toward a desired position by using the robot-assisted stage, and at the same time, it is oriented toward a desired angular position by using the fingertips. This paper provides a stochastic control framework for simultaneous orientation and transportation of micro- objects with arbitrary types in the micro-world, and thus bringing micro-manipulation using optical tweezers closer to robotic manipulation in the physical world.

- Soft Fingertips with Adaptive Sensing and Active Deformation for Robust Grasping of Delicate Objects

    Author: He, Liang | Imperial College London
    Author: Lu, Qiujie | Imperial College London
    Author: Abad Guaman, Sara Adela | Imperial College London
    Author: Rojas, Nicolas | Imperial College London
    Author: Nanayakkara, Thrishantha | Imperial College London
 
    keyword: Soft Sensors and Actuators; Grasping; Grippers and Other End-Effectors

    Abstract : Soft fingertips have shown significant adaptability for grasping a wide range of object shapes thanks to elasticity. This ability can be enhanced to grasp soft, delicate objects by adding touch sensing. However, in these cases, the complete restraint and robustness of the grasps have proved to be challenging, as the exertion of additional forces on the fragile object can result in damage. This paper presents a novel soft fingertip design for delicate objects based on the concept of embedded air cavities, which allow the dual ability of tactile sensing and active shape-changing. The pressurized air cavities act as soft tactile sensors to control gripper position from internal pressure variation; and active fingertip deformation is achieved by applying positive pressure to these cavities, which then enable a delicate object to be kept securely in position, despite externally applied forces, by form closure. We demonstrate this improved grasping capability by comparing the displacement of grasped delicate objects exposed to high-speed motions. Results show that passive soft fingertips fail to restrain fragile objects at accelerations as low as 0.1 m/s^2, in contrast, with the proposed fingertips delicate objects are completely secure even at accelerations of more than 5 m/s^2.

- Sensorization of a Continuum Body Gripper for High Force and Delicate Object Grasping

    Author: Hughes, Josie | MIT
    Author: Li, Shuguang | MIT/Harvard University
    Author: Rus, Daniela | MIT
 
    keyword: Soft Sensors and Actuators; Grasping; Perception for Grasping and Manipulation

    Abstract : The goal of achieving `universal grasping' where many objects can be handled with minimal control input is the focus of much research due to potential high impact applications ranging from grocery packing to recycling. However, many of the grippers developed suffer from limited sensing capabilities which can prevent handing of both heavy bulky items and also lightweight delicate objects which require fine control when grasping. Sensorizing such grippers is often challenging due to the highly deformable surfaces. We propose a novel sensing approach which uses highly flexible latex bladders. By measuring changes in the air pressure of the bladders, normal force and longitudinal strain can be measured. These sensors have been integrated into a 'Magic Ball' origami gripper to provide both tactile and proprioceptive sensing. The sensors show reasonable sensitivity and repeatability, are durable and low-cost, and can be easily integrated into the gripper without affecting performance. When the sensors are used for classification, they enabled identification of 10 objects with over 90% accuracy, and also allow failure to be detected through slippage detection. A control algorithm has been developed which uses the sensor feedback to extend the capabilities of the gripper to include both delicate and strong grasping. It is shown that this closed loop controller enables delicate grasping of potato chips; 80% of those tested were grasped without damage.

- Eye-In-Hand Visual Servoing Enhanced with Sparse Strain Measurement for Soft Continuum Robots

    Author: Wang, Xiaomei | The University of Hong Kong
    Author: Fang, Ge | The University of Hong Kong
    Author: Wang, Kui | The University of Hong Kong
    Author: Xie, Xiaochen | The University of Hong Kong
    Author: Lee, Kit-Hang | The University of Hong Kong
    Author: Ho, Justin Di-Lang | The University of Hong Kong
    Author: Tang, Wai Lun | The University of Hong Kong
    Author: Lam, James | University of Hong Kong
    Author: Kwok, Ka-Wai | The University of Hong Kong
 
    keyword: Soft Sensors and Actuators; Modeling, Control, and Learning for Soft Robots; Visual Servoing

    Abstract : In the feature/object tracking of eye-in-hand visual servoing, 2D motion estimation relying only on image plane feedback is easily affected by vision occlusion, blurring, or poor lighting. For the commonly-used template matching method, tracking performance greatly depends on the image quality. Fiber Bragg gratings (FBGs), a type of high-frequency flexible strain sensors, can be used as an assistant device for soft robot control. We propose a method to enhance motion estimation in soft robotic visual servoing by fusing the results from template matching and FBG wavelength shift to achieve more accurate tracking in applications such as minimally invasive surgery. Path following performance is validated in a simulated laparoscopic scene and LEGO�-constructed scene, demonstrating significant improvement to feature tracking and robot motion, even under external forces.

- A Soft Gripper with Retractable Nails for Advanced Grasping and Manipulation

    Author: Jain, Snehal | SUTD
    Author: Stalin, Thileepan | Singapore University of Technology and Design
    Author: Subramaniam, Vignesh | University of Florida
    Author: Agarwal, Jai | Birla Institute of Technology and Science, Pilani
    Author: Valdivia y Alvarado, Pablo | Singapore University of Technology and Design, MIT
 
    keyword: Grasping; Grippers and Other End-Effectors; Soft Robot Materials and Design

    Abstract : This study describes the enhancement of a vacuum actuated soft gripper's grasping capabilities using retractable finger nails and an active re-configurable palm. The finger nail mechanism is pneumatically actuated and enables the gripper to perform complex grasping and manipulation tasks with high repeatability. The retracted nails can exert normal grasping forces of up to 1.8N and enable grasping of objects up to 200 microns thick from flat surfaces, while allowing the gripper to execute delicate pinch grasps. A wide array of robotic grasping tasks that were not possible without nails are also described.

- A Sensorized Hybrid Gripper to Evaluate a Grasping Quality Based on a Largest Minimum Wrench

    Author: Park, Wookeun | UNIST
    Author: Seo, Seongmin | UNIST
    Author: Oh, Jinhyeok | UNIST
    Author: Bae, Joonbum | UNIST
 
    keyword: Soft Sensors and Actuators; Grasping; Soft Robot Applications

    Abstract : Soft pneumatic grippers, which are based on soft pneumatic actuators have been widely studied owing to their simple morphological structure, inherent compliance, and pliable grasp. Additionally, the integration of the soft gripper with various sensors to improve its functionality has also been extensively studied. Although the soft gripper is known to exhibit a robust grasping performance without accurate control, the grasping quality of the soft gripper has rarely been studied due to the lack of adequate embedded sensors and quality metrics of the soft gripper. Therefore, a hybrid gripper, which is a soft gripper with rigid components, was sensorized by embedding a soft force sensor and a bending sensor to evaluate the grasping quality. Furthermore, a new grasping quality metric for a soft gripper was proposed, which calculates the largest minimum wrench of a convex hull in the wrench space. The proposed grasping quality metric was experimentally verified, and a real-time program was developed to evaluate the grasping quality.

- A Soft Pressure Sensor Skin for Hand and Wrist Orthoses

    Author: Tan, Xinyang | Imperial College London
    Author: He, Liang | Imperial College London
    Author: Cao, Jiangang | Xuzhou Central Hospital
    Author: Chen, Wei | Xuzhou Central Hospital
    Author: Nanayakkara, Thrishantha | Imperial College London
 
    keyword: Soft Sensors and Actuators; Force and Tactile Sensing; Human Factors and Human-in-the-Loop

    Abstract : Side effects caused by excessive contact pressure such as discomfort and pressure sores are commonly complained by patients wearing orthoses. These problems leading to low patient compliance decrease the effectiveness of the device. To mitigate side effects, this study describes the design and fabrication of a soft sensor skin with strategically placed 12 sensor units for static contact pressure measurement beneath a hand and wrist orthosis. A Finite Element Model was built to simulate the pressure on the hand of a subject and sensor specifications were obtained from the result to guide the design. By testing the fabricated soft sensor skin on the subject, contact pressure between 0.012 MPa and 0.046 MPa was detected, revealing the maximum pressure at the thumb metacarpophalangeal joint which was the same location of the highest pressure of simulation. In this paper, a new fabrication method combining etching and multi-material additive manufacture was introduced to produce multiple sensor units as a whole. Furthermore, a novel fish-scale structure as the connection among sensors was introduced to stabilize sensor units and reinforce the soft skin. Experimental analysis reported that the sensor signal is repeatable, and the fish-scale structure facilitates baseline resuming of sensor signal during relaxation.

- Characterisation of Self-Locking High-Contraction Electro-Ribbon Actuators

    Author: Taghavi, Majid | University of Bristol
    Author: Helps, Tim | University of Bristol
    Author: Rossiter, Jonathan | University of Bristol
 
    keyword: Soft Sensors and Actuators; Soft Robot Applications; Compliant Joint/Mechanism

    Abstract : Actuators are essential devices that exert force and do work. The contraction of an actuator (how much it can shorten) is an important property that strongly influences its applications, especially in engineering and robotics. While high contractions have been achieved by thermally- or fluidically-driven technologies, electrically-driven actuators typically cannot contract by more than 50%. Recently developed electro-ribbon actuators are simple, low cost, scalable electroactive devices powered by dielectrophoretic liquid zipping (DLZ) that exhibit high efficiency (~70%), high power equivalent to mammalian muscle (~100 W/kg), contractions exceeding 99%. We characterise the electro-ribbon actuator and explore contraction variation with voltage and load. We describe the unique self-locking behaviour of the electro-ribbon actuator which could allow for low-power-consumption solenoids and valves. Finally, we show the interdependence of constituent material properties and the important role that material choice plays in maximising performance.

- Helically Wrapped Supercoiled Polymer (HW-SCP) Artificial Muscles: Design, Characterization, and Modeling

    Author: Tsabedze, Thulani | University of Nevada, Reno
    Author: Mullen, Christopher | University of Nevada Reno
    Author: Coulter, Ryan | University of Nevada Reno
    Author: Wade, Scott | Boston University
    Author: Zhang, Jun | University of Nevada Reno
 
    keyword: Soft Sensors and Actuators; Biomimetics; Compliant Joint/Mechanism

    Abstract : Supercoiled polymer (SCP) artificial muscles exhibit many desirable properties such as large contractions and high power density. However, their full potential as robotic muscles is challenged by insufficient strain or force generation -- non-mandrel-coiled SCP actuators produce up to 10-20% strain; mandrel-coiled SCP actuators often lift up to 10-30g of weight. It is strongly desired but difficult to obtain SCP actuators that produce large strain and large force. In this paper, the design, characterization, and modeling of helically wrapped SCP (HW-SCP) actuators are presented, which can produce up to 40-60% strain and lift more than 90g of weight. By adjusting their configuration parameters, their strain and force performance can be changed. Experiments are conducted to characterize the force production, strain, and speed of HW-SCP actuators. A Preisach hysteresis model and a polynomial model are adopted to accurately capture the actuator behaviors. This work contributes to the field of high-performance artificial muscles.

- A Variable Stiffness Soft Continuum Robot Based on Pre-Charged Air, Particle Jamming, and Origami

    Author: Li, Yujia | Southwest Petroleum University
    Author: Ren, Tao | Xihua University
    Author: Chen, Yonghua | The University of Hong Kong
    Author: Chen, Michael Z. Q. | Nanjing University of Science and Technology
 
    keyword: Soft Sensors and Actuators; Soft Robot Applications; Grippers and Other End-Effectors

    Abstract : Soft continuum robots have many applications such as medical surgeries, service industries, rescue tasks, and underwater exploration. Flexibility and good accessibility of such robots are the key reasons for their popularity. However, the complexity of their structural design and control systems limit their broader applications. In this paper, a novel variable stiffness soft continuum robot based on pre-charged air, particle jamming, and origami is proposed. The robot is a bellow-like origami structure with internal chambers. A spine-like chamber is filled with particles, and three identical chambers surrounding the spine chamber are filled with pressurized air. When the origami structure is compressed, the particles are jammed by the compression force and the increased pressure of the three air chambers, thus increasing the overall stiffness of the robot. The robot expansion-contraction and bending are controlled by three tendons. An analytical model of the proposed stiffness variation mechanism has been developed. The effects of various parameters on the lateral and axial stiffness of the soft continuum robot have been investigated by experimental studies. A prototype robot has been fabricated to demonstrate grasping operations.

- A Pneumatic/Cable-Driven Hybrid Linear Actuator with Combined Structure of Origami Chambers and Deployable Mechanism

    Author: Zhang, Zhuang | Shanghai Jiao Tong University
    Author: Chen, Genliang | Shanghai Jiao Tong University
    Author: Wu, Haiyu | Shanghai Jiao Tong University
    Author: Kong, Lingyu | Zhejiang Lab
    Author: Wang, Hao | Shanghai Jiao Tong University
 
    keyword: Soft Sensors and Actuators; Mechanism Design; Soft Robot Materials and Design

    Abstract : Pneumatic actuators have been widely used in robotics due to their inherent compliance and relatively high strength. In this letter, the     Authors report on the design, analysis and experimental validation of a novel pneumatic/cable-driven hybrid linear actuator whose structure is a combination of origami chambers and passive deployable mechanism. Under the joint actuation of the cable and compressed air, the proposed actuator can generate bidirectional motion; meanwhile, both thrust and tensile force can be produced. The combined structure of the rigid deployable mechanism and the axially soft origami chambers possesses high radial stiffness and an extension ratio up to 200% without radial expansion. The position can be controlled at whole motion range through a simple strategy and the actuation pressure can be as low as 2 kPa at no load. The kinematic as well as the quasi-static model is developed to accurately predict the behavior of the actuator and design the control strategy. A prototype is built based on a new fabrication method, on which the validation experiments are conducted. The experimental results prove the effectiveness of the model and show that the prototype possesses acceptable positioning accuracy, even when an external load is exerted on its moving plate.

- Simple, Low-Hysteresis, Foldable, Fabric Pneumatic Artificial Muscle

    Author: Naclerio, Nicholas | University of California, Santa Barbara
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
 
    keyword: Soft Sensors and Actuators; Hydraulic/Pneumatic Actuators; Soft Robot Materials and Design

    Abstract : Soft robots offer advantages over rigid robots in adaptability, robustness to uncertainty, and human safety. However, realizing soft actuators for these robots is still a challenge. We present a simple, highly conformable pneumatic artificial muscle made of a thin, single layer of woven, bias-cut fabric. The airtight fabric is adhered together with a flexible adhesive, negating the need for a bladder or sewing. Thus, it is foldable when depressurized and behaves like a McKibben muscle when pressurized, but without the friction of a bladder or braided sheath. Experiments show that the muscle exhibits repeatable, near-linear behavior with less than 1% hysteresis, over an order of magnitude less than that of McKibben muscles. Dynamic testing shows that the muscle responds quickly, even at lengths over 60 cm, contracting in 0.03 s, an order of magnitude quicker than series pouch motors. A fatigue test shows that its life exceeds 100,000 cycles. We also demonstrate that the muscles well suited for steering tip-extending robots, and actuating folding, deployable structures. Our muscle offers improvements over various existing pneumatic artificial muscles, providing a simple new option for soft robotic actuation that has potential to advance the field.

- Flat Inflatable Artificial Muscles with Large Stroke and Adjustable Force-Length Relations (I)

    Author: Kwon, Junghan | Seoul National University
    Author: Yoon, Sohee John | Seoul National University
    Author: Park, Yong-Lae | Seoul National University
 
    keyword: Soft Sensors and Actuators; Hydraulic/Pneumatic Actuators; Soft Robot Materials and Design

    Abstract : The performance of inflatable artificial muscles depends greatly on their designs and the output shapes resulting from the geometric constraints. Although there have been attempts to apply physical constraints on the air chamber to achieve larger stroke lengths and increased force-length ratios, it has been difficult to achieve the above two goals while maintaining a compact form factor. In this article, we propose flat inflatable artificial muscles that have large contraction ratios (up to 0.5) and show increased forces in wider ranges of contractions by adding an internal geometric constraint. Addition of an external constraint, such as rigid plates, further increased the maximum contraction ratio (up to 0.553) through a synergistic effect. We show that various force-length relations can be achieved by adjusting the height of the plates. Theoretical models based on the geometry and the principle of virtual work are experimentally validated using actuator prototypes made of heat-sealable plastic sheets. Also, compact capacitive sensors are integrated in design for proprioceptive feedback of the proposed actuators, and their feasibility and effectiveness are experimentally evaluated through closed-loop control.

- Joint Rotation Angle Sensing of Flexible Endoscopic Surgical Robots

    Author: Lai, Wenjie | Nanyang Technological University
    Author: Cao, Lin | Nanyang Technological University
    Author: Phan, Phuoc Thien | Nanyang Technological University
    Author: Wu, I-Wen | FBG Sensing Consulting Service Inc
    Author: Tjin, Swee Chuan | Nanyang Technological University
    Author: Phee, Louis | Nanyang Technological University
 
    keyword: Soft Sensors and Actuators; Medical Robots and Systems; Tendon/Wire Mechanism

    Abstract : Accurate motion control of surgical robots is critical for the efficiency and safety of both teleoperated robotic surgery and ultimate autonomous robotic surgery. However, fine motion control for a flexible endoscopic surgical robot is highly challenging because of the motion hysteresis of tendon-sheath mechanisms (TSMs) in the long, tortuous, and dynamically shape-changing robot body. Aiming to achieve precise closed-loop motion control, we propose a small and flexible sensor to directly sense the large and sharp rotations of the articulated joints of a flexible endoscopic surgical robot. The sensor, a Fiber Bragg Grating (FBG) eccentrically embedded in a thin and flexible epoxy substrate, can significantly bend with a large bending angle range of [-62.9°, 75.5°] and small bending radius of 6.9 mm. Mounted in-between the two pivot-connected links of a joint, the sensor will bend once the joint is actuated, resulting in the wavelength shift of the FBG. In this study, the relationship between the wavelength shift and the rotation angle of the joint was theoretically modeled and then experimentally verified before and after the installation of the sensor in a robotic endoscopic grasper. The sensor can track the rotation of the robotic joint with an RMSE of 3.34°. This small and flexible sensor has good repeatability, high sensitivity (~147.5 pm/degree), and low hysteresis (7.72%). It is suitable for surgical robots with large joint rotation angles and small bending radius.

- Soft, Round, High Resolution Tactile Fingertip Sensors for Dexterous Robotic Manipulation

    Author: Romero, Branden | Massachusetts Institute of Technology
    Author: Veiga, Filipe Fernandes | MIT
    Author: Adelson, Edward | MIT
 
    keyword: Soft Sensors and Actuators; Dexterous Manipulation; Sensor-based Control

    Abstract : High resolution tactile sensors are often bulky and have shape profiles that make them awkward for use in manipulation. This becomes important when using such sensors as fingertips for dexterous multi-fingered hands, where boxy or planar fingertips limit the available set of smooth manipulation strategies. High resolution optical based sensors such as GelSight have until now been constrained to relatively flat geometries due to constraints on illumination geometry. Here, we show how to construct a rounded fingertip that utilizes a form of light piping for directional illumination. Our sensors can replace the standard rounded fingertips of the Allegro hand. They can capture high resolution maps of the contact surfaces, and can be used to support various dexterous manipulation tasks.

- Creating a Soft Tactile Skin Employing Fluorescence Based Optical Sensing

    Author: De Chiara, Federica | King's College London
    Author: Wang, Shuxin | Tianjin University
    Author: Liu, Hongbin | King's College London
 
    keyword: Soft Sensors and Actuators; Haptics and Haptic Interfaces; Surgical Robotics: Steerable Catheters/Needles

    Abstract : Currently, optical tactile sensors propose solutions to measure contact forces at the tip of flexible medical instruments. However, the sensing capability of normal pressures applied to the surface along the tool body is still an open challenge. To deal with this challenge, this paper proposes a sensor design employing an angled tip optical fiber to measure the intensity modulation of a fluorescence signal proportional to the applied force. The fiber is used as both emitter of the excitation light and receiver of the fluorescence signal. This configuration allows to (i) halve the number of optical fibers and (ii) improve the signal to noise ratio thanks to the wavelength shift between excitation and fluorescence emission. The proposed design makes use of soft and flexible materials only, avoiding the size constraints given by rigid optical components and facilitating further miniaturization. The employed materials are bio-compatible and guarantee chemical inertness and non-toxicity for medical uses. In this work, the sensing principle is validated using a single optical fiber. Then, a soft stretchable skin pad, containing four tactile sensing elements, is presented to demonstrate the feasibility of this new force sensor design.

- FootTile: A Rugged Foot Sensor for Force and Center of Pressure Sensing in Soft Terrain

    Author: Ruppert, Felix | Max Planck Institute for Intelligent Systems
    Author: Badri-Spröwitz, Alexander | Max Planck Institute for Intelligent Systems
 
    keyword: Soft Sensors and Actuators; Legged Robots; Force and Tactile Sensing

    Abstract : In this paper, we present FootTile, a foot sensor for reaction force and center of pressure sensing in challenging terrain. We compare our sensor design to standard biomechanical devices, force plates and pressure plates. We show that FootTile can accurately estimate force and pressure distribution during legged locomotion. FootTile weighs 0.9 g, has a sampling rate of 330 Hz, a footprint of 10x10 mm and can easily be adapted in sensor range to the required load case. In three experiments, we validate: first, the performance of the individual sensor, second an array of FootTiles for center of pressure sensing and third the ground reaction force estimation during locomotion in granular substrate. We then go on to show the accurate sensing capabilities of the waterproof sensor in liquid mud, as a showcase for real world rough terrain use.

- A Vision-Based Soft Somatosensory Approach for Distributed Pressure and Temperature Sensing

    Author: Yu, Chen | King's College London
    Author: Lindenroth, Lukas | University College London
    Author: Hu, Jian | King's College London
    Author: Back, Junghwan | King's College London
    Author: Abrahams, George | King's College London
    Author: Liu, Hongbin | King's College London
 
    keyword: Soft Sensors and Actuators; Soft Robot Applications; Soft Robot Materials and Design

    Abstract : Emulating the somatosensory system in instruments such as robotic hands and surgical grippers has the potential to revolutionize these domains. Using a combination of different sensing modalities is problematic due to the limited space and incompatibility of these sensing principles. Therefore, in contrast to the natural world, it is currently difficult to concurrently measure the force, geometry, and temperature of contact in conventional tactile sensing. To this end, here we present a soft multifunctional tactile sensing principle. The temperature is estimated using a thermos chromic liquid crystal ink layer which exhibits colour variation under temperature change. The shape and force of contact is estimated through the 3-D reconstruction of a deformed soft silicone surface. Our experiments have demonstrated high accuracy in all three modalities, which can be measured at the same time. The resolution of the distributed force and temperature sensing was found to be 0.7N and 0.4&#8451; respectively.

- A Stretchable Capacitive Sensory Skin for Exploring Cluttered Environments

    Author: Gruebele, Alexander | Stanford University
    Author: Roberge, Jean-Philippe | École De Technologie Supérieure
    Author: Zerbe, Andrew | Stanford University
    Author: Ruotolo, Wilson | Stanford University
    Author: Huh, Tae Myung | Stanford University
    Author: Cutkosky, Mark | Stanford University
 
    keyword: Soft Sensors and Actuators; Sensor Networks; Grasping

    Abstract : We present a design and fabrication method using low-cost materials for a new tactile sensor that is highly stretchable (up to 60%) and for which the signal is substantially unaffected by stretching, immersion in water, or electromag- netic noise. The sensor can be wrapped around the front, side and back surfaces of fingers and allows them to flex without affecting the signal. The sensor consists of multiple layers of UV laser-patterned metallic capacitive elements and interconnects, encapsulated in a silicone membrane. It survives large impacts and over a thousand stretching cycles without a change in performance. We use low-cost capacitance-to-digital converters for filtering and communication. To meet different requirements for the front and back surfaces of fingers, the sensitivity of taxels can be tuned by varying the dielectric pattern and in CDC firmware. The skin detects contacts as low as 60 Pa and is intended for manipulation in cluttered environments where the back of the hand may accidentally brush objects.

## Wearable Robots

- SwarmRail: A Novel Overhead Robot System for Indoor Transport and Mobile Manipulation

    Author: Görner, Martin | German Aerospace Center (DLR)
    Author: Benedikt, Fabian | German Aerospace Center (DLR)
    Author: Grimmel, Ferdinand | German Aerospace Center (DLR)
    Author: Hulin, Thomas | German Aerospace Center (DLR)
 
    keyword: Wheeled Robots; Mobile Manipulation; Logistics

    Abstract : SwarmRail represents a novel solution to overhead manipulation from a mobile unit that drives in an aboveground rail-structure. The concept is based on the combination of omnidirectional mobile platform and L-shaped rail profiles that form a through-going central gap. This gap makes possible mounting a robotic manipulator arm overhead at the underside of the mobile platform. Compared to existing solutions, SwarmRail enables continuous overhead manipulation while traversing rail crossings. It also can be operated in a robot swarm, as it allows for concurrent operation of a group of mobile SwarmRail units inside a single rail network. Experiments on a first functional demonstrator confirm the functional capability of the concept. Potential fields of applications reach from industry over logistics to vertical farming.

- Fast Local Planning and Mapping in Unknown Off-Road Terrain

    Author: Overbye, Timothy | Texas A&amp;M University
    Author: Saripalli, Srikanth | Texas A&amp;M
 
    keyword: Wheeled Robots; Field Robots; Motion and Path Planning

    Abstract : In this paper, we present a fast, on-line mapping and planning solution for operation in unknown, off-road, environments. We combine obstacle detection along with a terrain gradient map to make simple and adaptable cost map. This map can be created and updated at 10~Hz. An A* planner finds optimal paths over the map. Finally, we take multiple samples over the control input space and do a kinematic forward simulation to generated feasible trajectories. Then the most optimal trajectory, as determined by the cost map and proximity to A* path, is chosen and sent to the controller. Our method allows real time operation at rates of 30~Hz. We demonstrate the efficiency of our method in various off-road terrain at high speed.

- Multifunctional 3-DOF Wearable Supernumerary Robotic Arm Based on Magnetorheological Clutches

    Author: Veronneau, Catherine | Universite De Sherbrooke
    Author: Denis, Jeff | Université De Sherbrooke
    Author: Lebel, Louis-Philippe | Université De Sherbrooke
    Author: Denninger, Marc | Université De Sherbrooke
    Author: Blanchard, Vincent | Université De Sherbrooke
    Author: Girard, Alexandre | Université De Sehrbrooke
    Author: Plante, Jean-Sebastien | Université De Sherbrooke
 
    keyword: Wearable Robots; Product Design, Development and Prototyping; Physical Human-Robot Interaction

    Abstract : Supernumerary robotic limbs (SRL) are wearable extra limbs intended to help humans perform physical tasks beyond conventional capabilities in domestic or industrial applications. However, unique design challenges are associated with SRLs as they are mounted on the human body. SRLs must 1) be lightweight to avoid burdening the user, 2) be fast enough to compensate for human motions, 3) be strong enough to accomplish various tasks, 4) have high force-bandwidth and good backdrivability to control interaction forces. This paper studies the potential of a 3-DOF supernumerary robotic arm powered by magnetorheological (MR) clutches and hydrostatic transmission lines. The tethered configuration allows the power-unit to be located on the ground, which minimizes the mass worn (4.2 kg) by the user. MR clutches minimize the actuation inertia in order to provide fast dynamics and backdrivability. An experimental open-loop force-bandwidth of 18 Hz is founded at each joint and the maximal speed reached by the end-effector is 3.4 m/s, which is sufficient for compensating human motions. Also, the two first joints provide 35 Nm and the third joint, 29 Nm, which is strong enough to hold manual tools. Finally,the SRL is put in real practical situations, as fruit picking, painting, tools holding and badminton playing. The capability of the proposed SRL to perform successfully various tasks with high speed and smoothness suggests a strong potential of SRLs to become future commonly used devices.

- Leveraging the Human Operator in the Design and Control of Supernumerary Robotic Limbs

    Author: Guggenheim, Jacob | MIT
    Author: Hoffman, Rachel | Massachusetts Institute of Technology
    Author: Song, Hanjun | Massachusetts Institute of Technology
    Author: Asada, Harry | MIT
 
    keyword: Wearable Robots; Physical Human-Robot Interaction; Human-Centered Robotics

    Abstract : A human has over 200 muscles in the body, creating a high degree of flexibility and redundancy in movement. This paper exploits this high degree of redundancy for the actuation and control of Supernumerary Robotic Limbs (SuperLimbs), which are attached to a human body. SuperLimbs containing many active joints tend to be too heavy to wear comfortably. Since SuperLimbs are attached to a human body at their base, Superlimbs can be positioned directly by moving the base with movements of the human body. No active joints are needed for the SuperLimbs in certain directions if the human body can generate the same movements as the SuperLimbs, thus allowing for the design of reduced-actuator Superlimbs. Here, we present a method for quantifying the usable degrees of freedom (DOFs) of a human body for a specific task so that the number of Superlimb actuators can be reduced. The high degree of redundant human DOFs can also be utilized for communication and control. Human's fingers are often redundant for performing a task, e.g. holding a box. Although both hands are busy, some combination of the finger forces is still available for generating signal patterns. An algorithm is developed for generating coded finger force patterns without interfering with the performance of the primary task. Both methods are implemented on a simple SuperLimb and a human subject demonstrates the usefulness of the methods.

- Revisiting Scaling Laws for Robotic Mobility in Granular Media

    Author: Thoesen, Andrew | Arizona State University
    Author: McBryan, Teresa | Arizona State University
    Author: Green, Marko | Arizona State University
    Author: Martia, Justin | Arizona State University
    Author: Mick, Darwin | Arizona State University
    Author: Marvi, Hamidreza | Arizona State University
 
    keyword: Wheeled Robots; Field Robots; Space Robotics and Automation

    Abstract : The development, building, and testing of robotic vehicles for applications in deformable media can be costly. Typical approaches rely on full-sized builds empirically evaluating performance metrics such as drawbar pull and slip. Recently developed granular scaling laws offer a new opportunity for terramechanics as a field. Using non-dimensional analysis on the wheel characteristics and treating the terrain as a deformable continuum, the performance of a larger, more massive wheel may be predicted from a smaller one. This allows for new wheel design approaches. However, robot-soil interaction and specific characteristics of the soil or robot dynamics may create discrepancies in prediction. In particular, we find that for a lightweight rover (2-5 kg), the scaling laws significantly overpredicted mechanical power requirements. To further explore the limitations of the current granular scaling laws, a pair of differently sized grousered wheels were tested at three masses and a pair of differently sized sandpaper wheels were tested at two masses across five speeds. Analysis indicates similar error for both designs, a mass dependency for all five pairs that explains the laws' overprediction, and a speed dependency for both of the heaviest sets. The findings create insights for using the laws with lightweight robots in granular media and generalizing granular scaling laws.

- Learning a Control Policy for Fall Prevention on an Assistive Walking Device

    Author: C V Kumar, Visak | Georgia Institute of Technology
    Author: Ha, Sehoon | Google Brain
    Author: Sawicki, Gregory | Georgia Institute of Technology
    Author: Liu, Karen | Georgia Tech
 
    keyword: Wearable Robots; Deep Learning in Robotics and Automation; Physically Assistive Devices

    Abstract : Fall prevention is one of the most important components in senior care. We present a technique to augment an assistive walking device with the ability to prevent falls. Given an existing walking device, our method develops a fall predictor and a recovery policy by utilizing the onboard sensors and actuators. The key component of our method is a robust human walking policy that models realistic human gait under a moderate level of perturbations. We use this human walking policy to provide training data for the fall predictor, as well as to teach the recovery policy how to best modify the person's gait when a fall is imminent. Our evaluation shows that the human walking policy generates walking sequences similar to those reported in biomechanics literature. Our experiments in simulation show that the augmented assistive device can indeed help recover balance from a variety of external perturbations. We also provide a quantitative method to evaluate the design choices for an assistive device.

- Assistive Force of a Belt-Type Hip Assist Suit for Lifting the Swing Leg During Walking

    Author: Guo, Shijie | Hebei University of Technology
    Author: Xiang, Qian | Hebei University of Technology
    Author: Hashimoto, Kazunobu | Ningbo Intelligent Manufacturing Industry Research Institute
    Author: Jin, Shanhai | Yanbian University
 
    keyword: Wearable Robots; Rehabilitation Robotics; Physically Assistive Devices

    Abstract : This paper proposes a relatively simple function of assistive force for a belt-type hip assist suit developed by the     Authors' group. The function, which is inspired by the muscle force of the rectus femoris, contains only two parameters, the magnitude and a phase shift factor. Thus, it can reduce the amount of calculation in generating the desired assistive force during walking. Tests were performed on three healthy subjects to confirm its effect and to investigate its influence on the motions of hip, knee and ankle joints. It was demonstrated that the effect of the assist depended greatly on the phase shift factor, i.e., the location of the peak of the assistive force in a swing period. A large effect was observed when the peak of the assistive force came at mid-swing phase. The results of the tests showed that the proposed force function could help to increase walk ratio (the ratio of step length to the number of steps per minute) by an average value of 11.2% at a force magnitude of 35 N, which could produce an assistive torque of the same order as the magnitude of the muscle force of the rectus femoris around the hip joint.

- Soft Pneumatic System for Interface Pressure Regulation and Automated Hands-Free Donning in Robotic Prostheses

    Author: Ambrose, Alexander | Georgia Institute of Technology
    Author: Hammond III, Frank L. | Georgia Institute of Technology
 
    keyword: Wearable Robots; Prosthetics and Exoskeletons; Soft Robot Applications

    Abstract : This paper discusses the design and preliminary evaluation of a soft pneumatic socket (SPS) with real-time pressure regulation and an automated underactuated donning mechanism (UDM). The ability to modulate the pressure at the human-socket interface of a prosthesis or wearable device to accommodate user's activities has the potential to make the user more comfortable. Furthermore, a hands-free, underactuated donning mechanism designed to reliably and safely don the socket onto the user may increase the convenience of prostheses and wearable devices. The pneumatic socket and donning mechanism are evaluated on synthetic forearm model designed to closely match the mechanical properties of the human forearm. The pneumatic socket was tested to determine the maximum loads it can withstand before slipping and the displacement of the socket after loading. The donning mechanism was able to successfully don the socket on to the replica forearm with a 100% success rate for the 30 trials that were tested. Both devices were also tested to determine the pressures they impart on the user. The highest pressures the socket can impart on the user is 4psi and the maximum pressure the donning mechanism imparts on the user is 0.83psi. These pressures were found to be lower than the reported pressures that cause pain and tissue damage.

- Automated Detection of Soleus Concentric Contraction in Variable Gait Conditions for Improved Exosuit Control

    Author: Nuckols, Richard | Harvard University
    Author: Swaminathan, Krithika | Harvard University
    Author: Lee, Sangjun | Harvard University
    Author: Awad, Louis | Harvard University
    Author: Walsh, Conor James | Harvard University
    Author: Howe, Robert D. | Harvard University
 
    keyword: Wearable Robots; Computer Vision for Other Robotic Applications; Physical Human-Robot Interaction

    Abstract : Exosuits can reduce metabolic demand and improve gait. Controllers explicitly derived from biological mechanisms that reflect the user's joint or muscle dynamics should in theory allow for individualized assistance and enable adaptation to changing gait. With the goal of developing an exosuit control strategy based on muscle power, we present an approach for estimating, at real time rates, when the soleus muscle begins to generate positive power. A low-profile ultrasound system recorded B-mode images of the soleus in walking individuals. An automated routine using optical flow segmented the data to a normalized gait cycle and estimated the onset of concentric contraction at real-time rates (~130Hz). Segmentation error was within 1% of the gait cycle compared to using ground reaction forces. Estimation of onset of concentric contraction had a high correlation (R2=0.92) and an RMSE of 2.6% gait cycle relative to manual estimation. We demonstrated the ability to estimate the onset of concentric contraction during fixed speed walking in healthy individuals that ranged from 39.3% to 45.8% of the gait cycle and feasibility in two persons post-stroke walking at comfortable walking speed. We also showed the ability to measure a shift in onset timing to 7% earlier when the biological system adapts from level to incline walking. Finally, we provided an initial evaluation for how the onset of concentric contraction might be used to inform exosuit control in level and incline walking.

- Soft Sensing Shirt for Shoulder Kinematics Estimation

    Author: Jin, Yichu | Harvard University
    Author: Glover, Christina | Harvard University
    Author: Cho, Haedo | Harvard University
    Author: Araromi, Oluwaseun Adelowo | Harvard University
    Author: Graule, Moritz A. | Harvard University
    Author: Li, Na | Harvard University
    Author: Wood, Robert | Harvard University
    Author: Walsh, Conor James | Harvard University
 
    keyword: Wearable Robots; Soft Sensors and Actuators

    Abstract : Soft strain sensors have been explored as an unobtrusive approach for wearable motion tracking. However, accurate tracking of multi-DOF noncyclic joint movements remains a challenge. This paper presents a soft sensing shirt for tracking shoulder kinematics of both cyclic and random arm movements in 3 DOFs: adduction/abduction, horizontal flexion/extension, and internal/external rotation. The sensing shirt consists of 8 textile-based capacitive strain sensors sewn around the shoulder joint that communicate to a customized readout electronics board through sewn micro coaxial cables. An optimized sensor design includes passive shielding and demonstrates high linearity and low hysteresis, making it suitable for wearable motion tracking. In a study with a single human subject, we evaluated the tracking capability of the integrated shirt in comparison with a ground truth optical motion capture system. An ensemble-based regression algorithm was implemented in post-processing to estimate joint angles and angular velocities from the strain sensor data. Results demonstrated root mean square errors less than 4.5 deg for joint angle estimation and normalized root mean square errors less than 4% for joint velocity estimation. Furthermore, we applied a recursive feature elimination-based sensor selection analysis to down select the number of sensors for future shirt designs. This sensor selection analysis found that 5 sensors out of 8 were sufficient to generate comparable accuracies.

- Characterizing Torso Stiffness in Female Adolescents with and without Scoliosis

    Author: Murray, Rosemarie | Columbia University
    Author: Ophaswongse, Chawin | Columbia University
    Author: Park, Joon-Hyuk | University of Central Florida
    Author: Agrawal, Sunil | Columbia University
 
    keyword: Wearable Robots; Rehabilitation Robotics; Physical Human-Robot Interaction

    Abstract : Adolescent Idiopathic Scoliosis (AIS) is a spinal curvature that affects 3% of the population and disproportionately affects females. It is treated with bracing and many researchers are developing models of the torso to optimize the effectiveness of brace designs. Unfortunately, the data available to create these models is limited by the experimental methods employed. One method, in vitro spine cadaver stiffness measurements, is generally based on specimens from the elderly, which are not representative of the adolescent population. The other method, radiographic studies, can only provide a limited amount of information because of the radiation exposure that multiple images require. In this work, we present a Robotic Spine Exoskeleton (RoSE) tailored to the population in greatest need of AIS interventions--female adolescents. We use it to create a three-dimensional stiffness characterization of the torso in vivo for eight female adolescents with scoliosis and eight without this condition. The key findings include an interaction effect of DOF and torso segment on translational collinear stiffnesses, and an interaction effect of DOF and group on rotational collinear stiffnesses. We also found that the 3D coupling stiffness pattern is in line with that of the human spine, regardless of spinal deformity. Also, the magnitude of the torso stiffness for the tested population is less than that of the adult male population previously characterized.

## Cognitive Human-Robot Interaction
- Scaled Autonomy: Enabling Human Operators to Control Robot Fleets

    Author: Swamy, Gokul | UC Berkeley
    Author: Reddy, Siddharth | UC Berkeley
    Author: Levine, Sergey | UC Berkeley
    Author: Dragan, Anca | University of California Berkeley
 
    keyword: Cognitive Human-Robot Interaction; Telerobotics and Teleoperation; Learning from Demonstration

    Abstract : Autonomous robots often encounter challenging situations where their control policies fail and an expert human operator must briefly intervene, e.g., through teleoperation. In settings where multiple robots act in separate environments, a single human operator can manage a fleet of robots by identifying and teleoperating one robot at any given time. The key challenge is that users have limited attention: as the number of robots increases, users lose the ability to decide which robot requires teleoperation the most. Our goal is to automate this decision, thereby enabling users to supervise more robots than their attention would normally allow for. Our insight is that we can model the user's choice of which robot to control as an approximately optimal decision that maximizes the user's utility function. We learn a model of the user's preferences from observations of the user's choices in easy settings with a few robots, and use it in challenging settings with more robots to automatically identify which robot the user would most likely choose to control, if they were able to evaluate the states of all robots at all times. We run simulation experiments and a user study with twelve participants that show our method can be used to assist users in performing a simulated navigation task. We also run a hardware demonstration that illustrates how our method can be applied to a real-world mobile robot navigation task.

- An Actor-Critic Approach for Legible Robot Motion Planner

    Author: Zhao, Xuan | City University of Hong Kong
    Author: Fan, Tingxiang | The University of Hong Kong
    Author: Wang, Dawei | The University of Hong Kong
    Author: Hu, Zhe | City University of Hong Kong
    Author: Han, Tao | City University of Hong Kong
    Author: Pan, Jia | University of Hong Kong
 
    keyword: Cognitive Human-Robot Interaction; AI-Based Methods

    Abstract : In human-robot collaboration, it is crucial for the robot to make its intentions clear and predictable to the human partners. Inspired by the mutual learning and adaptation of human partners, we suggest an actor-critic approach for a legible robot motion planner. This approach includes two neural networks and a legibility evaluator: 1) A policy network based on deep reinforcement learning (DRL); 2) A Recurrent Neural Networks (RNNs) based sequence to sequence (Seq2Seq) model as a motion predictor; 3) A legibility evaluator that maps motion to legible reward. Through a series of human-subject experiments, we demonstrate that with a simple handicraft function and no real-human data, our method lead to improved collaborative performance against a baseline method and a non-prediction method.

- May I Draw Your Attention? Initial Lessons from the Large-Scale Generative Mark Maker

    Author: Phillips, Aidan | University of Pittsburgh
    Author: Vinoo, Ashwin | Oregon State University
    Author: Fitter, Naomi T. | University of Southern California
 
    keyword: Entertainment Robotics; Human-Centered Robotics; Cognitive Human-Robot Interaction

    Abstract : Everyday robots are emerging in contexts from household chores to entertainment, and an accompanying need arises to understand perceptions of these systems. We propose a sleek 2D hanging pen plotter as a useful tool for both furthering art and studying human responses to ambient everyday robots. Aspects of this system and the pieces it produced were grounded in ideas from conceptual and generative art. We installed our robotic system in three different spaces in the wild, recorded video footage of the installations, consulted with expert artists, and interviewed a subset of exhibit visitors. Key findings included visitor assumptions of robot responsiveness and differences in user behaviors between performing arts and engineering environments. The exploratory products of this work can benefit artists who employ generative concepts in their work, as well as roboticists who seek further understanding of human-robot interaction in the wild.

- Intuitive 3D Control of a Quadrotor in User Proximity with Pointing Gestures

    Author: Gromov, Boris | IDSIA
    Author: Guzzi, Jerome | IDSIA, USI-SUPSI
    Author: Gambardella, Luca | USI-SUPSI
    Author: Giusti, Alessandro | IDSIA Lugano, SUPSI
 
    keyword: Cognitive Human-Robot Interaction; Gesture, Posture and Facial Expressions; Human-Centered Robotics

    Abstract : We present an approach for controlling the position of a quadrotor in 3D space using pointing gestures; the task is difficult because it is in general ambiguous to infer where, along the pointing ray, the robot should go. We propose and validate a pragmatic solution based on a push button acting as a simple additional input device which switches between different virtual workspace surfaces. Results of a study involving ten subjects show that the approach performs well on a challenging 3D piloting task, where it compares favorably with joystick control.

- Joint Inference of States, Robot Knowledge, and Human (False-)Beliefs

    Author: Yuan, Tao | University of California, Los Angeles
    Author: Liu, Hangxin | University of California, Los Angeles
    Author: Fan, Lifeng | University of California, Los Angeles
    Author: Zheng, Zilong | UCLA
    Author: Gao, Tao | Massachusetts Institute of Technology
    Author: Zhu, Yixin | University of California, Los Angeles
    Author: Zhu, Song-Chun | UCLA
 
    keyword: Cognitive Human-Robot Interaction; Semantic Scene Understanding; Visual Learning

    Abstract : Aiming to understand how human (false-)belief---a core socio-cognitive ability---would affect human interactions with robots, this paper proposes to adopt a graphical model to unify the representation of object states, robot knowledge, and human (false-)beliefs. Specifically, a pg is learned from a single-view spatiotemporal parsing by aggregating various object states along the time; such a learned representation is accumulated as the robot's knowledge. An inference algorithm is derived to fuse individual pg from all robots across multi-views into a joint pg, which affords more effective reasoning and inference capability to overcome the errors originated from a single view. In the experiments, through the joint inference over pgs, the system correctly recognizes human (false-)belief in various settings and achieves better cross-view accuracy on a challenging small object tracking dataset.

- Visual-Audio Cognitive Architecture for Autonomous Learning of Face Localisation by a Humanoid Robot

    Author: Gonzalez-Billandon, Jonas | Istituto Italiano Di Tecnologia, University of Genova
    Author: Sciutti, Alessandra | Italian Institute of Technology
    Author: Tata, Matthew | University of Lethbridge
    Author: Sandini, Giulio | Italian Institute of Technology
    Author: Rea, Francesco | Istituto Italiano Di Tecnologia
 
    keyword: Cognitive Human-Robot Interaction; Social Human-Robot Interaction; Learning and Adaptive Systems

    Abstract : Newborn infants are naturally attracted to human faces, a crucial source of information for social interaction. In robotics, acquisition of such information is crucial and social robots should also learn to exhibit such social skill. Deep learning algorithms have been used to resemble hierarchical processes of learning and generalization inspired by biological models. However, a major drawback of these methods is that they are not autonomous. They require a large amount of data and human supervision is extensively involved in the process. The challenge is to propose autonomous behaviours that can guide a robot to learn relevant social skills. In this work, we address this problem in the field of facial localisation by proposing a learning system based on a biologically-inspired cognitive framework. The proposed cognitive architecture builds on existing work and uses visual-audio orienting attention and a proactive stereo vision mechanism to autonomously direct a robot's attentive focus towards human faces. The gathered information is used to incrementally generate a dataset that can be used to train a state-of-the-art deep network. The learning system replicates the typical learning process of infants and enhances the learning generalization process by leveraging on the interaction experience with people. The integration of HRI with machine learning, inspired by early development in humans, constitutes an innovative approach for improving autonomous learning in robots.

- Motion Reasoning for Goal-Based Imitation Learning

    Author: Huang, De-An | Stanford University
    Author: Chao, Yu-Wei | Univeristy of Michigan
    Author: Paxton, Chris | NVIDIA Research
    Author: Deng, Xinke | University of Illinois at Urbana-Champaign
    Author: Fei-Fei, Li | Stanford University
    Author: Niebles, Juan Carlos | Stanford University
    Author: Garg, Animesh | University of Toronto
    Author: Fox, Dieter | University of Washington
 
    keyword: Cognitive Human-Robot Interaction; Perception for Grasping and Manipulation; Task Planning

    Abstract : We address goal-based imitation learning, where the aim is to output the symbolic goal from a third-person video demonstration. This enables the robot to plan for execution and reproduce the same goal in a completely different environment. The key challenge is that the goal of a video demonstration is often ambiguous at the level of semantic actions. The human demonstrators might unintentionally achieve certain subgoals in the demonstrations with their actions. Our main contribution is to propose a motion reasoning framework that combines task and motion planning to disambiguate the true intention of the demonstrator in the video demonstration. This allows us to recognize the goals that cannot be disambiguated by previous action-based approaches. We evaluate our approach on a new dataset of 96 video demonstrations in a mockup kitchen environment. We show that our motion reasoning plays an important role in recognizing the actual goal of the demonstrator and improves the success rate by over 20%. We further show that by using the automatically inferred goal from the video demonstration, our robot is able to reproduce the same task in a real kitchen environment.

- Flexible Online Adaptation of Learning Strategy Using EEG-Based Reinforcement Signals in Real-World Robotic Applications

    Author: Kim, Su Kyoung | German Research Center for Artificial Intelligence (DFKI)
    Author: Kirchner, Elsa Andrea | University of Bremen
    Author: Kirchner, Frank | University of Bremen
 
    keyword: Cognitive Human-Robot Interaction; Brain-Machine Interface; Learning and Adaptive Systems

    Abstract : Flexible adaptation of learning strategy depending on online changes of the user's current intents have a high relevance in human-robot collaboration. In our previous study, we proposed an intrinsic interactive reinforcement learning approach for human-robot interaction, in which a robot learns his/her action strategy based on intrinsic human feedback that is generated in the human's brain as neural signature of the human's implicit evaluation of the robot's actions. Our approach has an inherent property that allows robots to adapt their behavior depending on online changes of the human's current intents. Such flexible adaptation is possible, since robot learning is updated in real time by human's online feedback. In this paper, the adaptivity of robot learning is tested on eight subjects who change their current control strategy by adding a new gesture to the previous used gestures. This paper evaluates the learning progress by analyzing learning phases (before and after adding a new gesture for control). The results show that the robot can adapt the previously learned policy depending on online changes of the user's intents. Especially, learning progress is interrelated with the classification performance of electroencephalograms (EEGs), which are used to measure the human's implicit evaluation of the robot's actions.

- Object-Oriented Semantic Graph Based Natural Question Generation

    Author: Moon, Jiyoun | Seoul National University
    Author: Lee, Beom-Hee | Seoul National University
 
    keyword: Cognitive Human-Robot Interaction; Cooperating Robots; SLAM

    Abstract : Generating a natural question can enable autonomous robots to propose problems according to their surroundings. However, recent studies on question generation rarely consider semantic graph mapping, which is widely used to understand environments. In this paper, we introduce a method to generate natural questions using object-oriented semantic graphs. First, a graph convolutional network extracts features from the graph. Then, a recurrent neural network generates the natural question from the extracted features. Using graphs, we can generate natural questions for both single and sequential scenes. The proposed method outperforms conventional methods on a publicly available dataset for single scenes and can generate questions for sequential scenes.

- Towards Safe Human-Robot Collaboration Using Deep Reinforcement Learning

    Author: El-Shamouty, Mohamed | Fraunhofer IPA
    Author: Wu, Xinyang | Fraunhofer IPA
    Author: Yang, Shanqi | Fraunhofer IPA
    Author: Albus, Marcel | Fraunhofer IPA
    Author: Huber, Marco F. | University of Stuttgart
 
    keyword: Cognitive Human-Robot Interaction; Robot Safety; Motion and Path Planning

    Abstract : Safety in Human-Robot Collaboration (HRC) is a bottleneck to HRC-productivity in industry. With robots being the main source of hazards, safety engineers use over-emphasized safety measures, and carry out lengthy and expensive risk assessment processes on each HRC-layout reconfiguration. Recent advances in deep Reinforcement Learning (RL) offer solutions to add intelligence and comprehensibility of the environment to robots. In this paper, we propose a framework that uses deep RL as an enabling technology to enhance intelligence and safety of the robots in HRC scenarios and, thus, reduce hazards incurred by the robots. The framework offers a systematic methodology to encode the task and safety requirements and context of applicability into RL settings. The framework also considers core components, such as behavior explainer and verifier, which aim for transferring learned behaviors from research labs to industry. In the evaluations, the proposed framework shows the capability of deep RL agents learning collision-free point-to-point motion on different robots inside simulation, as shown in the supplementary video.

- Deep Compositional Robotic Planners That Follow Natural Language Commands

    Author: Kuo, Yen-Ling | MIT
    Author: Katz, Boris | MIT
    Author: Barbu, Andrei | MIT
 
    keyword: Cognitive Human-Robot Interaction; Learning from Demonstration

    Abstract : We demonstrate how a sampling-based robotic planner can be augmented to learn to understand a sequence of natural language commands in a continuous configuration space to move and manipulate objects. Our approach combines a deep network structured according to the parse of a complex command that includes objects, verbs, spatial relations, and attributes, with a sampling-based planner, RRT. A recurrent hierarchical deep network controls how the planner explores the environment, determines when a planned path is likely to achieve a goal, and estimates the confidence of each move to trade off exploitation and exploration between the network and the planner. Planners are designed to have near-optimal behavior when information about the task is missing, while networks learn to exploit observations which are available from the environment, making the two naturally complementary. Combining the two enables generalization to new maps, new kinds of obstacles,and more complex sentences that do not occur in the training set. Little data is required to train the model despite it jointly acquiring a CNN that extracts features from the environment as it learns the meanings of words. The model provides a level of interpretability through the use of attention maps allowing users to see its reasoning steps despite being an end-to-end model. This end-to-end model allows robots to learn to follow natural language commands in challenging continuous environments.

- Learning User Preferences from Corrections on State Lattices

    Author: Wilde, Nils | University of Waterloo
    Author: Kulic, Dana | Monash University
    Author: Smith, Stephen L. | University of Waterloo
 
    keyword: Cognitive Human-Robot Interaction; Motion and Path Planning

    Abstract :  Enabling a broader range of users to efficiently deploy autonomous mobile robots requires intuitive frameworks for specifying a robot's task and behaviour. We present a novel approach using learning from corrections (LfC), where a user is iteratively presented with a solution to a motion planning problem. Users might have preferences about parts of a robot's environment that are suitable for robot traffic or that should be avoided as well as preferences on the control actions a robot can take. The robot is initially unaware of these preferences; thus, we ask the user to provide a correction to the presented path. We assume that the user evaluates paths based on environment and motion features. From a sequence of corrections we learn weights for these features, which are then considered by the motion planner, resulting in future paths that better fit the user's preferences. We prove completeness of our algorithm and demonstrate its performance in simulations. Thereby, we show that the learned preferences yield good results not only for a set of training tasks but also for test tasks, as well as for different types of user behaviour.

## Robotics in Agriculture and Forestry
- Visual Servoing-Based Navigation for Monitoring Row-Crop Fields

    Author: Ahmadi, Alireza | University of Bonn
    Author: Nardi, Lorenzo | University of Bonn
    Author: Chebrolu, Nived | University of Bonn
    Author: Stachniss, Cyrill | University of Bonn
 
    keyword: Robotics in Agriculture and Forestry; Visual-Based Navigation; Visual Servoing

    Abstract : Autonomous navigation is a pre-requisite for field robots to carry out precision agriculture tasks. Typically, a robot has to navigate through a whole crop field several times during a season for monitoring the plants, for applying agro-chemicals, or for performing targeted intervention actions. In this paper, we propose a framework tailored for navigation in row-crop fields by exploiting the regular crop-row structure present in the fields. Our approach uses only the images from on-board cameras without the need for performing explicit localization or maintaining a map of the field and thus can operate without expensive RTK-GPS solutions often used in agriculture automation systems. Our navigation approach allows the robot to follow the crop-rows accurately and handles the switch to the next row seamlessly within the same framework. We implemented our approach using C++ and ROS and thoroughly tested it in several simulated environments with different shapes and sizes of field. We also demonstrated the system running at frame-rate on an actual robot operating on a test row-crop field. The code and data have been published.

- Optimal Routing Schedules for Robots Operating in Aisle-Structures

    Author: Betti Sorbelli, Francesco | University of Perugia
    Author: Carpin, Stefano | University of California, Merced
    Author: Cor�, Federico | Gran Sasso Science Institute
    Author: Navarra, Alfredo | University of Perugia
    Author: Pinotti, Cristina M. | University of Perugia
 
    keyword: Robotics in Agriculture and Forestry; Inventory Management; Planning, Scheduling and Coordination

    Abstract : In this paper, we consider the Constant-cost Orienteering Problem (COP) where a robot, constrained by a limited travel budget, aims at selecting a path with the largest reward in an aisle-graph. The aisle-graph consists of a set of loosely connected rows where the robot can change lane only at either end, but not in the middle. Even when considering this special type of graphs, the orienteering problem is known to be NP-hard. We optimally solve in polynomial time two special cases, COP-FR where the robot can only traverse full rows, and COP-SC where the robot can access the rows only from one side. To solve the general COP, we then apply our special case algorithms as well as a new heuristic that suitably combines them. Despite its light computational complexity and being confined into a very limited class of paths, the optimal solutions for COP-FR turn out to be competitive even for COP in both real and synthetic scenarios. Furthermore, our new heuristic for the general case outperforms state-of-art algorithms, especially for input with highly unbalanced rewards.

- Time Optimal Motion Planning with ZMP Stability Constraint for Timber Manipulation

    Author: Song, Jiazhi | McGill University
    Author: Sharf, Inna | McGill University
 
    keyword: Robotics in Agriculture and Forestry; Motion and Path Planning; Optimization and Optimal Control

    Abstract : This paper presents a dynamic stability-constrained optimal motion planning algorithm developed for a timber harvesting machine working on rough terrain. First, the kinematics model of the machine, and the Zero Moment Point (ZMP) stability measure is presented. Then, an approach to simplify the model to gain insight and achieve a fast solution of the optimization problem is introduced. The performance and computation time of the motion plan obtained with the simplified model is compared against that obtained with the full kinematics model of the machine with the help of MATLAB simulations. The results demonstrate feasibility of fast motion planning while satisfying the dynamic stability constraint.

- Plucking Motions for Tea Harvesting Robots Using Probabilistic Movement Primitives

    Author: Motokura, Kurena | Keio University
    Author: Takahashi, Masaki | Keio University
    Author: Ewerton, Marco | Idiap Research Institute
    Author: Peters, Jan | Technische Universitét Darmstadt
 
    keyword: Robotics in Agriculture and Forestry; Learning from Demonstration; Manipulation Planning

    Abstract : This study proposes a harvesting robot capable of plucking tea leaves. In order to harvest high-quality tea, the robot is required to pluck the petiole of the leaf without cutting it using blades. To pluck the leaves, it is necessary to reproduce a complicated human hand motion of pulling while rotating. Furthermore, the rotation and pulling of the hand, and the time taken, vary greatly depending on conditions that include the maturity of the leaves, thickness of the petioles, and thickness and length of the branches. Therefore, it is necessary to determine the amount of translational and rotational movements, and the length of time of the motion, according to each situation. In this study, the complicated motion is reproduced by learning from demonstration. The condition is judged in terms of the stiffness of the branches, which is defined as the force received from the branches per unit length when the gripped leaf is slightly pulled up. Combining the learned motions probabilistically at a ratio determined by the branch stiffness, the appropriate motion is generated, even for situations where no motion is taught. We compared the motions generated by the proposed method with the motions taught by humans, and verified the effectiveness of the proposed method. It was confirmed by experiment that the proposed method can harvest high-quality tea.

- SLOAM: Semantic Lidar Odometry and Mapping for Forest Inventory

    Author: Chen, Steven W | University of Pennsylvania
    Author: Vicentim Nardari, Guilherme | University of S�o Paulo
    Author: Lee, Elijah S. | University of Pennsylvania
    Author: Qu, Chao | University of Pennsylvania
    Author: Liu, Xu | University of Pennsylvania
    Author: Romero, Roseli Ap. Francelin | Universidade De Sao Paulo
    Author: Kumar, Vijay | University of Pennsylvania, School of Engineering and Applied Sc
 
    keyword: Robotics in Agriculture and Forestry; Deep Learning in Robotics and Automation; Aerial Systems: Applications

    Abstract : This paper describes an end-to-end pipeline for tree diameter estimation based on semantic segmentation and lidar odometry and mapping. Accurate mapping of this type of environment is challenging since the ground and the trees are surrounded by leaves, thorns and vines, and the sensor typically experiences extreme motion. We propose a semantic feature based pose optimization that simultaneously refines the tree models while estimating the robot pose. The pipeline utilizes a custom virtual reality tool for labeling 3D scans that is used to train a semantic segmentation network. The masked point cloud is used to compute a trellis graph that identifies individual instances and extracts relevant features that are used by the SLAM module. We show that traditional lidar and image based methods fail in the forest environment on both Unmanned Aerial Vehicle (UAV) and hand-carry systems, while our method is more robust, scalable, and automatically generates tree diameter estimations.

- Push and Drag: An Active Obstacle Separation Method for Fruit Harvesting Robots

    Author: Xiong, Ya | Norwegian University of Life Sciences
    Author: Ge, Yuanyue | Norwegian University of Life Sciences
    Author: From, P�l Johan | Norwegian University of Life Sciences
 
    keyword: Robotics in Agriculture and Forestry; Manipulation Planning; Field Robots

    Abstract : Selectively picking a target fruit surrounded by obstacles is one of the major challenges for fruit harvesting robots. Different from traditional obstacle avoidance methods, this paper presents an active obstacle separation strategy that combines push and drag motions. The separation motion and trajectory are generated based on the 3D visual perception of the obstacle information around the target. A linear push is used to clear the obstacles from the area below the target, while a zig-zag push that contains several linear motions is proposed to push aside more dense obstacles. The zig-zag push can generate multi-directional pushes and the side-to-side motion can break the static contact force between the target and obstacles, thus helping the gripper to receive a target in more complex situations. Moreover, we propose a novel drag operation to address the issue of mis-capturing obstacles located above the target, in which the gripper drags the target to a place with fewer obstacles and then pushes back to move the obstacles aside for further detachment. Furthermore, an image processing pipeline consisting of color thresholding, object detection using deep learning and point cloud operation, is developed to implement the proposed method on a harvesting robot. Field tests show that the proposed method can improve the picking performance substantially. This method helps to enable complex clusters of fruits to be harvested with a higher success rate than conventional methods.

## Calibration and Identification

- Unified Intrinsic and Extrinsic Camera and LiDAR Calibration under Uncertainties

    Author: K�mmerle, Julius | FZI Forschungszentrum Informatik
    Author: K�hner, Tilman | FZI Forschungszentrum Informatik
 
    keyword: Calibration and Identification; Sensor Fusion

    Abstract : Many approaches for camera and LiDAR calibration are presented in literature but none of them estimates all intrinsic and extrinsic parameters simultaneously and therefore optimally in a probabilistic sense. In this work, we present a method to simultaneously estimate intrinsic and extrinsic parameters of cameras and LiDARs in a unified problem.We derive a probabilistic formulation that enables flawless integration of different measurement types without hand-tuned weights.An arbitrary number of cameras and LiDARs can be calibrated simultaneously.Measurements are not required to be synchronized.The method is designed to work with any camera model. In evaluation, we show that additional LiDAR measurements significantly improve intrinsic camera calibration.Further, we show on real data that our method achieves state-of-the-art calibration precision with high reliability.

- AC/DCC : Accurate Calibration of Dynamic Camera Clusters for Visual SLAM

    Author: Rebello, Jason | University of Toronto
    Author: Fung, Angus | University of Toronto
    Author: Waslander, Steven Lake | University of Toronto
 
    keyword: Calibration and Identification; SLAM; Visual-Based Navigation

    Abstract : In order to relate information across cameras in a Dynamic Camera Cluster (DCC), an accurate time-varying set of extrinsic calibration transformations need to be determined. Previous calibration approaches rely solely on collecting measurements from a known fiducial target which limits calibration accuracy as insufficient excitation of the gimbal is achieved. In this paper, we improve DCC calibration accuracy by collecting measurements over the entire configuration space of the gimbal and achieve a 10X improvement in pixel re-projection error. We perform a joint optimization over the calibration parameters between any number of cameras and unknown joint angles using a pose-loop error optimization approach, thereby avoiding the need for overlapping fields-of-view. We test our method in simulation and provide a calibration sensitivity analysis for different levels of camera intrinsic and joint angle noise. In addition, we provide a novel analysis of the degenerate parameters in the calibration when joint angle values are unknown, which avoids situations in which the calibration cannot be uniquely recovered. The calibration code will be made available at https://github.com/TRAILab/AC-DCC

- Analytic Plane Covariances Construction for Precise Planarity-Based Extrinsic Calibration of Camera and LiDAR

    Author: Koo, Gunhee | Korea University
    Author: Kang, Jaehyeon | Korea University
    Author: Jang, Bumchul | Korea University
    Author: Doh, Nakju | Korea University
 
    keyword: Calibration and Identification; Sensor Fusion; RGB-D Perception

    Abstract : Planarity of checkerboards is a widely used feature for extrinsic calibration of camera and LiDAR. In this study, we propose two analytically derived covariances of (i) plane parameters and (ii) plane measurement, for precise extrinsic calibration of camera and LiDAR. These covariances allow the graded approach in planar feature correspondences by exploiting the uncertainty of a set of given features in calibration. To construct plane parameter covariance, we employ the error model of 3D corner points and the analytically formulated plane parameter errors. Next, plane measurement covariance is directly derived from planar regions of point clouds using the out-of-plane errors. In simulation validation, our method is compared to an existing uncertainty-excluding method using the different number of target poses and the different levels of noise. In field experiment, we validated the applicability of the proposed analytic plane covariances for precise calibration using the basic planarity-based method and the latest planarity-and-linearity-based method.

-  A Stable Adaptive Observer for Hard-Iron and Soft-Iron Bias Calibration and Compensation for Two-Axis Magnetometers: Theory and Experimental Evaluation

    Author: Spielvogel, Andrew Robert | Johns Hopkins University
    Author: Whitcomb, Louis | The Johns Hopkins University

- Extrinsic Calibration of an Eye-In-Hand 2D LiDAR Sensor in Unstructured Environments Using ICP

    Author: Peters, Arne | Technical University of Munich
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
    Author: Schmidt, Adam | TNO - the Netherlands Organisation for Applied Scientific Resear
 
    keyword: Calibration and Identification; Range Sensing; Computer Vision for Other Robotic Applications

    Abstract : We propose a calibration method for the six degrees of freedom (DOF) extrinsic pose of a 2D laser rangefinder mounted to a robot arm. Our goal is to design a system that allows on-site re-calibration without requiring any kind of special environment or calibration objects. By moving the sensor we generate 3D scans of the surrounding area on which we run a iterative closest point (ICP) variant to estimate the missing part of the kinematic chain. With this setup we can simply scale the density and format of our 3D scan by adjusting the robot speed and trajectory, allowing us to exploit the power of a high resolution 3D scanner for a variety of tasks such as mapping, object recognition and grasp planning. Our evaluation, performed on synthetic datasets as well as from real-data shows that the presented approach provides good results both in terms of convergence on crude initial parameters as well as in the precision of the final estimate.

- Geometric Robot Dynamic Identification: A Convex Programming Approach (I)

    Author: Lee, Taeyoon | Seoul National University
    Author: Wensing, Patrick M. | University of Notre Dame
    Author: Park, Frank | Seoul National University
 
    keyword: Calibration and Identification; Dynamics

    Abstract : Recent work has shed light on the often unreliable performance of constrained least-squares estimation methods for robot mass-inertial parameter identification, particularly for high degree-of-freedom systems subject to noisy and incomplete measurements. Instead, differential geometric identification methods have proven to be significantly more accurate and robust. These methods account for the fact that the mass-inertial parameters reside in a curved Riemannian space, and allow perturbations in the mass-inertial properties to be measured in a coordinate-invariant manner. Yet, a continued drawback of existing geometric methods is that the corresponding optimization problems are inherently nonconvex, have numerous local minima, and are computationally highly intensive to solve. In this paper, we propose a convex formulation under the same coordinate-invariant Riemannian geometric framework that directly addresses these and other deficiencies of the geometric approach. Our convex formulation leads to a globally optimal solution, reduced computations, faster and more reliable convergence, and easy inclusion of additional convex constraints. The main idea behind our approach is an entropic divergence measure that allows for the convex regularization of the inertial parameter identification problem. Extensive experiments with the 3-DoF MIT Cheetah leg, the 7-DoF AMBIDEX tendon-driven arm, and a 16-link articulated human model show markedly improved robustness and generalizability.

- A Novel Calibration Method between a Camera and a 3D LiDAR with Infrared Images

    Author: Chen, Shoubin | Wuhan University
    Author: Liu, Jingbin | Wuhan University
    Author: Liang, Xinlian | Finnish Geospatial Research Institute
    Author: Zhang, Shuming | Wuhan University
    Author: Ruizhi, Chen | Wuhan University
    Author: Hyypp�, Juha | Finnish Geospatial Research Institute
 
    keyword: Calibration and Identification; Sensor Fusion; Range Sensing

    Abstract : Fusions of LiDARs (light detection and ranging) and cameras have been effectively and widely employed in the communities of autonomous vehicles, virtual reality and mobile mapping systems (MMS) for different purposes, such as localization, high definition map or simultaneous location and mapping. However, the extrinsic calibration between a camera and a 3D LiDAR is a fundamental prerequisite to guarantee its performance. Some previous methods are inaccurate, have calibration error that is several times the beam divergence, and often require special calibration objects, thereby limiting their ubiquitous use for calibration. To overcome these shortcomings, we propose a novel and high-accuracy method for the extrinsic calibration between a camera and a 3D LiDAR. Our approach relies on the infrared images from a camera with an infrared filter, and the 2D-3D corresponding points in a scene with the corners of a wall can be extracted to calculate the six extrinsic parameters. Experiments using the Velodyne VLP-16 sensor show that the method can achieve an extrinsic accuracy at the level of the beam divergence, which is fully analyzed and validated from two different aspects. Therefore, the calibration method in this paper is highly accurate, effective and does not require special complicated calibration objects; thus, it meets the requirements of practical applications.

- Online Camera-LiDAR Calibration with Sensor Semantic Information

    Author: Zhu, Yufeng | Pony.ai
    Author: Li, Chenghui | Carnegie Mellon University
    Author: Zhang, Yubo | Pony.ai
 
    keyword: Calibration and Identification; Object Detection, Segmentation and Categorization

    Abstract : As a crucial step of sensor data fusion, sensor calibration plays a vital role in many cutting-edge machine vision applications, such as autonomous vehicles and AR/VR. Existing techniques either require quite amount of manual work and complex settings, or are unrobust and prone to produce suboptimal results. In this paper, we investigate the extrinsic calibration of an RGB camera and a light detection and ranging (LiDAR) sensor, which are two of the most widely used sensors in autonomous vehicles for perceiving the outdoor environment. Specifically, we introduce an online calibration technique that automatically computes the optimal rigid motion transformation between the aforementioned two sensors and maximizes their mutual information of perceived data, without the need of tuning environment settings. By formulating the calibration as an optimization problem with a novel calibration quality metric based on semantic features, we successfully and robustly align pairs of temporally synchronized camera and LiDAR frames in real time. Demonstrated on several autonomous driving tasks, our method outperforms state-of-the-art edge feature based auto-calibration approaches in terms of robustness and accuracy.

- Precise 3D Calibration of Wafer Handling Robot by Visual Detection and Tracking of Elliptic-Shape Wafers

    Author: Wang, Zining | University of California, Berkeley
    Author: Tomizuka, Masayoshi | University of California
 
    keyword: Calibration and Identification; Visual Tracking; Computer Vision for Automation

    Abstract : This work provides a framework for the 3D calibration of wafers and a wafer handling robot by monocular. The proposed method precisely reconstructs the 3D pose of wafers from a set of images captured by the camera mounted on the robot. Besides, it calibrates the robot kinematics simultaneously. A robust ellipse detection and tracking algorithm based on the edge arcs is developed to recognize wafers among images. Then a joint optimization is constructed from pose graph to simultaneously solve the 3D poses of wafers and other calibration parameters of the robot-camera system. The proposed tracking method is able to associate multiple incomplete elliptic segments using a GMM-based registration model. And it is point-based where no feature descriptor is required. The proposed 3D pose optimization incorporates shape constraint and is more accurate than the point-wise reconstruction of classic bundle adjustment methods.

- Globally Optimal Relative Pose Estimation for Camera on a Selfie Stick

    Author: Joo, Kyungdon | Korea Advanced Institute of Science and Technology (KAIST)
    Author: Li, Hongdong | Australian National University and NICTA
    Author: Oh, Tae-Hyun | MIT
    Author: Bok, Yunsu | Electronics and Telecommunication Research Institute (ETRI)
    Author: Kweon, In So | KAIST
 
    keyword: Calibration and Identification; Computer Vision for Other Robotic Applications; SLAM

    Abstract : Taking selfies has become a photographic trend nowadays. We envision the emergence of the �video selfie� capturing a short continuous video clip (or burst photography) of the user, themselves. A selfie stick is usually used, whereby a camera is mounted on a stick for taking selfie photos. In this scenario, we observe that the camera typically goes through a special trajectory along a sphere surface. Motivated by this observation, in this work, we propose an efficient and globally optimal relative camera pose estimation between a pair of two images captured by a camera mounted on a selfie stick. We exploit the special geometric structure of the camera motion constrained by a selfie stick and define its motion as spherical joint motion. By the new parametrization and calibration scheme, we show that the pose estimation problem can be reduced to a 3-DoF (degrees of freedom) search problem, instead of a generic 6-DoF problem. This allows us to derive a fast branch-and-bound global optimization, which guarantees a global optimum. Thereby, we achieve efficient and robust estimation even in the presence of outliers. By experiments on both synthetic and real-world data, we validate the performance as well as the guaranteed optimality of the proposed method.

- Online Calibration of Exterior Orientations of a Vehicle-Mounted Surround-View Camera System

    Author: Ouyang, Zhanpeng | Shanghaitech University
    Author: Hu, Lan | ShanghaiTech University
    Author: Lu, Yukan | Motovis
    Author: Wang, Zhirui | Motovis
    Author: Peng, Xin | ShanghaiTech University
    Author: Kneip, Laurent | ShanghaiTech
 
    keyword: Calibration and Identification; Omnidirectional Vision; Localization

    Abstract : The increasing availability of surround-view camera systems in passenger vehicles motivates their use as an exterior perception modality for intelligent vehicle behaviour. An important problem within this context is the extrinsic calibration between the cameras, which is challenging due to the often reduced overlap between the fields of view of neighbouring views. Our work is motivated by two insights. First, we argue that the accuracy of vision-based vehicle motion estimation depends crucially on the quality of exterior orientation calibration, while design parameters for camera positions typically provide sufficient accuracy. Second, we demonstrate how planar vehicle motion related direction vectors can be used to accurately identify individual camera-to-vehicle rotations, which are more useful than the commonly and tediously derived camera-to-camera transformations. We present a complete and highly practicable online optimisation strategy to obtain the exterior orientation parameters and conclude with successful tests on simulated, indoor, and large-scale outdoor experiments.

- Learning Camera Miscalibration Detection

    Author: Cramariuc, Andrei | ETHZ
    Author: Petrov, Aleksandar | ETH Zurich
    Author: Suri, Rohit | ETH Zurich
    Author: Mittal, Mayank | ETH
    Author: Siegwart, Roland | ETH Zurich
    Author: Cadena Lerma, Cesar | ETH Zurich
 
    keyword: Calibration and Identification; Deep Learning in Robotics and Automation; Computer Vision for Other Robotic Applications

    Abstract : Self-diagnosis and self-repair are some of the key challenges in deploying robotic platforms for long-term real-world applications. One of the issues that can occur to a robot is miscalibration of its sensors due to aging, environmental transients, or external disturbances. Precise calibration lies at the core of a variety of applications, due to the need to accurately perceive the world. However, while a lot of work has focused on calibrating the sensors, not much has been done towards identifying when a sensor needs to be recalibrated. This paper focuses on a data-driven approach to learn the detection of miscalibration in vision sensors, specifically RGB cameras. Our contributions include a proposed miscalibration metric for RGB cameras and a novel semi-synthetic dataset generation pipeline based on this metric. Additionally, by training a deep convolutional neural network, we demonstrate the effectiveness of our pipeline to identify whether a recalibration of the camera's intrinsic parameters is required or not. The code is available at http://github.com/ethz-asl/camera_miscalib_detection.

## Industrial Robots

- An End-Effector Wrist Module for the Kinematically Redundant Manipulation of Arm-Type Robots

    Author: Chang, Yu-Hsiang | National Cheng Kung University
    Author: Liu, Yen-Chun | National Cheng Kung University
    Author: Lan, Chao-Chieh | National Cheng Kung University
 
    keyword: Kinematics; Dexterous Manipulation; Mechanism Design

    Abstract : Industrial arm-type robots have multiple degrees-of-freedom (DoFs) and high dexterity but the use of the roll-pitch-roll wrist configuration yields singularities inside the reachable workspace. Excessive joint velocities will occur when encountering these singularities. Arm-type robots currently don�t have enough dexterity to move the end-effector path away from the wrist singularities. Robots with redundant DoFs can be used to provide additional dexterity required to avoid the singularities and reduce the excessive joint velocity. An end-effector wrist module is proposed in this paper to provide two redundant DoFs when interfaced with an existing 6-DoF robot. The new 8-DoF robot has a compact roll-pitch-yaw wrist that has no singularities inside the reachable workspace. The highly redundant robot can also be used to avoid collisions in various directions. Path tracking simulation examples are provided to show the advantages of the proposed design when compared with existing redundant or nonredundant robots. We expect that this module can serve as a cost-effective solution in applications where singularity-free motion or collision-free motion is required.

- Robust Path Following of the Tractor-Trailers System in GPS-Denied Environments

    Author: Zhou, Shunbo | The Chinese University of Hong Kong
    Author: Zhao, Hongchao | The Chinese University of Hong Kong
    Author: Chen, Wen | The Chinese University of Hong Kong
    Author: Miao, Zhiqiang | Hunan University
    Author: Liu, Zhe | The Chinese University of Hong Kong
    Author: Wang, Hesheng | Shanghai Jiao Tong University
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Industrial Robots; Visual Servoing

    Abstract : This paper reports a general path following framework for the tractor-trailers system in Global Positioning System (GPS)-denied environments. Compared to existing methods, this approach prioritizes a robust, cost-optimized, and easy-to-implement solution. First, to achieve accurate path following, a force sensor is subtly introduced to capture the impact of an arbitrary number of trailers and varying payloads on the tractor dynamics. A robust controller is then designed on the basis of a newly derived tractor dynamic model and the lateral force compensation. Second, a novel visual-inertial estimator, which explicitly considers the nonlinear velocity dynamics, is developed to allow real-time translational velocity and position estimation for dynamic feedback control. It is rigorously proved by the Lyapunov theory that the stability of the proposed estimation and control system is guaranteed. Full-scale experiments are conducted to demonstrate the feasibility of the approach.

- Online Trajectory Planning for an Industrial Tractor Towing Multiple Full Trailers

    Author: Zhao, Hongchao | The Chinese University of Hong Kong
    Author: Chen, Wen | The Chinese University of Hong Kong
    Author: Zhou, Shunbo | The Chinese University of Hong Kong
    Author: Liu, Zhe | The Chinese University of Hong Kong
    Author: Zheng, Fan | The Chinese University of Hong Kong
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Industrial Robots; Motion and Path Planning

    Abstract : This paper presents a novel solution for online trajectory planning of a full-size tractor-trailers vehicle composed of a car-like tractor and arbitrary number of passive full trailers. The motion planning problem for such systems was rarely addressed due to the complex nonlinear dynamics. A simulation-based prediction method is proposed to easily handle the complicated nonlinear dynamics and efficiently generate the obstacle-free and dynamically feasible trajectories. The vehicle dynamics model and a two-layer controller are used in the prediction. Implementation results on the real-world full-size industrial tractor-trailers vehicle are presented to validate the performance of the proposed methods.

- Towards Efficient Human Robot Collaboration with Robust Plan Recognition and Trajectory Prediction

    Author: Cheng, Yujiao | University of California, Berkeley
    Author: Sun, Liting | University of California, Berkeley
    Author: Liu, Changliu | Carnegie Mellon University
    Author: Tomizuka, Masayoshi | University of California
 
    keyword: Industrial Robots; Recognition; Cognitive Human-Robot Interaction

    Abstract : Human-robot collaboration (HRC) is becoming increasingly important as the paradigm of manufacturing is shifting from mass production to mass customization. The introduction of HRC can significantly improve the flexibility and intelligence of automation. To efficiently finish tasks in HRC systems, the robots need to not only predict the future movements of human, but also more high-level plans, i.e., the sequence of actions to finish the tasks. However, due to the stochastic and time-varying nature of human collaborators, it is quite challenging for the robot to efficiently and accurately identify such task plans and respond in a safe manner. To address this challenge, we propose an integrated human-robot collaboration framework. Both plan recognition and trajectory prediction modules are included for the generation of safe and efficient robotic motions. Such a framework enables the robots to perceive, predict and adapt their actions to the human's work plan and intelligently avoid collisions with the human. Moreover, by explicitly leveraging the hierarchical relationship between plans and trajectories, more robust plan recognition performance can be achieved. Physical experiments were conducted on an industrial robot to verify the proposed framework. The results show that the proposed framework could accurately recognize the human workers' plans and thus significantly improve the time efficiency of the HRC team even in the presence of motion classification noises.

- Collaborative Human-Robot Framework for Delicate Sanding of Complex Shape Surface

    Author: Maric, Bruno | Univeristy of Zagreb, Faculty of Electrical Engineering and Comp
    Author: Mutka, Alan | Faculty of EE&amp;C
    Author: Orsag, Matko | University of Zagreb, Faculty of Electrical Engineering and Comp
 
    keyword: Industrial Robots; Contact Modeling; Force Control

    Abstract : This letter presents a collaborative human-robot framework for delicate sanding of complex shape surfaces. Delicate sanding is performed using a standard industrial manipulator, equipped with the force/torque sensor and specially designed compliant control algorithm. Together with the compliant control, we discuss trajectory planning and safety problem of such an approach. The experience of the human workers is exploited trough the intuitive framework and applied to plan the trajectories for the robot. The flexibility and the reliability of the proposed framework is tested in real working conditions in the factory.

- External Force Estimation for Industrial Robots with Flexible Joints

    Author: Lin, Yang | Huazhong University of Science and Technology
    Author: Zhao, Huan | Huazhong University of Science and Technology
    Author: Ding, Han | Huazhong University of Science and Technology
 
    keyword: Industrial Robots; Dynamics

    Abstract : With no force sensors, estimating forces for robotic manipulation has gained a lot of attention. However, for industrial robots with harmonic drives (flexible joints), deviations between joint and motor positions inevitably deteriorate the force estimation performance due to the lack of encoders on the joint side. To this end, this paper presents a method to estimate not only joint states but also external forces. The method includes an extended disturbance state observer (DSO) and a proposed task-oriented disturbance modeling (TDM). First, a robust DSO is extended to robots with flexible joints aiming to estimate joint states and disturbances. Then, due to the observed disturbances including (possibly) external forces and also uncertainties such as measurement noises and model errors of the robot dynamics, a learning part is proposed to model the task-oriented disturbances during no-contact motion and to improve the effect of uncertainties on the force estimation performance. Finally, when the robot comes in contact with the environment, external forces are estimated as the differences between the modeled (no-contact) disturbances and the real-time observed disturbances. Experimental results obtained on a six-degrees-of-freedom (6-DOF) industrial robot with flexible joints, show the feasibility and effectiveness of the proposed method.

- Robotic General Parts Feeder: Bin-Picking, Regrasping, and Kitting

    Author: Domae, Yukiyasu | The National Institute of Advanced Industrial Science and Techno
    Author: Noda, Akio | Osaka Institute of Technology
    Author: Nagatani, Tatsuya | Mitsubishi Electric Corp
    Author: Wan, Weiwei | Osaka University
 
    keyword: Industrial Robots; Manipulation Planning; Multi-Robot Systems

    Abstract : Parts feeding of multiple objects is an unsolved problem which should be tackled by robotics. We propose a systematic approach to make general parts feeders for various shape's rigid parts. We divided the problem into three subcomponents: bin-picking, regrasping, and kitting. The subcomponents are designed to solve the large systematic problem by using a coarse-to-fine approach. Multiple robot arms are connected as a pipe line system to solve each subcomponent. In addition, we proposed a semi-automatic method for teaching of regrasping by multiple robots. Thus the robot systems can supply multiple types of parts. In our experiment by using eleven types of industrial parts which has various shapes, the Mean Picks Per Hour (MPPH) of the system becomes about 350. The score is faster than the state-of-the-art robotic bin- picking system. In addition, lead time by the proposed method for changing parts is less than by a combination of traditional parts feeders or a manual labor.

- Planning, Learning and Reasoning Framework for Robot Truck Unloading

    Author: Islam, Fahad | National University of Sciences and Technology
    Author: Vemula, Anirudh | Carnegie Mellon University
    Author: Kim, Sung-Kyun | NASA Jet Propulsion Laboratory, Caltech
    Author: Dornbush, Andrew | Carnegie Mellon University
    Author: Salzman, Oren | Technion
    Author: Likhachev, Maxim | Carnegie Mellon University
 
    keyword: Factory Automation; Industrial Robots; Task Planning

    Abstract : We consider the task of autonomously unloading boxes from trucks using an industrial manipulator robot. There are multiple challenges that arise: (1) real-time motion planning for a complex robotic system carrying two articulated mechanisms, an arm and a scooper, (2) decision-making in terms of what action to execute next given imperfect information about boxes such as their masses, (3) accounting for the sequential nature of the problem where current actions affect future state of the boxes, and (4) real-time execution that interleaves high-level decision-making with lower level motion planning. In this work, we propose a planning, learning, and reasoning framework to tackle these challenges, and describe its components including motion planning, belief space planning for offline learning, online decision-making based on offline learning, and an execution module to combine decision-making with motion planning. We analyze the performance of the framework on real-world scenarios. In particular, motion planning and execution modules are evaluated in simulation and on a real robot, while offline learning and online decision-making are evaluated in simulated real-world scenarios.

- Evaluation of Perception Latencies in a Human-Robot Collaborative Environment

    Author: Aalerud, Atle | University of Agder
    Author: Hovland, Geir | University of Agder
 
    keyword: Industrial Robots; Computer Vision for Other Robotic Applications; RGB-D Perception

    Abstract : The latency in vision-based sensor systems used in human-robot collaborative environments is an important safety parameter which in most cases has been neglected by researchers. The main reason for this neglect is the lack of an accurate ground-truth sensor system with a minimal delay to benchmark the vision-sensors against. In this paper the latencies of 3D vision-based sensors are experimentally evaluated and analyzed using an accurate laser-tracker system which communicates on a dedicated EtherCAT channel with minimal delay. The experimental results in the paper demonstrate that the latency in the vision-based sensor system is many orders higher than the latency in the control and actuation system.

- Assembly of Randomly Placed Parts Realized by Using Only One Robot Arm with a General Parallel-Jaw Gripper

    Author: Zhao, Jie | Harbin Institute of Technology, Shenzhen
    Author: Wang, Xiaoman | Harbin Institute of Technology, Shenzhen
    Author: Wang, Shengfan | Harbin Institute of Technology
    Author: Jiang, Xin | Harbin Institute of Technology, Shenzhen
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Factory Automation; Dexterous Manipulation; Grippers and Other End-Effectors

    Abstract : In industry assembly lines, parts feeding machines are widely employed as the prologue of the whole procedure. They play the role of sorting the parts randomly placed in bins to the state with specified pose.	With the help of the parts feeding machines, the subsequent assembly processes by robot arm can always start from the same condition. Thus it is expected that function of parting feeding machine and the robotic assembly can be integrated with one robot arm. This scheme can provide great flexibility and can also contribute to reduce the cost. The difficulties involved in this scheme lie in the fact that in the part feeding phase, the pose of the part after grasping may be not proper for the subsequent assembly. Sometimes it can not even guarantee a stable grasp. In this paper, we proposed a method to integrate parts feeding and assembly within one robot arm.	This proposal utilizes a specially designed gripper tip mounted on the jaws of a two-fingered gripper. With the modified gripper, in-hand manipulation of the grasped object is realized, which can ensure the control of the orientation and offset position of the grasped object. The proposal in this paper is verified by a simulated assembly in which a robot arm completed the assembly process including parts picking from bin and a subsequent peg-in-hole assembly.

- Drive-Based Vibration Damping Control for Robot Machining

    Author: Mesmer, Patrick | University of Stuttgart
    Author: Neubauer, Michael | University of Stuttgart
    Author: Lechler, Armin | University Stuttgart
    Author: Verl, Alexander | University of Stuttgart
 
    keyword: Industrial Robots; Motion Control

    Abstract : The objective of this letter is to propose a drive-based control method for damping joint vibrations in order to improve the machining quality and the productivity of robot machining. The machining quality is significantly determined by the transfer behavior of the gearboxes installed in the robot joints. This letter presents a novel approach for drive-based damping of these vibrations and thus for increasing the dynamic path accuracy on the basis of secondary encoders. The approach represents an enhancement of the independent joint control, which is widely used in industry. The simulation as well as the experimental results on a KUKA KR210-2 demonstrate the effectiveness of the presented control method for damping gearbox vibrations and increasing the dynamic path accuracy of industrial robot manipulators.

- Toward Fast and Optimal Robotic Pick-And-Place on a Moving Conveyor

    Author: Han, Shuai D. | Rutgers University
    Author: Feng, Si Wei | Rutgers University
    Author: Yu, Jingjin | Rutgers University
 
    keyword: Factory Automation; Planning, Scheduling and Coordination; Industrial Robots

    Abstract : Robotic pick-and-place (PNP) operations on moving conveyors find a wide range of industrial applications. In practice, simple greedy heuristics (e.g., prioritization based on the time to process a single object) are applied that achieve reasonable efficiency. We show analytically that, under a simplified telescoping robot model, these greedy approaches do not ensure time optimality of PNP operations. To address the shortcomings of classical solutions, we develop algorithms that compute optimal object picking sequences for a predetermined finite horizon. Employing dynamic programming techniques and additional heuristics, our methods scale to up to tens to hundreds of objects. In particular, the fast algorithms we develop come with running time guarantees, making them suitable for real-time PNP applications demanding high throughput. Extensive evaluation of our algorithmic solution over dominant industrial PNP robots used in real-world applications, i.e., Delta robots and Selective Compliance Assembly Robot Arm (SCARA) robots, shows that a typical efficiency gain of around 10%-40% over greedy approaches can be realized.

## Biomimetics

- A Bio-Inspired Transportation Network for Scalable Swarm Foraging

    Author: Lu, Qi | University of New Mexico
    Author: Fricke, George Matthew | The University of New Mexico
    Author: Tsuno, Takaya | Mie University
    Author: Moses, Melanie | University of New Mexico
 
    keyword: Collision Avoidance; Swarms; Biologically-Inspired Robots

    Abstract : Scalability is a significant challenge for robot swarms. Generally, larger groups of cooperating robots produce more inter-robot collisions, and in swarm robot foraging, larger search arenas result in larger travel costs. This paper demonstrates a scale-invariant swarm foraging algorithm that ensures that each robot finds and delivers resources to a central collection zone at the same rate regardless of the size of the swarm or the search area. Dispersed mobile depots aggregate locally foraged resources and transport them to a central place via a hierarchical branching transportation network. This approach is inspired by ubiquitous fractal branching networks such as tree branches and mammal cardiovascular networks that deliver resources to cells and determine the scale and pace of life. We demonstrate that biological scaling laws predict how quickly robots forage in simulations of up to thousands of robots searching over thousands of square meters. We then use biological scaling claims to determine the capacity of depot robots in order to overcome scaling constraints and produce scale-invariant robot swarms. We verify the claims for large swarms in simulation and implement a simple depot design in simulation and hardware.

- Stance Control Inspired by Cerebellum Stabilizes Reflex-Based Locomotion on HyQ Robot

    Author: Urbain, Gabriel | Ghent University
    Author: Barasuol, Victor | Istituto Italiano Di Tecnologia
    Author: Semini, Claudio | Istituto Italiano Di Tecnologia
    Author: Dambre, Joni | Ghent University
    Author: Wyffels, Francis | Ghent University
 
    keyword: Neurorobotics; Legged Robots; Neural and Fuzzy Control

    Abstract : Advances in legged robotics are strongly rooted in animal observations. A clear illustration of this claim is the generalization of Central Pattern Generators (CPG), first identified in the cat spinal cord, to generate cyclic motion in robotic locomotion. Despite a global endorsement of this model, physiological and functional experiments in mammals have also indicated the presence of descending signals from the cerebellum, and reflex feedback from the lower limb sensory cells, that closely interact with CPGs. To this day, these interactions are not fully understood. In some studies, it was demonstrated that pure reflex-based locomotion in the absence of oscillatory signals could be achieved in realistic musculoskeletal simulation models or small compliant quadruped robots. At the same time, biological evidence has attested the functional role of the cerebellum for predictive control of balance and stance within mammals. In this paper, we promote both approaches and successfully apply reflex-based dynamic locomotion, coupled with a balance and gravity compensation mechanism, on the state-of-art HyQ robot. We discuss the importance of this stability module to ensure a correct foot lift-off and maintain a reliable gait. The robotic platform is further used to test two different architectural hypotheses inspired by the cerebellum. An analysis of experimental results demonstrates that the most biologically plausible alternative also leads to better results for robust locomotion.

- Error Estimation and Correction in a Spiking Neural Network for Map Formation in Neuromorphic Hardware

    Author: Kreiser, Raphaela | Institute of Neuroinformatics, Univeristy Zurich and ETH Zurich
    Author: Waibel, Gabriel Guenter | ETH Zurich
    Author: Armengol, Nuria | ETH Zurich
    Author: Renner, Alpha | Institute of Neuroinformatics, University of Zurich and ETH Zuri
    Author: Sandamirskaya, Yulia | University and ETH Zurich
 
    keyword: Neurorobotics; Biologically-Inspired Robots; Biomimetics

    Abstract : Neuromorphic hardware offers computing platforms for the efficient implementation of spiking neural networks (SNNs) that can be used for robot control. Here, we present such an SNN on a neuromorphic chip that solves a number of tasks related to simultaneous localization and mapping (SLAM): forming a map of an unknown environment and, at the same time, estimating the robot's pose. In particular, we present an SNN mechanism to detect and estimate errors when the robot revisits a known landmark and updates both the map and the path integration speed to reduce the error. The whole system is fully realized in a neuromorphic device, showing the feasibility of a purely SNN-based SLAM, which could be efficiently implemented in a small form-factor neuromorphic chip.

- A Hybrid Compact Neural Architecture for Visual Place Recognition

    Author: Chanc�n Le�n, Marvin Aldo | Queensland University of Technology
    Author: Hernandez-Nunez, Luis | Harvard University
    Author: Narendra, Ajay | Macquarie University
    Author: Barron, Andrew | Macquarie University
    Author: Milford, Michael J | Queensland University of Technology
 
    keyword: Biomimetics; Localization; Visual-Based Navigation

    Abstract : State-of-the-art algorithms for visual place recognition, and related visual navigation systems, can be broadly split into two categories: computer-science-oriented models including deep learning or image retrieval-based techniques with minimal biological plausibility, and neuroscience-oriented dynamical networks that model temporal properties underlying spatial navigation in the brain. In this letter, we propose a new compact and high-performing place recognition model that bridges this divide for the first time. Our approach comprises two key neural models of these categories: (1) FlyNet, a compact, sparse two-layer neural network inspired by brain architectures of fruit flies, Drosophila melanogaster, and (2) a one-dimensional continuous attractor neural network (CANN). The resulting FlyNet+CANN network incorporates the compact pattern recognition capabilities of our FlyNet model with the powerful temporal filtering capabilities of an equally compact CANN, replicating entirely in a hybrid neural implementation the functionality that yields high performance in algorithmic localization approaches like SeqSLAM. We evaluate our model, and compare it to three state-of-the-art methods, on two benchmark real-world datasets with small viewpoint variations and extreme environmental changes - achieving 87% AUC results under day to night transitions compared to 60% for Multi-Process Fusion, 46% for LoST-X and 1% for SeqSLAM, while being 6.5, 310, and 1.5 times faster, respectively.

- Musculoskeletal AutoEncoder: A Unified Online Acquisition Method of Intersensory Networks for State Estimation, Control, and Simulation of Musculoskeletal Humanoids

    Author: Kawaharazuka, Kento | The University of Tokyo
    Author: Tsuzuki, Kei | University of Tokyo
    Author: Onitsuka, Moritaka | The University of Tokyo
    Author: Asano, Yuki | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Kawasaki, Koji | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Biomimetics; Humanoid Robots; Tendon/Wire Mechanism

    Abstract : While the musculoskeletal humanoid has various biomimetic benefits, the modeling of its complex structure is difficult, and many learning-based systems have been developed so far. There are various methods, such as control methods using acquired relationships between joints and muscles represented by a data table or neural network, and state estimation methods using Extended Kalman Filter or table search. In this study, we construct a Musculoskeletal AutoEncoder representing the relationship among joint angles, muscle tensions, and muscle lengths, and propose a unified method of state estimation, control, and simulation of musculoskeletal humanoids using it. By updating the Musculoskeletal AutoEncoder online using the actual robot sensor information, we can continuously conduct more accurate state estimation, control, and simulation than before the online learning. We conducted several experiments using the musculoskeletal humanoid Musashi, and verified the effectiveness of this study.

- Snake-Inspired Kirigami Skin for Lateral Undulation of a Soft Snake Robot

    Author: Branyan, Callie | Oregon State University
    Author: Hatton, Ross | Oregon State University
    Author: Menguc, Yigit | Facebook Reality Labs
 
    keyword: Soft Robot Materials and Design; Biologically-Inspired Robots; Flexible Robots

    Abstract : Frictional anisotropy, as produced by the directionality of scales in snake skin, is necessary to propel snakes across flat, hard surfaces. This work illustrates the design, fabrication, and testing of a snake-inspired skin based on kirigami techniques that, when attached to a soft snake robot, improves the robot's locomotion capabilities when implementing a lateral undulation gait. Examination of snake scales in nature informed the shape and texture of the synthetic scales, which are activated through the buckling of kirigami lattices. Biological snakes have microornamentation on their scales, which is replicated by scoring ridges into the plastic skin. This microornamentation contributes to the lateral resistance necessary for lateral undulation. The skin's frictional properties were experimentally determined, as were their contributions to the locomotion of the robot across a flat, hard, textured surface. Contributions to locomotion from scale profile geometry, scale microornamentation, and scale angle of attack were identified. The range of longitudinal COF ratios was 1.0 to 3.0 and the range of lateral COF ratios was 0.9 to 3.3. The highest performing skin was the triangular scale profile with microornamentation, producing a velocity of 6 mm/s (0.03 BL/s) which is an increase of 335% over the robot with no skin when activated to maximum achievable curvature.

- Bio-Inspired Distance Estimation Using the Self-Induced Acoustic Signature of a Motor-Propeller System

    Author: Calkins, Luke | Duke University
    Author: Lingevitch, Joseph | U.S. Naval Research Laboratory
    Author: McGuire, Loy | U.S. Naval Research Laboratory
    Author: Geder, Jason | U.S. Naval Research Laboratory
    Author: Kelly, Matthew | U.S. Naval Research Laboratory
    Author: Zavlanos, Michael M. | Duke University
    Author: Sofge, Donald | Naval Research Laboratory
    Author: Lofaro, Daniel | George Mason University
 
    keyword: Biomimetics; Range Sensing; Robot Audition

    Abstract : In this paper we propose an algorithm to actively control the distance of a motor-propeller system (MPS) to a large obstacle using data from a single microphone. The method is based upon a broadband constructive/destructive interference pattern across the audible frequency band that is present when the MPS is near an obstacle. By taking the difference between the power spectrum in the obstacle-free case and the spectrum when recording near an obstacle, a broadband oscillation with respect to frequency is revealed. The frequency of this oscillation is linearly-related to the distance from the microphone to the wall. We present both static and dynamic experiments showcasing the ability of the proposed method to estimate the distance to a wall as well as actively control it.

- A Bio-Inspired 3-DOF Light-Weight Manipulator with Tensegrity X-Joints

    Author: Fasquelle, Benjamin | Ecole Centrale De Nantes, LS2N
    Author: Furet, Matthieu | Laboratoire Des Sciences Du Num�rique De Nantes (LS2N)
    Author: Khanna, Parag | Laboratoire Des Sciences Du Num�rique De Nantes, École Centrale
    Author: Chablat, Damien | Laboratoire Des Sciences Du Num�rique De Nantes
    Author: Chevallereau, Christine | CNRS
    Author: Wenger, Philippe | Ecole Centrale De Nantes
 
    keyword: Biomimetics; Motion Control of Manipulators; Tendon/Wire Mechanism

    Abstract : This paper proposes a new kind of light-weight manipulators suitable for safe interactions. The proposed manipulators use anti-parallelogram joints in series, referred to as X-joints. Each X-joint is remotely actuated with cables and springs in parallel, thus realizing a tensegrity one-degree-of-freedom mechanism. As compared to manipulators built with simple revolute joints in series, manipulators with tensegrity X-joint offer a number of advantages, such as an intrinsic stability, variable stiffness and lower inertia. This new design was inspired by the musculosleketon architecture of the bird neck that is known to have remarkable features such as a high dexterity. The paper analyzes in detail the kinetostatics of a X-joint and proposes a 3-degree-of-freedom manipulator made of three such joints in series. Both simulation results and experiment results conducted on a test-bed prototype are presented and discussed.

- The Lobster-Inspired Antagonistic Actuation Mechanism towards a Bending Module

    Author: Chen, Yaohui | Monash University
    Author: Chung, Hoam | Monash University
    Author: Chen, Bernard | Monash University
    Author: , Baoyinjiya | Monash University
    Author: Sun, Yonghang | Monash Univerity
 
    keyword: Biomimetics; Soft Sensors and Actuators

    Abstract : This paper describes a new type of bending module inspired, in part, by the musculoskeletal structure of the lobster leg joint. The bending module proposed combines enhanced torque output, reconfigurability in assembling, safe compliant actuation, and accurate control on its mechanical performance. In this module, antagonistic soft chambers are enveloped by exoskeleton shells, and the bending angle and the stiffness can be independently adjusted by controlling the input pressure in the two chambers. Theoretical models are developed to characterize the relationships between the input pressure, bending angle, and stiffness, and a controller for angle control and stiffness tuning is constructed with experimental validation. The fabricated module can reach the maximum torque output of 109.7 Ncdotmm under 40 kPa and the stiffness range from 40 to 220 Ncdotmm / rad, demonstrating its capacity to fulfill both safe interactions and forceful tasks.

- Emulating Duration and Curvature of Coral Snake Anti-Predator Thrashing Behaviors Using a Soft-Robotic Platform

    Author: Danforth, Shannon | University of Michigan
    Author: Kohler, Margaret | University of Michigan
    Author: Bruder, Daniel | University of Michigan
    Author: Davis Rabosky, Alison | University of Michigan
    Author: Kota, Sridhar | University of Michigan
    Author: Vasudevan, Ram | University of Michigan
    Author: Moore, Talia | University of Michigan
 
    keyword: Biomimetics; Soft Robot Applications; Soft Robot Materials and Design

    Abstract : This paper presents a soft-robotic platform for exploring the ecological relevance of non-locomotory movements via animal-robot interactions. Coral snakes (genus <i>Micrurus</i>) and their mimics use vigorous, non-locomotory, and arrhythmic thrashing to deter predation. There is variation across snake species in the duration and curvature of anti-predator thrashes, and it is unclear how these aspects of motion interact to contribute to snake survival. This paper applies soft robots composed of fiber-reinforced elastomeric enclosures (FREEs) to emulate the anti-predator behaviors of three genera of snake. Curvature and duration of motion are estimated for both live snakes and robots, providing a quantitative assessment of the robots' ability to emulate snake poses. The curvature values of the fabricated soft-robotic head, midsection, and tail segments are found to overlap with those exhibited by live snakes. Soft robot motion durations were less than or equal to those of snakes for all three genera. Additionally, combinations of segments were selected to emulate three specific snake genera with distinct anti-predatory behavior, producing curvature values that aligned well with live snake observations.

- Directional Mechanical Impedance of the Human Ankle During Standing with Active Muscles

    Author: Aramizo Ribeiro, Guilherme | Purdue University
    Author: Knop, Lauren | Michigan Technological University
    Author: Rastgaar, Mo | Purdue University
 
    keyword: Biomimetics; Kinematics; Neurorobotics

    Abstract : The directional mechanical impedance of the human ankle was identified from subjects in a standing posture with varying levels of muscle activity. The impedance modeled the different torque responses to angle perturbations about different axes of rotation. This work proposed a novel impedance model that incorporated the coupling between multiple degrees of freedom of the ankle and was validated theoretically and experimentally. The reconstructed torque had an average variance accounted above 94% across twelve subjects. In addition, the impedance varied between and within trials and this variation was explained by changes in the ankle states, i.e., the ankle angle, torque, and muscle activities. These results have implications in the design of new prostheses controllers and the understanding of the human ankle function.

- Insect�Computer Hybrid Robot Achieves a Walking Gait Rarely Seen in Nature by Replacing the Anisotropic Natural Leg Spines with Isotropic Artificial Leg Spines (I)
 
    Author: Cao, Feng | Nanyang Technological University
    Author: Sato, Hirotaka | Nanyang Technological University
 
    keyword: Biomimetics; Biologically-Inspired Robots; Legged Robots

    Abstract : This paper demonstrates that our developed beetle-computer hybrid legged robot achieves backward walk which is impossible by intact beetles themselves in nature. Judging from the curvature of the natural leg spine, we hypothesized that the natural spine has anisotropic function: the natural spine would provide foot traction only in forward walk but not in backward. The hypothesis was verified as beetles hardly walk backward due to often slips. We then designed an artificial leg spine which isotropically functions in walk: the foot traction was increased and slip-less walk was achieved in both backward and forward walk with the artificial spines being implanted into the beetle legs. For these investigations, a wireless communication device, or "backpack", was mounted and wired to a live beetle for electrically stimulating leg muscles to remotely modulate leg motions and to perform the forward and backward walking on demand. Overall, the beetle hybrid robot revealed the anisotropic function of the natural leg spine and also achieved the backward walk which the intact beetle cannot intrinsically perform.

## Robust/Adaptive Control of Robotic Systems

- Adaptive Visual Shock Absorber with Visual-Based Maxwell Model Using Magnetic Gear

    Author: Tanaka, Satoshi | The University of Tokyo
    Author: Koyama, Keisuke | University of Tokyo
    Author: Senoo, Taku | University of Tokyo
    Author: Ishikawa, Masatoshi | University of Tokyo
 
    keyword: Robust/Adaptive Control of Robotic Systems; Force Control; Mechanism Design

    Abstract : In this study, a visual shock absorber capable of adapting to free-fall objects with various weights and speeds is designed and realized. The key element is a magnetic gear to passively absorb shock in the moment of contact, which is difficult for traditional feedback control to deal with. The magnetic gear allows the seamless transfer of control from the non-contact state to the contact state. 1000 Hz high-speed visual object tracking is used for preparation with position and velocity control in the object non-contact state. In the moment of object contact, the high backdrivability of the magnetic gear response by hardware provides high responsiveness to external force. After the impact, the plastic deformation control of a parallel-expressed Maxwell model handles the contact state.

- Slip-Based Nonlinear Recursive Backstepping Path Following Controller for Autonomous Ground Vehicles

    Author: Xin, Ming | Inceptio Technology
    Author: Zhang, Kai | Inceptio Technology, Inc
    Author: Lackner, David | Inceptio Technology
    Author: Minor, Mark | University of Utah
 
    keyword: Robust/Adaptive Control of Robotic Systems; Dynamics; Wheeled Robots

    Abstract :  Path following accuracy and error convergence with graceful motion in vehicle steering control is challenging due to the competing nature of these requirements, especially across a range of operating speeds. This work is founded upon slip-based kinematic and dynamic models, which allow derivation of controllers considering error due to sideslip and the mapping between steering commands and graceful lateral motion. A novel recursive backstepping steering controller is proposed that better couples yaw-rate based path following commands to steering angle and rate. Observer based sideslip estimates are combined with heading error in the kinematic controller to provide feedforward slip compensation. Path following error is compensated by a Variable Structure Controller (VSC) to balance graceful motion, path following error, and robustness. Yaw rate commands are used by a backstepping dynamic controller to generate robust steering commands. A High Gain Observer (HGO) estimates sideslip and yaw rate for output feedback control. Stability analysis is provided and peaking is addressed. Field experimental results evaluate the work and provide comparisons to MPC.

- Fast and Safe Path-Following Control Using a State-Dependent Directional Metric

    Author: Li, Zhichao | University of California San Diego
    Author: Arslan, Omur | Eindhoven University of Technology
    Author: Atanasov, Nikolay | University of California, San Diego
 
    keyword: Robust/Adaptive Control of Robotic Systems; Reactive and Sensor-Based Planning; Robot Safety

    Abstract : This paper considers the problem of fast and safe autonomous navigation in partially known environments. Our main contribution is a control policy design based on ellipsoidal trajectory bounds obtained from a quadratic state-dependent distance metric. The ellipsoidal bounds are used to embed directional preference in the control design, leading to system behavior that is adapted to local environment geometry, carefully considering medial obstacles while paying less attention to lateral ones. We use a virtual reference governor system to adaptively follow a desired navigation path, slowing down when system safety may be violated and speeding up otherwise. The resulting controller is able to navigate complex environments faster than common Euclidean-norm and Lyapunov-function-based designs, while retaining stability and collision avoidance guarantees.

- Backlash-Compensated Active Disturbance Rejection Control of Nonlinear Multi-Input Series Elastic Actuators

    Author: DeBoon, Brayden | University of Ontario Institute of Technology
    Author: Nokleby, Scott | University of Ontario Institute of Technology
    Author: Rossa, Carlos | Ontario Tech University
 
    keyword: Robust/Adaptive Control of Robotic Systems; Physical Human-Robot Interaction; Robot Safety

    Abstract : Series elastic actuators with passive compliance have been gaining increasing popularity in force-controlled robotic manipulators. One of the reasons is the actuator's ability to infer the applied torque by measuring the deflection of the elastic element as opposed to directly with dedicated torque sensors. Proper deflection control is pinnacle to achieve a desired output torque and, therefore, small deviances in positional measurements or a nonlinear deformation can have adverse effects on performance. In applications with larger torque requirements, the actuators typically use gear reductions which inherently result in mechanical backlash. This combined with the nonlinear behaviour of the elastic element and unmodelled dynamics, can severely compromise force fidelity.<p>This paper proposes a backlash compensating active disturbance rejection controller (ADRC) for multi-input series elastic actuators. In addition to proper deflection control, a multi-input active disturbance rejection controller is derived and implemented experimentally to mitigate any unmodelled nonlinearities or perturbations to the plant model. The controller is experimentally validated on a hybrid motor-brake-clutch series elastic actuator and the controller performance is compared against traditional error-based controllers. It is shown that the backlash compensated ADRC outperforms classical PID and ADRC methods and is a viable solution to positional measurement error in elastic actuators.

- On Generalized Homogenization of Linear Quadrotor Controller

    Author: Wang, Siyuan | Inria
    Author: Polyakov, Andrey | INRIA Lille
    Author: Zheng, Gang | INRIA
 
    keyword: Robust/Adaptive Control of Robotic Systems; Aerial Systems: Mechanics and Control; Motion Control

    Abstract : A novel scheme for an "upgrade" of a linear control algorithm to a non-linear one is developed based on the concepts of a generalized homogeneity and an implicit homogeneous feedback design. Some tuning rules for a guaranteed improvement of a regulation quality are proposed. Theoretical results are confirmed by real experiments with the quadrotor QDrone of Quanser(TM).

- Coordinated Optical Tweezing and Manipulation of Multiple Microscopic Objects with Stochastic Perturbations

    Author: Ta, Quang Minh | Nanyang Technological University
    Author: Cheah, C. C. | Nanyang Technological University
 
    keyword: Robust/Adaptive Control of Robotic Systems; Motion Control; Automation at Micro-Nano Scales

    Abstract : The Brownian motion of micro-objects in fluid mediums is a fundamental distinction between optical manipulation and robotic manipulation in the macro-world. Besides, current control techniques for optical manipulation generally assume that the manipulated micro-objects are initially trapped prior to the manipulation processes. This paper proposes a robotic control technique for fully automated optical trapping and manipulation of multiple micro-objects with stochastic perturbations. Cooperative control of robotic stage and optical traps is performed to achieve the control objective, in which multiple micro-objects are trapped in sequence by using the robotic stage, and the trapped micro-objects are then manipulated toward a desired region by using laser-steering system. The transition from the trapping operation to manipulation of the trapped micro-objects is fully automated. In this paper, a closed-loop control approach of the optical traps is formulated, and thus ensuring the completeness of the manipulation tasks. The stability of the control system is investigated from a stochastic perspective, and the performance of the proposed control technique is illustrated with experimental results.

- Contact Surface Estimation Via Hapic Perception

    Author: Lin, Hsiu-Chin | McGIll University
    Author: Mistry, Michael | University of Edinburgh
 
    keyword: Robust/Adaptive Control of Robotic Systems; Legged Robots; Field Robots

    Abstract : Legged systems need to optimize contact force in order to maintain contacts. For this, the controller needs to have the knowledge of the surface geometry and how slippery the terrain is. We can use a vision system to realize the terrain, but the accuracy of the vision system degrades in harsh weather, and it cannot visualize the terrain if it is covered with water or grass. Also, the degree of friction cannot be directly visualized. In this paper, we propose an online method to estimate the surface information via haptic exploration. We also introduce a probabilistic criterion to measure the quality of the estimation. The method is validated on both simulation and a real robot platform.

- Local Policy Optimization for Trajectory-Centric Reinforcement Learning

    Author: Kolaric, Patrik | University of Texas at Arlington Research Institute, TX, USA
    Author: Jha, Devesh | Mitsubishi Electric Research Laboratories
    Author: Raghunathan, Arvind | Mitsubishi Electric Research Laboratories
    Author: Lewis, Frank | The University of Texas at Arlington
    Author: Benosman, Mouhacine | Mitsubishi Electric Research Laboratories
    Author: Romeres, Diego | Mitsubishi Electric Research Laboratories
    Author: Nikovski, Daniel | MERL
 
    keyword: Robust/Adaptive Control of Robotic Systems; Optimization and Optimal Control; Learning and Adaptive Systems

    Abstract : The goal of this paper is to present a method for simultaneous trajectory and local stabilizing policy optimization to generate local policies for trajectory-centric model-based reinforcement learning (MBRL). This is motivated by the fact that global policy optimization for non-linear systems could be a very challenging problem both algorithmically and numerically. However, a lot of robotic manipulation tasks are trajectory-centric, and thus do not require a global model or policy. Due to inaccuracies in the learned model estimates, an open-loop trajectory optimization process mostly results in very poor performance when used on the real system. Motivated by these problems, we try to formulate the problem of trajectory optimization and local policy synthesis as a single optimization problem. It is then solved simultaneously as an instance of nonlinear programming. We provide some results for analysis as well as achieved performance of the proposed technique under some simplifying assumptions.

- Automatic Snake Gait Generation Using Model Predictive Control

    Author: Hannigan, Emily | Columbia University
    Author: Song, Bing | Columbia University
    Author: Khandate, Gagan | Columbia University
    Author: Haas-Heger, Maximilian | Columbia University
    Author: Yin, Ji | Columbia University
    Author: Ciocarlie, Matei | Columbia University
 
    keyword: Robust/Adaptive Control of Robotic Systems; Dynamics

    Abstract : Snake robots have the potential to perform jobs like rescue operations in challenging terrains. There is less work focused on snake gait generation across multiple environmental conditions, i.e., the Coulomb fricion, the viscous friction, and fluid dynamics. We propose a MPC-based gait generation approach that can automatically produce effective gaits over different environments with the same implementation, i.e., without human intuition or retuning parameters. The automatic generated gaits can be both faster and more energy efficient than Pareto-optimal serpenoid gaits. This MPC-based approach can also automatically generate complex gaits like obstacle avoidance. Study on practical applicabilty shows the potential of the online implementation of our approach. To satisfy the dynamic smoothness requirement of MPC, We also propose a more accurate anisotropic Coulomb friction model derived by maximum dissipation principle. This is the first time this anisotropic Coulomb fricton model introduced to snake robots.

- Safe and Fast Tracking on a Robot Manipulator: Robust MPC and Neural Network Control

    Author: Nubert, Julian | ETH Zurich
    Author: Koehler, Johannes | University of Stuttgart
    Author: Berenz, Vincent | Max Planck Institute for Intelligent Systems
    Author: Allgower, Frank | University of Stuttgart
    Author: Trimpe, Sebastian | Max Planck Institute for Intelligent Systems
 
    keyword: Robust/Adaptive Control of Robotic Systems; Motion Control; Deep Learning in Robotics and Automation

    Abstract : Fast feedback control and safety guarantees are essential in modern robotics. We present an approach that achieves both by combining novel robust model predictive control (MPC) with function approximation via (deep) neural networks (NNs). The result is a new approach for complex tasks with nonlinear, uncertain, and constrained dynamics as are common in robotics. Specifically, we leverage recent results in MPC research to propose a new robust setpoint tracking MPC algorithm, which achieves reliable and safe tracking of a dynamic setpoint while guaranteeing stability and constraint satisfaction. The presented robust MPC scheme constitutes a one-layer approach that unifies the often separated planning and control layers, by directly computing the control command based on a reference and possibly obstacle positions. As a separate contribution, we show how the computation time of the MPC can be drastically reduced by approximating the MPC law with a NN controller. The NN is trained and validated from offline samples of the MPC, yielding statistical guarantees, and used in lieu thereof at run time. Our experiments on a state-of-the-art robot manipulator are the first to show that both the proposed robust and approximate MPC schemes scale to real-world robotic systems.

- 3D Path-Following Using MRAC on a Millimeter-Scale Spiral-Type Magnetic Robot

    Author: Zhao, Haoran | University of Houston
    Author: Julien, Leclerc | University of Houston
    Author: Feucht, Maria | Baylor University
    Author: Bailey, Olivia | University of Maryland Baltimore County
    Author: Becker, Aaron | University of Houston
 
    keyword: Robust/Adaptive Control of Robotic Systems; Motion Control; Biologically-Inspired Robots

    Abstract : This paper focuses on the 3D path-following of a spiral-type helical magnetic swimmer in a water-filled workspace. The swimmer has a diameter of 2.5 mm, a length of 6 mm, and is controlled by an external time-varying magnetic field. A method to compensate undesired magnetic gradient forces is proposed and tested. Five swimmer designs with different thread pitch values were experimentally analyzed. All were controlled by the same model reference adaptive controller (MRAC). Compared to a conventional hand-tuned PI controller, their 3D path-following performance is significantly improved by using MRAC. At an average speed of 50 mm/s, the path-following mean error of the MRAC is 3.8+/-1.8 mm, less than one body length of the swimmer. The versatility

- Adaptive Model-Based Myoelectric Control for a Soft Wearable Arm Exosuit (I)

    Author: Lotti, Nicola | University of Heidelberg
    Author: Xiloyannis, Michele | Eidgen�ssische Technische Hochschule (ETH) Zurich
    Author: Durandau, Guillaume | University of Twente
    Author: Galofaro, Elisa | University of Genoa
    Author: Sanguineti, Vittorio | University of Genoa
    Author: Masia, Lorenzo | Heidelberg University
    Author: Sartori, Massimo | University of Twente
 
    keyword: Robust/Adaptive Control of Robotic Systems; Soft Robot Applications; Wearable Robots

    Abstract : Despite advances in mechatronic design, the widespread adoption of wearable robots for supporting human mobility has been hampered by (i) ergonomic limitations in rigid exoskeletal structures, and by (ii) the lack of human machine interfaces capable of sensing musculoskeletal states and translating them into robot control commands. We have developed a new framework that combines, for the first time, a model-based HMI with a soft wearable arm exosuit, that has the potential of addressing key limitations in current HMI and wearable robots. Results showed that the model controlled exosuit operated synchronously with biological muscle contraction. Remarkably, the exosuit dynamically modulated mechanical assistance across all investigated loads, thereby displaying adaptive behavior. As a result, both the exosuit's intrinsic dynamics and the external mechanical loads appeared to be transparent to the individuals' musculoskeletal systems. This was reflected by the fact that, with exosuit assistance, both muscle electromyograms and resulting forces, always varied within comparable ranges across all investigated rotational velocities and loads, i.e. the external load effect on muscle function was minimized. The ability of seamlessly combining musculoskeletal force estimators with wearable soft mechatronics opens new avenues for assisting human movement both in healthy and impaired individuals.

## Space Robotics and Automation
- Planetary Rover Exploration Combining Remote and in Situ Measurements for Active Spectroscopic Mapping

    Author: Candela, Alberto | Carnegie Mellon University
    Author: Kodgule, Suhit | Carnegie Mellon University
    Author: Edelson, Kevin | Carnegie Mellon University
    Author: Vijayarangan, Srinivasan | Carnegie Mellon University
    Author: Thompson, David | Jet Propulsion Laboratory / California Institute of Technology
    Author: Noe Dobrea, Eldar | Planetary Science Institute
    Author: Wettergreen, David | Carnegie Mellon University
 
    keyword: Space Robotics and Automation; Field Robots; Learning and Adaptive Systems

    Abstract : Maintaining high levels of productivity for planetary rover missions is very difficult due to limited communication and heavy reliance on ground control. There is a need for autonomy that enables more adaptive and efficient actions based on real-time information. This paper presents an autonomous mapping and exploration approach for planetary rovers. We first describe a machine learning model that actively combines remote and rover measurements for mapping. We focus on spectroscopic data because they are commonly used to investigate surface composition. We then incorporate notions from information theory and non-myopic path planning to improve exploration productivity. Finally, we demonstrate the feasibility and successful performance of our approach via spectroscopic investigations of Cuprite, Nevada; a well-studied region of mineralogical and geological interest. We first perform a detailed analysis in simulations, and then validate those results with an actual rover in the field in Nevada.

- Magnetic Docking Mechanism for Free-Flying Space Robots with Spherical Surfaces

    Author: Watanabe, Keisuke | Japan Aerospace Exploration Agency
 
    keyword: Space Robotics and Automation; Mechanism Design; Compliant Joint/Mechanism

    Abstract : Autonomous operation of robots in the International Space Station (ISS) is required to maximize the use of limited resources and enable astronauts to concentrate more on the valuable tasks. To achieve this goal, we are developing a station where the robot approaches, docks, charges, and then undocks. In this paper, the     Authors proposed a magnetic docking mechanism for free-flying robots having spherical surfaces, which makes it possible for the robot to dock securely without requiring highly precise guidance, navigation and control capability. By making use of a slider guide and repelling magnet pairs, the mechanism can achieve tolerance for larger robot position error as compared to the conventional fixed guide mechanism. The experimental results showed that the proposed mechanism can effectively enlarge the acceptable error range of poses, and also reduce acceleration at the moment of impact. We also introduced the model to predict whether docking will be succeeded or not from the contact condition of the robot and the guide, using a machine learning technique, Gaussian Process Regression (GPR). The prediction results shows that the learnt model can express the contact condition of successful docking.

- Barefoot Rover: A Sensor-Embedded Rover Wheel Demonstrating In-Situ Engineering and Science Extractions Using Machine Learning

    Author: Marchetti, Yuliya | Jet Propulsion Laboratory
    Author: Lightholder, Jack | Jet Propulsion Laboratory
    Author: Junkins, Eric | Jet Propulsion Laboratory
    Author: Cross, Matthew | Western University
    Author: Mandrake, Lukas | Jet Propulsion Laboratory
    Author: Fraeman, Abigail | Jet Propulsion Laboratory
 
    keyword: Space Robotics and Automation; Force and Tactile Sensing; Learning and Adaptive Systems

    Abstract : In this work, we demonstrate an instrumented wheel concept which utilizes a 2D pressure grid, an electrochemical impedance spectroscopy (EIS) sensor and machine learning to extract meaningful metrics from the interaction between the wheel and surface terrain. These include continuous slip/skid estimation, balance, and sharpness for engineering applications. Estimates of surface hydration, texture, terrain patterns, and regolith physical properties such as cohesion and angle of internal friction are additionally calculated for science applications. Traditional systems rely on post-processing of visual images and vehicle telemetry to estimate these metrics. Through in-situ sensing, these metrics can be calculated in near real time and made available to onboard science and engineering autonomy applications. This work aims to provide a deployable system for future planetary exploration missions to increase science and engineering capabilities through increased knowledge of the terrain.

- Deep Learning for Spacecraft Pose Estimation from Photorealistic Rendering

    Author: Proen�a, Pedro F. | University of Surrey
    Author: Gao, Yang | University of Surrey
 
    keyword: Space Robotics and Automation; Simulation and Animation; Computer Vision for Automation

    Abstract : On-orbit proximity operations in space rendezvous, docking and debris removal require precise and robust 6D pose estimation under a wide range of lighting conditions and against highly textured background, i.e., the Earth. This paper investigates leveraging deep learning and photorealistic rendering for monocular pose estimation of known uncooperative spacecraft. We first present a simulator built on Unreal Engine 4, named URSO, to generate labeled images of spacecraft orbiting the Earth, which can be used to train and evaluate neural networks. Secondly, we propose a deep learning framework for pose estimation based on orientation soft classification, which allows modelling orientation ambiguity as a mixture model. This framework was evaluated both on URSO datasets and the European Space Agency pose estimation challenge. In this competition, our best model achieved 3rd place on the synthetic test set and 2nd place on the real test set. Moreover, our results show the impact of several architectural and training aspects, and we demonstrate qualitatively how models learned on URSO datasets can perform on real images from space.

- Concurrent Parameter Identification and Control for Free-Floating Robotic Systems During On-Orbit Servicing

    Author: Christidi-Loumpasefski, Olga-Orsalia | National Technical University of Athens
    Author: Rekleitis, Georgios | National Technical University of Athens
    Author: Papadopoulos, Evangelos | National Technical University of Athens
 
    keyword: Space Robotics and Automation; Calibration and Identification; Motion Control of Manipulators

    Abstract : To control a free-floating robotic system with uncertain parameters in OOS tasks with high accuracy, a fast parameter identification method, previously developed by the     Authors, is enhanced further and used concurrently with a controller. The method provides accurate parameter estimates, without any prior knowledge of any system dynamic properties. This control scheme compensates for the accumulated angular momentum on the reaction wheels (RWs), which acts as a disturbance to the robotic servicer base. While any controller using parameter information can be used, a transposed Jacobian controller, modified to include RW angular momentum disturbance rejection, is employed here. Three-dimensional simulations demonstrate the method's validity.

- A Dual Quaternion-Based Discrete Variational Approach for Accurate and Online Inertial Parameter Estimation in Free-Flying Robots

    Author: Ekal, Monica | Instituto Superior Tecnico
    Author: Ventura, Rodrigo | Instituto Superior Técnico
 
    keyword: Space Robotics and Automation; Calibration and Identification

    Abstract : The performance of model-based motion control for free-flying robots relies on accurate estimation of their parameters. In this work, a method of rigid body inertial parameter estimation which relies on a variational approach is presented. Instead of discretizing the continuous equations of motion, discrete dual quaternion equations based on variational mechanics are used to formulate a linear parameter estimation problem. This method depends only on the pose of the rigid body obtained from standard localization algorithms. Recursive semi-definite programming is used to estimate the inertial parameters (mass, rotational inertia and center of mass offset) online. Linear Matrix Inequality constraints based on the pseudo-inertia matrix ensure that the estimates obtained are fully physically consistent. Simulation results demonstrate that this method is robust to disturbances and the produced estimates are at least one order of magnitude more accurate when compared to discretization using finite differences.

## Perception for Grasping and Manipulation

- Transferable Task Execution from Pixels through Deep Planning Domain Learning

    Author: Kase, Kei | Waseda University
    Author: Paxton, Chris | NVIDIA Research
    Author: Mazhar, Hammad | NVIDIA
    Author: Ogata, Tetsuya | Waseda University
    Author: Fox, Dieter | University of Washington
 
    keyword: Deep Learning in Robotics and Automation; Reactive and Sensor-Based Planning; Task Planning

    Abstract : While robots can learn models to solve many manipulation tasks from raw visual input, they cannot usually use these models to solve new problems. On the other hand, symbolic planning methods such as STRIPS have long been able to solve new problems given only a domain definition and a symbolic goal, but these approaches often struggle on the real world robotic tasks due to the challenges of grounding these symbols from sensor data in a partially-observable world. We propose Deep Planning Domain Learning (DPDL), an approach that combines the strengths of both methods to learn a hierarchical model. DPDL learns a high-level model which predicts values for a large set of logical predicates consisting of the current symbolic world state, and separately learns a low-level policy which translates symbolic operators into executable actions on the robot. This allows us to perform complex, multi-step tasks even when the robot has not been explicitly trained on them. We show our method on manipulation tasks in a photorealistic kitchen scenario.

- Depth by Poking: Learning to Estimate Depth from Self-Supervised Grasping

    Author: Goodrich, Ben | Osaro, Inc
    Author: Kuefler, Alex | Osaro, Inc
    Author: Richards, William | Osaro, Inc
 
    keyword: Deep Learning in Robotics and Automation; RGB-D Perception; Perception for Grasping and Manipulation

    Abstract : Accurate depth estimation remains an open problem for robotic manipulation; even state of the art techniques including structured light and LiDAR sensors fail on reflective or transparent surfaces. We address this problem by training a neural network model to estimate depth from RGB-D images, using labels from physical interactions between a robot and its environment. Our network predicts, for each pixel in an input image, the z position that a robot's end effector would reach if it attempted to grasp or poke at the corresponding position. Given an autonomous grasping policy, our approach is self-supervised as end effector position labels can be recovered through forward kinematics, without human annotation. Although gathering such physical interaction data is expensive, it is necessary for training and routine operation of state of the art manipulation systems. Therefore, this depth estimator comes �for free� while collecting data for other tasks (e.g., grasping, pushing, placing). We show our approach achieves significantly lower root mean squared error than traditional structured light sensors and unsupervised deep learning methods on difficult, industry-scale jumbled bin datasets.

- Online Learning of Object Representations by Appearance Space Feature Alignment

    Author: Pirk, Soren | Robotics at Google
    Author: Khansari, Mohi | Google X
    Author: Bai, Yunfei | Google X
    Author: Lynch, Corey | Google Brain
    Author: Sermanet, Pierre | Google
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization

    Abstract : We propose a self-supervised approach for learning representations of objects from monocular videos and demonstrate it is particularly useful for robotics. The main contributions of this paper are: 1) a self-supervised model called Object-Contrastive Network (OCN) that can discover and disentangle object attributes from video without using any labels; 2) we leverage self-supervision for online adaptation: the longer our online model looks at objects in a video, the lower the object identification error, while the offline baseline remains with a large fixed error; 3) we show the usefulness of our approach for a robotic pointing task; a robot can point to objects similar to the one presented in front of it. Videos illustrating online object adaptation and robotic pointing are provided as supplementary material.

- Visual Prediction of Priors for Articulated Object Interaction

    Author: Moses, Caris | Massachusetts Institute of Technology
    Author: Noseworthy, Michael | Massachusetts Institute of Technology
    Author: Kaelbling, Leslie | MIT
    Author: Lozano-Perez, Tomas | MIT
    Author: Roy, Nicholas | Massachusetts Institute of Technology
 
    keyword: Learning and Adaptive Systems; Deep Learning in Robotics and Automation; AI-Based Methods

    Abstract : Exploration in novel settings can be challenging without prior experience in similar domains. However, humans are able to build on prior experience quickly and efficiently. Children exhibit this behavior when playing with toys. For example, given a toy with a yellow and blue door, a child will explore with no clear objective, but once they have discovered how to open the yellow door, they will most likely be able to open the blue door much faster. Adults also exhibit this behaviour when entering new spaces such as kitchens. We develop a method, Contextual Prior Prediction, which provides a means of transferring knowledge between interactions in similar domains through vision. We develop agents that exhibit exploratory behavior with increasing efficiency, by learning visual features that are shared across environments, and how they correlate to actions. Our problem is formulated as a Contextual Multi-Armed Bandit where the contexts are images, and the robot has access to a parameterized action space. Given a novel object, the objective is to maximize reward with few interactions. A domain which strongly exhibits correlations between visual features and motion is kinemetically constrained mechanisms. We evaluate our method on simulated prismatic and revolute joints.

- MT-DSSD: Deconvolutional Single Shot Detector Using Multi Task Learning for Object Detection, Segmentation, and Grasping Detection

    Author: Araki, Ryosuke | Chubu University
    Author: Onishi, Takeshi | Chubu University
    Author: Hirakawa, Tsubasa | Chubu University
    Author: Yamashita, Takayoshi | Chubu University
    Author: Fujiyoshi, Hironobu | Chubu University
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; Computer Vision for Automation

    Abstract : This paper presents the multi-task Deconvolutional Single Shot Detector (MT-DSSD), which runs three tasks---object detection, semantic object segmentation, and grasping detection for a suction cup---in a single network based on the DSSD. Simultaneous execution of object detection and segmentation by multi-task learning improves the accuracy of these two tasks. Additionally, the model detects grasping points and performs the three recognition tasks necessary for robot manipulation. The proposed model can perform fast inference, which reduces the time required for grasping operation. Evaluations using the Amazon Robotics Challenge (ARC) dataset showed that our model has better object detection and segmentation performance than comparable methods, and robotic experiments for grasping show that our model can detect the appropriate grasping point.

- Using Synthetic Data and Deep Networks to Recognize Primitive Shapes for Object Grasping

    Author: Tang, Chao | Georgia Institute of Technology
    Author: Lin, Yunzhi | Georgia Institute of Technology
    Author: Chu, Fu-Jen | University of Michigan
    Author: Vela, Patricio | Georgia Institute of Technology
 
    keyword: Perception for Grasping and Manipulation; Grasping; Deep Learning in Robotics and Automation

    Abstract : A segmentation-based architecture is proposed to decompose objects into multiple primitive shapes from monocular depth input for robotic manipulation. The backbone deep network is trained on synthetic data with 6 classes of primitive shapes generated by a simulation engine. Each primitive shape is designed with parametrized grasp families, permitting the pipeline to identify multiple grasp candidates per shape primitive region. The grasps are priority ordered via proposed ranking algorithm, with the first feasible one chosen for execution. On task-free grasping of individual objects, the method achieves a 94% success rate. On task-oriented grasping, it achieves a 76% success rate.

- Real-Time, Highly Accurate Robotic Grasp Detection Using Fully Convolutional Neural Network with Rotation Ensemble Module

    Author: Park, Dongwon | UNIST
    Author: Seo, YongHyeok | Unist
    Author: Chun, Se Young | Ulsan National Institute of Science and Technology
 
    keyword: Perception for Grasping and Manipulation; Grasping; RGB-D Perception

    Abstract : Rotation invariance has been an important topic in computer vision tasks. Ideally, robot grasp detection should be rotation-invariant. However, rotation-invariance in robotic grasp detection has been only recently studied by using rotation anchor box that are often time-consuming and unreliable for multiple objects. In this paper, we propose a rotation ensemble module (REM) for robotic grasp detection using convolutions that rotates network weights. Our proposed REM was able to outperform current state-of-the-art methods by achieving up to 99.2% (image-wise), 98.6% (object-wise) accuracies on the Cornell dataset with real-time computation (50 frames per second). Our proposed method was also able to yield reliable grasps for multiple objects and up to 93.8% success rate for the real-time robotic grasping task with a 4-axis robot arm for small novel objects that was significantly higher than the baseline methods by 11-56%.

- Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly

    Author: Zakka, Kevin | Stanford, Google
    Author: Zeng, Andy | Google
    Author: Lee, Johnny | Google
    Author: Song, Shuran | Columbia University
 
    keyword: Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation; RGB-D Perception

    Abstract : Is it possible to learn policies for robotic assembly that can generalize to new objects? We explore this idea in the context of the kit assembly task. Since classic methods rely heavily on object pose estimation, they often struggle to generalize to new objects without 3D CAD models or task- specific training data. In this work, we propose to formulate the kit assembly task as a shape matching problem, where the goal is to learn a shape descriptor that establishes geometric correspondences between object surfaces and their target place- ment locations from visual input. This formulation enables the model to acquire a broader understanding of how shapes and surfaces fit together for assembly - allowing it to generalize to new objects and kits. To obtain training data for our model, we present a self-supervised data-collection pipeline that obtains ground truth object-to-placement correspondences by disassembling complete kits. Our resulting real-world system, Form2Fit, learns effective pick and place strategies for assem- bling objects into a variety of kits - achieving 90% average success rates under different initial conditions (e.g. varying object and kit poses), 94% success under new configurations of multiple kits, and over 86% success with completely new objects and kits. Code, videos, and supplemental material are available at https://form2fit.github.io

- Learning Rope Manipulation Policies Using Dense Object Descriptors Trained on Synthetic Depth Data

    Author: Sundaresan, Priya | University of California, Berkeley
    Author: Grannen, Jennifer | UC Berkeley
    Author: Thananjeyan, Brijen | UC Berkeley
    Author: Balakrishna, Ashwin | University of California, Berkeley
    Author: Laskey, Michael | University of California, Berkeley
    Author: Stone, Kevin | Toyota Research Institute
    Author: Gonzalez, Joseph E. | UC Berkeley
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation; Manipulation Planning

    Abstract : Robotic manipulation of deformable 1D objects such as ropes, cables, and hoses is challenging due to the lack of high-fidelity analytic models and large configuration spaces. Furthermore, learning end-to-end manipulation policies directly from images and physical interaction requires significant time on a robot and can fail to generalize across tasks. We address these challenges using interpretable deep visual representations for rope, extending recent work on dense object descriptors for robot manipulation. This facilitates the design of interpretable and transferable geometric policies built on top of the learned representations, decoupling visual reasoning and control. We present an approach that learns point-pair correspondences between initial and goal rope configurations, which implicitly encodes geometric structure, entirely in simulation from synthetic depth images. We demonstrate that the learned representation - dense depth object descriptors (DDODs) - can be used to manipulate a real rope into a variety of different arrangements either by learning from demonstrations or using interpretable geometric policies. In 50 trials of a knot-tying task with the ABB YuMi Robot, the system achieves a 66% knot-tying success rate from previously unseen configurations. See https://tinyurl.com/rope-learning for supplementary material and videos.

- Efficient Two Step Optimization for Large Embedded Deformation Graph Based SLAM

    Author: Song, Jingwei | University of Technology, Sydney
    Author: Bai, Fang | University of Technology, Sydney
    Author: Zhao, Liang | University of Technology Sydney
    Author: Huang, Shoudong | University of Technology, Sydney
    Author: Xiong, Rong | Zhejiang University
 
    keyword: Perception for Grasping and Manipulation; Computer Vision for Other Robotic Applications; SLAM

    Abstract : Embedded deformation graph is a widely used technique in deformable geometry and graphical problems. Recently the technique has been transmitted to stereo (or RGBD) sensor based SLAM applications, it remains challenging to compromise the computational cost as the model grows. In practice, the processing time grows rapidly in accordance with the expansion of maps. In this paper, we propose an approach to decouple nodes of deformation graph in large scale dense deformable SLAM and keep the estimation time to be constant. We observe that only partial deformable nodes in the graph are connected to visible points. Based on this fact, the sparsity of the original Hessian matrix is utilized to split parameter estimation into two independent steps. With this new technique, we achieve faster parameter estimation with amortized computation complexity reduced from O(n^2) to closing O(1). As a result, the computation cost barely increases as the map keeps growing. Based on our strategy, computational bottleneck in large scale embedded deformation graph based applications will be greatly mitigated. The effectiveness is validated by experiments, featuring large scale deformation scenarios.

- Camera-To-Robot Pose Estimation from a Single Image

    Author: Lee, Timothy Edward | Carnegie Mellon University
    Author: Tremblay, Jonathan | Nvidia
    Author: To, Thang | Nvidia Corp
    Author: Cheng, Jia | Nvidia Corp
    Author: Mosier, Terry | NVIDIA
    Author: Kroemer, Oliver | Carnegie Mellon University
    Author: Fox, Dieter | University of Washington
    Author: Birchfield, Stan | NVIDIA
 
    keyword: Perception for Grasping and Manipulation; Object Detection, Segmentation and Categorization; Computer Vision for Other Robotic Applications

    Abstract : We present an approach for estimating the pose of an external camera with respect to a robot using a single RGB image of the robot. The image is processed by a deep neural network to detect 2D projections of keypoints (such as joints) associated with the robot. The network is trained entirely on simulated data using domain randomization to bridge the reality gap. Perspective-n-point (PnP) is then used to recover the camera extrinsics, assuming that the camera intrinsics and joint configuration of the robot manipulator are known. Unlike classic hand-eye calibration systems, our method does not require an off-line calibration step. Rather, it is capable of computing the camera extrinsics from a single frame, thus opening the possibility of on-line calibration. We show experimental results for three different camera sensors, demonstrating that our approach is able to achieve accuracy with a single frame that is comparable to that of classic off-line hand-eye calibration using multiple frames. With additional frames from a static pose, accuracy improves even further. Code, datasets, and pretrained models for three widely-used robot manipulators are made available.

- DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation

    Author: Lambeta, Mike Maroje | Facebook
    Author: Chou, Po-Wei | Facebook
    Author: Tian, Stephen | UC Berkeley
    Author: Yang, Brian | University of California, Berkeley
    Author: Maloon, Benjamin | Facebook
    Author: Most, Victoria Rose | Facebook
    Author: Stroud, Dave | Facebook
    Author: Santos, Raymond | Facebook
    Author: Byagowi, Ahmad | Facebook
    Author: Kammerer, Gregg | Facebook
    Author: Jayaraman, Dinesh | Facebook AI Research and University of Pennsylvania
    Author: Calandra, Roberto | Facebook
 
    keyword: Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation; Force and Tactile Sensing

    Abstract : Despite decades of research, general purpose in-hand manipulation remains one of the unsolved challenges of robotics. One of the contributing factors that limit current robotic manipulation systems is the difficulty of precisely sensing contact forces -- sensing and reasoning about contact forces are crucial to accurately control interactions with the environment. As a step towards enabling better robotic manipulation, we introduce DIGIT, an inexpensive, compact, and high-resolution tactile sensor geared towards in-hand manipulation. DIGIT improves upon past vision-based tactile sensors by miniaturizing the form factor to be mountable on multi-fingered hands, and by providing several design improvements that result in an easier, more repeatable manufacturing process, and enhanced reliability. We demonstrate the capabilities of the DIGIT sensor by training deep neural network model-based controllers to manipulate glass marbles in-hand with a multi-finger robotic hand. To provide the robotic community access to reliable and low-cost tactile sensors, we open-source the DIGIT design at www.digit.ml.

- LyRN (Lyapunov Reaching Network): A Real-Time Closed Loop Approach from Monocular Vision

    Author: Zhuang, Zheyu | Australian National University
    Author: Yu, Xin | Australian National University
    Author: Mahony, Robert | Australian National University
 
    keyword: Perception for Grasping and Manipulation; Visual Servoing; Visual Learning

    Abstract : We propose a closed-loop, multi-instance control algorithm for visually guided reaching based on novel learning principles. A control Lyapunov function methodology is used to design a reaching action for a complex multi-instance task in the case where full state information (poses of all potential reaching points) is available. The proposed algorithm uses monocular vision and manipulator joint angles as the input to a deep convolution neural network to predict the value of the control Lyapunov function (cLf) and corresponding velocity control. The resulting network output is used in real-time as visual control for the grasping task with the multi-instance capability emerging naturally from the design of the control Lyapunov function.<p>We demonstrate the proposed algorithm grasping mugs (textureless and symmetric objects) on a table-top from an over-the-shoulder monocular RGB camera. The manipulator dynamically converges to the best-suited target among multiple identical instances from any random initial pose within the workspace. The system trained with simulated data only is able to achieve 90.3% grasp success rate in the real-world experiments with up to 85Hz closed-loop control on one GTX 1080Ti GPU and significantly outperforms a Pose-Based-Visual-Servo (PBVS) grasping system adapted from a state-of-the-art single shot RGB 6D pose estimation algorithm. A key contribution of the paper is the inclusion of a first-order differential constraint associate

- Object Finding in Cluttered Scenes Using Interactive Perception

    Author: Novkovic, Tonci | Autonomous Systems Lab, ETH Zurich
    Author: Pautrat, Remi | Inria Nancy Grand-Est
    Author: Furrer, Fadri | ETH Zurich
    Author: Breyer, Michel | Autonomous Systems Lab, ETH Zurich
    Author: Siegwart, Roland | ETH Zurich
    Author: Nieto, Juan | ETH Zurich
 
    keyword: Perception for Grasping and Manipulation; AI-Based Methods

    Abstract : Object finding in clutter is a skill that requires perception of the environment and in many cases physical interaction. In robotics, interactive perception defines a set of algorithms that leverage actions to improve the perception of the environment, and vice versa use perception to guide the next action. Scene interactions are difficult to model, therefore, most of the current systems use predefined heuristics. This limits their ability to efficiently search for the target object in a complex environment. In order to remove heuristics and the need for explicit models of the interactions, in this work we propose a reinforcement learning based active and interactive perception system for scene exploration and object search. We evaluate our work both in simulated and in real-world experiments using a robotic manipulator equipped with an RGB and a depth camera, and compare our system to two baselines. The results indicate that our approach, trained in simulation only, transfers smoothly to reality and can solve the object finding task efficiently and with more than 88% success rate.

- Multi-Modal Perception and Transfer Learning for Grasping Transparent and Specular Objects

    Author: Weng, Thomas | Carnegie Mellon University
    Author: Pallankize, Amith | BITS Pilani
    Author: Tang, Yimin | ShanghaiTech University
    Author: Kroemer, Oliver | Carnegie Mellon University
    Author: Held, David | Carnegie Mellon University
 
    keyword: Perception for Grasping and Manipulation; Grasping; RGB-D Perception

    Abstract : State-of-the-art object grasping methods rely on depth sensing to plan robust grasps, but commercially available depth sensors fail to detect transparent and specular objects. To improve grasping performance on such objects, we introduce a method for learning a multi-modal perception model by bootstrapping from an existing uni-modal model. This transfer learning approach requires only a pre-existing uni modal grasping model and paired multi-modal image data for training, foregoing the need for ground-truth grasp success labels nor real grasp attempts. Our experiments demonstrate that our approach is able to reliably grasp transparent and reflective objects. Video and supplementary material are available at: https://sites.google.com/view/transparent-specular-grasping.

- CCAN: Constraint Co-Attention Network for Instance Grasping

    Author: Cai, Junhao | Sun Yat-Sen University
    Author: Tao, Xuefeng | Sun Yat-Sen University
    Author: Cheng, Hui | Sun Yat-Sen University
    Author: Zhang, Zhanpeng | SenseTime Group Limited
 
    keyword: Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation; Visual Learning

    Abstract : Instance grasping is a challenging robotic grasping task when a robot aims to grasp a specified target object in cluttered scenes. In this paper, we propose a novel end-to-end instance grasping method using only monocular workspace and query images, where the workspace image includes several objects and the query image only contains the target object. To effectively extract discriminative features and facilitate the training process, a learning-based method, referred to as Constraint Co-Attention Network (CCAN), is proposed which consists of a constraint co-attention module and a grasp affordance predictor. An effective co-attention module is presented to construct the features of a workspace image from the extracted features of the query image. By introducing soft constraints into the co-attention module, it highlights the target object's features while trivializes other objects' features in the workspace image. Using the features extracted from the co-attention module, the cascaded grasp affordance interpreter network only predicts the grasp configuration for the target object. The training of the CCAN is totally based on simulated self-supervision. Extensive qualitative and quantitative experiments show the effectiveness of our method both in simulated and real-world environments even for totally unseen objects.

- RLBench: The Robot Learning Benchmark &amp; Learning Environment

    Author: James, Stephen | Imperial College London
    Author: Ma, Zicong | Imperial College London
    Author: Rovick Arrojo, David | Imperial College London
    Author: Davison, Andrew J | Imperial College London
 
    keyword: Perception for Grasping and Manipulation; Performance Evaluation and Benchmarking; Deep Learning in Robotics and Automation

    Abstract : We present a challenging new benchmark and learning-environment for robot learning: RLBench. The benchmark features 100 completely unique, hand-designed tasks, ranging in difficulty from simple target reaching and door opening to longer multi-stage tasks, such as opening an oven and placing a tray in it. We provide an array of both proprioceptive observations and visual observations, which include rgb, depth, and segmentation masks from an over-the-shoulder stereo camera and an eye-in-hand monocular camera. Uniquely, each task comes with an infinite supply of demos through the use of motion planners operating on a series of waypoints given during task creation time; enabling an exciting flurry of demonstration-based learning possibilities. RLBench has been designed with scalability in mind; new tasks, along with their motion-planned demos, can be easily created and then verified by a series of tools, allowing users to submit their own tasks to the RLBench task repository. This large-scale benchmark aims to accelerate progress in a number of vision-guided manipulation research areas, including: reinforcement learning, imitation learning, multi-task learning, geometric computer vision, and in particular, few-shot learning. With the benchmark's breadth of tasks and demonstrations, we propose the first large-scale few-shot challenge in robotics. We hope that the scale and diversity of RLBench offers unparalleled research opportunities in the robot learning community and beyond.

- Learning Task-Oriented Grasping from Human Activity Datasets

    Author: Kokic, Mia | KTH
    Author: Kragic, Danica | KTH
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation

    Abstract : We propose to leverage a real-world, human activity RGB dataset to teach a robot Task-Oriented Grasping (TOG). We develop a model that takes as input an RGB image and outputs a hand pose and configuration as well as an object pose and a shape. We follow the insight that jointly estimating hand and object poses increases accuracy compared to estimating these quantities independently of each other. Given the trained model, we process an RGB dataset to automatically obtain the data to train a TOG model. This model takes as input an object point cloud and outputs a suitable region for task-specific grasping. Our ablation study shows that training an object pose predictor with the hand pose information (and vice versa) is better than training without this information. Furthermore, our results on a real-world dataset show the applicability and competitiveness of our method over state-of-the-art. Experiments with a robot demonstrate that our method can allow a robot to preform TOG on novel objects.

- Inferring the Material Properties of Granular Media for Robotic Tasks

    Author: Matl, Carolyn | University of California, Berkeley
    Author: Narang, Yashraj | NVIDIA
    Author: Bajcsy, Ruzena | Univ of California, Berkeley
    Author: Ramos, Fabio | University of Sydney, NVIDIA
    Author: Fox, Dieter | University of Washington
 
    keyword: Perception for Grasping and Manipulation; Modeling, Control, and Learning for Soft Robots; Simulation and Animation

    Abstract : Granular media (e.g., cereal grains, plastic resin pellets, and pills) are ubiquitous in robotics-integrated industries, such as agriculture, manufacturing, and pharmaceutical development. This prevalence mandates the accurate and efficient simulation of these materials. This work presents a software and hardware framework that automatically calibrates a fast physics simulator to accurately simulate granular materials by inferring material properties from real-world depth images of granular formations (i.e., piles and rings). Specifically, coefficients of sliding friction, rolling friction, and restitution of grains are estimated from summary statistics of grain formations using likelihood-free Bayesian inference. The calibrated simulator accurately predicts unseen granular formations in both simulation and experiment; furthermore, simulator predictions are shown to generalize to more complex tasks, including using a robot to pour grains into a bowl, as well as to create a desired pattern of piles and rings.

- KETO: Learning Keypoint Representations for Tool Manipulation

    Author: Qin, Zengyi | Stanford
    Author: Fang, Kuan | Stanford University
    Author: Zhu, Yuke | Stanford University
    Author: Fei-Fei, Li | Stanford University
    Author: Savarese, Silvio | Stanford University
 
    keyword: Perception for Grasping and Manipulation; RGB-D Perception

    Abstract : We aim to develop an algorithm for robots to manipulate novel objects as tools for completing different task goals. An efficient and informative representation would facilitate the effectiveness and generalization of such algorithms. For this purpose, we present KETO, a framework of learning keypoint representations of tool-based manipulation. For each task, a set of task-specific keypoints is jointly predicted from 3D point clouds of the tool object by a deep neural network. These keypoints offer a concise and informative description of the object to determine grasps and subsequent manipulation actions. The model is learned from self-supervised robot interactions in the task environment without the need for explicit human annotations. We evaluate our framework in three manipulation tasks with tool use. Our model consistently outperforms state-of-the-art methods in terms of task success rates. Qualitative results of keypoint prediction and tool generation are shown to visualize the learned representations.

- Learning to See before Learning to Act: Visual Pre-Training for Manipulation

    Author: Lin, Yen-Chen | Massachusetts Institute of Technology
    Author: Zeng, Andy | Google
    Author: Song, Shuran | Columbia University
    Author: Isola, Phillip | UC Berkeley
    Author: Lin, Tsung-Yi | Google
 
    keyword: Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation; Visual Learning

    Abstract : Does having visual priors (e.g. the ability to detect objects) facilitate learning to perform vision-based manipula- tion (e.g. picking up objects)? We study this problem under the framework of transfer learning, where the model is first trained on a passive vision task (i.e., the data distribution does not depend on the agent's decisions), then adapted to perform an active manipulation task (i.e., the data distribution does depend on the agent's decisions). We find that pre-training on vision tasks significantly improves generalization and sample efficiency for learning to manipulate objects. However, realizing these gains requires careful selection of which parts of the model to transfer. Our key insight is that outputs of standard vision models highly correlate with affordance maps commonly used in manipulation. Therefore, we explore directly transferring model parameters from vision networks to affordance prediction networks, and show that this can result in successful zero- shot adaptation, where a robot can pick up certain objects with zero robotic experience. With just a small amount of robotic experience, we can further fine-tune the affordance model to achieve better results. With just 10 minutes of suction experience or 1 hour of grasping experience, our method achieves &#8764; 80% success rate at picking up novel objects.

-  Learning Continuous 3D Reconstructions for Geometrically Aware Grasping

    Author: Van der Merwe, Mark | University of Utah
    Author: Lu, Qingkai | University of Utah
    Author: Sundaralingam, Balakumar | University of Utah
    Author: Matak, Martin | University of Utah
    Author: Hermans, Tucker | University of Utah

- Contact-Based In-Hand Pose Estimation Using Particle Filtering

    Author: von Drigalski, Felix Wolf Hans Erich | OMRON SINIC X Corporation
    Author: Taniguchi, Shohei | The University of Tokyo
    Author: Lee, Robert | Australian Centre for Robotic Vision
    Author: Matsubara, Takamitsu | Nara Institute of Science and Technology
    Author: Hamaya, Masashi | OMRON SINIC X Corporation
    Author: Tanaka, Kazutoshi | OMRON SINIC X Corporation
    Author: Ijiri, Yoshihisa | OMRON Corp
 
    keyword: Perception for Grasping and Manipulation; Intelligent and Flexible Manufacturing; Factory Automation

    Abstract : In industrial assembly tasks, the position of an object grasped by the robot has to be known with high precision in order to insert or place it. In real applications, this problem is commonly solved by jigs that are specially produced for each part. However, they significantly limit flexibility and are prohibitive when the target parts change often, so a flexible method to localize parts with high accuracy after grasping is desired. To solve this problem, we propose a method that can estimate the position of an object in the robot's hand to sub-millimeter precision, and can improve its estimate incrementally, using only minimal calibration and a force sensor. Our method is applicable to any robotic gripper and any rigid object that the gripper can hold, and requires only a force sensor. We demonstrate that the method can determine the position of an object to a precision of under 1~mm without using any part-specific jigs or equipment.

- A Single Multi-Task Deep Neural Network with Post-Processing for Object Detection with Reasoning and Robotic Grasp Detection

    Author: Park, Dongwon | UNIST
    Author: Seo, YongHyeok | Unist
    Author: Shin, Dongju | UNIST
    Author: Choi, Jaesik | Ulsan National Institute of Science and Technology
    Author: Chun, Se Young | Ulsan National Institute of Science and Technology
 
    keyword: Perception for Grasping and Manipulation; Grasping; RGB-D Perception

    Abstract : Applications of deep neural network (DNN) based object and grasp detections could be expanded significantly when the network output is processed by a high-level reasoning over relationship of objects. Recently, robotic grasp detection and object detection with reasoning have been investigated using DNNs. There have been efforts to combine these multi-tasks using separate networks so that robots can deal with situations of grasping specific target objects in the cluttered, stacked, complex piles of novel objects from a single RGB-D camera. We propose a single multi-task DNN that yields accurate detections of objects, grasp position and relationship reasoning among objects. Our proposed methods yield state-of-the-art performance with the accuracy of 98.6% and 74.2% with the computation speed of 33 and 62 frame per second on VMRD and Cornell datasets, respectively. Our methods also yielded 95.3% grasp success rate for novel object grasping tasks with a 4-axis robot arm and 86.7% grasp success rate in cluttered novel objects with a humanoid robot.


- In-Hand Object Pose Tracking Via Contact Feedback and GPU-Accelerated Robotic Simulation

    Author: Liang, Jacky | Carnegie Mellon University
    Author: Handa, Ankur | IIIT Hyderabad
    Author: Van Wyk, Karl | NVIDIA
    Author: Makoviichuk, Viktor | NVIDIA
    Author: Kroemer, Oliver | Carnegie Mellon University
    Author: Fox, Dieter | University of Washington
 
    keyword: Perception for Grasping and Manipulation; Simulation and Animation; Force and Tactile Sensing

    Abstract : Tracking the pose of an object while it is being held and manipulated by a robot hand is difficult for vision-based methods due to significant occlusions. Prior works have explored using contact feedback and particle filters to localize in-hand objects. However, they have mostly focused on the static grasp setting and not when the object is in motion, as doing so requires explicit modeling of complex contact dynamics. In this work, we propose using GPU-accelerated parallel robot simulations and sample-based optimization algorithms to track the in-hand object pose with contact feedback during manipulation. We perform detailed ablation studies over 3 proposed optimizers in simulation, and we evaluate our method in the real world using a 4-fingered Allegro hand with SynTouch BioTac contact sensors, all mounted on a 7-DoF Kuka arm. Our algorithm runs in real-time (30Hz) on a single GPU, and it achieves an average point cloud distance error of 6mm in simulation and 13mm in the real world.

- Robust, Occlusion-Aware Pose Estimation for Objects Grasped by Adaptive Hands

    Author: Wen, Bowen | Rutgers University
    Author: Mitash, Chaitanya | Rutgers University
    Author: Soorian, Sruthi | Rutgers University
    Author: Kimmel, Andrew | Rutgers University
    Author: Sintov, Avishai | Tel-Aviv University
    Author: Bekris, Kostas E. | Rutgers, the State University of New Jersey
 
    keyword: Perception for Grasping and Manipulation; Computer Vision for Automation; RGB-D Perception

    Abstract : Many manipulation tasks, such as placement or within-hand manipulation, require the object's pose relative to a robot hand. The task is difficult when the hand significantly occludes the object. It is especially hard for adaptive hands, for which it is not easy to detect the finger's configuration. In addition, RGB-only approaches face issues with texture-less objects or when the hand and the object look similar. This paper presents a depth-based framework, which aims for robust pose estimation and short response times. The approach detects the adaptive hand's state via efficient parallel search given the highest overlap between the hand's model and the point cloud. The hand's point cloud is pruned and robust global registration is performed to generate object pose hypotheses, which are clustered. False hypotheses are pruned via physical reasoning. The remaining poses' quality is evaluated given agreement with observed data. Extensive evaluation on synthetic and real data demonstrates the accuracy and computational efficiency of the framework when applied on challenging, highly-occluded scenarios for different object types. An ablation study identifies how the framework's components help in performance. This work also provides a dataset for in-hand 6D object pose estimation. Code and dataset are available at: https://github.com/wenbowen123/icra20-hand-object-pose

- Robust 6D Object Pose Estimation by Learning RGB-D Features

    Author: Tian, Meng | National University of Singapore
    Author: Pan, Liang | National University of Singapore
    Author: Ang Jr, Marcelo H | National University of Singapore
    Author: Lee, Gim Hee | National University of Singapore
 
    keyword: Perception for Grasping and Manipulation; Object Detection, Segmentation and Categorization; RGB-D Perception

    Abstract : Accurate 6D object pose estimation is fundamental to robotic manipulation and grasping. Previous methods follow a local optimization approach which minimizes the distance between closest point pairs to handle the rotation ambiguity of symmetric objects. In this work, we propose a novel discrete-continuous formulation for rotation regression to resolve this local-optimum problem. We uniformly sample rotation anchors in SO(3), and predict a constrained deviation from each anchor to the target, as well as uncertainty scores for selecting the best prediction. Additionally, the object location is detected by aggregating point-wise vectors pointing to the 3D center. Experiments on two benchmarks: LINEMOD and YCB-Video, show that the proposed method outperforms state-of-the-art approaches. Our code is available at https://github.com/mentian/object-posenet.

- Split Deep Q-Learning for Robust Object Singulation

    Author: Sarantopoulos, Iason | Aristotle University of Thessaloniki
    Author: Kiatos, Marios | Aristotle University of Thessaloniki
    Author: Doulgeri, Zoe | Aristotle University of Thessaloniki
    Author: Malassiotis, Sotiris | Centre for Research and Technology Hellas
 
    keyword: Perception for Grasping and Manipulation; Learning and Adaptive Systems

    Abstract : Extracting a known target object from a pile of other objects in a cluttered environment is a challenging robotic manipulation task encountered in many robotic applications. In such conditions, the target object touches or is covered by adjacent obstacle objects, thus rendering traditional grasping techniques ineffective. In this paper, we propose a pushing policy aiming at singulating the target object from its surrounding clutter, by means of lateral pushing movements of both the neighboring objects and the target object until sufficient 'grasping room' has been achieved. To achieve the above goal we employ reinforcement learning and particularly Deep Q-learning (DQN) to learn optimal push policies by trial and error. A novel Split DQN is proposed to improve the learning rate and increase the modularity of the algorithm. Experiments show that although learning is performed in a simulated environment the transfer of learned policies to a real environment is effective thanks to robust feature selection. Finally, we demonstrate that the modularity of the algorithm allows the addition of extra primitives without retraining the model from scratch.

- 6-DOF Grasping for Target-Driven Object Manipulation in Clutter

    Author: Murali, Adithyavairavan | Carnegie Mellon University
    Author: Mousavian, Arsalan | NVIDIA
    Author: Eppner, Clemens | NVIDIA
    Author: Paxton, Chris | NVIDIA Research
    Author: Fox, Dieter | University of Washington
 
    keyword: Perception for Grasping and Manipulation; Grasping; RGB-D Perception

    Abstract : Grasping in cluttered environments is a fundamental but challenging robotic skill. It requires both reasoning about unseen object parts and potential collisions with the manipulator. Most existing data-driven approaches avoid this problem by limiting themselves to top-down planar grasps which is insufficient for many real-world scenarios and greatly limits possible grasps. We present a method that plans 6-DOF grasps for any desired object in a cluttered scene from partial point cloud observations. Our method achieves a grasp success of 80.3%, outperforming baseline approaches by 17.6% and clearing 9 cluttered table scenes that contain 51 objects in total on a real robotic platform. By using our learned collision checking module, we can even reason about effective grasp sequences to retrieve objects that are not immediately accessible.

- Single Shot 6D Object Pose Estimation

    Author: Kleeberger, Kilian | Fraunhofer IPA
    Author: Huber, Marco F. | University of Stuttgart
 
    keyword: Perception for Grasping and Manipulation; AI-Based Methods; Deep Learning in Robotics and Automation

    Abstract : In this paper, we introduce a novel single shot approach for 6D object pose estimation of rigid objects based on depth images. For this purpose, a fully convolutional neural network is employed, where the 3D input data is spatially discretized and pose estimation is considered as a regression task that is solved locally on the resulting volume elements. With 65 fps on a GPU, our Object Pose Network (OP-Net) is extremely fast, is optimized end-to-end, and estimates the 6D pose of multiple objects in the image simultaneously. Our approach does not require manually 6D pose-annotated real-world datasets and transfers to the real world, although being entirely trained on synthetic data. The proposed method is evaluated on public benchmark datasets, where we can demonstrate that state-of-the-art methods are significantly outperformed.

## Humanoid Robots

- HRP-4 Walks on Soft Feet

    Author: Catalano, Manuel Giuseppe | Istituto Italiano Di Tecnologia
    Author: Frizza, Irene | University of Pisa
    Author: Morandi, Cecilia | University of Pisa
    Author: Grioli, Giorgio | Istituto Italiano Di Tecnologia
    Author: Ayusawa, Ko | AIST
    Author: Ito, Takahiro | University of Tsukuba
    Author: Venture, Gentiane | Tokyo University of Agriculture and Technology
 
    keyword: Legged Robots; Humanoid and Bipedal Locomotion; Humanoid Robots

    Abstract : The majority of humanoid robots adopt flat feet, a choice that can limit their performance when maneuvering over uneven terrains. Recently, a soft robotic foot designed to adapt to the ground was proposed to overcome part of these limitations. This paper presents the results of testing two such feet on the humanoid robot HRP-4, and compares them to what obtained with the original flat feet of the robot. After describing the SoftFoot and how it has been adapted to the robot, the biped is tested while balancing, stepping and walking. Tests are carried out on flat ground and on obstacles of different heights. For comparison purposes, the original HRP-4 controller has been used for both types of feet with no changes (except for re-evaluation of the CoM position). Analysis of the ankle pitch angle, ankle pitch torque, knee pitch angle, knee pitch torque, waist roll angle and waist pitch angle, show a substantial improvement in obstacle negotiation performance of HRP-4, when using the SoftFoot, even without optimizing the controller to exploit the SoftFoot features.

- A Study on Sparse Hierarchical Inverse Kinematics Algorithms for Humanoid Robots

    Author: Mingo Hoffman, Enrico | Fondazione Istituto Italiano Di Tecnologia
    Author: Parigi Polverini, Matteo | Istituto Italiano Di Tecnologia (IIT)
    Author: Laurenzi, Arturo | Istituto Italiano Di Tecnologia
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
 
    keyword: Humanoid Robots; Optimization and Optimal Control; Kinematics

    Abstract : In humanoid robotic platforms, classical inverse kinematics algorithms based on L2-regularisation of joint velocities or accelerations, tends to engage the motion of all the available degrees of freedom, resulting in movements of the whole robot structure, which are inherently not sparse. The role of sparsity in motion control has recently gained interest in the robotics community for various reasons, e.g. human-like motions, human-robot interaction, actuation parsimony, yet an exhaustive mathematical analysis is still missing. In order to address this topic, we here propose and compare possible sparse optimization approaches applied to hierarchical inverse kinematics for humanoid robots. This is achieved through LASSO regression and MILP optimization to resolve the IK problem. A first order formulation of the sparse regression problem is further introduced to reduce chattering on the joint velocity profiles. This paper presents the theory behind the proposed approaches and performs a comparison analysis based on simulated and real experiments on different humanoid platforms.

- Inferring the Geometric Nullspace of Robot Skills from Human Demonstrations

    Author: Cai, Caixia | Institute for Infocomm Research, A*STAR
    Author: Liang, Ying Siu | Agency for Science, Technology and Research (A*STAR)
    Author: Somani, Nikhil | Agency for Science, Technology and Research (A*STAR)
    Author: Wu, Yan | A*STAR Institute for Infocomm Research
 
    keyword: Humanoid Robots; Behavior-Based Systems; Learning from Demonstration

    Abstract : In this paper we present a framework to learn skills from human demonstrations in the form of geometric nullspaces, which can be executed using a robot. We collect data of human demonstrations, fit geometric nullspaces to them, and also infer their corresponding geometric constraint models. These geometric constraints provide a powerful mathematical model as well as an intuitive representation of the skill in terms of the involved objects. To execute the skill using a robot, we combine this geometric skill description with the robot's kinematics and other environmental constraints, from which poses can be sampled for the robot's execution. The result of our framework is a system that takes the human demonstrations as input, learns the underlying skill model, and executes the learnt skill with different robots in different dynamic environments. We evaluate our approach on a simulated industrial robot, and execute the final task on the iCub humanoid robot.

- A Dynamical System Approach for Adaptive Grasping, Navigation and Co-Manipulation with Humanoid Robots

    Author: Figueroa, Nadia | Massachusetts Institute of Technology (MIT)
    Author: Faraji, Salman | EPFL
    Author: Koptev, Mikhail | École Polytechnique Fédérale De Lausanne
    Author: Billard, Aude | EPFL
 
    keyword: Motion and Path Planning; Humanoid Robots; Compliance and Impedance Control

    Abstract : In this paper, we present an integrated approach that provides compliant control of an iCub humanoid robot and adaptive reaching, grasping, navigating and co-manipulating capabilities. We use state-dependent dynamical systems (DS) to (i) coordinate and drive the robot's hands (in both position and orientation) to grasp an object using an intermediate virtual object, and (ii) drive the robot's base while walking/navigating. The use of DS as motion generators allows us to adapt smoothly as the object moves and to re-plan on-line motion of the arms and body to reach the object's new location. The desired motion generated by the DS are used in combination with a whole-body compliant control strategy that absorbs perturbations while walking and offers compliant behaviors for grasping and manipulation tasks. Further, the desired dynamics for the arm and body can be learned from demonstrations. By integrating these components, we achieve unprecedented adaptive behaviors for whole body manipulation. We showcase this in simulations and real-world experiments where iCub robots (i) walk-to-grasp objects, (ii) follow a human (or another iCub) through interaction and (iii) learn to navigate or co-manipulate an object from human guided demonstrations; whilst being robust to changing targets and perturbations.

- Humanoid Robots in Aircraft Manufacturing (I)
 
    Author: Kheddar, Abderrahmane | CNRS-AIST JRL (Joint Robotics Laboratory), UMI3218/CRT
    Author: Caron, Stephane | ANYbotics AG
    Author: Gergondet, Pierre | CNRS
    Author: Comport, Andrew Ian | CNRS-I3S/UNS
    Author: Tanguy, Arnaud | CNRS-UM LIRMM
    Author: Ott, Christian | German Aerospace Center (DLR)
    Author: Henze, Bernd | Agile Robots AG
    Author: Mesesan, George | German Aerospace Center (DLR)
    Author: Englsberger, Johannes | DLR (German Aerospace Center)
    Author: Roa, Maximo A. | DLR - German Aerospace Center
    Author: Wieber, Pierre-Brice | INRIA Rh�ne-Alpes
    Author: Chaumette, Francois | Inria Rennes-Bretagne Atlantique
    Author: Spindler, Fabien | INRIA
    Author: Oriolo, Giuseppe | Sapienza University of Rome
    Author: Lanari, Leonardo | Sapienza University of Rome
    Author: Escande, Adrien | AIST
    Author: Chappellet, Kevin | CNRS
    Author: Kanehiro, Fumio | National Inst. of AIST
    Author: Rabate, Patrice | Airbus Group
 
    keyword: Humanoid Robots; Industrial Robots; Additive Manufacturing

    Abstract : We report results from a collaborative project that investigated the deployment of humanoid robotic solutions in aircraft manufacturing for some assembly operations where access is not possible for wheeled or rail-ported robotic platforms. Recent developments in multi-contact planning and control, bipedal walking, embedded SLAM, whole-body multi-sensory task space optimization control, and contact detection and safety, suggest that humanoids could be a plausible solution for automation given the specific requirements in such large-scale manufacturing sites. The main challenge is to integrate these scientific and technological advances into two existing humanoid platforms: the position controlled HRP-4 and the torque controlled TORO. This integration effort was demonstrated in a bracket assembly operation inside a 1:1 scale A350 mock-up of the front part of the fuselage at the Airbus Saint-Nazaire site. We present and discuss the main results that have been achieved in this project and provide recommendations for future work.

- A Multi-Mode Teleoperation Framework for Humanoid Loco-Manipulation (I)

    Author: Penco, Luigi | INRIA
    Author: Scianca, Nicola | Sapienza University of Rome
    Author: Modugno, Valerio | Sapienza Université Di Roma
    Author: Lanari, Leonardo | Sapienza University of Rome
    Author: Oriolo, Giuseppe | Sapienza University of Rome
    Author: Ivaldi, Serena | INRIA
 
    keyword: Telerobotics and Teleoperation; Humanoid Robots; Human-Centered Robotics

    Abstract : Every year millions of people die due to work-related diseases or are severely injured as a result of accidents on the workplace. The introduction of humanoid	robots in the work environment can help us reduce the occurrence of such dramatic events. Thanks to their dexterity and agility, humanoids have a number of advantages over other kinds of mobile robots. They can move more easily in cluttered and human-centered environments, as well as unstructured and unpredictable environments such as disaster-response scenarios. %We propose a multi-mode teleoperation framework for controlling humanoid robots for loco-manipulation tasks. %One mode allows the operator to fully control the robot by wearing a motion capture suit, and thus having the robot replicate human movements in real-time. The other is a semi-autonomous control mode, where the teleoperator can give high-level directives to the robot about the task to be performed. We tested our framework on a real iCub robot for a whole body pick and place demo application.

- Balance of Humanoid Robots in a Mix of Fixed and Sliding Multi-Contact Scenarios

    Author: Samadi, Saeid | University of Montpellier
    Author: Caron, Stephane | ANYbotics AG
    Author: Tanguy, Arnaud | CNRS-UM LIRMM
    Author: Kheddar, Abderrahmane | CNRS-AIST JRL (Joint Robotics Laboratory), UMI3218/CRT
 
    keyword: Humanoid Robots; Dynamics; Force Control

    Abstract : This study deals with the balance of humanoid or multi-legged robots in a multi-contact setting where a chosen subset of contacts is undergoing desired sliding-task motions. One method to keep balance is to hold the center-of-mass (CoM) within an admissible convex area. This area is calculated based on the contact positions and forces. We introduce a methodology to compute this CoM support area (CSA) for multiple fixed and intentionally sliding contacts. To select the most appropriate CoM position within CSA, we account for (i) constraints of multiple fixed and sliding contacts, (ii) desired wrench distribution for contacts, and (iii) desired CoM position (eventually dictated by other tasks). These are formulated as a quadratic programming (QP) optimization problems. We illustrate our approach with pushing against a wall and wiping, and conducted experiments using the HRP-4 humanoid robot.

- Fast Whole-Body Motion Control of Humanoid Robots with Inertia Constraints

    Author: Ficht, Grzegorz | University of Bonn
    Author: Behnke, Sven | University of Bonn
 
    keyword: Humanoid Robots; Kinematics; Dynamics

    Abstract : We introduce a new, analytical method for generating whole body motions for humanoid robots, which approximate the desired Composite Rigid Body (CRB) inertia. Our approach uses a reduced five mass model, where four of the masses are attributed to the limbs and one is used for the trunk. This compact formulation allows for finding an analytical solution that combines the kinematics with mass distribution and inertial properties of a humanoid robot. The positioning of the masses in Cartesian space is then directly used to obtain joint angles with relations based on simple geometry. Motions are achieved through the time evolution of poses generated through the desired foot positioning and CRB inertia properties. As a result, we achieve short computation times in the order of tens of microseconds. This makes the method suited for applications with limited computation resources, or leaving them to be spent on higher-layer tasks such as model predictive control. The approach is evaluated by performing a dynamic kicking motion with an igus Humanoid Open Platform robot.

- SL1M: Sparse L1-Norm Minimization for Contact Planning on Uneven Terrain

    Author: Tonneau, Steve | The University of Edinburgh
    Author: Song, Daeun | Ewha Womans University
    Author: Fernbach, Pierre | Cnrs - Laas
    Author: Mansard, Nicolas | CNRS
    Author: Ta�x, Michel | LAAS-CNRS/Université Paul Sabatier
    Author: Del Prete, Andrea | Max Planck Institute for Intelligent Systems
 
    keyword: Humanoid and Bipedal Locomotion; Optimization and Optimal Control; Humanoid Robots

    Abstract : One of the main challenges of planning legged locomotion in complex environments is the combinatorial contact selection problem. Recent contributions propose to use integer variables to represent which contact surface is selected, and then to rely on modern mixed-integer (MI) optimization solvers to handle this combinatorial issue. To reduce the computational cost of MI, we exploit the sparsity properties of L1 norm minimization techniques to relax the contact planning problem into a feasibility linear program. Our approach accounts for kinematic reachability of the center of mass (COM) and of the contact effectors. We ensure the existence of a quasi-static COM trajectory by restricting our plan to quasi-flat contacts. For planning 10 steps with less than 10 potential contact surfaces for each phase, our approach is 50 to 100 times faster that its MI counterpart, which suggests potential applications for online contact re-planning. The method is demonstrated in simulation with the humanoid robots HRP-2 and Talos over various scenarios.

- Finding Locomanipulation Plans Quickly in the Locomotion Constrained Manifold

    Author: Jorgensen, Steven Jens | The University of Texas at Austin
    Author: Vedantam, Mihir | University of Texas at Austin
    Author: Gupta, Ryan | University of Texas at Austin
    Author: Cappel, Henry | University of Texas at Austin
    Author: Sentis, Luis | The University of Texas at Austin
 
    keyword: Humanoid Robots; Manipulation Planning; Humanoid and Bipedal Locomotion

    Abstract : We present a method that finds locomanipulation plans that perform simultaneous locomotion and manipulation of objects for a desired end-effector trajectory. Key to our approach is to consider an injective locomotion constraint manifold that defines the locomotion scheme of the robot and then using this constraint manifold to search for admissible manipulation trajectories. The problem is formulated as a weighted-A* graph search whose planner output is a sequence of contact transitions and a path progression trajectory to construct the whole-body kinodynamic locomanipulation plan. We also provide a method for computing, visualizing, and learning the locomanipulability region, which is used to efficiently evaluate the edge transition feasibility during the graph search. Numerical simulations are performed with the NASA Valkyrie robot platform that utilizes a dynamic locomotion approach, called the divergent-component-of-motion (DCM), on two example locomanipulation scenarios.

- Force-Based Control of Bipedal Balancing on Dynamic Terrain with the "Tallahassee Cassie" Robotic Platform

    Author: White, Jason | Florida State University
    Author: Swart, Dylan | Florida State University
    Author: Hubicki, Christian | Florida State University
 
    keyword: Legged Robots; Humanoid and Bipedal Locomotion; Humanoid Robots

    Abstract : Out in the field, bipedal robots need to travel on terrain that is uneven, non-rigid, and sometimes moving beneath its feet. We present a simple force-based balancing controller for such dynamic terrain scenarios for bipedal robots, and test it on the robotic bipedal platform ``Tallahassee Cassie.'' The presented controller relies on minimal information about the robot model, requiring its kinematics and overall weight, but not inertias of individual links or components. The controller is pelvis-centric, commanding pelvis positions in Cartesian space, which a model-free PD controller converts to motor torques in joint space. By commanding forces, torques, and a frontal pressure center in this simple fashion, Tallahassee Cassie is capable of balancing on a variety of dynamic terrain scenarios, from a lifting/sliding platform, to soft foam, to a sudden drop. These results show the potential for bipedal control to balance successfully despite minimal model information, the presence of large dynamic impacts (e.g. falling through trap door), and soft series-spring deflections. These results motivate future work for simple walking and running controllers on dynamic terrain with relatively low reliance on modeling information.

- Simultaneous Control Framework for Humanoid Tracking Human Movement with Interacting Wearable Assistive Device

    Author: Ito, Takahiro | University of Tsukuba
    Author: Ayusawa, Ko | AIST
    Author: Yoshida, Eiichi | National Inst. of AIST
    Author: Kobayashi, Hiroshi | Tokyo University of Science
 
    keyword: Humanoid Robots; Physically Assistive Devices; Motion Control

    Abstract : Instead of human subjects, humanoid robots can be used as human dummies to test the human-designed products. We propose a controller that uses wearable assistive devices (also referred to as exoskeletons) to reproduce human movement in the evaluation. The proposed control scheme consists two components: one is the torque controller designed for a simplified interaction model with the device, and the other is the tracking controller based on a vector field to reproduce human motion. We implemented the proposed controller on the human-sized humanoid HRP-4 and validated the feasibility of the human motion reproduction by wearing the assistive device. In the experiment, we tested the commercially available device "Muscle Suit" by using our control scheme. The experimental results showed that while the device applies its supporting strength, the humanoid robot could reproduce human movements. The assistive effect of the device was visualized effectively in our evaluation framework.

## Force Control
- Dynamic Control of a Rigid Pneumatic Gripper

    Author: Romeo, Rocco Antonio | Istituto Italiano Di Tecnologia
    Author: Fiorio, Luca | Istituto Italiano Di Tecnologia
    Author: L'Erario, Giuseppe | Istituto Italiano Di Tecnologia
    Author: Maggiali, Marco | Italian Institute of Technology
    Author: Metta, Giorgio | Istituto Italiano Di Tecnologia (IIT)
    Author: Pucci, Daniele | Italian Institute of Technology
 
    keyword: Force Control; Grippers and Other End-Effectors; Sensor-based Control

    Abstract : Pneumatic grippers are hugely employed in robotic applications. Nonetheless, their control is not easy due to difficulty in managing the pressure inside their air chambers. Pneumatic grippers have often simple structure though the lack of affordable control algorithms complicates their usage. Motivated by these reasons, we wish to deliver a new control architecture for the closed-loop control of pneumatic grippers actuated by pressure regulators. The proposed architecture is composed of a main controller resorting on an optimization algorithm and of a state observer that estimates pressures in both gripper chambers, along with the exerted force. Instead, measured quantities (i.e. physical pressure in the gripper chambers and force recorded by a load cell between the gripper fingers) are used as inputs for the state observer to improve its output. The pneumatic gripper performance benefits from the joint action of the controller and of the state observer, as experimentally demonstrated. The gripper response will be shown for different types of inputs and on different setups.

- A Control Framework Definition to Overcome Position/Interaction Dynamics Uncertainties in Force-Controlled Tasks

    Author: Roveda, Loris | SUPSI-IDSIA
    Author: Castaman, Nicola | University of Padova
    Author: Franceschi, Paolo | CNR-STIIMA
    Author: Ghidoni, Stefano | University of Padova
    Author: Pedrocchi, Nicola | National Research Council of Italy (CNR)
 
    keyword: Force Control; Intelligent and Flexible Manufacturing; RGB-D Perception

    Abstract : Within the Industry 4.0 context, industrial robots have to implement increasing autonomy. The manipulator has to be capable to react to uncertainties/changes in the working environment, displaying a robust behavior. In this paper, a control framework is proposed to perform industrial interaction tasks in uncertain working scenes. The proposed methodology relies on two components: i) a 6D pose estimation algorithm aiming to recognize large and featureless parts; ii) a variable damping impedance controller (inner loop) enhanced by an adaptive saturation PI (outer loop) for high accuracy force control (i.e., zero steady-state force error and force overshoots avoidance). The proposed methodology allows to be robust with respect to task uncertainties (i.e., positioning errors and interaction dynamics). The proposed approach has been evaluated in an assembly task of a side-wall panel to be installed inside the aircraft cabin. As a test platform, the KUKA iiwa 14 R820 has been used, together with the Microsoft Kinect 2.0 as RGB-D sensor. Experiments show the reliability in the 6D pose estimation and the high-performance in the force-tracking task, avoiding force overshoots while achieving the tracking of the reference force.

- Identification of Compliant Contact Parameters and Admittance Force Modulation on a Non-Stationary Compliant Surface

    Author: Wijayarathne, Lasitha | Georgia Institute of Technology
    Author: Hammond III, Frank L. | Georgia Institute of Technology
 
    keyword: Force Control; Robust/Adaptive Control of Robotic Systems; Motion Control of Manipulators

    Abstract : Although autonomous control of robotic manipulators has been studied for several decades, they are not commonly used in safety-critical applications due to lack of safety and performance guarantees - many of them concerning the modulation of interaction forces. This paper presents a mechanical probing strategy for estimating the environmental impedance parameters of compliant environments, independent a manipulator's controller design and configuration. The parameter estimates are used in a position-based adaptive force controller to enable control of interaction forces in compliant, stationary and non-stationary environments. This approach is targeted for applications where the workspace is constrained and non-stationary, and where force control is critical to task success. These applications include surgical tasks involving manipulation of compliant, delicate, moving tissues. Results show fast parameter estimation and successful force modulation that compensates for motion.

- Convex Controller Synthesis for Robot Contact

    Author: Pham, Hung | Nanyang Technological University
    Author: Pham, Quang-Cuong | NTU Singapore
 
    keyword: Force Control; Physical Human-Robot Interaction; Compliance and Impedance Control

    Abstract : Controlling contacts is truly challenging, and this has been a major hurdle to deploying industrial robots into unstructured/human-centric environments. More specifically, the main challenges are: (i) how to ensure stability at all times; (ii) how to satisfy task-specific performance specifications; (iii) how to achieve (i) and (ii) under environment uncertainty, robot parameters uncertainty, sensor and actuator time delays, external perturbations, etc. Here, we propose a new approach -- Convex Controller Synthesis (CCS) -- to tackle the above challenges based on robust control theory and convex optimization. In two physical interaction tasks -- robot hand guiding and sliding on surfaces with different and unknown stiffnesses -- we show that CCS controllers outperform their classical counterparts in an essential way.

- Force Adaptation in Contact Tasks with Dynamical Systems

    Author: Amanhoud, Walid | EPFL
    Author: Khoramshahi, Mahdi | EPFL
    Author: Bonnesoeur, Maxime | EPFL
    Author: Billard, Aude | EPFL
 
    keyword: Force Control; Compliance and Impedance Control; Physical Human-Robot Interaction

    Abstract : In many tasks such as finishing operations, achieving accurate force tracking is essential. However, uncertainties in the robot dynamics and the environment limit the force tracking accuracy. Learning a compensation model for these uncertainties to reduce the force error is an effective approach to overcome this limitation. However, this approach requires an adaptive and robust framework for motion and force generation. In this paper, we use the time-invariant Dynamical System (DS) framework for force adaptation in contact tasks. We propose to improve force tracking accuracy through online adaptation of a state-dependent force correction model encoded with Radial Basis Functions (RBFs). We evaluate our method with a KUKA LWR IV+ robotic arm. We show its efficiency to reduce the force error to a negligible amount with different target forces and robot velocities. Furthermore, we study the effect of the hyper-parameters and provide a guideline for their selection. We showcase a collaborative cleaning task with a human by integrating our method to previous works to achieve force, motion, and task adaptation at the same time. Thereby, we highlight the benefits of using adaptive force control in real-world environments where we need reactive and adaptive behaviours in response to interactions with the environment.

- Sensitivity Ellipsoids for Force Control of Magnetic Robots with Localization Uncertainty (I)

    Author: Slawinski, Piotr | Vanderbilt University
    Author: Simaan, Nabil | Vanderbilt University
    Author: Taddese, Addisu | Vanderbilt University
    Author: Obstein, Keith | Vanderbilt University
    Author: Valdastri, Pietro | University of Leeds
 
    keyword: Force Control; Localization; Medical Robots and Systems

    Abstract : The navigation of magnetic medical robots typically relies on localizing an actuated, intracorporeal, ferromagnetic body and back-computing a necessary field and gradient that would result in a desired wrench on the device. Uncertainty in this localization degrades the precision of force transmission. Reducing applied force uncertainty may enhance tasks such as in-vivo navigation of miniature robots, actuation of magnetically guided catheters, tissue palpation, as well as simply ensuring a bound on forces applied on sensitive tissue. In this paper, we analyzed the effects of localization noise on force uncertainty by using sensitivity ellipsoids of the magnetic force Jacobian and introduced an algorithm for uncertainty reduction. We validated the algorithm in both a simulation study and in a physical experiment. In simulation, we observed reductions in estimated force uncertainty by factors of up to 2.8 and 3.1 when using one and two actuating magnets, respectively. On a physical platform, we demonstrated a force uncertainty reduction by a factor of up to 2.5 as measured using an external sensor. Being the first consideration of force uncertainty resulting from noisy localization, this work provides a strategy for investigators to minimize uncertainty in magnetic force transmission.

## Semantic Scene Understanding

- Highly Parallelizable Plane Extraction for Organized Point Clouds Using Spherical Convex Hulls

    Author: M�ls, Hannes | Intelligent Sensor-Actuator-Systems Lab (ISAS), Karlsruhe Instit
    Author: Li, Kailai | Karlsruhe Institute of Technology (KIT)
    Author: Hanebeck, Uwe D. | Karlsruhe Institute of Technology (KIT)
 
    keyword: Semantic Scene Understanding; Computer Vision for Other Robotic Applications

    Abstract : We present a novel region growing algorithm for plane extraction of organized point clouds using the spherical convex hull. Instead of explicit plane parameterization, our approach interprets potential underlying planes as a series of geometric constraints on the sphere that are refined during region growing. Unlike existing schemes relying on downsampling for sequential execution in real time, our approach enables pixelwise plane extraction that is highly parallelizable. We further test the proposed approach with a fully parallel implementation on a GPU. Evaluation based on public data sets has shown state-of-the-art extraction accuracy and superior speed compared to existing approaches, while guaranteeing real-time processing at full input resolution of a typical RGB-D camera.

- Boosting Real-Time Driving Scene Parsing with Shared Semantics

    Author: Xiang, Zhenzhen | Shanghai Jiao Tong University
    Author: Bao, Anbo | Shanghai Jiao Tong University
    Author: Li, Jie | SAIC Motor
    Author: Su, Jianbo | Shanghai Jiao Tong University
 
    keyword: Semantic Scene Understanding; Autonomous Vehicle Navigation

    Abstract : Real-time scene parsing is a fundamental feature for autonomous driving vehicles with multiple cameras. In this letter we demonstrate that sharing semantics between cameras with different perspectives and overlapping views can boost the parsing performance when compared with traditional methods, which individually process the frames from each camera. Our framework is based on a deep neural network for semantic segmentation but with two kinds of additional modules for sharing and fusing semantics. On the one hand, a semantics sharing module is designed to establish the pixel-wise mapping between the input images. Features as well as semantics are shared by the map to reduce duplicated workload, which leads to more efficient computation. On the other hand, feature fusion modules are designed to combine different modalities of semantic features, which leverages the information from both inputs for better accuracy. To evaluate the effectiveness of the proposed framework, we have applied our network to a dual-camera vision system for driving scene parsing. Experimental results show that our network outperforms the baseline method on the parsing accuracy with comparable computations.

- CNN-Based Lidar Point Cloud De-Noising in Adverse Weather

    Author: Heinzler, Robin | Daimler AG
    Author: Piewak, Florian | Daimler AG
    Author: Schindler, Philipp | Daimler AG
    Author: Stork, Wilhelm | FZI Karlsruhe
 
    keyword: Semantic Scene Understanding; Visual Learning; Computer Vision for Transportation

    Abstract : Lidar sensors are frequently used in environment perception for autonomous vehicles and mobile robotics to com- plement camera, radar, and ultrasonic sensors. Adverse weather conditions are significantly impacting the performance of lidar- based scene understanding by causing undesired measurement points that in turn effect missing detections and false positives. In heavy rain or dense fog, water drops could be misinterpreted as objects in front of the vehicle which brings a mobile robot to a full stop. In this paper, we present the first CNN-based approach to understand and filter out such adverse weather effects in point cloud data. Using a large data set obtained in controlled weather environments, we demonstrate a significant performance improvement of our method over state-of-the-art involving geometric filtering.

- View-Invariant Loop Closure with Oriented Semantic Landmarks

    Author: Li, Jimmy | McGill University
    Author: Koreitem, Karim | McGill University
    Author: Meger, David Paul | McGill University
    Author: Dudek, Gregory | McGill University
 
    keyword: Semantic Scene Understanding; Visual-Based Navigation; SLAM

    Abstract : Recent work on semantic simultaneous localization and mapping (SLAM) have shown the utility of natural objects as landmarks for improving localization accuracy and robustness. In this paper we present a monocular semantic SLAM system that uses object identity and inter-object geometry for view-invariant loop detection and drift correction. Our system's ability to recognize an area of the scene even under large changes in viewing direction allows it to surpass the mapping accuracy of ORB-SLAM, which uses only local appearance-based features that are not robust to large viewpoint changes. Experiments on real indoor scenes show that our method achieves mean drift reduction of 70% when compared directly to ORB-SLAM. Additionally, we propose a method for object orientation estimation, where we leverage the tracked pose of a moving camera under the SLAM setting to overcome ambiguities caused by object symmetry. This allows our SLAM system to produce geometrically detailed semantic maps with object orientation, translation, and scale.

- Semantic Foreground Inpainting from Weak Supervision

    Author: Lu, Chenyang | Eindhoven University of Technology
    Author: Dubbelman, Gijs | Eindhoven University of Technology
 
    keyword: Semantic Scene Understanding; Computer Vision for Transportation

    Abstract : Semantic scene understanding is an essential task for self-driving vehicles and mobile robots. In our work, we aim to estimate a semantic segmentation map, in which the foreground objects are removed and semantically inpainted with background classes, from a single RGB image. This semantic foreground inpainting task is performed by a single-stage convolutional neural network (CNN) that contains our novel max-pooling as inpainting (MPI) module, which is trained with weak supervision, i.e., it does not require manual background annotations for the foreground regions to be inpainted. Our approach is inherently more efficient than the previous two-stage state-of-the-art method, and outperforms it by a margin of 3% IoU for the inpainted foreground regions on Cityscapes. The performance margin increases to 6% IoU, when tested on the unseen KITTI dataset. The code and the manually annotated datasets for testing are shared with the research community at https://github.com/Chenyang-Lu/semantic-foreground-inpainting.

- Fast Panoptic Segmentation Network

    Author: de Geus, Daan | Eindhoven University of Technology
    Author: Meletis, Panagiotis | Eindhoven University of Technology
    Author: Dubbelman, Gijs | Eindhoven University of Technology
 
    keyword: Semantic Scene Understanding; Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation

    Abstract : In this work, we present an end-to-end network for fast panoptic segmentation. This network, called Fast Panoptic Segmentation Network (FPSNet), does not require computationally costly instance mask predictions or rule-based merging operations. This is achieved by casting the panoptic task into a custom dense pixel-wise <i>classification</i> task, which assigns a class label or an instance <i>id</i> to each pixel. We evaluate FPSNet on the Cityscapes and Pascal VOC datasets, and find that FPSNet is faster than existing panoptic segmentation methods, while achieving better or similar panoptic segmentation performance. On the Cityscapes validation set, we achieve a Panoptic Quality score of 55.1%, at prediction times of 114 milliseconds for images with a resolution of 1024x2048 pixels. For lower resolutions of the Cityscapes dataset and for the Pascal VOC dataset, FPSNet achieves prediction times as low as 45 and 28 milliseconds, respectively.

- Weakly Supervised Silhouette-Based Semantic Scene Change Detection

    Author: Sakurada, Ken | National Institute of Advanced Industrial Science and Technology
    Author: Shibuya, Mikiya | Tokyo Institute of Technology University
    Author: Wang, Weimin | National Institute of Advanced Industrial Science and Technology
 
    keyword: Semantic Scene Understanding; Object Detection, Segmentation and Categorization; Recognition

    Abstract : This paper presents a novel semantic scene change detection scheme with only weak supervision. A straightforward approach for this task is to train a semantic change detection network directly from a large-scale dataset in an end-to-end manner. However, a specific dataset for this task, which is usually labor-intensive and time-consuming, becomes indispensable. To avoid this problem, we propose to train this kind of network from existing datasets by dividing this task into change detection and semantic extraction. On the other hand, the difference in camera viewpoints, for example, images of the same scene captured from a vehicle-mounted camera at different time points, usually brings a challenge to the change detection task. To address this challenge, we propose a new siamese network structure with the introduction of correlation layer. In addition, we create a publicly available dataset for semantic change detection to evaluate the proposed method. The experimental results verified both the robustness to viewpoint difference in change detection task and the effectiveness for semantic change detection of the proposed networks. Our code and dataset are available at https://github.com/xdspacelab/sscdnet.

- 3DCFS: Fast and Robust Joint 3D Semantic-Instance Segmentation Via Coupled Feature Selection

    Author: Du, Liang | Fudan University
    Author: Tan, Jingang | Shanghai Institute of Microsystem and Information Technology, Uni
    Author: Xue, Xiangyang | Fudan University
    Author: Chen, Lili | Shanghai Institute of Microsystem and Information Technology, Ch
    Author: Wen, Hongkai | University of Warwick
    Author: Feng, Jianfeng | Fudan University
    Author: Li, Jiamao | Shanghai Institute of Microsystem and Information Technology, Chi
    Author: Zhang, Xiaolin | Shanghai Institute of Microsystem and Information Technology, Chi
 
    keyword: Semantic Scene Understanding; RGB-D Perception; Object Detection, Segmentation and Categorization

    Abstract : We propose a novel fast and robust 3D point clouds segmentation framework via coupled feature selection, named 3DCFS, that jointly performs semantic and instance segmentation. Inspired by the human scene perception process, we design a novel coupled feature selection module, named CFSM, that adaptively selects and fuses the reciprocal semantic and instance features from two tasks in a coupled manner. To further boost the performance of the instance segmentation task in our 3DCFS, we investigate a loss function that helps the model learn to balance the magnitudes of the output embedding dimensions during training, which makes calculating the Euclidean distance more reliable and enhances the generalizability of the model. Extensive experiments demonstrate that our 3DCFS outperforms state-of-the-art methods on benchmark datasets in terms of accuracy, speed and computational cost. Codes are available at: https://github.com/Biotan/3DCFS.

- Who2com: Collaborative Perception Via Learnable Handshake Communication

    Author: Liu, Yen-Cheng | Georgia Tech
    Author: Tian, Junjiao | Georgia Institute of Technology
    Author: Ma, Chih-Yao | Georgia Tech
    Author: Glaser, Nathaniel | Georgia Institute of Technology
    Author: Kuo, Chia-Wen | Georgia Institute of Technology
    Author: Kira, Zsolt | Georgia Institute of Technology
 
    keyword: Semantic Scene Understanding; Object Detection, Segmentation and Categorization; Networked Robots

    Abstract : In this paper, we propose the problem of collaborative perception, where robots can combine their local observations with those of neighboring agents in a learnable way to improve accuracy on a perception task. Unlike existing work in robotics and multi-agent reinforcement learning, we formulate the problem as one where learned information must be shared across a set of agents in a bandwidth-sensitive manner to optimize for scene understanding tasks such as semantic segmentation. Inspired by networking communication protocols, we propose a handshake communication mechanism where the neural network can learn to compress relevant information needed for each stage. Specifically, a target agent with degraded sensor data sends a compressed request, the other agents respond with matching scores, and the target agent determines who to connect with(i.e., receive information from). We additionally develop the dataset and metrics based on the AirSim simulator where a group of aerial robots perform navigation and search missions over diverse landscapes, such as roads, grasslands, buildings, lakes, etc. We show that for the semantic segmentation task, our handshake communication method significantly improves accuracy by approximately 20% over decentralized baselines, and is comparable to centralized ones using a quarter of the bandwidth.

- Comparing View-Based and Map-Based Semantic Labelling in Real-Time SLAM

    Author: Landgraf, Zoe | Imperial College London
    Author: Falck, Fabian | Imperial College London
    Author: Bloesch, Michael | Imperial College
    Author: Leutenegger, Stefan | Imperial College London
    Author: Davison, Andrew J | Imperial College London
 
    keyword: Semantic Scene Understanding; RGB-D Perception; SLAM

    Abstract : Generally capable Spatial AI systems must build persistent scene representations where geometric models are combined with meaningful semantic labels. The many approaches to labelling scenes can be divided into two clear groups: emph{view-based} which estimate labels from the input view-wise data and then incrementally fuse them into the scene model as it is built; and emph{map-based} which label the generated scene model. However, there has so far been no attempt to quantitatively compare view-based and map-based labelling. Here, we present an experimental framework and comparison which uses real-time height map fusion as an accessible platform for a fair comparison, opening up the route to further systematic research in this area.

- Generative Modeling of Environments with Scene Grammars and Variational Inference

    Author: Izatt, Gregory | MIT
    Author: Tedrake, Russ | Massachusetts Institute of Technology
 
    keyword: Semantic Scene Understanding

    Abstract : In order to understand how a robot will perform in the open world, we aim to establish a quantitative understanding of the distribution of environments that a robot will face when when it is deployed. However, even restricting attention only to the distribution of objects in a scene, these distributions over environments are nontrivial: they describe mixtures of discrete and continuous variables related to the number, type, poses, and attributes of objects in the scene. We describe a probabilistic generative model that uses scene trees to capture hierarchical relationships between collections of objects, as well as a variational inference algorithm for tuning that model to best match a set of observed environments without any need for tediously labeled parse trees. We demonstrate that this model can accurately capture the distribution of a pair of nontrivial manipulation-relevant datasets and be deployed as a density estimator and outlier detector for novel environments.

- SHOP-VRB: A Visual Reasoning Benchmark for Object Perception

    Author: Nazarczuk, Michal | Imperial College London
    Author: Mikolajczyk, Krystian | University of Surrey
 
    keyword: Semantic Scene Understanding; Object Detection, Segmentation and Categorization; Computer Vision for Automation

    Abstract : In this paper we present an approach and a benchmark for visual reasoning in robotics applications, in particular small object grasping and manipulation. The approach and benchmark are focused on inferring object properties from visual and text data. It concerns small household objects with their properties, functionality, natural language descriptions as well as question-answer pairs for visual reasoning queries along with their corresponding scene semantic representations. We also present a method for generating synthetic data which allows to extend the benchmark to other objects or scenes and propose an evaluation protocol that is more challenging than in the existing datasets. We propose a reasoning system based on symbolic program execution. A disentangled representation of the visual and textual inputs is obtained and used to execute symbolic programs that represent a 'reasoning process' of the algorithm. We perform a set of experiments on the proposed benchmark and compare to results for the state of the art methods. These results expose the shortcomings of the existing benchmarks that may lead to misleading conclusions on the actual performance of the visual reasoning systems.

## Social Human-Robot Interaction
- Simultaneous Learning from Human Pose and Object Cues for Real-Time Activity Recognition

    Author: Reily, Brian | Colorado School of Mines
    Author: Zhu, Qingzhao | Colorado School of Mines
    Author: Reardon, Christopher M. | U.S. Army Research Laboratory
    Author: Zhang, Hao | Colorado School of Mines
 
    keyword: Human-Centered Robotics; Social Human-Robot Interaction; Computer Vision for Other Robotic Applications

    Abstract : Real-time human activity recognition plays an essential role in real-world human-centered robotics applications, such as assisted living and human-robot collaboration. Although previous methods based on skeletal data to encode human poses showed promising results on real-time activity recognition, they lacked the capability to consider the context provided by objects within the scene and in use by the humans, which can provide a further discriminant between human activity categories. In this paper, we propose a novel approach to real-time human activity recognition, through simultaneously learning from observations of both human poses and objects involved in the human activity. We formulate human activity recognition as a joint optimization problem under a unified mathematical framework, which uses a regression-like loss function to integrate human pose and object cues and defines structured sparsity-inducing norms to identify discriminative body joints and object attributes. To evaluate our method, we perform extensive experiments on two benchmark datasets and a physical robot in a home assistance setting. Experimental results have shown that our method outperforms previous methods and obtains real-time performance for human activity recognition with a processing speed of 10<sup>4</sup> Hz.

- Demonstration of Hospital Receptionist Robot with Extended Hybrid Code Network

    Author: Hwang, Eui Jun | The University of Auckland
    Author: Ahn, Byeong-Kyu | Sungkyunkwan University
    Author: MacDonald, Bruce | University of Auckland
    Author: Ahn, Ho Seok | The University of Auckland, Auckland
 
    keyword: Service Robots; Deep Learning in Robotics and Automation; Physical Human-Robot Interaction

    Abstract : Task-oriented dialogue system has a vital role in Human-Robot Interaction (HRI). However, it has been developed based on conventional pipeline approach which has several drawbacks; expensive, time-consuming, and so on. Based on this approach, developers manually define a robot's behaviour such as gestures and facial expressions on the corresponding dialogue states. Recently, end-to-end learning of Recurrent Neural Networks (RNNs) is an attractive solution for the dialogue system. In this paper, we proposed a social robot system using end-to-end dialogue system in the context of hospital receptionist. We utilized Hybrid Code Network (HCN) as an end-to-end dialogue system and extended to select both response and gesture using RNN based gesture selector. We evaluate its performance with human users and compare the results with one of the conventional methods. Empirical result shows that the proposed method has benefits in terms of dialogue efficiency, which indicates how efficient users were in performing the given tasks with the help of the robot. Moreover, we achieved the same performance regarding the robot's gesture with the proposed method compared to manually defined gestures.

- Can I Trust You? a User Study of Robot Mediation of a Support Group

    Author: Birmingham, Chris | University of Southern California
    Author: Hu, Zijian | University of Southern California
    Author: Mahajan, Kartik | University of Southern California
    Author: Reber, Elijah | Penn State University
    Author: Mataric, Maja | University of Southern California
 
    keyword: Social Human-Robot Interaction

    Abstract : Socially assistive robots have the potential to improve group dynamics when interacting with groups of people in social settings. This work contributes to the understanding of those dynamics through a user study of trust dynamics in the novel context of a robot mediated support group. For this study, a novel framework for robot mediation of a support group was developed and validated. To evaluate interpersonal trust in the multi-party setting, a dyadic trust scale was implemented and found to be uni-factorial, validating it as an appropriate measure of general trust. The results of this study demonstrate a significant increase in average interpersonal trust after the group interaction session, and qualitative post-session interview data report that participants found the interaction helpful and successfully supported and learned from one other. The results of the study validate that a robot-mediated support group can improve trust among strangers and allow them to share and receive support for their academic stress.

- Group Split and Merge Prediction with 3D Convolutional Networks

    Author: Wang, Allan | Carnegie Mellon University
    Author: Steinfeld, Aaron | Carnegie Mellon University
 
    keyword: Human-Centered Robotics; Human Detection and Tracking; Social Human-Robot Interaction

    Abstract : Mobile robots in crowds often have limited navigation capability due to insufficient evaluation of pedestrian behavior. We strengthen this capability by predicting splits and merges in multi-person groups. Successful predictions should lead to more efficient planning while also increasing human acceptance of robot behavior. We take a novel approach by formulating this as a video prediction problem, where group splits or merges are predicted given a history of geometric social group shape transformations. We take inspiration from the success of 3D convolution models for video-related tasks. By treating the temporal dimension as a spatial dimension, a modified C3D model successfully captures the temporal features required to perform the prediction task. We demonstrate performance on several datasets and analyze transfer ability to other settings. While current approaches for tracking human motion are not explicitly designed for this task, our approach performs significantly better. We also draw human interpretations from the model's learned features.

- TH�R: Human-Robot Navigation Data Collection and Accurate Motion Trajectories Dataset

    Author: Rudenko, Andrey | Robert Bosch GmbH
    Author: Kucner, Tomasz Piotr | Örebro Universitet
    Author: Swaminathan, Chittaranjan Srinivas | Örebro University
    Author: Chadalavada, Ravi Teja | Örebro University
    Author: Arras, Kai Oliver | Bosch Research
    Author: Lilienthal, Achim J. | Orebro University
 
    keyword: Social Human-Robot Interaction; Motion and Path Planning; Human Detection and Tracking

    Abstract : Understanding human behavior is key for robots and intelligent systems that share a space with people. Accordingly, research that enables such systems to perceive, track, learn and predict human behavior as well as to plan and interact with humans has received increasing attention over the last years. The availability of large human motion datasets that contain relevant levels of difficulty is fundamental to this research. Existing datasets are often limited in terms of information content, annotation quality or variability of human behavior. In this paper, we present TH�R, a new dataset with human motion trajectory and eye gaze data collected in an indoor environment with accurate ground truth for position, head orientation, gaze direction, social grouping, obstacles map and goal coordinates. TH�R also contains sensor data collected by a 3D lidar and involves a mobile robot navigating the space. We propose a set of metrics to quantitatively analyze motion trajectory datasets such as the average tracking duration, ground truth noise, curvature and speed variation of the trajectories. In comparison to prior art, our dataset has a larger variety in human motion behavior, is less noisy, and contains annotations at higher frequencies.

- Socially Assistive Infant-Robot Interaction: Using Robots to Encourage Infant Leg-Motion (I)

    Author: Fitter, Naomi T. | University of Southern California
    Author: Funke, Rebecca | University of Southern California
    Author: Pulido Pascual, Jos' Carlos | Universidad Carlos III De Madrid
    Author: Eisenman, Lauren E. | University of Southern California
    Author: Deng, Weiyang | University of Southern California
    Author: Rosales, Marcelo R. | University of Southern California
    Author: Bradley, Nina | University of Southern California
    Author: Sargent, Barbara | University of Southern California
    Author: Smith, Beth | University of Southern California
    Author: Mataric, Maja | University of Southern California
 
    keyword: Social Human-Robot Interaction; Rehabilitation Robotics; Robot Companions

    Abstract : Early interventions have the potential to positively influence infant movement patterns and support optimal neurodevelopmental outcomes. This work developed and validated a non-contact socially assistive infant-robot interaction system that aimed to use contingent reward learning and imitation to deliver effective early interventions that complement human-delivered therapy. <p>The described study explored if infants demonstrate contingent learning and imitation behavior in response to movements by a similarly-sized NAO humanoid robot. Twelve 6- to 8-month-old infants participated in a within-subjects study that compared different robot contingent reward policies for encouraging leg movement. Nine of the twelve participants learned the contingency. Of these learners, two responded less to the movement and lights reward than other rewards. Nine of the twelve infants imitated the NAO robot during at least one reward condition phase. These imitators displayed different learning rates and sometimes changed their behavior to imitate less during later reward conditions. Infants were generally alert and non-fussy when interacting with the robot. Parents of participants perceived the robot reward involving both movement and sound to be most engaging for their children.</p><p>As the new research area of infant-robot interaction develops, our results aim to inform continued work into targeted robot-assisted infant motion interventions.


- Real-Time Continuous Hand Motion Myoelectric Decoding by Automated Data Labeling

    Author: Hu, Xuhui | Southeast University
    Author: Zeng, Hong | Southeast University
    Author: Chen, Dapeng | Southeast University
    Author: Zhu, Jiahang | Southeast University
    Author: Song, Aiguo | Southeast University
 
    keyword: Social Human-Robot Interaction; Gesture, Posture and Facial Expressions; Prosthetics and Exoskeletons

    Abstract : In this paper an automated data labeling (ADL) neural network is proposed to streamline dataset collecting for real-time predicting the continuous motion of hand and wrist, these gestures are only decoded from a surface electromyography (sEMG) array of eight channels. Unlike collecting both the bio-signals and hand motion signals as samples and labels in supervised learning, this algorithm only collects unlabeled sEMG into an unsupervised neural network, in which the hand motion labels are auto-generated. The coefficient of determination (R^2) for three DOFs, i.e. wrist flex/extension, wrist pro/supination, hand open/close, was 0.86, 0.89 and 0.87 respectively. The comparison between real motion labels and auto-generated labels shows that the latter has earlier response than former. The results of Fitts' law test indicate that ADL has capability of controlling multi-DOFs simultaneously even though the training set only contains sEMG data from single DOF gesture. Moreover, no more hand motion measurement needed which greatly helps upper limb amputee imagine the gesture of residual limb to control a dexterous prosthesis.

- Towards Proactive Navigation: A Pedestrian-Vehicle Cooperation Based Behavioral Model

    Author: Kabtoul, Maria | Univ. Grenoble Alpes, Inria
    Author: Spalanzani, Anne | INRIA / Univ. Grenoble Alpes
    Author: Martinet, Philippe | INRIA
 
    keyword: Social Human-Robot Interaction; Autonomous Vehicle Navigation; Human-Centered Automation

    Abstract : Developing autonomous vehicles capable of navigating safely and socially around pedestrians is a major challenge in intelligent transportation. This challenge cannot be met without understanding pedestrians' behavioral response to an autonomous vehicle, and the task of building a clear and quantitative description of the pedestrian to vehicle interaction remains a key milestone in autonomous navigation research. As a step towards safe proactive navigation in a space shared with pedestrians, this work introduces a pedestrian-vehicle interaction behavioral model. The model estimates the pedestrian's cooperation with the vehicle in an interaction scenario by a quantitative time-varying function. Using this cooperation estimation the pedestrian's trajectory is predicted by a cooperation-based trajectory planning model. Both parts of the model are tested and validated using real-life recorded scenarios of pedestrian-vehicle interaction. The model is capable of describing and predicting agents' behaviors when interacting with a vehicle in both lateral and frontal crossing scenarios.

- Studying Navigation As a Form of Interaction: A Design Approach for Social Robot Navigation Methods

    Author: Scales, Philip | Université Grenoble Alpes
    Author: Aycard, Olivier | University Grenoble
    Author: Aubergé, Véronique | University Grenoble Alps, LIG, UMR CNRS 5217
 
    keyword: Social Human-Robot Interaction; Motion and Path Planning

    Abstract : Social Navigation methods attempt to integrate knowledge from Human Sciences fields such as the notion of Proxemics into mobile robot navigation. They are often evaluated in simulations, or lab conditions with informed participants, and studies of the impact of the robot behavior on humans are rare. Humans communicate and interact through many vectors, of which motion and positioning, which can be related to social hierarchy and the socio-physical context. If a robot is to be deployed among humans, the methods it uses should be designed with this in mind. This work acts as the first step in an ongoing project in which we explore how to design navigation methods for mobile robots destined to be deployed among humans. We aim to consider navigation as more than just a functionality of the robot, and to study the impact of robot motion on humans. In this paper, we focus on the person-following task. We selected a state of the art person-following method as the basis for our method, which we modified and extended in order for it to be more general and adaptable. We conducted pilot experiments using this method on a real mobile robot in ecological contexts. We used results from the experiments to study the Human-Robot Interaction as a whole by analysing both the person-following method and the human behavior. Our preliminary results show that the way in which the robot followed a person had an impact on the interaction that emerged between them.

- Robot Plan Model Generation and Execution with Natural Language Interface

    Author: Yang, Kyon-Mo | Korea Institute of Robot and Convergence
    Author: Seo, Kap-Ho | Korea Institute of Robot and Convergence
    Author: Kang, Sang Hoon | Ulsan National Institute of Science and Technology(UNIST) / U. O
    Author: Lim, Yoonseob | Korea Institute of Science and Technology
 
    keyword: Social Human-Robot Interaction; Cognitive Human-Robot Interaction; Task Planning

    Abstract : Verbal interaction between human and robot may play a key role in conveying suitable directions for a robot to achieve the goal of user's request. However, robot may need to correct task plans or make new decisions with human help, which would make the interaction inconvenient and also increase the interaction time. In this paper, we propose a new verbal interaction based method that can generate plan models and execute proper actions without human involvement in the middle of performing task by robot. To understand verbal behaviors of human when giving instructions to robot, we first conducted a brief user study and found that human user does not explicitly express the required task. To handle such unclear instructions by human, we propose two different algorithms that can generate component of new plan models based on intents and entities parsed from natural language, and can resolve the unclear entities existed in human instructions. Experimental scenario with robot, Cozmo was tried in the lab environment to test whether or not proposed method could generate appropriate plan model. As a result, we found that robot could successfully accomplish the task following human instructions and also found that number of interaction and components in the plan model could be reduced as opposed to general reactive plan model. In the future, we are going to improve the automated process of generating plan models and apply various scenarios under different service environments and robots

- Mapless Navigation among Dynamics with Social-Safety-Awareness: A Reinforcement Learning Approach from 2D Laser Scans

    Author: Jin, Jun | University of Alberta
    Author: Nguyen, Nhat | Huawei Technologies Canada
    Author: Sakib, Nazmus | University of Alberta
    Author: Graves, Daniel | Huawei Technologies Canada, Ltd
    Author: Yao, Hengshuai | Huawei
    Author: Jagersand, Martin | University of Alberta
 
    keyword: Social Human-Robot Interaction; Collision Avoidance; Autonomous Vehicle Navigation

    Abstract : We propose a method to tackle the problem of mapless collision-avoidance navigation where humans are present using 2D laser scans. Our proposed method uses ego-safety to measure collision from the robot's perspective while social-safety to measure the impact of our robot's actions on surrounding pedestrians. Specifically, the social-safety part predicts the intrusion impact of our robot's action into the interaction area with surrounding humans. We train the policy using reinforcement learning on a simple simulator and directly evaluate the learned policy in Gazebo and real robot tests. Experiments show the learned policy can be smoothly transferred without any fine tuning. We observe that our method demonstrates time-efficient path planning behavior with high success rate in mapless navigation tasks. Furthermore, we test our method in a navigation among dynamic crowds task considering both low and high volume traffic. Our learned policy demonstrates cooperative behavior that actively drives our robot into traffic flows while showing respect to nearby pedestrians. Evaluation videos are at https://sites.google.com/view/ssw-batman

- People's Adaptive Side-By-Side Model Evolved to Accompany Groups of People by Social Robots

    Author: Repiso, Ely | Institut De Robòtica I Informàtica Industrial, CSIC-UPC
    Author: Garrell, Anais | UPC-CSIC
    Author: Sanfeliu, Alberto | Universitat Politècnica De Cataluyna
 
    keyword: Social Human-Robot Interaction; Humanoid Robots; Service Robots

    Abstract : The presented method implements a robot accompaniment in an adaptive side-by-side formation of a single person or a group of people. The method enhances our previous robot adaptive side-by-side behavior allowing the robot to accompany a group of people, not only one person. Adaptive means that the robot is capable to adjust its position and velocity to the behavior of the group being accompanied, without bothering other pedestrians in the environment, as well as facilitating the group navigation to avoid static and dynamic obstacles. Furthermore, the robot can deal with the random factor of human behavior in several situations. Firstly, if other people interfere in the path of the companions, the robot leaves space for the person of the group that has to avoid those other people, by approaching the other companion. Also, without invading any personal space. Secondly, if the people of the group changes their physical position inside the group formation, then the robot adapts to them dynamically by changing from the lateral position to the central position of the formation or otherwise. Thirdly, the robot adapts to changes in the velocity of people in the group and other people that interfere in the path of the group, in magnitude and orientation. Fourthly, the robot can deal with occlusions of one accompanied person by the other. The method has been validated using synthetic experiments and real-life experiments with our robot. Finally,we developed an user study comparing the

## Biologically-Inspired Robots

- Coronal Plane Spine Twisting Composes Shape to Adjust the Energy Landscape for Grounded Reorientation

    Author: Caporale, J. Diego | University of Pennsylvania
    Author: McInroe, Benjamin | University of California, Berkeley
    Author: Ning, Chenze | University of Pennsylvania
    Author: Libby, Thomas | University of Washington
    Author: Full, Robert | University of California at Berkeley
    Author: Koditschek, Daniel | University of Pennsylvania
 
    keyword: Biologically-Inspired Robots; Legged Robots

    Abstract : Despite substantial evidence for the crucial role played by an active backbone or spine in animal locomotion,its adoption in legged robots remains limited because the added mechanical complexity and resulting dynamical challenges pose daunting obstacles to characterizing even a partial range of potential performance benefits. This paper takes a next step toward such a characterization by exploring the quasistatic terrestrial self-righting mechanics of a model system with coronal plane spine twisting (CPST). Reduction from a full 3D kinematic model of CPST to a two parameter, two degree of freedom coronal plane representation of body shape affordance predicts a substantial benefit to ground righting by lowering the barrier between stable potential energy basins. The reduced model predicts the most advantageous twist angle for several cross-sectional geometries, reducing the required righting torque by up to an order of magnitude depending on constituent shapes. Experiments with a three actuated degree of freedom physical mechanism corroborate the kinematic model predictions using two different quasistatic reorientation maneuvers for both elliptical and rectangular shaped bodies with a range of eccentricities or aspect ratios. More speculative experiments make intuitive use of the kinematic model in a highly dynamic maneuver to suggest still greater benefits of CPST achievable by coordinating kinetic as well as potential energy, for example as in a future multi-appendage system

- Significance of the Compliance of the Joints on the Dynamic Slip Resistance of a Bioinspired Hoof (I)

    Author: Abad Guaman, Sara Adela | University College London
    Author: Herzig, Nicolas | University of Sheffield
    Author: Sadati, Seyedmohammadhadi | King's College London
    Author: Nanayakkara, Thrishantha | Imperial College London
 
    keyword: Biologically-Inspired Robots; Compliant Joint/Mechanism; Legged Robots

    Abstract : Robust mechanisms for slip resistance are an open challenge in legged locomotion. Animals such as goats show impressive ability to resist slippage on cliffs. It is not fully known what attributes in their body determine this ability. Studying the slip resistance dynamics of the goat may offer insight toward the biologically inspired design of robotic hooves. This article tests how the embodiment of the hoof contributes to solving the problem of slip resistance. We ran numerical simulations and experiments using a passive robotic goat hoof for different compliance levels of its three joints. We established that compliant yaw and pitch and stiff roll can increase the energy required to slide the hoof by ~20% compared to the baseline (stiff hoof). Compliant roll and pitch allow the robotic hoof to adapt to the irregularities of the terrain. This produces an antilock braking system-like behavior of the robotic hoof for slip resistance. Therefore, the pastern and coffin joints have a substantial effect on the slip resistance of the robotic hoof, while the fetlock joint has the lowest contribution. These shed insights into how robotic hooves can be used to autonomously improve slip resistance.

- Motion Design for a Snake Robot Negotiating Complicated Pipe Structures of a Constant Diameter

    Author: Inazawa, Mariko | Kyoto University
    Author: Takemori, Tatsuya | Kyoto University
    Author: Tanaka, Motoyasu | The Univ. of Electro-Communications
    Author: Matsuno, Fumitoshi | Kyoto University
 
    keyword: Biologically-Inspired Robots; Motion and Path Planning; Field Robots

    Abstract : A method for designing the motion of a snake robot negotiating complicated pipe structures having a constant diameter is presented. For such robots moving inside pipes, there are various ``obstacles" such as junctions, bends, shears, and blockages. To surmount these obstacles, we propose a method that enables the robot to adapt to multiple pipe structures of a constant diameter. We designed the target form of the snake robot of two helices connected with an arbitrary shape. This method is applicable to various obstacles by designing a part of the target form conforming to the obstacle. The robot negotiates obstacles under shift control by employing a rolling motion. We demonstrated the effectiveness of the proposed method in various experiments.

- A Neuro-Inspired Computational Model for a Visually Guided Robotic Lamprey Using Frame and Event Based Cameras

    Author: Youssef, Ibrahim | Ecole Polytechnique Fédérale De Lausanne
    Author: Mutlu, Mehmet | École Polytechnique Fédérale De Lausanne (EPFL)
    Author: Bayat, Behzad | EPFL | École Polytechnique Fédérale De Lausanne
    Author: Crespi, Alessandro | Ecole Polytechnique Fédérale De Lausanne
    Author: Hauser, Simon | BIRL, University of Cambridge
    Author: Conradt, Jorg | KTH Royal Institute of Technology
    Author: Bernardino, Alexandre | IST - Técnico Lisboa
    Author: Ijspeert, Auke | EPFL
 
    keyword: Biologically-Inspired Robots; Marine Robotics; Computer Vision for Other Robotic Applications

    Abstract : The computational load associated with computer vision is often prohibitive, and limits the capacity for on-board image analysis in compact mobile robots. Replicating the kind of feature detection and neural processing that animals excel at remains a challenge in most biomimetic aquatic robots. Event-driven sensors use a biologically inspired sensing strategy to eliminate the need for complete frame capture. Systems employing event-driven cameras enjoy reduced latencies, power consumption, bandwidth, and benefit from a large dynamic range. However, to the best of our knowledge, no work has been done to evaluate the performance of these devices in underwater robotics. This work proposes a robotic lamprey design capable of supporting computer vision, and uses this system to validate a computational neuron model for driving anguilliform swimming. The robot is equipped with two different types of cameras: frame-based and event-based cameras. These were used to stimulate the neural network, yielding goal-oriented swimming. Finally, a study is conducted comparing the performance of the computational model when driven by the two different types of camera. It was observed that event-based cameras improved the accuracy of swimming trajectories and led to significant improvements in the rate at which visual inputs were processed by the network.

- Untethered Flight of an At-Scale Dual-Motor Hummingbird Robot with Bio-Inspired Decoupled Wings

    Author: Tu, Zhan | Purdue University
    Author: Fei, Fan | Purdue University
    Author: Deng, Xinyan | Purdue University
 
    keyword: Biologically-Inspired Robots; Biomimetics; Aerial Systems: Mechanics and Control

    Abstract : In this paper, we present the untethered flight of an at-scale tailless hummingbird robot with independently controlled wings. It represents the first untethered stable flight of a two actuator powered bio-inspired Flapping Wing Micro Air Vehicle (FWMAV) in both indoor and outdoor environment. The untethered flight of such FWMAVs is a challenging task due to stringent payload limitation from severe underactuation and power efficiency challenge caused by motor reciprocating motion. In this work, we present the detailed modeling, optimization, and system integration of onboard power, actuation, sensing, and flight control to address these unique challenges of such FWMAV during untethered flight. We performed untethered flight experiments in both indoor and outdoor environment and demonstrate sustained stable flight of the robot.

-  Model-Based Feedback Control of Live Zebrafish Behavior Via Interaction with a Robotic Replica (I)

    Author: De Lellis, Pietro | University of Naples Federico II
    Author: Cadolini, Eduardo | University of Naples Federico II
    Author: Croce, Arrigo | University of Naples Federico II
    Author: Yang, Yanpeng | New York University
    Author: Di Bernardo, Mario | University of Naples Federico II
    Author: Porfiri, Maurizio | New York University Polytechnic School of Engineering


- Steering Control of Magnetic Helical Swimmers in Swirling Flows Due to Confinement

    Author: Caldag, Hakan Osman | Sabanci University
    Author: Yesilyurt, Serhat | Sabanci University
 
    keyword: Biologically-Inspired Robots; Visual Servoing; Micro/Nano Robots

    Abstract : Artificial microswimmers are prospective robotic agents especially in biomedical applications. A rotating magnetic field can actuate a magnetized swimmer with a helical tail and enable propulsion. Such swimmers exhibit several modes of instability. Inside conduits, for example, hydrodynamic interactions with the boundaries lead to helical paths for pusher-mode swimmers; in this mode the helical tail pushes a rotating magnetic head. State-of-the-art in controlled navigation of microswimmers is based on aligning the swimmer orientation according to a reference path, thereby requiring both swimmer orientation and position to be known. Object-orientation is hard to track especially in in vivo scenarios which render orientation-based methods practically unfeasible. Here, we show that the kinematics for a confined swimmer can be linearized by assuming a low wobbling angle. This allows for a control law solely based on the swimmer position. The approach is demonstrated through experiments and two different numerical models: the first is based on the resistive force theory for a swimmer inside a swirling flow represented by a forced vortex and the second is a computational fluid dynamics model, which solves Stokes equations for a swimmer inside a circular channel. Helical pusher-mode trajectories are suppressed significantly for the straight path following problem. The error in real-life experiments remains comparable to those in the state-of-the-art methods.

- Sim2real Gap Is Non-Monotonic with Robot Complexity for Morphology-In-The-Loop Flapping Wing Design

    Author: Rosser, Kent Ashley | University of Vermont, University of South Australia, Defence Sc
    Author: Kok, Jia ming | DST Group
    Author: Chahl, Javaan | University of South Australia
    Author: Bongard, Josh | University of Vermont
 
    keyword: Biologically-Inspired Robots; Soft Robot Materials and Design; Aerial Systems: Mechanics and Control

    Abstract : Morphology of a robot design is important to its ability to achieve a stated goal and therefore applying machine learning approaches that incorporate morphology in the design space can provide scope for significant advantage. Our study is set in a domain known to be reliant on morphology: flapping wing flight. We developed a parameterised morphology design space that draws features from biological exemplars and apply automated design to produce a set of high performance robot morphologies in simulation. By performing sim2real transfer on a selection, for the first time we measure the shape of the reality gap for variations in design complexity. We found for the flapping wing that the reality gap changes non-monotonically with complexity, suggesting that certain morphology details narrow the gap more than others, and that such details could be identified and further optimised in a future end-to-end automated morphology design process.

- A Linearized Model for an Ornithopter in Gliding Flight: Experiments and Simulations

    Author: Lopez Lopez, Ricardo | University of Seville, GRVC
    Author: Perez Sanchez, Vicente | University of Seville, GRVC
    Author: Ramon Soria, Pablo | University of Seville
    Author: Martín-Alcántara, Antonio | University of Seville, GRVC
    Author: Fernandez-Feria, Ramon | University of Malaga
    Author: Arrue, Begoña C. | Universidad De Sevilla
    Author: Ollero, Anibal | University of Seville
 
    keyword: Biologically-Inspired Robots; Dynamics; Aerial Systems: Mechanics and Control

    Abstract : This work studies the accuracy of a simple but effective analytical model for a flapping-wings UAV in longitudinal gliding flight configuration comparing it with experimental results of a real ornithopter. The aerodynamic forces are modeled following the linearized potential theory for a flat plate in gliding configuration, extended to flapping wing episodes modeled also by the (now unsteady) linear potential theory, which are studied numerically. In the gliding configuration, the model reaches a steady-state descent at given terminal velocity and pitching and gliding angles, governed by the wings and tail position. In the flapping-wing configuration, it is noticed that the vehicle can increase its flight velocity and perform climbing episodes. A realistic simulation tool based on Unreal Engine 4 was developed to visualize the effect of the tail position and flapping frequencies and amplitudes on the ornithopter flight in realtime. The paper also includes the experimental validation of the gliding flight and the data has been released for the community.

- Towards Biomimicry of a Bat-Style Perching Maneuver on Structures: The Manipulation of Inertial Dynamics

    Author: Ramezani, Alireza | Northeastern University
 
    keyword: Biologically-Inspired Robots; Aerial Systems: Mechanics and Control; Dynamics

    Abstract : The flight characteristics of bats remarkably have been overlooked in aerial drone designs. Unlike other animals, bats leverage the manipulation of inertial dynamics to exhibit aerial flip turns when they perch. Inspired by this unique maneuver, this work develops and uses a tiny robot called Harpoon to demonstrate that the preparation for upside-down landing is possible through: 1) reorientation towards the landing surface through zero-angular-momentum turns and 2) reaching to the surface through shooting a detachable landing gear. The closed-loop manipulations of inertial dynamics takes place based on a symplectic description of the dynamical system (body and appendage), which is known to exhibit an excellent geometric conservation properties.

- Bioinspired Object Motion Filters As the Basis of Obstacle Negotiation in Micro Aerial Systems

    Author: Zhou, Rui | Imperial College London
    Author: Lin, Huai-Ti | Imperial College London
 
    keyword: Biologically-Inspired Robots; Collision Avoidance; Aerial Systems: Perception and Autonomy

    Abstract : All animals and robots that move in the world must navigate to a goal while clearing obstacles. Using vision to accomplish such task has several advantages in cost and payload, which explains the prevalence of biological visual guidance. However, the computational overhead has been an obvious concern when increasing number of pixels and frames that need to be analyzed in real-time for a machine vision system. The use of motion vision and optic flow has been a popular bio-inspired solution for this problem. However, many early-stage motion detection approaches rely on special hardware (e.g. event-cameras) or extensive computation (e.g. dense optic flow map). Here we demonstrate a method to combine an insect vision inspired object motion filter model with simple visual guidance rules to fly through a cluttered environment. We have implemented a complete feedback control loop in a micro racing drone and achieved proximal-distal object separation through only two object motion filters. We discuss the key constraints and the scalability of this approach for future development.

- Design and Architecture of ARCSnake: Archimedes' Screw-Propelled Serpentine Robot

    Author: Schreiber, Dimitri A. | University of California
    Author: Richter, Florian | University of California, San Diego
    Author: Bilan, Andrew | University of California San Diego
    Author: Gavrilov, Peter | University of California San Diego
    Author: Lam, Hoi Man | University of California San Diego
    Author: Price, Casey | University of California San Diego
    Author: Carpenter, Kalind | Jet Propulsion Laboratory
    Author: Yip, Michael C. | University of California, San Diego
 
    keyword: Biologically-Inspired Robots

    Abstract : This paper presents the design and performance of a screw-propelled serpentine robot. This robot comprises serially linked, identical modules, each incorporating an Archimedes' screw for propulsion and a universal joint (U-Joint) for orientation control. When serially chained, these modules form a versatile serpentine robot platform which enables the robot to reshape its body configuration for varying environments, typical of a snake. Furthermore, the Archimedes' screws allow for novel omni-wheel drive-like motions by speed controlling their screw threads. This paper considers the mechanical and electrical design, as well as the software architecture for realizing a fully integrated system. The system includes 3N actuators for N segments, each controlled using a BeagleBone Black with a customized power-electronics cape, a 9 Degrees of Freedom (DoF) Inertial Measurement Unit (IMU), and a scalable communication channel over ROS. The intended application for this robot is its use as an instrumentation mobility platform on terrestrial planets where the terrain may involve vents, caves, ice, and rocky surfaces. Additional experiments are shown on our website.

## Robotics in Agriculture, Construction and Mining
- GPR-Based Subsurface Object Detection and Reconstruction Using Random Motion and DepthNet

    Author: Feng, Jinglun | City College of New York
    Author: Liang, Yang | Shenyang Institute of Automation, Chinese Academy of Sciences
    Author: Wang, Haiyan | City College of New York
    Author: Song, Yifeng | Chinese Academy of Sciences, Shenyang InstituteofAutomation
    Author: Xiao, Jizhong | The City College of New York
 
    keyword: Robotics in Construction; AI-Based Methods; Deep Learning in Robotics and Automation

    Abstract : Ground Penetrating Radar (GPR) is one of the most important non-destructive evaluation (NDE) devices to detect the subsurface objects (i.e. rebars, utility pipes) and reveal the underground scene. One of the biggest challenges in GPR based inspection is the subsurface targets reconstruction. In order to address this issue, this paper presents a 3D GPR migration and dielectric prediction system to detect and reconstruct underground targets. This system is composed of three modules: 1) visual inertial fusion (VIF) module to generate the pose information of GPR device,	2) deep neural network module (i.e., DepthNet) which detects B-scan of GPR image, extracts hyperbola features to remove the noise in B-scan data and predicts dielectric to determine the depth of the objects, 3) 3D GPR migration module which synchronizes the pose information with GPR scan data processed by DepthNet to reconstruct and visualize the 3D underground targets. Our proposed DepthNet processes the GPR data by removing the noise in B-scan image as well as predicting depth of subsurface objects. In addition, the experimental results verify that our proposed method improve the migration accuracy and performance in generating 3D GPR image compared with the traditional migration methods.

- A Data-Driven Approach to Prediction and Optimal Bucket-Filling Control for Autonomous Excavators
 
    Author: Sandzimier, Ryan | MIT
    Author: Asada, Harry | MIT
 
    keyword: Robotics in Construction; Model Learning for Control; Mining Robotics

    Abstract : We develop a data-driven, statistical control method for autonomous excavators. Interactions between soil and an excavator bucket are highly complex and nonlinear, making traditional physical modeling difficult to use for real-time control. Here, we propose a data-driven method, exploiting data obtained from laboratory tests. We use the data to construct a nonlinear, non-parametric statistical model for predicting the behavior of soil scooped by an excavator bucket. The prediction model is built for controlling the amount of soil collected with a bucket. An excavator collects soil by dragging the bucket along the soil surface and scooping the soil by rotating the bucket. It is important to switch from the drag phase to the scoop phase with the correct timing to ensure an appropriate amount of soil has accumulated in front of the bucket. We model the process as a heteroscedastic Gaussian process (GP) based on the observation that the variance of the collected soil mass depends on the scooping trajectory, i.e. the input, as well as the shape of the soil surface immediately prior to scooping. We develop an optimal control algorithm for switching from the drag phase to the scoop phase at an appropriate time and for generating a scoop trajectory to capture a desired amount of soil with high confidence. We implement the method on a robotic excavator and collect experimental data. Experiments show promising results in terms of being able to achieve a desired bucket fill factor with

- Real-Time Stereo Visual Servoing for Rose Pruning with Robotic Arm

    Author: Cuevas-Velasquez, Hanz | University of Edinburgh
    Author: Gallego, Antonio-Javier | University of Alicante
    Author: Tylecek, Radim | University of Edinburgh
    Author: Hemming, Jochen | Wageningen University and Research Centre
    Author: Van Tuijl, Bart | Wageningen University &amp; Research - WUR
    Author: Mencarelli, Angelo | Wageningen University &amp; Research - WUR
    Author: Fisher, Robert | University of Edinburgh
 
    keyword: Robotics in Agriculture and Forestry; Visual Servoing; Grippers and Other End-Effectors

    Abstract : The paper presents a working pipeline which integrates hardware and software in an automated robotic rose cutter; to the best of our knowledge, the first robot able to prune rose bushes in a natural environment. Unlike similar approaches like tree stem cutting, the proposed method does not require to scan the full plant, have multiple cameras around the bush, or assume that a stem does not move. It relies on a single stereo camera mounted on the end-effector of the robot and real-time visual servoing to navigate to the desired cutting location on the stem. The evaluation of the whole pipeline shows a good performance in a garden with unconstrained conditions, where finding and approaching a specific location on a stem is challenging due to occlusions caused by other stems and dynamic changes caused by the wind.

- Canopy-Based Monte Carlo Localization in Orchards Using Top-View Imagery

    Author: Shalev, Omer | Technion - Israel Institute of Technology
    Author: Degani, Amir | Technion - Israel Institute of Technology
 
    keyword: Robotics in Agriculture and Forestry; Localization; Aerial Systems: Applications

    Abstract : Localization of ground mobile robots in orchards is a complex problem which is yet to be fully addressed. The typical localization approaches are not adjusted to the characteristics of the orchard environment, especially the homogeneous scenery. To alleviate these difficulties, we propose to use top-view images of the orchard acquired in real-time. The top-view observation of the orchard provides a unique signature of every tree formed by the shape of its canopy. This practically changes the homogeneity premise in orchards and paves the way for addressing the kidnapped robot problem. Using computer vision techniques, we build a virtual canopies laser scan around the ground robot which is generated from low-altitude top-view video streams. We apply Monte Carlo Localization on this virtual scan to localize the robot against a high-altitude top-view snapshot image which is used as a map. The suggested approach is examined in numerous offline experiments conducted on data acquired in real orchards and is compared against a typical simulated approach which relies on ground-level trunk observations. The canopy-based approach demonstrated better performance in all measures, including convergence to centimeter-level accuracy.

- In-Field Grape Cluster Size Assessment for Vine Yield Estimation Using a Mobile Robot and a Consumer Level RGB-D Camera

    Author: Kurtser, Polina | Örebro University
    Author: Ringdahl, Ola | Umeå University
    Author: Rotstein, Nati | Ben Gurion University of the Negev
    Author: Berenstein, Ron | Agricultural Research Organization Volcani Center
    Author: Edan, Yael | Ben-Gurion University of the Negev
 
    keyword: Field Robots; RGB-D Perception; Agricultural Automation

    Abstract : Current practice for vine yield estimation is based on RGB cameras and has limited performance. In this paper we present a method for outdoor vine yield estimation using a consumer grade RGB-D camera mounted on a mobile robotic platform. An algorithm for automatic grape cluster size estimation using depth information is evaluated both in controlled outdoor conditions and in commercial vineyard conditions. Ten video scans (3 camera viewpoints with 2 different backgrounds and 2 natural light conditions), acquired from a controlled outdoor experiment and a commercial vineyard setup, are used for analyses. The collected dataset (GRAPES3D) is released to the public. A total of 4542 regions of 49 grape clusters were manually labeled by a human annotator for comparison. Eight variations of the algorithm are assessed, both for manually labeled and auto-detected regions. The effect of viewpoint, presence of an artificial background, and the human annotator are analyzed using statistical tools. Results show 2.8-3.5 cm average error for all acquired data and reveal the potential of using low-cost commercial RGB-D cameras for improved robotic yield estimation.

- Autonomous Excavation of Rocks Using a Gaussian Process Model and Unscented Kalman Filter

    Author: Sotiropoulos, Filippos Edward | Massachusetts Institute of Technology
    Author: Asada, Harry | MIT
 
    keyword: Mining Robotics; Model Learning for Control; Robotics in Construction

    Abstract : In large-scale open-pit mining and construction works, excavators must deal with large rocks mixed with gravel and granular soil. Capturing and moving large rocks with the bucket of an excavator requires a high level of skill that only experienced human operators possess. In an attempt to develop autonomous rock excavators, this paper presents a control method that predicts the rock movement in response to bucket operation and computes an optimal bucket movement to capture the rock. The process is highly nonlinear and stochastic. A Gaussian process model, which is nonlinear, non-parametric and stochastic, is used for describing rock behaviors interacting with the bucket and surrounding soil. Experimental data is used directly for identifying the model. An Unscented Kalman Filter (UKF) is then integrated with the Gaussian process model for predicting the rock movements and estimating the length of the rock. A feedback controller that optimizes a cost function is designed based on the rock motion prediction and implemented on a robotic excavator prototype. Experiments demonstrate encouraging results towards autonomous mining and rock excavation.

## Kinematics
- Slip-Limiting Controller for Redundant Line-Suspended Robots: Application to LineRanger

    Author: Hamelin, Philippe | Hydro-Quebec Research Institute
    Author: Richard, Pierre-Luc | Hydro-Quebec Research Institute
    Author: Lepage, Marco | Hydro-Québec - IREQ
    Author: Lagacé, Marin | Hydro-Quebec Research Institute
    Author: Sartor, Alex | Hydro-Quebec Research Institute
    Author: Lambert, Ghislain | Hydro-Quebec Research Institute
    Author: Hébert, Camille | Hydro-Québec's Research Institute
    Author: Pouliot, Nicolas | IREQ Hydro-Québec Research Institute
 
    keyword: Redundant Robots; Wheeled Robots; Motion Control

    Abstract : In this paper, a slip-limiting controller for redundant line-suspended robots is presented. This kind of robot is usually equipped with v-shaped wheels, which brings uncertainty about the effective wheel radius, particularly when crossing obstacles. The proposed algorithm is able to estimate and limit wheel slippage in the presence of such uncertainty, relying only on wheel angular velocity measurements. Slip limitation occurs in the control allocation algorithm and hence is decoupled from the high-level velocity controller, allowing a broad applicability in centralized control approaches. Experimental results on LineRanger show that it effectively reduces wheel slippage compared to traditional centralized control while being more energy efficient than traditional decentralized control approaches.

- Interval Search Genetic Algorithm Based on Trajectory to Solve Inverse Kinematics of Redundant Manipulators and Its Application

    Author: Wu, Di | Central South University
    Author: Zhang, Wenting | Central South University
    Author: Qin, Mi | Central South University
    Author: Xie, Bin | Central South University
 
    keyword: Redundant Robots; Kinematics; Industrial Robots

    Abstract : In this paper, a new method is proposed to solve the inverse kinematics problem of redundant manipulators. This method demonstrates superior performance on continuous motion by combining interval search genetic algorithm based on trajectory which we propose with parametric joint angle method. In this method, population continuity strategy is utilized to improve search speed and reduce evolutionary generation, interval search strategy is introduced to enhance the search ability and overcome the influence of singularity, and reference point strategy is used to avoid sudden changes of joint variables. By introducing those three strategies, this method is especially suitable for redundant manipulators that perform continuous motion. It can not only obtain solutions of inverse kinematics quickly, but also ensure the motion continuity of manipulator and accuracy of the end effector. Moreover, this algorithm can also perform multi-objective tasks by adjusting the fitness function. Finally, this algorithm is applied to an 8 degree of freedom tunnel shotcrete robot. Field experiments and data analysis show that the algorithm can solve the problem quickly in industrial field, and ensure the motion continuity and accuracy.

- Analytical Expressions of Serial Manipulator Jacobians and Their High-Order Derivatives Based on Lie Theory

    Author: Fu, Zhongtao | Kings College London
    Author: Spyrakos-Papastavridis, Emmanouil | King's College London
    Author: Lin, Yen-Hua | King's College London
    Author: Dai, Jian | School of Natural and Mathematical Sciences, King's College Lond
 
    keyword: Kinematics; Motion Control; Flexible Robots

    Abstract : Serial manipulator kinematics provide a mapping between joint variables in joint-space coordinates, and end-effector configurations in task-space Cartesian coordinates. Velocity mappings are represented via the manipulator Jacobian produced by direct differentiation of the forward kinematics. Acquisition of acceleration, jerk, and snap expressions, typically utilized for accurate trajectory-tracking, requires the computation of high-order Jacobian derivatives. As compared to conventional numerical/D-H approaches, this paper proposes a novel methodology to derive the Jacobians and their high-order derivatives symbolically, based on Lie theory, which requires that the derivatives are calculated with respect to each joint variable and time. Additionally, the technique described herein yields a mathematically sound solution to the high-order Jacobian derivatives, which distinguishes it from other relevant works. Performing computations with respect to the two inertial-fixed and body-fixed frames, the analytical form of the spatial and body Jacobians are derived, as well as their higher-order derivatives, without resorting to any approximations, whose expressions depend explicitly on the joint state and the choice of reference frames. The proposed method provides more tractable computation of higher-order Jacobian derivatives, while its effectiveness has been verified by conducting a comparative analysis based on experimental data extracted from a KUKA LRB iiwa7 R800 manipulator.

- Inverse Kinematics for Serial Kinematic Chains Via Sum of Squares Optimization

    Author: Maric, Filip | University of Toronto Institute for Aerospace Studies
    Author: Giamou, Matthew | University of Toronto
    Author: Khoubyarian, Soroush | University of Toronto
    Author: Petrovic, Ivan | University of Zagreb
    Author: Kelly, Jonathan | University of Toronto
 
    keyword: Kinematics; Optimization and Optimal Control; Manipulation Planning

    Abstract : Inverse kinematics is a fundamental challenge for articulated robots: fast and accurate algorithms are needed for translating task-related workspace constraints and goals into feasible joint configurations. In general, inverse kinematics for serial kinematic chains is a difficult nonlinear problem, for which closed form solutions cannot easily be obtained. Therefore, computationally efficient numerical methods that can be adapted to a general class of manipulators are of great importance. In this paper, we use convex optimization techniques to solve the inverse kinematics problem with joint limit constraints for highly redundant serial kinematic chains with spherical joints in two and three dimensions.This is accomplished through a novel formulation of inverse kinematics as a nearest point problem, and with a fast sum of squares solver that exploits the sparsity of kinematic constraints for serial manipulators. Our method has the advantages of post-hoc certification of global optimality and a runtime that scales polynomially with the number of degrees of freedom. Additionally, we prove that our convex relaxation leads to a globally optimal solution when certain conditions are met, and demonstrate empirically that these conditions are common and represent many practical instances. Finally, we provide an open source implementation of our algorithm.

- Multi-Task Closed-Loop Inverse Kinematics Stability through Semidefinite Programming

    Author: Marti-Saumell, Josep | CSIC-UPC
    Author: Santamaria-Navarro, Angel | NASA Jet Propulsion Laboratory, Caltech
    Author: Ocampo-Martinez, Carlos | Technical University of Catalonia (UPC)
    Author: Andrade-Cetto, Juan | CSIC-UPC
 
    keyword: Kinematics; Redundant Robots; Motion Control

    Abstract : Today's complex robotic designs comprise in some cases a large number of degrees of freedom, enabling for multi-objective task resolution (e.g., humanoid robots or aerial manipulators). This paper tackles the stability problem of a hierarchical closed-loop inverse kinematics algorithm for such highly redundant robots. We present a method to guarantee system stability by performing an online tuning of the closedloop control gains. We define a semi-definite programming problem (SDP) with these gains as decision variables and a discrete-time Lyapunov stability condition as a linear matrix inequality, constraining the SDP optimization problem and guaranteeing the stability of the prioritized tasks. To the best of     Authors' knowledge, this work represents the first mathematical development of an SDP formulation that introduces stability conditions for a multi-objective closed-loop inverse kinematic problem for highly redundant robots. The validity of the proposed approach is demonstrated through simulation case studies, including didactic examples and a Matlab toolbox for the benefit of the community.

- Stable-By-Design Kinematic Control Based on Optimization (I)

    Author: Gon�alves, Vinicius Mariano | UFMG
    Author: Adorno, Bruno Vilhena | Federal University of Minas Gerais (UFMG)
    Author: Crosnier, André | LIRMM
    Author: Fraisse, Philippe | LIRMM
 
    keyword: Kinematics; Optimization and Optimal Control

    Abstract : This paper presents a new kinematic control paradigm for redundant robots based on optimization. The general approach takes into account convex objective functions with inequality constraints and a specific equality constraint resulting from a Lyapunov function, which ensures closed-loop stability by design. Furthermore, we tackle an important particular case by using a convex combination of quadratic and l_{1}-norm objective functions, making possible for the designer to choose different degrees of sparseness and smoothness in the control inputs. We provide a pseudo-analytical solution to this optimization problem and validate the approach by controlling the center of mass of the humanoid robot HOAP3.

## Robot Safety
- Securing Industrial Operators with Collaborative Robots: Simulation and Experimental Validation for a Carpentry Task

    Author: Benhabib, Nassim | Inria
    Author: Padois, Vincent | Inria Bordeaux
    Author: Daney, David | Inria Bordeaux - Sud Ouest
 
    keyword: Robot Safety; Physical Human-Robot Interaction; Physically Assistive Devices

    Abstract : In this work, a robotic assistance strategy is developed to improve the safety in an artisanal task that involves a strong interaction between a machine-tool and an operator. Wood milling is chosen as a pilot task due to its importance in carpentry and its accidentogenic aspect. A physical model of the tooling process including a human is proposed and a simulator is thereafter developed to better understand situations that are dangerous for the craftsman. This simulator is validated with experiments on three subjects using an harmless mock-up. This validation shows the pertinence of the proposed control approach for the collaborative robot used to increase the safety of the task.

- Learning Shape-Based Representation for Visual Localization in Extremely Changing Conditions

    Author: Jeon, Hae-Gon | GIST
    Author: Im, Sunghoon | DGIST
    Author: Oh, Jean | Carnegie Mellon University
    Author: Hebert, Martial | CMU
 
    keyword: Robot Safety; Computer Vision for Other Robotic Applications; Localization

    Abstract : Visual localization is an important task for applications such as navigation and augmented reality, but is a challenging problem when there are changes in scene appearances through day, seasons, or environments. In this paper, we present a convolutional neural network (CNN)-based approach for visual localization across normal to drastic appearance variations such as pre- and post-disaster cases. Our approach aims to address two key challenges: (1) to reduce the biases based on scene textures as in traditional CNNs, our model learns a shape-based representation by training on stylized images; (2) to make the model robust against layout changes, our approach uses the estimated dominant planes of query images as approximate scene coordinates. Our method is evaluated on various scenes including a simulated disaster dataset to demonstrate the effectiveness of our method in significant changes of scene layout. Experimental results show that our method provides reliable camera pose predictions in various changing conditions.

- Trajectory Planning with Safety Guaranty for a Multirotor Based on the Forward and Backward Reachability Analysis

    Author: Seo, Hoseong | Seoul National University
    Author: Son, Clark Youngdong | Seoul National University
    Author: Lee, Dongjae | Seoul National University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Robot Safety; Motion and Path Planning; Collision Avoidance

    Abstract : Planning a trajectory with guaranteed safety is a core part for a risk-free flight of a multirotor. If a trajectory planner only aims to ensure safety, it may generate trajectories which overly bypass risky regions and prevent the system from achieving specific missions. This work presents a robust trajectory planning algorithm which simultaneously guarantees the safety and reachability to the target state in the presence of unknown disturbances. We first characterize how the forward and backward reachable sets (FRSs and BRSs) are constructed by using Hamilton-Jacobi reachability analysis. Based on the analysis, we present analytic expressions for the reachable sets and then propose minimal ellipsoids which closely approximate the reachable sets. In the planning process, we optimize the reference trajectory to connect the FRSs and BRSs, while avoiding obstacles. By combining the FRSs and BRSs, we can guarantee that any state inside of the initial set reaches the target set. We validate the proposed algorithm through a simulation of traversing a narrow gap.

- A Hamilton-Jacobi Reachability-Based Framework for Predicting and Analyzing Human Motion for Safe Planning

    Author: Bansal, Somil | UC Berkeley
    Author: Bajcsy, Andrea | University of California Berkeley
    Author: Ratner, Ellis | University of California, Berkeley
    Author: Dragan, Anca | University of California Berkeley
    Author: Tomlin, Claire | UC Berkeley
 
    keyword: Robot Safety; Collision Avoidance; Cognitive Human-Robot Interaction

    Abstract : Real-world autonomous systems often employ probabilistic predictive models of human behavior during planning to reason about their future motion. Since accurately modeling human behavior a priori is challenging, such models are often parameterized, enabling the robot to adapt predictions based on observations by maintaining a distribution over the model parameters. Although this enables data and priors to improve the human model, observation models are difficult to specify and priors may be incorrect, leading to erroneous state predictions that can degrade the safety of the robot motion plan. In this work, we seek to design a predictor which is more robust to misspecified models and priors, but can still leverage human behavioral data online to reduce conservatism in a safe way. To do this, we cast human motion prediction as a Hamilton-Jacobi reachability problem in the joint state space of the human and the belief over the model parameters. We construct a new continuous-time dynamical system, where the inputs are the observations of human behavior, and the dynamics include how the belief over the model parameters change. The results of this reachability computation enable us to both analyze the effect of incorrect priors on future predictions in continuous state and time, as well as to make predictions of the human state in the future. We compare our approach to the worst-case forward reachable set and a stochastic predictor which produces full future state distributions.

- Enhancing Privacy in Robotics Via Judicious Sensor Selection

    Author: Eick, Stephen | Georgia Institute of Technology
    Author: Ant�n, Annie | Georgia Institute of Technology
 
    keyword: Ethics and Philosophy; Robot Safety

    Abstract : Roboticists are grappling with how to address privacy in robot design at a time when regulatory frameworks around the world increasingly require systems to be engineered to preserve and protect privacy. This paper surveys the top robotics journals and conferences over the past four decades to identify contributions with respect to privacy in robot design. Our survey revealed that less than half of one percent of the ~89,120 papers in our study even mention the word privacy. Herein, we propose privacy preserving approaches for roboticists to employ in robot design, including, assessing a robot's purpose and environment; ensuring privacy by design by selecting sensors that do not collect information that is not essential to the core objectives of that robot; embracing both privacy and performance as fundamental design challenges to be addressed early in the robot lifecycle; and performing privacy impact assessments.

- Robust Model Predictive Shielding for Safe Reinforcement Learning with Stochastic Dynamics

    Author: Li, Shuo | University of Pennsylvania
    Author: Bastani, Osbert | University of Pennsylvania
 
    keyword: Robot Safety; Robust/Adaptive Control of Robotic Systems; Deep Learning in Robotics and Automation

    Abstract : We propose a framework for safe reinforcement learning that can handle stochastic nonlinear dynamical systems. We focus on the setting where the nominal dynamics are known, and are subject to additive stochastic disturbances with known distribution. Our goal is to ensure the safety of a control policy trained using reinforcement learning, e.g., in a simulated environment. We build on the idea of model predictive shielding (MPS), where a backup controller is used to override the learned policy as needed to ensure safety. The key challenge is how to compute a backup policy in the context of stochastic dynamics. We propose to use a tube-based robust nonlinear model predictive controller (NMPC) as the backup controller. We estimate the tubes using sampled trajectories, leveraging ideas from statistical learning theory to obtain high-probability guarantees. We empirically demonstrate that our approach can ensure safety in stochastic systems, including cart-pole and a non-holonomic particle with random obstacles.

## Swarms
- Segregation of Heterogeneous Swarms of Robots in Curves

    Author: Bernardes Ferreira Filho, Edson | Universidade Federal De Minas Gerais
    Author: Pimenta, Luciano | Universidade Federal De Minas Gerais
 
    keyword: Swarms; Multi-Robot Systems; Cooperating Robots

    Abstract : This paper proposes a decentralized control strategy to reach segregation in heterogeneous robot swarms distributed in curves. The approach is based on a formation control algorithm applied to each robot and a heuristics to compute the distance between the groups, i.e. the distance from the beginning of the curve. We consider that robots can communicate through a fixed underlying topology and also when they are within a certain distance. A convergence proof with a collision avoidance strategy is presented. Simulations and experimental results show that our approach allows a swarm of multiple heterogeneous robots to segregate into groups.

- A Fast, Accurate, and Scalable Probabilistic Sample-Based Approach for Counting Swarm Size

    Author: Wang, Hanlin | Northwestern University
    Author: Rubenstein, Michael | Northwestern University
 
    keyword: Swarms; Multi-Robot Systems; Sensor Networks

    Abstract : This paper describes a distributed algorithm for computing the number of robots in a swarm, only requiring communication with neighboring robots.	The algorithm can adjust the estimated count when the number of robots in the swarm changes, such as the addition or removal of robots. Probabilistic guarantees are given, which show the accuracy of this method, and the trade-off between accuracy, speed, and adaptability to changing numbers. The proposed approach is demonstrated in simulation as well as a real swarm of robots.

- Bayes Bots: Collective Bayesian Decision-Making in Decentralized Robot Swarms

    Author: Ebert, Julia T | Harvard University
    Author: Gauci, Melvin | Harvard University
    Author: Mallmann-Trenn, Frederik | King's College London
    Author: Nagpal, Radhika | Harvard University
 
    keyword: Swarms; Multi-Robot Systems; Autonomous Agents

    Abstract : We present a distributed Bayesian algorithm for robot swarms to classify a spatially distributed feature of an environment. This type of ``go/no-go'' decision appears in applications where a group of robots must collectively choose whether to take action, such as determining if a farm field should be treated for pests. Previous bio-inspired approaches to decentralized decision-making in robotics lack a statistical foundation, while decentralized Bayesian algorithms typically require a strongly connected network of robots. In contrast, our algorithm allows simple, sparsely distributed robots to quickly reach accurate decisions about a binary feature of their environment. We investigate the speed vs. accuracy tradeoff in decision-making by varying the algorithm's parameters. We show that making fewer, less-correlated observations can improve decision-making accuracy, and that a well-chosen combination of prior and decision threshold allows for fast decisions with a small accuracy cost. Both speed and accuracy also improved with the addition of bio-inspired positive feedback. This algorithm is also adaptable to the difficulty of the environment. Compared to a fixed-time benchmark algorithm with accuracy guarantees, our Bayesian approach resulted in equally accurate decisions, while adapting its decision time to the difficulty of the environment.

- Supervisory Control of Robot Swarms Using Public Events

    Author: Kaszubowski Lopes, Yuri | Federal University of Technology - Paran�
    Author: Trenkwalder, Stefan M. | The University of Sheffield
    Author: Leal, André Bittencourt | Santa Catarina State University - UDESC
    Author: Dodd, Tony J | The University of Sheffield
    Author: Gross, Roderich | The University of Sheffield
 
    keyword: Swarms; Distributed Robot Systems; Multi-Robot Systems

    Abstract : Supervisory Control Theory (SCT) provides a formal framework for controlling	discrete event systems. It has recently been used	to generate correct-by-construction controllers for swarm robotics systems. Current SCT	frameworks are limited, as they support only (private) events that are observable within the same robot. In this paper, we propose an extended SCT framework that incorporates (public) events that are shared among robots. The extended framework	allows	to model formally the interactions among the robots. It is evaluated using a case study, where a group of mobile robots need to synchronise their movements in space	and time�a requirement that is specified at the formal level.	We validate our approach through experiments with groups of e-puck robots.

- Planetary Exploration with Robot Teams (I)

    Author: St-Onge, David | Ecole De Technologie Superieure
    Author: Kaufmann, Marcel | Polytechnique Montreal
    Author: Panerati, Jacopo | Polytechnique Montreal
    Author: Ramtoula, Benjamin | École Polytechnique De Montr�al, École Polytechnique Fédérale De
    Author: Cao, Yanjun | École Polytechnique De Montr�al (Université De Montr�al)
    Author: Coffey, Emily | Department of Psychology, Concordia University
    Author: Beltrame, Giovanni | Ecole Polytechnique De Montreal
 
    keyword: Swarms; Cognitive Human-Robot Interaction; Human Factors and Human-in-the-Loop

    Abstract : Since the beginning of space exploration, Mars and the Moon have been explored with orbiters, landers, and rovers. Over forty missions have targeted Mars, and more than a hundred, the Moon. Developing novel strategies and technologies for exploring celestial bodies continues to be a focus of space agencies. Multi-robot systems are particularly promising for planetary exploration, as they are more robust to individual failure and have the potential to explore larger areas; however, there are limits to how many robots an operator can individually control. We recently took part in the European Space Agency's interdisciplinary equipment test campaign (PANGAEA-X) at a Lunar/Mars analogue site in Lanzarote, Spain. We used a heterogeneous fleet of Unmanned Aerial Vehicles(UAVs)---a swarm---to study the interplay of systems operations and human factors. Human operators directed the swarm via ad-hoc networks and data sharing protocols to explore unknown areas under two control modes: one in which the operator instructed each robot separately; and the other in which the operator provided general guidance to the swarm, which self-organized via a combination of distributed decision-making, and consensus building. We assessed cognitive load via pupillometry for each condition, and perceived task demand and intuitiveness via self-report. Our results show that implementing higher autonomy with swarm intelligence can reduce workload, freeing the operator for other tasks.

- Statistics-Based Automated Control for a Swarm of Paramagnetic Nanoparticles in 2D Space (I)

    Author: Yang, Lidong | The Chinese University of Hong Kong
    Author: Yu, Jiangfan | University of Toronto
    Author: Zhang, Li | The Chinese University of Hong Kong
 
    keyword: Micro/Nano Robots; Automation at Micro-Nano Scales; Swarms

    Abstract : Swarm control is one of the primary challenges in microrobotics. For the automated control of such a microrobotic system with small size and large population, conventional methods using precise robot models and robot�robot communications lose effectiveness due to the complex locomotion of micro/nano agents in a swarm and difficult implementation of onboard actuators and sensors for individual motion control and motion feedback. This article proposes a statistics-based approach and reports the fully automated control of a swarm of paramagnetic nanoparticles including the swarm pattern formation, identification, tracking, motion control, and real-time distribution monitoring/control. By establishing the swarm statistics, collective behaviors of a nanoparticle swarm can be quantitatively analyzed by computers. Algorithms are designed based on the statistics to automatically generate and identify the vortex-like paramagnetic nanoparticle swarm (VPNS), which present robustness to the dose and initial distribution of the nanoparticle swarm. In order to robustly track a VPNS, a statistics-based tracking method is proposed, in which 500 boundary points of the VPNS are extracted and the VPNS distribution is optimally recognized. And, with the proposed gathering improvement control, experiments show that over 70% nanoparticles can be gathered in the VPNS. Furthermore, an automated motion control scheme for the VPNS is proposed which shows high-accuracy trajectory tracking performance.

## Simulation and Animation
- Automatic Tool for Gazebo World Construction: From a Grayscale Image to a 3D Solid Model

    Author: Abbyasov, Bulat | Kazan Federal University
    Author: Lavrenov, Roman | Kazan Federal University
    Author: Zakiev, Aufar | Kazan Federal University
    Author: Yakovlev, Konstantin | Federal Research Center "Computer Science and Control" of Russia
    Author: Svinin, Mikhail | Ritsumeikan University
    Author: Magid, Evgeni | Kazan Federal University
 
    keyword: Simulation and Animation; SLAM; Performance Evaluation and Benchmarking

    Abstract : Robot simulators provide an easy way for evaluation of new concepts and algorithms in a simulated physical environment reducing development time and cost. Therefore it is convenient to have a tool that quickly creates a 3D landscape from an arbitrary 2D image or 2D laser range finder data. This paper presents a new tool that automatically constructs such landscapes for Gazebo simulator. The tool converts a grayscale image into a 3D Collada format model, which could be directly imported into Gazebo. We run three different simultaneous localization and mapping (SLAM) algorithms within three varying complexity environments that were constructed with our tool. A real-time factor (RTF) was used as an efficiency benchmark. Successfully completed SLAM missions with acceptable RTF levels demonstrated the efficiency of the tool. The source code is available for free academic use.

- A ROS Gazebo Plugin to Simulate ARVA Sensors

    Author: Cacace, Jonathan | University of Naples
    Author: Mimmo, Nicola | University of Bologna
    Author: Marconi, Lorenzo | University of Bologna
 
    keyword: Simulation and Animation; Sensor-based Control

    Abstract : This paper addresses the problem to simulate ARVA sensors using ROS and Gazebo. ARVA is a French acronym which stands for Appareil de Recherche de Victims en Avalanche and represents the forefront technology adopted in Search &amp; Rescue operations to localize victims of avalanches buried under the snow. The aim of this paper is to describe the mathematical and theoretical background of the transceiver, discussing its implementation and integration with ROS allowing researchers to develop faster and smarter Search &amp; Rescue strategies based on ARVA receiver data. To assess the effectiveness of the proposed sensor model, We present a simulation scenario in which an Unmanned Aerial Vehicle equipped with the transceiver sensor performs a basic S&amp;R pattern using the output of ARVA system.

- Is That a Chair? Imagining Affordances Using Simulations of an Articulated Human Body

    Author: Wu, Hongtao | Johns Hopkins University
    Author: Misra, Deven | Reed College
    Author: Chirikjian, Gregory | Johns Hopkins University
 
    keyword: Simulation and Animation; AI-Based Methods; Humanoid Robots

    Abstract : For robots to exhibit a high level of intelligence in the real world, they must be able to assess objects for which they have no prior knowledge. Therefore, it is crucial for robots to perceive object affordances by reasoning about physical interactions with the object. In this paper, we propose a novel method to provide robots with an ability to imagine object affordances using physical simulations. The class of chair is chosen here as an initial category of objects to illustrate a more general paradigm. In our method, the robot �imagines' the affordance of an arbitrarily oriented object as a chair by simulating a physical sitting interaction between an articulated human body and the object. This object affordance reasoning is used as a cue for object classification (chair vs non-chair). Moreover, if an object is classified as a chair, the affordance reasoning can also predict the upright pose of the object which allows the sitting interaction to take place. We call this type of poses the functional pose. We demonstrate our method in chair classification on synthetic 3D CAD models. Although our method uses only 30 models for training, it outperforms appearance-based deep learning methods, which require a large amount of training data, when the upright orientation is not assumed to be known a priori. In addition, we showcase that the functional pose predictions of our method align well with human judgments on both synthetic models and real objects scanned by a depth camera.

- Toward Sim-To-Real Directional Semantic Grasping

    Author: Iqbal, Shariq | University of Southern California
    Author: Tremblay, Jonathan | Nvidia
    Author: To, Thang | Nvidia Corp
    Author: Cheng, Jia | Nvidia Corp
    Author: Leitch, Erik | Nvidia
    Author: Campbell, Andy | NVIDIA
    Author: Leung, Kirby | Nvidia
    Author: McKay, Duncan | NVIDIA
    Author: Birchfield, Stan | NVIDIA
 
    keyword: Simulation and Animation; Computer Vision for Automation; Deep Learning in Robotics and Automation

    Abstract : We address the problem of directional semantic grasping, that is, grasping a specific object from a specific direction. We approach the problem using deep reinforcement learning via a double deep Q-network (DDQN) that learns to map downsampled RGB input images from a wrist-mounted camera to Q-values, which are then translated into Cartesian robot control commands via the cross-entropy method (CEM). The network is learned entirely on simulated data generated by a custom robot simulator that models both physical reality (contacts) and perceptual quality (high-quality rendering). The reality gap is bridged using domain randomization. The system is an example of end-to-end (mapping input monocular RGB images to output Cartesian motor commands) grasping of objects from multiple pre-defined object-centric orientations, such as from the side or top. We show promising results in both simulation and the real world, along with some challenges faced and the need for future research in this area.

- Learning to Collaborate from Simulation for Robot-Assisted Dressing

    Author: Clegg, Alexander | Georgia Institute of Technology
    Author: Erickson, Zackory | Georgia Institute of Technology
    Author: Grady, Patrick | Georgia Institute of Technology
    Author: Turk, Greg | Georgia Institute of Technology
    Author: Kemp, Charlie | Georgia Institute of Technology
    Author: Liu, Karen | Georgia Tech
 
    keyword: Simulation and Animation; Deep Learning in Robotics and Automation; Physically Assistive Devices

    Abstract : We investigated the application of haptic feedback control and deep reinforcement learning (DRL) to robot-assisted dressing. Our method uses DRL to simultaneously train human and robot control policies as separate neural networks using physics simulations. In addition, we modeled variations in human impairments relevant to dressing, including unilateral muscle weakness, involuntary arm motion, and limited range of motion. Our approach resulted in control policies that successfully collaborate in a variety of simulated dressing tasks involving a hospital gown and a T-shirt. In addition, our approach resulted in policies trained in simulation that enabled a real PR2 robot to dress the arm of a humanoid robot with a hospital gown. We found that training policies for specific impairments dramatically improved performance; that controller execution speed could be scaled after training to reduce the robot's speed without steep reductions in performance; that curriculum learning could be used to lower applied forces; and that multi-modal sensing, including a simulated capacitive sensor, improved performance.

- Realtime Simulation of Thin-Shell Deformable Materials Using CNN-Based Mesh Embedding

    Author: Tan, Qingyang | University of Maryland at College Park
    Author: Pan, Zherong | The University of North Carolina at Chapel Hill
    Author: Gao, Lin | Institute of Computing Technology, Chinese Academy of Sciences
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Simulation and Animation; Dexterous Manipulation

    Abstract : We address the problem of accelerating thin-shell deformable object simulations by dimension reduction. We present a new algorithm to embed a high-dimensional configuration space of deformable objects in a low-dimensional feature space, where the configurations of objects and feature points have approximate one-to-one mapping. Our key technique is a graph-based convolutional neural network (CNN) defined on meshes with arbitrary topologies and a new mesh embedding approach based on physics-inspired loss term. We have applied our approach to accelerate high-resolution thin shell simulations corresponding to cloth-like materials, where the configuration space has tens of thousands of degrees of freedom. We show that our physics-inspired embedding approach leads to higher accuracy compared with prior mesh embedding methods. Finally, we show that the temporal evolution of the mesh in the feature space can also be learned using a recurrent neural network (RNN) leading to fully learnable physics simulators. After training our learned simulator runs 10&#8722;100� faster and the accuracy is high enough for robot manipulation tasks.

## Reinforcement Learning for Robotics

- Dynamic Actor-Advisor Programming for Scalable Safe Reinforcement Learning

    Author: Zhu, Lingwei | Nara Institute of Science and Technology
    Author: Cui, Yunduan | Nara Institute of Science and Technology
    Author: Matsubara, Takamitsu | Nara Institute of Science and Technology
 
    keyword: Learning and Adaptive Systems; Autonomous Agents

    Abstract : Real-world robots have complex strict constraints. Therefore, safe reinforcement learning algorithms that can simultaneously minimize the total cost and the risk of constraint violation are crucial. However, almost no algorithms exist that can scale to high-dimensional systems to the best of our knowledge. In this paper, we propose Dynamic Actor-Advisor Programming (DAAP), as an algorithm for sample-efficient and scalable safe reinforcement learning. DAAP employs two control policies, actor and advisor. They are updated to minimize total cost and risk of constraint violation intertwiningly and smoothly towards each other's direction by using the other as the baseline policy in the Kullback-Leibler divergence of Dynamic Policy Programming framework. We demonstrate the scalability and sample efficiency of DAAP through its application on simulated and real-robot arm control tasks with performance comparisons to baselines.

- Discrete Deep Reinforcement Learning for Mapless Navigation

    Author: Marchesini, Enrico | University of Verona
    Author: Farinelli, Alessandro | University of Verona
 
    keyword: Learning and Adaptive Systems; Autonomous Agents; Deep Learning in Robotics and Automation

    Abstract : Our goal is to investigate whether discrete state space algorithms are a viable solution to continuous alternatives for mapless navigation. To this end we present an approach based on Double Deep Q-Network and employ parallel asynchronous training and a multi-batch Priority Experience Replay to reduce the training time. Experiments show that our method trains faster and outperforms both the continuous Deep Deterministic Policy Gradient and Proximal Policy Optimization algorithms. Moreover, we train the models in a custom environment built on the recent Unity learning toolkit and show that they can be exported on the TurtleBot3 simulator and to the real robot without further training. Overall our optimized method is 40% faster compared to the original discrete algorithm. This setting significantly reduces the training times with respect to the continuous algorithms, maintaining a similar level of success rate hence being a viable alternative for mapless navigation.

- Learning Multi-Robot Decentralized Macro-Action-Based Policies Via a Centralized Q-Net

    Author: Xiao, Yuchen | Northeastern Univerisity
    Author: Hoffman, Joshua | Northeastern University
    Author: Xia, Tian | Northeastern University
    Author: Amato, Christopher | Northeastern University
 
    keyword: AI-Based Methods; Multi-Robot Systems; Deep Learning in Robotics and Automation

    Abstract : In many real-world multi-robot tasks, high-quality solutions often require a team of robots to perform asynchronous actions under decentralized control. Decentralized multi-agent reinforcement learning methods have difficulty learning decentralized policies because of the environment appearing to be non-stationary due to other agents also learning at the same time. In this paper, we address this challenge by proposing a macro-action-based decentralized multi-agent double deep recurrent Q-net (MacDec-MADDRQN) which trains each decentralized Q-net using a centralized Q-net for action selection. A generalized version of MacDec-MADDRQN with two separate training environments, called Parallel-MacDec- MADDRQN, is also presented to leverage either centralized or decentralized exploration. The advantages and the practical nature of our methods are demonstrated by achieving near- centralized results in simulation and having real robots accomplish a warehouse tool delivery task in an efficient way.

- Robust Model-Free Reinforcement Learning with Multi-Objective Bayesian Optimization

    Author: Turchetta, Matteo | ETH Zurich
    Author: Krause, Andreas | ETH Zurich
    Author: Trimpe, Sebastian | Max Planck Institute for Intelligent Systems
 
    keyword: Learning and Adaptive Systems

    Abstract : In reinforcement learning (RL), an autonomous agent learns to perform complex tasks by maximizing an exogenous reward signal while interacting with its environment. In real-world applications, test conditions may differ substantially from the training scenario and, therefore, focusing on pure reward maximization during training may lead to poor results at test time. In these cases, it is important to trade-off between performance and robustness while learning a policy. While several results exist for robust, model-based RL, the model-free case has not been widely investigated. In this paper, we cast the robust, model-free RL problem as a multi-objective optimization problem. To quantify the robustness of a policy, we use delay margin and gain margin, two robustness indicators that are common in control theory. We show how these metrics can be estimated from data in the model-free setting. We use multi-objective Bayesian optimization (MOBO) to solve efficiently this expensive-to-evaluate, multi-objective optimization problem. We show the benefits of our robust formulation both in sim-to-real and pure hardware experiments to balance a Furuta pendulum.

- Motor Synergy Development in High-Performing Deep Reinforcement Learning Algorithms

    Author: Chai, Jiazheng | Tohoku University
    Author: Hayashibe, Mitsuhiro | Tohoku University
 
    keyword: Deep Learning in Robotics and Automation; Performance Evaluation and Benchmarking

    Abstract : As human motor learning is hypothesized to use the motor synergy concept, we investigate if this concept could also be observed in deep reinforcement learning for robotics. From this point of view, we carried out a joint-space synergy analysis on multi-joint running agents in simulated environments trained using two state-of-the-art deep reinforcement learning algorithms. Although a synergy constraint has never been encoded into the reward function, the synergy emergence phenomenon could be observed statistically in the learning agent. To our knowledge, this is the first attempt to quantify the synergy development in detail and evaluate its emergence process during deep learning motor control tasks. We then demonstrate that there is a correlation between our synergy-related metrics and the performance and energy efficiency of a trained agent. Interestingly, the proposed synergy-related metrics reflected a better learning capability of SAC over TD3. It suggests that these metrics could be additional new indices to evaluate deep reinforcement learning algorithms for motor learning. It also indicates that synergy is required for multi-joints robots to move energy-efficiently.

- Barrier-Certified Adaptive Reinforcement Learning with Applications to Brushbot Navigation (I)

    Author: Ohnishi, Motoya | Paul G. Allen School of Computer Science &amp; Engineering
    Author: Wang, Li | Georgia Institute of Technology
    Author: Notomista, Gennaro | Georgia Institute of Technology
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Learning and Adaptive Systems; Robot Safety; Model Learning for Control

    Abstract : This paper presents a safe learning framework that employs an adaptive model learning algorithm together with barrier certificates for systems with possibly nonstationary agent dynamics. To extract the dynamic structure of the model, we use a sparse optimization technique. We use the learned model in combination with control barrier certificates that constrain policies (feedback controllers) in order to maintain safety, which refers to avoiding particular undesirable regions of the state space. Under certain conditions, recovery of safety in the sense of Lyapunov stability after violations of safety due to the nonstationarity is guaranteed.	In addition, we reformulate an action-value function approximation to make any kernel-based nonlinear function estimation method applicable to our adaptive learning framework. Lastly, solutions to the barrier-certified policy optimization are guaranteed to be globally optimal, ensuring the greedy policy improvement under mild conditions. The resulting framework is validated via simulations of a quadrotor, which has previously been used under stationarity assumptions in the safe learnings literature, and is then tested on a real robot, the brushbot, whose dynamics is unknown, highly complex, and nonstationary.


- On Simple Reactive Neural Networks for Behaviour-Based Reinforcement Learning

    Author: Pore, Ameya | University of Glasgow
    Author: Aragon-Camarasa, Gerardo | University of Glasgow
 
    keyword: Deep Learning in Robotics and Automation; Dexterous Manipulation; Grasping

    Abstract : We present a behaviour-based reinforcement learning approach, inspired by Brook's subsumption architecture, in which simple fully connected networks are trained as reactive behaviours. Our working assumption is that a pick and place robotic task can be simplified by leveraging domain knowledge of a robotics developer to decompose and train reactive behaviours; namely, approach, grasp, and retract. Then the robot autonomously learns how to combine reactive behaviours via an Actor-Critic architecture. We use an Actor-Critic policy to determine the activation and inhibition mechanisms of the reactive behaviours in a particular temporal sequence. We validate our approach in a simulated robot environment where the task is about picking a block and taking it to a target position while orienting the gripper from a top grasp. The latter represents an extra degree-of-freedom of which current end-to-end reinforcement learning approaches fail to generalise. Our findings suggest that robotic learning can be more effective if each behaviour is learnt in isolation and then combined them to accomplish the task. That is, our approach learns the pick and place task in 8,000 episodes, which represents a drastic reduction in the number of training episodes required by an end-to-end approach (~95,000 episodes) and existing state-of-the-art algorithms.

- Predicting Optimal Value Functions by Interpolating Reward Functions in Scalarized Multi-Objective Reinforcement Learning

    Author: Kusari, Arpan | Ford Motor Company
    Author: How, Jonathan Patrick | Massachusetts Institute of Technology
 
    keyword: Deep Learning in Robotics and Automation; AI-Based Methods; Learning and Adaptive Systems

    Abstract : A common approach for defining a reward function for multi-objective reinforcement learning (MORL) problems is the weighted sum of the multiple objectives. The weights are then treated as design parameters dependent on the expertise (and preference) of the person performing the learning, with the typical result that a new solution is required for any change in these settings. This paper investigates the relationship between the reward function and the optimal value function for MORL; specifically addressing the question of how to approximate the optimal value function well beyond the set of weights for which the optimization problem was actually solved, thereby avoiding the need to recompute for any particular choice. We prove that the value function transforms smoothly given a transformation of weights of the reward function (and thus a smooth interpolation in the policy space). A Gaussian process is used to obtain a smooth interpolation over the reward function weights of the optimal value function for three well-known examples: Gridworld, Objectworld and Pendulum. The results show that the interpolation can provide robust values for sample states and actions in both discrete and continuous domain problems. Significant advantages arise from utilizing this interpolation technique in the domain of autonomous vehicles: easy, instant adaptation of user preferences while driving and true randomization of obstacle vehicle behavior preferences during training.

- Integrated Moment-Based LGMD and Deep Reinforcement Learning for UAV Obstacle Avoidance

    Author: He, Lei | Northwestern Polytechnical University
    Author: Aouf, Nabil | City University of London
    Author: Whidborne, James | Cranfield University
    Author: Song, Bifeng | Northwestern Polytechnical University
 
    keyword: Deep Learning in Robotics and Automation; Visual-Based Navigation; Collision Avoidance

    Abstract : In this paper, a bio-inspired monocular vision perception method combined with a learning-based reaction local planner for obstacle avoidance of micro UAVs is presented. The system is more computationally efficient than other vision-based perception and navigation methods such as SLAM and optical flow because it does not need to calculate accurate distances. To improve the robustness of perception against illuminance change, the input image is remapped using image moment which is independent of illuminance variation. After perception, a local planner is trained using deep reinforcement learning for mapless navigation. The proposed perception and navigation methods are evaluated in some realistic simulation environments. The result shows that this light-weight monocular perception and navigation system works well in different complex environments without accurate depth information.

- Interactive Reinforcement Learning with Inaccurate Feedback

    Author: Kessler Faulkner, Taylor | University of Texas at Austin
    Author: Short, Elaine Schaertl | Tufts University
    Author: Thomaz, Andrea Lockerd | University of Texas at Austin
 
    keyword: Learning and Adaptive Systems; Human Factors and Human-in-the-Loop

    Abstract : Interactive Reinforcement Learning (RL) enables agents to learn from two sources: rewards taken from observations of the environment, and feedback or advice from a secondary critic source, such as human teachers or sensor feedback. The addition of information from a critic during the learning process allows the agents to learn more quickly than non-interactive RL. There are many methods that allow policy feedback or advice to be combined with RL. However, critics can often give imperfect information. In this work, we introduce a framework for characterizing Interactive RL methods with imperfect teachers and propose an algorithm, Revision Estimation from Partially Incorrect Resources (REPaIR), which can estimate corrections to imperfect feedback over time. We run experiments both in simulations and demonstrate performance on a physical robot, and find that when baseline algorithms do not have prior information on the exact quality of a feedback source, using REPaIR matches or improves the expected performance of these algorithms.

- Guided Uncertainty-Aware Policy Optimization: Combining Model-Free and Model-Based Strategies for Sample-Efficient Learning

    Author: Lee, Michelle | Stanford University
    Author: Florensa, Carlos | UC Berkeley
    Author: Tremblay, Jonathan | Nvidia
    Author: Ratliff, Nathan | Lula Robotics Inc
    Author: Garg, Animesh | University of Toronto
    Author: Ramos, Fabio | University of Sydney, NVIDIA
    Author: Fox, Dieter | University of Washington
 
    keyword: Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation; Learning and Adaptive Systems

    Abstract : Traditional robotic approaches rely on an accurate model of the environment, a detailed description of how to perform the task, and a robust perception system to keep track of the current state. On the other hand, Reinforcement Learning (RL) approaches can operate directly from raw sensory inputs with only a reward signal to describe the task, but are extremely sample-inefficient and brittle. In this work we combine the strengths of both to obtain a general method able to overcome inaccuracies of the elements in the traditional pipeline, while requiring minimal interaction with the environment. This is achieved by leveraging uncertainty estimates to divide the space in regions where the given model-based policy is reliable, and regions where it may have flaws or not be well defined. In these hard regions, we show that a local model-free policy can be learned directly from raw sensory inputs. This allows to build robotic systems faster from simple and cheap components, and only a high-level description of the task. We test our algorithm, Guided Uncertainty-Aware Policy Optimization (GUAPO), in a real-world robot performing tight-fitting peg insertion.

- High-Speed Autonomous Drifting with Deep Reinforcement Learning

    Author: Cai, Peide | Hong Kong University of Science and Technology
    Author: Mei, Xiaodong | HKUST
    Author: Tai, Lei | Alibaba Group
    Author: Sun, Yuxiang | Hong Kong University of Science and Technology
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Automation Technologies for Smart Cities; Service Robots; Field Robots

    Abstract : Drifting is a complicated task for autonomous vehicle control. Most traditional methods in this area are based on motion equations derived by the understanding of vehicle dynamics, which is difficult to be modeled precisely. We propose a robust drifting controller without explicit motion equations, which is based on the latest model-free deep reinforcement learning algorithm Soft Actor-Critic. The drift control problem is formulated as a trajectory following task, where error-based state and reward are designed. After trained on tracks with different levels of difficulty, our controller is capable of making the vehicle drift through various sharp corners safely and stably in the tough map. The proposed controller is further proved to have excellent generalization ability, which can directly handle unseen vehicle types with different physics properties, such as mass, tire friction, etc.

## Manipulation Planning

- Non-Prehensile Manipulation in Clutter with Human-In-The-Loop

    Author: Papallas, Rafael | The University of Leeds
    Author: Dogar, Mehmet R | University of Leeds
 
    keyword: Manipulation Planning; Human Factors and Human-in-the-Loop

    Abstract : We propose a human-operator guided planning approach to pushing-based manipulation in clutter. Most recent approaches to manipulation in clutter employs randomized planning. The problem, however, remains a challenging one where the planning times are still in the order of tens of seconds or minutes, and the success rates are low for difficult instances of the problem. We build on these control-based randomized planning approaches, but we investigate using them in conjunction with human-operator input. In our framework, the human operator supplies a high-level plan, in the form of an ordered sequence of objects and their approximate goal positions. We present experiments in simulation and on a real robotic setup, where we compare the success rate and planning times of our human-in-the-loop approach with fully autonomous sampling-based planners. We show that with a minimal amount of human input, the low-level planner can solve the problem faster and with higher success rates.

- PuzzleFlex: Kinematic Motion of Chains with Loose Joints

    Author: Lensgraf, Samuel | Dartmouth College
    Author: Itani, Karim | Dartmouth College
    Author: Zhang, Yinan | Dartmouth College
    Author: Sun, Zezhou | Boston University
    Author: Wu, Yijia | Beihang University
    Author: Quattrini Li, Alberto | Dartmouth College
    Author: Zhu, Bo | Dartmouth College
    Author: Whiting, Emily | Boston University
    Author: Wang, Weifu | University at Albany, SUNY
    Author: Balkcom, Devin | Dartmouth College
 
    keyword: Manipulation Planning; Assembly; Cellular and Modular Robots

    Abstract : This paper presents a method of computing free motions of a planar assembly of rigid bodies connected by loose joints. Joints are modeled using local distance constraints, which are then linearized with respect to con- figuration space velocities, yielding a linear programming formulation that allows analysis of systems with thousands of rigid bodies. Potential applications include analysis of collections of modular robots, structural stability perturbation analysis, tolerance analysis for mechanical systems, and formation control of mobile robots.

- Accurate Vision-Based Manipulation through Contact Reasoning

    Author: Kloss, Alina | Max-Planck-Institute for Intelligent Systems
    Author: Bauza Villalonga, Maria | Massachusetts Institute of Technology
    Author: Wu, Jiajun | Stanford University
    Author: Tenenbaum, Joshua | Massachusetts Institute of Technology
    Author: Rodriguez, Alberto | Massachusetts Institute of Technology
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Manipulation Planning; Contact Modeling; Perception for Grasping and Manipulation

    Abstract : Planning contact interactions is one of the core challenges of many robotic tasks. Optimizing contact locations while taking dynamics into account is computationally costly and, in environments that are only partially observable, executing contact-based tasks often suffers from low accuracy. We present an approach that addresses these two challenges for the problem of vision-based manipulation. First, we propose to disentangle contact from motion optimization. Thereby, we improve planning efficiency by focusing computation on promising contact locations. Second, we use a hybrid approach for perception and state estimation that combines neural networks with a physically meaningful state representation. In simulation and real-world experiments on the task of planar pushing, we show that our method is more efficient and achieves a higher manipulation accuracy than previous vision-based approaches.

- A Probabilistic Framework for Constrained Manipulations and Task and Motion Planning under Uncertainty

    Author: Ha, Jung-Su | University of Stuttgart
    Author: Driess, Danny | University of Stuttgart
    Author: Toussaint, Marc | Tu Berlin
 
    keyword: Manipulation Planning; Optimization and Optimal Control; Robust/Adaptive Control of Robotic Systems

    Abstract : Logic-Geometric Programming (LGP) is a powerful motion and manipulation planning framework, which represents hierarchical structure using logic rules that describe discrete aspects of problems, e.g., touch, grasp, hit, or push, and solves the resulting smooth trajectory optimization. The expressive power of logic allows LGP for handling complex, large-scale sequential manipulation and tool-use planning problems. In this paper, we extend the LGP formulation to stochastic domains. Based on the control-inference duality, we interpret LGP in a stochastic domain as fitting a mixture of Gaussians to the posterior path distribution, where each logic profile defines a single Gaussian path distribution. The proposed framework enables a robot to prioritize various interaction modes and to acquire interesting behaviors such as contact exploitation for uncertainty reduction, eventually providing a composite control scheme that is reactive to disturbance.

- Planning with Selective Physics-Based Simulation for Manipulation among Movable Objects

    Author: Saleem, Muhammad Suhail | Carnegie Mellon University
    Author: Likhachev, Maxim | Carnegie Mellon University
 
    keyword: Manipulation Planning; Motion and Path Planning

    Abstract : Use of physics-based simulation as a planning model enables a planner to reason and generate plans that involve non-trivial interactions with the world. For example, grasping a milk container out of a cluttered refrigerator may involve moving a robot manipulator in between other objects, pushing away the ones that are movable and avoiding interactions with certain fragile containers. A physics-based simulator allows a planner to reason about the effects of interactions with these objects and to generate a plan that grasps the milk container successfully. The use of physics-based simulation for planning however is underutilized. One of the reasons for it being that physics-based simulations are typically way too slow for being used within a planning loop that typically requires tens of thousands of actions to be evaluated within a matter of a second or two. In this work, we develop a planning algorithm that tries to address this challenge. In particular, it builds on the observation that only a small number of actions actually need to be simulated using physics, and the remaining set of actions, such as moving an arm around obstacles, can be evaluated using a much simpler internal planning model, e.g., a simple collision-checking model. Motivated by this, we develop an algorithm called Planning with Selective Physics-based Simulation that automatically discovers what should be simulated with physics and what can utilize an internal planning model for pick-and-place tasks.

- Hybrid Differential Dynamic Programming for Planar Manipulation Primitives

    Author: Doshi, Neel | MIT
    Author: Hogan, Francois | Massachusetts Institute of Technology
    Author: Rodriguez, Alberto | Massachusetts Institute of Technology
 
    keyword: Manipulation Planning; Dexterous Manipulation; Optimization and Optimal Control

    Abstract : We present a hybrid differential dynamic programming (DDP) algorithm for closed-loop execution of manipulation primitives with frictional contact switches. Planning and control of these primitives is challenging as they are hybrid, under-actuated, and stochastic. We address this by developing hybrid DDP both to plan finite horizon trajectories with a few contact switches and to create linear stabilizing controllers. We evaluate the performance and computational cost of our framework in ablations studies for two primitives: planar pushing and planar pivoting. We find that generating pose-to-pose closed-loop trajectories from most configurations requires only a couple (one to two) hybrid switches and can be done in reasonable time (one to five seconds). We further demonstrate that our controller stabilizes these hybrid trajectories on a real pushing system. A video describing our work can be found at https://youtu.be/YGSe4cUfq6Q.


- Human-Like Planning for Reaching in Cluttered Environments

    Author: Hasan, Mohamed | University of Leeds
    Author: Warburton, Matthew | University of Leeds
    Author: Agboh, Wisdom C. | University of Leeds
    Author: Dogar, Mehmet R | University of Leeds
    Author: Leonetti, Matteo | University of Leeds
    Author: Wang, He | University of Leeds
    Author: Mushtaq, Faisal | University of Leeds
    Author: Mon-Williams, Mark | University of Leeds
    Author: Cohn, Anthony | University of Leeds
 
    keyword: Manipulation Planning; Learning from Demonstration

    Abstract : Humans, in comparison to robots, are remarkably adept at reaching for objects in cluttered environments. The best existing robot planners are based on random sampling of configuration space- which becomes excessively high-dimensional with large number of objects. Consequently, most planners often fail to efficiently find object manipulation plans in such environments. We addressed this problem by identifying high-level manipulation plans in humans, and transferring these skills to robot planners. We used virtual reality to capture human participants reaching for a target object on a tabletop cluttered with obstacles. From this, we devised a qualitative representation of the task space to     Abstract the decision making, irrespective of the number of obstacles. Based on this representation, human demonstrations were segmented and used to train decision classifiers. Using these classifiers, our planner produced a list of waypoints in task space. These waypoints provided a high-level plan, which could be transferred to an arbitrary robot model and used to initialise a local trajectory optimiser. We evaluated this approach through testing on unseen human VR data, a physics-based robot simulation, and a real robot (dataset and code are publicly available). We found that the human-like planner outperformed a state-of-the-art standard trajectory optimisation algorithm, and was able to generate effective strategies for rapid planning

- Where to Relocate?: Object Rearrangement Inside Cluttered and Confined Environments for Robotic Manipulation

    Author: Cheong, Sang Hun | Korea University, KIST
    Author: Cho, Brian Younggil | Korea Institute of Science and Technology
    Author: Lee, JinHwi | Hanyang University
    Author: Kim, ChangHwan | Korea Institute of Science and Technology
    Author: Nam, Changjoo | Korea Institute of Science and Technology
 
    keyword: Manipulation Planning; Task Planning; Motion and Path Planning

    Abstract : We present an algorithm determining where to relocate objects inside a cluttered and confined space while rearranging objects to retrieve a target object. Although methods that decide what to remove have been proposed, planning for the placement of removed objects inside a workspace has not received much attention. Rather, removed objects are often placed outside the workspace, which incurs additional laborious work (e.g., motion planning and execution of the manipulator and the mobile base, perception of other areas). Some other methods manipulate objects only inside the workspace but without a principle so the rearrangement becomes inefficient. <p>In this work, we consider both monotone (each object is moved only once) and non-monotone arrangement problems which have shown to be NP-hard. Once the sequence of objects to be relocated is given by any existing algorithm, our method aims to minimize the number of pick-and-place actions to place the objects until the target becomes accessible. From extensive experiments, we show that our method reduces the number of pick-and-place actions and the total execution time (the reduction is up to 23.1% and 28.1% respectively) compared to baseline methods while achieving higher success rates.

- Autonomous Modification of Unstructured Environments with Found Material

    Author: Thangavelu, Vivekanandhan | University at Buffalo
    Author: Saboia Da Silva, Maira | University at Buffalo
    Author: Choi, Jiwon | University at Buffalo
    Author: Napp, Nils | SUNY Buffalo
 
    keyword: Autonomous Agents; Robotics in Construction; Reactive and Sensor-Based Planning

    Abstract : The ability to autonomously modify their environment dramatically increases the capability of robots to operate in unstructured environments. We develop a specialized construction algorithm and robotic system that can autonomously build motion support structures with previously unseen objects. The approach is based on our prior work on adaptive ramp building algorithms, but it eliminates the assumption of having specialized building materials that simplify manipulation and planning for stability. Utilizing irregularly shaped stones makes the problem significantly more challenging since the outcome of individual placements is sensitive to details of contact geometry and friction, which are difficult to observe. To reuse the same high-level algorithm, we develop a new physics-based planner that explicitly considers the uncertainty produced by incomplete in-situ sensing and imprecision during pickup and placement. We demonstrate the approach on a robotic system that uses a newly developed gripper to reliably pick up stones with minimal additional sensors or complex grasp planning. The resulting system can build structures with more than 70 stones, which in turn provide traversable paths to previously inaccessible locations.

- Tethered Tool Manipulation Planning with Cable Maneuvering

    Author: S�nchez, Daniel Enrique | Osaka University
    Author: Wan, Weiwei | Osaka University
    Author: Harada, Kensuke | Osaka University
 
    keyword: Manipulation Planning; Grippers and Other End-Effectors; Grasping

    Abstract : In this paper, we present a planner for manipulating tethered tools using dual-armed robots. The planner generates robot motion sequences to maneuver a tool and its cable while avoiding robot-cable entanglements. Firstly, the planner generates an Object Manipulation Motion Sequence (OMMS) to handle the tool and place it in desired poses. Secondly, the planner examines the tool movement associated with the OMMS and computes candidate positions for a cable slider, to maneuver the tool cable and avoid collisions. Finally, the planner determines the optimal slider positions to avoid entanglements and generates a Cable Manipulation Motion Sequence (CMMS) to place the slider in these positions. The robot executes both the OMMS and CMMS to handle the tool and its cable to avoid entanglements and excess cable bending. Simulations and real-world experiments help validate the proposed method.

- Optimization-Based Posture Generation for Whole-Body Contact Motion by Contact Point Search on the Body Surface

    Author: Murooka, Masaki | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Manipulation Planning; Kinematics; Humanoid Robots

    Abstract : Whole-body contact is an effective strategy for improving the stability and efficiency of the motion of robots. For robots to automatically perform such motions, we propose a posture generation method that employs all available surfaces of the robot links. By representing the contact point on the body surface by two-dimensional configuration variables, the joint positions and contact points are simultaneously determined through a gradient-based optimization. By generating motions with the proposed method, we present experiments in which robots manipulate objects effectively utilizing whole-body contact.

- Real-Time Conflict Resolution of Task-Constrained Manipulator Motion in Unforeseen Dynamic Environments (I)

    Author: Mao, Huitan | University of North Carolina at Charlotte
    Author: Xiao, Jing | Worcester Polytechnic Institute (WPI)
 
    keyword: Manipulation Planning; Natural Machine Motion; Task Planning

    Abstract : This paper introduces conflict resolution in task-constrained real-time adaptive motion planning (RAMP) to enable a robot manipulator performing tasks in an environment with dynamically unknown obstacles.The method continuously improves and maintains diverse task constrained as well as unconstrained robot trajectories to allow the manipulator switching to a better trajectory at any time and seamlessly resolving conflicts between satisfying task constraints and avoiding dynamically unknown obstacles. If dynamic obstacles block all available task-constrained trajectories, the algorithm allows the manipulator to change goals on the fly to be free of task constraints and resume the task whenever there is a collision-free,task-constrained trajectory. The method is validated in different dynamic environments with different task constraints in both simulation and real-world experiments.

## Contact Modeling
- Interaction Stability Analysis from the Input-Output Viewpoints
 
    Author: Huang, Yuancan | Beijing Institute of Technology
    Author: Huang, Qiang | Beijing Institute of Technology
 
    keyword: Contact Modeling; Physical Human-Robot Interaction; Dynamics

    Abstract : Interaction with the environment is arguably one of the necessary actions for many robot applications. Taxonomy of interaction behaviours is classified into three categories: cooperation, collaboration, and competition. In theory, interaction dynamics may be modelled by D'Alembert's principle or nonsmooth mechanics through seeking equality and/or inequality kinematic constraints. However, it is hard to gain these kinematic constraints in practice since they may be variable or be hardly described in a mathematical form. <p>In this paper, passivity and passivity indices with the differential operator are put forward by restricting its domain from the whole extended Hilbert function space to a set of all continuous function with finite derivative, and then the input-output stability condition, in this case, is derived. Next, mechanical impedance and admittance are defined, and a linear spatial impedance representation is given from the energetic point of view. Base on the bond graph theory, an ideal model is presented to model the idealized interaction, and invariance of port functions derived from the ideal interaction model is introduced; An interaction model is then proposed accounting for nonidealized factors and to describe cooperative, collaborative, and competitive interactions in a unified way. Finally, interaction stabilities are analyzed corresponding to different interaction models, and robustness of interaction stability is addressed based on the passivity indices.

- Improving the Contact Instant Detection of Sensing Antennae Using a Super-Twisting Algorithm

    Author: Feliu, Daniel | Robotics, Vision and Control Group at the University of Seville
    Author: Cortez-Vega, Ricardo | CINVESTAV-IPN
    Author: Feliu, Vicente | Escuela T�cnica Superior De IngenierosIndustriales/Universidad D
 
    keyword: Contact Modeling

    Abstract : Sensing antenna devices, that mimic insect antennae or mammal whiskers, is an active field of research that still needs new developments in order to become efficient and reliable components of robotic systems. This work reports a new result in the area of signal processing of these devices that allows to detect the instant of the impact of a flexible antenna with an object faster than other reported methods. Previous methods require the use of filters that introduce delays in the impact detection. A method based on the Super- Twisting algorithm is proposed here that avoids the use of these filters and reduces such delays improving the impact instant estimation. Experiments show that these delays can be reduced in more than 50%, allowing reliable estimation of the impact instant with an error of less than 5 ms in many cases requiring a limited computational effort.

- 6DFC: Efficiently Planning Soft Non-Planar Area Contact Grasps Using 6D Friction Cones

    Author: Xu, Jingyi | Technical University of Munich
    Author: Danielczuk, Michael | UC Berkeley
    Author: Steinbach, Eckehard | Technical University of Munich
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Contact Modeling; Grasping; Manipulation Planning

    Abstract : Analytic grasp planning algorithms typically approximate compliant contacts with soft point contact models to compute grasp quality, but these models are overly conservative and do not capture the full range of grasps available. While area contact models can reduce the number of false negatives predicted by point contact models, they have been restricted to a 3D analysis of the wrench applied at the contact and so are still overly conservative. We extend traditional 3D friction cones and present an efficient algorithm for calculating the 6D friction cone (6DFC) for a non-planar area contact between a compliant gripper and a rigid object. We introduce a novel sampling algorithm to find the 6D friction limit surface for a non-planar area contact and a linearization method for these ellipsoids that reduces the computation of 6DFC constraints to a quadratic program. We show that constraining the wrench applied at the contact in this way increases recall, a metric inversely related to the number of false negative predictions, by 17% and precision, a metric inversely related to the number of false positive predictions, by 2% over soft point contact models on results from 1500 physical grasps on 12 3D printed non-planar objects with an ABB YuMi robot. The 6DFC algorithm also achieves 6% higher recall with similar precision and 85x faster runtime than a previously proposed area contact model.

- Long-Horizon Prediction and Uncertainty Propagation with Residual Point Contact Learners
 
    Author: Fazeli, Nima | Massachusetts Institute of Technology
    Author: Ajay, Anurag | MIT
    Author: Rodriguez, Alberto | Massachusetts Institute of Technology
 
    keyword: Contact Modeling; Simulation and Animation; Performance Evaluation and Benchmarking

    Abstract : The ability to simulate and predict the outcome of contacts is paramount to the successful execution of many robotic tasks. Simulators are powerful tools for the design of robots and their behaviors, yet the discrepancy between their predictions and observed data limit their usability. In this paper, we propose a self-supervised approach to learning residual models for rigid-body simulators that exploits corrections of contact models to refine predictive performance and propagate uncertainty. We empirically evaluate the framework by predicting the outcomes of planar dice rolls and compare it's performance to state-of-the-art techniques.

- Versatile Trajectory Optimization Using a LCP Wheel Model for Dynamic Vehicle Maneuvers

    Author: Bellegarda, Guillaume | University of California, Santa Barbara
    Author: Byl, Katie | UCSB
 
    keyword: Contact Modeling; Optimization and Optimal Control; Wheeled Robots

    Abstract : Car models have been extensively studied at varying levels of     Abstraction, and planning and executing motions under ideal conditions is well researched and understood. For more aggressive maneuvers, for example when drifting or skidding, empirical and/or discontinuous friction models have been used to explain and approximate real world contact behavior. Separately, contact dynamics have been extensively studied by the robotics community, often times formulated as a linear complementarity problem (LCP) for dynamic multi-rigid-body contact problems with Coulomb friction cone approximations. In this work, we explore the validity of using such an anisotropic Coulomb friction cone to model tire dynamics to plan for vehicle motion, and present a versatile trajectory optimization framework using this model that can both avoid and/or exploit wheel skidding, depending on the cost function and planning horizon. Experimental evidence of planning and executing dynamic drift parking is shown on a 1/16 scale model car.

- A Transition-Aware Method for the Simulation of Compliant Contact with Regularized Friction

    Author: Castro, Alejandro | Toyota Research Institute
    Author: Qu, Ante | Stanford University, Toyota Research Institute
    Author: Kuppuswamy, Naveen | Toyota Research Institute
    Author: Alspach, Alex | Toyota Research Institute
    Author: Sherman, Michael | Toyota Research Institute
 
    keyword: Contact Modeling; Simulation and Animation; Grasping

    Abstract : Multibody simulation with frictional contact has been a challenging subject of research for the past thirty years. Rigid-body assumptions are commonly used to approximate the physics of contact, and together with Coulomb friction, lead to challenging-to-solve nonlinear complementarity problems (NCP). On the other hand, robot grippers often introduce significant compliance. Compliant contact, combined with regularized friction, can be modeled entirely with ODEs, avoiding NCP solves. Unfortunately, regularized friction introduces high-frequency stiff dynamics and even implicit methods struggle with these systems, especially during slip-stick transitions. To improve the performance of implicit integration for these systems we introduce a Transition-Aware Line Search (TALS), which greatly improves the convergence of the Newton-Raphson iterations performed by implicit integrators. We find that TALS works best with semi-implicit integration, but that the explicit treatment of normal compliance can be problematic. To address this, we develop a Transition-Aware Modified Semi-Implicit (TAMSI) integrator that has similar computational cost to semi-implicit methods but implicitly couples compliant contact forces, leading to a more robust method. We evaluate the robustness, accuracy and performance of TAMSI and demonstrate our approach alongside relevant sim-to-real manipulation tasks.


## Robotics in Hazardous Fields
- Single Actuator Peristaltic Robot for Subsurface Exploration and Device Emplacement

    Author: De la Fuente, Juan | University of Calgary
    Author: Shor, Roman | University of Calgary
    Author: Larter, Steve | University of Calgary
 
    keyword: Robotics in Hazardous Fields; Mechanism Design; Mining Robotics

    Abstract : In this work, we present the concept, design, and initial testing of a single actuator peristaltic motion robot for subsurface geological exploration and device emplacement. We are researching unconventional methods, including robotics, for the production of energy from oil reservoirs that do not liberate carbon to the atmosphere. For such application, we are developing autonomous robots for data acquisition and tool transportation inside petroleum reservoirs. The mechanism described in this work is a cam-follower configuration worm robot that utilizes peristaltic displacement. We confirmed that the mechanism works on a plane surface and in non-consolidated media.

- Improving Visual Feature Extraction in Glacial Environments

    Author: Morad, Steven | University of Arizona
    Author: Nash, Jeremy | Jet Propulsion Laboratory
    Author: Higa, Shoya | Jet Propulsion Laboratory
    Author: Smith, Russell G | Jet Propulsion Laboratory
    Author: Parness, Aaron | Nasa Jet Propulsion Laboratory
    Author: Barnard, Kobus | University of Arizona
 
    keyword: Visual-Based Navigation; Robotics in Hazardous Fields; Computer Vision for Other Robotic Applications

    Abstract : Glacial science could benefit tremendously from autonomous robots, but previous glacial robots have had perception issues in these colorless and featureless environments, specifically with visual feature extraction. Glaciologists use near-infrared imagery to reveal the underlying heterogeneous spatial structure of snow and ice, and we theorize that this hidden near-infrared structure could produce more and higher quality features than available in visible light. We took a custom camera rig to Igloo Cave at Mt. St. Helens to test our theory. The camera rig contains two identical machine vision cameras, one which was outfitted with multiple filters to see only near-infrared light. We extracted features from short video clips taken inside Igloo Cave at Mt. St. Helens, using three popular feature extractors (FAST, SIFT, and SURF). We quantified the number of features and their quality for visual navigation using feature correspondence and the epipolar constraint. Our results indicate that near-infrared imagery produces more features that tend to be of higher quality than that of visible light imagery.

- Unmanned Aerial Vehicle Based Hazardous Materials Response: Information-Theoretic Hazardous Source Search and Reconstruction (I)

    Author: Hutchinson, Michael | Loughborough University
    Author: Liu, Cunjia | Loughborough University
    Author: Thomas, Paul | Dstl
    Author: Chen, Wen-Hua | Loughborough University
 
    keyword: Robotics in Hazardous Fields; Reactive and Sensor-Based Planning; Environment Monitoring and Management

    Abstract : This article presents an airborne autonomous system to assist first responders in response to releases of hazardous material into the atmosphere. The system comprises of an Unmanned Aerial Vehicle (UAV), onboard chemical sensors and information-theoretic search and source term reconstruction algorithms. The methods presented have been validated in experiments by the     Authors and then demonstrated in field trials designed by end users and conducted at the UK's Fire Service College.

- Planning Maximum-Manipulability Cutting Paths

    Author: Pardi, Tommaso | University of Birmingham
    Author: Ortenzi, Valerio | University of Birmingham
    Author: Fairbairn, Colin | National Nuclear Laboratory
    Author: Pipe, Tony | University of the West of England
    Author: Ghalamzan Esfahani, Amir Masoud | University of Lincoln
    Author: Stolkin, Rustam | University of Birmingham
 
    keyword: Robotics in Hazardous Fields; Kinematics; Motion and Path Planning

    Abstract : This paper presents a method for constrained motion planning from vision, which enables a robot to move its end-effector over an observed surface, given start and destination points. The robot has no prior knowledge of the surface shape, but observes it from a noisy point cloud. We consider the multi-objective optimisation problem of finding robot trajectories which maximise the robot's manipulability throughout the motion, while also minimising surface-distance travelled between the two points. This work has application in industrial problems of rough robotic cutting, e.g., demolition of legacy nuclear plant, where the cut path needs not be precise as long as it achieves dismantling. We show how detours in the path can be leveraged to increase the manipulability of the robot at all points along the path. This helps to avoid singularities, while maximising the robot's capability to make small deviations during task execution. We show how a sampling-based planner can be projected onto the Riemannian manifold of a curved surface, and extended to include a term which maximises manipulability. We present the results of empirical experiments, with both simulated and real robots, which are tasked with moving over a variety of different surface shapes. Our planner enables successful task completion, while ensuring significantly greater manipulability when compared against a conventional RRT* planner.

- Robot Risk-Awareness by Formal Risk Reasoning and Planning

    Author: Xiao, Xuesu | The University of Texas at Austin
    Author: Dufek, Jan | Texas A&amp;M University
    Author: Murphy, Robin | Texas A&amp;M
 
    keyword: Robotics in Hazardous Fields; Robot Safety; Search and Rescue Robots

    Abstract : This paper proposes a formal robot motion risk reasoning framework and develops a risk-aware path planner that minimizes the proposed risk. While robots locomoting in unstructured or confined environments face a variety of risk, existing risk only focuses on collision with obstacles. Such risk is currently only addressed in ad hoc manners. Without a formal definition, ill-supported properties, e.g. additive or Markovian, are simply assumed. Relied on an incomplete and inaccurate representation of risk, risk-aware planners use ad hoc risk functions or chance constraints to minimize risk. The former inevitably has low fidelity when modeling risk, while the latter conservatively generates feasible path within a probability bound. Using propositional logic and probability theory, the proposed motion risk reasoning framework is formal. Building upon a universe of risk elements of interest, three major risk categories, i.e. locale-, action-, and traverse-dependent, are introduced. A risk-aware planner is also developed to plan minimum risk path based on the newly proposed risk framework. Results of the risk reasoning and planning are validated in physical experiments in a real-world unstructured or confined environment. With the proposed fundamental risk reasoning framework, safety of robot locomotion could be explicitly reasoned, quantified, and compared. The risk-aware planner finds safe path in t

- Experimental Evaluation and Characterization of Radioactive Source Effects on Robot Visual Localization and Mapping

    Author: Lee, Elijah S. | University of Pennsylvania
    Author: Loianno, Giuseppe | New York University
    Author: Thakur, Dinesh | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania
 
    keyword: Robotics in Hazardous Fields; Aerial Systems: Applications

    Abstract : Robots are ideally suited to performing simple tasks in dangerous environments. In this letter, we address the use of robots for inspection of nuclear reactors which may be contaminated by radiation. The geometry of a reactor vessel is three-dimensional with significant clutter. Accordingly, we propose the use of small-scale, flying robots that are able to localize themselves and autonomously navigate around obstacles. Because of the constraints on size, we rely on cameras which are the best low power and lightweight sensors. However, cameras perform poorly in the presence of radioactivity and the impact of radiation on robotics systems is not well understood. In this letter, we (a) analyze the effects of radioactive sources on camera sensors, affecting localization and mapping algorithms, (b) quantify these effects from a statistical viewpoint according to different source intensities; and (c) compare different solutions to mitigate these effects. Our analysis is supported and validated by experimental data collected on a Commercial-Off-The-Shelf (COTS) camera sensor exposed to radioactive sources in a hot cell.

## Dynamics
- Dynamic Modeling of Robotic Manipulators for Accuracy Evaluation

    Author: Zimmermann, Stefanie Antonia | ABB
    Author: Berninger, Tobias Franz Christian | TU Munich
    Author: Derkx, Jeroen | Jeroen.derkx@se.abb.com
    Author: Rixen, Daniel | Technische Universitét M�nchen
 
    keyword: Dynamics; Flexible Robots; Industrial Robots

    Abstract : In order to fulfill conflicting requirements in the development of industrial robots, such as increased accuracy of a weightreduced manipulator with lower mechanical stiffness, the robot's dynamical behavior must be evaluated early in the development process. This leads to the need of accurate multibody models of the manipulator under development. This paper deals with multibody models that include flexible bodies, which are exported from the corresponding Finite Element model of the structural parts. It is shown that such a flexible link manipulator model, which is purely based on development and datasheet data, is suitable for an accurate description of an industrial robot's dynamic behavior. No stiffness parameters need to be identified by experimental methods, making this approach especially relevant during the development of new manipulators. This paper presents results of experiments in time and frequency domain for analyzing the modeling approach and for validating the model performance against real robot behavior.

- A Real-Robot Dataset for Assessing Transferability of Learned Dynamics Models

    Author: Agudelo-Espa�a, Diego | Max Planck Institute for Intelligent Systems
    Author: Zadaianchuk, Andrii | Max Planck Institute for Intelligent Systems
    Author: Wenk, Philippe | ETH Zuerich
    Author: Garg, Aditya | Max Planck Institute for Intelligent Systems
    Author: Akpo, Joel | Max Planck Institute for Intelligent Systems
    Author: Grimminger, Felix | Max Planck Institute for Intelligent Systems
    Author: Viereck, Julian | Max Planck Institute for Intelligent Systems
    Author: Naveau, Maximilien | LAAS/CNRS
    Author: Righetti, Ludovic | New York University
    Author: Martius, Georg | Max Planck Institute for Intelligent Systems
    Author: Krause, Andreas | ETH Zurich
    Author: Sch�lkopf, Bernhard | Max Planck Institute for Intelligent Systems
    Author: Bauer, Stefan | MPI for Intelligent Systems
    Author: W�thrich, Manuel | Max-Planck-Institute for Intelligent Systems
 
    keyword: Dynamics; Model Learning for Control

    Abstract : In the context of model-based reinforcement learning and control, a large number of methods for learning system dynamics have been proposed in recent years. The purpose of these learned models is to synthesize new control policies. An important open question is how robust current dynamics-learning methods are to shifts in the data distribution due to changes in the control policy. We present a real-robot dataset which allows to systematically investigate this question. This dataset contains trajectories of a 3 degrees-of-freedom (DOF) robot being controlled by a diverse set of policies. For comparison, we also provide a simulated version of the dataset. Finally, we benchmark a few widely-used dynamics-learning methods using the proposed dataset. Our results show that the iid test error of a learned model is not necessarily a good indicator of its accuracy under control policies different from the one which generated the training data. This suggests that it may be important to evaluate dynamics-learning methods in terms of their transfer performance, rather than only their iid error.

- MagNet: Discovering Multi-Agent Interaction Dynamics Using Neural Network

    Author: Saha, Priyabrata | Georgia Institute of Technology
    Author: Ali, Arslan | Georgia Institute of Technology
    Author: Mudassar, Burhan | Georgia Institute of Technology
    Author: Long, Yun | GEORGIA TECH
    Author: Mukhopadhyay, Saibal | Georgia Institute of Technology
 
    keyword: Dynamics; Deep Learning in Robotics and Automation; Learning and Adaptive Systems

    Abstract : We present the MagNet, a neural network-based multi-agent interaction model to discover the governing dynamics and predict evolution of a complex multi-agent system from observations. We formulate a multi-agent system as a coupled non-linear network with a generic ordinary differential equation (ODE) based state evolution, and develop a neural network-based realization of its time-discretized model. MagNet is trained to discover the core dynamics of a multi-agent system from observations, and tuned on-line to learn agent-specific parameters of the dynamics to ensure accurate prediction even when physical or relational attributes of agents, or number of agents change. We evaluate MagNet on a point-mass system in two-dimensional space, Kuramoto phase synchronization dynamics and predator-swarm interaction dynamics demonstrating orders of magnitude improvement in prediction accuracy over traditional deep learning models.

- Modulation of Robot Orientation State Via Leg-Obstacle Contact Positions

    Author: Ramesh, Divya | University of Pennsylvania
    Author: Kathail, Anmol | University of Pennsylvania
    Author: Koditschek, Daniel | University of Pennsylvania
    Author: Qian, Feifei | University of Pennsylvania
 
    keyword: Dynamics; Biologically-Inspired Robots; Contact Modeling

    Abstract : We study a quadrupedal robot traversing a structured (i.e., periodically spaced) obstacle field driven by an open-loop quasi-static trotting walk. Despite complex, repeated collisions and slippage between robot legs and obstacles, the robot's horizontal plane body orientation trajectory can converge in the absence of body level feedback to stable steady state patterns. We classify these patterns into a series of �types' ranging from stable locked equilibria, to stable periodic oscillations, to unstable or mixed period oscillations. We observe that the stable equilibria can bifurcate to stable periodic oscillations and then to mixed period oscillations as the obstacle spacing is gradually increased. Using a 3D-reconstruction method, we experimentally characterize the robot leg-obstacle contact configurations at each step to show that the different steady patterns in robot orientation trajectories result from a self-stabilizing periodic pattern of leg-obstacle contact positions. We present a highly-simplified coupled oscillator model that predicts robot orientation pattern as a function of the leg-obstacle contact mechanism. We demonstrate that the model successfully captures the robot steady state for different obstacle spacing and robot initial conditions. We suggest in simulation that using the simplified coupled oscillator model we can create novel control strategies that allow multi-legged robots to exploit obstacle disturbances to negotiate randomly cluttered environments

- Beyond Basins of Attraction: Quantifying Robustness of Natural Dynamics (I)

    Author: Heim, Steve | Max Planck Institute for Intelligent Systems
    Author: Badri-Spröwitz, Alexander | Max Planck Institute for Intelligent Systems
 
    keyword: Dynamics; Legged Robots; Robust/Adaptive Control of Robotic Systems

    Abstract : Properly designing a system to exhibit favorable natural dynamics can greatly simplify designing or learning the control policy. However, it is still unclear what constitutes favorable natural dynamics and how to quantify its effect. Most studies of simple walking and running models have focused on the basins of attraction of passive limit cycles and the notion of self-stability. We instead emphasize the importance of stepping beyond basins of attraction. In this paper, we show an approach based on viability theory to quantify robust sets in state-action space. These sets are valid for the family of all robust control policies, which allows us to quantify the robustness inherent to the natural dynamics before designing the control policy or specifying a control objective. We illustrate our formulation using spring-mass models, simple low-dimensional models of running systems. We then show an example application by optimizing robustness of a simulated planar monoped, using a gradient-free optimization scheme. Both case studies result in a nonlinear effective stiffness providing more robustness.

- Stable Parking Control of a Robot Astronaut in a Space Station Based on Human Dynamics (I)
 
    Author: Jiang, Zhihong | Beijing Institute of Technology
    Author: Xu, Jiafeng | Beijing Institute of Technology
    Author: Li, Hui | Beijing Institute of Technology
    Author: Huang, Qiang | Beijing Institute of Technology
 
    keyword: Dynamics; Collision Avoidance; Space Robotics and Automation

    Abstract : Controlling a robot astronaut to move in the same way as a human astronaut to realize a wide range of motion in a space station is an important requirement for the robot astronauts that are meant to assist or replace human astronauts. However, a robot astronaut is a nonlinear and strongly coupled multibody dynamic system with multiple degrees of freedom, whose dynamic characteristics are complex. Therefore, implementing a robot astronaut with wide-ranging motion control in a space station is a tremendous challenge for robotic technology. This article presents a wide-ranging stable motion control method for robot astronauts in space stations based on human dynamics. Focusing on the astronauts' parking motion in a space station, a viscoelastic dynamic humanoid model of parking under microgravity environment was established using a mass�spring�damper system. The model was used as the expected model for stable parking control of a robot astronaut, and the complex dynamic characteristics were mapped into the robot astronaut system to control the stable parking of the robot astronaut in a manner similar to a human astronaut. This provides a critical basis for implementing robots that are capable of steady wide-ranging motion in space stations. The method was verified on a dynamic system of a robot astronaut that was constructed for this research. The experimental results showed that the method is feasible and effective and that it is a highly competitive solution for robot astronau

## Product Design, Development and Prototyping
- Development of a Robotic System for Automated Decaking of 3D-Printed Parts

    Author: Nguyen, Huy | Nanyang Technological University
    Author: Adrian, Nicholas | Nanyang Technological University
    Author: Lim, Joyce Xin-Yan | Nanyang Technological University
    Author: Salfity, Jonathan | HP Labs, HP Inc
    Author: Allen, William | HP Inc
    Author: Pham, Quang-Cuong | NTU Singapore
 
    keyword: Product Design, Development and Prototyping; Additive Manufacturing; Factory Automation

    Abstract : With the rapid rise of 3D-printing as a competitive mass manufacturing method, manual "decaking" - i.e. removing the residual powder that sticks to a 3D-printed part - has become a significant bottleneck. Here, we introduce, for the first time to our knowledge, a robotic system for automated decaking of 3D-printed parts. Combining Deep Learning for 3D perception, smart mechanical design, motion planning, and force control for industrial robots, we developed a system that can automatically decake parts in a fast and efficient way. Through a series of decaking experiments performed on parts printed by a Multi Jet Fusion printer, we demonstrated the feasibility of robotic decaking for 3D-printing-based mass manufacturing.

- A Novel Solar Tracker Driven by Waves: From Idea to Implementation

    Author: Xu, Ruoyu | The Chinese University of Hong Kong, Shenzhen
    Author: Liu, Hengli | The Chinese University of Hong Kong, Shenzhen
    Author: Liu, Chongfeng | The Chinese University of Hong Kong, Shenzhen
    Author: Sun, Zhenglong | Chinese University of Hong Kong, Shenzhen
    Author: Lam, Tin Lun | The Chinese University of Hong Kong, Shenzhen
    Author: Qian, Huihuan | The Chinese University of Hong Kong, Shenzhen
 
    keyword: Dynamics; Product Design, Development and Prototyping; Marine Robotics

    Abstract : Traditional solar trackers often adopt motors to automatically adjust the attitude of the solar panels towards the sun for maximum power efficiency. In this paper, a novel design of solar tracker for the ocean environment is introduced. Utilizing the fluctuations due to the waves, electromagnetic brakes are utilized instead of motors to adjust the attitude of the solar panels. Compared with the traditional solar trackers, the proposed one is simpler in hardware while the harvesting efficiency is similar. The desired attitude is calculated out of the local location and time. Then based on the dynamic model of the system, the angular acceleration of the solar panels is estimated and a control algorithm is proposed to decide the release and lock states of the brakes. In such a manner, the adjustment of the attitude of the solar panels can be achieved by using two brakes only. Experiments are conducted to validate the acceleration estimator and the dynamic model. At last, the feasibility of the proposed solar tracker is tested on the real water surface. The results show that the system is able to adjust 40^circ in two dimensions within 28 seconds.

- Design and Implementation of Hydraulic-Cable Driven Manipulator for Disaster Response Operation

    Author: Kim, JungYeong | University of Science and Technology(UST), Korea Institute of In
    Author: Seo, Jaehong | University of Science and Technology
    Author: Park, Sangshin | Korea Institute of Industrial Technology
    Author: Cho, Jungsan | KITECH(Korea Institute of Industrial Technology)
    Author: Han, SangChul | Korea Institute of Industrial Technology
 
    keyword: Product Design, Development and Prototyping; Hydraulic/Pneumatic Actuators; Tendon/Wire Mechanism

    Abstract : This paper introduces a new hydraulic manipulator with hydraulic-cable driven actuation (HCA) modules for disaster response mobile-manipulation. The hydraulic actuation system has the potential to apply disaster-response application, because it has a higher power-to-weight ratio and robustness to external impacts than electric motor actuation. However, using a conventional hydraulic manipulators is inappropriate because the revolute joint uses conventional actuators, such as linear cylinders and vanes, which have some limitations: 1) linear cylinder: small range of motion, 2) vane: low torque-to-weight ratio. To overcome these limitations, we propose new 3DOF manipulator which has a larger workspace than the conventional hydraulic manipulator and comparable payload-to-weight ratio. To this end, we use hydraulic-cable driven actuation modules from our previous research. Experimental results verify the basic performance of the actuator modules and manipulator and their capability to perform various disaster response tasks.

- Designs for an Expressive Mechatronic Chordophone

    Author: Yepez Placencia, Juan Pablo | Victoria University of Wellington
    Author: Carnegie, Dale Anthony | Victoria University of Wellington
    Author: Murphy, James Wassell | Victoria University of Wellington
 
    keyword: Product Design, Development and Prototyping; Mechanism Design; Entertainment Robotics

    Abstract : Plucked strings are an exciting sound generation model for technical and timbral exploration. Mechatronic chordophones take advantage of this model and have been the focus of extensive research and exploration in musical robotics, often used as stand-alone instruments or as part of sound art installations. However, no existing chordophone designs have utilised the expressive potential of plucked strings to their full extent.<p>In this paper, we introduce an expressive mechatronic monochord that serves as a prototyping platform for the construction of a polystring chordophone. This new chordophone has been developed to offer enhanced dynamic range, fast picking speeds, fast pitch shifter displacement, and additional expressive techniques compared to existing systems.

- Multi Directional Piezoelectric Plate Energy Harvesters Designed by Topology Optimization Algorithm

    Author: Homayouni-Amlashi, Abbas | FEMTO-ST Institute, Université Bourgogne Franche
    Author: Mohand Ousaid, Abdenbi | University of Franche-Comte
    Author: Rakotondrabe, Micky | Laboratoire G�nie De Production (LGP)
 
    keyword: Product Design, Development and Prototyping; Energy and Environment-Aware Automation; Optimization and Optimal Control

    Abstract : In this paper, piezoelectric plate energy harvesters are designed by using topology optimization algorithm to harvest the excitation from different directions. The goal is to minimize the volume and weight of the whole structure so the harvesters can be used in small scale applications. To this aim, the profile of polarization is optimized by the topology optimization to overcome charge cancellation which is the main challenge in random direction excitation. Two optimized designs with uniform and non-uniform polarization profiles are obtained. Separated electrodes in the surfaces of the optimized design with non-uniform polarization are used to simulate the polarization profile. Numerical simulations by COMSOL multi-physics software show that the optimized design with separated electrodes can provide 3 times higher voltage and power than those obtained with non-optimized piezoelectric plate. Experimental investigation demonstrated that the same design with separated electrodes can have 2.17 and 1.93 times higher voltage than the full plate for out of plane and in-plane forces respectively.

- OmBURo: A Novel Unicycle Robot with Active Omnidirectional Wheel

    Author: Shen, Junjie | UCLA
    Author: Hong, Dennis | UCLA
 
    keyword: Wheeled Robots; Underactuated Robots; Mechanism Design

    Abstract : A mobility mechanism for robots to be used in tight spaces shared with people requires it to have a small footprint, to move omnidirectionally, as well as to be highly maneuverable. However, currently there exist few such mobility mechanisms that satisfy all these conditions well. Here we introduce Omnidirectional Balancing Unicycle Robot (OmBURo), a novel unicycle robot with active omnidirectional wheel. The effect is that the unicycle robot can drive in both longitudinal and lateral directions simultaneously. Thus, it can dynamically balance itself based on the principle of dual-axis wheeled inverted pendulum. This paper discloses the early development of this novel unicycle robot involving the overall design, modeling, and control, as well as presents some preliminary results including station keeping and path following. With its very compact structure and agile mobility, it might be the ideal locomotion mechanism for robots to be used in human environments in the future.

## Cellular and Modular Robots
- Self-Reconfiguration in Response to Faults in Modular Aerial Systems

    Author: Gandhi, Neeraj | University of Pennsylvania
    Author: Salda�a, David | Lehigh University
    Author: Kumar, Vijay | University of Pennsylvania
    Author: Phan, Linh Thi Xuan | University of Pennsylvania
 
    keyword: Cellular and Modular Robots; Aerial Systems: Applications; Failure Detection and Recovery

    Abstract : We present a self-reconfiguration technique bywhich a modular flying platform can mitigate the impact of rotor failures. In this technique, the system adapts its configuration in response to rotor failures to be able to continue its mission while efficiently utilizing resources. A mixed integer linear program determines an optimal module-to-position allocation in the structure based on rotor faults and desired trajectories. We further propose an efficient dynamic programming algorithm that minimizes the number of disassembly and reassembly steps needed for reconfiguration. Evaluation results show that our technique can substantially increase the robustness of the system while utilizing resources efficiently, and that it can scale well with the number of modules.

- Recognition and Reconfiguration of Lattice-Based Cellular Structures by Simple Robots

    Author: Niehs, Eike | Technische Universitét Braunschweig
    Author: Schmidt, Arne | TU Braunschweig
    Author: Scheffer, Christian | Technische Universitét Braunschweig
    Author: Biediger, Dan | University of Houston
    Author: Yanuzzi, Mike | University of Houston
    Author: Jenett, Benjamin | Massachusetts Institute of Technology
    Author: Abdel-Rahman, Amira | MIT
    Author: Cheung, Kenneth C. | National Aeronautics and Space Administration (NASA)
    Author: Becker, Aaron | University of Houston
    Author: Fekete, S�ndor | Technische Universitét Braunschweig
 
    keyword: Cellular and Modular Robots; Distributed Robot Systems; Swarms

    Abstract : We consider recognition and reconfiguration of lattice-based cellular structures by very simple robots with only basic functionality. The underlying motivation is the construction and modification of space facilities of enormous dimensions, where the combination of new materials with extremely simple robots promises structures of previously unthinkable size and flexibility; this is also closely related to the newly emerging field of programmable matter. Aiming for large-scale scalability, both in terms of the number of the cellular components of a structure, as well as the number of robots that are being deployed for construction requires simple yet robust robots and mechanisms, while also dealing with various basic constraints, such as connectivity of a structure during reconfiguration. To this end, we propose an approach that combines ultra-light, cellular building materials with extremely simple robots. We develop basic algorithmic methods that are able to detect and reconfigure arbitrary cellular structures, based on robots that have only constant-sized memory. As a proof of concept, we demonstrate the feasibility of this approach for specific cellular materials and robots that have been developed at NASA.

- A Fast Configuration Space Algorithm for Variable Topology Truss Modular Robots

    Author: Liu, Chao | University of Pennsylvania
    Author: Yu, Sencheng | University of Pennsylvania
    Author: Yim, Mark | University of Pennsylvania
 
    keyword: Cellular and Modular Robots; Motion and Path Planning; Collision Avoidance

    Abstract : The Variable Topology Truss (VTT) is a new class of self-reconfigurable robot that can reconfigure its truss shape and topology depending on the task or environment requirements. Motion planning and avoiding self-collision are difficult as these systems usually have dozens of degrees-of-freedom with complex intersecting parallel actuation. There are two different types of shape changing actions for a VTT: geometry reconfiguration and topology reconfiguration. This paper focuses on the geometry reconfiguration actions. A new cell decomposition approach is presented based on a fast and complete method to compute the collision-free space of a node in a truss. A simple shape-morphing method is shown to quickly create motion paths for reconfiguration by moving one node at a time.

- ModQuad-DoF: A Novel Yaw Actuation for Modular Quadrotors

    Author: Teles Gabrich, Bruno | University of Pennsylvania
    Author: Li, Guanrui | New York University
    Author: Yim, Mark | University of Pennsylvania
 
    keyword: Cellular and Modular Robots; Aerial Systems: Mechanics and Control; Multi-Robot Systems

    Abstract : In this work we introduce ModQuad-DoF, a modular flying robotic structure with enhanced capabilities for yaw actuation. We propose a new module design that allows a one degree of freedom relative motion between the flying robot and the cage, with a docking mechanism allowing rigid connections between cages. A novel method of yaw actuation that increases the structure control     Authority is also presented. Our new method for the structure yaw control relies on the independent roll angles of each one of the modules, instead of the traditional drag moments from the propellers. In this paper, we propose a controller that allows the ModQuad-DoF to control its position and attitude. In our experiments, we tested a different number of modules flying in cooperation and validated the novel yaw actuation method.

- An Actuation Fault Tolerance Approach to Reconfiguration Planning of Modular Self-Folding Robots

    Author: Yao, Meibao | Jilin University
    Author: Xiao, Xueming | Changchun University of Science and Technology
    Author: Tian, Yang | School of Transportation Science and Engineering, Harbin Institu
    Author: Cui, Hutao | Harbin Institute of Technology
    Author: Paik, Jamie | Ecole Polytechnique Federale De Lausanne
 
    keyword: Cellular and Modular Robots; Failure Detection and Recovery; Redundant Robots

    Abstract : This paper presents a novel approach to fault tolerant reconfiguration of modular self-folding robots. Among various types of faults that probably occur in the modular system, we focus on the tolerance of complete actuation failure of active modules that might cause imprecise robotic motion and even reconfiguration failure. Our approach is to utilize the reconfigurability of modular self-folding robots and investigate intra-module connection to determine initial patterns that are inherently fault tolerant. We exploit the redundancy of actuation and distribute active modules in both layout-based and target-based scenarios, such that reconfiguration schemes with user-specified fault tolerant capability can be generated for an arbitrary input initial pattern or 3D configuration. Our methods are demonstrated in computer-aided simulation on the robotic platform of Mori, a modular origami robot. The simulation results validate that the proposed algorithms yield fault tolerant initial patterns and distribution schemes of active modules for several 2D and 3D configurations with Mori, while retaining generalizability for a large number of modular self-folding robots.

- Parallel Permutation for Linear Full-Resolution Reconfiguration of Heterogeneous Sliding-Only Cubic Modular Robots

    Author: Kawano, Hiroshi | NTT Corporation
 
    keyword: Cellular and Modular Robots

    Abstract : This paper presents a parallel permutation algorithm that achieves linear full-resolution reconfiguration of sliding-only cubic modular robots. We assume the use of a cubic module that can only slide across other modules' surfaces. The idea of a cubic modular robot with sliding-only motion primitive is a new concept that has advantages in simplifying the mechanisms of module hardware and space saving in its heterogeneous operations compared wtih previously studied cubic modules, such as those with sliding and convex motion primitives, or rotating motion primitives. However, because of its limited mobility, there are difficulties in managing the connectivity and scalability of the heterogeneous reconfiguration algorithm for it. To overcome these disadvantages, we introduce a parallel heterogeneous permutation method with linear operating time cost that can be incorporated into our previous full-resolution reconfiguration algorithm. We prove the correctness and completeness of the proposed algorithm. Simulation results show that the full-resolution reconfiguration algorithm that incorporates the proposed permutation algorithm reconfigures the robot structure with sliding-only cubic modules in linear operating-time cost.

## Performance Evaluation and Benchmarking
- Determining and Improving the Localization Accuracy of AprilTag Detection

    Author: Kallwies, Jan | Bundeswehr University Munich
    Author: Forkel, Bianca | Bundeswehr University Munich
    Author: Wuensche, Hans Joachim Joe | Bundeswehr University Munich
 
    keyword: Performance Evaluation and Benchmarking; Calibration and Identification; Computer Vision for Other Robotic Applications

    Abstract : Fiducial markers like AprilTags play an important role in robotics, e.g., for the calibration of cameras or the localization of robots. One of the most important properties of an algorithm for detecting such tags is its localization accuracy.<p>In this paper, we present the results of an extensive comparison of four freely available libraries capable of detecting AprilTags, namely AprilTag 3, AprilTags C++, ArUco as stand- alone libraries, and the OpenCV algorithm based on ArUco. The focus of the comparison is on localization accuracy, but the processing time is also examined. Besides working with pure tags, their extension to checkerboard corners is investigated.</p><p>In addition, we present two new post-processing techniques. Firstly, a method that can filter out very inaccurate detections resulting from partial occlusion, and secondly a new highly accurate method for edge refinement. With this we achieve a median pixel error of 0.017 px, compared to 0.17 px for standard OpenCV corner refinement.</p><p>The dataset used for the evaluation, as well as the developed post-processing techniques, are made publicly available to en- courage further comparison and improvement of the detection libraries.

- Change of Optimal Values: A Pre-Calculated Metric

    Author: Bai, Fang | University of Technology, Sydney
 
    keyword: Performance Evaluation and Benchmarking; Optimization and Optimal Control; SLAM

    Abstract : A variety of optimization problems takes the form of a minimum norm optimization. In this paper, we study the change of optimal values between two incrementally constructed least norm optimization problems, with new measurements included in the second one. We prove an exact equation to calculate the change of optimal values in the linear least norm optimization problem. With the result in this paper, the change of the optimal values can be pre-calculated as a metric to guide online decision makings, without solving the second optimization problem as long the solution and covariance of the first optimization problem are available. The result can be extended to linear least distance optimization problems, and nonlinear least distance optimization with (nonlinear) equality constraints through linearizations. This derivation in this paper provides a theoretically sound explanation to the empirical observations shown in cite{bai2018robust}. As an additional contribution, we propose another optimization problem, i.e. aligning two trajectories at given poses, to further demonstrate how to use the metric. The accuracy of the metric is validated with numerical examples, which is quite satisfactory in general (see the experiments in cite{bai2018robust} as well), unless in some extremely adverse scenarios. Last but not least, calculating the optimal value by the proposed metric is at least one magnitude faster than solving the corresponding optimization problems directly.

- A Flexible Method for Performance Evaluation of Robot Localization

    Author: Scheideman, Sean | University of Alberta
    Author: Ray, Nilanjan | University of Alberta
    Author: Zhang, Hong | University of Alberta
 
    keyword: Performance Evaluation and Benchmarking; Localization; SLAM

    Abstract : An important research issue in mobile robotics is performance assessment of robot SLAM algorithms in terms of their localization accuracy. Typically, SLAM algorithms are evaluated with the help of benchmark datasets or expensive equipment such as motion capture. Benchmark datasets however, are environment-specific, and use of motion capture constrains spatial coverage and affordability. In this paper, we present a novel method for SLAM performance evaluation, which only uses distinctive markers (such as AR tags), randomly placed in the robot navigation environment at arbitrary locations, and observes these markers with a camera onboard of the robot. Formulated as a generative latent optimization (GLO) problem, our method uses the local robot-to-marker poses to evaluate the global robot pose estimates by a SLAM algorithm and therefore its performance. Through extensive experiments on two robots, three localization/SLAM algorithms and both LiDAR and RGB-D sensors, we demonstrate the feasibility and accuracy of our proposed method.

- Quantifying Good Seamanship for Autonomous Surface Vessel Performance Evaluation

    Author: Stankiewicz, Paul | Johns Hopkins University Applied Physics Laboratory
    Author: Heistand, Michael | Johns Hopkins University Applied Physics Laboratory
    Author: Kobilarov, Marin | Johns Hopkins University
 
    keyword: Performance Evaluation and Benchmarking; Marine Robotics

    Abstract : The current state-of-the-art for testing and evaluation of autonomous surface vehicle (ASV) decision-making is currently limited to one-versus-one vessel interactions by determining compliance with the International Regulations for Prevention of Collisions at Sea, referred to as COLREGS. Strict measurement of COLREGS compliance, however, loses value in multi-vessel encounters, as there can be conflicting rules which make determining compliance extremely subjective. This work proposes several performance metrics to evaluate ASV decision-making based on the concept of "good seamanship," a practice which generalizes to multi-vessel encounters. Methodology for quantifying good seamanship is presented based on the criteria of reducing the overall collision risk of the situation and taking early, appropriate actions. Case study simulation results are presented to showcase the seamanship performance criteria against different ASV planning strategies.

- Action-Conditioned Benchmarking of Robotic Video Prediction Models: A Comparative Study

    Author: Serra Nunes, Manuel | Institute for Systems and Robotics
    Author: Dehban, Atabak | Ist-Id 509 830 072
    Author: Moreno, Plinio | IST-ID
    Author: Santos-Victor, Jos' | Instituto Superior Técnico - Lisbon
 
    keyword: Performance Evaluation and Benchmarking; Visual Learning

    Abstract : A defining characteristic of intelligent systems is the ability to make action decisions based on the anticipated outcomes. Video prediction systems have been demonstrated as a solution for predicting how the future will unfold visually, and thus, many models have been proposed that are capable of predicting future frames based on a history of observed frames~(and sometimes robot actions). However, a comprehensive method for determining the fitness of different video prediction models at guiding the selection of actions is yet to be developed. <p>Current metrics assess video prediction models based on human perception of frame quality. In contrast, we argue that if these systems are to be used to guide action, necessarily, the actions the robot performs should be encoded in the predicted frames. In this paper, we are proposing a new metric to compare different video prediction models based on this argument. More specifically, we propose an action inference system and quantitatively rank different models based on how well we can infer the robot actions from the predicted frames. Our extensive experiments show that models with high perceptual scores can perform poorly in the proposed action inference tests and thus, may not be suitable options to be used in robot planning systems.

- Performance Indicators for Wheeled Robots Traversing Obstacles
 
    Author: Nowac, William | McGill University
    Author: Gonzalez, Francisco | University of a Coruna
    Author: MacMahon, Sadhbh | MDA Corporation
    Author: Kovecses, Jozsef | McGill University
 
    keyword: Wheeled Robots; Dynamics; Space Robotics and Automation

    Abstract : An important element of wheeled robot operations on uneven and unstructured terrain is the ability to overcome obstacles. In this paper we deal with a part of this obstacle negotiation problem. We particularly investigate the ability of a wheeled robot, originating from its mechanical design, to successfully negotiate an obstacle. The work reported primarily investigates how the mechanism topologies and the resulting mass and inertia distributions influence obstacle negotiation. The kinematics of the obstacle and ground contact is described using the variables that represent the degrees of freedom of the articulated mechanical system of the robot; this enables the study of the effect of the robot topology on the contact dynamics. Based on this we develop a dynamics formulation that allows us to propose performance indicators to characterize the ability of the wheeled robot to overcome obstacles. This formulation accounts for the unilateral nature	 of interaction between robot, obstacle and ground. We illustrate the work with simulation and experimental results.






## Aerial Systems: Applications

- A Morphable Aerial-Aquatic Quadrotor with Coupled Symmetric Thrust Vectoring

    Author: Tan, Yu Herng | National University of Singapore
    Author: Chen, Ben M. | Chinese University of Hong Kong
 
    keyword: Aerial Systems: Applications; Mechanism Design; Product Design, Development and Prototyping

    Abstract : Hybrid aerial-aquatic vehicles have the unique ability of travelling in both air and water and can benefit from both lower fluid resistance in air and energy efficient position holding in water. However, they have to address the differing requirements which make optimising a single design difficult. While existing examples have shown the possibility of such vehicles, they are mostly structurally identical to normal aerial vehicles with minor adjustments to work underwater. Instead of using rotational acceleration to direct a component of thrust in surge and sway, we propose a quadrotor based vehicle that tilts its rotors about the respective arm so that a larger component of thrust can be directed in the lateral plane and opposite direction without rotating the vehicle body. A small scale prototype of this design is presented here, detailing the design considerations including mechanical actuation, static stability and waterproofing.

- An Autonomous Intercept Drone with Image-Based Visual Servo

    Author: Yang, Kun | School of Automation Science and Electrical Engineering, Beihang
    Author: Quan, Quan | Beihang University
 
    keyword: Aerial Systems: Applications; Visual Servoing

    Abstract : For most people on the ground, facing an unwanted drone buzzing around overhead, there is not a lot that we can do, especially if it is out of gun (radio wave gun or shotgun) range. A solution to this is to use intercept drones that seek out and bring down other drones. In order to make the interception autonomous, an image-based visual servo algorithm is designed with a forward-looking monocular camera. The control command, namely the angular velocity and thrust, is generated for intercept drones to implement accurate and fast interception. The proposed method is demonstrated in both hardware-in-the-loop simulation and demonstrative flight experiments.

- Real-Time Optimal Trajectory Generation and Control of a Multi-Rotor with a Suspended Load for Obstacle Avoidance

    Author: Son, Clark Youngdong | Seoul National University
    Author: Seo, Hoseong | Seoul National University
    Author: Jang, Dohyun | Seoul National University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Aerial Systems: Applications; Motion and Path Planning; Optimization and Optimal Control

    Abstract : This paper presents real-time optimization algorithms on trajectory generation and control for a multi-rotor with a suspended load. Since the load is suspended through a cable without any actuator, movement of the load must be controlled via maneuvers of the multi-rotor, which brings about difficulties in operating this system. Additionally, the highly nonlinear dynamics of the system exacerbates the difficulties. While trajectory generation and control are essential for safety, energy efficiency, and stability, the aforementioned characteristics of the system add challenges. With this in mind, the     Authors propose real-time path planning and optimal control algorithms for collision-free trajectory generation and trajectory tracking. For the dynamics, simplified dynamic models of the system are proposed by considering time delay in attitude control of the multi-rotor. For collision avoidance, the vehicle, cable, and load are considered as ellipsoids with different sizes and shapes, and collision-free constraints are expressed in an efficient and nonconservative way. The augmented Lagrangian method is applied to solve the nonlinear optimization problem with the nonlinear constraints in real-time. For control of the system, model predictive control with a sequential linear quadratic solver is used. Several simulations and experiments are conducted to validate the proposed algorithm.

- Wildfire Fighting by Unmanned Aerial System Exploiting Its Time-Varying Mass

    Author: Saikin, Diego | Czech Technical University in Prague
    Author: Baca, Tomas | Czech Technical Univerzity in Prague
    Author: Gurtner, Martin | Czech Technical University in Prague, Faculty of Electrical Engi
    Author: Saska, Martin | Czech Technical University in Prague
 
    keyword: Aerial Systems: Applications; Motion and Path Planning; Optimization and Optimal Control

    Abstract : This paper presents an approach for accurately dropping a relatively large amount of fire retardant, water or some other extinguishing agent onto a wildfire from an autonomous unmanned aerial vehicle (UAV), in close proximity to the epicenter of the fire. The proposed approach involves a risky maneuver outside of the safe flight envelope of the UAV. This maneuver exploits the expected weight reduction resulting from the release of the payload, enabling the UAV to recover without impacting the terrain. The UAV is tilted to high pitch angles, at which the thrust may be pointed almost horizontally. The vehicle can therefore achieve higher horizontal speeds than would be allowed by conventional motion planners. This high speed allows the UAV to significantly reduce the time spent close to the fire. As a result, the overall high heat exposure is reduced, and the payload can be dropped closer to the target, minimizing its dispersion. A constrained optimal control problem (OCP) is solved taking into account environmental parameters such as wind and terrain gradients, as well as various payload releasing mechanisms. The proposed approach was verified in simulations and in real experiments. Emphasis was put on the real time recalculation of the solution, which will enable future adaptation into a model predictive controller (MPC) scheme.

- On the Human Control of a Multiple Quadcopters with a Cable-Suspended Payload System

    Author: Prajapati, Pratik | Indian Institute of Technology Gandhinagar
    Author: Parekh, Sagar | Indian Institute of Technology, Gandhinagar
    Author: Vashista, Vineet | Indian Institute of Technology Gandhinagar
 
    keyword: Aerial Systems: Applications; Cooperating Robots; Human Factors and Human-in-the-Loop

    Abstract : A quadcopter is an under-actuated system with only four control inputs for six degrees of freedom, and yet the human control of a quadcopter is simple enough to be learned with some practice. In this work, we consider the problem of human control of a multiple quadcopters system to transport a cable-suspended payload. The coupled dynamics of the system, due to the inherent physical constraints, is used to develop a leader-follower architecture where the leader quadcopter is controlled directly by a human operator and the followers are controlled with the proposed Payload Attitude Controller and Cable Attitude Controller. Experiments, where a human operator flew a two quadcopters system to transport a cable-suspended payload, were conducted to study the performance of proposed controller. The results demonstrated successful implementation of human control in these systems. This work presents the possibility of enabling manual control for on-the-go maneuvering of the quadcopter-payload system which motivates aerial transportation in the unknown environments.

- Dronument: System for Reliable Deployment of Micro Aerial Vehicles in Dark Areas of Large Historical Monuments

    Author: Petráček, Pavel | Czech Technical University in Prague
    Author: Krátký, Vít | Czech Technical University in Prague
    Author: Saska, Martin | Czech Technical University in Prague
 
    keyword: Aerial Systems: Applications; Aerial Systems: Perception and Autonomy; Localization

    Abstract : This letter presents a self-contained system for robust deployment of autonomous aerial vehicles in environments without access to global navigation systems and with limited lighting conditions. The proposed system, application-tailored for documentation in dark areas of large historical monuments, uses a unique and reliable aerial platform with a multi-modal lightweight sensory setup to acquire data in human-restricted areas with adverse lighting conditions, especially in areas that are high above the ground. The introduced localization method relies on an easy-to-obtain 3-D point cloud of a historical building, while it copes with a lack of visible light by fusing active laser-based sensors. The approach does not rely on any external localization, or on a preset motion-capture system. This enables fast deployment in the interiors of investigated structures while being computationally undemanding enough to process data online, onboard an MAV equipped with ordinary processing resources. The reliability of the system is analyzed, is quantitatively evaluated on a set of aerial trajectories performed inside a real-world church, and is deployed onto the aerial platform in the position control feedback loop to demonstrate the reliability of the system in the safety-critical application of historical monuments documentation.

- Robust Real-Time UAV Replanning Using Guided Gradient-Based Optimization and Topological Paths

    Author: Zhou, Boyu | Hong Kong University of Science and Technology
    Author: Gao, Fei | Zhejiang University
    Author: Pan, Jie | Hong Kong University of Science and Technology
    Author: Shen, Shaojie | Hong Kong University of Science and Technology
 
    keyword: Aerial Systems: Applications; Motion and Path Planning; Collision Avoidance

    Abstract : Gradient-based trajectory optimization (GTO) has gained wide popularity for quadrotor trajectory replanning. However, it suffers from local minima, which is not only fatal to safety but also unfavorable for smooth navigation. In this paper, we propose a replanning method based on GTO addressing this issue systematically. A path-guided optimization (PGO) approach is devised to tackle infeasible local minima, which improves the replanning success rate significantly. A topological path searching algorithm is developed to capture a collection of distinct useful paths in 3-D environments, each of which then guides an independent trajectory optimization. It activates a more comprehensive exploration of the solution space and output superior replanned trajectories. Benchmark evaluation shows that our method outplays state-of-the-art methods regarding replanning success rate and optimality.Challenging experiments of aggressive autonomous flight are presented to demonstrate the robustness of our method. We will release our implementation as an open-source package.

- Learning-Based Path Planning for Autonomous Exploration of Subterranean Environments

    Author: Reinhart, Russell | University of Nevada Reno
    Author: Dang, Tung | University of Nevada, Reno
    Author: Hand, Emily | University of Nevada, Reno
    Author: Papachristos, Christos | University of Nevada Reno
    Author: Alexis, Kostas | University of Nevada, Reno
 
    keyword: Aerial Systems: Applications; Aerial Systems: Perception and Autonomy; Field Robots

    Abstract : In this work we present a new methodology on learning-based path planning for autonomous exploration of subterranean environments using aerial robots. Utilizing a recently proposed graph-based path planner as a "training expert" and following an approach relying on the concepts of imitation learning, we derive a trained policy capable of guiding the robot to autonomously explore underground mine drifts and tunnels. The algorithm utilizes only a short window of range data sampled from the onboard LiDAR and achieves an exploratory behavior similar to that of the training expert with a more than an order of magnitude reduction in computational cost, while simultaneously relaxing the need to maintain a consistent and online reconstructed map of the environment. The trained path planning policy is extensively evaluated both in simulation and experimentally within field tests relating to the autonomous exploration of underground mines.

- Visual-Inertial Telepresence for Aerial Manipulation

    Author: Lee, Jongseok | German Aerospace Center
    Author: Balachandran, Ribin | DLR
    Author: Sarkisov, Yuri | Skolkovo Institute of Science and Technology
    Author: De Stefano, Marco | German Aerospace Center (DLR)
    Author: Coelho, Andre | German Aerospace Center (DLR)
    Author: Shinde, Kashmira | German Aerospace Center (DLR)
    Author: Kim, Min Jun | DLR
    Author: Triebel, Rudolph | German Aerospace Center (DLR)
    Author: Kondak, Konstantin | German Aerospace Center
 
    keyword: Aerial Systems: Applications; Telerobotics and Teleoperation; Field Robots

    Abstract : This paper presents a novel vision-based telepresence system for enhancing aerial manipulation capabilities. It involves not only a haptic device, but also a virtual reality technology that provides a 3D visual feedback to a remotely-located teleoperator in real-time. We achieve this by utilizing onboard sensors, an object tracking algorithm and a pre-generated object database. As the virtual reality has to closely match the real remote scene, we propose an extension of a marker tracking algorithm with on-board Visual Inertial Odometry. Both of our indoor and outdoor experiments show benefits of our proposed system in achieving advanced aerial manipulation tasks, namely grasping, placing, pressing and peg-in-hole insertion.

- Distributed Rotor-Based Vibration Suppression for Flexible Object Transport and Manipulation

    Author: Yang, Hyunsoo | Seoul National University
    Author: Kim, Min Seong | Seoul National University
    Author: Lee, Dongjun | Seoul National University
 
    keyword: Aerial Systems: Applications; Flexible Robots; Multi-Robot Systems

    Abstract :  The RVM (Robot-based Vibration Suppression Modules) is proposed for the manipulation and transport of a large flexible object. Since the RVM is easily attachable/detachable to the object, this RVM allows distributing over the manipulated object so that it is scalable to the object size. The composition of the system is partly motivated by the MAGMaS (Multiple Aerial-Ground Manipulator System) [1]-[3], however, since the quadrotor usage is mechanically too complicated and its design is not optimized for manipulation, thus we overcome these limitations using distributed RVMs and newly developed theory. For this, we first provide a constrained optimization problem of RVM design with the minimum number of rotors, so that the feasible thrust force is maximized while it minimizes undesirable wrench and its own weight. Then, we derive the full dynamics and elucidate a controllability condition with multiple distributed RVMs and show that even if multiple, their structures turn out similar to [2] composed with a single quadrotor. We also elucidate the optimal placement of the RVM via the usage of controllability gramian which is not even alluded in [2] and established for the first time here. Experiments are performed to demonstrate the effectiveness of the proposed theory.

- Aerial Manipulation Using Model Predictive Control for Opening a Hinged Door

    Author: Lee, Dongjae | Seoul National University
    Author: Seo, Hoseong | Seoul National University
    Author: Kim, Dabin | Seoul National University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Aerial Systems: Applications; Optimization and Optimal Control; Motion and Path Planning

    Abstract : Existing studies for environment interaction with an aerial robot have been focused on interaction with static surroundings. However, to fully explore the concept of aerial manipulation, interaction with moving structures should also be considered. In this paper, a multirotor-based aerial manipulator opening a daily-life moving structure, a hinged door, is presented. In order to address the constrained motion of the structure and to avoid collisions during operation, model predictive control (MPC) is applied to the derived coupled system dynamics between the aerial manipulator and the door involving state constraints. By implementing a constrained version of differential dynamic programming (DDP), MPC can generate position setpoints to the disturbance observer (DOB)-based robust controller in real-time, which is validated by our experimental results.

- Integrated Motion Planner for Real-Time Aerial Videography with a Drone in a Dense Environment

    Author: Jeon, Boseong | Seoul National University
    Author: Lee, Yunwoo | Seoul National University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Reactive and Sensor-Based Planning; Visual Servoing

    Abstract : This work suggests an integrated approach for a drone (or multirotor) to perform an autonomous videography task in a 3-D obstacle environment by following a moving object. The proposed system includes 1) a target motion prediction module which can be applied to dense environments and 2) a hierarchical chasing planner. Leveraging covariant optimization, the prediction module estimates the future motion of the target assuming it efforts to avoid the obstacles. The other module, chasing planner, is in a bi-level structure composed of preplanner and smooth planner. In the first phase, we exploit a graph-search method to preplan a chasing corridor which incorporates safety and visibility of target. In the subsequent phase, we generate a smooth and dynamically feasible path within the corridor using quadratic programming (QP). We validate our approach with multiple complex scenarios and actual experiments.

- Stable Control in Climbing and Descending Flight under Upper Walls Using Ceiling Effect Model Based on Aerodynamics

    Author: Nishio, Takuzumi | The University of Tokyo
    Author: Zhao, Moju | The University of Tokyo
    Author: Shi, Fan | The University of Tokyo
    Author: Anzai, Tomoki | The University of Tokyo
    Author: Kawaharazuka, Kento | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Aerial Systems: Applications; Aerial Systems: Mechanics and Control; Dynamics

    Abstract : Stable flight control under ceilings is difficult for multi-rotor Unmanned Aerial Vehicles (UAVs). The wake interaction between rotors and the upper walls, called the "ceiling effect", causes an increase of rotor thrust. By the thrust increase, multi-rotors are drawn upward abruptly and collide with ceilings. In previous work, several thrust models in the ceiling effect have been proposed for stable flight under ceilings, assuming that the airflow around the rotor is in steady states. However, the airflow around rotors in vertical flight is not in steady states and each model is skillfully determined based on large amounts of experimental data. In this paper, we introduce an aerodynamics based thrust model and a stable control method under ceilings. The model is derived from the momentum theory and relationship between a vertical climbing/descending rate of a rotor and an induced velocity. To confirm the proposed model, we collect thrust data at various vertical rates in flight. Here, we use only onboard sensors to estimate self-state, for structural inspections. Consequently, we demonstrate that the proposed model is in agreement with the experimental results. Based on aerodynamics, we need not collect huge precise experimental data to construct the model. Furthermore, the vertical flight under ceilings demonstrate that the proposed unsteady-state model based controller outperforms the conventional steady-state ones.

- Motion Primitives-Based Path Planning for Fast and Agile Exploration Using Aerial Robots

    Author: Dharmadhikari, Mihir Rahul | Birla Institute of Technology and Science (BITS) - Pilani
    Author: Dang, Tung | University of Nevada, Reno
    Author: Solanka, Lukas | Flyability SA
    Author: Loje, Johannes Brakker | Flyability SA
    Author: Nguyen, Dinh Huan | University of Nevada, Reno
    Author: Khedekar, Nikhil Vijay | University of Nevada, Reno
    Author: Alexis, Kostas | University of Nevada, Reno
 
    keyword: Aerial Systems: Applications; Field Robots; Motion and Path Planning

    Abstract : This paper presents a novel path planning strategy for fast and agile exploration using aerial robots. Tailored to the combined need for large-scale exploration of challenging and confined environments, despite the limited endurance of micro aerial vehicles, the proposed planning employs motion primitives to identify admissible paths that search the configuration space, while exploiting the dynamic flight properties of small aerial robots. Utilizing a computationally efficient volumetric representation of the environment, the planner provides fast collision-free and future-safe paths that maximize the expected exploration gain and ensure continuous fast navigation through the unknown environment. The new method is field-verified in a set of deployments relating to subterranean exploration and specifically, in both modern and abandoned underground mines in Northern Nevada utilizing a 0.55m-wide collision-tolerant flying robot exploring with a speed of up to 2m/s and navigating sections with width as small as 0.8m.

- Unsupervised Anomaly Detection for Self-Flying Delivery Drones
 
    Author: Sindhwani, Vikas | Google Brain, NYC
    Author: Sidahmed, Hakim | Google
    Author: Choromanski, Krzysztof | Google Brain Robotics
    Author: Jones, Brandon | Alphabet
 
    keyword: Aerial Systems: Applications; Robot Safety; Model Learning for Control

    Abstract : We propose a novel anomaly detection framework for a fleet of hybrid aerial vehicles executing high-speed package pickup and delivery missions. The detection is based on machine learning models of normal flight profiles, trained on millions of flight log measurements of control inputs and sensor readings. We develop a new scalable algorithm for robust regression which can simultaneously fit predictive flight dynamics models while identifying and discarding abnormal flight missions from the training set. The resulting unsupervised estimator has a very high breakdown point and can withstand massive contamination of training data to uncover what normal flight patterns look like, without requiring any form of prior knowledge of aircraft aerodynamics or manual labeling of anomalies upfront. Across many different anomaly types, spanning simple 3-sigma statistical thresholds to turbulence and other equipment anomalies, our models achieve high detection rates across the board. Our method consistently outperforms alternative robust detection methods on benchmark problems. To the best of	our knowledge, dynamics modeling of hybrid delivery drones for anomaly detection at the scale of 100 million measurements from 5000 real flight missions in variable flight conditions is unprecedented.

- Keyfilter-Aware Real-Time UAV Object Tracking

    Author: Li, Yiming | Tongji University
    Author: Fu, Changhong | Tongji University
    Author: Huang, Ziyuan | National Universitu of Singapore
    Author: Zhang, Yinqiang | Technical University of Munich
    Author: Pan, Jia | University of Hong Kong
 
    keyword: Aerial Systems: Applications; Visual Tracking; Computer Vision for Other Robotic Applications

    Abstract : Correlation filter-based tracking has been widely applied in unmanned aerial vehicle (UAV) with high efficiency. However, it has two imperfections, i.e., boundary effect and filter corruption. Several methods enlarging the search area can mitigate boundary effect, yet introducing undesired background distraction. Existing frame-by-frame context learning strategies for repressing background distraction nevertheless lower the tracking speed. Inspired by keyframe-based simultaneous localization and mapping, keyfilter is proposed in visual tracking for the first time, in order to handle the above issues efficiently and effectively. Keyfilters generated by periodically selected keyframes learn the context intermittently and are used to restrain the learning of filters, so that 1) context awareness can be transmitted to all the filters via keyfilter restriction, and 2) filter corruption can be repressed. Compared to the state-of-the-art results, our tracker performs better on two challenging benchmarks, with enough speed for UAV real-time applications.

- Aerial Regrasping: Pivoting with Transformable Multilink Aerial Robot

    Author: Shi, Fan | The University of Tokyo
    Author: Zhao, Moju | The University of Tokyo
    Author: Murooka, Masaki | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Aerial Systems: Applications; Dexterous Manipulation; Underactuated Robots

    Abstract : Regrasping is one of the most common and important manipulation skills used in our daily life. However, aerial regrasping has not been seriously investigated yet, since most of the aerial manipulator lacks dexterous manipulation abilities except for the basic pick-and-place. In this paper, we focus on pivoting a long box, which is one of the most classical problems among regrasping researches, using a transformable multilink aerial robot. First, we improve our previous controller by compensating for the external wrench. Second, we optimize the joints configuration of our transformable multilink drone for stable grasping form under the constraints of thrust force and joints effort. Third, we sequentially optimize the grasping force in the pivoting process. The optimization goal is to generate continous grasping force whilst maximizing the friction force in case of the downwash, which would influence the grasped object and is difficult to model. Fourth, we develop the impedance controller in joint space and admittance controller in task space. As far as we know, it is the first research to achieve extrinsic contact-aware regrasping task on aerial robots.

- Grounding Language to Landmarks in Arbitrary Outdoor Environments

    Author: Berg, Matthew | Brown University
    Author: Bayazit, Deniz | Brown University
    Author: Mathew, Rebecca | Brown University
    Author: Rotter-Aboyoun, Ariel | Brown University
    Author: Pavlick, Ellie | Brown University
    Author: Tellex, Stefanie | Brown
 
    keyword: Aerial Systems: Applications; Task Planning

    Abstract : Robots operating in outdoor, urban environments need the ability to follow complex natural language commands which refer to never-before-seen landmarks. Existing approaches to this problem are limited because they require training a language model for the landmarks of a particular environment before a robot can understand commands referring to those landmarks. To generalize to new environments outside of the training set, we present a framework that parses references to landmarks, then assesses semantic similarities between the referring expression and landmarks in a predefined semantic map of the world, and ultimately translates natural language commands to motion plans for a drone. This framework allows the robot to ground natural language phrases to landmarks in a map when both the referring expressions to landmarks and the landmarks themselves have not been seen during training. We test our framework with a 14-person user evaluation demonstrating an end-to-end accuracy of 76.19% in an unseen environment. Subjective measures show that users find our system to have high performance and low workload. These results demonstrate our approach enables untrained users to control a robot in large unseen outdoor environments with unconstrained natural language.

## Learning and Adaptive Systems


- MANGA: Method Agnostic Neural-Policy Generalization and Adaptation

    Author: Bharadhwaj, Homanga | University of Toronto, Canada
    Author: Yamaguchi, Shoichiro | Preferred Networks, Inc
    Author: Maeda, Shin-ichi | Preferred Networks
 
    keyword: Deep Learning in Robotics and Automation; Learning and Adaptive Systems; Robust/Adaptive Control of Robotic Systems

    Abstract : In this paper, we target the problem of transferring policies across multiple environments with different dynamics parameters and motor noise variations, by introducing a framework that decouples the processes of policy learning and system identification. Efficiently transferring learned policies to an unknown environment with changes in dynamics configurations in the presence of motor noise is very important for operating robots in the real world, and our work is a novel attempt in that direction. We introduce MANGA: Method Agnostic Neural-policy Generalization and Adaptation, that trains dynamics conditioned policies and efficiently learns to estimate the dynamics parameters of the environment given off-policy state-transition rollouts in the environment. Our scheme is agnostic to the type of training method used - both reinforcement learning (RL) and imitation learning (IL) strategies can be used. We demonstrate the effectiveness of our approach by experimenting with four different MuJoCo agents and comparing against previously proposed transfer baselines.

- Fast Adaptation of Deep Reinforcement Learning-Based Navigation Skills to Human Preference

    Author: Choi, Jinyoung | NAVERLABS
    Author: Dance, Christopher | NAVER LABS Europe
    Author: Kim, Jung-eun | NAVER LABS
    Author: Park, Kyung-sik | NAVER LABS
    Author: Han, Jae-Hun | NAVER LABS
    Author: Seo, Joonho | NAVER LABS
    Author: Kim, Minsu | NAVERLABS
 
    keyword: Deep Learning in Robotics and Automation; Learning and Adaptive Systems; Service Robots

    Abstract : Deep reinforcement learning (RL) is being actively studied for robot navigation due to its promise of superior performance and robustness. However, most existing deep RL navigation agents are trained using fixed parameters, such as maximum velocities and weightings of reward components. Since the optimal choice of parameters depends on the use-case, it can be difficult to deploy such existing methods in a variety of real-world service scenarios. In this paper, we propose a novel deep RL navigation method that can adapt its policy to a wide range of parameters and reward functions without expensive retraining. Additionally, we explore a Bayesian deep learning method to optimize these parameters that requires only a small amount of preference data. We empirically show that our method can learn diverse navigation skills and quickly adapt its policy to a given performance metric or to human preference. We also demonstrate our method in real-world scenarios

- Model-Based Generalization under Parameter Uncertainty Using Path Integral Control

    Author: Abraham, Ian | Northwestern University
    Author: Handa, Ankur | IIIT Hyderabad
    Author: Ratliff, Nathan | Lula Robotics Inc
    Author: Lowrey, Kendall | University of Washington
    Author: Murphey, Todd | Northwestern University
    Author: Fox, Dieter | University of Washington
 
    keyword: Learning and Adaptive Systems; Optimization and Optimal Control; Reactive and Sensor-Based Planning

    Abstract : This work addresses the problem of robot interaction in complex environments where online control and adaptation is necessary. By expanding the sample space in the free energy formulation of path integral control, we derive a natural extension to the path integral control that embeds uncertainty into action and provides robustness for model-based robot planning. Our algorithm is applied to a diverse set of tasks using different robots and validate our results in simulation and real-world experiments. We further show that our method is capable of running in real-time without loss of performance.

- Memory of Motion for Warm-Starting Trajectory Optimization

    Author: Lembono, Teguh Santoso | Idiap Research Institute
    Author: Paolillo, Antonio | Idiap Research Institute
    Author: Pignat, Emmanuel | Idiap Research Institute, Martigny, Switzerland
    Author: Calinon, Sylvain | Idiap Research Institute
 
    keyword: Learning and Adaptive Systems; Motion and Path Planning

    Abstract : Trajectory optimization for motion planning requires good initial guesses to obtain good performance. In our proposed approach, we build a memory of motion based on a database of robot paths to provide good initial guesses. The memory of motion relies on function approximators and dimensionality reduction techniques to learn the mapping between the tasks and the robot paths. Three function approximators are compared: k-Nearest Neighbor, Gaussian Process Regression, and Bayesian Gaussian Mixture Regression. In addition, we show that the memory can be used as a metric to choose between several possible goals, and using an ensemble method to combine different function approximators results in a significantly improved warm-starting performance. We demonstrate the proposed approach with motion planning examples on the dual-arm robot PR2 and the humanoid robot Atlas.

- Safety Augmented Value Estimation from Demonstrations (SAVED): Safe Deep Model-Based RL for Sparse Cost Robotic Tasks

    Author: Thananjeyan, Brijen | UC Berkeley
    Author: Balakrishna, Ashwin | University of California, Berkeley
    Author: Rosolia, Ugo | 1990
    Author: Li, Felix | UC Berkeley
    Author: McAllister, Rowan | University of California, Berkeley
    Author: Gonzalez, Joseph E. | UC Berkeley
    Author: Levine, Sergey | UC Berkeley
    Author: Borrelli, Francesco | University of California, Berkeley
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Learning from Demonstration; Learning and Adaptive Systems; Deep Learning in Robotics and Automation

    Abstract : Reinforcement learning (RL) for robotics is challenging due to the difficulty in hand-engineering a dense cost function, which can lead to unintended behavior, and dynamical uncertainty, which makes exploration and constraint satisfaction challenging. We address these issues with a new model-based reinforcement learning algorithm, Safety Augmented Value Estimation from Demonstrations (SAVED), which uses supervision that only identifies task completion and a modest set of suboptimal demonstrations to constrain exploration and learn efficiently while handling complex constraints. We derive iterative improvement guarantees for SAVED under known stochastic nonlinear systems. We then compare SAVED with 3 state-of-the-art model-based and model-free RL algorithms on 6 standard simulation benchmarks involving navigation and manipulation and 2 real-world tasks on the da Vinci surgical robot. Results suggest that SAVED outperforms prior methods in terms of success rate, constraint satisfaction, and sample efficiency, making it feasible to safely learn complex maneuvers directly on a real robot in less than an hour. For tasks on the robot, baselines succeed less than 5% of the time while SAVED has a success rate of over 75% in the first 50 training iterations. Code and supplementary material is available at https://tinyurl.com/saved-rl.

- Variational Inference with Mixture Model Approximation for Applications in Robotics

    Author: Pignat, Emmanuel | Idiap Research Institute, Martigny, Switzerland
    Author: Lembono, Teguh Santoso | Idiap Research Institute
    Author: Calinon, Sylvain | Idiap Research Institute
 
    keyword: Learning and Adaptive Systems

    Abstract : We propose to formulate the problem of representing a distribution of robot configurations (e.g. joint angles) as that of approximating a product of experts. Our approach uses variational inference, a popular method in Bayesian computation, which has several practical advantages over sampling-based techniques. To be able to represent complex and multimodal distributions of configurations, mixture models are used as approximate distribution. We show that the problem of approximating a distribution of robot configurations while satisfying multiple objectives arises in a wide range of problems in robotics, for which the properties of the proposed approach have relevant consequences. Several applications are discussed, including learning objectives from demonstration, planning, and warm-starting inverse kinematics problems. Simulated experiments are presented with a 7-DoF Panda arm and a 28-DoF Talos humanoid.

- Preference-Based Learning for Exoskeleton Gait Optimization

    Author: Tucker, Maegan | California Institute of Technology
    Author: Novoseller, Ellen | California Institute of Technology
    Author: Kann, Claudia | California Institute of Technology
    Author: Sui, Yanan | Tsinghua University
    Author: Yue, Yisong | California Institute of Technology
    Author: Burdick, Joel | California Institute of Technology
    Author: Ames, Aaron | Caltech
 
    keyword: Learning and Adaptive Systems; Humanoid and Bipedal Locomotion; Prosthetics and Exoskeletons

    Abstract : This paper presents a personalized gait optimization framework for lower-body exoskeletons. Rather than optimizing numerical objectives such as the mechanical cost of transport, our approach directly learns from user preferences, e.g., for comfort. Building upon work in preference-based interactive learning, we present the CoSpar algorithm. CoSpar prompts the user to give pairwise preferences between trials and suggest improvements; as exoskeleton walking is a non-intuitive behavior, users can provide preferences more easily and reliably than numerical feedback. We show that CoSpar performs competitively in simulation and demonstrate a prototype implementation of CoSpar on a lower-body exoskeleton to optimize human walking trajectory features. In the experiments, CoSpar consistently found user-preferred parameters of the exoskeleton's walking gait, which suggests that it is a promising starting point for adapting and personalizing exoskeletons (or other assistive devices) to individual users.

- Adaptive Neural Trajectory Tracking Control for Flexible-Joint Robots with Online Learning

    Author: Chen, Shuyang | Rensselaer Polytechnic Institute
    Author: Wen, John | Rensselaer Polytechnic Institute
 
    keyword: Learning and Adaptive Systems; Neural and Fuzzy Control; Deep Learning in Robotics and Automation

    Abstract : Collaborative robots and space manipulators contain significant joint flexibility. It complicates the control design, compromises the control bandwidth, and limits the tracking accuracy. The imprecise knowledge of the flexible joint dynamics compounds the challenge. In this paper, we present a new control architecture for controlling flexible joint robots. Our approach uses a multi-layer neural network to approximate unknown dynamics needed for the feedforward control. The network may be viewed as a linear-in-parameter representation of the robot dynamics, with the nonlinear basis of the robot dynamics connected to the linear output layer. The output layer weights are updated based on the tracking error and the nonlinear basis. The internal weights of the nonlinear basis are updated by online backpropagation to further reduce the tracking error. To use time scale separation to reduce the coupling of the two steps - the update of the internal weights is at a lower rate compared to the update of the output layer weights. With the update of the output layer weights, our controller adapts quickly to the unknown dynamics change and disturbances. The update of the internal weights would continue to improve the converge of the nonlinear basis functions. We show the stability of the proposed scheme under the "outer loop" control. Simulation and physical experiments are conducted to demonstrate the performance of the proposed controller on a Baxter robot.

- BiCF: Learning Bidirectional Incongruity-Aware Correlation Filter for Efficient UAV Object Tracking

    Author: Lin, Fuling | Tongji University
    Author: Fu, Changhong | Tongji University
    Author: He, Yujie | Tongji University
    Author: Guo, Fuyu | Chongqing University
    Author: Tang, Qian | Chongqing University
 
    keyword: Learning and Adaptive Systems; Visual Tracking; Computer Vision for Other Robotic Applications

    Abstract : Correlation filters (CFs) have shown excellent performance in unmanned aerial vehicle (UAV) tracking scenarios due to their high computational efficiency. During the UAV tracking process, viewpoint variations are usually accompanied by changes in the object and background appearance, which poses a unique challenge to CF-based trackers. Since the appearance is gradually changing over time, an ideal tracker can not only forward predict the object position but also backtrack to locate its position in the previous frame. There exist response-based errors in the reversibility of the tracking process containing the information on the changes in appearance. However, some existing methods do not consider the forward and backward errors based on while using only the current training sample to learn the filter. For other ones, the applicants of considerable historical training samples impose a computational burden on the UAV. In this work, a novel bidirectional incongruity-aware correlation filter (BiCF) is proposed. By integrating the response-based bidirectional incongruity error into the CF, BiCF can efficiently learn the changes in appearance and suppress the inconsistent error. Extensive experiments on 243 challenging sequences from three UAV datasets (UAV123, UAVDT, and DTB70) are conducted to demonstrate that BiCF favorably outperforms other 25 stateof- the-art trackers and achieves a real-time speed of 45.4 FPS on a single CPU, which can be applied in UAV efficiently.

- Adaptive Unknown Object Rearrangement Using Low-Cost Tabletop Robot

    Author: Chai, Chun-Yu | National Chiao Tung University
    Author: Peng, Wen-Hsiao | National Chiao Tung University
    Author: Tsao, Shiao-Li | National Chiao Tung University
 
    keyword: Learning and Adaptive Systems; Motion and Path Planning; AI-Based Methods

    Abstract : Studies on object rearrangement planning typically consider known objects. Some learning-based methods can predict the movement of an unknown object after single-step interaction, but require intermediate targets, which are generated manually, to achieve the rearrangement task. In this work, we propose a framework for unknown object rearrangement. Our system &#64257;rst models an object through a small-amount of identi&#64257;cation actions and adjust the model parameters during task execution. We implement the proposed framework based on a low-cost tabletop robot (under 180 USD) to demonstrate the advantages of using a physics engine to assist action prediction. Experimental results reveal that after running our adaptive learning procedure, the robot can successfully arrange a novel object using an average of &#64257;ve discrete pushes on our tabletop environment and satisfy a precise 3.5 cm translation and 5° rotation criterion.

- Unsupervised Learning and Exploration of Reachable Outcome Space

    Author: Paolo, Giuseppe | Sorbonne University
    Author: Coninx, Alexandre | UPMC
    Author: Doncieux, Stéphane | Pierre and Marie Curie University
    Author: Laflaquière, Alban | AI Lab, SoftBank Robotics EU
 
    keyword: Learning and Adaptive Systems; Autonomous Agents; AI-Based Methods

    Abstract : Performing Reinforcement Learning in sparse rewards settings, with very little prior knowledge, is a challenging problem since there is no signal to properly guide the learning process. In such situations, a good search strategy is fundamental. At the same time, not having to adapt the algorithm to every single problem is very desirable. Here we introduce TAXONS, a Task Agnostic eXploration of Outcome spaces through Novelty and Surprise algorithm. Based on a population-based divergent-search approach, it learns a set of diverse policies directly from high-dimensional observations, without any task-specific information. TAXONS builds a repertoire of policies while training an autoencoder on the high-dimensional observation of the final state of the system to build a low-dimensional outcome space. The learned outcome space, combined with the reconstruction error, is used to drive the search for new policies. Results show that TAXONS can find a diverse set of controllers, covering a good part of the ground-truth outcome space, while having no information about such space.

- Context-Aware Cost Shaping to Reduce the Impact of Model Error in Safe, Receding Horizon Control

    Author: McKinnon, Christopher | University of Toronto
    Author: Schoellig, Angela P. | University of Toronto
 
    keyword: Learning and Adaptive Systems; Model Learning for Control; Field Robots

    Abstract : This paper presents a method to enable a robot using stochastic Model Predictive Control (MPC) to achieve high performance on a repetitive path-following task. In particular, we consider the case where the accuracy of the model for robot dynamics varies significantly over the path-motivated by the fact that the models used in MPC must be computationally efficient, which limits their expressive power. Our approach is based on correcting the cost predicted using a simple learned dynamics model over the MPC horizon. This discourages the controller from taking actions that lead to higher cost than would have been predicted using the dynamics model. In addition, stochastic MPC provides a quantitative measure of safety by limiting the probability of violating state and input constraints over the prediction horizon. Our approach is unique in that it combines both online model learning and cost learning over the prediction horizon and is geared towards operating a robot in changing conditions. We demonstrate our algorithm in simulation and experiment on a ground robot that uses a stereo camera for localization.

- Context-Aware Task Execution Using Apprenticeship Learning

    Author: Abdelrahman, Ahmed Faisal | Hochschule Bonn-Rhein-Sieg
    Author: Mitrevski, Alex | Hochschule Bonn-Rhein-Sieg
    Author: Pl�ger, Paul G. | Hochschule Bonn Rhein Sieg
 
    keyword: Learning and Adaptive Systems; Human-Centered Robotics; Domestic Robots

    Abstract : An essential measure of autonomy in assistive service robots is adaptivity to the various contexts of human-oriented tasks, which are subject to subtle variations in task parameters that determine optimal behaviour. In this work, we propose an apprenticeship learning approach to achieving context-aware action generalization on the task of robot-to- human object hand-over. The procedure combines learning from demonstration and reinforcement learning: a robot first imitates a demonstrator's execution of the task and then learns contextualized variants of the demonstrated action through experience. We use dynamic movement primitives as compact motion representations, and a model-based C-REPS algorithm for learning policies that can specify hand-over position, conditioned on context variables. Policies are learned using simulated task executions, before transferring them to the robot and evaluating emergent behaviours. We additionally conduct a user study involving participants assuming different postures and receiving an object from a robot, which executes hand-overs by either imitating a demonstrated motion, or adapting its motion to hand-over positions dictated by the learned policy. The results confirm the hypothesized improvements in the robot's perceived behaviour when it is context-aware and adaptive, and provide useful insights that can inform future developments.

- Hierarchical Interest-Driven Associative Goal Babbling for Efficient Bootstrapping of Sensorimotor Skills

    Author: Rayyes, Rania | TU Braunschweig
    Author: Donat, Heiko | Technical University Braunschweig
    Author: Steil, Jochen J. | Technische Universitét Braunschweig
 
    keyword: Learning and Adaptive Systems; AI-Based Methods; Neurorobotics

    Abstract : We propose a novel hierarchical online learning scheme for fast and efficient bootstrapping of sensorimotor skills. Our scheme permits rapid data-driven robot model learning in a "learning while behaving" fashion. It is updated continuously to adapt to time-dependent changes and driven by an intrinsic motivation signal. It utilizes an online associative radial basis function network, which is the first associative dynamic network to be constructed from scratch with high stability. Moreover, we propose a parameter-sharing technique to increase efficiency, stabilize the online scheme, avoid exhaustive parameter tuning, and speed up the learning process. We apply our proposed algorithms on a 7-DoF physical robot manipulator and demonstrate their performance and efficiency.

- Robot-Supervised Learning for Object Segmentation

    Author: Florence, Victoria | University of Michigan
    Author: Corso, Jason | University of Michigan
    Author: Griffin, Brent | University of Michigan
 
    keyword: Learning and Adaptive Systems; Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation

    Abstract : To be effective in unstructured and changing environments, robots must learn to recognize new objects. Deep learning has enabled rapid progress for object detection and segmentation in computer vision; however, this progress comes at the price of human annotators labeling many training examples. This paper addresses the problem of extending learning-based segmentation methods to robotics applications where annotated training data is not available. Our method enables pixelwise segmentation of grasped objects. We factor the problem of segmenting the object from the background into two sub-problems: (1) segmenting the robot manipulator and object from the background and (2) segmenting the object from the manipulator. We propose a kinematics-based foreground segmentation technique to solve (1). To solve (2), we train a self-recognition network that segments the robot manipulator. We train this network without human supervision, leveraging our foreground segmentation technique from (1) to label a training set of images containing the robot manipulator without a grasped object. We demonstrate experimentally that our method outperforms state-of-the-art adaptable in-hand object segmentation. We also show that a training set composed of automatically labelled images of grasped objects improves segmentation performance on a test set of images of the same objects in the environment.

- Gradient and Log-Based Active Learning for Semantic Segmentation of Crop and Weed for Agricultural Robots

    Author: Sheikh, Rasha | University of Bonn
    Author: Milioto, Andres | University of Bonn
    Author: Lottes, Philipp | University of Bonn
    Author: Stachniss, Cyrill | University of Bonn
    Author: Bennewitz, Maren | University of Bonn
    Author: Schultz, Thomas | University of Bonn
 
    keyword: Learning and Adaptive Systems; Robotics in Agriculture and Forestry; Deep Learning in Robotics and Automation

    Abstract : Annotated datasets are essential for supervised learning. However, annotating large datasets is a tedious and time-intensive task. This paper addresses active learning in the context of semantic segmentation with the goal of reducing the human labeling effort. Our application is agricultural robotics and we focus on the task of distinguishing between crop and weed plants from image data. A key challenge in this application is the transfer of an existing semantic segmentation CNN to a new field, in which growth stage, weeds, soil, and weather conditions differ. We propose a novel approach that, given a trained model on one field together with rough foreground segmentation, refines the network on a substantially different field providing an effective method of selecting samples to annotate for supporting the transfer. We evaluated our approach on two challenging datasets from the agricultural robotics domain and show that we achieve a higher accuracy with a smaller number of samples compared to random sampling as well as entropy based sampling, which consequently reduces the required human labeling effort.

- Learning How to Walk: Warm-Starting Optimal Control Solver with Memory of Motion

    Author: Lembono, Teguh Santoso | Idiap Research Institute
    Author: Mastalli, Carlos | University of Edinburgh
    Author: Fernbach, Pierre | Cnrs - Laas
    Author: Mansard, Nicolas | CNRS
    Author: Calinon, Sylvain | Idiap Research Institute
 
    keyword: Learning and Adaptive Systems; Optimization and Optimal Control; Legged Robots

    Abstract : In this paper, we propose a framework to build a memory of motion for warm-starting an optimal control solver for the locomotion task of a humanoid robot. We use HPP Loco3D, a versatile locomotion planner, to generate offline a set of dynamically consistent whole-body trajectory to be stored as the memory of motion. The learning problem is formulated as a regression problem to predict a single-step motion given the desired contact locations, which is used as a building block for producing multi-step motions. The predicted motion is then used as a warm-start for the fast optimal control solver Crocoddyl. We have shown that the approach manages to reduce the required number of iterations to reach the convergence from ~9.5 to only ~3.0 iterations for the single-step motion and from ~6.2 to ~4.5 iterations for the multi-step motion, while maintaining the solution's quality.

- Feedback Linearization for Unknown Systems Via Reinforcement Learning

    Author: Westenbroek, Tyler | University of California, Berkeley
    Author: Fridovich-Keil, David | University of California, Berkeley
    Author: Mazumdar, Eric | University of California, Berkeley
    Author: Arora, Shreyas | Mission San Jose High School
    Author: Prabhu, Valmik | University of California, Berkeley
    Author: Sastry, Shankar | University of California, Berkeley
    Author: Tomlin, Claire | UC Berkeley
 
    keyword: Learning and Adaptive Systems; Model Learning for Control; Deep Learning in Robotics and Automation

    Abstract : We present a novel approach to control design for nonlinear systems which leverages model-free policy optimization techniques to learn a linearizing controller for a physical plant with unknown dynamics. Feedback linearization is a technique from nonlinear control which renders the input-output dynamics of a nonlinear plant emph{linear} under application of an appropriate feedback controller. Once a linearizing controller has been constructed, desired output trajectories for the nonlinear plant can be tracked using a variety of linear control techniques. However, the calculation of a linearizing controller requires a precise dynamics model for the system. As a result, model-based approaches for learning exact linearizing controllers generally require a simple, highly structured model of the system with easily identifiable parameters. In contrast, the model-free approach presented in this paper is able to approximate the linearizing controller for the plant using general function approximation architectures. Specifically, we formulate a continuous-time optimization problem over the parameters of a learned linearizing controller whose optima are the set of parameters which best linearize the plant. We derive conditions under which the learning problem is (strongly) convex and provide guarantees which ensure the true linearizing controller for the plant is recovered. We then discuss how model-free policy optimization algorithms can be used to solve a discrete-time approximation

- Long-Term Robot Navigation in Indoor Environments Estimating Patterns in Traversability Changes

    Author: Nardi, Lorenzo | University of Bonn
    Author: Stachniss, Cyrill | University of Bonn
 
    keyword: Learning and Adaptive Systems; Mapping; Autonomous Vehicle Navigation

    Abstract : Nowadays, mobile robots are deployed in many indoor environments such as offices or hospitals. These environments are subject to changes in the traversability that often happen following patterns. In this paper, we investigate the problem of navigating in such environments over extended periods of time by capturing and exploiting these patterns to make informed decisions for navigation. Our approach uses a probabilistic graphical model to incrementally estimate a model of the traversability changes from the robot's observations and to make predictions at currently unobserved locations. In the belief space defined by the predictions, we plan paths that trade off the risk to encounter obstacles and the information gain of visiting unknown locations. We implemented our approach and tested it in different indoor environments. The experiments suggest that, in the long run, our approach leads robots to navigate along shorter paths compared to following a greedy shortest path policy.

- Sample-And-Computation-Efficient Probabilistic Model Predictive Control with Random Features

    Author: Kuo, Cheng-Yu | Nara Institute of Science and Technology
    Author: Cui, Yunduan | Nara Institute of Science and Technology
    Author: Matsubara, Takamitsu | Nara Institute of Science and Technology
 
    keyword: Learning and Adaptive Systems; Model Learning for Control

    Abstract : Gaussian processes (GPs) based Reinforcement Learning (RL) methods with Model Predictive Control (MPC) have demonstrated their excellent sample efficiency. However, since the computational cost of GPs largely depends on the training sample size, learning an accurate dynamics using GPs result in slow control frequency in MPC. To alleviate this trade-off and achieve a sample-and-computation-efficient nature, we propose a novel model-based RL method with MPC. Our approach employs a linear Gaussian model with randomized features using the Fastfood as an approximated GP dynamics. Then, we derive an analytic moment matching scheme in state prediction with the model and uncertain inputs. Through experiments with simulated and real robot control tasks, the sample efficiency, as well as the computational efficiency of our model-based RL method, are demonstrated.

- Sample-Efficient Robot Motion Learning Using Gaussian Process Latent Variable Models

    Author: Delgado-Guerrero, Juan Antonio | IRI
    Author: Colomé, Adrià | Institut De Robòtica I Informàtica Industrial (CSIC-UPC), Q28180
    Author: Torras, Carme | Csic - Upc
 
    keyword: Learning and Adaptive Systems; Redundant Robots; Learning from Demonstration

    Abstract : Robotic manipulators are reaching a state where we could see them in household environments in the following decade. Nevertheless, such robots need to be easy to instruct by lay people. This is why kinesthetic teaching has become very popular in recent years, in which the robot is taught a motion that is encoded as a parametric function - usually a Movement Primitive (MP)-. This approach produces trajectories that are usually suboptimal, and the robot needs to be able to improve them through trial-and-error. Such optimization is often done with Policy Search (PS) reinforcement learning, using a given reward function. PS algorithms can be classified as model-free, where neither the environment nor the reward function are modelled, or model-based, which can use a surrogate model of the reward function and/or a model for the dynamics of the task. However, MPs can become very high-dimensional in terms of parameters, which constitute the search space, so their optimization often requires too many samples. In this paper, we assume we have a robot motion task characterized with an MP of which we cannot model the dynamics. We build a surrogate model for the reward function, that maps an MP parameter latent space (obtained through a Mutual-information-weighted Gaussian Process Latent Variable Model) into a reward. While we do not model the task dynamics, using mutual information to shrink the task space makes it more consistent with the reward and so the policy improvement is faster.

- Iterative Learning Based Feedforward Control for Transition of a Biplane-Quadrotor Tailsitter UAS

    Author: Raj, Nidhish | Indian Institute of Technology Kanpur
    Author: Simha, Ashutosh | Tallinn University of Technology
    Author: Kothari, Mangal | Indian Institute of Technology Kanpur
    Author: Abhishek, Abhishek | Indian Institute of Technology Kanpur
    Author: Banavar, Ravi N | I. I. T. Bombay
 
    keyword: Learning and Adaptive Systems; Aerial Systems: Mechanics and Control; Aerial Systems: Perception and Autonomy

    Abstract : This paper provides a real time on-board algorithm for a biplane-quadrotor to iteratively learn a forward transition maneuver via repeated flight trials. The maneuver is controlled by regulating the pitch angle and propeller thrust according to feedforward control laws that are parameterized by polynomials. Based on a nominal model with simplified aerodynamics, the optimal coefficients of the polynomials are chosen through simulation such that the maneuver is completed with specified terminal conditions on altitude and air speed. In order to compensate for modeling errors, repeated flight trials are performed by updating the feedforward control paramters according to an iterative learning algorithm until the maneuver is perfected. A geometric attitude controller, valid for all flight modes is employed in order to track the pitch angle according to the feedforward law. Further, a high-fidelity thrust model of the propeller for varying advance-ratio and orientation angle is obtained from wind tunnel data which is captured using a neural network model. This facilitates accurate application of feedforward thrust for varying flow conditions during transition. Experimental flight trials are performed to demonstrate the robustness and rapid convergence of the proposed learning algorithm.

- Reinforcement Learning for Adaptive Illumination with X-Rays

    Author: Betterton, Jean-Raymond | Stanford University
    Author: Ratner, Daniel | SLAC National Accelerator Laboratory
    Author: Webb, Samuel | SLAC National Accelerator Laboratory
    Author: Kochenderfer, Mykel | Stanford University
 
    keyword: Learning and Adaptive Systems; AI-Based Methods

    Abstract : We propose a learning algorithm for automating image sampling in scientific applications. We consider settings where images are sampled by controlling a probe beam's scanning trajectory over the image surface. We explore alternatives to obtaining images by the standard rastering method. We formulate the scanner control problem as a reinforcement learning (RL) problem and train a policy to adaptively sample only the highest value regions of the image, choosing the acquisition time and resolution for each sample position based on an observation of previous readings. We use convolutional neural network (CNN) policies to control the scanner as a way to generalize our approach to larger samples. We show simulation results for a simple policy on both synthetic data and real world data from an archaeological application.

- Efficient Updates for Data Association with Mixtures of Gaussian Processes

    Author: Lee, Ki Myung Brian | University of Technology Sydney
    Author: Martens, Wolfram | Siemens Mobility GmbH
    Author: Khatkar, Jayant | University of Technology Sydney
    Author: Fitch, Robert | University of Technology Sydney
    Author: Mettu, Ramgopal | Tulane University
 
    keyword: Learning and Adaptive Systems; AI-Based Methods; Big Data in Robotics and Automation

    Abstract : Gaussian processes (GPs) enable a probabilistic approach to important estimation and classification tasks that arise in robotics applications. Meanwhile, most GP-based methods are often prohibitively slow, thereby posing a substantial barrier to practical applications. Existing ``sparse'' methods to speed up GPs seek to either make the model more sparse, or find ways to more efficiently manage a large covariance matrix. In this paper, we present an orthogonal approach that memoises (i.e. reuses) previous computations in GP inference. We demonstrate that a substantial speedup can be achieved by incorporating memoisation into applications in which GPs must be updated frequently. Moreover, we also show how memoisation can be used in conjunction with sparse methods and demonstrate a synergistic improvement in performance. Across three robotic vision applications, we demonstrate between 40-100% speed-up over the standard method for inference in GP mixtures.

## Surgical Robotics: Laparascopy I
- Hand-Eye Calibration of Surgical Instrument for Robotic Surgery Using Interactive Manipulation

    Author: Zhong, Fangxun | The Chinese University of Hong Kong
    Author: Wang, Zerui | The Chinese University of Hong Kong
    Author: Chen, Wei | The Chinese University of Hong Kong
    Author: Wang, Yaqing | The Chinese University of Hong Kong
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Surgical Robotics: Laparoscopy; Calibration and Identification

    Abstract : Conventional robot hand-eye calibration methods are impractical for localizing robotic instruments in minimally-invasive surgeries under intra-corporeal workspace after pre-operative set-up. In this letter, we present a new approach to autonomously calibrate a robotic instrument relative to a monocular camera without recognizing calibration objects or salient features. The algorithm leverages interactive manipulation (IM) of the instrument for tracking its rigid-body motion behavior subject to the remote center-of-motion constraint. An adaptive controller is proposed to regulate the IM-induced instrument trajectory, using visual feedback, within a 3D interactive feature plane which is observable from both the robot base and the camera. The eye-to-hand orientation and position is then computed via a dual-stage process allowing parameter estimation in low-dimensional spaces. The method also does not require exact knowledge of the instrument model or large-scale data collection. Results from simulations and experiments on the da Vinci Research Kit are demonstrated via a laparoscopy resembling set-up using the proposed framework.

- Real-Time Data Driven Precision Estimator for RAVEN-II Surgical Robot End Effector Position

    Author: Peng, Haonan | University of Washington
    Author: Yang, Xingjian | University of Washington
    Author: Su, Yun-Hsuan | University of Washington
    Author: Hannaford, Blake | University of Washington
 
    keyword: Surgical Robotics: Laparoscopy; Computer Vision for Medical Robotics; Medical Robots and Systems

    Abstract :     Abstract - Surgical robots have been introduced to operating rooms over the past few decades due to their high sensitivity, small size, and remote controllability. The cable-driven nature of many surgical robots allows the systems to be dexterous and lightweight, with diameters as low as 5mm. However, due to the slack and stretch of the cables and the backlash of the gears, inevitable uncertainties are brought into the kinematics calculation [1]. Since the reported end effector position of surgical robots like RAVEN-II [2] is directly calculated using the motor encoder measurements and forward kinematics, it may contain relatively large error up to 10mm, whereas semi-autonomous functions being introduced into abdominal surgeries require position inaccuracy of at most 1mm. To resolve the problem, a cost-effective, real-time and data-driven pipeline for robot end effector position precision estimation is proposed and tested on RAVEN-II. Analysis shows an improved end effector position error of around 1mm RMS traversing through the entire robot workspace without high-resolution motion tracker. The open source code, data sets, videos, and user guide can be found at //github.com/HaonanPeng/RAVEN_Neural_Network_Estimator.

- Vision-Based Dynamic Virtual Fixtures for Tools Collision Avoidance in Robotic Surgery

    Author: Moccia, Rocco | Université Degli Studi Di Napoli, Federico II
    Author: Iacono, Cristina | Université Degli Studi Di Napoli Federico II
    Author: Siciliano, Bruno | Univ. Napoli Federico II
    Author: Ficuciello, Fanny | Université Di Napoli Federico II
 
    keyword: Surgical Robotics: Laparoscopy; Collision Avoidance; Telerobotics and Teleoperation

    Abstract : In robot-aided surgery, during the execution of typical bimanual procedures such as dissection, surgical tools can collide and create serious damage to the robot or tissues. The da Vinci robot is one of the most advanced and certainly the most widespread robotic system dedicated to minimally invasive surgery. Although the procedures performed by da Vinci-like surgical robots are teleoperated, potential collisions between surgical tools are a very sensitive issue declared by surgeons. Shared control techniques based on Virtual Fixtures (VF) can be an effective way to help the surgeon prevent tools collision.<p>This paper presents a surgical tools collision avoidance method that uses Forbidden Region Virtual Fixtures. Tool clashing is avoided by rendering a repulsive force to the surgeon. To ensure the correct definition of the VF, a marker-less tool tracking method, using deep neural network architecture for tool segmentation, is adopted. The use of direct kinematics for tools collision avoidance is affected by tools position error introduced by robot component elasticity during tools interaction with the environment. On the other hand, kinematics information can help in case of occlusions of the camera. Therefore, this work proposes an Extended Kalman Filter (EKF) for pose estimation which ensures a more robust application of VF on the tool, coupling vision and kinematics information. The entire pipeline is tested in different tasks using the da Vinci Research Kit system.

- An Experimental Comparison towards Autonomous Camera Navigation to Optimize Training in Robot Assisted Surgery

    Author: Mariani, Andrea | Scuola Superiore Sant'Anna
    Author: Colaci, Giorgia | Politecnico Di Milano
    Author: Da Col, Tommaso | Politecnico Di Milano
    Author: Sanna, Nicole | Politecnico Di Milano
    Author: Vendrame, Eleonora | Politecnico Di Milano
    Author: Menciassi, Arianna | Scuola Superiore Sant'Anna - SSSA
    Author: De Momi, Elena | Politecnico Di Milano
 
    keyword: Telerobotics and Teleoperation; Virtual Reality and Interfaces; Surgical Robotics: Laparoscopy

    Abstract : Robot-Assisted Surgery enhances vision and it can restore depth perception, but it introduces the need for learning how to tele-operatively control both the surgical tools and the endoscope. Together with the complexity of selecting the optimal viewpoint to carry out the procedure, this requires distinct training. This work proposes an autonomous camera navigation during the initial stages of training in order to optimize the learning of these skills. A user study involving 26 novice participants was carried out using the master console of the da Vinci Research Kit and a virtual reality training environment. The subjects were randomly divided into two groups: the control group that manually controlled the camera as in the current practice and the experimental group that underwent the autonomous navigation. After training, the time-accuracy metrics of the users who underwent autonomous camera navigation were significantly higher with respect to the control group. Additionally, autonomous camera navigation seemed to be capable to provide an imprinting about endoscope management.

- Temporal Segmentation of Surgical Sub-Tasks through Deep Learning with Multiple Data Sources

    Author: Qin, Yidan | Intuitive Surgical
    Author: Aghajani Pedram, Sahba | University of California, Los Angeles
    Author: Feyzabadi, Seyedshams | UC Merced
    Author: Allan, Max | Intuitive Surgical
    Author: McLeod, Angus Jonathan | Intuitive Surgical
    Author: Burdick, Joel | California Institute of Technology
    Author: Azizian, Mahdi | Intuitive Surgical
 
    keyword: Surgical Robotics: Laparoscopy; Deep Learning in Robotics and Automation; Medical Robots and Systems

    Abstract : Many tasks in robot-assisted surgeries (RAS) can be represented by finite-state machines (FSMs), where each state represents either an action (such as picking up a needle) or an observation (such as bleeding). A crucial step towards the automation of such surgical tasks is the temporal perception of the current surgical scene, which requires a real-time estimation of the states in the FSMs. The objective of this work is to estimate the current state of the surgical task based on the actions performed or events occurred as the task progresses. We propose Fusion-KVE, a unified surgical state estimation model that incorporates multiple data sources including the Kinematics, Vision, and system Events. Additionally, we examine the strengths and weaknesses of different state estimation models in segmenting states with different representative features or levels of granularity. We evaluate our model on the JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS), as well as a more complex dataset involving robotic intra-operative ultrasound (RIOUS) imaging, created using the da Vinci Xi surgical system. Our model achieves a superior frame-wise state estimation accuracy up to 89.4%, which improves the state-of-the-art surgical state estimation models in both JIGSAWS suturing dataset and our RIOUS dataset.

- Controlling Assistive Robots with Learned Latent Actions

    Author: Losey, Dylan | Stanford University
    Author: Srinivasan, Krishnan | Stanford University
    Author: Mandlekar, Ajay Uday | Stanford University
    Author: Garg, Animesh | University of Toronto
    Author: Sadigh, Dorsa | Stanford University
 
    keyword: Physically Assistive Devices; Cognitive Human-Robot Interaction; Human-Centered Robotics

    Abstract : Assistive robotic arms enable users with physical disabilities to perform everyday tasks without relying on a caregiver. Unfortunately, the very dexterity that makes these arms useful also makes them challenging to teleoperate: the robot has more degrees-of-freedom than the human can directly coordinate with a handheld joystick. Our insight is that we can make assistive robots easier for humans to control by leveraging latent actions. Latent actions provide a low-dimensional embedding of high-dimensional robot behavior: for example, one latent dimension might guide the assistive arm along a pouring motion. In this paper, we design a teleoperation algorithm for assistive robots that learns latent actions from task demonstrations. We formulate the controllability, consistency, and scaling properties that user-friendly latent actions should have, and evaluate how different low-dimensional embeddings capture these properties. Finally, we conduct two user studies on a robotic arm to compare our latent action approach to both state-of-the-art shared autonomy baselines and a teleoperation strategy currently used by assistive arms. Participants completed assistive eating and cooking tasks more efficiently when leveraging our latent actions, and also subjectively reported that latent actions made the task easier to perform. The video accompanying this paper can be found at: https://youtu.be/wjnhrzugBj4.


## Surgical Robotics: Laparoscopy II

- SuPer: A Surgical Perception Framework for Endoscopic Tissue Manipulation with Surgical Robotics

    Author: Li, Yang | Zhejiang University
    Author: Richter, Florian | University of California, San Diego
    Author: Lu, Jingpei | University of California, San Diego
    Author: Funk, Emily | University of California, San Diego
    Author: Orosco, Ryan | University of California, San Diego
    Author: Zhu, Jianke | Zhejiang University
    Author: Yip, Michael C. | University of California, San Diego
 
    keyword: Computer Vision for Medical Robotics; Surgical Robotics: Laparoscopy; Perception for Grasping and Manipulation

    Abstract : Traditional control and task automation have been successfully demonstrated in a variety of structured, controlled environments through the use of highly specialized modeled robotic systems in conjunction with multiple sensors. However, the application of autonomy in endoscopic surgery is very challenging, particularly in soft tissue work, due to the lack of high-quality images and the unpredictable, constantly deforming environment. In this work, we propose a novel surgical perception framework, SuPer, for surgical robotic control. This framework continuously collects 3D geometric information that allows for mapping a deformable surgical field while tracking rigid instruments within the field. To achieve this, a model-based tracker is employed to localize the surgical tool with a kinematic prior in conjunction with a model-free tracker to reconstruct the deformable environment and provide an estimated point cloud as a mapping of the environment. The proposed framework was implemented on the da Vinci Surgical System in real-time with an end-effector controller where the target configurations are set and regulated through the framework. Our proposed framework successfully completed soft tissue manipulation tasks with high accuracy. The demonstration of this novel framework is promising for the future of surgical autonomy. In addition, we provide our dataset for further surgical research.

- Multi-Task Recurrent Neural Network for Surgical Gesture Recognition and Progress Prediction

    Author: van Amsterdam, Beatrice | University College London
    Author: Clarkson, Matt | University College London
    Author: Stoyanov, Danail | University College London
 
    keyword: Surgical Robotics: Laparoscopy; Object Detection, Segmentation and Categorization

    Abstract : Surgical gesture recognition is important for surgical data science and computer-aided intervention. Even with robotic kinematic information, automatically segmenting surgical steps presents numerous challenges because surgical demonstrations are characterized by high variability in style, duration and order of actions. In order to extract discriminative features from the kinematic signals and boost recognition accuracy, we propose a multi-task recurrent neural network for simultaneous recognition of surgical gestures and estimation of a novel formulation of surgical task progress. To show the effectiveness of the presented approach, we evaluate its application on the JIGSAWS dataset, that is currently the only publicly available dataset for surgical gesture recognition featuring robot kinematic data. We demonstrate that recognition performance improves in multi-task frameworks with progress estimation without any additional manual labelling and training.

- Neural Network Based Inverse Dynamics Identification and External Force Estimation on the Da Vinci Research Kit

    Author: Yilmaz, Nural | Marmara University
    Author: Wu, Jie Ying | Johns Hopkins University
    Author: Kazanzides, Peter | Johns Hopkins University
    Author: Tumerdem, Ugur | Marmara University
 
    keyword: Surgical Robotics: Laparoscopy; Force and Tactile Sensing; Telerobotics and Teleoperation

    Abstract : Most current surgical robotic systems lack the ability to sense tool/tissue interaction forces, which motivates research in methods to estimate these forces from other available measurements, primarily joint torques. These methods require the internal joint torques, due to the robot inverse dynamics, to be subtracted from the measured joint torques. This paper presents the use of neural networks to estimate the inverse dynamics of the da Vinci surgical robot, which enables estimation of the external environment forces. Experiments with motions in free space demonstrate that the neural networks can estimate the internal joint torques within 10% normalized rootmean-square error (NRMSE), which outperforms model-based approaches in the literature. Comparison with an external force sensor shows that the method is able to estimate environment forces within about 10% NRMSE.

- Visual Servo of a 6-DOF Robotic Stereo Flexible Endoscope Based on Da Vinci Research Kit (dVRK) System

    Author: Ma, Xin | Chinese Univerisity of HongKong
    Author: Song, Chengzhi | Chinese University of Hong Kong,
    Author: Chiu, Philip, Wai-yan | Chinese University of Hong Kong
    Author: Li, Zheng | The Chinese University of Hong Kong
 
    keyword: Surgical Robotics: Laparoscopy; Medical Robots and Systems; Flexible Robots

    Abstract : Endoscopes play an important role in minimally invasive surgery (MIS). Due to the advantages of less occupied motion space and enhanced safety, flexible endoscopes are drawing more and more attention. However, the structure of the flexible section makes it difficult for surgeons to manually rotate and guide the view of endoscopes. To solve these problems, we developed a 6-DOF robotic stereo flexible endoscope (RSFE) based on the da Vinci Research Kit (dVRK). Then an image-based endoscope guidance method with depth information is proposed for the RSFE. With this method, the view and insertion depth of the RSFE can be adjusted by tracking the surgical instruments automatically. Additionally, an image-based view rotation control method is proposed, with which the rotation of the view can be controlled by tracking two surgical instruments. The experimental results show that the proposed methods control the direction and rotation of the view of the flexible endoscope faster than the manual control method. Lastly, an ex vivo experiment is performed to demonstrate the feasibility of the proposed control method and system.

- Reflective-AR Display: An Interaction Methodology for Virtual-To-Real Alignment in Medical Robotics

    Author: Fotouhi, Javad | Johns Hopkins University
    Author: Song, Tianyu | Verb Surgical Inc
    Author: Mehrfard, Arian | Johns Hopkins University
    Author: Taylor, Giacomo | Verb Surgical Inc
    Author: Wang, Qiaochu | Johns Hopkins University
    Author: Xian, Fengfan | Johns Hopkins University
    Author: Martin-Gomez, Alejandro | Technical University of Munich
    Author: Fuerst, Bernhard | Verb Surgical Inc
    Author: Armand, Mehran | Johns Hopkins University Applied Physics Laboratory
    Author: Unberath, Mathias | Johns Hopkins University
    Author: Navab, Nassir | Johns Hopkins University
 
    keyword: Surgical Robotics: Laparoscopy

    Abstract : Robot-assisted minimally invasive surgery has shown to improve patient outcomes, as well as reduce complications and recovery time for several clinical applications. While increasingly configurable robotic arms can maximize reach and avoid collisions in cluttered environments, positioning them appropriately during surgery is complicated because safety regulations prevent automatic driving. We propose a head-mounted display (HMD) based augmented reality (AR) system designed to guide optimal surgical arm set up. The staff equipped with HMD aligns the robot with its planned virtual counterpart. In this user-centric setting, the main challenge is the perspective ambiguities hindering such collaborative robotic solution. To overcome this challenge, we introduce a novel registration concept for intuitive alignment of AR content to its physical counterpart by providing a multi-view AR experience via reflective-AR displays that simultaneously show the augmentations from multiple viewpoints. Using this system, users can visualize different perspectives while actively adjusting the pose to determine the registration transformation that most closely superimposes the virtual onto the real. The experimental results demonstrate improvement in the interactive alignment of a virtual and real robot when using a reflective-AR display. We also present measurements from configuring a robotic manipulator in a simulated trocar placement surgery using the AR guidance methodology.



## Surgical Robotics: Steerable Catheters/Needles
- Aortic 3D Deformation Reconstruction Using 2D X-Ray Fluoroscopy and 3D Pre-Operative Data for Endovascular Interventions

    Author: Zhang, Yanhao | University of Technology Sydney
    Author: Zhao, Liang | University of Technology Sydney
    Author: Huang, Shoudong | University of Technology, Sydney
 
    keyword: Surgical Robotics: Steerable Catheters/Needles; Computer Vision for Medical Robotics; Mapping

    Abstract : Current clinical endovascular interventions rely on 2D guidance for catheter manipulation. Although an aortic 3D surface is available from the pre-operative CT/MRI imaging, it cannot be used directly as a 3D intra-operative guidance since the vessel will deform during the procedure. This paper aims to reconstruct the live 3D aortic deformation by fusing the static 3D model from the pre-operative data and the 2D live imaging from fluoroscopy. In contrast to some existing deformation reconstruction frameworks which require 3D observations such as RGB-D or stereo images, fluoroscopy only presents 2D information. In the proposed framework, a 2D-3D registration is performed and the reconstruction process is formulated as a non-linear optimization problem based on the deformation graph approach. Detailed simulations and phantom experiments are conducted and the result demonstrates the reconstruction accuracy and robustness, as well as the potential clinical value of this framework.

- Design and Kinematic Modeling of a Novel Steerable Needle for Image-Guided Insertion

    Author: Chen, Yuyang | Shanghai Jiao Tong University
    Author: Yang, Haozhe | School of Mechanical Engineering, Shanghai Jiao Tong University,
    Author: Liu, Xu | Shanghai Jiao Tong University
    Author: Xu, Kai | Shanghai Jiao Tong University
 
    keyword: Surgical Robotics: Steerable Catheters/Needles; Medical Robots and Systems

    Abstract : Needle-based procedures, such as biopsy and percutaneous tumor ablation, highly depend on the accuracy of needle placement. The accuracy is significantly affected by the needle-tissue interaction no matter what needles (straight or steerable) are used. Due to the unknown tissue mechanics, it is challenging to achieve high accuracy in practice. This paper hence proposes a needle design with an articulated tip for increased steerability and improved needle path consistency. Due to the passive needle tip articulation, tissue mechanics always plays a dominant role such that the needle creates similar paths with approximately piece-wise constant curvature in different tissues. Kinematics model for the proposed needle is presented. The algorithms of path planning and needle tip pose estimation under external imaging modality are developed. Experimental verifications were conducted to demonstrate the needle's steerability as well as the target-reaching capability with obstacles avoidance.

- Robotic Needle Insertion in Moving Soft Tissues Using Constraint-Based Inverse Finite Element Simulation

    Author: Baksic, Paul | Université De Strasbourg
    Author: Courtecuisse, Hadrien | AVR, CNRS Strasbourg
    Author: Duriez, Christian | INRIA
    Author: Bayle, Bernard | University of Strasbourg
 
    keyword: Surgical Robotics: Steerable Catheters/Needles; Medical Robots and Systems

    Abstract : This paper introduces a method for robotic steering of a flexible needle inside moving and deformable tissues. The method relies on a set of objective functions allowing to automatically steer the needle along a predefined path. In order to follow the desired trajectory, an inverse problem linking the motion of the robot end effector with the objective functions is solved using a Finite Element simulation. The main contribution of the article is the new constraint-based formulation of the objective functions allowing to: 1) significantly reduce the computation time; 2) increase the accuracy and stability of the simulation-guided needle insertion. The method is illustrated, and its performances are characterized in a realistic framework, using a direct simulation of the respiratory motion generated from in vivo data of a pig. Despite the highly non-linear behavior of the numerical simulation and the significant deformations occurring during the insertion, the obtained performances enable the possibility to follow the trajectory with the desired accuracy for medical purpose.

- Collaborative Robot-Assisted Endovascular Catheterization withGenerative Adversarial Imitation Learning

    Author: Chi, Wenqiang | Imperial College London
    Author: Dagnino, Giulio | Imperial College London
    Author: Kwok, Trevor M Y | Imperial College London
    Author: Nguyen, Anh | Imperial College London
    Author: Kundrat, Dennis | Imperial College London
    Author: Abdelaziz, Mohamed Essam Mohamed Kassem | Imperial College London
    Author: Riga, Celia | Imperial College London
    Author: Bicknell, Colin | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Surgical Robotics: Planning; Medical Robots and Systems; Learning from Demonstration

    Abstract : Master-slave systems for endovascular catheterization have brought major clinical benefits including reduced radiation doses to the operators, improved precision and stability of the instruments, as well as reduced procedural duration. Emerging deep reinforcement learning (RL) technologies could potentially automate more complex endovascular tasks with enhanced success rates, more consistent motion and reduced fatigue and cognitive workload of the operators. However, the complexity of the pulsatile flows within the vasculature and non-linear behavior of the instruments hinder the use of model-based approaches for RL. This paper describes model-free generative adversarial imitation learning to automate a standard arterial catherization task. The automation policies have been trained in a pre-clinical setting. Detailed validation results show high success rates after skill transfer to a different vascular anatomical model. The quality of the catheter motions also shows less mean and maximum contact forces compared to manual-based approaches.

- A Novel Sensing Method to Detect Tissue Boundaries During Robotic Needle Insertion Based on Laser Doppler Flowmetry

    Author: Virdyawan, Vani | Imperial College London
    Author: Dessi, Orsina | Imperial College London
    Author: Rodriguez y Baena, Ferdinando | Imperial College, London, UK
 
    keyword: Surgical Robotics: Steerable Catheters/Needles

    Abstract : This study investigates the use of Laser Doppler Flowmetry (LDF) as a method to detect tissue transitions during robotic needle insertions. Insertions were performed in gelatin tissue phantoms with different optical and mechanical properties and into an ex-vivo sheep brain. The effect of changing the optical properties of gelatin tissue phantoms was first investigated and it was shown that using gelatin concentration to modify the stiffness of samples was suitable. Needle insertion experiments were conducted into both one-layer and two-layer gelatin phantoms. In both cases, three stages could be observed in the perfusion values: tissue loading, rupture and tissue cutting. These were correlated to force values measured from the tip of the needle during insertion. The insertions into an ex-vivo sheep brain also clearly showed the time of rupture in both force and perfusion values, demonstrating that tissue puncture can be detected using an LDF sensor at the tip of a needle.

- GA3C Reinforcement Learning for Surgical Steerable Catheter Path Planning

    Author: Segato, Alice | Politecnico Di Milano, Milano , Italy
    Author: Sestini, Luca | Politecnico Di Milano
    Author: Castellano, Antonella | Neuroradiology Unit and CERMAC, Vita-Salute San Raffaele Univers
    Author: De Momi, Elena | Politecnico Di Milano
 
    keyword: Surgical Robotics: Steerable Catheters/Needles; Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : Path planning algorithms for steerable catheters, must guarantee anatomical obstacles avoidance, reduce the insertion length and ensure the compliance with needle kinematics. The majority of the solutions in literature focuses on graph based or sampling based methods, both limited by the impossibility to directly obtain smooth trajectories. In this work we formulate the path planning problem as a reinforcement learning problem and show that the trajectory planning model, generated from the training, can provide the user with optimal trajectories in terms of obstacle clearance and kinematic constraints. We obtain 2D and 3D environments from MRI images processing and we implement a GA3C algorithm to create a path planning model, able to generalize on different patients anatomies. The curvilinear trajectories obtained from the model in 2D and 3D environments are compared to the ones obtained by A* and RRT* algorithms. Our method achieves state-of-the-art performances in terms of obstacle avoidance, trajectory smoothness and computational time proving this algorithm as valid planning method for complex environments.



## Path Planning for Multiple Mobile Robots or Agents
- Online Trajectory Generation with Distributed Model Predictive Control for Multi-Robot Motion Planning

    Author: Luis, Carlos E. | University of Toronto
    Author: Vukosavljev, Marijan | University of Toronto
    Author: Schoellig, Angela P. | University of Toronto
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems; Distributed Robot Systems

    Abstract : We present a distributed model predictive control (DMPC) algorithm to generate trajectories in real-time for multiple robots. We adopted the on-demand collision avoidance method presented in previous work to efficiently compute non-colliding trajectories in transition tasks. An event-triggered replanning strategy is proposed to account for disturbances. Our simulation results show that the proposed collision avoidance method can reduce, on average, around 50% of the travel time required to complete a multi-agent point-to-point transition when compared to the well-studied Buffered Voronoi Cells (BVC) approach. Additionally, it shows a higher success rate in transition tasks with a high density of agents, with more than 90% success rate with 30 palm-sized quadrotor agents in a 18 m^3 arena. The approach was experimentally validated with a swarm of up to 20 drones flying in close proximity.

- One-Shot Multi-Path Planning for Robotic Applications Using Fully Convolutional Networks

    Author: Kulvicius, Tomas | University of Goettingen
    Author: Herzog, Sebastian | Department of Computational Neuroscience, University of Goetting
    Author: Lüddecke, Timo | University of Göttingen
    Author: Tamosiunaite, Minija | University of Goettingen
    Author: Wörgötter, Florentin | University of Göttingen
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Deep Learning in Robotics and Automation

    Abstract : Path planning is important for robot action execution, since a path or a motion trajectory for a particular action has to be defined first before the action can be executed. Most of the current approaches are iterative methods where the trajectory is generated by predicting the next state based on the current state. Here we propose a novel method by utilising a fully convolutional neural network, which allows generation of complete paths, even for several agents without any iterations. We demonstrate that our method is able to successfully generate optimal or close to optimal paths (less than 10% longer) in more than 99% of the cases for single path predictions in 2D and 3D environments. Furthermore, we show that the network is - without specific training on such cases - able to create (close to) optimal paths in 96% of the cases for two and in 84% of the cases for three simultaneously generated paths.

- Walk, Stop, Count, and Swap: Decentralized Multi-Agent Path Finding with Theoretical Guarantees

    Author: Wang, Hanlin | Northwestern University
    Author: Rubenstein, Michael | Northwestern University
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Distributed Robot Systems; Swarms

    Abstract : For multi-agent path-finding (MAPF) problems, finding the optimal solution has been shown to be NP-Complete. Here we present WSCaS (Walk, Stop, Count, and Swap), a decentralized multi-agent path-finding algorithm that can provide theoretical completeness and optimality guarantees. That is, WSCaS is able to deliver a worst case O(1)-approximate distance-optimal solution to MAPF instances on most grid maps. Moreover, the algorithm's cost is independent of the swarm's size with respect to computation complexity, memory complexity, as well as communication complexity, therefore the algorithm can scale well with the number of agents in practice. The algorithm is executed on 1024 simulated agents as well as 100 physical robots, and the results show that WSCaS is robust to real-world non-idealitys.

- Efficient Iterative Linear-Quadratic Approximations for Nonlinear Multi-Player General-Sum Differential Games

    Author: Fridovich-Keil, David | University of California, Berkeley
    Author: Ratner, Ellis | University of California, Berkeley
    Author: Peters, Lasse | TU Hamburg
    Author: Dragan, Anca | University of California Berkeley
    Author: Tomlin, Claire | UC Berkeley
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Human-Centered Robotics; Motion and Path Planning

    Abstract : Differential games offer a powerful theoretical framework for formulating safety and robustness problems in optimal control. Unfortunately, numerical solution techniques for general nonlinear dynamical systems scale poorly with state dimension and are rarely used in applications requiring real- time computation. For single-agent optimal control problems, however, local methods based on efficiently solving iterated approximations with linear dynamics and quadratic costs are becoming increasingly popular. We take inspiration from one such method, the iterative linear quadratic regulator (ILQR), and observe that efficient algorithms also exist to solve multi-player linear-quadratic games. Whereas ILQR converges to a local solution of the optimal control problem, if our method converges it returns a local Nash equilibrium of the differential game. We benchmark our method in a three-player general- sum simulated example, in which it takes &lt; 0.75 s to identify a solution and &lt; 50 ms to solve warm-started subproblems in a receding horizon. We also demonstrate our approach in hardware, operating in real-time and following a 10 s receding horizon.

- Online Motion Planning for Deforming Maneuvering and Manipulation by Multilinked Aerial Robot Based on Differential Kinematics

    Author: Zhao, Moju | The University of Tokyo
    Author: Shi, Fan | The University of Tokyo
    Author: Anzai, Tomoki | The University of Tokyo
    Author: Okada, Kei | The University of Tokyo
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Aerial Systems: Applications; Motion and Path Planning; Motion Control

    Abstract : State-of-the-art work on deformable multirotor aerial robots has developed a strong maneuvering ability in such robots, whereas there is no versatile aerial robot that can perform both deforming maneuvering and aerial manipulation yet. However, a novel multilinked aerial robot presented in our previous work, called DRAGON, has both potential because of its serial-link structure. Therefore, an online motion planning method for such a multilinked aerial robot is required. In this paper, we first reveal the general statics model of the multilinked aerial robot, which involves the influence of joint torque, rotor thrust force, external wrench, and gravity, and further discuss the necessary rotor thrust force and joint torque required to compensate for external force and gravity under the quasi-static assumption. Then, we propose a real-time motion planning method, which sequentially solves the differential kinematics problem. This method considers the limitations of rotor thrust force and joint torque, as well as kinematics constraints. Furthermore, we introduce the integrated control framework, which can follow a quasi-static multilinks' trajectory and compensate for the external wrench. Finally, experiments to squeeze a virtual hatch covered by a movable plate are performed with quad-type DRAGON to demonstrate the feasibility of the proposed motion planning method in real-time.

- DDM: Fast Near-Optimal Multi-Robot Path Planning Using Diversified-Path and Optimal Sub-Problem Solution Database Heuristics

    Author: Han, Shuai D. | Rutgers University
    Author: Yu, Jingjin | Rutgers University
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Planning, Scheduling and Coordination; Multi-Robot Systems

    Abstract : We propose a novel centralized and decoupled algorithm, DDM, for solving multi-robot path planning problems in grid graphs, targeting on-demand and automated warehouse-like settings. Two settings are studied: a traditional one whose objective is to move a set of robots from their respective initial vertices to the goal vertices as quickly as possible, and a dynamic one which requires frequent re-planning to accommodate for goal configuration adjustments. Among other techniques, DDM is mainly enabled through exploiting two innovative heuristics: path diversification and optimal sub-problem solution databases. The two heuristics attack two distinct phases of a decoupling-based planner: while path diversification allows the more effective use of the entire workspace for robot travel, optimal sub-problem solution databases facilitate the fast resolution of local path conflicts. Extensive evaluation demonstrates that DDM achieves high levels of scalability and solution quality close to the optimum.

- UBAT: On Jointly Optimizing UAV Trajectories and Placement of Battery Swap Stations

    Author: Won, Myounggyu | University of Memphis
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Aerial Systems: Applications; Motion and Path Planning

    Abstract : Unmanned aerial vehicles (UAVs) have been widely used in many applications. The limited flight time of UAVs, however, still remains as a major challenge. Although numerous approaches have been developed to recharge the battery of UAVs effectively, little is known about optimal methodologies to deploy charging stations. In this paper, we address the charging station deployment problem with an aim to find the optimal number and locations of charging stations such that the system performance is maximized. We show that the problem is NP-Hard and propose UBAT, a heuristic framework based on the ant colony optimization (ACO) to solve the problem. Additionally, a suite of algorithms are designed to enhance the execution time and the quality of the solutions for UBAT. Through extensive simulations, we demonstrate that UBAT effectively performs multi-objective optimization of generation of UAV trajectories and placement of charging stations that are within 8.3% and 7.3% of the true optimal solutions, respectively.

- Efficient Multi-Agent Trajectory Planning with Feasibility Guarantee Using Relative Bernstein Polynomial

    Author: Park, Jungwon | Seoul National University
    Author: Kim, Junha | Seoul National University
    Author: Jang, Inkyu | Seoul National University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Collision Avoidance; Swarms

    Abstract : This paper presents a new efficient algorithm which guarantees a solution for a class of multi-agent trajectory planning problems in obstacle-dense environments. Our algorithm combines the advantages of both grid-based and optimization-based approaches, and generates safe, dynamically feasible trajectories without suffering from an erroneous optimization setup such as imposing infeasible collision constraints. We adopt a sequential optimization method with dummy agents to improve the scalability of the algorithm, and utilize the convex hull property of Bernstein and relative Bernstein polynomial to replace non-convex collision avoidance constraints to convex ones. The proposed method can compute the trajectory for 64 agents on average 6.36 seconds with Intel Core i7-7700 @ 3.60GHz CPU and 16G RAM, and it reduces more than 50% of the objective cost compared to our previous work. We validate the proposed algorithm through simulation and flight tests.

- Optimal Sequential Task Assignment and Path Finding for Multi-Agent Robotic Assembly Planning

    Author: Brown, Kyle | Stanford University
    Author: Peltzer, Oriana | Stanford University
    Author: Sehr, Martin | Siemens Corporation
    Author: Schwager, Mac | Stanford University
    Author: Kochenderfer, Mykel | Stanford University
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems; Intelligent and Flexible Manufacturing

    Abstract : We study the problem of sequential task assignment and collision-free routing for large teams of robots in applications with inter-task precedence constraints (e.g., task A and task B must both be completed before task C may begin). Such problems commonly occur in assembly planning for robotic manufacturing applications, in which sub-assemblies must be completed before they can be combined to form the final product. We propose a hierarchical algorithm for computing makespan-optimal solutions to the problem. The algorithm is evaluated on a set of randomly generated problem instances where robots must transport objects between stations in a ``factory'' grid world environment. In addition, we demonstrate in high-fidelity simulation that the output of our algorithm can be used to generate collision-free trajectories for non-holonomic differential-drive robots.

- Cooperative Multi-Robot Navigation in Dynamic Environment with Deep Reinforcement Learning

    Author: Han, Ruihua | Southern University of Science and Technology
    Author: Chen, Shengduo | Southern University of Science and Technology
    Author: Hao, Qi | Southern University of Science and Technology
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Deep Learning in Robotics and Automation; Multi-Robot Systems

    Abstract : The challenges of multi-robot navigation in dynamic environments lie in uncertainties in obstacle complexities, partially observation of robots, and policy implementation from simulations to the real world. This paper presents a cooperative approach to address the multi-robot navigation problem (MRNP) under dynamic environments using a deep reinforcement learning (DRL) framework, which can help multiple robots jointly achieve optimal paths despite a certain degree of obstacle complexities. The novelty of this work includes threefold: (1) developing a cooperative architecture that robots can exchange information with each other to select the optimal target locations; (2) developing a DRL based framework which can learn a navigation policy to generate the optimal paths for multiple robots; (3) developing a training mechanism based on dynamics randomization which can make the policy generalized and achieve the maximum performance in the real world. The method is tested with Gazebo simulations and 4 differential-driven robots. Both simulation and experiment results validate the superior performance of the proposed method in terms of success rate and travel time when compared with the other state-of-art technologies.

- Adaptive Directional Path Planner for Real-Time, Energy-Efficient, Robust Navigation of Mobile Robots

    Author: Nimmagadda, Mallikarjuna Rao | Intel Corporation
    Author: Dattawadkar, Shreela | Intel
    Author: Muthukumar, Sriram | Intel Corporation
    Author: Honkote, Vinayak | Intel Corporation
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Distributed Robot Systems; Autonomous Agents

    Abstract : Autonomous navigation through unknown and complex environments is a fundamental capability that is essential in almost all robotic applications. Optimal robot path planning is critical to enable efficient navigation. Path planning is a complex, compute and memory intensive task. Traditional methods employ either graph based search methods or sample based methods to implement path planning, which are sub-optimal and compute/memory-intensive. To this end, an Adaptive Directional Planner (ADP) algorithm is devised to achieve real-time, energy-efficient, memory-optimized, robust local path planning for enabling efficient autonomous navigation of mobile robots. The ADP algorithm ensures that the paths are optimal and kinematically-feasible. Further, the proposed algorithm is tested with different challenging scenarios verifying the functionality and robustness. The ADP algorithm implementation results demonstrate 40&#8722;60X less number of nodes and 40 &#8722; 50X less execution time compared to the standard TP-RRT schemes, without compromising on accuracy. Finally, the algorithm has also been implemented as an accelerator for non-holonomic, multi-shape, small form factor mobile robots to provide a silicon solution with high performance and low memory footprint (28KB).

- Distributed State Estimation Using Intermittently Connected Robot Networks (I)

    Author: Khodayi-mehr, Reza | Duke University
    Author: Kantaros, Yiannis | University of Pennsylvania
    Author: Zavlanos, Michael M. | Duke University
 
    keyword: Path Planning for Multiple Mobile Robots or Agents; Formal Methods in Robotics and Automation; Sensor Fusion

    Abstract : We consider the problem of distributed state estimation using multi-robot systems. The robots have limited communication capabilities and only communicate their measurements intermittently, when they are physically close. To decrease the travel distance needed only to communicate, we divide the robots into small teams that communicate at different locations. Then, we propose a new distributed scheme that combines (i) communication schedules that ensure that the network is intermittently connected, and (ii) sampling-based motion planning for the robots in every team to collect optimal measurements and decide on a meeting time and location. This is the first distributed state estimation framework that relaxes all network connectivity assumptions and controls intermittent communication events so that the estimation uncertainty is minimized. Our results show significant improvement in estimation accuracy compared to methods that maintain end-to-end connection for all time.

## Optimization and Optimal Control
- Whole-Body Motion Tracking for a Quadruped-On-Wheel Robot Via a Compact-Form Controller with Improved Prioritized Optimization

    Author: Du, Wenqian | Sorbonne University, ISIR, Paris 6
    Author: Fnadi, Mohamed | ISIR - Sorbonne University
    Author: Ben Amar, Faiz | Université Pierre Et Marie Curie, Paris 6
 
    keyword: Optimization and Optimal Control; Dynamics; Legged Robots

    Abstract : This paper develops a more general dynamics controller to generate whole-body behaviors for a quadruped-on-wheel robot. To track the quadruped centroidal motion, the wheeled motion is achieved by combining the wheel contact constraints and the centroidal momentum/dynamics model. The dynamics controller is based on a new hybrid hierarchical and prioritized weighted optimization framework. We propose one concept of a recursively updated dynamics model and this model enables to integrate the new prioritized weighted scheme in the hierarchical framework. In contrast with the conventional weighted scheme, we propose to use null-space projections among its sub-tasks. Then the prioritized impedance controller is proposed and integrated in our dynamics model, which enables to influence the hierarchical and prioritized weighted tasks in a decoupled way. The task accelerations in the two schemes are extracted with quadratic forms depending on the actuated torque and the prioritized impedance force using null-space based inverse dynamics. The inequality constraints are modified to ensure the compatibility with the hybrid convex optimization. This dynamics controller is more general and its algorithm is given completely which enables our robot to track the centroidal motion on rough terrain and handle other missions in three simulation scenarios.

- Optimal Control of an Energy-Recycling Actuator for Mobile Robotics Applications

    Author: Krimsky, Erez | Stanford University
    Author: Collins, Steven H. | Stanford University
 
    keyword: Optimization and Optimal Control; Force Control; Prosthetics and Exoskeletons

    Abstract : Actuator power consumption is a limiting factor in mobile robot design. In this paper we introduce the concept of an energy-recycling actuator, which uses an array of springs and clutches to capture and return elastic energy in parallel with an electric motor. Engaging and disengaging clutches appropriately could reduce electrical energy consumption without sacrificing controllability, but presents a challenging control problem. We formulated the optimal control objective of minimizing actuator power consumption as a mixed-integer quadratic program (MIQP) and solved for the global minimum. For a given actuator design and a wide range of simulated torque and rotation patterns, all corresponding to zero net work over one cycle, we compared optimized actuator energy consumption to that of an optimized gear motor with simple parallel elasticity. The simulated energy-recycling actuator consumed less electrical energy: 57% less on average and 80% less in the best case. These results demonstrate an effective approach to optimal control of this type of system, and suggest that energy-recycling actuators could substantially reduce power consumption in some robotics applications.

- Real-Time Nonlinear Model Predictive Control of Robots Using a Graphics Processing Unit

    Author: Hyatt, Phillip | Brigham Young University
    Author: Killpack, Marc | Brigham Young University
 
    keyword: Optimization and Optimal Control; Control Architectures and Programming; Deep Learning in Robotics and Automation

    Abstract : In past robotics applications, Model Predictive Control (MPC) has been limited to linear models and relatively short time horizons. In recent years however, research in optimization, optimal control, and simulation has enabled some forms of nonlinear model predictive control which find locally optimal solutions. The limiting factor for applying nonlinear MPC for robotics remains the computation necessary to solve the optimization, especially for complex systems and for long time horizons. This paper presents a new method which addresses computational concerns related to nonlinear MPC called nonlinear Evolutionary MPC (NEMPC), and then compares it to several existing methods. These comparisons include simulations on torque-limited robots performing a swing-up task and demonstrate that NEMPC is able to discover complex behaviors to accomplish the task. Comparisons with state-of-the-art nonlinear MPC algorithms show that NEMPC finds high quality control solutions very quickly using a global, instead of local, optimization. Finally, an application in hardware (a 24 state pneumatically actuated continuum soft robot) demonstrates that this method is tractable for real-time control of high degree of freedom systems.

- An NMPC Approach Using Convex Inner Approximations for Online Motion Planning with Guaranteed Collision Avoidance

    Author: Schoels, Tobias | University of Freiburg
    Author: Palmieri, Luigi | Robert Bosch GmbH
    Author: Arras, Kai Oliver | Bosch Research
    Author: Diehl, Moritz | Univ. of Heidelberg
 
    keyword: Optimization and Optimal Control; Collision Avoidance; Nonholonomic Motion Planning

    Abstract : Even though mobile robots have been around for decades, trajectory optimization and continuous time collision avoidance remain subject of active research. Existing methods trade off between path quality, computational complexity, and kinodynamic feasibility. This work approaches the problem using a nonlinear model predictive control (NMPC) framework, that is based on a novel convex inner approximation of the collision avoidance constraint. The proposed Convex Inner ApprOximation (CIAO) method finds kinodynamically feasible and continuous time collision free trajectories, in few iterations, typically one. For a feasible initialization, the approach is guaranteed to find a feasible solution, i.e. it preserves feasibility. Our experimental evaluation shows that CIAO outperforms state of the art baselines in terms of planning efficiency and path quality. Experiments show that it also efficiently scales to high-dimensional systems. Furthermore real-world experiments demonstrate its capability of unifying trajectory optimization and tracking for safe motion planning in dynamic environments.

- Multi-Contact Heavy Object Pushing with a Centaur-Type Humanoid Robot: Planning and Control for a Real Demonstrator

    Author: Parigi Polverini, Matteo | Istituto Italiano Di Tecnologia (IIT)
    Author: Laurenzi, Arturo | Istituto Italiano Di Tecnologia
    Author: Mingo Hoffman, Enrico | Fondazione Istituto Italiano Di Tecnologia
    Author: Ruscelli, Francesco | Istituto Italiano Di Tecnologia
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
 
    keyword: Optimization and Optimal Control; Humanoid Robots; Mobile Manipulation

    Abstract : Performing a demanding manipulation task with a multi-legged loco-manipulation platform may require the exploitation of multiple external contacts with different environment surfaces for counteracting the manipulation forces. This is the case of the pushing task of a heavy object, where the grip forces at ground may not be adequate and establishing leg contacts against a wall turns out to be an effective solution to the manipulation problem. In order to produce such behaviour, this paper presents a control architecture that is able to freely exploit the environment complexity to perform loco-manipulation actions, e.g. pushing a heavy object, while meeting the implementation requirements to achieve a real demonstrator. The proposed approach, conceived for torque-controlled platforms, combines the planning capabilities of nonlinear optimization over the robot centroidal statics, based on a continuous description of the environment through superquadric functions, with the instantaneous capabilities of hierarchical inverse kinematics and reactive contact force distribution. Experimental validation has been performed on the pushing task of a wooden cabinet loaded with bricks, using the CENTAURO robot developed at Istituto Italiano di Tecnologia (IIT).

- Hierarchical Stochastic Optimization with Application to Parameter Tuning for Electronically Controlled Transmissions

    Author: Karasawa, Hiroyuki | The University of Tokyo
    Author: Kanemaki, Tomohiro | Komatsu Ltd
    Author: Oomae, Kei | Komatsu Ltd
    Author: Fukui, Rui | The University of Tokyo
    Author: Nakao, Masayuki | The University of Tokyo
    Author: Osa, Takayuki | Kyushu Institute of Technology
 
    keyword: Optimization and Optimal Control; AI-Based Methods; Industrial Robots

    Abstract : In mechanical systems, control parameters are often manually tuned by an expert through trial and error, which is labor-intensive and time-consuming. In addition, the difficulty of this problem is that there often exist multiple solutions that provide high returns. As a designed objective function is often not optimal in practice, the solution that provides the highest return may not be the optimal solution. Therefore, it is often necessary to verify the multiple candidates of the solution to identify the one most suitable for the actual system. To address this issue, we propose a parameter optimization system using hierarchical stochastic optimization (HSO) that can handle multimodal objective functions. In a case study of electronically controlled transmissions, the optimizer learns multiple sets of parameters that satisfy all constraints and outperforms the parameters manually designed by human engineers. We demonstrate experimentally that our HSO can identify several modes of the objective function and is more sample-efficient than the existing methods, such as cross-entropy method and covariance matrix adaptation evolution strategy, as well as a human engineer.

- Targeted Drug Delivery: Algorithmic Methods for Collecting a Swarm of Particles with Uniform, External Forces

    Author: Becker, Aaron | University of Houston
    Author: Fekete, S�ndor | Technische Universitét Braunschweig
    Author: Huang, Li | University of Houston
    Author: Keldenich, Phillip | TU Braunschweig
    Author: Kleist, Linda | Technische Universitét Braunschweig
    Author: Krupke, Dominik Michael | TU Braunschweig, IBR, Algorithms Group
    Author: Rieck, Christian | Technische Universitét Braunschweig
    Author: Schmidt, Arne | TU Braunschweig
 
    keyword: Optimization and Optimal Control; Medical Robots and Systems; AI-Based Methods

    Abstract : We investigate algorithmic approaches for targeted drug delivery in a complex, maze-like environment, such as a vascular system. The basic scenario is given by a large swarm of micro-scale particles ("agents") and a particular target region ("tumor") within a system of passageways. Agents are too small to contain on-board power or computation and are instead controlled by a global external force that acts uniformly on all particles, such as an applied fluidic flow or electromagnetic field. The challenge is to deliver all agents to the target region with a minimum number of actuation steps. We provide a number of results for this challenge. We show that the underlying problem is NP-hard, which explains why previous work did not provide provably efficient algorithms. We also develop a number of algorithmic approaches that greatly improve the worst-case guarantees for the number of required actuation steps. We evaluate our algorithmic approaches by a number of simulations, both for deterministic algorithms and searches supported by deep learning, which show that the performance is practically promising.

-  Virtual Point Control Strategy with Power Optimization for Trajectory Planning of Autonomous Mobile Robots

    Author: Merzouki, Rochdi | CRIStAL, CNRS UMR 9189, University of Lille1
    Author: Bensekrane, Ismail | Polytech Lille, University of Lille 1
    Author: Drakunov, Sergey | IHMC

- Enhancing Bilevel Optimization for UAV Time-Optimal Trajectoryusing a Duality Gap Approach

    Author: Tang, Gao | University of Illinois at Urbana-Champaign
    Author: Sun, Weidong | Duke University
    Author: Hauser, Kris | University of Illinois at Urbana-Champaign
 
    keyword: Optimization and Optimal Control; Aerial Systems: Perception and Autonomy; Autonomous Agents

    Abstract : Time-optimal trajectories for dynamic robotic vehicles are difficult to compute even for state-of-the-art nonlinear programming (NLP) solvers, due to nonlinearity and bang-bang control structure. This paper presents a bilevel optimization framework that addresses these problems by decomposing the spatial and temporal variables into a hierarchical optimization. Specifically, the original problem is divided into an inner layer, which computes a time-optimal velocity profile along a given geometric path, and an outer layer, which refines the geometric path by a Quasi-Newton method. The inner optimization is convex and efficiently solved by interior-point methods. The gradients of the outer layer can be analytically obtained using sensitivity analysis of parametric optimization problems. A novel contribution is to introduce a duality gap in the inner optimization rather than solving it to optimality; this lets the optimizer realize warm-starting of the interior-point method, avoids non-smoothness of the outer cost function caused by active inequality constraint switching. Like prior bilevel frameworks, this method is guaranteed to return a feasible solution at any time, but converges faster than gap-free bilevel optimization. Numerical experiments on a drone model with velocity and acceleration limits show that the proposed method performs faster and more robustly than gap-free bilevel optimization and general NLP solvers.

- Constrained Sampling-Based Trajectory Optimization Using StochasticApproximation

    Author: Boutselis, George | Georgia Tech
    Author: Wang, Ziyi | Georgia Institute of Technology
    Author: Theodorou, Evangelos | Georgia Institute of Technology
 
    keyword: Optimization and Optimal Control; Collision Avoidance; Probability and Statistical Methods

    Abstract : We propose a sampling-based trajectory optimiza- tion methodology for constrained problems. We extend recent works on stochastic search to deal with box control constraints, as well as nonlinear state constraints for discrete dynamical systems. Regarding the former, our strategy is to optimize over truncated parameterized distributions on control inputs. Furthermore, we show how non-smooth penalty functions can be incorporated into our framework to handle state constraints. Numerical simulations show that our approach outperforms previous methods on constrained sampling-based optimization, in terms of quality of solutions and sample efficiency.

- Learning Control Policies from Optimal Trajectories

    Author: Zelch, Christoph | Technische Universitét Darmstadt
    Author: Peters, Jan | Technische Universitét Darmstadt
    Author: von Stryk, Oskar | Technische Universitét Darmstadt
 
    keyword: Optimization and Optimal Control; Learning and Adaptive Systems; Probability and Statistical Methods

    Abstract : The ability to optimally control robotic systems offers significant advantages for their performance. While time-dependent optimal trajectories can numerically be computed for high dimensional nonlinear system dynamic models, constraints and objectives, finding optimal feedback control policies for such systems is hard. This is unfortunate, as without a policy, the control of real-world systems requires frequent correction or replanning to compensate for disturbances and model errors. In this paper, a feedback control policy is learned from a set of optimal reference trajectories using Gaussian processes. Information from existing trajectories and the current policy is used to find promising start points for the computation of further optimal trajectories. This aspect is important as it avoids exhaustive sampling of the complete state space, which is impractical due to the high dimensional state space, and to focus on the relevant region. The presented method has been applied in simulation to a swing-up problem of an underactuated pendulum and an energy- minimal point-to-point movement of a 3-DOF industrial robot.

- Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control

    Author: Mastalli, Carlos | University of Edinburgh
    Author: Budhiraja, Rohan | LAAS, CNRS
    Author: Merkt, Wolfgang Xaver | The University of Edinburgh
    Author: Saurel, Guilhem | LAAS-CNRS
    Author: Hammoud, Bilal | Max Planck Institute
    Author: Naveau, Maximilien | LAAS/CNRS
    Author: Carpentier, Justin | INRIA
    Author: Righetti, Ludovic | New York University
    Author: Vijayakumar, Sethu | University of Edinburgh
    Author: Mansard, Nicolas | CNRS
 
    keyword: Optimization and Optimal Control; Legged Robots; Humanoid and Bipedal Locomotion

    Abstract : We introduce Crocoddyl (Contact RObot COntrol by Differential DYnamic Library), an open-source framework tailored for efficient multi-contact optimal control. Crocoddyl efficiently computes the state trajectory and the control policy for a given predefined sequence of contacts. Its efficiency is due to the use of sparse analytical derivatives, exploitation of the problem structure, and data sharing. It employs differential geometry to properly describe the state of any geometrical system, e.g. floating-base systems. Additionally, we propose a novel optimal control algorithm called Feasibility-driven Differential Dynamic Programming (FDDP). Our method does not add extra decision variables which often increases the computation time per iteration due to factorization. FDDP shows a greater globalization strategy compared to classical Differential Dynamic Programming (DDP) algorithms. Concretely, we propose two modifications to the classical DDP algorithm. First, the backward pass accepts infeasible state-control trajectories. Second, the rollout keeps the gaps open during the early "exploratory" iterations (as expected in multiple-shooting methods with only equality constraints). We showcase the performance of our framework using different tasks. With our method, we can compute highly-dynamic maneuvers (e.g. jumping, front-flip) within few milliseconds.

- Path-Following Model Predictive Control of Ballbots

    Author: Jespersen, Thomas Kølbæk | APTIV, Formerly NuTonomy
    Author: Al Ahdab, Mohamad | Aalborg University
    Author: Flores-Mendez, Juan de Dios | Aalborg University
    Author: Damgaard, Malte Rørmose | Aalborg University
    Author: Hansen, Karl Damkjær | Aalborg University
    Author: Pedersen, Rasmus | Aalborg University
    Author: Bak, Thomas | Aalborg University
 
    keyword: Optimization and Optimal Control; Underactuated Robots; Motion and Path Planning

    Abstract : This paper introduces a novel approach for model predictive control of ballbots for path-following tasks. Ballbots are dynamically unstable mobile robots which are designed to balance on a single ball. The model presented in this paper is a simplified version of a full quaternion-based model of ballbots' underactuated dynamics which is suited for online implementation. Furthermore, the approach is extended to handle nearby obstacles directly in the MPC formulation. The presented controller is validated through simulation on a high fidelity model as well as through real-world experiments on a physical ballbot system.

- Underactuated Waypoint Trajectory Optimization for Light Painting Photography

    Author: Eilers, Christian | Technische Universitét Darmstadt
    Author: Eschmann, Jonas | Technische Universitét Darmstadt
    Author: Menzenbach, Robin | Technische Universitét Darmstadt
    Author: Belousov, Boris | Technische Universitét Darmstadt
    Author: Muratore, Fabio | TU Darmstadt
    Author: Peters, Jan | Technische Universitét Darmstadt
 
    keyword: Optimization and Optimal Control; Motion and Path Planning; Underactuated Robots

    Abstract : Despite their abundance in robotics and nature, underactuated systems remain a challenge for control engineering. Trajectory optimization provides a generally applicable solution, however its efficiency strongly depends on the skill of the engineer to frame the problem in an optimizer-friendly way. This paper proposes a procedure that automates such problem reformulation for a class of tasks in which the desired trajectory is specified by a sequence of waypoints. The approach is based on introducing auxiliary optimization variables that represent waypoint activations. To validate the proposed method, a letter drawing task is set up where shapes traced by the tip of a rotary inverted pendulum are visualized using long exposure photography.

- Whole-Body Walking Generation Using Contact Parametrization: A Non-Linear Trajectory Optimization Approach

    Author: Dafarra, Stefano | Istituto Italiano Di Tecnologia
    Author: Romualdi, Giulio | Fondazione Istituto Italiano Di Tecnologia
    Author: Metta, Giorgio | Istituto Italiano Di Tecnologia (IIT)
    Author: Pucci, Daniele | Italian Institute of Technology
 
    keyword: Optimization and Optimal Control; Humanoid and Bipedal Locomotion; Motion and Path Planning

    Abstract : In this paper, we describe a planner capable of generating walking trajectories by using the centroidal dynamics and the full kinematics of a humanoid robot model. The interaction between the robot and the walking surface is modeled explicitly through a novel contact parametrization. The approach is complementarity-free and does not need a predefined contact sequence. By solving an optimal control problem we obtain walking trajectories. In particular, through a set of constraints and dynamic equations, we model the robot in contact with the ground. We describe the objective the robot needs to achieve with a set of tasks. The whole optimal control problem is transcribed into an optimization problem via a Direct Multiple Shooting approach and solved with an off-the-shelf solver. We show that it is possible to achieve walking motions automatically by specifying a minimal set of references, such as a constant desired Center of Mass velocity and a reference point on the ground.

- Controlling Fast Height Variation of an Actively Articulated Wheeled Humanoid Robot Using Center of Mass Trajectory

    Author: Otubela, Moyin | Trinity College Dublin
    Author: McGinn, Conor | Trinity College Dublin
 
    keyword: Optimization and Optimal Control; Dynamics; Kinematics

    Abstract : Hybrid wheel-legged robots have begun to demonstrate the ability to adapt to complex terrain traditionally inaccessible to purely wheeled morphologies. Further research is needed into how their dynamics can be optimally controlled for developing highly adaptive behaviours on challenging terrain. Using optimal center of mass (COM) kinematic trajectories, this work examines the nonlinear dynamics control problem for fast height adaptation on the hybrid humanoid platform known as Aerobot. We explore the dynamics control problem through experimentation with an offline trajectory optimisation (TO) method and a task-space inverse dynamics (TSID) controller for varying the robot's height. Our TO approach uses sequential quadratic programming (SQP) to solve optimal 7th order spline coefficients for the robot's kinematics. The nonlinear Zero Moment Point (ZMP) is used to model a stability criterion that is constrained in the TO problem to ensure dynamic stability. Our TSID controller follows motion plans based on using task jacobians and a simplified passive dynamics model of the Aerobot platform. Results exhibit fast height adaptation on the Aerobot platform with significantly differing results between the control methods that prompts new research into how it may be controlled online.

- Contact-Aware Controller Design for Complementarity Systems

    Author: Aydinoglu, Alp | University of Pennsylvania
    Author: Preciado, Victor | University of Pennsylvania
    Author: Posa, Michael | University of Pennsylvania
 
    keyword: Optimization and Optimal Control; Force Control; Contact Modeling

    Abstract : While many robotic tasks, like manipulation and locomotion, are fundamentally based in making and breaking contact with the environment, state-of-the-art control policies struggle to deal with the hybrid nature of multi-contact motion. Such controllers often rely heavily upon heuristics or, due to the combinatoric structure in the dynamics, are unsuitable for real-time control. Principled deployment of tactile sensors offers a promising mechanism for stable and robust control, but modern approaches often use this data in an ad hoc manner, for instance to guide guarded moves. In this work, by exploiting the complementarity structure of contact dynamics, we propose a control framework which can close the loop on rich, tactile sensors. Critically, this framework is non-combinatoric, enabling optimization algorithms to automatically synthesize provably stable control policies. We demonstrate this approach on three different underactuated, multi-contact robotics problems.

-  Trajectory Optimization of Robots with Regenerative Drive Systems: Numerical and Experimental Results (I)

    Author: Khalaf, Poya | Cleveland State University
    Author: Richter, Hanz | Cleveland State University


- Exploiting Sparsity in Robot Trajectory Optimization with Direct Collocation and Geometric Algorithms

    Author: Cardona-Ortiz, Daniel | Cinvestav
    Author: Alvaro Paz, Alveiro | CINVESTAV
    Author: Arechavaleta, Gustavo | CINVESTAV
 
    keyword: Optimization and Optimal Control; Dynamics

    Abstract : This paper presents a robot trajectory optimization formulation that builds upon numerical optimal control and Lie group methods. In particular, the inherent sparsity of direct collocation is carefully analyzed to dramatically reduce the number of floating-point operations to get first-order information of the problem. We describe how sparsity exploitation is employed with both numerical and analytical differentiation. Furthermore, the use of geometric algorithms based on Lie groups and their associated Lie algebras allow to analytically evaluate the state equations and their derivatives with efficient recursive algorithms. We demonstrate the scalability of the proposed formulation with three different articulated robots, such as a finger, a mobile manipulator and a humanoid composed of five, eight and more than twenty degrees of freedom, respectively. The performance of our implementation in C++ is also validated and compared against a state-of-the-art general purpose numerical optimal control solver.

- Bi-Convex Approximation of Non-Holonomic Trajectory Optimization

    Author: Singh, Arun Kumar | Tampere University of Technology, Finland
    Author: Theerthala, Raghu Ram | International Institute of Information Technology, Hyderabad
    Author: Nallana, Mithun Babu | International Institute of Information Technology, Hyderabad
    Author: Krishnan R Nair, Unni | IIITH
    Author: Krishna, Madhava | IIIT Hyderabad
 
    keyword: Optimization and Optimal Control; Nonholonomic Motion Planning; Autonomous Vehicle Navigation

    Abstract : Autonomous cars and fixed-wing aerial vehicles have the so-called non-holonomic kinematics which non-linearly maps control input to states. As a result, trajectory optimization with such a motion model becomes highly non-linear and non-convex. In this paper, we improve the computational tractability of non-holonomic trajectory optimization by reformulating it in terms of a set of bi-convex cost and constraint functions along with a non-linear penalty. The bi-convex part acts as a relaxation for the non-holonomic trajectory optimization while the residual of the penalty dictates how well its output obeys the non-holonomic behavior. We adopt an alternating minimization approach for solving the reformulated problem and show that it naturally leads to the replacement of the challenging non-linear penalty with a globally valid convex surrogate. Along with the common cost functions modeling goal-reaching, trajectory smoothness, etc., the proposed optimizer can also accommodate a class of non-linear costs for modeling goal-sets, while retaining the bi-convex structure. We benchmark the proposed optimizer against off-the-shelf solvers implementing sequential quadratic programming and interior-point methods and show that it produces solutions with similar or better cost as the former while significantly outperforming the latter. Furthermore, as compared to both off-the-shelf solvers, the proposed optimizer achieves more than 20x reduction in computation time.

- Fast, Versatile, and Open-Loop Stable Running Behaviors with Proprioceptive-Only Sensing Using Model-Based Optimization

    Author: Gao, Wei | Florida State University
    Author: Young, Charles | Florida State University
    Author: Nicholson, John | Florida State University
    Author: Hubicki, Christian | Florida State University
    Author: Clark, Jonathan | Florida State University
 
    keyword: Optimization and Optimal Control; Robust/Adaptive Control of Robotic Systems; Legged Robots

    Abstract : As we build our legged robots smaller and cheaper, stable and agile control without expensive inertial sensors becomes increasingly important. We seek to enable versatile dynamic behaviors on robots with limited modes of state feedback, specifically proprioceptive-only sensing. This work uses model-based trajectory optimization methods to design open-loop stable motion primitives. We specifically design running gaits for a single-legged planar robot, and can generate motion primitives in under 3 seconds, approaching online-capable speeds. A direct-collocation-formulated optimization generated axial force profiles for the direct-drive robot to achieve desired running speed and apex height. When implemented in hardware, these trajectories produced open-loop stable running. Further, the measured running achieved the desired speed within 10% of the speed specified for the optimization in spite of having no control loop actively measuring or controlling running speed. Additionally, we examine the shape of the optimized force profile and observe features that may be applicable to open-loop stable running in general.

- Wasserstein Distributionally Robust Motion Planning and Control with Safety Constraints Using Conditional Value-At-Risk

    Author: Hakobyan, Astghik | Seoul National University
    Author: Yang, Insoon | Seoul National University
 
    keyword: Optimization and Optimal Control; Collision Avoidance; Motion Control

    Abstract : In this paper, we propose an optimization-based decision-making tool for safe motion planning and control in an environment with randomly moving obstacles. The unique feature of the proposed method is that it limits the risk of unsafety by a pre-specified threshold even when the true probability distribution of the obstacles' movements deviates, within a Wasserstein ball, from an available empirical distribution. Another advantage is that it provides a probabilistic out-of-sample performance guarantee of the risk constraint. To develop a computationally tractable method for solving the distributionally robust model predictive control problem, we propose a set of reformulation procedures using (i) the Kantorovich duality principle, (ii) the extremal representation of conditional value-at-risk, and (iii) a geometric expression of the distance to the union of halfspaces. The performance and utility of this distributionally robust method are demonstrated through simulations using a 12D quadrotor model in a 3D environment.

- One Robot for Many Tasks: Versatile Co-Design through Stochastic Programming

    Author: Bravo, Gabriel | University of Notre Dame
    Author: Del Prete, Andrea | Max Planck Institute for Intelligent Systems
    Author: Wensing, Patrick M. | University of Notre Dame
 
    keyword: Optimization and Optimal Control; Legged Robots; Mechanism Design

    Abstract : Versatility is one of the main factors driving the adoption of robots on the assembly line and in other applications. Compared to fixed-automation solutions, a single industrial robot can perform a wide range of tasks (e.g., welding, lifting). In other platforms, such as legged robots, versatility is a necessity to negotiate varied terrains. The ability to balance performance across these anticipated scenarios is one of the main challenges to the process of design. To address this challenge, this paper proposes a new framework for the computational design of versatile robots by considering the interplay between mechanical design and control across multiple tasks and environments. The proposed method optimizes morphology parameters while simultaneously adjusting control parameters using trajectory optimization (TO) so that a single design can fulfill multiple tasks. As its main contribution, the paper details an approach to combine methods from stochastic programming (SP) with TO to address the scalability of these multi-task co-design problems. To assess the effects of this contribution, this paper considers the problems of designing a planar manipulator to transport a range of loads and a hopping monopod robot that must jump across a variety of terrains. The proposed formulation achieves faster solution times and improved scalability in comparison to state of the art co-design solutions.

- Inverse Optimal Control for Multiphase Cost Functions (I)
 
    Author: Jin, Wanxin | Purdue University
    Author: Kulic, Dana | Monash University
    Author: Lin, Jonathan Feng-Shun | University of Waterloo
    Author: Mou, Shaoshuai | Purdue University
    Author: Hirche, Sandra | Technische Universitét M�nchen
 
    keyword: Optimization and Optimal Control; Learning from Demonstration

    Abstract : We consider a dynamical system whose trajectory is a result of minimizing a multiphase cost function. The multiphase cost function is assumed to be a weighted sum of specified features (or basis functions) with phase-dependent weights that switch at some unknown phase transition points. A new inverse optimal control approach for recovering the cost weights of each phase and estimating the phase transition points is proposed. The key idea is to use a length-adapted window moving along the observed trajectory, where the window length is determined by finding the minimal observation length that suffices for a successful cost weight recovery. The effectiveness of the proposed method is first evaluated on a simulated robot arm, and then, demonstrated on a dataset of human participants performing a series of squatting tasks. The results demonstrate that the proposed method reliably retrieves the cost function of each phase and segments each phase of motion from the trajectory with a segmentation accuracy above 90%.
