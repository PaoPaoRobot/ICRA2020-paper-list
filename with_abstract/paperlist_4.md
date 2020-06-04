
# International Conference on Robotics and Automation 2020
 
Welcome to ICRA 2020, the 2020 IEEE International Conference on Robotics and Automation.

This list is edited by [PaopaoRobot, 泡泡机器人](https://github.com/PaoPaoRobot) , the Chinese academic nonprofit organization. Recently we will classify these papers by topics. Welcome to follow our github and our WeChat Public Platform Account ( [paopaorobot_slam](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=100000102&idx=1&sn=0a8a831a4f2c18443dbf436ef5d5ff8c&chksm=6c10bf625b6736748c9612879e166e510f1fe301b72ed5c5d7ecdd0f40726c5d757e975f37af&mpshare=1&scene=1&srcid=0530KxSLjUE9I38yLgfO2nVm&pass_ticket=0aB5tcjeTfmcl9u0eSVzN4Ag4tkpM2RjRFH8DG9vylE%3D#rd) ). Of course, you could contact with [daiwei.song@outlook.com](mailto://daiwei.song@outlook.com)

## Grasping


- Action Image Representation: Learning Deep Grasping Policies with Zero Real World Data

    Author: Khansari, Mohi | Google X
    Author: Kappler, Daniel | X (Google)
    Author: Luo, Jianlan | UC Berkeley
    Author: Bingham, Jeffrey | X
    Author: Kalakrishnan, Mrinal | X
 
    keyword: Deep Learning in Robotics and Automation

    Abstract : This paper introduces Action Image, a new grasp proposal representation that allows learning an end-to-end deep-grasping policy. Our model achieves 84% grasp success on 172 real world objects while being trained only in simulation on 48 objects with just naive domain randomization. Similar to computer vision problems, such as object detection, Action Image builds on the idea that object features are invariant to translation in image space. Therefore, grasp quality is invariant when evaluating the object-gripper relationship; a successful grasp for an object depends on its local context, but is independent of the surrounding environment. Action Image represents a grasp proposal as an image and uses a deep convolutional network to infer grasp quality. We show that by using an Action Image representation, trained networks are able to extract local, salient features of grasping tasks that generalize across different objects and environments. We show that this representation works on a variety of inputs, including color images (RGB), depth images (D), and combined color-depth (RGB-D). Our experimental results demonstrate that networks utilizing an Action Image representation exhibit strong domain transfer between training on simulated data and inference on real-world sensor streams. Finally, our experiments show that a network trained with Action Image improves grasp success (84% vs. 53%) over a baseline model with the same structure, but using actions encoded as vectors.

- High Accuracy and Efficiency Grasp Pose Detection Scheme with Dense Predictions

    Author: Cheng, Hu | The Chinese University of Hong Kong
    Author: Ho, Danny | The Chinese University of Hong Kong
    Author: Meng, Max Q.-H. | The Chinese University of Hong Kong
 
    keyword: Service Robots; Perception for Grasping and Manipulation; Grasping

    Abstract : Learning-based grasp pose detection algorithms have boosted the performance of robot grasping, but they usually need manually fine-tuning steps to find the balance between detection accuracy and efficiency. In this paper, we discard these intermediate procedures, like sampling grasps and generating grasp proposals, and propose an end-to-end grasp pose detection model. Our model uses the RGB image as the input and predicts the single grasp pose in each small grid of the image. Furthermore, the best grasps are found by non-maximum suppression (NMS) strategy. The clustering and ranking procedures are left for NMS while the network only generates dense grasp predictions, which keeps the network simple and efficient. To achieve dense predictions, the predicted grasps of our detection model are represented by the 6-channel images with each pixel location representing a rated grasp. To the best of our knowledge, our model is the first neural network that attaches a grasp pose in pixel level. The model achieves 96:5% accuracy which costs 14ms for prediction of a 480*360 resolution RGB image in Cornell Grasp Dataset, and 90.4% robot grasping success rate for unknown objects with a parallel plate gripper in the real environment.

- Transferable Active Grasping and Real Embodied Dataset

    Author: Chen, Xiangyu | Cornell University
    Author: Ye, Zelin | SJTU
    Author: Sun, Jiankai | The Chinese University of Hong Kong
    Author: Fan, Yuda | Shanghai Jiao Tong University
    Author: Hu, Fang | Shanghai Jiao Tong University
    Author: Wang, Chenxi | Shanghai Jiaotong University
    Author: Lu, Cewu | ShangHai Jiao Tong University
 
    keyword: Deep Learning in Robotics and Automation; Model Learning for Control; Visual Servoing

    Abstract : Grasping in cluttered scenes is challenging for robot vision systems, as detection accuracy can be hindered by partial occlusion of objects. We adopt a reinforcement learning (RL) framework and 3D vision architectures to search for feasible viewpoints for grasping by the use of hand-mounted RGB-D cameras. To overcome the disadvantages of photo-realistic environment simulation, we propose a large-scale dataset called Real Embodied Dataset (RED), which includes full-viewpoint real samples on the upper hemisphere with amodal annotation and enables a simulator that has real visual feedback. Based on this dataset, a practical 3-stage transferable active grasping pipeline is developed, that is adaptive to unseen clutter scenes. In our pipeline, we propose a novel mask-guided reward to overcome the sparse reward issue in grasping and ensure category-irrelevant behavior. The grasping pipeline and its possible variants are evaluated with extensive experiments both in simulation and on a real-world UR-5 robotic arm.

- PointNet++ Grasping: Learning an End-To-End Spatial Grasp Generation Algorithm from Sparse Point Clouds

    Author: Ni, Peiyuan | Shanghai Jiao Tong University
    Author: Zhang, Wenguang | Shanghai Jiao Tong University
    Author: Zhu, Xiaoxiao | SJTU
    Author: Cao, Qixin | Shanghai Jiao Tong University
 
    keyword: Deep Learning in Robotics and Automation; Grasping; Perception for Grasping and Manipulation

    Abstract : Grasping for novel objects is important for robot manipulation in unstructured environments. Most of current works require a grasp sampling process to obtain grasp candidates, combined with local feature extractor using deep learning. This pipeline is timecost, expecially when grasp points are sparse such as at the edge of a bowl.	In this paper, we propose an end-to-end approach to directly predict the poses, categories and scores (qualities) of all the grasps. It takes the whole sparse point clouds as the input and requires no sampling or search process. Moreover, to generate training data of muti-object scene, we propose a fast multi-object grasp detection algorithm based on Ferrari Canny metrics. A single-object dataset (79 objects from YCB object set, 23.7k grasps) and a multi-object dataset (20k point clouds with annotations and masks) are generated. A PointNet++ based network combined with multi-mask loss is introduced to deal with different training points. The whole weight size of our network is only about 11.6M, which takes about 102ms for a whole prediction process using a GeForce 840M GPU. Our experiment shows our work get 71.43% success rate and 91.60% completion rate, which performs better than current state-of-art works.

- UniGrasp: Learning a Unified Model to Grasp with Multifingered Robotic Hands

    Author: Shao, Lin | Stanford University
    Author: Ferreira, Fabio | Karlsruhe Institute of Technology
    Author: Jorda, Mikael | Stanford University
    Author: Nambiar, Varun | Stanford University
    Author: Luo, Jianlan | UC Berkeley
    Author: Solowjow, Eugen | Siemens Corporation
    Author: Aparicio Ojea, Juan | Siemens
    Author: Khatib, Oussama | Stanford University
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Deep Learning in Robotics and Automation; Grasping; Multifingered Hands

    Abstract : To achieve a successful grasp, gripper attributes such as its geometry and kinematics play a role as important as the object geometry. The majority of previous work has focused on developing grasp methods that generalize over novel object geometry but are specific to a certain robot hand. We propose UniGrasp, an efficient data-driven grasp synthesis method that considers both the object geometry and gripper attributes as inputs. UniGrasp is based on a novel deep neural network architecture that selects sets of contact points from the input point cloud of the object. The proposed model is trained on a large dataset to produce contact points that are in force closure and reachable by the robot hand. By using contact points as output, we can transfer between a diverse set of multifingered robotic hands. Our model produces over 90 percent valid contact points in Top10 predictions in simulation and more than 90 percent successful grasps in real world experiments for various known two-fingered and three-fingered grippers. Our model also achieves 93 percent, 83 percent and 90 percent successful grasps in real world experiments for an unseen two-fingered gripper and two unseen multi-fingered anthropomorphic robotic hands.

- Grasp for Stacking Via Deep Reinforcement Learning

    Author: Zhang, Junhao | Shandong University
    Author: Zhang, Wei | Shandong University
    Author: Song, Ran | Shandong University
    Author: Ma, Lin | Tencent
    Author: Li, Yibin | Shandong University
 
    keyword: Grasping; Perception for Grasping and Manipulation; Visual Learning

    Abstract : Integrated robotic arm system should contain both grasp and place actions. However, most grasping methods focus more on how to grasp objects, while ignoring the placement of the grasped objects, which limits their applications in various industrial environments. In this research, we propose a model-free deep Q-learning method to learn the grasping-stacking strategy end-to-end from scratch. Our method maps the images to the actions of the robotic arm through two deep networks: the grasping network (GNet) using the observation of the desk and the pile to infer the gripper's position and orientation for grasping, and the stacking network (SNet) using the observation of the platform to infer the optimal location when placing the grasped object. To make a long-range planning, the two observations are integrated in the grasping for stacking network (GSN). We evaluate the proposed GSN on a grasping-stacking task in both simulated and real-world scenarios.

- CAGE: Context-Aware Grasping Engine

    Author: Liu, Weiyu | Georgia Institute of Technology
    Author: Daruna, Angel | Georgia Institute of Technology, Atlanta, GA 30332
    Author: Chernova, Sonia | Georgia Institute of Technology
 
    keyword: Grasping; Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation

    Abstract : Semantic grasping is the problem of selecting stable grasps that are functionally suitable for specific object manipulation tasks. In order for robots to effectively perform object manipulation, a broad sense of contexts, including object and task constraints, needs to be accounted for. We introduce the Context-Aware Grasping Engine, which combines a novel semantic representation of grasp contexts with a neural network structure based on the Wide &amp; Deep model, capable of capturing complex reasoning patterns. We quantitatively validate our approach against three prior methods on a novel dataset consisting of 14,000 semantic grasps for 44 objects, 7 tasks, and 6 different object states. Our approach outperformed all baselines by statistically significant margins, producing new insights into the importance of balancing memorization and generalization of contexts for semantic grasping. We further demonstrate the effectiveness of our approach on robot experiments in which the presented model successfully achieved 31 of 32 suitable grasps. The code and data are available at: https://github.com/wliu88/rail_semantic_grasping

- Time Optimal Motion Planning and Admittance Control for Cooperative Grasping

    Author: Kaserer, Dominik | Johannes Kepler University Linz (JKU)
    Author: Gattringer, Hubert | Johannes Kepler University Linz
    Author: Mueller, Andreas | Johannes Kepler University
 
    keyword: Grasping; Dual Arm Manipulation; Compliance and Impedance Control

    Abstract : Cooperative grasping refers to the situation when an object is manipulated by multiple robots and the grasp is achieved by the unilateral contact between the robots and the object. This is different from the cooperation of multiple robots where each robot rigidly grasps the object. Motion planning of cooperative grasping tasks involves active force control of the interaction wrench in order to ensure stable grasp. This becomes particularly challenging when aiming at time optimal motions. It is crucial that the trajectories are continuous up to third-order, in order to satisfy velocity, acceleration, and jerk as well as torque limits of the robots. A solution approach is presented for the time optimal path following of two robots performing cooperative grasping tasks. The time optimal path is determined with a dynamic programing method. An admittance control scheme in task space is proposed and used to generate the contact wrench. The method is applicable to grasping of general objects that are in surface contact with the robot.

- Jamming-Free Immobilizing Grasps Using Dual-Friction Robotic Fingertips

    Author: Golan, Yoav | Ben Gurion University
    Author: Shapiro, Amir | Ben Gurion University of the Negev
    Author: Rimon, Elon | Technion - Israel Institute of Technology
 
    keyword: Grasping; Mechanism Design; Contact Modeling

    Abstract : Successful grasping of objects with robotic hands is still considered a difficult task. One aspect of the grasping problem is the physical contact interaction between the robotic fingertips and the object. Friction at the fingertip contacts can improve grasp robustness, but frictional fingertips may be difficult to precisely place on the object's perimeter. This paper describes a novel fingertip design that can switch from frictionless to frictional modes. The transformation from frictionless to frictional contact is achieved passively by the finger force exerted on the object at the target grasp. A novel swivel mechanism ensures that the force magnitude required to switch friction states is independent on the grasped object's contact normal direction, ensuring robustness. Analysis of the displacement and eventual sliding of the fingertip contacts in response to external torque is presented, taking into account the friction and compliant behavior of the fingertip mechanism. Experiments validate the analytic model and demonstrate the fingertip's ability to change friction modes by the applied force magnitude irrespective of the contact normal direction. In line with the analytic model predictions, the experiments show that when converted to frictional contacts, the fingertips provide a more robust and hence secure grasp in the presence of external disturbances. The robustness of the fingertips is further validated by real-world demonstrations shown in an external video.

- Force-Guided High-Precision Grasping Control of Fragile and Deformable Objects Using sEMG-Based Force Prediction

    Author: Wen, Ruoshi | Harbin Institute of Technology
    Author: Yuan, Kai | University of Edinburgh
    Author: Wang, Qiang | Harbin Institute of Technology
    Author: Heng, Shuai | Harbin Institute of Technology
    Author: Li, Zhibin | University of Edinburgh
 
    keyword: Grasping; Dexterous Manipulation; Human Factors and Human-in-the-Loop

    Abstract : Regulating contact forces with high precision is crucial for grasping and manipulating fragile or deformable objects. We aim to utilize the dexterity of human hands to regulate the contact forces for robotic hands and exploit human sensory-motor synergies in a wearable and non-invasive way. We extracted force information from the electric activities of skeletal muscles during their voluntary contractions through surface electromyography (sEMG). We built a regression model based on a Neural Network to predict the gripping force from the preprocessed sEMG signals and achieved high accuracy (R<sup>2</sup> = 0.982). Based on the force command predicted from human muscles, we developed a force-guided control framework, where force control was realized via an admittance controller that tracked the predicted gripping force reference to grasp delicate and deformable objects. We demonstrated the effectiveness of the proposed method on a set of representative fragile and deformable objects from daily life, all of which were successfully grasped without any damage or deformation.

- Grasp It Like a Pro: Grasp of Unknown Objects with Robotic Hands Based on Skilled Human Expertise

    Author: Gabellieri, Chiara | University of Pisa
    Author: Angelini, Franco | University of Pisa
    Author: Arapi, Visar | Centro E. Piaggio
    Author: Palleschi, Alessandro | University of Pisa
    Author: Catalano, Manuel Giuseppe | Istituto Italiano Di Tecnologia
    Author: Grioli, Giorgio | Istituto Italiano Di Tecnologia
    Author: Pallottino, Lucia | Université Di Pisa
    Author: Bicchi, Antonio | Université Di Pisa
    Author: Bianchi, Matteo | University of Pisa
    Author: Garabini, Manolo | Université Di Pisa
 
    keyword: Grasping; Perception for Grasping and Manipulation

    Abstract : This work proposes a method to grasp unknown objects with robotic hands based on demonstrations by a skilled human operator. We observed that humans not only are obviously better at grasping with their own hands than robots, but are also when using the same hardware hand as the robot, provided they train for some time. We therefore consider how the grasping skills of a human trained in robot hand use can be transferred to a robot using the same physical hand. The method we propose is that a skilled human user manually operates the robotic hand to grasp a number of elementary objects, consisting in different boxes. A Decision Tree Regressor is trained on the data acquired from the human operator so to generate hand poses able to grasp a general box. This is extended to grasp unknown objects leveraging upon the state of the art Minimum Volume Bounding Box (MVBB) decomposition algorithm that approximates with a number of boxes the shape of an unknown object, based on its point cloud. We report on extensive tests of the proposed approach on a Panda manipulator equipped with a Pisa/IIT SoftHand, achieving a success rate of 86.7% over 105 grasps of 21 different objects.

- Learning to Generate 6-DoF Grasp Poses with Reachability Awareness

    Author: Lou, Xibai | University of Minnesota Twin Cities
    Author: Yang, Yang | University of Minnesota
    Author: Choi, Changhyun | University of Minnesota, Twin Cities
 
    keyword: Grasping; Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation

    Abstract : Motivated by the stringent requirements of unstructured real-world where a plethora of unknown objects reside in arbitrary locations of the surface, we propose a voxel-based deep 3D Convolutional Neural Network (3D CNN) that generates feasible 6-DoF grasp poses in unrestricted workspace with reachability awareness. Unlike the majority of works that predict if a proposed grasp pose within the restricted workspace will be successful solely based on grasp pose stability, our approach further learns a reachability predictor that evaluates if the grasp pose is reachable or not from robot's own experience. To avoid the laborious real training data collection, we exploit the power of simulation to train our networks on a large-scale synthetic dataset. To our best knowledge, this work is the first attempt to take into account the reachability when proposing feasible grasp poses. Experimental results in both simulation and real-world demonstrate that our reachability aware 3D CNN grasping outperforms several other approaches and achieves 82.5% grasping success rate on unknown objects.

- Enhancing Grasp Pose Computation in Gripper Workspace Spheres

    Author: Sorour, Mohamed | University of Lincoln
    Author: Elgeneidy, Khaled | University of Lincoln
    Author: Hanheide, Marc | University of Lincoln
    Author: Abdalmjed, Mohamed | Ain Shams University
    Author: Srinivasan, Aravinda Ramakrishnan | University of Lincoln, UK
    Author: Neumann, Gerhard | University of Lincoln
 
    keyword: Grasping; Grippers and Other End-Effectors; Perception for Grasping and Manipulation

    Abstract : In this paper, enhancement to the novel grasp planning algorithm based on gripper workspace spheres is presented. Our development requires a registered point cloud of the target from different views, assuming no prior knowledge of the object, nor any of its properties. This work features a new set of metrics for grasp pose candidates evaluation, as well as exploring the impact of high object sampling on grasp success rates. In addition to gripper position sampling, we now perform orientation sampling about the x, y, and z-axes, hence the grasping algorithm no longer require object orientation estimation. Successful experiments have been conducted on a simple jaw gripper (Franka Panda gripper) as well as a complex, high Degree of Freedom (DoF) hand (Allegro hand) as a proof of its versatility. Higher grasp success rates of 76% and 85.5% respectively has been reported by real world experiments.

- Minimal Work: A Grasp Quality Metric for Deformable Hollow Objects

    Author: Xu, Jingyi | Technical University of Munich
    Author: Danielczuk, Michael | UC Berkeley
    Author: Ichnowski, Jeffrey | University of North Carolina at Chapel Hill
    Author: Mahler, Jeffrey | University of California, Berkeley
    Author: Steinbach, Eckehard | Technical University of Munich
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Grasping; Manipulation Planning; Task Planning

    Abstract : Robot grasping of deformable hollow objects such as plastic bottles and cups is challenging, as the grasp should resist disturbances while minimally deforming the object so as not to damage it or dislodge liquids. We propose minimal work as a novel grasp quality metric that combines wrench resistance and object deformation. We introduce an efficient algorithm to compute the work required to resist an external wrench for a manipulation task by solving a linear program. The algorithm first computes the minimum required grasp force and an estimation of the gripper jaw displacements based on the object's empirical stiffness at different locations. The work done by the jaws is the product of the grasp force and the displacements. Grasps requiring minimal work are considered to be of high quality. We collect 460 physical grasps with a UR5 robot and a Robotiq gripper. We consider a grasp to be successful if it completes the task without damaging the object or dislodging the content. Physical experiments suggest that the minimal work quality metric reaches 74.2% balanced accuracy, a metric that is the raw accuracy normalized by the number of successful and failed real-world grasps, and is up to 24.2% higher than classical wrench-based quality metrics.

- Hierarchical 6-DoF Grasping with Approaching Direction Selection

    Author: Choi, Yunho | Seoul National University
    Author: Kee, Hogun | Seoul National University
    Author: Lee, Kyungjae | Seoul National University
    Author: Choy, JaeGoo | Seoul National University
    Author: Min, Junhong | Samsung Electronics
    Author: Lee, Sohee | Technische Universitét M�nchen
    Author: Oh, Songhwai | Seoul National University
 
    keyword: Grasping; Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation

    Abstract : In this paper, we tackle the problem of 6-DoF grasp detection which is crucial for robot grasping in cluttered real-world scenes. Unlike existing approaches which synthesize 6-DoF grasp data sets and train grasp quality networks with input grasp representations based on point clouds, we rather take a novel hierarchical approach which does not use any 6-DoF grasp data. We cast the 6-DoF grasp detection problem as a robot arm approaching direction selection problem using the existing 4-DoF grasp detection algorithm, by exploiting a fully convolutional grasp quality network for evaluating the quality of an approaching direction. To select the best approaching direction with the highest grasp quality, we propose an approaching direction selection method which leverages a geometry-based prior and a derivative-free optimization method. Specifically, we optimize the direction iteratively using the cross entropy method with initial samples of surface normal directions. Our algorithm efficiently finds diverse 6-DoF grasps by the novel way of evaluating and optimizing approaching directions. We validate that the proposed method outperforms other selection methods in scenarios with cluttered objects in a physics-based simulator. Finally, we show that our method outperforms the state-of-the-art grasp detection method in real-world experiments with robots.

- Geometric Characterization of Two-Finger Basket Grasps of 2-D Objects: Contact Space Formulation

    Author: Rimon, Elon | Technion - Israel Institute of Technology
    Author: Pokorny, Florian T. | KTH Royal Institute of Technology
    Author: Wan, Weiwei | Osaka University
 
    keyword: Grasping; Multifingered Hands

    Abstract : This paper considers basket grasps, where a two- finger robot hand forms a basket that can safely lift and carry rigid objects in a 2-D gravitational environment. The two-finger basket grasps form special points in a high-dimensional configuration space of the object and two-finger robot hand. This paper establishes that all two- finger basket grasps can be found in a low-dimensional contact space that parametrizes the two-finger contacts along the supported object boundary. Using contact space, each basket grasp is associated with its depth that provides a security measure while carrying the object, as well as its safety margin away from a critical finger opening where the object drops-off into its intended destination. Geometric techniques that compute the depth and drop-off finger opening are described and illustrated with detailed graphical and numerical examples.

- A Multi-Level Optimization Framework for Simultaneous Grasping and Motion Planning

    Author: Zimmermann, Simon | ETH Zurich
    Author: Hakimifard, Ghazal | ETH Zurich
    Author: Zamora, Miguel | ETH Zurich
    Author: Poranne, Roi | ETHZ
    Author: Coros, Stelian | Carnegie Mellon University
 
    keyword: Grasping; Motion and Path Planning; Optimization and Optimal Control

    Abstract : We present an optimization framework for grasp and motion planning in the context of robotic assembly. Typically, grasping locations are provided by higher level planners or as input parameters. In contrast, our mathematical model simultaneously optimizes motion trajectories, grasping locations, and other parameters such as the position of an object during handover operations. The input to our framework consists of a set of objects placed in a known configuration, their target locations, and relative timing information describing when objects need to be picked up, optionally handed over, and dropped off. To allow robots to reason about the way in which grasping locations govern optimal motions, we formulate the problem using a multi-level optimization scheme: the top level optimizes grasping locations; the mid-layer level computes the configurations of the robot for pick, drop and handover states; and the bottom level computes optimal, collision-free motions. We leverage sensitivity analysis to compute derivatives analytically (how do grasping parameters affect IK solutions, and how these, in turn, affect motion trajectories etc.), and devise an efficient numerical solver to generate solutions to the resulting optimization problem. We demonstrate the efficacy of our approach on a variety of assembly and handover tasks performed by a dual-armed robot with parallel grippers.


- Grasping Fragile Objects Using a Stress-Minimization Metric

    Author: Pan, Zherong | The University of North Carolina at Chapel Hill
    Author: Gao, Xifeng | Florida State University
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Grasping

    Abstract : We present a new method to generate optimal grasps for brittle and fragile objects using a novel stress-minimization (SM) metric. Our approach is designed for objects that are composed of homogeneous isotopic materials. Our SM metric measures the maximal resistible external wrenches that would not result in fractures in the target objects. In this paper, we propose methods to compute our new metric. We also use our SM metric to design optimal grasp planning algorithms. Finally, we compare the performance of our metric and conventional grasp metrics, including Q_1, Q_inf, Q_G11, Q_MSV, Q_VEW. Our experiments show that our SM metric takes into account the material characteristics and object shapes to indicate the fragile regions, where prior methods may not work well. We also show that the computational cost of our SM metric is on par with prior methods. Finally, we show that grasp planners guided by our metric can lower the probability of breaking target objects.

- Grasp Control for Enhancing Dexterity of Parallel Grippers

    Author: Costanzo, Marco | Université Degli Studi Della Campania Luigi Vanvitelli
    Author: De Maria, Giuseppe | Université Degli Studi Della Campania Luigi Vanvitelli
    Author: Lettera, Gaetano | Université Degli Studi Della Campania Luigi Vanvitelli
    Author: Natale, Ciro | Université Degli Studi Della Campania "Luigi Vanvitelli"
 
    keyword: Grasping; Perception for Grasping and Manipulation; Manipulation Planning

    Abstract : A robust grasp controller for both slipping avoidance and controlled sliding is proposed based on force/tactile feedback only. The model-based algorithm exploits a modified LuGre friction model to consider both translational and rotational frictional sliding motions. The modification relies on the Limit Surface concept where a novel computationally efficient method is introduced to compute in real-time the minimum grasping force to balance tangential and torsional loads. The two control modalities are considered by the robot motion planning algorithm that automatically generates robot motions and gripper commands to solve complex manipulation tasks in a material handling application.

- Theoretical Derivation and Realization of Adaptive Grasping Based on Rotational Incipient Slip Detection

    Author: Narita, Tetsuya | Sony Corporation
    Author: Nagakari, Satoko | Sony Corporation
    Author: Conus, William | Sony Corporation
    Author: Tsuboi, Toshimitsu | Sony Corporation
    Author: Nagasaka, Kenichiro | Sony Corporation
 
    keyword: Grasping; Force and Tactile Sensing; Mobile Manipulation

    Abstract : Manipulating objects whose physical properties are unknown remains one of the greatest challenges in robotics. Controlling grasp force is an essential aspect of handling unknown objects without slipping or crushing them. Although extensive research has been carried out on grasp force control, unknown object manipulation is still difficult because conventional approaches assume that object properties (mass, center of gravity, friction coefficient, etc.) are known for grasp force control. One of the approaches to address this issue is incipient slip detection. However, there has been few detailed investigations of robust detection and control of incipient slip on rotational case. This study makes contributions on deriving the theoretical model of incipient slip and proposes a new algorithm to detect incipient slip. Additionally, a novel sensor configuration and a grasp force control algorithm based on the derived theoretical model are proposed. Finally, the proposed algorithm is evaluated by grasping objects with different weights and moments including a fragile pastry (�clair).

- Grasp State Assessment of Deformable Objects Using Visual-Tactile Fusion Perception

    Author: Cui, Shaowei | Institute of Automation, Chinese Academy of Sciences
    Author: Wang, Rui | Institute of Automation, Chinese Academy of Sciences
    Author: Wei, Junhang | Institute of Automation, Chinese Academy of Sciences
    Author: Li, Fanrong | Institute of Automation, Chinese Academy of Sciences
    Author: Wang, Shuo | Chinese Academy of Sciences
 
    keyword: Grasping; Force and Tactile Sensing; Sensor Fusion

    Abstract : Humans can quickly determine the force required to grasp a deformable object to prevent its sliding or excessive deformation through vision and touch, which is still a challenging task for robots. To address this issue, we propose a novel 3D convolution-based visual-tactile fusion deep neural network (C3D-VTFN) to evaluate the grasp state of various deformable objects in this paper. Specifically, we divide the grasp states of deformable objects into three categories of sliding, appropriate and excessive. Also, a dataset for training and testing the proposed network is built by extensive grasping and lifting experiments with different widths and forces on 16 various deformable objects with a robotic arm equipped with a wrist camera and a tactile sensor. As a result, a classification accuracy as high as 99.97% is achieved. Furthermore, some delicate grasp experiments based on the proposed network are implemented in this paper. The experimental results demonstrate that the C3D-VTFN is accurate and efficient enough for grasp state assessment, which can be widely applied to automatic force control, adaptive grasping, and other visual-tactile spatiotemporal sequence learning problems.

- Beyond Top-Grasps through Scene Completion

    Author: Lundell, Jens | Aalto University
    Author: Verdoja, Francesco | Aalto University
    Author: Kyrki, Ville | Aalto University
 
    keyword: Grasping; Perception for Grasping and Manipulation; Deep Learning in Robotics and Automation

    Abstract : Current end-to-end grasp planning methods propose grasps in the order of seconds that attain high grasp success rates on a diverse set of objects, but often by constraining the workspace to top-grasps. In this work, we present a method that allows end-to-end top-grasp planning methods to generate full six-degree-of-freedom grasps using a single RGB-D view as input. This is achieved by estimating the complete shape of the object to be grasped, then simulating different viewpoints of the object, passing the simulated viewpoints to an end-to-end grasp generation method, and finally executing the overall best grasp. The method was experimentally validated on a Franka Emika Panda by comparing 429 grasps generated by the state-of-the-art Fully Convolutional Grasp Quality CNN, both on simulated and real camera images. The results show statistically significant improvements in terms of grasp success rate when using simulated images over real camera images, especially when the real camera viewpoint is angled. Code and video are available at https://irobotics.aalto.fi/beyond-top-grasps-through-scene-completion/.

- Dex-Net AR: Distributed Deep Grasp Planning Using a Commodity Cellphone and Augmented Reality App

    Author: Zhang, Harry Haolun | UC Berkeley
    Author: Ichnowski, Jeffrey | University of North Carolina at Chapel Hill
    Author: Avigal, Yahav | UC Berkeley
    Author: Gonzalez, Joseph E. | UC Berkeley
    Author: Stoica, Ion | UC Berkeley
    Author: Goldberg, Ken | UC Berkeley
 
    keyword: Grasping; Perception for Grasping and Manipulation; Computer Vision for Automation

    Abstract : Consumer demand for augmented reality (AR) in mobile phone applications, such as the Apple ARKit. Such applications have potential to expand access to robot grasp planning systems such as Dex-Net. AR apps use structure from motion methods to compute a point cloud from a sequence of RGB images taken by the camera as it is moved around an object. However, the resulting point clouds are often noisy due to estimation errors. We present a distributed pipeline, Dex-Net AR, that allows point clouds to be uploaded to a server in our lab, cleaned, and evaluated by Dex-Net grasp planner to generate a grasp axis that is returned and displayed as an overlay on the object. We implement Dex-Net AR using the iPhone and ARKit and compare results with those generated with high-performance depth sensors. The success rates with AR on harder adversarial objects are higher than traditional depth images. The server URL is https://sites.google.com/berkeley.edu/dex-net-ar/home

## Omnidirectional Vision
- OmniSLAM: Omnidirectional Localization and Dense Mapping for Wide-Baseline Multi-Camera Systems

    Author: Won, Changhee | Hanyang University
    Author: Seok, Hochang | Hanyang Univ
    Author: Cui, Zhaopeng | ETH Zurich
    Author: Pollefeys, Marc | ETH Zurich
    Author: Lim, Jongwoo | Hanyang University
 
    keyword: Omnidirectional Vision; SLAM; Mapping

    Abstract : In this paper, we present an omnidirectional localization and dense mapping system for a wide-baseline multiview stereo setup with ultra-wide field-of-view (FOV) fisheye cameras, which has a 360&#9702; coverage of stereo observations of the environment. For more practical and accurate reconstruction, we first introduce improved and light-weighted deep neural networks for the omnidirectional depth estimation, which are faster and more accurate than the existing networks. Second, we integrate our omnidirectional depth estimates into the visual odometry (VO) and add a loop closing module for global consistency. Using the estimated depth map, we reproject keypoints onto each other view, which leads to better and more efficient feature matching process. Finally, we fuse the omnidirectional depth maps and the estimated rig poses into the truncated signed distance function (TSDF) volume to acquire a 3D map. We evaluate our method on synthetic datasets with ground-truth and real-world sequences of challenging environments, and the extensive experiments show that the proposed system generates excellent reconstruction results in both synthetic and real-world environments.

- What's in My Room? Object Recognition on Indoor Panoramic Images

    Author: Guerrero-Viu, Julia | University of Zaragoza
    Author: Fernandez-Labrador, Clara | University of Zaragoza
    Author: Demonceaux, C�dric | Université Bourgogne Franche-Comt�
    Author: Guerrero, Josechu | Universidad De Zaragoza
 
    keyword: Omnidirectional Vision; Object Detection, Segmentation and Categorization; Semantic Scene Understanding

    Abstract : In the last few years, there has been a growing interest in taking advantage of the 360� panoramic images potential, while managing the new challenges they imply. While several tasks have been improved thanks to the contextual information these images offer, object recognition in indoor scenes still remains a challenging problem that has not been deeply investigated. This paper provides an object recognition system that performs object detection and semantic segmentation tasks by using a deep learning model adapted to match the nature of equirectangular images. From these results, instance segmentation masks are recovered, refined and transformed into 3D bounding boxes that are placed into the 3D model of the room. Quantitative and qualitative results support that our method outperforms the state of the art by a large margin and show a complete understanding of the main objects in indoor scenes.

- FisheyeDistanceNet: Self-Supervised Scale-Aware Distance Estimation Using Monocular Fisheye Camera for Autonomous Driving

    Author: Ravi Kumar, Varun | Valeo
    Author: Athni Hiremath, Sandesh | Valeo Schalter Und Sensoren
    Author: Bach, Markus | Valeo
    Author: Milz, Stefan | Valeo Schalter Und Sensoren GmbH
    Author: Witt, Christian | Valeo
    Author: Pinard, Cl�ment | Ensta Paris
    Author: Yogamani, Senthil | Home
    Author: M�der, Patrick | Technische Universitét Ilmenau
 
    keyword: Omnidirectional Vision; Computer Vision for Transportation

    Abstract : Fisheye cameras are commonly used in applications like autonomous driving and surveillance to provide a large field of view greater tha 180degrees. However, they come at the cost of strong non-linear distortion which require more complex algorithms. In this paper, we explore Euclidean distance estimation on fisheye cameras for automotive scenes. Obtaining accurate and dense depth supervision is difficult in practice, but self-supervised learning approaches show promising results and could potentially overcome the problem. We present a novel self-supervised scale-aware framework for learning Euclidean distance and ego-motion from raw monocular fisheye videos without applying rectification. While it is possible to perform piece-wise linear approximation of fisheye projection surface and apply standard rectilinear models, it has its own set of issues like re-sampling distortion and discontinuities in transition region. To encourage further research in this area, we will release this dataset as part of our WoodScape dataset. We further evaluated the proposed algorithm on the KITTI dataset and obtained state-of-the-art results comparable to other self-supervised monocular methods.

- 360SD-Net: 360° Stereo Depth Estimation with Learnable Cost Volume

    Author: Wang, Ning-Hsu | National Tsing Hua University
    Author: Solarte, Bolivar | National Tsing Hua University
    Author: Tsai, Yi-Hsuan | NEC Labs America
    Author: Chiu, Wei-Chen | National Chiao Tung University
    Author: Sun, Min | National Tsing Hua University
 
    keyword: Omnidirectional Vision; AI-Based Methods; Visual Learning

    Abstract : Recently, end-to-end trainable deep neural networks have significantly improved stereo depth estimation for perspective images. However, 360� images captured under equirectangular projection cannot benefit from directly adopting existing methods due to distortion introduced (i.e., lines in 3D are not projected onto lines in 2D). To tackle this issue, we present a novel architecture specifically designed for spherical disparity using the setting of top-bottom 360� camera pairs. Moreover, we propose to mitigate the distortion issue by (1) an additional input branch capturing the position and relation of each pixel in the spherical coordinate, and (2) a cost volume built upon a learnable shifting filter. Due to the lack of 360� stereo data, we collect two 360� stereo datasets from Matterport3D and Stanford3D for training and evaluation. Extensive experiments and ablation study are provided to validate our method against existing algorithms. Finally, we show promising results on real-world environments capturing images with two consumer-level cameras. Our project page is at https://albert100121.github.io/360SD-Net-Project-Page.

- Omnidirectional Depth Extension Networks
 
    Author: Cheng, Xinjing | Baidu
    Author: Wang, Peng | Bytedance USA LLC
    Author: Zhou, Yanqi | Google
    Author: Guan, Chenye | Baidu
    Author: Yang, Ruigang | University of Kentucky
 
    keyword: Omnidirectional Vision; RGB-D Perception; Sensor Fusion

    Abstract : Omnidirectional 360&#9702; camera proliferates rapidly for autonomous robots since it significantly enhances the perception ability by widening the field of view (FoV). However, corresponding 360&#9702; depth sensors, which are also critical for the perception system, are still difficult or expensive to have. In this paper, we propose a low-cost 3D sensing system that combines an omnidirectional camera with a calibrated projective depth camera, where the depth from the limited FoV can be automatically extended to the rest of recorded omnidirectional image. To accurately recover the missing depths, we design an omnidirectional depth extension convolutional neural network (ODE-CNN), in which a spherical feature transform layer (SFTL) is embedded at the end of feature encoding layers, and a deformable convolutional spatial propagation network (D-CSPN) is appended at the end of feature decoding layers. The former re-samples the neighborhood of each pixel in the omnidirectional coordination to the projective coordination, which reduce the difficulty of feature learning, and the later automatically finds a proper context to well align the structures in the estimated depths via CNN w.r.t. the reference image, which significantly improves the visual quality. Finally, we demonstrate the effectiveness of proposed ODE-CNN over the popular 360D dataset, and show that ODE-CNN significantly outperforms (relatively 33% reduction in depth error) other state-of-the-art (SoTA) methods.

- 3D Orientation Estimation and Vanishing Point Extraction from Single Panoramas Using Convolutional Neural Network

    Author: Shi, Yongjie | Peking University
    Author: Tong, Xin | Peking University
    Author: Wen, Jingsi | Peking University
    Author: Zhao, He | Peking University
    Author: Ying, Xianghua | Peking University
    Author: Zha, Hongbin | Peking University
 
    keyword: Omnidirectional Vision; Calibration and Identification; Computer Vision for Automation

    Abstract : 3D orientation estimation is a key component of many important computer vision tasks such as autonomous navigation and 3D scene understanding. This paper presents a new CNN architecture to estimate the 3D orientation of an omnidirectional camera with respect to the world coordinate system from a single spherical panorama. To train the proposed architecture, we leverage a dataset of panoramas named VOP60K from Google Street View with labeled 3D orientation, including 50 thousand panoramas for training and 10 thousand panoramas for testing. Previous approaches usually estimate 3D orientation under pinhole cameras. However, for a panorama, due to its larger field of view, previous approaches cannot be suitable. In this paper, we propose an edge extractor layer to utilize the low-level and geometric information of panorama, an attention module to fuse different features generated by previous layers. A regression loss for two column vectors of the rotation matrix and classification loss for the position of vanishing points are added to optimize our network simultaneously. The proposed algorithm is validated on our benchmark, and experimental results clearly demonstrate that it outperforms previous methods.

## Force and Tactile Sensing



- Low-Cost GelSight with UV Markings: Feature Extraction of Objects Using AlexNet and Optical Flow without 3D Image Reconstruction

    Author: Abad, Alexander | Liverpool Hope University
    Author: Ranasinghe, Anuradha | Liverpool Hope University
 
    keyword: Haptics and Haptic Interfaces; Force and Tactile Sensing

    Abstract : GelSight sensor has been used to study microgeometry of objects since 2009 in tactile sensing applications. Elastomer, reflective coating, lighting, and camera were the main challenges of making a GelSight sensor within a short period. The recent addition of permanent markers to the GelSight was a new era in shear/slip studies. In our previous studies, we introduced Ultraviolet (UV) ink and UV LEDs as a new form of marker and lighting respectively. UV ink markers are invisible using ordinary LED but can be made visible using UV LED. Currently, recognition of objects or surface textures using GelSight sensor is done using fusion of camera-only images and GelSight captured images with permanent markings. Those images are fed to Convolutional Neural Networks (CNN) to classify objects. However, our novel approach in using low-cost GelSight sensor with UV markings, the 3D height map to 2D image conversion, and the additional non-Gelsight captured images for training the CNN can be eliminated. AlexNet and optical flow algorithm have been used for feature recognition of five coins without UV markings and shear/slip of the coin in GelSight with UV markings respectively. Our results on confusion matrix show that, on average coin recognition can reach 93.4% without UV markings using AlexNet. Therefore, our novel method of using GelSight with UV markings would be useful to recognize full/partial object, shear/slip, and force applied to the objects without any 3D image reconstruction.

- Evaluation of Non-Collocated Force Feedback Driven by Signal-Independent Noise

    Author: Chua, Zonghe | Stanford University
    Author: Okamura, Allison M. | Stanford University
    Author: Deo, Darrel | Stanford University
 
    keyword: Haptics and Haptic Interfaces; Prosthetics and Exoskeletons; Brain-Machine Interface

    Abstract : Individuals living with paralysis or amputation can operate robotic prostheses using input signals based on their intent or attempt to move. Because sensory function is lost or diminished in these individuals, haptic feedback must be non-collocated. The intracortical brain computer interface (iBCI) has enabled a variety of neural prostheses for people with paralysis. An important attribute of the iBCI is that its input signal contains signal-independent noise. To understand the effects of signal-independent noise on a system with non-collocated haptic feedback and inform iBCI-based prostheses control strategies, we conducted an experiment with a conventional haptic interface as a proxy for the iBCI. Able-bodied users were tasked with locating an indentation within a virtual environment using input from their right hand. Non-collocated haptic feedback of the interaction forces in the virtual environment was augmented with noise of three different magnitudes and simultaneously rendered on users' left hands. We found increases in distance error of the guess of the indentation location, mean time per trial, mean peak absolute displacement and speed of tool movements during localization for the highest noise level compared to the other two levels. The findings suggest that users have a threshold of disturbance rejection and that they attempt to increase their signal-to-noise ratio through their exploratory actions.

- Vibration-Based Multi-Axis Force Sensing: Design, Characterization, and Modeling

    Author: Kuang, Winnie | UCSD
    Author: Yip, Michael C. | University of California, San Diego
    Author: Zhang, Jun | University of Nevada Reno
 
    keyword: Haptics and Haptic Interfaces

    Abstract : It is strongly desirable but challenging to obtain force sensing mechanisms that are low-cost, volumetrically compact, away from contact location, and can be easily integrated into existing and emerging robot systems. For example, having a bulky force sensor near the tip of surgical robot tools may be impractical as it may require a large incision, infect biological tissues, and negatively affect surgeon's operation. In this study, a new vibration-based approach was proposed to measure the force applied to a structure utilizing the structure's acceleration signals. By exciting the structure using a vibration motor, the structure's acceleration signals in time domain showed discernible ellipse-shaped profiles when a force was applied. For the first time, these acceleration profiles were characterized via regression and employed for estimating the direction and magnitude of the applied force. Experimental results showed that, the achieved resolutions with the proposed approach in estimating the direction and magnitude of the applied force were 10{textdegree} and 0.098 N, respectively. The sensing errors were within the range of 8-18%. This force-sensing approach has strong potential for a wide area of robotic applications.

- Tactile Sensing Based on Fingertip Suction Flow for Submerged Dexterous Manipulation

    Author: Nadeau, Philippe | École De Technologie Supérieure
    Author: Abbott, Michael | UC Berkeley
    Author: Melville, Dominic | UC Berkeley
    Author: Stuart, Hannah | UC Berkeley
 
    keyword: Grasping; Force and Tactile Sensing; Dexterous Manipulation

    Abstract : The ocean is a harsh and unstructured environment for robotic systems; high ambient pressures, saltwater corrosion and low-light conditions demand machines with robust electrical and mechanical parts that are able to sense and respond to the environment. Prior work shows that the addition of gentle suction flow to the hands of underwater robots can aid in the handling of objects during mobile manipulation tasks. The current paper explores using this suction flow mechanism as a new modality for tactile sensing; by monitoring orifice occlusion we can get a sense of how objects make contact in the hand. The electronics required for this sensor can be located remotely from the hand and the signal is insensitive to large changes in ambient pressure associated with diving depth. In this study, suction is applied to the fingertips of a two-fingered compliant gripper and suction-based tactile sensing is monitored while an object is pulled out of a pinch grasp. As a proof of concept, a recurrent neural network model was trained to predict external force trends using only the suction signals. This tactile sensing modality holds the potential to enable automated robotic behaviors or to provide operators of remotely operated vehicles with additional feedback in a robust fashion suitable for ocean deployment.

- Discrete Bimanual Manipulation for Wrench Balancing

    Author: Cruciani, Silvia | KTH Royal Institute of Technology
    Author: Almeida, Diogo | Royal Institute of Technology, KTH
    Author: Kragic, Danica | KTH
    Author: Karayiannidis, Yiannis | Chalmers University of Technology &amp; KTH Royal Institute of Techn
 
    keyword: Dual Arm Manipulation; Dexterous Manipulation; Force and Tactile Sensing

    Abstract : Dual-arm robots can overcome grasping force and payload limitations of a single arm by jointly grasping an object. However, if the distribution of mass of the grasped object is not even, each arm will experience different wrenches that can exceed its payload limits. In this work, we consider the problem of balancing the wrenches experienced by a dual-arm robot grasping a rigid tray. The distribution of wrenches among the robot arms changes due to objects being placed on the tray. We present an approach to reduce the wrench imbalance among arms through discrete bimanual manipulation. Our approach is based on sequential sliding motions of the grasp points on the surface of the object, to attain a more balanced configuration. We validate our modeling approach and system design through a set of robot experiments.

- Shear, Torsion and Pressure Tactile Sensor Via Plastic Optofiber Guided Imaging

    Author: Baimukashev, Daulet | Nazarbayev University
    Author: Kappassov, Zhanat | Pierre and Marie Curie University
    Author: Varol, Huseyin Atakan | Nazarbayev University
 
    keyword: Force and Tactile Sensing; Soft Sensors and Actuators; Deep Learning in Robotics and Automation

    Abstract : Object manipulation performed by robots refers to the art of controlling the shape and location of an object through force constraints with robot end-effectors, both robot hands, and grippers. The success of task execution is usually guaranteed by the sense of touch. In this work, we present an optical tactile sensor - incorporating plastic optical fibers, transparent silicone rubber, and an off-the-shelf color camera - that can detect: translational and rotational shear forces, and contact location and its normal force. Contact localization is possible thanks to the shear strain. Specifically, one of the layers stretches so that its thickness decreases. The decrease in the thickness results in the color change at the point of contact. Elastic behavior of the sensing media provides a robust rotational and translational shear detection mechanism when torque and planar force, respectively, are applied onto the sensing surface. Thanks to the plastic optofibers, signal processing electronics are placed away from the sensing surface making the sensor immune to hazardous environments. Machine learning techniques were used to benchmark the sensing performance of the sensor. By implementing a multi-output CNN model, the contact type was classified into normal and shear or torsional deformation and their corresponding continuous contact features were estimated.

- Dynamically Reconfigurable Tactile Sensor for Robotic Manipulation

    Author: Huh, Tae Myung | Stanford University
    Author: Choi, Hojung | Stanford University
    Author: Willcox, Simone | Stanford University
    Author: Moon, Stephanie | Stanford University
    Author: Cutkosky, Mark | Stanford University
 
    keyword: Force and Tactile Sensing; Dexterous Manipulation

    Abstract : We present a new tactile sensor intended for manipulation by mobile robots, for example in the home. The surface consists of an array of small, rounded bumps or "nibs", which provide reliable traction on objects like wet dishes. When the nibs contact a surface they deflect, and capacitive sensors measure the corresponding local normal and shear forces. A key feature of the sensor is the ability to reconfigure dynamically depending on which combinations of sensing elements it samples. By interrogating different combinations of elements the sensor can detect and distinguish between linear and rotational sliding, and other dynamic events such as making and breaking contact. These dynamic events, combined with sensing the grasp and load forces, are useful for acquiring objects and performing simple in-hand manipulations. The proposed slip detection method estimates minimum required grasping force with an error less than 1.5N and uses tactile controlled rotational slips to reorient an unknown weight/surface object with 78% success rate.

- NeuroTac: A Neuromorphic Optical Tactile Sensor Applied to Texture Recognition

    Author: Ward-Cherrier, Benjamin | University of Bristol
    Author: Pestell, Nicholas | University of Bristol
    Author: Lepora, Nathan | University of Bristol
 
    keyword: Force and Tactile Sensing; Biomimetics; Neurorobotics

    Abstract : Developing artificial tactile sensing capabilities that rival human touch is a long-term goal in robotics and prosthetics. Gradually more elaborate biomimetic tactile sensors are being developed and applied to grasping and manipulation tasks to help achieve this goal. Here we present the neuroTac, a novel neuromorphic optical tactile sensor. The neuroTac combines the biomimetic hardware design from the TacTip sensor which mimicks the layered papillae structure of the human glabrous skin, with an event-based camera (DAVIS240, iniVation) and data encoding algorithms which transduce contact information in the form of spike trains. The performance of the neuroTac sensor is evaluated on a texture classification task, with four spike coding methods being implemented and compared: Intensive, Spatial, Temporal and Spatiotemporal. We found the timing-based coding methods performed with the highest accuracy over both artificial and natural textures. The spike-based output of the neuroTac could enable the development of biomimetic tactile perception algorithms in robotics as well as non-invasive and invasive haptic feedback methods in prosthetics.

- Reducing Uncertainty in Pose Estimation under Complex Contacts Via Force Forecast

    Author: Mao, Huitan | University of North Carolina at Charlotte
    Author: Xiao, Jing | Worcester Polytechnic Institute (WPI)
 
    keyword: Assembly; Force and Tactile Sensing; Manipulation Planning

    Abstract : How to reduce uncertainty in object pose estimation under complex contacts is crucial to autonomous robotic manipulation and assembly. In this paper, we introduce an approach through forecasting contact force from simulated complex contacts with calibration based on real force sensing. A constraint-based haptic simulation algorithm is used with sphere-tree representation of contacting objects to compute contact poses and forces, and through matching the computed forces to measured real force data via a regression model, the least-uncertain estimate of the relative contact pose is obtained. Our approach can handle multi-region complex contacts and does not make any assumption about contact types or contact locations. It also does not have restriction on object shapes. We have applied the force forecast approach to reducing uncertainty in estimating object poses in challenging peg-in-hole robotic assembly tasks and demonstrate the effectiveness of the approach by successful completion of contact-rich two-pin and three-pin real peg-in-hole assembly tasks with complex shapes of pins and holes.

- Comparison of Constrained and Unconstrained Human Grasp Forces Using Fingernail Imaging and Visual Servoing

    Author: Fallahinia, Navid | University of Utah
    Author: Mascaro, Stephen | University of Utah
 
    keyword: Force and Tactile Sensing; Grasping; Visual Servoing

    Abstract : Fingernail imaging has been proven to be effective in prior works [1], [2] for estimating the 3D fingertip forces with a maximum RMS estimation error of 7%. In the current research, fingernail imaging is used to perform unconstrained grasp force measurement on multiple fingers to study human grasping. Moreover, two robotic arms with mounted cameras and a visual tracking system have been devised to keep the human fingers in the camera frame during the experiments. Experimental tests have been conducted for six human subjects under both constrained and unconstrained grasping conditions, and the results indicate a significant difference in force collaboration among the fingers between the two grasping conditions. Another interesting result according to the experiments is that in comparison to constrained grasping, unconstrained grasp forces are more evenly distributed over the fingers and there is less force variation (more steadiness) in each finger force. These results validate the importance of measuring grasp forces in an unconstrained manner in order to study how humans naturally grasp objects.

- An ERT-Based Robotic Skin with Sparsely Distributed Electrodes: Structure, Fabrication, and DNN-Based Signal Processing

    Author: Park, Kyungseo | KAIST
    Author: Park, Hyunkyu | Korea Advanced Institute of Science and Technology
    Author: Lee, Hyosang | Max Planck Institute for Intelligent Systems
    Author: Park, Sungbin | Korea Advanced Institute of Science and Technology
    Author: Kim, Jung | KAIST
 
    keyword: Force and Tactile Sensing; Physical Human-Robot Interaction; Soft Robot Materials and Design

    Abstract : Electrical resistance tomography (ERT) has previously been utilized to develop a large-scale tactile sensor because this approach enables the estimation of the conductivity distribution among the electrodes based on a known physical model. Such a sensor made with a stretchable material can conform to a curved surface. However, this sensor cannot fully cover a cylindrical surface because in such a configuration, the edges of the sensor must meet each other. The electrode configuration becomes irregular in this edge region, which may degrade the sensor performance. In this paper, we introduce an ERT-based robotic skin with evenly and sparsely distributed electrodes. For implementation, we sprayed a carbon nanotube (CNT)-dispersed solution to form a conductive sensing domain on a cylindrical surface. The electrodes were firmly embedded in the surface so that the wires were not exposed to the outside. The sensor output images were estimated using a deep neural network (DNN), which was trained with noisy simulation data. An indentation experiment revealed that the localization error of the sensor was 5.2 - 3.3 mm, which is remarkable performance with only 30 electrodes. A frame rate of up to 120 Hz could be achieved with a sensing domain area of 90 cm^2. The proposed approach simplifies the fabrication of 3D-shaped sensors, allowing them to be easily applied to existing robot arms in a seamless and robust manner.

- FBG-Based Triaxial Force Sensor Integrated with an Eccentrically Configured Imaging Probe for Endoluminal Optical Biopsy

    Author: Wu, Zicong | Imperial College London
    Author: Gao, Anzhu | Shanghai Jiao Tong University
    Author: Liu, Ning | Imperial College London
    Author: Jin, Zhu | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Force and Tactile Sensing; Surgical Robotics: Steerable Catheters/Needles; Medical Robots and Systems

    Abstract : Accurate force sensing is important for endoluminal interventions in terms of both safety and lesion targeting. This paper develops an FBG-based force sensor for robotic bronchoscopy by configuring three FBG sensors at the lateral side of a conical substrate, which allows a large and eccentric inner lumen for the interventional instrument, like a flexible imaging probe to perform the optical biopsy. The force sensor is embodied with a laser-profiled continuum robot and thermo drift is fully compensated by three temperature sensors that are integrated on the circumference surface of the sensor substrate. Different decoupling approaches are investigated, and nonlinear decoupling is adopted based on the cross-validation SVM and a gaussian kernel function, achieving an accuracy of 10.58 mN, 14.57 mN and 26.32 mN along X, Y and Z axis, respectively. Besides, the tissue test is investigated to further demonstrate the feasibility of the developed triaxial force sensor

- Calibrating a Soft ERT-Based Tactile Sensor with a Multiphysics Model and Sim-To-Real Transfer Learning

    Author: Lee, Hyosang | Max Planck Institute for Intelligent Systems
    Author: Park, Hyunkyu | Korea Advanced Institute of Science and Technology
    Author: Serhat, Gokhan | Max Planck Institute for Intelligent Systems, Haptic Intelligenc
    Author: Sun, Huanbo | Max Planck Institute for Intelligent Systems
    Author: Kuchenbecker, Katherine J. | Max Planck Institute for Intelligent Systems
 
    keyword: Force and Tactile Sensing; Deep Learning in Robotics and Automation; Calibration and Identification

    Abstract : Tactile sensors based on electrical resistance tomography (ERT) have shown many advantages for implementing a soft and scalable whole-body robotic skin; however, calibration is challenging because pressure reconstruction is an ill-posed inverse problem. This paper introduces a method for calibrating soft ERT-based tactile sensors using sim-to-real transfer learning with a finite element multiphysics model. The model is composed of three simple models that together map contact pressure distributions to voltage measurements. We optimized the model parameters to reduce the gap between the simulation and reality. As a preliminary study, we discretized the sensing points into a 6 by 6 grid and synthesized single- and two-point contact datasets from the multiphysics model. We obtained another single-point dataset using the real sensor with the same contact location and force used in the simulation. Our new deep neural network architecture uses a de-noising network to capture the simulation-to-real gap and a reconstruction network to estimate contact force from voltage measurements. The proposed approach showed 82% hit rate for localization and 0.51 N of force estimation error performance in single-contact tests and 78.5% hit rate for localization and 5.0 N of force estimation error in two-point contact tests. We believe this new calibration method has the possibility to improve the sensing performance of ERT-based tactile sensors.

- Sim-To-Real Transfer for Optical Tactile Sensing

    Author: Ding, Zihan | Imperial College London
    Author: Lepora, Nathan | University of Bristol
    Author: Johns, Edward | Imperial College London
 
    keyword: Force and Tactile Sensing; Deep Learning in Robotics and Automation; Soft Sensors and Actuators

    Abstract : Deep learning and reinforcement learning methods have been shown to enable learning of flexible and complex robot controllers. However, the reliance on large amounts of training data often requires data collection to be carried out in simulation, with a number of sim-to-real transfer methods being developed in recent years. In this paper, we study these techniques for tactile sensing using the TacTip optical tactile sensor, which consists of a deformable tip with a camera observing the positions of pins inside this tip. We designed a model for soft body simulation which was implemented using the Unity physics engine, and trained a neural network to predict the locations and angles of edges when in contact with the sensor. Using domain randomisation techniques for sim-to-real transfer, we show how this framework can be used to accurately predict edges with less than 1 mm prediction error in real-world testing, without any real-world data at all.

- Semi-Empirical Simulation of Learned Force Response Models for Heterogeneous Elastic Objects

    Author: Zhu, Yifan | University of Illinois at Urbana-Champaign
    Author: Lu, Kai | Tsinghua Univerisity
    Author: Hauser, Kris | University of Illinois at Urbana-Champaign
 
    keyword: Force and Tactile Sensing; Contact Modeling; Simulation and Animation

    Abstract : This paper presents a semi-empirical method for simulating contact with elastically deformable objects whose force response is learned using entirely data-driven models. A point-based surface representation and an inhomogeneous, nonlinear force response model are learned from a robotic arm acquiring force-displacement curves from a small number of poking interactions. The simulator then estimates displacement and force response when the deformable object is in contact with an arbitrary rigid object. It does so by estimating displacements by solving a Hertzian contact model, and sums the expected forces at individual surface points through querying the learned point stiffness models as a function of their expected displacements. Experiments on a variety of challenging objects show that our approach learns force response with sufficient accuracy to generate plausible contact response for novel rigid objects.

- Low-Cost Fiducial-Based 6-Axis Force-Torque Sensor

    Author: Ouyang, Rui | Harvard University
    Author: Howe, Robert D. | Harvard University
 
    keyword: Force and Tactile Sensing; Computer Vision for Other Robotic Applications; Perception for Grasping and Manipulation

    Abstract : Commercial six-axis force-torque sensors suffer from being some combination of expensive, fragile, and hard-to-use. We propose a new fiducial-based design which addresses all three points. The sensor uses an inexpensive webcam and can be fabricated using a consumer-grade 3D printer. Open-source software is used to estimate the 3D pose of the fiducials on the sensor, which is then used to calculate the applied force-torque. A browser-based (installation free) interface demonstrates ease-of-use. The sensor is very light and can be dropped or thrown with little concern. We characterize our prototype in dynamic conditions under compound loading, finding a mean R^2 of over 0.99 for the F_x, F_y, M_x, and M_y axes, and over 0.87 and 0.90 for the F_z and M_z axes respectively. The open source design files allow the sensor to be adapted for diverse applications ranging from robot fingers to human-computer interfaces, while the simple design principle allows for quick changes with minimal technical expertise. This approach promises to bring six-axis force-torque sensing to new applications where the precision, cost, and fragility of traditional strain-gauge based sensors are not appropriate. The open-source sensor de sign can be viewed at http://sites.google.com/view/ fiducialforcesensor.

- Curvature Sensing with a Spherical Tactile Sensor Based on the Color-Interference of a Marker Array

    Author: Lin, Xi | ISM, CNRS, Aix-Marseille Université
    Author: Willemet, Laurence | ISM, CNRS, Aix-Marseille Université
    Author: Bailleul, Alexandre | ISM, Aix-Marseille Université
    Author: Wiertlewski, Michael | TU Delft
 
    keyword: Force and Tactile Sensing; Soft Robot Materials and Design

    Abstract : It is well accepted that touch is an important sensory channel to consider while planning of robotic manipulation tasks. Touch provides information about the state of contact and the local shape of the object which is central to fine manipulation. In this work, we present an evolution of our distributed tactile sensor which is able to measure the dense 3-dimensional displacement field of an elastic membrane, using the subtractive color-mixing principle. The manufacturing process employed allows us to design and manufacture the features of the sensor on a flat surface, then fold the resulting 2d structure into a spherical shape. The resulting 40mm-diameter spherical sensor has 77 measurement points, each of which gives an estimation of the local 3d displacement, normal and tangential to the surface. Each marker is built around 2 sets of colored patch placed at different depths. The first one reflects magenta light while the second is a translucent yellow filter that converts the magenta into the red. An embedded camera observes the lateral displacement and the resulting hue of the marker. To benchmark the sensor, we compared the measurement obtained while pressing the sensor on a curved surface with Hertz contact theory, which constitutes a classical contact mechanics problem. While Hertz contact assumes frictionless conditions, using the shear and normal sensing, ChromaTouch can estimate the curvature of an object after an indentation of the sensor of less than a millimeter.

- Center-Of-Mass-Based Robust Grasp Planning for Unknown Objects Using Tactile-Visual Sensors

    Author: Feng, Qian | Technical University of Munich
    Author: Chen, Zhaopeng | University of Hamburg
    Author: Deng, Jun | Agile Robots AG
    Author: Gao, Chunhui | Agile Robots AG
    Author: Zhang, Jianwei | University of Hamburg
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
 
    keyword: Force and Tactile Sensing; Grasping; Deep Learning in Robotics and Automation

    Abstract : An unstable grasp pose can lead to slip, thus an unstable grasp pose can be predicted by slip detection. A regrasp is required afterwards to correct the grasp pose in order to finish the task. In this work, we propose a novel regrasp planner with multi-sensor modules to plan grasp adjustments with the feedback from a slip detector. Then a regrasp planner is trained to estimate the location of center of mass, which helps robots find an optimal grasp pose. The dataset in this work consists of 1 025 slip experiments and 1347 regrasps collected by one pair of tactile sensors, an RGB-D camera and one Franka Emika robot arm equipped with joint force/torque sensors. We show that our algorithm can successfully detect and classify the slip for 5 unknown test objects with an accuracy of 76.88% and a regrasp planner increases the grasp success rate by 31.0% compared to the state-of-the-art vision-based grasping algorithm.

- OmniTact: A Multi-Directional High-Resolution Touch Sensor

    Author: Padmanabha, Akhil | UC Berkeley
    Author: Ebert, Frederik | UC Berkeley
    Author: Tian, Stephen | UC Berkeley
    Author: Calandra, Roberto | Facebook
    Author: Finn, Chelsea | Stanford University
    Author: Levine, Sergey | UC Berkeley
 
    keyword: Force and Tactile Sensing; Perception for Grasping and Manipulation; Soft Sensors and Actuators

    Abstract : Incorporating touch as a sensing modality for robots can enable finer and more robust manipulation skills. Existing tactile sensors are either flat, have small sensitive fields or only provide low-resolution signals. In this paper, we introduce OmniTact, a multi-directional high-resolution tactile sensor. OmniTact is designed to be used as a fingertip for robotic manipulation with robotic hands, and uses multiple micro-cameras to detect multi-directional deformations of a gel-based skin. This provides a rich signal from which a variety of different contact state variables can be inferred using modern image processing and computer vision methods. We evaluate the capabilities of OmniTact on a challenging robotic control task that requires inserting an electrical connector into an outlet, as well as a state estimation problem that is representative of those typically encountered in dexterous robotic manipulation, where the goal is to infer the angle of contact of a curved finger pressing against an object. Both tasks are performed using only touch sensing and convolutional neural networks to process images from the sensor's cameras. We compare with a state-of-the-art tactile sensor that is only sensitive on one side, as well as a state-of-the-art multi-directional tactile sensor, and find that the combination of high-resolution and multi-directional sensing is crucial for reliably inserting the electrical connector and allows for higher accuracy in the state estimation task.

- Highly Sensitive Bio-Inspired Sensor for Fine Surface Exploration and Characterization

    Author: Ribeiro, Pedro | Instituto Superior Tecnico
    Author: Cardoso, Susana | INESC-Microsistemas E Nanotecnologias and In
    Author: Bernardino, Alexandre | IST - Técnico Lisboa
    Author: Jamone, Lorenzo | Queen Mary University London
 
    keyword: Force and Tactile Sensing; Biomimetics; Soft Sensors and Actuators

    Abstract : Texture sensing is one of the types of information sensed by humans through touch, and is thus of interest to robotics that this type of information can be acquired and processed. In this work we present a texture topography sensor based on a ciliary structure, similar to a biological structure found in many organisms. The device consists on up to 9 elastic cilia with permanent magnetization assembled on top of a highly sensitive tunneling magnetoresistance (TMR) sensor, within a compact footprint of 6x6 mm2 . When these cilia brush against some textured surface, their movement and vibrations give rise to a signal that can be correlated to the characteristics of the texture being measured. We also present an electronic signal acquisition board used in this work. Various configurations of cilia sizes are tested, with the most precise being capable of differentiating different types of sandpaper from 9.2 �m to 213 �m average surface roughness with a 7 �m resolution. As a topography scanner the sensor was able to scan a 20 �m high step in a flat surface.

- Implementing Tactile and Proximity Sensing for Crack Detection

    Author: Palermo, Francesca | Queen Mary University of London
    Author: Konstantinova, Jelizaveta | Ocado Technology
    Author: Althoefer, Kaspar | Queen Mary University of London
    Author: Poslad, Stefan | Queen Mary University of London
    Author: Farkhatdinov, Ildar | Queen Mary University of London
 
    keyword: Force and Tactile Sensing; Robotics in Hazardous Fields; Sensor-based Control

    Abstract : Remote characterisation of the environment during physical robot-environment interaction is an important task commonly accomplished in telerobotics. This paper demonstrates how tactile and proximity sensing can be efficiently used to perform automatic crack detection. A custom-designed integrated tactile and proximity sensor is implemented. It measures the deformation of its body when interacting with the physical environment and distance to the environment's objects with the help of fibre optics. This sensor was used to slide across different surfaces and the data recorded during the experiments was used to detect and classify cracks, bumps and undulations. The proposed method uses machine learning techniques (mean absolute value as feature and random forest as classifier) to detect cracks and determine their width. An average crack detection accuracy of 86.46% and width classification accuracy of 57.30% is achieved. Kruskal-Wallis results (p&lt;0.001) indicate statistically significant differences among results obtained when analysing only force data, only proximity data and both force and proximity data. In contrast to previous techniques, which mainly rely on visual modality, the proposed approach based on optical fibres is suitable for operation in extreme environments, such as nuclear facilities in which nuclear radiation may damage the electronic components of video cameras.

- Novel Proximity Sensor for Realizing Tactile Sense in Suction Cups

    Author: Doi, Sayaka | OMRON Corporation
    Author: Koga, Hiroki | Omron Corporation
    Author: Seki, Tomonori | OMRON Corporation
    Author: Okuno, Yutaro | OMRON
 
    keyword: Force and Tactile Sensing; Sensor-based Control; Failure Detection and Recovery

    Abstract : We propose a new capacitive proximity sensor that detects deformations of a suction cup as a tactile sense. We confirmed that one sensor module provides three applications for reliable picking and a simplified setup. The first application is the picking height decision. The second one is the placing height decision for detecting whether the grasped object is placed on the placement surface. These two applications are achieved by detecting the push-in stroke of the suction cup. The final application is detection of whether the suction cup is in partial contact or full contact with the object. This function can correct the picking posture as well as detect whether picking is possible before the pull-up motion. We also demonstrate that the partial contact position can be estimated in real time.

## Visual-Based Navigation

- Exploring Performance Bounds of Visual Place Recognition Using Extended Precision

    Author: Ferrarini, Bruno | Universtiy of Essex
    Author: Waheed, Maria | COMSATS University
    Author: Waheed, Sania | National University of Sciences and Technology
    Author: Ehsan, Shoaib | University of Essex
    Author: Milford, Michael J | Queensland University of Technology
    Author: McDonald-Maier, Klaus | University of Essex
 
    keyword: Visual-Based Navigation; Localization

    Abstract : Recent advances in image description and matching allowed significant improvements in Visual Place Recognition (VPR). The wide variety of methods proposed so far and the increase of the interest in the field have rendered the problem of evaluating VPR methods an important task. As part of the localization process, VPR is a critical stage for many robotic applications and it is expected to perform reliably in any location of the operating environment. To design more reliable and effective localization systems this letter presents a generic evaluation framework based on the new Extended Precision performance metric for VPR. The proposed framework allows assessment of the upper and lower bounds of VPR performance and finds statistically significant performance differences between VPR methods. The proposed evaluation method is used to assess several state-of-the-art techniques with a variety of imaging conditions that an autonomous navigation system commonly encounters on long term runs. The results provide new insights into the behaviour of different VPR methods under varying conditions and help to decide which technique is more appropriate to the nature of the venture or the task assigned to an autonomous robot.

- Deep Reinforcement Learning for Instruction Following Visual Navigation in 3D Maze-Like Environments

    Author: Devo, Alessandro | University of Perugia
    Author: Costante, Gabriele | University of Perugia
    Author: Valigi, Paolo | Universita' Di Perugia
 
    keyword: Visual-Based Navigation; Deep Learning in Robotics and Automation; Visual Learning

    Abstract : In this work, we address the problem of visual navigation by following instructions. In this task, the robot must interpret a natural language instruction in order to follow a predefined path in a possibly unknown environment. Despite different approaches have been proposed in the last years, they are all based on the assumption that the environment contains objects or other elements that can be used to formulate instructions, such as houses or offices. On the contrary, we focus on situations where the environment objects cannot be used to specify a navigation path. In particular, we consider 3D maze-like environments as our test bench because they can be very large and offer very intricate structures. We show that without reference points, visual navigation and instruction following can be rather challenging, and that standard approaches can not be applied successfully. For this reason, we propose a new architecture that explicitly learns both visual navigation and instruction understanding. We demonstrate with simulated experiments that our method can effectively follow instructions and navigate in previously unseen mazes of various sizes.

- Aggressive Perception-Aware Navigation Using Deep Optical Flow Dynamics and PixelMPC

    Author: Lee, Keuntaek | Georgia Institute of Technology
    Author: Gibson, Jason | Georgia Institute of Technology
    Author: Theodorou, Evangelos | Georgia Institute of Technology
 
    keyword: Visual-Based Navigation; Visual Servoing; Visual Tracking

    Abstract : Recently, vision-based control has gained traction by leveraging the power of machine learning. In this work, we couple a model predictive control (MPC) framework to a visual pipeline. We introduce deep optical flow (DOF) dynamics, which is a combination of optical flow and robot dynamics. Using the DOF dynamics, MPC explicitly incorporates the predicted movement of relevant pixels into the planned trajectory of a robot. Our implementation of DOF is memory-efficient, data-efficient, and computationally cheap so that it can be computed in real-time for use in an MPC framework. The suggested Pixel Model Predictive Control (PixelMPC) algorithm controls the robot to accomplish a high-speed racing task while maintaining visibility of the important features (gates). This improves the reliability of vision-based estimators for localization and can eventually lead to safe autonomous flight. The proposed algorithm is tested in a photorealistic simulation with a high-speed drone racing task.

- Visual-Inertial Mapping with Non-Linear Factor Recovery

    Author: Usenko, Vladyslav | TU Munich
    Author: Demmel, Nikolaus | Technische Universitét M�nchen
    Author: Schubert, David | Technical University of Munich
    Author: Stueckler, Joerg | Max-Planck Institute for Intelligent Systems
    Author: Cremers, Daniel | Technical University of Munich
 
    keyword: Visual-Based Navigation; Mapping; Sensor Fusion

    Abstract : Cameras and inertial measurement units are complementary sensors for ego-motion estimation and environment mapping. Their combination makes visual-inertial odometry (VIO) systems more accurate and robust. For globally consistent mapping, however, combining visual and inertial information is not straightforward. To estimate the motion and geometry with a set of images large baselines are required. Because of that, most systems operate on keyframes that have large time intervals between each other. Inertial data on the other hand quickly degrades with the duration of the intervals and after several seconds of integration, it typically contains only little useful information.<p>In this paper, we propose to extract relevant information for visual-inertial mapping from visual-inertial odometry using non-linear factor recovery. We reconstruct a set of non-linear factors that make an optimal approximation of the information on the trajectory accumulated by VIO. To obtain a globally consistent map we combine these factors with loop-closing constraints using bundle adjustment. The VIO factors make the roll and pitch angles of the global map observable, and improve the robustness and the accuracy of the mapping. In experiments on a public benchmark, we demonstrate superior performance of our method over the state-of-the-art approaches.

- Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments

    Author: Xia, Fei | Stanford University
    Author: Shen, William B. | Stanford University
    Author: Li, Chengshu | Stanford University
    Author: Kasimbeg, Priya | Stanford University
    Author: Tchapmi, Micael Edmond | Stanford University
    Author: Toshev, Alexander | Google
    Author: Martín-Martín, Roberto | Stanford University
    Author: Savarese, Silvio | Stanford University
 
    keyword: Visual-Based Navigation; Deep Learning in Robotics and Automation; Mobile Manipulation

    Abstract : We present Interactive Gibson Benchmark, the first comprehensive benchmark for training and evaluating Interactive Navigation solutions. Interactive Navigation tasks are robot navigation problems where physical interaction with objects (e.g. pushing) is allowed and even encouraged to reach the goal. Our benchmark comprises two novel elements: 1) a new experimental simulated environment, the Interactive Gibson Environment, that generates photo-realistic images of indoor scenes and simulates realistic physical interactions of robots and common objects found in these scenes; 2) the Interactive Navigation Score, a novel metric to study the interplay between navigation and physical interaction of Interactive Navigation solutions. We present and evaluate multiple learning-based baselines in Interactive Gibson Benchmark, and provide insights into regimes of navigation with different trade-offs between navigation, path efficiency and disturbance of surrounding objects. We make our benchmark publicly available and encourage researchers from related robotics disciplines (e.g. planning, learning, control) to propose, evaluate, and compare their Interactive Navigation solutions in Interactive Gibson Benchmark.

- Highly Robust Visual Place Recognition through Spatial Matching of CNN Features

    Author: Camara, Luis Gomez | CIIRC CTU Prague
    Author: G�bert, Carl | Czech Institute of Informatics, Robotics and Cybernetics
    Author: Preucil, Libor | Czech Technical University in Prague
 
    keyword: Visual-Based Navigation; Localization; Deep Learning in Robotics and Automation

    Abstract : We revise, extend and consolidate the system previously introduced by us and named SSM-VPR (Semantic and Spatial Matching Visual Place Recognition), largely boosting its performance above the current state of the art. The system encodes images of places by employing the activations of different layers of a pre-trained, off-the-shelf, VGG16 Convolutional Neural Network (CNN) architecture. It consists of two stages: given a query image of a place, (1) a list of candidates is selected from a database of places and (2) the candidates are geometrically compared with the query by matching CNN features and, equally important, their spatial locations. The best matching candidate is then deemed as the recognized place. The performance of the system is maximized by finding optimal image resolutions during the second stage and by exploiting temporal correlation between consecutive frames in the employed datasets.

- Robust and Efficient Estimation of Absolute Camera Pose for Monocular Visual Odometry

    Author: Li, Haoang | The Chinese University of Hong Kong
    Author: Chen, Wen | The Chinese University of Hong Kong
    Author: Zhao, Ji | TuSimple
    Author: Bazin, Jean-Charles | KAIST
    Author: Luo, Lei | Wuhan University
    Author: Liu, Zhe | The Chinese University of Hong Kong
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Visual-Based Navigation; Localization; SLAM

    Abstract : Given a set of 3D-to-2D point correspondences corrupted by outliers, we aim to robustly estimate the absolute camera pose. Existing methods robust to outliers either fail to guarantee high robustness and efficiency simultaneously, or require an appropriate initial pose and thus lack generality. In contrast, we propose a novel approach based on the robust �L2-minimizing estimate�(L2E) loss. We first define a novel cost function by integrating the projection constraint into the L2E loss. Then to efficiently obtain the global minimum of this function, we propose a hybrid strategy of a local optimizer and branch-and-bound. For branch-and-bound, we derive effective function bounds. Our approach can handle high outlier ratios, leading to high robustness. It can run reliably regardless of whether the initial pose is appropriate, providing high generality. Moreover, given a decent initial pose, it is suitable for real-time applications. Experiments on synthetic and real-world datasets showed that our approach outperforms state-of-the-art methods in terms of robustness and/or efficiency.

- Robust Vision-Based Obstacle Avoidance for Micro Aerial Vehicles in Dynamic Environments

    Author: Lin, Jiahao | Delft University of Technology
    Author: Zhu, Hai | Delft University of Technology
    Author: Alonso-Mora, Javier | Delft University of Technology
 
    keyword: Visual-Based Navigation; Aerial Systems: Perception and Autonomy; Collision Avoidance

    Abstract : In this paper, we present an on-board vision-based approach for avoidance of moving obstacles in dynamic environments. Our approach relies on an efficient obstacle detection and tracking algorithm based on stereo image pairs, which provides the estimated position, velocity and size of the obstacles. Robust collision avoidance is achieved by formulating a chance-constrained model predictive controller (CC-MPC) to ensure that the collision probability between the micro aerial vehicle (MAV) and each moving obstacle is below a specified threshold. The method takes into account MAV dynamics, state estimation and obstacle sensing uncertainties. The proposed approach is implemented on a quadrotor equipped with a stereo camera and is tested in a variety of environments, showing effective on-line collision avoidance of moving obstacles.

- Proximity Estimation Using Vision Features Computed on Sensor

    Author: Chen, Jianing | The University of Manchester
    Author: Liu, Yanan | University of Bristol
    Author: Carey, Stephen J. | The University of Manchester
    Author: Dudek, Piotr | The University of Manchester
 
    keyword: Visual-Based Navigation; Reactive and Sensor-Based Planning; Collision Avoidance

    Abstract : This paper presents a monocular vision based proximity estimation system using     Abstract features, such as corner points, blobs and edges, as inputs to a neural network. An experimental vehicle was built using a vision system integrating the SCAMP-5 vision chip, a micro-controller, and an RC model car. The vision chip includes image sensor with embedded 256x256 processor SIMD array. The pixel processor array chip was programmed to capture images and run the feature algorithms directly on the focal plane, and then digest them so that only sparse feature description data were read-out in the form of 40 values. By logging the vision output and the output from three infrared proximity sensors, training data were obtained to train three fully connected layer-recurrent neural networks with fewer than 700 parameters each. The trained neural network was able to estimate the proximity to the level of accuracy sufficient for a reactive collision avoidance behaviour to be achieved. The latency of the control system, from image capture to neural network output, was under 4 msec, enabling the vehicles to avoid obstacles of while moving at 0.6 m/s to 1.8 m/s in the experiment.

- Efficient Globally-Optimal Correspondence-Less Visual Odometry for Planar Ground Vehicles

    Author: Gao, Ling | ShanghaiTech University
    Author: Su, Junyan | ShanghaiTech University
    Author: Cui, Jiadi | ShanghaiTech University
    Author: Zeng, Xiangchen | ShanghaiTech University
    Author: Peng, Xin | ShanghaiTech University
    Author: Kneip, Laurent | ShanghaiTech
 
    keyword: Visual-Based Navigation; Localization; Intelligent Transportation Systems

    Abstract : The motion of planar ground vehicles is often non-holonomic, and as a result may be modelled by the 2 DoF Ackermann steering model. We analyse the feasibility of estimating such motion with a downward facing camera that exerts fronto-parallel motion with respect to the ground plane. This turns the motion estimation into a simple image registration problem in which we only have to identify a 2-parameter planar homography. However, one difficulty that arises from this setup is that ground-plane features are indistinctive and thus hard to match between successive views. We encountered this difficulty by introducing the first globally-optimal, correspondence-less solution to plane-based Ackermann motion estimation. The solution relies on the branch-and-bound optimisation technique. Through the low-dimensional parametrisation, a derivation of tight bounds, and an efficient implementation, we demonstrate how this technique is eventually amenable to accurate real-time motion estimation. We prove its property of global optimality and analyse the impact of assuming a locally constant centre of rotation. Our results on real data finally demonstrate a significant advantage over the more traditional, correspondence-based hypothesise-and-test schemes.

- EgoTEB: Egocentric, Perception Space Navigation Using Timed-Elastic-Bands

    Author: Smith, Justin | Georgia Institute of Technology
    Author: Xu, Ruoyang | Georgia Institute of Technology
    Author: Vela, Patricio | Georgia Institute of Technology
 
    keyword: Visual-Based Navigation; Collision Avoidance; Motion and Path Planning

    Abstract : The TEB hierarchical planner for real-time navigation through unknown environments is highly effective at balancing collision avoidance with goal directed motion. Designed over several years and publications, it implements a multi-trajectory optimization based synthesis method for identifying topologically distinct trajectory candidates through navigable space. Unfortunately, the underlying factor graph approach to the optimization problem induces a mismatch between grid-based representations and the optimization graph, which leads to several time and optimization inefficiencies. This paper explores the impact of using egocentric, perception space representations for the local planning map. Doing so alleviates many of the identified issues related to TEB and leads to a new method called egoTEB. Timing experiments and Monte Carlo evaluations in benchmark worlds quantify the benefits of egoTEB for navigation through uncertain environments.

- Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection

    Author: Yang, Heng | MIT
    Author: Antonante, Pasquale | MIT
    Author: Tzoumas, Vasileios | Massachusetts Institute of Technology
    Author: Carlone, Luca | Massachusetts Institute of Technology
 
    keyword: Visual-Based Navigation; SLAM; Optimization and Optimal Control

    Abstract : Semidefinite Programming (SDP) and Sums-of-Squares (SOS) relaxations have led to certifiably optimal non-minimal solvers for several robotics and computer vision problems. However, most non-minimal solvers rely on least squares formulations, and, as a result, are brittle against outliers. While a standard approach to regain robustness against outliers is to use robust cost functions, the latter typically introduce other non-convexities, preventing the use of existing non-minimal solvers. In this letter, we enable the simultaneous use of non-minimal solvers and robust estimation by providing a general-purpose approach for robust global estimation, which can be applied to any problem where a non-minimal solver is available for the outlier-free case. To this end, we leverage the Black-Rangarajan duality between robust estimation and outlier processes, and show that graduated non-convexity (GNC) can be used in conjunction with non-minimal solvers to compute robust solutions, without requiring an initial guess. We demonstrate the resulting robust non-minimal solvers in applications, including point cloud and mesh registration, pose graph optimization, and image-based object pose estimation (also called shape alignment). Our solvers are robust to 70�80% of outliers, outperform RANSAC, are more accurate than specialized local solvers, and faster than specialized global solvers. We also propose the first certifiably optimal non-minimal solver for shape alignment using SOS relaxation.

- Reliable Frame-To-Frame Motion Estimation for Vehicle-Mounted Surround-View Camera Systems

    Author: Wang, Yifu | Australian National University
    Author: Huang, Kun | ShanghaiTech University
    Author: Peng, Xin | ShanghaiTech University
    Author: Li, Hongdong | Australian National University and NICTA
    Author: Kneip, Laurent | ShanghaiTech
 
    keyword: Visual-Based Navigation; Omnidirectional Vision; Localization

    Abstract : Modern vehicles are often equipped with a surround-view multi-camera system. The current interest in autonomous driving invites the investigation of how to use such systems for a reliable estimation of relative vehicle displacement. Existing camera pose algorithms either work for a single camera, make overly simplified assumptions, are computationally expensive, or simply become degenerate under non-holonomic vehicle motion. In this paper, we introduce a new, reliable solution able to handle all kinds of relative displacements in the plane despite the possibly non-holonomic characteristics. We furthermore introduce a novel two-view optimization scheme which minimizes a geometrically relevant error without relying on 3D point related optimization variables. Our method leads to highly reliable and accurate frame-to-frame visual odometry with a full-size, vehicle-mounted surround-view camera system.

- Enabling Topological Planning with Monocular Vision

    Author: Stein, Gregory | CSAIL, MIT
    Author: Bradley, Christopher | CSAIL, MIT
    Author: Preston, Victoria | Massachusetts Institute of Technology
    Author: Roy, Nicholas | Massachusetts Institute of Technology
 
    keyword: Visual-Based Navigation; Mapping; Motion and Path Planning

    Abstract : Topological strategies for navigation meaningfully reduce the space of possible actions available to a robot, allowing use of heuristic priors or learning to enable computationally efficient, intelligent planning. The challenges in estimating structure with monocular SLAM in low texture or highly cluttered environments have precluded its use for topological planning in the past. We propose a robust sparse map representation that can be built with monocular vision and overcomes these shortcomings. Using a learned sensor, we estimate high-level structure of an environment from streaming images by detecting sparse "vertices" (e.g., boundaries of walls) and reasoning about the structure between them. We also estimate the known free space in our map, a necessary feature for planning through previously unknown environments. We show that our mapping technique can be used on real data and is sufficient for planning and exploration in simulated multi-agent search and learned subgoal planning applications.

- DeepMEL: Compiling Visual Multi-Experience Localization into a Deep Neural Network

    Author: Gridseth, Mona | University of Toronto
    Author: Barfoot, Timothy | University of Toronto
 
    keyword: Visual-Based Navigation; Deep Learning in Robotics and Automation; Localization

    Abstract : Vision-based path following allows robots to autonomously repeat manually taught paths. Stereo Visual Teach and Repeat (VT&amp;R) [1] accomplishes accurate and robust long-range path following in unstructured outdoor environments across changing lighting, weather, and seasons by relying on colour-constant imaging [2] and multi-experience localization [3]. We leverage multi-experience VT&amp;R together with two datasets of outdoor driving on two separate paths spanning different times of day, weather, and seasons to teach a deep neural network to predict relative pose for visual odometry (VO) and for localization with respect to a path. In this paper we run experiments exclusively on datasets to study how the network generalizes across environmental conditions. Based on the results we believe that our system achieves relative pose estimates sufficiently accurate for in-the-loop path following and that it is able to localize radically different conditions against each other directly (i.e. winter to spring and day to night), a capability that our hand-engineered system does not have.

- SnapNav: Learning Mapless Visual Navigationwith Sparse Directional Guidance and Visual Reference

    Author: Xie, Linhai | University of Oxford
    Author: Markham, Andrew | Oxford University
    Author: Trigoni, Niki | University of Oxford
 
    keyword: Visual-Based Navigation; Deep Learning in Robotics and Automation

    Abstract : Learning-based visual navigation still remains a challenging problem in robotics, with two overarching issues: how to transfer the learnt policy to unseen scenarios, and how to deploy the system on real robots. In this paper, we propose a deep neural network based visual navigation system, SnapNav. Unlike map-based navigation or Visual-Teach-and-Repeat (VT&amp;R), SnapNav only receives a few snapshots of the environment combined with directional guidance to allow it to execute the navigation task. Additionally, SnapNav can be easily deployed on real robots due to a two-level hierarchy: a high level commander that provides directional commands and a low level controller that provides real-time control and obstacle avoidance. This also allows us to effectively use simulated and real data to train the different layers of the hierarchy, facilitating robust control. Extensive experimental results show that SnapNav achieves a highly autonomous navigation ability compared to baseline models, enabling sparse, map-less navigation in previously unseen environments.

- Kimera: An Open-Source Library for Real-Time Metric-Semantic Localization and Mapping

    Author: Rosinol, Antoni | MIT
    Author: Abate, Marcus | MIT
    Author: Chang, Yun | MIT
    Author: Carlone, Luca | Massachusetts Institute of Technology
 
    keyword: Visual-Based Navigation; SLAM; Mapping

    Abstract : We provide an open-source C++ library for real-time metric-semantic visual-inertial Simultaneous Localization And Mapping (SLAM). The library goes beyond existing visual and visual-inertial SLAM libraries (e.g., ORB-SLAM, VINS-Mono, OKVIS, ROVIO) by enabling mesh reconstruction and 3D semantic labeling in 3D. Kimera is designed with modularity in mind and has four key components: a visual-inertial odometry (VIO) module for fast and accurate state estimation, a robust pose graph optimizer for global trajectory estimation, a lightweight 3D mesher module for fast mesh reconstruction, and a dense 3D metric-semantic reconstruction module. The modules can be run in isolation or in combination, hence Kimera can easily fall back to a state-of-the-art VIO or a full SLAM system. Kimera runs in real-time on a CPU and produces a 3D metric-semantic mesh from semantically labeled images, which can be obtained by modern deep learning methods. We hope that the flexibility, computational efficiency, robustness, and accuracy afforded by Kimera will build a solid basis for future metric-semantic SLAM and perception research, and will allow researchers across multiple areas (e.g., VIO, SLAM, 3D reconstruction, segmentation) to benchmark and prototype their own efforts without having to start from scratch.

- CityLearn: Diverse Real-World Environments for Sample-Efficient Navigation Policy Learning

    Author: Chanc�n Le�n, Marvin Aldo | Queensland University of Technology
    Author: Milford, Michael J | Queensland University of Technology
 
    keyword: Visual-Based Navigation; Visual Learning; Deep Learning in Robotics and Automation

    Abstract : Visual navigation tasks in real-world environments often require both self-motion and place recognition feedback. While deep reinforcement learning has shown success in solving these perception and decision-making problems in an end-to-end manner, these algorithms require large amounts of experience to learn navigation policies from high-dimensional data, which is generally impractical for real robots due to sample complexity. In this paper, we address these problems with two main contributions. We first leverage place recognition and deep learning techniques combined with goal destination feedback to generate compact, bimodal image representations that can then be used to effectively learn control policies from a small amount of experience. Second, we present an interactive framework, CityLearn, that enables for the first time training and deployment of navigation algorithms across city-sized, realistic environments with extreme visual appearance changes. CityLearn features more than 10 benchmark datasets, often used in visual place recognition and autonomous driving research, including over 100 recorded traversals across 60 cities around the world. We evaluate our approach on two CityLearn environments, training our navigation policy on a single traversal per dataset. Results show our method can be over 2 orders of magnitude faster than when using raw images, and can also generalize across extreme visual changes including day to night and summer to winter transitions.

- Constrained Filtering-Based Fusion of Images, Events, and Inertial Measurements for Pose Estimation

    Author: Jung, jae Hyung | Seoul National University
    Author: Park, Chan Gook | Seoul National University
 
    keyword: Visual-Based Navigation; Localization; Sensor Fusion

    Abstract : In this paper, we propose a novel filtering-based method that fuses events from a dynamic vision sensor (DVS), images, and inertial measurements to estimate camera poses. A DVS is a bio-inspired sensor that generates events triggered by brightness changes. It can cover the drawbacks of a conventional camera by virtual of its independent pixels and high dynamic range. Specifically, we focus on optical flow obtained from both a stream of events and intensity images in which the former is much like a differential quantity, whereas the latter is a pixel difference in a much longer time interval than events. This nature characteristic motivates us to model optical flow estimated from events directly, but feature tracks for images in the filter design. An inequality constraint is considered in our method since the inverse scene-depth is larger than zero by its definition. Furthermore, we evaluate our proposed method in the benchmark DVS dataset and a dataset collected by the     Authors. The results reveal that the presented algorithm has reduced the position error by 49.9% on average and comparable accuracy only using events when compared to the state-of-the-art filtering-based estimator.

- Schmidt-EKF-Based Visual-Inertial Moving Object Tracking

    Author: Eckenhoff, Kevin | University of Delaware
    Author: Geneva, Patrick | University of Delaware
    Author: Merrill, Nathaniel | University of Delaware
    Author: Huang, Guoquan | University of Delaware
 
    keyword: Visual-Based Navigation; Sensor Fusion; Localization

    Abstract : In this paper we investigate the effect of tightly-coupled estimation on the performance of visual-inertial localization and dynamic object pose tracking. In particular, we show that while a joint estimation system outperforms its decoupled counterpart when given a ``proper'' model for the target's motion, inconsistent modeling, such as choosing improper levels for the target's propagation noises, can actually lead to a degradation in ego-motion accuracy. To address the realistic scenario where a good prior knowledge of the target's motion model is not available, we design a new system based on the Schmidt-Kalman Filter (SKF), in which target measurements do not update the navigation states, however all correlations are still properly tracked. This allows for both consistent modeling of the target errors and the ability to update target estimates whenever the tracking sensor receives non-target data such as bearing measurements to static, 3D environmental features. We show in extensive simulation that this system, along with a robot-centric representation of the target, leads to robust estimation performance even in the presence of an inconsistent target motion model. Finally, the system is validated in a real-world experiment, and is shown to offer accurate localization and object pose tracking performance.

- Learning View and Target Invariant Visual Servoing for Navigation

    Author: Li, Yimeng | George Mason University
    Author: Kosecka, Jana | George Mason University
 
    keyword: Visual-Based Navigation; Visual Servoing; Model Learning for Control

    Abstract : The advances in deep reinforcement learning recently revived interest in data-driven learning based approaches to navigation. In this paper we propose to learn viewpoint invariant and target invariant visual servoing for local mobile robot navigation; given an initial view and the goal view or an image of a target, we train deep convolutional network controller to reach the desired goal. We present a new architecture for this task which rests on the ability of establishing correspondences between the initial and goal view and novel reward structure motivated by the traditional feedback control error. The advantage of the proposed model is that it does not require calibration and depth information and achieves robust visual servoing in a variety of environments and targets without any parameter fine tuning. We present comprehensive evaluation of the approach and comparison with other deep learning architectures as well as classical visual servoing methods in visually realistic simulation environment. The presented model overcomes the brittleness of classical visual servoing based methods and achieves significantly higher generalization capability compared to the previous learning approaches.

- Tightly-Coupled Single-Anchor Ultra-Wideband-Aided Monocular Visual Odometry System

    Author: Nguyen, Thien Hoang | Nanyang Technological University
    Author: Nguyen, Thien-Minh | Nanyang Technological University
    Author: Xie, Lihua | NanyangTechnological University
 
    keyword: Visual-Based Navigation; Sensor Fusion; Localization

    Abstract : In this work, we propose a tightly-coupled odometry framework, which combines monocular visual feature observations with distance measurements provided by a single ultra-wideband (UWB) anchor with an initial guess for its location. Firstly, the scale factor and the anchor position in the vision frame will be simultaneously estimated using a variant of Levenberg-Marquardt non-linear least squares optimization scheme. Once the scale factor is obtained, the map of visual features is updated with the new scale. Subsequent ranging errors in a sliding window are continuously monitored and the estimation procedure will be reinitialized to refine the estimates. Lastly, range measurements and anchor position estimates are fused when needed into a pose-graph optimization scheme to minimize both the landmark reprojection errors and ranging errors, thus reducing the visual drift and improving the system robustness. The proposed method is implemented in Robot Operating System (ROS) and can function in real-time. The performance of the proposed system is compared with state-of-the-art methods on both public datasets and real-life experiments.

- Scaling Local Control to Large-Scale Topological Navigation

    Author: Meng, Xiangyun | University of Washington
    Author: Ratliff, Nathan | Lula Robotics Inc
    Author: Xiang, Yu | NVIDIA
    Author: Fox, Dieter | University of Washington
 
    keyword: Visual-Based Navigation; Deep Learning in Robotics and Automation; Motion and Path Planning

    Abstract : Visual topological navigation has been revitalized recently thanks to the advancement of deep learning that substantially improves robot perception. However, the scalability and reliability issue remain challenging due to the complexity and ambiguity of real world images and mechanical constraints of real robots. We present an intuitive solution to show that by accurately measuring the capability of a local controller, large-scale visual topological navigation can be achieved while being scalable and robust. Our approach achieves state-of-the-art results in trajectory following and planning in large-scale environments. It also generalizes well to real robots and new environments without finetuning.

- Zero-Shot Imitation Learning from Demonstrations for Legged Robot Visual Navigation

    Author: Pan, Xinlei | UC Berkeley
    Author: Zhang, Tingnan | Google
    Author: Ichter, Brian | Google Brain
    Author: Faust, Aleksandra | Google Brain
    Author: Tan, Jie | Google
    Author: Ha, Sehoon | Google Brain
 
    keyword: Visual-Based Navigation; Learning from Demonstration; Legged Robots

    Abstract : Imitation learning is a popular approach for training visual navigation policies. However, collecting expert demonstrations for legged robots is challenging as these robots can be hard to control, move slowly, and cannot operate continuously for a long time. Here, we propose a zero-shot imitation learning approach for training a visual navigation policy on legged robots from human (third-person perspective) demonstrations, enabling high-quality navigation and cost-effective data collection. However, imitation learning from third-person demonstrations raises unique challenges. First, these demonstrations are captured from different camera perspectives, which we address via a feature disentanglement network(FDN) that extracts perspective-invariant state features. Second, as transition dynamics vary across systems, we label missing actions by either building an inverse model of the robot's dynamics in the feature space and applying it to the human demonstrations or developing a Graphic User Interface(GUI) to label human demonstrations. To train a navigation policy we use a model-based imitation learning approach with FDN and labeled human demonstrations. We show that our framework can learn an effective policy for a legged robot, Laikago, from human demonstrations in both simulated and real-world environments. Our approach is zero-shot as the robot never navigates the same paths during training as those at testing time. We justify our framework by performing a comparative study.

## Soft Robot Applications

- High Resolution Soft Tactile Interface for Physical Human-Robot Interaction

    Author: Huang, Isabella | UC Berkeley
    Author: Bajcsy, Ruzena | Univ of California, Berkeley
 
    keyword: Soft Robot Applications; Physical Human-Robot Interaction; Modeling, Control, and Learning for Soft Robots

    Abstract : If robots and humans are to coexist and cooperate in society, it would be useful for robots to be able to engage in tactile interactions. Touch is an intuitive communication tool as well as a fundamental method by which we assist each other physically. Tactile abilities are challenging to engineer in robots, since both mechanical safety and sensory intelligence are imperative. Existing work reveals a trade-off between these principles--- tactile interfaces that are high in resolution are not easily adapted to human-sized geometries, nor are they generally compliant enough to guarantee safety. On the other hand, soft tactile interfaces deliver intrinsically safe mechanical properties, but their non-linear characteristics render them difficult for use in timely sensing and control. We propose a robotic system that is equipped with a completely soft and therefore safe tactile interface that is large enough to interact with human upper limbs, while producing high resolution tactile sensory readings via depth camera imaging of the soft interface. We present and validate a data-driven model that maps point cloud data to contact forces, and verify its efficacy by demonstrating two real-world applications. In particular, the robot is able to react to a human finger's pokes and change its pose based on the tactile input. In addition, we also demonstrate that the robot can act as an assistive device that dynamically supports and follows a human forearm from underneath.

- Learning-Based Fingertip Force Estimation for Soft Wearable Hand Robot with Tendon-Sheath Mechanism

    Author: Cho, Kyu-Jin | Seoul National University, Biorobotics Laboratory
    Author: Jo, Sungho | Korea Advanced Institute of Science and Technology (KAIST)
    Author: Kang, Brian Byunghyun | Seoul National University
    Author: Kim, Daekyum | Korea Advanced Institute of Science and Technology
    Author: Choi, Hyungmin | Seoul National University
    Author: Jeong, Useok | Korea Institute of Industrial Technology (KITECH)
    Author: Kim, Kyu Bum | Seoul National University
 
    keyword: Soft Robot Applications; Wearable Robots; Modeling, Control, and Learning for Soft Robots

    Abstract : Soft wearable hand robots with tendon-sheath mechanisms are being actively developed to assist people who have lost their hand mobility. For these robots, accurately estimating fingertip forces can lead to successful object grasping. One way of estimating fingertip forces is to place sensors on the glove. However, directly placing placing sensors on the glove increases bulkiness and does not allow for water resistance. This results in a lack of user mobility when performing daily tasks. While another approach can utilize information like wire tension and motor encoder values, non-linearity and hysteresis with regards to the sheath bending angles and the dynamic changes of the angle displacement hinder accurate fingertip force estimation. This paper proposes a deep learning-based method to estimate fingertip forces by integrating dynamic information of motor encoders, wire tension, and sheath bending angles. The hardware system includes a soft under-actuated wearable robot, complete with an actuation system and a sensing system designed to measure sheath bending angles. The proposed approach was evaluated under criteria ranging from different object sizes, bending angle ranges, and forces. The results show that the system including the bending angle sensors and the proposed model can accurately estimate fingertip forces for the soft wearable hand robot.

- Autonomous and Reversible Adhesion Using Elastomeric Suction Cups for In-Vivo Medical Treatments

    Author: Iwasaki, Haruna | Waseda University
    Author: Lefevre, Flavien | ESEO
    Author: Damian, Dana | University of Sheffield
    Author: Iwase, Eiji | Waseda University
    Author: Miyashita, Shuhei | University of Sheffield
 
    keyword: Soft Robot Applications; Medical Robots and Systems; Grippers and Other End-Effectors

    Abstract : Remotely controllable and reversible adhesion is highly desirable for surgical operations: it can provide the possibility of non-invasive surgery, flexibility in fixing a patch and surgical manipulation via sticking. In our previous work, we developed a remotely controllable, ingestible, and deployable pill for use as a patch in the human stomach. In this study, we focus on magnetically facilitated reversible adhesion and develop a suction-based adhesive mechanism as a solution for non-invasive and autonomous adhesion of patches. We present the design, model, and fabrication of a magnet-embedded elastomeric suction cup. The suction cup can be localised, navigated, and activated or deactivated in an autonomous way; all realised magnetically with a pre-programmed fashion. The use of the adhesion mechanism is demonstrated for anchoring and carrying, for patching an internal organ surface and for an object removal, respectively.

- Design of an Inflatable Wrinkle Actuator with Fast Inflation/Deflation Responses for Wearable Suits

    Author: Park, Junghoon | KAIST
    Author: Choi, Junhwan | KAIST
    Author: Kim, Sangjoon J. | KAIST
    Author: Seo, Kap-Ho | Korea Institute of Robot and Convergence
    Author: Kim, Jung | KAIST
 
    keyword: Soft Robot Applications; Soft Sensors and Actuators; Wearable Robots

    Abstract : In recent years, inflatable actuators have been widely used in wearable suits to assist humans who need help in moving their joints. Despite their lightweight and simple structure, they have long inflation and deflation times, which make their quick use difficult. To resolve this issue, we propose an inflatable wrinkle actuator with fast inflation and deflation responses. First, a theoretical model is proposed to develop an actuator that satisfies the design requirements: the desired assistive torque and the foam factor based on the wearability. Second, we reduce the inflation and deflation times by partially controlling the actuator layers and by designing pneumatic circuits using a vacuum ejector. To validate the usability of the actuator in wearable suits, we applied it to a wearable knee suit, and the inflation and deflation times were 0.40 s and 0.16 s, respectively. As a result, we ensured that the actuator did not interfere with human knee joint movement during walking by creating any residual resistance.

- Design and Validation of a Soft Ankle-Foot Orthosis Exosuit for Inversion and Eversion Support

    Author: Thalman, Carly | Arizona State University
    Author: Lee, Hyunglae | Arizona State University
 
    keyword: Soft Robot Applications; Wearable Robots; Rehabilitation Robotics

    Abstract : This paper presents a soft robotic ankle-foot orthosis (SR-AFO) exosuit designed to provide support to the human ankle in the frontal plane without restricting natural motion in the sagittal plane. The SR-AFO exosuit incorporates inflatable fabric-based actuators with a hollow cylinder design which requires less volume than the commonly used solid cylinder design for the same deflection. The actuators were modeled and characterized using finite element analysis techniques and experimentally validated. The SR-AFO exosuit was evaluated on healthy participants in both a sitting position using a wearable ankle robot and a standing position using a dual-axis robotic platform to characterize the effect of the exosuit on the change of 2D ankle stiffness in the sagittal and frontal planes. For both sitting and standing test protocols, a trend of increasing ankle stiffness in the frontal plane was observed up to 50 kPa while stiffness in the sagittal plane remained relatively constant over pressure levels. During quiet standing, the exosuit could effectively change eversion stiffness at the ankle joint from about 20 to 70 Nm/rad at relatively low-pressure levels (&lt; 30 kPa). Eversion stiffness was 84.9 Nm/rad at 50 kPa, an increase of 387.5% from the original free foot stiffness.

- Vine Robots: Design, Teleoperation, and Deployment for Navigation and Exploration (I)
 
    Author: Coad, Margaret M. | Stanford University
    Author: Blumenschein, Laura | Stanford University
    Author: Cutler, Sadie | Brigham Young University
    Author: Reyna Zepeda, Javier | Stanford University
    Author: Naclerio, Nicholas | University of California, Santa Barbara
    Author: El-Hussieny, Haitham | Faculty of Engineering(Shoubra), Benha University
    Author: Mehmood, Usman | Korea University of Technology and Education
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
    Author: Okamura, Allison M. | Stanford University
 
    keyword: Soft Robot Applications; Field Robots

    Abstract : A new class of continuum robots has recently been explored, characterized by tip extension, significant length change, and directional control. Here, we call this class of robots "vine robots," due to their similar behavior to plants with the growth habit of trailing. Due to their growth-based movement, vine robots are well suited for navigation and exploration in cluttered environments, but until now, they have not been deployed outside the lab. Portability of these robots and steerability at length scales relevant for navigation are key to field applications. In addition, intuitive human-in-the-loop teleoperation enables movement in unknown and dynamic environments. We present a vine robot system that is teleoperated using a custom designed flexible joystick and camera system, long enough for use in navigation tasks, and portable for use in the field. We report on deployment of this system in two scenarios: a soft robot navigation competition and exploration of an archaeological site. The competition course required movement over uneven terrain, past unstable obstacles, and through a small aperture. The archaeological site required movement over rocks and through horizontal and vertical turns. The robot tip successfully moved past the obstacles and through the tunnels, demonstrating the capability of vine robots to achieve navigation and exploration tasks in the field.

- Pressure-Driven Manipulator with Variable Stiffness Structure

    Author: Sozer, Canberk | Scuola Superiore Sant'Anna
    Author: Patern�, Linda | The BioRobotics Institute, Scuola Superiore Sant'Anna
    Author: Tortora, Giuseppe | Scuola Superiore Sant'Anna
    Author: Menciassi, Arianna | Scuola Superiore Sant'Anna - SSSA
 
    keyword: Soft Robot Applications; Soft Robot Materials and Design; Flexible Robots

    Abstract : The high deformability and compliance of soft robots allow safer interaction with the environment. On the other hand, these advantages bring along controllability and predictability challenges which result in loss of force and stiffness output. Such challenges should be addressed in order to improve the overall functional performance and to meet the requirements of real-scenario applications. In this paper, we present a bidirectional in-plane manipulator which consists of two unidirectional fiber-reinforced actuators (FRAs) and a hybrid soft-rigid stiffness control structure (SCS), all of them controlled by air pressure. Both controllability and predictability of the manipulator are enhanced by the hybrid soft-rigid structure. While the FRAs provide positioning and position dependent stiffness, the SCS increases the stiffness of the manipulator without position dependency. The SCS is able to increase the manipulator stiffness by 35%, 30%, and 18%, when one FRA is pressurized at 150 kPa, 75 kPa, and 0 kPa, respectively. Experiments are carried out to present the feasibility of the proposed manipulator.

- 3D Electromagnetic Reconfiguration Enabled by Soft Continuum Robots

    Author: Gan, Lucia | Stanford University
    Author: Blumenschein, Laura | Stanford University
    Author: Huang, Zhe | University of Illinois at Urbana-Champaign
    Author: Okamura, Allison M. | Stanford University
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
    Author: Fan, Jonathan | Stanford University
 
    keyword: Soft Robot Applications; Soft Robot Materials and Design

    Abstract : The properties of radio frequency electromagnetic systems can be manipulated by changing the 3D geometry of the system. Most reconfiguration schemes specify different conductive pathways using electrical switches in mechanically static systems, or they actuate and reshape a continuous metallic structure. Here, we demonstrate a novel strategy that utilizes soft continuum robots to both dynamically assemble electrical pathways and mechanically reconfigure 3D electromagnetic devices. Our concept consists of using soft robotic actuation to conductively connect multiple subwavelength-scale metallic building blocks, which form into electromagnetic structures when joined together. Soft robots offer an exciting avenue for electromagnetic device construction because they can form complex, high curvature shapes from low-loss dielectric materials using straightforward manufacturing methods. As a proof of concept, we experimentally implement a helical antenna that can switch chirality through tendon actuation of a soft pneumatic continuum robot. Our work introduces a new paradigm for electromagnetic reconfiguration using soft robotic platforms.

- VaLeNS: Design of a Novel Variable Length Nested Soft Arm

    Author: Uppalapati, Naveen Kumar | University of Illinois at Urbana-Champaign
    Author: Krishnan, Girish | University of Illinois Urbana Champaign
 
    keyword: Soft Robot Applications; Soft Robot Materials and Design; Modeling, Control, and Learning for Soft Robots

    Abstract : Over the last decade, soft continuum arms (SCAs) have successfully demonstrated the compliance needed to operate in unstructured environments and handle fragile objects. However, their inherent soft compliance limits their performance in situations where stiffness and force transfer is required. In this letter, we present a compact design architecture, which is a hybrid between soft arms and rigid links known as Variable Length Nested Soft (VaLeNS) arm. The design architecture involves a novel SCA nested inside a concentric rigid tube. The SCA can undergo a combination of spatial bending (B) and bidirectional axial twist (R^2), and can extrude out or retract back into the rigid tube with varying length. The resulting configuration is shown to modulate stiffness up to a factor of ten and exhibits enhanced workspace and dexterity. Furthermore, the VaLeNS arm mounted on a rigid robotic platform allows for bifurcation of the overall workspace into rigid and soft, and can achieve high reachability in constrained environments. The paper demonstrates the effectiveness of the VaLeNS arm system in manipulation tasks that require both the rigid and soft attributes. This design architecture is deemed useful in agricultural applications and in physical human robot interaction.

- A Programmably Compliant Origami Mechanism for Dynamically Dexterous Robots

    Author: Chen, Wei-Hsi | University of Pennsylvania
    Author: Misra, Shivangi | University of Pennsylvania
    Author: Gao, Yuchong | University of Pennsylvania
    Author: Lee, Young-Joo | University of Pennsylvania
    Author: Koditschek, Daniel | University of Pennsylvania
    Author: Yang, Shu | University of Pennsylvania
    Author: Sung, Cynthia | University of Pennsylvania
 
    keyword: Soft Robot Applications; Soft Robot Materials and Design; Compliant Joint/Mechanism

    Abstract : We present an approach to overcoming challenges in dynamical dexterity for robots through tunable origami structures. Our work leverages a one-parameter family of flat sheet crease patterns that folds into origami bellows, whose axial compliance can be tuned to select desired stiffness. Concentrically arranged cylinder pairs reliably manifest additive stiffness, extending the tunable range by nearly an order of magnitude and achieving bulk axial stiffness spanning 200--1500 N/m using 8 mil thick polyester-coated paper. Accordingly, we design origami energy-storing springs with a stiffness of 1035 N/m each and incorporate them into a three degree-of-freedom (DOF) tendon-driven spatial pointing mechanism that exhibits trajectory tracking accuracy less than 15% rms error within a (~2 cm)^3 volume. The origami springs can sustain high power throughput, enabling the robot to achieve asymptotically stable juggling for both highly elastic (1kg resilient shot put ball) and highly damped (medicine ball) collisions in the vertical direction with apex heights approaching 10 cm. The results demonstrate that �soft'' robotic mechanisms are able to perform a controlled, dynamically actuated task.	

- Human Interface for Teleoperated Object Manipulation with a Soft Growing Robot

    Author: Stroppa, Fabio | Stanford University
    Author: Luo, Ming | Stanford University
    Author: Yoshida, Kyle | Stanford University
    Author: Coad, Margaret M. | Stanford University
    Author: Blumenschein, Laura | Stanford University
    Author: Okamura, Allison M. | Stanford University
 
    keyword: Soft Robot Applications; Human Factors and Human-in-the-Loop; Gesture, Posture and Facial Expressions

    Abstract : Soft growing robots are proposed for use in applications such as complex manipulation tasks or navigation in disaster scenarios. Safe interaction and ease of production promote the usage of this technology, but soft robots can be challenging to teleoperate due to their unique degrees of freedom. In this paper, we propose a human-centered interface that allows users to teleoperate a soft growing robot for manipulation tasks using arm movements. A study was conducted to assess the intuitiveness of the interface and the performance of our soft robot, involving a pick-and-place manipulation task. The results show that users were able to complete the task 97% of the time and achieve placement errors below 2 cm on average. These results demonstrate that our body-movement-based interface is an effective method for control of a soft growing robot manipulator.

## Prosthetics and Exoskeletons

- A Closed-Loop and Ergonomic Control for Prosthetic Wrist Rotation

    Author: Legrand, Mathilde | Institute for Intelligent Systems and Robotics, Sorbonne Univers
    Author: Jarrass�, Nathanael | Sorbonne Université, ISIR UMR 7222 CNRS
    Author: Richer, Florian | Cnrs - Isir
    Author: Morel, Guillaume | Sorbonne Université, CNRS, INSERM
 
    keyword: Prosthetics and Exoskeletons; Rehabilitation Robotics; Physical Human-Robot Interaction

    Abstract : Beyond the ultimate goal of prosthetics, repairing all the capabilities of amputees, the development line of upper-limb prostheses control mainly relies on three aspects: the robustness, the intuitiveness and the reduction of mental fatigue. Many complex structures and algorithms are proposed but no one question a common open-loop nature, where the user is the one in charge of correcting errors. Yet, closing the control loop at the prosthetic level may help to improve the three main lines of research cited above. One major issue to build a closed-loop control is the definition of a reliable error signal; this paper proposes to use body compensations, naturally exhibited by prostheses users when the motion of their device is inaccurate, as such. The described control scheme measures these compensatory movements and makes the prosthesis move in order to bring back the user into an ergonomic posture. The function of the prosthesis is no longer to perform a given motion but rather to correct the posture of its user while s/he focus on performing an endpoint task. This concept was validated and compared to a standard open-loop scheme, for the control of a prosthetic wrist, with five healthy subjects completing a dedicated task with a customized transradial prosthesis. Results show that the presented closed-loop control allows for more intuitiveness and less mental burden without enhancing body compensation.

- Comparison of Online Algorithms for the Tracking of Multiple Magnetic Targets in a Myokinetic Control Interface

    Author: Montero-Arag�n, Jordan | Scuola Superiore Sant'Anna
    Author: Gherardini, Marta | The Biorobotics Institute, Sant'Anna School of Advanced Studies
    Author: Clemente, Francesco | Scuola Superiore Sant'Anna
    Author: Cipriani, Christian | Scuola Superiore Sant'Anna
 
    keyword: Prosthetics and Exoskeletons; Localization; Optimization and Optimal Control

    Abstract : Magnetic tracking algorithms can be used to determine the position and orientation of specially designed magnetic markers or devices. These techniques are particularly interesting for biomedical applications such as teleoperated surgical robots or the control of upper limb prostheses. The performance of different algorithms used for magnetic tracking were compared in the past. However, in most cases, those algorithms were required to track a single MM. Here we investigated the performance of three localization algorithms in tracking up to 9 magnetic markers: two optimization-based (Levenberg-Marquardt algorithm, LMA, and Trust Region Reflective algorithm, TRRA) and one recursion-based (unscented Kalman Filter, UKF) algorithm. The tracking accuracy of the algorithms and their computation time were investigated through simulations. The accuracy of the three algorithms was similar, leading to estimation errors varying from a fraction of a millimeter, to a couple of millimeters. They allowed to accurately track up to six magnets with computation times under 300 ms for the UKF and 45 ms for the LMA/TRRA. The TRRA showed the best tracking performance overall. These outcomes are of interest for a wide range of robotics applications that require remote tracking.

- SIMPA: Soft-Grasp Infant Myoelectric Prosthetic Arm

    Author: De Barrie, Daniel | University of Lincoln
    Author: Margetts, Rebecca | University of Lincoln
    Author: Goher, Khaled | University of Lincoln
 
    keyword: Prosthetics and Exoskeletons; Soft Robot Applications; Additive Manufacturing

    Abstract :     Abstract� Myoelectric prosthetic arms have primarily focused on adults, despite evidence showing the benefits of early adoption. This work presents SIMPA, a low-cost 3D-printed prosthetic arm with soft grippers. The arm has been designed using CAD and 3D-scaning and manufactured using predominantly 3D-printing techniques. A voluntary opening control system utilising an armband based sEMG has been developed concurrently. Grasp tests have resulted in an average effectiveness of 87%, with objects in excess of 400g being securely grasped. The results highlight the effectiveness of soft grippers as an end device in prosthetics, as well as viability of toddler scale myoelectric devices.

- Backdrivable and Fully-Portable Pneumatic Back Support Exoskeleton for Lifting Assistance

    Author: Heo, Ung | KAIST
    Author: Kim, Sangjoon J. | KAIST
    Author: Kim, Jung | KAIST
 
    keyword: Prosthetics and Exoskeletons; Hydraulic/Pneumatic Actuators; Mechanism Design

    Abstract : To reduce the possibility of lower back pain (LBP), which is the most frequent injury in manual labor, several back support exoskeletons have been developed and implemented for lifting motion assistance. Although pneumatic power transmission is attractive due to its inherent compliance and backdrivability, the portability of the pneumatic system is highly limited due to the bulky air compressors that provide compressed air to the system. Therefore, we aimed to develop a fully-portable pneumatic back support exoskeleton by integrating all pneumatic components in the system. The compressed air consumption and generation of pneumatic system were modeled to meet design requirements. The developed exoskeleton was completely stand-alone and provides 80Nm of maximum extension torque for 6 liftings per minute (6l/m). The upper limit of the resistance torque was estimated to be about 2Nm, which implies high backdrivability. Finally, lifting experiments were performed and surface electromyography (sEMG) was measured to validate the physical assistance of the developed exoskeleton system for ten subjects. Compared to the case with no exoskeleton, the back muscle activation was significantly reduced with the assistances.

-  Clinical Readiness of a Myoelectric Postural Control Algorithm for Persons with Transradial Amputation (I)

    Author: Segil, Jacob | University of Colorado
    Author: Kaliki, Rahul | Infinite Biomedical Technologies
    Author: Uellendahl, Jack | Hanger Prosthetics and Orthotics
    Author: Weir, Richard | University of Colorado Denver | Anschutz Medical Campus

- Force Control of SEA-Based Exoskeletons for Multimode Human-Robot Interactions (I)

    Author: Huo, Weiguang | Imperial College London
    Author: Alouane, Mohamed Amine | Université Paris Est Cr�teil, France
    Author: Amirat, Yacine | University of Paris Est Cr�teil (UPEC)
    Author: Mohammed, Samer | University of Paris Est Cr�teil - (UPEC)
 
    keyword: Prosthetics and Exoskeletons; Force Control; Physical Human-Robot Interaction

    Abstract : In this article, a proxy-based force control method is proposed for three important human-robot interaction modes: zero-impedance mode, force assistive mode, and large force mode. A two-mass dynamic model-based nonlinear disturbance observer is used to meet the zero impedance output and accurate force tracking requirements with respect to disturbances from the wearer and environment. Additionally, significant force compliance can be achieved to guarantee the wearer's safety when the interaction torque is large. The proposed method is evaluated via experiments by comparison to the conventional proportional-integral-derivative and proxy-based sliding mode control methods. The results indicate that the proposed approach achieves better force tracking accuracy, robustness, and force compliance in three-mode human-robot interactions.

- Velocity Field Based Active-Assistive Control for Upper Limb Rehabilitation Exoskeleton Robot

    Author: Chia, En-Yu | National Taiwan University
    Author: Chen, Yi-Lian | National Taiwan University
    Author: Chien, Tzu-Chieh | National Taiwan University
    Author: Chiang, Ming-Li | National Taiwan University
    Author: Fu, Li-Chen | National Taiwan University
    Author: Lai, Jin-Shin | National Taiwan University
    Author: Lu, Lu | National Taiwan University
 
    keyword: Rehabilitation Robotics; Physical Human-Robot Interaction; Prosthetics and Exoskeletons

    Abstract : There are limitations of conventional active-assistive control for upper limb rehabilitation with help from exoskeleton robot, such as 1). prior time-dependent trajectories are generally required, 2). task-based rehabilitation exercise involving multi-joint motion is hard to implement, and 3). assistive mechanism normally is so inflexible that the resulting exercise performed by the subjects becomes inefficient. In this paper, we propose a novel velocity field based active-assistive control system to address these issues, which leads to much more efficient and precise rehabilitation compared with the existing schemes. First, we design a Kalman filter based interactive torque observer to obtain subjects' active intention of motion. Next, a joint-position-dependent velocity field which can be automatically generated via the task motion pattern is proposed to provide the time-independent assistance to the subjects. We further propose an integration method that combines the active and assistive motions based on the performance and the involvement of subjects to guide them to perform the task more voluntarily and precisely. The experiment results show that both the execution time and the subjects' torque exertion are reduced while performing both given single joint tasks and task-oriented multi-joint tasks as compared with the related work in the literature.

- Design, Development and Control of a Tendon-Actuated Exoskeleton for Wrist Rehabilitation and Training

    Author: Dragusanu, Mihai | University of Siena
    Author: Lisini Baldi, Tommaso | University of Siena
    Author: Iqbal, Muhammad Zubair | University of Siena
    Author: Prattichizzo, Domenico | Université Di Siena
    Author: Malvezzi, Monica | University of Siena
 
    keyword: Rehabilitation Robotics; Physically Assistive Devices; Health Care Management

    Abstract : Robot rehabilitation is an emerging and promising topic that incorporates robotics with neuroscience and rehabilitation to define new methods for supporting patients with neurological diseases. As a consequence, the rehabilitation process could increase the efficacy exploiting the potentialities of robot-mediated therapies. Nevertheless, nowadays clinical effectiveness is not enough to widely introduce robotic technologies in such social contexts. In this paper we propose a step further, presenting an innovative exoskeleton for wrist flexion/extension and adduction/abduction motion training. It is designed to be wearable and easy to control and manage. It can be used by the patient in collaboration with the therapist or autonomously. The paper introduces the main steps of device design and development and presents some tests conducted with an user with limited wrist mobility.

- Impedance Control of a Transfemoral Prosthesis Using Continuously Varying Ankle Impedances and Multiple Equilibria

    Author: Anil Kumar, Namita | Texas A&amp;M University College Station
    Author: Hong, Woolim | Texas A&amp;M University
    Author: Hur, Pilwon | Texas A&amp;M University
 
    keyword: Prosthetics and Exoskeletons; Compliance and Impedance Control; Rehabilitation Robotics

    Abstract : Impedance controllers are popularly used in the field of lower limb prostheses and exoskeleton development. Such controllers assume the joint to be a spring-damper system described by a discrete set of equilibria and impedance param- eters. These parameters are estimated via a least squares opti- mization that minimizes the difference between the controller's output torque and human joint torque. Other researchers have used perturbation studies to determine empirical values for ankle impedance. The resulting values vary greatly from the prior least squares estimates. While perturbation studies are more credible, they require immense investment. This paper extended the least squares approach to reproduce the results of perturbation studies. The resulting ankle impedance parameters were successfully tested on a powered transfemoral prosthesis, AMPRO II. Further, the paper investigated the effect of multiple equilibria on the least squares estimation and the performance of the impedance controller. Finally, the paper uses the proposed least squares optimization method to estimate knee impedance.

- Towards Variable Assistance for Lower Body Exoskeletons

    Author: Gurriet, Thomas | California Institute of Technology
    Author: Tucker, Maegan | California Institute of Technology
    Author: Duburcq, Alexis | Wandercraft
    Author: Boeris, Guilhem | Wandercraft
    Author: Ames, Aaron | Caltech
 
    keyword: Prosthetics and Exoskeletons; Rehabilitation Robotics; Formal Methods in Robotics and Automation

    Abstract : This paper presents and experimentally demonstrates a novel framework for variable assistance on lower body exoskeletons, based upon safety-critical control methods. Existing work has shown that providing some freedom of movement around a nominal gait, instead of rigidly following it, accelerates the spinal learning process of people with a walking impediment when using a lower body exoskeleton. With this as motivation, we present a method to accurately control how much a subject is allowed to deviate from a given gait while ensuring robustness to patient perturbation. This method leverages control barrier functions to force certain joints to remain inside predefined trajectory tubes in a minimally invasive way. The effectiveness of the method is demonstrated experimentally with able-bodied subjects and the Atalante lower body exoskeleton.

- Offline Assistance Optimization of a Soft Exosuit for Augmenting Ankle Power of Stroke Survivors During Walking

    Author: Siviy, Christopher | Harvard University School of Engineering and Applied Sciences
    Author: Bae, Jaehyun | Apple Inc
    Author: Baker, Lauren | Harvard SEAS
    Author: Porciuncula, Franchino | Harvard SEAS
    Author: Baker, Teresa | Boston University
    Author: Ellis, Terry | Boston University
    Author: Awad, Louis | Harvard University
    Author: Walsh, Conor James | Harvard University
 
    keyword: Prosthetics and Exoskeletons; Rehabilitation Robotics; Wearable Robots

    Abstract : Locomotor impairments afflict more than 80% of people poststroke. Our group has previously developed a unilateral ankle exosuit aimed at assisting the paretic ankle joint of stroke survivors during walking. While studies to date have shown promising biomechanical and physiological changes, there remains opportunity to better understand how changes in plantarflexion (PF) assistance profiles impact wearer response. In healthy populations, studies explicitly varying augmentation power have been informative about how exosuit users are sensitive to changes in PF assistance; however there are challenges in applying existing methods to a medical population where significantly higher gait variability and limited walking capacity exist. This paper details an offline assistance optimization scheme that uses previously-recorded biomechanics data to generate torque profiles designed to deliver either positive or negative augmentation power in PF while being less sensitive to stride-by-stride variability. Additionally, we describe an admittance-control strategy that can effectively deliver PF force with RMS error less than 10 N. A preliminary study on six people poststroke demonstrates that offline assistance optimization can successfully isolate positive and negative augmentation power. Moreover, we show that in people poststroke, positive augmentation power effected changes in total positive ankle power while delivering negative augmentation power had no effect on total negative ankle p

- Gait Patterns Generation Based on Basis Functions Interpolation for the TWIN Lower-Limb Exoskeleton

    Author: Vassallo, Christian | Istituto Italiano Di Tecnologia
    Author: De Giuseppe, Samuele | Istituto Italiano Di Tecnologia
    Author: Piezzo, Chiara | Italian Institute of Technology
    Author: Maludrottu, Stefano | Italian Institute of Technology
    Author: Cerruti, Giulio | IIT - Italian Institute of Technology
    Author: D'Angelo, Maria Laura | Istituto Italiano Di Tecnologia
    Author: Gruppioni, Emanuele | INAIL Prosthesis Center
    Author: Marchese, Claudia | Centro Protesi INAIL, Vigorso Di Budrio
    Author: Castellano, Simona | Centro Protesi INAIL, Vigorso Di Budrio
    Author: Guanziroli, Eleonora | Valduce Hospital - Como
    Author: Molteni, Franco | Hospital Valduce - Villa Beretta, Via Nazario Sauro 17, 23845 Cos
    Author: Laffranchi, Matteo | Istituto Italiano Di Tecnologia
    Author: De Michieli, Lorenzo | Istituto Italiano Di Tecnologia
 
    keyword: Prosthetics and Exoskeletons; Medical Robots and Systems; Wearable Robots

    Abstract : Since the uprising of new biomedical orthotic devices, exoskeletons have been put in the spotlight for their possible use in rehabilitation. Even if these products might share some commonalities among them in terms of overall structure, degrees of freedom and possible actions, they quite often differ in their approach on how to generate a feasible, stable and comfortable gait trajectory pattern. This paper introduces three proposed trajectories that were generated by using a basis function interpolation method and by working closely with two major rehabilitation centers in Italy. The whole procedure has been focused on the concepts of a configurable walk for patients that suffer from spinal cord injuries. We tested the solutions on a group of healthy volunteers and on a spinal-cord injury patient with the use of the new TWIN exoskeleton developed at the Rehab Technologies Lab at the Italian Institute of Technology.

- Modulating Hip Stiffness with a Robotic Exoskeleton Immediately Changes Gait

    Author: Lee, Jongwoo | Massachusetts Institute of Technology (MIT)
    Author: Warren, Haley | University of Vermont
    Author: Agarwal, Vibha | MIT
    Author: Huber, Meghan | University of Massachusetts Amherst
    Author: Hogan, Neville | Massachusetts Institute of Technology
 
    keyword: Prosthetics and Exoskeletons; Wearable Robots; Physical Human-Robot Interaction

    Abstract : Restoring healthy kinematics is a critical component of assisting and rehabilitating impaired locomotion. Here we tested whether spatio-temporal gait patterns can be modulated by applying mechanical impedance to hip joints. Using the Samsung GEMS-H exoskeleton, we emulated a virtual spring (positive and negative) between the user's legs. We found that applying positive stiffness with the exoskeleton decreased stride time and hip range of motion for healthy subjects during treadmill walking. Conversely, the application of negative stiffness increased stride time and hip range of motion. These effects did not vary over long nor short repeated exposures to applied stiffness. In addition, minimal transient behavior was observed in spatio-temporal measures of gait when the stiffness controller transitioned between on and off states. These results suggest that changes in gait behavior induced by applying hip stiffness were purely a mechanical effect. Together, our findings indicate that applying mechanical impedance using lower-limb assistive devices may be an effective, minimally-encumbering intervention to restore healthy gait patterns.

- Swing-Assist for Enhancing Stair Ambulation in a Primarily-Passive Knee Prosthesis

    Author: Lee, Jantzen | Vanderbilt University
    Author: Goldfarb, Michael | Vanderbilt University
 
    keyword: Prosthetics and Exoskeletons; Rehabilitation Robotics; Physically Assistive Devices

    Abstract :     Abstract�This paper presents the design and implementation of a controller for stair ascent and descent in a primarily-passive stance-controlled swing-assist (SCSA) prosthesis. The prosthesis and controller enable users to perform both step-over and step-to stair ascent and descent. The efficacy of the controller and SCSA prosthesis prototype in providing improved stair ambulation was tested on a unilateral transfemoral amputee in experiments that employed motion capture apparatus to compare joint kinematics with the SCSA prosthesis, relative to performing the same activity with a microprocessor-controlled daily-use passive prosthesis. Results suggest that the SCSA knee significantly decreases compensatory motion during stair activity when compared to the passive prosthesis.

- Proof-Of-Concept of a Pneumatic Ankle Foot Orthosis Powered by a Custom Compressor for Drop Foot Correction

    Author: Kim, Sangjoon J. | KAIST
    Author: Park, Junghoon | KAIST
    Author: Shin, Wonseok | KAIST
    Author: Lee, Dong Yeon | Seoul National University Hospital
    Author: Kim, Jung | KAIST
 
    keyword: Prosthetics and Exoskeletons; Hydraulic/Pneumatic Actuators; Human Performance Augmentation

    Abstract : Pneumatic transmission has several advantages in developing powered ankle foot orthosis (AFO) systems, such as the flexibility in placing pneumatic components for mass distribution and providing high back-drivability via simple valve control. However, pneumatic systems are generally tethered to large stationary air compressors that restrict them for being used as daily assistive devices. In this study, we improved a previously developed wearable (untethered) custom compressor that can be worn (1.5 kg) at the waist of the body and can generate adequate amount of pressurized air (maximum pressure of 1050 kPa and a flow rate of 15.1 mL/sec at 550 kPa) to power a unilateral active AFO used to assist the dorsiflexion (DF) motion of drop-foot patients. The finalized system can provide a maximum assistive torque of 10 Nm and induces an average 0.03�0.06 Nm resistive torque when free movement is provided. The system was tested for two hemiparetic drop-foot patients. The proposed system showed an average improvement of 13.6� of peak dorsiflexion angle during the swing phase of the gait cycle.

- Knowledge-Guided Reinforcement Learning Control for Robotic Lower Limb Prosthesis

    Author: Gao, Xiang | Arizona State University
    Author: Si, Jennie | Arizona State University
    Author: Wen, Yue | University of North Carolina at Chapel Hill
    Author: Li, Minhan | North Carolina State University
    Author: Huang, He (Helen) | North Carolina State University and University of North Carolina
 
    keyword: Prosthetics and Exoskeletons; Learning and Adaptive Systems; AI-Based Methods

    Abstract :     Abstract--- Robotic prostheses provide new opportunities to better restore the lost functions than passive prostheses for transfemoral amputees. But controlling a prosthesis device automatically for individual users in different task environments is an unsolved problem. Reinforcement learning (RL) is a naturally promising tool. For prosthesis control with a user in the loop, it is desirable that the controlled prosthesis can adapt to different task environments as quickly and smoothly as possible. However, most RL agents learn or relearn from scratch when the environment changes. To address this issue, we propose the knowledge-guided Q-learning (KG-QL) control method as a principled way for the problem. In this report, we collected and used data from two able-bodied (AB) subjects wearing a RL controlled robotic prosthetic limb walking on level ground. Our ultimate goal is to build an efficient RL controller with reduced time and data requirement and transfer knowledge from AB subjects to amputee subjects. Toward this goal, we demonstrate its feasibility by employing OpenSim, a well-established human locomotion simulator. Our results show the OpenSim simulated amputee subject improved control tuning performance over learning from scratch by utilizing knowledge transfer from AB subjects. Also in this paper, we will explore the possibility of information transfer from AB subjects to help tuning for the amputee subjects.

- Development of a Twisted String Actuator-Based Exoskeleton for Hip Joint Assistance in Lifting Tasks

    Author: Seong, Hyeonseok | Korea University of Technology
    Author: Kim, Do-Hyeong | KAIST
    Author: Gaponov, Igor | Innopolis University
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
 
    keyword: Prosthetics and Exoskeletons; Physically Assistive Devices; Tendon/Wire Mechanism

    Abstract : This paper presents a study on a compliant cable-driven exoskeleton for hip assistance in lifting tasks that is aimed at preventing low-back pain and injuries in the vocational setting. In the proposed concept, we used twisted string actuator (TSA) to design a light-weight and powerful exoskeleton that benefits from inherent TSA advantages. We have noted that nonlinear nature of twisted strings�transmission ratio (decreasing with twisting) closely matched typical torque-speed requirements for hip assistance during lifting tasks and tried to use this fact in the exoskeleton design and motor selection. Hip-joint torque and speed required to lift a 10-kg load from stoop to stand were calculated, which gave us a baseline that we used to design and manufacture a practical exoskeleton prototype. Preliminary experimental trials demonstrated that the proposed device was capable of generating required torque and speed at the hip joint while weighing under 6 kg,including battery.

- A Novel Portable Lower Limb Exoskeleton for Gravity Compensation During Walking

    Author: Zhou, Libo | Beihang University
    Author: Chen, Weihai | Beihang University
    Author: Chen, Wenjie | Singapore Inst. of Manufacturing Technology
    Author: Bai, Shaoping | Aalborg University
    Author: Wang, Jianhua | Beijing University of Aeronautics and Astronautics
 
    keyword: Prosthetics and Exoskeletons; Physically Assistive Devices; Rehabilitation Robotics

    Abstract : This paper presents a novel portable passive lower limb exoskeleton for walking assistance. The exoskeleton is designed with built-in spring mechanisms at the hip and knee joints to realize gravity balancing of the human leg. A pair of mating gears is used to convert the tension force from the built-in springs into balancing torques at hip and knee joints for overcoming the influence of gravity. Such a design makes the exoskeleton has a compact layout with small protrusion, which improves its safety and user acceptance. In this paper, the design principle of gravity balancing is described. Simulation results show a significant reduction of driving torques at the limb joints. A prototype of single leg exoskeleton has been constructed and preliminary test results show the effectiveness of the exoskeleton.

## Human-Centered Robotics

- Human-Centric Active Perception for Autonomous Observation

    Author: Kent, David | Georgia Institute of Technology
    Author: Chernova, Sonia | Georgia Institute of Technology
 
    keyword: Space Robotics and Automation; Human-Centered Robotics; Planning, Scheduling and Coordination

    Abstract : As robot autonomy improves, robots are increasingly being considered in the role of autonomous observation systems - free-flying cameras capable of actively tracking human activity within some predefined area of interest. In this work, we formulate the autonomous observation problem through multi-objective optimization, presenting a novel Semi-MDP formulation of the autonomous human observation problem that maximizes observation rewards while accounting for both human- and robot-centric costs. We demonstrate that the problem can be solved with both scalarization-based Multi-Objective MDP methods and Constrained MDP methods, and discuss the relative benefits of each approach. We validate our work on activity tracking using a NASA Astrobee robot operating within a simulated International Space Station environment.

- Prediction of Human Full-Body Movements with Motion Optimization and Recurrent Neural Networks

    Author: Kratzer, Philipp | University of Stuttgart
    Author: Toussaint, Marc | University of Stuttgart
    Author: Mainprice, Jim | Max Planck Institute
 
    keyword: Human-Centered Robotics; Optimization and Optimal Control; Deep Learning in Robotics and Automation

    Abstract : Human movement prediction is difficult as humans naturally exhibit complex behaviors that can change drastically from one environment to the next. In order to alleviate this issue, we propose a prediction framework that decouples short-term prediction, linked to internal body dynamics, and long-term prediction, linked to the environment and task constraints. In this work we investigate encoding short-term dynamics in a recurrent neural network, while we account for environmental constraints, such as obstacle avoidance, using gradient-based trajectory optimization. Experiments on real motion data demonstrate that our framework improves the prediction with respect to state-of-the-art motion prediction methods, as it accounts to beforehand unseen environmental structures. Moreover we demonstrate on an example, how this framework can be used to plan robot trajectories that are optimized to coordinate with a human partner.

- Predicting and Optimizing Ergonomics in Physical Human-Robot Cooperation Tasks

    Author: van der Spaa, Linda F | Delft University of Technology
    Author: Gienger, Michael | Honda Research Institute Europe
    Author: Bates, Tamas | Technical University of Delft
    Author: Kober, Jens | TU Delft
 
    keyword: Human Factors and Human-in-the-Loop; Physical Human-Robot Interaction; Human-Centered Robotics

    Abstract : This paper presents a method to incorporate ergonomics into the optimization of action sequences for bi-manual human-robot cooperation tasks with continuous physical interaction. Our first contribution is a novel computational model of the human that allows prediction of an ergonomics assessment corresponding to each step in a task. The model is learned from human motion capture data in order to predict the human pose as realistically as possible. The second contribution is a combination of this prediction model with an informed graph search algorithm, which allows computation of human-robot cooperative plans with improved ergonomics according to the incorporated method for ergonomic assessment. The concepts have been evaluated in simulation and in a small user study in which the subjects manipulate a large object with a 32 DoF bimanual mobile robot as partner. For all subjects, the ergonomic-enhanced planner shows their reduced ergonomic cost compared to a baseline planner.

- Active Reward Learning for Co-Robotic Vision Based Exploration in Bandwidth Limited Environments

    Author: Jamieson, Stewart | Massachusetts Institute of Technology
    Author: How, Jonathan Patrick | Massachusetts Institute of Technology
    Author: Girdhar, Yogesh | Woods Hole Oceanographic Institution
 
    keyword: Human Factors and Human-in-the-Loop; Learning and Adaptive Systems; Marine Robotics

    Abstract : We present a novel POMDP problem formulation for a robot that must autonomously decide where to go to collect new and scientifically relevant images given a limited ability to communicate with its human operator. From this formulation we derive constraints and design principles for the observation model, reward model, and communication strategy of such a robot, exploring techniques to deal with the very high-dimensional observation space and scarcity of relevant training data. We introduce a novel active reward learning strategy based on making queries to help the robot minimize path "regret" online, and evaluate it for suitability in autonomous visual exploration through simulations. We demonstrate that, in some bandwidth-limited environments, this novel regret-based criterion enables the robotic explorer to collect up to 17% more reward per mission than the next-best criterion.

- Characterizing User Responses to Failures in Aerial Autonomous Systems

    Author: Kunde, Siya | University of Nebraska
    Author: Elbaum, Sebastian | University of Virginia
    Author: Duncan, Brittany | University of Nebraska, Lincoln
 
    keyword: Human Factors and Human-in-the-Loop; Human-Centered Automation; Human-Centered Robotics

    Abstract : Users are often the last barrier in the detection and correction of abnormal behavior in autonomous systems, so understanding what can be expected from users in various contexts is crucial to the performance of such systems. This paper presents the first study characterizing a user ability to timely report and correct autonomous system failures in the context of small Unmanned Aerial Vehicles (sUAVs). The study aims to explore the complex tradespace which designers will encounter when developing effective transitions of control in sUAVs. We have analyzed these tradeoffs in terms of accuracy and response time while manipulating several key contextual elements including subtlety of failure, transitions of control, introduction of noise in autonomous paths, and amount of user training. Results indicate that: 1) increased accuracy is achieved under longer deadlines without large delays in responses, 2) increased noise, user training, and user responsibility for correction lead to increased reporting times, but only increased user responsibility increased accuracy, 3) users, particularly those with additional training, wanted to remain engaged in failure correction even when such interactions were not requested, 4) asking users to fix failures that they did not report resulted in increased response times and reduced accuracy.

- VariPath: A Database for Modelling the Variance of Human Pathways in Manual and HRC Processes with Heavy-Duty Robots

    Author: Bdiwi, Mohamad | Fraunhofer Institute for Machine Tools and Forming Technology IW
    Author: Harsch, Ann-Kathrin | Fraunhofer Institute for Machine Tools and Forming Technology
    Author: Reindel, Paul | Fraunhofer Institute for Machine Tools and Forming Technology
    Author: Putz, Matthias | Fraunhofer Institute for Machine Tools and Forming Technology IW
 
    keyword: Human Factors and Human-in-the-Loop; Human-Centered Robotics; Cognitive Human-Robot Interaction

    Abstract : Unlike robots, humans do not have constant movements. Their pathways are individually changeable and influenced by circumstances. This paper presents a method to investigate human pathway variations in a real study. In systematically selected tasks, human pathways are examined for 100 participants in manual and human-robot collaboration (HRC) scenarios. As a result, the variations of pathways are presented depending on various features: e.g. in nearly all cases the variance of women's walking pathways is smaller than that of men. VariPath database can be used in any planning process of manual or HRC scenarios to ensure safety and efficiency.

- Congestion-Aware Evacuation Routing Using Augmented Reality Devices

    Author: Zhang, Zeyu | UCLA
    Author: Liu, Hangxin | University of California, Los Angeles
    Author: Jiao, Ziyuan | University of California, Los Angeles
    Author: Zhu, Yixin | University of California, Los Angeles
    Author: Zhu, Song-Chun | UCLA
 
    keyword: Virtual Reality and Interfaces

    Abstract : We present a congestion-aware routing solution for indoor evacuation, which produces real-time individual-customized evacuation routes among multiple destinations while keeping tracks of all evacuees' locations. A population density map, obtained on-the-fly by aggregating locations of evacuees from user-end Augmented Reality (AR) devices, is used to model the congestion distribution inside a building. To efficiently search the evacuation route among all destinations, a variant of A* algorithm is devised to obtain the optimal solution in a single pass. In a series of simulated studies, we show that the proposed algorithm is more computationally optimized compared to classic path planning algorithms; it generates a more time-efficient evacuation route for each individual that minimizes the overall congestion. A complete system using AR devices is implemented for a pilot study in real-world environments, demonstrating the efficacy of the proposed approach.

- Human-Robot Interaction for Robotic Manipulator Programming in Mixed Reality

    Author: Ostanin, Mikhail | Innopolis University
    Author: Mikhel, Stanislav | Innopolis University
    Author: Evlampiev, Alexey | Innopolis University
    Author: Skvortsova, Valeria | Innopolis University
    Author: Klimchik, Alexandr | Innopolis University
 
    keyword: Virtual Reality and Interfaces; Industrial Robots

    Abstract : The paper presents an approach for interactive programming of the robotic manipulator using mixed reality. The developed system is based on the HoloLens glasses connected through Robotic Operation System to Unity engine and robotic manipulators. The system gives a possibility to recognize the real robot location by the point cloud analysis, to use virtual markers and menus for the task creation, to generate a trajectory for execution in the simulator or on the real manipulator. It also provides the possibility of scaling virtual and real worlds for more accurate planning. The proposed framework has been tested on pick-and-place and contact operations execution by UR10e and KUKA iiwa robots.

- Heart Rate Sensing with a Robot Mounted mmWave Radar

    Author: Zhao, Peijun | University of Oxford
    Author: Lu, Chris Xiaoxuan | University of Oxford
    Author: Wang, Bing | University of Oxford
    Author: Chen, Changhao | University of Oxford
    Author: Xie, Linhai | University of Oxford
    Author: Wang, Mengyu | Peking University
    Author: Trigoni, Niki | University of Oxford
    Author: Markham, Andrew | Oxford University
 
    keyword: Human-Centered Robotics; Service Robots; Human Factors and Human-in-the-Loop

    Abstract : Heart-rate monitoring at home is a useful metric for assessing health e.g. of the elderly or patients in post-operative recovery. Although non-contact heart-rate monitoring has been widely explored, typically using a static, wall-mounted device, measurements are limited to a single room and sensitive to user orientation and position. In this work, we propose mBeats, a robot mounted millimeter wave (mmWave) radar system that provide periodic heart-rate measurements under different user poses, without interfering in a user's daily activities. mBeats contains a mmWave servoing module that adaptively adjusts the sensor angle to the best reflection profile. Furthermore, mBeats features a deep neural network predictor, which can estimate heart rate from the lower leg and additionally provides estimation uncertainty. Through extensive experiments, we demonstrate accurate and robust operation of mBeats in a range of scenarios. We believe by integrating mobility and adaptability, mBeats can empower many downstream healthcare applications at home, such as palliative care, post-operative rehabilitation and telemedicine.

- VibeRo: Vibrotactile Stiffness Perception Interface for Virtual Reality

    Author: Adilkhanov, Adilzhan | Nazarbayev University
    Author: Yelenov, Amir | Nazarbayev University
    Author: Singal Reddy, Ramakanth | ISIR, UPMC, Paris
    Author: Terekhov, Alexander V. | Paris Descartes University
    Author: Kappassov, Zhanat | Pierre and Marie Curie University
 
    keyword: Virtual Reality and Interfaces; Haptics and Haptic Interfaces; Force and Tactile Sensing

    Abstract : Haptic interfaces allow a more realistic experience with Virtual Reality (VR). They are used to manipulate virtual objects. These objects can be rigid and soft. In this letter, we have designed and evaluated a vibrotactile hand-held device (VibeRo) to achieve haptic cues of different soft objects. The proposed method is based on combining the vision-driven displacement produced by the pseudo-haptics effect with the haptic illusion of a limb displacement obtained from force-driven synthesis of vibration of a contact surface. VibeRo features a voice-coil actuator and force-sensitive resistors for generating squeeze forces at fingertips. We present an evaluation of VibeRo's pseudo-haptic and haptic illusion effects to render soft virtual objects. The efficacy of the approach was validated in experiments with human subjects.

- Detachable Body: The Impact of Binocular Disparity and Vibrotactile Feedback in Co-Presence Tasks

    Author: Iwasaki, Yukiko | Waseda University
    Author: Ando, Kozo | Waseda University
    Author: Iizuka, Shuhei | Waseda
    Author: Kitazaki, Michiteru | Toyohashi University of Technology
    Author: Iwata, Hiroyasu | Waseda University
 
    keyword: Virtual Reality and Interfaces; Human Performance Augmentation; Telerobotics and Teleoperation

    Abstract : Detachable Body is a new concept of a robot arm wearable as an extended part of the body. It can be detached from the user's natural body; it can be attached not only to another person but also anywhere in the environment. Humans can eventually perform tasks that involve co-presence, or tasks that are concurrent and performed in two separate places, by utilizing the Detachable Body. In this paper, we design an information presentation interface to concurrently manage both the natural and detached bodies that are located in two separate locations. The interface consists of a vision presentation system that superimposes two environment images with binocular disparity, and a proprioception presentation system that provides somatosensory feedback of the detached arm's position. The usability of the proposed interface was evaluated by measuring work efficiency and subjective evaluation in a task involving co-presence. The results suggest the existence of the effects of the binocular disparity in the vision presentation system and the tactile information provided via feedback.

- Prediction of Gait Cycle Percentage Using Instrumented Shoes with Artificial Neural Networks

    Author: Prado, Antonio | Columbia University
    Author: Cao, Xiya | Peking University
    Author: Ding, Xiangzhuo | Columbia University
    Author: Agrawal, Sunil | Columbia University
 
    keyword: Human Detection and Tracking; Rehabilitation Robotics; Deep Learning in Robotics and Automation

    Abstract : Gait training is widely used to treat gait abnormalities. Traditional gait measurement systems are limited to instrumented laboratories. Even though gait measurements can be made in these settings, it is challenging to estimate gait parameters robustly in real-time for gait rehabilitation, especially when walking over-ground. In this paper, we present a novel approach to track the continuous gait cycle during overground walking outside the laboratory. In this approach, we instrument standard footwear with a sensorized insole and an inertial measurement unit. Artificial neural networks are used on the raw data obtained from the insoles and IMUs to compute the continuous percentage of the gait cycle for the entire walking session. We show in this paper that when tested with novel subjects, we can predict the gait cycle with a Root Mean Square Error (RMSE) of 7.2%. The onset of each cycle can be detected within an RMSE time of 41.5 ms with a 99% detection rate. The algorithm was tested with 18840 strides collected from 24 adults. In this paper, we tested a combination of fully-connected layers, an Encoder-Decoder using convolutional layers, and recurrent layers to identify an architecture that provided the best performance.

- Perception-Action Coupling in Usage of Telepresence Cameras

    Author: Valiton, Alexandra | Worcester Polytechnic Institute
    Author: Li, Zhi | Worcester Polytechnic Institute
 
    keyword: Telerobotics and Teleoperation; Human Factors and Human-in-the-Loop; Human-Centered Robotics

    Abstract :  Telepresence tele-action robots enable human workers to reliably perform difficult tasks in remote, cluttered, and human environments. However, the effort to control co- ordinated manipulation and active perception motions may exhaust and intimidate novice workers. We hypothesize that such cognitive efforts would be effectively reduced if the teleoperators are provided with autonomous camera selection and control aligned with the natural perception-action coupling of the human motor system. Thus, we conducted a user study to investigate the coordination of active perception control and manipulation motions performed with visual feedback from various wearable and standalone cameras in a telepresence scenario. Our study discovered rich information about telepresence camera selection to inform telepresence system configuration and possible teleoperation assistance design for reduced cognitive effort in robot teleoperation.

- A Technical Framework for Human-Like Motion Generation with Autonomous Anthropomorphic Redundant Manipulators

    Author: Averta, Giuseppe | University of Pisa
    Author: Caporale, Danilo | Centro Di Ricerca E. Piaggio
    Author: Della Santina, Cosimo | Massachusetts Institute of Technology
    Author: Bicchi, Antonio | Université Di Pisa
    Author: Bianchi, Matteo | University of Pisa
 
    keyword: Natural Machine Motion; Humanoid Robots; Human-Centered Robotics

    Abstract : The need for users' safety and technology accept- ability has incredibly increased with the deployment of co-bots physically interacting with humans in industrial settings, and for people assistance. A well-studied approach to meet these requirements is to ensure human-like robot motions. Classic solutions for anthropomorphic movement generation usually rely on optimization procedures, which build upon hypotheses devised from neuroscientific literature, or capitalize on learning methods. However, these approaches come with limitations, e.g. limited motion variability or the need for high dimensional datasets. In this work, we present a technique to directly embed human upper limb principal motion modes computed through functional analysis in the robot trajectory optimization. We report on the implementation with manipulators with redundant anthropomorphic kinematic architectures - although dissimilar with respect to the human model used for functional mode extraction - via Cartesian impedance control. In our experiments, we show how human trajectories mapped onto a robotic manipulator still exhibit the main characteristics of human-likeness, e.g. low jerk values. We discuss the results with respect to the state of the art, and their implications for advanced human-robot interaction in industrial co-botics and for human assistance.

- Real-Time Adaptive Assembly Scheduling in Human-Multi-Robot Collaboration According to Human Capability

    Author: Zhang, Shaobo | Chang�an University
    Author: Chen, Yi | Clemson University
    Author: Zhang, Jun | Chang�an University
    Author: Jia, Yunyi | Clemson University
 
    keyword: Assembly; Human Factors and Human-in-the-Loop

    Abstract : Human-multi-robot collaboration is becoming more and more common in intelligent manufacturing. Optimal assembly scheduling of such systems plays a critical role in their production efficiency. Existing approaches mostly consider humans as agents with assumed or known capabilities, which leads to suboptimal performance in realistic applications where human capabilities usually change. In addition, most robot adaptation focuses on human-single-robot interaction and the adaptation in human-multi-robot interaction with changing human capability still remains challenging due to the complexity of the heterogeneous multi-agent interactions. This paper proposes a real-time adaptive assembly scheduling approach for human-multi-robot collaboration by modeling and incorporating changing human capability. A genetic algorithm is also designed to derive implementable solutions for the formulated adaptive assembly scheduling problem. The proposed approaches are validated through different simulated human-multi-robot assembly tasks and the results demonstrate the effectiveness and advantages of the proposed approaches.

- Microscope-Guided Autonomous Clear Corneal Incision

    Author: Xia, Jun | Sun Yat-Sen University
    Author: Bergunder, Sean Joseph | Sun Yat-Sen University
    Author: Lin, Duoru | Sun Yat-Sen University, Zhongshan Ophthalmic Center
    Author: Yan, Ying | Sun Yat-Sen University, Zhongshan Ophthalmic Center
    Author: Lin, Shengzhi | Sun Yat-Sen University
    Author: Nasseri, M. Ali | Technische Universitaet Muenchen
    Author: Zhou, Mingchuan | Technische Universitét M�nchen
    Author: Lin, Haotian | Sun Yat-Sen University, Zhongshan Ophthalmic Center
    Author: Huang, Kai | Sun Yat-Sen University
 
    keyword: Surgical Robotics: Planning; Computer Vision for Medical Robotics; Medical Robots and Systems

    Abstract : Clear Corneal Incision, a challenging step in cataract surgery, and important to the overall quality of the surgery. New surgeons usually spend one full year trying to perfect their incision, but even after such rigorous training de&#64257;cient incisions can still occur. This paper proposes an autonomous robotic system for this self-sealing incision. A conventional ophthalmic microscope system with a monocular camera is utilized to capture the surgical scene, ascertain the robot's position, and estimate depth information. Kinematics with a remote centre of motion (RCM) is designed for a multiaxes robot to perform the incision route. The experimental results on ex-vivo porcine eyes show the autonomous Clear Corneal Incision has a stricter three-plane structure than a surgeon-made incision, which is closer to the ideal incision.

- A Haptic Interface for the Teleoperation of Extensible Continuum Manipulators

    Author: Frazelle, Chase | Clemson University
    Author: Kapadia, Apoorva | Clemson University
    Author: Walker, Ian | Clemson University
 
    keyword: Tendon/Wire Mechanism; Soft Robot Materials and Design; Haptics and Haptic Interfaces

    Abstract : We describe a novel haptic interface designed specifically for the teleoperation of extensible continuum manipulators. The proposed device is based off of, and extends to the haptic domain, a kinematically similar input device for continuum manipulators called the MiniOct. This paper describes the physical design of the new device, the method of creating impedance type haptic feedback to users, and some of the requirements for implementing this device in a bilateral teleoperation scheme. We report a series of initial experiments to validate the operation of the system, including simulated and real-time conditions. The experimental results show that a user can identify the direction of planar obstacles from the feedback for both virtual and physical environments. Finally, we discuss the challenges for providing feedback to an operator about the state of a teleoperated continuum manipulator.

- From Crowd Simulation to Robot Navigation in Crowds
 
    Author: Fraichard, Thierry | INRIA
    Author: Levesy, Valentin | INRIA
 
    keyword: Human-Centered Robotics; Collision Avoidance; Simulation and Animation

    Abstract : This paper presents the result of a study aiming at investigating to what extent the results obtained in the Crowd Simulation domain could be used to control a mobile robot navigating among people. It turns out that Crowd Simulation relies on two assumptions that would not hold for a real mobile robot, a test protocol has therefore been designed in order to thoroughly evaluate how three representative Crowd Simulation techniques would perform when said assumptions are relaxed. The study shows that all those techniques entail safety problems, i.e. they would cause collisions in the real world. The study also highlights the most promising candidate for a transposition on a real mobile robot.

- Are We There Yet? Comparing Remote Learning Technologies in the University Classroom

    Author: Fitter, Naomi T. | University of Southern California
    Author: Raghunath, Nisha | Oregon State University
    Author: Cha, Elizabeth | University of Southern California
    Author: Sanchez, Christopher A. | Oregon State University
    Author: Takayama, Leila | University of California, Santa Cruz
    Author: Mataric, Maja | University of Southern California
 
    keyword: Human-Centered Robotics; Telerobotics and Teleoperation; Social Human-Robot Interaction

    Abstract : Telepresence robots can empower people to work, play, and learn along with others, despite geographic distance. To investigate the use of telepresence robots for remote attendance of university-level classes, we conducted a study in four courses at our university. We compared student experiences attending class during three distinct phases in three different ways: in person, via state-of-the-art university distance learning tools (DLT), and via a telepresence robot. The results from N = 18 student participants revealed that although class attendance method preferences were split between in-person and DLT attendance, students felt more present, self-aware, and expressive when using a telepresence robot than when using DLT. The instructors of the courses uniformly preferred in-person attendance, but they noted that for remote learning, telepresence would be preferable to DLT use. This work can help to inform telepresence robotics and higher education researchers who wish to improve distance learning technologies.

- Bilateral Haptic Collaboration for Human-Robot Cooperative Tasks

    Author: Salvietti, Gionata | University of Siena
    Author: Iqbal, Muhammad Zubair | University of Siena
    Author: Prattichizzo, Domenico | Université Di Siena
 
    keyword: Human-Centered Robotics; Haptics and Haptic Interfaces; Grippers and Other End-Effectors

    Abstract : The aim of this paper is to introduce the concept of bilateral haptic cooperation as a novel paradigm for human-robot cooperative tasks. The approach is demonstrated with a system composed of a soft gripper and a wearable interface. The soft gripper, called CoGripper, has been designed to guarantee a safe interaction giving the possibility to the operator to reconfigure the device according to the object to be grasped. The wearable interface is used to control the open/close motion of the gripper and to feedback information about important task parameters, e.g., the grasp tightness. The result is a bilateral haptic collaboration where human and robot bidirectionally communicate through the interface. The user interaction with the system is extremely intuitive and simple. We performed three user studies to prove the effectiveness of bilateral haptic collaboration involving ten subjects. Results confirmed that the use of the wearable interface reduces the time to accomplish a cooperative task and enables a better control of the grasp tightness.

- A Surgeon-Robot Shared Control for Ergonomic Pedicle Screw Fixation

    Author: Lauretti, Clemente | Université Campus Bio-Medico Di Roma
    Author: Cordella, Francesca | University Campus Biomedico of Rome
    Author: Tamantini, Christian | Campus Bio-Medico University of Rome
    Author: Gentile, Cosimo | Campus Bio-Medico Di Roma
    Author: Scotto di Luzio, Francesco | Université Campus Bio-Medico Di Roma
    Author: Zollo, Loredana | Université Campus Bio-Medico
 
    keyword: Human-Centered Robotics; Physically Assistive Devices; Medical Robots and Systems

    Abstract : Pedicle screw fixation is a fundamental surgical procedure which requires high accuracy and strength from the surgeon who may be exposed to uncomfortable postures and muscular fatigue. The objective of this paper is to propose a novel approach to robot-aided pedicle screw fixation based on a Surgeon-Robot Shared control that makes the tapping procedure, i.e. threading the patient's pedicle, semi-autonomous. The surgeon continues to have a full control over the surgical intervention by i) accurately move the robot end-effector using a hands-on control interface, i.e. the surgical tapper, along a pre-planned axis, ii) continuously controlling the forces exerted onto the patient spine during the tapping phase, iii) modulating the torque about the tapping axis by adequately tuning the tool-bone interaction force along the same axis. Furthermore, the procedure appears to be more comfortable and less tiring for the surgeon. The proposed approach was tested on eight subjects who were asked to perform the tapping procedure onto an anthropomorphic spine phantom. A comparative analysis among the proposed approach and the ones typically adopted in literature to perform pedicle screw placement was carried out. The experimental results demonstrated that proposed approach, with respect to the traditional robot-aided procedure, guarantees comparable accuracy and efficiency in the screw placement and lower fatigue and more comfortable postures for the surgeon.

- Improving Robotic Cooking Using Batch Bayesian Optimization

    Author: Junge, Kai | University of Cambridge
    Author: Hughes, Josie | MIT
    Author: George Thuruthel, Thomas | Bio-Inspired Robotics Lab, University of Cambridge
    Author: Iida, Fumiya | University of Cambridge
 
    keyword: Human-Centered Robotics; Domestic Robots; Optimization and Optimal Control

    Abstract : With advances in the field of robotic manipulation, sensing and machine learning, robotic chefs are expected to become prevalent in our kitchens and restaurants. Robotic chefs are envisioned to replicate human skills in order to reduce the burden of the cooking process. However, the potential of robots as a means to enhance the dining experience is unrecognised. This work introduces the concept of food quality optimization and its challenges with an automated omelette cooking robotic system. The design and control of the robotic system that uses general kitchen tools is presented first. Next, we investigate new optimization strategies for improving subjective food quality rating, a problem challenging because of the qualitative nature of the objective and strongly constrained number of function evaluations possible. Our results show that through appropriate design of the optimization routine using Batch Bayesian Optimization, improvements in the subjective evaluation of food quality can be achieved reliably, with very few trials and with the ability for bulk optimization. This study paves the way towards a broader vision of personalized food for taste-and-nutrition and transferable recipes.

- Adaptive Motion Planning for a Collaborative Robot Based on Prediction Uncertainty to Enhance Human Safety and Work Efficiency (I)

    Author: Kanazawa, Akira | Tohoku University
    Author: Kinugawa, Jun | Tohoku University
    Author: Kosuge, Kazuhiro | Tohoku University
 
    keyword: Human-Centered Automation; Optimization and Optimal Control; Factory Automation

    Abstract : Industrial robots are expected to share the same workspace with human workers and work in cooperation with humans to improve the productivity and maintain the quality of products. In this situation, the worker safety and work-time efficiency must be enhanced simultaneously. In this paper, we extend a task scheduling system proposed in the previous work by installing an online trajectory generation system. On the basis of the probabilistic prediction of the worker motion and the receding horizon scheme for the trajectory planning, the proposed motion planning system calculates an optimal trajectory that realizes collision avoidance and the reduction of waste time simultaneously. Moreover, the proposed system plans the robot trajectory adaptively based on updated predictions and its uncertainty to deal not only with the regular behavior of workers but also with their irregular behavior. We apply the proposed system to an assembly process where a two-link planarmanipulator supports a worker by delivering parts and tools. After implementing the proposed system, we experimentally evaluate the effectiveness of the adaptive motion planning.

## Mechanism Design

- Quadrupedal Locomotion on Uneven Terrain with Sensorized Feet

    Author: Valsecchi, Giorgio | Robotic System Lab, ETH
    Author: Grandia, Ruben | ETH Zurich
    Author: Hutter, Marco | ETH Zurich
 
    keyword: Mechanism Design; Legged Robots; Motion Control

    Abstract : Sensing of the terrain shape is crucial for legged robots deployed in the real world since the knowledge of the local terrain inclination at the contact points allows for an optimized force distribution that minimizes the risk of slipping. In this paper, we present a reactive locomotion strategy for torque controllable quadruped robots based on sensorized feet. Since the present approach works without exteroceptive sensing, it is robust against degraded vision. Inertial and Force/Torque sensors implemented in specially designed feet with articulated passive ankle joints measure the local terrain inclination and interaction forces. The proposed controller exploits the contact null-space in order to minimize the tangential forces to prevent slippage even in case of extreme contact conditions. We experimentally tested the proposed method in laboratory experiments and validated the approach with the quadrupedal robot ANYmal.

- Exploiting Singular Configurations for Controllable, Low-Power, Friction Enhancement on Unmanned Ground Vehicles

    Author: Foris, Adam | Georgia Institute of Technology
    Author: Wagener, Nolan | Georgia Tech
    Author: Boots, Byron | University of Washington
    Author: Mazumdar, Anirban | Georgia Institute of Technology
 
    keyword: Mechanism Design; Field Robots; Wheeled Robots

    Abstract : This paper describes the design, validation, and performance of a new type of adaptive wheel morphology for unmanned ground vehicles. Our adaptive wheel morphology uses a spiral cam to create a system that enables controllable deployment of high friction surfaces. The overall design is modular, battery powered, and can be mounted directly to the wheels of a vehicle without additional wiring. The use of a tailored cam profile exploits a singular configuration to minimize power consumption when deployed and protects the actuator from external forces. Component-level experiments demonstrate that friction on ice and grass can be increased by up to 170%. Two prototypes were also incorporated directly into a 1:5 scale radio-controlled rally car. The devices were able to controllably deploy, increase friction, and greatly improve acceleration capacity on a slippery, synthetic ice surface.

- Flow Compensation for Hydraulic Direct-Drive System with a Single-Rod Cylinder Applied to Biped Humanoid Robot

    Author: Shimizu, Juri | Waseda University
    Author: Otani, Takuya | Waseda University
    Author: Mizukami, Hideki | Waseda University
    Author: Hashimoto, Kenji | Meiji University
    Author: Takanishi, Atsuo | Waseda University
 
    keyword: Hydraulic/Pneumatic Actuators; Humanoid and Bipedal Locomotion; Mechanism Design

    Abstract : Biped robots require massive power on each leg while walking, hopping, and running. We have developed a flow-based control system�called hydraulic direct drive system�that can achieve high output while avoiding spatial limitations. To implement the proposed system with simple equipment configuration, a pump and single-rod cylinder are connected in a closed loop. However, because compensation for flow rate is impossible in a completely closed loop, owing to the difference in the pressure receiving area caused by the rod, a passive flow compensation valve is employed. This valve has a simple structure and is easy to implement. Further, an additional sensor is required to detect the open/close state because the valve state will cause an error in flow control. Therefore, we implemented a model in the controller to predict the state of the flow compensation valve and formulated a method of switching from flow control to pressure control according to the predicted state. Experimental results indicate that the error of the joint angle is reduced to less than 1.6 degrees for walking patterns, and stable walking is realized when the system is installed in biped humanoid robots.

- Development of Visible Manipulator with Multi-Gear Array Mechanism for Laparoscopic Surgery

    Author: Wang, Haibo | Tianjin University
    Author: Wang, Shuxin | Tianjin University
    Author: Zuo, Siyang | Tianjin University
 
    keyword: Mechanism Design; Medical Robots and Systems; Surgical Robotics: Laparoscopy

    Abstract : In recent years, robotic technology has been introduced to medical fields, and many surgical robots have been proposed for minimally invasive surgery (MIS). However, due to the limitations in dexterity imposed by surgical instruments and occlusion area, surgeons experience great difficulties during operations. In this paper, we propose a visible manipulator for laparoscopic surgery. Unlike other multiple degree-of-freedom (DOF) manipulators that utilize compliant parts or tendons and pulleys, our proposed manipulator adopts a multi-gear array mechanism to perform the yaw and pitch motions. The manipulator is integrated with a visualization unit to provide macroscopic images for observation in a constrained cavity. Moreover, flexible surgical tools with different functions can be inserted through the central channel of the manipulator to perform diagnostic or therapeutic procedures. A master-slave system is developed to control the bending motions. Bending characteristics experiments and load capacity experiments are performed. The experimental results demonstrate that the proposed manipulator can perform bending motions with a yaw angle range of -76.8�~76.2� and a pitch angle range of -75.2�~75.6�. The manipulator can lift a workload of 250 g during yaw motion and a workload of 150 g during pitch motion, demonstrating the potential clinical value of the visible manipulator for robot-assisted surgery.

- Mechanically Programmed Miniature Origami Grippers

    Author: Liu, Chang | Northeastern University
    Author: Orlofsky, Alec | Northeastern University
    Author: Kamrava, Soroush | Northeastern University
    Author: Vaziri, Ashkan | Northeastern University
    Author: Felton, Samuel | Northeastern University
 
    keyword: Mechanism Design; Grippers and Other End-Effectors; Grasping

    Abstract : This paper presents a robotic gripper design that can perform customizable grasping tasks at the millimeter scale. The design is based on the origami string, a mechanism with a single degree of freedom that can be mechanically programmed to approximate arbitrary paths in space. By using this concept, we create miniature fingers that bend at multiple joints with a single actuator input. The shape and stiffness of these fingers can be varied to fit different grasping tasks by changing the crease pattern of the string. We show that the experimental behavior of these strings follows their analytical models and that they can perform a variety of tasks including pinching, wrapping, and twisting common objects such as pencils, bottle caps, and blueberries.

- Design of a Novel Multiple-DOF Extendable Arm with Rigid Components Inspired by a Deployable Origami Structure

    Author: Matsuo, Hiroshi | Tokyo Institute of Technology
    Author: Asada, Harry | MIT
    Author: Takeda, Yukio | Tokyo Institute of Technology
 
    keyword: Mechanism Design; Kinematics

    Abstract : An extendable robot inspired by origami is designed, analyzed, and tested. Its deployable origami structure has a large extension ratio, allowing the robot to extend the body length multiple times. The new robot, however, differs from the existing origami structure in two aspects. One is that the robot mechanism consists of all rigid bodies, unlike the prior origami that exploits structural deformation for creating flexible configurations. The other is that new origami-inspired robot has multiple active degrees of freedom, allowing for taking various postures, unlike most deployable mechanisms composed of rigid components having a single DOF. When developing a mechanism based on an origami structure, we often encounter the deformations of parts during the transition from the contracted to the extended configurations. Previously, we analyzed the motion of a deployable origami structure considering the foldings' deformation and showed that they do not have any kinematic roles but give a large effect to constrain the motion. Thus, we come to the idea that by removing such parts, a novel rigid extendable mechanism with multiple DOF can be obtained, which can achieve a large extension ratio and a high transformability only by its kinematic structure, beyond an original origami structure.

- A Compact and Low-Cost Robotic Manipulator Driven by Supercoiled Polymer Actuators

    Author: Yang, Yang | The Hong Kong University of Science and Technology
    Author: Liu, Zhicheng | The Hong Kong University of Science and Technology
    Author: Wang, Yanhan | The Hong Kong University of Science and Technology
    Author: Liu, Shuai | Hong Kong University of Science and Technology
    Author: Wang, Michael Yu | Hong Kong University of Science &amp; Technology
 
    keyword: Mechanism Design; Biologically-Inspired Robots; Grippers and Other End-Effectors

    Abstract : The supercoiled polymer (SCP) actuator is a novel artificial muscle, which is manufactured by twisting and coiling polymer fibers. This new artificial muscle is soft, low-cost and shows good linearity. Being utilized as an actuator, the artificial muscle could generate significant mechanical power in a muscle-like form upon electrical activation by Joule heating. In this study, we adopt this new artificial muscle to actuate a novel designed robotic manipulator, which is composed of two parts. The first part is a robotic arm based on the inspiration of the musculoskeletal system. The arm is fabricated with two ball-and-socket joints as skeleton and SCP actuators as driven muscles. The second part is a Fin Ray Effect inspired soft gripper that can perform grasping tasks on fragile objects. The manipulator prototype is fabricated and experimental tests are conducted including both simple but effective control of the bio-inspired arm as well as characterization of the gripper. Lastly, a pick and place demonstration of a fragile fruit is performed utilizing the proposed manipulator. We envision that the bio-inspired robotic manipulator design driven by SCP actuators could potentially be used in other robotic applications.

- A Wall-Mounted Robot Arm Equipped with a 4-DOF Yaw-Pitch-Yaw-Pitch Counterbalance Mechanism

    Author: Min, Jae-Kyung | Korea University
    Author: Kim, Do-Won | Korea University
    Author: Song, Jae-Bok | Korea University
 
    keyword: Mechanism Design; Tendon/Wire Mechanism; Humanoid Robots

    Abstract : Because industrial robots are relatively heavy, most of motor torque are used to support the weight of a robot. Consequently, high-capacity motors and speed reducers are needed, resulting in a low energy efficiency and an increase in the manufacturing cost. To deal with this problem, a variety of spring-based counterbalance mechanisms (CBM) have been developed to mechanically compensate for the gravitational torque caused by the robot weight and payload. However, conventional CBMs are limited to pitch joints whose axis of rotation is horizontal to the ground and it is difficult to apply them to robot arms with different joint configurations, such as humanoid robot arms. In this study, we propose a CBM with a passive yaw-pitch structure consisting of a spring and wire. Through geometrical analysis and experiments, we demonstrate that the proposed CBM can effectively compensate for the gravitational torque due to robot weight and payload.

- Internally-Balanced Magnetic Mechanisms Using a Magnetic Spring for Producing a Large Amplified Clamping Force

    Author: Shimizu, Tori | Tohoku University
    Author: Tadakuma, Kenjiro | Tohoku University
    Author: Watanabe, Masahiro | Tohoku University
    Author: Takane, Eri | Tohoku University
    Author: Konyo, Masashi | Tohoku University
    Author: Tadokoro, Satoshi | Tohoku University
 
    keyword: Mechanism Design; Grippers and Other End-Effectors; Grasping

    Abstract : To detach a permanent magnet with a control force much smaller than its original attractive force, the Internally-Balanced Magnetic Unit (IB Magnet) was invented and has been applied to magnetic devices such as wall-climbing robots, ceil-dangling drones and modular swarm robots. In contrast to its drastic reduction rate on the control force, the IB Magnet has two major problems on its nonlinear spring which cancels out the internal force on the magnet: complicated design procedure and trade-off relationship between balancing precision and mechanism volume. This paper proposes a principle of a new balancing method for the IB Magnet which uses a like-pole pair of magnets as a magnetic spring, whose repulsive force ideally equals the attractive force of an unlike-pole pair exactly. To verify the proposed principle, the     Authors realized a prototype model of the IB Magnet using magnetic spring and verified through experiments its reduction rate is comparable to those of conventional IB Magnets. Moreover, the     Authors discussed and realized a robotic clamp as an application example containing proposed the proposed IB Magnets as its internal mechanism.

- A Continuum Manipulator with Closed-Form Inverse Kinematics and Independently Tunable Stiffness

    Author: Zhao, Bin | Shanghai Jiao Tong University
    Author: Zeng, Lingyun | Shanghai Jiao Tong University
    Author: Wu, Baibo | Shanghai Jiao Tong University
    Author: Xu, Kai | Shanghai Jiao Tong University
 
    keyword: Mechanism Design; Kinematics; Flexible Robots

    Abstract : Continuum manipulators can accomplish various tasks in confined spaces, benefiting from their compliant structures and improved dexterity. Confined and unstructured spaces may require both enhanced stiffness of a continuum manipulator for precision and payload, as well as compliance for safe interaction. Thus, studies have been consistently dedicated to design continuum or articulated manipulators with tunable stiffness to adapt to different operating conditions. This paper presents a simple continuum manipulator with independently tunable stiffness where the stiffness variation does not affect the movement of the manipulator's end-effector. Moreover, the proposed continuum manipulator is found to have analytical inverse kinematics. The design concept, analytical kinematics, system construction and experimental characterizations are presented. The results showed that the manipulator's stiffness can be increased up to 3.61 times of the minimal value, demonstrating the effectiveness of the proposed idea.

- Shape-Morphing Wheel Mechanism for Step Climbing in High Speed Locomotion

    Author: Ryu, Sijun | Hanyang University
    Author: Lee, Youngjoo | Hanyang University
    Author: Seo, TaeWon | Hanyang University
 
    keyword: Mechanism Design; Wheeled Robots

    Abstract : The ability of wheeled mobile robots to overcome steps is often limited by wheel size. Enhancing the ability of mobile robots to overcome obstacles is essential for extending their operation area, and many previous attempts have been made by using transformable wheels, a linkage mechanism, and a spoke-type wheel-leg mechanism. In this study, we propose a shape-morphing wheel mechanism for step climbing at high speed. In the general case of low speed locomotion, a robot's wheels can be used normally. However, to overcome relatively large obstacles, the robot's wheels can extend its shape by using the proposed morphing mechanism with centrifugal force at high speed locomotion. Two modes of step climbing are analyzed that use kinetic energy conversion or impact on the steps. Detail design issues with comprehensive analyses results are presented. Results demonstrate that a robot with morphing wheels can climb a 46.67 mm obstacle at 1.82 m/s, which is 1.33 times larger than the wheel radius. We expect that this method can be applied to other locomotion modes of wheeled mobile robots.

- Design and Compensation Control of a Flexible Instrument for Endoscopic Surgery

    Author: Hong, Wuzhou | Shanghai Jiao Tong University
    Author: Schmitz, Andreas | Imperial College London
    Author: Bai, Weibang | Imperial College London
    Author: Berthet-Rayne, Pierre | Imperial College London
    Author: Xie, Le | Shanghai Jiao Tong University
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Mechanism Design; Optimization and Optimal Control; Medical Robots and Systems

    Abstract : Snake-like robots for endoscopic surgery make it possible to reach deep-seated lesions. With the use of small flexible tendon-driven instruments, it is possible to perform bi-manual micro-surgical tasks that are challenging for standard endoscopic surgeries. Existing devices, however, lack articulated wrists and rolling motion of the end-effector. This paper presents a new instrument design with a distal-roll gripper for snake-like robots. The developed 5 DoFs miniaturized instruments with a diameter of 3 mm enable the deployment into narrow endoluminal channels. Issues related to actuation coupling, tendon slack, and backlash are addressed. Experimental results show that the distal-roll gripper can rotate 106�, and the actuated joints can achieve good repeatability and accuracy with the proposed compensation control scheme.

- Steerable Burrowing Robot: Design, Modeling and Experiments

    Author: Barenboim, Moran | Technion - Israel Institute of Technology
    Author: Degani, Amir | Technion - Israel Institute of Technology
 
    keyword: Mechanism Design; Underactuated Robots; Nonholonomic Mechanisms and Systems

    Abstract : This paper investigates a burrowing robot that can maneuver and steer while being submerged in a granular medium. The robot locomotes using an internal vibro-impact mechanism and steers using a rotating bevel-tip head. We formulate and investigate a non-holonomic model for the steering mechanism and a hybrid dynamics model for the thrusting mechanism. We perform a numerical analysis of the dynamics of the robot's thrusting mechanism using a simplified, orientation and depth dependent model for the drag forces acting on the robot. We first show, in simulation, that by carefully tuning various control input parameters, the thrusting mechanism can drive the robot both forward and backward. We present several experiments designed to evaluate and verify the simulative results using a proof-of-concept robot. We show that different input amplitudes indeed affect the direction of motion, as suggested by the simulation. We further demonstrate the ability of the robot to perform a simple S-shaped trajectory. These experiments demonstrate the feasibility of the robot's design and fidelity of the model.

- High Force Density Gripping with UV Activation and Sacrificial Adhesion

    Author: Lee, Esther | North Carolina State University
    Author: Goddard, Zachary | Georgia Institute of Technology
    Author: Ngotiaoco, Joshua | Georgia Institute of Technology
    Author: Monterrosa, Noe | Georgia Institute of Technology
    Author: Mazumdar, Anirban | Georgia Institute of Technology
 
    keyword: Mechanism Design; Mobile Manipulation

    Abstract : This paper presents a novel physical gripping framework intended for controlled, high force density attachment on a range of surfaces. Our framework utilizes a light-activated chemical adhesive to attach to surfaces. The cured adhesive is part of a ``sacrificial layer,'' which is shed when the gripper separates from the surface. In order to control adhesive behavior we utilize ultraviolet (UV) light sensitive acrylics which are capable of rapid curing when activated with 380nm light. Once cured, zero input power is needed to hold load. Thin plastic parts can be used as the sacrificial layers, and these can be released using an electric motor. This new gripping framework including the curing load capacity, adhesive deposition, and sacrificial methods are described in detail. Two proof-of concept prototypes are designed, built, and tested. The experimental results illustrate the response time (15-75s depending on load), high holding force density (10-30), and robustness to material type. Additionally, two drawbacks of this design are discussed: corruption of the gripped surface and a limited number of layers.

- Stiffness Optimization of a Cable Driven Parallel Robot for Additive Manufacturing

    Author: Gueners, Damien | Université Clermont Auvergne/Institut Pascal/Sigma-Clermont
    Author: Chanal, Hélène | SIGMA Clermont
    Author: Bouzgarrou, Chedli | Institut Pascal UMR 6602 - UCA/CNRS/SIGMA
 
    keyword: Mechanism Design; Parallel Robots; Additive Manufacturing

    Abstract : In this paper, the optimization of the anchor points of a cable driven parallel robot (CDPR) for 3D printing is proposed in order to maximize the rigidity. Indeed, in the context of 3D printing, robot stiffness should guarantee a high level of tool path following accuracy. The optimized platform showed a rigidity improvement in simulation, but also experimentally with a first study of vibration modes. In the same time, this study illustrates the influence of preload in cables on the platform rigidity.

- CAMI - Analysis, Design and Realization of a Force-Compliant Variable Cam System

    Author: Mannhart, Dominik | ETH Zurich
    Author: Dubois, Fabio | Eidgen�ssische Technische Hochschule
    Author: Bodie, Karen | ETH Zurich
    Author: Klemm, Victor | ETH Zurich
    Author: Morra, Alessandro | ETH Zurich
    Author: Hutter, Marco | ETH Zurich
 
    keyword: Mechanism Design; Compliant Joint/Mechanism; Legged Robots

    Abstract : This work presents a novel design concept that achieves multi-legged locomotion using a three-dimensional cam system. A computational framework has been developed to analyze and dimension this cam apparatus, that can perform arbitrary end effector motions within its design constraints. The mechanism enables continuous gait transition and inherent force compliance. With only two motors, any trajectory of a continuous set of gaits can be followed. One motor is used to actuate the system and a second one to morph its movement. To illustrate a possible application of this system, a working prototype of a bipedal robot is developed and validated in hardware. It showcases a smooth velocity change by transitioning through different gaits from standing still to walking fast at 124 mm/s within 2.0 s, while following the given end effector trajectory with an error of only 2.47 mm.

- Using Manipulation to Enable Adaptive Ground Mobility

    Author: Kim, Raymond | Georgia Institute of Technology
    Author: DeBate, Alex | Georgia Institute of Technology
    Author: Balakirsky, Stephen | Georgia Tech
    Author: Mazumdar, Anirban | Georgia Institute of Technology
 
    keyword: Mechanism Design; Mobile Manipulation; Wheeled Robots

    Abstract : In order to accomplish various missions, autonomous ground vehicles must operate on a wide range of terrain. While many systems such as wheels and whegs can navigate some types of terrain, none are optimal across all. This creates a need for physical adaptation. This paper presents a broad new approach to physical adaptation that relies on manipulation. Specifically, we explore how multipurpose manipulators can enable ground vehicles to dramatically modify their propulsion system in order to optimize performance across various terrain. While this approach is general and widely applicable, this work focuses on physically switching between wheels and legs. We outline the design of "swappable propulsors" that combine the powerful adhesion forces of permanent magnets with geometric features for easy detachment. We provide analysis on how the swappable propulsors can be manipulated, and use these results to create a functional prototype robot. This robot can use its manipulator to change between wheeled and legged locomotion. Our experimental results illustrate how this approach can enhance energy efficiency and versatility.

- SNIAE-SSE Deformation Mechanism Enabled Scalable Multicopter: Design, Modeling and Flight Performance Validation

    Author: Yang, Tao | Harbin Institute of Technology, Shenzhen
    Author: Zhang, Yujing | Harbin Institute of Technology, Shenzhen
    Author: Li, Peng | Harbin Institute of Technology (ShenZhen)
    Author: Shen, Yantao | University of Nevada, Reno
    Author: Liu, Yunhui | Chinese University of Hong Kong
    Author: Chen, Haoyao | Harbin Institute of Technology
 
    keyword: Mechanism Design; Product Design, Development and Prototyping

    Abstract : This paper focuses on designing, modeling and validating a novel scalable multicopter whose deformation mechanism, called SNIAE-SSE, relies on a combination of simple non-intersecting angulated elements (SNIAEs) and straight scissor-like elements (SSEs). The proposed SNIAE-SSE mechanism has the advantages of single degree-of-freedom, fast actuation capability and large deformation ratio. In this work, enabled by the SNIAE-SSE mechanism, a quadcopter prototype with symmetrical and synchronous deformation is firstly developed, which facilitates a novel and controllably scalable multicopter system for us to analyze its modeling, as well as to validate its flight performance and dynamics during the deformation in several flight missions including hover, throwing, and morphing flying through a narrow window. Experimental results demonstrate that the developed scalable multicopter can maintain its stable flight behavior even both the folding and unfolding body deformations are fast performed, which indicates an excellent capability of the scalable multicopter to rapidly adapt to complex and dynamically changed environments.

## Marine Robotics
- Distance and Steering Heuristics for Streamline-Based Flow Field Planning

    Author: To, Kwun Yiu Cadmus | University of Technology Sydney
    Author: Yoo, Chanyeol | University of Technology Sydney
    Author: Anstee, Stuart David | Defence Science and Technology Group
    Author: Fitch, Robert | University of Technology Sydney
 
    keyword: Marine Robotics; Motion and Path Planning; Field Robots

    Abstract : Motion planning for vehicles under the influence of flow fields can benefit from the idea of streamline-based planning, which exploits ideas from fluid dynamics to achieve computational efficiency. Important to such planners is an efficient means of computing the travel distance and direction between two points in free space, but this is difficult to achieve in strong incompressible flows such as ocean currents. We propose two useful distance functions in analytical form that combine Euclidean distance with values of the stream function associated with a flow field, and with an estimation of the strength of the opposing flow between two points. Further, we propose steering heuristics that are useful for steering towards a sampled point. We evaluate these ideas by integrating them with RRT&#8727; and comparing the algorithm's performance with state-of-the-art methods in an artificial flow field and in actual ocean prediction data in the region of the dominant East Australian Current between Sydney and Brisbane. Results demonstrate the method's computational efficiency and ability to find high-quality paths outperforming state-of-the-art methods, and show promise for practical use with autonomous marine robots.

- Enhancing Coral Reef Monitoring Utilizing a Deep Semi-Supervised Learning Approach

    Author: Modasshir, Md | University of South Carolina
    Author: Rekleitis, Ioannis | University of South Carolina
 
    keyword: Marine Robotics; Computer Vision for Other Robotic Applications; Semantic Scene Understanding

    Abstract : Coral species detection underwater is a challenging problem. There are many cases when even the experts (marine biologists) fail to recognize corals, hence limiting ground truth annotation for training a robust detection system. Identifying coral species is fundamental for enabling the monitoring of coral reefs, a task currently performed by humans, which can be automated with the use of underwater robots. By employing temporal cues using a tracker on a high confidence prediction by a convolutional neural network-based object detector, we augment the collected dataset for the retraining of the object detector. However, using trackers to extract examples also introduces hard or mislabelled samples, which is counterproductive and will deteriorate the performance of the detector. In this work, we show that employing a simple deep neural network to filter out hard or mislabelled samples can help regulate sample extraction. We empirically evaluate our approach in a coral object dataset, collected via an Autonomous Underwater Vehicle (AUV) and human divers, that shows the benefit of incorporating extracted examples obtained from tracking. This work also demonstrates how controlling sample generation by tracking using a simple deep neural network can further improve an object detector.

- DOB-Net: Actively Rejecting Unknown Excessive Time-Varying Disturbances

    Author: Wang, Tianming | University of Technology Sydney
    Author: Lu, Wenjie | University of Technology Sydney
    Author: Yan, Zheng | University of Technology Sydney
    Author: Liu, Dikai | University of Technology, Sydney
 
    keyword: Marine Robotics; Learning and Adaptive Systems

    Abstract : This paper presents an observer-integrated Reinforcement Learning (RL) approach, called Disturbance OBserver Network (DOB-Net), for robots operating in environments where disturbances are unknown and time-varying, and may frequently exceed robot control capabilities. The DOB-Net integrates a disturbance dynamics observer network and a controller network. Originated from conventional DOB mechanisms, the observer is built and enhanced via Recurrent Neural Networks (RNNs), encoding estimation of past values and prediction of future values of unknown disturbances in RNN hidden state. Such encoding allows the controller generate optimal control signals to actively reject disturbances, under the constraints of robot control capabilities. The observer and the controller are jointly learned within policy optimization by advantage actor critic. Numerical simulations on position regulation tasks have demonstrated that the proposed DOB-Net significantly outperforms conventional feedback controllers and classical RL policy.

- Demonstration of Autonomous Nested Search for Local Maxima Using an Unmanned Underwater Vehicle

    Author: Branch, Andrew | Jet Propulsion Laboratory
    Author: McMahon, James | The Naval Research Laboratory
    Author: Xu, Guangyu | Applied Physics Laboratory of University of Washington
    Author: Jakuba, Michael | Woods Hole Oceanographic Institution
    Author: German, Christopher R. | Woods Hole Oceanographic Institution
    Author: Chien, Steve | Jet Propulsion Laboratory
    Author: Kinsey, James | Woods Hole Oceanographic Institution
    Author: Bowen, Andrew D. | Woods Hole Oceanographic Institution
    Author: Hand, Kevin P. | Jet Propulsion Laboratory
    Author: Seewald, Jeffrey S. | Woods Hole Oceanographic Institution
 
    keyword: Marine Robotics; Space Robotics and Automation; Autonomous Agents

    Abstract : Ocean Worlds represent one of the best chances for extra-terrestrial life in our solar system. A new mission concept must be developed to explore these oceans. This mission would require traversing the 10s of km thick icy shell and releasing a submersible into the ocean below. During the transit of the icy shell and the exploration of the ocean, the vehicle(s) will be out of contact with Earth for weeks or potentially months at a time. During this time the vehicle must have sufficient autonomy to locate and study scientific targets of interest. One such target of interest is hydrothermal venting. We have previously developed an autonomous nested search method to locate and investigate sources of hydrothermal venting by locating local maxima in hydrothermal vent emissions. In this work we demonstrate this approach on board an OceanServer Iver2 AUV in Chesapeake Bay, MD using simulated sensor data from a hydrothermal plume model. This represents the first step towards the deployment of this approach in conditions analogous to those that we might expect on an Ocean World.

- Towards Distortion Based Underwater Domed Viewport Camera Calibration

    Author: Iscar, Eduardo | University of Michigan
    Author: Johnson-Roberson, Matthew | University of Michigan
 
    keyword: Marine Robotics; Computer Vision for Other Robotic Applications

    Abstract : Photogrammetry techniques used for 3D reconstructions and motion estimation from images are based on projective geometry that models the image formation process. However, in the underwater setting, refraction of light rays at the housing interface introduce non-linear effects in the image formation. These effects produce systematic errors if not accounted for, and severely degrade the quality of the acquired images. In this paper, we present a novel approach to the calibration of cameras inside spherical domes with large offsets between dome and camera centers. Such large offsets not only amplify the effect of refraction, but also introduce blur in the image that corrupts feature extractors used to establish image-world correspondences in existing refractive calibration methods. We propose using the point spread function (PSF) as a complete description of the optical system and introduce a procedure to recover the camera pose inside the dome based on the measurement of the distortions. Results on a collected dataset show the method is capable of recovering the camera pose with high accuracy.

- A Flapped Paddle-Fin for Improving Underwater Propulsive Efficiency of Oscillatory Actuation

    Author: Simha, Ashutosh | Tallinn University of Technology
    Author: Gkliva, Roza | Tallinn University of Technology
    Author: Kotta, �lle | Tallinn University of Technology
    Author: Kruusmaa, Maarja | Tallinn University of Technology
 
    keyword: Marine Robotics; Biologically-Inspired Robots; Mechanism Design

    Abstract : This paper presents a novel design of an oscillatory fin for thrust-efficient and agile underwater robotic locomotion. We propose a flat paddle-fin comprising a set of overlapping cascaded soft flaps that open in one half of the stroke cycle and close in the other. Consequently, asymmetry in the lateral drag force exerted by the fin during oscillatory actuation is passively achieved. This enables a substantially higher degree of efficiency in force generation than conventional oscillatory fins which rely on weaker longitudinal wake-induced forces. Experimental results show a high degree of improvement in net thrust and propulsive-efficiency over conventional fins. Locomotion with the proposed fin has been demonstrated on an underwater robotic platform. Various gaits were achieved using oscillatory actuation, via angular and phase offsets between the actuators.

- Bio-Inspired Tensegrity Fish Robot

    Author: Shintake, Jun | University of Electro-Communications
    Author: Zappetti, Davide | École Polytechnique Fédérale De Lausanne
    Author: Peter, Timoth�e | École Polytechnique Fédérale De Lausanne
    Author: Ikemoto, Yusuke | Meijo University
    Author: Floreano, Dario | Ecole Polytechnique Federal, Lausanne
 
    keyword: Soft Robot Applications; Biologically-Inspired Robots; Soft Robot Materials and Design

    Abstract : This paper presents a method to create fish-like robots with tensegrity systems and describes a prototype modeled on the body shape of the rainbow trout with a length of 400 mm and a mass of 102 g that is driven by a waterproof servomotor. The structure of the tensegrity robot consists of rigid body segments and elastic cables that represent bone/tissue and muscles of fish, respectively. This structural configuration employing the tensegrity class 2 is much simpler than other tensegrity-based underwater robots. It also allows the tuning of the mechanical stiffness, which is often said to be an important factor in fish swimming. In our robot, the body stiffness can be tuned by changing the cross-section of the cables and their pre-stretch ratio. We characterize the robot in terms of body stiffness, swimming speed, and thrust force while varying the body stiffness i.e., the cross-section of the elastic cables. The results show that the body stiffness of the robot can be designed to approximate that of the real fish and modulate its performance characteristics. The measured swimming speed of the robot is 0.23 m/s (0.58 BL/s), which is comparable to other fish robots of the same type. Strouhal number of the robot 0.54 is close to that of the natural counterpart, suggesting that the presented method is an effective engineering approach to realize the swimming characteristics of real fish.

- A Hybrid Underwater Manipulator System with Intuitive Muscle-Level sEMG Mapping Control

    Author: Zhong, Hua | The University of Hong Kong
    Author: Shen, Zhong | The University of Hong Kong
    Author: Zhao, Yafei | The University of Hong Kong
    Author: Tang, Keke | The University of Hong Kong
    Author: Wang, Wenping | The University of Hong Kong
    Author: Wang, Zheng | The University of Hong Kong
 
    keyword: Soft Robot Applications; Physical Human-Robot Interaction; Marine Robotics

    Abstract : Soft-robotic manipulators, with their closed-chamber elastomeric actuators, natural water-sealing and inherent compliance, are ideal for underwater applications for compact, lightweight, and dexterous manipulation tasks. However, their low structure rigidity makes soft robots highly prone to underwater disturbances, rendering traditional control methods unreliable, substantially increasing the challenges for high-dexterity control. To address this issue, we proposed an intuitive underwater hybrid manipulator system with a muscle-level mapping design concept. The manipulator was designed to construct an actuator-configuration which could directly map to the main muscles group in the human forearm. Exploiting this analogy, an electromyography-based wearable controller was developed using continuous bio-sensory data from the operator's arm to complement the intuitive manipulator control. A prototype of the proposed manipulator was constructed and validated in various experiments, where a human user could effectively use muscle activation to proportionally drive the soft-robotic manipulator in free-space motions, as well as performing object manipulation tasks both in air and underwater, only using visual feedback, with consistent performances under various time delays. The promising results of this work have demonstrated that the muscle-level analogy of soft robotics could lead to intuitive and effective underwater manipulation with simple structure and low control effort.

- Single-Hydrophone Low-Cost Underwater Vehicle Swarming
 
    Author: Fischell, Erin Marie | Woods Hole Oceanographic Institution
    Author: Kroo, Anne R. | Olin College of Engineering
    Author: O'Neill, Brendan W. | WHOI/MIT
 
    keyword: Marine Robotics; Multi-Robot Systems; Swarms

    Abstract : Swarms of robots are starting to appear in aerial and ground robotics across multiple operational domains: be it for search-and-rescue, mapping, or light shows, drones are able to be coordinated in groups. This capability does not currently extend underwater as attenuation of light in water is 10 orders of magnitude greater than that in air, rendering ineffective the navigation and communication technologies used in terrestrial robotics. Autonomous underwater vehicle (AUV) navigation is either expensive or unwieldy, requiring high-power sensors such as inertial navigation sensors that cost hundreds of thousands of dollars, frequent GPS surfacing, or deployment of geo-located acoustic beacons. To do �swarms' of underwater vehicles will require a navigation and communication scheme that allows vehicles to remain together in an area while prosecuting a mission without these constraints. This paper suggests a system that would make underwater swarming possible by using single-transducer, Doppler-based and multi-frequency attenuation-based acoustic navigation without time-synchronization for multi-vehicle swarming. In this solution, a leader with good navigation carries a multi-frequency sound source. Followers equipped with a custom low-cost acoustic package then adapt heading based on Doppler-shifted frequency and range using multi-frequency difference in absorption. The theory and design behind this system is presented and tested under different simulation conditions.

- 2D Estimation of Velocity Relative to Water and Tidal Currents Based on Differential Pressure for Autonomous Underwater Vehicles

    Author: Meurer, Christian | Tallinn University of Technology
    Author: Fuentes-P�rez, Juan Francisco | Tallinn University of Technology
    Author: Schwarzw�lder, Kordula Valerie Anne | Norwegian University of Science and Technology
    Author: Ludvigsen, Martin | Norwegian University of Science and Technology
    Author: S�rensen, Asgeir Johan | Norwegian University of Science and Technology
    Author: Kruusmaa, Maarja | Tallinn University of Technology
 
    keyword: Marine Robotics; Autonomous Vehicle Navigation; Sensor Fusion

    Abstract : Reliable navigation of autonomous underwater vehicles (AUVs) depends on the quality of their state estimation. Providing robust velocity estimation is thus an important role. While water currents are main contributors to the navigational uncertainty of AUVs, they are also an important variable for oceanographic research. For both reasons water current estimation is desirable during AUV operations. State of the art velocity estimation either relies on expensive acoustic sensors with considerable energy requirements and a large form factor such as Doppler Velocity Logs (DVL) and Acoustic Doppler Current Profilers (ADCP), while water currents are either estimated with the same sensors, or with algorithms that require accurate position feedback. We introduce a lightweight and energy efficient sensor to estimate fluid relative velocity in 2D based on differential pressure. The sensor is validated in field trials onboard of an AUV in the presence of tidal currents. We further show that, while moving against the currents, our device is capable of estimating tidal currents in situ with a comparable accuracy to a DVL, given a source for absolute vehicle velocity. Additionally, we establish the limitations of the current design of DPSSv2 while moving with the currents.

- Multi-Sensor Mapping for Low Contrast, Quasi-Dynamic, Large Objects

    Author: Shah, Vikrant | Northeastern University
    Author: Schild, Kristin | University of Maine
    Author: Lindeman, Margaret | Scripps Institution of Oceanography
    Author: Duncan, Daniel | The University of Texas at Austin
    Author: Sutherland, David | University of Oregon
    Author: Cenedese, Claudia | Woods Hole Oceanographic Institution
    Author: Straneo, Fiammetta | Scripps Institution of Oceanography
    Author: Singh, Hanumant | Northeatern University
 
    keyword: Marine Robotics; Mapping; Visual-Based Navigation

    Abstract : This paper proposes a systems level solution for addressing the problem of mapping large moving targets with slow but complicated dynamics with multiple sensing modalities. While this work is applicable to other domains we focus our efforts on mapping rotating and translating icebergs. Our solution involves a rigidly coupled combination of a line scan sensor - a subsurface multibeam sonar, with an area scan sensor - an optical camera. This allows the system to exploit the optical camera information to perform iceberg relative navigation which can directly be used by the multibeam sonar to map the iceberg underwater. This paper details the algorithm required to compute the scale of the navigation solution and corrections to find iceberg centric navigation and thus an accurate iceberg reconstruction. This approach is successfully demonstrated on real world iceberg data collected during the 2018 Sermilik campaign in Eastern Greenland. Due to the availability of iceberg mounted GPS observations during this research expedition we could also groundtruth our navigation and thus our systems level mapping efforts.

- Gaussian-Dirichlet Random Fields for Inference Over High Dimensional Categorical Observations

    Author: San Soucie, John E. | Massachusetts Institute of Technology
    Author: Girdhar, Yogesh | Woods Hole Oceanographic Institution
    Author: Sosik, Heidi M. | Woods Hole Oceanographic Institution
 
    keyword: Probability and Statistical Methods; Marine Robotics; Deep Learning in Robotics and Automation

    Abstract : We propose a generative model for the spatiotemporal distribution of high dimensional categorical observations. Such observations are commonly produced by robots equipped with an imaging sensor such as a camera, paired with an image classifier, potentially producing observations over thousands of categories. The proposed approach combines the use of Dirichlet distributions to model sparse co-occurrence relations between the observed categories using a latent variable, and Gaussian processes to model the spatiotemporal distribution of the latent variable. Experiments in this paper show that the resulting model is able to efficiently and accurately approximate the temporal distribution of high dimensional categorical measurements such as taxonomic observations of microscopic organisms in the ocean, even in unobserved (held out) locations, far from other samples. This work's primary motivation is to enable deployment of informative path planning techniques over high dimensional categorical fields, which until now have been limited to scalar or low dimensional vector observations.

- Cooperative Autonomy and Data Fusion for Underwater Surveillance with Networked AUVs

    Author: Ferri, Gabriele | NATO Centre for Maritime Research and Experimentation
    Author: Stinco, Pietro | Nato Sto Cmre
    Author: De Magistris, Giovanni | IBM Research AI
    Author: Tesei, Alessandra | Nato Sto Cmre
    Author: LePage, Kevin | NATO Undersea Research Centre
 
    keyword: Marine Robotics; Autonomous Agents; Networked Robots

    Abstract : Cooperative autonomy and data sharing can largely improve the mission performance of robotic networks in un- derwater surveillance applications. In this paper, we describe the cooperative autonomy used to control the Autonomous Underwater Vehicles (AUVs) acting as sonar receiver nodes in the CMRE Anti-Submarine Warfare (ASW) network. The paper focuses on a track management module that was integrated in the robot autonomy software for enabling the share of information. Track to track (T2T) associations are used for improving track classification and for creating a common tactical picture, necessary for AUV cooperative strategies. We also present a new cooperative data-driven AUV behaviour that exploits the spatial diversity of multiple robots for improving target tracking and for facilitating T2T associations. We report results with real data collected at sea that validate the approach. The reported results are one of the first examples that show the potential of cooperative autonomy and data fusion in realistic underwater surveillance scenarios characterised by limited communications.

- Bidirectional Resonant Propulsion and Localization for AUVs

    Author: Secord, Thomas | University of St. Thomas
    Author: Louwagie, Troy | University
 
    keyword: Marine Robotics; Compliant Joint/Mechanism; Localization

    Abstract : Battery life, reliability, and localization are prominent challenges in the design of autonomous underwater vehicles (AUVs). This work aims to address facets of these challenges using a single system. We describe the design of a bidirectional resonant pump that uses a single electromagnetic voice coil motor (VCM) capable of rotation around a central two degree-of-freedom flexure stage axis. This actuator design produces highly efficient resonant motion that drives two orthogonally oriented diaphragms simultaneously. The operation of this diaphragm pump mechanism produces both adjustable thrust vectors at the aft surface of the AUV and a monotonic relationship between thrust vectors and operating frequency. We propose using the unique frequency to thrust relationship to enhance AUV localization capabilities. We construct a prototype and use it to experimentally demonstrate the feasibility of the directionally-tunable resonance concept.

- Hierarchical Planning in Time-Dependent Flow Fields for Marine Robots

    Author: Lee, James Ju Heon | University of Technology Sydney
    Author: Yoo, Chanyeol | University of Technology Sydney
    Author: Anstee, Stuart David | Defence Science and Technology Group
    Author: Fitch, Robert | University of Technology Sydney
 
    keyword: Marine Robotics; Motion and Path Planning; Field Robots

    Abstract : We present an efficient approach for finding shortest paths in flow fields that vary as a sequence of flow predictions over time. This approach is applicable to motion planning for slow marine robots that are subject to dynamic ocean currents. Although the problem is NP-hard in general form, we incorporate recent results from the theory of finding shortest paths in time-dependent graphs to construct a polynomial-time algorithm that finds continuous trajectories in time-dependent flow fields. The algorithm has a hierarchical structure where a graph is constructed with time-varying edge costs that are derived from sets of continuous trajectories in the underlying flow field. We show that the continuous algorithm retains the time complexity and path quality properties of the discrete graph solution, and demonstrate its application to surface and underwater vehicles including a traversal along the East Australian Current with an autonomous surface vehicle. Results show that the algorithm performs efficiently in practice and can find paths that adapt to changing ocean currents. These results are significant to marine robotics because they allow for efficient use of time-varying ocean predictions for motion planning.

- Navigation in the Presence of Obstacles for an Agile Autonomous Underwater Vehicle

    Author: Xanthidis, Marios | University of South Carolina
    Author: Karapetyan, Nare | University of South Carolina
    Author: Damron, Hunter | University of South Carolina
    Author: Rahman, Sharmin | University of South Carolina
    Author: Johnson, James | University of South Carolina
    Author: O'Connell, Allison | Vassar College
    Author: O'Kane, Jason | University of South Carolina
    Author: Rekleitis, Ioannis | University of South Carolina
 
    keyword: Marine Robotics; Motion Control; Nonholonomic Motion Planning

    Abstract : Navigation underwater traditionally is done by keeping a safe distance from obstacles, resulting in "fly-overs" of the area of interest. Movement of an autonomous underwater vehicle (AUV) through a cluttered space, such as a shipwreck or a decorated cave, is an extremely challenging problem that has not been addressed in the past. This paper proposes a novel navigation framework utilizing an enhanced version of Trajopt for fast 3D path-optimization planning for AUVs. A sampling-based correction procedure ensures that the planning is not constrained by local minima, enabling navigation through narrow spaces. Two different modalities are proposed: planning with a known map results in efficient trajectories through cluttered spaces; operating in an unknown environment utilizes the point cloud from the visual features detected to navigate efficiently while avoiding the detected obstacles. The proposed approach is rigorously tested, both on simulation and in-pool experiments, proven to be fast enough to enable safe real-time 3D autonomous navigation for an AUV.

- Underwater Image Super-Resolution Using Deep Residual Multipliers

    Author: Islam, Md Jahidul | University of Minnesota-Twin Cities
    Author: Enan, Sadman Sakib | University of Minnesota, Twin Cities
    Author: Luo, Peigen | University of Minnesota, Twin Cities
    Author: Sattar, Junaed | University of Minnesota
 
    keyword: Marine Robotics; Deep Learning in Robotics and Automation; Computer Vision for Automation

    Abstract : We present a deep residual network-based generative model for single image super-resolution (SISR) of underwater imagery for use by autonomous underwater robots. We also provide an adversarial training pipeline for learning SISR from paired data. In order to supervise the training, we formulate an objective function that evaluates the perceptual quality of an image based on its global content, color, and local style information. Additionally, we present USR-248, a large-scale dataset of three sets of underwater images of 'high' (640x480) and 'low' (80x60, 160x120, and 320x240) resolution. USR-248 contains paired instances for supervised training of 2x, 4x, or 8x SISR models. Furthermore, we validate the effectiveness of our proposed model through qualitative and quantitative experiments and compare the results with several state-of-the-art models' performances. We also analyze its practical feasibility for applications such as scene understanding and attention modeling in noisy visual conditions.

- Nonlinear Synchronization Control for Short-Range Mobile Sensors Drifting in Geophysical Flows

    Author: Wei, Cong | University of Delaware
    Author: Tanner, Herbert G. | University of Delaware
    Author: Hsieh, M. Ani | University of Pennsylvania
 
    keyword: Marine Robotics; Cooperating Robots; Multi-Robot Systems

    Abstract : This paper presents a synchronization controller for mobile sensors that are minimally actuated and can only communicate with each other over a very short range. Thiswork is motivated by ocean monitoring applications wherelarge-scale sensor networks consisting of drifters with minimalactuation capabilities,i.e.,activedrifters, are employed. We assume drifters are tasked to monitor regions consisting of gyre flows where their trajectories are periodic. As driftersin neighboring regions move into each other's proximity, itpresents an opportunity for data exchange and synchronization to ensure future rendezvous. We present a nonlinearsynchronization control strategy to ensure that drifters willperiodically rendezvous and maximize the time they are intheir rendezvous regions. We present numerical simulations andsmall-scale experiments to validat the efficacy of the controlstrategy and discuss the extension to large-scale mobile sensornetworks.

## Compliant Joint/Mechanism
- Energy-Based Safety in Series Elastic Actuation

    Author: Roozing, Wesley | University of Twente
    Author: Groothuis, Stefan S. | University of Twente
    Author: Stramigioli, Stefano | University of Twente
 
    keyword: Compliant Joint/Mechanism; Compliance and Impedance Control; Robot Safety

    Abstract : This work presents the concept of energy-based safety for series-elastic actuation. Generic actuation passivity and safety is treated, defining several energy storage and power flow properties related to passivity. Safe behaviour is not guaranteed by passivity, but can be guaranteed by energy and power limits that adapt the nominal behaviour of an impedance controller. A discussion on power flows in series-elastic actuation is presented and an appropriate controller is developed. Experimental results validate the effectiveness of the energy-based safety in elastic actuation.

- Safe High Impedance Control of a Series-Elastic Actuator with a Disturbance Observer

    Author: Haninger, Kevin | Fraunhofer IPK
    Author: Asignacion, Abner Jr | Daegu Gyeongbuk Institute of Science and Technology
    Author: Oh, Sehoon | DGIST (Daegu Gyeongbuk Institute of Science and Technology)
 
    keyword: Compliant Joint/Mechanism; Compliance and Impedance Control; Force Control

    Abstract : In many series-elastic actuator applications, the ability to safely render a wide range of impedance is important. Advanced torque control techniques such as the disturbance observer (DOB) can improve torque tracking performance, but their impact on safe impedance range is not established. Here, safety is defined with load port passivity, and passivity conditions are developed for two variants of DOB torque control. These conditions are used to determine the maximum safe stiffness and Z-region of the DOB controllers, which are analyzed and compared with the no DOB case. A feedforward controller is proposed which increases the maximum safe stiffness of the DOB approaches. The results are experimentally validated by manual excitation and in a high-stiffness environment.

- Variable Stiffness Springs for Energy Storage Applications

    Author: Kim, Sung | Vanderbilt University
    Author: Tiange, Zhang | Vanderbilt University
    Author: Braun, David | Vanderbilt University
 
    keyword: Compliant Joint/Mechanism; Human Performance Augmentation; Hydraulic/Pneumatic Actuators

    Abstract : Theory suggests an inverse relation between the stiffness and the energy storage capacity for linear helical springs: reducing the active length of the spring by 50% increases its stiffness by 100%, but reduces its energy storage capacity by 50%. State-of-the-art variable stiffness actuators used to drive robots are characterized by a similar inverse relation, implying reduced energy storage capacity for increased spring stiffness. This relation limits the potential of the variable stiffness actuation technology when it comes to human performance augmentation in natural tasks, e.g., jumping, weight-bearing and running, which may necessitate a spring exoskeleton with large stiffness range and high energy storage capacity. In this paper, we theoretically show that the trade-off between stiffness range and energy storage capacity is not fundamental; it is possible to develop variable stiffness springs with simultaneously increasing stiffness and energy storage capacity. Consistent with the theory, we experimentally show that a controllable volume air spring, has a direct relation between its stiffness range and energy storage capacity. The mathematical conditions presented in this paper may be used to develop actuators that could bypass the limited energy storage capacity of current variable stiffness spring technology.

- Parallel-Motion Thick Origami Structure for Robotic Design

    Author: Liu, Shuai | Hong Kong University of Science and Technology
    Author: Wu, Huajie | University of Science and Technology of China
    Author: Yang, Yang | The Hong Kong University of Science and Technology
    Author: Wang, Michael Yu | Hong Kong University of Science &amp; Technology
 
    keyword: Compliant Joint/Mechanism; Flexible Robots; Soft Robot Materials and Design

    Abstract : Structures with origami design enable objects to transform into various three-dimensional shapes. Traditionally origami structures are designed with zero-thickness flat paper sheets. However, the thickness and intersection of origami facets are non-negligible in most cases, especially when origami design is integrated with robotic design because of the more efficient force transfer between thick plates compared with zero-thickness paper-sheets. Meanwhile, the single-layer-paper oriented initial design limited the shape transformation potential as multiple layer origami structures could conduct more variety of deformation. In this article, we are proposing a general design method of parallel-motion thick origami structures which could be used in robotic design like a parallel-motion gripper.

- Gyroscopic Tensegrity Robots

    Author: Goyal, Raman | Texas A&amp;M University
    Author: Chen, Muhao | Texas A&amp;M University
    Author: Majji, Manoranjan | Texas A&amp;M University
    Author: Skelton, Robert | Texas A&amp;M University
 
    keyword: Dynamics; Modeling, Control, and Learning for Soft Robots; Space Robotics and Automation

    Abstract : Mechanics and control of innovative gyroscopic structural systems are detailed in the paper. By adding controllable spinning wheels to a network of controllable, axially loaded strings and bars, it is shown that the mobility and manipulation of the structural system are enhanced. Using principles of mechanics, a nonlinear dynamic model is presented to modulate the torque produced by the network of spatially distributed gyroscopes. Equations of motion, formulated as a second-order matrix differential equation, provide a trajectory for the nodal displacement of the bars, along with the wheel's spin degree of freedom. While the gyroscopic robotics concept is scalable to an arbitrarily large network, this research aims to identify elemental modules to override fundamental design principles of the innovative structural systems. Dynamic simulation and experimental verification on a planar D-bar tensegrity structure are used to demonstrate the utility of one such fundamental building block of the gyroscopic robotic system.

## Search and Rescue Robots
- Real-Time Simulation of Non-Deformable Continuous Tracks with Explicit Consideration of Friction and Grouser Geometry

    Author: Okada, Yoshito | Tohoku University
    Author: Kojima, Shotaro | Tohoku University
    Author: Ohno, Kazunori | Tohoku University
    Author: Tadokoro, Satoshi | Tohoku University
 
    keyword: Search and Rescue Robots; Simulation and Animation; Field Robots

    Abstract : In this study, we developed a real-time simulation method for non-deformable continuous tracks having grousers for rough terrain by explicitly considering the collision and friction between the tracks and the ground. In the proposed simulation method, an arbitrary trajectory of a track is represented with multiple linear and circular segments, each of which is a link connected to a robot body. The proposed method sets velocity constraints between each segment link and the robot body, to simulate the track rotation around the body. To maintain the shape of a track, it also restores the positions of the segment links when required. Experimental comparisons with other existing real-time simulation methods demonstrated that while the proposed method considered the grousers and the friction with the ground, it was comparable to them in terms of the computational speed. Experimental comparison of the simulations based on the proposed method and a physical robot exhibited that the former was comparable to the precise motion of the robot on rough or uneven terrain.

- Test Your SLAM! the SubT-Tunnel Dataset and Metric for Mapping

    Author: Rogers III, John G. | US Army Research Laboratory
    Author: Gregory, Jason M. | US Army Research Laboratory
    Author: Fink, Jonathan | US Army Research Laborator
    Author: Stump, Ethan | US Army Research Laboratory
 
    keyword: Performance Evaluation and Benchmarking; SLAM; Search and Rescue Robots

    Abstract : This paper presents an approach and introduces new open-source tools which can be used to evaluate robotic mapping algorithms, in addition to an extensive subterranean mine rescue dataset based upon the DARPA Subterranean challenge including professionally surveyed ground truth. Finally, some commonly available approaches are evaluated using this metric.

- Uncertainty Measured Markov Decision Process in Dynamic Environments

    Author: Dutta, Sourav | University at Albany
    Author: Rekabdar, Banafsheh | Southern Illinois University Carbondale
    Author: Ekenna, Chinwe | University at Albany
 
    keyword: Surveillance Systems; Search and Rescue Robots; Localization

    Abstract : Successful robot path planning is challenging in the presence of visual occlusions and moving targets. Classical methods to solve this problem have used visioning and perception algorithms in addition to partially observable markov decision processes to aid in path planning for pursuit-evasion and robot tracking.<p>We present a predictive path planning process that measures and utilizes the uncertainty present during robot motion planning. We develop a variant of subjective logic in combination with the Markov decision process (MDP) and provide a measure for belief, disbelief, and uncertainty in relation to feasible trajectories being generated. We then model the MDP to identify the best path planning method from a list of possible choices. Our results show a high percentage accuracy based on the closest acquired proximity between a target and a tracking robot and a simplified pursuer trajectory in comparison with related work.

- A Minimally Actuated Reconfigurable Continuous Track Robot

    Author: Kislassi, Tal | Ben Gurion University of the Negev
    Author: Zarrouk, David | Ben Gurion University

- Cooperative Mapping and Target Search Over an Unknown Occupancy Graph Using Mutual Information

    Author: Wolek, Artur | University of Maryland
    Author: Cheng, Sheng | University of Maryland, College Park
    Author: Goswami, Debdipta | University of Maryland
    Author: Paley, Derek | University of Maryland
 
    keyword: Search and Rescue Robots; Cooperating Robots; Sensor Networks

    Abstract : A cooperative mapping and target-search algorithm is presented for detecting a single moving ground target in an urban environment that is initially unknown to a team of autonomous quadrotors equipped with noisy, range-limited sensors. The target moves according to a biased random-walk model, and search agents (quadrotors) build a target state graph that encodes past and present target positions. A track-before-detect algorithm assimilates target measurements into the log-likelihood ratio and anisotropic kriging interpolation predicts the location of occupancy nodes in unexplored regions. Mutual information evaluated at each location in the search area defines a sampling-priority surface that is partitioned by a weighted Voronoi algorithm into candidate waypoint tasks. Tasks are assigned to each agent by iteratively solving a utility-maximizing assignment problem. Numerical simulations show that the proposed approach compares favorably to non-adaptive lawnmower and random coverage strategies. The proposed strategy is also demonstrated experimentally through an outdoor flight test using two real and two virtual quadrotors.

- Flexible Disaster Response of Tomorrow - Final Presentation and Evaluation of the CENTAURO System (I)

    Author: Klamt, Tobias | University of Bonn
    Author: Rodriguez, Diego | University of Bonn
    Author: Baccelliere, Lorenzo | Istituto Italiano Di Tecnologia
    Author: Chen, Xi | KTH
    Author: Chiaradia, Domenico | Scuola Superiore Sant'Anna, TeCIP Institute, PERCRO Laboratory,
    Author: Cichon, Torben | RWTH Aachen University
    Author: Gabardi, Massimiliano | Scuola Superiore Sant'Anna PERCRO
    Author: Guria, Paolo | Istituto Italiano Di Tecnologia
    Author: Holmquist, Karl | Link�ping University
    Author: Kamedula, Malgorzata | Istituto Italiano Di Tecnologia
    Author: Karaoguz, Hakan | Royal Institute of Technology KTH
    Author: Kashiri, Navvab | Istituto Italiano Di Tecnologia
    Author: Laurenzi, Arturo | Istituto Italiano Di Tecnologia
    Author: Lenz, Christian | University of Bonn
    Author: Leonardis, Daniele | Scuola Superiore Sant'Anna - TeCIP Institute
    Author: Mingo, Enrico | Istituto Italiano Di Tecnologia
    Author: Muratore, Luca | Istituto Italiano Di Tecnologia
    Author: Pavlichenko, Dmytro | University of Bonn
    Author: Porcini, Francesco | PERCRO Laboratory, TeCIP Institute, Sant�Anna School of Advanced
    Author: Ren, Zeyu | Istituto Italiano Di Tecnologia
    Author: Schilling, Fabian | EPFL
    Author: Schwarz, Max | University Bonn
    Author: Solazzi, Massimiliano | Scuola Superiore Sant'Anna, TeCIP Institute
    Author: Felsberg, Michael | Link�ping University
    Author: Frisoli, Antonio | TeCIP Institute, Scuola Superiore Sant'Anna
    Author: Gustmann, Michael | Kerntechnische Hilfsdienst GmbH
    Author: Jensfelt, Patric | KTH - Royal Institute of Technology
    Author: Nordberg, Klas | Link�ping University
    Author: Rossmann, Juergen | RWTH Aachen University
    Author: Suess, Uwe | Kerntechnische Hilfsdienst GmbH
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
    Author: Behnke, Sven | University of Bonn
 
    keyword: Search and Rescue Robots; Mobile Manipulation; Robotics in Hazardous Fields

    Abstract : Mobile manipulation robots have high potential to support rescue forces in disaster-response missions. Despite the difficulties imposed by real-world scenarios, robots are promising to perform mission tasks from a safe distance. In the CENTAURO project, we developed a disaster-response system which consists of the highly flexible Centauro robot and suitable control interfaces including an immersive tele-presence suit and support-operator controls on different levels of autonomy. In this article, we give an overview of the final CENTAURO system. In particular, we explain several high-level design decisions and how those were derived from requirements and extensive experience of Kerntechnische Hilfsdienst GmbH, Karlsruhe, Germany (KHG). We focus on components which were recently integrated and report about a systematic evaluation which demonstrated system capabilities and revealed valuable insights.

## Human Detection and Tracking
- Natural Scene Facial Expression Recognitionwith Dimension Reduction Network

    Author: Hu, Shenhua | Institute of Automation, Chinese Academy of Sciences
    Author: Yiming, Hu | CASIA
    Author: Li, Jianquan | Institute of Automation, Chinese Academy of Sciences
    Author: Long, Xianlei | Institute of Automation, Chinese Academy of Sciences
    Author: Chen, Mengjuan | University of Chinese Academy of Sciences
    Author: Gu, Qingyi | Institute of Automation, Chinese Academy of Sciences
 
    keyword: Gesture, Posture and Facial Expressions; Recognition; Cognitive Human-Robot Interaction

    Abstract : As an external manifestation of human emotions, expression recognition plays an important role in human-computer interaction. Although existing expression recognition methods performs perfectly on constrained frontal faces, there are still many challenges in expression recognition in natural scenes due to different unrestricted conditions.Expression classification belongs to a pattern recognition problem where intra-class distance is greater than the inter-class distance, which leads to severe over-fitting when using neural networks for expression recognition. This paper proposes a novel network structure called Dimension Reduction Network which can effectively reduce generalization error. By adding a data dimension reduction module before the general classification network, a lot of redundant information is filtered, and only useful information is left.This can reduce the interference by irrelevant information when performing classification tasks and reduce generalization error. The proposed method does not require any modification to the classification network, only a small dimension reduction module needs to be added in front of the classification network. However, it can effectively reduce generalization error. We designed big and tiny versions of Dimension Reduction Network, both exceeds our baseline on AffectNet data set. The big version of our proposed method surpassed the state-of-the-art methods by more than 1.2% on AffectNet data set.

- Hand Pose Estimation for Hand-Object Interaction Cases Using Augmented Autoencoder

    Author: Li, Shile | Technische Universitét M�nchen
    Author: Wang, Haojie | Technische Universitét M�nchen
    Author: Lee, Dongheui | Technical University of Munich
 
    keyword: Human Detection and Tracking; RGB-D Perception; Deep Learning in Robotics and Automation

    Abstract : Hand pose estimation with objects is challenging due to object occlusion and the lack of large annotated datasets. To tackle these issues, we propose an Augmented Autoencoder based deep learning method using augmented clean hand data. Our method takes 3D point cloud of a hand with an augmented object as input and encodes the input to latent representation of the hand. From the latent representation, our method decodes 3D hand pose and we propose to use an auxiliary point cloud decoder to assist the formation of the latent space. Through quantitative and qualitative evaluation on both synthetic dataset and real captured data containing objects, we demonstrate state-of-the-art performance for hand pose estimation with objects.

- Accurate Detection and 3D Localization of Humans Using a Novel YOLO-Based RGB-D Fusion Approach and Synthetic Training Data

    Author: Linder, Timm | Robert Bosch GmbH
    Author: Pfeiffer, Kilian Yutaka | RWTH Aachen University
    Author: Vaskevicius, Narunas | Robert Bosch GmbH
    Author: Schirmer, Robert | Robert Bosch GmbH
    Author: Arras, Kai Oliver | Bosch Research
 
    keyword: Human Detection and Tracking; RGB-D Perception; Object Detection, Segmentation and Categorization

    Abstract : While 2D object detection has made significant progress, robustly localizing objects in 3D space under presence of occlusion is still an unresolved issue. Our focus in this work is on real-time detection of human 3D centroids in RGB-D data. We propose an image-based detection approach which extends the YOLOv3 architecture with a 3D centroid loss and mid-level feature fusion to exploit complementary information from both modalities. We employ a transfer learning scheme which can benefit from existing large-scale 2D object detection datasets, while at the same time learning end-to-end 3D localization from our highly randomized, diverse synthetic RGB-D dataset with precise 3D groundtruth. We further propose a geometrically more accurate depth-aware crop augmentation for training on RGB-D data, which helps to improve 3D localization accuracy. In experiments on our challenging intralogistics dataset, we achieve state-of-the-art performance even when learning 3D localization just from synthetic data.

- Pedestrian Planar LiDAR Pose (PPLP) Network for Oriented Pedestrian Detection Based on Planar LiDAR and Monocular Images

    Author: Bu, Fan | University of Michigan
    Author: Le, Trinh | University of Michigan
    Author: Du, Xiaoxiao | University of Michigan
    Author: Vasudevan, Ram | University of Michigan
    Author: Johnson-Roberson, Matthew | University of Michigan
 
    keyword: Human Detection and Tracking; Computer Vision for Automation; Recognition

    Abstract : Pedestrian detection is an important task for human-robot interaction and autonomous driving applications. Most previous pedestrian detection methods rely on data collected from three-dimensional (3D) Light Detection and Ranging (LiDAR) sensors in addition to camera imagery, which can be expensive to deploy. In this paper, we propose a novel Pedestrian Planar LiDAR Pose Network (PPLP Net) based on two-dimensional (2D) LiDAR data and monocular camera imagery, which offers a far more affordable solution to the oriented pedestrian detection problem. The proposed PPLP Net consists of three sub-networks: an orientation detection network (OrientNet), a Region Proposal Network (RPN), and a PredictorNet. The OrientNet leverages state-of-the-art neural-network-based 2D pedestrian detection algorithms, including Mask R-CNN and ResNet, to detect the Bird's Eye View (BEV) orientation of each pedestrian. The RPN transfers 2D LiDAR point clouds into occupancy grid map and uses a frustum-based matching strategy for estimating non-oriented 3D pedestrian bounding boxes. Outputs from both OrientNet and RPN are passed through the PredictorNet for a final regression. The overall outputs of our proposed network are 3D bounding box locations and orientation values for all pedestrians in the scene. We present oriented pedestrian detection results on two datasets, the CMU Panoptic Dataset and a newly collected FCAV M-Air Pedestrian (FMP) Dataset, and show that our proposed PPLP network based on 2D L

- Wide-Range Load Sensor Using Vacuum Sealed Quartz Crystal Resonator for Simultaneous Biosignals Measurement on Bed
 
    Author: Murozaki, Yuichi | Nagoya University
    Author: Arai, Fumihito | Nagoya University
 
    keyword: Human Detection and Tracking; Product Design, Development and Prototyping

    Abstract : Monitoring of biosignals on a daily basis plays important roles for the health management of elderly. The monitoring system for the daily life, the system should not require the subjects to take special effort like wearing a sensor. We propose biosignals measurement using wide-range load sensor on the bed. The sensing system can detect the body weight, heartbeat and respiration simultaneously by just lying on the bed. We have developed load sensor using quartz crystal resonator (QCR load sensor) as wide-range load sensor. However, the measurement range was not sufficient for the simultaneous measurement of biosgnals on bed. To realize such sensing system, we propose a QCR load sensor utilizing vacuum sealing technology for expanding the measurement range. We improved the oscillation characteristics of the QCR by the vacuum sealing to stabilize the sensor output. Accordingly, the resolution of the sensor was improved. Moreover, the load capacity of the sensor was increased by improving the bonding strength of sensor structure. The fabricated sensor had a measurement range of 0.27 mN - 1180 N (4.4 - 10^6). This wide enough compared with the conventional force sensor (10^3 - 10^4). Also, we developed mechanically robust jig of QCR load sensor for practical use of QCR load sensor. We succeed in simultaneous measurement of weight, heart rate, and respiration rate using fabricated QCR load sensing system. The accuracy of heart rate and respiration measurement are 0.6 % and 6.1 %,

- Joint Pedestrian Detection and Risk-Level Prediction with Motion-Representation-By-Detection

    Author: Kataoka, Hirokatsu | National Institute of Advanced Industrial Science and Technology
    Author: Suzuki, Teppei | Denso IT Laboratory, INC
    Author: Nakashima, Kodai | University of Tsukuba
    Author: Satoh, Yutaka | AIST
    Author: Aoki, Yoshimitsu | Keio University
 
    keyword: Human Detection and Tracking; Computer Vision for Other Robotic Applications; Object Detection, Segmentation and Categorization

    Abstract : The paper presents a pedestrian near-miss detector with temporal analysis that provides both pedestrian detection and risk-level predictions which are demonstrated on a self-collected database. Our work makes three primary contributions: (i) The framework of pedestrian near-miss detection is proposed by providing both a pedestrian detection and risk-level assignment. Specifically, we have created a Pedestrian Near-Miss (PNM) dataset that categorizes traffic near-miss incidents based on their risk levels (high-, low-, and no-risk). Unlike existing databases, our dataset also includes manually localized pedestrian labels as well as a large number of incident-related videos. (ii) Single-Shot MultiBox Detector with Motion Representation (SSD-MR) is implemented to effectively extract motion-based features in a detected pedestrian. (iii) Using the self-collected PNM dataset and SSD-MR, our proposed method achieved +19.38% (on risk-level prediction) and +13.00% (on joint pedestrian detection and risk-level prediction) higher scores than that of the baseline SSD and LSTM. Additionally, the running time of our system is over 50 fps on a graphics processing unit (GPU).

## Omnidirectional Vision and Audition
- Robust Sound Source Localization Considering Similarity of Back-Propagation Signals

    Author: An, Inkyu | KAIST
    Author: Jo, Byeongho | Korea Advanced Institute of Science and Technology
    Author: Kwon, Youngsun | KAIST
    Author: Choi, Jung-Woo | KAIST
    Author: Yoon, Sung-eui | KAIST
 
    keyword: Robot Audition; Localization

    Abstract : We present a novel, robust sound source localization algorithm considering back-propagation signals. Sound propagation paths are estimated by generating direct and reflection acoustic rays based on ray tracing in a backward manner. We then compute the back-propagation signals by designing and using the impulse response of the backward sound propagation based on the acoustic ray paths. For identifying the 3D source position, we use a well-established Monte Carlo localization method. Candidates for a source position are determined by identifying convergence regions of acoustic ray paths. Those candidates are validated by measuring similarities between back-propagation signals, under the assumption that the back-propagation signals of different acoustic ray paths should be similar near the ground-truth sound source position. Thanks to considering similarities of back-propagation signals, our approach can localize a source position with an averaged error of 0.55m in a room of 7m by 7m area with 3m height in tested environments. We also place additional 67dB and 77dB white noise at the background, to test the robustness of our approach. Overall, we observe a 7% to 100% improvement in accuracy over the state-of-the-art method.

- BatVision: Learning to See 3D Spatial Layout with Two Ears

    Author: Christensen, Jesper | Technical University of Denmark, ATLAS MARIDAN ApS
    Author: Hornauer, Sascha | International Computer Science Institute Berkeley
    Author: Yu, Stella | UC Berkeley / ICSI
 
    keyword: Robot Audition; Biologically-Inspired Robots; Deep Learning in Robotics and Automation

    Abstract : Images showing the 3D spatial layout of space ahead of a mobile agent can be generated by purely listening to the reflections of chirping sounds. Many species have evolved sophisticated non-visual perception while artificial systems fall behind. While radar and ultrasound are used where cameras fail, they either provide very limited information or require large, complex and expensive sensors. Sound, on the other hand, is used effortlessly by dolphins, bats, whales and humans as a sensor modality with many advantages over vision. However, it is challenging to harness useful and detailed information for machine perception. We train a network to generate representations of the world in 2D and 3D only from sounds, sent by one speaker and captured by two microphones. Inspired by examples from nature, we emit short frequency modulated sound chirps and record returning echoes through an artificial human pinnae pair. We then learn to generate disparity-like depth maps and grayscale images from the echoes in an end-to-end fashion. With only low-cost equipment, our models show good reconstruction performance while being robust to errors and even overcoming limitations of our vision-based ground truth. Finally, we introduce a large dataset consisting of binaural sound signals synchronized in time with both RGB images and depth maps.

- Self-Supervised Learning for Alignment of Objects and Sound

    Author: Liu, Xinzhu | Tsinghua University
    Author: Liu, XiaoYu | BeiJing Institute of Technology
    Author: Guo, Di | Tsinghua University
    Author: Liu, Huaping | Tsinghua University
    Author: Sun, Fuchun | Tsinghua Univerisity
    Author: Min, Haibo | Tsinghua University
 
    keyword: Robot Audition; Computer Vision for Other Robotic Applications

    Abstract : The sound source separation problem has many useful applications in the field of robotics, such as human-robot interaction, scene understanding, etc. However, it remains a very challenging problem. In this paper, we utilize both visual and audio information of videos to perform the sound source separation task. A self-supervised learning framework is proposed to implement the object detection and sound separation modules simultaneously. Such an approach is designed to better find the alignment between the detected objects and separated sound components. Our experiments, conducted on both the synthetic and real datasets, validate this approach and demonstrate the effectiveness of the proposed model in the task of object and sound alignment.

- Variational Fisheye Stereo

    Author: Roxas, Menandro | The University of Tokyo
    Author: Oishi, Takeshi | The University of Tokyo
 
    keyword: Omnidirectional Vision; Mapping

    Abstract : Dense 3D maps from wide-angle cameras is beneficial to robotics applications such as navigation and autonomous driving. In this work, we propose a real-time dense 3D mapping method for fisheye cameras without explicit rectification and undistortion. We extend the conventional variational stereo method by constraining the correspondence search along the epipolar curve using a trajectory field induced by camera motion. We also propose a fast way of generating the trajectory field without increasing the processing time compared to conventional rectified methods. With our implementation, we were able to achieve real-time processing using modern GPUs. Our results show the advantages of our non-rectified dense mapping approach compared to rectified variational methods and non-rectified discrete stereo matching methods.

- The OmniScape Dataset

    Author: Sekkat, Ahmed Rida | LITIS Lab, Université De Rouen Normandie
    Author: Dupuis, Yohan | ESIGELEC
    Author: Vasseur, Pascal | Université De Rouen
    Author: Honeine, Paul | LITIS Lab, Université De Rouen Normandie
 
    keyword: Omnidirectional Vision; Object Detection, Segmentation and Categorization; RGB-D Perception

    Abstract : Despite the utility and benefits of omnidirectional images in robotics and automotive applications, there are no dataset of omnidirectional images available with semantic segmentation, depth map, and dynamic properties. This is due to the time cost and human effort required to annotate ground truth images. This paper presents a framework for generating omnidirectional images using images that are acquired from a virtual environment. For this purpose, we demonstrate the relevance of the proposed framework on two well-known simulators: CARLA simulator, which is an open-source simulator for autonomous driving research, and Grand Theft Auto V (GTA V), which is a very high quality video game. We explain in details the generated OmniScape dataset, which includes stereo fisheye and catadioptric images acquired from the two front sides of a motorcycle, including semantic segmentation, depth map, intrinsic parameters of the cameras and the dynamic parameters of the motorcycle. It is worth noting that the case of two-wheeled vehicles is more challenging than cars due to the specific dynamic of these vehicles.

- Corners for Layout: End-To-End Layout Recovery from 360 Images

    Author: Fernandez-Labrador, Clara | University of Zaragoza
    Author: F�cil, Jos' M. | Universidad De Zaragoza
    Author: Perez-Yus, Alejandro | Universidad De Zaragoza
    Author: Demonceaux, C�dric | Université Bourgogne Franche-Comt�
    Author: Civera, Javier | Universidad De Zaragoza
    Author: Guerrero, Josechu | Universidad De Zaragoza
 
    keyword: Omnidirectional Vision; Semantic Scene Understanding; Computer Vision for Other Robotic Applications

    Abstract : The problem of 3D layout recovery in indoor scenes has been a core research topic for over a decade. However, there are still several major challenges that remain unsolved. Among the most relevant ones, a major part of the state-of-the-art methods make implicit or explicit assumptions on the scenes �e.g. box-shaped or Manhattan layouts. Also, current methods are computationally expensive and not suitable for real-time applications like robot navigation and AR/VR. In this work we present CFL (Corners for Layout), the first end-to-end model that predicts layout corners for 3D layout recovery on 360� images. Our experimental results show that we outperform the state of the art, making less assumptions on the scene than other works, and with lower cost. We also show that our model generalizes better to camera position variations than conventional approaches by using EquiConvs, a convolution applied directly on the spherical projection and hence invariant to the equirectangular distortions.


## Hydraulic/Pneumatic Actuators
- How Far Are Pneumatic Artificial Muscles from Biological Muscles?

    Author: Mohseni, Omid | University of Tehran
    Author: Gagey, Ferr�ol | École Normale Paris-Saclay
    Author: Zhao, Guoping | Technical University of Darmstadt
    Author: Seyfarth, Andre | TU Darmstadt
    Author: Ahmad Sharbafi, Maziar | Technical University of Darmstadt
 
    keyword: Hydraulic/Pneumatic Actuators

    Abstract : There is a long history demonstrating humans' tendency to create artificial copies of living creatures. For moving machines called robots, actuators play a key role in developing human-like movements. Among different types of actuation, PAMs (pneumatic artificial muscles) are known as the most similar ones to biological muscles. In addition to similarities in force generation mechanism (tension based), the well-accepted argumentation from Klute et al., states that the PAM force-length (<i>f<sub>l</sub></i>) behavior is close to biological muscles, while the force-velocity (<i>f<sub>v</sub></i>) pattern is different. Using the multiplicative formulation of the pressure (as an activation term), <i>f<sub>l</sub></i> and <i>f<sub>v</sub></i> beside an additive passive parallel elastic element, we present a new model of PAM. This muscle-based model can predict PAM dynamic behaviors with high precision. With a second experiment in a two-segmented leg, the proposed model is verified to predict the generated forces of PAMs in an antagonistic arrangement. Such a dynamic muscle-like model of artificial muscles can be used for the design and control of legged robots to generate robust, efficient and versatile gaits.

- Optically Sensorized Elastomer Air Chamber for Proprioceptive Sensing of Soft Pneumatic Actuator

    Author: Jung, Jaewoong | Seoul National University
    Author: Park, Myungsun | Seoul National University
    Author: Kim, DongWook | Seoul National University
    Author: Park, Yong-Lae | Seoul National University
 
    keyword: Hydraulic/Pneumatic Actuators; Soft Sensors and Actuators; Soft Robot Materials and Design

    Abstract : Soft robotics has proven the capability of robots interacting with their environments including humans by taking advantage of the property of high compliance in recent years. Soft pneumatic actuators are one of the most commonly used actuation systems in soft robotics. However, control of a highly compliant actuation system remains as a challenging issue due to its nonlinearity and hysteresis. Addressing this problem requires integration of a soft sensing mechanism with the actuator for proprioceptive feedback. A soft optical waveguide with a reflective metal coating is a promising sensing mechanism with high compliance and low hysteresis. In this paper, we propose design and fabrication of a soft pneumatic actuator integrated with an optical waveguide that can provide the proprioceptive information of the actuator. We describe the de- sign and fabrication, and present experimental characterization results of the proposed system. We also provide applications of the proposed system.

- A Compact McKibben Muscle Based Bending Actuator for Close-To-Body Application in Assistive Wearable Robots

    Author: Tschiersky, Martin | University of Twente
    Author: Hekman, Edsko E.G. | University of Twente
    Author: Brouwer, Dannis M. | University of Twente
    Author: Herder, Just | Delft University of Technology
    Author: Suzumori, Koichi | Tokyo Institute of Technology
 
    keyword: Hydraulic/Pneumatic Actuators; Wearable Robots; Physically Assistive Devices

    Abstract : In this letter we demonstrate a pneumatic bending actuator for upper-limb assistive wearable robots which uses thin McKibben muscles in combination with a flexure strip. The actuator features both active soft actuation and passive gravity support, and in terms of force transmission bridges the gap between the classic rigid type actuators and the emerging soft actuator technologies. Its flexure strip leverages the high-force low-displacement properties of McKibben muscles towards a large rotational range of motion and reduces localized forces at the attachments. We explain the synthesis method by which these actuators can be obtained and optimized for high specific moment output. Physical specimens of three optimized actuator designs are built and tested on a dedicated experimental setup, verifying the computational models. Furthermore, a proof-of-concept upper-limb assistive wearable robot is presented to illustrate a practical application of this actuator and its potential for close-to-body alignment. We found that based on our currently available components actuators can be built which, given a width of 80 mm, are able to produce a moment exceeding 4 Nm at an arm elevation of 90 deg.

- Proposal and Prototyping of Self-Excited Pneumatic Actuator Using Automatic-Flow-Path-Switching-Mechanism

    Author: Tani, Kosuke | Tokyo Institute of Technology
    Author: Nabae, Hiroyuki | Tokyo Institute of Technology
    Author: Endo, Gen | Tokyo Institute of Technology
    Author: Suzumori, Koichi | Tokyo Institute of Technology
 
    keyword: Hydraulic/Pneumatic Actuators; Additive Manufacturing; Mechanism Design

    Abstract : Robots currently have a wide range of practical applications. However, their widespread use is limited by long design and manufacturing times as well as increasingly complex drive system electronics and software, which have led to high development costs. Therefore, simpler manufacturing, driving, and control methods are required. In this study, we design a pneumatic actuator drive system that combines the printing technique and self-excited vibration. In the proposed actuator, a mechanism for automatically switching the airflow path is used to induce self-excited vibration. Moreover, the actuator is integrally molded by a 3D printer; therefore, no assembly process is required. This actuator can be used to easily build robots in a short time, contributing to more widespread use of robots. In this study, we also calculate the theoretical value of the moving frequency by modeling the actuator and verify the validity of this value through experiments using a prototype actuator. Based on the results, we were able to freely design the operating frequency of the actuator; by using this knowledge, we designed a flapping robot. The robot is also integrally molded by a 3D printer. Finally, we validate its motion through experiments, in order to illustrate one of the many applications of the proposed actuator.

- Development of Backdrivable Servovalve with Feedback Spring for Enhanced Electro-Hydraulic Torque Actuator

    Author: Nam, Seokho | POSTECH
    Author: Lee, Woongyong | POSTECH
    Author: Yoo, Sunkyum | POSTECH
    Author: Kim, Keehoon | POSTECH, Pohang University of Science and Technology
    Author: Chung, Wan Kyun | POSTECH
 
    keyword: Hydraulic/Pneumatic Actuators; Physical Human-Robot Interaction; Product Design, Development and Prototyping

    Abstract : This paper proposes a novel backdrivable servovalve to implement a torque-controlled electro-hydraulic actuator. The proposed backdrivable servovalve simultaneously contains the characteristics of a flow-control servo valve (= feedback spring) and a pressure-control servovalve (= pressure feedback port). Consequently, the torque control performance of electrohydraulic torque actuators (EHTAs), which consist of the backdrivable servovalve and rotary vane-type hydraulic actuators, can be enhanced. First, the effective torque bandwidth, which represents the region where the torque output remains constant within a 90deg phase lag, is improved. Second, torque-based control algorithms developed for electric motors are realized in a wide frequency region. Finally, although the effect of viscous friction increases, the effect of static friction is reduced. With these advantages, we achieved an enhanced EHTA with calculated torque output of -372~355Nm; this could facilitate in the development of a high-performance interactive robot system. The proposed backdrivable servovalve and enhanced EHTA were evaluated through experiments.

- Passivity-Based Robust Compliance Control of Electro-Hydraulic Robot Manipulators with Joint Angle Limit

    Author: Lee, Woongyong | POSTECH
    Author: Yoo, Sunkyum | POSTECH
    Author: Nam, Seokho | POSTECH
    Author: Kim, Keehoon | POSTECH, Pohang University of Science and Technology
    Author: Chung, Wan Kyun | POSTECH
 
    keyword: Hydraulic/Pneumatic Actuators; Compliance and Impedance Control; Robust/Adaptive Control of Robotic Systems

    Abstract : This paper presents a robust compliance control scheme for an electro-hydraulic robot manipulator with an electro-hydraulic torque actuators (EHTAs) and joint torque sensors. The EHTA, a torque-sourced hydraulic actuator, consists of electro-hydraulic backdrivable servovalve and a rotary hydraulic vane actuator and it allows us to design controllers similar to those for robot manipulators with electric motors. However, unlike in electric motors, the EHTA has a limited rotational angle and this may lead to instability when using a non-passive robust controller that generates the energy. As a solution to this problem, this paper proposes a robust two-loop control structure that has a passivity-based disturbance observer as an inner-loop controller and a nominal state feedback compliance controller as an outer-loop controller. The proposed control method was evaluated through single-degree-of-freedom experiments.

## Service Robots
- Shared Control Templates for Assistive Robotics

    Author: Quere, Gabriel | DLR
    Author: Hagengruber, Annette | German Aerospace Center
    Author: Iskandar, Maged | German Aerospace Center - DLR
    Author: Bustamante, Samuel | German Aeroespace Center (DLR), Robotics and Mechatronics Center
    Author: Leidner, Daniel | German Aerospace Center (DLR)
    Author: Stulp, Freek | DLR - Deutsches Zentrum F�r Luft Und Raumfahrt E.V
    Author: Vogel, J�rn | German Aerospace Center
 
    keyword: Service Robots; Control Architectures and Programming; Rehabilitation Robotics

    Abstract : Light-weight robotic manipulators can be used to restore the manipulation capability of people with motor disability. However, manipulating the environment poses a complex task, especially when the control interface is of low bandwidth, as may be the case for users with impairments. Therefore, we propose a constraint-based shared control scheme to define skills which provide support during task execution. This is achieved by representing a skill as a sequence of states, with specific user command mappings and different sets of constraints being applied in each state. New skills can be defined based on different types of constraints and conditions for state transition in a human readable manner. We demonstrate its versatility in a pilot experiment with three activities of daily living. Results show that even complex, high-dimensional tasks can be performed with a low-dimensional interface using our shared control approach.

- Enabling Robots to Understand Incomplete Natural Language Instructions Using Commonsense Reasoning

    Author: Chen, Haonan | University of North Carolina at Chapel Hill
    Author: Tan, Hao | UNC Chapel Hill
    Author: Kuntz, Alan | University of Utah
    Author: Bansal, Mohit | Unc Chapel Hill
    Author: Alterovitz, Ron | University of North Carolina at Chapel Hill
 
    keyword: Service Robots

    Abstract : Enabling robots to understand instructions provided via spoken natural language would facilitate interaction between robots and people in a variety of settings in homes and workplaces. However, natural language instructions are often missing information that would be obvious to a human based on environmental context and common sense, and hence does not need to be explicitly stated. In this paper, we introduce Language-Model-based Commonsense Reasoning (LMCR), a new method which enables a robot to listen to a natural language instruction from a human, observe the environment around it, and automatically fill in information missing from the instruction using environmental context and a new commonsense reasoning approach. Our approach first converts an instruction provided as unconstrained natural language into a form that a robot can understand by parsing it into verb frames. Our approach then fills in missing information in the instruction by observing objects in its vicinity and leveraging commonsense reasoning. To learn commonsense reasoning automatically, our approach distills knowledge from large unstructured textual corpora by training a language model. Our results show the feasibility of a robot learning commonsense knowledge automatically from web-based textual corpora, and the power of learned commonsense reasoning models in enabling a robot to autonomously perform tasks based on incomplete natural language instructions.

- A Holistic Approach in Designing Tabletop Robot's Expressivity

    Author: Gomez, Randy | Honda Research Institute Japan Co., Ltd
    Author: Nakamura, Keisuke | Honda Research Institute Japan Co., Ltd
    Author: Szapiro, Deborah | University of Technology Sydney
    Author: Merino, Luis | Universidad Pablo De Olavide
 
    keyword: Robot Companions; Gesture, Posture and Facial Expressions

    Abstract : Defining a robot's expressivity is a difficult task that requires thoughtful consideration of the potential of various robot modalities and a model of communication that humans understand. Humanoid and zoomorphic-designed robots can easily take cues from human and animals, respectively when designing their expressivity. However, a robot design that is neither human nor animal-like does not have a clear model to follow in terms of designing expressivity. Animation presents a potential model in these circumstances as animated characters in movies take various forms, sizes, shapes and styles, and are successful in defining expressivity that is widely accepted across different languages and cultures. In this paper, we discuss the development and design of the expressivity of a table top robot that is neither human nor animal-like and the application of animation expertise to the holistic treatment of the different modalities. The method maximises animation techniques and expertise normally applied to movies to generate expressivity that is then transferred to the robot hardware. Experimental results show that the robot's expressivity generated using our method is easily understood and are preferred than the conventional approach of generating expressions

- DirtNet: Visual Dirt Detection for Autonomous Cleaning Robots

    Author: Bormann, Richard | Fraunhofer IPA
    Author: Wang, Xinjie | Fraunhofer IPA
    Author: Xu, Jiawen | Fraunhofer IPA
    Author: Schmidt, Joel | University of Stuttgart
 
    keyword: Service Robots; Object Detection, Segmentation and Categorization; Computer Vision for Other Robotic Applications

    Abstract : Visual dirt detection is becoming an important capability for modern professional cleaning robots both for optimizing their wet cleaning results and for facilitating demand-oriented daily vacuum cleaning. This paper presents a robust, fast, and reliable dirt and office item detection system for these tasks based on an adapted YOLOv3 framework. Its superiority over state-of-the-art dirt detection systems is demonstrated in several experiments. The paper furthermore features a dataset generator for creating any number of realistic training images from a small set of real scene, dirt, and object examples.

- Semantic Linking Maps for Active Visual Object Search

    Author: Zeng, Zhen | University of Michigan
    Author: Röfer, Adrian | University of Bremen
    Author: Jenkins, Odest Chadwicke | University of Michigan
 
    keyword: Service Robots; Autonomous Agents; Domestic Robots 

    Abstract : We aim for mobile robots to function in a variety of common human environments. Such robots need to be able to reason about the locations of previously unseen target objects. Landmark objects can help this reasoning by narrowing down the search space significantly. More specifically, we can exploit background knowledge about common spatial relations between landmark and target objects. For example, seeing a table and knowing that cups can often be found on tables aids the discovery of a cup. Such correlations can be expressed as distributions over possible pairing relationships of objects. In this paper, we propose an active visual object search strategy method through our introduction of the Semantic Linking Maps (SLiM) model. SLiM simultaneously maintains the belief over a target object's location as well as landmark objects' locations, while accounting for probabilistic inter-object spatial relations. Based on SLiM, we describe a hybrid search strategy that selects the next best view pose for searching for the target object based on the maintained belief. We demonstrate the efficiency of our SLiM-based search strategy through comparative experiments in simulated environments. We further demonstrate the real-world applicability of SLiM-based search in scenarios with a Fetch mobile manipulation robot.

- ALTER-EGO: A Mobile Robot with Functionally Anthropomorphic Upper Body Designed for Physical Interaction (I)

    Author: Lentini, Gianluca | University of Pisa
    Author: Settimi, Alessandro | Université Di Pisa
    Author: Caporale, Danilo | Centro Di Ricerca E. Piaggio
    Author: Garabini, Manolo | Université Di Pisa
    Author: Grioli, Giorgio | Istituto Italiano Di Tecnologia
    Author: Pallottino, Lucia | Université Di Pisa
    Author: Catalano, Manuel Giuseppe | Istituto Italiano Di Tecnologia
    Author: Bicchi, Antonio | Université Di Pisa
 
    keyword: Humanoid Robots; Service Robots; Telerobotics and Teleoperation

    Abstract : In this work we present ALTER-EGO, an open-source mobile robot with a functionally anthropomorphic upper body, designed to operate in different environments, and equipped with soft robotics technologies to enable safe physical interactions with humans and the environment, to guarantee robustness and to allow versatility. ALTER-EGO is powered by Variable Stiffness Actuators and each arm mounts an anthropomorphic synergistically actuated artificial hand. The upper body is integrated with a two-wheels self-balancing mobile base to minimize the robot footprint and increase agility. ALTEREGO features also sensors and a computational system which, together with a modular pilot station, make the robot able to function either autonomously or in an, optionally immersive, teleoperation mode. Finally, a plausible use case scenario - a robot avatar in a domestic environment - is described and investigated through a preliminary experimental session.

## Robot Perception
- Active Depth Estimation: Stability Analysis and Its Applications

    Author: T. Rodrigues, R�mulo | Faculty of Engineering, University of Porto
    Author: Miraldo, Pedro | Instituto Superior Técnico, Lisboa
    Author: Dimarogonas, Dimos V. | KTH Royal Institute of Technology
    Author: Aguiar, A. Pedro | Faculty of Engineering, University of Porto (FEUP)
 
    keyword: Visual Servoing; Sensor-based Control; Mapping

    Abstract : Recovering the 3D structure of the surrounding environment is one of the more important tasks in any vision-controlled Structure-from-Motion (SfM) scheme. This paper focuses on the theoretical properties of the SfM known as the incremental active depth estimation. The term incremental stands for estimating the 3D structure of the scene over a chronological sequence of image frames. Active means that the camera actuation is such that it improves estimation performance. Starting from a known depth estimation filter, this paper presents the stability analysis of the filter in terms of the control inputs of the camera. By analyzing the convergence of the estimator using Lyapunov theory, we relax the constraints on the projection of the 3D point in the image plane when compared to previous results. The main results are validated through experiments with simulated data.

- VALID: A Comprehensive Virtual Aerial Image Dataset

    Author: Chen, Lyujie | Tsinghua University
    Author: Liu, Feng | Tsinghua University
    Author: Zhao, Yan | Tsinghua University
    Author: Wang, Wufan | Tsinghua University
    Author: Yuan, Xiaming | Tsinghua University
    Author: Zhu, Jihong | Tsinghua University
 
    keyword: Performance Evaluation and Benchmarking; Big Data in Robotics and Automation; Object Detection, Segmentation and Categorization

    Abstract : Aerial imagery plays an important role in land-use planning, population analysis, precision agriculture, and unmanned aerial vehicle tasks. However, existing aerial image datasets generally suffer from the problem of inaccurate labeling, single ground truth type, and few category numbers. In this work, we implement a simulator that can simultaneously acquire diverse visual ground truth data in the virtual environment. Based on that, we collect a comprehensive Virtual AeriaL Image Dataset named VALID, consisting of 6690 highresolution images, all annotated with panoptic segmentation on 30 categories, object detection with oriented bounding box, and binocular depth maps, collected in 6 different virtual scenes and 5 various ambient conditions (sunny, dusk, night, snow and fog). To our knowledge, VALID is the &#64257;rst aerial image dataset that can provide panoptic level segmentation and complete dense depth maps. We analyze the characteristics of VALID and evaluate state-of-the-art methods for multiple tasks to provide reference baselines. The experiment results demonstrate that VALID is well presented and challenging. The dataset is available at https://sites.google.com/view/valid-dataset/.

- Multiple Sound Source Position Estimation by Drone Audition Based on Data Association between Sound Source Localization and Identification

    Author: Wakabayashi, Mizuho | Kumamoto University
    Author: Okuno, Hiroshi G. | Waseda University
    Author: Kumon, Makoto | Kumamoto University
 
    keyword: Robot Audition; Aerial Systems: Perception and Autonomy; Aerial Systems: Applications

    Abstract : Drone audition, or auditory processing for drones equipped with a microphone array, is expected to compensate for problems affecting drones' visual processing, in particular occlusion and poor-illumination conditions. The current state of drone audition still assumes a single sound source. When a drone hears sounds originating from multiple sound sources, its sound-source localization function determines their directions. If two sources are very close to each other, the localization function cannot determine whether they are crossing or approaching-then-departing. This ambiguity in tracking multiple sound sources is resolved by data association. Typical methods of data association use each label of the separated sounds, but are prone to errors due to identification failures. Instead of labeling by classification, this study uses a set of classification measures determined by support vector machines (SVM) to avoid labeling failures and deal with unknown signals. The effectiveness of the proposed approach is validated through simulations and experiments conducted in the field.

- Augmented LiDAR Simulator for Autonomous Driving

    Author: Fang, Jin | Baidu
    Author: Zhou, Dingfu | BAIDU
    Author: Yan, Feilong | Baidu
    Author: Tongtong, Zhao | Baidu
    Author: Zhang, Feihu | University of Oxford
    Author: Ma, Yu | Baidu
    Author: Wang, Liang | Baidu USA
    Author: Yang, Ruigang | University of Kentucky
 
    keyword: Simulation and Animation; Computer Vision for Automation; Object Detection, Segmentation and Categorization

    Abstract : In Autonomous Driving (AD), detection and tracking of obstacles on the roads is a critical task. Deep-learning based methods using annotated LiDAR data have been the most widely adopted approach for this. Unfortunately, annotating 3D point cloud is a very challenging, time- and money-consuming task. In this paper, we propose a novel LiDAR simulator that augments real point cloud with synthetic obstacles (e.g., vehicles, pedestrians, and other movable objects). Unlike previous simulators that entirely rely on CG (Computer Graphics) models and game engines, our augmented simulator bypasses the requirement to create high-fidelity background CAD (Computer Aided Design) models. Instead, we can simply deploy a vehicle with a LiDAR scanner to sweep the street of interests to obtain the background point cloud, based on which annotated point cloud can be automatically generated. This unique "scan-and-simulate" capability makes our approach scalable and practical, ready for large-scale industrial applications. In this paper, we describe our simulator in detail, in particular the placement of obstacles that is critical for performance enhancement. We show that detectors with our simulated LiDAR point cloud alone can perform comparably (within two percentage points) with these trained with real data. Mixing real and simulated data can achieve over 95% accuracy.

- Purely Image-Based Pose Stabilization of Nonholonomic Mobile Robots with a Truly Uncalibrated Overhead Camera (I)

    Author: Liang, Xinwu | Shanghai Jiao Tong University
    Author: Wang, Hesheng | Shanghai Jiao Tong University
    Author: Liu, Yunhui | Chinese University of Hong Kong
    Author: Liu, Zhe | University of Cambridge
    Author: You, Bing | Fujian Fuqing Nuclear Power Co., Ltd
    Author: Jing, Zhongliang | Shanghai Jiao Tong University
    Author: Chen, Weidong | Shanghai Jiao Tong University
 
    keyword: Visual Servoing; Nonholonomic Mechanisms and Systems; Sensor-based Control

    Abstract : Though many vision-based control methods have been proposed for nonholonomic mobile robots, in their implementation, it is usually necessary to calibrate the camera intrinsic and/or extrinsic parameters using offline/online parameter estimation algorithms or online adaptation laws. To avoid the tediousness of camera calibration and to make the system performance highly robust to camera parameter uncertainties, in this paper, we propose novel image-based pose stabilization control approaches for nonholonomic mobile robots with a truly uncalibrated overhead fixed camera. In the proposed approaches, only image position information of three feature points from an overhead camera is used for controller design, while information from other sensors (such as wheel encoders) is not required. Furthermore, either offline or online camera calibration is not necessary, and no knowledge about the camera intrinsic and extrinsic parameters is needed, which also can greatly simplify the controller implementation. Simulation and experimental results are given to demonstrate the feasibility and effectiveness of the proposed purely image-based pose stabilization approaches

## Distributed Robot Systems
- Distributed Attack-Robust Submodular Maximization for Multi-Robot Planning

    Author: Zhou, Lifeng | Virginia Tech
    Author: Tzoumas, Vasileios | Massachusetts Institute of Technology
    Author: Pappas, George J. | University of Pennsylvania
    Author: Tokekar, Pratap | University of Maryland
 
    keyword: Distributed Robot Systems; Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems

    Abstract : We aim to guard swarm-robotics applications against denial-of-service (DoS) attacks that result in withdrawals of robots. We focus on applications requiring the selection of actions for each robot, among a set of available ones, e.g., which trajectory to follow. Such applications are central in large-scale robotic applications, e.g., multi-robot motion planning for target tracking. But the current attack-robust algorithms are centralized, and scale quadratically with the problem size (e.g., number of robots). In this paper, we propose a general-purpose distributed algorithm towards robust optimization at scale, with local communications only. We name it distributed robust maximization (DRM). DRM proposes a divide-and-conquer approach that distributively partitions the problem among K cliques of robots. The cliques optimize in parallel, independently of each other. That way, DRM also offers computational speed-ups up to 1/K^2 the running time of its centralized counterparts. K depends on the robot’ communication range, which is given as input to DRM. DRM also achieves a close-to-optimal performance. We demonstrate DRM's performance in Gazebo and MATLAB simulations, in scenarios of active target tracking with multiple robots. We observe DRM achieves significant computational speed-ups (it is 3 to 4 orders faster) and, yet, nearly matches the tracking performance of its centralized counterparts.

- Multirobot Patrolling against Adaptive Opponents with Limited Information

    Author: Diaz Alvarenga, Carlos | University of California at Merced
    Author: Basilico, Nicola | University of Milan
    Author: Carpin, Stefano | University of California, Merced
 
    keyword: Surveillance Systems; Multi-Robot Systems; Planning, Scheduling and Coordination

    Abstract : We study a patrolling problem where multiple agents are tasked with protecting an environment where one or more adversaries are trying to compromise targets of varying value. The objective of the patrollers is to move between targets to quickly spot when an attack is taking place and then diffuse it. Differently from most related literature, we do not assume that attackers have full knowledge of the strategies followed by the patrollers, but rather build a model at run time through repeated observations of how often they visit certain targets. We study three different solutions to this problem. The first two partition the environment using either a fast heuristic or an exact method that is significantly more time consuming. The third method, instead does not partition the environment, but rather lets every patroller roam over the entire environment. After having identified strengths and weaknesses of each method, we contrast their performances against attackers using different algorithms to decide whether to attack or not.

- Distributed Optimization of Nonlinear, Non-Gaussian, Communication-Aware Information Using Particle Methods

    Author: Moon, Sangwoo | University of Colorado Boulder
    Author: Frew, Eric W. | University of Colorado
 
    keyword: Distributed Robot Systems; Networked Robots; Multi-Robot Systems

    Abstract : This paper presents a distributed optimization framework and its local utility design for communication-aware information gathering by mobile robotic sensor networks. The main idea of the optimization is that each robot decides based on its local utility that considers the decisions of other neighbor robots higher in a given hierarchy. The local utility is designed as conditional mutual information that captures sensing and communication properties. Sampling procedures using a specific measurement set and particle methods are applied to compute the designed utility, which allows nonlinear, non-Gaussian properties of targets, sensing, and communication. Simulation results describe the presented distributed optimization shows more improved estimates and entropy reduction than another approach that does not consider communication properties. Simulation results also verify the presented distributed optimization using the described approach for information computation has better results than using other approaches that simplify the communication-aware information.

- Experimental Comparison of Decentralized Task Allocation Algorithms under Imperfect Communication

    Author: Nayak, Sharan | University of Maryland, College Park
    Author: Yeotikar, Suyash | University of Maryland, College Park
    Author: Carrillo, Estefany | Department of Aerospace Engineering, University of Maryland
    Author: Rudnick-Cohen, Eliot | University of Maryland, College Park
    Author: M Jaffar, Mohamed Khalid | University of Maryland, College Park
    Author: Patel, Ruchir | University of Maryland, College Park
    Author: Azarm, Shapour | University of Maryland
    Author: Herrmann, Jeffrey | University of Maryland
    Author: Xu, Huan | University of Maryland
    Author: Otte, Michael W. | University of Maryland
 
    keyword: Distributed Robot Systems; Task Planning; Networked Robots

    Abstract : We compare the performance of five state of the art decentralized task allocation algorithms under imperfect communication conditions. The decentralized algorithms considered are CBAA, ACBBA, DHBA, HIPC and PI. All algorithms are evaluated using three different models of communication, including the Bernoulli model, the Gilbert-Elliot model, and the Rayleigh Fading model. All 15 of the resulting combinations of an algorithm with a communication model are evaluated in two different problem scenarios: (1)Collaborative visit scenario where the agents have to collaboratively visit known stationary targets. (2)Collaborative search and visit scenario where the agents have to collaboratively search and visit unknown stationary target locations. We use two performance measures to evaluate algorithms: (1)max distance traveled by any agent (2)max number of messages sent by any agent. Real-time experimental simulations show the trade offs that exist between these five algorithms at different communication conditions.

-  Parallel Self-Assembly with SMORES-EP, a Modular Robot

    Author: Liu, Chao | University of Pennsylvania
    Author: Lin, Qian | Tsinghua University
    Author: Kim, Hyun | University of Pennsylvania
    Author: Yim, Mark | University of Pennsylvania


###　Scalable Cooperative Transport of Cable-Suspended Loads with UAVs Using Distributed Trajectory Optimization

    Author: Jackson, Brian | Stanford University
    Author: Howell, Taylor | Stanford University
    Author: Shah, Kunal | Stanford University
    Author: Schwager, Mac | Stanford University
    Author: Manchester, Zachary | Stanford University


## Range Sensing
- Super-Pixel Sampler: A Data-Driven Approach for Depth Sampling and Reconstruction

    Author: Wolff, Adam | Technion
    Author: Praisler, Shachar | Technion
    Author: Tcenov, Ilya | Technion
    Author: Gilboa, Guy | Technion
 
    keyword: Range Sensing; Sensor Fusion

    Abstract : Depth acquisition, based on active illumination, is essential for autonomous and robotic navigation. LiDARs (Light Detection And Ranging) with mechanical, fixed, sampling templates are commonly used in today's autonomous vehicles. An emerging technology, based on solid-state depth sensors, with no mechanical parts, allows fast and adaptive scans. <p>In this paper, we propose an adaptive, image-driven, fast, sampling and reconstruction strategy. First, we formulate a piece-wise planar depth model and estimate its validity for indoor and outdoor scenes. Our model and experiments predict that, in the optimal case, adaptive sampling strategies with about 20-60 piece-wise planar structures can approximate well a depth map. This translates to requiring a single depth sample for every 1200 RGB samples (less than 0.1%), providing strong motivation to investigate an adaptive framework. Second, we introduce SPS (Super-Pixel Sampler), a simple, generic, sampling and reconstruction algorithm, based on super-pixels. Our sampling improves grid and random sampling, consistently, for a wide variety of reconstruction methods. Third, we propose an extremely simple and fast reconstruction for our sampler. It achieves state-of-the-art results, compared to complex image-guided depth completion algorithms, reducing the required sampling rate by a factor of 3-4. A single-pixel prototype sampler built in our lab illustrates the concept.

- Physics-Based Simulation of Continuous-Wave LIDAR for Localization, Calibration and Tracking

    Author: Heiden, Eric | University of Southern California
    Author: Liu, Ziang | University of Southern California
    Author: Ramachandran, Ragesh Kumar | University Southern California
    Author: Sukhatme, Gaurav | University of Southern California
 
    keyword: Range Sensing; Simulation and Animation; Learning and Adaptive Systems

    Abstract : Light Detection and Ranging (LIDAR) sensors play an important role in the perception stack of autonomous robots, supplying mapping and localization pipelines with depth measurements of the environment. While their accuracy outperforms other types of depth sensors, such as stereo or time-of-flight cameras, the accurate modeling of LIDAR sensors requires laborious manual calibration that typically does not take into account the interaction of laser light with different surface types, incidence angles and other phenomena that significantly influence measurements. In this work, we introduce a physically plausible model of a 2D continuous-wave LIDAR that accounts for the surface-light interactions and simulates the measurement process in the Hokuyo URG-04LX LIDAR. Through automatic differentiation, we employ gradient-based optimization to estimate model parameters from real sensor measurements.

- A Spatial-Temporal Multiplexing Method for Dense 3D Surface Reconstruction of Moving Objects

    Author: Sui, Congying | The Chinese University of Hong Kong
    Author: He, Kejing | Chinese University of Hong Kong
    Author: Wang, Zerui | The Chinese University of Hong Kong
    Author: Lyu, Congyi | Beijing Institute of Technology
    Author: Guo, Huiwen | Smarteye Technique
    Author: Liu, Yunhui | Chinese University of Hong Kong
 
    keyword: Range Sensing; Computer Vision for Other Robotic Applications; Computer Vision for Automation

    Abstract : Three-dimensional reconstruction of dynamic objects is important for robotic applications, for example, the robotic recognition and manipulation. In this paper, we present a novel 3D surface reconstruction method for moving objects. The proposed method combines the spatial-multiplexing and time-multiplexing structured-light techniques that have advantages of less image acquisition time and accurate 3D reconstruction, respectively. A set of spatial-temporal encoded patterns are designed, where a spatial-encoded texture map is embedded into the temporal-encoded three-step phase-shifting fringes. The specifically designed spatial-coded texture assigns high-uniqueness codeword to any window on the image which helps to eliminate the phase ambiguity. In addition, the texture is robust to noise and image blur. Combining this texture with high-frequency phase-shifting fringes, high reconstruction accuracy would be ensured. This method only requires 3 patterns to uniquely encode a surface, which facilitates the fast image acquisition for each reconstruction step. A filtering stereo matching algorithm is proposed for the spatial-temporal multiplexing method to improve the matching reliability. Moreover, the reconstruction precision is further enhanced by a correspondence refinement algorithm. Experiments validate the performance of the proposed method including the high accuracy, the robustness to noise and the ability to reconstruct moving objects.

- Modeling of Architectural Components for Large-Scale Indoor Spaces from Point Cloud Measurement

    Author: Lim, Gahyeon | Korea University
    Author: Oh, Youjin | Korea University
    Author: Kim, Dongwoo | Korea University
    Author: Jun, ChangHyun | Korea University
    Author: Kang, Jaehyeon | Korea Institute of Industrial Technology
    Author: Doh, Nakju | Korea University
 
    keyword: Range Sensing; Object Detection, Segmentation and Categorization; Robotics in Construction

    Abstract : In this paper, we propose a method to model architectural components in large-scale indoor spaces from point cloud measurements. The proposed method enables the modeling of curved surfaces, cylindrical pillars, and slanted surfaces, which cannot be modeled using existing approaches. It operates by constructing the architectural points from the raw point cloud after removing non-architectural (objects) points and filling in the holes caused by their exclusion. Then, the architectural points are represented using a set of piece-wise planar segments. Finally, the adjacency graph of the planar segments is constructed to verify the fact that every planar segment is closed. This ensures a watertight mesh model generation. Experimentation using 14 different real-world indoor space datasets and 2 public datasets, comprising spaces of various sizes---from room-scale to large-scale (12,557m^2), verify the accuracy of the proposed method in modeling environments with curved surfaces, cylindrical pillars, and slanted surfaces.

- PhaRaO: Direct Radar Odometry Using Phase Correlation

    Author: Park, Yeong Sang | KAIST
    Author: Shin, Young-Sik | KAIST
    Author: Kim, Ayoung | Korea Advanced Institute of Science Technology
 
    keyword: Range Sensing; SLAM; Autonomous Vehicle Navigation

    Abstract : Recent studies in radar-based navigation present promising navigation performance using scanning radars. These scanning radar-based odometry methods are mostly feature-based; they detect and match salient features within a radar image. Differing from existing feature-based methods, this paper reports on a method using direct radar odometry, PhaRaO, which infers relative motion from a pair of radar scans via phase correlation. Specifically, we apply the Fourier Mellin transform (FMT) for Cartesian and log-polar radar images to sequentially estimate rotation and translation. In doing so, we decouple rotation and translation estimations in a coarse-to-fine manner to achieve real-time performance. The proposed method is evaluated using large-scale radar data obtained from various environments. The inferred trajectory yields a 2.34% (translation) and 2.93 (rotation) Relative Error (RE) over a 4 km path length on average for the odometry estimation.

- DeepTemporalSeg: Temporally Consistent Semantic Segmentation of 3D LiDAR Scans

    Author: Dewan, Ayush | University of Freibug
    Author: Burgard, Wolfram | Toyota Research Institute
 
    keyword: Range Sensing; Semantic Scene Understanding; Deep Learning in Robotics and Automation

    Abstract : Understanding the semantic characteristics of the environment is a key enabler for autonomous robot operation. In this paper, we propose a deep convolutional neural network (DCNN) for semantic segmentation of a LiDAR scan into the classes car, pedestrian and bicyclist. This architecture is based on dense blocks and efficiently utilizes depth separable convolutions to limit the number of parameters while still maintaining the state-of-the-art performance. To make the predictions from the DCNN temporally consistent, we propose a Bayes filter based method. This method uses the predictions from the neural network to recursively estimate the current semantic state of a point in a scan. This recursive estimation uses the knowledge gained from previous scans, thereby making the predictions temporally consistent and robust towards isolated erroneous predictions. We compare the performance of our proposed architecture with other state-of-the-art neural network architectures and report substantial improvement. For the proposed Bayes filter approach, we shows results on various sequences in the KITTI tracking benchmark.

## Transfer Learning
- Self-Supervised Sim-To-Real Adaptation for Visual Robotic Manipulation

    Author: Jeong, Rae | DeepMind
    Author: Aytar, Yusuf | Massachusetts Institute of Technology
    Author: Khosid, David | DeepMind
    Author: Zhou, Yuxiang | Google Uk, Ltd
    Author: Kay, Jackie | DeepMind
    Author: Lampe, Thomas | Google UK Ltd
    Author: Bousmalis, Konstantinos | Google
    Author: Nori, Francesco | DeepMind
 
    keyword: Deep Learning in Robotics and Automation; Simulation and Animation; AI-Based Methods

    Abstract : Collecting and automatically obtaining reward signals from real robotic visual data for the purposes of training reinforcement learning algorithms can be quite challenging and time-consuming. Methods for utilizing unlabeled data can have a huge potential to further accelerate robotic learning. We consider here the problem of performing manipulation tasks from pixels. In such tasks, choosing an appropriate state representation is crucial for planning and control. This is even more relevant with real images where noise, occlusions and resolution affect the accuracy and reliability of state estimation. In this work, we learn a latent state representation implicitly with deep reinforcement learning in simulation, and then adapt it to the real domain using unlabeled real robot data. We propose to do so by optimizing sequence-based self-supervised objectives. These use the temporal nature of robot experience, and can be common in both the simulated and real domains, without assuming any alignment of underlying states in simulated and unlabeled real images. We further propose a novel such objective, the textit{Contrastive Forward Dynamics} loss, which combines dynamics model learning with time-contrastive techniques. The learned state representation that results from our methods can be used to robustly solve a manipulation task in simulation and to successfully transfer the learned skill on a real system.

- Meta Reinforcement Learning for Sim-To-Real Domain Adaptation

    Author: Arndt, Karol | Aalto University
    Author: Hazara, Murtaza | KU Leuven
    Author: Ghadirzadeh, Ali | KTH Royal Institute of Technology, Aalto University
    Author: Kyrki, Ville | Aalto University
 
    keyword: Deep Learning in Robotics and Automation; Learning and Adaptive Systems

    Abstract : Modern reinforcement learning methods suffer from low sample efficiency and unsafe exploration, making it infeasible to train robotic policies entirely on real hardware. In this work, we propose to address the problem of sim-to-real domain transfer by using meta learning to train a policy that can adapt to a variety of dynamic conditions, and using a task-specific trajectory generation model to provide an action space that facilitates quick exploration. We evaluate the method by performing domain adaptation in simulation and analyzing the structure of the latent space during adaptation. We then deploy this policy on a KUKA LBR 4+ robot and evaluate its performance on a task of hitting a hockey puck to a target. Our method shows more consistent and stable domain adaptation than the baseline, resulting in better overall performance.

- Variational Auto-Regularized Alignment for Sim-To-Real Control

    Author: Hwasser, Martin | KTH / Northvolt
    Author: Kragic, Danica | KTH
    Author: Antonova, Rika | KTH Stockholm
 
    keyword: Learning and Adaptive Systems; Deep Learning in Robotics and Automation; Model Learning for Control

    Abstract : General-purpose simulators can be a valuable data source for flexible learning and control approaches. However, training models or control policies in simulation and then directly applying to hardware can yield brittle control. Instead, we propose a novel way to use simulators as regularizers. Our approach regularizes a decoder of a variational autoencoder to a black-box simulation, with the latent space bound to a subset of simulator parameters. This enables successful encoder training from a small number of real-world trajectories (10 in our experiments), yielding a latent space with simulation parameter distribution that matches the real-world setting. We use a learnable mixture for the latent prior/posterior, which implies a highly flexible class of densities for the posterior fit. Our approach is scalable and does not require restrictive distributional assumptions. We demonstrate ability to recover matching parameter distributions on a range of benchmarks, challenging custom simulation environments and several real-world scenarios. Our experiments using ABB YuMi robot hardware show ability to help reinforcement learning approaches overcome cases of severe sim-to-real mismatch.

- Experience Selection Using Dynamics Similarity for Efficient Multi-Source Transfer Learning between Robots

    Author: Sorocky, Michael | University of Toronto
    Author: Zhou, Siqi | University of Toronto
    Author: Schoellig, Angela P. | University of Toronto
 
    keyword: Learning and Adaptive Systems; Model Learning for Control

    Abstract : In the robotics literature, different knowledge transfer approaches have been proposed to leverage the experience from a source task or robot---real or virtual---to accelerate the learning process on a new task or robot. A commonly made but infrequently examined assumption is that incorporating experience from a source task or robot will be beneficial. For practical applications, inappropriate knowledge transfer can result in negative transfer or unsafe behaviour. In this work, inspired by a system gap metric from robust control theory, the nu-gap, we present a data-efficient algorithm for estimating the similarity between pairs of robot systems. In a multi-source inter-robot transfer learning setup, we show that this similarity metric allows us to predict relative transfer performance and thus informatively select experiences from a source robot before knowledge transfer. We demonstrate our approach with quadrotor experiments, where we transfer an inverse dynamics model from a real or virtual source quadrotor to enhance the tracking performance of a target quadrotor on arbitrary hand-drawn trajectories. We show that selecting experiences based on the proposed similarity metric effectively facilitates the learning of the target quadrotor, improving performance by 62% compared to a poorly selected experience.

- DeepRacer: Autonomous Racing Platform for Experimentation with Sim2Real Reinforcement Learning

    Author: Balaji, Bharathan | Amazon
    Author: Mallya, Sunil | Amazon Artificial Intelligence
    Author: Genc, Sahika | Amazon Artificial Intelligence
    Author: Gupta, Saurabh | Amazon
    Author: Dirac, Leo | Amazon
    Author: Khare, Vineet | Amazon
    Author: Roy, Gourav | Amazon
    Author: Sun, Tao | Amazon
    Author: Tao, Yunzhe | Amazon Web Services
    Author: Townsend, Brian | Amazon
    Author: Calleja, Eddie | Amazon
    Author: Muralidhara, Sunil | Amazon
    Author: Karuppasamy, Dhanasekar | Amazon
 
    keyword: AI-Based Methods; Autonomous Vehicle Navigation; Education Robotics

    Abstract : DeepRacer is a platform for end-to-end experimentation with RL and can be used to systematically investigate the key challenges in developing intelligent control systems. Using the platform, we demonstrate how a 1/18th scale car can learn to drive autonomously using RL with a monocular camera. It is trained in simulation with no additional tuning in physical world and demonstrates: 1) formulation and solution of a robust reinforcement learning algorithm, 2) narrowing the reality gap through joint perception and dynamics, 3) distributed on-demand compute architecture for training optimal policies, and 4) a robust evaluation method to identify when to stop training. It is the first successful large-scale deployment of deep reinforcement learning on a robotic control agent that uses only raw camera images as observations and a model-free learning method to perform robust path planning.

- Cross-Domain Motion Transfer Via Safety-Aware Shared Latent Space Modeling

    Author: Choi, Sungjoon | Disney Research
    Author: Kim, Joohyung | University of Illinois at Urbana-Champaign
 
    keyword: Deep Learning in Robotics and Automation; Motion and Path Planning; Collision Avoidance

    Abstract : This paper presents a data-driven motion retargeting method with safety considerations. In particular, we focus on handling self-collisions while transferring poses between different domains. To this end, we &#64257;rst propose leveraged Wasserstein auto-encoders (LWAE) which leverage both positive and negative data where negative data consist of self-collided poses. Then, we extend this idea to multiple domains to have a shared latent space to perform motion retargeting. We also present an effective self-collision handling method based on solving inverse kinematics with augmented targets that is used to collect collision-free poses. The proposed method is extensively evaluated in a diverse set of motions from human subjects and an animation character where we show that incorporating negative data dramatically reduces self-collisions while preserving the quality of the original motion.

## Flexible Robots
- Investigation of a Multistable Tensegrity Robot Applied As Tilting Locomotion System

    Author: Schorr, Philipp | TU Ilmenau
    Author: Schale, Florian | Ilmenau University of Technology
    Author: Otterbach, Jan Marc | TU Ilmenau
    Author: Zentner, Lena | TU Ilmenau
    Author: Zimmermann, Klaus | TU Ilmenau, Germany
    Author: Boehm, Valter | OTH Regensburg
 
    keyword: Flexible Robots; Soft Robot Applications; Dynamics

    Abstract : This paper describes the development of a tilting locomotion system based on a compliant tensegrity structure with multiple stable equilibrium configurations. A tensegrity structure featuring 4 stable equilibrium states is considered. The mechanical model of the structure is presented and the according equations of motion are derived. The variation of the length of selected structural members allows to influence the prestress state and the corresponding shape of the tensegrity structure. Based on bifurcation analyses a reliable actuation strategy to control the current equilibrium state is designed. In this work, the tensegrity structure is assumed to be in contact with a horizontal plane due to gravity. The derived actuation strategy is utilized to generate tilting locomotion by successively changing the equilibrium state. Numerical simulations are evaluated considering the locomotion characteristics. In order to validate this theoretical approach a prototype is developed. Experiments regarding to the equilibrium configurations, the actuation strategy and the locomotion characteristics are evaluated using image processing tools and motion capturing. The results verify the theoretical data and confirm the working principle of the investigated tilting locomotion system. This approach represents a feasible actuation strategy to realize a reliable tilting locomotion utilizing the multistability of compliant tensegrity structures.

- A Novel Articulated Soft Robot Capable of Variable Stiffness through Bistable Structure

    Author: Zhong, Yong | South China University of Technology
    Author: Du, Ruxu | The Chinese University of Hong Kong
    Author: Wu, Liao | University of New South Wales
    Author: Yu, Haoyong | National University of Singapore
 
    keyword: Flexible Robots; Kinematics; Mechanism Design

    Abstract : Soft robot has demonstrated promise in unstructured and dynamic environments due to unique advantages, such as safe interaction, adaptiveness, easy to actuate, and easy fabrication. However, the highly dissipative nature of elastic materials results in small stiffness of soft robot which limits certain functions, such as force transmission, position accuracy, and load capability. In this paper, we present a novel articulated soft robot with variable stiffness. The robot is constructed by rigid joints and compliant bistable structures in series. Each joint can be independently locked through triggering the bistable structure to touch the mechanical constrain. Thus, the bending stiffness of the joint can be magnified which increases the stiffness of the articulated soft robot. Through this construction method, even driven by only one servomotor, the robot demonstrates variable workspace and stiffness which have the potential of dexterous manipulation and maintaining shape under tip load.

- Modeling and Experiments on the Swallowing and Disgorging Characteristics of an Underwater Continuum Manipulator

    Author: Wang, Haihang | Harbin Engineering University
    Author: Xu, He | College of Mechanical and Electrical Engineering, Harbin Enginee
    Author: Yu, Fengshu | Harbin Engineering University
    Author: Li, Xin | Harbin Engineering University
    Author: Yang, Chen | Harbin Engineering University
    Author: Chen, Siqing | Harbin Engineering University
    Author: Chen, JunLong | Harbin Engineering University
    Author: Zhang, Yonghui | Harbin Engineering University
    Author: Zhou, Xueshan | Harbin Engineering University
 
    keyword: Flexible Robots; Hydraulic/Pneumatic Actuators; Soft Robot Applications

    Abstract : Soft robots apply compliant materials to perform motions and behaviors not typically achievable by rigid robots. An underwater, compliant, multi-segment continuum manipulator that can bend, swallow, disgorge is developed in this study. The manipulator is driven by McKibben water hydraulic artificial muscle (WHAM). The mechanical properties of the WHAM are tested and analyzed experimentally. The kinematics model, which concerns about the variable diameter structure of the soft grippers, are established to simulate the behaviors of the manipulator among the bending, swallowing and disgorging procedure. A mouth-tongue collaborative soft robot assembled with another single-segment soft robot arm is presented. And its functions are experimentally testified. The distinctive functions were verified according to the experimental results.

- Salamanderbot: A Soft-Rigid Composite Continuum Mobile Robotto Traverse Complex Environments

    Author: Sun, Yinan | Worcester Polytechnic Institute
    Author: Jiang, Yuqi | Worcester Polytechnic Institute
    Author: Yang, Hao | Worcester Polytechnic Institute
    Author: Walter, Louis-Claude | Ecole Nationale Supérieure d'Electricité Et De Mécanique
    Author: Santoso, Junius | WPI
    Author: Skorina, Erik | Worcester Polytechnic Institute
    Author: Onal, Cagdas | WPI
 
    keyword: Flexible Robots; Search and Rescue Robots; Soft Robot Applications

    Abstract : Soft robots are theoretically well-suited to rescueand exploration	applications where their flexibility allows forthe traversal	of highly cluttered environments. However, mostexisting mobile soft robots are not	fast or powerful enoughto effectively traverse three dimensional environments. In thispaper,	we introduce a new mobile robot with a continuouslydeformable slender body structure, the SalamanderBot, whichcombines the flexibility and maneuverability of soft robots, withthe speed	and power of traditional mobile robots. It consistsof a cable-driven bellows-like origami modules based on theYoshimura crease pattern mounted	between sets of poweredwheels.	The origami structure allows the body to deform asnecessary to adapt to complex environments and terrains, whilethe wheels allow the robot to reach speeds of up to 303.1 mm/s(2.05 body-length/s). Salamanderbot can climb up to 60-degreeslopes and perform sharp turns with a minimum turning radiusof 79.9 mm (0.54 body-length).

- Flexure Hinge-Based Biomimetic Thumb with a Rolling-Surface Metacarpal Joint

    Author: Pulleyking, Spenser | University of Tulsa
    Author: Schultz, Joshua | University of Tulsa
 
    keyword: Flexible Robots; Underactuated Robots; Tendon/Wire Mechanism

    Abstract : The human thumb's state contribution to grasping and dexterous manipulation of objects is a function of the kinematic multiplicity of joints and structure of the bones, joints, and ligaments. This paper looks at the design and evaluation of a human-like thumb for use in a robotic hand, where the thumb's state contribution to grasping and dexterous manipulation is a function of a simplified kinematic model based on that of the human thumb, but also on empirical trials of surgical techniques to retain functionality while reducing the number of joints in the thumb. Motion Capture Data of the End Effector is analyzed with the measured excursion of the tendons to determine the relationship between tendon velocities and task-space velocities. We propose a simplified metric to represent this data, after validating the procedure experimentally, and show that our prototype is predicted to have a relatively smooth mapping between tendon excursion velocity and end effector velocity.

- Stretchable Kirigami Components for Composite Meso-Scale Robots

    Author: Firouzeh, Amir | Seoul National University
    Author: Higashisaka, Tatsuya | The University of Tokyo
    Author: Nagato, Keisuke | The University of Tokyo
    Author: Cho, Kyu-Jin | Seoul National University, Biorobotics Laboratory
    Author: Paik, Jamie | Ecole Polytechnique Federale De Lausanne
 
    keyword: Flexible Robots; Soft Robot Materials and Design; Soft Sensors and Actuators

    Abstract : Layer-by-layer manufacturing of composite mechanisms allows fast and cost-effective fabrication of customized robots in millimeter and centimeter scales which is promising for research fields that rely on frequent and numerous physical iterations. Due to the limited number of components that can be directly integrated in composite structures, however, often an assembly step is necessary which diminishes the benefits of this manufacturing method. Inspired by the Japanese craft of cutting (kiri-) paper (-gami), Kirigami, we introduce quasi-2D and highly stretchable functional Kirigami layers for direct integration into the composite mechanisms. Depending on the material and geometrical design; functional Kirigami layers can perform as flat springs, stretchable electronics, sensors or actuators. These components will facilitate the design and manufacturing of composite robots for different applications. To illustrate the effectiveness of these components, we designed and realized a foldable composite inchworm robot with three Kirigami layers serving as actuator, sensor and contact pad with directional friction. We elaborate on the working principle of each layer and report on their combined performance in the robot.

## Field and Space Robots
- Ibex: A Reconfigurable Ground Vehicle with Adaptive Terrain Navigation Capability

    Author: Raj, Senthur | National Institute of Technology, Tiruchirappalli
    Author: Aatitya R P, Manu | National Institute of Technology, Tiruchirapalli
    Author: Samuel, Jack | NATIONAL INSTITUTE of TECHNOLOGY, Tiruchirapalli
    Author: Karthik, Veejay | National Institute of Technology Tiruchirappalli
    Author: D, Ezhilarasi | National Institute of Technology Tiruchirappalli
 
    keyword: Field Robots; Wheeled Robots; Compliance and Impedance Control

    Abstract : This paper presents a unique unmanned ground vehicle with a dynamic wheelbase and an adaptive thrust based friction optimization scheme that aids in the traversal of steep slopes and slippery surfaces. The vehicle is capable of adapting itself to the surface topography using an impedance-based stabilization module to minimize the mechanical oscillatory transients induced during its motion. A detailed analysis of its modules has been elucidated in this paper based on the vehicle parameters. The proposed methodologies have been integrated and tested on a customized prototype. Experimental validation and simulation for the proposed modules at various terrain conditions have been carried out to authenticate its performance.

- Day and Night Collaborative Dynamic Mapping in Unstructured Environment Based on Multimodal Sensors

    Author: Yue, Yufeng | Nanyang Technological University
    Author: Yang, Chule | Nanyang Technological University
    Author: Zhang, Jun | Nanyang Technological University
    Author: Wen, Mingxing | Nanyang Technological University
    Author: Wu, Zhenyu | Nanyang Technological University
    Author: Zhang, Haoyuan | Nanyang Technological University
    Author: Wang, Danwei | Nanyang Technological University
 
    keyword: Field Robots; Mapping; Sensor Fusion

    Abstract : Enabling long-term operation during day and night for collaborative robots requires a comprehensive understanding of the unstructured environment. Besides, in the dynamic environment, robots must be able to recognize dynamic objects and collaboratively build a global map. This paper proposes a novel approach for dynamic collaborative mapping based on multimodal environmental perception. For each mission, robots first apply heterogeneous sensor fusion model to detect humans and separate them to acquire static observations. Then, the collaborative mapping is performed to estimate the relative position between robots and local 3D maps are integrated into a globally consistent 3D map. The experiment is conducted in the day and night rainforest with moving people. The results show the accuracy, robustness, and versatility in 3D map fusion missions.

- Generating Locomotion with Effective Wheel Radius Manipulation

    Author: Hojnik, Tim | CSIRO
    Author: Pond, Lachlan | QUT
    Author: Dungavell, Ross | CSIRO
    Author: Flick, Paul | CSIRO
    Author: Roberts, Jonathan | Queensland University of Technology
 
    keyword: Field Robots; Space Robotics and Automation; Intelligent Transportation Systems

    Abstract : Travel over sloped terrain is difficult as an incline changes the interaction between each wheel and the ground resulting in an unbalanced load distribution which can lead to loss of traction and instability. This paper presents a novel approach to generating wheel rotation for primary locomotion by only changing its centre of rotation, or as a complimentary locomotion source to increase versatility of a plain centre hub drive. This is done using linear actuators within a wheel to control the position of the centre hub and induce a moment on the wheel from gravity. In doing so our platform allows for active ride height selection and individual wheel pose control. We present the system with calculations outlining the theoretical properties and perform experiments to validate the concept under loading via multiple gaits to show motion on slopes, and sustained motion over extended distance. We envision applications in conjunction to assist current motor drives and increasing slope traversability by allowing body pose and centre of gravity manipulation, or as a primary locomotion system.

- Where to Map? Iterative Mars Helicopter-Rover Path Planning for Long-Range Autonomous Exploration

    Author: Sasaki, Takahiro | Japan Aerospace Exploration Agency
    Author: Otsu, Kyohei | California Institute of Technology
    Author: Thakker, Rohan | Nasa's Jet Propulsion Laboratory, Caltech
    Author: Haesaert, Sofie | Eindhoven University of Technology
    Author: Agha-mohammadi, Ali-akbar | NASA-JPL, Caltech
 
    keyword: Space Robotics and Automation; Path Planning for Multiple Mobile Robots or Agents

    Abstract : Besides the conventional ground-crawling vehicles, the Mars 2020 mission decided to send a helicopter to Mars. The copter's high-resolution data should help the rover to identify smaller hazards and avoid small structural components such as rocks or pebbles. We consider a three-agent system composed of a Mars rover, copter, and orbiter. The objective is to compute an optimal rover path that minimizes the localization uncertainty accumulation after a traverse. To achieve such a goal, we primarily focus on the localizability, which is the goodness measure, and conduct a joint-space search over rover's path and copter's perceptive actions before a traverse. Then, we jointly address where to map by the copter and where to drive by the rover to minimize the uncertainty accumulation in rover localization using the proposed iterative copter-rover path planner. Numerical simulations demonstrate the effectiveness of the proposed planner.

- A GNC Architecture for Planetary Rovers with Autonomous Navigation

    Author: Azkarate, Martin | European Space Agency (ESA) - ESTEC
    Author: Gerdes, Levin | ESA/ESTEC
    Author: Perez-del-Pulgar, Carlos | Universidad De Málaga
    Author: Joudrier, Luc | ESA
 
    keyword: Space Robotics and Automation; Autonomous Agents; Visual-Based Navigation

    Abstract : This paper proposes a Guidance, Navigation, and Control (GNC) architecture for planetary rovers targeting the conditions of upcoming Mars exploration missions such as Mars 2020 and the Sample Fetching Rover (SFR). The navigation requirements of these missions demand a control architecture featuring autonomous capabilities to achieve a fast and long traverse. The proposed solution presents a two-level architecture where the efficient navigation (low) level is always active and the full navigation (upper) level is enabled according to the difficulty of the terrain. The first level is an efficient implementation of the basic functionalities for autonomous navigation based on hazard detection, local path replanning, and trajectory control with visual odometry. The second level implements an adaptive SLAM algorithm that improves the relative localization, evaluates the traversability of the terrain ahead for a more optimal path planning, and performs global (absolute) localization that corrects the pose drift during longer traverses. The architecture provides a solution for long range, low supervision and fast planetary exploration. Both navigation levels have been validated on planetary analogue field test campaigns.

- Mine Tunnel Exploration Using Multiple Quadrupedal Robots

    Author: Miller, Ian | University of Pennsylvania
    Author: Cladera, Fernando | University of Pennsylvania
    Author: Cowley, Anthony | University of Pennsylvania
    Author: Skandan, Shreyas | University of Pennsylvania
    Author: Lee, Elijah S. | University of Pennsylvania
    Author: Lipschitz, Laura | UPenn
    Author: Bhat, Akhilesh | University of Pennsylvania
    Author: Rodrigues, Neil | University of Pennsylvania
    Author: Zhou, Alex | University of Pennsylvania
    Author: Cohen, Avraham | Technion, Robotics Laboratory
    Author: Kulkarni, Adarsh | University of Pennsylvania
    Author: Laney, James | Ghost Robotics
    Author: Taylor, Camillo Jose | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania, School of Engineering and Applied Sc
 
    keyword: Mining Robotics; Field Robots; Legged Robots

    Abstract : Robotic exploration of underground environments is a particularly challenging problem due to communication, endurance, and traversability constraints which necessitate high degrees of autonomy and agility. These challenges are further exacerbated by the need to minimize human intervention forpractical applications. While legged robots have the ability to traverse extremely challenging terrain, they also engender newchallenges for planning, estimation, and control. In this work, we describe a fully autonomous system for multi-robot mine exploration and mapping using legged quadrupeds, as well as a distributed database mesh networking system for reporting data. In addition, we show results from the DARPA Subterranean Challenge (SubT) Tunnel Circuit demonstrating localization of artifacts after traversals of hundreds of meters. These experiments describe fully autonomous exploration of an unknown Global Navigation Satellite System (GNSS)-denied environment undertaken by legged robots.

## Recognition
- Learning Face Recognition Unsupervisedly by Disentanglement and Self-Augmentation

    Author: Lee, Yi-Lun | National Chiao Tung University
    Author: Tseng, Min-Yuan | National Chiao Tung University
    Author: Yu, Dung Ru | National Chiao Tung University
    Author: Lo, Yu-Cheng | National Chiao-Tung University
    Author: Chiu, Wei-Chen | National Chiao Tung University
 
    keyword: Recognition

    Abstract : As the growth of smart home, healthcare, and home robot applications, learning a face recognition system which is specific for a particular environment and capable of self-adapting to the temporal changes in appearance (e.g., caused by illumination or camera position) is nowadays an important topic. In this paper, given a video of a group of people, which simulates the surveillance video in a smart home environment, we propose a novel approach which unsupervisedly learns a face recognition model based on two main components: (1) a triplet network that extracts identity-aware feature from face images for performing face recognition by clustering, and (2) an augmentation network that is conditioned on the identity-aware features and aims at synthesizing more face samples. Particularly, the training data for the triplet network is obtained by using the spatiotemporal characteristic of face samples within a video, while the augmentation network learns to disentangle a face image into identity-aware and identity-irrelevant features thus is able to generate new faces of the same identity but with variance in appearance. With taking the richer training data produced by augmentation network, the triplet network is further fine-tuned and achieves better performance in face recognition. Extensive experiments not only show the efficacy of our model in learning an environment-specific face recognition model unsupervisedly, but also verify its adaptability to various appearance changes.

- PARC: A Plan and Activity Recognition Component for Assistive Robots

    Author: Massardi, Jean | Université Du Québec à Montréal
    Author: Gravel, Mathieu | University of Quebec at Montreal
    Author: Beaudry, Eric | Université Du Québec à Montréal
 
    keyword: RGB-D Perception; Human Detection and Tracking; Recognition

    Abstract : Mobile robot assistants have many applications, such as providing help for people in their daily living activities. These robots have to detect and to recognize the actions and goals of the assisted humans. While several plan and activity recognition solutions are widely spread for controlled environments with many built-in sensors, like smart-homes, there is a lack of such systems for mobile robots operating in open settings, such as an apartment. We propose a module for the recognition of the activities and goals for daily living on mobile robots, in real time and for complex activities. Our approach recognizes human-object interaction using an RGB-D camera to infer low-level actions which are sent to a goal recognition algorithm. Results show that our approach is both real time and requires little computational resources, which facilitates its deployment on a mobile and low-cost robotics platform.

- Image-Based Place Recognition on Bucolic Environment across Seasons from Semantic Edge Description

    Author: Benbihi, Assia | Umi 2958 Gt-Cnrs
    Author: Aravecchia, Stephanie | Georgia Tech Lorraine - UMI 2958 GT-CNRS
    Author: Geist, Matthieu | Université De Lorraine
    Author: Pradalier, Cedric | GeorgiaTech Lorraine
 
    keyword: Recognition; Semantic Scene Understanding; Localization

    Abstract : Most of the research effort on image-based place recognition is designed for urban environments. In bucolic environments such as natural scenes with low texture and little semantic content, the main challenge is to handle the variations in visual appearance across time such as illumination, weather, vegetation state or viewpoints. The nature of the variations is different and this leads to a different approach to describing a bucolic scene. We introduce a global image descriptor computed from its semantic and topological information. It is built from the wavelet transforms of the image semantic edges. Matching two images is then equivalent to matching their semantic edge descriptors. We show that this method reaches state-of-the-art image retrieval performance on two multi-season environment-monitoring datasets: the CMU-Seasons and the Symphony Lake dataset. It also generalises to urban scenes on which it is on par with the current baselines NetVLAD and DELF.

- A Multilayer-Multimodal Fusion Architecture for Pattern Recognition of Natural Manipulations in Percutaneous Coronary Interventions

    Author: Zhou, Xiao-Hu | Institute of Automation Chinese Academy of Sciences
    Author: Xie, Xiaoliang | Institutation of Automation, Chinese Academy of Sciences
    Author: Feng, Zhen-Qiu | Institute of Automation, Chinese Academy of Sciences
    Author: Hou, Zeng-Guang | Chinese Academy of Science
    Author: Bian, Gui-Bin | Institute of Automation, Chinese Academy of Sciences
    Author: Li, Rui-Qi | Institute of Automation, Chinese Academy of Sciences
    Author: Ni, ZhenLiang | Chinese Academy of Sciences
    Author: Liu, Shiqi | School of Automation, Harbin University of Science and Technolog
    Author: Zhou, Yan-Jie | Institute of Automation, Chinese Academy of Sciences
 
    keyword: Recognition; Sensor Fusion; Surgical Robotics: Steerable Catheters/Needles

    Abstract : The increasingly-used robotic systems can provide precise delivery and reduce X-ray radiation to medical staff in percutaneous coronary interventions (PCI), but natural manipulations of interventionalists are forgone in most robot-assisted procedures. Therefore, it is necessary to explore natural manipulations to design more advanced human-robot interfaces (HRI). In this study, a multilayer-multimodal fusion architecture is proposed to recognize six typical subpatterns of guidewire manipulations in conventional PCI. The synchronously acquired multimodal behaviors from ten subjects are used as the inputs of the fusion architecture. Six classification-based and two rule-based fusion algorithms are evaluated for performance comparisons. Experimental results indicate that the multimodal fusion brings significant accuracy improvement in comparison with single-modal schemes. Furthermore, the proposed architecture can achieve the overall accuracy of 96.90%, much higher than that of a single-layer recognition architecture (92.56%). These results have indicated the potential of the proposed method for facilitating the development of HRI for robot-assisted PCI.

- Action Description from 2D Human Postures in Care Facilities

    Author: Takano, Wataru | Osaka University
    Author: Haeyeon, Lee | Toyota Motor Corporation
 
    keyword: Recognition

    Abstract : This paper describes a novel approach to classification of whole-body motions from estimated human postures in 2D camera images and subsequent generation of their relevant descriptions. The motions are encoded into stochastic motion models referred to as motion primitives. Words are connected to the motion primitives, and word n-grams are represented stochastically. More specifically, the motion observation is classified into the motion primitive, from which its relevant words are generated. These words are arranged in a grammatically correct order to make the descriptions for the observation. This approach was tested on actions performed by older adults in the care facility and its validity was demonstrated.

- CoHOG: A Light-Weight, Compute-Efficient and Training-Free Visual Place Recognition Technique for Changing Environments

    Author: Zaffar, Mubariz | University of Essex
    Author: Ehsan, Shoaib | University of Essex
    Author: Milford, Michael J | Queensland University of Technology
    Author: McDonald-Maier, Klaus | University of Essex
 
    keyword: Recognition; SLAM; Computer Vision for Automation

    Abstract : This paper presents a novel, compute-efficient and training-free approach based on Histogram-of-Oriented-Gradients (HOG) descriptor for achieving state-of-the-art performance-per-compute-unit in Visual Place Recognition (VPR). The inspiration for this approach (namely CoHOG) is based on the convolutional scanning and regions-based feature extraction employed by Convolutional Neural Networks (CNNs). By using image entropy to extract regions-of-interest (ROI) and regional-convolutional descriptor matching, our technique performs successful place recognition in changing environments. We use viewpoint- and appearance-variant public VPR datasets to report this matching performance, at lower RAM commitment, zero training requirements and 20 times lesser feature encoding time compared to state-of-the-art neural networks. We also discuss the image retrieval time of CoHOG and the effect of CoHOG's parametric variation on its place matching performance and encoding time.

## Aerial Systems: Multi-Robots
- Distributed Consensus Control of Multiple UAVs in a Constrained Environment

    Author: Wang, Gang | University of Nevada
    Author: Yang, Weixin | University of Nevada, Reno
    Author: Zhao, Na | University of Nevada, Reno
    Author: Ji, Yunfeng | University of Shanghai for Science and Technology
    Author: Shen, Yantao | University of Nevada, Reno
    Author: Xu, Hao | University of Nevada, Reno
    Author: Li, Peng | Harbin Institute of Technology (ShenZhen)
 
    keyword: Autonomous Agents; Distributed Robot Systems

    Abstract : In this paper, we investigate the consensus problem of multiple unmanned aerial vehicles (UAVs) in the presence of environmental constraints under a general communication topology containing a directed spanning tree. First, based on a position transformation function, we propose a novel dynamic reference position and yaw angle for each UAV to cope with both the asymmetric topology and the constraints. Then, the backstepping-like design methodology is presented to derive a local tracking controller for each UAV such that its position and yaw angle can converge to the reference ones. The proposed protocol is distributed in the sense that, the input update of each UAV dynamically relies only on local state information from its neighborhood set and the constraints, and it does not require any additional centralized information. It is demonstrated that under the proposed protocol, all UAVs reach consensus without violation of the environmental constraints. Finally, simulation and experimental results are provided to demonstrate the performance of the protocol.

- Neural-Swarm: Decentralized Close-Proximity Multirotor Control Using Learned Interactions

    Author: Shi, Guanya | California Institute of Technology
    Author: Hoenig, Wolfgang | California Institute of Technology
    Author: Yue, Yisong | California Institute of Technology
    Author: Chung, Soon-Jo | Caltech
 
    keyword: Aerial Systems: Mechanics and Control; Deep Learning in Robotics and Automation; Multi-Robot Systems

    Abstract : In this paper, we present Neural-Swarm, a nonlinear decentralized stable controller for close-proximity flight of multirotor swarms. Close-proximity control is challenging due to the complex aerodynamic interaction effects between multirotors, such as downwash from higher vehicles to lower ones. Conventional methods often fail to properly capture these interaction effects, resulting in controllers that must maintain large safety distances between vehicles, and thus are not capable of close-proximity flight. Our approach combines a nominal dynamics model with a regularized permutation-invariant Deep Neural Network (DNN) that accurately learns the high-order multi-vehicle interactions. We design a stable nonlinear tracking controller using the learned model. Experimental results demonstrate that the proposed controller significantly outperforms a baseline nonlinear tracking controller with up to four times smaller worst-case height tracking errors. We also empirically demonstrate the ability of our learned model to generalize to larger swarm sizes.

- Line Coverage with Multiple Robots

    Author: Agarwal, Saurav | University of North Carolina at Charlotte
    Author: Akella, Srinivas | University of North Carolina at Charlotte
 
    keyword: Aerial Systems: Applications; Path Planning for Multiple Mobile Robots or Agents; Multi-Robot Systems

    Abstract : The <i>line coverage problem</i> is the coverage of linear environment features (e.g., road networks, power lines), modeled as 1D segments, by one or more robots while respecting resource constraints (e.g., battery capacity, flight time) for each of the robots. The robots incur direction dependent costs and resource demands as they traverse the edges. We treat the line coverage problem as an optimization problem, with the total cost of the tours as the objective, by formulating it as a mixed integer linear program (MILP). The line coverage problem is NP-hard and hence we develop a heuristic algorithm, Merge-Embed-Merge (MEM). We compare it against the optimal MILP approach and a baseline heuristic algorithm, Extended Path Scanning. We show the MEM algorithm is fast and suitable for real-time applications. To tackle large-scale problems, our approach performs graph simplification and graph partitioning, followed by robot tour generation for each of the partitioned subgraphs. We demonstrate our approach on a large graph with 4,658 edges and 4,504 vertices that represents an urban region of about 16 sq. km. We compare the performance of the algorithms on several small road networks and experimentally demonstrate the approach using UAVs on the UNC Charlotte campus road network.

- Visual Coverage Maintenance for Quadcopters Using Nonsmooth Barrier Functions

    Author: Funada, Riku | The University of Texas at Austin
    Author: Santos, Mar�a | Georgia Institute of Technology
    Author: Gencho, Takuma | Tokyo Institute of Technology
    Author: Yamauchi, Junya | Tokyo Institute of Technology
    Author: Fujita, Masayuki | Tokyo Institute of Technology
    Author: Egerstedt, Magnus | Georgia Institute of Technology
 
    keyword: Cooperating Robots; Multi-Robot Systems; Sensor Networks

    Abstract : This paper presents a coverage control algorithm for teams of quadcopters with downward facing visual sensors that prevents the appearance of coverage holes in-between the monitored areas while maximizing the coverage quality as much as possible. We derive necessary and sufficient conditions for preventing the appearance of holes in-between the fields of views among trios of robots. Because this condition can be expressed as logically combined constraints, control nonsmooth barrier functions are implemented to enforce it. An algorithm which extends control nonsmooth barrier functions to hybrid systems is implemented to manage the switching among barrier functions caused by the changes of the robots composing trio. The performance and validity of the proposed algorithm are evaluated in simulation as well as on a team of quadcopters.

- Autonomous Reflectance Transformation Imaging by a Team of Unmanned Aerial Vehicles

    Author: Kr�tk�, V�t | Czech Technical University in Prague
    Author: Petr�&#269;ek, Pavel | Czech Technical University in Prague
    Author: Spurny, Vojtech | Czech Technical University in Prague
    Author: Saska, Martin | Czech Technical University in Prague
 
    keyword: Aerial Systems: Applications; Cooperating Robots; Multi-Robot Systems

    Abstract : A Reflectance Transformation Imaging technique (RTI) realized by multi-rotor Unmanned Aerial Vehicles (UAVs) with a focus on deployment in difficult to access buildings is presented in this paper. RTI is a computational photographic method that captures a surface shape and color of a subject and enables its interactive re-lighting from any direction in a software viewer, revealing details that are not visible with the naked eye. The input of RTI is a set of images captured by a static camera, each one under illumination from a different known direction. We present an innovative approach applying two multi-rotor UAVs to perform this scanning procedure in locations that are hardly accessible or even inaccessible for people. The proposed system is designed for its safe deployment within real-world scenarios in historical buildings with priceless historical value.

-  Localization of Ionizing Radiation Sources by Cooperating Micro Aerial Vehicles with Pixel Detectors in Real-Time

    Author: Stibinger, Petr | Czech Technical University in Prague
    Author: Baca, Tomas | Czech Technical Univerzity in Prague
    Author: Saska, Martin | Czech Technical University in Prague


## Biological Cell Manipulation
- Design and Control of a Piezo Drill for Robotic Piezo-Driven Cell Penetration

    Author: Dai, Changsheng | University of Toronto
    Author: Xin, Liming | University of Toronto
    Author: Zhang, Zhuoran | University of Toronto
    Author: Shan, Guanqiao | University of Toronto
    Author: Wang, Tiancong | University of Toronto
    Author: Zhang, Kaiwen | University of Toronto
    Author: Wang, Xian | University of Toronto
    Author: Chu, Lap-Tak | University of Toronto
    Author: Ru, Changhai | Soochow University
    Author: Sun, Yu | University of Toronto
 
    keyword: Automation at Micro-Nano Scales; Biological Cell Manipulation

    Abstract : Cell penetration is an indispensable step in many cell surgery tasks. Conventionally, cell penetration is achieved by passively indenting and eventually puncturing the cell membrane, during which undesired large cell deformation is induced. Piezo drills have been developed to penetrate cells with less deformation. However, existing piezo drills suffer from large lateral vibration or are incompatible with standard clinical setup. Furthermore, it is challenging to accurately determine the time instance of cell membrane puncturing; thus, the time delay to stop piezo pulsing causes cytoplasm stirring and cell damage. This paper reports a new robotic piezo-driven cell penetration technique, in which the piezo drill device induces small lateral vibrations and is fully compatible with standard clinical setup. Techniques based on corner-feature probabilistic data association filter and motion history images were developed to automatically detect cell membrane breakage by piezo drilling. Experiments on hamster oocytes confirmed that the system is capable of achieving a small cell deformation of 5.68�2.74 �m (vs. 54.29�10.21 �m by conventional passive approach) during cell penetration. Automated detection of membrane breakage had a success rate of 95.0%, and the time delay to stop piezo vibration was 0.51�0.27 s vs. 2.32�0.98 s manually. This reduced time delay together with smaller cell deformation led to higher oocyte post-penetration survival rate (92.5% vs. 77.5% passively)

- Model-Based Robotic Cell Aspiration: Tackling Nonlinear Dynamics and Varying Cell Sizes

    Author: Shan, Guanqiao | University of Toronto
    Author: Zhang, Zhuoran | University of Toronto
    Author: Dai, Changsheng | University of Toronto
    Author: Wang, Xian | University of Toronto
    Author: Chu, Lap-Tak | University of Toronto
    Author: Sun, Yu | University of Toronto
 
    keyword: Biological Cell Manipulation; Automation at Micro-Nano Scales

    Abstract : Aspirating a single cell from the outside to the inside of a micropipette is widely used for cell transfer and manipulation. Due to the small volume of a single cell (picoliter) and nonlinear dynamics involved in the aspiration process, it is challenging to accurately and quickly position a cell to the target position inside a micropipette. This paper reports the first mathematical model that describes the nonlinear dynamics of cell motion inside a micropipette, which takes into account oil compressibility and connecting tube's deformation. Based on the model, an adaptive controller was designed to effectively compensate for the cell position error by estimating the time-varying cell medium length and speed in real time. In experiments, small-sized cells (human sperm, head width: ~3 �m), medium-sized cells (T24 cancer cells, diameter: ~15 �m), and large-sized cells (mouse embryos, diameter: ~90 �m) were aspirated using different-sized micropipettes for evaluating the performance of the model and the controller. Based on aspirating 150 cells, the model-based adaptive control method was able to complete the positioning of a cell inside a micropipette within 6 seconds with a positioning accuracy of �3 pixels and a success rate higher than 94%.

- Automated High-Productivity Microinjection System for Adherent Cells

    Author: Pan, Fei | City University of Hong Kong
    Author: Chen, Shuxun | City University of Hong Kong
    Author: Jiao, Yang | City University of Hong Kong
    Author: Guan, Zhangyan | City University of Hong Kong
    Author: Shakoor, Adnan | City University of Hong Kong
    Author: Sun, Dong | City University of Hong Kong
 
    keyword: Biological Cell Manipulation

    Abstract : Automated microinjection systems for suspension cells have been studied for years. Nevertheless, microinjection systems for adherent cells still suffer from laborious manual operations and low productivity. This paper presents a new automated microinjection system with high productivity for adherent cells. This system enhances productivity through four approaches. First, cells are detected automatically to replace manual selections. Second, the injection paths of detected cells are optimized rapidly to save time. Third, the penetration depth is adjusted adaptively according to the moving plane of the dish holder plate. Finally, constant outflow-based injection is adopted to minimize clogging. The first three approaches aim to improve the injection speed, and the last one aims to extend the usage time of micropipettes. Experiments of massive injections on MC3T3-E1 cells are performed to evaluate cell detection efficiency, injection speed, success rate, and survival rate. Results confirm that the system allows injections of over 1500 cells in one hour without much training and preparation a priori.

- High Fidelity Force Feedback Facilitates Manual Injection in Biological Samples

    Author: Mohand Ousaid, Abdenbi | University of Franche-Comte
    Author: Haliyo, Dogan Sinan | Sorbonne Université
    Author: R�gnier, Stéphane | Sorbonne University
    Author: Hayward, Vincent | Université Pierre Et Marie Curie
 
    keyword: Biological Cell Manipulation; Telerobotics and Teleoperation; Human Performance Augmentation

    Abstract : Micro-teleoperated interaction with biological cells is of special interest. The low fidelity of previous systems aimed at such small scale tasks prompted the design of a novel manual bilateral cell injection system. This systems employed the coupling of a null-displacement active force sensor with a haptic device having negligible effective inertia. This combination yielded a bilateral interaction system that was unconditionally stable even when the scaling gains were high. To demonstrate the capability of this system, two experiments were performed. A hard trout egg was delicately punctured and a small dye amount was injected in an embryo within a zebra fish egg without causing other forms of damage. The results demonstrate that the system let an operator dextrously interact with reduced reliance on visual feedback.

- Dynamic Response of Swimming Paramecium Induced by Local Stimulation Using a Threadlike-Microtool

    Author: Ahmad, Belal | FEMTO-ST Institute
    Author: Maeda, Hironobu | Kyushu Institute of Technology
    Author: Kawahara, Tomohiro | Kyushu Institute of Technology
 
    keyword: Biological Cell Manipulation; Automation at Micro-Nano Scales; Micro/Nano Robots

    Abstract : In this paper, an approach for realizing local area stimulation of single motile microorganism (Paramecium) by a microtool is described. In order to overcome the hydrodynamic drag force acting on the thin and long tool in the fluidic environment, a magnetic compensation approach to improve the positioning accuracy of the metallic microtool is introduced. The permanent magnets' arrangement that reduced the vertical component and enhanced the horizontal component of the magnetic force is modeled and clarified through numerical simulations and actual experiments on the microtool with a diameter of 50 �m. As a result, the positioning accuracy of the microtool using the magnetic compensation is improved to approximately 200 �m. Finally, the performance and practicality of the integrated platform are confirmed by conducting experiments on a freely swimming Paramecium. By virtue of the low fluidic disturbance generated by the microtool, the stimulation did not cause a tracking failure and the dynamic reaction of the Paramecium is confirmed without any immobilization manners for the first time ever. The avoiding reactions in response to mechanical stimulation are evaluated by analysing the captured image data with a spatial resolution of less than 5 �m and a time resolution of less than 5 ms.

- Injection of a Fluorescent Microsensor into a Specific Cell by Laser Manipulation and Heating with Multiple Wavelengths of Light

    Author: Maruyama, Hisataka | Nagoya University
    Author: Hashim, Hairulazwan | Universiti Tun Hussein Onn Malaysia
    Author: Yangawa, Ryota | Nagoya University
    Author: Arai, Fumihito | Nagoya University
 
    keyword: Micro/Nano Robots; Biological Cell Manipulation

    Abstract : In this study, we propose the manipulation and cell injection of a fluorescent microsensor using multiple wavelengths of light. The fluorescent microsensor is made of a 1-�m polystyrene particle containing infrared (IR: 808 nm) absorbing dye and Rhodamine B. The polystyrene particle can be manipulated in water using a 1064-nm laser because the refractive index of the polystyrene is 1.6 (refractive index of water: 1.3). The IR absorbing dye absorbs 808-nm light but does not absorb the 1064-nm laser. Rhodamine B is a temperature-sensitive fluorescent dye (excitation wavelength: 488 nm, emission wavelength: 560 nm). The functions of manipulation, heating for injection, and temperature measurement are achieved by different wavelengths of 1064 nm, 808 nm, and 488 nm, respectively. The temperature increase of fluorescent microsensor with 808-nm (40 mW, 10 s) laser was approximately 15�C, and enough for injection of fluorescent microsensor. We demonstrated manipulation and injection of the microsensor into Madin-Darby canine kidney cell using 1064-nm and 808-nm lasers. These results confirmed the effectiveness of our proposed cell injection of a fluorescent microsensor using multiple wavelengths of light.

## Cooperating Robots
- Correspondence Identification in Collaborative Robot Perception through Maximin Hypergraph Matching

    Author: Gao, Peng | Colorado School of Mines
    Author: Zhang, Ziling | Colorado School of Mines
    Author: Guo, Rui | Toyota InfoTechnology Center USA
    Author: Lu, Hongsheng | Toyota Motor North America
    Author: Zhang, Hao | Colorado School of Mines
 
    keyword: Cooperating Robots; RGB-D Perception

    Abstract : Correspondence identification is an essential problem for collaborative multi-robot perception, with the objective of deciding the correspondence of objects that are observed in the field of view of each robot. In this paper, we introduce a novel maximin hypergraph matching approach that formulates correspondence identification as a hypergraph matching problem. The proposed approach incorporates both spatial relationships and appearance features of objects to improve representation capabilities. It also integrates the maximin theorem to optimize the worst case scenario in order to address distractions caused by non-covisible objects. In addition, we design an optimization algorithm to address the formulated non-convex non-continuous optimization problem. We evaluate our approach and compare it with seven previous techniques in two application scenarios, including multi-robot coordination on real robots and connected autonomous driving in simulations. Experimental results have validated the effectiveness of our approach in identifying object correspondence from partially overlapped views in collaborative perception, and have shown that the proposed maximin hypergraph matching approach outperforms previous techniques and obtains state-of-the-art performance.

- Scalable Target-Tracking for Autonomous Vehicle Fleets

    Author: Shorinwa, Ola | Stanford University
    Author: Yu, Javier | Stanford University
    Author: Halsted, Trevor | Stanford University
    Author: Koufos, Alex | Stanford University
    Author: Schwager, Mac | Stanford University
 
    keyword: Sensor Networks; Distributed Robot Systems; Multi-Robot Systems

    Abstract : We present a scalable and distributed target tracking algorithm based on the Alternating Direction Method of Multipliers (ADMM), which is well-suited for a fleet of autonomous cars communicating over a vehicle-to-vehicle network. Each sensor executes iterations of a Kalman filter-like update followed by a local communication round with local neighbors, such that each agent's estimate converges to the joint textit{maximum a posteriori} solution of the corresponding estimation problem without requiring the communication of measurements or measurement models. We show that, given a fixed communication bandwidth, our method outperforms the Consensus Kalman Filter in recovering the centralized estimate.	We also demonstrate the algorithm in a high fidelity urban driving simulator (CARLA), in which 50 autonomous cars connected on a time-varying communication network track the locations of 50 target vehicles using a simulated segmented vision sensor.

- A Dynamic Weighted Area Assignment Based on a Particle Filter for Active Cooperative Perception

    Author: Acevedo, Jose Joaqu�n | University of Seville
    Author: Teixeira de Sousa Messias, Jo�o Vicente | Latent Logic
    Author: Capitan, Jesus | University of Seville
    Author: Ventura, Rodrigo | Instituto Superior Técnico
    Author: Merino, Luis | Universidad Pablo De Olavide
    Author: Lima, Pedro U. | Instituto Superior Técnico - Institute for Systems and Robotics
 
    keyword: Cooperating Robots; Path Planning for Multiple Mobile Robots or Agents; Motion and Path Planning

    Abstract : This paper addresses an Active Cooperative Perception problem for Networked Robots Systems. Given a team of networked robots, the goal is finding a target using their inherent uncertain sensor data. The paper proposes a particle filter to model the probability distribution of the position of the target, which is updated using detection measurements from all robots. Then, an information-theoretic approach based on the RRT* algorithm is used to determine the optimal robots trajectories that maximize the information gain while surveying the map. Finally, a dynamic area weighted allocation approach based on particle distribution and coordination variables is proposed to coordinate the networked robots in order to cooperate efficiently in this active perception problem. Simulated and real experimental results are provided to analyze, evaluate and validate the proposed approach.

- Flying Batteries: In-Flight Battery Switching to Increase Multirotor Flight Time

    Author: Jain, Karan | UC Berkeley
    Author: Mueller, Mark Wilfried | University of California, Berkeley
 
    keyword: Cooperating Robots; Mechanism Design; Aerial Systems: Applications

    Abstract : We present a novel approach to increase the flight time of a multirotor via mid-air docking and in-flight battery switching. A main quadcopter flying using a primary battery has a docking platform attached to it. A `flying battery' -- a small quadcopter carrying a secondary battery -- is equipped with docking legs that can mate with the main quadcopter's platform. Connectors between the legs and the platform establish electrical contact on docking, and enable power transfer from the secondary battery to the main quadcopter. A custom-designed circuit allows arbitrary switching between the primary battery and secondary battery. We demonstrate the concept in a flight experiment involving repeated docking, battery switching, and undocking. This is shown in the video attachment. The experiment increases the flight time of the main quadcopter by a factor of 4.7x compared to solo flight, and 2.2x a theoretical limit for that given multirotor. Importantly, this increase in flight time is not associated with a large increase in overall vehicle mass or size, leaving the main quadcopter in fundamentally the same safety class.

- Sensor Assignment Algorithms to Improve Observability While Tracking Targets (I)

    Author: Zhou, Lifeng | Virginia Tech
    Author: Tokekar, Pratap | University of Maryland
 
    keyword: Cooperating Robots; Planning, Scheduling and Coordination; Sensor-based Control

    Abstract : In this paper, we study two sensor assignment problems for multitarget tracking with the goal of improving the observability of the underlying estimator. We consider various measures of the observability matrix as the assignment value function. We first study the general version where the sensors must form teams to track individual targets. If the value function is monotonically increasing and submodular, then a greedy algorithm yields a 1/2�approximation. We then study a restricted version where exactly two sensors must be assigned to each target. We present a 1/3�approximation algorithm for this problem, which holds for arbitrary value functions (not necessarily submodular or monotone). In addition to approximation algorithms, we also present various properties of observability measures. We show that the inverse of the condition number of the observability matrix is neither monotone nor submodular, but present other measures that are. Specifically, we show that the trace and rank of the symmetric observability matrix are monotone and submodular and the log determinant of the symmetric observability matrix is monotone and submodular when the matrix is nonsingular. If the target's motion model is not known, the inverse cannot be computed exactly. Instead, we present a lower bound for distance sensors. In addition to theoretical results, we evaluate our results empirically through simulations.

- Coordinated Bayesian-Based Bioinspired Plume Source Term Estimation and Source Seeking for Mobile Robots (I)

    Author: Bourne, Joseph R. | University of Utah Robotics Center
    Author: Pardyjak, Eric | University of Utah
    Author: Leang, Kam K. | University of Utah
 
    keyword: Cooperating Robots; Multi-Robot Systems; Probability and Statistical Methods

    Abstract : A new nonparametric Bayesian-based motion planning algorithm for autonomous plume source term estimation (STE) and source seeking (SS) is presented. The algorithm is designed for mobile robots equipped with gas concentration sensors. Specifically, robots coordinate and utilize a Gaussian-plume likelihood model in a Bayesian-based STE process, then they simultaneously search for and navigate toward the source through model based, bioinspired SS methods such as biased-random-walk and surge-casting. Compared with the state-of-the-art Bayesian- and sensor-based STE/SS motion planners, the strategy described takes advantage of coordination between multiple robots and the estimated plume model for faster and more robust SS, rather than rely on direct or filtered sensor measurements. A set of Monte Carlo simulation studies are conducted to compare the performance between the uncoordinated and coordinated algorithms for different robot team sizes and starting conditions. Additionally, the algorithms are validated experimentally through a laboratory-safe, realistic humid-air plume that behaves similar to a gas plume, to test STE and SS using mobile ground robots equipped with humidity sensors. Simulation and experimental results show consistently that the algorithm involving coordination outperforms traditional bioinspired SS algorithms and it is approximately twice as fast as the uncoordinated case. Finally, the plume source is distorted to study the algorithm's limitations.

## RGB-D Perception
- ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation

    Author: Sajjan, Shreeyak | Synthesis.ai
    Author: Moore, Matthew | Synthesis.ai
    Author: Pan, Mike | Synthesis.ai
    Author: Nagaraja, Ganesh | Synthesis.ai
    Author: Lee, Johnny | Google
    Author: Zeng, Andy | Google
    Author: Song, Shuran | Columbia University
 
    keyword: Perception for Grasping and Manipulation; RGB-D Perception; Deep Learning in Robotics and Automation

    Abstract : Transparent objects are a common part of everyday life, yet they possess unique visual properties that make them incredibly difficult for standard 3D sensors to produce accurate depth estimates for, and often appear as noisy or distorted approximations of the surfaces that lie behind them. To address these challenges, we present ClearGrasp -- a deep learning approach for estimating accurate 3D geometry of transparent objects from a single RGB-D image for robotic manipulation. Given a single RGB-D image of transparent objects, ClearGrasp uses deep convolutional networks to infer a set of information from the color image (surface normals, masks of transparent surfaces, and occlusion boundaries), then uses these outputs to refine the initial depth estimates for all transparent surfaces in the scene. To train and test ClearGrasp, we construct a large-scale synthetic dataset of over 40,000 RGB-D images, as well as a real-world test benchmark with 286 RGB-D images of transparent objects and their ground truth geometries. The experiments demonstrate that ClearGrasp is substantially better than monocular depth estimation baselines and is capable of generalizing to real-world images and novel objects. We also demonstrate that ClearGrasp can be applied out-of-the-box to improve state-of-the-art grasping algorithms' performance on transparent objects. Code, data, and benchmarks will be released. Supplementary materials: https://sites.google.com/view/cleargrasp

- 6D Object Pose Regression Via Supervised Learning on Point Clouds

    Author: Gao, Ge | University of Hamburg
    Author: Lauri, Mikko | University of Hamburg
    Author: Wang, Yulong | Tsinghua University
    Author: Hu, Xiaolin | Tsinghua University
    Author: Zhang, Jianwei | University of Hamburg
    Author: Frintrop, Simone | University of Hamburg
 
    keyword: RGB-D Perception; Perception for Grasping and Manipulation

    Abstract : This paper addresses the task of estimating the 6 degrees of freedom pose of a known 3D object from depth information represented by a point cloud. Deep features learned by convolutional neural networks from color information have been the dominant features to be used for inferring object poses, while depth information receives much less attention. However, depth information contains rich geometric information of the object shape, which is important for inferring the object pose. We use depth information represented by point clouds as the input to both deep networks and geometry-based pose refinement and use separate networks for rotation and translation regression. We argue that the axis-angle representation is a suitable rotation representation for deep learning, and use a geodesic loss function for rotation regression. Ablation studies show that these design choices outperform alternatives such as the quaternion representation and L2 loss, or regressing translation and rotation with the same network. Our simple yet effective approach clearly outperforms state-of-the-art methods on the YCB-video dataset.

- YCB-M: A Multi-Camera RGB-D Dataset for Object Recognition and 6DoF Pose Estimation

    Author: Grenzd�rffer, Till | Osnabrueck University
    Author: G�nther, Martin | DFKI
    Author: Hertzberg, Joachim | University of Osnabrueck
 
    keyword: RGB-D Perception; Object Detection, Segmentation and Categorization; Computer Vision for Other Robotic Applications

    Abstract : While a great variety of 3D cameras have been introduced in recent years, most publicly available datasets for object recognition and pose estimation focus on one single camera. In this work, we present a dataset of 32 scenes that have been captured by 7 different 3D cameras, totaling 49,294 frames. This allows evaluating the sensitivity of pose estimation algorithms to the specifics of the used camera and the development of more robust algorithms that are more independent of the camera model. Vice versa, our dataset enables researchers to perform a quantitative comparison of the data from several different cameras and depth sensing technologies and evaluate their algorithms before selecting a camera for their specific task. The scenes in our dataset contain 20 different objects from the common benchmark YCB object and model set. We provide full ground truth 6DoF poses for each object, per-pixel segmentation, 2D and 3D bounding boxes and a measure of the amount of occlusion of each object. We have also performed an initial evaluation of the cameras using our dataset on a state-of-the-art object recognition and pose estimation system (DOPE).

- Depth Based Semantic Scene Completion with Position Importance Aware Loss

    Author: Liu, Yu | The University of Adelaide
    Author: Li, Jie | Nanjing University of Science and Technology
    Author: Yuan, Xia | Nanjing University of Science and Technology
    Author: Zhao, Chunxia | Nanjing University of Science and Technology
    Author: Siegwart, Roland | ETH Zurich
    Author: Reid, Ian | University of Adelaide
    Author: Cadena Lerma, Cesar | ETH Zurich
 
    keyword: RGB-D Perception; Semantic Scene Understanding; Object Detection, Segmentation and Categorization

    Abstract : Semantic Scene Completion (SSC) refers to the task of inferring the 3D semantic segmentation of a scene while simultaneously completing the 3D shapes. We propose PALNet, a novel hybrid network for SSC based on single depth. PALNet utilizes a two-stream network to extract both 2D and 3D features from multi-stages using fine-grained depth information to efficiently captures the context, as well as the geometric cues of the scene. Current methods for SSC treat all parts of the scene equally causing unnecessary attention to the interior of objects. To address this problem, we propose Position Aware Loss(PA-Loss) which is position importance aware while training the network. Specifically, PA-Loss considers Local Geometric Anisotropy to determine the importance of different positions within the scene. It is beneficial for recovering key details like the boundaries of objects and the corners of the scene. Comprehensive experiments on two benchmark datasets demonstrate the effectiveness of the proposed method and its superior performance. Code and demo are avaliable at: https://github.com/UniLauX/PALNet.

- Self-Supervised 6D Object Pose Estimation for Robot Manipulation

    Author: Deng, Xinke | University of Illinois at Urbana-Champaign
    Author: Xiang, Yu | NVIDIA
    Author: Mousavian, Arsalan | NVIDIA
    Author: Eppner, Clemens | NVIDIA
    Author: Bretl, Timothy | University of Illinois at Urbana-Champaign
    Author: Fox, Dieter | University of Washington
 
    keyword: Perception for Grasping and Manipulation; RGB-D Perception; Deep Learning in Robotics and Automation

    Abstract : To teach robots to learn skills, it is crucial to obtain data with supervision. Since annotating real world data is time-consuming and expensive, enabling robots to learn in a self-supervised way is important. In this work, we introduce a robot system for self-supervised 6D object pose estimation. Starting from modules trained in simulation, our system is able to label real world images with accurate 6D object poses for self-supervised learning. In addition, the robot interacts with objects in the environment to change the object configuration by grasping or pushing objects. In this way, our system is able to continuously collect data and improve its pose estimation modules. We show that the self-supervised learning improves object segmentation and 6D pose estimation performance, and consequently enables the system to grasp objects robustly.

- Panoptic 3D Mapping and Object Pose Estimation Using Adaptively Weighted Semantic Information

    Author: Hoang, Dinh-Cuong | Orebro University
    Author: Lilienthal, Achim J. | Orebro University
    Author: Stoyanov, Todor | Örebro University
 
    keyword: RGB-D Perception; Object Detection, Segmentation and Categorization

    Abstract : We present a system capable of reconstructing highly detailed object-level models and estimating the 6D pose of objects by means of an RGB-D camera. In this work, we integrate deep-learning-based semantic segmentation, instance segmentation, and 6D object pose estimation into a state of the art RGB-D mapping system. We leverage the pipeline of ElasticFusion as a backbone and propose modifications of the registration cost function to make full use of the semantic class labels in the process. The proposed objective function features tunable weights for the depth, appearance, and semantic information channels, which are learned from data. A fast semantic segmentation and registration weight prediction convolutional neural network (Fast-RGBD-SSWP) suited to efficient computation is introduced. In addition, our approach explores performing 6D object pose estimation from multiple viewpoints supported by the high-quality reconstruction system. The developed method has been verified through experimental validation on the YCB-Video dataset and a dataset of warehouse objects. Our results confirm that the proposed system performs favorably in terms of surface reconstruction, segmentation quality, and accurate object pose estimation in comparison to other state-of-the-art systems. Our code and video are available at https://sites.google.com/view/panoptic-mope.

## Task Planning
- Online Trajectory Planning through Combined Trajectory Optimization and Function Approximation: Application to the Exoskeleton Atalante

    Author: Duburcq, Alexis | Wandercraft
    Author: Chevaleyre, Yann | Univ. Paris Dauphine
    Author: Bredeche, Nicolas | Université Pierre Et Marie Curie
    Author: Boeris, Guilhem | Wandercraft
 
    keyword: Task Planning; Optimization and Optimal Control; Humanoid and Bipedal Locomotion

    Abstract : Autonomous robots require online trajectory planning capability to operate in the real world. Efficient offline trajectory planning methods already exist, but are computationally demanding, preventing their use online. In this paper, we present a novel algorithm called Guided Trajectory Learning that learns a function approximation of solutions computed through trajectory optimization while ensuring accurate and reliable predictions. This function approximation is then used online to generate trajectories. This algorithm is designed to be easy to implement, and practical since it does not require massive computing power. It is readily applicable to any robotics systems and effortless to set up on real hardware since robust control strategies are usually already available. We demonstrate the computational performance of our algorithm on flat-foot walking with a self-balanced exoskeleton.

- Act, Perceive, and Plan in Belief Space for Robot Localization

    Author: Colledanchise, Michele | IIT - Italian Institute of Technology
    Author: Malafronte, Damiano | Istituto Italiano Di Tecnologia
    Author: Natale, Lorenzo | Istituto Italiano Di Tecnologia
 
    keyword: Task Planning; Reactive and Sensor-Based Planning; Visual-Based Navigation

    Abstract : In this paper, we outline an interleaved acting and planning technique to rapidly reduce the uncertainty of the estimated robot's pose by perceiving relevant information from the environment, as recognizing an object or asking someone for a direction. Generally, existing localization approaches rely on low-level geometric features such as points, lines, and planes. While these approaches provide the desired accuracy, they may require time to converge, especially with incorrect initial guesses. In our approach, a task planner computes a sequence of action and perception tasks to actively obtain relevant information from the robot's perception system. We validate our approach in large state spaces, to show how the approach scales, and in real environments, to show the applicability of our method on real robots. We prove that our approach is sound, probabilistically complete, and tractable in practical cases.

- Decentralized Task Allocation in Multi-Agent Systems Using a Decentralized Genetic Algorithm

    Author: Patel, Ruchir | University of Maryland, College Park
    Author: Rudnick-Cohen, Eliot | University of Maryland, College Park
    Author: Azarm, Shapour | University of Maryland
    Author: Otte, Michael W. | University of Maryland
    Author: Xu, Huan | University of Maryland
    Author: Herrmann, Jeffrey | University of Maryland
 
    keyword: Task Planning; Cooperating Robots; Multi-Robot Systems

    Abstract : In multi-agent collaborative search missions, task allocation is required to determine which agents will perform which tasks. We propose a new approach for decentralized task allocation based on a decentralized genetic algorithm (GA). The approach parallelizes a genetic algorithm across the team of agents, making efficient use of their computational resources. In the proposed approach, the agents continuously search for and share better solutions during task execution. We conducted simulation experiments to compare the decentralized GA approach and several existing approaches. Two objectives were considered: a min-sum objective (minimizing the total distance traveled by all agents) and a min-time objective (minimizing the time to visit all locations of interest). The results showed that the decentralized GA approach yielded task allocations that were better on the min-time objective than those created by existing approaches and solutions that were reasonable on the min-sum objective. The decentralized GA improved min-time performance by an average of 5.6% on the larger instances. The results indicate that decentralized evolutionary approaches have a strong potential for solving the decentralized task allocation problem.

- Fast and Resilient Manipulation Planning for Target Retrieval in Clutter

    Author: Nam, Changjoo | Korea Institute of Science and Technology
    Author: Lee, JinHwi | Hanyang University
    Author: Cheong, Sang Hun | Korea University, KIST
    Author: Cho, Brian Younggil | Korea Institute of Science and Technology
    Author: Kim, ChangHwan | Korea Institute of Science and Technology
 
    keyword: Task Planning; Manipulation Planning; Motion and Path Planning

    Abstract : This paper presents a task and motion planning (TAMP) framework for a robotic manipulator in order to retrieve a target object from clutter. We consider a configuration of objects in a confined space with a high density so no collision-free path to the target exists. The robot must relocate some objects to retrieve the target without collisions. For fast completion of object rearrangement, the robot aims to optimize the number of pick-and-place actions which often determines the efficiency of a TAMP framework.<p>We propose a task planner incorporating motion planning to generate executable plans which aims to minimize the number of pick-and-place actions. In addition to fully known and static environments, our method can deal with uncertain and dynamic situations incurred by occluded views. Our method is shown to reduce the number of pick-and-place actions compared to baseline methods (e.g., at least 28.0% of reduction in a known static environment with 20 objects).

- Multi-Robot Task and Motion Planning with Subtask Dependencies

    Author: Motes, James | University of Illinois Urbana-Champaign
    Author: Sandstrom, Read | Texas A&amp;M University
    Author: Lee, Hannah | Colorado School of Mines
    Author: Thomas, Shawna | Texas A&amp;M University
    Author: Amato, Nancy | University of Illinois
 
    keyword: Task Planning; Motion and Path Planning; Multi-Robot Systems

    Abstract : We present a multi-robot integrated task and motion method capable of handling sequential subtask dependencies within multiply decomposable tasks. We map the multi-robot pathfinding method, Conflict Based Search, to task planning and integrate this with motion planning to create TMP-CBS. TMP-CBS couples task decomposition, allocation, and planning to support cases where the optimal solution depends on robot availability and inter-team conflict avoidance. We show improved planning time for simpler task sets and generate optimal solutions w.r.t. the state space representation for a broader range of problems than prior methods.

- Untethered Soft Millirobot with Magnetic Actuation

    Author: Bhattacharjee, Anuruddha | Southern Methodist University
    Author: Rogowski, Louis | Southern Methodist University
    Author: Zhang, Xiao | Southern Methodist University
    Author: Kim, MinJun | Southern Methodist University
 
    keyword: Task Planning; Soft Robot Materials and Design; Cellular and Modular Robots

    Abstract : This paper presents scalable designs and fabrication, actuation, and manipulation techniques for soft millirobots under uniform magnetic field control. The millirobots were fabricated through an economic and robust moulding technique using polydimethylsiloxane (PDMS), acrylonitrile butadiene styrene (ABS) filaments, and 3D printed polylactic acid (PLA) rings. The soft millirobots were simple hollow rod-like structures with different configurations of embedded permanent magnets inside of their soft-body or at their ends. The soft-robots were actuated using six different motion modes including: pivot walking, rolling, tumbling, side-tapping, wiggling, and wavy-motion under an external uniform magnetic field control system. The velocities of the millirobots under different motion modes were analyzed under varying magnetic flux densities (<i><b>B</b></i>). Moreover, deformation of the soft-robotic body in response to the magnetic field strength was measured and a deflection curve showing bending angle (<i>&#934;</i>) was produced. Soft millirobots were navigated through a maze using a combination of the available motion modes. Different arrangements of the embedded permanent magnets enabled individual soft millirobots to respond heterogeneously under the same magnetic field inputs towards performing assembly and disassembly operation as modular subunits. Overall, this soft millirobot platform shows enormous potential for minimally invasive <i>in vivo</i> applications.

## Brain-Machine Interfaces
- Accelerated Robot Learning Via Human Brain Signals

    Author: Akinola, Iretiayo | Columbia University
    Author: Wang, Zizhao | University of Michigan-Ann Arbor
    Author: Shi, Jack | Columbia University
    Author: He, Xiaomin | Columbia University
    Author: Lapborisuth, Pawan | Columbia University
    Author: Xu, Jingxi | Columbia University
    Author: Watkins-Valls, David | Columbia University
    Author: Sajda, Paul | Columbia University
    Author: Allen, Peter | Columbia University
 
    keyword: Brain-Machine Interface; Learning and Adaptive Systems; Cognitive Human-Robot Interaction

    Abstract :  In reinforcement learning (RL), sparse rewards are a natural way to specify the task to be learned. However, most RL algorithms struggle to learn in this setting since the learning signal is mostly zeros. In contrast, humans are good at assessing and predicting the future consequences of actions and can serve as good reward/policy shapers to accelerate the robot learning process. Previous works have shown that the human brain generates an error-related signal, measurable using electroencephelography (EEG), when the human perceives the task being done erroneously. In this work, we propose a method that uses evaluative feedback obtained from human brain signals measured via scalp EEG to accelerate RL for robotic agents in sparse reward settings. As the robot learns the task, the EEG of a human observer watching the robot attempts is recorded and decoded into noisy error feedback signal. From this feedback, we use supervised learning to obtain a policy that subsequently augments the behavior policy and guides exploration in the early stages of RL. This bootstraps the RL learning process to enable learning from sparse reward. Using a simple robotic navigation task as a test bed, we show that our method achieves a stable obstacle-avoidance policy with high success rate, outperforming learning from sparse rewards only that struggles to achieve obstacle avoidance behavior or fails to advance to the goal.

- Muscle and Brain Activations in Cylindrical Rotary Controller Manipulation with Index Finger and Thumb

    Author: Okatani, Rio | DOSHISHA University
    Author: Tsumugiwa, Toru | Doshisha University
    Author: Yokogawa, Ryuichi | Doshisha University
    Author: Narusue, Mitsuhiro | Mazda Motor Corporation
    Author: Nishimura, Hiroto | Mazda Corporation
    Author: Takeda, Yuusaku | Mazda Motor Corporation
    Author: Hara, Toshihiro | Mazda Motor Corporation
 
    keyword: Brain-Machine Interface; Human Factors and Human-in-the-Loop; Cognitive Human-Robot Interaction

    Abstract : This study aim to confirm the effect of viscosity characteristics differences on the rotational manipulation of a cylindrical rotary controller with the index finger and thumb through a quantitative analysis and evaluation of muscle and brain activations. The target motion was a rotary manipulation with the index finger and thumb of a cylindrical rotary controller with a 50 mm diameter. The rotary motion of the controller produces a click sensation at every 12 degrees in the rotation. The experimental conditions were three conditions with different viscosity characteristics related to the rotary motion of the controller. The subjects were six right-handed healthy males with a mean age of 21.7 (S. D.: 1.03) years. We analyzed the brain activity from a near�infrared spectroscopy measurement system, the muscles activity using a surface myoelectric potential measurement device, the force data at the index finger and thumb tip using two independent six-axis force/torque sensors, and the position data using a 3D position measurement device. The experimental results showed that there was no significant difference in the questionnaire survey, muscle activity, and grasping force, respectively; however, a significant difference in brain activity was observed with increased controller viscosity. Therefore, it became clear that there was a change in the brain activity when rotating the cylindrical rotary controller with the viscosity characteristics related to the rotary motion.

- Real-Time Robot Reach-To-Grasp Movements Control Via EOG and EMG Signals Decoding

    Author: Specht, Bernhard | TUM
    Author: Tayeb, Zied | Technical University of Munich
    Author: Dean-Leon, Emmanuel | Technischen Universitaet Muenchen
    Author: Soroushmojdehi, Rahil | IIT
    Author: Cheng, Gordon | Technical University of Munich
 
    keyword: Brain-Machine Interface; Cognitive Human-Robot Interaction; Rehabilitation Robotics

    Abstract : In this paper, we propose a real-time human-robot interface (HRI) system, where Electrooculography (EOG) and Electromyography (EMG) signals were decoded to perform reach-to-grasp movements. For that, five different eye movements (up, down, left, right and rest) were classified in real-time and translated into commands to steer an industrial robot (UR- 10) to one of the four approximate target directions. Thereafter, EMG signals were decoded to perform the grasping task using an attached gripper to the UR-10 robot arm. The proposed system was tested offline on three different healthy subjects, and mean validation accuracy of 93.62% and 99.50% were obtained across the three subjects for EOG and EMG decoding, respectively. Furthermore, the system was successfully tested in real-time with one subject, and mean online accuracy of 91.66% and 100% were achieved for EOG and EMG decoding, respectively. Our results obtained by combining real- time decoding of EOG and EMG signals for robot control show overall the potential of this approach to develop powerful and less complex HRI systems. Overall, this work provides proof- of-concept for successful real-time control of robot arms using EMG and EOG signals, paving the way for the development of more dexterous and human-controlled assistive devices.

- Simultaneous Estimations of Joint Angle and Torque in Interactions with Environments Using EMG

    Author: Kim, Dongwon | University of Michigan
    Author: Koh, Kyung | University of Maryland
    Author: Oppizzi, Giovanni | University of Maryland
    Author: Baghi, Raziyeh | University of Maryland, Baltimore
    Author: Lo, Li-Chuan | University of Maryland at Baltimore
    Author: Zhang, Chunyang | University of Maryland at Baltimore
    Author: Zhang, Li-Qun | Rehabilitation Institute of Chicago/Northwestern University
 
    keyword: Brain-Machine Interface; Wearable Robots; Rehabilitation Robotics

    Abstract : We develop a decoding technique that estimates both the position and torque of a joint of the limb in interaction with an environment based on activities of the agonist-antagonist pair of muscles using electromyography in real time. The long short-term memory (LSTM) network is employed as the core processor of the proposed technique that is capable of learning time series of a long-time span with varying time lags. A validation that is conducted on the wrist joint shows that the decoding approach provides an agreement of greater than 95% in kinetics (i.e. torque) estimation and an agreement of greater than 85% in kinematics (i.e. angle) estimation, between the actual and estimated variables, during interactions with an environment. Also, it is revealed that the proposed decoding method inherits the strengths of the LSTM network in terms of the capability of learning EMG signals and the corresponding responses with time dependency.

- High-Density Electromyography Based Control of Robotic Devices: On the Execution of Dexterous Manipulation Tasks

    Author: Dwivedi, Anany | University of Auckland
    Author: Lara, Jaime | The University of Auckland
    Author: Cheng, Leo K. | University of Auckland
    Author: Paskaranandavadivel, Niranchan | University of Auckland
    Author: Liarokapis, Minas | The University of Auckland
 
    keyword: Brain-Machine Interface; Neurorobotics

    Abstract : Electromyography (EMG) based interfaces have been used in various robotics studies ranging from teleoperation and telemanipulation applications to the EMG based control of prosthetic, assistive, or robotic rehabilitation devices. But most of these studies have focused on the decoding of user's motion or on the control of the robotic devices in the execution of simple tasks (e.g., grasping tasks). In this work, we present a learning scheme that employs High Density Electromyography (HD-EMG) sensors to decode a set of dexterous, in-hand manipulation motions (in the object space) based on the myoelectric activations of human forearm and hand muscles. To do that, the subjects were asked to perform roll, pitch, and yaw motions manipulating two different cubes. The first cube was designed to have a center of mass coinciding with the geometric center of the cube, while for the second cube the center of mass was shifted 14 mm to the right (off-centered design). Regarding the acquisition of the myoelectric data, custom HD-EMG electrode arrays were designed and fabricated. Using these arrays, a total of 89 EMG signals were extracted. The object motion decoding was formulated as a regression problem using the Random Forests (RF) technique and the muscle importances were studied using the inherent feature variables importance calculation procedure of the RF. The muscle importance results show that different subjects use different strategies to execute the same motions on same object whe

- The Role of the Control Framework for Continuous Teleoperation of a Brain�Machine Interface-Driven Mobile Robot (I)

    Author: Tonin, Luca | University of Padova
    Author: Bauer, Felix Christian | ETH Zurich, aiCTX AG
    Author: Mill�n, Jos' del R. | EPFL
 
    keyword: Neurorobotics; Brain-Machine Interface; Telerobotics and Teleoperation

    Abstract : Despite the growing interest in brain-machine interface (BMI) driven neuroprostheses, the optimization of the control framework and the translation of the BMI output into a suitable control signal are often neglected. In this study, we propose a novel approach based on dynamical systems that was explicitly designed to take into account the nature of the BMI output. We hypothesize that such a control framework would allow users to continuously drive a mobile robot and it would enhance the navigation performance. 13 healthy users evaluated the system by using a 2-class motor imagery BMI to drive the robot to 5 targets in two experimental conditions: with a discrete control strategy, traditionally exploited in the BMI field, and with the novel continuous control framework developed herein. Experimental results showed that the new approach: i) allowed users to continuously drive the mobile robot via BMI; ii) led to significant improvements in the navigation performance; iii) promoted a better coupling between user and robot. These results highlight the importance of designing a suitable control framework to improve the performance and the reliability of BMI driven neurorobotic devices.

## Tendon/Wire Mechanism
- Asynchronous and Decoupled Control of the Position and the Stiffness of a Spatial RCM Tensegrity Mechanism for Needle Manipulation

    Author: Jurado Realpe, Jr | Université De Montpellier
    Author: Aiche, Guillaume | Université De Montpellier
    Author: Abdelaziz, Salih | LIRMM, University of Montpellier 2
    Author: Poignet, Philippe | LIRMM University of Montpellier CNRS
 
    keyword: Tendon/Wire Mechanism; Motion Control; Compliance and Impedance Control

    Abstract : This paper introduces a 2-DOF spatial remote center of motion (RCM) tensegrity mechanism, based on a double parallelogram system, dedicated for percutaneous needle insertion. The originality of this mechanism is its ability to be reconfigured and its capacity to perform a decoupled modulation of its stiffness in an asynchronous way. To do so, an analytical stiffness model of the robot is established, and a control methodology is proposed. A prototype of the robot is developed and assessed experimentally. The position tracking is evaluated using a 6-DOF magnetic tracker sensor showing a root mean square error less than 0.8� in both directions of the needle guide.

- Redundancy Resolution Integrated Model Predictive Control of CDPRs: Concept, Implementation and Experiments

    Author: Cavalcanti Santos, Joao | University of Montpellier, LIRMM
    Author: Chemori, Ahmed | Cnrs / Lirmm
    Author: Gouttefarde, Marc | CNRS
 
    keyword: Tendon/Wire Mechanism; Parallel Robots; Motion Control of Manipulators

    Abstract : This paper introduces a Model Predictive Control (MPC) strategy for fully-constrained Cable-Driven Parallel Robots. The main advantage of the proposed scheme lies in its ability to explicitly handle cable tension limits. Indeed, the cable tension distribution is performed as an integral part of the main control architecture. This characteristic significantly improves the safety of the system. Experimental results demonstrate this advantage addressing a typical pick-and-place task with two different scenarios: nominal cable tension limits and reduced maximum tension. Satisfactory tracking errors were obtained in the first scenario. In the second scenario, the desired trajectory escapes from the workspace defined by the new set of tension limits. The proposed MPC scheme is able to minimize the tracking errors without violating the tension limits. Satisfying results were also obtained regarding robustness against uncertainties on the payload mass.

- Mechanics for Tendon Actuated Multisection Continuum Arms

    Author: Gonthina, Phanideep | Clemson University
    Author: Wooten, Michael | Clemson University
    Author: Godage, Isuru S. | Depaul University
    Author: Walker, Ian | Clemson University
 
    keyword: Tendon/Wire Mechanism; Compliant Joint/Mechanism; Biologically-Inspired Robots

    Abstract : Tendon actuated multisection continuum arms have high potential for inspection applications in highly constrained spaces. They generate motion by axial and bending deformations. However, because of the high mechanical coupling between continuum sections, variable length-based kinematic models produce poor results. A new mechanics model for tendon actuated multisection continuum arms is proposed in this paper. The model combines the continuum arm curve parameter kinematics and concentric tube kinematics to correctly account for the large axial and bending deformations observed in the robot. Also, the model is computationally efficient and utilizes tendon tensions as the joint space variables thus eliminating the actuator length related problems such as slack and backlash. A recursive generalization of the model is also presented. Despite the high coupling between continuum sections, numerical results show that the model can be used for generating correct forward and inverse kinematic results. The model is then tested on a thin and long multisection continuum arm. The results show that the model can be used to successfully model the deformation.

- Trajectory Optimization for a Six-DOF Cable-Suspended Parallel Robot with Dynamic Motions Beyond the Static Workspace

    Author: Xiang, Sheng | Harbin Institute of Technology
    Author: Gao, Haibo | Harbin Institute of Technology
    Author: Liu, Zhen | Harbin Institute of Technology
    Author: Gosselin, Clement | Université Laval
 
    keyword: Tendon/Wire Mechanism; Motion Control; Dynamics

    Abstract : This paper presents a trajectory optimization formulation for planning dynamic trajectories of a six-degree-of-freedom (six-DOF) cable-suspended parallel robot (CSPR) that extend beyond the static workspace. The optimization is guided by low-dimensional dynamic models to overcome the local minima and accelerate the exploration of the narrow feasible state space. The dynamic similarity between the six-DOF CSPR and the three-DOF point-mass CSPR is discussed with the analyses of their feasible force polyhedra. Finally, the transition trajectories of a three-DOF CSPR are used as the initial guess of the translational part of the six-DOF motion. With the proposed approach, highly dynamic motions for a six-DOF CSPR are efficiently generated with multiple oscillations. The feasibility is demonstrated by point-to-point and periodic trajectories in the physics simulation.

- Design of Tensegrity-Based Manipulators: Comparison of Two Approaches to Respect a Remote Center of Motion Constraint

    Author: Begey, J�r�my | University of Strasbourg
    Author: Vedrines, Marc | ICube - INSA De Strasbourg
    Author: Renaud, Pierre | ICube AVR
 
    keyword: Tendon/Wire Mechanism; Mechanism Design; Parallel Robots

    Abstract : Tensegrity mechanisms can offer key features such as compliance and deployability for high compactness. The absence of systematic design methods has however strongly limited the development of tensegrity mechanisms for manipulation up to now. In this paper, we consider how tensegrity mechanisms can be used to respect a Remote Center of Motion (RCM) constraint. Mechanisms of high compactness, respecting RCM constraint and offering compliance can indeed be of great interest in a challenging environment such as the medical context. Two architectures are elaborated, using cable- and bar-actuated Snelson crosses. Their analysis is performed, proofs of concept are built and experimentally evaluated, and their relative interest is discussed. This work brings at the same time initial results on design of tensegrity mechanisms for manipulation in a medical environment, and first guidelines to perform tensegrity mechanism synthesis.

- Accurate Dynamic Modeling of Twisted String Actuators Accounting for String Compliance and Friction

    Author: Nedelchev, Simeon | Innopolis University
    Author: Gaponov, Igor | Innopolis University
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
 
    keyword: Tendon/Wire Mechanism; Dynamics

    Abstract : This paper proposes a more accurate dynamic model of twisted string actuators (TSAs) that accounts for both elastic deformation of strings and frictional forces, which have not been considered by any mathematical models to date. The proposed model allows more accurate estimation of the required motor torque in both low-speed (statics) and highspeed (up to 2 Hz) motion. Consideration of both joint-space and task-space frictional forces enables us to better model the hysteresis in torque behavior, which was not possible by stateof-the-art models that only considered joint-space friction. In addition, the inclusion of elastic deformation of the strings during twisting in both longitudinal and radial (transverse) directions allows the proposed model to estimate the required torque with higher accuracy, especially in statics. This paper presents both theoretical derivation of the underlying dynamic equations and their experimental evaluation on a practical setup. The analysis of the experimental data has shown that the relative error between theoretical and practical curves never exceeded 7% for the range of tested motion frequencies between 0.1 and 2 Hz. The proposed dynamic model can be used for accurate model-based control of the TSA-based systems and for efficient TSA motor selection.

## Agricultural Automation
- An Intelligent Spraying System with Deep Learning-Based Semantic Segmentation of Fruit Trees in Orchards

    Author: Kim, Jeongeun | Chonnam National University
    Author: Seol, Jaehwi | Chonnam National University
    Author: Lee, SukWoo | Chonnam National University
    Author: Hong, Se-Woon | Chonnam National University
    Author: Son, Hyoung Il | Chonnam National University
 
    keyword: Agricultural Automation; Robotics in Agriculture and Forestry; Deep Learning in Robotics and Automation

    Abstract : This study proposes an intelligent spraying system with semantic segmentation of fruit trees in a pear orchard. A fruit tree detection system was developed using the SegNet model, a semantic segmentation structure. The system is trained with images categorized into five distinct classes. The learned deep learning model performed with an accuracy of 83.79%. Further, we fusion depth data from an RGB-D camera to prevent the tree in the background from being detected. To operate the nozzles, each image captured from the camera is separated lengthwise into quarters and mapped to the nozzles. Then, the nozzle was opened when the area of fruit trees in each zone exceeded 20%. Two types of field experiments were performed in a pear orchard to verify the effectiveness of our system. From the results obtained, we can confirm the satisfactory performance of our deep learning-based intelligent spraying system. It is expected that the introduction of this system to actual farms will significantly reduce the amount of pesticide used and will make the work environment safer for farmers.

- An Efficient Planning and Control Framework for Pruning Fruit Trees

    Author: You, Alexander | Oregon State University
    Author: Sukkar, Fouad | University of Technology Sydney
    Author: Fitch, Robert | University of Technology Sydney
    Author: Karkee, Manoj | Washington State University
    Author: Davidson, Joseph | Oregon State University
 
    keyword: Agricultural Automation; Manipulation Planning; Planning, Scheduling and Coordination

    Abstract : Dormant pruning is a major cost component of fresh market tree fruit production, nearly equal in scale to harvesting the fruit. However, relatively little focus has been given to the problem of pruning trees autonomously. In this paper, we introduce a robotic system consisting of an industrial manipulator, an eye-in-hand RGB-D camera configuration, and a custom pneumatic cutter. The system is capable of planning and executing a sequence of cuts while making minimal assumptions about the environment. We leverage a novel planning framework designed for high-throughput operation which builds upon previous work to reduce motion planning time and sequence cut points intelligently. In end-to-end experiments with a set of ten different branch configurations, the system achieved a high success rate in plan execution and a 1.5x speedup in throughput versus a baseline planner, representing a significant step towards the goal of practical implementation of robotic pruning.

- Context Dependant Iterative Parameter Optimisation for Robust Robot Navigation

    Author: Binch, Adam | Saga Robotics UK
    Author: Das, Gautham | University of Lincoln
    Author: Pulido Fentanes, Jaime | Saga Robotics
    Author: Hanheide, Marc | University of Lincoln
 
    keyword: Optimization and Optimal Control; Robotics in Agriculture and Forestry; Autonomous Vehicle Navigation

    Abstract : Progress in autonomous mobile robotics has seen significant advances in the development of many algorithms for motion control and path planning. However, robust performance from these algorithms can often only be expected if the parameters controlling them are tuned specifically for the respective robot model, and optimised for specific scenarios in the environment the robot is working in. Such parameter tuning can, depending on the underlying algorithm, amount to a substantial combinatorial challenge, often rendering extensive manual tuning of these parameters intractable. In this paper, we present a framework that permits the use of different navigation actions with different parameters depending on the spatial context of the navigation task. We consider the respective navigation algorithms themselves mostly as a "black box", and find suitable parameters by means of an iterative optimisation, improving for performance metrics in simulated environments. We present a genetic algorithm incorporated into the framework and empirically show that the resulting parameter sets lead to substantial performance improvements in both simulated and real-world environments.

- Combining Domain Adaptation and Spatial Consistency for Unseen Fruits Counting: A Quasi-Unsupervised Approach

    Author: Bellocchio, Enrico | University of Perugia
    Author: Costante, Gabriele | University of Perugia
    Author: Cascianelli, Silvia | University of Perugia
    Author: Fravolini, Mario | University of Perugia
    Author: Valigi, Paolo | Universita' Di Perugia
 
    keyword: Agricultural Automation; Robotics in Agriculture and Forestry; Visual Learning

    Abstract : Autonomous robotic platforms can be effectively used to perform automatic fruits yield estimation. To this aim, robots need data-driven models that process image streams and count, even approximately, the number of fruits in an orchard. However, training such models following a supervised paradigm is expensive and unpractical. Extending pre-trained models to perform yield estimation for a completely new type of fruit is even more challenging, but interesting since this situation is typical in practice. In this work, we combine a State-of-the-Art weakly-supervised fruit counting model with an unsupervised style transfer method for addressing the task above. In this sense, our proposed approach is quasi-unsupervised. In particular, we use a Cycle-Generative Adversarial Network (C-GAN) to perform unsupervised domain adaptation and train it alongside with a Presence-Absence Classifier (PAC) that discriminates images containing fruits or not. The PAC produces the weak-supervision signal for the counting network, that can then be used on the target orchard directly. Experiments on datasets collected in four different orchards show that the proposed approach is more accurate than the supervised baseline methods.

- A Navigation Architecture for Ackermann Vehicles in Precision Farming

    Author: Carpio, Renzo Fabrizio | Roma Tre University
    Author: Potena, Ciro | Sapienza University of Rome
    Author: Maiolini, Jacopo | University of Roma 3
    Author: Ulivi, Giovanni | Université Di Roma Tre
    Author: Bono Rossello, Nicolas | ULB-SAAS
    Author: Garone, Emanuele | Université Libre De Bruxelles
    Author: Gasparri, Andrea | Université Degli Studi Roma Tre
 
    keyword: Agricultural Automation; Robotics in Agriculture and Forestry; Motion Control

    Abstract : In this work, inspired by the needs of the European H2020 Project PANTHEON, we propose a full navigation stack purposely designed for the autonomous navigation of Ackermann steering vehicles in precision farming settings. The proposed stack is composed of a local planner and a pose regulation controller, both implemented in ROS. The local planner generates, in real-time, optimal trajectories described by a sequence of successive poses. The planning problem is formulated as a real-time cost-function minimization problem over a finite time horizon where the Ackermann kinematics and the presence of obstacles are encoded as constraints. The control law ensures the convergence toward each of these poses. To do so, in this paper we propose a novel non-smooth control law designed to ensure the solvability of the pose regulation problem for the Ackermann vehicle. Theoretical characterization of the convergence property of the proposed pose regulation controller is provided. Numerical simulations along with real-world experiments are provided to corroborate the effectiveness of the proposed navigation strategy.

- MinneApple: A Benchmark Dataset for Apple Detection and Segmentation

    Author: H�ni, Nicolai | University of Minnesota
    Author: Roy, Pravakar | University of Minnesota
    Author: Isler, Volkan | University of Minnesota
 
    keyword: Agricultural Automation; Robotics in Agriculture and Forestry; Object Detection, Segmentation and Categorization

    Abstract : In this work, we present a new dataset to advance the state-of-the-art in fruit detection, segmentation, and counting in orchard environments. While there has been significant recent interest in solving these problems, the lack of a unified dataset has made it difficult to compare results. We hope to enable direct comparisons by providing a large variety of high-resolution images acquired in orchards, together with human annotations of the fruit on trees. The fruits are labeled using polygonal masks for each object instance to aid in precise object detection, localization, and segmentation. Additionally, we provide data for patch-based counting of clustered fruits. Our dataset contains over 41, 000 annotated object instances in 1000 images. We present a detailed overview of the dataset together with baseline performance analysis for bounding box detection, segmentation, and fruit counting as well as representative results for yield estimation. We make this dataset publicly available and host a CodaLab challenge to encourage comparison of results on a common dataset. To download the data and learn more about MinneApple please see the project website: http://rsn.cs.umn.edu/index.php/MinneApple. Up to date information is available online.

## Underactuated Robots
- Extending Riemmanian Motion Policies to a Class of Underactuated Wheeled-Inverted-Pendulum Robots

    Author: Wingo, Bruce | Georgia Institute of Technology
    Author: Cheng, Ching-an | Georgia Institute of Technology
    Author: Murtaza, Muhammad Ali | Georgia Institute of Technology
    Author: Zafar, Munzir | Georgia Institute of Technology
    Author: Hutchinson, Seth | Georgia Institute of Technology
 
    keyword: Underactuated Robots; Motion and Path Planning; Manipulation Planning

    Abstract : Riemannian Motion Policies (RMPs) have recently been introduced as a way to specify motion policies for robot tasks in terms of a set of second order differential equations defined directly in the task space. RMP-based approaches have the advantage of being significantly more general than traditional operational space approaches; for example, when using RMPs, generalized task inertia can be fully state-dependent (rather than merely configuration dependent), leading to more effective motions that naturally incorporate the task dynamics, as well as task constraints such as collision avoidance. Until now, RMPs have been applied only to fully actuated systems, i.e., systems for which each degree of freedom (DoF) can be directly actuated by a control input. In this paper, we present a method that generalizes the RMP formalism to a class of underacutated systems whose dynamics are amenable to a particular class of decomposition such that the original underactuated dynamics can be effectively controlled by a fully actuated subsystem. We show the efficacy of the approach by constructing a suitable decomposition for a Wheeled Inverted Pendulum (WIP) humanoid robot, and applying our method to derive motion policies for combined locomotion and manipulation tasks. Simulation results are presented for a 7-DoF system with one degree of underactuation.

- Augmenting Self-Stability: Height Control of a Bernoulli Ball Via Bang-Bang Control

    Author: Howison, Toby | University of Cambridge
    Author: Giardina, Fabio | Harvard University
    Author: Iida, Fumiya | University of Cambridge
 
    keyword: Underactuated Robots; Motion Control; Dynamics

    Abstract : Mechanical self-stability is often useful for controlling systems in uncertain and unstructured environments because it can regulate processes without explicit state observation or feedback computation. However, the performance of such systems is often not optimised, which begs the question how their dynamics can be naturally augmented by a control law to improve performance metrics. We propose a minimalistic approach to controlling mechanically self-stabilising systems by utilising model-based, feedforward bang-bang control at a global level and self-stabilising dynamics at a local level. We demonstrate the approach in the height control problem of a sphere hovering in a vertical air jet�the so-called Bernoulli Ball. After developing a model to study the system and theoretically proving global asymptotic stability, we present the augmented controller and show how to enhance performance measures and plan behaviour. Our physical experiments show that the proposed control approach has a reduced time-to-target compared to the uncontrolled system without loss of stability (ranging from a 2.4 to 4.4 fold improvement) and that we can plan sequences of target positions at will.

- Singularity-Free Inverse Dynamics for Underactuated Systems with a Rotating Mass

    Author: Tafrishi, Seyed Amir | Kyushu University
    Author: Svinin, Mikhail | Ritsumeikan University
    Author: Yamamoto, Motoji | Kyushu University
 
    keyword: Underactuated Robots; Dynamics; Nonholonomic Mechanisms and Systems

    Abstract : Motion control of underactuated systems through the inverse dynamics contains configuration singularities. These limitations in configuration space mainly stem from the inertial coupling that passive joints/bodies create. In this study, we present a model that is free from singularity while the trajectory of the rotating mass has a small-amplitude sine wave around its circle. First, we derive the modified non-linear dynamics for a rolling system. Also, the singularity regions for this underactuated system is demonstrated. Then, the wave parameters are designed under certain conditions to remove the coupling singularities. We obtain these conditions from the positive definiteness of the inertia matrix in the inverse dynamics. Finally, the simulation results are confirmed by using a prescribed Beta function on the specified states of the rolling carrier. Because our algebraic method is integrated into the non-linear dynamics, the proposed solution has a great potential to be extended to the Lagrangian mechanics with multiple degrees-of-freedom.

- Coordinated Particle Relocation Using Finite Static Friction with Boundary Walls

    Author: Schmidt, Arne | TU Braunschweig
    Author: Montano, Victor | University of Houston
    Author: Becker, Aaron | University of Houston
    Author: Fekete, S�ndor | Technische Universitét Braunschweig
 
    keyword: Underactuated Robots; Path Planning for Multiple Mobile Robots or Agents; Manipulation Planning

    Abstract : We present theoretical and practical methods for achieving <i>arbitrary</i> reconfiguration of a set of objects, based on the use of external forces, such as a magnetic field or gravity: Upon actuation, each object is pushed in the same direction until it collides with an obstruction. This concept can be used for a wide range of applications in which particles do not have their own energy supply. <p>A crucial challenge for achieving any desired target configuration is breaking global symmetry in a controlled fashion. Previous work made use of specifically placed barriers; however, introducing precisely located obstacles into the workspace is impractical for many scenarios. In this paper, we present a different, less intrusive method: making use of the interplay between static friction with a boundary and the external force to achieve <i>arbitrary reconfiguration</i>. Our key contributions are a precise <i>theoretical</i> characterization of the critical coefficient of friction that is sufficient for rearranging two particles in triangles, convex polygons, and regular polygons; a method for reconfiguring multiple particles in rectangular workspaces, and deriving <i>practical</i> algorithms for these rearrangements. Hardware experiments show the efficacy of these procedures, demonstrating the usefulness of this novel approach.

- Robust Capture of Unknown Objects with a Highly Under-Actuated Gripper

    Author: Glick, Paul | UCSD Bioinspired Robotics and Design Lab
    Author: Van Crey, Nikko | University of Michigan Ann Arbor
    Author: Tolley, Michael T. | University of California, San Diego
    Author: Ruffatto III, Donald | NASA Jet Propulsion Lab
 
    keyword: Underactuated Robots; Grippers and Other End-Effectors; Tendon/Wire Mechanism

    Abstract : Capturing large objects of unknown shape and orientation remains a challenge for most robotic grippers. In this letter, we present a highly under-actuated gripper well suited for this task. Prior work shows that the stability of an under-actuated linkage depends on the configuration of the links and that grippers with many links are unlikely to be stable in arbitrary configurations. We unlock the potential of highly under-actuated grippers by implementing two methods of stabilization, allowing operation on unknown surfaces. We show highly under-actuated linkages successfully grasp in many configurations and without strict stability. The gripper, capable of holding over 30 N and conforming tightly to a set of test geometries, consists of two cable-driven linkages that are each 65 cm long. Furthermore, we show this type of gripper is well suited for tasks with space and mass constraints such as satellite servicing, and outfit the gripper with a gecko-inspired adhesive to improve performance.	

- TWISTER Hand: Underactuated Robotic Gripper Inspired by Origami Twisted Tower (I)
 
    Author: Lee, Kiju | Texas A&amp;M University
    Author: Wang, Yanzhou | Johns Hopkins University
    Author: Zheng, Chuanqi | Case Western Reserve University
 
    keyword: Underactuated Robots; Grippers and Other End-Effectors; Soft Robot Applications

    Abstract : This article presents a new cable-driven underactuated robotic gripper, called TWISTER Hand. It is designed for adaptable grasping of objects in different shapes, weights, sizes, and textures. Each finger of the gripper is made of a compliant and continuum mechanism inspired by an origami design. This design is converted into a computer-aided design (CAD) model and 3-D printed using flexible and rigid polymer composite materials. Two CAD modeling methods for this design are compared in terms of structural stiffness and durability in the printed outcomes. For each design, two soft materials are used for preliminary evaluation of the material effect in these properties. The best combination of the model and material is selected to fabricate the three fingers of the robotic gripper. Each finger has a single cable routed along the structure. All three cables are tied and actuated simultaneously using a single servo motor to generate closing and opening motions in the gripper. TWISTER Hand's adaptable grasping capability is tested using 36 different objects. The robot's grasping performance under object pose uncertainties is also experimentally tested and analyzed. This compact fully integrated gripper can be attached to a robotic arm for various manipulative tasks.

## Applications
- Robust Autonomous Navigation of Unmanned Aerial Vehicles (UAVs) for Warehouses' Inventory Applications

    Author: Kwon, Woong | Samsung Electronics Co., Ltd
    Author: Park, Junho | Samsung Electronics
    Author: Lee, Minsu | Samsung Electronics
    Author: Her, Jongbeom | Samsung Electronics
    Author: Kim, Sang-Hyeon | Samsung Electronics
    Author: Seo, Ja-Won | Samsung Electronics
 
    keyword: Aerial Systems: Applications; Aerial Systems: Perception and Autonomy

    Abstract : The inventory inspection using autonomous UAVs is beneficial in terms of cost, time and safety of human workers. However, in typical warehouses, it is very challenging for the autonomous UAVs to do inventory task motions safely because aisles are narrow and long, and the illumination is poor. Prior autonomous UAVs are not suitable for such environments, since they suffer from either localization methods prone to disturbance, drift and outliers; or expensive sensors. We present a low-cost sensing system with an Extended Kalman Filter(EKF)-based multi-sensor fusion framework to achieve practical autonomous navigation of UAVs in warehouse environments. To overcome the inherent drift, outliers, and disturbance problems of na�ve UAV localization methods, we suggest 1) exploiting component test of Mahalanobis norm to reject outliers efficiently, 2) introducing pseudo-covariance to incorporate a visual SLAM algorithm, and 3) recognizing floor lanes to get absolute information - as robust data fusion methods. Exemplar results are provided to demonstrate the effectiveness of the methods. The proposed system has been successfully implemented for diverse cyclic inventory inspection tasks in a materials warehouse.

- SUMMIT: A Simulator for Urban Driving in Massive Mixed Traffic

    Author: Cai, Panpan | National University of Singapore
    Author: Lee, Yiyuan | National University of Singapore
    Author: Luo, Yuanfu | School of Computing, National University of Singapore
    Author: Hsu, David | National University of Singapore
 
    keyword: Simulation and Animation; Autonomous Vehicle Navigation; Deep Learning in Robotics and Automation

    Abstract : Autonomous driving in an unregulated urban crowd is an outstanding challenge, especially, in the presence of many aggressive, high-speed traffic participants. This paper presents SUMMIT, a high-fidelity simulator that facilitates the development and testing of crowd-driving algorithms. By leveraging the open-source OpenStreetMap map database and a heterogeneous multi-agent motion prediction model developed in our earlier work, SUMMIT simulates dense, unregulated urban traffic for heterogeneous agents at any worldwide locations that OpenStreetMap supports. SUMMIT is built as an extension of CARLA and inherits from it the physics and visual realism for autonomous driving simulation. SUMMIT supports a wide range of applications, including perception, vehicle control and planning, and end-to-end learning. We provide a context-aware planner together with benchmark scenarios and show that SUMMIT generates complex, realistic traffic behaviors in challenging crowd-driving settings.

- A Model-Based Reinforcement Learning and Correction Framework for Process Control of Robotic Wire Arc Additive Manufacturing

    Author: Dharmawan, Audelia Gumarus | Singapore University of Technology and Design
    Author: Xiong, Yi | Singapore University of Technology and Design
    Author: Foong, Shaohui | Singapore University of Technology and Design
    Author: Soh, Gim Song | Singapore University of Technology and Design
 
    keyword: Additive Manufacturing; Intelligent and Flexible Manufacturing; Industrial Robots

    Abstract : Robotic Wire Arc Additive Manufacturing (WAAM) utilizes a robot arm as a motion system to build 3D metallic objects by depositing weld beads one above the other in a layer by layer fashion. A key part of this approach is the process study and control of Multi-Layer Multi-Bead (MLMB) deposition, which is very sensitive to process parameters and prone to error stacking. Despite its importance, it has been receiving less attention than its single bead counterpart in literature, probably due to the higher experimental overhead and complexity of modeling. To address these challenges, this paper proposes an integrated learning-correction framework, adapted from Model-Based Reinforcement Learning, to iteratively learn the direct effect of process parameters on MLMB print while simultaneously correct for any inter-layer geometric digression such that the final output is still satisfactory. The advantage is that this learning architecture can be used in conjunction with actual parts printing (hence, in-situ study), thus minimizing the required training time and material wastage. The proposed learning framework is implemented on an actual robotic WAAM system and experimentally evaluated.

- Toward Optimal FDM Toolpath Planning with Monte Carlo Tree Search

    Author: Yoo, Chanyeol | University of Technology Sydney
    Author: Lensgraf, Samuel | Dartmouth College
    Author: Fitch, Robert | University of Technology Sydney
    Author: Clemon, Lee | University of Technology Sydnet
    Author: Mettu, Ramgopal | Tulane University
 
    keyword: Additive Manufacturing; Intelligent and Flexible Manufacturing; Motion and Path Planning

    Abstract : The most widely used methods for toolpath planning in 3D printing slice the input model into successive 2D layers to construct the toolpath. Unfortunately the methods can incur a substantial amount of wasted motion (i.e., the extruder is moving while not printing). In recent years we have introduced a new paradigm that characterizes the space of feasible toolpaths using a dependency graph on the input model, along with several algorithms that optimize objective functions (wasted motion or print time). A natural question that arises is, under what circumstances can we efficiently compute an optimal toolpath? In this paper, we give an algorithm for computing fused deposition modeling (FDM) toolpaths that utilizes Monte Carlo Tree Search (MCTS), a powerful general-purpose method for navigating large search spaces that is guaranteed to converge to the optimal solution. Under reasonable assumptions on printer geometry that allow us to compress the dependency graph, our MCTS-based algorithm converges to find the optimal toolpath. We validate our algorithm on a dataset of 75 models and examine the performance on MCTS against our previous best local search-based algorithm in terms of toolpath quality. We show that a relatively short time budget for MCTS yields results on par with local search, while a larger time budget yields a 15% improvement in quality over local search. Additionally, we examine the properties of the models and MCTS executions that lead to better or worse results.

- Optimizing Performance in Automation through Modular Robots

    Author: Liu, Stefan Boson | Technical University of Munich
    Author: Althoff, Matthias | Technische Universitét M�nchen
 
    keyword: Cellular and Modular Robots; Industrial Robots

    Abstract : Flexible manufacturing and automation require robots that can be adapted to changing tasks. We propose to use modular robots that are customized from given modules for a specific task. This work presents an algorithm for proposing a module composition that is optimal with respect to performance metrics such as cycle time and energy efficiency, while considering kinematic, dynamic, and obstacle constraints. Tasks are defined as trajectories in Cartesian space, as a list of poses for the robot to reach as fast as possible, or as dexterity in a desired workspace. In a simulated comparison with commercially available industrial robots, we demonstrate the superiority of our approach in randomly generated tasks with respect to the chosen performance metrics. We use our modular robot proModular.1 for the comparison.

- Towards Practical Multi-Object Manipulation Using Relational Reinforcement Learning

    Author: Li, Richard | UC Berkeley
    Author: Jabri, Allan | UC Berkeley
    Author: Agrawal, Pulkit | MIT
    Author: Darrell, Trevor | UC Berkeley
 
    keyword: Learning and Adaptive Systems

    Abstract : Learning robotic manipulation tasks using rein- forcement learning with sparse rewards is currently impractical due to the outrageous data requirements. Many practical tasks require manipulation of multiple objects, and the complexity of such tasks increases with the number of objects. Learning from a curriculum of increasingly complex tasks appears to be a natural solution, but unfortunately, does not work for many scenarios. We hypothesize that the inability of the state- of-the-art algorithms to effectively utilize a task curriculum stems from the absence of inductive biases for transferring knowledge from simpler to complex tasks. We show that graph-based relational architectures overcome this limitation and enable learning of complex tasks when provided with a simple curriculum of tasks with increasing numbers of objects. We demonstrate the utility of our framework on a simulated block stacking task. Starting from scratch, our agent learns to stack six blocks into a tower. Despite using step-wise sparse rewards, our method is orders of magnitude more data- efficient and outperforms the existing state-of-the-art method that utilizes human demonstrations. Furthermore, the learned policy exhibits zero-shot generalization, successfully stacking blocks into taller towers and previously unseen configurations such as pyramids, without any further training.

- SwarmMesh: A Distributed Data Structure for Cooperative Multi-Robot Applications

    Author: Majcherczyk, Nathalie | Worcester Polytechnic Institute
    Author: Pinciroli, Carlo | Worcester Polytechnic Institute
 
    keyword: Networked Robots; Distributed Robot Systems; Swarms

    Abstract : We present an approach to the distributed storage of data across a swarm of mobile robots that forms a shared global memory. We assume that external storage infrastructure is absent, and that each robot is capable of devoting a quota of memory and bandwidth to distributed storage. Our approach is motivated by the insight that in many applications data is collected at the periphery of a swarm topology, but the periphery also happens to be the most dangerous location for storing data, especially in exploration missions. Our approach is designed to promote data storage in the locations in the swarm that best suit a specific feature of interest in the data, while accounting for the constantly changing topology due to individual motion. We analyze two possible features of interest: the data type and the data item position in the environment. We assess the performance of our approach in a large set of simulated experiments. The evaluation shows that our approach is capable of storing quantities of data that exceed the memory of individual robots, while maintaining near-perfect data retention in high-load conditions.

## Robust and Sensor-Based Control
- Avalanche Victim Search Via Robust Observers

    Author: Mimmo, Nicola | University of Bologna
    Author: Bernard, Pauline | MINES ParisTech, Université PSL
    Author: Marconi, Lorenzo | University of Bologna
 
    keyword: Sensor-based Control; Search and Rescue Robots

    Abstract : This paper introduces a new approach for victim localization in avalanches that will be exploited by UAVs using the ARVA sensor. We show that the nominal ARVA measurement can be linearly related to a quantity that is sufficient to reconstruct the victim position. We explicitly deal with a robust scenario in which the measurement is actually perturbed by a noise that grows with the distance to the victim and we propose an adaptive control scheme made of a least-square identifier and a trajectory generator whose role is both to guarantee the persistence of excitation for the identifier and to steer the ARVA receiver towards the victim. We show that the system succeeds in localizing the victim in a domain where the ARVA output is sufficiently informative and illustrate its performance in simulation. This new approach could significantly reduce the searching time by providing an exploitable estimate before having reached the victim. The work is framed within the EU project AirBorne whose goals is to develop at TRL8 a drone for quick localization of victims in avalanche scenarios.

- Reactive Control and Metric-Topological Planning for Exploration

    Author: Ohradzansky, Michael | University of Colorado Boulder
    Author: Mills, Andrew | University of Colorado, Boulder
    Author: Rush, Eugene | University of Colorado Boulder
    Author: Riley, Danny | University of Colorado Boulder
    Author: Frew, Eric W. | University of Colorado
    Author: Humbert, James Sean | University of Colorado Boulder
 
    keyword: Sensor-based Control; Motion and Path Planning; Biologically-Inspired Robots

    Abstract : Autonomous navigation in unknown environments with the intent of exploring all traversable areas is a significant challenge for robotic platforms. In this paper, a simple yet reliable method for exploring unknown environments is presented based on bio-inspired reactive control and metric-topological planning. The reactive control algorithm is modeled after the spatial decomposition of wide and small-field patterns of optic flow in the insect visuomotor system. Centering behaviour and small obstacle detection and avoidance are achieved through wide- field integration and Fourier residual analysis of instantaneous measured nearness respectively. A topological graph is estimated using image processing techniques on a continuous occupancy grid image. Node paths are rapidly generated to navigate to the nearest unexplored edge in the graph. It is shown through rigorous field-testing that the proposed control and planning method is robust, reliable, and computationally efficient.

- Information Theoretic Active Exploration in Signed Distance Fields

    Author: Saulnier, Kelsey | University of Pennsylvania
    Author: Atanasov, Nikolay | University of California, San Diego
    Author: Pappas, George J. | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania, School of Engineering and Applied Sc
 
    keyword: Sensor-based Control; Autonomous Agents; Reactive and Sensor-Based Planning

    Abstract : This paper focuses on exploration and occupancy mapping of unknown environments using a mobile robot. While a truncated signed distance field (TSDF) is a popular, efficient, and highly accurate representation of occupancy, few works have considered optimizing robot sensing trajectories for autonomous TSDF mapping. We propose an efficient approach for maintaining TSDF uncertainty and predicting its evolution from potential future sensor measurements without actually receiving them. Efficient uncertainty prediction is critical for long-horizon optimization of potential sensing trajectories. We develop a deterministic tree-search algorithm that evaluates the information gain between the TSDF distribution and potential observations along sequences of robot motion primitives. Efficient planning is achieved by branch-and-bound pruning of uninformative sensing trajectories. The effectiveness of our active TSDF mapping approach is evaluated in several simulated environments with complex visibility constraints.

- Adaptive Integral Inverse Kinematics Control for Lightweight Compliant Manipulators

    Author: Rodr�guez de Cos, Carlos | Universidad De Sevilla
    Author: Acosta, Jose Angel | University of Seville
    Author: Ollero, Anibal | University of Seville
 
    keyword: Robust/Adaptive Control of Robotic Systems; Compliance and Impedance Control; Aerial Systems: Mechanics and Control

    Abstract : In this paper, an adaptive to unknown stiffness algorithm for controlling low-cost lightweight compliant manipulators is presented. The proposed strategy is based on the well-known transpose inverse kinematics approach, that has been enhanced with an integral action and an update law for the unknown stiffness of the compliant links, making it valid for soft materials. Moreover, the algorithm is proven to guarantee global task-space regulation of the end-effector. This approach has been implemented on a very low-cost robotic manipulator setup (comprised of 4 actuated and 3 flexible links) equipped with a simple Arduino board running at 27Hz. Notwithstanding, the strategy is capable of achieving a first-order-like response when undisturbed, and recover from overshoots provided by unforeseen impacts, smoothly returning to its nominal behaviour. Moreover, the adaptive capabilities are also used to perform contact tasks, achieving zero steady-state error. The tracking performance and disturbance rejection capabilities are demonstrated with both theoretical and experimental results.

- Bayesian Learning-Based Adaptive Control for Safety Critical Systems

    Author: Fan, David D | Georgia Institute of Technology
    Author: Nguyen, Jennifer | West Virginia University
    Author: Thakker, Rohan | Nasa's Jet Propulsion Laboratory, Caltech
    Author: Alatur, Nikhilesh Athresh | ETH Zurich
    Author: Agha-mohammadi, Ali-akbar | NASA-JPL, Caltech
    Author: Theodorou, Evangelos | Georgia Institute of Technology
 
    keyword: Robust/Adaptive Control of Systems; Robot Safety; Probability and Statistical Methods

    Abstract : Deep learning has enjoyed much recent success, and applying state-of-the-art model learning methods to controls is an exciting prospect. However, there is a strong reluctance to use these methods on safety-critical systems, which have constraints on safety, stability, and real-time performance. We propose a framework which satisfies these constraints while allowing the use of deep neural networks for learning model uncertainties. Central to our method is the use of Bayesian model learning, which provides an avenue for maintaining appropriate degrees of caution in the face of the unknown. In the proposed approach, we develop an adaptive control framework leveraging the theory of stochastic CLFs (Control Lyapunov Functions) and stochastic CBFs (Control Barrier Functions) along with tractable Bayesian model learning via Gaussian Processes or Bayesian neural networks. Under reasonable assumptions, we guarantee stability and safety while adapting to unknown dynamics with probability 1. We demonstrate this architecture for high-speed terrestrial mobility targeting potential applications in safety-critical high-speed Mars rover missions.

- A Novel Adaptive Controller for Robot Manipulators Based on Active Inference

    Author: Pezzato, Corrado | Delft University of Technology
    Author: Ferrari, Riccardo M.G. | Delft University of Technology
    Author: Hern�ndez, Carlos | Delft University of Technology
 
    keyword: Robust/Adaptive Control of Robotic Systems; Industrial Robots; Biologically-Inspired Robots

    Abstract : More adaptive controllers for robot manipulators are needed, which can deal with large model uncertainties. This paper presents a novel active inference controller (AIC) as an adaptive control scheme for industrial robots. This scheme is easily scalable to high degrees-of-freedom, and it maintains high performance even in the presence of large unmodeled dynamics. The proposed method is based on active inference, a promising neuroscientific theory of the brain, which describes a biologically plausible algorithm for perception and action. In this work, we formulate active inference from a control perspective, deriving a model-free control law which is less sensitive to unmodeled dynamics. The performance and the adaptive properties of the algorithm are compared to a state-of-the-art model reference adaptive controller (MRAC) in an experimental setup with a real 7-DOF robot arm. The results showed that the AIC outperformed the MRAC in terms of adaptability, providing a more general control law. This confirmed the relevance of active inference for robot control.

## Object Detection, Segmentation and Categorization

- Stillleben: Realistic Scene Synthesis for Deep Learning in Robotics

    Author: Schwarz, Max | University Bonn
    Author: Behnke, Sven | University of Bonn
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization; Semantic Scene Understanding

    Abstract : Training data is the key ingredient for deep learning approaches, but difficult to obtain for the specialized domains often encountered in robotics. We describe a synthesis pipeline capable of producing training data for cluttered scene perception tasks such as semantic segmentation, object detection, and correspondence or pose estimation. Our approach arranges object meshes in physically realistic, dense scenes using physics simulation. The arranged scenes are rendered using high-quality rasterization with randomized appearance and material parameters. Noise and other transformations introduced by the camera sensors are simulated. Our pipeline can be run online during training of a deep neural network, yielding applications in life-long learning and in iterative render-and-compare approaches. We demonstrate the usability by learning semantic segmentation on the challenging YCB-Video dataset without actually using any training frames, where our method achieves performance comparable to a conventionally trained model. Additionally, we show successful application in a real-world regrasping system.

- From Planes to Corners: Multi-Purpose Primitive Detection in Unorganized 3D Point Clouds

    Author: Sommer, Christiane | Technical University of Munich
    Author: Sun, Yumin | Technical University of Munich
    Author: Guibas, Leonidas | Stanford University
    Author: Cremers, Daniel | Technical University of Munich
    Author: Birdal, Tolga | Technical University of Munich
 
    keyword: Object Detection, Segmentation and Categorization; Range Sensing; Computational Geometry

    Abstract : We propose a new method for segmentation-free joint estimation of orthogonal planes, their intersection lines, relationship graph and corners lying at the intersection of three orthogonal planes. Such unified scene exploration under orthogonality allows for multitudes of applications such as semantic plane detection or local and global scan alignment, which in turn can aid robot localization or grasping tasks. Our two-stage pipeline involves a rough yet joint estimation of orthogonal planes followed by a subsequent joint refinement of plane parameters respecting their orthogonality relations. We form a graph of these primitives, paving the way to the extraction of further reliable features: lines and corners. Our experiments demonstrate the validity of our approach in numerous scenarios from wall detection to 6D tracking, both on synthetic and real data.

- Addressing the Sim2Real Gap in Robotic 3D Object Classification

    Author: Weibel, Jean-Baptiste | TU Wien
    Author: Patten, Timothy | TU Wien
    Author: Vincze, Markus | Vienna University of Technology
 
    keyword: Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation; RGB-D Perception

    Abstract : Object classification with 3D data is an essential component of any scene understanding method. It has gained significant interest in a variety of communities, most notably in robotics and computer graphics. While the advent of deep learning has progressed the field of 3D object classification, most work using this data type are solely evaluated on CAD model datasets. Consequently, current work does not address the discrepancies existing between real and artificial data. In this work, we examine this gap in a robotic context by specifically addressing the problem of classification when transferring from artificial CAD models to real reconstructed objects. This is performed by training on ModelNet (CAD models) and evaluating on ScanNet (reconstructed objects). We show that standard methods do not perform well in this task. We thus introduce a method that carefully samples object parts that are reproducible under various transformations and hence robust. Using graph convolution to classify the composed graph of parts, our method significantly improves upon the baseline.

- A Generative Approach towards Improved Robotic Detection of Marine Litter

    Author: Hong, Jungseok | University of Minnesota
    Author: Fulton, Michael | University of Minnesota
    Author: Sattar, Junaed | University of Minnesota
 
    keyword: Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation; Marine Robotics

    Abstract : This paper presents an approach to address data scarcity problems in underwater image datasets for visual detection of marine debris. The proposed approach relies on a two-stage variational autoencoder (VAE) and a binary classifier to evaluate the generated imagery for quality and realism. From the images generated by the two-stage VAE, the binary classifier selects "good quality" images and augments the given dataset with them. Lastly, a multi-class classifier is used to evaluate the impact of the augmentation process by measuring the accuracy of an object detector trained on combinations of real and generated trash images. Our results show that the classifier trained with the augmented data outperforms the one trained only with the real data. This approach will not only be valid for the underwater trash classification problem presented in this paper, but it will also be useful for any data-dependent task for which collecting more images is challenging or infeasible.

- Learning to Optimally Segment Point Clouds

    Author: Hu, Peiyun | Carnegie Mellon University
    Author: Held, David | Carnegie Mellon University
    Author: Ramanan, Deva | Carnegie Mellon University
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization; Autonomous Vehicle Navigation

    Abstract : We focus on the problem of class-agnostic instance segmentation of LiDAR point clouds. We propose an approach that combines graph-theoretic search with data-driven learning: it searches over a set of candidate segmentations and returns one where individual segments score well according to a data-driven point-based model of "objectness". We prove that if we score a segmentation by the worst objectness among its individual segments, there is an efficient algorithm that finds the optimal worst-case segmentation among an exponentially large number of candidate segmentations. We also present an efficient algorithm for the average-case. For evaluation, we repurpose KITTI 3D detection as a segmentation benchmark and empirically demonstrate that our algorithms significantly outperform past bottom-up segmentation approaches and top-down object-based algorithms on segmenting point clouds.

- CNN Based Road User Detection Using the 3D Radar Cube

    Author: Palffy, Andras | Delft University of Technology
    Author: Dong, Jiaao | Daimler Greater China Ltd
    Author: Kooij, Julian | TU Delft
    Author: Gavrila, Dariu | Delft University of Technology
 
    keyword: Object Detection, Segmentation and Categorization; Sensor Fusion; Deep Learning in Robotics and Automation

    Abstract : This paper presents a novel radar based, single-frame, multi-class detection method for moving road users (pedestrian, cyclist, car), which utilizes low-level radar cube data. The method provides class information both on the radar target- and object-level. Radar targets are classified individually after extending the target features with a cropped block of the 3D radar cube around their positions, thereby capturing the motion of moving parts in the local speed distribution. A Convolutional Neural Network (CNN) is proposed for this classification step. Afterwards, object proposals are generated with a clustering step, which not only considers the radar targets' positions and speeds, but their calculated class scores as well. In experiments on a real-life dataset we demonstrate that our method outperforms the state-of-the-art methods both target and object-wise by reaching an average of 0.70 (baseline: 0.68) target-wise and 0.56 (baseline: 0.48) object-wise<p>F1 score. Furthermore, we examine the importance of the used features in an ablation study.

- 

- PST900: RGB-Thermal Calibration, Dataset and Segmentation Network

    Author: Skandan, Shreyas | University of Pennsylvania
    Author: Rodrigues, Neil | University of Pennsylvania
    Author: Zhou, Alex | University of Pennsylvania
    Author: Miller, Ian | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania
    Author: Taylor, Camillo Jose | University of Pennsylvania
 
    keyword: Object Detection, Segmentation and Categorization; Sensor Fusion; Deep Learning in Robotics and Automation

    Abstract : In this work we propose long wave infrared (LWIR) imagery as a viable supporting modality for semantic segmentation using learning-based techniques. We first address the problem of RGB-thermal camera calibration by proposing a passive calibration target and procedure that is both portable and easy to use. Second, we present PST900, a dataset of 894 synchronized and calibrated RGB and Thermal image pairs with per pixel human annotations across four distinct classes from the DARPA Subterranean Challenge. Lastly, we propose a CNN architecture for fast semantic segmentation that combines both RGB and Thermal imagery in a way that leverages RGB imagery independently. We compare our method against the state-of-the-art and show that our method outperforms them in our dataset.

- Instance Segmentation of LiDAR Point Clouds

    Author: Zhang, Feihu | University of Oxford
    Author: Guan, Chenye | Baidu
    Author: Fang, Jin | Baidu
    Author: Bai, Song | University of Oxford
    Author: Yang, Ruigang | University of Kentucky
    Author: Torr, Philip | University of Oxford
    Author: Prisacariu, Victor | University of Oxford
 
    keyword: Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation; Computer Vision for Transportation

    Abstract : We propose a robust baseline method for instance segmentation which are specially designed for large-scale outdoor LiDAR point clouds. Our method includes a novel dense feature encoding technique, allowing the localization and segmentation of small, far-away objects, a simple but effective solution for single-shot instance prediction and effective strategies for handling severe class imbalances. Since there is no public dataset for the study of LiDAR instance segmentation, we also build a new publicly available LiDAR point cloud dataset to include both precise 3D bounding box and point-wise labels for instance segmentation, while still being about 3~20 times as large as other existing LiDAR datasets. The dataset and the source code will be published along with the paper.

- Generation of Object Candidates through Simply Looking Around

    Author: Patar, Do&#287;an | Bogazici University
    Author: Bozma, H. Isil | Bogazici University
 
    keyword: Object Detection, Segmentation and Categorization; Visual Tracking; Sensor-based Control

    Abstract : In this paper, we consider the generation of generic object candidates by a mobile robot that is endowed with a pan-tilt monocular camera. This is an important problem because these candidates serve as basis for the robot to categorize and/or recognize the objects in its surroundings. The previously proposed methods either do not have a means of enabling the robot to look around through moving its camera or do not take advantage of the temporal coherence of the video data. We present a novel approach that enables the robot to achieve both of these capabilities simultaneously. In this approach, the robot's camera movements are governed by a family of controllers whose constructions depend on the set of object candidates that have been hitherto generated, but not directly looked at. In parallel, the robot discovers the object candidates through tracking segments and determining spatio-temporally coherent ones. The advantage of the proposed approach is that while the robot can explore its surroundings by simply looking around prior to more sophisticated exploration behavior involving possibly bodily locomotion the generated object candidates turn out to be consolidated across the visual stream in comparison to single-shot methods. This is demonstrated in extensive experimental results with a robot operating indoors varying in clutter as well as outdoors.

- Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds

    Author: Engelmann, Francis | RWTH Aachen University
    Author: Kontogianni, Theodora | University of Aachen
    Author: Leibe, Bastian | RWTH Aachen University
 
    keyword: Object Detection, Segmentation and Categorization; Semantic Scene Understanding; Deep Learning in Robotics and Automation

    Abstract : In this work, we propose Dilated Point Convolutions (DPC). In a thorough ablation study, we show that the receptive field size is directly related to the performance of 3D point cloud processing tasks, including semantic segmentation and object classification. Point convolutions are widely used to efficiently process 3D data representations such as point clouds or graphs. However, we observe that the receptive field size of recent point convolutional networks is inherently limited. Our dilated point convolutions alleviate this issue, they significantly increase the receptive field size of point convolutions. Importantly, our dilation mechanism can easily be integrated into most existing point convolutional networks. To evaluate the resulting network architectures, we visualize the receptive field and report competitive scores on popular point cloud benchmarks.

- A Water-Obstacle Separation and Refinement Network for Unmanned Surface Vehicles

    Author: Bovcon, Borja | Faculty of Computer and Information Science, University of Ljubl
    Author: Kristan, Matej | University of Ljubljana
 
    keyword: Object Detection, Segmentation and Categorization; Computer Vision for Transportation; Visual-Based Navigation

    Abstract : Obstacle detection by semantic segmentation shows a great promise for autonomous navigation in unmanned surface vehicles (USV). However, existing methods suffer from poor estimation of the water edge in the presence of visual ambiguities, poor detection of small obstacles and high false-positive rate on water reflections and wakes. We propose a new deep encoder-decoder architecture, a water-obstacle separation and refinement network (WaSR), to address these issues. Detection and water edge accuracy are improved by a novel decoder that gradually fuses inertial information from IMU with the visual features from the encoder. In addition, a novel loss function is designed to increase the separation between water and obstacle features early on in the network. Subsequently, the capacity of the remaining layers in the decoder is better utilised, leading to a significant reduction in false positives and increased true positives. Experimental results show that WaSR outperforms the current state-of-the-art by a large margin, yielding a 14% increase in F-measure over the second-best method.

- Dynamic Anchor Selection for Improving Object Localization

    Author: Shyam, Pranjay | Korea Advanced Institute of Science and Technology
    Author: Yoon, Kuk-Jin | KAIST
    Author: Kim, Kyung-Soo | KAIST(Korea Advanced Institute of Science and Technology)
 
    keyword: Object Detection, Segmentation and Categorization

    Abstract : Anchor boxes acting as potential object detection candidates allow single-stage detectors to achieve realtime performance, at the cost of localization accuracy when compared to state of the art two-stage detectors. Therefore, correct selection of the scale and aspect ratio associated with an anchor box is crucial for detector performance. In this work, we propose a novel architecture (DANet) for improving the localization performance of single-stage object detectors, while maintaining real-time inference. The proposed network achieves this by predicting (1) the combination of aspect ratios and scales per feature map based on object density and (2) localization confidence per anchor box. We evaluate the proposed network using the benchmark dataset. On the MS COCO dataset, DANet achieves 30.9% AP at 51.8 fps using ResNet-18 and 45.3% AP at 7.4 fps using ResNeXt-101. The code and models will be available at https://github.com/PS06/AnchorNet.

- 3D Object Detection and Tracking Based on Streaming Data

    Author: Guo, Xusen | Sun Yet-Sen University
    Author: Gu, Jianfeng | Sun Yat-Sen University
    Author: Guo, Silu | SunYat-senUniversity
    Author: Xu, Zixiao | Sun Yat-Sen University
    Author: Yang, Chengzhang | Sun Yat-Sen University
    Author: Liu, Shanghua | Sysu
    Author: Cheng, Long | Sun Yat-Sen University
    Author: Huang, Kai | Sun Yat-Sen University
 
    keyword: Object Detection, Segmentation and Categorization; Visual Tracking; Sensor Fusion

    Abstract : Recent approaches for 3D object detection have&#12288;made tremendous progresses due to the development of deep&#12288;learning. However, previous researches are mostly based on&#12288;individual frames, leading to limited exploitation of&#12288;information between frames. In this paper, we attempt to leverage&#12288;the temporal information in streaming data and explore 3D&#12288;streaming based object detection as well as tracking. Toward&#12288;this goal, we set up a dual-way network for 3D object detection&#12288;based on keyframes, and then propagate predictions to non-key&#12288;frames through a motion based interpolation algorithm guided&#12288;by temporal information. Our framework is not only shown&#12288;to have significant improvements on object detection compared&#12288;with frame-by-frame paradigm, but also proven to produce&#12288;competitive results on KITTI Object Tracking Benchmark, with&#12288;76.68% in MOTA and 81.65% in MOTP respectively.

- Object-Centric Stereo Matching for 3D Object Detection

    Author: Pon, Alexander | University of Toronto
    Author: Ku, Jason | University of Toronto
    Author: Li, Chengyao | University of Toronto
    Author: Waslander, Steven Lake | University of Toronto
 
    keyword: Object Detection, Segmentation and Categorization; Autonomous Vehicle Navigation

    Abstract : Safe autonomous driving requires reliable 3D object detection-determining the 6 DoF pose and dimensions of objects of interest. Using stereo cameras to solve this task is a cost-effective alternative to the widely used LiDAR sensor. The current state-of-the-art for stereo 3D object detection takes the existing PSMNet stereo matching network, with no modifications, and converts the estimated disparities into a 3D point cloud, and feeds this point cloud into a LiDAR-based 3D object detector. The issue with existing stereo matching networks is that they are designed for disparity estimation, not 3D object detection; the shape and accuracy of object point clouds are not the focus. Stereo matching networks commonly suffer from inaccurate depth estimates at object boundaries, which we define as streaking, because background and foreground points are jointly estimated. Existing networks also penalize disparity instead of the estimated position of object point clouds in their loss functions. We propose a novel 2D box association and object-centric stereo matching method that only estimates the disparities of the objects of interest to address these two issues. With the open-sourced 3D object detector AVOD, we achieve state-of-the-art results on the KITTI 3D and BEV benchmarks.

- The Relative Confusion Matrix, a Tool to Assess Classifiablility in Large Scale Picking Applications

    Author: Balasch, Alexander | TGW Logistics Group
    Author: Beinhofer, Maximilian | TGW Logistics Group
    Author: Zauner, Gerald | Upper Austria University of Applied Sciences, School of Engineer
 
    keyword: Object Detection, Segmentation and Categorization; Logistics; Perception for Grasping and Manipulation

    Abstract : For bin picking robots in real logistics installations, the certainty of picking the correct product out of a mixed-product bin is essential. This paper proposes an approach for the robot to efficiently decide whether it can robustly distinguish the product to pick from the others in the bin. If not, the pick has to be routed not to the robot workstation but to a manual picking station. For this, we introduce a modified version of the confusion matrix, which we call the relative confusion matrix. We show how this matrix can be used to make the required decision, taking into account that all other products in the warehouse can be logically ruled out as they are not contained in the bin. Considering only this sub-set of products would require a re-computation of the standard confusion matrix. With the relative confusion matrix, no such re-computation is needed, which makes our approach more efficient. We show the usefulness of our approach in extensive experiments with a real bin picking robot, on simulated data, and on a publicly available image dataset.

- Pose-Guided Auto-Encoder and Feature-Based Refinement for 6-DoF Object Pose Regression

    Author: Li, Zhigang | Tsinghua University
    Author: Ji, Xiangyang | Tsinghua University
 
    keyword: Object Detection, Segmentation and Categorization; Computer Vision for Other Robotic Applications; Perception for Grasping and Manipulation

    Abstract : Accurately estimating the 6-DoF object pose from a single RGB image is a challenging task in computer vision. Though pose regression approaches have achieved great progress, the performance is still limited. In this work, we propose Pose-guided Auto-Encoder (PAE), which can distill better pose-related features from the image by utilizing a suitable pose representation, 3D Location Field (3DLF), to guide the encoding process. The features from PAE show strong robustness to pose-irrelevant factors. Compared with traditional auto-encoder, PAE can not only improve the pose estimation performance but also handle the ambiguity viewpoints problem. Further, we propose Feature-based Pose Refiner (FPR), which refines the pose from the extracted features without rendering. Combining PAE with FPR, our approach achieved state-of-the-art performance on the widely used LINEMOD dataset. Our approach not only outperforms the direct regression-based approaches with a large margin but also thrillingly surpasses current state-of-the-art indirect PnP-based approach.

- PrimiTect: Fast Continuous Hough Voting for Primitive Detection

    Author: Sommer, Christiane | Technical University of Munich
    Author: Sun, Yumin | Technical University of Munich
    Author: Bylow, Erik | Technical University of Munich
    Author: Cremers, Daniel | Technical University of Munich
 
    keyword: Object Detection, Segmentation and Categorization; Semantic Scene Understanding; Range Sensing

    Abstract : This paper tackles the problem of data     Abstraction in the context of 3D point sets. Our method classifies points into different geometric primitives, such as planes and cones, leading to a compact representation of the data. Being based on a semi-global Hough voting scheme, the method does not need initialization and is robust, accurate and efficient. We use a local, low-dimensional parameterization of primitives to determine type, shape and pose of object that a point belongs to. This makes our algorithm suitable to run on devices with low computational power, as often required in robotics applications. The evaluation shows that our method outperforms state-of-the-art methods both in terms of accuracy and robustness.

- FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-Scale Context Aggregation and Feature Space Super-Resolution

    Author: Zhang, Zhanpeng | SenseTime Group Limited
    Author: Zhang, Kaipeng | The University of Tokyo
 
    keyword: Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation; Computer Vision for Transportation

    Abstract : Real-time semantic segmentation is desirable in many robotic applications with limited computation resources. One challenge of semantic segmentation is to deal with the object scale variations and leverage the context. How to perform multi-scale context aggregation within limited computation budget is important. In this paper, firstly, we introduce a novel and efficient module called Cascaded Factorized Atrous Spatial Pyramid Pooling (CF-ASPP). It is a lightweight cascaded structure for Convolutional Neural Networks (CNNs) to efficiently leverage context information. On the other hand, for runtime efficiency, state-of-the-art methods will quickly decrease the spatial size of the inputs or feature maps in the early network stages. The final high-resolution result is usually obtained by non-parametric up-sampling operation (e.g. bilinear interpolation). Differently, we rethink this pipeline and treat it as a super-resolution process. We use optimized superresolution operation in the up-sampling step and improve the accuracy, especially in sub-sampled input image scenario for real-time applications. By fusing the above two improvements, our methods provide better latency-accuracy trade-off than the other state-of-the-art methods. In particular, we achieve 68.4% mIoU at 84 fps on the Cityscapes test set with a single Nivida Titan X (Maxwell) GPU card. The proposed module can be plugged into any feature extraction CNN and benefits from the CNN structure development.

## Aerial Systems: Perception and Autonomy
- Pose-Estimate-Based Target Tracking for Human-Guided Remote Sensor Mounting with a UAV

    Author: McArthur, Daniel | Purdue University
    Author: An, Ze | Purdue University
    Author: Cappelleri, David | Purdue University
 
    keyword: Aerial Systems: Applications; Visual-Based Navigation

    Abstract : In this paper, we present a method for pose estimate-based target tracking (PBTT) that enables the performance of autonomous aerial manipulation operations in unstructured environments using fully on-board computation for both UAV localization and target tracking. The PBTT method does not depend on extracting traditional visual features (e.g. using SIFT, SURF, ORB, etc.) on or near the target. Instead, the algorithm combines input from an RGB-D camera and the UAV's position estimator (which utilizes a downward-facing optical flow camera for horizontal localization) to track a target point selected by a human operator. The effectiveness of the PBTT method is evaluated through several autonomous flight tests performed with the Interacting-Boomcopter (I-BC) UAV platform in unstructured environments and in the presence of light wind disturbances.

- Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation

    Author: Teixeira, Lucas | ETH Zurich
    Author: Oswald, Martin R. | ETH Zurich
    Author: Pollefeys, Marc | ETH Zurich
    Author: Chli, Margarita | ETH Zurich
 
    keyword: Aerial Systems: Perception and Autonomy; Deep Learning in Robotics and Automation

    Abstract : On the pursuit of autonomous flying robots, the scientific community has been developing onboard real-time algorithms for localisation and mapping. Despite recent progress, the available solutions still lack accuracy and robustness in many aspects. While mapping for autonomous cars had a substantive boost using learning techniques to enhance LIDAR measurements using image-based depth completion, the viewpoint variations experienced by aerial vehicles are still posing challenges for learning-based mapping approaches. In this paper, we propose a depth completion and uncertainty estimation approach that better handles the challenges of aerial platforms, such as viewpoint and depth variations, and limited computing resources. The core of our method is a novel compact network that performs both depth completion and confidence estimation using an image-guided approach. Real-time performance onboard a GPU suitable for small robots is achieved by sharing deep features between both tasks. Experiments demonstrate that our network outperforms the state-of-the-art in depth completion and uncertainty estimation for single-view methods on mobile GPUs. We further present a new aerial depth completion dataset that exhibits challenging depth completion scenarios and an open-source, visual-inertial UAV simulator for photo-realistic data generation. Our results show that our network trained on this dataset can be deployed on real-world aerial public datasets without tuning or style transfer.

- EVDodgeNet: Deep Dynamic Obstacle Dodging with Event Cameras

    Author: J Sanket, Nitin | University of Maryland, College Park
    Author: Parameshwara, Chethan | University of Maryland, College Park
    Author: Singh, Chahat | University of Maryland, College Park
    Author: Kuruttukulam, Ashwin Varghese | University of Maryland
    Author: Fermuller, Cornelia | University of Maryland
    Author: Scaramuzza, Davide | University of Zurich
    Author: Aloimonos, Yiannis | University of Maryland
 
    keyword: Aerial Systems: Perception and Autonomy; Deep Learning in Robotics and Automation; Visual-Based Navigation

    Abstract : Dynamic obstacle avoidance on quadrotors requires low latency. A class of sensors that are particularly suitable for such scenarios are event cameras. In this paper, we present a deep learning based solution for dodging multiple dynamic obstacles on a quadrotor with a single event camera and on-board computation. Our approach uses a series of shallow neural networks for estimating both the ego-motion and the motion of independently moving objects. The networks are trained in simulation and directly transfer to the real world without any fine-tuning or retraining. <p>We successfully evaluate and demonstrate the proposed approach in many real-world experiments with obstacles of different shapes and sizes, achieving an overall success rate of 70% including objects of unknown shape and a low light testing scenario. To our knowledge, this is the first deep learning - based solution to the problem of dynamic obstacle avoidance using event cameras on a quadrotor. Finally, we also extend our work to the pursuit task by merely reversing the control policy, proving that our navigation stack can cater to different scenarios.

- Direct Visual-Inertial Ego-Motion Estimation Via Iterated Extended Kalman Filter

    Author: Zhong, Shangkun | City University of Hong Kong
    Author: Chirarattananon, Pakpong | City University of Hong Kong
 
    keyword: Visual-Based Navigation; Aerial Systems: Perception and Autonomy; Range Sensing

    Abstract : This paper proposes a reactive navigation strategy for recovering the altitude, translational velocity and orientation of Micro Aerial Vehicles. The main contribution lies in the direct and tight fusion of Inertial Measurement Unit (IMU) measurements with monocular feedback under an assumption of a single planar scene. An Iterated Extended Kalman Filter (IEKF) scheme is employed. The state prediction makes use of IMU readings while the state update relies directly on photometric feedback as measurements. Unlike feature-based methods, the photometric difference for the innovation term renders an inherent and robust data association process in a single step. The proposed approach is validated using real-world datasets. The results show that the proposed method offers better robustness, accuracy, and efficiency than a feature-based approach. Further investigation suggests that the accuracy of the flight velocity estimates from the proposed approach is comparable to those of two state-of-the-art Visual Inertial Systems (VINS) while the proposed framework is &#8776;15-30 times faster thanks to the omission of reconstruction and mapping.

- A Robust UAV System for Operations in a Constrained Environment

    Author: Petrlik, Matej | Czech Technical University in Prague, Faculty of Electrical Engi
    Author: Baca, Tomas | Czech Technical Univerzity in Prague
    Author: Hert, Daniel | Czech Technical University in Prague
    Author: Vrba, Matous | Faculty of Electrical Engineering, Czech Technical University In
    Author: Krajn�k, Tom� | Czech Technical University
    Author: Saska, Martin | Czech Technical University in Prague
 
    keyword: Aerial Systems: Perception and Autonomy; Search and Rescue Robots; Robotics in Hazardous Fields

    Abstract : In this letter we present an autonomous system intended for aerial monitoring, inspection and assistance in Search and Rescue (SAR) operations within a constrained workspace. The proposed system is designed for deployment in demanding real-world environments with extremely narrow passages only slightly wider than the aerial platform, and with limited visibility due to the absence of illumination and the presence of dust. The focus is on precise localization in an unknown environment, high robustness, safety and fast deployment without any need to install an external infrastructure such as an external computer and localization system. These are the main requirements of the targeted SAR scenarios. The performance of the proposed system was successfully evaluated in the Tunnel Circuit of the DARPA Subterranean Challenge, where the UAV cooperated with ground robots to precisely localize artifacts in a coal mine tunnel system. The challenge was unique due to the intention of the organizers to emulate the unpredictable conditions of a real SAR operation, in which there is no prior knowledge of the obstacles that will be encountered.

- On Training Datasets for Machine Learning-Based Visual Relative Localization of Micro-Scale UAVs

    Author: Walter, Viktor | Czech Technical University
    Author: Vrba, Matous | Faculty of Electrical Engineering, Czech Technical University In
    Author: Saska, Martin | Czech Technical University in Prague
 
    keyword: Aerial Systems: Perception and Autonomy; Multi-Robot Systems; Deep Learning in Robotics and Automation

    Abstract : By leveraging our relative Micro-scale Unmanned Aerial Vehicle localization sensor UVDAR, we generated an automatically annotated dataset MIDGARD, which the community is invited to use for training and testing their machine learning systems for the detection and localization of Micro-scale Unmanned Aerial Vehicles (MAVs) by other MAVs. Furthermore, we provide our system as a mechanism for rapidly generating custom annotated datasets specifically tailored for the needs of a given application. The recent literature is rich in applications of machine learning methods in automation and robotics. One particular subset of these methods is visual object detection and localization, using means such as Convolutional Neural Networks, which nowadays enable objects to be detected and classified with previously inconceivable precision and reliability. Most of these applications, however, rely on a carefully crafted training dataset of annotated camera footage. These must contain the objects of interest in environments similar to those where the detector is expected to operate. Notably, the positions of the objects must be provided in annotations. For non-laboratory settings, the construction of such datasets requires many man-hours of manual annotation, which is especially the case for use onboard Micro-scale Unmanned Aerial Vehicles. In this paper, we are providing for the community a practical alternative to that kind of approach.

- Fast Frontier-Based Information-Driven Autonomous Exploration with an MAV

    Author: Dai, Anna | ETH Zurich
    Author: Papatheodorou, Sotiris | Imperial College London
    Author: Funk, Nils | Imperial College London
    Author: Tzoumanikas, Dimos | Imperial College London
    Author: Leutenegger, Stefan | Imperial College London
 
    keyword: Aerial Systems: Perception and Autonomy; Visual-Based Navigation

    Abstract : Exploration and collision-free navigation through an unknown environment is a fundamental task for autonomous robots. In this paper, a novel exploration strategy for Micro Aerial Vehicles (MAVs) is presented. The goal of the exploration strategy is the reduction of map entropy regarding occupancy probabilities, which is reflected in a utility function to be maximised. We achieve fast and efficient exploration performance with tight integration between our octree-based occupancy mapping approach, frontier extraction, and motion planning--as a hybrid between frontier-based and sampling-based exploration methods. The computationally expensive frontier clustering employed in classic frontier-based exploration is avoided by exploiting the implicit grouping of frontier voxels in the underlying octree map representation. Candidate next-views are sampled from the map frontiers and are evaluated using a utility function combining map entropy and travel time, where the former is computed efficiently using sparse raycasting. These optimisations along with the targeted exploration of frontier-based methods result in a fast and computationally efficient exploration planner. The proposed method is evaluated using both simulated and real-world experiments, demonstrating clear advantages over state-of-the-art approaches.

- Dynamic Landing of an Autonomous Quadrotor on a Moving Platform in Turbulent Wind Conditions

    Author: Paris, Aleix | Massachusetts Institute of Technology
    Author: Lopez, Brett Thomas | Massachusetts Institute of Technology
    Author: How, Jonathan Patrick | Massachusetts Institute of Technology
 
    keyword: Aerial Systems: Perception and Autonomy; Aerial Systems: Mechanics and Control

    Abstract : Autonomous landing on a moving platform presents unique challenges for multirotor vehicles, including the need to accurately localize the platform, fast trajectory planning, and precise/robust control. Previous works studied this problem but most lack explicit consideration of the wind disturbance, which typically leads to slow descents onto the platform. This work presents a fully autonomous vision-based system that addresses these limitations by tightly coupling the localization, planning, and control, thereby enabling fast and accurate landing on a moving platform. The platform's position, orientation, and velocity are estimated by an extended Kalman filter using simulated GPS measurements when the quadrotor-platform distance is large, and by a visual fiducial system when the platform is nearby. The landing trajectory is computed online using receding horizon control and is followed by a boundary layer sliding controller that provides tracking performance guarantees in the presence of unknown, but bounded, disturbances. To improve the performance, the characteristics of the turbulent conditions are accounted for in the controller. The landing trajectory is fast, direct, and does not require hovering over the platform, as is typical of most state-of-the-art approaches. Simulations and hardware experiments are presented to validate the robustness of the approach.

- Cross-Drone Binocular Coordination for Ground Moving Target Tracking in Occlusion-Rich Scenarios

    Author: Chang, Yuan | National University of Defense Technology
    Author: Zhou, Han | National University of Defense Technology
    Author: Wang, Xiangke | National University of Defense Technology
    Author: Shen, Lincheng | National University of Defense Technology
    Author: Hu, Tianjiang | Sun Yat-Sen University
 
    keyword: Aerial Systems: Perception and Autonomy; Multi-Robot Systems; Visual Tracking

    Abstract : How to work effectively under occlusion-rich environments remains a challenge for airborne vision-based ground target tracking, due to the natural limitation of monocular vision. Given this, a novel cross-drone binocular coordination approach, inspired by the efficient coordination of human eyes, is proposed and developed. The idea, derived from neural models of the human visual system, is to utilize distributed target measurements to overcome occlusion effects. Eventually, a binocular coordination controller is developed. It enables two distributed pan-tilt cameras to execute synergistic movements similar to human eyes. The proposed approach is able to work based on binocular or monocular vision, and hence it is practically appropriate for various environments. Both testbed experiments and field experiments are conducted for performance evaluation. Testbed experiments highlight its advantages over independent tracking in terms of accuracy while being robust to a partial perception ratio of up to 43%. Field experiments with a pair of drones further demonstrate its effectiveness in the real-world scenarios.

- Direct NMPC for Post-Stall Motion Planning with Fixed-Wing UAVs

    Author: Basescu, Max | Johns Hopkins University Applied Physics Lab
    Author: Moore, Joseph | Johns Hopkins University Applied Physics Lab
 
    keyword: Aerial Systems: Perception and Autonomy; Aerial Systems: Mechanics and Control; Optimization and Optimal Control

    Abstract : Fixed-wing unmanned aerial vehicles (UAVs) offer significant performance advantages over rotary-wing UAVs in terms of speed, endurance, and efficiency. However, these vehicles have traditionally been severely limited with regards to maneuverability. In this paper, we present a nonlinear control approach for enabling aerobatic fixed-wing UAVs to maneuver in constrained spaces. Our approach utilizes full-state direct trajectory optimization and a minimalistic, but representative, nonlinear aircraft model to plan aggressive fixed-wing trajectories in real-time at 5 Hz across high angles-of-attack. Randomized motion planning is used to avoid local minima and local-linear feedback is used to compensate for model inaccuracies between updates. We demonstrate our method in hardware and show that both local-linear feedback and re-planning are necessary for successful navigation of a complex environment in the presence of model uncertainty.

-  IMU-Based Inertia Estimation for a Quadrotor Using Newton-Euler Dynamics

    Author: Svacha, James | University of Pennsylvania
    Author: Paulos, James | University of Pennsylvania
    Author: Loianno, Giuseppe | New York University
    Author: Kumar, Vijay | University of Pennsylvania


- A Flight Envelope Determination and Protection System for Fixed-Wing UAVs

    Author: Zogopoulos-Papaliakos, Georgios | National Technical University of Athens
    Author: Kyriakopoulos, Kostas | National Technical Univ. of Athens
 
    keyword: Aerial Systems: Perception and Autonomy; Motion Control; Optimization and Optimal Control

    Abstract : In this work we present a novel, approximate, efficient algorithm for determining the Trim Flight Envelope of a fixed-wing UAV, based on a generic, nonlinear numerical model. The resulting Flight Envelope is expressed as a convex intersection of half-spaces. Subsequently, a Model Predictive Controller (MPC) is designed which takes into account the Flight Envelope constraints, to avoid Loss-of-Control. The overall system is shown to operate in real-time in a simulation environment.

- AU-AIR: A Multi-Modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance

    Author: Bozcan, &#304;lker | Middle East Technical University
    Author: Kayacan, Erdal | Aarhus University
 
    keyword: Aerial Systems: Perception and Autonomy; Object Detection, Segmentation and Categorization; Aerial Systems: Applications

    Abstract : Unmanned aerial vehicles (UAVs) with mounted cameras have the advantage of capturing aerial (bird-view) images. The availability of aerial visual data and the recent advances in object detection algorithms led the computer vision community to focus on object detection tasks on aerial images. As a result of this, several aerial datasets have been introduced, including visual data with object annotations. UAVs are used solely as flying-cameras in these datasets, discarding different data types regarding the flight (e.g., time, location, internal sensors). In this work, we propose a multi-purpose aerial dataset (AU-AIR) that has multi-modal sensor data (i.e., visual, time, location, altitude, IMU, velocity) collected in real-world outdoor environments. The AU-AIR dataset includes meta-data for extracted frames (i.e., bounding box annotations for trafficrelated object category) from recorded RGB videos. Moreover, we emphasize the differences between natural and aerial images in the context of object detection task. For this end, we train and test mobile object detectors (including YOLOv3- Tiny and MobileNetv2-SSDLite) on the AU-AIR dataset, which are applicable for real-time object detection using on-board computers with UAVs. Since our dataset has diversity in recorded data types, it contributes to filling the gap between computer vision and robotics. The dataset is available at https://bozcani.github.io/auairdataset.

- Design and Autonomous Stabilization of a Ballistically Launched Multirotor

    Author: Bouman, Amanda | Caltech
    Author: Nadan, Paul | Olin College
    Author: Anderson, Matthew | Jet Propulsion Laboratory
    Author: Pastor, Daniel | Caltech
    Author: Izraelevitz, Jacob | NASA Jet Propulsion Laboratory
    Author: Burdick, Joel | California Institute of Technology
    Author: Kennedy, Brett | Jet Propulsion Laboratory
 
    keyword: Aerial Systems: Perception and Autonomy; Aerial Systems: Applications

    Abstract : Aircraft that can launch ballistically and convert to autonomous, free-flying drones have applications in many areas such as emergency response, defense, and space exploration, where they can gather critical situational data using onboard sensors. This paper presents a ballistically-launched, autonomously-stabilizing multirotor prototype (SQUID - Streamlined Quick Unfolding Investigation Drone) with an onboard sensor suite, autonomy pipeline, and passive aerodynamic stability. We demonstrate autonomous transition from passive to vision-based, active stabilization, confirming the multirotor's ability to autonomously stabilize after a ballistic launch in a GPS-denied environment.

- Asynchronous Event-Based Clustering and Tracking for Intrusion Monitoring in UAS

    Author: Rodriguez-Gomez, Juan Pablo | University of Seville
    Author: G�mez Egu�luz, Augusto | University of Seville
    Author: Martinez-de-Dios, Jose Ramiro | University of Seville
    Author: Ollero, Anibal | University of Seville
 
    keyword: Aerial Systems: Perception and Autonomy; Aerial Systems: Applications; Computer Vision for Other Robotic Applications

    Abstract : Automatic surveillance and monitoring using Unmanned Aerial Systems (UAS) require the development of perception systems that robustly work under different illumination conditions. Event cameras are neuromorphic sensors that capture the illumination changes in the scene with very low latency and high dynamic range. Although recent advances in event-based vision have explored the use of event cameras onboard UAS, most techniques group events in frames and, therefore, do not fully exploit the sequential and asynchronous nature of the event stream. This paper proposes a fully asynchronous scheme for intruder monitoring using UAS. It employs efficient event clustering and feature tracking modules and includes a sampling mechanism to cope with the computational cost of event-by-event processing adapting to on-board hardware computational constraints. The proposed scheme was tested on a real multirotor in challenging scenarios showing significant accuracy and robustness to lighting conditions.

- SHIFT: Selective Heading Image for Translation, an Onboard Monocular Optical Flow Estimator for Fast Constantly Rotating UAVs

    Author: Ng, Matthew | Singapore University of Technology and Design
    Author: Tang, Emmanuel | Singapore University of Technology &amp; Design
    Author: Soh, Gim Song | Singapore University of Technology and Design
    Author: Foong, Shaohui | Singapore University of Technology and Design
 
    keyword: Aerial Systems: Perception and Autonomy; Aerial Systems: Applications; Computer Vision for Other Robotic Applications

    Abstract : Pose estimation is of paramount importance for flight control as well as localization and navigation of Unmanned Aerial Vehicles (UAVs) to enable autonomous operations. In environments without GPS, such estimation can only be determined using onboard sensors; optical flow using a monocular camera is a popular approach. Monocopters are a class of nature inspired UAVs known as free rotors where their design and flight dynamics are inspired by the falling samara seed. With a constantly rotating body frame, free rotors introduces some unique challenges for visual perception required during optical flow sensing. This paper addresses these problems with the introduction of SHIFT (Selective Heading Image for Translation) that selects optimal images for determining translation with optical flow. It achieves this by decoupling rotation vectors about the optical axis from translation vectors in a flow field through the separate tracking of orientation and position using an Unscented Kalman Filter with phase correlation in the log-polar and spatial domain. The experiments show that SHIFT's estimation in orientation is stable even under sinusoidal excitation with a median absolute percentage errors of less than 1%. It is able to track position and orientation of a UAV accurately.

- Flydar: Magnetometer-Based High Angular Rate Estimation During Gyro Saturation for SLAM

    Author: Tan, Chee How | Singapore University of Technology &amp; Design
    Author: Sufiyan, Danial | Singapore University of Technology &amp; Design
    Author: Tang, Emmanuel | Singapore University of Technology &amp; Design
    Author: Khaw, Jien-Yi | Singapore University of Technology &amp; Design
    Author: Soh, Gim Song | Singapore University of Technology and Design
    Author: Foong, Shaohui | Singapore University of Technology and Design
 
    keyword: Aerial Systems: Perception and Autonomy; Range Sensing; Sensor Fusion

    Abstract : In this paper, the high angular rate estimation for simultaneous localisation and mapping (SLAM) of a Flying Li-DAR (Flydar) is presented. The proposed EKF-based algorithm exploits the sinusoidal magnetometer measurement generated by the continuously rotating airframe for estimation of the robot hovering angular velocity. Significantly, the proposed method does not rely on additional sensors other than existing IMU sensors already being used for flight stabilization. The gyro measurement and the gyro bias are incorporated as a control input and a filter state respectively to enable estimation even under gyro saturation. Additionally, this work proposes leveraging on the inherently rotating locomotion to generate a planar lidar scan using only a single-point laser for possible lightweight autonomy. The proposed estimation method was experimentally evaluated on a ground rotating rig up to twice the gyro saturation limit with an effective rms error of 0.0045Hz; and on the proposed aerial platform - Flydar - hovering beyond the saturation limit with a rms error of 0.0056Hz. Lastly, the proposed method for SLAM using the rotating dynamics of Flydar was demonstrated with a localisation accuracy of 0.11m.

- Nonlinear MPC with Motor Failure Identification and Recovery for Safe and Aggressive Multicopter Flight

    Author: Tzoumanikas, Dimos | Imperial College London
    Author: Yan, Qingyue | Imperial College London
    Author: Leutenegger, Stefan | Imperial College London
 
    keyword: Aerial Systems: Perception and Autonomy

    Abstract : Safe and precise reference tracking is a crucial characteristic of Micro Aerial Vehicles (MAVs) that have to operate under the influence of external disturbances in cluttered environments. In this paper, we present a Model Predictive Controller (MPC) that exploits the fully physics based non-linear dynamics of the system. We furthermore show how the control inputs can be transformed into feasible actuator commands. In order to guarantee safe operation despite potential loss of a motor under which we show our system keeps operating safely, we developed an ac{EKF} based motor failure identification algorithm. We verify the effectiveness of the developed pipeline in flight experiments with and without motor failures.

## Autonomous Vehicle Navigation

- Autonomous Navigation in Inclement Weather Based on a Localizing Ground Penetrating Radar

    Author: Ort, Teddy | Massachusetts Institute of Technology
    Author: Gilitschenski, Igor | Massachusetts Institute of Technology
    Author: Rus, Daniela | MIT
 
    keyword: Autonomous Vehicle Navigation; Wheeled Robots; Intelligent Transportation Systems

    Abstract : Most autonomous driving solutions require some method of localization within their environment. The GPS has not been widely adopted for autonomous driving because it is neither sufficiently precise, nor robust. Instead, state-of-the art autonomous driving systems use onboard sensors such as cameras or LiDAR to localize the robot precisely in a previously recorded map. However, these solutions are sensitive to ambient lighting conditions such as darkness and inclement weather. Additionally the maps can become outdated in a rapidly changing environment and require continuous updating. While LiDAR systems don't require visible light, they are sensitive to weather such as fog, or snow, which can interfere with localization. In this paper, we utilize a Ground Penetrating Radar (GPR) sensor to obtain precise vehicle localization. By mapping and localizing using features beneath the ground, we obtain features that are both stable over time, and maintain their appearance during changing ambient weather and lighting conditions. We incorporate this solution into a full-scale autonomous vehicle and evaluate the performance on over 17 km of testing data in a variety of challenging weather conditions. We find that this novel sensing modality is capable of providing precise localization for autonomous navigation without using cameras or LiDAR sensors.

- Robot Navigation in Crowds by Graph Convolutional Networks with Attention Learned from Human Gaze

    Author: Chen, Yuying | Hong Kong University of Science and Technology
    Author: Liu, Congcong | Hong Kong University of Science and Technology
    Author: Shi, Bertram Emil | Hong Kong University of Science and Technology
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Autonomous Vehicle Navigation; Social Human-Robot Interaction; Deep Learning in Robotics and Automation

    Abstract : Safe and efficient crowd navigation for mobile robot is a crucial yet challenging task. Previous work has shown the power of deep reinforcement learning frameworks to train efficient policies. However, their performance deteriorates when the crowd size grows. We suggest that this can be addressed by enabling the network to identify and pay attention to the humans in the crowd that are most critical to navigation. We propose a novel network utilizing a graph representation to learn the policy. We first train a graph convolutional network based on human gaze data that accurately predicts human attention to different agents in the crowd as they perform a navigation task based on a top down view of the environment. We incorporate the learned attention into a graph-based reinforcement learning architecture. The proposed attention mechanism enables the assignment of meaningful weightings to the neighbors of the robot, and has the additional benefit of interpretability. Experiments on real-world dense pedestrian datasets with various crowd sizes demonstrate that our model outperforms state-of-art methods, increasing the task completion rate by 18.4% and decreasing navigation time by 16.4%.

- Wall Deadlock Evasion Control Based on Rotation Radius Adjustment

    Author: Kojima, Shotaro | Tohoku University
    Author: Ohno, Kazunori | Tohoku University
    Author: Suzuki, Takahiro | Tohoku University
    Author: Okada, Yoshito | Tohoku University
    Author: Westfechtel, Thomas | Tohoku University
    Author: Tadokoro, Satoshi | Tohoku University
 
    keyword: Autonomous Vehicle Navigation; Motion Control; Dynamics

    Abstract : This paper describes a wall deadlock evasion method for tracked vehicles. Wall deadlock is a phenomenon where the robot cannot rotate to the commanded direction when it collides with a wall, because the motion is restricted by the wall. The key idea behind solving this problem involves an adjustment of the rotation radius to generate sufficient rotational moment. There are several approaches to generate a rotational moment; however, no previous solution has been established to address this problem by adjusting the rotation radius based on the dynamics of wall deadlock. In this paper, the     Authors propose a new wall deadlock evasion method based on the sufficient rotation radius estimation. Experimental results show that the robot can generate rotational motion that satisfies conditions expected by the model. The wall deadlock evasion method is implemented and shows improved performance in terms of reproducibility of motion compared with the different approach proposed in our previous work. Wall deadlock evasion provides more choices of motion such as being as close to the obstacles as possible and ensures that the robot can continue locomotion after such motion. By handling wall deadlock, the robots can utilize surrounding walls for motion in situations such as relative positioning or driving in fixed lanes.

- Socially-Aware Reactive Obstacle Avoidance Strategy Based on Limit Cycle

    Author: Boldrer, Manuel | University of Trento
    Author: Andreetto, Marco | University of Trento
    Author: Divan, Stefano | University of Trento
    Author: Palopoli, Luigi | University of Trento
    Author: Fontanelli, Daniele | University of Trento
 
    keyword: Autonomous Vehicle Navigation; Collision Avoidance; Sensor-based Control

    Abstract :  The paper proposes a combination of ideas to support navigation for a<p> mobile robot across dynamic environments, cluttered with obstacles and populated by human beings. The combination of the classical potential field methods and limit cycle based approach with an innovative shape for the limit cycles, generates paths which are reasonably short, smooth and comfortable to follow (which can be very important for assistive robots), and it respects the safety and psychological comfort of the by-standers by staying clear of their private space.

- Multi-Head Attention for Multi-Modal Joint Vehicle Motion Forecasting

    Author: Mercat, Jean | 1991
    Author: Gilles, Thomas | Polytechnique
    Author: El Zoghby, Nicole | RENAULT
    Author: Sandou, Guillaume | SUPELEC
    Author: Dominique, Beauvois | Sup�lec
    Author: Guillermo, Pita Gil | Renault
 
    keyword: Intelligent Transportation Systems; Deep Learning in Robotics and Automation; Motion and Path Planning

    Abstract : This paper presents a novel vehicle motion forecasting method based on multi-head attention. It produces joint forecasts for all vehicles on a road scene as sequences of multi-modal probability density functions of their positions. Its architecture uses multi-head attention to account for complete interactions between all vehicles, and long short-term memory layers for encoding and forecasting. It relies solely on vehicle position tracks, does not need maneuver definitions, and does not represent the scene with a spatial grid. This allows it to be more versatile than similar model while combining many forecasting capabilities, namely joint forecast with interactions, uncertainty estimation, and multi-modality. The resulting prediction likelihood outperforms state-of-the-art models on the same dataset.

- Temporal Information Integration for Video Semantic Segmentation

    Author: Guarino, Guillaume | INSA Strasbourg
    Author: Chateau, Thierry | Clermont Auvergne University
    Author: Teuliere, Celine | Institut Pascal, Clermont Auvergne University
    Author: Antoine, Violaine | Clermont Auvergne University, LIMOS
 
    keyword: Autonomous Vehicle Navigation; Semantic Scene Understanding

    Abstract : We present a temporal Bayesian filter for semantic segmentation of a video sequence. Each pixel is a random variable following a discrete probabilistic distribution function representing possible semantic classes (eg. road, pedestrian, traffic sign,... for autonomous driving applications). Bayesian filtering consists in two main steps: 1) a prediction model and 2) an observation model (likelihood). We propose to use a data-driven prediction function derived from a dense optical flow between images t and t+1 achieved by a deep neural network. Moreover, the observation function uses a semantic segmentation network. The resulting approach is evaluated on the public dataset Cityscapes. We show that using the temporal filtering increases the accuracy of the semantic segmentation.

- Map-Predictive Motion Planning in Unknown Environments

    Author: Elhafsi, Amine | Stanford University
    Author: Ivanovic, Boris | Stanford University
    Author: Janson, Lucas | Harvard University
    Author: Pavone, Marco | Stanford University
 
    keyword: Autonomous Vehicle Navigation; Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : Algorithms for motion planning in unknown environments are generally limited in their ability to reason about the structure of the unobserved environment. As such, current methods generally navigate unknown environments by relying on heuristic methods to choose intermediate objectives along frontiers. We present a unified method that combines map prediction and motion planning for safe, time-efficient autonomous navigation of unknown environments by dynamically-constrained robots. We propose a data-driven method for predicting the map of the unobserved environment, using the robot's observations of its surroundings as context. These map predictions are then used to plan trajectories from the robot's position to the goal without requiring frontier selection. We applied this map-predictive motion planning strategy to randomly generated winding hallway environments, yielding substantial improvement in trajectory duration over a naive frontier pursuit method. We also experimentally demonstrate similar performance to methods using more sophisticated frontier selection heuristics while significantly reducing computation time.

- Using Multiple Short Hops for Multicopter Navigation with Only Inertial Sensors

    Author: Wu, Xiangyu | University of California, Berkeley
    Author: Mueller, Mark Wilfried | University of California, Berkeley
 
    keyword: Autonomous Vehicle Navigation; Aerial Systems: Applications; Localization

    Abstract : In certain challenging environments, such as inside buildings on fire, the main sensors (e.g. cameras, LiDARs and GPS systems) used for multicopter localization can become unavailable. Direct integration of the inertial navigation sensors (the accelerometer and rate gyroscope), is however unaffected by external disturbances, but the rapid error accumulation quickly makes a naive application of such a strategy feasible only for very short durations. In this work we propose a motion strategy for reducing the inertial navigation state estimation error of multicopters. The proposed strategy breaks a long duration flight into multiple short duration hops between which the vehicle remains stationary on the ground. When the vehicle is stationary, zero-velocity pseudo-measurements are introduced to an extended Kalman Filter to reduce the state estimation error. We perform experiments for closed-loop control of a multicopter for evaluation. The mean absolute position estimation error was 3.4% over a total flight distance of 5m in the experiments. The results showed a 80% reduction compared to the standard inertial navigation method without using this strategy. In addition, an additional experiment with total flight distance of 10m is conducted to demonstrate the ability of this method to navigate a multicopter in real-world environment. The final trajectory tracking error was 3% of the total flight distance.

- An Efficient and Continuous Approach to Information-Theoretic Exploration

    Author: Henderson, Theia | Massachusetts Institute of Technology
    Author: Sze, Vivienne | Massachusetts Institute of Technology
    Author: Karaman, Sertac | Massachusetts Institute of Technology
 
    keyword: Autonomous Vehicle Navigation; Motion and Path Planning; Search and Rescue Robots

    Abstract : Exploration of unknown environments is embedded and essential in many robotics applications. Traditional algorithms, that decide where to explore by computing the expected information gain of an incomplete map from future sensor measurements, are limited to very powerful computational platforms. In this paper, we describe a novel approach for computing this expected information gain efficiently, as principally derived via mutual information. The key idea behind the proposed approach is a continuous occupancy map framework and the recursive structure it reveals. This structure makes it possible to compute the expected information gain of sensor measurements across an entire map much faster than computing each measurements' expected gain independently. Specifically, for an occupancy map composed of |M| cells and a range sensor that emits |T| measurement beams, the algorithm (titled FCMI) computes the information gain corresponding to measurements made at each cell in O(|T||M|) steps. To the best of our knowledge, this complexity bound is better than all existing methods for computing information gain. In our experiments, we observe that this novel, continuous approach is two orders of magnitude faster than the state-of-the-art FSMI algorithm.

- A Feature-Based Underwater Path Planning Approach Using Multiple Perspective Prior Maps

    Author: Cagara, Daniel | Queensland University of Technology
    Author: Dunbabin, Matthew | Queensland University of Technology
    Author: Rigby, Paul | Australian Institute of Marine Science
 
    keyword: Autonomous Vehicle Navigation; Visual-Based Navigation; Marine Robotics

    Abstract : This paper presents a path planning methodology which enables Autonomous Underwater Vehicles (AUVs) to navigate in shallow complex environments such as coral reefs. The approach leverages prior information from an aerial photographic survey, and derived bathymetric information of the corresponding area. From these prior maps, a set of features is obtained which define an expected arrangement of objects and bathymetry likely to be perceived by the AUV when underwater. A navigation graph is then constructed by predicting the arrangement of features visible from a set of test points within the prior, which allows the calculation of the shortest paths from any pair of start and destination points. A maximum likelihood function is defined which allows the AUV to match its observations to the navigation graph as it undertakes its mission. To improve robustness, the history of observed features are retained to facilitate possible recovery from non-detectable or misclassified objects. The approach is evaluated using a photo-realistic simulated environment, and results illustrate the merits of the approach even when only a relatively small number of features can be identified from the prior map.

- Automatic LiDAR-Camera Calibration of Extrinsic Parameters Using a Spherical Target

    Author: T�th, Tekla | E�tv's Lor�nd University
    Author: Pusztai, Zolt�n | E�tv's Lorand University
    Author: Hajder, Levente | E�tv's Lor�nd University
 
    keyword: Autonomous Vehicle Navigation; Object Detection, Segmentation and Categorization; Sensor Fusion

    Abstract : This paper investigates a novel calibration process of devices with different modalities, which is a critical step of computer vision applications. We propose a fully automatic extrinsic calibration of a LiDAR-camera system. Our approach applies sphere as their surfaces and contours can be accurately detected on point clouds and camera images, respectively. Experiments on synthetic and real data exhibits that our automatic algorithm is fast and robust and it yields accurate camera and LiDAR extrinsic parameters.

## Mapping
- A Unified Framework for Piecewise Semantic Reconstruction in Dynamic Scenes Via Exploiting Superpixel Relations

    Author: Di, Yan | Tsinghua University
    Author: Morimitsu, Henrique | Tsinghua University
    Author: Lou, ZhiQiang | Tsinghua University
    Author: Ji, Xiangyang | Tsinghua University
 
    keyword: Mapping; SLAM; Localization

    Abstract : This paper presents a novel framework for dense piecewise semantic reconstruction in dynamic scenes containing complex background and moving objects via exploiting superpixel relations. We utilize two kinds of superpixel relations: motion relations and spatial relations, each having three subcategories: coplanar, hinge and crack. Spatial relations provide constraints on the spatial locations of neighboring superpixels and thus can be used to reconstruct dynamic scenes. However, spatial relations can not be estimated directly with epipolar geometry due to moving objects in dynamic scenes. We synthesize the results of semantic instance segmentation and motion relations to estimate spatial relations. Given consecutive frames, we mainly develop our method in five main stages: preprocessing, motion estimation, superpixel relation analysis, reconstruction and refinement. Extensive experiments on various datasets demonstrate that our method outperforms competitors in reconstruction quality. Furthermore, our method presents a feasible way to incorporate semantic information in Structure-from-Motion (SFM) based reconstruction pipelines.

- Keyframe-Based Dense Mapping with the Graph of View-Dependent Local Maps

    Author: Belter, Dominik | Poznan University of Technology
    Author: Zieli&#324;ski, Krzysztof | Institute of Control, Robotics and Information Engineering, Pozn
 
    keyword: Mapping; Range Sensing; RGB-D Perception

    Abstract : In this article, we propose a new keyframe-based mapping system. The proposed method updates local Normal Distribution Transform maps (NDT) using data from an RGB-D sensor. The cells of the NDT are stored in 2D view-dependent structures to better utilize the properties and uncertainty model of RGB-D cameras. This method naturally represents an object closer to the camera origin with higher precision. The local maps are stored in the pose graph which allows correcting global map after loop closure detection. We also propose a procedure that allows merging and filtering local maps to obtain a global map of the environment. Finally, we compare our method with Octomap and NDT-OM and provide example applications of the proposed mapping method.

- Informative Path Planning for Active Mapping under Localization Uncertainty

    Author: Popovic, Marija | Imperial College London
    Author: Vidal-Calleja, Teresa A. | University of Technology Sydney
    Author: Chung, Jen Jen | Eidgen�ssische Technische Hochschule Zurich
    Author: Nieto, Juan | ETH Zurich
    Author: Siegwart, Roland | ETH Zurich
 
    keyword: Mapping; Planning, Scheduling and Coordination; Motion and Path Planning

    Abstract : Information gathering algorithms play a key role in unlocking the potential of robots for efficient data collection in a wide range of applications. However, most existing strategies neglect the fundamental problem of the robot pose uncertainty, which is an implicit requirement for creating robust, high-quality maps. To address this issue, we introduce an informative planning framework for active mapping that explicitly accounts for the pose uncertainty in both the mapping and planning tasks. Our strategy exploits a Gaussian Process (GP) model to capture a target environmental field given the uncertainty on its inputs. For planning, we formulate a new utility function that couples the localization and field mapping objectives in GP-based mapping scenarios in a principled way, without relying on manually-tuned parameters. Extensive simulations show that our approach outperforms existing strategies, reducing mean pose uncertainty and map error. We present a proof of concept in an indoor temperature mapping scenario.

- Ensemble of Sparse Gaussian Process Experts for Implicit Surface Mapping with Streaming Data

    Author: Stork, Johannes A. | Orebro University
    Author: Stoyanov, Todor | Örebro University
 
    keyword: Mapping; Range Sensing; Learning and Adaptive Systems

    Abstract : Creating maps is an essential task in robotics and provides the basis for effective planning and navigation. In this paper, we learn a compact and continuous implicit surface map of an environment from a stream of range data with known poses.For this, we create and incrementally adjust an ensemble of approximate Gaussian process (GP) experts which are each responsible for a different part of the map. Instead of inserting all arriving data into the GP models, we greedily trade-off between model complexity and prediction error. Our algorithm therefore uses less resources on areas with few geometric features and more where the environment is rich in variety. We evaluate our approach on synthetic and real-world data sets and analyze sensitivity to parameters and measurement noise. The results show that we can learn compact and accurate implicit surface models under different conditions, with a performance comparable to or better than that of exact GP regression with subsampled data.

- Robust Method for Removing Dynamic Objects from Point Clouds

    Author: Pagad, Shishir | Nio
    Author: Agarwal, Divya | NIO Automotive, Purdue University
    Author: Kasturi Rangan, Sathya Narayanan | NIO
    Author: Kim, Hyungjin | NIO USA Inc
    Author: Yalla, Ganesh | Capella Space
 
    keyword: Mapping; SLAM; Object Detection, Segmentation and Categorization

    Abstract : 3D point cloud maps are an accumulation of laser scans obtained at different positions and times. Since laser scans represent a snapshot of the surrounding at the time of capture, they often contain moving objects which may not be observed at all times. Dynamic objects in point cloud maps decrease the quality of maps and affect localization accuracy, hence it is important to remove the dynamic objects from 3D point cloud maps. In this paper, we present a robust method to remove dynamic objects from 3D point cloud maps. Given a registered set of 3D point clouds, we build an occupancy map in which the voxels represent the occupancy state of the volume of space over an extended time period. After building the occupancy map, we use it as a filter to remove dynamic points in lidar scans before adding the points to the map. Furthermore, we accelerate the process of building occupancy maps using object detection and a novel voxel traversal method. Once the occupancy map is built, dynamic object removal can run in real-time. Our approach works well on wide urban roads with stopped or moving traffic and the occupancy maps get better with the inclusion of more lidar scans from the same scene.

- Skeleton-Based Conditionally Independent Gaussian Process Implicit Surfaces for Fusion in Sparse to Dense 3D Reconstruction

    Author: Wu, Lan | University of Technology Sydney
    Author: Falque, Raphael | University of Technology Sydney
    Author: Perez-Puchalt, Victor | EPFL
    Author: Liu, Liyang | University of Technology Sydney
    Author: Pietroni, Nico | University of Technology Sydney
    Author: Vidal-Calleja, Teresa A. | University of Technology Sydney
 
    keyword: Mapping; RGB-D Perception

    Abstract : 3D object reconstructions obtained from 2D or 3D cameras are typically noisy. Probabilistic algorithms are suitable for information fusion and can deal with noise robustly. Consequently, these algorithms can be useful for accurate surface reconstruction. This paper presents an approach to estimate a probabilistic representation of the implicit surface of 3D objects. One of the contributions of the paper is the pipeline for generating an accurate reconstruction, given a set of sparse points that are close to the surface and a dense noisy point cloud. A novel submapping method following the topology of the object is proposed to generate conditional independent Gaussian Process Implicit Surfaces. This allows inference and fusion mechanisms to be performed in parallel followed by information propagation through the submaps. Large datasets can efficiently be processed by the proposed pipeline producing not only a surface but also the uncertainty information of the reconstruction. We evaluate the performance of our algorithm using simulated and real datasets.

- Motion Estimation in Occupancy Grid Maps in Stationary Settings Using Recurrent Neural Networks

    Author: Schreiber, Marcel | Ulm University
    Author: Belagiannis, Vasileios | Universitét Ulm
    Author: Glaeser, Claudius | Robert Bosch GmbH
    Author: Dietmayer, Klaus | University of Ulm
 
    keyword: Mapping; Intelligent Transportation Systems; Deep Learning in Robotics and Automation

    Abstract : In this work, we tackle the problem of modeling the vehicle environment as dynamic occupancy grid map in complex urban scenarios using recurrent neural networks. Dynamic occupancy grid maps represent the scene in a bird's eye view, where each grid cell contains the occupancy probability and the two dimensional velocity. As input data, our approach relies on measurement grid maps, which contain occupancy probabilities, generated with lidar measurements. Given this configuration, we propose a recurrent neural network architecture to predict a dynamic occupancy grid map, i.e. filtered occupancy and velocity of each cell, by using a sequence of measurement grid maps. Our network architecture contains convolutional long-short term memories in order to sequentially process the input, makes use of spatial context, and captures motion. In the evaluation, we quantify improvements in estimating the velocity of braking and turning vehicles compared to the state-of-the-art. Additionally, we demonstrate that our approach provides more consistent velocity estimates for dynamic objects, as well as, less erroneous velocity estimates in static area.

- A Divide and Conquer Method for 3D Registration of Inhomogeneous, Partially Overlapping Scans with Fourier Mellin SOFT (FMS)

    Author: Buelow, Heiko | Jacobs University
    Author: Mueller, Christian Atanas | Jacobs University
    Author: Gomez Chavez, Arturo | Jacobs University Bremen GGmbH
    Author: Buda, Frederike | Jacobs University
    Author: Birk, Andreas | Jacobs University
 
    keyword: Mapping; Big Data in Robotics and Automation

    Abstract : High-end laser range-finders provide accurate 3D data over long ranges. But their scans are inhomogeneous, i.e., the environment is non-uniformly sampled, as there is denser data in the near range than in the far range. Furthermore, the generation of a scan is time-consuming. Thus, it is desirable to cover an area by as few scans as possible, i.e., scanning is more time-efficient if the overlap between scans is as small as possible. However, these factors pose significant challenges for state-of-the-art registration algorithms. In this work, we present a divide-and-conquer method that uses an efficient strategy to check for possible registrations between partitions of two scans. As underlying registration method, Fourier-Mellin-SOFT (FMS) is used. FMS is quite robust against partial overlaps but its performance is significantly boosted by the presented partitioning method. As concrete use case, results from the digitization of a WWII submarine bunker as a large-scale cultural heritage site are presented.

- Estimating Motion Uncertainty with Bayesian ICP

    Author: Afzal Maken, Fahira | The University of Sydney
    Author: Ramos, Fabio | University of Sydney, NVIDIA
    Author: Ott, Lionel | University of Sydney
 
    keyword: Mapping

    Abstract : Estimating the uncertainty associated with the pose transformation between two 3D point clouds is critical for autonomous navigation, grasping, and data fusion. Iterative closest point (ICP) is widely used to estimate the transformation between point cloud pairs by iteratively performing data association and motion estimation. Despite its success and popularity, ICP is effectively a deterministic algorithm, and attempts to formulate it in a probabilistic manner generally do not model all sources of uncertainty, such as data association errors and sensor noise. This leads to overconfident transformation estimates, potentially compromising the robustness of the system. In this paper we propose a novel method to estimate pose uncertainty in ICP with a Markov Chain Monte Carlo (MCMC) algorithm. Our method combines recent developments in optimization such as stochastic gradient Langevin dynamics (SGLD) and scalable Bayesian sampling to infer a full posterior distribution of the pose transformation given two point clouds and a prior distribution. We call this method Bayesian ICP. Experiments using 3D Kinect data shows that our method is capable of estimating pose uncertainty accurately, taking into account data association uncertainty as reflected by the shape of the objects.

- Actively Mapping Industrial Structures with Information Gain-Based Planning on a Quadruped Robot

    Author: Wang, Yiduo | University of Oxford
    Author: Ramezani, Milad | University of Oxford
    Author: Fallon, Maurice | University of Oxford
 
    keyword: Mapping; Legged Robots; Motion and Path Planning

    Abstract : In this paper, we develop an online active mapping system to enable a quadruped robot to autonomously survey large physical structures. We describe the perception, planning and control modules needed to scan and reconstruct an object of interest, without requiring a prior model. The system builds a voxel representation of the object, and iteratively determines the Next-Best-View (NBV) to extend the representation, according to both the reconstruction itself and to avoid collisions with the environment. By computing the expected information gain of a set of candidate scan locations sampled on the as-sensed terrain map, as well as the cost of reaching these candidates, the robot decides the NBV for further exploration. The robot plans an optimal path towards the NBV, avoiding obstacles and un-traversable terrain. Experimental results on both simulated and real-world environments show the capability and efficiency of our system. Finally we present a full system demonstration on the real robot, the ANYbotics ANYmal, autonomously reconstructing a building facade and an industrial structure.

- Efficient Covisibility-Based Image Matching for Large-Scale SfM

    Author: Ye, Zhichao | Zhejiang University
    Author: Zhang, Guofeng | Zhejiang University
    Author: Bao, Hujun | Zhejiang University
 
    keyword: Mapping; Visual Tracking

    Abstract : Obtaining accurate and sufficient feature matches is crucial for robust large-scale Structure-from-Motion. For unordered image collections, a traditional feature matching method with geometric verification requires a huge cost to find sufficient feature matches. Although several methods have been proposed to speed up this stage, none of them makes full use of existing matches. In this paper, we propose a novel efficient image matching method by using the transitivity of region covisibility. The overlapping image pairs can be efficiently found in an iterative matching strategy even only with few inlier feauture matches. The experimental results on unordered image datasets demonstrate that the proposed method is three times faster than the state-of-the-art and the matching result is high-quality enough for robust SfM.

- Probabilistic TSDF Fusion Using Bayesian Deep Learning for Dense 3D Reconstruction with a Single RGB Camera

    Author: Kim, Hanjun | Seoul National University
    Author: Lee, Beom-Hee | Seoul National University
 
    keyword: Mapping; Visual Learning; SLAM

    Abstract : In this paper, we address a 3D reconstruction problem using depth prediction from a single RGB image. Thanks to the recent advances in deep learning, depth prediction shows high performance. However, due to the gap between training environment and test environment, 3D reconstruction can be vulnerable to uncertainty of depth prediction. To consider uncertainty of depth prediction for robust 3D reconstruction, we adopt Bayesian deep learning framework. Conventional Bayesian deep learning requires a large amount of time and GPU memory to perform Monte Carlo sampling. To address this problem, we propose a lightweight Bayesian neural network consisting of U-net structure and summation-based skip connections, which is performed in real-time. Estimated uncertainty is utilized in probabilistic TSDF fusion for dense 3D reconstruction by maximizing the posterior of TSDF value per voxel. As a result, global TSDF robust to erroneous depth values can be obtained and then dense 3D reconstruction from the global TSDF is achievable more accurately. To evaluate the performance of depth prediction and 3D reconstruction using our method, we utilized two official datasets and demonstrated the outperformance of the proposed method over other conventional methods.

- A Volumetric Albedo Framework for 3D Imaging Sonar Reconstruction

    Author: Westman, Eric | Carnegie Mellon University
    Author: Gkioulekas, Ioannis | Carnegie Mellon University
    Author: Kaess, Michael | Carnegie Mellon University
 
    keyword: Mapping; Marine Robotics; Field Robots

    Abstract : In this work, we present a novel framework for object-level 3D underwater reconstruction using imaging sonar sensors. We demonstrate that imaging sonar reconstruction is analogous to the problem of confocal non-line-of-sight (NLOS) reconstruction. Drawing upon this connection, we formulate the problem as one of solving for volumetric albedo, wherein the scene of interest is modeled as a directionless albedo field. After discretization, reconstruction reduces to a convex linear optimization problem, which we can augment with a variety of priors and regularization terms. We show how to solve the resulting regularized problems using the alternating direction method of multipliers (ADMM) algorithm. We demonstrate the effectiveness of the proposed approach in simulation and on real-world datasets collected in a controlled, test tank environment with several different sonar elevation apertures.

- Map Management Approach for SLAM in Large-Scale Indoor and Outdoor Areas

    Author: Ehlers, Simon F. G. | Leibniz University Hannover
    Author: Stuede, Marvin | Leibniz University Hannover, Institute of Mechatronic Systems
    Author: Nuelle, Kathrin | Leibniz Universitét Hannover
    Author: Ortmaier, Tobias | Leibniz University Hanover
 
    keyword: Mapping; SLAM; Field Robots

    Abstract : This work presents a semantic map management approach for various environments by triggering multiple maps with different simultaneous localization and mapping (SLAM) configurations. A modular map structure allows to add, modify or delete maps without influencing other maps of different areas. The hierarchy level of our algorithm is above the utilized SLAM method. Evaluating laser scan data (e.g. the detection of passing a doorway) triggers a new map, automatically choosing the appropriate SLAM configuration from a manually predefined list. Single independent maps are connected by link-points, which are located in an overlapping zone of both maps, enabling global navigation over several maps. Loop-closures between maps are detected by an appearance-based method, using feature matching and iterative closest point (ICP) registration between point clouds. Based on the arrangement of maps and link-points, a topological graph is extracted for navigation purpose and tracking the global robot's position over several maps. Our approach is evaluated by mapping a university campus with multiple indoor and outdoor areas and     Abstracting a metrical-topological graph. It is compared to a single map running with different SLAM configurations. Our approach enhances the overall map quality compared to the single map approaches by automatically choosing predefined SLAM configurations for different environmental setups.

- A Hierarchical Framework for Collaborative Probabilistic Semantic Mapping

    Author: Yue, Yufeng | Nanyang Technological University
    Author: Zhao, Chunyang | Nanyang Technological University
    Author: Li, Ruilin | Nanyang Technological University
    Author: Yang, Chule | Nanyang Technological University
    Author: Zhang, Jun | Nanyang Technological University
    Author: Wen, Mingxing | Nanyang Technological University
    Author: Wang, Yuanzhe | Nanyang Technological University
    Author: Wang, Danwei | Nanyang Technological University
 
    keyword: Mapping; Semantic Scene Understanding; Cooperating Robots

    Abstract : Performing collaborative semantic mapping is a critical challenge for cooperative robots to maintain a comprehensive contextual understanding of the surroundings. Most of the existing work either focus on single robot semantic mapping or collaborative geometry mapping. In this paper, a novel hierarchical collaborative probabilistic semantic mapping framework is proposed, where the problem is formulated in a distributed setting. The key novelty of this work is the mathematical modeling of the overall collaborative semantic mapping problem and the derivation of its probability decomposition. In the single robot level, the semantic point cloud is obtained based on heterogeneous sensor fusion model and is used to generate local semantic maps. Since the voxel correspondence is unknown in collaborative robots level, an Expectation-Maximization approach is proposed to estimate the hidden data association, where Bayesian rule is applied to perform semantic and occupancy probability update.	The experimental results show the high quality global semantic map, demonstrating the accuracy and utility of 3D semantic map fusion algorithm in real missions.

- Autonomous Navigation in Unknown Environments Using Sparse Kernel-Based Occupancy Mapping

    Author: Duong, Thai | University of California, San Diego
    Author: Das, Nikhil | UCSD
    Author: Yip, Michael C. | University of California, San Diego
    Author: Atanasov, Nikolay | University of California, San Diego
 
    keyword: Mapping; Autonomous Vehicle Navigation; Collision Avoidance

    Abstract : This paper focuses on real-time occupancy mapping and collision checking onboard an autonomous robot navigating in an unknown environment. We propose a new map representation, in which occupied and free space are separated by the decision boundary of a kernel perceptron classifier. We develop an online training algorithm that maintains a very sparse set of support vectors to represent obstacle boundaries in configuration space. We also derive conditions that allow complete (without sampling) collision-checking for piecewise-linear and piecewise-polynomial robot trajectories. We demonstrate the effectiveness of our mapping and collision checking algorithms for autonomous navigation of an Ackermann-drive robot in unknown environments.

- Hybrid Topological and 3D Dense Mapping through Autonomous Exploration for Large Indoor Environments

    Author: Gomez, Clara | University Carlos III of Madrid
    Author: Fehr, Marius | ETH Zurich
    Author: Millane, Alexander James | ETH Zurich
    Author: Hernandez Silva, Alejandra Carolina | University Carlos III of Madrid
    Author: Nieto, Juan | ETH Zurich
    Author: Barber, Ramon | Universidad Carlos III of Madrid
    Author: Siegwart, Roland | ETH Zurich
 
    keyword: Mapping; Motion and Path Planning; Autonomous Agents

    Abstract : Robots require a detailed understanding of the 3D structure of the environment for autonomous navigation and path planning. A popular approach is to represent the environment using metric, dense 3D maps such as 3D occupancy grids. However, in large environments the computational power required for most state-of-the-art 3D dense mapping systems is compromising precision and real-time capability. In this work, we propose a novel mapping method that is able to build and maintain 3D dense representations for large indoor environments using standard CPUs. Topological global representations and 3D dense submaps are maintained as hybrid global map. Submaps are generated for every new visited place. A place (room) is identified as an isolated part of the environment connected to other parts through transit areas (doors). This semantic partitioning of the environment allows for a more efficient mapping and path-planning. We also propose a method for autonomous exploration that directly builds the hybrid representation in real time. We validate the real-time performance of our hybrid system on simulated and real environments regarding mapping and path-planning. The improvement in execution time and memory requirements upholds the contribution of the proposed work.

- Resolving Marker Pose Ambiguity by Robust Rotation Averaging with Clique Constraints

    Author: Ch'ng, Shin-Fang | The University of Adelaide
    Author: Sogi, Naoya | University of Tsukuba
    Author: Purkait, Pulak | The University of Adelaide
    Author: Chin, Tat-Jun | The University of Adelaide
    Author: Fukui, Kazuhiro | Tsukuba University
 
    keyword: Mapping; Localization; SLAM

    Abstract : Planar markers are useful in robotics and computer vision for mapping and localisation. Given a detected marker in an image, a frequent task is to estimate the 6DOF pose of the marker relative to the camera, which is an instance of planar pose estimation (PPE). Although there are mature techniques, PPE suffers from a fundamental ambiguity problem, in that there can be more than one plausible pose solutions for a PPE instance. Especially when localisation of the marker corners is noisy, it is often difficult to disambiguate the pose solutions based on reprojection error alone. Previous methods choose between the possible solutions using a heuristic criterion, or simply ignore ambiguous markers.<p>We propose to resolve the ambiguities by examining the consistencies of a set of markers across multiple views. Our specific contributions include a novel rotation averaging formulation that incorporates long-range dependencies between possible marker orientation solutions that arise from PPE ambiguities. We analyse the combinatorial complexity of the problem, and develop a novel lifted algorithm to effectively resolve marker pose ambiguities, without discarding any marker observations. Results on real and synthetic data show that our method is able to handle highly ambiguous inputs, and provides more accurate and/or complete marker-based mapping and localisation.



## Computer Vision for Other Robotic Applications

- Real-Time Semantic Stereo Matching

    Author: Dovesi, Pier | KTH, Univrses
    Author: Poggi, Matteo | University of Bologna
    Author: Andraghetti, Lorenzo | Univrses
    Author: Mart� i Rabad�n, Miquel | KTH Royal University of Technology, Univrses AB
    Author: Kjellstrom, Hedvig | KTH
    Author: Pieropan, Alessandro | KTH
    Author: Mattoccia, Stefano | University of Bologna
 
    keyword: Computer Vision for Transportation; Computer Vision for Other Robotic Applications; Computer Vision for Automation

    Abstract : Scene understanding is paramount in robotics, self-navigation, augmented reality, and many other fields. To fully accomplish this task, an autonomous agent has to infer the 3D structure of the sensed scene (to know where it looks at) and its content (to know what it sees). To tackle the two tasks, deep neural networks trained to infer semantic segmentation and depth from stereo images are often the preferred choices. Specifically, Semantic Stereo Matching can be tackled by either standalone models trained for the two tasks independently or joint end-to-end architectures. Nonetheless, as proposed so far, both solutions are inefficient because requiring two forward passes in the former case or due to the complexity of a single network in the latter, although jointly tackling both tasks is usually beneficial in terms of accuracy. In this paper, we propose a single compact and lightweight architecture for real-time semantic stereo matching. Our framework relies on coarse-to-fine estimations in a multi-stage fashion, allowing: i) very fast inference even on embedded devices, with marginal drops in accuracy, compared to state-of-the-art networks, ii) trade accuracy for speed, according to the specific application requirements. Experimental results on high-end GPUs as well as on an embedded Jetson TX2 confirm the superiority of semantic stereo matching compared to standalone tasks and highlight the versatility of our framework on any hardware and for any application.

- Multi-Task Learning for Single Image Depth Estimation and Segmentation Based on Unsupervised Network

    Author: Lu, Yawen | Rochester Institute of Technology
    Author: Sarkis, Michel | Qualcomm Technologies Inc
    Author: Lu, Guoyu | Rochester Institute of Technology
 
    keyword: Computer Vision for Transportation; Visual-Based Navigation; Autonomous Vehicle Navigation

    Abstract : Deep neural networks have significantly enhanced the performance of various computer vision tasks, including single image depth estimation and image segmentation. However, most existing approaches handle them in supervised manners and require a large number of ground truth labels that consume extensive human efforts and are not always available in real scenarios. In this paper, we propose a novel framework to estimate disparity maps and segment images simultaneously by jointly training an encoder-decoder-based interactive convolutional neural network (CNN) for single image depth estimation and a multiple class CNN for image segmentation. Learning the neural network for one task can be beneficial from simultaneously learning from another one under a multi-task learning framework. We show that our proposed model can learn per-pixel depth regression and segmentation from just a single image input. Extensive experiments on available public datasets, including KITTI, Cityscapes urban, and PASCAL-VOC demonstrate the effectiveness of our model compared with other state-of-the-art methods for both tasks.

- Learning Transformable and Plannable Se(3) Features for Scene Imitation of a Mobile Service Robot

    Author: Park, J. hyeon | Seoul National University
    Author: Kim, Jigang | Seoul National University
    Author: Jang, YoungSeok | Seoul National University
    Author: Jang, Inkyu | Seoul National University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Computer Vision for Other Robotic Applications; Deep Learning in Robotics and Automation; Learning from Demonstration

    Abstract : Deep neural networks facilitate visuosensory inputs for robotic systems. However, the features encoded in a network without specific constraints have little physical meaning. In this research, we add constraints on the network so that the trained features are forced to represent the actual twist coordinates of interactive objects in a scene. The trained coordinates describe 6d-pose of the objects, and SE(3) transformation is applied to change the coordinate system. This algorithm is developed for a mobile service robot that imitates an object-oriented task by watching human demonstrations. As the robot has mobility, the video demonstrations are collected from the different view points. Our feature trajectories of twist coordinates are synthesized in the global coordinate after SE(3) transformation is applied according to robot localization. Then, the trajectories are trained as probabilistic model and imitated by the robot with geometric dynamics of se(3). Our main contribution is to develop a trainable robot with visually demonstrated human performances. Additionally, our algorithmic contribution is to design a scene interpretation network where se(3) constraints are incorporated to estimate 6d-pose of objects.

- Multimodal Multispectral Imaging System for Small UAVs

    Author: Haavardsholm, Trym Vegard | Norwegian Defence Research Establishment (FFI)
    Author: Skauli, Torbj�rn | Forsvarets Forskningsinstitutt (FFI)
    Author: Stahl, Annette | Norwegian University of Science and Technology (NTNU)
 
    keyword: Computer Vision for Other Robotic Applications; Mapping; Surveillance Systems

    Abstract : Multispectral imaging is an attractive sensing modality for small unmanned aerial vehicles (UAVs) in numerous applications. The most compact spectral camera architecture is based on spectral filters in the focal plane. Vehicle movement can be used to scan the scene using multiple bandpass filters arranged perpendicular to the flight direction. With known camera trajectory and scene structure, it is possible to assemble a spectral image in software.<p>In this paper, we demonstrate the feasibility of a novel concept for low-cost wide area multispectral imaging with integrated spectral consistency testing. Six bandpass filters are arranged in a periodically repeating pattern.</p><p>Since different bands are recorded at different times and in different viewing directions, there is a risk of obtaining spectral artifacts in the image. We exploit the repeated sampling of bands to enable spectral consistency testing, which leads to significantly improved spectral integrity. In addition, an unfiltered region permits conventional 2D video imaging that can be used for image-based navigation and 3D reconstruction.</p><p>The proposed multimodal imaging system was tested on a UAV in a realistic experiment. The results demonstrate that spectral reconstruction and consistency testing can be performed by image processing alone, based on visual simultaneous localization and mapping (VSLAM).

- Unseen Salient Object Discovery for Monocular Robot Vision

    Author: Chan, Darren | University of California, San Diego
    Author: Riek, Laurel D. | University of California San Diego
 
    keyword: Computer Vision for Other Robotic Applications; Object Detection, Segmentation and Categorization

    Abstract : A key challenge in robotics is the capability to perceive unseen objects, which can improve a robot's ability to learn from and adapt to its surroundings. One approach is to employ unsupervised, salient object discovery methods, which have shown promise in the computer vision literature. However, most state-of-the-art methods are unsuitable for robotics because they are limited to processing whole video segments before discovering objects, which can constrain real-time perception. To address these gaps, we introduce Unsupervised Foraging of Objects (UFO), a novel, unsupervised, salient object discovery method designed for monocular robot vision. We designed UFO with a parallel discover-prediction paradigm, permitting it to discover arbitrary, salient objects on a frame-by-frame basis, which can help robots to engage in scalable object learning. We compared UFO to the two fastest and most accurate methods for unsupervised salient object discovery (Fast Segmentation and Saliency-Aware Geodesic), and show that UFO 6.5 times faster, achieving state-of-the-art precision, recall, and accuracy. Furthermore our evaluation suggests that UFO is robust to real-world perception challenges encountered by robots, including moving cameras and moving objects, motion blur, and occlusion. It is our goal that this work will be used with other robot perception methods, to design robots that can learn novel object concepts, leading to improved autonomy.

- CorsNet: 3D Point Cloud Registration by Deep Neural Network

    Author: Kurobe, Akiyoshi | Keio University
    Author: Sekikawa, Yusuke | Denso IT Laboratory
    Author: Ishikawa, Kohta | Denso IT Laboratory, Inc
    Author: Saito, Hideo | Keio University
 
    keyword: Computer Vision for Other Robotic Applications; Deep Learning in Robotics and Automation; Perception for Grasping and Manipulation

    Abstract : Point cloud registration is a key problem for robotics and computer vision communities. This represents estimating a rigid transform which aligns one point cloud to another. Iterative closest point (ICP) is a well-known classical method for this problem. ICP generally achieves high alignment only when the source and template point cloud are mostly pre-aligned. If each point cloud is far away or contains a repeating structure, the registration often fails because of being fallen into a local minimum. Recently, inspired by PointNet, several deep learning-based methods have been developed. PointNetLK is a representative approach, which directly optimizes the distance of aggregated features using gradient method by Jacobian. In this paper, we propose CorsNet: Point Cloud Registration based on Deep Learning. Since CorsNet concatenates the local features with the global features and regresses correspondences between point clouds, not directly pose or aggregated features, more useful information is integrated than the conventional approaches. For comparison, we developed the simplest baseline approach (DirectNet) which directly regresses the pose between point clouds. Through our experiments, we show that CorsNet achieves higher accuracy than not only the classic ICP method but also the recently proposed learning-based proposal PointNetLK and DirectNet, including on seen and unseen categories.

- Anticipating the Start of User Interaction for Service Robot in the Wild

    Author: Ito, Koichiro | Hitcahi, Ltd
    Author: Kong, Quan | Hitachi, Ltd
    Author: Horiguchi, Shota | Hitachi, Ltd
    Author: Sumiyoshi, Takashi | Hitachi, Ltd., Reseach &amp; Development Group
    Author: Nagamatsu, Kenji | Hitachi, Ltd
 
    keyword: Computer Vision for Other Robotic Applications; Service Robots; Deep Learning in Robotics and Automation

    Abstract : A service robot is expected to provide proactive service for visitors who require its help. In contrast to passive service, e.g., providing service only after being spoken to, proactive service initiates an interaction at an early stage, e.g., talking to potential visitors who need the robot's help in advance. This paper addresses how to anticipate the start of user interaction. We propose an approach using only a single RGB camera that anticipates whether a visitor will come to the robot for interaction or just pass it by. In the proposed approach, we (i) utilize the visitor's pose information from captured images incorporating facial information, (ii) train a CNN-LSTM--based model in an end-to-end manner with an exponential loss for early anticipation, and (iii) during the training, the network branch for facial keypoints acquired as the part of the human pose information is taught to mimic the branch trained with the face image from a specialized face detector with a human verification. By virtue of (iii), at the inference, we can run our model in an embedded system processing only the pose information without an additional face detector and typical accuracy drop. We evaluated the proposed approach on our collected real world data with a real service robot and publicly available JPL interaction dataset and found that it achieved accurate anticipation performance.

- Spin Detection in Robotic Table Tennis

    Author: Tebbe, Jonas | University of Tübingen
    Author: Klamt, Lukas | University of Tübingen
    Author: Gao, Yapeng | University of Tuebingen
    Author: Zell, Andreas | University of Tübingen
 
    keyword: Computer Vision for Other Robotic Applications; Object Detection, Segmentation and Categorization; Deep Learning in Robotics and Automation

    Abstract : In table tennis, the rotation (spin) of the ball plays a crucial role. A table tennis match will feature a variety of strokes. Each generates different amounts and types of spin. To develop a robot that can compete with a human player, the robot needs to detect spin, so it can plan an appropriate return stroke. In this paper we compare three methods to estimate spin. The first two approaches use a high-speed camera that captures the ball in flight at a frame rate of 380 Hz. This camera allows the movement of the circular brand logo printed on the ball to be seen. The first approach uses background difference to determine the position of the logo. In a second alternative, we train a CNN to predict the orientation of the logo. The third method evaluates the trajectory of the ball and derives the rotation from the effect of the Magnus force. This method gives the highest accuracy and is used for a demonstration. Our robot successfully copes with different spin types in a real table tennis rally against a human opponent.

- Look, Listen, and Act: Towards Audio-Visual Embodied Navigation

    Author: Gan, Chuang | IBM
    Author: Zhang, Yiwei | Tsinghua University
    Author: Wu, Jiajun | Stanford University
    Author: Gong, Boqing | Tencent AI Lab, Seattle
    Author: Tenenbaum, Joshua | Massachusetts Institute of Technology
 
    keyword: Computer Vision for Other Robotic Applications

    Abstract : A crucial ability of mobile intelligent agents is to integrate the evidence from multiple sensory inputs in an environment and to make a sequence of actions to reach their goals. In this paper, we attempt to approach the problem of Audio-Visual Embodied Navigation, the task of planning the shortest path from a random starting location in a scene to the sound source in an indoor environment, given only raw egocentric visual and audio sensory data. To accomplish this task, the agent is required to learn from various modalities, i.e., relating the audio signal to the visual environment. Here we describe an approach to audio-visual embodied navigation that takes advantage of both visual and audio pieces of evidence. Our solution is based on three key ideas: a visual perception mapper module that constructs its spatial memory of the environment, a sound perception module that infers the relative location of the sound source from the agent, and a dynamic path planner that plans a sequence of actions based on the audio-visual observations and the spatial memory of the environment to navigate toward the goal. Experimental results on a newly collected Visual-Audio-Room dataset using the simulated multi-modal environment demonstrate the effectiveness of our approach over several competitive baselines.

- Autonomous Tool Construction with Gated Graph Neural Network

    Author: Yang, Chenjie | Xi'an Jiaotong University
    Author: Lan, Xuguang | Xi'an Jiaotong University
    Author: Zhang, Hanbo | Xi'an Jiaotong University
    Author: Zheng, Nanning | Xi'an Jiaotong University
 
    keyword: Computer Vision for Other Robotic Applications; Semantic Scene Understanding; Perception for Grasping and Manipulation

    Abstract : Autonomous tool construction is a significant but challenging task in robotics. This task can be interpreted as when given a reference tool, selecting some available candidate parts to reconstruct it. Most of the existing works perform tool construction in the form of action part and grasp part, which is only a specific construction pattern and limits its application to some extent. In general scenarios, a tool can be constructed in various patterns with different part pairs. Therefore, whether a part pair is most suitable for constructing the tool depends not only on itself, but on other parts in the same scene. To solve this problem, we construct a Gated Graph Neural Network (GGNN) to model the relations between all part pairs, so that we can select the candidate parts in consideration of the global information. Afterwards, we embed the constructed GGNN into a RCNN-like structure to finally accomplish tool construction. The whole model will be named Tool Construction Graph RCNN (TC-GRCNN). In addition, we develop a mechanism that can generate large-scale training and testing data in simulation environments, by which we can save the time of data collection and annotation. Finally, the proposed model is deployed on the physical robot. The experiment results show that TC-GRCNN can perform well in the general scenarios of tool construction.

- Training-Set Distillation for Real-Time UAV Object Tracking

    Author: Li, Fan | Tongji University
    Author: Fu, Changhong | Tongji University
    Author: Lin, Fuling | Tongji University
    Author: Li, Yiming | Tongji University
    Author: Lu, Peng | The Hong Kong Polytechnic University
 
    keyword: Computer Vision for Other Robotic Applications; Visual Learning; Aerial Systems: Applications

    Abstract : Correlation filter (CF) has recently exhibited promising performance in visual object tracking for unmanned aerial vehicle (UAV). Such online learning method heavily depends on the quality of the training-set, yet complicated aerial scenarios like occlusion or out of view can reduce its reliability. In this work, a novel time slot-based distillation approach is proposed to efficiently and effectively optimize the training-set's quality on the fly. A cooperative energy minimization function is established to score the historical samples adaptively. To accelerate the scoring process, frames with high confident tracking results are employed as the keyframes to divide the tracking process into multiple time slots. After the establishment of a new slot, the weighted fusion of the previous samples generates one key-sample, in order to reduce the number of samples to be scored. Besides, when the current time slot exceeds the maximum frame number, which can be scored, the sample with the lowest score will be discarded. Consequently, the training-set can be efficiently and reliably distilled. Comprehensive tests on two well-known UAV benchmarks prove the effectiveness of our method with real-time speed on single CPU.

- CNN-Based Simultaneous Dehazing and Depth Estimation

    Author: Lee, Byeong-Uk | KAIST
    Author: Lee, Kyunghyun | KAIST
    Author: Oh, Jean | Carnegie Mellon University
    Author: Kweon, In So | KAIST
 
    keyword: Computer Vision for Other Robotic Applications; AI-Based Methods; Visual Learning

    Abstract : It is difficult for both cameras and depth sensors to obtain reliable information in hazy scenes. Therefore, image dehazing is still one of the most challenging problems to solve in computer vision and robotics. With the development of convolutional neural networks (CNNs), lots of dehazing and depth estimation algorithms using CNNs have emerged. However, very few of those try to solve these two problems at the same time. Focusing on the fact that traditional haze modeling contains depth information in its formula, we propose a CNN-based simultaneous dehazing and depth estimation network. Our network aims to estimate both a dehazed image and a fully scaled depth map from a single hazy RGB input with end-to-end training. The network contains a single dense encoder and four separate decoders; each of them shares the encoded image representation while performing individual tasks. We suggest a novel depth-transmission consistency loss in the training scheme to fully utilize the correlation between the depth information and transmission map. To demonstrate the robustness and effectiveness of our algorithm, we performed various ablation studies and compared our results to those of state-of-the-art algorithms in dehazing and single image depth estimation, both qualitatively and quantitatively. Furthermore, we show the generality of our network by applying it to some real-world examples.

- IF-Net: An Illumination-Invariant Feature Network

    Author: Chen, Po-Heng | National Chiao-Tung University
    Author: Luo, Zhao Xu | National Chiao Tung University
    Author: Huang, Tsu-Kuan | National Chiao Tung University
    Author: Yang, Chun | National Chiao-Tung University
    Author: Chen, Kuan-Wen | National Chiao Tung University
 
    keyword: Computer Vision for Other Robotic Applications; Visual Learning; AI-Based Methods

    Abstract : Feature descriptor matching is a critical step is many computer vision applications such as image stitching, image retrieval and visual localization. However, it is often affected by many practical factors which will degrade its performance. Among these factors, illumination variations are the most influential one, and especially no previous descriptor learning works focus on dealing with this problem. In this paper, we propose IF-Net, aimed to generate a robust and generic descriptor under crucial illumination changes conditions. We find out not only the kind of training data important but also the order it is presented. To this end, we investigate several dataset scheduling methods and propose a separation training scheme to improve the matching accuracy. Further, we propose a ROI loss and hard-positive mining strategy along with the training scheme, which can strengthen the ability of generated descriptor dealing with large illumination change conditions. We evaluate our approach on public patch matching benchmark and achieve the best results compared with several state-of-the-arts methods. To show the practicality, we further evaluate IF-Net on the task of visual localization under large illumination changes scenes, and achieves the best localization accuracy.

- Deep-Learning Assisted High-Resolution Binocular Stereo Depth Reconstruction

    Author: Hu, Yaoyu | Carnegie Mellon University
    Author: Zhen, Weikun | Carnegie Mellon University
    Author: Scherer, Sebastian | Carnegie Mellon University
 
    keyword: Computer Vision for Other Robotic Applications; Deep Learning in Robotics and Automation; Aerial Systems: Applications

    Abstract : This work presents dense stereo reconstruction using high-resolution images for infrastructure inspections. The state-of-the-art stereo reconstruction methods, both learning and non-learning ones, consume too much computational resource on high-resolution data. Recent learning-based methods achieve top ranks on most benchmarks. However, they suffer from the generalization issue due to lack of task-specific training data. We propose to use a less resource demanding non-learning method, guided by a learning-based model, to handle high-resolution images and achieve accurate stereo reconstruction. The deep-learning model produces an initial disparity prediction with uncertainty for each pixel of the down-sampled stereo image pair. The uncertainty serves as a self-measurement of its generalization ability and the per-pixel searching range around the initially predicted disparity. The downstream process performs a modified version of the Semi-Global Block Matching method with the up-sampled per-pixel searching range. The proposed deep-learning assisted method is evaluated on the Middlebury dataset and high-resolution stereo images collected by our customized binocular stereo camera. The combination of learning and non-learning methods achieves better performance on 12 out of 15 cases of the Middlebury dataset. In our infrastructure inspection experiments, the average 3D reconstruction error is less than 0.004m.

- Least-Squares Optimal Relative Planar Motion for Vehicle-Mounted Cameras

    Author: Hajder, Levente | E�tv's Lor�nd University
    Author: Barath, Daniel | MTA SZTAKI; Visual Recognition Group in CTU Prague
 
    keyword: Computer Vision for Other Robotic Applications; Visual-Based Navigation

    Abstract : A new closed-form solver is proposed minimizing the algebraic error optimally, in the least squares sense, to estimate the relative planar motion of two calibrated cameras. The main objective is to solve the over-determined case, i.e., when a larger-than-minimal sample of point correspondences is given - thus, estimating the motion from at least three correspondences. The algorithm requires the camera movement to be constrained to a plane, e.g. mounted to a vehicle, and the image plane to be orthogonal to the ground. The solver obtains the motion parameters as the roots of a 6-th degree polynomial. It is validated both in synthetic experiments and on publicly available real-world datasets that using the proposed solver leads to results superior to the state-of-the-art in terms of geometric accuracy with no noticeable deterioration in the processing time.

- Relative Planar Motion for Vehicle-Mounted Cameras from a Single Affine Correspondence

    Author: Hajder, Levente | E�tv's Lor�nd University
    Author: Barath, Daniel | MTA SZTAKI; Visual Recognition Group in CTU Prague
 
    keyword: Computer Vision for Other Robotic Applications; Visual-Based Navigation

    Abstract : Two solvers are proposed for estimating the extrinsic camera parameters from a single affine correspondence assuming general planar motion. In this case, the camera movement is constrained to a plane and the image plane is orthogonal to the ground. The algorithms do not assume other constraints, e.g. the non-holonomic one, to hold. A new minimal solver is proposed for the semi-calibrated case, i.e. the camera parameters are known except a common focal length. Another method is proposed for the fully calibrated case. Due to requiring a single correspondence, robust estimation, e.g. histogram voting, leads to a fast and accurate procedure. The proposed methods are tested in our synthetic environment and on publicly available real datasets consisting of videos through tens of kilometers. They are superior to the state-of-the-art both in terms of accuracy and processing time.

- Moving Object Detection for Visual Odometry in a Dynamic Environment Based on Occlusion Accumulation

    Author: Kim, Haram | Seoul National University
    Author: Kim, Pyojin | Simon Fraser University
    Author: Kim, H. Jin | Seoul National University
 
    keyword: Computer Vision for Other Robotic Applications; Visual-Based Navigation; Object Detection, Segmentation and Categorization

    Abstract : Detection of moving objects is an essential capability in dealing with dynamic environments. Most moving object detection algorithms have been designed for color images without depth. For robotic navigation where real-time RGB-D data is often readily available, utilization of the depth information would be beneficial for obstacle recognition. Here, we propose a simple moving object detection algorithm that uses RGB-D images. The proposed algorithm does not require estimating a background model. Instead, it uses an occlusion model which enables us to estimate the camera pose on a background confused with moving objects that dominate the scene. The proposed algorithm allows to separate the moving object detection and visual odometry (VO) so that an arbitrary robust VO method can be employed in a dynamic situation with a combination of moving object detection, whereas other VO algorithms for a dynamic environment are inseparable. In this paper, we use dense visual odometry (DVO) as a VO method with a bi-square regression weight. Experimental results show the segmentation accuracy and the performance improvement of DVO in the situations. We validate our algorithm in public datasets and our dataset which also publicly accessible.

- A Low-Rank Matrix Approximation Approach to Multiway Matching with Applications in Multi-Sensory Data Association

    Author: Leonardos, Spyridon | University of Pennsylvania
    Author: Zhou, Xiaowei | Zhejiang University
    Author: Daniilidis, Kostas | University of Pennsylvania
 
    keyword: Computer Vision for Other Robotic Applications

    Abstract : Consider the case of multiple visual sensors perceiving the same scene from different viewpoints. In order to achieve consistent visual perception, the problem of data association, in this case establishing correspondences between observed features, must be first solved. In this work, we consider multiway matching which is a specific instance of multi-sensory data association. Multiway matching refers to the problem of establishing correspondences among a set of images from noisy pairwise correspondences, typically by exploiting cycle-consistency.<p>We propose a novel optimization-based formulation of multiway matching problem as a	nonconvex low-rank matrix approximation problem. We propose two novel algorithms for numerically solving the problem at hand. The first one	is an algorithm based on the Alternating Direction Method of Multipliers (ADMM). The second one is a Riemannian trust-region method on	the multinomial manifold, the manifold of strictly positive stochastic matrices, equipped with the Fisher information metric. Experimental results demonstrate that the proposed methods have the state of the art performance in multiway matching while reducing the computational complexity compared to the state of the art.




## Humanoid and Bipedal Locomotion


- LQR-Assisted Whole-Body Control of a Wheeled Bipedal Robot with Kinematic Loops

    Author: Klemm, Victor | ETH Zurich
    Author: Morra, Alessandro | ETH Zurich
    Author: Gulich, Lionel | ETH Zurich
    Author: Mannhart, Dominik | ETH Zurich
    Author: Rohr, David | ETH Zurich
    Author: Kamel, Mina | Autonomous Systems Lab, ETH Zurich
    Author: de Viragh, Yvain | ETH Zurich
    Author: Siegwart, Roland | ETH Zurich
 
    keyword: Legged Robots; Wheeled Robots; Parallel Robots

    Abstract : We present a hierarchical whole-body controller leveraging the full rigid body dynamics of the wheeled bipedal robot Ascento. We derive closed-form expressions for the dynamics of its kinematic loops in a way that readily generalizes to more complex systems. The rolling constraint is incorporated using a compact analytic solution based on rotation matrices. The non-minimum phase balancing dynamics are accounted for by including a linear-quadratic regulator as a motion task. Robustness when driving curves is increased by regulating the lean angle as a function of the zero-moment point. The proposed controller is computationally lightweight and significantly extends the rough-terrain capabilities and robustness of the system, as we demonstrate in several experiments.

- Leveraging the Template and Anchor Framework for Safe, Online Robotic Gait Design

    Author: Liu, Jinsun | University of Michigan, Ann Arbor
    Author: Zhao, Pengcheng | University of Michigan
    Author: Gan, Zhenyu | University of Michigan
    Author: Johnson-Roberson, Matthew | University of Michigan
    Author: Vasudevan, Ram | University of Michigan
 
    keyword: Humanoid and Bipedal Locomotion; Robot Safety; Underactuated Robots

    Abstract : Online control design using a high-fidelity, full-order model for a bipedal robot can be challenging due to the size of the state space of the model. A commonly adopted solution to overcome this challenge is to approximate the full-order model (anchor) with a simplified, reduced-order model (template), while performing control synthesis. Unfortunately it is challenging to make formal guarantees about the safety of an anchor model using a controller designed in an online fashion using a template model. To address this problem, this paper proposes a method to generate safety-preserving controllers for anchor models by performing reachability analysis on template models while bounding the modeling error. This paper describes how this reachable set can be incorporated into a Model Predictive Control framework to select controllers that result in safe walking on the anchor model in an online fashion. The method is illustrated on a 5-link RABBIT model, and is shown to allow the robot to walk safely while utilizing controllers designed in an online fashion.

- Unified Push Recovery Fundamentals: Inspiration from Human Study

    Author: McGreavy, Christopher | University of Edinburgh
    Author: Yuan, Kai | University of Edinburgh
    Author: Gordon, Daniel F. N. | University of Edinburgh
    Author: Tan, Kang | The University of Glasgow
    Author: Wolfslag, Wouter | University of Edinburgh
    Author: Vijayakumar, Sethu | University of Edinburgh
    Author: Li, Zhibin | University of Edinburgh
 
    keyword: Humanoid and Bipedal Locomotion; Legged Robots; Motion Control

    Abstract : Currently for balance recovery, humans outperform humanoid robots which use hand-designed controllers in terms of the diverse actions. This study aims to close this gap by finding core control principles that are shared across ankle, hip, toe and stepping strategies by formulating experiments to test human balance recoveries and define criteria to quantify the strategy in use. To reveal fundamental principles of balance strategies, our study shows that a minimum jerk controller can accurately replicate comparable human behaviour at the Centre of Mass level. Therefore, we formulate a general Model-Predictive Control (MPC) framework to produce recovery motions in any system, including legged machines, where the framework parameters are tuned for time-optimal performance in robotic systems.

- Nonholonomic Virtual Constraint Design for Variable-Incline Bipedal Robotic Walking

    Author: Horn, Jonathan | University of Texas at Dallas
    Author: Mohammadi, Alireza | University of Michigan, Dearborn
    Author: Akbari Hamed, Kaveh | Virginia Tech
    Author: Gregg, Robert D. | University of Michigan
 
    keyword: Legged Robots; Underactuated Robots; Motion Control

    Abstract : This paper presents a method of designing relative-degree-two nonholonomic virtual constraints (NHVCs) that allow for stable bipedal robotic walking across variable terrain slopes. relative-degree-two NHVCs are virtual constraints that encode velocity-dependent walking gaits via momenta conjugate to the unactuated degrees of freedom for the robot. We recently introduced a systematic method of designing NHVCs, based on the hybrid zero dynamics (HZD) control framework, to achieve hybrid invariant flat ground walking without the use of dynamic reset variables. This work addresses the problem of walking over variable-inclined terrain disturbances. We propose a methodology for designing NHVCs, via an optimization problem, in order to achieve stable walking across variable terrain slopes. The end result is a single controller capable of walking over variable-inclined surfaces, that is also robust to inclines not considered in the optimization design problem, and uncertainties in the inertial parameters of the model.

- MPC for Humanoid Gait Generation: Stability and Feasibility (I)

    Author: Scianca, Nicola | Sapienza University of Rome
    Author: De Simone, Daniele | Sapienza University of Rome
    Author: Lanari, Leonardo | Sapienza University of Rome
    Author: Oriolo, Giuseppe | Sapienza University of Rome
 
    keyword: Humanoid and Bipedal Locomotion; Humanoid Robots

    Abstract : In this article, we present an intrinsically stable Model Predictive Control (IS-MPC) framework for humanoid gait generation that incorporates a stability constraint in the formulation. The method uses as prediction model a dynamically extended Linear Inverted Pendulum with Zero Moment Point (ZMP) velocities as control inputs, producing in real time a gait (including footsteps with timing) that realizes omnidirectional motion commands coming from an external source. The stability constraint links future ZMP velocities to the current state so as to guarantee that the generated Center of Mass (CoM) trajectory is bounded with respect to the ZMP trajectory. Being the MPC control horizon finite, only part of the future ZMP velocities are decision variables; the remaining part, called tail, must be either conjectured or anticipated using preview information on the reference motion. Several options for the tail are discussed, each corresponding to a specific terminal constraint. A feasibility analysis of the generic MPC iteration is developed and used to obtain sufficient conditions for recursive feasibility. Finally, we prove that recursive feasibility guarantees stability of the CoM/ZMP dynamics. Simulation and experimental results on NAO and HRP-4 are presented to highlight the performance of IS-MPC.

- A Robust Walking Controller Based on Online Optimization of Ankle, Hip, and Stepping Strategies (I)
 
    Author: Jeong, Hyobin | KAIST
    Author: Lee, Inho | IHMC
    Author: Oh, Jaesung | KAIST
    Author: Lee, Kang Kyu | KAIST Hubolab
    Author: Oh, Jun Ho | Korea Advanced Inst. of Sci. and Tech
 
    keyword: Humanoid and Bipedal Locomotion; Humanoid Robots; Legged Robots

    Abstract : In this paper, we propose a biped walking controller that optimized three push recovery strategies: the ankle, hip, and stepping strategies. We suggested formulations that related the effects of each strategy to the stability of walking based on the linear inverted pendulum with flywheel model.With these relations, we could set up an optimization problem that integrates all the strategies, including step time change. These strategies are not applied hierarchically, but applied according to each weighting factor. Various combinations of weighting factors can be used to determine how the robot should respond to an external push. The optimization problem derived here includes many nonlinear components, but it has been linearized though some assumptions and it can be applied to a robot in real time. Our method is designed to be robust to modeling errors or weak perturbations, by exploiting the advantages of the foot. Hence, it is very practical to apply this algorithm to a real robot. The effectiveness of the walking controller has been verified through     Abstracted model simulation, full dynamics simulation, and a practical robot experiments.


- Passive Dynamic Balancing and Walking in Actuated Environments

    Author: Reher, Jenna | California Institute of Technology
    Author: Csomay-Shanklin, Noel | California Institute of Technology
    Author: Christensen, David | Stanford University
    Author: Bristow, Robert | Walt Disney Imagineering
    Author: Ames, Aaron | California Institute of Technology
    Author: Smoot, Lanny | Walt Disney Imagineering R&amp;D
 
    keyword: Passive Walking; Underactuated Robots; Humanoid and Bipedal Locomotion

    Abstract : The control of passive dynamic systems remains a challenging problem in the field of robotics, and insights from their study can inform everything from dynamic behaviors on actuated robots to robotic assistive devices. In this work, we explore the use of flat actuated environments for realizing passive dynamic balancing and locomotion. Specifically, we utilize a novel omnidirectional actuated floor to dynamically stabilize two robotic systems. We begin with an inverted pendulum to demonstrate the ability to control a passive system through an active environment. We then consider a passive bipedal robot wherein dynamically stable periodic walking gaits are generated through an optimization that leverages the actuated floor. The end result is the ability to demonstrate passive dynamic walking experimentally through the use of actuated environments.

- Biped Stabilization by Linear Feedback of the Variable-Height Inverted Pendulum Model

    Author: Caron, Stephane | ANYbotics AG
 
    keyword: Humanoid and Bipedal Locomotion

    Abstract : The variable-height inverted pendulum (VHIP) model enables a new balancing strategy by height variations of the center of mass, in addition to the well-known ankle strategy. We propose a biped stabilizer based on linear feedback of the VHIP that is simple to implement, coincides with the state-of-the-art for small perturbations and is able to recover from larger perturbations thanks to this new strategy. This solution is based on "best-effort" pole placement of a 4D divergent component of motion for the VHIP under input feasibility and state viability constraints. We complement it with a suitable whole-body admittance control law and test the resulting stabilizer on the HRP-4 humanoid robot.

- Stability Criteria of Balanced and Steppable Unbalanced States for Full-Body Systems with Implications in Robotic and Human Gait

    Author: Peng, William | New York University
    Author: Mummolo, Carlotta | New York University
    Author: Kim, Joo H. | New York University
 
    keyword: Legged Robots; Humanoid and Bipedal Locomotion; Passive Walking

    Abstract : Biped walking involves a series of transitions between single support (SS) and double support (DS) contact configurations that can include both balanced and unbalanced states. The new concept of steppability is introduced to partition the set of unbalanced states into steppable states and falling (unsteppable) states based on the ability of a biped system to respond to forward velocity perturbations by stepping. In this work, a complete system-specific analysis of the stepping process including full-order nonlinear system dynamics is presented for the DARwIn-OP humanoid robot and a human subject in the sagittal plane with respect to both balance stability and steppability. The balance stability and steppability of each system are analyzed by numerical construction of its balance stability boundaries (BSB) for the initial SS and final DS contact configuration and the steppable unbalanced state boundary (SUB). These results are presented with center of mass (COM) trajectories obtained from walking experiments to benchmark robot controller performance and analyze the variation of balance stability and steppability with COM and swing foot position along the progression of a step cycle. For each system, DS BSBs were obtained with both constrained and unconstrained arms in order to demonstrate the ability of this approach to incorporate the effects of angular momentum and system-specific characteristics such as actuation torque, velocity, and angle limits.

- Material Handling by Humanoid Robot While Pushing Carts Using a Walking Pattern Based on Capture Point

    Author: Chagas Vaz, Jean M. | University of Nevada Las Vegas
    Author: Oh, Paul Y. | University of Nevada, Las Vegas (UNLV)
 
    keyword: Humanoid and Bipedal Locomotion; Legged Robots; Humanoid Robots

    Abstract : This paper presents a study that evaluates the effects on the walking pattern of a full-sized humanoid robot as it pushes different carts. Furthermore, it discuss a modified Zero Moment Point (ZMP) pattern based on a capture point method, and a friction compensation method for the arms. Humanoid researchers have demonstrated that robots can perform a wide range of tasks including handling tools, climbing ladders, and patrolling rough terrain. However, when it comes to handling objects while walking, humanoids are relatively limited; it becomes more apparent when humanoids have to push a cart. Many challenges become evident under such circumstances; for example, the walking pattern will be severely affected by the external force opposed by the cart. Therefore, an appropriate walking pattern dynamic model and arm compliance are needed to mitigate external forces. This becomes crucial in order to ensure the robot's self-balance and minimize external disturbances.

- Interconnection and Damping Assignment Passivity-Based Control for Gait Generation in Underactuated Compass-Like Robots

    Author: Arpenti, Pierluigi | CREATE Consortium
    Author: Ruggiero, Fabio | Université Di Napoli Federico II
    Author: Lippiello, Vincenzo | University of Naples FEDERICO II
 
    keyword: Passive Walking; Legged Robots

    Abstract : A compass-like biped robot can go down a gentle slope without the need of actuation through a proper choice of its dynamic parameter and starting from a suitable initial condition. Addition of control actions is requested to generate additional gaits and robustify the existing one. This paper designs an interconnection and damping assignment passivity- based control, rooted within the port-Hamiltonian framework, to generate further gaits with respect to state-of-the-art method- ologies, enlarge the basin of attraction of existing gaits, and further robustify the system against controller discretization and parametric uncertainties. The performance of the proposed algorithm is validated through numerical simulations and comparison with existing passivity-based techniques.

-  Safety-Critical Control of a Cassie Bipedal Robot Riding Hovershoes for Vision-Based Obstacle Avoidance

    Author: Zhang, Bike | University of California, Berkeley
    Author: Sreenath, Koushil | University of California, Berkeley


- A Methodology for the Incorporation of Arbitrarily-Shaped Feet in Passive Bipedal Walking Dynamics

    Author: Smyrli, Aikaterini | National Technical University of Athens
    Author: Papadopoulos, Evangelos | National Technical University of Athens
 
    keyword: Humanoid and Bipedal Locomotion; Passive Walking; Dynamics

    Abstract : A methodology for implementing arbitrary foot shapes in the passive walking dynamics of biped robots is developed. The dynamic model of a walking robot is defined in a way that allows shape-dependent foot kinetics to contribute to the robot's dynamics, for all convex foot shapes regardless of the exact foot geometry: for the developed method, only the set of points describing the foot profile curve is needed. The method is mathematically derived and then showcased with an application. The open-source pose estimation system OpenPose is used to determine the foot profile that enables the rigid-foot passive robot to reproduce the ankle trajectory of the actively powered, multi-DOF human foot complex. The passive gait of the biped robot walking on the specified foot shape is simulated and analyzed, and a stable walking cycle is found and evaluated. The proposed model enables the study of the effects of foot shape on the walking dynamics of biped robots, eliminating the necessity of solely using simple, and analytically defined geometric shapes as the walking robots' feet. The method can be used for foot shape optimization towards achieving any desired walking pattern in walking robots.

- Experimental Analysis of Structural Vibration Problems of a Biped Walking Robot

    Author: Berninger, Tobias Franz Christian | TU Munich
    Author: Sygulla, Felix | Technical University of Munich
    Author: Fuderer, Sebastian | Technical University of Munich
    Author: Rixen, Daniel | Technische Universitét M�nchen
 
    keyword: Humanoid and Bipedal Locomotion; Legged Robots; Calibration and Identification

    Abstract : Over the past decade we have been able to vastly improve the control algorithms of our biped walking robot Lola. Further enhancements, however, are limited by vibration problems caused by the dynamics of Lola's mechanical structure. In this work, we present small examples how structural dynamics limit our control design for walking control as well as low level position control of the joints. We also provide a procedure to identify weaknesses in the structural design of our biped using Experimental Modal Analysis. Using this method, we could successfully identify the structural modes of the system. Furthermore, we were able to use a closed-loop identification method to show a connection between the control loop resonances and the structural resonances of our robot.

- Dynamic Coupling As an Indicator of Gait Robustness for Underactuated Biped Robots

    Author: Fevre, Martin | University of Notre Dame
    Author: Schmiedeler, James | University of Notre Dame
 
    keyword: Legged Robots

    Abstract : This paper employs velocity decomposition of underactuated mechanical systems to determine the degree of dynamic coupling in the gaits of a two-link biped model. The degree of coupling between controlled and uncontrolled directions quantifies the control     Authority the system has over its unactuated degree of freedom. This paper shows that the amount of coupling is directly correlated to gait robustness, as seen through the size of the gait's region of attraction. The analytical measure of coupling is applied in the context of trajectory optimization to generate two-link gaits that maximize or minimize coupling. Simulation studies show that gaits maximizing coupling exhibit significantly superior robustness, as measured by 1) stochastic performance on uneven terrain, 2) ability to maintain desired walking speed under non-vanishing disturbances, 3) size of the region of attraction, and 4) robustness to model uncertainties.

- ZMP Constraint Restriction for Robust Gait Generation in Humanoids

    Author: Smaldone, Filippo Maria | Sapienza University of Rome
    Author: Scianca, Nicola | Sapienza University of Rome
    Author: Modugno, Valerio | Sapienza Université Di Roma
    Author: Lanari, Leonardo | Sapienza University of Rome
    Author: Oriolo, Giuseppe | Sapienza University of Rome
 
    keyword: Humanoid and Bipedal Locomotion; Humanoid Robots; Robust/Adaptive Control of Robotic Systems

    Abstract : We present an extension of our previously proposed IS-MPC method for humanoid gait generation aimed at obtaining robust performance in the presence of disturbances. The considered disturbance signals vary in a range of known amplitude around a mid-range value that can change at each sampling time, but whose current value is assumed to be available. The method consists in modifying the stability constraint that is at the core of IS-MPC by incorporating the current mid-range disturbance, and performing an appropriate restriction of the ZMP constraint in the control horizon on the basis of the range amplitude of the disturbance. We derive explicit conditions for recursive feasibility and internal stability of the IS-MPC method with constraint modification. Finally, we illustrate its superior performance with respect to the nominal version by performing dynamic simulations on the NAO robot.

- Hybrid Zero Dynamics Inspired Feedback Control Policy Design for 3D Bipedal Locomotion Using Reinforcement Learning

    Author: Castillo, Guillermo | The Ohio State University
    Author: Weng, Bowen | The Ohio State University
    Author: Zhang, Wei | Southern University of Science and Technology
    Author: Hereid, Ayonga | Ohio State University
 
    keyword: Legged Robots; Deep Learning in Robotics and Automation

    Abstract : This paper presents a novel model-free reinforcement learning (RL) framework to design feedback control policies for 3D bipedal walking. Existing RL algorithms are often trained in an end-to-end manner or rely on prior knowledge of some reference joint trajectories. Different from these studies, we propose a novel policy structure that appropriately incorporates physical insights gained from the hybrid nature of the walking dynamics and the well-established hybrid zero dynamics approach for 3D bipedal walking. As a result, the overall RL framework has several key advantages, including lightweight network structure, short training time, and less dependence on prior knowledge. We demonstrate the effectiveness of the proposed method on Cassie, a challenging 3D bipedal robot. The proposed solution produces stable limit walking cycles that can track various walking speed in different directions. Surprisingly, without specifically trained with disturbances to achieve robustness, it also performs robustly against various adversarial forces applied to the torso towards both the forward and the backward directions.

- Optimal Reduced-Order Modeling of Bipedal Locomotion

    Author: Chen, Yu-Ming | University of Pennsylvania
    Author: Posa, Michael | University of Pennsylvania
 
    keyword: Humanoid and Bipedal Locomotion; Legged Robots; Model Learning for Control

    Abstract : State-of-the-art approaches to legged locomotion are widely dependent on the use of models like the linear inverted pendulum (LIP) and the spring-loaded inverted pendulum (SLIP), popular because their simplicity enables a wide array of tools for planning, control, and analysis. However, they inevitably limit the ability to execute complex tasks or agile maneuvers. In this work, we aim to automatically synthesize models that remain low-dimensional but retain the capabilities of the high-dimensional system. For example, if one were to restore a small degree of complexity to LIP, SLIP, or a similar model, our approach discovers the form of that additional complexity which optimizes performance. In this paper, we define a class of reduced-order models and provide an algorithm for optimization within this class. To demonstrate our method, we optimize models for walking at a range of speeds and ground inclines, for both a five-link model and the Cassie bipedal robot.

## Motion Control
- Anti-Jackknife Control of Tractor-Trailer Vehicles Via Intrinsically Stable MPC

    Author: Beglini, Manuel | DIAG, Sapienza University of Rome
    Author: Lanari, Leonardo | Sapienza University of Rome
    Author: Oriolo, Giuseppe | Sapienza University of Rome
 
    keyword: Motion Control; Nonholonomic Mechanisms and Systems; Autonomous Vehicle Navigation

    Abstract : It is common knowledge that tractor-trailer vehicles are affected by jackknifing, a phenomenon that consists in the divergence of the trailer hitch angle and ultimately causes the vehicle to fold up. For the case of backwards motion, in which jackknifing can also occur at low speeds, we present a control method that drives the vehicle along a reference Cartesian trajectory while avoiding the divergence of the hitch angle. In particular, a feedback control law is obtained by combining two actions: a tracking term, computed using input-output linearization, and a corrective term, generated via IS-MPC, an intrinsically stable MPC scheme which is effective for stable inversion of nonminimum-phase systems. The proposed method has been verified in simulation and experimentally validated on a purposely built prototype.

- On Sensing-Aware Model Predictive Path-Following Control for a Reversing General 2-Trailer with a Car-Like Tractor

    Author: Ljungqvist, Oskar | Link�ping University, ISY, Automatic Control
    Author: Axehill, Daniel | Link�ping University
    Author: Pettersson, Henrik | Scania CV
 
    keyword: Motion Control; Optimization and Optimal Control; Wheeled Robots

    Abstract : The design of reliable path-following controllers is a key ingredient for successful deployment of self-driving vehicles. This controller-design problem is especially challenging for a general 2-trailer with a car-like tractor due to the vehicle's structurally unstable joint-angle kinematics in backward motion and the car-like tractor's curvature limitations which can cause the vehicle segments to fold and enter a jackknife state. Furthermore, optical sensors with a limited field of view have been proposed to solve the joint-angle estimation problem online, which introduce additional restrictions on which vehicle states that can be reliably estimated. To incorporate these restrictions at the level of control, a model predictive path-following controller is proposed. By taking the vehicle's physical and sensing limitations into account, it is shown in real-world experiments that the performance of the proposed path-following controller in terms of suppressing disturbances and recovering from non-trivial initial states is significantly improved compared to a previously proposed solution where the constraints have been neglected.

- Offline Practising and Runtime Training Framework for Autonomous Motion Control of Snake Robots

    Author: Cheng, Long | Sun Yat-Sen University
    Author: Huang, Jianping | Sun Yat-Sen University
    Author: Liu, Linlin | Sun Yat-Sen University
    Author: Jian, Zhiyong | Sun Yat-Sen University
    Author: Huang, Yuhong | National University of Defense Technology
    Author: Huang, Kai | Sun Yat-Sen University
 
    keyword: Motion Control; Robust/Adaptive Control of Robotic Systems; Biologically-Inspired Robots

    Abstract : This paper proposes an offline and runtime combined framework for the autonomous motion of snake robots. With the dynamic feedback of its state during runtime, the robot utilizes the linear regression to update its control parameters for better performance and thus adaptively reacts to the environment. To reduce interference from infeasible samples and improve efficiency, the data set for runtime training is chosen from one in several clusters categorized from samples collected in offline practice. Moreover, only the most sensitive control parameter is updated at one iteration for better robustness and efficiency. The effectiveness and efficiency of our approach are evaluated by a set of case studies of pole climbing. Experimental results demonstrate that with the proposed framework, the snake robot can adapt its locomotion gait to poles with different unknown diameters.

- Control of a Differentially Driven Nonholonomic Robot Subject to a Restricted Wheels Rotation

    Author: Pazderski, Dariusz | Poznan University of Technology
    Author: Kozlowski, Krzysztof R. | Poznan University of Technology
 
    keyword: Motion Control; Wheeled Robots; Nonholonomic Mechanisms and Systems

    Abstract : The paper deals with non-standard motion tasks specified for a two-wheeled nonholonomic robot. It is assumed that wheels cannot fully rotate which reduces a set of feasible movements significantly. In spite of these constraints, it is expected that position of the robot can be changed without violating nonholonomic constraints. Such a possibility comes from small time local controllability (STLC) of the kinematics described on four-dimensional configuration manifold.<p>In order to solve these specific tasks a feedback taking advantage of the transverse function approach is designed. Consequently, the system can be virtually released from nonholonomic constraints. The transverse function also defines a virtual geometry constraint which makes it possible to limit wheels rotation. </p><p>Properties of the designed controller are illustrated by results of numerical simulations in various motion task scenarios.

- Inferring Task-Space Central Pattern Generator Parameters for Closed-Loop Control of Underactuated Robots

    Author: Kent, Nathan | University of Rochester
    Author: Bhirangi, Raunaq Mahesh | Carnegie Mellon University
    Author: Travers, Matthew | Carnegie Mellon University
    Author: Howard, Thomas | University of Rochester
 
    keyword: Motion Control; Underactuated Robots; Model Learning for Control

    Abstract : The complexity associated with the control of highly-articulated legged robots scales quickly as the number of joints increases. Traditional approaches to the control of these robots are often impractical for many real-time applications. This work thus presents a novel sampling-based planning approach for highly-articulated robots that utilizes a probabilistic graphical model (PGM) to infer in real-time how to optimally modify goal-driven, locomotive behaviors for use in closed-loop control. Locomotive behaviors are quantified in terms of the parameters associated with a network of neural oscillators, or rather a central pattern generator (CPG). For the first time, we show that the PGM can be used to optimally modulate different behaviors in real-time (i.e., to select of optimal choice of parameter values across the CPG model) in response to changes both in the local environment and in the desired control signal. The PGM is trained offline using a library of optimal behaviors that are generated using a gradient-free optimization framework.

- Magnetically Actuated Simple Millirobots for Complex Navigation and Modular Assembly

    Author: Al Khatib, Ehab | Southern Methodist University
    Author: Bhattacharjee, Anuruddha | Southern Methodist University
    Author: Razzaghi, Pouria | Southern Methodist University
    Author: Rogowski, Louis | Southern Methodist University
    Author: Kim, MinJun | Southern Methodist University
    Author: Hurmuzlu, Yildirim | Southern Methodist University
 
    keyword: Motion Control; Task Planning; Additive Manufacturing

    Abstract : Magnetic millirobots can be controlled remotely by external magnetic fields, making them promising candidates for biomedical and engineering applications. This paper presents a low-cost millirobot that has simple in design, easy to fabricate, highly scalable, and can be used as modular sub-units within complex structures for large-scale manipulation. The rectangular-shaped millirobot was made by 3D printing by using polylactic acid (PLA) filaments. Two cylindrical permanent magnets are embedded in each end. Individual millirobots are highly agile and capable of performing a variety of locomotive tasks such as pivot walking, tapping, and tumbling. A comparative study is presented to demonstrate the advantages and disadvantages of each locomotion mode. However, among these modes of locomotion, pivot walking at millimeter length scale is demonstrated for the first time in this paper, and our experimental data shows that this is the fastest and the most stable. Further, we demonstrate that the millirobot could be deployed through an esophagus-like bent tube and a maze-like path with combined motion modes. Later, to extend the functionality of our millirobots, we present two systems utilizing multiple millirobots combined together: a stag beetle and a carbot. Using a powerful electromagnetic coil system, we conduct extensive experiments to establish feasibility and practical utility of the magnetically actuated millirobot.

## Dexterous Manipulation
- MagicHand: Context-Aware Dexterous Grasping Using an Anthropomorphic Robotic Hand

    Author: Li, Hui | Wichita State University
    Author: Tan, Jindong | University of Tennessee, Knoxville
    Author: He, Hongsheng | Wichita State University
 
    keyword: Dexterous Manipulation; Recognition; Grasping

    Abstract : Understanding of characteristics of objects such as fragility, rigidity, texture and dimensions facilitates and innovates robotic grasping. In this paper, we propose a context-aware anthropomorphic robotic hand (MagicHand) grasping system which is able to gather various information about its target object and generate grasping strategies based on the perceived information. In this work, NIR spectra of target objects are perceived to recognize materials on a molecular level and RGB-D images are collected to estimate dimensions of the objects. We selected six most used grasping poses and our system is able to decide the most suitable grasp type and grasp size based on the characteristics of an object. Through multiple experiments, the performance of the MagicHand system is demonstrated.

- Strategy for Roller Chain Assembly with Parallel Jaw Gripper

    Author: Tatemura, Keiki | Wakayama University
    Author: Dobashi, Hiroki | Wakayama University
 
    keyword: Dexterous Manipulation; Grippers and Other End-Effectors; Manufacturing, Maintenance and Supply Chains

    Abstract : For realizing a versatile, &#64258;exible robotic assembly system in manufacturing, robots need to handle not only rigid parts but also parts with &#64258;exible characteristics such as belts, cables, roller chains, and so on. This paper presents a strategy for assembling a roller chain to a set of sprockets with a conventional parallel jaw gripper, utilizing the property of the roller chain. Targeting the roller chain assembly of a belt drive unit designed for the World Robot Summit 2018, the validity and the utility of the proposed strategy are experimentally veri&#64257;ed.

- Distal Hyperextension Is Handy: High Range of Motion in Cluttered Environments

    Author: Ruotolo, Wilson | Stanford University
    Author: Thomasson, Rachel | University of California, Berkeley
    Author: Herrera, Joel | Stanford
    Author: Gruebele, Alexander | Stanford University
    Author: Cutkosky, Mark | Stanford University
 
    keyword: Dexterous Manipulation; Grippers and Other End-Effectors; Multifingered Hands

    Abstract : As robots branch out from the manufacturing sector into the home, there is a pressing need for new technology that can operate in cluttered and unstructured human environments. Loading and unloading a dishwasher serves as a difficult representative challenge for in-home robots, and a new robotic end-effector has been developed for this type of task. The actuation of the fingers is integrated with a bending degree of freedom that is nearly coincident with the proximal joints of the fingers, an arrangement that greatly increases the kinematic workspace in constrained environments. In addition, the distal joints of the fingers are capable of hyperextension (bending backwards), allowing them to pinch a wider range of surface curvatures and angles securely. A third feature of the hand is a palm that combines a granular jamming substrate with suction cups to adhere to wet and slippery objects of varying curvatures. Integration of these features into a single prototype allows the hand to grasp and manipulate dirty dishes reliably and with low gripping forces, as demonstrated in object acquisition and manipulation with less than 10N of applied force.

- Learning Pre-Grasp Manipulation for Objects in Un-Graspable Poses

    Author: Sun, Zhaole | Tsinghua University, the University of Edinburgh, Intel Lab Chin
    Author: Yuan, Kai | University of Edinburgh
    Author: Hu, Wenbin | University of Edinburgh
    Author: Yang, Chuanyu | University of Edinburgh
    Author: Li, Zhibin | University of Edinburgh
 
    keyword: Dexterous Manipulation; Dual Arm Manipulation; Grasping

    Abstract : In robotic grasping, objects are often occluded in ungraspable configurations such that no feasible grasp pose can be found, e.g. large flat boxes on the table that can only be grasped once lifted. Inspired by human bimanual manipulation, e.g. one hand to lift up things and the other to grasp, we address this type of problems by introducing pregrasp manipulation -- push and lift actions. We propose a model-free Deep Reinforcement Learning framework to train feedback control policies that utilize visual information and proprioceptive states of the robot to autonomously discover robust pregrasp manipulation. The robot arm learns to push the object first towards a support surface and then lift up one side of the object, creating an object-table clearance for possible grasping solutions. Furthermore, we show the robustness of the proposed learning framework in training pregrasp policies that can be directly transferred to a real robot. Lastly, we evaluate the effectiveness and generalization ability of the learned policy in real-world experiments, and demonstrate pregrasp manipulation of objects with various sizes, shapes, weights, and surface friction.

- Object-Level Impedance Control for Dexterous In-Hand Manipulation

    Author: Pfanne, Martin | DLR German Aerospace Center
    Author: Chalon, Maxime | German Aerospace Center (DLR)
    Author: Stulp, Freek | DLR - Deutsches Zentrum F�r Luft Und Raumfahrt E.V
    Author: Ritter, Helge Joachim | Bielefeld University
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Dexterous Manipulation; Grasping; Compliance and Impedance Control

    Abstract : This work presents a novel object-level control framework for the dexterous in-hand manipulation of objects with torque-controlled robotic hands. The proposed impedance-based controller realizes the compliant 6-DOF control of a grasped object. Enabled by the in-hand localization, the slippage of contacts is avoided by actively maintaining the desired grasp configuration. The internal forces on the object are determined by solving a quadratic optimization problem, which explicitly considers friction constraints on the contacts and allows to gradually shift the load between fingers. The proposed framework is capable of dealing with dynamic changes in the grasp configuration, which makes it applicable for the control of the object during grasp acquisition or the reconfiguration of fingers (i.e. finger gaiting). A nullspace controller avoids joint limits and singularities. Experiments with the DLR robot David demonstrate the efficiency and performance of the controller in various grasping scenarios.

- Picking Thin Objects by Tilt-And-Pivot Manipulation and Its Application to Bin Picking

    Author: Tong, Zhekai | The Hong Kong University of Science and Technology
    Author: He, Tierui | The Hong Kong University of Science and Technology
    Author: Kim, Chung Hee | The Hong Kong University of Science and Technology
    Author: Ng, Yu Hin | The Hong Kong University of Science and Technology
    Author: Xu, Qianyi | The Hong Kong University of Science and Technology
    Author: Seo, Jungwon | The Hong Kong University of Science and Technology
 
    keyword: Dexterous Manipulation; Grasping; Grippers and Other End-Effectors

    Abstract : This paper introduces the technique of tilt-and-pivot manipulation, a new method for picking thin, rigid objects lying on a flat surface through robotic dexterous in-hand manipulation. During the manipulation process, the gripper is controlled to reorient about the contact with the object such that its finger can get in the space between the object and the supporting surface, which is formed by tilting up the object, with no relative sliding motion at the contact. As a result, a pinch grasp can be obtained on the faces of the thin object with ease. We discuss issues regarding the kinematics and planning of tilt-and-pivot, effector shape design, and the overall practicality of the manipulation technique, which is general enough to be applicable to any rigid convex polygonal objects. We also present a set of experiments in a range of bin picking scenarios.

- In-Hand Manipulation of Objects with Unknown Shapes

    Author: Cruciani, Silvia | KTH Royal Institute of Technology
    Author: Yin, Hang | KTH
    Author: Kragic, Danica | KTH
 
    keyword: Dexterous Manipulation; Perception for Grasping and Manipulation

    Abstract : This work addresses the problem of changing grasp configurations on objects with an unknown shape through in-hand manipulation. Our approach leverages shape priors, learned as deep generative models, to infer novel object shapes from partial visual sensing. The Dexterous Manipulation Graph method is extended to build incrementally and account for object shape uncertainty when planning a sequence of manipulation actions. We show that our approach successfully solves in-hand manipulation tasks with unknown objects, and demonstrate the validity of these solutions with robot experiments.

- Learning Hierarchical Control for Robust In-Hand Manipulation

    Author: Li, Tingguang | The Chinese University of Hong Kong
    Author: Srinivasan, Krishnan | Stanford University
    Author: Meng, Max Q.-H. | The Chinese University of Hong Kong
    Author: Yuan, Wenzhen | Carnegie Mellon University
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Dexterous Manipulation; Deep Learning in Robotics and Automation; Motion Control of Manipulators

    Abstract : Robotic in-hand manipulation has been a long-standing challenge due to the complexity of modeling hand and object in contact and of coordinating finger motion for complex manipulation sequences. To address these challenges, the majority of prior work has either focused on model-based, low-level controllers or on model-free deep reinforcement learning that each have their own limitations. We propose a hierarchical method that relies on traditional, model-based controllers on the low-level and learned policies on the mid-level. The low-level controllers can robustly execute different manipulation primitives (reposing, sliding, flipping). The mid-level policy orchestrates these primitives. We extensively evaluate our approach in simulation with a 3-fingered hand that controls three degrees of freedom of elongated objects. We show that our approach can move objects between almost all the possible poses in the workspace while keeping them firmly grasped. We also show that our approach is robust to inaccuracies in the object models and to observation noise. Finally, we show how our approach generalizes to objects of other shapes.

- Tactile Dexterity: Manipulation Primitives with Tactile Feedback

    Author: Hogan, Francois | Massachusetts Institute of Technology
    Author: Ballester, Jose | Massachusetts Institute of Technology
    Author: Dong, Siyuan | MIT
    Author: Rodriguez, Alberto | Massachusetts Institute of Technology
 
    keyword: Dexterous Manipulation; Force and Tactile Sensing; Dual Arm Manipulation

    Abstract : This paper develops closed-loop tactile controllers for dexterous robotic manipulation with a dual-palm robotic system. Tactile dexterity is an approach to dexterous manipulation that plans for robot/object interactions that render interpretable tactile information for control. We divide the role of tactile control into two goals: 1) control the contact state between the end-effector and the object (contact/no-contact, stick/slip) by regulating the stability of planned contact configurations and monitoring undesired slip events; and 2) control the object state by tactile-based tracking and iterative replanning of the object and robot trajectories.<p>Key to this formulation is the decomposition of manipulation plans into sequences of manipulation primitives with simple mechanics and efficient planners. We consider the scenario of manipulating an object from an initial pose to a target pose on a flat surface while correcting for external perturbations and uncertainty in the initial pose of the object. We experimentally validate the approach with an ABB YuMi dual-arm robot and demonstrate the ability of the tactile controller to react to external perturbations.

- Design of a Roller-Based Dexterous Hand for Object Grasping and Within-Hand Manipulation

    Author: Yuan, Shenli | Stanford University
    Author: Epps, Austin | Dexterity Systems
    Author: Nowak, Jerome | Stanford University
    Author: Salisbury, Kenneth | Stanford University
 
    keyword: Dexterous Manipulation; Grasping; Grippers and Other End-Effectors

    Abstract : This paper describes the development of a novel non-anthropomorphic robot hand with the ability to manipulate objects by means of articulated, actively driven rollers located at the fingertips. An analysis is conducted and systems of equations for two-finger and three-finger manipulation of a sphere are formulated to demonstrate full six degree of freedom nonholonomic spatial motion capability. A prototype version of the hand was constructed and used to grasp and manipulate a variety of objects. Tests conducted with the prototype confirmed the validity of the mathematical analysis. Unlike conventional approaches to within-hand manipulation using legacy robotic hands, the continuous rotation capability of our rolling fingertips allows for unbounded rotation of a grasped object without the need for finger gaiting.

- High-Resolution Optical Fiber Shape Sensing of Continuum Robots: A Comparative Study

    Author: Monet, Fr�d�ric | Polytechnique Montreal
    Author: Sefati, Shahriar | Johns Hopkins University
    Author: Lorre, Pierre | Polytechnique Montreal
    Author: Poiffaut, Arthur | Polytechnique Montreal
    Author: Kadoury, Samuel | Polytechnique Montr�al
    Author: Armand, Mehran | Johns Hopkins University Applied Physics Laboratory
    Author: Iordachita, Ioan Iulian | Johns Hopkins University
    Author: Kashyap, Raman | Polytechnique Montreal
 
    keyword: Dexterous Manipulation; Medical Robots and Systems

    Abstract : Flexible medical instruments, such as Continuum Dexterous Manipulators (CDM), constitute an important class of tools for minimally invasive surgery. Fiber Bragg grating (FBG) sensors have demonstrated great potential in shape sensing and consequently tip position estimation of CDMs. However, due to the limited number of sensing locations, these sensors can only accurately recover basic shapes, and become unreliable in the presence of obstacles or many inflection points such as s-bends. Optical Frequency Domain Reflectometry (OFDR), on the other hand, can achieve much higher spatial resolution, and can therefore accurately reconstruct more complex shapes. Additionally, Random Optical Gratings by Ultraviolet laser Exposure (ROGUEs) can be written in the fibers to increase signal to noise ratio of the sensors. In this comparison study, the tip position error is used as a metric to compare both FBG and OFDR shape reconstructions for a CDM developed for orthopedic surgeries, using a pair of stereo cameras as ground truth. The tip position error for the OFDR (and FBG) technique was found to be 0.32 (0.83) mm in free-bending environment, 0.41 (0.80) mm when interacting with obstacles, and 0.45 (2.27) mm in s-bending. Moreover, the maximum tip position error remains sub-mm for the OFDR reconstruction, while it reaches 3.40 mm for FBG reconstruction. These results propose a cost-effective, robust and more accurate alternative to FBG sensors for reconstructing complex CDM shapes.

- Local Trajectory Stabilization for Dexterous Manipulation Via Piecewise Affine Approximations

    Author: Han, Weiqiao | Massachusetts Institute of Technology
    Author: Tedrake, Russ | Massachusetts Institute of Technology
 
    keyword: Dexterous Manipulation; Motion and Path Planning; Motion Control of Manipulators

    Abstract : We propose a model-based approach to design feedback policies for dexterous robotic manipulation. The manipulation problem is formulated as reaching the target region from an initial state for some non-smooth nonlinear system. First, we use trajectory optimization to find a feasible trajectory. Next, we characterize the local multi-contact dynamics around the trajectory as a piecewise affine system, and build a funnel around the linearization of the nominal trajectory using polytopes. We prove that the feedback controller at the vicinity of the linearization is guaranteed to drive the nonlinear system to the target region. During online execution, we solve linear programs to track the system trajectory. We validate the algorithm on hardware, showing that even under large external disturbances, the controller is able to accomplish the task.

## Computer Vision for Automation and Manufacturing
- Monocular Direct Sparse Localization in a Prior 3D Surfel Map

    Author: Ye, Haoyang | The Hong Kong University of Science and Technology
    Author: Huang, Huaiyang | The Hong Kong University of Science and Technology
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Computer Vision for Automation; Localization; Visual Tracking

    Abstract : In this paper, we introduce an approach to tracking the pose of a monocular camera in a prior surfel map. By rendering vertex and normal maps from the prior surfel map, the global planar information for the sparse tracked points in the image frame is obtained. The tracked points with and without the global planar information involve both global and local constraints of frames to the system. Our approach formulates all constraints in the form of direct photometric errors within a local window of the frames. The final optimization utilizes these constraints to provide the accurate estimation of global 6-DoF camera poses with the absolute scale. The extensive simulation and real-world experiments demonstrate that our monocular method can provide accurate camera localization results under various conditions.

- LINS: A Lidar-Inertial State Estimator for Robust and Efficient Navigation

    Author: Qin, Chao | Shenzhen Yiqing Inovation Co., Ltd
    Author: Ye, Haoyang | The Hong Kong University of Science and Technology
    Author: Pranata, Christian Edwin | Robotics and Multiperception Lab HKUST
    Author: Han, Jun | University of Notre Dame
    Author: Zhang, Shuyang | Shenzhen Yiqing Inovation Co., Ltd
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Computer Vision for Automation; Calibration and Identification

    Abstract : We present LINS, a lightweight lidar-inertial state estimator, for real-time ego-motion estimation. The proposed method enables robust and efficient navigation for ground vehicles in challenging environments, such as feature-less scenes, via fusing a 6-axis IMU and a 3D lidar in a tightly-coupled scheme. An iterated error-state Kalman filter (ESKF) is designed to correct the estimated state recursively by generating new feature correspondences in each iteration, and to keep the system computationally tractable. Moreover, we use a robocentric formulation that represents the state in a moving local frame in order to prevent filter divergence in a long run. To validate robustness and generalizability, extensive experiments are performed in various scenarios. Experimental results indicate that LINS offers comparable performance with the state-of-the-art lidar-inertial odometry in terms of stability and accuracy and has order-of-magnitude improvement in speed.

- Automated Eye-In-Hand Robot-3D Scanner Calibration for Low Stitching Errors

    Author: Madhusudanan, Harikrishnan | University of Toronto
    Author: Liu, Xingjian | University of Toronto
    Author: Chen, Wenyuan | University of Toronto
    Author: Li, Dahai | University of Toronto
    Author: Du, Linghao | University of Toronto
    Author: Li, Jianfeng | University of Toronto
    Author: Ge, Ji | University of Toronto
    Author: Sun, Yu | University of Toronto
 
    keyword: Computer Vision for Manufacturing; Factory Automation; Industrial Robots

    Abstract : A 3D measurement system consisting of a 3D scanner and an industrial robot (eye-in-hand) is commonly used to scan large object under test (OUT) from multiple field-of-views (FOVs) for complete measurement. A data stitching process is required to align multiple FOVs into a single coordinate system. Marker-free stitching assisted by robot's accurate positioning becomes increasingly attractive since it bypasses the cumbersome traditional fiducial marker-based method. Most existing methods directly use initial Denavit-Hartenberg (DH) parameters and hand-eye calibration to calculate the transformations between multiple FOVs. Since accuracy of DH parameters deteriorates over time, such methods suffer from high stitching errors (e.g., 0.2 mm) in long-term routine industrial use. This paper reports a new robot-scanner calibration approach to realize such measurement with low data stitching errors. During long-term continuous measurement, the robot periodically moves towards a 2D standard calibration board to optimize kinematic model's parameters to maintain a low stitching error. This capability is enabled by several techniques including virtual arm-based robot-scanner kinematic model, trajectory-based robot-world transformation calculation, nonlinear optimization. Experimental results demonstrated a low data stitching error (&lt; 0.1 mm) similar to the cumbersome marker-based method and a lower system downtime (&lt; 60 seconds vs. 10-15 minutes by traditional DH and hand-eye calibration).

- Monocular Visual Odometry Using Learned Repeatability and Description

    Author: Huang, Huaiyang | The Hong Kong University of Science and Technology
    Author: Ye, Haoyang | The Hong Kong University of Science and Technology
    Author: Sun, Yuxiang | Hong Kong University of Science and Technology
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Computer Vision for Automation; SLAM; Mapping

    Abstract : Robustness and accuracy for monocular visual odometry (VO) under challenging environments are widely concerned. In this paper, we present a monocular VO system leveraging learned repeatability and description. In a hybrid scheme, the camera pose is initially tracked on the predicted repeatability maps in a direct manner and then refined with the patch-wise 3D-2D association. The local feature parameterization and the adapted mapping module further boost different functionalities in the system. Extensive evaluations on challenging public datasets are performed. The competitive performance on camera pose estimation demonstrates the effectiveness of our method. Additional studies on the local reconstruction accuracy and running time exhibit that our system is capable of maintaining a robust and lightweight backend.

- Interaction Graphs for Object Importance Estimation in On-Road Driving Videos

    Author: Zhang, Zehua | Indiana University Bloomington
    Author: Tawari, Ashish | Honda Research Institute
    Author: Martin, Sujitha | Honda Research Institute
    Author: Crandall, David | Indiana University
 
    keyword: Computer Vision for Automation; Deep Learning in Robotics and Automation; Visual Learning

    Abstract : A vehicle driving along the road is surrounded by many objects, but only a small subset of them influence the driver's decisions and actions. Learning to estimate the importance of each object on the driver's real-time decision-making may help better understand human driving behavior and lead to more reliable autonomous driving systems. Solving this problem requires models that understand the interactions between the ego-vehicle and the surrounding objects. However, interactions among other objects in the scene can potentially also be very helpful, e.g., a pedestrian beginning to cross the road between the ego-vehicle and the car in front will make the car in front less important. We propose a novel framework for object importance estimation using an interaction graph, in which the features of each object node are updated by interacting with others through graph convolution. Experiments show that our model outperforms state-of-the-art baselines with much less input and pre-processing.

- A Robotics Inspection System for Detecting Defects on Semi-Specular Painted Automotive Surfaces

    Author: Akhtar, Sohail | University of Guelph
    Author: Tandiya, Adarsh | University of Guelph
    Author: Moussa, Medhat | Guelph
    Author: Tarry, Cole | University of Guelph
 
    keyword: Computer Vision for Manufacturing; Industrial Robots

    Abstract : This paper describes the design and implementation of a real-time robotics system for semi-specular/painted surface defect detection. The system can be used on moving parts, tolerate varying lighting conditions, and can accommodate small inherent vibrations of the inspected surface that is common in manufacturing operations. Topographical information of the inspected surface is first obtained by the analysis of reflections of a known pattern from this surface. Spectral analysis is then applied to identify defects through novelty detection. Finally, a defect tracking mechanism eliminates spurious defects. The proposed system operates continuously at 90 fps. The paper presents field testing results that show the system can be used as a consistent and cost-effective way of quality control.
