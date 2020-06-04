
# International Conference on Robotics and Automation 2020
 
Welcome to ICRA 2020, the 2020 IEEE International Conference on Robotics and Automation.

This list is edited by [PaopaoRobot, 泡泡机器人](https://github.com/PaoPaoRobot) , the Chinese academic nonprofit organization. Recently we will classify these papers by topics. Welcome to follow our github and our WeChat Public Platform Account ( [paopaorobot_slam](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=100000102&idx=1&sn=0a8a831a4f2c18443dbf436ef5d5ff8c&chksm=6c10bf625b6736748c9612879e166e510f1fe301b72ed5c5d7ecdd0f40726c5d757e975f37af&mpshare=1&scene=1&srcid=0530KxSLjUE9I38yLgfO2nVm&pass_ticket=0aB5tcjeTfmcl9u0eSVzN4Ag4tkpM2RjRFH8DG9vylE%3D#rd) ). Of course, you could contact with [daiwei.song@outlook.com](mailto://daiwei.song@outlook.com)

## Visual Servoing
- Active Deformation through Visual Servoing of Soft Objects

    Author: Lagneau, Romain | INSA Rennes
    Author: Krupa, Alexandre | INRIA Rennes - Bretagne Atlantique
    Author: Marchal, Maud | INSA/INRIA
 
    keyword: Visual Servoing

    Abstract : In this paper, we propose the ADVISEd (Active Deformation through VIsual SErvoing) method, a novel model-free deformation servoing method able to deform a soft object towards a desired shape. ADVISEd relies on an online estimation of the deformation Jacobian that relates the motion of the robot end-effector to the deformation behavior of the object. The estimation is based on a weighted least-squares minimization with a sliding window. The robustness of the method to observation noise is ensured using an eigenvalue-based confidence criterion. The ADVISEd method is validated through comparisons with a model-based and a model-free state-of-the-art methods. Two experimental setups are proposed to compare the methods, one to perform a marker-based active shaping task and one to perform several marker-less active shaping and shape preservation tasks. Experiments showed that our approach can interactively control the deformations of an object in different tasks while ensuring better robustness to external perturbations than the state-of-the-art methods.

- Visual Geometric Skill Inference by Watching Human Demonstration

    Author: Jin, Jun | University of Alberta
    Author: Petrich, Laura | University of Alberta
    Author: Zhang, Zichen | University of Alberta, Canada
    Author: Dehghan, Masood | University of Alberta
    Author: Jagersand, Martin | University of Alberta
 
    keyword: Visual Servoing; Visual Learning; Learning from Demonstration

    Abstract : We study the problem of learning manipulation skills from human demonstration video by inferring the association relationships between geometric features. Motivation for this work stems from the observation that humans perform eye-hand coordination tasks by using geometric primitives to define a task while a geometric control error drives the task through execution. We propose a graph based kernel regression method to directly infer the underlying association constraints from human demonstration video using Incremental Maximum Entropy Inverse Reinforcement Learning (InMaxEnt IRL). The learned skill inference provides human readable task definition and outputs control errors that can be directly plugged into traditional controllers. Our method removes the need for tedious feature selection and robust feature trackers required in traditional approaches (e.g. feature-based visual servoing). Experiments show our method infers correct geometric associations even with only one human demonstration video and can generalize well under variance.

- Direct Visual Servoing in the Frequency Domain

    Author: Marchand, Eric | Univ Rennes, Inria, CNRS, IRISA
 
    keyword: Visual Servoing

    Abstract : In this paper, we propose an original approach to extend direct visual servoing to the frequency domain. Whereas most of visual servoing approaches relied on the geometric features, recent works have highlighted the importance of taking into account the photometric information of the entire images. This leads to direct visual servoing (DVS) approaches. In this paper we propose no longer to consider the image itself in the spatial domain but its transformation in the frequency domain. The idea is to consider the Discrete Cosine Transform (DCT) which allows to represent the image in the frequency domain in terms of a sum of cosine functions that oscillate at various frequencies. This leads to a new set of coordinates in a new precomputed orthogonal basis, the coefficients of the DCT. We propose to use these coefficients as the visual features that are then consider in a visual servoing control law. We then exhibit the analytical formulation of the interaction matrix related to these coefficients. Experimental results validate our approach.

- DFVS: Deep Flow Guided Scene Agnostic Image Based Visual Servoing

    Author: Y V S, Harish | International Institute of Information Technology Hyderabad
    Author: Pandya, Harit | IIIT Hyderabad
    Author: Gaud, Ayush | International Institute of Information Technology Hyderabad
    Author: Terupally, Shreya | International Institute of Information Technology, Hyderabad
    Author: Narasimhan, Sai Shankar | IIIT Hyderabad
    Author: Krishna, Madhava | IIIT Hyderabad
 
    keyword: Visual Servoing

    Abstract : Existing deep learning based visual servoing approaches regress the relative camera pose between a pair of images. Therefore, they require a huge amount of training data and sometimes fine-tuning for adaptation to a novel scene. Furthermore, current approaches do not consider underlying geometry of the scene and rely on direct estimation of camera pose. Thus, inaccuracies in prediction of the camera pose, especially for distant goals, lead to a degradation in the servoing performance. In this paper, we propose a two-fold solution:(i)We consider optical flow as our visual features, which are predicted using a deep neural network.(ii)These flow features are then systematically integrated with depth estimates provided by another neural network using interaction matrix.We further present an extensive benchmark in a photo-realistic 3D simulation across diverse scenes to study the convergence and generalisation of visual servoing approaches. We show convergence for over 3m and 40 degrees while maintaining precise positioning of under 2cm and 1 degree on our challenging benchmark where the existing approaches that are unable to converge for majority of scenarios for over 1.5m and 20 degrees. Furthermore,we also evaluate our approach for a real scenario on an aerial robot. Our approach generalizes to novel scenarios producing precise and robust servoing performance for 6 degrees of freedom positioning tasks with even large camera transformations without any retraining or fine-tuning.

- Photometric Path Planning for Vision-Based Navigation

    Author: Rodriguez Martinez, Eder Alejandro | MIS Laboratory, University of Picardie Jules Verne
    Author: Caron, Guillaume | Université De Picardie Jules Verne
    Author: Pegard, Claude | Université De Picardie Jules Verne
    Author: Lara, David | Instituto Tecnologico Superior De Misantla
 
    keyword: Visual Servoing; Visual-Based Navigation

    Abstract : We present a vision-based navigation system that uses a visual memory to navigate. Such memory corresponds to a topological map of key images created from moving a virtual camera over a model of the real scene. The advantage of our approach is that it provides a useful insight into the navigability of a visual path without relying on a traditional learning stage. During the navigation stage, the robot is controlled by sequentially comparing the images stored in the memory with the images acquired by the onboard camera. The evaluation is conducted on a robotic arm equipped with a camera and the model of the environment corresponds to a top view image of an urban scene.

- A Memory of Motion for Visual Predictive Control Tasks

    Author: Paolillo, Antonio | Idiap Research Institute
    Author: Lembono, Teguh Santoso | Idiap Research Institute
    Author: Calinon, Sylvain | Idiap Research Institute
 
    keyword: Visual Servoing; Learning and Adaptive Systems

    Abstract : This paper addresses the problem of efficiently achieving visual predictive control tasks. To this end, a memory of motion, containing a set of trajectories built off-line, is used for leveraging precomputation and dealing with difficult visual tasks. Standard regression techniques, such as k-nearest neighbors and Gaussian process regression, are used to query the memory and provide on-line a warm-start and a way point to the control optimization process. The proposed technique allows the control scheme to achieve high performance and, at the same time, keep the computational time limited. Simulation and experimental results, carried out with a 7-axis manipulator, show the effectiveness of the approach.

## Soft Robot Materials and Design

- Designing Ferromagnetic Soft Robots (FerroSoRo) with Level-Set-Based Multiphysics Topology Optimization

    Author: Tian, Jiawei | The State University of New York at Stony Brook
    Author: Zhao, Xuanhe | MIT
    Author: Gu, Xianfeng | Stony Brook University
    Author: Chen, Shikui | State University of New York at Stony Brook
 
    keyword: Soft Robot Materials and Design; Dynamics

    Abstract : Soft active materials can generate flexible locomotion and change configurations through large deformations when subjected to an external environmental stimulus. They can be engineered to design �soft machines' such as soft robots, compliant actuators, flexible electronics, or bionic medical devices. By embedding ferromagnetic particles into soft elastomer matrix, the ferromagnetic soft matter can generate flexible movement and shift morphology in response to the external magnetic field. By taking advantage of this physical property, soft active structures undergoing desired motions can be generated by tailoring the layouts of the ferromagnetic soft elastomers. Structural topology optimization has emerged as an attractive tool to achieve innovative structures by optimizing the material layout within a design domain, and it can be utilized to architect ferromagnetic soft active structures. In this paper, the level-set-based topology optimization method is employed to design ferromagnetic soft robots (FerroSoRo). The objective function comprises a sub-objective function for the kinematics requirement and a sub-objective function for minimum compliance. Shape sensitivity analysis is derived using the material time derivative and adjoint variable method. Three examples, including a gripper, an actuator, and a flytrap structure, are studied to demonstrate the effectiveness of the proposed framework.

- Exoskeleton-Covered Soft Finger with Vision-Based Proprioception and Tactile Sensing

    Author: She, Yu | Massachusetts Institute of Technology
    Author: Liu, Sandra Q. | Massachusetts Institute of Technology
    Author: Yu, Peiyu | Tsinghua University
    Author: Adelson, Edward | MIT
 
    keyword: Soft Robot Materials and Design; Modeling, Control, and Learning for Soft Robots; Soft Robot Applications

    Abstract : Soft robots offer significant advantages in adapt-ability, safety, and dexterity compared to conventional rigid-body robots. However, it is challenging to equip soft robots with accurate proprioception and exteroception due to their high flexibility and elasticity. In this work, we describe the development of a vision-based proprioceptive and tactile sensor for soft robots called GelFlex. More specifically, we develop a novel exoskeleton-covered soft finger with embedded cameras and deep learning methods that enable high-resolution proprioceptive sensing and rich tactile sensing. To do so, we design features along the axial direction of the gripper, which enable high-resolution proprioceptive sensing, and incorporate a reflective ink coating on the surface of the gripper to enable rich tactile sensing. We design a highly underactuated exoskeleton with a tendon-driven mechanism to actuate the gripper. We then train neural networks for proprioception and shape classification using data from the embedded sensors. Finally, we perform a bar stock classification task, which requires both shape and tactile information. The accuracy for proprioception while grasping objects was within an accumulative positional and angle error of 2 mm and 5.5&#9702;. These proposed techniques offer soft robots the high-level ability to simultaneously perceive their proprioceptive state and peripheral environment, providing potential solutions for soft robots to solve everyday manipulation tasks.

- Tuning the Energy Landscape of Soft Robots for Fast and Strong Motion

    Author: Sun, Jiefeng | Colorado State University
    Author: Tighe, Brandon | Colorado State University
    Author: Zhao, Jianguo | Colorado State University
 
    keyword: Soft Robot Materials and Design; Soft Robot Applications; Compliant Joint/Mechanism

    Abstract : Soft robots demonstrate great potential compared with traditional rigid robots owing to their inherently soft body structures. Although researchers have made tremendous progress in recent years, existing soft robots are in general plagued by a main issue: slow speeds and small forces. In this work, we aim to address this issue by actively designing the energy landscape of the soft body: the total strain energy with respect to the robot's deformation. With such a strategy, a soft robot's dynamics can be tuned to have fast and strong motion. We introduce the general design principle using a soft module with two stable states that can rapidly switch from one state to the other under external forces. We characterize the required triggering (switching) force with respect to design parameters (e.g., the initial shape of the module). We then apply the soft bistable module to develop fast and strong soft robots, whose triggering forces are generated by a soft actuator -- twisted-and-coiled actuator (TCA). We demonstrate a soft gripper that can hold weights more than 8 times its own weight, and a soft jumping robot that can jump more than 5 times its body height. We envision our strategies will overcome the weakness of soft robots to unleash their potential for diverse applications.

- REBOund: Untethered Origami Jumping Robot with Controllable Jump Height

    Author: Carlson, Jaimie | University of Pennsylvania
    Author: Friedman, Jason | University of Pennsylvania
    Author: Kim, Christopher Yoon Jae | University of Pennsylvania
    Author: Sung, Cynthia | University of Pennsylvania
 
    keyword: Soft Robot Materials and Design; Modeling, Control, and Learning for Soft Robots; Soft Robot Applications

    Abstract : Origami robots are well-suited for jumping maneuvers because of their light weight and ability to incorporate actuation and control strategies directly into the robot body. However, existing origami robots often model fold patterns as rigidly foldable and fail to take advantage of the deformation in an origami sheet for potential energy storage. In this paper, we consider a parametric origami tessellation, the Reconfigurable Expanding Bistable Origami (REBO) pattern, which leverages face deformations to act as a nonlinear spring. We present a spring-based pseudo-rigid-body model for the REBO that calculates its energy when compressed to a given displacement and compare that model to experimental measurements taken on a mechanical testing system. This stored potential energy, when released quickly, can cause the pattern to jump. Using our model and experimental data, we design and fabricate a jumping robot, REBOund, that uses the spring-like REBO pattern as its body. Four lightweight servo motors with custom release mechanisms allow for quick compression and release of the origami pattern, allowing the fold pattern to jump over its own height even when carrying 5~times its own weight in electronics and power. We further demonstrate that small geometric changes to the pattern allow us to control the jump height without changing the actuation or control mechanism.

- Concentric Precurved Bellows: New Bending Actuators for Soft Robots

    Author: Childs, Jake | University of Tennessee, Knoxville
    Author: Rucker, Caleb | University of Tennessee
 
    keyword: Soft Robot Materials and Design; Modeling, Control, and Learning for Soft Robots

    Abstract : We present a new mechanical bending actuator for soft and continuum robots based on a pair of concentric precurved bellows. Each bellows is rotated axially at its base, allowing independent control of the curvature and bending plane of the concentric bellows pair. Rotation of precurved nested tubes is a well-known principle by which needle-sized concentric-tube robots operate, but the concept has never been scaled up to large diameters due to the trade-offs of increased actuation forces, decreased range of motion, strain limits, and torsional windup. In this letter, we show that using bellows structures instead of tubes allows two important breakthroughs: (1) actuation by rotation of precurved concentric elements can be achieved at much larger scales, and (2) torsional lag and instability are virtually eliminated due to the high ratio of torsional rigidity to flexural rigidity endowed by the bellows geometry. We discuss the development of two types of 3D printed concentric precurved bellows prototypes (revolute and helical), perform model parameter identification, and experimentally verify a torsionless mechanics model for the actuated shape which accounts for direction-dependent rigidities.

- Design of Deployable Soft Robots through Plastic Deformation of Kirigami Structures

    Author: Sedal, Audrey | University of Michigan
    Author: H. Memar, Amirhossein | Facebook Reality Labs
    Author: Liu, Tianshu | Facebook Reality Labs
    Author: Menguc, Yigit | Facebook Reality Labs
    Author: Corson, Nick | Oculus Research
 
    keyword: Soft Robot Materials and Design

    Abstract : Kirigami-patterned mechanisms are an emergent class of deployable structure that are easy to fabricate and offer the potential to be integrated into deployable robots. In this paper, we propose a design methodology for robotic kirigami structures that takes into consideration the deformation, loading, and stiffness of the structure under typical use cases. We show how loading-deformation behavior of a kirigami structure can be mechanically programmed by imposing plastic deformation. We develop a model for plasticity in the stretching of a kirigami structure. We show the creation of kirigami structures that have an increased elastic region, and specified stiffness, in their deployed states. We demonstrate the benefits of such a plastically-deformed structure by integrating it into a soft deployable crawling robot: the kirigami structure matches the stiffness of the soft actuator such that the deployed, coupled behavior serves to mechanically program the gait step size.

- Self-Excited Vibration Valve That Induces Traveling Waves in Pneumatic Soft Mobile Robots

    Author: Tsukagoshi, Hideyuki | Tokyo Institute of Technology
    Author: Miyaki, Yuji | Tokyo Institute of Technology
 
    keyword: Soft Sensors and Actuators

    Abstract : This paper presents a soft compact valve inducing self-excited vibration aiming at simplification of piping, power saving and downsizing the total system for pneumatic soft mobile robots generating traveling waves. The presented device, composed of flat tubes, resistant part, permanent magnet and holder, can induce the self-excited vibration from constant supplied pressure, enabling to switch the inner pressure of different chambers periodically. In this paper, we propose a new structure of the self-excited vibration valve, capable of switching more than three different chambers in order, which can detect the set pressure inside the chamber. Moreover, the developed valve is mounted on a flexible sheet-type mobile unit that is propelled by a traveling wave, and the verification of its effectiveness is discussed through the experimental results.

- A 1mm-Thick Miniatured Mobile Soft Robot with Mechanosensation and Multimodal Locomotion

    Author: Liu, Zemin | BeiHang University
    Author: Liu, Jiaqi | Beihang University
    Author: Wang, He | Beihang University
    Author: Yu, Xiao | Johns Hopkins University
    Author: Yang, Kang | Beihang University
    Author: Liu, Wenbo | Beihang University
    Author: Nie, Shilin | Beihang University
    Author: Sun, Wenguang | Beihang University
    Author: Xie, Zhexin | Beihang University
    Author: Chen, Bohan | Buaa
    Author: Liang, Shuzhang | Beihang University
    Author: Yingchun, Guan | Beihang University
    Author: Wen, Li | Beihang University
 
    keyword: Soft Robot Materials and Design; Micro/Nano Robots; Marine Robotics

    Abstract : The miniature soft robots have many promising applications, including micro-manipulations, endoscopy, and microsurgery, etc. Nevertheless, it remains challenging to fabricate a miniatured robot device that is thin, flexible, and can perform multimodal locomotor mobility with sensory capacity. In this study, we propose a miniatured, multi-layer (two shape memory polymer layers, a flexible copper heater, a silk particle enhanced actuator layer, and a sensory layer) four-limb soft robot (0.45-gram, 35mm-long, 12mm-wide) with a total thickness of 1mm. A precise flip-assembling technique is utilized to integrate multiple functional layers (fabricated by soft lithography, laser micromachining technologies). The actuator layer's elastic modulus increased ~100% by mixing with 20% silk particles by weight, which enhanced the mechanical properties of the miniature soft robot. We demonstrate that the soft robot can perform underwater crawling and jumping-gliding locomotion. The sensing data depicts the robot's multiple bending configurations after the sensory data been processed by the microprocessor mounted on the robot torso. The miniatured soft robot can also be reshaped to a soft miniatured gripper. The proposed miniatured soft robots can be helpful for studying soft organisms' body locomotion as well as medical applications in the future.

- Anisotropic Soft Robots Based on 3D Printed Meso-Structured Materials: Design, Modeling by Homogenization and Simulation

    Author: Vanneste, F�lix | INRIA
    Author: Goury, Olivier | Inria - Lille Nord Europe
    Author: Martínez, Jon's | INRIA
    Author: Lefebvre, Sylvain | Inria
    Author: Delingette, Herve | INRIA
    Author: Duriez, Christian | INRIA
 
    keyword: Soft Robot Materials and Design; Additive Manufacturing

    Abstract : In this paper, we propose to use new 3D-printed meso-structured materials to build soft robots and we present a modeling pipeline for design assistance and control. These meta-materials can be programmed before printing to target specific mechanical properties, in particular heterogeneous stiffness and anisotropic behaviour. Without changing the external shape, we show that using such meta-material can lead to a dramatic change in the kinematics of the robot. This highlights the importance of modeling. Therefore, to help the design and to control soft robots made of these meso-structured materials, we present a modeling method based on numerical homogenization and Finite Element Method (FEM) that captures the anisotropic deformations. The method is tested on a 3 axis parallel soft robot initially made of silicone. We demonstrate the change in kinematics when the robot is built with meso-structured materials and compare its behavior with modeling results.

- 3D-Printed Electroactive Hydraulic Valves for Use in Soft Robotic Applications

    Author: Bira, Nicholas | Oregon State University
    Author: Menguc, Yigit | Oregon State University
    Author: Davidson, Joseph | Oregon State University
 
    keyword: Modeling, Control, and Learning for Soft Robots; Soft Robot Applications; Hydraulic/Pneumatic Actuators

    Abstract : Soft robotics promises developments in the research areas of safety, bio-mimicry, manipulation, human-robot interaction, and alternative locomotion techniques. The research presented here is directed towards developing an improved, low-cost, and open-source method for soft robotic control using electrorheological fluids in compact, 3D-printed electroactive hydraulic valves. We construct high-pressure electrorheological valves and deformable actuators using only commercially available materials and accessible fabrication methods. The printed valves were characterized with industrial-grade electrorheological fluid (RheOil 3.0), but the design is generalizable to other electrorheological fluids. Valve performance was shown to be an improvement over comparable work with demonstrated higher yield pressures at lower voltages (up to 230 kPa), larger flow rates (up to 15 ml/min) and lower response times (1 to 3 seconds, depending on design). The resulting valve and actuator systems enable future novel applications of electrorheological fluid-based control and hydraulics in soft robotics and other disciplines.

- Design and Workspace Characterisation of Malleable Robots

    Author: Clark, Angus Benedict | Imperial College London
    Author: Rojas, Nicolas | Imperial College London
 
    keyword: Soft Robot Materials and Design; Flexible Robots; Mechanism Design

    Abstract : For the majority of tasks performed by traditional serial robot arms, such as bin picking or pick and place, only two or three degrees of freedom (DOF) are required for motion; however, by augmenting the number of degrees of freedom, further dexterity of robot arms for multiple tasks can be achieved. Instead of increasing the number of joints of a robot to improve flexibility and adaptation, which increases control complexity, weight, and cost of the overall system, malleable robots utilise a variable stiffness link between joints allowing the relative positioning of the revolute pairs at each end of the link to vary, thus enabling a low DOF serial robot to adapt across tasks by varying its workspace. In this paper, we present the design and prototyping of a 2-DOF malleable robot, calculate the general equation of its workspace using a parameterisation based on distance geometry-suitable for robot arms of variable topology, and characterise the workspace categories that the end effector of the robot can trace via reconfiguration. Through the design and construction of the malleable robot we explore design considerations, and demonstrate the viability of the overall concept. By using motion tracking on the physical robot, we show examples of the infinite number of workspaces that the introduced 2-DOF malleable robot can achieve.

- A Tri-Stable Soft Robotic Finger Capable of Pinch and Wrap Grasps

    Author: Nguyen, Aaron Khoa | University of California, Santa Barbara
    Author: Russell, Alexander | University of California, Santa Barbara
    Author: Vuong, Vu | University of California, Santa Barbara
    Author: Naclerio, Nicholas | University of California, Santa Barbara
    Author: Huang, Heming | UCSB
    Author: Chui, Kenny | University of California, Santa Barbara
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
 
    keyword: Soft Robot Materials and Design; Soft Robot Applications; Modeling, Control, and Learning for Soft Robots

    Abstract : Soft robotic pneumatic grippers have been shown to be versatile, robust to impacts, and safe for use on delicate objects. One type, fluidic elastomer grippers, are characterized by fingers with an inextensible gripping surface backed by extensible pneumatic chambers; when inflated, this mismatch in extensibility results in the finger curling. However, one drawback of these simple fingers is that they have one pre-programmed grasp, usually a simple constant-curvature wrap. While well-suited for finger-sized round objects, they do not grasp flat or small objects well. Here, we present an adaptable tri-stable soft robotic finger that can form either a pinch or wrap grasp based on the shape of the grasped object. We enable this by incorporating two bi-stable springs into the inextensible layer. The three stable positions are: i) open (unpressurized), ii) pinch (with only the proximal section bending), and iii) wrap (with the entire finger bending). We present a simple model of the behavior of our finger and experimental results verifying the model. Further, we apply forces and moments to grasped objects, and show that the tri-stable finger increases the grasping performance when compared to a control gripper with equal gripping force. Our work presents a novel design modification that is unobtrusive, simple, and passive. Our introduction of inexpensive programmable hardware advances the versatility and adaptability of soft grippers.

- A Dexterous Tip-Extending Robot with Variable-Length Shape-Locking

    Author: Wang, Sicheng | University of California, Santa Barbara
    Author: Zhang, Ruotong | Xi'an Jiaotong University
    Author: Haggerty, David Arthur | UC Santa Barbara
    Author: Naclerio, Nicholas | University of California, Santa Barbara
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
 
    keyword: Soft Robot Materials and Design; Soft Robot Applications; Flexible Robots

    Abstract : Soft, tip-extending "vine" robots offer a uniqueability for inspection	and manipulation in highly	constrainedenvironments. It is desirable that the distal end of the robotcan be manipulated freely, while	the body remains stationary.However, in previous vine robots, either the shape of the bodywas fixed after growth with no ability to manipulate the distalend, or the whole body moved together with the tip. Here, wepresent a concept for shape-locking that enables a vine robotto move only its distal end, while the rest of the body is lockedin place. This is	achieved using two inextensible, pressurized,tip-extending,	chambers that "grow" along the sides of therobot body, preserving curvature in the	section where	theyhave been deployed. The length of the locked and free sectionscan be varied by	 controlling the extension and retractionof	these chambers. We present models describing the shape-locking mechanism and workspace	of the robot	in both freeand constrained environments. We experimentaly validate thesemodel and show an increased dexterous workspace compared toprevious vine robots, as well as decreased contact forces of theproximal end with the environment. Our shape-locking concept allows improved	performance for vine robots, advancing the field of soft robotics for inspection and manipulation in highly constrained environments.

- Compliant Electromagnetic Actuator Architecture for Soft Robotics

    Author: Kohls, Noah | Georgia Institute of Technology
    Author: Dias, Beatriz | Georgia Institute of Technology
    Author: Mensah, Yaw | University of Tennessee - Knoxville
    Author: Ruddy, Bryan P. | University of Auckland
    Author: Mazumdar, Yi | Georgia Institute of Technology
 
    keyword: Soft Robot Materials and Design; Flexible Robots; Soft Sensors and Actuators

    Abstract : Soft materials and compliant actuation concepts have generated new design and control approaches in areas from robotics to wearable devices. Despite the potential of soft robotic systems, most designs currently use hard pumps, valves, and electromagnetic actuators. In this work, we take a step towards fully soft robots by developing a new compliant electromagnetic actuator architecture using gallium-indium liquid metal conductors, as well as compliant permanent magnetic and compliant iron composites. Properties of the new materials are first characterized and then co-fabricated to create an exemplary biologically-inspired soft actuator with pulsing or grasping motions, similar to Xenia soft corals. As current is applied to the liquid metal coil, the compliant permanent magnetic tips on passive silicone arms are attracted or repelled. The dynamics of the robotic actuator are characterized using stochastic system identification techniques and then operated at the resonant frequency of 7 Hz to generate high-stroke (&gt; 6 mm) motions.

- Dynamically Reconfigurable Discrete Distributed Stiffness for Inflated Beam Robots

    Author: Do, Brian | Stanford University
    Author: Banashek, Valory | Stanford University
    Author: Okamura, Allison M. | Stanford University
 
    keyword: Soft Robot Materials and Design; Mechanism Design; Compliant Joint/Mechanism

    Abstract : Inflated continuum robots are promising for a variety of navigation tasks, but controlling their motion with a small number of actuators is challenging. These inflated beam robots tend to buckle under compressive loads, producing extremely tight local curvature at difficult-to-control buckle point locations. In this paper, we present an inflated beam robot that uses distributed stiffness changing sections enabled by positive pressure layer jamming to control or prevent buckling. Passive valves are actuated by an electromagnet carried by an electromechanical device that travels inside the main inflated beam robot body. The valves themselves require no external connections or wiring, allowing the distributed stiffness control to be scaled to long beam lengths. Multiple layer jamming elements are stiffened simultaneously to achieve global stiffening, allowing the robot to support greater cantilevered loads and longer unsupported lengths. Local stiffening, achieved by leaving certain layer jamming elements unstiffened, allows the robot to produce ``virtual joints" that dynamically change the robot kinematics. Implementing these stiffening strategies is compatible with growth through tip eversion and tendon-steering, and enables a number of new capabilities for inflated beam robots and tip-everting robots.

- Retraction of Soft Growing Robots without Buckling

    Author: Coad, Margaret M. | Stanford University
    Author: Thomasson, Rachel | University of California, Berkeley
    Author: Blumenschein, Laura | Stanford University
    Author: Usevitch, Nathan | Stanford
    Author: Hawkes, Elliot Wright | University of California, Santa Barbara
    Author: Okamura, Allison M. | Stanford University
 
    keyword: Soft Robot Materials and Design; Modeling, Control, and Learning for Soft Robots

    Abstract : Tip-extending soft robots that "grow" via pneumatic eversion of their body material have demonstrated applications in exploration of cluttered environments. During growth, the motion and force of the robot tip can be controlled in three degrees of freedom using actuators that direct the tip in combination with extension. However, when reversal of the growth process is attempted by retracting the internal body material from the base, the robot body often responds by buckling rather than inversion of its body material, which makes control of tip motion and force impossible. We present and validate a model to predict when buckling occurs instead of inversion during retraction, and we present and evaluate an electromechanical device that can be added to a tip- extending soft robot to prevent buckling during retraction and enable control of all three degrees of freedom of tip actuation during inversion. Using our retraction device, we demonstrate completion of three previously impossible tasks: exploring different branches of a forking path, reversing growth while applying minimal force on the environment, and bringing back environment samples to the base.

## Rehabilitation Robotics

- Motion Intensity Extraction Scheme for Simultaneous Recognition of Wrist/Hand Motions

    Author: Kim, Minjae | KIST
    Author: Chung, Wan Kyun | POSTECH
    Author: Kim, Keehoon | POSTECH, Pohang University of Science and Technology
 
    keyword: Rehabilitation Robotics; Recognition

    Abstract : Surface electromyography contains muscular information representing gestures and corresponding forces. However, conventional sEMG-based motion recognition methods, such as pattern classification and regression, have intrinsic limitations due to the complex characteristics of sEMG signals. In this paper, motion intensity, a highly selective sEMG feature proportional to the level of muscle contraction, is proposed. The motion intensity feature allows proportional and simultaneous recognition of multiple degrees of freedom. The proposed method was demonstrated in terms of simultaneous recognition of wrist/hand motions. The result shows that the proposed method can successfully decompose sEMG signals into highly selective signals to target motions. In future works, the proposed method will be adapted for more subjects and to sEMG applications for practical evaluation considering various grasping motions.

- Simultaneous Online Motion Discrimination and Evaluation of Whole-Body Exercise by Synergy Probes for Home Rehabilitation

    Author: Moreira Ramos, Felipe | Tohoku University
    Author: Hayashibe, Mitsuhiro | Tohoku University
 
    keyword: Rehabilitation Robotics; Human Detection and Tracking; Neurorobotics

    Abstract : The development of algorithms for motion discrimination in home rehabilitation sessions poses numerous challenges. Recent studies have used the concept of synergies to discriminate a set of movements. However, the discrimination depends on the correlation of the reconstructed movement with the online data, and the training data requires well-defined movements. In this paper, we introduced the concept of a synergy probe, which makes a direct comparison between synergies and online data. The system represents synergies and movements in the same space and monitors their behavior. The results indicated that conventional methods are influenced by the segmentation of training data, and even though the reconstructed movement is similar to the ground-truth, it does not provide sufficient information to evaluate the data in real time. The synergy probes were used to discriminate and evaluate the performance of natural whole-body exercises without segmentation or previous determination of movements. An analysis of the results also demonstrated the possibility to identify the strategies used by the subjects for movement. Such information aids in gaining a better insight and can prove beneficial in home rehabilitation.

- IART: Learning from Demonstration for Assisted Robotic Therapy Using LSTM

    Author: Pareek, Shrey | University of Illinois at Urbana Champaign
    Author: Kesavadas, Thenkurussi | University of Illinois at Urbana-Champaign
 
    keyword: Rehabilitation Robotics; Learning from Demonstration; AI-Based Methods

    Abstract : In this paper, we present an intelligent Assistant for Robotic Therapy (iART), that provides robotic assistance during 3D trajectory tracking tasks. We propose a novel LSTM-based robot learning from demonstration (LfD) paradigm to mimic a therapist's assistance behavior. iART presents a trajectory agnostic LfD routine that can generalize learned behavior from a single trajectory to any 3D shape. Once the therapist's behavior has been learned, iART enables the patient to modify this behavior as per their preference. The system requires only a single demonstration of 2 minutes and exhibits a mean accuracy of 91.41% in predicting, and hence mimicking a therapist's assistance behavior. The system delivers stable assistance in realtime and successfully reproduces different types of assistance behaviors.

- Validation of a Forward Kinematics Based Controller for a Mobile Tethered Pelvic Assist Device to Augment Pelvic Forces During Walking

    Author: Stramel, Danielle | Columbia University in the City of New York
    Author: Agrawal, Sunil | Columbia University
 
    keyword: Physically Assistive Devices; Rehabilitation Robotics; Human-Centered Robotics

    Abstract : For those with irregular gait, re-calibration of motor control strategies and retraining of coordination are key goals. Thoughtful external forces or resistances during repetitive tasks can reprogram motor control patterns and strategies. Prior work in our lab has utilized this theory to improve gait in various patient groups using the Tethered Pelvic Assist Device (TPAD), a treadmill-based robotic trainer. In this paper, we propose a new, portable extension of the TPAD, which relies on an open-loop, forward kinematics based controller to remove the restriction of walking in the laboratory on a treadmill, and therefore accommodates overground ambulation. To evaluate the effects of this new control scheme and the effects of the users holding the mobile TPAD frame, a dataset of walking in four conditions was collected from eight healthy individuals. When applying a constant pelvic loading force of 10% body weight, the mean ground reaction force increased by 8.2�7.7% when the individual holds the walker frame and 11.1 - 7.8% when no hand contact is made. The mobile TPAD was shown to still induce a targeted loading on individuals during treadmill walking. The validation of this mobile device's controller and characterization of holding the frame allow overground studies to be conducted, and now opens the door to new training paradigms for overground gait training.

- Temporal Muscle Synergy Features Estimate Effects of Short-Term Rehabilitation in Sit-To-Stand of Post-Stroke Patients

    Author: Yang, Ningjia | The University of Tokyo
    Author: An, Qi | The University of Tokyo
    Author: Kogami, Hiroki | The University of Tokyo
    Author: Yamakawa, Hiroshi | The University of Tokyo
    Author: Tamura, Yusuke | The University of Tokyo
    Author: Takahashi, Kouji | Morinomiya Hospital
    Author: Kinomoto, Makoto | Morinomiya Hospital
    Author: Yamasaki, Hiroshi | BSI-TOYOTA Collaboration Center in the Nagoya Science Park Resea
    Author: Itkonen, Matti | RIKEN
    Author: Alnajjar, Fady | United Arab Emirates University,
    Author: Shimoda, Shingo | RIKEN
    Author: Hattori, Noriaki | Morinomiya Hospital
    Author: Fujii, Takanori | Morinomiya Hospital
    Author: Otomune, Hironori | Takanorifujii19@yahoo.co.jp
    Author: Miyai, Ichiro | Morinomiya Hospital
    Author: Yamashita, Atsushi | The University of Tokyo
    Author: Asama, Hajime | The University of Tokyo
 
    keyword: Rehabilitation Robotics; Physically Assistive Devices

    Abstract : Sit-to-stand (STS) motion is an important daily activity and many post-stroke patients have difficulty in performing the STS motion. Post-stroke patients who can perform STS independently, still utilize four muscle synergies (synchronized muscle activation) as seen in healthy people. In addition, temporal muscle synergy features can reflect motor impairment of post-stroke patients. However, it has been unclear whether post-stroke patients improve their STS movements in short-term rehabilitation and which muscle synergy features can estimate this improvement. Here, we demonstrate that temporal features of muscle synergies which contribute to body extension and balance maintenance can estimate the effect of short-term rehabilitation based on machine learning methods. By analyzing muscle synergies of post-stroke patients (n=33) before and with the intervention of physical therapists, we found that about half of the patients who were severely impaired, improved activation timing of muscle synergy to raise the hip with the intervention. Additionally, we identified the temporal features that can estimate whether severely impaired post-stroke patients improve. We conclude that temporal features of muscle synergies can estimate the motor recovery in short-term rehabilitation of post-stroke patients. This finding may lead to new rehabilitation strategies for post-stroke patients that focus on improving activation timing of different muscle synergies.

- Model Learning for Control of a Paralyzed Human Arm with Functional Electrical Stimulation

    Author: Wolf, Derek | Cleveland State University
    Author: Hall, Zinnia | University of Connecticut
    Author: Schearer, Eric | Cleveland State University
 
    keyword: Rehabilitation Robotics; Model Learning for Control; Physically Assistive Devices

    Abstract : Functional electrical stimulation (FES) is a promising technique for restoring reaching ability to individuals with tetraplegia. To this point, the complexities of goal-directed reaching motions and the shoulder-arm complex have prevented the realization of this potential in full-arm 3D reaching tasks. We trained a Gaussian process regression model to form the basis of a feedforward-feedback control structure capable of achieving reaching motions with a paralyzed upper limb. Over a series of 95 reaches of at least 10 cm in length, the controller achieved an average accuracy (measured by the Euclidean distance of the wrist to the final target position) of 3.8 cm and an average error along the path of 3.5 cm. This controller is the first demonstration of an accurate, complete-arm, FES-driven 3D reaching controller to be implemented with an individual with tetraplegia.

- Patient-Specific, Voice-Controlled, Robotic FLEXotendon Glove-II System for Spinal Cord Injury

    Author: Tran, Phillip | Georgia Institute of Technology
    Author: Jeong, Seokhwan | Georgia Institute of Technology
    Author: Wolf, Steven | Emory University School of Medicine
    Author: Desai, Jaydev P. | Georgia Institute of Technology
 
    keyword: Rehabilitation Robotics; Medical Robots and Systems; Soft Robot Applications

    Abstract : Reduced hand function in spinal cord injury (SCI) patients is commonly associated with a lower quality of life and limits the autonomy of the patient because he/she cannot perform most tasks independently. Robotic rehabilitation exoskeletons have been introduced as a method for assisting in hand function restoration. In this work, we propose a voice-controlled, tendon-actuated soft exoskeleton for improving hand function rehabilitation. The exoskeleton is constructed from soft materials to conform to the user's hand for improved fit and flexibility. A partially biomimetic tendon routing strategy independently actuates the index finger, middle finger, and thumb for a total of 4 degrees-of-freedom of the overall system. Nitinol wires are used for passive finger extension and screw-guided twisted tendon actuators are used for active finger flexion to create a compact, lightweight actuation mechanism. A continuous voice control strategy is implemented to provide a hands-free control interface and a simplified user interface experience while retaining distinct user intention. The exoskeleton was evaluated in a case study with a spinal cord injury patient. The patient used the exoskeleton and completed range-of-motion measurement as well as hand function tests, including the Box and Block Test and Jebsen-Taylor Hand Function Test.

- Integration of Self-Sealing Suction Cups on the FLEXotendon Glove-II Robotic Exoskeleton System

    Author: Jeong, Seokhwan | Georgia Institute of Technology
    Author: Tran, Phillip | Georgia Institute of Technology
    Author: Desai, Jaydev P. | Georgia Institute of Technology
 
    keyword: Rehabilitation Robotics; Medical Robots and Systems; Soft Robot Applications

    Abstract : This paper presents a hand exoskeleton using self-sealing suction cup modules to assist and simplify various grasping tasks. Robotic hands, grippers, and hand rehabilitation exoskeletons require complex motion planning and control algorithms to manipulate various objects, which increases system complexity. The proposed hand exoskeleton integrated with self-sealing suction cup modules provides simplified grasping with the assistance of suction. The suction cup has a self-sealing mechanism with a passive opening valve and it reduces vacuum consumption and pump noise. The gimbal mechanism allows the suction cup to have a wide range of contact angles, which increases adaptability of grasping of the exoskeleton. The fabrication process of the device is introduced with the suction cup design and material selection. The vacuum canister and solenoid valve that comprise the proposed pneumatic circuit provide a continuous vacuum supply source without continuous operation of a vacuum pump and autonomous suction/release motion, respectively. The performance of the hand exoskeleton was demonstrated with various grasping tasks and it provided stable grasping and pick-and-place task without complex finger manipulation. The proposed hand exoskeleton has the potential to simplify the grasping process and allow patients with hand dysfunction to expand their versatility of grasping tasks.

- A Novel End-Effector Robot System Enabling to Monitor Upper-Extremity Posture During Robot-Aided Reaching Movements

    Author: Hwang, Yeji | DGIST
    Author: Lee, Seongpung | DGIST
    Author: Hong, Jaesung | DGIST
    Author: Kim, Jonghyun | Sungkyunkwan University
 
    keyword: Rehabilitation Robotics; Sensor Fusion; Human Detection and Tracking

    Abstract : End-effector type robots have been popularly applied to robot-aided therapy for rehabilitation purpose. However, those robots have a key drawback for the purpose: lack of the user's posture (joint angle) information. This paper proposes a novel end-effector rehabilitation robot system that contains a contactless motion sensor to monitor upper- extremity posture during robot-aided reaching exercise. The sensor allows the posture estimation without complicated procedures but has an inaccuracy problem such as occlusion and an unreliable segment length. Therefore, we developed a posture monitoring method, which is an analytical method without training procedure, based on the combined use of the information obtained from the sensor and the robot. Eight healthy subjects participated in the experiment with planar reaching exercise for validation. The results of joint angle estimation, high correlation coefficient (0.95 - 0.03) and small errors (3.55 - 0.70 deg), show that the proposed system can provide affordable upper-extremity posture estimation.

- Optimal Design of a Novel 3-DOF Orientational Parallel Mechanism for Pelvic Assistance on a Wheelchair: An Approach Based on Kinematic Geometry and Screw Theory

    Author: Ophaswongse, Chawin | Columbia University
    Author: Agrawal, Sunil | Columbia University
 
    keyword: Physically Assistive Devices; Parallel Robots; Mechanism Design

    Abstract : Pelvis mobility is essential to the daily seated activities of wheelchair users, however it is not yet fully addressed in the field of active wearable devices. This paper presents a novel design and optimization methodology of an in-parallel actuated robotic brace for assisting the human pelvis during seated maneuvers on wheelchair. This design uses human data captured by cameras in conjunction with the knowledge of kinematic geometry and screw theory. The mechanism has full rotational three degrees-of-freedom (DOFs) and also accommodates coupled translation of the human pelvic segment. This type of motion was realized by employing three kinematic limbs that impose non-intersecting zero-pitch constraint wrenches on the platform. Our multi-objective optimization (MOO) routine consists of two stages: (I) platform constraint synthesis, where the geometric parameters of the limb constraints were determined to minimize the pelvis-platform errors of trajectories and instantaneous screw axes (ISAs); and (II) limb structural synthesis, where limb types and dimensions, workspace, transmission performances, singularities, and actuated joint displacements were considered. The optimized mechanism has an asymmetrical [RRR]U-2[RR]S architecture. This mechanism can also be integrated into our previously developed Wheelchair Robot for Active Postural Support (WRAPS).

- Using Human Ratings for Feedback Control: A Supervised Learning Approach with Application to Rehabilitation Robotics (I)

    Author: Menner, Marcel | ETH Zurich
    Author: Neuner, Lukas | ETH Zurich
    Author: Lunenburger, Lars | Hocoma AG
    Author: Zeilinger, Melanie N. | ETH Zurich
 
    keyword: Rehabilitation Robotics; Learning and Adaptive Systems

    Abstract : This paper presents a method for tailoring a parametric controller based on human ratings. The method leverages supervised learning concepts in order to train a reward model from data. It is applied to a gait rehabilitation robot with the goal of teaching the robot how to walk patients physiologically. In this context, the reward model judges the physiology of the gait cycle (instead of therapists) using sensor measurements provided by the robot and the automatic feedback controller chooses the input settings of the robot to maximize the reward. The key advantage of the proposed method is that only a few input adaptations are necessary to achieve a physiological gait cycle. Experiments with non-disabled subjects show that the proposed method permits the incorporation of human expertise into a control law and to automatically walk patients physiologically.

- Compliant Humanoids Moving Toward Rehabilitation Applications: Transparent Integration of Real-Time Control, Whole-Body Motion Generation, and Virtual Reality (I)

    Author: Mohammadi, Pouya | Braunschweig University of Technology
    Author: Mingo, Enrico | Istituto Italiano Di Tecnologia
    Author: Dehio, Niels | Karlsruhe Institute of Technology
    Author: Malekzadeh, Milad S. | Technical University of Btraunschweig, IRP
    Author: Giese, Martin | University Clinic Tübingen
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
    Author: Steil, Jochen J. | Technische Universitét Braunschweig
 
    keyword: Rehabilitation Robotics; Software, Middleware and Programming Environments; Humanoid Robots

    Abstract : Humanoids fascinate through their anthropomorphic appearance, high dexterity and the potential to work in human-tailored environment and to interact with humans in novel applications. In our research, we promote two real-world applications in physiotherapeutic juggling and assisted walking using two compliant humanoids COMAN and COMAN+. We focus on rehabilitation, which as a result of changing demographics becomes an increasingly crucial application field. But alike most humanoid experiments, the realization of these scenarios is challenging due to the fact that the hardware is brittle, software is complex and control remains highly demanding. In this article, we describe an integrative and transparent control architecture that alleviates this complexity by strictly adhering in design and implementation to a component-based approach. It promotes flexibility and reusability and allows transparent switching between different humanoid robots, between simulation and real world, between control paradigms. It also orchestrates the integration of real-time and non-real-time components, including a Virtual Reality framework, towards rich user interaction.

- Data-Driven Reinforcement Learning for Walking Assistance Control of a Lower Limb Exoskeleton with Hemiplegic Patients

    Author: Peng, Zhinan | Unversity of Electronic Science and Tehcnology of China
    Author: Luo, Rui | Unversity of Electronic Science and Tehcnology of China
    Author: Huang, Rui | University of Electronic Science and Technology of China
    Author: Hu, Jiangping | University of Electronic Science and Technology of China
    Author: Shi, Kecheng | The School of Automation Engineering, University of Electronic S
    Author: Cheng, Hong | University of Electronic Science and Technology
    Author: Ghosh, Bijoy | Texas Tech University
 
    keyword: Rehabilitation Robotics; Robust/Adaptive Control of Robotic Systems; Optimization and Optimal Control

    Abstract : Lower limb exoskeleton (LLE) has gained considerable interests in both strength augmentation, rehabilitation and walking assistance scenarios. In walking assistance of hemiplegic patients, the LLE should have the ability to control the affected leg to track the unaffected leg's motion naturally. A critical issue in this scenario is that the exoskeleton system needs to deal with unpredictable disturbance from the patient, which lead the controller of exoskeleton system should have the ability to adapt different patients. This paper presents a novel Data-Driven Reinforcement Learning (DDRL) control strategy for adapting different hemiplegic patients with unpredictable disturbances. In the proposed DDRL strategy, the interaction between two lower limbs of LLE and the legs of hemiplegic patient are modeled as a leader-follower multi-agent system framework. The walking assistance control problem is transformed into a optimal control problem. Then, a policy iteration (PI) algorithm is proposed to obtain optimal controller. In order to achieve online adaptation control for different patients, based on PI algorithm, an Actor-Critic Neural Network (ACNN) technology of the reinforcement learning (RL) is employed in the proposed DDRL strategy. We conduct experiments both on a simulation environment and a real LLE system with healthy subjects. The experimental results demonstrate that the proposed control strategy have strong robustness against disturbances and ability to adapt to differ

- On the Effects of Visual Anticipation of Floor Compliance Changes on Human Gait: Towards Model-Based Robot-Assisted Rehabilitation

    Author: Michael, Drolet | Arizona State University
    Author: Qui�ones Yumbla, Emiliano | Arizona State University
    Author: Hobbs, Bradley | Arizona State University
    Author: Artemiadis, Panagiotis | University of Delaware
 
    keyword: Rehabilitation Robotics

    Abstract : The role of various types of robot assistance in post-stroke gait rehabilitation has gained much attention in recent years. Naturally, this results in the need to study how the different robot-assisted interventions affect the various underlying sensorimotor mechanisms involved in rehabilitation. To answer this important question, this paper combines a virtual reality experience with a unique robotic rehabilitation device, the Variable Stiffness Treadmill (VST), as a way of understanding interactions across different sensorimotor mechanisms involved in gait. The VST changes the walking surface stiffness in order to simulate real-world compliant surfaces while seamlessly interacting with a virtual environment. Through the manipulated visual and proprioceptive feedback, this paper focuses on the muscle activation patterns before, during, and after surface changes that are both visually informed and uninformed. The results show that there are predictable and repeatable muscle activation patterns both before and after surface stiffness changes, and these patterns are affected by the perceived visual and proprioceptive feedback. The interaction of feedback mechanisms and their effect on evoked muscular activation can be used in future robot-assisted gait therapies, where the intended muscle responses are informed by deterministic models and are tailored to a specific patient's needs.

- A Visual Positioning System for Indoor Blind Navigation

    Author: Zhang, He | Virginia Commonwealth University
    Author: Ye, Cang | Virginia Commonwealth University
 
    keyword: Rehabilitation Robotics; SLAM; Visual-Based Navigation

    Abstract : This paper presents a visual positioning system (VPS) for real-time pose estimation of a robotic navigation aid (RNA) for assistive navigation. The core of the VPS is a new method called depth-enhanced visual-inertial odometry (DVIO) that uses an RGB-D camera and an inertial measurement unit (IMU) to estimate the RNA pose. The DVIO method extracts the geometric feature (the floor plane) from the camera's depth data and integrates its measurement residuals with that of the visual features and the inertial data in a graph optimization framework for pose estimation. A new measure based on the Sampson error is introduced to describe the measurement residuals of the near-range visual features with a known depth and that of the far-range visual features whose depths are unknown. The measure allows for the incorporation of both types of visual features into graph optimization. The use of the geometric feature and the Sampson error improves pose estimation accuracy and precision. The DVIO method is paired with a particle filter localization (PFL) method to locate the RNA in a 2D floor plan and the information is used to guide a visually impaired person. The PFL reduces the RNA's position and heading error by aligning the camera's depth data with the floor plan map. Together, the DVIO and the PFL allow for accurate pose estimation for wayfinding and 3D mapping for obstacle avoidance. Experimental results demonstrate the usefulness of the RNA in assistive navigation in indoor spaces.

- An Outsole-Embedded Optoelectronic Sensor to Measure Shear Ground Reaction Forces During Locomotion

    Author: Duong, Ton | Stevens Institute of Technology
    Author: Whittaker, David R. | U.S. Navy Bureau of Medicine and Surgery (BUMED)
    Author: Zanotto, Damiano | Stevens Institute of Technology
 
    keyword: Rehabilitation Robotics; Prosthetics and Exoskeletons; Wearable Robots

    Abstract : Online estimation of 3D ground reaction forces (GRFs) is becoming increasingly important for closed-loop control of lower-extremity robotic exoskeletons. Through inverse dynamics and optimization models, 3D GRFs can be used to estimate net joint torques and approximate muscle forces. Although instrumented footwear to measure vertical GRFs in out-of-the-lab environments is available, accurately measuring shear GRFs with foot-mounted sensors still remains a challenging task. In this paper, a new outsole-embedded optoelectronic sensor configuration that is able to measure biaxial shear GRFs is proposed. Compared with traditional strain-gauge based solutions, optoelectronic sensors allow for a more affordable design. To mitigate the risk of altering the wearer's natural gait, the proposed solution does not involve external modifications to the footwear structure. A preliminary validation of the outsole-embedded sensor was conducted against validated laboratory equipment. The test involved two sessions of treadmill walking at different speeds. Experimental results suggest that the proposed design may be a promising solution for measuring shear GRFs in unconstrained environments.

- Bump�em: An Open-Source, Bump-Emulation System for Studying Human Balance and Gait

    Author: Tan, Guan Rong | Stanford University
    Author: Raitor, Michael | Stanford University
    Author: Collins, Steven H. | Stanford University
 
    keyword: Rehabilitation Robotics; Legged Robots; Force Control

    Abstract : Fall-related injury is a significant health problem on a global scale and is expected to grow with the aging population. Laboratory-based perturbation systems have the capability of simulating various modes of fall-inducing perturbations in a repeatable way. These systems enable fundamental research on human gait and balance and facilitate the development of devices to assist human balance. We present a robotic, rope-driven system capable of rendering bumps and force-fields at a person's pelvis in any direction in the transverse plane with forces up to 200 N, and a 90% rise time of as little as 44 ms, which is faster than a human's ability to sense and respond to the force. These capabilities enable experiments that require stabilizing or destabilizing subjects as they stand or walk on a treadmill. To facilitate use by researchers from all backgrounds, we designed both a configuration with simpler open-loop force control, and another with higher-performance, closed-loop force control. Both configurations are modular, and the open-loop system is made entirely from 3D-printed and catalog components. The design files and assembly instructions for both are freely available in an online repository.

- A Hybrid, Soft Exoskeleton Glove Equipped with a Telescopic Extra Thumb and Abduction Capabilities

    Author: Gerez, Lucas | The University of Auckland
    Author: Dwivedi, Anany | University of Auckland
    Author: Liarokapis, Minas | The University of Auckland
 
    keyword: Prosthetics and Exoskeletons; Rehabilitation Robotics

    Abstract : Over the last years, hand exoskeletons have become a popular and efficient technical solution for assisting people that suffer from neurological and musculoskeletal diseases and enhance the capabilities of healthy individuals. These devices can vary from rigid and complex structures to soft, lightweight, wearable gloves. Despite the significant progress in the field, most existing solutions do not provide the same dexterity as the healthy human hand. In this paper, we focus on the development of a hybrid (tendon-driven and pneumatic), lightweight, affordable, wearable exoskeleton glove equipped with abduction/adduction capabilities and a pneumatic telescopic extra thumb that increases grasp stability. The efficiency of the proposed device is experimentally validated through three different types of experiments: i) abduction/adduction tests, ii) force exertion experiments that capture the maximum forces that can be applied by the proposed device, and iii) grasp quality assessment experiments that focus on the effect of the inflatable thumb on enhancing grasp stability. The hybrid assistive glove considerably improves the grasping capabilities of the user, being able to exert the forces required to assist people in the execution of activities of daily living.

## Physical Human-Robot Interaction

- Transient Behavior and Predictability in Manipulating Complex Objects

    Author: Nayeem, Rashida | Northeastern University
    Author: Bazzi, Salah | Northeastern University
    Author: Hogan, Neville | Massachusetts Institute of Technology
    Author: Sternad, Dagmar | Northeastern University
 
    keyword: Physical Human-Robot Interaction; Biologically-Inspired Robots; Dexterous Manipulation

    Abstract :  Relatively little work in human and robot control has examined the control of complex objects with intrinsic dynamics, such as carrying a cup of coffee, a task that presents little problems for humans. This study examined how humans move a �cup-of-coffee� with a view to identify principles that may be useful for robot control. The specific focus was on how humans choose initial conditions to safely reach a steady state. We hypothesized that subjects choose initial conditions that minimized the transient duration to reach the steady state faster as it presented more predictable dynamics. In the experiment the cup of coffee was reduced to a 2-D cup with a sliding ball inside which was simulated in a virtual environment. Human subjects interacted with this virtual object via a robotic manipulandum that provided haptic feedback. Participants moved the cup between two targets without losing the ball; they were instructed to explore different initial conditions before initiating the continuous interaction. Results showed that subjects converged to a small set of initial conditions that decreased their transient durations and achieved a predictable steady state faster. Simulations with a simple feedforward controller and inverse dynamics calculations confirmed that these initial conditions indeed led to shorter transients and less complex interaction forces. These results may inform robot control of complex objects where the effects of initial conditions need further investigation.

- A Variable-Fractional Order Admittance Controller for PHRI

    Author: Sirintuna, Doganay | Koc University, College of Engineering, Turkey
    Author: Aydin, Yusuf | Koc University
    Author: Caldiran, Ozan | Koc University
    Author: Tokatli, Ozan | UKAEA
    Author: Patoglu, Volkan | Sabanci University
    Author: Basdogan, Cagatay | Koc University
 
    keyword: Physical Human-Robot Interaction; Cooperating Robots; Human-Centered Automation

    Abstract : In today's automation driven manufacturing environments, emerging technologies like cobots (collaborative robots) and augmented reality interfaces can help integrating humans into the production workflow to benefit from their adaptability and cognitive skills. In such settings, humans are expected to work with robots side by side and physically interact with them. However, the trade-off between stability and transparency is a core challenge in the presence of physical human robot interaction (pHRI). While stability is of utmost importance for safety, transparency is required for fully exploiting the precision and ability of robots in handling labor intensive tasks. In this work, we propose a new variable admittance controller based on fractional order control to handle this trade-off more effectively. We compared the performance of fractional order variable admittance controller with a classical admittance controller with fixed parameters as a baseline and an integer order variable admittance controller during a realistic drilling task. Our comparisons indicate that the proposed controller led to a more transparent interaction compared to the other controllers without sacrificing the stability. We also demonstrate a use case for an augmented reality (AR) headset which can augment human sensory capabilities for reaching a certain drilling depth otherwise not possible without changing the role of the robot as the decision maker.

- Assistive Gym: A Physics Simulation Framework for Assistive Robotics

    Author: Erickson, Zackory | Georgia Institute of Technology
    Author: Gangaram, Vamsee | Georgia Institute of Technology
    Author: Kapusta, Ariel | Georgia Institute of Technology
    Author: Liu, Karen | Georgia Tech
    Author: Kemp, Charlie | Georgia Institute of Technology
 
    keyword: Physical Human-Robot Interaction; Simulation and Animation; Physically Assistive Devices

    Abstract : Autonomous robots have the potential to serve as versatile caregivers that improve quality of life for millions of people worldwide. Yet, conducting research in this area presents numerous challenges, including the risks of physical interaction between people and robots. Physics simulations have been used to optimize and train robots for physical assistance, but have typically focused on a single task. In this paper, we present Assistive Gym, an open source physics simulation framework for assistive robots that models multiple tasks. It includes six simulated environments in which a robotic manipulator can attempt to assist a person with activities of daily living (ADLs): itch scratching, drinking, feeding, body manipulation, dressing, and bathing. Assistive Gym models a person's physical capabilities and preferences for assistance, which are used to provide a reward function. We present baseline policies trained using reinforcement learning for four different commercial robots in the six environments. We demonstrate that modeling human motion results in better assistance and we compare the performance of different robots. Overall, we show that Assistive Gym is a promising tool for assistive robotics research.

- Learning Whole-Body Human-Robot Haptic Interaction in Social Contexts

    Author: Campbell, Joseph | Arizona State University
    Author: Yamane, Katsu | Honda
 
    keyword: Physical Human-Robot Interaction; Learning from Demonstration; Social Human-Robot Interaction

    Abstract : This paper presents a learning-from-demonstration (LfD) framework for teaching human-robot social interactions that involve whole-body haptic interaction, i.e. direct human-robot contact over the full robot body. The performance of existing LfD frameworks suffers in such interactions due to the high dimensionality and spatiotemporal sparsity of the demonstration data. We show that by leveraging this sparsity, we can reduce the data dimensionality without incurring a significant accuracy penalty, and introduce three strategies for doing so. By combining these techniques with an LfD framework for learning multimodal human-robot interactions, we can model the spatiotemporal relationship between the tactile and kinesthetic information during whole-body haptic interactions. Using a teleoperated bimanual robot equipped with 61 force sensors, we experimentally demonstrate that a model trained with 121 sample hugs from 4 participants generalizes well to unseen inputs and human partners.

- Human Preferences in Using Damping to Manage Singularities During Physical Human-Robot Collaboration

    Author: Carmichael, Marc | Centre for Autonomous Systems
    Author: Khonasty, Richardo | Centre for Autonomous Systems
    Author: Aldini, Stefano | University of Technology Sydney
    Author: Liu, Dikai | University of Technology, Sydney
 
    keyword: Physical Human-Robot Interaction; Human Factors and Human-in-the-Loop; Human-Centered Automation

    Abstract : When a robot manipulator approaches a kinematic singular configuration, control strategies need to be employed to ensure safe and robust operation. If this manipulator is being controlled by a human through physical human-robot collaboration, the choice of strategy for handling singularities can have a significant effect on the feelings and impressions of the user. To date the preferences of humans during physical human-robot collaboration regarding strategies for managing kinematic singularities have yet to be thoroughly explored.<p>This work presents an empirical study of a damping-based strategy for handling singularities with regard to the preferences of the human operator. Two different parameters, damping rate and damping asymmetry, are tested using a double-blind A/B pairwise comparison testing protocol. Participants included two cohorts made up of the general public (n=51) and people working within a robotic research centre (n=18). In total 105 individual trials were performed. Results indicate a preference for a faster, asymmetric damping behavior that slows motions towards singularities whilst allowing for faster motions away.

- MOCA-MAN: A MObile and Reconfigurable Collaborative Robot Assistant for Conjoined huMAN-Robot Actions

    Author: Kim, Wansoo | Istituto Italiano Di Tecnologia
    Author: Balatti, Pietro | Istituto Italiano Di Tecnologia
    Author: Lamon, Edoardo | Istituto Italiano Di Tecnologia
    Author: Ajoudani, Arash | Istituto Italiano Di Tecnologia
 
    keyword: Physical Human-Robot Interaction; Human-Centered Robotics; Physically Assistive Devices

    Abstract : The objective of this paper is to create a new collaborative robotic system that subsumes the advantages of mobile manipulators and supernumerary limbs. By exploiting the reconfiguration potential of a MObile Collaborative robot Assistant (MOCA), we create a collaborative robot that can function autonomously, in close proximity to humans, or be physically coupled to the human counterpart as a supernumerary body (MOCA-MAN). Through an admittance interface and a hand gesture recognition system, the controller can give higher priority to the mobile base (e.g., for long distance co-carrying tasks) or the arm movements (e.g., for manipulating tools), when performing conjoined actions. The resulting system has a high potential not only to reduce waste associated with the equipment waiting and setup times, but also to mitigate the human effort when performing heavy or prolonged manipulation tasks. The performance of the proposed system, i.e., MOCA-MAN, is evaluated by multiple subjects in two different use-case scenarios, which require large mobility or close-proximity manipulation.

- Variable Damping Control of a Robotic Arm to Improve Trade-Off between Agility and Stability and Reduce User Effort

    Author: Bitz, Tanner | Arizona State University
    Author: Zahedi, Fatemeh | Arizona State University
    Author: Lee, Hyunglae | Arizona State University
 
    keyword: Physical Human-Robot Interaction; Physically Assistive Devices; Human Performance Augmentation

    Abstract : This paper presents a variable damping controller to improve the trade-off between agility and stability in physical human-robot interaction (pHRI), while reducing user effort. Variable robotic damping, defined as a dual-sided logistic function, was determined in real time throughout a range of negative to positive values based on the user's intent of movement. To evaluate the effectiveness of the proposed controller, we performed a set of human experiments with subjects interacting with the end-effector of a 7 degree-of-freedom robot. Twelve subjects completed target reaching tasks under three robotic damping conditions: fixed positive, fixed negative, and variable damping. On average, the variable damping controller significantly shortened the rise time by 22.4% compared to the fixed positive damping. It is also important to note that the rise time in the variable damping condition was as fast as that in the fixed negative damping condition. The variable damping controller significantly decreased the percentage overshoot by 49.6% and shortened the settling time by 29.0% compared to the fixed negative damping. Both the maximum and mean root-mean-squared (RMS) interaction forces were significantly lower in the variable damping condition than the other two fixed damping conditions, i.e., the variable damping controller reduced user effort. The maximum and mean RMS interaction forces were at least 17.3% and 20.3% lower than any of the fixed damping conditions, respectively.

- Robustness in Human Manipulation of Dynamically Complex Objects through Control Contraction Metrics

    Author: Bazzi, Salah | Northeastern University
    Author: Sternad, Dagmar | Northeastern University
 
    keyword: Physical Human-Robot Interaction; Biologically-Inspired Robots; Dexterous Manipulation

    Abstract : Control and manipulation of objects with underactuated dynamics remains a challenge for robots.Due to their typically nonlinear dynamics, it is computationally taxing to implement model-based planning and control techniques. Yet humans can skillfully manipulate such objects, seemingly with ease. More insight into human control strategies may inform how to enhance control strategies in robots. This study examined human control of objects that exhibit complex - underactuated and nonlinear - dynamics. We hypothesized that humans seek to make their trajectories exponentially stable to achieve robustness in the face of external perturbations. A stable trajectory is also robust to the high levels of noise in the human neuromotor system. Motivated by the task of carrying a cup of coffee, a virtual implementation of transporting a cart-pendulum system was developed. Subjects interacted with the virtual system via a robotic manipulandum that provided a haptic and visual interface. Human subjects were instructed to transport this simplified system to a target position as fast as possible without �spilling coffee,� while accommodating different visible perturbations that could be anticipated. To test the hypothesis of exponential convergence, tools fromthe framework of control contraction metrics were leveraged to analyze human trajectories. Results showed that with practice the trajectories indeed became exponentially stable, selectively around the perturbation.

- Cooperative Human-Robot Grasping with Extended Contact Patches

    Author: Marullo, Sara | University of Siena
    Author: Pozzi, Maria | University of Siena
    Author: Prattichizzo, Domenico | University of Siena
    Author: Malvezzi, Monica | University of Siena
 
    keyword: Physical Human-Robot Interaction; Contact Modeling; Grasping

    Abstract : Grasping large and heavy objects with a robot assistant is one of the most interesting scenarios in physical Human-Robot Interaction. Many solutions have been proposed in the last 40 years, focusing not only on human safety, but also on comfort and ergonomics. When carrying objects with large planar surfaces, classical contact models developed in robotic grasping cannot be used. This is why we conceived a contact model explicitly considering the contact area properties and thus suitable to deal with grasps requiring large contact patches from the robot side. Together with the model, this work proposes a decentralized control strategy to implement cooperative object handling between humans and robots. Experimental results showed that the proposed method performs better than a simpler one, which does not account for contact patch properties. The control strategy is decentralized and needs minimal exteroceptive sensing (force/torque sensor at the robot wrist), so the proposed approach can easily be generalized to human-robot teams composed of more than two agents.

- The InSight Crutches: Analyzing the Role of Arm Support During Robot-Assisted Leg Movements (I)

    Author: Haufe, Florian Leander | ETH Zurich
    Author: Haji Hassani, Roushanak | University of Basel
    Author: Riener, Robert | ETH Zurich
    Author: Wolf, Peter | ETH Zurich, Institute of Robotics and Intelligent Systems
 
    keyword: Physical Human-Robot Interaction; Rehabilitation Robotics; Prosthetics and Exoskeletons

    Abstract : To complement the assistance of a wearable robot, users with a leg weakness often rely on balance and body-weight support through their arms and passive walking aids. A precise quantification of this arm support is crucial to better understand the real-world robot dynamics, the human-robot interaction, and the human user performance. In this article, we present a novel measurement system, the InSight Crutches, that allows such a quantification, and evaluate the crutches' functionality in three exemplary movement scenarios with different wearable robots and users with spinal cord injury.

- Cognitive and Motor Compliance in Intentional Human-Robot Interaction

    Author: Chame, Hendry | Okinawa Institute of Science and Technology Graduate University
    Author: Tani, Jun | Okinawa Institute of Science and Technology
 
    keyword: Neurorobotics; Physical Human-Robot Interaction; Humanoid Robots

    Abstract : Embodiment and subjective experience in human-robot interaction are important aspects to consider when studying both natural cognition and adaptive robotics to human environments. Although several researches have focused on nonverbal communication and collaboration, the study of autonomous physical interaction has obtained less attention. From the perspective of neurorobotics, we investigate the relation between intentionality, motor compliance, cognitive compliance, and behavior emergence. We propose a variational model inspired by the principles of predictive coding and active inference to study intentionality and cognitive compliance, and an intermittent control concept for motor deliberation and compliance based on torque feed-back. Our experiments with the humanoid Torobo portrait interesting perspectives for the bio-inspired study of developmental and social processes.

- Controlling an Upper-Limb Exoskeleton by EMG Signal While Carrying Unknown Load

    Author: Treussart, Benjamin | Cea-List, Diasi, Lri
    Author: Geffard, Franck | Atomic Energy Commissariat (CEA)
    Author: Vignais, Nicolas | Univ. Paris-Sud
    Author: Marin, Fr�d�ric | UTC
 
    keyword: Physical Human-Robot Interaction; Physically Assistive Devices; Prosthetics and Exoskeletons

    Abstract : Implementing an intuitive control law for an upper-limb exoskeleton dedicated to force augmentation is a challenging issue in the field of human-robot collaboration. The aim of this study is to design an innovative approach to assist carrying an unknown load. The method is based on user's intentions estimated through a wireless EMG armband allowing movement direction and intensity estimation along 1 Degree of Freedom. This control law aimed to behave like a gravity compensation except that the mass of the load does not need to be known. The proposed approach is tested by 10 participants on a lifting task with a single Degree of Freedom upper-limb exoskeleton. Participants perform it in three different conditions : without assistance, with an exact gravity compensation and with the proposed method based on EMG armband (Myo Armband). The evaluation of the efficiency of the assistance is based on EMG signals captured on seven muscles (objective indicator) and a questionnaire (subjective indicator). Results shows a statically significant reduction of mean activity of the biceps, erector spinae and deltoid by 20+/-14%, 18+/-12% and 25+/-16% respectively while comparing the proposed method with no assistance. In addition, similar muscle activities are found both in the proposed method and the traditional gravity compensation. Subjective evaluation shows better precision, efficiency and responsiveness of the proposed method compared to the traditional one.

- Learning Grasping Points for Garment Manipulation in Robot-Assisted Dressing

    Author: Zhang, Fan | Imperial College London
    Author: Demiris, Yiannis | Imperial College London
 
    keyword: Physical Human-Robot Interaction; Perception for Grasping and Manipulation; Human-Centered Robotics

    Abstract : Assistive robots have the potential to provide tremendous support for disabled and elderly people in their daily dressing activities. Recent studies on robot-assisted dressing usually simplify the setup of the initial robot configuration by manually attaching the garments on the robot end-effector and positioning them close to the user's arm. A fundamental challenge in automating such a process for robots is computing suitable grasping points on garments that facilitate robotic manipulation. In this paper, we address this problem by introducing a supervised deep neural network to locate a pre-defined grasping point on the garment, using depth images for their invariance to color and texture. To reduce the amount of real data required, which is costly to collect, we leverage the power of simulation to produce large amounts of labeled data. The network is jointly trained with synthetic datasets of depth images and a limited amount of real data. We introduce a robot-assisted dressing system that combines the grasping point prediction method, with a grasping and manipulation strategy which takes grasping orientation computation and robot-garment collision avoidance into account. The experimental results demonstrate that our method is capable of yielding accurate grasping point estimations. The proposed dressing system enables the Baxter robot to autonomously grasp a hospital gown hung on a rail, bring it close to the user and successfully dress the upper-body.

- TACTO-Selector: Enhanced Hierarchical Fusion of PBVS with Reactive Skin Control for Physical Human-Robot Interaction

    Author: Huezo Martin, Ana Elvira | Technische Universitét M�nchen
    Author: Dean-Leon, Emmanuel | Technischen Universitaet Muenchen
    Author: Cheng, Gordon | Technical University of Munich
 
    keyword: Physical Human-Robot Interaction; Collision Avoidance; Force and Tactile Sensing

    Abstract : In a physical Human-Robot Interaction for industrial scenarios is paramount to guarantee the safety of the user while keeping the robot's performance. Hierarchical task approaches are not sufficient since they tend to sacrifice the low priority tasks in order to guarantee the consistency of the main task. To handle this problem, we enhance the standard hierarchical fusion by introducing a novel interactive task-reconfiguring approach (TACTO-Selector) that uses the information of the tactile interaction to adapt the dimension of the tasks, therefore guaranteeing the execution of the safety task while performing the other task as good as possible. In this work, we hierarchically combine a 6 DOF Position-Based Visual Servoing (PBVS) task with a reactive skin control. This approach was evaluated on a 6 DOF industrial robot showing an improvement of 36.37% on average in tracking error reduction compared with the standard approach.

- Towards an Intelligent Collaborative Robotic System for Mixed Case Palletizing

    Author: Lamon, Edoardo | Istituto Italiano Di Tecnologia
    Author: Leonori, Mattia | Istituto Italiano Di Tecnologia
    Author: Kim, Wansoo | Istituto Italiano Di Tecnologia
    Author: Ajoudani, Arash | Istituto Italiano Di Tecnologia
 
    keyword: Physical Human-Robot Interaction; Logistics; Mobile Manipulation

    Abstract : In this paper, a novel human-robot collaborative framework for mixed case palletizing is presented. The frame- work addresses several challenges associated with the detection and localisation of boxes and pallets through visual perception algorithms, high-level optimisation of the collaborative effort through effective role-allocation principles, and maximisation of packing density. A graphical user interface (GUI) is additionally developed to ensure an intuitive allocation of roles and the optimal placement of the boxes on target pallets. The framework is evaluated in two conditions where humans operate with and without the support of a Mobile COllaborative robotic Assistant (MOCA). The results show that the optimised placement can improve up to the 20% with respect to a manual execution of the same task, and reveal the high potential of MOCA in increasing the performance of collaborative palletizing tasks.

- Treadmill Based Three Tether Parallel Robot for Evaluating Auditory Warnings While Running

    Author: Luttmer, Nathaniel | University of Utah
    Author: Truong, Takara | University of Utah
    Author: Boynton, Alicia | University of Utah, School of Biological Sciences
    Author: Carrier, David | University of Utah
    Author: Minor, Mark | University of Utah
 
    keyword: Physical Human-Robot Interaction; Tendon/Wire Mechanism; Human Detection and Tracking

    Abstract : We design and test a 3 DoF parallel cable system capable of applying precise and accurate impulses to walking and running subjects for the University of Utah's Treadport Active Wind Tunnel (TPAWT). Using Nexus VICON motion capture and gait algorithms, perturbations can be applied at different points in the subject's gait. The use of a PID force controller allow the system to create omnidirectional perturbations with walking and running subjects while having the capability to vary amplitude and direction of perturbations. Analysis is presented of the workspace of the large treadmill to test whether the workspace available to activate these perturbations is safe. This paper reports the efficacy of the system and evaluates how warning a runner before impact may affect their displacement. Participants experienced 48 perturbations while running applied with a random combination of a front/back/left/right impact at either toe-off or mid-stance with or without warning. A two sample T-test reveals that warning a runner before impact significantly reduced the magnitude they were displaced for both toe-off (t(46) = 4.98 p&lt;.001) and mid-stance (t(46) = 3.44, p = .001).

- Evaluation of Human-Robot Object Co-Manipulation under Robot Impedance Control

    Author: Mujica, Martin | INP-ENI of Tarbes
    Author: Benoussaad, Mourad | INP-ENI of Tarbes
    Author: Fourquet, Jean-Yves | ENIT
 
    keyword: Physical Human-Robot Interaction; Compliance and Impedance Control; Force Control

    Abstract : The human-robot collaboration is a promising and challenging field of robotics research. One of the main collaboration tasks is the object co-manipulation where the human and robot are in a continuous physical interaction and forces exerted must be handled. This involves some issues known in robotics as physical Human-Robot Interaction (pHRI), where human safety and interaction comfort are required. Moreover, a definition of interaction quality metrics would be relevant. In the current work, the assessment of Human-Robot object co-manipulation task was explored through the proposed metrics of interaction quality, based on human forces throughout the movement. This analysis is based on co-manipulation of objects with different dynamical properties (weight and inertia), with and without including these properties knowledge in the robot control law. Here, the human is a leader of task and the robot the follower without any information of the human trajectory and movement profile. For the robot control law, a well-known impedance control was applied on a 7-dof Kuka LBR iiwa 14 R820 robot. Results show that the consideration of object dynamical properties in the robot control law is crucial for a good and more comfortable interaction. Besides, human efforts are more significant with a higher no-considered weight, whereas it remains stable when these weights were considered.

## Telerobotics and Teleoperation
- Adaptive     Authority Allocation in Shared Control of Robots Using Bayesian Filters

    Author: Balachandran, Ribin | DLR
    Author: Mishra, Hrishik | German Aerospace Center (DLR)
    Author: Cappelli, Matteo | German Aerospace Center (DLR)
    Author: Weber, Bernhard | German Aerospace Center
    Author: Secchi, Cristian | Univ. of Modena &amp; Reggio Emilia
    Author: Ott, Christian | German Aerospace Center (DLR)
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Telerobotics and Teleoperation; Haptics and Haptic Interfaces; Sensor-based Control

    Abstract : In the present paper, we propose a novel system-driven adaptive shared control framework in which the autonomous system allocates the     Authority among the human operator and itself.     Authority allocation is based on a metric derived from a Bayesian filter, which is being adapted online according to real measurements. In this way, time-varying measurement noise characteristics are incorporated. We present the stability proof for the proposed shared control architecture with adaptive     Authority allocation, which includes time delay in the communication channel between the operator and the robot. Furthermore, the proposed method is validated through experiments and a user-study evaluation. the obtained results indicate significant improvements in task execution compared with pure teleoperation.

- Tactile Telerobots for Dull, Dirty, Dangerous, and Inaccessible Tasks

    Author: Fishel, Jeremy | Tangible Research
    Author: Oliver, Toni | Shadow Robot Company
    Author: Eichermueller, Michael | HaptX, Inc
    Author: Barbieri, Giuseppe | Shadow Robot Company
    Author: Fowler, Ethan | Shadow Robot Company
    Author: Hartikainen, Toivo | Shadow Robot Company
    Author: Moss, Luke | Shadow Robot Company
    Author: Walker, Rich | Shadow Robot Company
 
    keyword: Telerobotics and Teleoperation; Dexterous Manipulation; Haptics and Haptic Interfaces

    Abstract : The sense of touch, which is essential for human dexterity, is virtually absent from today's robotic hands. In this work we present progress in creating a highly-dexterous bimanual tactile telerobot, and evaluate its performance compared to bare hands. The system, consisting of anthropomorphic robot hands, biomimetic tactile sensors, and advanced haptic gloves enables a human operator to intuitively control and feel what the robotic hands are touching. Through carefully tuned tactile and kinematic mapping it was possible to intuitively perform dexterous operations, including pick and place tasks and even in-hand manipulation, a challenge for most autonomous robotic hands. Performance of the system was evaluated in standard measures of human and robotic dexterity such as the Box and Block test and other YCB benchmarks. This first-generation telerobot was found to have promising performance with the pilot able to do the same tasks in the telerobot between 1/4th to 1/12th the speed of their bare hands depending on the task complexity.

- A Teleoperation Framework for Mobile Robots Based on Shared Control

    Author: Luo, Jing | South China University of Technology
    Author: Lin, Zhidong | South China University of Technology
    Author: Li, Yanan | University of Sussex
    Author: Yang, Chenguang | University of the West of England
 
    keyword: Telerobotics and Teleoperation; Haptics and Haptic Interfaces; Cognitive Human-Robot Interaction

    Abstract : Mobile robots can complete a task in cooperation with a human partner. In this paper, a hybrid shared control method for a mobile robot with omnidirectional wheels is proposed. A human partner utilizes a six degrees of freedom haptic device and electromyography (EMG) signals sensor to control the mobile robot. A hybrid shared control approach based on EMG and artificial potential field is exploited to avoid obstacles according to the repulsive force and attractive force and to enhance the human perception of the remote environment based on force feedback of the mobile platform. This shared control method enables the human partner to tele-control the mobile robot's motion and achieve obstacles avoidance synchronously. Compared with conventional shared control methods, this proposed one provides a force feedback based on muscle activation and drives the human partners to update their control intention with predictability. Experimental results demonstrate the enhanced performance of the mobile robots in comparison with the methods in the literatures.

- A Novel Orientability Index and the Kinematic Design of the RemoT-ARM: A Haptic Master with Large and Dexterous Workspace

    Author: Li, Gaofeng | Italy Institute of Technology (IIT)
    Author: Del Bianco, Edoardo | Istituto Italiano Di Tecnologia
    Author: Caponetto, Fernando | Istituto Italiano Di Tecnologia
    Author: Katsageorgiou, Vasiliki-Maria | Humanoids and Human Centered Mechatronics, Istituto Italiano Di
    Author: Tsagarakis, Nikos | Istituto Italiano Di Tecnologia
    Author: Sarakoglou, Ioannis | Fondazione Istituto Italiano Di Tecnologia
 
    keyword: Telerobotics and Teleoperation; Haptics and Haptic Interfaces; Kinematics

    Abstract : Orientability is an important performance index to evaluate the dexterity of haptic master devices. Currently, most of the existing haptic master devices have limited workspace and limited dexterity. In this paper, we present the RemoT-ARM, a 6 Degree-of-Freedom (DOF) haptic master device that can provide larger and more dexterous workspace for operators. To evaluate its reachability of orientations, we propose a novel orientability index. Furthermore, a relative orientability index is proposed to characterize the matching degree of the workspace of a given manipulator to its target workspace. The volume, the manipulability and the condition number are also introduced as performance indices to evaluate the size and the isotropy of the workspace. According to these performance indices, all possible configurations for the RemoT-ARM have been taken into consideration, analyzed, and compared to finalize its optimal configuration.

- Enhancing the Transparency by Onomatopoeia for Passivity-Based Time-Delayed Teleoperation

    Author: Zhu, Yaonan | Nagoya University
    Author: Aoyama, Tadayoshi | Nagoya University
    Author: Hasegawa, Yasuhisa | Nagoya University
 
    keyword: Telerobotics and Teleoperation; Haptics and Haptic Interfaces; Virtual Reality and Interfaces

    Abstract : Robotic teleoperation with force feedback has been studied extensively since it was first developed in the 1940s. Time delay is a common problem of bilateral teleoperation systems. Although many efforts on optimizing the control architectures have been made, there is always a trade-off between transparency and stability for bilateral systems, and the perfect transparency and stability can only be achieved simultaneously in ideal situations. In this paper, we propose a novel approach to compensate for the degraded transparency while using the conventional passivity-based approach to maintain system stability under time-delay. The proposed approach is based on visual feedback and enhances the transparency by displaying different kinds of onomatopoeia according to contact force detected on the slave side. The basic performance is evaluated by conducting a stiffness classification task under constant round trip time delays (0 ms, 500 ms and 1000 ms). The preliminary results indicate that the subjects have higher accuracy for classifying the stiffness of a remote object by using onomatopoeia enhanced force feedback compared with the conventional passivity-based position-force feedback.

- RAVEN-S: Design and Simulation of a Robot for Teleoperated Microgravity Rodent Dissection under Time Delay
 
    Author: Lewis, Andrew | University of Washington
    Author: Drajeske, David | Applied Dexterity
    Author: Raiti, John | University of Washington
    Author: Berens, Angelique | University of Washington
    Author: Rosen, Jacob | &#8203;University of California, Los Angeles
    Author: Hannaford, Blake | University of Washington
 
    keyword: Telerobotics and Teleoperation; Space Robotics and Automation; Mechanism Design

    Abstract : The International Space Station (ISS) serves as a research lab for a wide variety of experiments including some that study the biological effects of microgravity and spaceflight using the Rodent Habitat and Microgravity Science Glovebox (MSG). Astronauts train for onboard dissections of rodents following basic training. An alternative approach for conducting these experiments is teleoperation of a robot located on the ISS from earth by a scientist who is proficient in rodent dissection. This pilot study addresses (1) the effects of extreme time delay on skill degradation during Fundamentals of Laparoscopic Surgery (FLS) tasks and rodent dissections using RAVEN II; (2) derivation and testing of rudimentary interaction force estimation; (3) elicitation of design requirements for an onboard dissection robot, RAVEN-S; and (4) simulation of the RAVEN-S prototype design with dissection data. The results indicate that the tasks' completion times increased by a factor of up to 9 for a 3 s time delay while performing manipulation and cutting tasks (FLS model) and by a factor of up to 3 for a 0.75 s time delay during mouse dissection tasks (animal model). Average robot forces/torques of 14N/0.1Nm (peak 90N/0.75Nm) were measured along with average linear/angular velocities of 0.02m/s / 4rad/s (peak 0.1m/s / 40rad/s) during dissection. A triangular configuration of three arms with respect to the operation site showed the best configuration given the MSG geometry and the dissection tasks.

- Closing the Force Loop to Enhance Transparency in Time-Delayed Teleoperation

    Author: Balachandran, Ribin | DLR
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
    Author: Jorda, Mikael | Stanford University
    Author: Ott, Christian | German Aerospace Center (DLR)
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Telerobotics and Teleoperation; Force Control; Force and Tactile Sensing

    Abstract : In the present paper, we first adopt explicit force control from general robotics and embed it into teleoperation systems to enhance the transparency by reducing the effect of the perceived inertia to the human operator and simultaneously improve contact perception. To ensure stability of the proposed teleoperation system considering time-delays, we propose a sequential design procedure based on time domain passivity approach. Experimental results of master-slave teleoperation system, based on KUKA light-weight-robots, for different values of delays are presented. Comparative analysis is conducted considering two existing approaches, namely 2-channel and 4-channel architecture based bilateral controllers, and its results clearly indicate significant improvement in force transparency owing to the proposed method. The proposed system is finally validated considering a real industrial assembly scenario.

- Evaluation of an Exoskeleton-Based Bimanual Teleoperation Architecture with Independently Passivated Slave Devices

    Author: Sant'Anna), Francesco Porcini (Scuola | PERCRO Laboratory, TeCIP Institute, Sant�Anna School of Advanced
    Author: Chiaradia, Domenico | Scuola Superiore Sant'Anna, TeCIP Institute, PERCRO Laboratory,
    Author: Marcheschi, Simone | PERCRO - Scuola Superiore S.Anna
    Author: Solazzi, Massimiliano | Scuola Superiore Sant'Anna, TeCIP Institute
    Author: Frisoli, Antonio | TeCIP Institute, Scuola Superiore Sant'Anna
 
    keyword: Telerobotics and Teleoperation; Physical Human-Robot Interaction; Wearable Robots

    Abstract : Search and rescue robotics is becoming a relevant topic in the last years. In this context, the possibility to drive a remote robot with an exoskeleton is a promising strategy to enhance dexterity, reduce operator effort and save time. However, the use of haptic feedback (bilateral teleoperation) may lead to instability in the presence of communication delay and more complex is the case of bimanual teleoperation where the two arms can exchange energy. In this work, we present a bimanual teleoperation system based on an exoskeletal master, where multi-degrees of freedom and kinematically different devices are involved. In the implemented architecture the two slaves are managed in parallel and independently passivated using the Time Domain Passivity Approach extended for multi-DoFs devices. To investigate the stability of the architecture we designed two tasks highly related to real disaster scenarios: the first one was useful to verify the system behavior in case of small movements and constrained configurations, whereas the second experiment was designed to involve larger contact forces and movements. Moreover, we compared the effect of both delay and low control loop frequency on the stability of the system when TDPA was applied. From the results, it was evident that the overall system exhibited a stable behavior with the use of the TDPA, even passivating the two slaves independently, under simulated time delay and in presence of a low control loop frequency.

- Hand-Worn Haptic Interface for Drone Teleoperation

    Author: Macchini, Matteo | EPFL
    Author: Havy, Thomas&nbsp;Clint&nbsp;Patrick | EPFL
    Author: Weber, Antoine | EPFL
    Author: Schiano, Fabrizio | Ecole Polytechnique Federale De Lausanne, EPFL
    Author: Floreano, Dario | Ecole Polytechnique Federal, Lausanne
 
    keyword: Telerobotics and Teleoperation; Haptics and Haptic Interfaces; Human Performance Augmentation

    Abstract : Drone teleoperation is usually accomplished using remote radio controllers, devices that can be hard to master for inexperienced users. Moreover, the limited amount of information fed back to the user about the robot's state, often limited to vision, can represent a bottleneck for operation in several conditions. In this work, we present a wearable interface for drone teleoperation and its evaluation through a user study. The two main features of the proposed system are a data glove to allow the user to control the drone trajectory by hand motion and a haptic system used to augment their awareness of the environment surrounding the robot. This interface can be employed for the operation of robotic systems in line of sight (LOS) by inexperienced operators and allows them to safely perform tasks common in inspection and search-and-rescue missions such as approaching walls and crossing narrow passages with limited visibility conditions. In addition to the design and implementation of the wearable interface, we performed a systematic study to assess the effectiveness of the system through three user studies (n = 36) to evaluate the users' learning path and their ability to perform tasks with limited visibility. We validated our ideas in both a simulated and a real-world environment. Our results demonstrate that the proposed system can improve teleoperation performance in different cases compared to standard remote controllers, making it a viable alternative to standard Human-Rob

- Toward Human-Like Teleoperated Robot Motion: Performance and Perception of a Choreography-Inspired Method in Static and Dynamic Tasks for Rapid Pose Selection of Articulated Robots

    Author: Bushman, Allison | University of Illinois
    Author: Asselmeier, Maxwell | University of Illinois at Urbana-Champaign
    Author: Won, Justin | University of Illinois
    Author: LaViers, Amy | University of Illinois at Urbana-Champaign
 
    keyword: Telerobotics and Teleoperation; Performance Evaluation and Benchmarking

    Abstract : In some applications, operators may want to create fluid, human-like motion on a remotely-operated robot, for example, a device used for remote telepresence. This paper examines two methods of controlling the pose of a Baxter robot via an Xbox One controller. The first method is a joint-by-joint (JBJ) method in which one joint of each limb is specified in sequence. The second method of control, named Robot Choreography Center (RCC), utilizes choreographic     Abstractions in order to simultaneously move multiple joints of the limb of the robot in a predictable manner. Thirty-eight users were asked to perform four tasks with each method. Success rate and duration of successfully completed tasks were used to analyze the performances of the participants. Analysis of the preferences of the users found that the joint-by-joint (JBJ) method was considered to be more precise, easier to use, safer, and more articulate, while the choreography-inspired (RCC) method of control was perceived as faster, more fluid, and more expressive. Moreover, performance data found that while both methods of control were over 80% successful for the two static tasks, the RCC method was an average of 11.85% more successful for the two more difficult, dynamic tasks. Future work will leverage this framework to investigate ideas of fluidity, expressivity, and human-likeness in robotic motion through online user studies with larger participant pools.

- Helping Robots Learn: A Human-Robot Master-Apprentice Model Using Demonstrations Via Virtual Reality Teleoperation

    Author: DelPreto, Joseph | Massachusetts Institute of Technology
    Author: Lipton, Jeffrey | University of Washington
    Author: Sanneman, Lindsay | Massachusetts Institute of Technology
    Author: Fay, Aidan | MIT CSAIL
    Author: Fourie, Christopher K | Massachusetts Institute of Technology (MIT)
    Author: Choi, Changhyun | University of Minnesota, Twin Cities
    Author: Rus, Daniela | MIT
 
    keyword: Telerobotics and Teleoperation; Learning from Demonstration; Human Factors and Human-in-the-Loop

    Abstract : As artificial intelligence becomes an increasingly prevalent method of enhancing robotic capabilities, it is important to consider effective ways to train these learning pipelines and to leverage human expertise. Working towards these goals, a master-apprentice model is presented and is evaluated during a grasping task for effectiveness and human perception. The apprenticeship model augments self-supervised learning with learning by demonstration, efficiently using the human's time and expertise while facilitating future scalability to supervision of multiple robots; the human provides demonstrations via virtual reality when the robot cannot complete the task autonomously. Experimental results indicate that the robot learns a grasping task with the apprenticeship model faster than with a solely self-supervised approach and with fewer human interventions than a solely demonstration-based approach; 100% grasping success is obtained after 150 grasps with 19 demonstrations. Preliminary user studies evaluating workload, usability, and effectiveness of the system yield promising results for system scalability and deployability. They also suggest a tendency for users to overestimate the robot's skill and to generalize its capabilities, especially as learning improves.

- A Framework for Interactive Virtual Fixture Generation for Shared Teleoperation in Unstructured Environments

    Author: Pruks, Vitalii | KAIST
    Author: Ryu, Jee-Hwan | Korea Advanced Institute of Science and Technology
 
    keyword: Telerobotics and Teleoperation; Haptics and Haptic Interfaces; Virtual Reality and Interfaces

    Abstract :  Virtual fixtures (VFs) improve human operator performance in teleoperation scenarios. However, the generation of VFs is challenging, especially in unstructured environments. In this work, we introduce a framework for the interactive generation of VF. The method is based on the observation that a human can easily understand just by looking at the remote environment which VF could help in task execution. We propose a user interface that detects features on camera images and permits interactive selection of the features. We demonstrate how the feature selection can be used for designing VF, providing 6-DOF haptic feedback. In order to make the proposed framework more generally applicable to a wider variety of applications, we formalize the process of virtual fixture generation (VFG) into the specification of features, geometric primitives, and constraints. The framework can be extended further by the introduction of additional components. Through the human subject study, we demonstrate the proposed framework is intuitive, easy to use while effective, especially for performing hard contact tasks.

- Whole-Body Bilateral Teleoperation of a Redundant Aerial Manipulator

    Author: Coelho, Andre | German Aerospace Center (DLR)
    Author: Singh, Harsimran | DLR German Aerospace Center
    Author: Kondak, Konstantin | German Aerospace Center
    Author: Ott, Christian | German Aerospace Center (DLR)
 
    keyword: Telerobotics and Teleoperation; Aerial Systems: Mechanics and Control; Haptics and Haptic Interfaces

    Abstract : Attaching a robotic manipulator to a flying base allows for significant improvements in the reachability and versatility of manipulation tasks. In order to explore such systems while taking advantage of human capabilities in terms of perception and cognition, bilateral teleoperation arises as a reasonable solution. However, since most telemanipulation tasks require visual feedback in addition to the haptic one, real-time (task-dependent) positioning of a video camera, which is usually attached to the flying base, becomes an additional objective to be fulfilled. Since the flying base is part of the kinematic structure of the robot, if proper care is not taken, moving the video camera could undesirably disturb the end-effector motion. For that reason, the necessity of controlling the base position in the null space of the manipulation task arises. In order to provide the operator with meaningful information about the limits of the allowed motions in the null space, this paper presents a novel haptic concept called Null-Space Wall. In addition, a framework to allow stable bilateral teleoperation of both tasks is presented. Numerical simulation data confirm that the proposed framework is able to keep the system passive while allowing the operator to perform time-delayed telemanipulation and command the base to a task-dependent optimal pose.

- Shared Autonomous Interface for Reducing Physical Effort in Robot Teleoperation Via Human Motion Mapping

    Author: Lin, Tsung-Chi | Worcester Polytechnic Institute
    Author: Unni Krishnan, Achyuthan | Worcester Polytechnic Institute
    Author: Li, Zhi | Worcester Polytechnic Institute
 
    keyword: Telerobotics and Teleoperation; Human Factors and Human-in-the-Loop; Medical Robots and Systems

    Abstract : Motion mapping is an intuitive method of teleoperation with a low learning curve. Our previous study investigates the physical fatigue caused by teleoperating a robot to perform general-purpose assistive tasks and this fatigue affects the operator's performance. The results from that study indicate that physical fatigue happens more in the tasks which involve more precise manipulation and steady posture maintenance. In this paper, we investigate how teleoperation assistance in terms of shared autonomy can reduce the physical workload in robot teleoperation via motion mapping. Specifically, we conduct a user study to compare the muscle effort in teleoperating a mobile humanoid robot to (1) reach and grasp an individual object and (2) collect objects in a cluttered workspace with and without an autonomous grasping function that can be triggered manually by the teleoperator. We also compare the participants' task performance, subjective user experience, and change in attitude towards the usage of teleoperation assistance in the future based on their experience using the assistance function. Our results show that: (1) teleoperation assistance like autonomous grasping can effectively reduce the physical effort, task completion time and number of errors; (2) based on their experience performing the tasks with and without assistance, the teleoperators reported that they would prefer to use automated functions for future teleoperation interfaces.

- DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System

    Author: Handa, Ankur | IIIT Hyderabad
    Author: Van Wyk, Karl | NVIDIA
    Author: Yang, Wei | NVIDIA
    Author: Liang, Jacky | Carnegie Mellon University
    Author: Chao, Yu-Wei | Univeristy of Michigan
    Author: Wan, Qian | Harvard University
    Author: Birchfield, Stan | NVIDIA
    Author: Ratliff, Nathan | Lula Robotics Inc
    Author: Fox, Dieter | University of Washington
 
    keyword: Telerobotics and Teleoperation; Learning from Demonstration; Grasping

    Abstract : Teleoperation relays natural human motion to control robotic systems that are relatively free of sophisticated reasoning skills, intuition, and creativity. However, tele-operation solutions for high degree-of-actuation (DoA), multi-fingered robots are generally cost-prohibitive, while low-cost offerings usually offer reduced degrees of control. Herein, a low-cost, RGB-D based teleoperation system was developed that allows for complete control over the full 23 DoA robotic system by merely observing the bare human hand. The system was used to solve a variety of manipulation tasks that go beyond simple pick-and-place operations accompanied by a thorough assessment of system performance. The videos of the experiments can be found at https://sites.google.com/view/dex-pilot.

- Distributed Winner-Take-All Teleoperation of a Multi-Robot System

    Author: Yang, Yuan | University of Victoria
    Author: Constantinescu, Daniela | University of Victoria
    Author: Shi, Yang | University of Victoria
 
    keyword: Telerobotics and Teleoperation; Multi-Robot Systems; Networked Robots

    Abstract : In a distributed multi-master-multi-slave teleoperation system, the human users may compete against each other for the control of the team of slave robots. To win the competition, one operator would send the largest command to the slave group. For the sake of team cohesion, the slave group should follow the command of the winning operator and ignore the commands of the other users. To enable (i) the slave team to identify the winning operator, and (ii) each slave to determine whether to admit or discard the command it receives from its operator, this paper proposes a dynamic decision-making protocol that distinguishes the decision variable of the slave commanded by the winner from the decision variables of all other slave robots. The protocol only requires the slaves to exchange and evaluate their decision variables locally. Lyapunov stability analysis proves the theoretical convergence of the proposed decision-making algorithm. An experimental distributed winner-take-all teleoperation in a 3-masters-11-slaves teleoperation testbed validates its practical efficacy.

- Enhanced Teleoperation Using Autocomplete

    Author: Kassem Zein, Mohammad | American University of Beirut (AUB)
    Author: Sidaoui, Abbas | American University of Beirut
    Author: Asmar, Daniel | American University of Beirut
    Author: Elhajj, Imad | American University of Beirut
 
    keyword: Telerobotics and Teleoperation; Human Performance Augmentation

    Abstract : Controlling and manning robots from a remote location is difficult because of the limitations one faces in perception and available degrees of actuation. Although humans can become skilled teleoperators, the amount of training time required to acquire such skills is typically very high. In this paper, we propose a novel solution (named Autocomplete) to aid novice teleoperators in manning robots adroitly. At the input side, Autocomplete relies on machine learning to detect and categorize human inputs as one from a group of motion primitives. Once a desired motion is recognized, at the actuation side an automated command replaces the human input in performing the desired action. So far, Autocomplete can recognize and synthesize lines, arcs, full circles, 3-D helices, and sine trajectories. Autocomplete was tested in simulation on the teleoperation of an unmanned aerial vehicle, and results demonstrate the advantages of the proposed solution versus manual steering.

## Collision Avoidance
- Collision-Free Navigation of Human-Centered Robots Via Markov Games

    Author: Ye, Guo | Northwestern University
    Author: Lin, Qinjie | Northwestern University
    Author: Juang, Tzung-Han | Northwestern University
    Author: Liu, Han | Northwestern University
 
    keyword: Collision Avoidance; Path Planning for Multiple Mobile Robots or Agents; Deep Learning in Robotics and Automation

    Abstract : We exploit Markov games as a framework for collision-free navigation of human-centered robots. Unlike the classical methods which formulate robot navigation as a single-agent Markov decision process with a static environment, our framework of Markov games adopts a multi-agent formulation with one primary agent representing the robot and the remaining auxiliary agents form a dynamic or even competing environment. Such a framework allows us to develop a path-following type adversarial training strategy to learn a robust decentralized collision avoidance policy. Through thorough experiments on both simulated and real-world mobile robots, we show that the learnt policy outperforms the state-of-the-art algorithms in both sample complexity and runtime robustness.

- DenseCAvoid: Real-Time Navigation in Dense Crowds Using Anticipatory Behaviors

    Author: Sathyamoorthy, Adarsh Jagan | University of Maryland
    Author: Liang, Jing | University of Maryland
    Author: Patel, Utsav | University of Maryland
    Author: Guan, Tianrui | University of Maryland
    Author: Chandra, Rohan | University of Maryland
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Collision Avoidance; Motion and Path Planning

    Abstract : We present DenseCAvoid, a novel algorithm for navigating a robot through dense crowds and avoiding collisions by anticipating pedestrian behaviors. Our formulation uses visual sensors and a pedestrian trajectory prediction algorithm to track pedestrians in a set of input frames and compute bounding boxes that extrapolate to the pedestrian positions in a future time. Our hybrid approach combines this trajectory prediction with a Deep Reinforcement Learning-based collision avoidance method to train a policy to generate smoother, safer, and more robust trajectories during run-time. We train our policy in realistic 3-D simulations of static and dynamic scenarios with multiple pedestrians. In practice, our hybrid approach generalizes well to unseen, real-world scenarios and can navigate a robot through dense crowds (~1-2 humans per square meter) in indoor scenarios, including narrow corridors and lobbies. As compared to cases where prediction was not used, we observe that our method reduces the occurrence of the robot freezing in a crowd by up to 48%, and performs comparably with respect to trajectory lengths and mean arrival times to goal.

- DEEPCRASHTEST: Turning Dashcam Videos into Virtual Crash Tests for Automated Driving Systems

    Author: Bashetty, Sai Krishna | ASU
    Author: Ben Amor, Heni | Arizona State University
    Author: Fainekos, Georgios | Arizona State University
 
    keyword: Collision Avoidance; Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization

    Abstract : The goal of this paper is to generate simulations with real-world collision scenarios for training and testing autonomous vehicles. We use numerous dashcam crash videos uploaded on the internet to extract valuable collision data and recreate the crash scenarios in a simulator. We tackle the problem of extracting 3D vehicle trajectories from videos recorded by an unknown and uncalibrated monocular camera source using a modular approach. A working architecture and demonstration videos along with the open-source implementation are provided with the paper.

- Observer-Extended Direct Method for Collision Monitoring in Robot Manipulators Using Proprioception and IMU Sensing

    Author: Baradaran Birjandi, Seyed Ali | Technical University of Munich
    Author: Kuehn, Johannes | Technical University of Munich
    Author: Haddadin, Sami | Technical University of Munich
 
    keyword: Collision Avoidance; Sensor Fusion; Robot Safety

    Abstract : In this paper a novel method for accurate and high-bandwidth real-time monitoring of robot collisions is presented. To the     Authors' knowledge this is the first time the so called direct method, which is mathematically the simplest and theoretically the ideal one, has been realized at practically relevant levels. For this, joint velocity and acceleration of serial chain robots are initially estimated using observer techniques that fuse joint position, Cartesian acceleration and angular velocity measurements. Consequently, this algorithm, which also extends our previous work in velocity and acceleration estimation, together with the available robot dynamics model are utilized to algebraically monitor external forces applied to the robot. Specifically, the proposed sensor fusion setup increases estimation bandwidth and decreases detection uncertainties compared to existing methods. Moreover, since neither inversion of large matrices nor their derivatives are required, our approach also shows increased numerical stability. Finally, the developed algorithm is evaluated based on a realistic simulation with the consideration of all parasitic effects and experimentally with a 7-DoF flexible joint robot.

- DCAD: Decentralized Collision Avoidance with Dynamics Constraints for Agile Quadrotor Swarms

    Author: Arul, Senthil Hariharan | University of Maryland, College Park
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Collision Avoidance; Path Planning for Multiple Mobile Robots or Agents; Simulation and Animation

    Abstract : We present DCAD, a novel, decentralized collision avoidance algorithm for navigating a swarm of quadrotors in dense environments populated with static and dynamic obstacles. Our algorithm relies on the concept of Optimal Reciprocal Collision Avoidance (ORCA) and utilizes a flatness-based Model Predictive Control (MPC) to generate local collision-free trajectories for each quadrotor. We feedforward linearize the non-linear dynamics of the quadrotor and subsequently use this linearized model in our MPC framework. Our approach tends to compute safe trajectories that avoid quadrotors from entering each other's downwash regions during close proximity maneuvers. In addition, we account for the uncertainty in the position and velocity sensor data using Kalman filter. We evaluate the performance of our algorithm with other state-of-the-art decentralized methods and demonstrate its superior performance in terms of smoothness of generated trajectories and lower probability of collision during high-velocity maneuvers.

- Forward Kinematics Kernel for Improved Proxy Collision Checking

    Author: Das, Nikhil | UCSD
    Author: Yip, Michael C. | University of California, San Diego
 
    keyword: Collision Avoidance; Kinematics; Motion and Path Planning

    Abstract : Kernel functions may be used in robotics for comparing different poses of a robot, such as in collision checking, inverse kinematics, and motion planning. These comparisons provide distance metrics often based on joint space measurements and are performed hundreds or thousands of times a second, continuously for changing environments. We introduce a new kernel function based on forward kinematics (FK) to compare robot manipulator configurations. We integrate our new FK kernel into our proxy collision checker, Fastron, that previously showed significant speed improvements to collision checking and motion planning. With the new FK kernel, we realize a two-fold speedup in proxy collision check speed, 8 times less memory, and a boost in classification accuracy from 74% to over 95% for a 7 degrees-of-freedom robot arm compared to the previously-used radial basis function kernel. Compared to state-of-the-art geometric collision checkers, with the FK kernel, collision checks are now 9 times faster. To show the broadness of the approach, we apply Fastron FK in OMPL across a wide variety of motion planners, showing unanimously faster robot planning.

- Local Obstacle-Skirting Path Planning for a Fast Bi-Steerable Rover Using B�ziers Curves

    Author: Fnadi, Mohamed | ISIR - Sorbonne University
    Author: Du, Wenqian | Sorbonne University, ISIR, Paris 6
    Author: Gomes da Silva, Rafael | ENSTA Paris
    Author: Plumet, Frederic | UPMC
    Author: Ben Amar, Faiz | Université Pierre Et Marie Curie, Paris 6
 
    keyword: Collision Avoidance; Dynamics; Field Robots

    Abstract :  This paper focuses on local path planning for obstacle avoidance tasks dedicated to double steering off-road mobile robots. This technique calculates a new local path for the vehicle using a set of cubic B�zier curves once the safety distance is not respected; otherwise, the vehicle follows the global reference path which is defined off-line. Two basic steps are used to determine this new path. Firstly, some significant points that should belong to the planned path are extracted on-line according to the obstacle's sizes and the current state of the vehicle, these points are adopted as waypoints. Secondly, on-line cubic B�zier curves are computed to create a smooth path for these points such that the safety and lateral stability of the vehicle are ensured (i.e., preventing huge curvatures and wide-variation in steering angles). This path will be used as a reference to be performed by the vehicle using a constrained model predictive control. The validation of our navigation strategy is performed via numerical simulations and experiments using a real off-road mobile robot.

- Collision Avoidance with Proximity Servoing for Redundant Serial Robot Manipulators

    Author: Ding, Yitao | Chemnitz University of Technology
    Author: Thomas, Ulrike | Chemnitz University of Technology
 
    keyword: Collision Avoidance; Reactive and Sensor-Based Planning; Motion Control of Manipulators

    Abstract : Collision avoidance is a key technology towards safe human-robot interaction, especially on-line and fast-reacting motions are required. Skins with proximity sensors mounted on the robot's outer shell provide an interesting approach to occlusion-free and low-latency perception. However, collision avoidance algorithms which make extensive use of these properties for fast-reacting motions have not yet been fully investigated. We present an improved collision avoidance algorithm for proximity sensing skins by formulating a quadratic optimization problem with inequality constraints to compute instantaneous optimal joint velocities. Compared to common repulsive force methods, our algorithm confines the approach velocity to obstacles and keeps motions pointing away from obstacles unrestricted. Since with repulsive motions the robot only moves in one direction, opposite to obstacles, our approach has better exploitation of the redundancy space to maintain the task motion and gets stuck less likely in local minima. Furthermore, our method incorporates an active behaviour for avoiding obstacles and evaluates all potentially colliding obstacles for the whole arm, rather than just the single nearest obstacle. We demonstrate the effectiveness of our method with simulations and on real robot manipulators in comparison with commonly used repulsive force methods and our prior proposed approach.

- Predicting Obstacle Footprints from 2D Occupancy Maps by Learning from Physical Interactions

    Author: Kollmitz, Marina | University of Freiburg
    Author: B�scher, Daniel | Albert-Ludwigs-Universitét Freiburg
    Author: Burgard, Wolfram | Toyota Research Institute
 
    keyword: Collision Avoidance; Deep Learning in Robotics and Automation; Learning and Adaptive Systems

    Abstract : Horizontally scanning 2D laser rangefinders are a popular approach for indoor robot localization because of the high accuracy of the sensors and the compactness of the required 2D maps. As the scanners in this configuration only provide information about one slice of the environment, the measurements typically do not capture the full extent of a large variety of obstacles, including chairs or tables. Accordingly, obstacle avoidance based on laser scanners mounted in such a fashion is likely to fail. In this paper, we propose a learning-based approach to predict collisions in 2D occupancy maps. Our approach is based on a convolutional neural network which is trained on a 2D occupancy map and collision events recorded with a bumper while the robot is navigating in its environment. As the network operates on local structures only, it can generalize to new environments. In addition, the robot can collect and integrate new collision examples after an initial training phase. Extensive experiments carried out in simulation and a realistic real-world environment confirm that our approach allows robots to learn from collision events to avoid collisions in the future.

- Path Planning in Dynamic Environments Using Generative RNNs and Monte Carlo Tree Search

    Author: Eiffert, Stuart | The University of Sydney: The Australian Centre for Field Roboti
    Author: Kong, He | University of Sydney
    Author: Pirmarzdashti, Navid | The University of Sydney: The Australian Centre for Field Roboti
    Author: Sukkarieh, Salah | The University of Sydney: The Australian Centre for Field Roboti
 
    keyword: Collision Avoidance; Autonomous Vehicle Navigation

    Abstract : State of the art methods for robotic path planning in dynamic environments, such as crowds or traffic, rely on hand crafted motion models for agents. These models often do not reflect interactions of agents in real world scenarios. To overcome this limitation, this paper proposes an integrated path planning framework using generative Recurrent Neural Networks within a Monte Carlo Tree Search (MCTS). This approach uses a learnt model of social response to predict crowd dynamics during planning across the action space. This extends our recent work using generative RNNs to learn the relationship between planned robotic actions and the likely response of a crowd. We show that the proposed framework can considerably improve motion prediction accuracy during interactions, allowing more effective path planning. The performance of our method is compared in simulation with existing methods for collision avoidance in a crowd of pedestrians, demonstrating the ability to control future states of nearby individuals. We also conduct preliminary real world tests to validate the effectiveness of our method.

- Safety-Critical Rapid Aerial Exploration of Unknown Environments

    Author: Singletary, Andrew | California Institute of Technology
    Author: Gurriet, Thomas | California Institute of Technology
    Author: Nilsson, Petter | California Institute of Technology
    Author: Ames, Aaron | California Institute of Technology
 
    keyword: Collision Avoidance; Aerial Systems: Perception and Autonomy; Robot Safety

    Abstract : This paper details a novel approach to collision avoidance for aerial vehicles that enables high-speed flight in uncertain environments. This framework is applied at the controller level and provides safety regardless of the planner that is used. The method is shown to be robust to state uncertainty and disturbances, and is computed entirely online utilizing the full nonlinear system dynamics. The effectiveness of this method is shown in a high-fidelity simulation of a quadrotor with onboard sensors rapidly and safely exploring a cave environment utilizing a simple planner.

- Reactive Navigation under Non-Parametric Uncertainty through Hilbert Space Embedding of Probabilistic Velocity Obstacles

    Author: Poonganam, SriSai Naga Jyotish | IIIT Hyderabad
    Author: Gopalakrishnan, Bharath | IIIT HYDERABAD
    Author: Avula, Venkata Seetharama Sai Bhargav Kumar | International Institute of Information Technology, Hyderabad
    Author: Singh, Arun Kumar | Tampere University of Technology, Finland
    Author: Krishna, Madhava | IIIT Hyderabad
    Author: Manocha, Dinesh | University of Maryland
 
    keyword: Collision Avoidance

    Abstract : The probabilistic velocity obstacle (PVO) extends the concept of velocity obstacle (VO) to work in uncertain dynamic environments. In this paper, we show how a robust model predictive control (MPC) with PVO constraints under non-parametric uncertainty can be made computationally tractable. At the core of our formulation is a novel yet simple interpretation of our robust MPC as a problem of matching the distribution of PVO with a certain desired distribution. To this end, we propose two methods. Our first baseline method is based on approximating the distribution of PVO with a Gaussian Mixture Model (GMM) and subsequently performing distribution matching using Kullback Leibler (KL) divergence metric. Our second formulation is based on the possibility of representing arbitrary distributions as functions in Reproducing Kernel Hilbert Space (RKHS). We use this foundation to interpret our robust MPC as a problem of minimizing the distance between the desired distribution and the distribution of the PVO in the RKHS. Both the RKHS and GMM based formulation can work with any uncertainty distribution and thus allowing us to relax the prevalent Gaussian assumption in the existing works. We validate our formulation by taking an example of 2D navigation of quadrotors with a realistic noise model for perception and ego-motion uncertainty.

- Contact-Based Bounding Volume Hierarchy for Assembly Tasks

    Author: Shellshear, Evan | FCC
    Author: Li, Yi | Fraunhofer-Chalmers Research Centre
    Author: Bohlin, Robert | Fraunhofer-Chalmers Research Centre
    Author: Carlson, Johan | Fraunhofer-Chalmers Research Centre
 
    keyword: Collision Avoidance; Motion and Path Planning; Assembly

    Abstract : Path planning of an object which is allowed to be in contact with other objects during assembly process is a significant challenge due to the variety of permitted or forbidden collisions between the distinct parts of the objects to be assembled. In order to put objects together in real-life scenarios, parts of assembled objects may be required to flex, whereas other parts may have to fit exactly. Consequently, existing collision checking and distance computation algorithms have to be modified to enable path planning of objects that can be in contact during the assembly process. In this paper, we analyze an improved broad phase proximity query algorithm to enable such contact-based assembly tasks we call CHAT (Contact-based Hierarchy for Assembly Tasks). We demonstrate that, compared to existing approaches, our proposed method is more than an order of magnitude faster for collision queries and up to three times faster for distance queries when the two objects contain a large number of parts (with some parts containing thousands or tens of thousands of triangles). Due to the nature of the algorithm, we expect the performance improvements to increase as the number of parts in an object becomes larger.

- Construction of Bounding Volume Hierarchies for Triangle Meshes with Mixed Face Sizes

    Author: Li, Yi | Fraunhofer-Chalmers Research Centre
    Author: Shellshear, Evan | FCC
    Author: Bohlin, Robert | Fraunhofer-Chalmers Research Centre
    Author: Carlson, Johan | Fraunhofer-Chalmers Research Centre
 
    keyword: Collision Avoidance; Motion and Path Planning; Assembly

    Abstract : We consider the problem of creating tighter-fitting bounding volumes (more specifically rectangular swept spheres) when constructing bounding volume hierarchies (BVHs) for complex 3D geometries given in the form of unstructured triangle meshes/soups with the aim of speeding up our IPS Path Planner for rigid bodies, where the triangles often have very different sizes. Currently, the underlying collision and distance computation module (IPS CDC) does not take into account the sizes of the triangles when it constructs BVHs using a top-down strategy. To split triangles in a BVH node into two BVH nodes, IPS CDC has to compute both the split axis and the split position. In this work, we use the principal axes of the tensor of inertia as the potential split axes and the center of mass as the split position, where the computations of both the tensor of inertia and the center of mass require knowledge of the areas of the triangles. We show that our method improves performance (up to 20% faster) of our IPS Path Planner when it is used to plan collision-free disassembly paths for three different test cases taken from manufacturing industries.

- Strategy for Automated Dense Parking: How to Navigate in Narrow Lanes

    Author: Polack, Philip | Stanley Robotics
    Author: Cord, Aur�lien | Stanley Robotics
    Author: Dallen, Louis-Marie | Stanley Robotics
 
    keyword: Collision Avoidance; Optimization and Optimal Control; Kinematics

    Abstract : This paper presents the architecture of a high-density parking solution based on car-like robots specifically designed to move cars. The main difficulty is to park the vehicles close to one another which implies hard constraints on the robot motion and localization. In particular, this paper focuses on navigation in narrow lanes. We propose a Lyapunov-based control strategy that has been derived after expressing the problem in a Configuration Space formulation. The current solution has been implemented and tested on Stanley Robotics' robots and has been running in production for several months. Thanks to the Configuration Space formulation, we are able to guarantee the obstacles' integrity. Moreover, a method for calibrating the GPS orientation with a high-precision is derived from the present control strategy.

- Multimodal Trajectory Predictions for Urban Environments Using Geometric Relationship between a Vehicle and Lanes

    Author: Kawasaki, Atsushi | Toshiba Corporation
    Author: Seki, Akihito | Toshiba Corporation
 
    keyword: Collision Avoidance; Deep Learning in Robotics and Automation; Motion and Path Planning

    Abstract : Implementation of safe and efficient autonomous driving systems requires accurate prediction of the long-term trajectories of surrounding vehicles. High uncertainty in traffic behavior makes it difficult to predict trajectories in urban environments, which have various road geometries. To overcome this problem, we propose a method called lane-based multimodal prediction network (LAMP-Net), which can handle arbitrary shapes and numbers of traffic lanes and predict both the future trajectory along each lane and the probability of each lane being selected. A vector map is used to define the lane geometry and a novel lane feature is introduced to represent the generalized geometric relationships between the vehicle state and lanes. Our network takes this feature as the input and is trained to be versatile for arbitrarily shaped lanes. Moreover, we introduce a vehicle motion model constraint to our network. Our prediction method combined with the constraint significantly enhances prediction accuracy. We evaluate the prediction performance on two datasets which contain a wide variety of real-world traffic scenarios. Experimental results show that our proposed LAMP-Net outperforms state-of-the-art methods.

- Online Optimal Motion Generation with Guaranteed Safety in Shared Workspace

    Author: Zheng, Pu | University Grenoble Alpes
    Author: Wieber, Pierre-Brice | INRIA Rh�ne-Alpes
    Author: Aycard, Olivier | University Grenoble
 
    keyword: Collision Avoidance; Optimization and Optimal Control; Robot Safety

    Abstract : For new, safer manipulator robots, the probability of serious injury due to collisions with humans remains low (5%), even at speeds as high as 2 m/s. Collisions would better be avoided nevertheless, because they disrupt the tasks of both the robot and the human. We propose in this paper to equip robots with exteroceptive sensors and online motion generation so that the robot is able to perceive and react to the motion of the human in order to reduce the occurrence of collisions. We adapt a Model Predictive Control scheme which has been demonstrated previously with two industrial manipulator robots avoiding collisions while sharing their workspace. It's impossible to guarantee that no collision will ever take place in a partially unknown dynamic environment such as a shared workspace, but we can guarantee instead that, if a collision takes place, the robot is at rest at the time of collision, so that it doesn�t inject its own kinetic energy in the collision. The proposed control scheme is validated in simulation.

- Episodic Koopman Learning of Nonlinear Robot Dynamics with Applications to Fast Multirotor Landing

    Author: Folkestad, Carl | California Institute of Technology
    Author: Pastor, Daniel | Caltech
    Author: Burdick, Joel | California Institute of Technology
 
    keyword: Learning and Adaptive Systems; Aerial Systems: Perception and Autonomy; Optimization and Optimal Control

    Abstract : This paper presents a novel episodic method to learn a robot's nonlinear dynamics model and an increasingly optimal control sequence for a set of tasks. The method is based on the Koopman operator approach to nonlinear dynamical systems analysis, which models the flow of observables in a function space, rather than a flow in a state space. Practically, this method estimates a nonlinear diffeomorphism that lifts the dynamics to a higher dimensional space where they are linear. Efficient Model Predictive Control methods can then be applied to the lifted model. This approach allows for real time implementation in on-board hardware, with rigorous incorporation of both input and state constraints during learning. We demonstrate the method in a real-time implementation of fast multirotor landing, where the nonlinear ground effect is learned and used to improve landing speed and quality.

## Micro/Nano Robots

- Reconfigurable Magnetic Microswarm for Thrombolysis under Ultrasound Imaging

    Author: Wang, Qianqian | The Chinese University of Hong Kong
    Author: Wang, Ben | The Chinese University of Hong Kong
    Author: Yu, Jiangfan | University of Toronto
    Author: Schweizer, Kathrin | ETH Zurich
    Author: Nelson, Bradley J. | ETH Zurich
    Author: Zhang, Li | The Chinese University of Hong Kong
 
    keyword: Micro/Nano Robots; Medical Robots and Systems; Swarms

    Abstract : We propose thrombolysis using a magnetic nanoparticle microswarm with tissue plasminogen activator (tPA) under ultrasound imaging. The microswarm is generated in blood using an oscillating magnetic field and can be navigated with locomotion along both the long and short axis. By modulating the input field, the aspect ratio of the microswarm can be reversibly tuned, showing the ability to adapt to different confined environments. Simulation results indicate that both in-plane and out-of-plane fluid convection are induced around the microswarm, which can be further enhanced by tuning the aspect ratio of the microswarm. Under ultrasound imaging, the microswarm is navigated in a microchannel towards a blood clot and deformed to obtain optimal lysis. Experimental results show that the lysis rate reaches -0.1725 - 0.0612 mm^3/min in the 37&#9702;C blood environment under the influence of the microswarm-induced fluid convection and tPA. The lysis rate is enhanced 2.5-fold compared to that without the microswarm (-0.0681 - 0.0263 mm^3/min). Our method provides a new strategy to increase the efficiency of thrombolysis by applying microswarm-induced fluid convection, indicating that swarming micro/nanorobots have the potential to act as effective tools towards targeted therapy.

- Improving Optical Micromanipulation with Force-Feedback Bilateral Coupling

    Author: Gerena, Edison | Sorbonne Université Facult� Pierre Et Mairie Curie
    Author: Legendre, Florent | Sorbonne Université
    Author: Vitry, Youen | ULB
    Author: R�gnier, Stéphane | Sorbonne University
    Author: Haliyo, Dogan Sinan | Sorbonne Université
 
    keyword: Micro/Nano Robots; Haptics and Haptic Interfaces; Biological Cell Manipulation

    Abstract : Micromanipulation is challenging due to the specific physical effects governing the microworld. Interactive approaches using only visual feedback are limited to the 2D image of the microscope, and have forcibly lower bandwidth. Recently, haptic feedback teleoperation systems have been developed to try to overcome those difficulties. This paper explores the case of an optical tweezers platform coupled to an haptic device providing transparent force feedback. The impact of haptic feedback regarding user dexterity on tactile exploration tasks is studied using 3 �m microbeads and a test bench with micro sized shapes. The results reveal a consistent improvement in both users' trajectory tracking and their control of the contact forces. This also validates the experimental setup which performed reliably on 140 different trials of the evaluation.

- Maneuver at Micro Scale: Steering by Actuation Frequency Control in Micro Bristle Robots

    Author: Hao, Zhijian | Georgia Institute of Technology
    Author: Kim, DeaGyu | Georgia Institute of Technology
    Author: Mohazab, Ali Reza | Foundation for the Advancement of Sciences, Humanities, Enginee
    Author: Ansari, Azadeh | Georgia Institute of Technology
 
    keyword: Micro/Nano Robots; Mechanism Design; Automation at Micro-Nano Scales

    Abstract : This paper presents a novel steering mechanism, which leads to frequency-controlled locomotion demonstrated for the first time in micro bristle robots. The miniaturized robots are 3D-printed, 12 mm - 8 mm - 6 mm in size, with bristle feature sizes down to 400 &#956;m. The robots can be steered by utilizing the distinct resonance behaviors of the asymmetrical bristle sets. The left and right sets of the bristles have different diameters, and thus different stiffnesses and resonant frequencies. The unique response of each bristle side to the vertical vibrations of a single on-board piezoelectric actuator causes differential steering of the robot. The robot can be modeled as two coupled uniform bristle robots, representing the left and the right sides. At distinct frequencies, the robots can move in all four principal directions: forward, backward, left and right. Furthermore, the full 360� 2D plane can be covered by superimposing the principal actuation frequency components with desired amplitudes. In addition to miniaturized robots, the presented resonance-based steering mechanism can be applied over multiple scales and to other mechanical systems.

- Scaling down an Insect-Size Microrobot, HAMR-VI into HAMR-Jr

    Author: Jayaram, Kaushik | University of Colorado Boulder
    Author: Shum, Jennifer | Harvard University
    Author: Castellanos, Sam | Harvard University
    Author: Helbling, Elizabeth Farrell | Harvard University
    Author: Wood, Robert | Harvard University
 
    keyword: Micro/Nano Robots; Biologically-Inspired Robots; Legged Robots

    Abstract : Here, we present HAMR-Jr, a 22.5 mm, 320 mg quadrupedal microrobot. With eight independently actuated degrees of freedom, HAMR-Jr is, to our knowledge, the most mechanically dexterous legged robot at its scale and is capable of high-speed locomotion (13.91 bodylengths/s) at a variety of stride frequency (1-200 Hz) using multiple gaits. We achieved this using a design and fabrication process that is flexible, allowing scaling with minimum changes to our workflow. We further characterized HAMR-Jr's open-loop locomotion and compared it with the larger scale HAMR-VI microrobot to demonstrate the effectiveness of scaling laws in predicting running performance.

- Model Predictive Control with Obstacle Avoidance for Inertia Actuated AFM Probes Inside a Scanning Electron Microscope

    Author: Liang, Shuai | Pierre and Marie Curie University, the Institute for Intelligent
    Author: Boudaoud, Mokrane | Sorbonne Université
    Author: Morin, Pascal | UPMC
    Author: Cagneau, Barth�lemy | Université De Versailles Saint-Quentin En Yvelines
    Author: Rong, Weibin | Harbin Institute of Technology, Harbin, China
    Author: R�gnier, Stéphane | Sorbonne University
 
    keyword: Micro/Nano Robots; Automation at Micro-Nano Scales

    Abstract : The Atomic Force Microscope (AFM) is as a reliable tool for 3D imaging and manipulation at the micrometer and nanometer scales. When used inside a Scanning Elec- tron Microscope (SEM), AFM probes can be localized and controlled with a nanometer resolution by visual feedback. However, achieving trajectory control and obstacles avoidance is still a major concern for manipulation tasks. We propose a Model Predictive Control (MPC) to address these two issues while AFM probes are actuated by Piezoelectric Inertia type Actuators (PIA). The novelty of this paper is that the model of our MPC-based approach relies on a velocity map of PIAs. It enables path following and obstacle avoidance while preserving safety margins. Control inputs are optimized by Quadratic Programming, referring to their increment and distance constraints. A cost function is defined to navigate the AFM probe with a specified velocity. Simulations and experiments are carried out to demonstrate that the proposed algorithm is suitable to perform path following with obstacle avoidance using map-based velocity references. This is the first time that MPC is implemented in micro/nano-robotic systems for autonomous control inside SEM.

- Double-Modal Locomotion and Application of Soft Cruciform Thin-Film Microrobot

    Author: Su, Meng | Shenzhen Institutes of Advanced Technology&#65292;Chinese Academ
    Author: Xu, Tiantian | Chinese Academy of Sciences
    Author: Lai, Zhengyu | Chinese Academy of Science, Shenzhen Institude of Advanced Techn
    Author: Huang, Chenyang | Shenzhen Institutes of Advanced Technology, Chinese Academy of S
    Author: Liu, Jia | ShenZhen Institutes of Advanced Technology, Chinese Academy of S
    Author: Wu, Xinyu | CAS
 
    keyword: Micro/Nano Robots; Soft Robot Applications; Biologically-Inspired Robots

    Abstract : Untethered, wirelessly controlled microrobots have a broad application prospect from industrial area, to the bioengineering due to their small scales. In a narrow environment containing viscous resistance and friction fluid, rigid body may damage the micro-objects that the microrobots manipulate. In this paper, we propose a new type of soft microrobot which is informed the Cruciform Thin-film Microrobot (CTM). This soft microrobot has two motion modes: jellyfish-like mode and forklift truck mode. The forklift truck mode helps to transport micro-objects. The heaviest object that CTM could carry is ten times weighter than its own. The CTM is controlled for s-shaped trajectory control. In this paper, we study the swimming properties of the microrobot. The maximum velocity of CTM is 8mm/s. The velocity of the microrobot is inversely proportional to solution viscosity and proportional to magnetic field frequency. And the velocity is proportional to leg length and thickness. Load testing and manipulation test of the three microbeads to the same location are completed. The CTMs are of great significance for bioengineering and industrial microoperation. In future works, we will conduct more accurate trajectory control and manipulation performance tests.

- Robotic Control of a Magnetic Swarm for On-Demand Intracellular Measurement

    Author: Wang, Xian | University of Toronto
    Author: Wang, Tiancong | University of Toronto
    Author: Shan, Guanqiao | University of Toronto
    Author: Law, Junhui | University of Toronto
    Author: Dai, Changsheng | University of Toronto
    Author: Zhang, Zhuoran | University of Toronto
    Author: Sun, Yu | University of Toronto
 
    keyword: Automation in Life Sciences: Biotechnology, Pharmaceutical and Health Care; Automation at Micro-Nano Scales; Micro/Nano Robots

    Abstract : Fluorescent dyes are routinely used for biochemical measurements such as pH and ion concentrations. They, especially when used for detecting a low concentration of ions, suffer from low signal-to-noise ratios (SNR); and increasing the concentration of dyes causes more sever cytotoxicity. We invented a new approach that uses a low amount of fluorescent dye-coated magnetic nanoparticles for on-demand, accurately aggregating the nanoparticles and thus fluorescent dyes in a local region inside a cell for measurement. Experiments proved this approach is capable of achieving a significantly higher SNR and lower cytotoxicity. Different from existing magnetic micromanipulation systems that generate large swarms (several microns and above) or cannot move the generated swarm to an arbitrary position, we developed a five-pole magnetic manipulation system and technique for generating a small swarm (e.g., 1 �m; controllable size from 0.52 �m to 52.7 �m with an error &lt;7.5%) and accurately positioning the swarm (positioning control accuracy: 0.76 �m). As an example, the system performed intracellular pH mapping using a 1 �m swarm of pH sensitive fluorescent dye-coated magnetic nanoparticles. The swarm had an SNR inside a cell 10 times that by the traditional dye treatment, with both cases using the same fluorescent dye concentration. Our intracellular measurement results, for the first time, quantitatively revealed the existence of pH gradient in live migrating cells.

- Acoustofluidic Tweezers for the 3D Manipulation of Microparticles

    Author: Guo, Xinyi | Max Planck Institute for Intelligent Systems; Tianjin University
    Author: Ma, Zhichao | Max Planck Institute for Intelligent Systems
    Author: Goyal, Rahul | Max Planck Institute for Intelligent Systems
    Author: Jeong, Moonkwang | Max Planck Institute for Intelligent Systems
    Author: Pang, Wei | Tianjin University
    Author: Fischer, Peer | Max-Planck-Institute for Intelligent Systems
    Author: Duan, Xuexin | Tianjin University
    Author: Qiu, Tian | University of Stuttgart
 
    keyword: Automation at Micro-Nano Scales; Micro/Nano Robots; Automation in Life Sciences: Biotechnology, Pharmaceutical and Health Care

    Abstract : Non-contact manipulation is of great importance in the actuation of micro-robotics. It is challenging to contactless manipulate micro-scale objects over large spatial distance in fluid. Here, we describe a novel approach for the dynamic position control of microparticles in three-dimensional (3D) space, based on high-speed acoustic streaming generated by a micro-fabricated gigahertz transducer. The hydrodynamic force generated by the streaming flow field has a vertical component against gravity and a lateral component towards the center, thus the microparticle is able to be stably trapped at a position far from the transducer surface, and to be manipulated over centimeter distance in 3D. Only the hydrodynamic force is utilized in the system for particle manipulation, making it a versatile tool regardless the material properties of the trapped particle. The system shows high reliability and manipulation velocity, revealing its potentials for the applications in robotics and automation at small scales.

- Task Space Motion Control for AFM-Based Nanorobot Using Optimal and Ultralimit Archimedean Spiral Local Scan

    Author: Sun, Zhiyong | The University of Hong Kong
    Author: Xi, Ning | The University of Hong Kong
    Author: Xue, Yuxuan | University of Hong Kong
    Author: Cheng, Yu | Michigan State University
    Author: Chen, Liangliang | Michigan State University
    Author: Yang, Ruiguo | Northwestern University
    Author: Song, Bo | Hefei Institutes of Physical Science, Chinese Academy of Science
 
    keyword: Micro/Nano Robots; Automation at Micro-Nano Scales; Visual Servoing

    Abstract : Atomic force microscopy (AFM) based nanorobotic technology provides a unique manner for delicate operations at the nanoscale in various ambient thanks to its ultrahigh spatial resolution, outstanding environmental adaptability, and numerous measurement approaches. However, one vital challenge behind nanoscale operations is the task space positioning problem, known as difficulty of realizing desirable relative position between the AFM sharp tip and the target. It is noted that though one AFM possesses nanometer imaging resolution, it is hard to achieve nanometer positioning accuracy due to system uncertainties, such as the uncompensated nonlinearity and the thermal drift generated internally/externally. In order to tackle the vital positioning problem in the task space, this paper proposes a specific visual servoing control framework using local ambient image as feedback to overcome positioning uncertainty at the nanoscale. In this study, we employ the optimal Archimedean spiral scanning strategy and try to exceed the speed criterion to pursue faster and uniform local scan for generating feedback images. To fulfill reliable precise tip locating with the possible non-ideal feedback images, a type of visual servoing control approach: extended non-vector space (ENVS) controller based on subset projection method was developed for tackling environmental noise and disturbances. Experimental studies were conducted to verify the effectiveness of the proposed methodology.

- Kinematic Model of a Magnetic-Microrobot Swarm in a Rotating Magnetic Dipole Field

    Author: Chaluvadi, BhanuKiran | Blue Ocean Robotics
    Author: Stewart, Kristen | University of Utah
    Author: Sperry, Adam | University of Utah
    Author: Fu, Henry | University of Utah
    Author: Abbott, Jake | University of Utah
 
    keyword: Micro/Nano Robots; Medical Robots and Systems

    Abstract : This letter describes how a rotating magnetic dipole field will manipulate the location and shape of a swarm of magnetic microrobots, specifically microrobots that convert rotation into forward propulsion, such as helical swimmers and screws. The analysis assumes a swarm that can be described by a centroid and a covariance matrix, with the swarm comprising an arbitrary and unknown number of homogenous microrobots. The result of this letter is a kinematic model that can be used as an <i>a priori</i> model for motion planners and feedback control systems. Because the model is fully three-dimensional and does not require any localization information beyond what could realistically be determined from medical images, the method has potential for <i>in vivo</i> medical applications. The model is experimentally verified using magnetic screws moving through a soft-tissue phantom, propelled by a rotating spherical permanent magnet.

- Magnetic Milli-Robot Swarm Platform: A Safety Barrier Certificate Enabled, Low-Cost Test Bed

    Author: Hsu, Allen | SRI International
    Author: Huihua, Zhao | Georgia Institute of Technology
    Author: Gaudreault, Martin | SRI International
    Author: Wong-Foy, Annjoe | SRI International
    Author: Pelrine, Ron | SRI International
 
    keyword: Micro/Nano Robots; Multi-Robot Systems; Swarms

    Abstract : Swarms of micro- and milli-sized robots have the potential to advance biological micro-manipulation, micro-assembly and manufacturing, and provide an ideal platform for studying large swarm behaviors and control. Due to their small size and low cost, tens to hundreds of micro/milli robots can function in parallel to perform a task that otherwise would be too cumbersome or costly for a larger macroscopic robot. Here, we demonstrate a scalable system and modular circuit architecture for controlling and coordinating the motion of &gt;10's of magnetic micro/milli robots. By modifying the concepts of safety barrier certificates to our magnetic robot hardware, we achieve minimally invasive, collision-free, 2D position control (x,y) of up to N = 16 robots in a low-cost tabletop (288mm x 288mm) magnetic milli-robot platform with up to 288 degrees of freedom. We show that the introduction of random dithering can achieve a 100% success rate (i.e., no deadlocking), enabling the system to serve as a platform for the study of various swarm-like behaviors or multi-agent robotic coordination.

- A Device for Rapid, Automated Trimming of Insect-Sized Flying Robots

    Author: Dhingra, Daksh | University of Washington
    Author: Chukewad, Yogesh Madhavrao | University of Washington
    Author: Fuller, Sawyer | University of Washington
 
    keyword: Micro/Nano Robots; Automation at Micro-Nano Scales; Aerial Systems: Mechanics and Control

    Abstract : Successful demonstrations of controlled flight in flying insect-sized robots (FIRs) &lt;500~mg have all relied on piezo-actuated flapping wings because of unfavorable downward size scaling in motor-driven propellers. In practice, the mechanical complexity of flapping wings typically results in large torque bias variability about pitch and roll axes, leading to rapid rotation in free flight for vehicles that are not properly trimmed. Manual trimming by watching high-speed video is tedious and error-prone. In this letter, we introduce an alternative, a trimming device that uses feedback from motion capture cameras to determine and correct for bias torques. It does so using an automated feedback loop, without the need for any visual feedback from the user, or airborne flights which can damage the robot. We validated the device on two different robot flies. After trimming with our device, the robots both took off approximately vertically in open-loop and were able to hover in free flight under feedback control. Our system, therefore, reduces the time of essential yet time-consuming step in robot fly fabrication, facilitating their eventual mass production and practical application.

- Eye-In-Hand 3D Visual Servoing of Helical Swimmers Using Parallel Mobile Coils

    Author: Yang, Zhengxin | The Chinese Univeristy of HongKong
    Author: Yang, Lidong | The Chinese University of Hong Kong
    Author: Zhang, Li | The Chinese University of Hong Kong
 
    keyword: Micro/Nano Robots; Visual Servoing; Motion Control

    Abstract : Magnetic helical microswimmers can be propelled by rotating magnetic field and are adept at passing through narrow space. To date, various magnetic actuation systems and control methods have been developed to drive these microswimmers. However, steering their spacial movement in a large workspace is still challenging, which could be significant for potential medical applications. In this regard, this paper designs an eye-in-hand stereo-vision module and corresponding refraction-rectified location algorithm. Combined with the motor module and the coil module, the mobile-coil system is capable of generating dynamic magnetic fields in a large 3D workspace. Based on the system, a robust triple-loop stereo visual servoing strategy is proposed that operates simultaneous tracking, locating, and steering, through which the helical swimmer is able to follow a long-distance 3D path. A scaled-up magnetic helical swimmer is employed in the path following experiment. Our prototype system reaches a cylindrical workspace with a diameter more than 200 mm, and the mean error of path tracking is less than 2 mm.

- A Mobile Paramagnetic Nanoparticle Swarm with Automatic Shape Deformation Control

    Author: Yang, Lidong | The Chinese University of Hong Kong
    Author: Yu, Jiangfan | University of Toronto
    Author: Zhang, Li | The Chinese University of Hong Kong
 
    keyword: Micro/Nano Robots; Swarms; Automation at Micro-Nano Scales

    Abstract : Recently, swarm control of micro-/nanorobots has drawn much attention in the field of microrobotics. This paper reports a mobile paramagnetic nanoparticle swarm with the capability of active shape deformation that can improve its environment adaptability. We show that, by applying elliptical rotating magnetic fields, a swarm pattern called the elliptical paramagnetic nanoparticle swarm (EPNS) would be formed. When changing the field ratio- (i.e. the strength ratio between the minor axis and major axis of the elliptical field), the shape ratio- of the EPNS (i.e. the length ratio between the major axis and minor axis) will change accordingly. However, automatically control this shape deformation process has difficulties because the deformation dynamics has strong nonlinearity, model variation and long time requirement. To solve this problem, we propose a fuzzy logic-based control scheme that utilizes the knowledge and control experience from skilled human operators. Experiments show that the proposed control scheme can stably maneuver the shape deformation of the EPNS with small overshoot, which cannot be achieved by conventional PI control. Moreover, experimental results show that, with the automatic shape deformation control, shape of the EPNS is controlled with high reversibility and also can be well maintained during the planar rotational and translational locomotion of the EPNS.

- Magnetic Miniature Swimmers with Multiple Flagella

    Author: Quispe, Johan Edilberto | Sorbonne University, CNRS Institut Des Syst�mes Intelligents Et
    Author: R�gnier, Stéphane | Sorbonne University
 
    keyword: Micro/Nano Robots; Biomimetics; Simulation and Animation

    Abstract : In this paper, we introduce novel miniature swimmers with multiple rigid tails based on spherical helices. The tail distribution of these prototypes enhances its swimming features as well as allowing to carry objects with it. The proposed swimmers are actuated by a rotating magnetic field, generating the robot rotation and thus producing a considerable thrust to start self-propelling. These prototypes achieved propulsion speeds up to 6 mm/s at 3.5 Hz for a 6-mm in size prototypes. We study the efficiency of different tail distribution for a 2-tailed swimmer by varying the angular position between both tails. Moreover, it is demonstrated that these swimmers experience great sensibility when changing their tail height. Besides, these swimmers demonstrate to be effective for cargo carrying tasks since they can displace objects up to 3.5 times their weight. Finally, wall effect is studied with multi-tailed swimmer robots considering 2 containers with 20 and 50-mm in width. Results showed speeds' increments up to 59% when swimmers are actuated in the smaller container.

- Design and Control of a Large-Range Nil-Stiffness Electro-Magnetic Active Force Sensor

    Author: Cailliez, Jonathan | Sorbonne Université, Institut Des Syst�mes Intelligents Et De Ro
    Author: Weill--Duflos, Antoine | McGill University
    Author: Boudaoud, Mokrane | Sorbonne Université
    Author: R�gnier, Stéphane | Sorbonne University
    Author: Haliyo, Dogan Sinan | Sorbonne Université
 
    keyword: Micro/Nano Robots; Sensor-based Control; Calibration and Identification

    Abstract : Active force sensors are key instruments to get around the tradeoff between the sensitivity and the measure- ment range of conventional passive force sensors. Thanks to their quasi-infinite stiffness in closed loop, active sensors can be applied for force measurements on samples with a wide range of stiffness without interference with the mechanical parameters of the sensor. MEMS (Micro-Electro Mechanical Systems) active force sensors have been wildly developed in the literature but they are ill adapted for force measurements at the Newton level needed in meso-scale robotics. In this article, a novel structure for a meso-scale active force sensor is proposed for the measurement of forces from the milli-newton to the newton.This novel meso-scale sensor is based on a nil-stiffness guidance and an electromagnetic actuation. This paper deals with its design, identification, calibration and closed loop control. The sensor exhibits nil-stiffness characteristic in open loop and an almost infinite stiffness in closed loop. This allows measuring forces with a large range of gradients. First experiments shows the ability of this new sensor architecture to measure low frequency forces up to 0.8 N with a precision of 0.03 N and a closed loop -20dB cutoff frequency of 73.9Hz.

- Modeling Electromagnetic Navigation Systems for Medical Applications Using Random Forests and Artificial Neural Networks

    Author: Yu, Ruoxi | The Chinese University of Hong Kong
    Author: Charreyron, Samuel L. | ETH Zurich
    Author: Boehler, Quentin | ETH Zurich
    Author: Weibel, Cameron | ETHZ
    Author: Chautems, Christophe | ETH Zurich
    Author: Poon, Carmen C. Y. | The Chinese University of Hong Kong
    Author: Nelson, Bradley J. | ETH Zurich
 
    keyword: Micro/Nano Robots; AI-Based Methods; Model Learning for Control

    Abstract : Electromagnetic Navigation Systems (eMNS) can be used to control a variety of multiscale devices within the human body for remote surgery. Accurate modeling of the magnetic fields generated by the electromagnets of an eMNS is crucial for the precise control of these devices. Existing methods assume a linear behavior of these systems, leading to significant modeling errors within nonlinear regions exhibited at higher magnetic fields, preventing these systems from operating at full capacity. In this paper, we use a random forest (RF) and an artificial neural network (ANN) to model the nonlinear behavior of the magnetic fields generated by an eMNS. Both machine learning methods outperformed the state- of-the-art linear multipole electromagnet model (MPEM). The RF and the ANN model reduced the root mean squared error (RMSE) of the MPEM when predicting the field magnitude by approximately 40% and 87%, respectively, over the entire current range of the eMNS. At high current regions, especially between 30 and 35 A, the field-magnitude RMSE improvement of the ANN model over the MPEM was 37 mT, equivalent to 90% error reduction. This study demonstrates the feasibility of using machine learning to model an eMNS for medical applications, and its ability to account for complex nonlinear behavior at high currents. The use of machine learning thus shows promise in developing accurate field predicting models, and ultimately improving surgical procedures that use magnetic navigation.

- Automated Tracking System with Head and Tail Recognition for Time-Lapse Observation of Free-Moving C. Elegans

    Author: Dong, Shengnan | Beijing Institute of Technology
    Author: Liu, Xiaoming | Beijing Institute of Technology
    Author: Li, Pengyun | Beijing Institute of Technology
    Author: Tang, Xiaoqing | Beijing Institute of Technology
    Author: Liu, Dan | Beijing Institute of Technology
    Author: Kojima, Masaru | Osaka University
    Author: Huang, Qiang | Beijing Institute of Technology
    Author: Arai, Tatsuo | University of Electro-Communications
 
    keyword: Micro/Nano Robots

    Abstract :     Abstract�In this paper, an automated tracking system with head and tail recognition for time-lapse observation of free-moving C. elegans is presented. In microscale field, active C. elegans can move out of the view easily without an automated tracking system because of the narrow field of view and rapid speed of C. elegans. In our previous works, we constructed an automated platform with 3D freedom to track centroid region of the nematode successfully. However, tracking time was not long enough to support a full time-lapse observation. Our proposed system in this study integrate the detection method in horizontal plane with depth evaluation more tightly. Tracking time and response speed have been greatly improved. Besides, we make full use of curvature calculation to make the system recognize the head and tail of C. elegans and the recognition rate can be up to 95%. The results demonstrate that the system can fully achieve automated long-term tracking of a free-living nematode and will be a nice tool for C. elegans behavioral analysis.

## AI-Based Methods
- Towards Adaptive Benthic Habitat Mapping

    Author: Shields, Jackson | University of Sydney
    Author: Pizarro, Oscar | Australian Centre for Field Robotics
    Author: Williams, Stefan Bernard | University of Sydney
 
    keyword: AI-Based Methods; Big Data in Robotics and Automation; Field Robots

    Abstract : Autonomous Underwater Vehicles (AUVs) are increasingly being used to support scientific research and monitoring studies. One such application is in benthic habitat mapping where these vehicles collect seafloor imagery that complements broadscale bathymetric data collected using sonar. Using these two data sources, the relationship between remotely-sensed acoustic data and the sampled imagery can be learned, creating a habitat model. As the areas to be mapped are often very large and AUV systems collecting seafloor imagery can only sample from a small portion of the survey area, the information gathered should be maximised for each deployment. This paper illustrates how the habitat models themselves can be used to plan more efficient AUV surveys by identifying where to collect further samples in order to most improve the habitat model. A Bayesian neural network is used to predict visually-derived habitat classes when given broad-scale bathymetric data. This network can also estimate the uncertainty associated with a prediction, which can be deconstructed into its aleatoric (data) and epistemic (model) components. We demonstrate how these structured uncertainty estimates can be utilised to improve the model with fewer samples. Such adaptive approaches to benthic surveys have the potential to reduce costs by prioritizing further sampling efforts. We illustrate the effectiveness of the proposed approach using data collected by an AUV on offshore reefs in Tasmania, Australia.

- Multispectral Domain Invariant Image for Retrieval-Based Place Recognition

    Author: Han, Daechan | Sejong University
    Author: Hwang, Yujin | Sejong University
    Author: Kim, Namil | NAVER LABS
    Author: Choi, Yukyung | Sejong University
 
    keyword: AI-Based Methods; Localization; Intelligent Transportation Systems

    Abstract : Multispectral recognition has attracted increasing attention from the research community due to its potential competence for many applications from day to night. However, due to the domain shift between RGB and thermal image, it has still many challenges to apply and to use RGB-based task. To reduce the domain gap, we propose Multispectral domain invariant framework, which leverages the unpaired image translation method to generate a semantic and strongly discriminative invariant image by enforcing novel constraints in the objective function. We demonstrate the efficacy of the proposed method on mainly place recognition task and achieve significant improvement compared to previous works. Furthermore, we test on multispectral semantic segmentation and unsupervised domain adaptations to prove the scalability and generality of the proposed method. We will open our source code and dataset.

- Probabilistic Effect Prediction through Semantic Augmentation and Physical Simulation

    Author: Bauer, Adrian Simon | German Aerospace Center (DLR)
    Author: Schmaus, Peter | German Aerospace Center (DLR)
    Author: Stulp, Freek | DLR - Deutsches Zentrum F�r Luft Und Raumfahrt E.V
    Author: Leidner, Daniel | German Aerospace Center (DLR)
 
    keyword: AI-Based Methods; Service Robots; Task Planning

    Abstract : Nowadays, robots are mechanically able to perform highly demanding tasks, where AI-based planning methods are used to schedule a sequence of actions that result in the desired effect. However, it is not always possible to know the exact outcome of an action in advance, as failure situations may occur at any time. To enhance failure tolerance, we propose to predict the effects of robot actions by augmenting collected experience with semantic knowledge and leveraging realistic physics simulations. That is, we consider semantic similarity of actions in order to predict outcome probabilities for previously unknown tasks. Furthermore, physical simulation is used to gather simulated experience that makes the approach robust even in extreme cases. We show how this concept is used to predict action success probabilities and how this information can be exploited throughout future planning trials. The concept is evaluated in a series of real world experiments conducted with the humanoid robot Rollin' Justin.

- Anytime Integrated Task and Motion Policies for Stochastic Environments

    Author: Shah, Naman | Arizona State University
    Author: Kala Vasudevan, Deepak | Arizona State University
    Author: Kumar, Kislay | Arizona State University
    Author: Kamojjhala, Pranav | Arizona State University
    Author: Srivastava, Siddharth | University of California Berkeley
 
    keyword: AI-Based Methods; Autonomous Agents; Task Planning

    Abstract : In order to solve complex, long-horizon tasks, intelligent robots need to carry out high-level,     Abstract planning and reasoning in conjunction with motion planning. However,     Abstract models are typically lossy and plans or policies computed using them can be unexecutable. These problems are exacerbated in stochastic situations where the robot needs to reason about, and plan for multiple contingencies. We present a new approach for integrated task and motion planning in stochastic settings. In contrast to prior work in this direction, we show that our approach can effectively compute integrated task and motion policies whose branching structures encoding agent behaviors handling multiple execution-time contingencies. We prove that our algorithm is probabilistically complete and can compute feasible solution policies in an anytime fashion so that the probability of encountering an unresolved contingency decreases over time. Empirical results on a set of challenging problems show the utility and scope of our methods.

- Context-Aware Human Activity Recognition

    Author: Mojarad, Roghayeh | UPEC
    Author: Attal, Ferhat | University Paris-Est Cr�teil (UPEC)
    Author: Chibani, Abdelghani | Lissi Lab Paris EST University
    Author: Amirat, Yacine | University of Paris Est Cr�teil (UPEC)
 
    keyword: AI-Based Methods; Human-Centered Automation; Human Detection and Tracking

    Abstract : One of the main challenges in designing Ambient Assisted Living (AAL) systems is Human Activity Recognition (HAR). The latter is crucial to improve the quality of people's lives in terms of autonomy, safety, and well-being. In this paper, a novel framework exploiting the contextual information of human activities is proposed for HAR. The proposed framework allows detecting and correcting classification errors automatically. Machine-learning models are firstly used to recognize human activities. These models may predict erroneous activities; therefore, detecting and correcting these errors is necessary to improve HAR. For this purpose, two Bayesian networks are used for classification error detection and classification error correction. The proposed framework is evaluated in terms of precision, recall, F-measure, and accuracy on the Opportunity dataset, a well-known dataset for multi-label human daily living activity recognition. The evaluation results demonstrate the ability of the proposed framework to improve HAR performance.

- Interactive Natural Language-Based Person Search

    Author: Shree, Vikram | Cornell University
    Author: Chao, Wei-Lun | Cornell University
    Author: Campbell, Mark | Cornell University
 
    keyword: AI-Based Methods; Human Detection and Tracking; Cognitive Human-Robot Interaction

    Abstract : In this work, we consider the problem of searching people in an unconstrained environment, with natural language descriptions. Specifically, we study how to systematically design an algorithm to effectively acquire descriptions from humans. An algorithm is proposed by adapting models, used for visual and language understanding, to search a person of interest (POI) in a principled way, achieving promising results without the need to re-design another complicated model. We then investigate an iterative question-answering (QA) strategy that enable robots to request additional information about the POI's appearance from the user. To this end, we introduce a greedy algorithm to rank questions in terms of their significance, and equip the algorithm with the capability to dynamically adjust the length of human-robot interaction according to model's uncertainty. Our approach is validated not only on benchmark datasets but on a mobile robot, moving in a dynamic and crowded environment.

## Climbing Robots
- CCRobot-III: A Split-Type Wire-Driven Cable Climbing Robot for Cable-Stayed Bridge Inspection

    Author: Ding, Ning | The Chinese University of Hong Kong (Shenzhen)
    Author: Zheng, Zhenliang | The Chinese University of Hong Kong, Shenzhen
    Author: Song, Junlin | Shenzhen Institute of Artificial Intelligence and Robotics for S
    Author: Sun, Zhenglong | Chinese University of Hong Kong, Shenzhen
    Author: Lam, Tin Lun | The Chinese University of Hong Kong, Shenzhen
    Author: Qian, Huihuan | The Chinese University of Hong Kong, Shenzhen
 
    keyword: Climbing Robots; Legged Robots; Biologically-Inspired Robots

    Abstract : This paper presents a novel Cable Climbing Robot CCRobot-III, which is the third version designed for bridge cable inspection tasks, aiming at surpassing previous versions in terms of climbing speed and payload capacity. Benefiting from Split-type Wire-driven design, CCRobot-III can climb along a 90-110mm diameter bridge cable in inchworm-like gait at a speed of up to 12m/min, and carrying more than 40kg payload at the same time. CCRobot-III consists of a climbing precursor and a main-body frame. The two parts are connected and driven by steel wires. The climbing precursor, acting as a mobile anchor, moves quickly on a bridge cable. The main-body frame, acting as a mobile winch, carries payload and pulls itself to a certain position with steel wires. Both parts have one or two pairs of palm-based gripper, which is the key component for providing strong adhesion to support the robot climbing. Experimental results have shown that CCRobot-III possesses outstanding climbing performance, high payload capacity, and good adaptability to complex conditions of cable surface. Moreover, it has potential engineering applications on the cable-stayed bridge for fieldwork.

- Omnidirectional Tractable Three Module Robot

    Author: Suryavanshi, Kartik | International Institute of Information Technology, Hyderabad
    Author: Vadapalli, Rama | International Institute of Information Technology, Hyderabad
    Author: Vucha, Ruchitha | International Institute of Information Technology, Hyderabad
    Author: Sarkar, Abhishek | International Institute of Information Technology, Hyderabad
    Author: Krishna, Madhava | IIIT Hyderabad
 
    keyword: Climbing Robots; Mechanism Design; Field Robots

    Abstract : This paper introduces the Omnidirectional Tractable Three Module Robot for traversing inside complex pipe networks. The robot consists of three omnidirectional modules fixed 120� apart circumferentially which can rotate about their own axis allowing holonomic motion of the robot. The holonomic motion enables the robot to overcome motion singularity when negotiating T-junctions and further allows the robot to arrive in a preferred orientation while taking turns inside a pipe. We have developed a closed-form kinematic model for the robot in the paper and propose the �Motion Singularity Region� that the robot needs to avoid while negotiating T-junction. The design and motion capabilities of the robot are demonstrated both by conducting simulations in MSC ADAMS on a simplified lumped-model of the robot and with experiments on its physical embodiment.

- A Practical Climbing Robot for Steel Bridge Inspection

    Author: Nguyen, Son | University of Nevada, Reno
    Author: Pham, Anh | University of Nevada, Reno
    Author: Motley, Cadence | University of Nevada, Reno
    Author: La, Hung | University of Nevada at Reno
 
    keyword: Robot Safety; Search and Rescue Robots; Service Robots

    Abstract :  The advanced robotic and automation (ARA) lab has developed and successfully implemented a design inspired by many of the various cutting edge steel inspection robots to date. The combination of these robots concepts into a unified design came with its own set of challenges since the parameters for these features sometimes conflicted. An extensive amount of design and analysis work was performed by the ARA lab in order to find a carefully tuned balance between the implemented features on the ARA robot and general functionality. Having successfully managed to implement this conglomerate of features represents a breakthrough to the industry of steel inspection robots as the ARA lab robot is capable of traversing most complex geometries found on steel structures while still maintaining its ability to efficiently travel along these structures; a feat yet to be done until now.

- Development of a Wheeled Wall-Climbing Robot with a Shape-Adaptive Magnetic Adhesion Mechanism

    Author: Eto, Haruhiko | Sumitomo Heavy Industries, Ltd
    Author: Asada, Harry | MIT
 
    keyword: Climbing Robots; Mechanism Design; Manufacturing, Maintenance and Supply Chains

    Abstract : This paper presents a wheeled wall-climbing robot with a shape-adaptive magnetic adhesion mechanism for large steel structures. To travel up and down various curved ferromagnetic surfaces, we developed a 2 DOF rotational magnetic adhesion mechanism installed on each wheel that can change the orientation of the magnets to keep the magnetic force direction always normal to the contact surface. These magnetic wheels have a spherical shape and can move relative to the main body by a non-elastic suspension mechanism so that the robot can climb up small obstacles on the ground and find contact points for each wheel on a wall with an arbitrary curved shape. Being geometrically stable is important for the robot because this robot is intended to be a mobile base for a welding manipulator. The detailed design of the mechanism and the results of climbing tests are presented.

- Towards More Possibilities: Motion Planning and Control for Hybrid Locomotion of Wheeled-Legged Robots

    Author: Sun, Jingyuan | National University of Singapore
    Author: You, Yangwei | Institute for Infocomm Research
    Author: Zhao, Xuran | National University of Singapore
    Author: Adiwahono, Albertus Hendrawan | I2R A-STAR
    Author: Chew, Chee Meng | National University of Singapore
 
    keyword: Legged Robots; Natural Machine Motion; Climbing Robots

    Abstract : This paper proposed a control framework to tackle the hybrid locomotion problem of wheeled-legged robots. It comes as a hierarchical structure with three layers: hybrid foot placement planning, Centre of Mass (CoM) trajectory optimization and whole-body control. General mathematical representation of foot movement is developed to analyze different motion modes and decide hybrid foot placements. Gait graph widely used in legged locomotion is extended to better describe the hybrid movement by adding extra foot velocity information. Thereafter, model predictive control is introduced to optimize the CoM trajectory based on the planned foot placements considering terrain height changing. The desired trajectories together with other kinematic and dynamic constraints are fed into a whole-body controller to produce joint commands. In the end, the feasibility of the proposed approach is demonstrated by the simulation and experiments of hybrid locomotion running on our wheeled quadrupedal robot Pholus.

- Navigation for Legged Mobility: Dynamic Climbing (I)

    Author: Austin, Max | Florida State University
    Author: Harper, Mario | Florida State University
    Author: Brown, Jason | Florida State University
    Author: Collins, Emmanuel | FAMU-FSU College of Engineering
    Author: Clark, Jonathan | Florida State University
 
    keyword: Climbing Robots; Legged Robots; Motion and Path Planning

    Abstract : Autonomous navigation through unstructured terrains has been most effectively demonstrated by animals, who utilize a large set of locomotive styles to move through their native habitats. While legged robots have recently demonstrated several of these locomotion modalities (such as walking, running, jumping, and climbing vertical walls), motion planners have yet to be able to leverage these unique mobility characteristics. In this article, we outline some of the	specific motion planning challenges faced when attempting to plan for legged systems with dynamic gaits, with specific instances of these demonstrated by the dynamic climbing platform TAILS. Using a unique implementation of sampling-based model predictive optimization, we demonstrate the ability to motion plan around obstacles on vertical walls and experimentally demonstrate this on TAILS by navigating through traditionally difficult narrow gap problems.

## Failure Detection and Recovery
- Algebraic Fault Detection and Identification for Rigid Robots

    Author: Lomakin, Alexander | Universitét Erlangen-N�rnberg
    Author: Deutscher, Joachim | Universitét Erlangen-N�rnberg
 
    keyword: Robot Safety; Failure Detection and Recovery

    Abstract : This paper presents a method for algebraic fault detection and identification of nonlinear mechanical systems, describing rigid robots, by using an approximation with orthonormal Jacobi polynomials. An explicit expression is derived for the fault from the equation of motion, which is decoupled from disturbances and only depends on measurable signals and their time derivatives. Fault detection and identification is then achieved by polynomial approximation of the determined fault term. The results are illustrated by a simulation for a faulty SCARA.

- Fault Tolerance Analysis of a Hexarotor with Reconfigurable Tilted Rotors

    Author: Pose, Claudio Daniel | Facultad De Ingenieria - Universidad De Buenos Aires
    Author: Giribet, Juan Ignacio | University of Buenos Aires
    Author: Mas, Ignacio | CONICET-ITBA
 
    keyword: Failure Detection and Recovery; Aerial Systems: Mechanics and Control

    Abstract : Tilted rotors in multirotor vehicles have shown to be useful for different practical reasons. For instance, increasing yaw maneuverability or enabling full position and attitude control of hexarotor vehicles. It has also been proven that a hexagon-shaped multirotor is capable of complete attitude and altitude control under failures of one of its rotors. However, when a rotor fails, the torque that can be reached in the worst-case direction decreases considerably. <p>This work proposes to actively change the tilt angle of the rotors when a failure occurs. This rotor reconfiguration increases the maximum torque that can be achieved in the most stressful direction, reducing maneuverability limitations. Experimental validations are shown, where the proposed reconfigurable tilted rotor is used in order to control a hexarotor vehicle when a failure appears mid-flight. The impact of the delay in the reconfiguration when a failure occurs is also addressed.

- Detecting Execution Anomalies As an Oracle for Autonomy Software Robustness

    Author: Katz, Deborah S. | Carnegie Mellon University
    Author: Hutchison, Casidhe | Carnegie Mellon University
    Author: Zizyte, Milda | Carnegie Mellon University
    Author: Le Goues, Claire | Carnegie Mellon University
 
    keyword: Failure Detection and Recovery; Robot Safety; Probability and Statistical Methods

    Abstract : We propose a method for detecting execution anomalies in robotics and autonomy software. The algorithm uses system monitoring techniques to obtain profiles of executions. It uses a clustering algorithm to create clusters of those executions, representing nominal execution. A distance metric determines whether additional execution profiles belong to the existing clusters or should be considered anomalies. The method is suitable for identifying faults in robotics and autonomy systems. We evaluate the technique in simulation on two robotics systems, one of which is a real-world industrial system. We find that our technique works well to detect possibly unsafe behavior in autonomous systems.

- When Your Robot Breaks: Active Learning During Plant Failure

    Author: Schrum, Mariah | Georgia Institute of Technology
    Author: Gombolay, Matthew | Georgia Institute of Technology
 
    keyword: Failure Detection and Recovery; Learning and Adaptive Systems; Model Learning for Control

    Abstract : Detecting and adapting to catastrophic failures in robotic systems requires a robot to learn its new dynamics quickly and safely to best accomplish its goals. To address this challenging problem, we propose probabilistically-safe, online learning techniques to infer the altered dynamics of a robot at the moment a failure (e.g., physical damage) occurs. We combine model predictive control and active learning within a chance-constrained optimization framework to safely and efficiently learn the new plant model of the robot. We leverage a neural network for function approximation in learning the latent dynamics of the robot under failure conditions. Our framework generalizes to various damage conditions while being computationally light-weight to advance real-time deployment. We empirically validate within a virtual environment that we can regain control of a severely damaged aircraft in seconds and require only 0.1 seconds to find safe, information-rich trajectories, outperforming state-of-the-art approaches.

- An Integrated Dynamic Fall Protection and Recovery System for Two-Wheeled Humanoids

    Author: Zambella, Grazia | University of Pisa
    Author: Monteleone, Simone | Research Center E. Piaggio, University of Pisa
    Author: Herrera Alarc�n, Edwin Pa�l | Scuola Superiore Sant'Anna
    Author: Negrello, Francesca | Istituto Italiano Di Tecnologia
    Author: Lentini, Gianluca | University of Pisa
    Author: Caporale, Danilo | Centro Di Ricerca E. Piaggio
    Author: Grioli, Giorgio | Istituto Italiano Di Tecnologia
    Author: Garabini, Manolo | Université Di Pisa
    Author: Catalano, Manuel Giuseppe | Istituto Italiano Di Tecnologia
    Author: Bicchi, Antonio | Université Di Pisa
 
    keyword: Robot Safety; Performance Evaluation and Benchmarking; Wheeled Robots

    Abstract : Robots face the eventuality of falling. Unplanned events, external disturbances and technical failures may lead a robot to a condition where even an effective dynamic stabilization is not sufficient to maintain the equilibrium. Therefore, it is essential to equip robotic platforms with both active and passive fall protection means to minimize damages, and enable the recovery and restart without physical human intervention.<p>This work introduces a method to design an integrated safety system for two-wheeled humanoids. As a case study, the method is applied to a robot and experimentally tested under several conditions corresponding to different causes of robot instability, such as motor jamming, external disturbances, and sudden shut-down.

- Reliability Validation of Learning Enabled Vehicle Tracking

    Author: Sun, Youcheng | Queen University of Belfast
    Author: Zhou, Yifan | University of Liverpool
    Author: Maskell, Simon | University of Liverpool
    Author: Sharp, James | Dstl
    Author: Huang, Xiaowei | University of Liverpool
 
    keyword: Failure Detection and Recovery

    Abstract : This paper studies the reliability of a real-world learning-enabled system, which conducts dynamic vehicle tracking based on a high-resolution wide-area motion imagery input. The system consists of multiple neural network components -- to process the imagery inputs -- and multiple symbolic (Kalman filter) components -- to analyse the processed information for vehicle tracking. It is known that neural networks suffer from adversarial examples, which make them lack robustness. However, it is unclear if and how the adversarial examples over learning components can affect the overall system-level reliability. By integrating a coverage-guided neural network testing tool, DeepConcolic, with the vehicle tracking system, we found that (1) the overall system can be resilient to some adversarial examples thanks to the existence of other components, and (2) the overall system presents an extra level of uncertainty which cannot be determined by analysing the deep learning components only. This research suggests the need for novel verification and validation methods for learning-enabled systems.

## Learning to Predict

- Spatiotemporal Representation Learning with GAN Trained LSTM-LSTM Networks

    Author: Fu, Yiwei | The Pennsylvania State University
    Author: Sen, Shiraj | General Electric
    Author: Theurer, Charles | GE Research
    Author: Reimann, Johan | GE Research
 
    keyword: Deep Learning in Robotics and Automation; Computer Vision for Automation; Model Learning for Control

    Abstract : Learning robot behaviors in unstructured environments often requires handcrafting the features for a given task. In this paper, we present and evaluate an unsupervised representation learning architecture, Layered Spatiotemporal Memory Long Short-Term Memory (LSTM-LSTM), that learns the underlying representation without knowledge of the task. The goal of this architecture is to learn the dynamics of the environment from high-dimensional raw video inputs. Using a Generative Adversarial Network (GAN) framework with the proposed network, this architecture is able to learn a spatiotemporal representation in its lower-dimensional latent space directly from raw input sequences. We show that our approach learns the spatial and temporal information simultaneously as opposed to a two-stage learning approach of alternating between training a Convolutional Neural Network (ConvNet) and a Long Short-Term Network (LSTM). Furthermore, by using LSTM-LSTM cells that shrink in size with the increase in the number of layers, the network learns a hierarchical representation with a low-dimensional representation at the top layer. We show that this architecture achieves state-of-the-art results with a substantially lower-dimensional representation than existing methods. We evaluate our approach on a video prediction task with standard benchmark datasets like Moving MNIST and KTH Action, as well as a simulated robot dataset.

- Belief Regulated Dual Propagation Nets for Learning Action Effects on Groups of Articulated Objects

    Author: Tekden, Ahmet | Bogazici University
    Author: Ugur, Emre | Bogazici University
    Author: Erdem, Erkut | Hacettepe University
    Author: Erdem, Aykut | Hacettepe University
    Author: Imre, Mert | Delft University of Technology
    Author: Seker, Muhammet Yunus | Bogazici University
 
    keyword: Deep Learning in Robotics and Automation

    Abstract : In recent years, graph neural networks have been successfully applied for learning the dynamics of complex and partially observable physical systems. However, their use in the robotics domain is, to date, still limited. In this paper, we introduce Belief Regulated Dual Propagation Networks (BRDPN), a general-purpose learnable physics engine, which enables a robot to predict the effects of its actions in scenes containing groups of articulated multi-part objects. Specifically, our framework extends recently proposed propagation networks (PropNets) and consists of two complementary components, a physics predictor and a belief regulator. While the former predicts the future states of the object(s) manipulated by the robot, the latter constantly corrects the robot's knowledge regarding the objects and their relations. Our results showed that after training in a simulator, the robot can reliably predict the consequences of its actions in object trajectory level and exploit its own interaction experience to correct its belief about the state of the environment, enabling better predictions in partially observable environments. Furthermore, the trained model was transferred to the real world and verified in predicting trajectories of pushed interacting objects whose joint relations were initially unknown. We compared BRDPN against PropNets, and showed that BRDPN performs consistently well. Moreover, BRDPN can adapt its physic predictions, since the relations can be predicted online.

- Deep Kinematic Models for Kinematically Feasible Vehicle Trajectory Predictions

    Author: Cui, Henggang | Uber Advanced Technologies Group
    Author: Nguyen, Thi Duong | Uber Technologies Inc
    Author: Chou, Fang-Chieh | Uber
    Author: Lin, Tsung-Han | Uber
    Author: Schneider, Jeff | Carnegie Mellon University
    Author: Bradley, David | Carnegie Mellon University
    Author: Djuric, Nemanja | Uber ATG
 
    keyword: AI-Based Methods; Deep Learning in Robotics and Automation

    Abstract : Self-driving vehicles (SDVs) hold great potential for improving traffic safety and are poised to positively affect the quality of life of millions of people. To unlock this potential one of the critical aspects of the autonomous technology is understanding and predicting future movement of vehicles surrounding the SDV. This work presents a deep-learning-based method for kinematically feasible motion prediction of such traffic actors. Previous work did not explicitly encode vehicle kinematics and instead relied on the models to learn the constraints directly from the data, potentially resulting in kinematically infeasible, suboptimal trajectory predictions. To address this issue we propose a method that seamlessly combines ideas from the AI with physically grounded vehicle motion models. In this way we employ best of the both worlds, coupling powerful learning models with strong feasibility guarantees for their outputs. The proposed approach is general, being applicable to any type of learning method. Extensive experiments using deep convnets on real-world data strongly indicate its benefits, outperforming the existing state-of-the-art.

- Human Driver Behavior Prediction Based on UrbanFlow

    Author: Qiao, Zhiqian | Carnegie Mellon University
    Author: Zhao, Jing | Carnegie Mellon University
    Author: Zhu, Jin | Carnegie Mellon University
    Author: Tyree, Zachariah | General Motors Research and Development
    Author: Mudalige, Priyantha | General Motors
    Author: Schneider, Jeff | Carnegie Mellon University
    Author: Dolan, John M. | Carnegie Mellon University
 
    keyword: Learning and Adaptive Systems; Behavior-Based Systems; AI-Based Methods

    Abstract : How autonomous vehicles and human drivers share public transportation systems is an important problem, as fully automatic transportation environments are still a long way off. Understanding human drivers' behavior can be beneficial for autonomous vehicle decision making and planning, especially when the autonomous vehicle is surrounded by human drivers who have various driving behaviors and patterns of interaction with other vehicles. In this paper, we propose an LSTM-based trajectory prediction method for human drivers which can help the autonomous vehicle make better decisions, especially in urban intersection scenarios. Meanwhile, in order to collect human drivers' driving behavior data in the urban scenario, we describe a system called UrbanFlow which includes the whole procedure from raw bird's-eye view data collection via drone to the final processed trajectories. The system is mainly intended for urban scenarios but can be extended to be used for any traffic scenarios.

- Environment Prediction from Sparse Samples for Robotic Information Gathering

    Author: Caley, Jeffrey | Oregon State University
    Author: Hollinger, Geoffrey | Oregon State University
 
    keyword: Deep Learning in Robotics and Automation; Environment Monitoring and Management

    Abstract : Robots often require a model of their environment to make informed decisions. In unknown environments, the ability to infer the value of a data field from a limited number of samples is essential to many robotics applications. In this work, we propose a neural network architecture to model these spatially correlated data fields based on a limited number of spatially continuous samples. Additionally, we provide a method based on biased loss functions to suggest future areas of exploration to minimize reconstruction error. We run simulated robotic information gathering trials on both the MNIST hand written digits dataset and a Regional Ocean Modeling System (ROMS) ocean dataset for ocean monitoring. Our method outperforms Gaussian process regression in both environments for modeling the data field and action selection.

- Predicting Pushing Action Effects on Spatial Object Relations by Learning Internal Prediction Models

    Author: Paus, Fabian | Karlsruhe Institute of Technology (KIT)
    Author: Teng, Huang | Karlsruhe Institute of Technology (KIT)
    Author: Asfour, Tamim | Karlsruhe Institute of Technology (KIT)
 
    keyword: Learning and Adaptive Systems; Manipulation Planning; Humanoid Robots

    Abstract : Understanding the effects of actions is essential for planning and executing robot tasks. By imagining possible action consequences, a robot can choose specific action parameters to achieve desired goal states. We present an approach for parametrizing pushing actions based on learning internal prediction models. These pushing actions must fulfill constraints given by a high-level planner, e.g., after the push the brown box must be to the right of the orange box. In this work, we represent the perceived scenes as object-centric graphs and learn an internal model, which predicts object pose changes due to pushing actions. We train this internal model on a large synthetic data set, which was generated in simulation, and record a smaller data set on the real robot for evaluation. For a given scene and goal state, the robot generates a set of possible pushing action candidates by sampling the parameter space and evaluating the candidates by comparing the predicted effect resulting from the internal model with the desired effect provided by the high-level planner. In the evaluation, we show that our model achieves high prediction accuracy in scenes with a varying number of objects and is able to generalize to scenes with more objects than seen during training. In experiments on the humanoid robot ARMAR-6, we validate the transfer from simulation and show that the learned internal model can be used to manipulate scenes into desired states effectively.

- Under the Radar: Learning to Predict Robust Keypoints for Odometry Estimation and Metric Localisation in Radar

    Author: Barnes, Dan | University of Oxford
    Author: Posner, Ingmar | Oxford University
 
    keyword: Deep Learning in Robotics and Automation; Autonomous Vehicle Navigation; Localization

    Abstract : This paper presents a self-supervised framework for learning to detect robust keypoints for odometry estimation and metric localisation in radar. By embedding a differentiable point-based motion estimator inside our architecture, we learn keypoint locations, scores and descriptors from localisation error alone. This approach avoids imposing any assumption on what makes a robust keypoint and crucially allows them to be optimised for our application. Furthermore the architecture is sensor agnostic and can be applied to most modalities. We run experiments on 280km of real world driving from the Oxford Radar RobotCar Dataset and improve on the state-of-the-art in point-based radar odometry, reducing errors by up to 45% whilst running an order of magnitude faster, simultaneously solving metric loop closures. Combining these outputs, we provide a framework capable of full mapping and localisation with radar in urban environments.

- SpAGNN: Spatially-Aware Graph Neural Networks for Relational Behavior Forecasting from Sensor Data

    Author: Casas Romero, Sergio | Uber ATG, University of Toronto
    Author: Gulino, Cole | Carnegie Mellon University
    Author: Liao, Renjie | Uber ATG Toronto
    Author: Urtasun, Raquel | University of Toronto
 
    keyword: Deep Learning in Robotics and Automation; Autonomous Agents; Computer Vision for Transportation

    Abstract : In this paper, we tackle the problem of relational behavior forecasting from sensor data. Towards this goal, we propose a novel, spatially-aware graph neural network (SpAGNN) that models the interactions between agents in the scene. Specifically, we first exploit convolutional neural networks to detect the actors and compute their initial states. We then design a graph neural network which iteratively updates the actor states via a message passing process. Inspired by Gaussian belief propagation, our SpAGNN takes the spatially-transformed parameters of the output distributions from neighboring agents as input. Our model is fully differentiable, thus enabling end-to-end training. Importantly, our probabilistic predictions can model uncertainty at the trajectory level. We demonstrate the effectiveness of our approach by achieving significant improvements over the state-of-the-art on two real-world, self-driving datasets.

- Any Motion Detector: Learning Class-Agnostic Scene Dynamics from a Sequence of LiDAR Point Clouds

    Author: Filatov, Artem | Yandex
    Author: Rykov, Andrey | Yandex
    Author: Murashkin, Viacheslav | Google
 
    keyword: Deep Learning in Robotics and Automation; Object Detection, Segmentation and Categorization

    Abstract : Object detection and motion parameters estima- tion are crucial tasks for self-driving vehicle safe navigation in a complex urban environment. In this work we propose a novel real-time approach of temporal context aggregation for motion detection and motion parameters estimation based on 3D point cloud sequence. We introduce an ego-motion compen- sation layer to achieve real-time inference with performance comparable to a naive odometric transform of the original point cloud sequence. Not only is the proposed architecture capable of estimating the motion of common road participants like vehicles or pedestrians but also generalizes to other object categories which are not present in training data. We also conduct an in-deep analysis of different temporal context aggregation strategies such as recurrent cells and 3D convolutions. Finally, we provide comparison results of our state-of-the-art model with existing solutions on KITTI Scene Flow dataset.

- Real Time Trajectory Prediction Using Deep Conditional Generative Models

    Author: Gomez-Gonzalez, Sebastian | Max Planck Institute for Intelligent Systems
    Author: Prokudin, Sergey | Max Planck Institute
    Author: Sch�lkopf, Bernhard | Max Planck Institute for Intelligent Systems
    Author: Peters, Jan | Technische Universitét Darmstadt
 
    keyword: Deep Learning in Robotics and Automation

    Abstract : Data driven methods for time series forecasting that quantify uncertainty open new important possibilities for robot tasks with hard real time constraints, allowing the robot system to make decisions that trade off between reaction time and accuracy in the predictions. Despite the recent advances in deep learning, it is still challenging to make long term accurate predictions with the low latency required by real time robotic systems. In this paper, we propose a deep conditional generative model for trajectory prediction that is learned from a data set of collected trajectories. Our method uses an encoder and decoder deep networks that maps complete or partial trajectories to a Gaussian distributed latent space and back, allowing for fast inference of the future values of a trajectory given previous observations. The encoder and decoder networks are trained using stochastic gradient variational Bayes. In the experiments, we show that our model provides more accurate long term predictions with a lower latency that popular models for trajectory forecasting like recurrent neural networks or physical models based on differential equations. Finally, we test our proposed approach in a robot table tennis scenario to evaluate the performance of the proposed method in a robotic task with hard real time constraints.

- Ambiguity in Sequential Data: Predicting Uncertain Futures with Recurrent Models

    Author: Berlati, Alessandro | Université Di Bologna
    Author: Scheel, Oliver | BMW Group
    Author: Di Stefano, Luigi | University of Bologna
    Author: Tombari, Federico | Technische Universitét M�nchen
 
    keyword: Deep Learning in Robotics and Automation; Autonomous Agents; Intelligent Transportation Systems

    Abstract : Ambiguity is inherently present in many machine learning tasks, but seldom accounted for, as most models only output a single prediction. In this work we propose a general framework to handle ambiguous predictions with sequential data, which is of special importance, as often multiple futures are equally likely. Our approach can be applied to the most common recurrent architectures and can be used with any loss function. Additionally, we introduce a novel metric for ambiguous problems, which is better suited to account for uncertainties and coincides with our intuitive understanding of correctness in the presence of multiple labels. We test our method on several experiments and across diverse tasks dealing with time series data, such as trajectory forecasting and maneuver prediction, achieving promising results.

- Where and When: Event-Based Spatiotemporal Trajectory Prediction from the iCub's Point-Of-View

    Author: Monforte, Marco | IIT
    Author: Arriandiaga, Ander | Istituto Italiano Di Tecnologia
    Author: Glover, Arren | Istituto Italiano Di Tecnologia
    Author: Bartolozzi, Chiara | Istituto Italiano Di Tecnologia
 
    keyword: Deep Learning in Robotics and Automation; Humanoid Robots; Neurorobotics

    Abstract : Fast, non-linear trajectories have been shown to be more accurately visually measured, and hence predicted, when sampled spatially (that is when the target position changes) rather than temporally, i.e. at a fixed-rate as in traditional frame-based cameras. Event-cameras, with their asynchronous, low latency information stream, allow for spatial sampling with very high temporal resolution, improving the quality of the data and the accuracy of post-processing operations. This paper investigates the use of Long Short-Term Memory (LSTM) networks with event-cameras spatial sampling for trajectory prediction. We show the benefit of using an Encoder-Decoder architecture over parameterised models for regression on event-based human-to-robot handover trajectories. In particular, we exploit the temporal information associated to the events stream to predict not only the incoming spatial trajectory points, but also when these will occur in time. After having studied the proper LSTM input/output sequence length, the network performance are compared to other regression models. Then, prediction behavior and computational time are analysed for the proposed method. We carry out the experiment using an iCub robot equipped with event-cameras, addressing the problem from the robot perspective.

## Learning for Motion Planning 

- Learning of Key Pose Evaluation for Efficient Multi-Contact Motion Planner

    Author: Noda, Shintaro | The University of Tokyo
    Author: Murooka, Masaki | The University of Tokyo
    Author: Asano, Yuki | The University of Tokyo
    Author: Ishizaki, Ryusuke | Honda Research Institute Japan Co, .Ltd
    Author: Kawakami, Tomohiro | Honda R&amp;D Co., Ltd
    Author: Watabe, Tomoki | Honda R&amp;D Co., Ltd
    Author: Okada, Kei | The University of Tokyo
    Author: Yoshiike, Takahide | Honda Research Institute Japan
    Author: Inaba, Masayuki | The University of Tokyo
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : It is necessary to use not only foot but also hand, knee and other body parts to support body weight for locomotion in uneven terrain. Such multi-contact motion planning is an important research topic including lots of previous works; however, a problem of computational speed of planning is still remaining. In this paper, we propose a learning-based algorithm to speed up the planning. The algorithm reduces replanning of contact states by learning an evaluation function of key pose to reach goal. We investigated the learning performance by comparing three neural network configurations and two activation function. This research aims at achieving robust robotics system in unknown environments.

- Differentiable Gaussian Process Motion Planning

    Author: Bhardwaj, Mohak | University of Washington
    Author: Boots, Byron | University of Washington
    Author: Mukadam, Mustafa | Facebook AI Research
 
    keyword: Motion and Path Planning; Learning and Adaptive Systems

    Abstract : Modern trajectory optimization based approaches to motion planning are fast, easy to implement, and effective on a wide range of robotics tasks. However, trajectory optimization algorithms have parameters that are typically set in advance (and rarely discussed in detail). Setting these parameters properly can have a significant impact on the practical performance of the algorithm, sometimes making the difference between finding a feasible plan or failing at the task entirely. We propose a method for leveraging past experience to learn how to automatically adapt the parameters of Gaussian Process Motion Planning (GPMP) algorithms. Specifically, we propose a differentiable extension to the GPMP2 algorithm, so that it can be trained end-to-end from data. We perform several experiments that validate our algorithm and illustrate the benefits of our proposed learning-based approach to motion planning.

- Learn and Link: Learning Critical Regions for Efficient Planning

    Author: Molina, Daniel | Arizona State University
    Author: Kumar, Kislay | Arizona State University
    Author: Srivastava, Siddharth | University of California Berkeley
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation; Learning from Demonstration

    Abstract : This paper presents a new approach to learning for motion planning (MP) where critical regions of an environment are learned from a given set of motion plans and used to improve performance on new environments and problem instances. We introduce a new suite of sampling-based motion planners, Learn and Link. Our planners leverage critical regions to overcome the limitations of uniform sampling, while still maintaining guarantees of correctness inherent to sampling-based algorithms. We also show that convolutional neural networks (CNNs) can be used to identify critical regions for motion planning problems. We evaluate Learn and Link against planners from the Open Motion Planning Library (OMPL) using an extensive suite of experiments on challenging motion planning problems. We show that our approach requires far less planning time than existing sampling-based planners.

- What the Constant Velocity Model Can Teach Us about Pedestrian Motion Prediction

    Author: Sch�ller, Christoph | Fortiss GmbH
    Author: Aravantinos, Vincent | Fortiss
    Author: Lay, Florian | Fortiss, Technical University of Munich
    Author: Knoll, Alois | Tech. Univ. Muenchen TUM
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : Pedestrian motion prediction is a fundamental task for autonomous robots and vehicles to operate safely. In recent years many complex approaches based on neural networks have been proposed to address this problem. In this work we show that - surprisingly - a simple Constant Velocity Model can outperform even state-of-the-art neural models. This indicates that either neural networks are not able to make use of the additional information they are provided with, or that this information is not as relevant as commonly believed. Therefore, we analyze how neural networks process their input and how it impacts their predictions. Our analysis reveals pitfalls in training neural networks for pedestrian motion prediction and clarifies false assumptions about the problem itself. In particular, neural networks implicitly learn environmental priors that negatively impact their generalization capability, the motion history of pedestrians is irrelevant and interactions are too complex to predict. Our work shows how neural networks for pedestrian motion prediction can be thoroughly evaluated and our results indicate which research directions for neural motion prediction are promising in future.

- Path Planning with Local Motion Estimations

    Author: Guzzi, Jerome | IDSIA, USI-SUPSI
    Author: Chavez-Garcia, R. Omar | University of Applied Sciences of Southern Switzerland
    Author: Nava, Mirko | IDSIA
    Author: Gambardella, Luca | USI-SUPSI
    Author: Giusti, Alessandro | IDSIA Lugano, SUPSI
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation; Probability and Statistical Methods

    Abstract : We introduce a novel approach to long-range path planning that relies on a learned model to predict the outcome of local motions using possibly partial knowledge. The model is trained from a dataset of trajectories acquired in a self-supervised way. Sampling-based path planners use this component to evaluate edges to be added to the planning tree. We illustrate the application of this pipeline with two robots: a complex, simulated, quadruped robot (ANYmal) moving on rough terrains; and a simple, real, differential-drive robot (Mighty Thymio), whose geometry is assumed unknown, moving among obstacles. We quantitatively evaluate the model performance in predicting the outcome of short moves and long-range paths; finally, we show that planning results in reasonable paths.

- Scene Compliant Trajectory Forecast with Agent-Centric Spatio-Temporal Grids

    Author: Ridel, Daniela | University of Sao Paulo
    Author: Deo, Nachiket | UC San Diego
    Author: Wolf, Denis Fernando | University of Sao Paulo
    Author: Trivedi, Mohan | University of California San Diego (UCSD)
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation; Intelligent Transportation Systems

    Abstract : Forecasting long-term human motion is a challenging task due to the non-linearity, multi-modality and inherent uncertainty in future trajectories. The underlying scene and past motion of agents can provide useful cues to predict their future motion. However, the heterogeneity of the two inputs poses a challenge for learning a joint representation of the scene and past trajectories. To address this challenge, we propose a model based on grid representations to forecast agent trajectories. We represent the past trajectories of agents using binary 2-D grids, and the underlying scene as a RGB birds-eye view (BEV) image, with an agent-centric frame of reference. We encode the scene and past trajectories using convolutional layers and generate trajectory forecasts using a Convolutional LSTM (ConvLSTM) decoder. Results on the publicly available Stanford Drone Dataset (SDD) show that our model outperforms prior approaches and outputs realistic future trajectories that comply with scene structure and past motion.

- A Data-Driven Planning Framework for Robotic Texture Painting on 3D Surfaces

    Author: Vempati, Anurag Sai | ETH Zurich, Disney Research Zurich
    Author: Siegwart, Roland | ETH Zurich
    Author: Nieto, Juan | ETH Zurich
 
    keyword: Deep Learning in Robotics and Automation; Visual Learning; Task Planning

    Abstract : Painting textures on 3D surfaces requires an understanding of the surface geometry, paint flow and paint mixing. This work formulates automated painting as a planning problem and proposes a solution based on a self-supervised learning framework that enables a robot to paint monochromatic non-uniform textures on 3D surfaces. We developed a method that iteratively decides the actions to take based on constant feedback of the painting process. Inspired by some recent results, we formulate our solution using a recurrent neural network (RNN) to decide where and what to paint on the surface at each time instant. Specifically, the paint delivery tool's flow rate, orientation and position relative to the surface at each time instant are evaluated. This data can then be processed by a robot's planner of choice for generating a painting mission that can achieve the desired end result. We evaluate the proposed approach by providing qualitative and quantitative results of the different components. Furthermore, we validate the effectiveness of the approach for the application by providing renderings from a paint simulation environment and show how a robot executes the planned painting mission on a generic 3D surface.

- Learned Critical Probabilistic Roadmaps for Robotic Motion Planning

    Author: Ichter, Brian | Google Brain
    Author: Schmerling, Edward | Waymo
    Author: Lee, Tsang-Wei Edward | Google
    Author: Faust, Aleksandra | Google Brain
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : Sampling-based motion planning techniques have emerged as an efficient algorithmic paradigm for solving complex motion planning problems. These approaches use a set of probing samples to construct an implicit graph representation of the robot's state space, allowing arbitrarily accurate representations as the number of samples increases to infinity. In practice, however, solution trajectories only rely on a few critical states, often defined by structure in the state space (e.g., doorways). In this work we propose a general method to identify these critical states via graph-theoretic techniques (betweenness centrality) and learn to predict criticality from only local environment features. These states are then leveraged more heavily via global connections within a hierarchical graph, termed Critical Probabilistic Roadmaps. Critical PRMs are demonstrated to achieve up to three orders of magnitude improvement over uniform sampling, while preserving the guarantees and complexity of sampling-based motion planning. A video is available at https://youtu.be/AYoD-pGd9ms.

- Learning Heuristic A*: Efficient Graph Search Using Neural Network

    Author: Kim, Soonkyum | Korea Institute of Science and Technology
    Author: An, Byungchul | Korea Institute of Science and Technology
 
    keyword: Motion and Path Planning; Deep Learning in Robotics and Automation

    Abstract : In this paper, we consider the path planning problem on a graph. To reduce computation load by efficiently exploring the graph, we model the heuristic function as a neural network, which is trained by a training set derived from optimal paths to estimate the optimal cost between a pair of vertices on the graph. As such heuristic function cannot be proved to be an admissible heuristic to guarantee the global optimality of the path, we adapt an admissible heuristic function for the terminating criteria. Thus, proposed Learning Heuristic A* (LHA*) guarantees the bounded suboptimality of the path. The performance of LHA* was demonstrated by simulations in a maze-like map and compared with the performance of weighted A* with the same suboptimality bound.

- 3D-CNN Based Heuristic Guided Task-Space Planner for Faster Motion Planning

    Author: Terasawa, Ryo | Sony Corporation
    Author: Ariki, Yuka | Sony Corporation
    Author: Narihira, Takuya | Sony Corporation
    Author: Tsuboi, Toshimitsu | Sony Corporation
    Author: Nagasaka, Kenichiro | Sony Corporation
 
    keyword: Deep Learning in Robotics and Automation; Motion and Path Planning; Learning and Adaptive Systems

    Abstract : Motion planning is important in a wide variety of applications such as robotic manipulation. However, it is still challenging to reliably find a collision-free path within a reasonable time. To address the issue, this paper proposes a novel framework which combines a sampling-based planner and deep learning for faster motion planning, focusing on heuristics. The proposed method extends Task-Space Rapidly-exploring Random Trees (TS-RRT) to guide the trees with a "heuristic map" where every voxel has a cost-to-go value toward the goal. It also utilizes fully convolutional neural networks (CNNs) for producing more appropriate heuristic maps, rather than manually-designed heuristics. To verify the effectiveness of the proposed method, experiments for motion planning using a real environment and mobile manipulator are carried out. The results indicate that it outperforms the existing planners, especially in terms of the average planning time with smaller variance.

- Learned Sampling Distributions for Efficient Planning in Hybrid Geometric and Object-Level Representations

    Author: Liu, Katherine | MIT
    Author: Stadler, Martina | Massachusetts Institute of Technology
    Author: Roy, Nicholas | Massachusetts Institute of Technology
 
    keyword: Deep Learning in Robotics and Automation; Autonomous Vehicle Navigation; Motion and Path Planning

    Abstract : We would like to enable a robotic agent to quickly and intelligently find promising trajectories through structured, unknown environments. Many approaches to navigation in unknown environments are limited to considering geometric information only, which leads to myopic behavior. In this work, we show that learning a sampling distribution that incorporates both geometric information and explicit, object-level semantics for sampling-based planners enables efficient planning at longer horizons in partially-known environments. We demonstrate that our learned planner is up to 2.7 times more likely to find a plan than the baseline, and can result in up to a 16% reduction in traversal costs as calculated by linear regression. We also show promising qualitative results on real-world data.

- Deep Visual Heuristics: Learning Feasibility of Mixed-Integer Programs for Manipulation Planning

    Author: Driess, Danny | University of Stuttgart
    Author: Oguz, Ozgur S. | Technical University of Munich
    Author: Ha, Jung-Su | University of Stuttgart
    Author: Toussaint, Marc | University of Stuttgart
 
    keyword: Deep Learning in Robotics and Automation; Manipulation Planning; Learning and Adaptive Systems

    Abstract : In this paper, we propose a deep neural network that predicts the feasibility of a mixed-integer program from visual input for robot manipulation planning. Integrating learning into task and motion planning is challenging, since it is unclear how the scene and goals can be encoded as input to the learning algorithm in a way that enables to generalize over a variety of tasks in environments with changing numbers of objects and goals. To achieve this, we propose to encode the scene and the target object directly in the image space.<p>Our experiments show that our proposed network generalizes to scenes with multiple objects, although during training only two objects are present at the same time. By using the learned network as a heuristic to guide the search over the discrete variables of the mixed-integer program, the number of optimization problems that have to be solved to find a feasible solution or to detect infeasibility can greatly be reduced.

- 

## Motion Control of Manipulators
- Segmentation and Averaging of sEMG Muscle Activations Prior to Synergy Extraction

    Author: Costa, �lvaro | BSI-TOYOTA Collaboration Center in the Nagoya Science Park Resea
    Author: Iáñez, Eduardo | Miguel Hern�ndez University of Elche
    Author: Sonoo, Moeka | RIKEN, CBS-TOYOTA Collaboration Center in the Nagoya Science Par
    Author: Okajima, Shotaro | RIKEN
    Author: Yamasaki, Hiroshi | BSI-TOYOTA Collaboration Center in the Nagoya Science Park Resea
    Author: Ueda, Sayako | RIKEN, CBS-TOYOTA Collaboration Center in the Nagoya Science Par
    Author: Shimoda, Shingo | RIKEN
 
    keyword: Motion Control; Rehabilitation Robotics; Kinematics

    Abstract : Averaging electromyographic activity prior to muscle synergy computation is a common method employed to compensate for the inter-repetition variability usually associated with this kind of physiological recording. Capturing muscle synergies requires the preservation of accurate temporal and spatial information for muscle activity. The natural variation in electromyography data across consecutive repetitions of the same task raises several related challenges that make averaging a non-trivial process. Duration and triggering times of muscle activity generally vary across different repetitions of the same task. Therefore, it is necessary to define a robust methodology to segment and average muscle activity that deals with these issues. Emerging from this need, the present work proposes a standard protocol for segmenting and averaging muscle activations from periodic motions in a way that accurately preserves the temporal and spatial information contained in the original data and enables the isolation of a single averaged motion period. This protocol has been validated with muscle activity data recorded from 15 participants performing elbow flexion/extension motions, a series of actions driven by well-established muscle synergies. Using the averaged data, muscle synergies were computed, permitting their behavior to be compared with previous results related to the evaluated task. The comparison between the method proposed and a widely used methodology based on motion flags, shown

- Energy-Optimal Cooperative Manipulation Via Provable Internal-Force Regulation

    Author: Verginis, Christos | Electrical Engineering, KTH Royal Institute of Technology
    Author: Dimarogonas, Dimos V. | KTH Royal Institute of Technology
 
    keyword: Motion Control of Manipulators; Cooperating Robots

    Abstract : This paper considers the optimal cooperative robotic manipulation problem in terms of energy resources. In particular, we consider rigid cooperative manipulation systems, i.e., with rigid grasping contacts, and study energy-optimal conditions in the sense of minimization of the arising internal forces, which are inter-agent forces that do not contribute to object motion. Firstly, we use recent results to derive a closed form expression for the internal forces. Secondly, by using a standard inverse dynamics control protocol, we provide novel conditions on the force distribution to the robotic agents for provable internal force minimization. Moreover, we derive novel results on the provable achievement of a desired non-zero inter-agent internal force vector. Extensive simulation results in a realistic environment verify the theoretical analysis.

- Robot Telekinesis: Application of a Unimanual and Bimanual Object Manipulation Technique to Robot Control

    Author: Lee, Joon Hyub | Korea Advanced Institute of Science and Technology
    Author: Kim, Yongkwan | Korea Advanced Institute of Science and Technology
    Author: An, Sang-Gyun | Korea Advanced Institute of Science and Technology
    Author: Bae, Seok-Hyung | Korea Advanced Institute of Science and Technology
 
    keyword: Motion Control of Manipulators; Telerobotics and Teleoperation; Human-Centered Robotics

    Abstract : Unlike large and dangerous industrial robots at production lines in factories that are strictly fenced off, collaborative robots are smaller and safer and can be installed adjacent to human workers and collaborate with them. However, controlling and teaching new moves to collaborative robots can be difficult and time-consuming when using existing methods, such as pressing buttons on a teaching pendant and physically grabbing and moving the robot via direct teaching. We present Robot Telekinesis, a novel robot interaction technique that lets the user remotely control the movement of the end effector of a robot arm with unimanual and bimanual hand gestures that closely resemble handling a physical object. Through formal evaluation, we show that using a teaching pendant is slow and confusing and that direct teaching is fast and intuitive but physically demanding. Robot Telekinesis is as fast and intuitive as direct teaching without the need for physical contact or physical effort.

- A Set-Theoretic Approach to Multi-Task Execution and Prioritization

    Author: Notomista, Gennaro | Georgia Institute of Technology
    Author: Mayya, Siddharth | University of Pennsylvania
    Author: Selvaggio, Mario | Université Degli Studi Di Napoli Federico II
    Author: Santos, Mar�a | Georgia Institute of Technology
    Author: Secchi, Cristian | Univ. of Modena &amp; Reggio Emilia
 
    keyword: Motion Control of Manipulators; Redundant Robots

    Abstract : Executing multiple tasks concurrently is important in many robotic applications. Moreover, the prioritization of tasks is essential in applications where safety-critical tasks need to precede application-related objectives, in order to protect both the robot from its surroundings and vice versa. Furthermore, the possibility of switching the priority of tasks during their execution gives the robotic system the flexibility of changing its objectives over time. In this paper, we present an optimization-based task execution and prioritization framework that lends itself to the case of time-varying priorities as well as variable number of tasks. We introduce the concept of extended set-based tasks, encode them using control barrier functions, and execute them by means of a constrained-optimization problem, which can be efficiently solved in an online fashion. Finally, we show the application of the proposed approach to the case of a redundant robotic manipulator.

- Task Space Control of Articulated Robot Near Kinematic Singularity: Forward Dynamics Approach

    Author: Lee, Donghyeon | Pohang University of Science and Technology(POSTECH)
    Author: Lee, Woongyong | POSTECH
    Author: Park, Jonghoon | Neuromeka
    Author: Chung, Wan Kyun | POSTECH
 
    keyword: Motion Control; Kinematics; Compliance and Impedance Control

    Abstract : In this study, a forward dynamics-based control (FDC) framework is proposed for task space control of a non-redundant robot manipulator. The FDC framework utilizes forward dynamic robot simulation and an impedance controller to solve the inverse kinematics problem. For the practical use of the proposed control framework, the accuracy, robustness, and stability of robot motion are considered. Taking advantage of the stability of the implicit Euler method, a high-gain PD controller enables accurate end-effector pose tracking in the task space without losing stability even near the kinematic singularities. Also, the robustness of the controller is enhanced by borrowing the structure of the nonlinear robust internal-loop compensator. Lastly, the selective joint damping injection and spring force saturation are applied to the impedance controller so that the robot motion can always stay within the given dynamic constraints. This study suggests a new, effective solution for the kinematic singularity problem of non-redundant robot manipulators.

- Variable Impedance Control in Cartesian Latent Space While Avoiding Obstacles in Null Space

    Author: Parent, David | Institut De Robòtica I Informàtica Industrial (CSIC-UPC)
    Author: Colomé, Adrià | Institut De Robòtica I Informàtica Industrial (CSIC-UPC), Q28180
    Author: Torras, Carme | Csic - Upc
 
    keyword: Motion Control of Manipulators; Compliance and Impedance Control; Redundant Robots

    Abstract : Human-robot interaction is one of the keys of assistive robots. Robots are expected to be compliant with people but at the same time correctly perform the tasks. In such applications, Cartesian impedance control is preferred over joint control, as the desired interaction and environmental feedback can be described more naturally, and the force to be exerted by the robot can be readily adjusted.<p>This paper addresses the problem of controlling a robot arm in the operational space with variable stiffness so as to continuously adapt the force exerted in each phase of motion according to the precision requirements. Moreover, performing dimensionality reduction we can separate the degrees of freedom (DoF) relevant for the task from the redundant ones. The stiffness of the former can be adjusted constantly to achieve the required accuracy, while task-redundant DoF can be used to achieve other goals such as avoiding obstacles by moving in the directions where accuracy is not critical. The designed method is tested teaching the robot to give water to drink to a model of human head. Our empirical results demonstrate that the robot can learn precision requirements from demonstration. Furthermore, dimensionality reduction is proved to be useful to avoid obstacles.

- 

## Computer Vision for Medical Robots
- Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments

    Author: Ni, ZhenLiang | Chinese Academy of Sciences
    Author: Bian, Gui-Bin | Institute of Automation, Chinese Academy of Sciences
    Author: Hou, Zeng-Guang | Chinese Academy of Science
    Author: Zhou, Xiao-Hu | Institute of Automation Chinese Academy of Sciences
    Author: Xie, Xiaoliang | Institutation of Automation, Chinese Academy of Sciences
    Author: Li, Zhen | Institute of Automation, Chinese Academy of Sciences
 
    keyword: Computer Vision for Medical Robotics; Object Detection, Segmentation and Categorization

    Abstract : The real-time segmentation of surgical instruments plays a crucial role in robot-assisted surgery. However, it is still a challenging task to implement deep learning models to do real-time segmentation for surgical instruments due to their high computational costs and slow inference speed. In this paper, we propose an attention-guided lightweight network (LWANet), which can segment surgical instruments in real-time. LWANet adopts encoder-decoder architecture, where the encoder is the lightweight network MobileNetV2, and the decoder consists of depthwise separable convolution, attention fusion block, and transposed convolution. Depthwise separable convolution is used as the basic unit to construct the decoder, which can reduce the model size and computational costs. Attention fusion block captures global contexts and encodes semantic dependencies between channels to emphasize target regions, contributing to locating the surgical instrument. Transposed convolution is performed to upsample feature maps for acquiring refined edges. LWANet can segment surgical instruments in real-time while takes little computational costs. Based on 960�544 inputs, its inference speed can reach 39 fps with only 3.39 GFLOPs. Also, it has a small model size and the number of parameters is only 2.06 M. The proposed network is evaluated on two datasets. It achieves state-of-the-art performance 94.10% mean IOU on Cata7 and obtains a new record on EndoVis 2017 with a 4.10% increase on mean IOU.

- Automated Robotic Breast Ultrasound Acquisition Using Ultrasound Feedback

    Author: Welleweerd, Marcel Klaas | University of Twente
    Author: De Groot, Antonius Gerardus | University of Twente
    Author: de Looijer, Stijn | Delft University of Technology
    Author: Siepel, Fran�oise J | University of Twente
    Author: Stramigioli, Stefano | University of Twente
 
    keyword: Computer Vision for Medical Robotics; Visual Servoing; Medical Robots and Systems

    Abstract : Current challenges in automated robotic breast ultrasound (US) acquisitions include keeping acoustic coupling between the breast and the US probe, minimizing tissue deformations and safety. In this paper, we present how an autonomous 3D breast US acquisition can be performed utilizing a 7DOF robot equipped with a linear US transducer. Robotic 3D breast US acquisitions would increase the diagnostic value of the modality since they allow patient specific scans and have a high reproducibility, accuracy and efficiency. Additionally, 3D US acquisitions allow more flexibility in examining the breast and simplify registration with preoperative images like MRI. To overcome the current challenges, the robot follows a reference-based trajectory adjusted by a visual servoing algorithm. The reference trajectory is a patient specific trajectory coming from e.g. an MRI. The visual servoing algorithm commands in-plane rotations and corrects the probe contact based on confidence maps. A safety aware, intrinsically passive framework is utilised to actuate the robot. The approach is illustrated with experiments on a phantom, which show that the robot only needs minor pre-procedural information to consistently image the phantom while relying mainly on US feedback.

- Robust and Accurate 3D Curve to Surface Registration with Tangent and Normal Vectors

    Author: Min, Zhe | The Chinese University of Hong Kong
    Author: Zhu, Delong | The Chinese University of Hong Kong
    Author: Meng, Max Q.-H. | The Chinese University of Hong Kong
 
    keyword: Computer Vision for Medical Robotics; Medical Robots and Systems

    Abstract : This paper presents a robust and accurate approach for the rigid registration of pre-operative and intra-operative point sets in textit{image-guided surgery (IGS)}. Three challenges are identified in the textit{pre-to-intraoperative} registration: the intra-operative 3D data (usually forms a 3D curve in space) (1) is often contaminated with noise and outliers; (2) usually only covers a partial region of the whole pre-operative model; (3) is usually sparse. To tackle those challenges, we utilize the tangent vectors extracted from the sparse intra-operative textit{data} points and the normal vectors extracted from the pre-operative textit{model} points. % the normal vectors and the tangent vectors are first extracted from the pre-and-intra operative point set and utilized in the registration. Our first contribution is to formulate a novel probabilistic distribution of the error between a pair of corresponding tangent and normal vectors. The second contribution is, based on the novel distribution, we formulate the registration of two multi-dimensional (6D) point sets as a textit{maximum likelihood (ML)} problem and solve it under the textit{expectation maximization (EM)} framework. Our last contribution is, in order to facilitate the computation process, the derivatives of the objective function with respect to desired parameters are presented.

- Single Shot Pose Estimation of Surgical Robot Instruments' Shafts from Monocular Endoscopic Images

    Author: Yoshimura, Masakazu | The University of Tokyo
    Author: Marques Marinho, Murilo | The University of Tokyo
    Author: Harada, Kanako | The University of Tokyo
    Author: Mitsuishi, Mamoru | The University of Tokyo
 
    keyword: Computer Vision for Medical Robotics; AI-Based Methods; Calibration and Identification

    Abstract : Surgical robots are used to perform minimally invasive surgery and alleviate much of the burden imposed on surgeons. Our group has developed a surgical robot to aid in the removal of tumors at the base of the skull via access through the nostrils. To avoid injuring the patients, a collision-avoidance algorithm that depends on having an accurate model for the poses of the instruments' shafts is used. Given that the model's parameters can change over time owing to interactions between instruments and other disturbances, the online estimation of the poses of the instrument's shaft is essential. In this work, we propose a new method to estimate the pose of the surgical instruments' shafts using a monocular endoscope. Our method is based on the use of an automatically annotated training dataset and an improved pose-estimation deep-learning architecture. In preliminary experiments, we show that our method can surpass state of the art vision-based marker-less pose estimation techniques (providing an error decrease of 55% in position estimation, 64% in pitch, and 69% in yaw) by using artificial images.

- End-To-End Real-Time Catheter Segmentation with Optical Flow-Guided Warping During Endovascular Intervention

    Author: Nguyen, Anh | Imperial College London
    Author: Kundrat, Dennis | Imperial College London
    Author: Dagnino, Giulio | Imperial College London
    Author: Chi, Wenqiang | Imperial College London
    Author: Abdelaziz, Mohamed Essam Mohamed Kassem | Imperial College London
    Author: Ma, YingLiang | School of Computing, Electronics and Mathematics, Coventry Unive
    Author: Kwok, Trevor M Y | Imperial College London
    Author: Riga, Celia | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Computer Vision for Medical Robotics; Object Detection, Segmentation and Categorization

    Abstract : Accurate real-time catheter segmentation is an important pre-requisite for robot-assisted endovascular intervention. Most of the existing learning-based methods for catheter segmentation and tracking are only trained on small-scale datasets or synthetic data due to the difficulties of ground-truth annotation. Furthermore, the temporal continuity in intraoperative imaging sequences is not fully utilised. In this paper, we present FW-Net, an end-to-end and real-time deep learning framework for endovascular intervention. The proposed FW-Net has three modules: a segmentation network with encoder-decoder architecture, a flow network to extract optical flow information, and a novel flow-guided warping function to learn the frame-to-frame temporal continuity. We show that by effectively learning temporal continuity, the network can successfully segment and track the catheters in real-time sequences using only raw ground-truth for training. Detailed validation results confirm that our FW-Net outperforms state-of-the-art techniques while achieving real-time performance.

- Pathological Airway Segmentation with Cascaded Neural Networks for Bronchoscopic Navigation

    Author: Zhang, Hanxiao | Imperial College London
    Author: Shen, Mali | The Hamlyn Centre for Robotic Surgery, Imperial College London
    Author: Shah, Pallav L | Imperial College London
    Author: Yang, Guang-Zhong | Shanghai Jiao Tong University
 
    keyword: Computer Vision for Medical Robotics; Visual Learning; Surgical Robotics: Planning

    Abstract : Robotic bronchoscopic intervention requires detailed 3D airway maps for both localisation and enhanced visualisation, especially at peripheral airways. Patient-specific airway maps can be generated from preoperative chest CT scans. Due to pathological abnormalities and anatomical variations, automatically delineating the airway tree with distal branches is a challenging task. In the paper, we propose a cascaded 2D+3D model that has been tailored for airway segmentation from pathological CT scans. A novel 2D neural network is developed to generate the initial predictions where the peripheral airways are refined by a 3D adversarial training model. A sampling strategy based on a sequence of morphological operations is employed for the concatenation of the 2D and 3D models. The method has been validated on 20 pathological CT scans with results demonstrating improved segmentation accuracy and consistency, especially in peripheral airways.

## Grippers and Other End-Effectors

- A Novel Underactuated End-Effector for Planar Sequential Grasping of Multiple Objects

    Author: Mucchiani, Caio | University of Pennsylvania
    Author: Yim, Mark | University of Pennsylvania
 
    keyword: Grasping; Mechanism Design

    Abstract : We propose a serpentine type tendon driven underactuated end-effector design with a closing mechanism that is triggered upon contact with an object. This end-effector can grasp objects without knowing the size a priori and is able to grasp a new object while securing another one previously grasped, and so grasp multiple objects sequentially with a single DOF actuation. Design parameters based on the object dimensions are proposed. A low-cost prototype demonstrates two implementations (radius estimation and autonomous grasp of circular objects by torque control, and sequential grasps of multiple objects) of the end-effector through several experiments. A method for estimating applied internal forces is also proposed. This end-effector can benefit robotic manipulation in tasks such as fetching applications, industrial pick-and place of single or multiple objects and human-robot hand-off interactions.

- Design and Analysis of a Synergy-Inspired Three-Fingered Hand

    Author: Chen, Wenrui | Hunan University
    Author: Xiao, ZhiLan | Hunan University
    Author: Lu, JingWen | HuNan University
    Author: Zhao, Zilong | Hunan University
    Author: Wang, Yaonan | Hunan University
 
    keyword: Multifingered Hands; Underactuated Robots; Compliant Joint/Mechanism

    Abstract : Hand synergy from neuroscience provides an effective tool for anthropomorphic hands to realize versatile grasping with simple planning and control. This paper aims to extend the synergy-inspired design from anthropomorphic hands to multi-fingered robot hands. The synergy-inspired hands are not necessarily humanoid in morphology but perform primary characteristics and functions similar to the human hand. At first, the biomechanics of hand synergy is investigated. Three biomechanical characteristics of the human hand synergy are explored as a basis for the mechanical simplification of the robot hands. Secondly, according to the synergy characteristics, a three-fingered hand is designed, and its kinematic model is developed for the analysis of some typical grasping and manipulation functions. Finally, a prototype is developed and preliminary grasping experiments validate the effectiveness of the design and analysis.

- Multiplexed Manipulation: Versatile Multimodal Grasping Via a Hybrid Soft Gripper

    Author: Chin, Lillian | Massachusetts Institute of Technology
    Author: Barscevicius, Felipe | MIT
    Author: Lipton, Jeffrey | University of Washington
    Author: Rus, Daniela | MIT
 
    keyword: Grippers and Other End-Effectors; Dexterous Manipulation; Soft Robot Materials and Design

    Abstract : The success of hybrid suction + parallel-jaw grippers in the Amazon Robotics/Picking Challenge have demonstrated the effectiveness of multimodal grasping approaches. However, existing multimodal grippers combine grasping modes in isolation and do not incorporate the benefits of compliance found in soft robotic manipulators. In this paper, we present a gripper that integrates three modes of grasping: suction, parallel jaw, and soft fingers. Using complaint handed shearing auxetics actuators as the foundation, this gripper is able to multiplex manipulation by creating unique grasping primitives through permutations of these grasping techniques. This gripper is able to grasp 88% of tested objects, 14% of which could only be grasped using a combination of grasping modes. The gripper is also able to perform in-hand object re-orientation of flat objects without the need for pre-grasp manipulation.

- Modeling, Optimization, and Experimentation of the ParaGripper for In-Hand Manipulation without Parasitic Rotation

    Author: Liu, Huan | University of Naples, Federico II
    Author: Zhao, Longhai | Shanghai Jiao Tong University
    Author: Siciliano, Bruno | Univ. Napoli Federico II
    Author: Ficuciello, Fanny | Université Di Napoli Federico II
 
    keyword: Grippers and Other End-Effectors; Dexterous Manipulation; Mechanism Design

    Abstract : Recently, underactuated robotic hands have been exploited for dexterous in-hand manipulation, after having been proven efficient in performing versatile adaptive grasps. However, the reported in-hand manipulation skills are usually associated with parasitic motion, which may complicate control and application of the hand. This paper presents the modeling, optimization and experimentation of the ParaGripper, an underactuated gripper capable of performing in-hand manipulation without parasitic rotation. The underactuated finger uses two serially connected parallelograms to ensure pure translation of the fingertips. If the object remains stationary within the fingertips, the gripper can translate the object without parasitic rotation. The kinematics and kinetostatics of the hand--object system are derived and the manipulation workspace is optimized. The ParaGripper is designed and fabricated according to suitable optimal parameters. Experiments show that the ParaGripper could perform non-parasitic in-hand manipulation and versatile adaptive grasps.

- Underactuated Gecko Adhesive Gripper for Simple and Versatile Grasp

    Author: Hirano, Daichi | Japan Aerospace Exploration Agency
    Author: Tanishima, Nobutaka | JAXA
    Author: Bylard, Andrew | Stanford University
    Author: Chen, Tony G. | Stanford University
 
    keyword: Grippers and Other End-Effectors; Biologically-Inspired Robots; Space Robotics and Automation

    Abstract : Gecko-inspired adhesives have several desirable characteristics in robotic grasping: controllable activation and deactivation of adhesion, ability to grasp and release with minimal disturbance, and grasping without the need of form closure. Previously proposed grippers with this technology either require a complex activation mechanism or multiple activation steps. In this paper, we present an underactuated gecko-inspired adhesive gripper that can grasp a wide range of curved surfaces using a single actuator through a simple tendon-driven mechanism that attaches and adheres in one step. We derive a theoretical model of the adhesive contact area and resulting gripper grasp force, which is verified experimentally. The actual performance of the proposed mechanism is demonstrated by successfully grasping several surfaces with different curvature diameters.

- Examining the Frictional Behavior of Primitive Contact Geometries for Use As Robotic Finger Pads

    Author: Leddy, Michael | Yale University
    Author: Dollar, Aaron | Yale University
 
    keyword: Grippers and Other End-Effectors; Grasping; Multifingered Hands

    Abstract : Prosthetic and robotic grippers rely on soft finger pads to better acquire objects of varying size, shape and surface. However, the frictional behavior of soft finger pads of different designs and geometries have yet to be quantitatively compared, in large part due to the difficulty in modeling soft contact mechanics. In this paper, we experimentally examine the frictional behavior of several common primitive contact geometries in terms of their performance under shear loads that would tend to cause the contact to slip and the grasp to potentially fail. The effective static and kinetic coefficients of friction were recorded for each finger pad under a range of common grasping loads. The results show that the variance in contact curvature, contact patch geometry and pressure distribution have influences on key parameters for grasping at low forces. The advantages and disadvantages of these simple geometries are discussed for design of single finger, multi-finger and manipulation-based robotic hands.

- Design of 3D-Printed Assembly Mechanisms Based on Special Wooden Joinery Techniques and Its Application to a Robotic Hand

    Author: Katsumaru, Akihiro | Ritsumeikan University
    Author: Ozawa, Ryuta | Meiji University
 
    keyword: Multifingered Hands; Mechanism Design

    Abstract : Recently, it has become possible to easily design and fabricate robotic systems in the laboratory and at home due to the recent development of 3D printer technology. On the other hand, the strength of the plastic materials used in reasonably priced 3D printers and the accuracy of the printed parts are generally low. These problems affect the part-joining quality. Therefore, this paper describes a design method inspired by ancient Japanese wooden joinery techniques for assembling 3D-printed parts and presents the design of a robotic hand as its application. The joinery techniques use special shapes to assemble components and allow us to assemble the robotic hand without glue, screws or nails and to easily disassemble it.

- Parallel Gripper with Displacement-Magnification Mechanism and Extendable Finger Mechanism

    Author: Tanaka, Junya | Toshiba Corporation
    Author: Sugahara, Atsushi | Toshiba Corporation
 
    keyword: Grippers and Other End-Effectors; Mechanism Design; Force and Tactile Sensing

    Abstract : We propose a gripper displacement-magnification mechanism and an extendable finger mechanism, both of which can be attached to a commercially available parallel gripper. We then verify the operation of the mechanism in order to expand applications of the parallel gripper. The displacement-magnification mechanism has a stacked rack-and-pinion system that doubles displacement. The extendable finger mechanism has two nails that extend and contract, reducing impact force and detecting changes in product height from expansion and contraction amounts. The parallel gripper has a width of 95 mm, a depth of 110 mm, and a height of 214 mm and weighs 1.36 kg. It has an open/close stroke of 60 mm, a gripping force of 7.4 N, and an opening/closing speed of 100 mm/s or more. Further, it was confirmed that the ends and inclinations of products can be reliably detected using the extending/contracting nail. The mechanism verification confirmed that our parallel gripper achieved the desired performance and is therefore useful.

- Sheet-Based Gripper Featuring Passive Pull-In Functionality for Bin Picking and for Picking up Thin Flexible Objects

    Author: Morino, Kota | Kanazawa University
    Author: Kikuchi, Shiro | YKK
    Author: Chikagawa, Shinichi | YKK AP
    Author: Izumi, Masakazu | YKK AP
    Author: Watanabe, Tetsuyou | Kanazawa University
 
    keyword: Factory Automation; Industrial Robots; Grippers and Other End-Effectors

    Abstract : This study investigates the effect of the surface texture of soft deformable fingertips on the maximum resistible force under dry and wet conditions, and proposes a new hybrid structure that provides a stable grasp under both conditions. One definition of stable grasp is the capability of balancing a large external force or moment while grasping. For soft fingertips, both the friction and surface deformation contribute to the stability. Therefore, we investigate the maximum resistible force, which is defined as the maximum tangential force at which the fingertip can maintain contact when applying and increasing the tangential/shear force. We investigate the slit textures with primitive patterns and demonstrate that the non-pattern performs the best under dry conditions, whereas the horizontal slit pattern performs the best under wet (oily) conditions. Based on this, a concentric hybrid texture of the two patterns is proposed, and its effectiveness is verified by a grasping test.

- An Origami-Inspired Variable Friction Surface for Increasing the Dexterity of Robotic Grippers

    Author: Lu, Qiujie | Imperial College London
    Author: Clark, Angus Benedict | Imperial College London
    Author: Shen, Matthew | Imperial College London
    Author: Rojas, Nicolas | Imperial College London
 
    keyword: Dexterous Manipulation; Mechanism Design; Grippers and Other End-Effectors

    Abstract : While the grasping capability of robotic grippers has shown significant development, the ability to manipulate objects within the hand is still limited. One explanation for this limitation is the lack of controlled contact variation between the grasped object and the gripper. For instance, human hands have the ability to firmly grip object surfaces, as well as slide over object faces, an aspect that aids the enhanced manipulation of objects within the hand without losing contact. In this letter, we present a parametric, origami-inspired thin surface capable of transitioning between a high friction and a low friction state, suitable for implementation as an epidermis in robotic fingers. A numerical analysis of the proposed surface based on its design parameters, force analysis, and performance in in-hand manipulation tasks is presented. Through the development of a simple two-fingered two-degree-of-freedom gripper utilizing the proposed variable-friction surfaces with different parameters, we experimentally demonstrate the improved manipulation capabilities of the hand when compared to the same gripper without changeable friction. Results show that the pattern density and valley gap are the main parameters that effect the in-hand manipulation performance. The origami-inspired thin surface with a higher pattern density generated a smaller valley gap and smaller height change, producing a more stable improvement of the manipulation capabilities of the hand.

- A Shape Memory Polymer Adhesive Gripper for Pick-And-Place Applications

    Author: Son, ChangHee | University of Illinois at Urbana-Champaign
    Author: Kim, Seok | University of Illinois at Urbana-Champaign
 
    keyword: Grippers and Other End-Effectors

    Abstract : Over the past few years, shape memory polymer (SMP) has been extensively studied in terms of its remarkable reversible dry adhesive properties and related smart adhesive applications. However, its exceptional properties have not been exploited for further opportunities such as pick-and-place applications, which would otherwise advance the robotic manipulation. This work explores the use of an SMP to design an adhesive gripper that picks and places a target solid object employing the reversible dry adhesion of an SMP. Compared with other single surface contact grippers including vacuum, electromagnetic, electroadhesion, and gecko grippers, the SMP adhesive gripper interacts with not only flat and smooth dry surfaces but also moderately rough and even wet surfaces for pick-and-place with high adhesion strength (&gt; 2 atmospheres). In this work, associated physical mechanisms, SMP adhesive mechanics, and thermal conditions are studied. In particular, the numerical and experimental study elucidates that the optimal compositional and topological SMP design may substantially enhance its adhesion strength and reversibility, which leads to a strong grip force simultaneously with a minimized releasing force. Finally, the versatility and utility of the SMP adhesive gripper are highlighted through diverse pick-and-place demonstrations.

- A High-Payload Proprioceptive Hybrid Robotic Gripper with Soft Origamic Actuators

    Author: Su, Yinyin | Southern University of Science and Technology
    Author: Fang, Zhonggui | Southern University of Science and Technology
    Author: Zhu, Wenpei | Southern University of Science and Technology
    Author: Sun, Xiaochen | Southern University of Science and Technology
    Author: Zhu, Yuming | SUSTech
    Author: Wang, Hexiang | Harbin Institute of Technology
    Author: Tang, Kailuan | Southern University of Science and Technology
    Author: Huang, Hailin | Harbin Institute of Technology, Shenzhen
    Author: Liu, Sicong | Southern University of Science and Technology
    Author: Wang, Zheng | The University of Hong Kong
 
    keyword: Grippers and Other End-Effectors; Soft Robot Materials and Design; Perception for Grasping and Manipulation

    Abstract : Proprioception is the ability to perceive environmental stimulations through internal sensory organs. Enabling proprioception is critical for robots to be aware of the environmental interactions and respond appropriately, particularly for high-payload grippers to ensure safety when handling delicate objects. State-of-the-art robotic grippers with soft actuators are typically equipped with pressure sensors for pneumatic regulation and control, but very few utilized them for proprioceptive purposes. This lack of environmental awareness was largely compensated by their inherent compliance and conformity, but also due to the generally limited force capabilities. Targeting at this gap, this work proposes a novel hybrid robotic gripper design with high-payload soft origami actuators and rigid supporting frames, achieving up to 567N actuator output force or 300N finger gripping force at 150kPa low pneumatic pressure and 3.2kg self-weight. Despite the substantially higher force capability over state-of-the-art soft grippers, the proposed hybrid gripper could retain the excellent inherent compliance thanks to the novel soft origami actuators being used. Moreover, by only using the embedded pneumatic pressure sensor, a novel scheme of multi-actuator proprioception is proposed to enable the hybrid gripper with environmental awareness, achieving real-time position and force estimations of errors at &lt;1% and 5.6%, respectively. The principles, design, prototyping, and experiments of the

## Formal Methods in Robotics and Automation
- Reality As a Simulation of Reality: Robot Illusions, Fundamental Limits, and a Physical Demonstration

    Author: Shell, Dylan | Texas A&amp;M University
    Author: O'Kane, Jason | University of South Carolina
 
    keyword: Formal Methods in Robotics and Automation; Simulation and Animation; Path Planning for Multiple Mobile Robots or Agents

    Abstract : We consider problems in which robots conspire to present a view of the world that differs from reality. The inquiry is motivated by the problem of validating robot behavior physically despite there being a discrepancy between the robots we have at hand and those we wish to study, or the environment for testing that is available versus that which is desired, or other potential mismatches in this vein. After formulating the concept of a convincing illusion, essentially a notion of system simulation that takes place in the real world, we examine the implications of this type of simulability in terms of infrastructure requirements. Time is one important resource: some robots may be able to simulate some others but, perhaps, only at a rate that is slower than real-time.	This difference gives a way of relating the simulating and the simulated systems in a form that is relative. We establish some theorems, including one with the flavor of an impossibility result, and providing several examples throughout. Finally, we present data from a simple multi-robot experiment based on this theory, with a robot navigating amid an unbounded field of obstacles.

- Finding Missing Skills for High-Level Behaviors

    Author: Pacheck, Adam | Cornell University
    Author: Moarref, Salar | Cornell University
    Author: Kress-Gazit, Hadas | Cornell University
 
    keyword: Formal Methods in Robotics and Automation; Task Planning

    Abstract : Recently, Linear Temporal Logic (LTL) has been used as a formalism for defining high-level robot tasks, and LTL synthesis has been used to automatically create correct-by-construction robot control. The underlying premise of this approach is that the robot has a set of actions, or skills, that can be composed to achieve the high-level task. In this paper we consider LTL specifications that cannot be synthesized into robot control due to lack of appropriate skills; we present algorithms for automatically suggesting new or modified skills for the robot that will guarantee the task will be achieved. We demonstrate our approach with a physical Baxter robot and a simulated KUKA IIWA arm.

- Near-Optimal Reactive Synthesis Incorporating RuntimeInformation

    Author: Bharadwaj, Sudarshanan | UT Austin
    Author: P. Vinod, Abraham | University of New Mexico
    Author: Dimitrova, Rayna | The University of Sheffield
    Author: Topcu, Ufuk | The University of Texas at Austin
 
    keyword: Formal Methods in Robotics and Automation; Autonomous Agents

    Abstract : We consider the problem of optimal reactive synthesis --- compute a strategy that satisfies a mission specification in a dynamic environment, and optimize a performance metric. We incorporate task-critical information, that is only available at runtime, into the strategy synthesis in order to improve performance. Existing approaches to utilize such time-varying information requires online re-synthesis, which is not computationally feasible in real-time applications. In this paper, we pre-synthesize a set of strategies corresponding to emph{candidate instantiations} (pre-specified representative information scenarios). We then propose a novel switching mechanism to dynamically switch between the strategies at runtime while guaranteeing all liveness goals are met. We also characterize bounds on the performance suboptimality. We demonstrate our approach in two examples --- robotic motion planning where the likelihood of the position of the robot's goal is updated in real-time, and an air traffic management problem for urban air mobility.

- Control Synthesis from Linear Temporal Logic Specifications Using Model-Free Reinforcement Learning

    Author: Bozkurt, Alper Kamil | Duke University
    Author: Wang, Yu | Duke University
    Author: Zavlanos, Michael M. | Duke University
    Author: Pajic, Miroslav | Duke University
 
    keyword: Formal Methods in Robotics and Automation; Motion and Path Planning; Probability and Statistical Methods

    Abstract : We present a reinforcement learning (RL) framework to synthesize a control policy from a given linear temporal logic (LTL) specification in an unknown stochastic environment that can be modeled as a Markov Decision Process (MDP). Specifically, we learn a policy that maximizes the probability of satisfying the LTL formula without learning the transition probabilities. We introduce a novel rewarding and discounting mechanism based on the LTL formula such that (i) an optimal policy maximizing the total discounted reward effectively maximizes the probabilities of satisfying LTL objectives, and (ii) a model-free RL algorithm using these rewards and discount factors is guaranteed to converge to such a policy. Finally, we illustrate the applicability of our RL-based synthesis approach on two motion planning case studies.

- A Framework for Formal Verification of Behavior Trees with Linear Temporal Logic

    Author: Biggar, Oliver | University of Melbourne
    Author: Zamani, Mohammad | DSTG
 
    keyword: Formal Methods in Robotics and Automation; Control Architectures and Programming

    Abstract : Despite the current increasing popularity of Behavior Trees (BTs) in the robotics community, there does not currently exist a method to formally verify their correctness without compromising their most valuable traits: modularity, flexibility and reusability. In this paper we present a new mathematical framework in which we formally express Behavior Trees in Linear Temporal Logic (LTL). We show how this framework equivalently represents classical BTs. Then we utilize the proposed framework to construct an algorithm to verify that a given BT satisfies a given LTL specification. We prove that this algorithm is sound. Importantly, we prove that this method does not compromise the flexible design process of BTs, i.e. changes to subtrees can be verified separately and their combination can be assured to be correct. We present an example of the proposed algorithm in use.

- Safety Assessment of Collaborative Robotics through Automated Formal Verification (I)

    Author: Vicentini, Federico | Stiima CNR
    Author: Askarpour, Mehrnoosh | Politecnico Di Milano
    Author: Rossi, Matteo Giovanni | Politecnico Di Milano
    Author: Dino, Mandrioli | Politecnico Di Milano
 
    keyword: Formal Methods in Robotics and Automation; Physical Human-Robot Interaction

    Abstract : A crucial aspect of physical human�robot collaboration (HRC) is to maintain a safe common workspace for human operator. However, close proximity between human�robot and unpredictability of human behavior raises serious challenges in terms of safety. This article proposes a risk analysis methodology for collaborative robotic applications, which is compatible with well-known standards in the area and relies on formal verification techniques to automate the traditional risk analysis methods. In particular, the methodology relies on temporal logic-based mod- els to describe the different possible ways in which tasks can be carried out, and on fully automated formal verification techniques to explore the corresponding state space to detect and modify the hazardous situations at early stages of system design.

## Parallel Robots
- R-Min: A Fast Collaborative Underactuated Parallel Robot for Pick-And-Place Operations

    Author: Jeanneau, Guillaume | École Centrale De Nantes
    Author: B�goc, Vincent | Icam
    Author: Briot, S�bastien | LS2N
    Author: Goldsztejn, Alexandre | CNRS IRCCyN
 
    keyword: Parallel Robots; Mechanism Design; Physical Human-Robot Interaction

    Abstract : This paper introduces an intrinsically safe parallel manipulator dedicated to fast pick-and-place operations, called R-Min. It has been designed to reduce the risk of injury during a collision with a human operator, while maintaining high speed and acceleration. The proposed architecture is based on a modification of the well-known planar five-bar mechanism, where additional passive joints are introduced to the distal links in order to create a planar seven-bar mechanism with two degrees of underactuation, so that it can passively reconfigure in case of collision. A supplementary passive leg, in which a tension spring is mounted, is added between the base and the end-effector in order to constrain the additional degrees of freedom.<p>A prototype of this new collaborative parallel robot is designed and its equilibrium configurations under several types of loadings are analyzed. Its dynamics is also studied. We analyze the impact force occurring during a collision between our prototype and the head of an operator and compare these results with those that would have been obtained with a rigid five-bar mechanism. Simulation results of impact during a standard pick-and-place trajectory of duration 0.3~s show that a regular five-bar mechanism would injure a human, while our robot would avoid the trauma.

- High-Flexibility Locomotion and Whole-Torso Control for a Wheel-Legged Robot on Challenging Terrain

    Author: Xu, Kang | Beijing Institute of Technology
    Author: Wang, Shoukun | Beijing Institute of Technology
    Author: Wang, Xiuwen | Beijing Institute of Technology
    Author: Wang, Junzheng | Beijing Institute of Technology
    Author: Chen, Zhihua | Beijing Institute of Technology
    Author: Liu, Daohe | Beijing Institute of Technology
 
    keyword: Parallel Robots; Wheeled Robots; Legged Robots

    Abstract : In this paper, we propose a parallel six-wheeled-legged robot that can traverse irregular terrain while carrying objectives to do heavy-duty work. This robot is equipped with six Stewart platforms as legs and tightly integrates the additional degrees of freedom introduced by the wheels. The presented control strategy with physical system used to adapt the diverse degrees of each leg to irregular terrain such that robot increases the traversability, and simultaneously to maintain the horizontal whole-torso pose. This strategy makes use of Contact Scheduler (CS) and Whole-Torso Control (WTC) to control the multiple degrees of freedom (DOF) leg for performing high-flexibility locomotion and adapting the rough terrain like actively parallel suspension system. We conducted experiments on flat, slope, soft and sand-gravel surface, which validate the proposed control method and physical system. Especially, we attempt to traverse over sand-gravel terrain with 3 people about 240kg payload.

- Prince's Tears, a Large Cable-Driven Parallel Robot for an Artistic Exhibition

    Author: Merlet, Jean-Pierre | INRIA
    Author: Papegay, Yves | INRIA
    Author: Gasc, Anne-Valerie | ENSAM-M
 
    keyword: Parallel Robots; Kinematics; Robotics in Construction

    Abstract : This paper presents the development and results of a large 3 d.o.f cable-driven parallel robot (CDPR) that has been extensively used between June and August 2019 for an artistic exhibition. The purpose of the exhibition was to 3D print a wall of glass powder, which will slowly collapse after the deposit of each layer. Positioning control on the assigned trajectory was an issue because of the CDPR geometry imposed by the specific configuration of the exhibition place. We describe how this problem was solved using a combination of cable length estimation based on the winch rotation measured by encoder, together with 3 on-board lidars that were used to provide a measure of the robot position. To the best of our knowledge this is the first time that such method was used for controlling a large CDPR. This CDPR has run for 174 hours since 6/18/2019, averaging a run time of 4h15mn per day. The 3D printing of the wall started on 7/18/2019 and stops on 8/31/2019. During this period the robot was used for 32 days with an average of 2h18mn run-time per day. The robot has traveled on a total distance of 4757 meters, of which 3893 meters on the assigned trajectory. During the period 76 layers have been deposited, representing a mass of 1.5 tons of glass powder.

- Singularity Analysis and Reconfiguration Mode of the 3-CRS Parallel Manipulator

    Author: Bouzgarrou, Chedli | Institut Pascal UMR 6602 - UCA/CNRS/SIGMA
    Author: Koessler, Adrien | Institut Pascal
    Author: Bouton, Nicolas | Institut Pascal - SIGMA Clermont-Ferrand
 
    keyword: Parallel Robots; Kinematics; Mechanism Design

    Abstract : The 3-CRS manipulator is an original parallel mechanism having 6 degrees of freedom (DOFs) with only 3 limbs. This mechanism uses a motorized cylindrical joint per limb. This new paradigm of actuation opens research fields on new families of robots that should particularly interest the parallel robotics community. According to its dimensional synthesis, this mechanism can have remarkable kinematic properties such as a large orientation workspace or reconfiguration capabilities. In this paper, we introduce this mechanism and we study its singularities by using a geometric approach. This approach simplifies considerably singularity analysis problem by considering the relative geometric configurations of three planes defined by the distal links of the limbs. Thanks to that, a reconfiguration mode of the 3-CRS, that doubles its reachable workspace, is highlighted. This property is illustrated on a physical prototype of the robot.

- Trajectory Optimization for a Class of Robots Belonging to Constrained Collaborative Mobile Agents (CCMA) Family

    Author: Kumar, Nitish | Computational Robotics Lab, Institute for Intelligent Interactiv
    Author: Coros, Stelian | Carnegie Mellon University
 
    keyword: Parallel Robots; Optimization and Optimal Control; Multi-Robot Systems

    Abstract : We present a novel class of robots belonging to Constrained Collaborative Mobile Agents (CCMA) family which consists of ground mobile bases with non-holonomic constraints. Moreover, these mobile robots are constrained by closed-loop kinematic chains consisting of revolute joints which can be either passive or actuated. We also describe a novel trajectory optimization method which is general with respect to number of mobile robots, topology of the closed-loop kinematic chains and placement of the actuators at the revolute joints. We also extend the standalone trajectory optimization method to optimize concurrently the design parameters and the control policy. We describe various CCMA system examples, in simulation, differing in design, topology, number of mobile robots and actuation space. The simulation results for standalone trajectory optimization with fixed design parameters is presented for CCMA system examples. We also show how this method can be used for tasks other than end-effector positioning such as internal collision avoidance and external obstacle avoidance. The concurrent design and control policy optimization is demonstrated, in simulations, to increase the CCMA system workspace and manipulation capabilities. Finally, the trajectory optimization method is validated in experiments through two 4-DOF prototypes consisting of 3 tracked mobile bases.

-  Multiaxis Reaction System (MARS) for Vibration Control of Planar Cable-Driven Parallel Robots (I)

    Author: Rushton, Mitchell | University of Waterloo
    Author: Jamshidifar, Hamed | University of Waterloo
    Author: Khajepour, Amir | University of Waterloo


## Mechanism and Verification
- Reconfiguration Solution of a Variable Topology Truss: Design and Experiment

    Author: Park, Eugene | Seoul National University
    Author: Bae, Jangho | Seoul National University
    Author: Park, Sumin | Seoul National University
    Author: Yim, Mark | University of Pennsylvania
    Author: Kim, Jongwon | Seoul National University
    Author: Seo, TaeWon | Hanyang University
 
    keyword: Cellular and Modular Robots; Mechanism Design

    Abstract : In this paper, an active ball joint actuator, called the master node, is developed for the purpose of reconfiguring truss structures. We propose a variable topology truss system, which is an advanced variable geometry truss system that can reconfigure its own topology to expand its functions. However, reconfiguration of a variable topology truss is difficult, because the controllability of trusses needs to be maintained during the process. We solve this problem by adding the master node to the system, which can move trusses without losing their controllability. The master node is designed and fabricated for a variable topology truss. The reconfiguration test using the master node is performed on a reduced prototype of the system. The results prove that using the master node for reconfiguration is viable.

- Development of Body Rotational Wheeled Robot and Its Verification of Effectiveness

    Author: Sim, Byeong-Seop | Chonbuk National University
    Author: Kim, Kun-Jung | Chonbuk National University
    Author: Yu, Kee-Ho | Chonbuk National University
 
    keyword: Field Robots; Mechanism Design; Wheeled Robots

    Abstract : A wheeled robot operating on various terrains such as scattered obstacles and slopes is required to cope with and overcome the driving environment. In this paper, in order to overcome a step-type obstacle and to steadily ascend on the slope, the main body rotation mechanism, which controls the load distribution on the robot wheels was proposed for a wheel-drive robot. By rotating the center of the body mass, the friction/traction force required for climbing step obstacles can be reduced. In the case of slope traveling, the slip can be suppressed, and the traveling ability improved by controlling the load distribution excessively increased on the downhill wheel due to the attitude change of the robot's body. The mechanical effect of the proposed body rotation mechanism was analyzed. In addition, based on the design and manufacture of the robot platform, the effectiveness of the proposed mechanism was convincingly demonstrated by indoor test for step-obstacle climbing and slope-traveling.

- Error Bounds for PD-Controlled Mechanical Systems under Bounded Disturbances Using Interval Arithmetic

    Author: Calzolari, Davide | German Aerospace Center, Technical University of Munich
    Author: Giordano, Alessandro Massimo | DLR (German Aerospace Center)
    Author: Albu-Sch�ffer, Alin | DLR - German Aerospace Center
 
    keyword: Performance Evaluation and Benchmarking; Industrial Robots

    Abstract : We present a numerical algorithm based on invariant set theory to evaluate the worst-case performance of PD-controlled mechanical systems affected by bounded disturbances. By performing a specific coordinate transformation, the search and computation of positive invariant sets is simplified. It is shown that, thanks to the preservation of problem structure, the proposed method allows to obtain tight, component-wise bounds on the states, which are especially useful for performance evaluation and tuning of a PD controller. The bounds are formally guaranteed and can be used for safety certification. The method is compared to ultimate boundedness, and the superior results are shown via numerical simulations.

- Hardware-In-The-Loop Iterative Optimal Feedback Control without Model-Based Future Prediction (I)

    Author: Chen, Yuqing | Singapore University of Technology and Design
    Author: Braun, David | Vanderbilt University
 
    keyword: Optimization and Optimal Control; Control Architectures and Programming

    Abstract : Optimal control provides a systematic approach to control robots. However, computing optimal controllers for hardware-in-the-loop control is sensitively affected by modeling assumptions, computationally expensive in online implementation, and time-consuming in practical application. This makes the theoretical appeal of optimization challenging to exploit in real-world implementation. In this paper, we present a novel online optimal control formulation that aims to address the above-mentioned limitations. The formulation combines a model with measured state information to efficiently find near-optimal feedback controllers. The idea to combine a model with measurements from the actual motion is similar to what is used in model predictive control formulations, with the difference that here the model is not used for future prediction, the optimization is performed along the measured trajectory of the system, and the online computation is reduced to a minimum; it requires a small-scale, one time step, static optimization, instead of a large-scale, finite time horizon, dynamic optimization. The formulation can be used to solve optimal control problems defined with nonlinear cost, nonlinear dynamics, and box-constrained control inputs. Numerical simulations and hardware-in-the-loop experiments demonstrate the effectiveness of the proposed hardware-in-the-loop optimal control approach.

- Analysis and Synthesis of Underactuated Compliant Mechanisms Based on Transmission Properties of Motion and Force (I)
 
    Author: Chen, Wenrui | Hunan University
    Author: Xiong, Caihua | Huazhong Univ. of Science &amp; Tech
    Author: Wang, Yaonan | Hunan University
 
    keyword: Underactuated Robots; Compliant Joint/Mechanism; Grasping

    Abstract : This paper analyzes and designs the transmission structure for underactuated compliant mechanisms (UCMs). The transmission structure of UCMs consists of serial and parallel transmission chains. At first, the UCMs are classified systematically according to the number and distribution of the serial and parallel transmissions. Next, the active and passive transmission properties of motion and force in UCMs are analyzed on the defined four subspaces of tangent and cotangent spaces of joint space. Synthesizing the classification and the transmission properties of UCMs, the congruent relationship between mechanical structure and transmission function is established, and different cases of UCMs are discussed and compared. A novel type of UCMs can achieve the independent regulation of passive stiffness, active force and active motion that is useful for improving the transmission performance in robotic and prosthetic hands. Finally, a functional oriented design method is proposed and used to design a single-actuator two-fingered gripper for enveloping and precision grasps. The results demonstrate the validity of the proposed method.

- Radar Sensors in Collaborative Robotics: Fast Simulation and Experimental Validation

    Author: Stetco, Christian | Alpen-Adria Universitét Klagenfurt
    Author: Ubezio, Barnaba | Joanneum Research Robotics Forschungsgesellschaft MbH
    Author: M�hlbacher-Karrer, Stephan | JOANNEUM RESEARCH Forschungsgesellschaft mbH - ROBOTICS
    Author: Zangl, Hubert | Alpen-Adria-Universitaet Klagenfurt
 
    keyword: Range Sensing; Simulation and Animation; Human Detection and Tracking

    Abstract : With the availability of small system in package realizations, radar systems become more and more attractive for a variety of applications in robotics, in particular also for collaborative robotics. As the simulation of robot systems in realistic scenarios has become an important tool, not only for design and optimization, but also e.g. for machine learning approaches, realistic simulation models are needed. In the case of radar sensor simulations, this means providing more realistic results than simple proximity sensors, e.g. in the presence of multiple objects and/or humans, objects with different relative velocities and differentiation between background and foreground movement. Due to the short wavelength in the millimeter range, we propose to utilize methods known from computer graphics (e.g. z-buffer approach, Lambertian reflectance model) to quickly acquire depth images and reflection estimates. This information is used to calculate an estimate of the received signal for a Frequency Modulated Continuous Wave (FMCW) radar by superposition of the corresponding signal contributions. Due to the moderate computational complexity, the approach can be used with various simulation environments such as V-Rep or Gazebo. Validity and benefits of the approach are demonstrated by means of a comparison with experimental data obtained with a radar sensor on a UR10 arm in different scenarios.

## Model Learning for Control
-  Sparse, Online, Locally Adaptive Regression Using Gaussian Processes for Bayesian Robot Model Learning and Control

    Author: Wilcox, Brian | UC San Diego
    Author: Yip, Michael C. | University of California, San Diego

- DISCO: Double Likelihood-Free Inference Stochastic Control

    Author: Barcelos, Lucas | The University of Sydney
    Author: Oliveira, Rafael | University of Sydney
    Author: Possas, Rafael | University of Sydney
    Author: Ott, Lionel | University of Sydney
    Author: Ramos, Fabio | University of Sydney, NVIDIA
 
    keyword: Model Learning for Control; Learning and Adaptive Systems; Robust/Adaptive Control of Robotic Systems

    Abstract : Accurate simulation of complex physical systems enables the development, testing, and certification of control strategies before they are deployed into the real systems. As simulators become more advanced, the analytical tractability of the differential equations and associated numerical solvers incorporated in the simulations diminishes, making them difficult to analyse. A potential solution is the use of probabilistic inference to assess the uncertainty of the simulation parameters given real observations of the system. Unfortunately the likelihood function required for inference is generally expensive to compute or totally intractable. In this paper we propose to leverage the power of modern simulators and recent techniques in Bayesian statistics for likelihood-free inference to design a control framework that is efficient and robust with respect to the uncertainty over simulation parameters. The posterior distribution over simulation parameters is propagated through a potentially non-analytical model of the system with the unscented transform, and a variant of the information theoretical model predictive control. This approach provides a more efficient way to evaluate trajectory roll outs than Monte Carlo sampling, reducing the online computation burden. Experiments show that the controller proposed attained superior performance and robustness on classical control and robotics tasks when compared to models not accounting for the uncertainty over model parameters.

- Discovering Interpretable Dynamics by Sparsity Promotion on Energy and the Lagrangian

    Author: Chu, Khanh Hoang | Tohoku University
    Author: Hayashibe, Mitsuhiro | Tohoku University
 
    keyword: Model Learning for Control; Dynamics; Calibration and Identification

    Abstract : Data-driven modeling frameworks that adopt sparse regression techniques, such as sparse identification of nonlinear dynamics (SINDy) and its modifications, are developed to resolve difficulties in extracting underlying dynamics from experimental data. In contrast to neural-network-based methods, these methods are designed to obtain white-box analytical models. In this work, we incorporate the concept of SINDy and knowledge in the field of classical mechanics to identify interpretable and sparse expressions of total energy and the Lagrangian that shelters the hidden dynamics. Moreover, our method (hereafter referred as Lagrangian-SINDy) is developed to use knowledge of simple systems that form the system being analyzed to ensure the likelihood of correct results and to improve the learning pace. Lagrangian-SINDy is highly accurate in discovering interpretable dynamics via energy-related physical quantities. Its performance is validated with three popular multi-DOF nonlinear dynamical systems, namely the spherical pendulum, double pendulum and cart-pendulum system. Comparisons with other SINDy-based methods are made and Lagrangian-SINDy is found to provide the most compact analytical models.

- Online Simultaneous Semi-Parametric Dynamics Model Learning

    Author: Smith, Joshua | University of Edinburgh
    Author: Mistry, Michael | University of Edinburgh
 
    keyword: Model Learning for Control; Robust/Adaptive Control of Robotic Systems; Dynamics

    Abstract : Accurate models of robots' dynamics are critical for control, stability, motion optimization, and interaction. Semi-Parametric approaches to dynamics learning combine physics-based Parametric models with unstructured Non-Parametric regression with the hope to achieve both accuracy and generalizability. In this paper, we highlight the non-stationary problem created when attempting to adapt both Parametric and Non-Parametric components simultaneously. We present a consistency transform designed to compensate for this non-stationary effect, such that the contributions of both models can adapt simultaneously without adversely affecting the performance of the platform. Thus, we are able to apply the Semi-Parametric learning approach for continuous iterative online adaptation, without relying on batch or offline updates. We validate the transform via a perfect virtual model as well as by applying the overall system on a Kuka LWR IV manipulator. We demonstrate improved tracking performance during online learning and show a clear transference of contribution between the two components with a learning bias towards the Parametric component.

- Sufficiently Accurate Model Learning

    Author: Zhang, Clark | University of Pennsylvania
    Author: Khan, Arbaaz | University of Pennsylvania
    Author: Paternain, Santiago | University of Pennsylvania
    Author: Ribeiro, Alejandro | University of Pennsylvania
 
    keyword: Model Learning for Control; Robust/Adaptive Control of Robotic Systems; Motion Control

    Abstract : Modeling how a robot interacts with the environment around it is an important prerequisite for designing control and planning algorithms. In fact, the performance of controllers and planners is highly dependent on the quality of the model. One popular approach is to learn data driven models in order to compensate for inaccurate physical measurements and to adapt to systems that evolve over time. In this paper, we investigate a method to regularize model learning techniques to provide better error characteristics for traditional control and planning algorithms. This work proposes learning ``Sufficiently Accurate" models of dynamics using a primal-dual method that can explicitly enforce constraints on the error in pre-defined parts of the state-space. The result of this method is that the error characteristics of the learned model is more predictable and can be better utilized by planning and control algorithms. The characteristics of Sufficiently Accurate models are analyzed through experiments on a simulated ball paddle system.

- Active Learning of Dynamics for Data-Driven Control Using Koopman Operators (I)

    Author: Abraham, Ian | Northwestern University
    Author: Murphey, Todd | Northwestern University
 
    keyword: Model Learning for Control; Optimization and Optimal Control

    Abstract : This paper presents an active learning strategy for robotic systems that takes into account task information, enables fast learning, and allows control to be readily synthesized by taking advantage of the Koopman operator representation. We first motivate the use of representing nonlinear systems as linear Koopman operator systems by illustrating the improved model-based control performance with an actuated Van der Pol system. Information-theoretic methods are then applied to the Koopman operator formulation of dynamical systems where we derive a controller for active learning of robot dynamics. The active learning controller is shown to increase the rate of information about the Koopman operator. In addition, our active learning controller can readily incorporate policies built on the Koopman dynamics, enabling the benefits of fast active learning and improved control. Results using a quadcopter illustrate single-execution active learning and stabilization capabilities during free-fall. The results for active learning are extended for automating Koopman observables and we implement our method on real robotic systems.

## Mobile Manipulation
- Towards Plan Transformations for Real-World Mobile Fetch and Place

    Author: Kazhoyan, Gayane | University of Bremen
    Author: Niedzwiecki, Arthur | Institute for Artificial Intelligence, University of Bremen
    Author: Beetz, Michael | University of Bremen
 
    keyword: Mobile Manipulation; Autonomous Agents; Service Robots

    Abstract : In this paper, we present an approach and an implemented framework for applying plan transformations to real-world mobile manipulation plans, in order to specialize them to the specific situation at hand. The framework can improve execution cost and achieve better performance by autonomously transforming robot's behavior at runtime. To demonstrate the feasibility of our approach, we apply three example transformations to the plan of a PR2 robot performing simple table setting and cleaning tasks in the real world. Based on a large amount of experiments in a fast plan projection simulator, we make conclusions on improved execution performance.

- Planning an Efficient and Robust Base Sequence for a Mobile Manipulator Performing Multiple Pick-And-Place Tasks

    Author: Xu, Jingren | Osaka University
    Author: Harada, Kensuke | Osaka University
    Author: Wan, Weiwei | Osaka University
    Author: Ueshiba, Toshio | National Institute of Advanced Industrial Science And
    Author: Domae, Yukiyasu | The National Institute of Advanced Industrial Science and Techno
 
    keyword: Mobile Manipulation

    Abstract : In this paper, we address efficiently and robustly collecting objects stored in different trays using a mobile manipulator. A resolution complete method, based on precomputed reachability database, is proposed to explore collision-free inverse kinematics (IK) solutions and then a resolution complete set of feasible base positions can be determined. This method approximates a set of representative IK solutions that are especially helpful when solving IK and checking collision are treated separately. For real world applications, we take into account the base positioning uncertainty and plan a sequence of base positions that reduce the number of necessary base movements for collecting the target objects, the base sequence is robust in that the mobile manipulator is able to complete the part-supply task even there is certain deviation from the planned base positions. Our experiments demonstrate both the efficiency compared to regular base sequence and the feasibility in real world applications.

- Towards Mobile Multi-Task Manipulation in a Confined and Integrated Environment with Irregular Objects

    Author: Han, Zhao | UMass Lowell
    Author: Allspaw, Jordan | University of Massachusetts Lowell
    Author: LeMasurier, Gregory | University of Massachusetts Lowell
    Author: Parrillo, Jenna | University of Massachusetts Lowell
    Author: Giger, Daniel | University of Massachusetts Lowell
    Author: Ahmadzadeh, S. Reza | University of Massachusetts Lowell
    Author: Yanco, Holly | UMass Lowell
 
    keyword: Mobile Manipulation

    Abstract : The FetchIt! Mobile Manipulation Challenge, held at the IEEE International Conference on Robots and Automation (ICRA) in May 2019, offered an environment with complex and integrated task sets, irregular objects, confined space, and machining, introducing new challenges in the mobile manipulation domain. Here we describe our efforts to address these challenges by demonstrating the assembly of a kit of mechanical parts in a caddy. In addition to implementation details, we examine the issues in this task set extensively, and we discuss our software architecture in the hope of providing a base for other researchers. To evaluate performance and consistency, we conducted 20 full runs, then examined failure cases with possible solutions. We conclude by identifying future research directions to address the open challenges.

- Linear Time-Varying MPC for Nonprehensile Object Manipulation with a Nonholonomic Mobile Robot

    Author: Bertoncelli, Filippo | University of Modena and Reggio Emilia
    Author: Ruggiero, Fabio | Université Di Napoli Federico II
    Author: Sabattini, Lorenzo | University of Modena and Reggio Emilia
 
    keyword: Mobile Manipulation; Contact Modeling; Optimization and Optimal Control

    Abstract : This paper proposes a technique to manipulate an object with a nonholonomic mobile robot by pushing, which is a nonprehensile manipulation motion primitive. Such a primitive involves unilateral constraints associated with the friction between the robot and the manipulated object. Violating this constraint produces the slippage of the object during the manipulation, preventing the correct achievement of the task. A linear time-varying model predictive control is designed to include the unilateral constraint within the control action properly. The approach is verified in a dynamic simulation environment through a Pioneer 3-DX wheeled robot executing the pushing manipulation of a package.

- A Mobile Manipulation System for One-Shot Teaching of Complex Tasks in Homes

    Author: Bajracharya, Max | Toyota Research Institute
    Author: Borders, James | Toyota Research Institute
    Author: Helmick, Daniel | Toyota Research Institute
    Author: Kollar, Thomas | Toyota Research Institute
    Author: Laskey, Michael | University of California, Berkeley
    Author: Leichty, John | Toyota Research Institute
    Author: Ma, Jeremy | California Institute of Technology
    Author: Nagarajan, Umashankar | Honda Research Institute USA Inc
    Author: Ochiai, Akiyoshi | Toyota Research Institute
    Author: Petersen, Joshua | Toyota Research Institute
    Author: Shankar, Krishna | Toyota Research Institute
    Author: Stone, Kevin | Toyota Research Institute
    Author: Takaoka, Yutaka | Toyota Mortor Corporation
 
    keyword: Mobile Manipulation; Learning from Demonstration; Domestic Robots

    Abstract : We describe a mobile manipulation hardware and software system capable of autonomously performing complex human-level tasks in real homes, after being taught the task with a single demonstration from a person in virtual reality. This is enabled by a highly capable mobile manipulation robot, whole-body task space hybrid position/force control, teaching of parameterized primitives linked to a robust learned dense visual embeddings representation of the scene, and a task graph of the taught behaviors. We demonstrate the robustness of the approach by presenting results for performing a variety of tasks, under different environmental conditions, in multiple real homes. Our approach achieves 85% overall success rate on three tasks that consist of an average of 45 behaviors each. The video is available at: https://youtu.be/HSyAGMGikLk.

## Computer Vision for Transportation
- 2D to 3D Line-Based Registration with Unknown Associations Via Mixed-Integer Programming

    Author: Parkison, Steven | University of Michigan
    Author: Walls, Jeffrey | University of Michigan
    Author: Wolcott, Ryan | University of Michigan
    Author: Saad, Mohammad | Toyota Research Institute
    Author: Eustice, Ryan | University of Michigan
 
    keyword: Computer Vision for Transportation; Localization; Mapping

    Abstract : Determining the rigid-body transformation between 2D image data and 3D point cloud data has applications for mobile robotics including sensor calibration and localizing into a prior map. Common approaches to 2D-3D registration use least-squares solvers assuming known associations often provided by heuristic front-ends, or iterative nearest-neighbor. We present a linear line-based 2D-3D registration algorithm formulated as a mixed-integer program to simultaneously solve for the correct transformation and data association. Our formulation is explicitly formulated to handle outliers, by modeling associations as integer variables. Additionally, we can constrain the registration to SE(2) to improve runtime and accuracy. We evaluate this search over multiple real-world data sets demonstrating adaptability to scene variation

- An Efficient Solution to the Relative Pose Estimation with a Common Direction

    Author: Ding, Yaqing | Nanjing University of Science and Technology
    Author: Yang, Jian | Nanjing University of Science &amp; Technology
    Author: Kong, Hui | Nanjing University of Science and Technology
 
    keyword: Computer Vision for Transportation; Sensor Fusion; Performance Evaluation and Benchmarking

    Abstract : In this paper, we propose an efficient solution to the calibrated camera motion estimation with a common direction. This case is relevant to smart phones, tablets, and other camera-IMU (Inertial measurement unit) systems, which have accelerometers to measure the gravity direction. We can align one of the axes of the camera with this common direction so that the relative rotation between the views reduces to only 1-DOF (degree of freedom). This allows us to use only three point correspondences for relative pose estimation. Unlike previous work, we derive new constraints on the simplified essential matrix using an elimination strategy based on Gr"{o}bner basis. In this case, computing the coefficients of these constraints require less computation and we only need to solve a polynomial eigenvalue problem. We show detailed analyses and comparisons against the existing 3-point algorithms, with satisfactory results obtained.

- Task-Aware Novelty Detection for Visual-Based Deep Learning in Autonomous Systems

    Author: Chen, Valerie | Yale University
    Author: Yoon, Man-Ki | Yale University
    Author: Shao, Zhong | Yale University
 
    keyword: Computer Vision for Transportation; Deep Learning in Robotics and Automation; Autonomous Vehicle Navigation

    Abstract : Deep-learning driven safety-critical autonomous systems, such as self-driving cars, must be able to detect situations where its trained model is not able to make a trustworthy prediction. This ability to determine the novelty of a new input with respect to a trained model is critical for such systems because novel inputs due to changes in the environment, adversarial attacks, or even unintentional noise can potentially lead to erroneous, perhaps life-threatening decisions. This paper proposes a learning framework that leverages information learned by the prediction model in a task-aware manner to detect novel scenarios. We use network saliency to provide the learning architecture with knowledge of the input areas that are most relevant to the decision-making and learn an association between the saliency map and the predicted output to determine the novelty of the input. We demonstrate the efficacy of this method through experiments on real-world driving datasets as well as through driving scenarios in our in-house indoor driving environment where the novel image can be sampled from another similar driving dataset with similar features or from adversarial attacked images from the training dataset. We find that our method is able to systematically detect novel inputs and quantify the deviation from the target prediction through this task-aware approach.

- DirectShape: Direct Photometric Alignment of Shape Priors for Visual Vehicle Pose and Shape Estimation

    Author: Wang, Rui | Technical University of Munich
    Author: Yang, Nan | Technical University of Munich
    Author: Stueckler, Joerg | Max-Planck Institute for Intelligent Systems
    Author: Cremers, Daniel | Technical University of Munich
 
    keyword: Computer Vision for Transportation; Semantic Scene Understanding; Autonomous Vehicle Navigation

    Abstract : Scene understanding from images is a challenging problem which is encountered in autonomous driving. On the object level, while 2D methods have gradually evolved from computing simple bounding boxes to delivering finer grained results like instance segmentations, the 3D family is still dominated by estimating 3D bounding boxes. In this paper, we propose a novel approach to jointly infer the 3D rigid-body poses and shapes of vehicles from a stereo image pair using shape priors. Unlike previous works that geometrically align shapes to point clouds from dense stereo reconstruction, our approach works directly on images by combining a photometric and a silhouette alignment term in the energy function. An adaptive sparse point selection scheme is proposed to efficiently measure the consistency with both terms. In experiments, we show superior performance of our method on 3D pose and shape estimation over the previous geometric approach. Moreover, we demonstrate that our method can also be applied as a refinement step and significantly boost the performances of several state-of-the-art deep learning based 3D object detectors. All related materials and a demonstration video are available at the project page https://vision.in.tum.de/research/vslam/direct-shape.

- RoadText-1K: Text Detection &amp; Recognition Dataset for Driving Videos

    Author: Battu, Sangeeth Reddy | IIIT Hyderabad
    Author: Mathew, Minesh | International Institute of Information Technology, Hyderabad
    Author: Gomez, Lluis | Computer Vision Center, Universitat Autonoma De Barcelona
    Author: Rusi�ol, Mar�al | Computer Vision Center
    Author: Karatzas, Dimosthenis | Computer Vision Center, Universitat Aut�noma De Barcelona
    Author: Jawahar, C.V. | IIIT, Hyderabad
 
    keyword: Computer Vision for Transportation; Intelligent Transportation Systems; Semantic Scene Understanding

    Abstract : Understanding text is crucial to understand the semantics of outdoor scenes and hence is a critical requirement to build intelligent systems for driver assistance and self- driving. Most of the existing datasets for text detection and recognition comprise still images and are mostly compiled keeping text in mind. This paper introduces a new �RoadText- 1K - dataset for text in driving videos. The dataset is 20 times larger than the existing largest dataset for text in videos. Our dataset comprises 1000 video clips of driving without any bias towards the text and with annotations for text bounding boxes and transcriptions in every frame. State of the art methods for text detection, recognition and tracking are evaluated on the new dataset and the results signify the challenges in unconstrained driving videos compared to existing datasets. This suggests that RoadText-1K is suited for research and development of reading systems robust enough to be incorporated into more complex downstream tasks like driver assistance and self-driving.

- End-To-End Learning for Inter-Vehicle Distance and Relative Velocity Estimation in ADAS with a Monocular Camera

    Author: Song, Zhenbo | Nanjing University of Science and Technology
    Author: Lu, Jianfeng | Nanjing University of Science &amp; Technology
    Author: Zhang, Tong | Australian National University, Motovis Australia Pty Ltd
    Author: Li, Hongdong | Australian National University and NICTA
 
    keyword: Computer Vision for Transportation; Deep Learning in Robotics and Automation; Intelligent Transportation Systems

    Abstract : Inter-vehicle distance and relative velocity estimations are two basic functions for any ADAS (Advanced driver-assistance systems). In this paper, we propose a monocular camera based inter-vehicle distance and relative velocity estimation method based on end-to-end training of a deep neural network. The key novelty of our method is the integration of multiple visual cues provided by any two time-consecutive monocular frames, which include deep feature cue, scene geometry cue, as well as temporal optical flow clue. We also propose a vehicle-centric sampling mechanism to alleviate the effect of perspective distortion in the motion field (i.e. optical flow). We implement the method by a light-weight deep neural network. Extensive experiments are conducted which confirm the superior performance of our method over other state-of-the-art methods, in terms of estimation accuracy, computational speed, and memory footprint.

## Haptics and Haptic Interfaces
- Learning an Action-Conditional Model for Haptic Texture Generation

    Author: Heravi, Negin | Stanford
    Author: Yuan, Wenzhen | Carnegie Mellon University
    Author: Okamura, Allison M. | Stanford University
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Haptics and Haptic Interfaces; Visual Learning

    Abstract : Rich haptic sensory feedback in response to user interactions is desirable for an effective, immersive virtual reality or teleoperation system. However, this feedback depends on material properties and user interactions in a complex, non-linear manner. Therefore, it is challenging to model the mapping from material and user interactions to haptic feedback in a way that generalizes over many variations of the user's input. Current methodologies are typically conditioned on user interactions, but require a separate model for each material. In this paper, we present a learned action-conditional model that uses data from a vision-based tactile sensor (GelSight) and user's action as input. This model predicts an induced acceleration that could be used to provide haptic vibration feedback to a user. We trained our proposed model on a publicly available dataset (Penn Haptic Texture Toolkit) that we augmented with GelSight measurements of the different materials. We show that a unified model over all materials outperforms previous methods and generalizes to new actions and new instances of the material categories in the dataset.

- Just Noticeable Differences for Joint Torque Feedback During Static Poses

    Author: Kim, Hubert | Virginia Tech
    Author: Guo, Hongxu | Virginia Polytechnic Institute and State University
    Author: Asbeck, Alan | Virginia Tech
 
    keyword: Haptics and Haptic Interfaces; Physical Human-Robot Interaction; Wearable Robots

    Abstract : Joint torque feedback is a new and promising means of kinesthetic feedback for providing information to a person or guiding them during a motion task. However, little work has been done in determining the psychophysical parameters of how well humans can detect external torques. In this study, we determine the human perceptual ability to detect kinesthetic feedback at the elbow during all possible combinations of preload torques and test stimulus torques, with the elbow in a static posture. To accomplish this, we constructed an exoskeleton for the elbow providing joint torque feedback. The device is designed to convey 0.54 Nm of stall torque for up to 120 seconds via a semi-rigid sleeve structure. Using this device, we assessed perception capability using the Interweaving Staircase Method. We found that users could detect average torques of 0.14-0.18 Nm in the extension or flexion directions with no preload. When a preload of 1.27 Nm was applied, this increased to 0.25-0.27 Nm for when flexion stimuli were applied, and 0.18-0.3 Nm when extension stimuli were applied, depending on the preload direction.

- Design of a Parallel Haptic Device with Gravity Compensation by Using Its System Weight

    Author: Hur, Sung-moon | Korea Institute of Science &amp; Technology (KIST)
    Author: Park, Jaeyoung | Korea Institute of Science and Technology
    Author: Park, Jaeheung | Seoul National University
    Author: Oh, Yonghwan | Korea Institute of Science &amp; Technology (KIST)
 
    keyword: Haptics and Haptic Interfaces; Mechanism Design

    Abstract : This paper proposes a 6 degree of freedom(DoF) manipulator for haptic application. The proposed haptic device, named GHap, is designed based on the four-bar-linkage mechanism for linear motion with the ring-type gimbal mechanism. To improve the force display ability, the device is designed to compensate the gravity force of the manipulator by its own weight. The conceptual mechanical design is compared by placing the third joint, which controls the four-bar mechanism, in two different configurations. The forward kinematics and the jacobian of GHap are presented. Finally, the gravity compensation method and open-loop force display performance of the proposed haptic device are validated by an experiment with the GHap prototype.

- Enhanced Haptic Sensations Using a Novel Electrostatic Vibration Actuator with Frequency Beating Phenomenon

    Author: Koo, Jeong-Hoi | Miami University
    Author: Schuster, Jeremy M. | Miami University
    Author: Tantiyartyanontha, Takdanai | Miami University
    Author: Kim, Young-Min | Kiom
    Author: Yang, Tae-Heon | Korea National University of Transportation
 
    keyword: Haptics and Haptic Interfaces

    Abstract : Generating haptic sensations in large touch displays is highly desirable, yet producing them for such application is challenging with conventional haptic actuators used in hand-held devices. This study proposes an electrostatic actuator utilizing frequency beating phenomenon with the goals of generating haptic sensations for large touch sensitive displays (TSDs). Unlike typical electrostatic actuators, the proposed haptic actuator incorporates two high-volt electrodes and a spring supported disk (moveable grounded mass) to enhance the intensity and pattern of haptic sensations. After fabricating a proof-of-concept prototype, its performance was experimentally evaluated by varying the beat frequency and the carrier frequency. Using mock-up LCD panels, a feasibility of the proposed actuator was evaluated for generating meaningful haptic sensations in large TSDs. Testing results show that the prototype generated a variety of unique vibration patterns at varying intensities based on combinations of the two input voltage signals. The results further show that the proposed actuator can produce sufficiently strong vibrotactile feedbacks in mock-up panels, indicating that the proposed electrostatic actuators can be a viable option for providing haptic sensations in large TSDs.

- Electromagnetic Haptic Feedback System for Use with a Graphical Display Using Flat Coils and Sensor Array

    Author: Berkelman, Peter | University of Hawaii-Manoa
    Author: Abdul-Ghani, Hamza | California Institute of Technology
 
    keyword: Haptics and Haptic Interfaces; Virtual Reality and Interfaces

    Abstract : We have developed a haptic interaction system which wirelessly generates 3D forces onto a magnet fixed to a user's fingertip or a stylus. Magnet position sensing and haptic force generation are both performed through a thin screen, and the actuation and sensing components are sufficiently thin to be easily mounted behind the screen in the same enclosure. Thus the system is well suited for an interactive co-located haptic and graphic display.<p>The location of the magnet during haptic interaction is obtained by an array of Hall effect sensors. Flat rectangular coils generate Lorentz forces on the magnet and can act in combination to produce 3D forces in any direction, both parallel and normal to the screen surface. The active area is approximately 120x120 mm, with effective force generation and position sensing from the screen to an elevation of approximately 15 mm. The localization methods do not depend on the size, shape, or magnetization strength of the magnet, so that larger or smaller magnets may be used for greater forces or more ease of manipulation, without modifying the software other than the force actuation model. The design of the system is presented, including modeling and analysis of the sensing and actuation methods. Experimental results are given for field sensing, magnet localization, and haptic interaction with simulated objects. Forces up to 3.0 N are demonstrated in different directions while sensing planar magnet position to an accuracy within 2.0 mm.

- An Instrumented Master Tool Manipulator (MTM) for Force Feedback in the Da Vinci Surgical Robot

    Author: Black, David Gregory | University of British Columbia
    Author: Hadi Hosseinabadi, Amir Hossein | University of British Columbia
    Author: Salcudean, Septimiu E. | University of British Columbia
 
    keyword: Haptics and Haptic Interfaces; Surgical Robotics: Laparoscopy; Force and Tactile Sensing

    Abstract : We integrated a force/torque sensor into the wrist of the Master Tool Manipulator (MTM) of the da Vinci Standard Surgical system. The added sensor can be used to monitor the surgeon interaction forces and to improve the haptic experience. The proposed mechanical design is expected to have little effect on the surgeon's operative experience and is simple and inexpensive to implement. We also developed a software package that allows for seamless integration of the force sensor into the da Vinci Research Kit (dVRK) and the Robot Operating System (ROS). The complete mechanical and electrical modifications, as well as the software packages are discussed. Two example applications of impedance control at the MTM and joystick control of the PSM are presented to demonstrate the successful integration of the sensor into the MTM and the interface to the dVRK.

## Visual Tracking

- Multi-Person Pose Tracking Using Sequential Monte Carlo with Probabilistic Neural Pose Predictor

    Author: Okada, Masashi | Panasonic Corporation
    Author: Takenaka, Shinji | Panasonic System Networks R&amp;D Lab. Co., Ltd
    Author: Taniguchi, Tadahiro | Ritsumeikan University
 
    keyword: Visual Tracking; Deep Learning in Robotics and Automation; Probability and Statistical Methods

    Abstract : It is an effective strategy for the multi-person pose tracking task in videos to employ prediction and pose matching in a frame-by-frame manner. For this type of approach, uncertainty-aware modeling is essential because precise prediction is impossible. However, previous studies have relied on only a single prediction without incorporating uncertainty, which can cause critical tracking errors if the prediction is unreliable. This paper proposes an extension to this approach with Sequential Monte Carlo (SMC). This naturally reformulates the tracking scheme to handle multiple predictions (or hypotheses) of poses, thereby mitigating the negative effect of prediction errors. An important component of SMC, i.e., a proposal distribution, is designed as a probabilistic neural pose predictor, which can propose diverse and plausible hypotheses by incorporating epistemic uncertainty and heteroscedastic aleatoric uncertainty. In addition, a recurrent architecture is introduced to our neural modeling to utilize time-sequence information of poses to manage difficult situations, such as the frequent disappearance and reappearances of poses. Compared to existing baselines, the proposed method achieves a state-of-the-art MOTA score on the PoseTrack2018 validation dataset by reducing approximately 50% of tracking errors from a state-of-the art baseline method.

- 4D Generic Video Object Proposals

    Author: Osep, Aljosa | Technical University Munich
    Author: Voigtlaender, Paul | RWTH Aachen University
    Author: Weber, Mark | RWTH Aachen University
    Author: Luiten, Jonathon | RWTH Aachen University
    Author: Leibe, Bastian | RWTH Aachen University
 
    keyword: Visual Tracking; Object Detection, Segmentation and Categorization; Computer Vision for Other Robotic Applications

    Abstract : Many high-level video understanding methods require input in the form of object proposals. Currently, such proposals are predominantly generated with the help of networks that were trained for detecting and segmenting a set of known object classes, which limits their applicability to cases where all objects of interest are represented in the training set. We propose an approach that can reliably extract spatio-temporal object proposals for both known and unknown object categories from stereo video. Our 4D Generic Video Tubes (4D-GVT) method combines motion cues, stereo data, and data-driven object instance segmentation in a probabilistic framework to compute a compact set of video-object proposals that precisely localizes object candidates and their contours in 3D space and time.

- Simultaneous Tracking and Elasticity Parameter Estimation of Deformable Objects

    Author: Sengupta, Agniva | INRIA
    Author: Lagneau, Romain | INSA Rennes
    Author: Krupa, Alexandre | INRIA Rennes - Bretagne Atlantique
    Author: Marchand, Eric | Univ Rennes, Inria, CNRS, IRISA
    Author: Marchal, Maud | INSA/INRIA
 
    keyword: Visual Tracking; RGB-D Perception

    Abstract : In this paper, we propose a novel method to simultaneously track the deformation of soft objects and estimate their elasticity parameters. The tracking of the deformable object is performed by combining the visual information acquired with a RGB-D sensor with interactive Finite Element Method simulations of the object. The visual information is more particularly used to distort the simulated object. In parallel, the elasticity parameter estimation minimizes the error between the tracked object and a simulated object deformed by the forces that are measured using a force sensor. Once the elasticity parameters are estimated, our tracking algorithm can then also be used to estimate the deformation forces applied on an object without the use of a force sensor. We validated our method on several soft objects with different shape complexities. Our evaluations show the ability of our method to estimate the elasticity parameters as well as its use to estimate the forces applied on a deformable object without any force sensor. These results open novel perspectives to better track and control deformable objects during robotic manipulations.

- AVOT: Audio-Visual Object Tracking of Multiple Objects for Robotics

    Author: Wilson, Justin | University of North Carolina at Chapel Hill
    Author: Lin, Ming C. | University of North Carolina
 
    keyword: Visual Tracking

    Abstract : Existing state-of-the-art object tracking can run into challenges when objects collide, occlude, or come close to one another. These visually based trackers may also fail to differentiate between objects with the same appearance but different materials. Existing methods may stop tracking or incorrectly start tracking another object. These failures are uneasy for trackers to recover from since they often use results from previous frames. By using audio of the impact sounds from object collisions, rolling, etc., our audio-visual object tracking (AVOT) neural network can reduce tracking error and drift. We train AVOT end to end and use audio-visual inputs over all frames. Our audio-based technique may be used in conjunction with other neural networks to augment visually based object detection and tracking methods. We evaluate its runtime frames-per-second (FPS) performance and intersection over union (IoU) performance against OpenCV object tracking implementations and a deep learning method. Our experiments, using the synthetic Sound-20K audio-visual dataset, demonstrate that AVOT outperforms single-modality deep learning methods, when there is audio from object collisions. A proposed scheduler network to switch between AVOT and other methods based on audio onset further maximizes accuracy and performance over all frames in multimodal object tracking.

- Efficient Pig Counting in Crowds with Keypoints Tracking and Spatial-Aware Temporal Response Filtering

    Author: Chen, Guang | JD.com
    Author: Shen, Shiwen | JD.com
    Author: Wen, Longyin | JD Digits
    Author: Luo, Si | JD Digits
    Author: Bo, Liefeng | University of Washington
 
    keyword: Visual Tracking; Computer Vision for Other Robotic Applications; Deep Learning in Robotics and Automation

    Abstract : Pig counting is a crucial task for large-scale pig farming. Pigs are usually visually counted by human. But this process is very time-consuming and error-prone. Few studies in literature developed automated pig counting method. The existing works only focused on pig counting using single image, and its level of accuracy faced challenges due to pig movements, occlusion and overlapping. Especially, the field of view of a single image is very limited, and could not meet the needs of pig counting for large pig grouping houses. Towards addressing these challenges, we presented a real-time automated pig counting system in crowds using only one monocular fisheye camera with an inspection robot. Our system showed that it achieved performance superior to human. Our pipeline began with a novel bottom-up pig detection algorithm to avoid false negatives due to overlapping, occlusion and deformable pig shapes. This detection included a deep convolution neural network (CNN) for pig body part keypoints detection and the keypoints association method to identify individual pigs. It then employed an efficient on-line tracking method to associate pigs across image frames. Finally, pig counts were estimated by a novel spatial-aware temporal response filtering (STRF) method to suppress false positives caused by pig or camera movements or tracking failures. The whole pipeline has been deployed in an edge computing device, and demonstrated the effectiveness.

- 6-PACK: Category-Level 6D Pose Tracker with Anchor-Based Keypoints

    Author: Wang, Chen | Shanghai Jiao Tong University
    Author: Martín-Martín, Roberto | Stanford University
    Author: Xu, Danfei | Stanford Univesity
    Author: Lv, Jun | Shanghai Jiao Tong University
    Author: Lu, Cewu | ShangHai Jiao Tong University
    Author: Fei-Fei, Li | Stanford University
    Author: Savarese, Silvio | Stanford University
    Author: Zhu, Yuke | Stanford University
 
    keyword: Visual Tracking; RGB-D Perception; Deep Learning in Robotics and Automation

    Abstract : We present 6-PACK, a deep learning approach to category-level 6D object pose tracking on RGB-D data. Our method tracks in real time novel object instances of known object categories such as bowls, laptops, and mugs. 6-PACK learns to compactly represent an object by a handful of 3D keypoints, based on which the interframe motion of an object instance can be estimated through keypoint matching. These keypoints are learned end-to-end without manual supervision in order to be most effective for tracking. Our experiments show that our method substantially outperforms existing methods on the NOCS category-level 6D pose estimation benchmark and supports a physical robot to perform simple vision-based closed-loop manipulation tasks.

- Multimodal Tracking Framework for Visual Odometry in Challenging Illumination Conditions

    Author: Beauvisage, Axel | City University, London
    Author: Ahiska, Kenan | Cranfield University
    Author: Aouf, Nabil | City University of London
 
    keyword: Visual Tracking; Visual-Based Navigation; Localization

    Abstract : Research on visual odometry and localisation is largely dominated by solutions developed in the visible spectrum, where illumination is a critical factor. Other parts of the electromagnetic spectrum are currently being investigated to generate solutions dealing with extreme illumination conditions. Multispectral setups are particularly interesting as they provide information from different parts of the spectrum at once. However, the main challenge of such camera setups is the lack of similarity between the images produced, which makes conventional stereo matching techniques obsolete.<p>This work investigates a new way of concurrently processing images from different spectra for application to visual odometry. It particularly focuses on the visible and Long Wave InfraRed (LWIR) spectral bands where dissimilarity between pixel intensities is maximal. A new Multimodal Monocular Visual Odometry solution (MMS-VO) is presented. With this novel approach, features are tracked simultaneously, but only the camera providing the best tracking quality is used to estimate motion. Visual odometry is performed within a windowed bundle adjustment framework, by alternating between the cameras as the nature of the scene changes. Furthermore, the motion estimation process is robustified by selecting adequate keyframes based on parallax.</p><p>The algorithm was tested on a series of visible-thermal datasets, acquired from a car with real driving conditions.

- Real-Time Multi-Diver Tracking and Re-Identification for Underwater Human-Robot Collaboration

    Author: de Langis, Karin Johanna Denton | University of Minnesota
    Author: Sattar, Junaed | University of Minnesota
 
    keyword: Visual Tracking; Marine Robotics; Human-Centered Robotics

    Abstract : Autonomous underwater robots working with teams of human divers may need to distinguish between different divers, e.g., to recognize a lead diver or to follow a specific team member. This paper describes a technique that enables autonomous underwater robots to track divers in real time as well as to reidentify them. The approach is an extension of Simple Online Realtime Tracking (SORT) with an appearance metric (deep SORT). Initial diver detection is performed with a custom CNN designed for realtime diver detection, and appearance features are subsequently extracted for each detected diver. Next, realtime tracking-by-detection is performed with an extension of the deep SORT algorithm. We evaluate this technique on a series of videos of divers performing human-robot collaborative tasks and show that our methods result in more divers being accurately identified during tracking. We also discuss the practical considerations of applying multi-person tracking to on-board autonomous robot operations, and we consider how failure cases can be addressed during on-board tracking.

- Autonomous Tissue Scanning under Free-Form Motion for Intraoperative Tissue Characterisation

    Author: Zhan, Jian | Imperial College London
    Author: Cartucho, Jo�o | Imperial College London
    Author: Giannarou, Stamatia | Imperial College London
 
    keyword: Visual Tracking; Visual Servoing; Surgical Robotics: Laparoscopy

    Abstract : In Minimally Invasive Surgery (MIS), tissue scanning with imaging probes is required for subsurface visualisation to characterise the state of the tissue. However, scanning of large tissue surfaces in the presence of motion is a challenging task for the surgeon. Recently, robot-assisted local tissue scanning has been investigated for motion stabilisation of imaging probes to facilitate the capturing of good quality images and reduce the surgeon's cognitive load. Nonetheless, these approaches require the tissue surface to be static or translating with periodic motion. To eliminate these assumptions, we propose a visual servoing framework for autonomous tissue scanning, able to deal with free-form tissue motion. The 3D structure of the surgical scene is recovered, and a feature-based method is proposed to estimate the motion of the tissue in real-time. The desired scanning trajectory is manually defined on a reference frame and continuously updated using projective geometry to follow the tissue motion and control the movement of the robotic arm. The advantage of the proposed method is that it does not require the learning of the tissue motion prior to scanning and can deal with free-form motion. We deployed this framework on the da Vinci surgical robot using the da Vinci Research Kit (dVRK) for Ultrasound tissue scanning. Our framework can be easily extended to other probe-based imaging modalities.

- High Speed Three Dimensional Tracking of Swimming Cell by Synchronous Modulation between TeCE Camera and TAG Lens

    Author: Yamato, Kazuki | Gunma University
    Author: Chiba, Hiroyuki | Gunma University
    Author: Oku, Hiromasa | Gunma University
 
    keyword: Micro/Nano Robots; Visual Tracking; Biological Cell Manipulation

    Abstract : In this study, high speed three dimensional (3D) tracking of a swimming cell was achieved by synchronizing a temporally coded exposure (TeCE) camera and a tunable acoustic gradient index (TAG) lens. Because a TeCE camera can acquire the specific focal plane from a TAG lens at high speed, high speed 3D information acquisition can be accomplished using both a TeCE camera and a TAG lens. In addition, a TeCE camera can acquire the focal plane from a TAG lens irrespective of illumination conditions because a TeCE camera can control exposure. To verify the superiority of the TeCE camera and TAG lens with respect to illumination conditions, we conduct a high speed 3D tracking experiment wherein a swimming cell is tracked via two observation methods, namely bright-field and phase difference observations. The experimental results indicate that high speed 3D tracking can be achieved with both observation methods. Thus, high speed 3D tracking of a swimming cell was confirmed to be independent of illumination conditions.

- Track to Reconstruct and Reconstruct to Track

    Author: Luiten, Jonathon | RWTH Aachen University
    Author: Fischer, Tobias | RWTH Aachen University
    Author: Leibe, Bastian | RWTH Aachen University
 
    keyword: Visual Tracking; Human Detection and Tracking; Deep Learning in Robotics and Automation

    Abstract : Object tracking and 3D reconstruction are often performed together, with tracking used as input for reconstruction. However, the obtained reconstructions also provide useful information for improving tracking. We propose a novel method that closes this loop, first tracking to reconstruct, and then reconstructing to track. Our approach, MOTSFusion (Multi-Object Tracking, Segmentation and dynamic object Fusion), exploits the 3D motion extracted from dynamic object reconstructions to track objects through long periods of complete occlusion and to recover missing detections. Our approach first builds up short tracklets using 2D optical flow, and then fuses these into dynamic 3D object reconstructions. The precise 3D object motion of these reconstructions is used to merge tracklets through occlusion into long-term tracks, and to locate objects when detections are missing. On KITTI, our reconstruction-based tracking reduces the number of ID switches of the initial tracklets by more than 50%, and outperforms all previous approaches for both bounding box and segmentation tracking.

- PointTrackNet: An End-To-End Network for 3-D Object Detection and Tracking from Point Clouds

    Author: Wang, Sukai | Robotics and Multi-Perception Lab (RAM-LAB), Robotics Institute,
    Author: Sun, Yuxiang | Hong Kong University of Science and Technology
    Author: Liu, Chengju | Tongji University
    Author: Liu, Ming | Hong Kong University of Science and Technology
 
    keyword: Automation Technologies for Smart Cities; Service Robots; Field Robots

    Abstract : Recent machine learning-based multi-object tracking (MOT) frameworks are becoming popular for 3-D point clouds. Most traditional tracking approaches use filters (e.g., Kalman filter or particle filter) to predict object locations in a time sequence, however, they are vulnerable to extreme motion conditions, such as sudden braking and turning. In this letter, we propose PointTrackNet, an end-to-end 3-D object detection and tracking network, to generate foreground masks, 3-D bounding boxes, and point-wise tracking association displacements for each detected object. The network merely takes as input two adjacent point-cloud frames. Experimental results on the KITTI tracking dataset show competitive results over the state-of-the-arts, especially in the irregularly and rapidly changing scenarios.


## Planning, Scheduling and Coordination
- An Online Scheduling Algorithm for Human-Robot Collaborative Kitting

    Author: Maderna, Riccardo | Politecnico Di Milano
    Author: Poggiali, Matteo | Politecnico Di Milano
    Author: Zanchettin, Andrea Maria | Politecnico Di Milano
    Author: Rocco, Paolo | Politecnico Di Milano
 
    keyword: Planning, Scheduling and Coordination; Human Factors and Human-in-the-Loop; Industrial Robots

    Abstract : In manufacturing, kitting is the process of grouping separate items together to be supplied as one unit to the assembly line. This is a key logistic task, which is usually performed manually by human operators. However, picking objects from the warehouse implies a great repetitiveness in arm motion. Moreover, the weight and position of items may increase the physical strain and induce the development of work-related musculoskeletal disorders. The inclusion of a collaborative robot in the process may help to reduce the operator's effort and increase productivity. This paper introduces an online scheduling algorithm to guide the picking operations of the human and the robot. The proposed approach has been experimentally evaluated and compared with an offline scheduler, as well as with the baseline case of manual kitting.

- A Model-Free Approach to Meta-Level Control of Anytime Algorithms

    Author: Svegliato, Justin | University of Massachusetts Amherst
    Author: Sharma, Prakhar | University of Massachusetts Amherst
    Author: Zilberstein, Shlomo | University of Massachusetts
 
    keyword: Planning, Scheduling and Coordination; Learning and Adaptive Systems

    Abstract : Anytime algorithms offer a trade-off between solution quality and computation time that has proven to be useful in autonomous systems for a wide range of real-time planning problems. In order to optimize this trade-off, an autonomous system has to solve a challenging meta-level control problem: it must decide when to interrupt the anytime algorithm and act on the current solution. Prevailing meta-level control techniques, however, make a number of unrealistic assumptions that reduce their effectiveness and usefulness in the real world. Eliminating these assumptions, we first introduce a model-free approach to meta-level control based on reinforcement learning and prove its optimality. We then offer a general meta-level control technique that can use different reinforcement learning methods. Finally, we show that our approach is effective across several common benchmark domains and a mobile robot domain.

- Simultaneous Task Allocation and Motion Scheduling for Complex Tasks Executed by Multiple Robots

    Author: Behrens, Jan Kristof | Czech Institute of Informatics, Robotics and Cybernetics, Czech
    Author: Stepanova, Karla | Czech Technical University
    Author: Babuska, Robert | Delft University of Technology
 
    keyword: Planning, Scheduling and Coordination; Intelligent and Flexible Manufacturing; Multi-Robot Systems

    Abstract : The coordination of multiple robots operating simultaneously in the same workspace requires the integration of task allocation and motion scheduling. We focus on tasks in which the robot's actions are not confined to small volumes, but can also occupy a large time-varying portion of the workspace, such as in welding along a line. The optimization of such tasks presents a considerable challenge mainly due to the fact that different variants of task execution exist, for instance, there can be multiple starting points of lines or closed curves, different filling patterns of areas, etc. We propose a generic and computationally efficient optimization method which is based on constraint programming. It takes into account the kinematics of the robots and guarantees that the motions of the robots are collision-free while minimizing the overall makespan. We evaluate our approach on several use-cases of varying complexity: cutting, additive manufacturing, spot welding, inserting and tightening bolts, performed by a dual-arm robot. In terms of the makespan, the result is superior to task execution by one robot arm as well as by two arms not working simultaneously.

- Efficient Planning for High-Speed MAV Flight in Unknown Environments Using Online Sparse Topological Graphs

    Author: Collins, Matthew | Carnegie Mellon University
    Author: Michael, Nathan | Carnegie Mellon University
 
    keyword: Planning, Scheduling and Coordination; Dynamics

    Abstract : Safe high-speed autonomous navigation for MAVs in unknown environments requires fast planning to enable the robot to react quickly to incoming information about obstacles within the world. Furthermore, when operating in environments not known a priori, the robot may make decisions that lead to dead ends, necessitating global replanning through a map of the environment outside of a local planning grid. This work proposes a computationally-efficient planning architecture for safe high-speed operation in unknown environments that incorporates a notion of longer-term memory into the planner enabling the robot to accurately plan to locations no longer contained within a local map. A motion primitive-based local receding horizon planner that uses a probabilistic collision avoidance methodology enables the robot to generate safe plans at fast replan rates. To provide global guidance, a memory-efficient sparse topological graph is created online from a time history of the robot's path and a geometric notion of visibility within the environment to search for alternate pathways towards the desired goal if a dead end is encountered.	The safety and performance of the proposed planning system is evaluated at speeds up to 10m/s, and the approach is tested in a set of large-scale, complex simulation environments containing dead ends. These scenarios lead to failure cases for competing methods; however, the proposed approach enables the robot to safely reroute and reach the goal.

- Evaluating Adaptation Performance of Hierarchical Deep Reinforcement Learning

    Author: Van Stralen, Neale | University of Illinois at Urbana Champaign
    Author: Kim, Seung Hyun | University of Illinois at Urbana-Champaign
    Author: Tran, Huy | University of Illinois at Urbana Champaign
    Author: Chowdhary, Girish | University of Illinois at Urbana Champaign
 
    keyword: Planning, Scheduling and Coordination; Path Planning for Multiple Mobile Robots or Agents

    Abstract : Deep Reinforcement Learning has been used to exploit specific environments, but has difficulty transferring learned policies to new situations. This issue poses a problem for practical applications of Reinforcement Learning, as real-world scenarios may introduce unexpected differences that drastically reduce policy performance. We propose the use of differentiated sub-policies governed by a hierarchical controller to support adaptation in such scenarios. We also introduce a confidence-based training process for the hierarchical controller which improves training stability and convergence times. We evaluate these methods in a new Capture the Flag environment designed to explore adaptation in autonomous multi-agent settings.

- An Approximation Algorithm for a Task Allocation, Sequencing and Scheduling Problem Involving a Human-Robot Team

    Author: Hari, Sai Krishna Kanth | Texas a &amp; M University
    Author: Nayak, Abhishek | Texas a &amp; M University
    Author: Rathinam, Sivakumar | TAMU
 
    keyword: Planning, Scheduling and Coordination; Path Planning for Multiple Mobile Robots or Agents; Task Planning

    Abstract : This article presents an approximation algorithm for a Task Allocation, Sequencing and Scheduling Problem (TASSP) involving a team of human operators and robots. The robots have to travel to a given set of targets and collaboratively work on the tasks at the targets with the human operators. The problem aims to find a sequence of targets for each robot to visit and schedule the tasks at the targets with the human operators such that each target is visited exactly once by some robot, the scheduling constraints are satisfied and the maximum mission time of any robot is minimum. This problem is a generalization of the single Traveling Salesman Problem and is NP-Hard. Given k robots and m human operators, an algorithm is developed for solving the TASSP with an approximation ratio equal to 5/2-1/k when m&gt;=k and equal to 7/2-1/k otherwise. Computational results are also presented to corroborate the performance of the proposed algorithm.

## Reactive and Sensor-Based Planning
- Iterator-Based Temporal Logic Task Planning

    Author: Zudaire, Sebastian | Universidad De Buenos Aires
    Author: Garrett, Martin | CNEA
    Author: Uchitel, Sebastian | Universidad De Buenos Aires
 
    keyword: Reactive and Sensor-Based Planning; Task Planning

    Abstract : Temporal logic task planning for robotic systems suffers from state explosion when specifications involve large numbers of discrete locations. We provide a novel approach, particularly suited for tasks specifications with universally quantified locations, that has constant time with respect to the number of locations, enabling synthesis of plans for an arbitrary number of them. We propose a hybrid control framework that uses an iterator to manage the discretized workspace hiding it from a plan enacted by a discrete event controller. A downside of our approach is that it incurs in increased overhead when executing a synthesised plan. We demonstrate that the overhead is reasonable for missions of a fixed-wing Unmanned Aerial Vehicle in simulated and real scenarios for up to 700 000 locations.

- Reactive Temporal Logic Planning for Multiple Robots in Unknown Environments

    Author: Kantaros, Yiannis | University of Pennsylvania
    Author: Malencia, Matthew | University of Pennsylvania
    Author: Kumar, Vijay | University of Pennsylvania
    Author: Pappas, George J. | University of Pennsylvania
 
    keyword: Reactive and Sensor-Based Planning; Path Planning for Multiple Mobile Robots or Agents; Autonomous Agents

    Abstract : This paper proposes a new reactive mission planning algorithm for multiple robots that operate in unknown environments. The robots are equipped with individual sensors that allow them to collectively learn and continuously update a map of the unknown environment. The goal of the robots is to accomplish complex tasks, captured by global co-safe Linear Temporal Logic (LTL) formulas. The majority of existing temporal logic planning approaches rely on discrete     Abstractions of the robot dynamics operating in known environments and, as a result, they cannot be applied to the more realistic scenarios where the environment is initially unknown. In this paper, we address this novel challenge by proposing the first reactive, and     Abstraction-free LTL planning algorithm that can be applied for complex mission planning of multiple robots operating in unknown environments. Our algorithm is reactive in the sense that temporal logic planning is adapting to the updated map of the environment and     Abstraction-free as it does not rely on designing     Abstractions of robot dynamics. Our proposed algorithm is complete under mild assumptions on the structure of the environment and the sensor models. Our paper provides extensive numerical simulations and hardware experiments that illustrate the theoretical analysis and show that the proposed algorithm can address complex planning tasks in unknown environments.

- Higher Order Function Networks for View Planning and Multi-View Reconstruction

    Author: Engin, Kazim Selim | University of Minnesota
    Author: Mitchell, Eric | Samsung AI Center NY
    Author: Lee, Daewon | Samsung AI Center New York
    Author: Isler, Volkan | University of Minnesota
    Author: Lee, Daniel | Cornell Tech
 
    keyword: Reactive and Sensor-Based Planning; Sensor Fusion; Learning and Adaptive Systems

    Abstract : We consider the problem of planning views for a robot to acquire images of an object for visual inspection and reconstruction. In contrast to offline methods which require a 3D model of the object as input or online methods which rely on only local measurements, our method uses a neural network which encodes shape information for a large number of objects. We build on recent deep learning methods capable of generating a complete 3D reconstruction of an object from a single image. Specifically, in this work, we extend a recent method which uses Higher Order Functions (HOF) to represent the shape of the object. We present a new generalization of this method to incorporate multiple images as input and establish a connection between visibility and reconstruction quality. This relationship forms the foundation of our view planning method where we compute viewpoints to visually cover the output of the multi-view HOF network with as few images as possible. Experiments indicate that our method provides a good compromise between online and offline methods: Similar to online methods, our method does not require the true object model as input. In terms of number of views, it is much more efficient. In most cases, its performance is comparable to the optimal offline case even on object classes the network has not been trained on.

- Residual Reactive Navigation: Combining Classical and Learned Navigation Strategies for Deployment in Unknown Environments

    Author: Rana, Krishan | Queensland University of Technology
    Author: Talbot, Ben | Queensland University of Technology
    Author: Dasagi, Vibhavari | Queensland University of Technology
    Author: Milford, Michael J | Queensland University of Technology
    Author: S�nderhauf, Niko | Queensland University of Technology
 
    keyword: Reactive and Sensor-Based Planning; Deep Learning in Robotics and Automation

    Abstract : In this work we focus on improving the efficiency and generalisation of learned navigation strategies when transferred from its training environment to previously unseen ones. We present an extension of the residual reinforcement learning framework from the robotic manipulation literature and adapt it to the vast and unstructured environments that mobile robots can operate in. The concept is based on learning a residual control effect to add to a typical sub-optimal classical controller in order to close the performance gap, whilst guiding the exploration process during training for improved data efficiency. We exploit this tight coupling and propose a novel deployment strategy, switching Residual Reactive Navigation (sRNN), which yields efficient trajectories whilst probabilistically switching to a classical controller in cases of high policy uncertainty. Our approach achieves improved performance over end-to-end alternatives and can be incorporated as part of a complete navigation stack for cluttered indoor navigation tasks in the real world. The code and training environment for this project is made publicly available at url{https://sites.google.com/view/srrn/home}.

- Online Grasp Plan Refinement for Reducing Defects During Robotic Layup of Composite Prepreg Sheets

    Author: Malhan, Rishi | University of Southern California
    Author: Jomy Joseph, Rex | University of Southern California
    Author: Shembekar, Aniruddha | University of Southern California
    Author: Kabir, Ariyan M | University of Southern California
    Author: Bhatt, Prahar | University of Southern California
    Author: Gupta, Satyandra K. | University of Southern California
 
    keyword: Reactive and Sensor-Based Planning; Failure Detection and Recovery; Industrial Robots

    Abstract : High-performance composites are increasingly being used in the industry. Sheet layup is a process of manufacturing composite components using deformable sheets. We have developed a robotic cell to automate the layup process and overcome the limitations of the manual layup. Generating offline trajectories for robots and executing them without online refinement can introduce defects in the process due to uncertainties in the model of the sheet and environmental factors. Our system computes layup and grasping trajectories for the robots and refines them during the layup process based on the sensor data. We use an approach that augments physical experiments with simulations to train a Gaussian process regression model offline. The use of GPR enables us to quickly refine grasp plans and perform a defect-free layup without slowing down the layup process. We present experimental results on two components.

- Object-Centric Task and Motion Planning in Dynamic Environments

    Author: Migimatsu, Toki | Stanford University
    Author: Bohg, Jeannette | Stanford University
 
    keyword: Reactive and Sensor-Based Planning; Task Planning; Optimization and Optimal Control

    Abstract : We address the problem of applying Task and Motion Planning (TAMP) in real world environments. TAMP combines symbolic and geometric reasoning to produce sequential manipulation plans, typically specified as joint-space trajectories, which are valid only as long as the environment is static and perception and control are highly accurate. In case of any changes in the environment, slow re-planning is required. We propose a TAMP algorithm that optimizes over Cartesian frames defined relative to target objects. The resulting plan then remains valid even if the objects are moving and can be executed by reactive controllers that adapt to these changes in real time. We apply our TAMP framework to a torque-controlled robot in a pick and place setting and demonstrate its ability to adapt to changing environments, inaccurate perception, and imprecise control, both in simulation and the real world.

