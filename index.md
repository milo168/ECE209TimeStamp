## Time Sync Study

This is the project repository website for UCLA ECE 209 AS-2 Winter 2021 supervised by Prof. Mani B. Srivastava. The project will focus on exploring state-of-the-art methods which try to mitigate faulty time stamps on multimodal data, evaluating them using different datasets and metrics, and trying to propose new methods based on them to improve the performance.

### Goals
The time synchronization of multiple sensor streams is a long-standing challenge. Inaccurate time stamps on multimodal data can cause a lot of problems like accuracy loss on models that try to classify a series of actions. Some works try to tackle the source of the problem and minimize the error by enforcing synchronization at the hardware or software level during capture. Some works exploit the cross-correlation of the data from different sensors to predict and correct the time shifts before using them further. Other works have focused on training deep learning models that can be robust to bad timestamps using data augmentation techniques.

This project will focus on the cross-correlation-based and deep-learning-based methods. Specifically, we choose two state-of-the-art works “SyncWISE” and “TimeAwareness” to explore and investigate.

#### Basic Objectives
- Propose a common metric to handle their different kinds of outputs to better compare the effects
- Explore the range of shifts that can be handled by each component
- Explore to combine SyncWISE and Time Awareness and evaluate the performance

#### Future Objectives
- Propose new methods with these two works as foundations for better performance measured on different metrics

### Approach (to be updated dynamically)
#### Proposing a Common Measurement Metric
Though SyncWISE and Time Awareness try to handle similar time synchronization errors across multimodal data, they have different specific goals, generate different kinds of outputs, and use different metrics to evaluate their methods. Therefore, to compare these two methods, we first need to propose a common metric to show their performance under the same criteria. 

Window Induced Shift Estimation method for Synchronization (SyncWISE) mainly aims to synchronize video data with adjacent sensor streams, which can enable temporal alignment of video-derived labels to diverse wearable sensor signals. It uses cross-correlation to match multimodal data to the offsets. The output is the time shifts between video and other sensor data. SyncWISE uses two metrics to evaluate performance: 

1. The average of the absolute value of the synchronization error, $Eavg$.
2. The percentage of clips which are synchronized to an offset error of less than $n$ ms, $Pv-n$. 

Time Awareness tries to use the data directly instead of aligning them. It induces time synchronization errors when training the deep learning models to improve the model's robustness. They add artificial shifts in the dataset and check the model's test accuracies to evaluate the effectiveness. 

Here, we propose four combinations to understand the effectiveness of each work. The details of training and testing of these three approaches are shown below. We will use the same original model architecture and dataset if possible.

Approach Baseline (NO SyncWISE + Time Awareness Non-Robust): 
- Training + Validation WITHOUT data augmentation (domain randomization) of shifted data
- Testing: Introduce artificial shifts in test data, WITHOUT SyncWISE correction, and then feed to classifier

Approach 2 (NO SyncWise + Time Awareness Robust):
- Training + Validation WITH data augmentation (domain randomization) of shifted data
  - Expand the training and validation data with additional copies that have random shifts
- Testing: introduce artificial shifts in test data, WITHOUT SyncWISE correction, and then feed to classifier
  - The classifier here is to show the effects of shifted data augmentation

Approach 3 (SyncWISE + Time Awareness Non-Robust): 
- Training + Validation WITHOUT data augmentation (domain randomization) of shifted data
- Testing: Introduce artificial shifts in test data, WITH SyncWISE correction, and then feed to classifier
  - The classifier here is to show the effects of SyncWISE
  - Also try to run another cross-correlation based approach (baseline method in SyncWISE paper)


#### Exploring Time Shift Range Effectivness

Time Awareness introduces at most 1000ms and 2000ms shifts in its 10-Sec training and testing data respectively, and it can preserve classifier accuracy up to 600ms of timing error. For SyncWISE, a synthetic testing dataset based on S2S-Sync dataset by adding random offsets in the range [-3 sec, 3 sec] is used. The original offsets of the S2S-Sync dataset has a complex distribution, with an average offset of 21s, max offset of 180s, and min offset of 387ms. S2S-Sync dataset has 163 video clips totaling 45.2 hours, which means the average period of each clip is 998.28s.

We will explore and compare how long time shift Time Awareness and SyncWISE can handle if given the same length of timed data. We will observe how well each algorithm works by gradually reducing the length of the data and record their performance change to see its limit.


#### Combining SyncWISE + Time Awareness
Since both SyncWISE and Time Awareness aims to handle time shifts across multimodal data, we will explore combining the two and evaluate the performance.

Approach 4 (SyncWISE + TimeAwareness Robust):
- Training + Validation with data augmentation (domain randomization) of shifted data
- Testing: introduce artificial shifts in test data, WITH SyncWISE correction, and then feed to classifier

#### Modify Current Deep Learning Model
In Time Awareness, they use a simple model with two convolutional layers and several fully connected layers. To improve the capacity of the model, we can try more complex architectures like LSTM to get better performance. 

### Implementation and Results (to be updated dynamically)
#### Re-run SyncWISE and TimeAwareness
Though they provides most codes on their GitHub, it took some time to debug and re-run their codes. Finally, we get results as seen in the papers.

SyncWISE：

TimeAwareness：


### Prior Work
1. SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors

2. TimeAwareness：Time Awareness in Deep Learning-Based Multimodal Fusion Across Smartphone Platforms


### Analysis and Future Direction (to be updated dynamically)

### Project Timeline (to be updated dynamically)
- Week 1 and 2
  - Search for project ideas and discuss with instructor
- Week 3 and 4
  - Literature review on Bad TimeStamps, especially SyncWISE and Time Awareness
  - Team up
  - Second discussion about final idea with instructor
- Week 5
  - Determine the final project idea
  - Initinal project website and Github repository
- Week 6 and 7
  - Re-run the codes of SyncWISE
  - Re-run the codes of Time Awareness
  - Update project website and Github repository
  - Prepaer Project Midterm Presentation
- Week 8
  - Project Midterm Presentation
  - Trying Approach Baseline


### Contribution (to be updated dynamically)

### PDF

### References
1. SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors
- Paper: https://dl.acm.org/doi/abs/10.1145/3411824
- Website: https://github.com/HAbitsLab/SyncWISE

2. Time Awareness in Deep Learning-Based Multimodal Fusion Across Smartphone Platforms
- Paper: https://ieeexplore.ieee.org/document/9097594
- Website: https://github.com/nesl/CMActivities-DataSet

3. Automated Synchronization of Driving Data Using Vibration and Steering Events
- Paper: https://arxiv.org/abs/1510.06113

4. 



