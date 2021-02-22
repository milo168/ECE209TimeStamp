## Time Sync Study

This is the project repository website for UCLA ECE 209 AS-2 Winter 2021 supervised by Prof. Mani B. Srivastava. The project will focus on exploring state-of-the-art methods which try to mitigate faulty time stamps on multimodal data, evaluating them using different datasets and metrics, and trying to propose new methods based on them to improve the performance.

### Goals
The time synchronization of multiple sensor streams is a long-standing challenge. Inaccurate time stamps on multimodal data can cause a lot of problems like accuracy loss on models that try to classify a series of actions. Some works try to tackle the source of the problem and minimize the error by enforcing synchronization at the hardware or software level during capture. Some works exploit the cross-correlation of the data from different sensors to predict and correct the time shifts before using them further. Other works have focused on training deep learning models that can be resilient to bad time stamps using data augmentation technologies.

This project will focus on the cross-correlation-based and deep-learning-based methods. Specifically, we choose two state-of-the-art works “SyncWISE” and “TimeAwareness” to explore and investigate.

#### Basic Objectives
- Propose a common metric to handle their different kinds of outputs to beter compare them
- Explore how they generalize on different multimodal data
- Explore how long time shift and how short data they can handle
- Explore to combine SyncWISE and TimeAwareness and evaluate the performance

#### Stretch Objectives
- Try to propose new methods based on them with better performance across multiple metrics

### Approach
#### Propose a common metric to handle their different kinds of outputs to beter compare them
Though SyncWISE and TimeAwareness try to handle similar time synchronization errors across multimodal data, they have different specific aim, generate different kinds of outputs, and use different metrics to evaluate their methods. Therefore, to compare these two methods, we first need to propose a common metric to show their performance under the same criteria. 

Window Induced Shift Estimation method for Synchronization (SyncWISE) mainly aims to synchronize video data with adjacent sensor streams, which can enable temporal alignment of video-derived labels to diverse wearable sensor signals. It uses the cross-correlation of the multimodal data to estimate the offsets. It outputs the time shifts between video and other sensor data, so they use two metrics to evaluate SyncWISe's performance: 1) the average of the absolute value of the synchronization error, $Eavg$, and 2) the percentage of clips which are synchronized to an offset error of less than $n$ ms, $Pv-n$. TimeAwareness tries to use the data directly instead of aligning them. It induces time synchronization errors when training the deep learning models to improve models' robustness. They add artificial shifts in test data and check the models' test accuracies to evaluate their method. Here, we first train a baseline model, and then add SyncWISE (when testing) and TimeAwareness (when training) separately, and use models' accuracies as the metric to show their effect and performance.

The details of training and testing of these three approaches are shown below. We will use the same original model architecture and dataset.

Approach Baseline: 
- Training + Validation without data augmentation (domain randomization)
- Testing: introduce artificial shifts in test data, without SyncWISE correction, and then classify to check the accuracy

Approach 1 (SyncWISE): 
- Training + Validation without data augmentation (domain randomization)
- Testing: introduce artificial shifts in test data, correct using SyncWISE, and then classify to check the accuracy
  - The classifier here is to show the effects of SyncWISE
  - Also try to run another cross-correlation based approach (baseline method in SyncWISE paper)
	
Approach 2 (TimeAwareness):
- Training + Validation with data augmentation (domain randomization)
  - Expand the training and validation data with additional copies that have random shifts
- Testing: introduce artificial shifts in test data, without SyncWISE correction, and then classify to check the accuracy
  - The classifier here is to show the effect of data augmentation

#### Explore how they generalize on different multimodal data
Although SyncWISE and TimeAwareness claim that their algorithm can be further adapted to many other sensing modalities, they only evaluate their methods on limited sensor data. SyncWISE uses video and accelerometry data from S2S-Sync dataset and CMU-MMAC dataset. TimeAwareness uses audio and IMU data from CMActivities dataset. So it will be interesting to explore how they generalize on different multimodal data. 

Since the experiments can take a long time even using many computation resources, like for SyncWISE "It will take about 5 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores" for simulated shifts and "It will take 10 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores" for real shifts, we plan to explore video-accelerometry and audio-IMU data on them first.

#### Explore how long time shift and how short data they can handle

TimeAwareness introduce at most 1000ms and 2000ms shifts in their 10-Sec training and testing data respectively, and it can preserve classifier accuracy at most 600ms of timing error. For SyncWISE, they generate a synthetic testing dataset based on S2S-Sync dataset by adding random offsets in the range [-3 sec, 3 sec]. And the original offsets of S2S-Sync dataset have a complex distribution, with an average offset of 21s, max offset of 180s, and min offset of 387ms. S2S-Sync dataset have 163 video clips of 45.2 hours, which means the average period of each clip is 998.28s.

We will explore and compare how long time shift TimeAwareness and SyncWISE can handle given the same data of the same length. Besides, we will gradually reduce the length of the data and record their performance changes to see how short data they can handle.


#### Explore to combine SyncWISE and TimeAwareness and evaluate the performance
Since both SyncWISE and TimeAwareness aims to handle time shifts across multimodal data, we can explore to combinie them and evaluate the performance.

Approach SyncWISE + TimeAwareness:
- Training + Validation with data augmentation (domain randomization)
- Testing: introduce artificial shifts in test data, correct using SyncWISE, and then classify to check the accuracy

#### Try to propose new methods based on them with better performance across multiple metrics



### Implementation and Results

### Prior Work

### Analysis and Future Direction

### Project Timeline 

### Contribution

### PDF

### References
