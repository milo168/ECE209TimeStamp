## Time Sync Study

This is the project repository website for UCLA ECE 209 AS-2 Winter 2021 supervised by Prof. Mani B. Srivastava. The project will focus on exploring state-of-the-art methods which try to mitigate faulty time stamps on multimodal data, evaluating them using different datasets and metrics, and trying to propose new methods based on them to improve the performance.

### Goals
The time synchronization of multiple sensor streams is a long-standing challenge. Inaccurate time stamps on multimodal data can cause a lot of problems like accuracy loss on models that try to classify a series of actions. Some works try to tackle the source of the problem and minimize the error by enforcing synchronization at the hardware or software level during capture. Some works exploit the cross-correlation of the data from different sensors to predict and correct the time shifts before using them further. Other works have focused on training deep learning models that can be resilient to bad time stamps using data augmentation technologies.

This project will focus on the cross-correlation-based and deep-learning-based methods. Specifically, we choose two state-of-the-art works “SyncWISE” and “TimeAwareness” to explore and investigate.

#### Basic Objectives
- Propose a common metric to handle their different kinds of outputs to beter compare them
- Explore how they generalize on different multimodal data
- Explore how long time shift they can handle
- Explore how short data they need
- Explore their tolerance to noise

#### Stretch Objectives
- Try to propose new methods based on them with better performance across multiple metrics

### Approach
#### Propose a common metric to handle their different kinds of outputs to beter compare them

Baseline: 
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


### Implementation and Results

### Prior Work

### Analysis and Future Direction

### Contribution

### PDF

### References
