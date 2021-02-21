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

Approach Baseline: 
- Training+Validation without data augmentation (domain randomization)
- Testing: subject test data to artificial shifts, no fixing, and then classify

Approach 1: 
- Training+Validation without data augmentation (domain randomization)
  - Use syncwise to realign data and labels first (But SyncWISE may be inaccurate and will introduce shifts
  - Training data may be not sync, except that make sure the data are sync 
  - Need to use syncwise to align the testing data too
- Testing: subject test data to artificial shifts, then fix using syncwise, and then classify
  - The classifier here is to show the effect of syncwise
  - (Fix: Syncwise, there was another approach in Syncwise paper
		
Approach 2:
- Training+Validation with data augmentation (domain randomization)
  - I.e. you’d expand the training and validation data with additional copies that have random shifts. 
- Testing: subject test data to artificial shifts, no fixing, and then classify
  - The classifier here is to show the effect of data augmentation



### Implementation and Results

### Prior Work

### Analysis and Future Direction

### Contribution

### PDF

### References
