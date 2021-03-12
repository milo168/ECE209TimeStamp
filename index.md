## Time Sync Study

This is the project repository website for UCLA ECE 209 AS-2 Winter 2021 supervised by Prof. Mani B. Srivastava. The project will focus on exploring state-of-the-art methods which try to mitigate faulty time stamps on multimodal data, evaluating them using different metrics, and trying to propose new methods based on them to improve the performance.

<div align=center><img width="600" height="330" src="./Images/TimeShift_SyncWISE.png"/></div>

<center>Time Shifts in Multimodal Data from SyncWISE Paper</center>

------
### Background and Goals
Multi-modality sensors offer a wider insight into understanding complex behaviors. For example, recognizing human activities such as differentiating washing hands or walking using more than one type of sensors. However, there is a challenge in lining up the data to a reference point. It may be possible one sensor has timestamps relatively faster or slower than a reference point. This issue of misaligned timestamps present challenges to deep learning models that use more than one input. During training, the model would have a hard time learning features as features corresponding to one activity leaks into another. 

Some works try to tackle the issue of misalignment through several ways. Some try to fix the source of the problem and minimize the error by enforcing synchronization at the hardware or software level during capture. Some exploit the cross-correlation of the data from different sensors to predict and correct the time shifts before using them further. Other works have focused on training deep learning models that can be robust to bad timestamps using data augmentation techniques.

This project will focus on the cross-correlation-based and deep-learning-based methods. Specifically, we will explore and investigate two state-of-the-art works “SyncWISE” and “Time Awareness”.

#### Basic Objectives
- Explore the range of shifts that can be handled by each component
- Explore to combine SyncWISE and Time Awareness and evaluate the performance

#### Future Objectives
- Propose new methods with these two works as foundations for better performance measured on different metrics

------
### Approach
#### Measuring Contribution
Though SyncWISE and Time Awareness try to handle similar time synchronization errors across multimodal data, they have different specific goals, generate different kinds of outputs, and use different metrics to evaluate their methods. Therefore, to measure their effectivness, we need to evaluate both contributions individually. Finally, we will combine both methods to see if further improvement can be gained.

Window Induced Shift Estimation method for Synchronization (SyncWISE) mainly aims to synchronize video data with adjacent sensor streams, which can enable temporal alignment of video-derived labels to diverse wearable sensor signals. It uses cross-correlation to match multimodal data to the offsets. The output is the time shifts between video and other sensor data. SyncWISE uses two metrics to evaluate performance: 

1. The average of the absolute value of the synchronization error, $Eavg$.
2. The percentage of clips which are synchronized to an offset error of less than $n$ ms, $Pv-n$. 

Time Awareness tries to use the data directly instead of aligning them. It induces time synchronization errors during training to improve the model's robustness. They add artificial shifts in the dataset and check the model's test accuracies to evaluate the effectiveness. 

Here, we will evaluate four combinations to understand the effectiveness of each work. The details of training and testing of these three approaches are shown below. We will use the same original model architecture and dataset if possible.

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

Approach 4 (SyncWISE + TimeAwareness Robust):
- Training + Validation with data augmentation (domain randomization) of shifted data
- Testing: introduce artificial shifts in test data, WITH SyncWISE correction, and then feed to classifier

#### Exploring the Dataset
Time Awareness uses the CMActivity dataset which contains 3 different sensor modalities. These modalities are video, audio, and inertial measurement units. In this dataset, there are 7 human activities roughly distributed equally. There are roughly 12,000 train+validate samples and 1,400 test samples total. It is assumed that the CMActvity dataset does not have any time shifts and so we will need to generate fake shifts ourselves. To do this, we pick the IMU modality to be shifted. To generate shifted samples, we first rearrange the samples into a long sequence. Inside the sequence contains windows of the activities of roughly 10 seconds. Then we shift the IMU samples. For the CMActivity dataset, we generated shifts ranging from 50ms to 2000ms.

SyncWISE uses the CMU-MMAC dataset which contains 2 sensor modalities. These modalities are video and inertial measurement units. There are a total of 163 videos representing 45.2 hours of data, averaging 998.28s per clip. The dataset is augmented by adding random offsets in the range [-3 sec, 3 sec] is used. The original offsets of the S2S-Sync dataset has a complex distribution, with an average offset of 21s, max offset of 180s, and min offset of 387ms.

The dataset we will be using is the CMActivity dataset since the IMU deep learning model is provided by Time Awareness. In addition, we will be using CMAcitvity dataset for the model and IMU samples. Since we are more familiar with thise dataset, this will provide us an anchor point as we will be able to tell if something went wrong.

------
### Expectations
We were able to run the codes from both paper to get a feel. The Time Awareness code was straight forward to run and can easily be modified to use video instead of audio. We expect that most of our time will be spent on tweaking SyncWISE to be able to use the CMActivity dataset. 

We believe that SyncWISE will play a significant role in error correction since we think that for Time Awareness, the model learns by feature. By introducing shifts into the training dataset, we think that the model will only memorize the shifts it has seen. If we were to give it shifts that were more than what it has seen during training, the accuracy would drop. This is confirmed in the paper as it does poorly pass 1000ms.

##### Re-run SyncWISE and TimeAwareness
Though they provides most codes and dataset on their GitHub, it took some time to debug and re-run their codes, especially for SincWISE "It will take about 5 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores" for simulated shifts and "It will take 10 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores" for real shifts. Finally, we run the codes successfully and get results as seen in the papers.

##### SyncWISE：

<center>Results from our re-run codes</center>

|Method|Ave #Win Pairs|Ave Error (ms)|PV300 (%)|PV700 (%)|
| :----: | :----: | :----: | :----: | :----: |
|Baseline-xx|1|30032.676735022145|55.38461538461539|83.07692307692308|
|Baseline-PCA|1|53093.94500744793| 50.0| 70.0|
|SyncWISE-xx|1403.4546153846159|447.0069581348005|62.35897435897436|89.7179487179487|
|SyncWISE|1403.4546153846159|416.2898679848237|73.38461538461539|88.7179487179487|


<div align=center><img width="700" height="160" src="./Images/Results_SyncWISE.png"/></div>

<center>Results from SyncWISE Paper</center>

##### TimeAwareness：


<div align=center><img width="400" height="160" src="./Images/Result_TimeAwareness.png"/></div>


<div align=center><img width="400" height="160" src="./Images/Result_Baseline.png"/></div>

<center>Results from Time Awareness where top model is from the paper</center>

------
### Implementation and Results
#### Deep Learning Models
We will keep the same IMU model used in Time Awareness. This model contains 2 convolution layers and 3 fully connected layers. For the video model we will be using C3D model. This model has 4 3d convolution layers and 3 fully connected layers. When independently training these models, the IMU accuracy is 91.65% and the video accuracy is 93.25%. We then build a fusion model using these 2. The fusion model has an additional fully connected layer and has an accuracy of 97.17%, showing that multi-modal models help improve accuracy.


<div align=center><img width="400" height="160" src="./Images/C3D_model.png"/></div>

<center>Model chosen for the video modality</center>

#### Approach Baseline (NO SyncWISE + Time Awareness Non-Robust): 
The training of this fusion model did not involve and shifted data. The testing dataset contained shifts ranging from 50ms to 2000ms. It was not corrected by SyncWISE and was given to the fusion model as is. As we can see in the image below, as the shifts become more pronounced, the accuracy of the model drops.

#### Approach 2 (NO SyncWise + Time Awareness Robust):
The training of this fusion model included shifted data. The testing dataset contained shifts ranging from 50ms to 2000ms. It was not corrected by SyncWISE was was given to the fusion model as is. As expected, this model performs well up to the shifts it has been trained on. As we can see in the images below, when trained only up to 1000ms, the accuracy past 1000ms starts to drop. However, if trained up to 2000ms the accuracy does not drop as the previous one.

<div align=center><img width="400" height="160" src="./Images/Result_1s.png"/></div>

<div align=center><img width="400" height="160" src="./Images/Result_2s.png"/></div>

<center>Results when training with different amount of shifts</center>

------
### Prior Work
1. SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors

2. TimeAwareness：Time Awareness in Deep Learning-Based Multimodal Fusion Across Smartphone Platforms



------
### Analysis and Future Directions (to be updated dynamically)
Strengths and weakness, and future directions.
#### Future directions
Due to time limitation, here are some topics may be interesting and can be explored in the future:
- Explore Generalizing On Different Multimodal Data	
  - Although SyncWISE and Time Awareness claim that their algorithms can be further adapted to many other sensing modalities, they only evaluate their methods on limited sensor data. SyncWISE uses video and accelerometry data from S2S-Sync dataset and CMU-MMAC dataset. Time Awareness uses audio and IMU data from CMActivities dataset. It will be interesting to explore how they generalize on different multimodal data. 	
  - Since the experiments can take a long time even using many computation resources, like for SyncWISE "It will take about 5 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores" for simulated shifts and "It will take 10 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores" for real shifts, we may can explore multimodal data in the future.
- Explore how to predict time shifts using deep learning models
  - Time Awareness uses deep learning models to handle multimodal data with faulty time stamps directly. It will be interesting to explore how to predict time shifts using deep learning models.


------
### Project Timeline (to be updated dynamically)
- Week 1 and 2
  - Search for project ideas and discuss with instructor
- Week 3 and 4
  - Literature review for faulty timestamp and its consequences, especially SyncWISE and Time Awareness
  - Team up
  - Second discussion about final idea with instructor
- Week 5
  - Determine the final project idea
  - Initial project website and Github repository
- Week 6 and 7
  - Re-run the codes of SyncWISE
  - Re-run the codes of Time Awareness
  - Update project website and Github repository
  - Prepare Project Midterm Presentation
- Week 8
  - Project Midterm Presentation
  - Trying Approach Baseline
- Week 9
  - Modify SyncWISE to use CMACtivity
  - Train video model and fusion model

------
### Contribution (to be updated dynamically)
Gaofeng Dong 
- Implementation and maintenance of the project website and Github repository
- Survey of literature related to bad timestamp, especially SyncWISE and Time Awareness
- Rerun the codes of SyncWISE and Time Awareness
- Prepare Project Midterm Presentation

Michael Lo
- Design and maintenance of the project website and Github repository
- Survey of literature related to bad timestamp, especially SyncWISE and Time Awareness
- Rerun the codes of SyncWISE and Time Awareness
- Prepare Project Midterm Presentation


------
### PDF
Section with links to PDF of your final presentation slides, and any data sets not in your repo.

------
### References
1. Zhang Y C, Zhang S, Liu M, et al. SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors[J]. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2020, 4(3): 1-26.
  - Paper: <https://dl.acm.org/doi/abs/10.1145/3411824>
  - Website: <https://github.com/HAbitsLab/SyncWISE>
  - Short Talk: <https://www.youtube.com/watch?v=p86hH8O5xhs>

2. Sandha S S, Noor J, Anwar F M, et al. Time awareness in deep learning-based multimodal fusion across smartphone platforms[C]//2020 IEEE/ACM Fifth International Conference on Internet-of-Things Design and Implementation (IoTDI). IEEE, 2020: 149-156.
  - Paper: <https://ieeexplore.ieee.org/document/9097594>
  - Website: <https://github.com/nesl/CMActivities-DataSet>

3. Fridman L, Brown D E, Angell W, et al. Automated synchronization of driving data using vibration and steering events[J]. Pattern Recognition Letters, 2016, 75: 9-15.
  - Paper: <https://arxiv.org/abs/1510.06113>

4. Adams R, Marlin B M. Learning Time Series Segmentation Models from Temporally Imprecise Labels[C]//UAI. 2018: 135-144.
  - Paper: <http://auai.org/uai2018/proceedings/papers/50.pdf>

5. Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresami, Manohar Paluri Learning Spatiotemporal Features with 3D Convolution Neural Networks[C]. 2015 IEEE International Conference on Computer Vision. IEEE, 2015: 4489-4497
  - Paper: <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf>


