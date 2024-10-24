# Updates
**\* This repository will be fully available after publication.**

This repository includes our pretrained models with the demonstration.

**October 24, 2024**
Our paper is under review.

## Installation
- To setup the environment
```
# For Python2.x
python setup.py

# For Python3.x
python3 setup.py
```

## Training Process
![alt text for screen readers](Images/Overall_Process.png  "Training Process").

## Model Architecture
Our model architecture includes an additional feature from the contrastive learning with the inductive features from the end-to-end model.
![alt text for screen readers](Images/CTN2Nmodel.png  "End-to-end classification with contrastive feature architecture").

## Performance
![alt text for screen readers](Images/Performance_table.png  "Performance comparison on test set").

<!-- 
### Contrastive learning 
This learning method learns to extract the contrast between an input and represent them as a contrastive feature map which are learnt through the similarity classification.
![alt text for screen readers](Images/Screenshot_CT.png "Contrastive learning architecture").
### End-to-end classification
The end-to-end model takes a pair of sky images and make a prediction whether a moving object is exists.
![alt text for screen readers](Images/Screenshot_n2n.png  "End-to-end classification architecture").

## Implementation
We provide two approaches of implmentations. First, we provides test.py and test.ipynb to show how our model make a prediction on 64*64 patches. Moreover, for full_img_test.py, we demonstrate how to use our model for .fits images. The latter approach includes both patch extractor and classifier heads to complete the whole process.
![alt text for screen readers](Images/Screenshot_sliding_windows.png  "Sliding windows for patch extraction").
-->

## Acknowledgement
We would like to thank to National Astronomical Research Institute of Thailand(NARIT) and The Gravitational-wave Optical Transient Observer (GOTO) as the data provider. Moreover, Researchers from the institutes also supported the comparable results from Astrometrica.
