 # AcceleratedPerception- Deep Learning model for Breast Cancer Diagnosis(BCD)
Cancer Segmentation and Classification using Connected Seg-Net Architecture inspired by the following research paper:

https://www.mdpi.com/2072-6694/14/16/4030?__cf_chl_tk=HDfgZg7b.79q4s6YG0yBCEfBoKd0VXUb4dLQiFrGnLU-1687956882-0-gaNycGzNDqU

Breast Cancer in South Asian countries such as Pakistan ranks among the most fatal diseases to affect women; with an average of 1/9 women developing it in their lifetime and a devastating misdiagnosis rate of more than 50%, it is of utmost importance to improve the misdiagnosis rate to save countless lives. Additionally, around 30% of cancer patients are diagnosed at the malignant stage, which has made detection at the benign stage crucial in the fight against Breast Cancer.These problems are accentuated by the incredible burden on the Pakistani healthcare system combined with the meagre income of the average patient which does not allow for biopsies and further examinations. Therefore, it is imperative to extract as many features and deductions as we can from mammograms alone in the Pakistani healthcare system. 

As a solution to the above problems, AcceleratedPerception utilizes the power of Deep Learning and Convolutional Neural Networks to detect and segment lesions present in mammographic images along with their classification as malignant/benign. 3 datasets have been used in the training of this program, including CBIS-DDSM and In Breast along with a private dataset consisting of mammograms of Pakistani origin. This has been implemented keeping in mind the problems of domain generalization and limited data in state-of-the-art Computer Assisted Diagnosis(CAD) algorithms.

The databases used to train the model can be found here:

CBIS-DDDSM: Available a https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

INBreast: Available at https://www.kaggle.com/martholi/inbreast?select=inbreast.tgz

Our algorithm utilizes the state of the art YOLOv3 algorithm for object detection and fine tunes it for automatic detection and classfication of lesions within digital mammograms. We have trained the model on mammograms with ROI data as well as mammograms with labels only to allow our classification and detection system to generalize well. It also allows us to expand our dataset since it is very rare to find mammograms with annotated ROI's. To further improve the accuracy of the algorithm, we will be experimenting with YOLOv4 architecture in the future as well.
The second part of our model uses the predictions made by our YOLOv3 algorithm to crop 256x256 sized ROI's from the mammograms and  perform semantic segmentation on them using Connected Seg-Net architecture to highlight lesion boundaries and size. 
