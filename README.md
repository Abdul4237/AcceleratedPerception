# BreastCancerDiagnosis
Cancer Segmentation and Classification using Connected Seg-Net Architecture inspired by the following research paper:

https://www.mdpi.com/2072-6694/14/16/4030?__cf_chl_tk=HDfgZg7b.79q4s6YG0yBCEfBoKd0VXUb4dLQiFrGnLU-1687956882-0-gaNycGzNDqU

Breast Cancer in South Asian countries such as Pakistan ranks among the most fatal diseases to affect women; with an average of 1/9 women developing it in their lifetime and a devastating misdiagnosis rate of more than 50%, it is of utmost importance to improve the misdiagnosis rate to save countless lives. Around 30% of cancer patients are diagnosed at the malignant stage, which has made detection at the benign stage crucial in the fight against Breast Cancer.

Our project utilizes the power of Deep Learning and Convolutional Neural Networks to segment lesions present in mammographic images followed by their classification as malignant/benign. 3 datasets have been used in the training of this program, including CBIS-DDSM and In Breast along with a private dataset consisting of mammograms of Pakistani origin. This has been implemented keeping in mind the problems of domain generalization and limited data in state-of-the-art Computer Assisted Diagnosis(CAD) algorithms.

The databases used to train the model can be found here:

CBIS-DDDSM: Available a https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

INBreast: Available at https://www.kaggle.com/martholi/inbreast?select=inbreast.tgz

Our algorithm takes in 256x256-sized image patches extracted from digital mammograms and  performs semantic segmentation on them using Connected Seg-Net architecture to highlight lesion boundaries and size. This is followed by a classification of the segmented mass as Malignant/Benign using VGG-19-inspired architectures in the second part.   
