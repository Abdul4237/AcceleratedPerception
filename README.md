# BreastCancerDiagnosis
Cancer Segmentation and Prediction using Seg-Net Architecture inspired from the following research paper:

https://www.mdpi.com/2072-6694/14/16/4030?__cf_chl_tk=HDfgZg7b.79q4s6YG0yBCEfBoKd0VXUb4dLQiFrGnLU-1687956882-0-gaNycGzNDqU

Breast Cancer in South Asia and specifically Pakistan is one of the most fatal diseases to affect women with an average of 1/9 women developing it in their lifetime and a devastating misdiagnosis rate of more than 50%. As such, it is of utmost importance to improve the misdiagnosis rate to ensure that the lives of countless women are saved which can be done through training an AI program on data of Pakistani origin. Our program is(as of now) trained on the CBIS-DDSM dataset of digital mammograms and performs semantic segmentation on 256x256 ROIs extracted from digital mammograms to outline the extent to which a tumor is present. This is followed by a classification of the mass as Malignant/Benign using VGG-19 inspired architectures.   
