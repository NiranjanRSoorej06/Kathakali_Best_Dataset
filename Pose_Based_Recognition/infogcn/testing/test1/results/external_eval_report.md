# EXP10 InfoGCN External Evaluation Report (`testing/test1`)

## Execution & Environment
- **Device**: `cuda:0` (`NVIDIA H100 PCIe MIG 2g.20gb`)
- **CUDA Version**: `12.1`
- **Source Manifest**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40/processed/manifest.csv`
- **Checkpoint**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/runs/exp10/exp10_infogcn_baseline/best_model.pt`
- **Checkpoint Epoch**: `28` (selected via lowest canonical validation loss: `0.148193`)
- **Model Parameters**: `324,080` (trainable: `324,080`)
- **Evaluated Videos**: `40` / `40`
- **Skipped / Invalid Videos**: `0`

## Summary Metrics
- **Overall Accuracy**: `0.5750` (`57.50%`, `23/40`)
- **Macro F1 (all 41 canonical classes)**: `0.5122`
- **Macro F1 (40 represented classes)**: `0.5250`
- **Weighted F1**: `0.5250`
- **Macro Precision (41 / represented)**: `0.4927` / `0.5050`
- **Macro Recall (41 / represented)**: `0.5610` / `0.5750`
- **Mean Cross-Entropy Loss**: `1.2925`

## Class Coverage
- **Represented Classes (40)**: Aakasham, Aana, Aayudham, Acchan, Albutham, Amma, Anugraham, Ariyikkuka, Baalan, Bhoomi, Bhrahmaavu, Chandran, Chundu, Gunam, Irikkuka, Kaadu, Kallan, Kannu, Kashtam, Kuyil, Manassu, Nadakkuka, Nee_Roudram, Nee_Shringaaram, Paadam, Pazham, Pushpam, Raajyam, Samudram, Satyam, Shivan, Simham, Sooryan, Svargam, Thaamara, Thalamudi, Thaliru, Veedu, Vishnu, Vriksham
- **Missing Classes (1)**: Urakkam

## Per-Class Precision, Recall, and F1
| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Aakasham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Aana | 1 | 1.0000 | 1.0000 | 1.0000 |
| Aayudham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Acchan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Albutham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Amma | 1 | 1.0000 | 1.0000 | 1.0000 |
| Anugraham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Ariyikkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Baalan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Bhoomi | 1 | 0.0000 | 0.0000 | 0.0000 |
| Bhrahmaavu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Chandran | 1 | 0.0000 | 0.0000 | 0.0000 |
| Chundu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Gunam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Irikkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kaadu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kallan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kannu | 1 | 0.5000 | 1.0000 | 0.6667 |
| Kashtam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Kuyil | 1 | 0.0000 | 0.0000 | 0.0000 |
| Manassu | 1 | 0.5000 | 1.0000 | 0.6667 |
| Nadakkuka | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Roudram | 1 | 1.0000 | 1.0000 | 1.0000 |
| Nee_Shringaaram | 1 | 1.0000 | 1.0000 | 1.0000 |
| Paadam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Pazham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Pushpam | 1 | 0.2000 | 1.0000 | 0.3333 |
| Raajyam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Samudram | 1 | 1.0000 | 1.0000 | 1.0000 |
| Satyam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Shivan | 1 | 0.5000 | 1.0000 | 0.6667 |
| Simham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Sooryan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Svargam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thaamara | 1 | 0.5000 | 1.0000 | 0.6667 |
| Thalamudi | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thaliru | 1 | 0.0000 | 0.0000 | 0.0000 |
| Urakkam | 0 | 0.0000 | 0.0000 | 0.0000 |
| Veedu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Vishnu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Vriksham | 1 | 0.0000 | 0.0000 | 0.0000 |

## Incorrect Predictions (17)
| Video | True Class | Predicted Class | Confidence | True Class Prob |
|---|---|---|---:|---:|
| Aakasham_4_5.mp4 | Aakasham | Pushpam | 0.7221 | 0.0078 |
| Acchan_3_2.mp4 | Acchan | Satyam | 0.7988 | 0.1366 |
| Anugraham_4_4.mp4 | Anugraham | Veedu | 0.6973 | 0.0082 |
| Bhoomi_3_2.mp4 | Bhoomi | Gunam | 0.2627 | 0.0740 |
| Chandran_5_4.mp4 | Chandran | Gunam | 0.4861 | 0.3697 |
| Chundu_2_1.mp4 | Chundu | Kannu | 0.3917 | 0.1752 |
| Gunam_3_5.mp4 | Gunam | Sooryan | 0.2739 | 0.0126 |
| Kashtam_4_5.mp4 | Kashtam | Pushpam | 0.6951 | 0.2290 |
| Kuyil_5_1.mp4 | Kuyil | Thaamara | 0.4656 | 0.1786 |
| Nadakkuka_5_2.mp4 | Nadakkuka | Manassu | 0.5275 | 0.0325 |
| Satyam_3_3.mp4 | Satyam | Acchan | 0.3187 | 0.3147 |
| Simham_3_3.mp4 | Simham | Pushpam | 0.4328 | 0.2181 |
| Sooryan_2_3.mp4 | Sooryan | Chandran | 0.5312 | 0.0050 |
| Thaliru_2_3.mp4 | Thaliru | Pushpam | 0.8041 | 0.0147 |
| Veedu_3_1.mp4 | Veedu | Vriksham | 0.3390 | 0.1875 |
| Vishnu_4_3.mp4 | Vishnu | Shivan | 0.6432 | 0.0589 |
| Vriksham_5_4.mp4 | Vriksham | Gunam | 0.4762 | 0.0854 |

## Correct Predictions (23)
| Video | Class | Confidence |
|---|---|---:|
| Aana_5_1.mp4 | Aana | 0.9796 |
| Aayudham_2_4.mp4 | Aayudham | 0.8963 |
| Albutham_2_4.mp4 | Albutham | 0.5018 |
| Amma_3_2.mp4 | Amma | 0.4754 |
| Ariyikkuka_5_3.mp4 | Ariyikkuka | 0.9975 |
| Baalan_2_4.mp4 | Baalan | 0.6374 |
| Bhrahmaavu_4_2.mp4 | Bhrahmaavu | 0.6035 |
| Irikkuka_2_5.mp4 | Irikkuka | 0.8514 |
| Kaadu_3_1.mp4 | Kaadu | 0.9763 |
| Kallan_2_2.mp4 | Kallan | 0.9965 |
| Kannu_3_1.mp4 | Kannu | 0.9423 |
| Manassu_4_4.mp4 | Manassu | 0.9964 |
| Nee_Roudram_2_3.mp4 | Nee_Roudram | 0.5784 |
| Nee_Shringaaram_3_3.mp4 | Nee_Shringaaram | 0.9889 |
| Paadam_4_3.mp4 | Paadam | 0.7179 |
| Pazham_3_2.mp4 | Pazham | 0.9414 |
| Pushpam_2_4.mp4 | Pushpam | 0.9015 |
| Raajyam_5_5.mp4 | Raajyam | 0.9928 |
| Samudram_2_3.mp4 | Samudram | 0.9472 |
| Shivan_2_4.mp4 | Shivan | 0.9787 |
| Svargam_3_2.mp4 | Svargam | 0.9186 |
| Thaamara_4_3.mp4 | Thaamara | 0.5791 |
| Thalamudi_3_4.mp4 | Thalamudi | 0.9898 |
