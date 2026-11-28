# EXP10RERUN Seed 13 External New40 Evaluation Report (`exp10_seed13`)

## Execution & Environment
- **Device**: `cuda:0` (`NVIDIA H100 PCIe MIG 2g.20gb`)
- **CUDA Version**: `12.1`
- **Run Name**: `exp10_seed13` (seed `13`)
- **Source Manifest**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40/processed/manifest.csv`
- **Checkpoint**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/runs/exp10rerun/exp10_seed13/best_model.pt`
- **Checkpoint Epoch**: `47` (lowest canonical val loss: `0.055219`)
- **Model Parameters**: `324,080` (trainable: `324,080`)
- **Evaluated Videos**: `40` / `40` (skipped/invalid: `0`)

## Summary Metrics (New40 External)
- **Overall Accuracy**: `0.7250` (`72.50%`, `29/40`)
- **Macro F1 (all 41 canonical classes)**: `0.6341`
- **Macro F1 (40 represented classes)**: `0.6500`
- **Weighted F1**: `0.6500`
- **Macro Precision (41 / represented)**: `0.5976` / `0.6125`
- **Macro Recall (41 / represented)**: `0.7073` / `0.7250`
- **Mean Cross-Entropy Loss**: `1.1431`

## Class Coverage
- **Represented Classes**: `40`
- **Missing Classes (1)**: Urakkam

## Per-Class Precision, Recall, and F1
| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Aakasham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Aana | 1 | 0.5000 | 1.0000 | 0.6667 |
| Aayudham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Acchan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Albutham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Amma | 1 | 1.0000 | 1.0000 | 1.0000 |
| Anugraham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Ariyikkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Baalan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Bhoomi | 1 | 1.0000 | 1.0000 | 1.0000 |
| Bhrahmaavu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Chandran | 1 | 1.0000 | 1.0000 | 1.0000 |
| Chundu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Gunam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Irikkuka | 1 | 0.5000 | 1.0000 | 0.6667 |
| Kaadu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kallan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kannu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kashtam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Kuyil | 1 | 1.0000 | 1.0000 | 1.0000 |
| Manassu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Nadakkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Nee_Roudram | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Shringaaram | 1 | 0.5000 | 1.0000 | 0.6667 |
| Paadam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Pazham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Pushpam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Raajyam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Samudram | 1 | 1.0000 | 1.0000 | 1.0000 |
| Satyam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Shivan | 1 | 0.5000 | 1.0000 | 0.6667 |
| Simham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Sooryan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Svargam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Thaamara | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thalamudi | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thaliru | 1 | 0.0000 | 0.0000 | 0.0000 |
| Urakkam | 0 | 0.0000 | 0.0000 | 0.0000 |
| Veedu | 1 | 0.5000 | 1.0000 | 0.6667 |
| Vishnu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Vriksham | 1 | 0.0000 | 0.0000 | 0.0000 |

## Incorrect Predictions (11)
| Video | True Class | Predicted Class | Confidence | True Class Prob |
|---|---|---|---:|---:|
| Aakasham_4_5.mp4 | Aakasham | Svargam | 0.8906 | 0.0285 |
| Acchan_3_2.mp4 | Acchan | Satyam | 0.8793 | 0.0960 |
| Anugraham_4_4.mp4 | Anugraham | Veedu | 0.8180 | 0.0071 |
| Baalan_2_4.mp4 | Baalan | Irikkuka | 0.8945 | 0.1008 |
| Gunam_3_5.mp4 | Gunam | Sooryan | 0.3787 | 0.0321 |
| Kashtam_4_5.mp4 | Kashtam | Aana | 0.3111 | 0.0338 |
| Nee_Roudram_2_3.mp4 | Nee_Roudram | Nee_Shringaaram | 0.9769 | 0.0131 |
| Sooryan_2_3.mp4 | Sooryan | Paadam | 0.5995 | 0.0026 |
| Thaliru_2_3.mp4 | Thaliru | Pushpam | 0.6739 | 0.2411 |
| Vishnu_4_3.mp4 | Vishnu | Shivan | 0.8824 | 0.0078 |
| Vriksham_5_4.mp4 | Vriksham | Gunam | 0.3004 | 0.0705 |

## Correct Predictions (29)
| Video | Class | Confidence |
|---|---|---:|
| Aana_5_1.mp4 | Aana | 0.9875 |
| Aayudham_2_4.mp4 | Aayudham | 0.9335 |
| Albutham_2_4.mp4 | Albutham | 0.8971 |
| Amma_3_2.mp4 | Amma | 0.4685 |
| Ariyikkuka_5_3.mp4 | Ariyikkuka | 0.9319 |
| Bhoomi_3_2.mp4 | Bhoomi | 0.4436 |
| Bhrahmaavu_4_2.mp4 | Bhrahmaavu | 0.9290 |
| Chandran_5_4.mp4 | Chandran | 0.9270 |
| Chundu_2_1.mp4 | Chundu | 0.9603 |
| Irikkuka_2_5.mp4 | Irikkuka | 0.9655 |
| Kaadu_3_1.mp4 | Kaadu | 0.9768 |
| Kallan_2_2.mp4 | Kallan | 0.9991 |
| Kannu_3_1.mp4 | Kannu | 0.9904 |
| Kuyil_5_1.mp4 | Kuyil | 0.6435 |
| Manassu_4_4.mp4 | Manassu | 0.9960 |
| Nadakkuka_5_2.mp4 | Nadakkuka | 0.1966 |
| Nee_Shringaaram_3_3.mp4 | Nee_Shringaaram | 0.9875 |
| Paadam_4_3.mp4 | Paadam | 0.8395 |
| Pazham_3_2.mp4 | Pazham | 0.9316 |
| Pushpam_2_4.mp4 | Pushpam | 0.7322 |
| Raajyam_5_5.mp4 | Raajyam | 0.9909 |
| Samudram_2_3.mp4 | Samudram | 0.9493 |
| Satyam_3_3.mp4 | Satyam | 0.4154 |
| Shivan_2_4.mp4 | Shivan | 0.9671 |
| Simham_3_3.mp4 | Simham | 0.9360 |
| Svargam_3_2.mp4 | Svargam | 0.9967 |
| Thaamara_4_3.mp4 | Thaamara | 0.6799 |
| Thalamudi_3_4.mp4 | Thalamudi | 0.9922 |
| Veedu_3_1.mp4 | Veedu | 0.6893 |
