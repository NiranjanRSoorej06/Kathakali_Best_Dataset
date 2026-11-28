# EXP10RERUN Seed 7 External New40 Evaluation Report (`exp10_seed7`)

## Execution & Environment
- **Device**: `cuda:0` (`NVIDIA H100 PCIe MIG 2g.20gb`)
- **CUDA Version**: `12.1`
- **Run Name**: `exp10_seed7` (seed `7`)
- **Source Manifest**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40/processed/manifest.csv`
- **Checkpoint**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/runs/exp10rerun/exp10_seed7/best_model.pt`
- **Checkpoint Epoch**: `43` (lowest canonical val loss: `0.100816`)
- **Model Parameters**: `324,080` (trainable: `324,080`)
- **Evaluated Videos**: `40` / `40` (skipped/invalid: `0`)

## Summary Metrics (New40 External)
- **Overall Accuracy**: `0.6500` (`65.00%`, `26/40`)
- **Macro F1 (all 41 canonical classes)**: `0.5691`
- **Macro F1 (40 represented classes)**: `0.5833`
- **Weighted F1**: `0.5833`
- **Macro Precision (41 / represented)**: `0.5407` / `0.5542`
- **Macro Recall (41 / represented)**: `0.6341` / `0.6500`
- **Mean Cross-Entropy Loss**: `1.1559`

## Class Coverage
- **Represented Classes**: `40`
- **Missing Classes (1)**: Urakkam

## Per-Class Precision, Recall, and F1
| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Aakasham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Aana | 1 | 1.0000 | 1.0000 | 1.0000 |
| Aayudham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Acchan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Albutham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Amma | 1 | 1.0000 | 1.0000 | 1.0000 |
| Anugraham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Ariyikkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Baalan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Bhoomi | 1 | 0.0000 | 0.0000 | 0.0000 |
| Bhrahmaavu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Chandran | 1 | 0.3333 | 1.0000 | 0.5000 |
| Chundu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Gunam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Irikkuka | 1 | 0.5000 | 1.0000 | 0.6667 |
| Kaadu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kallan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kannu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kashtam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Kuyil | 1 | 1.0000 | 1.0000 | 1.0000 |
| Manassu | 1 | 0.5000 | 1.0000 | 0.6667 |
| Nadakkuka | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Roudram | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Shringaaram | 1 | 0.3333 | 1.0000 | 0.5000 |
| Paadam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Pazham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Pushpam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Raajyam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Samudram | 1 | 0.5000 | 1.0000 | 0.6667 |
| Satyam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Shivan | 1 | 0.5000 | 1.0000 | 0.6667 |
| Simham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Sooryan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Svargam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Thaamara | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thalamudi | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thaliru | 1 | 0.0000 | 0.0000 | 0.0000 |
| Urakkam | 0 | 0.0000 | 0.0000 | 0.0000 |
| Veedu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Vishnu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Vriksham | 1 | 1.0000 | 1.0000 | 1.0000 |

## Incorrect Predictions (14)
| Video | True Class | Predicted Class | Confidence | True Class Prob |
|---|---|---|---:|---:|
| Aakasham_4_5.mp4 | Aakasham | Svargam | 0.6446 | 0.0128 |
| Aayudham_2_4.mp4 | Aayudham | Sooryan | 0.4834 | 0.2637 |
| Acchan_3_2.mp4 | Acchan | Satyam | 0.7359 | 0.2146 |
| Anugraham_4_4.mp4 | Anugraham | Vishnu | 0.5199 | 0.0186 |
| Baalan_2_4.mp4 | Baalan | Irikkuka | 0.5647 | 0.4207 |
| Bhoomi_3_2.mp4 | Bhoomi | Chandran | 0.3391 | 0.0638 |
| Kashtam_4_5.mp4 | Kashtam | Samudram | 0.3285 | 0.0740 |
| Nadakkuka_5_2.mp4 | Nadakkuka | Manassu | 0.2266 | 0.0918 |
| Nee_Roudram_2_3.mp4 | Nee_Roudram | Nee_Shringaaram | 0.9784 | 0.0023 |
| Pushpam_2_4.mp4 | Pushpam | Nee_Shringaaram | 0.5140 | 0.4063 |
| Satyam_3_3.mp4 | Satyam | Acchan | 0.2861 | 0.2172 |
| Sooryan_2_3.mp4 | Sooryan | Chandran | 0.5304 | 0.0004 |
| Thaliru_2_3.mp4 | Thaliru | Pushpam | 0.5919 | 0.2356 |
| Vishnu_4_3.mp4 | Vishnu | Shivan | 0.7038 | 0.0146 |

## Correct Predictions (26)
| Video | Class | Confidence |
|---|---|---:|
| Aana_5_1.mp4 | Aana | 0.9442 |
| Albutham_2_4.mp4 | Albutham | 0.9564 |
| Amma_3_2.mp4 | Amma | 0.6320 |
| Ariyikkuka_5_3.mp4 | Ariyikkuka | 0.9818 |
| Bhrahmaavu_4_2.mp4 | Bhrahmaavu | 0.9629 |
| Chandran_5_4.mp4 | Chandran | 0.8426 |
| Chundu_2_1.mp4 | Chundu | 0.5064 |
| Gunam_3_5.mp4 | Gunam | 0.6841 |
| Irikkuka_2_5.mp4 | Irikkuka | 0.9423 |
| Kaadu_3_1.mp4 | Kaadu | 0.9946 |
| Kallan_2_2.mp4 | Kallan | 0.9995 |
| Kannu_3_1.mp4 | Kannu | 0.9586 |
| Kuyil_5_1.mp4 | Kuyil | 0.8707 |
| Manassu_4_4.mp4 | Manassu | 0.9755 |
| Nee_Shringaaram_3_3.mp4 | Nee_Shringaaram | 0.9055 |
| Paadam_4_3.mp4 | Paadam | 0.7227 |
| Pazham_3_2.mp4 | Pazham | 0.8754 |
| Raajyam_5_5.mp4 | Raajyam | 0.9879 |
| Samudram_2_3.mp4 | Samudram | 0.9713 |
| Shivan_2_4.mp4 | Shivan | 0.9933 |
| Simham_3_3.mp4 | Simham | 0.9848 |
| Svargam_3_2.mp4 | Svargam | 0.9764 |
| Thaamara_4_3.mp4 | Thaamara | 0.7268 |
| Thalamudi_3_4.mp4 | Thalamudi | 0.9993 |
| Veedu_3_1.mp4 | Veedu | 0.5783 |
| Vriksham_5_4.mp4 | Vriksham | 0.4665 |
