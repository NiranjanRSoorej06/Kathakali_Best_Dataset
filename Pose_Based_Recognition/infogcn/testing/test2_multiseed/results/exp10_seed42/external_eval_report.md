# EXP10RERUN Seed 42 External New40 Evaluation Report (`exp10_seed42`)

## Execution & Environment
- **Device**: `cuda:0` (`NVIDIA H100 PCIe MIG 2g.20gb`)
- **CUDA Version**: `12.1`
- **Run Name**: `exp10_seed42` (seed `42`)
- **Source Manifest**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40/processed/manifest.csv`
- **Checkpoint**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/runs/exp10rerun/exp10_seed42/best_model.pt`
- **Checkpoint Epoch**: `58` (lowest canonical val loss: `0.058067`)
- **Model Parameters**: `324,080` (trainable: `324,080`)
- **Evaluated Videos**: `40` / `40` (skipped/invalid: `0`)

## Summary Metrics (New40 External)
- **Overall Accuracy**: `0.6000` (`60.00%`, `24/40`)
- **Macro F1 (all 41 canonical classes)**: `0.5098`
- **Macro F1 (40 represented classes)**: `0.5225`
- **Weighted F1**: `0.5225`
- **Macro Precision (41 / represented)**: `0.4817` / `0.4938`
- **Macro Recall (41 / represented)**: `0.5854` / `0.6000`
- **Mean Cross-Entropy Loss**: `1.5684`

## Class Coverage
- **Represented Classes**: `40`
- **Missing Classes (1)**: Urakkam

## Per-Class Precision, Recall, and F1
| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Aakasham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Aana | 1 | 0.0000 | 0.0000 | 0.0000 |
| Aayudham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Acchan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Albutham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Amma | 1 | 1.0000 | 1.0000 | 1.0000 |
| Anugraham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Ariyikkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Baalan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Bhoomi | 1 | 0.0000 | 0.0000 | 0.0000 |
| Bhrahmaavu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Chandran | 1 | 0.3333 | 1.0000 | 0.5000 |
| Chundu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Gunam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Irikkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kaadu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kallan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kannu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kashtam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Kuyil | 1 | 1.0000 | 1.0000 | 1.0000 |
| Manassu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Nadakkuka | 1 | 1.0000 | 1.0000 | 1.0000 |
| Nee_Roudram | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Shringaaram | 1 | 0.5000 | 1.0000 | 0.6667 |
| Paadam | 1 | 0.3333 | 1.0000 | 0.5000 |
| Pazham | 1 | 1.0000 | 1.0000 | 1.0000 |
| Pushpam | 1 | 0.3333 | 1.0000 | 0.5000 |
| Raajyam | 1 | 1.0000 | 1.0000 | 1.0000 |
| Samudram | 1 | 1.0000 | 1.0000 | 1.0000 |
| Satyam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Shivan | 1 | 0.2500 | 1.0000 | 0.4000 |
| Simham | 1 | 0.0000 | 0.0000 | 0.0000 |
| Sooryan | 1 | 0.0000 | 0.0000 | 0.0000 |
| Svargam | 1 | 0.5000 | 1.0000 | 0.6667 |
| Thaamara | 1 | 0.0000 | 0.0000 | 0.0000 |
| Thalamudi | 1 | 1.0000 | 1.0000 | 1.0000 |
| Thaliru | 1 | 0.0000 | 0.0000 | 0.0000 |
| Urakkam | 0 | 0.0000 | 0.0000 | 0.0000 |
| Veedu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Vishnu | 1 | 0.0000 | 0.0000 | 0.0000 |
| Vriksham | 1 | 0.0000 | 0.0000 | 0.0000 |

## Incorrect Predictions (16)
| Video | True Class | Predicted Class | Confidence | True Class Prob |
|---|---|---|---:|---:|
| Aakasham_4_5.mp4 | Aakasham | Svargam | 0.5391 | 0.0024 |
| Aana_5_1.mp4 | Aana | Paadam | 0.6258 | 0.1842 |
| Aayudham_2_4.mp4 | Aayudham | Sooryan | 0.7947 | 0.1291 |
| Acchan_3_2.mp4 | Acchan | Satyam | 0.9710 | 0.0140 |
| Anugraham_4_4.mp4 | Anugraham | Veedu | 0.9349 | 0.0006 |
| Bhoomi_3_2.mp4 | Bhoomi | Chandran | 0.5195 | 0.0299 |
| Gunam_3_5.mp4 | Gunam | Chandran | 0.7431 | 0.0012 |
| Kashtam_4_5.mp4 | Kashtam | Pushpam | 0.9621 | 0.0019 |
| Nee_Roudram_2_3.mp4 | Nee_Roudram | Nee_Shringaaram | 0.8327 | 0.1640 |
| Simham_3_3.mp4 | Simham | Shivan | 0.5661 | 0.0567 |
| Sooryan_2_3.mp4 | Sooryan | Paadam | 0.4030 | 0.0035 |
| Thaamara_4_3.mp4 | Thaamara | Shivan | 0.3840 | 0.1483 |
| Thaliru_2_3.mp4 | Thaliru | Pushpam | 0.4957 | 0.4102 |
| Veedu_3_1.mp4 | Veedu | Thaliru | 0.4133 | 0.3223 |
| Vishnu_4_3.mp4 | Vishnu | Shivan | 0.6070 | 0.0226 |
| Vriksham_5_4.mp4 | Vriksham | Gunam | 0.3598 | 0.3095 |

## Correct Predictions (24)
| Video | Class | Confidence |
|---|---|---:|
| Albutham_2_4.mp4 | Albutham | 0.9098 |
| Amma_3_2.mp4 | Amma | 0.3023 |
| Ariyikkuka_5_3.mp4 | Ariyikkuka | 0.9992 |
| Baalan_2_4.mp4 | Baalan | 0.6258 |
| Bhrahmaavu_4_2.mp4 | Bhrahmaavu | 0.9974 |
| Chandran_5_4.mp4 | Chandran | 0.7107 |
| Chundu_2_1.mp4 | Chundu | 0.4456 |
| Irikkuka_2_5.mp4 | Irikkuka | 0.9778 |
| Kaadu_3_1.mp4 | Kaadu | 0.9887 |
| Kallan_2_2.mp4 | Kallan | 0.9998 |
| Kannu_3_1.mp4 | Kannu | 0.9853 |
| Kuyil_5_1.mp4 | Kuyil | 0.9322 |
| Manassu_4_4.mp4 | Manassu | 0.9953 |
| Nadakkuka_5_2.mp4 | Nadakkuka | 0.3607 |
| Nee_Shringaaram_3_3.mp4 | Nee_Shringaaram | 0.9961 |
| Paadam_4_3.mp4 | Paadam | 0.5261 |
| Pazham_3_2.mp4 | Pazham | 0.9546 |
| Pushpam_2_4.mp4 | Pushpam | 0.7511 |
| Raajyam_5_5.mp4 | Raajyam | 0.9985 |
| Samudram_2_3.mp4 | Samudram | 0.9937 |
| Satyam_3_3.mp4 | Satyam | 0.6451 |
| Shivan_2_4.mp4 | Shivan | 0.9826 |
| Svargam_3_2.mp4 | Svargam | 0.9613 |
| Thalamudi_3_4.mp4 | Thalamudi | 0.9887 |
