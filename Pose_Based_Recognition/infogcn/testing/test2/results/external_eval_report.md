# EXP10RERUN Seed 21 External Evaluation Report (`testing/test2`)

## Execution & Environment
- **Device**: `cuda:0` (`NVIDIA H100 PCIe MIG 2g.20gb`)
- **CUDA Version**: `12.1`
- **Run Name**: `exp10rerun_seed21` (seed `21`)
- **Source Manifest**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40/processed/manifest.csv`
- **Checkpoint**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/runs/exp10rerun/exp10rerun_seed21/best_model.pt`
- **Checkpoint Epoch**: `71` (selected via lowest canonical validation loss: `0.076947`)
- **Model Parameters**: `324,080` (trainable: `324,080`)
- **Evaluated Videos**: `40` / `40`
- **Skipped / Invalid Videos**: `0`

## Summary Metrics
- **Overall Accuracy**: `0.7000` (`70.00%`, `28/40`)
- **Macro F1 (all 41 canonical classes)**: `0.6016`
- **Macro F1 (40 represented classes)**: `0.6167`
- **Weighted F1**: `0.6167`
- **Macro Precision (41 / represented)**: `0.5610` / `0.5750`
- **Macro Recall (41 / represented)**: `0.6829` / `0.7000`
- **Mean Cross-Entropy Loss**: `1.3684`

## Class Coverage
- **Represented Classes (40)**: Aakasham, Aana, Aayudham, Acchan, Albutham, Amma, Anugraham, Ariyikkuka, Baalan, Bhoomi, Bhrahmaavu, Chandran, Chundu, Gunam, Irikkuka, Kaadu, Kallan, Kannu, Kashtam, Kuyil, Manassu, Nadakkuka, Nee_Roudram, Nee_Shringaaram, Paadam, Pazham, Pushpam, Raajyam, Samudram, Satyam, Shivan, Simham, Sooryan, Svargam, Thaamara, Thalamudi, Thaliru, Veedu, Vishnu, Vriksham
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
| Chandran | 1 | 0.5000 | 1.0000 | 0.6667 |
| Chundu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Gunam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Irikkuka | 1 | 0.5000 | 1.0000 | 0.6667 |
| Kaadu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kallan | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kannu | 1 | 1.0000 | 1.0000 | 1.0000 |
| Kashtam | 1 | 0.0000 | 0.0000 | 0.0000 |
| Kuyil | 1 | 1.0000 | 1.0000 | 1.0000 |
| Manassu | 1 | 0.5000 | 1.0000 | 0.6667 |
| Nadakkuka | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Roudram | 1 | 0.0000 | 0.0000 | 0.0000 |
| Nee_Shringaaram | 1 | 0.5000 | 1.0000 | 0.6667 |
| Paadam | 1 | 1.0000 | 1.0000 | 1.0000 |
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

## Incorrect Predictions (12)
| Video | True Class | Predicted Class | Confidence | True Class Prob |
|---|---|---|---:|---:|
| Aakasham_4_5.mp4 | Aakasham | Svargam | 0.9553 | 0.0040 |
| Acchan_3_2.mp4 | Acchan | Satyam | 0.8898 | 0.0959 |
| Anugraham_4_4.mp4 | Anugraham | Veedu | 0.6005 | 0.0024 |
| Baalan_2_4.mp4 | Baalan | Irikkuka | 0.6869 | 0.2751 |
| Gunam_3_5.mp4 | Gunam | Chandran | 0.4920 | 0.0250 |
| Kashtam_4_5.mp4 | Kashtam | Aana | 0.2998 | 0.0217 |
| Nadakkuka_5_2.mp4 | Nadakkuka | Manassu | 0.7643 | 0.0739 |
| Nee_Roudram_2_3.mp4 | Nee_Roudram | Nee_Shringaaram | 0.9641 | 0.0013 |
| Sooryan_2_3.mp4 | Sooryan | Gunam | 0.6296 | 0.0001 |
| Thaliru_2_3.mp4 | Thaliru | Pushpam | 0.9070 | 0.0621 |
| Vishnu_4_3.mp4 | Vishnu | Shivan | 0.2842 | 0.0294 |
| Vriksham_5_4.mp4 | Vriksham | Gunam | 0.2091 | 0.1376 |

## Correct Predictions (28)
| Video | Class | Confidence |
|---|---|---:|
| Aana_5_1.mp4 | Aana | 0.9992 |
| Aayudham_2_4.mp4 | Aayudham | 0.9639 |
| Albutham_2_4.mp4 | Albutham | 0.9150 |
| Amma_3_2.mp4 | Amma | 0.8678 |
| Ariyikkuka_5_3.mp4 | Ariyikkuka | 0.9979 |
| Bhoomi_3_2.mp4 | Bhoomi | 0.3013 |
| Bhrahmaavu_4_2.mp4 | Bhrahmaavu | 0.9877 |
| Chandran_5_4.mp4 | Chandran | 0.5661 |
| Chundu_2_1.mp4 | Chundu | 0.6040 |
| Irikkuka_2_5.mp4 | Irikkuka | 0.8089 |
| Kaadu_3_1.mp4 | Kaadu | 0.9982 |
| Kallan_2_2.mp4 | Kallan | 0.9999 |
| Kannu_3_1.mp4 | Kannu | 0.9978 |
| Kuyil_5_1.mp4 | Kuyil | 0.7039 |
| Manassu_4_4.mp4 | Manassu | 0.9941 |
| Nee_Shringaaram_3_3.mp4 | Nee_Shringaaram | 0.8722 |
| Paadam_4_3.mp4 | Paadam | 0.6004 |
| Pazham_3_2.mp4 | Pazham | 0.9942 |
| Pushpam_2_4.mp4 | Pushpam | 0.9882 |
| Raajyam_5_5.mp4 | Raajyam | 0.9922 |
| Samudram_2_3.mp4 | Samudram | 0.9832 |
| Satyam_3_3.mp4 | Satyam | 0.6872 |
| Shivan_2_4.mp4 | Shivan | 0.6445 |
| Simham_3_3.mp4 | Simham | 0.9819 |
| Svargam_3_2.mp4 | Svargam | 0.9978 |
| Thaamara_4_3.mp4 | Thaamara | 0.9202 |
| Thalamudi_3_4.mp4 | Thalamudi | 0.9934 |
| Veedu_3_1.mp4 | Veedu | 0.5581 |
