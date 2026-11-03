\# RoadGuard AI — Model Evaluation Report



\## 1. Objective



The objective of this evaluation is to measure the pothole detection performance of the selected YOLOv8s model using a prepared test dataset.



\## 2. Evaluation Setup



| Parameter | Value |

|---|---|

| Model weights | `models/pothole\_best.pt` |

| Model architecture | YOLOv8s |

| Evaluation framework | Ultralytics 8.4.176 |

| PyTorch version | 2.14.1+cpu |

| Device | CPU |

| Image size | 640 × 640 |

| Evaluation images | 133 |

| Ground-truth pothole instances | 299 |

| Classes | 1 — pothole |



\## 3. Evaluation Metrics



| Metric | Result |

|---|---:|

| Precision | 92.2% |

| Recall | 86.3% |

| F1-score (approximate) | 89.2% |

| mAP@50 | 95.1% |

| mAP@50–95 | 70.7% |



The approximate F1-score is calculated from the rounded precision and recall values:



`F1 = 2 × Precision × Recall / (Precision + Recall)`



\## 4. Interpretation



\- \*\*Precision (92.2%):\*\* A high proportion of the model's pothole detections were correct.

\- \*\*Recall (86.3%):\*\* The model detected a substantial proportion of the actual potholes in the evaluation set.

\- \*\*mAP@50 (95.1%):\*\* The model achieved high average precision at an intersection-over-union threshold of 0.50.

\- \*\*mAP@50–95 (70.7%):\*\* Performance was lower across stricter overlap thresholds, indicating room to improve bounding-box localization.



\## 5. Evaluation Output Files



The evaluation run generated the following artifacts in `runs/detect/val-2/`:



\- `BoxF1\_curve.png`

\- `BoxPR\_curve.png`

\- `BoxP\_curve.png`

\- `BoxR\_curve.png`

\- `confusion\_matrix.png`

\- `confusion\_matrix\_normalized.png`

\- `val\_batch0\_labels.jpg` and `val\_batch0\_pred.jpg`

\- `val\_batch1\_labels.jpg` and `val\_batch1\_pred.jpg`

\- `val\_batch2\_labels.jpg` and `val\_batch2\_pred.jpg`



The standalone evaluation did not generate `results.csv` or `results.png`. These files are not included in the current evaluation artifacts.



\## 6. Inference Speed



The reported average inference time was approximately 406.9 ms per image on the CPU.



This is the model inference time reported by the evaluation run, not the complete end-to-end application response time.



\## 7. Known Limitations and Evaluation Scope



\- Evaluation was performed on 133 images from the supplied test split.

\- The dataset contains one detection class: pothole.

\- The current evaluation does not establish performance on every road type, weather condition, lighting condition, or camera angle.

\- Performance in night-time scenes, on unpaved roads, and around puddles requires additional testing.

\- The evaluation was performed on CPU hardware; performance may differ on other devices.

\- The original training dataset and its relationship to this test dataset have not yet been independently verified.



\## 8. Conclusion



The selected YOLOv8s pothole detection model achieved 92.2% precision, 86.3% recall, 95.1% mAP@50, and 70.7% mAP@50–95 on the current 133-image evaluation set.



These results indicate promising pothole detection performance on this dataset. Further testing with varied road and environmental conditions is required before making broader reliability claims.



