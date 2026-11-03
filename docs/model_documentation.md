\# RoadGuard AI — Model Documentation



\## 1. Selected Model



The model used for the current pothole detection evaluation is:



\- \*\*Weights file:\*\* `models/pothole\_best.pt`

\- \*\*Architecture:\*\* YOLOv8s

\- \*\*Task:\*\* Object detection

\- \*\*Base pretrained weights:\*\* `yolov8s.pt`

\- \*\*Model size:\*\* Approximately 22.5 MB

\- \*\*Number of parameters:\*\* 11,125,971

\- \*\*Classes:\*\* Pothole (single-class detection)



\## 2. Model Source and Licence



\- \*\*Model repository:\*\* \[Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8)

\- \*\*Checkpoint date:\*\* 23 June 2026

\- \*\*Licence field recorded in checkpoint:\*\* AGPL-3.0

\- \*\*Original Git repository metadata:\*\* Not available in the checkpoint.



The checkpoint's embedded licence field is recorded here as metadata. The exact redistribution terms of the downloaded model weights should be confirmed with the model repository before redistribution.



\## 3. Training Configuration



The checkpoint records the following training configuration:



| Parameter | Value |

|---|---|

| Base model | `yolov8s.pt` |

| Configured epochs | 60 |

| Recorded training history | 60 epochs |

| Image size | 640 × 640 |

| Batch size | 16 |

| Optimizer | Auto |

| Early stopping patience | 15 |

| Pretrained | True |

| Training task | Detection |

| Checkpoint Ultralytics version | 8.4.75 |



The checkpoint's `epoch` field is -1. However, its `train\_results` contains training records for epochs 1 through 60.



\## 4. Evaluation Environment



| Component | Version / Setting |

|---|---|

| Ultralytics | 8.4.176 |

| PyTorch | 2.14.1+cpu |

| Device | CPU |

| Evaluation image size | 640 × 640 |



\## 5. Model Evaluation



The selected weights were evaluated on the prepared test split containing 133 images and 299 ground-truth pothole instances.



| Metric | Result |

|---|---:|

| Precision | 92.2% |

| Recall | 86.3% |

| F1-score (calculated from rounded precision and recall) | 89.2% |

| mAP@50 | 95.1% |

| mAP@50–95 | 70.7% |



These are the results of the current evaluation run, not the original training metrics stored in the checkpoint.



\## 6. Training Provenance



The checkpoint records the base model, training configuration and training history. Its Git origin metadata is empty, so the original training repository and trainer identity could not be established from the checkpoint alone.



The team should retain the original model download information and verify the model repository's licence before redistribution.



\## 7. Model Selection



`pothole\_best.pt` is the selected model for the current pothole detection evaluation. The additional `best.pt` file is not included in this evaluation.



