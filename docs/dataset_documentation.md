\# RoadGuard AI — Dataset Documentation



\## 1. Dataset Information



\- \*\*Dataset name:\*\* A Fully Annotated Image Dataset for Pothole Detection

\- \*\*Source:\*\* Kaggle

\- \*\*Dataset link:\*\* https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset

\- \*\*Downloaded archive:\*\* `archive (3).zip`

\- \*\*Extracted directory:\*\* `data/potholes\_raw/`

\- \*\*Annotation format:\*\* Pascal VOC XML

\- \*\*Detection class used:\*\* `pothole`



\## 2. Dataset Split



According to the dataset's `splits.json` file:



| Split | Number of images |

|---|---:|

| Training | 532 |

| Test | 133 |

| \*\*Total\*\* | \*\*665\*\* |



The supplied split information contains training and test lists. A separate validation split was not provided in this file.



\## 3. Dataset Preparation



The original dataset contains images and XML annotation files in the `annotated-images/` directory.



For the current evaluation:



1\. The test image list was read from `splits.json`.

2\. The corresponding XML annotations were parsed.

3\. Pothole bounding boxes were extracted.

4\. Pascal VOC bounding boxes were converted into normalized YOLO format.

5\. The 133 test images and their corresponding label files were copied into the `evaluation/` directory.



The prepared evaluation dataset contains:



\- 133 images

\- 133 YOLO-format label files

\- 299 pothole bounding-box instances

\- One detection class: `pothole`



\## 4. Dataset Directory Structure



```text

data/

├── Road.jpg

├── road\_video.mp4

└── potholes\_raw/

&#x20;   ├── annotated-images/

&#x20;   ├── README.md

&#x20;   └── splits.json



evaluation/

├── images/

├── labels/

├── data.yaml

└── prepare\_dataset.py

```



`Road.jpg` and `road\_video.mp4` are separate project demo inputs. They are not counted among the 133 evaluation images.



\## 5. Annotation Conversion



The preparation script is:



`evaluation/prepare\_dataset.py`



It converts Pascal VOC XML bounding boxes into YOLO detection labels.



Each YOLO label follows this structure:



```text

class\_id x\_center y\_center width height

```



The coordinates are normalized relative to the image width and height. The pothole class uses class ID `0`.



\## 6. Dataset Configuration



The evaluation configuration is stored in:



`evaluation/data.yaml`



The configuration identifies the evaluation images and the class name.



\*\*Important:\*\* This configuration is intended for the current evaluation procedure. It does not represent a separate training and validation split and should not be used to claim that the model was trained on these images.



\## 7. Dataset Licence and Additional Data



The dataset's exact licence and redistribution conditions must be confirmed from the Kaggle dataset page or its accompanying documentation before redistribution.



No claim is made here that the separate demo images were added to the training dataset.



\## 8. Dataset Preparation Results



The preparation script reported:



\- Test annotations in split: 133

\- Images and labels converted: 133

\- Missing images: 0

\- Total pothole bounding boxes: 299



