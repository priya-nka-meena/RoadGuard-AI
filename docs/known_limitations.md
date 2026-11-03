\# RoadGuard AI — Known Limitations



\## 1. Dataset Limitations



\- The current evaluation uses 133 images from the supplied pothole test split.

\- The dataset contains only one detection class: `pothole`.

\- The dataset may not represent every type of road, location, weather condition, or camera angle.

\- A separate validation split was not provided in the supplied split information.



\## 2. Model Limitations



\- The model is designed for pothole detection and does not currently identify other road hazards such as cracks or damaged road markings.

\- Small, distant, partially hidden, or poorly illuminated potholes may be difficult to detect.

\- The model's performance may vary with image quality, camera angle, lighting, and road conditions.

\- The current evaluation measures detection performance on a prepared image dataset. It does not establish real-world reliability in every environment.



\## 3. Environmental Conditions Requiring Further Testing



The following conditions require additional testing:



\- Night-time and low-light road scenes.

\- Potholes containing water or covered by puddles.

\- Unpaved or uneven roads.

\- Rain, fog, shadows, and glare.

\- Potholes partially blocked by vehicles or other objects.



These are potential areas of weakness to investigate, not confirmed failures from the current evaluation.



\## 4. Hardware and Performance Limitations



\- The current evaluation was performed on CPU hardware.

\- Average model inference time was approximately 406.9 ms per image in the reported evaluation run.

\- Performance may differ on systems with different processors or GPU acceleration.

\- Real-time performance must be evaluated separately using the complete application and video pipeline.



\## 5. Future Improvements



\- Collect and evaluate more road images from different locations and environmental conditions.

\- Include additional road-damage classes if the project scope is expanded.

\- Test the model on night-time, rainy, and challenging road scenes.

\- Evaluate performance on additional hardware platforms.

\- Investigate model optimization for faster inference.

\- Perform further testing before making broad real-world reliability claims.



\## 6. Conclusion



The current model demonstrates promising pothole detection performance on the prepared evaluation dataset. However, additional data collection and testing under varied environmental conditions are necessary to understand its limitations and improve its reliability.



