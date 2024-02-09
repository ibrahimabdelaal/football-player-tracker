 ## Football Player Tracking using ByteTrack 
**Dual Tracking** 
* ByteTrack: 
Responsible for tracking individual soccer players. 
Offers a balance between accuracy and speed. 
* ByteTrack_Reid:
Specifically designed for ball tracking, enhancing capabilities to follow the ball effectively.
Leveraging Strengths
ByteTrack provides efficient player tracking while maintaining accuracy.
ByteTrack_Reid improves ball tracking precision through feature extraction using res_net.
Enhanced Performance
By integrating both trackers, we aim to achieve comprehensive and accurate tracking of players and the ball throughout the match.

**Fine-Tuning Process**
* Matching Threshold: Optimized to identify the best threshold for accurately matching detections across frames.
* Confidence Threshold: Adjusted to ensure confidence in the tracking results, minimizing false positives.
* Kalman Filter: Fine-tuned to predict player and ball movements, improving tracking accuracy.
Feature Extractor
* ByteTrack_Reid: Employs ResNet as the feature extractor, although it hasn't been fine-tuned specifically for this task.
  For improved performance, consider fine-tuning ResNet on a dataset dedicated to soccer player and ball tracking.
## Bird's Eye View
We provide a bird's eye view of the field, visually displaying player and ball movements for better understanding of the game dynamics.

## Result Visualization
https://github.com/ibrahimabdelaal/football-player-tracker/assets/49596777/bb23d41c-f7d1-43d9-b51d-1d46cebc9984

## Usage
* downlod yolov5 [weights] (https://drive.google.com/file/d/1-7UUm0XAZhVwzBHHL-zvh7WHGzYXN8df/view?usp=sharing)
  and Bird eye view  [weights] (https://drive.google.com/file/d/1-5wsJH4mnOGrcJ6exoSC3y3zPC8L94lS/view?usp=sharing)
*Create Conda Environment: First, create a new Conda environment using the provided YAML file.
    conda env create -f environment.yml
*Activate Environment: Activate the newly created environment.
    conda activate football-tracking
*Verify Installation: Verify that all dependencies have been installed correctly.
    conda list


