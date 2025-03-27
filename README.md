# Project 1: Orientation Tracking
* Orientatin tracking:
  Estimate the orientation of rotating camera through IMU measurements, including linear velocitys and angular velocity.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/0a5ce50f-e931-4699-80cc-faf399f2e523" />
  </p>
* Panorama:
  Use the orientation estimated by previous step to stitch the images captured by camera to a augmented picture.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/48ea8f78-403a-4223-b6f3-e9f2f59aba07" />
  </p>

# Project 2: LiDAR Based SLAM
* Encoder and IMU odometry:
  Reconstruct the rudimentary odometry of robot through encoder and IMU measurements.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/d6d9cd60-cbc7-4e2a-98b1-f2e582afc581" />
  </p>
  
* Point-cloud registration via iterative closest point (ICP):
  Applying iterative closest point (ICP) algorithm to match LiDAR scans. By aligning the LiDAR scans,
  we can estimate the rotation of the robots.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/e16e087a-b71b-46ac-a90f-64756b05fe5d" />
  </p>

* Occupancy and texture mapping:
  Using the LiDAR scans with robot position to reconstruct occupancy grid maps. Furthermore, transforming
  images captured by RGB-D camera to find the pixels belong to floor.
  <p align="center" width="100%">
    <img width="49%" src="https://github.com/user-attachments/assets/42bd92ed-8efa-4d91-9692-c033a65a2e8d">
    <img width="43%" src="https://github.com/user-attachments/assets/ab0a358e-9546-4ae2-8c3c-92eb697f04e9">
  </p>
  
# Project 3: Visual SLAM
* Landmark mapping via EKF update::
  Estimate the landmarks through sequence of images.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/f2b9c08b-80cd-4f29-acc6-ab5b78b060bf" />
  </p>
  
* Visual-inertial SLAM::
  Applying Extended Kalman filter to optimized robot odometry.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/7134c07f-0009-4b95-9e6c-d11fee44723f" />
  </p>
