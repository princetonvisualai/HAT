## Percentage of Synthetic Pixels

We plot the accuracy of [SlowFast](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/slowfast_r152_r50_4x16x1_256e_kinetics400_rgb.py) by the percentage of the synthetic pixels each video has. 

|   |  |  |
| ------------- | ------------- | ------------- |
| ![hoa](hoa.png)   | ![hoa](boa.png)   | ![hoa](shacc.png)  |

## Percentage of Human Mask

Here we show some of the sameple frames per human mask percentage (±2%).  
To assure that the Action Swap Videos have enough information for both background and foreground, we only select frames that have 5%-50% of human coverage.

| 0% | 10% | 20% | 30% | 40% | 50% |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ![10](000.png) | ![10](010.png) | ![10](020.png) | ![10](030.png) | ![10](040.png) | ![10](050.png) |


| 60% | 70% | 80% | 90% | 100% |
| :---: | :---: | :---: | :---: | :---: |
| ![10](060.png) | ![10](070.png) | ![10](080.png) | ![10](090.png) | ![10](100.png) |