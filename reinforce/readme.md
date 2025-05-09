REINFORCE formula pushes the parameters of the policy
in the direction of the better action (multiplied proportionally by the size of the
estimated action value) to know which action is best.


![Training_Screenshot](Reinforce_training.png)


| Episode | With bias=False |               | With bias=True |               |
|---------|-----------------|---------------|----------------|---------------|
|         | Last length     | Average length| Last length    | Average length|
|---------|-----------------|---------------|----------------|---------------|
| 0       | 29              | 10.19         | 8              | 9.98          |
| 50      | 95              | 26.00         | 9              | 9.68          |
| 100     | 499             | 69.84         | 36             | 20.27         |
| 150     | 256             | 165.41        | 101            | 53.99         |
| 200     | 114             | 223.36        | 41             | 49.02         |
| 250     | 499             | 272.73        | 112            | 71.04         |
| 300     | 499             | 360.64        | 438            | 123.43        |
| 350     | -               | -             | 499            | 263.39        |
| 400     | -               | -             | 499            | 319.13        |
| 450     | -               | -             | 95             | 355.69        |
