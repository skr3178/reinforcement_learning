REINFORCE formula pushes the parameters of the policy
in the direction of the better action (multiplied proportionally by the size of the
estimated action value) to know which action is best.


In the Reinforce algorithm, we use a model based system (without learning the exact env physics).

Key elements of the Reinforce algo:

1. Policy
  Fundamentally a simply neural network that takes in the input state values and outputs the actions in      the form of logits
  This NN is trained with a loss function that improves the prediction of the Policy over time.
 
2. Action selection
   Actions are selected as output from the Policy network--> converted to Categorical form--> sampled-->      converted into a log form. This log is appended to the policy.history to be later used for loss            function optimisation.
   
3. Batch update
   For a batch of episodes (run within the function), rewards are accumulated/collected for future       
   rewards. 
   
4. Policy update
   Loss and therefore the policy update happens when a batch_update function is called to collect all the     batch_rewards. The derivatation is for discounted rewards i.e. immediate rewards are given more            weightage

   
![Discounted_reward](https://miro.medium.com/v2/resize:fit:366/format:webp/1*P2W3I2gwbFphCvBAqqkLLg.png)
   
![Reward_update](https://miro.medium.com/v2/resize:fit:638/format:webp/1*VFRng5GHkOzNrx8wG2BlqA.png)

![Training_Screenshot](Reinforce_training.png)

Notes:
Empirically it was noted that with Normalization of rewards yield better/faster and more stable training than without.

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
