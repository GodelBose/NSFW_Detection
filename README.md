# NSFW_Detection
A NSFW detecting neural network mainly based on the description by yahoo "https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for". Also provided is a script for downloading training/testing data from Imgur and a quick demo script of hosting the network on the https://21.co/ platform.  
Before the network can be trained, training and testing data has to be aquired which can be done using the provided Imgur Script. Run this script several times by calling it with different Imgur Subreddits e.g.  
```bash
python crawl_subr.py sports  
```
The script should create two directories "nsfw" and "normal". Once they contain enough data put them into the directory called "train" and then repeat the crawling process with different subreddits to gather testing/validation data which is then to be put into the "test" directory.  
The model is then trained by running the "train_detector" script and providing it with a directory name where accuracy on the testing/validation data is logged and a model is saved for each epoch the training is running.  
So for instance run it by typing the following line in the console.
```bash
python train_detector.py detector_test_1  
```  
The network should be getting 95+ percent accuracy on an evenly distributed validation set without much problems.  

The 21 application should be working once you installed all the needed software. For a detailed explanation to get up and running with it checkout "https://21.co/features/"
