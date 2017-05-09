# NSFW_Detection
A NSFW detecting neural network mainly based on the description by yahoo "https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for". Also provided is a script for downloading training/testing data from Imgur and a quick demo script of hosting the network on the https://21.co/ platform.  
Before the network can be trained, training and testing data has to be aquired which can be done using the provided Imgur Script. Run this script several times by calling it with different Imgur Subreddits e.g.  
    python crawl_subr.py sports  
