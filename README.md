
---

## Setting up the environment

Launch an EC2 Deep learning instance - "Deep Learning Base AMI (Amazon Linux) Version 4.0 - ami-3de63140" on AWS with 32 GB of RAM. 

SSH into the instance using the key-pair used to create the EC2 instance - ssh -i your_keypair.pem ec2-user@ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com

###  Steps to set up the environment
1. sudo yum -y update
2. sudo yum -y install yum-utils
3. sudo yum -y groupinstall development
4. sudo yum install python36
5. python --version
8. pip --version
9. ssh-keygen
10. cat ~/.ssh/id_rsa.pub
11. git clone git@bitbucket.org:saikrishna7ak/final_project.git
12. cd final_project/
13. pip install -r requirements.txt


##### open python environment to install nltk words, stopwords and punkt. Run following code in python terminal.

	import nltk
	nltk.download("words")
	nltk.download("stopwords")
	nltk.download("punkt")
	nltk.download("vader_lexicon")


##### Once the environment is set up you can download a twitter specific pretrained word2vec model from the web. The file is supposed to be saved at location final_project/twitter/data/. cd into the path and run below command

	wget http://www-nlp.stanford.edu/data/glove.twitter.27B.zip


##### unzip the file contents using unzip. There are 4 different files with different sizes. We will keep the file glove.twitter.27B.200d.txt and delete remaining files. 

	unzip glove.twitter.27B.zip

##### Add following line of text as first line in the file "glove.twitter.27B.200d.txt". It tells word2vec how many rows and columns are there in the model. You can add the line using 'sed' as follows - "sed  -i '1i 1193514 200' glove.twitter.27B.200d.txt"
	
	1193514 200


##### You are all set to run the application

###### cd into either final_project/rssfeed_analysis or final_project/twitter and run

	python main.py


---

## Clone a repository

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You�ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you�d like to and then click **Clone**.
4. Open the directory you just created to see your repository�s files.

Now that you're more familiar with your Bitbucket repository, go ahead and add a new file locally. You can [push your change back to Bitbucket with SourceTree](https://confluence.atlassian.com/x/iqyBMg), or you can [add, commit,](https://confluence.atlassian.com/x/8QhODQ) and [push from the command line](https://confluence.atlassian.com/x/NQ0zDQ).