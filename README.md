# AWS-DeepRacer-Student-Notes

## What is Machine Learning (ML)?

Machine learning (ML) is a modern software development technique, and a type of artificial intelligence (AI), that enables computers to solve problems by using examples of real-world data. It allows computers to automatically learn and improve from experience without being explicitly programmed to do so.

Relationship between Artificial Intelligence and Machine Learning
Artificial Intelligence (AI) is the broad capability of machines to perform activities using human-like intelligence
Machine Learning (ML) is a type of AI that allows computers to automatically learn and improve from experience without being explicilty programmed to do so

ML has other subs such as 
### 1. Supervised Learning:
   - is a type of ML learning where samples from a dataset has a corresponding label or output value; used to predict cell types or identify objects in an image
### 2. Unsupervised Learning:
   - there are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data. 
### 3. Reinforced Leaning:
  - the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal. This is a completely different approach than supervised and unsupervised learning

=======================
Nearly all tasks solved with machine learning involve three primary components:

1. A machine learning model
2. A model training algorithm
3. A model inference algorithm

1. A machine learning model - a machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided. *A model is an extremely generic program (or block of code), made specific by the data used to train it. It is used to solve different problems.*

2. A model training algorithm - they work through an interactive process where the current model iteration is analyzed to determine what changes can be made to get closer to the goal. Those changes are made and the iteration continues until the model is evaluated to meet the goals.

3. A model inference algorithm - is used to generate predictions using a trained model.

=======================

Major steps in Machine Learning Process
1. Define the problem
2. Build the dataset
3. Train the model
4. Evaluate the model
5. Use the model

---------

Here’s a quick recap of the terms introduced in this lesson:

Clustering is an unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
A categorical label has a discrete set of possible values, such as "is a cat" and "is not a cat."
A continuous (regression) label does not have a discrete set of possible values, which means there are potentially an unlimited number of possibilities.
Discrete is a term taken from statistics referring to an outcome that takes only a finite number of values (such as days of the week).
A label refers to data that already contains the solution.
Using unlabeled data means you don't need to provide the model with any kind of label or solution while the model is being trained.

---------

The first step in model training is to randomly split the dataset.

Splitting your dataset gives you two sets of data:

Training dataset: The data on which the model will be trained. Most of your data will be here. Many developers estimate about 80%.
Test dataset: The data withheld from the model during training, which is used to test how well your model will generalize to new data.

The model training algorithm iteratively updates a model's parameters to minimize some loss function.

===================

Model parameters: Model parameters are settings or configurations that the training algorithm can update to change how the model behaves. Depending on the context, you’ll also hear other specific terms used to describe model parameters such as weights and biases. Weights, which are values that change as the model learns, are more specific to neural networks.

Loss function: A loss function is used to codify the model’s distance from a goal. For example, if you were trying to predict the number of snow cone sales based on the day’s weather, you would care about making predictions that are as accurate as possible. So you might define a loss function to be “the average distance between your model’s predicted number of snow cone sales and the correct number.” 

===================

Extended Learning texts (copied as is):

This information wasn't covered in the video from the previous section, but it is provided for the advanced reader.

Linear models

One of the most common models covered in introductory coursework, linear models simply describe the relationship between a set of input numbers and a set of output numbers through a linear function (think of y = mx + b or a line on a x vs y chart). Classification tasks often use a strongly related logistic model, which adds an additional transformation mapping the output of the linear function to the range [0, 1], interpreted as “probability of being in the target class.” Linear models are fast to train and give you a great baseline against which to compare more complex models. A lot of media buzz is given to more complex models, but for most new problems, consider starting with a simple model.

Tree-based models

Tree-based models are probably the second most common model type covered in introductory coursework. They learn to categorize or regress by building an extremely large structure of nested if/else blocks, splitting the world into different regions at each if/else block. Training determines exactly where these splits happen and what value is assigned at each leaf region. For example, if you’re trying to determine if a light sensor is in sunlight or shadow, you might train tree of depth 1 with the final learned configuration being something like if (sensor_value > 0.698), then return 1; else return 0;. The tree-based model XGBoost is commonly used as an off-the-shelf implementation for this kind of model and includes enhancements beyond what is discussed here. Try tree-based models to quickly get a baseline before moving on to more complex models.

Deep learning models

Extremely popular and powerful, deep learning is a modern approach that is based around a conceptual model of how the human brain functions. The model (also called a neural network) is composed of collections of neurons (very simple computational units) connected together by weights (mathematical representations of how much information thst is allowed to flow from one neuron to the next). The process of training involves finding values for each weight. Various neural network structures have been determined for modeling different kinds of problems or processing different kinds of data.
A short (but not complete!) list of noteworthy examples includes:

FFNN: The most straightforward way of structuring a neural network, the Feed Forward Neural Network (FFNN) structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the previous layer.
CNN: Convolutional Neural Networks (CNN) represent nested filters over grid-organized data. They are by far the most commonly used type of model when processing images.
RNN/LSTM: Recurrent Neural Networks (RNN) and the related Long Short-Term Memory (LSTM) model types are structured to effectively represent for loops in traditional computing, collecting state while iterating over some object. They can be used for processing sequences of data.
Transformer: A more modern replacement for RNN/LSTMs, the transformer architecture enables training over larger datasets involving sequences of data.
Machine learning using Python libraries

For more classical models (linear, tree-based) as well as a set of common ML-related tools, take a look at scikit-learn. The web documentation for this library is also organized for those getting familiar with space and can be a great place to get familiar with some extremely useful tools and techniques.
For deep learning, mxnet, tensorflow, and pytorch are the three most common libraries. For the purposes of the majority of machine learning needs, each of these is feature-paired and equivalent.

=========================

Reinforcement Learning

Reinforcement learning consists of several key concepts:

Agent is the entity being trained. In our example, this is a dog.
Environment is the “world” in which the agent interacts, such as a park.
Actions are performed by the agent in the environment, such as running around, sitting, or playing ball.
Rewards are issued to the agent for performing good actions.

AWS DeepRacer offers two training algorithms:

Proximal Policy Optimization (PPO)
Soft Actor Critic (SAC)

------
- Deterministic policy
- Stochastic policy

A policy defines the action that the agent should take for a given state. This could conceptually be represented as a table - given a particular state, perform this action.

This is called a deterministic policy, where there is a direct relationship between state and action. This is often used when the agent has a full understanding of the environment and, given a state, always performs the same action.

Consider the classic game of rock, paper, scissors. An example of a deterministic policy is always playing rock. Eventually the other players are going to realize that you are always playing rock and then adapt their strategy to win, most likely by always playing paper. So in this situation it’s not optimal to use a deterministic policy.

So, we can alternatively use a stochastic policy. In a stochastic policy you have a range of possible actions for a state, each with a probability of being selected. When the policy is queried to return an action for a state it selects one of these actions based on the probability distribution.

This would obviously be a much better policy option for our rock, paper, scissors game as our opponents will no longer know exactly which action we will choose each time we play.


-----

Value function
- This is used to determine possible future rewards from being in a certain state

Policy update
- This is generally the process of selecting the preferred actions, by querying the policy given a certain state

--------------------

Difference between PPO and SAC Policy

PPO uses “on-policy” learning. This means it learns only from observations made by the current policy exploring the environment - using the most recent and relevant data. Say you are learning to drive a car, on-policy learning would be analogous to you reviewing a video of your most recent lesson and taking note of what you did well, and what needs improvement.

In contrast, SAC uses “off-policy” learning. This means it can use observations made from previous policies exploration of the environment - so it can also use old data. Going back to our learning to drive analogy, this would involve reviewing videos of your driving lessons from the last few weeks. 

--------------------

Machine Learning Inference
- this refers to how a device uses the trained model loaded onto it to make decisions. It involves running input data through a machine learning model which then outputs its prediction

We can measure the performance of inference with two metrics:

The "inference rate" which is the number of inferences which can be done per second; and
The "inference time" or "inference latency" which is time taken to run a single inference.

---------------

we want to maximize the rate of inference and therefore the number of decisions made every second. However, the rate of inference is dependent on many factors. The most obvious is the performance of the machine running the inference, such as the CPU (Central Processing Unit) speed, whether any GPU (Graphical Processing Unit) acceleration is present, and the amount of system RAM (Random Access Memory) available.

There are also other factors. One is the machine learning framework used for training. There are many different frameworks available, such as TensorFlow, PyTorch, and Apache MXNet. These frameworks provide the tools and services to build machine learning models, including the training algorithms such as PPO (Proximal Policy Optimization) and SAC (Soft Actor Critic)



