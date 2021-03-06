# pyApi



# Description
Currently:
 - Using Allen NLP to analyze sentences stored in elasticsearch, converting the sentences into triples (subject, predicate, object) and storing them in a graph database.

Working Towards:
- Converting this into an API where it will store sentences as tripples in a graph database


Details

Need elasticsearch
Need a graph database
Need allennlp which uses python version 3.6 or 3.7


# SetUp
Set Up VS Code for python
[video](https://www.youtube.com/watch?v=W--_EOzdTHk)

Get Anaconda working on VS Code

install Anaconda

Link anaconda to VS Code through 
[here](https://code.visualstudio.com/docs/python/environments)

Set up pipenv [here](https://www.youtube.com/watch?v=ArDT5NsROMk)


Set up Python and Flask
[Video](https://www.youtube.com/watch?v=iwhuPxJ0dig&list=PLei96ZX_m9sU3dLM2I5LKcqXeqth-GTe6)

Thanks @Chris Hawkes


# Windows python setup
- install python
 
- set up python path

- run cmd in root directory
```sh
$python get-pip.py
```
- then add pip to your enviorment variables
by adding 

 `<pathwhere python is>`\Python\Python37\Scripts




For using GPU for allennlp predictor you need to install pytorch with CUDA
- conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
- https://github.com/pytorch/pytorch/issues/30664


might need to reinstall anaconda 
or might need to remove all python versions and reinstall python
https://stackoverflow.com/questions/14627047/windowserror-error-127-the-specified-procedure-could-not-be-found