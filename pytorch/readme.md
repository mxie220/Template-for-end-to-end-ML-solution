# Building a text classifier with PyTorch 

###### Source: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

## What is PyTorch?

[PyTorch](https://pytorch.org/) is a framework that contains lots of modules to support machine learning and deep learning workloads. You can use PyTorch to train neural networks to [classify images](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [do object detection](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) all the way through to [classfiying text](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html) (which is the focus of this tutorial) and many, many more. It is well-integrated into Python (that programming language that has 'snaked' its way through to becoming one of the most popular languages used in data science and machine learning) and is designed to train models on GPU and CPU machines. 

## How do I get started?
You need to install PyTorch of course! But before you can do that, you need to make sure you have the following installed: 
> [Python](https://www.python.org/downloads/) (preferrably Python 3.x where x is a variable so just replace x with a number). For this tutorial, we are using Python 3.8.

> [Anaconda](https://docs.continuum.io/anaconda/) (not to be confused with Nicki Minaj's song, Anaconda is a package manager, environment manager and a Python data science distribution. When you install Anaconda, you get an Anaconda prompt and the Anaconda Navigator which allows you to install a LOT of Anaconda packages for free.) Remember to download 64-Bit Graphical Installer and not the 32-Bit Graphical Installer because you might run into issues later. 

> [Jupyter](https://jupyter.org/install) Notebooks allows us to execute our text classifier code in cells. You can install Jupyter with conda or pip. You can think of both conda and pip as package managers for python however conda can do more. While pip allows you install Python packages from [PyPI](https://pypi.org/), conda is actually cross-platform which means conda can also be used to install packages from C, C++, and R libraries. 

## Getting set up 

Open up Jupyter Notebook (Anaconda 3). 
This will browse to your localhost and open up one of your file system's directories. Create a folder and navigate into that folder. From there, create a new Python 3 notebook like so: 

![Create Jupyter Notebook](images/create-jupyter-notebook.png)

Give your notebook some meaningful name like *'Text Classifier'*. 
In the first cell, execute the following code to check you are running the correct Python version. 

```python
# First we want to check what version of python we are running here: 
import sys
print("Python version: " + str(sys.version_info.major) + "." + str(sys.version_info.minor) + "." + str(sys.version_info.micro))
```

Once you have identified the version, add another cell below by clicking on the '+' sign from the toolbar at the top:

![Jupyter toolbar](images/jupyter-topbar.png)

This will add another cell for you to run a separate block of code. In this cell, we want to install PyTorch: 

```python 
# Install PyTorch and TorchText
!pip install torch
!pip install torchtext
```

We can use either conda or pip to install PyTorch, I'm using pip here but either one works. Run that cell and once it successfully executes, we want to import the modules we have just installed to our environment: 

```python 
import torch
import torchtext
print("Successfully imported Torch and TorchText module from PyTorch.")
```

Now we want to load some of the datasets from TorchText: 

```python 
# Import some datasets from TorchText
from torchtext.datasets import text_classification
print("Successfully imported some datasets from TorchText.")
```

TorchText has a couple of datasets but I'm just going to bring the AG_NEWS dataset in by running the following in another cell: 

```python 
# Creating a folder called 'data'
import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
    
# Specifying how to break up the dataset into n-words 
NGRAMS = 2

# Load the training dataset and testing dataset
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
```

## Building the model 

Once we have loaded our datasets, we want to start building out the model which we will then train our data on. We'll start by importing the neural network modules from PyTorch: 

```python 
# Import the neural network package from PyTorch 
import torch.nn as nn
```

We also need to import the functional module which contains all the functions in the torch.nn library:

```python 
# Import the functional module from torch.nn
import torch.nn.functional as F
```
Now we create a class called TextSentiment. 
Within this class, we want to initiate a few things like the embedding. An embedding is essentially these vectors you can use to represent items that share similarities with each other e.g. let's imagine a group of words and let's say these words are reprsented by A, B, C, and D. 

How do we determine if word A is closer to word B or word C or word D? 

We map its features into vectors that span over multiple dimensions (these are called tensors) we can use embeddings to find which words share the most similarities across all its dimensions. Anyway, an EmbeddingBag in PyTorch allows you to find the sum or the mean of all these embeddings. 

We also need to initalize a linear transformation which is a function that allows our model to learn the weights between the embeddings and map it to output classes or labels.

The init_weights method initalizes some random weights at the start (which we can then use the model to learn as we do our training).

The forward method is what's called forward propagation in Machine Learning. It takes some input data and feeds it forward into the next layer of the neural network. 
```python
# Creating a class called TextSentiment that takes in the neural network module from Pytorch.
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # The EmbeddingBag module finds the sum or the mean of all the embeddings.
        # An embedding allows you to map low-dimensional real vectors that can represent each words to other words that are similar. 
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # This does a Linear Transformation which allows the model to learn the weights between the embeddings and maps it to an output class
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        # intialization range set at 0.5
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    # This is the forward propagation which essentially feeds input data into each layer of the model 
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
```

The next section is just defining some variables which we will pass into our model as parameters:

```python
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUM_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS)

print("The vocabulary size is: ", VOCAB_SIZE)
print("The number of classes is: ", NUM_CLASS)
```

According to the PyTorch tutorial documentation, the generate_batch function is used to generate batches of data and offsets. Let's take a more in-depth look at what it actually does. 

The function takes in an argument called 'batch' and batch has several entries. Running through each entry in the batch which looks like some form of array that can be accessed by its index. The first part of every entry (at index = 0), 
```python
def generate_batch(batch):
    # A tensor is used to represent n-Dimensions of features. It looks like a matrix but it's not a matrix, a matrix is simply used to visualize a tensor. 
    # The first part of the entry is the class label (e.g. what type of news article).
    label = torch.tensor([entry[0] for entry in batch])
    # The second part of the entry is the text. 
    text = [entry[1] for entry in batch]
    # These are the offset values in storage which indicates where the tensors start from.
    offsets = [0] + [len(entry) for entry in text]
    
    # torch.Tensor.cumsum returns a cumulative sum of all the elements in the dimension.
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    
    # torch.cat concatenates whatever you give it together. 
    text = torch.cat(text)
    return text, offsets, label
```




```python
from torch.utils.data import DataLoader

#Dataloder is used to load the data and then send it to the model for traning and validation 
def train_func(sub_train_):

    # Train the model 
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    #  a function dataloder is called here with the parameters to be passed for thr validation of our model
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate  fir the number of epoches taken
    scheduler.step()
    
    return train_loss / len(sub_train_), train_acc / len(sub_train_)
    
#testing the model
def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)
```



```python

#Evaluating the data with the test dataset..
print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
#finally the locc and the accuracy are calculated
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

```


```python
#Evaluating the model on some random data 
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

#different labels taken into consideration
ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

#Use of the trained model and the bgrams for the classification
def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1
#example input text taken
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")
#final classification of the taken input text
print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])

```