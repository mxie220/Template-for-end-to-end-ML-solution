# Template to build an end-to-end machine learning solution

#### [2020 Facebook Developer Circles Community Challenge](https://developercircles2020.devpost.com/)
#### Team: Michelle Xie, Victor Livingstone, Sirani McNeill, Adarsh Vulli
___________________________________________________________________________________________________________________________________________________________________________________

## How to use this guide: 

The tutorial is split into 3 sections, the frontend tier with React, the middle tier with Django, the backend tier with Pytorch. Each section takes you through a simple tutorial on how to leverage open-source technologies to build an end-to-end machine learning solution. For the purposes of this tutorial, we will show you how to build an application that can classify texts but the aim of this tutorial is to act as a template for any machine learning solutions, models, and packages that can be easily consumed by applications. 

## Part 1: React App

To create and start your React App, do the following steps below:

### npx create-react-app my-app
### cd my-app
### npm start

## About React JS more in detail
 JavaScript build toolchain typically consists of:
- A package manager, such as Yarn or npm. It lets you take advantage of a vast ecosystem of third-party packages, and easily install or update them.
- A bundler, such as webpack or Parcel. It lets you write modular code and bundle it together into small packages to optimize load time.
- A compiler such as Babel. It lets you write modern JavaScript code that still works in older browsers.

**NOTE:** If you've previously installed create-react-app globally via npm install -g create-react-app, we recommend you uninstall the package using npm uninstall -g create-react-app or yarn global remove create-react-app to ensure that npx always uses the latest version.


## Part 2: Django APIs
...

## Part 3: PyTorch Model
Install the required dependences for the pytorch! whether it may be for local or through the colab! and swe can use the text classifier using pytorch
___________________________________________________________________________________________________________________________________________________________________________________

**NOTE** Ensure you have python interpreter in your system. visit https://www.python.org/downloads/

### clone the project

create a virtual environment run this command pip install virtualenv in your termital, make sure your in the project directory SVM_API run this command virtual env this will create a folder called env activate the virtual enviroment by running this env\scripts\activate

python pakages in your virtual enviroment run pip install django djangorestframework

navigate to the svm folder by running cd svm

then run python manage.py runserver
