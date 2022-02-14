# 1 Installation
***

## 1.1 Building the stable package
In the early stages, the stable package isn't bug free - tracking issues and fixing bugs will still. The most recent stable build is 0.0.2-stable.

To install:
```
$ pip3 install stable_builds/flares-segmentation-tlincke125-0.0.2-stable.tar.gz
```

## 1.2 Building the latest package 
First, checkout the **master** branch. Then, in the root directory, build the package and documentation using the following command:

```
$ ./install.sh
``` 

This should produce two files in the ./dist folder and update all files in docs/documentation/flares.

To install the package, go into the dist folder and install the package using pip:

```
$ cd dist
$ pip3 install flares-segmentation-tlincke125-<version>.tar.gz
```

Where \<version\> is the highest version (or any other version you wish to install).

You can then open docs/documentation/flares/index.html in your prefered browser.

# 2 Getting Started
***
See ./docs/guides/getting_started.ipynb
