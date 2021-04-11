# 3202

MUN Computer Science Intro to Machine Learning

Marty Whelan - martyw@mun.ca

Elliott Barnes - enbarnes@mun.ca

## Task Lists for A1

**Specifications**

- [x] program named A1_KNN.py
- Command Line Arguments
  - [x] Filename for `.txt` value
  - [x] `int` indicating the number of closest neighbours k to consider
  - [x] `int` indicating the number of instances in the data to be set apart as unknown instances

**Functionality**

- [x] Read command line args
  - [x] U < # of observations
- [x] read the input data, assume in working dir
- [ ] randomly select U-instances to be the unknown instances in UnInstance
  - **slide 5, lec 2**
- [ ] remove the unknown instances in UnInstance from the input data
  - remaining instances are the training data
- [ ] perform KNN Algorithm for classification to predict the class of the unknown instances in UnInstance
  - [ ] Use a distance metric suitable for categorical attibutes
- [ ] Calc proportion of correctly classified unknown instances
  - the # of unknown instances for which the actual class is equal to the predicted class divided by U.
- [ ] print to screen (terminal output?) the value of **k followed by a tab and then the proportion of correctly classified unknown instances.**

**Submission Requirements**

- [ ] single `.py` file called `A1_KNN.py`
- [ ] `.pdf` file containing...
  - [ ] description and justification of the distance metric you used.
  - [ ] explanation of the effect of `k` in the performance of your algorithm and recommendation of what value of k to use.
  - [ ] description of how you tested your implementation
  - [ ] acknowledgements section
  - [ ] program specification section listing Python version
  - [ ] program specificatino section listing libraries

**Bonus Marks**

- [ ] PNG with a plot showing the accuracy of your program as a function of the number of neighbours considered
