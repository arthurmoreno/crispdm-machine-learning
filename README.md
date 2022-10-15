# CRISP-MD Machine Learning

## Project definition:

- Execute all the steps using Python Scripts
- Build a report about all the decisions taken in all steps of the project. Explain the choices made.
- Make experiments with K-NN, LVQ, Decision Tree, SVM, Random Forest and an heterogen "committee".
    - It will be necessary to make searches in the configuration committee.
    - Utilize the cross validation with stratified k-fold with k=10
    - Show the average accuracy and the standard deviation, confusion matrix, boxplot. Analyse the results with a classification report, present the tests that had most statistical significance.
- Extract a decision tree and identify which are the first attributes utilized on the decision tree building.
    - Propose suggestions of the decisions based on the found attributes. (?)
- Deliver the final report that has all the decisions and discussion about each step of the CRISP-DM.

## Libraries decision:

https://www.simplilearn.com/scikit-learn-vs-tensorflow-article

"Scikit-learn’s true strength resides in its model assessment and selection architecture, which allows us to cross-validate and perform multiple hyperparameter searches on our models. Scikit-learn also helps us choose the best model for our work."

"The machine learning algorithm is also implemented using Scikit-learn, a higher-level library."

## Running the project:

### Install python and the requirements:

Install Python:
* [Windows](https://www.digitalocean.com/community/tutorials/install-python-windows-10)

Install the Pyhton package manager (PIP):
* [Windows](https://phoenixnap.com/kb/install-pip-windows)

[Create a virtualenv](https://docs.python.org/3/library/venv.html):
```
python3 -m venv .
```

Install the dependencies:
```
pip3 install -r requirements.txt
```

### Running the script:
```
python3 main.py
```

## Examples for testing with dataset.csv:

```
0,unknown,3,1,1,20060421120648,20161028084502,157,3844,97,117,2,0,0,10,4,7,5,130,0.997535302103113,994
0,unknown,3,1,0,20161027234947,20170911201328,165,320,23,15,0,0,0,5,0,31,27,101,1.29126262950745,974
0,unknown,3,3,1,20060110204256,20160108221641,1093,3651,298,378,64,0,0,34,22,122,9,898,0.661673179614133,477
1,male,3,2,0,20130107110046,20170915092339,590,1713,261,501,12,0,0,8,1,0,9,540,1.20440316450483,647
1,male,3,0,1,20150419181322,20160308133552,59,325,8,16,0,0,0,2,1,0,2,54,1.05714931610548,1074
1,male,3,3,3,20061201001006,20150821220942,4272,3186,286,1020,53,0,0,583,197,447,472,1721,1.48165413798567,1596
1,unknown,1,2,3,20061011041038,20160317170845,761,3446,121,67,19,0,0,48,1,5,4,697,0.831953524648453,917
1,unknown,1,1,3,20060427064625,20170803155631,202,4117,76,72,6,0,0,21,3,2,2,156,0.594368207936839,493
1,male,1,2,2,20130527062953,20170812234148,622,1539,128,107,4,0,0,73,0,10,3,525,0.915024129447388,978
0,unknown,3,1,0,20071209021632,20121022102507,195,1780,67,165,1,0,0,2,1,0,5,179,1.29126262950745,974
0,unknown,3,3,3,20060626215437,20170919213053,1048,4104,306,546,17,0,0,29,10,34,27,927,1.48165413798567,1596
1,male,3,3,3,20060626183311,20170930094718,18002,4115,2207,6729,135,0,0,319,117,120,865,11509,1.48165413798567,1596
1,male,1,3,1,20090425142743,20121129165121,1923,1315,195,82,30,0,0,3,1,2,13,1559,0.661673179614133,477
0,unknown,3,2,2,20110525182045,20120728152607,584,431,83,124,69,0,0,2,5,0,3,514,0.915024129447388,978
0,unknown,3,1,1,20061113122506,20150224132455,108,3026,37,94,0,0,0,1,6,4,4,76,0.997535302103113,994
1,male,3,2,3,20110115173128,20130501151035,277,838,83,70,24,0,0,45,0,6,0,207,0.831953524648453,917
1,unknown,1,1,1,20151031204155,20170709154025,143,618,34,45,1,0,0,8,0,0,4,131,0.997535302103113,994
1,unknown,1,0,3,20060827105658,20130506113219,72,2445,8,18,1,0,0,50,2,0,5,13,0.411985187306913,297
1,male,3,0,2,20160418225057,20170721190210,53,460,19,19,2,0,0,18,0,0,2,33,0.800528377424059,664
1,unknown,1,0,0,20050918184525,20161229145000,66,4121,32,30,1,0,0,4,0,4,0,57,1.86500803160946,1277
0,unknown,3,0,2,20080830145603,20170612005630,56,3209,36,26,1,0,0,8,3,0,4,39,0.800528377424059,664
0,unknown,3,3,2,20060309094510,20170720140109,5508,4152,1034,1162,124,0,0,34,1162,949,586,2550,1.00522694513289,817
0,unknown,3,0,2,20070206114958,20170614083345,87,3782,39,38,1,0,0,19,0,13,10,45,0.800528377424059,664
1,male,1,2,3,20121112160822,20170913164019,464,1767,129,116,1,0,0,136,9,24,17,258,0.831953524648453,917
1,male,1,3,1,20090601172308,20170917044950,2907,3031,864,1500,16,0,0,69,56,76,60,2485,0.661673179614133,477
1,male,3,3,3,20081228012822,20170930130637,30539,3199,1264,8704,16,0,0,148,3,73,184,29485,1.48165413798567,1596
2,female,2,0,3,20120114025941,20120517044032,68,125,8,37,2,0,0,3,1,1,6,57,0.411985187306913,297
0,unknown,3,0,1,20070816200615,20130731185925,79,2177,23,48,13,0,0,14,0,0,0,63,1.05714931610548,1074
1,male,3,3,1,20090511040004,20160108175052,1191,2434,156,89,6,0,0,4,0,71,50,1065,0.661673179614133,477
0,unknown,3,0,3,20111012103037,20170512141844,58,2040,18,12,0,0,0,40,0,0,3,15,0.411985187306913,297
1,male,3,3,3,20110114063813,20170930072552,17092,2452,1345,2674,1111,0,0,982,175,469,328,14682,1.48165413798567,1596
1,male,3,2,2,20120227100614,20170930073013,266,2043,155,135,0,0,0,6,0,2,3,250,0.915024129447388,978
0,unknown,3,3,1,20111108054659,20170906055641,1217,2130,381,646,7,0,0,21,4,15,41,910,0.661673179614133,477
2,unknown,2,1,2,20120405102902,20170302073010,122,1793,33,43,7,0,0,8,1,0,2,111,1.02771709743701,841
2,female,3,3,2,20091014131349,20161112122730,962,2587,307,334,5,0,0,49,273,33,106,493,1.00522694513289,817
1,unknown,1,2,0,20050901045004,20151022222845,284,3704,44,207,0,0,0,34,171,0,20,40,1.20440316450483,647
```