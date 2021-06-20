# Development of AI model for Racing Driving Level Classification

This repository describes a team project conducted at Yonsei University's MACHINE LEARNING AND PROGRAMMING (MEU5053) in the first semester of 2021. All codes were written by jupyter notebook. Required packages are written in 'requirements.txt'

## 1. Problem Formulation
### 1-1. Motivation & Background
![image](https://user-images.githubusercontent.com/57740749/122670504-6c5e9100-d1fd-11eb-843c-06090994aa74.png)

Most people aren't familiar with car racing, so it's hard to get training without experts. The racing techniques required to be performed are different depending on each driving level of drivers. We want to develop racing assistant program that can help low-skilled drivers, providing right information or recommending driving skills in real time. For this purpose, first thing that must be done is classifying driver level of input drivers.
### 1-2. Goal of project
Goal of this project is to classify driving level of input driver using vehicle's sensor data in virtual game environment, "Assetto Corsa". Assetto Corsa is the popular racing game that sensor data is as realistic as it really is. We wanted to implement binary classification of driving level as beginner or expert.


## 2. Data Generation
### 2-1. Game setup
![image](https://user-images.githubusercontent.com/57740749/122665031-d9632e00-d1df-11eb-8938-c8ddc4bbecac.png)

We downloaded and played assetto corsa which can be purchased in STEAM. We used Logitech G27 Steering Wheel, Accelerator, Brake to play the game and generate data. To collect data in game, We used Assetto Corsa Telemetry Interface(ACTI) that can convert game data into csv file from racedepartment community(https://www.racedepartment.com/downloads/acti-assetto-corsa-telemetry-interface.3948/). We chose one specific vehicle and map (Pagani Zonda R, ks_highlands track).
### 2-2. Data Description
![image](https://user-images.githubusercontent.com/57740749/122665313-7377a600-d1e1-11eb-938c-a7ed911f75cf.png)

Three people of our group played with above setup. Although all of us weren't experts, but one had good skills. So we collected 19 lap data that is one person's played data as expert, and another 19 lap data that two people's played data as beginner. Each lap data has 180 features that includes ground speed, tire temperature, tire pressure, etc. Each lap data is collected at 20Hz. This played data are in folder 'beginner_expert_RawData'. Because there is no columns about lateral/longitudinal Velocity and acceleration, We calculated them using GPS and ground speed columns. In addition to this, we added CG distance data column that describes euclidean distance of lateral/longitudinal acceleration. This added data are in folder 'beginner_expert_processedData'.
### 2-3. Data analysis
![image](https://user-images.githubusercontent.com/57740749/122666503-98234c00-d1e8-11eb-933f-7cc705d0e091.png)

We thought dividing driving levels was data when driving a curve. To ensure if the collected data were collected well by driving level when driving curve, we plotted data and saw the difference. You can see this process in dataplotting.ipynb
```
(in jupyter notebook or google colab) look description and run each blocks of dataplotting.ipynb
```

## 3. Method
### 3-1. Used Features
![image](https://user-images.githubusercontent.com/57740749/122666127-71fcac80-d1e6-11eb-9404-f07fb318af46.png)

We decided to use two kinds of features. one is all features except game settings and the other is realistic sensor features that is only 13 features that can be collected in realistic environment. We wanted to look difference between these two kinds of features.
### 3-2. Problem solving methods
we used three methods that is SVM(Support Vector Machine), LSTM(Long Short-Term Memory), GRU(Gated Recurrent Units). 

**- Using SVM in curve data**

![image](https://user-images.githubusercontent.com/57740749/122667007-84c5b000-d1eb-11eb-8d4f-8b8e4dac84b4.png)

referred to <Naiwala P. Chandrasiri, Kazunary Nawa, Akira Ishii, 2016, “Driving skill classification curve driving scenes using machine learning”>, we wanted to do similar process of it.
1. Pre-processing step : convert driving data into distance-indexed data, extract curve data based on distance and normalize it. 
2. Feature extraction : In this paper, they implemented PCA but we thought that this process isn't necessary if normal data can be classified well. So we except this process
3. Driving Skill classification using k-NN & SVM : this paper shows that SVM worked better than kNN all cases, so we did just svm in each/all curve data.



**- Using LSTM in curve data**

![image](https://user-images.githubusercontent.com/57740749/122667112-264d0180-d1ec-11eb-9d4f-ad2d17dcca3b.png)

Compared to SVM, LSTM is more suitable to adopt time series data. It has cell state which can remember and forget information during time series data. We thought racing skills based on lap times will be relevant with continuos data in curve data. Thus, it seems to be reasonalbe to conclude that we need to us LSTM. We implemented it for each curve and all curve as same with SVM process. 



**- Using GRU in curve data**

![image](https://user-images.githubusercontent.com/57740749/122667124-36fd7780-d1ec-11eb-84bf-f0dfe9f56545.png)

LSTM has three gates including input, output, forget but GRU has only two gates, which are reset and update. It is less complex then LSTM. Since GRU is also effective in time series data, we thought it would be appropriate for our drive data same as LSTM. We implemented each curve and all curve as same with above methods.



## 4. Results
### 4-1. SVM
![image](https://user-images.githubusercontent.com/57740749/122667254-dae72300-d1ec-11eb-8846-9ab1b1f5fb65.png)
![image](https://user-images.githubusercontent.com/57740749/122667261-e20e3100-d1ec-11eb-82e3-c3ecd79c5492.png)

Table shows average of cross-validation score. Parameters of SVM were selected using grid searching. Results show that all accuracy were higher than 90% regardless of curves and features. We can see that using all features were slightly higher than using selected features as we thought. However even using selected features were worked well. As you can see, ROC curve that can illustrates the diagnostic ability of a binary classification shows good results. In addition, Most of ROC AUC(Area Under the Curve) were 0.99. Future importances of one data that is sampled in linear kernel and all features show that Toe-in of front tires were most important things. This is reasonable because toe-in refers to front wheels tilting towards the centerline of a vehicle and also means stability of vehicle.
```
(in jupyter notebook or google colab) look description and run each blocks of implementSVM.ipynb
```
### 4-2. LSTM
![image](https://user-images.githubusercontent.com/57740749/122667963-aaa18380-d1f0-11eb-938c-aa5f9d8f49d2.png)
![image](https://user-images.githubusercontent.com/57740749/122667969-b3925500-d1f0-11eb-92d2-f75d81e89fd7.png)

Table shows the accuracy of LSTM. Compared to SVM, most of results were mixed up and lower than 90%. We used one lstm layer at each curve and two stacked LSTM at all curve classification. We thought this result was due to the form of the data. While SVM uses each time step data when driving curve into each train data, LSTM uses whole time step of curve data into one single sequence data. So the total amount of data was much larger in SVM compared to LSTM. Moreover, because of the unique features of driving data, beginners' driving wasn't smooth as experts so that there was less correlation compared to general time series data.
```
(in jupyter notebook or google colab) look description and run each blocks of implementLSTMandGRU.ipynb
```
### 4-3. GRU
![image](https://user-images.githubusercontent.com/57740749/122668415-ee958800-d1f2-11eb-9e59-321a904aa90e.png)
![image](https://user-images.githubusercontent.com/57740749/122668423-f6edc300-d1f2-11eb-99a8-20b7676fdbbc.png)

Similar with LSTM, GRU shows mixed up and low accuracy than SVM. We just used one gru layer, and adding other layers results lower accuracy. As same with description in LSTM, we thought that shortages of data and uncorrelated data results low accuracy. 
```
(in jupyter notebook or google colab) look description and run each blocks of implementLSTMandGRU.ipynb
```

## 5. Code explanation
### implementSVM.ipynb
- generated_data() : generate each curve data and make it to csv file using curve distance. It has 6 kinds of curves and result csv file 'data.csv' is saved in cornerData folder.
- load_data(left_column, curve_number) : load data.csv file to X and y(label) data. we can load data as your purpose. when you want to classify with all columns at curve 3, you can just put like load_data(all_column,3).
- processing_data(X,y) : splits X and y data to train and test data with 8:2, and normalize it.
- evaluate_model(X_train, X_test, y_train, y_test,kernel='rbf', C=1, gamma=0.01) : with specific parameter, you can fit model and evaluate it using cross validation score.
- confusion_matrix(y_test, y_pred) : show confusion matrix using ground truth of y and predicted y.
- draw_ROC(y_test,y_pred) : draw ROC curve using ground truth of y and predicted y.
- grid_searching(X_train, y_train) : implement grid searching with user's defined parameters.
- run_experiment(column, corner) :  main code of SVM. it receives which features to use and which corner to classifying. it uses all above function and show results.
- f_importances(coef, names, top=-1) : when using linear kernel of svm, it can shows importances of columns. 

### implementLSTMandGRU.ipynb
- generate_allcorner_data(features) : depending on which features to use, make entire laps' all corner data into one sequence which are padded with length of 60. it returns train, test, validation and its' label.
- generate_onecorner_sequences(corner_num, features) : depending on which corner & features, make entire laps' one corner data into one sequence which are padded with length of 60. it returns train, test, validation and its' label.
- 'load data' : load data using one of above functions.
- 'model making' : it has 2 options, LSTM and GRU. we can select which model to use and generate sequential layers.
- 'learning' : select hyperparameters, and implement learning process. it saves model if validation accuracy improve.
- 'load model and evaluate' : plotting history of model and evaluate with loaded model.  

### implementGRU_torch_new
- pytorch version of above keras version process

## 6. Reference
1. Naiwala P. Chandrasiri, Kazunary Nawa, Akira Ishii, 2016, “Driving skill classification curve driving scenes using machine learning”
2. Shuguang LI, Shigeyuki YAMABE, Yoichi SATO, Takayuki HIRASAWA, Yoshihiro SUDA, Naiwala P. CHANDRASIRI and Kazunari NAWA, 2013, “Driving Feature Extraction from High and Low Skilled Drivers in CURVE Sections Based on Machine Learning”
3. Shuguang LI, Shigeyuki YAMABE, Yoichi SATO, Yoshihiro SUDA, Naiwala P. CHANDRASIRI and Kazunari NAWA, 2014, “Learning Characteristic Driving Operations in Curve Sections that Reflect Drivers’ Skill Levels”
