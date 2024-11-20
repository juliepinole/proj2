> [!WARNING]
> WIP.
> 
> Reproducing Task 2 of Project 2.
> 
> This folder is still WIP and does not comply with Python style guidelines. 


# Project ECG (Project 2): Time Series and Representation Learning

In this part, we explore how to **leverage information from the larger MIT-BIH database to improve learning on the smaller PTB dataset**.
  *  In *transfer learning*, we use representations from supervised learning on a related but distinct task (here, arrhythmia classification).
  *  In *representation learning*, we use unsupervised learning strategies.

## Notebooks by questions

> [!IMPORTANT]
> The following links show the original notebooks, before creating the reproducing folder.

###  Part 1: Supervised Learning on Time Series

*  **Q1**: [Exploratory Analysis](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q1_Exploration.ipynb)
*  **Q2**: [Classic ML Methods](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q2_Classic_ML_Models.ipynb) , [Features Extraction](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q2_features_extraction.ipynb)
> [!NOTE]    
> The features extraction takes some time, this is why they are done separately in the second colab [Features Extraction](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q2_features_extraction.ipynb) (two extractions, one for the train set, the other one for the test set). They are then imported back in the first colab ([Classic ML Methods](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q2_Classic_ML_Models.ipynb)). These extracted features are uplaoded top github so the path should match and there should not be any action required. See:
> ![image](https://github.com/juliepinole/proj2/assets/166155962/9625efc2-6965-45ee-8379-e694817c4650)

*  **Q3**: [LSTM](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q3_LSTM.ipynb)
*  **Q4**: [CNN](https://github.com/juliepinole/proj2/blob/main/task_1/Part_1_Q4_CNN.ipynb)
*  **Q5**: Unfinished.

###  Part 2: Transfer and Representation Learning
*  **Q1**: [Classic ML Methods + encoder training - MIT BIH Dataset](https://github.com/juliepinole/proj2/blob/main/task_1/Part_2_Q1_Classic_ML.ipynb)
*  **Q2**: [Representation Learning](https://github.com/juliepinole/proj2/blob/main/task_1/Part_2_Q2_Representation.ipynb)
*  **Q3**: [Visualization](https://github.com/juliepinole/proj2/blob/main/task_1/Part_2_Q3_Visualization.ipynb)
*  **Q4**: [Fine_tuning](https://github.com/juliepinole/proj2/blob/main/task_1/Part_2_Q4_Fine_Tuning.ipynb)

## Virtual Environment

> [!IMPORTANT]
> The result of `pip list` in the virtual environment that I used in in the file [requirements.txt](https://github.com/juliepinole/proj2/blob/main/requirements.txt).


> [!TIP]
> TBD

