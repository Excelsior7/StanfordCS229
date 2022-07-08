```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python

```


```python
# MULTINOMIAL EVENT MODEL
```


```python

```


```python
stop_words = np.array(["Subject","'",".",":","?","§",";","&","~","#","{","[","|","`","\\","^","@","]","}",",","é","(","-","è","ç","à",")","=","+","°","0","9","8","7","6","5","4","3","2","1","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]);
```


```python
email_dataset = pd.read_csv("/home/excelsior/Desktop/StandordSC229/spam_ham_dataset.csv");
```


```python
email_dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>label</th>
      <th>text</th>
      <th>label_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>605</td>
      <td>ham</td>
      <td>Subject: enron methanol ; meter # : 988291\r\n...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2349</td>
      <td>ham</td>
      <td>Subject: hpl nom for january 9 , 2001\r\n( see...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3624</td>
      <td>ham</td>
      <td>Subject: neon retreat\r\nho ho ho , we ' re ar...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4685</td>
      <td>spam</td>
      <td>Subject: photoshop , windows , office . cheap ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2030</td>
      <td>ham</td>
      <td>Subject: re : indian springs\r\nthis deal is t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5166</th>
      <td>1518</td>
      <td>ham</td>
      <td>Subject: put the 10 on the ft\r\nthe transport...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5167</th>
      <td>404</td>
      <td>ham</td>
      <td>Subject: 3 / 4 / 2000 and following noms\r\nhp...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5168</th>
      <td>2933</td>
      <td>ham</td>
      <td>Subject: calpine daily gas nomination\r\n&gt;\r\n...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5169</th>
      <td>1409</td>
      <td>ham</td>
      <td>Subject: industrial worksheets for august 2000...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5170</th>
      <td>4807</td>
      <td>spam</td>
      <td>Subject: important online banking alert\r\ndea...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5171 rows × 4 columns</p>
</div>




```python
email_training_dataset = email_dataset[0:4136]; # Get 80% of the email dataset as training.
email_testing_dataset = email_dataset[4136:5172]; # Get 20% of the email dataset as testing.

print("% of spam email in training:", email_training_dataset["label_num"].sum()/len(email_training_dataset));
print("% of spam email in testing:", email_testing_dataset["label_num"].sum()/len(email_testing_dataset));
```

    % of spam email in training: 0.28505802707930367
    % of spam email in testing: 0.30917874396135264



```python
# An example of what an email looks like.

print(email_dataset.iloc[0].text);
```

    Subject: enron methanol ; meter # : 988291
    this is a follow up to the note i gave you on monday , 4 / 3 / 00 { preliminary
    flow data provided by daren } .
    please override pop ' s daily volume { presently zero } to reflect daily
    activity you can obtain from gas control .
    this change is needed asap for economics purposes .



```python
# Construct my vocabulary from all emails (email_dataset)
# Exclude all the words that are considered irrelevant to classification (stop_words)

vocabulary = [];

for i in range(len(email_dataset)):
    temp = email_dataset.iloc[i].text.split();
    for j in range(len(temp)):
        w = temp[j];
        if (w not in vocabulary) and (w not in stop_words):
            vocabulary.append(w);
```


```python
# TRAINING OF THE MODEL

# Compute the parameters of the model:
# Let prob_y1[k] be the probability that the kth word in the vocabulary appears in a spam email;
# Let prob_y0[k] be the probability that the kth word in the vocabulary appears in a non-spam email;
# Let prob_s be the probability of having a spam email (class prior);

prob_y1 = np.ones(len(vocabulary));
prob_y0 = np.ones(len(vocabulary));
prob_s = 0;


for i in range(len(email_training_dataset)):
    temp = email_training_dataset.iloc[i].text.split();
    label = email_training_dataset.iloc[i].label_num;
    
    if label == 0:
        for j in range(len(temp)):
            w = temp[j];
            if w not in stop_words:
                index_w_in_vocabulary = vocabulary.index(w);
                prob_y0[index_w_in_vocabulary] += 1;

    else:
        prob_s += 1;
        for j in range(len(temp)):
            w = temp[j];
            if w not in stop_words:
                index_w_in_vocabulary = vocabulary.index(w);
                prob_y1[index_w_in_vocabulary] += 1;

prob_y1 = prob_y1/np.sum(prob_y1);
prob_y0 = prob_y0/np.sum(prob_y0);
prob_s = prob_s/len(email_training_dataset);
```


```python
# TESTING OF THE MODEL

# To test the model and in order to avoid multiplying a lot of small numbers that might cause problems,
# I will compute the log(odds) of an email being spam, i.e. log(p(spam | email) / p(non-spam | email));
# Decision rule : 
# If log(odds) > 0, predict spam!
# If log(odds) <= 0, predict non-spam! (Note: I predict equality to be spam because the prior on spam is higher)

percentage_of_success = 0;

for i in range(len(email_testing_dataset)):
    temp = email_testing_dataset.iloc[i].text.split();
    label = email_testing_dataset.iloc[i].label_num;
    log_odds = 0;
    
    for j in range(len(temp)):
        w = temp[j];
        if w not in stop_words:
            index_w_in_voc = vocabulary.index(w);
            log_odds += np.log(prob_y1[index_w_in_voc]) - np.log(prob_y0[index_w_in_voc]) + np.log(prob_s) - np.log(1-prob_s);
    
    if log_odds > 0 and label == 1:
        percentage_of_success +=1;
    if log_odds <= 0 and label == 0:        
        percentage_of_success +=1;

percentage_of_success = percentage_of_success/len(email_testing_dataset);

percentage_of_success
```




    0.8318840579710145


