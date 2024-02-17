import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import tree
from sklearn.model_selection import cross_val_score

# Importing csv's and dropping useless columns
train = pd.read_csv('train.csv').drop(columns=['id', 'Unnamed: 32'])
test = pd.read_csv('test.csv').drop(columns=['id', 'Unnamed: 32'])

# Splitting the features and targets for training and test data
X_train = train.drop(columns='diagnosis')
X_test = test.drop(columns='diagnosis')

y_train = train['diagnosis']
y_test = test['diagnosis']

# Preforming preprocessing techniques
X_train_clean = X_train
X_train_zscore = StandardScaler().fit_transform(X_train)
X_train_minmax = MinMaxScaler().fit_transform(X_train)

X_train_preprocessed = [X_train_clean, X_train_zscore, X_train_minmax]

X_test_clean = X_test
X_test_zscore = StandardScaler().fit_transform(X_test)
X_test_minmax = MinMaxScaler().fit_transform(X_test)

X_test_preprocessed = [X_test_clean, X_test_zscore, X_test_minmax]

def run_variations_with_options(config):
  """
  Uses config options from streamlit form to run permutations on the decision tree classifier
  
  Args:
    Dictionary of configuration options from streamlit    
  
  Returns:
    Dictionary of permutation results from decision tree classifier
  """

  outputs = []
  X_train_preprocessed = []
  preprocessing_labels = []
  if config['clean']: 
    X_train_preprocessed.append(X_train_clean)
    preprocessing_labels.append("No Preprocessing")

  if config['zscore']: 
    X_train_preprocessed.append(X_train_zscore)
    preprocessing_labels.append("Z-Score")
  if config['minmax']: 
    X_train_preprocessed.append(X_train_minmax)
    preprocessing_labels.append("MinMax")

  splitter_values = []
  if config['best']: splitter_values.append('best')
  if config['random']: splitter_values.append('random')

  for X_train, X_test, label in zip(X_train_preprocessed, X_test_preprocessed, preprocessing_labels):
    for d in config['max_depth_values']:
      for s in splitter_values:
        clf = tree.DecisionTreeClassifier(max_depth=d, splitter=s)
        clf = clf.fit(X_train, y_train)
        score = cross_val_score(clf, X_test, y_test, cv = 5, scoring = 'accuracy')
        outputs.append({
            'Normalization': label,
            'Max Depth': d,
            'Splitter': s,
            'Accuracy %': score.mean()
        })

  return pd.DataFrame(outputs)

# Setting Streamlit Form title
st.title("Decision Tree Classifier")

# Creating Streamlit Form
with st.form("my_form"):
    st.write("Preprocessing Options")
    preprocess_clean = st.checkbox("No Preprocessing")
    preprocess_zscore = st.checkbox("Z-Score")
    preprocess_minmax = st.checkbox("MinMax")


    st.write("Max Depth Values")
    max_depth_str = st.text_input(label="max_depth: ex: 3,5,7 no spaces", value="")

    st.write("Splitter Choice")
    splitter_best = st.checkbox("best")
    splitter_random = st.checkbox("random")

    submitted = st.form_submit_button("Submit")
    if submitted:
        config = {}
        config['clean'] = preprocess_clean
        config['zscore'] = preprocess_zscore
        config['minmax'] = preprocess_minmax
        config['max_depth_values'] = [int(i) for i in max_depth_str.split(",")]
        config['best'] = splitter_best
        config['random'] = splitter_random
        output = run_variations_with_options(config)
        st.dataframe(output)
        
    

