{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "27f7fd07-5dd0-4755-9720-97e0fd10322b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load the pre-trained model\n",
    "model_pipeline = joblib.load('logistic_regression_model.pkl')\n",
    "\n",
    "# Create the Streamlit app\n",
    "st.title(\"Titanic Survival Prediction\")\n",
    "\n",
    "# User inputs for the model\n",
    "sex = st.selectbox(\"Sex\", ['male', 'female'])\n",
    "age = st.number_input(\"Age\", min_value=0, max_value=100, value=30)\n",
    "pclass = st.selectbox(\"Pclass\", [1, 2, 3])\n",
    "fare = st.number_input(\"Fare\", min_value=0, max_value=500, value=50)\n",
    "embarked = st.selectbox(\"Embarked\", ['C', 'Q', 'S'])\n",
    "\n",
    "# Create a dataframe from the user inputs\n",
    "user_input = pd.DataFrame({\n",
    "    'Sex': [sex],\n",
    "    'Age': [age],\n",
    "    'Pclass': [pclass],\n",
    "    'Fare': [fare],\n",
    "    'Embarked': [embarked]\n",
    "})\n",
    "\n",
    "# Make prediction\n",
    "if st.button(\"Predict Survival\"):\n",
    "    prediction = model_pipeline.predict(user_input)\n",
    "    if prediction == 1:\n",
    "        st.write(\"This passenger survived.\")\n",
    "    else:\n",
    "        st.write(\"This passenger did not survive.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
