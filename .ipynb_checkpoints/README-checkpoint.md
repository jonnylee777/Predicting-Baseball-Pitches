# MLB Pitch Prediction Machine Learning Project
This project develops machine learning models to predict the next pitch type thrown in MLB games using Statcast pitch-by-pitch data. We wish to train a model to accurately predict what a pitch an individual pitcher will throw in a given situation. Multiple models are compared while iteratively engineering features to improve predictive performance.  

## Motivation
With the rise of analytics in sports, or sabermetrics, teams are finding ways to use data science techniques to turn the game into a science and find competitive advantages over their opponents. While fans only see the war being waged on the diamond, the battles being fought outside of the white lines plays a key supporting role in helping the players succeed.

While pitch selection is an inherently noisy classification task, pitchers are famously creatures of habit, and tend to fall into patters given the circumstances.  Knowing what pitch is coming, or even being able to anticipate a certain pitch with an increased probability, is a major advantage for the hitter. While hitters often have a "hunch" or anecdotal experiences for which to build intuition on which is coming, a more systematic and data-centered approach could be beneficial in developing scouting reports and helping hitters have a prepared approach or plan of attack during each at-bat. 





## Datasets
**Kevin Gausman Dataset:**
[Baseball Savant](https://baseballsavant.mlb.com/statcast_search?hfGT=R%7C&hfSea=2026%7C2025%7C2024%7C2023%7C2022%7C2021%7C2020%7C2019%7C2018%7C2017%7C2016%7C2015%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C&player_type=pitcher&pitchers_lookup%5B%5D=592332&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&sort_order=desc#results)

This dataset contains information about Kevin Gausman's career pitches up to June 30th, 2026. This data was collected via Statcast and made available through Baseball Savant.

**Shape:** 25,000 rows with 117 features

**Dates included:** 5/23/13 - 6/30/26

**Features:** Each pitch has associated features that describe the information of the pitcher and batter, the physics characteristics of the pitch, the situation preceding the pitch, the setup of the defensive alignment, the results of the pitch, the results of the corresponding at-bat, and other descriptive variables.

**To access exact same dataset:**
1. Click above link
2. Set *Game Date* parameter as 6/31/26
3. Scroll down to Search Results
4. Click on Graphs
5. Download as CSV

**MLB League-Wide 2025 Pitch Dataset:**
[Baseball Savant](https://baseballsavant.mlb.com/statcast_search?hfGT=R%7C&hfSea=2025%7C&player_type=pitcher&group_by=pitch-type&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&sort_order=desc#results)

This dataset contains information about league-wide pitch usage for the MLB 2025 season. This data was collected via Statcast and made available through Baseball Savant.

**Shape:** 3,603 rows with 20 features

**Features:** It contains information on an individual pitch in each pitcher's arsenal, with features describing statistics of total usage, frequency and efficacy, such as run value.

**To access exact same dataset:**
1. Click above link
2. Scroll down to Search Results
3. Click on the icon that looks like a printer
4. Download as CSV
   
## Problem Statement
Given the game situation, we seek to predict the pitcher's next pitch. This is a classification problem seeking to predict the target variable **pitch_type** using the existing and engineered features.


## Methodology
We used a ML pipeline to create and evaluate our model:

```text
Raw MLB 2025 League-Wide Pitch Data (452 × 279)
        │
        ▼
  Exploratory Data Analysis
  ├── League-Wide Pitch Distribution Analysis
  ├── Pitch Categorization
  └── Pitch Categories Distribution Analysis
        │
        ▼
Raw Kevin Gausman Career Pitch Data (452 × 279)
        │
        ▼
  Exploratory Data Analysis
  ├── Kevin Gausman Career Pitch Distribution Analysis
        │
        ▼
  Data Cleaning Preprocessing
  ├── Handle missing values (Removed intentional walks)
  ├── Transformed pitch results to pre-pitch variables for next pitch
  └── Transformed at-bat results to pre-ab features for pitches of next at-bat
        │
        ▼
  Modeling Setup
  └── One-Hot Encoding
  └── Data preprocessing for tree and linear models
  └── Train/Test split by games
  └── Model fit, cross-validation, evaluation pipeline setup
  └── Feature importance, comparison table setup
        │
        ▼
  Model Training & Evaluation
  ├── Naive Stratified Baseline
  ├── Random Forest: kg1, kg2, kg3, kg4
  ├── Logistic Regression
  ├── Gradient Boosting
  ├── Linear SVM
        │
        ▼
  Evaluation: Testing Accuracy, Cross-Validation 
```
Due to the individualistic habits and tendencies of each pitcher, we decided to train our models on a singular case study as a trial into the efficacy of our prediction efforts. For our first iteration, we chose Kevin Gausman of the Toronto Blue Jays. He is a good case study, as he has been in the league for a decent amount of time, and therefore has accumulated many pitches as data. In addition, he primarily throws only three main pitches (fastball, split-finger, slider), which will provide clear parameters for our model to study.

## Feature Engineering
```text
  kg0: Raw dataset
        │
        ▼
  kg1
  ├── Pitch results-> Pre-pitch variables
  ├── At-bat results-> Pre-AB variables
        │
        ▼
  kg2
  └── Remove dense physics-ey pitch characteristics (i.e. acceleration of pitch at 50ft)
  └── Remove API + derived break variables, IDs
  └── Remove Hawkeye + swing characteristic variables
  └── Remove unnecessary spin/release/location characteristics
  └── Add previous pitch type variable
  └── Add Count Leverage variable
  └── Add Pitcher/Hitter Same/Opposite Hand variable
  └── Add Pitch No. of Game variable
  └── Add Career Pitcher/Hitter Matchup variable
  └── Add Pitcher Team Lead variable
  └── Add Strike Zone Height variable
        │
        ▼
  kg3
  └── Remove redundant variables
  └── Remove fielder ID
  └── Add previous 3-Pitch History variable
        │
        ▼
  kg4
  └── Remove all variables not readily available at game-level (i.e. only include all variables available to viewers watching via MLB Gameday)

```

## Models Tested
1. Stratified Naive Model
2. Random Forest
3. Logistic Regression
4. Gradient Boosting
5. Linear SVM

## Results
| KG Dataset | Model | Train Accuracy | Test Accuracy |
|------------|----------------------|---------------:|--------------:|
| KG1 | Naive Stratified | 0.4270 | 0.4265 |
| KG1 | Random Forest | 0.6907 | 0.5772 |
| KG2 | Random Forest | 0.6859 | 0.5852 |
| KG3 | Random Forest | 0.6759 | 0.5874 |
| KG4 | Random Forest | 0.6664 | 0.5800 |
| KG4 | Logistic Regression | 0.5826 | 0.5543 |
| KG4 | Gradient Boosting | 0.6861 | 0.5811 |
| KG4 | Linear SVM | 0.5839 | 0.5563 |

## Key Findings
Our random forest model was able to predict pitches at roughly a 58% rate, a roughly 36% increase over our stratified naive baseline. 

## Repository Structure
```text
Pitch Prediction Project/
│
├── 📂 Data/
│   ├── 592332_data-2.csv               # Raw Kevin Gausman career pitches dataset
│   └── kg1_cleaned                     # Cleaned Kevin Gausman dataset produced by cleaning notebook
│   └── pitch-arsenal-stats             # 2025 MLB league-wide pitch arsenal data
├── 📂 Notebooks/
│   ├── Cleaning.ipynb                  # Data exploration, transformation
│   └── Model1.ipynb                    # Model preparation, feature engineering, model comparisons
├── requirements.txt                  # Python dependencies
└── README.md                         # You are here
```

## How to Set Up

### 1. Clone the repository

```bash
git clone https://github.com/jonnylee777/Predicting-Baseball-Pitches.git
cd Predicting-Baseball-Pitches
```

### 2. Create a virtual environment (recommended)

**Mac/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Prepare the data

Place the required CSV files inside the `Data/` directory:

```
Data/
├── 592332_data-2.csv
└── pitch-arsenal-stats.csv
```

*(If these datasets are already included in the repository, this step can be skipped.)*

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

### 6. Run the notebooks

Execute the notebooks in the following order:

1. `Cleaning.ipynb`
2. `Model1.ipynb`

The cleaning notebook produces the processed dataset (`kg1_cleaned.csv`), which is used by the modeling notebook.

## Future Improvements
Other tree models could be tried in order to see if they could provide more predictive power. Examples include XG Boost and CatBoost.
