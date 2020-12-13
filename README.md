# Airbnb analysis & price prediction

**IDS 2020 Course Project**

Using [New York City Airbnb Open Data on Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data).

The project is described in-depth in the [report](https://github.com/tonysln/ids2020-project/blob/main/C4_report.pdf).


## Files
```shell
/
├── data
│     ├── AB_NYC_2019.csv        # the dataset from Kaggle, contains information on Airbnb listing activity and metrics in New York, 2019
│     ├── AB_NYC_2019_xlsx.xlsx  # the original dataset ported to Microsoft Excel
│     └── NYC.jpg                # illustrative map of New York City
├── notebooks
│     ├── Analysis.ipynb         # Jupyter notebook (Python 3.8) for analyzing the dataset
│     └── Project.ipynb          # Jupyter notebook (Python 3.8) for the project, containing debugging and testing code
├── output                       # dataset visualizations in .png format, created by the main script
│     ├── fig1.png
│     └── ...
├── C4_report.pdf                # CRISP-DM report of the project
├── C4-KAGGLE-NYC-AIRBNB.py      # main source code for the project, using Python 3.8
└── C4-NYC-AIRBNB_poster.pdf     # introductory poster for our project
```

## Requirements

Python version: `3.8` or newer

Libraries: `pandas, numpy, seaborn, matplotlib, sklearn`

## Usage

1. Clone this repository

```shell
$ git clone https://github.com/tonysln/ids2020-project.git
```

2. Install missing libraries using `pip`

3. Run the main script

```shell
$ python C4-KAGGLE-NYC-AIRBNB.py
```

## Team
* Anton Slavin (Group 6)
* Elen Liivapuu (Group 6)
