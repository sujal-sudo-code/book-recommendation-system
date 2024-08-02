## Book Recommendation System

A Book Recommendation System using collaborative filtering to suggest books based on user ratings.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Features](#features)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project builds a book recommendation system using a dataset of books, users, and ratings. It employs collaborative filtering to suggest books similar to those a user has rated highly.

## Dataset
The datasets used in this project are:
- BX-Books.csv
- BX-Users.csv
- BX-Book-Ratings.csv
- Link for dataset is - https://www.kaggle.com/datasets/ra4u12/bookrecommendation

## Usage
1. Ensure you have the datasets (`BX-Books.csv`, `BX-Users.csv`, `BX-Book-Ratings.csv`) in the `Dataset` directory.
2. Run the `book_recommender_system.ipynb` script to process the data and generate recommendations:
    ```bash
    python main.py
    ```
3. To recommend books for a given title, use the `recommend_book` function:
    ```python
    from recommendation import recommend_book
    recommend_book('Wish You Well')
    ```

## Features
- Load and preprocess book, user, and rating data
- Filter users and books to focus on those with significant interactions
- Build a sparse matrix and apply Nearest Neighbors algorithm
- Save and load the trained model and data
- Generate book recommendations based on user input


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any feature requests or bug reports.
