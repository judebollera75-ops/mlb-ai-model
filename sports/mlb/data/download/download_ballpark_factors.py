
import os
import pandas as pd
import requests

def download_ballpark_factors():

    os.makedirs("data/ballpark", exist_ok=True)

    url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"

    parks = pd.DataFrame({
        "team":[
            "Arizona Diamondbacks",
            "Atlanta Braves",
            "Baltimore Orioles",
            "Boston Red Sox",
            "Chicago Cubs",
            "Chicago White Sox",
            "Cincinnati Reds",
            "Cleveland Guardians",
            "Colorado Rockies",
            "Detroit Tigers",
            "Houston Astros",
            "Kansas City Royals",
            "Los Angeles Angels",
            "Los Angeles Dodgers",
            "Miami Marlins",
            "Milwaukee Brewers",
            "Minnesota Twins",
            "New York Mets",
            "New York Yankees",
            "Athletics",
            "Philadelphia Phillies",
            "Pittsburgh Pirates",
            "San Diego Padres",
            "San Francisco Giants",
            "Seattle Mariners",
            "St. Louis Cardinals",
            "Tampa Bay Rays",
            "Texas Rangers",
            "Toronto Blue Jays",
            "Washington Nationals"
        ],

        "park_factor":[
            99,101,99,101,101,98,104,99,118,99,
            98,99,101,103,96,101,100,98,100,98,
            102,99,96,97,94,100,97,102,101,100
        ]
    })

    parks.to_csv(
        "data/ballpark/ballpark_factors.csv",
        index=False
    )

    return parks

if __name__ == "__main__":
    print(download_ballpark_factors())
