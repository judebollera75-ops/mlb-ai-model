import numpy as np
import pandas as pd
from scipy.stats import poisson


def probability_over(projected_ks, line):
    return 1 - poisson.cdf(np.floor(line), projected_ks)


def probability_under(projected_ks, line):
    return poisson.cdf(np.floor(line), projected_ks)


def fair_odds(prob):

    if prob >= 0.5:
        return round(-(prob/(1-prob))*100)

    return round(((1-prob)/prob)*100)


def build_probability_table(df):

    lines = [3.5,4.5,5.5,6.5,7.5,8.5]

    rows=[]

    for _,row in df.iterrows():

        for line in lines:

            over = probability_over(row["calibrated_projected_ks"], line)
            under = probability_under(row["calibrated_projected_ks"], line)

            rows.append({

                "pitcher":row["pitcher_name"],
                "projected_ks": round(row["calibrated_projected_ks"], 2),

                "line":line,

                "over_prob":round(over,3),
                "under_prob":round(under,3),

                "fair_over_odds":fair_odds(over),
                "fair_under_odds":fair_odds(under)

            })

    return pd.DataFrame(rows)


if __name__=="__main__":

    projections = pd.read_csv(
    "outputs/calibrated_strikeout_projections.csv"
)

    table=build_probability_table(projections)

    table.to_csv(
        "outputs/probability_table.csv",
        index=False
    )

    print(table.head(30))
