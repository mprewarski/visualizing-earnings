import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from train import MyClassifier, val_tt

ckp_path = Path(
    "/Users/marcusprewarski/projects/visualizing-earnings/lightning_logs/version_0/checkpoints/chk--epoch=13-val_acc=0.64.ckpt"
)
model = MyClassifier.load_from_checkpoint(ckp_path)
model.eval()


def get_all_earn_tickers():
    return [
        v.strip(".json") for v in os.listdir("/Volumes/extdisk/alpha_vantage/earnings")
    ]


# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    # Fallback to CPU
    device = torch.device("cpu")


def create_earnings_matrix(earn):
    if earn.min() < 0:
        rng = earn.max() - earn.min()
        earn = earn / rng
        earn = np.round(earn * 24).astype(int)
        base = int(24 - earn.max())  # calculate the bottom value of the chart
    else:
        earn /= earn.max()
        earn = np.round(earn * 24).astype(int)
        base = 0

    mat = np.zeros((24, 24)).astype(int)
    for i, v in enumerate(earn):
        col = i * 3 + 1
        if v >= 0:
            mat[base : v + base, col] = 255
        else:
            mat[base + v : base, col] = 255
    mat = np.flipud(mat)
    mat = mat.astype(np.uint8)
    return mat


earn_data_path = Path("/Volumes/extdisk/alpha_vantage/earnings")


def read_earn_data(ticker):
    with open(earn_data_path / f"{ticker}.json", "r") as file:
        earn_data = json.load(file)
        df = pd.DataFrame(earn_data["quarterlyEarnings"])
        df["reportedDate"] = pd.to_datetime(df["reportedDate"])
        df["ticker"] = ticker
    return df


def calc_earnings_score(earnings, e_index):
    # create an image from the earnings array
    earn_mat = create_earnings_matrix(earnings)
    # convert the matrix to a 3 channel matrix, each channel has the same values
    e3 = np.stack([earn_mat] * 3, axis=2)
    # turn the matrix into an image
    e3_img = Image.fromarray(e3)

    # run it through the same transform used in training
    img_ten = val_tt(e3_img).to(device)

    # run the image through the model
    model_out = model(torch.unsqueeze(img_ten, 0))
    y_hat = F.softmax(model_out, dim=-1)
    # the first value is the score
    earn_score = y_hat.cpu().detach().numpy()[0, 0]

    # path = Path('images')
    # name = f"nvda-{e_index}-{earn_score}.png"
    # print(name)
    # e3_img.save(path/name)

    return earn_score


def calc_earnings_for_ticker(df, ticker):
    earn_list = []

    # loop through the earnings taking 8 at a time
    for i in range(df.shape[0] - 8 + 1):
        try:
            # get latest 8 earnings and convert df column to numpy array
            earn_subset = df[i : i + 8]
            earnings = earn_subset.reportedEPS.to_numpy().astype(np.float64)

            report_date = earn_subset.reportedDate[i]
            # if the earnings were premarket, use the date, otherwise want the next days date
            time_shift = 1 if earn_subset.reportTime[i] == "post-market" else 0

            # switch order so latest earnings is last
            earnings = np.flip(earnings)

            earn_score = calc_earnings_score(earnings, i)

            corrected_date = report_date + datetime.timedelta(days=time_shift)
            earn_list.append((corrected_date, ticker, earn_score))
        except Exception as e:
            print(f"Exception when calculating earnings for ticker {ticker}, index {i}")
            print(e)

    return earn_list


if __name__ == "__main__":
    tickers = get_all_earn_tickers()
    print(f"Number of tickers: {len(tickers)}")

    es_path = Path("/Volumes/extdisk/ml_trading_data/earnings_scores")

    for ticker in tickers:
        print(ticker)
        df = read_earn_data(ticker)
        earn_list = calc_earnings_for_ticker(df, ticker)
        earn_df = pd.DataFrame(earn_list, columns=["date", "ticker", "earnings_score"])
        earn_df.to_csv(es_path / f"{ticker}.csv")

